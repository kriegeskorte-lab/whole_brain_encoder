import sys

sys.path.append("/engram/nklab/algonauts/ethan/whole_brain_encoder")

import numpy as np
from pathlib import Path
from attention_maps.utils import calc_overlap

from plot_run_results import plot_parcels_return_img
import matplotlib.pyplot as plt
from tqdm import tqdm
import fetch
import argparse


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--subj", type=int)
    argparser.add_argument("--hemi", type=str)
    argparser.add_argument("--parcel_strategy", type=str)
    args = argparser.parse_args()

    overlap_selection(args.subj, args.hemi, args.parcel_strategy)
    save_top_parcels(args.subj, args.hemi, args.parcel_strategy)


def overlap_selection(subj, hemi, strategy):
    metadata = fetch.metadata(subj)

    parcels = fetch.all_parcels(subj, hemi, strategy)
    save_dir = (
        Path("/engram/nklab/algonauts/ethan/whole_brain_encoder/img_gen")
        / strategy
        / "overlap_labeled_parcels"
        / f"subj_{subj:02}"
        / hemi
    )
    save_dir.mkdir(exist_ok=True, parents=True)

    for roi in tqdm(
        metadata[f"{hemi}_rois"],
        desc=f"Subj {subj}, {hemi}",
        total=len(metadata[f"{hemi}_rois"]),
        leave=False,
    ):
        candidate_parcels = parcels[hemi]

        roi_mask = np.zeros(163842)
        roi_mask[metadata[f"{hemi}_rois"][roi]] = 1
        values, indices = calc_overlap(roi_mask, candidate_parcels)

        if values.max() == 0:
            print(f"No overlap found for {roi} in {hemi}")
            continue

        for i in range(3):
            parcel_with_most_overlap = indices[i]
            parcel_with_most_overlap_mask = np.zeros(163842)
            parcel_with_most_overlap_mask[
                candidate_parcels[parcel_with_most_overlap]
            ] = 1

            if hemi == "lh":
                img = plot_parcels_return_img(
                    parcel_with_most_overlap_mask,
                    np.zeros(163842),
                    cmap="YlOrRd_r",
                    colorbar=False,
                )
                width, height = img.size
                crop_box = (0, 0, width * 11 // 20, height)
            elif hemi == "rh":
                img = plot_parcels_return_img(
                    np.zeros(163842),
                    parcel_with_most_overlap_mask,
                    cmap="YlOrRd_r",
                    colorbar=False,
                )
                width, height = img.size
                crop_box = (width * 9 // 20, 0, width, height)

            img = img.crop(crop_box)
            parcel_info = {
                "parcel_num": parcel_with_most_overlap,
                "parcel": candidate_parcels[parcel_with_most_overlap],
                "pycortex_img": img,
            }
            img.save(save_dir / f"{roi}_{i}.png")
            np.save(save_dir / f"{roi}_{i}.npy", parcel_info)


def save_top_parcels(subj, hemi, strategy):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    metadata = fetch.metadata(subj)
    parcels = fetch.all_parcels(subj, hemi, strategy)

    nsd_roi_labels = {}
    nsd_roi_labels[hemi] = metadata[f"{hemi}_rois"]

    candidates = parcels[hemi]

    # pick parcels that don't overlap with labeled areas
    def filter_for_overlap(candidate_parcels, allowed_overlap=0.1):
        labeled_areas = np.zeros(163842, dtype=bool)
        for roi in nsd_roi_labels[hemi]:
            if roi in [
                "early",
                "midventral",
                "midlateral",
                "midparietal",
                "ventral",
                "lateral",
                "parietal",
            ]:
                continue
            labeled_areas = np.logical_or(labeled_areas, nsd_roi_labels[hemi][roi])

        final_indices = []
        other_indices = []
        overlaps = []
        for i, p in enumerate(candidate_parcels):
            parcel_area = np.zeros(163842, dtype=bool)
            parcel_area[p] = 1
            overlap = np.logical_and(parcel_area, labeled_areas).sum()
            overlaps.append(overlap)
            if overlap < allowed_overlap * len(p):
                final_indices.append(i)
            else:
                other_indices.append(i)

        final_indices = np.array(final_indices)
        other_indices = np.array(other_indices)
        overlaps = np.array(overlaps)
        not_qualifying = overlaps[other_indices]
        not_qualifying = not_qualifying[not_qualifying > 0]
        qualifying = overlaps[final_indices]
        qualifying = qualifying[qualifying > 0]
        axs[0, 0].hist(not_qualifying, bins=100, alpha=0.5, label="not qualifying")
        axs[0, 0].hist(
            qualifying,
            bins=10,
            alpha=0.5,
            label="qualifying",
            color="red",
        )
        axs[0, 0].legend()
        axs[0, 0].set_title(
            f"parcels with nonzero overlap ({(overlaps == 0).sum()} have no overlap)"
        )
        axs[0, 0].set_xlabel("# overlapping voxels")

        # candidate_parcels = set(
        #     [
        #         i
        #         for i, p in enumerate(candidate_parcels)
        #         if values[i] < allowed_overlap * len(p)
        #     ]
        # )
        return final_indices

    overlap_ok_indices = filter_for_overlap(candidates)
    candidates = [p for i, p in enumerate(candidates) if i in overlap_ok_indices]

    # choose high noise ceiling areas

    def filter_for_noise_ceiling(candidate_parcels):
        nc = metadata[f"{hemi}_ncsnr"].squeeze()
        norm_term = 1 / 3
        nc = (nc**2) / ((nc**2) + norm_term)
        nc = np.sqrt(nc)

        parcel_nc = [nc[parcel].mean() for parcel in candidate_parcels]
        axs[0, 1].hist(parcel_nc, bins=100)
        axs[0, 1].set_title(
            f"mean parcel noise ceiling (normalized), {(parcel_nc > np.percentile(parcel_nc, 75)).sum()} qualify"
        )
        axs[0, 1].axvline(
            np.percentile(parcel_nc, 75), color="r", linestyle="dashed", linewidth=1
        )

        candidate_parcels = {
            i: nc
            for i, (p, nc) in enumerate(zip(candidate_parcels, parcel_nc))
            if nc > np.percentile(parcel_nc, 75)
        }

        return candidate_parcels, parcel_nc

    highnc_indices, parcel_nc = filter_for_noise_ceiling(candidates)

    # load test correlation info
    # enc_model_dir = fetch.ensemble_model_dir(subj, strategy, runs=[1, 2])
    test_corr = fetch.split_corr(subj, strategy, runs=[1, 2])

    def filter_for_correlation(candidate_parcels):
        parcel_corr = [test_corr[hemi][parcel].mean() for parcel in candidate_parcels]

        axs[1, 0].hist(parcel_corr, bins=100)
        axs[1, 0].set_title(
            f"mean parcel correlation on test set, {(parcel_corr > np.percentile(parcel_corr, 75)).sum()} qualify"
        )
        axs[1, 0].axvline(
            np.percentile(parcel_corr, 75), color="r", linestyle="dashed", linewidth=1
        )

        candidate_parcels = {
            i: corr
            for i, (p, corr) in enumerate(zip(candidate_parcels, parcel_corr))
            if corr > np.percentile(parcel_corr, 75)
        }

        return candidate_parcels, parcel_corr

    highcorr_indices, parcel_corr = filter_for_correlation(candidates)

    final_indices = set(highcorr_indices.keys()).intersection(
        set(highnc_indices.keys())
    )
    selected_nc = []
    selected_corr = []
    nonselected_nc = []
    nonselected_corr = []
    for i in range(len(candidates)):
        if i in final_indices:
            selected_nc.append(parcel_nc[i])
            selected_corr.append(parcel_corr[i])
        else:
            nonselected_nc.append(parcel_nc[i])
            nonselected_corr.append(parcel_corr[i])

    candidates = [p for i, p in enumerate(candidates) if i in final_indices]
    axs[1, 1].scatter(nonselected_nc, nonselected_corr, label="nonselected")
    axs[1, 1].scatter(selected_nc, selected_corr, label="selected")
    axs[1, 1].set_title(f"nc, corr for {len(candidates)} parcels")
    axs[1, 1].set_xlabel("nc (normalized)")
    axs[1, 1].set_ylabel("corr")
    axs[1, 1].plot([0, 0.8], [0, 0.8], "r--", label="noise ceiling")
    axs[1, 1].legend()
    print(f"final number of parcels: {len(candidates)}")

    # save parcels

    anterior_vertices = np.zeros(163842, dtype=bool)
    anterior_vertices[metadata[f"{hemi}_anterior_vertices"]] = 1
    posterior_vertices = np.logical_not(anterior_vertices)

    fig.suptitle(f"Subject {subj} - {hemi}")
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    # plt.show()

    for i, p in enumerate(candidates):
        for parcel_idx, og_parcel in enumerate(parcels[hemi]):
            if np.array_equal(og_parcel, p):
                break
        else:
            raise ValueError("parcel not found")
        assert np.array_equal(parcels[hemi][parcel_idx], p)

        labeled_areas = np.zeros(163842, dtype=bool)
        for roi in nsd_roi_labels[hemi]:
            if roi in [
                "early",
                "midventral",
                "midlateral",
                "midparietal",
                "ventral",
                "lateral",
                "parietal",
            ]:
                continue
            labeled_areas = np.logical_or(labeled_areas, nsd_roi_labels[hemi][roi])

        parcel_area = np.zeros(163842, dtype=bool)
        parcel_area[p] = 1
        overlap = np.logical_and(parcel_area, labeled_areas).sum()
        print(f"overlap: {overlap}")

        a = np.zeros(163842, dtype=bool)
        a[p] = 1

        # check how much back of brain overlap
        back_overlap = np.logical_and(a, posterior_vertices)
        back_overlap = back_overlap.sum() / a.sum()
        if back_overlap == 0:
            posterior = False
        else:
            posterior = True

        if hemi == "lh":
            img = plot_parcels_return_img(
                a,
                np.zeros(163842),
                cmap="YlOrRd_r",
                colorbar=False,
            )
            width, height = img.size
            crop_box = (0, 0, width * 11 // 20, height)
        elif hemi == "rh":
            img = plot_parcels_return_img(
                np.zeros(163842),
                a,
                cmap="YlOrRd_r",
                colorbar=False,
            )
            width, height = img.size
            crop_box = (width * 9 // 20, 0, width, height)

        img = img.crop(crop_box)

        parcel_info = {
            "parcel": p,
            "test_corr": test_corr[hemi][p],
            "posterior": posterior,
            "desc": "in top 75% nc and test_corr, no overlap with labeled areas",
            "pycortex_img": img,
        }
        save_dir = Path("/engram/nklab/algonauts/ethan/whole_brain_encoder/img_gen")
        save_dir = save_dir / strategy / "candidate_parcels" / f"subj_{subj:02}" / hemi
        print(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        np.save(save_dir / f"{parcel_idx}.npy", parcel_info)
        img.save(save_dir / f"{parcel_idx}.png")
        print(f"saving image at {save_dir / f'{parcel_idx}.png'}")


if __name__ == "__main__":
    main()
