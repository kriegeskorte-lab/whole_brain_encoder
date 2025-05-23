import sys
import os

sys.path.append("/engram/nklab/algonauts/ethan/whole_brain_encoder")
os.chdir("/engram/nklab/algonauts/ethan/whole_brain_encoder")
import fetch
import matplotlib.pyplot as plt
import h5py
import numpy as np
import math
from tqdm import tqdm
from pathlib import Path
from scipy.stats import spearmanr
from scipy import stats
from scipy.stats import permutation_test
import statsmodels.stats.multitest as smm


def ensure_path(d, keys):
    """
    Walks down dict d following the sequence of keys.
    At each step, if the key is missing or not a dict, replaces it with {}.
    Returns the final (leaf) dict.
    """
    for key in keys:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    return d


def run_rankcorr_test(
    split,
    img_type,
    source_subj,
    target_subj,
    hemi,
    nsd_test_imgs,
    nsd_test_clip_feats,
    top_n=32,
):
    p_values = []
    encoder_p_values = []

    if isinstance(source_subj, list):
        parcel_list = set.intersection(
            *[set(fetch.get_parcel_list(subj)[hemi]) for subj in source_subj]
        )
    elif isinstance(source_subj, int):
        parcel_list = set(fetch.get_parcel_list(target_subj)[hemi]).intersection(
            set(fetch.get_parcel_list(source_subj)[hemi])
        )

    res = {}

    for i, parcel_dir in tqdm(
        enumerate(parcel_list),
        desc=f"s{source_subj} t{target_subj} {hemi}",
        total=len(parcel_list),
        leave=False,
    ):
        res[parcel_dir] = {}

        # 1) nsd ground truth activation
        ground_truth_activation = np.array(
            [img.activation[parcel_dir] for img in nsd_test_imgs]
        )
        res[parcel_dir]["ground_truth_activation"] = ground_truth_activation

        # 2) imgnet clip features
        if isinstance(source_subj, int):
            with h5py.File("clip_hypothesis/clip_embeds_patch.h5", "r") as f:
                imgnet_clip_feat = f[
                    f"{img_type}_cls/subj_{source_subj:02}/{hemi}/p{parcel_dir}"
                ][:]
        elif isinstance(source_subj, list):
            with h5py.File("clip_hypothesis/clip_embeds_patch.h5", "r") as f:
                imgnet_clip_feat = np.array(
                    [
                        f[f"{img_type}_cls/subj_{subj:02}/{hemi}/p{parcel_dir}"][:]
                        for subj in source_subj
                    ]
                )
            imgnet_clip_feat = imgnet_clip_feat.mean(axis=0)
        imgnet_clip_feat = imgnet_clip_feat[:top_n].mean(axis=0)
        imgnet_clip_feat = imgnet_clip_feat / np.linalg.norm(imgnet_clip_feat)
        res[parcel_dir]["imgnet_clip_feat"] = imgnet_clip_feat

        retrieved_ranking = nsd_test_clip_feats @ imgnet_clip_feat
        res[parcel_dir]["retrieved_ranking"] = retrieved_ranking

        retrieved_corr = spearmanr(retrieved_ranking, ground_truth_activation)[0]
        res[parcel_dir]["retrieved_corr"] = retrieved_corr

        # 3) encoder activation
        if isinstance(source_subj, int):
            encoder_acts = np.array(
                [img.model_activation[parcel_dir][source_subj] for img in nsd_test_imgs]
            )
            res[parcel_dir]["encoder_acts"] = encoder_acts
            encoder_retrieved_corr = spearmanr(ground_truth_activation, encoder_acts)[0]
            res[parcel_dir]["encoder_retrieved_corr"] = encoder_retrieved_corr

            # nulls
            n_perms = 10_000
            rnd_activation = np.random.rand(n_perms, len(ground_truth_activation))
            null_corr = np.array(
                [
                    spearmanr(rnd_activation[i], ground_truth_activation).correlation
                    for i in range(n_perms)
                ]
            )
            res[parcel_dir]["null_corr"] = null_corr

            p = (np.sum(null_corr >= retrieved_corr) + 1) / (len(null_corr) + 1)
            # print(t, p)
            p_values.append(p)

            encoder_p = (np.sum(null_corr >= encoder_retrieved_corr) + 1) / (
                len(null_corr) + 1
            )
            encoder_p_values.append(encoder_p)

    if isinstance(source_subj, int):
        p_values = np.array(p_values)
        reject, pvals_corr = smm.fdrcorrection(p_values, alpha=0.05, method="indep")
        encoder_p_values = np.array(encoder_p_values)
        encoder_reject, encoder_pvals_corr = smm.fdrcorrection(
            encoder_p_values, alpha=0.05, method="indep"
        )
    else:
        pvals_corr = None
        encoder_pvals_corr = None
    # print(
    #     f"source_s{source_subj} target_s{target_subj} {img_type} p<0.05: {np.sum(pvals_corr < 0.05)}/{len(pvals_corr)}, p<0.01: {np.sum(pvals_corr < 0.01)}/{len(pvals_corr)}"
    # )

    return encoder_pvals_corr, pvals_corr, res


def main():
    img_type = "imgnet"

    for split in ["train"]:
        # preload once
        for img_type in ["imgnet", "gen"]:
            res = {}

            for target_subj in [1, 2, 5, 7]:
                with h5py.File("clip_hypothesis/clip_embeds_patch.h5", "r") as f:
                    nsd_test_clip_feats = f[f"nsd_{split}_cls/subj_{target_subj:02}"][:]
                for hemi in ["lh", "rh"]:
                    # fetch your data
                    nsd_test_imgs, _, _ = fetch.top_NSD_imgs(
                        subj=target_subj,
                        hemi=hemi,
                        split=split,
                        split_subjs=[1, 2, 5, 7],
                    )

                    # for source_subj in [1, 2, 5, 7]: # TODO remove this
                    source_subj = [1, 2, 5, 7]
                    source_subj.remove(target_subj)
                    encoder_pvals_corr, pvals_corr, intermediates = run_rankcorr_test(
                        split,
                        img_type,
                        source_subj=source_subj,
                        target_subj=target_subj,
                        hemi=hemi,
                        nsd_test_clip_feats=nsd_test_clip_feats,
                        nsd_test_imgs=nsd_test_imgs,
                    )
                    source_subj = target_subj  # TODO remove this
                    ensure_path(res, [source_subj, target_subj, hemi])
                    res[source_subj][target_subj][hemi]["encoder_pvals_corr"] = (
                        encoder_pvals_corr
                    )
                    res[source_subj][target_subj][hemi]["pvals_corr"] = pvals_corr
                    res[source_subj][target_subj][hemi]["intermediates"] = intermediates

            np.save(
                f"clip_hypothesis/cross_subj_p/clip_retrieval_rankcorr_{img_type}_{split}_holdout.npy",
                res,
            )


if __name__ == "__main__":
    main()
