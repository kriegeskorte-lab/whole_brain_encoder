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
from scipy.stats import pearsonr as corr
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


def run_retrieval(
    top_nsd_idxs,
    source_subj,
    target_subj,
    hemi,
    split,
    img_type,
    nimgs_to_retrieve=32,
    top_n=32,
):
    p_values = []

    # parcel_list = fetch.get_parcel_list(target_subj)[hemi]
    parcel_list = set(fetch.get_parcel_list(target_subj)[hemi]).intersection(
        set(fetch.get_parcel_list(source_subj)[hemi])
    )

    # preload once
    with h5py.File("clip_hypothesis/clip_embeds_patch.h5", "r") as f:
        nsd_test_clip_feats = f[f"nsd_{split}_cls/subj_{target_subj:02}"][:]

    for i, parcel_dir in enumerate(parcel_list):
        # 1) full NSD test activations
        top_idxs = top_nsd_idxs[parcel_dir][:nimgs_to_retrieve]
        nsd_clip_retrieved = nsd_test_clip_feats[top_idxs].mean(axis=0)

        # 2) gen/img‐type CLIP retrieval
        with h5py.File("clip_hypothesis/clip_embeds_patch.h5", "r") as f:
            imgnet_clip_feat = f[
                f"{img_type}_cls/subj_{source_subj:02}/{hemi}/p{parcel_dir}"
            ][:]
        imgnet_clip_feat = imgnet_clip_feat[:top_n].mean(axis=0)

        # cosine similarity
        sim = (nsd_clip_retrieved @ imgnet_clip_feat) / (
            np.linalg.norm(nsd_clip_retrieved) * np.linalg.norm(imgnet_clip_feat)
        )
        # sim = corr(nsd_clip_retrieved, imgnet_clip_feat)[0]

        # nulls
        n_perms = 10_000
        rnds = np.random.randint(
            0,
            len(nsd_test_clip_feats),
            size=(n_perms, nimgs_to_retrieve),
        )
        samples = nsd_test_clip_feats[rnds].mean(axis=1)
        samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)
        imgnet_clip_feat /= np.linalg.norm(imgnet_clip_feat)
        null_cos_sims = samples @ imgnet_clip_feat

        null_cos_sims = np.array(null_cos_sims)

        p = (np.sum(null_cos_sims >= sim) + 1) / (len(null_cos_sims) + 1)
        # print(t, p)
        p_values.append(p)

    p_values = np.array(p_values)

    return p_values


def main():
    nimgs_to_retrieve = 32
    top_n = 32

    p_values = []

    for split in ["train"]:
        for img_type in ["imgnet", "gen"]:
            passing_parcels = {}
            for target_subj in [1, 2, 5, 7]:
                for hemi in tqdm(
                    ["lh", "rh"], desc=f"Subj {target_subj} {img_type}", leave=False
                ):
                    # fetch your data
                    nsd_test_imgs, top_nsd_idxs, top_nsd_model_idxs = (
                        fetch.top_NSD_imgs(subj=target_subj, hemi=hemi, split=split)
                    )

                    for source_subj in [1, 2, 5, 7]:
                        p_values = run_retrieval(
                            top_nsd_idxs,
                            source_subj,
                            target_subj,
                            hemi,
                            split,
                            img_type,
                            nimgs_to_retrieve=nimgs_to_retrieve,
                            top_n=top_n,
                        )

                        reject, pvals_corr = smm.fdrcorrection(
                            p_values, alpha=0.05, method="indep"
                        )
                        # print(f"s{subj} {img_type} p<0.05: {np.sum(pvals_corr < 0.05)}/{len(pvals_corr)}, p<0.01: {np.sum(pvals_corr < 0.01)}/{len(pvals_corr)}")
                        ensure_path(passing_parcels, [source_subj, target_subj])
                        passing_parcels[source_subj][target_subj][hemi] = pvals_corr

            np.save(
                f"clip_hypothesis/cross_subj_p/clip_retrieval_stats_{img_type}_{split}.npy",
                passing_parcels,
            )


if __name__ == "__main__":
    main()
