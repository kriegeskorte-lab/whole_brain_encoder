from pathlib import Path
import numpy as np
import h5py
import os
import utils.args as args
from tqdm import tqdm

model_results_dir = Path("/engram/nklab/algonauts/ethan/whole_brain_encoder/results")


def get_parcel_list(subj, parcel_strategy="schaefer"):
    base_parcels_dir = (
        Path("/engram/nklab/algonauts/ethan/whole_brain_encoder/img_gen")
        / parcel_strategy
        / "candidate_parcels"
    )
    parcels = {}

    for hemi in ["lh", "rh"]:
        parcels_dir = base_parcels_dir / f"subj_{subj:02}" / hemi
        parcels[hemi] = sorted([int(p.stem) for p in parcels_dir.glob("*.npy")])

    return parcels


def all_parcel_list(subj, hemi, parcel_strategy="schaefer"):
    parcel_list = get_parcel_list(subj, parcel_strategy)[hemi] + list(
        overlap_labeled_parcels(subj, hemi, parcel_strategy).keys()
    )
    return parcel_list


def overlap_labeled_parcels(
    subj, hemi, return_parcel_num=False, parcel_strategy="schaefer"
):
    base_parcels_dir = (
        Path("/engram/nklab/algonauts/ethan/whole_brain_encoder/img_gen")
        / parcel_strategy
        / "overlap_labeled_parcels"
    )
    parcels_dir = base_parcels_dir / f"subj_{subj:02}" / hemi

    parcels = {}
    for parcel_path in parcels_dir.glob("*.npy"):
        p = np.load(parcel_path, allow_pickle=True).item()
        if return_parcel_num:
            parcels[parcel_path.stem] = p["parcel_num"]
        else:
            parcels[parcel_path.stem] = p["parcel"]

    return parcels


def parcel_dir(subj, hemi, parcel_dir, parcel_strategy="schaefer"):
    base_parcels_dir = (
        Path("/engram/nklab/algonauts/ethan/whole_brain_encoder/img_gen/")
        / parcel_strategy
    )
    if isinstance(parcel_dir, int) or parcel_dir.isdigit():
        base_parcels_dir = base_parcels_dir / "candidate_parcels"
    else:
        base_parcels_dir = base_parcels_dir / "overlap_labeled_parcels"
    parcels_dir = base_parcels_dir / f"subj_{subj:02}" / hemi

    return parcels_dir


def parcel(subj, hemi, parcel_path, parcel_strategy="schaefer"):
    parcel_info_dir = parcel_dir(
        subj, hemi, parcel_path, parcel_strategy=parcel_strategy
    )
    parcel = np.load(parcel_info_dir / f"{parcel_path}.npy", allow_pickle=True).item()

    return parcel


def all_parcels(subj, hemi, strategy):
    parcels = {}
    for hemi in ["lh", "rh"]:
        parcels[hemi] = np.load(
            Path("/engram/nklab/algonauts/ethan/whole_brain_encoder")
            / "parcels"
            / strategy
            / f"{hemi}_labels_s{subj:02}.npy",
            allow_pickle=True,
        ).item()["parcels"]
    return parcels


def top_imgs_grid_path(
    imgtype, nimgs, subj, hemi, parcel_dir, parcel_strategy="schaefer", cgs=130
):
    if imgtype not in ["nsd", "imgnet", "generated"]:
        print(f"imgtype {imgtype} not recognized")
        return None
    base_imgs_dir = Path("/engram/nklab/algonauts/ethan/images/_curated")
    imgs_dir = base_imgs_dir / parcel_strategy / f"top_{imgtype}_imgs_grid"
    imgs_dir = imgs_dir / f"subj_{subj:02}"
    imgs_dir = imgs_dir / hemi

    if not imgs_dir.exists():
        print(f"imgs_dir {imgs_dir} does not exist")
        return None

    filepath = imgs_dir / f"top{nimgs}_p{parcel_dir}_cgs{int(cgs)}.png"
    if not filepath.exists():
        print(f"file does not exist in {imgs_dir}")
        return None

    return filepath


def pretty_summary(subj, hemi, parcel_name, parcel_strategy="schaefer"):
    base_imgs_dir = (
        Path(
            f"/engram/nklab/algonauts/ethan/images/_curated/{parcel_strategy}/_pretty_overlap_labeled_parcels_reranked_cgs130"
        )
        / f"subj_{subj:02}"
        / hemi
    )
    matching_files = list(base_imgs_dir.glob(f"*_p{parcel_name}_*"))
    assert len(matching_files) <= 1, (
        f"Searching for {parcel_name}, more than one matching file found at {base_imgs_dir}: {[p.name for p in matching_files]}!"
    )

    return matching_files[0] if matching_files else None


def common_parcels_summary_dir(num_subj, hemi, parcel_name, parcel_strategy="schaefer"):
    return (
        Path("/engram/nklab/algonauts/ethan/images/_curated/schaefer/_common_parcels")
        / str(num_subj)
        / hemi
        / str(parcel_name)
    )


def imagenet_activations_path(subj, parcel_strategy):
    imagenet_results_dir = (
        model_results_dir
        / parcel_strategy
        / "enc_1_3_5_7_run_1_2/imagenet"
        / f"subj_{subj:02}"
    )
    fp = imagenet_results_dir / "activation_dist.h5"
    return fp


def imagenet_paths(subj, parcel_strategy):
    fp = imagenet_activations_path(subj, parcel_strategy)
    with h5py.File(fp, "r") as f:
        return f["img_paths"][()]


def imagenet_activations(subj, hemi, parcel_dir, parcel_strategy="schaefer"):
    fp = imagenet_activations_path(subj, parcel_strategy)
    with h5py.File(fp, "r") as f:
        return f[hemi][parcel_dir][()]


def metadata(subj):
    neural_data_path = Path(
        "/engram/nklab/datasets/natural_scene_dataset/model_training_datasets/neural_data"
    )

    metadata = np.load(
        neural_data_path / f"metadata_sub-{subj:02}.npy", allow_pickle=True
    ).item()

    return metadata


def gen_imgs_dir(subj, hemi, parcel_dir, cgs=130, parcel_strategy="schaefer"):
    base_imgs_dir = Path("/engram/nklab/algonauts/ethan/images/")
    imgs_dir = base_imgs_dir / "unlabeled_parcels"
    imgs_dir = imgs_dir / parcel_strategy
    imgs_dir = imgs_dir / f"subj_{subj:02}"
    imgs_dir = imgs_dir / hemi
    imgs_dir = imgs_dir / str(parcel_dir)
    imgs_dir = imgs_dir / f"cgs_{int(cgs)}"

    return imgs_dir


def gen_imgs_activations(subj, hemi, parcel_dir, cgs=130, parcel_strategy="schaefer"):
    imgs_dir = gen_imgs_dir(subj, hemi, parcel_dir, cgs, parcel_strategy)
    if not imgs_dir.exists():
        print(f"imgs_dir {imgs_dir} does not exist")
        return None

    if not (imgs_dir / "activations.npy").exists():
        print(f"activations.npy does not exist in {imgs_dir}")
        return None

    activations = np.load(imgs_dir / "activations.npy", allow_pickle=True).item()
    return activations


def nsd_activations(subj, split="test", parcel_strategy="schaefer"):
    model_dir = (
        model_results_dir / parcel_strategy / "enc_1_3_5_7_run_1_2/" / f"subj_{subj:02}"
    )
    model_test_file = np.load(model_dir / f"{split}.npy", allow_pickle=True).item()

    return model_test_file


def ensemble_model_dir(subj, strategy, enc_output_layers=[1, 3, 5, 7], runs=[1, 2]):
    save_dir = (
        Path("/engram/nklab/algonauts/ethan/whole_brain_encoder/results/")
        / strategy
        / f"enc_{'_'.join([str(s) for s in enc_output_layers])}_run_{'_'.join([str(s) for s in runs])}"
        / f"subj_{subj:02}"
    )
    return save_dir


def split_corr(
    subj, strategy, enc_output_layers=[1, 3, 5, 7], runs=[1, 2], split="test"
):
    save_dir = ensemble_model_dir(subj, strategy, enc_output_layers, runs)
    test_corr = np.load(save_dir / f"{split}_corr_avg.npy", allow_pickle=True).item()

    return test_corr


def nsd_data(subj, hemi, split="test"):
    from datasets.nsd import nsd_dataset_avg


    os.chdir("/engram/nklab/algonauts/ethan/whole_brain_encoder")
    a = args.get_default_args()
    a.subj = subj
    a.hemi = hemi

    if isinstance(split, str):
        split = [split]
    
    result = []
    for sp in split:
        data = []
        dataset = nsd_dataset_avg(a, split=sp)

        try:
            import torch

            dataset = torch.utils.data.DataLoader(
                dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
            )
            for img, fmri in tqdm(dataset, desc=f"loading NSD {sp} data", leave=False):
                img_data = {}
                img_data["img"] = img
                img_data["betas"] = fmri["betas"]

                data.append(img_data)


        except:
            for img, fmri in tqdm(dataset, desc=f"loading NSD {sp} data", leave=False):
                img_data = {}
                img_data["img"] = img
                img_data["betas"] = fmri["betas"]

                data.append(img_data)

        data = {key: torch.cat([d[key] for d in data], dim=0).numpy() for key in data[0]}
        result.append(data)
    
    if len(result) == 1:
        return result[0]
    else:
        data = {key: np.concatenate([d[key] for d in result], axis=0) for key in result[0]}

    return data


def nsd_labeled_area_mask(subj, hemi):
    metadata_file = metadata(subj)

    la = np.zeros(163842, dtype=bool)
    for roi in metadata_file[f"{hemi}_rois"]:
        la = np.logical_or(la, metadata_file[f"{hemi}_rois"][roi])

    return la
