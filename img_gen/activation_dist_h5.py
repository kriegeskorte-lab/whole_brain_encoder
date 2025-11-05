import sys

import os

sys.path.insert(0, "/engram/nklab/algonauts/ethan/whole_brain_encoder")
os.chdir("/engram/nklab/algonauts/ethan/whole_brain_encoder")

import numpy as np
from pathlib import Path
import h5py
from tqdm import tqdm
import fetch
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor

import os
import argparse

os.chdir("/engram/nklab/algonauts/ethan/whole_brain_encoder")


def recursively_save_dict_to_group(group, dict_obj):
    """
    Recursively saves a dictionary to an HDF5 group.

    Parameters:
    - group: The current h5py.Group or h5py.File where data is saved.
    - dict_obj: The dictionary to save.
    """
    for key, item in dict_obj.items():
        # If the item is a dictionary, create a subgroup and call the function recursively.
        key = str(key)
        if isinstance(item, dict):
            subgroup = group.create_group(key)
            recursively_save_dict_to_group(subgroup, item)
        else:
            # If the item is a list, convert it to a numpy array.
            if isinstance(item, list):
                item = [str(i) for i in item]
                vlen_str = h5py.special_dtype(vlen=str)
                item = np.array(item, dtype=vlen_str)
            # For strings, use the proper HDF5 string data type.
            if isinstance(item, str):
                dt = h5py.string_dtype(encoding="utf-8")
                group.create_dataset(key, data=item, dtype=dt)
            elif isinstance(item, np.ndarray):
                try:
                    group.create_dataset(key, data=item)
                except Exception as e:
                    print(e)
                    print(f"Item {key} {item} is a problem")
            else:
                print(f"Item {key} {item} is a problem")


if __name__ == "__main__":
    parcel_strategy = "schaefer"
    # parcel_strategy = "kmeans"
    parser = argparse.ArgumentParser(description="NSD Training", add_help=False)
    parser.add_argument("--subj", default=1, type=int)
    args = parser.parse_args()
    subj = args.subj

    imagenet_results_dir = (
        Path("/engram/nklab/algonauts/ethan/whole_brain_encoder/results/")
        / parcel_strategy
        / "enc_1_3_5_7_run_1_2"
        / "imagenet_linear/"
        / f"subj_{subj:02}"
    )

    acts = {"lh": {}, "rh": {}, "img_paths": []}
    for hemi in ["lh", "rh"]:
        overlap_parcels = fetch.overlap_labeled_parcels(
            subj, hemi, return_parcel_num=True, parcel_strategy=parcel_strategy
        )
        for key in overlap_parcels:
            acts[hemi][key] = []
        for key in fetch.get_parcel_list(subj, parcel_strategy)[hemi]:
            acts[hemi][key] = []

    def process_activation_file(imagenet_activations_path):
        try:
            act = np.load(imagenet_activations_path, allow_pickle=True).item()
        except Exception as e:
            print(f"Error loading {imagenet_activations_path}: {e}")
            return None
        result = {
            "img_paths": act["img_paths"],
            "lh": {},
            "rh": {},
        }
        for hemi in ["lh", "rh"]:
            overlap_parcels = fetch.overlap_labeled_parcels(
                subj, hemi, return_parcel_num=True, parcel_strategy=parcel_strategy
            )
            for parcel_name, parcel_num in overlap_parcels.items():
                result[hemi].setdefault(parcel_name, []).append(
                    act["parcel_mean_activity"][hemi][:, parcel_num]
                )

            for key in fetch.get_parcel_list(subj, parcel_strategy)[hemi]:
                result[hemi].setdefault(key, []).append(
                    act["parcel_mean_activity"][hemi][:, key]
                )
        return result

    files = list(sorted(imagenet_results_dir.glob("*.npy")))

    results = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_activation_file, file) for file in files]
        with tqdm(total=len(files), desc=f"Subj {subj}", leave=False) as pbar:
            for future in as_completed(futures):
                if future.result() is not None:
                    results.append(future.result())
                pbar.update(1)

    for res in results:
        acts["img_paths"] += res["img_paths"]
        for hemi in ["lh", "rh"]:
            for key, values in res[hemi].items():
                acts[hemi][key] += values

    for hemi in ["lh", "rh"]:
        for key, data in acts[hemi].items():
            acts[hemi][key] = np.concatenate(data, axis=0)

    with h5py.File(imagenet_results_dir / "activation_dist.h5", "w") as f:
        recursively_save_dict_to_group(f, acts)

    print(f"Saved {imagenet_results_dir / 'activation_dist.h5'}")
    print(f"Number of img_paths: {len(acts['img_paths'])}")
