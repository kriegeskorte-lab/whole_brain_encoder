from img_gen.utils import BrainGuidedImageGenerator
import argparse
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--subj", type=int, default=1)
parser.add_argument("--hemi", type=str, default="lh")
parser.add_argument("--cgs", type=float, default=130)
parser.add_argument("--parcel_dir", type=str, default="parcel_0")
parser.add_argument("--num_imgs_to_generate", type=int, default=400)
parser.add_argument("--parcel_strategy", type=str, default="schaefer")
args = parser.parse_args()

if args.parcel_dir.isdigit():
    parcel_dir = (
        Path("/engram/nklab/algonauts/ethan/whole_brain_encoder/img_gen/")
        / args.parcel_strategy
        / "candidate_parcels"
    )
else:
    parcel_dir = (
        Path("/engram/nklab/algonauts/ethan/whole_brain_encoder/img_gen")
        / args.parcel_strategy
        / "overlap_labeled_parcels"
    )
parcel_info = np.load(
    parcel_dir / f"subj_{args.subj:02}" / args.hemi / f"{args.parcel_dir}.npy",
    allow_pickle=True,
).item()

# print(parcel_info)
roi_mask = np.zeros(163842)
roi_mask[parcel_info["parcel"]] = 1
roi_name = str(args.parcel_dir)

model = BrainGuidedImageGenerator(args.subj, args.hemi, roi_mask, roi_name, args.cgs)
model.generate_imgs(args.num_imgs_to_generate)
