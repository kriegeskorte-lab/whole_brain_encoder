import sys
import os

sys.path.append("/engram/nklab/algonauts/ethan/whole_brain_encoder")
os.chdir("/engram/nklab/algonauts/ethan/whole_brain_encoder")
import fetch
import h5py
from PIL import Image
import numpy as np
from tqdm import tqdm
import open_clip
import torch
import torch.nn.functional as F
import argparse


parser = argparse.ArgumentParser(description="Train model for a given subject")
parser.add_argument("--subject", "-s", type=int, required=True, help="Subject ID")
args = parser.parse_args()
subj = args.subject

model, preprocess = open_clip.create_model_from_pretrained(
    "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384"
)
model = model.cuda()
model.eval()
tokenizer = open_clip.get_tokenizer("ViT-H-14")
model.visual.output_tokens = True


def embed_imgs(imgs, batch_size=64):
    all_cls = []

    with torch.no_grad(), torch.amp.autocast("cuda"):
        for chunk in torch.split(imgs, batch_size):
            c, f = model.visual(chunk.cuda())
            all_cls.append(c.detach().cpu())
            # all_feat.append(f.detach().cpu())
        cls = torch.cat(all_cls, dim=0)
        # img_feat = torch.cat(all_feat, dim=0)

        # img_feat = F.normalize(img_feat.flatten(start_dim=1), dim=-1).cpu().numpy()
        cls = F.normalize(cls, dim=-1).cpu().numpy()

    return cls


def write_h5(f, data, path, **kwargs):
    grp = f.require_group(os.path.dirname(path))
    if os.path.basename(path) in grp:
        del grp[os.path.basename(path)]
    grp.create_dataset(os.path.basename(path), data=data, **kwargs)


with h5py.File(
    f"/engram/nklab/algonauts/ethan/whole_brain_encoder/clip_hypothesis/rewrite/retrieval_stats_test_imgnet_gen_s{subj}.h5",
    "w",
) as f:
    for hemi in ["lh", "rh"]:
        for parcel_dir in tqdm(
            fetch.get_parcel_list(subj)[hemi], desc=f"Processing s{subj} {hemi}"
        ):
            for imgtype, top_imgtype_imgs in zip(
                ["imagenet", "gen"],
                [fetch.top_imgnet_imgs, fetch.top_generated_imgs],
            ):
                top_imgnet_imgs, top_imgnet_paths, top_imgnet_activations = (
                    top_imgtype_imgs(subj, hemi, parcel_dir, max_num_imgs=64)
                )

                key = f"{imgtype}/subj_{subj:02}/{hemi}/p{parcel_dir}/image"
                arr_imgs = np.array(
                    [np.array(img.convert("RGB")) for img in top_imgnet_imgs]
                )
                write_h5(f, arr_imgs, key)

                key = f"{imgtype}/subj_{subj:02}/{hemi}/p{parcel_dir}/clip"
                preprocessed_imgs = [preprocess(img) for img in top_imgnet_imgs]
                preprocessed_imgs = torch.stack(preprocessed_imgs)
                cls = embed_imgs(preprocessed_imgs)
                write_h5(f, cls, key)

                key = f"{imgtype}/subj_{subj:02}/{hemi}/p{parcel_dir}/path"
                top_imgnet_paths = np.array(
                    [
                        p.decode("utf-8")
                        if isinstance(p, (bytes, bytearray))
                        else str(p)
                        for p in top_imgnet_paths
                    ],
                    dtype=object,
                )
                write_h5(f, top_imgnet_paths, key, dtype=h5py.special_dtype(vlen=str))

                key = f"{imgtype}/subj_{subj:02}/{hemi}/p{parcel_dir}/model_activations"
                top_imgnet_activations = np.array(top_imgnet_activations)
                write_h5(f, top_imgnet_activations, key)
