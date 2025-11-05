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
        for chunk in tqdm(torch.split(imgs, batch_size), desc="embedding images"):
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
    f"/engram/nklab/algonauts/ethan/whole_brain_encoder/clip_hypothesis/rewrite/retrieval_stats_test_s{subj}.h5",
    "w",
) as f:
    for split in ["test", "train"]:
        for hemi in tqdm(["lh", "rh"], desc=f"Processing s{subj} {split}"):
            nsd, _, _ = fetch.top_NSD_imgs(
                subj=subj, hemi=hemi, split=split, split_subjs=[subj]
            )

            key = f"{split}/subj_{subj:02}/{hemi}/ground_truth_activations"
            gt_acts = np.array([[img.activation[i] for i in range(501)] for img in nsd])
            write_h5(f, gt_acts, key)
            print(f"Written {key} with shape {gt_acts.shape}")

            key = f"{split}/subj_{subj:02}/{hemi}/model_activations"
            model_acts = np.array(
                [[img.model_activation[i][subj] for i in range(501)] for img in nsd]
            )
            write_h5(f, model_acts, key)
            print(f"Written {key} with shape {model_acts.shape}")

        key = f"{split}/subj_{subj:02}/clip"

        nsd_imgs = [img.img for img in nsd]
        preprocessed_nsd_imgs = [preprocess(Image.fromarray(img)) for img in nsd_imgs]
        nsd_imgs_cat = torch.stack(preprocessed_nsd_imgs)

        cls = embed_imgs(nsd_imgs_cat)
        write_h5(f, cls, key)
        print(f"Written {key} with shape {cls.shape}")

        key = f"{split}/subj_{subj:02}/image"
        nsd_imgs = np.array(nsd_imgs)
        write_h5(f, nsd_imgs, key)
        print(f"Written {key} with shape {nsd_imgs.shape}")
