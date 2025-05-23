import sys
import os

sys.path.append("/engram/nklab/algonauts/ethan/whole_brain_encoder")
os.chdir("/engram/nklab/algonauts/ethan/whole_brain_encoder")
import fetch
import torch
from PIL import Image
import open_clip
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import h5py

model, preprocess = open_clip.create_model_from_pretrained(
    "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384"
)
model = model.cuda()
tokenizer = open_clip.get_tokenizer("ViT-H-14")

for subj in range(1, 9):
    for hemi in ["lh", "rh"]:
        for parcel_dir in tqdm(
            fetch.get_parcel_list(subj)[hemi], desc=f"Subj {subj} {hemi}", leave=False
        ):
            imgs, img_paths, img_acts = fetch.top_generated_imgs(
                subj=subj, hemi=hemi, parcel_dir=parcel_dir, max_num_imgs=64
            )
            imgs = [preprocess(img) for img in imgs]
            imgs_cat = torch.stack(imgs).cuda()
            with torch.no_grad(), torch.cuda.amp.autocast():
                img_feat = model.encode_image(imgs_cat)
            img_feat = F.normalize(img_feat, dim=-1)
            img_feat = img_feat.cpu().numpy()
            with h5py.File("clip_hypothesis/clip_embeds.h5", "a") as f:
                if f"gen_cls/subj_{subj:02}/{hemi}/p{parcel_dir}" in f:
                    del f[f"gen_cls/subj_{subj:02}/{hemi}/p{parcel_dir}"]
                f[f"gen_cls/subj_{subj:02}/{hemi}/p{parcel_dir}"] = img_feat

            imgs, img_paths, img_acts = fetch.top_imgnet_imgs(
                subj=subj, hemi=hemi, parcel_dir=parcel_dir, max_num_imgs=64
            )
            imgs = [preprocess(img) for img in imgs]
            imgs_cat = torch.stack(imgs).cuda()
            with torch.no_grad(), torch.cuda.amp.autocast():
                img_feat = model.encode_image(imgs_cat)
            img_feat = F.normalize(img_feat, dim=-1)
            img_feat = img_feat.cpu().numpy()
            with h5py.File("clip_hypothesis/clip_embeds.h5", "a") as f:
                if f"imgnet_cls/subj_{subj:02}/{hemi}/p{parcel_dir}" in f:
                    del f[f"imgnet_cls/subj_{subj:02}/{hemi}/p{parcel_dir}"]
                f[f"imgnet_cls/subj_{subj:02}/{hemi}/p{parcel_dir}"] = img_feat