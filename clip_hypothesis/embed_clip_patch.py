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
import gc

model, preprocess = open_clip.create_model_from_pretrained(
    "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384"
)
model = model.cuda()
tokenizer = open_clip.get_tokenizer("ViT-H-14")
model.visual.output_tokens = True

split = "train"

batch_size = 64

fp = "clip_hypothesis/clip_embeds_patch.h5"

with h5py.File(fp, "r") as f:
    grp = f.get(f"nsd_{split}_cls", None)
    if grp is not None:
        done = set(grp.keys())

for subj in [1, 2, 5, 7]:
    if f"subj_{subj:02}" in done:
        print(f"subj_{subj:02} already done")
        continue

    nsd_imgs = fetch.nsd_data(subj=subj, hemi="lh", split=split)
    nsd_imgs = [preprocess(Image.fromarray(img)) for img in nsd_imgs["img"]]
    nsd_imgs_cat = torch.stack(nsd_imgs)
    model.visual.output_tokens = True

    all_cls, all_feat = [], []
    with torch.no_grad(), torch.cuda.amp.autocast():
        # <-- replaced single forward with a tiny loop
        for chunk in torch.split(nsd_imgs_cat, batch_size):
            c, f = model.visual(chunk.cuda())
            all_cls.append(c.detach().cpu())
            all_feat.append(f.detach().cpu())
        cls = torch.cat(all_cls, dim=0)
        img_feat = torch.cat(all_feat, dim=0)

        img_feat = F.normalize(img_feat.flatten(start_dim=1), dim=-1).cpu().numpy()
        cls = F.normalize(cls, dim=-1).cpu().numpy()

        # cls, img_feat = model.visual(nsd_imgs_cat.cuda())
        # img_feat = F.normalize(img_feat.flatten(start_dim=1), dim=-1)
        # img_feat = img_feat.cpu().numpy()
        # cls = F.normalize(cls, dim=-1)
        # cls = cls.cpu().numpy()
    with h5py.File("clip_hypothesis/clip_embeds_patch.h5", "a") as f:
        if f"nsd_{split}/subj_{subj:02}" in f:
            del f[f"nsd_{split}/subj_{subj:02}"]
        f.require_group(f"nsd_{split}")
        f[f"nsd_{split}/subj_{subj:02}"] = img_feat
        f.require_group(f"nsd_{split}_cls")
        f[f"nsd_{split}_cls/subj_{subj:02}"] = cls
        print(f"nsd_{split}_cls/subj_{subj:02} saved")

    del cls, img_feat
    del nsd_imgs, nsd_imgs_cat
    torch.cuda.empty_cache()

    gc.collect()

# for subj in range(1, 9):
#     for hemi in ["lh", "rh"]:
#         for parcel_dir in tqdm(
#             fetch.get_parcel_list(subj)[hemi], desc=f"Subj {subj} {hemi}", leave=False
#         ):
#             imgs, img_paths, img_acts = fetch.top_generated_imgs(
#                 subj=subj, hemi=hemi, parcel_dir=parcel_dir, max_num_imgs=64
#             )
#             imgs = [preprocess(img) for img in imgs]
#             imgs_cat = torch.stack(imgs).cuda()
#             with torch.no_grad(), torch.cuda.amp.autocast():
#                 cls, img_feat = model.visual(imgs_cat)
#             img_feat = F.normalize(img_feat.flatten(start_dim=1), dim=-1)
#             img_feat = img_feat.cpu().numpy()
#             cls = F.normalize(cls, dim=-1)
#             cls = cls.cpu().numpy()
#             with h5py.File("clip_hypothesis/clip_embeds_patch.h5", "a") as f:
#                 # if f"gen/subj_{subj:02}/{hemi}/p{parcel_dir}" in f:
#                 #     del f[f"gen/subj_{subj:02}/{hemi}/p{parcel_dir}"]
#                 # f[f"gen/subj_{subj:02}/{hemi}/p{parcel_dir}"] = img_feat
#                 if f"gen_cls/subj_{subj:02}/{hemi}/p{parcel_dir}" in f:
#                     del f[f"gen_cls/subj_{subj:02}/{hemi}/p{parcel_dir}"]
#                 f[f"gen_cls/subj_{subj:02}/{hemi}/p{parcel_dir}"] = cls

#             imgs, img_paths, img_acts = fetch.top_imgnet_imgs(
#                 subj=subj, hemi=hemi, parcel_dir=parcel_dir, max_num_imgs=64
#             )
#             imgs = [preprocess(img) for img in imgs]
#             imgs_cat = torch.stack(imgs).cuda()
#             with torch.no_grad(), torch.cuda.amp.autocast():
#                 cls, img_feat = model.visual(imgs_cat)
#             img_feat = F.normalize(img_feat.flatten(start_dim=1), dim=-1)
#             img_feat = img_feat.cpu().numpy()
#             cls = F.normalize(cls, dim=-1)
#             cls = cls.cpu().numpy()
#             with h5py.File("clip_hypothesis/clip_embeds_patch.h5", "a") as f:
#                 # if f"imgnet/subj_{subj:02}/{hemi}/p{parcel_dir}" in f:
#                 #     del f[f"imgnet/subj_{subj:02}/{hemi}/p{parcel_dir}"]
#                 # f[f"imgnet/subj_{subj:02}/{hemi}/p{parcel_dir}"] = img_feat
#                 if f"imgnet_cls/subj_{subj:02}/{hemi}/p{parcel_dir}" in f:
#                     del f[f"imgnet_cls/subj_{subj:02}/{hemi}/p{parcel_dir}"]
#                 f[f"imgnet_cls/subj_{subj:02}/{hemi}/p{parcel_dir}"] = cls
