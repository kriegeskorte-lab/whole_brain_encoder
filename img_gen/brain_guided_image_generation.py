import os

# os.chdir("/engram/nklab/algonauts/ethan/whole_brain_encoder/")
import numpy as np
import torch
from brain_encoder_wrapper import BrainEncoderWrapper
from diffusers import DPMSolverMultistepScheduler
from brain_guide_pipeline import mypipelineSAG
import gc
import time
import argparse
from pathlib import Path
from attention_maps.utils import calc_overlap
from torchvision import transforms
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--rois_str", help="rois_str", type=str, default="OFA")
parser.add_argument("--subj", help="subj", type=int, default=1)
parser.add_argument(
    "--num_parcels_included", help="num_parcels_included", type=int, default=1
)
parser.add_argument(
    "--num_imgs_to_generate", help="num_imgs_to_generate", type=int, default=1000
)
parser.add_argument("--cgs", help="cgs", type=float, default=200.0)
args = parser.parse_args()
rois_str = args.rois_str
# rois_list = rois_str.split("_")

# print(rois_str, rois_list, flush=True)
#################################
# model = brain_encoder_wrapper()
# model_name = 'default'
##################################
subj = args.subj

# readout_res = "rois_all"

runs = np.arange(1, 3)
enc_output_layer = [1, 3, 5, 7]
model = BrainEncoderWrapper(subj=subj, enc_output_layer=[1, 3, 5, 7], runs=[1, 2])
model.preload_models()

for m in model.models["lh"]:
    m.to_device("cuda:0")
model.models["lh"] = [m.to("cuda:0") for m in model.models["lh"]]
for m in model.models["rh"]:
    m.to_device("cuda:1")
model.models["rh"] = [m.to("cuda:1") for m in model.models["rh"]]

# model_name = "model"
# model_name += "Voxels" if readout_res == "voxels" else ""
# model_name += "Layer" + "".join([str(cur) for cur in enc_output_layer])
# model_name += "Runs" + "".join([str(cur) for cur in runs])
##################################
# print(model_name, flush=True)


# for cur_model in model.models:
#     cur_model.lr_backbone = 1
#     for name, param in cur_model.named_parameters():
#         param.requires_grad = False
model.lr_backbone = 1
model.transform = transforms.Compose(
    [
        # transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


repo_id = "stabilityai/stable-diffusion-2-1-base"
pipe = mypipelineSAG.from_pretrained(
    repo_id, torch_dtype=torch.float16, revision="fp16"
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe2 = pipe.to("cuda:0")

# for dn in range(3):
#     device = torch.device(f"cuda:{dn}")
#     # Print allocated memory (in MB)
#     print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
#     # Print reserved memory (in MB)
#     print(f"Memory reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")


nsd_roi_labels = model.load_roi_labels()
parcels = model.load_parcels()

# overlap_values = {}
# overlap_indices = {}


def generate_imgs(
    rois_str,
    num_parcels_included=args.num_parcels_included,
    num_imgs_to_generate=args.num_imgs_to_generate,
):
    roi_mask = {
        "lh": torch.zeros(len(nsd_roi_labels["lh"][rois_str])),
        "rh": torch.zeros(len(nsd_roi_labels["rh"][rois_str])),
    }

    for hemi in ["lh", "rh"]:
        vals, indices = calc_overlap(nsd_roi_labels[hemi][rois_str], parcels[hemi])
        if (vals == 0).all():
            continue
        indices = torch.from_numpy(indices)
        # print("parcels", parcels[hemi])
        # print("parcel indices", indices[:num_parcels_included])
        for parcel in [parcels[hemi][i.item()] for i in indices[:num_parcels_included]]:
            roi_mask[hemi][parcel] = 1

    def loss_function(image_input):
        outputs = model.forward(image_input, use_dataloader=False)
        roi_acts = 0
        for hemi in ["lh", "rh"]:
            outputs[hemi] = outputs[hemi].cpu()
            roi_acts += torch.mean(outputs[hemi] * roi_mask[hemi], dim=1).sum()

        return -roi_acts

    fld = Path("/engram/nklab/algonauts/ethan/images")
    # fld = fld / f"{rois_str}_top{num_parcels_included}_s{args.subj}"
    cgs = args.cgs
    fld = fld / rois_str
    fld = fld / f"cgs_{int(cgs)}"
    fld = fld / f"topparcels_{num_parcels_included}"
    fld = fld / f"subj_{args.subj:02}"
    fld.mkdir(exist_ok=True, parents=True)
    pipe.brain_tweak = loss_function

    time_st = time.time()
    num_imgs_per_seed = 4

    for seed in tqdm(
        range(num_imgs_to_generate // num_imgs_per_seed), desc="Image Generation"
    ):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        g = torch.Generator(device="cuda").manual_seed(int(seed))
        image = pipe(
            "",
            sag_scale=0.75,
            guidance_scale=0.0,
            num_inference_steps=50,
            generator=g,
            clip_guidance_scale=cgs,
            num_images_per_prompt=num_imgs_per_seed,
        )

        for i, im in enumerate(image.images):
            im.save(
                fld / f"seed{seed}_{i}.png",
                format="PNG",
                compress_level=6,
            )
        # print("generated", len(image.images), "images")

        # fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        # ax.imshow(image.images[0])
    print(time.time() - time_st)  # 240/3 #29.9-30.1


if __name__ == "__main__":
    neural_data_path = Path(
        "/engram/nklab/datasets/natural_scene_dataset/model_training_datasets/neural_data"
    )
    metadata = np.load(
        neural_data_path / f"metadata_sub-{subj:02}.npy", allow_pickle=True
    ).item()

    if rois_str == "all":
        for rois_str in metadata["lh_rois"]:
            generate_imgs(rois_str)

    else:
        generate_imgs(rois_str)
