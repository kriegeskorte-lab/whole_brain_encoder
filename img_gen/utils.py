from brain_encoder_wrapper import BrainEncoderWrapper
from torchvision import transforms
import torch
from diffusers import DPMSolverMultistepScheduler
from pathlib import Path
from tqdm import tqdm
import numpy as np
import sys
import gc

sys.path.append("/engram/nklab/algonauts/ethan/whole_brain_encoder/img_gen")
from brain_guide_pipeline import mypipelineSAG


class BrainGuidedImageGenerator:
    def __init__(
        self,
        subj,
        hemi,
        roi_mask,
        roi_name,
        cgs,
        runs=[1, 2],
        enc_output_layer=[1, 3, 5, 7],
        save_dir=None,
        parcel_strategy="schaefer",
    ):
        self.subj = subj
        self.hemi = hemi
        self.roi_mask = roi_mask
        if isinstance(self.roi_mask, np.ndarray):
            self.roi_mask = torch.tensor(self.roi_mask)
        if self.roi_mask.dtype != torch.bool:
            self.roi_mask = self.roi_mask.bool()

        self.roi_name = roi_name
        self.cgs = cgs
        self.num_imgs_per_seed = 4

        self.model = BrainEncoderWrapper(
            subj=subj,
            enc_output_layer=enc_output_layer,
            runs=runs,
            num_gpus=2,
            parcel_strategy=parcel_strategy,
        )
        self.model.lr_backbone = 1
        self.model.transform = transforms.Compose(
            [
                # transforms.ToTensor(),
                # transforms.Resize(425),
                # transforms.CenterCrop(425),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # self.nsd_roi_labels = self.model.load_roi_labels()
        # self.parcels = self.model.load_parcels()

        repo_id = "stabilityai/stable-diffusion-2-1-base"
        self.pipe = mypipelineSAG.from_pretrained(
            repo_id, torch_dtype=torch.float16, revision="fp16"
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.brain_tweak = self.loss_function
        pipe2 = self.pipe.to("cuda:0")

        if save_dir is None:
            self.save_dir = Path("/engram/nklab/algonauts/ethan/images")
            cgs = self.cgs
            self.save_dir = self.save_dir / "unlabeled_parcels"
            self.save_dir = self.save_dir / parcel_strategy
            self.save_dir = self.save_dir / f"subj_{self.subj:02}"
            self.save_dir = self.save_dir / self.hemi
            self.save_dir = self.save_dir / self.roi_name
            self.save_dir = self.save_dir / f"cgs_{int(cgs)}"
            self.save_dir.mkdir(exist_ok=True, parents=True)
        else:
            self.save_dir = save_dir

        print("Saving to", self.save_dir, flush=True)

    def loss_function(self, image_input):
        outputs = self.model.forward_hemi(self.hemi, image_input, use_dataloader=False)
        outputs = outputs.cpu()
        roi_acts = torch.mean(outputs[:, self.roi_mask], dim=1)

        return -roi_acts

    def generate_imgs(self, num_imgs_to_generate):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        for seed in tqdm(
            range(num_imgs_to_generate),
            desc="Image Generation",
        ):
            if (self.save_dir / f"seed{seed}.png").exists():
                continue

            image = self.generate_step(seed)
            image[0].save(
                self.save_dir / f"seed{seed}.png",
                format="PNG",
                compress_level=6,
            )

    def generate_step(self, seed):
        g = torch.Generator(device="cuda").manual_seed(int(seed))
        image = self.pipe(
            "",
            sag_scale=0.75,
            guidance_scale=0.0,
            num_inference_steps=50,
            generator=g,
            clip_guidance_scale=self.cgs,
            # num_images_per_prompt=self.num_imgs_per_seed,
        )

        return image.images
