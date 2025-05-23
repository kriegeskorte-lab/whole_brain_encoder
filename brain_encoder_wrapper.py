import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F

from models.activations import get_transformer_activations
from models.brain_encoder import brain_encoder

from datasets.nsd import nsd_dataset_custom, nsd_dataset_avg, nsd_dataset
from engine import evaluate
from scipy.special import softmax
from utils.args import get_model_dir, get_args_parser, get_default_args
from pathlib import Path, PosixPath
import argparse
import copy
from tqdm import tqdm
from scipy.stats import pearsonr as corr
from huggingface_hub import snapshot_download
import shutil
import fetch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from scipy import ndimage

import h5py


# argparser needs: subj
class BrainEncoderWrapper:
    def __init__(
        self,
        subj=1,
        backbone_arch="dinov2_q",
        encoder_arch="transformer",
        enc_output_layer=[1, 3, 5, 7],
        runs=[1, 2],
        num_gpus=2,
        parcel_strategy="schaefer",
    ):
        torch.serialization.add_safe_globals([argparse.Namespace, PosixPath])
        parser = get_args_parser()
        default_args = {
            action.dest: action.default
            for action in parser._actions
            if action.dest != "help"
        }
        args = argparse.Namespace(**default_args)

        self.enc_output_layer = enc_output_layer
        self.runs = runs
        self.subj = subj
        self.default_args = args
        self.neural_data_path = Path(args.data_dir)
        self.lr_backbone = None
        self.backbone_arch = backbone_arch

        self.metadata = np.load(
            Path(args.data_dir) / f"metadata_sub-{self.subj:02}.npy", allow_pickle=True
        ).item()

        self.num_voxels = len(self.metadata["lh_anterior_vertices"]) + len(
            self.metadata["lh_posterior_vertices"]
        )

        if "radio" in backbone_arch:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(496),
                    transforms.CenterCrop(496),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        elif "clip" in backbone_arch:
            import open_clip

            _, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai"
            )

            self.transform = transforms.Compose(
                [transforms.ToPILImage()] + list(preprocess.transforms)
            )
        elif backbone_arch == "dinov2_q":
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(425),
                    transforms.CenterCrop(425),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),  # convert the images to a PyTorch tensor
                    transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),  # normalize the images color channels
                ]
            )

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model_paths = {
            "lh": [],
            "rh": [],
        }
        print(f"backbone arch: {self.backbone_arch}")
        for hemi in ["lh", "rh"]:
            for r in runs:
                for layer_num in self.enc_output_layer:
                    model_path = get_model_dir(
                        args.output_path,
                        self.backbone_arch,
                        encoder_arch,
                        self.subj,
                        layer_num,
                        r,
                        hemi,
                        parcel_strategy,
                    )
                    if self.is_valid_model(model_path, hemi):
                        self.model_paths[hemi].append(model_path)

                    else:
                        if not (
                            1 <= self.subj <= 8
                            and r in [1, 2]
                            and layer_num in [1, 3, 5, 7]
                            and parcel_strategy == "checkpoints"
                        ):
                            print(f"WARNING: Model path {model_path} is not valid")
                            continue

                        print(
                            f"Downloading checkpoint from huggingface into checkpoints/nsd_test/dinov2_q_transformer/subj_{self.subj:02}/enc_{layer_num}/run_{r}/{hemi}/"
                        )
                        fp = snapshot_download(
                            repo_id="ehwang/brain_encoder_weights",
                            allow_patterns=f"checkpoints/nsd_test/dinov2_q_transformer/subj_{self.subj:02}/enc_{layer_num}/run_{r}/{hemi}/*",
                        )
                        output_path = (
                            Path(args.output_path)
                            / f"nsd_test/dinov2_q_transformer/subj_{self.subj:02}/enc_{layer_num}/run_{r}/{hemi}"
                        )
                        output_path.mkdir(exist_ok=True, parents=True)
                        src_path = (
                            Path(fp)
                            / f"checkpoints/nsd_test/dinov2_q_transformer/subj_{self.subj:02}/enc_{layer_num}/run_{r}/{hemi}"
                        )
                        for item in src_path.rglob(
                            "*"
                        ):  # Recursively match all files and subdirectories
                            dest = output_path / item.relative_to(src_path)
                            if item.is_symlink():
                                # Resolve the symlink to the actual file
                                resolved_path = item.resolve(strict=True)
                                dest.parent.mkdir(
                                    parents=True, exist_ok=True
                                )  # Ensure the parent directory exists
                                shutil.move(str(resolved_path), str(dest))
                            elif item.is_dir():
                                dest.mkdir(parents=True, exist_ok=True)
                            else:
                                dest.parent.mkdir(
                                    parents=True, exist_ok=True
                                )  # Ensure the parent directory exists
                                shutil.move(str(item), str(dest))

                        if self.is_valid_model(model_path, hemi):
                            self.model_paths[hemi].append(model_path)

            assert self.model_paths[hemi], f"No valid models found for {hemi}"

            print(f"Found {len(self.model_paths[hemi])} valid models for {hemi}")

        ## TODO what is the best way to load multiple models?
        val_correlation = {
            "lh": np.zeros((len(self.model_paths[hemi]), self.num_voxels)),
            "rh": np.zeros((len(self.model_paths[hemi]), self.num_voxels)),
        }
        self.corr_sm = copy.deepcopy(val_correlation)
        for hemi in ["lh", "rh"]:
            for idx, model_path in enumerate(self.model_paths[hemi]):
                region_val_corr = np.load(model_path / f"{hemi}_val_corr_nonavg.npy")
                region_val_corr = np.nan_to_num(region_val_corr)
                val_correlation[hemi][idx] += region_val_corr

            self.corr_sm[hemi] = torch.from_numpy(
                softmax(20 * val_correlation[hemi], axis=0)
            )

        print(
            "max lh corr",
            val_correlation["lh"].mean(axis=1).max(),
            "min",
            val_correlation["lh"].mean(axis=1).min(),
        )
        print(
            "max rh corr",
            val_correlation["rh"].mean(axis=1).max(),
            "min",
            val_correlation["rh"].mean(axis=1).min(),
        )

        self.preload_models(num_gpus)

    def is_valid_model(self, model_path, hemi):
        paths = [
            model_path,
            model_path / "checkpoint_nonavg.pth",
            model_path / f"{hemi}_val_corr_nonavg.npy",
        ]
        for p in paths:
            if not p.exists():
                return False

        return True

    def load_model_path(
        self,
        model_path,
        images,
        device="cpu",
    ):
        checkpoint = torch.load(
            model_path / "checkpoint_nonavg.pth", map_location="cpu", weights_only=True
        )

        pretrained_dict = checkpoint["model"]
        args = checkpoint["args"]

        dataset = nsd_dataset_avg(args)

        args.device = device
        model = brain_encoder(args, dataset)

        if len([k for k in pretrained_dict.keys() if ".orig_mod" in k]) > 0:
            raise ValueError(
                "Model has nonmatching keys with .orig_mod, should manually inspect"
            )

        pretrained_dict = {
            key.replace("_orig_mod.", ""): value
            for key, value in pretrained_dict.items()
        }
        # pretrained_dict = {
        #     k: v
        #     for k, v in pretrained_dict.items()
        # if "backbone_model" not in k or "input_proj" in k
        # }
        # print("input proj keys:", pretrained_dict.keys())
        model_dict = model.state_dict()
        # print("original model dict", model_dict.keys())
        # print("pretrained dict", pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

        model = model.to(device, non_blocking=True)
        model.eval()

        return model, args, dataset

    def extract_transformer_features(self, model, imgs, enc_layers=0, dec_layers=1):
        model_features = {}

        outputs, enc_output, enc_attn_weights, dec_output, dec_attn_weights = (
            get_transformer_activations(model, imgs, enc_layers, dec_layers)
        )

        return outputs, enc_output, enc_attn_weights, dec_output, dec_attn_weights

    def attention(self, images):
        model_features = {}
        # Will hold results for both hemispheres.
        dec_attn_weights_all = {}
        outputs = {}

        # Define a helper function to process one hemisphere.
        def process_hemi(hemi):
            models = self.models[hemi]
            hemi_preds = torch.zeros(
                len(images), len(self.model_paths[hemi]), self.num_voxels
            )
            dec_attn_weights_list = []
            for idx, model in enumerate(
                tqdm(
                    models,
                    desc=f"Running inference on {hemi} models",
                    leave=False,
                )
            ):
                # Use a local copy so as not to overwrite the shared "images" tensor.
                images_to_device = images.to(model.device, non_blocking=True)
                pred, _, _, _, dec_attn_weights = self.extract_transformer_features(
                    model, images_to_device
                )
                # Append the first decoder attention weights from the output.
                dec_attn_weights_list.append(dec_attn_weights[0].detach().cpu().numpy())

                pred = torch.nan_to_num(pred["pred"]).cpu()
                hemi_preds[:, idx, :] += pred

            # Stack the attention weights and compute normalized prediction.
            dec_attn_weights_all_hemi = np.stack(dec_attn_weights_list, axis=0)
            normalized_pred = (self.corr_sm[hemi].cpu() * hemi_preds.cpu()).sum(1)
            outputs_hemi = normalized_pred.detach().numpy()
            return dec_attn_weights_all_hemi, outputs_hemi

        # Run the two hemispheres in parallel (each on its own GPU).

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(process_hemi, hemi): hemi for hemi in ["lh", "rh"]
            }
            for future in futures:
                hemi = futures[future]
                dec_attn_weights_all[hemi], outputs[hemi] = future.result()

        model_features["dec_attn_weights"] = dec_attn_weights_all
        model_features["pred"] = outputs

        return model_features

    def forward_hemi(self, hemi, images, use_dataloader):
        hemi_preds = torch.zeros(
            len(images), len(self.model_paths[hemi]), self.num_voxels
        )

        if use_dataloader:
            models = self.models[hemi]
            for idx, model in enumerate(
                tqdm(
                    models,
                    desc=f"Running inference on {hemi} models",
                    leave=False,
                )
            ):
                dataset = nsd_dataset_custom(images, transform=self.transform)
                imgs_loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=32,
                    num_workers=4,
                    pin_memory=True,
                    shuffle=False,
                )
                preds = []
                for imgs, _ in tqdm(
                    imgs_loader, desc="running forward pass", leave=False
                ):
                    # print("imgs", imgs.shape)
                    pred = self.forward_batch(model, imgs)
                    # print("pred", pred.shape)
                    pred = torch.nan_to_num(pred).cpu()
                    preds.append(pred)
                preds = torch.cat(preds, dim=0)
                hemi_preds[:, idx, :] += preds
        else:
            models = self.models[hemi]
            for idx, model in enumerate(models):
                preds = self.forward_batch(model, images)
                preds = torch.nan_to_num(preds).cpu()
                hemi_preds[:, idx, :] += preds

        normalized_pred = (self.corr_sm[hemi].cpu() * hemi_preds.cpu()).sum(1)

        return normalized_pred

    def compile_models(self):
        for hemi in ["lh", "rh"]:
            for idx, model in enumerate(self.models[hemi]):
                self.models[hemi][idx] = torch.compile(model)

    def forward(self, images, use_dataloader=True):
        # You can remove these placeholders if they are not needed.
        pred = {
            "lh": np.zeros((len(images), self.num_voxels)),
            "rh": np.zeros((len(images), self.num_voxels)),
        }

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                hemi: executor.submit(self.forward_hemi, hemi, images, use_dataloader)
                for hemi in ["lh", "rh"]
            }
            for hemi, future in futures.items():
                pred[hemi] = future.result()

        return pred

    def preload_models(self, num_gpus):
        self.models = {}
        self.parcel_dir = None

        def load_hemi_models(hemi):
            models = []
            for model_path in self.model_paths[hemi]:
                if num_gpus == 2:
                    device = "cuda:0" if hemi == "lh" else "cuda:1"
                elif num_gpus == 4:
                    for part in model_path.parts:
                        if part.startswith("enc_"):
                            layer_int = int(part.split("_", 1)[1])
                            break
                    device = f"cuda:{int((layer_int - 1) / 2)}"
                else:
                    device = "cuda:0"  # default device if not using 2 GPUs
                model, args, _ = self.load_model_path(
                    model_path,
                    torch.zeros(1, 3, 224, 224),
                    device,
                )
                models.append(model)

                if self.parcel_dir is not None and self.parcel_dir != args.parcel_dir:
                    raise ValueError("Model paths have different parcel directories")
                self.parcel_dir = args.parcel_dir
            return models

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                hemi: executor.submit(load_hemi_models, hemi) for hemi in ["lh", "rh"]
            }
            for hemi, future in futures.items():
                self.models[hemi] = future.result()

        self.parcel_dir = Path(self.parcel_dir)

        return self.models

    def forward_batch(self, model, images):
        if self.lr_backbone is not None:
            model.lr_backbone = self.lr_backbone
            for name, param in model.named_parameters():
                param.requires_grad = False

        imgs = images.to(next(model.parameters()).device, non_blocking=True)
        # print("imgs", imgs.shape)
        outputs = model(imgs)
        outputs = outputs["pred"]

        return outputs

    def load_roi_labels(self):
        metadata = np.load(
            self.neural_data_path / f"metadata_sub-{self.subj:02}.npy",
            allow_pickle=True,
        ).item()

        return {"lh": metadata["lh_rois"], "rh": metadata["rh_rois"]}

    def load_parcels(self):
        parcels = {}
        for hemi in ["lh", "rh"]:
            parcels[hemi] = torch.load(
                self.parcel_dir / f"{hemi}_labels_s{self.subj:02}.pt", weights_only=True
            )

        return parcels


def forward(model, imgs_data, input_args):
    dataset = nsd_dataset_custom(
        imgs_data, transform=model.transform, backbone_arch=model.backbone_arch
    )
    if model.backbone_arch == "dinov2_q":
        batch_size = 26
    else:
        batch_size = 4
    imgs_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    out = {
        "lh": np.zeros((len(dataset), 163842)),
        "rh": np.zeros((len(dataset), 163842)),
    }
    dec_attn_weights = {
        "lh": np.zeros((len(dataset), model.models["lh"][0].num_queries, 31, 31)),
        "rh": np.zeros((len(dataset), model.models["rh"][0].num_queries, 31, 31)),
    }
    for idx, (imgs, _) in tqdm(
        enumerate(imgs_loader),
        desc="running forward pass",
        leave=False,
        total=len(imgs_loader),
    ):
        with torch.no_grad():
            if input_args.save_attention:
                model_features = model.attention(imgs)
                pred = model_features["pred"]
            else:
                pred = model.forward(imgs, use_dataloader=False)

        for hemi in ["lh", "rh"]:
            if torch.is_tensor(pred[hemi]):
                pred[hemi] = pred[hemi].detach().cpu().numpy()
            out[hemi][idx * batch_size : idx * batch_size + len(pred[hemi])] = pred[
                hemi
            ]

            if input_args.save_attention:
                dec_attn_weights_batch = model_features["dec_attn_weights"][hemi]
                hemi_dec_attn_weights = dec_attn_weights_batch.mean(0)
                h, w = 31, 31
                new_shape = hemi_dec_attn_weights.shape[:-1] + (h, w)
                hemi_dec_attn_weights = hemi_dec_attn_weights.reshape(new_shape)
                dec_attn_weights[hemi][
                    idx * batch_size : idx * batch_size + len(imgs)
                ] = hemi_dec_attn_weights

        torch.cuda.empty_cache()
    res = {}
    activations = np.concatenate([out["lh"], out["rh"]], axis=1)
    corr_sims, mse_sims = calc_sims(model.subj, activations)
    res["corr_sims"] = corr_sims
    res["mse_sims"] = mse_sims

    # Compute parcel mean activity
    parcels = model.load_parcels()
    roi_labels = model.load_roi_labels()
    parcel_mean_activity = {}
    for hemi in ["lh", "rh"] if input_args.hemi is None else [input_args.hemi]:
        parcel_mean_activity[hemi] = np.zeros((len(imgs_data), len(parcels[hemi])))
        for idx, parcel in enumerate(parcels[hemi]):
            parcel_mask = np.zeros(163842, dtype=bool)
            parcel_mask[parcel] = True
            parcel_mean_activity[hemi][:, idx] = out[hemi][:, parcel_mask].mean(axis=1)
        la = np.zeros(163842, dtype=bool)
        for roi in roi_labels[hemi]:
            la = np.logical_or(la, roi_labels[hemi][roi])

        if input_args.split in ["train", "val", "test"] or input_args.save == "all":
            continue
        elif input_args.save == "parcel":
            mask = fetch.parcel(
                input_args.subj, hemi, input_args.parcel_dir, input_args.parcel_strategy
            )["parcel"]
            out[hemi] = out[hemi][:, mask]
        elif input_args.save == "nsd_labeled":
            out[hemi] = out[hemi][:, la]

    res["path"] = input_args.target_dir
    if input_args.save != "none":
        res["out"] = out

    # Save attention weights
    if input_args.save_attention:
        if input_args.parcel_dir is not None:
            res["attention"] = {}
            if input_args.parcel_dir.isdigit():
                p = int(input_args.parcel_dir)
            else:
                parcel_map = fetch.overlap_labeled_parcels(
                    input_args.subj,
                    input_args.hemi,
                    return_parcel_num=True,
                    parcel_strategy=input_args.parcel_strategy,
                )
                p = parcel_map[input_args.parcel_dir]
            res["attention"] = dec_attn_weights[input_args.hemi][:, p]
        else:
            res["labeled_attention"] = {}
            res["candidate_attention"] = {}
            candidate_parcels = fetch.get_parcel_list(
                input_args.subj, input_args.parcel_strategy
            )
            for hemi in ["lh", "rh"]:
                # zoom_factors = (1, 1, 8 / 31, 8 / 31)
                data = dec_attn_weights[hemi]
                # data = ndimage.zoom(data, zoom_factors, order=1)
                # mins = data.min(axis=(-2, -1), keepdims=True)
                # maxs = data.max(axis=(-2, -1), keepdims=True)

                # # normalize to 0-255
                # data = (data - mins) / (maxs - mins) * 255
                # data = np.rint(data).astype(np.uint8)

                dec_attn_weights[hemi] = data
                labeled_parcels = fetch.overlap_labeled_parcels(
                    input_args.subj,
                    hemi,
                    return_parcel_num=True,
                    parcel_strategy=input_args.parcel_strategy,
                )
                res["labeled_attention"][hemi] = {
                    k: dec_attn_weights[hemi][:, labeled_parcels[k]]
                    for k in labeled_parcels.keys()
                }
                res["candidate_attention"][hemi] = {
                    k: dec_attn_weights[hemi][:, k] for k in candidate_parcels[hemi]
                }
    res["parcel_mean_activity"] = parcel_mean_activity
    res["parcels"] = parcels

    return res


def pairwise_mse(x, y):
    # x: tensor of shape (N, D)
    # y: tensor of shape (M, D)
    N, D = x.shape
    M, _ = y.shape

    # Compute squared norms for each row.
    # Shape of x_norm_sq: (N, 1), shape of y_norm_sq: (M, 1)
    x_norm_sq = (x**2).sum(dim=1, keepdim=True)  # (N, 1)
    y_norm_sq = (y**2).sum(dim=1, keepdim=True)  # (M, 1)

    # Compute the inner product between all pairs using matrix multiplication.
    # This results in an (N, M) matrix.
    dot_product = x @ y.t()  # (N, M)

    # To apply the formula, we need to broadcast:
    # - x_norm_sq over axis 1 (making it (N, M))
    # - y_norm_sq over axis 0 (make it (N, M) by transposing or unsqueezing)
    # Then subtract 2 * dot_product and divide by D.
    mse_matrix = (x_norm_sq + y_norm_sq.t() - 2 * dot_product) / D

    return mse_matrix


def calc_sims(subj, activations, device="cpu"):
    activations = torch.from_numpy(activations).to(device)
    test_betas = []
    ncs = []
    for hemi in ["lh", "rh"]:
        nsd_data = fetch.nsd_data(subj, hemi, split=["test", "val"])
        nsd_data = torch.from_numpy(nsd_data["betas"])

        nc = torch.from_numpy(fetch.metadata(subj)[f"{hemi}_ncsnr"].squeeze())
        ncs.append(nc)
        test_betas.append(nsd_data)
    betas_torch = torch.cat(test_betas, dim=1)
    # normed_betas = F.normalize(betas_torch, p=2, dim=1)
    betas_centered = betas_torch - betas_torch.mean(dim=1, keepdim=True)
    betas_centered = betas_centered.to(device)

    # activations = torch.from_numpy(activations)

    # # Cosine similarity
    # normed_acts = F.normalize(activations, p=2, dim=1)  # (N_class, 70949)
    # cos_sims = torch.matmul(normed_betas, normed_acts.T)  # (515, N_class)

    # Pearson correlation: center the activations
    acts_centered = activations - activations.mean(dim=1, keepdim=True)
    corr_sims = torch.matmul(betas_centered, acts_centered.T)
    corr_sims /= (
        betas_centered.norm(dim=1, keepdim=True)
        * acts_centered.norm(dim=1).unsqueeze(0)
        + 1e-6  # numerical stability
    )

    ncs = torch.cat(ncs, dim=0).to(device)
    betas_torch *= ncs
    activations *= ncs

    # MSE loss
    mse_sims = pairwise_mse(betas_torch, activations)

    # cos_sims = cos_sims.to(torch.float32)  # shape (515, 1000)
    corr_sims = corr_sims.to(torch.float32).cpu().numpy()  # shape (515, 1000)
    mse_sims = mse_sims.to(torch.float32).cpu().numpy()  # shape (515, 1000)

    return corr_sims, mse_sims


def main():
    torch.serialization.add_safe_globals([argparse.Namespace, PosixPath])

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--subj", type=int, default=None)
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--split_subj", type=int, default=None)
    argparser.add_argument("--target_dir", type=str, default=None)
    argparser.add_argument("--save_path", type=str, default=None)
    argparser.add_argument("--exist_skip", type=bool, default=False)
    argparser.add_argument("--save_attention", type=bool, default=False)
    argparser.add_argument("--parcel_dir", type=str, default=None)
    argparser.add_argument("--hemi", type=str, default=None)
    argparser.add_argument("--parcel_strategy", type=str, default="schaefer")
    argparser.add_argument("--backbone_arch", type=str, default="dinov2_q")
    argparser.add_argument(
        "--save",
        type=str,
        choices=["none", "all", "parcel", "nsd_labeled"],
    )
    argparser.add_argument("--runs", nargs="+", type=int, default=[1, 2])
    input_args = argparser.parse_args()

    print("save attention", input_args.save_attention)

    assert input_args.subj is not None, "Please specify a subject number"

    if input_args.backbone_arch == "dinov2_q_large":
        num_gpus = 4
    else:
        num_gpus = 2
    model = BrainEncoderWrapper(
        subj=input_args.subj,
        enc_output_layer=[1, 3, 5, 7],
        runs=input_args.runs,
        num_gpus=num_gpus,
        parcel_strategy=input_args.parcel_strategy,
        backbone_arch=input_args.backbone_arch,
    )

    assert model.transform is not None, "No transform found for this model"

    if input_args.split == "folder":
        assert input_args.target_dir is not None
        print(f"Running inference on images in {input_args.target_dir}")

        if input_args.backbone_arch != "dinov2_q":
            fn = f"{input_args.backbone_arch}_activations.npy"
        else:
            fn = "activations.npy"
        args = get_default_args()
        target_dir = Path(input_args.target_dir)

        if (target_dir / fn).exists() and input_args.exist_skip:
            print(f"Activations already exist for {target_dir}, skipping")
            return

        imgs_data = []
        img_paths = []
        for img_file in sorted(target_dir.glob("*")):
            if img_file.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                continue

            image = Image.open(img_file).convert("RGB")
            imgs_data.append(image)
            img_paths.append(img_file)

        print(f"Found {len(imgs_data)} images in {target_dir}")
        if not imgs_data:
            print(f"No images found in {target_dir}")
            return

        res = forward(model, imgs_data, input_args)
        res["img_paths"] = img_paths

        if input_args.save_path is not None and input_args.save_path:
            save_path = Path(input_args.save_path)
        else:
            save_path = target_dir / fn
        save_path.parent.mkdir(exist_ok=True, parents=True)

        np.save(save_path, res)
        print(f"Saved activations to {save_path}")

    if input_args.split in ["train", "val", "test"]:
        split = input_args.split

        save_dir = fetch.ensemble_model_dir(
            input_args.subj,
            input_args.parcel_strategy,
            enc_output_layers=[1, 3, 5, 7],
            runs=[1, 2],
        )

        print(f"Saving results to {save_dir}")

        args = get_default_args()
        if input_args.split_subj is not None:
            args.subj = input_args.split_subj
        else:
            args.subj = input_args.subj
        imgs = {}
        betas = {}
        for hemi in ["lh", "rh"]:
            args.hemi = hemi
            val_dataset = nsd_dataset_avg(args, transform=None, split=input_args.split)
            val_dataset.backbone_arch = False

            imgs[hemi] = []
            betas[hemi] = []
            for img, beta in val_dataset:
                imgs[hemi].append(img)
                betas[hemi].append(beta["betas"])
            imgs[hemi] = np.stack(imgs[hemi])
            betas[hemi] = np.stack(betas[hemi])
        imgs = imgs["lh"]

        res = forward(model, imgs, input_args)
        out = res["out"]

        save_dir.mkdir(exist_ok=True, parents=True)

        if input_args.backbone_arch != "dinov2_q":
            fn = f"{input_args.backbone_arch}_{split}.npy"
        else:
            fn = f"{split}_splitsubj{input_args.split_subj}.npy"
        # np.save(save_dir / fn, res)
        out_file = (save_dir / fn).with_suffix(".h5")
        with h5py.File(out_file, "w") as hf:
            for key, arr in out.items():
                # you can adjust compression or chunking per‐dataset here if you like
                hf.create_dataset(
                    key,
                    data=arr,
                    compression="gzip",  # or "lzf"
                    chunks=True,  # auto-choose a good chunk shape
                )

        val_correlation = {}
        for hemi in ["lh", "rh"]:
            ys = torch.from_numpy(betas[hemi])

            num_valid_voxels = ys.shape[1]
            val_correlation[hemi] = torch.zeros(num_valid_voxels)
            for v in range(num_valid_voxels):
                val_correlation[hemi][v] = corr(ys[:, v], out[hemi][:, v])[0]

            val_correlation[hemi] = val_correlation[hemi].numpy()

            print(
                f"Validation correlation for {hemi} hemi: {val_correlation[hemi].mean()}"
            )

            with (save_dir / f"{split}_corr_avg.txt").open("a") as f:
                f.write(
                    f"Validation correlation for {split} split: {val_correlation[hemi].mean()}\n"
                )
        if input_args.backbone_arch != "dinov2_q":
            fn = f"{input_args.backbone_arch}_{split}_corr_avg.npy"
        else:
            fn = f"{split}_splitsubj{input_args.split_subj}_corr_avg.npy"
        np.save(save_dir / fn, val_correlation)


if __name__ == "__main__":
    main()
