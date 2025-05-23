from utils.utils import NestedTensor
import torch
from torch import nn
from typing import Dict
import torch.nn.functional as F

class eradio_model(nn.Module):
    def __init__(self, enc_output_layer):
        super().__init__()
        self.enc_output_layer = enc_output_layer

        model_version = "radio_v2.5-h"  # for RADIOv2.5-B model (ViT-B/16)
        self.model = torch.hub.load(
            "NVlabs/RADIO",
            "radio_model",
            version=model_version,
            progress=True,
            skip_validation=True,
        )
        self.model.cuda().eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.num_channels = 768

        self.input_proj = nn.Conv2d(
            1280, self.num_channels, kernel_size=1
        ).to("cuda")

    def forward(self, tensor_list: NestedTensor):
        xs = tensor_list.tensors
        h, w = int(xs.shape[2] / 16), int(xs.shape[3] / 16)

        xs = self.model.forward(xs).features



        xs = torch.reshape(xs, (xs.shape[0], h, w, 1280)).permute(
            0, 3, 1, 2
        )

        xs = self.input_proj(xs)

        m = tensor_list.mask
        assert m is not None
        mask = F.interpolate(m[None].float(), size=xs.shape[-2:]).to(torch.bool)[0]
        xs = NestedTensor(xs, mask)

        return {"layer_top": xs}

class radio_model_h(nn.Module):
    def __init__(self, enc_output_layer):
        super().__init__()
        self.enc_output_layer = enc_output_layer

        model_version = "e-radio_v2"  # for RADIOv2.5-B model (ViT-B/16)
        self.model = torch.hub.load(
            "NVlabs/RADIO",
            "radio_model",
            version=model_version,
            progress=True,
            skip_validation=True,
        )
        self.model.cuda().eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.num_channels = 768

        self.input_proj = nn.Conv2d(
            self.model.embed_dim, self.num_channels, kernel_size=1
        ).to("cuda")

    def forward(self, tensor_list: NestedTensor):
        xs = tensor_list.tensors
        h, w = int(xs.shape[2] / 16), int(xs.shape[3] / 16)

        xs = self.model.forward_intermediates(
            xs, indices=self.enc_output_layer, intermediates_only=True
        )[-1 * self.enc_output_layer]
        xs = self.input_proj(xs)
        xs = xs.flatten(-2).permute(0, 2, 1)

        xs = {"layer_top": xs}
        #         xs = self.body(tensor_list.tensors)

        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None

            x = torch.reshape(x, (x.shape[0], h, w, self.num_channels)).permute(
                0, 3, 1, 2
            )

            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)

        return out


class radio_model(nn.Module):
    def __init__(self, enc_output_layer):
        super().__init__()
        self.enc_output_layer = enc_output_layer

        model_version = "radio_v2.5-b"  # for RADIOv2.5-B model (ViT-B/16)
        self.model = torch.hub.load(
            "NVlabs/RADIO",
            "radio_model",
            version=model_version,
            progress=True,
            skip_validation=True,
        )
        self.model.cuda().eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.num_channels = 768

    def forward(self, tensor_list: NestedTensor):
        xs = tensor_list.tensors
        h, w = int(xs.shape[2] / 16), int(xs.shape[3] / 16)

        xs = self.model.forward_intermediates(
            xs, indices=self.enc_output_layer, intermediates_only=True
        )[-1 * self.enc_output_layer]
        xs = xs.flatten(-2).permute(0, 2, 1)

        xs = {"layer_top": xs}
        #         xs = self.body(tensor_list.tensors)

        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None

            x = torch.reshape(x, (x.shape[0], h, w, self.num_channels)).permute(
                0, 3, 1, 2
            )

            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)

        return out
