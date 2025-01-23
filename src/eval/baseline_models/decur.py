from pathlib import Path
from typing import List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DeCurWrapper(nn.Module):
    def __init__(
        self, weights_path: Path, modality: str, do_pool=True, temporal_pooling: str = "mean"
    ):
        super().__init__()
        assert modality in ["SAR", "optical"]

        self.encoder = timm.create_model("vit_small_patch16_224", pretrained=False)
        self.dim = 384
        self.modality = modality
        if modality == "optical":
            self.encoder.patch_embed.proj = torch.nn.Conv2d(
                13, 384, kernel_size=(16, 16), stride=(16, 16)
            )
            state_dict = torch.load(weights_path / "vits16_ssl4eo-s12_ms_decur_ep100.pth")
            msg = self.encoder.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"head.weight", "head.bias"}
        else:
            self.encoder.patch_embed.proj = torch.nn.Conv2d(
                2, 384, kernel_size=(16, 16), stride=(16, 16)
            )
            state_dict = torch.load(weights_path / "vits16_ssl4eo-s12_sar_decur_ep100.pth")
            msg = self.encoder.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"head.weight", "head.bias"}

        self.image_resolution = 224
        self.patch_size = 16
        self.grid_size = int(self.image_resolution / self.patch_size)
        self.do_pool = do_pool
        if temporal_pooling not in ["mean", "max"]:
            raise ValueError(
                f"Expected temporal_pooling to be in ['mean', 'max'], got {temporal_pooling}"
            )
        self.temporal_pooling = temporal_pooling

    def resize(self, images):
        images = F.interpolate(
            images,
            size=(self.image_resolution, self.image_resolution),
            mode="bilinear",
            align_corners=False,
        )
        return images

    def preproccess(self, images):
        images = rearrange(images, "b h w c -> b c h w")
        assert (images.shape[1] == 13) or (images.shape[1] == 2)
        return self.resize(images)  # (bsz, C, H, W)

    def forward(self, s2=None, s1=None, months=None):
        if s1 is not None:
            assert self.modality == "SAR"
            if len(s1.shape) == 5:
                outputs_l: List[torch.Tensor] = []
                for timestep in range(s1.shape[3]):
                    image = self.preproccess(s1[:, :, :, timestep])
                    output = self.encoder.forward_features(image)
                    if self.do_pool:
                        output = output.mean(dim=1)
                    else:
                        output = output[:, 1:]
                    outputs_l.append(output)
                outputs_t = torch.stack(outputs_l, dim=-1)  # b h w d t
                if self.temporal_pooling == "mean":
                    return outputs_t.mean(dim=-1)
                else:
                    return torch.amax(outputs_t, dim=-1)
            else:
                s1 = self.preproccess(s1)
                output = self.encoder.forward_features(s1)
                if self.do_pool:
                    return output.mean(dim=1)
                else:
                    return output[:, 1:]
        elif s2 is not None:
            assert self.modality == "optical"
            if len(s2.shape) == 5:
                outputs_l: List[torch.Tensor] = []
                for timestep in range(s2.shape[3]):
                    image = self.preproccess(s2[:, :, :, timestep])
                    output = self.encoder.forward_features(image)
                    if self.do_pool:
                        output = output.mean(dim=1)
                    else:
                        output = output[:, 1:]
                    outputs_l.append(output)
                outputs_t = torch.stack(outputs_l, dim=-1)  # b h w d t
                if self.temporal_pooling == "mean":
                    return outputs_t.mean(dim=-1)
                else:
                    return torch.amax(outputs_t, dim=-1)
            else:
                s2 = self.preproccess(s2)
                output = self.encoder.forward_features(s2)
                if self.do_pool:
                    return output.mean(dim=1)
                else:
                    return output[:, 1:]
