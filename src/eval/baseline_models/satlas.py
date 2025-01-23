from pathlib import Path
from typing import List

import satlaspretrain_models
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from satlaspretrain_models.utils import Backbone


class SatlasWrapper(nn.Module):
    def __init__(
        self, weights_path: Path, size="base", do_pool=True, temporal_pooling: str = "mean"
    ):
        super().__init__()
        if size == "base":
            self.dim = 1024
            weights = torch.load(
                weights_path / "satlas-model-v1-lowres-band.pth", map_location="cpu"
            )
            self.satlas = satlaspretrain_models.Model(
                num_channels=9,
                multi_image=False,
                backbone=Backbone.SWINB,
                fpn=False,
                head=None,
                num_categories=None,
                weights=weights,
            )
        elif size == "tiny":
            self.dim = 768
            weights = torch.load(weights_path / "sentinel2_swint_si_ms.pth", map_location="cpu")
            self.satlas = satlaspretrain_models.Model(
                num_channels=9,
                multi_image=False,
                backbone=Backbone.SWINT,
                fpn=False,
                head=None,
                num_categories=None,
                weights=weights,
            )
        else:
            raise ValueError(f"size must be base or tiny, not {size}")

        self.image_resolution = 512
        self.grid_size = 16  # Swin spatially pools
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
        assert images.shape[1] == 13
        # From: https://github.com/allenai/satlas/blob/main/Normalization.md
        images = images[:, (1, 2, 3, 4, 5, 6, 0, 7, 8), :, :]
        return self.resize(images)  # (bsz, 12, 120, 120)

    def forward(self, s2=None, s1=None, months=None):
        if s2 is None:
            raise ValueError("S2 can't be None for Satlas")

        # not using the FPN
        # we should get output shapes, for base:
        # [[bsz, 128, 128, 128], [bsz, 256, 64, 64], [bsz, 512, 32, 32], [bsz, 1024, 16, 16]]
        # and for tiny:
        # [[bsz, 96, 128, 128], [bsz, 192, 64, 64], [bsz, 384, 32, 32], [bsz, 768, 16, 16]]

        if len(s2.shape) == 5:
            outputs_l: List[torch.Tensor] = []
            for timestep in range(s2.shape[3]):
                image = self.preproccess(s2[:, :, :, timestep])
                output = self.satlas(image)
                # output shape for atto: (bsz, 320, 7, 7)
                # output shape for tiny: (bsz, 768, 6, 6)
                if self.do_pool:
                    output = output[-1].mean(dim=-1).mean(dim=-1)
                else:
                    output = rearrange(output[-1], "b c h w -> b (h w) c")
                outputs_l.append(output)
            outputs_t = torch.stack(outputs_l, dim=-1)  # b h w d t
            if self.temporal_pooling == "mean":
                return outputs_t.mean(dim=-1)
            else:
                return torch.amax(outputs_t, dim=-1)
        else:
            s2 = self.preproccess(s2)
            output = self.satlas(s2)
            if self.do_pool:
                return output[-1].mean(dim=-1).mean(dim=-1)  # (bsz, dim)
            else:
                return rearrange(output[-1], "b c h w -> b (h w) c")  # (bsz, seq_len, dim)
