from datetime import datetime
from typing import Dict, Optional

import torch
from einops import rearrange, repeat
from torch import nn

from .hubconf import AnySat


class AnySatWrapper(nn.Module):
    # we assume any data passed to this wrapper
    # will contain S2 data with the following channels
    INPUT_S2_BAND_ORDERING = [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B10",
        "B11",
        "B12",
    ]
    INPUT_S1_BAND_ORDERING = [
        "VV",
        "VH",
    ]

    # these are the bands which AnySat accepts
    # https://github.com/gastruc/AnySat?tab=readme-ov-file#format-your-data
    ANYSAT_S2_BAND_ORDERING = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    ANYSAT_S1_BAND_ORDERING = ["VV", "VH", "ratio"]

    def __init__(self, do_pool=True, temporal_pooling: str = "mean"):
        super().__init__()
        self.model = AnySat.from_pretrained("base", flash_attn=False)

        self.grid_size: Optional[int] = None

        self.kept_s2_band_idx = [
            self.INPUT_S2_BAND_ORDERING.index(v) for v in self.ANYSAT_S2_BAND_ORDERING
        ]
        self.kept_s1_band_idx = [
            self.INPUT_S1_BAND_ORDERING.index(v)
            for v in self.ANYSAT_S1_BAND_ORDERING
            if v in self.INPUT_S1_BAND_ORDERING
        ]
        self.month = 5  # default month, if none is given (indexing from 0)
        self.do_pool = do_pool
        if do_pool:
            self.output = "tile"  # single vector per tile
            self.dim = 768
        else:
            self.output = "dense"
            self.dim = 768 * 2  # this is the case for dense for both s1 and s2
            self.patch_size = 1  # a token is output for every pixel

    @staticmethod
    def months_to_day_of_year(months: torch.Tensor):
        output_tensors = []
        for i in range(months.shape[0]):
            output_tensors.append(
                torch.tensor([datetime(2025, m + 1, 1).timetuple().tm_yday for m in months[i]]).to(
                    device=months.device
                )
            )
        return torch.stack(output_tensors)

    def calculate_patch_size_and_update_grid_size(self, h):
        # based on https://arxiv.org/pdf/2412.14123, a patch size of
        # 40 is the minimum used for images of 128x128. Since smaller patches
        # = more tokens, this should lead to the best performance
        h_in_m = h * 10
        patch_size = min(40, h_in_m)
        self.grid_size = h_in_m  # with dense, a token is outputted for every pixel
        return patch_size

    def forward(self, s2=None, s1=None, months=None):
        input_dictionary: Dict = {}
        if s2 is not None:
            output_modality = "s2"
            patch_size = self.calculate_patch_size_and_update_grid_size(s2.shape[1])
            if len(s2.shape) == 4:
                s2 = repeat(s2, "b h w d -> b t d h w", t=1)[:, :, self.kept_s2_band_idx, :, :]
                if months is None:
                    months = repeat(
                        torch.tensor([self.month]).to(device=s2.device), "d -> b d", b=s2.shape[0]
                    )
                s2_doy = self.months_to_day_of_year(months)
            else:
                assert months is not None
                s2 = rearrange(s2, "b h w t d -> b t d h w")[:, :, self.kept_s2_band_idx, :, :]
                s2_doy = self.months_to_day_of_year(months)
            # months should always be passed unless
            input_dictionary.update({"s2": s2, "s2_dates": s2_doy})
        if s1 is not None:
            output_modality = "s1"
            patch_size = self.calculate_patch_size_and_update_grid_size(s1.shape[1])
            if len(s1.shape) == 4:
                s1 = repeat(s1, "b h w d -> b t d h w", t=1)[:, :, self.kept_s1_band_idx, :, :]
                if months is None:
                    months = repeat(
                        torch.tensor([self.month]).to(device=s1.device), "d -> b d", b=s1.shape[0]
                    )
                s1_doy = self.months_to_day_of_year(months)
            else:
                assert months is not None
                s1 = rearrange(s1, "b h w t d -> b t d h w")[:, :, self.kept_s1_band_idx, :, :]
                s1_doy = self.months_to_day_of_year(months)
            # add a ratio band
            ratio_band = s1[:, :, :1, :, :] / (s1[:, :, 1:, :, :] + 1e-6)
            s1 = torch.concat((s1, ratio_band), dim=2)
            input_dictionary.update({"s1": s1, "s1_dates": s1_doy})

        output_patches = self.model(
            x=input_dictionary,
            patch_size=patch_size,
            output=self.output,
            output_modality=output_modality,
        )
        if self.output == "dense":
            output_patches = rearrange(output_patches, "b h w d -> b (h w) d")
        return output_patches
