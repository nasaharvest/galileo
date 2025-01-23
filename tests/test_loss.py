import unittest

import torch

from src.data.dataset import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    TIME_BAND_GROUPS_IDX,
)
from src.loss import mae_loss


class TestLoss(unittest.TestCase):
    def test_mae_loss(self):
        b, t_h, t_w, t, patch_size = 16, 4, 4, 3, 2
        pixel_h, pixel_w = t_h * patch_size, t_w * patch_size
        max_patch_size = 8
        max_group_length = max(
            [
                max([len(v) for _, v in SPACE_TIME_BANDS_GROUPS_IDX.items()]),
                max([len(v) for _, v in TIME_BAND_GROUPS_IDX.items()]),
                max([len(v) for _, v in SPACE_BAND_GROUPS_IDX.items()]),
                max([len(v) for _, v in STATIC_BAND_GROUPS_IDX.items()]),
            ]
        )
        p_s_t = torch.randn(
            (
                b,
                t_h,
                t_w,
                t,
                len(SPACE_TIME_BANDS_GROUPS_IDX),
                max_group_length * (max_patch_size**2),
            )
        )
        p_sp = torch.randn(
            (b, t_h, t_w, len(SPACE_BAND_GROUPS_IDX), max_group_length * (max_patch_size**2))
        )
        p_t = torch.randn(
            (b, t, len(TIME_BAND_GROUPS_IDX), max_group_length * (max_patch_size**2))
        )
        p_st = torch.randn(
            (b, len(STATIC_BAND_GROUPS_IDX), max_group_length * (max_patch_size**2))
        )
        s_t_x = torch.randn(
            b, pixel_h, pixel_w, t, sum([len(x) for _, x in SPACE_TIME_BANDS_GROUPS_IDX.items()])
        )
        sp_x = torch.randn(
            b, pixel_h, pixel_w, sum([len(x) for _, x in SPACE_BAND_GROUPS_IDX.items()])
        )
        t_x = torch.randn(b, t, sum([len(x) for _, x in TIME_BAND_GROUPS_IDX.items()]))
        st_x = torch.randn(b, sum([len(x) for _, x in STATIC_BAND_GROUPS_IDX.items()]))
        s_t_m = torch.ones((b, pixel_h, pixel_w, t, len(SPACE_TIME_BANDS_GROUPS_IDX))) * 2
        sp_m = torch.ones((b, pixel_h, pixel_w, len(SPACE_BAND_GROUPS_IDX))) * 2
        t_m = torch.ones((b, t, len(TIME_BAND_GROUPS_IDX))) * 2
        st_m = torch.ones((b, len(STATIC_BAND_GROUPS_IDX))) * 2
        max_patch_size = 8

        loss = mae_loss(
            p_s_t,
            p_sp,
            p_t,
            p_st,
            s_t_x,
            sp_x,
            t_x,
            st_x,
            s_t_m,
            sp_m,
            t_m,
            st_m,
            patch_size,
            max_patch_size,
        )
        self.assertFalse(torch.isnan(loss))
