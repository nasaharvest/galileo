import unittest

import torch

from src.data.utils import (
    S2_BANDS,
    SPACE_TIME_BANDS,
    SPACE_TIME_BANDS_GROUPS_IDX,
    construct_galileo_input,
)


class TestDataUtils(unittest.TestCase):
    def test_construct_galileo_input_s2(self):
        t, h, w = 2, 4, 4
        s2 = torch.randn((t, h, w, len(S2_BANDS)))
        masked_output = construct_galileo_input(s2=s2)

        self.assertTrue((masked_output.space_mask == 1).all())
        self.assertTrue((masked_output.time_mask == 1).all())
        self.assertTrue((masked_output.static_mask == 1).all())

        # check that only the s2 bands got unmasked
        not_s2 = [idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S2" not in key]
        self.assertTrue((masked_output.space_time_mask[:, :, :, not_s2] == 1).all())
        # and that s2 got unmasked
        s2_mask_indices = [
            idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S2" in key
        ]
        self.assertTrue((masked_output.space_time_mask[:, :, :, s2_mask_indices] == 0).all())

        # and got assigned to the right indices
        s2_indices = [idx for idx, val in enumerate(SPACE_TIME_BANDS) if val in S2_BANDS]
        self.assertTrue(torch.equal(masked_output.space_time_x[:, :, :, s2_indices], s2))
