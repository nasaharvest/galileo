import random
import unittest

import torch
from einops import repeat

from src.masking import (
    MASKING_MODES,
    MAX_MASKING_STRATEGIES,
    SPACE_BAND_GROUPS_IDX,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    TIME_BAND_GROUPS_IDX,
    batch_mask_random,
    batch_mask_space,
    batch_mask_time,
    check_modes_for_conflicts,
    filter_unmasking_mode_candidates,
    weighted_sample_without_replacement,
)


class TestMasking(unittest.TestCase):
    def check_all_values_in_masks(
        self, space_time_mask, space_mask, time_mask, static_mask, masking_modes, unmasking_modes
    ):
        self.assertTrue(
            (space_time_mask == 2).any()
            | (space_mask == 2).any()
            | (time_mask == 2).any()
            | (static_mask == 2).any(),
            f"2 check failed for {masking_modes}, {unmasking_modes}",
        )
        self.assertTrue(
            (space_time_mask == 0).any()
            | (space_mask == 0).any()
            | (time_mask == 0).any()
            | (static_mask == 0).any(),
            f"0 check failed for {masking_modes}, {unmasking_modes}",
        )
        self.assertTrue(
            (space_time_mask == 1).any()
            | (space_mask == 1).any()
            | (time_mask == 1).any()
            | (static_mask == 1).any(),
            f"1 check failed for {masking_modes}, {unmasking_modes}",
        )

    def test_mask_by_time(self):
        # testing specific failure modes
        self._test_mask_by_for_f(
            batch_mask_time,
            [
                ("static", "LS"),
                ("static", "location"),
                ("space", "WC"),
                ("space_time", "S2_SWIR"),
                ("space", "DW"),
                ("space_time", "S2_NIR_20m"),
            ],
            [("time", "TC"), ("time", "VIIRS")],
        )

        for _ in range(100):
            num_masking_modes = random.choice(list(range(2, MAX_MASKING_STRATEGIES + 1)))
            num_unmasking_modes = 1

            masking_modes = weighted_sample_without_replacement(
                MASKING_MODES, weights=[1] * len(MASKING_MODES), k=num_masking_modes
            )
            unmasking_modes = weighted_sample_without_replacement(
                MASKING_MODES, weights=[1] * len(MASKING_MODES), k=num_unmasking_modes
            )
            self.assertTrue(
                len(unmasking_modes) == num_unmasking_modes, f"Got {len(unmasking_modes)}"
            )
            masking_modes, unmasking_modes = check_modes_for_conflicts(
                masking_modes, unmasking_modes
            )
            self.assertTrue(
                len(unmasking_modes) == num_unmasking_modes, f"Got {len(unmasking_modes)}"
            )
            self.assertTrue(len(masking_modes) >= 1, f"Got {len(masking_modes)}")
            for m_m in masking_modes:
                self.assertTrue(m_m not in unmasking_modes, f"{m_m} in {unmasking_modes}")
            for u_m in unmasking_modes:
                self.assertTrue(u_m not in masking_modes, f"{u_m} in {masking_modes}")
            self.assertTrue(len(masking_modes) >= 1)
            self.assertTrue(len(unmasking_modes) >= 1)
            self._test_mask_by_for_f(batch_mask_space, masking_modes, unmasking_modes)

    def test_mask_by_space(self):
        for _ in range(100):
            num_masking_modes = random.choice(list(range(2, MAX_MASKING_STRATEGIES + 1)))
            num_unmasking_modes = 1

            masking_modes = weighted_sample_without_replacement(
                MASKING_MODES, weights=[1] * len(MASKING_MODES), k=num_masking_modes
            )
            unmasking_modes = weighted_sample_without_replacement(
                MASKING_MODES, weights=[1] * len(MASKING_MODES), k=num_unmasking_modes
            )
            self.assertTrue(
                len(unmasking_modes) == num_unmasking_modes, f"Got {len(unmasking_modes)}"
            )
            masking_modes, unmasking_modes = check_modes_for_conflicts(
                masking_modes, unmasking_modes
            )
            self.assertTrue(
                len(unmasking_modes) == num_unmasking_modes, f"Got {len(unmasking_modes)}"
            )
            self.assertTrue(len(masking_modes) >= 1, f"Got {len(masking_modes)}")
            for m_m in masking_modes:
                self.assertTrue(m_m not in unmasking_modes, f"{m_m} in {unmasking_modes}")
            for u_m in unmasking_modes:
                self.assertTrue(u_m not in masking_modes, f"{u_m} in {masking_modes}")
            self.assertTrue(len(masking_modes) >= 1)
            self.assertTrue(len(unmasking_modes) >= 1)
            self._test_mask_by_for_f(batch_mask_space, masking_modes, unmasking_modes)

    def _test_mask_by_for_f(self, f, masking_modes, unmasking_modes):
        for t in range(4, 8):
            b, h, w = 2, 16, 16
            space_time_input = torch.ones((b, h, w, t, 8))
            space_input = torch.ones((b, h, w, 8))
            time_input = torch.ones((b, t, 8))
            static_input = torch.ones((b, 8))
            months = repeat(torch.arange(0, t), "t -> b t", b=b)
            ratio = 0.25
            output = f(
                space_time_input,
                space_input,
                time_input,
                static_input,
                months,
                encode_ratio=ratio,
                decode_ratio=ratio,
                mode=masking_modes,
                decoder_mode=unmasking_modes,
                patch_size=4,
            )
            self.check_all_values_in_masks(
                output.space_time_mask,
                output.space_mask,
                output.time_mask,
                output.static_mask,
                masking_modes,
                unmasking_modes,
            )
            self.assertEqual(
                (b, h, w, t, len(SPACE_TIME_BANDS_GROUPS_IDX)),
                output.space_time_mask.shape,
            )
            self.assertEqual((b, h, w, len(SPACE_BAND_GROUPS_IDX)), output.space_mask.shape)
            self.assertEqual((b, t, len(TIME_BAND_GROUPS_IDX)), output.time_mask.shape)
            self.assertEqual((b, len(STATIC_BAND_GROUPS_IDX)), output.static_mask.shape)

    def test_mask_by_random(self):
        b, t, h, w, p = 2, 8, 16, 16, 4
        h_tokens, w_tokens = h / p, w / p
        space_time_input = torch.ones((b, h, w, t, 8))
        space_input = torch.ones((b, h, w, 8))
        time_input = torch.ones((b, t, 8))
        static_input = torch.ones((b, 8))
        months = repeat(torch.arange(0, t), "t -> b t", b=b)
        ratio = 0.25

        output = batch_mask_random(
            space_time_input,
            space_input,
            time_input,
            static_input,
            months,
            encode_ratio=ratio,
            decode_ratio=ratio,
            patch_size=p,
            ignore_band_groups=["DW", "DW_static"],
        )
        self.check_all_values_in_masks(
            output.space_time_mask,
            output.space_mask,
            output.time_mask,
            output.static_mask,
            "random",
            "random",
        )
        self.assertEqual(
            (b, h, w, t, len(SPACE_TIME_BANDS_GROUPS_IDX)), output.space_time_mask.shape
        )
        self.assertEqual((b, h, w, len(SPACE_BAND_GROUPS_IDX)), output.space_mask.shape)
        self.assertEqual((b, t, len(TIME_BAND_GROUPS_IDX)), output.time_mask.shape)
        self.assertEqual((b, len(STATIC_BAND_GROUPS_IDX)), output.static_mask.shape)

        for i in range(1, p):
            self.assertTrue(
                torch.equal(
                    output.space_time_mask[:, i::p, i::p],
                    output.space_time_mask[:, i - 1 :: p, i - 1 :: p],
                )
            )
            self.assertTrue(
                torch.equal(
                    output.space_mask[:, i::p, i::p],
                    output.space_mask[:, i - 1 :: p, i - 1 :: p],
                )
            )
        space_time_per_token = output.space_time_mask[:, i::p, i::p]
        space_time_encode_per_instance = (
            space_time_per_token[space_time_per_token == 0] + 1
        ).sum()
        space_time_decode_per_instance = space_time_per_token[space_time_per_token == 2].sum()
        space_per_token = output.space_mask[:, i::p, i::p]
        space_encode_per_instance = (space_per_token[space_per_token == 0] + 1).sum()
        space_decode_per_instance = space_per_token[space_per_token == 2].sum()
        time_per_token = output.time_mask
        time_encode_per_instance = (time_per_token[time_per_token == 0] + 1).sum()
        time_decode_per_instance = time_per_token[time_per_token == 2].sum()
        static_per_token = output.static_mask
        static_encode_per_instance = (static_per_token[static_per_token == 0] + 1).sum()
        static_decode_per_instance = static_per_token[static_per_token == 2].sum()
        total_tokens = (
            (h_tokens * w_tokens * t * len(SPACE_TIME_BANDS_GROUPS_IDX))
            # -1 because we have now masked out dynamic world
            + (h_tokens * w_tokens * (len(SPACE_BAND_GROUPS_IDX) - 1))
            + (t * len(TIME_BAND_GROUPS_IDX))
            # -1 because we have now masked out dynamic world static
            + (len(STATIC_BAND_GROUPS_IDX) - 1)
        ) * b
        # handles off by one errors
        self.assertTrue(
            (
                space_time_encode_per_instance
                + space_encode_per_instance
                + time_encode_per_instance
                + static_encode_per_instance
                <= int(total_tokens * ratio) + 1
            ).all()
            and (
                space_time_encode_per_instance
                + space_encode_per_instance
                + time_encode_per_instance
                + static_encode_per_instance
                >= int(total_tokens * ratio) - 1
            ).all()
        )
        self.assertTrue(
            (
                # hacky but the / 2 lets us easily handle the fact
                # we are summing over values == 2, not 1
                (
                    space_time_decode_per_instance
                    + space_decode_per_instance
                    + time_decode_per_instance
                    + static_decode_per_instance
                )
                / 2
                <= (int(total_tokens * ratio) + 1)
            ).all()
            and (
                (
                    space_time_decode_per_instance
                    + space_decode_per_instance
                    + time_decode_per_instance
                    + static_decode_per_instance
                )
                / 2
                >= int(total_tokens * ratio) - 1
            ).all()
        )

        # check DW was masked out
        expected_mask_index = list(SPACE_BAND_GROUPS_IDX.keys()).index("DW")
        self.assertTrue((output.space_mask[:, :, :, expected_mask_index] == 1).all())

        expected_mask_index = list(STATIC_BAND_GROUPS_IDX.keys()).index("DW_static")
        self.assertTrue((output.static_mask[:, expected_mask_index] == 1).all())

    def test_filter_candidates(self):
        candidates = [
            [("space", "SRTM"), ("space_time", "S2_RGB")],
            [("space", "SRTM"), ("space", "DW")],
            [("space", "DW")],
        ]
        ignore_bands = ["DW"]
        outputs = filter_unmasking_mode_candidates(candidates, ignore_bands)
        print(outputs)
        self.assertEqual(outputs, [[("space", "SRTM"), ("space_time", "S2_RGB")]])
