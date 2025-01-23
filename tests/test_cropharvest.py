import unittest

import numpy as np

from src.data.dataset import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_BANDS,
    SPACE_TIME_BANDS,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    STATIC_BANDS,
    TIME_BAND_GROUPS_IDX,
    TIME_BANDS,
    Normalizer,
)
from src.eval.cropharvest.cropharvest_eval import BANDS, BinaryCropHarvestEval


class TestCropHarvest(unittest.TestCase):
    def test_to_galileo_arrays(self):
        for ignore_band_groups in [["S1"], None]:
            for include_latlons in [True, False]:

                class BinaryCropHarvestEvalNoDownload(BinaryCropHarvestEval):
                    def __init__(self, ignore_band_groups, include_latlons: bool = True):
                        self.include_latlons = include_latlons
                        self.normalizer = Normalizer(std=False)
                        self.ignore_s_t_band_groups = self.indices_of_ignored(
                            SPACE_TIME_BANDS_GROUPS_IDX, ignore_band_groups
                        )
                        self.ignore_sp_band_groups = self.indices_of_ignored(
                            SPACE_BAND_GROUPS_IDX, ignore_band_groups
                        )
                        self.ignore_t_band_groups = self.indices_of_ignored(
                            TIME_BAND_GROUPS_IDX, ignore_band_groups
                        )
                        self.ignore_st_band_groups = self.indices_of_ignored(
                            STATIC_BAND_GROUPS_IDX, ignore_band_groups
                        )

                eval = BinaryCropHarvestEvalNoDownload(ignore_band_groups, include_latlons)
                b, t = 8, 12
                array = np.ones((b, t, len(BANDS)))
                latlons = np.ones((b, 2))
                (
                    s_t_x,
                    sp_x,
                    t_x,
                    st_x,
                    s_t_m,
                    sp_m,
                    t_m,
                    st_m,
                    months,
                ) = eval.cropharvest_array_to_normalized_galileo(array, latlons, start_month=1)

                self.assertEqual(s_t_x.shape, (b, 1, 1, t, len(SPACE_TIME_BANDS)))
                self.assertEqual(s_t_m.shape, (b, 1, 1, t, len(SPACE_TIME_BANDS_GROUPS_IDX)))
                if ignore_band_groups is not None:
                    # check s1 got masked
                    self.assertTrue(
                        (
                            s_t_m[:, :, :, :, list(SPACE_TIME_BANDS_GROUPS_IDX.keys()).index("S1")]
                            == 1
                        ).all()
                    )
                self.assertEqual(sp_x.shape, (b, 1, 1, len(SPACE_BANDS)))
                self.assertEqual(sp_m.shape, (b, 1, 1, len(SPACE_BAND_GROUPS_IDX)))
                self.assertTrue(
                    (sp_m[:, :, :, list(SPACE_BAND_GROUPS_IDX.keys()).index("SRTM")] == 0).all()
                )
                self.assertTrue(
                    (sp_m[:, :, :, list(SPACE_BAND_GROUPS_IDX.keys()).index("DW")] == 1).all()
                )
                self.assertEqual(t_x.shape, (b, t, len(TIME_BANDS)))
                self.assertEqual(t_m.shape, (b, t, len(TIME_BAND_GROUPS_IDX)))
                self.assertEqual(st_x.shape, (b, len(STATIC_BANDS)))
                self.assertEqual(st_m.shape, (b, len(STATIC_BAND_GROUPS_IDX)))
                self.assertEqual(months.shape, (b, t))
