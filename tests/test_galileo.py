import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from einops import repeat

from single_file_galileo import Encoder as SingleFileEncoder
from src.data import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    TIME_BAND_GROUPS_IDX,
    Dataset,
)
from src.data.config import CONFIG_FILENAME, ENCODER_FILENAME
from src.data.dataset import DatasetOutput
from src.galileo import (
    SPACE_BANDS,
    SPACE_TIME_BANDS,
    STATIC_BANDS,
    TIME_BANDS,
    Decoder,
    Encoder,
)
from src.masking import (
    MASKING_MODES,
    MaskingFunctions,
    batch_mask_space,
    batch_mask_time,
    batch_subset_mask_galileo,
)
from src.utils import device, load_check_config

DATA_FOLDER = Path(__file__).parents[1] / "data"
TIFS_FOLDER = DATA_FOLDER / "tifs"
TEST_MODEL_FOLDER = Path(__file__).parents[0] / "141"


class TestGalileo(unittest.TestCase):
    @staticmethod
    def to_tensor_with_batch_d(input: DatasetOutput):
        return (
            torch.from_numpy(input.space_time_x).float().unsqueeze(0),
            torch.from_numpy(input.space_x).float().unsqueeze(0),
            torch.from_numpy(input.time_x).float().unsqueeze(0),
            torch.from_numpy(input.static_x).float().unsqueeze(0),
            torch.from_numpy(input.months).long().unsqueeze(0),
        )

    def test_end_to_end(self):
        self._end_to_end_run(16, 8)

    def test_end_to_end_different_inputs_per_dim_than_default(self):
        self._end_to_end_run(16, 4)

    def _end_to_end_run(self, embedding_size, patch_size):
        image_size = patch_size * 4
        num_timesteps = 3
        encoder = Encoder(embedding_size=embedding_size, num_heads=1)
        decoder = Decoder(
            encoder_embedding_size=embedding_size,
            decoder_embedding_size=embedding_size,
            num_heads=1,
        )
        ds = Dataset(TIFS_FOLDER, False)
        for i in range(len(ds)):
            s_t_x, sp_x, t_x, st_x, months = self.to_tensor_with_batch_d(ds[i])
            masked_output = batch_subset_mask_galileo(
                s_t_x,
                sp_x,
                t_x,
                st_x,
                months,
                encode_ratio=0.25,
                decode_ratio=0.25,
                patch_size=patch_size,
                image_size=image_size,
                num_timesteps=num_timesteps,
                augmentation_strategies=None,
                masking_probabilities=[1] * len(MASKING_MODES),
                masking_function=MaskingFunctions.SPACE,
                max_unmasking_channels=4,
            )

            # for now, we just make sure it all runs
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                encoder_output = encoder(
                    masked_output.space_time_x,
                    masked_output.space_x,
                    masked_output.time_x,
                    masked_output.static_x,
                    masked_output.space_time_mask,
                    masked_output.space_mask,
                    masked_output.time_mask,
                    masked_output.static_mask,
                    masked_output.months.long(),
                    patch_size=patch_size,
                )
                output = decoder(*encoder_output)

                with torch.no_grad():
                    t_s_t, t_sp, t_t, t_st, _, _, _, _ = encoder.apply_linear_projection(
                        masked_output.space_time_x,
                        masked_output.space_x,
                        masked_output.time_x,
                        masked_output.static_x,
                        ~(masked_output.space_time_mask == 2),  # we want 0s where the mask == 2
                        ~(masked_output.space_mask == 2),
                        ~(masked_output.time_mask == 2),
                        ~(masked_output.static_mask == 2),
                        patch_size,
                    )
                    t_s_t = encoder.blocks[0].norm1(t_s_t)
                    t_sp = encoder.blocks[0].norm1(t_sp)
                    t_sp = encoder.blocks[0].norm1(t_sp)
                    t_st = encoder.blocks[0].norm1(t_st)

            self.assertFalse(
                torch.isnan(
                    t_s_t[masked_output.space_time_mask[:, 0::patch_size, 0::patch_size] == 2]
                ).any()
            )
            self.assertFalse(
                torch.isnan(
                    t_sp[masked_output.space_mask[:, 0::patch_size, 0::patch_size] == 2]
                ).any()
            )
            self.assertFalse(torch.isnan(t_t[masked_output.time_mask == 2]).any())
            self.assertFalse(torch.isnan(t_st[masked_output.static_mask == 2]).any())
            self.assertTrue(
                list(encoder_output[0].shape)
                == [
                    1,
                    image_size / patch_size,
                    image_size / patch_size,
                    num_timesteps,
                    len(SPACE_TIME_BANDS_GROUPS_IDX),
                    embedding_size,
                ]
            )
            self.assertTrue(
                list(encoder_output[1].shape)
                == [
                    1,
                    image_size / patch_size,
                    image_size / patch_size,
                    len(SPACE_BAND_GROUPS_IDX),
                    embedding_size,
                ]
            )
            self.assertTrue(
                list(encoder_output[2].shape)
                == [
                    1,
                    num_timesteps,
                    len(TIME_BAND_GROUPS_IDX),
                    embedding_size,
                ]
            )
            self.assertTrue(
                list(encoder_output[3].shape)
                == [
                    1,
                    len(STATIC_BAND_GROUPS_IDX),
                    embedding_size,
                ]
            )
            self.assertFalse(
                torch.isnan(
                    encoder_output[0][
                        masked_output.space_time_mask[:, 0::patch_size, 0::patch_size] == 0
                    ]
                ).any()
            )
            self.assertFalse(
                torch.isnan(
                    encoder_output[1][
                        masked_output.space_mask[:, 0::patch_size, 0::patch_size] == 0
                    ]
                ).any()
            )
            self.assertFalse(torch.isnan(encoder_output[2][masked_output.time_mask == 0]).any())
            self.assertFalse(torch.isnan(encoder_output[3][masked_output.static_mask == 0]).any())

            self.assertTrue(
                list(output[0].shape)
                == [
                    1,
                    image_size / patch_size,
                    image_size / patch_size,
                    num_timesteps,
                    len(SPACE_TIME_BANDS_GROUPS_IDX),
                    embedding_size,
                ]
            )
            self.assertTrue(
                list(output[1].shape)
                == [
                    1,
                    image_size / patch_size,
                    image_size / patch_size,
                    len(SPACE_BAND_GROUPS_IDX),
                    embedding_size,
                ]
            )
            self.assertTrue(
                list(output[2].shape)
                == [1, num_timesteps, len(TIME_BAND_GROUPS_IDX), embedding_size]
            )
            self.assertTrue(
                list(output[3].shape) == [1, len(STATIC_BAND_GROUPS_IDX), embedding_size]
            )

            self.assertFalse(
                torch.isnan(
                    output[0][masked_output.space_time_mask[:, 0::patch_size, 0::patch_size] == 2]
                ).any()
            )
            self.assertFalse(
                torch.isnan(
                    output[1][masked_output.space_mask[:, 0::patch_size, 0::patch_size] == 2]
                ).any()
            )
            self.assertFalse(torch.isnan(output[2][masked_output.time_mask == 2]).any())
            self.assertFalse(torch.isnan(output[3][masked_output.static_mask == 2]).any())

            # check we can call backwards, with the loss
            summed_output = sum([torch.sum(o) for o in output])
            summed_output.backward()

    def test_decoder_add_masks(self):
        embedding_size = 16
        decoder = Decoder(
            encoder_embedding_size=embedding_size,
            decoder_embedding_size=embedding_size,
            num_heads=1,
        )
        b, h, w, t = 5, 6, 7, 8
        s_t_x = torch.ones(b, h, w, t, len(SPACE_TIME_BANDS_GROUPS_IDX), embedding_size)
        s_t_m = torch.zeros(b, h, w, t, len(SPACE_TIME_BANDS_GROUPS_IDX))
        s_t_m[:, :, :, 0] = 2  # the first timestep will get processed by the decoder
        s_t_m[:, :, :, 1] = 1  # the second timestep gets masked but not processed

        sp_x = torch.ones(b, h, w, len(SPACE_BAND_GROUPS_IDX), embedding_size)
        sp_m = torch.zeros(b, h, w, len(SPACE_BAND_GROUPS_IDX))
        sp_m[:, 0] = 2
        sp_m[:, 1] = 1

        t_x = torch.ones(b, t, len(TIME_BAND_GROUPS_IDX), embedding_size)
        t_m = torch.zeros(b, t, len(TIME_BAND_GROUPS_IDX))
        t_m[:, 0] = 2
        t_m[:, 1] = 1

        st_x = torch.ones(b, len(STATIC_BAND_GROUPS_IDX), embedding_size)
        st_m = torch.zeros(b, len(STATIC_BAND_GROUPS_IDX))
        st_m[:, 0] = 2
        st_m[:, 1] = 1

        with torch.no_grad():
            o = decoder.add_masks(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m)

        self.assertTrue((o[0][:, :, :, 0] == 0).all())
        self.assertTrue((o[0][:, :, :, 1:] == 1).all())
        self.assertTrue((o[1][:, 0] == 0).all())
        self.assertTrue((o[1][:, 1:] == 1).all())
        self.assertTrue((o[2][:, 0] == 0).all())
        self.assertTrue((o[2][:, 1:] == 1).all())
        self.assertTrue((o[3][:, 0] == 0).all())
        self.assertTrue((o[3][:, 1:] == 1).all())

    def test_mean_of_tokens(self):
        b, t, d, h, w, s_t_c_g, sp_c_g, t_c_g, st_c_g = 1, 2, 8, 3, 3, 5, 6, 2, 4
        s_t_x = torch.ones((b, h, w, t, s_t_c_g, d))
        sp_x = torch.ones((b, h, w, sp_c_g, d))
        t_x = torch.ones((b, t, t_c_g, d))
        st_x = torch.ones((b, st_c_g, d))

        # the first timestep and the first column are masked
        s_t_m = torch.zeros((b, h, w, t, s_t_c_g))
        s_t_m[:, :, 0, :] = 1
        s_t_m[:, :, :, 0] = 1
        s_t_x[:, :, 0, :] = 0
        s_t_x[:, :, :, 0] = 0
        # the last row is masked
        sp_m = torch.zeros((b, h, w, sp_c_g))
        sp_m[:, -1, :] = 1
        sp_x[:, -1, :] = 0
        # the first timestep is masked
        t_m = torch.zeros((b, t, t_c_g))
        t_m[:, 0] = 1
        t_x[:, 0] = 0
        # the last column is masked
        st_m = torch.zeros((b, st_c_g))
        st_m[:, -1] = 1
        st_x[:, -1] = 0

        mean = Encoder.average_tokens(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m)
        self.assertEqual(mean.shape, (b, d))
        self.assertTrue((mean == 1).all())

    def test_mask_and_unmask_tokens(self):
        b, d = 2, 2
        x = torch.tensor([[0, 1, 0], [1, 0, 1]]).float()
        x = repeat(x, "b n -> b n d", d=d)
        mask = torch.tensor([[1, 0, 1], [0, 1, 0]]).float()

        out_x, indices, updated_mask = Encoder.remove_masked_tokens(x, mask)
        self.assertEqual(out_x.dtype, x.dtype)
        self.assertEqual(updated_mask.dtype, mask.dtype)
        self.assertEqual(out_x.shape, (b, 2, d))
        # for the 2nd item in the batch, there should be only 0s
        self.assertTrue(torch.equal(out_x[1], torch.ones_like(out_x[1])))
        # for the first item in the batch, only the first index is unmasked so
        # it should be at the front
        self.assertEqual(indices[0, 0], 1)
        # for the second item, the 0th and 2nd are masked
        self.assertTrue(torch.equal(indices[1, :2], torch.tensor([0, 2])))
        self.assertEqual(updated_mask.shape, (b, 2))
        self.assertTrue(torch.equal(updated_mask, torch.Tensor([[0, 1], [0, 0]])))

        # check that when we add things back, they are once again what we had originally
        final_x, final_mask = Encoder.add_removed_tokens(out_x, indices, updated_mask)
        self.assertEqual(final_x.dtype, x.dtype)
        self.assertEqual(final_mask.dtype, mask.dtype)
        self.assertTrue(torch.equal(final_x, x))
        self.assertTrue(torch.equal(final_mask, mask))

    def test_combine_x_y(self):
        # x is the query (i.e. the masked tokens)
        x = torch.tensor([[14, 15, 16], [15, 16, 1]]).unsqueeze(-1)
        # y is the keys and values (i.e. the unmasked tokens)
        y = torch.tensor([[5, 6, 7, 8], [4, 5, 6, 7]]).unsqueeze(-1)
        x_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        y_mask = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]])
        indices = torch.tensor([[6, 7, 8, 4, 5, 0, 1, 2, 3], [7, 8, 3, 4, 5, 6, 0, 1, 2]])

        tokens = Decoder.combine_x_y(x, y, x_mask, y_mask, indices)
        self.assertTrue(
            torch.equal(
                tokens,
                torch.tensor(
                    [[5, 6, 7, 8, 0, 0, 14, 15, 16], [5, 6, 7, 0, 0, 0, 0, 15, 16]]
                ).unsqueeze(-1),
            )
        )

    def test_split_x_y(self):
        tokens = torch.tensor(
            [[5, 6, 7, 8, 2, 13, 14, 15, 16], [5, 6, 7, 1, 2, 3, 4, 15, 16]]
        ).unsqueeze(-1)
        mask = torch.tensor([[0, 0, 0, 0, 1, 1, 2, 2, 2], [0, 0, 0, 1, 1, 1, 1, 2, 2]])

        x, y, x_mask, y_mask, _ = Decoder.split_x_y(tokens, mask)
        self.assertTrue(torch.equal(x, torch.tensor([[14, 15, 16], [15, 16, 1]]).unsqueeze(-1)))
        self.assertTrue(torch.equal(y, torch.tensor([[5, 6, 7, 8], [4, 5, 6, 7]]).unsqueeze(-1)))
        self.assertTrue(torch.equal(x_mask, torch.tensor([[1, 1, 1], [1, 1, 0]])))
        self.assertTrue(torch.equal(y_mask, torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]])))

    def test_x_y_there_and_back_again(self):
        tokens = torch.tensor(
            [[5, 6, 7, 8, 2, 13, 14, 15, 16], [5, 6, 7, 1, 2, 3, 4, 15, 16]]
        ).unsqueeze(-1)
        mask = torch.tensor([[0, 0, 0, 0, 1, 1, 2, 2, 2], [0, 0, 0, 1, 1, 1, 1, 2, 2]])
        x, y, x_mask, y_mask, indices = Decoder.split_x_y(tokens, mask)
        new_tokens = Decoder.combine_x_y(x, y, x_mask, y_mask, indices)
        tokens[mask == 1] = 0
        self.assertTrue(torch.equal(tokens, new_tokens))

    def test_load_from_device(self):
        config = load_check_config("nano.json")
        original_encoder = Encoder(**config["model"]["encoder"])

        with tempfile.TemporaryDirectory() as tempdir:
            torch.save(original_encoder.state_dict(), Path(tempdir) / ENCODER_FILENAME)
            with (Path(tempdir) / CONFIG_FILENAME).open("w") as f:
                json.dump(config, f)

            new_encoder = Encoder.load_from_folder(Path(tempdir))

        for key, val in new_encoder.state_dict().items():
            self.assertTrue(torch.equal(val, original_encoder.state_dict()[key]))

    def test_decoder_and_mask_static(self):
        patch_size = 4
        ratio = 0.25

        ds = Dataset(TIFS_FOLDER, False)
        tensor_batch = self.to_tensor_with_batch_d(ds[0])
        self.assertTrue(tensor_batch[0].shape[1] == tensor_batch[0].shape[2])
        for f in [batch_mask_time, batch_mask_space]:
            masked_output = f(
                *tensor_batch,
                encode_ratio=ratio,
                decode_ratio=ratio,
                mode=[("space", "DW")],
                decoder_mode=[("static", "LS")],
                patch_size=patch_size,
            )

            encoder = Encoder(embedding_size=32, num_heads=1)
            decoder = Decoder(
                encoder_embedding_size=32,
                decoder_embedding_size=32,
                num_heads=1,
            )
            encoder_output = encoder(
                masked_output.space_time_x,
                masked_output.space_x,
                masked_output.time_x,
                masked_output.static_x,
                masked_output.space_time_mask,
                masked_output.space_mask,
                masked_output.time_mask,
                masked_output.static_mask,
                masked_output.months.long(),
                patch_size=patch_size,
            )
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, _ = encoder_output
            x, m = decoder.collapse_and_combine_hwtc(
                s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m
            )
            x, _, _, _, _ = decoder.split_x_y(x, m)
            self.assertTrue(x.shape[1] == 1, x.shape)

    def test_token_exit_cfgs_single_exit_equivalency(self):
        self._token_exit_cfgs_single_exit_equivalency(0)
        self._token_exit_cfgs_single_exit_equivalency(6)
        self._token_exit_cfgs_single_exit_equivalency(12)

    @torch.no_grad()
    def _token_exit_cfgs_single_exit_equivalency(self, depth):
        embedding_size, patch_size = 16, 1
        image_size = patch_size * 4
        num_timesteps = 3
        encoder = Encoder(embedding_size=embedding_size, num_heads=1, depth=12)
        encoder.eval()
        ds = Dataset(TIFS_FOLDER, False)
        for i in range(len(ds)):
            s_t_x, sp_x, t_x, st_x, months = self.to_tensor_with_batch_d(ds[i])
            masked_output = batch_subset_mask_galileo(
                s_t_x,
                sp_x,
                t_x,
                st_x,
                months,
                encode_ratio=0.25,
                decode_ratio=0.25,
                patch_size=patch_size,
                image_size=image_size,
                num_timesteps=num_timesteps,
                augmentation_strategies=None,
                masking_probabilities=[1] * len(MASKING_MODES),
                masking_function=MaskingFunctions.SPACE,
                max_unmasking_channels=4,
            )

            # for this test, we will keep the same
            # values per shape since we call layer norm
            # on each shape output
            token_exit_cfgs = {
                "S1": depth,
                "S2_RGB": depth,
                "S2_Red_Edge": depth,
                "S2_NIR_10m": depth,
                "S2_NIR_20m": depth,
                "S2_SWIR": depth,
                "NDVI": depth,
                "ERA5": depth,
                "TC": depth,
                "VIIRS": depth,
                "SRTM": depth,
                "DW": depth,
                "WC": depth,
                "LS": depth,
                "location": depth,
                "DW_static": depth,
                "WC_static": depth,
            }

            encoder_output_depth = encoder(
                masked_output.space_time_x,
                masked_output.space_x,
                masked_output.time_x,
                masked_output.static_x,
                torch.zeros_like(masked_output.space_time_mask),
                torch.zeros_like(masked_output.space_mask),
                torch.zeros_like(masked_output.time_mask),
                torch.zeros_like(masked_output.static_mask),
                masked_output.months.long(),
                patch_size=patch_size,
                exit_after=depth,
            )

            encoder_output_depth_varied = encoder(
                masked_output.space_time_x,
                masked_output.space_x,
                masked_output.time_x,
                masked_output.static_x,
                torch.zeros_like(masked_output.space_time_mask),
                torch.zeros_like(masked_output.space_mask),
                torch.zeros_like(masked_output.time_mask),
                torch.zeros_like(masked_output.static_mask),
                masked_output.months.long(),
                patch_size=patch_size,
                token_exit_cfg=token_exit_cfgs,
                exit_after=None,
            )

            # s_t_x
            self.assertTrue(torch.equal(encoder_output_depth_varied[0], encoder_output_depth[0]))

            # sp_x
            self.assertTrue(torch.equal(encoder_output_depth_varied[1], encoder_output_depth[1]))

            # t_x
            self.assertTrue(torch.equal(encoder_output_depth_varied[2], encoder_output_depth[2]))

            # st_x
            self.assertTrue(torch.equal(encoder_output_depth_varied[3], encoder_output_depth[3]))

    def test_single_file_galileo_matches_galileo(self):
        org_model = Encoder.load_from_folder(DATA_FOLDER / "models/nano")
        sf_model = SingleFileEncoder.load_from_folder(
            DATA_FOLDER / "models/nano", device=torch.device("cpu")
        )

        for model_p, sf_model_p in zip(org_model.parameters(), sf_model.parameters()):
            self.assertTrue(torch.equal(model_p, sf_model_p))

    def test_nans(self):
        m = Encoder.load_from_folder(DATA_FOLDER / "models/nano")

        B = 2
        H = 1
        W = 1
        T = 5

        s_t_x = torch.randn(B, H, W, T, len(SPACE_TIME_BANDS))
        sp_x = torch.randn(B, H, W, len(SPACE_BANDS))
        t_x = torch.randn(B, T, len(TIME_BANDS))
        st_x = torch.randn(B, len(STATIC_BANDS))
        s_t_m = torch.ones(B, H, W, T, len(SPACE_TIME_BANDS_GROUPS_IDX))
        sp_m = torch.ones(B, H, W, len(SPACE_BAND_GROUPS_IDX))
        t_m = torch.ones(B, T, len(TIME_BAND_GROUPS_IDX))
        st_m = torch.ones(B, len(STATIC_BAND_GROUPS_IDX))
        months = torch.ones(B, T, dtype=torch.int)

        # Half of the samples have a sequence length of 3
        s_t_m[:1, :, :, 0:3, :2] = 0
        # Other half of the samples have a sequence length of 4
        s_t_m[1:, :, :, 0:4, :2] = 0
        for r in range(100):
            # Allocate some random amount of memory and set it to inf
            # making it more likely that inf values will end up in memory returned by 'torch.empty'
            N = np.random.randint(1, 10) * B * H * W * T
            _ = torch.full((N,), fill_value=torch.inf)
            e = m(
                s_t_x,
                sp_x,
                t_x,
                st_x,
                s_t_m,
                sp_m,
                t_m,
                st_m,
                months,
                patch_size=1,
                input_resolution_m=10,
                add_layernorm_on_exit=False,
            )
            assert not m.average_tokens(*(e[:-1])).isnan().any()
            assert not e[0][:1, :, :, 0:3, 0].isnan().any()
            assert not e[0][1:, :, :, 0:4, 0].isnan().any()
