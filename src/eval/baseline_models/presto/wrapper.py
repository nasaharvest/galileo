from pathlib import Path
from typing import Optional

import torch
from einops import repeat
from torch import nn

from .single_file_presto import (
    NUM_DYNAMIC_WORLD_CLASSES,
    PRESTO_BANDS,
    PRESTO_S1_BANDS,
    PRESTO_S2_BANDS,
    Presto,
)

WEIGHTS_PATH = Path(__file__).parent / "default_model.pt"
assert WEIGHTS_PATH.exists()

INPUT_PRESTO_BANDS = [b for b in PRESTO_BANDS if b != "B9"]
INPUT_PRESTO_S2_BANDS = [b for b in PRESTO_S2_BANDS if b != "B9"]


class PrestoWrapper(nn.Module):
    # we assume any data passed to this wrapper
    # will contain S2 data with the following channels
    S2_BAND_ORDERING = [
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
    S1_BAND_ORDERING = [
        "VV",
        "VH",
    ]

    def __init__(self, do_pool=True, temporal_pooling: str = "mean"):
        super().__init__()

        model = Presto.construct()
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))

        self.encoder = model.encoder
        self.dim = self.encoder.embedding_size
        self.do_pool = do_pool
        if temporal_pooling != "mean":
            raise ValueError("Only mean temporal pooling supported by Presto")
        if not do_pool:
            raise ValueError("Presto cannot output spatial tokens")

        self.kept_s2_band_idx = [
            i for i, v in enumerate(self.S2_BAND_ORDERING) if v in INPUT_PRESTO_S2_BANDS
        ]
        self.kept_s1_band_idx = [
            i for i, v in enumerate(self.S1_BAND_ORDERING) if v in PRESTO_S1_BANDS
        ]
        kept_s2_band_names = [val for val in self.S2_BAND_ORDERING if val in INPUT_PRESTO_S2_BANDS]
        kept_s1_band_names = [val for val in self.S1_BAND_ORDERING if val in PRESTO_S1_BANDS]
        self.to_presto_s2_map = [PRESTO_BANDS.index(val) for val in kept_s2_band_names]
        self.to_presto_s1_map = [PRESTO_BANDS.index(val) for val in kept_s1_band_names]

        self.month = 6  # default month

    def preproccess(
        self,
        s2: Optional[torch.Tensor] = None,
        s1: Optional[torch.Tensor] = None,
        months: Optional[torch.Tensor] = None,
    ):
        # images should have shape (b h w c) or (b h w t c)
        if s2 is not None:
            data_device = s2.device
            if len(s2.shape) == 4:
                b, h, w, c_s2 = s2.shape
                t = 1
                s2 = repeat(torch.mean(s2, dim=(1, 2)), "b d -> b t d", t=1)
            else:
                assert len(s2.shape) == 5
                b, h, w, t, c_s2 = s2.shape
                s2 = torch.mean(s2, dim=(1, 2))
            assert c_s2 == len(self.S2_BAND_ORDERING)

            x = torch.zeros((b, t, len(INPUT_PRESTO_BANDS)), dtype=s2.dtype, device=s2.device)
            x[:, :, self.to_presto_s2_map] = s2[:, :, self.kept_s2_band_idx]

        elif s1 is not None:
            data_device = s1.device
            if len(s1.shape) == 4:
                b, h, w, c_s1 = s1.shape
                t = 1
                s1 = repeat(torch.mean(s1, dim=(1, 2)), "b d -> b t d", t=1)
            else:
                assert len(s1.shape) == 5
                b, h, w, t, c_s1 = s1.shape
                s1 = torch.mean(s1, dim=(1, 2))
            assert c_s1 == len(self.S1_BAND_ORDERING)

            # add a single timestep
            x = torch.zeros((b, t, len(INPUT_PRESTO_BANDS)), dtype=s1.dtype, device=s1.device)
            x[:, :, self.to_presto_s1_map] = s1[:, :, self.kept_s1_band_idx]

        else:
            raise ValueError("no s1 or s2?")
        s_t_m = torch.ones(
            (b, t, len(INPUT_PRESTO_BANDS)),
            dtype=x.dtype,
            device=x.device,
        )
        if s2 is not None:
            s_t_m[:, :, self.to_presto_s2_map] = 0
        elif s1 is not None:
            s_t_m[:, :, self.to_presto_s1_map] = 0

        if months is None:
            months = torch.ones((b, t), device=data_device) * self.month
        else:
            assert months.shape[-1] == t

        dymamic_world = torch.ones((b, t), device=data_device) * NUM_DYNAMIC_WORLD_CLASSES

        return (
            x,
            s_t_m,
            dymamic_world.long(),
            months.long(),
        )

    def forward(self, s2=None, s1=None, months=None):
        x, mask, dynamic_world, months = self.preproccess(s2=s2, s1=s1, months=months)
        return self.encoder(
            x=x, dynamic_world=dynamic_world, mask=mask, month=months, eval_task=True
        )  # [B, self.dim]
