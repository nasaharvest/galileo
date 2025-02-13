from typing import cast

import torch

from src.data import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    TIME_BAND_GROUPS_IDX,
)
from src.data.dataset import (
    SPACE_BANDS,
    SPACE_TIME_BANDS,
    STATIC_BANDS,
    TIME_BANDS,
    Normalizer,
    to_cartesian,
)
from src.data.earthengine.eo import (
    DW_BANDS,
    ERA5_BANDS,
    LANDSCAN_BANDS,
    LOCATION_BANDS,
    S1_BANDS,
    S2_BANDS,
    SRTM_BANDS,
    TC_BANDS,
    VIIRS_BANDS,
    WC_BANDS,
)
from src.masking import MaskedOutput

DEFAULT_MONTH = 5


def construct_galileo_input(
    s1: torch.Tensor | None = None,  # [H, W, T, D]
    s2: torch.Tensor | None = None,  # [H, W, T, D]
    era5: torch.Tensor | None = None,  # [T, D]
    tc: torch.Tensor | None = None,  # [T, D]
    viirs: torch.Tensor | None = None,  # [T, D]
    srtm: torch.Tensor | None = None,  # [H, W, D]
    dw: torch.Tensor | None = None,  # [H, W, D]
    wc: torch.Tensor | None = None,  # [H, W, D]
    landscan: torch.Tensor | None = None,  # [D]
    latlon: torch.Tensor | None = None,  # [D]
    months: torch.Tensor | None = None,  # [T]
    normalize: bool = False,
):
    space_time_inputs = [s1, s2]
    time_inputs = [era5, tc, viirs]
    space_inputs = [srtm, dw, wc]
    static_inputs = [landscan, latlon]
    devices = [
        x.device
        for x in space_time_inputs + time_inputs + space_inputs + static_inputs
        if x is not None
    ]

    if len(devices) == 0:
        raise ValueError("At least one input must be not None")
    if not all(devices[0] == device for device in devices):
        raise ValueError("Received tensors on multiple devices")
    device = devices[0]

    # first, check all the input shapes are consistent
    timesteps_list = [x.shape[2] for x in space_time_inputs if x is not None] + [
        x.shape[1] for x in time_inputs if x is not None
    ]
    height_list = [x.shape[0] for x in space_time_inputs if x is not None] + [
        x.shape[0] for x in space_inputs if x is not None
    ]
    width_list = [x.shape[1] for x in space_time_inputs if x is not None] + [
        x.shape[1] for x in space_inputs if x is not None
    ]

    if len(timesteps_list) > 0:
        if not all(timesteps_list[0] == timestep for timestep in timesteps_list):
            raise ValueError("Inconsistent number of timesteps per input")
        t = timesteps_list[0]
    else:
        t = 1

    if len(height_list) > 0:
        if not all(height_list[0] == height for height in height_list):
            raise ValueError("Inconsistent heights per input")
        if not all(width_list[0] == width for width in width_list):
            raise ValueError("Inconsistent widths per input")
        h = height_list[0]
        w = width_list[0]
    else:
        h, w = 1, 1

    # now, we can construct our empty input tensors. By default, everything is masked
    s_t_x = torch.zeros((h, w, t, len(SPACE_TIME_BANDS)), dtype=torch.float, device=device)
    s_t_m = torch.ones(
        (h, w, t, len(SPACE_TIME_BANDS_GROUPS_IDX)), dtype=torch.float, device=device
    )
    sp_x = torch.zeros((h, w, len(SPACE_BANDS)), dtype=torch.float, device=device)
    sp_m = torch.ones((h, w, len(SPACE_BAND_GROUPS_IDX)), dtype=torch.float, device=device)
    t_x = torch.zeros((t, len(TIME_BANDS)), dtype=torch.float, device=device)
    t_m = torch.ones((t, len(TIME_BAND_GROUPS_IDX)), dtype=torch.float, device=device)
    st_x = torch.zeros((len(STATIC_BANDS)), dtype=torch.float, device=device)
    st_m = torch.ones((len(STATIC_BAND_GROUPS_IDX)), dtype=torch.float, device=device)

    for x, bands_list, group_key in zip([s1, s2], [S1_BANDS, S2_BANDS], ["S1", "S2"]):
        if x is not None:
            indices = [idx for idx, val in enumerate(SPACE_TIME_BANDS) if val in bands_list]
            groups_idx = [
                idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if group_key in key
            ]
            s_t_x[:, :, :, indices] = x
            s_t_m[:, :, :, groups_idx] = 0

    for x, bands_list, group_key in zip(
        [srtm, dw, wc], [SRTM_BANDS, DW_BANDS, WC_BANDS], ["SRTM", "DW", "WC"]
    ):
        if x is not None:
            indices = [idx for idx, val in enumerate(SPACE_BANDS) if val in bands_list]
            groups_idx = [idx for idx, key in enumerate(SPACE_BAND_GROUPS_IDX) if group_key in key]
            sp_x[:, :, indices] = x
            sp_m[:, :, groups_idx] = 0

    for x, bands_list, group_key in zip(
        [era5, tc, viirs], [ERA5_BANDS, TC_BANDS, VIIRS_BANDS], ["ERA5", "TC", "VIIRS"]
    ):
        if x is not None:
            indices = [idx for idx, val in enumerate(TIME_BANDS) if val in bands_list]
            groups_idx = [idx for idx, key in enumerate(TIME_BAND_GROUPS_IDX) if group_key in key]
            t_x[:, indices] = x
            t_m[:, groups_idx] = 0

    for x, bands_list, group_key in zip(
        [landscan, latlon], [LANDSCAN_BANDS, LOCATION_BANDS], ["LS", "location"]
    ):
        if x is not None:
            if group_key == "location":
                # transform latlon to cartesian
                x = cast(torch.Tensor, to_cartesian(x[0], x[1]))
            indices = [idx for idx, val in enumerate(STATIC_BANDS) if val in bands_list]
            groups_idx = [
                idx for idx, key in enumerate(STATIC_BAND_GROUPS_IDX) if group_key in key
            ]
            st_x[indices] = x
            st_m[groups_idx] = 0

    if months is None:
        months = torch.ones((t), dtype=torch.long, device=device) * DEFAULT_MONTH
    else:
        if months.shape[0] != t:
            raise ValueError("Incorrect number of input months")

    if normalize:
        normalizer = Normalizer(std=False)
        s_t_x = torch.from_numpy(normalizer(s_t_x.cpu().numpy())).to(device)
        sp_x = torch.from_numpy(normalizer(sp_x.cpu().numpy())).to(device)
        t_x = torch.from_numpy(normalizer(t_x.cpu().numpy())).to(device)
        st_x = torch.from_numpy(normalizer(st_x.cpu().numpy())).to(device)

    return MaskedOutput(
        space_time_x=s_t_x,
        space_time_mask=s_t_m,
        space_x=sp_x,
        space_mask=sp_m,
        time_x=t_x,
        time_mask=t_m,
        static_x=st_x,
        static_mask=st_m,
        months=months,
    )
