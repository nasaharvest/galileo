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


# this normalizing dict is sourced from
# https://github.com/nasaharvest/galileo/blob/main/config/normalization.json
# its used to normalize the data. The keys (e.g. "13") are used to query
# which tensor (e.g. space_time_x) is associated to the means and stds,
# where the key represents the number of dimensions in the tensor (i.e. x.shape[-1])
PRETRAINING_NORMALIZING_DICT = {
    "total_n": 127155,
    "sampled_n": 10000,
    "13": {
        "mean": [
            -11.728724389184965,
            -18.85558188024017,
            1395.3408730676722,
            1338.4026921784578,
            1343.09883810357,
            1543.8607982512297,
            2186.2022069512263,
            2525.0932853316694,
            2410.3377187373408,
            2750.2854646886753,
            2234.911100061487,
            1474.5311266077113,
            0.2892116502999044,
        ],
        "std": [
            4.887145774840316,
            5.730270320384293,
            917.7041440370853,
            913.2988423581528,
            1092.678723527555,
            1047.2206083460424,
            1048.0101611156767,
            1143.6903026819996,
            1098.979177731649,
            1204.472755085893,
            1145.9774063078878,
            980.2429840007796,
            0.2720939024500081,
        ],
    },
    "16": {
        "mean": [
            673.0152819503361,
            5.930092668915115,
            0.10470439140978786,
            0.23965913270066183,
            0.08158044385860364,
            0.04246976254259546,
            0.11304392863520317,
            0.17329647890362473,
            0.0698981691616277,
            0.12130267132802142,
            0.04671318615236216,
            10.973119802517362,
            1.0927069179958768,
            1.6991394232855903,
            0.03720594618055555,
            1.3671352688259548,
        ],
        "std": [
            983.0697298296237,
            8.167406789813247,
            0.18771647977504985,
            0.2368313455675914,
            0.08024268534756586,
            0.04045374496146404,
            0.11350342472061795,
            0.1279898111718168,
            0.12042341550438586,
            0.13602408145504347,
            0.043971116096060345,
            31.255340146970997,
            10.395974878206689,
            12.92380617159917,
            1.9285254295940466,
            11.612179775408928,
        ],
    },
    "6": {
        "mean": [
            271.5674963541667,
            0.08554303677156568,
            657.3181260091111,
            692.1291795806885,
            562.781331880633,
            1.5647115934036673,
        ],
        "std": [
            79.80828940314429,
            0.11669547098151486,
            704.0008695557707,
            925.0116126406431,
            453.2434022278578,
            7.513020170832818,
        ],
    },
    "18": {
        "mean": [
            188.20315880851746,
            0.2804946561574936,
            0.11371652073860168,
            0.058778801321983334,
            0.10474256777763366,
            0.2396918488264084,
            0.08152248692512512,
            0.04248040814399719,
            0.11303179881572724,
            0.17326324067115784,
            0.06998309404850006,
            0.12122812910079957,
            0.04671641788482666,
            10.98456594619751,
            1.0968475807189941,
            1.6947754135131836,
            0.03320046615600586,
            1.3602827312469483,
        ],
        "std": [
            1154.5919128300602,
            0.5276998078079327,
            0.7021637331734328,
            0.36528892213195063,
            0.17470213191865785,
            0.20411195416718833,
            0.0660782470089761,
            0.03380702424871257,
            0.09809195568521663,
            0.11292471052124119,
            0.09720748930233268,
            0.12912217763726777,
            0.0399973913151906,
            23.725471823867462,
            5.715238079725388,
            9.030481416228302,
            0.9950220242487364,
            7.754429123862099,
        ],
    },
}


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
        normalizer = Normalizer(normalizing_dicts=PRETRAINING_NORMALIZING_DICT, std=True)
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
