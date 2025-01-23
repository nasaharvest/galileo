import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import dateutil.tz
import numpy as np
import torch

from .config import DEFAULT_SEED
from .data.dataset import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    TIME_BAND_GROUPS_IDX,
)
from .masking import MASKING_MODES, MaskedOutput

data_dir = Path(__file__).parent.parent / "data"
logging_dir = Path(__file__).parent.parent / "logs"
config_dir = Path(__file__).parent.parent / "config"

if not torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)


def will_cause_nans(x: torch.Tensor):
    return torch.isnan(x).any() or torch.isinf(x).any()


# From https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
def seed_everything(seed: int = DEFAULT_SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def masked_output_np_to_tensor(
    s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, month
) -> MaskedOutput:
    """converts eval task"""
    return MaskedOutput(
        torch.as_tensor(s_t_x, dtype=torch.float32),
        torch.as_tensor(sp_x, dtype=torch.float32),
        torch.as_tensor(t_x, dtype=torch.float32),
        torch.as_tensor(st_x, dtype=torch.float32),
        torch.as_tensor(s_t_m, dtype=torch.float32),
        torch.as_tensor(sp_m, dtype=torch.float32),
        torch.as_tensor(t_m, dtype=torch.float32),
        torch.as_tensor(st_m, dtype=torch.float32),
        torch.as_tensor(month, dtype=torch.long),
    )


class AverageMeter:
    """computes and stores the average and current value"""

    average: float
    sum: float
    count: int

    def __init__(self):
        self.average = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.average = self.sum / self.count


def check_config(config):
    """
    this function modifies the config
    primarily so that legacy configs
    work with updates to the code
    """
    expected_training_keys_type = {
        "num_epochs": int,
        "batch_size": int,
        "effective_batch_size": int,
        "patch_sizes": list,
        "max_lr": float,
        "final_lr": float,
        "warmup_epochs": (int, float),
        "shape_time_combinations": list,
        "augmentation": dict,
        "masking_probabilities": list,
        "grad_clip": bool,
        "normalization": str,
        "random_masking": str,
    }
    optional_training_keys_type_default = {
        "target_masking": (str, "decoder_only"),
        "double_loss": (bool, False),
        "double_predictors": (bool, False),
        "ignore_band_groups": (list, None),
    }
    training_dict = config["training"]

    for key, val in expected_training_keys_type.items():
        assert key in training_dict, f"Expected {key} in training dict"
        assert isinstance(
            training_dict[key],
            val,  # type: ignore
        ), f"Expected {key} to be {val}, got {type(training_dict[key])}"
    for key, val in optional_training_keys_type_default.items():
        if key in training_dict:
            assert isinstance(
                training_dict[key], val[0]
            ), f"Expected {key} to be {val}, got {type(training_dict[key])}"
        else:
            print(f"{key} missing from training dict. Filling with default value {val[1]}")
            config["training"][key] = val[1]

    assert ("target_exit_after" in training_dict.keys()) or (
        "token_exit_cfg" in training_dict.keys()
    )

    # legacy: encode_ratio, decode_ratio
    # new - specific for st and random
    if "encode_ratio" in training_dict:
        # then we assign it for both st and random
        training_dict["st_encode_ratio"] = training_dict["encode_ratio"]
        training_dict["random_encode_ratio"] = training_dict["encode_ratio"]
        del training_dict["encode_ratio"]
    else:
        assert "st_encode_ratio" in training_dict
        assert "random_encode_ratio" in training_dict
    if "decode_ratio" in training_dict:
        # then we assign it for both st and random
        training_dict["st_decode_ratio"] = training_dict["decode_ratio"]
        training_dict["random_decode_ratio"] = training_dict["decode_ratio"]
        del training_dict["decode_ratio"]
    else:
        assert "st_decode_ratio" in training_dict
        assert "random_decode_ratio" in training_dict

    # we have three options for the losses:
    # 1. the loss keys are in the training config at a top level (legacy)
    # 2. the loss dict has a single entry "loss_dict" (if double loss is False)
    # 3. the loss dict has two entries, "loss_dict_random", "loss_dict_st"
    #    if double loss is True
    if "loss_type" in training_dict:
        loss_dict = {"loss_type": training_dict["loss_type"]}
        for additional_key in ["loss_mask_other_samples", "pred2unit", "tau"]:
            if additional_key in training_dict:
                loss_dict[additional_key] = training_dict[additional_key]
        training_dict["loss_dict"] = loss_dict

    if training_dict["double_loss"]:
        assert "target_exit_after" in training_dict
        assert isinstance(training_dict["target_exit_after"], int)
        assert "token_exit_cfg" in training_dict
        assert isinstance(training_dict["token_exit_cfg"], dict)
        assert (
            training_dict["random_masking"] == "half"
        ), "double_loss only possible with random_masking == half"
        if "loss_dict" not in training_dict:
            assert "loss_dict_st" in training_dict
            assert "loss_dict_random" in training_dict
    else:
        if "target_exit_after" in training_dict.keys():
            assert isinstance(training_dict["target_exit_after"], int)
            assert "token_exit_cfg" not in training_dict.keys()
            training_dict["token_exit_cfg"] = None
        elif "token_exit_cfg" in training_dict.keys():
            assert isinstance(training_dict["token_exit_cfg"], dict)
            assert "target_exit_after" not in training_dict.keys()
            training_dict["target_exit_after"] = None

    if training_dict["double_predictors"]:
        if not training_dict["double_loss"]:
            raise ValueError("double_predictors only possible with double loss")

    if isinstance(training_dict["warmup_epochs"], float):
        training_dict["warmup_epochs"] = int(
            training_dict["warmup_epochs"] * training_dict["num_epochs"]
        )
    assert isinstance(training_dict["warmup_epochs"], int)
    assert training_dict["num_epochs"] > training_dict["warmup_epochs"]
    assert training_dict["normalization"] in ["std", "scaling"]
    assert training_dict["random_masking"] in ["half", "full", "none"]

    assert len(training_dict["masking_probabilities"]) == len(
        MASKING_MODES
    ), f"Expected {len(MASKING_MODES)}, got {len(training_dict['masking_probabilities'])}"

    for combination in training_dict["shape_time_combinations"]:
        assert "timesteps" in combination.keys()
        assert "size" in combination.keys()
        assert combination["timesteps"] >= 3

    expected_encoder_decoder_keys_type = {
        "embedding_size": int,
        "depth": int,
        "mlp_ratio": int,
        "num_heads": int,
        "max_sequence_length": int,
    }

    expected_encoder_only_keys_type = {"freeze_projections": bool, "drop_path": float}
    expected_decoder_only_keys_type = {"learnable_channel_embeddings": bool}

    model_dict = config["model"]
    for model in ["encoder", "decoder"]:
        assert model in model_dict
        for key, val in expected_encoder_decoder_keys_type.items():
            assert key in model_dict[model], f"Expected {key} in {model} dict"
            assert isinstance(model_dict[model][key], val)
        if model == "encoder":
            for key, val in expected_encoder_only_keys_type.items():
                assert key in model_dict[model], f"Expected {key} in {model} dict"
                assert isinstance(model_dict[model][key], val)
        elif model == "decoder":
            for key, val in expected_decoder_only_keys_type.items():
                assert key in model_dict[model], f"Expected {key} in {model} dict"
                assert isinstance(model_dict[model][key], val)

    config["model"]["encoder"]["max_patch_size"] = max(config["training"]["patch_sizes"])
    config["model"]["decoder"]["max_patch_size"] = max(config["training"]["patch_sizes"])
    config["model"]["decoder"]["encoder_embedding_size"] = config["model"]["encoder"][
        "embedding_size"
    ]
    config["model"]["decoder"]["decoder_embedding_size"] = config["model"]["decoder"].pop(
        "embedding_size"
    )

    if ("loss_dict" in config["training"]) and (config["training"]["loss_dict"] == "MAE"):
        max_patch_size = max(config["training"]["patch_sizes"])
        max_group_length = max(
            [
                max([len(v) for _, v in SPACE_TIME_BANDS_GROUPS_IDX.items()]),
                max([len(v) for _, v in TIME_BAND_GROUPS_IDX.items()]),
                max([len(v) for _, v in SPACE_BAND_GROUPS_IDX.items()]),
                max([len(v) for _, v in STATIC_BAND_GROUPS_IDX.items()]),
            ]
        )
        config["model"]["decoder"]["output_embedding_size"] = (
            max_patch_size**2
        ) * max_group_length

    return config


def load_check_config(name: str) -> Dict:
    with (config_dir / "models" / name).open("r") as f:
        config = json.load(f)
    config = check_config(config)

    return config


def timestamp_dirname(suffix: Optional[str] = None) -> str:
    ts = datetime.now(dateutil.tz.tzlocal()).strftime("%Y_%m_%d_%H_%M_%S_%f")
    return f"{ts}_{suffix}" if suffix is not None else ts


def is_bf16_available():
    # https://github.com/huggingface/transformers/blob/d91841315aab55cf1347f4eb59332858525fad0f/src/transformers/utils/import_utils.py#L275
    # https://github.com/pytorch/pytorch/blob/2289a12f21c54da93bf5d696e3f9aea83dd9c10d/torch/testing/_internal/common_cuda.py#L51
    # to succeed:
    # 1. the hardware needs to support bf16 (arch >= Ampere)
    # 2. torch >= 1.10 (1.9 should be enough for AMP API has changed in 1.10, so using 1.10 as minimal)
    # 3. CUDA >= 11
    # 4. torch.autocast exists
    # XXX: one problem here is that it may give invalid results on mixed gpus setup, so it's
    # really only correct for the 0th gpu (or currently set default device if different from 0)

    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8:
        return False
    if int(torch.version.cuda.split(".")[0]) < 11:
        return False
    if not hasattr(torch, "autocast"):
        return False

    return True
