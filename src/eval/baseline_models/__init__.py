from pathlib import Path
from typing import Dict

from .anysat import AnySatWrapper
from .croma import CROMAWrapper
from .decur import DeCurWrapper
from .dofa import DOFAWrapper
from .mmearth import MMEarthWrapper  # type: ignore
from .presto import PrestoWrapper, UnWrappedPresto
from .prithvi import PrithviWrapper  # type: ignore
from .satlas import SatlasWrapper
from .satmae import SatMAEWrapper
from .softcon import SoftConWrapper

__all__ = [
    "CROMAWrapper",
    "DOFAWrapper",
    "MMEarthWrapper",
    "SatlasWrapper",
    "SatMAEWrapper",
    "SoftConWrapper",
    "DeCurWrapper",
    "PrestoWrapper",
    "AnySatWrapper",
    "UnWrappedPresto",
    "PrithviWrapper",
]


def construct_model_dict(weights_path: Path, s1_or_s2: str) -> Dict:
    model_dict = {
        "mmearth_atto": {
            "model": MMEarthWrapper,
            "args": {
                "weights_path": weights_path,
                "size": "atto",
            },
        },
        "satmae_pp": {
            "model": SatMAEWrapper,
            "args": {
                "pretrained_path": weights_path / "satmae_pp.pth",
                "size": "large",
            },
        },
        "satlas_tiny": {
            "model": SatlasWrapper,
            "args": {
                "weights_path": weights_path,
                "size": "tiny",
            },
        },
        "croma_base": {
            "model": CROMAWrapper,
            "args": {
                "weights_path": weights_path,
                "size": "base",
                "modality": "SAR" if s1_or_s2 == "s1" else "optical",
            },
        },
        "softcon_small": {
            "model": SoftConWrapper,
            "args": {
                "weights_path": weights_path,
                "size": "small",
                "modality": "SAR" if s1_or_s2 == "s1" else "optical",
            },
        },
        "satmae_base": {
            "model": SatMAEWrapper,
            "args": {
                "pretrained_path": weights_path / "pretrain-vit-base-e199.pth",
                "size": "base",
            },
        },
        "dofa_base": {
            "model": DOFAWrapper,
            "args": {
                "weights_path": weights_path,
                "size": "base",
            },
        },
        "satlas_base": {
            "model": SatlasWrapper,
            "args": {
                "weights_path": weights_path,
                "size": "base",
            },
        },
        "croma_large": {
            "model": CROMAWrapper,
            "args": {
                "weights_path": weights_path,
                "size": "large",
                "modality": "SAR" if s1_or_s2 == "s1" else "optical",
            },
        },
        "softcon_base": {
            "model": SoftConWrapper,
            "args": {
                "weights_path": weights_path,
                "size": "base",
                "modality": "SAR" if s1_or_s2 == "s1" else "optical",
            },
        },
        "satmae_large": {
            "model": SatMAEWrapper,
            "args": {
                "pretrained_path": weights_path / "pretrain-vit-large-e199.pth",
                "size": "large",
            },
        },
        "dofa_large": {
            "model": DOFAWrapper,
            "args": {
                "weights_path": weights_path,
                "size": "large",
            },
        },
        "decur": {
            "model": DeCurWrapper,
            "args": {
                "weights_path": weights_path,
                "modality": "SAR" if s1_or_s2 == "s1" else "optical",
            },
        },
        "presto": {
            "model": PrestoWrapper if s1_or_s2 in ["s1", "s2"] else UnWrappedPresto,
            "args": {},
        },
        "anysat": {"model": AnySatWrapper, "args": {}},
        "prithvi": {"model": PrithviWrapper, "args": {"weights_path": weights_path}},
    }
    return model_dict


def get_model_config(model_name: str, weights_path: Path, s1_or_s2: str):
    return construct_model_dict(weights_path, s1_or_s2)[model_name]


def get_all_model_names():
    model_dict = construct_model_dict(Path("."), "s2")  # placeholder Path and pooling
    return list(model_dict.keys())


BASELINE_MODELS = get_all_model_names()
