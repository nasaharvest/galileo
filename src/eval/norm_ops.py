from typing import List, Tuple

from .baseline_models import BASELINE_MODELS


def norm_type_from_model_name(model_name: str) -> Tuple[str, int]:
    standardizing_models = [
        "dofa_large",
        "dofa_base",
        "mmearth_atto",
        "presto",
        "anysat",
        "prithvi",
    ]
    for m in standardizing_models:
        assert m in BASELINE_MODELS, f"{m} not in BASELINE_MODELS"
    if model_name in standardizing_models:
        norm_type = "standardize"
        std_dividor = 2
    elif model_name in BASELINE_MODELS:
        norm_type = "norm_yes_clip_int"
        std_dividor = 1
    else:
        norm_type = "norm_no_clip"
        std_dividor = 1
    return norm_type, std_dividor


def get_all_norm_strats(model_name, s1_or_s2: str = "s2") -> List:
    std_multiplier_range = list(range(14, 27, 2))

    norm_type, std_dividor = norm_type_from_model_name(model_name)
    if s1_or_s2 == "s2":
        datasets = ["dataset", "SATMAE", "S2A", "S2C", "OURS", "presto_s2"]
    else:
        if s1_or_s2 != "s1":
            raise ValueError(f"Expected s1_or_s2 to be 's1' or 's2', got {s1_or_s2}")
        datasets = ["dataset", "S1", "OURS_S1", "presto_s1"]

    if model_name == "prithvi":
        # the Prithvi norm bands only cover a subset of bands,
        # so they are not applicable for other models
        datasets.append("prithvi2")

    # std_multiplier = 1.4, 1.6, ... 2.6
    norm_stats = [
        {"stats": s, "type": norm_type, "std_multiplier": m / (10 * std_dividor)}
        for s in datasets
        for m in std_multiplier_range
    ]

    if s1_or_s2 == "s2":
        norm_stats.append({"type": "satlas"})
    return norm_stats
