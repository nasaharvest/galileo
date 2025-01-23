import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from breizhcrops import BreizhCrops
from breizhcrops.datasets.breizhcrops import SELECTED_BANDS
from einops import repeat
from torch.utils.data import ConcatDataset, Dataset

from src.data.config import DATA_FOLDER

from ..preprocess import normalize_bands

LEVEL = "L1C"
DATAPATH = DATA_FOLDER / "breizhcrops"
OUTPUT_BAND_ORDER = [
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
INPUT_TO_OUTPUT_BAND_MAPPING = [SELECTED_BANDS[LEVEL].index(b) for b in OUTPUT_BAND_ORDER]


class BreizhCropsDataset(Dataset):
    def __init__(
        self,
        path_to_splits: Path,
        split: str,
        norm_operation,
        augmentation,
        partition,
        monthly_average: bool = True,
    ):
        """
        https://isprs-archives.copernicus.org/articles/XLIII-B2-2020/1545/2020/
        isprs-archives-XLIII-B2-2020-1545-2020.pdf

        We partitioned all acquired field parcels
        according to the NUTS-3 regions and suggest to subdivide the
        dataset into training (FRH01, FRH02), validation (FRH03), and
        evaluation (FRH04) subsets based on these spatially distinct
        regions.
        """
        kwargs = {
            "root": path_to_splits,
            "preload_ram": False,
            "level": LEVEL,
            "transform": raw_transform,
        }
        # belle-ille is small, so its useful for testing
        assert split in ["train", "valid", "test", "belle-ile"]
        if split == "train":
            self.ds: Dataset = ConcatDataset(
                [BreizhCrops(region=r, **kwargs) for r in ["frh01", "frh02"]]
            )
        elif split == "valid":
            self.ds = BreizhCrops(region="frh03", **kwargs)
        elif split == "test":
            self.ds = BreizhCrops(region="frh04", **kwargs)
        else:
            self.ds = BreizhCrops(region="belle-ile", **kwargs)
        self.monthly_average = monthly_average

        with (Path(__file__).parents[0] / Path("configs") / Path("breizhcrops.json")).open(
            "r"
        ) as f:
            config = json.load(f)
        self.band_info = config["band_info"]
        self.norm_operation = norm_operation
        self.augmentation = augmentation
        warnings.warn("Augmentations ignored for time series")
        if partition != "default":
            raise NotImplementedError(f"partition {partition} not implemented yet")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y_true, _ = self.ds[idx]
        if self.monthly_average:
            x = self.average_over_month(x)
        eo = normalize_bands(
            x[:, INPUT_TO_OUTPUT_BAND_MAPPING], self.norm_operation, self.band_info
        )
        eo = repeat(eo, "t d -> h w t d", h=1, w=1)
        months = x[:, SELECTED_BANDS[LEVEL].index("doa")]
        return {"s2": torch.tensor(eo), "months": torch.tensor(months), "target": y_true}

    @staticmethod
    def average_over_month(x: np.ndarray):
        x[:, SELECTED_BANDS[LEVEL].index("doa")] = np.array(
            [t.month - 1 for t in pd.to_datetime(x[:, SELECTED_BANDS[LEVEL].index("doa")])]
        )
        per_month = np.split(
            x, np.unique(x[:, SELECTED_BANDS[LEVEL].index("doa")], return_index=True)[1]
        )[1:]
        return np.array([per_month[idx].mean(axis=0) for idx in range(len(per_month))])


def raw_transform(input_timeseries):
    return input_timeseries
