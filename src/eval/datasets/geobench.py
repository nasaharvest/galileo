import json
from pathlib import Path
from typing import Optional

import geobench
import numpy as np
import torch.multiprocessing
from sklearn.utils import shuffle
from torch.utils.data import Dataset

from src.utils import DEFAULT_SEED

from ..preprocess import impute_bands, impute_normalization_stats, normalize_bands

torch.multiprocessing.set_sharing_strategy("file_system")


class GeobenchDataset(Dataset):
    """
    Class implementation inspired by: https://github.com/vishalned/MMEarth-train/tree/main
    """

    def __init__(
        self,
        dataset_config_file: str,
        split: str,
        norm_operation,
        augmentation,
        partition,
        manual_subsetting: Optional[float] = None,
    ):
        with (Path(__file__).parents[0] / Path("configs") / Path(dataset_config_file)).open(
            "r"
        ) as f:
            config = json.load(f)

        assert split in ["train", "valid", "test"]

        self.split = split
        self.config = config
        self.norm_operation = norm_operation
        self.augmentation = augmentation
        self.partition = partition

        if config["task_type"] == "cls":
            self.tiles_per_img = 1
        elif config["task_type"] == "seg":
            assert self.config["dataset_name"] in ["m-SA-crop-type", "m-cashew-plant"]
            # for cashew plant and SA crop type
            # images are 256x256, we want 64x64
            self.tiles_per_img = 16
        else:
            raise ValueError(f"task_type must be cls or seg, not {config['task_type']}")

        for task in geobench.task_iterator(benchmark_name=self.config["benchmark_name"]):
            if task.dataset_name == self.config["dataset_name"]:
                break

        self.dataset = task.get_dataset(split=self.split, partition_name=self.partition)
        print(
            f"In dataset length for split {split} and partition {partition}: length = {len(self.dataset)}"
        )

        original_band_names = [
            self.dataset[0].bands[i].band_info.name for i in range(len(self.dataset[0].bands))
        ]

        self.band_names = list(self.config["band_info"].keys())
        self.band_indices = [original_band_names.index(band_name) for band_name in self.band_names]
        self.band_info = impute_normalization_stats(
            self.config["band_info"], self.config["imputes"]
        )
        self.manual_subsetting = manual_subsetting

        if self.manual_subsetting is not None:
            num_vals_to_keep = int(self.manual_subsetting * len(self.dataset) * self.tiles_per_img)
            active_indices = list(range(int(len(self.dataset) * self.tiles_per_img)))
            self.active_indices = shuffle(
                active_indices, random_state=DEFAULT_SEED, n_samples=num_vals_to_keep
            )
        else:
            self.active_indices = list(range(int(len(self.dataset) * self.tiles_per_img)))

    def __getitem__(self, idx):
        dataset_idx = self.active_indices[idx]
        img_idx = dataset_idx // self.tiles_per_img  # thanks Gabi / Marlena
        label = self.dataset[img_idx].label

        x = []
        for band_idx in self.band_indices:
            x.append(self.dataset[img_idx].bands[band_idx].data)

        x = impute_bands(x, self.band_names, self.config["imputes"])

        x = np.stack(x, axis=2)  # (h, w, 13)
        assert x.shape[-1] == 13, f"All datasets must have 13 channels, not {x.shape[-1]}"
        if self.config["dataset_name"] == "m-so2sat":
            x = x * 10_000

        x = torch.tensor(normalize_bands(x, self.norm_operation, self.band_info))

        # check if label is an object or a number
        if not (isinstance(label, int) or isinstance(label, list)):
            label = label.data
            # label is a memoryview object, convert it to a list, and then to a numpy array
            label = np.array(list(label))

        target = torch.tensor(label, dtype=torch.long)

        if self.tiles_per_img == 16:
            # thanks Gabi / Marlena
            # for cashew plant and SA crop type
            subtiles_per_dim = 4
            h = 256
            assert h % subtiles_per_dim == 0
            pixels_per_dim = h // subtiles_per_dim
            subtile_idx = idx % self.tiles_per_img

            row_idx = subtile_idx // subtiles_per_dim
            col_idx = subtile_idx % subtiles_per_dim

            x = x[
                row_idx * pixels_per_dim : (row_idx + 1) * pixels_per_dim,
                col_idx * pixels_per_dim : (col_idx + 1) * pixels_per_dim,
                :,
            ]

            target = target[
                row_idx * pixels_per_dim : (row_idx + 1) * pixels_per_dim,
                col_idx * pixels_per_dim : (col_idx + 1) * pixels_per_dim,
            ]

        x, target = self.augmentation.apply(x, target, self.config["task_type"])
        return {"s2": x, "target": target}

    def __len__(self):
        return int(len(self.active_indices))
