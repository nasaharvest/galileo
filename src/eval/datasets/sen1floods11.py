import json
from pathlib import Path

import pandas as pd
import rioxarray
import torch
from einops import rearrange
from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils import data_dir

from ..preprocess import normalize_bands

flood_folder = data_dir / "sen1floods"


class Sen1Floods11Processor:
    input_hw = 512
    output_tile_size = 64

    s1_bands = ("VV", "VH")
    s2_bands = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12")

    def __init__(self, folder: Path, split_path: Path):
        split_labelnames = pd.read_csv(split_path, header=None)[1].tolist()
        all_labels = list(folder.glob("LabelHand/*.tif"))
        split_labels = []
        for label in all_labels:
            if label.name in split_labelnames:
                split_labels.append(label)
        self.all_labels = split_labels

    def __len__(self):
        return len(self.all_labels)

    @classmethod
    def split_and_filter_tensors(cls, s1, s2, labels):
        """
        Split image and label tensors into 9 tiles and filter based on label content.
        Args:
        image_tensor (torch.Tensor): Input tensor of shape (13, 240, 240)
        label_tensor (torch.Tensor): Label tensor of shape (240, 240)
        Returns:
        list of tuples: Each tuple contains (image_tile, label_tile)
        """
        assert s1.shape == (
            len(cls.s1_bands),
            cls.input_hw,
            cls.input_hw,
        ), (
            f"s1 tensor must be of shape ({len(cls.s1_bands)}, {cls.input_hw}, {cls.input_hw}), "
            f"got {s1.shape}"
        )
        assert s2.shape == (
            len(cls.s2_bands),
            cls.input_hw,
            cls.input_hw,
        ), f"s2 tensor must be of shape ({len(cls.s2_bands)}, {cls.input_hw}, {cls.input_hw})"
        assert labels.shape == (
            1,
            cls.input_hw,
            cls.input_hw,
        ), f"labels tensor must be of shape (1, {cls.input_hw}, {cls.input_hw})"

        tile_size = cls.output_tile_size
        s1_list, s2_list, labels_list = [], [], []

        num_tiles_per_dim = cls.input_hw // cls.output_tile_size
        for i in range(num_tiles_per_dim):
            for j in range(num_tiles_per_dim):
                # Extract image tile
                s1_tile = s1[
                    :, i * tile_size : (i + 1) * tile_size, j * tile_size : (j + 1) * tile_size
                ]

                s2_tile = s2[
                    :, i * tile_size : (i + 1) * tile_size, j * tile_size : (j + 1) * tile_size
                ]

                # Extract corresponding label tile
                label_tile = labels[
                    :, i * tile_size : (i + 1) * tile_size, j * tile_size : (j + 1) * tile_size
                ]

                # Check if label tile has any non-zero values
                if torch.any(label_tile > 0):
                    s1_list.append(s1_tile)
                    s2_list.append(s2_tile)
                    labels_list.append(label_tile)

        return s1_list, s2_list, labels_list

    @staticmethod
    def label_to(label: Path, to: str = "s1"):
        sen_root = label.parents[1]
        location, tile_id, _ = label.stem.split("_")
        if to == "s1":
            return sen_root / f"s1/{location}_{tile_id}_S1Hand.tif"
        elif to == "s2":
            return sen_root / f"s2/{location}_{tile_id}_S2Hand.tif"
        else:
            raise ValueError(f"Expected `to` to be s1 or s2, got {to}")

    def __getitem__(self, idx: int):
        labels_path = self.all_labels[idx]

        with rioxarray.open_rasterio(labels_path) as ds:  # type: ignore
            labels = torch.from_numpy(ds.values)  # type: ignore

        with rioxarray.open_rasterio(self.label_to(labels_path, "s1")) as ds:  # type: ignore
            s1 = torch.from_numpy(ds.values)  # type: ignore

        with rioxarray.open_rasterio(self.label_to(labels_path, "s2")) as ds:  # type: ignore
            s2 = torch.from_numpy(ds.values)  # type: ignore
        return self.split_and_filter_tensors(s1, s2, labels)


def get_sen1floods11(split_name: str = "flood_bolivia_data.csv"):
    split_path = flood_folder / split_name
    dataset = Sen1Floods11Processor(folder=flood_folder, split_path=split_path)
    all_s1, all_s2, all_labels = [], [], []
    for i in tqdm(range(len(dataset))):
        b = dataset[i]
        all_s1 += b[0]
        all_s2 += b[1]
        all_labels += b[2]

    save_path = flood_folder / f"{split_path.stem}.pt"
    torch.save(
        obj={
            "s1": torch.stack(all_s1),
            "labels": torch.stack(all_labels),
            "s2": torch.stack(all_s2),
        },
        f=save_path,
    )


def remove_nan(s1, target):
    # s1 is shape (N, H, W, C)
    # target is shape (N, H, W)

    new_s1, new_target = [], []
    for i in range(s1.shape[0]):
        if torch.any(torch.isnan(s1[i])) or torch.any(torch.isinf(s1[i])):
            continue
        new_s1.append(s1[i])
        new_target.append(target[i])

    return torch.stack(new_s1), torch.stack(new_target)


class Sen1Floods11Dataset(Dataset):
    def __init__(
        self,
        path_to_splits: Path,
        split: str,
        norm_operation,
        augmentation,
        partition,
        mode: str = "s1",  # not sure if we would ever want s2?
    ):
        with (Path(__file__).parents[0] / Path("configs") / Path("sen1floods11.json")).open(
            "r"
        ) as f:
            config = json.load(f)

        assert split in ["train", "val", "valid", "test", "bolivia"]
        if split == "val":
            split = "valid"

        self.band_info = config["band_info"]["s1"]
        self.split = split
        self.augmentation = augmentation
        self.norm_operation = norm_operation

        torch_obj = torch.load(path_to_splits / f"flood_{split}_data.pt")
        self.s1 = torch_obj["s1"]  # (N, 2, 64, 64)
        self.s1 = rearrange(self.s1, "n c h w -> n h w c")
        # print(f"Before removing nans, we have {self.s1.shape[0]} tiles")
        self.labels = torch_obj["labels"]
        self.s1, self.labels = remove_nan(
            self.s1, self.labels
        )  # should we remove the tile or impute the pixel?
        # print(f"After removing nans, we have {self.s1.shape[0]} tiles")

        if (partition != "default") and (split == "train"):
            with open(path_to_splits / f"{partition}_partition.json", "r") as json_file:
                subset_indices = json.load(json_file)

            self.s1 = self.s1[subset_indices]
            self.labels = self.labels[subset_indices]

        if mode != "s1":
            raise ValueError(f"Modes other than s1 not yet supported, got {mode}")

    def __len__(self):
        return self.s1.shape[0]

    def __getitem__(self, idx):
        image = self.s1[idx]
        label = self.labels[idx][0]
        image = torch.tensor(normalize_bands(image.numpy(), self.norm_operation, self.band_info))
        image, label = self.augmentation.apply(image, label, "seg")
        return {"s1": image, "target": label.long()}
