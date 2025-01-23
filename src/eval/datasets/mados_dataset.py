import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from ..preprocess import normalize_bands

torch.multiprocessing.set_sharing_strategy("file_system")


def split_and_filter_tensors(image_tensor, label_tensor):
    """
    Split image and label tensors into 9 tiles and filter based on label content.

    Args:
    image_tensor (torch.Tensor): Input tensor of shape (13, 240, 240)
    label_tensor (torch.Tensor): Label tensor of shape (240, 240)

    Returns:
    list of tuples: Each tuple contains (image_tile, label_tile)
    """
    assert image_tensor.shape == (13, 240, 240), "Image tensor must be of shape (13, 240, 240)"
    assert label_tensor.shape == (240, 240), "Label tensor must be of shape (240, 240)"

    tile_size = 80
    tiles = []
    labels = []

    for i in range(3):
        for j in range(3):
            # Extract image tile
            image_tile = image_tensor[
                :, i * tile_size : (i + 1) * tile_size, j * tile_size : (j + 1) * tile_size
            ]

            # Extract corresponding label tile
            label_tile = label_tensor[
                i * tile_size : (i + 1) * tile_size, j * tile_size : (j + 1) * tile_size
            ]

            # Check if label tile has any non-zero values
            if torch.any(label_tile > 0):
                tiles.append(image_tile)
                labels.append(label_tile)

    return tiles, labels


class PrepMADOSDataset(Dataset):
    def __init__(self, root_dir, split_file):
        self.root_dir = root_dir

        with open(os.path.join(root_dir, "splits", split_file), "r") as f:
            self.scene_list = [line.strip() for line in f]

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, idx):
        scene_name = self.scene_list[idx]
        scene_num_1 = scene_name.split("_")[1]
        scene_num_2 = scene_name.split("_")[2]

        # Load all bands
        B1 = self._load_band(scene_num_1, scene_num_2, [442, 443], 60)
        B2 = self._load_band(scene_num_1, scene_num_2, [492], 10)
        B3 = self._load_band(scene_num_1, scene_num_2, [559, 560], 10)
        B4 = self._load_band(scene_num_1, scene_num_2, [665], 10)
        B5 = self._load_band(scene_num_1, scene_num_2, [704], 20)
        B7 = self._load_band(scene_num_1, scene_num_2, [780, 783], 20)
        B8 = self._load_band(scene_num_1, scene_num_2, [833], 10)
        B8A = self._load_band(scene_num_1, scene_num_2, [864, 865], 20)
        B11 = self._load_band(scene_num_1, scene_num_2, [1610, 1614], 20)
        B12 = self._load_band(scene_num_1, scene_num_2, [2186, 2202], 20)

        B1 = self._resize(B1)
        B5 = self._resize(B5)
        B7 = self._resize(B7)
        B8A = self._resize(B8A)
        B11 = self._resize(B11)
        B12 = self._resize(B12)

        # Interpolate missing bands
        B6 = (B5 + B7) / 2
        B9 = B8A
        B10 = (B8A + B11) / 2

        image = torch.cat(
            [B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12], axis=1
        ).squeeze(0)  # (13, 240, 240)
        mask = self._load_mask(scene_num_1, scene_num_2).squeeze(0).squeeze(0)  # (240, 240)
        images, masks = split_and_filter_tensors(image, mask)

        return images, masks

    def _load_band(self, scene_num_1, scene_num_2, bands, resolution):
        for band in bands:
            band_path = f"{self.root_dir}/Scene_{scene_num_1}/{resolution}/Scene_{scene_num_1}_L2R_rhorc_{band}_{scene_num_2}.tif"
            if os.path.exists(band_path):
                return (
                    torch.from_numpy(np.array(Image.open(band_path)))
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
        print(f"COULDNT FIND {scene_num_1, scene_num_2, bands, resolution}")

    def _resize(self, image):
        return F.interpolate(image, size=240, mode="bilinear", align_corners=False)

    def _load_mask(self, scene_num_1, scene_num_2):
        mask_path = (
            f"{self.root_dir}/Scene_{scene_num_1}/10/Scene_{scene_num_1}_L2R_cl_{scene_num_2}.tif"
        )
        return torch.from_numpy(np.array(Image.open(mask_path))).long().unsqueeze(0).unsqueeze(0)


def get_mados(save_path, root_dir="MADOS", split_file="test_X.txt"):
    dataset = PrepMADOSDataset(root_dir=root_dir, split_file=split_file)
    all_images = []
    all_masks = []
    for i in dataset:
        all_images += i[0]
        all_masks += i[1]

    split_images = torch.stack(all_images)  # shape (N, 13, 80, 80)
    split_masks = torch.stack(all_masks)  # shape (N, 80, 80)
    torch.save(obj={"images": split_images, "labels": split_masks}, f=save_path)


class MADOSDataset(Dataset):
    def __init__(self, path_to_splits: Path, split: str, norm_operation, augmentation, partition):
        with (Path(__file__).parents[0] / Path("configs") / Path("mados.json")).open("r") as f:
            config = json.load(f)

        # NOTE: I imputed bands for this dataset before saving the tensors, so no imputation is necessary
        assert split in ["train", "val", "valid", "test"]
        if split == "valid":
            split = "val"

        self.band_info = config["band_info"]
        self.split = split
        self.augmentation = augmentation
        self.norm_operation = norm_operation

        torch_obj = torch.load(path_to_splits / f"MADOS_{split}.pt")
        self.images = torch_obj["images"]
        self.labels = torch_obj["labels"]

        if (partition != "default") and (split == "train"):
            with open(path_to_splits / f"{partition}_partition.json", "r") as json_file:
                subset_indices = json.load(json_file)

            self.images = self.images[subset_indices]
            self.labels = self.labels[subset_indices]

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]  # (80, 80, 13)
        label = self.labels[idx]  # (80, 80)
        image = torch.tensor(normalize_bands(image.numpy(), self.norm_operation, self.band_info))
        image, label = self.augmentation.apply(image, label, "seg")
        return {"s2": image, "target": label}
