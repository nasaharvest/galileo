import json
from pathlib import Path

import torch
import torch.multiprocessing
from einops import rearrange
from torch.utils.data import Dataset

from ..preprocess import normalize_bands

torch.multiprocessing.set_sharing_strategy("file_system")


class PASTISDataset(Dataset):
    def __init__(self, path_to_splits: Path, split: str, norm_operation, augmentation, partition):
        with (Path(__file__).parents[0] / Path("configs") / Path("pastis.json")).open("r") as f:
            config = json.load(f)

        # NOTE: I imputed bands for this dataset before saving the tensors, so no imputation is necessary
        assert split in ["train", "val", "valid", "test"]
        if split == "val":
            split = "valid"

        self.band_info = config["band_info"]
        self.split = split
        self.augmentation = augmentation
        self.norm_operation = norm_operation

        torch_obj = torch.load(path_to_splits / f"pastis_{split}.pt")
        self.images = torch_obj["images"]  # (N, 12, 13, 64, 64)
        self.months = torch_obj["months"] - 1  # subtract 1 for zero-indexing , shape (N, 12)
        self.labels = torch_obj["targets"]  # (N, 64, 64)

        if (partition != "default") and (split == "train"):
            with open(path_to_splits / f"{partition}_partition.json", "r") as json_file:
                subset_indices = json.load(json_file)

            self.images = self.images[subset_indices]
            self.months = self.months[subset_indices]
            self.labels = self.labels[subset_indices]

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        images = self.images[idx]  # (12, 13, 64, 64)
        months = self.months[idx]  # (12)
        labels = self.labels[idx]  # (64, 64)

        assert images.shape[0] == 12

        # normalize one timestep at a time
        normed_images = []
        for i in range(12):
            # sorry for the ugly code
            single_timestep_image = rearrange(images[i], "c h w -> h w c").numpy()
            normed_image = torch.tensor(
                normalize_bands(single_timestep_image, self.norm_operation, self.band_info)
            )
            normed_images.append(normed_image)

        normed_images = torch.stack(normed_images)  # (12, 64, 64, 13)
        normed_images = rearrange(normed_images, "t h w c -> h w t c")  # (64, 64, 12, 13)

        assert normed_images.shape[-2] == 12
        assert normed_images.shape[-1] == 13

        # important note: augmentation for timeseries is not supported
        # there is obviously a better way to do this but oh well, I'll remember it
        return {"s2": normed_images, "target": labels, "months": months}
