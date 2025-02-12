import logging
import random
from copy import deepcopy
from typing import Dict, Optional

import torch
import torchvision.transforms.v2.functional as TVF
from einops import rearrange
from torch.utils.data import DataLoader

logger = logging.getLogger("__main__")


def get_embeddings(data_loader, model, device, subsample_tokens: Optional[float] = None):
    embeddings = []
    labels = []
    if subsample_tokens:
        print(f"Subsampling tokens with ratio {subsample_tokens}")

    model = model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch_labels = batch.pop("target")
            if "s1" in batch:
                batch["s1"] = batch["s1"].to(device).to(torch.bfloat16)
            if "s2" in batch:
                batch["s2"] = batch["s2"].to(device).to(torch.bfloat16)
            if "months" in batch:
                batch["months"] = batch["months"].to(device).long()

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                batch_embeddings = model(**batch)  # (bsz, dim) or (bsz, tokens, dim)

            if subsample_tokens is not None:
                if len(batch_embeddings.shape) < 3:
                    raise ValueError("subsample tokens only works for segmentation tasks")
                num_tokens_per_instance = batch_embeddings.shape[1]
                num_instances_to_keep = int(num_tokens_per_instance * subsample_tokens)
                sampled_indices = torch.randperm(num_tokens_per_instance)[:num_instances_to_keep]
                batch_embeddings = batch_embeddings[:, sampled_indices]

                tokens_per_dim = int(num_tokens_per_instance**0.5)
                pixels_per_token_dim = int(batch_labels.shape[1] / tokens_per_dim)

                batch_labels_per_token = rearrange(
                    batch_labels,
                    "b (t_h p_h) (t_w p_w) -> b (t_h t_w) (p_h p_w)",
                    t_h=tokens_per_dim,
                    t_w=tokens_per_dim,
                    p_h=pixels_per_token_dim,
                    p_w=pixels_per_token_dim,
                )
                batch_labels = batch_labels_per_token[:, sampled_indices]

            embeddings.append(batch_embeddings.to(torch.bfloat16).cpu())
            labels.append(batch_labels)

    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)


class DownstreamAugs(object):
    """
    For now, lets have no parameters
    Choose 1 of 8 transformations and apply it to space_x and the segmentation map (if needed)
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.transformations = [
            self.no_transform,  # No transformation
            self.rotate_90,  # 90-degree rotation
            self.rotate_180,  # 180-degree rotation
            self.rotate_270,  # 270-degree rotation
            self.hflip,  # Horizontal flip
            self.vflip,  # Vertical flip
            self.hflip_rotate_90,  # Horizontal flip of 90-degree rotated image
            self.vflip_rotate_90,  # Vertical flip of 90-degree rotated image
        ]

    def no_transform(self, x):
        return x

    def rotate_90(self, x):
        return TVF.rotate(x, 90)

    def rotate_180(self, x):
        return TVF.rotate(x, 180)

    def rotate_270(self, x):
        return TVF.rotate(x, 270)

    def hflip(self, x):
        return TVF.hflip(x)

    def vflip(self, x):
        return TVF.vflip(x)

    def hflip_rotate_90(self, x):
        return TVF.hflip(TVF.rotate(x, 90))

    def vflip_rotate_90(self, x):
        return TVF.vflip(TVF.rotate(x, 90))

    def apply(self, image, target, task_type):
        assert task_type in ["cls", "seg"]
        # image is (H, W, C)
        # target is either (1,) for classification or (H, W) for segmentation
        if not self.enabled:
            return image, target

        # choose 1 of 8 possible augmentations
        transformation = random.choice(self.transformations)

        # transform image and rearrange
        image = rearrange(image, "h w c -> c h w")
        image = transformation(image)
        image = rearrange(image, "c h w -> h w c")

        if task_type == "cls":
            return image, target
        else:
            # transform segmentation map and rearrange
            assert target.shape[-1] == image.shape[-1]
            assert target.shape[-2] == image.shape[-2]
            target = rearrange(target, "h w -> 1 h w")
            target = transformation(target)
            target = rearrange(target, "1 h w -> h w")
            return image, target


def get_loaders(
    benchmark,
    config,
    model_name,
    batch_size,
    num_workers,
    eval_type,
    train_partition: Optional[str] = None,
    valtest_partition: Optional[str] = None,
    norm_ops: Optional[Dict] = None,
):
    use_train_augs = True if eval_type == "FT" else False

    dataclass_kwargs = deepcopy(benchmark["kwargs"])
    if norm_ops is None:
        dataclass_kwargs["norm_operation"] = config["models"][model_name]
    else:
        dataclass_kwargs["norm_operation"] = norm_ops

    train_kwargs = deepcopy(dataclass_kwargs)
    valtest_kwargs = deepcopy(dataclass_kwargs)
    if train_partition is not None:
        train_kwargs["partition"] = train_partition
        if valtest_partition is None:
            valtest_partition = "default"
        valtest_kwargs["partition"] = valtest_partition
    elif valtest_partition:
        raise ValueError("Shouldn't have not None val_partition but None train_partiton")

    return {
        "train": DataLoader(
            benchmark["class"](
                **train_kwargs,
                split="train",
                augmentation=DownstreamAugs(use_train_augs),
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "valid": DataLoader(
            benchmark["class"](
                **valtest_kwargs,
                split="valid",
                augmentation=DownstreamAugs(False),
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            benchmark["class"](
                **valtest_kwargs,
                split="test",
                augmentation=DownstreamAugs(False),
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
