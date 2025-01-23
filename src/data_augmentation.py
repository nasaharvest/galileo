import random
from typing import Dict, Tuple

import torch
import torchvision.transforms.v2.functional as F
from einops import rearrange


class FlipAndRotateSpace(object):
    """
    For now, lets have no parameters
    Choose 1 of 8 transformations and apply it to space_time_x and space_x
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
        return F.rotate(x, 90)

    def rotate_180(self, x):
        return F.rotate(x, 180)

    def rotate_270(self, x):
        return F.rotate(x, 270)

    def hflip(self, x):
        return F.hflip(x)

    def vflip(self, x):
        return F.vflip(x)

    def hflip_rotate_90(self, x):
        return F.hflip(F.rotate(x, 90))

    def vflip_rotate_90(self, x):
        return F.vflip(F.rotate(x, 90))

    def apply(
        self,
        space_time_x: torch.Tensor,
        space_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.enabled:
            return space_time_x, space_x

        space_time_x = rearrange(
            space_time_x.float(), "b h w t c -> b t c h w"
        )  # rearrange for transforms
        space_x = rearrange(space_x.float(), "b h w c -> b c h w")  # rearrange for transforms

        transformation = random.choice(self.transformations)

        space_time_x = rearrange(
            transformation(space_time_x), "b t c h w -> b h w t c"
        )  # rearrange back
        space_x = rearrange(transformation(space_x), "b c h w -> b h w c")  # rearrange back

        return space_time_x.half(), space_x.half()


class Augmentation(object):
    def __init__(self, aug_config: Dict):
        self.flip_and_rotate = FlipAndRotateSpace(enabled=aug_config.get("flip+rotate", False))

    def apply(
        self,
        space_time_x: torch.Tensor,
        space_x: torch.Tensor,
        time_x: torch.Tensor,
        static_x: torch.Tensor,
        months: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        space_time_x, space_x = self.flip_and_rotate.apply(space_time_x, space_x)

        return space_time_x, space_x, time_x, static_x, months
