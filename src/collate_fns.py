from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import default_collate

from src.masking import (
    MASKING_MODES,
    MaskingFunctions,
    batch_subset_mask_galileo,
)


class CollateFnOutput(NamedTuple):
    s_t_x: torch.Tensor
    sp_x: torch.Tensor
    t_x: torch.Tensor
    st_x: torch.Tensor
    s_t_m: torch.Tensor
    sp_m: torch.Tensor
    t_m: torch.Tensor
    st_m: torch.Tensor
    months: torch.Tensor
    patch_size: float


def collated_batch_to_output(
    s_t_x: torch.Tensor,
    sp_x: torch.Tensor,
    t_x: torch.Tensor,
    st_x: torch.Tensor,
    months: torch.Tensor,
    patch_sizes,
    shape_time_combinations,
    encode_ratio,
    decode_ratio,
    masking_function: MaskingFunctions,
    augmentation_strategies=None,
    fixed_patch_size=None,
    fixed_space_time_combination=None,
    masking_probabilities=None,
    max_unmasking_channels=4,
    unmasking_channels_combo: str = "shapes",
    ignore_band_groups: Optional[List[str]] = None,
) -> CollateFnOutput:
    if fixed_patch_size is not None:
        patch_size = fixed_patch_size
    else:
        # randomly sample a patch size, and a corresponding image size
        patch_size = np.random.choice(patch_sizes)

    if fixed_space_time_combination is not None:
        space_time_combination = fixed_space_time_combination
    else:
        space_time_combination = np.random.choice(shape_time_combinations)
        spatial_patches_per_dim = space_time_combination["size"]
        if int(spatial_patches_per_dim * patch_size) > s_t_x.shape[1]:
            spatial_patches_per_dim = int(s_t_x.shape[1] / patch_size)

    timesteps = space_time_combination["timesteps"]

    image_size = patch_size * spatial_patches_per_dim
    if masking_probabilities is None:
        masking_probabilities = [1] * len(MASKING_MODES)

    # randomly select a masking strategy
    s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = batch_subset_mask_galileo(
        s_t_x,
        sp_x,
        t_x,
        st_x,
        months,
        encode_ratio=encode_ratio,
        patch_size=patch_size,
        image_size=image_size,
        num_timesteps=timesteps,
        decode_ratio=decode_ratio,
        augmentation_strategies=augmentation_strategies,
        masking_probabilities=masking_probabilities,
        masking_function=masking_function,
        max_unmasking_channels=max_unmasking_channels,
        unmasking_channels_combo=unmasking_channels_combo,
        ignore_band_groups=ignore_band_groups,
    )

    return CollateFnOutput(
        s_t_x,
        sp_x,
        t_x,
        st_x,
        s_t_m,
        sp_m,
        t_m,
        st_m,
        months,
        patch_size,
    )


@torch.no_grad()
def galileo_collate_fn(
    batch,
    patch_sizes,
    shape_time_combinations,
    st_encode_ratio=None,
    st_decode_ratio=None,
    random_encode_ratio=None,
    random_decode_ratio=None,
    augmentation_strategies=None,
    fixed_patch_size=None,
    fixed_space_time_combination=None,
    masking_probabilities=None,
    max_unmasking_channels=4,
    random_masking: str = "None",
    unmasking_channels_combo: str = "shapes",
    ignore_band_groups: Optional[List[str]] = None,
) -> Tuple[CollateFnOutput, CollateFnOutput, CollateFnOutput, CollateFnOutput]:
    s_t_x, sp_x, t_x, st_x, months = default_collate(batch)

    input_args = {
        "s_t_x": s_t_x,
        "sp_x": sp_x,
        "t_x": t_x,
        "st_x": st_x,
        "months": months,
        "patch_sizes": patch_sizes,
        "augmentation_strategies": augmentation_strategies,
        "fixed_patch_size": fixed_patch_size,
        "fixed_space_time_combination": fixed_space_time_combination,
        "masking_probabilities": masking_probabilities,
        "shape_time_combinations": shape_time_combinations,
        "max_unmasking_channels": max_unmasking_channels,
        "unmasking_channels_combo": unmasking_channels_combo,
        "ignore_band_groups": ignore_band_groups,
    }
    if random_masking == "none":
        if st_encode_ratio is None:
            raise ValueError("st_encode_ratio can't be None for random_masking='none'")
        if st_decode_ratio is None:
            raise ValueError("st_decode_ratio can't be None for random_masking='none'")
        input_args.update({"encode_ratio": st_encode_ratio, "decode_ratio": st_decode_ratio})
        return (
            collated_batch_to_output(
                **input_args,
                masking_function=MaskingFunctions.TIME,
            ),
            collated_batch_to_output(
                **input_args,
                masking_function=MaskingFunctions.SPACE,
            ),
            collated_batch_to_output(
                **input_args,
                masking_function=MaskingFunctions.TIME,
            ),
            collated_batch_to_output(
                **input_args,
                masking_function=MaskingFunctions.SPACE,
            ),
        )
    elif random_masking == "half":
        if st_encode_ratio is None:
            raise ValueError("st_encode_ratio can't be None for random_masking='half'")
        if st_decode_ratio is None:
            raise ValueError("st_decode_ratio can't be None for random_masking='half'")
        if random_encode_ratio is None:
            raise ValueError("random_encode_ratio can't be None for random_masking='half'")
        if random_decode_ratio is None:
            raise ValueError("random_decode_ratio can't be None for random_masking='half'")
        return (
            collated_batch_to_output(
                **input_args,
                encode_ratio=st_encode_ratio,
                decode_ratio=st_decode_ratio,
                masking_function=MaskingFunctions.TIME,
            ),
            collated_batch_to_output(
                **input_args,
                encode_ratio=st_encode_ratio,
                decode_ratio=st_decode_ratio,
                masking_function=MaskingFunctions.SPACE,
            ),
            collated_batch_to_output(
                **input_args,
                encode_ratio=random_encode_ratio,
                decode_ratio=random_decode_ratio,
                masking_function=MaskingFunctions.RANDOM,
            ),
            collated_batch_to_output(
                **input_args,
                encode_ratio=random_encode_ratio,
                decode_ratio=random_decode_ratio,
                masking_function=MaskingFunctions.RANDOM,
            ),
        )
    elif random_masking == "full":
        if random_encode_ratio is None:
            raise ValueError("random_encode_ratio can't be None for random_masking='full'")
        if random_decode_ratio is None:
            raise ValueError("random_decode_ratio can't be None for random_masking='full'")
        input_args.update(
            {"encode_ratio": random_encode_ratio, "decode_ratio": random_decode_ratio}
        )
        return (
            collated_batch_to_output(
                **input_args,
                masking_function=MaskingFunctions.RANDOM,
            ),
            collated_batch_to_output(
                **input_args,
                masking_function=MaskingFunctions.RANDOM,
            ),
            collated_batch_to_output(
                **input_args,
                masking_function=MaskingFunctions.RANDOM,
            ),
            collated_batch_to_output(
                **input_args,
                masking_function=MaskingFunctions.RANDOM,
            ),
        )
    else:
        raise ValueError(f"Expected random_masking to be (none, half full), got {random_masking}")
