import random
from collections import OrderedDict
from enum import Enum
from itertools import chain, combinations, product
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from einops import rearrange, repeat

from .data.dataset import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    TIME_BAND_GROUPS_IDX,
)
from .data_augmentation import Augmentation

# This is to allow a quick expansion of the mask from
# group-channel space into real-channel space
SPACE_TIME_BAND_EXPANSION = torch.tensor(
    [len(x) for x in SPACE_TIME_BANDS_GROUPS_IDX.values()]
).long()
SPACE_BAND_EXPANSION = torch.tensor([len(x) for x in SPACE_BAND_GROUPS_IDX.values()]).long()
TIME_BAND_EXPANSION = torch.tensor([len(x) for x in TIME_BAND_GROUPS_IDX.values()]).long()
STATIC_BAND_EXPANSION = torch.tensor([len(x) for x in STATIC_BAND_GROUPS_IDX.values()]).long()


STR2DICT = OrderedDict(
    {
        "space_time": SPACE_TIME_BANDS_GROUPS_IDX,
        "space": SPACE_BAND_GROUPS_IDX,
        "time": TIME_BAND_GROUPS_IDX,
        "static": STATIC_BAND_GROUPS_IDX,
    }
)

REVERSED_STR2DICT = {}
for key, values in STR2DICT.items():
    for v in values:
        REVERSED_STR2DICT[v] = key

SHAPES = list(STR2DICT.keys())
MASKING_MODES: List[Tuple[str, str]] = [
    ("space", "SRTM"),
    ("space", "DW"),
    ("space", "WC"),
    ("space_time", "NDVI"),
    ("space_time", "S1"),
    ("space_time", "S2_RGB"),
    ("space_time", "S2_SWIR"),
    ("space_time", "S2_Red_Edge"),
    ("space_time", "S2_NIR_10m"),
    ("space_time", "S2_NIR_20m"),
    ("time", "ERA5"),
    ("time", "TC"),
    ("time", "VIIRS"),
    ("static", "LS"),
    ("static", "location"),
    ("static", "DW_static"),
    ("static", "WC_static"),
]

UNMASKING_CHANNEL_GROUPS: List[Tuple[str, str]] = MASKING_MODES

MAX_MASKING_STRATEGIES = 6
NUM_RECON_OBJS = 2


def generate_combinations():
    all_combinations = []
    for r in range(1, 5):
        shape_combos = combinations(SHAPES, r)

        for shape_combo in shape_combos:
            mode_lists = [STR2DICT[shape] for shape in shape_combo]
            mode_combos = product(*mode_lists)
            for mode_combo in mode_combos:
                all_combinations.append([(REVERSED_STR2DICT[x], x) for x in mode_combo])

    return all_combinations


def powerset(iterable):
    "powerset([1,2,3]) → (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return_list = list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))
    return [item for item in return_list if len(item) > 0]


# Generate all 639 combinations
ALL_MASKING_COMBINATIONS_SHAPES = generate_combinations()
ALL_MASKING_COMBINATIONS = powerset(MASKING_MODES)


class MaskingFunctions(Enum):
    SPACE = 1
    TIME = 0
    RANDOM = 2


def return_masked_unmasked_bands(
    bands: List[str], band_groups: Dict[str, List]
) -> Tuple[List[int], List[int]]:
    def in_masked_bands(x):
        for b in bands:
            if b in x:
                return True
        return False

    return [idx for idx, val in enumerate(band_groups.keys()) if in_masked_bands(val)], [
        idx for idx, val in enumerate(band_groups.keys()) if not in_masked_bands(val)
    ]


class MaskedOutput(NamedTuple):
    """
    A mask can take 3 values:
    0: seen by the encoder (i.e. makes the key and value tokens in the decoder)
    1: not seen by the encoder, and ignored by the decoder
    2: not seen by the encoder, and processed by the decoder (the decoder's query values)
    """

    space_time_x: torch.Tensor  # [B, H, W, T, len(SPACE_TIME_BANDS)]
    space_x: torch.Tensor  # [B, H, W, len(SPACE_BANDS)]
    time_x: torch.Tensor  # [B, T, len(TIME_BANDS)]
    static_x: torch.Tensor  # [B, len(STATIC_BANDS)]
    space_time_mask: torch.Tensor  # [B, H, W, T, len(SPACE_TIME_BANDS_GROUPS_IDX)]
    space_mask: torch.Tensor  # [B, H, W, len(SPACE_BAND_GROUPS_IDX)]
    time_mask: torch.Tensor  # [B, T, len(TIME_BAND_GROUPS_IDX)]
    static_mask: torch.Tensor  # [B, len(STATIC_BAND_GROUPS_IDX)]
    months: torch.Tensor  # [B, T]


def weighted_sample_without_replacement(population, weights, k, rng=random):
    if len(population) != len(weights):
        raise ValueError("Population and weights must have the same length")

    non_zero_indices = [i for i, w in enumerate(weights) if w > 0]
    if len(non_zero_indices) < k:
        raise ValueError("Not enough non-zero weights to sample k items")

    non_zero_population = [population[i] for i in non_zero_indices]
    non_zero_weights = [weights[i] for i in non_zero_indices]

    v = [rng.random() ** (1 / w) for w in non_zero_weights]
    order = sorted(range(len(non_zero_population)), key=lambda i: v[i])
    return [non_zero_population[i] for i in order[-k:]]


def check_modes_for_conflicts(
    modes: List[Tuple[str, str]], unmasking_modes: List[Tuple[str, str]]
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    output_modes: List[Tuple[str, str]] = []
    for mode in modes:
        assert mode in MASKING_MODES
        if mode in unmasking_modes:
            if len(unmasking_modes) == 1:
                # don't remove any more from the unmasking modes
                continue
            elif len(output_modes) == 0:
                output_modes.append(mode)
                unmasking_modes.remove(mode)
            else:
                # neither modes or unmasking_modes are bottlenecked;
                # randomly select which one to remove
                if random.random() <= 0.5:
                    output_modes.append(mode)
                    unmasking_modes.remove(mode)
        else:
            output_modes.append(mode)
    assert len(output_modes) >= 1
    assert len(unmasking_modes) >= 1
    return output_modes, unmasking_modes


def check_mode_and_return_channels(unmasking_modes: List[Tuple[str, str]]):
    outputs = []
    for data_type in STR2DICT.keys():
        relevant_bands = [x[1] for x in unmasking_modes if x[0] == data_type]
        if len(relevant_bands) > 0:
            outputs.append(return_masked_unmasked_bands(relevant_bands, STR2DICT[data_type]))
        else:
            outputs.append(([], []))
    return outputs


def round_school(x: float) -> float:
    i, f = divmod(x, 1)
    return int(i + ((f >= 0.5) if (x > 0) else (f > 0.5)))


def filter_unmasking_mode_candidates(
    unmasking_mode_candidates, ignore_band_groups: Optional[List[str]]
):
    def check_if_overlap(candidate, ignore_band_groups):
        for channel_group in candidate:
            if channel_group[1] in ignore_band_groups:
                return True
        return False

    if ignore_band_groups is None:
        return unmasking_mode_candidates
    output_candidates = []
    for candidate in unmasking_mode_candidates:
        if check_if_overlap(candidate, ignore_band_groups):
            continue
        else:
            output_candidates.append(candidate)
    return output_candidates


def batch_subset_mask_galileo(
    s_t_x: torch.Tensor,
    sp_x: torch.Tensor,
    t_x: torch.Tensor,
    st_x: torch.Tensor,
    months: torch.Tensor,
    encode_ratio: float,
    decode_ratio: float,
    patch_size: int,
    image_size: int,
    num_timesteps: int,
    augmentation_strategies: Optional[Dict],
    masking_probabilities: List[float],
    masking_function: MaskingFunctions,
    max_unmasking_channels: int,
    unmasking_channels_combo: str = "shapes",
    ignore_band_groups: Optional[List[str]] = None,
) -> MaskedOutput:
    assert len(masking_probabilities) == len(MASKING_MODES)

    if masking_function.value < 2:
        f: Callable = batch_mask_space if masking_function.value == 1 else batch_mask_time
        num_masking_modes = random.choice(list(range(2, MAX_MASKING_STRATEGIES + 1)))
        if ignore_band_groups is not None:
            masking_modes, kept_masking_probs = zip(
                *(
                    (m, p)
                    for m, p in zip(MASKING_MODES, masking_probabilities)
                    if m[1] not in ignore_band_groups
                )
            )
        else:
            masking_modes, kept_masking_probs = MASKING_MODES, masking_probabilities  # type: ignore
        masking_modes = weighted_sample_without_replacement(
            masking_modes, weights=kept_masking_probs, k=num_masking_modes
        )  # type: ignore

        # isolate the unmasking candidates which (1) have the right number of channels and
        # (b) don't intersect with the masking_modes
        if unmasking_channels_combo == "shapes":
            unmasking_mode_candidates = [
                x
                for x in filter_unmasking_mode_candidates(
                    ALL_MASKING_COMBINATIONS_SHAPES, ignore_band_groups
                )
                if ((len(x) <= max_unmasking_channels) and (len(set(x) & set(masking_modes)) == 0))
            ]
        elif unmasking_channels_combo == "all":
            unmasking_mode_candidates = [
                x
                for x in filter_unmasking_mode_candidates(
                    ALL_MASKING_COMBINATIONS, ignore_band_groups
                )
                if ((len(x) <= max_unmasking_channels) and (len(set(x) & set(masking_modes)) == 0))
            ]
        else:
            raise ValueError(
                "Expected unmasking_channels_combo to be "
                f"'shapes' or 'all', got {unmasking_channels_combo}"
            )
        unmasking_modes = random.choice(unmasking_mode_candidates)

        masking_modes, unmasking_modes = check_modes_for_conflicts(masking_modes, unmasking_modes)  # type: ignore
        masked_output = f(
            *subset_and_augment_batch_of_images(
                s_t_x,
                sp_x,
                t_x,
                st_x,
                months,
                size=image_size,
                num_timesteps=num_timesteps,
                augmentation_strategies=augmentation_strategies,
            ),
            encode_ratio=encode_ratio,
            decode_ratio=decode_ratio,
            patch_size=patch_size,
            mode=masking_modes,
            decoder_mode=unmasking_modes,
        )

    elif masking_function.value == 2:
        # 2 is random
        masked_output = batch_mask_random(
            *subset_and_augment_batch_of_images(
                s_t_x,
                sp_x,
                t_x,
                st_x,
                months,
                size=image_size,
                num_timesteps=num_timesteps,
                augmentation_strategies=augmentation_strategies,
            ),
            encode_ratio=encode_ratio,
            decode_ratio=decode_ratio,
            patch_size=patch_size,
            ignore_band_groups=ignore_band_groups,
        )

    else:
        raise AssertionError(f"Unexpected strategy {masking_function}")

    return masked_output


def subset_and_augment_batch_of_images(
    space_time_x: torch.Tensor,
    space_x: torch.Tensor,
    time_x: torch.Tensor,
    static_x: torch.Tensor,
    months: torch.Tensor,
    size: int,
    num_timesteps: int,
    augmentation_strategies: Optional[Dict],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert (space_time_x.shape[1] == space_x.shape[1]) & (
        space_time_x.shape[2] == space_x.shape[2]
    )
    assert time_x.shape[1] == space_time_x.shape[3] == months.shape[1]
    possible_h = space_time_x.shape[1] - size
    possible_w = space_time_x.shape[2] - size
    possible_t = space_time_x.shape[3] - num_timesteps
    assert (possible_h >= 0) & (possible_w >= 0) & (possible_t >= 0)

    if possible_h > 0:
        start_h = np.random.choice(possible_h)
    else:
        start_h = possible_h

    if possible_w > 0:
        start_w = np.random.choice(possible_w)
    else:
        start_w = possible_w

    if possible_t > 0:
        start_t = np.random.choice(possible_t)
    else:
        start_t = possible_t

    # do augmentations, if enabled
    space_time_x = space_time_x[
        :,
        start_h : start_h + size,
        start_w : start_w + size,
        start_t : start_t + num_timesteps,
    ]
    space_x = space_x[:, start_h : start_h + size, start_w : start_w + size]
    time_x = time_x[:, start_t : start_t + num_timesteps]
    months = months[:, start_t : start_t + num_timesteps]

    if augmentation_strategies is not None:
        return Augmentation(augmentation_strategies).apply(
            space_time_x, space_x, time_x, static_x, months
        )
    return space_time_x, space_x, time_x, static_x, months


def _random_mask_for_b(
    b: int, device: torch.device, encode_ratio: float, decode_ratio: float
) -> torch.Tensor:
    mask = torch.rand(b, device=device)
    mask[mask >= (1 - encode_ratio)] = 0
    mask[mask <= decode_ratio] = 2
    # all the rest is ignored by both the encoder and decoder
    mask[(mask != 0) | (mask != 2)] = 1
    return mask


def batch_mask_time(
    space_time_x: torch.Tensor,
    space_x: torch.Tensor,
    time_x: torch.Tensor,
    static_x: torch.Tensor,
    months: torch.Tensor,
    encode_ratio: float,
    decode_ratio: float,
    patch_size: int,
    decoder_mode: List[Tuple[str, str]],
    mode: List[Tuple[str, str]],
):
    """
    Masks out blocks of hxwx1xBAND_GROUPs.
    e.g. if mask_ratio=0.25, then 1/4 of the timeteps
    (and the static channel groups, with 1/4 probability) will be masked out

    Operates over batches where each item in the batch has independently masked timesteps
    """
    b, h, w, t, _ = space_time_x.shape
    assert t >= 3

    bands_to_encode = check_mode_and_return_channels(mode)
    bands_to_decode = check_mode_and_return_channels(decoder_mode)
    # if there is only a single timestep, decode it
    num_timesteps_to_decode = max(int(t * decode_ratio), 1)
    num_timesteps_to_encode = max(int(t * encode_ratio), 1)
    # we do this as a numpy array to take advantage of
    # numpy's permuted function
    flat_timesteps = np.concatenate(
        (
            np.ones(t - (num_timesteps_to_decode + num_timesteps_to_encode), dtype=np.int_),
            np.ones(num_timesteps_to_decode, dtype=np.int_) * 2,
            np.zeros(num_timesteps_to_encode, dtype=np.int_),
        )
    )
    b_flat_timesteps = repeat(flat_timesteps, "t -> b t", b=b)
    # hopefully this will allow for reproducibility, since random is seeded
    rng = np.random.default_rng(random.randint(0, 100))
    b_flat_timesteps_t = torch.from_numpy(rng.permuted(b_flat_timesteps, axis=1)).to(
        space_time_x.device
    )
    space_time_mask = repeat(
        b_flat_timesteps_t,
        "b t-> b h w t c_g",
        h=h,
        w=w,
        c_g=len(SPACE_TIME_BANDS_GROUPS_IDX),
    ).clone()
    # make the mask as if bands_to_mask and bands_to_decode both = None
    time_mask = repeat(
        b_flat_timesteps_t,
        "b t-> b t c_g",
        c_g=len(TIME_BAND_GROUPS_IDX),
    ).clone()
    space_mask = _random_mask_for_b(b, space_x.device, encode_ratio, decode_ratio)
    space_mask = repeat(
        space_mask, "b -> b h w c_g", h=h, w=w, c_g=len(SPACE_BAND_GROUPS_IDX)
    ).clone()
    static_mask = _random_mask_for_b(b, static_x.device, encode_ratio, decode_ratio)
    static_mask = repeat(static_mask, "b -> b c_g", c_g=len(STATIC_BAND_GROUPS_IDX)).clone()
    if max([len(x[0]) for x in bands_to_encode]) >= 1:  # encoder mode != random
        # for static in time data,
        # ignore all previous calculations about what should be encoded
        static_mask = torch.clamp(static_mask, min=1)
        space_mask = torch.clamp(space_mask, min=1)

        s_t_e, s_e, t_e, st_e = bands_to_encode

        if len(s_t_e[0]) > 0:
            # there are space time bands to decode
            s_t_bands_to_mask = s_t_e[1]
            space_time_mask[:, :, :, :, s_t_bands_to_mask] = torch.clamp(
                space_time_mask[:, :, :, :, s_t_bands_to_mask], min=1
            )
        else:
            space_time_mask = torch.clamp(space_time_mask, min=1)

        if len(s_e[0]) > 0:
            s_bands_to_encode = s_e[0]
            # there are space bands to mask
            space_mask[:, :, :, s_bands_to_encode] = 0

        if len(t_e[0]) > 0:
            t_bands_to_mask = t_e[1]
            time_mask[:, :, t_bands_to_mask] = torch.clamp(time_mask[:, :, t_bands_to_mask], min=1)
        else:
            time_mask = torch.clamp(time_mask, min=1)

        if len(st_e[0]) > 0:
            st_bands_to_encode = st_e[0]
            static_mask[:, st_bands_to_encode] = 0

    if max([len(x[0]) for x in bands_to_decode]) >= 1:  # decoder mode != random
        # for static in time data,
        # ignore all previous calculations about what should be decoded
        static_mask = torch.clamp(static_mask, max=1)
        space_mask = torch.clamp(space_mask, max=1)

        s_t_d, s_d, t_d, st_d = bands_to_decode

        if len(s_t_d[0]) > 0:
            # there are space time bands to decode
            s_t_bands_to_mask = s_t_d[1]
            space_time_mask[:, :, :, :, s_t_bands_to_mask] = torch.clamp(
                space_time_mask[:, :, :, :, s_t_bands_to_mask], max=1
            )
        else:
            space_time_mask = torch.clamp(space_time_mask, max=1)

        if len(s_d[0]) > 0:
            s_bands_to_decode = s_d[0]
            # there are space bands to mask
            space_mask[:, :, :, s_bands_to_decode] = 2

        if len(t_d[0]) > 0:
            t_bands_to_mask = t_d[1]
            time_mask[:, :, t_bands_to_mask] = torch.clamp(time_mask[:, :, t_bands_to_mask], max=1)
        else:
            time_mask = torch.clamp(time_mask, max=1)

        if len(st_d[0]) > 0:
            st_bands_to_decode = st_d[0]
            static_mask[:, st_bands_to_decode] = 2

    return MaskedOutput(
        space_time_x.clone(),
        space_x.clone(),
        time_x.clone(),
        static_x.clone(),
        space_time_mask,
        space_mask,
        time_mask,
        static_mask,
        months,
    )


def batch_mask_space(
    space_time_x: torch.Tensor,
    space_x: torch.Tensor,
    time_x: torch.Tensor,
    static_x: torch.Tensor,
    months: torch.Tensor,
    patch_size: int,
    encode_ratio: float,
    decode_ratio: float,
    mode: List[Tuple[str, str]],
    decoder_mode: List[Tuple[str, str]],
):
    """
    Masks out patches (blocks of of pxpxtxBAND_GROUPs).
    e.g. if mask_ratio=0.25, h = w = 8 and p=2, then a mask might be:
    [0 0 1 1]
    [0 0 1 1]
    [0 0 0 0]
    [0 0 0 0]
    repeated over all dynamic timesteps + channel groups and static channel groups
    Operates over batches where each item in the batch is independently masked
    """
    bands_to_encode = check_mode_and_return_channels(mode)
    bands_to_decode = check_mode_and_return_channels(decoder_mode)
    b, h, w, t, _ = space_time_x.shape
    assert (h % patch_size == 0) and (w % patch_size == 0)
    h_p = int(h / patch_size)
    w_p = int(w / patch_size)
    total_patches = h_p * w_p
    num_patches_to_encode = int(total_patches * encode_ratio)
    num_patches_to_decode = int(total_patches * decode_ratio)
    # we do this as a numpy array to take advantage of
    # numpy's permuted function
    flat_patches = np.concatenate(
        (
            np.ones(
                total_patches - (num_patches_to_encode + num_patches_to_decode), dtype=np.int_
            ),
            np.ones(num_patches_to_decode, dtype=np.int_) * 2,
            np.zeros(num_patches_to_encode, dtype=np.int_),
        )
    )
    b_flat_patches = repeat(flat_patches, "p -> b p", b=b)
    # hopefully this will allow for reproducibility, since random is seeded
    rng = np.random.default_rng(random.randint(0, 100))
    b_flat_patches = rng.permuted(b_flat_patches, axis=1)
    two_d_patch_mask = rearrange(b_flat_patches, "b (h w) -> b h w", h=h_p, w=w_p)
    two_d_mask = np.repeat(
        np.repeat(two_d_patch_mask, repeats=patch_size, axis=1), repeats=patch_size, axis=2
    )
    space_time_mask = (
        torch.from_numpy(
            repeat(
                two_d_mask,
                "b h w -> b h w t c_g",
                t=t,
                c_g=len(SPACE_TIME_BANDS_GROUPS_IDX),
            )
        )
        .clone()
        .to(space_time_x.device)
    )

    space_mask = (
        torch.from_numpy(
            repeat(
                two_d_mask,
                "b h w -> b h w c_g",
                c_g=len(SPACE_BAND_GROUPS_IDX),
            )
        )
        .clone()
        .to(space_x.device)
    )
    time_mask = _random_mask_for_b(b, time_x.device, encode_ratio, decode_ratio)
    time_mask = repeat(time_mask, "b -> b t c_g", t=t, c_g=len(TIME_BAND_GROUPS_IDX)).clone()
    static_mask = _random_mask_for_b(b, static_x.device, encode_ratio, decode_ratio)
    static_mask = repeat(static_mask, "b -> b c_g", c_g=len(STATIC_BAND_GROUPS_IDX)).clone()

    if max([len(x[0]) for x in bands_to_encode]) >= 1:  # encoder mode != random
        # for static in space data,
        # ignore all previous calculations about what should be encoded
        static_mask = torch.clamp(static_mask, min=1)
        time_mask = torch.clamp(time_mask, min=1)

        s_t_e, s_e, t_e, st_e = bands_to_encode

        if len(s_t_e[0]) > 0:
            # there are space time bands to decode
            s_t_bands_to_mask = s_t_e[1]
            space_time_mask[:, :, :, :, s_t_bands_to_mask] = torch.clamp(
                space_time_mask[:, :, :, :, s_t_bands_to_mask], min=1
            )
        else:
            space_time_mask = torch.clamp(space_time_mask, min=1)

        if len(s_e[0]) > 0:
            s_bands_to_mask = s_e[1]
            # there are space bands to mask
            space_mask[:, :, :, s_bands_to_mask] = torch.clamp(
                space_mask[:, :, :, s_bands_to_mask], min=1
            )
        else:
            space_mask = torch.clamp(space_mask, min=1)

        if len(t_e[0]) > 0:
            t_bands_to_encode = t_e[0]
            time_mask[:, :, t_bands_to_encode] = 0

        if len(st_e[0]) > 0:
            st_bands_to_encode = st_e[0]
            static_mask[:, st_bands_to_encode] = 0

    if max([len(x[0]) for x in bands_to_decode]) >= 1:  # decoder mode != random
        # for static in space data,
        # ignore all previous calculations about what should be decoded
        static_mask = torch.clamp(static_mask, max=1)
        time_mask = torch.clamp(time_mask, max=1)

        s_t_d, s_d, t_d, st_d = bands_to_decode

        if len(s_t_d[0]) > 0:
            # there are space time bands to decode
            s_t_bands_to_mask = s_t_d[1]
            space_time_mask[:, :, :, :, s_t_bands_to_mask] = torch.clamp(
                space_time_mask[:, :, :, :, s_t_bands_to_mask], max=1
            )
        else:
            space_time_mask = torch.clamp(space_time_mask, max=1)

        if len(s_d[0]) > 0:
            s_bands_to_mask = s_d[1]
            # there are space bands to mask
            space_mask[:, :, :, s_bands_to_mask] = torch.clamp(
                space_mask[:, :, :, s_bands_to_mask], max=1
            )
        else:
            space_mask = torch.clamp(space_mask, max=1)

        if len(t_d[0]) > 0:
            t_bands_to_decode = t_d[0]
            time_mask[:, :, t_bands_to_decode] = 2

        if len(st_d[0]) > 0:
            st_bands_to_decode = st_d[0]
            static_mask[:, st_bands_to_decode] = 2

    return MaskedOutput(
        space_time_x.clone(),
        space_x.clone(),
        time_x.clone(),
        static_x.clone(),
        space_time_mask,
        space_mask,
        time_mask,
        static_mask,
        months,
    )


def batch_mask_random(
    space_time_x: torch.Tensor,
    space_x: torch.Tensor,
    time_x: torch.Tensor,
    static_x: torch.Tensor,
    months: torch.Tensor,
    encode_ratio: float,
    decode_ratio: float,
    patch_size: int,
    ignore_band_groups: Optional[List[str]] = None,
):
    """
    Masks out random tokens (blocks of of pxpx1x1).
    e.g. if mask_ratio=0.25, h = w = 8 and p=2, then a mask (for one timestep)
    and channel group) might be
    [0 0 1 1]
    [0 0 1 1]
    [0 0 0 0]
    [0 0 0 0]
    Operates over batches where each item in the batch is independently masked
    """

    def indices_of_ignored(band_groups: OrderedDict, ignore_band_groups: Optional[List]):
        if ignore_band_groups is None:
            return len(band_groups), []
        else:
            ignored_band_indices = []
            for idx, (band, _) in enumerate(band_groups.items()):
                if band in ignore_band_groups:
                    ignored_band_indices.append(idx)
            return len(band_groups) - len(ignored_band_indices), ignored_band_indices

    b, h, w, t, _ = space_time_x.shape
    c_s_t, c_s_t_ignore = indices_of_ignored(SPACE_TIME_BANDS_GROUPS_IDX, ignore_band_groups)
    c_sp, c_sp_ignore = indices_of_ignored(SPACE_BAND_GROUPS_IDX, ignore_band_groups)
    c_t, c_t_ignore = indices_of_ignored(TIME_BAND_GROUPS_IDX, ignore_band_groups)
    c_st, c_st_ignore = indices_of_ignored(STATIC_BAND_GROUPS_IDX, ignore_band_groups)
    assert (h % patch_size == 0) and (w % patch_size == 0)
    h_p = int(h / patch_size)
    w_p = int(w / patch_size)

    num_space_time_tokens = h_p * w_p * t * c_s_t
    num_space_tokens = h_p * w_p * c_sp
    num_time_tokens = t * c_t
    num_static_tokens = c_st

    total_tokens = num_space_time_tokens + num_space_tokens + num_time_tokens + num_static_tokens
    tokens_the_decoder_will_unmask = int(total_tokens * decode_ratio)
    tokens_the_encoder_will_encode = int(total_tokens * encode_ratio)
    # we do this as a numpy array to take advantage of
    # numpy's permuted function
    flat_tokens = np.concatenate(
        (
            np.ones(
                total_tokens - (tokens_the_encoder_will_encode + tokens_the_decoder_will_unmask),
                dtype=np.int_,
            ),
            np.ones(tokens_the_decoder_will_unmask, dtype=np.int_) * 2,
            np.zeros(
                tokens_the_encoder_will_encode,
                dtype=np.int_,
            ),
        )
    )
    b_flat_tokens = repeat(flat_tokens, "t -> b t", b=b)
    # hopefully this will allow for reproducibility, since random is seeded
    rng = np.random.default_rng(random.randint(0, 100))
    b_flat_tokens = rng.permuted(b_flat_tokens, axis=1)

    s_t_tokens = b_flat_tokens[:, :num_space_time_tokens]
    s_t_tokens = rearrange(s_t_tokens, "b (h w t c) -> b h w t c", h=h_p, w=w_p, t=t, c=c_s_t)
    for s_t_ignored_cg in c_s_t_ignore:
        # make the empty array
        ignored_mask = np.ones_like(s_t_tokens[:, :, :, :, 0])
        s_t_tokens = np.insert(s_t_tokens, obj=s_t_ignored_cg, values=ignored_mask, axis=-1)
    space_time_mask = torch.from_numpy(
        np.repeat(np.repeat(s_t_tokens, repeats=patch_size, axis=1), repeats=patch_size, axis=2)
    ).to(space_time_x.device)

    space_tokens = b_flat_tokens[:, num_space_time_tokens : -(num_time_tokens + num_static_tokens)]
    space_tokens = rearrange(space_tokens, "b (h w c) -> b h w c", h=h_p, w=w_p, c=c_sp)
    for sp_ignored_cg in c_sp_ignore:
        # make the empty array
        ignored_mask = np.ones_like(space_tokens[:, :, :, 0])
        space_tokens = np.insert(space_tokens, obj=sp_ignored_cg, values=ignored_mask, axis=-1)
    space_mask = torch.from_numpy(
        np.repeat(np.repeat(space_tokens, repeats=patch_size, axis=1), repeats=patch_size, axis=2)
    ).to(space_x.device)

    time_tokens = b_flat_tokens[:, -(num_time_tokens + num_static_tokens) : -num_static_tokens]
    time_mask = rearrange(time_tokens, "b (t c) -> b t c", t=t, c=c_t)
    for t_ignored_cg in c_t_ignore:
        # make the empty array
        ignored_mask = np.ones_like(time_mask[:, :, 0])
        time_mask = np.insert(time_mask, obj=t_ignored_cg, values=ignored_mask, axis=-1)
    time_mask_t = torch.from_numpy(time_mask).to(time_x.device)

    static_tokens = b_flat_tokens[:, -num_static_tokens:]
    for st_ignored_cg in c_st_ignore:
        # make the empty array
        ignored_mask = np.ones_like(static_tokens[:, 0])
        static_tokens = np.insert(static_tokens, obj=st_ignored_cg, values=ignored_mask, axis=-1)
    static_mask = torch.from_numpy(static_tokens).to(static_x.device)
    return MaskedOutput(
        space_time_x.clone(),
        space_x.clone(),
        time_x.clone(),
        static_x.clone(),
        space_time_mask,
        space_mask,
        time_mask_t,
        static_mask,
        months,
    )
