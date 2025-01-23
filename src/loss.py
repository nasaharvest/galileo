from typing import Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torchvision.transforms.functional import resize

from src.data.dataset import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    TIME_BAND_GROUPS_IDX,
)


def construct_target_encoder_masks(
    s_t_m: torch.Tensor, sp_m: torch.Tensor, t_m: torch.Tensor, st_m: torch.Tensor, method: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if method == "decoder_only":
        # we want 0s where the mask == 2
        return ~(s_t_m == 2), ~(sp_m == 2), ~(t_m == 2), ~(st_m == 2)
    elif method == "all":
        # we want all zeros
        return (
            torch.zeros_like(s_t_m),
            torch.zeros_like(sp_m),
            torch.zeros_like(t_m),
            torch.zeros_like(st_m),
        )
    elif method == "decoder_and_encoder":
        # we want 0s where the mask is not equal to 1
        return s_t_m == 1, sp_m == 1, t_m == 1, st_m == 1
    else:
        raise ValueError(f"Unexpected method {method}")


def mse_loss(
    t_s_t,
    t_sp,
    t_t,
    t_st,
    p_s_t,
    p_sp,
    p_t,
    p_st,
    s_t_m,
    sp_m,
    t_m,
    st_m,
):
    encoder_size = t_s_t.shape[-1]
    expanded_s_t_m = repeat(s_t_m, "b h w t c_g -> b h w t c_g d", d=encoder_size)
    expanded_sp_m = repeat(sp_m, "b h w c_g -> b h w c_g d", d=encoder_size)
    expanded_t_m = repeat(t_m, "b t c_g -> b t c_g d", d=encoder_size)
    expanded_st_m = repeat(st_m, "b c_g -> b c_g d", d=encoder_size)
    return F.mse_loss(
        torch.concat(
            [
                p_s_t[expanded_s_t_m == 2],
                p_sp[expanded_sp_m == 2],
                p_t[expanded_t_m == 2],
                p_st[expanded_st_m == 2],
            ]
        ),
        torch.concat(
            [
                t_s_t[expanded_s_t_m == 2],
                t_sp[expanded_sp_m == 2],
                t_t[expanded_t_m == 2],
                t_st[expanded_st_m == 2],
            ]
        ).float(),
    )


def seq_and_cat(s_t, sp, t, st):
    s_t = rearrange(s_t, "b h w t c_g d -> b (h w t c_g) d")
    sp = rearrange(sp, "b h w c_g d -> b (h w c_g) d")
    t = rearrange(t, "b t c_g d -> b (t c_g) d")
    # st is already a sequence
    return torch.cat([s_t, sp, t, st], dim=1)


def expand_and_reciprocate(t):
    reciprocals = torch.reciprocal(t.float())
    return torch.repeat_interleave(reciprocals, t)


def patch_disc_loss(
    t_s_t,
    t_sp,
    t_t,
    t_st,
    p_s_t,
    p_sp,
    p_t,
    p_st,
    s_t_m,
    sp_m,
    t_m,
    st_m,
    mask_other_samples: bool,
    pred2unit: bool = True,
    tau: float = 0.2,
):
    # create tensors of shape (bsz, seq_len, dim)
    all_masks = seq_and_cat(
        s_t_m.unsqueeze(dim=-1),
        sp_m.unsqueeze(dim=-1),
        t_m.unsqueeze(dim=-1),
        st_m.unsqueeze(dim=-1),
    ).squeeze(-1)
    all_preds = seq_and_cat(p_s_t, p_sp, p_t, p_st)
    all_targets = seq_and_cat(t_s_t, t_sp, t_t, t_st)

    pred = all_preds[all_masks == 2].unsqueeze(dim=0)
    target = all_targets[all_masks == 2].unsqueeze(dim=0)

    bs, nt, d = pred.shape

    if pred2unit:
        pred_mu = pred.mean(1, keepdims=True)
        pred_std = pred.std(1, keepdims=True)
        pred = (pred - pred_mu) / (pred_std + 1e-4)

    pred = F.normalize(pred, p=2, dim=-1)
    target = F.normalize(target, p=2, dim=-1)

    scores = torch.einsum("npd,nqd->npq", pred, target) / tau
    count = (all_masks == 2).sum(dim=-1)

    if mask_other_samples:
        logit_mask = torch.full_like(scores, -torch.finfo(scores.dtype).max)
        start = 0
        for c in count:
            end = start + c
            logit_mask[:, start:end, start:end] = 0
            start += c

        scores = scores + logit_mask

    labels = torch.arange(nt, dtype=torch.long, device=pred.device)[None].repeat(bs, 1)
    loss = F.cross_entropy(scores.flatten(0, 1), labels.flatten(0, 1), reduction="none") * (
        tau * 2
    )

    # emulate averaging across the batch dimension
    loss_multiplier = expand_and_reciprocate(count)
    loss = (loss * loss_multiplier).sum() / t_s_t.shape[0]
    return loss


def mae_loss(
    p_s_t,
    p_sp,
    p_t,
    p_st,
    s_t_x,
    sp_x,
    t_x,
    st_x,
    s_t_m,
    sp_m,
    t_m,
    st_m,
    patch_size,
    max_patch_size,
):
    SPACE_TIME_BAND_EXPANSION = torch.tensor(
        [len(x) for x in SPACE_TIME_BANDS_GROUPS_IDX.values()], device=sp_m.device
    ).long()
    SPACE_BAND_EXPANSION = torch.tensor(
        [len(x) for x in SPACE_BAND_GROUPS_IDX.values()], device=sp_m.device
    ).long()
    TIME_BAND_EXPANSION = torch.tensor(
        [len(x) for x in TIME_BAND_GROUPS_IDX.values()], device=sp_m.device
    ).long()
    STATIC_BAND_EXPANSION = torch.tensor(
        [len(x) for x in STATIC_BAND_GROUPS_IDX.values()], device=sp_m.device
    ).long()

    pixel_s_t_m = torch.repeat_interleave(s_t_m, repeats=SPACE_TIME_BAND_EXPANSION, dim=-1)
    pixel_sp_m = torch.repeat_interleave(sp_m, repeats=SPACE_BAND_EXPANSION, dim=-1)
    pixel_st_m = torch.repeat_interleave(st_m, repeats=STATIC_BAND_EXPANSION, dim=-1)
    pixel_t_m = torch.repeat_interleave(t_m, repeats=TIME_BAND_EXPANSION, dim=-1)

    output_p_s_t = []
    for idx, (_, c_g) in enumerate(SPACE_TIME_BANDS_GROUPS_IDX.items()):
        channel_group_p_s_t = p_s_t[:, :, :, :, idx, : ((max_patch_size**2) * len(c_g))]
        channel_group_p_s_t = rearrange(
            channel_group_p_s_t,
            "b t_h t_w t (c_g p_h p_w) -> b (t_h p_h) (t_w p_w) t c_g",
            c_g=len(c_g),
            p_w=max_patch_size,
            p_h=max_patch_size,
        )
        if patch_size < max_patch_size:
            channel_group_p_s_t = rearrange(
                resize(
                    rearrange(channel_group_p_s_t, "b h w t d -> b (t d) h w"),
                    size=(s_t_x.shape[1], s_t_x.shape[2]),
                ),
                "b (t d) h w -> b h w t d",
                t=s_t_x.shape[3],
                d=len(c_g),
            )

        output_p_s_t.append(channel_group_p_s_t)

    output_p_sp = []
    for idx, (_, c_g) in enumerate(SPACE_BAND_GROUPS_IDX.items()):
        channel_group_p_sp = p_sp[:, :, :, idx, : ((max_patch_size**2) * len(c_g))]
        channel_group_p_sp = rearrange(
            channel_group_p_sp,
            "b t_h t_w (c_g p_h p_w) -> b (t_h p_h) (t_w p_w) c_g",
            c_g=len(c_g),
            p_w=max_patch_size,
            p_h=max_patch_size,
        )
        if patch_size < max_patch_size:
            channel_group_p_sp = rearrange(
                resize(
                    rearrange(channel_group_p_sp, "b h w d -> b d h w"),
                    size=(s_t_x.shape[1], s_t_x.shape[2]),
                ),
                "b d h w -> b h w d",
                d=len(c_g),
            )
        output_p_sp.append(channel_group_p_sp)

    output_p_t = []
    for idx, (_, c_g) in enumerate(TIME_BAND_GROUPS_IDX.items()):
        channel_group_p_t = p_t[:, :, idx, : len(c_g)]
        output_p_t.append(channel_group_p_t)

    output_p_st = []
    for idx, (_, c_g) in enumerate(STATIC_BAND_GROUPS_IDX.items()):
        channel_group_st_t = p_st[:, idx, : len(c_g)]
        output_p_st.append(channel_group_st_t)

    # these now have the same shape as s_t_x, etc.
    p_s_t = torch.cat(output_p_s_t, dim=-1)
    p_sp = torch.cat(output_p_sp, dim=-1)
    p_st = torch.cat(output_p_st, dim=-1)
    p_t = torch.cat(output_p_t, dim=-1)
    return F.smooth_l1_loss(
        torch.concat(
            [
                p_s_t[pixel_s_t_m == 2],
                p_sp[pixel_sp_m == 2],
                p_t[pixel_t_m == 2],
                p_st[pixel_st_m == 2],
            ]
        ),
        torch.concat(
            [
                s_t_x[pixel_s_t_m == 2],
                sp_x[pixel_sp_m == 2],
                t_x[pixel_t_m == 2],
                st_x[pixel_st_m == 2],
            ]
        ),
    )


def do_loss(config, loss_inputs):
    if config["loss_type"] == "patch_disc":
        loss = patch_disc_loss(
            *loss_inputs,
            mask_other_samples=config["loss_mask_other_samples"],
            pred2unit=config["pred2unit"],
            tau=config["tau"],
        )
    elif config["loss_type"] == "mse":
        loss = mse_loss(*loss_inputs)
    elif config["loss_type"] == "MAE":
        loss = mae_loss(*loss_inputs)
    else:
        raise f"loss_type must be patch_disc, MAE or mse, not {config['loss_type']}"

    return loss
