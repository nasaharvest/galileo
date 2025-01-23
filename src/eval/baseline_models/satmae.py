from functools import partial
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import PatchEmbed, VisionTransformer


class SatMAEWrapper(nn.Module):
    def __init__(
        self,
        pretrained_path,
        size="large",
        img_size=96,
        do_pool=True,
        temporal_pooling: str = "mean",
    ):
        super().__init__()
        if size == "large":
            self.encoder = vit_large(img_size=img_size, patch_size=8, in_chans=10)
            self.dim = 1024
        elif size == "base":
            self.encoder = vit_base(img_size=img_size, patch_size=8, in_chans=10)
            self.dim = 768

        checkpoint = torch.load(pretrained_path, map_location="cpu")["model"]

        if img_size != 96:
            checkpoint = interpolate_pos_embed(self.encoder, checkpoint)

        self.encoder.load_state_dict(checkpoint, strict=False)
        self.image_resolution = img_size
        self.do_pool = do_pool
        self.patch_size = 8
        self.grid_size = int(self.image_resolution / self.patch_size)
        if temporal_pooling not in ["mean", "max"]:
            raise ValueError(
                f"Expected temporal_pooling to be in ['mean', 'max'], got {temporal_pooling}"
            )
        self.temporal_pooling = temporal_pooling

    def resize(self, images):
        images = F.interpolate(
            images,
            size=(self.image_resolution, self.image_resolution),
            mode="bilinear",
            align_corners=False,
        )
        return images

    def preproccess(self, images):
        if len(images.shape) == 5:
            # take the mean along the temporal dimension
            images = torch.mean(images, dim=2)
        images = rearrange(images, "b h w c -> b c h w")
        assert images.shape[1] == 13
        return self.resize(images)  # (bsz, C, H, W)

    def forward(self, s2=None, s1=None, months=None):
        if s2 is None:
            raise ValueError("S2 can't be None for SatMAE")
        if len(s2.shape) == 5:
            outputs_l: List[torch.Tensor] = []
            for timestep in range(s2.shape[3]):
                image = self.preproccess(s2[:, :, :, timestep])
                output = self.encoder.forward_features(image)
                # output shape for atto: (bsz, 320, 7, 7)
                # output shape for tiny: (bsz, 768, 6, 6)
                if self.do_pool:
                    output = output.mean(dim=1)
                else:
                    output = rearrange(output, "b (c_g l) d -> b l c_g d", c_g=3).mean(dim=-2)
                outputs_l.append(output)
            outputs_t = torch.stack(outputs_l, dim=-1)  # b h w d t
            if self.temporal_pooling == "mean":
                return outputs_t.mean(dim=-1)
            else:
                return torch.amax(outputs_t, dim=-1)
        else:
            s2 = self.preproccess(s2)
            output = self.encoder.forward_features(s2)
            if self.do_pool:
                return output.mean(dim=1)
            else:
                return rearrange(output, "b (c_g l) d -> b l c_g d", c_g=3).mean(dim=-2)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=float, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.double()


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        try:
            num_patches = model.patch_embed.num_patches
        except AttributeError:
            num_patches = model.patch_embed[0].num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(
                0, 3, 1, 2
            )
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed

        return checkpoint_model


class GroupChannelsVisionTransformer(VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(
        self,
        global_pool=False,
        channel_embed=256,
        channel_groups=((0, 1, 2, 6), (3, 4, 5, 7), (8, 9)),
        **kwargs,
    ):
        super().__init__(**kwargs)
        img_size = kwargs["img_size"]
        patch_size = kwargs["patch_size"]
        embed_dim = kwargs["embed_dim"]

        self.channel_groups = channel_groups

        self.patch_embed = nn.ModuleList(
            [PatchEmbed(img_size, patch_size, len(group), embed_dim) for group in channel_groups]
        )
        num_patches = self.patch_embed[0].num_patches

        # Positional and channel embed
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim - channel_embed))
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(num_patches**0.5), cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        num_groups = len(channel_groups)
        self.channel_embed = nn.Parameter(torch.zeros(1, num_groups, channel_embed))
        chan_embed = get_1d_sincos_pos_embed_from_grid(
            self.channel_embed.shape[-1], torch.arange(num_groups).numpy()
        )
        self.channel_embed.data.copy_(torch.from_numpy(chan_embed).float().unsqueeze(0))

        # Extra embedding for cls to fill embed_dim
        self.channel_cls_embed = nn.Parameter(torch.zeros(1, 1, channel_embed))
        channel_cls_embed = torch.zeros((1, channel_embed))
        self.channel_cls_embed.data.copy_(channel_cls_embed.float().unsqueeze(0))

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        b, c, h, w = x.shape

        x_c_embed = []
        for i, group in enumerate(self.channel_groups):
            x_c = x[:, group, :, :]
            x_c_embed.append(self.patch_embed[i](x_c))  # (N, L, D)

        x = torch.stack(x_c_embed, dim=1)  # (N, G, L, D)
        _, G, L, D = x.shape

        # add channel embed
        channel_embed = self.channel_embed.unsqueeze(2)  # (1, c, 1, cD)
        pos_embed = self.pos_embed[:, 1:, :].unsqueeze(1)  # (1, 1, L, pD)

        # Channel embed same across (x,y) position, and pos embed same across channel (c)
        channel_embed = channel_embed.expand(-1, -1, pos_embed.shape[2], -1)  # (1, c, L, cD)
        pos_embed = pos_embed.expand(-1, channel_embed.shape[1], -1, -1)  # (1, c, L, pD)
        pos_channel = torch.cat((pos_embed, channel_embed), dim=-1)  # (1, c, L, D)

        # add pos embed w/o cls token
        x = x + pos_channel  # (N, G, L, D)
        x = x.view(b, -1, D)  # (N, G*L, D)

        cls_pos_channel = torch.cat(
            (self.pos_embed[:, :1, :], self.channel_cls_embed), dim=-1
        )  # (1, 1, D)
        cls_tokens = cls_pos_channel + self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (N, 1 + c*L, D)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        return x[:, 1:, :]  # remove cls token


def vit_base(**kwargs):
    model = GroupChannelsVisionTransformer(
        channel_embed=256,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large(**kwargs):
    model = GroupChannelsVisionTransformer(
        channel_embed=256,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
