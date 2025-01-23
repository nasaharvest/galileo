from functools import partial
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
from timm.models.vision_transformer import Block


class DOFAWrapper(nn.Module):
    def __init__(
        self, weights_path: Path, size="base", do_pool=True, temporal_pooling: str = "mean"
    ):
        super().__init__()

        if size == "base":
            self.encoder = vit_base_patch16()
            checkpoint = torch.load(weights_path / "DOFA_ViT_base_e100.pth", map_location="cpu")
            self.dim = 768
        elif size == "large":
            self.encoder = vit_large_patch16()
            checkpoint = torch.load(weights_path / "DOFA_ViT_large_e100.pth", map_location="cpu")
            self.dim = 1024
        else:
            raise ValueError(f"size must be base or large, not {size}")

        self.encoder.load_state_dict(checkpoint, strict=False)
        self.image_resolution = 224
        self.patch_size = 16
        self.grid_size = int(self.image_resolution / self.patch_size)
        # Sentinel-2 wavelengths, with RGB re-ordered
        self.s2_waves = [0.665, 0.56, 0.49, 0.705, 0.74, 0.783, 0.842, 1.61, 2.19]
        self.s1_waves = [3.75, 3.75]
        self.do_pool = do_pool

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
        assert images.shape[1] in (13, 2)
        # need to re-order RGB and remove coastal aerosol, water vapour, narrow NIR, and cirrus
        if images.shape[1] == 13:
            channel_ids = [3, 2, 1, 4, 5, 6, 7, 11, 12]
            images = images[:, channel_ids, :, :]
        return self.resize(images)  # (bsz, C, H, W)

    def forward(self, s2=None, s1=None, months=None):
        # TODO add support for s1 with s1 waves
        if s2 is not None:
            if len(s2.shape) == 5:
                outputs_l: List[torch.Tensor] = []
                for timestep in range(s2.shape[3]):
                    image = self.preproccess(s2[:, :, :, timestep])
                    output = self.encoder.forward_features(image, wave_list=self.s2_waves)
                    if self.do_pool:
                        output = output.mean(dim=1)
                    else:
                        output = output[:, 1:]
                    outputs_l.append(output)
                outputs_t = torch.stack(outputs_l, dim=-1)  # b h w d t
                if self.temporal_pooling == "mean":
                    return outputs_t.mean(dim=-1)
                else:
                    return torch.amax(outputs_t, dim=-1)
            else:
                s2 = self.preproccess(s2)
                output = self.encoder.forward_features(s2, wave_list=self.s2_waves)
                if self.do_pool:
                    return output.mean(dim=1)
                else:
                    return output[:, 1:]
        elif s1 is not None:
            if len(s1.shape) == 5:
                outputs_l: List[torch.Tensor] = []
                for timestep in range(s1.shape[3]):
                    image = self.preproccess(s1[:, :, :, timestep])
                    output = self.encoder.forward_features(image, wave_list=self.s1_waves)
                    if self.do_pool:
                        output = output.mean(dim=1)
                    else:
                        output = output[:, 1:]
                    outputs_l.append(output)
                outputs_t = torch.stack(outputs_l, dim=-1)  # b h w d t
                if self.temporal_pooling == "mean":
                    return outputs_t.mean(dim=-1)
                else:
                    return torch.amax(outputs_t, dim=-1)
            else:
                s1 = self.preproccess(s1)
                output = self.encoder.forward_features(s1, wave_list=self.s1_waves)
                if self.do_pool:
                    return output.mean(dim=1)
                else:
                    return output[:, 1:]


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


class TransformerWeightGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads=4, num_layers=1):
        super(TransformerWeightGenerator, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation="gelu",
            norm_first=False,
            batch_first=False,
            dropout=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Linear layer to map transformer output to desired weight shape
        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)
        self.wt_num = 128
        self.weight_tokens = nn.Parameter(torch.empty([self.wt_num, input_dim]))
        self.bias_token = nn.Parameter(torch.empty([1, input_dim]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is
        # too big (2.)
        torch.nn.init.normal_(self.weight_tokens, std=0.02)
        torch.nn.init.normal_(self.bias_token, std=0.02)

    def forward(self, x):
        # x should have shape [seq_len, batch, input_dim]
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)
        transformer_output = self.transformer_encoder(x)
        weights = self.fc_weight(transformer_output[self.wt_num : -1] + pos_wave)
        bias = self.fc_bias(transformer_output[-1])  # Using the last output to generate bias
        return weights, bias


class Basic1d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        conv = nn.Linear(in_channels, out_channels, bias)
        self.conv = nn.Sequential(
            conv,
        )
        if not bias:
            self.conv.add_module("ln", nn.LayerNorm(out_channels))
        self.conv.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out


class FCResLayer(nn.Module):
    def __init__(self, linear_size=128):
        super(FCResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out


class Dynamic_MLP_OFA(nn.Module):
    """
    Input: channels of wavelength (normalized): List -> List
           kernel size of the depth-wise convolution: kernel_size, default 3x3
           wv_planes
           inplanes
    """

    def __init__(self, wv_planes, inter_dim=128, kernel_size=3, embed_dim=1024):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
        self.inter_dim = inter_dim
        self.patch_size = (kernel_size, kernel_size)
        self.num_patches = -1

        self.weight_generator = TransformerWeightGenerator(wv_planes, self._num_kernel, embed_dim)
        self.scaler = 0.01

        self.fclayer = FCResLayer(wv_planes)

        self._init_weights()

    def _get_weights(self, waves):
        dynamic_weights = self.weight_generator(waves)

        return dynamic_weights

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self):
        """
        initialize the base weights and dynamic mlp weights
        """
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def forward(self, img_feat, wvs):
        inplanes = wvs.size(0)
        # wv_feats: 9,128 -> 9, 3x3x3
        waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, wvs * 1000)
        waves = self.fclayer(waves)
        weight, bias = self._get_weights(waves)  # 3x3x3

        dynamic_weight = weight.view(
            self.embed_dim, inplanes, self.kernel_size, self.kernel_size
        )  # 3xoutdx16x16
        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler

        weights = dynamic_weight * self.scaler

        dynamic_out = F.conv2d(
            img_feat, weights, bias=bias, stride=self.kernel_size, padding=1, dilation=1
        )

        x = dynamic_out
        x = x.flatten(2).transpose(1, 2)

        return x, waves


class OFAViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        drop_rate=0.0,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        wv_planes=128,
        num_classes=45,
        global_pool=True,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.wv_planes = wv_planes
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = norm_layer
            embed_dim = embed_dim
            self.fc_norm = norm_layer(embed_dim)
        else:
            self.norm = norm_layer(embed_dim)

        self.patch_embed = Dynamic_MLP_OFA(
            wv_planes=128, inter_dim=128, kernel_size=16, embed_dim=embed_dim
        )
        self.num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(depth)
            ]
        )

        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, wave_list):
        # embed patches
        wavelist = torch.tensor(wave_list, device=x.device).float()
        self.waves = wavelist

        x, _ = self.patch_embed(x, self.waves)

        x = x + self.pos_embed[:, 1:, :]
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for block in self.blocks:
            x = block(x)

        return x

    def forward_head(self, x, pre_logits=False):
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, wave_list):
        x = self.forward_features(x, wave_list)
        x = self.forward_head(x)
        return x


def vit_small_patch16(**kwargs):
    model = OFAViT(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_base_patch16(**kwargs):
    model = OFAViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large_patch16(**kwargs):
    model = OFAViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge_patch14(**kwargs):
    model = OFAViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
