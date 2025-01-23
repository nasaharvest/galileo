# type: ignore

import math
from argparse import Namespace
from collections import OrderedDict
from pathlib import Path
from typing import AnyStr, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from torch import Tensor

PIXEL_WISE_MODALITIES = [
    "sentinel2",
    "sentinel1",
    "aster",
    "canopy_height_eth",
    "esa_worldcover",
    "dynamic_world",
]

# Input modalities for training
INP_MODALITIES = {
    "sentinel2": [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8A",
        "B8",
        "B9",
        "B11",
        "B12",
    ],
}


# Output modalities for training
OUT_MODALITIES = {
    "sentinel2": [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8A",
        "B8",
        "B9",
        "B11",
        "B12",
    ],
    "sentinel1": "all",
    "aster": "all",
    "era5": "all",
    "dynamic_world": "all",
    "canopy_height_eth": "all",
    "lat": "all",
    "lon": "all",
    "biome": "all",
    "eco_region": "all",
    "month": "all",
    "esa_worldcover": "all",
}

# an example of all the modalities. DO NOT CHANGE THIS, ALWAYS CHANGE THE INP and OUT MODALITIES ABOVE
MODALITIES_FULL = {
    "sentinel2": [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8A",
        "B8",
        "B9",
        "B10",
        "B11",
        "B12",
    ],
    "sentinel2_cloudmask": ["QA60"],
    "sentinel2_cloudprod": ["MSK_CLDPRB"],
    "sentinel2_scl": ["SCL"],
    "sentinel1": [
        "asc_VV",
        "asc_VH",
        "asc_HH",
        "asc_HV",
        "desc_VV",
        "desc_VH",
        "desc_HH",
        "desc_HV",
    ],
    "aster": ["elevation", "slope"],
    "era5": [
        "prev_month_avg_temp",
        "prev_month_min_temp",
        "prev_month_max_temp",
        "prev_month_total_precip",
        "curr_month_avg_temp",
        "curr_month_min_temp",
        "curr_month_max_temp",
        "curr_month_total_precip",
        "year_avg_temp",
        "year_min_temp",
        "year_max_temp",
        "year_total_precip",
    ],
    "dynamic_world": ["landcover"],
    "canopy_height_eth": ["height", "std"],
    "lat": ["sin", "cos"],
    "lon": ["sin", "cos"],
    "biome": ["biome"],
    "eco_region": ["eco_region"],
    "month": ["sin_month", "cos_month"],
    "esa_worldcover": ["map"],
}


class MMEarthWrapper(nn.Module):
    def __init__(
        self, weights_path: Path, size="atto", do_pool=True, temporal_pooling: str = "mean"
    ):
        super().__init__()
        if size == "atto":
            self.dim = 320
            check = weights_path / "mmearth-atto-checkpoint-199.pth"
            checkpoint = torch.load(check, map_location="cpu")
            weights = remap_checkpoint_keys(checkpoint["model"])
            args = Namespace(
                checkpoint_dir=check,
                random_crop=True,
                random_crop_size=112,
                patch_size=16,
                loss_aggr="uncertainty",
                use_orig_stem=False,
                mask_ratio=0.6,
                linear_probe=False,
            )
            args.inp_modalities = INP_MODALITIES
            args.out_modalities = OUT_MODALITIES

            args.modalities = args.inp_modalities.copy()
            args.modalities.update(args.out_modalities)
            args.modalities_full = MODALITIES_FULL

            model = convnextv2_atto(
                mask_ratio=args.mask_ratio,
                decoder_depth=1,
                decoder_embed_dim=512,
                norm_pix_loss=True,
                patch_size=args.patch_size,
                img_size=args.random_crop_size,
                args=args,
            )
            self.encoder = model.encoder
            self.encoder.load_state_dict(weights, strict=False)
            self.image_resolution = 112
            self.grid_size = 7

        elif size == "tiny":
            self.dim = 768
            check = weights_path / "mmearth-tiny-checkpoint-199.pth"
            checkpoint = torch.load(check, map_location="cpu")
            weights = remap_checkpoint_keys(checkpoint["model"])
            args = Namespace(
                checkpoint_dir=check,
                random_crop=True,
                random_crop_size=56,
                patch_size=8,
                loss_aggr="uncertainty",
                use_orig_stem=False,
                mask_ratio=0.6,
                linear_probe=False,
            )
            args.inp_modalities = INP_MODALITIES
            args.out_modalities = OUT_MODALITIES

            args.modalities = args.inp_modalities.copy()
            args.modalities.update(args.out_modalities)
            args.modalities_full = MODALITIES_FULL

            model = convnextv2_tiny(
                mask_ratio=args.mask_ratio,
                decoder_depth=1,
                decoder_embed_dim=512,
                norm_pix_loss=True,
                patch_size=args.patch_size,
                img_size=args.random_crop_size,
                args=args,
            )
            self.encoder = model.encoder
            self.encoder.load_state_dict(weights, strict=False)
            self.image_resolution = 56
            self.grid_size = 6
        else:
            raise ValueError(f"size must be atto or tiny, not {size}")

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
            raise ValueError(f"Unexpected input shape {images.shape}")
        images = rearrange(images, "b h w c -> b c h w")
        assert images.shape[1] == 13
        # MMEarth does not use B10 as input
        remove_idx = 10
        images = torch.cat(
            [images[:, :remove_idx, :, :], images[:, (remove_idx + 1) :, :, :]], dim=1
        )
        assert images.shape[1] == 12
        return self.resize(images)  # (bsz, 12, 112, 112)

    def forward(self, s2=None, s1=None, months=None):
        if s2 is None:
            raise ValueError("S2 can't be None for MMEarth")

        if len(s2.shape) == 5:
            outputs_l: List[torch.Tensor] = []
            for timestep in range(s2.shape[3]):
                image = self.preproccess(s2[:, :, :, timestep])
                output = self.encoder(image)
                # output shape for atto: (bsz, 320, 7, 7)
                # output shape for tiny: (bsz, 768, 6, 6)
                if self.do_pool:
                    output = output.mean(dim=-1).mean(dim=-1)
                else:
                    output = rearrange(output, "b c h w -> b (h w) c")
                outputs_l.append(output)
            outputs_t = torch.stack(outputs_l, dim=-1)  # b h w d t
            if self.temporal_pooling == "mean":
                return outputs_t.mean(dim=-1)
            else:
                return torch.amax(outputs_t, dim=-1)
        else:
            s2 = self.preproccess(s2)
            output = self.encoder(s2)
            if self.do_pool:
                return output.mean(dim=-1).mean(dim=-1)  # (bsz, dim)
            else:
                return rearrange(output, "b c h w -> b (h w) c")  # (bsz, seq_len, dim)


def remap_checkpoint_keys(ckpt):
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith("encoder"):
            k = ".".join(k.split(".")[1:])  # remove encoder in the name
        if k.endswith("kernel"):
            k = ".".join(k.split(".")[:-1])  # remove kernel in the name
            new_k = k + ".weight"
            if len(v.shape) == 3:  # resahpe standard convolution
                kv, in_dim, out_dim = v.shape
                ks = int(math.sqrt(kv))
                new_ckpt[new_k] = (
                    v.permute(2, 1, 0).reshape(out_dim, in_dim, ks, ks).transpose(3, 2)
                )
            elif len(v.shape) == 2:  # reshape depthwise convolution
                kv, dim = v.shape
                ks = int(math.sqrt(kv))
                new_ckpt[new_k] = v.permute(1, 0).reshape(dim, 1, ks, ks).transpose(3, 2)
            continue
        elif "ln" in k or "linear" in k:
            k = k.split(".")
            k.pop(-2)  # remove ln and linear in the name
            new_k = ".".join(k)
        elif "backbone.resnet" in k:
            # sometimes the resnet model is saved with the prefix backbone.resnet
            # we need to remove this prefix
            new_k = k.split("backbone.resnet.")[1]
        else:
            new_k = k
        new_ckpt[new_k] = v

    # reshape grn affine parameters and biases
    for k, v in new_ckpt.items():
        if k.endswith("bias") and len(v.shape) != 1:
            new_ckpt[k] = v.reshape(-1)
        elif "grn" in k:
            new_ckpt[k] = v.unsqueeze(0).unsqueeze(1)
    return new_ckpt


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-4)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    """ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv: nn.Module = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depth-wise conv
        self.norm: nn.Module = LayerNorm(dim, eps=1e-6)
        self.pwconv1: nn.Module = nn.Linear(
            dim, 4 * dim
        )  # point-wise/1x1 convs, implemented with linear layers
        self.act: nn.Module = nn.GELU()
        self.grn: nn.Module = GRN(4 * dim)
        self.pwconv2: nn.Module = nn.Linear(4 * dim, dim)
        self.drop_path: nn.Module = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXtV2(nn.Module):
    """ConvNeXt V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        patch_size: int = 32,
        img_size: int = 128,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: Optional[list[int]] = None,
        dims: Optional[list[int]] = None,
        drop_path_rate: float = 0.0,
        head_init_scale: float = 1.0,
        use_orig_stem: bool = False,
        args: Optional[Namespace] = None,
    ):
        super().__init__()
        self.depths = depths
        if self.depths is None:  # set default value
            self.depths = [3, 3, 9, 3]
        self.img_size = img_size
        self.use_orig_stem = use_orig_stem
        assert depths is not None
        self.num_stage = len(depths)
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layer
        self.patch_size = patch_size
        if dims is None:
            dims = [96, 192, 384, 768]

        if self.use_orig_stem:
            self.stem_orig = nn.Sequential(
                nn.Conv2d(
                    in_chans,
                    dims[0],
                    kernel_size=patch_size // (2 ** (self.num_stage - 1)),
                    stride=patch_size // (2 ** (self.num_stage - 1)),
                ),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            )
        else:
            self.initial_conv = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=1),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                nn.GELU(),
            )
            # depthwise conv for stem
            self.stem = nn.Sequential(
                nn.Conv2d(
                    dims[0],
                    dims[0],
                    kernel_size=patch_size // (2 ** (self.num_stage - 1)),
                    stride=patch_size // (2 ** (self.num_stage - 1)),
                    padding=(patch_size // (2 ** (self.num_stage - 1))) // 2,
                    groups=dims[0],
                ),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            )

        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(self.num_stage):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        if self.use_orig_stem:
            x = self.stem_orig(x)
        else:
            x = self.initial_conv(x)
            x = self.stem(x)

        x = self.stages[0](x)
        for i in range(3):
            x = self.downsample_layers[i](x)
            x = self.stages[i + 1](x)

        return x  # pool with wrapper

    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** 0.5)
        return (
            mask.reshape(-1, p, p)
            .repeat_interleave(scale, axis=1)
            .repeat_interleave(scale, axis=2)
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # no masking
        return self.forward_features(x)


class FCMAE(nn.Module):
    """Fully Convolutional Masked Autoencoder with ConvNeXtV2 backbone"""

    def __init__(
        self,
        img_size: int = 112,
        depths: list[int] = None,
        dims: list[int] = None,
        decoder_depth: int = 1,
        decoder_embed_dim: int = 512,
        patch_size: float = 16,
        mask_ratio: float = 0.6,
        norm_pix_loss: bool = False,
        args: Namespace = None,
        loss_fn=None,
        sparse: bool = True,
    ):
        super().__init__()

        print("using the multi-modal fcmae model")
        # configs
        self.args = args
        self.img_size = img_size
        if depths is None:  # set default value
            depths = [3, 3, 9, 3]
        self.depths = depths
        if dims is None:
            dims = [96, 192, 384, 768]
        self.dims = dims
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.norm_pix_loss = norm_pix_loss
        self.loss_fn = loss_fn
        self.sparse = sparse

        self.in_chans = (
            len(args.modalities["sentinel2"])
            if args.modalities["sentinel2"] != "all"
            else len(args.modalities_full["sentinel2"])
        )
        self.out_chans = {}
        for modality in self.args.modalities.keys():
            if modality in ["sentinel2", "sentinel1", "aster", "canopy_height_eth"]:
                # all the conituous pixel level modalities
                if self.args.modalities[modality] == "all":
                    self.out_chans[modality] = len(self.args.modalities_full[modality])
                else:
                    self.out_chans[modality] = len(self.args.modalities[modality])
            elif modality == "biome":
                self.out_chans[modality] = 14  # 14 biomes
            elif modality == "eco_region":
                self.out_chans[modality] = 846  # 846 eco regions
            elif modality in ["lat", "lon", "month", "era5"]:
                if self.args.modalities[modality] == "all":
                    self.out_chans[modality] = len(self.args.modalities_full[modality])
                else:
                    self.out_chans[modality] = len(self.args.modalities[modality])
            elif modality == "esa_worldcover":
                self.out_chans[modality] = 11  # 11 classes for esa worldcover
            elif modality == "dynamic_world":
                self.out_chans[modality] = 9  # 9 classes for dynamic world

        # encoder
        self.encoder = ConvNeXtV2(
            in_chans=self.in_chans,
            depths=depths,
            dims=dims,
            patch_size=patch_size,
            img_size=img_size,
            use_orig_stem=args.use_orig_stem,
        )
        self.proj = nn.Conv2d(in_channels=dims[-1], out_channels=decoder_embed_dim, kernel_size=1)

        # mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1))
        decoder = [Block(dim=decoder_embed_dim, drop_path=0.0) for _ in range(decoder_depth)]

        # creating a decoder for each modality
        self.decoder_dict = nn.ModuleDict()
        self.pred_dict = nn.ModuleDict()
        for modality in self.args.out_modalities.keys():
            if modality in [
                "sentinel2",
                "sentinel1",
                "aster",
                "canopy_height_eth",
                "dynamic_world",
                "esa_worldcover",
                "IMNET",
            ]:
                # all the pixel level modalities
                self.decoder_dict[modality] = nn.Sequential(*decoder)
                self.pred_dict[modality] = nn.Conv2d(
                    in_channels=decoder_embed_dim,
                    out_channels=patch_size**2 * self.out_chans[modality],
                    kernel_size=1,
                )
            elif modality in ["biome", "eco_region", "lat", "lon", "month", "era5"]:
                # all the non-pixel level modalities along with a global average pooling
                self.decoder_dict[modality] = nn.Sequential(*decoder)
                self.layer_norm_tmp = LayerNorm(
                    decoder_embed_dim, eps=1e-6, data_format="channels_first"
                )
                self.pred_dict[modality] = nn.Linear(
                    in_features=decoder_embed_dim, out_features=self.out_chans[modality]
                )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            trunc_normal_(w.view([w.shape[0], -1]))
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)
        if hasattr(self, "mask_token"):
            torch.nn.init.normal_(self.mask_token, std=0.02)

    def patchify(self, imgs: Tensor, modality: str) -> Tensor:
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        if modality in ["dynamic_world", "esa_worldcover"]:
            # for these modalities, we only have one channel
            channels = 1
        else:
            channels = self.out_chans[modality]
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], channels, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * channels))
        return x

    def unpatchify(self, x: Tensor) -> Tensor:
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        print("shape of x:", x.shape)
        h = w = self.img_size // p
        # assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs

    def gen_random_mask(self, x: Tensor, mask_ratio: float) -> Tensor:
        N = x.shape[0]  # number of samples
        L = (x.shape[2] // self.patch_size) ** 2  # number of patches
        len_keep = int(L * (1 - mask_ratio))  # number of patches to keep

        # the following lines generate a mask with 0s and 1s at random locations
        noise = torch.randn(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask  # (batch_size, no_patches**2)

    def upsample_mask(self, mask: Tensor, scale: float):
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** 0.5)
        return (
            mask.reshape(-1, p, p).repeat_interleave(scale, dim=1).repeat_interleave(scale, dim=2)
        )

    def forward_encoder(self, imgs: Tensor, mask_ratio: float) -> Tuple[Tensor, Tensor]:
        # generate random masks
        mask = self.gen_random_mask(imgs, mask_ratio)
        # encoding
        x = self.encoder(imgs, mask)
        return x, mask

    def forward_decoder(self, x: Tensor, mask: Tensor) -> Dict[AnyStr, Tensor]:
        pred = {}
        x = self.proj(x)
        n, c, h, w = x.shape
        mask = mask.reshape(-1, h, w).unsqueeze(1).type_as(x)
        mask_token = self.mask_token.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        x = x * (1.0 - mask) + mask_token * mask
        for modalities in self.args.out_modalities.keys():
            # decoding
            x_ = self.decoder_dict[modalities](x)
            if modalities in ["biome", "eco_region", "lat", "lon", "month", "era5"]:
                x_ = self.layer_norm_tmp(x_)
                # for the image level modalities we use global average pooling followed by the linear layer in pred_dict
                x_ = x_.mean(dim=[-2, -1])
            # pred
            pred[modalities] = self.pred_dict[modalities](x_)
        return pred

    def forward_loss(
        self, imgs_dict: Dict[AnyStr, Tensor], preds: Dict[AnyStr, Tensor], mask: Tensor
    ) -> Tuple[Tensor, Dict, Tensor, Tensor]:
        """
        imgs_dict: A dict of different modalities, each with shape of [N, C, H, W], C is the number of channels/bands
        preds: A dict of predictions for different modalities each of shape [N, L, p*p*C]
        mask: [N, L], 0 is keep, 1 is remove
        """

        loss_dict = {}
        for modality in self.args.out_modalities.keys():
            if modality in ["biome", "eco_region", "lat", "lon", "month", "era5"]:
                # all the image level modalities
                # we still further divide this into categorical and continuous modalities
                if modality in ["biome", "eco_region"]:
                    # categorical modalities
                    imgs = imgs_dict[modality]
                    pred = preds[modality]
                    imgs_classes = torch.argmax(imgs, dim=-1)
                    # we don't need to patchify the image for these modalities
                    # compute the loss
                    loss = nn.CrossEntropyLoss()(pred, imgs_classes)
                    loss_dict[modality] = loss
                elif modality in ["lat", "lon", "month", "era5"]:
                    # continuous modalities
                    imgs = imgs_dict[modality]
                    pred = preds[modality]
                    # we don't need to patchify the image for these modalities but we can still ignore any nan values
                    nan_mask = torch.isnan(imgs)
                    pred = pred[~nan_mask]
                    imgs = imgs[~nan_mask]
                    # compute the loss
                    loss = nn.MSELoss()(pred, imgs)
                    loss_dict[modality] = loss
            elif modality in ["dynamic_world", "esa_worldcover"]:
                # pixel level modalities but categorical
                imgs = imgs_dict[modality]
                pred = preds[modality]

                if len(pred.shape) == 4:
                    n, c, _, _ = pred.shape
                    pred = pred.reshape(n, c, -1)
                    pred = torch.einsum("ncl->nlc", pred)

                # pred is of the shape [N, L, C] where C is patch_size**2 * num_classes. we need to first convert this to [N, L, patch_size**2, num_classes]
                # L is the number of patches
                pred = pred.reshape(pred.shape[0], pred.shape[1], self.patch_size**2, -1)

                target = self.patchify(imgs, modality)

                # we only compute the loss on the patches where the mask is 1
                # mask is of the shape [N, L]
                # target is of the shape [N, L, patch_size**2 * num_classes]
                # pred is of the shape [N, L, patch_size**2, num_classes]
                # we need to apply the mask on target and pred for every channel

                target = target.reshape(target.shape[0], target.shape[1], self.patch_size**2, -1)
                mask_tmp = mask.unsqueeze(-1).repeat(1, 1, self.patch_size**2).unsqueeze(-1)

                target = target.reshape(target.shape[0], -1)
                pred = pred.reshape(pred.shape[0], -1, self.out_chans[modality])
                mask_tmp = mask_tmp.reshape(mask.shape[0], -1)

                # we only compute the loss on the patches where the mask is 1
                target = target[mask_tmp == 1]
                pred = pred[mask_tmp == 1]

                # we also apply a nan mask on the target and pred, since sometimes the target can be nan
                nan_mask = target == -1
                target = target[~nan_mask]
                pred = pred[~nan_mask]
                loss = nn.CrossEntropyLoss()(pred, target)
                loss_dict[modality] = loss

            elif modality == "IMNET":
                imgs = imgs_dict[modality]
                pred = preds[modality]
                if len(pred.shape) == 4:
                    n, c, _, _ = pred.shape
                    pred = pred.reshape(n, c, -1)
                    pred = torch.einsum("ncl->nlc", pred)

                target = self.patchify(imgs, modality)
                if self.norm_pix_loss:
                    mean = target.mean(dim=-1, keepdim=True)
                    var = target.var(dim=-1, keepdim=True)
                    target = (target - mean) / (var + 1.0e-6) ** 0.5
                loss = (pred - target) ** 2
                loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

                loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
                loss_dict[modality] = loss
            else:
                # pixel level modalities but continuous
                imgs = imgs_dict[modality]
                pred = preds[modality]

                if len(pred.shape) == 4:
                    n, c, _, _ = pred.shape  # [N, C, H, W]
                    pred = pred.reshape(n, c, -1)
                    pred = torch.einsum("ncl->nlc", pred)
                target = self.patchify(imgs, modality)

                if (
                    self.norm_pix_loss and modality == "sentinel2"
                ):  # we only compute the per-patch norm on sentinel2
                    mean = target.mean(dim=-1, keepdim=True)
                    var = target.var(dim=-1, keepdim=True)
                    target = (target - mean) / (var + 1.0e-6) ** 0.5

                loss = (pred - target) ** 2  # using mean squared error
                nan_mask = torch.isnan(loss)
                count = torch.count_nonzero(~nan_mask, dim=-1)
                loss[nan_mask] = 0
                loss = loss.sum(dim=-1) / count

                # uncomment the below line to compute the loss on the whole image - this results in better reconstructions, but
                # not better representations for downstream tasks
                # mask = torch.ones_like(mask)

                # counting the number of pixels where mask is 1 and loss is not nan. since we only compute the loss on these.
                # we create the nan mask again, since sometimes count can be 0.
                nan_mask = torch.isnan(loss * mask)
                tmp = loss * mask
                tmp[nan_mask] = 0
                sum_ = tmp.sum()

                count = torch.count_nonzero(tmp)
                loss = sum_ / count  # mean loss on removed patches
                loss_dict[modality] = loss

        loss_list = [loss_dict[modality] for modality in loss_dict.keys()]
        if self.args.loss_aggr == "uncertainty":
            uncertainty_loss_, log_vars = self.loss_fn(loss_list)
            loss_combined = sum(uncertainty_loss_)
            return loss_combined, loss_dict, log_vars, uncertainty_loss_
        elif self.args.loss_aggr == "unweighted":
            loss_combined = sum(loss_list)
            return loss_combined, loss_dict, None, None

    def forward(self, imgs_dict: Dict[AnyStr, Tensor], labels=None, mask_ratio: float = 0.6):
        # apply random crop to all pixel-wise modalities
        params = self.random_crop.generate_parameters(imgs_dict["sentinel2"].shape)

        # Apply the same transform to all images in the batch
        for modality in imgs_dict:
            if modality in PIXEL_WISE_MODALITIES:
                imgs_dict[modality] = self.random_crop.apply_transform(
                    imgs_dict[modality], params, None
                )

        # here imgs_dict is a dictionary with every modality, we set imgs to be the input which in this case
        # is always sentinel2.
        imgs = imgs_dict["sentinel2"]

        # convert nan to 0 for "sentinel2", "sentinel1", "aster", "canopy_height_eth".
        # This is done since the data is normalized to have a mean of 0 and std of 1. hence
        # effectively we are setting the nan values to the mean. In the case of the input,
        # setting to 0 also ensures that these values become sparse.
        for modality in imgs_dict.keys():
            if modality in ["sentinel2", "sentinel1", "aster", "canopy_height_eth"]:
                imgs_dict[modality] = torch.nan_to_num(
                    imgs_dict[modality], nan=0.0, posinf=0.0, neginf=0.0
                )

        x, mask = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(x, mask)
        loss, loss_dict, log_vars, normalized_loss_list = self.forward_loss(imgs_dict, pred, mask)
        return loss, pred, mask, loss_dict, log_vars, normalized_loss_list


def convnextv2_atto(**kwargs):
    model = FCMAE(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model


def convnextv2_femto(**kwargs):
    model = FCMAE(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model


def convnextv2_pico(**kwargs):
    model = FCMAE(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model


def convnextv2_nano(**kwargs):
    model = FCMAE(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model


def convnextv2_tiny(**kwargs):
    model = FCMAE(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def convnextv2_base(**kwargs):
    model = FCMAE(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model


def convnextv2_large(**kwargs):
    model = FCMAE(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model


def convnextv2_huge(**kwargs):
    model = FCMAE(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model
