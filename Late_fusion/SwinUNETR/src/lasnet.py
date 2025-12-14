from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from monai.networks.blocks import (
    Convolution,
    PatchEmbed,
    UnetOutBlock,
    UnetrBasicBlock,
    UnetrUpBlock,
)
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, optional_import

try:
    from modules import SwinTransformer
except:
    from .modules import SwinTransformer

rearrange, _ = optional_import("einops", name="rearrange")


class LASNet(nn.Module):
    """
    Longitudinally-aware segmentation network (LAS-Net) based on the implemtantion of Swin UNETR in MONAI
    <https://docs.monai.io/en/stable/_modules/monai/networks/nets/swin_unetr.html>
    """

    def __init__(
        self,
        img_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        patch_size: Sequence[int] | int = 2,
        feature_size: Sequence[int] | int = 48,
        embed_dim: int = 64,
        num_heads: Sequence[int] = [4, 8, 16, 32],
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        spatial_dims: int = 3,
        downsample="merging",
        deep_supr_num: int = 1,
        use_checkpoint: bool = True,
        use_v2: bool = True,
    ) -> None:

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)  # 7*7*7
        patch_size = ensure_tuple_rep(2, spatial_dims)
        feature_size = ensure_tuple_rep(feature_size, 2)

        if not (spatial_dims == 3):
            raise ValueError("spatial dimension should be 3.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        self.normalize = normalize
        self.deep_supr_num = deep_supr_num

        # level 0
        self.encoder_head_0 = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=1,
                out_channels=feature_size[0],
                kernel_size=(3, 3, 1),
                adn_ordering="NA",
                act=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
                # same wtih UnetrBasicBlock
            ),
            UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size[0],
                out_channels=feature_size[0],
                kernel_size=(3, 3, 1),
                stride=1,
                norm_name=norm_name,
                res_block=True,
            ),
        )
        self.encoder_head_ref_0 = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels - 1,
                out_channels=feature_size[0],
                kernel_size=(3, 3, 1),
                adn_ordering="NA",
                act=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
                # same wtih UnetrBasicBlock
            ),
            UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size[0],
                out_channels=feature_size[0],
                kernel_size=(3, 3, 1),
                stride=1,
                norm_name=norm_name,
                res_block=True,
            ),
        )

        # level 1
        self.encoder_head_1 = nn.Sequential(
            UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size[0],
                out_channels=feature_size[1],
                kernel_size=3,
                stride=(2, 2, 1),
                norm_name=norm_name,
                res_block=True,
            ),
            UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size[1],
                out_channels=feature_size[1],
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=True,
            ),
        )

        self.encoder_head_ref_1 = nn.Sequential(
            UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size[0],
                out_channels=feature_size[1],
                kernel_size=3,
                stride=(2, 2, 1),
                norm_name=norm_name,
                res_block=True,
            ),
            UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size[1],
                out_channels=feature_size[1],
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=True,
            ),
        )

        self.swinViT = SwinTransformer(
            in_chans=feature_size[1],
            embed_dim=embed_dim,
            window_size=window_size,
            patch_size=patch_size,
            num_heads=num_heads,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            spatial_dims=spatial_dims,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            use_v2=use_v2,
        )

        # level 0
        self.encoder0 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size[0] * 2,
            out_channels=feature_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        # level 1
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size[1] * 2,
            out_channels=feature_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * embed_dim,
            out_channels=2 * embed_dim,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * embed_dim,
            out_channels=4 * embed_dim,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.bottleneck = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * embed_dim,
            out_channels=8 * embed_dim,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=embed_dim * 8,
            out_channels=embed_dim * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=embed_dim * 2,
            out_channels=embed_dim,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=embed_dim,
            out_channels=feature_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder0 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size[1],
            out_channels=feature_size[0],
            kernel_size=3,
            upsample_kernel_size=(2, 2, 1),
            norm_name=norm_name,
            res_block=True,
        )

        feature_size_list = [
            feature_size[0],
            feature_size[1],
            embed_dim,
            2 * embed_dim,
            4 * embed_dim,
            8 * embed_dim,
        ]
        self.out = nn.ModuleList(
            [
                UnetOutBlock(
                    spatial_dims=spatial_dims,
                    in_channels=feature_size_list[i],
                    out_channels=out_channels,
                )
                for i in range(deep_supr_num)
            ]
        )

    def forward(self, x_in):
        # decode the hidden states
        x = x_in[:, -1:, ...]  # planning CT images
        x_ref = x_in[:, :-1, ...]  # prior PET/CT images

        x = self.encoder_head_0(x)
        x_ref = self.encoder_head_ref_0(x_ref)
        enc0 = self.encoder0(torch.cat([x, x_ref], dim=1))

        x1 = self.encoder_head_1(x)
        x1_ref = self.encoder_head_ref_1(x_ref)
        enc1 = self.encoder1(torch.cat([x1, x1_ref], dim=1))

        hidden_states_out = self.swinViT(x1, x1_ref, self.normalize)

        enc2 = self.encoder2(hidden_states_out[0])
        enc3 = self.encoder3(hidden_states_out[1])
        enc4 = self.encoder4(hidden_states_out[2])

        dec5 = self.bottleneck(hidden_states_out[3])

        decs: list[torch.Tensor] = []
        dec4 = self.decoder4(dec5, enc4)
        decs.append(dec4)
        dec3 = self.decoder3(dec4, enc3)
        decs.append(dec3)
        dec2 = self.decoder2(dec3, enc2)
        decs.append(dec2)
        dec1 = self.decoder1(dec2, enc1)
        decs.append(dec1)
        dec0 = self.decoder0(dec1, enc0)
        decs.append(dec0)

        decs.reverse()  # the first element is the output of the last layer
        outs: list[torch.Tensor] = []
        for i in range(self.deep_supr_num):
            outs.append(self.out[i](decs[i]))

        if not self.training or len(outs) == 1:
            return outs[0]
        return outs
