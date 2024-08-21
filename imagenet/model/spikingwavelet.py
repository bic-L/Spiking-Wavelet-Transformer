from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from module import *

backend = 'torch'


class SpikingWaveletTransformer(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        in_channels=2,
        num_classes=11,
        embed_dims=512,
        mlp_ratios=4,
        drop_rate=0.0,
        drop_path_rate=0.0,
        depths=[6, 8, 6],
        T=4,
        pooling_stat="1111",
        haar_vth=1.,
        FL_blocks = 16,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule

        patch_embed = MS_SPS(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            pooling_stat=pooling_stat,
        )

        blocks = nn.ModuleList(
            [
                MS_Block_Conv(
                    dim=embed_dims,
                    mlp_ratio=mlp_ratios,
                    haar_vth=haar_vth,
                    FL_blocks = FL_blocks,
                )
                for j in range(depths)
            ]
        )

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", blocks)

        # classification head
        self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.head = (nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x = patch_embed(x)
        for blk in block:
            x = blk(x)

        x = x.flatten(3).mean(3)
        return x

    def forward(self, x):
        if len(x.shape) < 5:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        else:
            x = x.transpose(0, 1).contiguous()

        x = self.forward_features(x)

        x = self.head_lif(x)
        x = self.head(x)
        x = x.mean(0)

        return x


@register_model
def swformer(pretrained = False, pretrained_cfg=None, **kwargs):
    model = SpikingWaveletTransformer(
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model
