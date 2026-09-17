"""timm image backbone returning a spatial feature map (B, C, H, W) for every architecture."""

from __future__ import annotations

import math

import timm
import torch
from torch import nn


class ImageEncoder(nn.Module):
    def __init__(self, backbone: str = "convnext_tiny", pretrained: bool = True,
                 freeze_stages: int = 0):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0,
                                          global_pool="")
        self.num_features: int = self.backbone.num_features
        self.is_vit = not hasattr(self.backbone, "stages")
        if freeze_stages > 0 and not self.is_vit:
            frozen = [self.backbone.stem, *self.backbone.stages[:freeze_stages]]
            for module in frozen:
                for p in module.parameters():
                    p.requires_grad = False

    @property
    def gradcam_layer(self) -> nn.Module:
        """Layer whose activations/gradients Grad-CAM uses (last spatial stage)."""
        if self.is_vit:
            return self.backbone.blocks[-1]
        return self.backbone.stages[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)
        if feats.ndim == 4:
            return feats
        # ViT: drop prefix (CLS/register) tokens and fold patches back into a grid.
        n_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
        tokens = feats[:, n_prefix:]
        side = int(math.isqrt(tokens.shape[1]))
        return tokens.transpose(1, 2).reshape(tokens.shape[0], -1, side, side)
