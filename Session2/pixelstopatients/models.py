from __future__ import annotations

import torch
from torch import nn


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm3d(out_channels),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm3d(out_channels),
        nn.LeakyReLU(0.1, inplace=True),
    )


class TinyUNet3D(nn.Module):
    """A compact 3D U-Net for workshop-scale CT segmentation examples."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, features: int = 8) -> None:
        super().__init__()
        self.enc1 = _block(in_channels, features)
        self.down1 = nn.MaxPool3d(2)
        self.enc2 = _block(features, features * 2)
        self.down2 = nn.MaxPool3d(2)
        self.bottleneck = _block(features * 2, features * 4)
        self.up2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec2 = _block(features * 4, features * 2)
        self.up1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.dec1 = _block(features * 2, features)
        self.head = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)

