"""Visual backbone feature extractor for 2.5D MRI windows.
Extracts deep spatial feature representations from 3-channel slice stacks (center slice ± 1).
Supports ConvNeXt / ResNet / CoAtNet style blocks with center-channel initializations.
"""

from typing import Tuple, Optional
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

if torch is not None:
    class ResidualBlock(nn.Module):
        """Standard 2D Residual block for feature extraction."""
        def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
                
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            return F.relu(out)

    class MRIBackbone(nn.Module):
        """Backbone for 2.5D MRI windows (3 adjacent slices: center ± 1).
        Projects (B_win, 3, H, W) -> (B_win, feature_dim).
        """
        def __init__(self, in_channels: int = 3, feature_dim: int = 512):
            super().__init__()
            self.in_channels = in_channels
            self.feature_dim = feature_dim
            
            # Stem conv: 3 -> 64
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            
            # Initialize stem weights: center slice prominence (Silver solution insight)
            with torch.no_grad():
                w = self.stem[0].weight.data  # (64, 3, 7, 7)
                w_sum = w.sum(dim=1, keepdim=True)
                w[:, 0:1, :, :] = 0.05 * w[:, 0:1, :, :]
                w[:, 1:2, :, :] = 0.90 * w_sum  # Center slice receives 90% initial weight
                w[:, 2:3, :, :] = 0.05 * w[:, 2:3, :, :]
                
            # Stages: 64 -> 128 -> 256 -> 512
            self.stage1 = nn.Sequential(ResidualBlock(64, 128, stride=2), ResidualBlock(128, 128))
            self.stage2 = nn.Sequential(ResidualBlock(128, 256, stride=2), ResidualBlock(256, 256))
            self.stage3 = nn.Sequential(ResidualBlock(256, feature_dim, stride=2), ResidualBlock(feature_dim, feature_dim))
            
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(feature_dim, feature_dim)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Args:
                x: Tensor of shape (B_windows, 3, H, W)
            Returns:
                Tensor of shape (B_windows, feature_dim)
            """
            feats = self.stem(x)
            feats = self.stage1(feats)
            feats = self.stage2(feats)
            feats = self.stage3(feats)
            pooled = self.global_pool(feats).flatten(1)
            out = F.relu(self.fc(pooled))
            return out

