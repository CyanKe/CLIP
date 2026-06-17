"""
Simple CNN-1D Encoder for Time-Domain Signals.

A lightweight, progressively deeper Conv1d stack for radar I/Q signal
classification. 5 conv blocks with increasing channel depth, designed as
a simple but solid baseline for comparison with Conformer and ResNet1D.

Input:  (B, 2, 8000)  — I and Q channels
Output: (B, embed_dim) — feature vector for contrastive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Conv Block
# ---------------------------------------------------------------------------

class ConvBlock1D(nn.Module):
    """Conv1d → BatchNorm1d → ReLU → MaxPool1d."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, pool: int = 2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(kernel_size=pool, stride=pool) if pool > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → (B, C_out, T_out)"""
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


# ---------------------------------------------------------------------------
# SimpleCNN1D Encoder
# ---------------------------------------------------------------------------

class SimpleCNN1DEncoder(nn.Module):
    """Simple stacked CNN-1D encoder for time-domain signal feature extraction.

    Architecture:
        5 × ConvBlock1D (progressive channel doubling + pooling)
        → AdaptiveAvgPool1d(1)
        → Flatten
        → Linear(512, embed_dim)

    Channel progression: 2 → 32 → 64 → 128 → 256 → 512
    Pooling: ×2 each block → total downsampling 32×  (8000 → ~250 → pool → 1)

    Input:  (B, 2, 8000)   — I/Q time-domain signal
    Output: (B, embed_dim)  — feature vector
    """

    def __init__(
        self,
        in_channels: int = 2,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Channel progression & kernel sizes
        channels = [in_channels, 32, 64, 128, 256, 512]
        kernels = [7, 5, 5, 3, 3]

        self.blocks = nn.ModuleList([
            ConvBlock1D(channels[i], channels[i + 1], kernel_size=kernels[i], pool=2)
            for i in range(5)
        ])

        # Output
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Linear(512, embed_dim)

        self._init_weights()
        print(f"SimpleCNN1DEncoder: {sum(p.numel() for p in self.parameters()):,} params, "
              f"{embed_dim} output dim")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, 8000) — I/Q time-domain signal

        Returns:
            (B, embed_dim) — feature vector
        """
        for block in self.blocks:
            x = block(x)

        # Global pooling and projection
        x = self.avg_pool(x)         # (B, 512, 1)
        x = x.squeeze(-1)            # (B, 512)
        x = self.output_proj(x)      # (B, embed_dim)
        return x


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing SimpleCNN1DEncoder...")
    encoder = SimpleCNN1DEncoder(
        in_channels=2,
        embed_dim=512,
    )
    dummy = torch.randn(2, 2, 8000)
    with torch.no_grad():
        out = encoder(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Output norm: {out.norm(dim=-1).mean():.2f}")

    # Test with batch_size=1
    single = torch.randn(1, 2, 8000)
    with torch.no_grad():
        out_single = encoder(single)
    print(f"Single output: {out_single.shape}")
    print("Done!")
