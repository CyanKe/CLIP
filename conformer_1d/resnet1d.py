"""
ResNet-18 1D Encoder for Time-Domain Signals.

Mirrors the standard ResNet-18 architecture with all 2D convolutions replaced
by 1D convolutions, adapted for radar I/Q signal processing:

Input:  (B, 2, 8000)  — I and Q channels
Output: (B, embed_dim) — feature vector for contrastive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# BasicBlock (1D version)
# ---------------------------------------------------------------------------

class BasicBlock1D(nn.Module):
    """ResNet BasicBlock with 1D convolutions.

    Two Conv1d-BN-ReLU layers with a residual connection.
    If stride != 1 or in_channels != out_channels, the shortcut uses
    a 1×1 Conv1d to match dimensions.
    """

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut: 1×1 conv when dimensions don't match
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → (B, C_out, T_out)"""
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = F.relu(out)
        return out


# ---------------------------------------------------------------------------
# ResNet1D Encoder
# ---------------------------------------------------------------------------

class ResNet1DEncoder(nn.Module):
    """ResNet-18 1D encoder for time-domain signal feature extraction.

    Pipeline:
        stem(Conv1d → BN → ReLU → MaxPool1d)
        → layer1 (2× BasicBlock1D, 64 ch)
        → layer2 (2× BasicBlock1D, 128 ch, stride=2)
        → layer3 (2× BasicBlock1D, 256 ch, stride=2)
        → layer4 (2× BasicBlock1D, 512 ch, stride=2)
        → AdaptiveAvgPool1d(1) → flatten → Linear(512, embed_dim)

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

        # Stem: initial downsampling
        # (B, 2, 8000) → (B, 64, 4000) → (B, 64, 1333)
        self.stem_conv = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
        self.stem_bn = nn.BatchNorm1d(64)
        self.stem_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # Output
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Linear(512, embed_dim)

        self._init_weights()
        print(f"ResNet1DEncoder: {sum(p.numel() for p in self.parameters()):,} params, "
              f"{embed_dim} output dim")

    def _make_layer(self, in_ch: int, out_ch: int, blocks: int, stride: int):
        layers = [BasicBlock1D(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

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
        # Stem
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = F.relu(x)
        x = self.stem_pool(x)

        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Pooling and projection
        x = self.avg_pool(x)         # (B, 512, 1)
        x = x.squeeze(-1)            # (B, 512)
        x = self.output_proj(x)      # (B, embed_dim)
        return x


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing ResNet1DEncoder...")
    encoder = ResNet1DEncoder(
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
