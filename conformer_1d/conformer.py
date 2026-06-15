"""
Conformer Encoder for 1D Time-Domain Signals.

Macaron-style Conformer blocks with convolution module, as described in:
"Conformer: Convolution-augmented Transformer for Speech Recognition" (Gulati et al., 2020)

Adapted for radar I/Q signal processing:
- Input: (B, 2, 8000) — I and Q channels
- Output: (B, embed_dim) — feature vector for contrastive learning
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sub-sampling front-end
# ---------------------------------------------------------------------------

class SubSamplingFrontEnd(nn.Module):
    """3-layer Conv1d stack that reduces sequence length by ~12x.

    Input:  (B, 2, 8000)
    Output: (B, 256, ~667)
    """

    def __init__(self, in_channels: int = 2, out_channels: int = 256):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=3, padding=3)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → (B, out_channels, T')"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x


# ---------------------------------------------------------------------------
# Conformer sub-modules
# ---------------------------------------------------------------------------

class FFNModule(nn.Module):
    """Position-wise feed-forward with Swish activation and dropout."""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        inner_dim = dim * expansion
        self.layer_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, inner_dim)
        self.swish = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(inner_dim, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D)"""
        residual = x
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x + residual


class MHSA(nn.Module):
    """Multi-Head Self-Attention with Pre-LN."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.layer_norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D)"""
        residual = x
        x = self.layer_norm(x)
        B, T, D = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, nh, T, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, T, D)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x + residual


class ConvolutionModule(nn.Module):
    """Conformer convolution module: pointwise + GLU → depthwise → BN → Swish → pointwise.

    This is the key innovation of Conformer — it captures local patterns
    that self-attention may miss.
    """

    def __init__(self, dim: int, kernel_size: int = 15, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)  # halves channel dim: 2D → D
        self.depthwise_conv = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=dim,  # depthwise
        )
        self.batch_norm = nn.BatchNorm1d(dim)
        self.swish = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D)"""
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)              # (B, T, D) → (B, D, T)
        x = self.pointwise_conv1(x)        # (B, 2D, T)
        x = self.glu(x)                    # (B, D, T)
        x = self.depthwise_conv(x)         # (B, D, T)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)        # (B, D, T)
        x = self.dropout(x)
        x = x.transpose(1, 2)              # (B, D, T) → (B, T, D)
        return x + residual


# ---------------------------------------------------------------------------
# Conformer Block (macaron style)
# ---------------------------------------------------------------------------

class ConformerBlock(nn.Module):
    """Macaron-style Conformer block.

    Architecture:
        LN → ½FFN → LN → MHSA → LN → Conv → LN → ½FFN → LN
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        ffn_expansion: int = 4,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ffn1 = FFNModule(dim, ffn_expansion, dropout)
        self.mhsa = MHSA(dim, num_heads, dropout)
        self.conv = ConvolutionModule(dim, conv_kernel_size, dropout)
        self.ffn2 = FFNModule(dim, ffn_expansion, dropout)
        self.layer_norm = nn.LayerNorm(dim)

        # Halve the FFN residual contribution (macaron half-step)
        self.ffn_scale = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D)"""
        # First macaron half-step FFN
        x = x + self.ffn_scale * self.ffn1(x)

        # MHSA
        x = self.mhsa(x)

        # Convolution
        x = self.conv(x)

        # Second macaron half-step FFN
        x = x + self.ffn_scale * self.ffn2(x)

        # Final layer norm
        x = self.layer_norm(x)
        return x


# ---------------------------------------------------------------------------
# Full Conformer Encoder
# ---------------------------------------------------------------------------

class ConformerEncoder(nn.Module):
    """Complete 1D Conformer encoder for time-domain signal feature extraction.

    Pipeline:
        subsampling(2,8000) → permute → linear_proj → pos_encoding
        → N × ConformerBlock → LayerNorm → mean pool → output_proj

    Input:  (B, 2, 8000)   — I/Q time-domain signal
    Output: (B, embed_dim)  — feature vector
    """

    def __init__(
        self,
        in_channels: int = 2,
        input_len: int = 8000,
        hidden_dim: int = 256,
        num_blocks: int = 6,
        num_heads: int = 4,
        ffn_expansion: int = 4,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
        embed_dim: int = 512,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        # Sub-sampling: (B, 2, 8000) → (B, hidden_dim, ~667)
        self.subsampling = SubSamplingFrontEnd(in_channels, hidden_dim)

        # Compute expected output length after sub-sampling
        # Conv1d output len = floor((L + 2*pad - kernel) / stride + 1)
        def conv_out_len(L, k, s, p):
            return math.floor((L + 2 * p - k) / s + 1)

        L1 = conv_out_len(input_len, 7, 3, 3)
        L2 = conv_out_len(L1, 5, 2, 2)
        L3 = conv_out_len(L2, 3, 2, 1)
        self.subsampled_len = L3

        # Linear projection (redundant if hidden_dim == out_channels, but keeps flexibility)
        self.linear_proj = nn.Linear(hidden_dim, hidden_dim)

        # Sinusoidal positional encoding
        self.pos_encoding = self._make_sinusoidal(max_seq_len, hidden_dim)

        # Conformer blocks
        self.blocks = nn.ModuleList([
            ConformerBlock(hidden_dim, num_heads, ffn_expansion, conv_kernel_size, dropout)
            for _ in range(num_blocks)
        ])

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, embed_dim)

        self._init_weights()
        print(f"ConformerEncoder: {input_len} → {self.subsampled_len} frames, "
              f"{num_blocks} blocks, {hidden_dim} dim, {embed_dim} output")

    @staticmethod
    def _make_sinusoidal(max_len: int, dim: int) -> nn.Parameter:
        """Create sinusoidal positional encoding (non-learned)."""
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)  # (1, max_len, dim)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, 8000) — I/Q time-domain signal

        Returns:
            (B, embed_dim) — feature vector
        """
        # Sub-sampling: (B, 2, 8000) → (B, hidden_dim, T')
        x = self.subsampling(x)

        # Permute: (B, D, T) → (B, T, D)
        x = x.transpose(1, 2)

        # Linear projection
        x = self.linear_proj(x)

        # Add positional encoding (truncate/pad as needed)
        T = x.shape[1]
        if T <= self.pos_encoding.shape[1]:
            x = x + self.pos_encoding[:, :T, :]
        else:
            # Should not happen with 8000 input, but be safe
            x = x + self.pos_encoding[:, :T, :]  # pos_encoding was made for max_seq_len=1024 ≥ 667

        # Conformer blocks
        for block in self.blocks:
            x = block(x)

        # Output norm + global mean pooling
        x = self.output_norm(x)
        x = x.mean(dim=1)  # (B, hidden_dim)

        # Project to embed_dim
        x = self.output_proj(x)  # (B, embed_dim)
        return x


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing ConformerEncoder...")
    encoder = ConformerEncoder(
        in_channels=2,
        input_len=8000,
        hidden_dim=256,
        num_blocks=6,
        num_heads=4,
        embed_dim=512,
    )
    dummy = torch.randn(2, 2, 8000)
    with torch.no_grad():
        out = encoder(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in encoder.parameters()):,}")

    # Check L2 norm (should be reasonable, not exploding)
    print(f"Output norm: {out.norm(dim=-1).mean():.2f}")
    print("Done!")
