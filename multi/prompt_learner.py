"""
Feature-Conditioned Learnable Context (CoOp-style)

用域感知 Meta-Net 将各域信号特征映射为连续上下文向量，
注入到冻结文本 Transformer 中，使文本特征随输入信号物理特性动态调整。

参考: CoOp - Learning Prompt for Vision-Language Models (IJCV 2022)
"""

import torch
import torch.nn as nn


class DomainMetaNet(nn.Module):
    """单域特征 → 上下文 token 向量

    每个 Domain 一个 2 层 MLP，将域特征映射为 n_ctx 个上下文 token。
    最后一层零初始化，保证训练初期 context 接近零向量，不破坏预训练表征。
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_ctx: int, transformer_width: int):
        super().__init__()
        self.n_ctx = n_ctx
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_ctx * transformer_width),
        )
        # 零初始化最后一层，保证初期 context ≈ 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)  # [B, n_ctx * transformer_width]
        return out.view(x.shape[0], self.n_ctx, -1)  # [B, n_ctx, transformer_width]


class FeatureConditionedPromptLearner(nn.Module):
    """管理所有域的 Meta-Net，生成完整的上下文向量序列

    5 个信号特征域:
    - time (5): skewness, kurtosis, envelope_variation, modulation_bandwidth, modulation_rate
    - freq (4): spectral_skewness, spectral_kurtosis, carrier_factor, awgn_factor
    - bispectrum (2): bispectrum_variance, bispectrum_mean
    - wavelet (8): variance, mean, max, scale_centroid, max_singular_value, central_moment_2/3/4
    - statistical (3): shannon_entropy, exponential_entropy, norm_entropy
    """

    DOMAIN_ORDER = ['time', 'freq', 'bispectrum', 'wavelet', 'statistical']
    DOMAIN_DIMS = {'time': 5, 'freq': 4, 'bispectrum': 2, 'wavelet': 8, 'statistical': 3}
    # JSON 中的域前缀 → 内部域名的映射
    DOMAIN_JSON_PREFIX = {
        'time': 'time_domain',
        'freq': 'freq_domain',
        'bispectrum': 'bispectrum',
        'wavelet': 'wavelet',
        'statistical': 'statistical',
    }

    def __init__(
        self,
        transformer_width: int,
        n_ctx_per_domain: dict = None,
        hidden_dim: int = None,
    ):
        super().__init__()
        if n_ctx_per_domain is None:
            n_ctx_per_domain = {d: 2 for d in self.DOMAIN_ORDER}
            n_ctx_per_domain['bispectrum'] = 1  # 2 维特征只需 1 个 token
            n_ctx_per_domain['wavelet'] = 3  # 8 维特征用 3 个 token

        self.n_ctx_per_domain = n_ctx_per_domain
        self.total_n_ctx = sum(n_ctx_per_domain.values())

        hidden = hidden_dim or (transformer_width // 2)

        self.meta_nets = nn.ModuleDict()
        for domain in self.DOMAIN_ORDER:
            self.meta_nets[domain] = DomainMetaNet(
                in_dim=self.DOMAIN_DIMS[domain],
                hidden_dim=hidden,
                n_ctx=n_ctx_per_domain[domain],
                transformer_width=transformer_width,
            )

    def forward(self, features_dict: dict) -> torch.Tensor:
        """生成上下文向量序列

        Args:
            features_dict: {'time': [B, 5], 'freq': [B, 4], ...}
                          各域特征张量（已标准化）

        Returns:
            context_vectors: [B, total_n_ctx, transformer_width]
        """
        ctx_parts = []
        for domain in self.DOMAIN_ORDER:
            ctx = self.meta_nets[domain](features_dict[domain])  # [B, n_ctx_d, D]
            ctx_parts.append(ctx)
        return torch.cat(ctx_parts, dim=1)  # [B, total_n_ctx, D]
