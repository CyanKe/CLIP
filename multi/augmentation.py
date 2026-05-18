"""
STFT 数据增强模块 — 针对雷达干扰 CLIP CZSL 的 shortcut learning 问题

包含四种增强策略:
1. SpecAugment      — 时频带掩码 (最稳定, 物理意义最明确)
2. PatchRandomMasking — ViT patch-grid 对齐的矩形块掩码 (破坏几何连续性)
3. EnergyAwareMasking — 针对高能量前景区域的概率掩码 (强行压制欺骗特征)
4. AsymmetricAugmentation — 对纯压制样本施加更强增强

所有增强在 CLIP 标准化之后应用 (mean≈0, mask with 0 = "no information").
"""

import random
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, List


# ============================================================================
# 1. SpecAugment — 时频带掩码
# ============================================================================

class SpecAugment(nn.Module):
    """
    STFT 时频带掩码。

    在时间轴 (dim 1, rows) 和频率轴 (dim 2, cols) 上随机掩码连续区间。
    横轴=时间, 纵轴=频率 的约定: 时间掩码遮住连续行, 频率掩码遮住连续列。
    两个方向都掩码, 无论显示约定如何都能覆盖。

    应用在 CLIP 标准化之后 (值域约 [-2, 2], 均值为 0, 所以 0 = 无信息).
    """

    def __init__(
        self,
        time_mask_param: float = 0.1,
        freq_mask_param: float = 0.1,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
        mask_value: float = 0.0,
        p: float = 1.0,
    ):
        """
        Args:
            time_mask_param: 时间掩码最大宽度 (占 H 的比例)
            freq_mask_param: 频率掩码最大高度 (占 W 的比例)
            num_time_masks: 时间掩码条数
            num_freq_masks: 频率掩码条数
            mask_value: 掩码填充值 (0 = CLIP 均值)
            p: 应用概率
        """
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.mask_value = mask_value
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [C, H, W] post-CLIP-norm tensor
        Returns:
            masked x
        """
        if self.p < 1.0 and torch.rand(1).item() > self.p:
            return x

        C, H, W = x.shape

        # 时间掩码 (遮住连续行, dim 1)
        for _ in range(self.num_time_masks):
            t = random.randint(0, max(1, int(H * self.time_mask_param)))
            t0 = random.randint(0, H - t)
            x[:, t0:t0 + t, :] = self.mask_value

        # 频率掩码 (遮住连续列, dim 2)
        for _ in range(self.num_freq_masks):
            f = random.randint(0, max(1, int(W * self.freq_mask_param)))
            f0 = random.randint(0, W - f)
            x[:, :, f0:f0 + f] = self.mask_value

        return x


# ============================================================================
# 2. PatchRandomMasking — ViT patch-grid 对齐的矩形块掩码
# ============================================================================

class PatchRandomMasking(nn.Module):
    """
    以 ViT 的 patch 为最小单位随机掩码整块区域。

    通过 reshape 实现完全向量化 (无 Python 循环), 高效兼容 DataLoader 多进程。
    """

    def __init__(
        self,
        patch_size: Tuple[int, int] = (16, 16),
        img_size: int = 224,
        mask_ratio: float = 0.3,
        mask_value: float = 0.0,
        p: float = 1.0,
    ):
        """
        Args:
            patch_size: (patch_height, patch_width)
            img_size: 图像尺寸 (正方形)
            mask_ratio: 掩码 patch 的比例 (0.0-1.0)
            mask_value: 填充值
            p: 应用概率
        """
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.p = p

        self.grid_h = img_size // patch_size[0]
        self.grid_w = img_size // patch_size[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [C, H, W]
        Returns:
            masked x
        """
        if self.p < 1.0 and torch.rand(1).item() > self.p:
            return x

        C, H, W = x.shape
        ph, pw = self.patch_size
        grid_h, grid_w = self.grid_h, self.grid_w

        # [C, H, W] -> [C, grid_h, ph, grid_w, pw] -> [C, grid_h, grid_w, ph, pw]
        x_r = x.view(C, grid_h, ph, grid_w, pw).permute(0, 1, 3, 2, 4)

        # 生成保留 mask: True = 保留, False = 丢弃
        keep_mask = torch.rand(grid_h, grid_w, device=x.device) > self.mask_ratio
        keep_mask = keep_mask.view(1, grid_h, grid_w, 1, 1).expand_as(x_r)

        fill = torch.tensor(self.mask_value, dtype=x.dtype, device=x.device)
        x_r = torch.where(keep_mask, x_r, fill)

        # 恢复形状
        x = x_r.permute(0, 1, 3, 2, 4).contiguous().view(C, H, W)
        return x


# ============================================================================
# 3. EnergyAwareMasking — 能量感知掩码
# ============================================================================

class EnergyAwareMasking(nn.Module):
    """
    针对高能量像素 (通常对应欺骗干扰前景) 进行概率掩码。

    使用 torch.quantile 高效计算阈值, 完全向量化。
    mask_prob 不应设为 1.0, 保留 0.5~0.8 让模型保留一定 "透视" 能力。
    """

    def __init__(
        self,
        top_percentile: float = 20.0,
        mask_prob: float = 0.8,
        mask_value: float = 0.0,
        p: float = 1.0,
    ):
        """
        Args:
            top_percentile: 高能量百分位阈值 (如前 20% = 80th percentile 以上)
            mask_prob: 对高能量像素的掩码概率 (0.5~0.8 推荐)
            mask_value: 填充值
            p: 应用概率
        """
        super().__init__()
        self.top_percentile = top_percentile
        self.mask_prob = mask_prob
        self.mask_value = mask_value
        self.p = p

    def forward(self, x: torch.Tensor, energy_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [C, H, W] post-CLIP-norm tensor
            energy_map: [H, W] pre-CLIP-norm magnitude (值域约 [0, 1])
        Returns:
            masked x
        """
        if self.p < 1.0 and torch.rand(1).item() > self.p:
            return x

        # 计算能量阈值: top N% 意味着取 (100-N)th percentile
        q = 1.0 - (self.top_percentile / 100.0)
        threshold = torch.quantile(energy_map.view(-1), q)

        # 高能量区域
        high_energy = energy_map > threshold  # [H, W] bool

        # 以 mask_prob 概率随机决定是否真的掩码
        apply_mask = torch.rand_like(energy_map) < self.mask_prob
        final_mask = high_energy & apply_mask  # [H, W] bool

        # 扩展到 3 通道并应用
        final_mask = final_mask.unsqueeze(0).expand_as(x)
        x[final_mask] = self.mask_value

        return x


# ============================================================================
# 4. AsymmetricAugmentation — 非对称增强
# ============================================================================

class AsymmetricAugmentation(nn.Module):
    """
    对纯压制干扰样本施加更强的增强。

    逻辑: 纯压制样本没有欺骗前景, 额外增加 mask ratio 可以迫使模型
    从碎片化的背景纹理中学习更全局的底噪特征, 防止对局部特征过拟合。
    """

    DECEPTION_CLASSES = {"DFTJ", "ISRJ", "SMSPJ", "C&IJ", "CSJ"}

    def __init__(
        self,
        extra_patch_mask_ratio: float = 0.2,
        extra_specaug_ratio: float = 0.05,
    ):
        """
        Args:
            extra_patch_mask_ratio: 纯压制样本额外增加的 patch mask 比例
            extra_specaug_ratio: 纯压制样本额外增加的 specaug 参数
        """
        super().__init__()
        self.extra_patch_mask_ratio = extra_patch_mask_ratio
        self.extra_specaug_ratio = extra_specaug_ratio

    @classmethod
    def is_suppression_only(cls, metadata: Optional[dict]) -> bool:
        """判断样本是否只有压制干扰 (无欺骗干扰)"""
        if metadata is None:
            return False

        jam_types = metadata.get('jam_types', [])
        if isinstance(jam_types, str):
            jam_types = [jam_types] if jam_types else []
        elif not isinstance(jam_types, list):
            return False

        if not jam_types:
            return False

        has_deception = any(
            t.strip() in cls.DECEPTION_CLASSES if isinstance(t, str) else str(t) in cls.DECEPTION_CLASSES
            for t in jam_types
        )
        has_any = len(jam_types) > 0

        return has_any and not has_deception

    def forward(self, metadata: Optional[dict] = None) -> dict:
        """
        Args:
            metadata: 样本 metadata 字典 (含 jam_types)
        Returns:
            overrides dict: {'extra_patch_mask_ratio': float, 'extra_specaug_ratio': float}
            如果非纯压制样本, 返回全零
        """
        if self.is_suppression_only(metadata):
            return {
                'extra_patch_mask_ratio': self.extra_patch_mask_ratio,
                'extra_specaug_ratio': self.extra_specaug_ratio,
            }
        return {'extra_patch_mask_ratio': 0.0, 'extra_specaug_ratio': 0.0}


# ============================================================================
# 5. STFTAugmentation — 组合增强管线
# ============================================================================

class STFTAugmentation(nn.Module):
    """
    可组合的 STFT 数据增强管线。

    通过 config dict 控制各子增强的启用/禁用和参数。
    所有子增强均在 __getitem__ 中对单张图像 [C, H, W] 操作。

    Usage:
        aug = STFTAugmentation(config['augmentation'])
        augmented, _ = aug(stft_tensor, metadata=meta, energy_map=energy_map)
    """

    def __init__(self, config: dict):
        """
        Args:
            config: augmentation 配置字典 (来自 config.yaml 的 augmentation 段)
        """
        super().__init__()
        self.config = config
        self.transforms = nn.ModuleDict()
        self._asymmetric = None

        # --- SpecAugment ---
        sa_cfg = config.get('specaugment', {})
        if sa_cfg.get('enabled', False):
            self.transforms['specaugment'] = SpecAugment(
                time_mask_param=sa_cfg.get('time_mask_param', 0.1),
                freq_mask_param=sa_cfg.get('freq_mask_param', 0.1),
                num_time_masks=sa_cfg.get('num_time_masks', 2),
                num_freq_masks=sa_cfg.get('num_freq_masks', 2),
                mask_value=sa_cfg.get('mask_value', 0.0),
                p=sa_cfg.get('p', 1.0),
            )

        # --- PatchRandomMasking ---
        pm_cfg = config.get('patch_mask', {})
        if pm_cfg.get('enabled', False):
            patch_size = tuple(pm_cfg.get('patch_size', [16, 16]))
            self.transforms['patch_mask'] = PatchRandomMasking(
                patch_size=patch_size,
                img_size=224,
                mask_ratio=pm_cfg.get('mask_ratio', 0.3),
                mask_value=pm_cfg.get('mask_value', 0.0),
                p=pm_cfg.get('p', 1.0),
            )
            self._base_patch_mask_ratio = pm_cfg.get('mask_ratio', 0.3)

        # --- EnergyAwareMasking ---
        em_cfg = config.get('energy_mask', {})
        if em_cfg.get('enabled', False):
            self.transforms['energy_mask'] = EnergyAwareMasking(
                top_percentile=em_cfg.get('top_percentile', 20.0),
                mask_prob=em_cfg.get('mask_prob', 0.8),
                mask_value=em_cfg.get('mask_value', 0.0),
                p=em_cfg.get('p', 1.0),
            )

        # --- Asymmetric ---
        asym_cfg = config.get('asymmetric', {})
        if asym_cfg.get('enabled', False):
            self._asymmetric = AsymmetricAugmentation(
                extra_patch_mask_ratio=asym_cfg.get('extra_patch_mask_ratio', 0.2),
                extra_specaug_ratio=asym_cfg.get('extra_specaug_ratio', 0.05),
            )

    def forward(
        self,
        x: torch.Tensor,
        metadata: Optional[dict] = None,
        energy_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Args:
            x: [C, H, W] post-CLIP-norm STFT tensor
            metadata: 样本 metadata (含 jam_types)
            energy_map: [H, W] pre-CLIP-norm magnitude

        Returns:
            (augmented_x, suppression_weight): 增强后的 tensor 和抑制分支权重
        """
        supp_weight = 1.0

        # 检查是否为纯压制样本
        extra_patch = 0.0
        extra_spec = 0.0
        if self._asymmetric is not None:
            overrides = self._asymmetric(metadata)
            extra_patch = overrides['extra_patch_mask_ratio']
            extra_spec = overrides['extra_specaug_ratio']

        for name, transform in self.transforms.items():
            if name == 'energy_mask':
                x = transform(x, energy_map=energy_map)
            elif name == 'patch_mask' and extra_patch > 0:
                # 临时增加 mask_ratio
                orig_ratio = transform.mask_ratio
                transform.mask_ratio = min(orig_ratio + extra_patch, 0.9)
                x = transform(x)
                transform.mask_ratio = orig_ratio
            elif name == 'specaugment' and extra_spec > 0:
                # 临时增加 specaug 参数
                orig_t = transform.time_mask_param
                orig_f = transform.freq_mask_param
                transform.time_mask_param = min(orig_t + extra_spec, 0.5)
                transform.freq_mask_param = min(orig_f + extra_spec, 0.5)
                x = transform(x)
                transform.time_mask_param = orig_t
                transform.freq_mask_param = orig_f
            else:
                x = transform(x)

        return x, supp_weight


# ============================================================================
# Unit Test / Visual Verification
# ============================================================================

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    print("=" * 60)
    print("STFT Augmentation — Unit Verification")
    print("=" * 60)

    # 创建模拟 STFT 图像: 3x224x224
    # 亮点 = 欺骗干扰前景 (高能量)
    # 暗背景 = 压制干扰底噪 (低能量)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    dummy = torch.zeros(3, 224, 224)
    # 添加底噪背景
    dummy += torch.randn(3, 224, 224) * 0.05
    # 添加亮点模拟欺骗干扰
    dummy[:, 60:80, 50:170] = 1.5   # 横向亮条
    dummy[:, 140:160, 30:190] = 1.2  # 另一片高能量
    # 添加一些弱的连续纹理模拟压制干扰
    for i in range(224):
        dummy[:, :, i] += 0.15 * torch.sin(torch.linspace(0, 4 * np.pi, 224))

    # Pre-CLIP-norm magnitude (simulated, range ~[0, 1.5])
    energy_map = dummy[0].clone()

    # CLIP 标准化后 (simulated)
    x = torch.stack([
        (dummy[0] - 0.48145466) / 0.26862954,
        (dummy[1] - 0.4578275) / 0.26130258,
        (dummy[2] - 0.40821073) / 0.27577711,
    ], dim=0)

    def show_and_save(tensor, title, filename):
        """可视化 [C, H, W] tensor (取第一通道)"""
        img = tensor[0].detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(img, cmap='viridis', aspect='auto', origin='upper')
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        plt.savefig(f"multi/aug_debug_{filename}.png", dpi=100)
        plt.close()
        print(f"  Saved: multi/aug_debug_{filename}.png")

    # Test 1: SpecAugment
    print("\n[1/4] Testing SpecAugment...")
    sa = SpecAugment(time_mask_param=0.1, freq_mask_param=0.1,
                     num_time_masks=3, num_freq_masks=3, p=1.0)
    show_and_save(x, "Original (post CLIP-norm)", "00_original")
    show_and_save(sa(x.clone()), "SpecAugment (time + freq bands)", "01_specaugment")

    # Test 2: PatchRandomMasking
    print("[2/4] Testing PatchRandomMasking...")
    pm = PatchRandomMasking(patch_size=(16, 16), mask_ratio=0.3, p=1.0)
    show_and_save(pm(x.clone()), "PatchRandomMasking (16x16, ratio=0.3)", "02_patchmask")

    # Test 3: EnergyAwareMasking
    print("[3/4] Testing EnergyAwareMasking...")
    em = EnergyAwareMasking(top_percentile=20.0, mask_prob=0.8, p=1.0)
    show_and_save(em(x.clone(), energy_map=energy_map),
                  "EnergyAwareMasking (top 20%, prob=0.8)", "03_energymask")

    # Test 4: Asymmetric — check logic
    print("[4/4] Testing AsymmetricAugmentation logic...")
    asym = AsymmetricAugmentation(extra_patch_mask_ratio=0.2, extra_specaug_ratio=0.05)
    assert asym.is_suppression_only({'jam_types': ['AJ']}) is True
    assert asym.is_suppression_only({'jam_types': ['BJ', 'SJ']}) is True
    assert asym.is_suppression_only({'jam_types': ['DFTJ']}) is False
    assert asym.is_suppression_only({'jam_types': ['DFTJ', 'AJ']}) is False  # composite
    assert asym.is_suppression_only({'jam_types': []}) is False
    assert asym.is_suppression_only(None) is False
    print("  All assertions passed!")
    print("    - ['AJ']          -> suppression_only = True")
    print("    - ['BJ', 'SJ']    -> suppression_only = True")
    print("    - ['DFTJ']        -> suppression_only = False")
    print("    - ['DFTJ', 'AJ']  -> suppression_only = False (composite)")
    print("    - []              -> suppression_only = False")

    # Test 5: Full pipeline with asymmetric
    print("\n[Bonus] Testing full STFTAugmentation pipeline...")
    config = {
        'specaugment': {'enabled': True, 'time_mask_param': 0.1, 'freq_mask_param': 0.1,
                        'num_time_masks': 2, 'num_freq_masks': 2, 'mask_value': 0.0, 'p': 1.0},
        'patch_mask': {'enabled': True, 'patch_size': [16, 16], 'mask_ratio': 0.2,
                       'mask_value': 0.0, 'p': 1.0},
        'energy_mask': {'enabled': True, 'top_percentile': 20.0, 'mask_prob': 0.7, 'p': 1.0},
        'asymmetric': {'enabled': True, 'extra_patch_mask_ratio': 0.2, 'extra_specaug_ratio': 0.05},
    }
    pipeline = STFTAugmentation(config)

    # Suppression-only sample → should get extra masking
    aug_supp, w_supp = pipeline(x.clone(), metadata={'jam_types': ['AJ']}, energy_map=energy_map)
    show_and_save(aug_supp, "Full pipeline (suppression-only, extra mask)", "04_full_suppression")

    # Deception-only sample → standard masking
    aug_decep, w_decep = pipeline(x.clone(), metadata={'jam_types': ['DFTJ']}, energy_map=energy_map)
    show_and_save(aug_decep, "Full pipeline (deception-only, standard mask)", "05_full_deception")

    print("\n" + "=" * 60)
    print("All tests passed! Check multi/aug_debug_*.png for visual results.")
    print("=" * 60)
