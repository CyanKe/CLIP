# STFT 数据增强框架说明

## 问题背景

在组合干扰（欺骗+压制线性叠加）零样本泛化场景中，欺骗干扰的 STFT 特征能量更高、形态更显著，模型容易产生 **捷径学习（Shortcut Learning）**——看
到高亮前景就判定为欺骗干扰，忽略背景中的压制干扰纹理，导致压制分支在组合测试集上准确率下降。

## 设计思路

双分支架构（欺骗分支 + 压制分支）已经将两类干扰解耦到独立的对比学习任务中，但单独的压制干扰样本画面上缺乏"前景遮挡"这种真实组合场景的挑战。本增强框架通过在训练时随机破坏图像，迫使模型：
- 不能只依赖局部高亮特征
- 必须整合全局纹理做决策
- 在压制分支上学会"透过前景看背景"

## 四种增强策略

| 策略 | 核心机制 | 特点 |
|---|---|---|
| **SpecAugment** | 随机遮住连续的时频带（行=时间，列=频率） | 物理意义明确，最稳定 |
| **PatchRandomMasking** | 以 ViT patch 为最小单位整块随机丢弃 | 破坏几何连续性，强制全局整合 |
| **EnergyAwareMasking** | 识别高能量像素（通常=欺骗前景），概率掩码 | 精准打击 shortcut 源头 |
| **AsymmetricAugmentation** | 检测纯压制样本，对其施加更强增强 | 让压制分支在困难条件下学习 |

## 配置结构

```yaml
augmentation:
  enabled: true          # 总开关，false 则全部绕过

  specaugment:
    enabled: true
    time_mask_param: 0.1   # 单条时间掩码最大宽度（占 H 比例）
    freq_mask_param: 0.1   # 单条频率掩码最大高度（占 W 比例）
    num_time_masks: 2      # 时间方向掩码条数
    num_freq_masks: 2      # 频率方向掩码条数
    mask_value: 0.0        # 填充值（0 = CLIP 均值）
    p: 1.0                 # 对该样本应用增强的概率

  patch_mask:
    enabled: true
    patch_size: [16, 16]   # 掩码块尺寸，独立于 ViT 的 patch_sizes
    mask_ratio: 0.3        # 丢弃 patch 的比例
    mask_value: 0.0
    p: 1.0

  energy_mask:
    enabled: true
    top_percentile: 20.0   # 高能量阈值（前 N%）
    mask_prob: 0.8         # 对候选像素的实际掩码概率（不要设 1.0）
    mask_value: 0.0
    p: 1.0

  asymmetric:
    enabled: true
    extra_patch_mask_ratio: 0.2   # 仅对纯压制样本追加的 mask 比例
    extra_specaug_ratio: 0.05     # 仅对纯压制样本追加的 specaug 参数
```

## 关键设计决策

**增强在 `__getitem__` 中应用，位于 CLIP 标准化之后**
- CLIP 标准化后张量均值 ≈ 0，mask_value=0 等价于"填充均值"即"无信息"
- 不需要修改 collate_fn、trainer、loss 函数

**仅训练集生效**
- `create_dual_branch_dataloaders` 内部根据 `split_name == 'train'` 判断是否传入 augmentation
- 验证集和测试集始终使用原始图像，保证评估不受增强干扰

**energy_map 的获取**
- 在 `__getitem__` 中，CLIP 标准化之前保存 `stft_tensor[0].clone()`
- 此时张量已是 `[3, 224, 224]`（已 resize），取通道 0 即幅度图，值域约 [0, 1]
- 无需额外插值计算，无运行时开销

**PatchRandomMasking 的向量化实现**
- 通过 reshape → permute → torch.where → permute → reshape 实现完全向量化
- 无 Python 循环，兼容 DataLoader 多进程

**mask_prob 不要设 1.0**
- `energy_mask.mask_prob=1.0` 会遮死所有高能量区域，导致欺骗分支无法学习
- 推荐 0.5~0.8，保留"透视"能力

## 推荐实验路径

| 阶段 | SpecAug | PatchMask | EnergyMask | Asymmetric | 观察重点 |
|---|---|---|---|---|---|
| Baseline | - | - | - | - | 确认代码不改动时性能复现 |
| A | ✓ | - | - | - | 压制分支准确率是否开始爬升 |
| B | ✓ | ✓ | - | - | 欺骗分支略有下降可接受，压制继续提升 |
| C | ✓ | ✓ | ✓ | - | 调 mask_prob 0.5~0.8，避免欺骗分支崩掉 |
| D | ✓ | ✓ | ✓ | ✓ | 四项全开，综合最优 |

核心评估指标：组合干扰测试集上**压制分支的 Subset Accuracy / Jaccard**。

## 回滚

```yaml
augmentation:
  enabled: false
```

## 文件清单

| 文件 | 说明 |
|---|---|
| `multi/augmentation.py` | 增强实现（~260 行） |
| `multi/config.yaml` | 配置入口（`augmentation` 段） |
| `multi/data.py` | 增强集成点（`__getitem__` + dataloader 工厂） |
