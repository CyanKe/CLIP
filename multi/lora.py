"""
LoRA (Low-Rank Adaptation) 实现
用于高效微调 CLIP 模型

参考: "LoRA: Low-Rank Adaptation of Large Language Models"
"""
import torch
import torch.nn as nn
from typing import List
import math


class LoRALayer(nn.Module):
    """
    单个 LoRA 层

    LoRA 通过低秩分解来近似权重更新：
    W' = W + BA
    其中 B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d, k)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        """
        初始化 LoRA 层

        Args:
            in_features: 输入维度
            out_features: 输出维度
            rank: LoRA 秩
            alpha: LoRA 缩放因子
            dropout: Dropout 比率
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA 低秩矩阵
        # A: [in_features, rank], B: [rank, out_features]
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # 初始化：A 使用 Kaiming，B 初始化为 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        LoRA 增量输出

        Args:
            x: 输入张量 [batch_size, in_features]

        Returns:
            LoRA 增量 [batch_size, out_features]
        """
        return self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling


class LoRALinear(nn.Module):
    """
    带 LoRA 的 Linear 层

    输出 = 原始输出 + LoRA 增量
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        """
        初始化

        Args:
            original_linear: 原始 Linear 层
            rank: LoRA 秩
            alpha: LoRA 缩放因子
            dropout: Dropout 比率
        """
        super().__init__()
        self.original = original_linear

        # 冻结原始权重
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        # LoRA 层
        self.lora = LoRALayer(
            original_linear.in_features,
            original_linear.out_features,
            rank, alpha, dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量

        Returns:
            输出张量
        """
        return self.original(x) + self.lora(x)


def apply_lora_to_model(
    model: nn.Module,
    target_modules: List[str] = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0
) -> int:
    """
    对模型应用 LoRA

    Args:
        model: 要应用 LoRA 的模型
        target_modules: 目标模块名称关键词列表，默认为 Attention 层
        rank: LoRA 秩
        alpha: LoRA 缩放因子
        dropout: Dropout 比率

    Returns:
        替换的层数
    """
    if target_modules is None:
        target_modules = ["attn", "attention", "q_proj", "k_proj", "v_proj", "out_proj"]

    replaced_count = 0
    replacements = []  # 收集要替换的层

    # 遍历模型找到目标模块
    for name, module in model.named_modules():
        # 检查是否为目标模块
        is_target = any(t in name.lower() for t in target_modules)

        if is_target and isinstance(module, nn.Linear):
            # 获取父模块和属性名
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name

            replacements.append((parent, attr_name, name, module))

    # 执行替换
    for parent, attr_name, name, module in replacements:
        lora_linear = LoRALinear(module, rank, alpha, dropout)
        setattr(parent, attr_name, lora_linear)
        print(f"  Applied LoRA to: {name}")
        replaced_count += 1

    return replaced_count


def freeze_non_lora_params(model: nn.Module, verbose: bool = True):
    """
    冻结所有非 LoRA 参数

    Args:
        model: 模型
        verbose: 是否打印可训练参数
    """
    trainable_count = 0
    frozen_count = 0

    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            trainable_count += 1
            if verbose:
                print(f"  Trainable: {name} ({param.numel()} params)")
        else:
            param.requires_grad = False
            frozen_count += 1

    print(f"\n  Frozen: {frozen_count} parameters")
    print(f"  Trainable: {trainable_count} parameters (LoRA only)")


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """
    获取所有 LoRA 参数

    Args:
        model: 模型

    Returns:
        LoRA 参数列表
    """
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            lora_params.append(param)
    return lora_params


def count_parameters(model: nn.Module) -> dict:
    """
    统计模型参数数量

    Args:
        model: 模型

    Returns:
        参数统计字典
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_ratio": trainable / total if total > 0 else 0
    }


if __name__ == "__main__":
    # 测试 LoRA 实现
    print("=" * 60)
    print("Testing LoRA Implementation")
    print("=" * 60)

    # 创建测试模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(512, 256)
            self.linear2 = nn.Linear(256, 128)

        def forward(self, x):
            x = self.linear1(x)
            x = torch.relu(x)
            x = self.linear2(x)
            return x

    model = SimpleModel()
    print("\nOriginal model parameters:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")

    # 应用 LoRA
    print("\nApplying LoRA...")
    replaced = apply_lora_to_model(model, target_modules=["linear"], rank=4, alpha=8.0)
    print(f"Replaced {replaced} layers")

    print("\nAfter LoRA:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")

    # 冻结非 LoRA 参数
    print("\nFreezing non-LoRA parameters...")
    freeze_non_lora_params(model)

    # 统计参数
    stats = count_parameters(model)
    print(f"\nParameter statistics:")
    print(f"  Total: {stats['total']:,}")
    print(f"  Trainable: {stats['trainable']:,}")
    print(f"  Frozen: {stats['frozen']:,}")
    print(f"  Trainable ratio: {stats['trainable_ratio']:.2%}")

    # 测试前向传播
    print("\nTesting forward pass...")
    x = torch.randn(2, 512)
    y = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")

    print("\n" + "=" * 60)
    print("LoRA test completed!")
    print("=" * 60)
