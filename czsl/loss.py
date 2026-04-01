"""
损失函数模块 - 支持多标签分类和对比学习
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiLabelContrastiveLoss(nn.Module):
    """
    多标签对比损失
    扩展CLIP的InfoNCE损失到多标签场景
    """

    def __init__(
        self,
        temperature: float = 0.07,
        label_smoothing: float = 0.0
    ):
        """
        初始化

        Args:
            temperature: 温度参数
            label_smoothing: 标签平滑系数
        """
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        labels: torch.Tensor,
        logit_scale: torch.Tensor
    ) -> torch.Tensor:
        """
        计算多标签对比损失

        Args:
            image_features: 图像特征 [batch_size, embed_dim]
            text_features: 文本特征 [num_classes, embed_dim]
            labels: 多热标签 [batch_size, num_classes]
            logit_scale: CLIP的logit_scale参数

        Returns:
            损失值
        """
        batch_size = image_features.shape[0]
        num_classes = text_features.shape[0]

        # 归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # 计算相似度矩阵 [batch_size, num_classes]
        logits = logit_scale.exp() * (image_features @ text_features.T)
        logits = logits / self.temperature

        # 标签平滑
        if self.label_smoothing > 0:
            labels = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # 多标签对比损失：每个正样本对都要最大化
        # 使用二元交叉熵损失
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        return loss


class AsymmetricLoss(nn.Module):
    """
    非对称损失 (ASL)
    适用于多标签分类中的类别不平衡问题
    参考: "Asymmetric Loss For Multi-Label Classification"
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8
    ):
        """
        初始化

        Args:
            gamma_neg: 负样本的聚焦参数
            gamma_pos: 正样本的聚焦参数
            clip: 概率裁剪值
            eps: 数值稳定性
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算ASL损失

        Args:
            logits: 模型输出logits [batch_size, num_classes]
            labels: 标签 [batch_size, num_classes]

        Returns:
            损失值
        """
        # 计算概率
        probs = torch.sigmoid(logits)

        # 正样本损失
        pos_loss = labels * torch.log(probs.clamp(min=self.eps))
        pos_loss = pos_loss * (1 - probs).pow(self.gamma_pos)

        # 负样本损失
        probs_neg = probs.clamp(max=1 - self.clip) if self.clip > 0 else probs
        neg_loss = (1 - labels) * torch.log((1 - probs_neg).clamp(min=self.eps))
        neg_loss = neg_loss * probs_neg.pow(self.gamma_neg)

        loss = -pos_loss - neg_loss
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for Multi-label Classification
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean"
    ):
        """
        初始化

        Args:
            gamma: 聚焦参数
            alpha: 正样本权重
            reduction: 归约方式
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算Focal Loss

        Args:
            logits: 模型输出logits
            labels: 标签

        Returns:
            损失值
        """
        probs = torch.sigmoid(logits)

        # 正样本损失
        pos_loss = -self.alpha * (1 - probs).pow(self.gamma) * torch.log(probs + 1e-8) * labels

        # 负样本损失
        neg_loss = -(1 - self.alpha) * probs.pow(self.gamma) * torch.log(1 - probs + 1e-8) * (1 - labels)

        loss = pos_loss + neg_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    """
    组合损失函数
    结合对比损失和分类损失
    """

    def __init__(
        self,
        contrastive_weight: float = 1.0,
        classification_weight: float = 1.0,
        temperature: float = 0.07,
        use_asl: bool = True
    ):
        """
        初始化

        Args:
            contrastive_weight: 对比损失权重
            classification_weight: 分类损失权重
            temperature: 温度参数
            use_asl: 是否使用ASL损失
        """
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.classification_weight = classification_weight

        self.contrastive_loss = MultiLabelContrastiveLoss(temperature=temperature)
        self.classification_loss = AsymmetricLoss() if use_asl else nn.BCEWithLogitsLoss()

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        logit_scale: torch.Tensor
    ) -> torch.Tensor:
        """
        计算组合损失

        Args:
            image_features: 图像特征
            text_features: 文本特征
            logits: 分类logits
            labels: 标签
            logit_scale: CLIP的logit_scale

        Returns:
            总损失
        """
        # 对比损失
        contrastive = self.contrastive_loss(image_features, text_features, labels, logit_scale)

        # 分类损失
        classification = self.classification_loss(logits, labels)

        # 组合
        total_loss = (self.contrastive_weight * contrastive +
                      self.classification_weight * classification)

        return total_loss


def create_loss_function(config: dict) -> nn.Module:
    """
    根据配置创建损失函数

    Args:
        config: 配置字典

    Returns:
        损失函数实例
    """
    loss_config = config.get("loss", {})
    loss_type = loss_config.get("type", "bce")

    if loss_type == "multilabel_contrastive":
        return MultiLabelContrastiveLoss(
            temperature=loss_config.get("temperature", 0.07),
            label_smoothing=loss_config.get("label_smoothing", 0.0)
        )
    elif loss_type == "asymmetric":
        return AsymmetricLoss()
    elif loss_type == "focal":
        return FocalLoss()
    elif loss_type == "combined":
        return CombinedLoss(
            temperature=loss_config.get("temperature", 0.07)
        )
    else:
        # 默认BCE损失
        return nn.BCEWithLogitsLoss()


if __name__ == "__main__":
    # 测试损失函数
    batch_size = 4
    num_classes = 16
    embed_dim = 512

    # 模拟数据
    image_features = torch.randn(batch_size, embed_dim)
    text_features = torch.randn(num_classes, embed_dim)
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, 2, (batch_size, num_classes)).float()
    logit_scale = torch.tensor(2.6595)  # CLIP默认值

    # 测试各种损失
    print("Testing loss functions...")

    # 1. 多标签对比损失
    mlc_loss = MultiLabelContrastiveLoss()
    loss1 = mlc_loss(image_features, text_features, labels, logit_scale)
    print(f"MultiLabelContrastiveLoss: {loss1.item():.4f}")

    # 2. ASL损失
    asl_loss = AsymmetricLoss()
    loss2 = asl_loss(logits, labels)
    print(f"AsymmetricLoss: {loss2.item():.4f}")

    # 3. Focal损失
    focal_loss = FocalLoss()
    loss3 = focal_loss(logits, labels)
    print(f"FocalLoss: {loss3.item():.4f}")

    # 4. 组合损失
    combined_loss = CombinedLoss()
    loss4 = combined_loss(image_features, text_features, logits, labels, logit_scale)
    print(f"CombinedLoss: {loss4.item():.4f}")

    print("\nAll loss functions tested successfully!")