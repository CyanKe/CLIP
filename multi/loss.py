"""
损失函数模块 - 支持多标签分类和对比学习
包含CLIP风格的InfoNCE对比损失用于CZSL
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class InfoNCELoss(nn.Module):
    """
    InfoNCE对比损失 - CLIP风格
    用于训练图像-文本对齐

    参考: "Learning Transferable Visual Models From Natural Language Supervision"
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temperature: bool = True
    ):
        """
        初始化

        Args:
            temperature: 温度参数
            learnable_temperature: 是否学习温度参数
        """
        super().__init__()
        self.temperature = temperature

        if learnable_temperature:
            # 可学习的温度参数（类似于CLIP）
            self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature)))
        else:
            self.register_buffer('logit_scale', torch.ones([]) * torch.log(torch.tensor(1 / temperature)))

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        labels: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算InfoNCE损失

        Args:
            image_features: 图像特征 [batch_size, embed_dim]
            text_features: 文本特征 [batch_size, embed_dim]
            labels: 可选的标签（用于多标签扩展）

        Returns:
            loss: 对比损失
            logits: 相似度矩阵
        """
        batch_size = image_features.shape[0]

        # 归一化特征
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # 获取温度缩放因子
        logit_scale = self.logit_scale.exp()

        # 计算相似度矩阵
        # [batch_size, batch_size]
        logits_per_image = logit_scale * (image_features @ text_features.t())
        logits_per_text = logits_per_image.t()

        # 创建标签（对角线为正样本）
        # 在标准CLIP中，每个图像与对应的文本是正样本
        targets = torch.arange(batch_size, device=image_features.device)

        # 计算交叉熵损失
        # 图像到文本的损失
        loss_i2t = F.cross_entropy(logits_per_image, targets)
        # 文本到图像的损失
        loss_t2i = F.cross_entropy(logits_per_text, targets)

        # 总损失
        loss = (loss_i2t + loss_t2i) / 2

        return loss, logits_per_image


class MultiLabelInfoNCELoss(nn.Module):
    """
    多标签InfoNCE损失
    支持一个图像对应多个文本描述的场景
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
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算多标签InfoNCE损失

        Args:
            image_features: 图像特征 [batch_size, embed_dim]
            text_features: 文本特征 [num_texts, embed_dim]
            labels: 多标签指示 [batch_size, num_texts]
                   labels[i, j] = 1 表示图像i与文本j匹配

        Returns:
            损失值
        """
        batch_size = image_features.shape[0]
        num_texts = text_features.shape[0]

        # 归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # 计算相似度 [batch_size, num_texts]
        logits = (image_features @ text_features.t()) / self.temperature

        # 标签平滑
        if self.label_smoothing > 0:
            labels = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # 使用二元交叉熵损失
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        return loss


class CZSLContrastiveLoss(nn.Module):
    """
    CZSL对比损失
    结合图像-文本对齐和多标签分类
    """

    def __init__(
        self,
        temperature: float = 0.07,
        contrastive_weight: float = 1.0,
        classification_weight: float = 0.0,
        learnable_temperature: bool = True
    ):
        """
        初始化

        Args:
            temperature: 温度参数
            contrastive_weight: 对比损失权重
            classification_weight: 分类损失权重（可选）
            learnable_temperature: 是否学习温度参数
        """
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.classification_weight = classification_weight

        # 对比损失
        self.contrastive_loss = InfoNCELoss(
            temperature=temperature,
            learnable_temperature=learnable_temperature
        )

        # 分类损失（可选）
        if classification_weight > 0:
            self.classification_loss = AsymmetricLoss()
        else:
            self.classification_loss = None

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        labels: torch.Tensor = None,
        classifier_logits: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算CZSL损失

        Args:
            image_features: 图像特征 [batch_size, embed_dim]
            text_features: 文本特征 [batch_size, embed_dim]
            labels: 标签 [batch_size, num_classes] (可选)
            classifier_logits: 分类器输出 [batch_size, num_classes] (可选)

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失字典
        """
        loss_dict = {}

        # 对比损失
        contrastive_loss, logits = self.contrastive_loss(image_features, text_features)
        loss_dict['contrastive_loss'] = contrastive_loss.item()

        total_loss = self.contrastive_weight * contrastive_loss

        # 分类损失（如果启用）
        if self.classification_loss is not None and classifier_logits is not None and labels is not None:
            classification_loss = self.classification_loss(classifier_logits, labels)
            loss_dict['classification_loss'] = classification_loss.item()
            total_loss = total_loss + self.classification_weight * classification_loss

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

    def get_logit_scale(self) -> torch.Tensor:
        """获取温度缩放因子"""
        return self.contrastive_loss.logit_scale.exp()


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

    if loss_type == "infonce":
        return InfoNCELoss(
            temperature=loss_config.get("temperature", 0.07),
            learnable_temperature=loss_config.get("learnable_temperature", True)
        )
    elif loss_type == "multilabel_infonce":
        return MultiLabelInfoNCELoss(
            temperature=loss_config.get("temperature", 0.07),
            label_smoothing=loss_config.get("label_smoothing", 0.0)
        )
    elif loss_type == "czsl_contrastive":
        return CZSLContrastiveLoss(
            temperature=loss_config.get("temperature", 0.07),
            contrastive_weight=loss_config.get("contrastive_weight", 1.0),
            classification_weight=loss_config.get("classification_weight", 0.0),
            learnable_temperature=loss_config.get("learnable_temperature", True)
        )
    elif loss_type == "multilabel_contrastive":
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
    text_features = torch.randn(batch_size, embed_dim)  # 注意：batch_size个文本
    all_text_features = torch.randn(num_classes, embed_dim)  # 所有类别的文本特征
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, 2, (batch_size, num_classes)).float()
    logit_scale = torch.tensor(2.6595)  # CLIP默认值

    # 测试各种损失
    print("=" * 60)
    print("Testing CZSL Loss Functions")
    print("=" * 60)

    # 1. InfoNCE对比损失
    print("\n1. InfoNCE Loss (CLIP-style):")
    infonce_loss = InfoNCELoss(temperature=0.07)
    loss1, sim_matrix = infonce_loss(image_features, text_features)
    print(f"   Loss: {loss1.item():.4f}")
    print(f"   Similarity matrix shape: {sim_matrix.shape}")
    print(f"   Logit scale: {infonce_loss.logit_scale.exp().item():.4f}")

    # 2. 多标签InfoNCE损失
    print("\n2. Multi-Label InfoNCE Loss:")
    ml_infonce = MultiLabelInfoNCELoss(temperature=0.07)
    # 创建匹配标签（模拟）
    match_labels = torch.zeros(batch_size, num_classes)
    for i in range(batch_size):
        match_labels[i, i % num_classes] = 1
    loss2 = ml_infonce(image_features, all_text_features, match_labels)
    print(f"   Loss: {loss2.item():.4f}")

    # 3. CZSL对比损失
    print("\n3. CZSL Contrastive Loss:")
    czsl_loss = CZSLContrastiveLoss(
        temperature=0.07,
        contrastive_weight=1.0,
        classification_weight=0.5
    )
    loss3, loss_dict = czsl_loss(image_features, text_features, labels, logits)
    print(f"   Total Loss: {loss3.item():.4f}")
    print(f"   Loss dict: {loss_dict}")

    # 4. 多标签对比损失
    print("\n4. Multi-Label Contrastive Loss:")
    mlc_loss = MultiLabelContrastiveLoss()
    loss4 = mlc_loss(image_features, all_text_features, labels, logit_scale)
    print(f"   Loss: {loss4.item():.4f}")

    # 5. ASL损失
    print("\n5. Asymmetric Loss:")
    asl_loss = AsymmetricLoss()
    loss5 = asl_loss(logits, labels)
    print(f"   Loss: {loss5.item():.4f}")

    # 6. Focal损失
    print("\n6. Focal Loss:")
    focal_loss = FocalLoss()
    loss6 = focal_loss(logits, labels)
    print(f"   Loss: {loss6.item():.4f}")

    # 7. 组合损失
    print("\n7. Combined Loss:")
    combined_loss = CombinedLoss()
    loss7 = combined_loss(image_features, all_text_features, logits, labels, logit_scale)
    print(f"   Loss: {loss7.item():.4f}")

    print("\n" + "=" * 60)
    print("All loss functions tested successfully!")
    print("=" * 60)