"""
CLIP模型适配器 - 用于雷达干扰信号的组合零样本学习
支持多标签分类和零样本推理
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import sys
import os

# 添加CLIP路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clip


class CLIPAdapter(nn.Module):
    """
    CLIP模型适配器
    将CLIP适配到雷达干扰信号的多标签分类任务
    """

    def __init__(
        self,
        clip_model: str = "ViT-B/32",
        num_classes: int = 16,
        freeze_vision: bool = False,
        freeze_text: bool = True,
        vision_layers_unfreeze: int = 2,
        device: str = "cuda"
    ):
        """
        初始化CLIP适配器

        Args:
            clip_model: CLIP模型名称
            num_classes: 类别数
            freeze_vision: 是否冻结视觉编码器
            freeze_text: 是否冻结文本编码器
            vision_layers_unfreeze: 解冻视觉编码器最后N层
            device: 计算设备
        """
        super().__init__()
        self.device = device
        self.num_classes = num_classes

        # 加载预训练CLIP模型
        self.model, self.preprocess = clip.load(clip_model, device=device)

        # 将CLIP模型转换为FP32以支持微调（避免FP16梯度问题）
        self.model = self.model.float()

        # 获取特征维度
        self.embed_dim = self.model.text_projection.shape[1]

        # 冻结策略
        self._apply_freeze_strategy(
            freeze_vision=freeze_vision,
            freeze_text=freeze_text,
            vision_layers_unfreeze=vision_layers_unfreeze
        )

        # 可学习的分类头（用于有监督微调）
        # 注意：分类器保持FP32以避免梯度溢出，输入时会将FP16特征转为FP32
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim // 2, num_classes)
        ).to(device)

        # 文本特征缓存
        self.text_features_cache = None
        self.cached_class_names = None

    def _apply_freeze_strategy(
        self,
        freeze_vision: bool,
        freeze_text: bool,
        vision_layers_unfreeze: int
    ):
        """应用冻结策略"""
        # 冻结文本编码器
        if freeze_text:
            for param in self.model.transformer.parameters():
                param.requires_grad = False
            self.model.token_embedding.weight.requires_grad = False
            self.model.positional_embedding.requires_grad = False
            self.model.text_projection.requires_grad = False

        # 冻结视觉编码器
        if freeze_vision:
            for param in self.model.visual.parameters():
                param.requires_grad = False
        elif vision_layers_unfreeze > 0:
            # 部分解冻：只解冻最后N层
            self._freeze_vision_partially(vision_layers_unfreeze)

    def _freeze_vision_partially(self, layers_unfreeze: int):
        """部分冻结视觉编码器"""
        # 先冻结所有视觉参数
        for param in self.model.visual.parameters():
            param.requires_grad = False

        # 根据模型类型解冻最后N层
        if hasattr(self.model.visual, 'transformer'):
            # ViT类型
            num_layers = len(self.model.visual.transformer.resblocks)
            for i in range(num_layers - layers_unfreeze, num_layers):
                for param in self.model.visual.transformer.resblocks[i].parameters():
                    param.requires_grad = True
        elif hasattr(self.model.visual, 'layer4'):
            # ResNet类型
            for param in self.model.visual.layer4.parameters():
                param.requires_grad = True
            if layers_unfreeze > 1:
                for param in self.model.visual.layer3.parameters():
                    param.requires_grad = True

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        编码文本

        Args:
            text_tokens: 文本token张量 [batch_size, seq_len]

        Returns:
            文本特征 [batch_size, embed_dim]
        """
        return self.model.encode_text(text_tokens)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        编码图像（STFT时频图）

        Args:
            image: 图像张量 [batch_size, 3, H, W]

        Returns:
            图像特征 [batch_size, embed_dim]
        """
        return self.model.encode_image(image)

    def cache_text_features(self, class_descriptions: Dict[str, List[str]]):
        """
        预计算并缓存文本特征

        Args:
            class_descriptions: 类别描述字典 {class_name: [description1, ...]}
        """
        self.model.eval()
        all_features = []
        class_names = []

        with torch.no_grad():
            for class_name, descriptions in class_descriptions.items():
                # 使用第一个描述作为默认
                text = clip.tokenize(descriptions[0], truncate=True).to(self.device)
                text_features = self.encode_text(text)
                text_features = F.normalize(text_features, dim=-1)
                all_features.append(text_features)
                class_names.append(class_name)

        self.text_features_cache = torch.cat(all_features, dim=0)  # [num_classes, embed_dim]
        self.cached_class_names = class_names

        print(f"Cached text features for {len(class_names)} classes")

    def forward_zero_shot(
        self,
        image: torch.Tensor,
        use_cached: bool = True
    ) -> torch.Tensor:
        """
        零样本推理

        Args:
            image: 图像张量 [batch_size, 3, H, W]
            use_cached: 是否使用缓存的文本特征

        Returns:
            相似度分数 [batch_size, num_classes]
        """
        # 编码图像
        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)

        # 计算相似度
        if use_cached and self.text_features_cache is not None:
            text_features = self.text_features_cache
        else:
            raise ValueError("No cached text features. Call cache_text_features() first.")

        # 计算余弦相似度
        logits = (image_features @ text_features.T) * self.model.logit_scale.exp()

        return logits

    def forward_supervised(self, image: torch.Tensor) -> torch.Tensor:
        """
        有监督分类

        Args:
            image: 图像张量 [batch_size, 3, H, W]

        Returns:
            分类logits [batch_size, num_classes]
        """
        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)
        logits = self.classifier(image_features)
        return logits

    def forward(
        self,
        image: torch.Tensor,
        mode: str = "supervised"
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            image: 图像张量 [batch_size, 3, H, W]
            mode: 模式 "supervised" 或 "zero_shot"

        Returns:
            输出logits
        """
        if mode == "zero_shot":
            return self.forward_zero_shot(image)
        else:
            return self.forward_supervised(image)


class CLIPForMultiLabel(CLIPAdapter):
    """
    专门用于多标签分类的CLIP适配器
    支持组合标签的零样本推理
    """

    def __init__(
        self,
        clip_model: str = "ViT-B/32",
        num_classes: int = 16,
        class_names: List[str] = None,
        class_descriptions: Dict[str, List[str]] = None,
        **kwargs
    ):
        """
        初始化多标签CLIP

        Args:
            clip_model: CLIP模型名称
            num_classes: 类别数
            class_names: 类别名称列表
            class_descriptions: 类别描述字典
        """
        super().__init__(clip_model, num_classes, **kwargs)

        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.class_descriptions = class_descriptions or {}

        # 如果提供了类别描述，缓存文本特征
        if class_descriptions:
            self.cache_text_features(class_descriptions)

    def build_combination_descriptions(
        self,
        class_indices: List[List[int]],
        template: str = "a radar signal with {} and {}"
    ) -> List[str]:
        """
        构建组合类别的文本描述

        Args:
            class_indices: 组合索引列表 [[0,1], [2,3], ...]
            template: 描述模板

        Returns:
            组合描述列表
        """
        descriptions = []
        for indices in class_indices:
            if len(indices) == 2:
                name1 = self.class_names[indices[0]]
                name2 = self.class_names[indices[1]]
                desc = template.format(
                    self.class_descriptions.get(name1, [name1])[0].replace("a radar signal with ", ""),
                    self.class_descriptions.get(name2, [name2])[0].replace("a radar signal with ", "")
                )
                descriptions.append(desc)
        return descriptions

    def predict_multilabel(
        self,
        image: torch.Tensor,
        threshold: float = 0.5,
        mode: str = "supervised"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        多标签预测

        Args:
            image: 图像张量 [batch_size, 3, H, W]
            threshold: 预测阈值
            mode: 预测模式

        Returns:
            概率和预测标签
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(image, mode=mode)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

        return probs, preds


def create_clip_model(config: dict, device: str = "cuda") -> CLIPForMultiLabel:
    """
    根据配置创建CLIP模型

    Args:
        config: 配置字典
        device: 计算设备

    Returns:
        CLIP模型实例
    """
    model_config = config.get("model", {})

    # 构建类别描述
    class_descriptions = {}
    class_names = []
    for cls_info in config.get("jamming_classes", []):
        name = cls_info["name"]
        class_names.append(name)
        class_descriptions[name] = cls_info["descriptions"]

    model = CLIPForMultiLabel(
        clip_model=model_config.get("clip_model", "ViT-B/32"),
        num_classes=len(class_names),
        class_names=class_names,
        class_descriptions=class_descriptions,
        freeze_vision=model_config.get("freeze_vision", False),
        freeze_text=model_config.get("freeze_text", True),
        vision_layers_unfreeze=model_config.get("vision_layers_unfreeze", 2),
        device=device
    )

    return model


if __name__ == "__main__":
    # 测试代码
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 创建模型
    model = CLIPAdapter(clip_model="ViT-B/32", num_classes=16, device=device)
    print(f"Model created with embed_dim={model.embed_dim}")

    # 测试前向传播
    dummy_image = torch.randn(2, 3, 224, 224).to(device)
    output = model.forward_supervised(dummy_image)
    print(f"Output shape: {output.shape}")