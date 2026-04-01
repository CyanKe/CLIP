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


class CLIPForCZSL(nn.Module):
    """
    CLIP模型用于组合零样本学习 (CZSL)

    支持两种训练模式:
    1. 对比学习模式：训练图像-文本对齐
    2. 零样本推理模式：使用文本描述识别未见过的组合
    """

    def __init__(
        self,
        clip_model: str = "ViT-B/32",
        num_classes: int = 16,
        class_names: List[str] = None,
        class_descriptions: Dict[str, List[str]] = None,
        freeze_vision: bool = False,
        freeze_text: bool = True,
        vision_layers_unfreeze: int = 2,
        device: str = "cuda"
    ):
        """
        初始化CZSL-CLIP模型

        Args:
            clip_model: CLIP模型名称
            num_classes: 类别数
            class_names: 类别名称列表
            class_descriptions: 类别描述字典
            freeze_vision: 是否冻结视觉编码器
            freeze_text: 是否冻结文本编码器
            vision_layers_unfreeze: 解冻视觉编码器最后N层
            device: 计算设备
        """
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.class_descriptions = class_descriptions or {}

        # 加载预训练CLIP模型
        self.model, self.preprocess = clip.load(clip_model, device=device)
        self.model = self.model.float()  # 转换为FP32

        # 获取特征维度
        self.embed_dim = self.model.text_projection.shape[1]

        # 应用冻结策略
        self._apply_freeze_strategy(
            freeze_vision=freeze_vision,
            freeze_text=freeze_text,
            vision_layers_unfreeze=vision_layers_unfreeze
        )

        # 可选的分类头（用于混合训练）
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim // 2, num_classes)
        ).to(device)

        # 文本特征缓存（用于零样本推理）
        self._text_features_cache = None
        self._combination_features_cache = None
        self._combination_names = None

    def _apply_freeze_strategy(self, freeze_vision, freeze_text, vision_layers_unfreeze):
        """应用冻结策略"""
        if freeze_text:
            for param in self.model.transformer.parameters():
                param.requires_grad = False
            self.model.token_embedding.weight.requires_grad = False
            self.model.positional_embedding.requires_grad = False
            self.model.text_projection.requires_grad = False

        if freeze_vision:
            for param in self.model.visual.parameters():
                param.requires_grad = False
        elif vision_layers_unfreeze > 0:
            self._freeze_vision_partially(vision_layers_unfreeze)

    def _freeze_vision_partially(self, layers_unfreeze: int):
        """部分冻结视觉编码器"""
        for param in self.model.visual.parameters():
            param.requires_grad = False

        if hasattr(self.model.visual, 'transformer'):
            num_layers = len(self.model.visual.transformer.resblocks)
            for i in range(num_layers - layers_unfreeze, num_layers):
                for param in self.model.visual.transformer.resblocks[i].parameters():
                    param.requires_grad = True
        elif hasattr(self.model.visual, 'layer4'):
            for param in self.model.visual.layer4.parameters():
                param.requires_grad = True
            if layers_unfreeze > 1:
                for param in self.model.visual.layer3.parameters():
                    param.requires_grad = True

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """编码图像"""
        return self.model.encode_image(image)

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """编码文本"""
        return self.model.encode_text(text_tokens)

    def forward_contrastive(
        self,
        image: torch.Tensor,
        text_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对比学习前向传播

        Args:
            image: 图像张量 [batch_size, 3, H, W]
            text_tokens: 文本tokens [batch_size, seq_len]

        Returns:
            image_features: 图像特征 [batch_size, embed_dim]
            text_features: 文本特征 [batch_size, embed_dim]
        """
        image_features = self.encode_image(image)
        text_features = self.encode_text(text_tokens)

        # 归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        return image_features, text_features

    def forward_classifier(self, image: torch.Tensor) -> torch.Tensor:
        """
        分类器前向传播

        Args:
            image: 图像张量 [batch_size, 3, H, W]

        Returns:
            logits: 分类logits [batch_size, num_classes]
        """
        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)
        logits = self.classifier(image_features)
        return logits

    def forward(
        self,
        image: torch.Tensor,
        text_tokens: torch.Tensor = None,
        mode: str = "contrastive"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            image: 图像张量
            text_tokens: 文本tokens (对比学习模式需要)
            mode: "contrastive" 或 "classifier"

        Returns:
            根据模式返回不同的结果
        """
        if mode == "contrastive" and text_tokens is not None:
            return self.forward_contrastive(image, text_tokens)
        else:
            return self.forward_classifier(image)

    @torch.no_grad()
    def cache_text_features(
        self,
        max_combination_size: int = 2,
        include_single: bool = True
    ):
        """
        缓存所有类别和组合的文本特征

        Args:
            max_combination_size: 最大组合大小
            include_single: 是否包含单干扰
        """
        self.eval()
        from itertools import combinations

        all_features = []
        all_names = []

        # 单干扰特征
        if include_single:
            for cls_name in self.class_names:
                desc = self.class_descriptions.get(cls_name, [cls_name])[0]
                tokens = clip.tokenize(desc, truncate=True).to(self.device)
                features = self.encode_text(tokens)
                features = F.normalize(features, dim=-1)
                all_features.append(features)
                all_names.append(cls_name)

        # 组合特征
        if max_combination_size >= 2:
            for i, j in combinations(range(len(self.class_names)), 2):
                cls1, cls2 = self.class_names[i], self.class_names[j]
                desc1 = self.class_descriptions.get(cls1, [cls1])[0]
                desc2 = self.class_descriptions.get(cls2, [cls2])[0]

                # 提取关键部分
                if desc1.startswith("a radar signal with "):
                    desc1 = desc1[len("a radar signal with "):]
                if desc2.startswith("a radar signal with "):
                    desc2 = desc2[len("a radar signal with "):]

                combined_desc = f"a radar signal with {desc1} and {desc2}"
                tokens = clip.tokenize(combined_desc, truncate=True).to(self.device)
                features = self.encode_text(tokens)
                features = F.normalize(features, dim=-1)
                all_features.append(features)
                all_names.append(f"{cls1}+{cls2}")

        self._combination_features_cache = torch.cat(all_features, dim=0)
        self._combination_names = all_names
        self._text_features_cache = self._combination_features_cache[:len(self.class_names)]

        print(f"Cached {len(all_names)} text features:")
        print(f"  - Single classes: {len(self.class_names)}")
        print(f"  - Combinations: {len(all_names) - len(self.class_names)}")

    def get_cached_text_features(self) -> torch.Tensor:
        """获取缓存的文本特征"""
        if self._text_features_cache is None:
            self.cache_text_features()
        return self._text_features_cache

    def get_cached_combination_features(self) -> torch.Tensor:
        """获取缓存的组合特征"""
        if self._combination_features_cache is None:
            self.cache_text_features()
        return self._combination_features_cache

    @torch.no_grad()
    def zero_shot_predict(
        self,
        image: torch.Tensor,
        use_combinations: bool = True,
        top_k: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        零样本预测

        Args:
            image: 图像张量 [batch_size, 3, H, W]
            use_combinations: 是否使用组合特征
            top_k: 返回top-k个预测

        Returns:
            similarities: 相似度分数 [batch_size, num_candidates]
            indices: 预测索引 [batch_size, top_k]
            names: 预测名称列表
        """
        self.eval()

        # 编码图像
        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)

        # 选择文本特征
        if use_combinations:
            text_features = self.get_cached_combination_features()
            names = self._combination_names
        else:
            text_features = self.get_cached_text_features()
            names = self.class_names

        # 计算相似度
        logit_scale = self.model.logit_scale.exp()
        similarities = logit_scale * (image_features @ text_features.T)

        # 获取top-k预测
        top_k = min(top_k, text_features.shape[0])
        values, indices = torch.topk(similarities, k=top_k, dim=-1)

        # 获取预测名称
        pred_names = [[names[idx.item()] for idx in batch_indices] for batch_indices in indices]

        return similarities, indices, pred_names

    @torch.no_grad()
    def zero_shot_multilabel_predict(
        self,
        image: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        零样本多标签预测
        使用单类别特征进行多标签预测

        Args:
            image: 图像张量 [batch_size, 3, H, W]
            threshold: 预测阈值

        Returns:
            probs: 概率 [batch_size, num_classes]
            preds: 预测标签 [batch_size, num_classes]
        """
        self.eval()

        # 编码图像
        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)

        # 使用单类别特征
        text_features = self.get_cached_text_features()

        # 计算相似度
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * (image_features @ text_features.T)

        # 转换为概率
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        return probs, preds


def create_clip_model(config: dict, device: str = "cuda", model_type: str = "multilabel") -> nn.Module:
    """
    根据配置创建CLIP模型

    Args:
        config: 配置字典
        device: 计算设备
        model_type: 模型类型 "multilabel" 或 "czsl"

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

    if model_type == "czsl":
        # 创建CZSL模型
        model = CLIPForCZSL(
            clip_model=model_config.get("clip_model", "ViT-B/32"),
            num_classes=len(class_names),
            class_names=class_names,
            class_descriptions=class_descriptions,
            freeze_vision=model_config.get("freeze_vision", False),
            freeze_text=model_config.get("freeze_text", True),
            vision_layers_unfreeze=model_config.get("vision_layers_unfreeze", 2),
            device=device
        )
    else:
        # 创建多标签分类模型
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


def create_czsl_model(config: dict, device: str = "cuda") -> CLIPForCZSL:
    """
    创建CZSL模型

    Args:
        config: 配置字典
        device: 计算设备

    Returns:
        CLIPForCZSL模型实例
    """
    return create_clip_model(config, device, model_type="czsl")


if __name__ == "__main__":
    import yaml

    # 测试代码
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("=" * 60)
    print("Testing CLIP Models for CZSL")
    print("=" * 60)

    # 创建测试配置
    test_config = {
        "model": {
            "clip_model": "ViT-B/32",
            "freeze_vision": False,
            "freeze_text": True,
            "vision_layers_unfreeze": 2
        },
        "jamming_classes": [
            {"name": "DFTJ", "descriptions": ["a radar signal with dense false target jamming"]},
            {"name": "ISRJ", "descriptions": ["a radar signal with interrupted sampling repeater jamming"]},
            {"name": "VDJ", "descriptions": ["a radar signal with velocity deception jamming"]},
            {"name": "DDJ", "descriptions": ["a radar signal with distance and velocity joint deception jamming"]},
        ]
    }

    # 测试CZSL模型
    print("\n1. Testing CLIPForCZSL:")
    czsl_model = create_czsl_model(test_config, device)
    print(f"   Embed dim: {czsl_model.embed_dim}")
    print(f"   Num classes: {czsl_model.num_classes}")
    print(f"   Class names: {czsl_model.class_names}")

    # 缓存文本特征
    czsl_model.cache_text_features(max_combination_size=2, include_single=True)

    # 测试对比学习前向传播
    print("\n2. Testing contrastive forward:")
    dummy_image = torch.randn(2, 3, 224, 224).to(device)
    dummy_text = clip.tokenize(["a radar signal with DFTJ", "a radar signal with ISRJ and VDJ"], truncate=True).to(device)

    img_feat, txt_feat = czsl_model.forward_contrastive(dummy_image, dummy_text)
    print(f"   Image features shape: {img_feat.shape}")
    print(f"   Text features shape: {txt_feat.shape}")

    # 测试零样本预测
    print("\n3. Testing zero-shot prediction:")
    sims, indices, names = czsl_model.zero_shot_predict(dummy_image, use_combinations=True, top_k=3)
    print(f"   Similarities shape: {sims.shape}")
    print(f"   Top predictions:")
    for i, batch_names in enumerate(names):
        print(f"      Batch {i}: {batch_names}")

    # 测试多标签预测
    print("\n4. Testing multi-label zero-shot prediction:")
    probs, preds = czsl_model.zero_shot_multilabel_predict(dummy_image, threshold=0.5)
    print(f"   Probs shape: {probs.shape}")
    print(f"   Predictions: {preds}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)