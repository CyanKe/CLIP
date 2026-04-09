"""
CLIP 模型 - 用于雷达干扰信号的组合零样本学习 (CZSL)
仅保留对比学习所需的核心功能
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import sys
import os
import math

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clip


class TimeDomainTransformerEncoder(nn.Module):
    """
    基于 Transformer 的时域信号编码器

    输入: (batch, seq_len) 的 1D 时域信号
    输出: (batch, embed_dim) 的特征向量
    """

    def __init__(self, embed_dim=512, num_heads=8, num_layers=3, seq_len=8000, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len

        # 输入投影：将 1D 信号投影到 embed_dim
        self.input_proj = nn.Linear(1, embed_dim)

        # 位置编码 (可学习)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

        # Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出投影
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len) 的时域信号

        Returns:
            (batch, embed_dim) 的特征向量
        """
        # x: (batch, seq_len) -> (batch, seq_len, 1)
        x = x.unsqueeze(-1)

        # 投影到 embed_dim
        x = self.input_proj(x)  # (batch, seq_len, embed_dim)

        # 添加位置编码
        x = x + self.pos_encoder

        # Transformer 编码
        x = self.transformer(x)  # (batch, seq_len, embed_dim)

        # Global average pooling
        x = x.mean(dim=1)  # (batch, embed_dim)

        # 输出投影
        x = self.output_proj(x)

        return x


class CLIPForCZSL(nn.Module):
    """
    CLIP 模型用于组合零样本学习 (CZSL)

    支持对比学习训练和零样本推理
    """

    def __init__(
        self,
        clip_model: str = "ViT-B/32",
        num_classes: int = 16,
        class_names: List[str] = None,
        freeze_vision: bool = False,
        freeze_text: bool = True,
        vision_layers_unfreeze: int = 2,
        device: str = "cuda",
        use_time_domain: bool = False,
        time_seq_len: int = 8000,
    ):
        """
        初始化 CZSL-CLIP 模型

        Args:
            clip_model: CLIP 模型名称
            num_classes: 类别数
            class_names: 类别名称列表
            freeze_vision: 是否冻结视觉编码器
            freeze_text: 是否冻结文本编码器
            vision_layers_unfreeze: 解冻视觉编码器最后 N 层
            device: 计算设备
            use_time_domain: 是否使用时域信号
            time_seq_len: 时域信号序列长度
        """
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.use_time_domain = use_time_domain
        self.time_seq_len = time_seq_len

        # 加载预训练 CLIP 模型
        self.model, self.preprocess = clip.load(clip_model, device=device)
        self.model = self.model.float()  # 转换为 FP32

        # 获取特征维度
        self.embed_dim = self.model.text_projection.shape[1]

        # 应用冻结策略
        self._apply_freeze_strategy(
            freeze_vision=freeze_vision,
            freeze_text=freeze_text,
            vision_layers_unfreeze=vision_layers_unfreeze
        )

        # 初始化时域编码器和融合层（如果启用时域信号）
        if self.use_time_domain:
            # 时域编码器
            self.time_encoder = TimeDomainTransformerEncoder(
                embed_dim=self.embed_dim,
                num_heads=8,
                num_layers=3,
                seq_len=time_seq_len
            ).to(device)

            # 融合投影层：拼接后投影回 embed_dim
            self.fusion_projection = nn.Linear(self.embed_dim * 2, self.embed_dim).to(device)

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
        text_tokens: torch.Tensor,
        time_signal: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对比学习前向传播

        Args:
            image: STFT 图像张量 [batch_size, 3, H, W]
            text_tokens: 文本 tokens [batch_size, seq_len]
            time_signal: 时域信号张量 [batch_size, seq_len] (可选)

        Returns:
            image_features: 图像特征 [batch_size, embed_dim]
            text_features: 文本特征 [batch_size, embed_dim]
        """
        # 提取 STFT 特征
        stft_features = self.encode_image(image)

        # 如果启用时域信号且提供了时域数据，进行融合
        if self.use_time_domain and time_signal is not None:
            # 提取时域特征
            time_features = self.time_encoder(time_signal)

            # 拼接特征
            fused_features = torch.cat([stft_features, time_features], dim=-1)

            # 投影回 embed_dim
            image_features = self.fusion_projection(fused_features)
        else:
            # 仅使用 STFT 特征
            image_features = stft_features

        # 提取文本特征
        text_features = self.encode_text(text_tokens)

        # 归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        return image_features, text_features

    def forward(
        self,
        image: torch.Tensor,
        text_tokens: torch.Tensor = None,
        time_signal: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播（对比学习模式）

        Args:
            image: STFT 图像张量
            text_tokens: 文本 tokens (对比学习模式需要)
            time_signal: 时域信号张量 (可选)

        Returns:
            image_features, text_features
        """
        return self.forward_contrastive(image, text_tokens, time_signal)

    @torch.no_grad()
    def cache_text_features(
        self,
        max_combination_size: int = 2,
        include_single: bool = True,
        seen_combinations: list = None
    ):
        """
        缓存所有类别和组合的文本特征
        使用 text_templates 生成描述

        Args:
            max_combination_size: 最大组合大小
            include_single: 是否包含单干扰
            seen_combinations: 已见组合列表（用于 CZSL）
        """
        self.eval()
        from itertools import combinations
        from multi.text_templates import JAM_TYPE_NAMES, VISUAL_TEMPLATES

        all_features = []
        all_names = []

        # 单干扰特征
        if include_single:
            for cls_name in self.class_names:
                jam_type = None
                for k, v in JAM_TYPE_NAMES.items():
                    if v == cls_name:
                        jam_type = k
                        break

                if jam_type:
                    template = VISUAL_TEMPLATES.get(jam_type, {'base': cls_name})
                    desc = f"{cls_name} looks like {template['base']}"
                else:
                    desc = f"a radar signal with {cls_name}"

                tokens = clip.tokenize(desc, truncate=True).to(self.device)
                features = self.encode_text(tokens)
                features = F.normalize(features, dim=-1)
                all_features.append(features)
                all_names.append(cls_name)

        # 组合特征
        if max_combination_size >= 2:
            if seen_combinations:
                combo_only = [c for c in seen_combinations if len(c) > 1]
                for combo in combo_only:
                    descs = []
                    for cls_name in combo:
                        jam_type = None
                        for k, v in JAM_TYPE_NAMES.items():
                            if v == cls_name:
                                jam_type = k
                                break
                        if jam_type:
                            template = VISUAL_TEMPLATES.get(jam_type, {'base': cls_name})
                            descs.append(f"{cls_name} looks like {template['base']}")
                        else:
                            descs.append(f"a radar signal with {cls_name}")

                    combined_desc = ', '.join(descs)
                    tokens = clip.tokenize(combined_desc, truncate=True).to(self.device)
                    features = self.encode_text(tokens)
                    features = F.normalize(features, dim=-1)
                    all_features.append(features)
                    all_names.append('+'.join(combo))
            else:
                for i, j in combinations(range(len(self.class_names)), 2):
                    cls1, cls2 = self.class_names[i], self.class_names[j]

                    jam_type1 = jam_type2 = None
                    for k, v in JAM_TYPE_NAMES.items():
                        if v == cls1:
                            jam_type1 = k
                        if v == cls2:
                            jam_type2 = k

                    descs = []
                    for cls_name, jam_type in [(cls1, jam_type1), (cls2, jam_type2)]:
                        if jam_type:
                            template = VISUAL_TEMPLATES.get(jam_type, {'base': cls_name})
                            descs.append(f"{cls_name} looks like {template['base']}")
                        else:
                            descs.append(f"a radar signal with {cls_name}")

                    combined_desc = ', '.join(descs)
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
        if seen_combinations:
            print(f"  - Seen combinations (from config): {len(all_names) - len(self.class_names)}")
        else:
            print(f"  - Combinations (all pairs): {len(all_names) - len(self.class_names)}")

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
        time_signal: torch.Tensor = None,
        use_combinations: bool = True,
        top_k: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        零样本预测

        Args:
            image: 图像张量 [batch_size, 3, H, W]
            use_combinations: 是否使用组合特征
            top_k: 返回 top-k 个预测

        Returns:
            similarities: 相似度分数 [batch_size, num_candidates]
            indices: 预测索引 [batch_size, top_k]
            names: 预测名称列表
        """
        self.eval()

        # 编码 STFT 图像
        stft_features = self.encode_image(image)

        # 如果启用时域信号且提供了时域数据，进行融合
        if self.use_time_domain and time_signal is not None:
            # 提取时域特征
            time_features = self.time_encoder(time_signal)

            # 拼接特征
            fused_features = torch.cat([stft_features, time_features], dim=-1)

            # 投影回 embed_dim
            image_features = self.fusion_projection(fused_features)
        else:
            # 仅使用 STFT 特征
            image_features = stft_features

        # 归一化
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

        # 获取 top-k 预测
        top_k = min(top_k, text_features.shape[0])
        values, indices = torch.topk(similarities, k=top_k, dim=-1)

        # 获取预测名称
        pred_names = [[names[idx.item()] for idx in batch_indices] for batch_indices in indices]

        return similarities, indices, pred_names


def create_czsl_model(config: dict, device: str = "cuda") -> CLIPForCZSL:
    """
    创建 CZSL 模型

    Args:
        config: 配置字典
        device: 计算设备

    Returns:
        CLIPForCZSL 模型实例
    """
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    class_names = [cls_info["name"] for cls_info in config.get("jamming_classes", [])]

    # 时域配置
    use_time_domain = config.get("use_time_domain", False)
    time_seq_len = data_config.get("time_seq_len", 8000)

    model = CLIPForCZSL(
        clip_model=model_config.get("clip_model", "ViT-B/32"),
        num_classes=len(class_names),
        class_names=class_names,
        freeze_vision=model_config.get("freeze_vision", False),
        freeze_text=model_config.get("freeze_text", True),
        vision_layers_unfreeze=model_config.get("vision_layers_unfreeze", 2),
        device=device,
        use_time_domain=use_time_domain,
        time_seq_len=time_seq_len
    )

    return model


if __name__ == "__main__":
    # 测试代码
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("=" * 60)
    print("Testing CLIPForCZSL")
    print("=" * 60)

    test_config = {
        "model": {
            "clip_model": "ViT-B/32",
            "freeze_vision": False,
            "freeze_text": True,
            "vision_layers_unfreeze": 2
        },
        "jamming_classes": [
            {"name": "DFTJ"},
            {"name": "ISRJ"},
            {"name": "VGPO"},
            {"name": "RGPO"},
        ]
    }

    # 测试模型创建
    print("\n1. Testing model creation:")
    czsl_model = create_czsl_model(test_config, device)
    print(f"   Embed dim: {czsl_model.embed_dim}")
    print(f"   Num classes: {czsl_model.num_classes}")
    print(f"   Class names: {czsl_model.class_names}")

    # 缓存文本特征
    czsl_model.cache_text_features(max_combination_size=2, include_single=True)

    # 测试对比学习前向传播
    print("\n2. Testing contrastive forward:")
    dummy_image = torch.randn(2, 3, 224, 224).to(device)
    dummy_text = clip.tokenize(["a radar signal with DFTJ", "a radar signal with ISRJ"], truncate=True).to(device)

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

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
