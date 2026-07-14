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
import torchvision.models as models
from multi.lora import apply_lora_to_model, freeze_non_lora_params, count_parameters
from multi.prompt_learner import FeatureConditionedPromptLearner


class ResNet18Visual(nn.Module):
    """
    Wrapper for ResNet18 visual encoder to mimic CLIP's visual interface.
    Exposes layer3, layer4 for freezing logic compatibility.
    """
    def __init__(self, resnet):
        super().__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.flatten = nn.Flatten()
        self.proj = None  # Will be set externally
        self.input_resolution = 224

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.proj(x)
        return x


class HybridResNetCLIP(nn.Module):
    """
    Hybrid model combining ResNet18 (ImageNet pretrained) visual encoder
    with CLIP RN50 text encoder.
    """
    def __init__(self, device="cuda", target_embed_dim=1024):
        super().__init__()

        # 1. Visual Encoder (ResNet18)
        resnet = models.resnet18(pretrained=True)
        resnet = resnet.to(device)
        self.visual = ResNet18Visual(resnet)
        self.visual.proj = nn.Linear(512, target_embed_dim).to(device)

        # 2. Text Encoder (Load from CLIP RN50)
        # Load base CLIP model to get text components
        base_model, _ = clip.load("RN50", device=device)

        # Copy text components (ensure they are registered as parameters/buffers)
        self.transformer = base_model.transformer
        self.token_embedding = base_model.token_embedding
        self.positional_embedding = base_model.positional_embedding
        self.ln_final = base_model.ln_final
        self.text_projection = base_model.text_projection
        self.logit_scale = nn.Parameter(base_model.logit_scale.clone())
        self.context_length = base_model.context_length

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        # Logic from clip.model.CLIP.encode_text
        x = self.token_embedding(text).type(self.visual.proj.weight.dtype)
        x = x + self.positional_embedding.type(self.visual.proj.weight.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.visual.proj.weight.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x


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
        use_feature_context: bool = False,
        n_ctx_per_domain: dict = None,
        patch_pooling: str = "cls",
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
            use_feature_context: 是否使用特征条件上下文 (CoOp-style)
            n_ctx_per_domain: 各域上下文 token 数
            patch_pooling: 视觉特征池化方式 ("cls" | "mean")
        """
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.use_time_domain = use_time_domain
        self.time_seq_len = time_seq_len
        self.use_feature_context = use_feature_context

        # 加载预训练 CLIP 模型
        if clip_model == "resnet18":
            # Use HybridResNetCLIP
            self.model = HybridResNetCLIP(device=device, target_embed_dim=1024) # Match RN50 dim
            self.preprocess = clip.clip._transform(self.model.visual.input_resolution)
        else:
            self.model, self.preprocess = clip.load(clip_model, device=device)

        self.model = self.model.float()  # 转换为 FP32

        # 设置 patch pooling 方式
        if hasattr(self.model, 'visual') and hasattr(self.model.visual, 'pooling'):
            self.model.visual.pooling = patch_pooling

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

        # 特征条件上下文 (CoOp-style)
        if self.use_feature_context:
            transformer_width = self.model.transformer.width
            self.prompt_learner = FeatureConditionedPromptLearner(
                transformer_width=transformer_width,
                n_ctx_per_domain=n_ctx_per_domain,
            ).to(device)
        else:
            self.prompt_learner = None

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
            # ResNet: 逐步解冻 block layer4 → layer3 → layer2 → layer1
            resnet_layers = ['layer4', 'layer3', 'layer2', 'layer1']
            n_to_unfreeze = min(layers_unfreeze, len(resnet_layers))
            for i in range(n_to_unfreeze):
                layer = getattr(self.model.visual, resnet_layers[i], None)
                if layer is not None:
                    for param in layer.parameters():
                        param.requires_grad = True
            # layers_unfreeze >= 5: 再解冻 stem (conv1, bn1, attnpool)
            if layers_unfreeze >= 5:
                for attr in ['conv1', 'bn1', 'attnpool']:
                    module = getattr(self.model.visual, attr, None)
                    if module is not None:
                        if isinstance(module, nn.Parameter):
                            module.requires_grad = True
                        else:
                            for p in module.parameters():
                                p.requires_grad = True

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """编码图像"""
        return self.model.encode_image(image)

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """编码文本"""
        return self.model.encode_text(text_tokens)

    def encode_text_with_context(
        self,
        text: torch.Tensor,
        context_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """带特征条件上下文的文本编码

        将 context_vectors 插入到 [SOT_emb | context_vectors | word_embs | EOT_emb | padding] 中，
        通过冻结的文本 Transformer 编码。

        Args:
            text: token IDs [B, 77]
            context_vectors: 上下文向量 [B, M, transformer_width]

        Returns:
            text_features: [B, embed_dim]
        """
        B = text.shape[0]
        M = context_vectors.shape[1]
        dtype = self.model.visual.conv1.weight.dtype

        # 1. Token embeddings
        token_embs = self.model.token_embedding(text).type(dtype)  # [B, 77, D]

        # 2. 定位 EOT
        eot_pos = text.argmax(dim=-1)  # [B]

        # 3. 拼接: [SOT_emb | context_vectors | word_embs(1:eot+1) | padding]
        sot_emb = token_embs[:, 0:1, :]                          # [B, 1, D]
        max_eot = eot_pos.max().item()
        word_embs = token_embs[:, 1:max_eot + 1, :]              # [B, max_eot, D]

        x = torch.cat([sot_emb, context_vectors.type(dtype), word_embs], dim=1)  # [B, 1+M+max_eot, D]

        # 补零到 77
        seq_len = x.shape[1]
        if seq_len > 77:
            x = x[:, :77, :]
        elif seq_len < 77:
            pad = torch.zeros(B, 77 - seq_len, x.shape[-1], device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

        # 4. 加位置编码
        x = x + self.model.positional_embedding.type(dtype)

        # 5. 冻结 Transformer
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # 6. LayerNorm
        x = self.model.ln_final(x).type(dtype)

        # 7. 提取偏移后的 EOT 位置特征
        new_eot_pos = (eot_pos + M).clamp(max=76)
        text_features = x[torch.arange(B, device=x.device), new_eot_pos] @ self.model.text_projection

        return text_features

    def forward_contrastive(
        self,
        image: torch.Tensor,
        text_tokens: torch.Tensor,
        time_signal: torch.Tensor = None,
        features_dict: dict = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对比学习前向传播

        Args:
            image: STFT 图像张量 [batch_size, 3, H, W]
            text_tokens: 文本 tokens [batch_size, seq_len]
            time_signal: 时域信号张量 [batch_size, seq_len] (可选)
            features_dict: 各域特征 {'time': [B,5], ...} (可选，用于 CoOp)

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
        if features_dict is not None and self.prompt_learner is not None:
            context_vectors = self.prompt_learner(features_dict)  # [B, M, D]
            text_features = self.encode_text_with_context(text_tokens, context_vectors)
        else:
            text_features = self.encode_text(text_tokens)

        # 归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        return image_features, text_features

    def forward(
        self,
        image: torch.Tensor,
        text_tokens: torch.Tensor = None,
        time_signal: torch.Tensor = None,
        features_dict: dict = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播（对比学习模式）

        Args:
            image: STFT 图像张量
            text_tokens: 文本 tokens (对比学习模式需要)
            time_signal: 时域信号张量 (可选)
            features_dict: 各域特征 (可选，用于 CoOp)

        Returns:
            image_features, text_features
        """
        return self.forward_contrastive(image, text_tokens, time_signal, features_dict)

    @torch.no_grad()
    def cache_text_features(
        self,
        max_combination_size: int = 2,
        include_single: bool = True,
        seen_combinations: list = None,
        use_translation: bool = False,
        text_style: str = "class_only"
    ):
        """
        缓存所有类别和组合的文本特征
        使用 text_templates 生成描述

        Args:
            max_combination_size: 最大组合大小
            include_single: 是否包含单干扰
            seen_combinations: 已见组合列表（用于 CZSL）
            use_translation: 是否使用翻译后的类别名称（如 "Dense False Target Jamming"）
            text_style: 文本描述风格 ("class_only" | "simple_clip")
        """
        self.eval()

        # 特征条件上下文模式下，文本特征依赖逐样本特征，无法预缓存
        if self.prompt_learner is not None:
            print("Feature-conditioned context active: skipping text feature caching (computed per-sample)")
            return

        from itertools import combinations

        # 选择描述生成函数
        if text_style == "simple_clip":
            from multi.text_templates import get_simple_clip_description as _get_desc
        else:
            from multi.text_templates import get_inference_description as _get_desc

        all_features = []
        all_names = []

        # 单干扰特征
        if include_single:
            for cls_name in self.class_names:
                desc = _get_desc([cls_name], use_translation=use_translation)

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
                    combined_desc = _get_desc(list(combo), use_translation=use_translation)
                    tokens = clip.tokenize(combined_desc, truncate=True).to(self.device)
                    features = self.encode_text(tokens)
                    features = F.normalize(features, dim=-1)
                    all_features.append(features)
                    all_names.append('+'.join(combo))
            else:
                for i, j in combinations(range(len(self.class_names)), 2):
                    cls1, cls2 = self.class_names[i], self.class_names[j]

                    combined_desc = _get_desc([cls1, cls2], use_translation=use_translation)
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
        if use_translation:
            print(f"  - Using translated names: True")

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
    def compute_text_features_for_sample(
        self,
        features_dict: dict,
        class_names: list = None,
        use_combinations: bool = True,
        seen_combinations: list = None,
        use_translation: bool = False,
    ) -> Tuple[torch.Tensor, List[str]]:
        """基于单样本特征，计算所有候选类别的文本特征

        Args:
            features_dict: 各域特征 {'time': [1,5], ...}
            class_names: 候选类别名列表
            use_combinations: 是否包含组合
            seen_combinations: 已见组合列表
            use_translation: 是否使用翻译

        Returns:
            text_features: [K, embed_dim]
            names: 类别名列表
        """
        from itertools import combinations
        from multi.text_templates import get_inference_description

        class_names = class_names or self.class_names
        context_vectors = self.prompt_learner(features_dict)  # [1, M, D]

        all_features = []
        all_names = []

        # 单类别
        for cls_name in class_names:
            desc = get_inference_description([cls_name], use_translation=use_translation)
            tokens = clip.tokenize(desc, truncate=True).to(self.device)  # [1, 77]
            ctx = context_vectors.expand(1, -1, -1)  # [1, M, D]
            feat = self.encode_text_with_context(tokens, ctx)
            feat = F.normalize(feat, dim=-1)
            all_features.append(feat)
            all_names.append(cls_name)

        # 组合
        if use_combinations and seen_combinations:
            for combo in seen_combinations:
                if len(combo) <= 1:
                    continue
                combined_desc = get_inference_description(list(combo), use_translation=use_translation)
                tokens = clip.tokenize(combined_desc, truncate=True).to(self.device)
                ctx = context_vectors.expand(1, -1, -1)
                feat = self.encode_text_with_context(tokens, ctx)
                feat = F.normalize(feat, dim=-1)
                all_features.append(feat)
                all_names.append('+'.join(combo))
        elif use_combinations:
            for i, j in combinations(range(len(class_names)), 2):
                cls1, cls2 = class_names[i], class_names[j]
                combined_desc = get_inference_description([cls1, cls2], use_translation=use_translation)
                tokens = clip.tokenize(combined_desc, truncate=True).to(self.device)
                ctx = context_vectors.expand(1, -1, -1)
                feat = self.encode_text_with_context(tokens, ctx)
                feat = F.normalize(feat, dim=-1)
                all_features.append(feat)
                all_names.append(f"{cls1}+{cls2}")

        return torch.cat(all_features, dim=0), all_names

    @torch.no_grad()
    def zero_shot_predict(
        self,
        image: torch.Tensor,
        time_signal: torch.Tensor = None,
        features_dict: dict = None,
        use_combinations: bool = True,
        top_k: int = 1,
        seen_combinations: list = None,
        use_translation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        零样本预测

        Args:
            image: 图像张量 [batch_size, 3, H, W]
            time_signal: 时域信号张量 (可选)
            features_dict: 各域特征 (可选，用于 CoOp)
            use_combinations: 是否使用组合特征
            top_k: 返回 top-k 个预测
            seen_combinations: 已见组合列表
            use_translation: 是否使用翻译

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

        # 特征条件上下文路径: 逐样本计算文本特征
        if features_dict is not None and self.prompt_learner is not None:
            B = image.shape[0]
            all_similarities = []

            for i in range(B):
                sample_features = {d: f[i:i+1] for d, f in features_dict.items()}
                text_features, names = self.compute_text_features_for_sample(
                    sample_features,
                    use_combinations=use_combinations,
                    seen_combinations=seen_combinations,
                    use_translation=use_translation,
                )
                logit_scale = self.model.logit_scale.exp()
                sim = logit_scale * (image_features[i:i+1] @ text_features.T)
                all_similarities.append(sim)

            similarities = torch.cat(all_similarities, dim=0)
        else:
            # 原始缓存路径
            if use_combinations:
                text_features = self.get_cached_combination_features()
                names = self._combination_names
            else:
                text_features = self.get_cached_text_features()
                names = self.class_names

            logit_scale = self.model.logit_scale.exp()
            similarities = logit_scale * (image_features @ text_features.T)

        # 获取 top-k 预测
        top_k = min(top_k, similarities.shape[-1])
        values, indices = torch.topk(similarities, k=top_k, dim=-1)

        # 获取预测名称
        pred_names = [[names[idx.item()] for idx in batch_indices] for batch_indices in indices]

        return similarities, indices, pred_names

def create_czsl_model(config: dict, device: str = "cuda"):
    """
    创建 CZSL 模型（支持 CLIP) 

    Args:
        config: 配置字典
        device: 计算设备

    Returns:
        CLIPForCZSL
    """

    model_config = config.get("model", {})
    data_config = config.get("data", {})
    class_names = [cls_info["name"] for cls_info in config.get("jamming_classes", [])]

    # 时域配置
    use_time_domain = config.get("use_time_domain", False)
    time_seq_len = data_config.get("time_seq_len", 8000)

    # 特征条件上下文配置
    use_feature_context = config.get("use_feature_context", False)
    n_ctx_per_domain = config.get("n_ctx_per_domain", None)

    clip_model = model_config.get("clip_model", "ViT-B/32")

    # 使用 CLIP 模型
    print(f"\nCreating CLIPForCZSL with model: {clip_model}")
    model = CLIPForCZSL(
        clip_model=clip_model,
        num_classes=len(class_names),
        class_names=class_names,
        freeze_vision=model_config.get("freeze_vision", False),
        freeze_text=model_config.get("freeze_text", True),
        vision_layers_unfreeze=model_config.get("vision_layers_unfreeze", 2),
        device=device,
        use_time_domain=use_time_domain,
        time_seq_len=time_seq_len,
        use_feature_context=use_feature_context,
        n_ctx_per_domain=n_ctx_per_domain,
        patch_pooling=model_config.get("patch_pooling", "cls"),
    )

    # LoRA 配置
    lora_config = config.get("lora", {})
    if lora_config.get("enabled", False):
        print("\n" + "=" * 60)
        print("Applying LoRA to model...")
        print("=" * 60)

        target_modules = lora_config.get("target_modules", ["attn"])
        if isinstance(target_modules, str):
            target_modules = [target_modules]

        replaced = apply_lora_to_model(
            model.model,
            target_modules=target_modules,
            rank=lora_config.get("rank", 8),
            alpha=lora_config.get("alpha", 16.0),
            dropout=lora_config.get("dropout", 0.0)
        )

        # 冻结非 LoRA 参数
        freeze_non_lora_params(model.model)

        # 统计参数
        stats = count_parameters(model.model)
        print(f"\nLoRA applied to {replaced} layers")
        print(f"Trainable parameters: {stats['trainable']:,} ({stats['trainable_ratio']:.2%})")
        print("=" * 60)

    return model


class DualBranchCLIPForCZSL(nn.Module):
    """
    双分支 CLIP 模型 - 用于欺骗/压制干扰分类

    架构:
    - 共享 CLIP 视觉编码器
    - 欺骗干扰分支: 独立文本特征空间 (5种欺骗 + 无欺骗)
    - 压制干扰分支: 独立文本特征空间 (9种压制 + 无压制)
    """

    def __init__(
        self,
        clip_model: str = "ViT-B/32",
        deception_classes: List[str] = None,
        suppression_classes: List[str] = None,
        freeze_vision: bool = False,
        freeze_text: bool = True,
        vision_layers_unfreeze: int = 2,
        device: str = "cuda",
        use_time_domain: bool = False,
        time_seq_len: int = 8000,
        use_feature_context: bool = False,
        n_ctx_per_domain: dict = None,
        patch_pooling: str = "cls",
    ):
        """
        初始化双分支 CZSL-CLIP 模型

        Args:
            clip_model: CLIP 模型名称
            deception_classes: 欺骗干扰类型列表
            suppression_classes: 压制干扰类型列表
            freeze_vision: 是否冻结视觉编码器
            freeze_text: 是否冻结文本编码器
            vision_layers_unfreeze: 解冻视觉编码器最后 N 层
            device: 计算设备
            use_time_domain: 是否使用时域信号
            time_seq_len: 时域信号序列长度
            use_feature_context: 是否使用特征条件上下文 (CoOp-style)
            n_ctx_per_domain: 各域上下文 token 数
            patch_pooling: 视觉特征池化方式 ("cls" | "mean")
        """
        super().__init__()
        self.device = device
        self.deception_classes = deception_classes or []
        self.suppression_classes = suppression_classes or []

        # 添加 "无XX干扰" 类
        self.deception_classes_with_none = self.deception_classes + ["无欺骗干扰"]
        self.suppression_classes_with_none = self.suppression_classes + ["无压制干扰"]

        self.num_deception_classes = len(self.deception_classes_with_none)
        self.num_suppression_classes = len(self.suppression_classes_with_none)

        self.use_time_domain = use_time_domain
        self.time_seq_len = time_seq_len
        self.use_feature_context = use_feature_context

        # 加载预训练 CLIP 模型
        if clip_model == "resnet18":
            self.model = HybridResNetCLIP(device=device, target_embed_dim=1024)
            self.preprocess = clip.clip._transform(self.model.visual.input_resolution)
        else:
            self.model, self.preprocess = clip.load(clip_model, device=device)

        self.model = self.model.float()

        # 设置 patch pooling 方式
        if hasattr(self.model, 'visual') and hasattr(self.model.visual, 'pooling'):
            self.model.visual.pooling = patch_pooling

        # 获取特征维度
        self.embed_dim = self.model.text_projection.shape[1]

        # 应用冻结策略
        self._apply_freeze_strategy(
            freeze_vision=freeze_vision,
            freeze_text=freeze_text,
            vision_layers_unfreeze=vision_layers_unfreeze
        )

        # 时域编码器 (如果启用)
        if self.use_time_domain:
            self.time_encoder = TimeDomainTransformerEncoder(
                embed_dim=self.embed_dim,
                num_heads=8,
                num_layers=3,
                seq_len=time_seq_len
            ).to(device)
            self.fusion_projection = nn.Linear(self.embed_dim * 2, self.embed_dim).to(device)

        # 缓存文本特征
        self._deception_text_features = None
        self._suppression_text_features = None
        self._deception_names = None
        self._suppression_names = None

        # 特征条件上下文 (CoOp-style)
        if self.use_feature_context:
            transformer_width = self.model.transformer.width
            self.prompt_learner = FeatureConditionedPromptLearner(
                transformer_width=transformer_width,
                n_ctx_per_domain=n_ctx_per_domain,
            ).to(device)
        else:
            self.prompt_learner = None

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
            # ResNet: 逐步解冻 block layer4 → layer3 → layer2 → layer1
            resnet_layers = ['layer4', 'layer3', 'layer2', 'layer1']
            n_to_unfreeze = min(layers_unfreeze, len(resnet_layers))
            for i in range(n_to_unfreeze):
                layer = getattr(self.model.visual, resnet_layers[i], None)
                if layer is not None:
                    for param in layer.parameters():
                        param.requires_grad = True
            # layers_unfreeze >= 5: 再解冻 stem (conv1, bn1, attnpool)
            if layers_unfreeze >= 5:
                for attr in ['conv1', 'bn1', 'attnpool']:
                    module = getattr(self.model.visual, attr, None)
                    if module is not None:
                        if isinstance(module, nn.Parameter):
                            module.requires_grad = True
                        else:
                            for p in module.parameters():
                                p.requires_grad = True

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """编码图像"""
        return self.model.encode_image(image)

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """编码文本"""
        return self.model.encode_text(text_tokens)

    def encode_text_with_context(
        self,
        text: torch.Tensor,
        context_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """带特征条件上下文的文本编码（双分支共享）"""
        B = text.shape[0]
        M = context_vectors.shape[1]
        dtype = self.model.visual.conv1.weight.dtype

        token_embs = self.model.token_embedding(text).type(dtype)
        eot_pos = text.argmax(dim=-1)

        sot_emb = token_embs[:, 0:1, :]
        max_eot = eot_pos.max().item()
        word_embs = token_embs[:, 1:max_eot + 1, :]

        x = torch.cat([sot_emb, context_vectors.type(dtype), word_embs], dim=1)

        seq_len = x.shape[1]
        if seq_len > 77:
            x = x[:, :77, :]
        elif seq_len < 77:
            pad = torch.zeros(B, 77 - seq_len, x.shape[-1], device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

        x = x + self.model.positional_embedding.type(dtype)
        x = x.permute(1, 0, 2)
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.model.ln_final(x).type(dtype)

        new_eot_pos = (eot_pos + M).clamp(max=76)
        text_features = x[torch.arange(B, device=x.device), new_eot_pos] @ self.model.text_projection

        return text_features

    def forward_dual(
        self,
        image: torch.Tensor,
        text_tokens_deception: torch.Tensor,
        text_tokens_suppression: torch.Tensor,
        time_signal: torch.Tensor = None,
        features_dict: dict = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        双分支前向传播

        Args:
            image: STFT 图像
            text_tokens_deception: 欺骗分支文本 tokens
            text_tokens_suppression: 压制分支文本 tokens
            time_signal: 时域信号 (可选)
            features_dict: 各域特征 (可选，用于 CoOp)

        Returns:
            image_features: 图像特征
            text_features_deception: 欺骗分支文本特征
            text_features_suppression: 压制分支文本特征
        """
        # 提取图像特征
        stft_features = self.encode_image(image)

        if self.use_time_domain and time_signal is not None:
            time_features = self.time_encoder(time_signal)
            fused_features = torch.cat([stft_features, time_features], dim=-1)
            image_features = self.fusion_projection(fused_features)
        else:
            image_features = stft_features

        # 提取两个分支的文本特征
        if features_dict is not None and self.prompt_learner is not None:
            context_vectors = self.prompt_learner(features_dict)
            text_features_deception = self.encode_text_with_context(text_tokens_deception, context_vectors)
            text_features_suppression = self.encode_text_with_context(text_tokens_suppression, context_vectors)
        else:
            text_features_deception = self.encode_text(text_tokens_deception)
            text_features_suppression = self.encode_text(text_tokens_suppression)

        # 归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features_deception = F.normalize(text_features_deception, dim=-1)
        text_features_suppression = F.normalize(text_features_suppression, dim=-1)

        return image_features, text_features_deception, text_features_suppression

    def forward(
        self,
        image: torch.Tensor,
        text_tokens_deception: torch.Tensor = None,
        text_tokens_suppression: torch.Tensor = None,
        time_signal: torch.Tensor = None,
        features_dict: dict = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        return self.forward_dual(image, text_tokens_deception, text_tokens_suppression, time_signal, features_dict)

    @torch.no_grad()
    def cache_text_features_dual(self, use_translation: bool = False):
        """
        缓存双分支文本特征
        """
        self.eval()

        # 特征条件上下文模式下，文本特征依赖逐样本特征，无法预缓存
        if self.prompt_learner is not None:
            print("Feature-conditioned context active: skipping dual-branch text feature caching")
            return

        from multi.text_templates import get_dual_branch_inference_descriptions

        descriptions = get_dual_branch_inference_descriptions(
            self.deception_classes,
            self.suppression_classes,
            use_translation=use_translation
        )

        # 欺骗分支
        deception_texts = descriptions['deception']
        self._deception_names = descriptions['deception_names']
        deception_tokens = clip.tokenize(deception_texts, truncate=True).to(self.device)
        self._deception_text_features = self.encode_text(deception_tokens)
        self._deception_text_features = F.normalize(self._deception_text_features, dim=-1)

        # 压制分支
        suppression_texts = descriptions['suppression']
        self._suppression_names = descriptions['suppression_names']
        suppression_tokens = clip.tokenize(suppression_texts, truncate=True).to(self.device)
        self._suppression_text_features = self.encode_text(suppression_tokens)
        self._suppression_text_features = F.normalize(self._suppression_text_features, dim=-1)

        print(f"Cached dual-branch text features:")
        print(f"  - Deception: {len(self._deception_names)} classes")
        print(f"  - Suppression: {len(self._suppression_names)} classes")

    def get_deception_text_features(self) -> torch.Tensor:
        """获取欺骗分支文本特征"""
        if self._deception_text_features is None:
            self.cache_text_features_dual()
        return self._deception_text_features

    def get_suppression_text_features(self) -> torch.Tensor:
        """获取压制分支文本特征"""
        if self._suppression_text_features is None:
            self.cache_text_features_dual()
        return self._suppression_text_features

    @torch.no_grad()
    def zero_shot_predict_dual(
        self,
        image: torch.Tensor,
        time_signal: torch.Tensor = None,
        top_k: int = 1
    ) -> Tuple[dict, dict]:
        """
        双分支零样本预测

        Args:
            image: 图像张量
            time_signal: 时域信号 (可选)
            top_k: 返回 top-k 预测

        Returns:
            deception_result: {'similarities', 'indices', 'names'}
            suppression_result: {'similarities', 'indices', 'names'}
        """
        self.eval()

        # 编码图像
        stft_features = self.encode_image(image)

        if self.use_time_domain and time_signal is not None:
            time_features = self.time_encoder(time_signal)
            fused_features = torch.cat([stft_features, time_features], dim=-1)
            image_features = self.fusion_projection(fused_features)
        else:
            image_features = stft_features

        image_features = F.normalize(image_features, dim=-1)

        logit_scale = self.model.logit_scale.exp()

        # 欺骗分支预测
        deception_features = self.get_deception_text_features()
        deception_similarities = logit_scale * (image_features @ deception_features.T)
        top_k_deception = min(top_k, deception_features.shape[0])
        deception_values, deception_indices = torch.topk(deception_similarities, k=top_k_deception, dim=-1)
        deception_names = [[self._deception_names[idx.item()] for idx in batch_indices]
                          for batch_indices in deception_indices]

        deception_result = {
            'similarities': deception_similarities,
            'indices': deception_indices,
            'names': deception_names
        }

        # 压制分支预测
        suppression_features = self.get_suppression_text_features()
        suppression_similarities = logit_scale * (image_features @ suppression_features.T)
        top_k_suppression = min(top_k, suppression_features.shape[0])
        suppression_values, suppression_indices = torch.topk(suppression_similarities, k=top_k_suppression, dim=-1)
        suppression_names = [[self._suppression_names[idx.item()] for idx in batch_indices]
                            for batch_indices in suppression_indices]

        suppression_result = {
            'similarities': suppression_similarities,
            'indices': suppression_indices,
            'names': suppression_names
        }

        return deception_result, suppression_result


def create_dual_branch_model(config: dict, device: str = "cuda") -> DualBranchCLIPForCZSL:
    """
    创建双分支 CZSL 模型

    Args:
        config: 配置字典
        device: 计算设备

    Returns:
        DualBranchCLIPForCZSL 模型实例
    """
    model_config = config.get("model", {})
    data_config = config.get("data", {})

    # 从 jamming_groups 获取分类
    jamming_groups = config.get("jamming_groups", {})
    deception_classes = jamming_groups.get("deception", {}).get("classes", [])
    suppression_classes = jamming_groups.get("suppression", {}).get("classes", [])

    if not deception_classes or not suppression_classes:
        # 回退到默认分组
        all_classes = [cls_info["name"] for cls_info in config.get("jamming_classes", [])]
        # 根据常见干扰类型分组
        deception_classes = ["DFTJ", "ISRJ", "SMSPJ", "C&IJ", "CSJ"]
        suppression_classes = ["AJ", "BJ", "SJ", "NCJ", "NPJ", "NFMJ", "NPMJ", "NAMJ", "PJ"]
        # 过滤出实际存在的类别
        deception_classes = [c for c in deception_classes if c in all_classes]
        suppression_classes = [c for c in suppression_classes if c in all_classes]

    # 时域配置
    use_time_domain = config.get("use_time_domain", False)
    time_seq_len = data_config.get("time_seq_len", 8000)

    # 特征条件上下文配置
    use_feature_context = config.get("use_feature_context", False)
    n_ctx_per_domain = config.get("n_ctx_per_domain", None)

    model = DualBranchCLIPForCZSL(
        clip_model=model_config.get("clip_model", "ViT-B/32"),
        deception_classes=deception_classes,
        suppression_classes=suppression_classes,
        freeze_vision=model_config.get("freeze_vision", False),
        freeze_text=model_config.get("freeze_text", True),
        vision_layers_unfreeze=model_config.get("vision_layers_unfreeze", 2),
        device=device,
        use_time_domain=use_time_domain,
        time_seq_len=time_seq_len,
        use_feature_context=use_feature_context,
        n_ctx_per_domain=n_ctx_per_domain,
        patch_pooling=model_config.get("patch_pooling", "cls"),
    )

    # LoRA 配置
    lora_config = config.get("lora", {})
    if lora_config.get("enabled", False):
        print("\n" + "=" * 60)
        print("Applying LoRA to dual-branch model...")
        print("=" * 60)

        target_modules = lora_config.get("target_modules", ["attn"])
        if isinstance(target_modules, str):
            target_modules = [target_modules]

        replaced = apply_lora_to_model(
            model.model,
            target_modules=target_modules,
            rank=lora_config.get("rank", 8),
            alpha=lora_config.get("alpha", 16.0),
            dropout=lora_config.get("dropout", 0.0)
        )

        freeze_non_lora_params(model.model)

        stats = count_parameters(model.model)
        print(f"\nLoRA applied to {replaced} layers")
        print(f"Trainable parameters: {stats['trainable']:,} ({stats['trainable_ratio']:.2%})")
        print("=" * 60)

    return model


# ============================================================================
# ResNet18 双分支分类模型 - 用于消融实验对比
# ============================================================================

class DualBranchResNet18(nn.Module):
    """
    双分支 ResNet18 分类模型 - 用于消融实验

    架构:
    - 共享 ResNet18 视觉编码器
    - 欺骗干扰分类头 (num_deception_classes)
    - 压制干扰分类头 (num_suppression_classes)

    使用标准交叉熵损失训练，不是 CLIP 对比学习。
    """

    def __init__(
        self,
        deception_classes: List[str] = None,
        suppression_classes: List[str] = None,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        device: str = "cuda"
    ):
        """
        初始化双分支 ResNet18 模型

        Args:
            deception_classes: 欺骗干扰类型列表
            suppression_classes: 压制干扰类型列表
            pretrained: 是否使用 ImageNet 预训练权重
            freeze_backbone: 是否冻结骨干网络
            device: 计算设备
        """
        super().__init__()
        self.device = device
        self.deception_classes = deception_classes or []
        self.suppression_classes = suppression_classes or []

        # 包含 "无XX干扰"
        self.deception_classes_with_none = self.deception_classes + ["无欺骗干扰"]
        self.suppression_classes_with_none = self.suppression_classes + ["无压制干扰"]

        self.num_deception_classes = len(self.deception_classes_with_none)
        self.num_suppression_classes = len(self.suppression_classes_with_none)

        # 加载预训练 ResNet18
        resnet = models.resnet18(pretrained=pretrained)

        # 移除原始全连接层
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # 输出: [batch, 512, 1, 1]
        self.feature_dim = 512

        # 欺骗分支分类头
        self.deception_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_deception_classes)
        )

        # 压制分支分类头
        self.suppression_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_suppression_classes)
        )

        # 冻结骨干网络
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.to(device)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取图像特征"""
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # [batch, 512]
        return features

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入图像 [batch, 3, H, W]

        Returns:
            logits_deception: 欺骗分支 logits [batch, num_deception_classes]
            logits_suppression: 压制分支 logits [batch, num_suppression_classes]
        """
        features = self.extract_features(x)

        logits_deception = self.deception_classifier(features)
        logits_suppression = self.suppression_classifier(features)

        return logits_deception, logits_suppression

    def predict(self, x: torch.Tensor) -> Tuple[dict, dict]:
        """
        预测

        Args:
            x: 输入图像

        Returns:
            deception_result: {'indices', 'names', 'probs'}
            suppression_result: {'indices', 'names', 'probs'}
        """
        self.eval()
        with torch.no_grad():
            logits_deception, logits_suppression = self.forward(x)

            probs_deception = F.softmax(logits_deception, dim=-1)
            probs_suppression = F.softmax(logits_suppression, dim=-1)

            pred_deception = torch.argmax(probs_deception, dim=-1)
            pred_suppression = torch.argmax(probs_suppression, dim=-1)

        batch_size = x.size(0)

        deception_names = [[self.deception_classes_with_none[idx.item()]] for idx in pred_deception]
        suppression_names = [[self.suppression_classes_with_none[idx.item()]] for idx in pred_suppression]

        deception_result = {
            'indices': pred_deception.unsqueeze(1),
            'names': deception_names,
            'probs': probs_deception
        }

        suppression_result = {
            'indices': pred_suppression.unsqueeze(1),
            'names': suppression_names,
            'probs': probs_suppression
        }

        return deception_result, suppression_result


def create_resnet18_dual_branch_model(config: dict, device: str = "cuda") -> DualBranchResNet18:
    """
    创建双分支 ResNet18 模型

    Args:
        config: 配置字典
        device: 计算设备

    Returns:
        DualBranchResNet18 模型实例
    """
    model_config = config.get("model", {})

    # 从 jamming_groups 获取分类
    jamming_groups = config.get("jamming_groups", {})
    deception_classes = jamming_groups.get("deception", {}).get("classes", ["DFTJ", "ISRJ", "SMSPJ", "C&IJ", "CSJ"])
    suppression_classes = jamming_groups.get("suppression", {}).get("classes", ["AJ", "BJ", "SJ", "NCJ", "NPJ", "NFMJ", "NPMJ", "NAMJ", "PJ"])

    model = DualBranchResNet18(
        deception_classes=deception_classes,
        suppression_classes=suppression_classes,
        pretrained=model_config.get("pretrained", True),
        freeze_backbone=model_config.get("freeze_backbone", False),
        device=device
    )

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nDualBranchResNet18 created:")
    print(f"  - Deception classes: {len(deception_classes)} + 1 (无欺骗干扰)")
    print(f"  - Suppression classes: {len(suppression_classes)} + 1 (无压制干扰)")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")

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
            "clip_model": "resnet18",
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
