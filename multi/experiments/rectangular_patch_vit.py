import os
import sys
"""
条形 Patch ViT 视觉编码器

支持非正方形的 patch,适合 STFT 时频图的特征提取：
- 横向条形 (如 8x32): 覆盖较长时间段、较窄频段
- 纵向条形 (如 32x8): 覆盖较宽频段、较短时间

用于消融实验对比。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PatchEmbed(nn.Module):
    """
    条形 Patch Embedding

    将图像分割为非正方形的 patch 并投影到嵌入空间
    支持任意 patch size，不要求整除图像尺寸
    不能整除时使用补零（Padding）而非裁切（Cropping）:
    - 时间轴 (Width): 仅右侧补零（原点 t=0 在左侧）
    - 频率轴 (Height): 上下对称补零（原点 f=0 在顶部）
    以保护 STFT 时频图的绝对坐标映射。
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: Tuple[int, int] = (16, 16),  # (height, width)
        in_chans: int = 3,
        embed_dim: int = 768,
        strict_mode: bool = False,  # True: 要求整除; False: 自动处理
    ):
        """
        Args:
            img_size: 输入图像尺寸 (正方形)
            patch_size: patch 尺寸 (patch_height, patch_width)
            in_chans: 输入通道数
            embed_dim: 嵌入维度
            strict_mode: 是否严格要求 patch_size 整除 img_size
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_height = patch_size[0]
        self.patch_width = patch_size[1]
        self.strict_mode = strict_mode

        # 计算每个维度上的 patch 数量
        # 使用 ceil，不足部分通过 padding 补齐（而非裁切）
        self.num_patches_h = math.ceil(img_size / self.patch_height)
        self.num_patches_w = math.ceil(img_size / self.patch_width)
        self.num_patches = self.num_patches_h * self.num_patches_w

        # 计算 padding 后的有效尺寸
        self.padded_h = self.num_patches_h * self.patch_height
        self.padded_w = self.num_patches_w * self.patch_width

        # 检查是否能整除
        self.is_perfect_fit = (img_size % self.patch_height == 0) and (img_size % self.patch_width == 0)

        if not self.is_perfect_fit:
            pad_h = self.padded_h - img_size
            pad_w = self.padded_w - img_size
            if strict_mode:
                raise ValueError(
                    f"patch_size {patch_size} cannot evenly divide img_size {img_size}. "
                    f"Would pad: {pad_h}x{pad_w}. "
                    f"Set strict_mode=False to allow this."
                )
            else:
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                print(f"  Warning: patch_size {patch_size} doesn't evenly divide {img_size}. "
                      f"Padding: top={pad_top} bottom={pad_bottom} right={pad_w}. "
                      f"Effective grid: {self.num_patches_h}x{self.num_patches_w} = {self.num_patches} patches")

        # 使用 Conv2d 实现 patch embedding
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] 输入图像

        Returns:
            [B, num_patches, embed_dim] patch embeddings
        """
        B, C, H, W = x.shape

        # 验证输入尺寸
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}x{W}) doesn't match expected ({self.img_size}x{self.img_size})"

        # 如果不能完美覆盖，补齐像素
        # F.pad format: (left, right, top, bottom) — last dimension (width) first!
        #
        # STFT 时频图坐标约定（原点在左上角: t=0, f=0）:
        #   - 时间轴 (Width, dim=-1): 原点在左侧 → 仅右侧补零 (left=0, right=pad_w)
        #   - 频率轴 (Height, dim=-2): 原点在顶部 → 上下对称补零 (top=pad_top, bottom=pad_bottom)
        #     对称补零保持频率中心不变，避免频域坐标偏移
        if not self.is_perfect_fit:
            pad_w = self.padded_w - W
            pad_h = self.padded_h - H
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top  # 奇数像素余量分配给底部
            if pad_w > 0 or pad_h > 0:
                x = F.pad(x, (0, pad_w, pad_top, pad_bottom))

        # Conv2d: [B, C, H, W] -> [B, embed_dim, num_patches_h, num_patches_w]
        x = self.proj(x)

        # Flatten: [B, embed_dim, num_patches_h, num_patches_w] -> [B, embed_dim, num_patches]
        x = x.flatten(2)

        # Transpose: [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        x = x.transpose(1, 2)

        return x


class Attention(nn.Module):
    """Multi-head Self Attention"""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP with GELU activation"""

    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer Block with Pre-Norm"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop: float = 0., attn_drop: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class RectangularPatchViT(nn.Module):
    """
    条形 Patch ViT 视觉编码器

    支持:
    - 非正方形 patch (条形)
    - 与 CLIP 文本编码器兼容的输出维度
    - 可选的 [CLS] token 或全局平均池化
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: Tuple[int, int] = (16, 16),  # (height, width) - 条形 patch
        in_chans: int = 3,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        use_cls_token: bool = True,
        output_dim: int = 512,  # CLIP 兼容的输出维度
    ):
        """
        Args:
            img_size: 输入图像尺寸
            patch_size: patch 尺寸 (height, width)
            in_chans: 输入通道数
            embed_dim: 嵌入维度
            depth: Transformer 层数
            num_heads: 注意力头数
            mlp_ratio: MLP 隐藏层倍数
            use_cls_token: 是否使用 [CLS] token
            output_dim: 输出特征维度
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.use_cls_token = use_cls_token

        # Patch Embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 打印 patch 信息
        patch_h, patch_w = patch_size
        print(f"\nRectangularPatchViT Configuration:")
        print(f"  Image size: {img_size}x{img_size}")
        print(f"  Patch size: {patch_h}x{patch_w} ({'horizontal strip' if patch_w > patch_h else 'vertical strip' if patch_h > patch_w else 'square'})")
        print(f"  Num patches: {self.patch_embed.num_patches_h}x{self.patch_embed.num_patches_w} = {num_patches}")
        print(f"  Embed dim: {embed_dim}")
        print(f"  Depth: {depth}")
        print(f"  Num heads: {num_heads}")

        # CLS Token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.cls_token = None
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.pos_drop = nn.Dropout(drop_rate)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Output projection (to CLIP compatible dimension)
        self.proj = nn.Linear(embed_dim, output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] 输入图像

        Returns:
            [B, output_dim] 图像特征
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # Add CLS token
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.norm(x)

        # Get output features
        if self.cls_token is not None:
            # Use CLS token
            x = x[:, 0]
        else:
            # Global average pooling
            x = x.mean(dim=1)

        # Project to output dimension
        x = self.proj(x)

        return x


class RectangularPatchViTForCLIP(nn.Module):
    """
    条形 Patch ViT + CLIP 文本编码器

    完整的 CLIP 风格模型，用于 CZSL 任务
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: Tuple[int, int] = (16, 16),
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        device: str = "cuda",
    ):
        """
        Args:
            img_size: 输入图像尺寸
            patch_size: patch 尺寸 (height, width)
            embed_dim: 嵌入维度 (需要与 CLIP 文本编码器匹配)
            depth: Transformer 层数
            num_heads: 注意力头数
        """
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim

        # 加载 CLIP 文本编码器
        import clip
        base_model, self.preprocess = clip.load("ViT-B/32", device=device)

        # 转换为 FP32 (重要: CLIP 默认使用 FP16)
        base_model = base_model.float()

        # 复用 CLIP 文本编码器组件
        self.transformer = base_model.transformer
        self.token_embedding = base_model.token_embedding
        self.positional_embedding = base_model.positional_embedding
        self.ln_final = base_model.ln_final
        self.text_projection = base_model.text_projection
        self.context_length = base_model.context_length

        # 冻结文本编码器
        for param in self.transformer.parameters():
            param.requires_grad = False
        self.token_embedding.weight.requires_grad = False
        self.positional_embedding.requires_grad = False
        self.text_projection.requires_grad = False

        # 自定义视觉编码器 (条形 patch)
        self.visual = RectangularPatchViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            output_dim=embed_dim,
        ).to(device)

        # 可学习的温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """编码图像"""
        return self.visual(image)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """编码文本 (与 CLIP 相同)"""
        x = self.token_embedding(text).type(self.visual.proj.weight.dtype)
        x = x + self.positional_embedding.type(self.visual.proj.weight.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.visual.proj.weight.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: [B, 3, H, W] 图像
            text: [B, seq_len] 文本 tokens

        Returns:
            image_features, text_features
        """
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # 归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        return image_features, text_features


# ============================================================================
# 预定义配置
# ============================================================================

def create_horizontal_strip_vit(
    img_size: int = 224,
    strip_height: int = 8,
    strip_width: int = 32,
    embed_dim: int = 512,
    depth: int = 6,
    device: str = "cuda",
) -> RectangularPatchViTForCLIP:
    """
    创建横向条形 patch ViT

    特点: 覆盖较长时间段、较窄频段
    适合: 捕捉频率局部变化
    """
    return RectangularPatchViTForCLIP(
        img_size=img_size,
        patch_size=(strip_height, strip_width),
        embed_dim=embed_dim,
        depth=depth,
        device=device,
    )


def create_vertical_strip_vit(
    img_size: int = 224,
    strip_height: int = 32,
    strip_width: int = 8,
    embed_dim: int = 512,
    depth: int = 6,
    device: str = "cuda",
) -> RectangularPatchViTForCLIP:
    """
    创建纵向条形 patch ViT

    特点: 覆盖较宽频段、较短时间
    适合: 捕捉时间瞬态特征
    """
    return RectangularPatchViTForCLIP(
        img_size=img_size,
        patch_size=(strip_height, strip_width),
        embed_dim=embed_dim,
        depth=depth,
        device=device,
    )


def create_square_patch_vit(
    img_size: int = 224,
    patch_size: int = 16,
    embed_dim: int = 512,
    depth: int = 6,
    device: str = "cuda",
) -> RectangularPatchViTForCLIP:
    """
    创建正方形 patch ViT (对照实验基线)
    """
    return RectangularPatchViTForCLIP(
        img_size=img_size,
        patch_size=(patch_size, patch_size),
        embed_dim=embed_dim,
        depth=depth,
        device=device,
    )


# ============================================================================
# 与现有 CLIPForCZSL 兼容的包装类
# ============================================================================

class _BaseCLIPWrapper(nn.Module):
    """
    CLIP CZSL 包装器基类

    统一管理 CLIP ViT-B/32 文本编码器的加载、冻结和推理，
    消除子类中的重复代码。子类只需实例化 self.visual。
    """

    def __init__(self, device: str = "cuda", embed_dim: int = 512):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim

        # 加载 CLIP 文本编码器
        import clip
        base_model, self.preprocess = clip.load("ViT-B/32", device=device)
        base_model = base_model.float()

        self.transformer = base_model.transformer
        self.token_embedding = base_model.token_embedding
        self.positional_embedding = base_model.positional_embedding
        self.ln_final = base_model.ln_final
        self.text_projection = base_model.text_projection
        self.context_length = base_model.context_length

        # 冻结文本编码器
        for param in self.transformer.parameters():
            param.requires_grad = False
        self.token_embedding.weight.requires_grad = False
        self.positional_embedding.requires_grad = False
        self.text_projection.requires_grad = False

        # 继承预训练 CLIP 的温度参数 (不要重新初始化！
        # 重置 logit_scale 会改变绝对相似度尺度，破坏 zero-shot softmax 分布)
        import copy
        self.logit_scale = nn.Parameter(copy.deepcopy(base_model.logit_scale.data))

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """编码图像 — 委托给 self.visual"""
        return self.visual(image)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """编码文本 (与 CLIP 相同)"""
        dtype = self._get_visual_dtype()
        x = self.token_embedding(text).type(dtype)
        x = x + self.positional_embedding.type(dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def _get_visual_dtype(self):
        """获取视觉组件的 dtype 以保持一致性"""
        return next(self.visual.parameters()).dtype


class RectangularPatchViTForCZSL(_BaseCLIPWrapper):
    """
    条形 Patch ViT 用于 CZSL 任务

    与 CLIPForCZSL 接口兼容，可以直接替换使用
    """

    def __init__(
        self,
        patch_size: Tuple[int, int] = (16, 16),
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        device: str = "cuda",
        num_classes: int = 14,
        class_names: list = None,
    ):
        """
        Args:
            patch_size: patch 尺寸 (height, width)
            embed_dim: 嵌入维度
            depth: Transformer 层数
            num_heads: 注意力头数
            device: 计算设备
            num_classes: 类别数
            class_names: 类别名称列表
        """
        super().__init__(device=device, embed_dim=embed_dim)
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]

        # 直接创建视觉编码器（不再通过 RectangularPatchViTForCLIP 中转，避免双重文本编码器）
        self.visual = RectangularPatchViT(
            img_size=224,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            output_dim=embed_dim,
        ).to(device)

        # 文本特征缓存
        self._text_features_cache = None
        self._combination_features_cache = None
        self._combination_names = None

    def forward_contrastive(
        self,
        image: torch.Tensor,
        text_tokens: torch.Tensor,
        time_signal: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对比学习前向传播 (与 CLIPForCZSL 接口兼容)
        """
        image_features = self.encode_image(image)
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
        """前向传播"""
        return self.forward_contrastive(image, text_tokens, time_signal)

    @torch.no_grad()
    def cache_text_features(
        self,
        max_combination_size: int = 2,
        include_single: bool = True,
        seen_combinations: list = None,
        use_translation: bool = False
    ):
        """
        缓存所有类别和组合的文本特征
        """
        self.eval()
        import clip
        from itertools import combinations

        all_features = []
        all_names = []

        if include_single:
            from multi.text_templates import get_inference_description
            for cls_name in self.class_names:
                desc = get_inference_description([cls_name], use_translation=use_translation)
                tokens = clip.tokenize(desc, truncate=True).to(self.device)
                features = self.encode_text(tokens)
                features = F.normalize(features, dim=-1)
                all_features.append(features)
                all_names.append(cls_name)

        if max_combination_size >= 2:
            from multi.text_templates import get_inference_description
            if seen_combinations:
                combo_only = [c for c in seen_combinations if len(c) > 1]
                for combo in combo_only:
                    combined_desc = get_inference_description(list(combo), use_translation=use_translation)
                    tokens = clip.tokenize(combined_desc, truncate=True).to(self.device)
                    features = self.encode_text(tokens)
                    features = F.normalize(features, dim=-1)
                    all_features.append(features)
                    all_names.append('+'.join(combo))
            else:
                for i, j in combinations(range(len(self.class_names)), 2):
                    cls1, cls2 = self.class_names[i], self.class_names[j]
                    combined_desc = get_inference_description([cls1, cls2], use_translation=use_translation)
                    tokens = clip.tokenize(combined_desc, truncate=True).to(self.device)
                    features = self.encode_text(tokens)
                    features = F.normalize(features, dim=-1)
                    all_features.append(features)
                    all_names.append(f"{cls1}+{cls2}")

        self._combination_features_cache = torch.cat(all_features, dim=0)
        self._combination_names = all_names
        self._text_features_cache = self._combination_features_cache[:len(self.class_names)]

        print(f"Cached {len(all_names)} text features")

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
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        零样本预测
        """
        self.eval()

        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)

        if use_combinations:
            text_features = self.get_cached_combination_features()
            names = self._combination_names
        else:
            text_features = self.get_cached_text_features()
            names = self.class_names

        logit_scale = self.logit_scale.exp()
        similarities = logit_scale * (image_features @ text_features.T)

        top_k = min(top_k, text_features.shape[0])
        values, indices = torch.topk(similarities, k=top_k, dim=-1)

        pred_names = [[names[idx.item()] for idx in batch_indices] for batch_indices in indices]

        return similarities, indices, pred_names


def create_rectangular_patch_model(
    config: dict,
    device: str = "cuda"
) -> RectangularPatchViTForCZSL:
    """
    从配置创建条形 Patch ViT 模型

    配置示例:
    ```yaml
    model:
      patch_size: [8, 32]  # [height, width]
      embed_dim: 512
      depth: 6
      num_heads: 8
    ```
    """
    model_config = config.get("model", {})
    class_names = [cls_info["name"] for cls_info in config.get("jamming_classes", [])]

    patch_size = model_config.get("patch_size", (16, 16))
    if isinstance(patch_size, list):
        patch_size = tuple(patch_size)

    embed_dim = model_config.get("embed_dim", 512)
    depth = model_config.get("depth", 6)
    num_heads = model_config.get("num_heads", 8)

    model = RectangularPatchViTForCZSL(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        device=device,
        num_classes=len(class_names),
        class_names=class_names,
    )

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nRectangularPatchViTForCZSL:")
    print(f"  Patch size: {patch_size[0]}x{patch_size[1]}")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    return model


# ============================================================================
# 多形状 Patch ViT - 综合三种patch形状
# ============================================================================

class MultiShapePatchViT(nn.Module):
    """
    多形状 Patch ViT

    综合三种patch形状的特征：
    1. 每种形状独立提取patch特征
    2. 通过Transformer进一步处理
    3. 融合输出

    支持两种架构模式：
    - "early_fusion": 早期融合，patch层面直接拼接后统一处理
    - "late_fusion": 晚期融合，各分支独立处理后融合
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_sizes: list = None,
        in_chans: int = 3,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        fusion_mode: str = "late_fusion",  # "early_fusion" or "late_fusion"
        output_dim: int = 512,
    ):
        """
        Args:
            img_size: 输入图像尺寸
            patch_sizes: patch尺寸列表 [(h,w), ...]
            embed_dim: 嵌入维度
            depth: Transformer层数
            num_heads: 注意力头数
            fusion_mode: 融合模式
        """
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.fusion_mode = fusion_mode

        if patch_sizes is None:
            patch_sizes = [(8, 32), (32, 8), (16, 16)]

        self.patch_sizes = patch_sizes
        self.num_branches = len(patch_sizes)

        if fusion_mode == "early_fusion":
            # 早期融合：各分支patch拼接后统一处理
            self.patch_embeds = nn.ModuleList([
                PatchEmbed(img_size, ps, in_chans, embed_dim)
                for ps in patch_sizes
            ])

            total_patches = sum(pe.num_patches for pe in self.patch_embeds)

            # 位置编码（用于所有patch）
            self.pos_embed = nn.Parameter(torch.zeros(1, total_patches, embed_dim))
            self.pos_drop = nn.Dropout(drop_rate)

            # 形状/类型编码 (类似BERT的segment embeddings)
            # 让Transformer能够区分token来自哪个patch分支
            self.shape_embeds = nn.ParameterList([
                nn.Parameter(torch.zeros(1, 1, embed_dim))
                for _ in patch_sizes
            ])

            # 共享的 Transformer blocks
            self.blocks = nn.ModuleList([
                Block(embed_dim, num_heads, mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate)
                for _ in range(depth)
            ])
            self.norm = nn.LayerNorm(embed_dim)

        elif fusion_mode == "late_fusion":
            # 晚期融合：各分支独立处理后融合
            self.patch_embeds = nn.ModuleList([
                PatchEmbed(img_size, ps, in_chans, embed_dim)
                for ps in patch_sizes
            ])

            # 每个分支独立的位置编码 (修复致命Bug: 之前缺少位置编码)
            self.branch_pos_embeds = nn.ParameterList([
                nn.Parameter(torch.zeros(1, pe.num_patches, embed_dim))
                for pe in self.patch_embeds
            ])
            self.branch_pos_drops = nn.ModuleList([
                nn.Dropout(drop_rate) for _ in range(self.num_branches)
            ])

            # 每个分支独立的 Transformer
            self.branch_transformers = nn.ModuleList([
                nn.ModuleList([
                    Block(embed_dim, num_heads, mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate)
                    for _ in range(depth)
                ])
                for _ in range(self.num_branches)
            ])

            # 每个分支的 LayerNorm
            self.branch_norms = nn.ModuleList([
                nn.LayerNorm(embed_dim)
                for _ in range(self.num_branches)
            ])

            # 融合层
            self.fusion_layer = nn.Linear(embed_dim * self.num_branches, embed_dim)

        # 输出投影
        self.proj = nn.Linear(embed_dim, output_dim)

        # 初始化
        self._init_weights()

        print(f"\nMultiShapePatchViT Configuration:")
        print(f"  Image size: {img_size}x{img_size}")
        print(f"  Patch sizes: {patch_sizes}")
        print(f"  Fusion mode: {fusion_mode}")
        print(f"  Embed dim: {embed_dim}")
        print(f"  Depth: {depth}")
        print(f"  Num heads: {num_heads}")

    def _init_weights(self):
        if self.fusion_mode == "early_fusion":
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            for se in self.shape_embeds:
                nn.init.trunc_normal_(se, std=0.02)
        elif self.fusion_mode == "late_fusion":
            for pe in self.branch_pos_embeds:
                nn.init.trunc_normal_(pe, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] 输入图像

        Returns:
            [B, output_dim] 图像特征
        """
        B = x.shape[0]

        if self.fusion_mode == "early_fusion":
            # 早期融合
            all_patches = []
            for i, patch_embed in enumerate(self.patch_embeds):
                patches = patch_embed(x)  # [B, num_patches, embed_dim]
                patches = patches + self.shape_embeds[i]  # 添加形状标识
                all_patches.append(patches)

            # 拼接所有patch
            x = torch.cat(all_patches, dim=1)  # [B, total_patches, embed_dim]

            # 添加位置编码
            x = x + self.pos_embed
            x = self.pos_drop(x)

            # Transformer blocks
            for block in self.blocks:
                x = block(x)

            x = self.norm(x)

            # 全局平均池化
            x = x.mean(dim=1)  # [B, embed_dim]

        else:  # late_fusion
            # 晚期融合：各分支独立处理
            branch_features = []

            for i, patch_embed in enumerate(self.patch_embeds):
                # 提取 patch
                patches = patch_embed(x)  # [B, num_patches_i, embed_dim]

                # 添加位置编码 (修复致命Bug: 之前缺失)
                patches = patches + self.branch_pos_embeds[i]
                patches = self.branch_pos_drops[i](patches)

                # Transformer 处理
                for block in self.branch_transformers[i]:
                    patches = block(patches)

                # LayerNorm
                patches = self.branch_norms[i](patches)

                # 全局平均池化
                feature = patches.mean(dim=1)  # [B, embed_dim]
                branch_features.append(feature)

            # 拼接各分支特征
            concat_features = torch.cat(branch_features, dim=-1)  # [B, embed_dim * num_branches]

            # 融合
            x = self.fusion_layer(concat_features)  # [B, embed_dim]

        # 输出投影
        x = self.proj(x)

        return x


class MultiShapePatchViTForCZSL(_BaseCLIPWrapper):
    """
    多形状 Patch ViT 用于 CZSL 任务

    与 CLIPForCZSL 接口兼容
    """

    def __init__(
        self,
        patch_sizes: list = None,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        fusion_mode: str = "late_fusion",
        device: str = "cuda",
        num_classes: int = 14,
        class_names: list = None,
    ):
        super().__init__(device=device, embed_dim=embed_dim)
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]

        if patch_sizes is None:
            patch_sizes = [(8, 32), (32, 8), (16, 16)]

        # 多形状视觉编码器
        self.visual = MultiShapePatchViT(
            img_size=224,
            patch_sizes=patch_sizes,
            in_chans=3,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            fusion_mode=fusion_mode,
            output_dim=embed_dim,
        ).to(device)

        # 缓存
        self._text_features_cache = None
        self._combination_features_cache = None
        self._combination_names = None

    def forward_contrastive(
        self,
        image: torch.Tensor,
        text_tokens: torch.Tensor,
        time_signal: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_features = self.encode_image(image)
        text_features = self.encode_text(text_tokens)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        return image_features, text_features

    def forward(self, image, text_tokens=None, time_signal=None):
        return self.forward_contrastive(image, text_tokens, time_signal)

    @torch.no_grad()
    def cache_text_features(self, max_combination_size=2, include_single=True,
                           seen_combinations=None, use_translation=False):
        self.eval()
        import clip
        from itertools import combinations

        all_features = []
        all_names = []

        if include_single:
            from multi.text_templates import get_inference_description
            for cls_name in self.class_names:
                desc = get_inference_description([cls_name], use_translation=use_translation)
                tokens = clip.tokenize(desc, truncate=True).to(self.device)
                features = self.encode_text(tokens)
                features = F.normalize(features, dim=-1)
                all_features.append(features)
                all_names.append(cls_name)

        if max_combination_size >= 2:
            from multi.text_templates import get_inference_description
            if seen_combinations:
                combo_only = [c for c in seen_combinations if len(c) > 1]
                for combo in combo_only:
                    combined_desc = get_inference_description(list(combo), use_translation=use_translation)
                    tokens = clip.tokenize(combined_desc, truncate=True).to(self.device)
                    features = self.encode_text(tokens)
                    features = F.normalize(features, dim=-1)
                    all_features.append(features)
                    all_names.append('+'.join(combo))
            else:
                for i, j in combinations(range(len(self.class_names)), 2):
                    cls1, cls2 = self.class_names[i], self.class_names[j]
                    combined_desc = get_inference_description([cls1, cls2], use_translation=use_translation)
                    tokens = clip.tokenize(combined_desc, truncate=True).to(self.device)
                    features = self.encode_text(tokens)
                    features = F.normalize(features, dim=-1)
                    all_features.append(features)
                    all_names.append(f"{cls1}+{cls2}")

        self._combination_features_cache = torch.cat(all_features, dim=0)
        self._combination_names = all_names
        self._text_features_cache = self._combination_features_cache[:len(self.class_names)]

        print(f"Cached {len(all_names)} text features")

    def get_cached_text_features(self):
        if self._text_features_cache is None:
            self.cache_text_features()
        return self._text_features_cache

    def get_cached_combination_features(self):
        if self._combination_features_cache is None:
            self.cache_text_features()
        return self._combination_features_cache

    @torch.no_grad()
    def zero_shot_predict(self, image, time_signal=None, use_combinations=True, top_k=1):
        self.eval()

        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)

        if use_combinations:
            text_features = self.get_cached_combination_features()
            names = self._combination_names
        else:
            text_features = self.get_cached_text_features()
            names = self.class_names

        logit_scale = self.logit_scale.exp()
        similarities = logit_scale * (image_features @ text_features.T)

        top_k = min(top_k, text_features.shape[0])
        values, indices = torch.topk(similarities, k=top_k, dim=-1)

        pred_names = [[names[idx.item()] for idx in batch_indices] for batch_indices in indices]

        return similarities, indices, pred_names


def create_multi_shape_patch_model(config: dict, device: str = "cuda") -> MultiShapePatchViTForCZSL:
    """
    从配置创建多形状 Patch ViT 模型

    配置示例:
    ```yaml
    model:
      patch_sizes: [[8, 32], [32, 8], [16, 16]]  # horizontal, vertical, square
      embed_dim: 512
      depth: 6
      num_heads: 8
      fusion_mode: "late_fusion"  # or "early_fusion"
    ```
    """
    model_config = config.get("model", {})
    class_names = [cls_info["name"] for cls_info in config.get("jamming_classes", [])]

    patch_sizes = model_config.get("patch_sizes", [(8, 32), (32, 8), (16, 16)])
    if isinstance(patch_sizes[0], list):
        patch_sizes = [tuple(ps) for ps in patch_sizes]

    embed_dim = model_config.get("embed_dim", 512)
    depth = model_config.get("depth", 6)
    num_heads = model_config.get("num_heads", 8)
    fusion_mode = model_config.get("fusion_mode", "late_fusion")

    model = MultiShapePatchViTForCZSL(
        patch_sizes=patch_sizes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        fusion_mode=fusion_mode,
        device=device,
        num_classes=len(class_names),
        class_names=class_names,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nMultiShapePatchViTForCZSL:")
    print(f"  Patch sizes: {patch_sizes}")
    print(f"  Fusion mode: {fusion_mode}")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    return model


# ============================================================================
# 双分支多形状 Patch ViT - 用于欺骗/压制干扰分类
# ============================================================================

class MultiShapePatchViTForDualBranch(_BaseCLIPWrapper):
    """
    多形状 Patch ViT 双分支版本

    用于欺骗/压制干扰分类，与 DualBranchCLIPForCZSL 接口兼容
    使用双投影头解耦两个语义维度的特征子空间
    """

    def __init__(
        self,
        patch_sizes: list = None,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        fusion_mode: str = "early_fusion",
        device: str = "cuda",
        deception_classes: list = None,
        suppression_classes: list = None,
    ):
        super().__init__(device=device, embed_dim=embed_dim)

        # 干扰类型分组
        self.deception_classes = deception_classes or ["DFTJ", "ISRJ", "SMSPJ", "C&IJ", "CSJ"]
        self.suppression_classes = suppression_classes or ["AJ", "BJ", "SJ", "NCJ", "NPJ", "NFMJ", "NPMJ", "NAMJ", "PJ"]

        # 添加 "无XX干扰" 类
        self.deception_classes_with_none = self.deception_classes + ["无欺骗干扰"]
        self.suppression_classes_with_none = self.suppression_classes + ["无压制干扰"]

        self.num_deception_classes = len(self.deception_classes_with_none)
        self.num_suppression_classes = len(self.suppression_classes_with_none)

        if patch_sizes is None:
            patch_sizes = [(8, 32), (32, 8), (16, 16)]

        # 多形状视觉编码器 (共享主干)
        # output_dim=embed_dim: 内部proj退化为恒等映射，由外部task heads做真正的投影
        self.visual = MultiShapePatchViT(
            img_size=224,
            patch_sizes=patch_sizes,
            in_chans=3,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            fusion_mode=fusion_mode,
            output_dim=embed_dim,  # 恒等 — 避免双重线性投影
        ).to(device)

        # 任务特定的投影头 (解决共享投影头无法解耦两个语义维度的问题)
        self.proj_deception = nn.Linear(embed_dim, embed_dim).to(device)
        self.proj_suppression = nn.Linear(embed_dim, embed_dim).to(device)

        # 缓存
        self._deception_text_features = None
        self._suppression_text_features = None
        self._deception_names = None
        self._suppression_names = None

        print(f"\nMultiShapePatchViTForDualBranch:")
        print(f"  Patch sizes: {patch_sizes}")
        print(f"  Fusion mode: {fusion_mode}")
        print(f"  Deception classes: {self.num_deception_classes}")
        print(f"  Suppression classes: {self.num_suppression_classes}")

    def forward_dual(
        self,
        image: torch.Tensor,
        text_tokens_deception: torch.Tensor,
        text_tokens_suppression: torch.Tensor,
        time_signal: torch.Tensor = None,
        features_dict: dict = None,
    ) -> dict:
        """双分支前向传播

        返回字典格式以保持向后兼容：其他模型（如 DualBranchCLIPForCZSL）
        仍返回 3-tuple，调用方通过 isinstance 判断或直接解析 dict。
        """
        shared = self.encode_image(image)  # 共享主干输出

        # 任务特定投影
        img_feat_d = self.proj_deception(shared)
        img_feat_s = self.proj_suppression(shared)

        img_feat_d = F.normalize(img_feat_d, dim=-1)
        img_feat_s = F.normalize(img_feat_s, dim=-1)

        txt_feat_d = F.normalize(self.encode_text(text_tokens_deception), dim=-1)
        txt_feat_s = F.normalize(self.encode_text(text_tokens_suppression), dim=-1)

        return {
            "deception": (img_feat_d, txt_feat_d),
            "suppression": (img_feat_s, txt_feat_s),
        }

    def forward(self, image, text_tokens_deception=None, text_tokens_suppression=None,
                time_signal=None, features_dict=None):
        return self.forward_dual(image, text_tokens_deception, text_tokens_suppression,
                                 time_signal, features_dict)

    @torch.no_grad()
    def cache_text_features_dual(self, use_translation: bool = False):
        """缓存双分支文本特征"""
        self.eval()
        import clip
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

    def get_deception_text_features(self):
        if self._deception_text_features is None:
            self.cache_text_features_dual()
        return self._deception_text_features

    def get_suppression_text_features(self):
        if self._suppression_text_features is None:
            self.cache_text_features_dual()
        return self._suppression_text_features

    @torch.no_grad()
    def zero_shot_predict_dual(self, image, time_signal=None, top_k=1):
        """双分支零样本预测 — 使用任务特定投影头"""
        self.eval()

        shared = self.encode_image(image)

        # 使用任务特定投影头
        img_feat_d = F.normalize(self.proj_deception(shared), dim=-1)
        img_feat_s = F.normalize(self.proj_suppression(shared), dim=-1)

        logit_scale = self.logit_scale.exp()

        # 欺骗分支预测
        deception_features = self.get_deception_text_features()
        deception_similarities = logit_scale * (img_feat_d @ deception_features.T)
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
        suppression_similarities = logit_scale * (img_feat_s @ suppression_features.T)
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


def create_multi_shape_dual_branch_model(config: dict, device: str = "cuda") -> MultiShapePatchViTForDualBranch:
    """
    从配置创建多形状双分支模型

    配置示例:
    ```yaml
    model:
      patch_sizes: [[8, 32], [32, 8], [16, 16]]
      embed_dim: 512
      depth: 6
      num_heads: 8
      fusion_mode: "early_fusion"
    ```
    """
    model_config = config.get("model", {})

    # 从配置获取干扰类型分组
    jamming_groups = config.get("jamming_groups", {})
    deception_classes = jamming_groups.get("deception", {}).get("classes", ["DFTJ", "ISRJ", "SMSPJ", "C&IJ", "CSJ"])
    suppression_classes = jamming_groups.get("suppression", {}).get("classes", ["AJ", "BJ", "SJ", "NCJ", "NPJ", "NFMJ", "NPMJ", "NAMJ", "PJ"])

    patch_sizes = model_config.get("patch_sizes", [(8, 32), (32, 8), (16, 16)])
    if isinstance(patch_sizes[0], list):
        patch_sizes = [tuple(ps) for ps in patch_sizes]

    embed_dim = model_config.get("embed_dim", 512)
    depth = model_config.get("depth", 6)
    num_heads = model_config.get("num_heads", 8)
    fusion_mode = model_config.get("fusion_mode", "early_fusion")

    model = MultiShapePatchViTForDualBranch(
        patch_sizes=patch_sizes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        fusion_mode=fusion_mode,
        device=device,
        deception_classes=deception_classes,
        suppression_classes=suppression_classes,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    return model


if __name__ == "__main__":
    import sys
    import os
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    import clip

    # ==================== 测试单形状 ViT ====================
    print("\n" + "="*70)
    print("Testing Single-Shape Patch ViT")
    print("="*70)

    configs = [
        ("Horizontal (8x32)", (8, 32)),
        ("Vertical (32x8)", (32, 8)),
        ("Square (16x16)", (16, 16)),
    ]

    for name, patch_size in configs:
        print(f"\n{'-'*60}")
        print(f"Testing: {name}")
        print('-'*60)

        model = RectangularPatchViTForCLIP(
            img_size=224,
            patch_size=patch_size,
            embed_dim=512,
            depth=6,
            device=device,
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")

        dummy_image = torch.randn(2, 3, 224, 224).to(device)
        dummy_text = torch.randint(0, 49408, (2, 77)).to(device)

        with torch.no_grad():
            img_feat, txt_feat = model(dummy_image, dummy_text)

        print(f"Image features shape: {img_feat.shape}")
        print(f"Text features shape: {txt_feat.shape}")

    # ==================== 测试多形状 ViT ====================
    print("\n" + "="*70)
    print("Testing Multi-Shape Patch ViT")
    print("="*70)

    for fusion_mode in ["early_fusion", "late_fusion"]:
        print(f"\n{'-'*60}")
        print(f"Fusion mode: {fusion_mode}")
        print('-'*60)

        model = MultiShapePatchViTForCZSL(
            patch_sizes=[(8, 32), (32, 8), (16, 16)],
            embed_dim=512,
            depth=6,
            num_heads=8,
            fusion_mode=fusion_mode,
            device=device,
            num_classes=4,
            class_names=["DFTJ", "ISRJ", "AJ", "BJ"],
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")

        # 测试前向传播
        dummy_image = torch.randn(2, 3, 224, 224).to(device)
        dummy_text = clip.tokenize(["a radar signal with DFTJ", "a radar signal with ISRJ"], truncate=True).to(device)

        img_feat, txt_feat = model.forward_contrastive(dummy_image, dummy_text)
        print(f"Image features shape: {img_feat.shape}")
        print(f"Text features shape: {txt_feat.shape}")

    # ==================== 测试配置创建 ====================
    print("\n" + "="*70)
    print("Testing create_multi_shape_patch_model")
    print("="*70)

    test_config = {
        "model": {
            "patch_sizes": [[8, 32], [32, 8], [16, 16]],
            "embed_dim": 512,
            "depth": 6,
            "num_heads": 8,
            "fusion_mode": "late_fusion",
        },
        "jamming_classes": [
            {"name": "DFTJ"},
            {"name": "ISRJ"},
            {"name": "AJ"},
            {"name": "BJ"},
        ]
    }

    model = create_multi_shape_patch_model(test_config, device)

    # 缓存文本特征
    model.cache_text_features(max_combination_size=2, include_single=True)

    # 测试零样本预测
    dummy_image = torch.randn(2, 3, 224, 224).to(device)
    sims, indices, names = model.zero_shot_predict(dummy_image, use_combinations=True, top_k=3)
    print(f"\nZero-shot prediction:")
    print(f"  Similarities shape: {sims.shape}")
    print(f"  Top predictions: {names}")

    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)
