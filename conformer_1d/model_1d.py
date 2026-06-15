"""
ConformerForCZSL: 1D Conformer + CLIP Text Encoder for CZSL.

Replaces the 2D ViT vision encoder with a 1D Conformer that processes
raw I/Q time-domain signals directly, while reusing CLIP's text encoder
for contrastive learning.
"""

import os
import sys
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make parent directory importable for clip and multi modules
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from conformer_1d.conformer import ConformerEncoder


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class ConformerForCZSL(nn.Module):
    """1D Conformer vision encoder + CLIP text encoder for CZSL.

    Forward returns (signal_features, text_features) — both L2-normalized —
    for contrastive learning with standard InfoNCE / LabelAwareInfoNCE loss.
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        num_classes: int = 14,
        class_names: List[str] = None,
        freeze_text: bool = True,
        device: str = "cuda",
        # Conformer params
        in_channels: int = 2,
        input_len: int = 8000,
        hidden_dim: int = 256,
        num_blocks: int = 6,
        num_heads: int = 4,
        ffn_expansion: int = 4,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
        embed_dim: int = 512,
        # Feature-conditioned context (CoOp-style)
        use_feature_context: bool = False,
        n_ctx_per_domain: dict = None,
    ):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.embed_dim = embed_dim
        self.use_feature_context = use_feature_context

        # ---- Load CLIP text encoder (discard vision encoder) ----
        import clip
        base_model, _ = clip.load(clip_model_name, device=device)
        base_model = base_model.float()

        # Extract text-only components
        self.token_embedding = base_model.token_embedding
        self.positional_embedding = base_model.positional_embedding  # (77, transformer_width)
        self.transformer = base_model.transformer                    # Transformer (text)
        self.ln_final = base_model.ln_final
        self.text_projection = base_model.text_projection            # (transformer_width, embed_dim)
        self.logit_scale = nn.Parameter(base_model.logit_scale.clone())

        self.transformer_width = self.transformer.width
        self.context_length = base_model.context_length

        # Discard the original model (ViT no longer needed)
        del base_model

        # ---- Freeze text encoder ----
        if freeze_text:
            for param in self.transformer.parameters():
                param.requires_grad = False
            self.token_embedding.weight.requires_grad = False
            self.positional_embedding.requires_grad = False
            self.text_projection.requires_grad = False
        self._freeze_text = freeze_text

        # ---- 1D Conformer encoder ----
        self.signal_encoder = ConformerEncoder(
            in_channels=in_channels,
            input_len=input_len,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            ffn_expansion=ffn_expansion,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
            embed_dim=embed_dim,
        )

        # ---- Feature-conditioned context (CoOp-style) ----
        if use_feature_context:
            from multi.prompt_learner import FeatureConditionedPromptLearner
            self.prompt_learner = FeatureConditionedPromptLearner(
                transformer_width=self.transformer_width,
                n_ctx_per_domain=n_ctx_per_domain,
            ).to(device)
        else:
            self.prompt_learner = None

        # Text feature caches
        self._text_features_cache = None
        self._combination_features_cache = None
        self._combination_names = None

        # Move to device
        self.to(device)

    # ------------------------------------------------------------------
    # dtype helper
    # ------------------------------------------------------------------

    @property
    def dtype(self):
        return self.token_embedding.weight.dtype

    # Compatibility: CZSLEvaluator expects model.model.logit_scale
    @property
    def model(self):
        return self

    # ------------------------------------------------------------------
    # Signal encoding
    # ------------------------------------------------------------------

    def encode_image(self, time_signal: torch.Tensor) -> torch.Tensor:
        """Alias for encode_signal — compatibility with CZSLEvaluator API."""
        return self.encode_signal(time_signal)

    def encode_signal(self, time_signal: torch.Tensor) -> torch.Tensor:
        """Encode I/Q time-domain signal with Conformer.

        Args:
            time_signal: (B, 2, T) — I/Q channels

        Returns:
            (B, embed_dim) — L2-normalized features
        """
        return self.signal_encoder(time_signal.to(device=self.device, dtype=self.dtype))

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Standard CLIP text encoding (no context vectors).

        Args:
            text_tokens: (B, context_length) token IDs

        Returns:
            (B, embed_dim) text features
        """
        x = self.token_embedding(text_tokens).type(self.dtype)  # (B, L, D)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD → LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND → NLD
        x = self.ln_final(x).type(self.dtype)

        # Feature at EOT position
        eot_pos = text_tokens.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eot_pos] @ self.text_projection
        return x

    def encode_text_with_context(
        self,
        text: torch.Tensor,
        context_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """Insert context vectors into the token sequence and encode.

        Sequence: [SOT_emb | context_vectors | word_embs(1:eot+1) | padding]

        Args:
            text: token IDs (B, 77)
            context_vectors: (B, M, transformer_width)

        Returns:
            (B, embed_dim) text features
        """
        B = text.shape[0]
        M = context_vectors.shape[1]
        dtype = self.dtype

        token_embs = self.token_embedding(text).type(dtype)  # (B, 77, D)
        eot_pos = text.argmax(dim=-1)  # (B,)

        sot_emb = token_embs[:, 0:1, :]                       # (B, 1, D)
        max_eot = eot_pos.max().item()
        word_embs = token_embs[:, 1:max_eot + 1, :]           # (B, max_eot, D)

        x = torch.cat([sot_emb, context_vectors.type(dtype), word_embs], dim=1)

        # Pad/truncate to context_length (77)
        seq_len = x.shape[1]
        if seq_len > self.context_length:
            x = x[:, :self.context_length, :]
        elif seq_len < self.context_length:
            pad = torch.zeros(B, self.context_length - seq_len, x.shape[-1],
                              device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

        x = x + self.positional_embedding.type(dtype)
        x = x.permute(1, 0, 2)  # LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # NLD
        x = self.ln_final(x).type(dtype)

        # Feature at (eot_pos + M), clamped
        new_eot = (eot_pos + M).clamp(max=self.context_length - 1)
        x = x[torch.arange(B), new_eot] @ self.text_projection
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        time_signal: torch.Tensor,
        text_tokens: torch.Tensor = None,
        features_dict: dict = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for contrastive learning.

        Args:
            time_signal: (B, 2, T) I/Q time-domain signal
            text_tokens: (B, 77) CLIP token IDs
            features_dict: optional per-sample features for CoOp context

        Returns:
            (signal_features, text_features) — both (B, embed_dim), L2-normalized
        """
        # 1. Encode signal with Conformer
        signal_features = self.encode_signal(time_signal)

        # 2. Encode text
        if features_dict is not None and self.prompt_learner is not None:
            context_vectors = self.prompt_learner(features_dict)
            text_features = self.encode_text_with_context(text_tokens, context_vectors)
        else:
            text_features = self.encode_text(text_tokens)

        # 3. L2 normalize
        signal_features = F.normalize(signal_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        return signal_features, text_features

    forward_contrastive = forward  # alias for compatibility

    # ------------------------------------------------------------------
    # Text feature caching (for zero-shot inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def cache_text_features(
        self,
        max_combination_size: int = 2,
        include_single: bool = True,
        seen_combinations: list = None,
        use_translation: bool = False,
    ):
        """Build and cache text features for all classes and combinations."""
        self.eval()

        if self.prompt_learner is not None:
            print("Feature-conditioned context active: skipping text feature caching (computed per-sample)")
            return

        from itertools import combinations
        from multi.text_templates import get_inference_description
        import clip

        all_features = []
        all_names = []

        # Single classes
        if include_single:
            for cls_name in self.class_names:
                desc = get_inference_description([cls_name], use_translation=use_translation)
                tokens = clip.tokenize(desc, truncate=True).to(self.device)
                features = self.encode_text(tokens)
                features = F.normalize(features, dim=-1)
                all_features.append(features)
                all_names.append(cls_name)

        # Combinations
        if max_combination_size >= 2:
            if seen_combinations:
                for combo in seen_combinations:
                    if len(combo) <= 1:
                        continue
                    desc = get_inference_description(list(combo), use_translation=use_translation)
                    tokens = clip.tokenize(desc, truncate=True).to(self.device)
                    features = self.encode_text(tokens)
                    features = F.normalize(features, dim=-1)
                    all_features.append(features)
                    all_names.append('+'.join(combo))
            else:
                for i, j in combinations(range(len(self.class_names)), 2):
                    cls1, cls2 = self.class_names[i], self.class_names[j]
                    desc = get_inference_description([cls1, cls2], use_translation=use_translation)
                    tokens = clip.tokenize(desc, truncate=True).to(self.device)
                    features = self.encode_text(tokens)
                    features = F.normalize(features, dim=-1)
                    all_features.append(features)
                    all_names.append(f"{cls1}+{cls2}")

        self._combination_features_cache = torch.cat(all_features, dim=0)
        self._combination_names = all_names
        self._text_features_cache = self._combination_features_cache[:len(self.class_names)]

        print(f"Cached {len(all_names)} text features: {len(self.class_names)} singles + {len(all_names) - len(self.class_names)} combos")

    def get_cached_text_features(self) -> torch.Tensor:
        if self._text_features_cache is None:
            self.cache_text_features()
        return self._text_features_cache

    def get_cached_combination_features(self) -> torch.Tensor:
        if self._combination_features_cache is None:
            self.cache_text_features()
        return self._combination_features_cache

    def get_cached_combination_names(self) -> List[str]:
        if self._combination_names is None:
            self.cache_text_features()
        return self._combination_names

    # ------------------------------------------------------------------
    # Zero-shot prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def zero_shot_predict(
        self,
        time_signal: torch.Tensor,
        features_dict: dict = None,
        use_combinations: bool = True,
        top_k: int = 1,
        seen_combinations: list = None,
        use_translation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Zero-shot prediction via similarity to cached text features.

        Args:
            time_signal: (B, 2, T) I/Q time-domain signal
            features_dict: optional per-sample features for CoOp context
            use_combinations: include combination features
            top_k: number of top predictions
            seen_combinations: list of seen combination lists
            use_translation: use translated names

        Returns:
            similarities: (B, num_candidates)
            indices: (B, top_k)
            names: list of candidate names
        """
        self.eval()

        # Encode signal
        signal_features = self.encode_signal(time_signal)
        signal_features = F.normalize(signal_features, dim=-1)

        # Get text features
        if features_dict is not None and self.prompt_learner is not None:
            from multi.text_templates import get_inference_description
            import clip
            import itertools

            context_vectors = self.prompt_learner(features_dict)
            all_features = []
            all_names = []

            for cls_name in self.class_names:
                desc = get_inference_description([cls_name], use_translation=use_translation)
                tokens = clip.tokenize(desc, truncate=True).to(self.device)
                ctx = context_vectors.expand(tokens.shape[0], -1, -1)
                feat = self.encode_text_with_context(tokens, ctx)
                feat = F.normalize(feat, dim=-1)
                all_features.append(feat)
                all_names.append(cls_name)

            if use_combinations and seen_combinations:
                for combo in seen_combinations:
                    if len(combo) <= 1:
                        continue
                    desc = get_inference_description(list(combo), use_translation=use_translation)
                    tokens = clip.tokenize(desc, truncate=True).to(self.device)
                    ctx = context_vectors.expand(tokens.shape[0], -1, -1)
                    feat = self.encode_text_with_context(tokens, ctx)
                    feat = F.normalize(feat, dim=-1)
                    all_features.append(feat)
                    all_names.append('+'.join(combo))

            text_features = torch.cat(all_features, dim=0)  # (K, D)
        else:
            if use_combinations:
                text_features = self.get_cached_combination_features()
                all_names = self._combination_names
            else:
                text_features = self.get_cached_text_features()
                all_names = self.class_names

        # Compute similarity
        logit_scale = self.logit_scale.exp()
        similarities = logit_scale * (signal_features @ text_features.T)  # (B, K)

        if top_k > 1:
            top_scores, top_indices = similarities.topk(min(top_k, similarities.shape[-1]), dim=-1)
        else:
            top_scores, top_indices = similarities.topk(1, dim=-1)

        return similarities, top_indices, all_names


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_conformer_model(config: dict, device: str = None) -> ConformerForCZSL:
    """Build a ConformerForCZSL model from a config dict.

    Args:
        config: full config dict (from config_1d.yaml)
        device: torch device string (default: "cuda" if available else "cpu")

    Returns:
        ConformerForCZSL model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg = config['model']
    conformer_cfg = model_cfg.get('conformer', {})
    train_cfg = config.get('train', {})

    # Class names
    jamming_classes = config.get('jamming_classes', [])
    class_names = [jc['name'] if isinstance(jc, dict) else jc for jc in jamming_classes]

    model = ConformerForCZSL(
        clip_model_name=model_cfg.get('clip_model', 'ViT-B/32'),
        num_classes=len(class_names),
        class_names=class_names,
        freeze_text=model_cfg.get('freeze_text', True),
        device=device,
        # Conformer params
        in_channels=conformer_cfg.get('in_channels', 2),
        input_len=config['data'].get('time_seq_len', 8000),
        hidden_dim=conformer_cfg.get('hidden_dim', 256),
        num_blocks=conformer_cfg.get('num_blocks', 6),
        num_heads=conformer_cfg.get('num_heads', 4),
        ffn_expansion=conformer_cfg.get('ffn_expansion', 4),
        conv_kernel_size=conformer_cfg.get('conv_kernel_size', 15),
        dropout=conformer_cfg.get('dropout', 0.1),
        embed_dim=model_cfg.get('embed_dim', 512),
        # Feature context
        use_feature_context=config.get('use_feature_context', False),
        n_ctx_per_domain=config.get('n_ctx_per_domain', None),
    )

    print(f"ConformerForCZSL created:")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Classes: {len(class_names)}")
    print(f"  Embed dim: {model.embed_dim}")

    return model


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing ConformerForCZSL...")

    jamming_classes = [
        "DFTJ", "ISRJ", "AJ", "BJ", "SJ", "NCJ", "NPJ",
        "SMSPJ", "C&IJ", "NFMJ", "NPMJ", "NAMJ", "CSJ", "PJ",
    ]

    model = ConformerForCZSL(
        clip_model_name="ViT-B/32",
        num_classes=14,
        class_names=jamming_classes,
        freeze_text=True,
        device="cpu",
    )

    # Test forward pass
    import clip
    dummy_signal = torch.randn(2, 2, 8000)
    dummy_text = clip.tokenize(["DFTJ", "AJ"], truncate=True)

    with torch.no_grad():
        sig_feat, txt_feat = model(dummy_signal, dummy_text)
        print(f"Signal features: {sig_feat.shape}, norm={sig_feat.norm(dim=-1)}")
        print(f"Text features:   {txt_feat.shape}, norm={txt_feat.norm(dim=-1)}")

    # Test caching
    with torch.no_grad():
        model.cache_text_features()
        combos = model.get_cached_combination_features()
        print(f"Combination features: {combos.shape}")

    # Test zero-shot
    with torch.no_grad():
        sims, idxs, names = model.zero_shot_predict(dummy_signal[:1])
        print(f"Similarities: {sims.shape}")
        print(f"Top-1 indices: {idxs.shape}")
        print(f"Candidate names ({len(names)}): {names[:5]}...")

    print("All tests passed!")
