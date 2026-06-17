"""
PersistenceCLIPForCZSL: CLIP ViT + Text Encoder for persistence spectrum CZSL.

Uses the standard CLIP ViT visual encoder on persistence spectrum images
(3×224×224, created by stacking the 1-channel probability distribution),
with CLIP's frozen text encoder for contrastive learning.
"""

import os
import sys
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class PersistenceCLIPForCZSL(nn.Module):
    """CLIP model for persistence spectrum CZSL.

    Keeps both CLIP's ViT visual encoder and text encoder. The visual encoder
    processes 3-channel persistence spectrum images; the text encoder is frozen
    and generates text features for contrastive learning.

    Forward returns (image_features, text_features) — both L2-normalized —
    for use with InfoNCE / LabelAwareInfoNCE loss.
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        num_classes: int = 14,
        class_names: List[str] = None,
        freeze_vision: bool = False,
        vision_layers_unfreeze: int = 2,
        freeze_text: bool = True,
        device: str = "cuda",
        # Feature-conditioned context (CoOp-style)
        use_feature_context: bool = False,
        n_ctx_per_domain: dict = None,
    ):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.use_feature_context = use_feature_context

        # ---- Load full CLIP model (ViT visual + text) ----
        import clip
        self.clip_model, _ = clip.load(clip_model_name, device=device)
        self.clip_model = self.clip_model.float()

        # Extract reference to key components
        self.visual = self.clip_model.visual
        self.logit_scale = self.clip_model.logit_scale

        # Text encoder components for direct access
        self.token_embedding = self.clip_model.token_embedding
        self.positional_embedding = self.clip_model.positional_embedding
        self.transformer = self.clip_model.transformer
        self.ln_final = self.clip_model.ln_final
        self.text_projection = self.clip_model.text_projection
        self.transformer_width = self.transformer.width
        self.context_length = self.clip_model.context_length

        # ---- Freeze strategy ----
        if freeze_text:
            self._freeze_text_encoder()

        if freeze_vision:
            self._freeze_vision_encoder()
        elif vision_layers_unfreeze > 0:
            self._partial_unfreeze_vision(vision_layers_unfreeze)

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
    # Freeze helpers
    # ------------------------------------------------------------------

    def _freeze_text_encoder(self):
        """Freeze all text encoder parameters."""
        for param in self.transformer.parameters():
            param.requires_grad = False
        self.token_embedding.weight.requires_grad = False
        self.positional_embedding.requires_grad = False
        self.text_projection.requires_grad = False
        self.ln_final.weight.requires_grad = False
        self.ln_final.bias.requires_grad = False

    def _freeze_vision_encoder(self):
        """Freeze all vision encoder parameters."""
        for param in self.visual.parameters():
            param.requires_grad = False

    def _partial_unfreeze_vision(self, num_layers: int):
        """Unfreeze only the last N layers of the vision encoder.

        For ViT-B/32: self.visual.transformer.resblocks is a ModuleList of
        Transformer blocks. We freeze all then unfreeze the last N.
        """
        self._freeze_vision_encoder()

        if hasattr(self.visual, 'transformer') and hasattr(self.visual.transformer, 'resblocks'):
            resblocks = self.visual.transformer.resblocks
            total_blocks = len(resblocks)
            unfreeze_start = max(0, total_blocks - num_layers)

            for i in range(unfreeze_start, total_blocks):
                for param in resblocks[i].parameters():
                    param.requires_grad = True

            print(f"Vision encoder: {total_blocks} blocks total, "
                  f"unfrozen last {num_layers} (blocks {unfreeze_start}-{total_blocks - 1})")

        # Also unfreeze the final LayerNorm before projection
        if hasattr(self.visual, 'ln_post'):
            for param in self.visual.ln_post.parameters():
                param.requires_grad = True

    # ------------------------------------------------------------------
    # dtype helper
    # ------------------------------------------------------------------

    @property
    def dtype(self):
        return self.token_embedding.weight.dtype

    # Compatibility: evaluator expects model.model.logit_scale
    @property
    def model(self):
        return self

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode persistence spectrum image with ViT.

        Args:
            image: (B, 3, H, W) — 3-channel persistence spectrum

        Returns:
            (B, embed_dim) image features
        """
        return self.clip_model.encode_image(
            image.to(device=self.device, dtype=self.dtype)
        )

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
        image: torch.Tensor,
        text_tokens: torch.Tensor = None,
        features_dict: dict = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for contrastive learning.

        Args:
            image: (B, 3, H, W) persistence spectrum images
            text_tokens: (B, 77) CLIP token IDs
            features_dict: optional per-sample features for CoOp context

        Returns:
            (image_features, text_features) — both (B, embed_dim), L2-normalized
        """
        # 1. Encode image with ViT
        image_features = self.encode_image(image)

        # 2. Encode text
        if features_dict is not None and self.prompt_learner is not None:
            context_vectors = self.prompt_learner(features_dict)
            text_features = self.encode_text_with_context(text_tokens, context_vectors)
        else:
            text_features = self.encode_text(text_tokens)

        # 3. L2 normalize
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        return image_features, text_features

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
            print("Feature-conditioned context active: skipping text feature caching "
                  "(computed per-sample)")
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
                all_features.append(features)
                all_names.append(cls_name)

        # Combinations
        for size in range(2, max_combination_size + 1):
            for combo in combinations(self.class_names, size):
                if seen_combinations is not None:
                    combo_sorted = sorted(list(combo))
                    is_seen = any(
                        sorted(sc) == combo_sorted if isinstance(sc, list) else False
                        for sc in seen_combinations
                    )
                    if not is_seen:
                        continue

                combo_name = "+".join(combo)
                desc = get_inference_description(list(combo), use_translation=use_translation)
                tokens = clip.tokenize(desc, truncate=True).to(self.device)
                features = self.encode_text(tokens)
                all_features.append(features)
                all_names.append(combo_name)

        if all_features:
            self._text_features_cache = torch.cat(all_features, dim=0)
            self._combination_names = all_names
            print(f"Cached {len(all_names)} text features for zero-shot inference")
        else:
            self._text_features_cache = None
            self._combination_names = None

    # ------------------------------------------------------------------
    # Zero-shot prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def zero_shot_predict(
        self,
        image: torch.Tensor,
        threshold: float = 0.5,
        top_k: int = None,
    ) -> List[List[str]]:
        """Zero-shot prediction using cached text features.

        Args:
            image: (B, 3, H, W)
            threshold: multi-label confidence threshold
            top_k: if set, return top-k predictions regardless of threshold

        Returns:
            List of lists of predicted class/combination names
        """
        self.eval()

        if self._text_features_cache is None:
            raise RuntimeError("Text features not cached. Call cache_text_features() first.")

        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)

        text_features = F.normalize(self._text_features_cache, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * (image_features @ text_features.T)  # (B, num_texts)

        probs = F.softmax(logits, dim=-1)

        predictions = []
        for i in range(image.size(0)):
            if top_k is not None:
                top_indices = probs[i].topk(top_k).indices.cpu().tolist()
                preds = [self._combination_names[idx] for idx in top_indices]
            else:
                preds = [
                    self._combination_names[j]
                    for j in range(len(self._combination_names))
                    if probs[i, j].item() > threshold
                ]
                if not preds:
                    best = probs[i].argmax().item()
                    preds = [self._combination_names[best]]
            predictions.append(preds)

        return predictions


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_persistence_model(config: dict, device: str = "cuda") -> PersistenceCLIPForCZSL:
    """Create a PersistenceCLIPForCZSL model from a config dictionary.

    Args:
        config: full YAML config dict
        device: torch device string

    Returns:
        Configured PersistenceCLIPForCZSL instance
    """
    model_config = config.get('model', {})

    jamming_classes = config.get('jamming_classes', [])
    class_names = [jc['name'] if isinstance(jc, dict) else jc for jc in jamming_classes]

    use_feature_context = config.get('use_feature_context', False)
    n_ctx_per_domain = config.get('n_ctx_per_domain', None)

    model = PersistenceCLIPForCZSL(
        clip_model_name=model_config.get('clip_model', 'ViT-B/32'),
        num_classes=len(class_names),
        class_names=class_names,
        freeze_vision=model_config.get('freeze_vision', False),
        vision_layers_unfreeze=model_config.get('vision_layers_unfreeze', 2),
        freeze_text=model_config.get('freeze_text', True),
        device=device,
        use_feature_context=use_feature_context,
        n_ctx_per_domain=n_ctx_per_domain,
    )

    print(f"Created PersistenceCLIPForCZSL:")
    print(f"  CLIP model: {model_config.get('clip_model', 'ViT-B/32')}")
    print(f"  Classes: {len(class_names)}")
    print(f"  Device: {device}")
    print(f"  Freeze text: {model_config.get('freeze_text', True)}")
    print(f"  Freeze vision: {model_config.get('freeze_vision', False)}")
    print(f"  Vision layers unfrozen: {model_config.get('vision_layers_unfreeze', 0)}")

    return model
