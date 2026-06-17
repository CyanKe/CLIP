"""
MoE Hybrid Interference Evaluation Script.

Fuses two independently trained CZSL models at prediction level:
  - Conformer (1D time-domain) → Expert 1 — contributes DECEPTION predictions
  - Persistence (2D spectrum)    → Expert 2 — contributes SUPPRESSION predictions

Supports two fusion modes:
  - zero_shot:       top-1 prediction fusion via moe_fuse()
  - by_combination:  per-class probability splicing (deception from E1, suppression from E2)

Usage:
    python -m moe.evaluate_hybrid \
        --checkpoint_persistence checkpoints/persistence/best_model.pt \
        --checkpoint_conformer checkpoints/conformer/best_model.pt \
        --mode all --split test --output_dir results/moe_hybrid
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from functools import partial
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from sklearn.manifold import TSNE

# ── matplotlib rcParams for Chinese font support (matching evaluate_czsl.py) ──
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Ensure the project root is on sys.path
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

# ── Reuse dataset classes from persistence/ and conformer_1d/ ──────────────
from persistence.data import PersistenceDataset, TokenizerWrapper as PersistTokenizer
from conformer_1d.data_1d import TimeSignalDataset, TokenizerWrapper as ConformTokenizer
from persistence.model import create_persistence_model
from conformer_1d.model_1d import create_conformer_model

# ── Reuse text template generation ────────────────────────────────────────
from multi.text_templates import generate_text_descriptions, get_inference_description


# ============================================================================
# Constants
# ============================================================================

DECEPTION_CLASSES = ["DFTJ", "ISRJ", "ISCJ", "ISDJ", "MISRJ", "SMSPJ", "C&IJ", "CSJ"]
SUPPRESSION_CLASSES = ["AJ", "BJ", "SJ", "NCJ", "NPJ", "NFMJ", "NPMJ", "NAMJ", "PJ"]


# ============================================================================
# Helper: load config
# ============================================================================

def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ============================================================================
# Fusion helper functions (adapted from multi/evaluate_moe.py)
# ============================================================================

def parse_prediction(pred_name: str):
    """
    Parse a prediction name into (deception_class, suppression_class) pair.

    Single jamming: one component is None; combined: both present.

    Returns:
        (deception: str|None, suppression: str|None)
    """
    if pred_name is None or pred_name == "" or pred_name == "None":
        return None, None

    parts = [p.strip() for p in pred_name.split('+')]
    d, s = None, None
    for part in parts:
        if part in DECEPTION_CLASSES:
            d = part
        elif part in SUPPRESSION_CLASSES:
            s = part
    return d, s


def components_to_pred_name(d: str | None, s: str | None, class_names: list) -> str:
    """Convert (d, s) components back to a prediction name string."""
    parts = []
    if d is not None:
        parts.append(d)
    if s is not None:
        parts.append(s)
    if not parts:
        return "None"
    return "+".join(sorted(parts, key=lambda x: class_names.index(x) if x in class_names else 99))


def components_to_multihot(d: str | None, s: str | None, class_to_idx: dict) -> np.ndarray:
    """Convert (d, s) components to a multi-hot vector (num_classes-dim)."""
    vec = np.zeros(len(class_to_idx), dtype=np.float32)
    if d is not None and d in class_to_idx:
        vec[class_to_idx[d]] = 1.0
    if s is not None and s in class_to_idx:
        vec[class_to_idx[s]] = 1.0
    return vec


def moe_fuse(pred_name_conformer: str, pred_name_persistence: str):
    """
    MoE fusion core logic.

    Rule:
      - Deception component → trust Expert 1 (Conformer, 欺骗偏好)
      - Suppression component → trust Expert 2 (Persistence, 压制偏好)
      - Fallback to the other expert when a component is missing.

    Args:
        pred_name_conformer:  Expert 1 (Conformer) top-1 prediction
        pred_name_persistence: Expert 2 (Persistence) top-1 prediction

    Returns:
        (final_deception: str|None, final_suppression: str|None)
    """
    d1, s1 = parse_prediction(pred_name_conformer)
    d2, s2 = parse_prediction(pred_name_persistence)

    # Deception: prefer Conformer (E1), fallback to Persistence (E2)
    final_d = d1 if d1 is not None else d2
    # Suppression: prefer Persistence (E2), fallback to Conformer (E1)
    final_s = s2 if s2 is not None else s1

    return final_d, final_s


def classify_fusion_case(d1, s1, d2, s2):
    """
    Classify the fusion case for analysis.

    d1, s1: components from Expert 1 (Conformer)
    d2, s2: components from Expert 2 (Persistence)
    """
    has_d1, has_s1 = d1 is not None, s1 is not None
    has_d2, has_s2 = d2 is not None, s2 is not None
    is_combined_1 = has_d1 and has_s1
    is_combined_2 = has_d2 and has_s2
    is_single_1 = (has_d1 and not has_s1) or (not has_d1 and has_s1)
    is_single_2 = (has_d2 and not has_s2) or (not has_d2 and has_s2)
    is_none_1 = not has_d1 and not has_s1
    is_none_2 = not has_d2 and not has_s2

    if is_none_1 or is_none_2:
        return "至少一方无预测"

    # Case 1: Both agree
    if d1 == d2 and s1 == s2:
        if is_combined_1:
            return "双方一致->组合"
        elif has_d1 and not has_s1:
            return "双方一致->欺骗单干扰"
        elif not has_d1 and has_s1:
            return "双方一致->压制单干扰"

    # Case 2: Both single but different
    if is_single_1 and is_single_2:
        if has_d1 and has_s2:  # E1→deception, E2→suppression → natural complement
            return "交叉互补->单+单=组合"
        elif has_d1 and has_d2:
            return "双方单欺骗但不同类"
        elif has_s1 and has_s2:
            return "双方单压制但不同类"
        elif has_s1 and has_d2:
            return "交叉互补(反向)->组合"

    # Case 3: One combined, one single
    if is_combined_1 and is_single_2:
        if has_d2 and d2 in {d1}:
            return "一方组合一方单(覆盖)"
        elif has_s2 and s2 in {s1}:
            return "一方组合一方单(覆盖)"
        else:
            return "一方组合一方单(不覆盖)"
    if is_single_1 and is_combined_2:
        if has_d1 and d1 in {d2}:
            return "一方单一组合(覆盖)"
        elif has_s1 and s1 in {s2}:
            return "一方单一组合(覆盖)"
        else:
            return "一方单一组合(不覆盖)"

    # Case 4: Both combined but different
    if is_combined_1 and is_combined_2:
        if d1 == d2 and s1 == s2:
            return "双方一致->组合"
        elif d1 == d2 and s1 != s2:
            return "双方组合->欺骗同压制不同"
        elif d1 != d2 and s1 == s2:
            return "双方组合->欺骗不同压制同"
        else:
            return "双方组合->完全不同"

    return "其他"


def _get_component(idx_set: set, class_names: list, component_classes: list) -> str | None:
    """Extract a specific component from a set of class indices."""
    for cls_name in component_classes:
        if cls_name in class_names and class_names.index(cls_name) in idx_set:
            return cls_name
    return None


# ============================================================================
# Multi-label metrics computation (adapted from persistence/evaluate.py)
# ============================================================================

def compute_multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute multi-label classification metrics.

    Args:
        y_true: (N, C) ground-truth multi-hot
        y_pred: (N, C) predicted multi-hot

    Returns:
        dict with f1_micro, f1_macro, f1_samples, precision_micro, precision_macro,
        recall_micro, recall_macro, subset_accuracy, per_class_f1
    """
    n_classes = y_true.shape[1]

    metrics = {
        "f1_micro": f1_score(y_true, y_pred, average='micro', zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "f1_samples": f1_score(y_true, y_pred, average='samples', zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average='micro', zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average='micro', zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "subset_accuracy": accuracy_score(y_true, y_pred),
    }

    # Per-class F1
    per_class_f1 = np.zeros(n_classes)
    for i in range(n_classes):
        if y_true[:, i].sum() > 0 or y_pred[:, i].sum() > 0:
            per_class_f1[i] = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        else:
            per_class_f1[i] = 1.0
    metrics["per_class_f1"] = per_class_f1

    # Combination accuracy (exact match)
    metrics["combination_accuracy"] = metrics["subset_accuracy"]

    return metrics


def compute_czsl_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         seen_indices: list, unseen_indices: list) -> dict:
    """
    Compute CZSL-specific metrics: seen/unseen accuracy and harmonic mean.

    Args:
        y_true: (N, C) ground-truth multi-hot
        y_pred: (N, C) predicted multi-hot
        seen_indices: list of sample indices for seen combinations
        unseen_indices: list of sample indices for unseen combinations

    Returns:
        dict with seen_acc, unseen_acc, harmonic_mean
    """
    metrics = {}
    for name, indices in [("seen", seen_indices), ("unseen", unseen_indices)]:
        if len(indices) == 0:
            metrics[f"{name}_accuracy"] = 0.0
        else:
            correct = np.all(y_true[indices] == y_pred[indices], axis=1).sum()
            metrics[f"{name}_accuracy"] = correct / len(indices)

    s = metrics["seen_accuracy"]
    u = metrics["unseen_accuracy"]
    if s + u > 0:
        metrics["harmonic_mean"] = 2 * s * u / (s + u)
    else:
        metrics["harmonic_mean"] = 0.0

    return metrics


# ============================================================================
# Dual-Modal Data Loading
# ============================================================================

class DualModalDataset(Dataset):
    """
    Dataset that loads BOTH persistence spectrum AND time-domain signal
    for the same sample index — guaranteeing alignment by construction.

    Returns per sample:
        (persistence_tensor, time_signal, label_multihot, metadata_dict)
    """

    def __init__(self, persistence_dataset: PersistenceDataset,
                 time_dataset: TimeSignalDataset):
        if len(persistence_dataset) != len(time_dataset):
            raise ValueError(
                f"Dataset size mismatch: persistence={len(persistence_dataset)}, "
                f"time={len(time_dataset)}. Data may not be aligned!"
            )
        self.persist_ds = persistence_dataset
        self.time_ds = time_dataset

    def __len__(self):
        return len(self.persist_ds)

    def __getitem__(self, idx):
        # Load both modalities from the same sample index
        p_item = self.persist_ds[idx]
        t_item = self.time_ds[idx]

        if len(p_item) >= 3:
            p_img, p_label, p_meta = p_item[0], p_item[1], p_item[2]
        else:
            p_img, p_label = p_item[0], p_item[1]
            p_meta = {}

        if len(t_item) >= 3:
            t_sig, t_label, t_meta = t_item[0], t_item[1], t_item[2]
        else:
            t_sig, t_label = t_item[0], t_item[1]
            t_meta = {}

        return p_img, t_sig, p_label, p_meta


def dual_modal_collate_fn(batch, tokenizer_fn):
    """
    Collate function for DualModalDataset batches.

    Returns:
        (persistence_imgs, time_signals, labels, metadata_list)
        - persistence_imgs: (B, 3, 224, 224)
        - time_signals: (B, 2, 8000)
        - labels: (B, num_classes) multi-hot
        - metadata_list: list of metadata dicts
    """
    p_imgs, t_sigs, labels, metas = zip(*batch)

    p_imgs = torch.stack(p_imgs, dim=0)
    t_sigs = torch.stack(t_sigs, dim=0)
    labels = torch.stack(labels, dim=0)

    return p_imgs, t_sigs, labels, list(metas)


# ============================================================================
# DataLoader factory
# ============================================================================

def create_dual_modal_dataloaders(config: dict, split: str = "test"):
    """
    Create a DataLoader that yields both modalities aligned by sample index.

    Iterates over JNR levels, creates PersistenceDataset and TimeSignalDataset
    for the same split at each JNR level (both use the same metadata JSON),
    wraps them in DualModalDataset, concatenates across JNR levels.

    Args:
        config: full YAML config dict
        split: "train", "val", or "test"

    Returns:
        (loader, jnr_loaders_dict, class_names)
        - loader: DataLoader yielding (p_img, t_sig, labels, metas)
        - jnr_loaders_dict: {jnr: DataLoader} for per-JNR evaluation
        - class_names: list of class name strings
    """
    data_config = config.get('data', {})

    base_path = data_config.get('base_path')
    jnr_start = data_config.get('jnr_start', 0)
    jnr_end = data_config.get('jnr_end', 20)
    jnr_step = data_config.get('jnr_step', 5)
    batch_size = data_config.get('batch_size', 16)
    num_workers = data_config.get('num_workers', 0)
    pin_memory = data_config.get('pin_memory', True)
    image_size = data_config.get('image_size', 224)
    time_seq_len = data_config.get('time_seq_len', 8000)
    persistence_var_name = data_config.get('persistence_var_name', 'all_persistences')
    persistence_suffix = data_config.get('persistence_suffix', 'echo_persistences')
    time_var_name = data_config.get('time_var_name', 'all_times')

    # Class names from config
    jamming_classes = config.get('jamming_classes', [])
    class_names = [jc['name'] if isinstance(jc, dict) else jc for jc in jamming_classes]

    jnr_levels = list(range(jnr_start, jnr_end + 1, jnr_step))
    print(f"JNR levels: {jnr_levels}")
    print(f"Class names ({len(class_names)}): {class_names}")

    tokenizer = PersistTokenizer(model_type="clip")
    collate = partial(dual_modal_collate_fn, tokenizer_fn=tokenizer)

    all_datasets = []
    jnr_loaders = {}

    for jnr in jnr_levels:
        data_folder = os.path.join(base_path, f'JNR_+{jnr}')

        # Persistence files
        persistence_file = os.path.join(data_folder, f'{split}_{persistence_suffix}.mat')
        # Time-domain files
        time_file = os.path.join(data_folder, f'{split}_echo_times.mat')
        # Shared metadata
        metadata_file = os.path.join(data_folder, f'{split}_echo_metadata.json')

        if not os.path.exists(persistence_file):
            print(f"  Warning: missing persistence file for JNR_+{jnr}, skipping")
            continue
        if not os.path.exists(time_file):
            print(f"  Warning: missing time file for JNR_+{jnr}, skipping")
            continue
        if not os.path.exists(metadata_file):
            print(f"  Warning: missing metadata for JNR_+{jnr}, skipping")
            continue

        # Create both datasets with the SAME metadata file (ensures alignment)
        persist_ds = PersistenceDataset(
            persistence_file=persistence_file,
            metadata_file=metadata_file,
            persistence_var_name=persistence_var_name,
            class_names=class_names,
            image_size=image_size,
            apply_clip_norm=True,
            augmentation=None,
            features_file=None,
            feature_dims=None,
            use_feature_context=False,
        )

        time_ds = TimeSignalDataset(
            time_file=time_file,
            metadata_file=metadata_file,
            time_var_name=time_var_name,
            class_names=class_names,
            time_seq_len=time_seq_len,
            features_file=None,
            feature_dims=None,
            use_feature_context=False,
        )

        dual_ds = DualModalDataset(persist_ds, time_ds)
        all_datasets.append(dual_ds)

        # Create per-JNR loader
        jnr_loader = DataLoader(
            dual_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            collate_fn=collate,
        )
        jnr_loaders[jnr] = jnr_loader

    if not all_datasets:
        raise RuntimeError(f"No data found for split '{split}' across JNR levels {jnr_levels}")

    # Concatenate across JNR levels
    combined_dataset = ConcatDataset(all_datasets)
    print(f"Combined {split} dataset: {len(combined_dataset)} samples across {len(all_datasets)} JNR levels")

    loader = DataLoader(
        combined_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate,
    )

    return loader, jnr_loaders, class_names


# ============================================================================
# Model loading
# ============================================================================

def load_persistence_model(config: dict, checkpoint_path: str, device: str = "cuda"):
    """
    Create a PersistenceCLIPForCZSL from config and load checkpoint weights.

    Args:
        config: full YAML config dict
        checkpoint_path: path to .pt checkpoint
        device: torch device

    Returns:
        PersistenceCLIPForCZSL in eval mode
    """
    print(f"\n[Persistence Model] Loading from: {checkpoint_path}")

    # Build a config compatible with create_persistence_model
    persist_cfg = {
        'model': config.get('persistence_model', {}),
        'jamming_classes': config.get('jamming_classes', []),
        'use_feature_context': False,
        'n_ctx_per_domain': None,
    }

    model = create_persistence_model(persist_cfg, device)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Safe load with shape mismatch tolerance
    model_state = model.state_dict()
    filtered_state = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                filtered_state[k] = v
            else:
                skipped.append(f"{k}: ckpt={list(v.shape)} model={list(model_state[k].shape)}")
        else:
            skipped.append(f"{k}: not in model")

    if skipped:
        print(f"  Warning: skipped {len(skipped)} mismatched/missing keys")
        for s in skipped[:5]:
            print(f"    - {s}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")

    model.load_state_dict(filtered_state, strict=False)
    model.to(device)
    model.eval()

    # Cache text features now
    _cache_text_features_on_model(model, config, device)

    print(f"  Loaded {len(filtered_state)}/{len(state_dict)} parameter tensors")
    return model


def load_conformer_model(config: dict, checkpoint_path: str, device: str = "cuda"):
    """
    Create a ConformerForCZSL from config and load checkpoint weights.

    Args:
        config: full YAML config dict
        checkpoint_path: path to .pt checkpoint
        device: torch device

    Returns:
        ConformerForCZSL in eval mode
    """
    print(f"\n[Conformer Model] Loading from: {checkpoint_path}")

    # Build config in the format create_conformer_model expects (flat, like config_1d.yaml)
    moe_data = config.get('data', {})
    conf_cfg = {
        'model': config.get('conformer_model', {}),
        'data': {
            'time_seq_len': moe_data.get('time_seq_len', 8000),
        },
        'jamming_classes': config.get('jamming_classes', []),
        'use_feature_context': False,
        'n_ctx_per_domain': None,
        'label_translations': {},
    }

    model = create_conformer_model(conf_cfg, device)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Safe load with shape mismatch tolerance
    model_state = model.state_dict()
    filtered_state = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                filtered_state[k] = v
            else:
                skipped.append(f"{k}: ckpt={list(v.shape)} model={list(model_state[k].shape)}")
        else:
            skipped.append(f"{k}: not in model")

    if skipped:
        print(f"  Warning: skipped {len(skipped)} mismatched/missing keys")
        for s in skipped[:5]:
            print(f"    - {s}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")

    model.load_state_dict(filtered_state, strict=False)
    model.to(device)
    model.eval()

    # Cache text features now
    _cache_text_features_on_model(model, config, device)

    print(f"  Loaded {len(filtered_state)}/{len(state_dict)} parameter tensors")
    return model


def _cache_text_features_on_model(model, config: dict, device: str):
    """Cache text features on a model, handling both model types."""
    czsl_config = config.get('czsl', {})
    seen_combinations = czsl_config.get('seen_combinations', [])

    # Normalize seen_combinations to list-of-lists format
    if seen_combinations and isinstance(seen_combinations, dict):
        seen_combinations = seen_combinations.get('seen_combinations', [])

    try:
        model.cache_text_features(
            max_combination_size=2,
            include_single=True,
            seen_combinations=seen_combinations if seen_combinations else None,
            use_translation=False,
        )
        # Diagnostic: verify text feature cache
        if hasattr(model, '_text_features_cache') and model._text_features_cache is not None:
            tf = model._text_features_cache
            print(f"  Text features cached: shape={list(tf.shape)}, dtype={tf.dtype}, "
                  f"norm_range=[{tf.norm(dim=-1).min().item():.4f}, {tf.norm(dim=-1).max().item():.4f}]")
        if hasattr(model, '_combination_names') and model._combination_names:
            print(f"  Cached names ({len(model._combination_names)}): "
                  f"{model._combination_names[:5]}...")
        if hasattr(model, 'logit_scale'):
            print(f"  logit_scale: {model.logit_scale.item():.4f} (exp={model.logit_scale.exp().item():.4f})")
        print(f"  Text features cached successfully")
    except Exception as e:
        print(f"  Warning: text feature caching failed ({e})")


def _get_logit_scale(model) -> torch.Tensor:
    """Get logit_scale as a scalar tensor, handling both model types."""
    if hasattr(model, 'logit_scale'):
        return model.logit_scale.exp()
    elif hasattr(model, 'model') and hasattr(model.model, 'logit_scale'):
        return model.model.logit_scale.exp()
    else:
        raise AttributeError("Cannot find logit_scale on model")


def _get_single_text_features(model, num_classes: int) -> torch.Tensor:
    """
    Get single-class text features (normalized), handling both model types.

    PersistenceCLIPForCZSL: singles are first N entries of _text_features_cache
    ConformerForCZSL: has get_cached_text_features() method
    """
    if hasattr(model, 'get_cached_text_features'):
        feats = model.get_cached_text_features()
    elif hasattr(model, '_text_features_cache') and model._text_features_cache is not None:
        feats = model._text_features_cache[:num_classes]
    else:
        raise RuntimeError(
            "Text features not cached. Call cache_text_features() first."
        )
    return F.normalize(feats.float(), dim=-1)


def _encode_for_model(model, x: torch.Tensor) -> torch.Tensor:
    """
    Encode input with the appropriate method.
    Conformer uses encode_signal/encode_image; Persistence uses encode_image.
    Both models support encode_image() (Conformer has it as an alias).
    """
    return model.encode_image(x)


# ============================================================================
# Hybrid MoE Evaluator
# ============================================================================

class HybridMoEEvaluator:
    """
    Evaluator for hybrid MoE fusion of Persistence + Conformer models.

    Runs dual-modal inference and fuses predictions using the MoE rule:
      - Deception class probs/predictions → Conformer (Expert 1)
      - Suppression class probs/predictions → Persistence (Expert 2)
    """

    def __init__(self, persistence_model, conformer_model, config: dict, device: str = "cuda"):
        self.p_model = persistence_model
        self.c_model = conformer_model
        self.config = config
        self.device = device

        # Class names and indices
        jamming_classes = config.get('jamming_classes', [])
        self.class_names = [jc['name'] if isinstance(jc, dict) else jc for jc in jamming_classes]
        self.num_classes = len(self.class_names)
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        # Deception / suppression indices
        self.deception_indices = [self.class_to_idx[c] for c in DECEPTION_CLASSES
                                  if c in self.class_to_idx]
        self.suppression_indices = [self.class_to_idx[c] for c in SUPPRESSION_CLASSES
                                    if c in self.class_to_idx]

        # CZSL splits
        czsl_config = config.get('czsl', {})
        self.seen_combinations = czsl_config.get('seen_combinations', [])
        if isinstance(self.seen_combinations, dict):
            self.seen_combinations = self.seen_combinations.get('seen_combinations', [])
        self.unseen_combinations = czsl_config.get('unseen_combinations', [])

        # Text features should already be cached from model loading

        print(f"\n[HybridMoEEvaluator] Initialized:")
        print(f"  Classes: {self.num_classes}")
        print(f"  Deception indices ({len(self.deception_indices)}): {self.deception_indices}")
        print(f"  Suppression indices ({len(self.suppression_indices)}): {self.suppression_indices}")

    # ── Main evaluation dispatch ───────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, mode: str = "by_combination"):
        """
        Main evaluation entry point.

        Args:
            dataloader: DataLoader yielding (p_img, t_sig, labels, metas)
            mode: "zero_shot" | "by_combination"

        Returns:
            results dict with metrics, labels, predictions, and per_sample records
        """
        if mode == "zero_shot":
            return self._evaluate_zero_shot(dataloader)
        elif mode == "by_combination":
            return self._evaluate_by_combination(dataloader)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @torch.no_grad()
    def evaluate_by_jnr(self, jnr_loaders: dict, mode: str = "by_combination"):
        """Per-JNR evaluation. Returns {jnr: results_dict}."""
        jnr_results = {}
        for jnr, loader in sorted(jnr_loaders.items()):
            print(f"\n{'='*60}")
            print(f"Evaluating JNR = +{jnr}")
            print(f"{'='*60}")
            jnr_results[jnr] = self.evaluate(loader, mode=mode)
        return jnr_results

    # ── by_combination mode ──────────────────────────────────────────

    @torch.no_grad()
    def _evaluate_by_combination(self, dataloader: DataLoader) -> dict:
        """
        Combination-aware evaluation matching standalone inference.

        Each model uses its native zero_shot_predict() with threshold —
        exactly the same inference path as persistence/evaluate.py and
        conformer_1d/evaluate_conformer.py standalone evaluation.

        MoE fusion: name-level component splicing via moe_fuse():
          - Deception component from Conformer (Expert 1)
          - Suppression component from Persistence (Expert 2)
        """
        self.p_model.eval()
        self.c_model.eval()

        czsl_cfg = self.config.get('czsl', {})
        zero_shot_cfg = czsl_cfg.get('zero_shot', {})
        threshold = zero_shot_cfg.get('threshold', 0.14)

        all_labels = []
        all_preds_moe = []
        all_preds_p = []    # Persistence (Expert 2)
        all_preds_c = []    # Conformer (Expert 1)
        per_sample_records = []
        case_counter = Counter()

        total_moe_correct = 0
        total_p_correct = 0
        total_c_correct = 0
        total_samples = 0
        debug_done = False

        desc = "MoE Hybrid [by_combination]"
        for batch in tqdm(dataloader, desc=desc):
            p_imgs, t_sigs, labels, metas = batch

            p_imgs = p_imgs.to(self.device)
            t_sigs = t_sigs.to(self.device)
            labels = labels.to(self.device)
            batch_size = p_imgs.shape[0]

            if p_imgs.shape[-1] != 224:
                p_imgs = F.interpolate(p_imgs, size=(224, 224), mode='bilinear',
                                       align_corners=False)

            # ── Expert 1: Conformer — EXACT match to ConformerEvaluator._predict_batch ──
            #   Single-class text features + softmax + top-3 > 1/num_classes
            text_feat_c = self.c_model.get_cached_text_features()   # [17, D] single-class
            signal_feat = _encode_for_model(self.c_model, t_sigs)
            signal_feat = F.normalize(signal_feat.float(), dim=-1)
            logit_scale_c = _get_logit_scale(self.c_model)
            logits_c = logit_scale_c * (signal_feat @ text_feat_c.T)  # [B, 17]
            probs_c = torch.softmax(logits_c, dim=-1)
            c_threshold = 1.0 / self.num_classes  # matches standalone (not config threshold)
            preds_c_model = self._decode_multihot(probs_c, c_threshold, top_k=3)

            # ── Expert 2: Persistence — EXACT match to PersistenceEvaluator.evaluate_by_combination ──
            #   zero_shot_predict(threshold) against combination text features
            try:
                preds_p_raw = self.p_model.zero_shot_predict(p_imgs, threshold=threshold)
                pred_p_names_list = [p if isinstance(p, list) else [p] for p in preds_p_raw]
            except Exception:
                pred_p_names_list = self._fallback_zero_shot(
                    self.p_model, p_imgs, threshold, use_combinations=True)

            # Convert persistence names → multi-hot
            preds_p_model = torch.zeros(batch_size, self.num_classes)
            for i in range(batch_size):
                for name in pred_p_names_list[i]:
                    for part in name.split('+'):
                        part = part.strip()
                        if part in self.class_to_idx:
                            preds_p_model[i, self.class_to_idx[part]] = 1.0

            # ── MoE Fusion: component-level name fusion ──
            #   Deception from Conformer (E1), Suppression from Persistence (E2)
            preds_moe = torch.zeros(batch_size, self.num_classes)
            for i in range(batch_size):
                c_name = self._multihot_to_name(preds_c_model[i].cpu().numpy())
                p_name = self._multihot_to_name(preds_p_model[i].cpu().numpy())
                fd, fs = moe_fuse(c_name, p_name)
                moe_vec = components_to_multihot(fd, fs, self.class_to_idx)
                preds_moe[i] = torch.from_numpy(moe_vec)

            # ── Per-sample recording ──
            for i in range(batch_size):
                true_vec = labels[i].cpu().numpy()
                pred_vec_moe = preds_moe[i].cpu().numpy()
                pred_vec_c = preds_c_model[i].cpu().numpy()
                pred_vec_p = preds_p_model[i].cpu().numpy()

                true_set = set(np.where(true_vec == 1)[0].tolist())
                pred_set_moe = set(np.where(pred_vec_moe == 1)[0].tolist())
                pred_set_c = set(np.where(pred_vec_c == 1)[0].tolist())
                pred_set_p = set(np.where(pred_vec_p == 1)[0].tolist())

                moe_ok = (true_set == pred_set_moe)
                c_ok = (true_set == pred_set_c)
                p_ok = (true_set == pred_set_p)

                total_moe_correct += moe_ok
                total_c_correct += c_ok
                total_p_correct += p_ok

                name_c = self._multihot_to_name(pred_vec_c)
                name_p = self._multihot_to_name(pred_vec_p)
                name_moe = self._multihot_to_name(pred_vec_moe)

                d_c, s_c = parse_prediction(name_c)
                d_p, s_p = parse_prediction(name_p)
                fd, fs = parse_prediction(name_moe)
                case_label = classify_fusion_case(d_c, s_c, d_p, s_p)
                case_counter[case_label] += 1

                all_labels.append(true_vec)
                all_preds_moe.append(pred_vec_moe)
                all_preds_c.append(pred_vec_c)
                all_preds_p.append(pred_vec_p)

                true_name = components_to_pred_name(
                    _get_component(true_set, self.class_names, DECEPTION_CLASSES),
                    _get_component(true_set, self.class_names, SUPPRESSION_CLASSES),
                    self.class_names
                )

                per_sample_records.append({
                    "true": true_name,
                    "conformer": name_c,
                    "persistence": name_p,
                    "moe": name_moe,
                    "conformer_correct": c_ok,
                    "persistence_correct": p_ok,
                    "moe_correct": moe_ok,
                    "case": case_label,
                    "d_conformer": d_c, "s_conformer": s_c,
                    "d_persistence": d_p, "s_persistence": s_p,
                    "fd": fd, "fs": fs,
                })

                total_samples += 1

            if not debug_done:
                print("\n" + "=" * 80)
                print(f"[DEBUG] MoE Hybrid [by_combination] — First batch predictions")
                print("=" * 80)
                # Conformer diagnostic: top-3 probs per sample
                print(f"\n--- Conformer diagnostics ---")
                print(f"  text_feat_c shape={text_feat_c.shape}, dtype={text_feat_c.dtype}, "
                      f"norm_range=[{text_feat_c.norm(dim=-1).min().item():.4f}, "
                      f"{text_feat_c.norm(dim=-1).max().item():.4f}]")
                print(f"  signal_feat norm_range=[{signal_feat.norm(dim=-1).min().item():.4f}, "
                      f"{signal_feat.norm(dim=-1).max().item():.4f}]")
                print(f"  logit_scale: {logit_scale_c.item():.4f}")
                top3_vals, top3_idx = torch.topk(probs_c, k=3, dim=-1)
                for i in range(min(5, batch_size)):
                    pred_parts = []
                    for j, idx in enumerate(top3_idx[i]):
                        if top3_vals[i, j] > c_threshold:
                            pred_parts.append(f"{self.class_names[idx.item()]}={top3_vals[i,j]:.3f}")
                        else:
                            pred_parts.append(f"[{self.class_names[idx.item()]}={top3_vals[i,j]:.3f}]")
                    true_idx = torch.where(labels[i] == 1)[0].tolist()
                    true_names = [self.class_names[j] for j in true_idx]
                    print(f"  [{i}] True={true_names} | Top-3: {', '.join(pred_parts)}")

                # Persistence diagnostic
                print(f"\n--- Persistence diagnostics ---")
                for i in range(min(5, batch_size)):
                    tset = set(torch.where(labels[i] == 1)[0].tolist())
                    tn = components_to_pred_name(
                        _get_component(tset, self.class_names, DECEPTION_CLASSES),
                        _get_component(tset, self.class_names, SUPPRESSION_CLASSES),
                        self.class_names)
                    print(f"  [{i}] True={tn} | Pred names: {pred_p_names_list[i][:3]}")

                print(f"\n--- Per-sample comparison ---")
                for i in range(min(5, batch_size)):
                    r = per_sample_records[-batch_size + i]
                    print(f"  [{i}] True: {r['true']:<20} | C: {r['conformer']:<20} | "
                          f"P: {r['persistence']:<20} | MoE: {r['moe']:<20} | "
                          f"C:{'T' if r['conformer_correct'] else 'F'} "
                          f"P:{'T' if r['persistence_correct'] else 'F'} "
                          f"MoE:{'T' if r['moe_correct'] else 'F'} | {r['case']}")
                debug_done = True

        # ── Aggregate metrics ──
        all_labels_np = np.array(all_labels)
        all_preds_moe_np = np.array(all_preds_moe)
        all_preds_c_np = np.array(all_preds_c)
        all_preds_p_np = np.array(all_preds_p)

        moe_metrics = compute_multilabel_metrics(all_labels_np, all_preds_moe_np)
        c_metrics = compute_multilabel_metrics(all_labels_np, all_preds_c_np)
        p_metrics = compute_multilabel_metrics(all_labels_np, all_preds_p_np)

        # CZSL metrics: identify seen/unseen sample indices
        seen_indices, unseen_indices = self._split_seen_unseen(all_labels_np)
        moe_czsl = compute_czsl_metrics(all_labels_np, all_preds_moe_np, seen_indices, unseen_indices)
        c_czsl = compute_czsl_metrics(all_labels_np, all_preds_c_np, seen_indices, unseen_indices)
        p_czsl = compute_czsl_metrics(all_labels_np, all_preds_p_np, seen_indices, unseen_indices)

        moe_metrics.update({f"moe_{k}": v for k, v in moe_czsl.items()})
        c_metrics.update({f"conformer_{k}": v for k, v in c_czsl.items()})
        p_metrics.update({f"persistence_{k}": v for k, v in p_czsl.items()})

        # Case accuracy
        case_accuracy = self._compute_case_accuracy(per_sample_records)

        return {
            "mode": "by_combination",
            "moe_metrics": moe_metrics,
            "conformer_metrics": c_metrics,
            "persistence_metrics": p_metrics,
            "labels": all_labels_np,
            "predictions_moe": all_preds_moe_np,
            "predictions_conformer": all_preds_c_np,
            "predictions_persistence": all_preds_p_np,
            "per_sample": per_sample_records,
            "case_stats": {
                "counter": dict(case_counter),
                "accuracy": case_accuracy,
            },
            "total_samples": total_samples,
            "seen_indices": seen_indices,
            "unseen_indices": unseen_indices,
        }

    # ── zero_shot mode ──────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate_zero_shot(self, dataloader: DataLoader) -> dict:
        """
        Top-1 prediction fusion via zero_shot_predict() on each model.

        Conformer (E1) → top-1 prediction from combination features
        Persistence (E2) → top-1 prediction from combination features
        moe_fuse(E1_top1, E2_top1) → final prediction
        """
        self.p_model.eval()
        self.c_model.eval()

        all_labels = []
        all_preds_moe = []
        all_preds_c = []
        all_preds_p = []
        per_sample_records = []
        case_counter = Counter()

        total_moe_correct = 0
        total_c_correct = 0
        total_p_correct = 0
        total_samples = 0
        debug_done = False

        desc = "MoE Hybrid [zero_shot]"
        for batch in tqdm(dataloader, desc=desc):
            p_imgs, t_sigs, labels, metas = batch

            p_imgs = p_imgs.to(self.device)
            t_sigs = t_sigs.to(self.device)
            labels = labels.to(self.device)
            batch_size = p_imgs.shape[0]

            if p_imgs.shape[-1] != 224:
                p_imgs = F.interpolate(p_imgs, size=(224, 224), mode='bilinear',
                                       align_corners=False)

            # Expert 1: Conformer — use _manual_zero_shot directly
            # Conformer.zero_shot_predict returns (similarities, indices, all_names)
            # where all_names is the FULL candidate list, NOT per-sample predictions.
            # _manual_zero_shot correctly returns per-sample top-1 names.
            pred_c_names = self._manual_zero_shot(self.c_model, t_sigs, top_k=1)

            # Expert 2: Persistence zero_shot_predict → top-1
            try:
                # Persistence zero_shot_predict has different signature
                preds_p_raw = self.p_model.zero_shot_predict(p_imgs, top_k=1)
                pred_p_names = [p[0] if isinstance(p, list) else p for p in preds_p_raw]
            except Exception:
                pred_p_names = self._manual_zero_shot(self.p_model, p_imgs, top_k=1)

            # ── MoE Fusion ──
            for i in range(batch_size):
                d_c, s_c = parse_prediction(pred_c_names[i])
                d_p, s_p = parse_prediction(pred_p_names[i])
                fd, fs = moe_fuse(pred_c_names[i], pred_p_names[i])

                moe_name = components_to_pred_name(fd, fs, self.class_names)
                moe_vec = components_to_multihot(fd, fs, self.class_to_idx)

                true_vec = labels[i].cpu().numpy()
                true_set = set(np.where(true_vec == 1)[0].tolist())

                c_vec = components_to_multihot(d_c, s_c, self.class_to_idx)
                p_vec = components_to_multihot(d_p, s_p, self.class_to_idx)

                moe_set = set(np.where(moe_vec == 1)[0].tolist())
                c_set = set(np.where(c_vec == 1)[0].tolist())
                p_set = set(np.where(p_vec == 1)[0].tolist())

                moe_ok = (true_set == moe_set)
                c_ok = (true_set == c_set)
                p_ok = (true_set == p_set)

                total_moe_correct += moe_ok
                total_c_correct += c_ok
                total_p_correct += p_ok

                case_label = classify_fusion_case(d_c, s_c, d_p, s_p)
                case_counter[case_label] += 1

                all_labels.append(true_vec)
                all_preds_moe.append(moe_vec)
                all_preds_c.append(c_vec)
                all_preds_p.append(p_vec)

                true_name = components_to_pred_name(
                    _get_component(true_set, self.class_names, DECEPTION_CLASSES),
                    _get_component(true_set, self.class_names, SUPPRESSION_CLASSES),
                    self.class_names
                )

                per_sample_records.append({
                    "true": true_name,
                    "conformer": pred_c_names[i],
                    "persistence": pred_p_names[i],
                    "moe": moe_name,
                    "conformer_correct": c_ok,
                    "persistence_correct": p_ok,
                    "moe_correct": moe_ok,
                    "case": case_label,
                    "d_conformer": d_c, "s_conformer": s_c,
                    "d_persistence": d_p, "s_persistence": s_p,
                    "fd": fd, "fs": fs,
                })

                total_samples += 1

            # Debug
            if not debug_done:
                print("\n" + "=" * 80)
                print(f"[DEBUG] MoE Hybrid [zero_shot] — First batch predictions")
                print("=" * 80)
                for i in range(min(5, batch_size)):
                    r = per_sample_records[-batch_size + i]
                    print(f"  [{i}] True: {r['true']:<20} | C: {r['conformer']:<20} | "
                          f"P: {r['persistence']:<20} | MoE: {r['moe']:<20} | "
                          f"C:{'T' if r['conformer_correct'] else 'F'} "
                          f"P:{'T' if r['persistence_correct'] else 'F'} "
                          f"MoE:{'T' if r['moe_correct'] else 'F'} | {r['case']}")
                debug_done = True

        # ── Aggregate ──
        all_labels_np = np.array(all_labels)
        all_preds_moe_np = np.array(all_preds_moe)
        all_preds_c_np = np.array(all_preds_c)
        all_preds_p_np = np.array(all_preds_p)

        moe_metrics = compute_multilabel_metrics(all_labels_np, all_preds_moe_np)
        c_metrics = compute_multilabel_metrics(all_labels_np, all_preds_c_np)
        p_metrics = compute_multilabel_metrics(all_labels_np, all_preds_p_np)

        seen_indices, unseen_indices = self._split_seen_unseen(all_labels_np)
        moe_czsl = compute_czsl_metrics(all_labels_np, all_preds_moe_np, seen_indices, unseen_indices)
        c_czsl = compute_czsl_metrics(all_labels_np, all_preds_c_np, seen_indices, unseen_indices)
        p_czsl = compute_czsl_metrics(all_labels_np, all_preds_p_np, seen_indices, unseen_indices)

        moe_metrics.update({f"moe_{k}": v for k, v in moe_czsl.items()})
        c_metrics.update({f"conformer_{k}": v for k, v in c_czsl.items()})
        p_metrics.update({f"persistence_{k}": v for k, v in p_czsl.items()})

        case_accuracy = self._compute_case_accuracy(per_sample_records)

        return {
            "mode": "zero_shot",
            "moe_metrics": moe_metrics,
            "conformer_metrics": c_metrics,
            "persistence_metrics": p_metrics,
            "labels": all_labels_np,
            "predictions_moe": all_preds_moe_np,
            "predictions_conformer": all_preds_c_np,
            "predictions_persistence": all_preds_p_np,
            "per_sample": per_sample_records,
            "case_stats": {
                "counter": dict(case_counter),
                "accuracy": case_accuracy,
            },
            "total_samples": total_samples,
            "seen_indices": seen_indices,
            "unseen_indices": unseen_indices,
        }

    # ── Helpers ──────────────────────────────────────────────────────

    def _decode_multihot(self, probs: torch.Tensor, threshold: float, top_k: int) -> torch.Tensor:
        """Decode probabilities to multi-hot via top-k > threshold."""
        batch_size, num_classes = probs.shape
        topk_vals, topk_idx = torch.topk(probs, k=min(top_k, num_classes), dim=-1)
        preds = torch.zeros(batch_size, num_classes, device=probs.device)
        for b in range(batch_size):
            for j, idx in enumerate(topk_idx[b]):
                if topk_vals[b, j] > threshold:
                    preds[b, idx] = 1.0
        return preds

    def _multihot_to_name(self, vec: np.ndarray) -> str:
        """Convert a multi-hot vector to a prediction name string."""
        parts = []
        for j, v in enumerate(vec):
            if v == 1:
                parts.append(self.class_names[j])
        return "+".join(sorted(parts, key=lambda x: self.class_names.index(x))) if parts else "None"

    def _manual_zero_shot(self, model, x: torch.Tensor, top_k: int = 1) -> list:
        """Manual zero-shot computation as fallback."""
        feat = _encode_for_model(model, x)
        feat = F.normalize(feat.float(), dim=-1)

        if hasattr(model, 'get_cached_combination_features'):
            text_feat = model.get_cached_combination_features()
        elif hasattr(model, '_text_features_cache') and model._text_features_cache is not None:
            text_feat = model._text_features_cache
        else:
            raise RuntimeError("No cached text features available")

        text_feat = F.normalize(text_feat.float(), dim=-1)
        logit_scale = _get_logit_scale(model)
        sim = logit_scale * (feat @ text_feat.T)

        if top_k == 1:
            best_idx = sim.argmax(dim=-1).cpu().tolist()
        else:
            best_idx = sim.topk(top_k, dim=-1).indices.cpu().tolist()

        # Get names
        if hasattr(model, '_combination_names') and model._combination_names is not None:
            names_list = model._combination_names
        elif hasattr(model, 'get_cached_combination_names'):
            names_list = model.get_cached_combination_names()
        else:
            names_list = self.class_names

        if top_k == 1:
            return [names_list[idx] for idx in best_idx]
        else:
            return [[names_list[idx] for idx in indices] for indices in best_idx]

    def _fallback_zero_shot(self, model, x: torch.Tensor, threshold: float,
                            use_combinations: bool = True) -> list:
        """
        Fallback zero-shot with threshold-based decoding.

        Matches the standalone persistence evaluate_by_combination logic:
          1. Compute similarities against cached combination features
          2. Apply softmax
          3. Keep predictions above threshold (fallback to argmax if none)
          4. Parse prediction names to multi-hot via splitting '+'

        Returns:
            List[List[str]] — list of predicted name lists per sample
        """
        feat = _encode_for_model(model, x)
        feat = F.normalize(feat.float(), dim=-1)

        # Get text features (combination or single-class)
        if use_combinations:
            if hasattr(model, 'get_cached_combination_features'):
                text_feat = model.get_cached_combination_features()
                names = model.get_cached_combination_names()
            elif hasattr(model, '_text_features_cache') and model._text_features_cache is not None:
                text_feat = model._text_features_cache
                names = (model._combination_names if hasattr(model, '_combination_names')
                         and model._combination_names is not None
                         else self.class_names)
            else:
                raise RuntimeError("No cached text features available")
        else:
            text_feat = _get_single_text_features(model, self.num_classes)
            names = self.class_names

        text_feat = F.normalize(text_feat.float(), dim=-1)
        logit_scale = _get_logit_scale(model)
        similarities = logit_scale * (feat @ text_feat.T)  # (B, K)
        probs = torch.softmax(similarities, dim=-1)

        predictions = []
        for i in range(x.size(0)):
            preds = [
                names[j]
                for j in range(len(names))
                if probs[i, j].item() > threshold
            ]
            if not preds:
                best = probs[i].argmax().item()
                preds = [names[best]]
            predictions.append(preds)

        return predictions

    def _split_seen_unseen(self, labels: np.ndarray):
        """
        Split sample indices into seen and unseen based on class combination.

        Seen = single-class samples (only one active label)
        Unseen = multi-class samples (deception + suppression combination)
        """
        seen_indices = []
        unseen_indices = []

        for i, label_vec in enumerate(labels):
            active = np.where(label_vec == 1)[0].tolist()
            active_names = [self.class_names[j] for j in active]

            # Classify based on whether it's a known combination
            is_seen = False
            for sc in self.seen_combinations:
                if isinstance(sc, list):
                    sc_names = sorted(sc)
                elif isinstance(sc, str):
                    sc_names = sorted(sc.split('+'))
                else:
                    continue
                if sorted(active_names) == sc_names:
                    is_seen = True
                    break

            if is_seen:
                seen_indices.append(i)
            else:
                unseen_indices.append(i)

        return seen_indices, unseen_indices

    def _compute_case_accuracy(self, per_sample_records: list) -> dict:
        """Compute per-fusion-case accuracy."""
        case_correct = defaultdict(lambda: {"moe": 0, "conformer": 0, "persistence": 0})
        case_total = Counter()

        for r in per_sample_records:
            case = r["case"]
            case_total[case] += 1
            if r["moe_correct"]:
                case_correct[case]["moe"] += 1
            if r["conformer_correct"]:
                case_correct[case]["conformer"] += 1
            if r["persistence_correct"]:
                case_correct[case]["persistence"] += 1

        result = {}
        for case, total in case_total.items():
            result[case] = {
                "count": total,
                "expert1_accuracy": case_correct[case]["conformer"] / total,
                "expert2_accuracy": case_correct[case]["persistence"] / total,
                "moe_accuracy": case_correct[case]["moe"] / total,
            }

        return result


# ============================================================================
# Results output & visualization
# ============================================================================

def print_comparison(results: dict, class_names: list):
    """Print 3-column comparison: Conformer (Expert 1) vs Persistence (Expert 2) vs MoE."""
    c = results["conformer_metrics"]
    p = results["persistence_metrics"]
    m = results["moe_metrics"]

    print("\n" + "=" * 90)
    print("MoE Hybrid — Comparison: Conformer (欺骗) vs Persistence (压制) vs MoE")
    print("=" * 90)
    print(f"{'Metric':<25} {'Conformer (E1)':<18} {'Persistence (E2)':<18} {'MoE Fusion':<18}")
    print("-" * 90)

    for key, name in [
        ("combination_accuracy", "Combination Acc"),
        ("f1_macro", "F1 Macro"),
        ("f1_micro", "F1 Micro"),
        ("f1_samples", "F1 Samples"),
        ("precision_macro", "Precision Macro"),
        ("recall_macro", "Recall Macro"),
    ]:
        vc, vp, vm = c.get(key, 0), p.get(key, 0), m.get(key, 0)
        best = max(vc, vp, vm)
        markers = [
            "<- best" if vc == best and vc > max(vp, vm) else "",
            "<- best" if vp == best and vp > max(vc, vm) else "",
            "<- best" if vm == best and vm > max(vc, vp) else "",
        ]
        print(f"{name:<25} {vc:<18.4f} {vp:<18.4f} {vm:<18.4f}")
        if any(markers):
            print(f"{'':<25} {markers[0]:<18} {markers[1]:<18} {markers[2]:<18}")

    # CZSL metrics — seen / unseen / harmonic mean
    print("-" * 90)
    print(f"{'CZSL Metrics':<25} {'Conformer (E1)':<18} {'Persistence (E2)':<18} {'MoE Fusion':<18}")
    print("-" * 90)
    for c_key, p_key, m_key, name in [
        ("conformer_seen_accuracy", "persistence_seen_accuracy", "moe_seen_accuracy", "Seen Accuracy"),
        ("conformer_unseen_accuracy", "persistence_unseen_accuracy", "moe_unseen_accuracy", "Unseen Accuracy"),
        ("conformer_harmonic_mean", "persistence_harmonic_mean", "moe_harmonic_mean", "Harmonic Mean"),
    ]:
        vc, vp, vm = c.get(c_key, 0), p.get(p_key, 0), m.get(m_key, 0)
        best = max(vc, vp, vm)
        markers = [
            "<- best" if vc == best and vc > max(vp, vm) else "",
            "<- best" if vp == best and vp > max(vc, vm) else "",
            "<- best" if vm == best and vm > max(vc, vp) else "",
        ]
        print(f"{name:<25} {vc:<18.4f} {vp:<18.4f} {vm:<18.4f}")
        if any(markers):
            print(f"{'':<25} {markers[0]:<18} {markers[1]:<18} {markers[2]:<18}")

    print("-" * 90)
    print(f"\nTotal samples: {results['total_samples']}")


def print_case_analysis(results: dict):
    """Print fusion case distribution and accuracy analysis."""
    case_stats = results["case_stats"]
    counter = case_stats["counter"]
    accuracy = case_stats["accuracy"]

    total = sum(counter.values())

    print("\n" + "=" * 90)
    print("Fusion Case Analysis — Distribution & Accuracy")
    print("=" * 90)
    print(f"{'Case':<35} {'Count':>7} {'Ratio':>8} {'C Acc':>8} {'P Acc':>8} {'MoE Acc':>8}")
    print("-" * 90)

    for case_label, count in sorted(counter.items(), key=lambda x: -x[1]):
        acc = accuracy.get(case_label, {})
        ratio = count / total * 100 if total > 0 else 0
        print(f"{case_label:<35} {count:>7} {ratio:>7.1f}% "
              f"{acc.get('expert1_accuracy', 0):>8.4f} "
              f"{acc.get('expert2_accuracy', 0):>8.4f} "
              f"{acc.get('moe_accuracy', 0):>8.4f}")

    print("-" * 90)
    print(f"{'Total':<35} {total:>7}")

    # Synergy analysis
    improved = sum(1 for r in results["per_sample"]
                   if r["moe_correct"] and not r["conformer_correct"] and not r["persistence_correct"])
    degraded = sum(1 for r in results["per_sample"]
                   if not r["moe_correct"] and (r["conformer_correct"] or r["persistence_correct"]))
    both_right = sum(1 for r in results["per_sample"]
                     if r["conformer_correct"] and r["persistence_correct"])
    any_right_moe_wrong = sum(1 for r in results["per_sample"]
                               if (r["conformer_correct"] or r["persistence_correct"])
                               and not r["moe_correct"])

    print(f"\nSynergy Analysis:")
    print(f"  Both experts correct:         {both_right} ({both_right/total*100:.1f}%)")
    print(f"  MoE > both alone (synergy):   {improved} ({improved/total*100:.1f}%)")
    print(f"  MoE < at least one expert:    {degraded} ({degraded/total*100:.1f}%)")
    print(f"  At least 1 expert ✓, MoE ✗:   {any_right_moe_wrong} ({any_right_moe_wrong/total*100:.1f}%)")


def print_per_class_comparison(results: dict, class_names: list):
    """Print per-class F1 comparison."""
    c_f1 = results["conformer_metrics"]["per_class_f1"]
    p_f1 = results["persistence_metrics"]["per_class_f1"]
    m_f1 = results["moe_metrics"]["per_class_f1"]

    print("\n" + "=" * 90)
    print("Per-Class F1 Comparison")
    print("=" * 90)
    print(f"{'Class':<10} {'Type':<12} {'Conformer':>10} {'Persistence':>10} {'MoE':>10} {'Δ(MoE-Best)':>15}")
    print("-" * 90)

    for i, name in enumerate(class_names):
        if i < len(c_f1):
            jtype = "Deception" if name in DECEPTION_CLASSES else "Suppression"
            best_single = max(c_f1[i], p_f1[i])
            delta = m_f1[i] - best_single
            marker = " ▲" if delta > 0.01 else (" ▼" if delta < -0.01 else "")
            print(f"{name:<10} {jtype:<12} {c_f1[i]:>10.4f} {p_f1[i]:>10.4f} {m_f1[i]:>10.4f} {delta:>+14.4f}{marker}")


# ── Visualization ────────────────────────────────────────────────────────
#
# All visualization functions follow the style of multi/evaluate_czsl.py:
#   - Confusion matrix: fig_size = max(10, n * 0.5), raw counts, annot=True, fmt='d', cmap='Blues'
#   - [S]/[U] prefix for seen/unseen combinations
#   - Axis labels: "Predicted Combination" / "True Combination"
#   - DPI 150, bbox_inches='tight' on all saves
#   - grid(True, alpha=0.3) on line/scatter plots
#


def _build_combination_confusion_data(results: dict, class_names: list,
                                       seen_combinations=None, unseen_combinations=None):
    """Shared helper: extract combination labels and build confusion-ready data.

    Returns:
        dict with keys: unique_combs, n_combs, comb_names, comb_to_idx,
                        seen_set, unseen_set, true_combs, pred_combs_moe,
                        pred_combs_c, pred_combs_p, name_to_idx
    """
    def get_comb_labels(labels_matrix):
        combos = []
        for row in labels_matrix:
            active = tuple(sorted(np.where(row == 1)[0].tolist()))
            combos.append(active)
        return combos

    def indices_to_name(indices):
        if not indices:
            return "None"
        return "+".join([class_names[i] for i in indices])

    name_to_idx = {name: i for i, name in enumerate(class_names)}

    true_combs = get_comb_labels(results["labels"])
    pred_moe_combs = get_comb_labels(results["predictions_moe"])
    pred_c_combs = get_comb_labels(results["predictions_conformer"])
    pred_p_combs = get_comb_labels(results["predictions_persistence"])

    # Build seen / unseen sets
    seen_set = set()
    unseen_set = set()
    for comb_list, target_set in [(seen_combinations, seen_set), (unseen_combinations, unseen_set)]:
        if comb_list:
            for comb in comb_list:
                if isinstance(comb, list):
                    indices = tuple(sorted([name_to_idx.get(c, -1) for c in comb if c in name_to_idx]))
                    if indices:
                        target_set.add(indices)

    # Build ordered combination list: seen first, then unseen (matching evaluate_czsl.py)
    configured = []
    for comb_list in [seen_combinations, unseen_combinations]:
        if comb_list:
            for comb in comb_list:
                if isinstance(comb, list):
                    indices = tuple(sorted([name_to_idx.get(c, -1) for c in comb if c in name_to_idx]))
                    if indices and indices not in configured:
                        configured.append(indices)

    if configured:
        unique_combs = configured
    else:
        unique_combs = sorted(set(true_combs + pred_moe_combs + pred_c_combs + pred_p_combs))

    n = len(unique_combs)
    comb_to_idx = {c: i for i, c in enumerate(unique_combs)}

    # Display names with [S]/[U] prefix (matching evaluate_czsl.py)
    comb_names = []
    for comb in unique_combs:
        name = indices_to_name(comb)
        if comb in seen_set:
            name = f"[S] {name}"
        elif comb in unseen_set:
            name = f"[U] {name}"
        comb_names.append(name)

    return {
        "unique_combs": unique_combs,
        "n_combs": n,
        "comb_names": comb_names,
        "comb_to_idx": comb_to_idx,
        "seen_set": seen_set,
        "unseen_set": unseen_set,
        "true_combs": true_combs,
        "pred_combs_moe": pred_moe_combs,
        "pred_combs_c": pred_c_combs,
        "pred_combs_p": pred_p_combs,
        "name_to_idx": name_to_idx,
    }


def plot_confusion_comparison(results: dict, class_names: list, save_path: str,
                               seen_combinations=None, unseen_combinations=None):
    """
    3-panel confusion matrix comparison: Conformer | Persistence | MoE.

    Each panel follows evaluate_czsl.py style:
      - Square figure, fig_size = max(10, n_combs * 0.5)
      - Raw counts with annot=True, fmt='d', cmap='Blues'
      - Labels: "Predicted Combination" / "True Combination"
      - [S]/[U] prefix on combination names
    """
    data = _build_combination_confusion_data(results, class_names,
                                              seen_combinations, unseen_combinations)
    unique_combs = data["unique_combs"]
    n = data["n_combs"]
    comb_names = data["comb_names"]
    comb_to_idx = data["comb_to_idx"]
    true_combs = data["true_combs"]

    def build_cm(pred_combs):
        cm = np.zeros((n, n), dtype=int)
        other = 0
        for tc, pc in zip(true_combs, pred_combs):
            if tc in comb_to_idx and pc in comb_to_idx:
                cm[comb_to_idx[tc], comb_to_idx[pc]] += 1
            else:
                other += 1
        if other > 0:
            print(f"  ({other} samples outside configured combinations)")
        return cm

    cms = [
        build_cm(data["pred_combs_c"]),
        build_cm(data["pred_combs_p"]),
        build_cm(data["pred_combs_moe"]),
    ]
    titles = [
        "Conformer (Expert 1 - Deception)",
        "Persistence (Expert 2 - Suppression)",
        "MoE Hybrid Fusion",
    ]

    # Each panel as a separate square figure (matching evaluate_czsl single-panel style)
    # plus a combined 3-panel overview for comparison
    fig_size = max(10, n * 0.5)

    # ── Single-panel MoE confusion matrix (exact match to evaluate_czsl.py) ──
    cm_moe = cms[2]
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(cm_moe, annot=True, fmt='d', cmap='Blues',
                xticklabels=comb_names, yticklabels=comb_names, ax=ax)
    ax.set_xlabel('Predicted Combination')
    ax.set_ylabel('True Combination')
    ax.set_title('MoE Hybrid Combination Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    moe_path = save_path.replace('.png', '_moe.png')
    os.makedirs(os.path.dirname(moe_path), exist_ok=True)
    fig.savefig(moe_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"MoE confusion matrix saved to: {moe_path}")

    # ── 3-panel comparison overview ──
    fig, axes = plt.subplots(1, 3, figsize=(min(fig_size * 3, 48), fig_size))
    for ax_idx, (cm, title) in enumerate(zip(cms, titles)):
        ax = axes[ax_idx]
        sns.heatmap(cm, annot=(n <= 15), fmt='d', cmap='Blues',
                    xticklabels=comb_names if n <= 30 else [],
                    yticklabels=comb_names if n <= 30 else [],
                    ax=ax, cbar=(ax_idx == 2))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Predicted Combination' if n <= 30 else '')
        ax.set_ylabel('True Combination' if n <= 30 else '')
        if n <= 30:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            plt.setp(ax.get_yticklabels(), rotation=0)

    plt.suptitle("MoE Hybrid — Confusion Matrix Comparison", fontsize=13, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Confusion comparison saved to: {save_path}")


def plot_roc_curves(labels: np.ndarray, probabilities: np.ndarray,
                    class_names: list, save_dir: str,
                    prefix: str = "moe_hybrid", mode_title: str = ""):
    """
    ROC curves — per-class + micro/macro average (matching evaluate_czsl.py).

    Generates two figures:
      1. {prefix}_roc_per_class.png
      2. {prefix}_roc_average.png
    """
    n_classes = labels.shape[1]

    # Per-class ROC + AUC
    fpr_dict, tpr_dict, auc_dict = {}, {}, {}
    for i in range(n_classes):
        if np.sum(labels[:, i]) == 0:
            continue
        fpr_dict[i], tpr_dict[i], _ = roc_curve(labels[:, i], probabilities[:, i])
        auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

    # Micro-average
    fpr_micro, tpr_micro, _ = roc_curve(labels.ravel(), probabilities.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)

    # Macro-average
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in fpr_dict]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in fpr_dict:
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= len(fpr_dict)
    auc_macro = auc(all_fpr, mean_tpr)

    title_suffix = f" ({mode_title})" if mode_title else ""

    # === Per-class ROC ===
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(figsize=(8, 7))
    for idx, i in enumerate(fpr_dict):
        color = colors[idx % 10]
        ax.plot(fpr_dict[i], tpr_dict[i], color=color, linewidth=1.2,
                label=f'{class_names[i]} (AUC={auc_dict[i]:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', linewidth=1.0, linestyle=':', alpha=0.7)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'Per-class ROC Curves{title_suffix}', fontsize=14)
    ax.legend(loc='lower right', fontsize=7)
    ax.grid(True, alpha=0.3)

    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f'{prefix}_roc_per_class.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # === Average ROC ===
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr_micro, tpr_micro, color='darkorange', linewidth=2.5,
            label=f'Micro-average (AUC={auc_micro:.3f})')
    ax.plot(all_fpr, mean_tpr, color='darkgreen', linewidth=2.5, linestyle='--',
            label=f'Macro-average (AUC={auc_macro:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', linewidth=1.0, linestyle=':', alpha=0.7)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'Average ROC Curves{title_suffix}', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(save_dir, f'{prefix}_roc_average.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"ROC curves saved to: {save_dir}/{prefix}_roc_*.png")


def plot_pr_curves(labels: np.ndarray, probabilities: np.ndarray,
                   class_names: list, save_dir: str,
                   prefix: str = "moe_hybrid", mode_title: str = ""):
    """
    Precision-Recall curves — per-class + micro/macro average (matching evaluate_czsl.py).

    Generates two figures:
      1. {prefix}_pr_per_class.png
      2. {prefix}_pr_average.png
    """
    n_classes = labels.shape[1]
    n_positives = labels.sum(axis=0)
    baseline = n_positives.sum() / labels.size

    # Per-class PR
    prec_dict, rec_dict, ap_dict = {}, {}, {}
    for i in range(n_classes):
        if np.sum(labels[:, i]) == 0:
            continue
        prec_dict[i], rec_dict[i], _ = precision_recall_curve(labels[:, i], probabilities[:, i])
        ap_dict[i] = average_precision_score(labels[:, i], probabilities[:, i])

    # Micro-average
    prec_micro, rec_micro, _ = precision_recall_curve(labels.ravel(), probabilities.ravel())
    ap_micro = average_precision_score(labels, probabilities, average='micro')

    # Macro-average
    all_rec = np.linspace(0, 1, 1000)
    mean_prec = np.zeros_like(all_rec)
    for i in prec_dict:
        mean_prec += np.interp(all_rec, rec_dict[i][::-1], prec_dict[i][::-1])[::-1]
    mean_prec /= len(prec_dict)
    ap_macro = float(np.trapezoid(mean_prec, all_rec))

    title_suffix = f" ({mode_title})" if mode_title else ""

    # === Per-class PR ===
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(figsize=(8, 7))
    for idx, i in enumerate(prec_dict):
        color = colors[idx % 10]
        ax.plot(rec_dict[i], prec_dict[i], color=color, linewidth=1.2,
                label=f'{class_names[i]} (AP={ap_dict[i]:.3f})')
    ax.axhline(y=baseline, color='navy', linewidth=1.0, linestyle=':', alpha=0.7,
               label=f'Baseline ({baseline:.3f})')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Per-class Precision-Recall Curves{title_suffix}', fontsize=14)
    ax.legend(loc='lower left', fontsize=7)
    ax.grid(True, alpha=0.3)

    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f'{prefix}_pr_per_class.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # === Average PR ===
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(rec_micro, prec_micro, color='darkorange', linewidth=2.5,
            label=f'Micro-average (AP={ap_micro:.3f})')
    ax.plot(all_rec, mean_prec, color='darkgreen', linewidth=2.5, linestyle='--',
            label=f'Macro-average (AP={ap_macro:.3f})')
    ax.axhline(y=baseline, color='navy', linewidth=1.0, linestyle=':', alpha=0.7,
               label=f'Baseline ({baseline:.3f})')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Average Precision-Recall Curves{title_suffix}', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(save_dir, f'{prefix}_pr_average.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"PR curves saved to: {save_dir}/{prefix}_pr_*.png")


def plot_improvement_scatter(results: dict, save_path: str):
    """Scatter plot: MoE improvement/degradation per sample."""
    per_sample = results["per_sample"]

    c_correct = np.array([r["conformer_correct"] for r in per_sample])
    p_correct = np.array([r["persistence_correct"] for r in per_sample])
    m_correct = np.array([r["moe_correct"] for r in per_sample])

    categories = np.full(len(per_sample), "other", dtype=object)
    both_right = c_correct & p_correct
    both_wrong = ~c_correct & ~p_correct
    only_c = c_correct & ~p_correct
    only_p = ~c_correct & p_correct

    categories[both_right & m_correct] = "both_right_moe_right"
    categories[both_right & ~m_correct] = "both_right_moe_wrong"
    categories[both_wrong & m_correct] = "synergy_gain"
    categories[both_wrong & ~m_correct] = "both_wrong"
    categories[only_c & m_correct] = "c_right_moe_right"
    categories[only_c & ~m_correct] = "c_right_moe_wrong"
    categories[only_p & m_correct] = "p_right_moe_right"
    categories[only_p & ~m_correct] = "p_right_moe_wrong"

    color_map = {
        "both_right_moe_right": "#2ecc71",
        "both_right_moe_wrong": "#e74c3c",
        "synergy_gain": "#3498db",
        "both_wrong": "#95a5a6",
        "c_right_moe_right": "#27ae60",
        "c_right_moe_wrong": "#c0392b",
        "p_right_moe_right": "#1abc9c",
        "p_right_moe_wrong": "#e67e22",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for cat, color in color_map.items():
        mask = categories == cat
        if mask.sum() > 0:
            jitter_x = np.random.uniform(-0.15, 0.15, mask.sum())
            jitter_y = np.random.uniform(-0.15, 0.15, mask.sum())
            ax.scatter(0 + jitter_x, 1 + jitter_y, c=color, alpha=0.5, s=20,
                      label=f"{cat} ({mask.sum()})")

    ax.set_xlim(-1, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Conformer Correct", "Persistence Correct"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Wrong", "Correct"])
    ax.set_title("MoE Improvement Scatter (per sample)", fontsize=12)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Improvement scatter saved to: {save_path}")


def plot_case_distribution(results: dict, save_path: str):
    """Pie chart + bar chart of fusion case distribution."""
    case_stats = results["case_stats"]
    counter = case_stats["counter"]
    accuracy = case_stats["accuracy"]

    total = sum(counter.values())
    sorted_cases = sorted(counter.items(), key=lambda x: -x[1])

    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(18, 8))

    # Pie chart
    labels = []
    sizes = []
    for case, count in sorted_cases[:8]:
        labels.append(f"{case}\n({count/total*100:.1f}%)")
        sizes.append(count)
    if len(sorted_cases) > 8:
        other_count = sum(c for _, c in sorted_cases[8:])
        labels.append(f"Other ({other_count/total*100:.1f}%)")
        sizes.append(other_count)

    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    ax_pie.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90)
    ax_pie.set_title("Fusion Case Distribution", fontsize=12)

    # Bar chart — accuracy per case
    case_names = []
    moe_accs = []
    c_accs = []
    p_accs = []
    for case, count in sorted_cases[:10]:
        case_names.append(case[:30])
        acc = accuracy.get(case, {})
        moe_accs.append(acc.get("moe_accuracy", 0))
        c_accs.append(acc.get("expert1_accuracy", 0))
        p_accs.append(acc.get("expert2_accuracy", 0))

    x = np.arange(len(case_names))
    w = 0.25
    ax_bar.bar(x - w, c_accs, w, label='Conformer', color='#3498db')
    ax_bar.bar(x, p_accs, w, label='Persistence', color='#e74c3c')
    ax_bar.bar(x + w, moe_accs, w, label='MoE', color='#2ecc71')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(case_names, rotation=45, ha='right', fontsize=8)
    ax_bar.set_ylabel("Accuracy")
    ax_bar.set_title("Accuracy by Fusion Case", fontsize=12)
    ax_bar.legend()
    ax_bar.set_ylim(0, 1)

    plt.suptitle("MoE Hybrid — Case Analysis", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Case distribution saved to: {save_path}")


def plot_jnr_curves(jnr_results: dict, save_path: str, class_names: list = None):
    """
    Per-JNR performance: 2x2 subplot matching evaluate_czsl.py style.

    Top-left:  Accuracy (MoE overall, seen, unseen)
    Top-right: F1 Macro vs JNR
    Bottom-left:  Per-class recall vs JNR
    Bottom-right: Per-class F1 vs JNR
    """
    jnrs = sorted(jnr_results.keys())
    m_accs, m_f1s = [], []
    seen_accs, unseen_accs = [], []
    per_class_recall = defaultdict(list)
    per_class_f1 = defaultdict(list)

    for jnr in jnrs:
        r = jnr_results[jnr]
        mm = r["moe_metrics"]
        m_accs.append(mm["combination_accuracy"])
        m_f1s.append(mm["f1_macro"])
        seen_accs.append(mm.get("moe_seen_accuracy", 0))
        unseen_accs.append(mm.get("moe_unseen_accuracy", 0))

        if class_names:
            pcf1 = mm.get("per_class_f1", [])
            for c in range(len(class_names)):
                if c < len(pcf1):
                    per_class_f1[c].append(pcf1[c])
                else:
                    per_class_f1[c].append(0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: Accuracy curves
    axes[0, 0].plot(jnrs, m_accs, 'b-o', label='MoE Overall Accuracy', linewidth=2)
    axes[0, 0].plot(jnrs, seen_accs, 'g-s', label='MoE Seen Accuracy', linewidth=2)
    axes[0, 0].plot(jnrs, unseen_accs, 'r-^', label='MoE Unseen Accuracy', linewidth=2)
    axes[0, 0].set_xlabel('JNR (dB)')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy vs JNR Level')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: F1 Macro
    axes[0, 1].plot(jnrs, m_f1s, 'b-o', label='MoE F1 Macro', linewidth=2)
    axes[0, 1].set_xlabel('JNR (dB)')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('F1 Score vs JNR Level')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: Per-class F1
    if class_names and per_class_f1:
        ax = axes[1, 0]
        for c, name in enumerate(class_names):
            if c in per_class_f1 and len(per_class_f1[c]) == len(jnrs):
                ax.plot(jnrs, per_class_f1[c], '-o', label=name, linewidth=1.5, markersize=4)
        ax.set_xlabel('JNR (dB)')
        ax.set_ylabel('F1 Score')
        ax.set_title('Per-Class F1 Score vs JNR Level')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)

    # Bottom-right: MoE harmonic mean comparison (if available)
    ax = axes[1, 1]
    m_hm = [r["moe_metrics"].get("moe_harmonic_mean", 0) for r in [jnr_results[j] for j in jnrs]]
    c_hm = [r["conformer_metrics"].get("conformer_harmonic_mean", 0) for r in [jnr_results[j] for j in jnrs]]
    p_hm = [r["persistence_metrics"].get("persistence_harmonic_mean", 0) for r in [jnr_results[j] for j in jnrs]]
    ax.plot(jnrs, c_hm, 'o-', color='#3498db', label='Conformer', linewidth=2)
    ax.plot(jnrs, p_hm, 's-', color='#e74c3c', label='Persistence', linewidth=2)
    ax.plot(jnrs, m_hm, 'D-', color='#2ecc71', label='MoE', linewidth=2.5)
    ax.set_xlabel('JNR (dB)')
    ax.set_ylabel('Harmonic Mean')
    ax.set_title('CZSL Harmonic Mean vs JNR Level')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"JNR curves saved to: {save_path}")


def plot_per_class_f1(results: dict, class_names: list, save_path: str):
    """
    Per-class F1 vertical bar chart matching evaluate_unified.py style.

    Two panels: Deception classes (Blues) + Suppression classes (Greens).
    Gradient colors, macro F1 reference line, text labels above bars.
    """
    m_f1 = results["moe_metrics"]["per_class_f1"]
    m_f1_macro = results["moe_metrics"]["f1_macro"]

    # Split into deception / suppression
    d_names = [n for n in class_names if n in DECEPTION_CLASSES]
    s_names = [n for n in class_names if n in SUPPRESSION_CLASSES]
    d_indices = [class_names.index(n) for n in d_names]
    s_indices = [class_names.index(n) for n in s_names]

    d_f1 = [m_f1[i] for i in d_indices]
    s_f1 = [m_f1[i] for i in s_indices]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (classes, f1_vals, cmap_name, title) in enumerate([
        (d_names, d_f1, "Blues", "Deception Classes - Per-Class F1 (MoE)"),
        (s_names, s_f1, "Greens", "Suppression Classes - Per-Class F1 (MoE)"),
    ]):
        ax = axes[idx]
        x = np.arange(len(classes))
        colors = plt.cm.get_cmap(cmap_name)(np.linspace(0.3, 0.9, max(len(f1_vals), 1)))
        bars = ax.bar(x, f1_vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_ylabel('F1 Score')
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.axhline(y=m_f1_macro, color='r', linestyle='--',
                   label=f'MoE Macro F1: {m_f1_macro:.4f}')
        ax.legend(fontsize=9)
        for bar, val in zip(bars, f1_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Per-class F1 saved to: {save_path}")


def plot_feature_tsne(features: np.ndarray, labels: np.ndarray,
                      class_names: list, save_path: str,
                      seen_combinations=None, unseen_combinations=None):
    """
    t-SNE feature space visualization (matching evaluate_czsl.py style).

    [S]=Seen (circle), [U]=Unseen (triangle), other (square).
    """
    num_samples = features.shape[0]
    if num_samples <= 1:
        print("Too few samples for t-SNE, skipping.")
        return

    perplexity = min(30, num_samples - 1) if num_samples > 1 else 1

    print(f"Running t-SNE on {num_samples} samples (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embedded = tsne.fit_transform(features)

    # Build combination labels
    seen_set = set()
    unseen_set = set()
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    if seen_combinations:
        for comb in seen_combinations:
            if isinstance(comb, list):
                indices = tuple(sorted([name_to_idx.get(c, -1) for c in comb if c in name_to_idx]))
                if indices:
                    seen_set.add(indices)
    if unseen_combinations:
        for comb in unseen_combinations:
            if isinstance(comb, list):
                indices = tuple(sorted([name_to_idx.get(c, -1) for c in comb if c in name_to_idx]))
                if indices:
                    unseen_set.add(indices)

    comb_labels = []
    for i in range(num_samples):
        active = tuple(sorted(np.where(labels[i] == 1)[0].tolist()))
        comb_labels.append(active)

    unique_combs = sorted(set(comb_labels), key=lambda c: (c not in seen_set, c))
    num_combs = len(unique_combs)

    fig, ax = plt.subplots(figsize=(14, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, num_combs)))

    for comb_idx, comb in enumerate(unique_combs):
        mask = [c == comb for c in comb_labels]
        color = colors[comb_idx % 20]
        if comb in unseen_set:
            marker, label_suffix = '^', ' [U]'
        elif comb in seen_set:
            marker, label_suffix = 'o', ' [S]'
        else:
            marker, label_suffix = 's', ''

        name = "+".join([class_names[i] for i in comb]) if comb else "None"
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                   c=[color], marker=marker, alpha=0.6, s=30,
                   label=f"{label_suffix} {name}")

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('Feature Space Visualization (t-SNE)\n[S]=Seen, [U]=Unseen')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"t-SNE visualization saved to: {save_path}")


def plot_feature_umap(features: np.ndarray, labels: np.ndarray,
                      class_names: list, save_path: str,
                      seen_combinations=None, unseen_combinations=None):
    """
    UMAP feature space visualization (matching evaluate_czsl.py style).

    [S]=Seen (circle), [U]=Unseen (triangle), other (square).
    """
    try:
        import umap
    except ImportError:
        print("umap-learn not installed, skipping UMAP visualization.")
        return

    num_samples = features.shape[0]
    if num_samples <= 1:
        print("Too few samples for UMAP, skipping.")
        return

    n_neighbors = min(15, num_samples - 1) if num_samples > 1 else 2
    print(f"Running UMAP on {num_samples} samples (n_neighbors={n_neighbors})...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
    embedded = reducer.fit_transform(features)

    seen_set = set()
    unseen_set = set()
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    if seen_combinations:
        for comb in seen_combinations:
            if isinstance(comb, list):
                indices = tuple(sorted([name_to_idx.get(c, -1) for c in comb if c in name_to_idx]))
                if indices:
                    seen_set.add(indices)
    if unseen_combinations:
        for comb in unseen_combinations:
            if isinstance(comb, list):
                indices = tuple(sorted([name_to_idx.get(c, -1) for c in comb if c in name_to_idx]))
                if indices:
                    unseen_set.add(indices)

    comb_labels = []
    for i in range(num_samples):
        active = tuple(sorted(np.where(labels[i] == 1)[0].tolist()))
        comb_labels.append(active)

    unique_combs = sorted(set(comb_labels))
    num_combs = len(unique_combs)

    fig, ax = plt.subplots(figsize=(14, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, num_combs)))

    for comb_idx, comb in enumerate(unique_combs):
        mask = [c == comb for c in comb_labels]
        color = colors[comb_idx % 20]
        if comb in unseen_set:
            marker, label_suffix = '^', ' [U]'
        elif comb in seen_set:
            marker, label_suffix = 'o', ' [S]'
        else:
            marker, label_suffix = 's', ''

        name = "+".join([class_names[i] for i in comb]) if comb else "None"
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                   c=[color], marker=marker, alpha=0.6, s=30,
                   label=f"{label_suffix} {name}")

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Feature Space Visualization (UMAP)\n[S]=Seen, [U]=Unseen')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"UMAP visualization saved to: {save_path}")


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MoE Hybrid Interference Evaluation — Conformer (欺骗) + Persistence (压制)"
    )
    parser.add_argument("--config", default=None,
                        help="Path to config YAML (default: moe/config.yaml relative to project root)")
    parser.add_argument("--checkpoint_persistence", required=True,
                        help="Path to Persistence model checkpoint (.pt)")
    parser.add_argument("--checkpoint_conformer", required=True,
                        help="Path to Conformer model checkpoint (.pt)")
    parser.add_argument("--mode", choices=["zero_shot", "by_combination", "all"], default="all",
                        help="Evaluation mode (default: all)")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test",
                        help="Data split (default: test)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for results and plots (overrides config)")
    parser.add_argument("--device", default="cuda",
                        help="Torch device (default: cuda)")
    parser.add_argument("--by_jnr", action="store_true",
                        help="Also run per-JNR evaluation")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")
    args = parser.parse_args()

    # ── Resolve config path ──
    if args.config is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    else:
        config_path = args.config

    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    # ── Output directory ──
    output_dir = args.output_dir or config.get('evaluation', {}).get('output_dir', 'results/moe_hybrid')
    os.makedirs(output_dir, exist_ok=True)

    # ── Load models ──
    device = args.device
    p_model = load_persistence_model(config, args.checkpoint_persistence, device)
    c_model = load_conformer_model(config, args.checkpoint_conformer, device)

    # ── Load data ──
    print(f"\n[Data] Loading dual-modal {args.split} data...")
    loader, jnr_loaders, class_names = create_dual_modal_dataloaders(
        config, split=args.split
    )

    # ── Create evaluator ──
    evaluator = HybridMoEEvaluator(p_model, c_model, config, device)

    # ── Run evaluation ──
    modes = ["zero_shot", "by_combination"] if args.mode == "all" else [args.mode]

    for mode in modes:
        print(f"\n{'#'*80}")
        print(f"# MoE Hybrid Evaluation — Mode: {mode}")
        print(f"{'#'*80}")

        results = evaluator.evaluate(loader, mode=mode)

        # ── Print results ──
        print_comparison(results, class_names)
        print_case_analysis(results)
        print_per_class_comparison(results, class_names)

        # ── Save predictions ──
        if config.get('evaluation', {}).get('save_predictions', True):
            pred_file = os.path.join(output_dir, f"moe_hybrid_{mode}_predictions.json")
            save_data = {
                "config": {
                    "checkpoint_persistence": args.checkpoint_persistence,
                    "checkpoint_conformer": args.checkpoint_conformer,
                    "mode": mode,
                    "split": args.split,
                },
                "moe_metrics": {k: float(v) if isinstance(v, (np.floating, np.integer))
                                else v.tolist() if isinstance(v, np.ndarray) else v
                                for k, v in results["moe_metrics"].items()},
                "conformer_metrics": {k: float(v) if isinstance(v, (np.floating, np.integer))
                                      else v.tolist() if isinstance(v, np.ndarray) else v
                                      for k, v in results["conformer_metrics"].items()},
                "persistence_metrics": {k: float(v) if isinstance(v, (np.floating, np.integer))
                                        else v.tolist() if isinstance(v, np.ndarray) else v
                                        for k, v in results["persistence_metrics"].items()},
                "per_sample": results["per_sample"],
                "total_samples": results["total_samples"],
            }
            with open(pred_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
            print(f"\nPredictions saved to: {pred_file}")

        # ── Generate visualizations ──
        plot_confusion_comparison(
            results, class_names,
            save_path=os.path.join(output_dir, f"moe_hybrid_{mode}_confusion.png"),
            seen_combinations=evaluator.seen_combinations,
            unseen_combinations=evaluator.unseen_combinations,
        )
        plot_improvement_scatter(
            results,
            save_path=os.path.join(output_dir, f"moe_hybrid_{mode}_improvement.png"),
        )
        plot_case_distribution(
            results,
            save_path=os.path.join(output_dir, f"moe_hybrid_{mode}_cases.png"),
        )
        plot_per_class_f1(
            results, class_names,
            save_path=os.path.join(output_dir, f"moe_hybrid_{mode}_per_class_f1.png"),
        )

    # ── Per-JNR evaluation (optional) ──
    if args.by_jnr:
        print(f"\n{'#'*80}")
        print(f"# Per-JNR Evaluation")
        print(f"{'#'*80}")

        jnr_results = evaluator.evaluate_by_jnr(jnr_loaders, mode=modes[0])

        # Summary table
        print("\n" + "=" * 80)
        print("Per-JNR Summary")
        print("=" * 80)
        print(f"{'JNR':>6} {'Conformer':>12} {'Persistence':>12} {'MoE':>12} {'MoE HM':>12}")
        print("-" * 80)
        for jnr in sorted(jnr_results.keys()):
            r = jnr_results[jnr]
            print(f"{'+'+str(jnr):>6} "
                  f"{r['conformer_metrics']['combination_accuracy']:>12.4f} "
                  f"{r['persistence_metrics']['combination_accuracy']:>12.4f} "
                  f"{r['moe_metrics']['combination_accuracy']:>12.4f} "
                  f"{r['moe_metrics'].get('moe_harmonic_mean', 0):>12.4f}")

        plot_jnr_curves(
            jnr_results,
            save_path=os.path.join(output_dir, f"moe_hybrid_jnr_curves.png"),
            class_names=class_names,
        )

        # Save per-JNR results
        jnr_save = {}
        for jnr, r in jnr_results.items():
            jnr_save[str(jnr)] = {
                "moe_combination_acc": float(r["moe_metrics"]["combination_accuracy"]),
                "conformer_combination_acc": float(r["conformer_metrics"]["combination_accuracy"]),
                "persistence_combination_acc": float(r["persistence_metrics"]["combination_accuracy"]),
                "moe_harmonic_mean": float(r["moe_metrics"].get("moe_harmonic_mean", 0)),
            }
        with open(os.path.join(output_dir, "moe_hybrid_jnr_summary.json"), 'w') as f:
            json.dump(jnr_save, f, indent=2)
        print(f"\nPer-JNR summary saved to: {os.path.join(output_dir, 'moe_hybrid_jnr_summary.json')}")

    print(f"\n{'='*80}")
    print(f"MoE Hybrid Evaluation complete. Results in: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
