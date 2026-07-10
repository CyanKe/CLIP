"""
Evaluation script for Persistence Spectrum CLIP CZSL model.

Supports three modes:
    all             — zero_shot + by_combination (like evaluate_czsl)
    zero_shot       — global zero-shot evaluation
    by_combination  — per-combination (seen/unseen) breakdown
    by_jnr          — per-JNR performance analysis

Visualization flags (matching evaluate_czsl):
    --visualize     — all visualizations (confusion matrix, ROC, PR, t-SNE, UMAP)
    --tsne          — t-SNE only
    --umap          — UMAP only
    --roc           — ROC curves only
    --pr            — PR curves only

Usage:
    python persistence/evaluate.py --checkpoint CHKPT --mode all --split test --visualize
    python persistence/evaluate.py --checkpoint CHKPT --mode by_jnr --split test --output_dir results --visualize
    python persistence/evaluate.py --checkpoint CHKPT --mode zero_shot --roc --pr

Adapted from multi/evaluate_czsl.py for PersistenceCLIPForCZSL.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _parent)

from persistence.model import PersistenceCLIPForCZSL, create_persistence_model
from persistence.data import (
    PersistenceDataset, collate_fn, TokenizerWrapper,
    AblationDataset,
)


# ===========================================================================
# Standalone plot functions (decoupled from evaluator, like evaluate_czsl)
# ===========================================================================

def plot_roc_curves(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: list,
    save_dir: str = None,
    prefix: str = "roc",
    mode_title: str = "",
):
    """Draw ROC curves — per-class + micro/macro average.

    Generates two figures:
      1. {prefix}_roc_per_class.png — one curve per class, different colors
      2. {prefix}_roc_average.png  — Micro / Macro average curves

    Args:
        labels: one-hot labels [n_samples, n_classes]
        probabilities: softmax probabilities [n_samples, n_classes]
        class_names: list of class name strings
        save_dir: output directory (None = show)
        prefix: filename prefix
        mode_title: title suffix (e.g. "by_combination" or "JNR=+10")
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

    # === Figure 1: Per-class ROC curves ===
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(figsize=(8, 7))

    for idx, i in enumerate(fpr_dict):
        color = colors[idx % 10]
        ax.plot(
            fpr_dict[i], tpr_dict[i],
            color=color, linewidth=1.2,
            label=f'{class_names[i]} (AUC={auc_dict[i]:.3f})',
        )

    ax.plot([0, 1], [0, 1], color='navy', linewidth=1.0, linestyle=':', alpha=0.7)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'Per-class ROC Curves{title_suffix}', fontsize=14)
    ax.legend(loc='lower right', fontsize=7, ncol=1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, f"{prefix}_roc_per_class.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Per-class ROC saved to {path}")
    else:
        plt.show()
    plt.close()

    # === Figure 2: Micro / Macro average ROC ===
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(
        fpr_micro, tpr_micro,
        color='darkorange', linewidth=2.5,
        label=f'Micro-average (AUC={auc_micro:.3f})',
    )
    ax.plot(
        all_fpr, mean_tpr,
        color='darkgreen', linewidth=2.5, linestyle='--',
        label=f'Macro-average (AUC={auc_macro:.3f})',
    )
    ax.plot([0, 1], [0, 1], color='navy', linewidth=1.0, linestyle=':', alpha=0.7)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'Average ROC Curves{title_suffix}', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, f"{prefix}_roc_average.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Average ROC saved to {path}")
    else:
        plt.show()
    plt.close()


def plot_pr_curves(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: list,
    save_dir: str = None,
    prefix: str = "pr",
    mode_title: str = "",
):
    """Draw Precision-Recall curves — per-class + micro/macro average.

    Generates two figures:
      1. {prefix}_pr_per_class.png — one curve per class, different colors
      2. {prefix}_pr_average.png  — Micro / Macro average curves
    """
    n_classes = labels.shape[1]

    # Per-class PR + AP
    precision_dict, recall_dict, ap_dict = {}, {}, {}
    for i in range(n_classes):
        if np.sum(labels[:, i]) == 0:
            continue
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(
            labels[:, i], probabilities[:, i],
        )
        ap_dict[i] = average_precision_score(labels[:, i], probabilities[:, i])

    # Micro-average
    precision_micro, recall_micro, _ = precision_recall_curve(
        labels.ravel(), probabilities.ravel(),
    )
    ap_micro = average_precision_score(labels.ravel(), probabilities.ravel())

    # Macro-average
    all_recall = np.unique(np.concatenate([recall_dict[i] for i in recall_dict]))
    mean_precision = np.zeros_like(all_recall)
    for i in precision_dict:
        mean_precision += np.interp(all_recall, recall_dict[i][::-1], precision_dict[i][::-1])
    mean_precision /= len(precision_dict)
    ap_macro = np.trapezoid(mean_precision, all_recall)

    # No-skill baseline
    baseline = labels.sum() / labels.size

    title_suffix = f" ({mode_title})" if mode_title else ""

    # === Figure 1: Per-class PR curves ===
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(figsize=(8, 7))

    for idx, i in enumerate(precision_dict):
        color = colors[idx % 10]
        ax.plot(
            recall_dict[i], precision_dict[i],
            color=color, linewidth=1.2,
            label=f'{class_names[i]} (AP={ap_dict[i]:.3f})',
        )

    ax.axhline(y=baseline, color='navy', linewidth=1.0, linestyle=':', alpha=0.7,
               label=f'Baseline ({baseline:.3f})')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Per-class Precision-Recall Curves{title_suffix}', fontsize=14)
    ax.legend(loc='lower left', fontsize=7, ncol=1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, f"{prefix}_pr_per_class.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Per-class PR saved to {path}")
    else:
        plt.show()
    plt.close()

    # === Figure 2: Micro / Macro average PR ===
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(
        recall_micro, precision_micro,
        color='darkorange', linewidth=2.5,
        label=f'Micro-average (AP={ap_micro:.3f})',
    )
    ax.plot(
        all_recall, mean_precision,
        color='darkgreen', linewidth=2.5, linestyle='--',
        label=f'Macro-average (AP={ap_macro:.3f})',
    )
    ax.axhline(y=baseline, color='navy', linewidth=1.0, linestyle=':', alpha=0.7,
               label=f'Baseline ({baseline:.3f})')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Average Precision-Recall Curves{title_suffix}', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, f"{prefix}_pr_average.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Average PR saved to {path}")
    else:
        plt.show()
    plt.close()


# ===========================================================================
# Helper: convert combination names → indices
# ===========================================================================

def _unwrap_combinations(raw, key: str = "seen_combinations") -> list:
    """Safely extract combination lists from config, handling nested structures.

    Handles two config formats:
      Normal:   czsl.seen_combinations: [...]           → list
      Nested:   czsl.seen_combinations.seen_combinations: [...] → dict then list
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        # Try the same key nested inside
        inner = raw.get(key, [])
        if isinstance(inner, list):
            return inner
        # Try common keys
        for k in ['combinations', 'seen', 'unseen']:
            v = raw.get(k)
            if isinstance(v, list):
                return v
    return []


def _convert_combination_names_to_indices(combinations: list, class_names: list) -> list:
    """Convert combination name-lists to index-lists.

    e.g. [["DFTJ", "AJ"], ["ISRJ"]]  →  [[0, 9], [1]]
    """
    if not combinations:
        return []
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    result = []
    for comb in combinations:
        indices = []
        for name in comb:
            if name in name_to_idx:
                indices.append(name_to_idx[name])
            else:
                print(f"Warning: Unknown class name '{name}' in combination {comb}")
        if indices:
            result.append(sorted(indices))
    return result


# ===========================================================================
# Evaluator
# ===========================================================================

class PersistenceEvaluator:
    """Evaluation suite for PersistenceCLIPForCZSL — matching CZSLEvaluator API."""

    def __init__(self, model: PersistenceCLIPForCZSL, config: dict, device: torch.device):
        self.model = model
        self.config = config
        self.device = device

        jamming_classes = config.get('jamming_classes', [])
        self.class_names = [jc['name'] if isinstance(jc, dict) else jc for jc in jamming_classes]
        self.num_classes = len(self.class_names)

        self.czsl_config = config.get('czsl', {})
        self.eval_config = config.get('evaluation', {})

    # ------------------------------------------------------------------
    # Helpers: single-class probabilities & image features
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_single_class_probs(self, images: torch.Tensor) -> np.ndarray:
        """Compute single-class softmax probabilities for ROC/PR curves.

        Uses only the single-class entries from the cached text features
        (they are stored first by cache_text_features with include_single=True).
        """
        single_features = self.model._text_features_cache[:self.num_classes]
        single_features = F.normalize(single_features, dim=-1)

        image_features = self.model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)

        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * (image_features @ single_features.T)
        probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy(), image_features.cpu().numpy()

    # ------------------------------------------------------------------
    # Shared: build a single-JNR loader on demand
    # ------------------------------------------------------------------

    def _build_single_jnr_loader(self, config: dict, split: str, jnr: int,
                                  ablation_mode: str = 'persistence'):
        """Build a DataLoader for a single JNR level. Returns (loader, num_samples) or None."""
        data_config = config.get('data', {})
        base_path = data_config.get('base_path')
        image_size = data_config.get('image_size', 224)
        batch_size = config.get('train', {}).get('batch_size', 32)
        persistence_var_name = data_config.get('persistence_var_name', 'all_persistences')
        persistence_suffix = data_config.get('persistence_suffix', 'echo_persistences')
        stft_suffix = data_config.get('stft_suffix', 'echo_stfts')
        stft_var_name = data_config.get('stft_var_name', 'all_stfts')
        pin_memory = data_config.get('pin_memory', True)

        data_folder = os.path.join(base_path, f'JNR_+{jnr}')
        metadata_file = os.path.join(data_folder, f'{split}_echo_metadata.json')
        if not os.path.exists(metadata_file):
            return None

        persistence_file = None
        if ablation_mode in ('persistence', 'fusion'):
            pf = os.path.join(data_folder, f'{split}_{persistence_suffix}.mat')
            if not os.path.exists(pf):
                return None
            persistence_file = pf

        stft_file = None
        if ablation_mode in ('stft', 'fusion'):
            sf = os.path.join(data_folder, f'{split}_{stft_suffix}.mat')
            if not os.path.exists(sf):
                return None
            stft_file = sf

        if ablation_mode == 'persistence':
            ds = PersistenceDataset(
                persistence_file=persistence_file,
                metadata_file=metadata_file,
                persistence_var_name=persistence_var_name,
                class_names=self.class_names,
                image_size=image_size,
                apply_clip_norm=True,
            )
        else:
            ds = AblationDataset(
                persistence_file=persistence_file,
                metadata_file=metadata_file,
                persistence_var_name=persistence_var_name,
                stft_file=stft_file,
                stft_var_name=stft_var_name,
                class_names=self.class_names,
                image_size=image_size,
                apply_clip_norm=True,
                ablation_mode=ablation_mode,
            )

        tokenizer = TokenizerWrapper(model_type="clip")
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=pin_memory,
            collate_fn=partial(collate_fn, tokenizer_fn=tokenizer, model_type="clip"),
        )
        return loader

    # ------------------------------------------------------------------
    # Zero-shot evaluation — per-JNR lazy loading
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_zero_shot(
        self,
        config: dict = None,
        split: str = 'test',
        ablation_mode: str = 'persistence',
        dataloader=None,
        threshold: float = None,
    ) -> dict:
        """Run zero-shot evaluation with per-JNR lazy loading.

        Accepts either a pre-built dataloader (backward compat) or
        config+split+ablation_mode for on-demand per-JNR loading.
        """
        if threshold is None:
            threshold = self.eval_config.get('threshold', 0.5)

        self.model.eval()

        # Backward compat: if a pre-built dataloader is passed, use it directly
        if dataloader is not None:
            return self._evaluate_zero_shot_on_loader(dataloader, threshold)

        # Per-JNR lazy loading
        data_config = config.get('data', {})
        jnr_start = data_config.get('jnr_start', 0)
        jnr_end = data_config.get('jnr_end', 20)
        jnr_step = data_config.get('jnr_step', 1)
        jnr_levels = list(range(jnr_start, jnr_end + 1, jnr_step))

        all_preds = []
        all_labels = []
        all_probs = []
        all_features = []

        for jnr in jnr_levels:
            loader = self._build_single_jnr_loader(config, split, jnr, ablation_mode)
            if loader is None:
                continue
            result = self._evaluate_zero_shot_on_loader(loader, threshold)
            all_preds.append(result["predictions"])
            all_labels.append(result["labels"])
            all_probs.append(result["probabilities"])
            all_features.append(result["features"])
            del loader

        all_preds = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
        all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
        all_probs = np.concatenate(all_probs, axis=0) if all_probs else np.array([])
        all_features = np.concatenate(all_features, axis=0) if all_features else np.array([])

        metrics = self._compute_multilabel_metrics(all_labels, all_preds)
        metrics['num_samples'] = len(all_preds)

        return {
            "metrics": metrics,
            "labels": all_labels,
            "predictions": all_preds,
            "probabilities": all_probs,
            "features": all_features,
        }

    def _evaluate_zero_shot_on_loader(self, dataloader, threshold: float) -> dict:
        """Internal: run zero-shot eval on a single DataLoader."""
        all_preds = []
        all_labels = []
        all_probs = []
        all_features = []

        for batch_data in tqdm(dataloader, desc="Zero-shot eval", leave=False):
            if len(batch_data) >= 6:
                images, _, text_tokens, labels, texts, metas = batch_data[:6]
            else:
                images, text_tokens, labels, texts, metas = batch_data

            images = images.to(self.device)
            labels = labels.to(self.device)

            probs_single, img_feats = self._compute_single_class_probs(images)
            all_probs.append(probs_single)
            all_features.append(img_feats)

            predictions = self.model.zero_shot_predict(images, threshold=threshold)

            for pred_names in predictions:
                pred_vec = np.zeros(self.num_classes, dtype=np.float32)
                for name in pred_names:
                    for part in name.split('+'):
                        part = part.strip()
                        if part in self.class_names:
                            pred_vec[self.class_names.index(part)] = 1.0
                all_preds.append(pred_vec)

            all_labels.append(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        all_features = np.concatenate(all_features, axis=0)

        return {
            "predictions": all_preds,
            "labels": all_labels,
            "probabilities": all_probs,
            "features": all_features,
        }

    # ------------------------------------------------------------------
    # Multilabel metrics
    # ------------------------------------------------------------------

    def _compute_multilabel_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict:
        """Compute multilabel classification metrics."""
        # Per-class F1
        per_class_f1 = {}
        for i, cls_name in enumerate(self.class_names):
            if y_true[:, i].sum() > 0 or y_pred[:, i].sum() > 0:
                f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
                per_class_f1[cls_name] = f1

        # Micro / Macro / Sample F1
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_samples = f1_score(y_true, y_pred, average='samples', zero_division=0)

        # Precision / Recall
        precision = precision_score(y_true, y_pred, average='samples', zero_division=0)
        recall = recall_score(y_true, y_pred, average='samples', zero_division=0)

        # Subset accuracy (exact match)
        subset_acc = accuracy_score(y_true, y_pred)

        return {
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_samples': f1_samples,
            'precision': precision,
            'recall': recall,
            'subset_accuracy': subset_acc,
            'per_class_f1': per_class_f1,
        }

    # ------------------------------------------------------------------
    # By-combination evaluation (seen vs unseen)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_by_combination(
        self,
        config: dict = None,
        split: str = 'test',
        ablation_mode: str = 'persistence',
        dataloader=None,
        threshold: float = None,
    ) -> dict:
        """Evaluate broken down by seen and unseen combinations — per-JNR lazy loading.

        Accepts either a pre-built dataloader (backward compat) or
        config+split+ablation_mode for on-demand per-JNR loading.
        """
        if threshold is None:
            threshold = self.czsl_config.get('zero_shot', {}).get('threshold', 0.14)

        # Backward compat: pre-built dataloader
        if dataloader is not None:
            return self._evaluate_by_combination_on_loader(dataloader, threshold)

        # Per-JNR lazy loading
        data_config = config.get('data', {})
        jnr_start = data_config.get('jnr_start', 0)
        jnr_end = data_config.get('jnr_end', 20)
        jnr_step = data_config.get('jnr_step', 1)
        jnr_levels = list(range(jnr_start, jnr_end + 1, jnr_step))

        all_results = []
        for jnr in jnr_levels:
            loader = self._build_single_jnr_loader(config, split, jnr, ablation_mode)
            if loader is None:
                continue
            result = self._evaluate_by_combination_on_loader(loader, threshold)
            all_results.append(result)
            del loader

        # Merge results across JNRs
        if not all_results:
            return {'_labels': np.array([]), '_predictions': np.array([]),
                    '_probabilities': np.array([]), '_features': np.array([])}

        merged = {}
        # Merge seen
        seen_preds = np.concatenate([r.get('_seen_preds', np.array([])) for r in all_results if r.get('_seen_preds') is not None and len(r['_seen_preds']) > 0], axis=0) if any(r.get('_seen_preds') is not None and len(r['_seen_preds']) > 0 for r in all_results) else []
        seen_labels = np.concatenate([r.get('_seen_labels', np.array([])) for r in all_results if r.get('_seen_labels') is not None and len(r['_seen_labels']) > 0], axis=0) if any(r.get('_seen_labels') is not None and len(r['_seen_labels']) > 0 for r in all_results) else []
        if len(seen_preds) > 0:
            merged['seen'] = self._compute_multilabel_metrics(seen_labels, seen_preds)
            merged['seen']['count'] = len(seen_preds)
            print(f"\nSeen combinations ({len(seen_preds)} samples):")
            print(f"  F1_macro: {merged['seen']['f1_macro']:.4f}")
            print(f"  F1_micro: {merged['seen']['f1_micro']:.4f}")

        # Merge unseen
        unseen_preds = np.concatenate([r.get('_unseen_preds', np.array([])) for r in all_results if r.get('_unseen_preds') is not None and len(r['_unseen_preds']) > 0], axis=0) if any(r.get('_unseen_preds') is not None and len(r['_unseen_preds']) > 0 for r in all_results) else []
        unseen_labels = np.concatenate([r.get('_unseen_labels', np.array([])) for r in all_results if r.get('_unseen_labels') is not None and len(r['_unseen_labels']) > 0], axis=0) if any(r.get('_unseen_labels') is not None and len(r['_unseen_labels']) > 0 for r in all_results) else []
        if len(unseen_preds) > 0:
            merged['unseen'] = self._compute_multilabel_metrics(unseen_labels, unseen_preds)
            merged['unseen']['count'] = len(unseen_preds)
            print(f"\nUnseen combinations ({len(unseen_preds)} samples):")
            print(f"  F1_macro: {merged['unseen']['f1_macro']:.4f}")
            print(f"  F1_micro: {merged['unseen']['f1_micro']:.4f}")

        # Merge all
        all_preds = np.concatenate([r['_predictions'] for r in all_results], axis=0)
        all_labels = np.concatenate([r['_labels'] for r in all_results], axis=0)
        all_probs = np.concatenate([r['_probabilities'] for r in all_results], axis=0)
        all_features = np.concatenate([r['_features'] for r in all_results], axis=0)

        merged['_labels'] = all_labels
        merged['_predictions'] = all_preds
        merged['_probabilities'] = all_probs
        merged['_features'] = all_features

        return merged

    def _evaluate_by_combination_on_loader(self, dataloader, threshold: float) -> dict:
        """Internal: run by-combination eval on a single DataLoader."""
        seen_combos_raw = _unwrap_combinations(
            self.czsl_config.get('seen_combinations', []), 'seen_combinations')
        unseen_combos_raw = _unwrap_combinations(
            self.czsl_config.get('unseen_combinations', []), 'unseen_combinations')

        def combo_key(combo):
            return tuple(sorted(combo))

        seen_keys = set()
        for sc in seen_combos_raw:
            seen_keys.add(combo_key(sc))
        unseen_keys = set()
        for uc in unseen_combos_raw:
            unseen_keys.add(combo_key(uc))

        self.model.eval()
        seen_preds, seen_labels = [], []
        unseen_preds, unseen_labels = [], []
        all_preds_list, all_labels_list = [], []
        all_probs_list, all_features_list = [], []

        for batch_data in tqdm(dataloader, desc="By-combination eval", leave=False):
            if len(batch_data) >= 6:
                images, _, text_tokens, labels, texts, metas = batch_data[:6]
            else:
                images, text_tokens, labels, texts, metas = batch_data

            images = images.to(self.device)
            labels_np = labels.cpu().numpy()

            probs_single, img_feats = self._compute_single_class_probs(images)
            all_probs_list.append(probs_single)
            all_features_list.append(img_feats)

            predictions = self.model.zero_shot_predict(images, threshold=threshold)

            for i, pred_names in enumerate(predictions):
                pred_vec = np.zeros(self.num_classes, dtype=np.float32)
                for name in pred_names:
                    for part in name.split('+'):
                        part = part.strip()
                        if part in self.class_names:
                            pred_vec[self.class_names.index(part)] = 1.0

                all_preds_list.append(pred_vec)
                all_labels_list.append(labels_np[i])

                true_classes = tuple(sorted([
                    self.class_names[j] for j in range(self.num_classes)
                    if labels_np[i, j] > 0
                ]))

                if true_classes in seen_keys or len(true_classes) <= 1:
                    seen_preds.append(pred_vec)
                    seen_labels.append(labels_np[i])
                elif true_classes in unseen_keys:
                    unseen_preds.append(pred_vec)
                    unseen_labels.append(labels_np[i])

        return {
            '_labels': np.array(all_labels_list),
            '_predictions': np.array(all_preds_list),
            '_probabilities': np.concatenate(all_probs_list, axis=0) if all_probs_list else np.array([]),
            '_features': np.concatenate(all_features_list, axis=0) if all_features_list else np.array([]),
            '_seen_preds': np.array(seen_preds) if seen_preds else None,
            '_seen_labels': np.array(seen_labels) if seen_labels else None,
            '_unseen_preds': np.array(unseen_preds) if unseen_preds else None,
            '_unseen_labels': np.array(unseen_labels) if unseen_labels else None,
        }

    # ------------------------------------------------------------------
    # By-JNR evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_by_jnr(
        self,
        config: dict,
        split: str = 'test',
        ablation_mode: str = 'persistence',
        threshold: float = None,
        output_dir: str = None,
    ) -> dict:
        """Evaluate separately at each JNR level — loads one JNR at a time to save memory.

        Args:
            config: full YAML config dict
            split: data split to evaluate
            ablation_mode: 'persistence', 'stft', or 'fusion'
            threshold: prediction threshold override
            output_dir: directory for JNR metrics panel plot

        Returns:
            dict: {jnr: metrics, ...}
        """
        if threshold is None:
            threshold = self.czsl_config.get('zero_shot', {}).get('threshold', 0.14)

        # Data config for on-demand loader building
        data_config = config.get('data', {})
        base_path = data_config.get('base_path')
        jnr_start = data_config.get('jnr_start', 0)
        jnr_end = data_config.get('jnr_end', 20)
        jnr_step = data_config.get('jnr_step', 1)
        image_size = data_config.get('image_size', 224)
        batch_size = config.get('train', {}).get('batch_size', 32)
        persistence_var_name = data_config.get('persistence_var_name', 'all_persistences')
        persistence_suffix = data_config.get('persistence_suffix', 'echo_persistences')
        stft_suffix = data_config.get('stft_suffix', 'echo_stfts')
        stft_var_name = data_config.get('stft_var_name', 'all_stfts')
        pin_memory = data_config.get('pin_memory', True)

        tokenizer = TokenizerWrapper(model_type="clip")

        # Build seen/unseen sets
        seen_combos_raw = _unwrap_combinations(
            self.czsl_config.get('seen_combinations', []), 'seen_combinations')
        unseen_combos_raw = _unwrap_combinations(
            self.czsl_config.get('unseen_combinations', []), 'unseen_combinations')

        def combo_key(c):
            return tuple(sorted(c))

        seen_set = set(combo_key(sc) for sc in seen_combos_raw)
        unseen_set = set(combo_key(uc) for uc in unseen_combos_raw)

        jnr_levels = list(range(jnr_start, jnr_end + 1, jnr_step))
        jnr_results = {}
        all_jnr_metrics = []

        for jnr in jnr_levels:
            # --- Build single-JNR loader on demand (freed after eval) ---
            data_folder = os.path.join(base_path, f'JNR_+{jnr}')
            metadata_file = os.path.join(data_folder, f'{split}_echo_metadata.json')
            if not os.path.exists(metadata_file):
                continue

            persistence_file = None
            if ablation_mode in ('persistence', 'fusion'):
                pf = os.path.join(data_folder, f'{split}_{persistence_suffix}.mat')
                if os.path.exists(pf):
                    persistence_file = pf
                else:
                    continue

            stft_file = None
            if ablation_mode in ('stft', 'fusion'):
                sf = os.path.join(data_folder, f'{split}_{stft_suffix}.mat')
                if os.path.exists(sf):
                    stft_file = sf
                else:
                    continue

            if ablation_mode == 'persistence':
                ds = PersistenceDataset(
                    persistence_file=persistence_file,
                    metadata_file=metadata_file,
                    persistence_var_name=persistence_var_name,
                    class_names=self.class_names,
                    image_size=image_size,
                    apply_clip_norm=True,
                )
            else:
                ds = AblationDataset(
                    persistence_file=persistence_file,
                    metadata_file=metadata_file,
                    persistence_var_name=persistence_var_name,
                    stft_file=stft_file,
                    stft_var_name=stft_var_name,
                    class_names=self.class_names,
                    image_size=image_size,
                    apply_clip_norm=True,
                    ablation_mode=ablation_mode,
                )

            loader = DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=pin_memory,
                collate_fn=partial(collate_fn, tokenizer_fn=tokenizer, model_type="clip"),
            )

            self.model.eval()

            _all_labels = []
            _all_preds = []
            _all_probs = []
            _all_features = []
            class_stats = np.zeros((self.num_classes, 4), dtype=np.int64)
            seen_correct, seen_total = 0, 0
            unseen_correct, unseen_total = 0, 0

            for batch_data in tqdm(loader, desc=f"JNR=+{jnr}"):
                if len(batch_data) >= 6:
                    images, _, text_tokens, labels, texts, metas = batch_data[:6]
                else:
                    images, text_tokens, labels, texts, metas = batch_data

                images = images.to(self.device)
                labels_dev = labels.to(self.device)
                labels_np = labels_dev.cpu().numpy()

                # Single-class probabilities + image features
                probs_single, img_feats = self._compute_single_class_probs(images)
                _all_probs.append(probs_single)
                _all_features.append(img_feats)

                # Zero-shot prediction
                predictions = self.model.zero_shot_predict(images, threshold=threshold)

                preds = np.zeros((images.size(0), self.num_classes), dtype=np.float32)
                for i, pred_names in enumerate(predictions):
                    for name in pred_names:
                        for part in name.split('+'):
                            part = part.strip()
                            if part in self.class_names:
                                preds[i, self.class_names.index(part)] = 1.0

                _all_preds.append(preds)
                _all_labels.append(labels_np)

                # Per-class statistics
                for c in range(self.num_classes):
                    true_c = labels_np[:, c]
                    pred_c = preds[:, c]
                    class_stats[c, 0] += np.sum((true_c == 1) & (pred_c == 1))  # TP
                    class_stats[c, 1] += np.sum((true_c == 0) & (pred_c == 1))  # FP
                    class_stats[c, 2] += np.sum((true_c == 1) & (pred_c == 0))  # FN
                    class_stats[c, 3] += np.sum((true_c == 0) & (pred_c == 0))  # TN

                # Seen/unseen accuracy
                for i in range(images.size(0)):
                    true_comb = tuple(sorted(
                        np.where(labels_np[i] == 1)[0].tolist()
                    ))
                    pred_comb = tuple(sorted(
                        np.where(preds[i] == 1)[0].tolist()
                    ))
                    is_correct = (true_comb == pred_comb)

                    if true_comb in seen_set:
                        seen_total += 1
                        if is_correct:
                            seen_correct += 1
                    elif true_comb in unseen_set:
                        unseen_total += 1
                        if is_correct:
                            unseen_correct += 1

            all_labels_np = np.concatenate(_all_labels, axis=0)
            all_preds_np = np.concatenate(_all_preds, axis=0)
            all_probs_np = np.concatenate(_all_probs, axis=0)
            all_feats_np = np.concatenate(_all_features, axis=0)

            # Per-class metrics
            per_class_recall = np.zeros(self.num_classes)
            per_class_precision = np.zeros(self.num_classes)
            per_class_f1 = np.zeros(self.num_classes)
            for c in range(self.num_classes):
                tp, fp, fn, tn = class_stats[c]
                per_class_recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0
                per_class_precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0
                if per_class_precision[c] + per_class_recall[c] > 0:
                    per_class_f1[c] = (2 * per_class_precision[c] * per_class_recall[c]
                                       / (per_class_precision[c] + per_class_recall[c]))

            total_samples = len(all_preds_np)
            metrics = {
                'f1_macro': f1_score(all_labels_np, all_preds_np, average='macro', zero_division=0),
                'f1_micro': f1_score(all_labels_np, all_preds_np, average='micro', zero_division=0),
                'f1_samples': f1_score(all_labels_np, all_preds_np, average='samples', zero_division=0),
                'precision': precision_score(all_labels_np, all_preds_np, average='samples', zero_division=0),
                'recall': recall_score(all_labels_np, all_preds_np, average='samples', zero_division=0),
                'subset_accuracy': accuracy_score(all_labels_np, all_preds_np),
                'jnr': jnr,
                'num_samples': total_samples,
                'combination_accuracy': seen_correct + unseen_correct,
                'total_samples': seen_total + unseen_total,
                'seen_accuracy': seen_correct / seen_total if seen_total > 0 else 0,
                'seen_samples': seen_total,
                'unseen_accuracy': unseen_correct / unseen_total if unseen_total > 0 else 0,
                'unseen_samples': unseen_total,
                'per_class_recall': per_class_recall,
                'per_class_precision': per_class_precision,
                'per_class_f1': per_class_f1,
                # For per-JNR ROC/PR:
                'labels': all_labels_np,
                'probabilities': all_probs_np,
                'features': all_feats_np,
            }

            jnr_results[jnr] = metrics
            all_jnr_metrics.append(metrics)
            print(f"  JNR=+{jnr}: F1_macro={metrics['f1_macro']:.4f}, "
                  f"F1_micro={metrics['f1_micro']:.4f}")

            # --- Per-JNR confusion matrix ---
            if output_dir:
                jnr_dir = os.path.join(output_dir, "by_jnr")
                os.makedirs(jnr_dir, exist_ok=True)
                # Convert combo name-lists to index-lists for the plot
                seen_idx = _convert_combination_names_to_indices(seen_combos_raw, self.class_names)
                unseen_idx = _convert_combination_names_to_indices(unseen_combos_raw, self.class_names)
                prefix = "persistence" if ablation_mode == "persistence" else f"persistence_{ablation_mode}"
                self.plot_confusion_by_combination(
                    all_labels_np, all_preds_np,
                    save_path=os.path.join(jnr_dir, f"{prefix}_confusion_jnr_{jnr:+.0f}.png"),
                    seen_combinations=seen_idx,
                    unseen_combinations=unseen_idx,
                )

            # Free this JNR's data before loading the next one
            del loader, ds

        # Plot JNR metrics panel → save in by_jnr subfolder
        if output_dir and all_jnr_metrics:
            jnr_dir = os.path.join(output_dir, "by_jnr")
            os.makedirs(jnr_dir, exist_ok=True)
            prefix = "persistence" if ablation_mode == "persistence" else f"persistence_{ablation_mode}"
            self.plot_jnr_metrics(jnr_results, save_path=os.path.join(jnr_dir, f'{prefix}_jnr_metrics.png'))

        return jnr_results

    # ==================================================================
    #  Visualization methods (ported from evaluate_czsl.py)
    # ==================================================================

    # ------------------------------------------------------------------
    # JNR metrics panel (2×2 — like evaluate_czsl)
    # ------------------------------------------------------------------

    def plot_jnr_metrics(self, results: dict, save_path: str = None):
        """Plot JNR metrics panel: Accuracy, F1, Per-class Recall, Per-class F1."""
        jnrs = sorted(results.keys())
        accuracies = []
        f1_macros = []
        seen_accs = []
        unseen_accs = []

        for jnr in jnrs:
            m = results[jnr]
            total = m.get('total_samples', m.get('num_samples', 1))
            acc = m.get('combination_accuracy', 0) / total if total > 0 else 0
            accuracies.append(acc)
            f1_macros.append(m['f1_macro'])
            seen_accs.append(m.get('seen_accuracy', 0))
            unseen_accs.append(m.get('unseen_accuracy', 0))

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Accuracy curve
        axes[0, 0].plot(jnrs, accuracies, 'b-o', label='Overall Accuracy', linewidth=2)
        axes[0, 0].plot(jnrs, seen_accs, 'g-s', label='Seen Accuracy', linewidth=2)
        axes[0, 0].plot(jnrs, unseen_accs, 'r-^', label='Unseen Accuracy', linewidth=2)
        axes[0, 0].set_xlabel('JNR (dB)')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy vs JNR Level')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # F1 curve
        axes[0, 1].plot(jnrs, f1_macros, 'b-o', label='F1 Macro', linewidth=2)
        axes[0, 1].set_xlabel('JNR (dB)')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('F1 Score vs JNR Level')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Per-class Recall
        ax = axes[1, 0]
        for c, name in enumerate(self.class_names):
            recalls = [results[jnr].get('per_class_recall', [0] * self.num_classes)[c]
                       for jnr in jnrs]
            ax.plot(jnrs, recalls, '-o', label=name, linewidth=1.5, markersize=4)
        ax.set_xlabel('JNR (dB)')
        ax.set_ylabel('Recall')
        ax.set_title('Per-Class Recall vs JNR Level')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)

        # Per-class F1
        ax = axes[1, 1]
        for c, name in enumerate(self.class_names):
            f1s = [results[jnr].get('per_class_f1', [0] * self.num_classes)[c]
                   for jnr in jnrs]
            ax.plot(jnrs, f1s, '-o', label=name, linewidth=1.5, markersize=4)
        ax.set_xlabel('JNR (dB)')
        ax.set_ylabel('F1 Score')
        ax.set_title('Per-Class F1 Score vs JNR Level')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"JNR metrics panel saved to {save_path}")
        else:
            plt.show()
        plt.close()

    # ------------------------------------------------------------------
    # Combination confusion matrix
    # ------------------------------------------------------------------

    def plot_confusion_by_combination(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        save_path: str = None,
        seen_combinations: list = None,
        unseen_combinations: list = None,
    ):
        """Plot combination confusion matrix.

        Args:
            labels: true multilabel matrix [N, C]
            preds: predicted multilabel matrix [N, C]
            save_path: output path
            seen_combinations: list of index-lists for seen combos
            unseen_combinations: list of index-lists for unseen combos
        """
        # Determine which combinations to show
        if seen_combinations is not None or unseen_combinations is not None:
            configured_combs = []
            if seen_combinations:
                for comb in seen_combinations:
                    ct = tuple(sorted(comb))
                    if ct not in configured_combs:
                        configured_combs.append(ct)
            if unseen_combinations:
                for comb in unseen_combinations:
                    ct = tuple(sorted(comb))
                    if ct not in configured_combs:
                        configured_combs.append(ct)
            unique_combs = configured_combs
            print(f"\nUsing configured combinations: {len(unique_combs)} "
                  f"(Seen: {len(seen_combinations or [])}, Unseen: {len(unseen_combinations or [])})")
        else:
            true_combs = [tuple(sorted(np.where(labels[i] == 1)[0].tolist()))
                          for i in range(len(labels))]
            pred_combs = [tuple(sorted(np.where(preds[i] == 1)[0].tolist()))
                          for i in range(len(preds))]
            unique_combs = sorted(set(true_combs + pred_combs))

        comb_to_idx = {comb: i for i, comb in enumerate(unique_combs)}
        n_combs = len(unique_combs)
        confusion = np.zeros((n_combs, n_combs), dtype=int)

        other_count = 0
        for i in range(len(labels)):
            true_comb = tuple(sorted(np.where(labels[i] == 1)[0].tolist()))
            pred_comb = tuple(sorted(np.where(preds[i] == 1)[0].tolist()))
            if true_comb in comb_to_idx and pred_comb in comb_to_idx:
                confusion[comb_to_idx[true_comb], comb_to_idx[pred_comb]] += 1
            else:
                other_count += 1

        if other_count > 0:
            print(f"Samples not in configured combinations: {other_count}")

        # Build names with [S]/[U] markers
        seen_set = set(tuple(sorted(c)) for c in (seen_combinations or []))
        unseen_set = set(tuple(sorted(c)) for c in (unseen_combinations or []))

        comb_names = []
        for comb in unique_combs:
            if len(comb) == 0:
                name = "None"
            else:
                name = "+".join([self.class_names[i] for i in comb])

            if seen_combinations is not None or unseen_combinations is not None:
                if comb in seen_set:
                    name = f"[S] {name}"
                elif comb in unseen_set:
                    name = f"[U] {name}"
            comb_names.append(name)

        fig_size = max(10, n_combs * 0.5)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
                    xticklabels=comb_names, yticklabels=comb_names, ax=ax)
        ax.set_xlabel('Predicted Combination')
        ax.set_ylabel('True Combination')
        ax.set_title('Combination Confusion Matrix (Zero-Shot)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        plt.close()

    # ------------------------------------------------------------------
    # t-SNE feature visualization
    # ------------------------------------------------------------------

    def plot_feature_tsne(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        save_path: str = None,
        seen_combinations: list = None,
        unseen_combinations: list = None,
    ):
        """Plot t-SNE of image features."""
        print("Computing t-SNE projection...")

        num_samples = len(features)
        perplexity = min(30, num_samples - 1) if num_samples > 1 else 1

        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        features_2d = tsne.fit_transform(features)

        # Convert multilabel to combination labels
        comb_labels = []
        for label in labels:
            active = tuple(sorted(np.where(label == 1)[0].tolist()))
            comb_labels.append(active)

        seen_set = set(tuple(sorted(c)) for c in (seen_combinations or []))
        unseen_set = set(tuple(sorted(c)) for c in (unseen_combinations or []))

        unique_combs = sorted(set(comb_labels))
        comb_to_name = {}
        for comb in unique_combs:
            if len(comb) == 0:
                comb_to_name[comb] = "None"
            else:
                comb_to_name[comb] = "+".join([self.class_names[i] for i in comb])

        num_combs = len(unique_combs)
        colors = plt.cm.tab20(np.linspace(0, 1, max(20, num_combs)))

        fig, ax = plt.subplots(figsize=(14, 10))

        for idx, comb in enumerate(unique_combs):
            mask = np.array([c == comb for c in comb_labels])
            if mask.sum() > 0:
                name = comb_to_name[comb]
                if comb in seen_set:
                    name = f"[S] {name}"
                    marker = 'o'
                elif comb in unseen_set:
                    name = f"[U] {name}"
                    marker = '^'
                else:
                    marker = 's'

                ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                           c=[colors[idx % 20]], label=name, alpha=0.6, s=30,
                           marker=marker)

        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title('Feature Space Visualization (t-SNE)\n[S]=Seen, [U]=Unseen, □=Other')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"t-SNE plot saved to {save_path}")
        else:
            plt.show()
        plt.close()

    # ------------------------------------------------------------------
    # UMAP feature visualization
    # ------------------------------------------------------------------

    def plot_feature_umap(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        save_path: str = None,
        seen_combinations: list = None,
        unseen_combinations: list = None,
    ):
        """Plot UMAP of image features."""
        try:
            import umap
        except ImportError:
            print("UMAP not installed. Install with: pip install umap-learn")
            return

        print("Computing UMAP projection...")

        reducer = umap.UMAP(
            n_components=2, random_state=42,
            n_neighbors=15, min_dist=0.1,
        )
        features_2d = reducer.fit_transform(features)

        comb_labels = []
        for label in labels:
            active = tuple(sorted(np.where(label == 1)[0].tolist()))
            comb_labels.append(active)

        seen_set = set(tuple(sorted(c)) for c in (seen_combinations or []))
        unseen_set = set(tuple(sorted(c)) for c in (unseen_combinations or []))

        unique_combs = sorted(set(comb_labels))
        comb_to_name = {}
        for comb in unique_combs:
            if len(comb) == 0:
                comb_to_name[comb] = "None"
            else:
                comb_to_name[comb] = "+".join([self.class_names[i] for i in comb])

        num_combs = len(unique_combs)
        colors = plt.cm.tab20(np.linspace(0, 1, max(20, num_combs)))

        fig, ax = plt.subplots(figsize=(14, 10))

        for idx, comb in enumerate(unique_combs):
            mask = np.array([c == comb for c in comb_labels])
            if mask.sum() > 0:
                name = comb_to_name[comb]
                if comb in seen_set:
                    name = f"[S] {name}"
                    marker = 'o'
                elif comb in unseen_set:
                    name = f"[U] {name}"
                    marker = '^'
                else:
                    marker = 's'

                ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                           c=[colors[idx % 20]], label=name, alpha=0.6, s=30,
                           marker=marker)

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('Feature Space Visualization (UMAP)\n[S]=Seen, [U]=Unseen, □=Other')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"UMAP plot saved to {save_path}")
        else:
            plt.show()
        plt.close()

    # ------------------------------------------------------------------
    # Label co-occurrence
    # ------------------------------------------------------------------

    def plot_label_cooccurrence(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        save_path: str = None,
    ):
        """Plot true / predicted label co-occurrence matrices."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        true_cooc = (labels.T @ labels).astype(int)
        nc = labels.shape[1]
        sns.heatmap(true_cooc, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names[:nc],
                    yticklabels=self.class_names[:nc],
                    ax=axes[0])
        axes[0].set_title('True Label Co-occurrence')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].tick_params(axis='y', rotation=0)

        pred_cooc = (preds.T @ preds).astype(int)
        sns.heatmap(pred_cooc, annot=True, fmt='d', cmap='Greens',
                    xticklabels=self.class_names[:nc],
                    yticklabels=self.class_names[:nc],
                    ax=axes[1])
        axes[1].set_title('Predicted Label Co-occurrence')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].tick_params(axis='y', rotation=0)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Label co-occurrence plot saved to {save_path}")
        else:
            plt.show()
        plt.close()

    # ------------------------------------------------------------------
    # Print helpers
    # ------------------------------------------------------------------

    def print_metrics(self, metrics: dict):
        """Pretty-print evaluation metrics."""
        print(f"\n{'='*60}")
        print(f"Evaluation Results ({metrics.get('num_samples', '?')} samples)")
        print(f"{'='*60}")
        print(f"  F1_macro:       {metrics['f1_macro']:.4f}")
        print(f"  F1_micro:       {metrics['f1_micro']:.4f}")
        print(f"  F1_samples:     {metrics['f1_samples']:.4f}")
        print(f"  Precision:      {metrics['precision']:.4f}")
        print(f"  Recall:         {metrics['recall']:.4f}")
        print(f"  Subset accuracy: {metrics['subset_accuracy']:.4f}")
        if metrics.get('per_class_f1'):
            print(f"\nPer-class F1:")
            for cls_name, f1 in sorted(metrics['per_class_f1'].items()):
                print(f"  {cls_name:8s}: {f1:.4f}")

    def print_jnr_results(self, results: dict, save_path: str = None):
        """Print and optionally save per-JNR results table."""
        print("\n" + "=" * 80)
        print("Evaluation Results by JNR Level")
        print("=" * 80)
        print(f"{'JNR':>6} | {'Accuracy':>10} | {'F1_Macro':>10} | "
              f"{'Seen_Acc':>10} | {'Unseen_Acc':>10} | {'Samples':>8}")
        print("-" * 80)

        lines = []
        for jnr, m in sorted(results.items()):
            total = m.get('total_samples', m.get('num_samples', 1))
            acc = m.get('combination_accuracy', 0) / total if total > 0 else 0
            print(f"{jnr:>6} | {acc:>10.4f} | {m['f1_macro']:>10.4f} | "
                  f"{m.get('seen_accuracy', 0):>10.4f} | "
                  f"{m.get('unseen_accuracy', 0):>10.4f} | "
                  f"{total:>8}")
            lines.append(f"{jnr},{acc:.4f},{m['f1_macro']:.4f},"
                        f"{m.get('seen_accuracy', 0):.4f},"
                        f"{m.get('unseen_accuracy', 0):.4f},"
                        f"{total}")

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("JNR,Accuracy,F1_Macro,Seen_Acc,Unseen_Acc,Samples\n")
                f.write("\n".join(lines))
            print(f"\nResults saved to {save_path}")


# ===========================================================================
# Main entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Persistence Spectrum CLIP CZSL Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file (default: use config from checkpoint)")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "zero_shot", "by_combination", "by_jnr"],
                        help="Evaluation mode: 'all'=zero_shot+by_combination, "
                             "'by_jnr'=per-JNR breakdown")
    parser.add_argument("--split", type=str, default="test",
                        help="Data split to evaluate on")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override prediction threshold")
    parser.add_argument("--output_dir", type=str, default="results/persistence",
                        help="Output directory for results and plots")
    parser.add_argument("--ablation_mode", type=str, default=None,
                        choices=["persistence", "stft", "fusion"],
                        help="Override ablation mode from checkpoint config")
    # Visualization flags (matching evaluate_czsl)
    parser.add_argument("--visualize", action="store_true",
                        help="Generate all visualizations (confusion, ROC, PR, t-SNE, UMAP)")
    parser.add_argument("--tsne", action="store_true",
                        help="Generate t-SNE visualization only")
    parser.add_argument("--umap", action="store_true",
                        help="Generate UMAP visualization only")
    parser.add_argument("--roc", action="store_true",
                        help="Generate ROC curves only")
    parser.add_argument("--pr", action="store_true",
                        help="Generate PR curves only")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config — default to config.yaml, checkpoint config as fallback
    default_config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    if args.config:
        config_path = args.config
    elif os.path.exists(default_config_path):
        config_path = default_config_path
    else:
        config = checkpoint.get('config', {})
        if not config:
            raise ValueError("No config found and --config not specified")
        config_path = None

    if config_path:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        # Merge ablation mode from checkpoint if present (allows training-time mode to carry over)
        if 'ablation' not in config:
            ckpt_ablation = checkpoint.get('config', {}).get('ablation', {})
            if ckpt_ablation:
                config['ablation'] = ckpt_ablation

    # Detect ablation mode (CLI override takes precedence)
    ablation_config = config.get('ablation', {})
    ablation_mode = args.ablation_mode or ablation_config.get('mode', 'persistence')
    print(f"Ablation mode: {ablation_mode}")

    # Mode-aware output directory — isolate results per mode
    output_dir = Path(args.output_dir)
    if ablation_mode != 'persistence' and args.output_dir == "results/persistence":
        # Auto-suffix default output dir with mode
        output_dir = Path(f"results/persistence_{ablation_mode}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mode-aware prefix for output files
    mode_prefix = "persistence" if ablation_mode == "persistence" else f"persistence_{ablation_mode}"

    # Class names
    jamming_classes = config.get('jamming_classes', [])
    class_names = [jc['name'] if isinstance(jc, dict) else jc for jc in jamming_classes]

    # Create model
    print("\nReconstructing model from checkpoint...")
    model = create_persistence_model(config, device=str(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # Cache text features
    czsl_config = config.get("czsl", {})
    _seen_raw = _unwrap_combinations(czsl_config.get("seen_combinations", None), 'seen_combinations')
    seen_comb_names = _seen_raw if _seen_raw else None
    model.cache_text_features(
        max_combination_size=2,
        include_single=True,
        seen_combinations=seen_comb_names,
    )

    evaluator = PersistenceEvaluator(model, config, device)

    # Load seen/unseen combinations (as indices)
    seen_comb_names = _unwrap_combinations(czsl_config.get("seen_combinations", []), 'seen_combinations')
    unseen_comb_names = _unwrap_combinations(czsl_config.get("unseen_combinations", []), 'unseen_combinations')
    seen_combinations = _convert_combination_names_to_indices(seen_comb_names, class_names)
    unseen_combinations = _convert_combination_names_to_indices(unseen_comb_names, class_names)

    # ==================================================================
    # Mode: all — zero_shot + by_combination (like evaluate_czsl)
    # ==================================================================
    if args.mode == "all":
        # ── 1) Zero-Shot ──
        print(f"\n{'='*60}")
        print(f"  [1/3] Zero-Shot Evaluation on {args.split}")
        print(f"{'='*60}")
        results_zs = evaluator.evaluate_zero_shot(
            config=config, split=args.split, ablation_mode=ablation_mode,
            threshold=args.threshold)
        evaluator.print_metrics(results_zs["metrics"])

        np.savez(str(output_dir / f"{mode_prefix}_zeroshot_{args.split}.npz"),
                 labels=results_zs["labels"],
                 predictions=results_zs["predictions"],
                 features=results_zs["features"])

        if args.visualize:
            evaluator.plot_confusion_by_combination(
                results_zs["labels"], results_zs["predictions"],
                save_path=str(output_dir / f"{mode_prefix}_confusion_zs_{args.split}.png"),
                seen_combinations=seen_combinations,
                unseen_combinations=unseen_combinations,
            )
        if args.visualize or args.roc:
            plot_roc_curves(
                results_zs["labels"], results_zs["probabilities"], class_names,
                save_dir=str(output_dir), prefix=f"{mode_prefix}_{args.split}_zs",
                mode_title=f"zero_shot ({args.split})",
            )
        if args.visualize or args.pr:
            plot_pr_curves(
                results_zs["labels"], results_zs["probabilities"], class_names,
                save_dir=str(output_dir), prefix=f"{mode_prefix}_{args.split}_zs",
                mode_title=f"zero_shot ({args.split})",
            )
        if args.tsne:
            print("\nGenerating t-SNE (zero-shot)...")
            evaluator.plot_feature_tsne(
                results_zs["features"], results_zs["labels"],
                save_path=str(output_dir / f"{mode_prefix}_tsne_{args.split}.png"),
                seen_combinations=seen_combinations,
                unseen_combinations=unseen_combinations,
            )
        if args.umap:
            print("\nGenerating UMAP (zero-shot)...")
            evaluator.plot_feature_umap(
                results_zs["features"], results_zs["labels"],
                save_path=str(output_dir / f"{mode_prefix}_umap_{args.split}.png"),
                seen_combinations=seen_combinations,
                unseen_combinations=unseen_combinations,
            )
        if args.visualize:
            evaluator.plot_label_cooccurrence(
                results_zs["labels"], results_zs["predictions"],
                save_path=str(output_dir / f"{mode_prefix}_cooccurrence_{args.split}.png"),
            )

        # ── 2) By-Combination ──
        print(f"\n{'='*60}")
        print(f"  [2/3] By-Combination Evaluation on {args.split}")
        print(f"{'='*60}")
        results_bc = evaluator.evaluate_by_combination(
            config=config, split=args.split, ablation_mode=ablation_mode,
            threshold=args.threshold)

        bc_labels = results_bc.get('_labels')
        bc_preds = results_bc.get('_predictions')
        bc_probs = results_bc.get('_probabilities')
        bc_feats = results_bc.get('_features')

        if bc_labels is not None:
            np.savez(str(output_dir / f"{mode_prefix}_bycombo_{args.split}.npz"),
                     labels=bc_labels, predictions=bc_preds, features=bc_feats)

            if args.visualize:
                evaluator.plot_confusion_by_combination(
                    bc_labels, bc_preds,
                    save_path=str(output_dir / f"{mode_prefix}_confusion_bc_{args.split}.png"),
                    seen_combinations=seen_combinations,
                    unseen_combinations=unseen_combinations,
                )
            if args.visualize or args.roc:
                plot_roc_curves(
                    bc_labels, bc_probs, class_names,
                    save_dir=str(output_dir), prefix=f"{mode_prefix}_{args.split}_byc",
                    mode_title=f"by_combination ({args.split})",
                )
            if args.visualize or args.pr:
                plot_pr_curves(
                    bc_labels, bc_probs, class_names,
                    save_dir=str(output_dir), prefix=f"{mode_prefix}_{args.split}_byc",
                    mode_title=f"by_combination ({args.split})",
                )

        # ── 3) By-JNR ──
        print(f"\n{'='*60}")
        print(f"  [3/3] By-JNR Evaluation on {args.split}")
        print(f"{'='*60}")
        results_jnr = evaluator.evaluate_by_jnr(
            config=config, split=args.split, ablation_mode=ablation_mode,
            threshold=args.threshold, output_dir=str(output_dir),
        )
        evaluator.print_jnr_results(
            results_jnr,
            save_path=str(output_dir / "by_jnr" / f"{mode_prefix}_jnr_results_{args.split}.csv"),
        )
        # Per-JNR ROC / PR curves
        do_jnr_curves = args.visualize or args.roc or args.pr
        if do_jnr_curves:
            jnr_viz_dir = str(output_dir / "by_jnr")
            for jnr_val, jnr_result in sorted(results_jnr.items()):
                if "probabilities" in jnr_result and jnr_result["probabilities"].size > 0:
                    prefix = f"{mode_prefix}_jnr_{jnr_val:+.0f}_{args.split}"
                    if args.visualize or args.roc:
                        plot_roc_curves(
                            jnr_result["labels"], jnr_result["probabilities"],
                            class_names, save_dir=jnr_viz_dir, prefix=prefix,
                            mode_title=f"JNR={jnr_val:+d}",
                        )
                    if args.visualize or args.pr:
                        plot_pr_curves(
                            jnr_result["labels"], jnr_result["probabilities"],
                            class_names, save_dir=jnr_viz_dir, prefix=prefix,
                            mode_title=f"JNR={jnr_val:+d}",
                        )

    # ==================================================================
    # Mode: zero_shot
    # ==================================================================
    elif args.mode == "zero_shot":
        print(f"\n{'='*60}")
        print(f"Zero-Shot Evaluation on {args.split} split")
        print(f"{'='*60}")

        results = evaluator.evaluate_zero_shot(
            config=config, split=args.split, ablation_mode=ablation_mode,
            threshold=args.threshold)
        evaluator.print_metrics(results["metrics"])

        np.savez(str(output_dir / f"{mode_prefix}_zeroshot_{args.split}.npz"),
                 labels=results["labels"],
                 predictions=results["predictions"],
                 features=results["features"])

        if args.visualize:
            evaluator.plot_confusion_by_combination(
                results["labels"], results["predictions"],
                save_path=str(output_dir / f"{mode_prefix}_confusion_zs_{args.split}.png"),
                seen_combinations=seen_combinations,
                unseen_combinations=unseen_combinations,
            )
        if args.visualize or args.roc:
            plot_roc_curves(
                results["labels"], results["probabilities"], class_names,
                save_dir=str(output_dir), prefix=f"{mode_prefix}_{args.split}_zs",
                mode_title=f"zero_shot ({args.split})",
            )
        if args.visualize or args.pr:
            plot_pr_curves(
                results["labels"], results["probabilities"], class_names,
                save_dir=str(output_dir), prefix=f"{mode_prefix}_{args.split}_zs",
                mode_title=f"zero_shot ({args.split})",
            )
        if args.tsne:
            print("\nGenerating t-SNE...")
            evaluator.plot_feature_tsne(
                results["features"], results["labels"],
                save_path=str(output_dir / f"{mode_prefix}_tsne_{args.split}.png"),
                seen_combinations=seen_combinations,
                unseen_combinations=unseen_combinations,
            )
        if args.umap:
            print("\nGenerating UMAP...")
            evaluator.plot_feature_umap(
                results["features"], results["labels"],
                save_path=str(output_dir / f"{mode_prefix}_umap_{args.split}.png"),
                seen_combinations=seen_combinations,
                unseen_combinations=unseen_combinations,
            )
        if args.visualize:
            evaluator.plot_label_cooccurrence(
                results["labels"], results["predictions"],
                save_path=str(output_dir / f"{mode_prefix}_cooccurrence_{args.split}.png"),
            )

    # ==================================================================
    # Mode: by_combination
    # ==================================================================
    elif args.mode == "by_combination":
        print(f"\n{'='*60}")
        print(f"By-Combination Evaluation on {args.split} split")
        print(f"{'='*60}")

        results = evaluator.evaluate_by_combination(
            config=config, split=args.split, ablation_mode=ablation_mode,
            threshold=args.threshold)

        bc_labels = results.get('_labels')
        bc_preds = results.get('_predictions')
        bc_probs = results.get('_probabilities')
        bc_feats = results.get('_features')

        if bc_labels is not None:
            np.savez(str(output_dir / f"{mode_prefix}_bycombo_{args.split}.npz"),
                     labels=bc_labels, predictions=bc_preds, features=bc_feats)

            if args.visualize:
                evaluator.plot_confusion_by_combination(
                    bc_labels, bc_preds,
                    save_path=str(output_dir / f"{mode_prefix}_confusion_bc_{args.split}.png"),
                    seen_combinations=seen_combinations,
                    unseen_combinations=unseen_combinations,
                )
            if args.visualize or args.roc:
                plot_roc_curves(
                    bc_labels, bc_probs, class_names,
                    save_dir=str(output_dir), prefix=f"{mode_prefix}_{args.split}_byc",
                    mode_title=f"by_combination ({args.split})",
                )
            if args.visualize or args.pr:
                plot_pr_curves(
                    bc_labels, bc_probs, class_names,
                    save_dir=str(output_dir), prefix=f"{mode_prefix}_{args.split}_byc",
                    mode_title=f"by_combination ({args.split})",
                )

    # ==================================================================
    # Mode: by_jnr
    # ==================================================================
    elif args.mode == "by_jnr":
        print(f"\n{'='*60}")
        print(f"By-JNR Evaluation on {args.split} split")
        print(f"{'='*60}")

        results = evaluator.evaluate_by_jnr(
            config=config,
            split=args.split,
            ablation_mode=ablation_mode,
            threshold=args.threshold,
            output_dir=str(output_dir),
        )

        if not results:
            print("No JNR data found!")
            return

        evaluator.print_jnr_results(
            results,
            save_path=str(output_dir / "by_jnr" / f"{mode_prefix}_jnr_results_{args.split}.csv"),
        )

        # Per-JNR ROC / PR curves
        do_jnr_curves = args.visualize or args.roc or args.pr
        if do_jnr_curves:
            jnr_viz_dir = str(output_dir / "by_jnr")
            for jnr_val, jnr_result in sorted(results.items()):
                if "probabilities" in jnr_result and jnr_result["probabilities"].size > 0:
                    prefix = f"{mode_prefix}_jnr_{jnr_val:+.0f}_{args.split}"
                    if args.visualize or args.roc:
                        plot_roc_curves(
                            jnr_result["labels"],
                            jnr_result["probabilities"],
                            class_names,
                            save_dir=jnr_viz_dir,
                            prefix=prefix,
                            mode_title=f"JNR={jnr_val:+d}",
                        )
                    if args.visualize or args.pr:
                        plot_pr_curves(
                            jnr_result["labels"],
                            jnr_result["probabilities"],
                            class_names,
                            save_dir=jnr_viz_dir,
                            prefix=prefix,
                            mode_title=f"JNR={jnr_val:+d}",
                        )

    print(f"\nEvaluation completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
