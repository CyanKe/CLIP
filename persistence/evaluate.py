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
from multi.metrics_czsl import (
    compute_metrics_bundle,
    compute_subset_bundles,
    print_metrics_report,
    print_jnr_metrics_table,
    save_metrics_json,
    build_seen_unseen_sets,
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
    def _compute_single_class_probs(self, images: torch.Tensor) -> tuple:
        """Compute single-class softmax probabilities for ROC/PR curves.

        Matches multi/evaluate_czsl.py zero_shot: probs from single-class
        text features only (not combination-space).
        """
        single_features = self.model.get_cached_text_features()
        if single_features is None:
            raise RuntimeError(
                "Text features cache is empty. Call cache_text_features() first."
            )
        single_features = F.normalize(single_features, dim=-1)

        image_features = self.model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)

        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * (image_features @ single_features.T)
        probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy(), image_features.cpu().numpy()

    # ------------------------------------------------------------------
    # Single-class prediction (single softmax + topk) — matching conformer
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_batch(self, images: torch.Tensor, threshold: float = None):
        """Single-class softmax + top-k(3) > threshold prediction.

        Uses only single-class text features (not combination-space).
        Matches conformer_1d/evaluate_conformer.py _predict_batch pattern.

        Returns:
            probs: (B, num_classes) softmax probabilities
            preds: (B, num_classes) multi-hot predictions
        """
        num_classes = self.num_classes
        if threshold is None:
            threshold = (self.eval_config.get('by_combination_threshold')
                         or self.eval_config.get('threshold')
                         or 1.0 / num_classes)

        # Use single-class text features only
        single_features = self.model.get_cached_text_features()
        single_features = F.normalize(single_features, dim=-1)

        image_features = self.model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)

        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * (image_features @ single_features.T)

        probs = torch.softmax(logits, dim=-1)
        batch_size = images.size(0)
        top_k = 3
        topk_values, topk_indices = torch.topk(probs, k=top_k, dim=-1)

        preds = torch.zeros(batch_size, num_classes, device=images.device)
        for b in range(batch_size):
            for j, idx in enumerate(topk_indices[b]):
                if topk_values[b, j] > threshold:
                    preds[b, idx] = 1.0

        return probs.cpu(), preds.cpu()

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
    # Zero-shot evaluation — delegates to unified per-JNR method
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_zero_shot(
        self,
        config: dict = None,
        split: str = 'test',
        ablation_mode: str = 'persistence',
        dataloader=None,
        threshold: float = None,
        print_samples: bool = False,
    ) -> dict:
        """Zero-shot eval: combo-space top-1 (multi-style) via JNR lazy load.

        Prediction matches multi/evaluate_czsl.py:
          zero_shot_predict(use_combinations=True, top_k=1) → decode "A+B"
        ROC/PR probabilities use single-class softmax.
        Data is loaded one JNR at a time then concatenated globally.
        """
        self.model.eval()

        # Backward compat: pre-built dataloader
        if dataloader is not None:
            raw = self._evaluate_zero_shot_on_loader(dataloader, print_samples=print_samples)
        else:
            # JNR lazy-load → global concat (same as multi ChainedLoaderIterable intent)
            result = self._evaluate_unified_per_jnr(
                config, split, ablation_mode,
                collect_per_jnr=False,
                print_samples=print_samples,
                predict_mode="combo_top1",
            )
            raw = result['global']

        seen_set, unseen_set = build_seen_unseen_sets(
            self.czsl_config.get('seen_combinations', []),
            self.czsl_config.get('unseen_combinations', []),
            self.class_names,
        )
        metrics = compute_metrics_bundle(
            raw['labels'], raw['predictions'], self.class_names,
            y_prob=raw.get('probabilities'),
            seen_set=seen_set, unseen_set=unseen_set,
            dual_write=True,
        )

        return {
            "metrics": metrics,
            "labels": raw['labels'],
            "predictions": raw['predictions'],
            "probabilities": raw['probabilities'],
            "features": raw['features'],
        }

    def _decode_combo_top1_preds(
        self, images: torch.Tensor,
    ) -> tuple:
        """Multi-style combo-space top-1 → multi-hot + single-class probs + features.

        Returns:
            preds (B, C) np.float32, probs (B, C) np, features (B, D) np
        """
        probs_single, img_feats = self._compute_single_class_probs(images)
        _, _, pred_names = self.model.zero_shot_predict(
            images, use_combinations=True, top_k=1,
        )
        batch_size = images.size(0)
        preds = np.zeros((batch_size, self.num_classes), dtype=np.float32)
        for i, names in enumerate(pred_names):
            if not names:
                continue
            comb_name = names[0]
            for part in comb_name.split('+'):
                part = part.strip()
                if part in self.class_names:
                    preds[i, self.class_names.index(part)] = 1.0
        return preds, probs_single, img_feats

    def _evaluate_zero_shot_on_loader(
        self, dataloader, print_samples: bool = False,
    ) -> dict:
        """Internal: multi-style zero-shot on a single DataLoader."""
        all_preds = []
        all_labels = []
        all_probs = []
        all_features = []
        sample_idx = 0

        for batch_data in tqdm(dataloader, desc="Zero-shot eval", leave=False):
            if len(batch_data) >= 6:
                images, _, text_tokens, labels, texts, metas = batch_data[:6]
            else:
                images, text_tokens, labels, texts, metas = batch_data

            images = images.to(self.device)
            labels_np = labels.cpu().numpy()

            preds, probs_single, img_feats = self._decode_combo_top1_preds(images)

            if print_samples:
                for i in range(images.size(0)):
                    true_names = [self.class_names[j] for j in range(self.num_classes)
                                  if labels_np[i, j] > 0]
                    pred_names = [self.class_names[j] for j in range(self.num_classes)
                                  if preds[i, j] > 0]
                    correct = sorted(true_names) == sorted(pred_names)
                    print(f"  [SAMPLE {sample_idx}] true={'+'.join(true_names):30s} | "
                          f"pred={'+'.join(pred_names):30s} | correct={correct}", flush=True)
                    sample_idx += 1

            all_preds.append(preds)
            all_probs.append(probs_single)
            all_features.append(img_feats)
            all_labels.append(labels_np)

        return {
            "predictions": np.concatenate(all_preds, axis=0) if all_preds else np.array([]),
            "labels": np.concatenate(all_labels, axis=0) if all_labels else np.array([]),
            "probabilities": np.concatenate(all_probs, axis=0) if all_probs else np.array([]),
            "features": np.concatenate(all_features, axis=0) if all_features else np.array([]),
        }

    # ------------------------------------------------------------------
    # Multilabel metrics
    # ------------------------------------------------------------------

    def _compute_multilabel_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray = None,
    ) -> dict:
        """Compute multilabel metrics via unified MetricsBundle."""
        seen_set, unseen_set = build_seen_unseen_sets(
            self.czsl_config.get('seen_combinations', []),
            self.czsl_config.get('unseen_combinations', []),
            self.class_names,
        )
        return compute_metrics_bundle(
            y_true, y_pred, self.class_names,
            y_prob=y_prob, seen_set=seen_set, unseen_set=unseen_set,
            dual_write=True,
        )

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
        print_samples: bool = False,
    ) -> dict:
        """Evaluate broken down by seen and unseen combinations — per-JNR lazy loading.

        Accepts either a pre-built dataloader (backward compat) or
        config+split+ablation_mode for on-demand per-JNR loading.

        Threshold priority: CLI > evaluation.by_combination_threshold > evaluation.threshold > 1/num_classes
        """

        # Backward compat: pre-built dataloader
        if dataloader is not None:
            return self._evaluate_by_combination_on_loader(dataloader, threshold,
                                                           print_samples=print_samples)

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
            result = self._evaluate_by_combination_on_loader(loader, threshold,
                                                           print_samples=print_samples)
            all_results.append(result)
            del loader

        # Merge results across JNRs
        if not all_results:
            return {'_labels': np.array([]), '_predictions': np.array([]),
                    '_probabilities': np.array([]), '_features': np.array([])}

        # Merge all predictions, then global + seen/unseen subset bundles
        all_preds = np.concatenate([r['_predictions'] for r in all_results], axis=0)
        all_labels = np.concatenate([r['_labels'] for r in all_results], axis=0)
        all_probs = np.concatenate([r['_probabilities'] for r in all_results], axis=0)
        all_features = np.concatenate([r['_features'] for r in all_results], axis=0)

        seen_set, unseen_set = build_seen_unseen_sets(
            self.czsl_config.get('seen_combinations', []),
            self.czsl_config.get('unseen_combinations', []),
            self.class_names,
        )
        bundles = compute_subset_bundles(
            all_labels, all_preds, self.class_names,
            seen_set=seen_set, unseen_set=unseen_set,
            y_prob=all_probs, dual_write=True,
        )
        merged = {
            "global": bundles["global"],
            "seen": bundles["seen"],
            "unseen": bundles["unseen"],
            # flat = global for backward compat
            **bundles["global"],
            "_labels": all_labels,
            "_predictions": all_preds,
            "_probabilities": all_probs,
            "_features": all_features,
        }
        print_metrics_report(bundles["global"], title="By-Combination — global")
        if bundles["seen"].get("num_samples", 0):
            print_metrics_report(bundles["seen"], title="By-Combination — seen subset")
        if bundles["unseen"].get("num_samples", 0):
            print_metrics_report(bundles["unseen"], title="By-Combination — unseen subset")

        return merged

    def _evaluate_by_combination_on_loader(self, dataloader, threshold: float,
                                             print_samples: bool = False) -> dict:
        """Internal: run by-combination eval on a single DataLoader.

        Uses single-class softmax + topk(3) > threshold (_predict_batch),
        matching conformer_1d's by_combination pattern.
        """
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

            # _predict_batch returns (probs, preds) — single-class softmax + topk
            probs, preds_batch = self._predict_batch(images, threshold)
            all_probs_list.append(probs.numpy())

            # Image features for visualization (separate lightweight forward pass)
            img_feats = self.model.encode_image(images)
            img_feats = F.normalize(img_feats, dim=-1).cpu().numpy()
            all_features_list.append(img_feats)

            for i in range(images.size(0)):
                pred_vec = preds_batch[i].numpy()
                all_preds_list.append(pred_vec)
                all_labels_list.append(labels_np[i])

                true_classes = tuple(sorted([
                    self.class_names[j] for j in range(self.num_classes)
                    if labels_np[i, j] > 0
                ]))
                pred_classes = tuple(sorted([
                    self.class_names[j] for j in range(self.num_classes)
                    if pred_vec[j] > 0
                ]))

                # ---- Per-sample debug print ----
                if print_samples:
                    correct = (true_classes == pred_classes)
                    print(f"  [SAMPLE {len(all_labels_list)-1}] "
                          f"true={'+'.join(true_classes):30s} | "
                          f"pred={'+'.join(pred_classes):30s} | correct={correct}", flush=True)

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
    # Unified per-JNR evaluation (zero_shot + by_jnr combined)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate_unified_per_jnr(
        self,
        config: dict,
        split: str = 'test',
        ablation_mode: str = 'persistence',
        threshold: float = None,
        output_dir: str = None,
        collect_per_jnr: bool = False,
        print_samples: bool = False,
        predict_mode: str = "combo_top1",
    ) -> dict:
        """Unified evaluation: load each JNR once, accumulate global + per-JNR stats.

        Prediction (default ``combo_top1``, multi-aligned):
          zero_shot_predict(use_combinations=True, top_k=1) + single-class probs

        Args:
            collect_per_jnr: if True, compute per-JNR metrics dict
            output_dir: if set, generate per-JNR confusion matrices
            predict_mode: ``combo_top1`` (default, multi-style) | reserved for future

        Returns:
            dict with keys:
                'global': {labels, predictions, probabilities, features} — concatenated
                'per_jnr': {jnr: metrics_dict, ...} — only if collect_per_jnr=True
        """
        del threshold  # multi-style top-1 does not use threshold
        del predict_mode  # currently only combo_top1

        data_config = config.get('data', {})
        jnr_start = data_config.get('jnr_start', 0)
        jnr_end = data_config.get('jnr_end', 20)
        jnr_step = data_config.get('jnr_step', 1)
        jnr_levels = list(range(jnr_start, jnr_end + 1, jnr_step))

        # Seen/unseen sets for per-JNR breakdown
        seen_combos_raw = _unwrap_combinations(
            self.czsl_config.get('seen_combinations', []), 'seen_combinations')
        unseen_combos_raw = _unwrap_combinations(
            self.czsl_config.get('unseen_combinations', []), 'unseen_combinations')

        # Convert combination names → class indices for fast matching
        seen_set = set()
        for sc in seen_combos_raw:
            seen_set.add(tuple(sorted(self.class_names.index(c) for c in sc if c in self.class_names)))
        unseen_set = set()
        for uc in unseen_combos_raw:
            unseen_set.add(tuple(sorted(self.class_names.index(c) for c in uc if c in self.class_names)))

        self.model.eval()

        # Global accumulation
        all_labels = []
        all_preds = []
        all_probs = []
        all_features = []

        # Per-JNR accumulation
        jnr_results = {}
        all_jnr_metrics = []
        global_sample_idx = 0

        for jnr in jnr_levels:
            loader = self._build_single_jnr_loader(config, split, jnr, ablation_mode)
            if loader is None:
                print(f"  Skipping JNR=+{jnr}: data not found")
                continue

            # Per-JNR accumulators
            jnr_labels = []
            jnr_preds = []
            jnr_probs = []
            jnr_feats = []
            class_stats = np.zeros((self.num_classes, 4), dtype=np.int64)
            seen_correct, seen_total = 0, 0
            unseen_correct, unseen_total = 0, 0

            for batch_data in tqdm(loader, desc=f"JNR=+{jnr}", leave=False):
                if len(batch_data) >= 6:
                    images, _, text_tokens, labels, texts, metas = batch_data[:6]
                else:
                    images, text_tokens, labels, texts, metas = batch_data

                images = images.to(self.device)
                labels_np = labels.cpu().numpy()
                batch_size = images.size(0)

                # Multi-style: combo top-1 preds + single-class softmax for ROC/PR
                preds, probs_single, img_feats = self._decode_combo_top1_preds(images)

                # ---- Per-sample debug print ----
                if print_samples:
                    for i in range(batch_size):
                        true_names = [self.class_names[j] for j in range(self.num_classes)
                                      if labels_np[i, j] > 0]
                        pred_names = [self.class_names[j] for j in range(self.num_classes)
                                      if preds[i, j] > 0]
                        correct = (sorted(true_names) == sorted(pred_names))
                        print(f"  [SAMPLE {global_sample_idx}] true={'+'.join(true_names):30s} | "
                              f"pred={'+'.join(pred_names):30s} | correct={correct}", flush=True)
                        global_sample_idx += 1

                # Accumulate global
                all_labels.append(labels_np)
                all_preds.append(preds)
                all_probs.append(probs_single)
                all_features.append(img_feats)

                # Accumulate per-JNR
                jnr_labels.append(labels_np)
                jnr_preds.append(preds)
                jnr_probs.append(probs_single)
                jnr_feats.append(img_feats)

                # Per-class TP/FP/FN/TN
                for c in range(self.num_classes):
                    true_c = labels_np[:, c]
                    pred_c = preds[:, c]
                    class_stats[c, 0] += np.sum((true_c == 1) & (pred_c == 1))  # TP
                    class_stats[c, 1] += np.sum((true_c == 0) & (pred_c == 1))  # FP
                    class_stats[c, 2] += np.sum((true_c == 1) & (pred_c == 0))  # FN
                    class_stats[c, 3] += np.sum((true_c == 0) & (pred_c == 0))  # TN

                # Seen/unseen accuracy
                for i in range(batch_size):
                    true_comb = tuple(sorted(np.where(labels_np[i] == 1)[0].tolist()))
                    pred_comb = tuple(sorted(np.where(preds[i] == 1)[0].tolist()))
                    is_correct = (true_comb == pred_comb)
                    if true_comb in seen_set:
                        seen_total += 1
                        if is_correct:
                            seen_correct += 1
                    elif true_comb in unseen_set:
                        unseen_total += 1
                        if is_correct:
                            unseen_correct += 1

            # ---- Per-JNR metrics ----
            jnr_labels_np = np.concatenate(jnr_labels, axis=0)
            jnr_preds_np = np.concatenate(jnr_preds, axis=0)
            jnr_probs_np = np.concatenate(jnr_probs, axis=0)
            jnr_feats_np = np.concatenate(jnr_feats, axis=0)

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

            bundle = compute_metrics_bundle(
                jnr_labels_np, jnr_preds_np, self.class_names,
                y_prob=jnr_probs_np,
                seen_set=seen_set, unseen_set=unseen_set,
                dual_write=True,
            )
            jnr_metrics = {
                **bundle,
                'jnr': jnr,
                'total_samples': bundle['num_samples'],  # dual-write legacy
                'per_class_recall': per_class_recall,
                'per_class_precision': per_class_precision,
                'per_class_f1': per_class_f1,
                'labels': jnr_labels_np,
                'probabilities': jnr_probs_np,
                'features': jnr_feats_np,
            }
            _ = (seen_correct, unseen_correct, seen_total, unseen_total)
            if collect_per_jnr:
                jnr_results[jnr] = jnr_metrics
                all_jnr_metrics.append(jnr_metrics)

            # Per-JNR confusion matrix
            if output_dir:
                jnr_dir = os.path.join(output_dir, "by_jnr")
                os.makedirs(jnr_dir, exist_ok=True)
                seen_idx = _convert_combination_names_to_indices(seen_combos_raw, self.class_names)
                unseen_idx = _convert_combination_names_to_indices(unseen_combos_raw, self.class_names)
                prefix = "persistence" if ablation_mode == "persistence" else f"persistence_{ablation_mode}"
                self.plot_confusion_by_combination(
                    jnr_labels_np, jnr_preds_np,
                    save_path=os.path.join(jnr_dir, f"{prefix}_confusion_jnr_{jnr:+.0f}.png"),
                    seen_combinations=seen_idx,
                    unseen_combinations=unseen_idx,
                )

            # Free this JNR's data
            del loader

        # ---- Aggregate global ----
        global_data = {}
        if all_labels:
            global_data['labels'] = np.concatenate(all_labels, axis=0)
            global_data['predictions'] = np.concatenate(all_preds, axis=0)
            global_data['probabilities'] = np.concatenate(all_probs, axis=0)
            global_data['features'] = np.concatenate(all_features, axis=0)
        else:
            global_data = {'labels': np.array([]), 'predictions': np.array([]),
                           'probabilities': np.array([]), 'features': np.array([])}

        # JNR metrics panel plot (after all JNRs)
        if output_dir and all_jnr_metrics:
            jnr_dir = os.path.join(output_dir, "by_jnr")
            os.makedirs(jnr_dir, exist_ok=True)
            prefix = "persistence" if ablation_mode == "persistence" else f"persistence_{ablation_mode}"
            self.plot_jnr_metrics(jnr_results, save_path=os.path.join(jnr_dir, f'{prefix}_jnr_metrics.png'))

        return {
            'global': global_data,
            'per_jnr': jnr_results if collect_per_jnr else {},
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
        print_samples: bool = False,
    ) -> dict:
        """Evaluate separately at each JNR level — delegates to _evaluate_unified_per_jnr.

        Args:
            config: full YAML config dict
            split: data split to evaluate
            ablation_mode: 'persistence', 'stft', or 'fusion'
            threshold: prediction threshold override
            output_dir: directory for JNR metrics panel + per-JNR confusion matrices

        Returns:
            dict: {jnr: metrics, ...}
        """
        result = self._evaluate_unified_per_jnr(
            config, split, ablation_mode,
            output_dir=output_dir,
            collect_per_jnr=True,
            predict_mode="combo_top1",
            print_samples=print_samples,
        )

        jnr_results = result['per_jnr']

        # Print per-JNR summary
        for jnr, metrics in sorted(jnr_results.items()):
            print(f"  JNR=+{jnr}: F1_macro={metrics['f1_macro']:.4f}, "
                  f"F1_micro={metrics['f1_micro']:.4f}")

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
            n = m.get('num_samples', m.get('total_samples', 0))
            acc = m.get('subset_accuracy', m.get('combination_accuracy', 0.0))
            if isinstance(acc, (int, float)) and acc > 1.0 and n:
                acc = acc / n
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
                  f"(Seen: {len(seen_combinations or [])}, Unseen: {len(unseen_combinations or [])})",
                  flush=True)
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
            print(f"Confusion matrix saved to {save_path}", flush=True)
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

    def print_metrics(self, metrics: dict, title: str = "Evaluation Results"):
        """Pretty-print evaluation metrics (unified MetricsBundle)."""
        if isinstance(metrics.get("global"), dict) and "subset_accuracy" in metrics.get("global", {}):
            print_metrics_report(metrics["global"], title=f"{title} — global")
            if metrics.get("seen", {}).get("num_samples", 0):
                print_metrics_report(metrics["seen"], title=f"{title} — seen subset")
            if metrics.get("unseen", {}).get("num_samples", 0):
                print_metrics_report(metrics["unseen"], title=f"{title} — unseen subset")
            return
        print_metrics_report(metrics, title=title)

    def print_jnr_results(self, results: dict, save_path: str = None):
        """Print and optionally save per-JNR results table (rates only; no global)."""
        print_jnr_metrics_table(results)

        lines = []
        for jnr, m in sorted(results.items()):
            n = m.get('num_samples', m.get('total_samples', 0))
            acc = m.get('subset_accuracy', m.get('combination_accuracy', 0.0))
            if isinstance(acc, (int, float)) and acc > 1.0 and n:
                acc = acc / n
            lines.append(
                f"{jnr},{acc:.4f},{m.get('f1_macro', 0):.4f},"
                f"{m.get('seen_accuracy', 0):.4f},"
                f"{m.get('unseen_accuracy', 0):.4f},"
                f"{m.get('harmonic_mean', 0):.4f},"
                f"{n}"
            )

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("JNR,SubsetAcc,F1_Macro,Seen_Acc,Unseen_Acc,HM,Samples\n")
                f.write("\n".join(lines))
            print(f"\nResults saved to {save_path}")


# ===========================================================================
# Main entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Persistence Spectrum CLIP CZSL Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt) — defaults to config evaluation.checkpoint")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file (default: config.yaml)")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["all", "zero_shot", "by_combination", "by_jnr"],
                        help="Evaluation mode (overrides config)")
    parser.add_argument("--split", type=str, default=None,
                        choices=["train", "val", "test"],
                        help="Data split (overrides config)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override prediction threshold")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (overrides config)")
    parser.add_argument("--ablation_mode", type=str, default=None,
                        choices=["persistence", "stft", "fusion"],
                        help="Override ablation mode")
    # Visualization flags
    parser.add_argument("--visualize", action="store_true", default=None,
                        help="Generate all visualizations")
    parser.add_argument("--tsne", action="store_true", default=None,
                        help="Generate t-SNE visualization only")
    parser.add_argument("--umap", action="store_true", default=None,
                        help="Generate UMAP visualization only")
    parser.add_argument("--roc", action="store_true", default=None,
                        help="Generate ROC curves only")
    parser.add_argument("--pr", action="store_true", default=None,
                        help="Generate PR curves only")
    parser.add_argument("--print_samples", action="store_true", default=None,
                        help="Print true and predicted labels for every sample")
    parser.add_argument("--combo_only", action="store_true", default=None,
                        help="Use only combination texts (no single-class) for zero_shot cache")
    args = parser.parse_args()

    # ── Load config ──
    default_config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config_path = args.config or default_config_path
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from: {config_path}")
    else:
        config = {}
        print("No config file found, using CLI args only")

    # ── Resolve checkpoint path: CLI > config > error ──
    eval_cfg = config.get('evaluation', {})
    checkpoint_path = args.checkpoint or eval_cfg.get('checkpoint')
    if not checkpoint_path:
        raise ValueError(
            "No checkpoint specified. Set --checkpoint or add evaluation.checkpoint to config.yaml"
        )
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Merge ablation mode from checkpoint if config doesn't have it ──
    if 'ablation' not in config:
        ckpt_ablation = checkpoint.get('config', {}).get('ablation', {})
        if ckpt_ablation:
            config['ablation'] = ckpt_ablation

    # ── Resolve all params: CLI > config > default ──
    ablation_cfg = config.get('ablation', {})
    ablation_mode = args.ablation_mode or ablation_cfg.get('mode', 'persistence')

    mode = args.mode or eval_cfg.get('mode', 'all')
    split = args.split or eval_cfg.get('split', 'test')
    threshold = args.threshold if args.threshold is not None else eval_cfg.get('threshold')
    print_samples = args.print_samples if args.print_samples is not None else False
    combo_only = args.combo_only if args.combo_only is not None else eval_cfg.get('combo_only', False)
    output_dir_str = args.output_dir or eval_cfg.get('output_dir', 'results/persistence')

    # Visualization flags: CLI --flag (explicitly set) > config > false
    def _viz_flag(cli_val, config_key):
        if cli_val is not None:
            return cli_val
        return eval_cfg.get(config_key, False)
    do_viz = _viz_flag(args.visualize, 'visualize')
    do_tsne = _viz_flag(args.tsne, 'tsne')
    do_umap = _viz_flag(args.umap, 'umap')
    do_roc = _viz_flag(args.roc, 'roc')
    do_pr = _viz_flag(args.pr, 'pr')

    print(f"Ablation mode: {ablation_mode}")
    print(f"Mode: {mode}, Split: {split}, Threshold: {threshold}")

    # ── Mode-aware output directory ──
    output_dir = Path(output_dir_str)
    if ablation_mode != 'persistence' and output_dir_str == "results/persistence":
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
    seen_comb_names = _seen_raw if _seen_raw else []
    _unseen_raw = _unwrap_combinations(czsl_config.get("unseen_combinations", None), 'unseen_combinations')
    unseen_comb_names = _unseen_raw if _unseen_raw else []
    # Merge seen + unseen → the full set of class/combinations we want in the cache
    # e.g. 17 single-class + 72 unseen combos = 89 text entries
    all_comb_names = seen_comb_names + unseen_comb_names
    print(f"CZSL text cache: {len(seen_comb_names)} seen + {len(unseen_comb_names)} unseen "
          f"= {len(all_comb_names)} total class/combinations")

    model.cache_text_features(
        max_combination_size=2,
        include_single=not combo_only,
        # combo_only: cache ALL 2-class combos indiscriminately — for testing
        # whether the model can recognize single jamming via combo texts only.
        seen_combinations=None if combo_only else (all_comb_names if all_comb_names else None),
    )

    if combo_only:
        print("⚠ combo_only=True: using only combination texts (no single-class) for zero_shot cache")
        if mode in ("by_combination", "all"):
            print("⚠ WARNING: by_combination/all modes use single-class features for _predict_batch; "
                  "results may be incorrect with combo_only")

    evaluator = PersistenceEvaluator(model, config, device)

    # Load seen/unseen combinations (as indices)
    seen_comb_names = _unwrap_combinations(czsl_config.get("seen_combinations", []), 'seen_combinations')
    unseen_comb_names = _unwrap_combinations(czsl_config.get("unseen_combinations", []), 'unseen_combinations')
    seen_combinations = _convert_combination_names_to_indices(seen_comb_names, class_names)
    unseen_combinations = _convert_combination_names_to_indices(unseen_comb_names, class_names)

    # ==================================================================
    # Mode: all — unified (zero_shot + by_jnr) + by_combination
    # ==================================================================
    if mode == "all":
        # ── 1) Unified pass: zero_shot + by_jnr simultaneously ──
        print(f"\n{'='*60}")
        print(f"  [1/2] Unified Evaluation (zero_shot + by_jnr) on {split}")
        print(f"{'='*60}")
        unified = evaluator._evaluate_unified_per_jnr(
            config=config, split=split, ablation_mode=ablation_mode,
            output_dir=str(output_dir),
            collect_per_jnr=True,
            print_samples=print_samples,
            predict_mode="combo_top1",
        )
        global_data = unified['global']
        jnr_results = unified['per_jnr']

        # Zero-shot metrics from global accumulation (separate JSON from by_jnr)
        zs_metrics = evaluator._compute_multilabel_metrics(
            global_data['labels'], global_data['predictions'],
            y_prob=global_data.get('probabilities'),
        )
        evaluator.print_metrics(zs_metrics, title="Zero-Shot Evaluation Results")
        save_metrics_json(
            zs_metrics, output_dir,
            filename=f"metrics_zero_shot_{split}.json",
            mode="zero_shot", split=split,
            extra_meta={"checkpoint": str(checkpoint_path), "ablation_mode": ablation_mode},
        )

        np.savez(str(output_dir / f"{mode_prefix}_zeroshot_{split}.npz"),
                 labels=global_data["labels"],
                 predictions=global_data["predictions"],
                 features=global_data["features"])

        print(f"\n[DEBUG ZS] Calling plot_confusion_by_combination: labels={global_data['labels'].shape}, preds={global_data['predictions'].shape}", flush=True)
        # Clear any accumulated matplotlib state from per-JNR plots
        import matplotlib.pyplot as _plt
        _plt.close('all')
        try:
            evaluator.plot_confusion_by_combination(
                global_data["labels"], global_data["predictions"],
                save_path=str(output_dir / f"{mode_prefix}_confusion_zs_{split}.png"),
                seen_combinations=seen_combinations,
                unseen_combinations=unseen_combinations,
            )
            print(f"[DEBUG ZS] plot_confusion_by_combination OK", flush=True)
        except Exception as e:
            print(f"[DEBUG ZS] plot_confusion_by_combination FAILED: {e}", flush=True)
            import traceback
            traceback.print_exc()
        if do_viz or do_roc:
            plot_roc_curves(
                global_data["labels"], global_data["probabilities"], class_names,
                save_dir=str(output_dir), prefix=f"{mode_prefix}_{split}_zs",
                mode_title=f"zero_shot ({split})",
            )
        if do_viz or do_pr:
            plot_pr_curves(
                global_data["labels"], global_data["probabilities"], class_names,
                save_dir=str(output_dir), prefix=f"{mode_prefix}_{split}_zs",
                mode_title=f"zero_shot ({split})",
            )
        if do_tsne:
            print("\nGenerating t-SNE (zero-shot)...")
            evaluator.plot_feature_tsne(
                global_data["features"], global_data["labels"],
                save_path=str(output_dir / f"{mode_prefix}_tsne_{split}.png"),
                seen_combinations=seen_combinations,
                unseen_combinations=unseen_combinations,
            )
        if do_umap:
            print("\nGenerating UMAP (zero-shot)...")
            evaluator.plot_feature_umap(
                global_data["features"], global_data["labels"],
                save_path=str(output_dir / f"{mode_prefix}_umap_{split}.png"),
                seen_combinations=seen_combinations,
                unseen_combinations=unseen_combinations,
            )
        if do_viz:
            evaluator.plot_label_cooccurrence(
                global_data["labels"], global_data["predictions"],
                save_path=str(output_dir / f"{mode_prefix}_cooccurrence_{split}.png"),
            )

        # By-JNR output only (no global mixed into by_jnr JSON — decision E)
        evaluator.print_jnr_results(
            jnr_results,
            save_path=str(output_dir / "by_jnr" / f"{mode_prefix}_jnr_results_{split}.csv"),
        )
        jnr_json = {}
        for jnr, m in jnr_results.items():
            jnr_json[str(jnr)] = {
                k: v for k, v in m.items()
                if k not in ("labels", "probabilities", "features",
                             "per_class_recall", "per_class_precision", "per_class_f1")
            }
        save_metrics_json(
            {"per_jnr": jnr_json}, output_dir,
            filename=f"metrics_by_jnr_{split}.json",
            mode="by_jnr", split=split,
            extra_meta={"checkpoint": str(checkpoint_path), "ablation_mode": ablation_mode},
        )
        # Per-JNR ROC / PR curves
        do_jnr_curves = do_viz or do_roc or do_pr
        if do_jnr_curves:
            jnr_viz_dir = str(output_dir / "by_jnr")
            for jnr_val, jnr_result in sorted(jnr_results.items()):
                if "probabilities" in jnr_result and jnr_result["probabilities"].size > 0:
                    prefix = f"{mode_prefix}_jnr_{jnr_val:+.0f}_{split}"
                    if do_viz or do_roc:
                        plot_roc_curves(
                            jnr_result["labels"], jnr_result["probabilities"],
                            class_names, save_dir=jnr_viz_dir, prefix=prefix,
                            mode_title=f"JNR={jnr_val:+d}",
                        )
                    if do_viz or do_pr:
                        plot_pr_curves(
                            jnr_result["labels"], jnr_result["probabilities"],
                            class_names, save_dir=jnr_viz_dir, prefix=prefix,
                            mode_title=f"JNR={jnr_val:+d}",
                        )

        # ── 2) By-Combination (separate pass — uses single-class _predict_batch) ──
        print(f"\n{'='*60}")
        print(f"  [2/2] By-Combination Evaluation on {split}")
        print(f"{'='*60}")
        results_bc = evaluator.evaluate_by_combination(
            config=config, split=split, ablation_mode=ablation_mode,
            threshold=threshold, print_samples=print_samples)

        bc_labels = results_bc.get('_labels')
        bc_preds = results_bc.get('_predictions')
        bc_probs = results_bc.get('_probabilities')
        bc_feats = results_bc.get('_features')

        # JSON: global + seen + unseen subset bundles only
        bc_export = {
            "global": results_bc.get("global", {}),
            "seen": results_bc.get("seen", {}),
            "unseen": results_bc.get("unseen", {}),
        }
        save_metrics_json(
            bc_export, output_dir,
            filename=f"metrics_by_combination_{split}.json",
            mode="by_combination", split=split,
            extra_meta={"checkpoint": str(checkpoint_path), "ablation_mode": ablation_mode},
        )

        if bc_labels is not None:
            np.savez(str(output_dir / f"{mode_prefix}_bycombo_{split}.npz"),
                     labels=bc_labels, predictions=bc_preds, features=bc_feats)

            print(f"\n[DEBUG BC] Calling plot_confusion_by_combination: labels={bc_labels.shape}, preds={bc_preds.shape}", flush=True)
            import matplotlib.pyplot as _plt; _plt.close('all')
            try:
                evaluator.plot_confusion_by_combination(
                    bc_labels, bc_preds,
                    save_path=str(output_dir / f"{mode_prefix}_confusion_bc_{split}.png"),
                    seen_combinations=seen_combinations,
                    unseen_combinations=unseen_combinations,
                )
                print(f"[DEBUG BC] plot_confusion_by_combination OK", flush=True)
            except Exception as e:
                print(f"[DEBUG BC] FAILED: {e}", flush=True)
                import traceback
                traceback.print_exc()

            if do_viz or do_roc:
                plot_roc_curves(
                    bc_labels, bc_probs, class_names,
                    save_dir=str(output_dir), prefix=f"{mode_prefix}_{split}_byc",
                    mode_title=f"by_combination ({split})",
                )
            if do_viz or do_pr:
                plot_pr_curves(
                    bc_labels, bc_probs, class_names,
                    save_dir=str(output_dir), prefix=f"{mode_prefix}_{split}_byc",
                    mode_title=f"by_combination ({split})",
                )

    # ==================================================================
    # Mode: zero_shot
    # ==================================================================
    elif mode == "zero_shot":
        print(f"\n{'='*60}")
        print(f"Zero-Shot Evaluation on {split} split")
        print(f"{'='*60}")

        results = evaluator.evaluate_zero_shot(
            config=config, split=split, ablation_mode=ablation_mode,
            threshold=threshold, print_samples=print_samples)
        evaluator.print_metrics(results["metrics"], title="Zero-Shot Evaluation Results")
        save_metrics_json(
            results["metrics"], output_dir,
            filename=f"metrics_zero_shot_{split}.json",
            mode="zero_shot", split=split,
            extra_meta={"checkpoint": str(checkpoint_path), "ablation_mode": ablation_mode},
        )

        np.savez(str(output_dir / f"{mode_prefix}_zeroshot_{split}.npz"),
                 labels=results["labels"],
                 predictions=results["predictions"],
                 features=results["features"])

        evaluator.plot_confusion_by_combination(
            results["labels"], results["predictions"],
            save_path=str(output_dir / f"{mode_prefix}_confusion_zs_{split}.png"),
            seen_combinations=seen_combinations,
            unseen_combinations=unseen_combinations,
        )
        if do_viz or do_roc:
            plot_roc_curves(
                results["labels"], results["probabilities"], class_names,
                save_dir=str(output_dir), prefix=f"{mode_prefix}_{split}_zs",
                mode_title=f"zero_shot ({split})",
            )
        if do_viz or do_pr:
            plot_pr_curves(
                results["labels"], results["probabilities"], class_names,
                save_dir=str(output_dir), prefix=f"{mode_prefix}_{split}_zs",
                mode_title=f"zero_shot ({split})",
            )
        if do_tsne:
            print("\nGenerating t-SNE...")
            evaluator.plot_feature_tsne(
                results["features"], results["labels"],
                save_path=str(output_dir / f"{mode_prefix}_tsne_{split}.png"),
                seen_combinations=seen_combinations,
                unseen_combinations=unseen_combinations,
            )
        if do_umap:
            print("\nGenerating UMAP...")
            evaluator.plot_feature_umap(
                results["features"], results["labels"],
                save_path=str(output_dir / f"{mode_prefix}_umap_{split}.png"),
                seen_combinations=seen_combinations,
                unseen_combinations=unseen_combinations,
            )
        if do_viz:
            evaluator.plot_label_cooccurrence(
                results["labels"], results["predictions"],
                save_path=str(output_dir / f"{mode_prefix}_cooccurrence_{split}.png"),
            )

    # ==================================================================
    # Mode: by_combination
    # ==================================================================
    elif mode == "by_combination":
        print(f"\n{'='*60}")
        print(f"By-Combination Evaluation on {split} split")
        print(f"{'='*60}")

        results = evaluator.evaluate_by_combination(
            config=config, split=split, ablation_mode=ablation_mode,
            threshold=threshold, print_samples=print_samples)

        bc_labels = results.get('_labels')
        bc_preds = results.get('_predictions')
        bc_probs = results.get('_probabilities')
        bc_feats = results.get('_features')

        save_metrics_json(
            {
                "global": results.get("global", {}),
                "seen": results.get("seen", {}),
                "unseen": results.get("unseen", {}),
            },
            output_dir,
            filename=f"metrics_by_combination_{split}.json",
            mode="by_combination", split=split,
            extra_meta={"checkpoint": str(checkpoint_path), "ablation_mode": ablation_mode},
        )

        if bc_labels is not None:
            np.savez(str(output_dir / f"{mode_prefix}_bycombo_{split}.npz"),
                     labels=bc_labels, predictions=bc_preds, features=bc_feats)

            print(f"\n[DEBUG BC] Calling plot_confusion_by_combination: labels={bc_labels.shape}, preds={bc_preds.shape}", flush=True)
            import matplotlib.pyplot as _plt; _plt.close('all')
            try:
                evaluator.plot_confusion_by_combination(
                    bc_labels, bc_preds,
                    save_path=str(output_dir / f"{mode_prefix}_confusion_bc_{split}.png"),
                    seen_combinations=seen_combinations,
                    unseen_combinations=unseen_combinations,
                )
                print(f"[DEBUG BC] plot_confusion_by_combination OK", flush=True)
            except Exception as e:
                print(f"[DEBUG BC] FAILED: {e}", flush=True)
                import traceback
                traceback.print_exc()

            if do_viz or do_roc:
                plot_roc_curves(
                    bc_labels, bc_probs, class_names,
                    save_dir=str(output_dir), prefix=f"{mode_prefix}_{split}_byc",
                    mode_title=f"by_combination ({split})",
                )
            if do_viz or do_pr:
                plot_pr_curves(
                    bc_labels, bc_probs, class_names,
                    save_dir=str(output_dir), prefix=f"{mode_prefix}_{split}_byc",
                    mode_title=f"by_combination ({split})",
                )

    # ==================================================================
    # Mode: by_jnr
    # ==================================================================
    elif mode == "by_jnr":
        print(f"\n{'='*60}")
        print(f"By-JNR Evaluation on {split} split")
        print(f"{'='*60}")

        results = evaluator.evaluate_by_jnr(
            config=config,
            split=split,
            ablation_mode=ablation_mode,
            threshold=threshold,
            output_dir=str(output_dir),
            print_samples=print_samples,
        )

        if not results:
            print("No JNR data found!")
            return

        evaluator.print_jnr_results(
            results,
            save_path=str(output_dir / "by_jnr" / f"{mode_prefix}_jnr_results_{split}.csv"),
        )
        jnr_json = {
            str(jnr): {
                k: v for k, v in m.items()
                if k not in ("labels", "probabilities", "features",
                             "per_class_recall", "per_class_precision", "per_class_f1")
            }
            for jnr, m in results.items()
        }
        save_metrics_json(
            {"per_jnr": jnr_json}, output_dir,
            filename=f"metrics_by_jnr_{split}.json",
            mode="by_jnr", split=split,
            extra_meta={"checkpoint": str(checkpoint_path), "ablation_mode": ablation_mode},
        )

        # Per-JNR ROC / PR curves
        do_jnr_curves = do_viz or do_roc or do_pr
        if do_jnr_curves:
            jnr_viz_dir = str(output_dir / "by_jnr")
            for jnr_val, jnr_result in sorted(results.items()):
                if "probabilities" in jnr_result and jnr_result["probabilities"].size > 0:
                    prefix = f"{mode_prefix}_jnr_{jnr_val:+.0f}_{split}"
                    if do_viz or do_roc:
                        plot_roc_curves(
                            jnr_result["labels"],
                            jnr_result["probabilities"],
                            class_names,
                            save_dir=jnr_viz_dir,
                            prefix=prefix,
                            mode_title=f"JNR={jnr_val:+d}",
                        )
                    if do_viz or do_pr:
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
