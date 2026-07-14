"""
Evaluation script for 1D CZSL models (Conformer / ResNet1D / CNN1D).

Supports three modes:
    zero_shot       — global zero-shot evaluation
    by_combination  — per-combination (seen/unseen) breakdown
    by_jnr          — per-JNR performance analysis

参数优先级: CLI > config_1d.yaml 的 evaluation: > 内置默认值。
在 evaluation 中写好 checkpoint / mode / split 后，可省略超长 CLI:

Usage:
    python conformer_1d/evaluate_conformer.py
    python conformer_1d/evaluate_conformer.py --mode by_jnr --split test
    python conformer_1d/evaluate_conformer.py --checkpoint CHKPT --mode zero_shot
    python conformer_1d/evaluate_conformer.py --checkpoint CHKPT --backbone resnet1d --mode by_combination
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_curve, auc

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _parent)

from conformer_1d.model_1d import Base1DCZSLModel, create_1d_model
from conformer_1d.data_1d import TimeSignalDataset, create_1d_dataloaders, collate_fn_conformer, TokenizerWrapper, set_collate_use_translation
from multi.metrics_czsl import (
    compute_metrics_bundle,
    compute_subset_bundles,
    print_metrics_report,
    print_jnr_metrics_table,
    save_metrics_json,
)


# ---------------------------------------------------------------------------
# JNR-by-JNR dataloader factory (1D time-domain version)
# ---------------------------------------------------------------------------

def create_1d_jnr_dataloaders(
    config: dict,
    split: str = 'test',
    by_jnr: bool = True,
) -> dict:
    """Create one DataLoader per JNR level for 1D time-domain data."""
    data_config = config.get('data', {})
    base_path = data_config.get('base_path')
    jnr_start = data_config.get('jnr_start', 0)
    jnr_end = data_config.get('jnr_end', 20)
    jnr_step = data_config.get('jnr_step', 1)
    time_var_name = data_config.get('time_var_name', 'all_times')
    time_seq_len = data_config.get('time_seq_len', 8000)
    batch_size = config.get('train', {}).get('batch_size', 16)
    num_workers = data_config.get('num_workers', 0)  # 0 = main process only (safer)
    pin_memory = data_config.get('pin_memory', True)

    jamming_classes = config.get('jamming_classes', [])
    class_names = [jc['name'] if isinstance(jc, dict) else jc for jc in jamming_classes]

    jnr_levels = list(range(jnr_start, jnr_end + 1, jnr_step))

    clip_model_name = config.get('model', {}).get('clip_model', 'ViT-B/32')
    tokenizer = TokenizerWrapper(model_type="clip")

    set_collate_use_translation(config.get('use_translation', False))

    jnr_loaders = {}

    for jnr in jnr_levels:
        data_folder = os.path.join(base_path, f'JNR_+{jnr}')
        time_file = os.path.join(data_folder, f'{split}_echo_times.mat')
        metadata_file = os.path.join(data_folder, f'{split}_echo_metadata.json')

        if not os.path.exists(time_file) or not os.path.exists(metadata_file):
            print(f"  Skipping JNR={jnr}: data not found")
            continue

        ds = TimeSignalDataset(
            time_file=time_file,
            metadata_file=metadata_file,
            time_var_name=time_var_name,
            class_names=class_names,
            time_seq_len=time_seq_len,
        )

        from functools import partial
        collate = partial(collate_fn_conformer, tokenizer_fn=tokenizer, model_type="clip")

        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            collate_fn=collate,
        )
        jnr_loaders[jnr] = loader
        print(f"  JNR=+{jnr}: {len(ds)} samples")

    return jnr_loaders


# ---------------------------------------------------------------------------
# CZSLEvaluator — adapted for Conformer
# ---------------------------------------------------------------------------

class ConformerEvaluator:
    """Evaluation suite for 1D CZSL models (Conformer / ResNet1D / CNN1D).

    Works with any Base1DCZSLModel subclass. Mirrors CZSLEvaluator
    (evaluate_czsl.py:214) but handles time-domain signals instead of STFT images.
    """

    def __init__(self, model: Base1DCZSLModel, config: dict, device: torch.device):
        self.model = model
        self.config = config
        self.device = device

        jamming_classes = config.get('jamming_classes', [])
        self.class_names = [jc['name'] if isinstance(jc, dict) else jc for jc in jamming_classes]
        self.num_classes = len(self.class_names)

        # Build seen/unseen sets from CZSL config
        czsl_config = config.get('czsl', {})
        self.seen_combos = czsl_config.get('seen_combinations', [])
        self.unseen_combos = czsl_config.get('unseen_combinations', [])

        seen_set = set()
        self._seen_indices = []  # 保持配置顺序的列表
        for c in self.seen_combos:
            indices = tuple(sorted(self.class_names.index(x) for x in c if x in self.class_names))
            if indices:
                seen_set.add(indices)
                self._seen_indices.append(indices)
        self.seen_set = seen_set

        unseen_set = set()
        self._unseen_indices = []  # 保持配置顺序的列表
        for c in self.unseen_combos:
            indices = tuple(sorted(self.class_names.index(x) for x in c if x in self.class_names))
            if indices:
                unseen_set.add(indices)
                self._unseen_indices.append(indices)
        self.unseen_set = unseen_set

        self.output_dir = None
        self._all_probs = None
        self._all_labels = None

    # ------------------------------------------------------------------
    # Prediction helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_batch(self, time_signals):
        """Run prediction on a batch of time-domain signals.

        Returns (probs, preds) as CPU tensors.
        """
        time_signals = time_signals.to(self.device)
        text_features = self.model.get_cached_text_features()
        signal_features = self.model.encode_image(time_signals)
        signal_features = F.normalize(signal_features, dim=-1)
        logit_scale = self.model.model.logit_scale.exp()
        logits = logit_scale * (signal_features @ text_features.T)

        probs = torch.softmax(logits, dim=-1)
        batch_size = time_signals.shape[0]
        threshold = 1.0 / self.num_classes
        top_k = 3
        topk_values, topk_indices = torch.topk(probs, k=top_k, dim=-1)

        preds = torch.zeros(batch_size, self.num_classes, device=self.device)
        for b in range(batch_size):
            for j, idx in enumerate(topk_indices[b]):
                if topk_values[b, j] > threshold:
                    preds[b, idx] = 1.0

        return probs.cpu(), preds.cpu()

    # ------------------------------------------------------------------
    # Zero-shot evaluation
    # ------------------------------------------------------------------

    def _build_single_jnr_loader(self, config: dict, split: str, jnr: int):
        """Build one JNR DataLoader on demand (lazy; freed by caller after use)."""
        data_config = config.get('data', {})
        base_path = data_config.get('base_path')
        time_var_name = data_config.get('time_var_name', 'all_times')
        time_seq_len = data_config.get('time_seq_len', 8000)
        batch_size = config.get('train', {}).get('batch_size', 16)
        pin_memory = data_config.get('pin_memory', True)

        data_folder = os.path.join(base_path, f'JNR_+{jnr}')
        time_file = os.path.join(data_folder, f'{split}_echo_times.mat')
        metadata_file = os.path.join(data_folder, f'{split}_echo_metadata.json')
        if not os.path.exists(time_file) or not os.path.exists(metadata_file):
            return None

        ds = TimeSignalDataset(
            time_file=time_file,
            metadata_file=metadata_file,
            time_var_name=time_var_name,
            class_names=self.class_names,
            time_seq_len=time_seq_len,
        )
        from functools import partial
        tokenizer = TokenizerWrapper(model_type="clip")
        collate = partial(collate_fn_conformer, tokenizer_fn=tokenizer, model_type="clip")
        return DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=pin_memory, collate_fn=collate,
        )

    def _run_zero_shot_on_loader(self, data_loader, debug: bool = False, debug_done: bool = False):
        """Process one loader; returns batch accumulators + debug_done flag."""
        all_labels, all_preds, all_probs = [], [], []
        seen_correct = unseen_correct = other_correct = 0
        seen_total = unseen_total = other_total = 0
        has_seen_unseen = bool(self.seen_set or self.unseen_set)

        for batch in tqdm(data_loader, desc="Zero-shot eval", leave=False):
            time_signals, _, _, labels, texts, metas = batch[:6]
            labels = labels.to(self.device)
            batch_size = time_signals.shape[0]

            # Single-class probs for ROC/PR (match multi)
            single_text_features = self.model.get_cached_text_features()
            single_text_features = F.normalize(single_text_features, dim=-1)
            signal_features = self.model.encode_image(time_signals)
            signal_features = F.normalize(signal_features, dim=-1)
            logit_scale = self.model.model.logit_scale.exp()
            single_logits = logit_scale * (signal_features @ single_text_features.T)
            single_probs = torch.softmax(single_logits, dim=-1)
            all_probs.append(single_probs.cpu())

            # Combo-space top-1 (match multi)
            _, top_indices, all_names = self.model.zero_shot_predict(
                time_signals, use_combinations=True, top_k=1,
                use_translation=self.config.get('use_translation', False),
            )
            preds = torch.zeros(batch_size, self.num_classes, device=self.device)
            pred_names_list = []
            for i in range(batch_size):
                comb_name = all_names[top_indices[i, 0].item()]
                pred_names_list.append(comb_name)
                for part in comb_name.split('+'):
                    part = part.strip()
                    if part in self.class_names:
                        preds[i, self.class_names.index(part)] = 1.0

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

            if has_seen_unseen:
                for i in range(batch_size):
                    true_comb = tuple(sorted(torch.where(labels[i] == 1)[0].tolist()))
                    pred_comb = tuple(sorted(torch.where(preds[i] == 1)[0].tolist()))
                    is_correct = (true_comb == pred_comb)
                    if true_comb in self.seen_set:
                        seen_total += 1
                        if is_correct:
                            seen_correct += 1
                    elif true_comb in self.unseen_set:
                        unseen_total += 1
                        if is_correct:
                            unseen_correct += 1
                    else:
                        other_total += 1
                        if is_correct:
                            other_correct += 1

            if debug and not debug_done:
                print(f"\n{'='*60}")
                print(f"[DEBUG] batch_size={batch_size}")
                for i in range(min(3, batch_size)):
                    true_idx = torch.where(labels[i] == 1)[0].tolist()
                    pred_idx = torch.where(preds[i] == 1)[0].tolist()
                    top3_vals, top3_idx = torch.topk(single_probs[i], k=3)
                    print(f"  [{i}] True={[self.class_names[j] for j in true_idx]} "
                          f"Pred={[self.class_names[j] for j in pred_idx]}")
                    print(f"       Combo pred: {pred_names_list[i]}")
                    print(f"       Top-3 single: {[(self.class_names[j.item()], f'{v:.3f}') for v, j in zip(top3_vals, top3_idx)]}")
                debug_done = True

        stats = {
            'seen_correct': seen_correct, 'seen_total': seen_total,
            'unseen_correct': unseen_correct, 'unseen_total': unseen_total,
            'other_correct': other_correct, 'other_total': other_total,
        }
        return all_labels, all_preds, all_probs, stats, debug_done

    @torch.no_grad()
    def evaluate_zero_shot(
        self,
        data_loader=None,
        config: dict = None,
        split: str = 'test',
        debug: bool = False,
        output_dir: str = None,
    ):
        """Global zero-shot: combo-space top-1 (multi-style).

        Prefer JNR lazy-load (config+split): load one JNR at a time, then concat.
        Falls back to a pre-built data_loader if provided.
        """
        self.model.eval()
        all_labels, all_preds, all_probs = [], [], []
        seen_correct = unseen_correct = other_correct = 0
        seen_total = unseen_total = other_total = 0
        has_seen_unseen = bool(self.seen_set or self.unseen_set)
        debug_done = False

        if config is not None:
            # JNR lazy-load → global concat
            set_collate_use_translation(config.get('use_translation', False))
            data_config = config.get('data', {})
            jnr_start = data_config.get('jnr_start', 0)
            jnr_end = data_config.get('jnr_end', 20)
            jnr_step = data_config.get('jnr_step', 1)
            jnr_levels = list(range(jnr_start, jnr_end + 1, jnr_step))
            print(f"Zero-shot: JNR lazy-load on {split} "
                  f"(JNR {jnr_start}..{jnr_end} step {jnr_step})")

            for jnr in jnr_levels:
                loader = self._build_single_jnr_loader(config, split, jnr)
                if loader is None:
                    print(f"  Skipping JNR=+{jnr}: data not found")
                    continue
                print(f"  Evaluating JNR=+{jnr}...")
                labs, preds, probs, stats, debug_done = self._run_zero_shot_on_loader(
                    loader, debug=debug, debug_done=debug_done,
                )
                all_labels.extend(labs)
                all_preds.extend(preds)
                all_probs.extend(probs)
                seen_correct += stats['seen_correct']
                seen_total += stats['seen_total']
                unseen_correct += stats['unseen_correct']
                unseen_total += stats['unseen_total']
                other_correct += stats['other_correct']
                other_total += stats['other_total']
                del loader
        elif data_loader is not None:
            labs, preds, probs, stats, debug_done = self._run_zero_shot_on_loader(
                data_loader, debug=debug, debug_done=False,
            )
            all_labels, all_preds, all_probs = labs, preds, probs
            seen_correct = stats['seen_correct']
            seen_total = stats['seen_total']
            unseen_correct = stats['unseen_correct']
            unseen_total = stats['unseen_total']
            other_correct = stats['other_correct']
            other_total = stats['other_total']
        else:
            raise ValueError("evaluate_zero_shot requires config=... or data_loader=...")

        if not all_labels:
            empty = np.zeros((0, self.num_classes))
            return {}, empty, empty, empty

        all_labels_np = torch.cat(all_labels).numpy()
        all_preds_np = torch.cat(all_preds).numpy()
        all_probs_np = torch.cat(all_probs).numpy()

        metrics = compute_metrics_bundle(
            all_labels_np, all_preds_np, self.class_names,
            y_prob=all_probs_np,
            seen_set=self.seen_set, unseen_set=self.unseen_set,
            dual_write=True,
        )
        _ = (has_seen_unseen, seen_correct, seen_total, unseen_correct, unseen_total,
             other_correct, other_total)

        if output_dir:
            self.plot_confusion_matrix(
                all_labels_np, all_preds_np, output_dir,
                title="Zero-Shot", filename="zero_shot_cm.png",
                seen_combinations=self._seen_indices if self._seen_indices else None,
                unseen_combinations=self._unseen_indices if self._unseen_indices else None,
            )

        return metrics, all_labels_np, all_preds_np, all_probs_np

    # ------------------------------------------------------------------
    # By-combination evaluation (seen / unseen separation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_by_combination(self, data_loader, debug: bool = False, output_dir: str = None):
        """Evaluate breaking down results by seen, unseen, and other combinations.

        Uses single-class softmax + topk(3) + threshold (matching multi's by_combination).
        """
        self.model.eval()
        all_labels, all_preds, all_probs = [], [], []
        seen_correct, seen_total = 0, 0
        unseen_correct, unseen_total = 0, 0
        other_correct, other_total = 0, 0

        debug_done = False
        for batch in tqdm(data_loader, desc="Combination eval"):
            time_signals, _, _, labels, texts, metas = batch[:6]
            labels = labels.to(self.device)
            probs, preds = self._predict_batch(time_signals)
            all_labels.append(labels.cpu())
            all_preds.append(preds)
            all_probs.append(probs)

            for i in range(time_signals.shape[0]):
                true_comb = tuple(sorted(torch.where(labels[i] == 1)[0].tolist()))
                pred_comb = tuple(sorted(torch.where(preds[i] == 1)[0].tolist()))
                is_correct = (true_comb == pred_comb)

                if true_comb in self.seen_set:
                    seen_total += 1
                    if is_correct:
                        seen_correct += 1
                elif true_comb in self.unseen_set:
                    unseen_total += 1
                    if is_correct:
                        unseen_correct += 1
                else:
                    other_total += 1
                    if is_correct:
                        other_correct += 1

            if debug and not debug_done:
                print(f"\n[DEBUG] Seen set size: {len(self.seen_set)}, Unseen set size: {len(self.unseen_set)}")
                for i in range(min(3, time_signals.shape[0])):
                    tc = tuple(sorted(torch.where(labels[i] == 1)[0].tolist()))
                    print(f"  [{i}] True={tc} in_seen={tc in self.seen_set} in_unseen={tc in self.unseen_set}")
                debug_done = True

        all_labels_np = torch.cat(all_labels).numpy()
        all_preds_np = torch.cat(all_preds).numpy()
        all_probs_np = torch.cat(all_probs).numpy()

        # Global + seen/unseen subset bundles
        bundles = compute_subset_bundles(
            all_labels_np, all_preds_np, self.class_names,
            seen_set=self.seen_set, unseen_set=self.unseen_set,
            y_prob=all_probs_np, dual_write=True,
        )
        metrics = dict(bundles["global"])
        metrics["global"] = bundles["global"]
        metrics["seen"] = bundles["seen"]
        metrics["unseen"] = bundles["unseen"]
        _ = (seen_correct, seen_total, unseen_correct, unseen_total, other_correct, other_total)

        # Save confusion matrices
        if output_dir:
            self.plot_confusion_matrix(
                all_labels_np, all_preds_np, output_dir,
                title="By-Combination", filename="by_combination_cm.png",
                seen_combinations=self._seen_indices if self._seen_indices else None,
                unseen_combinations=self._unseen_indices if self._unseen_indices else None,
            )

            # Separate seen-only and unseen-only confusion matrices
            seen_mask = np.array([
                tuple(sorted(np.where(all_labels_np[i] == 1)[0].tolist())) in self.seen_set
                for i in range(len(all_labels_np))
            ])
            unseen_mask = np.array([
                tuple(sorted(np.where(all_labels_np[i] == 1)[0].tolist())) in self.unseen_set
                for i in range(len(all_labels_np))
            ])

            if seen_mask.sum() > 0:
                self.plot_confusion_matrix(
                    all_labels_np[seen_mask], all_preds_np[seen_mask], output_dir,
                    title="By-Combination — Seen Only", filename="by_combination_cm_seen.png",
                )
            if unseen_mask.sum() > 0:
                self.plot_confusion_matrix(
                    all_labels_np[unseen_mask], all_preds_np[unseen_mask], output_dir,
                    title="By-Combination — Unseen Only", filename="by_combination_cm_unseen.png",
                )

        return metrics, all_labels_np, all_preds_np, all_probs_np

    # ------------------------------------------------------------------
    # By-JNR evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_by_jnr(self, jnr_loaders: dict):
        """Evaluate per JNR level."""
        self.model.eval()
        num_classes = self.num_classes
        results = {}

        for jnr, data_loader in sorted(jnr_loaders.items()):
            print(f"\nEvaluating JNR=+{jnr}...")
            all_labels, all_preds, all_probs = [], [], []
            seen_correct, seen_total = 0, 0
            unseen_correct, unseen_total = 0, 0
            class_stats = np.zeros((num_classes, 4), dtype=np.int64)

            for batch in tqdm(data_loader, desc=f"JNR=+{jnr}"):
                time_signals, _, _, labels, texts, metas = batch[:6]
                labels = labels.to(self.device)
                probs, preds = self._predict_batch(time_signals)
                all_labels.append(labels.cpu())
                all_preds.append(preds)
                all_probs.append(probs)

                labels_np_b = labels.cpu().numpy()
                preds_np_b = preds.numpy()
                for c in range(num_classes):
                    true_c = labels_np_b[:, c]
                    pred_c = preds_np_b[:, c]
                    class_stats[c, 0] += np.sum((true_c == 1) & (pred_c == 1))
                    class_stats[c, 1] += np.sum((true_c == 0) & (pred_c == 1))
                    class_stats[c, 2] += np.sum((true_c == 1) & (pred_c == 0))
                    class_stats[c, 3] += np.sum((true_c == 0) & (pred_c == 0))

                for i in range(time_signals.shape[0]):
                    true_comb = tuple(sorted(torch.where(labels[i] == 1)[0].tolist()))
                    pred_comb = tuple(sorted(torch.where(preds[i] == 1)[0].tolist()))
                    is_correct = (true_comb == pred_comb)
                    if true_comb in self.seen_set:
                        seen_total += 1
                        if is_correct:
                            seen_correct += 1
                    elif true_comb in self.unseen_set:
                        unseen_total += 1
                        if is_correct:
                            unseen_correct += 1

            all_labels_np = torch.cat(all_labels).numpy()
            all_preds_np = torch.cat(all_preds).numpy()
            all_probs_np = torch.cat(all_probs).numpy()

            per_class_recall = np.zeros(num_classes)
            per_class_precision = np.zeros(num_classes)
            per_class_f1 = np.zeros(num_classes)
            for c in range(num_classes):
                tp, fp, fn, tn = class_stats[c]
                per_class_recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0
                per_class_precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0
                if per_class_precision[c] + per_class_recall[c] > 0:
                    per_class_f1[c] = (2 * per_class_precision[c] * per_class_recall[c] /
                                       (per_class_precision[c] + per_class_recall[c]))

            bundle = compute_metrics_bundle(
                all_labels_np, all_preds_np, self.class_names,
                y_prob=all_probs_np,
                seen_set=self.seen_set, unseen_set=self.unseen_set,
                dual_write=True,
            )
            results[jnr] = {
                **bundle,
                'total_samples': bundle['num_samples'],
                'per_class_recall': per_class_recall,
                'per_class_precision': per_class_precision,
                'per_class_f1': per_class_f1,
                'all_labels': all_labels_np,
                'all_preds': all_preds_np,
                'all_probs': all_probs_np,
            }
            _ = (seen_correct, unseen_correct, seen_total, unseen_total)

        return results

    # ------------------------------------------------------------------
    # Report & visualization
    # ------------------------------------------------------------------

    def print_report(self, metrics, title="Evaluation Results"):
        """Print formatted evaluation results (unified MetricsBundle)."""
        if isinstance(metrics.get("global"), dict) and "subset_accuracy" in metrics.get("global", {}):
            print_metrics_report(metrics["global"], title=f"{title} — global")
            if metrics.get("seen", {}).get("num_samples", 0):
                print_metrics_report(metrics["seen"], title=f"{title} — seen subset")
            if metrics.get("unseen", {}).get("num_samples", 0):
                print_metrics_report(metrics["unseen"], title=f"{title} — unseen subset")
            return
        print_metrics_report(metrics, title=title)

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    def plot_confusion_matrix(
        self, labels: np.ndarray, preds: np.ndarray,
        output_dir: str, title: str = "Confusion Matrix",
        filename: str = "confusion_matrix.png",
        seen_combinations: list = None,
        unseen_combinations: list = None,
    ):
        """Plot multi-label confusion matrix.

        Generates two figures:
          1. per_class_{filename}  — Per-class TP/FP/FN bar chart + F1
          2. combination_{filename} — Combination confusion matrix
             (sorted: Seen first, then Unseen, per config order)

        Args:
            labels: (N, num_classes) multi-hot
            preds:  (N, num_classes) multi-hot
            output_dir: where to save the .png
            title: plot title
            filename: output filename
            seen_combinations: list of index tuples, e.g. [(0,), (1, 2)]
            unseen_combinations: list of index tuples
        """
        os.makedirs(output_dir, exist_ok=True)
        num_classes = self.num_classes

        # ---- Figure 1: Per-class TP/FP/FN bar chart + F1 ----
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        tp = np.zeros(num_classes, dtype=np.int64)
        fp = np.zeros(num_classes, dtype=np.int64)
        fn = np.zeros(num_classes, dtype=np.int64)
        for c in range(num_classes):
            tp[c] = np.sum((labels[:, c] == 1) & (preds[:, c] == 1))
            fp[c] = np.sum((labels[:, c] == 0) & (preds[:, c] == 1))
            fn[c] = np.sum((labels[:, c] == 1) & (preds[:, c] == 0))

        x = np.arange(num_classes)
        width = 0.25
        axes[0].bar(x - width, tp, width, label='TP', color='#2ecc71')
        axes[0].bar(x, fp, width, label='FP', color='#e74c3c')
        axes[0].bar(x + width, fn, width, label='FN', color='#f39c12')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(self.class_names, rotation=45, ha='right', fontsize=8)
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'{title} — Per-Class TP/FP/FN')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        per_class_f1 = np.zeros(num_classes)
        for c in range(num_classes):
            per_class_f1[c] = (2 * tp[c] / (2 * tp[c] + fp[c] + fn[c])
                               if (2 * tp[c] + fp[c] + fn[c]) > 0 else 0)

        bars = axes[1].bar(x, per_class_f1, color=plt.cm.YlOrRd(per_class_f1))
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(self.class_names, rotation=45, ha='right', fontsize=8)
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title(f'{title} — Per-Class F1')
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, per_class_f1):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{val:.2f}', ha='center', va='bottom', fontsize=7)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'per_class_{filename}'), dpi=150)
        plt.close(fig)

        # ---- Figure 2: Combination confusion matrix ----
        seen_set = set(tuple(sorted(c)) for c in (seen_combinations or []))
        unseen_set = set(tuple(sorted(c)) for c in (unseen_combinations or []))
        has_config = bool(seen_set or unseen_set)

        if has_config:
            # 按配置顺序排列: Seen 在前, Unseen 在后
            ordered_combs = []
            if seen_combinations:
                for comb in seen_combinations:
                    comb_tuple = tuple(sorted(comb))
                    if comb_tuple not in ordered_combs:
                        ordered_combs.append(comb_tuple)
            if unseen_combinations:
                for comb in unseen_combinations:
                    comb_tuple = tuple(sorted(comb))
                    if comb_tuple not in ordered_combs:
                        ordered_combs.append(comb_tuple)
            unique_combs = ordered_combs
        else:
            # 从数据中提取所有出现的组合, 按字母序排列
            all_combs_set = set()
            for i in range(len(labels)):
                all_combs_set.add(tuple(sorted(np.where(labels[i] == 1)[0].tolist())))
                all_combs_set.add(tuple(sorted(np.where(preds[i] == 1)[0].tolist())))
            unique_combs = sorted(all_combs_set)

        combo_to_idx = {c: i for i, c in enumerate(unique_combs)}
        Nc = len(unique_combs)

        # 构建混淆矩阵
        cm = np.zeros((Nc, Nc), dtype=np.int64)
        other_count = 0
        for i in range(len(labels)):
            true_comb = tuple(sorted(np.where(labels[i] == 1)[0].tolist()))
            pred_comb = tuple(sorted(np.where(preds[i] == 1)[0].tolist()))
            if true_comb in combo_to_idx and pred_comb in combo_to_idx:
                cm[combo_to_idx[true_comb], combo_to_idx[pred_comb]] += 1
            else:
                other_count += 1
        if other_count > 0:
            print(f"  Samples not in configured combinations: {other_count}")

        # 组合名 + [S]/[U] 标记
        combo_names = []
        for comb in unique_combs:
            if len(comb) == 0:
                name = "None"
            else:
                name = "+".join(self.class_names[i] for i in comb)
            if has_config:
                if comb in seen_set:
                    name = f"[S] {name}"
                elif comb in unseen_set:
                    name = f"[U] {name}"
            combo_names.append(name)

        # 绘图 — 外观与 multi/evaluate_czsl.py 对齐
        fig_size = max(10, Nc * 0.5)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        sns.heatmap(cm, ax=ax, xticklabels=combo_names, yticklabels=combo_names,
                     annot=True, fmt='d', cmap='Blues')
        ax.set_xlabel('Predicted Combination')
        ax.set_ylabel('True Combination')
        ax.set_title(f'{title} — Combination Confusion Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'combination_{filename}'), dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"Confusion matrices saved to {output_dir}")

    # ------------------------------------------------------------------
    # JNR curves
    # ------------------------------------------------------------------

    def plot_jnr_curve(self, jnr_results: dict, output_dir: str):
        """Plot accuracy vs JNR curves."""
        os.makedirs(output_dir, exist_ok=True)
        jnrs = sorted(jnr_results.keys())
        seen_accs = [jnr_results[j]['seen_accuracy'] for j in jnrs]
        unseen_accs = [jnr_results[j]['unseen_accuracy'] for j in jnrs]
        overall = [
            jnr_results[j].get('subset_accuracy', jnr_results[j].get('combination_accuracy', 0))
            for j in jnrs
        ]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(jnrs, seen_accs, 'o-', label='Seen', linewidth=2)
        ax.plot(jnrs, unseen_accs, 's-', label='Unseen', linewidth=2)
        ax.plot(jnrs, overall, '^-', label='Overall', linewidth=2)
        ax.set_xlabel('JNR (dB)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Conformer CZSL Accuracy vs JNR', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'jnr_accuracy_curve.png'), dpi=150)
        plt.close(fig)

        # Per-class F1 vs JNR heatmap
        num_classes = self.num_classes
        f1_matrix = np.zeros((num_classes, len(jnrs)))
        for j_idx, jnr in enumerate(jnrs):
            f1_matrix[:, j_idx] = jnr_results[jnr]['per_class_f1']

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(f1_matrix, ax=ax, xticklabels=jnrs, yticklabels=self.class_names,
                     annot=True, fmt='.2f', cmap='YlOrRd', cbar_kws={'label': 'F1'})
        ax.set_xlabel('JNR (dB)', fontsize=12)
        ax.set_ylabel('Class', fontsize=12)
        ax.set_title('Per-Class F1 vs JNR', fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'jnr_per_class_f1.png'), dpi=150)
        plt.close(fig)

        print(f"JNR curves saved to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """1D CZSL evaluation entry.

    参数优先级: CLI > config.evaluation > 内置默认值。
    在 conformer_1d/config_1d.yaml 的 evaluation: 中写好 checkpoint/mode/split 等后，
    可直接: python conformer_1d/evaluate_conformer.py
    """
    parser = argparse.ArgumentParser(description="1D CZSL Model Evaluation (Conformer / ResNet1D / CNN1D)")
    parser.add_argument("--config", type=str, default="conformer_1d/config_1d.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (defaults to evaluation.checkpoint)")
    parser.add_argument("--backbone", type=str, default=None,
                        choices=["conformer", "resnet1d", "cnn1d"],
                        help="Override config model.backbone (auto-detected if not set)")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["zero_shot", "by_combination", "by_jnr"],
                        help="Evaluation mode (overrides config)")
    parser.add_argument("--split", type=str, default=None,
                        choices=["train", "val", "test"],
                        help="Data split (overrides config)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (overrides config)")
    parser.add_argument("--debug", action="store_true", default=None,
                        help="Print per-sample debug info")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    eval_cfg = config.get("evaluation", {}) or {}

    # ── Resolve params: CLI > config.evaluation > default ──
    checkpoint_path = args.checkpoint or eval_cfg.get("checkpoint")
    if not checkpoint_path:
        raise ValueError(
            "No checkpoint specified. Set --checkpoint or add evaluation.checkpoint to config.yaml"
        )

    mode = args.mode or eval_cfg.get("mode", "by_combination")
    split = args.split or eval_cfg.get("split", "test")
    output_dir_str = args.output_dir or eval_cfg.get("output_dir", "results/conformer_1d")
    backbone = args.backbone or eval_cfg.get("backbone")
    if args.debug is not None:
        debug = args.debug
    else:
        debug = bool(eval_cfg.get("debug", False))

    args.checkpoint = checkpoint_path
    args.mode = mode
    args.split = split
    args.output_dir = output_dir_str
    args.backbone = backbone
    args.debug = debug

    print(f"Evaluation config → checkpoint={args.checkpoint}")
    print(f"  mode={args.mode}, split={args.split}, output_dir={args.output_dir}, "
          f"backbone={args.backbone}, debug={args.debug}")

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Override backbone from CLI / evaluation config if specified
    if args.backbone is not None:
        config.setdefault('model', {})['backbone'] = args.backbone
        print(f"Backbone override: {args.backbone}")

    # Create model (backbone from config)
    print("\nBuilding model...")
    model = create_1d_model(config, str(device))

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Filter shape-mismatched keys
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)
    if skipped:
        print(f"  Skipped {len(skipped)} mismatched keys: {skipped[:5]}...")
    model.load_state_dict(filtered, strict=False)
    model.eval()
    print(f"  Loaded {len(filtered)} / {len(state_dict)} keys")

    # Cache text features: singles + seen ∪ unseen combinations (align multi)
    czsl_config = config.get('czsl', {})
    seen_combos = czsl_config.get('seen_combinations', []) or []
    unseen_combos = czsl_config.get('unseen_combinations', []) or []
    all_comb_names = []
    _seen_keys = set()
    for c in list(seen_combos) + list(unseen_combos):
        key = tuple(sorted(c))
        if key not in _seen_keys:
            _seen_keys.add(key)
            all_comb_names.append(c)
    print(f"  CZSL text cache: {len(seen_combos)} seen + {len(unseen_combos)} unseen "
          f"→ {len(all_comb_names)} unique entries (incl. singles in list)")
    model.cache_text_features(
        max_combination_size=2,
        include_single=True,
        seen_combinations=all_comb_names if all_comb_names else None,
        use_translation=config.get('use_translation', False),
    )
    print(f"  Cached {len(model._combination_names)} text features")

    # Create evaluator
    evaluator = ConformerEvaluator(model, config, device)

    out_dir = args.output_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Run evaluation
    if args.mode == "by_jnr":
        print(f"\nLoading {args.split} data per JNR...")
        jnr_loaders = create_1d_jnr_dataloaders(config, split=args.split)
        jnr_results = evaluator.evaluate_by_jnr(jnr_loaders)

        # Print summary (rates only; no global mixed with zero_shot)
        print_jnr_metrics_table(jnr_results, title=f"Per-JNR Results (split={args.split})")
        jnr_json = {
            str(jnr): {
                k: v for k, v in m.items()
                if k not in ("all_labels", "all_preds", "all_probs",
                             "per_class_recall", "per_class_precision", "per_class_f1")
            }
            for jnr, m in jnr_results.items()
        }
        save_metrics_json(
            {"per_jnr": jnr_json}, out_dir,
            filename=f"metrics_by_jnr_{args.split}.json",
            mode="by_jnr", split=args.split,
            extra_meta={"checkpoint": args.checkpoint},
        )

        # Plot
        evaluator.plot_jnr_curve(jnr_results, out_dir)

    elif args.mode == "zero_shot":
        # JNR lazy-load → global concat (align multi / persistence)
        print(f"\nZero-Shot Evaluation on {args.split} (JNR lazy-load)...")
        metrics, labels, preds, probs = evaluator.evaluate_zero_shot(
            config=config, split=args.split,
            debug=args.debug, output_dir=out_dir,
        )
        evaluator.print_report(metrics, f"Zero-Shot Evaluation ({args.split})")
        save_metrics_json(
            metrics, out_dir,
            filename=f"metrics_zero_shot_{args.split}.json",
            mode="zero_shot", split=args.split,
            extra_meta={"checkpoint": args.checkpoint},
        )
    else:
        print(f"\nLoading {args.split} data...")
        train_loader, val_loader, test_loader = create_1d_dataloaders(config)
        if args.split == 'test':
            loader = test_loader
        elif args.split == 'val':
            loader = val_loader
        else:
            loader = train_loader

        metrics, labels, preds, probs = evaluator.evaluate_by_combination(
            loader, debug=args.debug, output_dir=out_dir
        )
        evaluator.print_report(metrics, f"By-Combination Evaluation ({args.split})")
        save_metrics_json(
            {
                "global": metrics.get("global", metrics),
                "seen": metrics.get("seen", {}),
                "unseen": metrics.get("unseen", {}),
            },
            out_dir,
            filename=f"metrics_by_combination_{args.split}.json",
            mode="by_combination", split=args.split,
            extra_meta={"checkpoint": args.checkpoint},
        )

    print(f"\nDone!")


if __name__ == "__main__":
    main()
