"""
多形状 Patch ViT 评估脚本 — 支持多模式评估和可视化

模式:
  zero_shot      — 零样本组合识别评估
  by_combination — Seen/Unseen 组合分类评估
  by_jnr         — 按 JNR 级别分别评估

用法:
  python -m multi.evaluate_multishape_vit --checkpoint checkpoints/multishape_vit_best.pt --mode zero_shot
  python -m multi.evaluate_multishape_vit --checkpoint checkpoints/multishape_vit_best.pt --mode by_jnr --visualize
"""
import os
import sys
import yaml
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi.rectangular_patch_vit import MultiShapePatchViTForCZSL, create_multi_shape_patch_model
from multi.data import create_czsl_dataloaders
from multi.evaluate_czsl import (
    plot_roc_curves, plot_pr_curves, create_jnr_dataloaders, convert_combination_names_to_indices
)


class MultiShapeViTEvaluator:
    """多形状 Patch ViT 评估器"""

    def __init__(
        self,
        model: MultiShapePatchViTForCZSL,
        device: torch.device,
        class_names: list,
        seen_combinations: list = None,
        unseen_combinations: list = None,
    ):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.seen_combinations = seen_combinations or []
        self.unseen_combinations = unseen_combinations or []

    # ── 模式 1: 零样本评估 ──────────────────────────────────────
    @torch.no_grad()
    def evaluate_zero_shot(
        self,
        data_loader,
        use_combinations: bool = True,
        debug: bool = False,
    ) -> dict:
        self.model.eval()

        all_labels = []
        all_preds = []
        all_features = []
        all_combination_correct = 0
        all_multilabel_correct = 0
        total_samples = 0

        # Seen / Unseen 统计
        seen_set = set(tuple(sorted(c)) for c in self.seen_combinations)
        unseen_set = set(tuple(sorted(c)) for c in self.unseen_combinations)
        has_seen_unseen = bool(seen_set or unseen_set)
        seen_correct, seen_total = 0, 0
        unseen_correct, unseen_total = 0, 0
        other_correct, other_total = 0, 0

        eval_bar = tqdm(data_loader, desc="Zero-Shot Evaluation")

        if debug:
            print("\n" + "=" * 80)
            print("[DEBUG] Cached text features for inference:")
            print("=" * 80)
            cached_names = self.model._combination_names if use_combinations else self.class_names
            for i, name in enumerate(cached_names[:20]):
                print(f"  [{i}] {name}")
            if len(cached_names) > 20:
                print(f"  ... ({len(cached_names) - 20} more)")
            print("=" * 80 + "\n")

        debug_done = False
        for batch_idx, (images, _, text_tokens, labels, texts, metas, *_) in enumerate(eval_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            if images.shape[-1] != 224:
                images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

            batch_size = images.shape[0]
            image_features = self.model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            all_features.append(image_features.cpu().numpy())

            if use_combinations:
                similarities, indices, pred_names = self.model.zero_shot_predict(
                    images, use_combinations=True, top_k=1
                )
                preds = torch.zeros(batch_size, len(self.class_names), device=self.device)
                for i, name in enumerate(pred_names):
                    if name and len(name) > 0:
                        parts = name[0].split('+')
                        for part in parts:
                            if part in self.class_names:
                                preds[i, self.class_names.index(part)] = 1
            else:
                text_features = self.model.get_cached_text_features()
                logit_scale = self.model.logit_scale.exp()
                logits = logit_scale * (image_features @ text_features.T)
                probs = torch.softmax(logits, dim=-1)
                threshold = 1.0 / self.num_classes
                topk_values, topk_indices = torch.topk(probs, k=3, dim=-1)
                preds = torch.zeros(batch_size, self.num_classes, device=self.device)
                for b in range(batch_size):
                    for j, idx in enumerate(topk_indices[b]):
                        if topk_values[b, j] > threshold:
                            preds[b, idx] = 1.0

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

            if debug and not debug_done:
                print("\n" + "=" * 80)
                print(f"[DEBUG] Batch {batch_idx} — batch_size={batch_size}")
                print("Predictions vs Labels:")
                for i in range(min(5, batch_size)):
                    true_idx = torch.where(labels[i] == 1)[0].tolist()
                    pred_idx = torch.where(preds[i] == 1)[0].tolist()
                    true_n = [self.class_names[j] for j in true_idx]
                    pred_n = [self.class_names[j] for j in pred_idx]
                    print(f"  [{i}] True: {true_n} | Pred: {pred_n}")
                print("=" * 80 + "\n")
                debug_done = True

            for i in range(batch_size):
                true_set = set(torch.where(labels[i] == 1)[0].tolist())
                pred_set = set(torch.where(preds[i] == 1)[0].tolist())
                if true_set == pred_set:
                    all_combination_correct += 1
                if true_set.issubset(pred_set) or pred_set.issubset(true_set):
                    all_multilabel_correct += 1

                # 按 seen/unseen/other 分类统计
                if has_seen_unseen:
                    true_comb = tuple(sorted(true_set))
                    is_correct = (true_set == pred_set)
                    if true_comb in seen_set:
                        seen_total += 1
                        if is_correct:
                            seen_correct += 1
                    elif true_comb in unseen_set:
                        unseen_total += 1
                        if is_correct:
                            unseen_correct += 1
                    else:
                        other_total += 1
                        if is_correct:
                            other_correct += 1
            total_samples += batch_size

        all_labels_np = torch.cat(all_labels).numpy()
        all_preds_np = torch.cat(all_preds).numpy()
        all_features_np = np.concatenate(all_features, axis=0)

        metrics = {
            "combination_accuracy": all_combination_correct / total_samples,
            "partial_match_accuracy": all_multilabel_correct / total_samples,
            "f1_macro": f1_score(all_labels_np, all_preds_np, average='macro', zero_division=0),
            "f1_micro": f1_score(all_labels_np, all_preds_np, average='micro', zero_division=0),
            "precision_macro": precision_score(all_labels_np, all_preds_np, average='macro', zero_division=0),
            "recall_macro": recall_score(all_labels_np, all_preds_np, average='macro', zero_division=0),
        }
        # 添加 seen/unseen 准确率
        if has_seen_unseen:
            metrics["seen_accuracy"] = seen_correct / seen_total if seen_total > 0 else 0.0
            metrics["seen_samples"] = seen_total
            metrics["unseen_accuracy"] = unseen_correct / unseen_total if unseen_total > 0 else 0.0
            metrics["unseen_samples"] = unseen_total
            if other_total > 0:
                metrics["other_accuracy"] = other_correct / other_total
                metrics["other_samples"] = other_total
            if metrics["seen_accuracy"] + metrics["unseen_accuracy"] > 0:
                metrics["harmonic_mean"] = (2 * metrics["seen_accuracy"] * metrics["unseen_accuracy"]
                                            / (metrics["seen_accuracy"] + metrics["unseen_accuracy"]))
            else:
                metrics["harmonic_mean"] = 0.0

        per_class_f1 = f1_score(all_labels_np, all_preds_np, average=None, zero_division=0)
        metrics["per_class"] = {
            "f1": per_class_f1,
            "precision": precision_score(all_labels_np, all_preds_np, average=None, zero_division=0),
            "recall": recall_score(all_labels_np, all_preds_np, average=None, zero_division=0),
        }
        return {"metrics": metrics, "labels": all_labels_np, "predictions": all_preds_np, "features": all_features_np}

    # ── 模式 2: 按组合类型评估 ──────────────────────────────────
    @torch.no_grad()
    def evaluate_by_combination_type(
        self,
        data_loader,
        debug: bool = False,
    ) -> dict:
        self.model.eval()

        seen_set = set(tuple(sorted(c)) for c in self.seen_combinations)
        unseen_set = set(tuple(sorted(c)) for c in self.unseen_combinations)

        seen_results = {"correct": 0, "total": 0}
        unseen_results = {"correct": 0, "total": 0}
        other_results = {"correct": 0, "total": 0}

        all_labels = []
        all_preds = []
        all_features = []
        all_probs = []

        eval_bar = tqdm(data_loader, desc="Evaluating by Combination Type")

        for batch_idx, (images, _, text_tokens, labels, texts, metas, *_) in enumerate(eval_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            if images.shape[-1] != 224:
                images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

            batch_size = images.shape[0]
            text_features = self.model.get_cached_text_features()
            image_features = self.model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            logit_scale = self.model.logit_scale.exp()
            logits = logit_scale * (image_features @ text_features.T)
            probs = torch.softmax(logits, dim=-1)

            threshold = 1.0 / self.num_classes
            topk_values, topk_indices = torch.topk(probs, k=3, dim=-1)
            preds = torch.zeros(batch_size, self.num_classes, device=self.device)
            for b in range(batch_size):
                for j, idx in enumerate(topk_indices[b]):
                    if topk_values[b, j] > threshold:
                        preds[b, idx] = 1.0

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_features.append(image_features.cpu().numpy())

            for i in range(batch_size):
                true_comb = tuple(sorted(torch.where(labels[i] == 1)[0].tolist()))
                pred_comb = tuple(sorted(torch.where(preds[i] == 1)[0].tolist()))
                is_correct = (true_comb == pred_comb)

                if true_comb in seen_set:
                    seen_results["total"] += 1
                    if is_correct:
                        seen_results["correct"] += 1
                elif true_comb in unseen_set:
                    unseen_results["total"] += 1
                    if is_correct:
                        unseen_results["correct"] += 1
                else:
                    other_results["total"] += 1
                    if is_correct:
                        other_results["correct"] += 1

        all_labels_np = torch.cat(all_labels).numpy()
        all_preds_np = torch.cat(all_preds).numpy()
        all_probs_np = torch.cat(all_probs).numpy()
        all_features_np = np.concatenate(all_features, axis=0)

        metrics = {
            "seen_accuracy": seen_results["correct"] / seen_results["total"] if seen_results["total"] > 0 else 0,
            "seen_samples": seen_results["total"],
            "unseen_accuracy": unseen_results["correct"] / unseen_results["total"] if unseen_results["total"] > 0 else 0,
            "unseen_samples": unseen_results["total"],
            "other_accuracy": other_results["correct"] / other_results["total"] if other_results["total"] > 0 else 0,
            "other_samples": other_results["total"],
        }
        return {"metrics": metrics, "labels": all_labels_np, "predictions": all_preds_np,
                "probabilities": all_probs_np, "features": all_features_np}

    # ── 模式 3: 按 JNR 评估 ──────────────────────────────────────
    @torch.no_grad()
    def evaluate_by_jnr(self, jnr_loaders: dict) -> dict:
        seen_set = set(tuple(sorted(c)) for c in self.seen_combinations)
        unseen_set = set(tuple(sorted(c)) for c in self.unseen_combinations)
        results = {}

        for jnr, data_loader in sorted(jnr_loaders.items()):
            print(f"\nEvaluating JNR={jnr}...")

            all_labels = []
            all_preds = []
            all_probs = []
            seen_correct, seen_total = 0, 0
            unseen_correct, unseen_total = 0, 0
            class_stats = np.zeros((self.num_classes, 4), dtype=np.int64)

            for images, _, _, labels, _, _, *_ in tqdm(data_loader, desc=f"JNR={jnr}"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                if images.shape[-1] != 224:
                    images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

                batch_size = images.shape[0]
                text_features = self.model.get_cached_text_features()
                image_features = self.model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logit_scale = self.model.logit_scale.exp()
                logits = logit_scale * (image_features @ text_features.T)
                probs = torch.softmax(logits, dim=-1)

                threshold = 1.0 / self.num_classes
                topk_values, topk_indices = torch.topk(probs, k=3, dim=-1)
                preds = torch.zeros(batch_size, self.num_classes, device=self.device)
                for b in range(batch_size):
                    for j, idx in enumerate(topk_indices[b]):
                        if topk_values[b, j] > threshold:
                            preds[b, idx] = 1.0

                all_labels.append(labels.cpu())
                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())

                for c in range(self.num_classes):
                    true_c = labels[:, c].cpu().numpy()
                    pred_c = preds[:, c].cpu().numpy()
                    class_stats[c, 0] += np.sum((true_c == 1) & (pred_c == 1))
                    class_stats[c, 1] += np.sum((true_c == 0) & (pred_c == 1))
                    class_stats[c, 2] += np.sum((true_c == 1) & (pred_c == 0))
                    class_stats[c, 3] += np.sum((true_c == 0) & (pred_c == 0))

                for i in range(batch_size):
                    true_comb = tuple(sorted(torch.where(labels[i] == 1)[0].tolist()))
                    pred_comb = tuple(sorted(torch.where(preds[i] == 1)[0].tolist()))
                    is_correct = (true_comb == pred_comb)
                    if true_comb in seen_set:
                        seen_total += 1
                        if is_correct:
                            seen_correct += 1
                    elif true_comb in unseen_set:
                        unseen_total += 1
                        if is_correct:
                            unseen_correct += 1

            all_labels_np = torch.cat(all_labels).numpy()
            all_preds_np = torch.cat(all_preds).numpy()
            all_probs_np = torch.cat(all_probs).numpy()

            per_class_recall = np.zeros(self.num_classes)
            per_class_precision = np.zeros(self.num_classes)
            per_class_f1 = np.zeros(self.num_classes)
            for c in range(self.num_classes):
                tp, fp, fn, tn = class_stats[c]
                per_class_recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0
                per_class_precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0
                if per_class_precision[c] + per_class_recall[c] > 0:
                    per_class_f1[c] = 2 * per_class_precision[c] * per_class_recall[c] / (
                        per_class_precision[c] + per_class_recall[c])

            results[jnr] = {
                "combination_accuracy": seen_correct + unseen_correct,
                "total_samples": seen_total + unseen_total,
                "seen_accuracy": seen_correct / seen_total if seen_total > 0 else 0,
                "seen_samples": seen_total,
                "unseen_accuracy": unseen_correct / unseen_total if unseen_total > 0 else 0,
                "unseen_samples": unseen_total,
                "f1_macro": f1_score(all_labels_np, all_preds_np, average='macro', zero_division=0),
                "f1_micro": f1_score(all_labels_np, all_preds_np, average='micro', zero_division=0),
                "precision_macro": precision_score(all_labels_np, all_preds_np, average='macro', zero_division=0),
                "recall_macro": recall_score(all_labels_np, all_preds_np, average='macro', zero_division=0),
                "labels": all_labels_np,
                "probabilities": all_probs_np,
                "per_class_recall": per_class_recall,
                "per_class_precision": per_class_precision,
                "per_class_f1": per_class_f1,
            }
        return results

    # ── 打印 ─────────────────────────────────────────────────────
    def print_metrics(self, metrics: dict):
        print("\n" + "=" * 60)
        print("Zero-Shot Evaluation Results")
        print("=" * 60)

        # Seen / Unseen accuracy (CZSL 核心指标)
        if 'seen_accuracy' in metrics:
            print(f"  Seen Accuracy:           {metrics['seen_accuracy']:.4f}  ({metrics.get('seen_samples', 0)} samples)")
            print(f"  Unseen Accuracy:         {metrics['unseen_accuracy']:.4f}  ({metrics.get('unseen_samples', 0)} samples)")
            if 'other_accuracy' in metrics:
                print(f"  Other Accuracy:          {metrics['other_accuracy']:.4f}  ({metrics.get('other_samples', 0)} samples)")
            if 'harmonic_mean' in metrics:
                print(f"  Harmonic Mean:           {metrics['harmonic_mean']:.4f}")
            print(f"  {'-'*50}")

        print(f"Combination Accuracy:     {metrics['combination_accuracy']:.4f}")
        print(f"Partial Match Accuracy:   {metrics['partial_match_accuracy']:.4f}")
        print(f"Macro F1 Score:           {metrics['f1_macro']:.4f}")
        print(f"Micro F1 Score:           {metrics['f1_micro']:.4f}")
        print(f"Macro Precision:          {metrics['precision_macro']:.4f}")
        print(f"Macro Recall:             {metrics['recall_macro']:.4f}")
        print("\nPer-class Metrics:")
        print("-" * 60)
        print(f"{'Class':<10} {'F1':>8} {'Precision':>12} {'Recall':>10}")
        print("-" * 60)
        for i, name in enumerate(self.class_names[:len(metrics['per_class']['f1'])]):
            print(f"{name:<10} {metrics['per_class']['f1'][i]:>8.4f} "
                  f"{metrics['per_class']['precision'][i]:>12.4f} "
                  f"{metrics['per_class']['recall'][i]:>10.4f}")

    def print_jnr_results(self, results: dict, save_path: str = None):
        print("\n" + "=" * 80)
        print("Evaluation Results by JNR Level")
        print("=" * 80)
        print(f"{'JNR':>6} | {'Accuracy':>10} | {'F1_Macro':>10} | "
              f"{'Seen_Acc':>10} | {'Unseen_Acc':>10} | {'Samples':>8}")
        print("-" * 80)

        lines = []
        for jnr, m in sorted(results.items()):
            acc = (m["combination_accuracy"] / m["total_samples"] if m["total_samples"] > 0 else 0)
            print(f"{jnr:>6} | {acc:>10.4f} | {m['f1_macro']:>10.4f} | "
                  f"{m['seen_accuracy']:>10.4f} | {m['unseen_accuracy']:>10.4f} | {m['total_samples']:>8}")
            lines.append(f"{jnr},{acc:.4f},{m['f1_macro']:.4f},{m['seen_accuracy']:.4f},{m['unseen_accuracy']:.4f},{m['total_samples']}")

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("JNR,Accuracy,F1_Macro,Seen_Acc,Unseen_Acc,Samples\n")
                f.write("\n".join(lines))
            print(f"\nResults saved to {save_path}")

        # Per-class table
        print("\n" + "=" * 80)
        print("Per-Class Recall by JNR Level")
        print("=" * 80)
        header = f"{'JNR':>6} | " + " | ".join(f"{name:>8}" for name in self.class_names)
        print(header)
        print("-" * len(header))
        for jnr, m in sorted(results.items()):
            recalls = m.get("per_class_recall", [])
            print(f"{jnr:>6} | " + " | ".join(f"{r:>8.4f}" for r in recalls))

    # ── 可视化 ────────────────────────────────────────────────────
    def plot_confusion_by_combination(
        self, labels: np.ndarray, preds: np.ndarray,
        save_path: str = None, seen_combinations: list = None, unseen_combinations: list = None
    ):
        configured_combs = []
        seen_list = seen_combinations or self.seen_combinations
        unseen_list = unseen_combinations or self.unseen_combinations
        for comb in seen_list:
            t = tuple(sorted(comb))
            if t not in configured_combs:
                configured_combs.append(t)
        for comb in unseen_list:
            t = tuple(sorted(comb))
            if t not in configured_combs:
                configured_combs.append(t)
        unique_combs = configured_combs

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

        seen_set = set(tuple(sorted(c)) for c in seen_list)
        unseen_set = set(tuple(sorted(c)) for c in unseen_list)

        comb_names = []
        for comb in unique_combs:
            name = "+".join([self.class_names[i] for i in comb]) if comb else "None"
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
        ax.set_title('Combination Confusion Matrix (MultiShape ViT)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_feature_tsne(
        self, features: np.ndarray, labels: np.ndarray,
        save_path: str = None, seen_combinations: list = None, unseen_combinations: list = None
    ):
        print("Computing t-SNE projection...")
        num_samples = len(features)
        perplexity = min(30, num_samples - 1) if num_samples > 1 else 1

        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        features_2d = tsne.fit_transform(features)

        comb_labels = [tuple(sorted(np.where(label == 1)[0].tolist())) for label in labels]

        seen_list = seen_combinations or self.seen_combinations
        unseen_list = unseen_combinations or self.unseen_combinations
        seen_set = set(tuple(sorted(c)) for c in seen_list)
        unseen_set = set(tuple(sorted(c)) for c in unseen_list)

        unique_combs = sorted(set(comb_labels))
        comb_to_name = {}
        for comb in unique_combs:
            comb_to_name[comb] = "+".join([self.class_names[i] for i in comb]) if comb else "None"

        num_combs = len(unique_combs)
        colors = plt.cm.tab20(np.linspace(0, 1, max(20, num_combs)))

        fig, ax = plt.subplots(figsize=(14, 10))
        for idx, comb in enumerate(unique_combs):
            mask = np.array([c == comb for c in comb_labels])
            if mask.sum() > 0:
                name = comb_to_name[comb]
                if comb in seen_set:
                    name, marker = f"[S] {name}", 'o'
                elif comb in unseen_set:
                    name, marker = f"[U] {name}", '^'
                else:
                    marker = 's'
                ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                          c=[colors[idx % 20]], label=name, alpha=0.6, s=30, marker=marker)

        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title('Feature Space (t-SNE) — MultiShape ViT\n[S]=Seen, [U]=Unseen')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"t-SNE plot saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_feature_umap(
        self, features: np.ndarray, labels: np.ndarray,
        save_path: str = None, seen_combinations: list = None, unseen_combinations: list = None
    ):
        try:
            import umap
        except ImportError:
            print("UMAP not installed. Install with: pip install umap-learn")
            return

        print("Computing UMAP projection...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        features_2d = reducer.fit_transform(features)

        comb_labels = [tuple(sorted(np.where(label == 1)[0].tolist())) for label in labels]

        seen_list = seen_combinations or self.seen_combinations
        unseen_list = unseen_combinations or self.unseen_combinations
        seen_set = set(tuple(sorted(c)) for c in seen_list)
        unseen_set = set(tuple(sorted(c)) for c in unseen_list)

        unique_combs = sorted(set(comb_labels))
        comb_to_name = {}
        for comb in unique_combs:
            comb_to_name[comb] = "+".join([self.class_names[i] for i in comb]) if comb else "None"

        num_combs = len(unique_combs)
        colors = plt.cm.tab20(np.linspace(0, 1, max(20, num_combs)))

        fig, ax = plt.subplots(figsize=(14, 10))
        for idx, comb in enumerate(unique_combs):
            mask = np.array([c == comb for c in comb_labels])
            if mask.sum() > 0:
                name = comb_to_name[comb]
                if comb in seen_set:
                    name, marker = f"[S] {name}", 'o'
                elif comb in unseen_set:
                    name, marker = f"[U] {name}", '^'
                else:
                    marker = 's'
                ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                          c=[colors[idx % 20]], label=name, alpha=0.6, s=30, marker=marker)

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('Feature Space (UMAP) — MultiShape ViT\n[S]=Seen, [U]=Unseen')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"UMAP plot saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_label_cooccurrence(self, labels: np.ndarray, preds: np.ndarray, save_path: str = None):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        num_classes = labels.shape[1]
        true_cooc = (labels.T @ labels).astype(int)
        sns.heatmap(true_cooc, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names[:num_classes],
                    yticklabels=self.class_names[:num_classes], ax=axes[0])
        axes[0].set_title('True Label Co-occurrence')
        axes[0].tick_params(axis='x', rotation=45)

        pred_cooc = (preds.T @ preds).astype(int)
        sns.heatmap(pred_cooc, annot=True, fmt='d', cmap='Greens',
                    xticklabels=self.class_names[:num_classes],
                    yticklabels=self.class_names[:num_classes], ax=axes[1])
        axes[1].set_title('Predicted Label Co-occurrence')
        axes[1].tick_params(axis='x', rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Co-occurrence plot saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_jnr_metrics(self, results: dict, save_path: str = None):
        jnrs = sorted(results.keys())
        accuracies = []
        f1_macros = []
        seen_accs = []
        unseen_accs = []

        for jnr in jnrs:
            m = results[jnr]
            acc = (m["combination_accuracy"] / m["total_samples"] if m["total_samples"] > 0 else 0)
            accuracies.append(acc)
            f1_macros.append(m["f1_macro"])
            seen_accs.append(m["seen_accuracy"])
            unseen_accs.append(m["unseen_accuracy"])

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        axes[0, 0].plot(jnrs, accuracies, 'b-o', label='Overall Accuracy', linewidth=2)
        axes[0, 0].plot(jnrs, seen_accs, 'g-s', label='Seen Accuracy', linewidth=2)
        axes[0, 0].plot(jnrs, unseen_accs, 'r-^', label='Unseen Accuracy', linewidth=2)
        axes[0, 0].set_xlabel('JNR (dB)')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy vs JNR Level')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(jnrs, f1_macros, 'b-o', label='F1 Macro', linewidth=2)
        axes[0, 1].set_xlabel('JNR (dB)')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('F1 Score vs JNR Level')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        ax = axes[1, 0]
        for c, name in enumerate(self.class_names):
            recalls = [results[jnr].get("per_class_recall", [0] * self.num_classes)[c] for jnr in jnrs]
            ax.plot(jnrs, recalls, '-o', label=name, linewidth=1.5, markersize=4)
        ax.set_xlabel('JNR (dB)')
        ax.set_ylabel('Recall')
        ax.set_title('Per-Class Recall vs JNR Level')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        for c, name in enumerate(self.class_names):
            f1s = [results[jnr].get("per_class_f1", [0] * self.num_classes)[c] for jnr in jnrs]
            ax.plot(jnrs, f1s, '-o', label=name, linewidth=1.5, markersize=4)
        ax.set_xlabel('JNR (dB)')
        ax.set_ylabel('F1 Score')
        ax.set_title('Per-Class F1 Score vs JNR Level')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"JNR metrics plot saved to {save_path}")
        else:
            plt.show()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Multi-Shape Patch ViT Evaluation")
    parser.add_argument("--config", type=str, default="multi/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "by_jnr"],
                        help="Evaluation mode: 'all' runs zero-shot + by-combination, 'by_jnr' runs per-JNR")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=str, default="results/multishape_vit")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate all visualizations (confusion matrix, ROC, PR, t-SNE, UMAP)")
    parser.add_argument("--tsne", action="store_true",
                        help="Generate t-SNE visualization only")
    parser.add_argument("--umap", action="store_true",
                        help="Generate UMAP visualization only")
    parser.add_argument("--roc", action="store_true",
                        help="Generate ROC curves only")
    parser.add_argument("--pr", action="store_true",
                        help="Generate PR curves only")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载检查点
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # 模型架构配置: 使用 checkpoint 中保存的配置（保证结构匹配）
    if "config" in checkpoint:
        config = checkpoint["config"]
        print("  Model architecture from checkpoint config")
    else:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("  Model architecture from file config (no config in checkpoint)")

    # 数据路径: 始终从当前 YAML 文件读取（可能已变更）
    with open(args.config, 'r', encoding='utf-8') as f:
        file_config = yaml.safe_load(f)
    config["data"] = file_config.get("data", config.get("data", {}))
    print(f"  Data config from {args.config}")

    # 确保模型配置存在
    model_config = config.get("model", {})
    model_config.setdefault("patch_sizes", [(8, 32), (32, 8), (16, 16)])
    model_config.setdefault("fusion_mode", "early_fusion")
    model_config.setdefault("embed_dim", 512)
    model_config.setdefault("depth", 6)
    model_config.setdefault("num_heads", 8)
    config["model"] = model_config

    # 类别名称
    class_names = [cls["name"] for cls in config.get("jamming_classes", [])]

    # CZSL 组合
    czsl_config = config.get("czsl", {})
    seen_comb_names = czsl_config.get("seen_combinations", [])
    unseen_comb_names = czsl_config.get("unseen_combinations", [])
    seen_combinations = convert_combination_names_to_indices(seen_comb_names, class_names)
    unseen_combinations = convert_combination_names_to_indices(unseen_comb_names, class_names)

    # 创建模型
    print("\nCreating model...")
    model = create_multi_shape_patch_model(config, device=str(device))
    model.load_state_dict(checkpoint["model_state_dict"])

    # 缓存文本特征
    model.cache_text_features(max_combination_size=2, include_single=True,
                              seen_combinations=seen_combinations)

    # 评估器
    evaluator = MultiShapeViTEvaluator(
        model=model, device=device, class_names=class_names,
        seen_combinations=seen_combinations, unseen_combinations=unseen_combinations,
    )

    # ── 按模式执行 ────────────────────────────────────────────
    if args.mode == "all":
        # 加载数据
        print(f"\nLoading {args.split} dataset...")
        train_loader, val_loader, test_loader, _ = create_czsl_dataloaders(
            config=config,
            batch_size=config.get("train", {}).get("batch_size", 16),
            num_workers=config.get("data", {}).get("num_workers", 4),
            pin_memory=config.get("data", {}).get("pin_memory", True),
            load_test=(args.split == "test"),
        )
        data_loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]
        if data_loader is None:
            print(f"Error: No {args.split} data available!")
            return

        # ── 1) Zero-Shot (组合特征匹配) ──
        print(f"\n{'='*60}")
        print(f"  [1/2] Zero-Shot Evaluation (combination matching)")
        print(f"{'='*60}")
        results_zs = evaluator.evaluate_zero_shot(data_loader, use_combinations=True, debug=True)
        evaluator.print_metrics(results_zs["metrics"])

        np.savez(str(output_dir / f"multishape_vit_zeroshot_{args.split}.npz"),
                 labels=results_zs["labels"], predictions=results_zs["predictions"],
                 features=results_zs["features"])

        if args.visualize:
            evaluator.plot_confusion_by_combination(
                results_zs["labels"], results_zs["predictions"],
                save_path=str(output_dir / f"multishape_vit_confusion_zs_{args.split}.png"),
            )
        if args.visualize:
            evaluator.plot_label_cooccurrence(
                results_zs["labels"], results_zs["predictions"],
                save_path=str(output_dir / f"multishape_vit_cooccurrence_{args.split}.png"),
            )
        if args.tsne:
            print("\nGenerating t-SNE visualization...")
            evaluator.plot_feature_tsne(
                results_zs["features"], results_zs["labels"],
                save_path=str(output_dir / f"multishape_vit_tsne_{args.split}.png"),
            )
        if args.umap:
            print("\nGenerating UMAP visualization...")
            evaluator.plot_feature_umap(
                results_zs["features"], results_zs["labels"],
                save_path=str(output_dir / f"multishape_vit_umap_{args.split}.png"),
            )

        # ── 2) By-Combination (单类特征 + softmax) ──
        print(f"\n{'='*60}")
        print(f"  [2/2] By-Combination Evaluation (single-class + softmax)")
        print(f"{'='*60}")
        results_bc = evaluator.evaluate_by_combination_type(data_loader)
        metrics_bc = results_bc["metrics"]
        print(f"  Seen Accuracy:    {metrics_bc['seen_accuracy']:.4f} ({metrics_bc['seen_samples']} samples)")
        print(f"  Unseen Accuracy:  {metrics_bc['unseen_accuracy']:.4f} ({metrics_bc['unseen_samples']} samples)")
        print(f"  Other Accuracy:   {metrics_bc['other_accuracy']:.4f} ({metrics_bc.get('other_samples', 0)} samples)")
        print(f"{'='*60}")

        np.savez(str(output_dir / f"multishape_vit_bycombo_{args.split}.npz"),
                 labels=results_bc["labels"], predictions=results_bc["predictions"],
                 features=results_bc["features"])

        if args.visualize:
            evaluator.plot_confusion_by_combination(
                results_bc["labels"], results_bc["predictions"],
                save_path=str(output_dir / f"multishape_vit_confusion_bc_{args.split}.png"),
            )
        if args.visualize or args.roc:
            plot_roc_curves(
                results_bc["labels"], results_bc["probabilities"], class_names,
                save_dir=str(output_dir), prefix=f"multishape_vit_{args.split}_byc",
                mode_title=f"by_combination ({args.split})"
            )
        if args.visualize or args.pr:
            plot_pr_curves(
                results_bc["labels"], results_bc["probabilities"], class_names,
                save_dir=str(output_dir), prefix=f"multishape_vit_{args.split}_byc",
                mode_title=f"by_combination ({args.split})"
            )

    elif args.mode == "by_jnr":
        print(f"\nEvaluating by JNR level on {args.split} set...")

        # 加载归一化统计量
        stats_file = os.path.join(os.path.dirname(args.config), 'normalization_stats.json')
        normalization_stats = None
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                normalization_stats = json.load(f)

        jnr_loaders = create_jnr_dataloaders(
            config=config, split=args.split,
            normalization_stats=normalization_stats,
            batch_size=config.get('train', {}).get('batch_size', 32),
            num_workers=config.get('data', {}).get('num_workers', 4),
        )

        results = evaluator.evaluate_by_jnr(jnr_loaders)
        evaluator.print_jnr_results(
            results, save_path=str(output_dir / f"jnr_results_{args.split}.csv"))

        evaluator.plot_jnr_metrics(
            results, save_path=str(output_dir / f"jnr_metrics_{args.split}.png"))

        do_jnr_curves = args.visualize or args.roc or args.pr
        if do_jnr_curves:
            for jnr_val, jnr_results in sorted(results.items()):
                if "probabilities" in jnr_results and jnr_results["probabilities"].size > 0:
                    if args.visualize or args.roc:
                        plot_roc_curves(
                            jnr_results["labels"], jnr_results["probabilities"], class_names,
                            save_dir=str(output_dir), prefix=f"jnr_{jnr_val:+.0f}_{args.split}",
                            mode_title=f"JNR={jnr_val:+d}"
                        )
                    if args.visualize or args.pr:
                        plot_pr_curves(
                            jnr_results["labels"], jnr_results["probabilities"], class_names,
                            save_dir=str(output_dir), prefix=f"jnr_{jnr_val:+.0f}_{args.split}",
                            mode_title=f"JNR={jnr_val:+d}"
                        )

    # 保存评估摘要
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
