"""
统一评估脚本 - 通过 config.yaml 控制评估模式
python -m multi.evaluate_unified --checkpoint checkpoints/czsl_best_model.pt --mode zero_shot

支持模式 (train.mode in config.yaml):
  - "czsl": 对比学习评估 (CLIP/MultiShapeViT/OriginalCLIP)
  - "dual_branch": 双分支评估 (CLIP双分支/MultiShapeViT双分支/ResNet18双分支)

评估子模式 (--mode):
  - "zero_shot": 零样本评估
  - "by_combination": 按组合类型评估 (seen vs unseen)
  - "by_jnr": 按 JNR 级别评估 (仅 CZSL 模式)
"""
import os
import sys
import yaml
import argparse
import json
from abc import ABC, abstractmethod
from pathlib import Path
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.manifold import TSNE

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



# ============================================================================
# 公共函数
# ============================================================================

def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def convert_combination_names_to_indices(combinations_list: list, class_names: list) -> list:
    if not combinations_list:
        return []
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    result = []
    for comb in combinations_list:
        indices = []
        for name in comb:
            if name in name_to_idx:
                indices.append(name_to_idx[name])
            else:
                print(f"Warning: Unknown class name '{name}' in combination {comb}")
        if indices:
            result.append(sorted(indices))
    return result


def plot_roc_curves(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: list,
    save_dir: str = None,
    prefix: str = "roc",
    mode_title: str = ""
):
    n_classes = labels.shape[1]
    fpr_dict, tpr_dict, auc_dict = {}, {}, {}
    for i in range(n_classes):
        if np.sum(labels[:, i]) == 0:
            continue
        fpr_dict[i], tpr_dict[i], _ = roc_curve(labels[:, i], probabilities[:, i])
        auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

    fpr_micro, tpr_micro, _ = roc_curve(labels.ravel(), probabilities.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)

    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in fpr_dict]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in fpr_dict:
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= len(fpr_dict)
    auc_macro = auc(all_fpr, mean_tpr)

    title_suffix = f" ({mode_title})" if mode_title else ""

    # Per-class ROC
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(figsize=(8, 7))
    for idx, i in enumerate(fpr_dict):
        color = colors[idx % 10]
        ax.plot(fpr_dict[i], tpr_dict[i], color=color, linewidth=1.2,
                label=f'{class_names[i]} (AUC={auc_dict[i]:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', linewidth=1.0, linestyle=':', alpha=0.7)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
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

    # Average ROC
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr_micro, tpr_micro, color='darkorange', linewidth=2.5,
            label=f'Micro-average (AUC={auc_micro:.3f})')
    ax.plot(all_fpr, mean_tpr, color='darkgreen', linewidth=2.5, linestyle='--',
            label=f'Macro-average (AUC={auc_macro:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', linewidth=1.0, linestyle=':', alpha=0.7)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
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


# ============================================================================
# DualBranchPlotMixin: 双分支共享可视化方法
# ============================================================================

class DualBranchPlotMixin:
    """双分支评估器共享的可视化方法 (evaluate_dual_branch 与 evaluate_resnet18 完全相同)"""

    def _compute_metrics(self, labels: np.ndarray, preds: np.ndarray) -> dict:
        return {
            "f1_macro": f1_score(labels, preds, average='macro', zero_division=0),
            "f1_micro": f1_score(labels, preds, average='micro', zero_division=0),
            "precision_macro": precision_score(labels, preds, average='macro', zero_division=0),
            "recall_macro": recall_score(labels, preds, average='macro', zero_division=0),
            "accuracy": accuracy_score(labels, preds),
            "per_class_f1": f1_score(labels, preds, average=None, zero_division=0).tolist(),
            "per_class_precision": precision_score(labels, preds, average=None, zero_division=0).tolist(),
            "per_class_recall": recall_score(labels, preds, average=None, zero_division=0).tolist(),
        }

    def print_metrics(self, metrics: dict):
        print("\n" + "=" * 80)
        print(f"{self._model_label} Evaluation Results")
        print("=" * 80)
        for branch in ["deception", "suppression"]:
            b = metrics[branch]
            label = "Deception" if branch == "deception" else "Suppression"
            print(f"\n[{label} Branch]")
            print(f"  F1 Macro:    {b['f1_macro']:.4f}")
            print(f"  F1 Micro:    {b['f1_micro']:.4f}")
            print(f"  Precision:   {b['precision_macro']:.4f}")
            print(f"  Recall:      {b['recall_macro']:.4f}")
            print(f"  Accuracy:    {b['accuracy']:.4f}")
        print(f"\n[Combined]")
        print(f"  Combined Accuracy: {metrics['combined_accuracy']:.4f}")
        print(f"  Total Samples: {metrics['total_samples']}")

        for branch, classes in [("deception", self.deception_classes_with_none),
                                ("suppression", self.suppression_classes_with_none)]:
            label = "Deception" if branch == "deception" else "Suppression"
            print(f"\n[Per-Class F1 - {label}]")
            for i, name in enumerate(classes):
                f1 = metrics[branch]["per_class_f1"][i]
                print(f"  {name:<15}: {f1:.4f}")

    def plot_confusion_matrices(self, metrics: dict, save_path: str = None):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for idx, (branch, classes, cmap) in enumerate([
            ("deception", self.deception_classes_with_none, "Blues"),
            ("suppression", self.suppression_classes_with_none, "Greens"),
        ]):
            labels = metrics.get(f"labels_{branch}")
            preds = metrics.get(f"preds_{branch}")
            if labels is not None and preds is not None:
                pred_classes = np.argmax(preds, axis=1)
                true_classes = np.argmax(labels, axis=1)
                cm = confusion_matrix(true_classes, pred_classes)
                sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                            xticklabels=classes, yticklabels=classes, ax=axes[idx])
                title = "Deception" if branch == "deception" else "Suppression"
                axes[idx].set_title(f'{title} Branch Confusion Matrix')
                axes[idx].set_xlabel('Predicted')
                axes[idx].set_ylabel('True')
                axes[idx].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrices saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_per_class_f1(self, metrics: dict, save_path: str = None):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for idx, (branch, classes, cmap_name) in enumerate([
            ("deception", self.deception_classes_with_none, "Blues"),
            ("suppression", self.suppression_classes_with_none, "Greens"),
        ]):
            f1 = metrics[branch]["per_class_f1"]
            x = np.arange(len(classes))
            colors = plt.cm.get_cmap(cmap_name)(np.linspace(0.3, 0.9, len(f1)))
            bars = axes[idx].bar(x, f1, color=colors)
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(classes, rotation=45, ha='right')
            axes[idx].set_ylabel('F1 Score')
            title = "Deception" if branch == "deception" else "Suppression"
            axes[idx].set_title(f'{title} Branch - Per-Class F1 Score')
            axes[idx].set_ylim(0, 1.0)
            axes[idx].axhline(y=metrics[branch]["f1_macro"], color='r', linestyle='--',
                              label=f'Macro F1: {metrics[branch]["f1_macro"]:.4f}')
            axes[idx].legend()
            for bar, val in zip(bars, f1):
                axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                               f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Per-class F1 plot saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_metrics_summary(self, metrics: dict, save_path: str = None):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        metric_names = ['F1 Macro', 'F1 Micro', 'Precision', 'Recall', 'Accuracy']
        for idx, (branch, color, title) in enumerate([
            ("deception", "steelblue", "Deception Branch Metrics"),
            ("suppression", "seagreen", "Suppression Branch Metrics"),
        ]):
            values = [
                metrics[branch]["f1_macro"], metrics[branch]["f1_micro"],
                metrics[branch]["precision_macro"], metrics[branch]["recall_macro"],
                metrics[branch]["accuracy"]
            ]
            x = np.arange(len(metric_names))
            bars = axes[idx].bar(x, values, width=0.6, color=color)
            axes[idx].set_ylabel('Score')
            axes[idx].set_title(title)
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(metric_names, rotation=45, ha='right')
            axes[idx].set_ylim(0, 1.0)
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2, height + 0.02,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Metrics summary plot saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_combination_confusion_matrix(
        self, metrics: dict, save_path: str = None,
        seen_combinations: list = None, unseen_combinations: list = None,
        all_class_names: list = None
    ):
        labels_d = metrics.get("labels_deception")
        labels_s = metrics.get("labels_suppression")
        preds_d = metrics.get("preds_deception")
        preds_s = metrics.get("preds_suppression")
        if labels_d is None or labels_s is None:
            print("No label data available for combination confusion matrix")
            return

        name_to_idx = {name: i for i, name in enumerate(all_class_names or [])}

        def get_comb_indices(label_d, label_s):
            d_idx, s_idx = np.argmax(label_d), np.argmax(label_s)
            d_name = self.deception_classes_with_none[d_idx]
            s_name = self.suppression_classes_with_none[s_idx]
            indices = []
            if d_name != "无欺骗干扰" and d_name in name_to_idx:
                indices.append(name_to_idx[d_name])
            if s_name != "无压制干扰" and s_name in name_to_idx:
                indices.append(name_to_idx[s_name])
            return tuple(sorted(indices))

        def indices_to_name(indices):
            if not indices:
                return "None"
            return "+".join([all_class_names[i] for i in indices])

        true_comb = [get_comb_indices(labels_d[i], labels_s[i]) for i in range(len(labels_d))]
        pred_comb = [get_comb_indices(preds_d[i], preds_s[i]) for i in range(len(preds_d))]

        # Determine order
        if seen_combinations is not None or unseen_combinations is not None:
            configured = []
            for comb_list in [seen_combinations, unseen_combinations]:
                if comb_list:
                    for comb in comb_list:
                        if isinstance(comb, list):
                            indices = tuple(sorted([name_to_idx.get(c, -1) for c in comb if c in name_to_idx]))
                            if indices and indices not in configured:
                                configured.append(indices)
            for comb in sorted(set(true_comb) | set(pred_comb)):
                if comb not in configured:
                    configured.append(comb)
            unique_combs = configured
        else:
            unique_combs = sorted(set(true_comb) | set(pred_comb))

        comb_to_idx = {comb: i for i, comb in enumerate(unique_combs)}
        n = len(unique_combs)
        confusion = np.zeros((n, n), dtype=int)
        for i in range(len(labels_d)):
            tc, pc = true_comb[i], pred_comb[i]
            if tc in comb_to_idx and pc in comb_to_idx:
                confusion[comb_to_idx[tc], comb_to_idx[pc]] += 1

        seen_set = set()
        unseen_set = set()
        for comb_list, target_set in [(seen_combinations, seen_set), (unseen_combinations, unseen_set)]:
            if comb_list:
                for comb in comb_list:
                    if isinstance(comb, list):
                        indices = tuple(sorted([name_to_idx.get(c, -1) for c in comb if c in name_to_idx]))
                        if indices:
                            target_set.add(indices)

        comb_names = []
        for comb in unique_combs:
            name = indices_to_name(comb)
            if seen_set or unseen_set:
                if comb in seen_set:
                    name = f"[S] {name}"
                elif comb in unseen_set:
                    name = f"[U] {name}"
            comb_names.append(name)

        fig_size = max(12, n * 0.6)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
                    xticklabels=comb_names, yticklabels=comb_names, ax=ax)
        ax.set_xlabel('Predicted Combination')
        ax.set_ylabel('True Combination')
        ax.set_title(f'{self._model_label} Combination Confusion Matrix - [S]=Seen, [U]=Unseen')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Combination confusion matrix saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def print_combination_metrics(
        self, metrics: dict,
        seen_combinations: list = None, unseen_combinations: list = None
    ):
        labels_d = metrics.get("labels_deception")
        labels_s = metrics.get("labels_suppression")
        preds_d = metrics.get("preds_deception")
        preds_s = metrics.get("preds_suppression")
        if labels_d is None:
            return

        seen_set = set()
        unseen_set = set()
        for comb_list, target_set in [(seen_combinations, seen_set), (unseen_combinations, unseen_set)]:
            if comb_list:
                for comb in comb_list:
                    if isinstance(comb, list):
                        target_set.add("+".join(sorted(comb)))

        seen_correct, seen_total = 0, 0
        unseen_correct, unseen_total = 0, 0

        def get_comb_label(label_d, label_s):
            d_idx, s_idx = np.argmax(label_d), np.argmax(label_s)
            d_name = self.deception_classes_with_none[d_idx]
            s_name = self.suppression_classes_with_none[s_idx]
            parts = []
            if d_name != "无欺骗干扰":
                parts.append(d_name)
            if s_name != "无压制干扰":
                parts.append(s_name)
            return "+".join(sorted(parts)) if parts else "None"

        for i in range(len(labels_d)):
            true_comb = get_comb_label(labels_d[i], labels_s[i])
            pred_comb = get_comb_label(preds_d[i], preds_s[i])
            is_correct = (true_comb == pred_comb)
            if true_comb in seen_set:
                seen_total += 1
                if is_correct:
                    seen_correct += 1
            elif true_comb in unseen_set:
                unseen_total += 1
                if is_correct:
                    unseen_correct += 1

        print("\n" + "=" * 80)
        print("Combination-Level Results")
        print("=" * 80)
        if seen_total > 0:
            print(f"Seen Accuracy:    {seen_correct / seen_total:.4f} ({seen_correct}/{seen_total})")
        if unseen_total > 0:
            print(f"Unseen Accuracy:  {unseen_correct / unseen_total:.4f} ({unseen_correct}/{unseen_total})")
        print(f"\nOverall Combined Accuracy: {metrics['combined_accuracy']:.4f}")

    def save_stft_predictions(
        self, data_loader: DataLoader, output_dir: str,
        max_samples: int = 100, random_sample: bool = True
    ):
        import cv2
        self.model.eval()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        correct_dir = output_path / "correct"
        wrong_dir = output_path / "wrong"
        correct_dir.mkdir(exist_ok=True)
        wrong_dir.mkdir(exist_ok=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        all_samples = []
        print(f"Collecting samples for STFT saving...")
        for batch_data in tqdm(data_loader, desc="Collecting"):
            (stft_images, tok_d, tok_s,
             labels_d, labels_s, txt_d, txt_s, metas) = batch_data
            for i in range(stft_images.size(0)):
                all_samples.append({
                    'image': stft_images[i].cpu(),
                    'label_deception': labels_d[i].cpu(),
                    'label_suppression': labels_s[i].cpu(),
                })

        total = len(all_samples)
        if random_sample and max_samples < total:
            import random
            indices = random.sample(range(total), max_samples)
            selected = [all_samples[i] for i in indices]
        else:
            selected = all_samples[:max_samples]

        saved_count, correct_count, wrong_count = 0, 0, 0
        for sample in tqdm(selected, desc="Saving STFT images"):
            image = sample['image'].unsqueeze(0).to(self.device)
            if image.shape[-1] != 224:
                image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)

            # Predict
            pred_d_idx, pred_s_idx = self._predict_sample(image)
            true_d_idx = torch.argmax(sample['label_deception']).item()
            true_s_idx = torch.argmax(sample['label_suppression']).item()

            true_d_name = self.deception_classes_with_none[true_d_idx]
            true_s_name = self.suppression_classes_with_none[true_s_idx]
            pred_d_name = self.deception_classes_with_none[pred_d_idx]
            pred_s_name = self.suppression_classes_with_none[pred_s_idx]

            def get_comb_name(d_name, s_name):
                parts = []
                if d_name != "无欺骗干扰":
                    parts.append(d_name)
                if s_name != "无压制干扰":
                    parts.append(s_name)
                return "+".join(sorted(parts)) if parts else "None"

            true_name = get_comb_name(true_d_name, true_s_name)
            pred_name = get_comb_name(pred_d_name, pred_s_name)
            is_correct = (pred_d_idx == true_d_idx and pred_s_idx == true_s_idx)

            img = image[0].cpu().numpy()
            mean = np.array([0.48145466, 0.4578275, 0.40821073])
            std = np.array([0.26862954, 0.26130258, 0.27577711])
            img = img * std[:, None, None] + mean[:, None, None]
            img = np.clip(img, 0, 1)
            img = np.transpose(img, (1, 2, 0))
            img = (img * 255).astype(np.uint8)

            save_dir = correct_dir if is_correct else wrong_dir
            if is_correct:
                correct_count += 1
            else:
                wrong_count += 1

            filename = f"{saved_count:05d}_true_{true_name}_pred_{pred_name}.png"
            cv2.imwrite(str(save_dir / filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            saved_count += 1

        print(f"\nSaved {saved_count} STFT images:")
        print(f"  Correct: {correct_count} -> {correct_dir}")
        print(f"  Wrong: {wrong_count} -> {wrong_dir}")
        with open(output_path / "stft_summary.txt", 'w', encoding='utf-8') as f:
            f.write(f"Total: {saved_count}\nCorrect: {correct_count}\nWrong: {wrong_count}\nAccuracy: {correct_count/saved_count:.4f}\n")


# ============================================================================
# EvaluationStrategy 抽象基类
# ============================================================================

class EvaluationStrategy(ABC):
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device

    @abstractmethod
    def create_model(self) -> nn.Module: ...
    @abstractmethod
    def create_dataloaders(self) -> tuple: ...
    @abstractmethod
    def load_checkpoint(self, model: nn.Module, checkpoint_path: str) -> nn.Module: ...
    @abstractmethod
    def cache_text_features(self, model: nn.Module) -> None: ...
    @abstractmethod
    def evaluate(self, model: nn.Module, data_loader: DataLoader, **kwargs) -> dict: ...
    @abstractmethod
    def print_metrics(self, results: dict): ...
    @abstractmethod
    def run_visualizations(self, model: nn.Module, results: dict, output_dir: Path, **kwargs): ...


# ============================================================================
# CZSLEvaluationStrategy
# ============================================================================

class CZSLEvaluationStrategy(EvaluationStrategy):
    def __init__(self, config: dict, device: torch.device, use_original_clip: bool = False):
        super().__init__(config, device)
        model_config = config.get("model", {})
        self.model_type = model_config.get("clip_model", "ViT-B/32")
        self.use_multishape = model_config.get("use_multishape_vit", False)
        self.use_original_clip = use_original_clip

        # Class info
        self.class_names = [cls["name"] for cls in config.get("jamming_classes", [])]
        self.num_classes = len(self.class_names)

        # CZSL config
        czsl_config = config.get("czsl", {})
        self.seen_combinations = czsl_config.get("seen_combinations", [])
        self.unseen_combinations = czsl_config.get("unseen_combinations", [])

    @property
    def _model_label(self):
        if self.use_original_clip:
            return "Original CLIP"
        if self.use_multishape:
            return "MultiShapeViT"
        return "CZSL"

    def create_model(self) -> nn.Module:
        if self.use_original_clip:
            import clip
            model, _ = clip.load(self.model_type, device=self.device)
            model = model.float().eval()
            print(f"Loading original CLIP model: {self.model_type}")
            return model
        elif self.use_multishape:
            from multi.rectangular_patch_vit import create_multi_shape_patch_model
            print("Creating Multi-Shape Patch ViT model...")
            return create_multi_shape_patch_model(self.config, device=str(self.device))
        else:
            from multi.model import create_czsl_model
            print(f"Creating CZSL model ({self.model_type})...")
            return create_czsl_model(self.config, device=str(self.device))

    def create_dataloaders(self) -> tuple:
        from multi.data import create_czsl_dataloaders
        return create_czsl_dataloaders(
            config=self.config,
            batch_size=self.config.get("train", {}).get("batch_size", 16),
            num_workers=self.config.get("data", {}).get("num_workers", 4),
            pin_memory=self.config.get("data", {}).get("pin_memory", True),
            load_test=True,
        )

    def load_checkpoint(self, model: nn.Module, checkpoint_path: str) -> nn.Module:
        if self.use_original_clip:
            return model  # No checkpoint for original CLIP

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        # Try checkpoint config first
        if "config" in checkpoint and not self.use_multishape:
            # Use mismatch filtering for CZSL (classifier might differ)
            state_dict = checkpoint["model_state_dict"]
            model_state = model.state_dict()
            filtered = {}
            for key, value in state_dict.items():
                if key in model_state and model_state[key].shape == value.shape:
                    filtered[key] = value
            model.load_state_dict(filtered, strict=False)
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")
        return model

    def cache_text_features(self, model: nn.Module) -> None:
        if self.use_original_clip:
            self._cache_original_clip_features(model)
        else:
            czsl_config = self.config.get("czsl", {})
            seen_combos = czsl_config.get("seen_combinations", None)
            model.cache_text_features(
                max_combination_size=2, include_single=True, seen_combinations=seen_combos
            )

    def _cache_original_clip_features(self, model):
        import clip
        from multi.text_templates import get_inference_description
        all_features, all_names = [], []
        for cls_name in self.class_names:
            desc = get_inference_description([cls_name])
            tokens = clip.tokenize(desc, truncate=True).to(self.device)
            features = model.encode_text(tokens)
            features = F.normalize(features, dim=-1)
            all_features.append(features)
            all_names.append(cls_name)
        for i, j in combinations(range(len(self.class_names)), 2):
            cls1, cls2 = self.class_names[i], self.class_names[j]
            desc = get_inference_description([cls1, cls2])
            tokens = clip.tokenize(desc, truncate=True).to(self.device)
            features = model.encode_text(tokens)
            features = F.normalize(features, dim=-1)
            all_features.append(features)
            all_names.append(f"{cls1}+{cls2}")
        model._combination_features = torch.cat(all_features, dim=0)
        model._combination_names = all_names
        model._text_features = model._combination_features[:len(self.class_names)]
        print(f"Cached {len(all_names)} text features")

    @torch.no_grad()
    def evaluate(self, model: nn.Module, data_loader: DataLoader,
                 use_combinations: bool = True, debug: bool = False) -> dict:
        model.eval()
        all_labels, all_preds, all_features = [], [], []
        all_combination_correct, all_multilabel_correct, total_samples = 0, 0, 0

        eval_bar = tqdm(data_loader, desc=f"{self._model_label} Evaluation")
        debug_done = False

        for batch_idx, batch_data in enumerate(eval_bar):
            # 解析 batch，支持 6 元素 (无 features) 和 7 元素 (有 features)
            features_dict = None
            if len(batch_data) == 7:
                images, time_signals, text_tokens, labels, texts, metas, features_dict = batch_data
            elif len(batch_data) == 6:
                images, time_signals, text_tokens, labels, texts, metas = batch_data
            else:
                images, text_tokens, labels, texts, metas = batch_data
                time_signals = None

            images = images.to(self.device)
            labels = labels.to(self.device)
            if images.shape[-1] != 224:
                images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            batch_size = images.shape[0]

            # Encode image
            if self.use_original_clip:
                image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
            elif self.use_multishape:
                image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
            else:
                image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
            all_features.append(image_features.cpu().numpy())

            # Zero-shot predict
            if use_combinations:
                # 准备 features_dict (如果可用)
                feat_kwargs = {}
                if features_dict is not None and hasattr(model, 'prompt_learner') and model.prompt_learner is not None:
                    feat_kwargs['features_dict'] = {d: f.to(self.device) for d, f in features_dict.items()}
                    feat_kwargs['seen_combinations'] = self.config.get("czsl", {}).get("seen_combinations", None)

                if self.use_original_clip or self.use_multishape:
                    similarities, indices, pred_names = model.zero_shot_predict(
                        images, use_combinations=True, top_k=1, **feat_kwargs
                    ) if not self.use_original_clip else self._original_clip_predict(model, images, use_combinations=True)
                else:
                    similarities, indices, pred_names = model.zero_shot_predict(
                        images, use_combinations=True, top_k=1, **feat_kwargs
                    )

                preds = torch.zeros(batch_size, self.num_classes, device=self.device)
                if self.use_original_clip:
                    pred_names_list = pred_names
                    for i, name_list in enumerate(pred_names_list):
                        if name_list and len(name_list) > 0:
                            parts = name_list[0].split('+')
                            for part in parts:
                                if part in self.class_names:
                                    preds[i, self.class_names.index(part)] = 1
                else:
                    for i, name in enumerate(pred_names):
                        if name and len(name) > 0:
                            comb_name = name[0]
                            parts = comb_name.split('+')
                            for part in parts:
                                if part in self.class_names:
                                    preds[i, self.class_names.index(part)] = 1
            else:
                # Single-class prediction
                if self.use_original_clip:
                    text_features = model._text_features
                else:
                    text_features = model.get_cached_text_features()
                if self.use_original_clip:
                    logit_scale = model.logit_scale.exp()
                else:
                    logit_scale = model.model.logit_scale.exp() if hasattr(model, 'model') else model.logit_scale.exp()
                logits = logit_scale * (image_features @ text_features.T)
                probs = torch.softmax(logits, dim=-1)
                threshold = 1.0 / self.num_classes
                top_k = 3
                topk_values, topk_indices = torch.topk(probs, k=top_k, dim=-1)
                preds = torch.zeros(batch_size, self.num_classes, device=self.device)
                for b in range(batch_size):
                    for j, idx in enumerate(topk_indices[b]):
                        if topk_values[b, j] > threshold:
                            preds[b, idx] = 1.0

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

            # Combination accuracy
            for i in range(batch_size):
                true_set = set(torch.where(labels[i] == 1)[0].tolist())
                pred_set = set(torch.where(preds[i] == 1)[0].tolist())
                if true_set == pred_set:
                    all_combination_correct += 1
                if true_set.issubset(pred_set) or pred_set.issubset(true_set):
                    all_multilabel_correct += 1
            total_samples += batch_size

            # Debug
            if debug and not debug_done:
                print(f"\n[DEBUG] Batch {batch_idx}, batch_size={batch_size}")
                for i in range(min(3, batch_size)):
                    true_names = [self.class_names[idx] for idx in torch.where(labels[i] == 1)[0].tolist()]
                    pred_names_list = [self.class_names[idx] for idx in torch.where(preds[i] == 1)[0].tolist()]
                    print(f"  [{i}] True: {true_names} | Pred: {pred_names_list}")
                debug_done = True

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
            "per_class": {
                "f1": f1_score(all_labels_np, all_preds_np, average=None, zero_division=0),
                "precision": precision_score(all_labels_np, all_preds_np, average=None, zero_division=0),
                "recall": recall_score(all_labels_np, all_preds_np, average=None, zero_division=0),
            }
        }

        return {
            "metrics": metrics,
            "labels": all_labels_np,
            "predictions": all_preds_np,
            "features": all_features_np,
            "total_samples": total_samples,
        }

    def _original_clip_predict(self, model, images, use_combinations=True):
        """Predict with original CLIP model"""
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        text_features = model._combination_features if use_combinations else model._text_features
        names = model._combination_names if use_combinations else self.class_names
        logit_scale = model.logit_scale.exp()
        similarities = logit_scale * (image_features @ text_features.T)
        _, indices = torch.topk(similarities, k=1, dim=-1)
        pred_names = [[names[idx.item()]] for idx in indices]
        return similarities, indices, pred_names

    @torch.no_grad()
    def evaluate_by_combination(self, model: nn.Module, data_loader: DataLoader,
                                seen_combinations: list = None, unseen_combinations: list = None,
                                debug: bool = False) -> dict:
        """Evaluate by combination type (seen vs unseen)"""
        model.eval()
        seen_set = set(tuple(sorted(c)) for c in (seen_combinations or []))
        unseen_set = set(tuple(sorted(c)) for c in (unseen_combinations or []))
        seen_results = {"correct": 0, "total": 0}
        unseen_results = {"correct": 0, "total": 0}
        other_results = {"correct": 0, "total": 0}
        all_labels, all_preds, all_probs, all_features = [], [], [], []

        for batch_data in tqdm(data_loader, desc="By combination"):
            features_dict = None
            if len(batch_data) == 7:
                images, _, _, labels, texts, metas, features_dict = batch_data
            elif len(batch_data) == 6:
                images, _, _, labels, texts, metas = batch_data
            else:
                images, labels, texts, metas = batch_data[:4]
            images = images.to(self.device)
            labels = labels.to(self.device)
            if images.shape[-1] != 224:
                images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            batch_size = images.shape[0]

            # Predict using single-class features
            if features_dict is not None and hasattr(model, 'prompt_learner') and model.prompt_learner is not None:
                # 特征条件上下文: 逐样本计算
                feat_kwargs = {'features_dict': {d: f.to(self.device) for d, f in features_dict.items()},
                               'use_combinations': False}
                similarities, indices, pred_names = model.zero_shot_predict(images, **feat_kwargs)
                if self.use_original_clip:
                    logit_scale = model.logit_scale.exp()
                else:
                    logit_scale = model.model.logit_scale.exp() if hasattr(model, 'model') else model.logit_scale.exp()
                probs = torch.softmax(similarities, dim=-1)
                image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
            else:
                text_features = model.get_cached_text_features() if not self.use_original_clip else model._text_features
                image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                if self.use_original_clip:
                    logit_scale = model.logit_scale.exp()
                else:
                    logit_scale = model.model.logit_scale.exp() if hasattr(model, 'model') else model.logit_scale.exp()
                logits = logit_scale * (image_features @ text_features.T)
                probs = torch.softmax(logits, dim=-1)

            threshold = 1.0 / self.num_classes
            top_k = 3
            topk_values, topk_indices = torch.topk(probs, k=top_k, dim=-1)
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

        metrics = {
            "seen_accuracy": seen_results["correct"] / seen_results["total"] if seen_results["total"] > 0 else 0,
            "seen_samples": seen_results["total"],
            "unseen_accuracy": unseen_results["correct"] / unseen_results["total"] if unseen_results["total"] > 0 else 0,
            "unseen_samples": unseen_results["total"],
            "other_accuracy": other_results["correct"] / other_results["total"] if other_results["total"] > 0 else 0,
            "other_samples": other_results["total"],
        }
        all_labels_np = torch.cat(all_labels).numpy()
        all_preds_np = torch.cat(all_preds).numpy()
        all_probs_np = torch.cat(all_probs).numpy()
        all_features_np = np.concatenate(all_features, axis=0)

        return {
            "metrics": metrics,
            "labels": all_labels_np,
            "predictions": all_preds_np,
            "probabilities": all_probs_np,
            "features": all_features_np,
        }

    def print_metrics(self, results: dict):
        metrics = results["metrics"]
        print("\n" + "=" * 60)
        print(f"{self._model_label} Evaluation Results")
        print("=" * 60)
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

    def run_visualizations(self, model: nn.Module, results: dict, output_dir: Path, **kwargs):
        labels = results["labels"]
        preds = results["predictions"]
        features = results.get("features")
        split = kwargs.get("split", "test")

        # Combination confusion matrix
        seen_combos = convert_combination_names_to_indices(self.seen_combinations, self.class_names)
        unseen_combos = convert_combination_names_to_indices(self.unseen_combinations, self.class_names)

        self._plot_combination_confusion(
            labels, preds, output_dir / f"czsl_confusion_{split}.png",
            seen_combos, unseen_combos
        )

        # ROC curves
        if "probabilities" in results:
            plot_roc_curves(
                labels, results["probabilities"], self.class_names,
                save_dir=str(output_dir), prefix="czsl", mode_title=split
            )

        # t-SNE
        if features is not None and len(features) > 10:
            print("Computing t-SNE...")
            perplexity = min(30, len(features) - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            features_2d = tsne.fit_transform(features)
            comb_labels = [tuple(sorted(np.where(label == 1)[0].tolist())) for label in labels]
            seen_set = set(tuple(sorted(c)) for c in seen_combos)
            unseen_set = set(tuple(sorted(c)) for c in unseen_combos)
            unique_combs = sorted(set(comb_labels))

            fig, ax = plt.subplots(figsize=(14, 10))
            colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(unique_combs))))
            for idx, comb in enumerate(unique_combs):
                mask = np.array([c == comb for c in comb_labels])
                if mask.sum() > 0:
                    name = "+".join([self.class_names[i] for i in comb]) if comb else "None"
                    marker = '^' if comb in unseen_set else ('o' if comb in seen_set else 's')
                    prefix = "[U] " if comb in unseen_set else ("[S] " if comb in seen_set else "")
                    ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                               c=[colors[idx % 20]], label=f"{prefix}{name}", alpha=0.6, s=30, marker=marker)
            ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
            ax.set_title(f'Feature Space (t-SNE) - [S]=Seen, [U]=Unseen')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(str(output_dir / f"czsl_tsne_{split}.png"), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"t-SNE plot saved")

        # Label co-occurrence
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        nc = labels.shape[1]
        sns.heatmap((labels.T @ labels).astype(int), annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names[:nc], yticklabels=self.class_names[:nc], ax=axes[0])
        axes[0].set_title('True Label Co-occurrence')
        axes[0].tick_params(axis='x', rotation=45)
        sns.heatmap((preds.T @ preds).astype(int), annot=True, fmt='d', cmap='Greens',
                    xticklabels=self.class_names[:nc], yticklabels=self.class_names[:nc], ax=axes[1])
        axes[1].set_title('Predicted Label Co-occurrence')
        axes[1].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(str(output_dir / f"czsl_cooccurrence_{split}.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Co-occurrence plot saved")

    def _plot_combination_confusion(self, labels, preds, save_path, seen_combos, unseen_combos):
        true_combs = [tuple(sorted(np.where(labels[i] == 1)[0].tolist())) for i in range(len(labels))]
        pred_combs = [tuple(sorted(np.where(preds[i] == 1)[0].tolist())) for i in range(len(preds))]
        unique = sorted(set(true_combs) | set(pred_combs))
        comb_to_idx = {c: i for i, c in enumerate(unique)}
        n = len(unique)
        confusion = np.zeros((n, n), dtype=int)
        for tc, pc in zip(true_combs, pred_combs):
            if tc in comb_to_idx and pc in comb_to_idx:
                confusion[comb_to_idx[tc], comb_to_idx[pc]] += 1
        seen_set = set(seen_combos)
        unseen_set = set(unseen_combos)
        comb_names = []
        for c in unique:
            name = "+".join([self.class_names[i] for i in c]) if c else "None"
            if c in seen_set:
                name = f"[S] {name}"
            elif c in unseen_set:
                name = f"[U] {name}"
            comb_names.append(name)
        fig_size = max(10, n * 0.5)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
                    xticklabels=comb_names, yticklabels=comb_names, ax=ax)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'{self._model_label} Combination Confusion Matrix')
        plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Combination confusion matrix saved to {save_path}")


# ============================================================================
# DualBranchEvaluationStrategy
# ============================================================================

class DualBranchEvaluationStrategy(EvaluationStrategy, DualBranchPlotMixin):
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        model_config = config.get("model", {})
        self.use_multishape = model_config.get("use_multishape_vit", False)
        self.use_resnet18 = model_config.get("use_resnet18", False)

        # Class info
        jamming_groups = config.get("jamming_groups", {})
        self.deception_classes = jamming_groups.get("deception", {}).get("classes",
            ["DFTJ", "ISRJ", "SMSPJ", "C&IJ", "CSJ"])
        self.suppression_classes = jamming_groups.get("suppression", {}).get("classes",
            ["AJ", "BJ", "SJ", "NCJ", "NPJ", "NFMJ", "NPMJ", "NAMJ", "PJ"])
        self.deception_classes_with_none = self.deception_classes + ["无欺骗干扰"]
        self.suppression_classes_with_none = self.suppression_classes + ["无压制干扰"]

    @property
    def _model_label(self):
        if self.use_resnet18:
            return "ResNet18 Dual-Branch"
        elif self.use_multishape:
            return "MultiShapeViT Dual-Branch"
        return "CLIP Dual-Branch"

    def create_model(self) -> nn.Module:
        if self.use_resnet18:
            from multi.model import create_resnet18_dual_branch_model
            print("Creating ResNet18 dual-branch model...")
            return create_resnet18_dual_branch_model(self.config, device=str(self.device))
        elif self.use_multishape:
            from multi.rectangular_patch_vit import create_multi_shape_dual_branch_model
            print("Creating Multi-Shape Patch ViT dual-branch model...")
            return create_multi_shape_dual_branch_model(self.config, device=str(self.device))
        else:
            from multi.model import create_dual_branch_model
            print("Creating CLIP dual-branch model...")
            return create_dual_branch_model(self.config, device=str(self.device))

    def create_dataloaders(self) -> tuple:
        from multi.data import create_dual_branch_dataloaders
        return create_dual_branch_dataloaders(
            config=self.config,
            batch_size=self.config.get("train", {}).get("batch_size", 16),
            num_workers=self.config.get("data", {}).get("num_workers", 4),
            pin_memory=self.config.get("data", {}).get("pin_memory", True),
            load_test=True,
        )

    def load_checkpoint(self, model: nn.Module, checkpoint_path: str) -> nn.Module:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")
        return model

    def cache_text_features(self, model: nn.Module) -> None:
        if not self.use_resnet18:
            model.cache_text_features_dual()

    @torch.no_grad()
    def evaluate(self, model: nn.Module, data_loader: DataLoader, debug: bool = False) -> dict:
        model.eval()
        all_labels_d, all_labels_s, all_preds_d, all_preds_s = [], [], [], []
        all_combined_correct, total_samples = 0, 0

        eval_bar = tqdm(data_loader, desc=f"{self._model_label} Evaluation")
        debug_done = False

        for batch_idx, batch_data in enumerate(eval_bar):
            (stft_images, tok_d, tok_s,
             labels_d, labels_s, txt_d, txt_s, metas) = batch_data

            stft_images = stft_images.to(self.device)
            labels_d = labels_d.to(self.device)
            labels_s = labels_s.to(self.device)
            if stft_images.shape[-1] != 224:
                stft_images = F.interpolate(stft_images, size=(224, 224), mode='bilinear', align_corners=False)
            batch_size = stft_images.size(0)

            # Predict
            if self.use_resnet18:
                deception_result, suppression_result = model.predict(stft_images)
            else:
                deception_result, suppression_result = model.zero_shot_predict_dual(stft_images)

            pred_d_idx = deception_result['indices'][:, 0]
            pred_s_idx = suppression_result['indices'][:, 0]

            preds_d = torch.zeros(batch_size, len(self.deception_classes_with_none), device=self.device)
            preds_s = torch.zeros(batch_size, len(self.suppression_classes_with_none), device=self.device)
            for i in range(batch_size):
                preds_d[i, pred_d_idx[i]] = 1.0
                preds_s[i, pred_s_idx[i]] = 1.0

            all_labels_d.append(labels_d.cpu())
            all_labels_s.append(labels_s.cpu())
            all_preds_d.append(preds_d.cpu())
            all_preds_s.append(preds_s.cpu())

            # Debug
            if debug and not debug_done:
                print(f"\n[DEBUG] Batch {batch_idx}")
                for i in range(min(3, batch_size)):
                    true_d = [self.deception_classes_with_none[idx] for idx in torch.where(labels_d[i] == 1)[0].tolist()]
                    pred_d = self.deception_classes_with_none[pred_d_idx[i].item()]
                    print(f"  [{i}] True_D={true_d} Pred_D={pred_d}")
                debug_done = True

            # Combined accuracy
            labels_d_idx = torch.argmax(labels_d, dim=1)
            labels_s_idx = torch.argmax(labels_s, dim=1)
            for i in range(batch_size):
                if pred_d_idx[i] == labels_d_idx[i] and pred_s_idx[i] == labels_s_idx[i]:
                    all_combined_correct += 1
            total_samples += batch_size

        all_labels_d_np = torch.cat(all_labels_d).numpy()
        all_labels_s_np = torch.cat(all_labels_s).numpy()
        all_preds_d_np = torch.cat(all_preds_d).numpy()
        all_preds_s_np = torch.cat(all_preds_s).numpy()

        deception_metrics = self._compute_metrics(all_labels_d_np, all_preds_d_np)
        suppression_metrics = self._compute_metrics(all_labels_s_np, all_preds_s_np)
        combined_accuracy = all_combined_correct / total_samples

        return {
            "deception": deception_metrics,
            "suppression": suppression_metrics,
            "combined_accuracy": combined_accuracy,
            "total_samples": total_samples,
            "labels_deception": all_labels_d_np,
            "labels_suppression": all_labels_s_np,
            "preds_deception": all_preds_d_np,
            "preds_suppression": all_preds_s_np,
        }

    def _predict_sample(self, image: torch.Tensor) -> tuple:
        """Predict a single image (for STFT saving). Returns (pred_d_idx, pred_s_idx)."""
        if self.use_resnet18:
            d_result, s_result = self.model.predict(image)
        else:
            d_result, s_result = self.model.zero_shot_predict_dual(image)
        return d_result['indices'][0, 0].item(), s_result['indices'][0, 0].item()

    def run_visualizations(self, model: nn.Module, results: dict, output_dir: Path, **kwargs):
        split = kwargs.get("split", "test")

        # Confusion matrices
        self.plot_confusion_matrices(results, save_path=str(output_dir / f"confusion_matrices_{split}.png"))

        # Per-class F1
        self.plot_per_class_f1(results, save_path=str(output_dir / f"per_class_f1_{split}.png"))

        # Metrics summary
        self.plot_metrics_summary(results, save_path=str(output_dir / f"metrics_summary_{split}.png"))

        # Combination confusion matrix
        czsl_config = self.config.get("czsl", {})
        seen_combos = czsl_config.get("seen_combinations", [])
        unseen_combos = czsl_config.get("unseen_combinations", [])
        all_class_names = [cls["name"] for cls in self.config.get("jamming_classes", [])]
        self.plot_combination_confusion_matrix(
            results, save_path=str(output_dir / f"combination_confusion_{split}.png"),
            seen_combinations=seen_combos, unseen_combinations=unseen_combos,
            all_class_names=all_class_names
        )
        self.print_combination_metrics(results, seen_combinations=seen_combos, unseen_combinations=unseen_combos)


# ============================================================================
# Strategy 工厂
# ============================================================================

def create_evaluation_strategy(config: dict, device: torch.device,
                                use_original_clip: bool = False) -> EvaluationStrategy:
    mode = config.get("train", {}).get("mode", "czsl")
    if mode == "czsl":
        return CZSLEvaluationStrategy(config, device, use_original_clip=use_original_clip)
    elif mode == "dual_branch":
        return DualBranchEvaluationStrategy(config, device)
    else:
        raise ValueError(f"Unknown train.mode: '{mode}'. Expected 'czsl' or 'dual_branch'.")


# ============================================================================
# main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified CLIP/CZSL Evaluation")
    parser.add_argument("--config", type=str, default="multi/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--mode", type=str, default="zero_shot",
                        choices=["zero_shot", "by_combination", "by_jnr"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save_stft", action="store_true")
    parser.add_argument("--max_stft_samples", type=int, default=100)
    parser.add_argument("--original-clip", action="store_true", help="Use original untrained CLIP (baseline)")
    parser.add_argument("--use-combinations", action="store_true", default=True, help="Use combination features")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create strategy
    strategy = create_evaluation_strategy(config, device, use_original_clip=args.original_clip)

    # Create model
    model = strategy.create_model()

    # Load checkpoint
    if args.checkpoint:
        model = strategy.load_checkpoint(model, args.checkpoint)
    elif not args.original_clip:
        print("Warning: No checkpoint specified. Using random weights.")

    # Cache text features
    strategy.cache_text_features(model)

    # Create dataloaders
    dataloaders = strategy.create_dataloaders()
    train_loader, val_loader, test_loader = dataloaders[0], dataloaders[1], dataloaders[2]
    if args.split == "train":
        data_loader = train_loader
    elif args.split == "val":
        data_loader = val_loader
    else:
        data_loader = test_loader

    # Evaluate
    if isinstance(strategy, CZSLEvaluationStrategy) and args.mode == "by_combination":
        print(f"\nEvaluating by combination type on {args.split} set...")
        seen_combos = convert_combination_names_to_indices(
            strategy.seen_combinations, strategy.class_names)
        unseen_combos = convert_combination_names_to_indices(
            strategy.unseen_combinations, strategy.class_names)
        results = strategy.evaluate_by_combination(
            model, data_loader, seen_combinations=seen_combos,
            unseen_combinations=unseen_combos, debug=args.debug
        )
        m = results["metrics"]
        print(f"\nSeen Accuracy:    {m['seen_accuracy']:.4f} ({m['seen_samples']} samples)")
        print(f"Unseen Accuracy:  {m['unseen_accuracy']:.4f} ({m['unseen_samples']} samples)")
        print(f"Other Accuracy:   {m['other_accuracy']:.4f} ({m['other_samples']} samples)")
    else:
        print(f"\nEvaluating on {args.split} set ({args.mode} mode)...")
        results = strategy.evaluate(
            model, data_loader,
            use_combinations=args.use_combinations,
            debug=args.debug
        )
        strategy.print_metrics(results)

    # Save results
    if isinstance(strategy, CZSLEvaluationStrategy):
        np.savez(
            str(output_dir / f"results_{args.split}.npz"),
            labels=results["labels"],
            predictions=results["predictions"],
            features=results.get("features", np.array([]))
        )
    else:
        # Dual-branch: save JSON (exclude numpy arrays)
        metrics_to_save = {}
        for key in ["deception", "suppression"]:
            metrics_to_save[key] = {k: v for k, v in results[key].items() if not isinstance(v, np.ndarray)}
        metrics_to_save["combined_accuracy"] = results["combined_accuracy"]
        metrics_to_save["total_samples"] = results["total_samples"]
        label = "resnet18" if strategy.use_resnet18 else ("multishape" if strategy.use_multishape else "dual_branch")
        with open(output_dir / f"{label}_results_{args.split}.json", 'w', encoding='utf-8') as f:
            json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)

    # Visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        strategy.run_visualizations(model, results, output_dir, split=args.split)

    # Save STFT images (dual-branch only)
    if args.save_stft and isinstance(strategy, DualBranchEvaluationStrategy):
        print("\nSaving STFT images with predictions...")
        # Recreate dataloader with lower memory usage
        from multi.data import create_dual_branch_dataloaders
        _, _, stft_loader, _, _ = create_dual_branch_dataloaders(
            config, load_test=True, pin_memory=False, num_workers=0
        )
        if args.split == "train":
            stft_loader = stft_loader
        elif args.split == "val":
            stft_loader = stft_loader
        strategy.save_stft_predictions(
            data_loader=stft_loader,
            output_dir=str(output_dir / "stft"),
            max_samples=args.max_stft_samples,
        )

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
