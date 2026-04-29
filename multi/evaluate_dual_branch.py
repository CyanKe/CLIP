"""
双分支 CZSL 评估脚本 - 用于欺骗/压制干扰分类
python -m multi.evaluate_dual_branch --checkpoint checkpoints/dual_branch_best_model.pt --split test
"""
import os
import sys
import yaml
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi.model import create_dual_branch_model
from multi.data import create_dual_branch_dataloaders, DualBranchSTFTDataset, collate_fn_dual_branch
from multi.rectangular_patch_vit import create_multi_shape_dual_branch_model


class DualBranchEvaluator:
    """双分支 CZSL 评估器"""

    def __init__(
        self,
        model,  # 支持 DualBranchCLIPForCZSL 或 MultiShapePatchViTForDualBranch
        device: torch.device,
        deception_classes: list,
        suppression_classes: list
    ):
        self.model = model
        self.device = device
        self.deception_classes = deception_classes
        self.suppression_classes = suppression_classes

        # 包含 "无XX干扰"
        self.deception_classes_with_none = deception_classes + ["无欺骗干扰"]
        self.suppression_classes_with_none = suppression_classes + ["无压制干扰"]

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, debug: bool = False) -> dict:
        """
        评估双分支模型

        Returns:
            metrics: 包含各项指标的字典
        """
        self.model.eval()

        all_labels_deception = []
        all_labels_suppression = []
        all_preds_deception = []
        all_preds_suppression = []
        all_combined_correct = 0
        total_samples = 0

        eval_bar = tqdm(data_loader, desc="Evaluating")

        debug_done = False
        for batch_idx, batch_data in enumerate(eval_bar):
            (stft_images, text_tokens_deception, text_tokens_suppression,
             labels_deception, labels_suppression, texts_deception, texts_suppression,
             metadata_list) = batch_data

            stft_images = stft_images.to(self.device)
            labels_deception = labels_deception.to(self.device)
            labels_suppression = labels_suppression.to(self.device)

            if stft_images.shape[-1] != 224:
                stft_images = nn.functional.interpolate(
                    stft_images, size=(224, 224), mode='bilinear', align_corners=False
                )

            batch_size = stft_images.size(0)

            # 双分支预测
            deception_result, suppression_result = self.model.zero_shot_predict_dual(stft_images)

            # 获取预测类别
            pred_deception_idx = deception_result['indices'][:, 0]  # top-1
            pred_suppression_idx = suppression_result['indices'][:, 0]

            # 转换为多热编码
            preds_deception = torch.zeros(batch_size, len(self.deception_classes_with_none), device=self.device)
            preds_suppression = torch.zeros(batch_size, len(self.suppression_classes_with_none), device=self.device)

            for i in range(batch_size):
                preds_deception[i, pred_deception_idx[i]] = 1.0
                preds_suppression[i, pred_suppression_idx[i]] = 1.0

            all_labels_deception.append(labels_deception.cpu())
            all_labels_suppression.append(labels_suppression.cpu())
            all_preds_deception.append(preds_deception.cpu())
            all_preds_suppression.append(preds_suppression.cpu())

            # Debug
            if debug and not debug_done:
                print(f"\n{'='*80}")
                print(f"[DEBUG] Batch {batch_idx}")
                for i in range(min(5, batch_size)):
                    true_deception = [self.deception_classes_with_none[idx] for idx in torch.where(labels_deception[i] == 1)[0].tolist()]
                    true_suppression = [self.suppression_classes_with_none[idx] for idx in torch.where(labels_suppression[i] == 1)[0].tolist()]
                    pred_deception = [self.deception_classes_with_none[idx.item()] for idx in [pred_deception_idx[i]]]
                    pred_suppression = [self.suppression_classes_with_none[idx.item()] for idx in [pred_suppression_idx[i]]]

                    print(f"  [{i}] True: Deception={true_deception}, Suppression={true_suppression}")
                    print(f"       Pred: Deception={pred_deception}, Suppression={pred_suppression}")
                print(f"{'='*80}")
                debug_done = True

            # 计算组合准确率
            for i in range(batch_size):
                true_deception_set = set(torch.where(labels_deception[i] == 1)[0].tolist())
                true_suppression_set = set(torch.where(labels_suppression[i] == 1)[0].tolist())
                pred_deception_set = {pred_deception_idx[i].item()}
                pred_suppression_set = {pred_suppression_idx[i].item()}

                # 两个分支都正确才算正确
                if true_deception_set == pred_deception_set and true_suppression_set == pred_suppression_set:
                    all_combined_correct += 1

            total_samples += batch_size

        # 合并结果
        all_labels_deception = torch.cat(all_labels_deception).numpy()
        all_labels_suppression = torch.cat(all_labels_suppression).numpy()
        all_preds_deception = torch.cat(all_preds_deception).numpy()
        all_preds_suppression = torch.cat(all_preds_suppression).numpy()

        # 计算欺骗分支指标
        deception_metrics = self._compute_metrics(all_labels_deception, all_preds_deception, "deception")

        # 计算压制分支指标
        suppression_metrics = self._compute_metrics(all_labels_suppression, all_preds_suppression, "suppression")

        # 组合指标
        combined_accuracy = all_combined_correct / total_samples

        metrics = {
            "deception": deception_metrics,
            "suppression": suppression_metrics,
            "combined_accuracy": combined_accuracy,
            "total_samples": total_samples,
            # 保存原始数据用于可视化
            "labels_deception": all_labels_deception,
            "labels_suppression": all_labels_suppression,
            "preds_deception": all_preds_deception,
            "preds_suppression": all_preds_suppression
        }

        return metrics

    def _compute_metrics(self, labels: np.ndarray, preds: np.ndarray, branch_name: str) -> dict:
        """计算单个分支的指标"""
        # 排除 "无XX干扰" 类 (最后一个类) 的指标
        num_classes = labels.shape[1] - 1
        labels_exclude_none = labels[:, :-1]
        preds_exclude_none = preds[:, :-1]

        return {
            "f1_macro": f1_score(labels, preds, average='macro', zero_division=0),
            "f1_micro": f1_score(labels, preds, average='micro', zero_division=0),
            "precision_macro": precision_score(labels, preds, average='macro', zero_division=0),
            "recall_macro": recall_score(labels, preds, average='macro', zero_division=0),
            "accuracy": accuracy_score(labels, preds),
            "per_class_f1": f1_score(labels, preds, average=None, zero_division=0).tolist(),
            "per_class_precision": precision_score(labels, preds, average=None, zero_division=0).tolist(),
            "per_class_recall": recall_score(labels, preds, average=None, zero_division=0).tolist()
        }

    def print_metrics(self, metrics: dict):
        """打印评估指标"""
        print("\n" + "=" * 80)
        print("Dual-Branch Evaluation Results")
        print("=" * 80)

        print("\n[Deception Branch]")
        d = metrics["deception"]
        print(f"  F1 Macro:    {d['f1_macro']:.4f}")
        print(f"  F1 Micro:    {d['f1_micro']:.4f}")
        print(f"  Precision:   {d['precision_macro']:.4f}")
        print(f"  Recall:      {d['recall_macro']:.4f}")
        print(f"  Accuracy:    {d['accuracy']:.4f}")

        print("\n[Suppression Branch]")
        s = metrics["suppression"]
        print(f"  F1 Macro:    {s['f1_macro']:.4f}")
        print(f"  F1 Micro:    {s['f1_micro']:.4f}")
        print(f"  Precision:   {s['precision_macro']:.4f}")
        print(f"  Recall:      {s['recall_macro']:.4f}")
        print(f"  Accuracy:    {s['accuracy']:.4f}")

        print("\n[Combined]")
        print(f"  Combined Accuracy: {metrics['combined_accuracy']:.4f}")
        print(f"  Total Samples: {metrics['total_samples']}")

        # 打印每个类别的指标
        print("\n[Per-Class F1 - Deception]")
        for i, name in enumerate(self.deception_classes_with_none):
            f1 = metrics["deception"]["per_class_f1"][i]
            print(f"  {name:<15}: {f1:.4f}")

        print("\n[Per-Class F1 - Suppression]")
        for i, name in enumerate(self.suppression_classes_with_none):
            f1 = metrics["suppression"]["per_class_f1"][i]
            print(f"  {name:<15}: {f1:.4f}")

    def plot_confusion_matrices(self, metrics: dict, save_path: str = None):
        """绘制混淆矩阵"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # 欺骗分支混淆矩阵
        labels_deception = metrics.get("labels_deception")
        preds_deception = metrics.get("preds_deception")

        if labels_deception is not None and preds_deception is not None:
            # 获取每个样本的预测类别
            pred_classes = np.argmax(preds_deception, axis=1)
            true_classes = np.argmax(labels_deception, axis=1)

            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(true_classes, pred_classes)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.deception_classes_with_none,
                        yticklabels=self.deception_classes_with_none,
                        ax=axes[0])
            axes[0].set_title('Deception Branch Confusion Matrix')
            axes[0].set_xlabel('Predicted')
            axes[0].set_ylabel('True')
            axes[0].tick_params(axis='x', rotation=45)

        # 压制分支混淆矩阵
        labels_suppression = metrics.get("labels_suppression")
        preds_suppression = metrics.get("preds_suppression")

        if labels_suppression is not None and preds_suppression is not None:
            pred_classes = np.argmax(preds_suppression, axis=1)
            true_classes = np.argmax(labels_suppression, axis=1)

            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(true_classes, pred_classes)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                        xticklabels=self.suppression_classes_with_none,
                        yticklabels=self.suppression_classes_with_none,
                        ax=axes[1])
            axes[1].set_title('Suppression Branch Confusion Matrix')
            axes[1].set_xlabel('Predicted')
            axes[1].set_ylabel('True')
            axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrices saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_per_class_f1(self, metrics: dict, save_path: str = None):
        """绘制每个类别的 F1 分数条形图"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 欺骗分支 F1 分数
        f1_deception = metrics["deception"]["per_class_f1"]
        x_deception = np.arange(len(self.deception_classes_with_none))
        colors_deception = plt.cm.Blues(np.linspace(0.3, 0.9, len(f1_deception)))

        bars1 = axes[0].bar(x_deception, f1_deception, color=colors_deception)
        axes[0].set_xticks(x_deception)
        axes[0].set_xticklabels(self.deception_classes_with_none, rotation=45, ha='right')
        axes[0].set_ylabel('F1 Score')
        axes[0].set_title('Deception Branch - Per-Class F1 Score')
        axes[0].set_ylim(0, 1.0)
        axes[0].axhline(y=metrics["deception"]["f1_macro"], color='r', linestyle='--',
                        label=f'Macro F1: {metrics["deception"]["f1_macro"]:.4f}')
        axes[0].legend()

        # 在柱子上添加数值
        for bar, val in zip(bars1, f1_deception):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)

        # 压制分支 F1 分数
        f1_suppression = metrics["suppression"]["per_class_f1"]
        x_suppression = np.arange(len(self.suppression_classes_with_none))
        colors_suppression = plt.cm.Greens(np.linspace(0.3, 0.9, len(f1_suppression)))

        bars2 = axes[1].bar(x_suppression, f1_suppression, color=colors_suppression)
        axes[1].set_xticks(x_suppression)
        axes[1].set_xticklabels(self.suppression_classes_with_none, rotation=45, ha='right')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('Suppression Branch - Per-Class F1 Score')
        axes[1].set_ylim(0, 1.0)
        axes[1].axhline(y=metrics["suppression"]["f1_macro"], color='r', linestyle='--',
                        label=f'Macro F1: {metrics["suppression"]["f1_macro"]:.4f}')
        axes[1].legend()

        for bar, val in zip(bars2, f1_suppression):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Per-class F1 plot saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_metrics_summary(self, metrics: dict, save_path: str = None):
        """绘制指标汇总图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 指标名称
        metric_names = ['F1 Macro', 'F1 Micro', 'Precision', 'Recall', 'Accuracy']

        # 欺骗分支指标
        deception_values = [
            metrics["deception"]["f1_macro"],
            metrics["deception"]["f1_micro"],
            metrics["deception"]["precision_macro"],
            metrics["deception"]["recall_macro"],
            metrics["deception"]["accuracy"]
        ]

        x = np.arange(len(metric_names))
        width = 0.35

        bars1 = axes[0].bar(x, deception_values, width, color='steelblue', label='Deception')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Deception Branch Metrics')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(metric_names, rotation=45, ha='right')
        axes[0].set_ylim(0, 1.0)

        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2, height + 0.02,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        # 压制分支指标
        suppression_values = [
            metrics["suppression"]["f1_macro"],
            metrics["suppression"]["f1_micro"],
            metrics["suppression"]["precision_macro"],
            metrics["suppression"]["recall_macro"],
            metrics["suppression"]["accuracy"]
        ]

        bars2 = axes[1].bar(x, suppression_values, width, color='seagreen', label='Suppression')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Suppression Branch Metrics')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(metric_names, rotation=45, ha='right')
        axes[1].set_ylim(0, 1.0)

        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2, height + 0.02,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Metrics summary plot saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_combination_confusion_matrix(
        self,
        metrics: dict,
        save_path: str = None,
        seen_combinations: list = None,
        unseen_combinations: list = None,
        all_class_names: list = None
    ):
        """
        绘制组合混淆矩阵 - 显示所有欺骗+压制组合

        排序逻辑与 evaluate_czsl.py 一致：
        1. 先按 seen_combinations 配置顺序
        2. 再按 unseen_combinations 配置顺序
        3. 最后添加数据中出现但配置中没有的组合（按元组排序）

        Args:
            metrics: 评估结果
            save_path: 保存路径
            seen_combinations: 已见组合列表 (如 [["DFTJ"], ["DFTJ", "AJ"]])
            unseen_combinations: 未见组合列表
            all_class_names: 所有类别名称列表（用于确定排序顺序）
        """
        labels_deception = metrics.get("labels_deception")
        labels_suppression = metrics.get("labels_suppression")
        preds_deception = metrics.get("preds_deception")
        preds_suppression = metrics.get("preds_suppression")

        if labels_deception is None or labels_suppression is None:
            print("No label data available for combination confusion matrix")
            return

        # 构建类别名称到索引的映射
        name_to_idx = {name: i for i, name in enumerate(all_class_names or [])}

        # 构建组合标签 (欺骗+压制)
        def get_combination_indices(label_d, label_s):
            """从双分支标签获取组合的类索引元组"""
            d_idx = np.argmax(label_d)
            s_idx = np.argmax(label_s)

            d_name = self.deception_classes_with_none[d_idx]
            s_name = self.suppression_classes_with_none[s_idx]

            # 构建组合索引列表
            indices = []
            if d_name != "无欺骗干扰" and d_name in name_to_idx:
                indices.append(name_to_idx[d_name])
            if s_name != "无压制干扰" and s_name in name_to_idx:
                indices.append(name_to_idx[s_name])

            return tuple(sorted(indices))

        def indices_to_name(indices):
            """将索引元组转换为组合名称"""
            if not indices:
                return "None"
            return "+".join([all_class_names[i] for i in indices])

        # 提取真实组合和预测组合（使用索引元组）
        true_comb_indices = []
        pred_comb_indices = []

        for i in range(len(labels_deception)):
            true_idx = get_combination_indices(labels_deception[i], labels_suppression[i])
            pred_idx = get_combination_indices(preds_deception[i], preds_suppression[i])
            true_comb_indices.append(true_idx)
            pred_comb_indices.append(pred_idx)

        # 确定组合顺序（与 evaluate_czsl.py 一致）
        if seen_combinations is not None or unseen_combinations is not None:
            # 将配置中的组合名称转换为索引元组
            configured_combs = []

            # 先添加 seen_combinations（保持配置顺序）
            if seen_combinations:
                for comb in seen_combinations:
                    if isinstance(comb, list):
                        indices = tuple(sorted([name_to_idx.get(c, -1) for c in comb if c in name_to_idx]))
                        if indices and indices not in configured_combs:
                            configured_combs.append(indices)

            # 再添加 unseen_combinations（保持配置顺序）
            if unseen_combinations:
                for comb in unseen_combinations:
                    if isinstance(comb, list):
                        indices = tuple(sorted([name_to_idx.get(c, -1) for c in comb if c in name_to_idx]))
                        if indices and indices not in configured_combs:
                            configured_combs.append(indices)

            # 添加数据中出现的其他组合（按元组排序）
            data_combs = set(true_comb_indices) | set(pred_comb_indices)
            for comb in sorted(data_combs):
                if comb not in configured_combs:
                    configured_combs.append(comb)

            unique_combs = configured_combs

            print(f"\nUsing configured combinations: {len(seen_combinations or []) + len(unseen_combinations or [])} "
                  f"(Seen: {len(seen_combinations or [])}, Unseen: {len(unseen_combinations or [])})")
        else:
            # 没有配置，直接从数据中提取
            unique_combs = sorted(set(true_comb_indices) | set(pred_comb_indices))

        comb_to_idx = {comb: i for i, comb in enumerate(unique_combs)}

        # 构建混淆矩阵
        n_combs = len(unique_combs)
        confusion = np.zeros((n_combs, n_combs), dtype=int)

        for i in range(len(labels_deception)):
            true_comb = true_comb_indices[i]
            pred_comb = pred_comb_indices[i]

            if true_comb in comb_to_idx and pred_comb in comb_to_idx:
                confusion[comb_to_idx[true_comb], comb_to_idx[pred_comb]] += 1

        # 标记 Seen/Unseen
        seen_set = set()
        unseen_set = set()

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

        # 构建组合名称列表
        comb_names = []
        for comb in unique_combs:
            name = indices_to_name(comb)
            if seen_set or unseen_set:
                if comb in seen_set:
                    name = f"[S] {name}"
                elif comb in unseen_set:
                    name = f"[U] {name}"
            comb_names.append(name)

        # 绘图
        fig_size = max(12, n_combs * 0.6)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
                    xticklabels=comb_names, yticklabels=comb_names, ax=ax)
        ax.set_xlabel('预测组合 (Predicted Combination)')
        ax.set_ylabel('真实组合 (True Combination)')
        ax.set_title('组合混淆矩阵 - [S]=已见(Seen), [U]=未见(Unseen)')
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
        self,
        metrics: dict,
        seen_combinations: list = None,
        unseen_combinations: list = None
    ):
        """打印组合级别的评估指标"""
        labels_deception = metrics.get("labels_deception")
        labels_suppression = metrics.get("labels_suppression")
        preds_deception = metrics.get("preds_deception")
        preds_suppression = metrics.get("preds_suppression")

        if labels_deception is None:
            return

        # 构建 seen/unseen 集合
        seen_set = set()
        unseen_set = set()

        if seen_combinations:
            for comb in seen_combinations:
                if isinstance(comb, list):
                    seen_set.add("+".join(sorted(comb)))
        if unseen_combinations:
            for comb in unseen_combinations:
                if isinstance(comb, list):
                    unseen_set.add("+".join(sorted(comb)))

        # 统计 seen/unseen 准确率
        seen_correct, seen_total = 0, 0
        unseen_correct, unseen_total = 0, 0

        def get_combination_label(label_d, label_s):
            d_idx = np.argmax(label_d)
            s_idx = np.argmax(label_s)
            d_name = self.deception_classes_with_none[d_idx]
            s_name = self.suppression_classes_with_none[s_idx]
            parts = []
            if d_name != "无欺骗干扰":
                parts.append(d_name)
            if s_name != "无压制干扰":
                parts.append(s_name)
            return "+".join(sorted(parts)) if parts else "None"

        for i in range(len(labels_deception)):
            true_comb = get_combination_label(labels_deception[i], labels_suppression[i])
            pred_comb = get_combination_label(preds_deception[i], preds_suppression[i])

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
        print("组合级别评估结果 (Combination-Level Results)")
        print("=" * 80)

        if seen_total > 0:
            print(f"Seen Accuracy:    {seen_correct / seen_total:.4f} ({seen_correct}/{seen_total} samples)")
        if unseen_total > 0:
            print(f"Unseen Accuracy:  {unseen_correct / unseen_total:.4f} ({unseen_correct}/{unseen_total} samples)")

        print(f"\nOverall Combined Accuracy: {metrics['combined_accuracy']:.4f}")

    def save_stft_predictions(
        self,
        data_loader: DataLoader,
        output_dir: str,
        max_samples: int = 100,
        random_sample: bool = True
    ):
        """
        保存STFT图像和预测结果

        Args:
            data_loader: 数据加载器
            output_dir: 输出目录
            max_samples: 最大保存样本数
            random_sample: 是否随机采样
        """
        import cv2

        self.model.eval()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        correct_dir = output_path / "correct"
        wrong_dir = output_path / "wrong"
        correct_dir.mkdir(exist_ok=True)
        wrong_dir.mkdir(exist_ok=True)

        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 先收集所有数据（用于随机采样）
        all_samples = []

        print(f"Collecting samples for STFT saving (random_sample={random_sample})...")

        for batch_data in tqdm(data_loader, desc="Collecting"):
            (stft_images, text_tokens_deception, text_tokens_suppression,
             labels_deception, labels_suppression, texts_deception, texts_suppression,
             metadata_list) = batch_data

            batch_size = stft_images.size(0)

            for i in range(batch_size):
                all_samples.append({
                    'image': stft_images[i].cpu(),  # 存储在CPU上
                    'label_deception': labels_deception[i].cpu(),
                    'label_suppression': labels_suppression[i].cpu(),
                    'meta': metadata_list[i]
                })

        # 随机采样
        total_samples = len(all_samples)
        if random_sample and max_samples < total_samples:
            import random
            indices = random.sample(range(total_samples), max_samples)
            selected_samples = [all_samples[i] for i in indices]
            print(f"Randomly selected {max_samples} samples from {total_samples} total")
        else:
            selected_samples = all_samples[:max_samples]
            print(f"Selected first {len(selected_samples)} samples")

        # 预测并保存
        saved_count = 0
        correct_count = 0
        wrong_count = 0

        for sample in tqdm(selected_samples, desc="Saving STFT images"):
            image = sample['image'].unsqueeze(0).to(self.device)
            label_deception = sample['label_deception']
            label_suppression = sample['label_suppression']

            # 调整图像尺寸
            if image.shape[-1] != 224:
                image = nn.functional.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)

            # 双分支预测
            deception_result, suppression_result = self.model.zero_shot_predict_dual(image)

            pred_deception_idx = deception_result['indices'][0, 0].item()
            pred_suppression_idx = suppression_result['indices'][0, 0].item()

            # 获取真实标签
            true_deception_idx = torch.argmax(label_deception).item()
            true_suppression_idx = torch.argmax(label_suppression).item()

            # 获取名称
            true_deception_name = self.deception_classes_with_none[true_deception_idx]
            true_suppression_name = self.suppression_classes_with_none[true_suppression_idx]
            pred_deception_name = self.deception_classes_with_none[pred_deception_idx]
            pred_suppression_name = self.suppression_classes_with_none[pred_suppression_idx]

            # 构建组合名称
            def get_comb_name(d_name, s_name):
                parts = []
                if d_name != "无欺骗干扰":
                    parts.append(d_name)
                if s_name != "无压制干扰":
                    parts.append(s_name)
                return "+".join(sorted(parts)) if parts else "None"

            true_name = get_comb_name(true_deception_name, true_suppression_name)
            pred_name = get_comb_name(pred_deception_name, pred_suppression_name)

            # 判断是否正确
            is_correct = (pred_deception_idx == true_deception_idx and
                          pred_suppression_idx == true_suppression_idx)

            # 反归一化图像
            img = image[0].cpu().numpy()
            mean = np.array([0.48145466, 0.4578275, 0.40821073])
            std = np.array([0.26862954, 0.26130258, 0.27577711])
            img = img * std[:, None, None] + mean[:, None, None]
            img = np.clip(img, 0, 1)
            img = np.transpose(img, (1, 2, 0))
            img = (img * 255).astype(np.uint8)

            # 保存图像
            if is_correct:
                save_dir = correct_dir
                correct_count += 1
            else:
                save_dir = wrong_dir
                wrong_count += 1

            filename = f"{saved_count:05d}_true_{true_name}_pred_{pred_name}.png"
            filepath = save_dir / filename
            cv2.imwrite(str(filepath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            saved_count += 1

        print(f"\nSaved {saved_count} STFT images:")
        print(f"  Correct predictions: {correct_count} -> {correct_dir}")
        print(f"  Wrong predictions: {wrong_count} -> {wrong_dir}")
        print(f"  Accuracy: {correct_count / saved_count:.4f}")

        # 保存统计信息
        with open(output_path / "stft_summary.txt", 'w', encoding='utf-8') as f:
            f.write(f"Total samples: {saved_count}\n")
            f.write(f"Correct predictions: {correct_count}\n")
            f.write(f"Wrong predictions: {wrong_count}\n")
            f.write(f"Accuracy: {correct_count / saved_count:.4f}\n")


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Dual-Branch CZSL Model")
    parser.add_argument("--config", type=str, default="multi/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=str, default="results/dual_branch")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_stft", action="store_true", help="Save STFT images with predictions")
    parser.add_argument("--max_stft_samples", type=int, default=100, help="Maximum number of STFT images to save")
    parser.add_argument("--random_sample", action="store_true", default=True, help="Randomly sample STFT images")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取干扰类型分组
    jamming_groups = config.get("jamming_groups", {})
    deception_classes = jamming_groups.get("deception", {}).get("classes", ["DFTJ", "ISRJ", "SMSPJ", "C&IJ", "CSJ"])
    suppression_classes = jamming_groups.get("suppression", {}).get("classes", ["AJ", "BJ", "SJ", "NCJ", "NPJ", "NFMJ", "NPMJ", "NAMJ", "PJ"])

    # 获取所有类别名称（用于排序）
    all_class_names = [cls["name"] for cls in config.get("jamming_classes", [])]

    # 获取 seen/unseen 组合配置
    czsl_config = config.get("czsl", {})
    seen_combinations = czsl_config.get("seen_combinations", [])
    unseen_combinations = czsl_config.get("unseen_combinations", [])

    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # 检测模型类型
    state_dict = checkpoint["model_state_dict"]
    is_multishape = any(k.startswith("visual.patch_embeds") for k in state_dict.keys())

    # 创建模型
    if is_multishape:
        print("\nDetected Multi-Shape Patch ViT model from checkpoint")
        model = create_multi_shape_dual_branch_model(config, device=str(device))
    else:
        print("\nDetected CLIP Dual-Branch model from checkpoint")
        model = create_dual_branch_model(config, device=str(device))

    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # 缓存文本特征
    model.cache_text_features_dual()

    # 创建数据加载器
    train_loader, val_loader, test_loader, _, _ = create_dual_branch_dataloaders(config, load_test=True)

    if args.split == "train":
        data_loader = train_loader
    elif args.split == "val":
        data_loader = val_loader
    else:
        data_loader = test_loader

    # 创建评估器
    evaluator = DualBranchEvaluator(
        model=model,
        device=device,
        deception_classes=deception_classes,
        suppression_classes=suppression_classes
    )

    # 评估
    print(f"\nEvaluating on {args.split} set...")
    metrics = evaluator.evaluate(data_loader, debug=args.debug)

    # 打印结果
    evaluator.print_metrics(metrics)

    # 保存结果 (排除 numpy 数组)
    import json
    metrics_to_save = {
        "deception": {k: v for k, v in metrics["deception"].items() if not isinstance(v, np.ndarray)},
        "suppression": {k: v for k, v in metrics["suppression"].items() if not isinstance(v, np.ndarray)},
        "combined_accuracy": metrics["combined_accuracy"],
        "total_samples": metrics["total_samples"]
    }
    results_path = output_dir / f"dual_branch_results_{args.split}.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")

    # 可视化
    print("\nGenerating visualizations...")

    # 混淆矩阵
    evaluator.plot_confusion_matrices(
        metrics,
        save_path=str(output_dir / f"confusion_matrices_{args.split}.png")
    )

    # 每个类别的 F1 分数
    evaluator.plot_per_class_f1(
        metrics,
        save_path=str(output_dir / f"per_class_f1_{args.split}.png")
    )

    # 指标汇总
    evaluator.plot_metrics_summary(
        metrics,
        save_path=str(output_dir / f"metrics_summary_{args.split}.png")
    )

    # 组合混淆矩阵
    evaluator.plot_combination_confusion_matrix(
        metrics,
        save_path=str(output_dir / f"combination_confusion_{args.split}.png"),
        seen_combinations=seen_combinations,
        unseen_combinations=unseen_combinations,
        all_class_names=all_class_names
    )

    # 打印组合级别指标
    evaluator.print_combination_metrics(
        metrics,
        seen_combinations=seen_combinations,
        unseen_combinations=unseen_combinations
    )

    # 保存 STFT 图像
    if args.save_stft:
        # 释放原数据加载器的内存
        del train_loader, val_loader, test_loader, data_loader
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 创建新的数据加载器，禁用pin_memory以避免显存问题
        print(f"\nCreating data loader for STFT saving (pin_memory=False)...")
        train_loader, val_loader, test_loader, _, _ = create_dual_branch_dataloaders(
            config, load_test=True, pin_memory=False, num_workers=0
        )
        if args.split == "train":
            stft_loader = train_loader
        elif args.split == "val":
            stft_loader = val_loader
        else:
            stft_loader = test_loader

        print(f"Saving STFT images with predictions...")
        stft_dir = output_dir / "stft"
        evaluator.save_stft_predictions(
            data_loader=stft_loader,
            output_dir=str(stft_dir),
            max_samples=args.max_stft_samples,
            random_sample=args.random_sample
        )

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
