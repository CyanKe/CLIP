"""
CZSL评估脚本 - 支持零样本组合识别评估
python -m multi.evaluate_czsl --checkpoint checkpoints/czsl_best_model.pt --mode zero_shot

python -m multi.evaluate_czsl --checkpoint checkpoints/czsl_best_model.pt --mode zero_shot --split test --visualize --output_dir results
 --save_stft

python -m multi.evaluate_czsl --checkpoint checkpoints/czsl_best_model.pt --mode by_jnr --split test --output_dir results
"""
# pylint: disable=no-member

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
from sklearn.manifold import TSNE

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi.model import create_czsl_model, CLIPForCZSL
from multi.data import create_czsl_dataloaders, STFTDataset, collate_fn
import clip


def create_jnr_dataloaders(
    config: dict,
    split: str = 'test',
    normalization_stats: dict = None,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> dict:
    """
    为每个JNR级别创建独立的数据加载器

    Args:
        config: 配置字典
        split: 数据划分 ('train', 'val', 'test')
        normalization_stats: 归一化统计量
        batch_size: 批次大小
        num_workers: worker数量
        pin_memory: 是否pin memory

    Returns:
        字典 {jnr_level: DataLoader}
    """
    data_config = config.get('data', {})
    base_path = data_config.get('base_path')
    jnr_start = data_config.get('jnr_start', 0)
    jnr_end = data_config.get('jnr_end', 20)
    jnr_step = data_config.get('jnr_step', 5)

    class_names = [cls['name'] for cls in config.get('jamming_classes', [])]
    jnr_levels = list(range(jnr_start, jnr_end + 1, jnr_step))

    jnr_loaders = {}

    for jnr in jnr_levels:
        jnr_folder = f"JNR_{'+' if jnr >= 0 else ''}{jnr}"
        data_folder = os.path.join(base_path, jnr_folder)

        stft_file = os.path.join(data_folder, f'{split}_echo_stfts.mat')
        metadata_file = os.path.join(data_folder, f'{split}_echo_metadata.json')

        if not os.path.exists(stft_file) or not os.path.exists(metadata_file):
            print(f"Warning: Data not found for {jnr_folder}/{split}, skipping...")
            continue

        dataset = STFTDataset(
            stft_file=stft_file,
            metadata_file=metadata_file,
            normalization_stats=normalization_stats,
            class_names=class_names,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # 评估时不打乱
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

        jnr_loaders[jnr] = loader
        print(f"Loaded {split} data from {jnr_folder}: {len(dataset)} samples")

    return jnr_loaders


class CZSLEvaluator:
    """
    CZSL评估器
    支持零样本组合识别评估
    """

    def __init__(
        self,
        model: CLIPForCZSL,
        device: torch.device,
        class_names: list
    ):
        """
        初始化评估器

        Args:
            model: CZSL模型
            device: 计算设备
            class_names: 类别名称列表
        """
        self.model = model
        self.device = device
        self.class_names = class_names

    @torch.no_grad()
    def evaluate_zero_shot(
        self,
        data_loader: DataLoader,
        use_combinations: bool = True,
        debug: bool = False
    ) -> dict:
        """
        零样本评估

        Args:
            data_loader: 数据加载器
            use_combinations: 是否使用组合特征

        Returns:
            评估结果字典
        """
        self.model.eval()

        all_labels = []
        all_preds = []
        all_features = []  # 添加特征收集
        all_combination_correct = 0
        all_multilabel_correct = 0
        total_samples = 0

        eval_bar = tqdm(data_loader, desc="Zero-Shot Evaluation")

        # Debug: 显示缓存的文本特征
        if debug:
            print("\n" + "=" * 80)
            print("[DEBUG] Cached text features for inference:")
            print("=" * 80)
            cached_names = self.model._combination_names
            for i, name in enumerate(cached_names[:20]):  # 只显示前20个
                print(f"  [{i}] {name}")
            if len(cached_names) > 20:
                print(f"  ... ({len(cached_names) - 20} more)")
            print("=" * 80 + "\n")

        debug_done = False
        for batch_idx, (images, _, text_tokens, labels, texts, metas) in enumerate(eval_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 调整图像尺寸
            if images.shape[-1] != 224:
                images = nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

            batch_size = images.shape[0]

            # 提取图像特征
            image_features = self.model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            all_features.append(image_features.cpu().numpy())

            # 零样本预测
            if use_combinations:
                # 使用组合特征进行精确匹配
                similarities, indices, pred_names = self.model.zero_shot_predict(
                    images, use_combinations=True, top_k=1
                )

                # 解析预测的组合
                preds = torch.zeros(batch_size, len(self.class_names), device=self.device)
                for i, name in enumerate(pred_names):
                    if name and len(name) > 0:
                        comb_name = name[0]
                        # 解析组合名称 (如 "DFTJ+ISRJ")
                        parts = comb_name.split('+')
                        for part in parts:
                            if part in self.class_names:
                                preds[i, self.class_names.index(part)] = 1
            else:
                # 使用单类别特征进行多标签预测
                text_features = self.model.get_cached_text_features()
                image_features = self.model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logit_scale = self.model.model.logit_scale.exp()
                logits = logit_scale * (image_features @ text_features.T)

                # 多标签预测策略：softmax + top-k > threshold
                probs = torch.softmax(logits, dim=-1)
                num_classes = len(self.class_names)
                threshold = 1.0 / num_classes
                top_k = 3
                topk_values, topk_indices = torch.topk(probs, k=top_k, dim=-1)

                preds = torch.zeros(batch_size, num_classes, device=self.device)
                for b in range(batch_size):
                    for j, idx in enumerate(topk_indices[b]):
                        if topk_values[b, j] > threshold:
                            preds[b, idx] = 1.0

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

            # Debug: 显示第一个批次的详细信息
            if debug and not debug_done:
                print("\n" + "=" * 80)
                print(f"[DEBUG] Batch {batch_idx} — batch_size={batch_size}")
                print("=" * 80)
                # 显示缓存的文本描述（推理用）
                print("Cached text templates (for inference):")
                for i, name in enumerate(self.model._combination_names[:10]):
                    print(f"  [{i}] {name}")

                # 显示训练时的文本描述
                print("\nTraining texts (from data loader):")
                for i, txt in enumerate(texts[:5]):
                    print(f"  [{i}] {txt}")

                # 显示预测结果
                print("\nPredictions vs Labels:")
                for i in range(min(5, batch_size)):
                    true_label_indices = torch.where(labels[i] == 1)[0].tolist()
                    pred_label_indices = torch.where(preds[i] == 1)[0].tolist()
                    true_names = [self.class_names[idx] for idx in true_label_indices]
                    pred_names_list = [self.class_names[idx] for idx in pred_label_indices]
                    print(f"  [{i}] True: {true_names} | Pred: {pred_names_list}")

                # 显示相似度分数分布
                if use_combinations:
                    print("\nTop-5 similarity scores for sample 0:")
                    top5_vals, top5_idx = torch.topk(similarities[0], k=5)
                    for val, idx in zip(top5_vals, top5_idx):
                        print(f"  {self.model._combination_names[idx.item()]}: {val.item():.3f}")

                print("=" * 80 + "\n")
                debug_done = True

            # 计算组合匹配准确率
            for i in range(batch_size):
                true_set = set(torch.where(labels[i] == 1)[0].tolist())
                pred_set = set(torch.where(preds[i] == 1)[0].tolist())

                if true_set == pred_set:
                    all_combination_correct += 1

                # 多标签子集准确率
                if true_set.issubset(pred_set) or pred_set.issubset(true_set):
                    all_multilabel_correct += 1

            total_samples += batch_size

        # 合并结果
        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()

        # 计算指标
        metrics = {
            "combination_accuracy": all_combination_correct / total_samples,
            "partial_match_accuracy": all_multilabel_correct / total_samples,
            "f1_macro": f1_score(all_labels, all_preds, average='macro', zero_division=0),
            "f1_micro": f1_score(all_labels, all_preds, average='micro', zero_division=0),
            "precision_macro": precision_score(all_labels, all_preds, average='macro', zero_division=0),
            "recall_macro": recall_score(all_labels, all_preds, average='macro', zero_division=0),
        }

        # 每个类别的指标
        per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
        metrics["per_class"] = {
            "f1": per_class_f1,
            "precision": precision_score(all_labels, all_preds, average=None, zero_division=0),
            "recall": recall_score(all_labels, all_preds, average=None, zero_division=0)
        }

        # 合并特征
        all_features = np.concatenate(all_features, axis=0)

        return {
            "metrics": metrics,
            "labels": all_labels,
            "predictions": all_preds,
            "features": all_features
        }

    @torch.no_grad()
    def evaluate_by_combination_type(
        self,
        data_loader: DataLoader,
        seen_combinations: list = None,
        unseen_combinations: list = None,
        debug: bool = False
    ) -> dict:
        """
        按组合类型评估（Seen vs Unseen）

        Args:
            data_loader: 数据加载器
            seen_combinations: 已见组合索引列表
            unseen_combinations: 未见组合索引列表

        Returns:
            评估结果字典
        """
        self.model.eval()

        seen_results = {"correct": 0, "total": 0}
        unseen_results = {"correct": 0, "total": 0}
        other_results = {"correct": 0, "total": 0}

        # 转换为set便于查找
        seen_set = set(tuple(sorted(c)) for c in (seen_combinations or []))
        unseen_set = set(tuple(sorted(c)) for c in (unseen_combinations or []))

        # 收集所有预测结果（用于保存混淆矩阵等）
        all_labels = []
        all_preds = []
        all_features = []

        eval_bar = tqdm(data_loader, desc="Evaluating by Combination Type")

        # Debug: 显示缓存的文本特征
        if debug:
            print("\n" + "=" * 80)
            print("[DEBUG] Cached text features for inference:")
            print("=" * 80)
            for i, name in enumerate(self.class_names):
                print(f"  [{i}] {name}")
            print("=" * 80 + "\n")

        debug_done = False
        for batch_idx, (images, _, text_tokens, labels, texts, metas) in enumerate(eval_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            if images.shape[-1] != 224:
                images = nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

            # 零样本预测 - 使用单类别特征
            text_features = self.model.get_cached_text_features()
            image_features = self.model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            logit_scale = self.model.model.logit_scale.exp()
            logits = logit_scale * (image_features @ text_features.T)

            # 多标签预测策略：
            # 使用 softmax 获取概率分布，然后选择超过阈值的 top-k
            probs = torch.softmax(logits, dim=-1)  # [batch, num_classes]
            batch_size = images.shape[0]

            # 策略：选择概率超过阈值 且是 top-k 的类别
            num_classes = len(self.class_names)
            threshold = 1.0 / num_classes  # 基础阈值（均匀分布）

            # 最多选择 top-3（因为最多是2个干扰的组合）
            top_k = 3
            topk_values, topk_indices = torch.topk(probs, k=top_k, dim=-1)

            preds = torch.zeros(batch_size, num_classes, device=self.device)
            for b in range(batch_size):
                for j, idx in enumerate(topk_indices[b]):
                    # 只选择概率超过阈值的 top-k
                    if topk_values[b, j] > threshold:
                        preds[b, idx] = 1.0

            # 收集预测结果
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_features.append(image_features.cpu().numpy())

            # Debug: 显示第一个批次的详细信息
            if debug and not debug_done:
                print("\n" + "=" * 80)
                print(f"[DEBUG] Batch {batch_idx} — batch_size={batch_size}")
                print("=" * 80)

                # 显示训练时的文本描述
                print("Training texts (from data loader):")
                for i, txt in enumerate(texts[:5]):
                    print(f"  [{i}] {txt}")

                # 显示推理用的缓存模板
                print("\nInference templates (cached):")
                for i, cls in enumerate(self.class_names):
                    print(f"  [{i}] a radar signal with single jamming: {cls}")

                # 显示预测结果
                print("\nPredictions vs Labels (softmax + top-k > threshold):")
                for i in range(min(5, batch_size)):
                    true_label_indices = torch.where(labels[i] == 1)[0].tolist()
                    pred_label_indices = torch.where(preds[i] == 1)[0].tolist()
                    true_names = [self.class_names[idx] for idx in true_label_indices]
                    pred_names_list = [self.class_names[idx] for idx in pred_label_indices]
                    # 显示 top-5 概率
                    top5_vals, top5_idx = torch.topk(probs[i], k=5)
                    print(f"  [{i}] True: {true_names} | Pred: {pred_names_list}")
                    print(f"       Top-5: {[(self.class_names[idx.item()], f'{val.item():.3f}') for val, idx in zip(top5_vals, top5_idx)]}")

                print("=" * 80 + "\n")
                debug_done = True

            for i in range(batch_size):
                true_comb = tuple(sorted(torch.where(labels[i] == 1)[0].tolist()))
                pred_comb = tuple(sorted(torch.where(preds[i] == 1)[0].tolist()))

                is_correct = (true_comb == pred_comb)

                # 分类
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
            "other_samples": other_results["total"]
        }

        # 合并所有结果
        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()
        all_features = np.concatenate(all_features, axis=0)

        return {
            "metrics": metrics,
            "labels": all_labels,
            "predictions": all_preds,
            "features": all_features
        }

    def print_metrics(self, metrics: dict):
        """打印评估指标"""
        print("\n" + "=" * 60)
        print("Zero-Shot Evaluation Results")
        print("=" * 60)
        print(f"Combination Accuracy:     {metrics['combination_accuracy']:.4f}")
        print(f"Partial Match Accuracy:   {metrics['partial_match_accuracy']:.4f}")
        print(f"Macro F1 Score:           {metrics['f1_macro']:.4f}")
        print(f"Micro F1 Score:           {metrics['f1_micro']:.4f}")
        print(f"Macro Precision:          {metrics['precision_macro']:.4f}")
        print(f"Macro Recall:             {metrics['recall_macro']:.4f}")

        # 每个类别的指标
        print("\nPer-class Metrics:")
        print("-" * 60)
        print(f"{'Class':<10} {'F1':>8} {'Precision':>12} {'Recall':>10}")
        print("-" * 60)
        for i, name in enumerate(self.class_names[:len(metrics['per_class']['f1'])]):
            print(f"{name:<10} {metrics['per_class']['f1'][i]:>8.4f} "
                  f"{metrics['per_class']['precision'][i]:>12.4f} "
                  f"{metrics['per_class']['recall'][i]:>10.4f}")

    def plot_confusion_by_combination(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        save_path: str = None,
        seen_combinations: list = None,
        unseen_combinations: list = None
    ):
        """
        绘制组合混淆矩阵

        Args:
            labels: 真实标签
            preds: 预测标签
            save_path: 保存路径
            seen_combinations: 已见组合索引列表 (如 [[0], [1], [0, 2]])
            unseen_combinations: 未见组合索引列表
        """
        # 确定使用哪些组合
        if seen_combinations is not None or unseen_combinations is not None:
            # 使用配置中定义的组合，保持配置顺序
            configured_combs = []

            # 先添加 seen_combinations（保持配置顺序）
            if seen_combinations:
                for comb in seen_combinations:
                    comb_tuple = tuple(sorted(comb))
                    if comb_tuple not in configured_combs:
                        configured_combs.append(comb_tuple)

            # 再添加 unseen_combinations（保持配置顺序）
            if unseen_combinations:
                for comb in unseen_combinations:
                    comb_tuple = tuple(sorted(comb))
                    if comb_tuple not in configured_combs:
                        configured_combs.append(comb_tuple)

            unique_combs = configured_combs
            print(f"\nUsing configured combinations: {len(unique_combs)} "
                  f"(Seen: {len(seen_combinations or [])}, Unseen: {len(unseen_combinations or [])})")
        else:
            # 从数据中提取所有出现的组合
            true_combinations = []
            pred_combinations = []

            for i in range(len(labels)):
                true_comb = tuple(sorted(np.where(labels[i] == 1)[0].tolist()))
                pred_comb = tuple(sorted(np.where(preds[i] == 1)[0].tolist()))
                true_combinations.append(true_comb)
                pred_combinations.append(pred_comb)

            unique_combs = sorted(set(true_combinations + pred_combinations))

        comb_to_idx = {comb: i for i, comb in enumerate(unique_combs)}

        # 构建混淆矩阵
        n_combs = len(unique_combs)
        confusion = np.zeros((n_combs, n_combs), dtype=int)

        # 统计不在配置中的样本数
        other_count = 0

        for i in range(len(labels)):
            true_comb = tuple(sorted(np.where(labels[i] == 1)[0].tolist()))
            pred_comb = tuple(sorted(np.where(preds[i] == 1)[0].tolist()))

            # 检查是否在配置的组合中
            if true_comb in comb_to_idx and pred_comb in comb_to_idx:
                confusion[comb_to_idx[true_comb], comb_to_idx[pred_comb]] += 1
            else:
                other_count += 1

        if other_count > 0:
            print(f"Samples not in configured combinations: {other_count}")

        # 转换为名称，标记 Seen/Unseen
        seen_set = set(tuple(sorted(c)) for c in (seen_combinations or []))
        unseen_set = set(tuple(sorted(c)) for c in (unseen_combinations or []))

        comb_names = []
        for comb in unique_combs:
            if len(comb) == 0:
                name = "None"
            else:
                name = "+".join([self.class_names[i] for i in comb])

            # 添加标记
            if seen_combinations is not None or unseen_combinations is not None:
                if comb in seen_set:
                    name = f"[S] {name}"
                elif comb in unseen_set:
                    name = f"[U] {name}"

            comb_names.append(name)

        # 绘图
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
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_feature_tsne(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        save_path: str = None,
        seen_combinations: list = None,
        unseen_combinations: list = None
    ):
        """
        绘制特征的t-SNE可视化

        Args:
            features: 特征向量 [num_samples, embed_dim]
            labels: 标签 [num_samples, num_classes]
            save_path: 保存路径
            seen_combinations: 已见组合索引列表
            unseen_combinations: 未见组合索引列表
        """
        print("Computing t-SNE projection...")

        num_samples = len(features)
        perplexity = min(30, num_samples - 1) if num_samples > 1 else 1

        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        features_2d = tsne.fit_transform(features)

        # 将多标签转换为组合名称
        comb_labels = []
        for label in labels:
            active = tuple(sorted(np.where(label == 1)[0].tolist()))
            comb_labels.append(active)

        # 转换为名称
        seen_set = set(tuple(sorted(c)) for c in (seen_combinations or []))
        unseen_set = set(tuple(sorted(c)) for c in (unseen_combinations or []))

        unique_combs = sorted(set(comb_labels))
        comb_to_name = {}
        for comb in unique_combs:
            if len(comb) == 0:
                comb_to_name[comb] = "None"
            else:
                comb_to_name[comb] = "+".join([self.class_names[i] for i in comb])

        # 为每个组合分配颜色
        num_combs = len(unique_combs)
        colors = plt.cm.tab20(np.linspace(0, 1, max(20, num_combs)))

        # 绘图
        fig, ax = plt.subplots(figsize=(14, 10))

        for idx, comb in enumerate(unique_combs):
            # 使用列表推导式而不是numpy数组比较
            mask = np.array([c == comb for c in comb_labels])
            if mask.sum() > 0:
                name = comb_to_name[comb]
                # 添加标记
                if comb in seen_set:
                    name = f"[S] {name}"
                    marker = 'o'
                elif comb in unseen_set:
                    name = f"[U] {name}"
                    marker = '^'
                else:
                    marker = 's'

                ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                          c=[colors[idx % 20]], label=name, alpha=0.6, s=30, marker=marker)

        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title('Feature Space Visualization (t-SNE)\n[S]=Seen, [U]=Unseen, □=Other')
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
        self,
        features: np.ndarray,
        labels: np.ndarray,
        save_path: str = None,
        seen_combinations: list = None,
        unseen_combinations: list = None
    ):
        """
        绘制特征的UMAP可视化

        Args:
            features: 特征向量 [num_samples, embed_dim]
            labels: 标签 [num_samples, num_classes]
            save_path: 保存路径
            seen_combinations: 已见组合索引列表
            unseen_combinations: 未见组合索引列表
        """
        try:
            import umap
        except ImportError:
            print("UMAP not installed. Install with: pip install umap-learn")
            return

        print("Computing UMAP projection...")

        # UMAP降维
        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=15,
            min_dist=0.1
        )
        features_2d = reducer.fit_transform(features)

        # 将多标签转换为组合名称
        comb_labels = []
        for label in labels:
            active = tuple(sorted(np.where(label == 1)[0].tolist()))
            comb_labels.append(active)

        # 转换为名称
        seen_set = set(tuple(sorted(c)) for c in (seen_combinations or []))
        unseen_set = set(tuple(sorted(c)) for c in (unseen_combinations or []))

        unique_combs = sorted(set(comb_labels))
        comb_to_name = {}
        for comb in unique_combs:
            if len(comb) == 0:
                comb_to_name[comb] = "None"
            else:
                comb_to_name[comb] = "+".join([self.class_names[i] for i in comb])

        # 为每个组合分配颜色
        num_combs = len(unique_combs)
        colors = plt.cm.tab20(np.linspace(0, 1, max(20, num_combs)))

        # 绘图
        fig, ax = plt.subplots(figsize=(14, 10))

        for idx, comb in enumerate(unique_combs):
            mask = np.array([c == comb for c in comb_labels])
            if mask.sum() > 0:
                name = comb_to_name[comb]
                # 添加标记
                if comb in seen_set:
                    name = f"[S] {name}"
                    marker = 'o'
                elif comb in unseen_set:
                    name = f"[U] {name}"
                    marker = '^'
                else:
                    marker = 's'

                ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                          c=[colors[idx % 20]], label=name, alpha=0.6, s=30, marker=marker)

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('Feature Space Visualization (UMAP)\n[S]=Seen, [U]=Unseen, □=Other')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"UMAP plot saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_label_cooccurrence(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        save_path: str = None
    ):
        """
        绘制标签共现矩阵

        Args:
            labels: 真实标签
            preds: 预测标签
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 真实标签共现
        true_cooc = (labels.T @ labels).astype(int)
        num_classes = labels.shape[1]
        sns.heatmap(true_cooc, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names[:num_classes],
                    yticklabels=self.class_names[:num_classes],
                    ax=axes[0])
        axes[0].set_title('True Label Co-occurrence')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].tick_params(axis='y', rotation=0)

        # 预测标签共现
        pred_cooc = (preds.T @ preds).astype(int)
        sns.heatmap(pred_cooc, annot=True, fmt='d', cmap='Greens',
                    xticklabels=self.class_names[:num_classes],
                    yticklabels=self.class_names[:num_classes],
                    ax=axes[1])
        axes[1].set_title('Predicted Label Co-occurrence')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].tick_params(axis='y', rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Label co-occurrence plot saved to {save_path}")
        else:
            plt.show()
        plt.close()

    @torch.no_grad()
    def evaluate_by_jnr(
        self,
        jnr_loaders: dict,
        seen_combinations: list = None,
        unseen_combinations: list = None,
        debug: bool = False
    ) -> dict:
        """
        按JNR级别分别评估

        Args:
            jnr_loaders: 字典 {jnr_level: DataLoader}
            seen_combinations: 已见组合索引列表
            unseen_combinations: 未见组合索引列表
            debug: 是否显示调试信息

        Returns:
            各JNR级别的评估结果
        """
        results = {}
        seen_set = set(tuple(sorted(c)) for c in (seen_combinations or []))
        unseen_set = set(tuple(sorted(c)) for c in (unseen_combinations or []))
        num_classes = len(self.class_names)

        for jnr, data_loader in sorted(jnr_loaders.items()):
            print(f"\nEvaluating JNR={jnr}...")

            all_labels = []
            all_preds = []
            seen_correct, seen_total = 0, 0
            unseen_correct, unseen_total = 0, 0

            # 每个类别的统计: TP, FP, FN, TN
            class_stats = np.zeros((num_classes, 4), dtype=np.int64)

            for images, _, _, labels, _, _ in tqdm(
                data_loader, desc=f"JNR={jnr}"
            ):
                images = images.to(self.device)
                labels = labels.to(self.device)

                if images.shape[-1] != 224:
                    images = nn.functional.interpolate(
                        images, size=(224, 224), mode='bilinear', align_corners=False
                    )

                # 预测
                text_features = self.model.get_cached_text_features()
                image_features = self.model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logit_scale = self.model.model.logit_scale.exp()
                logits = logit_scale * (image_features @ text_features.T)

                # 多标签预测
                probs = torch.softmax(logits, dim=-1)
                batch_size = images.shape[0]
                threshold = 1.0 / num_classes
                top_k = 3
                topk_values, topk_indices = torch.topk(probs, k=top_k, dim=-1)

                preds = torch.zeros(batch_size, num_classes, device=self.device)
                for b in range(batch_size):
                    for j, idx in enumerate(topk_indices[b]):
                        if topk_values[b, j] > threshold:
                            preds[b, idx] = 1.0

                all_labels.append(labels.cpu())
                all_preds.append(preds.cpu())

                # 统计每个类别的 TP, FP, FN
                for c in range(num_classes):
                    true_c = labels[:, c].cpu().numpy()
                    pred_c = preds[:, c].cpu().numpy()
                    class_stats[c, 0] += np.sum((true_c == 1) & (pred_c == 1))  # TP
                    class_stats[c, 1] += np.sum((true_c == 0) & (pred_c == 1))  # FP
                    class_stats[c, 2] += np.sum((true_c == 1) & (pred_c == 0))  # FN
                    class_stats[c, 3] += np.sum((true_c == 0) & (pred_c == 0))  # TN

                # 统计seen/unseen准确率
                for i in range(batch_size):
                    true_comb = tuple(
                        sorted(torch.where(labels[i] == 1)[0].tolist())
                    )
                    pred_comb = tuple(
                        sorted(torch.where(preds[i] == 1)[0].tolist())
                    )
                    is_correct = (true_comb == pred_comb)

                    if true_comb in seen_set:
                        seen_total += 1
                        if is_correct:
                            seen_correct += 1
                    elif true_comb in unseen_set:
                        unseen_total += 1
                        if is_correct:
                            unseen_correct += 1

            # 计算指标
            all_labels = torch.cat(all_labels).numpy()
            all_preds = torch.cat(all_preds).numpy()

            # 计算每个类别的准确率 (recall)
            per_class_recall = np.zeros(num_classes)
            per_class_precision = np.zeros(num_classes)
            per_class_f1 = np.zeros(num_classes)
            for c in range(num_classes):
                tp, fp, fn, tn = class_stats[c]
                per_class_recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0
                per_class_precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0
                if per_class_precision[c] + per_class_recall[c] > 0:
                    per_class_f1[c] = 2 * per_class_precision[c] * per_class_recall[c] / (
                        per_class_precision[c] + per_class_recall[c]
                    )

            results[jnr] = {
                "combination_accuracy": seen_correct + unseen_correct,
                "total_samples": seen_total + unseen_total,
                "seen_accuracy": seen_correct / seen_total if seen_total > 0 else 0,
                "seen_samples": seen_total,
                "unseen_accuracy": unseen_correct / unseen_total if unseen_total > 0 else 0,
                "unseen_samples": unseen_total,
                "f1_macro": f1_score(all_labels, all_preds, average='macro', zero_division=0),
                "f1_micro": f1_score(all_labels, all_preds, average='micro', zero_division=0),
                "precision_macro": precision_score(
                    all_labels, all_preds, average='macro', zero_division=0
                ),
                "recall_macro": recall_score(
                    all_labels, all_preds, average='macro', zero_division=0
                ),
                "per_class_recall": per_class_recall,
                "per_class_precision": per_class_precision,
                "per_class_f1": per_class_f1,
            }

        return results

    def print_jnr_results(self, results: dict, save_path: str = None):
        """打印并保存JNR评估结果"""
        # 打印总体结果
        print("\n" + "=" * 80)
        print("Evaluation Results by JNR Level")
        print("=" * 80)
        print(f"{'JNR':>6} | {'Accuracy':>10} | {'F1_Macro':>10} | "
              f"{'Seen_Acc':>10} | {'Unseen_Acc':>10} | {'Samples':>8}")
        print("-" * 80)

        lines = []
        for jnr, metrics in sorted(results.items()):
            acc = (metrics["combination_accuracy"] / metrics["total_samples"]
                   if metrics["total_samples"] > 0 else 0)
            print(f"{jnr:>6} | {acc:>10.4f} | {metrics['f1_macro']:>10.4f} | "
                  f"{metrics['seen_accuracy']:>10.4f} | "
                  f"{metrics['unseen_accuracy']:>10.4f} | "
                  f"{metrics['total_samples']:>8}")
            lines.append(f"{jnr},{acc:.4f},{metrics['f1_macro']:.4f},"
                        f"{metrics['seen_accuracy']:.4f},"
                        f"{metrics['unseen_accuracy']:.4f},"
                        f"{metrics['total_samples']}")

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("JNR,Accuracy,F1_Macro,Seen_Acc,Unseen_Acc,Samples\n")
                f.write("\n".join(lines))
            print(f"\nResults saved to {save_path}")

        # 打印每个类别的指标 (Recall, Precision, F1)
        print("\n" + "=" * 80)
        print("Per-Class Recall by JNR Level")
        print("=" * 80)
        header = f"{'JNR':>6} | " + " | ".join(f"{name:>8}" for name in self.class_names)
        print(header)
        print("-" * len(header))

        per_class_recall_lines = []
        for jnr, metrics in sorted(results.items()):
            recalls = metrics.get("per_class_recall", [])
            row = f"{jnr:>6} | " + " | ".join(f"{r:>8.4f}" for r in recalls)
            print(row)
            per_class_recall_lines.append(
                f"{jnr}," + ",".join(f"{r:.4f}" for r in recalls)
            )

        print("\n" + "=" * 80)
        print("Per-Class Precision by JNR Level")
        print("=" * 80)
        print(header)
        print("-" * len(header))

        per_class_precision_lines = []
        for jnr, metrics in sorted(results.items()):
            precisions = metrics.get("per_class_precision", [])
            row = f"{jnr:>6} | " + " | ".join(f"{p:>8.4f}" for p in precisions)
            print(row)
            per_class_precision_lines.append(
                f"{jnr}," + ",".join(f"{p:.4f}" for p in precisions)
            )

        print("\n" + "=" * 80)
        print("Per-Class F1 by JNR Level")
        print("=" * 80)
        print(header)
        print("-" * len(header))

        per_class_f1_lines = []
        for jnr, metrics in sorted(results.items()):
            f1s = metrics.get("per_class_f1", [])
            row = f"{jnr:>6} | " + " | ".join(f"{f:>8.4f}" for f in f1s)
            print(row)
            per_class_f1_lines.append(
                f"{jnr}," + ",".join(f"{f:.4f}" for f in f1s)
            )

        # 保存每个类别的结果
        if save_path:
            # Recall
            recall_path = save_path.replace('.csv', '_per_class_recall.csv')
            with open(recall_path, 'w', encoding='utf-8') as f:
                f.write("JNR," + ",".join(self.class_names) + "\n")
                f.write("\n".join(per_class_recall_lines))
            print(f"\nPer-class recall saved to {recall_path}")

            # Precision
            precision_path = save_path.replace('.csv', '_per_class_precision.csv')
            with open(precision_path, 'w', encoding='utf-8') as f:
                f.write("JNR," + ",".join(self.class_names) + "\n")
                f.write("\n".join(per_class_precision_lines))
            print(f"Per-class precision saved to {precision_path}")

            # F1
            f1_path = save_path.replace('.csv', '_per_class_f1.csv')
            with open(f1_path, 'w', encoding='utf-8') as f:
                f.write("JNR," + ",".join(self.class_names) + "\n")
                f.write("\n".join(per_class_f1_lines))
            print(f"Per-class F1 saved to {f1_path}")

    def plot_jnr_metrics(self, results: dict, save_path: str = None):
        """绘制JNR指标曲线图"""
        jnrs = sorted(results.keys())
        accuracies = []
        f1_macros = []
        seen_accs = []
        unseen_accs = []

        for jnr in jnrs:
            m = results[jnr]
            acc = (m["combination_accuracy"] / m["total_samples"]
                   if m["total_samples"] > 0 else 0)
            accuracies.append(acc)
            f1_macros.append(m["f1_macro"])
            seen_accs.append(m["seen_accuracy"])
            unseen_accs.append(m["unseen_accuracy"])

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 准确率曲线
        axes[0, 0].plot(jnrs, accuracies, 'b-o', label='Overall Accuracy', linewidth=2)
        axes[0, 0].plot(jnrs, seen_accs, 'g-s', label='Seen Accuracy', linewidth=2)
        axes[0, 0].plot(jnrs, unseen_accs, 'r-^', label='Unseen Accuracy', linewidth=2)
        axes[0, 0].set_xlabel('JNR (dB)')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy vs JNR Level')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # F1曲线
        axes[0, 1].plot(jnrs, f1_macros, 'b-o', label='F1 Macro', linewidth=2)
        axes[0, 1].set_xlabel('JNR (dB)')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('F1 Score vs JNR Level')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 每个类别的召回率曲线
        ax = axes[1, 0]
        for c, name in enumerate(self.class_names):
            recalls = [results[jnr].get("per_class_recall", [0] * len(self.class_names))[c]
                       for jnr in jnrs]
            ax.plot(jnrs, recalls, '-o', label=name, linewidth=1.5, markersize=4)
        ax.set_xlabel('JNR (dB)')
        ax.set_ylabel('Recall (Detection Rate)')
        ax.set_title('Per-Class Recall vs JNR Level')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)

        # 每个类别的F1曲线
        ax = axes[1, 1]
        for c, name in enumerate(self.class_names):
            f1s = [results[jnr].get("per_class_f1", [0] * len(self.class_names))[c]
                   for jnr in jnrs]
            ax.plot(jnrs, f1s, '-o', label=name, linewidth=1.5, markersize=4)
        ax.set_xlabel('JNR (dB)')
        ax.set_ylabel('F1 Score')
        ax.set_title('Per-Class F1 Score vs JNR Level')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"JNR metrics plot saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def save_stft_predictions(
        self,
        data_loader: DataLoader,
        output_dir: str,
        max_samples: int = None,
        use_combinations: bool = True
    ):
        """
        保存STFT图像和预测结果

        Args:
            data_loader: 数据加载器
            output_dir: 输出目录
            max_samples: 最大保存样本数（None表示全部保存）
            use_combinations: 是否使用组合特征预测
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

        saved_count = 0
        correct_count = 0
        wrong_count = 0

        eval_bar = tqdm(data_loader, desc="Saving STFT images")

        for images, _, text_tokens, labels, texts, metas in eval_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 调整图像尺寸
            if images.shape[-1] != 224:
                images = nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

            batch_size = images.shape[0]

            # 零样本预测 - 使用单类别特征
            text_features = self.model.get_cached_text_features()
            image_features = self.model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            logit_scale = self.model.model.logit_scale.exp()
            logits = logit_scale * (image_features @ text_features.T)

            # 多标签预测策略：softmax + top-k > threshold
            probs = torch.softmax(logits, dim=-1)
            num_classes = len(self.class_names)
            threshold = 1.0 / num_classes
            top_k = 3
            topk_values, topk_indices = torch.topk(probs, k=top_k, dim=-1)

            preds = torch.zeros(batch_size, num_classes, device=self.device)
            for b in range(batch_size):
                for j, idx in enumerate(topk_indices[b]):
                    if topk_values[b, j] > threshold:
                        preds[b, idx] = 1.0

            # 保存每张图像
            for i in range(batch_size):
                if max_samples and saved_count >= max_samples:
                    break

                # 获取标签名称
                true_indices = torch.where(labels[i] == 1)[0].tolist()
                pred_indices = torch.where(preds[i] == 1)[0].tolist()

                true_name = "+".join([self.class_names[idx] for idx in true_indices]) if true_indices else "None"
                pred_name = "+".join([self.class_names[idx] for idx in pred_indices]) if pred_indices else "None"

                # 判断是否正确
                is_correct = set(true_indices) == set(pred_indices)

                # 转换图像
                img = images[i].cpu().numpy()
                # 反归一化
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

            if max_samples and saved_count >= max_samples:
                break

        print(f"\nSaved {saved_count} STFT images:")
        print(f"  Correct predictions: {correct_count} -> {correct_dir}")
        print(f"  Wrong predictions: {wrong_count} -> {wrong_dir}")

        # 保存统计信息
        with open(output_path / "summary.txt", 'w', encoding='utf-8') as f:
            f.write(f"Total samples: {saved_count}\n")
            f.write(f"Correct predictions: {correct_count}\n")
            f.write(f"Wrong predictions: {wrong_count}\n")
            f.write(f"Accuracy: {correct_count / saved_count:.4f}\n")


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def convert_combination_names_to_indices(
    combinations: list,
    class_names: list
) -> list:
    """
    将组合中的类别名称转换为索引

    Args:
        combinations: 组合列表，如 [["DFTJ"], ["DFTJ", "ISRJ"]]
        class_names: 类别名称列表

    Returns:
        索引组合列表，如 [[0], [0, 1]]
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


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Evaluate CZSL Model")
    parser.add_argument("--config", type=str, default="multi/config.yaml",
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--mode", type=str, default="by_combination",
                        choices=["zero_shot", "by_combination", "by_jnr"],
                        help="Evaluation mode")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Data split to evaluate")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations (t-SNE, confusion matrix, etc.)")
    parser.add_argument("--save_stft", action="store_true",
                        help="Save STFT images with predictions")
    parser.add_argument("--max_stft_samples", type=int, default=None,
                        help="Maximum number of STFT images to save (default: all)")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 类别名称
    class_names = [cls["name"] for cls in config.get("jamming_classes", [])]

    # 创建模型
    model = create_czsl_model(config, device=str(device))

    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # 过滤掉尺寸不匹配的层（分类器）
    state_dict = checkpoint["model_state_dict"]
    model_state = model.state_dict()

    # 检测并过滤不匹配的键
    filtered_state_dict = {}
    mismatched_keys = []
    for key, value in state_dict.items():
        if key in model_state:
            if model_state[key].shape != value.shape:
                mismatched_keys.append(f"{key}: checkpoint {value.shape} vs model {model_state[key].shape}")
            else:
                filtered_state_dict[key] = value
        else:
            # 键不存在于模型中，跳过
            pass

    if mismatched_keys:
        print(f"Warning: Skipping mismatched layers due to size mismatch:")
        for key in mismatched_keys:
            print(f"  - {key}")
        print(f"  (This is OK for zero-shot prediction which doesn't use the classifier)")

    model.load_state_dict(filtered_state_dict, strict=False)

    print(f"Loaded checkpoint from {args.checkpoint}")

    # 缓存文本特征
    model.cache_text_features(max_combination_size=2, include_single=True)

    # 创建数据加载器
    train_loader, val_loader, test_loader, num_classes = create_czsl_dataloaders(config)

    # 选择数据集
    if args.split == "train":
        data_loader = train_loader
    elif args.split == "val":
        data_loader = val_loader
    else:
        data_loader = test_loader

    # 创建评估器
    evaluator = CZSLEvaluator(
        model=model,
        device=device,
        class_names=class_names
    )

    # 评估
    if args.mode == "zero_shot":
        print(f"\nEvaluating on {args.split} set with zero-shot mode...")
        results = evaluator.evaluate_zero_shot(data_loader, use_combinations=True, debug=True)
        evaluator.print_metrics(results["metrics"])

        # 从配置获取seen/unseen组合
        czsl_config = config.get("czsl", {})
        seen_comb_names = czsl_config.get("seen_combinations", [])
        unseen_comb_names = czsl_config.get("unseen_combinations", [])

        # 将类别名称转换为索引
        seen_combinations = convert_combination_names_to_indices(seen_comb_names, class_names)
        unseen_combinations = convert_combination_names_to_indices(unseen_comb_names, class_names)

        # 保存结果
        np.savez(
            str(output_dir / f"czsl_results_{args.split}.npz"),
            labels=results["labels"],
            predictions=results["predictions"],
            features=results["features"]
        )

        # 绘制混淆矩阵
        evaluator.plot_confusion_by_combination(
            results["labels"],
            results["predictions"],
            save_path=str(output_dir / f"czsl_confusion_{args.split}.png"),
            seen_combinations=seen_combinations,
            unseen_combinations=unseen_combinations
        )

        # 可视化
        if args.visualize:
            print("\nGenerating visualizations...")

            # t-SNE可视化
            evaluator.plot_feature_tsne(
                results["features"],
                results["labels"],
                save_path=str(output_dir / f"czsl_tsne_{args.split}.png"),
                seen_combinations=seen_combinations,
                unseen_combinations=unseen_combinations
            )

            # UMAP可视化
            evaluator.plot_feature_umap(
                results["features"],
                results["labels"],
                save_path=str(output_dir / f"czsl_umap_{args.split}.png"),
                seen_combinations=seen_combinations,
                unseen_combinations=unseen_combinations
            )

            # 标签共现矩阵
            evaluator.plot_label_cooccurrence(
                results["labels"],
                results["predictions"],
                save_path=str(output_dir / f"czsl_cooccurrence_{args.split}.png")
            )

        # 保存STFT图像和预测结果
        if args.save_stft:
            print("\nSaving STFT images with predictions...")
            stft_dir = output_dir / "stft"
            evaluator.save_stft_predictions(
                data_loader=data_loader,
                output_dir=str(stft_dir),
                max_samples=args.max_stft_samples,
                use_combinations=True
            )

    elif args.mode == "by_combination":
        # 从配置获取seen/unseen组合
        czsl_config = config.get("czsl", {})
        seen_comb_names = czsl_config.get("seen_combinations", [])
        unseen_comb_names = czsl_config.get("unseen_combinations", [])

        # 将类别名称转换为索引
        seen_combinations = convert_combination_names_to_indices(seen_comb_names, class_names)
        unseen_combinations = convert_combination_names_to_indices(unseen_comb_names, class_names)

        print(f"\nEvaluating by combination type on {args.split} set...")
        print(f"Seen combinations: {len(seen_combinations)}")
        print(f"Unseen combinations: {len(unseen_combinations)}")

        results = evaluator.evaluate_by_combination_type(
            data_loader,
            seen_combinations=seen_combinations,
            unseen_combinations=unseen_combinations,
            debug=True
        )

        metrics = results["metrics"]

        print("\n" + "=" * 60)
        print("Evaluation by Combination Type")
        print("=" * 60)
        print(f"Seen Accuracy:    {metrics['seen_accuracy']:.4f} ({metrics['seen_samples']} samples)")
        print(f"Unseen Accuracy:  {metrics['unseen_accuracy']:.4f} ({metrics['unseen_samples']} samples)")
        print(f"Other Accuracy:   {metrics['other_accuracy']:.4f} ({metrics['other_samples']} samples)")

        # 保存结果
        np.savez(
            str(output_dir / f"czsl_results_{args.split}.npz"),
            labels=results["labels"],
            predictions=results["predictions"],
            features=results["features"]
        )

        # 绘制混淆矩阵
        evaluator.plot_confusion_by_combination(
            results["labels"],
            results["predictions"],
            save_path=str(output_dir / f"czsl_confusion_{args.split}.png"),
            seen_combinations=seen_combinations,
            unseen_combinations=unseen_combinations
        )

        # 保存文本结果
        with open(output_dir / "combination_results.txt", 'w') as f:
            f.write("Evaluation by Combination Type\n")
            f.write("=" * 40 + "\n")
            f.write(f"Seen Accuracy:    {metrics['seen_accuracy']:.4f} ({metrics['seen_samples']} samples)\n")
            f.write(f"Unseen Accuracy:  {metrics['unseen_accuracy']:.4f} ({metrics['unseen_samples']} samples)\n")
            f.write(f"Other Accuracy:   {metrics['other_accuracy']:.4f} ({metrics['other_samples']} samples)\n")

    elif args.mode == "by_jnr":
        # 从配置获取seen/unseen组合
        czsl_config = config.get("czsl", {})
        seen_comb_names = czsl_config.get("seen_combinations", [])
        unseen_comb_names = czsl_config.get("unseen_combinations", [])
        seen_combinations = convert_combination_names_to_indices(seen_comb_names, class_names)
        unseen_combinations = convert_combination_names_to_indices(unseen_comb_names, class_names)

        print(f"\nEvaluating by JNR level on {args.split} set...")

        # 加载归一化统计量
        stats_file = os.path.join(os.path.dirname(args.config), 'normalization_stats.json')
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                import json
                normalization_stats = json.load(f)
        else:
            normalization_stats = None

        # 为每个JNR创建独立的数据加载器
        jnr_loaders = create_jnr_dataloaders(
            config=config,
            split=args.split,
            normalization_stats=normalization_stats,
            batch_size=config.get('train', {}).get('batch_size', 32),
            num_workers=config.get('data', {}).get('num_workers', 4),
        )

        # 按JNR评估
        results = evaluator.evaluate_by_jnr(
            jnr_loaders,
            seen_combinations=seen_combinations,
            unseen_combinations=unseen_combinations,
        )

        # 打印和保存结果
        evaluator.print_jnr_results(
            results,
            save_path=str(output_dir / f"jnr_results_{args.split}.csv")
        )

        # 绘制JNR指标曲线
        evaluator.plot_jnr_metrics(
            results,
            save_path=str(output_dir / f"jnr_metrics_{args.split}.png")
        )


if __name__ == "__main__":
    main()