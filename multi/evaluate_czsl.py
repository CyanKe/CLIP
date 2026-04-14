"""
CZSL评估脚本 - 支持零样本组合识别评估
python -m multi.evaluate_czsl --checkpoint checkpoints/czsl_best_model.pt --mode zero_shot
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
from multi.data import create_czsl_dataloaders
import clip


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
        use_combinations: bool = True
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

        for images, text_tokens, labels, texts, metas in eval_bar:
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
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

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
        unseen_combinations: list = None
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

        eval_bar = tqdm(data_loader, desc="Evaluating by Combination Type")

        for images, text_tokens, labels, texts, metas in eval_bar:
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
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            batch_size = images.shape[0]
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

        return metrics

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
            # 使用配置中定义的组合
            configured_combs = []
            if seen_combinations:
                configured_combs.extend([tuple(sorted(c)) for c in seen_combinations])
            if unseen_combinations:
                configured_combs.extend([tuple(sorted(c)) for c in unseen_combinations])
            unique_combs = sorted(set(configured_combs))
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

        for images, text_tokens, labels, texts, metas in eval_bar:
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
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

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
    parser.add_argument("--mode", type=str, default="zero_shot",
                        choices=["zero_shot", "by_combination"],
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
        results = evaluator.evaluate_zero_shot(data_loader, use_combinations=True)
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

        metrics = evaluator.evaluate_by_combination_type(
            data_loader,
            seen_combinations=seen_combinations,
            unseen_combinations=unseen_combinations
        )

        print("\n" + "=" * 60)
        print("Evaluation by Combination Type")
        print("=" * 60)
        print(f"Seen Accuracy:    {metrics['seen_accuracy']:.4f} ({metrics['seen_samples']} samples)")
        print(f"Unseen Accuracy:  {metrics['unseen_accuracy']:.4f} ({metrics['unseen_samples']} samples)")
        print(f"Other Accuracy:   {metrics['other_accuracy']:.4f} ({metrics['other_samples']} samples)")

        # 保存结果
        with open(output_dir / "combination_results.txt", 'w') as f:
            f.write("Evaluation by Combination Type\n")
            f.write("=" * 40 + "\n")
            f.write(f"Seen Accuracy:    {metrics['seen_accuracy']:.4f} ({metrics['seen_samples']} samples)\n")
            f.write(f"Unseen Accuracy:  {metrics['unseen_accuracy']:.4f} ({metrics['unseen_samples']} samples)\n")
            f.write(f"Other Accuracy:   {metrics['other_accuracy']:.4f} ({metrics['other_samples']} samples)\n")


if __name__ == "__main__":
    main()