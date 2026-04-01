"""
评估脚本 - 支持组合零样本学习的评估
包括多标签分类指标和零样本推理评估
python -m czsl.evaluate --checkpoint checkpoints/best_model.pt --split val

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
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    multilabel_confusion_matrix, classification_report,
    average_precision_score, roc_auc_score
)
from sklearn.manifold import TSNE

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from czsl.model import create_clip_model, CLIPForMultiLabel
from czsl.data import load_multi_jnr_dataset


class Evaluator:
    """
    评估器类
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_names: list = None,
        threshold: float = 0.5
    ):
        """
        初始化评估器

        Args:
            model: 模型
            device: 计算设备
            class_names: 类别名称列表
            threshold: 预测阈值
        """
        self.model = model
        self.device = device
        self.class_names = class_names or [f"Class_{i}" for i in range(16)]
        self.threshold = threshold

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, mode: str = "zero_shot") -> dict:
        """
        评估模型

        Args:
            data_loader: 数据加载器
            mode: 评估模式 "supervised" 或 "zero_shot"

        Returns:
            评估指标字典
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        all_features = []

        eval_bar = tqdm(data_loader, desc="Evaluating")

        for images, labels in eval_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 调整图像尺寸
            if images.shape[-1] != 224:
                images = nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

            # 前向传播
            image_features = self.model.encode_image(images)
            logits = self.model.forward(images, mode=mode)

            # 获取预测
            probs = torch.sigmoid(logits)
            preds = (probs > self.threshold).float()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
            all_features.append(image_features.cpu())

        # 合并结果
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()
        all_features = torch.cat(all_features).numpy()

        # 计算指标
        metrics = self._compute_metrics(all_labels, all_preds, all_probs)

        return {
            "metrics": metrics,
            "predictions": all_preds,
            "labels": all_labels,
            "probabilities": all_probs,
            "features": all_features
        }

    def _compute_metrics(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        probs: np.ndarray
    ) -> dict:
        """
        计算评估指标

        Args:
            labels: 真实标签 [N, num_classes] one-hot编码
            preds: 预测标签 [N, num_classes] one-hot编码
            probs: 预测概率

        Returns:
            指标字典
        """
        metrics = {
            # 整体指标
            "accuracy": accuracy_score(labels.flatten(), preds.flatten()),
            "f1_macro": f1_score(labels, preds, average='macro', zero_division=0),
            "f1_micro": f1_score(labels, preds, average='micro', zero_division=0),
            "precision_macro": precision_score(labels, preds, average='macro', zero_division=0),
            "recall_macro": recall_score(labels, preds, average='macro', zero_division=0),

            # 子集准确率（完全匹配）
            "subset_accuracy": (labels == preds).all(axis=1).mean(),

            # 平均精度（需要每个类别有正样本）
            "map": self._safe_average_precision(labels, probs),
        }

        # 基于标签集合的评估指标 (TP_sub, FN_sub, FP_sub, TN_sub)
        sub_metrics = self._compute_sub_metrics(labels, preds)
        metrics.update(sub_metrics)

        # 每个类别的指标
        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
        per_class_precision = precision_score(labels, preds, average=None, zero_division=0)
        per_class_recall = recall_score(labels, preds, average=None, zero_division=0)

        metrics["per_class"] = {
            "f1": per_class_f1,
            "precision": per_class_precision,
            "recall": per_class_recall
        }

        return metrics

    def _compute_sub_metrics(self, labels: np.ndarray, preds: np.ndarray) -> dict:
        """
        计算基于标签集合的TP_sub, FN_sub, FP_sub, TN_sub

        计算方法:
        - K = 真实标签与预测标签中非0的数量总和(并集)
        - TP_sub: 真实和预测都有的标签数量 / K
        - FN_sub: 真实有但预测没有的标签数量 / K
        - FP_sub: 预测有但真实没有的标签数量 / K
        - TN_sub: 真实和预测都没有的标签数量 / K

        Args:
            labels: 真实标签 [N, num_classes]
            preds: 预测标签 [N, num_classes]

        Returns:
            包含TP_sub, FN_sub, FP_sub, TN_sub, precision_sub, recall_sub, f1_sub的字典
        """
        tp_total = 0.0
        fn_total = 0.0
        fp_total = 0.0
        tn_total = 0.0

        for i in range(len(labels)):
            # 获取真实标签和预测标签的索引集合
            true_set = set(np.where(labels[i] == 1)[0])
            pred_set = set(np.where(preds[i] == 1)[0])

            # 并集作为K
            union_set = true_set | pred_set
            K = len(union_set)

            if K == 0:
                # 真实和预测都为空，跳过
                continue

            # 计算TP, FN, FP, TN
            tp_i = len(true_set & pred_set)  # 真实和预测都有
            fn_i = len(true_set - pred_set)  # 真实有但预测没有
            fp_i = len(pred_set - true_set)  # 预测有但真实没有

            # TN: 在并集之外的类别，真实和预测都没有
            # 按文档定义，TN_sub = (总类别数 - 并集大小) / K，但这里我们按文档的逻辑
            # 文档中TN是基于并集计算的，在并集中真实和预测都没有的元素
            # 但实际上并集中的元素要么在true_set，要么在pred_set
            # 所以TN_sub通常为0
            tn_i = 0

            tp_total += tp_i / K
            fn_total += fn_i / K
            fp_total += fp_i / K
            tn_total += tn_i / K

        # 计算precision, recall, f1
        precision_sub = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        recall_sub = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1_sub = 2 * precision_sub * recall_sub / (precision_sub + recall_sub) if (precision_sub + recall_sub) > 0 else 0.0

        return {
            "tp_sub": tp_total,
            "fn_sub": fn_total,
            "fp_sub": fp_total,
            "tn_sub": tn_total,
            "precision_sub": precision_sub,
            "recall_sub": recall_sub,
            "f1_sub": f1_sub
        }

    def _safe_average_precision(self, labels: np.ndarray, probs: np.ndarray) -> float:
        """
        安全计算平均精度

        Args:
            labels: 真实标签
            probs: 预测概率

        Returns:
            平均精度
        """
        try:
            # 只计算有正样本的类别
            valid_classes = labels.sum(axis=0) > 0
            if valid_classes.sum() == 0:
                return 0.0
            return average_precision_score(labels[:, valid_classes], probs[:, valid_classes], average='macro')
        except ValueError:
            return 0.0

    def print_metrics(self, metrics: dict):
        """打印评估指标"""
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(f"Overall Accuracy:       {metrics['accuracy']:.4f}")
        print(f"Subset Accuracy:        {metrics['subset_accuracy']:.4f}")
        print(f"Macro F1 Score:         {metrics['f1_macro']:.4f}")
        print(f"Micro F1 Score:         {metrics['f1_micro']:.4f}")
        print(f"Macro Precision:        {metrics['precision_macro']:.4f}")
        print(f"Macro Recall:           {metrics['recall_macro']:.4f}")
        print(f"Mean Average Precision: {metrics['map']:.4f}")

        # 基于标签集合的评估指标
        print("\n" + "-" * 60)
        print("Label-Set Based Metrics (TP/FN/FP/TN):")
        print("-" * 60)
        print(f"TP_sub: {metrics['tp_sub']:.4f}")
        print(f"FN_sub: {metrics['fn_sub']:.4f}")
        print(f"FP_sub: {metrics['fp_sub']:.4f}")
        print(f"TN_sub: {metrics['tn_sub']:.4f}")
        print(f"Precision (sub): {metrics['precision_sub']:.4f}")
        print(f"Recall (sub):    {metrics['recall_sub']:.4f}")
        print(f"F1 (sub):        {metrics['f1_sub']:.4f}")

        # 每个类别的指标
        print("\nPer-class Metrics:")
        print("-" * 60)
        print(f"{'Class':<10} {'F1':>8} {'Precision':>12} {'Recall':>10}")
        print("-" * 60)
        for i, name in enumerate(self.class_names[:len(metrics['per_class']['f1'])]):
            print(f"{name:<10} {metrics['per_class']['f1'][i]:>8.4f} "
                  f"{metrics['per_class']['precision'][i]:>12.4f} "
                  f"{metrics['per_class']['recall'][i]:>10.4f}")

    def plot_confusion_matrices(self, labels: np.ndarray, preds: np.ndarray, save_path: str = None):
        """
        绘制每个类别的混淆矩阵

        Args:
            labels: 真实标签
            preds: 预测标签
            save_path: 保存路径
        """
        cm_per_class = multilabel_confusion_matrix(labels, preds)
        num_classes = len(self.class_names[:cm_per_class.shape[0]])

        fig, axes = plt.subplots(
            nrows=(num_classes - 1) // 3 + 1,
            ncols=3,
            figsize=(15, 4 * ((num_classes - 1) // 3 + 1))
        )
        axes = axes.flatten()

        for i, cm in enumerate(cm_per_class):
            ax = axes[i]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                        xticklabels=['Pred 0', 'Pred 1'],
                        yticklabels=['True 0', 'True 1'])
            ax.set_title(self.class_names[i])

        # 删除多余的子图
        for i in range(num_classes, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrices saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_feature_tsne(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        save_path: str = None
    ):
        """
        绘制特征的t-SNE可视化

        Args:
            features: 特征向量
            labels: 标签
            save_path: 保存路径
        """
        print("Computing t-SNE projection...")

        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
        features_2d = tsne.fit_transform(features)

        # 将多标签转换为单一标签（取最大的一个）
        single_labels = labels.argmax(axis=1)

        # 绘图
        fig, ax = plt.subplots(figsize=(12, 10))

        for i, name in enumerate(self.class_names[:labels.shape[1]]):
            mask = single_labels == i
            if mask.sum() > 0:
                ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                          label=name, alpha=0.6, s=20)

        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title('Feature Space Visualization (t-SNE)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

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
        sns.heatmap(true_cooc, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names[:true_cooc.shape[0]],
                    yticklabels=self.class_names[:true_cooc.shape[0]],
                    ax=axes[0])
        axes[0].set_title('True Label Co-occurrence')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].tick_params(axis='y', rotation=0)

        # 预测标签共现
        pred_cooc = (preds.T @ preds).astype(int)
        sns.heatmap(pred_cooc, annot=True, fmt='d', cmap='Greens',
                    xticklabels=self.class_names[:pred_cooc.shape[0]],
                    yticklabels=self.class_names[:pred_cooc.shape[0]],
                    ax=axes[1])
        axes[1].set_title('Predicted Label Co-occurrence')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].tick_params(axis='y', rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Co-occurrence plot saved to {save_path}")
        else:
            plt.show()

        plt.close()


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_datasets_by_jnr(base_path, jnr_start, jnr_end, jnr_step, split="val"):
    """
    按JNR级别分别加载数据集

    Returns:
        dict: {jnr_level: dataset}
    """
    jnr_numbers = range(jnr_start, jnr_end + 1, jnr_step)
    jnr_levels = [f"+{jnr}" if jnr >= 0 else str(jnr) for jnr in jnr_numbers]

    datasets = {}
    split_files = {
        "train": ("train_echo_stfts.mat", "train_echo_label.mat"),
        "test": ("test_echo_stfts.mat", "test_echo_label.mat"),
        "val": ("val_echo_stfts.mat", "val_echo_label.mat")
    }

    from czsl.data import STFTDataset3

    for jnr in jnr_levels:
        jnr_folder_name = f"JNR_{jnr}"
        data_folder_path = os.path.join(base_path, jnr_folder_name)

        stft_file = os.path.join(data_folder_path, split_files[split][0])
        label_file = os.path.join(data_folder_path, split_files[split][1])

        if os.path.exists(stft_file) and os.path.exists(label_file):
            dataset = STFTDataset3(stft_file, label_file)
            datasets[jnr] = dataset
            print(f"  JNR {jnr}: {len(dataset)} samples")

    return datasets


def evaluate_single_jnr(model, dataset, device, class_names, threshold=0.5, mode="supervised"):
    """
    评估单个JNR级别的数据集

    Returns:
        dict: 包含metrics, predictions, labels, features
    """
    evaluator = Evaluator(
        model=model,
        device=device,
        class_names=class_names,
        threshold=threshold
    )

    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0  # Windows兼容性
    )

    return evaluator.evaluate(data_loader, mode=mode)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Evaluate CLIP for CZSL")
    parser.add_argument("--config", type=str, default="czsl/config.yaml",
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--mode", type=str, default="supervised",
                        choices=["supervised", "zero_shot"],
                        help="Evaluation mode")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "test", "val"],
                        help="Data split to evaluate")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--by_jnr", action="store_true",
                        help="Evaluate and visualize by JNR level")
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
    data_config = config.get("data", {})

    # 创建模型
    model = create_clip_model(config, device=str(device))

    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {args.checkpoint}")
    if "metrics" in checkpoint:
        print(f"Checkpoint metrics: {checkpoint['metrics']}")

    if args.by_jnr:
        # ============ 按JNR级别分别评估 ============
        print(f"\n{'='*60}")
        print(f"Evaluating by JNR level on {args.split} set")
        print(f"{'='*60}")

        datasets_by_jnr = load_datasets_by_jnr(
            base_path=data_config.get("base_path"),
            jnr_start=data_config.get("jnr_start", -5),
            jnr_end=data_config.get("jnr_end", 40),
            jnr_step=data_config.get("jnr_step", 5),
            split=args.split
        )

        all_results = {}

        for jnr, dataset in datasets_by_jnr.items():
            print(f"\n--- JNR {jnr} ({len(dataset)} samples) ---")

            # 评估
            results = evaluate_single_jnr(
                model=model,
                dataset=dataset,
                device=device,
                class_names=class_names,
                threshold=config.get("evaluation", {}).get("threshold", 0.5),
                mode=args.mode
            )
            all_results[jnr] = results

            # 打印指标
            metrics = results["metrics"]
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Subset Acc:{metrics['subset_accuracy']:.4f}")
            print(f"  Macro F1:  {metrics['f1_macro']:.4f}")
            print(f"  F1 (sub):  {metrics['f1_sub']:.4f}")
            print(f"  mAP:       {metrics['map']:.4f}")

            # 可视化
            if args.visualize:
                jnr_dir = output_dir / f"JNR_{jnr}"
                jnr_dir.mkdir(parents=True, exist_ok=True)

                # 混淆矩阵
                cm_per_class = multilabel_confusion_matrix(
                    results["labels"], results["predictions"]
                )
                num_classes = len(class_names)
                ncols = min(4, num_classes)
                nrows = (num_classes - 1) // ncols + 1

                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
                axes = axes.flatten()

                for i, cm in enumerate(cm_per_class):
                    ax = axes[i]
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                                xticklabels=['Pred 0', 'Pred 1'],
                                yticklabels=['True 0', 'True 1'])
                    ax.set_title(class_names[i])

                for i in range(num_classes, len(axes)):
                    fig.delaxes(axes[i])

                plt.suptitle(f'Confusion Matrices - JNR {jnr}', fontsize=14)
                plt.tight_layout()
                plt.savefig(str(jnr_dir / "confusion_matrices.png"), dpi=150, bbox_inches='tight')
                plt.close()

                # t-SNE可视化
                if len(results["features"]) > 10:
                    print(f"  Computing t-SNE for JNR {jnr}...")
                    tsne = TSNE(n_components=2, random_state=42,
                                perplexity=min(30, len(results["features"]) - 1))
                    features_2d = tsne.fit_transform(results["features"])

                    single_labels = results["labels"].argmax(axis=1)

                    fig, ax = plt.subplots(figsize=(12, 10))
                    for i, name in enumerate(class_names[:results["labels"].shape[1]]):
                        mask = single_labels == i
                        if mask.sum() > 0:
                            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                                      label=name, alpha=0.6, s=20)

                    ax.set_xlabel('t-SNE 1')
                    ax.set_ylabel('t-SNE 2')
                    ax.set_title(f'Feature Space (t-SNE) - JNR {jnr}')
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                    plt.tight_layout()
                    plt.savefig(str(jnr_dir / "feature_tsne.png"), dpi=150, bbox_inches='tight')
                    plt.close()

                # 标签共现矩阵
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))

                # 真实标签共现
                true_cooc = (results["labels"].T @ results["labels"]).astype(int)
                sns.heatmap(true_cooc, annot=True, fmt='d', cmap='Blues',
                            xticklabels=class_names[:true_cooc.shape[0]],
                            yticklabels=class_names[:true_cooc.shape[0]],
                            ax=axes[0])
                axes[0].set_title(f'True Label Co-occurrence - JNR {jnr}')
                axes[0].tick_params(axis='x', rotation=45)
                axes[0].tick_params(axis='y', rotation=0)

                # 预测标签共现
                pred_cooc = (results["predictions"].T @ results["predictions"]).astype(int)
                sns.heatmap(pred_cooc, annot=True, fmt='d', cmap='Greens',
                            xticklabels=class_names[:pred_cooc.shape[0]],
                            yticklabels=class_names[:pred_cooc.shape[0]],
                            ax=axes[1])
                axes[1].set_title(f'Predicted Label Co-occurrence - JNR {jnr}')
                axes[1].tick_params(axis='x', rotation=45)
                axes[1].tick_params(axis='y', rotation=0)

                plt.tight_layout()
                plt.savefig(str(jnr_dir / "label_cooccurrence.png"), dpi=150, bbox_inches='tight')
                plt.close()

                # 保存结果
                np.savez(
                    str(jnr_dir / "results.npz"),
                    predictions=results["predictions"],
                    labels=results["labels"],
                    probabilities=results["probabilities"],
                    features=results["features"]
                )
                print(f"  Saved results to {jnr_dir}")

        # 汇总表格
        print(f"\n{'='*80}")
        print("Summary by JNR Level")
        print(f"{'='*80}")
        print(f"{'JNR':<8} {'Samples':<8} {'Accuracy':<10} {'SubsetAcc':<10} {'F1_sub':<10} {'mAP':<10}")
        print("-" * 80)
        for jnr in sorted(all_results.keys(), key=lambda x: int(x.replace('+', ''))):
            r = all_results[jnr]
            print(f"{jnr:<8} {r['labels'].shape[0]:<8} "
                  f"{r['metrics']['accuracy']:<10.4f} "
                  f"{r['metrics']['subset_accuracy']:<10.4f} "
                  f"{r['metrics']['f1_sub']:<10.4f} "
                  f"{r['metrics']['map']:<10.4f}")

        # 保存汇总结果
        summary_path = output_dir / "summary_by_jnr.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Evaluation Results by JNR Level\n")
            f.write(f"Mode: {args.mode}, Split: {args.split}\n")
            f.write(f"{'='*80}\n")
            f.write(f"{'JNR':<8} {'Samples':<8} {'Accuracy':<10} {'SubsetAcc':<10} {'F1_sub':<10} {'mAP':<10}\n")
            f.write("-" * 80 + "\n")
            for jnr in sorted(all_results.keys(), key=lambda x: int(x.replace('+', ''))):
                r = all_results[jnr]
                f.write(f"{jnr:<8} {r['labels'].shape[0]:<8} "
                       f"{r['metrics']['accuracy']:<10.4f} "
                       f"{r['metrics']['subset_accuracy']:<10.4f} "
                       f"{r['metrics']['f1_sub']:<10.4f} "
                       f"{r['metrics']['map']:<10.4f}\n")
        print(f"\nSummary saved to {summary_path}")

    else:
        # ============ 整体评估（原有逻辑）============
        train_dataset, test_dataset, val_dataset, num_classes, jnr_levels = load_multi_jnr_dataset(
            base_path=data_config.get("base_path"),
            jnr_start=data_config.get("jnr_start", 0),
            jnr_end=data_config.get("jnr_end", 40),
            jnr_step=data_config.get("jnr_step", 10)
        )

        # 选择数据集
        if args.split == "train":
            dataset = train_dataset
        elif args.split == "test":
            dataset = test_dataset
        else:
            dataset = val_dataset

        # 创建数据加载器
        data_loader = DataLoader(
            dataset,
            batch_size=config.get("train", {}).get("batch_size", 32),
            shuffle=False,
            num_workers=data_config.get("num_workers", 4)
        )

        # 创建评估器
        evaluator = Evaluator(
            model=model,
            device=device,
            class_names=class_names,
            threshold=config.get("evaluation", {}).get("threshold", 0.5)
        )

        # 评估
        print(f"\nEvaluating on {args.split} set with mode={args.mode}...")
        results = evaluator.evaluate(data_loader, mode=args.mode)

        # 打印指标
        evaluator.print_metrics(results["metrics"])

        # 可视化
        if args.visualize:
            print("\nGenerating visualizations...")

            # 混淆矩阵
            evaluator.plot_confusion_matrices(
                results["labels"],
                results["predictions"],
                save_path=str(output_dir / "confusion_matrices.png")
            )

            # 特征可视化
            evaluator.plot_feature_tsne(
                results["features"],
                results["labels"],
                save_path=str(output_dir / "feature_tsne.png")
            )

            # 标签共现
            evaluator.plot_label_cooccurrence(
                results["labels"],
                results["predictions"],
                save_path=str(output_dir / "label_cooccurrence.png")
            )

        # 保存结果
        results_path = output_dir / f"results_{args.split}_{args.mode}.npz"
        np.savez(
            results_path,
            predictions=results["predictions"],
            labels=results["labels"],
            probabilities=results["probabilities"],
            features=results["features"]
        )
        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()