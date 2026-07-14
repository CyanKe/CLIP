"""
CZSL评估脚本 - 支持零样本组合识别评估

参数优先级: CLI > multi/config.yaml 的 evaluation 段 > 默认值。

推荐（改 config 后免长 CLI）:
    python -m multi.evaluate_czsl
    python -m multi.evaluate_czsl --config multi/config.yaml

仍可用 CLI 覆盖:
    python -m multi.evaluate_czsl --checkpoint checkpoints/xxx_best.pt --mode by_jnr --split test
    python -m multi.evaluate_czsl --mode all --visualize --output_dir results
"""
# pylint: disable=no-member

import os
import sys
import json
import h5py
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.manifold import TSNE

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi.model import create_czsl_model, CLIPForCZSL
from multi.data import STFTDataset
from multi.metrics_czsl import (
    compute_metrics_bundle,
    compute_subset_bundles,
    print_metrics_report,
    print_jnr_metrics_table,
    save_metrics_json,
)
import clip


class ChainedLoaderIterable:
    """将多个 JNR 的 DataLoader 串成一个可迭代对象。

    与预先创建所有 DataLoader 不同，此实现按需创建：
    - 迭代到某个 JNR 时才创建其 STFTDataset（触发 HDD 顺序读取）
    - 切换到下一个 JNR 时释放上一个 JNR 的内存
    - 同一时刻只有一个 JNR 的 STFT 数据驻留内存

    用法:
        configs = [dict(stft_file=..., metadata_file=..., ...), ...]
        chain = ChainedLoaderIterable(configs, batch_size=32, collate_fn=...)
        for batch in chain:
            ...
    """
    def __init__(self, jnr_dataset_configs: list, batch_size: int = 32,
                 num_workers: int = 0, pin_memory: bool = False,
                 collate_fn=None, normalization_stats: dict = None):
        self._configs = jnr_dataset_configs  # list of dicts with STFTDataset kwargs
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._collate_fn = collate_fn
        self._norm_stats = normalization_stats
        self._total_batches = 0
        self._samples_per_jnr = []

        # 预计算总 batch 数（轻量，只读 HDF5 shape）
        for cfg in self._configs:
            try:
                with h5py.File(cfg['stft_file'], 'r') as f:
                    n = f[cfg.get('stft_var_name', 'all_stfts')].shape[2]
            except Exception:
                n = 0
            self._samples_per_jnr.append(n)
            self._total_batches += (n + batch_size - 1) // batch_size if n > 0 else 0

    def __iter__(self):
        for i, cfg in enumerate(self._configs):
            if self._samples_per_jnr[i] == 0:
                continue

            # ── 按需创建当前 JNR 的 STFTDataset ──
            # 这一步触发 HDF5 顺序读取，将整个 JNR 的 STFT 加载到内存
            dataset = STFTDataset(
                stft_file=cfg['stft_file'],
                metadata_file=cfg['metadata_file'],
                stft_var_name=cfg.get('stft_var_name', 'all_stfts'),
                normalization_stats=self._norm_stats,
                class_names=cfg.get('class_names', []),
                normalize_mode=cfg.get('normalize_mode', 'per_sample'),
                normalize_method=cfg.get('normalize_method', 'p99'),
                image_size=cfg.get('image_size', 224),
                apply_clip_norm=cfg.get('apply_clip_norm', True),
            )

            loader = DataLoader(
                dataset,
                batch_size=self._batch_size,
                shuffle=False,
                num_workers=self._num_workers,
                pin_memory=self._pin_memory,
                collate_fn=self._collate_fn,
            )

            label = cfg.get('label', f'JNR_{i}')
            print(f"  [{label}] {len(dataset)} samples loaded into memory")

            yield from loader

            # ── 释放当前 JNR 的数据，为下一个 JNR 腾内存 ──
            del loader
            del dataset

    def __len__(self):
        return self._total_batches


def create_jnr_dataloaders(
    config: dict,
    split: str = 'test',
    normalization_stats: dict = None,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    text_style: str = "class_only",
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
        text_style: 文本描述风格

    Returns:
        字典 {jnr_level: DataLoader}
    """
    from functools import partial
    data_config = config.get('data', {})
    base_path = data_config.get('base_path')
    jnr_start = data_config.get('jnr_start', 0)
    jnr_end = data_config.get('jnr_end', 20)
    jnr_step = data_config.get('jnr_step', 5)

    class_names = [cls['name'] for cls in config.get('jamming_classes', [])]
    jnr_levels = list(range(jnr_start, jnr_end + 1, jnr_step))

    # 使用指定 text_style 的 collate
    _jnr_collate = partial(_collate_fn_with_style, text_style=text_style)

    jnr_loaders = {}

    for jnr in jnr_levels:
        jnr_folder = f"JNR_{'+' if jnr >= 0 else ''}{jnr}"
        data_folder = os.path.join(base_path, jnr_folder)

        stft_suffix = data_config.get('stft_suffix', 'echo_stfts')
        stft_file = os.path.join(data_folder, f'{split}_{stft_suffix}.mat')
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
            collate_fn=_jnr_collate,
        )

        jnr_loaders[jnr] = loader
        print(f"Loaded {split} data from {jnr_folder}: {len(dataset)} samples")

    return jnr_loaders


def _collate_fn_with_style(batch, text_style="class_only"):
    """带 text_style 的 collate 函数（模块级，支持 pickle）"""
    from multi.text_templates import generate_text_descriptions
    import clip

    sample = batch[0]
    has_features = isinstance(sample[-1], dict) and len(sample) == 4

    if has_features:
        stft_images, labels, metadata_list, features_list = zip(*batch)
        features_batched = {}
        for domain in features_list[0].keys():
            features_batched[domain] = torch.stack([f[domain] for f in features_list], dim=0)
    else:
        stft_images, labels, metadata_list = zip(*batch)
        features_batched = None

    stft_images = torch.stack(stft_images, dim=0)
    labels = torch.stack(labels, dim=0)

    texts = []
    for meta in metadata_list:
        text = generate_text_descriptions(meta, style=text_style)
        texts.append(text)

    text_tokens = clip.tokenize(texts, truncate=True)

    return stft_images, None, text_tokens, labels, texts, metadata_list, features_batched


def plot_roc_curves(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: list,
    save_dir: str = None,
    prefix: str = "roc",
    mode_title: str = ""
):
    """
    绘制 ROC 曲线（独立函数，解耦于评估器）

    生成两张图:
      1. {prefix}_roc_per_class.png — 每个类别不同颜色的 ROC 曲线
      2. {prefix}_roc_average.png  — Micro/Macro 平均 ROC 曲线

    Args:
        labels: one-hot 标签 [n_samples, n_classes]
        probabilities: softmax 概率 [n_samples, n_classes]
        class_names: 类别名称列表
        save_dir: 保存目录（None 则显示）
        prefix: 文件名前缀
        mode_title: 标题后缀（如 "by_combination" 或 "JNR=+10"）
    """
    n_classes = labels.shape[1]

    # 计算每个类别的 ROC 和 AUC
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

    # === 图1: 每个类别的 ROC 曲线（不同颜色） ===
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(figsize=(8, 7))

    for idx, i in enumerate(fpr_dict):
        color = colors[idx % 10]
        ax.plot(
            fpr_dict[i], tpr_dict[i],
            color=color, linewidth=1.2,
            label=f'{class_names[i]} (AUC={auc_dict[i]:.3f})'
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

    # === 图2: Micro / Macro 平均 ROC 曲线 ===
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(
        fpr_micro, tpr_micro,
        color='darkorange', linewidth=2.5,
        label=f'Micro-average (AUC={auc_micro:.3f})'
    )
    ax.plot(
        all_fpr, mean_tpr,
        color='darkgreen', linewidth=2.5, linestyle='--',
        label=f'Macro-average (AUC={auc_macro:.3f})'
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
    mode_title: str = ""
):
    """
    绘制 Precision-Recall 曲线（独立函数，解耦于评估器）

    生成两张图:
      1. {prefix}_pr_per_class.png — 每个类别不同颜色的 PR 曲线
      2. {prefix}_pr_average.png  — Micro/Macro 平均 PR 曲线

    Args:
        labels: one-hot 标签 [n_samples, n_classes]
        probabilities: softmax 概率 [n_samples, n_classes]
        class_names: 类别名称列表
        save_dir: 保存目录（None 则显示）
        prefix: 文件名前缀
        mode_title: 标题后缀（如 "by_combination" 或 "JNR=+10"）
    """
    n_classes = labels.shape[1]

    # 计算每个类别的 PR 和 AP
    precision_dict, recall_dict, ap_dict = {}, {}, {}
    for i in range(n_classes):
        if np.sum(labels[:, i]) == 0:
            continue
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(
            labels[:, i], probabilities[:, i]
        )
        ap_dict[i] = average_precision_score(labels[:, i], probabilities[:, i])

    # Micro-average: ravel all labels & probs
    precision_micro, recall_micro, _ = precision_recall_curve(
        labels.ravel(), probabilities.ravel()
    )
    ap_micro = average_precision_score(labels.ravel(), probabilities.ravel())

    # Macro-average: interpolate precision on common recall grid
    all_recall = np.unique(np.concatenate([recall_dict[i] for i in recall_dict]))
    mean_precision = np.zeros_like(all_recall)
    for i in precision_dict:
        mean_precision += np.interp(all_recall, recall_dict[i][::-1], precision_dict[i][::-1])
    mean_precision /= len(precision_dict)
    ap_macro = np.trapezoid(mean_precision, all_recall)

    # No-skill baseline: overall positive ratio
    baseline = labels.sum() / labels.size

    title_suffix = f" ({mode_title})" if mode_title else ""

    # === 图1: 每个类别的 PR 曲线（不同颜色） ===
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(figsize=(8, 7))

    for idx, i in enumerate(precision_dict):
        color = colors[idx % 10]
        ax.plot(
            recall_dict[i], precision_dict[i],
            color=color, linewidth=1.2,
            label=f'{class_names[i]} (AP={ap_dict[i]:.3f})'
        )

    # No-skill baseline
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

    # === 图2: Micro / Macro 平均 PR 曲线 ===
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(
        recall_micro, precision_micro,
        color='darkorange', linewidth=2.5,
        label=f'Micro-average (AP={ap_micro:.3f})'
    )
    ax.plot(
        all_recall, mean_precision,
        color='darkgreen', linewidth=2.5, linestyle='--',
        label=f'Macro-average (AP={ap_macro:.3f})'
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
        seen_combinations: list = None,
        unseen_combinations: list = None,
        debug: bool = False
    ) -> dict:
        """
        零样本评估

        Args:
            data_loader: 数据加载器
            use_combinations: 是否使用组合特征
            seen_combinations: 已见组合索引列表（用于 seen/unseen 准确率统计）
            unseen_combinations: 未见组合索引列表

        Returns:
            评估结果字典
        """
        self.model.eval()

        all_labels = []
        all_preds = []
        all_features = []
        all_probs = []      # 单类 softmax 概率 (用于 ROC/PR)
        all_combination_correct = 0
        all_multilabel_correct = 0
        total_samples = 0

        # 构建 seen/unseen 集合
        seen_set = set(tuple(sorted(c)) for c in (seen_combinations or []))
        unseen_set = set(tuple(sorted(c)) for c in (unseen_combinations or []))
        has_seen_unseen = bool(seen_set or unseen_set)

        seen_results = {"correct": 0, "total": 0}
        unseen_results = {"correct": 0, "total": 0}
        other_results = {"correct": 0, "total": 0}

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
        for batch_idx, (images, _, text_tokens, labels, texts, metas, *_) in enumerate(eval_bar):
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

            # 单类 softmax 概率 (用于 ROC/PR 曲线)
            text_features_single = self.model.get_cached_text_features()
            logit_scale = self.model.model.logit_scale.exp()
            logits_single = logit_scale * (image_features @ text_features_single.T)
            probs_single = torch.softmax(logits_single, dim=-1)
            all_probs.append(probs_single.cpu())

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

                is_correct = (true_set == pred_set)

                if is_correct:
                    all_combination_correct += 1

                # 多标签子集准确率
                if true_set.issubset(pred_set) or pred_set.issubset(true_set):
                    all_multilabel_correct += 1

                # 按 seen/unseen/other 分类统计
                if has_seen_unseen:
                    true_comb = tuple(sorted(true_set))
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

            total_samples += batch_size

        # 合并结果
        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()
        all_features = np.concatenate(all_features, axis=0)
        all_probs_np = torch.cat(all_probs).numpy()

        # 统一 MetricsBundle（subset_accuracy 主字段；combination_accuracy 双写）
        seen_set = set(tuple(sorted(c)) for c in (seen_combinations or []))
        unseen_set = set(tuple(sorted(c)) for c in (unseen_combinations or []))
        metrics = compute_metrics_bundle(
            all_labels, all_preds, self.class_names,
            y_prob=all_probs_np,
            seen_set=seen_set, unseen_set=unseen_set,
            dual_write=True,
        )
        # 保留循环内统计作为校验用（与 bundle 应一致）
        _ = (all_combination_correct, all_multilabel_correct, has_seen_unseen, seen_results, unseen_results, other_results)

        return {
            "metrics": metrics,
            "labels": all_labels,
            "predictions": all_preds,
            "probabilities": all_probs_np,
            "features": all_features,
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
        all_probs = []

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
        for batch_idx, (images, _, text_tokens, labels, texts, metas, *_) in enumerate(eval_bar):
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
            all_probs.append(probs.cpu())
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

        # 合并所有结果
        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()
        all_probs = torch.cat(all_probs).numpy()
        all_features = np.concatenate(all_features, axis=0)

        # 全局 + seen/unseen 两子集 MetricsBundle
        seen_set = set(tuple(sorted(c)) for c in (seen_combinations or []))
        unseen_set = set(tuple(sorted(c)) for c in (unseen_combinations or []))
        bundles = compute_subset_bundles(
            all_labels, all_preds, self.class_names,
            seen_set=seen_set, unseen_set=unseen_set,
            y_prob=all_probs, dual_write=True,
        )
        # Flat metrics = global (backward compat) + nested subsets
        metrics = dict(bundles["global"])
        metrics["global"] = bundles["global"]
        metrics["seen"] = bundles["seen"]
        metrics["unseen"] = bundles["unseen"]
        _ = (seen_results, unseen_results, other_results)

        return {
            "metrics": metrics,
            "labels": all_labels,
            "predictions": all_preds,
            "probabilities": all_probs,
            "features": all_features,
        }

    def print_metrics(self, metrics: dict, title: str = "Evaluation Results"):
        """打印评估指标（统一 MetricsBundle 报表）"""
        # Nested by_combination report
        if isinstance(metrics.get("global"), dict) and "subset_accuracy" in metrics.get("global", {}):
            print_metrics_report(metrics["global"], title=f"{title} — global")
            if metrics.get("seen", {}).get("num_samples", 0):
                print_metrics_report(metrics["seen"], title=f"{title} — seen subset")
            if metrics.get("unseen", {}).get("num_samples", 0):
                print_metrics_report(metrics["unseen"], title=f"{title} — unseen subset")
            return
        print_metrics_report(metrics, title=title)

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
            all_probs = []
            seen_correct, seen_total = 0, 0
            unseen_correct, unseen_total = 0, 0

            # 每个类别的统计: TP, FP, FN, TN
            class_stats = np.zeros((num_classes, 4), dtype=np.int64)

            for images, _, _, labels, _, _, *_ in tqdm(
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
                all_probs.append(probs.cpu())

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
            all_labels_np = torch.cat(all_labels).numpy()
            all_preds_np = torch.cat(all_preds).numpy()
            all_probs_np = torch.cat(all_probs).numpy()

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

            bundle = compute_metrics_bundle(
                all_labels_np, all_preds_np, self.class_names,
                y_prob=all_probs_np,
                seen_set=seen_set, unseen_set=unseen_set,
                dual_write=True,
            )
            # Keep arrays for optional plotting; not written into metrics JSON core
            results[jnr] = {
                **bundle,
                "total_samples": bundle["num_samples"],  # dual-write legacy key
                "labels": all_labels_np,
                "probabilities": all_probs_np,
                "per_class_recall": per_class_recall,
                "per_class_precision": per_class_precision,
                "per_class_f1": per_class_f1,
            }

        return results

    def print_jnr_results(self, results: dict, save_path: str = None):
        """打印并保存JNR评估结果（subset_accuracy 为比率）"""
        print_jnr_metrics_table(results)

        lines = []
        for jnr, metrics in sorted(results.items()):
            acc = metrics.get("subset_accuracy", metrics.get("combination_accuracy", 0.0))
            n = metrics.get("num_samples", metrics.get("total_samples", 0))
            lines.append(
                f"{jnr},{acc:.4f},{metrics.get('f1_macro', 0):.4f},"
                f"{metrics.get('seen_accuracy', 0):.4f},"
                f"{metrics.get('unseen_accuracy', 0):.4f},"
                f"{metrics.get('harmonic_mean', 0):.4f},"
                f"{n}"
            )

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("JNR,SubsetAcc,F1_Macro,Seen_Acc,Unseen_Acc,HM,Samples\n")
                f.write("\n".join(lines))
            print(f"\nResults saved to {save_path}")

        # 打印每个类别的指标 (Recall, Precision, F1) — 使用 bundle.per_class 或 legacy 数组
        print("\n" + "=" * 80)
        print("Per-Class Recall by JNR Level")
        print("=" * 80)
        header = f"{'JNR':>6} | " + " | ".join(f"{name:>8}" for name in self.class_names)
        print(header)
        print("-" * len(header))

        per_class_recall_lines = []
        for jnr, metrics in sorted(results.items()):
            recalls = metrics.get("per_class_recall")
            if recalls is None and isinstance(metrics.get("per_class"), dict):
                recalls = [metrics["per_class"].get(n, {}).get("recall", 0.0) for n in self.class_names]
            recalls = list(recalls) if recalls is not None else [0.0] * len(self.class_names)
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
            acc = m.get("subset_accuracy", m.get("combination_accuracy", 0.0))
            n = m.get("num_samples", m.get("total_samples", 0))
            # legacy: combination_accuracy stored as count
            if isinstance(acc, (int, float)) and acc > 1.0 and n:
                acc = acc / n
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
            use_combinations: 是否使用组合特征预测（与 evaluate_zero_shot 一致）
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

        for images, _, text_tokens, labels, texts, metas, *_ in eval_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 调整图像尺寸
            if images.shape[-1] != 224:
                images = nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

            batch_size = images.shape[0]

            # 零样本预测 — 与 evaluate_zero_shot 使用相同的预测逻辑
            if use_combinations:
                # 使用组合特征进行精确匹配
                _, _, pred_names = self.model.zero_shot_predict(
                    images, use_combinations=True, top_k=1
                )
                # 解析预测的组合名称
                preds = torch.zeros(batch_size, len(self.class_names), device=self.device)
                for i, name in enumerate(pred_names):
                    if name and len(name) > 0:
                        comb_name = name[0]
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
    """主函数

    参数优先级: CLI > config.yaml 的 evaluation 段 > 内置默认值。
    日常可只改 multi/config.yaml 的 evaluation，然后:
        python -m multi.evaluate_czsl
    """
    parser = argparse.ArgumentParser(description="Evaluate CZSL Model")
    parser.add_argument("--config", type=str, default="multi/config.yaml",
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (default: evaluation.checkpoint in config)")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["all", "by_jnr"],
                        help="Evaluation mode (default: evaluation.mode in config)")
    parser.add_argument("--split", type=str, default=None,
                        choices=["train", "val", "test"],
                        help="Data split (default: evaluation.split in config)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: evaluation.output_dir in config)")
    # store_true + default=None → 未传 CLI 时可读 config
    parser.add_argument("--visualize", action="store_true", default=None,
                        help="Generate all visualizations (overrides evaluation.visualize)")
    parser.add_argument("--tsne", action="store_true", default=None,
                        help="Generate t-SNE only")
    parser.add_argument("--umap", action="store_true", default=None,
                        help="Generate UMAP only")
    parser.add_argument("--roc", action="store_true", default=None,
                        help="Generate ROC curves only")
    parser.add_argument("--pr", action="store_true", default=None,
                        help="Generate PR curves only")
    parser.add_argument("--save_stft", action="store_true", default=None,
                        help="Save STFT images with predictions")
    parser.add_argument("--max_stft_samples", type=int, default=None,
                        help="Maximum number of STFT images to save (default: all)")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    eval_cfg = config.get("evaluation", {}) or {}

    # ── CLI > config > default ──
    def _flag(cli_val, key, default=False):
        if cli_val is not None:
            return bool(cli_val)
        return bool(eval_cfg.get(key, default))

    checkpoint_path = args.checkpoint or eval_cfg.get("checkpoint")
    if not checkpoint_path:
        raise ValueError(
            "No checkpoint specified. Set --checkpoint or evaluation.checkpoint in config.yaml"
        )
    mode = args.mode or eval_cfg.get("mode", "all")
    split = args.split or eval_cfg.get("split", "test")
    output_dir_str = args.output_dir or eval_cfg.get("output_dir", "results")
    do_viz = _flag(args.visualize, "visualize", False)
    do_tsne = _flag(args.tsne, "tsne", False)
    do_umap = _flag(args.umap, "umap", False)
    do_roc = _flag(args.roc, "roc", False)
    do_pr = _flag(args.pr, "pr", False)
    do_save_stft = _flag(args.save_stft, "save_stft", False)
    max_stft_samples = (
        args.max_stft_samples
        if args.max_stft_samples is not None
        else eval_cfg.get("max_stft_samples")
    )

    print(f"Config: {args.config}")
    print(f"Evaluation — mode={mode}, split={split}, checkpoint={checkpoint_path}")
    print(f"  output_dir={output_dir_str}  viz={do_viz} tsne={do_tsne} umap={do_umap} "
          f"roc={do_roc} pr={do_pr} save_stft={do_save_stft}")

    # 回写到 args，后续逻辑统一读 args.*（已合并 config）
    args.checkpoint = checkpoint_path
    args.mode = mode
    args.split = split
    args.output_dir = output_dir_str
    args.visualize = do_viz
    args.tsne = do_tsne
    args.umap = do_umap
    args.roc = do_roc
    args.pr = do_pr
    args.save_stft = do_save_stft
    args.max_stft_samples = max_stft_samples

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 类别名称和文本风格
    class_names = [cls["name"] for cls in config.get("jamming_classes", [])]
    text_style = config.get("czsl", {}).get("text_style", "class_only")

    # 创建模型
    model = create_czsl_model(config, device=str(device))

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

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

    # 从配置加载所有需要缓存的组合（seen + unseen），避免生成全部 C(17,2)=136 种
    czsl_config = config.get("czsl", {})
    seen_comb_names = czsl_config.get("seen_combinations", [])
    unseen_comb_names = czsl_config.get("unseen_combinations", [])
    all_comb_names = seen_comb_names + unseen_comb_names
    # 去重（保持顺序）
    seen_set = set()
    unique_combos = []
    for c in all_comb_names:
        key = tuple(sorted(c))
        if key not in seen_set:
            seen_set.add(key)
            unique_combos.append(c)

    # 缓存文本特征（仅缓存 config 中定义的 seen+unseen 组合）
    model.cache_text_features(max_combination_size=2, include_single=True,
                              seen_combinations=unique_combos, text_style=text_style)

    # 选择数据集
    if args.split == "train":
        split_name = "train"
    elif args.split == "val":
        split_name = "val"
    else:
        split_name = "test"

    # 创建评估器
    evaluator = CZSLEvaluator(
        model=model,
        device=device,
        class_names=class_names
    )

    # 评估
    if args.mode == "all":
        # ── 逐 JNR 按需加载：同一时刻只有一个 JNR 的 STFT 在内存中 ──
        # ChainedLoaderIterable 迭代到每个 JNR 时才创建 STFTDataset
        # （触发 HDD 顺序读取），切换 JNR 时释放上一个。

        # 加载归一化统计量
        stats_file = os.path.join(os.path.dirname(args.config), 'normalization_stats.json')
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                normalization_stats = json.load(f)
        else:
            normalization_stats = None

        # 构建每个 JNR 的数据集配置（不实际加载数据）
        data_config = config.get('data', {})
        base_path = data_config.get('base_path')
        jnr_start = data_config.get('jnr_start', 10)
        jnr_end = data_config.get('jnr_end', 10)
        jnr_step = data_config.get('jnr_step', 1)
        stft_suffix = data_config.get('stft_suffix', 'echo_stfts')
        class_names = [cls['name'] for cls in config.get('jamming_classes', [])]

        jnr_configs = []
        for jnr in range(jnr_start, jnr_end + 1, jnr_step):
            jnr_folder = f"JNR_{'+' if jnr >= 0 else ''}{jnr}"
            data_folder = os.path.join(base_path, jnr_folder)
            stft_file = os.path.join(data_folder, f'{split_name}_{stft_suffix}.mat')
            metadata_file = os.path.join(data_folder, f'{split_name}_echo_metadata.json')
            if os.path.exists(stft_file) and os.path.exists(metadata_file):
                jnr_configs.append({
                    'stft_file': stft_file,
                    'metadata_file': metadata_file,
                    'class_names': class_names,
                    'label': jnr_folder,
                })

        if not jnr_configs:
            print(f"Error: No data found for {split_name} split!")
            return

        print(f"\nCreating on-demand per-JNR loader chain for {split_name} split "
              f"({len(jnr_configs)} JNR levels)")

        # 链式迭代器：按需创建 → 迭代 → 释放 → 下一个
        data_loader = ChainedLoaderIterable(
            jnr_configs,
            batch_size=config.get('train', {}).get('batch_size', 32),
            num_workers=config.get('data', {}).get('num_workers', 0),
            pin_memory=config.get('data', {}).get('pin_memory', False) if sys.platform == 'win32' else False,
            collate_fn=partial(_collate_fn_with_style, text_style=text_style),
            normalization_stats=normalization_stats,
        )

        print(f"Total batches: {len(data_loader)}")

        # 公共: 加载 seen/unseen 组合
        czsl_config = config.get("czsl", {})
        seen_comb_names = czsl_config.get("seen_combinations", [])
        unseen_comb_names = czsl_config.get("unseen_combinations", [])
        seen_combinations = convert_combination_names_to_indices(seen_comb_names, class_names)
        unseen_combinations = convert_combination_names_to_indices(unseen_comb_names, class_names)

        # ── 1) Zero-Shot (组合特征匹配) ──
        print(f"\n{'='*60}")
        print(f"  [1/2] Zero-Shot Evaluation (combination matching)")
        print(f"{'='*60}")
        results_zs = evaluator.evaluate_zero_shot(
            data_loader, use_combinations=True, debug=True,
            seen_combinations=seen_combinations,
            unseen_combinations=unseen_combinations,
        )
        evaluator.print_metrics(results_zs["metrics"], title="Zero-Shot Evaluation Results")
        save_metrics_json(
            results_zs["metrics"], output_dir,
            filename=f"metrics_zero_shot_{args.split}.json",
            mode="zero_shot", split=args.split,
            extra_meta={"checkpoint": args.checkpoint},
        )

        np.savez(str(output_dir / f"czsl_zeroshot_{args.split}.npz"),
                 labels=results_zs["labels"], predictions=results_zs["predictions"],
                 features=results_zs["features"])

        if args.visualize:
            evaluator.plot_confusion_by_combination(
                results_zs["labels"], results_zs["predictions"],
                save_path=str(output_dir / f"czsl_confusion_zs_{args.split}.png"),
                seen_combinations=seen_combinations, unseen_combinations=unseen_combinations,
            )
        if args.visualize or args.roc:
            plot_roc_curves(
                results_zs["labels"], results_zs["probabilities"], class_names,
                save_dir=str(output_dir), prefix=f"czsl_{args.split}_zs",
                mode_title=f"zero_shot ({args.split})",
            )
        if args.visualize or args.pr:
            plot_pr_curves(
                results_zs["labels"], results_zs["probabilities"], class_names,
                save_dir=str(output_dir), prefix=f"czsl_{args.split}_zs",
                mode_title=f"zero_shot ({args.split})",
            )
        if args.tsne:
            print("\nGenerating t-SNE (zero-shot)...")
            evaluator.plot_feature_tsne(
                results_zs["features"], results_zs["labels"],
                save_path=str(output_dir / f"czsl_tsne_{args.split}.png"),
                seen_combinations=seen_combinations, unseen_combinations=unseen_combinations,
            )
        if args.umap:
            print("\nGenerating UMAP (zero-shot)...")
            evaluator.plot_feature_umap(
                results_zs["features"], results_zs["labels"],
                save_path=str(output_dir / f"czsl_umap_{args.split}.png"),
                seen_combinations=seen_combinations, unseen_combinations=unseen_combinations,
            )
        if args.visualize:
            evaluator.plot_label_cooccurrence(
                results_zs["labels"], results_zs["predictions"],
                save_path=str(output_dir / f"czsl_cooccurrence_{args.split}.png"),
            )
        if args.save_stft:
            print("\nSaving STFT images with predictions...")
            evaluator.save_stft_predictions(
                data_loader=data_loader, output_dir=str(output_dir / "stft"),
                max_samples=args.max_stft_samples, use_combinations=True,
            )

        # ── 2) By-Combination (单类特征 + softmax) ──
        print(f"\n{'='*60}")
        print(f"  [2/2] By-Combination Evaluation (single-class + softmax)")
        print(f"{'='*60}")
        results_bc = evaluator.evaluate_by_combination_type(
            data_loader,
            seen_combinations=seen_combinations,
            unseen_combinations=unseen_combinations,
            debug=True,
        )
        metrics_bc = results_bc["metrics"]
        evaluator.print_metrics(metrics_bc, title="By-Combination Evaluation Results")
        save_metrics_json(
            metrics_bc, output_dir,
            filename=f"metrics_by_combination_{args.split}.json",
            mode="by_combination", split=args.split,
            extra_meta={"checkpoint": args.checkpoint},
        )

        np.savez(str(output_dir / f"czsl_bycombo_{args.split}.npz"),
                 labels=results_bc["labels"], predictions=results_bc["predictions"],
                 features=results_bc["features"])

        if args.visualize:
            evaluator.plot_confusion_by_combination(
                results_bc["labels"], results_bc["predictions"],
                save_path=str(output_dir / f"czsl_confusion_bc_{args.split}.png"),
                seen_combinations=seen_combinations, unseen_combinations=unseen_combinations,
            )
        if args.visualize or args.roc:
            plot_roc_curves(
                results_bc["labels"], results_bc["probabilities"], class_names,
                save_dir=str(output_dir), prefix=f"czsl_{args.split}_byc",
                mode_title=f"by_combination ({args.split})",
            )
        if args.visualize or args.pr:
            plot_pr_curves(
                results_bc["labels"], results_bc["probabilities"], class_names,
                save_dir=str(output_dir), prefix=f"czsl_{args.split}_byc",
                mode_title=f"by_combination ({args.split})",
            )
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
            text_style=text_style,
        )

        # 按JNR评估
        results = evaluator.evaluate_by_jnr(
            jnr_loaders,
            seen_combinations=seen_combinations,
            unseen_combinations=unseen_combinations,
        )

        # 打印和保存结果（仅 per-JNR，不拼 global / 不与 zero_shot 合并）
        evaluator.print_jnr_results(
            results,
            save_path=str(output_dir / f"jnr_results_{args.split}.csv")
        )
        # JSON: strip heavy arrays
        jnr_json = {}
        for jnr, m in results.items():
            jnr_json[str(jnr)] = {
                k: v for k, v in m.items()
                if k not in ("labels", "probabilities", "features",
                             "per_class_recall", "per_class_precision", "per_class_f1")
            }
        save_metrics_json(
            {"per_jnr": jnr_json}, output_dir,
            filename=f"metrics_by_jnr_{args.split}.json",
            mode="by_jnr", split=args.split,
            extra_meta={"checkpoint": args.checkpoint},
        )

        # 绘制JNR指标曲线
        evaluator.plot_jnr_metrics(
            results,
            save_path=str(output_dir / f"jnr_metrics_{args.split}.png")
        )

        # 每个JNR等级的ROC曲线和PR曲线
        do_jnr_curves = args.visualize or args.roc or args.pr
        if do_jnr_curves:
            for jnr_val, jnr_results in sorted(results.items()):
                if "probabilities" in jnr_results and jnr_results["probabilities"].size > 0:
                    prefix = f"jnr_{jnr_val:+.0f}_{args.split}"
                    if args.visualize or args.roc:
                        plot_roc_curves(
                            jnr_results["labels"],
                            jnr_results["probabilities"],
                            class_names,
                            save_dir=str(output_dir),
                            prefix=prefix,
                            mode_title=f"JNR={jnr_val:+d}"
                        )
                    if args.visualize or args.pr:
                        plot_pr_curves(
                            jnr_results["labels"],
                            jnr_results["probabilities"],
                            class_names,
                            save_dir=str(output_dir),
                            prefix=prefix,
                            mode_title=f"JNR={jnr_val:+d}"
                        )


if __name__ == "__main__":
    main()