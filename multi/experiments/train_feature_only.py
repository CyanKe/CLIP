"""
仅使用特征条件上下文 (Feature-Conditioned Context) 进行训练和评估
不使用 STFT 图像数据，仅依赖 22 维信号特征

用法:
  python -m multi.experiments.train_feature_only --config multi/config.yaml
  python -m multi.experiments.train_feature_only --config multi/config.yaml --eval_only --checkpoint path/to/model.pt --split test
  python -m multi.experiments.train_feature_only --config multi/config.yaml --eval_only --checkpoint path/to/model.pt --split val

架构:
  22-dim features → FeatureConditionedPromptLearner → context vectors [B, M, D]
  → mean pool → [B, D] → MLP classifier → [B, num_classes]
  → BCE loss (multi-label)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from multi.prompt_learner import FeatureConditionedPromptLearner


# ============================================================================
# Models
# ============================================================================

class PlainMLPModel(nn.Module):
    """Plain MLP baseline: 22-dim → 2-layer MLP → logits

    不做 domain 分离，不做 context token 膨胀，直接全连接。
    使用 LayerNorm（非 BatchNorm）避免多标签小批次统计不稳定。
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(22, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.drop2 = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, features_dict: dict, return_features: bool = False):
        # Concat 5 domains → [B, 22]
        flat = torch.cat([
            features_dict[d]
            for d in FeatureConditionedPromptLearner.DOMAIN_ORDER
        ], dim=1)

        h = self.drop1(F.relu(self.ln1(self.fc1(flat))))
        h = self.drop2(F.relu(self.ln2(self.fc2(h))))
        logits = self.head(h)

        if return_features:
            return logits, h
        return logits


class MetaNetModel(nn.Module):
    """Meta-Net 模型（保留作对比）

    22维信号特征 → FeatureConditionedPromptLearner (Meta-Net)
    → context vectors [B, M, D] → mean pool → [B, D]
    → MLP classifier → [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int,
        context_dim: int = 128,
        n_ctx_per_domain: dict = None,
        classifier_hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.context_dim = context_dim

        self.prompt_learner = FeatureConditionedPromptLearner(
            transformer_width=context_dim,
            n_ctx_per_domain=n_ctx_per_domain,
        )

        self.classifier = nn.Sequential(
            nn.Linear(context_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, features_dict: dict, return_features: bool = False):
        context = self.prompt_learner(features_dict)  # [B, M, D]
        pooled = context.mean(dim=1)                   # [B, D]
        logits = self.classifier(pooled)               # [B, num_classes]
        if return_features:
            return logits, pooled
        return logits


# ============================================================================
# FeatureOnlyDataset
# ============================================================================

FEATURE_DOMAINS = {
    
    'time': {
        'prefix': 'time_domain',
        'keys': ['skewness', 'kurtosis', 'envelope_variation', 'modulation_bandwidth', 'modulation_rate'],
    },
    'freq': {
        'prefix': 'freq_domain',
        'keys': ['spectral_skewness', 'spectral_kurtosis', 'carrier_factor', 'awgn_factor'],
    },
    'bispectrum': {
        'prefix': 'bispectrum',
        'keys': ['bispectrum_variance', 'bispectrum_mean'],
    },
    'wavelet': {
        'prefix': 'wavelet',
        'keys': ['variance', 'mean', 'max', 'scale_centroid', 'max_singular_value',
                 'central_moment_2', 'central_moment_3', 'central_moment_4'],
    },
    'statistical': {
        'prefix': 'statistical',
        'keys': ['shannon_entropy', 'exponential_entropy', 'norm_entropy'],
    },
}


class FeatureOnlyDataset(Dataset):
    """只加载特征和标签，不加载 STFT 数据"""

    def __init__(
        self,
        data_dirs: List[str],
        class_names: List[str],
        feature_norm_stats: dict,
        split_name: str = "train",
    ):
        self.class_names = class_names
        self.split_name = split_name

        # 加载所有目录的特征和元数据
        self._samples = []       # list of features_dict (raw, un-normalized)
        self._labels_list = []   # list of np.ndarray (multi-hot)

        for data_dir in data_dirs:
            features_file = os.path.join(data_dir, f'{split_name}_echo_features.json')
            metadata_file = os.path.join(data_dir, f'{split_name}_echo_metadata.json')

            if not os.path.exists(features_file):
                print(f"  Skip {data_dir}: no features file")
                continue
            if not os.path.exists(metadata_file):
                print(f"  Skip {data_dir}: no metadata file")
                continue

            with open(features_file, 'r', encoding='utf-8') as f:
                features_list = json.load(f)
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)

            # 确保 metadata 是 list
            if isinstance(metadata_list, dict):
                metadata_list = list(metadata_list.values()) if 'samples' not in metadata_list else metadata_list.get('samples', [])

            for i, raw_features in enumerate(features_list):
                feat_dict = self._parse_features(raw_features, feature_norm_stats)
                if feat_dict is None:
                    continue

                # Build label
                label = np.zeros(len(class_names), dtype=np.float32)
                if i < len(metadata_list):
                    meta = metadata_list[i]
                    jam_types = meta.get('jam_types', [])
                    if isinstance(jam_types, str):
                        jam_types = [jam_types] if jam_types and jam_types != 'None' else []
                    for jt in jam_types:
                        jt = jt.strip() if isinstance(jt, str) else str(jt)
                        if jt in class_names:
                            label[class_names.index(jt)] = 1.0

                self._samples.append(feat_dict)
                self._labels_list.append(label)

        print(f"  [{split_name}] Loaded {len(self._samples)} samples from {len(data_dirs)} directories")

    def _parse_features(self, raw: dict, norm_stats: dict) -> Optional[dict]:
        feat_dict = {}
        for domain, info in FEATURE_DOMAINS.items():
            values = []
            for key in info['keys']:
                flat_key = f"{info['prefix']}.{key}"
                val = raw
                for part in [info['prefix'], key]:
                    val = val.get(part, 0.0) if isinstance(val, dict) else 0.0
                if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                    val = 0.0
                if flat_key in norm_stats:
                    s = norm_stats[flat_key]
                    val = (val - s.get('mean', 0.0)) / (s.get('std', 1.0) + 1e-8)
                values.append(val)
            feat_dict[domain] = torch.tensor(values, dtype=torch.float32)
        return feat_dict

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        return self._samples[index], torch.from_numpy(self._labels_list[index])


# ============================================================================
# collate function
# ============================================================================

def collate_features(batch):
    """将 batch 中的 features_dict 合并为 batched tensors"""
    features_list, labels_list = zip(*batch)
    labels = torch.stack(labels_list)

    # Merge features_dicts: {domain: [tensor_per_sample]} → {domain: tensor[B, dim]}
    batched_features = {}
    domain_names = features_list[0].keys()
    for domain in domain_names:
        batched_features[domain] = torch.stack([f[domain] for f in features_list])

    return batched_features, labels


# ============================================================================
# Training utilities
# ============================================================================

def build_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    data_config = config.get("data", {})
    base_path = data_config.get("base_path", "D:/output/output/20us_single")
    jnr_start = data_config.get("jnr_start", 0)
    jnr_end = data_config.get("jnr_end", 20)
    jnr_step = data_config.get("jnr_step", 5)
    batch_size = config.get("train", {}).get("batch_size", 32)
    num_workers = data_config.get("num_workers", 4)

    class_names = [cls["name"] for cls in config.get("jamming_classes", [])]

    # Load normalization stats
    norm_stats_path = config.get("feature_norm_stats_path", "multi/feature_normalization_stats.json")
    if os.path.exists(norm_stats_path):
        with open(norm_stats_path, 'r', encoding='utf-8') as f:
            norm_stats = json.load(f)
    else:
        norm_stats = {}

    # Gather directories per JNR level and split
    jnr_levels = list(range(jnr_start, jnr_end + 1, jnr_step))

    def make_dirs(split: str) -> List[str]:
        dirs = []
        for jnr in jnr_levels:
            jnr_folder = f"JNR_{'+' if jnr >= 0 else ''}{jnr}"
            d = os.path.join(base_path, jnr_folder)
            if os.path.isdir(d):
                dirs.append(d)
        return dirs

    train_dirs = make_dirs("train")
    val_dirs = make_dirs("val")
    test_dirs = make_dirs("test")

    train_dataset = FeatureOnlyDataset(train_dirs, class_names, norm_stats, "train")
    val_dataset = FeatureOnlyDataset(val_dirs, class_names, norm_stats, "val") if val_dirs else None
    test_dataset = FeatureOnlyDataset(test_dirs, class_names, norm_stats, "test") if test_dirs else None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_features,
        pin_memory=data_config.get("pin_memory", True),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_features,
        pin_memory=data_config.get("pin_memory", True),
    ) if val_dataset else None
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_features,
        pin_memory=data_config.get("pin_memory", True),
    ) if test_dataset else None

    return train_loader, val_loader, test_loader


# ============================================================================
# Train / Eval
# ============================================================================

def train_epoch(model, dataloader, loss_fn, optimizer, device, threshold: float = 0.5):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for features_dict, labels in tqdm(dataloader, desc="Train"):
        features_dict = {k: v.to(device) for k, v in features_dict.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features_dict)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = (torch.sigmoid(logits) > threshold).float()
        total_correct += (preds == labels).all(dim=1).sum().item()
        total_samples += labels.size(0)

    return {
        "loss": total_loss / total_samples,
        "exact_match": total_correct / total_samples,
    }


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device, return_details: bool = False, threshold: float = 0.5):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_labels = [], []
    all_probs = [] if return_details else None
    all_features = [] if return_details else None

    for features_dict, labels in tqdm(dataloader, desc="Eval"):
        features_dict = {k: v.to(device) for k, v in features_dict.items()}
        labels = labels.to(device)

        if return_details:
            logits, feats = model(features_dict, return_features=True)
            all_features.append(feats.cpu().numpy())
        else:
            logits = model(features_dict)
        loss = loss_fn(logits, labels)

        total_loss += loss.item() * labels.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        total_correct += (preds == labels).all(dim=1).sum().item()
        total_samples += labels.size(0)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        if return_details:
            all_probs.append(probs.cpu().numpy())

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Per-class metrics
    per_class = {}
    for c in range(all_labels.shape[1]):
        tp = ((all_preds[:, c] == 1) & (all_labels[:, c] == 1)).sum()
        fp = ((all_preds[:, c] == 1) & (all_labels[:, c] == 0)).sum()
        fn = ((all_preds[:, c] == 0) & (all_labels[:, c] == 1)).sum()
        tn = ((all_preds[:, c] == 0) & (all_labels[:, c] == 0)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        per_class[c] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                        "precision": precision, "recall": recall,
                        "f1": 2 * precision * recall / (precision + recall + 1e-8)}

    # Micro/macro F1
    tp_all = ((all_preds == 1) & (all_labels == 1)).sum()
    fp_all = ((all_preds == 1) & (all_labels == 0)).sum()
    fn_all = ((all_preds == 0) & (all_labels == 1)).sum()
    micro_p = tp_all / (tp_all + fp_all + 1e-8)
    micro_r = tp_all / (tp_all + fn_all + 1e-8)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-8)

    macro_f1 = np.mean([per_class[c]["f1"] for c in range(all_labels.shape[1])])

    result = {
        "loss": total_loss / total_samples,
        "exact_match": total_correct / total_samples,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }
    if return_details:
        result["all_preds"] = all_preds
        result["all_labels"] = all_labels
        result["all_features"] = np.concatenate(all_features)
        result["all_probs"] = np.concatenate(all_probs)
        result["per_class"] = per_class
        # Probability diagnostics
        all_probs_arr = result["all_probs"]
        result["prob_mean"] = float(all_probs_arr.mean())
        result["prob_std"] = float(all_probs_arr.std())
        result["prob_max"] = float(all_probs_arr.max())
        result["pred_positive_rate"] = float(all_preds.mean())
    return result


# ============================================================================
# Confusion Matrix
# ============================================================================

def plot_confusion_matrices(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    class_names: List[str],
    save_path: str,
    title_prefix: str = "",
):
    """为每个类别绘制 2×2 混淆矩阵，并汇总为多标签概览图

    Args:
        all_preds:  [N, C] 二值预测
        all_labels: [N, C] 真实标签
        class_names: 类别名称列表
        save_path: 图片保存路径
        title_prefix: 标题前缀 (e.g. "Test" / "Val")
    """
    if not HAS_MPL:
        print("Warning: matplotlib not available, skipping confusion matrix plot.")
        return

    n_classes = len(class_names)
    ncols = 4
    nrows = (n_classes + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten() if n_classes > 1 else [axes]

    for c in range(n_classes):
        ax = axes[c]
        tp = ((all_preds[:, c] == 1) & (all_labels[:, c] == 1)).sum()
        fp = ((all_preds[:, c] == 1) & (all_labels[:, c] == 0)).sum()
        fn = ((all_preds[:, c] == 0) & (all_labels[:, c] == 1)).sum()
        tn = ((all_preds[:, c] == 0) & (all_labels[:, c] == 0)).sum()

        cm = np.array([[tn, fp], [fn, tp]])
        im = ax.imshow(cm, cmap='Blues', vmin=0)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha='center', va='center',
                        fontsize=11, fontweight='bold',
                        color='white' if cm[i, j] > cm.max() * 0.5 else 'black')

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Neg', 'Pos'], fontsize=9)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Neg', 'Pos'], fontsize=9)
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('Actual', fontsize=9)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        ax.set_title(f"{class_names[c]}\nF1={f1:.3f} P={precision:.3f} R={recall:.3f}", fontsize=10)

    # 隐藏多余子图
    for c in range(n_classes, len(axes)):
        axes[c].set_visible(False)

    fig.suptitle(f"{title_prefix} Per-Class Confusion Matrices", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrices saved to {save_path}")


def plot_combined_confusion_matrix(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    class_names: List[str],
    save_path: str,
    title_prefix: str = "",
):
    """汇总混淆矩阵: C×C 矩阵，行=真实类别，列=预测类别

    对于多标签，每个样本可能有多个正类。这里统计：
    行 i，列 j = 真实类别 i 被预测为类别 j 的次数（共现/混淆统计）
    """
    if not HAS_MPL:
        return

    n_classes = len(class_names)
    co_occur = np.zeros((n_classes, n_classes), dtype=np.float64)

    for c_true in range(n_classes):
        mask_true = all_labels[:, c_true] == 1
        if mask_true.sum() == 0:
            continue
        preds_for_true = all_preds[mask_true]
        co_occur[c_true] = preds_for_true.sum(axis=0) / mask_true.sum()

    fig, ax = plt.subplots(figsize=(max(10, n_classes * 0.8), max(8, n_classes * 0.7)))
    im = ax.imshow(co_occur, cmap='YlOrRd', vmin=0, vmax=1)

    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, f"{co_occur[i, j]:.2f}", ha='center', va='center',
                    fontsize=8, color='white' if co_occur[i, j] > 0.5 else 'black')

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title(f"{title_prefix} Label Co-occurrence / Confusion\n(row=actual, col=predicted, values=ratio)", fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Combined confusion matrix saved to {save_path}")


# ============================================================================
# Raw Feature Collector (for t-SNE on original 22-dim features)
# ============================================================================

def collect_raw_features(dataloader: DataLoader, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """从 DataLoader 收集展平的原始 22 维特征和标签

    Returns:
        raw_feats: [N, 22] 拼接的原始特征向量
        all_labels: [N, C] 多热标签
    """
    all_feats, all_labels = [], []
    for features_dict, labels in tqdm(dataloader, desc="Collecting raw features"):
        # 按 DOMAIN_ORDER 拼接 → [B, 22]
        flat = torch.cat([features_dict[d] for d in FeatureConditionedPromptLearner.DOMAIN_ORDER], dim=1)
        all_feats.append(flat.numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_feats), np.concatenate(all_labels)


# ============================================================================
# t-SNE Visualization
# ============================================================================

def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    save_path: str,
    title_prefix: str = "",
    max_samples: int = 3000,
    perplexity: float = 30.0,
):
    """t-SNE 可视化学习到的特征表示

    生成两种图:
    1. 按主要类别着色 (primary class = 第一个正标签)
    2. 每类高亮图 (positive vs negative)

    Args:
        features: [N, D] 特征向量
        labels:  [N, C] 多热标签
        class_names: 类别名称列表
        save_path: 保存路径 (会衍生 _perclass 图)
        title_prefix: 标题前缀
        max_samples: t-SNE 最大样本数 (超过则随机采样)
        perplexity: t-SNE perplexity 参数
    """
    if not HAS_MPL:
        print("Warning: matplotlib not available, skipping t-SNE.")
        return

    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("Warning: sklearn not available, skipping t-SNE.")
        return

    N = features.shape[0]
    if N > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(N, max_samples, replace=False)
        features = features[idx]
        labels = labels[idx]
        N = max_samples

    print(f"  Running t-SNE on {N} samples (perplexity={perplexity})...")

    # Standardize
    feats_scaled = StandardScaler().fit_transform(features)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=min(perplexity, N - 1),
                random_state=42, verbose=0)
    feats_2d = tsne.fit_transform(feats_scaled)

    n_classes = len(class_names)
    cmap = plt.cm.tab10

    # Determine primary class per sample
    primary = np.full(N, -1, dtype=int)
    for i in range(N):
        pos = np.where(labels[i] == 1)[0]
        primary[i] = pos[0] if len(pos) > 0 else -1
    has_label = primary >= 0

    # ---- Plot 1: colored by primary class ----
    fig, ax = plt.subplots(figsize=(11, 9))
    unique_classes = sorted(set(primary[has_label]))

    for c in unique_classes:
        mask = primary == c
        ax.scatter(feats_2d[mask, 0], feats_2d[mask, 1],
                   c=[cmap(c % 10)], label=class_names[c],
                   s=8, alpha=0.6, edgecolors='none')

    # Unlabeled
    if (~has_label).sum() > 0:
        ax.scatter(feats_2d[~has_label, 0], feats_2d[~has_label, 1],
                   c='gray', label='No label', s=8, alpha=0.3, edgecolors='none')

    ax.legend(loc='lower left', fontsize=7, markerscale=2, ncol=2)
    ax.set_title(f"{title_prefix} t-SNE by Primary Class (perplexity={perplexity})", fontsize=13, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  t-SNE (primary class) saved to {save_path}")

    # ---- Plot 2: per-class highlight grid ----
    ncols = 4
    nrows = (n_classes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    for c in range(n_classes):
        ax = axes[c]
        pos_mask = labels[:, c] == 1
        neg_mask = ~pos_mask

        # Negative samples (background)
        ax.scatter(feats_2d[neg_mask, 0], feats_2d[neg_mask, 1],
                   c='lightgray', s=3, alpha=0.3, edgecolors='none')
        # Positive samples (highlighted)
        ax.scatter(feats_2d[pos_mask, 0], feats_2d[pos_mask, 1],
                   c=[cmap(c % 10)], s=12, alpha=0.7, edgecolors='none',
                   label=f'{class_names[c]} (n={pos_mask.sum()})')

        ax.set_title(f"{class_names[c]}", fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=7, loc='upper right')

    for c in range(n_classes, len(axes)):
        axes[c].set_visible(False)

    fig.suptitle(f"{title_prefix} t-SNE Per-Class Highlight (perplexity={perplexity})", fontsize=13, fontweight='bold')
    plt.tight_layout()
    perclass_path = save_path.replace('.png', '_perclass.png')
    plt.savefig(perclass_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  t-SNE (per-class) saved to {perclass_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Feature-Only Training (no STFT)")
    parser.add_argument("--config", type=str, default="multi/config.yaml")
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "meta_net"],
                        help="Model architecture: mlp (plain baseline) or meta_net")
    parser.add_argument("--context_dim", type=int, default=128, help="[meta_net] Prompt learner output dim")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden layer dim")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Which split to evaluate on (default: test)")
    parser.add_argument("--tsne_raw", action="store_true",
                        help="Run t-SNE on raw 22-dim features (no model needed)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Prediction threshold (default: from config or 0.5)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume/load checkpoint")
    parser.add_argument("--output_dir", type=str, default="checkpoints/feature_only")
    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Threshold: CLI > config.evaluation.threshold > default 0.5
    threshold = args.threshold
    if threshold is None:
        threshold = config.get("evaluation", {}).get("threshold", 0.5)
    print(f"Prediction threshold: {threshold}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, test_loader = build_dataloaders(config)
    class_names = [cls["name"] for cls in config.get("jamming_classes", [])]
    num_classes = len(class_names)
    print(f"Classes: {num_classes}")

    # --- Data diagnostics ---
    print("\n--- Data Diagnostics ---")
    # Sample one batch to check feature stats
    sample_batch, sample_labels = next(iter(train_loader))
    flat_feats = torch.cat([sample_batch[d] for d in FeatureConditionedPromptLearner.DOMAIN_ORDER], dim=1)
    print(f"  Feature shape: {list(flat_feats.shape)}")
    print(f"  Feature value range: [{flat_feats.min().item():.3f}, {flat_feats.max().item():.3f}]")
    print(f"  Feature std (mean): {flat_feats.std(dim=0).mean().item():.3f}")
    # Label distribution
    pos_counts = sample_labels.sum(dim=0)
    print(f"  Label distribution (batch):")
    for i, name in enumerate(class_names):
        print(f"    {name:6s}: {int(pos_counts[i]):4d}")
    print(f"  Total positive labels: {int(sample_labels.sum())} (avg {sample_labels.sum()/sample_labels.size(0):.1f} per sample)")
    print("-----------------------------\n")

    # Raw feature t-SNE (no model needed)
    if args.tsne_raw:
        print(f"\n=== t-SNE on Raw 22-dim Features (split={args.split}) ===")
        split_loader = {"train": train_loader, "val": val_loader, "test": test_loader}.get(args.split)
        if split_loader is None:
            print(f"No {args.split} data found!")
            return
        raw_feats, raw_labels = collect_raw_features(split_loader)
        print(f"  Raw features shape: {raw_feats.shape}, Labels shape: {raw_labels.shape}")
        if HAS_MPL:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            prefix = f"{args.split.capitalize()} Raw Features"
            plot_tsne(raw_feats, raw_labels, class_names,
                      str(out_dir / f"tsne_raw_{args.split}.png"),
                      title_prefix=prefix)
        return

    # Model
    if args.model_type == "mlp":
        model = PlainMLPModel(
            num_classes=num_classes,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        ).to(device)
        print(f"Model: PlainMLP ({args.hidden_dim} hidden)")
    else:
        n_ctx_config = config.get("n_ctx_per_domain", None)
        model = MetaNetModel(
            num_classes=num_classes,
            context_dim=args.context_dim,
            n_ctx_per_domain=n_ctx_config,
            classifier_hidden=args.hidden_dim,
            dropout=args.dropout,
        ).to(device)
        print(f"Model: MetaNet (context_dim={args.context_dim})")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total_params:,} total, {trainable_params:,} trainable")

    # Loss — compute pos_weight to handle class imbalance
    # Count positives per class across all training data
    pos_counts = torch.zeros(num_classes)
    total_samples_train = 0
    for _, labels in train_loader:
        pos_counts += labels.sum(dim=0)
        total_samples_train += labels.size(0)
    neg_counts = total_samples_train - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-8)  # higher weight for rare classes
    pos_weight = pos_weight.to(device)
    print(f"Pos weight (neg/pos per class): {[f'{w:.1f}' for w in pos_weight.tolist()]}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(500, total_steps // 5)
    warmup = LinearLR(optimizer, start_factor=0.5, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    # Resume
    start_epoch = 0
    best_f1 = 0.0
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if not args.eval_only:
            optimizer.load_state_dict(ckpt.get("optimizer_state_dict", optimizer.state_dict()))
            start_epoch = ckpt.get("epoch", 0) + 1
            best_f1 = ckpt.get("best_f1", 0.0)
        print(f"Loaded checkpoint from {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    # Eval only
    if args.eval_only:
        print(f"\n=== Evaluation Only (split={args.split}) ===")
        split_loader = {"train": train_loader, "val": val_loader, "test": test_loader}.get(args.split)
        if split_loader is None:
            print(f"No {args.split} data found! Make sure the split exists in the data.")
            return

        metrics = evaluate(model, split_loader, loss_fn, device, return_details=True, threshold=threshold)
        for k in ["loss", "exact_match", "micro_f1", "macro_f1"]:
            print(f"  {k}: {metrics[k]:.4f}")

        # Confusion matrices
        if HAS_MPL:
            cm_dir = Path(args.output_dir)
            cm_dir.mkdir(parents=True, exist_ok=True)
            prefix = args.split.capitalize()
            plot_confusion_matrices(
                metrics["all_preds"], metrics["all_labels"],
                class_names, str(cm_dir / f"confusion_per_class_{args.split}.png"),
                title_prefix=prefix,
            )
            plot_combined_confusion_matrix(
                metrics["all_preds"], metrics["all_labels"],
                class_names, str(cm_dir / f"confusion_combined_{args.split}.png"),
                title_prefix=prefix,
            )
            plot_tsne(
                metrics["all_features"], metrics["all_labels"],
                class_names, str(cm_dir / f"tsne_{args.split}.png"),
                title_prefix=prefix,
            )
        return

    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nTraining {args.epochs} epochs, saving to {output_dir}")

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device, threshold=threshold)
        scheduler.step()

        val_metrics = evaluate(model, val_loader or train_loader, loss_fn, device,
                               return_details=((epoch + 1) % 5 == 0 or epoch == 0),
                               threshold=threshold)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            per_class = val_metrics.get("per_class", {})
            f1s = [per_class[c]["f1"] for c in range(num_classes)]
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Train loss={train_metrics['loss']:.4f} em={train_metrics['exact_match']:.4f} | "
                  f"Val loss={val_metrics['loss']:.4f} micro_f1={val_metrics['micro_f1']:.4f} macro_f1={val_metrics['macro_f1']:.4f}")
            print(f"  Prob: mean={val_metrics.get('prob_mean', 0):.3f} max={val_metrics.get('prob_max', 0):.3f} pos_rate={val_metrics.get('pred_positive_rate', 0):.3f}")
            print(f"  Per-class F1: " + " | ".join(
                f"{class_names[c]}:{f1s[c]:.3f}" for c in range(num_classes)))
        else:
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Train loss={train_metrics['loss']:.4f} em={train_metrics['exact_match']:.4f} | "
                  f"Val loss={val_metrics['loss']:.4f} micro_f1={val_metrics['micro_f1']:.4f} macro_f1={val_metrics['macro_f1']:.4f}")

        current_f1 = val_metrics["micro_f1"]

        # Save best
        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1,
                "config": {k: v for k, v in args.__dict__.items() if k != 'config'},
            }, output_dir / "best_model.pt")
            print(f"  → Best model saved (micro_f1={best_f1:.4f})")

        # Save periodically
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1,
            }, output_dir / f"checkpoint_epoch_{epoch+1}.pt")

    # Final test evaluation
    print("\n=== Final Test Evaluation ===")
    if test_loader:
        test_metrics = evaluate(model, test_loader, loss_fn, device, return_details=True, threshold=threshold)
        for k in ["loss", "exact_match", "micro_f1", "macro_f1"]:
            print(f"  {k}: {test_metrics[k]:.4f}")

        if HAS_MPL:
            plot_confusion_matrices(
                test_metrics["all_preds"], test_metrics["all_labels"],
                class_names, str(output_dir / "confusion_per_class_test.png"),
                title_prefix="Test",
            )
            plot_combined_confusion_matrix(
                test_metrics["all_preds"], test_metrics["all_labels"],
                class_names, str(output_dir / "confusion_combined_test.png"),
                title_prefix="Test",
            )
            plot_tsne(
                test_metrics["all_features"], test_metrics["all_labels"],
                class_names, str(output_dir / "tsne_test.png"),
                title_prefix="Test",
            )
    else:
        print("No test data available.")

    print(f"\nBest val micro_f1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
