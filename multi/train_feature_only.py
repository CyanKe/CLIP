"""
仅使用特征条件上下文 (Feature-Conditioned Context) 进行训练和评估
不使用 STFT 图像数据，仅依赖 22 维信号特征

用法:
  python -m multi.train_feature_only --config multi/config.yaml
  python -m multi.train_feature_only --config multi/config.yaml --eval_only --checkpoint path/to/model.pt

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi.prompt_learner import FeatureConditionedPromptLearner


# ============================================================================
# FeatureOnlyModel
# ============================================================================

class FeatureOnlyModel(nn.Module):
    """仅使用信号特征进行分类，不依赖 STFT 图像

    22维信号特征 → FeatureConditionedPromptLearner (Meta-Net)
    → context vectors [B, M, D] → mean pool → [B, D]
    → MLP classifier → [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int,
        context_dim: int = 512,
        n_ctx_per_domain: dict = None,
        classifier_hidden: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.context_dim = context_dim

        self.prompt_learner = FeatureConditionedPromptLearner(
            transformer_width=context_dim,
            n_ctx_per_domain=n_ctx_per_domain,
            hidden_dim=context_dim // 2,
        )

        self.classifier = nn.Sequential(
            nn.Linear(context_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, features_dict: dict) -> torch.Tensor:
        context = self.prompt_learner(features_dict)  # [B, M, D]
        pooled = context.mean(dim=1)                   # [B, D]
        return self.classifier(pooled)                 # [B, num_classes]


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

def train_epoch(model, dataloader, loss_fn, optimizer, device):
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
        preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += (preds == labels).all(dim=1).sum().item()
        total_samples += labels.size(0)

    return {
        "loss": total_loss / total_samples,
        "exact_match": total_correct / total_samples,
    }


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_labels = [], []

    for features_dict, labels in tqdm(dataloader, desc="Eval"):
        features_dict = {k: v.to(device) for k, v in features_dict.items()}
        labels = labels.to(device)

        logits = model(features_dict)
        loss = loss_fn(logits, labels)

        total_loss += loss.item() * labels.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        total_correct += (preds == labels).all(dim=1).sum().item()
        total_samples += labels.size(0)

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
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        per_class[c] = {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall + 1e-8)}

    # Micro/macro F1
    tp_all = ((all_preds == 1) & (all_labels == 1)).sum()
    fp_all = ((all_preds == 1) & (all_labels == 0)).sum()
    fn_all = ((all_preds == 0) & (all_labels == 1)).sum()
    micro_p = tp_all / (tp_all + fp_all + 1e-8)
    micro_r = tp_all / (tp_all + fn_all + 1e-8)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-8)

    macro_f1 = np.mean([per_class[c]["f1"] for c in range(all_labels.shape[1])])

    return {
        "loss": total_loss / total_samples,
        "exact_match": total_correct / total_samples,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Feature-Only Training (no STFT)")
    parser.add_argument("--config", type=str, default="multi/config.yaml")
    parser.add_argument("--context_dim", type=int, default=512, help="Prompt learner output dim")
    parser.add_argument("--classifier_hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume/load checkpoint")
    parser.add_argument("--output_dir", type=str, default="checkpoints/feature_only")
    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, test_loader = build_dataloaders(config)
    class_names = [cls["name"] for cls in config.get("jamming_classes", [])]
    num_classes = len(class_names)
    print(f"Classes: {num_classes}")

    # Model
    n_ctx_config = config.get("n_ctx_per_domain", None)
    model = FeatureOnlyModel(
        num_classes=num_classes,
        context_dim=args.context_dim,
        n_ctx_per_domain=n_ctx_config,
        classifier_hidden=args.classifier_hidden,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total_params:,} total, {trainable_params:,} trainable")

    # Loss
    loss_fn = nn.BCEWithLogitsLoss()

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(500, total_steps // 5)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
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
        print("\n=== Evaluation Only ===")
        eval_loader = test_loader or val_loader
        if eval_loader is None:
            print("No evaluation data found!")
            return
        metrics = evaluate(model, eval_loader, loss_fn, device)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        return

    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nTraining {args.epochs} epochs, saving to {output_dir}")

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device)
        scheduler.step()

        val_metrics = evaluate(model, val_loader or train_loader, loss_fn, device)

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
        test_metrics = evaluate(model, test_loader, loss_fn, device)
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
    else:
        print("No test data available.")

    print(f"\nBest val micro_f1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
