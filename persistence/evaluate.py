"""
Evaluation script for Persistence Spectrum CLIP CZSL model.

Supports three modes:
    zero_shot       — global zero-shot evaluation
    by_combination  — per-combination (seen/unseen) breakdown
    by_jnr          — per-JNR performance analysis

Usage:
    python persistence/evaluate.py --checkpoint checkpoints/persistence/persistence_best_model.pt --mode zero_shot
    python persistence/evaluate.py --checkpoint CHKPT --mode by_combination --split test
    python persistence/evaluate.py --checkpoint CHKPT --mode by_jnr --split test --output_dir results

Adapted from conformer_1d/evaluate_conformer.py for PersistenceCLIPForCZSL.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _parent)

from persistence.model import PersistenceCLIPForCZSL, create_persistence_model
from persistence.data import PersistenceDataset, create_persistence_dataloaders, collate_fn, TokenizerWrapper


# ---------------------------------------------------------------------------
# JNR-by-JNR dataloader factory
# ---------------------------------------------------------------------------

def create_persistence_jnr_dataloaders(
    config: dict,
    split: str = 'test',
) -> dict:
    """Create one DataLoader per JNR level for persistence spectrum data.

    Args:
        config: full config dict
        split: 'train', 'val', or 'test'

    Returns:
        dict mapping jnr_level → DataLoader
    """
    data_config = config.get('data', {})
    base_path = data_config.get('base_path')
    jnr_start = data_config.get('jnr_start', 0)
    jnr_end = data_config.get('jnr_end', 20)
    jnr_step = data_config.get('jnr_step', 1)
    persistence_var_name = data_config.get('persistence_var_name', 'all_persistences')
    persistence_suffix = data_config.get('persistence_suffix', 'echo_persistences')
    image_size = data_config.get('image_size', 224)
    batch_size = config.get('train', {}).get('batch_size', 32)
    num_workers = data_config.get('num_workers', 0)
    pin_memory = data_config.get('pin_memory', True)

    jamming_classes = config.get('jamming_classes', [])
    class_names = [jc['name'] if isinstance(jc, dict) else jc for jc in jamming_classes]

    jnr_levels = list(range(jnr_start, jnr_end + 1, jnr_step))

    tokenizer = TokenizerWrapper(model_type="clip")
    collate = partial(collate_fn, tokenizer_fn=tokenizer, model_type="clip")

    jnr_loaders = {}

    for jnr in jnr_levels:
        data_folder = os.path.join(base_path, f'JNR_+{jnr}')
        persistence_file = os.path.join(data_folder, f'{split}_{persistence_suffix}.mat')
        metadata_file = os.path.join(data_folder, f'{split}_echo_metadata.json')

        if not os.path.exists(persistence_file) or not os.path.exists(metadata_file):
            print(f"  Skipping JNR={jnr}: data not found")
            continue

        ds = PersistenceDataset(
            persistence_file=persistence_file,
            metadata_file=metadata_file,
            persistence_var_name=persistence_var_name,
            class_names=class_names,
            image_size=image_size,
            apply_clip_norm=True,
        )

        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            collate_fn=collate,
        )
        jnr_loaders[jnr] = loader
        print(f"  JNR=+{jnr}: {len(ds)} samples")

    return jnr_loaders


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class PersistenceEvaluator:
    """Evaluation suite for PersistenceCLIPForCZSL."""

    def __init__(self, model: PersistenceCLIPForCZSL, config: dict, device: torch.device):
        self.model = model
        self.config = config
        self.device = device

        jamming_classes = config.get('jamming_classes', [])
        self.class_names = [jc['name'] if isinstance(jc, dict) else jc for jc in jamming_classes]
        self.num_classes = len(self.class_names)

        self.czsl_config = config.get('czsl', {})
        self.eval_config = config.get('evaluation', {})

    # ------------------------------------------------------------------
    # Zero-shot evaluation on a single DataLoader
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_zero_shot(
        self,
        dataloader,
        threshold: float = None,
    ) -> dict:
        """Run zero-shot evaluation on a dataloader.

        Returns per-sample predictions, labels, and metrics.
        """
        if threshold is None:
            threshold = self.eval_config.get('threshold', 0.5)

        self.model.eval()

        all_preds = []
        all_labels = []
        all_metas = []

        for batch_data in tqdm(dataloader, desc="Zero-shot eval"):
            if len(batch_data) >= 6:
                images, _, text_tokens, labels, texts, metas = batch_data[:6]
            else:
                images, text_tokens, labels, texts, metas = batch_data

            images = images.to(self.device)
            labels = labels.to(self.device)

            predictions = self.model.zero_shot_predict(images, threshold=threshold)

            for pred_names in predictions:
                pred_vec = np.zeros(self.num_classes, dtype=np.float32)
                for name in pred_names:
                    # Handle single class names and "A+B" combo names
                    for part in name.split('+'):
                        part = part.strip()
                        if part in self.class_names:
                            pred_vec[self.class_names.index(part)] = 1.0
                all_preds.append(pred_vec)

            all_labels.append(labels.cpu().numpy())
            all_metas.extend(metas)

        all_preds = np.array(all_preds)
        all_labels = np.concatenate(all_labels, axis=0)

        metrics = self._compute_multilabel_metrics(all_labels, all_preds)
        metrics['num_samples'] = len(all_preds)

        return metrics, all_preds, all_labels, all_metas

    # ------------------------------------------------------------------
    # Multilabel metrics
    # ------------------------------------------------------------------

    def _compute_multilabel_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict:
        """Compute multilabel classification metrics."""
        n_samples = y_true.shape[0]

        # Per-class F1
        per_class_f1 = {}
        for i, cls_name in enumerate(self.class_names):
            if y_true[:, i].sum() > 0 or y_pred[:, i].sum() > 0:
                f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
                per_class_f1[cls_name] = f1

        # Micro / Macro / Sample F1
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_samples = f1_score(y_true, y_pred, average='samples', zero_division=0)

        # Precision / Recall
        precision = precision_score(y_true, y_pred, average='samples', zero_division=0)
        recall = recall_score(y_true, y_pred, average='samples', zero_division=0)

        # Subset accuracy (exact match)
        subset_acc = accuracy_score(y_true, y_pred)

        return {
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_samples': f1_samples,
            'precision': precision,
            'recall': recall,
            'subset_accuracy': subset_acc,
            'per_class_f1': per_class_f1,
        }

    # ------------------------------------------------------------------
    # By-combination evaluation (seen vs unseen)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_by_combination(
        self,
        dataloader,
        threshold: float = None,
    ) -> dict:
        """Evaluate broken down by seen and unseen combinations."""
        if threshold is None:
            threshold = self.czsl_config.get('zero_shot', {}).get('threshold', 0.14)

        seen_combos_raw = self.czsl_config.get('seen_combinations', [])
        unseen_combos_raw = self.czsl_config.get('unseen_combinations', [])

        # Build sets of combination signatures
        def combo_key(combo):
            return tuple(sorted(combo))

        seen_keys = set()
        for sc in seen_combos_raw:
            seen_keys.add(combo_key(sc))

        unseen_keys = set()
        for uc in unseen_combos_raw:
            unseen_keys.add(combo_key(uc))

        self.model.eval()

        seen_preds, seen_labels = [], []
        unseen_preds, unseen_labels = [], []

        for batch_data in tqdm(dataloader, desc="By-combination eval"):
            if len(batch_data) >= 6:
                images, _, text_tokens, labels, texts, metas = batch_data[:6]
            else:
                images, text_tokens, labels, texts, metas = batch_data

            images = images.to(self.device)
            labels_np = labels.cpu().numpy()

            predictions = self.model.zero_shot_predict(images, threshold=threshold)

            for i, pred_names in enumerate(predictions):
                pred_vec = np.zeros(self.num_classes, dtype=np.float32)
                for name in pred_names:
                    for part in name.split('+'):
                        part = part.strip()
                        if part in self.class_names:
                            pred_vec[self.class_names.index(part)] = 1.0

                true_classes = tuple(sorted([
                    self.class_names[j] for j in range(self.num_classes)
                    if labels_np[i, j] > 0
                ]))

                if true_classes in seen_keys or len(true_classes) <= 1:
                    seen_preds.append(pred_vec)
                    seen_labels.append(labels_np[i])
                elif true_classes in unseen_keys:
                    unseen_preds.append(pred_vec)
                    unseen_labels.append(labels_np[i])

        results = {}
        if seen_preds:
            seen_preds = np.array(seen_preds)
            seen_labels = np.array(seen_labels)
            results['seen'] = self._compute_multilabel_metrics(seen_labels, seen_preds)
            results['seen']['count'] = len(seen_preds)
            print(f"\nSeen combinations ({len(seen_preds)} samples):")
            print(f"  F1_macro: {results['seen']['f1_macro']:.4f}")
            print(f"  F1_micro: {results['seen']['f1_micro']:.4f}")

        if unseen_preds:
            unseen_preds = np.array(unseen_preds)
            unseen_labels = np.array(unseen_labels)
            results['unseen'] = self._compute_multilabel_metrics(unseen_labels, unseen_preds)
            results['unseen']['count'] = len(unseen_preds)
            print(f"\nUnseen combinations ({len(unseen_preds)} samples):")
            print(f"  F1_macro: {results['unseen']['f1_macro']:.4f}")
            print(f"  F1_micro: {results['unseen']['f1_micro']:.4f}")

        return results

    # ------------------------------------------------------------------
    # By-JNR evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_by_jnr(
        self,
        jnr_loaders: dict,
        threshold: float = None,
        output_dir: str = None,
    ) -> dict:
        """Evaluate separately at each JNR level."""
        if threshold is None:
            threshold = self.czsl_config.get('zero_shot', {}).get('threshold', 0.14)

        jnr_results = {}
        all_jnr_metrics = []

        for jnr in sorted(jnr_loaders.keys()):
            loader = jnr_loaders[jnr]
            metrics, preds, labels, metas = self.evaluate_zero_shot(loader, threshold)
            metrics['jnr'] = jnr
            metrics['num_samples'] = len(preds)
            jnr_results[jnr] = metrics
            all_jnr_metrics.append(metrics)
            print(f"  JNR=+{jnr}: F1_macro={metrics['f1_macro']:.4f}, "
                  f"F1_micro={metrics['f1_micro']:.4f}")

        # Plot per-JNR F1 curve
        if output_dir and all_jnr_metrics:
            self._plot_jnr_curve(all_jnr_metrics, output_dir)

        return jnr_results

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_jnr_curve(self, jnr_metrics: list, output_dir: str):
        """Plot F1 vs JNR curve."""
        os.makedirs(output_dir, exist_ok=True)

        jnrs = [m['jnr'] for m in jnr_metrics]
        f1_macros = [m['f1_macro'] for m in jnr_metrics]
        f1_micros = [m['f1_micro'] for m in jnr_metrics]

        plt.figure(figsize=(10, 6))
        plt.plot(jnrs, f1_macros, 'o-', label='F1 Macro', linewidth=2)
        plt.plot(jnrs, f1_micros, 's-', label='F1 Micro', linewidth=2)
        plt.xlabel('JNR (dB)', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.title('Persistence Spectrum — F1 vs JNR', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'persistence_f1_vs_jnr.png'), dpi=150)
        plt.close()
        print(f"\nSaved JNR curve to {output_dir}/persistence_f1_vs_jnr.png")

        # Per-class heatmap
        if jnr_metrics and 'per_class_f1' in jnr_metrics[0]:
            classes = list(jnr_metrics[0]['per_class_f1'].keys())
            heatmap_data = np.zeros((len(classes), len(jnrs)))
            for j, m in enumerate(jnr_metrics):
                for i, cls in enumerate(classes):
                    heatmap_data[i, j] = m['per_class_f1'].get(cls, 0.0)

            plt.figure(figsize=(14, 8))
            sns.heatmap(heatmap_data, xticklabels=jnrs, yticklabels=classes,
                        annot=True, fmt='.3f', cmap='YlOrRd')
            plt.xlabel('JNR (dB)', fontsize=12)
            plt.ylabel('Class', fontsize=12)
            plt.title('Persistence Spectrum — Per-Class F1 vs JNR', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'persistence_per_class_f1_heatmap.png'), dpi=150)
            plt.close()
            print(f"Saved per-class heatmap to {output_dir}/persistence_per_class_f1_heatmap.png")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Persistence Spectrum CLIP CZSL Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file (default: use config from checkpoint)")
    parser.add_argument("--mode", type=str, default="zero_shot",
                        choices=["zero_shot", "by_combination", "by_jnr"],
                        help="Evaluation mode")
    parser.add_argument("--split", type=str, default="test",
                        help="Data split to evaluate on")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override prediction threshold")
    parser.add_argument("--output_dir", type=str, default="results/persistence",
                        help="Output directory for results and plots")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = checkpoint.get('config', {})
        if not config:
            raise ValueError("No config found in checkpoint and --config not specified")

    # Create model
    print("\nReconstructing model from checkpoint...")
    jamming_classes = config.get('jamming_classes', [])
    class_names = [jc['name'] if isinstance(jc, dict) else jc for jc in jamming_classes]

    model = create_persistence_model(config, device=str(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # Cache text features
    czsl_config = config.get("czsl", {})
    seen_combos = czsl_config.get("seen_combinations", None)
    model.cache_text_features(
        max_combination_size=2,
        include_single=True,
        seen_combinations=seen_combos,
    )

    evaluator = PersistenceEvaluator(model, config, device)

    if args.mode == "zero_shot":
        print(f"\n{'='*60}")
        print(f"Zero-Shot Evaluation on {args.split} split")
        print(f"{'='*60}")

        train_loader, val_loader, test_loader, num_classes, jnr_levels = \
            create_persistence_dataloaders(config)

        loader = {'train': train_loader, 'val': val_loader, 'test': test_loader}[args.split]
        if loader is None:
            print(f"No {args.split} data found!")
            return

        metrics, preds, labels, metas = evaluator.evaluate_zero_shot(
            loader, threshold=args.threshold
        )

        print(f"\nResults ({metrics['num_samples']} samples):")
        print(f"  F1_macro:       {metrics['f1_macro']:.4f}")
        print(f"  F1_micro:       {metrics['f1_micro']:.4f}")
        print(f"  F1_samples:     {metrics['f1_samples']:.4f}")
        print(f"  Precision:      {metrics['precision']:.4f}")
        print(f"  Recall:         {metrics['recall']:.4f}")
        print(f"  Subset accuracy: {metrics['subset_accuracy']:.4f}")
        print(f"\nPer-class F1:")
        for cls_name, f1 in sorted(metrics['per_class_f1'].items()):
            print(f"  {cls_name:8s}: {f1:.4f}")

    elif args.mode == "by_combination":
        print(f"\n{'='*60}")
        print(f"By-Combination Evaluation on {args.split} split")
        print(f"{'='*60}")

        train_loader, val_loader, test_loader, num_classes, jnr_levels = \
            create_persistence_dataloaders(config)

        loader = {'train': train_loader, 'val': val_loader, 'test': test_loader}[args.split]
        if loader is None:
            print(f"No {args.split} data found!")
            return

        evaluator.evaluate_by_combination(loader, threshold=args.threshold)

    elif args.mode == "by_jnr":
        print(f"\n{'='*60}")
        print(f"By-JNR Evaluation on {args.split} split")
        print(f"{'='*60}")

        jnr_loaders = create_persistence_jnr_dataloaders(config, split=args.split)

        if not jnr_loaders:
            print(f"No JNR data found!")
            return

        evaluator.evaluate_by_jnr(
            jnr_loaders,
            threshold=args.threshold,
            output_dir=args.output_dir,
        )

    print(f"\nEvaluation completed!")


if __name__ == "__main__":
    main()
