"""
Visualize saved evaluation results from .npz files — no inference needed.

Usage:
python persistence/visualize_results.py   -z results/persistence_fusion/persistence_fusion_zeroshot_test.npz   -b results/persistence_fusion/persistence_fusion_bycombo_test.npz   --visualize

    python persistence/visualize_results.py --zeroshot results/persistence_fusion/persistence_fusion_zeroshot_test.npz --tsne --roc
    python persistence/visualize_results.py --bycombo results/persistence_fusion/persistence_fusion_bycombo_test.npz --visualize
    python persistence/visualize_results.py -z zs.npz -b bc.npz --visualize --output_dir results/viz
"""
import os, sys, argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.manifold import TSNE

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _parent)

from persistence.evaluate import (
    plot_roc_curves, plot_pr_curves, _unwrap_combinations, _convert_combination_names_to_indices,
)


# ===========================================================================
# Plot functions (copied from evaluate.py for standalone use)
# ===========================================================================

def plot_confusion_by_combination(labels, preds, class_names, save_path,
                                   seen_combinations=None, unseen_combinations=None):
    """Plot combination confusion matrix."""
    seen_set = set(tuple(sorted(c)) for c in (seen_combinations or []))
    unseen_set = set(tuple(sorted(c)) for c in (unseen_combinations or []))

    has_markers = bool(seen_set or unseen_set)

    if has_markers:
        unique_combs = sorted(set(
            tuple(sorted(np.where(labels[i] == 1)[0].tolist())) for i in range(len(labels))
        ) | set(
            tuple(sorted(np.where(preds[i] == 1)[0].tolist())) for i in range(len(preds))
        ))
    else:
        true_combs = [tuple(sorted(np.where(labels[i] == 1)[0].tolist())) for i in range(len(labels))]
        pred_combs = [tuple(sorted(np.where(preds[i] == 1)[0].tolist())) for i in range(len(preds))]
        unique_combs = sorted(set(true_combs + pred_combs))

    comb_to_idx = {comb: i for i, comb in enumerate(unique_combs)}
    n_combs = len(unique_combs)
    confusion = np.zeros((n_combs, n_combs), dtype=int)

    for i in range(len(labels)):
        true_comb = tuple(sorted(np.where(labels[i] == 1)[0].tolist()))
        pred_comb = tuple(sorted(np.where(preds[i] == 1)[0].tolist()))
        if true_comb in comb_to_idx and pred_comb in comb_to_idx:
            confusion[comb_to_idx[true_comb], comb_to_idx[pred_comb]] += 1

    comb_names = []
    for comb in unique_combs:
        if len(comb) == 0:
            name = "None"
        else:
            name = "+".join([class_names[i] for i in comb])
        if has_markers:
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
    ax.set_title('Combination Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_label_cooccurrence(labels, preds, class_names, save_path):
    """Plot true / predicted label co-occurrence matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    nc = labels.shape[1]
    sns.heatmap(labels.T @ labels, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names[:nc], yticklabels=class_names[:nc], ax=axes[0])
    axes[0].set_title('True Label Co-occurrence')
    axes[0].tick_params(axis='x', rotation=45)

    sns.heatmap(preds.T @ preds, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names[:nc], yticklabels=class_names[:nc], ax=axes[1])
    axes[1].set_title('Predicted Label Co-occurrence')
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Co-occurrence saved to {save_path}")
    plt.close()


def plot_feature_tsne(features, labels, class_names, save_path,
                      seen_combinations=None, unseen_combinations=None):
    """Plot t-SNE of image features."""
    print("Computing t-SNE projection...")
    num_samples = len(features)
    perplexity = min(30, num_samples - 1) if num_samples > 1 else 1
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    features_2d = tsne.fit_transform(features)

    comb_labels = [tuple(sorted(np.where(labels[i] == 1)[0].tolist())) for i in range(len(labels))]
    seen_set = set(tuple(sorted(c)) for c in (seen_combinations or []))
    unseen_set = set(tuple(sorted(c)) for c in (unseen_combinations or []))

    unique_combs = sorted(set(comb_labels))
    num_combs = len(unique_combs)
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, num_combs)))

    fig, ax = plt.subplots(figsize=(14, 10))
    for idx, comb in enumerate(unique_combs):
        mask = np.array([c == comb for c in comb_labels])
        if mask.sum() > 0:
            name = "+".join([class_names[i] for i in comb]) if comb else "None"
            marker = 'o'
            if comb in seen_set:
                name, marker = f"[S] {name}", 'o'
            elif comb in unseen_set:
                name, marker = f"[U] {name}", '^'
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                       c=[colors[idx % 20]], label=name, alpha=0.6, s=30, marker=marker)

    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    ax.set_title('Feature Space (t-SNE)\n[S]=Seen, [U]=Unseen')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"t-SNE saved to {save_path}")
    plt.close()


def plot_feature_umap(features, labels, class_names, save_path,
                      seen_combinations=None, unseen_combinations=None):
    """Plot UMAP of image features."""
    try:
        import umap
    except ImportError:
        print("UMAP not installed. Run: pip install umap-learn")
        return
    print("Computing UMAP projection...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    features_2d = reducer.fit_transform(features)

    comb_labels = [tuple(sorted(np.where(labels[i] == 1)[0].tolist())) for i in range(len(labels))]
    seen_set = set(tuple(sorted(c)) for c in (seen_combinations or []))
    unseen_set = set(tuple(sorted(c)) for c in (unseen_combinations or []))

    unique_combs = sorted(set(comb_labels))
    num_combs = len(unique_combs)
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, num_combs)))

    fig, ax = plt.subplots(figsize=(14, 10))
    for idx, comb in enumerate(unique_combs):
        mask = np.array([c == comb for c in comb_labels])
        if mask.sum() > 0:
            name = "+".join([class_names[i] for i in comb]) if comb else "None"
            marker = 'o'
            if comb in seen_set:
                name, marker = f"[S] {name}", 'o'
            elif comb in unseen_set:
                name, marker = f"[U] {name}", '^'
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                       c=[colors[idx % 20]], label=name, alpha=0.6, s=30, marker=marker)

    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    ax.set_title('Feature Space (UMAP)\n[S]=Seen, [U]=Unseen')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"UMAP saved to {save_path}")
    plt.close()


# ===========================================================================
# Default class names (same as persistence/config.yaml)
# ===========================================================================

DEFAULT_CLASS_NAMES = [
    "DFTJ", "ISRJ", "ISDJ", "ISCJ", "MISRJ",
    "AJ", "BJ", "SJ", "NCJ", "NPJ",
    "SMSPJ", "C&IJ", "NFMJ", "NPMJ", "NAMJ", "CSJ", "PJ",
]


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize saved .npz evaluation results")
    parser.add_argument("--zeroshot", "-z", type=str, default=None,
                        help="Path to zeroshot .npz file")
    parser.add_argument("--bycombo", "-b", type=str, default=None,
                        help="Path to by-combination .npz file")
    parser.add_argument("--class_names", type=str, nargs="*", default=None,
                        help="Class names (space-separated). Default: standard 17 classes")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                        help="Output directory (default: same dir as first .npz)")
    # Visualization flags
    parser.add_argument("--visualize", action="store_true",
                        help="Generate all visualizations")
    parser.add_argument("--tsne", action="store_true", help="t-SNE only")
    parser.add_argument("--umap", action="store_true", help="UMAP only")
    parser.add_argument("--roc", action="store_true", help="ROC curves only")
    parser.add_argument("--pr", action="store_true", help="PR curves only")
    parser.add_argument("--confusion", action="store_true", help="Confusion matrix only")
    parser.add_argument("--cooccurrence", action="store_true", help="Co-occurrence only")
    args = parser.parse_args()

    if not args.zeroshot and not args.bycombo:
        parser.error("At least one of --zeroshot or --bycombo must be provided")

    class_names = args.class_names or DEFAULT_CLASS_NAMES

    # Determine output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        first = args.zeroshot or args.bycombo
        output_dir = Path(first).parent / "viz"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    do_all = args.visualize
    prefix = Path(args.zeroshot or args.bycombo).stem.replace("persistence_fusion_", "").replace("persistence_", "")

    # --- Zeroshot results ---
    if args.zeroshot:
        print(f"\nLoading zeroshot: {args.zeroshot}")
        data = np.load(args.zeroshot)
        labels = data["labels"]
        predictions = data["predictions"]
        features = data.get("features", None)

        if do_all or args.confusion:
            plot_confusion_by_combination(
                labels, predictions, class_names,
                save_path=str(output_dir / f"{prefix}_confusion_zs.png"),
            )
        if do_all or args.roc:
            plot_roc_curves(
                labels, data.get("probabilities", predictions.astype(float)),
                class_names, save_dir=str(output_dir), prefix=f"{prefix}_zs",
                mode_title="zero_shot",
            )
        if do_all or args.pr:
            plot_pr_curves(
                labels, data.get("probabilities", predictions.astype(float)),
                class_names, save_dir=str(output_dir), prefix=f"{prefix}_zs",
                mode_title="zero_shot",
            )
        if (do_all or args.tsne) and features is not None:
            plot_feature_tsne(features, labels, class_names,
                              save_path=str(output_dir / f"{prefix}_tsne_zs.png"))
        if (do_all or args.umap) and features is not None:
            plot_feature_umap(features, labels, class_names,
                              save_path=str(output_dir / f"{prefix}_umap_zs.png"))
        if do_all or args.cooccurrence:
            plot_label_cooccurrence(
                labels, predictions, class_names,
                save_path=str(output_dir / f"{prefix}_cooccurrence_zs.png"),
            )

    # --- By-combination results ---
    if args.bycombo:
        print(f"\nLoading by-combination: {args.bycombo}")
        data = np.load(args.bycombo)
        labels = data["labels"]
        predictions = data["predictions"]
        features = data.get("features", None)

        if do_all or args.confusion:
            plot_confusion_by_combination(
                labels, predictions, class_names,
                save_path=str(output_dir / f"{prefix}_confusion_bc.png"),
            )
        if do_all or args.roc:
            plot_roc_curves(
                labels, data.get("probabilities", predictions.astype(float)),
                class_names, save_dir=str(output_dir), prefix=f"{prefix}_byc",
                mode_title="by_combination",
            )
        if do_all or args.pr:
            plot_pr_curves(
                labels, data.get("probabilities", predictions.astype(float)),
                class_names, save_dir=str(output_dir), prefix=f"{prefix}_byc",
                mode_title="by_combination",
            )
        if (do_all or args.tsne) and features is not None:
            plot_feature_tsne(features, labels, class_names,
                              save_path=str(output_dir / f"{prefix}_tsne_bc.png"))
        if (do_all or args.umap) and features is not None:
            plot_feature_umap(features, labels, class_names,
                              save_path=str(output_dir / f"{prefix}_umap_bc.png"))
        if do_all or args.cooccurrence:
            plot_label_cooccurrence(
                labels, predictions, class_names,
                save_path=str(output_dir / f"{prefix}_cooccurrence_bc.png"),
            )

    print(f"\nDone! All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
