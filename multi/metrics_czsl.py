"""
Unified CZSL evaluation metrics for multi / persistence / conformer_1d.

Primary exact-match field: subset_accuracy
Dual-write (legacy alias): combination_accuracy == subset_accuracy

Schema (MetricsBundle):
  num_samples, subset_accuracy, combination_accuracy (alias),
  partial_match_accuracy,
  seen_accuracy, seen_samples, unseen_accuracy, unseen_samples,
  other_accuracy, other_samples, harmonic_mean,
  f1_macro, f1_micro, f1_samples,
  precision_macro, recall_macro, precision_samples, recall_samples,
  per_class: {name: {f1, precision, recall, support}},
  optional: auc_macro, auc_micro, ap_macro, ap_micro
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

Number = Union[int, float, np.integer, np.floating]


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def _combo_key_from_row(row: np.ndarray) -> Tuple[int, ...]:
    return tuple(sorted(np.where(row > 0.5)[0].tolist()))


def build_seen_unseen_sets(
    seen_combinations: Optional[Sequence] = None,
    unseen_combinations: Optional[Sequence] = None,
    class_names: Optional[Sequence[str]] = None,
) -> Tuple[Set[Tuple[int, ...]], Set[Tuple[int, ...]]]:
    """Build sets of combination index-tuples.

    Accepts either index lists ``[0, 3]`` or name lists ``["DFTJ", "AJ"]``
    when ``class_names`` is provided.
    """
    def _normalize(combos) -> Set[Tuple[int, ...]]:
        out: Set[Tuple[int, ...]] = set()
        if not combos:
            return out
        for c in combos:
            if c is None:
                continue
            if isinstance(c, (list, tuple)) and len(c) == 0:
                continue
            items = list(c) if isinstance(c, (list, tuple)) else [c]
            # name → index
            if items and isinstance(items[0], str):
                if class_names is None:
                    continue
                name_to_idx = {n: i for i, n in enumerate(class_names)}
                idxs = sorted(name_to_idx[x] for x in items if x in name_to_idx)
            else:
                idxs = sorted(int(x) for x in items)
            if idxs:
                out.add(tuple(idxs))
        return out

    return _normalize(seen_combinations), _normalize(unseen_combinations)


def _partial_match_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction where true⊆pred or pred⊆true (multi-hot sets)."""
    n = y_true.shape[0]
    if n == 0:
        return 0.0
    correct = 0
    for i in range(n):
        t = set(np.where(y_true[i] > 0.5)[0].tolist())
        p = set(np.where(y_pred[i] > 0.5)[0].tolist())
        if t.issubset(p) or p.issubset(t):
            correct += 1
    return correct / n


def _prob_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Macro/micro ROC-AUC and average precision when probabilities exist."""
    out: Dict[str, float] = {}
    if y_prob is None or y_true.size == 0:
        return out
    y_prob = _to_numpy(y_prob).astype(np.float64)
    y_true = y_true.astype(np.float64)
    if y_prob.shape != y_true.shape:
        return out

    # Micro
    try:
        out["auc_micro"] = float(roc_auc_score(y_true.ravel(), y_prob.ravel()))
    except ValueError:
        out["auc_micro"] = float("nan")
    try:
        out["ap_micro"] = float(average_precision_score(y_true.ravel(), y_prob.ravel()))
    except ValueError:
        out["ap_micro"] = float("nan")

    # Macro (skip classes with no positive or no negative)
    aucs, aps = [], []
    for c in range(y_true.shape[1]):
        yt, yp = y_true[:, c], y_prob[:, c]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        try:
            aucs.append(roc_auc_score(yt, yp))
        except ValueError:
            pass
        try:
            aps.append(average_precision_score(yt, yp))
        except ValueError:
            pass
    out["auc_macro"] = float(np.mean(aucs)) if aucs else float("nan")
    out["ap_macro"] = float(np.mean(aps)) if aps else float("nan")
    return out


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None:
        return None
    return obj


# ---------------------------------------------------------------------------
# Core bundle
# ---------------------------------------------------------------------------

def compute_metrics_bundle(
    y_true,
    y_pred,
    class_names: Sequence[str],
    y_prob=None,
    seen_set: Optional[Set[Tuple[int, ...]]] = None,
    unseen_set: Optional[Set[Tuple[int, ...]]] = None,
    seen_combinations: Optional[Sequence] = None,
    unseen_combinations: Optional[Sequence] = None,
    dual_write: bool = True,
) -> Dict[str, Any]:
    """Compute a unified MetricsBundle from multi-hot labels/preds.

    Args:
        y_true, y_pred: [N, C] multi-hot
        class_names: length C
        y_prob: optional [N, C] probabilities for AUC/AP
        seen_set / unseen_set: optional prebuilt sets of index-tuples
        seen_combinations / unseen_combinations: name or index lists (if sets not given)
        dual_write: if True, also write combination_accuracy = subset_accuracy
    """
    y_true = _to_numpy(y_true).astype(np.float32)
    y_pred = _to_numpy(y_pred).astype(np.float32)
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)

    n, c = y_true.shape
    names = list(class_names) if class_names else [f"Class_{i}" for i in range(c)]

    if seen_set is None or unseen_set is None:
        s, u = build_seen_unseen_sets(seen_combinations, unseen_combinations, names)
        seen_set = seen_set if seen_set is not None else s
        unseen_set = unseen_set if unseen_set is not None else u
    seen_set = seen_set or set()
    unseen_set = unseen_set or set()

    if n == 0:
        empty = {
            "num_samples": 0,
            "subset_accuracy": 0.0,
            "partial_match_accuracy": 0.0,
            "seen_accuracy": 0.0,
            "seen_samples": 0,
            "unseen_accuracy": 0.0,
            "unseen_samples": 0,
            "other_accuracy": 0.0,
            "other_samples": 0,
            "harmonic_mean": 0.0,
            "f1_macro": 0.0,
            "f1_micro": 0.0,
            "f1_samples": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "precision_samples": 0.0,
            "recall_samples": 0.0,
            "per_class": {nm: {"f1": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}
                          for nm in names},
        }
        if dual_write:
            empty["combination_accuracy"] = 0.0
        return empty

    # Exact match (subset accuracy)
    subset_acc = float(accuracy_score(y_true, y_pred))
    partial_acc = float(_partial_match_accuracy(y_true, y_pred))

    # Seen / unseen / other by true combination
    seen_c = seen_t = unseen_c = unseen_t = other_c = other_t = 0
    has_split = bool(seen_set or unseen_set)
    if has_split:
        for i in range(n):
            true_comb = _combo_key_from_row(y_true[i])
            pred_comb = _combo_key_from_row(y_pred[i])
            ok = true_comb == pred_comb
            if true_comb in seen_set:
                seen_t += 1
                if ok:
                    seen_c += 1
            elif true_comb in unseen_set:
                unseen_t += 1
                if ok:
                    unseen_c += 1
            else:
                other_t += 1
                if ok:
                    other_c += 1

    seen_acc = seen_c / seen_t if seen_t > 0 else 0.0
    unseen_acc = unseen_c / unseen_t if unseen_t > 0 else 0.0
    other_acc = other_c / other_t if other_t > 0 else 0.0
    if seen_acc + unseen_acc > 0:
        hm = 2 * seen_acc * unseen_acc / (seen_acc + unseen_acc)
    else:
        hm = 0.0

    metrics: Dict[str, Any] = {
        "num_samples": int(n),
        "subset_accuracy": subset_acc,
        "partial_match_accuracy": partial_acc,
        "seen_accuracy": float(seen_acc),
        "seen_samples": int(seen_t),
        "unseen_accuracy": float(unseen_acc),
        "unseen_samples": int(unseen_t),
        "other_accuracy": float(other_acc),
        "other_samples": int(other_t),
        "harmonic_mean": float(hm),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_samples": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_samples": float(precision_score(y_true, y_pred, average="samples", zero_division=0)),
        "recall_samples": float(recall_score(y_true, y_pred, average="samples", zero_division=0)),
    }
    if dual_write:
        metrics["combination_accuracy"] = subset_acc  # legacy alias

    per_class = {}
    for i, nm in enumerate(names[:c]):
        support = int(y_true[:, i].sum())
        per_class[nm] = {
            "f1": float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "precision": float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "recall": float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "support": support,
        }
    metrics["per_class"] = per_class

    # Probability-based (dict; plots remain elsewhere)
    if y_prob is not None:
        metrics.update(_prob_metrics(y_true, y_prob))

    return metrics


def compute_subset_bundles(
    y_true,
    y_pred,
    class_names: Sequence[str],
    seen_set: Set[Tuple[int, ...]],
    unseen_set: Set[Tuple[int, ...]],
    y_prob=None,
    dual_write: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Global + seen-subset + unseen-subset MetricsBundles (by_combination)."""
    y_true = _to_numpy(y_true).astype(np.float32)
    y_pred = _to_numpy(y_pred).astype(np.float32)
    y_prob_np = _to_numpy(y_prob) if y_prob is not None else None

    global_m = compute_metrics_bundle(
        y_true, y_pred, class_names, y_prob=y_prob_np,
        seen_set=seen_set, unseen_set=unseen_set, dual_write=dual_write,
    )

    seen_mask, unseen_mask = [], []
    for i in range(len(y_true)):
        key = _combo_key_from_row(y_true[i])
        seen_mask.append(key in seen_set)
        unseen_mask.append(key in unseen_set)
    seen_mask = np.asarray(seen_mask, dtype=bool)
    unseen_mask = np.asarray(unseen_mask, dtype=bool)

    def _slice(mask):
        if mask.sum() == 0:
            return compute_metrics_bundle(
                np.zeros((0, y_true.shape[1])), np.zeros((0, y_true.shape[1])),
                class_names, dual_write=dual_write,
            )
        yp = y_prob_np[mask] if y_prob_np is not None else None
        return compute_metrics_bundle(
            y_true[mask], y_pred[mask], class_names, y_prob=yp,
            seen_set=seen_set, unseen_set=unseen_set, dual_write=dual_write,
        )

    return {
        "global": global_m,
        "seen": _slice(seen_mask),
        "unseen": _slice(unseen_mask),
    }


# ---------------------------------------------------------------------------
# Report / JSON
# ---------------------------------------------------------------------------

def format_metrics_report(
    metrics: Dict[str, Any],
    title: str = "Evaluation Results",
    indent: str = "  ",
) -> str:
    """Human-readable multi-line report (does not print nested subset dicts)."""
    lines = [
        "=" * 60,
        f"  {title}",
        "=" * 60,
    ]
    n = metrics.get("num_samples", "?")
    lines.append(f"{indent}num_samples              : {n}")

    # Core match
    for k in ("subset_accuracy", "partial_match_accuracy", "combination_accuracy"):
        if k in metrics and isinstance(metrics[k], (int, float)):
            lines.append(f"{indent}{k:24s}: {metrics[k]:.4f}")

    # CZSL
    if "seen_accuracy" in metrics:
        lines.append(
            f"{indent}{'seen_accuracy':24s}: {metrics['seen_accuracy']:.4f} "
            f"({metrics.get('seen_samples', 0)} samples)"
        )
        lines.append(
            f"{indent}{'unseen_accuracy':24s}: {metrics['unseen_accuracy']:.4f} "
            f"({metrics.get('unseen_samples', 0)} samples)"
        )
        if metrics.get("other_samples", 0):
            lines.append(
                f"{indent}{'other_accuracy':24s}: {metrics['other_accuracy']:.4f} "
                f"({metrics.get('other_samples', 0)} samples)"
            )
        lines.append(f"{indent}{'harmonic_mean':24s}: {metrics.get('harmonic_mean', 0):.4f}")

    lines.append(f"{indent}{'-' * 40}")
    for k in (
        "f1_macro", "f1_micro", "f1_samples",
        "precision_macro", "recall_macro",
        "precision_samples", "recall_samples",
        "auc_macro", "auc_micro", "ap_macro", "ap_micro",
    ):
        if k in metrics and isinstance(metrics[k], (int, float)):
            v = metrics[k]
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                lines.append(f"{indent}{k:24s}: nan")
            else:
                lines.append(f"{indent}{k:24s}: {float(v):.4f}")

    pc = metrics.get("per_class")
    if isinstance(pc, dict) and pc:
        lines.append(f"\n{indent}Per-class:")
        lines.append(f"{indent}  {'Class':10s} {'F1':>8} {'P':>8} {'R':>8} {'Sup':>6}")
        for name, vals in pc.items():
            if not isinstance(vals, dict):
                # legacy name→f1 float
                lines.append(f"{indent}  {name:10s} {float(vals):>8.4f}")
                continue
            lines.append(
                f"{indent}  {name:10s} {vals.get('f1', 0):>8.4f} "
                f"{vals.get('precision', 0):>8.4f} {vals.get('recall', 0):>8.4f} "
                f"{vals.get('support', 0):>6d}"
            )

    lines.append("=" * 60)
    return "\n".join(lines)


def print_metrics_report(metrics: Dict[str, Any], title: str = "Evaluation Results") -> None:
    print(format_metrics_report(metrics, title=title))


def print_jnr_metrics_table(
    jnr_results: Dict[Any, Dict[str, Any]],
    title: str = "Evaluation Results by JNR Level",
) -> str:
    """Print and return a text table; expects each value is a MetricsBundle (rates)."""
    header = (
        f"{'JNR':>6} | {'SubsetAcc':>10} | {'F1_Macro':>10} | "
        f"{'Seen_Acc':>10} | {'Unseen_Acc':>10} | {'HM':>8} | {'N':>8}"
    )
    lines = ["", "=" * 90, title, "=" * 90, header, "-" * 90]
    for jnr in sorted(jnr_results.keys(), key=lambda x: (isinstance(x, str), x)):
        m = jnr_results[jnr]
        acc = m.get("subset_accuracy", m.get("combination_accuracy", 0.0))
        # Guard: if legacy count was stored, convert when > 1 and total known
        n = m.get("num_samples", m.get("total_samples", 0))
        if isinstance(acc, (int, float)) and acc > 1.0 and n:
            acc = acc / n
        lines.append(
            f"{jnr:>6} | {float(acc):>10.4f} | {m.get('f1_macro', 0):>10.4f} | "
            f"{m.get('seen_accuracy', 0):>10.4f} | {m.get('unseen_accuracy', 0):>10.4f} | "
            f"{m.get('harmonic_mean', 0):>8.4f} | {n:>8}"
        )
    lines.append("=" * 90)
    text = "\n".join(lines)
    print(text)
    return text


def save_metrics_json(
    metrics: Dict[str, Any],
    output_dir: Union[str, Path],
    filename: str = "metrics.json",
    mode: Optional[str] = None,
    split: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Save metrics JSON under output_dir.

    Layout::
        {
          "mode": "...",
          "split": "...",
          "results": { ... MetricsBundle or nested ... },
          "meta": { ... optional ... }
        }

    Returns absolute path of written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    payload: Dict[str, Any] = {"results": _json_safe(metrics)}
    if mode is not None:
        payload["mode"] = mode
    if split is not None:
        payload["split"] = split
    if extra_meta:
        payload["meta"] = _json_safe(extra_meta)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Metrics JSON saved to {path}")
    return str(path.resolve())
