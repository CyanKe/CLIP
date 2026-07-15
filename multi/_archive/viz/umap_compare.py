#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
umap_compare.py - 对比预训练 vs 训练后 CLIP 模型图像特征的 UMAP 可视化

Usage:
d:\Anaconda\envs\clip\python.exe -m multi.umap_compare --checkpoint checkpoints/multishape_vit_best__model.pt --dataset test --jnr 0:1:20 --save-html --skip-pretrained
    python multi/_archive/viz/umap_compare.py --checkpoint checkpoints/czsl_best_model.pt --dataset test --jnr 10
    python multi/_archive/viz/umap_compare.py --checkpoint checkpoints/czsl_best_model.pt --dataset test --jnr 0:5:20
    python multi/_archive/viz/umap_compare.py --checkpoint checkpoints/czsl_best_model.pt --dataset test --jnr 0:5:20 --model-type multishape_vit
    python multi/_archive/viz/umap_compare.py --checkpoint checkpoints/czsl_best_model.pt --dir D:/path/to/data --dataset test --jnr 0:1:10

输出: results/umap_pretrained_{dataset}_{jnr_tag}.png
      results/umap_trained_{dataset}_{jnr_tag}.png
"""
import math
import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import umap
import yaml

warnings.filterwarnings('ignore', category=UserWarning, module='umap')

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from multi.data import STFTDataset
from multi.model import create_czsl_model
import clip

# ============================================================
# Color map (from umap_stft.py)
# ============================================================
JAM_TYPE_COLORS = {
    'DFTJ': "#018df1", 'ISRJ': '#ff7f0e', 'SMSPJ': "#4f755c",
    'CIJ': '#d62728', 'CSJ': "#67bd75", 'MISRJ': "#002fff",
    'ISDJ': "#e101ff",'ISCJ': "#6200ff",
    'AJ': '#8c564b', 'BJ': '#e377c2', 'SJ': '#7f7f7f',
    'NCJ': '#bcbd22', 'NPJ': "#17c9cf", 'NFMJ': '#aec7e8',
    'NPMJ': '#ffbb78', 'NAMJ': '#98df8a', 'PJ': '#c5b0d5',
}


# ============================================================
# CLI helpers
# ============================================================

def parse_jnr_range(spec):
    """Parse JNR specification into list of integer values.
    "10" -> [10]; "0 5 10" -> [0,5,10]; "0:5:20" -> [0,5,10,15,20]; "0:20" -> [0,1,...,20]
    """
    spec = spec.strip()
    if re.match(r'^-?\d+:-?\d+:-?\d+$', spec):
        start, step, end = (int(x) for x in spec.split(':'))
        return list(range(start, end + 1, step))
    if re.match(r'^-?\d+:-?\d+$', spec):
        start, end = (int(x) for x in spec.split(':'))
        step = 1 if start <= end else -1
        return list(range(start, end + step, step))
    if ' ' in spec:
        return [int(x) for x in spec.split()]
    return [int(spec)]


def normalize_label(raw_label):
    """Normalize C&IJ -> CIJ for color mapping consistency."""
    return raw_label.replace('C&IJ', 'CIJ')


# ============================================================
# Data loading
# ============================================================

def collate_for_features(batch):
    """Simple collate: stack images, collect labels and metadata. Optionally raw_mag."""
    images = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    metadata = [item[2] for item in batch]
    if len(batch[0]) >= 4:  # STFTDataset with return_raw_mag: (stft, label, meta, raw_mag)
        raw_mags = torch.stack([item[3] for item in batch])
        return images, labels, metadata, raw_mags
    return images, labels, metadata


def load_data(base_path, dataset, jnr_list, stft_suffix, class_names,
              image_size, batch_size, num_workers):
    """Load STFT data across multiple JNR levels, return DataLoader."""
    datasets = []

    for jnr in jnr_list:
        jnr_str = f"+{jnr}" if jnr >= 0 else str(jnr)
        data_folder = os.path.join(base_path, f"JNR_{jnr_str}")
        stft_file = os.path.join(data_folder, f"{dataset}_{stft_suffix}.mat")
        metadata_file = os.path.join(data_folder, f"{dataset}_echo_metadata.json")

        if not os.path.exists(stft_file) or not os.path.exists(metadata_file):
            print(f"  [SKIP] JNR={jnr_str}: data not found")
            continue

        ds = STFTDataset(
            stft_file=stft_file,
            metadata_file=metadata_file,
            class_names=class_names,
            image_size=image_size,
            apply_clip_norm=True,
            normalize_mode='per_sample',
            normalize_method='p99',
            return_raw_mag=True,
        )
        datasets.append(ds)
        print(f"  JNR={jnr_str}: {len(ds)} samples")

    if not datasets:
        raise RuntimeError("No valid data found in the specified JNR range.")

    combined = ConcatDataset(datasets)
    print(f"  Total: {len(combined)} samples")

    loader = DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_for_features,
    )
    return loader


# ============================================================
# Feature extraction
# ============================================================

@torch.no_grad()
def extract_features(model, data_loader, device, desc="Extracting",
                     stft_dir=None):
    """Extract L2-normalized image features, string labels, JNR values, indices.

    If stft_dir is set, also saves per-sample STFT preview PNGs and returns paths.
    """
    model.eval()
    all_features = []
    all_label_strings = []
    all_jnr = []
    all_indices = []
    all_stft_raw = []
    all_stft_clip = []
    global_idx = 0

    for batch in tqdm(data_loader, desc=desc):
        has_raw = len(batch) >= 4 and batch[3] is not None
        if has_raw:
            images, _labels, metadata, raw_mags = batch
        else:
            images, _labels, metadata = batch
            raw_mags = None

        images = images.to(device)
        if images.shape[-1] != 224:
            images = F.interpolate(images, size=(224, 224),
                                   mode='bilinear', align_corners=False)

        features = model.encode_image(images)
        features = F.normalize(features, dim=-1)
        all_features.append(features.cpu().numpy())

        for i, meta in enumerate(metadata):
            jam_types = meta.get('jam_types', [])
            if isinstance(jam_types, list):
                label_str = '+'.join(sorted([normalize_label(t) for t in jam_types]))
            else:
                label_str = normalize_label(str(jam_types))
            all_label_strings.append(label_str)
            all_jnr.append(meta.get('JNR', '?'))
            all_indices.append(global_idx)

            if stft_dir and has_raw:
                os.makedirs(stft_dir, exist_ok=True)
                raw_fname = f'sample_{global_idx:06d}_raw.png'
                clip_fname = f'sample_{global_idx:06d}_clip.png'

                # Raw magnitude
                raw_img = raw_mags[i].cpu().numpy()
                plt.imsave(os.path.join(stft_dir, raw_fname),
                           raw_img, cmap='inferno')

                # CLIP input: average of 3 channels
                clip_arr = images[i].cpu().numpy()
                clip_avg = clip_arr.mean(axis=0)
                clip_avg = (clip_avg - clip_avg.min()) / \
                           (clip_avg.max() - clip_avg.min() + 1e-8)
                plt.imsave(os.path.join(stft_dir, clip_fname),
                           clip_avg, cmap='inferno')

                # Store relative paths for JSON (from results/ dir)
                all_stft_raw.append(f'stft_preview/{raw_fname}')
                all_stft_clip.append(f'stft_preview/{clip_fname}')
            elif stft_dir:
                all_stft_raw.append(None)
                all_stft_clip.append(None)

            global_idx += 1

    result = (np.concatenate(all_features, axis=0),
              all_label_strings, all_jnr, all_indices)
    if stft_dir:
        result = result + (all_stft_raw, all_stft_clip)
    return result


# ============================================================
# Model loading
# ============================================================

def load_pretrained_model(clip_model_name, device):
    """Load pretrained CLIP model (visual encoder only for feature extraction)."""
    model, _ = clip.load(clip_model_name, device=device)
    model.eval()
    print(f"Loaded pretrained CLIP model: {clip_model_name}")
    return model


def load_trained_model(config, checkpoint_path, device):
    """Load trained model from checkpoint. Supports vit and multishape_vit backbones.

    Uses the checkpoint's saved config (if available) for architecture-critical
    parameters like patch_sizes, so the model shape always matches the weights.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Prefer checkpoint's own config for architecture params
    ckpt_config = checkpoint.get('config', None)
    if ckpt_config is not None:
        config = ckpt_config
        print("Using checkpoint's saved config for model architecture")
    else:
        print("Checkpoint has no saved config — using current config file")

    model_type = config.get('model', {}).get('backbone', 'vit')

    if model_type == 'multishape_vit':
        from multi.experiments.rectangular_patch_vit import create_multi_shape_patch_model
        model = create_multi_shape_patch_model(config, device=str(device))
    else:
        model = create_czsl_model(config, device=str(device))

    model_state = model.state_dict()

    filtered_state_dict = {}
    mismatched_keys = []
    for key, value in state_dict.items():
        if key in model_state:
            if model_state[key].shape != value.shape:
                mismatched_keys.append(
                    f"{key}: checkpoint {value.shape} vs model {model_state[key].shape}"
                )
            else:
                filtered_state_dict[key] = value

    if mismatched_keys:
        print(f"Warning: Skipping {len(mismatched_keys)} mismatched layers:")
        for key in mismatched_keys[:10]:
            print(f"  - {key}")
        if len(mismatched_keys) > 10:
            print(f"  ... and {len(mismatched_keys) - 10} more")

    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Matched {len(filtered_state_dict)} / {len(state_dict)} parameter keys")
    return model


# ============================================================
# Visualization
# ============================================================

def plot_umap(embedding, labels, title_info, save_path=None, highlight=None):
    """Plot UMAP scatter: left=single colored+composite gray, right=all colored.

    Args:
        highlight: optional jamming type (e.g. 'DFTJ'). When set, only composites
                   containing this type are colored; other composites remain gray.
                   Single types are always colored regardless.
    """
    single_cats = sorted(
        {l for l in labels if '+' not in l},
        key=lambda x: list(JAM_TYPE_COLORS.keys()).index(x)
        if x in JAM_TYPE_COLORS else 999,
    )
    composite_cats = sorted({l for l in labels if '+' in l})
    n_single = len(single_cats)
    n_composite = len(composite_cats)
    cmap = plt.cm.tab20

    # Determine which composites to highlight
    if highlight:
        highlighted_composites = sorted(
            {l for l in composite_cats if highlight in l.split('+')}
        )
        gray_composites = sorted(
            {l for l in composite_cats if highlight not in l.split('+')}
        )
    else:
        highlighted_composites = composite_cats
        gray_composites = []

    _, axes = plt.subplots(1, 2, figsize=(22, 10))

    # --- Left: single colored, composites: highlighted=colored, rest=gray ---
    ax = axes[0]
    ax.set_facecolor('#f5f5f5')
    # Gray composites (not highlighted)
    if gray_composites:
        gray_mask = np.array([l in gray_composites for l in labels])
        if gray_mask.any():
            ax.scatter(embedding[gray_mask, 0], embedding[gray_mask, 1],
                       c='#dddddd', s=6, alpha=0.35, marker='o',
                       label=f'Other composite ({gray_mask.sum()})', edgecolors='none')
    # Highlighted composites — use the highlight type's color
    for hcat in highlighted_composites:
        mask = np.array([l == hcat for l in labels])
        if mask.any():
            color = JAM_TYPE_COLORS.get(highlight, '#999999')
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[color], s=14, alpha=0.7, marker='s',
                       label=f'{hcat} ({mask.sum()})',
                       edgecolors='white', linewidth=0.3)
    # Single types
    for cat in single_cats:
        mask = np.array([l == cat for l in labels])
        if mask.any():
            color = JAM_TYPE_COLORS.get(cat, '#999999')
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[color], s=18, alpha=0.85, label=f'{cat} ({mask.sum()})',
                       edgecolors='white', linewidth=0.3)
    n_hl = len(highlighted_composites)
    n_gray = len(gray_composites)
    title_left = f'UMAP ({n_single} single + {n_hl} highlighted composite'
    if n_gray:
        title_left += f', {n_gray} composite gray'
    title_left += ')'
    ax.set_title(title_left, fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(loc='upper left', fontsize=7, ncol=2, framealpha=0.9,
              bbox_to_anchor=(1.0, 1.0))

    # --- Right: single + highlighted composites colored, rest gray ---
    ax = axes[1]
    ax.set_facecolor('#f5f5f5')
    # Gray composites (not highlighted)
    if gray_composites:
        gray_mask = np.array([l in gray_composites for l in labels])
        if gray_mask.any():
            ax.scatter(embedding[gray_mask, 0], embedding[gray_mask, 1],
                       c='#dddddd', s=6, alpha=0.35, marker='o',
                       label=f'Other composite ({gray_mask.sum()})', edgecolors='none')
    # Highlighted composites — individually colored
    for gi, cat in enumerate(highlighted_composites):
        mask = np.array([l == cat for l in labels])
        if not mask.any():
            continue
        color = cmap(gi / max(len(highlighted_composites), 1))
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[color], s=14, alpha=0.8, label=f'{cat} ({mask.sum()})',
                   edgecolors='white', linewidth=0.2)
    # Single types
    for cat in single_cats:
        mask = np.array([l == cat for l in labels])
        if not mask.any():
            continue
        color = JAM_TYPE_COLORS.get(cat, '#999999')
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[color], s=14, alpha=0.8, label=f'{cat} ({mask.sum()})',
                   edgecolors='white', linewidth=0.2)
    n_colored = n_single + n_hl
    title_right = f'UMAP ({n_colored} colored'
    if n_gray:
        title_right += f' + {n_gray} composite gray'
    title_right += f' = {n_colored + n_gray} classes)'
    ax.set_title(title_right, fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    total_all = len(single_cats) + len(composite_cats)
    ncol = 3 if total_all > 30 else 2
    ax.legend(loc='upper left', fontsize=5, ncol=ncol, framealpha=0.85,
              bbox_to_anchor=(1.0, 1.0))

    plt.suptitle(title_info, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    else:
        plt.show()
    plt.close()


# ============================================================
# JSON data export
# ============================================================

def _json_safe(obj):
    """Convert numpy arrays and Python objects to JSON-safe types."""
    if isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.floating):
            obj = np.where(np.isnan(obj) | np.isinf(obj), None, obj)
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    return obj


def build_umap_data(pretrained_embedding, trained_embedding, all_labels, all_jnr,
                    all_indices, clip_model_name, dataset, jnr_tag, highlight,
                    stft_raw_paths=None, stft_clip_paths=None):
    """Build a JSON-serializable data dict from UMAP embeddings."""
    single_cats = sorted(
        {l for l in all_labels if '+' not in l},
        key=lambda x: list(JAM_TYPE_COLORS.keys()).index(x)
        if x in JAM_TYPE_COLORS else 999,
    )
    composite_cats = sorted({l for l in all_labels if '+' in l})

    if highlight:
        composite_cats = sorted(
            {l for l in composite_cats if highlight in l.split('+')}
        )

    def build_dataset(embedding):
        if embedding is None:
            return None
        traces_data = {}
        for cat in single_cats + composite_cats:
            mask = np.array([l == cat for l in all_labels])
            if mask.any():
                idx = np.where(mask)[0]
                traces_data[cat] = {
                    'x': _json_safe(embedding[mask, 0]),
                    'y': _json_safe(embedding[mask, 1]),
                    'jnr': _json_safe([all_jnr[i] for i in idx]),
                    'idx': _json_safe([all_indices[i] for i in idx]),
                }
                if stft_raw_paths is not None:
                    traces_data[cat]['stft_raw'] = _json_safe(
                        [stft_raw_paths[i] for i in idx])
                if stft_clip_paths is not None:
                    traces_data[cat]['stft_clip'] = _json_safe(
                        [stft_clip_paths[i] for i in idx])
        return traces_data

    return {
        'pretrained': build_dataset(pretrained_embedding),
        'trained': build_dataset(trained_embedding),
        'categories': {
            'single': single_cats,
            'composite': composite_cats,
        },
        'colors': {k: v for k, v in JAM_TYPE_COLORS.items() if k in single_cats},
        'info': {
            'clip_model': clip_model_name,
            'dataset': dataset,
            'jnr_tag': jnr_tag,
            'n_samples': len(all_labels),
            'highlight': highlight,
        },
    }


def save_umap_data_json(data_dict, output_path):
    """Save UMAP data dict as a standalone JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'  JSON saved: {output_path} ({size_mb:.1f} MB)')


# ============================================================
# Interactive HTML generation
# ============================================================

def build_html(data_filename, clip_model_name, dataset, jnr_tag,
               n_samples, output_path):
    """Generate an HTML file that loads UMAP data from an external JSON file.

    Features:
      - Auto-loads the default JSON on open (via sync XHR, works with file://)
      - File picker to load a different JSON at any time
      - Data-driven tabs: if JSON has both pretrained & trained, tabs appear
    """
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CLIP Feature UMAP</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
:root {{
  --bg: #1a1a2e; --bg2: #16213e; --border: #0f3460; --accent: #e94560;
  --text: #eee; --text2: #ccc; --text3: #888; --grid: #2a2a4a; --plot-bg: #1a1a2e;
}}
:root.light {{
  --bg: #f0f2f5; --bg2: #fff; --border: #d0d5dd; --accent: #d6336c;
  --text: #1a1a2e; --text2: #555; --text3: #999; --grid: #e0e0e0; --plot-bg: #fff;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; }}
.header {{ background: var(--bg2); padding: 8px 20px; display: flex; align-items: center; gap: 10px;
           border-bottom: 1px solid var(--border); flex-shrink: 0; min-height: 44px; }}
.header h2 {{ font-size: 15px; font-weight: 600; white-space: nowrap; }}
.header .spacer {{ flex: 1; }}
#fileInput {{ display: none; }}
.load-btn {{ padding: 6px 14px; border: 1px solid var(--accent); background: transparent; color: var(--accent);
            cursor: pointer; border-radius: 4px; font-size: 12px; white-space: nowrap; }}
.load-btn:hover {{ background: var(--accent); color: #fff; }}
.theme-btn {{ padding: 6px 10px; border: 1px solid var(--border); background: transparent; color: var(--text2);
             cursor: pointer; border-radius: 4px; font-size: 14px; line-height: 1; }}
.theme-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
#fileName {{ font-size: 11px; color: var(--text3); max-width: 300px; overflow: hidden;
            text-overflow: ellipsis; white-space: nowrap; }}
.tab-bar {{ display: flex; gap: 4px; margin-left: 8px; }}
.tab-btn {{ padding: 6px 16px; border: 1px solid var(--border); background: var(--bg); color: var(--text2);
           cursor: pointer; border-radius: 6px 6px 0 0; font-size: 12px; transition: all 0.2s; display: none; }}
.tab-btn.active {{ background: var(--border); color: #fff; border-color: var(--accent); }}
.tab-btn:hover:not(.active) {{ color: var(--text); border-color: var(--text3); }}
#infoTag {{ font-size: 11px; color: var(--text3); white-space: nowrap; }}
.main {{ display: flex; flex: 1; overflow: hidden; }}
.sidebar {{ width: 260px; background: var(--bg2); border-right: 1px solid var(--border);
           overflow-y: auto; padding: 12px; flex-shrink: 0; }}
.sidebar h3 {{ font-size: 13px; color: var(--accent); margin: 12px 0 6px 0; text-transform: uppercase;
               letter-spacing: 0.5px; }}
.sidebar h3:first-child {{ margin-top: 0; }}
.sidebar label {{ display: flex; align-items: center; padding: 4px 8px; font-size: 12px;
                  cursor: pointer; border-radius: 4px; transition: background 0.15s; gap: 6px; }}
.sidebar label:hover {{ background: var(--bg); }}
.sidebar input[type="checkbox"] {{ accent-color: var(--accent); }}
.color-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
.btn-row {{ display: flex; gap: 6px; margin: 8px 0; }}
.btn-row button {{ flex: 1; padding: 5px 0; font-size: 11px; border: 1px solid var(--border);
                   background: var(--bg); color: var(--text2); border-radius: 4px; cursor: pointer; }}
.btn-row button:hover {{ background: var(--border); color: var(--text); }}
.plot-area {{ flex: 1; position: relative; }}
#plot {{ width: 100%; height: 100%; }}
.info-bar {{ padding: 4px 12px; font-size: 11px; color: var(--text3); background: var(--bg2);
            border-top: 1px solid var(--border); flex-shrink: 0; }}
#loading {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
           font-size: 18px; color: var(--text3); z-index: 10; white-space: pre-line; text-align: center; }}
/* Modal */
.modal-overlay {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                  background: rgba(0,0,0,0.7); z-index: 1000; justify-content: center; align-items: center; }}
.modal-overlay.show {{ display: flex; }}
.modal-box {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 8px;
             padding: 16px; max-width: 900px; width: 95%; max-height: 90vh; overflow-y: auto; }}
.modal-box h3 {{ font-size: 14px; margin-bottom: 12px; color: var(--text); }}
.modal-close {{ float: right; background: none; border: none; color: var(--text2); font-size: 20px;
               cursor: pointer; line-height: 1; }}
.modal-close:hover {{ color: var(--accent); }}
.modal-imgs {{ display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; }}
.modal-img-wrap {{ text-align: center; }}
.modal-img-wrap img {{ max-width: 400px; max-height: 400px; border-radius: 4px;
                       border: 1px solid var(--border); }}
.modal-img-wrap p {{ font-size: 11px; color: var(--text3); margin-top: 4px; }}
.modal-actions {{ display: flex; gap: 8px; justify-content: center; margin-top: 12px; }}
.modal-actions button {{ padding: 6px 16px; border: 1px solid var(--accent); background: transparent;
                         color: var(--accent); border-radius: 4px; cursor: pointer; font-size: 12px; }}
.modal-actions button:hover {{ background: var(--accent); color: #fff; }}
</style>
</head>
<body>

<div class="header">
  <h2>CLIP Feature UMAP</h2>
  <input type="file" id="fileInput" accept=".json" onchange="window._onFilePicked(event)">
  <button class="load-btn" onclick="document.getElementById('fileInput').click()">Load JSON</button>
  <button class="theme-btn" id="themeBtn" onclick="window._toggleTheme()" title="Toggle dark/light">&#9788;</button>
  <span id="fileName">{data_filename}</span>
  <span class="spacer"></span>
  <div class="tab-bar">
    <button class="tab-btn" data-tab="pretrained" onclick="window.switchTab('pretrained')">Pretrained</button>
    <button class="tab-btn" data-tab="trained" onclick="window.switchTab('trained')">Trained</button>
  </div>
  <span id="infoTag"></span>
</div>

<div class="main">
  <div class="sidebar" id="sidebar"></div>
  <div class="plot-area">
    <div id="loading">Drag & drop a JSON file or click "Load JSON"</div>
    <div id="plot"></div>
  </div>
</div>

<div class="info-bar" id="infoBar">Ready — waiting for data</div>

<div class="modal-overlay" id="modalOverlay" onclick="window._closeModal(event)">
  <div class="modal-box" onclick="event.stopPropagation()">
    <button class="modal-close" onclick="window._closeModal()">&times;</button>
    <h3 id="modalTitle"></h3>
    <div class="modal-imgs" id="modalImgs">
      <div class="modal-img-wrap" id="modalWrapRaw" style="display:none">
        <img id="modalImgRaw" src="" alt="">
        <p>Raw |STFT| Magnitude</p>
        <button onclick="window._saveImg('raw')">Save</button>
      </div>
      <div class="modal-img-wrap" id="modalWrapClip" style="display:none">
        <img id="modalImgClip" src="" alt="">
        <p>CLIP Input (channel avg)</p>
        <button onclick="window._saveImg('clip')">Save</button>
      </div>
      <p id="modalNoStft" style="display:none;color:var(--text3);font-size:13px">
        STFT preview not available for this sample.<br>
        Re-run with --save-html to generate STFT previews.
      </p>
    </div>
  </div>
</div>

<script>
(function() {{
  var DATA = null, SINGLE_COLORS, SINGLE, COMPOSITE, activeTab = null;
  var TAB20 = [
    '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2',
    '#7f7f7f','#bcbd22','#17becf','#aec7e8','#ffbb78','#98df8a','#c5b0d5',
    '#f7b6d2','#c7c7c7','#dbdb8d','#9edae5','#c49c94','#f0e442'
  ];

  // ============================================================
  // Data loading
  // ============================================================
  function setData(jsonObj, filename) {{
    DATA = jsonObj;
    SINGLE_COLORS = DATA.colors;
    SINGLE = DATA.categories.single;
    COMPOSITE = DATA.categories.composite;
    document.getElementById('fileName').textContent = filename || 'loaded';
    // Show/hide tabs based on what models are available
    var hasPre = !!DATA.pretrained, hasTra = !!DATA.trained;
    document.querySelectorAll('.tab-btn').forEach(function(b) {{
      b.style.display = (b.dataset.tab === 'pretrained' && hasPre) ||
                        (b.dataset.tab === 'trained' && hasTra) ? '' : 'none';
    }});
    if (hasPre && hasTra) {{
      activeTab = activeTab || 'pretrained';
    }} else if (hasPre) {{
      activeTab = 'pretrained';
    }} else if (hasTra) {{
      activeTab = 'trained';
    }}
    document.querySelectorAll('.tab-btn').forEach(function(b) {{
      b.classList.toggle('active', b.dataset.tab === activeTab);
    }});
    var info = DATA.info;
    document.getElementById('infoTag').textContent =
      (info.clip_model||'') + ' | ' + (info.dataset||'') + ' | JNR=' + (info.jnr_tag||'') +
      ' | ' + (info.n_samples||0) + ' samples';
    buildSidebar();
    renderPlot();
  }}

  function loadFromXHR(url) {{
    var xhr = new XMLHttpRequest();
    xhr.open('GET', url, false);
    xhr.overrideMimeType('application/json');
    try {{
      xhr.send(null);
      if (xhr.status === 0 || xhr.status === 200) {{
        setData(JSON.parse(xhr.responseText), url);
        return true;
      }}
    }} catch(e) {{}}
    return false;
  }}

  function loadFromFile(file) {{
    var reader = new FileReader();
    reader.onload = function(e) {{
      try {{
        setData(JSON.parse(e.target.result), file.name);
      }} catch(err) {{
        document.getElementById('loading').textContent = 'Parse error: ' + err.message;
      }}
    }};
    reader.readAsText(file);
  }}

  // ============================================================
  // Sidebar
  // ============================================================
  function buildSidebar() {{
    var sidebar = document.getElementById('sidebar');
    var html = '';

    html += '<div class="btn-row">'
         + '<button onclick="window._selectAll()">Select All</button>'
         + '<button onclick="window._deselectAll()">Deselect All</button>'
         + '</div>';

    html += '<h3>Single (' + SINGLE.length + ')</h3>';
    SINGLE.forEach(function(cat) {{
      var color = SINGLE_COLORS[cat] || '#999';
      html += '<label><span class="color-dot" style="background:' + color + '"></span>'
           + '<input type="checkbox" checked onchange="window._onFilter()" data-cat="' + cat + '">'
           + cat + '</label>';
    }});

    if (COMPOSITE.length > 0) {{
      html += '<h3>Composite (' + COMPOSITE.length + ')</h3>';
      COMPOSITE.forEach(function(cat, i) {{
        var color = TAB20[i % TAB20.length];
        html += '<label><span class="color-dot" style="background:' + color + '"></span>'
             + '<input type="checkbox" checked onchange="window._onFilter()" data-cat="' + cat + '">'
             + cat + '</label>';
      }});
    }}

    sidebar.innerHTML = html;
  }}

  function getVisibleMap() {{
    var map = {{}};
    document.querySelectorAll('#sidebar input[type="checkbox"]').forEach(function(cb) {{
      map[cb.dataset.cat] = cb.checked;
    }});
    return map;
  }}

  // ============================================================
  // Plotly traces
  // ============================================================
  function buildTraces(tab) {{
    var ds = DATA[tab];
    if (!ds) return [];
    var traces = [];
    var visibleMap = getVisibleMap();

    SINGLE.forEach(function(cat) {{
      if (!ds[cat]) return;
      var color = SINGLE_COLORS[cat] || '#999';
      var d = ds[cat];
      traces.push({{
        x: d.x, y: d.y,
        mode: 'markers', type: 'scatter', name: cat,
        marker: {{ color: color, size: 5, opacity: 0.85,
                   line: {{ color: 'white', width: 0.2 }} }},
        text: d.jnr.map(function(v, i) {{ return cat + '<br>JNR: ' + v + '<br>idx: ' + d.idx[i]; }}),
        hoverinfo: 'text', visible: visibleMap[cat] !== false, showlegend: true,
      }});
    }});

    COMPOSITE.forEach(function(cat, i) {{
      if (!ds[cat]) return;
      var color = TAB20[i % TAB20.length];
      var d = ds[cat];
      traces.push({{
        x: d.x, y: d.y,
        mode: 'markers', type: 'scatter', name: cat,
        marker: {{ color: color, size: 5, opacity: 0.8,
                   line: {{ color: 'white', width: 0.2 }} }},
        text: d.jnr.map(function(v, j) {{ return cat + '<br>JNR: ' + v + '<br>idx: ' + d.idx[j]; }}),
        hoverinfo: 'text', visible: visibleMap[cat] !== false, showlegend: true,
      }});
    }});

    return traces;
  }}

  function getLayout() {{
    var isLight = document.documentElement.classList.contains('light');
    return {{
      paper_bgcolor: isLight ? '#fff' : '#1a1a2e',
      plot_bgcolor: isLight ? '#fff' : '#1a1a2e',
      font: {{ color: isLight ? '#333' : '#ccc', size: 12 }},
      xaxis: {{ title: 'UMAP 1', gridcolor: isLight ? '#e0e0e0' : '#2a2a4a',
               zerolinecolor: isLight ? '#ccc' : '#444' }},
      yaxis: {{ title: 'UMAP 2', gridcolor: isLight ? '#e0e0e0' : '#2a2a4a',
               zerolinecolor: isLight ? '#ccc' : '#444' }},
      margin: {{ l: 50, r: 30, t: 10, b: 50 }},
      hovermode: 'closest',
      legend: {{ font: {{ size: 10 }}, itemsizing: 'constant',
                itemclick: false, itemdoubleclick: false }},
      dragmode: 'pan',
    }};
  }}

  var CONFIG = {{
    displayModeBar: true, modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    displaylogo: false, responsive: true,
  }};

  function renderPlot() {{
    if (!DATA || !activeTab) return;
    document.getElementById('loading').style.display = 'block';
    var traces = buildTraces(activeTab);
    Plotly.newPlot('plot', traces, getLayout(), CONFIG).then(function() {{
      document.getElementById('loading').style.display = 'none';
      updateInfoBar(traces);
      // Register click handler for STFT preview
      var plotEl = document.getElementById('plot');
      plotEl.removeAllListeners && plotEl.removeAllListeners('plotly_click');
      plotEl.on('plotly_click', function(ev) {{
        if (!ev || !ev.points || ev.points.length === 0) return;
        var pt = ev.points[0];
        var cat = pt.data.name;
        var pi = pt.pointIndex;
        var ds = DATA[activeTab];
        if (!ds || !ds[cat]) return;
        var d = ds[cat];
        var sidx = String(d.idx[pi]).padStart(6, '0');
        var rawPath = (d.stft_raw && d.stft_raw[pi]) || ('stft_preview/sample_' + sidx + '_raw.png');
        var clipPath = (d.stft_clip && d.stft_clip[pi]) || ('stft_preview/sample_' + sidx + '_clip.png');
        window._modalData = {{
          cat: cat, jnr: d.jnr[pi], idx: d.idx[pi],
          rawPath: rawPath, clipPath: clipPath,
        }};
        document.getElementById('modalTitle').textContent =
          cat + '  |  JNR=' + d.jnr[pi] + '  |  idx=' + d.idx[pi];
        document.getElementById('modalImgRaw').src = rawPath;
        document.getElementById('modalImgClip').src = clipPath;
        document.getElementById('modalWrapRaw').style.display = '';
        document.getElementById('modalWrapClip').style.display = '';
        document.getElementById('modalNoStft').style.display = 'none';
        document.getElementById('modalOverlay').classList.add('show');
      }});
    }}).catch(function(err) {{
      document.getElementById('loading').textContent = 'Plot error: ' + err.message;
    }});
  }}

  function updateInfoBar(traces) {{
    var visible = 0;
    traces.forEach(function(t) {{ if (t.visible !== false && t.x) visible += t.x.length; }});
    document.getElementById('infoBar').textContent =
      'Showing ' + visible + ' / ' + DATA.info.n_samples + ' samples  |  '
      + SINGLE.length + ' single + ' + COMPOSITE.length + ' composite  |  ' + (activeTab||'');
  }}

  // ============================================================
  // Interactions
  // ============================================================
  window._onFilter = function() {{
    if (!DATA) return;
    var traces = buildTraces(activeTab);
    Plotly.react('plot', traces, getLayout(), CONFIG).then(function() {{ updateInfoBar(traces); }});
  }};

  window.switchTab = function(tab) {{
    if (!DATA || !DATA[tab]) return;
    activeTab = tab;
    document.querySelectorAll('.tab-btn').forEach(function(b) {{
      b.classList.toggle('active', b.dataset.tab === tab);
    }});
    window._onFilter();
  }};

  window._onFilePicked = function(e) {{
    if (e.target.files.length > 0) loadFromFile(e.target.files[0]);
  }};

  window._selectAll = function() {{
    document.querySelectorAll('#sidebar input[type="checkbox"]').forEach(function(cb) {{ cb.checked = true; }});
    window._onFilter();
  }};

  window._deselectAll = function() {{
    document.querySelectorAll('#sidebar input[type="checkbox"]').forEach(function(cb) {{ cb.checked = false; }});
    window._onFilter();
  }};

  // ============================================================
  // Modal — STFT preview (shows pre-saved PNGs directly)
  // ============================================================

  window._closeModal = function(ev) {{
    if (ev && ev.target !== document.getElementById('modalOverlay')) return;
    document.getElementById('modalOverlay').classList.remove('show');
    window._modalData = null;
  }};

  window._saveImg = function(which) {{
    var md = window._modalData;
    if (!md) return;
    var a = document.createElement('a');
    a.href = which === 'raw' ? md.rawPath : md.clipPath;
    a.download = md.cat + '_JNR' + md.jnr + '_idx' + md.idx
               + (which === 'raw' ? '_raw.png' : '_clip.png');
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }};

  // ============================================================
  // Theme toggle
  // ============================================================
  window._toggleTheme = function() {{
    var root = document.documentElement;
    var isLight = root.classList.toggle('light');
    document.getElementById('themeBtn').innerHTML = isLight ? '&#9790;' : '&#9788;';
    localStorage.setItem('umap-theme', isLight ? 'light' : 'dark');
    if (DATA) renderPlot();
  }};
  (function() {{
    if (localStorage.getItem('umap-theme') === 'light') {{
      document.documentElement.classList.add('light');
      document.getElementById('themeBtn').innerHTML = '&#9790;';
    }}
  }})();

  // ============================================================
  // Drag & drop
  // ============================================================
  document.addEventListener('dragover', function(e) {{ e.preventDefault(); }});
  document.addEventListener('drop', function(e) {{
    e.preventDefault();
    if (e.dataTransfer.files.length > 0) loadFromFile(e.dataTransfer.files[0]);
  }});

  // ============================================================
  // Init — try to auto-load the default JSON
  // ============================================================
  if (!loadFromXHR('{data_filename}')) {{
    document.getElementById('loading').textContent =
      'Could not auto-load "{data_filename}".\\nDrag & drop a JSON file or click "Load JSON".';
  }}
}})();
</script>
</body>
</html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'  HTML saved: {output_path}')


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='UMAP comparison: pretrained vs trained CLIP image features'
    )
    parser.add_argument('--config', type=str, default='multi/config.yaml',
                        help='Path to config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--model-type', type=str, default=None,
                        choices=['clip', 'multishape_vit'],
                        help='Model backbone type (default: from config)')
    parser.add_argument('--dir', type=str, default=None,
                        help='Base data directory (overrides config data.base_path)')
    parser.add_argument('--dataset', type=str, default='test',
                        help='Dataset split: train / val / test')
    parser.add_argument('--jnr', type=str, default=None,
                        help='JNR spec: "10", "0:5:20", "0:20" (default: from config)')
    parser.add_argument('--n_neighbors', type=int, default=30,
                        help='UMAP n_neighbors')
    parser.add_argument('--min_dist', type=float, default=0.3,
                        help='UMAP min_dist')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for feature extraction')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for plots')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--highlight', type=str, default=None,
                        help='Highlight composites containing this jamming type '
                             '(e.g. DFTJ). Other composites remain gray.')
    parser.add_argument('--save-html', action='store_true', default=False,
                        help='Generate an interactive HTML file for browser viewing')
    parser.add_argument('--skip-pretrained', action='store_true', default=False,
                        help='Skip pretrained model feature extraction')
    parser.add_argument('--skip-trained', action='store_true', default=False,
                        help='Skip trained model feature extraction')
    parser.add_argument('--stft-dir', type=str, default=None,
                        help='STFT preview directory (default: results/stft_preview)')
    parser.add_argument('--no-stft', action='store_true', default=False,
                        help='Skip saving STFT preview PNGs')
    args = parser.parse_args()

    # ============================================================
    # 1. Load config
    # ============================================================
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config not found: {args.config}, using defaults")
        config = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ============================================================
    # 2. Parse data parameters
    # ============================================================
    data_config = config.get('data', {})
    base_path = args.dir or data_config.get('base_path', '')
    stft_suffix = data_config.get('stft_suffix', 'echo_stfts')
    num_workers = data_config.get('num_workers', 4)
    class_names = [cls['name'] for cls in config.get('jamming_classes', [])]
    clip_model_name = config.get('model', {}).get('clip_model', 'ViT-B/32')

    # Override backbone only if --model-type is explicitly given
    if args.model_type:
        if 'model' not in config:
            config['model'] = {}
        config['model']['backbone'] = args.model_type

    # JNR range
    if args.jnr:
        jnr_list = parse_jnr_range(args.jnr)
    else:
        jnr_start = data_config.get('jnr_start', 0)
        jnr_end = data_config.get('jnr_end', 20)
        jnr_step = data_config.get('jnr_step', 5)
        jnr_list = list(range(jnr_start, jnr_end + 1, jnr_step))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jnr_tag = args.jnr.replace(':', '_').replace(' ', '_') if args.jnr else 'all'

    print(f"\n{'=' * 60}")
    print(f"Data base:  {base_path}")
    print(f"Dataset:    {args.dataset}")
    print(f"JNR:        {jnr_list}")
    print(f"Classes:    {len(class_names)}")
    print(f"Model type: {args.model_type}")
    print(f"CLIP model: {clip_model_name}")

    # ============================================================
    # 3. Load data
    # ============================================================
    print(f"\n--- Loading data ---")
    data_loader = load_data(
        base_path=base_path,
        dataset=args.dataset,
        jnr_list=jnr_list,
        stft_suffix=stft_suffix,
        class_names=class_names,
        image_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=num_workers,
    )

    # ============================================================
    # 4. Load pretrained model & extract features
    # ============================================================
    all_labels = None
    all_jnr = None
    all_indices = None
    all_stft_raw = None
    all_stft_clip = None
    pretrained_features = None

    if args.no_stft:
        stft_dir = None
    elif args.stft_dir:
        stft_dir = args.stft_dir
    else:
        stft_dir = str(output_dir / 'stft_preview')
    if stft_dir:
        os.makedirs(stft_dir, exist_ok=True)
        print(f"STFT preview dir: {stft_dir}")

    if not args.skip_pretrained:
        print(f"\n--- Pretrained model ---")
        pretrained_model = load_pretrained_model(clip_model_name, device)
        result = extract_features(
            pretrained_model, data_loader, device, desc="Pretrained features",
            stft_dir=stft_dir,
        )
        pretrained_features = result[0]
        all_labels, all_jnr, all_indices = result[1], result[2], result[3]
        if stft_dir and len(result) >= 6:
            all_stft_raw, all_stft_clip = result[4], result[5]
        print(f"  Features: {pretrained_features.shape}")

        del pretrained_model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    else:
        print(f"\n--- Skipping pretrained model ---")

    # ============================================================
    # 5. Load trained model & extract features
    # ============================================================
    trained_features = None
    if not args.skip_trained:
        print(f"\n--- Trained model ---")
        trained_model = load_trained_model(config, args.checkpoint, device)
        result = extract_features(
            trained_model, data_loader, device, desc="Trained features",
            stft_dir=(stft_dir if all_stft_raw is None else None),
        )
        trained_features = result[0]
        _lbls, _jnr, _idx = result[1], result[2], result[3]
        if all_labels is None:
            all_labels, all_jnr, all_indices = _lbls, _jnr, _idx
        if all_stft_raw is None and stft_dir and len(result) >= 6:
            all_stft_raw, all_stft_clip = result[4], result[5]
        print(f"  Features: {trained_features.shape}")

        del trained_model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    else:
        print(f"\n--- Skipping trained model ---")

    if pretrained_features is None and trained_features is None:
        raise RuntimeError("Both models are skipped — nothing to do.")

    # ============================================================
    # 6. UMAP
    # ============================================================
    unique_labels = sorted(set(all_labels))
    n_neighbors = min(args.n_neighbors, len(all_labels) - 1)
    print(f"\n--- UMAP (n_neighbors={n_neighbors}, min_dist={args.min_dist}) ---")
    print(f"  Samples: {len(all_labels)}, Classes: {len(unique_labels)}")

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=args.min_dist,
        n_components=2,
        metric='euclidean',
        random_state=args.seed,
        verbose=True,
    )

    pretrained_embedding = None
    trained_embedding = None

    if pretrained_features is not None:
        print("  Fitting UMAP on pretrained features...")
        pretrained_embedding = reducer.fit_transform(pretrained_features)

    if trained_features is not None:
        print("  Fitting UMAP on trained features...")
        trained_embedding = reducer.fit_transform(trained_features)

    # ============================================================
    # 7. Plot & save
    # ============================================================
    if pretrained_embedding is not None:
        pretrained_save = str(
            output_dir / f'umap_pretrained_{args.dataset}_JNR_{jnr_tag}.png'
        )
        plot_umap(
            pretrained_embedding, all_labels,
            f'Pretrained CLIP ({clip_model_name}) | {len(all_labels)} samples, '
            f'{len(unique_labels)} classes | JNR={jnr_tag}',
            save_path=pretrained_save,
            highlight=args.highlight,
        )

    if trained_embedding is not None:
        trained_save = str(
            output_dir / f'umap_trained_{args.dataset}_JNR_{jnr_tag}.png'
        )
        plot_umap(
            trained_embedding, all_labels,
            f'Trained CLIP ({clip_model_name}) | {len(all_labels)} samples, '
            f'{len(unique_labels)} classes | JNR={jnr_tag}',
            save_path=trained_save,
            highlight=args.highlight,
        )

    # ============================================================
    # 8. Save JSON data + Interactive HTML (optional)
    # ============================================================
    if args.save_html:
        print(f"\n--- Saving UMAP data JSON ---")
        json_filename = f'umap_data_{args.dataset}_JNR_{jnr_tag}.json'
        json_path = str(output_dir / json_filename)

        data_dict = build_umap_data(
            pretrained_embedding=pretrained_embedding,
            trained_embedding=trained_embedding,
            all_labels=all_labels,
            all_jnr=all_jnr,
            all_indices=all_indices,
            clip_model_name=clip_model_name,
            dataset=args.dataset,
            jnr_tag=jnr_tag,
            highlight=args.highlight,
            stft_raw_paths=all_stft_raw,
            stft_clip_paths=all_stft_clip,
        )
        save_umap_data_json(data_dict, json_path)

        print(f"\n--- Generating interactive HTML ---")
        html_path = str(
            output_dir / f'umap_view_{args.dataset}_JNR_{jnr_tag}.html'
        )
        build_html(
            data_filename=json_filename,
            clip_model_name=clip_model_name,
            dataset=args.dataset,
            jnr_tag=jnr_tag,
            n_samples=len(all_labels),
            output_path=html_path,
        )

    print(f"\nDone. Output saved to {output_dir}/")


if __name__ == '__main__':
    main()
