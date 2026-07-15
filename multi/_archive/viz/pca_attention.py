#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pca_attention.py — CLIP ViT 注意力 PCA 可视化

对最后一层 encoder 的 patch hidden states 做 per-sample PCA，
同时提取最后一层 self-attention 中 CLS→patches 的注意力权重，
生成热力图叠加到 STFT 上。

两种互补视角：
  - PCA on patch features: 揭示特征空间中哪些 patch 最具区分性
  - CLS self-attention: 直接显示哪些 patch 对最终 CLS 特征贡献最大

与 SigLIP 版的核心差异：
  - SigLIP 有 pooling head attention (可学习的 probe token)
  - CLIP 使用最后一层 ViT self-attention (CLS token 对所有 patch 的注意力)

Usage:
  python multi/_archive/viz/pca_attention.py --checkpoint checkpoints/czsl_best_model.pt --dataset test --jnr 10 --n_samples_per_class 5
  python multi/_archive/viz/pca_attention.py --checkpoint checkpoints/czsl_best_model.pt --dataset test --jnr 0:1:20 --n_samples_per_class 10 --save-html
  python multi/_archive/viz/pca_attention.py --checkpoint checkpoints/czsl_best_model.pt --dataset test --jnr 10 --n_samples_per_class 20 --save-html

输出: results/pca_attention/
  pca_data_{dataset}_JNR_{jnr_tag}.json        # PCA + attention 原始数据
  pca_view_{dataset}_JNR_{jnr_tag}.html         # 交互式浏览器页面
  stft_preview/sample_XXXXXX_{raw,clip}.png     # STFT 预览图
"""
import argparse
import json
import math
import os
import re
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
import yaml

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from multi.data import STFTDataset
from multi.model import create_czsl_model
import clip


# ============================================================
# Constants (consistent with umap_compare.py)
# ============================================================
JAM_TYPE_COLORS = {
    'DFTJ': '#1f77b4', 'ISRJ': '#ff7f0e', 'SMSPJ': '#2ca02c',
    'CIJ': '#d62728', 'CSJ': '#9467bd',
    'AJ': '#8c564b', 'BJ': '#e377c2', 'SJ': '#7f7f7f',
    'NCJ': '#bcbd22', 'NPJ': '#17becf', 'NFMJ': '#aec7e8',
    'NPMJ': '#ffbb78', 'NAMJ': '#98df8a', 'PJ': '#c5b0d5',
}


# ============================================================
# CLI helpers (from umap_compare.py)
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


def collate_for_features(batch):
    """Simple collate: stack images, collect labels and metadata. Optionally raw_mag."""
    images = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    metadata = [item[2] for item in batch]
    if len(batch[0]) >= 4:
        raw_mags = torch.stack([item[3] for item in batch])
        return images, labels, metadata, raw_mags
    return images, labels, metadata


# ============================================================
# Data loading (from umap_compare.py)
# ============================================================
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

    from torch.utils.data import ConcatDataset
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
# Model loading (from umap_compare.py)
# ============================================================
def load_trained_model(config, checkpoint_path, device):
    """Load trained model from checkpoint. Supports vit and multishape_vit backbones.

    Uses the checkpoint's saved config (if available) for architecture-critical
    parameters like patch_sizes, so the model shape always matches the weights.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

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
# Vision model access
# ============================================================
def get_vision_model(model):
    """Extract VisionTransformer from CLIPForCZSL or DualBranchCLIPForCZSL.

    Returns:
        vision_model: CLIP VisionTransformer instance
        grid_size: int — patch grid size (7 for ViT-B/32, 14 for ViT-B/16, 16 for ViT-L/14)
        is_vit: bool — True if model uses ViT visual encoder
    """
    if hasattr(model, 'model') and hasattr(model.model, 'visual'):
        vision = model.model.visual
    elif hasattr(model, 'visual'):
        vision = model.visual
    else:
        raise RuntimeError(f"Cannot find visual encoder in model of type {type(model)}")

    # Check if it's a ViT (has transformer with resblocks)
    if hasattr(vision, 'transformer') and hasattr(vision.transformer, 'resblocks'):
        grid_size = int(math.sqrt(vision.positional_embedding.shape[0] - 1))
        return vision, grid_size, True
    else:
        # ResNet or other non-ViT encoder
        return vision, None, False


# ============================================================
# Feature extraction (hook-based for CLIP ViT)
# ============================================================
@torch.no_grad()
def extract_patch_features(model, data_loader, device, desc="Extracting",
                           stft_dir=None, max_samples_per_class=None):
    """Extract patch hidden states and CLS self-attention weights from CLIP ViT.

    Uses hooks to capture intermediate outputs that CLIP ViT doesn't natively expose:
      - Registers forward hook on last transformer block to capture hidden states
      - Temporarily patches last block's attention() to capture attention weights

    For each sample, returns:
      - patch_hidden: [grid^2, D] — last encoder layer patch outputs (PCA-ready)
      - attn_weights: [grid^2] — CLS self-attention averaged across heads
      - per_head_attn: [num_heads, grid^2] — per-head attention weights
      - label_str, jnr_val, global_idx
      - Optionally saves STFT preview PNGs

    Args:
        model: CLIPForCZSL or DualBranchCLIPForCZSL
        data_loader: DataLoader yielding (images, labels, metadata, [raw_mags])
        device: torch device
        desc: tqdm description
        stft_dir: if set, saves STFT preview PNGs
        max_samples_per_class: if set, limit samples per class (applied post-hoc)

    Returns:
        results: list of dicts per sample
        all_stft_raw, all_stft_clip: lists of STFT preview paths (or None)
    """
    vision_model, grid_size, is_vit = get_vision_model(model)

    if not is_vit:
        raise RuntimeError(
            "PCA attention visualization only supports ViT-based CLIP models.\n"
            "ResNet-based CLIP (RN50, RN101) does not have patch tokens."
        )

    vision_model.eval()
    print(f"Vision model: ViT, grid_size={grid_size} ({grid_size*grid_size} patches)")
    num_layers = len(vision_model.transformer.resblocks)
    print(f"Layers: {num_layers}, using last layer for attention")

    # ---- Setup hooks on last transformer block ----
    last_block = vision_model.transformer.resblocks[-1]
    captured = {}

    # Hook 1: capture output of last block (hidden states)
    def last_block_hook(module, input, output):
        captured['last_hidden'] = output  # [L, B, D] where L = 1 + grid^2

    hook_handle = last_block.register_forward_hook(last_block_hook)

    # Hook 2: patch attention method to capture weights
    # The attention() method of ResidualAttentionBlock calls
    #   self.attn(x, x, x, need_weights=False, attn_mask=...)
    # We patch it to use need_weights=True and capture the weights.

    def patched_attention(self, x):
        self.attn_mask = (self.attn_mask.to(dtype=x.dtype, device=x.device)
                          if self.attn_mask is not None else None)
        attn_out, attn_weights = self.attn(
            x, x, x,
            need_weights=True,
            attn_mask=self.attn_mask,
            average_attn_weights=False,
        )
        captured['attn_weights'] = attn_weights
        return attn_out

    original_attention = last_block.attention
    last_block.attention = types.MethodType(patched_attention, last_block)

    try:
        results = []
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

            # Forward pass — hooks capture intermediate values
            _ = vision_model(images)

            # Extract patch hidden states: [L, B, D] → skip CLS (index 0)
            #                              → permute → [B, grid^2, D]
            patch_hidden = captured['last_hidden'][1:, :, :].permute(1, 0, 2)

            # Extract attention weights
            # nn.MultiheadAttention return shape varies by PyTorch version:
            #   PyTorch < 2.0 (batch_first=False): [B * num_heads, L, L] (3D)
            #   PyTorch >= 2.0 (batch_first=False): [B, num_heads, L, L] (4D)
            # We normalize to [B, num_heads, L, L] then extract CLS→patches
            attn_raw = captured['attn_weights']
            L = patch_hidden.shape[1] + 1  # grid^2 + 1
            B = images.shape[0]
            num_heads = vision_model.transformer.resblocks[-1].attn.num_heads

            # Robust reshape: detect shape pattern and normalize
            if attn_raw.dim() == 3:
                # [B * num_heads, L, L] — reshape directly
                if attn_raw.shape[0] == B * num_heads:
                    attn_reshaped = attn_raw.reshape(B, num_heads, L, L)
                elif attn_raw.shape[1] == B * num_heads:
                    # [L, B * num_heads, L] — permute and reshape
                    attn_reshaped = attn_raw.permute(1, 0, 2).reshape(B, num_heads, L, L)
                else:
                    # Fallback: search for B*num_heads dim
                    total = B * num_heads
                    found = False
                    for d in range(3):
                        if attn_raw.shape[d] == total:
                            perm = [d] + [i for i in range(3) if i != d]
                            attn_reshaped = attn_raw.permute(*perm).reshape(B, num_heads, L, L)
                            found = True
                            break
                    if not found:
                        raise RuntimeError(
                            f"Cannot interpret attention weights shape {attn_raw.shape} "
                            f"(B={B}, num_heads={num_heads}, L={L})")
            elif attn_raw.dim() == 4:
                # [B, num_heads, L, L] — already in expected format
                attn_reshaped = attn_raw
            else:
                raise RuntimeError(
                    f"Unexpected attention weights dim: {attn_raw.dim()}D, "
                    f"shape={attn_raw.shape}")

            # CLS token (index 0) attending to patches (indices 1:)
            per_head_attn = attn_reshaped[:, :, 0, 1:]  # [B, num_heads, grid^2]
            avg_attn = per_head_attn.mean(dim=1)  # [B, grid^2]

            for i, meta in enumerate(metadata):
                jam_types = meta.get('jam_types', [])
                if isinstance(jam_types, list):
                    label_str = '+'.join(sorted(
                        [normalize_label(t) for t in jam_types]))
                else:
                    label_str = normalize_label(str(jam_types))

                results.append({
                    'idx': global_idx,
                    'label': label_str,
                    'jnr': meta.get('JNR', '?'),
                    'patch_hidden': patch_hidden[i].cpu().numpy(),
                    'attn_weights': avg_attn[i].cpu().numpy(),
                    'per_head_attn': per_head_attn[i].cpu().numpy(),
                })

                if stft_dir and has_raw:
                    os.makedirs(stft_dir, exist_ok=True)
                    raw_fname = f'sample_{global_idx:06d}_raw.png'
                    clip_fname = f'sample_{global_idx:06d}_clip.png'

                    raw_img = raw_mags[i].cpu().numpy()
                    plt.imsave(os.path.join(stft_dir, raw_fname),
                               raw_img, cmap='inferno')

                    clip_arr = images[i].cpu().numpy()
                    clip_avg = clip_arr.mean(axis=0)
                    clip_avg = (clip_avg - clip_avg.min()) / \
                               (clip_avg.max() - clip_avg.min() + 1e-8)
                    plt.imsave(os.path.join(stft_dir, clip_fname),
                               clip_avg, cmap='inferno')

                    all_stft_raw.append(f'stft_preview/{raw_fname}')
                    all_stft_clip.append(f'stft_preview/{clip_fname}')
                elif stft_dir:
                    all_stft_raw.append(None)
                    all_stft_clip.append(None)

                global_idx += 1

    finally:
        hook_handle.remove()
        last_block.attention = original_attention

    # Apply per-class sampling limit (post-hoc)
    # Shuffle first to ensure JNR diversity — otherwise early JNR levels
    # (e.g. JNR=0) fill the quota and later JNRs are completely excluded.
    if max_samples_per_class is not None:
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(results))
        shuffled = [results[i] for i in indices]

        class_counts = {}
        filtered = []
        for r in shuffled:
            lbl = r['label']
            cnt = class_counts.get(lbl, 0)
            if cnt < max_samples_per_class:
                filtered.append(r)
                class_counts[lbl] = cnt + 1

        # Restore original order (by idx) for consistent display
        filtered.sort(key=lambda r: r['idx'])

        # Log per-JNR distribution after filtering
        jnr_counts_after = {}
        for r in filtered:
            j = r['jnr']
            jnr_counts_after[j] = jnr_counts_after.get(j, 0) + 1
        jnr_summary = ', '.join(
            f'JNR={j}: {c}' for j, c in sorted(jnr_counts_after.items()))
        print(f"  Per-class sampling: {len(results)} -> {len(filtered)} samples "
              f"(max {max_samples_per_class}/class)")
        print(f"  JNR distribution after sampling: {jnr_summary}")
        results = filtered

    stft_paths = (all_stft_raw, all_stft_clip) if stft_dir else (None, None)
    return results, stft_paths, grid_size


# ============================================================
# PCA (same as SigLIP version)
# ============================================================
def compute_per_sample_pca(results, n_components=3):
    """Compute per-sample PCA on patch hidden states.

    Adds to each result dict:
      - pca_pc{1..n}: [N_patches] — PCA component projections
      - pca_variance_ratio: [n] — explained variance ratio per component
    """
    for r in tqdm(results, desc="PCA"):
        features = r['patch_hidden']  # [N_patches, D]
        pca = PCA(n_components=n_components, random_state=42)
        transformed = pca.fit_transform(features)  # [N_patches, n_components]
        for k in range(n_components):
            r[f'pca_pc{k + 1}'] = transformed[:, k].astype(np.float32)
        r['pca_variance_ratio'] = pca.explained_variance_ratio_.astype(
            np.float32).tolist()
        # Free memory: don't need full hidden states in output
        del r['patch_hidden']
    return results


# ============================================================
# Heatmap overlay (static PNG)
# ============================================================
def _make_heatmap(values_N, grid_size):
    """Reshape [grid^2] -> [grid, grid] -> upsample to [224, 224] heatmap array."""
    grid = values_N.reshape(grid_size, grid_size)
    grid = torch.from_numpy(np.asarray(grid, dtype=np.float32)
                            ).unsqueeze(0).unsqueeze(0)
    up = F.interpolate(grid, size=(224, 224), mode='bilinear',
                       align_corners=False)
    return up.squeeze().numpy()


def save_heatmap_png(result, stft_clip_path, output_dir, grid_size, prefix=''):
    """Save a single heatmap overlay PNG for one sample.

    Creates a 3-panel figure: STFT | PCA PC1 | Attention
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load STFT clip preview as background
    if stft_clip_path:
        # stft_clip_path is relative like 'stft_preview/sample_000001_clip.png'
        stft_full = os.path.join(output_dir, '..', stft_clip_path)
    else:
        stft_full = None

    if stft_full and os.path.exists(stft_full):
        bg = plt.imread(stft_full)
    else:
        bg = np.ones((224, 224))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Left: STFT background
    axes[0].imshow(bg, cmap='inferno', aspect='equal')
    axes[0].set_title('STFT (clip input avg)', fontsize=10)
    axes[0].axis('off')

    # Middle: PCA PC1 overlay
    pca_hm = _make_heatmap(result['pca_pc1'], grid_size)
    axes[1].imshow(bg, cmap='gray', aspect='equal')
    im1 = axes[1].imshow(pca_hm, cmap='jet', alpha=0.55, aspect='equal')
    axes[1].set_title(
        f"PCA PC1 (var={result['pca_variance_ratio'][0]:.3f})", fontsize=10)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Right: Attention overlay
    attn_hm = _make_heatmap(result['attn_weights'], grid_size)
    axes[2].imshow(bg, cmap='gray', aspect='equal')
    im2 = axes[2].imshow(attn_hm, cmap='jet', alpha=0.55, aspect='equal')
    axes[2].set_title('CLS Self-Attention', fontsize=10)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    label = result['label']
    idx = result['idx']
    jnr = result['jnr']
    plt.suptitle(f'{label} | idx={idx} | JNR={jnr}', fontsize=12,
                 fontweight='bold')
    plt.tight_layout()

    fname = f'sample_{idx:06d}.png'
    save_path = os.path.join(output_dir, fname)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    return f'heatmaps/{fname}'


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


def build_pca_json(results, clip_model_name, dataset, jnr_tag,
                   stft_raw_paths, stft_clip_paths):
    """Build JSON-serializable dict from PCA+attention results.

    Simplified from SigLIP version — single model only, no pretrained/trained tabs.
    """
    data = {}
    for r in results:
        lbl = r['label']
        if lbl not in data:
            data[lbl] = []
        entry = {
            'idx': r['idx'],
            'jnr': r['jnr'],
            'pca_pc1': _json_safe(r['pca_pc1']),
            'pca_pc2': _json_safe(r.get('pca_pc2', np.zeros(r['attn_weights'].shape[0]))),
            'pca_pc3': _json_safe(r.get('pca_pc3', np.zeros(r['attn_weights'].shape[0]))),
            'pca_variance_ratio': r['pca_variance_ratio'],
            'attn_weights': _json_safe(r['attn_weights']),
        }
        idx = r['idx']
        if stft_raw_paths is not None and idx < len(stft_raw_paths):
            entry['stft_raw'] = stft_raw_paths[idx]
            entry['stft_clip'] = stft_clip_paths[idx] if stft_clip_paths else None
        data[lbl].append(entry)

    all_labels = sorted(set(r['label'] for r in results))
    all_jnrs = sorted(set(r['jnr'] for r in results))
    single_cats = sorted(
        {l for l in all_labels if '+' not in l},
        key=lambda x: list(JAM_TYPE_COLORS.keys()).index(x)
        if x in JAM_TYPE_COLORS else 999,
    )
    composite_cats = sorted({l for l in all_labels if '+' in l})

    return {
        'data': data,
        'categories': {'single': single_cats, 'composite': composite_cats},
        'jnr_levels': all_jnrs,
        'colors': {k: v for k, v in JAM_TYPE_COLORS.items() if k in single_cats},
        'info': {
            'clip_model': clip_model_name,
            'dataset': dataset,
            'jnr_tag': jnr_tag,
            'n_samples': len(results),
        },
    }


def save_pca_json(data_dict, output_path):
    """Save PCA data dict as JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'  JSON saved: {output_path} ({size_mb:.1f} MB)')


# ============================================================
# Interactive HTML (simplified — single model, no tabs)
# ============================================================
def build_pca_html(data_filename, clip_model_name, dataset, jnr_tag,
                   n_samples, output_path):
    """Generate interactive HTML for PCA + attention heatmap browsing.

    Single-model version (no pretrained/trained tabs).
    Uses pre-rendered PNG heatmaps (3-panel: STFT | PCA PC1 | Attention).
    JSON provides metadata for navigation: labels, indices, JNR values.
    """
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CLIP Attention PCA</title>
<style>
:root {{
  --bg: #1a1a2e; --bg2: #16213e; --border: #0f3460; --accent: #e94560;
  --text: #eee; --text2: #ccc; --text3: #888;
}}
:root.light {{
  --bg: #f0f2f5; --bg2: #fff; --border: #d0d5dd; --accent: #d6336c;
  --text: #1a1a2e; --text2: #555; --text3: #999;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; }}
.header {{ background: var(--bg2); padding: 6px 16px; display: flex; align-items: center; gap: 8px;
           border-bottom: 1px solid var(--border); flex-shrink: 0; min-height: 40px; }}
.header h2 {{ font-size: 14px; font-weight: 600; white-space: nowrap; }}
.header .spacer {{ flex: 1; }}
#fileInput {{ display: none; }}
.load-btn, .theme-btn, .nav-btn {{ padding: 4px 12px; border: 1px solid var(--border); background: transparent;
    color: var(--text2); cursor: pointer; border-radius: 4px; font-size: 11px; }}
.load-btn {{ border-color: var(--accent); color: var(--accent); }}
.load-btn:hover, .theme-btn:hover, .nav-btn:hover {{ background: var(--border); }}
#fileName {{ font-size: 10px; color: var(--text3); max-width: 250px; overflow: hidden;
            text-overflow: ellipsis; white-space: nowrap; }}
.main {{ display: flex; flex: 1; overflow: hidden; }}
.sidebar {{ width: 240px; background: var(--bg2); border-right: 1px solid var(--border);
           overflow-y: auto; padding: 10px; flex-shrink: 0; }}
.sidebar h3 {{ font-size: 11px; color: var(--accent); margin: 8px 0 4px 0; text-transform: uppercase; }}
.sidebar label {{ display: flex; align-items: center; padding: 2px 6px; font-size: 11px;
                  cursor: pointer; border-radius: 3px; gap: 5px; }}
.sidebar label:hover {{ background: var(--bg); }}
.sidebar input[type="radio"] {{ accent-color: var(--accent); transform: scale(0.85); }}
.sidebar select {{ width: 100%; padding: 4px; font-size: 11px; background: var(--bg); color: var(--text);
                  border: 1px solid var(--border); border-radius: 4px; margin: 4px 0; }}
.color-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
.hm-area {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}
.hm-controls {{ padding: 6px 10px; border-bottom: 1px solid var(--border); display: flex;
                gap: 8px; align-items: center; flex-shrink: 0; }}
.hm-controls label {{ font-size: 10px; color: var(--text3); white-space: nowrap; }}
.hm-controls select {{ font-size: 11px; background: var(--bg); color: var(--text);
    border: 1px solid var(--border); border-radius: 4px; padding: 3px; }}
.hm-view {{ flex: 1; display: flex; justify-content: center; align-items: center;
            overflow: auto; padding: 10px; }}
.hm-view img {{ max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 6px;
                border: 1px solid var(--border); }}
.info-bar {{ padding: 3px 10px; font-size: 10px; color: var(--text3); background: var(--bg2);
            border-top: 1px solid var(--border); flex-shrink: 0; }}
#loading {{ font-size: 16px; color: var(--text3); text-align: center; padding: 40px; }}
</style>
</head>
<body>

<div class="header">
  <h2>CLIP Attention PCA</h2>
  <input type="file" id="fileInput" accept=".json" onchange="window._onFilePicked(event)">
  <button class="load-btn" onclick="document.getElementById('fileInput').click()">Load JSON</button>
  <button class="theme-btn" id="themeBtn" onclick="_toggleTheme()" title="Toggle dark/light">&#9788;</button>
  <span id="fileName">{data_filename}</span>
  <span class="spacer"></span>
  <span id="modelTag" style="font-size:10px;color:var(--text3)"></span>
</div>

<div class="main">
  <div class="sidebar" id="sidebar"></div>
  <div class="hm-area">
    <div class="hm-controls">
      <button class="nav-btn" onclick="prevSample()" title="Previous">&#9664; Prev</button>
      <select id="sampleSelect" onchange="onSampleSelect()" style="min-width:260px"></select>
      <button class="nav-btn" onclick="nextSample()" title="Next">Next &#9654;</button>
      <span id="sampleInfo" style="font-size:10px;color:var(--text3)"></span>
    </div>
    <div class="hm-view" id="hmView">
      <div id="loading">Select a class from the sidebar</div>
    </div>
  </div>
</div>

<div class="info-bar" id="infoBar">Ready — waiting for data</div>

<script>
var DATA = null, SINGLE_COLORS, SINGLE, COMPOSITE, JNR_LEVELS;
var currentLabel = null, currentSampleIdx = 0, currentJnrFilter = 'all';

function setData(jsonObj, filename) {{
  DATA = jsonObj;
  SINGLE_COLORS = DATA.colors || {{}};
  SINGLE = DATA.categories.single;
  COMPOSITE = DATA.categories.composite;
  JNR_LEVELS = DATA.jnr_levels || [];
  document.getElementById('fileName').textContent = filename || 'loaded';
  var info = DATA.info;
  document.getElementById('modelTag').textContent =
    (info.clip_model||'') + ' | ' + (info.dataset||'') + ' | JNR=' + (info.jnr_tag||'');
  buildSidebar();
  updateView();
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
    try {{ setData(JSON.parse(e.target.result), file.name); }}
    catch(err) {{ document.getElementById('loading').textContent = 'Parse error: ' + err.message; }}
  }};
  reader.readAsText(file);
}}

function buildSidebar() {{
  var sidebar = document.getElementById('sidebar');
  var html = '<div style="margin-bottom:8px"><i style="font-size:10px;color:var(--text3)">'
           + (DATA.info.clip_model||'') + ' | ' + (DATA.info.dataset||'')
           + ' | JNR range: ' + (DATA.info.jnr_tag||'') + '</i></div>';

  // JNR filter
  if (JNR_LEVELS.length > 0) {{
    html += '<h3>JNR Filter</h3>';
    html += '<select onchange="onJnrFilterChange()" id="jnrFilter">';
    html += '<option value="all" selected>All JNRs</option>';
    JNR_LEVELS.forEach(function(j) {{
      html += '<option value="' + j + '">JNR = ' + j + '</option>';
    }});
    html += '</select>';
  }}

  if (SINGLE.length > 0) {{
    html += '<h3>Single (' + SINGLE.length + ')</h3>';
    SINGLE.forEach(function(cat) {{
      var color = SINGLE_COLORS[cat] || '#999';
      var count = countClassSamples(cat);
      html += '<label><span class="color-dot" style="background:' + color + '"></span>'
           + '<input type="radio" name="catSel" onchange="onCatChange()" data-cat="' + cat + '">'
           + cat + ' (' + count + ')</label>';
    }});
  }}
  if (COMPOSITE.length > 0) {{
    html += '<h3>Composite (' + COMPOSITE.length + ')</h3>';
    COMPOSITE.forEach(function(cat) {{
      var count = countClassSamples(cat);
      html += '<label><input type="radio" name="catSel" onchange="onCatChange()" data-cat="' + cat + '">'
           + cat + ' (' + count + ')</label>';
    }});
  }}
  sidebar.innerHTML = html;
}}

function countClassSamples(cat) {{
  var allSamples = DATA.data && DATA.data[cat] ? DATA.data[cat] : [];
  if (currentJnrFilter === 'all') return allSamples.length;
  return allSamples.filter(function(s) {{ return String(s.jnr) === currentJnrFilter; }}).length;
}}

function getSamples() {{
  if (!DATA || !DATA.data || !currentLabel) return [];
  var allSamples = DATA.data[currentLabel] || [];
  if (currentJnrFilter === 'all') return allSamples;
  return allSamples.filter(function(s) {{ return String(s.jnr) === currentJnrFilter; }});
}}

function onJnrFilterChange() {{
  currentJnrFilter = document.getElementById('jnrFilter').value;
  currentSampleIdx = 0;
  currentLabel = null;
  // Refresh sidebar counts
  buildSidebar();
  // Clear view
  document.getElementById('hmView').innerHTML = '<div id="loading">Select a class from the sidebar</div>';
  document.getElementById('sampleSelect').innerHTML = '';
  document.getElementById('infoBar').textContent = 'JNR filter: ' + (currentJnrFilter === 'all' ? 'All' : currentJnrFilter);
}}

function updateView() {{
  var samples = getSamples();
  var sel = document.getElementById('sampleSelect');
  sel.innerHTML = '';
  if (samples.length === 0) {{
    document.getElementById('hmView').innerHTML = '<div id="loading">Select a class from the sidebar</div>';
    document.getElementById('infoBar').textContent = 'Ready';
    return;
  }}
  samples.forEach(function(s, i) {{
    var opt = document.createElement('option');
    opt.value = i;
    opt.textContent = 'idx=' + s.idx + '  JNR=' + s.jnr
                    + (s.pca_variance_ratio ? '  PC1_var=' + s.pca_variance_ratio[0].toFixed(3) : '');
    sel.appendChild(opt);
  }});
  sel.value = Math.min(currentSampleIdx, samples.length - 1);
  currentSampleIdx = parseInt(sel.value);
  showSample();
}}

function showSample() {{
  var samples = getSamples();
  if (samples.length === 0) return;
  var s = samples[currentSampleIdx];
  var imgPath = 'heatmaps/sample_' + String(s.idx).padStart(6, '0') + '.png';
  document.getElementById('hmView').innerHTML =
    '<img src="' + imgPath + '" alt="Heatmap ' + s.idx + '" '
    + 'onerror="this.parentElement.innerHTML=\\'<div id=loading>Heatmap not found:<br>' + imgPath + '</div>\\'">';
  document.getElementById('sampleInfo').textContent =
    ' | ' + (currentSampleIdx + 1) + '/' + samples.length;
  document.getElementById('infoBar').textContent =
    currentLabel + ' | idx=' + s.idx + ' | JNR=' + s.jnr
    + (s.pca_variance_ratio ? ' | PC1 var=' + s.pca_variance_ratio[0].toFixed(3) : '');
}}

function onCatChange() {{
  var checked = document.querySelector('input[name="catSel"]:checked');
  if (checked) {{
    currentLabel = checked.dataset.cat;
    currentSampleIdx = 0;
    updateView();
  }}
}}

function onSampleSelect() {{
  currentSampleIdx = parseInt(document.getElementById('sampleSelect').value);
  showSample();
}}

function prevSample() {{
  if (currentSampleIdx > 0) {{ currentSampleIdx--; updateView(); }}
}}

function nextSample() {{
  var samples = getSamples();
  if (currentSampleIdx < samples.length - 1) {{ currentSampleIdx++; updateView(); }}
}}

window._onFilePicked = function(e) {{
  if (e.target.files.length > 0) loadFromFile(e.target.files[0]);
}};

function _toggleTheme() {{
  var root = document.documentElement;
  var isLight = root.classList.toggle('light');
  document.getElementById('themeBtn').innerHTML = isLight ? '&#9790;' : '&#9788;';
  localStorage.setItem('pca-theme', isLight ? 'light' : 'dark');
}};
(function() {{
  if (localStorage.getItem('pca-theme') === 'light') {{
    document.documentElement.classList.add('light');
    document.getElementById('themeBtn').innerHTML = '&#9790;';
  }}
}})();

document.addEventListener('dragover', function(e) {{ e.preventDefault(); }});
document.addEventListener('drop', function(e) {{
  e.preventDefault();
  if (e.dataTransfer.files.length > 0) loadFromFile(e.dataTransfer.files[0]);
}});

// Auto-load default JSON
if (!loadFromXHR('{data_filename}')) {{
  document.getElementById('loading').textContent =
    'Could not auto-load "{data_filename}".\\nDrag & drop a JSON file or click "Load JSON".';
}}
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
        description='CLIP ViT Attention PCA visualization'
    )
    parser.add_argument('--config', type=str, default='multi/config.yaml',
                        help='Path to config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--dir', type=str, default=None,
                        help='Base data directory (overrides config data.base_path)')
    parser.add_argument('--dataset', type=str, default='test',
                        help='Dataset split: train / val / test')
    parser.add_argument('--jnr', type=str, default=None,
                        help='JNR spec: "10", "0:5:20", "0:20"')
    parser.add_argument('--n_samples_per_class', type=int, default=20,
                        help='Max samples per class (default 20)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='results/pca_attention')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--save-html', action='store_true', default=False)
    parser.add_argument('--stft-dir', type=str, default=None)
    parser.add_argument('--no-stft', action='store_true', default=False)
    args = parser.parse_args()

    # ============================================================
    # 1. Config
    # ============================================================
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ============================================================
    # 2. Data params
    # ============================================================
    data_config = config.get('data', {})
    base_path = args.dir or data_config.get('base_path', '')
    stft_suffix = data_config.get('stft_suffix', 'echo_stfts')
    num_workers = data_config.get('num_workers', 0)
    class_names = [cls['name'] for cls in config.get('jamming_classes', [])]
    clip_model_name = config.get('model', {}).get('clip_model', 'ViT-B/32')

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
    print(f"Data base:    {base_path}")
    print(f"Dataset:      {args.dataset}")
    print(f"JNR:          {jnr_list}")
    print(f"Classes:      {len(class_names)}")
    print(f"CLIP model:   {clip_model_name}")
    print(f"Max/class:    {args.n_samples_per_class}")

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

    # STFT dir setup
    if args.no_stft:
        stft_dir = None
    elif args.stft_dir:
        stft_dir = args.stft_dir
    else:
        stft_dir = str(output_dir / 'stft_preview')
    if stft_dir:
        os.makedirs(stft_dir, exist_ok=True)

    # ============================================================
    # 4. Load trained model & extract patch features + attention
    # ============================================================
    print(f"\n--- Trained model ---")
    trained_model = load_trained_model(config, args.checkpoint, device)

    results, stft_paths, grid_size = extract_patch_features(
        trained_model, data_loader, device, desc="Extracting",
        stft_dir=stft_dir, max_samples_per_class=args.n_samples_per_class,
    )
    all_stft_raw, all_stft_clip = stft_paths if stft_paths[0] is not None else (None, None)
    print(f"  Samples: {len(results)}")
    print(f"  Grid size: {grid_size}x{grid_size} ({grid_size * grid_size} patches)")

    del trained_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ============================================================
    # 5. PCA
    # ============================================================
    print(f"\n--- PCA ---")
    results = compute_per_sample_pca(results)

    # ============================================================
    # 6. Save static PNG heatmaps
    # ============================================================
    heatmap_dir = str(output_dir / 'heatmaps')
    os.makedirs(heatmap_dir, exist_ok=True)

    for r in tqdm(results, desc="Saving PNGs"):
        clip_path = (all_stft_clip[r['idx']]
                     if all_stft_clip and r['idx'] < len(all_stft_clip)
                     else None)
        save_heatmap_png(r, clip_path, heatmap_dir, grid_size)
    print(f"  Heatmaps saved to {heatmap_dir}")

    # ============================================================
    # 7. JSON + HTML (optional)
    # ============================================================
    if args.save_html:
        print(f"\n--- Saving PCA data JSON ---")
        json_filename = f'pca_data_{args.dataset}_JNR_{jnr_tag}.json'
        json_path = str(output_dir / json_filename)

        data_dict = build_pca_json(
            results=results,
            clip_model_name=clip_model_name,
            dataset=args.dataset,
            jnr_tag=jnr_tag,
            stft_raw_paths=all_stft_raw,
            stft_clip_paths=all_stft_clip,
        )
        save_pca_json(data_dict, json_path)

        print(f"\n--- Generating interactive HTML ---")
        html_path = str(output_dir / f'pca_view_{args.dataset}_JNR_{jnr_tag}.html')
        build_pca_html(
            data_filename=json_filename,
            clip_model_name=clip_model_name,
            dataset=args.dataset,
            jnr_tag=jnr_tag,
            n_samples=len(results),
            output_path=html_path,
        )

    print(f"\nDone. Output saved to {output_dir}/")


if __name__ == '__main__':
    main()
