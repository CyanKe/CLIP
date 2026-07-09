"""
Attention Weight Visualization for 1D Conformer Time-Series Signals.

Extracts self-attention weights from each ConformerBlock's MHSA layer and
overlays them as an attention "envelope" on top of the original I/Q time-domain
waveform — dual-axis plots for IEEE TAES publication-quality figures.

Usage:
    # Single-layer (last block) high-res plot
    python -m conformer_1d.visualize_attention \\
        --checkpoint checkpoints/conformer_best_model.pt --mode single --samples 3

    # All 6 layers in a 3×2 grid
    python -m conformer_1d.visualize_attention \\
        --checkpoint checkpoints/conformer_best_model.pt --mode layers --samples 2

    # Per-head breakdown for a specific layer
    python -m conformer_1d.visualize_attention \\
        --checkpoint checkpoints/conformer_best_model.pt --mode heads --layer 5 --samples 2

    # Compare specific jamming types
    python -m conformer_1d.visualize_attention \\
        --checkpoint checkpoints/conformer_best_model.pt --mode compare --classes DFTJ AJ CSJ

    # With backbone override
    python -m conformer_1d.visualize_attention \\
        --checkpoint checkpoints/conformer_best_model.pt --backbone conformer --mode layers
"""

import os
import sys
import yaml
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _parent)

from conformer_1d.model_1d import create_1d_model
from conformer_1d.data_1d import (TimeSignalDataset, TokenizerWrapper,
                                   collate_fn_conformer)


# ═════════════════════════════════════════════════════════════════════════════
# Colours & style (IEEE TAES publication quality)
# ═════════════════════════════════════════════════════════════════════════════

SIGNAL_COLOR = '#2B5B84'       # deep blue — I-channel waveform
ATTN_FILL_COLOR = '#E87200'    # warm orange — attention envelope
ATTN_LINE_COLOR = '#E87200'
GRID_COLOR = '#CCCCCC'

# Per-head colours (distinct, colourblind-friendly)
HEAD_COLORS = ['#E87200', '#009E73', '#CC79A7', '#56B4E9']


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def upsample_attention(attn_1d: np.ndarray, target_len: int = 8000) -> np.ndarray:
    """Linearly interpolate sub-sampled attention weights to original signal length.

    The Conformer's 3-layer conv subsampling reduces 8000 → ~667 time steps.
    This function upsamples the 1D attention curve back to 8000 samples so it
    can be overlaid on the raw I/Q waveform.

    Args:
        attn_1d: (T_sub,) attention weights at the sub-sampled resolution.
        target_len: original signal length (default 8000).

    Returns:
        (target_len,) float32 numpy array, normalized to [0, 1].
    """
    t = torch.as_tensor(attn_1d, dtype=torch.float32)
    t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, T_sub)
    t = F.interpolate(t, size=target_len, mode='linear', align_corners=False)
    t = t.squeeze().numpy()

    # Normalise to [0, 1]
    t_min, t_max = t.min(), t.max()
    if t_max - t_min > 1e-8:
        t = (t - t_min) / (t_max - t_min)
    return t


def average_attention(attn_matrix: torch.Tensor) -> np.ndarray:
    """Reduce a per-layer attention matrix to a 1D importance curve.

    attn_matrix: (num_heads, T, T) — self-attention weights for one layer.
    Strategy: mean over query positions, then mean over heads → (T,).

    The model uses global mean pooling (no [CLS] token), so averaging over the
    query dimension captures "how much each time position is attended to on
    average by all other positions."
    """
    # Mean over query dim (dim=1 → rows of attn matrix)
    attn_query_mean = attn_matrix.mean(dim=1)  # (num_heads, T)
    # Mean over heads
    attn_1d = attn_query_mean.mean(dim=0)      # (T,)
    return attn_1d.cpu().numpy()


def average_attention_per_head(attn_matrix: torch.Tensor) -> np.ndarray:
    """Reduce per-layer attention to per-head 1D curves.

    Returns: (num_heads, T) — one curve per head.
    """
    attn_query_mean = attn_matrix.mean(dim=1)  # (num_heads, T)
    return attn_query_mean.cpu().numpy()


def get_jamming_label(metadata: dict, class_names: list) -> str:
    """Build a concise label string from sample metadata."""
    jam_types = metadata.get('jam_types', [])
    if isinstance(jam_types, str):
        jam_types = [jam_types] if jam_types and jam_types != 'None' else []
    if not jam_types:
        return 'Unknown'
    return '+'.join(jam_types)


# ═════════════════════════════════════════════════════════════════════════════
# Figure-drawing functions
# ═════════════════════════════════════════════════════════════════════════════

def _style_axes(ax1, ax2, x_time, title_str):
    """Apply consistent IEEE styling to a dual-axis attention plot."""
    ax1.set_xlabel('Time (samples)', fontsize=11)
    ax1.set_ylabel('I-Channel Amplitude', fontsize=11, color=SIGNAL_COLOR)
    ax1.tick_params(axis='y', labelcolor=SIGNAL_COLOR, labelsize=9)
    ax1.tick_params(axis='x', labelsize=9)
    ax1.set_xlim(0, len(x_time) - 1)
    ax1.grid(True, color=GRID_COLOR, alpha=0.4, linestyle='--')

    ax2.set_ylabel('Attention Weight', fontsize=11, color=ATTN_LINE_COLOR)
    ax2.tick_params(axis='y', labelcolor=ATTN_LINE_COLOR, labelsize=9)
    ax2.set_ylim(0, 1.05)

    ax1.set_title(title_str, fontsize=12, fontweight='bold')


def plot_single_layer(
    signal_i: np.ndarray,
    attn_1d: np.ndarray,
    label_str: str,
    jnr: int,
    layer_idx: int,
    save_path: str,
):
    """Single dual-axis plot: I-channel waveform + attention envelope.

    Args:
        signal_i: (8000,) I-channel samples.
        attn_1d: (667,) raw attention weights (will be upsampled).
        label_str: jamming type label (e.g. "DFTJ+AJ").
        jnr: JNR value for title annotation.
        layer_idx: 0-based ConformerBlock index.
        save_path: output PNG path.
    """
    attn_aligned = upsample_attention(attn_1d, target_len=len(signal_i))
    x_time = np.arange(len(signal_i))

    # Normalise signal amplitude for visual alignment
    sig_max = np.abs(signal_i).max()
    if sig_max > 1e-8:
        signal_norm = signal_i / sig_max
    else:
        signal_norm = signal_i

    fig, ax1 = plt.subplots(figsize=(14, 5), dpi=150)

    # I-channel waveform (deep blue)
    ax1.plot(x_time, signal_norm, color=SIGNAL_COLOR, alpha=0.85,
             linewidth=0.8, label='I-Channel Signal')

    # Attention envelope (warm orange)
    ax2 = ax1.twinx()
    ax2.plot(x_time, attn_aligned, color=ATTN_LINE_COLOR, linewidth=1.5,
             alpha=0.9, label='Attention Weight')
    ax2.fill_between(x_time, 0, attn_aligned, color=ATTN_FILL_COLOR, alpha=0.30)

    title = f'{label_str}  |  Layer {layer_idx + 1}  |  JNR=+{jnr} dB'
    _style_axes(ax1, ax2, x_time, title)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper right', fontsize=9, framealpha=0.7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {save_path}')


def plot_all_layers(
    signal_i: np.ndarray,
    attn_list: list,
    label_str: str,
    jnr: int,
    save_path: str,
):
    """3×2 grid — one dual-axis panel per ConformerBlock (6 layers total).

    Args:
        signal_i: (8000,) I-channel samples.
        attn_list: list of 6 tensors, each (B, num_heads, T', T').
                   We take the first batch element.
        label_str: jamming type label.
        jnr: JNR value.
        save_path: output PNG path.
    """
    num_layers = len(attn_list)
    ncols = 3
    nrows = math.ceil(num_layers / ncols)

    x_time = np.arange(len(signal_i))
    sig_max = np.abs(signal_i).max()
    signal_norm = signal_i / sig_max if sig_max > 1e-8 else signal_i

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows), dpi=150)
    axes = axes.flatten()

    for i in range(num_layers):
        ax1 = axes[i]
        attn_1d = average_attention(attn_list[i][0])  # [0] = first batch elem
        attn_aligned = upsample_attention(attn_1d, target_len=len(signal_i))

        ax1.plot(x_time, signal_norm, color=SIGNAL_COLOR, alpha=0.85,
                 linewidth=0.6)
        ax2 = ax1.twinx()
        ax2.plot(x_time, attn_aligned, color=ATTN_LINE_COLOR, linewidth=1.2,
                 alpha=0.9)
        ax2.fill_between(x_time, 0, attn_aligned, color=ATTN_FILL_COLOR,
                         alpha=0.25)

        _style_axes(ax1, ax2, x_time, f'Layer {i + 1}')

    # Hide unused subplots
    for i in range(num_layers, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'{label_str}  |  All {num_layers} ConformerBlock Attention Layers  |  JNR=+{jnr} dB',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {save_path}')


def plot_per_head(
    signal_i: np.ndarray,
    attn_matrix: torch.Tensor,
    label_str: str,
    jnr: int,
    layer_idx: int,
    save_path: str,
):
    """2×2 grid — one panel per attention head for a specific layer.

    Args:
        signal_i: (8000,) I-channel samples.
        attn_matrix: (num_heads, T', T') for one layer, first batch element.
        label_str: jamming type label.
        jnr: JNR value.
        layer_idx: 0-based layer index.
        save_path: output PNG path.
    """
    num_heads = attn_matrix.shape[0]
    ncols = 2
    nrows = math.ceil(num_heads / ncols)

    x_time = np.arange(len(signal_i))
    sig_max = np.abs(signal_i).max()
    signal_norm = signal_i / sig_max if sig_max > 1e-8 else signal_i

    per_head = average_attention_per_head(attn_matrix)  # (num_heads, T')

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), dpi=150)
    axes = axes.flatten()

    for h in range(num_heads):
        ax1 = axes[h]
        head_color = HEAD_COLORS[h % len(HEAD_COLORS)]
        attn_aligned = upsample_attention(per_head[h], target_len=len(signal_i))

        ax1.plot(x_time, signal_norm, color=SIGNAL_COLOR, alpha=0.85,
                 linewidth=0.6)
        ax2 = ax1.twinx()
        ax2.plot(x_time, attn_aligned, color=head_color, linewidth=1.2,
                 alpha=0.9)
        ax2.fill_between(x_time, 0, attn_aligned, color=head_color, alpha=0.20)
        ax2.set_ylim(0, 1.05)

        ax1.set_title(f'Head {h + 1}', fontsize=11, fontweight='bold',
                      color=head_color)
        ax1.set_xlabel('Time (samples)', fontsize=9)
        ax1.set_ylabel('I-Channel Amplitude', fontsize=9, color=SIGNAL_COLOR)
        ax1.tick_params(axis='y', labelcolor=SIGNAL_COLOR, labelsize=8)
        ax1.tick_params(axis='x', labelsize=8)
        ax1.set_xlim(0, len(x_time) - 1)
        ax1.grid(True, color=GRID_COLOR, alpha=0.3, linestyle='--')

        ax2.set_ylabel('Attention', fontsize=9, color=head_color)
        ax2.tick_params(axis='y', labelcolor=head_color, labelsize=8)

    for i in range(num_heads, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'{label_str}  |  Layer {layer_idx + 1} — Per-Head Attention  |  JNR=+{jnr} dB',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {save_path}')


def plot_compare_types(
    samples: list,
    layer_idx: int,
    save_path: str,
):
    """N-row comparison: one row per jamming type, side-by-side.

    Args:
        samples: list of dicts, each with keys:
            signal_i, attn_list, label_str, jnr
        layer_idx: which ConformerBlock layer to visualise.
        save_path: output PNG path.
    """
    n = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3.5 * n), dpi=150)
    if n == 1:
        axes = [axes]

    for row, (sample, ax1) in enumerate(zip(samples, axes)):
        signal_i = sample['signal_i']
        attn_matrix = sample['attn_list'][layer_idx][0]  # [0] = first batch
        attn_1d = average_attention(attn_matrix)
        attn_aligned = upsample_attention(attn_1d, target_len=len(signal_i))

        x_time = np.arange(len(signal_i))
        sig_max = np.abs(signal_i).max()
        signal_norm = signal_i / sig_max if sig_max > 1e-8 else signal_i

        ax1.plot(x_time, signal_norm, color=SIGNAL_COLOR, alpha=0.85,
                 linewidth=0.7)
        ax2 = ax1.twinx()
        ax2.plot(x_time, attn_aligned, color=ATTN_LINE_COLOR, linewidth=1.2,
                 alpha=0.9)
        ax2.fill_between(x_time, 0, attn_aligned, color=ATTN_FILL_COLOR,
                         alpha=0.25)
        _style_axes(ax1, ax2, x_time,
                    f'{sample["label_str"]}  |  JNR=+{sample["jnr"]} dB')

    fig.suptitle(f'Attention Comparison Across Jamming Types  |  Layer {layer_idx + 1}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {save_path}')


# ═════════════════════════════════════════════════════════════════════════════
# Data extraction
# ═════════════════════════════════════════════════════════════════════════════

def extract_attention_data(
    model,
    data_loader: DataLoader,
    device: str,
    max_samples: int,
    target_classes: list = None,
) -> list:
    """Run inference and collect raw signals + attention weights.

    Args:
        model: ConformerForCZSL (or any Base1DCZSLModel subclass).
        data_loader: DataLoader yielding (time_signal, text_tokens, labels, metadata).
        device: torch device string.
        max_samples: stop after collecting this many samples.
        target_classes: if set, only keep samples whose jam_types intersect.

    Returns:
        list of dicts, each with:
            signal_i: (8000,) float32 — I-channel raw signal
            signal_q: (8000,) float32 — Q-channel raw signal
            attn_list: list of 6 tensors, each (1, num_heads, T', T')
            label_str: e.g. "DFTJ+AJ"
            metadata: original metadata dict
            jnr: JNR value as int
    """
    results = []
    model.eval()

    class_names = model.class_names

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Extracting attention'):
            if len(results) >= max_samples:
                break

            # collate_fn_conformer returns:
            # (time_signals, None, text_tokens, labels, texts, metadata_list)
            time_signal, _, text_tokens, labels, texts, meta_list = batch
            time_signal = time_signal.to(device)

            # Encode with attention
            _, attn_list = model.encode_signal(time_signal, return_attn=True)
            # attn_list: list of 6 tensors, each (B, num_heads, T', T')

            # Move signal to CPU for plotting
            signal_np = time_signal.cpu().numpy()  # (B, 2, 8000)

            for b in range(signal_np.shape[0]):
                if len(results) >= max_samples:
                    break

                meta = meta_list[b] if b < len(meta_list) else {}
                label_str = get_jamming_label(meta, class_names)

                # Filter by target classes
                if target_classes:
                    jam_types = meta.get('jam_types', [])
                    if isinstance(jam_types, str):
                        jam_types = [jam_types] if jam_types and jam_types != 'None' else []
                    if not any(t in target_classes for t in jam_types):
                        continue

                jnr = meta.get('JNR', 0)

                results.append({
                    'signal_i': signal_np[b, 0, :],   # I-channel
                    'signal_q': signal_np[b, 1, :],   # Q-channel
                    'attn_list': [a[b:b+1].cpu() for a in attn_list],  # keep batch dim
                    'label_str': label_str,
                    'metadata': meta,
                    'jnr': jnr,
                })

    print(f'  Collected {len(results)} samples')
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═════════════════════════════════════════════════════════════════════════════

def build_single_jnr_loader(config: dict, jnr: int, split: str,
                            class_names: list, batch_size: int = 8):
    """Build a DataLoader for a single JNR level."""
    data_config = config.get('data', {})
    base_path = data_config.get('base_path')
    time_var_name = data_config.get('time_var_name', 'all_times')
    time_seq_len = data_config.get('time_seq_len', 8000)
    num_workers = data_config.get('num_workers', 0)
    pin_memory = data_config.get('pin_memory', True)

    data_folder = os.path.join(base_path, f'JNR_+{jnr}')
    time_file = os.path.join(data_folder, f'{split}_echo_times.mat')
    metadata_file = os.path.join(data_folder, f'{split}_echo_metadata.json')

    if not os.path.exists(time_file):
        raise FileNotFoundError(f'Time file not found: {time_file}')
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f'Metadata file not found: {metadata_file}')

    ds = TimeSignalDataset(
        time_file=time_file,
        metadata_file=metadata_file,
        time_var_name=time_var_name,
        class_names=class_names,
        time_seq_len=time_seq_len,
    )

    tokenizer = TokenizerWrapper(model_type='clip')
    from functools import partial
    collate_fn = partial(collate_fn_conformer, tokenizer_fn=tokenizer, model_type='clip')

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    return loader


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='1D Conformer Attention Map Visualization')
    parser.add_argument('--config', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'config_1d.yaml'),
                        help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--backbone', type=str, default=None,
                        choices=['conformer', 'resnet1d', 'cnn1d'],
                        help='Override backbone in config')
    parser.add_argument('--dataset', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split')
    parser.add_argument('--jnr', type=int, default=10,
                        help='JNR level to visualise')
    parser.add_argument('--samples', type=int, default=3,
                        help='Max samples to extract')
    parser.add_argument('--classes', type=str, nargs='*', default=None,
                        help='Specific jamming classes (e.g. DFTJ AJ CSJ)')
    parser.add_argument('--mode', type=str, default='layers',
                        choices=['single', 'layers', 'heads', 'compare'],
                        help='Visualisation mode')
    parser.add_argument('--layer', type=int, default=5,
                        help='Layer index for single/heads/compare mode (0-based)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: results/attention_viz)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    args = parser.parse_args()

    # ---- Seed ----
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- Device ----
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # ---- Load config ----
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.backbone is not None:
        config.setdefault('model', {})['backbone'] = args.backbone
        print(f'Backbone override: {args.backbone}')

    # ---- Prepare output dir ----
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(__file__), '..',
                                       'results', 'attention_viz')
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Class names ----
    jamming_classes = config.get('jamming_classes', [])
    class_names = [jc['name'] if isinstance(jc, dict) else jc
                   for jc in jamming_classes]

    # ---- Build data loader (single JNR) ----
    print(f'\nLoading {args.dataset} data for JNR=+{args.jnr}...')
    loader = build_single_jnr_loader(
        config, args.jnr, args.dataset, class_names, args.batch_size)
    print(f'  {len(loader.dataset)} samples')

    # ---- Build model ----
    print('\nBuilding model...')
    model = create_1d_model(config, device)

    # ---- Load checkpoint ----
    print(f'Loading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)

    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)
    if skipped:
        print(f'  Skipped {len(skipped)} mismatched keys: {skipped[:5]}...')
    model.load_state_dict(filtered, strict=False)
    model.eval()
    model.to(device)
    print(f'  Loaded {len(filtered)} / {len(state_dict)} keys')

    # ---- Validate backbone supports attention extraction ----
    encoder_name = type(model.signal_encoder).__name__
    if 'conformer' not in encoder_name.lower():
        print(f'\nERROR: Attention extraction requires a Conformer backbone.')
        print(f'Current encoder: {encoder_name}')
        print(f'Use --backbone conformer or load a conformer checkpoint.')
        print(f'Supported backbones for attention viz: conformer')
        sys.exit(1)
    print(f'  Encoder: {encoder_name} (supports attention extraction)')

    # ---- Extract attention data ----
    print(f'\nExtracting attention (max {args.samples} samples)...')
    samples = extract_attention_data(
        model, loader, device,
        max_samples=args.samples,
        target_classes=args.classes,
    )

    if not samples:
        print('ERROR: No samples collected. Check --classes filter or data availability.')
        sys.exit(1)

    # ---- Generate figures ----
    print(f'\nGenerating {args.mode} visualisations...')
    num_layers = len(samples[0]['attn_list'])
    layer_idx = min(args.layer, num_layers - 1)

    for i, sample in enumerate(samples):
        prefix = f'sample{i:02d}_{sample["label_str"].replace("+", "_")}'

        if args.mode == 'single':
            save_path = os.path.join(
                args.output_dir,
                f'{prefix}_layer{layer_idx + 1}_jnr{args.jnr}.png')
            plot_single_layer(
                sample['signal_i'],
                average_attention(sample['attn_list'][layer_idx][0]),
                sample['label_str'], args.jnr, layer_idx, save_path)

        elif args.mode == 'layers':
            save_path = os.path.join(
                args.output_dir,
                f'{prefix}_all_layers_jnr{args.jnr}.png')
            plot_all_layers(
                sample['signal_i'], sample['attn_list'],
                sample['label_str'], args.jnr, save_path)

        elif args.mode == 'heads':
            save_path = os.path.join(
                args.output_dir,
                f'{prefix}_layer{layer_idx + 1}_heads_jnr{args.jnr}.png')
            plot_per_head(
                sample['signal_i'],
                sample['attn_list'][layer_idx][0],
                sample['label_str'], args.jnr, layer_idx, save_path)

        elif args.mode == 'compare':
            # compare uses ALL collected samples in one figure
            save_path = os.path.join(
                args.output_dir,
                f'compare_layer{layer_idx + 1}_jnr{args.jnr}.png')
            plot_compare_types(samples, layer_idx, save_path)
            break  # only one combined figure needed

    print(f'\nDone. Results saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
