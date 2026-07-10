"""
Draw all test set persistence spectrum images from base_path,
organized by category (jam type) name.

Reads persistence/config.yaml for configuration, iterates all JNR levels,
loads test persistence data + metadata, and saves each 2D spectrum as
a viridis heatmap PNG under {base_path}/persistence_spectrum_test_images/{category}/.
"""

import json
import os
import sys
import time

import h5py
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import yaml
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_image(img: np.ndarray,
                    low_percentile: float = 2,
                    high_percentile: float = 98) -> np.ndarray:
    """Per-image percentile-based normalization to [0, 1].

    Uses 2nd–98th percentile stretch to clip outliers while preserving
    the structure of the persistence probability distribution.
    """
    vmin = np.percentile(img, low_percentile)
    vmax = np.percentile(img, high_percentile)
    if vmax - vmin < 1e-8:
        return np.zeros_like(img)
    return np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)


def draw_and_save(img_2d: np.ndarray,
                  save_path: str,
                  title: str = None) -> None:
    """Draw a 2D persistence spectrum as a viridis heatmap and save as PNG.

    Args:
        img_2d: (H, W) float32 array in [0, 1]
        save_path: output PNG path
        title: optional title printed above the image
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(img_2d, cmap='viridis', aspect='auto', origin='lower')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=7, pad=2)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(save_path, dpi=75, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def category_name_from_meta(meta: dict) -> str:
    """Extract a filesystem-safe category name from metadata."""
    jam_types = meta.get('jam_types', 'unknown')
    if isinstance(jam_types, list):
        return '_'.join(sorted(jam_types))
    return str(jam_types)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- Load config --------------------------------------------------------
    config_path = Path(__file__).resolve().parent / 'config.yaml'
    print(f'Loading config from: {config_path}')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    data_cfg = config['data']
    base_path = data_cfg['base_path']
    jnr_start = data_cfg.get('jnr_start', 0)
    jnr_end = data_cfg.get('jnr_end', 20)
    jnr_step = data_cfg.get('jnr_step', 5)
    var_name = data_cfg.get('persistence_var_name', 'all_persistences')
    suffix = data_cfg.get('persistence_suffix', 'echo_persistences')

    jnr_levels = list(range(jnr_start, jnr_end + 1, jnr_step))
    print(f'base_path:  {base_path}')
    print(f'JNR levels: {jnr_levels}')

    # ---- Output root --------------------------------------------------------
    output_root = os.path.join(base_path, 'persistence_spectrum_test_images')
    os.makedirs(output_root, exist_ok=True)
    print(f'Output dir: {output_root}\n')

    total_saved = 0
    t_start = time.time()

    for jnr in jnr_levels:
        data_dir = os.path.join(base_path, f'JNR_+{jnr}')
        mat_path = os.path.join(data_dir, f'test_{suffix}.mat')
        meta_path = os.path.join(data_dir, 'test_echo_metadata.json')

        if not os.path.exists(mat_path):
            print(f'[SKIP] JNR_+{jnr}: missing {mat_path}')
            continue
        if not os.path.exists(meta_path):
            print(f'[SKIP] JNR_+{jnr}: missing {meta_path}')
            continue

        # ---- Load data ------------------------------------------------------
        with h5py.File(mat_path, 'r') as f:
            raw = f[var_name][:]            # (freq, power, N)
            # MATLAB column-major → C row-major: (freq, power, N) → (N, freq, power)
            data = np.transpose(raw, axes=(2, 1, 0)).astype(np.float32)

        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        N = data.shape[0]
        assert N == len(metadata), \
            f'Mismatch: {N} samples vs {len(metadata)} metadata entries'
        print(f'JNR_+{jnr}: {N} samples — processing...', end='', flush=True)

        jnr_saved = 0
        for i in range(N):
            meta = metadata[i]
            cat_name = category_name_from_meta(meta)
            sample_idx = meta.get('sample_idx', i)

            # Normalize and draw
            img = data[i]                     # (freq, power)
            img_norm = normalize_image(img)

            cat_dir = os.path.join(output_root, cat_name)
            os.makedirs(cat_dir, exist_ok=True)

            fname = f'{cat_name}_JNR{jnr}_idx{sample_idx:04d}.png'
            save_path = os.path.join(cat_dir, fname)

            title = f'{cat_name} | JNR=+{jnr}dB | #{sample_idx}'
            draw_and_save(img_norm, save_path, title=title)
            jnr_saved += 1

        total_saved += jnr_saved
        elapsed = time.time() - t_start
        print(f' done ({jnr_saved} images) [{elapsed:.1f}s]')

    # ---- Summary ------------------------------------------------------------
    total_elapsed = time.time() - t_start
    categories = sorted(os.listdir(output_root))
    print(f'\n{"="*60}')
    print(f'Done! {total_saved} images in {len(categories)} categories')
    print(f'Time: {total_elapsed:.1f}s ({total_saved / total_elapsed:.1f} img/s)')
    print(f'Output: {output_root}')
    for cat in categories:
        cat_dir = os.path.join(output_root, cat)
        n_files = len(os.listdir(cat_dir))
        print(f'  {cat}: {n_files} images')


if __name__ == '__main__':
    main()
