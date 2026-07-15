#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stft_server.py — Lightweight HTTP server that reads STFT from .mat on demand.

Usage:
    python multi/_archive/dev/stft_server.py --config multi/config.yaml --port 8765
    python multi/_archive/dev/stft_server.py --base D:/path/to/data --port 8765

Endpoints:
    GET /stft?jnr=10&idx=42&type=raw   → raw |STFT| magnitude PNG
    GET /stft?jnr=10&idx=42&type=clip  → CLIP-preprocessed PNG
    GET /health                          → OK
"""

import argparse
import io
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np
import h5py
import yaml
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ============================================================
# STFT loader — replicates STFTDataset preprocessing
# ============================================================

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class STFTReader:
    """Reads individual STFT samples from .mat files on demand, with caching."""

    def __init__(self, base_path, stft_suffix='echo_stfts', image_size=224):
        self.base_path = base_path
        self.stft_suffix = stft_suffix
        self.image_size = image_size
        self._cache = {}  # jnr → (h5_file, num_samples)

    def _get_file(self, jnr):
        if jnr not in self._cache:
            jnr_str = f"+{jnr}" if jnr >= 0 else str(jnr)
            path = os.path.join(self.base_path, f"JNR_{jnr_str}",
                                f"test_{self.stft_suffix}.mat")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data not found: {path}")
            f = h5py.File(path, 'r')
            n = f['all_stfts'].shape[2]
            self._cache[jnr] = (f, n)
        return self._cache[jnr]

    def get_raw_mag(self, jnr, idx):
        """Return raw per-sample-normalized magnitude as [H, W] uint8 numpy array."""
        h5f, n_samples = self._get_file(jnr)
        if idx < 0 or idx >= n_samples:
            raise IndexError(f"idx {idx} out of range [0, {n_samples})")

        raw = h5f['all_stfts'][:, :, idx]
        stft_cpx = raw['real'] + 1j * raw['imag']
        mag = np.abs(stft_cpx)

        # Per-sample P99 normalization
        ref = np.percentile(mag, 99)
        if ref > 0:
            mag = mag / ref
        mag = np.clip(mag, 0, 1)

        # Transpose: [freq, time] → [time, freq] for display
        mag = mag.T

        # Resize to target size
        mag_t = torch.from_numpy(mag).float().unsqueeze(0).unsqueeze(0)
        mag_t = F.interpolate(mag_t, size=(self.image_size, self.image_size),
                              mode='bilinear', align_corners=False)
        return mag_t.squeeze().numpy()

    def get_clip_input(self, jnr, idx):
        """Return CLIP-preprocessed 3-channel average as [H, W] in [0,1]."""
        raw_mag = self.get_raw_mag(jnr, idx)  # already [0,1], [H, W]

        # Build 3-channel [mag, mag, mag]
        ch3 = np.stack([raw_mag, raw_mag, raw_mag], axis=0)  # [3, H, W]
        ch3_t = torch.from_numpy(ch3).float()

        # CLIP normalization
        mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
        std = torch.tensor(CLIP_STD).view(3, 1, 1)
        ch3_t = (ch3_t - mean) / std

        # Average channels, scale to [0,1]
        avg = ch3_t.mean(dim=0).numpy()
        avg = (avg - avg.min()) / (avg.max() - avg.min() + 1e-8)
        return avg


# ============================================================
# HTTP handler
# ============================================================

def _array_to_png(arr):
    """Convert [0,1] numpy array to PNG bytes using matplotlib inferno colormap."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.imshow(arr, cmap='inferno', aspect='auto', origin='lower',
              vmin=0, vmax=1, interpolation='bilinear')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, pad_inches=0, facecolor='black')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


class STFTRequestHandler(BaseHTTPRequestHandler):
    reader = None  # Set before starting server

    def log_message(self, format, *args):
        print(f"  {args[0]}")  # Compact logging

    def _send_png(self, png_bytes):
        self.send_response(200)
        self.send_header('Content-Type', 'image/png')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', len(png_bytes))
        self.end_headers()
        self.wfile.write(png_bytes)

    def _send_error(self, code, msg):
        self.send_response(code)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(msg.encode())

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b'OK')
            return

        if parsed.path == '/stft':
            qs = parse_qs(parsed.query)
            try:
                jnr = int(qs.get('jnr', [None])[0])
                idx = int(qs.get('idx', [None])[0])
                stft_type = qs.get('type', ['raw'])[0]
            except (TypeError, ValueError):
                self._send_error(400, 'Missing or invalid jnr/idx')
                return

            try:
                if stft_type == 'clip':
                    arr = self.reader.get_clip_input(jnr, idx)
                else:
                    arr = self.reader.get_raw_mag(jnr, idx)
                png = _array_to_png(arr)
                self._send_png(png)
            except FileNotFoundError as e:
                self._send_error(404, str(e))
            except IndexError as e:
                self._send_error(400, str(e))
            except Exception as e:
                self._send_error(500, str(e))
            return

        self._send_error(404, 'Not found')


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='STFT on-demand HTTP server')
    parser.add_argument('--config', type=str, default='multi/config.yaml',
                        help='Path to config.yaml')
    parser.add_argument('--base', type=str, default=None,
                        help='Base data path (overrides config)')
    parser.add_argument('--port', type=int, default=8765,
                        help='Server port')
    parser.add_argument('--dataset', type=str, default='test',
                        help='Dataset split to serve')
    args = parser.parse_args()

    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

    data_config = config.get('data', {})
    base_path = args.base or data_config.get('base_path', '')
    stft_suffix = data_config.get('stft_suffix', 'echo_stfts')
    image_size = data_config.get('image_size', 224)

    if not base_path:
        print("ERROR: No data path set. Use --base or set data.base_path in config.")
        sys.exit(1)

    print(f"STFT Server starting...")
    print(f"  Data:   {base_path}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Port:   {args.port}")
    print(f"  Image:  {image_size}x{image_size}")

    STFTRequestHandler.reader = STFTReader(
        base_path, stft_suffix, image_size,
    )

    server = HTTPServer(('0.0.0.0', args.port), STFTRequestHandler)
    print(f"\n  Ready at http://localhost:{args.port}")
    print(f"  Endpoint: /stft?jnr=10&idx=42&type=raw")
    print(f"  Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == '__main__':
    main()
