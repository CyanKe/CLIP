"""
STFT 预处理脚本 — 一次性将 .mat 原始 STFT 转为预处理后的 .pt 文件

用法:
    python -m multi.tools.preprocess_stft --config multi/config.yaml

处理流程 (与 STFTDataset.__getitem__ 一致):
    1. 读取 structured complex64 → 3 通道 (mag 三份)
    2. per_sample p99 归一化
    3. Resize 到 224x224
    4. CLIP 标准化

输出:
    {data_folder}/train_echo_stfts_preprocessed.pt  (dict: tensors, labels, metadata_indices)
    {data_folder}/val_echo_stfts_preprocessed.pt
    {data_folder}/test_echo_stfts_preprocessed.pt

预处理后加载速度提升 ~10-20x (只需 torch.load 索引，无需 h5py/percentile/interpolate)
"""
import os
import sys
import json
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

clip_norm = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)


def preprocess_sample(raw_stft, image_size=224):
    """与 STFTDataset.__getitem__ 保持一致的预处理"""
    stft_complex = raw_stft['real'] + 1j * raw_stft['imag']
    mag = np.abs(stft_complex)
    ref = np.percentile(mag, 99)
    if ref > 0:
        stft_complex = stft_complex / ref
    stft_mag = np.abs(stft_complex).T
    stft_tensor = torch.from_numpy(np.stack([stft_mag, stft_mag, stft_mag], axis=0)).float()

    if stft_tensor.shape[-2:] != (image_size, image_size):
        stft_tensor = F.interpolate(
            stft_tensor.unsqueeze(0),
            size=(image_size, image_size),
            mode='bilinear', align_corners=False
        ).squeeze(0)

    stft_tensor = clip_norm(stft_tensor)
    return stft_tensor


def preprocess_split(stft_file: str, metadata_file: str, stft_var_name: str = 'all_stfts'):
    """预处理一个 split 的所有样本"""
    with h5py.File(stft_file, 'r') as f:
        num_samples = f[stft_var_name].shape[2]

    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)

    tensors = []
    labels = []
    metas = []

    with h5py.File(stft_file, 'r') as f:
        print(f"  Loading {stft_var_name} into RAM...")
        all_raw_stfts = f[stft_var_name][()]  # 一次性顺序读取，避免逐 slice 的随机磁盘 I/O

    for i in tqdm(range(num_samples), desc=f"  Processing {Path(stft_file).name}"):
        raw_stft = all_raw_stfts[:, :, i]  # 纯内存操作
        tensor = preprocess_sample(raw_stft)
        tensors.append(tensor)  # float32 精度，与在线处理一致

        meta = metadata_list[i] if i < len(metadata_list) else {}
        jam_types = meta.get('jam_types', [])
        if isinstance(jam_types, str):
            jam_types = [jam_types] if jam_types else []
        elif not isinstance(jam_types, list):
            jam_types = []
        metas.append({'jam_types': jam_types})

    all_tensors = torch.stack(tensors, dim=0)
    # labels 在加载时从 metadata + class_names 动态构建，这里不存
    return {'tensors': all_tensors, 'metadata': metas, 'num_samples': num_samples}


def main():
    parser = argparse.ArgumentParser(description="Preprocess STFT data for fast loading")
    parser.add_argument("--config", type=str, default="multi/config.yaml")
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    args = parser.parse_args()

    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    data_config = config.get('data', {})
    base_path = data_config.get('base_path')
    jnr_start = data_config.get('jnr_start', 0)
    jnr_end = data_config.get('jnr_end', 20)
    jnr_step = data_config.get('jnr_step', 5)
    stft_suffix = data_config.get('stft_suffix', 'echo_stfts')

    jnr_levels = list(range(jnr_start, jnr_end + 1, jnr_step))
    splits = [args.split] if args.split else ['train', 'val', 'test']

    for split_name in splits:
        print(f"\n{'='*60}")
        print(f"Preprocessing split: {split_name}")
        print(f"{'='*60}")

        for jnr in jnr_levels:
            jnr_folder = f"JNR_{'+' if jnr >= 0 else ''}{jnr}"
            data_folder = os.path.join(base_path, jnr_folder)

            stft_file = os.path.join(data_folder, f'{split_name}_{stft_suffix}.mat')
            metadata_file = os.path.join(data_folder, f'{split_name}_echo_metadata.json')

            if not os.path.exists(stft_file):
                print(f"  Skip {jnr_folder} — STFT file not found")
                continue

            output_file = os.path.join(data_folder, f'{split_name}_{stft_suffix}_preprocessed.pt')
            if os.path.exists(output_file):
                print(f"  Skip {jnr_folder} — already preprocessed ({output_file})")
                continue

            result = preprocess_split(stft_file, metadata_file)
            torch.save(result, output_file)
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"  Saved: {output_file} ({size_mb:.1f} MB, {result['num_samples']} samples)")


if __name__ == '__main__':
    main()
