"""
选取 10 张 STFT 样本，展示增强前后的对比图。

用法:
    python multi/visualize_augmentation.py

输出:
    multi/augmentation_examples.png — 10 行 x 2 列的对比图
"""

import os, sys, yaml, random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_config():
    config_path = Path(__file__).resolve().parent / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_augmentation(config):
    from multi.augmentation import STFTAugmentation
    aug_cfg = config.get('augmentation', {}).copy()
    # 强制开启四项增强以展示完整效果
    aug_cfg['specaugment'] = dict(aug_cfg.get('specaugment', {}), enabled=True, p=1.0)
    aug_cfg['patch_mask']  = dict(aug_cfg.get('patch_mask', {}),  enabled=True, p=1.0)
    aug_cfg['energy_mask'] = dict(aug_cfg.get('energy_mask', {}), enabled=True, p=1.0)
    aug_cfg['asymmetric']  = dict(aug_cfg.get('asymmetric', {}),  enabled=True)
    return STFTAugmentation(aug_cfg)


def main():
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    config = load_config()
    data_cfg = config.get('data', {})
    base_path = data_cfg.get('base_path')
    stft_suffix = data_cfg.get('stft_suffix', 'echo_stfts')
    jnr_start = data_cfg.get('jnr_start', 0)
    jnr_end = data_cfg.get('jnr_end', 20)
    jnr_step = data_cfg.get('jnr_step', 5)

    # ---- 收集 10 个样本 (尽量覆盖不同干扰类型) ----
    samples = []
    used_types = set()

    # 先扫一遍所有 JNR 文件夹，按类型建立索引
    type_to_indices = {}  # jam_type -> [(jnr, idx, stft_path, meta_path)]
    for jnr in range(jnr_start, jnr_end + 1, jnr_step):
        folder = f"JNR_{'+' if jnr >= 0 else ''}{jnr}"
        # 自动检测可用 split: train > val > test
        for split in ['train', 'val', 'test']:
            stft_path = os.path.join(base_path, folder, f'{split}_{stft_suffix}.mat')
            meta_path = os.path.join(base_path, folder, f'{split}_echo_metadata.json')
            if os.path.exists(stft_path) and os.path.exists(meta_path):
                break
        else:
            continue
        import h5py, json
        print(f"Scanning {folder}/{split}...")
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta_list = json.load(f)
        for idx, meta in enumerate(meta_list):
            jt = meta.get('jam_types', ['?'])
            key = tuple(sorted(jt)) if isinstance(jt, list) else str(jt)
            if key not in type_to_indices:
                type_to_indices[key] = []
            type_to_indices[key].append((jnr, idx, stft_path, meta_path))

    from torchvision import transforms
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
    norm = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)

    def load_one(idx, stft_path, meta_path):
        """加载并预处理单个 STFT 样本"""
        with h5py.File(stft_path, 'r') as f:
            stft_key = list(f.keys())[0]
            raw = f[stft_key][:, :, idx]
            stft_c = raw['real'] + 1j * raw['imag']
        mag = np.abs(stft_c)
        ref = np.percentile(mag, 99)
        if ref > 0:
            stft_c = stft_c / ref
        mag = np.abs(stft_c).T
        tensor = torch.from_numpy(np.stack([mag, mag, mag], axis=0)).float()
        if tensor.shape[-2:] != (224, 224):
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), size=(224, 224),
                mode='bilinear', align_corners=False
            ).squeeze(0)
        energy_map = tensor[0].clone()
        tensor_normed = norm(tensor)
        with open(meta_path, 'r', encoding='utf-8') as f_meta:
            meta = json.load(f_meta)[idx]
        return tensor, tensor_normed, energy_map, meta

    if not type_to_indices:
        print("ERROR: No STFT data found under base_path:", base_path)
        print("Check that the path exists and contains JNR_+*/ folders with train_{stft_suffix}.mat files")
        print("Current config data.base_path:", base_path)
        sys.exit(1)

    # 每种类型取 1 个，确保覆盖不同干扰类别
    for key in sorted(type_to_indices.keys()):
        if len(samples) >= 10:
            break
        jnr, idx, stft_path, meta_path = random.choice(type_to_indices[key])
        tensor, tensor_normed, energy_map, meta = load_one(idx, stft_path, meta_path)
        samples.append({
            'original': tensor, 'normed': tensor_normed,
            'energy_map': energy_map, 'meta': meta,
        })
        used_types.add(key)

    # 还不够 10 个则随机补
    while len(samples) < 10:
        key = random.choice(list(type_to_indices.keys()))
        jnr, idx, stft_path, meta_path = random.choice(type_to_indices[key])
        tensor, tensor_normed, energy_map, meta = load_one(idx, stft_path, meta_path)
        samples.append({
            'original': tensor, 'normed': tensor_normed,
            'energy_map': energy_map, 'meta': meta,
        })

    print(f"Collected {len(samples)} samples covering {len(used_types)} distinct types: {sorted(used_types)}")

    # ---- 构建增强流水线 ----
    aug = build_augmentation(config)

    # ---- 绘图 ----
    fig, axes = plt.subplots(10, 2, figsize=(8, 24))
    fig.suptitle('STFT Augmentation — 10 Samples (Left=Original, Right=Augmented)', fontsize=13, y=0.995)

    for i, s in enumerate(samples):
        # 原始图 (取第一通道, pre-CLIP-norm, 值域约 [0,1])
        orig = s['original'][0].numpy()

        # 增强后的图 (post-CLIP-norm 输入, 输出取第一通道)
        with torch.no_grad():
            aug_out, _ = aug(s['normed'].clone(), metadata=s['meta'], energy_map=s['energy_map'])
        # 反标准化回 [0,1] 范围以便可视化
        aug_np = aug_out[0].numpy()
        aug_np = np.clip(aug_np * 0.26862954 + 0.48145466, 0, 1)

        # 标注 jamming 类型
        jam = s['meta'].get('jam_types', ['?'])
        label = '+'.join(jam) if isinstance(jam, list) else str(jam)

        ax0, ax1 = axes[i][0], axes[i][1]
        ax0.imshow(orig, cmap='viridis', aspect='auto', origin='upper')
        ax0.set_title(f'[{label}] Original', fontsize=9)
        ax0.axis('off')

        ax1.imshow(aug_np, cmap='viridis', aspect='auto', origin='upper')
        ax1.set_title(f'[{label}] Augmented', fontsize=9)
        ax1.axis('off')

    plt.tight_layout()
    out_path = Path(__file__).resolve().parent / 'augmentation_examples.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
