"""
可视化脚本 - 对每个 JNR 下的 val 样本随机绘制 100 个 STFT 图
同时展示对应干扰类型和生成的文本描述
"""
import os
import sys
import yaml
import json
import random
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi.text_templates import JAM_TYPE_NAMES, generate_text_descriptions

# CLIP 反归一化参数（用于恢复原始 [0,1] 范围以便可视化）
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)


def denormalize_stft(tensor, stats=None):
    """
    将 CLIP 归一化的 tensor 反归一化回 [0,1] 附近的可视化范围
    tensor: (3, H, W) 的 torch tensor
    返回: (H, W) 的 numpy 数组 (使用 magnitude 通道)
    """
    arr = tensor.numpy()  # (3, H, W)
    # 反 CLIP 归一化
    arr = arr * CLIP_STD + CLIP_MEAN
    arr = np.clip(arr, 0, 1)
    # 使用 magnitude 通道做可视化
    mag = arr[2]  # (H, W)
    return mag


def get_jam_type_names_list(jam_types_int):
    """将整型干扰类型转为名称列表"""
    types = jam_types_int if isinstance(jam_types_int, list) else [jam_types_int]
    names = []
    for t in types:
        names.append(JAM_TYPE_NAMES.get(t, f'Type{t}'))
    return names


def visualize_val_samples(config_path, num_samples_per_jnr=100, output_dir=None):
    """
    对每个 JNR 的 val 样本随机采样并可视化 STFT 图

    Args:
        config_path: config.yaml 路径
        num_samples_per_jnr: 每个 JNR 采样数量
        output_dir: 输出目录
    """
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    data_config = config.get('data', {})
    base_path = data_config.get('base_path')
    jnr_start = data_config.get('jnr_start', 0)
    jnr_end = data_config.get('jnr_end', 20)
    jnr_step = data_config.get('jnr_step', 5)
    image_size = data_config.get('image_size', 224)

    # 加载归一化统计量
    stats_file = 'multi/normalization_stats.json'
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            normalization_stats = json.load(f)
    else:
        normalization_stats = None

    # JNR 级别
    jnr_levels = list(range(jnr_start, jnr_end + 1, jnr_step))
    print(f"JNR levels to process: {jnr_levels}")

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(output_dir, 'val_stft_visualizations')
    os.makedirs(output_dir, exist_ok=True)

    # 每个 JNR 分别处理
    for jnr in jnr_levels:
        jnr_folder = f"JNR_{'+' if jnr >= 0 else ''}{jnr}"
        data_folder = os.path.join(base_path, jnr_folder)

        stft_file = os.path.join(data_folder, 'val_echo_stfts.mat')
        label_file = os.path.join(data_folder, 'val_echo_label.mat')
        metadata_file = os.path.join(data_folder, 'val_echo_metadata.json')

        if not (os.path.exists(stft_file) and os.path.exists(label_file)):
            print(f"Warning: Val data not found for {jnr_folder}, skipping...")
            continue

        print(f"\nProcessing {jnr_folder}...")

        # 获取样本总数
        with h5py.File(stft_file, 'r') as f:
            num_total = f['all_stfts'].shape[2]

        # 随机采样
        indices = random.sample(range(num_total), min(num_samples_per_jnr, num_total))
        indices.sort()

        # 加载 metadata
        metadata = None
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

        # 每行5个图绘制
        ncols = 5
        nrows = (len(indices) + ncols - 1) // ncols

        fig = plt.figure(figsize=(ncols * 4, nrows * 3.5))
        fig.suptitle(f'JNR {jnr}dB - Val STFT Visualization ({len(indices)} samples)',
                     fontsize=14, fontweight='bold')
        gs = GridSpec(nrows, ncols, figure=fig, hspace=0.5, wspace=0.3)

        with h5py.File(stft_file, 'r') as stft_f, \
             h5py.File(label_file, 'r') as label_f:

            for idx, sample_idx in enumerate(indices):
                row = idx // ncols
                col = idx % ncols
                ax = fig.add_subplot(gs[row, col])

                # 读取 STFT 数据
                raw_stft = stft_f['all_stfts'][:, :, sample_idx]
                stft_complex = raw_stft['real'] + 1j * raw_stft['imag']
                stft_mag = np.abs(stft_complex)

                # 可视化幅度谱
                ax.imshow(stft_mag, cmap='jet', aspect='auto', origin='lower')
                ax.axis('off')

                # 获取标签
                label = label_f['all_label'][:, sample_idx]
                jam_types_int = np.where(label > 0)[0].tolist()
                # h5py 的 label 是 (num_classes, ) 且索引从 0 开始
                # 但 JAM_TYPE_NAMES 用的是 1-based，所以 +1
                jam_types_int = [int(t + 1) for t in jam_types_int]

                jam_names = get_jam_type_names_list(jam_types_int)

                # 获取 metadata 生成文本
                meta = None
                if metadata is not None and sample_idx < len(metadata):
                    meta = metadata[sample_idx].copy()
                    if 'jam_types' in meta and isinstance(meta['jam_types'], int):
                        meta['jam_types'] = [meta['jam_types']]

                text = generate_text_descriptions(meta, style='meta')
                text_template = generate_text_descriptions(meta, style='template')

                # 标题
                type_str = ' + '.join(jam_names) if jam_names else 'Unknown'
                ax.set_title(
                    f'Sample #{sample_idx}\n{type_str}',
                    fontsize=7,
                    fontweight='bold',
                    pad=2
                )

                # 底部添加生成的文本描述
                ax.text(
                    0.05, 0.05,
                    f'Meta: {text}\nTemplate: {text_template}',
                    transform=ax.transAxes,
                    fontsize=6,
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85),
                )

        plt.savefig(os.path.join(output_dir, f'val_stft_jnr{jnr}.png'),
                    dpi=150, bbox_inches='tight')
        print(f"  Saved: val_stft_jnr{jnr}.png ({len(indices)} samples)")
        plt.close(fig)

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
    visualize_val_samples(config_path, num_samples_per_jnr=100)
