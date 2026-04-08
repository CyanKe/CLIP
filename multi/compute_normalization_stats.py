"""
统计训练集的全局分位数，用于 STFT 数据归一化
数据格式：[225, 224, N] structured complex64 (real + imag)
"""
import os
import json
import h5py
import numpy as np
from pathlib import Path


def compute_normalization_stats(
    base_path: str,
    jnr_levels: list,
    stft_var_name: str = 'all_stfts',
    max_samples: int = 5000
):
    """
    计算训练集的全局统计量（用于归一化）

    数据结构：[freq_bins, time_frames, N_samples] structured complex64
    """
    all_real = []
    all_imag = []
    all_mag = []

    for jnr in jnr_levels:
        jnr_folder = f"JNR_{jnr}"
        stft_file = os.path.join(base_path, jnr_folder, 'train_echo_stfts.mat')

        if not os.path.exists(stft_file):
            print(f"Warning: {stft_file} not found, skipping...")
            continue

        print(f"Loading {stft_file}...")
        with h5py.File(stft_file, 'r') as f:
            stft_data = f[stft_var_name]
            total_samples = stft_data.shape[2]

            # 采样
            if total_samples > max_samples // len(jnr_levels):
                step = total_samples // (max_samples // len(jnr_levels))
                indices = range(0, total_samples, step)
            else:
                indices = range(total_samples)

            for i in indices:
                # 读取 structured complex 数据
                raw = stft_data[:, :, i]
                # 转换为复数
                stft_complex = raw['real'] + 1j * raw['imag']

                # 收集实部、虚部、幅度
                all_real.append(np.real(stft_complex).flatten())
                all_imag.append(np.imag(stft_complex).flatten())
                all_mag.append(np.abs(stft_complex).flatten())

                if len(all_real) % 1000 == 0:
                    print(f"  Processed {len(all_real)} samples...")

    # 合并所有数据
    all_real = np.concatenate(all_real)
    all_imag = np.concatenate(all_imag)
    all_mag = np.concatenate(all_mag)

    print(f"\nTotal values: {len(all_real):,} per channel")

    # 计算统计量
    stats = {
        # 实部
        'real_min': float(np.min(all_real)),
        'real_max': float(np.max(all_real)),
        'real_mean': float(np.mean(all_real)),
        'real_std': float(np.std(all_real)),
        'real_p50': float(np.percentile(all_real, 50)),
        'real_p99': float(np.percentile(all_real, 99)),

        # 虚部
        'imag_min': float(np.min(all_imag)),
        'imag_max': float(np.max(all_imag)),
        'imag_mean': float(np.mean(all_imag)),
        'imag_std': float(np.std(all_imag)),
        'imag_p50': float(np.percentile(all_imag, 50)),
        'imag_p99': float(np.percentile(all_imag, 99)),

        # 幅度（用于归一化）
        'mag_min': float(np.min(all_mag)),
        'mag_max': float(np.max(all_mag)),
        'mag_mean': float(np.mean(all_mag)),
        'mag_std': float(np.std(all_mag)),
        'mag_p50': float(np.percentile(all_mag, 50)),
        'mag_p90': float(np.percentile(all_mag, 90)),
        'mag_p95': float(np.percentile(all_mag, 95)),
        'mag_p99': float(np.percentile(all_mag, 99)),
    }

    # 打印结果
    print("\n" + "=" * 50)
    print("全局统计量")
    print("=" * 50)

    print("\n实部:")
    print(f"  Range: [{stats['real_min']:.2f}, {stats['real_max']:.2f}]")
    print(f"  Mean: {stats['real_mean']:.2f}, Std: {stats['real_std']:.2f}")
    print(f"  P50: {stats['real_p50']:.2f}, P99: {stats['real_p99']:.2f}")

    print("\n虚部:")
    print(f"  Range: [{stats['imag_min']:.2f}, {stats['imag_max']:.2f}]")
    print(f"  Mean: {stats['imag_mean']:.2f}, Std: {stats['imag_std']:.2f}")
    print(f"  P50: {stats['imag_p50']:.2f}, P99: {stats['imag_p99']:.2f}")

    print("\n幅度（推荐用于归一化参考）:")
    print(f"  Range: [{stats['mag_min']:.2f}, {stats['mag_max']:.2f}]")
    print(f"  Mean: {stats['mag_mean']:.2f}, Std: {stats['mag_std']:.2f}")
    print(f"  P50: {stats['mag_p50']:.2f}, P90: {stats['mag_p90']:.2f}")
    print(f"  P95: {stats['mag_p95']:.2f}, P99: {stats['mag_p99']:.2f}")

    return stats


def save_stats(stats: dict, output_path: str):
    """保存统计量到 JSON 文件"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print(f"\nStats saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute normalization statistics for STFT data")
    parser.add_argument(
        "--base_path",
        type=str,
        default="D:/VScode/Jamming_signal_simulation/output/260403"
    )
    parser.add_argument(
        "--jnr_levels",
        type=str,
        nargs="+",
        default=["+10"]
    )
    parser.add_argument(
        "--output",
        type=str,
        default="multi/normalization_stats.json"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5000
    )

    args = parser.parse_args()

    stats = compute_normalization_stats(
        base_path=args.base_path,
        jnr_levels=args.jnr_levels,
        max_samples=args.max_samples
    )

    save_stats(stats, args.output)
