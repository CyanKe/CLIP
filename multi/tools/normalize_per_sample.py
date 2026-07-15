"""
样本级 STFT 数据归一化
每个样本独立归一化，消除不同 JNR 级别间的能量差异
"""
import os
import sys
import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm


def normalize_sample(stft_complex: np.ndarray, method: str = 'p99') -> np.ndarray:
    """
    对单个 STFT 样本进行归一化

    Args:
        stft_complex: 复数 STFT 数据 [freq_bins, time_frames]
        method: 归一化方法
            - 'max': 按最大幅度归一化
            - 'p99': 按 99 分位数归一化（更鲁棒，避免极值影响）
            - 'p95': 按 95 分位数归一化

    Returns:
        归一化后的复数 STFT 数据
    """
    mag = np.abs(stft_complex)

    if method == 'max':
        ref = np.max(mag)
    elif method == 'p99':
        ref = np.percentile(mag, 99)
    elif method == 'p95':
        ref = np.percentile(mag, 95)
    else:
        raise ValueError(f"Unknown method: {method}")

    if ref > 0:
        return stft_complex / ref
    return stft_complex


def process_stft_file(
    input_path: str,
    output_path: str,
    method: str = 'p99',
    stft_var_name: str = 'all_stfts'
):
    """
    处理单个 STFT 文件，对所有样本进行样本级归一化

    Args:
        input_path: 输入 .mat 文件路径
        output_path: 输出 .mat 文件路径
        method: 归一化方法
        stft_var_name: STFT 数据在 .mat 文件中的变量名
    """
    print(f"Processing: {input_path}")

    with h5py.File(input_path, 'r') as f_in:
        stft_data = f_in[stft_var_name]
        shape = stft_data.shape
        print(f"  Shape: {shape} [freq, time, samples]")

        # 创建输出文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with h5py.File(output_path, 'w') as f_out:
            # 创建 structured complex64 数据集
            complex_dtype = np.dtype([('real', '<f4'), ('imag', '<f4')])
            normalized = f_out.create_dataset(
                stft_var_name,
                shape=shape,
                dtype=complex_dtype
            )

            # 逐样本处理
            for i in tqdm(range(shape[2]), desc="  Normalizing"):
                # 读取原始数据
                raw = stft_data[:, :, i]
                stft_complex = raw['real'] + 1j * raw['imag']

                # 样本级归一化
                stft_norm = normalize_sample(stft_complex, method=method)

                # 写入 structured complex 格式
                normalized[:, :, i] = np.stack([
                    np.real(stft_norm).astype(np.float32),
                    np.imag(stft_norm).astype(np.float32)
                ], axis=-1).view(complex_dtype)[:, :, 0]

    print(f"  Saved to: {output_path}")


def process_all_jnr(
    base_path: str,
    jnr_levels: list,
    output_base: str,
    method: str = 'p99',
    splits: list = None,
    stft_var_name: str = 'all_stfts',
    stft_suffix: str = 'echo_stfts'
):
    """
    处理所有 JNR 级别的数据

    Args:
        base_path: 数据根目录
        jnr_levels: JNR 级别列表
        output_base: 输出根目录
        method: 归一化方法
        splits: 数据划分列表，默认 ['train', 'val', 'test']
        stft_var_name: STFT 变量名
    """
    if splits is None:
        splits = ['train', 'val', 'test']

    for jnr in jnr_levels:
        jnr_folder = f"JNR_{jnr}"
        print(f"\n{'='*50}")
        print(f"Processing JNR_{jnr}")
        print(f"{'='*50}")

        for split in splits:
            input_file = os.path.join(base_path, jnr_folder, f'{split}_{stft_suffix}.mat')

            if not os.path.exists(input_file):
                print(f"  [Skip] {input_file} not found")
                continue

            output_file = os.path.join(output_base, jnr_folder, f'{split}_{stft_suffix}.mat')

            process_stft_file(
                input_path=input_file,
                output_path=output_file,
                method=method,
                stft_var_name=stft_var_name
            )


def verify_normalization(output_path: str, stft_var_name: str = 'all_stfts'):
    """验证归一化结果"""
    print(f"\nVerifying: {output_path}")

    with h5py.File(output_path, 'r') as f:
        stft_data = f[stft_var_name]
        shape = stft_data.shape

        # 随机抽样验证
        sample_indices = np.random.choice(shape[2], min(10, shape[2]), replace=False)

        for idx in sample_indices:
            raw = stft_data[:, :, idx]
            stft_complex = raw['real'] + 1j * raw['imag']
            mag = np.abs(stft_complex)

            print(f"  Sample {idx}: mag_max={np.max(mag):.4f}, mag_p99={np.percentile(mag, 99):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample-level STFT normalization")
    parser.add_argument(
        "--base_path",
        type=str,
        default="D:/VScode/Jamming_signal_simulation/output/260421"
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="D:/VScode/Jamming_signal_simulation/output/260421_normalized"
    )
    parser.add_argument(
        "--jnr_levels",
        type=str,
        nargs="+",
        default=["+10", "+5", "0", "-5", "-10"]
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=['max', 'p99', 'p95'],
        default='p99',
        help="Normalization reference: max, p99 (99th percentile), or p95"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"]
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify normalization results after processing"
    )
    parser.add_argument(
        "--stft_suffix",
        type=str,
        default="echo_stfts",
        help="STFT filename suffix"
    )

    args = parser.parse_args()

    process_all_jnr(
        base_path=args.base_path,
        jnr_levels=args.jnr_levels,
        output_base=args.output_base,
        method=args.method,
        splits=args.splits,
        stft_suffix=args.stft_suffix
    )

    if args.verify:
        # 验证第一个 JNR 的 train 数据
        first_jnr = args.jnr_levels[0]
        verify_path = os.path.join(
            args.output_base,
            f"JNR_{first_jnr}",
            f'train_{args.stft_suffix}.mat'
        )
        if os.path.exists(verify_path):
            verify_normalization(verify_path)
