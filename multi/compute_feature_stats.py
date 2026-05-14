"""
计算多域特征的归一化统计量 (均值/标准差)

从训练集的 *_echo_features.json 文件计算各特征的全局均值和标准差，
输出 feature_normalization_stats.json 供 STFTDataset 使用。

Usage:
    python -m multi.compute_feature_stats --config multi/config.yaml
"""
import os
import json
import argparse
import numpy as np


# 特征键定义 (与 FEATURES_README.md 一致)
FEATURE_KEYS = [
    # 时域 (5)
    "time_domain.skewness",
    "time_domain.kurtosis",
    "time_domain.envelope_variation",
    "time_domain.modulation_bandwidth",
    "time_domain.modulation_rate",
    # 频域 (4)
    "freq_domain.spectral_skewness",
    "freq_domain.spectral_kurtosis",
    "freq_domain.carrier_factor",
    "freq_domain.awgn_factor",
    # 双谱 (2)
    "bispectrum.bispectrum_variance",
    "bispectrum.bispectrum_mean",
    # 小波 (8)
    "wavelet.variance",
    "wavelet.mean",
    "wavelet.max",
    "wavelet.scale_centroid",
    "wavelet.max_singular_value",
    "wavelet.central_moment_2",
    "wavelet.central_moment_3",
    "wavelet.central_moment_4",
    # 统计 (3)
    "statistical.shannon_entropy",
    "statistical.exponential_entropy",
    "statistical.norm_entropy",
]


def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def extract_feature_value(raw: dict, flat_key: str):
    """从嵌套 dict 中按 'domain.key' 取值"""
    parts = flat_key.split('.')
    val = raw
    for part in parts:
        val = val.get(part, None) if isinstance(val, dict) else None
        if val is None:
            return None
    if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
        return None
    return val


def compute_stats(config: dict, output_path: str):
    data_config = config.get('data', {})
    base_path = data_config.get('base_path')
    jnr_start = data_config.get('jnr_start', 0)
    jnr_end = data_config.get('jnr_end', 20)
    jnr_step = data_config.get('jnr_step', 5)

    jnr_levels = list(range(jnr_start, jnr_end + 1, jnr_step))

    # 收集所有特征值
    all_values = {k: [] for k in FEATURE_KEYS}

    for jnr in jnr_levels:
        jnr_folder = f"JNR_{'+' if jnr >= 0 else ''}{jnr}"
        features_file = os.path.join(base_path, jnr_folder, "train_echo_features.json")

        if not os.path.exists(features_file):
            print(f"  Skipping {jnr_folder}: features file not found")
            continue

        with open(features_file, 'r', encoding='utf-8') as f:
            features_list = json.load(f)

        print(f"  {jnr_folder}: {len(features_list)} samples")

        for raw in features_list:
            for key in FEATURE_KEYS:
                val = extract_feature_value(raw, key)
                if val is not None:
                    all_values[key].append(val)

    # 计算统计量
    stats = {}
    for key in FEATURE_KEYS:
        vals = np.array(all_values[key])
        if len(vals) > 0:
            stats[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }
        else:
            stats[key] = {"mean": 0.0, "std": 1.0}
            print(f"  Warning: no valid values for {key}")

    # 输出
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nSaved normalization stats to {output_path}")
    print(f"  Features: {len(stats)}")
    for key in FEATURE_KEYS:
        s = stats[key]
        print(f"    {key}: mean={s['mean']:.6f}, std={s['std']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Compute feature normalization stats")
    parser.add_argument("--config", type=str, default="multi/config.yaml", help="Config file path")
    parser.add_argument("--output", type=str, default="multi/feature_normalization_stats.json",
                        help="Output stats file path")
    args = parser.parse_args()

    config = load_config(args.config)
    compute_stats(config, args.output)


if __name__ == "__main__":
    main()
