"""
分析 STFT 数据的分布，检查是否需要 log 变换
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt

# 读取一个样本
stft_file = r"D:\VScode\Jamming_signal_simulation\output\260403\JNR_+10\train_echo_stfts.mat"

with h5py.File(stft_file, 'r') as f:
    stft_data = f['all_stfts']

    # 取前 10 个样本分析
    samples = []
    for i in range(min(10, stft_data.shape[2])):
        stft_complex = stft_data[:, :, i].view(np.complex128)
        stft_mag = np.abs(stft_complex)
        samples.append(stft_mag.flatten())

    samples = np.concatenate(samples)

print(f"样本数：{len(samples):,}")
print(f"Min: {np.min(samples):.4f}")
print(f"Max: {np.max(samples):.4f}")
print(f"Mean: {np.mean(samples):.4f}")
print(f"Std: {np.std(samples):.4f}")

# 分位数
print("\n分位数:")
for p in [50, 90, 95, 99, 99.9]:
    print(f"  {p}%: {np.percentile(samples, p):.4f}")

# 检查异常值比例
threshold_99 = np.percentile(samples, 99)
threshold_999 = np.percentile(samples, 99.9)
print(f"\n>99% 分位数的样本比例：{np.sum(samples > threshold_99) / len(samples) * 100:.2f}%")
print(f">99.9% 分位数的样本比例：{np.sum(samples > threshold_999) / len(samples) * 100:.2f}%")

# log 变换后的统计
log_samples = np.log1p(samples)
print(f"\nLog 变换后:")
print(f"Min: {np.min(log_samples):.4f}")
print(f"Max: {np.max(log_samples):.4f}")
print(f"Mean: {np.mean(log_samples):.4f}")
print(f"Std: {np.std(log_samples):.4f}")

# 画图
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(samples[samples < np.percentile(samples, 99)], bins=100, alpha=0.7)
axes[0].set_xlabel('Magnitude (clipped at 99%)')
axes[0].set_ylabel('Count')
axes[0].set_title('Original Distribution (99% clipped)')

axes[1].hist(log_samples, bins=100, alpha=0.7, color='orange')
axes[1].set_xlabel('Log(Magnitude)')
axes[1].set_ylabel('Count')
axes[1].set_title('Log-Transformed Distribution')

plt.tight_layout()
plt.savefig('multi/stft_distribution.png', dpi=150)
print("\nDistribution plot saved to multi/stft_distribution.png")
plt.close()
