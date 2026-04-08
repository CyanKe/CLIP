"""
检查 MATLAB STFT 数据的结构和数据类型
数据格式：[num_classes, 224, 225] complex single
"""
import h5py
import numpy as np

stft_file = r"D:\VScode\Jamming_signal_simulation\output\260403\JNR_+10\train_echo_stfts.mat"

print("Opening file:", stft_file)

with h5py.File(stft_file, 'r') as f:
    print("\nKeys:", list(f.keys()))

    stft_data = f['all_stfts']
    print("\nDataset shape:", stft_data.shape)
    print("Dataset dtype:", stft_data.dtype)

    # 读取第一个样本 - 形状应该是 [num_classes, 224, 225]
    # 但用户说是复数 single，所以需要正确解释
    sample_0 = stft_data[:, :, 0]
    print("\nRaw sample shape:", sample_0.shape)
    print("Raw sample dtype:", sample_0.dtype)

    # 如果原始是 complex64，h5py 可能存为 float32 的 view
    # 需要重新解释为复数
    if sample_0.dtype == np.float32:
        # 假设最后一维是实部虚部交替
        print("\nData is float32, need to convert to complex...")
        # 尝试重新 reshape
        print("Trying to interpret as complex...")
    elif sample_0.dtype.names:
        # 可能是 structured array
        print("\nStructured array, fields:", sample_0.dtype.names)
        # 尝试读取 real 和 imag 字段
        if 'real' in sample_0.dtype.names and 'imag' in sample_0.dtype.names:
            sample_complex = sample_0['real'] + 1j * sample_0['imag']
            print("Complex shape:", sample_complex.shape)
        else:
            # 可能是 r 和 i 字段
            first_field = sample_0.dtype.names[0]
            sample_complex = stft_data[:, :, 0].view(np.complex64)
            print("After view(complex64):", sample_complex.shape)
    else:
        # 直接就是复数
        sample_complex = stft_data[:, :, 0]
        print("\nData is already complex:", sample_complex.dtype)

    print("\nComplex sample shape:", sample_complex.shape)
    print("Complex min:", np.min(sample_complex))
    print("Complex max:", np.max(sample_complex))

    # 检查实部虚部
    print("\nReal - Min:", np.real(sample_complex).min(), "Max:", np.real(sample_complex).max())
    print("Imag - Min:", np.imag(sample_complex).min(), "Max:", np.imag(sample_complex).max())

    # 幅度
    stft_mag = np.abs(sample_complex)
    print("\nMagnitude - Min:", stft_mag.min(), "Max:", stft_mag.max())
    print("Magnitude - Mean:", stft_mag.mean())
    print("Magnitude - Std:", stft_mag.std())
