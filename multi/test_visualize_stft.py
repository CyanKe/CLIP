"""
可视化测试脚本 - 展示STFT图和文本描述

从config.yaml读取路径，展示所有数据的STFT图和对应的文本描述
"""
import os
import sys
import yaml
import numpy as np
import h5py
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi.metadata_template import generate_short_description, JAM_TYPE_NAMES


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def dereference_hdf5_value(f, ref_array):
    """解引用HDF5 object reference数组"""
    result = []
    for ref in ref_array:
        if isinstance(ref, h5py.Reference):
            data = f[ref]
            if isinstance(data, h5py.Dataset):
                val = data[()]
                if hasattr(val, 'item'):
                    val = val.item()
                result.append(float(val) if isinstance(val, (np.floating, float)) else int(val))
            elif isinstance(data, h5py.Group):
                # 返回group本身，后续处理
                result.append(data)
        else:
            result.append(ref)
    return result


def parse_metadata_h5py(meta_group, index, h5file):
    """解析h5py Group格式的metadata"""
    metadata_dict = {}

    # sample_idx
    if 'sample_idx' in meta_group:
        refs = meta_group['sample_idx'][index]
        vals = dereference_hdf5_value(h5file, refs)
        metadata_dict['sample_idx'] = int(vals[0]) if vals else index

    # jam_types
    if 'jam_types' in meta_group:
        refs = meta_group['jam_types'][index]
        vals = dereference_hdf5_value(h5file, refs)
        metadata_dict['jam_types'] = [int(v) for v in vals if v > 0]

    # JNR
    if 'JNR' in meta_group:
        refs = meta_group['JNR'][index]
        vals = dereference_hdf5_value(h5file, refs)
        metadata_dict['JNR'] = float(vals[0]) if vals else 0.0

    # pos
    if 'pos' in meta_group:
        refs = meta_group['pos'][index]
        vals = dereference_hdf5_value(h5file, refs)
        metadata_dict['pos'] = int(vals[0]) if vals else 0

    # jam_params - 嵌套Group
    metadata_dict['jam_params'] = {}
    if 'jam_params' in meta_group:
        refs = meta_group['jam_params'][index]
        for ref in refs:
            if isinstance(ref, h5py.Reference):
                params_group = h5file[ref]
                if isinstance(params_group, h5py.Group):
                    for field in params_group.keys():
                        field_data = params_group[field]
                        if isinstance(field_data, h5py.Dataset):
                            val = field_data[()]
                            if hasattr(val, 'item'):
                                val = val.item()
                            # 处理字符串
                            if isinstance(val, bytes):
                                val = val.decode('utf-8')
                            elif isinstance(val, str):
                                val = val
                            else:
                                val = float(val) if isinstance(val, (np.floating, float)) else val
                            metadata_dict['jam_params'][field] = val

    return metadata_dict


def load_data_from_config(config, split='test'):
    """根据配置加载数据"""
    data_config = config.get('data', {})
    base_path = data_config.get('base_path')

    # JNR范围
    jnr_start = data_config.get('jnr_start', 0)
    jnr_end = data_config.get('jnr_end', 40)
    jnr_step = data_config.get('jnr_step', 10)

    jnr_numbers = range(jnr_start, jnr_end + 1, jnr_step)
    jnr_levels = [f"+{jnr}" if jnr >= 0 else str(jnr) for jnr in jnr_numbers]

    # 文件名映射
    split_files = {
        'train': ('train_echo_stfts.mat', 'train_echo_label.mat', 'train_echo_metadata.mat'),
        'val': ('val_echo_stfts.mat', 'val_echo_label.mat', 'val_echo_metadata.mat'),
        'test': ('test_echo_stfts.mat', 'test_echo_label.mat', 'test_echo_metadata.mat')
    }

    stft_name, label_name, metadata_name = split_files[split]

    # 收集所有数据
    all_stfts = []
    all_labels = []
    all_metadata = []  # 存储(metadata_group, h5file) tuple
    all_jnr = []

    for jnr in jnr_levels:
        jnr_folder = f"JNR_{jnr}"
        data_path = os.path.join(base_path, jnr_folder)

        stft_file = os.path.join(data_path, stft_name)
        label_file = os.path.join(data_path, label_name)
        metadata_file = os.path.join(data_path, metadata_name)

        if not os.path.exists(stft_file):
            print(f"Skip JNR {jnr}: {stft_name} not found")
            continue

        # 加载STFT - 结构化数组格式
        with h5py.File(stft_file, 'r') as f:
            stft_data = f['all_stfts']
            # 结构化数组: (freq, time, samples) with fields 'real' and 'imag'
            n_samples = stft_data.shape[2]
            stft_real = stft_data['real'][:]
            stft_imag = stft_data['imag'][:]
            stft_complex = stft_real + 1j * stft_imag
            stft_combined = stft_complex.transpose(1, 0, 2)

        # 加载标签
        with h5py.File(label_file, 'r') as f:
            label_data = f['all_label'][:]

        # 加载metadata - 保持文件句柄打开以支持解引用
        metadata_handle = None
        metadata_group = None
        if os.path.exists(metadata_file):
            try:
                metadata_handle = h5py.File(metadata_file, 'r')
                metadata_group = metadata_handle['all_metadata']
                print(f"Loaded metadata for JNR {jnr}")
            except Exception as e:
                print(f"Warning: Failed to load metadata for JNR {jnr}: {e}")

        all_stfts.append(stft_combined)
        all_labels.append(label_data)
        all_metadata.append((metadata_group, metadata_handle))
        all_jnr.extend([jnr] * n_samples)

        print(f"JNR {jnr}: {n_samples} samples")

    if not all_stfts:
        raise ValueError("No data found!")

    # 合并
    combined_stfts = np.concatenate(all_stfts, axis=2)
    combined_labels = np.concatenate(all_labels, axis=1)

    return combined_stfts, combined_labels, all_metadata, all_jnr, jnr_levels


def visualize_stft(stft_complex, title="", ax=None, cmap='viridis'):
    """可视化STFT图 - 输入为(time, freq)的复数数组"""
    # 转换为幅度
    magnitude = np.abs(stft_complex)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(magnitude, aspect='auto', origin='lower', cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')

    return im


def get_jam_type_names(label_vector):
    """根据标签向量获取干扰类型名称"""
    active_indices = np.where(label_vector == 1)[0]
    names = []
    for idx in active_indices:
        jam_type = idx + 1  # 类别从1开始
        name = JAM_TYPE_NAMES.get(jam_type, f'Type{jam_type}')
        names.append(name)
    return names


def main():
    """主函数"""
    # 加载配置
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'multi', 'config.yaml')
    config = load_config(config_path)

    print("=" * 70)
    print("STFT可视化与文本描述测试")
    print("=" * 70)
    print(f"\n配置路径: {config_path}")
    print(f"数据路径: {config['data']['base_path']}")

    # 加载测试数据
    print(f"\n加载测试数据...")
    stfts, labels, metadata_list, jnr_tags, jnr_levels = load_data_from_config(config, split='test')

    n_samples = stfts.shape[2]
    n_classes = labels.shape[0]
    print(f"\n总样本数: {n_samples}")
    print(f"类别数: {n_classes}")
    print(f"JNR级别: {jnr_levels}")

    # 计算每个metadata段的样本数
    metadata_offsets = []
    offset = 0
    for meta_group, meta_handle in metadata_list:
        if meta_group is not None and meta_handle is not None:
            try:
                n = meta_group['sample_idx'].shape[0]
            except Exception:
                n = 0
        else:
            n = 0
        metadata_offsets.append((offset, offset + n))
        offset += n

    # 保存每个样本的单独图片
    print(f"\n保存 {n_samples} 个样本到 result/test_stft...")
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', 'test_stft')
    os.makedirs(save_dir, exist_ok=True)

    for idx in range(n_samples):
        # 获取STFT数据
        stft_slice = stfts[:, :, idx]
        label_vec = labels[:, idx]
        jam_names = get_jam_type_names(label_vec)
        jnr = jnr_tags[idx]

        # 获取metadata
        metadata_dict = None
        for meta_idx, (meta_start, meta_end) in enumerate(metadata_offsets):
            if meta_start <= idx < meta_end:
                meta_group, meta_handle = metadata_list[meta_idx]
                if meta_group is not None and meta_handle is not None:
                    local_idx = idx - meta_start
                    try:
                        metadata_dict = parse_metadata_h5py(meta_group, local_idx, meta_handle)
                    except Exception:
                        pass
                break

        # 生成文本描述
        if metadata_dict is not None and len(metadata_dict.get('jam_types', [])) > 0:
            text_desc = generate_short_description(metadata_dict, style='type_visual')
        else:
            text_desc = f"a radar signal with {' + '.join(jam_names)}"

        # 创建单个样本的图片
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Sample {idx} | JNR={jnr} | Types: {" + ".join(jam_names)}',
                     fontsize=12, fontweight='bold')

        # 左侧：STFT图
        magnitude = np.abs(stft_slice)
        im = axes[0].imshow(magnitude, aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title('STFT Magnitude')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Frequency')
        plt.colorbar(im, ax=axes[0])

        # 右侧：信息文本
        axes[1].axis('off')
        info_text = f"JNR: {jnr}\n"
        info_text += f"Types: {' + '.join(jam_names)}\n\n"
        info_text += f"Description:\n{text_desc}\n\n"

        if metadata_dict is not None:
            info_text += "Metadata:\n"
            info_text += f"  Sample Index: {metadata_dict.get('sample_idx', 'N/A')}\n"
            info_text += f"  Position: {metadata_dict.get('pos', 'N/A')}\n"
            info_text += f"  Jam Types: {metadata_dict.get('jam_types', [])}\n\n"
            if metadata_dict.get('jam_params'):
                info_text += "Parameters:\n"
                for key, val in metadata_dict.get('jam_params', {}).items():
                    info_text += f"  {key}: {val}\n"

        axes[1].text(0.05, 0.95, info_text, transform=axes[1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        plt.tight_layout()

        # 保存图片
        save_path = os.path.join(save_dir, f'sample_{idx:03d}_{"_".join(jam_names)}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        if (idx + 1) % 10 == 0:
            print(f"Saved {idx + 1}/{n_samples} samples")

    print(f"\n所有 {n_samples} 个样本已保存到: {save_dir}")

    # 关闭metadata文件句柄
    for meta_group, meta_handle in metadata_list:
        if meta_handle is not None:
            meta_handle.close()

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()