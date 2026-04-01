"""
测试metadata数据加载 - 模拟MATLAB .mat文件读取

使用numpy structured array模拟MATLAB struct格式
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import tempfile
import os

# 导入metadata模板
from multi.metadata_template import (
    generate_abstract_description,
    JAM_TYPE_NAMES,
    JNR_LEVELS,
    FALSE_TARGET_LEVELS,
    ISRJ_LEVELS,
    get_level_desc
)


def create_mock_metadata_mat(num_samples=100, save_path=None):
    """
    创建模拟的metadata .mat文件格式

    MATLAB struct在Python中的对应格式:
    - 使用numpy structured array
    - 每个字段是一个数组
    """
    # 定义结构化数组的数据类型
    # MATLAB的struct会转换为此格式
    dt = np.dtype([
        ('sample_idx', 'i4'),           # int32
        ('jam_types', 'i4', (2,)),      # 最大2个干扰类型
        ('JNR', 'f4'),                  # float32
        ('pos', 'i4'),                  # int32
        ('dftj_k', 'i4'),               # DFTJ假目标数量
        ('isrj_M', 'i4'),               # ISRJ转发次数
        ('isrj_N', 'i4'),               # ISRJ采样次数
        ('jam_BJ', 'f4'),               # 干扰带宽 Hz
    ])

    # 创建结构化数组
    metadata = np.zeros(num_samples, dtype=dt)

    # 填充数据
    for i in range(num_samples):
        metadata[i]['sample_idx'] = i

        # 随机选择干扰类型
        if i < 30:  # DFTJ单干扰
            metadata[i]['jam_types'][0] = 1
            metadata[i]['jam_types'][1] = 0
            metadata[i]['dftj_k'] = np.random.randint(4, 9)
        elif i < 60:  # ISRJ单干扰
            metadata[i]['jam_types'][0] = 2
            metadata[i]['jam_types'][1] = 0
            metadata[i]['isrj_M'] = np.random.randint(2, 5)
            metadata[i]['isrj_N'] = np.random.randint(2, 5)
        else:  # DFTJ+AJ组合
            metadata[i]['jam_types'][0] = 1
            metadata[i]['jam_types'][1] = 5
            metadata[i]['dftj_k'] = np.random.randint(4, 9)
            metadata[i]['jam_BJ'] = (18.5 + 5 * np.random.random()) * 1e6

        metadata[i]['JNR'] = np.random.choice([0, 10, 15, 20, 30])
        metadata[i]['pos'] = 1000 + np.random.randint(0, 6000)

    if save_path:
        np.save(save_path, metadata)
        print(f"Mock metadata saved to {save_path}")

    return metadata


def parse_metadata_to_dict(metadata_row):
    """
    将numpy结构化数组的一行转换为Python字典格式

    Args:
        metadata_row: numpy structured array的一行

    Returns:
        dict: Python字典格式的metadata
    """
    # 提取有效的干扰类型
    jam_types_raw = metadata_row['jam_types']
    jam_types = [int(j) for j in jam_types_raw if j > 0]

    # 构建jam_params字典
    jam_params = {}
    for jam_type in jam_types:
        name = JAM_TYPE_NAMES.get(jam_type, f'type{jam_type}').lower()
        if jam_type == 1:  # DFTJ
            jam_params[name] = {'k': int(metadata_row['dftj_k'])}
        elif jam_type == 2:  # ISRJ
            jam_params[name] = {
                'M': int(metadata_row['isrj_M']),
                'N': int(metadata_row['isrj_N'])
            }
        elif jam_type in [5, 6]:  # AJ/BJ
            jam_params[name] = {'BJ': float(metadata_row['jam_BJ'])}

    return {
        'sample_idx': int(metadata_row['sample_idx']),
        'jam_types': jam_types,
        'JNR': float(metadata_row['JNR']),
        'pos': int(metadata_row['pos']),
        'jam_params': jam_params,
    }


class MetadataSTFTDataset(Dataset):
    """
    支持metadata的STFT数据集

    返回: (stft_tensor, label_tensor, text_description, metadata_dict)
    """

    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        metadata_array: np.ndarray,
        num_classes: int = 16,
        image_size: int = 224,
        use_abstract_description: bool = True
    ):
        """
        初始化

        Args:
            metadata_array: numpy结构化数组格式的metadata
            num_classes: 类别数
            image_size: 输出图像尺寸
            use_abstract_description: 是否使用抽象描述
        """
        self.metadata = metadata_array
        self.num_samples = len(metadata_array)
        self.num_classes = num_classes
        self.image_size = image_size
        self.use_abstract_description = use_abstract_description

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # 获取metadata
        meta = parse_metadata_to_dict(self.metadata[index])

        # 生成one-hot标签
        label_tensor = torch.zeros(self.num_classes)
        for jam_type in meta['jam_types']:
            if jam_type > 0 and jam_type <= self.num_classes:
                label_tensor[jam_type - 1] = 1.0

        # 生成文本描述
        if self.use_abstract_description:
            text_desc = generate_abstract_description(meta)
        else:
            # 使用类级别描述
            jam_names = [JAM_TYPE_NAMES[j] for j in meta['jam_types']]
            text_desc = f"a radar signal with {', '.join(jam_names)}"

        # 模拟STFT数据（实际使用时从h5文件读取）
        stft_tensor = torch.randn(3, self.image_size, self.image_size)

        return stft_tensor, label_tensor, text_desc, meta


def test_metadata_loading():
    """测试metadata加载流程"""
    print("=" * 60)
    print("测试 metadata 数据加载")
    print("=" * 60)

    # 1. 创建模拟metadata
    print("\n1. 创建模拟metadata (100样本)...")
    mock_metadata = create_mock_metadata_mat(100)

    print(f"   Metadata shape: {mock_metadata.shape}")
    print(f"   Metadata dtype: {mock_metadata.dtype}")

    # 2. 测试单样本解析
    print("\n2. 解析第一个样本...")
    meta_dict = parse_metadata_to_dict(mock_metadata[0])
    print(f"   原始数据: {mock_metadata[0]}")
    print(f"   解析后: {meta_dict}")

    # 3. 测试抽象描述生成
    print("\n3. 生成抽象描述...")
    desc = generate_abstract_description(meta_dict)
    print(f"   描述: {desc}")

    # 4. 创建数据集并测试
    print("\n4. 创建MetadataSTFTDataset...")
    dataset = MetadataSTFTDataset(mock_metadata, num_classes=16)
    print(f"   数据集大小: {len(dataset)}")

    # 5. 测试__getitem__
    print("\n5. 测试数据集访问...")
    for i in [0, 50, 99]:
        stft, label, text, meta = dataset[i]
        jam_names = [JAM_TYPE_NAMES[j] for j in meta['jam_types']]
        print(f"\n   样本 {i}:")
        print(f"     干扰类型: {jam_names}")
        print(f"     JNR: {meta['JNR']} dB")
        print(f"     位置: {meta['pos']} 采样点")
        print(f"     文本描述: {text}")

    # 6. 测试批量加载
    print("\n6. 测试DataLoader批量加载...")
    from torch.utils.data import DataLoader

    # 自定义collate函数处理metadata字典
    def metadata_collate_fn(batch):
        """
        处理包含metadata字典的batch
        """
        stfts = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        texts = [item[2] for item in batch]  # 文本保持为list
        metas = [item[3] for item in batch]  # metadata保持为list
        return stfts, labels, texts, metas

    loader = DataLoader(dataset, batch_size=8, shuffle=True,
                        collate_fn=metadata_collate_fn)
    batch = next(iter(loader))
    stft_batch, label_batch, text_batch, meta_batch = batch

    print(f"   STFT batch shape: {stft_batch.shape}")
    print(f"   Label batch shape: {label_batch.shape}")
    print(f"   Text descriptions (前3个):")
    for j in range(3):
        print(f"     [{j}]: {text_batch[j]}")
        print(f"     metadata: {meta_batch[j]['jam_params']}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

    return mock_metadata


if __name__ == "__main__":
    test_metadata_loading()