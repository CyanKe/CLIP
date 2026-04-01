"""
测试metadata数据加载完整流程

测试:
1. 单个欺骗干扰 (DFTJ, ISRJ, RGPO, VGPO, SMSPJ, C&IJ, CSJ)
2. 欺骗干扰+AJ组合
3. 文本描述生成
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi.data import CZSLSTFTDatasetWithMetadata
from multi.metadata_template import generate_short_description, generate_abstract_description


# 干扰类型名称映射
JAM_TYPE_NAMES = {
    1: 'DFTJ', 2: 'ISRJ', 3: 'RGPO', 4: 'VGPO',
    5: 'AJ', 6: 'BJ', 7: 'SJ', 8: 'NCJ',
    9: 'NPJ', 10: 'SMSPJ', 11: 'C&IJ', 12: 'NFMJ',
    13: 'NPMJ', 14: 'NAMJ', 15: 'CSJ', 16: 'PJ'
}


def create_test_metadata_deceptive(num_samples=30):
    """
    创建欺骗干扰测试metadata
    包含单个欺骗干扰和欺骗干扰+AJ组合
    """
    dtype = np.dtype([
        ('sample_idx', 'i4'),
        ('jam_types', 'i4', (2,)),
        ('JNR', 'f4'),
        ('pos', 'i4'),
        ('jam_params', [
            ('dftj_k', 'i4'),
            ('isrj_M', 'i4'),
            ('isrj_N', 'i4'),
            ('rgpo_position_rel', 'i4'),  # 0=after, 1=before, 2=overlap
            ('rgpo_final_delay_us', 'f4'),
            ('vgpo_doppler_dir', 'i4'),   # 0=up, 1=down
            ('vgpo_final_fd_kHz', 'f4'),
            ('smspj_M', 'i4'),
            ('cij_a', 'i4'),
            ('cij_b', 'i4'),
            ('cij_is_continuous', 'i4'),
            ('csj_M', 'i4'),
            ('aj_BJ', 'f4'),
        ])
    ])

    metadata = np.zeros((1, num_samples), dtype=dtype)
    idx = 0

    # ========== 单个欺骗干扰 ==========
    # DFTJ (3个样本)
    for k in [4, 6, 8]:
        metadata[0, idx]['sample_idx'] = idx
        metadata[0, idx]['jam_types'][0] = 1
        metadata[0, idx]['jam_types'][1] = 0
        metadata[0, idx]['JNR'] = 15
        metadata[0, idx]['pos'] = 2000 + idx * 200
        metadata[0, idx]['jam_params']['dftj_k'] = k
        idx += 1

    # ISRJ (3个样本)
    for M in [2, 3, 4]:
        metadata[0, idx]['sample_idx'] = idx
        metadata[0, idx]['jam_types'][0] = 2
        metadata[0, idx]['jam_types'][1] = 0
        metadata[0, idx]['JNR'] = 15
        metadata[0, idx]['pos'] = 2000 + idx * 200
        metadata[0, idx]['jam_params']['isrj_M'] = M
        metadata[0, idx]['jam_params']['isrj_N'] = M
        idx += 1

    # RGPO (3个样本: after, before, overlap)
    for pos_rel, delay in [(0, 50), (1, -30), (2, 0)]:
        metadata[0, idx]['sample_idx'] = idx
        metadata[0, idx]['jam_types'][0] = 3
        metadata[0, idx]['jam_types'][1] = 0
        metadata[0, idx]['JNR'] = 15
        metadata[0, idx]['pos'] = 2000 + idx * 200
        metadata[0, idx]['jam_params']['rgpo_position_rel'] = pos_rel
        metadata[0, idx]['jam_params']['rgpo_final_delay_us'] = delay
        idx += 1

    # VGPO (2个样本: up, down)
    for dop_dir, fd in [(0, 100), (1, -80)]:
        metadata[0, idx]['sample_idx'] = idx
        metadata[0, idx]['jam_types'][0] = 4
        metadata[0, idx]['jam_types'][1] = 0
        metadata[0, idx]['JNR'] = 15
        metadata[0, idx]['pos'] = 2000 + idx * 200
        metadata[0, idx]['jam_params']['vgpo_doppler_dir'] = dop_dir
        metadata[0, idx]['jam_params']['vgpo_final_fd_kHz'] = abs(fd)
        idx += 1

    # SMSPJ (3个样本: 不同M值)
    for M in [4, 6, 8]:
        metadata[0, idx]['sample_idx'] = idx
        metadata[0, idx]['jam_types'][0] = 10
        metadata[0, idx]['jam_types'][1] = 0
        metadata[0, idx]['JNR'] = 15
        metadata[0, idx]['pos'] = 2000 + idx * 200
        metadata[0, idx]['jam_params']['smspj_M'] = M
        idx += 1

    # C&IJ (2个样本)
    for a, b in [(2, 3), (4, 4)]:
        metadata[0, idx]['sample_idx'] = idx
        metadata[0, idx]['jam_types'][0] = 11
        metadata[0, idx]['jam_types'][1] = 0
        metadata[0, idx]['JNR'] = 15
        metadata[0, idx]['pos'] = 2000 + idx * 200
        metadata[0, idx]['jam_params']['cij_a'] = a
        metadata[0, idx]['jam_params']['cij_b'] = b
        metadata[0, idx]['jam_params']['cij_is_continuous'] = 1
        idx += 1

    # CSJ (3个样本: 不同梳齿数)
    for M in [3, 5, 8]:
        metadata[0, idx]['sample_idx'] = idx
        metadata[0, idx]['jam_types'][0] = 15
        metadata[0, idx]['jam_types'][1] = 0
        metadata[0, idx]['JNR'] = 15
        metadata[0, idx]['pos'] = 2000 + idx * 200
        metadata[0, idx]['jam_params']['csj_M'] = M
        idx += 1

    # ========== 欺骗干扰+AJ组合 ==========
    # DFTJ+AJ (2个样本)
    for k in [5, 7]:
        metadata[0, idx]['sample_idx'] = idx
        metadata[0, idx]['jam_types'][0] = 1
        metadata[0, idx]['jam_types'][1] = 5
        metadata[0, idx]['JNR'] = 20
        metadata[0, idx]['pos'] = 3000 + idx * 200
        metadata[0, idx]['jam_params']['dftj_k'] = k
        metadata[0, idx]['jam_params']['aj_BJ'] = 25e6
        idx += 1

    # ISRJ+AJ (2个样本)
    for M in [2, 4]:
        metadata[0, idx]['sample_idx'] = idx
        metadata[0, idx]['jam_types'][0] = 2
        metadata[0, idx]['jam_types'][1] = 5
        metadata[0, idx]['JNR'] = 20
        metadata[0, idx]['pos'] = 3000 + idx * 200
        metadata[0, idx]['jam_params']['isrj_M'] = M
        metadata[0, idx]['jam_params']['isrj_N'] = M
        metadata[0, idx]['jam_params']['aj_BJ'] = 30e6
        idx += 1

    # RGPO+AJ
    metadata[0, idx]['sample_idx'] = idx
    metadata[0, idx]['jam_types'][0] = 3
    metadata[0, idx]['jam_types'][1] = 5
    metadata[0, idx]['JNR'] = 20
    metadata[0, idx]['pos'] = 3000 + idx * 200
    metadata[0, idx]['jam_params']['rgpo_position_rel'] = 0
    metadata[0, idx]['jam_params']['aj_BJ'] = 25e6
    idx += 1

    # VGPO+AJ
    metadata[0, idx]['sample_idx'] = idx
    metadata[0, idx]['jam_types'][0] = 4
    metadata[0, idx]['jam_types'][1] = 5
    metadata[0, idx]['JNR'] = 20
    metadata[0, idx]['pos'] = 3000 + idx * 200
    metadata[0, idx]['jam_params']['vgpo_doppler_dir'] = 0
    metadata[0, idx]['jam_params']['aj_BJ'] = 25e6
    idx += 1

    # SMSPJ+AJ
    metadata[0, idx]['sample_idx'] = idx
    metadata[0, idx]['jam_types'][0] = 10
    metadata[0, idx]['jam_types'][1] = 5
    metadata[0, idx]['JNR'] = 20
    metadata[0, idx]['pos'] = 3000 + idx * 200
    metadata[0, idx]['jam_params']['smspj_M'] = 6
    metadata[0, idx]['jam_params']['aj_BJ'] = 25e6
    idx += 1

    # C&IJ+AJ
    metadata[0, idx]['sample_idx'] = idx
    metadata[0, idx]['jam_types'][0] = 11
    metadata[0, idx]['jam_types'][1] = 5
    metadata[0, idx]['JNR'] = 20
    metadata[0, idx]['pos'] = 3000 + idx * 200
    metadata[0, idx]['jam_params']['cij_a'] = 3
    metadata[0, idx]['jam_params']['cij_b'] = 3
    metadata[0, idx]['jam_params']['aj_BJ'] = 25e6
    idx += 1

    # CSJ+AJ
    metadata[0, idx]['sample_idx'] = idx
    metadata[0, idx]['jam_types'][0] = 15
    metadata[0, idx]['jam_types'][1] = 5
    metadata[0, idx]['JNR'] = 20
    metadata[0, idx]['pos'] = 3000 + idx * 200
    metadata[0, idx]['jam_params']['csj_M'] = 5
    metadata[0, idx]['jam_params']['aj_BJ'] = 25e6
    idx += 1

    return metadata[:, :idx]


def parse_metadata_for_description(meta_struct):
    """解析metadata结构为字典格式"""
    jam_types_raw = meta_struct['jam_types']
    jam_types = [int(j) for j in jam_types_raw if j > 0]

    jam_params = {}
    params = meta_struct['jam_params']

    # DFTJ
    if params['dftj_k'] > 0:
        jam_params['dftj_k'] = int(params['dftj_k'])

    # ISRJ
    if params['isrj_M'] > 0:
        jam_params['isrj_M'] = int(params['isrj_M'])
        jam_params['isrj_N'] = int(params['isrj_N'])

    # RGPO
    if 3 in jam_types:
        pos_rel_val = int(params['rgpo_position_rel'])
        jam_params['rgpo_position_relation'] = ['after', 'before', 'overlap'][pos_rel_val]

    # VGPO
    if 4 in jam_types:
        dop_dir_val = int(params['vgpo_doppler_dir'])
        jam_params['vgpo_doppler_direction'] = 'up' if dop_dir_val == 0 else 'down'

    # SMSPJ
    if params['smspj_M'] > 0:
        jam_params['smspj_M'] = int(params['smspj_M'])

    # C&IJ
    if params['cij_a'] > 0:
        jam_params['cij_a'] = int(params['cij_a'])
        jam_params['cij_b'] = int(params['cij_b'])
        jam_params['cij_is_continuous'] = bool(params['cij_is_continuous'])

    # CSJ
    if params['csj_M'] > 0:
        jam_params['csj_M'] = int(params['csj_M'])

    # AJ
    if params['aj_BJ'] > 0:
        jam_params['aj_BJ'] = float(params['aj_BJ'])

    return {
        'sample_idx': int(meta_struct['sample_idx']),
        'jam_types': jam_types,
        'JNR': float(meta_struct['JNR']),
        'pos': int(meta_struct['pos']),
        'jam_params': jam_params
    }


class MockCZSLDataset(torch.utils.data.Dataset):
    """模拟数据集用于测试"""

    def __init__(self, metadata, num_classes=16):
        self.metadata = metadata
        self.num_samples = metadata.shape[1]
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # 模拟STFT数据
        stft = torch.randn(3, 224, 224)

        # 解析metadata
        meta = self.metadata[0, index]
        meta_dict = parse_metadata_for_description(meta)

        # 生成one-hot标签
        label = torch.zeros(self.num_classes)
        for jt in meta_dict['jam_types']:
            if 0 < jt <= self.num_classes:
                label[jt - 1] = 1.0

        # 生成文本描述
        text = generate_short_description(meta_dict, style='type_visual')

        return stft, label, text, meta_dict


def metadata_collate_fn(batch):
    """处理metadata的collate函数"""
    stfts = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    texts = [item[2] for item in batch]
    metas = [item[3] for item in batch]
    return stfts, labels, texts, metas


def test_deceptive_jamming_descriptions():
    """测试欺骗干扰描述生成"""
    print("=" * 70)
    print("欺骗干扰描述测试")
    print("=" * 70)

    # 创建测试metadata
    print("\n创建测试metadata...")
    metadata = create_test_metadata_deceptive()
    num_samples = metadata.shape[1]
    print(f"总样本数: {num_samples}")

    # 创建数据集
    dataset = MockCZSLDataset(metadata)

    # 测试每个样本
    print("\n" + "=" * 70)
    print("单个欺骗干扰描述")
    print("=" * 70)

    idx = 0

    # DFTJ
    print("\n【DFTJ - 密集假目标干扰】")
    for i in range(3):
        stft, label, text, meta = dataset[idx]
        k = meta['jam_params'].get('dftj_k', 0)
        print(f"  k={k}: {text}")
        idx += 1

    # ISRJ
    print("\n【ISRJ - 间歇采样转发干扰】")
    for i in range(3):
        stft, label, text, meta = dataset[idx]
        M = meta['jam_params'].get('isrj_M', 0)
        print(f"  M={M}: {text}")
        idx += 1

    # RGPO
    print("\n【RGPO - 距离拖引干扰】")
    for i in range(3):
        stft, label, text, meta = dataset[idx]
        pos_rel = meta['jam_params'].get('rgpo_position_relation', '')
        print(f"  {pos_rel}: {text}")
        idx += 1

    # VGPO
    print("\n【VGPO - 速度拖引干扰】")
    for i in range(2):
        stft, label, text, meta = dataset[idx]
        dop_dir = meta['jam_params'].get('vgpo_doppler_direction', '')
        print(f"  {dop_dir}: {text}")
        idx += 1

    # SMSPJ
    print("\n【SMSPJ - 弥散谱干扰】")
    for i in range(3):
        stft, label, text, meta = dataset[idx]
        M = meta['jam_params'].get('smspj_M', 0)
        print(f"  M={M}: {text}")
        idx += 1

    # C&IJ
    print("\n【C&IJ - 切片交织干扰】")
    for i in range(2):
        stft, label, text, meta = dataset[idx]
        a = meta['jam_params'].get('cij_a', 0)
        b = meta['jam_params'].get('cij_b', 0)
        print(f"  {a}x{b}: {text}")
        idx += 1

    # CSJ
    print("\n【CSJ - 梳状谱干扰】")
    for i in range(3):
        stft, label, text, meta = dataset[idx]
        M = meta['jam_params'].get('csj_M', 0)
        print(f"  M={M}: {text}")
        idx += 1

    # ========== 组合干扰 ==========
    print("\n" + "=" * 70)
    print("欺骗干扰+AJ组合描述")
    print("=" * 70)

    # DFTJ+AJ
    print("\n【DFTJ+AJ】")
    for i in range(2):
        stft, label, text, meta = dataset[idx]
        k = meta['jam_params'].get('dftj_k', 0)
        print(f"  k={k}: {text}")
        idx += 1

    # ISRJ+AJ
    print("\n【ISRJ+AJ】")
    for i in range(2):
        stft, label, text, meta = dataset[idx]
        M = meta['jam_params'].get('isrj_M', 0)
        print(f"  M={M}: {text}")
        idx += 1

    # RGPO+AJ
    print("\n【RGPO+AJ】")
    stft, label, text, meta = dataset[idx]
    print(f"  {text}")
    idx += 1

    # VGPO+AJ
    print("\n【VGPO+AJ】")
    stft, label, text, meta = dataset[idx]
    print(f"  {text}")
    idx += 1

    # SMSPJ+AJ
    print("\n【SMSPJ+AJ】")
    stft, label, text, meta = dataset[idx]
    print(f"  {text}")
    idx += 1

    # C&IJ+AJ
    print("\n【C&IJ+AJ】")
    stft, label, text, meta = dataset[idx]
    print(f"  {text}")
    idx += 1

    # CSJ+AJ
    print("\n【CSJ+AJ】")
    stft, label, text, meta = dataset[idx]
    print(f"  {text}")
    idx += 1

    # 测试DataLoader
    print("\n" + "=" * 70)
    print("DataLoader批量加载测试")
    print("=" * 70)

    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=metadata_collate_fn)
    batch = next(iter(loader))
    stft_batch, label_batch, text_batch, meta_batch = batch

    print(f"STFT batch shape: {stft_batch.shape}")
    print(f"Label batch shape: {label_batch.shape}")
    print(f"\n随机8个样本的描述:")
    for j in range(min(8, len(text_batch))):
        jam_names = [JAM_TYPE_NAMES[jt] for jt in meta_batch[j]['jam_types']]
        print(f"  [{j}] {'+'.join(jam_names)}: {text_batch[j]}")

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    test_deceptive_jamming_descriptions()