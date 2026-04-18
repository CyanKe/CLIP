"""
精简版数据加载模块 - 用于 CZSL 训练
支持:
- 从 MATLAB .mat 文件加载 STFT 数据 (structured complex64)
- 从 .json 文件加载 metadata
- 复数 STFT → RGB 三通道 (real, imag, mag)
- 全局归一化 (基于预计算的统计量)
- CLIP 标准化
"""
import os
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms

# CLIP 标准化参数
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def collate_fn(batch):
    """
    Collate 函数（模块级别，支持多进程 pickle）

    将 batch 转换为 tensor，并生成文本 tokens

    Args:
        batch: list of (stft_image, label, metadata) 或 (stft_image, time_signal, label, metadata)

    Returns:
        stft_images, time_signals, text_tokens, labels, texts, metadata_list
    """
    from multi.text_templates import generate_text_descriptions
    from multi.metadata_template import generate_short_description
    import clip

    # 检查第一个样本的长度以确定是否包含时域信号
    sample = batch[0]
    has_time_signal = len(sample) == 4  # (stft_image, time_signal, label, metadata)

    if has_time_signal:
        stft_images, time_signals, labels, metadata_list = zip(*batch)
        # 堆叠时域信号
        time_signals = torch.stack(time_signals, dim=0)
    else:
        stft_images, labels, metadata_list = zip(*batch)
        time_signals = None

    # 堆叠 STFT 图像和标签
    stft_images = torch.stack(stft_images, dim=0)
    labels = torch.stack(labels, dim=0)

    # 使用 metadata_template 生成文本描述并 tokenize
    # 使用 'template' 风格：固定的视觉特征描述（与推理一致）
    texts = []
    for meta in metadata_list:
        # 使用 template 风格，与推理时的缓存文本格式一致
        # text = generate_text_descriptions(meta, style='imagenet')
        text = generate_short_description(meta,'meta')
        text = generate_text_descriptions(meta,style = 'meta')
        texts.append(text)

    text_tokens = clip.tokenize(texts, truncate=True)

    return stft_images, time_signals, text_tokens, labels, texts, metadata_list


class STFTDataset(Dataset):
    """
    STFT 数据集 - 用于 CLIP 训练

    数据流:
    1. 从 .mat 读取 structured complex64 STFT 数据
    2. 分解为实部、虚部、幅度三通道
    3. 全局归一化 (使用预计算的统计量)
    4. Resize 到 224x224
    5. CLIP 标准化

    标签来源: metadata.json 中的 jam_types 字段 (数组格式: ["DFTJ", "AJ"])
    """

    def __init__(
        self,
        stft_file: str,
        metadata_file: str,
        stft_var_name: str = 'all_stfts',
        normalization_stats: dict = None,
        image_size: int = 224,
        apply_clip_norm: bool = True,
        class_names: list = None,
    ):
        """
        Args:
            stft_file: STFT 数据 .mat 文件路径
            metadata_file: metadata .json 文件路径 (包含 jam_types 字段)
            stft_var_name: STFT 变量名
            normalization_stats: 归一化统计量 (real_max, mag_max 等)
            image_size: 输出图像尺寸
            apply_clip_norm: 是否应用 CLIP 标准化
            class_names: 类别名称列表 (用于将 jam_types 转换为多热编码)
        """
        super().__init__()

        self.stft_file = stft_file
        self.metadata_file = metadata_file
        self.stft_var_name = stft_var_name

        # 延迟加载 STFT 样本数
        with h5py.File(stft_file, 'r') as f:
            self.num_samples = f[stft_var_name].shape[2]

        self.class_names = class_names or []
        self.num_classes = len(class_names)

        # 加载 metadata
        if metadata_file and os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self._metadata = json.load(f)
            # 从 metadata 构建 label 映射
            self._labels = self._build_labels_from_metadata()
        else:
            raise ValueError(f"Metadata file required: {metadata_file}")

        # 归一化参数
        if normalization_stats is None:
            # 默认值 (应使用预计算的统计量)
            normalization_stats = {
                'real_max': 450.0,
                'imag_max': 450.0,
                'mag_max': 455.0,
            }
        self.norm_scale_real = normalization_stats.get('real_max', 450.0)
        self.norm_scale_imag = normalization_stats.get('imag_max', 450.0)
        self.norm_scale_mag = normalization_stats.get('mag_max', 455.0)

        self.image_size = image_size
        self.apply_clip_norm = apply_clip_norm

        if apply_clip_norm:
            self.clip_norm = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
        else:
            self.clip_norm = None

        # 延迟加载的文件句柄
        self._h5_file = None

    def _build_labels_from_metadata(self) -> np.ndarray:
        """
        从 metadata 的 jam_types 字段构建多热编码标签

        jam_types 格式: ["DFTJ", "AJ"] 或 ["CSJ"]

        Returns:
            labels: np.ndarray, shape (num_samples, num_classes)
        """
        labels = np.zeros((self.num_samples, self.num_classes), dtype=np.float32)

        # 构建类别名称到索引的映射
        name_to_idx = {name: i for i, name in enumerate(self.class_names)}

        for i, meta in enumerate(self._metadata):
            if i >= self.num_samples:
                break

            jam_types = meta.get('jam_types', [])

            # 解析 jam_types (数组格式)
            if isinstance(jam_types, list):
                types_list = jam_types
            elif isinstance(jam_types, str):
                # 兼容字符串格式
                types_list = [jam_types] if jam_types and jam_types != 'None' else []
            else:
                types_list = []

            # 转换为多热编码
            for jam_type in types_list:
                jam_type = jam_type.strip() if isinstance(jam_type, str) else str(jam_type)
                if jam_type in name_to_idx:
                    labels[i, name_to_idx[jam_type]] = 1.0

        return labels

    def _lazy_load(self):
        """延迟加载 h5 文件 (每个 worker 独立)"""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.stft_file, 'r')
        return self._h5_file

    def __len__(self):
        return self.num_samples

    def _get_metadata(self, index: int) -> dict:
        """获取样本的 metadata，并标准化格式"""
        if self._metadata is None:
            return None
        if isinstance(self._metadata, list) and index < len(self._metadata):
            meta = self._metadata[index].copy()
            # 将 jam_types 从整数转换为列表 (兼容旧格式)
            if 'jam_types' in meta and isinstance(meta['jam_types'], int):
                meta['jam_types'] = [meta['jam_types']]
            return meta
        return None

    def __getitem__(self, index: int):
        h5_file = self._lazy_load()

        # 1. 读取 STFT 数据 (structured complex64)
        raw_stft = h5_file[self.stft_var_name][:, :, index]

        # 2. 转换为复数
        stft_complex = raw_stft['real'] + 1j * raw_stft['imag']

        # 3. 提取三通道：实部、虚部、幅度
        stft_real = np.real(stft_complex).T  # 转置到 (224, 225)
        stft_imag = np.imag(stft_complex).T
        stft_mag = np.abs(stft_complex).T

        # 4. 归一化到 [0, 1] (使用全局统计量)
        stft_real = np.clip(stft_real, -self.norm_scale_real, self.norm_scale_real) / self.norm_scale_real
        stft_imag = np.clip(stft_imag, -self.norm_scale_imag, self.norm_scale_imag) / self.norm_scale_imag
        stft_mag = np.clip(stft_mag, 0, self.norm_scale_mag) / self.norm_scale_mag

        # 5. 构建三通道张量
        stft_tensor = torch.from_numpy(np.stack([stft_real, stft_imag, stft_mag], axis=0)).float()
        stft_tensor = torch.from_numpy(np.stack([stft_mag, stft_mag, stft_mag], axis=0)).float()

        # 6. Resize 到目标尺寸 (如果还不是 224x224)
        if stft_tensor.shape[-2:] != (self.image_size, self.image_size):
            stft_tensor = torch.nn.functional.interpolate(
                stft_tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # 7. CLIP 标准化
        if self.clip_norm is not None:
            stft_tensor = self.clip_norm(stft_tensor)

        # 8. 从预构建的标签数组获取标签
        label_tensor = torch.from_numpy(self._labels[index])

        # 9. 获取 metadata (用于生成文本描述)
        metadata = self._get_metadata(index)

        return stft_tensor, label_tensor, metadata

def create_czsl_dataloaders(
    config: dict,
    normalization_stats: dict = None,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple:
    """
    创建 CZSL 数据加载器 (train/val/test)

    Args:
        config: 配置字典
        normalization_stats: 归一化统计量
        batch_size: 批次大小
        num_workers: 数据加载 worker 数
        pin_memory: 是否 pin memory

    Returns:
        (train_loader, val_loader, test_loader, num_classes)
    """

    data_config = config.get('data', {})
    base_path = data_config.get('base_path')
    jnr_start = data_config.get('jnr_start', 10)
    jnr_end = data_config.get('jnr_end', 10)
    jnr_step = data_config.get('jnr_step', 1)

    # 时域数据配置
    use_time_domain = config.get('use_time_domain', False)
    time_seq_len = data_config.get('time_seq_len', 2048)
    time_var_name = data_config.get('time_var_name', 'raw_time')

    # JNR 级别
    jnr_levels = list(range(jnr_start, jnr_end + 1, jnr_step))

    # 类别名称
    class_names = [cls['name'] for cls in config.get('jamming_classes', [])]

    # 加载所有 JNR 级别的数据
    def load_split(split_name: str):
        """加载单个 split 的数据集"""
        datasets = []

        for jnr in jnr_levels:
            jnr_folder = f"JNR_{'+' if jnr >= 0 else ''}{jnr}"
            data_folder = os.path.join(base_path, jnr_folder)

            stft_file = os.path.join(data_folder, f'{split_name}_echo_stfts.mat')
            metadata_file = os.path.join(data_folder, f'{split_name}_echo_metadata.json')

            # 时域数据文件路径 (假设命名规则)
            time_file = os.path.join(data_folder, f'{split_name}_echo_times.mat')

            if not os.path.exists(stft_file):
                print(f"Warning: STFT data not found for {jnr_folder}/{split_name}, skipping...")
                continue

            if not os.path.exists(metadata_file):
                print(f"Warning: Metadata not found for {jnr_folder}/{split_name}, skipping...")
                continue

            # 创建 STFT 数据集
            stft_dataset = STFTDataset(
                stft_file=stft_file,
                metadata_file=metadata_file,
                normalization_stats=normalization_stats,
                class_names=class_names,
            )
            datasets.append(stft_dataset)
            print(f"Loaded {split_name} data from {jnr_folder}: {len(stft_dataset)} samples")

        if not datasets:
            raise ValueError(f"No data found for {split_name} split!")

        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    # 创建数据集
    train_dataset = load_split('train')
    val_dataset = load_split('val')
    test_dataset = load_split('test')

    # 使用模块级别的 collate_fn
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader, len(class_names)


if __name__ == "__main__":
    # 测试数据加载
    import yaml

    with open('multi/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 加载统计量
    stats_file = 'multi/normalization_stats.json'
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            normalization_stats = json.load(f)
    else:
        normalization_stats = None

    train_loader, val_loader, test_loader, num_classes = create_czsl_dataloaders(
        config=config,
        normalization_stats=normalization_stats,
        batch_size=4,
        num_workers=0,
    )

    print(f"\nNumber of classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # 测试一个 batch
    images, text_tokens, labels, texts, metadata_list = next(iter(train_loader))
    print(f"\nBatch shape: images={images.shape}, text_tokens={text_tokens.shape}")
    print(f"Sample text: {texts[0]}")
