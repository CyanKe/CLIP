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
    # 使用 'visual_param' 风格：视觉特征 + 参数强度
    texts = []
    for meta in metadata_list:
        text = generate_text_descriptions(meta, style='meta')
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
    """

    def __init__(
        self,
        stft_file: str,
        label_file: str,
        metadata_file: str = None,
        stft_var_name: str = 'all_stfts',
        label_var_name: str = 'all_label',
        metadata_var_name: str = 'all_metadata',
        normalization_stats: dict = None,
        image_size: int = 224,
        apply_clip_norm: bool = True,
        class_names: list = None,
    ):
        """
        Args:
            stft_file: STFT 数据 .mat 文件路径
            label_file: 标签 .mat 文件路径
            metadata_file: metadata .json 文件路径 (可选)
            stft_var_name: STFT 变量名
            label_var_name: 标签变量名
            metadata_var_name: metadata 变量名
            normalization_stats: 归一化统计量 (real_max, mag_max 等)
            image_size: 输出图像尺寸
            apply_clip_norm: 是否应用 CLIP 标准化
            class_names: 类别名称列表
        """
        super().__init__()

        self.stft_file = stft_file
        self.label_file = label_file
        self.metadata_file = metadata_file
        self.stft_var_name = stft_var_name
        self.label_var_name = label_var_name
        self.metadata_var_name = metadata_var_name

        # 延迟加载元数据
        with h5py.File(stft_file, 'r') as f:
            self.num_samples = f[stft_var_name].shape[2]
        with h5py.File(label_file, 'r') as f:
            self.num_classes = f[label_var_name].shape[0]

        self.class_names = class_names or []

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
        self._h5_files = None
        self._metadata = None

    def _lazy_load(self):
        """延迟加载 h5 文件 (每个 worker 独立)"""
        if self._h5_files is None:
            self._h5_files = {
                'stft': h5py.File(self.stft_file, 'r'),
                'label': h5py.File(self.label_file, 'r'),
            }
            # 加载 metadata (如果存在)
            if self.metadata_file and os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self._metadata = json.load(f)
        return self._h5_files

    def __len__(self):
        return self.num_samples

    def _get_metadata(self, index: int) -> dict:
        """获取样本的 metadata，并标准化格式"""
        if self._metadata is None:
            return None
        if isinstance(self._metadata, list) and index < len(self._metadata):
            meta = self._metadata[index].copy()  # 复制一份，避免修改原始数据
            # 将 jam_types 从整数转换为列表
            if 'jam_types' in meta and isinstance(meta['jam_types'], int):
                meta['jam_types'] = [meta['jam_types']]
            return meta
        return None

    def __getitem__(self, index: int):
        h5_files = self._lazy_load()

        # 1. 读取 STFT 数据 (structured complex64)
        raw_stft = h5_files['stft'][self.stft_var_name][:, :, index]

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

        # 8. 读取标签
        label = h5_files['label'][self.label_var_name][:, index]
        label_tensor = torch.from_numpy(label).float()

        # 9. 获取 metadata (用于生成文本描述)
        metadata = self._get_metadata(index)

        return stft_tensor, label_tensor, metadata


class TimeDomainDataset(Dataset):
    """
    时域信号数据集 - 用于处理 complex single 格式的 HDF5 时域数据

    数据流:
    1. 从 .mat 读取 complex single 时域数据
    2. 提取幅度 (magnitude)
    3. 归一化并 pad/truncate 到固定长度
    4. 返回 1D 张量
    """

    def __init__(
        self,
        time_file: str,
        time_var_name: str = 'raw_time',
        seq_len: int = 2048,
        normalization_stats: dict = None,
    ):
        """
        Args:
            time_file: 时域数据 .mat 文件路径
            time_var_name: 时域数据变量名
            seq_len: 固定序列长度
            normalization_stats: 归一化统计量 (min, max)
        """
        super().__init__()

        self.time_file = time_file
        self.time_var_name = time_var_name
        self.seq_len = seq_len

        # 延迟加载元数据
        with h5py.File(time_file, 'r') as f:
            # 用户的数据是二维数组 (样本数, 8000)
            data_shape = f[time_var_name].shape
            if len(data_shape) == 2:
                self.num_samples = data_shape[0]  # 第一维是样本数
            else:
                raise ValueError(f"Expected 2D array (samples, seq_len), got shape {data_shape}")

        # 归一化参数
        if normalization_stats is None:
            # 默认值 (应使用预计算的统计量)
            normalization_stats = {
                'time_min': -1.0,
                'time_max': 1.0,
            }
        self.norm_min = normalization_stats.get('time_min', -1.0)
        self.norm_max = normalization_stats.get('time_max', 1.0)

        # 延迟加载的文件句柄
        self._h5_file = None

    def _lazy_load(self):
        """延迟加载 h5 文件 (每个 worker 独立)"""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.time_file, 'r')
        return self._h5_file

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index: int):
        h5_file = self._lazy_load()

        # 1. 读取时域数据
        # 用户提供的是二维数组 (样本数, 8000)
        # 所以我们需要读取第 index 行
        raw_time = h5_file[self.time_var_name][index, :]  # Shape: (8000,)

        # 2. 处理数据类型 - 检查是否是结构化数组
        if raw_time.dtype.names is not None:
            # 结构化数组，包含 real 和 imag 字段
            # 提取实部和虚部
            real_part = raw_time['real'] if 'real' in raw_time.dtype.names else raw_time['f0']
            imag_part = raw_time['imag'] if 'imag' in raw_time.dtype.names else raw_time['f1']
            # 组合为复数并提取幅度
            time_data = np.sqrt(real_part**2 + imag_part**2)
        elif np.iscomplexobj(raw_time):
            # 直接的复数数组
            time_data = np.abs(raw_time)
        else:
            # 实数数组
            time_data = raw_time

        # 3. 归一化到 [0, 1] 或 [-1, 1]
        time_data = np.clip(time_data, self.norm_min, self.norm_max)
        range_span = self.norm_max - self.norm_min
        if range_span > 0:
            time_data = (time_data - self.norm_min) / range_span

        # 4. Pad or Truncate 到固定长度
        if len(time_data) > self.seq_len:
            time_data = time_data[:self.seq_len]
        elif len(time_data) < self.seq_len:
            pad_len = self.seq_len - len(time_data)
            time_data = np.pad(time_data, (0, pad_len), mode='constant')

        # 5. 转换为张量
        time_tensor = torch.from_numpy(time_data).float()

        return time_tensor


class CombinedRadarDataset(Dataset):
    """
    组合雷达数据集 - 同时加载 STFT 和时域信号
    """

    def __init__(
        self,
        stft_dataset: STFTDataset,
        time_dataset: TimeDomainDataset,
    ):
        """
        Args:
            stft_dataset: STFT 数据集实例
            time_dataset: 时域数据集实例
        """
        super().__init__()
        self.stft_dataset = stft_dataset
        self.time_dataset = time_dataset

        # 验证两个数据集的样本数是否一致
        if len(stft_dataset) != len(time_dataset):
            raise ValueError(
                f"STFT dataset has {len(stft_dataset)} samples, "
                f"but time domain dataset has {len(time_dataset)} samples. "
                "They must have the same number of samples."
            )

    def __len__(self):
        return len(self.stft_dataset)

    def __getitem__(self, index: int):
        # 从两个数据集获取对应样本
        stft_tensor, label, metadata = self.stft_dataset[index]
        time_tensor = self.time_dataset[index]

        return stft_tensor, time_tensor, label, metadata


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
            label_file = os.path.join(data_folder, f'{split_name}_echo_label.mat')
            metadata_file = os.path.join(data_folder, f'{split_name}_echo_metadata.json')

            # 时域数据文件路径 (假设命名规则)
            time_file = os.path.join(data_folder, f'{split_name}_echo_time.mat')

            if not (os.path.exists(stft_file) and os.path.exists(label_file)):
                print(f"Warning: Data not found for {jnr_folder}/{split_name}, skipping...")
                continue

            # Metadata 文件可选
            if not os.path.exists(metadata_file):
                metadata_file = None

            # 创建 STFT 数据集
            stft_dataset = STFTDataset(
                stft_file=stft_file,
                label_file=label_file,
                metadata_file=metadata_file,
                normalization_stats=normalization_stats,
                class_names=class_names,
            )

            # 如果启用时域数据，创建时域数据集并组合
            if use_time_domain:
                if os.path.exists(time_file):
                    time_dataset = TimeDomainDataset(
                        time_file=time_file,
                        time_var_name=time_var_name,
                        seq_len=time_seq_len,
                        normalization_stats=normalization_stats,
                    )
                    # 组合两个数据集
                    combined_dataset = CombinedRadarDataset(stft_dataset, time_dataset)
                    datasets.append(combined_dataset)
                    print(f"Loaded {split_name} data (STFT + Time) from {jnr_folder}: {len(combined_dataset)} samples")
                else:
                    print(f"Warning: Time domain data not found for {jnr_folder}/{split_name}, using STFT only...")
                    datasets.append(stft_dataset)
            else:
                datasets.append(stft_dataset)
                print(f"Loaded {split_name} data (STFT only) from {jnr_folder}: {len(stft_dataset)} samples")

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

    with open('multi/config.yaml', 'r') as f:
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
