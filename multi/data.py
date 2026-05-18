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


def tokenize_texts(
    texts: list,
    model_type: str = "clip",
    processor=None,
    max_length: int = None
) -> torch.Tensor:
    """
    统一的文本分词函数，支持 CLIP 和 SigLIP

    Args:
        texts: 文本列表
        model_type: 模型类型 ("clip" 或 "siglip-xxx")
        processor: SigLIP processor（SigLIP 模型需要）
        max_length: 最大序列长度（可选）

    Returns:
        tokens: token 张量 [batch_size, seq_len]
    """
    if model_type.startswith("siglip"):
        # SigLIP tokenization
        from multi.siglip_loader import get_siglip_text_length

        if max_length is None:
            max_length = get_siglip_text_length(model_type)

        encoded = processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        return encoded["input_ids"]
    else:
        # CLIP tokenization
        import clip
        return clip.tokenize(texts, truncate=True)


def collate_fn(batch):
    """
    Collate 函数（模块级别，支持多进程 pickle）

    将 batch 转换为 tensor，并生成文本 tokens

    Args:
        batch: list of (stft_image, label, metadata) 或 (stft_image, label, metadata, features)

    Returns:
        stft_images, time_signals, text_tokens, labels, texts, metadata_list [, features_batched]
    """
    from multi.text_templates import generate_text_descriptions
    import clip

    # 检测样本格式
    sample = batch[0]
    has_features = isinstance(sample[-1], dict) and len(sample) == 4

    if has_features:
        stft_images, labels, metadata_list, features_list = zip(*batch)
        # 拼接各域特征
        features_batched = {}
        for domain in features_list[0].keys():
            features_batched[domain] = torch.stack([f[domain] for f in features_list], dim=0)
    else:
        stft_images, labels, metadata_list = zip(*batch)
        features_batched = None

    # 堆叠 STFT 图像和标签
    stft_images = torch.stack(stft_images, dim=0)
    labels = torch.stack(labels, dim=0)

    # 使用 metadata_template 生成文本描述并 tokenize
    texts = []
    for meta in metadata_list:
        text = generate_text_descriptions(meta, style='class_only')
        texts.append(text)

    text_tokens = clip.tokenize(texts, truncate=True)

    return stft_images, None, text_tokens, labels, texts, metadata_list, features_batched


def _collate_fn(batch, model_type="clip", processor=None):
    """Collate 函数，支持 CLIP 和 SigLIP tokenization（模块级别以便 pickle）"""
    from multi.text_templates import generate_text_descriptions

    # 检测样本格式
    sample = batch[0]
    has_features = isinstance(sample[-1], dict) and len(sample) == 4

    if has_features:
        stft_images, labels, metadata_list, features_list = zip(*batch)
        features_batched = {}
        for domain in features_list[0].keys():
            features_batched[domain] = torch.stack([f[domain] for f in features_list], dim=0)
    else:
        stft_images, labels, metadata_list = zip(*batch)
        features_batched = None

    # 堆叠 STFT 图像和标签
    stft_images = torch.stack(stft_images, dim=0)
    labels = torch.stack(labels, dim=0)

    # 生成文本描述
    texts = []
    for meta in metadata_list:
        text = generate_text_descriptions(meta, style='class_only')
        texts.append(text)

    # 使用统一的 tokenize 函数
    text_tokens = tokenize_texts(texts, model_type, processor)

    return stft_images, None, text_tokens, labels, texts, metadata_list, features_batched


def create_collate_fn(model_type: str = "clip", processor=None):
    """
    创建支持 CLIP 或 SigLIP 的 collate 函数

    Args:
        model_type: 模型类型 ("clip" 或 "siglip-xxx")
        processor: SigLIP processor（SigLIP 模型需要）

    Returns:
        collate 函数
    """
    from functools import partial
    return partial(_collate_fn, model_type=model_type, processor=processor)


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
        normalize_mode: str = 'per_sample', # 'global'  'per_sample'
        normalize_method: str = 'p99', #'p99' 'p95' 或 'max'
        features_file: str = None,
        feature_norm_stats: dict = None,
        use_feature_context: bool = False,
        augmentation: object = None,
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
            features_file: 多域特征 .json 文件路径 (可选)
            feature_norm_stats: 特征归一化统计量 (可选)
            use_feature_context: 是否启用特征条件上下文
        """
        super().__init__()

        self.stft_file = stft_file
        self.metadata_file = metadata_file
        self.stft_var_name = stft_var_name
        self.use_feature_context = use_feature_context
        self.augmentation = augmentation

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
        self.normalize_mode = normalize_mode
        self.normalize_method = normalize_method

        if apply_clip_norm:
            self.clip_norm = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
        else:
            self.clip_norm = None

        # 延迟加载的文件句柄
        self._h5_file = None

        # 特征条件上下文
        self._raw_features = None
        self._feature_norm_stats = feature_norm_stats or {}
        if self.use_feature_context and features_file and os.path.exists(features_file):
            with open(features_file, 'r', encoding='utf-8') as f:
                self._raw_features = json.load(f)
            print(f"  Loaded features from {features_file}: {len(self._raw_features)} samples")
        elif self.use_feature_context:
            print(f"  Warning: use_feature_context=True but features file not found: {features_file}")

    # 特征域定义
    FEATURE_DOMAINS = {
        'time': {
            'prefix': 'time_domain',
            'keys': ['skewness', 'kurtosis', 'envelope_variation', 'modulation_bandwidth', 'modulation_rate'],
        },
        'freq': {
            'prefix': 'freq_domain',
            'keys': ['spectral_skewness', 'spectral_kurtosis', 'carrier_factor', 'awgn_factor'],
        },
        'bispectrum': {
            'prefix': 'bispectrum',
            'keys': ['bispectrum_variance', 'bispectrum_mean'],
        },
        'wavelet': {
            'prefix': 'wavelet',
            'keys': ['variance', 'mean', 'max', 'scale_centroid', 'max_singular_value',
                     'central_moment_2', 'central_moment_3', 'central_moment_4'],
        },
        'statistical': {
            'prefix': 'statistical',
            'keys': ['shannon_entropy', 'exponential_entropy', 'norm_entropy'],
        },
    }

    def _get_features(self, index: int) -> dict:
        """获取样本的多域特征（已标准化）

        Returns:
            features_dict: {'time': tensor([5]), 'freq': tensor([4]), ...}
        """
        if self._raw_features is None or index >= len(self._raw_features):
            # 回退: 返回零特征
            return {d: torch.zeros(len(info['keys'])) for d, info in self.FEATURE_DOMAINS.items()}

        raw = self._raw_features[index]
        features_dict = {}

        for domain, info in self.FEATURE_DOMAINS.items():
            prefix = info['prefix']
            values = []
            for key in info['keys']:
                flat_key = f"{prefix}.{key}"
                # 嵌套取值
                val = raw
                for part in [prefix, key]:
                    val = val.get(part, 0.0) if isinstance(val, dict) else 0.0
                # 处理 NaN/Inf
                if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                    val = 0.0
                # 标准化
                if flat_key in self._feature_norm_stats:
                    stats = self._feature_norm_stats[flat_key]
                    val = (val - stats.get('mean', 0.0)) / (stats.get('std', 1.0) + 1e-8)
                values.append(val)
            features_dict[domain] = torch.tensor(values, dtype=torch.float32)

        return features_dict

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
        # 4. 归一化
        if self.normalize_mode == 'per_sample':
            # 样本级归一化
            mag = np.abs(stft_complex)
            if self.normalize_method == 'max':
                ref = np.max(mag)
            elif self.normalize_method == 'p95':
                ref = np.percentile(mag, 95)
            else:  # p99
                ref = np.percentile(mag, 99)

            if ref > 0:
                stft_complex = stft_complex / ref

            stft_real = np.real(stft_complex).T
            stft_imag = np.imag(stft_complex).T
            stft_mag = np.abs(stft_complex).T
        else:
            # 全局归一化
            stft_real = np.real(stft_complex).T
            stft_imag = np.imag(stft_complex).T
            stft_mag = np.abs(stft_complex).T

            stft_real = np.clip(stft_real, -self.norm_scale_real, self.norm_scale_real) / self.norm_scale_real
            stft_imag = np.clip(stft_imag, -self.norm_scale_imag, self.norm_scale_imag) / self.norm_scale_imag
            stft_mag = np.clip(stft_mag, 0, self.norm_scale_mag) / self.norm_scale_mag

        # 5. 构建三通道张量
        # stft_tensor = torch.from_numpy(np.stack([stft_real, stft_imag, stft_mag], axis=0)).float()
        stft_tensor = torch.from_numpy(np.stack([stft_mag, stft_mag, stft_mag], axis=0)).float()

        # 6. Resize 到目标尺寸 (如果还不是 224x224)
        if stft_tensor.shape[-2:] != (self.image_size, self.image_size):
            stft_tensor = torch.nn.functional.interpolate(
                stft_tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # 6b. 保存 CLIP 标准化前的能量图 (用于 Energy-Aware Masking)
        energy_map = None
        if self.augmentation is not None:
            energy_map = stft_tensor[0].clone()  # [H, W], 值域约 [0, 1]

        # 7. CLIP 标准化
        if self.clip_norm is not None:
            stft_tensor = self.clip_norm(stft_tensor)

        # 8. 获取 metadata (用于文本描述和数据增强)
        metadata = self._get_metadata(index)

        # 9. 数据增强 (在 CLIP 标准化之后, mean≈0 所以 mask=0 表示无信息)
        if self.augmentation is not None:
            stft_tensor, _ = self.augmentation(stft_tensor, metadata=metadata, energy_map=energy_map)

        # 10. 从预构建的标签数组获取标签
        label_tensor = torch.from_numpy(self._labels[index])

        # 11. 获取多域特征 (如果启用)
        if self.use_feature_context:
            features = self._get_features(index)
            return stft_tensor, label_tensor, metadata, features
        else:
            return stft_tensor, label_tensor, metadata


# ============================================================================
# 双分支数据加载 - 用于欺骗/压制干扰分类
# ============================================================================

class DualBranchSTFTDataset(Dataset):
    """
    双分支 STFT 数据集 - 用于欺骗/压制干扰分类

    标签生成策略:
    - 欺骗干扰样本: labels_deception = [1,0,0,...], labels_suppression = [0,0,...,1] (无压制)
    - 压制干扰样本: labels_deception = [0,0,...,1] (无欺骗), labels_suppression = [1,0,...]
    """

    def __init__(
        self,
        stft_file: str,
        metadata_file: str,
        stft_var_name: str = 'all_stfts',
        normalization_stats: dict = None,
        image_size: int = 224,
        apply_clip_norm: bool = True,
        deception_classes: list = None,
        suppression_classes: list = None,
        normalize_mode: str = 'per_sample',
        normalize_method: str = 'p99',
        augmentation: object = None,
    ):
        """
        Args:
            stft_file: STFT 数据文件路径
            metadata_file: metadata 文件路径
            deception_classes: 欺骗干扰类型列表
            suppression_classes: 压制干扰类型列表
        """
        super().__init__()

        self.stft_file = stft_file
        self.metadata_file = metadata_file
        self.stft_var_name = stft_var_name

        # 延迟加载 STFT 样本数
        with h5py.File(stft_file, 'r') as f:
            self.num_samples = f[stft_var_name].shape[2]

        # 干扰类型分组
        self.deception_classes = deception_classes or ["DFTJ", "ISRJ", "SMSPJ", "C&IJ", "CSJ"]
        self.suppression_classes = suppression_classes or ["AJ", "BJ", "SJ", "NCJ", "NPJ", "NFMJ", "NPMJ", "NAMJ", "PJ"]

        # 类别数 (含 "无XX干扰")
        self.num_deception_classes = len(self.deception_classes) + 1  # +1 for "无欺骗干扰"
        self.num_suppression_classes = len(self.suppression_classes) + 1  # +1 for "无压制干扰"

        # 加载 metadata
        if metadata_file and os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self._metadata = json.load(f)
            # 构建双分支标签
            self._labels_deception, self._labels_suppression = self._build_dual_branch_labels()
        else:
            raise ValueError(f"Metadata file required: {metadata_file}")

        # 归一化参数
        if normalization_stats is None:
            normalization_stats = {
                'real_max': 450.0,
                'imag_max': 450.0,
                'mag_max': 455.0,
            }
        self.norm_scale_mag = normalization_stats.get('mag_max', 455.0)

        self.image_size = image_size
        self.apply_clip_norm = apply_clip_norm
        self.normalize_mode = normalize_mode
        self.normalize_method = normalize_method
        self.augmentation = augmentation

        if apply_clip_norm:
            self.clip_norm = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
        else:
            self.clip_norm = None

        self._h5_file = None

    def _build_dual_branch_labels(self) -> tuple:
        """
        构建双分支标签

        Returns:
            labels_deception: [num_samples, num_deception_classes]
            labels_suppression: [num_samples, num_suppression_classes]
        """
        labels_deception = np.zeros((self.num_samples, self.num_deception_classes), dtype=np.float32)
        labels_suppression = np.zeros((self.num_samples, self.num_suppression_classes), dtype=np.float32)

        # 构建类别名称到索引的映射
        deception_name_to_idx = {name: i for i, name in enumerate(self.deception_classes)}
        suppression_name_to_idx = {name: i for i, name in enumerate(self.suppression_classes)}

        # "无XX干扰" 的索引
        none_deception_idx = len(self.deception_classes)  # 最后一个位置
        none_suppression_idx = len(self.suppression_classes)

        for i, meta in enumerate(self._metadata):
            if i >= self.num_samples:
                break

            jam_types = meta.get('jam_types', [])
            if isinstance(jam_types, str):
                jam_types = [jam_types] if jam_types else []
            elif not isinstance(jam_types, list):
                jam_types = []

            # 分类干扰类型
            has_deception = False
            has_suppression = False

            for jam_type in jam_types:
                jam_type = jam_type.strip() if isinstance(jam_type, str) else str(jam_type)

                if jam_type in deception_name_to_idx:
                    labels_deception[i, deception_name_to_idx[jam_type]] = 1.0
                    has_deception = True
                elif jam_type in suppression_name_to_idx:
                    labels_suppression[i, suppression_name_to_idx[jam_type]] = 1.0
                    has_suppression = True

            # 如果没有欺骗干扰，标记为 "无欺骗干扰"
            if not has_deception:
                labels_deception[i, none_deception_idx] = 1.0

            # 如果没有压制干扰，标记为 "无压制干扰"
            if not has_suppression:
                labels_suppression[i, none_suppression_idx] = 1.0

        return labels_deception, labels_suppression

    def _lazy_load(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.stft_file, 'r')
        return self._h5_file

    def __len__(self):
        return self.num_samples

    def _get_metadata(self, index: int) -> dict:
        if self._metadata is None:
            return None
        if isinstance(self._metadata, list) and index < len(self._metadata):
            meta = self._metadata[index].copy()
            if 'jam_types' in meta and isinstance(meta['jam_types'], int):
                meta['jam_types'] = [meta['jam_types']]
            return meta
        return None

    def __getitem__(self, index: int):
        h5_file = self._lazy_load()

        # 读取 STFT 数据
        raw_stft = h5_file[self.stft_var_name][:, :, index]
        stft_complex = raw_stft['real'] + 1j * raw_stft['imag']

        # 归一化
        if self.normalize_mode == 'per_sample':
            mag = np.abs(stft_complex)
            if self.normalize_method == 'max':
                ref = np.max(mag)
            elif self.normalize_method == 'p95':
                ref = np.percentile(mag, 95)
            else:
                ref = np.percentile(mag, 99)

            if ref > 0:
                stft_complex = stft_complex / ref
            stft_mag = np.abs(stft_complex).T
        else:
            stft_mag = np.abs(stft_complex).T
            stft_mag = np.clip(stft_mag, 0, self.norm_scale_mag) / self.norm_scale_mag

        # 构建三通道张量
        stft_tensor = torch.from_numpy(np.stack([stft_mag, stft_mag, stft_mag], axis=0)).float()

        # Resize
        if stft_tensor.shape[-2:] != (self.image_size, self.image_size):
            stft_tensor = torch.nn.functional.interpolate(
                stft_tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # 保存 CLIP 标准化前的能量图 (用于 Energy-Aware Masking)
        energy_map = None
        if self.augmentation is not None:
            energy_map = stft_tensor[0].clone()  # [H, W], 值域约 [0, 1]

        # CLIP 标准化
        if self.clip_norm is not None:
            stft_tensor = self.clip_norm(stft_tensor)

        # 获取 metadata (用于文本描述和数据增强)
        metadata = self._get_metadata(index)

        # 数据增强 (在 CLIP 标准化之后, mean≈0 所以 mask=0 表示无信息)
        if self.augmentation is not None:
            stft_tensor, _ = self.augmentation(stft_tensor, metadata=metadata, energy_map=energy_map)

        # 获取双分支标签
        label_deception = torch.from_numpy(self._labels_deception[index])
        label_suppression = torch.from_numpy(self._labels_suppression[index])

        return stft_tensor, label_deception, label_suppression, metadata


def collate_fn_dual_branch(batch):
    """
    双分支 Collate 函数

    Args:
        batch: list of (stft_image, label_deception, label_suppression, metadata)

    Returns:
        stft_images, text_tokens_deception, text_tokens_suppression,
        labels_deception, labels_suppression, texts_deception, texts_suppression, metadata_list
    """
    from multi.text_templates import get_dual_branch_descriptions
    import clip

    stft_images, labels_deception, labels_suppression, metadata_list = zip(*batch)

    # 堆叠
    stft_images = torch.stack(stft_images, dim=0)
    labels_deception = torch.stack(labels_deception, dim=0)
    labels_suppression = torch.stack(labels_suppression, dim=0)

    # 从配置获取干扰类型分组
    deception_classes = ["DFTJ", "ISRJ", "SMSPJ", "C&IJ", "CSJ"]
    suppression_classes = ["AJ", "BJ", "SJ", "NCJ", "NPJ", "NFMJ", "NPMJ", "NAMJ", "PJ"]

    # 生成双分支文本描述
    texts_deception = []
    texts_suppression = []

    for meta in metadata_list:
        jam_types = meta.get('jam_types', [])
        if isinstance(jam_types, str):
            jam_types = [jam_types] if jam_types else []
        elif not isinstance(jam_types, list):
            jam_types = []

        desc_deception, desc_suppression = get_dual_branch_descriptions(
            jam_types, deception_classes, suppression_classes
        )
        texts_deception.append(desc_deception)
        texts_suppression.append(desc_suppression)

    # Tokenize
    text_tokens_deception = clip.tokenize(texts_deception, truncate=True)
    text_tokens_suppression = clip.tokenize(texts_suppression, truncate=True)

    return (stft_images, text_tokens_deception, text_tokens_suppression,
            labels_deception, labels_suppression, texts_deception, texts_suppression, metadata_list)


def create_dual_branch_dataloaders(
    config: dict,
    normalization_stats: dict = None,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    load_test: bool = True,
) -> tuple:
    """
    创建双分支 CZSL 数据加载器

    Args:
        config: 配置字典
        normalization_stats: 归一化统计量
        batch_size: 批次大小
        num_workers: 数据加载 worker 数
        pin_memory: 是否 pin memory
        load_test: 是否加载测试集

    Returns:
        (train_loader, val_loader, test_loader, num_deception_classes, num_suppression_classes)
    """
    data_config = config.get('data', {})
    base_path = data_config.get('base_path')
    jnr_start = data_config.get('jnr_start', 10)
    jnr_end = data_config.get('jnr_end', 10)
    jnr_step = data_config.get('jnr_step', 1)
    stft_suffix = data_config.get('stft_suffix', 'echo_stfts')

    jnr_levels = list(range(jnr_start, jnr_end + 1, jnr_step))

    # 从配置获取干扰类型分组
    jamming_groups = config.get("jamming_groups", {})
    deception_classes = jamming_groups.get("deception", {}).get("classes", ["DFTJ", "ISRJ", "SMSPJ", "C&IJ", "CSJ"])
    suppression_classes = jamming_groups.get("suppression", {}).get("classes", ["AJ", "BJ", "SJ", "NCJ", "NPJ", "NFMJ", "NPMJ", "NAMJ", "PJ"])

    # 构建数据增强 (仅训练集)
    train_aug = None
    aug_config = config.get('augmentation', {})
    if aug_config.get('enabled', False):
        from multi.augmentation import STFTAugmentation
        train_aug = STFTAugmentation(aug_config)
        print(f"Data augmentation enabled: {[k for k, v in aug_config.items() if isinstance(v, dict) and v.get('enabled')]}")

    def load_split(split_name: str, required: bool = True):
        datasets = []
        # 仅训练集使用增强
        aug = train_aug if split_name == 'train' else None

        for jnr in jnr_levels:
            jnr_folder = f"JNR_{'+' if jnr >= 0 else ''}{jnr}"
            data_folder = os.path.join(base_path, jnr_folder)

            stft_file = os.path.join(data_folder, f'{split_name}_{stft_suffix}.mat')
            metadata_file = os.path.join(data_folder, f'{split_name}_echo_metadata.json')

            if not os.path.exists(stft_file):
                print(f"Warning: STFT data not found for {jnr_folder}/{split_name}, skipping...")
                continue

            if not os.path.exists(metadata_file):
                print(f"Warning: Metadata not found for {jnr_folder}/{split_name}, skipping...")
                continue

            dataset = DualBranchSTFTDataset(
                stft_file=stft_file,
                metadata_file=metadata_file,
                normalization_stats=normalization_stats,
                deception_classes=deception_classes,
                suppression_classes=suppression_classes,
                augmentation=aug,
            )
            datasets.append(dataset)
            print(f"Loaded {split_name} data from {jnr_folder}: {len(dataset)} samples")

        if not datasets:
            if required:
                raise ValueError(f"No data found for {split_name} split!")
            else:
                print(f"Warning: No data found for {split_name} split, returning None")
                return None

        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    # 创建数据集
    train_dataset = load_split('train', required=True)
    val_dataset = load_split('val', required=True)
    test_dataset = load_split('test', required=False) if load_test else None

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_dual_branch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_dual_branch,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn_dual_branch,
        )

    num_deception_classes = len(deception_classes) + 1
    num_suppression_classes = len(suppression_classes) + 1

    return train_loader, val_loader, test_loader, num_deception_classes, num_suppression_classes


def create_czsl_dataloaders(
    config: dict,
    normalization_stats: dict = None,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    load_test: bool = True,
    model_type: str = "clip",
    processor=None,
) -> tuple:
    """
    创建 CZSL 数据加载器 (train/val/test)

    Args:
        config: 配置字典
        normalization_stats: 归一化统计量
        batch_size: 批次大小
        num_workers: 数据加载 worker 数
        pin_memory: 是否 pin memory
        load_test: 是否加载测试集
        model_type: 模型类型 ("clip" 或 "siglip-xxx")
        processor: SigLIP processor（SigLIP 模型需要）

    Returns:
        (train_loader, val_loader, test_loader, num_classes)
        如果 load_test=False 或 test 数据不存在，test_loader 为 None
    """

    data_config = config.get('data', {})
    base_path = data_config.get('base_path')
    jnr_start = data_config.get('jnr_start', 10)
    jnr_end = data_config.get('jnr_end', 10)
    jnr_step = data_config.get('jnr_step', 1)
    stft_suffix = data_config.get('stft_suffix', 'echo_stfts')

    # 时域数据配置
    use_time_domain = config.get('use_time_domain', False)
    time_seq_len = data_config.get('time_seq_len', 2048)
    time_var_name = data_config.get('time_var_name', 'raw_time')

    # 特征条件上下文配置
    use_feature_context = config.get('use_feature_context', False)
    feature_norm_stats_path = config.get('feature_norm_stats_path', None)
    feature_norm_stats = None
    if use_feature_context and feature_norm_stats_path and os.path.exists(feature_norm_stats_path):
        with open(feature_norm_stats_path, 'r', encoding='utf-8') as f:
            feature_norm_stats = json.load(f)
        print(f"Loaded feature normalization stats from {feature_norm_stats_path}")

    # JNR 级别
    jnr_levels = list(range(jnr_start, jnr_end + 1, jnr_step))

    # 类别名称
    class_names = [cls['name'] for cls in config.get('jamming_classes', [])]

    # 构建数据增强 (仅训练集)
    train_aug = None
    aug_config = config.get('augmentation', {})
    if aug_config.get('enabled', False):
        from multi.augmentation import STFTAugmentation
        train_aug = STFTAugmentation(aug_config)
        print(f"Data augmentation enabled: {[k for k, v in aug_config.items() if isinstance(v, dict) and v.get('enabled')]}")

    # 加载所有 JNR 级别的数据
    def load_split(split_name: str, required: bool = True):
        aug = train_aug if split_name == 'train' else None
        """加载单个 split 的数据集

        Args:
            split_name: split 名称
            required: 是否必需，如果为 False 则在找不到时返回 None
        """
        datasets = []

        for jnr in jnr_levels:
            jnr_folder = f"JNR_{'+' if jnr >= 0 else ''}{jnr}"
            data_folder = os.path.join(base_path, jnr_folder)

            stft_file = os.path.join(data_folder, f'{split_name}_{stft_suffix}.mat')
            metadata_file = os.path.join(data_folder, f'{split_name}_echo_metadata.json')
            features_file = os.path.join(data_folder, f'{split_name}_echo_features.json')

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
                features_file=features_file if use_feature_context else None,
                feature_norm_stats=feature_norm_stats,
                use_feature_context=use_feature_context,
                augmentation=aug,
            )
            datasets.append(stft_dataset)
            print(f"Loaded {split_name} data from {jnr_folder}: {len(stft_dataset)} samples")

        if not datasets:
            if required:
                raise ValueError(f"No data found for {split_name} split!")
            else:
                print(f"Warning: No data found for {split_name} split, returning None")
                return None

        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    # 创建数据集
    train_dataset = load_split('train', required=True)
    val_dataset = load_split('val', required=True)
    test_dataset = load_split('test', required=False) if load_test else None

    # 创建 collate 函数（支持 CLIP 和 SigLIP）
    _collate_fn = create_collate_fn(model_type, processor)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_fn,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=_collate_fn,
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
