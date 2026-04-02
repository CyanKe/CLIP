"""
数据加载模块 - 支持CLIP模型的STFT数据集
定义STFT数据集类，支持单通道、三通道和CLIP适配
"""
import os
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader
from torchvision import transforms


class STFTDataset3(Dataset):
    """三通道数据集 (实部、虚部、幅度) - 支持多进程DataLoader"""
    def __init__(self, stft_file, label_file, stft_var_name='all_stfts', label_var_name='all_label'):
        super().__init__()
        # 只保存文件路径，不立即打开文件（支持多进程）
        self.stft_file = stft_file
        self.label_file = label_file
        self.stft_var_name = stft_var_name
        self.label_var_name = label_var_name

        # 延迟加载：先读取元数据
        with h5py.File(stft_file, 'r') as f:
            self.num_samples = f[stft_var_name].shape[2]
        with h5py.File(label_file, 'r') as f:
            self.num_classes = f[label_var_name].shape[0]

        # 用于缓存的文件句柄（每个worker独立）
        self._h5_files = None

    def _lazy_load(self):
        """延迟加载h5文件（每个worker进程独立调用）"""
        if self._h5_files is None:
            self._h5_files = {
                'stft': h5py.File(self.stft_file, 'r'),
                'label': h5py.File(self.label_file, 'r')
            }
        return self._h5_files

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # 延迟加载h5文件
        h5_files = self._lazy_load()

        stft_struct = h5_files['stft'][self.stft_var_name][:, :, index]
        one_hot_label_np = h5_files['label'][self.label_var_name][:, index]

        stft_complex = stft_struct.view(np.complex128)
        stft_real = np.real(stft_complex).T
        stft_imag = np.imag(stft_complex).T
        stft_Modulus = np.abs(stft_complex).T

        stft_multi_channel = np.stack([stft_real, stft_imag, stft_Modulus], axis=0)
        stft_tensor = torch.from_numpy(stft_multi_channel).float()
        label_tensor = torch.from_numpy(one_hot_label_np).float()

        return stft_tensor, label_tensor

    def close(self):
        """关闭h5文件句柄"""
        if self._h5_files is not None:
            self._h5_files['stft'].close()
            self._h5_files['label'].close()
            self._h5_files = None

    def __del__(self):
        """析构时关闭文件"""
        self.close()


class CLIPSTFTDataset(Dataset):
    """
    适配CLIP模型的STFT数据集
    - 输出224x224分辨率的图像
    - 应用CLIP标准的数据标准化
    - 支持多进程DataLoader（延迟加载h5文件）
    """
    # CLIP标准化参数
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        stft_file,
        label_file,
        stft_var_name='all_stfts',
        label_var_name='all_label',
        image_size=224,
        normalize_to_clip=True,
        channel_mode='real_imag_mag'  # 'real_imag_mag', 'real_imag_phase', 'magnitude_only'
    ):
        """
        初始化CLIP适配的数据集

        Args:
            stft_file: STFT数据文件路径
            label_file: 标签文件路径
            stft_var_name: STFT数据变量名
            label_var_name: 标签变量名
            image_size: 输出图像尺寸（默认224x224，CLIP标准）
            normalize_to_clip: 是否应用CLIP标准化
            channel_mode: 通道模式
                - 'real_imag_mag': 实部、虚部、幅度
                - 'real_imag_phase': 实部、虚部、相位
                - 'magnitude_only': 仅幅度（复制三通道）
        """
        super().__init__()
        # 只保存文件路径，不立即打开文件（支持多进程）
        self.stft_file = stft_file
        self.label_file = label_file
        self.stft_var_name = stft_var_name
        self.label_var_name = label_var_name

        # 延迟加载：先读取元数据
        with h5py.File(stft_file, 'r') as f:
            self.num_samples = f[stft_var_name].shape[2]
        with h5py.File(label_file, 'r') as f:
            self.num_classes = f[label_var_name].shape[0]

        self.image_size = image_size
        self.normalize_to_clip = normalize_to_clip
        self.channel_mode = channel_mode

        # 创建预处理transform
        if normalize_to_clip:
            self.transform = transforms.Normalize(mean=self.CLIP_MEAN, std=self.CLIP_STD)
        else:
            self.transform = None

        # 用于缓存的文件句柄（每个worker独立）
        self._h5_files = None

    def _lazy_load(self):
        """延迟加载h5文件（每个worker进程独立调用）"""
        if self._h5_files is None:
            self._h5_files = {
                'stft': h5py.File(self.stft_file, 'r'),
                'label': h5py.File(self.label_file, 'r')
            }
        return self._h5_files

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # 延迟加载h5文件
        h5_files = self._lazy_load()

        # 读取原始数据
        stft_struct = h5_files['stft'][self.stft_var_name][:, :, index]
        one_hot_label_np = h5_files['label'][self.label_var_name][:, index]

        # 转换为复数
        stft_complex = stft_struct.view(np.complex128)

        # 根据通道模式构建三通道图像
        stft_real = np.real(stft_complex).T
        stft_imag = np.imag(stft_complex).T

        if self.channel_mode == 'real_imag_mag':
            stft_channel3 = np.abs(stft_complex).T
        elif self.channel_mode == 'real_imag_phase':
            stft_channel3 = np.angle(stft_complex).T
        else:  # magnitude_only
            stft_channel3 = np.abs(stft_complex).T

        # 构建三通道数据
        stft_multi_channel = np.stack([stft_channel3, stft_channel3, stft_channel3], axis=0)

        # 转换为tensor
        stft_tensor = torch.from_numpy(stft_multi_channel).float()

        # Resize到目标尺寸
        if stft_tensor.shape[-2:] != (self.image_size, self.image_size):
            stft_tensor = torch.nn.functional.interpolate(
                stft_tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # 归一化到[0, 1]范围
        for c in range(3):
            channel = stft_tensor[c]
            c_min, c_max = channel.min(), channel.max()
            if c_max > c_min:
                stft_tensor[c] = (channel - c_min) / (c_max - c_min)
            else:
                stft_tensor[c] = 0.5

        # 应用CLIP标准化
        if self.transform is not None:
            stft_tensor = self.transform(stft_tensor)

        label_tensor = torch.from_numpy(one_hot_label_np).float()

        return stft_tensor, label_tensor

    def close(self):
        """关闭h5文件句柄"""
        if self._h5_files is not None:
            self._h5_files['stft'].close()
            self._h5_files['label'].close()
            self._h5_files = None

    def __del__(self):
        """析构时关闭文件"""
        self.close()

class STFTDataset11(Dataset):
    """三通道数据集 (实部、虚部、相位) - 支持11类"空类别"输出"""
    def __init__(self, stft_file, label_file, stft_var_name='all_stfts', 
                 label_var_name='all_label', num_classes=11):
        super().__init__()
        self.stft_h5 = h5py.File(stft_file, 'r')
        self.label_h5 = h5py.File(label_file, 'r')
        
        self.stft_data = self.stft_h5[stft_var_name]
        self.label_data = self.label_h5[label_var_name]
        
        self.num_samples = self.stft_data.shape[2]
        
        # 关键修改1: 设定总类别数为11
        self.num_classes = num_classes
        
        # 关键修改2: 记录原始标签维度
        self.original_label_dim = self.label_data.shape[0]
        
        # 打印信息，帮助调试
        print(f"STFT数据维度: {self.stft_data.shape}")  # (freq_bins, time_frames, n_samples)
        print(f"原始标签维度: {self.original_label_dim}")
        print(f"目标标签维度: {self.num_classes}")
        
        # 验证：原始标签数应该小于等于目标类别数
        if self.original_label_dim > self.num_classes:
            raise ValueError(f"原始标签维度({self.original_label_dim})大于目标类别数({self.num_classes})")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        # 1. 获取STFT数据（保持不变）
        stft_struct = self.stft_data[:, :, index]
        
        # 2. 获取原始标签（原始为9维）
        original_label = self.label_data[:, index]  # 形状: (9,)
        
        # 3. 处理STFT：复数转三通道数据
        # 将平坦的复数数组重塑为复数矩阵
        # 注意：根据你的数据格式，可能需要调整reshape
        stft_complex = stft_struct.view(np.complex128)
        
        # 如果stft_struct是平坦的一维数组，需要先重塑
        # 这里假设形状已经正确，如果报错请根据实际情况调整
        # 常见情况：需要先 reshape 到 (freq_bins, time_frames)
        if stft_complex.ndim == 1:
            # 尝试推断形状
            n_freq = int(np.sqrt(len(stft_complex)))
            stft_complex = stft_complex.reshape(n_freq, n_freq)  # 假设是平方矩阵
        
        stft_real = np.real(stft_complex).T  # 转置，通常时间轴在第0维
        stft_imag = np.imag(stft_complex).T
        stft_phase = np.angle(stft_complex).T
        
        stft_multi_channel = np.stack([stft_real, stft_imag, stft_phase], axis=0)
        stft_tensor = torch.from_numpy(stft_multi_channel).float()
        
        # 4. 处理标签：从9维扩展到11维
        # 方案A: 前9个位置保持原样，后2个位置补0（推荐）
        if self.original_label_dim < self.num_classes:
            extended_label = np.zeros(self.num_classes, dtype=np.float32)
            extended_label[:self.original_label_dim] = original_label
            
            # 验证：如果原始标签中有'空类别'的数据（标签为1），需要处理
            # 但根据你的描述，原本只有9类数据，最后2类在训练集中应该没有正样本
            
        else:
            # 如果已经是11维，直接使用（可能用于训练集）
            extended_label = original_label.astype(np.float32)
        
        # 5. 转换为PyTorch张量
        label_tensor = torch.from_numpy(extended_label).float()
        
        return stft_tensor, label_tensor
    
    def close(self):
        """关闭h5文件以释放资源"""
        self.stft_h5.close()
        self.label_h5.close()


class STFTDataset(Dataset):
    """单通道数据集 (幅度)"""
    def __init__(self, stft_file, label_file, stft_var_name='all_stfts', label_var_name='all_label'):
        super().__init__()
        self.stft_h5 = h5py.File(stft_file, 'r')
        self.label_h5 = h5py.File(label_file, 'r')
        
        self.stft_data = self.stft_h5[stft_var_name]
        self.label_data = self.label_h5[label_var_name]
        
        self.num_samples = self.stft_data.shape[2]
        self.num_classes = self.label_data.shape[0]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        stft_struct = self.stft_data[:, :, index]
        one_hot_label_np = self.label_data[:, index]
        
        stft_complex = stft_struct.view(np.complex128)
        stft_magnitude = np.abs(stft_complex).T
        
        stft_tensor = torch.from_numpy(stft_magnitude).float().unsqueeze(0)
        label_tensor = torch.from_numpy(one_hot_label_np).float()
        
        return stft_tensor, label_tensor
    
    def close(self):
        self.stft_h5.close()
        self.label_h5.close()


def load_multi_jnr_dataset(base_path, jnr_start, jnr_end, jnr_step, n_samples_per_dataset=450,compatibility = False,OpenSet = False):
    """
    加载多个JNR级别的数据集并合并
    
    Args:
        base_path: 数据基础路径
        jnr_start, jnr_end, jnr_step: JNR范围
        n_samples_per_dataset: 每个数据集采样数
    
    Returns:
        tuple: (combined_train, combined_test, combined_val, num_classes)
    """
    jnr_numbers = range(jnr_start, jnr_end + 1, jnr_step)
    jnr_levels = [f"+{jnr}" if jnr >= 0 else str(jnr) for jnr in jnr_numbers]
    
    train_datasets_subsets = []
    all_train_datasets = []
    all_test_datasets = []
    all_val_datasets = []
    
    for jnr in jnr_levels:
        jnr_folder_name = f"JNR_{jnr}"
        data_folder_path = os.path.join(base_path, jnr_folder_name)
        
        train_stfts = os.path.join(data_folder_path, 'train_echo_stfts.mat')
        train_label = os.path.join(data_folder_path, 'train_echo_label.mat')
        test_stfts = os.path.join(data_folder_path, 'test_echo_stfts.mat')
        test_label = os.path.join(data_folder_path, 'test_echo_label.mat')
        val_stfts = os.path.join(data_folder_path, 'val_echo_stfts.mat')
        val_label = os.path.join(data_folder_path, 'val_echo_label.mat')
        
        if not (os.path.exists(train_stfts) and os.path.exists(train_label)):
            continue
        
        # 加载数据集
        if compatibility :
            full_train = STFTDataset11(train_stfts, train_label)
            full_test  = STFTDataset11(test_stfts, test_label)
            full_val   = STFTDataset11(val_stfts, val_label)
        else :
            full_train = STFTDataset3(train_stfts, train_label)
            full_test  = STFTDataset3(test_stfts, test_label)
            full_val   = STFTDataset3(val_stfts, val_label)
        # 采样训练集
        n_samples = min(n_samples_per_dataset, len(full_train))
        train_subset = Subset(full_train, indices=range(n_samples))
        train_datasets_subsets.append(train_subset)
        
        all_train_datasets.append(full_train)
        all_test_datasets.append(full_test)
        all_val_datasets.append(full_val)
    
    if not train_datasets_subsets:
        raise ValueError("没有成功加载任何数据，请检查路径")
    
    # 合并数据集
    if OpenSet:
        combined_train = ConcatDataset(train_datasets_subsets)
    else:
        combined_train = ConcatDataset(all_train_datasets)  # 考虑是用完整集还是子集

    combined_test = ConcatDataset(all_test_datasets)
    combined_val = ConcatDataset(all_val_datasets)
    
    num_classes = all_test_datasets[0].num_classes
    
    print(f"加载完成: 训练集 {len(combined_train)} 样本, 测试集 {len(combined_test)} 样本")
    print(f"类别数: {num_classes}")

    return combined_train, combined_test, combined_val, num_classes, jnr_levels


def load_clip_dataset(
    base_path,
    jnr_start,
    jnr_end,
    jnr_step,
    image_size=224,
    normalize_to_clip=True,
    channel_mode='real_imag_mag'
):
    """
    加载适配CLIP的数据集

    Args:
        base_path: 数据基础路径
        jnr_start, jnr_end, jnr_step: JNR范围
        image_size: 输出图像尺寸
        normalize_to_clip: 是否应用CLIP标准化
        channel_mode: 通道模式

    Returns:
        tuple: (combined_train, combined_test, combined_val, num_classes, jnr_levels)
    """
    jnr_numbers = range(jnr_start, jnr_end + 1, jnr_step)
    jnr_levels = [f"+{jnr}" if jnr >= 0 else str(jnr) for jnr in jnr_numbers]

    all_train_datasets = []
    all_test_datasets = []
    all_val_datasets = []

    for jnr in jnr_levels:
        jnr_folder_name = f"JNR_{jnr}"
        data_folder_path = os.path.join(base_path, jnr_folder_name)

        train_stfts = os.path.join(data_folder_path, 'train_echo_stfts.mat')
        train_label = os.path.join(data_folder_path, 'train_echo_label.mat')
        test_stfts = os.path.join(data_folder_path, 'test_echo_stfts.mat')
        test_label = os.path.join(data_folder_path, 'test_echo_label.mat')
        val_stfts = os.path.join(data_folder_path, 'val_echo_stfts.mat')
        val_label = os.path.join(data_folder_path, 'val_echo_label.mat')

        if not (os.path.exists(train_stfts) and os.path.exists(train_label)):
            print(f"Warning: Data not found for JNR {jnr}, skipping...")
            continue

        # 使用CLIP适配的数据集
        full_train = CLIPSTFTDataset(
            train_stfts, train_label,
            image_size=image_size,
            normalize_to_clip=normalize_to_clip,
            channel_mode=channel_mode
        )
        full_test = CLIPSTFTDataset(
            test_stfts, test_label,
            image_size=image_size,
            normalize_to_clip=normalize_to_clip,
            channel_mode=channel_mode
        )
        full_val = CLIPSTFTDataset(
            val_stfts, val_label,
            image_size=image_size,
            normalize_to_clip=normalize_to_clip,
            channel_mode=channel_mode
        )

        all_train_datasets.append(full_train)
        all_test_datasets.append(full_test)
        all_val_datasets.append(full_val)

    if not all_train_datasets:
        raise ValueError("没有成功加载任何数据，请检查路径")

    # 合并数据集
    combined_train = ConcatDataset(all_train_datasets)
    combined_test = ConcatDataset(all_test_datasets)
    combined_val = ConcatDataset(all_val_datasets)

    num_classes = all_test_datasets[0].num_classes

    print(f"加载完成: 训练集 {len(combined_train)} 样本, 测试集 {len(combined_test)} 样本, 验证集 {len(combined_val)} 样本")
    print(f"类别数: {num_classes}")
    print(f"JNR级别: {jnr_levels}")

    return combined_train, combined_test, combined_val, num_classes, jnr_levels


def create_dataloaders(config, use_clip_dataset=True):
    """
    根据配置创建数据加载器

    Args:
        config: 配置字典
        use_clip_dataset: 是否使用CLIP适配数据集

    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes)
    """
    data_config = config.get("data", {})
    train_config = config.get("train", {})

    if use_clip_dataset:
        # 使用CLIP适配的数据集
        train_dataset, test_dataset, val_dataset, num_classes, jnr_levels = load_clip_dataset(
            base_path=data_config.get("base_path"),
            jnr_start=data_config.get("jnr_start", 0),
            jnr_end=data_config.get("jnr_end", 40),
            jnr_step=data_config.get("jnr_step", 10),
            image_size=data_config.get("image_size", 224),
            normalize_to_clip=True,
            channel_mode='real_imag_mag'
        )
    else:
        # 使用原始数据集
        train_dataset, test_dataset, val_dataset, num_classes, jnr_levels = load_multi_jnr_dataset(
            base_path=data_config.get("base_path"),
            jnr_start=data_config.get("jnr_start", 0),
            jnr_end=data_config.get("jnr_end", 40),
            jnr_step=data_config.get("jnr_step", 10)
        )

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_config.get("batch_size", 32),
        shuffle=True,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True)
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_config.get("batch_size", 32),
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True)
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=train_config.get("batch_size", 32),
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True)
    )

    return train_loader, val_loader, test_loader, num_classes


# ============================================================================
# CZSL 数据集类 - 支持图像-文本对格式
# ============================================================================

class CZSLSTFTDataset(Dataset):
    """
    CZSL数据集 - 支持组合零样本学习
    返回: (image, label, text_description)
    使用metadata_template生成文本描述
    """
    # CLIP标准化参数
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        stft_file,
        label_file,
        class_names: list,
        stft_var_name='all_stfts',
        label_var_name='all_label',
        image_size=224,
        normalize_to_clip=True,
        channel_mode='real_imag_mag'
    ):
        """
        初始化CZSL数据集

        Args:
            stft_file: STFT数据文件路径
            label_file: 标签文件路径
            class_names: 类别名称列表
            stft_var_name: STFT数据变量名
            label_var_name: 标签变量名
            image_size: 输出图像尺寸
            normalize_to_clip: 是否应用CLIP标准化
            channel_mode: 通道模式
        """
        super().__init__()
        self.stft_file = stft_file
        self.label_file = label_file
        self.stft_var_name = stft_var_name
        self.label_var_name = label_var_name
        self.class_names = class_names
        self.image_size = image_size
        self.normalize_to_clip = normalize_to_clip
        self.channel_mode = channel_mode

        # 延迟加载：先读取元数据
        with h5py.File(stft_file, 'r') as f:
            self.num_samples = f[stft_var_name].shape[2]
        with h5py.File(label_file, 'r') as f:
            self.num_classes = f[label_var_name].shape[0]

        # 创建预处理transform
        if normalize_to_clip:
            self.transform = transforms.Normalize(mean=self.CLIP_MEAN, std=self.CLIP_STD)
        else:
            self.transform = None

        # 用于缓存的文件句柄
        self._h5_files = None

    def _lazy_load(self):
        """延迟加载h5文件"""
        if self._h5_files is None:
            self._h5_files = {
                'stft': h5py.File(self.stft_file, 'r'),
                'label': h5py.File(self.label_file, 'r')
            }
        return self._h5_files

    def __len__(self):
        return self.num_samples

    def _build_text_description(self, label_np):
        """根据标签生成文本描述，使用metadata_template"""
        from multi.metadata_template import JAM_TYPE_NAMES, VISUAL_TEMPLATES

        active_indices = np.where(label_np == 1)[0]

        if len(active_indices) == 0:
            return "a radar signal with no jamming"

        active_classes = [self.class_names[i] for i in active_indices]

        # 生成描述
        descs = []
        for cls_name in active_classes:
            # 查找干扰类型编号
            jam_type = None
            for k, v in JAM_TYPE_NAMES.items():
                if v == cls_name:
                    jam_type = k
                    break

            if jam_type:
                template = VISUAL_TEMPLATES.get(jam_type, {'base': cls_name, 'param': {}})
                descs.append(f"{cls_name} looks like {template['base']}")
            else:
                descs.append(f"a radar signal with {cls_name}")

        if len(descs) == 1:
            return descs[0]
        else:
            return ', '.join(descs)

    def __getitem__(self, index):
        h5_files = self._lazy_load()

        # 读取STFT数据
        stft_struct = h5_files['stft'][self.stft_var_name][:, :, index]
        one_hot_label_np = h5_files['label'][self.label_var_name][:, index]

        # 转换为复数
        stft_complex = stft_struct.view(np.complex128)

        # 构建三通道图像
        stft_real = np.real(stft_complex).T
        stft_imag = np.imag(stft_complex).T

        if self.channel_mode == 'real_imag_mag':
            stft_channel3 = np.abs(stft_complex).T
        elif self.channel_mode == 'real_imag_phase':
            stft_channel3 = np.angle(stft_complex).T
        else:
            stft_channel3 = np.abs(stft_complex).T

        stft_multi_channel = np.stack([stft_real, stft_imag, stft_channel3], axis=0)
        stft_tensor = torch.from_numpy(stft_multi_channel).float()

        # Resize到目标尺寸
        if stft_tensor.shape[-2:] != (self.image_size, self.image_size):
            stft_tensor = torch.nn.functional.interpolate(
                stft_tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # 归一化到[0, 1]范围
        for c in range(3):
            channel = stft_tensor[c]
            c_min, c_max = channel.min(), channel.max()
            if c_max > c_min:
                stft_tensor[c] = (channel - c_min) / (c_max - c_min)
            else:
                stft_tensor[c] = 0.5

        # 应用CLIP标准化
        if self.transform is not None:
            stft_tensor = self.transform(stft_tensor)

        label_tensor = torch.from_numpy(one_hot_label_np).float()

        # 生成文本描述
        text_description = self._build_text_description(one_hot_label_np)

        return stft_tensor, label_tensor, text_description

    def close(self):
        """关闭h5文件句柄"""
        if self._h5_files is not None:
            self._h5_files['stft'].close()
            self._h5_files['label'].close()
            self._h5_files = None

    def __del__(self):
        self.close()


class CZSLSTFTDatasetWithMetadata(Dataset):
    """
    CZSL数据集 - 支持metadata和样本级描述
    返回: (image, label, text_description, metadata_dict)
    使用metadata_template生成文本描述
    """
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    # 干扰类型名称映射
    JAM_TYPE_NAMES = {
        1: 'DFTJ', 2: 'ISRJ', 3: 'RGPO', 4: 'VGPO',
        5: 'AJ', 6: 'BJ', 7: 'SJ', 8: 'NCJ',
        9: 'NPJ', 10: 'SMSPJ', 11: 'C&IJ', 12: 'NFMJ',
        13: 'NPMJ', 14: 'NAMJ', 15: 'CSJ', 16: 'PJ'
    }

    def __init__(
        self,
        stft_file,
        label_file,
        metadata_file=None,
        class_names: list = None,
        stft_var_name='all_stfts',
        label_var_name='all_label',
        metadata_var_name='all_metadata',
        image_size=224,
        normalize_to_clip=True,
        channel_mode='real_imag_mag',
        use_abstract_description=True
    ):
        """
        初始化支持metadata的CZSL数据集

        Args:
            stft_file: STFT数据文件路径
            label_file: 标签文件路径
            metadata_file: metadata文件路径 (可选)
            class_names: 类别名称列表
            stft_var_name: STFT数据变量名
            label_var_name: 标签变量名
            metadata_var_name: metadata变量名
            image_size: 输出图像尺寸
            normalize_to_clip: 是否应用CLIP标准化
            channel_mode: 通道模式
            use_abstract_description: 是否使用抽象描述
        """
        super().__init__()
        self.stft_file = stft_file
        self.label_file = label_file
        self.metadata_file = metadata_file
        self.stft_var_name = stft_var_name
        self.label_var_name = label_var_name
        self.metadata_var_name = metadata_var_name
        self.class_names = class_names or []
        self.image_size = image_size
        self.normalize_to_clip = normalize_to_clip
        self.channel_mode = channel_mode
        self.use_abstract_description = use_abstract_description

        # 延迟加载：先读取元数据
        with h5py.File(stft_file, 'r') as f:
            self.num_samples = f[stft_var_name].shape[2]
        with h5py.File(label_file, 'r') as f:
            self.num_classes = f[label_var_name].shape[0]

        # 创建预处理transform
        if normalize_to_clip:
            self.transform = transforms.Normalize(mean=self.CLIP_MEAN, std=self.CLIP_STD)
        else:
            self.transform = None

        # 用于缓存的文件句柄
        self._h5_files = None
        self._metadata = None
        self._metadata_fields = []  # 字段名称列表 (HDF5格式)
        self._metadata_list = []    # 样本列表 (JSON格式)
        self._metadata_num_samples = 0

        # 加载metadata（如果存在）
        if metadata_file and os.path.exists(metadata_file):
            self._load_metadata(metadata_file)

    def _load_metadata(self, metadata_file):
        """加载metadata文件（支持JSON和MATLAB v7.3 HDF5格式）"""
        # 检查文件扩展名
        if metadata_file.endswith('.json'):
            self._load_metadata_json(metadata_file)
        else:
            self._load_metadata_matlab(metadata_file)

    def _load_metadata_json(self, metadata_file):
        """加载JSON格式的metadata文件"""
        import json
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                # JSON格式：直接是样本列表 [{...}, {...}, ...]
                self._metadata_list = json.load(f)

            if isinstance(self._metadata_list, list):
                self._metadata_num_samples = len(self._metadata_list)
                self._metadata = 'json_list'  # 标记为JSON列表格式
                print(f"Loaded metadata from {metadata_file} (JSON format, {self._metadata_num_samples} samples)")
            else:
                print(f"Warning: JSON metadata should be a list, got {type(self._metadata_list)}")
                self._metadata = None

        except Exception as e:
            print(f"Warning: Failed to load JSON metadata: {e}")
            self._metadata = None

    def _load_metadata_matlab(self, metadata_file):
        """加载MATLAB格式的metadata文件"""
        try:
            import h5py
            with h5py.File(metadata_file, 'r') as f:
                if self.metadata_var_name not in f:
                    print(f"Warning: {self.metadata_var_name} not found in {metadata_file}")
                    self._metadata = None
                    return

                meta_group = f[self.metadata_var_name]

                # MATLAB v7.3 struct数组存储为HDF5 Group
                # 每个字段是一个独立的Dataset
                if isinstance(meta_group, h5py.Group):
                    # 只存储字段名称和形状信息，不读取实际数据
                    # 因为HDF5引用需要保持文件打开才能有效
                    self._metadata_fields = list(meta_group.keys())
                    print(f"Metadata fields found: {self._metadata_fields}")

                    # 检测样本数量（从第一个字段获取形状）
                    first_field = meta_group[self._metadata_fields[0]]
                    self._metadata_num_samples = first_field.shape[0]

                    print(f"Loaded metadata from {metadata_file} (HDF5 Group format, {self._metadata_num_samples} samples)")
                    self._metadata = 'hdf5_group'  # 标记为HDF5 Group格式
                else:
                    # 直接是Dataset（旧格式）
                    self._metadata = meta_group[:]
                    print(f"Loaded metadata from {metadata_file} (HDF5 Dataset format)")

        except Exception as e:
            print(f"HDF5 loading failed: {e}")
            # 如果h5py失败，尝试scipy.io（支持旧版MATLAB格式）
            try:
                import scipy.io as sio
                data = sio.loadmat(metadata_file)
                if self.metadata_var_name in data:
                    self._metadata = data[self.metadata_var_name]
                    print(f"Loaded metadata from {metadata_file} (MATLAB format)")
                else:
                    print(f"Warning: {self.metadata_var_name} not found in {metadata_file}")
                    self._metadata = None
            except Exception as e2:
                print(f"Warning: Failed to load metadata: {e2}")
                self._metadata = None

    def _lazy_load(self):
        """延迟加载h5文件"""
        if self._h5_files is None:
            self._h5_files = {
                'stft': h5py.File(self.stft_file, 'r'),
                'label': h5py.File(self.label_file, 'r')
            }
            # 如果metadata文件存在且是HDF5 Group格式，也加载它
            if self._metadata == 'hdf5_group' and self.metadata_file:
                self._h5_files['meta'] = h5py.File(self.metadata_file, 'r')
        return self._h5_files

    def __len__(self):
        return self.num_samples

    def _parse_metadata(self, index):
        """解析metadata结构为Python字典（支持JSON和MATLAB v7.3 HDF5格式）"""
        if self._metadata is None:
            return None

        try:
            # JSON列表格式（推荐）
            if self._metadata == 'json_list':
                if index < len(self._metadata_list):
                    meta = self._metadata_list[index]
                    # 直接返回，JSON已经是Python字典格式
                    # 确保jam_types是列表
                    if isinstance(meta.get('jam_types'), int):
                        meta['jam_types'] = [meta['jam_types']]
                    return meta
                return None

            # HDF5 Group格式（MATLAB v7.3）
            if self._metadata == 'hdf5_group':
                metadata_dict = {}

                # 获取metadata h5文件句柄
                h5_files = self._lazy_load()
                h5_meta = h5_files.get('meta')
                if h5_meta is None:
                    return None

                # 获取metadata group
                meta_group = h5_meta[self.metadata_var_name]
                fields = self._metadata_fields  # 现在是字段名称列表

                # MATLAB v7.3格式：每个字段是cell array，元素是HDF5 object reference
                # 需要解引用来获取实际数据

                # sample_idx
                if 'sample_idx' in fields:
                    field_data = meta_group['sample_idx']
                    ref = field_data[index, 0]
                    if isinstance(ref, h5py.h5r.Reference):
                        target = h5_meta[ref]
                        metadata_dict['sample_idx'] = int(target[0, 0])
                    else:
                        metadata_dict['sample_idx'] = int(ref)

                # jam_types
                if 'jam_types' in fields:
                    field_data = meta_group['jam_types']
                    ref = field_data[index, 0]
                    if isinstance(ref, h5py.h5r.Reference):
                        target = h5_meta[ref]
                        # target可能是单个数值或数组
                        data = target[:]
                        if data.ndim == 2 and data.shape == (1, 1):
                            metadata_dict['jam_types'] = [int(data[0, 0])] if data[0, 0] > 0 else []
                        else:
                            # 多个干扰类型
                            jam_types_flat = data.flatten()
                            metadata_dict['jam_types'] = [int(j) for j in jam_types_flat if j > 0]
                    else:
                        metadata_dict['jam_types'] = [int(ref)] if ref > 0 else []

                # JNR
                if 'JNR' in fields:
                    field_data = meta_group['JNR']
                    ref = field_data[index, 0]
                    if isinstance(ref, h5py.h5r.Reference):
                        target = h5_meta[ref]
                        metadata_dict['JNR'] = float(target[0, 0])
                    else:
                        metadata_dict['JNR'] = float(ref)

                # pos
                if 'pos' in fields:
                    field_data = meta_group['pos']
                    ref = field_data[index, 0]
                    if isinstance(ref, h5py.h5r.Reference):
                        target = h5_meta[ref]
                        metadata_dict['pos'] = int(target[0, 0]) if target.size > 0 else 0
                    else:
                        metadata_dict['pos'] = int(ref) if ref else 0

                # jam_params - 可能是Group或Dataset
                metadata_dict['jam_params'] = {}
                if 'jam_params' in fields:
                    field_data = meta_group['jam_params']
                    ref = field_data[index, 0]
                    if isinstance(ref, h5py.h5r.Reference):
                        target = h5_meta[ref]
                        if isinstance(target, h5py.Group):
                            # Group包含各个参数字段
                            for param_name in target.keys():
                                param_data = target[param_name][:]
                                if param_data.size == 1:
                                    val = param_data.item()
                                    # 处理bytes类型
                                    if isinstance(val, bytes):
                                        metadata_dict['jam_params'][param_name] = val.decode('utf-8')
                                    else:
                                        metadata_dict['jam_params'][param_name] = val
                                else:
                                    metadata_dict['jam_params'][param_name] = param_data.tolist()
                        elif isinstance(target, h5py.Dataset):
                            # Dataset可能是struct
                            if target.dtype.names:
                                for field in target.dtype.names:
                                    val = target[field][0, 0]
                                    metadata_dict['jam_params'][field] = val.item() if hasattr(val, 'item') else float(val)
                            else:
                                param_data = target[:]
                                if param_data.size == 1:
                                    metadata_dict['jam_params']['value'] = param_data.item()

                return metadata_dict

            # MATLAB struct数组格式（旧格式）
            meta_struct = self._metadata[0, index]

            # 提取字段
            metadata_dict = {}

            # sample_idx
            if 'sample_idx' in meta_struct.dtype.names:
                metadata_dict['sample_idx'] = int(meta_struct['sample_idx'][0, 0])

            # jam_types
            if 'jam_types' in meta_struct.dtype.names:
                jam_types_raw = meta_struct['jam_types'][0]
                metadata_dict['jam_types'] = [int(j) for j in jam_types_raw if j > 0]

            # JNR
            if 'JNR' in meta_struct.dtype.names:
                metadata_dict['JNR'] = float(meta_struct['JNR'][0, 0])

            # pos
            if 'pos' in meta_struct.dtype.names:
                metadata_dict['pos'] = int(meta_struct['pos'][0, 0])

            # jam_params
            metadata_dict['jam_params'] = {}
            if 'jam_params' in meta_struct.dtype.names:
                jam_params_struct = meta_struct['jam_params'][0, 0]
                for field in jam_params_struct.dtype.names:
                    value = jam_params_struct[field][0, 0]
                    # 转换numpy类型为Python类型
                    if hasattr(value, 'item'):
                        metadata_dict['jam_params'][field] = value.item()
                    else:
                        metadata_dict['jam_params'][field] = float(value)

            return metadata_dict

        except Exception as e:
            return None

    def _generate_abstract_description(self, metadata_dict, label_np):
        """根据metadata生成简短描述（每句2-3个特点）"""
        if metadata_dict is None:
            return self._build_class_description(label_np)

        jam_types = metadata_dict.get('jam_types', [])
        jam_params = metadata_dict.get('jam_params', {})

        # 收集描述片段
        parts = []
        for jam_type in jam_types:
            type_desc = ""
            visual_desc = ""
            param_desc = ""

            if jam_type == 1:  # DFTJ
                type_desc = "dense false target jamming"
                visual_desc = "diagonal lines in time-frequency domain"
                k = jam_params.get('dftj_k', 5)
                if k <= 4:
                    param_desc = "a few false targets"
                elif k <= 6:
                    param_desc = "several false targets"
                else:
                    param_desc = "many false targets"

            elif jam_type == 2:  # ISRJ
                type_desc = "intermittent sampling jamming"
                visual_desc = "discontinuous signal slices"
                M = jam_params.get('isrj_M', 3)
                if M <= 2:
                    param_desc = "simple sampling pattern"
                elif M <= 3:
                    param_desc = "moderate sampling pattern"
                else:
                    param_desc = "complex sampling pattern"

            elif jam_type == 3:  # RGPO
                type_desc = "range gate pull-off jamming"
                visual_desc = "false targets in range domain"
                pos_rel = jam_params.get('rgpo_position_relation', 'after')
                param_desc = f"trailing behind" if pos_rel == 'after' else "leading ahead"

            elif jam_type == 4:  # VGPO
                type_desc = "velocity gate pull-off jamming"
                visual_desc = "doppler shift pattern"
                dop_dir = jam_params.get('vgpo_doppler_direction', 'up')
                param_desc = "increasing frequency" if dop_dir == 'up' else "decreasing frequency"

            elif jam_type == 5:  # AJ
                type_desc = "aimed suppression jamming"
                visual_desc = "concentrated frequency energy"

            elif jam_type == 6:  # BJ
                type_desc = "barrage suppression jamming"
                visual_desc = "wide frequency coverage"

            elif jam_type == 10:  # SMSPJ
                type_desc = "smeared spectrum jamming"
                visual_desc = "steep diagonal lines"
                M = jam_params.get('smspj_M', 5)
                if M <= 4:
                    param_desc = "a few steep lines"
                elif M <= 6:
                    param_desc = "several steep lines"
                else:
                    param_desc = "many steep lines"

            elif jam_type == 11:  # C&IJ
                type_desc = "chopping and interleaved jamming"
                visual_desc = "continuous signal segments"
                is_cont = jam_params.get('cij_is_continuous', True)
                param_desc = "continuous pattern" if is_cont else "discontinuous pattern"

            elif jam_type == 15:  # CSJ
                type_desc = "comb spectrum jamming"
                visual_desc = "comb-like diagonal lines"
                M = jam_params.get('csj_M', 5)
                if M <= 3:
                    param_desc = "a few comb teeth"
                elif M <= 6:
                    param_desc = "several comb teeth"
                else:
                    param_desc = "many comb teeth"

            else:
                type_desc = self.JAM_TYPE_NAMES.get(jam_type, f'type{jam_type}')

            parts.append({'type': type_desc, 'visual': visual_desc, 'param': param_desc})

        # 组合描述（type + visual，每句2个特点）
        if len(parts) == 1:
            p = parts[0]
            if p['param']:
                return f"a radar signal with {p['param']} showing {p['visual']}"
            return f"a radar signal with {p['type']} showing {p['visual']}"
        else:
            types = ' and '.join([p['type'] for p in parts])
            return f"a radar signal with combined {types}"

    def _build_class_description(self, label_np):
        """回退：使用metadata_template生成类级别描述"""
        from multi.metadata_template import JAM_TYPE_NAMES, VISUAL_TEMPLATES

        active_indices = np.where(label_np == 1)[0]
        if len(active_indices) == 0:
            return "a radar signal with no jamming"

        active_classes = [self.class_names[i] for i in active_indices if i < len(self.class_names)]
        if len(active_classes) == 0:
            return "a radar signal with jamming"

        # 使用metadata_template生成描述
        descs = []
        for cls_name in active_classes:
            jam_type = None
            for k, v in JAM_TYPE_NAMES.items():
                if v == cls_name:
                    jam_type = k
                    break

            if jam_type:
                template = VISUAL_TEMPLATES.get(jam_type, {'base': cls_name, 'param': {}})
                descs.append(f"{cls_name} looks like {template['base']}")
            else:
                descs.append(f"a radar signal with {cls_name}")

        if len(descs) == 1:
            return descs[0]
        else:
            return ', '.join(descs)

    def __getitem__(self, index):
        h5_files = self._lazy_load()

        # 读取STFT数据
        stft_struct = h5_files['stft'][self.stft_var_name][:, :, index]
        one_hot_label_np = h5_files['label'][self.label_var_name][:, index]

        # 转换为复数
        stft_complex = stft_struct.view(np.complex128)

        # 构建三通道图像
        stft_real = np.real(stft_complex).T
        stft_imag = np.imag(stft_complex).T

        if self.channel_mode == 'real_imag_mag':
            stft_channel3 = np.abs(stft_complex).T
        elif self.channel_mode == 'real_imag_phase':
            stft_channel3 = np.angle(stft_complex).T
        else:
            stft_channel3 = np.abs(stft_complex).T

        stft_multi_channel = np.stack([stft_real, stft_imag, stft_channel3], axis=0)
        stft_tensor = torch.from_numpy(stft_multi_channel).float()

        # Resize到目标尺寸
        if stft_tensor.shape[-2:] != (self.image_size, self.image_size):
            stft_tensor = torch.nn.functional.interpolate(
                stft_tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # 归一化到[0, 1]范围
        for c in range(3):
            channel = stft_tensor[c]
            c_min, c_max = channel.min(), channel.max()
            if c_max > c_min:
                stft_tensor[c] = (channel - c_min) / (c_max - c_min)
            else:
                stft_tensor[c] = 0.5

        # 应用CLIP标准化
        if self.transform is not None:
            stft_tensor = self.transform(stft_tensor)

        label_tensor = torch.from_numpy(one_hot_label_np).float()

        # 解析metadata
        metadata_dict = self._parse_metadata(index)

        # 生成文本描述
        if self.use_abstract_description and metadata_dict:
            text_description = self._generate_abstract_description(metadata_dict, one_hot_label_np)
        else:
            text_description = self._build_class_description(one_hot_label_np)

        return stft_tensor, label_tensor, text_description, metadata_dict

    def close(self):
        """关闭h5文件句柄"""
        if self._h5_files is not None:
            self._h5_files['stft'].close()
            self._h5_files['label'].close()
            if 'meta' in self._h5_files:
                self._h5_files['meta'].close()
            self._h5_files = None

    def __del__(self):
        self.close()


class CZSLContrastiveDataset(Dataset):
    """
    CZSL对比学习数据集
    用于训练图像-文本对齐
    返回: (image, text_tokens, label)
    使用metadata_template生成文本描述
    """

    def __init__(
        self,
        base_dataset,
        class_names: list
    ):
        """
        初始化

        Args:
            base_dataset: 基础数据集 (返回 image, label)
            class_names: 类别名称列表
        """
        self.base_dataset = base_dataset
        self.class_names = class_names

    def _build_text_description(self, label):
        """根据标签生成文本描述，使用metadata_template"""
        from multi.metadata_template import JAM_TYPE_NAMES, VISUAL_TEMPLATES

        if isinstance(label, torch.Tensor):
            label_np = label.numpy()
        else:
            label_np = label

        active_indices = np.where(label_np == 1)[0]

        if len(active_indices) == 0:
            return "a radar signal with no jamming"

        active_classes = [self.class_names[i] for i in active_indices]

        # 生成描述
        descs = []
        for cls_name in active_classes:
            jam_type = None
            for k, v in JAM_TYPE_NAMES.items():
                if v == cls_name:
                    jam_type = k
                    break

            if jam_type:
                template = VISUAL_TEMPLATES.get(jam_type, {'base': cls_name, 'param': {}})
                descs.append(f"{cls_name} looks like {template['base']}")
            else:
                descs.append(f"a radar signal with {cls_name}")

        if len(descs) == 1:
            return descs[0]
        else:
            return ', '.join(descs)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        result = self.base_dataset[index]

        # 处理不同的返回格式
        if len(result) == 3:
            # CZSLSTFTDataset 返回 (image, label, text)
            image, label, text = result
        else:
            # 其他数据集返回 (image, label)
            image, label = result
            text = self._build_text_description(label)

        return image, text, label


def load_czsl_dataset(config, split='train'):
    """
    加载CZSL数据集

    Args:
        config: 配置字典
        split: 'train', 'val', 或 'test'

    Returns:
        dataset: CZSL数据集
        num_classes: 类别数
        jnr_levels: JNR级别列表
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from multi.data_split import CombinationSplitter, generate_default_splits

    data_config = config.get("data", {})

    # 获取类别名称
    class_names = [cls["name"] for cls in config.get("jamming_classes", [])]

    # JNR范围
    jnr_numbers = range(
        data_config.get("jnr_start", 0),
        data_config.get("jnr_end", 40) + 1,
        data_config.get("jnr_step", 10)
    )
    jnr_levels = [f"+{jnr}" if jnr >= 0 else str(jnr) for jnr in jnr_numbers]

    # 根据split选择文件名
    split_files = {
        'train': ('train_echo_stfts.mat', 'train_echo_label.mat'),
        'val': ('val_echo_stfts.mat', 'val_echo_label.mat'),
        'test': ('test_echo_stfts.mat', 'test_echo_label.mat')
    }
    stft_name, label_name = split_files[split]

    # 加载所有JNR级别的数据
    all_datasets = []

    for jnr in jnr_levels:
        jnr_folder_name = f"JNR_{jnr}"
        data_folder_path = os.path.join(data_config.get("base_path"), jnr_folder_name)

        stft_file = os.path.join(data_folder_path, stft_name)
        label_file = os.path.join(data_folder_path, label_name)

        if os.path.exists(stft_file) and os.path.exists(label_file):
            dataset = CZSLSTFTDataset(
                stft_file=stft_file,
                label_file=label_file,
                class_names=class_names,
                image_size=data_config.get("image_size", 224),
                normalize_to_clip=True,
                channel_mode='magnitude_only'
            )
            all_datasets.append(dataset)
            print(f"Loaded {split} data for JNR {jnr}: {len(dataset)} samples")

    if not all_datasets:
        raise ValueError(f"No {split} data found!")

    # 合并数据集
    combined_dataset = ConcatDataset(all_datasets)
    num_classes = all_datasets[0].num_classes

    print(f"Total {split} samples: {len(combined_dataset)}")
    print(f"Number of classes: {num_classes}")

    return combined_dataset, num_classes, jnr_levels


# ============================================================================
# 模块级别的collate函数（用于Windows多进程支持）
# ============================================================================

def czsl_collate_fn(batch):
    """
    CZSL数据集的collate函数
    必须定义在模块级别以支持Windows多进程

    Args:
        batch: 批次数据列表

    Returns:
        images, text_tokens, labels, texts
    """
    import clip

    images = torch.stack([item[0] for item in batch])
    texts = [item[1] for item in batch]
    labels = torch.stack([item[2] for item in batch])

    # tokenize texts
    text_tokens = clip.tokenize(texts, truncate=True)

    return images, text_tokens, labels, texts


def czsl_metadata_collate_fn(batch):
    """
    支持metadata的CZSL collate函数
    使用metadata_template生成文本描述并tokenize

    Args:
        batch: 批次数据列表，每个item为 (image, label, text, metadata_dict)

    Returns:
        images, text_tokens, labels, texts, metas
    """
    import clip
    from multi.metadata_template import generate_short_description

    images = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    original_texts = [item[2] for item in batch]
    metas = [item[3] for item in batch]

    # 使用metadata生成更丰富的文本描述
    texts = []
    for i, meta in enumerate(metas):
        if meta is not None:
            # 使用metadata_template生成描述
            text = generate_short_description(meta, style='visual_param')
        else:
            # 回退到原始文本
            text = original_texts[i]
        texts.append(text)

    # tokenize texts
    text_tokens = clip.tokenize(texts, truncate=True)

    return images, text_tokens, labels, texts, metas


def create_czsl_dataloaders(config):
    """
    创建CZSL数据加载器

    Args:
        config: 配置字典

    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes)
    """
    data_config = config.get("data", {})
    train_config = config.get("train", {})

    # 获取类别名称
    class_names = [cls["name"] for cls in config.get("jamming_classes", [])]

    # 加载数据集
    train_dataset, num_classes, jnr_levels = load_czsl_dataset(config, split='train')
    val_dataset, _, _ = load_czsl_dataset(config, split='val')
    test_dataset, _, _ = load_czsl_dataset(config, split='test')

    # 包装为对比学习数据集
    train_contrastive = CZSLContrastiveDataset(
        base_dataset=train_dataset,
        class_names=class_names
    )

    # 创建数据加载器（使用模块级别的czsl_collate_fn）
    train_loader = DataLoader(
        train_contrastive,
        batch_size=train_config.get("batch_size", 32),
        shuffle=True,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True),
        collate_fn=czsl_collate_fn
    )

    val_contrastive = CZSLContrastiveDataset(
        base_dataset=val_dataset,
        class_names=class_names
    )

    val_loader = DataLoader(
        val_contrastive,
        batch_size=train_config.get("batch_size", 32),
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True),
        collate_fn=czsl_collate_fn
    )

    test_contrastive = CZSLContrastiveDataset(
        base_dataset=test_dataset,
        class_names=class_names
    )

    test_loader = DataLoader(
        test_contrastive,
        batch_size=train_config.get("batch_size", 32),
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True),
        collate_fn=czsl_collate_fn
    )

    return train_loader, val_loader, test_loader, num_classes


def create_czsl_dataloaders_with_metadata(config, use_abstract_description=True):
    """
    创建支持metadata的CZSL数据加载器

    Args:
        config: 配置字典
        use_abstract_description: 是否使用抽象描述

    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes)
    """
    data_config = config.get("data", {})
    train_config = config.get("train", {})

    # 获取类别名称
    class_names = [cls["name"] for cls in config.get("jamming_classes", [])]

    # JNR范围
    jnr_numbers = range(
        data_config.get("jnr_start", 0),
        data_config.get("jnr_end", 40) + 1,
        data_config.get("jnr_step", 10)
    )
    jnr_levels = [f"+{jnr}" if jnr >= 0 else str(jnr) for jnr in jnr_numbers]

    # split文件映射
    split_files = {
        'train': ('train_echo_stfts.mat', 'train_echo_label.mat', 'train_echo_metadata'),
        'val': ('val_echo_stfts.mat', 'val_echo_label.mat', 'val_echo_metadata'),
        'test': ('test_echo_stfts.mat', 'test_echo_label.mat', 'test_echo_metadata')
    }

    # 加载数据集
    def load_split(split):
        stft_name, label_name, metadata_base = split_files[split]
        all_datasets = []

        for jnr in jnr_levels:
            jnr_folder_name = f"JNR_{jnr}"
            data_folder_path = os.path.join(data_config.get("base_path"), jnr_folder_name)

            stft_file = os.path.join(data_folder_path, stft_name)
            label_file = os.path.join(data_folder_path, label_name)

            # 优先使用JSON格式metadata，回退到MAT格式
            metadata_file_json = os.path.join(data_folder_path, metadata_base + '.json')
            metadata_file_mat = os.path.join(data_folder_path, metadata_base + '.mat')

            if os.path.exists(metadata_file_json):
                metadata_file = metadata_file_json
            elif os.path.exists(metadata_file_mat):
                metadata_file = metadata_file_mat
            else:
                metadata_file = None

            if not (os.path.exists(stft_file) and os.path.exists(label_file)):
                continue

            # 使用支持metadata的数据集
            dataset = CZSLSTFTDatasetWithMetadata(
                stft_file=stft_file,
                label_file=label_file,
                metadata_file=metadata_file if os.path.exists(metadata_file) else None,
                class_names=class_names,
                image_size=data_config.get("image_size", 224),
                normalize_to_clip=True,
                channel_mode='real_imag_mag',
                use_abstract_description=use_abstract_description
            )
            all_datasets.append(dataset)

        if not all_datasets:
            raise ValueError(f"No data found for split: {split}")

        return ConcatDataset(all_datasets)

    train_dataset = load_split('train')
    val_dataset = load_split('val')
    test_dataset = load_split('test')

    num_classes = 16  # 默认16类

    # 创建数据加载器（使用模块级别的czsl_metadata_collate_fn）
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get("batch_size", 32),
        shuffle=True,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True),
        collate_fn=czsl_metadata_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.get("batch_size", 32),
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True),
        collate_fn=czsl_metadata_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config.get("batch_size", 32),
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True),
        collate_fn=czsl_metadata_collate_fn
    )

    return train_loader, val_loader, test_loader, num_classes
