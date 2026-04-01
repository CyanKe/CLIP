"""
数据加载模块 - 支持CLIP模型的STFT数据集
定义STFT数据集类，支持单通道、三通道和CLIP适配
"""
import os
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, Subset, ConcatDataset
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
        stft_multi_channel = np.stack([stft_real, stft_imag, stft_channel3], axis=0)

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
