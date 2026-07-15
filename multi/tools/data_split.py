"""
数据划分模块 - 支持Seen/Unseen组合的划分
用于组合零样本学习的数据管理
"""
import os
import sys
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, Subset, DataLoader
from typing import List, Dict, Tuple, Optional
from itertools import combinations


class CombinationSplitter:
    """
    组合数据划分器
    将数据按seen/unseen组合划分
    """

    def __init__(
        self,
        class_names: List[str],
        seen_combinations: List[List[int]],
        unseen_combinations: List[List[int]] = None
    ):
        """
        初始化组合划分器

        Args:
            class_names: 类别名称列表
            seen_combinations: 已见组合的类别索引列表 (用于训练)
            unseen_combinations: 未见组合的类别索引列表 (用于零样本测试)
        """
        self.class_names = class_names
        self.num_classes = len(class_names)

        # 转换为tuple便于比较
        self.seen_set = set(tuple(sorted(c)) for c in seen_combinations)
        self.unseen_set = set(tuple(sorted(c)) for c in (unseen_combinations or []))

        # 所有组合
        self.all_combinations = seen_combinations + (unseen_combinations or [])

        print(f"CombinationSplitter initialized:")
        print(f"  Total classes: {self.num_classes}")
        print(f"  Seen combinations: {len(self.seen_set)}")
        print(f"  Unseen combinations: {len(self.unseen_set)}")

    def is_seen(self, label_vector: np.ndarray) -> bool:
        """
        检查样本是否属于seen组合

        Args:
            label_vector: [num_classes] 标签向量

        Returns:
            是否属于seen组合
        """
        active_indices = tuple(sorted(np.where(label_vector == 1)[0]))
        return active_indices in self.seen_set

    def is_unseen(self, label_vector: np.ndarray) -> bool:
        """
        检查样本是否属于unseen组合

        Args:
            label_vector: [num_classes] 标签向量

        Returns:
            是否属于unseen组合
        """
        active_indices = tuple(sorted(np.where(label_vector == 1)[0]))
        return active_indices in self.unseen_set

    def get_combination_type(self, label_vector: np.ndarray) -> str:
        """
        获取样本的组合类型

        Args:
            label_vector: [num_classes] 标签向量

        Returns:
            'seen', 'unseen', 或 'unknown'
        """
        active_indices = tuple(sorted(np.where(label_vector == 1)[0]))

        if active_indices in self.seen_set:
            return 'seen'
        elif active_indices in self.unseen_set:
            return 'unseen'
        else:
            return 'unknown'

    def filter_dataset_indices(
        self,
        dataset: Dataset,
        combination_type: str = 'seen'
    ) -> List[int]:
        """
        过滤数据集，返回指定类型的样本索引

        Args:
            dataset: 数据集
            combination_type: 'seen', 'unseen', 或 'all'

        Returns:
            样本索引列表
        """
        indices = []

        for idx in range(len(dataset)):
            # 获取标签
            if hasattr(dataset, 'label_data'):
                # 直接访问h5数据
                label = dataset.label_data[:, idx]
            else:
                # 通过__getitem__获取
                _, label = dataset[idx]
                if isinstance(label, torch.Tensor):
                    label = label.numpy()

            if combination_type == 'seen' and self.is_seen(label):
                indices.append(idx)
            elif combination_type == 'unseen' and self.is_unseen(label):
                indices.append(idx)
            elif combination_type == 'all':
                indices.append(idx)

        return indices

    def split_dataset(
        self,
        dataset: Dataset
    ) -> Tuple[Subset, Subset]:
        """
        将数据集划分为seen和unseen两部分

        Args:
            dataset: 原始数据集

        Returns:
            (seen_dataset, unseen_dataset)
        """
        seen_indices = self.filter_dataset_indices(dataset, 'seen')
        unseen_indices = self.filter_dataset_indices(dataset, 'unseen')

        seen_dataset = Subset(dataset, seen_indices) if seen_indices else None
        unseen_dataset = Subset(dataset, unseen_indices) if unseen_indices else None

        print(f"Dataset split:")
        print(f"  Seen samples: {len(seen_indices)}")
        print(f"  Unseen samples: {len(unseen_indices)}")

        return seen_dataset, unseen_dataset

    def get_combination_name(self, label_vector: np.ndarray) -> str:
        """
        获取组合的名称

        Args:
            label_vector: [num_classes] 标签向量

        Returns:
            组合名称字符串
        """
        active_indices = np.where(label_vector == 1)[0]
        active_names = [self.class_names[i] for i in active_indices]
        return " + ".join(active_names)


def generate_default_splits(
    class_names: List[str],
    unseen_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    生成默认的seen/unseen划分

    策略：
    1. 所有单干扰都是seen
    2. 随机选择部分两两组合作为unseen

    Args:
        class_names: 类别名称列表
        unseen_ratio: unseen组合比例
        seed: 随机种子

    Returns:
        (seen_combinations, unseen_combinations)
    """
    np.random.seed(seed)
    num_classes = len(class_names)

    # 所有单干扰都是seen
    seen_combinations = [[i] for i in range(num_classes)]

    # 生成所有两两组合
    all_pairs = list(combinations(range(num_classes), 2))
    all_pairs = [list(p) for p in all_pairs]

    # 随机选择一部分作为unseen
    np.random.shuffle(all_pairs)
    num_unseen = int(len(all_pairs) * unseen_ratio)

    unseen_combinations = all_pairs[:num_unseen]
    seen_combinations.extend(all_pairs[num_unseen:])

    print(f"Generated default splits:")
    print(f"  Total single classes: {num_classes} (all seen)")
    print(f"  Total pairs: {len(all_pairs)}")
    print(f"  Seen pairs: {len(all_pairs) - num_unseen}")
    print(f"  Unseen pairs: {num_unseen}")

    return seen_combinations, unseen_combinations


def parse_combination_config(config_str: str, class_names: List[str]) -> List[List[int]]:
    """
    解析配置字符串为组合索引列表

    Args:
        config_str: 组合字符串，如 "DFTJ+ISRJ, VDJ+DDJ"
        class_names: 类别名称列表

    Returns:
        组合索引列表
    """
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    combinations_list = []

    # 分割不同的组合
    comb_strs = [s.strip() for s in config_str.split(',')]

    for comb_str in comb_strs:
        # 分割组合中的类别
        parts = [p.strip() for p in comb_str.split('+')]
        indices = [name_to_idx[p] for p in parts if p in name_to_idx]
        if len(indices) > 0:
            combinations_list.append(indices)

    return combinations_list


def print_split_statistics(
    dataset: Dataset,
    splitter: CombinationSplitter,
    split_name: str = "Dataset"
):
    """
    打印数据划分统计信息

    Args:
        dataset: 数据集
        splitter: 组合划分器
        split_name: 数据集名称
    """
    seen_count = 0
    unseen_count = 0
    unknown_count = 0

    combination_counts = {}

    for idx in range(len(dataset)):
        if hasattr(dataset, 'label_data'):
            label = dataset.label_data[:, idx]
        else:
            _, label = dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.numpy()

        comb_type = splitter.get_combination_type(label)

        if comb_type == 'seen':
            seen_count += 1
        elif comb_type == 'unseen':
            unseen_count += 1
        else:
            unknown_count += 1

        # 统计各组合数量
        comb_name = splitter.get_combination_name(label)
        combination_counts[comb_name] = combination_counts.get(comb_name, 0) + 1

    print(f"\n{'='*60}")
    print(f"{split_name} Statistics")
    print(f"{'='*60}")
    print(f"Total samples: {len(dataset)}")
    print(f"Seen samples: {seen_count}")
    print(f"Unseen samples: {unseen_count}")
    print(f"Unknown samples: {unknown_count}")

    print(f"\nCombination distribution:")
    for comb_name, count in sorted(combination_counts.items(), key=lambda x: -x[1]):
        print(f"  {comb_name}: {count}")


class CZSLDatasetWrapper(Dataset):
    """
    CZSL数据集包装器
    返回 (image, label, text_description, combination_index)
    """

    def __init__(
        self,
        base_dataset: Dataset,
        splitter: CombinationSplitter,
        text_encoder,
        include_text: bool = True
    ):
        """
        初始化包装器

        Args:
            base_dataset: 基础数据集
            splitter: 组合划分器
            text_encoder: 文本编码器
            include_text: 是否包含文本描述
        """
        self.base_dataset = base_dataset
        self.splitter = splitter
        self.text_encoder = text_encoder
        self.include_text = include_text

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, label = self.base_dataset[index]

        if isinstance(label, torch.Tensor):
            label_np = label.numpy()
        else:
            label_np = label

        result = [image, label]

        if self.include_text:
            # 生成文本描述
            text = self.text_encoder.build_text_description(label)
            result.append(text)

            # 获取组合索引
            comb_idx = self.splitter.label_to_combination_index(label)
            result.append(comb_idx)

        return tuple(result)


# 预定义的干扰类型组合建议
RECOMMENDED_UNSEEN_COMBINATIONS = [
    # 基于16种干扰类型，建议一些作为零样本测试的组合
    # 这些组合在训练时不出现，但在测试时需要识别
    ["DFTJ", "NFMJ"],    # Dense False Target + Noise FM
    ["ISRJ", "NPMJ"],    # Interrupted Sampling + Noise PM
    ["VDJ", "NPJ"],      # Velocity Deception + Noise Product
    ["DDJ", "NCJ"],      # Distance-Velocity Deception + Noise Convolution
    ["AJ", "CSJ"],       # Aiming + Comb Spectrum
    ["BJ", "PJ"],        # Barrage + Pulse
    ["SJ", "NAMJ"],      # Swept Frequency + Noise AM
    ["SMSPJ", "C&IJ"],   # Smeared Spectrum + Chopping Interleaved
]


def create_combination_splitter_from_config(config: dict) -> CombinationSplitter:
    """
    从配置创建组合划分器

    Args:
        config: 配置字典

    Returns:
        CombinationSplitter实例
    """
    class_names = [cls["name"] for cls in config.get("jamming_classes", [])]

    # 从配置中读取seen/unseen组合
    czsl_config = config.get("czsl", {})
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}

    def parse_combinations(comb_list):
        """解析组合列表"""
        result = []
        for comb in comb_list:
            if isinstance(comb, str):
                if comb in name_to_idx:
                    result.append([name_to_idx[comb]])
            elif isinstance(comb, list):
                indices = [name_to_idx[name] for name in comb if name in name_to_idx]
                if len(indices) > 0:
                    result.append(indices)
        return result

    seen_combinations = parse_combinations(czsl_config.get("seen_combinations", []))
    unseen_combinations = parse_combinations(czsl_config.get("unseen_combinations", []))

    # 如果配置中没有指定，使用默认划分
    if not seen_combinations:
        print("No seen combinations specified, using default split...")
        seen_combinations, unseen_combinations = generate_default_splits(
            class_names,
            unseen_ratio=czsl_config.get("unseen_ratio", 0.2),
            seed=czsl_config.get("split_seed", 42)
        )

    return CombinationSplitter(
        class_names=class_names,
        seen_combinations=seen_combinations,
        unseen_combinations=unseen_combinations
    )


if __name__ == "__main__":
    # 测试代码
    import yaml

    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    class_names = [cls["name"] for cls in config.get("jamming_classes", [])]

    print("="*60)
    print("Testing Combination Splitter")
    print("="*60)

    # 测试默认划分
    print("\n1. 测试默认划分生成:")
    seen, unseen = generate_default_splits(class_names, unseen_ratio=0.2, seed=45)

    print(f"\n见过组合 ({len(seen)}):")
    print("  单一类别:", len([c for c in seen if len(c) == 1]))
    print("  成对组合:", len([c for c in seen if len(c) == 2]))

    print(f"\n没见过组合 ({len(unseen)}):")
    for comb in unseen[:5]:
        names = [class_names[i] for i in comb]
        print(f"  {' + '.join(names)}")

    # 测试划分器
    print("\n2. 测试组合划分器:")
    splitter = CombinationSplitter(
        class_names=class_names,
        seen_combinations=seen,
        unseen_combinations=unseen
    )

    # 测试标签向量
    print("\n3. 测试标签向量分类器(区分是否见过):")
    test_label = np.zeros(len(class_names))
    test_label[0] = 1  # DFTJ
    test_label[1] = 1  # ISRJ

    comb_type = splitter.get_combination_type(test_label)
    comb_name = splitter.get_combination_name(test_label)
    print(f"  Label: {comb_name}")
    print(f"  Type: {comb_type}")

    # 打印推荐的unseen组合
    print("\n4. 推荐的没见过的组合:")
    for comb in RECOMMENDED_UNSEEN_COMBINATIONS:
        print(f"  {' + '.join(comb)}")