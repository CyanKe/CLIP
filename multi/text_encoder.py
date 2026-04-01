"""
文本编码器模块 - 用于CZSL的文本描述生成与编码
支持单干扰和组合干扰的文本描述生成
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from itertools import combinations
import sys
import os

# 添加CLIP路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clip


class TextEncoder:
    """
    文本编码器
    负责生成文本描述并编码为特征向量
    """

    def __init__(
        self,
        class_names: List[str],
        class_descriptions: Dict[str, List[str]],
        clip_model,
        device: str = "cuda"
    ):
        """
        初始化文本编码器

        Args:
            class_names: 类别名称列表
            class_descriptions: 类别描述字典 {class_name: [description1, ...]}
            clip_model: CLIP模型实例
            device: 计算设备
        """
        self.class_names = class_names
        self.class_descriptions = class_descriptions
        self.clip_model = clip_model
        self.device = device
        self.num_classes = len(class_names)

        # 文本特征缓存
        self._single_class_features = None  # 单类别特征
        self._all_combination_features = None  # 所有组合特征
        self._all_combination_names = None  # 所有组合名称

    def build_text_description(
        self,
        label_vector: torch.Tensor,
        template: str = "a photo of {}"
    ) -> str:
        """
        根据标签向量生成文本描述

        Args:
            label_vector: [num_classes] 标签向量 (0/1)
            template: 文本模板

        Returns:
            文本描述字符串
        """
        # 获取活跃类别
        if isinstance(label_vector, torch.Tensor):
            active_indices = torch.where(label_vector == 1)[0].tolist()
        else:
            active_indices = [i for i, v in enumerate(label_vector) if v == 1]

        if len(active_indices) == 0:
            return "a radar signal with no jamming"

        active_classes = [self.class_names[i] for i in active_indices]

        # 生成描述
        if len(active_classes) == 1:
            # 单干扰
            desc = self.class_descriptions.get(active_classes[0], [active_classes[0]])[0]
            return desc
        else:
            # 组合干扰
            desc_parts = []
            for cls in active_classes:
                cls_desc = self.class_descriptions.get(cls, [cls])[0]
                # 提取关键部分（去掉"a radar signal with"前缀）
                if cls_desc.startswith("a radar signal with "):
                    cls_desc = cls_desc[len("a radar signal with "):]
                desc_parts.append(cls_desc)

            return f"a radar signal with {' and '.join(desc_parts)}"

    def build_all_combination_descriptions(
        self,
        max_combination_size: int = 2,
        include_single: bool = True
    ) -> Tuple[List[str], List[List[int]]]:
        """
        构建所有可能的组合描述

        Args:
            max_combination_size: 最大组合大小 (1=单干扰, 2=两两组合)
            include_single: 是否包含单干扰

        Returns:
            descriptions: 文本描述列表
            combinations: 对应的类别索引组合列表
        """
        descriptions = []
        comb_indices = []

        # 单干扰
        if include_single:
            for i, cls_name in enumerate(self.class_names):
                desc = self.class_descriptions.get(cls_name, [cls_name])[0]
                descriptions.append(desc)
                comb_indices.append([i])

        # 两两组合
        if max_combination_size >= 2:
            for i, j in combinations(range(self.num_classes), 2):
                cls1, cls2 = self.class_names[i], self.class_names[j]

                # 获取描述并提取关键部分
                desc1 = self.class_descriptions.get(cls1, [cls1])[0]
                desc2 = self.class_descriptions.get(cls2, [cls2])[0]

                if desc1.startswith("a radar signal with "):
                    desc1 = desc1[len("a radar signal with "):]
                if desc2.startswith("a radar signal with "):
                    desc2 = desc2[len("a radar signal with "):]

                combined_desc = f"a radar signal with {desc1} and {desc2}"
                descriptions.append(combined_desc)
                comb_indices.append([i, j])

        return descriptions, comb_indices

    @torch.no_grad()
    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 64
    ) -> torch.Tensor:
        """
        批量编码文本描述

        Args:
            texts: 文本描述列表
            batch_size: 批处理大小

        Returns:
            文本特征 [num_texts, embed_dim]
        """
        self.clip_model.eval()
        all_features = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            tokens = clip.tokenize(batch_texts, truncate=True).to(self.device)
            features = self.clip_model.encode_text(tokens)
            features = F.normalize(features, dim=-1)
            all_features.append(features)

        return torch.cat(all_features, dim=0)

    def cache_single_class_features(self):
        """
        缓存所有单类别的文本特征
        """
        descriptions = []
        for cls_name in self.class_names:
            desc = self.class_descriptions.get(cls_name, [cls_name])[0]
            descriptions.append(desc)

        self._single_class_features = self.encode_texts(descriptions)
        print(f"Cached text features for {len(self.class_names)} single classes")

    def cache_all_combination_features(
        self,
        max_combination_size: int = 2,
        include_single: bool = True
    ):
        """
        缓存所有组合的文本特征

        Args:
            max_combination_size: 最大组合大小
            include_single: 是否包含单干扰
        """
        descriptions, comb_indices = self.build_all_combination_descriptions(
            max_combination_size=max_combination_size,
            include_single=include_single
        )

        self._all_combination_features = self.encode_texts(descriptions)
        self._all_combination_names = descriptions
        self._all_combination_indices = comb_indices

        print(f"Cached text features for {len(descriptions)} combinations")

    def get_single_class_features(self) -> torch.Tensor:
        """获取单类别特征"""
        if self._single_class_features is None:
            self.cache_single_class_features()
        return self._single_class_features

    def get_all_combination_features(self) -> torch.Tensor:
        """获取所有组合特征"""
        if self._all_combination_features is None:
            self.cache_all_combination_features()
        return self._all_combination_features

    def get_combination_indices(self) -> List[List[int]]:
        """获取所有组合的类别索引"""
        if self._all_combination_indices is None:
            self.cache_all_combination_features()
        return self._all_combination_indices

    def get_combination_names(self) -> List[str]:
        """获取所有组合的名称"""
        if self._all_combination_names is None:
            self.cache_all_combination_features()
        return self._all_combination_names


class CombinationRegistry:
    """
    组合注册表
    管理Seen和Unseen组合的划分
    """

    def __init__(
        self,
        class_names: List[str],
        seen_combinations: Optional[List[List[int]]] = None,
        unseen_combinations: Optional[List[List[int]]] = None
    ):
        """
        初始化组合注册表

        Args:
            class_names: 类别名称列表
            seen_combinations: 已见组合的类别索引列表
            unseen_combinations: 未见组合的类别索引列表
        """
        self.class_names = class_names
        self.num_classes = len(class_names)

        # 默认：所有单干扰为seen，部分组合为unseen
        if seen_combinations is None:
            # 所有单干扰都是seen
            self.seen_combinations = [[i] for i in range(self.num_classes)]
        else:
            self.seen_combinations = seen_combinations

        if unseen_combinations is None:
            self.unseen_combinations = []
        else:
            self.unseen_combinations = unseen_combinations

        # 构建索引映射
        self._build_mappings()

    def _build_mappings(self):
        """构建组合到索引的映射"""
        self.comb_to_idx = {}
        self.idx_to_comb = {}

        all_combinations = self.seen_combinations + self.unseen_combinations
        for idx, comb in enumerate(all_combinations):
            key = tuple(sorted(comb))
            self.comb_to_idx[key] = idx
            self.idx_to_comb[idx] = comb

    def is_seen(self, combination: List[int]) -> bool:
        """检查组合是否在seen集合中"""
        key = tuple(sorted(combination))
        return key in [tuple(sorted(c)) for c in self.seen_combinations]

    def get_all_seen_indices(self) -> List[int]:
        """获取所有seen组合的索引"""
        return list(range(len(self.seen_combinations)))

    def get_all_unseen_indices(self) -> List[int]:
        """获取所有unseen组合的索引"""
        return list(range(len(self.seen_combinations),
                         len(self.seen_combinations) + len(self.unseen_combinations)))

    def label_to_combination_index(self, label_vector: torch.Tensor) -> int:
        """
        将标签向量转换为组合索引

        Args:
            label_vector: [num_classes] 标签向量

        Returns:
            组合索引，如果未找到返回-1
        """
        active_indices = tuple(sorted(torch.where(label_vector == 1)[0].tolist()))
        return self.comb_to_idx.get(active_indices, -1)

    @classmethod
    def from_config(cls, config: dict) -> 'CombinationRegistry':
        """
        从配置创建组合注册表

        Args:
            config: 配置字典

        Returns:
            CombinationRegistry实例
        """
        class_names = [cls["name"] for cls in config.get("jamming_classes", [])]
        name_to_idx = {name: idx for idx, name in enumerate(class_names)}

        # 从配置中读取seen和unseen组合
        czsl_config = config.get("czsl", {})

        def parse_combinations(comb_list):
            """解析组合列表"""
            result = []
            for comb in comb_list:
                if isinstance(comb, str):
                    # 单干扰名称
                    if comb in name_to_idx:
                        result.append([name_to_idx[comb]])
                elif isinstance(comb, list):
                    # 组合干扰名称列表
                    indices = [name_to_idx[name] for name in comb if name in name_to_idx]
                    if len(indices) > 0:
                        result.append(indices)
            return result

        seen_combinations = parse_combinations(czsl_config.get("seen_combinations", []))
        unseen_combinations = parse_combinations(czsl_config.get("unseen_combinations", []))

        return cls(
            class_names=class_names,
            seen_combinations=seen_combinations if seen_combinations else None,
            unseen_combinations=unseen_combinations
        )


def create_text_encoder(
    config: dict,
    clip_model,
    device: str = "cuda"
) -> TextEncoder:
    """
    根据配置创建文本编码器

    Args:
        config: 配置字典
        clip_model: CLIP模型实例
        device: 计算设备

    Returns:
        TextEncoder实例
    """
    class_names = []
    class_descriptions = {}

    for cls_info in config.get("jamming_classes", []):
        name = cls_info["name"]
        class_names.append(name)
        class_descriptions[name] = cls_info["descriptions"]

    return TextEncoder(
        class_names=class_names,
        class_descriptions=class_descriptions,
        clip_model=clip_model,
        device=device
    )


def print_combination_info(registry: CombinationRegistry, class_names: List[str]):
    """
    打印组合划分信息

    Args:
        registry: 组合注册表
        class_names: 类别名称列表
    """
    print("\n" + "=" * 60)
    print("Combination Registry Info")
    print("=" * 60)

    print(f"\nTotal classes: {len(class_names)}")
    print(f"Seen combinations: {len(registry.seen_combinations)}")
    print(f"Unseen combinations: {len(registry.unseen_combinations)}")

    print("\nSeen combinations:")
    for comb in registry.seen_combinations:
        names = [class_names[i] for i in comb]
        print(f"  {' + '.join(names)}")

    if registry.unseen_combinations:
        print("\nUnseen combinations (Zero-Shot):")
        for comb in registry.unseen_combinations:
            names = [class_names[i] for i in comb]
            print(f"  {' + '.join(names)}")


if __name__ == "__main__":
    # 测试代码
    import yaml

    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 提取类别信息
    class_names = [cls["name"] for cls in config.get("jamming_classes", [])]
    class_descriptions = {}
    for cls_info in config.get("jamming_classes", []):
        class_descriptions[cls_info["name"]] = cls_info["descriptions"]

    print(f"Found {len(class_names)} jamming classes:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")

    # 测试文本描述生成
    print("\n" + "-" * 40)
    print("Testing text description generation...")

    # 创建一个模拟的标签向量
    label_vector = torch.zeros(len(class_names))
    label_vector[0] = 1  # DFTJ
    label_vector[1] = 1  # ISRJ

    # 手动构建描述
    active = [class_names[i] for i in torch.where(label_vector == 1)[0].tolist()]
    desc1 = class_descriptions.get(active[0], [active[0]])[0]
    desc2 = class_descriptions.get(active[1], [active[1]])[0]
    if desc1.startswith("a radar signal with "):
        desc1 = desc1[len("a radar signal with "):]
    if desc2.startswith("a radar signal with "):
        desc2 = desc2[len("a radar signal with "):]

    print(f"Label: {active}")
    print(f"Description: a radar signal with {desc1} and {desc2}")

    # 测试所有组合
    print("\n" + "-" * 40)
    print("Building all combinations...")

    total = len(class_names)  # 单干扰
    total += len(class_names) * (len(class_names) - 1) // 2  # 两两组合
    print(f"Total combinations (single + pairs): {total}")

    # 显示前几个组合
    print("\nFirst 5 single-class descriptions:")
    for i in range(min(5, len(class_names))):
        print(f"  {i}: {class_descriptions[class_names[i]][0]}")

    print("\nFirst 5 pair combinations:")
    count = 0
    for i, j in combinations(range(len(class_names)), 2):
        if count >= 5:
            break
        print(f"  {class_names[i]} + {class_names[j]}")
        count += 1