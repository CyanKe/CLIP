"""
SigLIP 模型加载工具 - 使用 HuggingFace transformers
"""
import torch
from typing import Tuple

SIGLIP_MODEL_MAP = {
    "siglip-base-patch16-224": "google/siglip-base-patch16-224",
    "siglip-large-patch16-384": "google/siglip-large-patch16-384",
    "siglip-so400m-patch14-384": "google/siglip-so400m-patch14-384",
}


def load_siglip(model_name: str, device: str = "cuda") -> Tuple[torch.nn.Module, object]:
    """
    加载 SigLIP 模型和预处理器

    Args:
        model_name: 模型名称（短名称或完整的 HuggingFace model ID）
        device: 目标设备

    Returns:
        model: SigLIP 模型
        processor: AutoProcessor 预处理器
    """
    from transformers import AutoModel, AutoProcessor

    # 解析模型 ID
    if model_name in SIGLIP_MODEL_MAP:
        model_id = SIGLIP_MODEL_MAP[model_name]
    elif model_name.startswith("google/"):
        model_id = model_name
    else:
        model_id = f"google/{model_name}"

    # 加载模型和预处理器
    model = AutoModel.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    # 移动到目标设备并转换为 float32
    model = model.to(device)
    model = model.float()

    return model, processor


def is_siglip_model(model_name: str) -> bool:
    """
    检测是否为 SigLIP 模型

    Args:
        model_name: 模型名称

    Returns:
        bool: 是否为 SigLIP 模型
    """
    model_name_lower = model_name.lower()
    return "siglip" in model_name_lower


def get_siglip_image_size(model_name: str) -> int:
    """
    获取 SigLIP 模型的图像输入尺寸

    Args:
        model_name: 模型名称

    Returns:
        int: 图像尺寸
    """
    if "384" in model_name:
        return 384
    return 224


def get_siglip_text_length(model_name: str) -> int:
    """
    获取 SigLIP 模型的文本最大长度

    Args:
        model_name: 模型名称

    Returns:
        int: 文本最大长度
    """
    # SigLIP 默认 64 tokens
    return 64
