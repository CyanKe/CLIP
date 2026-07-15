"""
测试数据加载并保存STFT图像和文本描述
python multi/_archive/dev/test_data_save.py --output_dir test_output --max_samples 50
"""
import os
import argparse
from pathlib import Path

import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
import cv2

# 添加路径
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from multi.data import create_czsl_dataloaders


def save_stft_with_context(
    stft_tensor: torch.Tensor,
    text: str,
    metadata: dict,
    class_names: list,
    save_path: Path,
    denormalize: bool = True
):
    """
    保存STFT图像和文本描述

    Args:
        stft_tensor: [3, H, W] STFT张量
        text: 文本描述
        metadata: 元数据
        class_names: 类别名称列表
        save_path: 保存路径
        denormalize: 是否反归一化
    """
    # 转换为numpy
    img = stft_tensor.cpu().numpy()

    # 反归一化 (CLIP标准化)
    if denormalize:
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        img = img * std[:, None, None] + mean[:, None, None]

    img = np.clip(img, 0, 1)
    img = np.transpose(img, (1, 2, 0))  # [H, W, 3]
    img_uint8 = (img * 255).astype(np.uint8)

    # 获取标签名称
    jam_types = metadata.get('jam_types', [])
    if isinstance(jam_types, list):
        label_str = '+'.join(jam_types) if jam_types else 'None'
    else:
        label_str = str(jam_types)

    # 创建带文本的图像
    fig_height = img_uint8.shape[0] + 150  # 额外空间放文本
    fig_width = max(img_uint8.shape[1], 800)

    # 创建白色背景
    fig = np.ones((fig_height, fig_width, 3), dtype=np.uint8) * 255

    # 放置STFT图像
    fig[:img_uint8.shape[0], :img_uint8.shape[1]] = img_uint8

    # 添加文本信息
    y_offset = img_uint8.shape[0] + 20

    # 使用OpenCV绘制文本
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 0, 0)
    thickness = 1

    # 标签
    cv2.putText(fig, f"Label: {label_str}", (10, y_offset), font, font_scale, color, thickness)

    # 文本描述 (换行显示)
    y_offset += 25
    cv2.putText(fig, "Context:", (10, y_offset), font, font_scale, color, thickness)

    # 长文本换行
    y_offset += 20
    words = text.split()
    line = ""
    max_chars = 80
    for word in words:
        if len(line + word) > max_chars:
            cv2.putText(fig, line, (10, y_offset), font, font_scale, color, thickness)
            y_offset += 18
            line = word + " "
        else:
            line += word + " "
    if line:
        cv2.putText(fig, line, (10, y_offset), font, font_scale, color, thickness)

    # 保存
    cv2.imwrite(str(save_path), cv2.cvtColor(fig, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description="Test data loading and save STFT images")
    parser.add_argument("--config", type=str, default="multi/config.yaml",
                        help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="test_output",
                        help="Output directory")
    parser.add_argument("--max_samples", type=int, default=50,
                        help="Maximum number of samples to save")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test"],
                        help="Data split to use")
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 类别名称
    class_names = [cls["name"] for cls in config.get("jamming_classes", [])]
    print(f"Classes: {class_names}")

    # 加载归一化统计量
    stats_file = 'multi/normalization_stats.json'
    normalization_stats = None
    if os.path.exists(stats_file):
        import json
        with open(stats_file, 'r') as f:
            normalization_stats = json.load(f)
        print(f"Loaded normalization stats from {stats_file}")

    # 创建数据加载器
    print("\nLoading data...")
    train_loader, val_loader, test_loader, num_classes = create_czsl_dataloaders(
        config=config,
        normalization_stats=normalization_stats,
        batch_size=1,
        num_workers=0,
    )

    # 选择数据集
    if args.split == "train":
        data_loader = train_loader
    elif args.split == "val":
        data_loader = val_loader
    else:
        data_loader = test_loader

    print(f"Dataset size: {len(data_loader.dataset)}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 遍历并保存
    print(f"\nSaving up to {args.max_samples} samples...")
    saved_count = 0

    for idx, (images, time_signals, text_tokens, labels, texts, metadata_list) in enumerate(data_loader):
        if saved_count >= args.max_samples:
            break

        # 获取单个样本
        image = images[0]
        text = texts[0]
        metadata = metadata_list[0]
        label = labels[0]

        # 构建文件名
        jam_types = metadata.get('jam_types', [])
        if isinstance(jam_types, list):
            label_str = '+'.join(jam_types) if jam_types else 'None'
        else:
            label_str = str(jam_types)

        filename = f"{saved_count:04d}_{label_str}.png"
        save_path = output_dir / filename

        # 保存
        save_stft_with_context(
            stft_tensor=image,
            text=text,
            metadata=metadata,
            class_names=class_names,
            save_path=save_path
        )

        saved_count += 1
        if saved_count % 10 == 0:
            print(f"  Saved {saved_count}/{args.max_samples}")

    print(f"\nDone! Saved {saved_count} images to {output_dir}")

    # 打印样本示例
    print("\n" + "="*60)
    print("Sample Information:")
    print("="*60)
    print(f"Image shape: {image.shape}")
    print(f"Label shape: {label.shape}")
    print(f"Active labels: {[class_names[i] for i, v in enumerate(label) if v == 1]}")
    print(f"\nContext text:\n{text}")


if __name__ == "__main__":
    main()
