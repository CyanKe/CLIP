"""
t-SNE 可视化脚本 - 展示单标签和多标签干扰的特征分布
欺骗干扰用黑色+不同形状，压制干扰用不同颜色，组合干扰用形状+颜色组合

Usage:
    python -m multi.visualize_tsne --checkpoint checkpoints/czsl_best_model.pt --split test
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi.model import create_czsl_model
from multi.data import STFTDataset, collate_fn, create_czsl_dataloaders


# 欺骗干扰类别 - 黑色 + 不同形状
DECEPTION_CLASSES = {
    "DFTJ": {"color": "black", "marker": "*"},
    "ISRJ": {"color": "black", "marker": "+"},
    "SMSPJ": {"color": "black", "marker": "o"},
    "C&IJ": {"color": "black", "marker": "^"},  # 三角形
    "CSJ": {"color": "black", "marker": "s"},   # 正方形
}

# 压制干扰类别 - 不同颜色 + 点
SUPPRESSION_CLASSES = [
    "AJ", "BJ", "SJ", "NCJ", "NPJ", "NFMJ", "NPMJ", "NAMJ", "PJ"
]

# 为压制干扰生成颜色
SUPPRESSION_COLORS = {
    "AJ": "#e41a1c",    # 红
    "BJ": "#377eb8",    # 蓝
    "SJ": "#4daf4a",    # 绿
    "NCJ": "#984ea3",   # 紫
    "NPJ": "#ff7f00",   # 橙
    "NFMJ": "#ffff33",  # 黄
    "NPMJ": "#a65628",  # 棕
    "NAMJ": "#f781bf",  # 粉
    "PJ": "#999999",    # 灰
}


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def extract_features(model, data_loader, device):
    """提取图像特征和标签"""
    model.eval()
    all_features = []
    all_labels = []
    all_metas = []

    with torch.no_grad():
        for images, _, _, labels, _, metas in tqdm(data_loader, desc="Extracting features"):
            images = images.to(device)

            # 调整图像尺寸
            if images.shape[-1] != 224:
                images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

            # 提取特征
            features = model.encode_image(images)
            features = F.normalize(features, dim=-1)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            all_metas.extend(metas)

    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0), all_metas


def get_label_info(label_indices, class_names):
    """获取标签的组合信息，返回 (欺骗类别, 压制类别)"""
    label_names = [class_names[i] for i in label_indices]

    deception = None
    suppression = None

    for name in label_names:
        if name in DECEPTION_CLASSES:
            deception = name
        elif name in SUPPRESSION_CLASSES:
            suppression = name

    return deception, suppression


def plot_tsne(features_2d, labels, class_names, save_path=None, figsize=(14, 10)):
    """
    绘制 t-SNE 图

    原则：
    - 单独欺骗干扰：黑色 + 特定形状
    - 单独压制干扰：特定颜色 + 点
    - 组合干扰：欺骗形状 + 压制颜色
    """
    num_samples = features_2d.shape[0]
    num_classes = len(class_names)

    # 解析每个样本的标签类型
    sample_info = []  # (deception, suppression, is_single_deception, is_single_suppression, is_combination)

    for i in range(num_samples):
        label_indices = np.where(labels[i] == 1)[0].tolist()
        deception, suppression = get_label_info(label_indices, class_names)

        is_single_deception = (deception is not None and suppression is None)
        is_single_suppression = (deception is None and suppression is not None)
        is_combination = (deception is not None and suppression is not None)

        sample_info.append({
            "deception": deception,
            "suppression": suppression,
            "is_single_deception": is_single_deception,
            "is_single_suppression": is_single_suppression,
            "is_combination": is_combination
        })

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制顺序：先压制干扰，再组合干扰，最后欺骗干扰（让欺骗干扰在最上层）
    # 1. 单独压制干扰
    for supp_name in SUPPRESSION_CLASSES:
        mask = np.array([info["is_single_suppression"] and info["suppression"] == supp_name
                        for info in sample_info])
        if mask.sum() > 0:
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                      c=SUPPRESSION_COLORS[supp_name], marker='o', s=50, alpha=0.7,
                      label=f"{supp_name} (压制)", edgecolors='white', linewidth=0.5)

    # 2. 组合干扰（欺骗形状 + 压制颜色）
    for dep_name, dep_style in DECEPTION_CLASSES.items():
        for supp_name in SUPPRESSION_CLASSES:
            mask = np.array([info["is_combination"] and
                           info["deception"] == dep_name and
                           info["suppression"] == supp_name
                           for info in sample_info])
            if mask.sum() > 0:
                ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                          c=SUPPRESSION_COLORS[supp_name], marker=dep_style["marker"],
                          s=80, alpha=0.8, edgecolors='black', linewidth=0.5,
                          label=f"{dep_name}+{supp_name}")

    # 3. 单独欺骗干扰（黑色 + 特定形状）
    for dep_name, dep_style in DECEPTION_CLASSES.items():
        mask = np.array([info["is_single_deception"] and info["deception"] == dep_name
                        for info in sample_info])
        if mask.sum() > 0:
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                      c='black', marker=dep_style["marker"], s=100, alpha=0.9,
                      label=f"{dep_name} (欺骗)", edgecolors='gray', linewidth=0.5)

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('t-SNE Visualization of Jamming Signal Features\n'
                 '(Shape = Deception Type, Color = Suppression Type)', fontsize=14)

    # 图例放在右侧
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, ncol=1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved t-SNE plot to {save_path}")

    plt.show()
    return fig


def plot_tsne_by_category(features_2d, labels, class_names, save_dir=None, figsize=(12, 10)):
    """
    分类别绘制 t-SNE 图，更清晰地展示每种类型
    """
    num_samples = features_2d.shape[0]

    # 解析标签
    sample_info = []
    for i in range(num_samples):
        label_indices = np.where(labels[i] == 1)[0].tolist()
        deception, suppression = get_label_info(label_indices, class_names)
        sample_info.append({
            "deception": deception,
            "suppression": suppression
        })

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # ========== 子图1：欺骗干扰分布（黑色形状）==========
    ax1 = axes[0, 0]
    for dep_name, dep_style in DECEPTION_CLASSES.items():
        # 包含该欺骗干扰的所有样本
        mask = np.array([info["deception"] == dep_name for info in sample_info])
        if mask.sum() > 0:
            ax1.scatter(features_2d[mask, 0], features_2d[mask, 1],
                       c='black', marker=dep_style["marker"], s=50, alpha=0.6,
                       label=dep_name)
    ax1.set_title('欺骗干扰分布 (所有样本)', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.set_xlabel('t-SNE Dim 1')
    ax1.set_ylabel('t-SNE Dim 2')

    # ========== 子图2：压制干扰分布（不同颜色）==========
    ax2 = axes[0, 1]
    for supp_name in SUPPRESSION_CLASSES:
        mask = np.array([info["suppression"] == supp_name for info in sample_info])
        if mask.sum() > 0:
            ax2.scatter(features_2d[mask, 0], features_2d[mask, 1],
                       c=SUPPRESSION_COLORS[supp_name], marker='o', s=50, alpha=0.6,
                       label=supp_name)
    ax2.set_title('压制干扰分布 (所有样本)', fontsize=12)
    ax2.legend(fontsize=9, ncol=2)
    ax2.set_xlabel('t-SNE Dim 1')
    ax2.set_ylabel('t-SNE Dim 2')

    # ========== 子图3：单标签 vs 多标签 ==========
    ax3 = axes[1, 0]
    single_mask = np.array([info["deception"] is None or info["suppression"] is None
                           for info in sample_info])
    multi_mask = np.array([info["deception"] is not None and info["suppression"] is not None
                          for info in sample_info])

    ax3.scatter(features_2d[single_mask, 0], features_2d[single_mask, 1],
               c='blue', marker='o', s=30, alpha=0.5, label='单标签')
    ax3.scatter(features_2d[multi_mask, 0], features_2d[multi_mask, 1],
               c='red', marker='^', s=50, alpha=0.7, label='多标签(组合)')
    ax3.set_title('单标签 vs 多标签分布', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.set_xlabel('t-SNE Dim 1')
    ax3.set_ylabel('t-SNE Dim 2')

    # ========== 子图4：组合干扰详细 ==========
    ax4 = axes[1, 1]
    for dep_name, dep_style in DECEPTION_CLASSES.items():
        for supp_name in SUPPRESSION_CLASSES:
            mask = np.array([info["deception"] == dep_name and info["suppression"] == supp_name
                           for info in sample_info])
            if mask.sum() > 0:
                ax4.scatter(features_2d[mask, 0], features_2d[mask, 1],
                           c=SUPPRESSION_COLORS[supp_name], marker=dep_style["marker"],
                           s=60, alpha=0.7, edgecolors='black', linewidth=0.3,
                           label=f"{dep_name}+{supp_name}")
    ax4.set_title('组合干扰分布 (形状=欺骗, 颜色=压制)', fontsize=12)
    ax4.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax4.set_xlabel('t-SNE Dim 1')
    ax4.set_ylabel('t-SNE Dim 2')

    plt.tight_layout()

    if save_dir:
        save_path = os.path.join(save_dir, 'tsne_multi_panel.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved multi-panel t-SNE plot to {save_path}")

    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description='t-SNE visualization for CZSL')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='multi/config.yaml', help='Path to config file')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--output_dir', type=str, default='results/tsne', help='Output directory')
    parser.add_argument('--perplexity', type=int, default=30, help='t-SNE perplexity')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to use (for testing)')
    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载配置
    config = load_config(args.config)
    class_names = [cls_info["name"] for cls_info in config.get("jamming_classes", [])]
    print(f"Class names: {class_names}")

    # 创建模型
    model = create_czsl_model(config, device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint}")

    # 创建数据加载器
    train_loader, val_loader, test_loader, _ = create_czsl_dataloaders(
        config,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True
    )

    # 选择 split
    if args.split == 'train':
        data_loader = train_loader
    elif args.split == 'val':
        data_loader = val_loader
    else:
        data_loader = test_loader

    dataset = data_loader.dataset

    if args.max_samples:
        # 随机采样
        indices = np.random.choice(len(dataset), min(args.max_samples, len(dataset)), replace=False)
        from torch.utils.data import Subset
        dataset = Subset(dataset, indices)
        data_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True, collate_fn=collate_fn
        )

    print(f"Dataset size: {len(dataset)}")

    # 提取特征
    features, labels, metas = extract_features(model, data_loader, device)
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    # t-SNE 降维
    print(f"Running t-SNE with perplexity={args.perplexity}...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=42, n_iter=1000)
    features_2d = tsne.fit_transform(features)
    print(f"t-SNE completed. Shape: {features_2d.shape}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 绘制单图
    plot_tsne(features_2d, labels, class_names,
              save_path=output_dir / 'tsne_main.png')

    # 绘制多面板图
    plot_tsne_by_category(features_2d, labels, class_names,
                          save_dir=output_dir)

    # 保存数据
    np.save(output_dir / 'features_2d.npy', features_2d)
    np.save(output_dir / 'labels.npy', labels)
    print(f"Saved features and labels to {output_dir}")


if __name__ == '__main__':
    main()
