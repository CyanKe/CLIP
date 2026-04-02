"""
单张图片干扰类型识别脚本

使用方法:
    python -m multi.predict_single --image path/to/image.png --checkpoint checkpoints/czsl_best_model.pt
    python -m multi.predict_single --image path/to/image.png --checkpoint checkpoints/czsl_best_model.pt --top_k 3
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi.model import create_czsl_model
from multi.metadata_template import JAM_TYPE_NAMES
import clip


def load_image(image_path: str, target_size: int = 224) -> torch.Tensor:
    """
    加载并预处理图像

    Args:
        image_path: 图像路径
        target_size: 目标尺寸

    Returns:
        预处理后的图像张量 [1, 3, H, W]
    """
    # 支持多种图像格式
    if image_path.endswith('.npy'):
        # numpy数组
        img_array = np.load(image_path)
        if img_array.ndim == 2:
            # 单通道转三通道
            img_array = np.stack([img_array] * 3, axis=0)
        elif img_array.ndim == 3:
            if img_array.shape[0] in [1, 3]:
                pass  # 已经是正确的格式
            else:
                img_array = np.transpose(img_array, (2, 0, 1))
        img_tensor = torch.from_numpy(img_array).float()
        if img_tensor.max() > 1:
            img_tensor = img_tensor / 255.0
    else:
        # 普通图像文件 (png, jpg, etc.)
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        if img_tensor.max() > 1:
            img_tensor = img_tensor / 255.0

    # 确保是4D张量 [batch, channel, height, width]
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)

    # 调整尺寸
    if img_tensor.shape[-2:] != (target_size, target_size):
        img_tensor = F.interpolate(img_tensor, size=(target_size, target_size), mode='bilinear', align_corners=False)

    return img_tensor


def predict_single_image(
    model,
    image_tensor: torch.Tensor,
    device: torch.device,
    top_k: int = 3,
    threshold: float = 0.5
):
    """
    对单张图像进行预测

    Args:
        model: CZSL模型
        image_tensor: 图像张量
        device: 计算设备
        top_k: 返回top-k个预测
        threshold: 多标签预测阈值

    Returns:
        dict: 预测结果
    """
    model.eval()

    with torch.no_grad():
        image_tensor = image_tensor.to(device)

        # 1. 零样本组合预测 (精确匹配)
        _, indices, pred_names = model.zero_shot_predict(
            image_tensor, use_combinations=True, top_k=top_k
        )

        # 2. 多标签预测
        probs, preds = model.zero_shot_multilabel_predict(
            image_tensor, threshold=threshold
        )

        # 3. 获取所有类别的概率
        image_features = model.encode_image(image_tensor)
        image_features = F.normalize(image_features, dim=-1)
        text_features = model.get_cached_text_features()

        logit_scale = model.model.logit_scale.exp()
        logits = logit_scale * (image_features @ text_features.T)
        all_probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    # 整理结果
    results = {
        'top_k_prediction': pred_names[0] if pred_names else [],
        'top_k_indices': indices[0].cpu().numpy().tolist() if indices is not None else [],
        'multilabel_probs': probs.squeeze().cpu().numpy(),
        'multilabel_preds': preds.squeeze().cpu().numpy(),
        'all_probs': all_probs,
        'class_names': model.class_names
    }

    return results


def print_results(results: dict, top_k: int = 3):
    """打印预测结果"""
    print("\n" + "=" * 60)
    print("干扰类型识别结果")
    print("=" * 60)

    # Top-K 预测
    print(f"\n【Top-{top_k} 组合预测】")
    for i, name in enumerate(results['top_k_prediction'][:top_k]):
        print(f"  {i+1}. {name}")

    # 多标签预测
    print(f"\n【多标签预测】(阈值=0.5)")
    preds = results['multilabel_preds']
    probs = results['multilabel_probs']
    class_names = results['class_names']

    predicted_classes = []
    for i, (pred, prob) in enumerate(zip(preds, probs)):
        if pred > 0:
            predicted_classes.append((class_names[i], prob))

    if predicted_classes:
        for name, prob in sorted(predicted_classes, key=lambda x: -x[1]):
            print(f"  - {name}: {prob:.2%}")
    else:
        print("  (无预测类别)")

    # 所有类别概率
    print(f"\n【所有类别概率】")
    prob_list = [(class_names[i], p) for i, p in enumerate(probs)]
    prob_list.sort(key=lambda x: -x[1])

    for name, prob in prob_list[:10]:  # 只显示前10
        bar = "█" * int(prob * 20)
        print(f"  {name:6s}: {prob:.2%} {bar}")

    print("\n" + "=" * 60)


def visualize_results(results: dict, save_path: str = None):
    """可视化预测结果"""
    class_names = results['class_names']
    probs = results['all_probs']

    # 排序
    sorted_indices = np.argsort(probs)[::-1]
    sorted_names = [class_names[i] for i in sorted_indices]
    sorted_probs = probs[sorted_indices]

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#ff6b6b' if p > 0.5 else '#4ecdc4' for p in sorted_probs]
    bars = ax.barh(range(len(sorted_names)), sorted_probs, color=colors)

    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('Probability')
    ax.set_title('Jamming Type Classification Results')
    ax.axvline(x=0.5, color='red', linestyle='--', label='Threshold')

    # 添加概率标签
    for bar, prob in zip(bars, sorted_probs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{prob:.1%}', va='center', fontsize=8)

    ax.set_xlim(0, 1.1)
    ax.invert_yaxis()
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"结果已保存到: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='单张图片干扰类型识别')
    parser.add_argument('--image', type=str, required=True, help='图像路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--top_k', type=int, default=3, help='返回top-k预测')
    parser.add_argument('--threshold', type=float, default=0.5, help='多标签预测阈值')
    parser.add_argument('--visualize', action='store_true', help='可视化结果')
    parser.add_argument('--output', type=str, default=None, help='可视化结果保存路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # 检查文件存在
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"图像文件不存在: {args.image}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"模型文件不存在: {args.checkpoint}")

    print(f"使用设备: {args.device}")
    print(f"加载模型: {args.checkpoint}")

    # 加载模型
    device = torch.device(args.device)

    # 从checkpoint加载配置
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {})

    # 确保配置结构完整
    if 'model' not in config:
        config['model'] = {}
    if 'jamming_classes' not in config:
        # 默认16类干扰
        config['jamming_classes'] = [
            {'name': JAM_TYPE_NAMES[i]} for i in range(1, 17)
        ]
    if 'clip_model' not in config['model']:
        config['model']['clip_model'] = 'ViT-B/32'

    # 创建模型
    model = create_czsl_model(config, device=args.device)

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # 缓存文本特征 (CLIPForCZSL内部使用self.class_names，同时生成组合特征)
    model.cache_text_features(max_combination_size=2)

    print(f"加载图像: {args.image}")

    # 加载图像
    image_tensor = load_image(args.image)

    # 预测
    results = predict_single_image(
        model, image_tensor, device,
        top_k=args.top_k,
        threshold=args.threshold
    )

    # 打印结果
    print_results(results, top_k=args.top_k)

    # 可视化
    if args.visualize:
        visualize_results(results, save_path=args.output)


if __name__ == '__main__':
    main()