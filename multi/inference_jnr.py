"""
对指定JNR和干扰类型进行推理
python -m multi.inference_jnr --checkpoint checkpoints/czsl_best_model.pt --jnr 20 --jam_type ISRJ
"""
import os
import sys
import json
import yaml
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi.model import create_czsl_model
from multi.data import STFTDataset, collate_fn
from torch.utils.data import DataLoader
import clip


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def inference_specific_jnr_jamtype(
    config: dict,
    checkpoint_path: str,
    jnr: int,
    jam_type: str = None,
    split: str = 'test',
    max_samples: int = None,
    output_dir: str = 'results/inference',
    save_images: bool = True,
):
    """
    对指定JNR进行推理

    Args:
        jam_type: 指定干扰类型筛选，None表示处理所有样本
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 类别名称
    class_names = [cls["name"] for cls in config.get("jamming_classes", [])]
    print(f"Class names: {class_names}")

    # 数据路径
    data_config = config.get('data', {})
    base_path = data_config.get('base_path')
    jnr_folder = f"JNR_{'+' if jnr >= 0 else ''}{jnr}"
    data_folder = os.path.join(base_path, jnr_folder)

    stft_suffix = data_config.get('stft_suffix', 'echo_stfts')
    stft_file = os.path.join(data_folder, f'{split}_{stft_suffix}.mat')
    metadata_file = os.path.join(data_folder, f'{split}_echo_metadata.json')

    if not os.path.exists(stft_file):
        print(f"Error: STFT file not found: {stft_file}")
        return
    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file not found: {metadata_file}")
        return

    # 加载归一化统计量
    stats_file = 'multi/normalization_stats.json'
    normalization_stats = None
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            normalization_stats = json.load(f)

    # 创建数据集
    dataset = STFTDataset(
        stft_file=stft_file,
        metadata_file=metadata_file,
        normalization_stats=normalization_stats,
        class_names=class_names,
    )

    print(f"Total samples in dataset: {len(dataset)}")

    # 确定要处理的样本索引
    jam_type_idx = -1
    if jam_type:
        jam_type_idx = class_names.index(jam_type) if jam_type in class_names else -1
        if jam_type_idx == -1:
            print(f"Error: Unknown jam type '{jam_type}'")
            print(f"Available types: {class_names}")
            return

        # 找出包含该干扰类型的样本索引
        target_indices = []
        for i in range(len(dataset)):
            label = dataset._labels[i]
            if label[jam_type_idx] == 1:
                target_indices.append(i)
        print(f"Found {len(target_indices)} samples with {jam_type}")

        if len(target_indices) == 0:
            print("No samples found for the specified jam type")
            return

        if max_samples:
            target_indices = target_indices[:max_samples]
            print(f"Limiting to {len(target_indices)} samples")
    else:
        # 不筛选，处理所有样本
        target_indices = list(range(len(dataset)))
        if max_samples:
            target_indices = target_indices[:max_samples]
        print(f"Processing all {len(target_indices)} samples")

    # 创建子集
    from torch.utils.data import Subset
    subset = Subset(dataset, target_indices)
    loader = DataLoader(subset, batch_size=16, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

    # 加载模型
    model = create_czsl_model(config, device=str(device))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = checkpoint["model_state_dict"]
    model_state = model.state_dict()
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            filtered_state_dict[key] = value

    model.load_state_dict(filtered_state_dict, strict=False)
    print(f"Loaded checkpoint from {checkpoint_path}")

    # 缓存文本特征
    model.cache_text_features(max_combination_size=2, include_single=True)
    model.eval()

    # 推理
    all_labels = []
    all_preds = []
    all_probs = []
    all_sample_info = []  # 保存每个样本的详细信息
    correct = 0
    total = 0

    # 用于统计每个类别的预测情况
    class_tp = np.zeros(len(class_names))
    class_fp = np.zeros(len(class_names))
    class_fn = np.zeros(len(class_names))
    class_total = np.zeros(len(class_names))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stft_dir = output_path / f"JNR{jnr}_all" if not jam_type else output_path / f"JNR{jnr}_{jam_type}"
    if save_images:
        stft_dir.mkdir(exist_ok=True)

    info_str = f"JNR={jnr}dB, Split={split}"
    if jam_type:
        info_str += f", Type={jam_type}"
    print(f"\nInference: {info_str}")
    print("=" * 80)

    with torch.no_grad():
        for batch_idx, (images, _, _, labels, texts, metas) in enumerate(tqdm(loader, desc="Inference")):
            images = images.to(device)
            labels = labels.to(device)

            if images.shape[-1] != 224:
                images = nn.functional.interpolate(
                    images, size=(224, 224), mode='bilinear', align_corners=False
                )

            # 预测
            text_features = model.get_cached_text_features()
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            logit_scale = model.model.logit_scale.exp()
            logits = logit_scale * (image_features @ text_features.T)

            # 多标签预测
            probs = torch.softmax(logits, dim=-1)
            batch_size = images.shape[0]
            num_classes = len(class_names)
            threshold = 1.0 / num_classes
            top_k = 3
            topk_values, topk_indices = torch.topk(probs, k=top_k, dim=-1)

            preds = torch.zeros(batch_size, num_classes, device=device)
            for b in range(batch_size):
                for j, idx in enumerate(topk_indices[b]):
                    if topk_values[b, j] > threshold:
                        preds[b, idx] = 1.0

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

            # 统计
            for i in range(batch_size):
                true_set = set(torch.where(labels[i] == 1)[0].tolist())
                pred_set = set(torch.where(preds[i] == 1)[0].tolist())
                is_correct = (true_set == pred_set)

                if is_correct:
                    correct += 1
                total += 1

                # 统计每个类别
                for c in range(num_classes):
                    true_c = labels[i, c].item()
                    pred_c = preds[i, c].item()
                    class_total[c] += true_c
                    if true_c == 1 and pred_c == 1:
                        class_tp[c] += 1
                    elif true_c == 0 and pred_c == 1:
                        class_fp[c] += 1
                    elif true_c == 1 and pred_c == 0:
                        class_fn[c] += 1

                # 保存样本信息
                true_names = "+".join([class_names[idx] for idx in true_set])
                pred_names = "+".join([class_names[idx] for idx in pred_set])
                top5_vals, top5_idx = torch.topk(probs[i], k=5)
                sample_info = {
                    "true_labels": true_names,
                    "pred_labels": pred_names,
                    "is_correct": is_correct,
                    "top5_probs": [(class_names[idx.item()], round(val.item(), 4))
                                   for val, idx in zip(top5_vals, top5_idx)]
                }
                all_sample_info.append(sample_info)

                # 显示前几个样本的详细信息
                if batch_idx == 0 and i < 5:
                    print(f"\nSample {i}:")
                    print(f"  True labels: {true_names}")
                    print(f"  Pred labels: {pred_names}")
                    print(f"  Top-5 probs: {sample_info['top5_probs']}")

            # 保存图像
            if save_images:
                import cv2
                global_idx = batch_idx * 16  # 全局样本索引
                for i in range(batch_size):
                    img = images[i].cpu().numpy()
                    mean = np.array([0.48145466, 0.4578275, 0.40821073])
                    std = np.array([0.26862954, 0.26130258, 0.27577711])
                    img = img * std[:, None, None] + mean[:, None, None]
                    img = np.clip(img, 0, 1)
                    img = np.transpose(img, (1, 2, 0))
                    img = (img * 255).astype(np.uint8)

                    true_indices = torch.where(labels[i] == 1)[0].tolist()
                    pred_indices = torch.where(preds[i] == 1)[0].tolist()
                    true_set = set(true_indices)
                    pred_set = set(pred_indices)

                    true_names = "+".join([class_names[idx] for idx in true_indices])
                    pred_names = "+".join([class_names[idx] for idx in pred_indices])

                    # 判断匹配程度
                    if true_set == pred_set:
                        match_type = "exact"  # 完全匹配
                    elif true_set.issubset(pred_set):
                        match_type = "over"    # 预测过多（包含真实类别但多了）
                    elif pred_set.issubset(true_set):
                        match_type = "under"   # 预测不足（漏了部分真实类别）
                    elif len(true_set & pred_set) > 0:
                        match_type = "partial" # 部分匹配
                    else:
                        match_type = "miss"    # 完全错误

                    filename = f"{global_idx + i:04d}_{match_type}_true_{true_names}_pred_{pred_names}.png"
                    cv2.imwrite(str(stft_dir / filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # 汇总结果
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()

    print("\n" + "=" * 80)
    print("Inference Results Summary")
    print("=" * 80)
    print(f"JNR: {jnr} dB")
    if jam_type:
        print(f"Target Jamming Type: {jam_type}")
    print(f"Total samples: {total}")
    print(f"Exact match accuracy: {correct / total:.4f} ({correct}/{total})")

    # 每个类别的统计
    print("\nPer-Class Statistics:")
    print(f"{'Class':<10} {'Recall':>10} {'Precision':>12} {'F1':>10} {'Support':>10}")
    print("-" * 60)
    for c, name in enumerate(class_names):
        recall = class_tp[c] / (class_tp[c] + class_fn[c]) if (class_tp[c] + class_fn[c]) > 0 else 0
        precision = class_tp[c] / (class_tp[c] + class_fp[c]) if (class_tp[c] + class_fp[c]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"{name:<10} {recall:>10.4f} {precision:>12.4f} {f1:>10.4f} {int(class_total[c]):>10}")

    # 保存结果
    result_name = f"JNR{jnr}_all" if not jam_type else f"JNR{jnr}_{jam_type}"
    result_file = output_path / f"{result_name}_results.json"
    results = {
        "jnr": jnr,
        "split": split,
        "target_jam_type": jam_type,
        "total_samples": total,
        "exact_match_accuracy": correct / total,
        "per_class_recall": {name: float(class_tp[c] / (class_tp[c] + class_fn[c]) if (class_tp[c] + class_fn[c]) > 0 else 0)
                            for c, name in enumerate(class_names)},
        "per_class_precision": {name: float(class_tp[c] / (class_tp[c] + class_fp[c]) if (class_tp[c] + class_fp[c]) > 0 else 0)
                               for c, name in enumerate(class_names)},
        "samples": all_sample_info,
    }
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {result_file}")

    # 绘制概率分布图（针对每个类别绘制）
    for c, name in enumerate(class_names):
        if class_total[c] > 0:  # 只绘制有样本的类别
            plot_probability_distribution(
                all_probs, all_labels, class_names, c, name,
                save_path=str(output_path / f"{result_name}_{name}_probs.png"),
                jnr=jnr, split=split
            )

    return results


def plot_probability_distribution(all_probs, all_labels, class_names, target_idx, target_name, save_path=None, jnr=None, split=None):
    """绘制目标类别的概率分布"""
    target_probs = all_probs[:, target_idx]
    target_labels = all_labels[:, target_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 标题添加JNR和数据集信息
    info_str = f"JNR={jnr}dB, Split={split}" if jnr is not None else ""

    # 概率分布直方图
    ax = axes[0]
    ax.hist(target_probs[target_labels == 1], bins=30, alpha=0.7, label=f'True {target_name}', color='green')
    ax.hist(target_probs[target_labels == 0], bins=30, alpha=0.7, label=f'Not {target_name}', color='red')
    ax.axvline(x=1.0 / len(class_names), color='blue', linestyle='--', label='Threshold')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Count')
    ax.set_title(f'{target_name} Probability Distribution\n{info_str}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 所有类别的平均概率
    ax = axes[1]
    mean_probs = all_probs.mean(axis=0)
    colors = ['green' if i == target_idx else 'steelblue' for i in range(len(class_names))]
    bars = ax.bar(class_names, mean_probs, color=colors)
    ax.set_xlabel('Jamming Type')
    ax.set_ylabel('Mean Predicted Probability')
    ax.set_title(f'Mean Probability per Class\n{info_str}')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Probability plot saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Inference on specific JNR")
    parser.add_argument("--config", type=str, default="multi/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--jnr", type=int, default=20, help="JNR level in dB")
    parser.add_argument("--jam_type", type=str, default=None, help="Target jamming type (None for all)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--output_dir", type=str, default="results/inference")
    parser.add_argument("--no_save_images", action="store_true", help="Don't save STFT images")
    args = parser.parse_args()

    config = load_config(args.config)

    inference_specific_jnr_jamtype(
        config=config,
        checkpoint_path=args.checkpoint,
        jnr=args.jnr,
        jam_type=args.jam_type,
        split=args.split,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        save_images=not args.no_save_images,
    )


if __name__ == "__main__":
    main()
