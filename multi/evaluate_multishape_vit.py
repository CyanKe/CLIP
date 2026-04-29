"""
多形状 Patch ViT 推理脚本

支持:
- 零样本分类
- 组合零样本学习 (CZSL)
- 详细评估报告

使用方法:
    python -m multi.evaluate_multishape_vit --checkpoint checkpoints/multishape_vit_best.pt --config multi/config.yaml
"""
import os
import sys
import yaml
import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi.rectangular_patch_vit import MultiShapePatchViTForCZSL, create_multi_shape_patch_model
from multi.data import create_czsl_dataloaders


class MultiShapeViTEvaluator:
    """多形状 Patch ViT 评估器"""

    def __init__(
        self,
        model: MultiShapePatchViTForCZSL,
        test_loader,
        device: torch.device,
        config: dict
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.config = config

        # 获取类别信息
        self.class_names = model.class_names
        self.num_classes = len(self.class_names)

        # CZSL 配置
        czsl_config = config.get("czsl", {})
        self.seen_combinations = czsl_config.get("seen_combinations", [])
        self.unseen_combinations = czsl_config.get("unseen_combinations", [])

    @torch.no_grad()
    def evaluate_zero_shot(
        self,
        use_combinations: bool = True,
        threshold: float = 0.5
    ) -> dict:
        """
        零样本评估

        Args:
            use_combinations: 是否使用组合特征
            threshold: 多标签预测阈值

        Returns:
            评估结果字典
        """
        self.model.eval()

        # 缓存文本特征
        self.model.cache_text_features(
            max_combination_size=2,
            include_single=True,
            seen_combinations=self.seen_combinations
        )

        all_preds = []
        all_labels = []
        all_similarities = []

        test_bar = tqdm(self.test_loader, desc="Zero-shot evaluation")

        for batch_data in test_bar:
            if len(batch_data) == 6:
                stft_images, time_signals, text_tokens, labels, texts, metas = batch_data
            else:
                stft_images, text_tokens, labels, texts, metas = batch_data
                time_signals = None

            stft_images = stft_images.to(self.device)

            # 调整图像尺寸
            if stft_images.shape[-1] != 224:
                stft_images = F.interpolate(stft_images, size=(224, 224), mode='bilinear', align_corners=False)

            # 零样本预测
            similarities, indices, pred_names = self.model.zero_shot_predict(
                stft_images,
                time_signal=time_signals.to(self.device) if time_signals is not None else None,
                use_combinations=use_combinations,
                top_k=self.num_classes
            )

            all_similarities.append(similarities.cpu())
            all_labels.append(labels)

        # 合并结果
        all_similarities = torch.cat(all_similarities, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # 获取预测文本特征
        if use_combinations:
            text_features = self.model.get_cached_combination_features()
            names = self.model._combination_names
        else:
            text_features = self.model.get_cached_text_features()
            names = self.class_names

        # 计算多标签预测
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * all_similarities

        # 转换为单干扰类型预测
        # 对于组合预测，需要分解为单个类别
        preds_binary = torch.zeros(all_labels.shape[0], self.num_classes)

        for i in range(all_similarities.shape[0]):
            top_k_indices = torch.topk(all_similarities[i], k=5).indices
            for idx in top_k_indices:
                pred_name = names[idx]
                if '+' in pred_name:
                    # 组合预测
                    parts = pred_name.split('+')
                    for part in parts:
                        if part in self.class_names:
                            cls_idx = self.class_names.index(part)
                            preds_binary[i, cls_idx] = 1
                else:
                    # 单类别预测
                    if pred_name in self.class_names:
                        cls_idx = self.class_names.index(pred_name)
                        preds_binary[i, cls_idx] = 1

        # 计算指标
        labels_np = all_labels.numpy()
        preds_np = preds_binary.numpy()

        results = {
            "hamming_accuracy": (preds_np == labels_np).mean(),
            "exact_match": (preds_np == labels_np).all(axis=1).mean(),
            "macro_f1": f1_score(labels_np, preds_np, average='macro', zero_division=0),
            "micro_f1": f1_score(labels_np, preds_np, average='micro', zero_division=0),
            "samples_f1": f1_score(labels_np, preds_np, average='samples', zero_division=0),
        }

        # 每个类别的详细指标
        per_class_metrics = {}
        for i, cls_name in enumerate(self.class_names):
            per_class_metrics[cls_name] = {
                "precision": precision_score(labels_np[:, i], preds_np[:, i], zero_division=0),
                "recall": recall_score(labels_np[:, i], preds_np[:, i], zero_division=0),
                "f1": f1_score(labels_np[:, i], preds_np[:, i], zero_division=0),
                "support": labels_np[:, i].sum()
            }

        results["per_class_metrics"] = per_class_metrics

        return results

    @torch.no_grad()
    def evaluate_seen_unseen(
        self,
        threshold: float = 0.5
    ) -> dict:
        """
        分别评估 seen 和 unseen 组合的性能

        Returns:
            seen 和 unseen 的评估结果
        """
        self.model.eval()

        # 缓存文本特征
        self.model.cache_text_features(
            max_combination_size=2,
            include_single=True,
            seen_combinations=self.seen_combinations
        )

        # 解析 seen/unseen 组合
        seen_set = set()
        for combo in self.seen_combinations:
            if len(combo) == 1:
                seen_set.add(combo[0])
            else:
                seen_set.add('+'.join(sorted(combo)))

        unseen_set = set()
        for combo in self.unseen_combinations:
            if len(combo) == 1:
                unseen_set.add(combo[0])
            else:
                unseen_set.add('+'.join(sorted(combo)))

        seen_correct = 0
        seen_total = 0
        unseen_correct = 0
        unseen_total = 0

        test_bar = tqdm(self.test_loader, desc="Seen/Unseen evaluation")

        for batch_data in test_bar:
            if len(batch_data) == 6:
                stft_images, time_signals, text_tokens, labels, texts, metas = batch_data
            else:
                stft_images, text_tokens, labels, texts, metas = batch_data
                time_signals = None

            stft_images = stft_images.to(self.device)

            if stft_images.shape[-1] != 224:
                stft_images = F.interpolate(stft_images, size=(224, 224), mode='bilinear', align_corners=False)

            # 零样本预测
            similarities, indices, pred_names = self.model.zero_shot_predict(
                stft_images,
                time_signal=time_signals.to(self.device) if time_signals is not None else None,
                use_combinations=True,
                top_k=1
            )

            # 获取真实标签组合
            for i, meta in enumerate(metas):
                jam_types = meta.get('jam_types', [])
                true_combo = '+'.join(sorted(jam_types)) if jam_types else ''

                pred_name = pred_names[i][0] if pred_names[i] else ''

                if true_combo in seen_set:
                    seen_total += 1
                    if pred_name == true_combo:
                        seen_correct += 1
                elif true_combo in unseen_set:
                    unseen_total += 1
                    if pred_name == true_combo:
                        unseen_correct += 1

        results = {
            "seen_accuracy": seen_correct / seen_total if seen_total > 0 else 0,
            "unseen_accuracy": unseen_correct / unseen_total if unseen_total > 0 else 0,
            "seen_total": seen_total,
            "unseen_total": unseen_total,
            "harmonic_mean": self._harmonic_mean(
                seen_correct / seen_total if seen_total > 0 else 0,
                unseen_correct / unseen_total if unseen_total > 0 else 0
            )
        }

        return results

    def _harmonic_mean(self, a: float, b: float) -> float:
        if a + b == 0:
            return 0
        return 2 * a * b / (a + b)

    def generate_report(self, results: dict, save_path: str = None):
        """生成详细报告"""
        report = []
        report.append("=" * 70)
        report.append("Multi-Shape Patch ViT Evaluation Report")
        report.append("=" * 70)

        # 模型配置
        model_config = self.config.get("model", {})
        report.append(f"\nModel Configuration:")
        report.append(f"  Patch sizes: {model_config.get('patch_sizes', [(8,32), (32,8), (16,16)])}")
        report.append(f"  Fusion mode: {model_config.get('fusion_mode', 'early_fusion')}")
        report.append(f"  Embed dim: {model_config.get('embed_dim', 512)}")
        report.append(f"  Depth: {model_config.get('depth', 6)}")

        # 总体指标
        report.append(f"\nOverall Metrics:")
        report.append(f"  Hamming Accuracy: {results.get('hamming_accuracy', 0):.4f}")
        report.append(f"  Exact Match:      {results.get('exact_match', 0):.4f}")
        report.append(f"  Macro F1:         {results.get('macro_f1', 0):.4f}")
        report.append(f"  Micro F1:         {results.get('micro_f1', 0):.4f}")
        report.append(f"  Samples F1:       {results.get('samples_f1', 0):.4f}")

        # Seen/Unseen 结果
        if 'seen_accuracy' in results:
            report.append(f"\nCZSL Results:")
            report.append(f"  Seen Accuracy:   {results['seen_accuracy']:.4f} ({results['seen_total']} samples)")
            report.append(f"  Unseen Accuracy: {results['unseen_accuracy']:.4f} ({results['unseen_total']} samples)")
            report.append(f"  Harmonic Mean:   {results['harmonic_mean']:.4f}")

        # 每个类别的指标
        if 'per_class_metrics' in results:
            report.append(f"\nPer-Class Metrics:")
            report.append(f"  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
            report.append(f"  {'-'*55}")
            for cls_name, metrics in results['per_class_metrics'].items():
                report.append(f"  {cls_name:<15} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
                            f"{metrics['f1']:>10.4f} {int(metrics['support']):>10}")

        report.append("\n" + "=" * 70)

        report_text = '\n'.join(report)
        print(report_text)

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"\nReport saved to {save_path}")

        return report_text


def main():
    parser = argparse.ArgumentParser(description="Multi-Shape Patch ViT Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="multi/config.yaml", help="Path to config file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold")
    parser.add_argument("--output", type=str, default="results/multishape_vit_eval.json", help="Output file path")
    parser.add_argument("--use-combinations", action="store_true", default=True, help="Use combination features")
    args = parser.parse_args()

    # 加载检查点获取模型配置
    print(f"\nLoading checkpoint: {args.checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # 优先使用检查点中保存的配置
    if "config" in checkpoint:
        config = checkpoint["config"]
        print("  Using config from checkpoint")
    else:
        # 回退到配置文件
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("  Using config from file")

    # 确保模型配置存在
    model_config = config.get("model", {})
    if "patch_sizes" not in model_config:
        # 设置默认值
        model_config["patch_sizes"] = [(8, 32), (32, 8), (16, 16)]
        model_config["fusion_mode"] = "early_fusion"
        model_config["embed_dim"] = 512
        model_config["depth"] = 6
        model_config["num_heads"] = 8
        config["model"] = model_config
        print("  Using default model config (not found in checkpoint)")

    print(f"Using device: {device}")

    # 加载数据
    print("\nLoading test dataset...")
    train_loader, val_loader, test_loader, num_classes = create_czsl_dataloaders(
        config=config,
        batch_size=config.get("train", {}).get("batch_size", 16),
        num_workers=config.get("data", {}).get("num_workers", 4),
        pin_memory=config.get("data", {}).get("pin_memory", True),
        load_test=True,
    )

    if test_loader is None:
        print("Error: No test data available!")
        return

    # 创建模型
    print("\nCreating model...")
    model = create_multi_shape_patch_model(config, device=str(device))

    # 加载模型权重
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")

    # 创建评估器
    evaluator = MultiShapeViTEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        config=config
    )

    # 运行评估
    print("\n" + "=" * 70)
    print("Starting evaluation...")
    print("=" * 70)

    # 零样本评估
    results = evaluator.evaluate_zero_shot(
        use_combinations=args.use_combinations,
        threshold=args.threshold
    )

    # Seen/Unseen 评估
    czsl_results = evaluator.evaluate_seen_unseen(threshold=args.threshold)
    results.update(czsl_results)

    # 保存结果
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 转换为可序列化格式
    def convert_to_serializable(obj):
        """将numpy类型转换为Python原生类型"""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    results_to_save = convert_to_serializable(results)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")

    # 生成报告
    report_path = args.output.replace('.json', '_report.txt')
    evaluator.generate_report(results, save_path=report_path)


if __name__ == "__main__":
    main()
