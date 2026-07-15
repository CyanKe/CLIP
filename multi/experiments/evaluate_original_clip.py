"""
使用原始 CLIP 模型进行零样本推理
python -m multi.experiments.evaluate_original_clip --split test
"""
import os
import sys
import yaml
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import clip
from multi.data import create_czsl_dataloaders
from multi.text_templates import get_inference_description


class OriginalCLIPEvaluator:
    """原始 CLIP 模型的零样本评估器"""

    def __init__(self, model, device, class_names, preprocess):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.preprocess = preprocess
        self._text_features = None
        self._combination_features = None
        self._combination_names = None

    @torch.no_grad()
    def cache_text_features(self, max_combination_size=2, include_single=True, use_translation=False):
        """缓存文本特征"""
        self.model.eval()
        from itertools import combinations

        all_features = []
        all_names = []

        # 单干扰特征
        if include_single:
            for cls_name in self.class_names:
                desc = get_inference_description([cls_name], use_translation=use_translation)
                tokens = clip.tokenize(desc, truncate=True).to(self.device)
                features = self.model.encode_text(tokens)
                features = F.normalize(features, dim=-1)
                all_features.append(features)
                all_names.append(cls_name)

        # 组合特征
        if max_combination_size >= 2:
            for i, j in combinations(range(len(self.class_names)), 2):
                cls1, cls2 = self.class_names[i], self.class_names[j]
                desc = get_inference_description([cls1, cls2], use_translation=use_translation)
                tokens = clip.tokenize(desc, truncate=True).to(self.device)
                features = self.model.encode_text(tokens)
                features = F.normalize(features, dim=-1)
                all_features.append(features)
                all_names.append(f"{cls1}+{cls2}")

        self._combination_features = torch.cat(all_features, dim=0)
        self._combination_names = all_names
        self._text_features = self._combination_features[:len(self.class_names)]

        print(f"Cached {len(all_names)} text features")
        if use_translation:
            print("  (Using translated class names)")

    def get_text_features(self):
        if self._text_features is None:
            self.cache_text_features()
        return self._text_features

    @torch.no_grad()
    def evaluate_zero_shot(self, data_loader, use_combinations=True):
        """零样本评估"""
        self.model.eval()

        all_labels = []
        all_preds = []
        all_combination_correct = 0
        total_samples = 0

        # 选择文本特征
        if use_combinations:
            text_features = self._combination_features
            names = self._combination_names
        else:
            text_features = self._text_features
            names = self.class_names

        logit_scale = self.model.logit_scale.exp()

        for images, _, text_tokens, labels, texts, metas in tqdm(data_loader, desc="Zero-Shot (Original CLIP)"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 调整图像尺寸
            if images.shape[-1] != 224:
                images = nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

            batch_size = images.shape[0]

            # 编码图像
            image_features = self.model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)

            # 计算相似度
            logits = logit_scale * (image_features @ text_features.T)

            if use_combinations:
                # 组合模式：选择 top-1
                _, indices = torch.topk(logits, k=1, dim=-1)
                preds = torch.zeros(batch_size, len(self.class_names), device=self.device)

                for i, idx in enumerate(indices):
                    comb_name = names[idx.item()]
                    parts = comb_name.split('+')
                    for part in parts:
                        if part in self.class_names:
                            preds[i, self.class_names.index(part)] = 1
            else:
                # 多标签模式：softmax + top-k
                probs = torch.softmax(logits, dim=-1)
                num_classes = len(self.class_names)
                threshold = 1.0 / num_classes
                top_k = 3
                topk_values, topk_indices = torch.topk(probs, k=top_k, dim=-1)

                preds = torch.zeros(batch_size, num_classes, device=self.device)
                for b in range(batch_size):
                    for j, idx in enumerate(topk_indices[b]):
                        if topk_values[b, j] > threshold:
                            preds[b, idx] = 1.0

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

            # 计算组合准确率
            for i in range(batch_size):
                true_set = set(torch.where(labels[i] == 1)[0].tolist())
                pred_set = set(torch.where(preds[i] == 1)[0].tolist())
                if true_set == pred_set:
                    all_combination_correct += 1

            total_samples += batch_size

        # 合并结果
        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()

        # 计算指标
        metrics = {
            "combination_accuracy": all_combination_correct / total_samples,
            "partial_match_accuracy": np.mean([
                len(set(np.where(all_labels[i] == 1)[0]) & set(np.where(all_preds[i] == 1)[0])) > 0
                for i in range(len(all_labels))
            ]),
            "f1_macro": f1_score(all_labels, all_preds, average='macro', zero_division=0),
            "f1_micro": f1_score(all_labels, all_preds, average='micro', zero_division=0),
            "precision_macro": precision_score(all_labels, all_preds, average='macro', zero_division=0),
            "recall_macro": recall_score(all_labels, all_preds, average='macro', zero_division=0),
        }

        # 每个类别的指标
        metrics["per_class"] = {
            "f1": f1_score(all_labels, all_preds, average=None, zero_division=0),
            "precision": precision_score(all_labels, all_preds, average=None, zero_division=0),
            "recall": recall_score(all_labels, all_preds, average=None, zero_division=0)
        }

        return {"metrics": metrics, "labels": all_labels, "predictions": all_preds}

    def print_metrics(self, metrics):
        """打印评估指标"""
        print("\n" + "=" * 60)
        print("Original CLIP Zero-Shot Evaluation Results")
        print("=" * 60)
        print(f"Combination Accuracy:     {metrics['combination_accuracy']:.4f}")
        print(f"Partial Match Accuracy:   {metrics['partial_match_accuracy']:.4f}")
        print(f"Macro F1 Score:           {metrics['f1_macro']:.4f}")
        print(f"Micro F1 Score:           {metrics['f1_micro']:.4f}")
        print(f"Macro Precision:          {metrics['precision_macro']:.4f}")
        print(f"Macro Recall:             {metrics['recall_macro']:.4f}")

        print("\nPer-class Metrics:")
        print("-" * 60)
        print(f"{'Class':<10} {'F1':>8} {'Precision':>12} {'Recall':>10}")
        print("-" * 60)
        for i, name in enumerate(self.class_names[:len(metrics['per_class']['f1'])]):
            print(f"{name:<10} {metrics['per_class']['f1'][i]:>8.4f} "
                  f"{metrics['per_class']['precision'][i]:>12.4f} "
                  f"{metrics['per_class']['recall'][i]:>10.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate with Original CLIP")
    parser.add_argument("--config", type=str, default="multi/config.yaml")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32",
                        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--mode", type=str, default="zero_shot", choices=["zero_shot", "multilabel"])
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading original CLIP model: {args.clip_model}")

    # 加载原始 CLIP
    model, preprocess = clip.load(args.clip_model, device=device)
    model = model.float().eval()

    # 类别名称
    class_names = [cls["name"] for cls in config.get("jamming_classes", [])]
    print(f"Classes: {class_names}")

    # 创建评估器
    evaluator = OriginalCLIPEvaluator(model, device, class_names, preprocess)

    # 缓存文本特征
    evaluator.cache_text_features(max_combination_size=2, include_single=True)

    # 创建数据加载器
    train_loader, val_loader, test_loader, _ = create_czsl_dataloaders(config)

    if args.split == "train":
        data_loader = train_loader
    elif args.split == "val":
        data_loader = val_loader
    else:
        data_loader = test_loader

    # 评估
    use_combinations = (args.mode == "zero_shot")
    results = evaluator.evaluate_zero_shot(data_loader, use_combinations=use_combinations)
    evaluator.print_metrics(results["metrics"])


if __name__ == "__main__":
    main()
