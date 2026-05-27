"""
多形状 Patch ViT 训练脚本

使用 early_fusion 模式融合三种patch形状：
- 横向条形 (8x32): 捕捉时间维度特征
- 纵向条形 (32x8): 捕捉频率维度特征
- 正方形 (16x16): 捕捉局部空间特征

使用方法:
    python -m multi.train_multishape_vit --config multi/config.yaml

配置示例 (在 config.yaml 中添加):
    model:
      patch_sizes: [[8, 32], [32, 8], [16, 16]]
      embed_dim: 512
      depth: 6
      num_heads: 8
      fusion_mode: "early_fusion"
"""
import os
import sys
import yaml
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi.rectangular_patch_vit import MultiShapePatchViTForCZSL, create_multi_shape_patch_model
from multi.data import create_czsl_dataloaders, create_preprocessed_dataloaders
from multi.loss import create_loss_function, LabelAwareInfoNCELoss, MultiLabelInfoNCELoss, MultiLabelSigmoidLoss


class MultiShapeViTTrainer:
    """多形状 Patch ViT 训练器"""

    def __init__(
        self,
        model: MultiShapePatchViTForCZSL,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device: torch.device,
        config: dict,
        use_wandb: bool = True
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.use_wandb = use_wandb and HAS_WANDB

        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.train_config = config.get("train", {})
        self.grad_clip = self.train_config.get("grad_clip", 1.0)

        self.checkpoint_config = config.get("checkpoint", {})
        self.save_dir = Path(self.checkpoint_config.get("save_dir", "checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 损失函数
        self.loss_fn = create_loss_function(config)
        self.use_label_aware_loss = isinstance(self.loss_fn, LabelAwareInfoNCELoss)
        self.use_multilabel_infonce = isinstance(self.loss_fn, MultiLabelInfoNCELoss)
        self.use_sigmoid_loss = isinstance(self.loss_fn, MultiLabelSigmoidLoss)
        print(f"Using loss function: {type(self.loss_fn).__name__}")

    def train_epoch(self, debug: bool = False) -> dict:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        train_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")

        debug_done = False
        for batch_idx, batch_data in enumerate(train_bar):
            if len(batch_data) >= 7:
                stft_images, time_signals, text_tokens, labels, texts, metas, features_batched = batch_data
                has_time_signal = time_signals is not None
            elif len(batch_data) == 6:
                stft_images, time_signals, text_tokens, labels, texts, metas = batch_data
                features_batched = None
                has_time_signal = True
            else:
                stft_images, text_tokens, labels, texts, metas = batch_data
                time_signals = None
                features_batched = None
                has_time_signal = False

            stft_images = stft_images.to(self.device)
            text_tokens = text_tokens.to(self.device)
            labels = labels.to(self.device)

            # 调整图像尺寸
            if stft_images.shape[-1] != 224:
                stft_images = nn.functional.interpolate(
                    stft_images, size=(224, 224), mode='bilinear', align_corners=False
                )

            batch_size = stft_images.size(0)

            # Debug
            if debug and not debug_done and batch_idx == 0:
                print(f"\n[DEBUG] Batch {batch_idx}, batch_size={batch_size}")
                print(f"  texts[0]: {texts[0]}")
                print(f"  labels[0]: {labels[0].tolist()}")
                debug_done = True

            self.optimizer.zero_grad()

            # 前向传播
            image_features, text_features = self.model(stft_images, text_tokens, time_signals)

            # 计算损失
            if self.use_sigmoid_loss:
                loss, logits_per_image, _ = self.loss_fn(image_features, text_features, labels)
                logits_per_text = logits_per_image.T
            elif self.use_label_aware_loss or self.use_multilabel_infonce:
                loss, logits_per_image, _ = self.loss_fn(image_features, text_features, labels)
                logits_per_text = logits_per_image.T
            else:
                logit_scale = self.model.logit_scale.exp()
                logits_per_image = logit_scale * (image_features @ text_features.t())
                logits_per_text = logits_per_image.T

                targets = torch.arange(batch_size, device=self.device)
                loss_i2t = F.cross_entropy(logits_per_image, targets)
                loss_t2i = F.cross_entropy(logits_per_text, targets)
                loss = (loss_i2t + loss_t2i) / 2

            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            total_loss += loss.item() * batch_size

            # 标签感知准确率（argmax 命中共享标签的样本即算正确）
            with torch.no_grad():
                labels_f = labels.float()
                pos_mask = (labels_f @ labels_f.T) > 0
                pred_i2t = logits_per_image.argmax(dim=1)
                total_correct += pos_mask[torch.arange(batch_size, device=self.device), pred_i2t].sum().item()
                pred_t2i = logits_per_text.argmax(dim=1)
                total_correct += pos_mask[torch.arange(batch_size, device=self.device), pred_t2i].sum().item()
                total_samples += batch_size * 2

            train_bar.set_postfix(loss=loss.item())

        return {
            "loss": total_loss / len(self.train_loader.dataset),
            "accuracy": total_correct / total_samples
        }

    @torch.no_grad()
    def validate(self) -> dict:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        val_bar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")

        for batch_data in val_bar:
            if len(batch_data) >= 7:
                stft_images, time_signals, text_tokens, labels, texts, metas, features_batched = batch_data
                has_time_signal = time_signals is not None
            elif len(batch_data) == 6:
                stft_images, time_signals, text_tokens, labels, texts, metas = batch_data
                features_batched = None
                has_time_signal = True
            else:
                stft_images, text_tokens, labels, texts, metas = batch_data
                time_signals = None
                features_batched = None
                has_time_signal = False

            stft_images = stft_images.to(self.device)
            text_tokens = text_tokens.to(self.device)
            labels = labels.to(self.device)

            if stft_images.shape[-1] != 224:
                stft_images = nn.functional.interpolate(
                    stft_images, size=(224, 224), mode='bilinear', align_corners=False
                )

            batch_size = stft_images.size(0)

            image_features, text_features = self.model(stft_images, text_tokens, time_signals)

            if self.use_sigmoid_loss:
                loss, logits_per_image, _ = self.loss_fn(image_features, text_features, labels)
                logits_per_text = logits_per_image.T
            elif self.use_label_aware_loss or self.use_multilabel_infonce:
                loss, logits_per_image, _ = self.loss_fn(image_features, text_features, labels)
                logits_per_text = logits_per_image.T
            else:
                logit_scale = self.model.logit_scale.exp()
                logits_per_image = logit_scale * (image_features @ text_features.t())
                logits_per_text = logits_per_image.T

                targets = torch.arange(batch_size, device=self.device)
                loss_i2t = F.cross_entropy(logits_per_image, targets)
                loss_t2i = F.cross_entropy(logits_per_text, targets)
                loss = (loss_i2t + loss_t2i) / 2

            total_loss += loss.item() * batch_size

            # 标签感知准确率
            labels_f = labels.float()
            pos_mask = (labels_f @ labels_f.T) > 0
            pred_i2t = logits_per_image.argmax(dim=1)
            total_correct += pos_mask[torch.arange(batch_size, device=self.device), pred_i2t].sum().item()
            pred_t2i = logits_per_text.argmax(dim=1)
            total_correct += pos_mask[torch.arange(batch_size, device=self.device), pred_t2i].sum().item()
            total_samples += batch_size * 2

            val_bar.set_postfix(loss=loss.item())

        return {
            "loss": total_loss / len(self.val_loader.dataset),
            "accuracy": total_correct / total_samples
        }

    def save_checkpoint(self, metrics: dict, is_best: bool = False):
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "metrics": metrics,
            "config": self.config
        }

        # 保存最新检查点
        latest_path = self.save_dir / "multishape_vit_latest.pt"
        torch.save(checkpoint, latest_path)

        # 只保存最佳模型
        if is_best:
            best_path = self.save_dir / "multishape_vit_best.pt"
            torch.save(checkpoint, best_path)
            print(f"  ★ Saved best model (loss: {metrics['loss']:.4f}, acc: {metrics['accuracy']:.4f})")

    def fit(self, num_epochs: int, debug: bool = False) -> dict:
        print(f"\n{'='*70}")
        print(f"Multi-Shape Patch ViT Training")
        print(f"  Patch sizes: {self.config.get('model', {}).get('patch_sizes', [(8,32), (32,8), (16,16)])}")
        print(f"  Fusion mode: {self.config.get('model', {}).get('fusion_mode', 'early_fusion')}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Device: {self.device}")
        print(f"{'='*70}\n")

        if self.use_wandb:
            wandb_config = self.config.get("logging", {}).get("wandb", {})
            wandb.init(
                project=wandb_config.get("project", "CLIP-CZSL-Jamming"),
                entity=wandb_config.get("entity", None),
                name=f"MultiShapeViT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=wandb_config.get("tags", []) + ["MultiShapeViT", "early_fusion"],
                notes="Multi-shape patch ViT with early fusion",
                config=self.config
            )
            wandb.watch(self.model, log="all", log_freq=100)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            train_metrics = self.train_epoch(debug=debug and epoch == 0)
            self.scheduler.step()

            val_metrics = self.validate()

            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "lr": self.scheduler.get_last_lr()[0],
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["accuracy"]
                })

            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]
                self.best_val_acc = val_metrics["accuracy"]

            self.save_checkpoint(val_metrics, is_best)

        if self.use_wandb:
            wandb.finish()

        print(f"\n{'='*70}")
        print(f"Training completed!")
        print(f"  Best validation loss: {self.best_val_loss:.4f}")
        print(f"  Best validation acc:  {self.best_val_acc:.4f}")
        print(f"{'='*70}")

        return {"best_val_loss": self.best_val_loss, "best_val_acc": self.best_val_acc}


def create_optimizer_and_scheduler(model: nn.Module, config: dict, num_training_steps: int) -> tuple:
    train_config = config.get("train", {})

    optimizer = AdamW(
        model.parameters(),
        lr=float(train_config.get("lr", 1e-5)),
        weight_decay=float(train_config.get("weight_decay", 0.01))
    )

    scheduler_config = train_config.get("scheduler", {})
    scheduler_type = scheduler_config.get("type", "cosine")
    warmup_epochs = train_config.get("warmup_epochs", 2)
    num_epochs = train_config.get("epochs", 20)

    if scheduler_type == "cosine":
        actual_warmup = min(warmup_epochs, num_epochs - 1)
        t_max = max(1, num_epochs - actual_warmup)

        if actual_warmup > 0:
            warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=actual_warmup)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=scheduler_config.get("min_lr", 1e-7))
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[actual_warmup])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=scheduler_config.get("min_lr", 1e-7))
    else:
        scheduler = None

    return optimizer, scheduler


def main():
    parser = argparse.ArgumentParser(description="Multi-Shape Patch ViT Training")
    parser.add_argument("--config", type=str, default="multi/config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    parser.add_argument("--preprocessed", action="store_true", help="Use preprocessed .pt files (run preprocess_stft.py first)")
    parser.add_argument("--patch-sizes", type=str, default=None, help="Override patch sizes, e.g., '8,32;32,8;16,16'")
    parser.add_argument("--fusion-mode", type=str, default="late_fusion", choices=["early_fusion", "late_fusion"])
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 命令行参数覆盖配置文件
    if args.patch_sizes:
        patch_sizes = []
        for ps in args.patch_sizes.split(';'):
            h, w = map(int, ps.split(','))
            patch_sizes.append((h, w))
        config['model']['patch_sizes'] = patch_sizes

    config['model']['fusion_mode'] = args.fusion_mode
    config['model']['embed_dim'] = args.embed_dim
    config['model']['depth'] = args.depth

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建数据加载器
    print("\nLoading datasets...")
    if args.preprocessed:
        train_loader, val_loader, test_loader, num_classes = create_preprocessed_dataloaders(
            config=config,
            batch_size=config.get("train", {}).get("batch_size", 16),
            num_workers=config.get("data", {}).get("num_workers", 4),
            pin_memory=config.get("data", {}).get("pin_memory", True),
            load_test=False,
        )
    else:
        train_loader, val_loader, test_loader, num_classes = create_czsl_dataloaders(
            config=config,
            batch_size=config.get("train", {}).get("batch_size", 16),
            num_workers=config.get("data", {}).get("num_workers", 4),
            pin_memory=config.get("data", {}).get("pin_memory", True),
            load_test=False,
        )

    # 创建模型
    print("\nCreating Multi-Shape Patch ViT model...")
    model = create_multi_shape_patch_model(config, device=str(device))

    # 缓存文本特征
    czsl_config = config.get("czsl", {})
    seen_combos = czsl_config.get("seen_combinations", None)
    model.cache_text_features(
        max_combination_size=2,
        include_single=True,
        seen_combinations=seen_combos
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Resumed from {args.resume}")

    # 创建优化器和调度器
    num_training_steps = len(train_loader) * config.get("train", {}).get("epochs", 20)
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, num_training_steps)

    # WandB
    log_type = config.get("logging", {}).get("type", "console")
    use_wandb = (log_type == "wandb") and HAS_WANDB

    # 创建训练器
    trainer = MultiShapeViTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        use_wandb=use_wandb
    )

    # 开始训练
    num_epochs = config.get("train", {}).get("epochs", 20)
    trainer.fit(num_epochs, debug=args.debug)


if __name__ == "__main__":
    main()
