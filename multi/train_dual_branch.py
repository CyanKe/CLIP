"""
双分支 CZSL 训练脚本 - 用于欺骗/压制干扰分类
python -m multi.train_dual_branch --config multi/config.yaml

支持两种模型:
1. CLIP 双分支模型 (默认): 使用预训练 CLIP 视觉编码器
2. 多形状 Patch ViT 双分支模型: 使用自定义多形状 patch 视觉编码器

配置示例 (使用多形状 Patch ViT):
```yaml
model:
  use_multishape_vit: true
  patch_sizes: [[8, 32], [32, 8], [16, 16]]
  embed_dim: 512
  depth: 6
  num_heads: 8
  fusion_mode: "early_fusion"
```
"""
import os
import sys
import yaml
import argparse
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
    print("Warning: wandb not installed. Using console logging only.")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi.model import create_dual_branch_model, DualBranchCLIPForCZSL
from multi.rectangular_patch_vit import create_multi_shape_dual_branch_model, MultiShapePatchViTForDualBranch
from multi.data import create_dual_branch_dataloaders
from multi.loss import DualBranchContrastiveLoss


class DualBranchTrainer:
    """双分支 CZSL 训练器"""

    def __init__(
        self,
        model: DualBranchCLIPForCZSL,
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
        self.train_config = config.get("train", {})
        self.grad_clip = self.train_config.get("grad_clip", 1.0)

        self.checkpoint_config = config.get("checkpoint", {})
        self.save_dir = Path(self.checkpoint_config.get("save_dir", "checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 初始化损失函数
        loss_config = config.get("loss", {})
        self.loss_fn = DualBranchContrastiveLoss(
            temperature=loss_config.get("temperature", 0.07),
            learnable_temperature=loss_config.get("learnable_temperature", True),
            label_smoothing=loss_config.get("label_smoothing", 0.0)
        )
        print(f"Using DualBranchContrastiveLoss")

    def train_epoch(self, debug: bool = False) -> dict:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct_deception = 0
        total_correct_suppression = 0
        total_samples = 0

        train_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")

        debug_done = False
        for batch_idx, batch_data in enumerate(train_bar):
            (stft_images, text_tokens_deception, text_tokens_suppression,
             labels_deception, labels_suppression, texts_deception, texts_suppression,
             metadata_list) = batch_data

            stft_images = stft_images.to(self.device)
            text_tokens_deception = text_tokens_deception.to(self.device)
            text_tokens_suppression = text_tokens_suppression.to(self.device)
            labels_deception = labels_deception.to(self.device)
            labels_suppression = labels_suppression.to(self.device)

            # 调整图像尺寸
            if stft_images.shape[-1] != 224:
                stft_images = nn.functional.interpolate(
                    stft_images, size=(224, 224), mode='bilinear', align_corners=False
                )

            batch_size = stft_images.size(0)

            # Debug
            if debug and not debug_done and batch_idx == 0:
                print(f"\n{'='*80}")
                print(f"[DEBUG Train] Batch {batch_idx} — batch_size={batch_size}")
                print(f"  labels_deception[0]: {labels_deception[0].tolist()}")
                print(f"  labels_suppression[0]: {labels_suppression[0].tolist()}")
                print(f"  text_deception[0]: {texts_deception[0]}")
                print(f"  text_suppression[0]: {texts_suppression[0]}")
                print(f"{'='*80}")
                debug_done = True

            self.optimizer.zero_grad()

            # 前向传播
            image_features, text_features_deception, text_features_suppression = self.model(
                stft_images, text_tokens_deception, text_tokens_suppression
            )

            # 计算损失
            loss, loss_info = self.loss_fn(
                image_features,
                text_features_deception,
                text_features_suppression,
                labels_deception,
                labels_suppression
            )

            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            total_loss += loss.item() * batch_size

            # 计算准确率 (对角线准确率作为参考)
            with torch.no_grad():
                logit_scale = self.model.model.logit_scale.exp()

                # 欺骗分支准确率
                logits_deception = logit_scale * (image_features @ text_features_deception.t())
                targets = torch.arange(batch_size, device=self.device)
                pred_deception = logits_deception.argmax(dim=1)
                total_correct_deception += (pred_deception == targets).sum().item()

                # 压制分支准确率
                logits_suppression = logit_scale * (image_features @ text_features_suppression.t())
                pred_suppression = logits_suppression.argmax(dim=1)
                total_correct_suppression += (pred_suppression == targets).sum().item()

                total_samples += batch_size

            train_bar.set_postfix(loss=loss.item())

        return {
            "loss": total_loss / len(self.train_loader.dataset),
            "accuracy_deception": total_correct_deception / total_samples,
            "accuracy_suppression": total_correct_suppression / total_samples
        }

    @torch.no_grad()
    def validate(self, debug: bool = False) -> dict:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        total_correct_deception = 0
        total_correct_suppression = 0
        total_samples = 0

        val_bar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")

        debug_done = False
        for batch_idx, batch_data in enumerate(val_bar):
            (stft_images, text_tokens_deception, text_tokens_suppression,
             labels_deception, labels_suppression, texts_deception, texts_suppression,
             metadata_list) = batch_data

            stft_images = stft_images.to(self.device)
            text_tokens_deception = text_tokens_deception.to(self.device)
            text_tokens_suppression = text_tokens_suppression.to(self.device)
            labels_deception = labels_deception.to(self.device)
            labels_suppression = labels_suppression.to(self.device)

            if stft_images.shape[-1] != 224:
                stft_images = nn.functional.interpolate(
                    stft_images, size=(224, 224), mode='bilinear', align_corners=False
                )

            batch_size = stft_images.size(0)

            if debug and not debug_done:
                print(f"\n{'='*80}")
                print(f"[DEBUG Val] Batch {batch_idx} — batch_size={batch_size}")
                print(f"  text_deception[0]: {texts_deception[0]}")
                print(f"  text_suppression[0]: {texts_suppression[0]}")
                print(f"{'='*80}")
                debug_done = True

            # 前向传播
            image_features, text_features_deception, text_features_suppression = self.model(
                stft_images, text_tokens_deception, text_tokens_suppression
            )

            # 计算损失
            loss, loss_info = self.loss_fn(
                image_features,
                text_features_deception,
                text_features_suppression,
                labels_deception,
                labels_suppression
            )

            total_loss += loss.item() * batch_size

            # 计算准确率
            logit_scale = self.model.model.logit_scale.exp()
            targets = torch.arange(batch_size, device=self.device)

            logits_deception = logit_scale * (image_features @ text_features_deception.t())
            pred_deception = logits_deception.argmax(dim=1)
            total_correct_deception += (pred_deception == targets).sum().item()

            logits_suppression = logit_scale * (image_features @ text_features_suppression.t())
            pred_suppression = logits_suppression.argmax(dim=1)
            total_correct_suppression += (pred_suppression == targets).sum().item()

            total_samples += batch_size

            val_bar.set_postfix(loss=loss.item())

        return {
            "loss": total_loss / len(self.val_loader.dataset),
            "accuracy_deception": total_correct_deception / total_samples,
            "accuracy_suppression": total_correct_suppression / total_samples
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

        latest_path = self.save_dir / "dual_branch_latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = self.save_dir / "dual_branch_best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  Saved best model with loss: {metrics['loss']:.4f}")

    def fit(self, num_epochs: int, debug: bool = False) -> dict:
        print(f"\n{'='*60}")
        print(f"Starting Dual-Branch CZSL Training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        if self.use_wandb:
            wandb_config = self.config.get("logging", {}).get("wandb", {})
            wandb.init(
                project=wandb_config.get("project", "CLIP-CZSL-Jamming"),
                entity=wandb_config.get("entity", None),
                name=f"DualBranch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=wandb_config.get("tags", []) + ["DualBranch", "CZSL"],
                notes="Dual-branch training for deception/suppression jamming classification",
                config=self.config
            )
            wandb.watch(self.model, log="all", log_freq=100)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            train_metrics = self.train_epoch(debug=debug and epoch == 0)
            self.scheduler.step()

            val_metrics = self.validate(debug=debug and epoch == 0)

            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Deception Acc: {train_metrics['accuracy_deception']:.4f}, "
                  f"Suppression Acc: {train_metrics['accuracy_suppression']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Deception Acc: {val_metrics['accuracy_deception']:.4f}, "
                  f"Suppression Acc: {val_metrics['accuracy_suppression']:.4f}")

            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "lr": self.scheduler.get_last_lr()[0],
                    "train_loss": train_metrics["loss"],
                    "train_acc_deception": train_metrics["accuracy_deception"],
                    "train_acc_suppression": train_metrics["accuracy_suppression"],
                    "val_loss": val_metrics["loss"],
                    "val_acc_deception": val_metrics["accuracy_deception"],
                    "val_acc_suppression": val_metrics["accuracy_suppression"]
                })

            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]

            save_best_only = self.checkpoint_config.get("save_best_only", False)
            if not save_best_only or is_best:
                self.save_checkpoint(val_metrics, is_best)

        if self.use_wandb:
            wandb.finish()

        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        return {"best_val_loss": self.best_val_loss}


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: dict,
    num_training_steps: int
) -> tuple:
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
    parser = argparse.ArgumentParser(description="Dual-Branch CZSL Training")
    parser.add_argument("--config", type=str, default="multi/config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true", help="Print debug info for first batch")
    parser.add_argument("--use-multishape", action="store_true", help="Use Multi-Shape Patch ViT instead of CLIP")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建数据加载器
    print("\nLoading Dual-Branch datasets...")
    train_loader, val_loader, test_loader, num_deception, num_suppression = create_dual_branch_dataloaders(
        config=config,
        batch_size=config.get("train", {}).get("batch_size", 16),
        num_workers=config.get("data", {}).get("num_workers", 4),
        pin_memory=config.get("data", {}).get("pin_memory", True),
        load_test=False,
    )

    # 选择模型类型
    model_config = config.get("model", {})
    use_multishape = args.use_multishape or model_config.get("use_multishape_vit", False)

    if use_multishape:
        print("\nCreating Multi-Shape Patch ViT Dual-Branch model...")
        model = create_multi_shape_dual_branch_model(config, device=str(device))
    else:
        print("\nCreating CLIP Dual-Branch model...")
        model = create_dual_branch_model(config, device=str(device))

    # 缓存文本特征
    model.cache_text_features_dual()

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Resumed from {args.resume}")

    # 创建优化器和调度器
    num_training_steps = len(train_loader) * config.get("train", {}).get("epochs", 20)
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, num_training_steps)

    # 确定是否使用 WandB
    log_type = config.get("logging", {}).get("type", "console")
    use_wandb = (log_type == "wandb") and HAS_WANDB

    # 创建训练器
    trainer = DualBranchTrainer(
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
