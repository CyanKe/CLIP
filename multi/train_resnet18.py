"""
ResNet18 双分支分类训练脚本 - 用于消融实验对比
python -m multi.train_resnet18 --config multi/config.yaml
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi.model import create_resnet18_dual_branch_model, DualBranchResNet18
from multi.data import create_dual_branch_dataloaders


class ResNet18Trainer:
    """ResNet18 双分支训练器"""

    def __init__(
        self,
        model: DualBranchResNet18,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device: torch.device,
        config: dict
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config

        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_config = config.get("train", {})
        self.grad_clip = self.train_config.get("grad_clip", 1.0)

        self.checkpoint_config = config.get("checkpoint", {})
        self.save_dir = Path(self.checkpoint_config.get("save_dir", "checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 损失函数
        self.criterion_deception = nn.CrossEntropyLoss()
        self.criterion_suppression = nn.CrossEntropyLoss()

        print(f"Using CrossEntropyLoss for both branches")

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

            # 将多热标签转换为类别索引
            labels_deception_idx = torch.argmax(labels_deception, dim=1).to(self.device)
            labels_suppression_idx = torch.argmax(labels_suppression, dim=1).to(self.device)

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
                print(f"  labels_deception_idx[0]: {labels_deception_idx[0].item()}")
                print(f"  labels_suppression_idx[0]: {labels_suppression_idx[0].item()}")
                print(f"{'='*80}")
                debug_done = True

            self.optimizer.zero_grad()

            # 前向传播
            logits_deception, logits_suppression = self.model(stft_images)

            # 计算损失
            loss_deception = self.criterion_deception(logits_deception, labels_deception_idx)
            loss_suppression = self.criterion_suppression(logits_suppression, labels_suppression_idx)
            loss = loss_deception + loss_suppression

            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            total_loss += loss.item() * batch_size

            # 计算准确率
            pred_deception = torch.argmax(logits_deception, dim=1)
            pred_suppression = torch.argmax(logits_suppression, dim=1)

            total_correct_deception += (pred_deception == labels_deception_idx).sum().item()
            total_correct_suppression += (pred_suppression == labels_suppression_idx).sum().item()
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

        for batch_data in val_bar:
            (stft_images, text_tokens_deception, text_tokens_suppression,
             labels_deception, labels_suppression, texts_deception, texts_suppression,
             metadata_list) = batch_data

            stft_images = stft_images.to(self.device)
            labels_deception_idx = torch.argmax(labels_deception, dim=1).to(self.device)
            labels_suppression_idx = torch.argmax(labels_suppression, dim=1).to(self.device)

            if stft_images.shape[-1] != 224:
                stft_images = nn.functional.interpolate(
                    stft_images, size=(224, 224), mode='bilinear', align_corners=False
                )

            batch_size = stft_images.size(0)

            # 前向传播
            logits_deception, logits_suppression = self.model(stft_images)

            # 计算损失
            loss_deception = self.criterion_deception(logits_deception, labels_deception_idx)
            loss_suppression = self.criterion_suppression(logits_suppression, labels_suppression_idx)
            loss = loss_deception + loss_suppression

            total_loss += loss.item() * batch_size

            # 计算准确率
            pred_deception = torch.argmax(logits_deception, dim=1)
            pred_suppression = torch.argmax(logits_suppression, dim=1)

            total_correct_deception += (pred_deception == labels_deception_idx).sum().item()
            total_correct_suppression += (pred_suppression == labels_suppression_idx).sum().item()
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

        latest_path = self.save_dir / "resnet18_dual_branch_latest.pt"
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = self.save_dir / "resnet18_dual_branch_best.pt"
            torch.save(checkpoint, best_path)
            print(f"  ★ Saved best model with loss: {metrics['loss']:.4f}")

    def fit(self, num_epochs: int, debug: bool = False) -> dict:
        print(f"\n{'='*60}")
        print(f"Starting ResNet18 Dual-Branch Training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

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

            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]

            save_best_only = self.checkpoint_config.get("save_best_only", False)
            if not save_best_only or is_best:
                self.save_checkpoint(val_metrics, is_best)

        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        return {"best_val_loss": self.best_val_loss}


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_optimizer_and_scheduler(model, config, num_training_steps):
    train_config = config.get("train", {})

    optimizer = AdamW(
        model.parameters(),
        lr=float(train_config.get("lr", 1e-4)),
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
    parser = argparse.ArgumentParser(description="ResNet18 Dual-Branch Training")
    parser.add_argument("--config", type=str, default="multi/config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    args = parser.parse_args()

    config = load_config(args.config)

    # 覆盖参数
    if args.lr:
        config["train"]["lr"] = args.lr
    if args.epochs:
        config["train"]["epochs"] = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建数据加载器
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader, _, _ = create_dual_branch_dataloaders(
        config=config,
        batch_size=config.get("train", {}).get("batch_size", 32),
        num_workers=config.get("data", {}).get("num_workers", 4),
        load_test=False,
    )

    # 创建模型
    print("\nCreating ResNet18 dual-branch model...")
    model = create_resnet18_dual_branch_model(config, device=str(device))

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Resumed from {args.resume}")

    # 创建优化器和调度器
    num_training_steps = len(train_loader) * config.get("train", {}).get("epochs", 20)
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, num_training_steps)

    # 创建训练器
    trainer = ResNet18Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )

    # 开始训练
    num_epochs = config.get("train", {}).get("epochs", 20)
    trainer.fit(num_epochs, debug=args.debug)


if __name__ == "__main__":
    main()
