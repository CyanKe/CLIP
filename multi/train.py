"""
训练脚本 - CLIP微调用于组合零样本学习
支持WandB日志记录
python -m czsl.train --config czsl/config.yaml
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
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Using console logging only.")

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from czsl.model import create_clip_model, CLIPForMultiLabel
from czsl.loss import create_loss_function, MultiLabelContrastiveLoss, AsymmetricLoss
from czsl.data import STFTDataset3, load_multi_jnr_dataset


class Trainer:
    """
    训练器类
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        config: dict,
        use_wandb: bool = True
    ):
        """
        初始化训练器

        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 计算设备
            config: 配置字典
            use_wandb: 是否使用WandB
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.use_wandb = use_wandb and HAS_WANDB

        # 训练状态
        self.current_epoch = 0
        self.best_val_f1 = 0.0
        self.train_config = config.get("train", {})
        self.threshold = self.train_config.get("threshold", 0.5)

        # 混合精度 - CLIP模型内部已使用FP16，需要特殊处理
        self.use_amp = self.train_config.get("use_amp", True) and device.type == "cuda"

        # CLIP模型使用FP16，不需要额外的scaler
        # 检查模型是否已经是半精度
        self.model_is_fp16 = next(model.parameters()).dtype == torch.float16
        if self.model_is_fp16:
            self.use_amp = False  # CLIP已经是FP16，不需要autocast

        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # 梯度裁剪
        self.grad_clip = self.train_config.get("grad_clip", 1.0)

        # 检查点保存
        self.checkpoint_config = config.get("checkpoint", {})
        self.save_dir = Path(self.checkpoint_config.get("save_dir", "checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self) -> dict:
        """
        训练一个epoch

        Returns:
            训练指标字典
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        train_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")

        for batch_idx, (images, labels) in enumerate(train_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 调整图像尺寸到224x224（CLIP要求）
            if images.shape[-1] != 224:
                images = nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

            self.optimizer.zero_grad()

            # 混合精度前向传播
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    # 获取图像特征
                    image_features = self.model.encode_image(images)
                    image_features = F.normalize(image_features, dim=-1)  # 归一化
                    # 分类输出
                    logits = self.model.classifier(image_features)
                    # 计算损失
                    loss = self.criterion(logits, labels)

                # 反向传播
                self.scaler.scale(loss).backward()

                # 梯度裁剪
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 标准前向传播
                image_features = self.model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)  # 归一化
                logits = self.model.classifier(image_features)
                loss = self.criterion(logits, labels)

                loss.backward()

                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()

            # 记录
            total_loss += loss.item() * images.size(0)

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > self.threshold).float()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

            train_bar.set_postfix(loss=loss.item())

        # 计算指标
        avg_loss = total_loss / len(self.train_loader.dataset)
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # 计算subset_accuracy和F1_sub
        subset_acc = (all_labels == all_preds).all(axis=1).mean()
        f1_sub = self._compute_f1_sub(all_labels, all_preds)

        metrics = {
            "loss": avg_loss,
            "f1_macro": f1_score(all_labels, all_preds, average='macro', zero_division=0),
            "f1_micro": f1_score(all_labels, all_preds, average='micro', zero_division=0),
            "precision": precision_score(all_labels, all_preds, average='macro', zero_division=0),
            "recall": recall_score(all_labels, all_preds, average='macro', zero_division=0),
            "subset_accuracy": subset_acc,
            "f1_sub": f1_sub
        }

        return metrics

    @torch.no_grad()
    def validate(self) -> dict:
        """
        验证

        Returns:
            验证指标字典
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        val_bar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")

        for images, labels in val_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 调整图像尺寸
            if images.shape[-1] != 224:
                images = nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

            # 前向传播
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    image_features = self.model.encode_image(images)
                    image_features = F.normalize(image_features, dim=-1)  # 归一化
                    logits = self.model.classifier(image_features)
                    loss = self.criterion(logits, labels)
            else:
                image_features = self.model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)  # 归一化
                logits = self.model.classifier(image_features)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs > self.threshold).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            val_bar.set_postfix(loss=loss.item())

        # 计算指标
        avg_loss = total_loss / len(self.val_loader.dataset)
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # 计算subset_accuracy和F1_sub
        subset_acc = (all_labels == all_preds).all(axis=1).mean()
        f1_sub = self._compute_f1_sub(all_labels, all_preds)

        metrics = {
            "loss": avg_loss,
            "f1_macro": f1_score(all_labels, all_preds, average='macro', zero_division=0),
            "f1_micro": f1_score(all_labels, all_preds, average='micro', zero_division=0),
            "precision": precision_score(all_labels, all_preds, average='macro', zero_division=0),
            "recall": recall_score(all_labels, all_preds, average='macro', zero_division=0),
            "subset_accuracy": subset_acc,
            "f1_sub": f1_sub
        }

        return metrics

    def _compute_f1_sub(self, labels: np.ndarray, preds: np.ndarray) -> float:
        """
        计算基于标签集合的F1值

        Args:
            labels: 真实标签 [N, num_classes]
            preds: 预测标签 [N, num_classes]

        Returns:
            F1_sub值
        """
        tp_total = 0.0
        fn_total = 0.0
        fp_total = 0.0

        for i in range(len(labels)):
            true_set = set(np.where(labels[i] == 1)[0])
            pred_set = set(np.where(preds[i] == 1)[0])
            union_set = true_set | pred_set
            K = len(union_set)

            if K == 0:
                continue

            tp_i = len(true_set & pred_set)
            fn_i = len(true_set - pred_set)
            fp_i = len(pred_set - true_set)

            tp_total += tp_i / K
            fn_total += fn_i / K
            fp_total += fp_i / K

        precision_sub = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        recall_sub = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1_sub = 2 * precision_sub * recall_sub / (precision_sub + recall_sub) if (precision_sub + recall_sub) > 0 else 0.0

        return f1_sub

    def save_checkpoint(self, metrics: dict, is_best: bool = False):
        """
        保存检查点

        Args:
            metrics: 当前指标
            is_best: 是否是最佳模型
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "metrics": metrics,
            "config": self.config
        }

        # 保存最新检查点
        latest_path = self.save_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)

        # 保存最佳检查点
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  Saved best model with F1: {metrics['f1_macro']:.4f}")

    def fit(self, num_epochs: int) -> dict:
        """
        完整训练流程

        Args:
            num_epochs: 训练轮数

        Returns:
            最佳验证指标
        """
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"{'='*60}\n")

        # 初始化WandB
        if self.use_wandb:
            wandb_config = self.config.get("logging", {}).get("wandb", {})
            wandb.init(
                project=wandb_config.get("project", "CLIP-CZSL-Jamming"),
                entity=wandb_config.get("entity", None),
                name=self.config.get("logging", {}).get("name", f"CLIP_CZSL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                tags=wandb_config.get("tags", []),
                notes=wandb_config.get("notes", ""),
                config=self.config
            )
            wandb.watch(self.model, log="all", log_freq=100)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # 训练
            train_metrics = self.train_epoch()
            self.scheduler.step()

            # 验证
            val_metrics = self.validate()

            # 打印
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"F1: {train_metrics['f1_macro']:.4f}, "
                  f"Acc: {train_metrics['subset_accuracy']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"F1: {val_metrics['f1_macro']:.4f}, "
                  f"Acc: {val_metrics['subset_accuracy']:.4f}")

            # 记录到WandB
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "lr": self.scheduler.get_last_lr()[0],
                    "train_loss": train_metrics["loss"],
                    "train_f1": train_metrics["f1_macro"],
                    "train_acc": train_metrics["subset_accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_f1": val_metrics["f1_macro"],
                    "val_acc": val_metrics["subset_accuracy"]
                })

            # 保存检查点
            is_best = val_metrics["f1_macro"] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics["f1_macro"]

            save_best_only = self.checkpoint_config.get("save_best_only", False)
            if not save_best_only or is_best:
                self.save_checkpoint(val_metrics, is_best)

        # 结束WandB
        if self.use_wandb:
            wandb.finish()

        print(f"\nTraining completed!")
        print(f"Best validation F1: {self.best_val_f1:.4f}")

        return {"best_val_f1": self.best_val_f1}


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: dict) -> tuple:
    """
    创建数据加载器

    Args:
        config: 配置字典

    Returns:
        (train_loader, test_loader, num_classes)
    """
    data_config = config.get("data", {})
    train_config = config.get("train", {})

    # 加载数据集
    train_dataset, test_dataset, val_dataset, num_classes, jnr_levels = load_multi_jnr_dataset(
        base_path=data_config.get("base_path"),
        jnr_start=data_config.get("jnr_start", 0),
        jnr_end=data_config.get("jnr_end", 40),
        jnr_step=data_config.get("jnr_step", 10)
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get("batch_size", 32),
        shuffle=True,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True)
    )

    # 使用test_dataset作为验证集（因为val_dataset太大）
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config.get("batch_size", 32),
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True)
    )

    print(f"Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test (val) samples")
    print(f"Number of classes: {num_classes}")
    print(f"JNR levels: {jnr_levels}")

    return train_loader, test_loader, num_classes


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: dict,
    num_training_steps: int
) -> tuple:
    """
    创建优化器和学习率调度器

    Args:
        model: 模型
        config: 配置字典
        num_training_steps: 总训练步数

    Returns:
        (optimizer, scheduler)
    """
    train_config = config.get("train", {})

    # 优化器（确保参数类型正确）
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_config.get("lr", 1e-5)),
        weight_decay=float(train_config.get("weight_decay", 0.01))
    )

    # 学习率调度器
    scheduler_config = train_config.get("scheduler", {})
    scheduler_type = scheduler_config.get("type", "cosine")
    warmup_epochs = train_config.get("warmup_epochs", 2)
    num_epochs = train_config.get("epochs", 20)

    if scheduler_type == "cosine":
        # 预热 + 余弦退火
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=scheduler_config.get("min_lr", 1e-7)
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
    else:
        scheduler = None

    return optimizer, scheduler


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CLIP Fine-tuning for CZSL")
    parser.add_argument("--config", type=str, default="czsl/config.yaml",
                        help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建数据加载器
    train_loader, val_loader, num_classes = create_dataloaders(config)

    # 创建模型
    model = create_clip_model(config, device=str(device))

    # 如果恢复训练
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Resumed from {args.resume}")

    # 创建损失函数
    criterion = create_loss_function(config)

    # 创建优化器和调度器
    num_training_steps = len(train_loader) * config.get("train", {}).get("epochs", 20)
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, num_training_steps)

    # 确定是否使用WandB
    log_type = config.get("logging", {}).get("type", "console")
    use_wandb = (log_type == "wandb") and HAS_WANDB

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        use_wandb=use_wandb
    )

    # 开始训练
    num_epochs = config.get("train", {}).get("epochs", 20)
    trainer.fit(num_epochs)


if __name__ == "__main__":
    main()