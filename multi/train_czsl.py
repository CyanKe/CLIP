"""
CZSL训练脚本 - 使用对比学习训练CLIP用于组合零样本学习
支持metadata-based文本描述
python -m multi.train_czsl --config multi/config.yaml
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

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi.model import create_czsl_model, CLIPForCZSL
from multi.loss import InfoNCELoss, CZSLContrastiveLoss
from multi.data import create_czsl_dataloaders_with_metadata, czsl_metadata_collate_fn


class CZSLTrainer:
    """
    CZSL训练器
    使用对比学习训练图像-文本对齐
    """

    def __init__(
        self,
        model: CLIPForCZSL,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device: torch.device,
        config: dict,
        use_wandb: bool = True
    ):
        """
        初始化训练器

        Args:
            model: CZSL模型
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
        self.best_val_loss = float('inf')
        self.train_config = config.get("train", {})

        # 梯度裁剪
        self.grad_clip = self.train_config.get("grad_clip", 1.0)

        # 检查点保存
        self.checkpoint_config = config.get("checkpoint", {})
        self.save_dir = Path(self.checkpoint_config.get("save_dir", "checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self) -> dict:
        """
        训练一个epoch
        使用样本级文本描述进行对比学习（InfoNCE风格）

        Returns:
            训练指标字典
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0  # 正确预测的样本数
        total_samples = 0

        train_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")

        for batch_idx, (images, text_tokens, labels, texts, metas) in enumerate(train_bar):
            images = images.to(self.device)
            text_tokens = text_tokens.to(self.device)
            labels = labels.to(self.device)

            # 调整图像尺寸
            if images.shape[-1] != 224:
                images = nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

            batch_size = images.size(0)
            self.optimizer.zero_grad()

            # 编码图像和样本级文本
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(text_tokens)

            # 归一化特征
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            # 获取温度缩放因子
            logit_scale = self.model.model.logit_scale.exp()

            # 计算样本级相似度矩阵 [batch_size, batch_size]
            logits_per_image = logit_scale * (image_features @ text_features.t())
            logits_per_text = logits_per_image.t()

            # 标准InfoNCE：对角线为正样本（图像i匹配文本i）
            targets = torch.arange(batch_size, device=self.device)

            # 双向对比损失
            loss_i2t = F.cross_entropy(logits_per_image, targets)
            loss_t2i = F.cross_entropy(logits_per_text, targets)
            loss = (loss_i2t + loss_t2i) / 2

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            # 记录
            total_loss += loss.item() * batch_size

            # 计算准确率：检查是否正确匹配到对角线
            with torch.no_grad():
                # 图像到文本的预测准确率
                pred_i2t = logits_per_image.argmax(dim=1)
                correct_i2t = (pred_i2t == targets).sum().item()
                # 文本到图像的预测准确率
                pred_t2i = logits_per_text.argmax(dim=1)
                correct_t2i = (pred_t2i == targets).sum().item()
                # 平均准确率
                total_correct += (correct_i2t + correct_t2i) / 2
                total_samples += batch_size

            train_bar.set_postfix(loss=loss.item())

        # 计算指标
        avg_loss = total_loss / len(self.train_loader.dataset)
        accuracy = total_correct / total_samples

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy
        }

        return metrics

    @torch.no_grad()
    def validate(self) -> dict:
        """
        验证 - 使用样本级文本描述进行对比学习

        Returns:
            验证指标字典
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        val_bar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")

        for images, text_tokens, labels, texts, metas in val_bar:
            images = images.to(self.device)
            text_tokens = text_tokens.to(self.device)

            # 调整图像尺寸
            if images.shape[-1] != 224:
                images = nn.functional.interpolate(
                    images, size=(224, 224), mode='bilinear', align_corners=False
                )

            batch_size = images.size(0)

            # 编码图像和样本级文本
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(text_tokens)

            # 归一化特征
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            # 获取温度缩放因子
            logit_scale = self.model.model.logit_scale.exp()

            # 计算样本级相似度矩阵
            logits_per_image = logit_scale * (image_features @ text_features.t())
            logits_per_text = logits_per_image.t()

            # InfoNCE目标：对角线为正样本
            targets = torch.arange(batch_size, device=self.device)

            # 双向对比损失
            loss_i2t = F.cross_entropy(logits_per_image, targets)
            loss_t2i = F.cross_entropy(logits_per_text, targets)
            loss = (loss_i2t + loss_t2i) / 2

            total_loss += loss.item() * batch_size

            # 计算准确率
            pred_i2t = logits_per_image.argmax(dim=1)
            correct_i2t = (pred_i2t == targets).sum().item()
            pred_t2i = logits_per_text.argmax(dim=1)
            correct_t2i = (pred_t2i == targets).sum().item()
            total_correct += (correct_i2t + correct_t2i) / 2
            total_samples += batch_size

            val_bar.set_postfix(loss=loss.item())

        # 计算指标
        avg_loss = total_loss / len(self.val_loader.dataset)
        accuracy = total_correct / total_samples

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy
        }

        return metrics

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
        latest_path = self.save_dir / "czsl_latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)

        # 保存最佳检查点
        if is_best:
            best_path = self.save_dir / "czsl_best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  Saved best model with loss: {metrics['loss']:.4f}")

    def fit(self, num_epochs: int) -> dict:
        """
        完整训练流程

        Args:
            num_epochs: 训练轮数

        Returns:
            最佳验证指标
        """
        print(f"\n{'='*60}")
        print(f"Starting CZSL Training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        # 初始化WandB
        if self.use_wandb:
            wandb_config = self.config.get("logging", {}).get("wandb", {})
            wandb.init(
                project=wandb_config.get("project", "CLIP-CZSL-Jamming"),
                entity=wandb_config.get("entity", None),
                name=f"CZSL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=wandb_config.get("tags", []) + ["CZSL", "Contrastive"],
                notes="CZSL training with contrastive learning",
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
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

            # 记录到WandB
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "lr": self.scheduler.get_last_lr()[0],
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["accuracy"]
                })

            # 保存检查点
            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]

            save_best_only = self.checkpoint_config.get("save_best_only", False)
            if not save_best_only or is_best:
                self.save_checkpoint(val_metrics, is_best)

        # 结束WandB
        if self.use_wandb:
            wandb.finish()

        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        return {"best_val_loss": self.best_val_loss}


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


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

    # 优化器
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
        # 确保 epochs > warmup_epochs，否则调整 warmup
        actual_warmup = min(warmup_epochs, num_epochs - 1)
        t_max = max(1, num_epochs - actual_warmup)

        if actual_warmup > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=actual_warmup
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=scheduler_config.get("min_lr", 1e-7)
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[actual_warmup]
            )
            print(f"Scheduler: warmup={actual_warmup} epochs, cosine={t_max} epochs")
        else:
            # 没有 warmup，直接用 cosine
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=scheduler_config.get("min_lr", 1e-7)
            )
            print(f"Scheduler: cosine only, T_max={num_epochs}")
    else:
        scheduler = None

    return optimizer, scheduler


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CZSL Training with CLIP")
    parser.add_argument("--config", type=str, default="multi/config.yaml",
                        help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建数据加载器（使用metadata版本）
    print("\nLoading CZSL datasets with metadata...")
    train_loader, val_loader, test_loader, num_classes = create_czsl_dataloaders_with_metadata(config)

    # 创建模型
    print("\nCreating CZSL model...")
    model = create_czsl_model(config, device=str(device))

    # 缓存文本特征（用于零样本推理）
    model.cache_text_features(max_combination_size=2, include_single=True)

    # 如果恢复训练
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Resumed from {args.resume}")

    # 创建损失函数
    loss_config = config.get("loss", {})
    criterion = CZSLContrastiveLoss(
        temperature=loss_config.get("temperature", 0.07),
        contrastive_weight=loss_config.get("contrastive_weight", 1.0),
        classification_weight=loss_config.get("classification_weight", 0.0),
        learnable_temperature=loss_config.get("learnable_temperature", True)
    )

    # 创建优化器和调度器
    num_training_steps = len(train_loader) * config.get("train", {}).get("epochs", 20)
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, num_training_steps)

    # 确定是否使用WandB
    log_type = config.get("logging", {}).get("type", "console")
    use_wandb = (log_type == "wandb") and HAS_WANDB

    # 创建训练器
    trainer = CZSLTrainer(
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