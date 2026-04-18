"""
CZSL 训练脚本 - 使用标准 InfoNCE 对比损失
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi.model import create_czsl_model, CLIPForCZSL
from multi.data import create_czsl_dataloaders
from multi.loss import create_loss_function, LabelAwareInfoNCELoss


class CZSLTrainer:
    """CZSL 训练器 - 使用标准 InfoNCE 对比损失"""

    def __init__(
        self,
        model: CLIPForCZSL,
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
        self.loss_fn = create_loss_function(config)
        self.use_label_aware_loss = isinstance(self.loss_fn, LabelAwareInfoNCELoss)
        print(f"Using loss function: {type(self.loss_fn).__name__}")
        if self.use_label_aware_loss:
            print(f"  handle_zero_sum: {self.loss_fn.handle_zero_sum}")

    def train_epoch(self, debug: bool = False) -> dict:
        """训练一个 epoch - 使用标准 InfoNCE"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        train_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")

        debug_done = False
        # 数据解包：现在 collate_fn 返回 (stft_images, time_signals, text_tokens, labels, texts, metas)
        for batch_idx, batch_data in enumerate(train_bar):
            if len(batch_data) == 6:
                stft_images, time_signals, text_tokens, labels, texts, metas = batch_data
                has_time_signal = True
            else:
                stft_images, text_tokens, labels, texts, metas = batch_data
                time_signals = None
                has_time_signal = False

            stft_images = stft_images.to(self.device)
            text_tokens = text_tokens.to(self.device)
            labels = labels.to(self.device)
            if has_time_signal and time_signals is not None:
                time_signals = time_signals.to(self.device)

            # 调整图像尺寸
            if stft_images.shape[-1] != 224:
                stft_images = nn.functional.interpolate(stft_images, size=(224, 224), mode='bilinear', align_corners=False)

            batch_size = stft_images.size(0)

            # Debug: 打印第一批次的详细信息
            if debug and not debug_done and batch_idx == 0:
                print(f"\n{'='*80}")
                print(f"[DEBUG Train] Batch {batch_idx} — batch_size={batch_size}")
                print(f"  labels[0]: {labels[0].tolist()}")
                print(f"  texts[0]: {texts[0]}")
                if metas:
                    print(f"  metas[0]: {metas[0]}")
                unique_texts = list(dict.fromkeys(texts))
                print(f"  unique texts in batch: {len(unique_texts)} / {batch_size}")
                for t in unique_texts[:8]:
                    print(f"    -> {t}")
                if len(unique_texts) > 8:
                    print(f"    ... ({len(unique_texts) - 8} more)")
                print(f"{'='*80}")
                debug_done = True

            self.optimizer.zero_grad()

            # 对比学习前向传播 (传入时域信号)
            image_features, text_features = self.model(stft_images, text_tokens, time_signals)

            # 计算损失
            if self.use_label_aware_loss:
                # 使用标签感知损失函数
                loss, logits_per_image, loss_info = self.loss_fn(
                    image_features, text_features, labels
                )
                logits_per_text = logits_per_image.T
            else:
                # 标准 InfoNCE 损失：对角线为正样本对
                logit_scale = self.model.model.logit_scale.exp()
                logits_per_image = logit_scale * (image_features @ text_features.t())
                logits_per_text = logits_per_image.t()

                targets = torch.arange(batch_size, device=self.device)
                loss_i2t = F.cross_entropy(logits_per_image, targets)
                loss_t2i = F.cross_entropy(logits_per_text, targets)
                loss = (loss_i2t + loss_t2i) / 2

            loss.backward()

            # 梯度裁剪
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            total_loss += loss.item() * batch_size

            # 计算对比学习准确率（对角线准确率作为参考指标）
            with torch.no_grad():
                # 对于 label-aware 模式，对角线准确率仅供参考
                # 实际评估应使用 zero-shot 或 KNN
                targets = torch.arange(batch_size, device=self.device)
                pred_i2t = logits_per_image.argmax(dim=1)
                pred_t2i = logits_per_text.argmax(dim=1)
                total_correct += (pred_i2t == targets).sum().item()
                total_correct += (pred_t2i == targets).sum().item()
                total_samples += batch_size * 2

            train_bar.set_postfix(loss=loss.item())

        return {
            "loss": total_loss / len(self.train_loader.dataset),
            "accuracy": total_correct / total_samples
        }

    @torch.no_grad()
    def validate(self, debug: bool = False) -> dict:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        val_bar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")

        debug_done = False
        for batch_idx, batch_data in enumerate(val_bar):
            if len(batch_data) == 6:
                stft_images, time_signals, text_tokens, labels, texts, metas = batch_data
                has_time_signal = True
            else:
                stft_images, text_tokens, labels, texts, metas = batch_data
                time_signals = None
                has_time_signal = False

            stft_images = stft_images.to(self.device)
            text_tokens = text_tokens.to(self.device)
            if has_time_signal and time_signals is not None:
                time_signals = time_signals.to(self.device)

            if stft_images.shape[-1] != 224:
                stft_images = nn.functional.interpolate(stft_images, size=(224, 224), mode='bilinear', align_corners=False)

            batch_size = stft_images.size(0)

            # Debug: 打印第一批次的详细信息
            if debug and not debug_done:
                print(f"\n{'='*80}")
                print(f"[DEBUG] Batch {batch_idx} — batch_size={batch_size}")
                print(f"  labels[0]: {labels[0].tolist()}")
                print(f"  texts[0]: {texts[0]}")
                if metas:
                    print(f"  metas[0]: {metas[0]}")
                # 统计 unique texts
                unique_texts = list(dict.fromkeys(texts))
                print(f"  unique texts in batch: {len(unique_texts)} / {batch_size}")
                if len(unique_texts) <= 10:
                    for t in unique_texts:
                        print(f"    -> {t}")
                else:
                    for t in unique_texts[:5]:
                        print(f"    -> {t}")
                    print(f"    ... ({len(unique_texts) - 5} more unique texts)")
                print(f"{'='*80}")
                debug_done = True

            image_features, text_features = self.model(stft_images, text_tokens, time_signals)

            # 计算损失
            if self.use_label_aware_loss:
                # 使用标签感知损失函数
                loss, logits_per_image, loss_info = self.loss_fn(
                    image_features, text_features, labels
                )
                logits_per_text = logits_per_image.T
            else:
                # 标准 InfoNCE 损失
                logit_scale = self.model.model.logit_scale.exp()
                logits_per_image = logit_scale * (image_features @ text_features.t())
                logits_per_text = logits_per_image.t()

                targets = torch.arange(batch_size, device=self.device)
                loss_i2t = F.cross_entropy(logits_per_image, targets)
                loss_t2i = F.cross_entropy(logits_per_text, targets)
                loss = (loss_i2t + loss_t2i) / 2

            total_loss += loss.item() * batch_size

            # 计算对角线准确率（仅供参考）
            targets = torch.arange(batch_size, device=self.device)
            pred_i2t = logits_per_image.argmax(dim=1)
            pred_t2i = logits_per_text.argmax(dim=1)
            total_correct += (pred_i2t == targets).sum().item()
            total_correct += (pred_t2i == targets).sum().item()
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

        latest_path = self.save_dir / "czsl_latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = self.save_dir / "czsl_best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  Saved best model with loss: {metrics['loss']:.4f}")

    def fit(self, num_epochs: int, debug: bool = False) -> dict:
        print(f"\n{'='*60}")
        print(f"Starting CZSL Training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        if self.use_wandb:
            wandb_config = self.config.get("logging", {}).get("wandb", {})
            wandb.init(
                project=wandb_config.get("project", "CLIP-CZSL-Jamming"),
                entity=wandb_config.get("entity", None),
                name=f"CZSL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=wandb_config.get("tags", []) + ["CZSL", "InfoNCE"],
                notes="CZSL training with standard InfoNCE loss",
                config=self.config
            )
            wandb.watch(self.model, log="all", log_freq=100)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            train_metrics = self.train_epoch(debug=debug and epoch == 0)
            self.scheduler.step()

            val_metrics = self.validate(debug=debug and epoch == 0)

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
            print(f"Scheduler: warmup={actual_warmup} epochs, cosine={t_max} epochs")
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=scheduler_config.get("min_lr", 1e-7))
            print(f"Scheduler: cosine only, T_max={num_epochs}")
    else:
        scheduler = None

    return optimizer, scheduler


def main():
    parser = argparse.ArgumentParser(description="CZSL Training with CLIP")
    parser.add_argument("--config", type=str, default="multi/config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true", help="Print debug info for first batch of train/val")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建数据加载器
    print("\nLoading CZSL datasets...")
    train_loader, val_loader, test_loader, num_classes = create_czsl_dataloaders(
        config=config,
        batch_size=config.get("train", {}).get("batch_size", 16),
        num_workers=config.get("data", {}).get("num_workers", 4),
        pin_memory=config.get("data", {}).get("pin_memory", True),
    )

    # 创建模型
    print("\nCreating CZSL model...")
    model = create_czsl_model(config, device=str(device))

    # 缓存文本特征（使用配置文件中的 seen_combinations）
    czsl_config = config.get("czsl", {})
    seen_combos = czsl_config.get("seen_combinations", None)
    model.cache_text_features(max_combination_size=2, include_single=True, seen_combinations=seen_combos)

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
    trainer = CZSLTrainer(
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
