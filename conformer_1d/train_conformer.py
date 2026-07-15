"""
Unified training script for 1D CZSL models (Conformer / ResNet1D / CNN1D).

Select backbone via config:  model.backbone = "conformer" | "resnet1d" | "cnn1d"

Usage:
    python conformer_1d/train_conformer.py --config conformer_1d/config_1d.yaml
    python conformer_1d/train_conformer.py --config conformer_1d/config_1d.yaml --resume checkpoints/resnet1d/resnet1d_latest.pt
    python conformer_1d/train_conformer.py --config conformer_1d/config_1d.yaml --debug
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

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _parent)

from conformer_1d.model_1d import Base1DCZSLModel, create_1d_model
from conformer_1d.data_1d import create_1d_dataloaders
from multi.loss import create_loss_function, LabelAwareInfoNCELoss, MultiLabelSigmoidLoss


# ============================================================================
# Backbone strategies — lightweight model-selection helpers
# ============================================================================

BACKBONE_CONFIG = {
    "conformer": {
        "checkpoint_prefix": "conformer",
        "wandb_tag": "Conformer",
        "display_name": "Conformer 1D",
    },
    "resnet1d": {
        "checkpoint_prefix": "resnet1d",
        "wandb_tag": "ResNet1D",
        "display_name": "ResNet-18 1D",
    },
    "cnn1d": {
        "checkpoint_prefix": "cnn1d",
        "wandb_tag": "CNN1D",
        "display_name": "CNN 1D",
    },
}


def get_backbone_info(config: dict) -> dict:
    """Return the backbone config dict for the selected backbone."""
    backbone = config.get("model", {}).get("backbone", "conformer")
    if backbone not in BACKBONE_CONFIG:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Choose from: {list(BACKBONE_CONFIG.keys())}"
        )
    return BACKBONE_CONFIG[backbone]


# ---------------------------------------------------------------------------
# Trainer (model-agnostic)
# ---------------------------------------------------------------------------

class Trainer1D:
    """Unified trainer for 1D signal encoder + CLIP text encoder CZSL models.

    Works with any Base1DCZSLModel subclass — Conformer, ResNet1D, or CNN1D.
    """

    def __init__(
        self,
        model: Base1DCZSLModel,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device: torch.device,
        config: dict,
        use_wandb: bool = True,
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
        self.use_amp = self.train_config.get("use_amp", False)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # Checkpoint dir based on backbone
        backbone_info = get_backbone_info(config)
        self.checkpoint_prefix = backbone_info["checkpoint_prefix"]
        self.display_name = backbone_info["display_name"]
        self.wandb_tags = backbone_info.get("wandb_tags", [])
        if isinstance(self.wandb_tags, str):
            self.wandb_tags = [self.wandb_tags]

        self.checkpoint_config = config.get("checkpoint", {})
        self.save_dir = Path(self.checkpoint_config.get("save_dir", "checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Loss function (reused from multi/loss.py)
        self.loss_fn = create_loss_function(config, model_type="ViT-B/32")
        self.use_label_aware_loss = isinstance(self.loss_fn, LabelAwareInfoNCELoss)
        self.use_sigmoid_loss = isinstance(self.loss_fn, MultiLabelSigmoidLoss)

        print(f"Loss function: {type(self.loss_fn).__name__}")

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_epoch(self, debug: bool = False) -> dict:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        train_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        debug_done = False

        # Data: (time_signals, None, text_tokens, labels, texts, metas [, features])
        for batch_idx, batch_data in enumerate(train_bar):
            if len(batch_data) >= 7:
                time_signals, _, text_tokens, labels, texts, metas, *rest = batch_data
                features_dict = rest[0] if rest else None
            elif len(batch_data) == 6:
                time_signals, _, text_tokens, labels, texts, metas = batch_data
                features_dict = None
            else:
                time_signals, text_tokens, labels, texts, metas = batch_data
                features_dict = None

            time_signals = time_signals.to(self.device)
            text_tokens = text_tokens.to(self.device)
            labels = labels.to(self.device)

            if debug and not debug_done:
                print(f"\n{'='*80}")
                print(f"[DEBUG Train] Batch {batch_idx} — batch_size={time_signals.size(0)}")
                print(f"  time_signals shape: {time_signals.shape}")
                print(f"  labels[0]: {labels[0].tolist()}")
                print(f"  texts[0]: {texts[0]}")
                unique_texts = list(dict.fromkeys(texts))
                print(f"  unique texts: {len(unique_texts)} / {time_signals.size(0)}")
                for t in unique_texts[:5]:
                    print(f"    -> {t}")
                print(f"{'='*80}")
                debug_done = True

            self.optimizer.zero_grad()

            # Forward
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                signal_features, text_features = self.model(time_signals, text_tokens, features_dict)

                if self.use_sigmoid_loss or self.use_label_aware_loss:
                    loss, logits_per_image, loss_info = self.loss_fn(
                        signal_features, text_features, labels
                    )
                    logits_per_text = logits_per_image.T
                else:
                    # Standard InfoNCE
                    logit_scale = self.model.logit_scale.exp()
                    logits_per_image = logit_scale * (signal_features @ text_features.T)
                    logits_per_text = logits_per_image.T

                    targets = torch.arange(time_signals.size(0), device=self.device)
                    loss_i2t = F.cross_entropy(logits_per_image, targets)
                    loss_t2i = F.cross_entropy(logits_per_text, targets)
                    loss = (loss_i2t + loss_t2i) / 2

            self.scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = time_signals.size(0)
            total_loss += loss.item() * batch_size

            # 对角线准确率（与 multi/train_czsl.py 对齐，仅供参考）
            with torch.no_grad():
                targets = torch.arange(batch_size, device=self.device)
                pred_i2t = logits_per_image.argmax(dim=1)
                pred_t2i = logits_per_text.argmax(dim=1)
                total_correct += (pred_i2t == targets).sum().item()
                total_correct += (pred_t2i == targets).sum().item()
                total_samples += batch_size * 2

            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        return {
            "loss": total_loss / len(self.train_loader.dataset),
            "accuracy": total_correct / total_samples,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, debug: bool = False) -> dict:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        if self.val_loader is None:
            return {"loss": float('nan'), "accuracy": 0.0}

        val_bar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
        debug_done = False

        for batch_data in val_bar:
            if len(batch_data) >= 7:
                time_signals, _, text_tokens, labels, texts, metas, *rest = batch_data
                features_dict = rest[0] if rest else None
            elif len(batch_data) == 6:
                time_signals, _, text_tokens, labels, texts, metas = batch_data
                features_dict = None
            else:
                time_signals, text_tokens, labels, texts, metas = batch_data
                features_dict = None

            time_signals = time_signals.to(self.device)
            text_tokens = text_tokens.to(self.device)
            labels = labels.to(self.device)

            if debug and not debug_done:
                print(f"\n{'='*80}")
                print(f"[DEBUG Val] batch_size={time_signals.size(0)}")
                print(f"  labels[0]: {labels[0].tolist()}")
                print(f"  texts[0]: {texts[0]}")
                unique_texts = list(dict.fromkeys(texts))
                print(f"  unique texts: {len(unique_texts)} / {time_signals.size(0)}")
                print(f"{'='*80}")
                debug_done = True

            signal_features, text_features = self.model(time_signals, text_tokens, features_dict)

            if self.use_sigmoid_loss or self.use_label_aware_loss:
                loss, logits_per_image, loss_info = self.loss_fn(
                    signal_features, text_features, labels
                )
                logits_per_text = logits_per_image.T
            else:
                logit_scale = self.model.logit_scale.exp()
                logits_per_image = logit_scale * (signal_features @ text_features.T)
                logits_per_text = logits_per_image.T

                targets = torch.arange(time_signals.size(0), device=self.device)
                loss_i2t = F.cross_entropy(logits_per_image, targets)
                loss_t2i = F.cross_entropy(logits_per_text, targets)
                loss = (loss_i2t + loss_t2i) / 2

            batch_size = time_signals.size(0)
            total_loss += loss.item() * batch_size

            # 对角线准确率（与 multi/train_czsl.py 对齐，仅供参考）
            targets = torch.arange(batch_size, device=self.device)
            pred_i2t = logits_per_image.argmax(dim=1)
            pred_t2i = logits_per_text.argmax(dim=1)
            total_correct += (pred_i2t == targets).sum().item()
            total_correct += (pred_t2i == targets).sum().item()
            total_samples += batch_size * 2

            val_bar.set_postfix(loss=f"{loss.item():.4f}")

        return {
            "loss": total_loss / len(self.val_loader.dataset),
            "accuracy": total_correct / total_samples,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, metrics: dict, is_best: bool = False):
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "metrics": metrics,
            "config": self.config,
        }

        latest_path = self.save_dir / f"{self.checkpoint_prefix}_latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = self.save_dir / f"{self.checkpoint_prefix}_best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  ★ Saved best model with loss: {metrics['loss']:.4f}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def fit(self, num_epochs: int, debug: bool = False) -> dict:
        print(f"\n{'='*60}")
        print(f"Starting {self.display_name} CZSL Training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Checkpoint prefix: {self.checkpoint_prefix}")
        print(f"Save dir: {self.save_dir}")
        print(f"{'='*60}\n")

        if self.use_wandb:
            wandb_config = self.config.get("logging", {}).get("wandb", {})
            wandb.init(
                project=wandb_config.get("project", "CLIP-CZSL-Jamming"),
                entity=wandb_config.get("entity", None),
                name=f"{self.checkpoint_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=wandb_config.get("tags", []) + self.wandb_tags + ["1D", "CZSL"],
                notes=wandb_config.get("notes", f"{self.display_name} CZSL training"),
                config=self.config,
            )
            wandb.watch(self.model, log="all", log_freq=100)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            train_metrics = self.train_epoch(debug=debug and epoch == 0)
            if self.scheduler:
                self.scheduler.step()

            val_metrics = self.validate(debug=debug and epoch == 0)

            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

            lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]['lr']

            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "lr": lr,
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["accuracy"],
                })

            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]
                self.save_checkpoint(val_metrics, is_best=True)
            elif (epoch + 1) % self.checkpoint_config.get("save_interval", 10) == 0 or epoch == num_epochs - 1:
                self.save_checkpoint(val_metrics, is_best=False)

        if self.use_wandb:
            wandb.finish()

        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        return {"best_val_loss": self.best_val_loss}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: dict,
) -> tuple:
    """Create AdamW optimizer + warmup cosine scheduler."""
    train_config = config.get("train", {})

    optimizer = AdamW(
        model.parameters(),
        lr=float(train_config.get("lr", 1e-4)),
        weight_decay=float(train_config.get("weight_decay", 0.01)),
    )

    scheduler_config = train_config.get("scheduler", {})
    scheduler_type = scheduler_config.get("type", "cosine")
    warmup_epochs = train_config.get("warmup_epochs", 3)
    num_epochs = train_config.get("epochs", 30)

    if scheduler_type == "cosine":
        actual_warmup = min(warmup_epochs, num_epochs - 1)
        t_max = max(1, num_epochs - actual_warmup)

        if actual_warmup > 0:
            warmup_scheduler = LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=actual_warmup
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=scheduler_config.get("min_lr", 1e-6)
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[actual_warmup],
            )
            print(f"Scheduler: warmup={actual_warmup} epochs, cosine={t_max} epochs")
        else:
            scheduler = CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=scheduler_config.get("min_lr", 1e-6)
            )
            print(f"Scheduler: cosine only, T_max={num_epochs}")
    else:
        scheduler = None

    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="1D CZSL Model Training (Conformer / ResNet1D / CNN1D)")
    parser.add_argument("--config", type=str, default="conformer_1d/config_1d.yaml",
                        help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug info for first batch of train/val")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model from config backbone
    print("\nCreating CZSL model...")
    model = create_1d_model(config, device=str(device))

    # Create dataloaders
    print("\nLoading 1D time-domain datasets...")
    train_loader, val_loader, test_loader = create_1d_dataloaders(config)

    # Cache text features: singles + seen ∪ unseen (align multi evaluate)
    czsl_config = config.get("czsl", {})
    seen_combos = czsl_config.get("seen_combinations", []) or []
    unseen_combos = czsl_config.get("unseen_combinations", []) or []
    all_comb_names = []
    _seen_keys = set()
    for c in list(seen_combos) + list(unseen_combos):
        key = tuple(sorted(c))
        if key not in _seen_keys:
            _seen_keys.add(key)
            all_comb_names.append(c)
    model.cache_text_features(
        max_combination_size=2,
        include_single=True,
        seen_combinations=all_comb_names if all_comb_names else None,
        use_translation=config.get('use_translation', False),
    )

    # Resume if needed
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Resumed from {args.resume} (epoch {checkpoint.get('epoch', '?')})")

    # Create optimizer & scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)

    # Create trainer
    trainer = Trainer1D(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
    )
    trainer.current_epoch = start_epoch

    # Train
    num_epochs = config.get("train", {}).get("epochs", 30)
    trainer.fit(num_epochs, debug=args.debug)


if __name__ == "__main__":
    main()
