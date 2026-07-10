"""
Training script for Persistence Spectrum CLIP CZSL.

Usage:
    python persistence/train.py --config persistence/config.yaml
    python persistence/train.py --config persistence/config.yaml --resume checkpoints/persistence/persistence_latest.pt
    python persistence/train.py --config persistence/config.yaml --debug
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

from persistence.model import PersistenceCLIPForCZSL, create_persistence_model
from persistence.data import create_persistence_dataloaders, create_ablation_dataloaders
from multi.loss import create_loss_function, LabelAwareInfoNCELoss, MultiLabelSigmoidLoss


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PersistenceTrainer:
    """Trainer for Persistence Spectrum CLIP CZSL."""

    def __init__(
        self,
        model: PersistenceCLIPForCZSL,
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

        self.checkpoint_config = config.get("checkpoint", {})
        # Mode-aware checkpoint directory
        ablation_mode = config.get('ablation', {}).get('mode', 'persistence')
        base_save_dir = self.checkpoint_config.get("save_dir", "checkpoints/persistence")
        if ablation_mode != 'persistence':
            # Append mode suffix to avoid overwriting baseline checkpoints
            base_save_dir = f"{base_save_dir}_{ablation_mode}"
        self.save_dir = Path(base_save_dir)
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

        # Data: (images, None, text_tokens, labels, texts, metas [, features])
        for batch_idx, batch_data in enumerate(train_bar):
            if len(batch_data) >= 7:
                images, _, text_tokens, labels, texts, metas, *rest = batch_data
                features_dict = rest[0] if rest else None
            elif len(batch_data) == 6:
                images, _, text_tokens, labels, texts, metas = batch_data
                features_dict = None
            else:
                images, text_tokens, labels, texts, metas = batch_data
                features_dict = None

            images = images.to(self.device)
            text_tokens = text_tokens.to(self.device)
            labels = labels.to(self.device)

            if debug and not debug_done:
                print(f"\n{'='*80}")
                print(f"[DEBUG Train] Batch {batch_idx} — batch_size={images.size(0)}")
                print(f"  images shape: {images.shape}")
                print(f"  images value range: [{images.min().item():.4f}, {images.max().item():.4f}]")
                print(f"  labels[0]: {labels[0].tolist()}")
                print(f"  texts[0]: {texts[0]}")
                unique_texts = list(dict.fromkeys(texts))
                print(f"  unique texts: {len(unique_texts)} / {images.size(0)}")
                for t in unique_texts[:5]:
                    print(f"    -> {t}")
                print(f"{'='*80}")
                debug_done = True

            self.optimizer.zero_grad()

            # Forward
            image_features, text_features = self.model(images, text_tokens, features_dict)

            if self.use_sigmoid_loss or self.use_label_aware_loss:
                loss, logits_per_image, loss_info = self.loss_fn(
                    image_features, text_features, labels
                )
                logits_per_text = logits_per_image.T
            else:
                # Standard InfoNCE
                logit_scale = self.model.logit_scale.exp()
                logits_per_image = logit_scale * (image_features @ text_features.T)
                logits_per_text = logits_per_image.T

                targets = torch.arange(images.size(0), device=self.device)
                loss_i2t = F.cross_entropy(logits_per_image, targets)
                loss_t2i = F.cross_entropy(logits_per_text, targets)
                loss = (loss_i2t + loss_t2i) / 2

            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size

            with torch.no_grad():
                # Label-aware accuracy: matching any sample with identical labels counts as correct
                # This avoids penalizing the model when batch contains multiple samples of the same class
                pred_i2t = logits_per_image.argmax(dim=1)  # [B]
                pred_t2i = logits_per_text.argmax(dim=1)   # [B]
                # Build [B, B] boolean matrix: label_eq[i,j]==True if labels[i]==labels[j]
                label_eq = (labels.unsqueeze(1) == labels.unsqueeze(0)).all(dim=2)  # [B, B]
                # Count correct: predicted index falls in any position with identical labels
                correct_i2t = label_eq[torch.arange(batch_size, device=self.device), pred_i2t].sum().item()
                correct_t2i = label_eq[torch.arange(batch_size, device=self.device), pred_t2i].sum().item()
                total_correct += correct_i2t + correct_t2i
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
                images, _, text_tokens, labels, texts, metas, *rest = batch_data
                features_dict = rest[0] if rest else None
            elif len(batch_data) == 6:
                images, _, text_tokens, labels, texts, metas = batch_data
                features_dict = None
            else:
                images, text_tokens, labels, texts, metas = batch_data
                features_dict = None

            images = images.to(self.device)
            text_tokens = text_tokens.to(self.device)
            labels = labels.to(self.device)

            if debug and not debug_done:
                print(f"\n{'='*80}")
                print(f"[DEBUG Val] batch_size={images.size(0)}")
                print(f"  images value range: [{images.min().item():.4f}, {images.max().item():.4f}]")
                print(f"  labels[0]: {labels[0].tolist()}")
                print(f"  texts[0]: {texts[0]}")
                unique_texts = list(dict.fromkeys(texts))
                print(f"  unique texts: {len(unique_texts)} / {images.size(0)}")
                print(f"{'='*80}")
                debug_done = True

            image_features, text_features = self.model(images, text_tokens, features_dict)

            if self.use_sigmoid_loss or self.use_label_aware_loss:
                loss, logits_per_image, loss_info = self.loss_fn(
                    image_features, text_features, labels
                )
                logits_per_text = logits_per_image.T
            else:
                logit_scale = self.model.logit_scale.exp()
                logits_per_image = logit_scale * (image_features @ text_features.T)
                logits_per_text = logits_per_image.T

                targets = torch.arange(images.size(0), device=self.device)
                loss_i2t = F.cross_entropy(logits_per_image, targets)
                loss_t2i = F.cross_entropy(logits_per_text, targets)
                loss = (loss_i2t + loss_t2i) / 2

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size

            # Label-aware accuracy: matching any sample with identical labels counts as correct
            pred_i2t = logits_per_image.argmax(dim=1)
            pred_t2i = logits_per_text.argmax(dim=1)
            label_eq = (labels.unsqueeze(1) == labels.unsqueeze(0)).all(dim=2)
            correct_i2t = label_eq[torch.arange(batch_size, device=self.device), pred_i2t].sum().item()
            correct_t2i = label_eq[torch.arange(batch_size, device=self.device), pred_t2i].sum().item()
            total_correct += correct_i2t + correct_t2i
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
        ablation_mode = self.config.get('ablation', {}).get('mode', 'persistence')
        prefix = f"persistence_{ablation_mode}" if ablation_mode != 'persistence' else "persistence"

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "metrics": metrics,
            "config": self.config,
        }

        latest_path = self.save_dir / f"{prefix}_latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = self.save_dir / f"{prefix}_best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  ★ Saved best model with loss: {metrics['loss']:.4f}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def fit(self, num_epochs: int, debug: bool = False) -> dict:
        print(f"\n{'='*60}")
        print(f"Starting Persistence Spectrum CLIP CZSL Training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        if self.val_loader is not None:
            print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"{'='*60}\n")

        if self.use_wandb:
            wandb_config = self.config.get("logging", {}).get("wandb", {})
            ablation_mode = self.config.get('ablation', {}).get('mode', 'persistence')
            run_name = f"Persistence_{ablation_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project=wandb_config.get("project", "CLIP-CZSL-Jamming"),
                entity=wandb_config.get("entity", None),
                name=run_name,
                tags=wandb_config.get("tags", []) + ["Persistence", "CZSL", ablation_mode],
                notes=wandb_config.get("notes", "Persistence spectrum CLIP CZSL training"),
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

            lr = (self.scheduler.get_last_lr()[0] if self.scheduler
                  else self.optimizer.param_groups[0]['lr'])

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

            save_best_only = self.checkpoint_config.get("save_best_only", True)
            if not save_best_only or is_best:
                self.save_checkpoint(val_metrics, is_best)

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
        lr=float(train_config.get("lr", 1e-5)),
        weight_decay=float(train_config.get("weight_decay", 0.01)),
    )

    scheduler_config = train_config.get("scheduler", {})
    scheduler_type = scheduler_config.get("type", "cosine")
    warmup_epochs = train_config.get("warmup_epochs", 2)
    num_epochs = train_config.get("epochs", 20)

    if scheduler_type == "cosine":
        actual_warmup = min(warmup_epochs, num_epochs - 1)
        t_max = max(1, num_epochs - actual_warmup)

        if actual_warmup > 0:
            warmup_scheduler = LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=actual_warmup
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=scheduler_config.get("min_lr", 1e-7)
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[actual_warmup],
            )
            print(f"Scheduler: warmup={actual_warmup} epochs, cosine={t_max} epochs")
        else:
            scheduler = CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=scheduler_config.get("min_lr", 1e-7)
            )
            print(f"Scheduler: cosine only, T_max={num_epochs}")
    else:
        scheduler = None

    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Persistence Spectrum CLIP CZSL Training")
    parser.add_argument("--config", type=str, default="persistence/config.yaml",
                        help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug info for first batch of train/val")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    print("\nCreating Persistence CLIP CZSL model...")
    model = create_persistence_model(config, device=str(device))

    # Determine ablation mode from config
    ablation_config = config.get('ablation', {})
    ablation_mode = ablation_config.get('mode', 'persistence')

    # Create dataloaders — use ablation-aware loader for stft/fusion, persistence for baseline
    print(f"\nLoading datasets [ablation mode: {ablation_mode}]...")
    if ablation_mode == 'persistence':
        train_loader, val_loader, test_loader, num_classes, jnr_levels = \
            create_persistence_dataloaders(config)
    else:
        train_loader, val_loader, test_loader, num_classes, jnr_levels = \
            create_ablation_dataloaders(config)

    # Cache text features for zero-shot inference
    czsl_config = config.get("czsl", {})
    seen_combos = czsl_config.get("seen_combinations", None)
    model.cache_text_features(
        max_combination_size=2,
        include_single=True,
        seen_combinations=seen_combos,
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

    # Determine whether to use wandb from config
    logging_config = config.get("logging", {})
    logging_type = logging_config.get("type", "console")
    use_wandb = (logging_type == "wandb")

    # Create trainer
    trainer = PersistenceTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        use_wandb=use_wandb,
    )
    trainer.current_epoch = start_epoch

    # Train
    num_epochs = config.get("train", {}).get("epochs", 20)
    trainer.fit(num_epochs, debug=args.debug)


if __name__ == "__main__":
    main()
