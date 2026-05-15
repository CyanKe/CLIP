"""
统一训练脚本 - 通过 config.yaml 控制训练模式
python -m multi.train_unified --config multi/config.yaml

配置方式:
  model.backbone: "vit" | "multishape_vit" | "resnet18"
  model.dual_branch: true (双分支) / false (单分支对比学习)

模型分发:
  dual_branch=false:
    - backbone="vit"            → CLIP 对比学习
    - backbone="multishape_vit" → MultiShapeViT 对比学习
  dual_branch=true:
    - backbone="vit"            → CLIP 双分支
    - backbone="multishape_vit" → MultiShapeViT 双分支
    - backbone="resnet18"       → ResNet18 双分支分类
"""
import os
import sys
import yaml
import argparse
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



# ============================================================================
# StepResult: forward_pass 的统一返回类型
# ============================================================================

@dataclass
class StepResult:
    loss: torch.Tensor
    logits_for_metrics: Dict[str, torch.Tensor]
    batch_size: int


# ============================================================================
# 公共函数
# ============================================================================

def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: dict,
    num_training_steps: int
) -> tuple:
    train_config = config.get("train", {})
    use_feature_context = config.get("use_feature_context", False)

    # 特征条件上下文: prompt_learner 参数使用更高学习率
    if use_feature_context and hasattr(model, 'prompt_learner') and model.prompt_learner is not None:
        prompt_params = list(model.prompt_learner.parameters())
        prompt_param_ids = {id(p) for p in prompt_params}
        other_params = [p for p in model.parameters()
                       if p.requires_grad and id(p) not in prompt_param_ids]

        prompt_lr = float(train_config.get("prompt_lr", 2e-3))
        base_lr = float(train_config.get("lr", 1e-5))

        print(f"  Prompt learner LR: {prompt_lr}, Base LR: {base_lr}")
        print(f"  Prompt learner params: {sum(p.numel() for p in prompt_params):,}")
        print(f"  Other trainable params: {sum(p.numel() for p in other_params):,}")

        optimizer = AdamW([
            {"params": other_params, "lr": base_lr, "weight_decay": float(train_config.get("weight_decay", 0.01))},
            {"params": prompt_params, "lr": prompt_lr, "weight_decay": 0.0},
        ])
    else:
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


# ============================================================================
# DualBranchClassificationLoss: ResNet18 双分支分类用的薄包装
# ============================================================================

class DualBranchClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_deception = nn.CrossEntropyLoss()
        self.ce_suppression = nn.CrossEntropyLoss()

    def forward(self, logits_deception, logits_suppression, labels_d_idx, labels_s_idx):
        loss_d = self.ce_deception(logits_deception, labels_d_idx)
        loss_s = self.ce_suppression(logits_suppression, labels_s_idx)
        return loss_d + loss_s, {"loss_deception": loss_d.item(), "loss_suppression": loss_s.item()}


# ============================================================================
# TrainingStrategy 抽象基类
# ============================================================================

class TrainingStrategy(ABC):
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device

    @abstractmethod
    def create_model(self) -> nn.Module: ...

    @abstractmethod
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]: ...

    @abstractmethod
    def create_loss(self, model: nn.Module) -> nn.Module: ...

    @abstractmethod
    def cache_text_features(self, model: nn.Module) -> None: ...

    @abstractmethod
    def prepare_batch(self, batch_data: tuple) -> dict: ...

    @abstractmethod
    def forward_pass(self, model: nn.Module, batch: dict, loss_fn: nn.Module) -> StepResult: ...

    @abstractmethod
    def compute_epoch_metrics(self, accumulated: dict, dataset_len: int) -> Dict[str, float]: ...

    @abstractmethod
    def init_accumulators(self) -> dict: ...

    @abstractmethod
    def accumulate(self, acc: dict, result: StepResult, batch: dict) -> None: ...

    @abstractmethod
    def checkpoint_prefix(self) -> str: ...

    @abstractmethod
    def wandb_run_name(self) -> str: ...

    @abstractmethod
    def wandb_tags(self) -> List[str]: ...

    @abstractmethod
    def metric_display_keys(self) -> List[str]: ...

    @abstractmethod
    def best_metric_key(self) -> str: ...

    @abstractmethod
    def best_metric_direction(self) -> str: ...


# ============================================================================
# CZSLStrategy: 对比学习模式
# ============================================================================

class CZSLStrategy(TrainingStrategy):
    def __init__(self, config: dict, device: torch.device):
        super().__init__(config, device)
        model_config = config.get("model", {})
        self.model_type = model_config.get("clip_model", "ViT-B/32")
        self.backbone = model_config.get("backbone", "vit")

    def create_model(self) -> nn.Module:
        if self.backbone == "multishape_vit":
            from multi.rectangular_patch_vit import create_multi_shape_patch_model
            print("Creating Multi-Shape Patch ViT CZSL model...")
            return create_multi_shape_patch_model(self.config, device=str(self.device))
        else:
            from multi.model import create_czsl_model
            print(f"Creating CZSL model ({self.model_type})...")
            return create_czsl_model(self.config, device=str(self.device))

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        from multi.data import create_czsl_dataloaders
        return create_czsl_dataloaders(
            config=self.config,
            batch_size=self.config.get("train", {}).get("batch_size", 16),
            num_workers=self.config.get("data", {}).get("num_workers", 4),
            pin_memory=self.config.get("data", {}).get("pin_memory", True),
            load_test=False,
        )

    def create_loss(self, model: nn.Module) -> nn.Module:
        from multi.loss import create_loss_function
        return create_loss_function(self.config, model_type=self.model_type)

    def cache_text_features(self, model: nn.Module) -> None:
        czsl_config = self.config.get("czsl", {})
        seen_combos = czsl_config.get("seen_combinations", None)
        model.cache_text_features(
            max_combination_size=2,
            include_single=True,
            seen_combinations=seen_combos,
        )

    def prepare_batch(self, batch_data: tuple) -> dict:
        # batch_data: (images, time_signals, text_tokens, labels, texts, metas, features_batched)
        if len(batch_data) == 7:
            images, time_signals, text_tokens, labels, texts, metas, features = batch_data
        elif len(batch_data) == 6:
            images, time_signals, text_tokens, labels, texts, metas = batch_data
            features = None
        else:
            images, text_tokens, labels, texts, metas = batch_data
            time_signals = None
            features = None

        images = images.to(self.device)
        text_tokens = text_tokens.to(self.device)
        labels = labels.to(self.device)
        if time_signals is not None:
            time_signals = time_signals.to(self.device)

        if images.shape[-1] != 224:
            images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

        result = {
            "images": images,
            "text_tokens": text_tokens,
            "labels": labels,
            "time_signals": time_signals,
        }

        if features is not None:
            result["features"] = {d: f.to(self.device) for d, f in features.items()}
        else:
            result["features"] = None

        return result

    def forward_pass(self, model: nn.Module, batch: dict, loss_fn: nn.Module) -> StepResult:
        from multi.loss import LabelAwareInfoNCELoss, MultiLabelSigmoidLoss

        image_features, text_features = model(
            batch["images"], batch["text_tokens"], batch["time_signals"],
            features_dict=batch.get("features"),
        )
        labels = batch["labels"]
        batch_size = batch["images"].size(0)

        if isinstance(loss_fn, MultiLabelSigmoidLoss):
            loss, logits_per_image, _ = loss_fn(image_features, text_features, labels)
            logits_per_text = logits_per_image.T
        elif isinstance(loss_fn, LabelAwareInfoNCELoss):
            loss, logits_per_image, _ = loss_fn(image_features, text_features, labels)
            logits_per_text = logits_per_image.T
        else:
            if hasattr(model, 'model'):
                logit_scale = model.model.logit_scale.exp()
            else:
                logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * (image_features @ text_features.t())
            logits_per_text = logits_per_image.T
            targets = torch.arange(batch_size, device=self.device)
            loss_i2t = F.cross_entropy(logits_per_image, targets)
            loss_t2i = F.cross_entropy(logits_per_text, targets)
            loss = (loss_i2t + loss_t2i) / 2

        return StepResult(
            loss=loss,
            logits_for_metrics={"i2t": logits_per_image, "t2i": logits_per_text},
            batch_size=batch_size,
        )

    def init_accumulators(self) -> dict:
        return {"total_loss": 0.0, "total_correct": 0, "total_samples": 0}

    def accumulate(self, acc: dict, result: StepResult, batch: dict) -> None:
        acc["total_loss"] += result.loss.item() * result.batch_size
        with torch.no_grad():
            targets = torch.arange(result.batch_size, device=self.device)
            logits = result.logits_for_metrics
            acc["total_correct"] += (logits["i2t"].argmax(1) == targets).sum().item()
            acc["total_correct"] += (logits["t2i"].argmax(1) == targets).sum().item()
            acc["total_samples"] += result.batch_size * 2

    def compute_epoch_metrics(self, accumulated: dict, dataset_len: int) -> Dict[str, float]:
        return {
            "loss": accumulated["total_loss"] / dataset_len,
            "accuracy": accumulated["total_correct"] / accumulated["total_samples"],
        }

    def checkpoint_prefix(self) -> str:
        return "multishape_vit_" if self.backbone == "multishape_vit" else "czsl_"

    def wandb_run_name(self) -> str:
        prefix = "MultiShapeViT" if self.backbone == "multishape_vit" else "CZSL"
        return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def wandb_tags(self) -> List[str]:
        if self.backbone == "multishape_vit":
            return ["MultiShapeViT", "early_fusion"]
        return ["CZSL", "InfoNCE"]

    def metric_display_keys(self) -> List[str]:
        return ["loss", "accuracy"]

    def best_metric_key(self) -> str:
        return "loss"

    def best_metric_direction(self) -> str:
        return "min"


# ============================================================================
# DualBranchStrategy: 双分支模式
# ============================================================================

class DualBranchStrategy(TrainingStrategy):
    def __init__(self, config: dict, device: torch.device):
        super().__init__(config, device)
        model_config = config.get("model", {})
        self.backbone = model_config.get("backbone", "vit")

    def create_model(self) -> nn.Module:
        if self.backbone == "resnet18":
            from multi.model import create_resnet18_dual_branch_model
            print("Creating ResNet18 dual-branch model...")
            return create_resnet18_dual_branch_model(self.config, device=str(self.device))
        elif self.backbone == "multishape_vit":
            from multi.rectangular_patch_vit import create_multi_shape_dual_branch_model
            print("Creating Multi-Shape Patch ViT dual-branch model...")
            return create_multi_shape_dual_branch_model(self.config, device=str(self.device))
        else:
            from multi.model import create_dual_branch_model
            print("Creating CLIP dual-branch model...")
            return create_dual_branch_model(self.config, device=str(self.device))

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        from multi.data import create_dual_branch_dataloaders
        return create_dual_branch_dataloaders(
            config=self.config,
            batch_size=self.config.get("train", {}).get("batch_size", 16),
            num_workers=self.config.get("data", {}).get("num_workers", 4),
            pin_memory=self.config.get("data", {}).get("pin_memory", True),
            load_test=False,
        )

    def create_loss(self, model: nn.Module) -> nn.Module:
        if self.backbone == "resnet18":
            print("Using CrossEntropyLoss for both branches")
            return DualBranchClassificationLoss()
        else:
            from multi.loss import DualBranchContrastiveLoss
            loss_config = self.config.get("loss", {})
            loss_fn = DualBranchContrastiveLoss(
                temperature=loss_config.get("temperature", 0.07),
                learnable_temperature=loss_config.get("learnable_temperature", True),
                label_smoothing=loss_config.get("label_smoothing", 0.0),
            )
            print("Using DualBranchContrastiveLoss")
            return loss_fn

    def cache_text_features(self, model: nn.Module) -> None:
        if not self.backbone == "resnet18":
            model.cache_text_features_dual()

    def prepare_batch(self, batch_data: tuple) -> dict:
        (images, tok_d, tok_s, lab_d, lab_s, txt_d, txt_s, metas) = batch_data

        images = images.to(self.device)
        lab_d = lab_d.to(self.device)
        lab_s = lab_s.to(self.device)

        if images.shape[-1] != 224:
            images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

        result = {"images": images, "labels_deception": lab_d, "labels_suppression": lab_s}

        if not self.backbone == "resnet18":
            result["text_tokens_deception"] = tok_d.to(self.device)
            result["text_tokens_suppression"] = tok_s.to(self.device)
        else:
            result["labels_deception_idx"] = torch.argmax(lab_d, dim=1).to(self.device)
            result["labels_suppression_idx"] = torch.argmax(lab_s, dim=1).to(self.device)

        return result

    def forward_pass(self, model: nn.Module, batch: dict, loss_fn: nn.Module) -> StepResult:
        batch_size = batch["images"].size(0)

        if self.backbone == "resnet18":
            logits_d, logits_s = model(batch["images"])
            loss, _ = loss_fn(logits_d, logits_s, batch["labels_deception_idx"], batch["labels_suppression_idx"])
            return StepResult(
                loss=loss,
                logits_for_metrics={"logits_d": logits_d, "logits_s": logits_s},
                batch_size=batch_size,
            )
        else:
            img_feat, txt_d_feat, txt_s_feat = model(
                batch["images"],
                batch["text_tokens_deception"],
                batch["text_tokens_suppression"],
                features_dict=batch.get("features"),
            )
            loss, loss_info = loss_fn(
                img_feat, txt_d_feat, txt_s_feat,
                batch["labels_deception"], batch["labels_suppression"],
            )
            # 计算对角线准确率用的 logits
            if hasattr(model, 'model'):
                logit_scale = model.model.logit_scale.exp()
            else:
                logit_scale = model.logit_scale.exp()
            logits_d = logit_scale * (img_feat @ txt_d_feat.t())
            logits_s = logit_scale * (img_feat @ txt_s_feat.t())
            return StepResult(
                loss=loss,
                logits_for_metrics={"logits_d": logits_d, "logits_s": logits_s},
                batch_size=batch_size,
            )

    def init_accumulators(self) -> dict:
        return {
            "total_loss": 0.0,
            "total_correct_deception": 0,
            "total_correct_suppression": 0,
            "total_samples": 0,
        }

    def accumulate(self, acc: dict, result: StepResult, batch: dict) -> None:
        acc["total_loss"] += result.loss.item() * result.batch_size
        with torch.no_grad():
            logits = result.logits_for_metrics
            bs = result.batch_size
            if self.backbone == "resnet18":
                acc["total_correct_deception"] += (logits["logits_d"].argmax(1) == batch["labels_deception_idx"]).sum().item()
                acc["total_correct_suppression"] += (logits["logits_s"].argmax(1) == batch["labels_suppression_idx"]).sum().item()
            else:
                targets = torch.arange(bs, device=self.device)
                acc["total_correct_deception"] += (logits["logits_d"].argmax(1) == targets).sum().item()
                acc["total_correct_suppression"] += (logits["logits_s"].argmax(1) == targets).sum().item()
            acc["total_samples"] += bs

    def compute_epoch_metrics(self, accumulated: dict, dataset_len: int) -> Dict[str, float]:
        return {
            "loss": accumulated["total_loss"] / dataset_len,
            "accuracy_deception": accumulated["total_correct_deception"] / accumulated["total_samples"],
            "accuracy_suppression": accumulated["total_correct_suppression"] / accumulated["total_samples"],
        }

    def checkpoint_prefix(self) -> str:
        if self.backbone == "resnet18":
            return "resnet18_dual_branch_"
        elif self.backbone == "multishape_vit":
            return "multishape_dual_branch_"
        return "dual_branch_"

    def wandb_run_name(self) -> str:
        if self.backbone == "resnet18":
            prefix = "ResNet18DualBranch"
        elif self.backbone == "multishape_vit":
            prefix = "MultiShapeDualBranch"
        else:
            prefix = "DualBranch"
        return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def wandb_tags(self) -> List[str]:
        return ["DualBranch", "CZSL"]

    def metric_display_keys(self) -> List[str]:
        return ["loss", "accuracy_deception", "accuracy_suppression"]

    def best_metric_key(self) -> str:
        return "loss"

    def best_metric_direction(self) -> str:
        return "min"


# ============================================================================
# Strategy 工厂
# ============================================================================

def create_strategy(config: dict, device: torch.device) -> TrainingStrategy:
    dual_branch = config.get("model", {}).get("dual_branch", False)
    if dual_branch:
        return DualBranchStrategy(config, device)
    else:
        return CZSLStrategy(config, device)


# ============================================================================
# UnifiedTrainer: 统一训练器
# ============================================================================

class UnifiedTrainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        config: dict,
        strategy: TrainingStrategy,
        use_wandb: bool = True,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.strategy = strategy
        self.use_wandb = use_wandb and HAS_WANDB

        self.current_epoch = 0
        self.train_config = config.get("train", {})
        self.grad_clip = self.train_config.get("grad_clip", 1.0)

        # Best model tracking
        self.best_metric_key = strategy.best_metric_key()
        self.best_direction = strategy.best_metric_direction()
        self.best_val_metric = float('inf') if self.best_direction == "min" else 0.0

        # Checkpoint
        self.checkpoint_config = config.get("checkpoint", {})
        self.save_dir = Path(self.checkpoint_config.get("save_dir", "checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _run_epoch(self, is_train: bool, debug: bool = False) -> dict:
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        accumulated = self.strategy.init_accumulators()
        loader = self.train_loader if is_train else self.val_loader
        phase = "Train" if is_train else "Val"
        bar = tqdm(loader, desc=f"Epoch {self.current_epoch + 1} [{phase}]")

        ctx = nullcontext() if is_train else torch.no_grad()
        with ctx:
            for batch_idx, batch_data in enumerate(bar):
                batch = self.strategy.prepare_batch(batch_data)

                if is_train:
                    self.optimizer.zero_grad()

                step_result = self.strategy.forward_pass(self.model, batch, self.loss_fn)

                if is_train:
                    step_result.loss.backward()
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                self.strategy.accumulate(accumulated, step_result, batch)
                bar.set_postfix(loss=step_result.loss.item())

        return self.strategy.compute_epoch_metrics(accumulated, len(loader.dataset))

    def train_epoch(self, debug: bool = False) -> dict:
        return self._run_epoch(is_train=True, debug=debug)

    @torch.no_grad()
    def validate(self, debug: bool = False) -> dict:
        return self._run_epoch(is_train=False, debug=debug)

    def save_checkpoint(self, metrics: dict, is_best: bool = False):
        prefix = self.strategy.checkpoint_prefix()
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "metrics": metrics,
            "config": self.config,
        }
        latest_path = self.save_dir / f"{prefix}latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)
        if is_best:
            best_path = self.save_dir / f"{prefix}best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  Saved best model ({self.best_metric_key}: {metrics[self.best_metric_key]:.4f})")

    def fit(self, num_epochs: int, debug: bool = False) -> dict:
        mode_label = self.strategy.__class__.__name__.replace("Strategy", "")
        print(f"\n{'='*60}")
        print(f"Starting {mode_label} Training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        if self.use_wandb:
            wandb_config = self.config.get("logging", {}).get("wandb", {})
            wandb.init(
                project=wandb_config.get("project", "CLIP-CZSL-Jamming"),
                entity=wandb_config.get("entity", None),
                name=self.strategy.wandb_run_name(),
                tags=wandb_config.get("tags", []) + self.strategy.wandb_tags(),
                config=self.config,
            )
            wandb.watch(self.model, log="all", log_freq=100)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            train_metrics = self.train_epoch(debug=debug and epoch == 0)
            self.scheduler.step()

            val_metrics = self.validate(debug=debug and epoch == 0)

            display_keys = self.strategy.metric_display_keys()
            train_str = ", ".join(f"{k}: {train_metrics[k]:.4f}" for k in display_keys)
            val_str = ", ".join(f"{k}: {val_metrics[k]:.4f}" for k in display_keys)
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train - {train_str}")
            print(f"  Val   - {val_str}")

            if self.use_wandb:
                log_dict = {"epoch": epoch + 1, "lr": self.scheduler.get_last_lr()[0]}
                for k in display_keys:
                    log_dict[f"train_{k}"] = train_metrics[k]
                    log_dict[f"val_{k}"] = val_metrics[k]
                wandb.log(log_dict)

            val_metric = val_metrics[self.best_metric_key]
            if self.best_direction == "min":
                is_best = val_metric < self.best_val_metric
            else:
                is_best = val_metric > self.best_val_metric
            if is_best:
                self.best_val_metric = val_metric

            save_best_only = self.checkpoint_config.get("save_best_only", False)
            if not save_best_only or is_best:
                self.save_checkpoint(val_metrics, is_best)

        if self.use_wandb:
            wandb.finish()

        print(f"\nTraining completed!")
        print(f"Best val {self.best_metric_key}: {self.best_val_metric:.4f}")
        return {f"best_val_{self.best_metric_key}": self.best_val_metric}


# ============================================================================
# main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified CLIP/CZSL Training")
    parser.add_argument("--config", type=str, default="multi/config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true", help="Print debug info for first batch")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--patch-sizes", type=str, default=None, help="Override patch sizes, e.g., '8,32;32,8;16,16'")
    parser.add_argument("--fusion-mode", type=str, default=None, choices=["early_fusion", "late_fusion"])
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # CLI 覆盖
    if args.lr:
        config["train"]["lr"] = args.lr
    if args.epochs:
        config["train"]["epochs"] = args.epochs
    if args.patch_sizes:
        patch_sizes = []
        for ps in args.patch_sizes.split(';'):
            h, w = map(int, ps.split(','))
            patch_sizes.append((h, w))
        config["model"]["patch_sizes"] = patch_sizes
    if args.fusion_mode:
        config["model"]["fusion_mode"] = args.fusion_mode
    if args.embed_dim:
        config["model"]["embed_dim"] = args.embed_dim
    if args.depth:
        config["model"]["depth"] = args.depth

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建策略
    strategy = create_strategy(config, device)

    # 创建模型
    model = strategy.create_model()

    # 创建数据加载器
    print("\nLoading datasets...")
    dataloaders = strategy.create_dataloaders()
    train_loader, val_loader = dataloaders[0], dataloaders[1]

    # 缓存文本特征
    strategy.cache_text_features(model)

    # 恢复训练
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Resumed from {args.resume}")

    # 创建损失函数
    loss_fn = strategy.create_loss(model)

    # 创建优化器和调度器
    num_training_steps = len(train_loader) * config.get("train", {}).get("epochs", 20)
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, num_training_steps)

    # WandB
    log_type = config.get("logging", {}).get("type", "console")
    use_wandb = (log_type == "wandb") and HAS_WANDB

    # 创建训练器
    trainer = UnifiedTrainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        strategy=strategy,
        use_wandb=use_wandb,
    )

    # 开始训练
    num_epochs = config.get("train", {}).get("epochs", 20)
    trainer.fit(num_epochs, debug=args.debug)


if __name__ == "__main__":
    main()
