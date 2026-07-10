"""
Persistence Spectrum CLIP Training Module.

Loads persistence spectrum (持续时间谱) data from .mat files and trains
a CLIP-based model for Compositional Zero-Shot Learning (CZSL) on radar
jamming signal classification.
"""

from .data import (
    PersistenceDataset, create_persistence_dataloaders, collate_fn,
    AblationDataset, collate_fn_ablation, create_ablation_dataloaders,
)
from .model import PersistenceCLIPForCZSL, create_persistence_model
from .train import PersistenceTrainer, main as train_main

__all__ = [
    "PersistenceDataset",
    "create_persistence_dataloaders",
    "collate_fn",
    "AblationDataset",
    "collate_fn_ablation",
    "create_ablation_dataloaders",
    "PersistenceCLIPForCZSL",
    "create_persistence_model",
    "PersistenceTrainer",
    "train_main",
]
