"""
MoE Hybrid Interference Evaluation Module.

Fuses two independently trained CZSL models at prediction level:
  - Conformer (1D time-domain) — biased towards deception (欺骗) recognition
  - Persistence (2D spectrum)    — biased towards suppression (压制) recognition

Fusion rule: deception predictions from Conformer, suppression predictions
from Persistence, with fallback to the other expert when a component is missing.
"""

from .evaluate_hybrid import (
    DualModalDataset,
    create_dual_modal_dataloaders,
    HybridMoEEvaluator,
    main,
)

__all__ = [
    "DualModalDataset",
    "create_dual_modal_dataloaders",
    "HybridMoEEvaluator",
    "main",
]
