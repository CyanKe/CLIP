"""
1D Models for CZSL Radar Jamming Signal Classification.

Supports three backbones sharing the same CLIP text encoder interface:
  - Conformer (ConformerEncoder, ConformerBlock, ConformerForCZSL)
  - ResNet-18 1D (ResNet1DEncoder, ResNet1DForCZSL)
  - Simple CNN-1D (SimpleCNN1DEncoder, CNN1DForCZSL)

All process raw I/Q time-domain signals (B, 2, 8000) with a unified
training and evaluation pipeline controlled by model.backbone in config.
"""

from .conformer import ConformerEncoder, ConformerBlock
from .resnet1d import ResNet1DEncoder
from .cnn1d import SimpleCNN1DEncoder
from .model_1d import (
    Base1DCZSLModel,
    ConformerForCZSL,
    ResNet1DForCZSL,
    CNN1DForCZSL,
    create_conformer_model,
    create_resnet1d_model,
    create_cnn1d_model,
    create_1d_model,
)
from .data_1d import TimeSignalDataset, create_1d_dataloaders
