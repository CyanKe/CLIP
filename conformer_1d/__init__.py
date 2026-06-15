"""
1D Conformer for CZSL Radar Jamming Signal Classification.

Replaces the 2D ViT vision encoder with a 1D Conformer that directly
processes raw I/Q time-domain signals (B, 2, 8000) while reusing
CLIP's text encoder for contrastive learning.
"""

from .conformer import ConformerEncoder, ConformerBlock
from .model_1d import ConformerForCZSL, create_conformer_model
from .data_1d import TimeSignalDataset, create_1d_dataloaders
