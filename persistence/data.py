"""
Data loading for Persistence Spectrum: 2D probability distributions from HDF5.

Mirrors multi/data.py but loads from *_echo_persistences.mat instead of
*_echo_stfts.mat. Persistence data is already float32 [0,1] probabilities,
so no complex conversion or per-sample normalization is needed.
"""

import json
import os
import sys
from functools import partial

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# Reuse text template generation from multi/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from multi.text_templates import generate_text_descriptions


# ---------------------------------------------------------------------------
# CLIP normalization constants
# ---------------------------------------------------------------------------

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


# ---------------------------------------------------------------------------
# Tokenizer wrapper (reused from conformer_1d pattern)
# ---------------------------------------------------------------------------

class TokenizerWrapper:
    """Wrapper around clip.tokenize for use in collate functions."""

    def __init__(self, model_type: str = "clip"):
        import clip
        self._model_type = model_type
        self._tokenizer = clip.tokenize
        self._truncate = True

    def __call__(self, texts):
        return self._tokenizer(texts, truncate=self._truncate)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PersistenceDataset(Dataset):
    """Load persistence spectrum data from HDF5 files.

    Each sample is a 2D probability distribution: (freq_bins, power_bins)
    where each frequency bin's power distribution sums to ~1.

    The data is stacked 3× to create a 3-channel input for CLIP's ViT encoder,
    matching the pattern used for STFT magnitude images.
    """

    def __init__(
        self,
        persistence_file: str,
        metadata_file: str,
        persistence_var_name: str = "all_persistences",
        class_names: list = None,
        image_size: int = 224,
        apply_clip_norm: bool = True,
        augmentation=None,
        features_file: str = None,
        feature_dims: dict = None,
        use_feature_context: bool = False,
    ):
        """
        Args:
            persistence_file: path to *_echo_persistences.mat (HDF5)
            metadata_file: path to *_echo_metadata.json
            persistence_var_name: HDF5 variable name (default 'all_persistences')
            class_names: list of jam type names for label building
            image_size: output spatial size (default 224)
            apply_clip_norm: apply CLIP ImageNet normalization
            augmentation: optional STFTAugmentation object
            features_file: optional path to signal features JSON
            feature_dims: dict of per-domain feature dimensions
            use_feature_context: enable CoOp-style feature-conditioned context
        """
        self.image_size = image_size
        self.apply_clip_norm = apply_clip_norm
        self.augmentation = augmentation
        self.use_feature_context = use_feature_context

        # ---- Load persistence spectrum (float32, already [0,1] probabilities) ----
        with h5py.File(persistence_file, 'r') as f:
            # HDF5 stores MATLAB column-major: [power_bins, freq, samples]
            # Transpose to [samples, freq, power_bins] = [N, 224, 224]
            raw = f[persistence_var_name][:]
            # Generic transpose: move last dim to first, reverse the rest
            # For 3D: (2, 1, 0) → [samples, freq, power]
            axes = tuple(range(raw.ndim - 1, -1, -1))
            self._all_data = np.transpose(raw, axes=axes).astype(np.float32)
            self.num_samples = self._all_data.shape[0]

        # ---- Load metadata ----
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self._metadata = json.load(f)

        # ---- Build labels ----
        self.class_names = class_names or []
        self.labels = self._build_labels_from_metadata()

        # ---- Optional features for CoOp context ----
        self._features = None
        self.feature_dims = feature_dims
        if use_feature_context and features_file and os.path.exists(features_file):
            with open(features_file, 'r', encoding='utf-8') as f:
                self._features = json.load(f)

    # ------------------------------------------------------------------
    # Label building (same logic as STFTDataset)
    # ------------------------------------------------------------------

    def _build_labels_from_metadata(self) -> np.ndarray:
        """Build multi-hot label matrix from metadata.

        Returns:
            (num_samples, num_classes) float32 array
        """
        num_classes = len(self.class_names)
        labels = np.zeros((self.num_samples, num_classes), dtype=np.float32)

        for i, meta in enumerate(self._metadata):
            jam_types = meta.get('jam_types', [])
            if isinstance(jam_types, str):
                jam_types = [jam_types]
            for jt in jam_types:
                if jt in self.class_names:
                    j = self.class_names.index(jt)
                    labels[i, j] = 1.0

        return labels

    # ------------------------------------------------------------------

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # 1. Get 2D persistence spectrum [freq, power] → [224, 224]
        persistence = self._all_data[index]  # (H, W) = (freq_bins, power_bins)

        # 2. Stack 3 times → [3, H, W] for CLIP ViT (same pattern as STFT mag×3)
        tensor = torch.from_numpy(
            np.stack([persistence, persistence, persistence], axis=0)
        ).float()

        # 3. Resize if needed (data is already 224×224 from simulation)
        if tensor.shape[1] != self.image_size or tensor.shape[2] != self.image_size:
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

        # 4. CLIP normalization
        if self.apply_clip_norm:
            from torchvision.transforms import Normalize
            norm = Normalize(mean=CLIP_MEAN, std=CLIP_STD)
            tensor = norm(tensor)

        # 5. Optional augmentation (post-CLIP-norm, mask_value=0 = ~mean)
        if self.augmentation is not None:
            tensor = self.augmentation(tensor, None)  # No label needed for basic aug

        # 6. Get label and metadata
        label = torch.from_numpy(self.labels[index])
        meta = self._metadata[index] if index < len(self._metadata) else {}

        # 7. Optional features for CoOp context
        if self.use_feature_context and self._features is not None:
            feats = self._features[index] if index < len(self._features) else {}
            return tensor, label, meta, feats

        return tensor, label, meta


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(
    batch,
    tokenizer_fn=None,
    model_type: str = "clip",
    use_feature_context: bool = False,
):
    """Collate persistence spectrum samples into a batch.

    Returns a tuple matching the conformer_1d/stft convention:
        (images, None, text_tokens, labels, texts, meta_list [, features_batched])

    The second element (None) is a placeholder for time-domain signals,
    keeping the interface consistent across data modalities.
    """
    has_features = use_feature_context and len(batch[0]) >= 4

    if has_features:
        images, labels, metas, feats = zip(*[
            (item[0], item[1], item[2], item[3]) for item in batch
        ])
    else:
        images, labels, metas = zip(*[
            (item[0], item[1], item[2]) for item in batch
        ])

    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)

    # Generate text descriptions from metadata (per-sample, matching conformer pattern)
    texts = [generate_text_descriptions(meta, style='class_only') for meta in metas]

    # Tokenize
    if tokenizer_fn is not None:
        text_tokens = tokenizer_fn(texts)
    else:
        import clip
        text_tokens = clip.tokenize(texts, truncate=True)

    if has_features:
        # Batch features dict by domain
        all_keys = set()
        for f in feats:
            if isinstance(f, dict):
                all_keys.update(f.keys())
        features_batched = {}
        for k in sorted(all_keys):
            vals = []
            for f in feats:
                v = f.get(k, 0.0) if isinstance(f, dict) else 0.0
                vals.append(v)
            features_batched[k] = torch.tensor(vals, dtype=torch.float32)

        return images, None, text_tokens, labels, texts, list(metas), features_batched

    return images, None, text_tokens, labels, texts, list(metas)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_persistence_dataloaders(config: dict, load_test: bool = False):
    """Create train/val/test DataLoaders from config.

    Iterates over JNR levels, creates PersistenceDataset per split,
    concatenates across JNR levels, and returns DataLoader instances.

    Args:
        config: full YAML config dict
        load_test: whether to load test split (default False — skip during training)

    Returns:
        (train_loader, val_loader, test_loader) — val_loader / test_loader may be None
        if no data is found.
    """
    data_config = config.get('data', {})
    train_config = config.get('train', {})

    base_path = data_config.get('base_path')
    jnr_start = data_config.get('jnr_start', 0)
    jnr_end = data_config.get('jnr_end', 20)
    jnr_step = data_config.get('jnr_step', 1)
    # Force num_workers=0: PersistenceDataset loads all data into RAM,
    # so multiprocessing workers provide no I/O benefit and break on
    # Windows (spawn mode pickles the entire dataset, causing OOM).
    num_workers = 0
    pin_memory = data_config.get('pin_memory', True)
    image_size = data_config.get('image_size', 224)
    persistence_var_name = data_config.get('persistence_var_name', 'all_persistences')
    persistence_suffix = data_config.get('persistence_suffix', 'echo_persistences')
    batch_size = train_config.get('batch_size', 32)

    jamming_classes = config.get('jamming_classes', [])
    class_names = [jc['name'] if isinstance(jc, dict) else jc for jc in jamming_classes]

    use_feature_context = config.get('use_feature_context', False)
    feature_dims = config.get('n_ctx_per_domain', None)
    feature_norm_stats_path = config.get('feature_norm_stats_path', None)

    # Optional augmentation
    augmentation = None
    aug_config = config.get('augmentation', {})
    if aug_config.get('enabled', False):
        from multi.augmentation import STFTAugmentation
        augmentation = STFTAugmentation(aug_config)

    jnr_levels = list(range(jnr_start, jnr_end + 1, jnr_step))

    tokenizer = TokenizerWrapper(model_type="clip")
    collate = partial(
        collate_fn,
        tokenizer_fn=tokenizer,
        model_type="clip",
        use_feature_context=use_feature_context,
    )

    train_datasets = []
    val_datasets = []
    test_datasets = []

    for jnr in jnr_levels:
        data_folder = os.path.join(base_path, f'JNR_+{jnr}')

        for split, dataset_list in [('train', train_datasets),
                                     ('val', val_datasets),
                                     ('test', test_datasets)]:
            # Skip test split during training to avoid unnecessary loading
            if split == 'test' and not load_test:
                continue

            persistence_file = os.path.join(data_folder, f'{split}_{persistence_suffix}.mat')
            metadata_file = os.path.join(data_folder, f'{split}_echo_metadata.json')

            if not os.path.exists(persistence_file) or not os.path.exists(metadata_file):
                continue

            features_file = None
            if use_feature_context:
                feat_path = os.path.join(data_folder, f'{split}_echo_features.json')
                if os.path.exists(feat_path):
                    features_file = feat_path

            ds = PersistenceDataset(
                persistence_file=persistence_file,
                metadata_file=metadata_file,
                persistence_var_name=persistence_var_name,
                class_names=class_names,
                image_size=image_size,
                apply_clip_norm=True,
                augmentation=augmentation,
                features_file=features_file,
                feature_dims=feature_dims,
                use_feature_context=use_feature_context,
            )
            dataset_list.append(ds)

    # Build concatenated datasets
    train_dataset = ConcatDataset(train_datasets) if train_datasets else None
    val_dataset = ConcatDataset(val_datasets) if val_datasets else None
    test_dataset = ConcatDataset(test_datasets) if test_datasets else None

    # Create DataLoaders
    train_loader = None
    val_loader = None
    test_loader = None

    if train_dataset is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate,
        )
        print(f"Train: {len(train_dataset)} samples across {len(train_datasets)} JNR levels")

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate,
        )
        print(f"Val:   {len(val_dataset)} samples across {len(val_datasets)} JNR levels")

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate,
        )
        print(f"Test:  {len(test_dataset)} samples across {len(test_datasets)} JNR levels")

    num_classes = len(class_names)
    return train_loader, val_loader, test_loader, num_classes, jnr_levels


# ===========================================================================
# AblationDataset — multi-modal data loading for ablation study
# ===========================================================================

class AblationDataset(Dataset):
    """Dataset supporting three ablation modes for persistence vs STFT comparison.

    Modes:
        "persistence"  — only *_echo_persistences.mat, 3× persistence
                         (output identical to PersistenceDataset)
        "stft"         — only *_echo_stfts.mat, 3× STFT magnitude
        "fusion"       — both files, channels = [persistence, stft_mag, stft_phase]
    """

    def __init__(
        self,
        persistence_file: str = None,
        metadata_file: str = None,
        persistence_var_name: str = "all_persistences",
        stft_file: str = None,
        stft_var_name: str = "all_stfts",
        class_names: list = None,
        image_size: int = 224,
        apply_clip_norm: bool = True,
        augmentation=None,
        ablation_mode: str = "persistence",
        # STFT normalization
        normalize_mode: str = "per_sample",
        normalize_method: str = "p99",
        # Global normalization stats (only used when normalize_mode="global")
        normalization_stats: dict = None,
    ):
        super().__init__()

        self.ablation_mode = ablation_mode
        self.image_size = image_size
        self.apply_clip_norm = apply_clip_norm
        self.augmentation = augmentation
        self.normalize_mode = normalize_mode
        self.normalize_method = normalize_method

        # ---- Validate mode ----
        if ablation_mode not in ("persistence", "stft", "fusion"):
            raise ValueError(f"Unknown ablation_mode: {ablation_mode}. "
                             f"Must be 'persistence', 'stft', or 'fusion'.")

        # ---- Load persistence data (needed for "persistence" and "fusion") ----
        self._all_persistences = None
        if ablation_mode in ("persistence", "fusion"):
            if persistence_file is None or not os.path.exists(persistence_file):
                raise ValueError(f"Persistence file required for mode '{ablation_mode}': "
                                 f"{persistence_file}")
            with h5py.File(persistence_file, 'r') as f:
                raw = f[persistence_var_name][:]
                # HDF5 stores MATLAB column-major: [power_bins, freq, samples]
                # Transpose to [samples, freq, power_bins]
                axes = tuple(range(raw.ndim - 1, -1, -1))
                self._all_persistences = np.transpose(raw, axes=axes).astype(np.float32)
            self.num_samples = self._all_persistences.shape[0]

        # ---- Load STFT data (needed for "stft" and "fusion") ----
        self._all_stfts = None
        if ablation_mode in ("stft", "fusion"):
            if stft_file is None or not os.path.exists(stft_file):
                raise ValueError(f"STFT file required for mode '{ablation_mode}': "
                                 f"{stft_file}")
            with h5py.File(stft_file, 'r') as f:
                self._all_stfts = f[stft_var_name][()]
            num_stft = self._all_stfts.shape[2]
            if ablation_mode == "stft":
                self.num_samples = num_stft
            elif self.num_samples != num_stft:
                raise ValueError(
                    f"Sample count mismatch: persistence={self.num_samples}, "
                    f"stft={num_stft}. Both files must have the same number of samples."
                )

        # ---- STFT normalization stats (global mode only) ----
        if normalization_stats is None:
            normalization_stats = {
                'real_max': 450.0, 'imag_max': 450.0, 'mag_max': 455.0,
            }
        self.norm_scale_real = normalization_stats.get('real_max', 450.0)
        self.norm_scale_imag = normalization_stats.get('imag_max', 450.0)
        self.norm_scale_mag = normalization_stats.get('mag_max', 455.0)

        # ---- Load metadata ----
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self._metadata = json.load(f)

        # ---- Build labels ----
        self.class_names = class_names or []
        self.labels = self._build_labels_from_metadata()

        # ---- CLIP norm ----
        if apply_clip_norm:
            from torchvision.transforms import Normalize
            self.clip_norm = Normalize(mean=CLIP_MEAN, std=CLIP_STD)
        else:
            self.clip_norm = None

    # ------------------------------------------------------------------
    # Label building (identical to PersistenceDataset)
    # ------------------------------------------------------------------

    def _build_labels_from_metadata(self) -> np.ndarray:
        num_classes = len(self.class_names)
        labels = np.zeros((self.num_samples, num_classes), dtype=np.float32)
        for i, meta in enumerate(self._metadata):
            jam_types = meta.get('jam_types', [])
            if isinstance(jam_types, str):
                jam_types = [jam_types]
            for jt in jam_types:
                if jt in self.class_names:
                    j = self.class_names.index(jt)
                    labels[i, j] = 1.0
        return labels

    # ------------------------------------------------------------------

    def __len__(self):
        return self.num_samples

    # ------------------------------------------------------------------
    # Per-mode __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, index):
        if self.ablation_mode == "persistence":
            tensor = self._get_persistence_item(index)
        elif self.ablation_mode == "stft":
            tensor = self._get_stft_item(index)
        elif self.ablation_mode == "fusion":
            tensor = self._get_fusion_item(index)
        else:
            raise ValueError(f"Unknown ablation_mode: {self.ablation_mode}")

        # CLIP normalization
        if self.clip_norm is not None:
            tensor = self.clip_norm(tensor)

        # Optional augmentation
        if self.augmentation is not None:
            tensor = self.augmentation(tensor, None)

        # Label & metadata
        label = torch.from_numpy(self.labels[index])
        meta = self._metadata[index] if index < len(self._metadata) else {}

        return tensor, label, meta

    # ------------------------------------------------------------------
    # Mode: persistence (identical to PersistenceDataset.__getitem__)
    # ------------------------------------------------------------------

    def _get_persistence_item(self, index: int) -> torch.Tensor:
        persistence = self._all_persistences[index]  # (H, W)

        tensor = torch.from_numpy(
            np.stack([persistence, persistence, persistence], axis=0)
        ).float()

        if tensor.shape[1] != self.image_size or tensor.shape[2] != self.image_size:
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

        return tensor

    # ------------------------------------------------------------------
    # Mode: stft (follows multi/data.py STFTDataset)
    # ------------------------------------------------------------------

    def _get_stft_item(self, index: int) -> torch.Tensor:
        raw_stft = self._all_stfts[:, :, index]  # structured complex64
        stft_complex = raw_stft['real'] + 1j * raw_stft['imag']

        if self.normalize_mode == 'per_sample':
            mag = np.abs(stft_complex)
            if self.normalize_method == 'max':
                ref = np.max(mag)
            elif self.normalize_method == 'p95':
                ref = np.percentile(mag, 95)
            else:  # p99
                ref = np.percentile(mag, 99)
            if ref > 0:
                stft_complex = stft_complex / ref
            stft_mag = np.abs(stft_complex).T
        else:
            # Global normalization
            stft_mag = np.abs(stft_complex).T
            stft_mag = np.clip(stft_mag, 0, self.norm_scale_mag) / self.norm_scale_mag

        # Stack 3× magnitude
        tensor = torch.from_numpy(
            np.stack([stft_mag, stft_mag, stft_mag], axis=0)
        ).float()

        if tensor.shape[1] != self.image_size or tensor.shape[2] != self.image_size:
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

        return tensor

    # ------------------------------------------------------------------
    # Mode: fusion — [persistence, stft_mag, stft_phase]
    # ------------------------------------------------------------------

    def _get_fusion_item(self, index: int) -> torch.Tensor:
        # Channel 1: Persistence spectrum (already [0,1])
        persistence = self._all_persistences[index]  # (H, W)
        ch_persistence = torch.from_numpy(persistence).float().unsqueeze(0)  # (1, H, W)

        # Channel 2 & 3: STFT magnitude + phase
        raw_stft = self._all_stfts[:, :, index]
        stft_complex = raw_stft['real'] + 1j * raw_stft['imag']

        # Per-sample normalize STFT magnitude
        mag = np.abs(stft_complex)
        if self.normalize_mode == 'per_sample':
            if self.normalize_method == 'max':
                ref = np.max(mag)
            elif self.normalize_method == 'p95':
                ref = np.percentile(mag, 95)
            else:  # p99
                ref = np.percentile(mag, 99)
            if ref > 0:
                stft_complex = stft_complex / ref
            stft_mag = np.abs(stft_complex).T
        else:
            stft_mag = np.abs(stft_complex).T
            stft_mag = np.clip(stft_mag, 0, self.norm_scale_mag) / self.norm_scale_mag

        # Phase normalized to [0, 1]
        phase = np.angle(stft_complex).T  # (H, W), range [-π, π]
        phase_norm = (phase + np.pi) / (2.0 * np.pi)  # range [0, 1]

        ch_stft_mag = torch.from_numpy(stft_mag).float().unsqueeze(0)      # (1, H, W)
        ch_stft_phase = torch.from_numpy(phase_norm).float().unsqueeze(0)  # (1, H, W)

        # Independently resize each channel to 224×224
        channels = []
        for ch in [ch_persistence, ch_stft_mag, ch_stft_phase]:
            if ch.shape[1] != self.image_size or ch.shape[2] != self.image_size:
                ch = torch.nn.functional.interpolate(
                    ch.unsqueeze(0),
                    size=(self.image_size, self.image_size),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0)
            channels.append(ch)

        tensor = torch.cat(channels, dim=0)  # (3, H, W)
        return tensor


# ---------------------------------------------------------------------------
# Ablation collate function
# ---------------------------------------------------------------------------

def collate_fn_ablation(
    batch,
    tokenizer_fn=None,
):
    """Collate ablation dataset samples. Same signature as persistence collate_fn."""
    images, labels, metas = zip(*[
        (item[0], item[1], item[2]) for item in batch
    ])
    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)

    texts = [generate_text_descriptions(meta, style='class_only') for meta in metas]

    if tokenizer_fn is not None:
        text_tokens = tokenizer_fn(texts)
    else:
        import clip
        text_tokens = clip.tokenize(texts, truncate=True)

    return images, None, text_tokens, labels, texts, list(metas)


# ---------------------------------------------------------------------------
# Ablation DataLoader factory
# ---------------------------------------------------------------------------

def create_ablation_dataloaders(config: dict, load_test: bool = False):
    """Create train/val/test DataLoaders for ablation study.

    Reads config.ablation.mode to determine data source:
        "persistence" — only echo_persistences.mat
        "stft"        — only echo_stfts.mat
        "fusion"      — both files, 3 channels [persistence, stft_mag, stft_phase]

    Args:
        config: full YAML config dict
        load_test: whether to load test split (default False — skip during training)

    Returns:
        (train_loader, val_loader, test_loader, num_classes, jnr_levels)
    """
    ablation_config = config.get('ablation', {})
    ablation_mode = ablation_config.get('mode', 'persistence')

    data_config = config.get('data', {})
    train_config = config.get('train', {})

    base_path = data_config.get('base_path')
    jnr_start = data_config.get('jnr_start', 0)
    jnr_end = data_config.get('jnr_end', 20)
    jnr_step = data_config.get('jnr_step', 1)
    num_workers = 0  # preloaded data — no worker benefit, avoids Windows pickle OOM
    pin_memory = data_config.get('pin_memory', True)
    image_size = data_config.get('image_size', 224)
    batch_size = train_config.get('batch_size', 32)

    # Persistence config
    persistence_var_name = data_config.get('persistence_var_name', 'all_persistences')
    persistence_suffix = data_config.get('persistence_suffix', 'echo_persistences')

    # STFT config
    stft_suffix = data_config.get('stft_suffix', 'echo_stfts')
    stft_var_name = data_config.get('stft_var_name', 'all_stfts')

    # Normalization config
    normalize_mode = data_config.get('normalize_mode', 'per_sample')
    normalize_method = data_config.get('normalize_method', 'p99')
    normalization_stats = config.get('normalization_stats', None)

    jamming_classes = config.get('jamming_classes', [])
    class_names = [jc['name'] if isinstance(jc, dict) else jc for jc in jamming_classes]

    # Optional augmentation
    augmentation = None
    aug_config = config.get('augmentation', {})
    if aug_config.get('enabled', False):
        from multi.augmentation import STFTAugmentation
        augmentation = STFTAugmentation(aug_config)

    jnr_levels = list(range(jnr_start, jnr_end + 1, jnr_step))

    tokenizer = TokenizerWrapper(model_type="clip")
    collate = partial(collate_fn_ablation, tokenizer_fn=tokenizer)

    train_datasets = []
    val_datasets = []
    test_datasets = []

    for jnr in jnr_levels:
        data_folder = os.path.join(base_path, f'JNR_+{jnr}')

        for split, dataset_list in [('train', train_datasets),
                                     ('val', val_datasets),
                                     ('test', test_datasets)]:
            # Skip test split during training to avoid unnecessary loading
            if split == 'test' and not load_test:
                continue

            metadata_file = os.path.join(data_folder, f'{split}_echo_metadata.json')
            if not os.path.exists(metadata_file):
                continue

            # Persistence file (only for persistence and fusion modes)
            persistence_file = None
            if ablation_mode in ('persistence', 'fusion'):
                pf = os.path.join(data_folder, f'{split}_{persistence_suffix}.mat')
                if not os.path.exists(pf):
                    continue
                persistence_file = pf

            # STFT file (only for stft and fusion modes)
            stft_file = None
            if ablation_mode in ('stft', 'fusion'):
                sf = os.path.join(data_folder, f'{split}_{stft_suffix}.mat')
                if not os.path.exists(sf):
                    continue
                stft_file = sf

            ds = AblationDataset(
                persistence_file=persistence_file,
                metadata_file=metadata_file,
                persistence_var_name=persistence_var_name,
                stft_file=stft_file,
                stft_var_name=stft_var_name,
                class_names=class_names,
                image_size=image_size,
                apply_clip_norm=True,
                augmentation=augmentation,
                ablation_mode=ablation_mode,
                normalize_mode=normalize_mode,
                normalize_method=normalize_method,
                normalization_stats=normalization_stats,
            )
            dataset_list.append(ds)

    # Build concatenated datasets
    train_dataset = ConcatDataset(train_datasets) if train_datasets else None
    val_dataset = ConcatDataset(val_datasets) if val_datasets else None
    test_dataset = ConcatDataset(test_datasets) if test_datasets else None

    # Create DataLoaders
    train_loader = None
    val_loader = None
    test_loader = None

    if train_dataset is not None:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate,
        )
        print(f"[Ablation/{ablation_mode}] Train: {len(train_dataset)} samples "
              f"across {len(train_datasets)} JNR levels")

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate,
        )
        print(f"[Ablation/{ablation_mode}] Val:   {len(val_dataset)} samples "
              f"across {len(val_datasets)} JNR levels")

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate,
        )
        print(f"[Ablation/{ablation_mode}] Test:  {len(test_dataset)} samples "
              f"across {len(test_datasets)} JNR levels")

    num_classes = len(class_names)
    return train_loader, val_loader, test_loader, num_classes, jnr_levels
