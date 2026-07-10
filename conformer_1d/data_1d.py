"""
Data loading for 1D Conformer: time-domain I/Q signals from HDF5.

Mirrors multi/data.py but reads from *_echo_times.mat instead of *_echo_stfts.mat.
"""

import json
import math
import os
import sys
from functools import partial

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# Reuse text template generation from multi/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from multi.text_templates import generate_text_descriptions, load_label_translations


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TimeSignalDataset(Dataset):
    """Load 1D I/Q time-domain signals from HDF5 files.

    Each sample: (2, time_seq_len) = [I_channel, Q_channel].

    Metadata is loaded from a companion JSON file with the same structure
    as STFTDataset (jam_types, JNR, etc.).
    """

    def __init__(
        self,
        time_file: str,
        metadata_file: str,
        time_var_name: str = "all_times",
        class_names: list = None,
        time_seq_len: int = 8000,
        features_file: str = None,
        feature_dims: dict = None,
        use_feature_context: bool = False,
    ):
        """
        Args:
            time_file: path to *_echo_times.mat (HDF5 with structured complex64)
            metadata_file: path to *_echo_metadata.json
            time_var_name: HDF5 variable name for time-domain data
            class_names: list of jamming class name strings
            time_seq_len: expected signal length (truncate/pad)
            features_file: optional path to *_echo_features.json (for CoOp-style context)
            feature_dims: dict mapping domain names to their feature dimensions
            use_feature_context: whether to load and return signal features
        """
        self.time_file = time_file
        self.time_var_name = time_var_name
        self.time_seq_len = time_seq_len
        self.class_names = class_names or []
        self.use_feature_context = use_feature_context

        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        if isinstance(metadata, list):
            self._metadata = metadata
            self.num_samples = len(metadata)
        elif isinstance(metadata, dict):
            # Handle dict-of-lists format: {sample_idx: {...}, ...}
            # Not expected from this pipeline, but be safe
            self._metadata = [metadata[k] for k in sorted(metadata.keys(), key=int)]
            self.num_samples = len(self._metadata)
        else:
            raise TypeError(f"Unexpected metadata format: {type(metadata)}")

        # Build labels
        self.labels = self._build_labels_from_metadata()

        # Preload time-domain data into memory (避免 HDF5 列切片磁盘 I/O)
        # 原始格式: (time_seq_len, num_samples), structured complex (real/imag)
        # 转换后: (num_samples, 2, time_seq_len), float32 [I, Q]
        with h5py.File(time_file, 'r') as f:
            raw_times = f[time_var_name][()]
            if raw_times.dtype.names and 'real' in raw_times.dtype.names:
                real = np.array(raw_times['real'], dtype=np.float32)  # (T, N)
                imag = np.array(raw_times['imag'], dtype=np.float32)  # (T, N)
                # Transpose to (N, 2, T) for fast sample indexing
                self._all_times = np.stack([real, imag], axis=1)      # (T, 2, N)
                self._all_times = np.transpose(self._all_times, (2, 1, 0))  # (N, 2, T)
                self._all_times = np.ascontiguousarray(self._all_times)
            else:
                raw = np.array(raw_times, dtype=np.float32)
                if raw.ndim == 2:
                    raw = raw.T  # (T, N) -> (N, T)
                    raw = raw[:, np.newaxis, :]  # (N, 1, T)
                    raw = np.repeat(raw, 2, axis=1)  # (N, 2, T) - duplicate as I/Q
                self._all_times = np.ascontiguousarray(raw)
        print(f"  Preloaded time data: {self._all_times.shape} ({self._all_times.nbytes / 1024**2:.1f} MB)")

        # Feature context (CoOp-style)
        self.feature_dims = feature_dims or {}
        self._features_data = None

        if self.use_feature_context and features_file and os.path.exists(features_file):
            with open(features_file, 'r') as f:
                self._features_data = json.load(f)
            print(f"Loaded features from {features_file}")

    # ------------------------------------------------------------------
    # Label building (identical logic to STFTDataset)
    # ------------------------------------------------------------------

    def _build_labels_from_metadata(self) -> np.ndarray:
        """Build multi-hot labels from metadata jam_types field.

        Compatible with both single-string ('CSJ') and list (['DFTJ', 'AJ']) formats.
        """
        if not self.class_names:
            return np.zeros((self.num_samples, 1), dtype=np.float32)

        labels = np.zeros((self.num_samples, len(self.class_names)), dtype=np.float32)
        name_to_idx = {name: i for i, name in enumerate(self.class_names)}

        for i, meta in enumerate(self._metadata):
            if i >= self.num_samples:
                break
            jam_types = meta.get('jam_types', [])
            if isinstance(jam_types, list):
                types_list = jam_types
            elif isinstance(jam_types, str):
                types_list = [jam_types] if jam_types and jam_types != 'None' else []
            else:
                types_list = []

            for jam_type in types_list:
                jam_type = jam_type.strip() if isinstance(jam_type, str) else str(jam_type)
                if jam_type in name_to_idx:
                    labels[i, name_to_idx[jam_type]] = 1.0

        return labels

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index: int):
        # 1. 从预加载的内存数组中取样本: (2, T) float32 [I, Q]
        time_signal = self._all_times[index]  # (2, T)

        # 2. Per-sample standardization
        mean = time_signal.mean(axis=-1, keepdims=True)
        std = time_signal.std(axis=-1, keepdims=True) + 1e-8
        time_signal = (time_signal - mean) / std

        # 3. Truncate or pad to time_seq_len
        T = time_signal.shape[1]
        if T > self.time_seq_len:
            time_signal = time_signal[:, :self.time_seq_len]
        elif T < self.time_seq_len:
            pad = np.zeros((2, self.time_seq_len - T), dtype=np.float32)
            time_signal = np.concatenate([time_signal, pad], axis=1)

        time_tensor = torch.from_numpy(time_signal).float()

        # Label
        label = torch.from_numpy(self.labels[index].copy()).float()

        # Metadata
        meta = self._metadata[index]

        # Features (optional)
        if self.use_feature_context and self._features_data is not None:
            features_dict = self._get_features(index)
            return time_tensor, label, meta, features_dict
        else:
            return time_tensor, label, meta

    def _get_features(self, index: int) -> dict:
        """Extract 22-D signal features for FeatureConditionedPromptLearner."""
        if self._features_data is None:
            return None

        feats = self._features_data[index]
        features_dict = {}

        # Group feature values by domain based on feature_dims config
        offset = 0
        for domain, dim in self.feature_dims.items():
            domain_feats = []
            for i in range(dim):
                key = f"{domain}_{i}"
                if key in feats:
                    domain_feats.append(float(feats[key]))
                else:
                    # Try reading as pre-built sub-dict
                    break
            if len(domain_feats) == dim:
                features_dict[domain] = torch.tensor(domain_feats, dtype=torch.float32)
            offset += dim

        # Fallback: if indexing by domain failed, try flat list
        if not features_dict and isinstance(feats, list):
            offset = 0
            for domain, dim in self.feature_dims.items():
                features_dict[domain] = torch.tensor(
                    feats[offset:offset + dim], dtype=torch.float32
                )
                offset += dim

        return features_dict if features_dict else None


_COLLATE_USE_TRANSLATION = False


def set_collate_use_translation(use_translation: bool) -> None:
    """Set whether collate feeds translated (full) class names into CLIP text."""
    global _COLLATE_USE_TRANSLATION
    _COLLATE_USE_TRANSLATION = bool(use_translation)


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn_conformer(batch, tokenizer_fn, model_type="clip", processor=None):
    """Collate time-signal samples into a batch.

    Returns a 7-tuple compatible with existing training code:
        (time_signals, None, text_tokens, labels, texts, metadata_list)

    The second element (None) is a placeholder for stft_images to keep the
    interface consistent with CZSLTrainer.
    """
    use_features = len(batch[0]) == 4

    if use_features:
        time_signals, labels, metas, features_list = zip(*batch)
    else:
        time_signals, labels, metas = zip(*batch)
        features_list = None

    # Stack tensors
    time_batch = torch.stack(time_signals, dim=0)  # (B, 2, T)
    labels_batch = torch.stack(labels, dim=0)      # (B, num_classes)

    # Generate text descriptions from metadata
    texts = [generate_text_descriptions(meta, style='class_only', use_translation=_COLLATE_USE_TRANSLATION) for meta in metas]

    # Tokenize
    if model_type == "siglip":
        text_tokens = tokenizer_fn(
            texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt"
        )
    else:
        text_tokens = tokenizer_fn(texts)

    metas_list = list(metas)

    if use_features and features_list is not None:
        # features_list is a tuple of dicts or None values
        features_batched = {}
        # Collect keys from first non-None feature dict
        sample_feats = next((f for f in features_list if f is not None), None)
        if sample_feats is not None:
            for domain in sample_feats.keys():
                vals = []
                for f in features_list:
                    vals.append(f[domain] if f is not None else torch.zeros_like(sample_feats[domain]))
                features_batched[domain] = torch.stack(vals, dim=0)
        else:
            features_batched = None
        return time_batch, None, text_tokens, labels_batch, texts, metas_list, features_batched
    else:
        return time_batch, None, text_tokens, labels_batch, texts, metas_list


# ---------------------------------------------------------------------------
# Tokenizer wrapper (reuses CLIP tokenizer)
# ---------------------------------------------------------------------------

class TokenizerWrapper:
    """Wraps CLIP tokenize as a callable for collate_fn."""

    def __init__(self, model_type="clip"):
        if model_type == "siglip":
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
        else:
            import clip
            self.tokenizer = clip.tokenize

    def __call__(self, texts):
        return self.tokenizer(texts)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_1d_dataloaders(config: dict):
    """Create train/val/test dataloaders for 1D time-domain data.

    Args:
        config: full config dict (from config_1d.yaml)

    Returns:
        train_loader, val_loader, test_loader
    """
    data_config = config['data']
    model_config = config['model']
    base_path = data_config['base_path']

    # Whether CLIP text comparison uses translated (full) class names.
    set_collate_use_translation(config.get('use_translation', False))
    jnr_start = data_config.get('jnr_start', 0)
    jnr_end = data_config.get('jnr_end', 20)
    jnr_step = data_config.get('jnr_step', 1)
    time_var_name = data_config.get('time_var_name', 'all_times')
    time_seq_len = data_config.get('time_seq_len', 8000)
    batch_size = config['train']['batch_size']
    num_workers = data_config.get('num_workers', 4)
    pin_memory = data_config.get('pin_memory', True)
    use_feature_context = config.get('use_feature_context', False)

    # Class names
    jamming_classes = config.get('jamming_classes', [])
    class_names = [jc['name'] if isinstance(jc, dict) else jc for jc in jamming_classes]
    print(f"Class names ({len(class_names)}): {class_names}")

    # Feature dims
    feature_dims = {}
    if use_feature_context:
        n_ctx_per_domain = config.get('n_ctx_per_domain', {})
        # Map domain names to their feature dimensions from prompt_learner conventions
        feature_dims = {
            'time': 5,
            'freq': 4,
            'bispectrum': 2,
            'wavelet': 8,
            'statistical': 3,
        }
        feature_dims = {k: v for k, v in feature_dims.items() if k in n_ctx_per_domain}

    # Collect datasets per JNR level
    jnr_values = list(range(jnr_start, jnr_end + 1, jnr_step))
    print(f"JNR levels: {jnr_values}")

    train_datasets, val_datasets, test_datasets = [], [], []

    for jnr in jnr_values:
        data_folder = os.path.join(base_path, f'JNR_+{jnr}')
        if not os.path.exists(data_folder):
            print(f"  Warning: {data_folder} not found, skipping")
            continue

        for split_name in ['train', 'val', 'test']:
            time_file = os.path.join(data_folder, f'{split_name}_echo_times.mat')
            metadata_file = os.path.join(data_folder, f'{split_name}_echo_metadata.json')
            features_file = os.path.join(data_folder, f'{split_name}_echo_features.json')

            if not os.path.exists(time_file) or not os.path.exists(metadata_file):
                print(f"  Warning: missing {split_name} data for JNR_+{jnr}, skipping")
                continue

            ds = TimeSignalDataset(
                time_file=time_file,
                metadata_file=metadata_file,
                time_var_name=time_var_name,
                class_names=class_names,
                time_seq_len=time_seq_len,
                features_file=features_file if os.path.exists(features_file) else None,
                feature_dims=feature_dims,
                use_feature_context=use_feature_context,
            )

            if split_name == 'train':
                train_datasets.append(ds)
            elif split_name == 'val':
                val_datasets.append(ds)
            else:
                test_datasets.append(ds)

    # Concat across JNR levels
    if not train_datasets:
        raise RuntimeError("No training datasets found!")
    train_ds = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_ds = ConcatDataset(val_datasets) if len(val_datasets) > 1 else (val_datasets[0] if val_datasets else None)
    test_ds = ConcatDataset(test_datasets) if len(test_datasets) > 1 else (test_datasets[0] if test_datasets else None)

    print(f"Dataset sizes — Train: {len(train_ds)}, Val: {len(val_ds) if val_ds else 0}, Test: {len(test_ds) if test_ds else 0}")

    # Tokenizer
    clip_model_name = model_config.get('clip_model', 'ViT-B/32')
    model_type = "siglip" if "siglip" in clip_model_name.lower() else "clip"
    tokenizer = TokenizerWrapper(model_type=model_type)

    # Collate function (must be pickleable for Windows multiprocessing)
    collate = partial(collate_fn_conformer, tokenizer_fn=tokenizer, model_type=model_type)

    # Create dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate,
    ) if val_ds else None
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate,
    ) if test_ds else None

    print("Dataloaders created.")
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing TimeSignalDataset...")

    test_time = r"D:\VSCODE\Jamming_signal_simulation\output\17fine\JNR_+0\train_echo_times.mat"
    test_meta = r"D:\VSCODE\Jamming_signal_simulation\output\17fine\JNR_+0\train_echo_metadata.json"

    if not os.path.exists(test_time):
        print(f"File not found: {test_time}")
        print("Skipping test — data path unavailable.")
    else:
        jamming_classes = [
            "DFTJ", "ISRJ", "AJ", "BJ", "SJ", "NCJ", "NPJ",
            "SMSPJ", "C&IJ", "NFMJ", "NPMJ", "NAMJ", "CSJ", "PJ",
        ]

        ds = TimeSignalDataset(
            time_file=test_time,
            metadata_file=test_meta,
            class_names=jamming_classes,
            time_seq_len=8000,
        )
        print(f"Dataset size: {len(ds)}")
        print(f"Labels shape: {ds.labels.shape}")

        # Load first sample
        signal, label, meta = ds[0]
        print(f"Signal shape: {signal.shape}")   # (2, 8000)
        print(f"Label shape: {label.shape}")      # (14,)
        print(f"Label: {label}")
        print(f"Meta jam_type: {meta.get('jam_types')}")
        print(f"Signal stats: mean={signal.mean():.4f}, std={signal.std():.4f}")
        print("Test passed!")
