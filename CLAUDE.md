# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of OpenAI's CLIP repurposed for **Compositional Zero-Shot Learning (CZSL) on radar jamming signals**. The original `clip/` package (OpenAI's CLIP model + tokenizer) is used as a frozen text encoder; the surrounding code trains encoders on radar data and evaluates zero-shot generalization to *unseen jamming-type combinations* (e.g. the model trains on single jamming types and on some pairs, then is tested on held-out pairs like `[DFTJ, AJ]`).

There are three modality-specific model families, each in its own top-level package, that all share the same CLIP text encoder + contrastive-learning interface:

- **`multi/`** — 2D STFT spectrogram images fed to CLIP ViT / MultiShape-Patch-ViT / ResNet18 visual encoders. The largest package; contains shared infra (data, loss, text templates, aug) reused by the others.
- **`persistence/`** — 2D persistence-spectrum images fed to a standard CLIP ViT visual encoder. Has an `ablation.mode` switch (`persistence` | `stft` | `fusion`).
- **`conformer_1d/`** — raw 1D I/Q time-domain signals fed to a from-scratch Conformer / ResNet1D / CNN1D encoder. No CLIP vision tower; only CLIP's text encoder is used.
- **`moe/`** — evaluation-only Mixture-of-Experts that fuses a trained persistence model + a trained conformer model at prediction time (deception classes from Conformer, suppression classes from Persistence).

## Commands

All training/eval scripts are run from the repo root with `python -m <package>.<module> --config <path>`. The `--config` YAML is the single source of truth for model architecture, backbone, loss, data paths, and CZSL seen/unseen splits.

```bash
# 1D time-domain (Conformer / ResNet1D / CNN1D — selected by model.backbone)
python conformer_1d/train_conformer.py  --config conformer_1d/config_1d.yaml [--resume CHKPT] [--debug]
python conformer_1d/evaluate_conformer.py --checkpoint CHKPT --mode zero_shot|by_combination|by_jnr [--backbone conformer|resnet1d|cnn1d]

# 2D persistence spectrum
python persistence/train.py    --config persistence/config.yaml [--resume CHKPT] [--debug]
python persistence/evaluate.py --checkpoint CHKPT --mode all|by_jnr|zero_shot --split test [--visualize]

# 2D STFT (multi/) — one train script per backbone, all driven by multi/config.yaml
python -m multi.train_czsl           --config multi/config.yaml   # plain CLIP ViT / RN
python -m multi.train_multishape_vit  --config multi/config.yaml   # multi-patch-shape ViT
python -m multi.train_resnet18        --config multi/config.yaml   # ResNet18 dual-branch
python -m multi.train_dual_branch     --config multi/config.yaml   # CLIP deception+suppression branches
python -m multi.train_unified         --config multi/config.yaml   # config-backbone-dispatching trainer
python -m multi.evaluate_czsl         --checkpoint CHKPT --mode zero_shot|by_jnr|by_combination [--split test] [--visualize]
python -m multi.inference_jnr         --checkpoint CHKPT --jnr 20 --jam_type ISRJ

# MoE fusion eval (needs BOTH a persistence + a conformer checkpoint)
python -m moe.evaluate_hybrid --checkpoint_persistence CHKPT --checkpoint_conformer CHKPT --mode all --split test --output_dir results/moe_hybrid
```

Common flags: `--debug` prints first-batch shapes/labels/texts; `--resume` loads optimizer+scheduler+epoch from a checkpoint. Checkpoints are written under `checkpoints/<prefix>/` with names `<prefix>_best_model.pt` / `<prefix>_latest_checkpoint.pt`, where the prefix is the backbone (`conformer`, `resnet1d`, `cnn1d`, `persistence`, `multishape_vit`, …) — see `BACKBONE_CONFIG` in `conformer_1d/train_conformer.py`.

### Dependencies and tests

- `pip install -r requirements.txt` (torch, torchvision, ftfy, regex, tqdm). Code also uses `h5py`, `scikit-learn`, `matplotlib`, `seaborn`, `wandb`, `pyyaml`, `umap-learn` — these are imported lazily and warn on absence rather than being in requirements.txt.
- Original CLIP sanity test (JIT vs eager parity, downloads real CLIP weights): `pytest tests/test_consistency.py`. There are no tests for the CZSL code itself.
- CLIP weights are downloaded to `~/.cache/clip` on first `clip.load(...)`.

## Architecture

### Shared design: signal encoder + frozen CLIP text encoder + contrastive loss

Every model class follows the same contract, enforced by a `Base*` class or convention:
- `forward(...)` returns `(signal_features, text_features)`, **both already L2-normalized** (B, 512).
- `encode_image()` / `encode_signal()` is an alias used by evaluators so `model.encode_image(x)` works regardless of modality.
- `cache_text_features(...)` pre-encodes text for every seen class and every seen/unseen combination, stored in `_text_features_cache` / `_combination_features_cache` / `_combination_names`.
- `zero_shot_predict(...)` ranks cached combination text features against a signal's features via `logit_scale * (sig @ text.T)` and returns top-k names.
- The model exposes `model.model.logit_scale` (the 1D base class aliases `model = self` for this reason) — evaluators reach in for the temperature.

The text side is a frozen (`freeze_text: true`) CLIP text transformer; only `logit_scale` and (optionally) the last few vision layers are trainable. Trainable vision layers are controlled by `vision_layers_unfreeze`.

### The config is the architecture

`model.backbone` and the `model.*` / `conformer.*` sub-dicts dispatch to a `create_*_model(config, device)` factory. There is no code-level registration — each train script either hard-codes its factory (`train_multishape_vit.py`) or reads `config["model"]["backbone"]` (`train_unified.py`, `conformer_1d/create_1d_model`). To add or switch a backbone, change the config and use the matching train script.

`loss.type` in the same YAML dispatches via `multi/loss.py:create_loss_function` — supported values: `infonce`, `label_aware_infonce`, `multilabel_infonce`, `multilabel_contrastive`, `asymmetric`, `bce`, `focal`, `czsl_contrastive`. The default across configs is `multilabel_infonce` (SigLIP-style sigmoid with IoU similarity). Trainers branch on `isinstance(loss_fn, MultiLabelSigmoidLoss | LabelAwareInfoNCELoss)` to decide the forward path; standard `infonce` takes a separate per-batch diagonal-target path.

### Data pipeline

Radar data lives outside this repo (see `data.base_path` in each config — paths like `D:\VScode\Jamming_signal_simulation\output\...`, **not** the `data/` dir here, which is gitignored). Layout consumed by the loaders:

```
<base_path>/JNR_+<n>/<split>_<suffix>.mat      # HDF5, structured complex64
<base_path>/JNR_+<n>/<split>_echo_metadata.json
```

`<suffix>` is `echo_stfts` (multi/persistence STFT mode), `echo_persistences` (persistence mode), or `echo_times` (1D time domain). `jnr_start:jnr_end:jnr_step` selects which `JNR_+n` folders to include. STFT complex spectrograms are converted to 3-channel (real/imag/mag) RGB and CLIP-normalized (`multi/data.py`); 1D signals are read as I/Q channel pairs.

Collate produces a 7-tuple `(images, time_signals, text_tokens, labels, texts, metas, features_dict)` — trainers unpack defensively by length since not every modality/ablation emits all fields. `features_dict` carries per-sample physical features (I/O bispectrum/wavelet/etc.) only when `use_feature_context: true`.

### CZSL seen/unseen splits

Each config's `czsl.seen_combinations` / `czsl.unseen_combinations` (lists of 1- or 2-element `[jam_type, ...]` lists) partition the jamming-type combination space. Singles are always seen (used to build text feature caches); unseen pairs are evaluated only at test time via `zero_shot_predict`. `use_combinations`, `max_combination_size`, and `zero_shot.threshold` control inference. `use_translation` toggles whether text prompts use short abbreviations or full English names (`label_translations` dict) — this is an ablation knob that affects only the text fed to the encoder.

### Optional CoOp-style feature-conditioned context

`use_feature_context: true` enables a `FeatureConditionedPromptLearner` (`multi/prompt_learner.py`) that maps 22-dim physical signal features (distributed across time/freq/bispectrum/wavelet/statistical domains per `n_ctx_per_domain`) to learnable context tokens injected into the frozen text transformer. When active, `cache_text_features` is skipped and context vectors are computed per-sample. `prompt_lr` is a separate, ~200× higher LR for these parameters. Feature stats for normalization are in `multi/feature_normalization_stats.json`.

### MoE fusion (moe/)

`moe/evaluate_hybrid.py` is evaluation-only — it loads a persistence checkpoint and a conformer checkpoint, runs both on the same data, and combines per-class predictions using the deception/suppression class split (`deception_classes` / `suppression_classes`). Conformer is trusted for deception classes, Persistence for suppression classes. Two fusion modes: `zero_shot` (top-1 fusion) and `by_combination` (per-class probability splicing). It reuses `PersistenceDataset` and `TimeSignalDataset` directly and is driven by `moe/config.yaml`.

## Conventions and gotchas

- **Stale legacy entry points:** `multi/train.py` and `multi/evaluate.py` import from a `czsl.*` package that does not exist in this repo. They are leftovers from a rename. The live multi/ entry points are `train_czsl.py` / `evaluate_czsl.py` (and the per-backbone variants). Do not "fix" the `czsl` imports by creating a `czsl` package — prefer redirecting to the named multi/* scripts.
- **`data.base_path` differs per config** and points to absolute Windows paths outside the repo. Persistence's config also has a duplicated `seen_combinations:` key (a YAML quirk — only the second block is read). The MoE config uses a third `base_path`.
- **Comments are in Chinese** throughout; class/dict names are English. Match the surrounding language when editing.
- **Git:** `data/`, `results/`, `wandb/`, `*.pt` checkpoints, and `.claude/` are gitignored — checkpoints and generated figures live only on disk.
- ** Plot font:** evaluation scripts set `plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', ...]` for Chinese axis labels; keep this when adding plots that use Chinese text, or labels render as boxes.
- The repo root is on `sys.path` via a `sys.path.insert` snippet at the top of every train/eval/conform script, so `import clip`, `from multi.loss import ...`, etc. work regardless of CWD. New scripts that live in a subpackage should replicate this snippet.
