# multi/ — 2D STFT CZSL

雷达干扰 STFT + CLIP/CZSL。目录已分层，避免根目录堆满实验脚本。

## 布局

```
multi/
  # 主线（日常训练 / 评估）
  config.yaml
  train_czsl.py
  evaluate_czsl.py
  data.py  model.py  loss.py  metrics_czsl.py
  text_templates.py  lora.py  prompt_learner.py  augmentation.py

  experiments/   # 平行实验（双分支 / MultiShape / ResNet / Unified …）
  tools/         # 数据预处理与工具
  _archive/      # 旧入口、notebook、可视化、一次性脚本
```

## 主线

```bash
# 仓库根目录执行
python -m multi.train_czsl    --config multi/config.yaml
python -m multi.evaluate_czsl --config multi/config.yaml
```

## 实验变体

```bash
python -m multi.experiments.train_dual_branch
python -m multi.experiments.train_multishape_vit
python -m multi.experiments.train_resnet18
python -m multi.experiments.train_unified
python -m multi.experiments.train_feature_only
python -m multi.experiments.evaluate_*   # 与上面对应
python -m multi.experiments.evaluate_original_clip
```

MultiShape 骨干实现：`multi.experiments.rectangular_patch_vit`。

## 工具

```bash
python -m multi.tools.preprocess_stft --config multi/config.yaml
python -m multi.tools.compute_normalization_stats
python -m multi.tools.compute_feature_stats --config multi/config.yaml
python -m multi.tools.inference_jnr --checkpoint CHKPT --jnr 20 --jam_type ISRJ
python -m multi.tools.data_split
```

## 归档

`_archive/` 不作为正式入口：

| 子目录 | 内容 |
|--------|------|
| `legacy/` | 已损坏的 `train.py` / `evaluate.py`（`czsl.*` 遗留） |
| `viz/` | UMAP / t-SNE / PCA 注意力等 |
| `notebooks/` | 探索用 notebook |
| `docs/` | 架构 html、增强说明 |
| `dev/` | stft_server、test_data_save |

运行示例：`python multi/_archive/viz/umap_compare.py ...`
