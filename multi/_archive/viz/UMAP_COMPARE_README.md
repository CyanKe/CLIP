# umap_compare.py — CLIP 特征 UMAP 可视化工具

对比预训练与训练后 CLIP 模型对 STFT 样本的图像特征提取差异，通过 UMAP 降维 + 交互式 HTML 进行可视化分析。

## 框架结构

```
umap_compare.py
├── 数据加载      load_data()         → STFTDataset + ConcatDataset
├── 特征提取      extract_features()  → model.encode_image() → L2 normalize
├── 模型加载      load_pretrained_model()  /  load_trained_model()
├── UMAP 降维      umap.UMAP().fit_transform()
├── 静态图输出    plot_umap()         → PNG (暗背景, 双面板)
├── JSON 导出     build_umap_data()   → {pretrained, trained, categories, info}
│                 save_umap_data_json() → 纯 JSON 文件
├── HTML 生成     build_html()        → 自包含交互页面 (Plotly.js CDN)
│                 - 类别筛选 (checkbox)
│                 - 暗/亮主题切换
│                 - Pretrained/Trained Tab
│                 - STFT 点击预览 (模态框)
└── STFT 预览     extract_features()  → 按需保存 sample_XXXXXX_{raw,clip}.png
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--config` | `multi/config.yaml` | 配置文件路径 |
| `--checkpoint` | (必需) | 训练后模型 checkpoint 路径 |
| `--model-type` | `None` (从 config) | `clip` / `multishape_vit` |
| `--dir` | config `data.base_path` | 数据根目录 |
| `--dataset` | `test` | 数据划分: `train` / `val` / `test` |
| `--jnr` | config `jnr_start:end:step` | JNR 范围规格, 如 `"0:5:20"` |
| `--highlight` | `None` | 指定高亮的干扰类型, 如 `DFTJ` |
| `--n_neighbors` | `30` | UMAP n_neighbors |
| `--min_dist` | `0.3` | UMAP min_dist |
| `--seed` | `42` | 随机种子 |
| `--batch_size` | `64` | 特征提取批次大小 |
| `--output_dir` | `results` | 输出目录 |
| `--img_size` | `224` | 输入图像尺寸 |
| `--save-html` | `False` | 生成交互式 HTML + JSON |
| `--skip-pretrained` | `False` | 跳过预训练模型 |
| `--skip-trained` | `False` | 跳过训练后模型 |
| `--stft-dir` | `results/stft_preview` | STFT 预览图目录 |
| `--no-stft` | `False` | 不保存 STFT 预览图 |

## 输入 / 输出

### 输入
```
{base_path}/
  JNR_+0/
    test_echo_stfts.mat          # HDF5 structured complex64 STFT
    test_echo_metadata.json      # [{jam_types: [...], JNR: ...}, ...]
  JNR_+5/
    ...
```

### 输出 (`results/`)
```
results/
  umap_pretrained_test_JNR_0_1_20.png    # 预训练 UMAP 静态图
  umap_trained_test_JNR_0_1_20.png       # 训练后 UMAP 静态图
  umap_data_test_JNR_0_1_20.json         # UMAP 坐标 + 标签 + STFT 路径
  umap_view_test_JNR_0_1_20.html         # 交互式浏览器页面
  stft_preview/                           # (当 --save-html 且未 --no-stft)
    sample_000000_raw.png                # 原始幅度谱 (inferno 色表)
    sample_000000_clip.png               # CLIP 输入 (三通道平均)
    ...
```

### JSON 数据结构
```json
{
  "pretrained": {
    "DFTJ": {"x": [...], "y": [...], "jnr": [...], "idx": [...], "stft_raw": [...], "stft_clip": [...]},
    ...
  },
  "trained": { ... },
  "categories": {"single": ["DFTJ", ...], "composite": ["DFTJ+AJ", ...]},
  "colors": {"DFTJ": "#1f77b4", ...},
  "info": {"clip_model": "ViT-B/32", "dataset": "test", "jnr_tag": "0_1_20", "n_samples": 30975}
}
```

## 常用操作

```bash
# 只跑训练后模型, 生成 HTML + STFT 预览到自定义目录
python -m multi.umap_compare \
    --checkpoint checkpoints/multishape_vit_best.pt \
    --dataset test --jnr 0:1:20 \
    --save-html --skip-pretrained \
    --stft-dir results/my_stft_preview

# 使用已有 STFT 预览, 不重新生成
python -m multi.umap_compare \
    --checkpoint checkpoints/xxx.pt \
    --dataset test --jnr 10 \
    --save-html --skip-pretrained \
    --stft-dir results/my_stft_preview --no-stft

# 预训练 vs 训练后对比, 指定高亮类型
python -m multi.umap_compare \
    --checkpoint checkpoints/xxx.pt \
    --dataset test --jnr 0:5:20 \
    --save-html --highlight DFTJ

# 查看 HTML (浏览器打开即可, 无需服务器)
start results/umap_view_test_JNR_0_1_20.html
```

## HTML 交互功能

- **类别筛选**: 左侧 sidebar 勾选/取消 jamming type, 图表实时更新
- **主题切换**: 顶栏日月按钮切换暗/亮配色, 自动记忆
- **JSON 切换**: `Load JSON` 按钮可随时加载其他 JSON 数据文件
- **拖拽加载**: 直接拖 JSON 文件到页面
- **散点点击**: 弹出模态框, 直接显示原始幅度谱 + CLIP 输入版 STFT 图
- **图片保存**: 模态框中 Save 按钮下载当前 STFT 图
- **兼容旧 JSON**: 无 `stft_raw`/`stft_clip` 字段时自动按 `sample_{idx}_raw.png` 推导路径
