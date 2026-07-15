# Code Audit Findings

Generated 2026-07-11. Issues ranked by severity.

---

## CRITICAL (7) — Will crash or silently produce wrong results

| # | File | Category | Description |
|---|---|---|---|
| 1 | [multi/train.py:33-35](multi/train.py#L33-L35) | **BUG** | ~~Imports from `czsl.model`…~~ **FIXED: Deleted the stale file.** |
| 2 | [multi/evaluate.py:31-32](multi/evaluate.py#L31-L32) | **BUG** | ~~Same `czsl.*` import issue.~~ **FIXED: Deleted the stale file.** |
| 3 | [persistence/config.yaml:147-151](persistence/config.yaml#L147-L151) | **BUG** | ~~Duplicated nested `seen_combinations:` key…~~ **FIXED by user.** |
| 4 | [multi/data.py:1149](multi/data.py#L1149) | **BUG** | ~~`__main__` test block unpacks 5 elements…~~ **FIXED: Updated unpack to 7-tuple `(images, _, text_tokens, labels, texts, metadata_list, _features)`.** |
| 5 | [multi/evaluate_original_clip.py:101](multi/evaluate_original_clip.py#L101) | **BUG** | ~~Batch unpack destructures 6 elements…~~ **FIXED: Added `*_` to absorb extras.** |
| 6 | [multi/train_feature_only.py:384-385](multi/train_feature_only.py#L384-L385) | **BUG** | ~~`all_preds.append(preds)` executed twice…~~ **FIXED: Removed duplicate append block.** |
| 7 | [multi/loss.py:795,884-938](multi/loss.py#L795) | **LOGIC** | ~~`MultiLabelSigmoidLoss` defined but no factory route…~~ **FIXED: Added `"multilabel_sigmoid"` and `"sigmoid"` dispatch routes in `create_loss_function`.** |

---

## HIGH (14) — Silently wrong behavior, or design flaws likely to cause bugs

| # | File | Category | Description |
|---|---|---|---|
| 8 | [multi/config.yaml:132-146](multi/config.yaml#L132-L146) | **BUG** | ~~Only 14 jamming classes defined…~~ **FIXED: Extended to 17 classes matching persistence; added ISCJ/ISDJ/MISRJ to deception group; expanded seen_combinations (17 singles) and unseen_combinations (8×9=72 pairs).** |
| 9 | [multi/model.py:677-747](multi/model.py#L677-L747) | **BUG** | ~~`create_czsl_model(config)` ignores `backbone`…~~ **FIXED: Refactored into `create_czsl_model` (dispatches on backbone) + `_create_vit_model` (plain ViT/RN). `multishape_vit` now correctly routes to MultiShapePatchViT; `resnet18` raises a clear error directing to `train_resnet18.py`.** |
| 10 | [multi/loss.py:884-938](multi/loss.py#L884-L938) | **BUG** | ~~`create_loss_function` accepts `model_type` but never uses it; else fallback crashes…~~ **FIXED: Added `"multilabel_sigmoid"` and `"sigmoid"` dispatch routes before the `else` block so the BCE fallback is never reached with valid configs. `model_type` kept in signature for backward compatibility (ignored).** |
| 11 | [multi/loss.py:911-913](multi/loss.py#L911-L913) | **BUG** | ~~`"multilabel_infonce"` dispatch omits `learnable_temperature` and `max_temperature`…~~ **FIXED: Added both params read from config.** |
| 12 | [conformer_1d/model_1d.py:154-163](conformer_1d/model_1d.py#L154-L163) | **BUG** | ~~`return_attn` detection relies on catching `TypeError` — fragile…~~ **FIXED: Replaced try/except with `inspect.signature` check — verifies `return_attn` is in the encoder's forward signature before calling, raising a clear `RuntimeError` if not.** |
| 13 | [multi/model.py:437-498](multi/model.py#L437-L498) | **LOGIC** | ~~`cache_text_features` only caches seen combos…~~ **FIXED: Added `unseen_combinations` parameter; when provided, unseen combo text features are also cached. Updated conformer_1d/model_1d.py and persistence/model.py with the same parameter.** |
| 14 | [multi/train_feature_only.py:227-234](multi/train_feature_only.py#L227-L234) | **BUG** | `_parse_features` nested loop logic — **NOT FIXED** (user indicated this file is not currently in use). |
| 15 | [conformer_1d/config_1d.yaml:52](conformer_1d/config_1d.yaml#L52), [persistence/config.yaml:49](persistence/config.yaml#L49), [moe/config.yaml:37](moe/config.yaml#L37) | **BUG** | Hardcoded absolute Windows paths — **LEFT AS-IS** per user request (environment-specific). |
| 16 | [persistence/config.yaml:42](persistence/config.yaml#L42) | **LOGIC** | `ablation.mode: "stft"` default — **LEFT AS-IS** per user request (post-test artifact). |
| 17 | [persistence/model.py:293-349](persistence/model.py#L293-L349) | **BUG** | ~~Due to YAML duplicate key: `seen_combinations` dict causes empty cache…~~ **FIXED: YAML duplicate key removed (same fix as #3). `unseen_combinations` parameter added for robustness.** |
| 18 | [conformer_1d/model_1d.py:126-128](conformer_1d/model_1d.py#L126-L128), [persistence/model.py:156-157](persistence/model.py#L156-L157) | **LOGIC** | `self.model` property returns `self` — **NOT FIXED** per user request (deferred, high blast radius). |
| 19 | [moe/evaluate_hybrid.py:67-68](moe/evaluate_hybrid.py#L67-L68) vs [moe/config.yaml:72-91](moe/config.yaml#L72-L91) | **REDUNDANCY** | Hardcoded class split + YAML dead data — **NOT FIXED** (code logic correct; YAML serves as documentation). |
| 20 | Across configs | **LOGIC** | ~~Inconsistent CZSL splits across modalities…~~ **PARTIALLY FIXED: MoE evaluator now loads checkpoint configs and warns if seen_combinations differ between Persistence and Conformer models.** |
| 21 | [conformer_1d/train_conformer.py:132-143](conformer_1d/train_conformer.py#L132-L143), [persistence/train.py:162-173](persistence/train.py#L162-L173) | **LOGIC** | ~~Training accuracy semantics differ…~~ **FIXED: Replaced Conformer's `_batch_class_accuracy` with `_label_aware_accuracy` matching multi/train_czsl.py's approach (pos_mask-based i2t + t2i accuracy). Persistence already used label-aware; now consistent.** |

---

## MEDIUM (12) — Redundancy, confusing design, one-off issues

| # | File | Category | Description |
|---|---|---|---|
| 22 | 6× train_*.py | **REDUNDANCY** | `load_config()` (3 lines) duplicated in 6 places. `create_optimizer_and_scheduler()` duplicated across 5 training scripts — ~150 near-identical lines. |
| 23 | [conformer_1d/train_conformer.py](conformer_1d/train_conformer.py) + [persistence/train.py](persistence/train.py) | **REDUNDANCY** | Training loop logic ~90% identical (batch unpack, forward, loss dispatch, backward, grad clip, checkpoint save) — ~200 duplicated lines. Any fix must be applied twice. |
| 24 | [multi/evaluate_czsl.py](multi/evaluate_czsl.py) + [multi/evaluate_multishape_vit.py](multi/evaluate_multishape_vit.py) | **REDUNDANCY** | Evaluator classes (zero-shot inference, by-combination, by-JNR, t-SNE, UMAP, ROC, confusion matrices) are structurally ~70% identical. Multishape evaluator imports utilities from `evaluate_czsl.py` but reimplements the evaluator. |
| 25 | [conformer_1d/evaluate_conformer.py](conformer_1d/evaluate_conformer.py) + [persistence/evaluate.py](persistence/evaluate.py) | **REDUNDANCY** | Similar duplication across packages — `ConformerEvaluator` and `PersistenceEvaluator` share identical plotting/metrics structure, differing only in `encode_*` call and YAML unwrap. |
| 26 | [multi/data.py:240-262](multi/data.py#L240-L262) + [train_feature_only.py:136-159](multi/train_feature_only.py#L136-L159) + [prompt_learner.py](multi/prompt_learner.py) | **REDUNDANCY** | `FEATURE_DOMAINS` (22 feature keys across 5 domains) duplicated in 3 files. Changing the schema requires updating 3 locations. |
| 27 | [multi/data.py:45-119](multi/data.py#L45-L119) | **REDUNDANCY** | `collate_fn` and `_collate_fn` implement identical logic — former hardcodes clip, latter is parameterized. Both exist side by side and are both used. |
| 28 | [persistence/model.py:335-348](persistence/model.py#L335-L348) | **LOGIC** | `_text_features_cache[:self.num_classes]` slice relies on the convention that singles are cached **first**. Pure convention with no enforcement — easy to break during refactoring. |
| 29 | [conformer_1d/train_conformer.py:118](conformer_1d/train_conformer.py#L118) | **BUG** | ~~`backbone_info.get("wandb_tags", [])` (plural)…~~ **FIXED: Changed to `"wandb_tag"` (singular) matching `BACKBONE_CONFIG` key.** |
| 35 | [multi/evaluate_czsl.py:1238-1240](multi/evaluate_czsl.py#L1238-L1240) | **REDUNDANCY** | ~~Double consecutive `plt.tight_layout()`…~~ **FIXED: Removed duplicate call.** |
| 38 | [persistence/evaluate.py:509](persistence/evaluate.py#L509) | **BUG** | ~~`np.trapezoid` requires numpy ≥ 2.0…~~ **FIXED: Added `getattr(np, 'trapezoid', getattr(np, 'trapz', None))` fallback in both persistence/evaluate.py and moe/evaluate_hybrid.py.** |
