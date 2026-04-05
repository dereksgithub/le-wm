# Progress Log: LeWM Fork

## Fork Origin
- **Original repo:** LeWorldModel (LeWM) by Lucas Maes
- **Forked at:** commit `83f97d7` (2026-03-12, "Initial commit")
- **Fork author:** Derek Ding

---

## What the Original Repo Contained
- A JEPA-based world model (~15M params) using CLS-token pooled latents
- `ARPredictor` for temporal prediction, `SIGReg` for regularisation
- Hydra configs for training on PushT, TwoRoom, DMC, OGB datasets
- Evaluation configs with CEM and Adam solvers for 4 tasks (cube, pusht, reacher, tworoom)
- Single training variant only (CLS-token transformer predictor)

---

## Changes Made After Forking

### PR #1 — Device Fix (2026-03-23, by Haiyang Luo)
- Fixed hardcoded `cuda` to use `proj.device` in `module.py`
- Fixed README typos

### Commit `ac5592f` — Gitignore (2026-03-27)
- Added `__pycache__/` to `.gitignore`

### Commit `379e66c` — Main Work: Spatial PDE Hybrid (2026-03-27)
This is the bulk of the new work. Summary of everything added/changed:

#### 1. Three-Variant Architecture (`jepa.py`, `module.py`, `train.py`)
Refactored the codebase from a single CLS-token model to support **three trainable variants** under one entry point:

| Variant | Config key | Predictor | Representation |
|---|---|---|---|
| LeWM-CLS (baseline) | `cls_transformer` | `ARPredictor` | Pooled CLS token |
| Spatial-LeWM-Transformer | `spatial_transformer` | `SpatialTransformerPredictor` | Patch-token grid |
| Spatial-LeWM-PDE | `spatial_pde` | `PDEPredictor` | Patch-token grid + reaction-diffusion |

#### 2. New Neural Network Modules (`module.py`)
- **`SpatialTransformerPredictor`** — spatial-token transformer that mixes each frame's patch tokens before per-token temporal prediction
- **`PDEPredictor`** — reaction-diffusion belief-state model with:
  - Multi-scale Laplacian diffusion (learnable strength, configurable dilations)
  - Learned reaction terms via Conv2d
  - Action conditioning through gating
  - `ChannelRMSNorm` for spatial normalisation
  - Write gates for state updates
  - Configurable evolve steps (default 4) and dilations (1, 2, 4)
- **DDP-safe `SIGReg`** — added `sync_ddp` flag for distributed training

#### 3. JEPA Core Updates (`jepa.py`, +243/-153 lines effectively)
- Added spatial latent support with grid embeddings (`info["emb_grid"]`)
- New methods: `pool_grid()`, `predict_grid()`, `rollout_latents()`
- `encode()` extended to return both pooled and spatial representations
- Preserved external planning interface (`encode`, `predict`, `rollout`, `criterion`, `get_cost`)

#### 4. Training Loop Updates (`train.py`, +293/-183 lines effectively)
- New `build_world_model()` factory function that dispatches on variant
- Grid-based prediction loss for spatial variants
- Spatial diagnostics logging: `spatial_std`, `effective_rank`, `dead_channels`
- Updated `lejepa_forward()` for both grid and pooled code paths

#### 5. Configuration (`config/`)
- **New model configs:** `cls_transformer.yaml`, `spatial_transformer.yaml`, `spatial_pde.yaml`
- **New eval ablation configs:** `*_h1.yaml` for each task (history_size=1 ablations)
- **Updated defaults:** `history_size` bumped from 1 to 3 in all eval configs; default model set to `spatial_pde` in `lewm.yaml`
- PDE-specific hyperparams added to main training config (`pde_predictor` section)

#### 6. Tests (`tests/test_spatial_hybrid.py`, 205 lines)
- Parametrised tests covering all 3 variants
- Validates encoding, prediction, and training loss computation
- Confirms interface compatibility across variants

#### 7. Research Plan (`plan/research_plan.md`)
- Documents the three-variant strategy, experiment matrix, metrics, and acceptance criteria

#### 8. Notebook (`run_new_model.ipynb`, 426 lines)
- Notebook for running and experimenting with the new model variants

#### 9. Training Artifact
- One captured training run output at `outputs/2026-03-27/14-49-48/` (PushT + spatial_pde config)

---

## Current State
- All three variants are implemented and tested
- Default config points to `spatial_pde`
- One training run has been executed (PushT with spatial_pde)
- No evaluation results recorded yet
- Experiment matrix from the research plan has not been fully executed

## Next Steps (from research_plan.md)
- Run all 3 variants on PushT and TwoRoom for architecture selection
- Validate selected models on OGB-Cube and Reacher
- Compare rollout robustness, spatial diagnostics, and planning success
- Determine if PDE branch beats baselines per acceptance criteria
