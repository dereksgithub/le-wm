# Research Plan: Spatial Fluid-LeWM Hybrid

## Summary
- LeJEPA remains the foundation: LE-WM already uses predictive JEPA training plus SIGReg, and this branch upgrades SIGReg toward DDP-safe operation.
- FluidWorld is useful only with spatial latents. The current CLS-only predictor path cannot exploit PDE diffusion meaningfully, so this implementation moves the hybrid to patch-token grids.
- The repo now supports three trainable variants under one training entrypoint:
  - `LeWM-CLS`
  - `Spatial-LeWM-Transformer`
  - `Spatial-LeWM-PDE`

## Implemented Changes
- Preserved the external JEPA planning interface: `encode`, `predict`, `rollout`, `criterion`, and `get_cost`.
- Added dense encoder outputs for spatial variants:
  - `info["emb_grid"]` for projected patch-token grids
  - `info["emb"]` as pooled global embeddings derived from those grids
- Added two spatial predictors:
  - `SpatialTransformerPredictor` as the spatial-token transformer baseline
  - `PDEPredictor` as the reaction-diffusion belief-state model
- Kept the LeWM objective family:
  - multi-step latent prediction loss
  - pooled-embedding `SIGReg`
- Increased the default training horizon to `history_size=3`, `num_preds=3`.
- Updated eval configs to use `history_size=3` by default and added `*_h1.yaml` ablation configs with `history_size=1`.

## Experiment Matrix
- Train on PushT and TwoRoom first for fast architecture selection.
- Validate the selected models on OGB-Cube and Reacher.
- Compare:
  - `python train.py model=cls_transformer`
  - `python train.py model=spatial_transformer`
  - `python train.py model=spatial_pde`
- Use the same optimizer family, split logic, and data preprocessing for all three variants.

## Primary Metrics
- Long-horizon latent prediction error
- Rollout stability over multiple autoregressive steps
- Surprise under visual and physical perturbations
- Physical-probe quality
- Spatial diagnostics:
  - `spatial_std`
  - `effective_rank`
  - `dead_channels`

## Acceptance Criteria
- The PDE branch should beat both baselines on rollout robustness on at least two tasks.
- It should improve or match robustness diagnostics without obvious collapse.
- It should not materially degrade planning success; otherwise it remains a robustness-focused branch rather than the default replacement.
