from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from einops import rearrange
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import (
    ARPredictor,
    Embedder,
    MLP,
    PDEPredictor,
    SIGReg,
    SpatialTransformerPredictor,
)
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


def subsample_rows(features, max_rows):
    if features.size(0) <= max_rows:
        return features

    step = max(1, features.size(0) // max_rows)
    return features[::step][:max_rows]


def compute_effective_rank(features, eps=1e-8):
    if features.size(0) < 2:
        return features.new_tensor(0.0)

    centered = features.float() - features.float().mean(dim=0, keepdim=True)
    cov = centered.T @ centered
    cov = cov / max(centered.size(0) - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(eps)
    probs = eigvals / eigvals.sum().clamp_min(eps)
    return torch.exp(-(probs * torch.log(probs.clamp_min(eps))).sum())


def compute_spatial_diagnostics(emb_grid, max_rank_samples=4096):
    with torch.no_grad():
        flat = rearrange(emb_grid.float(), "b t c h w -> (b t h w) c")
        flat = subsample_rows(flat, max_rank_samples)
        channel_std = flat.std(dim=0)
        diagnostics = {
            "spatial_std": emb_grid.float().std(dim=(-1, -2)).mean(),
            "effective_rank": compute_effective_rank(flat),
            "dead_channels": (channel_std < 1e-3).float().mean(),
        }
    return diagnostics


def build_world_model(cfg, encoder):
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim
    use_spatial_latents = cfg.wm.variant != "cls_transformer"

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

    if cfg.wm.variant == "cls_transformer":
        predictor = ARPredictor(
            num_frames=cfg.wm.history_size,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            **cfg.predictor,
        )
        projector = MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        )
        pred_proj = MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        )
        grid_projector = None

    elif cfg.wm.variant == "spatial_transformer":
        predictor = SpatialTransformerPredictor(
            num_frames=cfg.wm.history_size,
            grid_size=cfg.wm.grid_size,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            **cfg.predictor,
        )
        projector = MLP(
            input_dim=embed_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        )
        pred_proj = MLP(
            input_dim=embed_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        )
        grid_projector = MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        )

    elif cfg.wm.variant == "spatial_pde":
        predictor = PDEPredictor(
            input_dim=embed_dim,
            action_dim=embed_dim,
            output_dim=embed_dim,
            **cfg.pde_predictor,
        )
        projector = MLP(
            input_dim=embed_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        )
        pred_proj = MLP(
            input_dim=embed_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        )
        grid_projector = MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        )

    else:
        raise ValueError(f"Unknown wm.variant: {cfg.wm.variant}")

    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
        grid_projector=grid_projector,
        use_spatial_latents=use_spatial_latents,
        history_size=cfg.wm.history_size,
    )


def lejepa_forward(self, batch, stage, cfg):
    """Encode observations, predict future states, and compute losses."""
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)

    emb = output["emb"]
    act_emb = output["act_emb"]
    emb_grid = output.get("emb_grid")

    ctx_emb = emb[:, :ctx_len]
    tgt_emb = emb[:, ctx_len : ctx_len + n_preds]
    ctx_grid = emb_grid[:, :ctx_len] if emb_grid is not None else None

    rollout = self.model.rollout_latents(
        ctx_emb,
        act_emb[:, : ctx_len + n_preds - 1],
        num_preds=n_preds,
        emb_grid_context=ctx_grid,
        history_size=ctx_len,
    )

    output["pred_emb"] = rollout["pred_emb"]
    output["pred_loss"] = F.mse_loss(rollout["pred_emb"], tgt_emb)

    if emb_grid is not None and "pred_emb_grid" in rollout:
        tgt_grid = emb_grid[:, ctx_len : ctx_len + n_preds]
        output["grid_pred_loss"] = F.mse_loss(rollout["pred_emb_grid"], tgt_grid)
        output["pred_loss"] = output["grid_pred_loss"]
        if cfg.wm.get("log_spatial_metrics", True):
            output.update(
                compute_spatial_diagnostics(
                    emb_grid,
                    max_rank_samples=cfg.wm.get("max_metric_samples", 4096),
                )
            )

    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    log_keys = {
        "pred_loss",
        "grid_pred_loss",
        "sigreg_loss",
        "loss",
        "spatial_std",
        "effective_rank",
        "dead_channels",
    }
    metrics = {f"{stage}/{k}": v.detach() for k, v in output.items() if k in log_keys}
    self.log_dict(metrics, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]

    with open_dict(cfg):
        cfg.wm.grid_size = cfg.img_size // cfg.patch_size
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )

    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        shuffle=True,
        drop_last=True,
        generator=rnd_gen,
    )
    val = torch.utils.data.DataLoader(
        val_set,
        **cfg.loader,
        shuffle=False,
        drop_last=False,
    )

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )
    world_model = build_world_model(cfg, encoder)

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )
    manager()


if __name__ == "__main__":
    run()
