from types import SimpleNamespace

import pytest
import torch
from torch import nn

from jepa import JEPA
from module import ARPredictor, MLP, PDEPredictor, SIGReg, SpatialTransformerPredictor


class DummyEncoder(nn.Module):
    def __init__(self, hidden_dim=12, grid_size=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size

    def forward(self, pixels, interpolate_pos_encoding=True):
        batch = pixels.size(0)
        num_patches = self.grid_size * self.grid_size
        cls = pixels.mean(dim=tuple(range(1, pixels.ndim)), keepdim=False).view(batch, 1, 1)
        cls = cls.expand(-1, 1, self.hidden_dim)
        patch_tokens = torch.linspace(
            0.0,
            1.0,
            steps=num_patches,
            device=pixels.device,
            dtype=pixels.dtype,
        ).view(1, num_patches, 1)
        patch_tokens = patch_tokens.expand(batch, -1, self.hidden_dim)
        return SimpleNamespace(last_hidden_state=torch.cat([cls, patch_tokens], dim=1))


def build_model(variant, embed_dim=8, hidden_dim=12, grid_size=4, history_size=3):
    encoder = DummyEncoder(hidden_dim=hidden_dim, grid_size=grid_size)
    projector_input = hidden_dim if variant == "cls_transformer" else embed_dim
    projector = MLP(
        input_dim=projector_input,
        hidden_dim=2 * embed_dim,
        output_dim=embed_dim,
        norm_fn=None,
    )
    pred_proj = MLP(
        input_dim=projector_input,
        hidden_dim=2 * embed_dim,
        output_dim=embed_dim,
        norm_fn=None,
    )
    grid_projector = None
    use_spatial = variant != "cls_transformer"

    if variant == "cls_transformer":
        predictor = ARPredictor(
            num_frames=history_size,
            depth=2,
            heads=2,
            mlp_dim=32,
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            output_dim=hidden_dim,
            dim_head=4,
            dropout=0.0,
            emb_dropout=0.0,
        )
    elif variant == "spatial_transformer":
        predictor = SpatialTransformerPredictor(
            num_frames=history_size,
            grid_size=grid_size,
            depth=2,
            heads=2,
            mlp_dim=32,
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            dim_head=4,
            dropout=0.0,
            emb_dropout=0.0,
        )
        grid_projector = MLP(
            input_dim=hidden_dim,
            hidden_dim=2 * embed_dim,
            output_dim=embed_dim,
            norm_fn=None,
        )
        pred_proj = MLP(
            input_dim=embed_dim,
            hidden_dim=2 * embed_dim,
            output_dim=embed_dim,
            norm_fn=None,
        )
    elif variant == "spatial_pde":
        predictor = PDEPredictor(
            input_dim=embed_dim,
            action_dim=embed_dim,
            output_dim=embed_dim,
            num_evolve_steps=2,
            dilations=(1, 2, 4),
            reaction_hidden_dim=16,
            norm_every=1,
        )
        grid_projector = MLP(
            input_dim=hidden_dim,
            hidden_dim=2 * embed_dim,
            output_dim=embed_dim,
            norm_fn=None,
        )
        pred_proj = MLP(
            input_dim=embed_dim,
            hidden_dim=2 * embed_dim,
            output_dim=embed_dim,
            norm_fn=None,
        )
    else:
        raise ValueError(variant)

    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=nn.Identity(),
        projector=projector,
        pred_proj=pred_proj,
        grid_projector=grid_projector,
        use_spatial_latents=use_spatial,
        history_size=history_size,
    )


def make_batch(batch_size=2, steps=6, action_dim=8):
    return {
        "pixels": torch.randn(batch_size, steps, 3, 8, 8),
        "action": torch.randn(batch_size, steps, action_dim),
    }


def test_spatial_encode_produces_grid_and_pooled_embeddings():
    model = build_model("spatial_pde")
    batch = make_batch(batch_size=2, steps=3)
    output = model.encode(batch)

    assert output["emb"].shape == (2, 3, 8)
    assert output["emb_grid"].shape == (2, 3, 8, 4, 4)
    assert torch.isfinite(output["emb"]).all()
    assert torch.isfinite(output["emb_grid"]).all()


@pytest.mark.parametrize("variant", ["cls_transformer", "spatial_transformer", "spatial_pde"])
def test_tiny_training_loss_is_finite_for_all_variants(variant):
    history_size = 3
    num_preds = 3
    model = build_model(variant, history_size=history_size)
    sigreg = SIGReg(knots=5, num_proj=8, sync_ddp=False)

    batch = make_batch(batch_size=2, steps=history_size + num_preds)
    output = model.encode(batch)

    rollout = model.rollout_latents(
        output["emb"][:, :history_size],
        output["act_emb"][:, : history_size + num_preds - 1],
        num_preds=num_preds,
        emb_grid_context=output.get("emb_grid", None)[:, :history_size]
        if output.get("emb_grid", None) is not None
        else None,
        history_size=history_size,
    )

    target_emb = output["emb"][:, history_size:]
    pred_loss = torch.nn.functional.mse_loss(rollout["pred_emb"], target_emb)

    if "pred_emb_grid" in rollout:
        target_grid = output["emb_grid"][:, history_size:]
        pred_loss = torch.nn.functional.mse_loss(rollout["pred_emb_grid"], target_grid)

    total_loss = pred_loss + 0.1 * sigreg(output["emb"].transpose(0, 1))
    assert torch.isfinite(pred_loss)
    assert torch.isfinite(total_loss)


def test_rollout_and_get_cost_keep_planning_interface():
    history_size = 3
    model = build_model("spatial_pde", history_size=history_size)
    batch_size = 2
    num_samples = 3
    action_dim = 8
    horizon = 5

    info = {
        "pixels": torch.randn(batch_size, num_samples, history_size, 3, 8, 8),
        "goal": torch.randn(batch_size, num_samples, 1, 3, 8, 8),
        "action": torch.randn(batch_size, num_samples, 1, action_dim),
    }
    action_candidates = torch.randn(batch_size, num_samples, horizon, action_dim)

    rollout = model.rollout(
        {k: v.clone() for k, v in info.items()},
        action_candidates.clone(),
        history_size=history_size,
    )
    assert "predicted_emb" in rollout
    assert rollout["predicted_emb"].shape[:2] == (batch_size, num_samples)

    cost = model.get_cost(
        {k: v.clone() for k, v in info.items()},
        action_candidates.clone(),
    )
    assert cost.shape == (batch_size, num_samples)
    assert torch.isfinite(cost).all()
