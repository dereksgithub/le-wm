"""JEPA implementation."""

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


class JEPA(nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        grid_projector=None,
        use_spatial_latents=False,
        history_size=3,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.grid_projector = grid_projector or nn.Identity()
        self.use_spatial_latents = use_spatial_latents
        self.history_size = history_size

    def pool_grid(self, emb_grid):
        return emb_grid.mean(dim=(-1, -2))

    def encode(self, info):
        """Encode observations and actions into pooled and optional spatial embeddings."""
        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)

        if self.use_spatial_latents:
            tokens = output.last_hidden_state[:, 1:]
            num_patches = tokens.size(1)
            grid_size = math.isqrt(num_patches)
            if grid_size * grid_size != num_patches:
                raise ValueError(f"Expected a square number of patches, received {num_patches}.")

            grid_tokens = self.grid_projector(tokens)
            emb_grid = rearrange(
                grid_tokens,
                "(b t) (h w) d -> b t d h w",
                b=b,
                h=grid_size,
                w=grid_size,
            )
            pooled = self.pool_grid(emb_grid)
            info["emb_grid"] = emb_grid
            info["emb"] = self.projector(pooled)
        else:
            pixels_emb = output.last_hidden_state[:, 0]
            emb = self.projector(pixels_emb)
            info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict_grid(self, emb_grid, act_emb):
        if not self.use_spatial_latents:
            raise RuntimeError("predict_grid is only available for spatial JEPA variants.")
        return self.predictor(emb_grid, act_emb)

    def predict(self, emb, act_emb, emb_grid=None):
        """Predict pooled embeddings from context embeddings and actions."""
        if self.use_spatial_latents:
            if emb_grid is None:
                raise ValueError("Spatial JEPA variants require emb_grid for prediction.")
            pred_grid = self.predict_grid(emb_grid, act_emb)
            return self.pred_proj(self.pool_grid(pred_grid))

        preds = self.predictor(emb, act_emb)
        return self.pred_proj(preds)

    def rollout_latents(
        self,
        emb_context,
        act_emb,
        *,
        num_preds,
        emb_grid_context=None,
        history_size=None,
    ):
        """Autoregressively predict future pooled and optional spatial embeddings."""
        history_size = history_size or self.history_size
        if emb_context.size(1) < history_size:
            raise ValueError(
                f"Need at least {history_size} context steps, received {emb_context.size(1)}."
            )

        required_actions = history_size + max(num_preds - 1, 0)
        if act_emb.size(1) < required_actions:
            raise ValueError(
                f"Need at least {required_actions} encoded actions, received {act_emb.size(1)}."
            )

        current_emb = emb_context.clone()
        current_grid = emb_grid_context.clone() if emb_grid_context is not None else None
        pred_emb_steps = []
        pred_grid_steps = []

        for step in range(num_preds):
            emb_window = current_emb[:, -history_size:]
            act_window = act_emb[:, step : step + history_size]
            if current_grid is not None:
                grid_window = current_grid[:, -history_size:]
                pred_grid = self.predict_grid(grid_window, act_window)
                pred_emb = self.pred_proj(self.pool_grid(pred_grid))
                current_grid = torch.cat([current_grid, pred_grid], dim=1)
                pred_grid_steps.append(pred_grid)
            else:
                pred_seq = self.predict(emb_window, act_window)
                pred_emb = pred_seq[:, -1:]

            current_emb = torch.cat([current_emb, pred_emb], dim=1)
            pred_emb_steps.append(pred_emb)

        result = {
            "pred_emb": torch.cat(pred_emb_steps, dim=1),
            "full_emb": current_emb,
        }
        if pred_grid_steps:
            result["pred_emb_grid"] = torch.cat(pred_grid_steps, dim=1)
            result["full_emb_grid"] = current_grid
        return result

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size=None):
        """Roll out the model given an initial info dict and action sequence."""
        history_size = history_size or self.history_size
        assert "pixels" in info, "pixels not in info_dict"

        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        if H < history_size:
            raise ValueError(f"Expected at least {history_size} history frames, received {H}.")

        info["action"] = action_sequence[:, :, :H]
        init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        init = self.encode(init)

        emb = init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        emb_grid = None
        if "emb_grid" in init:
            emb_grid = init["emb_grid"].unsqueeze(1).expand(B, S, -1, -1, -1, -1)
            emb_grid = rearrange(emb_grid, "b s ... -> (b s) ...").clone()

        act = rearrange(action_sequence, "b s ... -> (b s) ...")
        act_emb = self.action_encoder(act)
        rollout = self.rollout_latents(
            emb,
            act_emb,
            num_preds=T - H + 1,
            emb_grid_context=emb_grid,
            history_size=history_size,
        )

        info["predicted_emb"] = rearrange(
            rollout["full_emb"],
            "(b s) ... -> b s ...",
            b=B,
            s=S,
        )
        if "full_emb_grid" in rollout:
            info["predicted_emb_grid"] = rearrange(
                rollout["full_emb_grid"],
                "(b s) ... -> b s ...",
                b=B,
                s=S,
            )
        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]
        goal_emb = info_dict["goal_emb"]
        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        return F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """Compute action-sequence cost given an initial state and goal."""
        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for key in list(info_dict.keys()):
            if torch.is_tensor(info_dict[key]):
                info_dict[key] = info_dict[key].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for key in list(goal.keys()):
            if key.startswith("goal_"):
                goal[key[len("goal_") :]] = goal.pop(key)

        goal.pop("action", None)
        goal = self.encode(goal)
        info_dict["goal_emb"] = goal["emb"]

        info_dict = self.rollout(info_dict, action_candidates)
        return self.criterion(info_dict)
