import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch import nn

try:
    from torch.distributed.nn.functional import all_gather as dist_all_gather
except Exception:
    dist_all_gather = None


def modulate(x, shift, scale):
    """AdaLN-zero modulation."""
    return x * (1 + scale) + shift


class SIGReg(torch.nn.Module):
    """Sketched Isotropic Gaussian Regularizer with optional DDP sync."""

    def __init__(self, knots=17, num_proj=1024, sync_ddp=True):
        super().__init__()
        self.num_proj = num_proj
        self.sync_ddp = sync_ddp
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def _maybe_gather(self, proj):
        if not self.sync_ddp or dist_all_gather is None:
            return proj
        if not dist.is_available() or not dist.is_initialized():
            return proj
        if dist.get_world_size() == 1:
            return proj

        gathered = dist_all_gather(rearrange(proj, "t b d -> b t d").contiguous())
        proj = torch.cat(gathered, dim=0)
        return rearrange(proj, "b t d -> t b d")

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        proj = self._maybe_gather(proj)
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0).clamp_min_(1e-12))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class FeedForward(nn.Module):
    """Feed-forward network used in Transformer blocks."""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x: (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):
        x = self.input_proj(x)
        if c is not None:
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)
        return self.output_proj(x)


class Embedder(nn.Module):
    """Simple action embedder."""

    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        return self.embed(x)


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_layer = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_layer,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        x = self.net(x)
        return x.reshape(*shape[:-1], -1)


class ARPredictor(nn.Module):
    """Autoregressive predictor for pooled latent sequences."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, D)
        c: (B, T, A)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        return self.transformer(x, c)


class SpatialTransformerPredictor(nn.Module):
    """Spatial token predictor with per-frame spatial mixing and per-token temporal dynamics."""

    def __init__(
        self,
        *,
        num_frames,
        grid_size,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        spatial_depth = max(1, depth // 2)
        temporal_depth = max(1, depth - spatial_depth)
        self.spatial_pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, input_dim)
        )
        self.spatial_mixer = Transformer(
            input_dim,
            hidden_dim,
            input_dim,
            spatial_depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=Block,
        )
        self.temporal_predictor = ARPredictor(
            num_frames=num_frames,
            depth=temporal_depth,
            heads=heads,
            mlp_dim=mlp_dim,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim or input_dim,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )

    def forward(self, x, c):
        """
        x: (B, T, C, H, W)
        c: (B, T, A)
        """
        B, T, _, H, W = x.shape
        num_patches = H * W
        if num_patches != self.num_patches:
            raise ValueError(
                f"Expected {self.num_patches} spatial tokens, received {num_patches}."
            )

        tokens = rearrange(x, "b t c h w -> (b t) (h w) c")
        tokens = tokens + self.spatial_pos_embedding[:, :num_patches]
        tokens = self.spatial_mixer(tokens)
        tokens = rearrange(tokens, "(b t) n c -> (b n) t c", b=B, t=T, n=num_patches)

        act_tokens = c.unsqueeze(1).expand(-1, num_patches, -1, -1)
        act_tokens = rearrange(act_tokens, "b n t c -> (b n) t c")

        pred = self.temporal_predictor(tokens, act_tokens)[:, -1:]
        return rearrange(
            pred,
            "(b n) t c -> b t c h w",
            b=B,
            n=num_patches,
            h=H,
            w=W,
        )


class ChannelRMSNorm(nn.Module):
    """Channel-wise RMS normalization for spatial feature maps."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        scale = torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        return x * scale * self.weight


class MultiScaleDiffusion(nn.Module):
    """Fixed Laplacian filters with learned per-channel diffusion strength."""

    def __init__(self, channels, dilations=(1, 2, 4)):
        super().__init__()
        self.channels = channels
        self.dilations = tuple(int(d) for d in dilations)
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        self.register_buffer("kernel", kernel.view(1, 1, 3, 3))
        self.log_diffusion = nn.Parameter(torch.zeros(len(self.dilations), channels))

    def forward(self, x):
        kernel = self.kernel.expand(self.channels, 1, 3, 3)
        out = x.new_zeros(x.shape)
        for idx, dilation in enumerate(self.dilations):
            lap = F.conv2d(x, kernel, padding=dilation, dilation=dilation, groups=self.channels)
            strength = F.softplus(self.log_diffusion[idx]).view(1, self.channels, 1, 1)
            out = out + strength * lap
        return out


class PDEPredictor(nn.Module):
    """Reaction-diffusion predictor over spatial latent grids."""

    def __init__(
        self,
        *,
        input_dim,
        action_dim,
        output_dim=None,
        num_evolve_steps=4,
        dilations=(1, 2, 4),
        reaction_hidden_dim=None,
        norm_every=2,
    ):
        super().__init__()
        self.num_evolve_steps = num_evolve_steps
        self.norm_every = max(1, norm_every)

        hidden_dim = reaction_hidden_dim or (2 * input_dim)
        self.write_gate = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        self.write_value = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        self.action_proj = nn.Linear(action_dim, input_dim)
        self.diffusion = MultiScaleDiffusion(input_dim, dilations=dilations)
        self.reaction = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, input_dim, kernel_size=1),
        )
        self.norm = ChannelRMSNorm(input_dim)
        self.readout = nn.Conv2d(input_dim, output_dim or input_dim, kernel_size=1)
        self.log_dt = nn.Parameter(torch.tensor(math.log(0.1), dtype=torch.float32))
        self.state_decay = nn.Parameter(torch.zeros(1))

    def _write(self, state, obs):
        gate = torch.sigmoid(self.write_gate(obs))
        value = torch.tanh(self.write_value(obs))
        decay = 0.5 + 0.49 * torch.sigmoid(self.state_decay)
        return decay * state + gate * value

    def _evolve(self, state, action_term):
        dt = F.softplus(self.log_dt)
        for step in range(self.num_evolve_steps):
            update = self.diffusion(state) + self.reaction(state) + action_term
            state = state + dt * update
            if (step + 1) % self.norm_every == 0:
                state = self.norm(state)
        return state

    def forward(self, x, c):
        """
        x: (B, T, C, H, W)
        c: (B, T, A)
        """
        B, T, C, H, W = x.shape
        state = x.new_zeros(B, C, H, W)

        for idx in range(T):
            obs = x[:, idx]
            action = self.action_proj(c[:, idx]).view(B, C, 1, 1)
            state = self._write(state, obs)
            state = self._evolve(state, action)

        pred = self.readout(state)
        return pred.unsqueeze(1)
