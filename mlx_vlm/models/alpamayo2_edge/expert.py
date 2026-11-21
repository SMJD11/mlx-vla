"""Pure-MLX Action Expert and Euler Flow Matching for Alpamayo 2 Edge."""

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..cosmos3_edge.language import Cosmos3EdgeTextModel
from .action_space import build_action_space
from .config import ExpertConfig


class FourierEncoderV2(nn.Module):
    """Fourier feature encoder with logarithmically spaced frequencies."""

    def __init__(self, dim: int = 20, max_freq: float = 100.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"FourierEncoderV2 dim must be even, got {dim}")
        half = dim // 2
        freqs = np.logspace(0.0, math.log10(max_freq), num=half, dtype=np.float32)
        self.out_dim = dim
        self._freqs = mx.array(freqs[None, :])

    def __call__(self, x: mx.array) -> mx.array:
        arg = x.astype(mx.float32)[..., None] * self._freqs * (2.0 * math.pi)
        return mx.concatenate([mx.sin(arg), mx.cos(arg)], axis=-1) * math.sqrt(2.0)


class MLPEncoder(nn.Module):
    """MLP encoder preserving checkpoint parameter path `encoder.trunk.{idx}.*`."""

    def __init__(
        self,
        num_input_feats: int,
        num_enc_layers: int,
        hidden_size: int,
        outdim: int,
    ):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(num_input_feats, hidden_size), nn.SiLU()]
        for layer_idx in range(num_enc_layers):
            layers.append(nn.RMSNorm(hidden_size, eps=1e-5))
            if layer_idx < num_enc_layers - 1:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.SiLU())
            else:
                layers.append(nn.Linear(hidden_size, outdim))
        self.trunk = layers

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.trunk:
            x = layer(x)
        return x


class PerWaypointActionInProjV2(nn.Module):
    """Project per-waypoint noisy actions and diffusion timesteps into expert tokens."""

    def __init__(
        self,
        in_dims: Tuple[int, int],
        out_dim: int,
        num_enc_layers: int = 2,
        hidden_size: int = 512,
        max_freq: float = 100.0,
        num_fourier_feats: int = 20,
    ):
        super().__init__()
        self.in_dims = tuple(in_dims)
        self.out_dim = out_dim
        self.sinus = [
            FourierEncoderV2(dim=num_fourier_feats, max_freq=max_freq)
            for _ in range(in_dims[-1])
        ]
        self.timestep_fourier_encoder = FourierEncoderV2(
            dim=num_fourier_feats, max_freq=max_freq
        )
        num_input_feats = (in_dims[-1] + 1) * num_fourier_feats
        self.encoder = MLPEncoder(
            num_input_feats=num_input_feats,
            num_enc_layers=num_enc_layers,
            hidden_size=hidden_size,
            outdim=out_dim,
        )
        self.norm = nn.LayerNorm(out_dim)

    def __call__(self, x: mx.array, timesteps: mx.array) -> mx.array:
        batch_size, n_waypoints, _ = x.shape
        action_feats = mx.concatenate(
            [enc(x[:, :, idx]) for idx, enc in enumerate(self.sinus)], axis=-1
        )
        t_feats = self.timestep_fourier_encoder(timesteps[..., -1])
        if t_feats.shape[1] != n_waypoints:
            t_feats = mx.broadcast_to(
                t_feats, (batch_size, n_waypoints, t_feats.shape[-1])
            )
        feats = mx.concatenate([action_feats, t_feats], axis=-1)
        target_dtype = self.encoder.trunk[0].weight.dtype
        feats = feats.astype(target_dtype)
        encoded = self.encoder(feats.reshape(batch_size * n_waypoints, -1)).reshape(
            batch_size, n_waypoints, self.out_dim
        )
        return self.norm(encoded)


class ExpertModel(nn.Module):
    """512-wide 28-layer Cosmos3-Edge action expert sharing VLM KV-cache geometry."""

    def __init__(self, config: ExpertConfig):
        super().__init__()
        self.config = config
        self.action_space = build_action_space(config.action_space_cfg)
        self.action_dims = self.action_space.get_action_space_dims()

        self.expert = Cosmos3EdgeTextModel(
            config.llm_config, include_embed_tokens=False
        )
        aip_cfg = dict(config.action_in_proj_cfg)
        aip_cfg.pop("_target_", None)
        self.action_in_proj = PerWaypointActionInProjV2(
            in_dims=self.action_dims,
            out_dim=config.llm_config.hidden_size,
            **aip_cfg,
        )
        self.action_out_proj = nn.Linear(
            config.llm_config.hidden_size, self.action_dims[-1], bias=True
        )
        self.default_inference_steps = int(
            config.diffusion_cfg.get("num_inference_steps", 10)
        )

    def sample_actions(
        self,
        cache: list,
        prefix_length: int,
        position_ids: mx.array,
        attention_mask: mx.array,
        batch_size: int,
        inference_steps: Optional[int] = None,
        temperature: float = 1.0,
        initial_noise: Optional[mx.array] = None,
        seed: Optional[int] = None,
    ) -> mx.array:
        steps = inference_steps or self.default_inference_steps
        n_tokens, action_dim = self.action_dims

        # Pre-reserve cache capacity so 10 denoising steps update & trim in-place with zero reallocations
        for c in cache:
            if c is not None and hasattr(c, "prefix_cache_reserve"):
                c.prefix_cache_reserve(prefix_length + n_tokens)

        if initial_noise is not None:
            x = initial_noise.astype(mx.float32)
        else:
            if seed is not None:
                mx.random.seed(seed)
            x = (
                mx.random.normal(shape=(batch_size, n_tokens, action_dim), dtype=mx.float32)
                * temperature
            )

        time_steps = np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)
        mask_cast = attention_mask.astype(self.expert.layers[0].self_attn.q_proj.weight.dtype)

        for i in range(steps):
            dt = float(time_steps[i + 1] - time_steps[i])
            t_start = mx.full(
                (batch_size, 1, 1), float(time_steps[i]), dtype=mx.float32
            )
            embeds = self.action_in_proj(x, t_start)
            try:
                hidden = self.expert(
                    inputs_embeds=embeds,
                    mask=mask_cast,
                    cache=cache,
                    position_ids=position_ids,
                    is_causal=False,
                )
            finally:
                for c in cache:
                    if c is not None:
                        if hasattr(c, "trim"):
                            c.trim(c.offset - prefix_length)
                        elif hasattr(c, "offset"):
                            c.offset = prefix_length

            v = self.action_out_proj(hidden).astype(mx.float32)
            x = x + dt * v
            mx.eval(
                x,
                *[c.keys for c in cache if c is not None and c.keys is not None],
                *[c.values for c in cache if c is not None and c.values is not None],
            )

        return x
