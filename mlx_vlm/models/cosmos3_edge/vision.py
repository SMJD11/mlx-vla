from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import ensure_fused_sdpa
from .config import ModelConfig, VisionConfig


def _bilinear_antialias_weights(src: int, dst: int) -> mx.array:
    """Exact separable weights matching PyTorch F.interpolate(mode='bilinear', align_corners=False, antialias=True)."""
    scale = src / dst
    fscale = max(1.0, scale)
    centers = (np.arange(dst, dtype=np.float64) + 0.5) * scale
    j = np.arange(src, dtype=np.float64) + 0.5
    x = np.abs(j[None, :] - centers[:, None]) / fscale
    w = np.maximum(0.0, 1.0 - x)
    w /= w.sum(axis=1, keepdims=True)
    return mx.array(w.astype(np.float32))


class Cosmos3EdgeVisionEmbeddings(nn.Module):
    """SigLIP2 patch and learned-position embeddings for packed Cosmos3-Edge vision inputs."""

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Linear(
            config.num_channels * self.patch_size * self.patch_size,
            self.embed_dim,
            bias=True,
        )
        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    def resize_positional_embeddings(self, grid_thw: mx.array) -> mx.array:
        source_dtype = self.position_embedding.weight.dtype
        src = self.position_embedding_size
        dim = self.embed_dim
        merge_size = self.config.spatial_merge_size

        pos_grid = self.position_embedding.weight.astype(mx.float32).reshape(
            src, src, dim
        )
        weight_cache = {}
        chunks = []

        for temporal, height, width in grid_thw.tolist():
            t, h, w = int(temporal), int(height), int(width)
            if (src, h) not in weight_cache:
                weight_cache[(src, h)] = _bilinear_antialias_weights(src, h)
            if (src, w) not in weight_cache:
                weight_cache[(src, w)] = _bilinear_antialias_weights(src, w)

            w_h = weight_cache[(src, h)]
            w_w = weight_cache[(src, w)]

            # Separable 2D bilinear+antialias resize in float32: W_h @ pos_grid @ W_w^T
            resized_h = (w_h @ pos_grid.reshape(src, src * dim)).reshape(h, src, dim)
            resized = (
                w_w @ resized_h.transpose(1, 0, 2).reshape(src, h * dim)
            ).reshape(w, h, dim).transpose(1, 0, 2)

            resized = resized.astype(source_dtype)
            # Reorder into processor's block-major 2x2 patch order
            resized = (
                resized.reshape(
                    h // merge_size,
                    merge_size,
                    w // merge_size,
                    merge_size,
                    dim,
                )
                .transpose(0, 2, 1, 3, 4)
                .reshape(h * w, dim)
            )
            if t > 1:
                resized = mx.tile(resized, (t, 1))
            chunks.append(resized)

        return mx.concatenate(chunks, axis=0)

    def __call__(self, pixel_values: mx.array, grid_thw: mx.array) -> mx.array:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.astype(target_dtype))
        pos_embeds = self.resize_positional_embeddings(grid_thw)
        if patch_embeds.shape[0] != pos_embeds.shape[0]:
            raise ValueError(
                f"Packed visual patch count {patch_embeds.shape[0]} does not match grid_thw {pos_embeds.shape[0]}"
            )
        return patch_embeds + pos_embeds


class Cosmos3EdgeVisionAttention(nn.Module):
    """Packed non-causal SigLIP2 attention with one sequence per image or video frame."""

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def __call__(
        self, hidden_states: mx.array, frame_runs: List[Tuple[int, int, int, int]]
    ) -> mx.array:
        seq_len = hidden_states.shape[0]
        q = self.q_proj(hidden_states).reshape(seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(seq_len, self.num_heads, self.head_dim)

        # Group contiguous frames with identical patch count into batched SDPA calls
        outputs = []
        for start, end, count, length in frame_runs:
            q_run = q[start:end].reshape(count, length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            k_run = k[start:end].reshape(count, length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            v_run = v[start:end].reshape(count, length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            attn_out = ensure_fused_sdpa(q_run, k_run, v_run, scale=self.scale)
            outputs.append(
                attn_out.transpose(0, 2, 1, 3).reshape(count * length, self.embed_dim)
            )

        out = outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=0)
        return self.out_proj(out)


class Cosmos3EdgeVisionMLP(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.hidden_act = config.hidden_act
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        if self.hidden_act in ("gelu_pytorch_tanh", "gelu_fast", "gelu_new"):
            x = nn.gelu_approx(x)
        else:
            x = nn.gelu(x)
        return self.fc2(x)


class Cosmos3EdgeVisionEncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = Cosmos3EdgeVisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Cosmos3EdgeVisionMLP(config)

    def __call__(
        self, hidden_states: mx.array, frame_runs: List[Tuple[int, int, int, int]]
    ) -> mx.array:
        hidden_states = hidden_states + self.self_attn(
            self.layer_norm1(hidden_states), frame_runs
        )
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class Cosmos3EdgeEncoder(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.layers = [
            Cosmos3EdgeVisionEncoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ]

    @staticmethod
    def _build_frame_runs(grid_thw: mx.array) -> List[Tuple[int, int, int, int]]:
        """Group contiguous frames of identical patch count into (start, end, count, length) runs."""
        frame_lengths: List[int] = []
        for t, h, w in grid_thw.tolist():
            frame_lengths.extend([int(h) * int(w)] * int(t))

        runs: List[Tuple[int, int, int, int]] = []
        offset = 0
        idx = 0
        n = len(frame_lengths)
        while idx < n:
            length = frame_lengths[idx]
            count = 1
            while idx + count < n and frame_lengths[idx + count] == length:
                count += 1
            end = offset + count * length
            runs.append((offset, end, count, length))
            offset = end
            idx += count
        return runs

    def __call__(self, inputs_embeds: mx.array, grid_thw: mx.array) -> mx.array:
        frame_runs = self._build_frame_runs(grid_thw)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, frame_runs)
        return hidden_states


class VisionModel(nn.Module):
    """Packed variable-resolution SigLIP2 vision tower used by Cosmos3 Edge."""

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.spatial_merge_size = config.spatial_merge_size
        self.embeddings = Cosmos3EdgeVisionEmbeddings(config)
        self.encoder = Cosmos3EdgeEncoder(config)
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def __call__(self, pixel_values: mx.array, grid_thw: mx.array) -> mx.array:
        hidden_states = self.embeddings(pixel_values, grid_thw)
        hidden_states = self.encoder(hidden_states, grid_thw)
        return self.post_layernorm(hidden_states)


class Cosmos3EdgePatchMerger(nn.Module):
    """2x2 spatial patch merger projecting SigLIP2 features into LLM hidden space."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.input_hidden_size = config.vision_config.hidden_size
        self.hidden_size = self.input_hidden_size * (self.spatial_merge_size**2)
        self.norm = nn.LayerNorm(self.input_hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(
            self.hidden_size, config.projector_hidden_size, bias=True
        )
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(
            config.projector_hidden_size, config.text_config.hidden_size, bias=True
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = x.reshape(-1, self.spatial_merge_size**2, self.input_hidden_size)
        x = self.norm(x).reshape(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))
