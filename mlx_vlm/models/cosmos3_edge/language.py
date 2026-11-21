import itertools
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    project_logits,
    scaled_dot_product_attention,
)
from ..cache import KVCache
from .config import ModelConfig, TextConfig


def rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_multimodal_rotary_pos_emb(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array, unsqueeze_dim: int = 1
):
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Cosmos3EdgeRotaryEmbedding:
    """3-axis interleaved M-RoPE matching Cosmos3EdgeTextRotaryEmbedding."""

    def __init__(self, args: TextConfig):
        self.dim = args.head_dim
        self.base = float(args.rope_theta)
        rope_params = args.rope_parameters or {}
        self.mrope_section = list(rope_params.get("mrope_section", [24, 20, 20]))
        self.inv_freq = 1.0 / (
            self.base ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        # Precompute channel selector [0=T, 1=H, 2=W] for zero-loop interleaving
        axis_selector = [0] * (self.dim // 2)
        for dim_idx, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim_idx] * 3
            for idx in range(offset, min(length, self.dim // 2), 3):
                axis_selector[idx] = dim_idx
        self._axis_selector = mx.array(axis_selector, dtype=mx.int32)

    def __call__(self, x: mx.array, position_ids: mx.array):
        if position_ids.ndim == 2:
            position_ids = mx.broadcast_to(
                position_ids[None, ...],
                (3, position_ids.shape[0], position_ids.shape[1]),
            )
        elif position_ids.shape[0] == 1:
            position_ids = mx.broadcast_to(
                position_ids, (3, position_ids.shape[1], position_ids.shape[2])
            )

        # Select per-frequency-channel position directly: (B, L, dim//2)
        # position_ids: (3, B, L) -> transpose (B, L, 3) -> gather along axis=-1
        pos_bl3 = position_ids.transpose(1, 2, 0).astype(mx.float32)
        selected_pos = mx.take(pos_bl3, self._axis_selector, axis=-1)
        freqs = selected_pos * self.inv_freq[None, None, :]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb).astype(x.dtype)
        sin = mx.sin(emb).astype(x.dtype)
        return cos, sin


class Cosmos3EdgeAttention(nn.Module):
    """Dense GQA attention with Cosmos3-Edge M-RoPE (no Q/K RMSNorm)."""

    def __init__(self, args: TextConfig):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            dim, self.n_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, dim, bias=args.attention_bias
        )

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries = (
            self.q_proj(x)
            .reshape(B, L, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        keys = (
            self.k_proj(x)
            .reshape(B, L, self.n_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        values = (
            self.v_proj(x)
            .reshape(B, L, self.n_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        queries, keys = apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        if mask is not None and isinstance(mask, mx.array):
            mask = mask[..., : keys.shape[-2]]

        output = scaled_dot_product_attention(
            queries, keys, values, cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class Cosmos3EdgeMLP(nn.Module):
    """Dense two-projection ReLU-squared MLP used by Cosmos3 Edge."""

    def __init__(self, args: TextConfig):
        super().__init__()
        self.hidden_act = args.hidden_act
        self.fc1 = nn.Linear(args.hidden_size, args.intermediate_size, bias=args.mlp_bias)
        self.fc2 = nn.Linear(args.intermediate_size, args.hidden_size, bias=args.mlp_bias)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.fc1(x)
        if self.hidden_act == "relu2":
            r = nn.relu(h)
            h = r * r
        elif self.hidden_act == "silu":
            h = nn.silu(h)
        else:
            h = nn.gelu(h)
        return self.fc2(h)


class Cosmos3EdgeDecoderLayer(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.self_attn = Cosmos3EdgeAttention(args)
        self.mlp = Cosmos3EdgeMLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), cos, sin, mask=mask, cache=cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class Cosmos3EdgeTextModel(nn.Module):
    """Cosmos3-Edge decoder stack (shared architecture for VLM LLM and Action Expert)."""

    def __init__(self, args: TextConfig, include_embed_tokens: bool = True):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        if include_embed_tokens:
            self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Cosmos3EdgeDecoderLayer(args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.rotary_emb = Cosmos3EdgeRotaryEmbedding(args)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        position_ids: Optional[mx.array] = None,
        is_causal: bool = True,
    ) -> mx.array:
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds

        B, L, _ = h.shape
        if cache is None:
            cache = [None] * len(self.layers)

        if position_ids is None:
            offset = 0
            if cache and cache[0] is not None:
                offset = cache[0].offset
            pos = mx.arange(L, dtype=mx.int32)[None, :] + offset
            pos = mx.broadcast_to(pos, (B, L))
            position_ids = mx.broadcast_to(pos[None, :, :], (3, B, L))

        if mask is None and is_causal:
            mask = create_attention_mask(
                h, cache[0] if cache and cache[0] is not None else cache
            )

        cos, sin = self.rotary_emb(h, position_ids)
        for layer, c in zip(self.layers, cache):
            h = layer(h, cos, sin, mask=mask, cache=c)
        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig, config: Optional[ModelConfig] = None):
        super().__init__()
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = Cosmos3EdgeTextModel(args, include_embed_tokens=True)
        self._rope_deltas = None
        self._position_ids = None
        self._logit_mask: Optional[mx.array] = None

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ):
        """Compute 3D M-RoPE position IDs and deltas matching Cosmos3EdgeModel.get_rope_index."""
        batch_size, seq_length = input_ids.shape
        has_vision = image_grid_thw is not None or video_grid_thw is not None

        if not has_vision or self.config is None:
            if attention_mask is not None:
                pos = np.cumsum(np.array(attention_mask, dtype=np.int64), axis=-1) - 1
                mask_np = np.array(attention_mask)
                pos = np.where(mask_np == 0, 0, pos)
                pos_3d = np.broadcast_to(pos[None, :, :], (3, batch_size, seq_length))
                deltas = pos.max(axis=-1, keepdims=True) + 1 - seq_length
                return mx.array(pos_3d, dtype=mx.int32), mx.array(deltas, dtype=mx.int32)
            pos = mx.arange(seq_length, dtype=mx.int32)[None, None, :]
            pos_3d = mx.broadcast_to(pos, (3, batch_size, seq_length))
            deltas = mx.zeros((batch_size, 1), dtype=mx.int32)
            return pos_3d, deltas

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id

        img_grids = image_grid_thw.tolist() if image_grid_thw is not None else []
        vid_grids = []
        if video_grid_thw is not None:
            for t, h, w in video_grid_thw.tolist():
                vid_grids.extend([[1, int(h), int(w)]] * int(t))

        img_iter = iter(img_grids)
        vid_iter = iter(vid_grids)

        ids_np = np.array(input_ids, dtype=np.int64)
        mask_np = (
            np.array(attention_mask, dtype=bool)
            if attention_mask is not None
            else np.ones_like(ids_np, dtype=bool)
        )

        position_ids = np.zeros((3, batch_size, seq_length), dtype=np.int32)
        mrope_deltas = []

        for b in range(batch_size):
            valid_ids = ids_np[b][mask_np[b]]
            token_types = np.zeros_like(valid_ids)
            token_types[valid_ids == image_token_id] = 1
            token_types[valid_ids == video_token_id] = 2

            groups = []
            for mod_type, grp in itertools.groupby(
                enumerate(token_types.tolist()), key=lambda x: x[1]
            ):
                grp_list = list(grp)
                groups.append((mod_type, grp_list[0][0], grp_list[-1][0] + 1))

            current_pos = 0
            pos_chunks = []
            for mod_type, start_idx, end_idx in groups:
                if mod_type == 0:
                    t_len = end_idx - start_idx
                    txt_pos = np.arange(t_len, dtype=np.int32)[None, :] + current_pos
                    pos_chunks.append(np.broadcast_to(txt_pos, (3, t_len)))
                    current_pos += t_len
                else:
                    # Each contiguous run of image_token_id / video_token_id may span one image
                    # (wrapped in <|vision_start|> ... <|vision_end|> so adjacent images are separated by text tokens)
                    remaining = end_idx - start_idx
                    while remaining > 0:
                        t_g, h_g, w_g = next(img_iter if mod_type == 1 else vid_iter)
                        llm_t = int(t_g)
                        llm_h = int(h_g) // spatial_merge_size
                        llm_w = int(w_g) // spatial_merge_size
                        n_vis = llm_t * llm_h * llm_w
                        if n_vis > remaining:
                            raise ValueError(
                                f"Vision grid {llm_t}x{llm_h}x{llm_w}={n_vis} exceeds placeholder run {remaining}"
                            )
                        p_t = np.arange(llm_t, dtype=np.int32) + current_pos
                        p_h = np.arange(llm_h, dtype=np.int32) + current_pos
                        p_w = np.arange(llm_w, dtype=np.int32) + current_pos
                        tg, hg, wg = np.meshgrid(p_t, p_h, p_w, indexing="ij")
                        vis_pos = np.stack([tg, hg, wg], axis=0).reshape(3, -1)
                        pos_chunks.append(vis_pos)
                        current_pos += max(llm_h, llm_w)
                        remaining -= n_vis

            llm_positions = (
                np.concatenate(pos_chunks, axis=1)
                if pos_chunks
                else np.zeros((3, 0), dtype=np.int32)
            )
            position_ids[:, b, mask_np[b]] = llm_positions
            max_pos = int(llm_positions.max()) + 1 if llm_positions.size > 0 else 0
            mrope_deltas.append([max_pos - len(valid_ids)])

        return mx.array(position_ids, dtype=mx.int32), mx.array(
            mrope_deltas, dtype=mx.int32
        )

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        logits_to_keep: Optional[int] = 0,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        rope_deltas_kw = kwargs.pop("rope_deltas", None)

        if pixel_values is not None:
            self._rope_deltas = None
            self._position_ids = None

        if rope_deltas_kw is not None:
            self._rope_deltas = rope_deltas_kw

        cache_offset = 0
        cache_offset_array = None
        if cache and cache[0] is not None:
            c0 = cache[0]
            cache_offset = c0._idx if hasattr(c0, "_idx") else c0.offset
            if isinstance(c0.offset, mx.array) and c0.offset.ndim > 0:
                cache_offset_array = c0.offset

        seq_length = inputs.shape[1] if inputs is not None else inputs_embeds.shape[1]
        batch_size = inputs.shape[0] if inputs is not None else inputs_embeds.shape[0]

        rope_mask = mask
        if mask is not None and mask.shape[-1] != seq_length:
            rope_mask = None

        if position_ids is not None:
            if position_ids.shape[-1] != seq_length:
                self._position_ids = position_ids
                position_ids = position_ids[
                    :, :, cache_offset : cache_offset + seq_length
                ]
        elif rope_mask is None or rope_mask.ndim == 2:
            recalc_condition = (
                (cache is not None and cache[0] is not None and cache_offset == 0)
                or self._rope_deltas is None
                or cache is None
            )
            if recalc_condition:
                if self._position_ids is not None:
                    position_ids = self._position_ids[
                        :, :, cache_offset : cache_offset + seq_length
                    ]
                else:
                    position_ids, rope_deltas = self.get_rope_index(
                        inputs, image_grid_thw, video_grid_thw, rope_mask
                    )
                    self._rope_deltas = rope_deltas
                    self._position_ids = position_ids
            else:
                if cache_offset_array is not None:
                    base_offset = cache_offset_array[:batch_size].reshape(-1, 1)
                else:
                    base_offset = mx.array([[int(cache_offset)]] * batch_size, dtype=mx.int32)

                rope_delta = self._rope_deltas
                if rope_delta is not None:
                    if rope_delta.ndim == 1:
                        rope_delta = rope_delta[:, None]
                    if rope_delta.shape[0] < batch_size:
                        repeats = batch_size // rope_delta.shape[0]
                        rope_delta = mx.repeat(rope_delta, repeats, axis=0)
                    else:
                        rope_delta = rope_delta[:batch_size]
                    delta = base_offset + rope_delta
                else:
                    delta = base_offset

                pos = mx.arange(seq_length, dtype=mx.int32)[None, :] + delta
                position_ids = mx.broadcast_to(
                    pos[None, :, :], (3, batch_size, seq_length)
                )

        # Ensure 2D attention mask is sliced to active KV length for chunked prefill
        attn_mask = None
        if mask is not None and mask.ndim == 4:
            attn_mask = mask
        elif mask is not None and mask.ndim == 2:
            active_k_len = int(cache_offset) + seq_length
            cur_mask = mask[:, :active_k_len]
            if int((cur_mask == 0).sum().item()) > 0:
                k_len = cur_mask.shape[-1]
                q_len = seq_length
                pad_mask = mx.where(
                    cur_mask == 0,
                    mx.array(-1e9, dtype=mx.float32),
                    mx.array(0.0, dtype=mx.float32),
                )[:, None, None, :]
                if q_len > 1:
                    idx_q = mx.arange(k_len - q_len, k_len)[:, None]
                    idx_k = mx.arange(k_len)[None, :]
                    causal = mx.where(
                        idx_k > idx_q,
                        mx.array(-1e9, dtype=mx.float32),
                        mx.array(0.0, dtype=mx.float32),
                    )
                    attn_mask = pad_mask + causal[None, None, :, :]
                else:
                    attn_mask = pad_mask

        hidden = self.model(
            inputs,
            inputs_embeds=inputs_embeds,
            mask=attn_mask,
            cache=cache,
            position_ids=position_ids,
        )
        proj = (
            self.model.embed_tokens.as_linear
            if self.args.tie_word_embeddings
            else self.lm_head
        )
        logits = project_logits(hidden, proj, logits_to_keep)
        if self._logit_mask is not None:
            logits = logits + self._logit_mask.astype(logits.dtype)
        return LanguageModelOutput(logits=logits)

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
