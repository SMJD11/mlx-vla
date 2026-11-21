import re
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from . import processing_cosmos3_edge  # noqa: F401
from .config import ModelConfig
from .language import LanguageModel
from .vision import Cosmos3EdgePatchMerger, VisionModel

_DROPPED_GENERATOR_KEY_RE = re.compile(
    r"^(?:vlm\.)?(?:model\.language_model\.)?(?:"
    r"action_modality_embed$|"
    r"action_proj_(?:in|out)\.(?:bias|fc)\.weight$|"
    r"time_embedder\.|"
    r"proj_(?:in|out)\.|"
    r"norm_moe_gen\.|"
    r"layers\.\d+\.input_layernorm_moe_gen\.|"
    r"layers\.\d+\.post_attention_layernorm_moe_gen\.|"
    r"layers\.\d+\.mlp_moe_gen\.|"
    r"layers\.\d+\.self_attn\.(?:add_[qkv]_proj|to_add_out|norm_added_[qk]|k_norm_und_for_gen)\."
    r")"
)


def masked_scatter(
    final_embedding: mx.array,
    mask_expanded: mx.array,
    features: mx.array,
) -> mx.array:
    orig_shape = final_embedding.shape
    flat_emb = mx.flatten(final_embedding)
    flat_feat = mx.flatten(features.astype(final_embedding.dtype))
    flat_mask = mx.flatten(mask_expanded)
    positions = mx.array(np.where(np.array(flat_mask))[0], dtype=mx.uint32)
    flat_emb[positions] = flat_feat
    return mx.reshape(flat_emb, orig_shape)


class Model(nn.Module):
    """MLX implementation of NVIDIA Cosmos 3 Edge (Cosmos3EdgeForConditionalGeneration)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = (
            None if config.skip_vision else VisionModel(config.vision_config)
        )
        self.projector = (
            None if config.skip_vision else Cosmos3EdgePatchMerger(config)
        )
        self.language_model = LanguageModel(config.text_config, config)

    def encode_vision(
        self,
        pixel_values: mx.array,
        grid_thw: mx.array,
        max_frames_per_batch: int = 2,
    ) -> mx.array:
        """Encode packed vision patches in small frame micro-batches to bound Metal peak RAM & command buffer time."""
        if self.vision_tower is None or self.projector is None:
            raise ValueError("Cosmos3-Edge model was loaded without vision_tower/projector")
        dtype = self.vision_tower.post_layernorm.weight.dtype
        grids = grid_thw.tolist()
        n_items = len(grids)

        if n_items <= max_frames_per_batch:
            vis_hidden = self.vision_tower(pixel_values.astype(dtype), grid_thw)
            projected = self.projector(vis_hidden)
            mx.eval(projected)
            return projected

        projected_chunks = []
        patch_offset = 0
        for start in range(0, n_items, max_frames_per_batch):
            sub_grids = grids[start : start + max_frames_per_batch]
            n_patches = sum(int(t) * int(h) * int(w) for t, h, w in sub_grids)
            sub_pixels = pixel_values[patch_offset : patch_offset + n_patches].astype(dtype)
            sub_thw = mx.array(sub_grids, dtype=mx.int64)
            vis_hidden = self.vision_tower(sub_pixels, sub_thw)
            proj_chunk = self.projector(vis_hidden)
            mx.eval(proj_chunk)
            projected_chunks.append(proj_chunk)
            patch_offset += n_patches
            del vis_hidden, sub_pixels
            mx.clear_cache()

        out = mx.concatenate(projected_chunks, axis=0)
        mx.eval(out)
        del projected_chunks
        mx.clear_cache()
        return out

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        mask = kwargs.get("mask", None)
        if mask is None:
            mask = kwargs.get("attention_mask", None)
        pixel_values_videos = kwargs.get("pixel_values_videos", None)

        if pixel_values is None and pixel_values_videos is None:
            position_ids, rope_deltas = self.language_model.get_rope_index(
                input_ids, attention_mask=mask
            )
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids),
                position_ids=position_ids,
                rope_deltas=rope_deltas,
            )

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        cached = kwargs.get("cached_image_features", None)

        if cached is not None:
            vis_mask = (input_ids == self.config.image_token_index) | (
                input_ids == self.config.video_token_index
            )
            inputs_embeds = masked_scatter(
                inputs_embeds,
                mx.broadcast_to(vis_mask[..., None], inputs_embeds.shape),
                cached,
            )
        else:
            if pixel_values is not None:
                img_features = self.encode_vision(pixel_values, image_grid_thw)
                img_mask = input_ids == self.config.image_token_index
                n_img_tokens = int(img_mask.sum().item())
                if n_img_tokens != img_features.shape[0]:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens={n_img_tokens}, features={img_features.shape[0]}"
                    )
                inputs_embeds = masked_scatter(
                    inputs_embeds,
                    mx.broadcast_to(img_mask[..., None], inputs_embeds.shape),
                    img_features,
                )

            if pixel_values_videos is not None:
                vid_features = self.encode_vision(pixel_values_videos, video_grid_thw)
                vid_mask = input_ids == self.config.video_token_index
                n_vid_tokens = int(vid_mask.sum().item())
                if n_vid_tokens != vid_features.shape[0]:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens={n_vid_tokens}, features={vid_features.shape[0]}"
                    )
                inputs_embeds = masked_scatter(
                    inputs_embeds,
                    mx.broadcast_to(vid_mask[..., None], inputs_embeds.shape),
                    vid_features,
                )

        position_ids, rope_deltas = self.language_model.get_rope_index(
            input_ids, image_grid_thw, video_grid_thw, mask
        )
        mx.eval(inputs_embeds)
        mx.clear_cache()
        return InputEmbeddingsFeatures(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
        )

    @property
    def layers(self):
        return self.language_model.model.layers

    def quant_predicate(self, path: str, module: nn.Module) -> bool:
        if path.startswith(("vision_tower", "projector")):
            return False
        return True

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        features = self.get_input_embeddings(
            input_ids, pixel_values, mask=mask, **kwargs
        )
        kwargs.update(
            {
                "pixel_values": pixel_values,
                **features.to_dict(),
            }
        )
        return self.language_model(input_ids, mask=mask, cache=cache, **kwargs)

    def sanitize(self, weights: dict) -> dict:
        sanitized = {}
        for key, value in weights.items():
            if _DROPPED_GENERATOR_KEY_RE.match(key):
                continue
            if key.startswith("expert."):
                # Ignore Alpamayo expert weights when loading standalone Cosmos3Edge VLM
                continue
            if key.startswith("vlm."):
                key = key[len("vlm.") :]
            if key.startswith("model.language_model."):
                key = "language_model.model." + key[len("model.language_model.") :]
            elif key.startswith("model.visual."):
                key = "vision_tower." + key[len("model.visual.") :]
            elif key.startswith("model.projector."):
                key = "projector." + key[len("model.projector.") :]
            elif key.startswith("lm_head."):
                key = "language_model." + key
            sanitized[key] = value
        return sanitized
