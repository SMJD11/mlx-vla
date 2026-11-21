"""Full-precision MLX implementation of NVIDIA Alpamayo 2 Edge (VLM + Action Expert)."""

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import LanguageModelOutput
from ..cache import KVCache, make_prompt_cache
from ..cosmos3_edge import processing_cosmos3_edge  # noqa: F401
from ..cosmos3_edge.cosmos3_edge import Model as Cosmos3EdgeModel
from .action_space import DeltaTrajectoryTokenizer, fuse_history_traj_tokens
from .config import ModelConfig
from .expert import ExpertModel

CAMERA_DISPLAY_NAMES = {
    0: "Front left camera",
    1: "Front camera",
    2: "Front right camera",
    3: "Rear left camera",
    5: "Rear right camera",
    6: "Front telephoto camera",
}
CAMERA_FRAME_COUNTS = {0: 3, 1: 5, 2: 3, 3: 3, 5: 3, 6: 5}
CAMERA_PIXEL_BOUNDS = {
    cam: (163840, 196608) if cam in (1, 6) else (81920, 98304)
    for cam in CAMERA_FRAME_COUNTS
}
VQA_PIXEL_BOUNDS = (163840, 196608)
DEFAULT_SYSTEM_PROMPT = "You are a driving assistant that generates safe and accurate actions."
DRIVING_TASK = (
    "Your task: output the chain-of-thought reasoning of the driving process, "
    "then output the future trajectory."
)
CLOSED_THINKING = "<think></think>"
_META_ACTION_SPLIT_RE = re.compile(r"(?P<head>(?:Longitudinal|Lateral|Lane)\s*:)")


def _sample_next_tokens(
    logits: mx.array,
    temperature: float = 0.6,
    top_p: float = 0.98,
    top_k: Optional[int] = None,
) -> mx.array:
    """Sample `(B,)` token IDs from `(B, V)` logits."""
    if temperature <= 0.0:
        return mx.argmax(logits, axis=-1)

    scaled = logits.astype(mx.float32) / float(temperature)
    if top_k is not None and top_k > 0 and top_k < scaled.shape[-1]:
        kth = mx.topk(scaled, k=top_k, axis=-1)[..., -1:]
        scaled = mx.where(scaled < kth, mx.array(-float("inf"), dtype=mx.float32), scaled)

    if top_p is not None and 0.0 < top_p < 1.0:
        probs = mx.softmax(scaled, axis=-1)
        sorted_indices = mx.argsort(probs, axis=-1)
        sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
        cum_probs = mx.cumsum(sorted_probs, axis=-1)
        keep_sorted = cum_probs > (1.0 - float(top_p))
        last_col = mx.arange(probs.shape[-1])[None, :] == (probs.shape[-1] - 1)
        keep_sorted = keep_sorted | last_col
        filtered_sorted = mx.where(
            keep_sorted, sorted_probs, mx.array(0.0, dtype=mx.float32)
        )
        filtered_sorted = filtered_sorted / mx.sum(
            filtered_sorted, axis=-1, keepdims=True
        )
        sampled_sorted_idx = mx.random.categorical(mx.log(filtered_sorted + 1e-20))
        return mx.take_along_axis(
            sorted_indices, sampled_sorted_idx[:, None], axis=-1
        )[:, 0]

    return mx.random.categorical(scaled)


def _repeat_kv_cache(cache: List[KVCache], repeats: int) -> List[KVCache]:
    if repeats == 1:
        return cache
    out = []
    for c in cache:
        new_c = KVCache()
        if c.keys is not None:
            k, v = c.state
            new_c.keys = mx.repeat(k, repeats, axis=0)
            new_c.values = mx.repeat(v, repeats, axis=0)
            new_c.offset = c.offset
        out.append(new_c)
    return out


def _decode_cot_texts(tokenizer: Any, sequences_np: np.ndarray) -> Dict[str, List[str]]:
    decoded_batch = tokenizer.batch_decode(sequences_np.tolist(), skip_special_tokens=False)
    assistant_marker = "<|im_start|>assistant\n"
    traj_future_start = "<|traj_future_start|>"
    im_end = "<|im_end|>"

    cots = []
    meta_actions = []
    for raw in decoded_batch:
        text = raw
        if assistant_marker in text:
            text = text[text.rfind(assistant_marker) + len(assistant_marker) :]
        text = text.split(traj_future_start, 1)[0].split(im_end, 1)[0].strip()
        text = text.removeprefix(CLOSED_THINKING).strip()
        m = _META_ACTION_SPLIT_RE.search(text)
        if m is not None:
            cots.append(text[: m.start()].rstrip())
            meta_actions.append(text[m.start() :].lstrip())
        else:
            cots.append(text.rstrip())
            meta_actions.append("")
    return {"raw_outputs": decoded_batch, "cot": cots, "meta_action": meta_actions}


class Model(Cosmos3EdgeModel):
    """MLX Alpamayo 2 Edge Vision-Language-Action model.

    Compatible with standard ``mlx_vlm`` APIs (`load`, `generate`, `stream_generate`,
    `convert`, `server`) for VQA/chat, and provides `prepare_driving_inputs`,
    `generate_vqa`, and `sample_trajectories` for end-to-end Chain-of-Causation
    reasoning and 10-step Euler flow-matching trajectory prediction on Apple Silicon.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config
        self.history_traj_tokenizer = DeltaTrajectoryTokenizer(
            **config.hist_traj_tokenizer_cfg
        )
        self.expert = (
            ExpertModel(config.expert_config)
            if config.enable_expert and config.expert_config is not None
            else None
        )
        self._traj_logit_mask: Optional[mx.array] = None
        self.language_model._logit_mask = self._get_discrete_traj_mask()[0]

    def _get_discrete_traj_mask(self, dtype=mx.float32) -> mx.array:
        if self._traj_logit_mask is None:
            mask_np = np.zeros((1, 1, self.config.vocab_size), dtype=np.float32)
            h0 = int(self.config.traj_ids["history_id0"])
            f0 = int(self.config.traj_ids["future_id0"])
            mask_np[:, :, h0 : h0 + self.config.history_vocab_size] = -1e9
            mask_np[:, :, f0 : f0 + self.config.future_vocab_size] = -1e9
            self._traj_logit_mask = mx.array(mask_np)
        return self._traj_logit_mask.astype(dtype)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        """Standard mlx_vlm forward pass (discrete trajectory tokens are masked in language_model)."""
        return super().__call__(
            input_ids, pixel_values=pixel_values, mask=mask, cache=cache, **kwargs
        )

    def quant_predicate(self, path: str, module: nn.Module) -> bool:
        if path.startswith(
            (
                "vision_tower",
                "projector",
                "expert.action_in_proj",
                "expert.action_out_proj",
            )
        ):
            return False
        return True

    def sanitize(self, weights: dict) -> dict:
        vlm_part = {}
        expert_part = {}
        for k, v in weights.items():
            if k.startswith("expert."):
                if k.endswith(".freqs") or any(
                    k.endswith(s)
                    for s in (
                        "accel_mean",
                        "accel_std",
                        "curvature_mean",
                        "curvature_std",
                    )
                ):
                    continue
                expert_part[k] = v
            else:
                vlm_part[k] = v
        sanitized = super().sanitize(vlm_part)
        sanitized.update(expert_part)
        return sanitized

    def _chunked_prefill(
        self,
        input_ids: mx.array,
        full_embeds: mx.array,
        full_pos_ids: mx.array,
        prompt_mask: mx.array,
        cache: List[KVCache],
        prefill_step_size: int = 256,
        reserve_extra_tokens: int = 0,
    ) -> None:
        """Prefill `input_ids[:, :-1]` in bounded token chunks so Metal command buffers stay tiny."""
        prompt_length = input_ids.shape[1]
        prefill_len = prompt_length - 1
        if prefill_len <= 0:
            return

        for c in cache:
            if hasattr(c, "prefix_cache_reserve"):
                c.prefix_cache_reserve(prompt_length + reserve_extra_tokens)

        has_padding = int((prompt_mask[:, :prefill_len] == 0).sum().item()) > 0
        for start in range(0, prefill_len, prefill_step_size):
            end = min(start + prefill_step_size, prefill_len)
            chunk_mask = prompt_mask[:, :end] if has_padding else None
            _ = self.language_model(
                input_ids[:, start:end],
                inputs_embeds=full_embeds[:, start:end, :],
                mask=chunk_mask,
                cache=cache,
                position_ids=full_pos_ids[:, :, start:end],
                logits_to_keep=1,
            )
            mx.eval([c.state for c in cache])
        mx.clear_cache()

    def generate_vqa(
        self,
        images_or_sample: Any,
        processor: Any,
        question: str,
        max_new_tokens: int = 256,
        prefill_step_size: int = 256,
    ) -> str:
        """Generate a freeform driving VQA answer matching `alpamayo2_edge.text_tasks.generate_vqa_answer`."""
        mx.set_cache_limit(256 * 1024 * 1024)
        content: List[Dict[str, Any]] = []
        images: List[Any] = []
        pixel_bounds: List[Tuple[int, int]] = []

        if isinstance(images_or_sample, Mapping) and "image_frames" in images_or_sample:
            raw_frames = np.asarray(images_or_sample["image_frames"])
            if raw_frames.ndim == 5 and raw_frames.shape[1] == 1:
                raw_frames = raw_frames[:, 0]
            cam_ids = np.asarray(images_or_sample["camera_indices"], dtype=np.int64).reshape(-1)
            timestamps = np.asarray(images_or_sample["absolute_timestamps"], dtype=np.int64).reshape(-1)
            by_time = np.argsort(timestamps, kind="stable")
            order = by_time[np.argsort(cam_ids[by_time], kind="stable")]
            raw_frames, cam_ids, timestamps = raw_frames[order], cam_ids[order], timestamps[order]
            t_ref = int(timestamps[-1])
            prev_cam = None
            for frame, cam, ts in zip(raw_frames, cam_ids.tolist(), timestamps.tolist()):
                if self.config.include_camera_ids and cam != prev_cam:
                    content.append({"type": "text", "text": f"{CAMERA_DISPLAY_NAMES.get(cam, f'Camera {cam}')}: "})
                content.append({"type": "text", "text": f"<t={(ts - t_ref) / 1_000_000:z.1f}s>"})
                content.append({"type": "image"})
                images.append(frame)
                pixel_bounds.append(VQA_PIXEL_BOUNDS)
                prev_cam = cam
            sys_prompt = images_or_sample.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
        else:
            img_list = images_or_sample if isinstance(images_or_sample, (list, tuple)) else [images_or_sample]
            content.append({"type": "text", "text": "Front camera: "})
            for idx, im in enumerate(img_list):
                dt_s = (idx - (len(img_list) - 1)) * 0.5
                content.append({"type": "text", "text": f"<t={dt_s:z.1f}s>"})
                content.append({"type": "image"})
                images.append(im)
                pixel_bounds.append(VQA_PIXEL_BOUNDS)
            sys_prompt = DEFAULT_SYSTEM_PROMPT

        content.extend([{"type": "text", "text": "\n"}, {"type": "text", "text": question.strip()}])
        messages = [
            {"role": "system", "content": [{"type": "text", "text": str(sys_prompt)}]},
            {"role": "user", "content": content},
        ]
        prompt_text = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # VQA omits the trailing empty `<think></think>` block per `alpamayo2_edge.helper.prepare_vqa_inputs`
        if prompt_text.endswith(CLOSED_THINKING):
            prompt_text = prompt_text[: -len(CLOSED_THINKING)]

        proc_out = processor(
            text=prompt_text,
            images=images,
            pixel_bounds=pixel_bounds,
            padding=False,
            return_mm_token_type_ids=True,
        )
        input_ids = proc_out["input_ids"].astype(mx.int32)
        prompt_mask = mx.ones_like(input_ids)
        emb = self.get_input_embeddings(
            input_ids,
            pixel_values=proc_out.get("pixel_values"),
            image_grid_thw=proc_out.get("image_grid_thw"),
            mask=prompt_mask,
        )
        full_embeds = emb.inputs_embeds
        full_pos_ids = emb.position_ids
        rope_deltas = emb.rope_deltas

        cache = make_prompt_cache(self.language_model)
        self._chunked_prefill(
            input_ids,
            full_embeds,
            full_pos_ids,
            prompt_mask,
            cache,
            prefill_step_size=prefill_step_size,
            reserve_extra_tokens=max_new_tokens,
        )
        cur_tokens = input_ids[:, -1:]
        cur_embeds = full_embeds[:, -1:, :]
        cur_pos = full_pos_ids[:, :, -1:]
        del full_embeds
        mx.clear_cache()

        stop_ids = {11, int(self.config.traj_ids["future_start"])}
        if isinstance(self.config.eos_token_id, int):
            stop_ids.add(self.config.eos_token_id)
        elif self.config.eos_token_id:
            stop_ids.update(int(x) for x in self.config.eos_token_id)

        traj_mask_2d = self._get_discrete_traj_mask(mx.float32)[:, 0, :]
        out_ids: List[int] = []
        for _ in range(max_new_tokens):
            out = self.language_model(
                cur_tokens,
                inputs_embeds=cur_embeds,
                cache=cache,
                position_ids=cur_pos,
                logits_to_keep=1,
            )
            logits = out.logits[:, -1, :].astype(mx.float32) + traj_mask_2d
            tok_mx = mx.argmax(logits, axis=-1)
            mx.eval(tok_mx)
            tok = int(tok_mx.item())
            if tok in stop_ids:
                break
            out_ids.append(tok)
            cur_tokens = tok_mx[:, None]
            cur_embeds = None
            next_pos = mx.full((1, 1), cache[0].offset, dtype=mx.int32) + rope_deltas
            cur_pos = mx.broadcast_to(next_pos[None, :, :], (3, 1, 1))

        del cache
        mx.clear_cache()
        answer = processor.tokenizer.decode(out_ids, skip_special_tokens=True).strip()
        return answer.removeprefix(CLOSED_THINKING).strip()

    def prepare_driving_inputs(
        self,
        sample: Mapping[str, Any],
        processor: Any,
        nav_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Prepare raw multi-camera frames + ego history + nav text for `sample_trajectories`."""
        raw_frames = np.asarray(sample["image_frames"])
        if raw_frames.ndim == 5 and raw_frames.shape[1] == 1:
            raw_frames = raw_frames[:, 0]  # (N, 3, H, W)
        cam_ids = np.asarray(sample["camera_indices"], dtype=np.int64).reshape(-1)
        timestamps = np.asarray(sample["absolute_timestamps"], dtype=np.int64).reshape(-1)

        by_time = np.argsort(timestamps, kind="stable")
        order = by_time[np.argsort(cam_ids[by_time], kind="stable")]
        raw_frames = raw_frames[order]
        cam_ids = cam_ids[order]
        timestamps = timestamps[order]

        bounds_map = {**CAMERA_PIXEL_BOUNDS}
        if "camera_pixel_bounds" in sample:
            for k, v in sample["camera_pixel_bounds"].items():
                bounds_map[int(k)] = (int(v[0]), int(v[1]))

        t_ref = timestamps[-1]
        labels = [
            f"<t={(int(ts) - int(t_ref)) / 1_000_000:z.1f}s>" for ts in timestamps
        ]

        content: List[Dict[str, Any]] = []
        images: List[np.ndarray] = []
        pixel_bounds: List[Tuple[int, int]] = []
        prev_cam = None
        for frame, cam, label in zip(raw_frames, cam_ids.tolist(), labels):
            if self.config.include_camera_ids and cam != prev_cam:
                cam_name = CAMERA_DISPLAY_NAMES.get(cam, f"Camera {cam}")
                content.append({"type": "text", "text": f"{cam_name}: "})
            content.append({"type": "text", "text": label})
            content.append({"type": "image"})
            images.append(frame)
            pixel_bounds.append(
                bounds_map.get(cam, (self.config.min_pixels, self.config.max_pixels))
            )
            prev_cam = cam

        effective_nav = (
            nav_text
            if nav_text is not None
            else (
                sample["nav_text"][0]
                if isinstance(sample.get("nav_text"), (list, tuple))
                else str(sample.get("nav_text", "Continue straight"))
            )
        )

        content.extend(
            [
                {"type": "text", "text": "\n"},
                {
                    "type": "text",
                    "text": "<|traj_history_start|>"
                    + "<|traj_history|>" * self.config.tokens_per_history_traj
                    + "<|traj_history_end|>",
                },
                {"type": "text", "text": "\n"},
                {"type": "text", "text": f"Navigation input is: {effective_nav}"},
                {"type": "text", "text": "\n"},
                {"type": "text", "text": DRIVING_TASK},
            ]
        )

        sys_prompt = sample.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": [{"type": "text", "text": str(sys_prompt)}]},
            {"role": "user", "content": content},
        ]

        prompt_text = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not prompt_text.endswith(CLOSED_THINKING):
            prompt_text = prompt_text + CLOSED_THINKING

        proc_out = processor(
            text=prompt_text,
            images=images,
            pixel_bounds=pixel_bounds,
            padding=False,
            return_mm_token_type_ids=True,
        )

        ego_xyz = np.asarray(sample["ego_history_xyz"], dtype=np.float32)
        ego_rot = np.asarray(sample["ego_history_rot"], dtype=np.float32)
        if ego_xyz.ndim == 3:
            ego_xyz = ego_xyz[None, ...]
        if ego_rot.ndim == 4:
            ego_rot = ego_rot[None, ...]

        return {
            "tokenized_data": dict(proc_out),
            "ego_history_xyz": ego_xyz,
            "ego_history_rot": ego_rot,
            "prompt_text": prompt_text,
        }

    def _decode_and_diffuse_on_cache(
        self,
        cache: List[KVCache],
        last_tokens: mx.array,
        last_embeds: mx.array,
        last_pos: mx.array,
        prompt_mask: mx.array,
        effective_rope_deltas: mx.array,
        logit_mask: mx.array,
        anchor_id: int,
        prompt_length: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
        max_generation_length: int,
        diffusion_steps: int,
        diffusion_temperature: float,
        initial_noise: Optional[mx.array],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run CoT decode + 10-step expert diffusion on `cache`, returning `(sampled_action, gen_ids, first_anchor_pos)`."""
        cur_batch = last_tokens.shape[0]
        has_padding = int((prompt_mask == 0).sum().item()) > 0
        cur_tokens = last_tokens
        cur_embeds: Optional[mx.array] = last_embeds
        cur_pos = last_pos

        generated_cols: List[np.ndarray] = []
        anchor_found = np.zeros((cur_batch,), dtype=bool)
        anchor_cached = False

        for step_idx in range(max_generation_length + 1):
            cur_len = cache[0].offset + 1
            step_mask = (
                mx.concatenate(
                    [
                        prompt_mask,
                        mx.ones((cur_batch, cur_len - prompt_length), dtype=mx.int32),
                    ],
                    axis=1,
                )
                if has_padding
                else None
            )

            out = self.language_model(
                cur_tokens,
                inputs_embeds=cur_embeds,
                mask=step_mask,
                cache=cache,
                position_ids=cur_pos,
                logits_to_keep=1,
            )

            if anchor_found.all():
                anchor_cached = True
                mx.eval([c.state for c in cache])
                break

            if step_idx >= max_generation_length:
                break

            next_logits = out.logits[:, -1, :].astype(mx.float32) + logit_mask
            next_tok = _sample_next_tokens(
                next_logits, temperature=temperature, top_p=top_p, top_k=top_k
            )
            mx.eval(next_tok)
            next_tok_np = np.array(next_tok, dtype=np.int64)
            generated_cols.append(next_tok_np)
            anchor_found = anchor_found | (next_tok_np == anchor_id)

            cur_tokens = next_tok[:, None]
            cur_embeds = None
            next_pos_1d = (
                mx.full((cur_batch, 1), cache[0].offset, dtype=mx.int32)
                + effective_rope_deltas
            )
            cur_pos = mx.broadcast_to(next_pos_1d[None, :, :], (3, cur_batch, 1))

        if not generated_cols or not anchor_found.all():
            missing = np.where(~anchor_found)[0].tolist()
            raise ValueError(
                f"No <|traj_future_start|> token found in generated rows {missing} "
                f"within max_generation_length={max_generation_length}."
            )

        gen_np = np.stack(generated_cols, axis=1)
        first_anchor_pos = (gen_np == anchor_id).argmax(axis=1)
        offsets_np = first_anchor_pos + 1 + prompt_length
        prefix_length = int(offsets_np.max())

        if not anchor_cached and cache[0].offset < prefix_length:
            cur_out_mask = (
                mx.concatenate(
                    [
                        prompt_mask,
                        mx.ones(
                            (cur_batch, prefix_length - prompt_length), dtype=mx.int32
                        ),
                    ],
                    axis=1,
                )
                if has_padding
                else None
            )
            _ = self.language_model(
                cur_tokens,
                mask=cur_out_mask,
                cache=cache,
                position_ids=cur_pos,
                logits_to_keep=1,
            )
            mx.eval([c.state for c in cache])

        for c in cache:
            if c.offset > prefix_length:
                c.trim(c.offset - prefix_length)

        n_diff = self.expert.action_dims[0]
        offsets_mx = mx.array(offsets_np[:, None], dtype=mx.int32)
        diff_pos = (
            mx.arange(n_diff, dtype=mx.int32)[None, :]
            + effective_rope_deltas
            + offsets_mx
        )
        expert_pos_ids = mx.broadcast_to(
            diff_pos[None, :, :], (3, cur_batch, n_diff)
        )

        attn_mask_np = np.zeros(
            (cur_batch, 1, n_diff, prefix_length + n_diff), dtype=np.float32
        )
        cache_cols = np.arange(prefix_length)[None, :]
        invalid_cols = cache_cols >= offsets_np[:, None]
        attn_mask_np[..., :prefix_length] = np.where(
            invalid_cols[:, None, None, :], -1e9, 0.0
        )
        if has_padding:
            pm_np = np.array(prompt_mask)
            attn_mask_np[..., :prompt_length] = np.where(
                pm_np[:, None, None, :] == 0,
                -1e9,
                attn_mask_np[..., :prompt_length],
            )
        expert_attn_mask = mx.array(attn_mask_np)

        sampled_action_mx = self.expert.sample_actions(
            cache=cache,
            prefix_length=prefix_length,
            position_ids=expert_pos_ids,
            attention_mask=expert_attn_mask,
            batch_size=cur_batch,
            inference_steps=diffusion_steps,
            temperature=diffusion_temperature,
            initial_noise=initial_noise,
            seed=None,
        )
        return np.array(sampled_action_mx, dtype=np.float32), gen_np, first_anchor_pos

    def sample_trajectories(
        self,
        data: Mapping[str, Any],
        tokenizer: Any = None,
        top_p: float = 0.98,
        top_k: Optional[int] = None,
        temperature: float = 0.6,
        num_traj_sets: int = 1,
        num_traj_samples: int = 1,
        max_generation_length: int = 256,
        diffusion_steps: int = 10,
        diffusion_temperature: float = 1.0,
        initial_noise: Optional[mx.array] = None,
        seed: Optional[int] = None,
        prefill_step_size: int = 256,
        low_memory: bool = True,
        return_extra: bool = True,
    ) -> tuple:
        """Run VLM chunked prefill -> CoT generation -> 10-step Euler flow matching -> trajectory decode.

        When `low_memory=True` (default) and `batch_size == 1`, multiple trajectory candidates
        (`num_traj_sets * num_traj_samples > 1`) reuse the single shared prefill KV cache in-place
        by trimming back to `prompt_length - 1` after each candidate instead of duplicating multi-hundred-MB
        KV caches across batch rows.
        """
        if self.expert is None:
            raise ValueError("Model was initialized without an expert.")

        mx.set_cache_limit(256 * 1024 * 1024)
        if seed is not None:
            mx.random.seed(seed)

        tokenized = dict(data["tokenized_data"])
        raw_ids = np.asarray(tokenized["input_ids"], dtype=np.int64)
        ego_xyz = np.asarray(data["ego_history_xyz"], dtype=np.float32)
        ego_rot = np.asarray(data["ego_history_rot"], dtype=np.float32)
        if ego_xyz.ndim == 3:
            ego_xyz = ego_xyz[None, ...]
        if ego_rot.ndim == 4:
            ego_rot = ego_rot[None, ...]

        hist_pad_id = int(self.config.traj_ids["history_pad"])
        if (raw_ids == hist_pad_id).any():
            fused_ids_np = fuse_history_traj_tokens(
                raw_ids,
                ego_xyz,
                ego_rot,
                self.history_traj_tokenizer,
                self.config.traj_ids,
            )
        else:
            fused_ids_np = raw_ids

        input_ids = mx.array(fused_ids_np, dtype=mx.int32)
        if "attention_mask" in tokenized and tokenized["attention_mask"] is not None:
            prompt_mask = mx.array(np.asarray(tokenized["attention_mask"]), dtype=mx.int32)
        else:
            prompt_mask = mx.ones_like(input_ids)

        pixel_values = tokenized.get("pixel_values")
        if pixel_values is not None and not isinstance(pixel_values, mx.array):
            pixel_values = mx.array(np.asarray(pixel_values))
        image_grid_thw = tokenized.get("image_grid_thw")
        if image_grid_thw is not None and not isinstance(image_grid_thw, mx.array):
            image_grid_thw = mx.array(np.asarray(image_grid_thw, dtype=np.int64))

        batch_size, prompt_length = input_ids.shape
        n_samples_total = num_traj_sets * num_traj_samples
        expert_batch = batch_size * n_samples_total
        n_diff = self.expert.action_dims[0]

        # 1. Micro-batched vision encode + multimodal embedding assembly
        emb_features = self.get_input_embeddings(
            input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mask=prompt_mask,
        )
        full_embeds = emb_features.inputs_embeds
        full_pos_ids = emb_features.position_ids
        rope_deltas = emb_features.rope_deltas

        last_tokens_base = input_ids[:, -1:]
        last_embeds_base = full_embeds[:, -1:, :]
        last_pos_base = full_pos_ids[:, :, -1:]
        mx.eval(last_embeds_base)

        # 2. Chunked shared prefill of `input_ids[:, :-1]`
        cache = make_prompt_cache(self.language_model)
        use_inplace_reuse = low_memory and batch_size == 1 and n_samples_total > 1
        reserve_extra = (max_generation_length + n_diff + 4) if (n_samples_total == 1 or use_inplace_reuse) else 0
        self._chunked_prefill(
            input_ids,
            full_embeds,
            full_pos_ids,
            prompt_mask,
            cache,
            prefill_step_size=prefill_step_size,
            reserve_extra_tokens=reserve_extra,
        )
        del full_embeds
        mx.clear_cache()

        # Build static logit mask blocking discrete trajectory codes & non-anchor EOS
        anchor_id = int(self.config.traj_ids["future_start"])
        vocab_size = self.config.vocab_size
        logit_mask_np = np.zeros((1, vocab_size), dtype=np.float32)
        h0 = int(self.config.traj_ids["history_id0"])
        f0 = int(self.config.traj_ids["future_id0"])
        logit_mask_np[:, h0 : h0 + self.config.history_vocab_size] = -np.inf
        logit_mask_np[:, f0 : f0 + self.config.future_vocab_size] = -np.inf
        eos_cfg = self.config.eos_token_id
        eos_list = [eos_cfg] if isinstance(eos_cfg, int) else list(eos_cfg or [])
        for eid in eos_list:
            if int(eid) != anchor_id:
                logit_mask_np[:, int(eid)] = -np.inf
        logit_mask = mx.array(logit_mask_np)

        pad_counts_base = mx.sum((prompt_mask == 0).astype(mx.int32), axis=-1, keepdims=True)
        effective_rope_base = rope_deltas - pad_counts_base

        # 3. Run candidate(s): in-place cache reuse on multi-sample saves gigabytes of Metal RAM
        if use_inplace_reuse:
            prefill_saved_offset = prompt_length - 1
            actions_list = []
            gens_list = []
            anchors_list = []
            for s_idx in range(n_samples_total):
                for c in cache:
                    if c.offset > prefill_saved_offset:
                        c.trim(c.offset - prefill_saved_offset)
                noise_s = (
                    initial_noise[s_idx : s_idx + 1]
                    if initial_noise is not None
                    else None
                )
                act_s, gen_s, anc_s = self._decode_and_diffuse_on_cache(
                    cache=cache,
                    last_tokens=last_tokens_base,
                    last_embeds=last_embeds_base,
                    last_pos=last_pos_base,
                    prompt_mask=prompt_mask,
                    effective_rope_deltas=effective_rope_base,
                    logit_mask=logit_mask,
                    anchor_id=anchor_id,
                    prompt_length=prompt_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_generation_length=max_generation_length,
                    diffusion_steps=diffusion_steps,
                    diffusion_temperature=diffusion_temperature,
                    initial_noise=noise_s,
                )
                actions_list.append(act_s)
                gens_list.append(gen_s[0])
                anchors_list.append(int(anc_s[0]))
                mx.clear_cache()

            sampled_action_np = np.concatenate(actions_list, axis=0)
            max_gen_w = max(g.shape[0] for g in gens_list)
            pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
            gen_np = np.full((n_samples_total, max_gen_w), pad_id, dtype=np.int64)
            for idx, g in enumerate(gens_list):
                gen_np[idx, : g.shape[0]] = g
            first_anchor_pos = np.array(anchors_list, dtype=np.int64)
        else:
            cache = _repeat_kv_cache(cache, n_samples_total)
            sampled_action_np, gen_np, first_anchor_pos = self._decode_and_diffuse_on_cache(
                cache=cache,
                last_tokens=mx.repeat(last_tokens_base, n_samples_total, axis=0),
                last_embeds=mx.repeat(last_embeds_base, n_samples_total, axis=0),
                last_pos=mx.repeat(last_pos_base, n_samples_total, axis=1),
                prompt_mask=mx.repeat(prompt_mask, n_samples_total, axis=0),
                effective_rope_deltas=mx.repeat(effective_rope_base, n_samples_total, axis=0),
                logit_mask=logit_mask,
                anchor_id=anchor_id,
                prompt_length=prompt_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_generation_length=max_generation_length,
                diffusion_steps=diffusion_steps,
                diffusion_temperature=diffusion_temperature,
                initial_noise=initial_noise,
            )

        del cache
        mx.clear_cache()

        # 4. Decode actions -> ego-frame XYZ + SO(3) rotations via Course Action Space
        hist_xyz_last = np.repeat(ego_xyz[:, -1], n_samples_total, axis=0)
        hist_rot_last = np.repeat(ego_rot[:, -1], n_samples_total, axis=0)
        pred_xyz, pred_rot = self.expert.action_space.action_to_traj(
            sampled_action_np, hist_xyz_last, hist_rot_last
        )

        group_shape = (batch_size, num_traj_sets, num_traj_samples)
        pred_xyz = pred_xyz.reshape(*group_shape, *pred_xyz.shape[1:])
        pred_rot = pred_rot.reshape(*group_shape, *pred_rot.shape[1:])
        logprob = np.zeros(pred_xyz.shape[:-1], dtype=np.float32)

        if not return_extra:
            return pred_xyz, pred_rot, logprob

        pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        clean_gen = gen_np.copy()
        for r in range(expert_batch):
            clean_gen[r, first_anchor_pos[r] + 1 :] = pad_id
        full_seqs = np.concatenate(
            [np.repeat(fused_ids_np, n_samples_total, axis=0), clean_gen], axis=1
        )

        extra: Dict[str, Any] = {
            "sampled_action": sampled_action_np.reshape(
                *group_shape, *self.expert.action_dims
            )
        }
        if tokenizer is not None:
            decoded = _decode_cot_texts(tokenizer, full_seqs)
            for k, v in decoded.items():
                extra[k] = np.asarray(v, dtype=object).reshape(group_shape)
        gen_masked = gen_np.copy()
        for r in range(expert_batch):
            gen_masked[r, first_anchor_pos[r] + 1 :] = -1
        extra["generated_ids"] = gen_masked
        return pred_xyz, pred_rot, logprob, extra
