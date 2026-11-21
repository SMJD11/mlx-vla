"""Processor for Cosmos3-Edge and Alpamayo2-Edge (NumPy + PIL, torch-free)."""

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import ImageProcessingMixin
from transformers.processing_utils import ProcessorMixin
from transformers.video_processing_utils import BaseVideoProcessor

from ..base import install_auto_processor_patch, load_chat_template, to_mlx


def smart_resize_image(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 163840,
    max_pixels: int = 196608,
) -> Tuple[int, int]:
    """Port of HF qwen2_vl.smart_resize used by Cosmos3-Edge & Alpamayo2-Edge."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"Absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _to_chw_uint8(img) -> np.ndarray:
    """Convert PIL.Image, path, or HWC/CHW array to (3, H, W) uint8 numpy array."""
    if isinstance(img, (str, Path)):
        img = Image.open(img)
    if hasattr(img, "convert"):
        arr = np.array(img.convert("RGB"), dtype=np.uint8)  # (H, W, 3)
        return np.transpose(arr, (2, 0, 1))
    arr = np.asarray(img)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=0)
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4) and arr.shape[0] not in (1, 3, 4):
        arr = np.transpose(arr, (2, 0, 1))
    if arr.shape[0] == 4:
        arr = arr[:3]
    elif arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
            arr = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _bicubic_resize_chw_uint8(chw: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Bicubic resize on uint8 BEFORE normalization (matches Alpamayo2-Edge training/inference)."""
    _, h, w = chw.shape
    if h == target_h and w == target_w:
        return chw
    pil = Image.fromarray(np.transpose(chw, (1, 2, 0)), mode="RGB")
    resized = pil.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
    return np.transpose(np.array(resized, dtype=np.uint8), (2, 0, 1))


class Cosmos3EdgeImageProcessor(ImageProcessingMixin):
    """NumPy/PIL port of Cosmos3EdgeImageProcessor.

    Crucial layout detail: Cosmos3-Edge flattens each 2x2 block of patches in
    HWC-temporal order ``(grid_h//ms, grid_w//ms, ms, ms, ps, ps, C, tps)``,
    NOT Qwen3-VL's ``(..., C, tps, ps, ps)``.
    """

    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 1,
        merge_size: int = 2,
        min_pixels: int = 163840,
        max_pixels: int = 196608,
        do_resize: bool = True,
        do_rescale: bool = True,
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = True,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or [0.5, 0.5, 0.5]
        self.image_std = image_std or [0.5, 0.5, 0.5]
        self.do_convert_rgb = do_convert_rgb

    def _process_one(
        self,
        image_chw_uint8: np.ndarray,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        do_resize: Optional[bool] = None,
    ) -> Tuple[np.ndarray, List[int]]:
        c, h, w = image_chw_uint8.shape
        factor = self.patch_size * self.merge_size
        should_resize = self.do_resize if do_resize is None else do_resize

        if should_resize:
            target_h, target_w = smart_resize_image(
                h,
                w,
                factor=factor,
                min_pixels=self.min_pixels if min_pixels is None else min_pixels,
                max_pixels=self.max_pixels if max_pixels is None else max_pixels,
            )
            frame = _bicubic_resize_chw_uint8(image_chw_uint8, target_h, target_w)
        else:
            target_h, target_w = h, w
            frame = image_chw_uint8

        img = frame.astype(np.float32)
        if self.do_rescale:
            img = img * np.float32(self.rescale_factor)
        if self.do_normalize:
            mean = np.asarray(self.image_mean, dtype=np.float32)[:, None, None]
            std = np.asarray(self.image_std, dtype=np.float32)[:, None, None]
            img = (img - mean) / std

        ps = self.patch_size
        ms = self.merge_size
        tps = self.temporal_patch_size
        grid_t = 1
        grid_h = target_h // ps
        grid_w = target_w // ps

        # (C, gh//ms, ms, ps, gw//ms, ms, ps) -> (gh//ms, gw//ms, ms, ms, ps, ps, C)
        patches = img.reshape(c, grid_h // ms, ms, ps, grid_w // ms, ms, ps)
        patches = np.transpose(patches, (1, 4, 2, 5, 3, 6, 0))
        if tps > 1:
            patches = np.repeat(patches[..., None], tps, axis=-1)
        else:
            patches = patches[..., None]

        flat_patches = patches.reshape(grid_h * grid_w, ps * ps * c * tps)
        return flat_patches, [grid_t, grid_h, grid_w]

    def __call__(
        self,
        images,
        pixel_bounds: Optional[Sequence[Tuple[int, int]]] = None,
        do_resize: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        if not isinstance(images, list):
            images = [images]
        all_patches = []
        all_thw = []
        for idx, raw_img in enumerate(images):
            chw = _to_chw_uint8(raw_img)
            lo, hi = (
                pixel_bounds[idx]
                if pixel_bounds is not None
                else (kwargs.get("min_pixels"), kwargs.get("max_pixels"))
            )
            patches, thw = self._process_one(
                chw, min_pixels=lo, max_pixels=hi, do_resize=do_resize
            )
            all_patches.append(patches)
            all_thw.append(thw)
        return {
            "pixel_values": np.concatenate(all_patches, axis=0),
            "image_grid_thw": np.asarray(all_thw, dtype=np.int64),
        }

    def preprocess(self, images, **kwargs):
        return self(images, **kwargs)


class Cosmos3EdgeVideoProcessor(BaseVideoProcessor):
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 1,
        merge_size: int = 2,
        min_pixels: int = 64 * 64,
        max_pixels: int = 24 * 1024 * 1024,
        do_rescale: bool = True,
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = True,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        fps: float = 2.0,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or [0.5, 0.5, 0.5]
        self.image_std = image_std or [0.5, 0.5, 0.5]
        self.fps = fps
        self._img_proc = Cosmos3EdgeImageProcessor(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=self.image_mean,
            image_std=self.image_std,
        )

    def __call__(self, videos, **kwargs) -> Dict[str, np.ndarray]:
        if not isinstance(videos, list):
            videos = [videos]
        all_patches = []
        all_thw = []
        factor = self.patch_size * self.merge_size
        for video in videos:
            vid = np.asarray(video)  # (T, C, H, W) or (T, H, W, C)
            if vid.ndim == 4 and vid.shape[-1] == 3 and vid.shape[1] != 3:
                vid = np.transpose(vid, (0, 3, 1, 2))
            t_frames, _, h, w = vid.shape
            per_frame_max = max(factor * factor, self.max_pixels // max(1, t_frames))
            per_frame_min = max(factor * factor, self.min_pixels // max(1, t_frames))
            target_h, target_w = smart_resize_image(
                h, w, factor=factor, min_pixels=per_frame_min, max_pixels=per_frame_max
            )
            frame_patches = []
            for f_idx in range(t_frames):
                chw = _bicubic_resize_chw_uint8(_to_chw_uint8(vid[f_idx]), target_h, target_w)
                p, thw = self._img_proc._process_one(chw, do_resize=False)
                frame_patches.append(p)
            all_patches.append(np.concatenate(frame_patches, axis=0))
            all_thw.append([t_frames, target_h // self.patch_size, target_w // self.patch_size])
        return {
            "pixel_values_videos": np.concatenate(all_patches, axis=0),
            "video_grid_thw": np.asarray(all_thw, dtype=np.int64),
        }


def _load_json_if_exists(model_path_or_id, filename: str) -> Optional[dict]:
    local = Path(model_path_or_id) / filename
    if local.is_file():
        return json.loads(local.read_text(encoding="utf-8"))
    if Path(model_path_or_id).exists():
        return None
    try:
        from huggingface_hub import hf_hub_download

        fetched = Path(hf_hub_download(str(model_path_or_id), filename))
        return json.loads(fetched.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_image_proc_kwargs(model_path_or_id) -> dict:
    proc_cfg = _load_json_if_exists(model_path_or_id, "processor_config.json") or {}
    raw = _load_json_if_exists(model_path_or_id, "preprocessor_config.json") or {}
    raw.update(proc_cfg.get("image_processor") or {})
    out = {}
    for k in (
        "patch_size",
        "temporal_patch_size",
        "merge_size",
        "image_mean",
        "image_std",
        "rescale_factor",
        "do_resize",
        "do_rescale",
        "do_normalize",
        "do_convert_rgb",
    ):
        if k in raw:
            out[k] = raw[k]
    size = raw.get("size") or {}
    if size.get("shortest_edge") is not None:
        out["min_pixels"] = size["shortest_edge"]
    if size.get("longest_edge") is not None:
        out["max_pixels"] = size["longest_edge"]
    if raw.get("min_pixels") is not None:
        out["min_pixels"] = raw["min_pixels"]
    if raw.get("max_pixels") is not None:
        out["max_pixels"] = raw["max_pixels"]
    return out


class Cosmos3EdgeProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer", "video_processor"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    video_processor_class = "AutoVideoProcessor"

    def check_argument_for_proper_class(self, argument_name, argument):
        return type(argument)

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs,
    ):
        self.image_token = getattr(tokenizer, "image_token", "<|image_pad|>")
        self.video_token = getattr(tokenizer, "video_token", "<|video_pad|>")
        self.vision_start_token = getattr(
            tokenizer, "vision_start_token", "<|vision_start|>"
        )
        self.vision_end_token = getattr(tokenizer, "vision_end_token", "<|vision_end|>")
        self.image_token_id = (
            tokenizer.convert_tokens_to_ids(self.image_token)
            if tokenizer is not None
            else 19
        )
        self.video_token_id = (
            tokenizer.convert_tokens_to_ids(self.video_token)
            if tokenizer is not None
            else 18
        )
        self.vision_start_token_id = (
            tokenizer.convert_tokens_to_ids(self.vision_start_token)
            if tokenizer is not None
            else 20
        )
        self.vision_end_token_id = (
            tokenizer.convert_tokens_to_ids(self.vision_end_token)
            if tokenizer is not None
            else 21
        )
        super().__init__(
            image_processor, tokenizer, video_processor, chat_template=chat_template
        )

    def __call__(
        self,
        images=None,
        text: Optional[Union[str, List[str]]] = None,
        videos=None,
        pixel_bounds: Optional[Sequence[Tuple[int, int]]] = None,
        **kwargs,
    ) -> BatchFeature:
        return_tensors = kwargs.pop("return_tensors", None)
        return_mm_token_type_ids = kwargs.pop("return_mm_token_type_ids", False)
        do_resize = kwargs.pop("do_resize", None)
        kwargs.pop("do_rescale", None)
        kwargs.pop("do_normalize", None)

        image_inputs = {}
        video_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(
                images=images, pixel_bounds=pixel_bounds, do_resize=do_resize
            )
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_grid_thw = None

        if videos is not None:
            video_inputs = self.video_processor(videos=videos)
            video_grid_thw = video_inputs["video_grid_thw"]
        else:
            video_grid_thw = None

        if isinstance(text, str):
            text = [text]
        else:
            text = list(text)

        if image_grid_thw is not None:
            merge_len = self.image_processor.merge_size**2
            idx = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    n_tok = int(image_grid_thw[idx].prod()) // merge_len
                    text[i] = text[i].replace(
                        self.image_token, "<|img_ph|>" * n_tok, 1
                    )
                    idx += 1
                text[i] = text[i].replace("<|img_ph|>", self.image_token)

        if video_grid_thw is not None:
            merge_len = self.video_processor.merge_size**2
            vid_wrapper = (
                self.vision_start_token + self.video_token + self.vision_end_token
            )
            pattern = f"({re.escape(vid_wrapper)}|{re.escape(self.video_token)})"
            idx = 0
            for i in range(len(text)):
                def _repl(_m):
                    nonlocal idx
                    t_f, h_g, w_g = video_grid_thw[idx]
                    idx += 1
                    n_tok = int(h_g * w_g) // merge_len
                    fps = getattr(self.video_processor, "fps", 2.0) or 2.0
                    return "".join(
                        f"<{f_i / fps:.1f} seconds>{self.vision_start_token}"
                        f"{self.video_token * n_tok}{self.vision_end_token}"
                        for f_i in range(int(t_f))
                    )
                text[i] = re.sub(pattern, _repl, text[i])

        text_inputs = self.tokenizer(text, **kwargs)
        if return_mm_token_type_ids:
            arr_ids = np.asarray(text_inputs["input_ids"])
            mm_types = np.zeros_like(arr_ids)
            mm_types[arr_ids == self.image_token_id] = 1
            mm_types[arr_ids == self.video_token_id] = 2
            text_inputs["mm_token_type_ids"] = mm_types

        combined = {**text_inputs, **image_inputs, **video_inputs}
        if return_tensors == "np":
            return BatchFeature(
                data={
                    k: np.asarray(v) if not isinstance(v, np.ndarray) else v
                    for k, v in combined.items()
                }
            )
        return BatchFeature(data=to_mlx(combined))

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def save_pretrained(self, save_directory, **kwargs):
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(save_dir))
            if getattr(self, "chat_template", None):
                (save_dir / "chat_template.jinja").write_text(
                    self.chat_template, encoding="utf-8"
                )
        proc_cfg = {
            "processor_class": "Cosmos3EdgeProcessor",
            "image_processor": {
                "image_processor_type": "Cosmos3EdgeImageProcessor",
                "do_convert_rgb": self.image_processor.do_convert_rgb,
                "do_normalize": self.image_processor.do_normalize,
                "do_rescale": self.image_processor.do_rescale,
                "do_resize": self.image_processor.do_resize,
                "image_mean": list(self.image_processor.image_mean),
                "image_std": list(self.image_processor.image_std),
                "merge_size": self.image_processor.merge_size,
                "patch_size": self.image_processor.patch_size,
                "resample": 3,
                "rescale_factor": self.image_processor.rescale_factor,
                "size": {
                    "shortest_edge": self.image_processor.min_pixels,
                    "longest_edge": self.image_processor.max_pixels,
                },
                "temporal_patch_size": self.image_processor.temporal_patch_size,
            },
            "video_processor": {
                "video_processor_type": "Cosmos3EdgeVideoProcessor",
                "do_convert_rgb": True,
                "do_normalize": self.video_processor.do_normalize,
                "do_rescale": self.video_processor.do_rescale,
                "do_resize": True,
                "fps": self.video_processor.fps,
                "image_mean": list(self.video_processor.image_mean),
                "image_std": list(self.video_processor.image_std),
                "merge_size": self.video_processor.merge_size,
                "patch_size": self.video_processor.patch_size,
                "resample": 3,
                "rescale_factor": self.video_processor.rescale_factor,
                "size": {
                    "shortest_edge": self.video_processor.min_pixels,
                    "longest_edge": self.video_processor.max_pixels,
                },
                "temporal_patch_size": self.video_processor.temporal_patch_size,
            },
        }
        (save_dir / "processor_config.json").write_text(
            json.dumps(proc_cfg, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import AutoTokenizer

        kwargs.pop("use_fast", None)
        kwargs.pop("trust_remote_code", None)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        load_chat_template(tokenizer, pretrained_model_name_or_path)

        img_kwargs = _extract_image_proc_kwargs(pretrained_model_name_or_path)
        image_processor = Cosmos3EdgeImageProcessor(**img_kwargs)
        video_processor = Cosmos3EdgeVideoProcessor(
            patch_size=img_kwargs.get("patch_size", 16),
            merge_size=img_kwargs.get("merge_size", 2),
            image_mean=img_kwargs.get("image_mean"),
            image_std=img_kwargs.get("image_std"),
        )
        chat_template = getattr(tokenizer, "chat_template", None)
        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
        )


__all__ = [
    "Cosmos3EdgeImageProcessor",
    "Cosmos3EdgeVideoProcessor",
    "Cosmos3EdgeProcessor",
    "smart_resize_image",
]

install_auto_processor_patch(["cosmos3_edge", "alpamayo2_edge"], Cosmos3EdgeProcessor)
