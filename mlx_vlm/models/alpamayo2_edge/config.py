import copy
import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..base import BaseModelConfig
from ..cosmos3_edge.config import ModelConfig as Cosmos3EdgeModelConfig
from ..cosmos3_edge.config import TextConfig, VisionConfig


@dataclass
class ExpertConfig(BaseModelConfig):
    llm_config: TextConfig = field(
        default_factory=lambda: TextConfig(hidden_size=512, intermediate_size=3840)
    )
    action_in_proj_cfg: Dict[str, Any] = field(
        default_factory=lambda: {
            "hidden_size": 512,
            "max_freq": 100.0,
            "num_enc_layers": 2,
            "num_fourier_feats": 20,
        }
    )
    action_out_proj_cfg: Dict[str, Any] = field(default_factory=dict)
    action_space_cfg: Dict[str, Any] = field(
        default_factory=lambda: {
            "_target_": "alpamayo2_edge.action_space.UnicycleAccelCurvatureCourseActionSpace",
            "accel_mean": 0.02902694707164455,
            "accel_std": 0.6810426736454882,
            "curvature_mean": 0.0002692167976330542,
            "curvature_std": 0.026148280660833106,
            "accel_bounds": [-9.8, 9.8],
            "curvature_bounds": [-0.33, 0.33],
            "dt": 0.1,
            "n_waypoints": 64,
            "slip_gradient": 0.0025,
            "theta_lambda": 3e-6,
            "theta_ridge": 1e-8,
            "v_lambda": 1e-6,
            "v_ridge": 1e-4,
            "a_lambda": 1e-4,
            "a_ridge": 1e-4,
            "kappa_lambda": 1e-4,
            "kappa_ridge": 1e-4,
        }
    )
    diffusion_cfg: Dict[str, Any] = field(
        default_factory=lambda: {"int_method": "euler", "num_inference_steps": 10}
    )
    expert_non_causal_attention: bool = True

    def __post_init__(self):
        if isinstance(self.llm_config, dict):
            self.llm_config = TextConfig.from_dict(self.llm_config)


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    vlm_config: Optional[Cosmos3EdgeModelConfig] = None
    expert_config: Optional[ExpertConfig] = field(default_factory=ExpertConfig)
    model_type: str = "alpamayo2_edge"
    projector_hidden_size: int = 11520
    image_token_id: int = 19
    video_token_id: int = 18
    vision_start_token_id: int = 20
    vision_end_token_id: int = 21
    image_token_index: Optional[int] = None
    video_token_index: Optional[int] = None
    tie_word_embeddings: bool = False
    skip_vision: bool = False
    enable_expert: bool = True
    history_vocab_size: int = 1000
    future_vocab_size: int = 3000
    traj_vocab_size: int = 4000
    tokens_per_history_traj: int = 57
    tokens_per_future_traj: int = 128
    min_pixels: int = 163840
    max_pixels: int = 196608
    frame_label: str = "relative_time"
    token_layout: str = "camera_ts"
    include_camera_ids: bool = True
    padding_side: str = "left"
    hist_traj_tokenizer_cfg: Dict[str, Any] = field(
        default_factory=lambda: {"pad_origin_at_beginning": False}
    )
    future_traj_tokenizer_cfg: Dict[str, Any] = field(default_factory=dict)
    traj_ids: Dict[str, int] = field(
        default_factory=lambda: {
            "future_end": 135086,
            "future_id0": 132072,
            "future_pad": 135088,
            "future_start": 135084,
            "history_end": 135079,
            "history_id0": 131072,
            "history_pad": 135087,
            "history_start": 135077,
        }
    )
    vocab_size: int = 135168
    eos_token_id: Optional[Union[int, List[int]]] = 11
    quantization: Optional[Dict[str, Any]] = None
    quantization_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if isinstance(self.vlm_config, dict):
            self.vlm_config = Cosmos3EdgeModelConfig.from_dict(self.vlm_config)
        if self.vlm_config is not None:
            self.text_config = self.vlm_config.text_config
            self.vision_config = self.vlm_config.vision_config
            self.projector_hidden_size = self.vlm_config.projector_hidden_size
            self.image_token_id = self.vlm_config.image_token_id
            self.video_token_id = self.vlm_config.video_token_id
            self.vision_start_token_id = self.vlm_config.vision_start_token_id
            self.vision_end_token_id = self.vlm_config.vision_end_token_id
        else:
            if isinstance(self.text_config, dict):
                self.text_config = TextConfig.from_dict(self.text_config)
            if isinstance(self.vision_config, dict):
                self.vision_config = VisionConfig.from_dict(self.vision_config)
            self.vlm_config = Cosmos3EdgeModelConfig(
                text_config=self.text_config,
                vision_config=self.vision_config,
                projector_hidden_size=self.projector_hidden_size,
                image_token_id=self.image_token_id,
                video_token_id=self.video_token_id,
                vision_start_token_id=self.vision_start_token_id,
                vision_end_token_id=self.vision_end_token_id,
                skip_vision=self.skip_vision,
            )
        if isinstance(self.expert_config, dict):
            self.expert_config = ExpertConfig.from_dict(self.expert_config)
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id
        if self.video_token_index is None:
            self.video_token_index = self.video_token_id
        self.vocab_size = self.text_config.vocab_size
        self.traj_vocab_size = self.history_vocab_size + self.future_vocab_size
        # Include <|traj_future_start|> as an EOS boundary for VQA/text generation
        anchor_id = int(self.traj_ids.get("future_start", 135084))
        if self.eos_token_id is None:
            self.eos_token_id = [11, anchor_id]
        elif isinstance(self.eos_token_id, int):
            self.eos_token_id = list(dict.fromkeys([self.eos_token_id, anchor_id]))
        else:
            self.eos_token_id = list(dict.fromkeys([*self.eos_token_id, anchor_id]))
        axes = 4 if self.hist_traj_tokenizer_cfg.get("predict_yaw", False) else 3
        self.history_num_steps = self.tokens_per_history_traj // axes + int(
            not self.hist_traj_tokenizer_cfg.get("pad_origin_at_beginning", True)
        )

    @classmethod
    def from_dict(cls, params: dict):
        merged = copy.deepcopy(params)
        vlm_dict = merged.get("vlm_config")
        if isinstance(vlm_dict, dict):
            if not merged.get("text_config") and vlm_dict.get("text_config"):
                merged["text_config"] = vlm_dict["text_config"]
                params["text_config"] = vlm_dict["text_config"]
            if not merged.get("vision_config") and vlm_dict.get("vision_config"):
                merged["vision_config"] = vlm_dict["vision_config"]
                params["vision_config"] = vlm_dict["vision_config"]
            for k in (
                "projector_hidden_size",
                "image_token_id",
                "video_token_id",
                "vision_start_token_id",
                "vision_end_token_id",
            ):
                if k not in merged and k in vlm_dict:
                    merged[k] = vlm_dict[k]
        return cls(
            **{
                k: v
                for k, v in merged.items()
                if k in inspect.signature(cls).parameters
            }
        )
