import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "cosmos3_edge_vision"
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    num_patches: int = 256
    patch_size: int = 16
    spatial_merge_size: int = 2
    layer_norm_eps: float = 1e-6
    hidden_act: str = "gelu_pytorch_tanh"
    attention_dropout: float = 0.0


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "cosmos3_edge_text"
    vocab_size: int = 135168
    hidden_size: int = 2048
    intermediate_size: int = 9216
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: Optional[int] = 8
    head_dim: int = 128
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    hidden_act: str = "relu2"
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = False
    rope_theta: float = 100_000_000.0
    rope_parameters: Optional[Dict[str, Union[float, str, List[int]]]] = None
    rope_scaling: Optional[Dict[str, Union[float, str, List[int]]]] = None
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: Union[int, List[int]] = 11

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.rope_parameters is None:
            if self.rope_scaling is not None:
                self.rope_parameters = dict(self.rope_scaling)
            else:
                self.rope_parameters = {
                    "rope_type": "default",
                    "rope_theta": self.rope_theta,
                    "mrope_section": [24, 20, 20],
                }
        if "rope_theta" in self.rope_parameters:
            self.rope_theta = float(self.rope_parameters["rope_theta"])
        else:
            self.rope_parameters["rope_theta"] = self.rope_theta
        if "mrope_section" not in self.rope_parameters:
            self.rope_parameters["mrope_section"] = [24, 20, 20]


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    model_type: str = "cosmos3_edge"
    projector_hidden_size: int = 11520
    image_token_id: int = 19
    video_token_id: int = 18
    vision_start_token_id: int = 20
    vision_end_token_id: int = 21
    image_token_index: Optional[int] = None
    video_token_index: Optional[int] = None
    tie_word_embeddings: bool = False
    skip_vision: bool = False
    vocab_size: int = 135168
    eos_token_id: Optional[Union[int, List[int]]] = 11

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        if isinstance(self.vision_config, dict):
            self.vision_config = VisionConfig.from_dict(self.vision_config)
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id
        if self.video_token_index is None:
            self.video_token_index = self.video_token_id
        if self.text_config is not None:
            self.vocab_size = self.text_config.vocab_size

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
