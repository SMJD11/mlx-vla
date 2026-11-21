"""Unit tests for Cosmos3-Edge and Alpamayo2-Edge MLX models and processors."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from mlx_vlm.models.alpamayo2_edge import (
    ExpertConfig,
    Model as Alpamayo2EdgeModel,
    ModelConfig as Alpamayo2EdgeConfig,
)
from mlx_vlm.models.alpamayo2_edge.action_space import (
    DeltaTrajectoryTokenizer,
    UnicycleAccelCurvatureCourseActionSpace,
    fuse_history_traj_tokens,
)
from mlx_vlm.models.cosmos3_edge import (
    Model as Cosmos3EdgeModel,
    ModelConfig as Cosmos3EdgeConfig,
    TextConfig,
    VisionConfig,
)
from mlx_vlm.models.cosmos3_edge.processing_cosmos3_edge import (
    Cosmos3EdgeImageProcessor,
    smart_resize_image,
)


def _tiny_cosmos_config() -> Cosmos3EdgeConfig:
    return Cosmos3EdgeConfig(
        text_config=TextConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 100_000_000.0,
                "mrope_section": [2, 3, 3],
            },
        ),
        vision_config=VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_channels=3,
            num_patches=16,
            patch_size=16,
            spatial_merge_size=2,
        ),
        projector_hidden_size=64,
        image_token_id=19,
        video_token_id=18,
        vision_start_token_id=20,
        vision_end_token_id=21,
    )


def test_cosmos3_edge_image_processor_hwc_patch_layout():
    proc = Cosmos3EdgeImageProcessor(
        patch_size=16,
        temporal_patch_size=1,
        merge_size=2,
        min_pixels=64 * 64,
        max_pixels=64 * 64,
    )
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    out = proc([img])
    assert out["image_grid_thw"].tolist() == [[1, 4, 4]]
    assert out["pixel_values"].shape == (16, 16 * 16 * 3)
    assert np.allclose(out["pixel_values"], -1.0)


def test_cosmos3_edge_vlm_forward_text_and_multimodal():
    cfg = _tiny_cosmos_config()
    model = Cosmos3EdgeModel(cfg)
    mx.eval(model.parameters())

    # Text-only forward
    ids_txt = mx.array([[1, 5, 7, 9]], dtype=mx.int32)
    out_txt = model(ids_txt)
    mx.eval(out_txt.logits)
    assert out_txt.logits.shape == (1, 4, 256)

    # Multimodal forward with 4x4 patch grid -> (4*4)//4 = 4 image tokens
    ids_mm = mx.array([[1, 20, 19, 19, 19, 19, 21, 5]], dtype=mx.int32)
    patches = mx.random.normal(shape=(16, 16 * 16 * 3))
    grid_thw = mx.array([[1, 4, 4]], dtype=mx.int64)
    out_mm = model(ids_mm, pixel_values=patches, image_grid_thw=grid_thw)
    mx.eval(out_mm.logits)
    assert out_mm.logits.shape == (1, 8, 256)


def test_alpamayo2_edge_end_to_end_tiny_trajectory_sampling():
    vlm_cfg = _tiny_cosmos_config()
    expert_llm = TextConfig(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 100_000_000.0,
            "mrope_section": [2, 3, 3],
        },
    )
    cfg = Alpamayo2EdgeConfig(
        vlm_config=vlm_cfg,
        expert_config=ExpertConfig(
            llm_config=expert_llm,
            action_in_proj_cfg={
                "hidden_size": 32,
                "num_enc_layers": 2,
                "num_fourier_feats": 8,
                "max_freq": 100.0,
            },
            action_space_cfg={
                "_target_": "alpamayo2_edge.action_space.UnicycleAccelCurvatureCourseActionSpace",
                "n_waypoints": 8,
                "dt": 0.1,
            },
            diffusion_cfg={"int_method": "euler", "num_inference_steps": 3},
        ),
        history_vocab_size=10,
        future_vocab_size=10,
        tokens_per_history_traj=6,
        tokens_per_future_traj=16,
        traj_ids={
            "history_id0": 200,
            "future_id0": 210,
            "history_pad": 225,
            "future_pad": 226,
            "future_start": 224,
            "future_end": 227,
            "history_start": 222,
            "history_end": 223,
        },
    )
    model = Alpamayo2EdgeModel(cfg)
    # Deterministic AnchorHead (mirrors NVIDIA's test_model.py AnchorHead) so random init emits anchor 224
    anchor_head = nn.Linear(64, 256, bias=True)
    anchor_head.weight = mx.zeros((256, 64))
    bias = np.full((256,), -100.0, dtype=np.float32)
    bias[224] = 100.0
    anchor_head.bias = mx.array(bias)
    model.language_model.lm_head = anchor_head
    mx.eval(model.parameters())

    # 3 history poses -> 2 deltas * 3 axes = 6 history tokens
    ego_xyz = np.zeros((1, 1, 3, 3), dtype=np.float32)
    ego_xyz[0, 0, :, 0] = np.array([-0.2, -0.1, 0.0], dtype=np.float32)
    ego_rot = np.broadcast_to(np.eye(3, dtype=np.float32), (1, 1, 3, 3, 3)).copy()

    input_ids = np.array([[1, 222, 225, 225, 225, 225, 225, 225, 223, 5]], dtype=np.int64)
    data = {
        "tokenized_data": {"input_ids": input_ids},
        "ego_history_xyz": ego_xyz,
        "ego_history_rot": ego_rot,
    }
    xyz, rot, logprob, extra = model.sample_trajectories(
        data,
        temperature=0.0,
        num_traj_sets=1,
        num_traj_samples=2,
        max_generation_length=4,
        diffusion_steps=3,
        seed=0,
    )
    assert xyz.shape == (1, 1, 2, 8, 3)
    assert rot.shape == (1, 1, 2, 8, 3, 3)
    assert np.isfinite(xyz).all()
    assert np.isfinite(rot).all()

    # Zero-thought fast path (`reasoning_mode="none"`)
    xyz_z, rot_z, _, _ = model.sample_trajectories(
        data,
        temperature=0.0,
        num_traj_sets=1,
        num_traj_samples=1,
        diffusion_steps=3,
        reasoning_mode="none",
        seed=0,
    )
    assert xyz_z.shape == (1, 1, 1, 8, 3)
    assert np.isfinite(xyz_z).all() and np.isfinite(rot_z).all()


if __name__ == "__main__":
    test_cosmos3_edge_image_processor_hwc_patch_layout()
    test_cosmos3_edge_vlm_forward_text_and_multimodal()
    test_alpamayo2_edge_end_to_end_tiny_trajectory_sampling()
    print("ALL UNIT TESTS PASSED!")

