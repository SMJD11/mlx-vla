# Alpamayo 2 Edge & Cosmos 3 Edge (MLX-VLA)

Pure Apple Silicon **MLX** implementation of **NVIDIA Alpamayo 2 Edge (2B)** (`alpamayo2_edge`) and **NVIDIA Cosmos 3 Edge** (`cosmos3_edge`) with zero PyTorch/Torchvision runtime dependency, zero-copy unified-memory KV handoff into the 10-step Euler flow-matching Action Expert, and optional Apple Neural Engine (CoreML) async vision pipelining.

## Pre-Converted Hugging Face Checkpoints

### Alpamayo 2 Edge (2B) — Native `mlx_vlm.load(...)` Support

| Checkpoint | Precision | Disk / Metal RAM | Verified M3 Pro Notes |
|---|---|---:|---|
| **[`Safman/Alpamayo2-Edge-MLX-bf16`](https://huggingface.co/Safman/Alpamayo2-Edge-MLX-bf16)** | Full `BF16` | `5.30 GB` | Reference full precision (`ADE@3s = 0.507 m` on 5-clip HD nav suite) |
| **[`Safman/Alpamayo2-Edge-MLX-q8`](https://huggingface.co/Safman/Alpamayo2-Edge-MLX-q8)** | `Q8` LLM + Expert, `BF16` Vision/Action | `3.28 GB` | Virtually lossless vs `BF16` (`1.35 s` load) |
| **[`Safman/Alpamayo2-Edge-MLX-q6`](https://huggingface.co/Safman/Alpamayo2-Edge-MLX-q6)** | `Q6` LLM + Expert, `BF16` Vision/Action | `2.74 GB` | Balanced size & accuracy (`1.32 s` load) |
| **[`Safman/Alpamayo2-Edge-MLX-q4`](https://huggingface.co/Safman/Alpamayo2-Edge-MLX-q4)** | `Q4` LLM + Expert, `BF16` Vision/Action | `2.20 GB` | Smallest & fastest (`1.21 s` load; `1.73–2.5 Hz` w/ CoreML ANE async pipeline) |

### Alpamayo 1.5 (10B) — Mac MLX Port

| Checkpoint | Contents | Size | Notes |
|---|---|---:|---|
| **[`Safman/Alpamayo-1.5-Mac-MLX-4bit`](https://huggingface.co/Safman/Alpamayo-1.5-Mac-MLX-4bit)** | `vlm/` (4-bit `Cosmos-Reason2-8B` w/ 4,054 driving tokens) + `expert/` (36L Diffusion Action Expert) | `10.36 GB` | Complete standalone 10B Mac bundle (load `vlm/` via `mlx_vlm.load`) |
| **[`Safman/Alpamayo-1.5-Mac-DFlash`](https://huggingface.co/Safman/Alpamayo-1.5-Mac-DFlash)** | 2-layer DFlash speculative draft head | `0.94 GB` | Accelerates Chain-of-Causation decoding on Alpamayo 1.5 |
| [`Safman/Alpamayo-1.5-Mac-MLX-8bit`](https://huggingface.co/Safman/Alpamayo-1.5-Mac-MLX-8bit) | Action Expert weights only (`int8` + `fp16`) | `9.13 GB` | Requires `vlm/` from the 4-bit bundle |
| [`Safman/Alpamayo-1.5-Mac-MLX-16bit`](https://huggingface.co/Safman/Alpamayo-1.5-Mac-MLX-16bit) | Action Expert weights only (`fp16`) | `9.12 GB` | Requires `vlm/` from the 4-bit bundle |

---

## Installation

```bash
pip install "git+https://github.com/SMJD11/mlx-vla.git"
```

---

## Usage

### 1. Driving Reasoning (CoT) + 64-Waypoint Trajectory Prediction
```python
import numpy as np
from mlx_vlm import load

model, processor = load("Safman/Alpamayo2-Edge-MLX-q4")  # or -bf16, -q8, -q6

# sample dict:
#   image_frames:        (N, 3, H, W) uint8 CHW frames
#   camera_indices:      (N,) int64 camera IDs (e.g. 1=front_wide, 6=front_tele)
#   absolute_timestamps: (N,) int64 timestamps in microseconds
#   ego_history_xyz:     (1, 1, 20, 3) float32 local frame (2.0s @ 10Hz)
#   ego_history_rot:     (1, 1, 20, 3, 3) float32 rotation matrices
inputs = model.prepare_driving_inputs(
    sample,
    processor,
    nav_text="Turn right in 30m",
    use_vision_cache=True,  # rolling per-frame vision feature cache for streaming
)

pred_xyz, pred_rot, _, extra = model.sample_trajectories(
    inputs,
    tokenizer=processor.tokenizer,
    num_traj_sets=1,
    num_traj_samples=1,
    reasoning_mode="adaptive",  # "adaptive" (CoT) or "nominal" (zero-thought fast path)
    temperature=0.6,
    diffusion_steps=10,
    seed=42,
)
print("CoT:", extra["cot"][0, 0, 0])
print("Trajectory (64x3):", pred_xyz[0, 0, 0].shape)
```

### 2. Freeform Driving VQA
```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("Safman/Alpamayo2-Edge-MLX-q4")
prompt = apply_chat_template(processor, model.config, "Describe hazards ahead.", num_images=1)
out = generate(model, processor, prompt, image=["front_camera.jpg"], max_tokens=128)
print(out.text)
```

---

## Architecture & Apple Silicon Optimizations (`mlx_vlm/models/`)

- **Cosmos 3 Edge (`cosmos3_edge/`)**: 27-layer packed SigLIP2 vision tower (float32 antialiased bilinear positional embedding resize + 2×2 `Cosmos3EdgePatchMerger`), 28-layer `ReLU²` GQA language model with 3D interleaved M-RoPE (`rope_theta=1e8`, `mrope_section=[24,20,20]`), and torch-free NumPy/PIL `Cosmos3EdgeProcessor`.
- **Alpamayo 2 Edge (`alpamayo2_edge/`)**: Unified VLM + 512-wide 28-layer non-causal Diffusion Action Expert (`PerWaypointActionInProjV2` + zero-reallocation in-place `KVCache` Euler flow matching) + float64-verified `UnicycleAccelCurvatureCourseActionSpace` (`max_err = 8.53e-14` vs NVIDIA reference).
- **18 GB Unified Memory Safeguards**: Micro-batched vision encoding (`max_frames_per_batch=2`), chunked LLM prefill (`prefill_step_size=256`), single-cache multi-sample reuse (`KVCache.trim`), and capped Metal wired memory limit.
- **Real-Time Acceleration Stack (M3 Pro, 18 GB)**:
  - Rolling per-frame vision feature cache & external **CoreML Apple Neural Engine** vision hook (`0.0 ms` GPU wait with async prefetch).
  - Hybrid `BF16` prefill + `Q4` shadow CoT decode (`enable_hybrid_q4_decode`).
  - **Temporal-DFlash** speculative CoT verification (`84–90%` token acceptance across consecutive driving frames).
  - Zero-sync RoPE **FastV / Pyramid** visual token pruning + 3-step Flow-Matching Expert (`534 ms` / `1.87 Hz` Turbo CoT; `398–447 ms` / `2.24–2.51 Hz` interleaved 5 Hz).

## Benchmarks (Apple M3 Pro, 18 GB — 5-Clip Full-HD `1080×1920` PhysicalAI-AV Nav Suite)

| Configuration | p50 Latency | Rate | Peak Metal RAM | ADE@1s | ADE@3s | Lat@3s | Brake Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| **BF16 Adaptive CoT** (GPU Vision, 10 diff) | `2,577 ms` | `0.39 Hz` | `6.11 GB` | `0.053 m` | **`0.507 m`** | `0.216 m` | `100%` |
| **Hybrid 16/4 Adaptive CoT** (GPU Vision, 10 diff) | `2,226 ms` | `0.45 Hz` | `7.22 GB` | `0.054 m` | `0.517 m` | **`0.215 m`** | `100%` |
| **Q4 Adaptive CoT** (GPU Vision, 10 diff) | `2,268 ms` | `0.44 Hz` | **`3.25 GB`** | `0.055 m` | `0.520 m` | `0.221 m` | `100%` |
| **Lossless Full-`163k` CoT** (Hybrid 16/4 + Temporal-DFlash + 3-step) | `1,408 ms` | `0.71 Hz` | `7.22 GB` | — | **`0.498 m`** | — | `100%` |
| **Turbo Full-`163k` CoT** (Hybrid 16/4 + Async ANE + FastV + DFlash + 3-step) | **`534 ms`** | **`1.87 Hz`** | `6.94 GB` | — | `0.552 m` | `0.24 m` | `91%` |
| **Low-RAM Turbo Full-`163k` CoT** (Q4 + Async ANE + FastV + DFlash + 3-step) | **`578 ms`** | **`1.73 Hz`** | **`3.10 GB`** | — | `0.564 m` | — | `91%` |

## License

Code in `SMJD11/mlx-vla` is licensed under the MIT License (built on [Neywa-Labs/mlx-vlm](https://github.com/Neywa-Labs/mlx-vlm)). NVIDIA Alpamayo model weights remain subject to their respective NVIDIA model licenses.
