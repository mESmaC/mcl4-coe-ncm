# 08 — Native Blackwell `NVFP4` Fused Multi-Expert Grouped MoE Kernel & Pure RL Handoff

**Date:** 2026-10-21  
**Active Run ID:** `colab_mcl4_coe_v7_granite_e32_pure_rl`  
**Completed Step:** `3000 / 3000` (`status = "completed"`)  
**Hugging Face Backup Repo:** `mesmac/mcl4-coe-runs` (`colab_mcl4_coe_v7_granite_e32_pure_rl/**`)

---

## 1. Summary of Work Completed

### A. Local Inference Platform Setup & Verification (`scripts/chat_web.py`)
1. Downloaded the `2.06B` (`E = 48`) Mitotic RL Soup checkpoint (`step_06000_mitotic_rl_soup.pt`) from `mesmac/mcl4-coe-runs` into `X:\MCL4-COE\runs\colab_mcl4_coe_v7_granite_overnight_mitosis\checkpoints\step_06000.pt`.
2. Fixed two dtype/config bugs when loading `bfloat16` donor checkpoints:
   - [`scripts/chat_web.py`](../scripts/chat_web.py): Preserved `float32` on `nn.LayerNorm` weights/biases after `model.to(dtype=dtype)` so `RecurrentFeedback.norm` (`F.layer_norm`) does not raise `RuntimeError: expected scalar type Float but found BFloat16` when given `float32` recurrent states.
   - [`src/mcl4_coe/trainer.py`](../src/mcl4_coe/trainer.py): Applied the same `nn.LayerNorm` `float32` preservation during `train_dtype="bfloat16"` initialization and added fallback to `payload.get("config")` when `.meta.json` lacks an inline `"config"` block.

---

### B. Fused Multi-Expert Blackwell `NVFP4` Grouped Kernel (`nvfp4_grouped_linear`)
When `use_nvfp4 = True` was initially enabled on the `1.45B` (`E = 32`) Granite 3.1 MoE checkpoint (`1,536` `NVFP4Linear` expert projections), training throughput dropped from `~1,350 tok/s` (`BF16` `torch._grouped_mm`) to `~360 tok/s` because PyTorch's `F.scaled_mm` only supports single 2D matrices (`[M, K] @ [K, N]`), forcing `SparseMoE.forward` into a 32-iteration Python loop with `72` blocking `.tolist()` CPU-GPU syncs and `30,754` tiny kernel launches per step.

To unlock the `NVIDIA RTX PRO 6000 Blackwell` (`sm_120`) `NVFP4` Tensor Cores on fine-grained MoE layers, we built a custom **Fused Multi-Expert Blackwell `NVFP4` Grouped Kernel Suite** in [`src/mcl4_coe/nvfp4.py`](../src/mcl4_coe/nvfp4.py) and wired it into all three dispatch paths in [`SparseMoE`](../src/mcl4_coe/backbone.py):

1. **`_nvfp4_quantize_1d_kernel` (`0.010 ms` for `4096 × 1024`):**
   - Fused 2D-grid Triton kernel (`BLOCK_M = 32, BLOCK_K = 128`) that quantizes routed token activations (`[M_total, K]`) directly into packed `uint8` `E2M1` nibbles (`c0 | (c1 << 4)`) and contiguous `float8_e4m3fn` (`tl.float8e4nv`) `1×16` block scales with zero swizzle padding.
   - Supports deterministic causal row-wise scaling (`global_scale = 1.0`) in `forward` and tensor-scaled stochastic rounding (`STOCHASTIC = True`) in `backward` (`dgrad`).
2. **`_nvfp4_grouped_mm_kernel` (`0.026 ms` / `325.1 TFLOPS` across all `32` experts):**
   - Maps thread blocks to variable-length expert segments using the GPU-resident cumulative offset tensor `offs` (`int32 [E]`) in registers — **zero `.tolist()` CPU-GPU synchronizations** and **zero 128-row padding waste** on small expert chunks.
   - Executes `tl.dot_scaled(a, a_scale, "e2m1", b, b_scale, "e2m1", acc=acc, rhs_k_pack=True)`, which Triton `3.6.0` + CUDA `13.0` compiles directly to native `sm_120` Blackwell Tensor Core PTX:
     ```ptx
     mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
     ```
3. **`_grouped_wgrad_kernel` (`0.109 ms` across `32` experts):**
   - Computes `dW[e] = Grad[start_m:end_m].T @ X[start_m:end_m]` across all `E` experts in a single kernel launch while reading `MaskReq_ptr` (`requires_grad` mask) so frozen donor experts exit in 1 instruction and empty experts (`start_m >= end_m`) skip memory stores.
4. **Version-Keyed 3D `NVFP4` Expert Weight Cache (`_get_or_build_grouped_nvfp4_cache`):**
   - Caches persistent 3D `NVFP4` weight stacks (`B_fwd [E, K//2, N]`, `Bs_fwd [E, N, K//16]`, `Bg [E]`, `B_bwd [E, N//2, K]`, `Bs_bwd [E, K, N//16]`) on `experts`.
   - Checks `weight._version` per expert so frozen donor experts are quantized once at startup and the `8` plastic experts per layer are re-quantized only once per `optimizer.step()` (`0.176 ms` batched via `@torch.compile`), serving cached `NVFP4` weights in `< 1 µs` across micro-batches, recurrence depth steps, all 9 COE operators (`forward_all_operators`), Ornith GRPO rollouts, and Dream-RSI cycles.
5. **Full `SparseMoE` Integration ([`src/mcl4_coe/backbone.py`](../src/mcl4_coe/backbone.py)):**
   - Wired `_run_grouped_experts` into:
     1. Main top-$k$ routed `SparseMoE.forward` path
     2. `combine_dense_and_sparse` residual top-$k$ path (used by all 9 COE operators)
     3. `forward_dense_subset` coalition path (used by all 9 COE operators)

#### Benchmark Results on Colab `g4` (`NVIDIA RTX PRO 6000 Blackwell`, `sm_120`)

| Metric | Un-Grouped `NVFP4` (`F.scaled_mm` Loop) | Fused Multi-Expert `NVFP4` (`nvfp4_grouped_linear`) | Speedup |
| :--- | :---: | :---: | :---: |
| **`SparseMoE` Layer (`fwd + bwd`, `E=32, k=8, 8×128`)** | `37.279 ms` | **`1.820 ms`** (`0.851 ms` fwd / `0.969 ms` bwd) | **`20.5×` faster** |
| **End-to-End RL + 9-Op COE + Ornith + Dream `tok/s`** | `290 – 369 tok/s` | **`1,340 – 1,601 tok/s`** | **`>4.3×` faster** (exceeds `BF16`!) |

---

### C. Pure RL + Dream-RSI Progression (`step 400 -> 3000`, `enable_mitosis = False`)
Starting from the unsplit `1.45B` (`E = 32`) Granite 3.1 MoE checkpoint (`colab_mcl4_coe_v7_granite_slingshot_r2/checkpoints/step_00400.pt`), we ran [`scripts/launch_v7_granite_rl.py`](../scripts/launch_v7_granite_rl.py) (`enable_mitosis = False`, `plastic_experts_per_layer = 8`, `ornith_every = 1`, `dream_every = 1`, `operator_every = 1`, `batch_size = 8`, `use_nvfp4 = True`, `nvfp4_sparsity_mode = "none"`):

| Step | Mode | Held-Out `val_bpb` ($\downarrow$) | Held-Out `val_mcqa` ($\uparrow$) | `NVFP4` GEMMs (Session) | Throughput (`tok/s`) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| `400` | Base Slingshot R2 (`BF16`) | `0.8123` | `0.2891` | `0` | — |
| `600` | Pure RL (`BF16`, pre-PTQ) | `0.7334` | `0.3359` | `0` | `~1,350 tok/s` |
| `601` | Cold PTQ into `NVFP4` (`1,536` layers) | `0.7662` | `0.3047` | `87,750` | `333 tok/s` (un-grouped) |
| `700` | `NVFP4` Pure RL (un-grouped) | `0.7629` | `0.3359` | `4,085,000` | `331 tok/s` (un-grouped) |
| `701` | Switched to Fused `nvfp4_grouped_linear` | `0.7621` | `0.3203` | `88,704` | **`1,434 – 1,553 tok/s`** |
| `1400` | Fused `NVFP4` Pure RL + Dream-RSI | `0.7495` (`0.7494` best) | `0.3516` peak | `27,042,048` | **`1,255 – 1,601 tok/s`** |
| `3000` | Completed Checkpoint (`step_03000.pt`) | **`0.7391`** | **`0.4141`** (`0.4375` peak) | **`61,927,680`** (`>93M` total) | **`1,273 – 1,601 tok/s`** |

---

## 2. Backed-Up Checkpoints & Artifacts

### On Hugging Face (`mesmac/mcl4-coe-runs`)
All checkpoints and logs are backed up under `colab_mcl4_coe_v7_granite_e32_pure_rl/`:
- `colab_mcl4_coe_v7_granite_e32_pure_rl/checkpoints/step_03000.pt` (`1.45B`, `E = 32`, `val_bpb = 0.7391`, `val_mcqa = 0.4141 / 0.4375 peak`, calibrated for `NVFP4`)
- `colab_mcl4_coe_v7_granite_e32_pure_rl/checkpoints/step_03000.meta.json`
- `colab_mcl4_coe_v7_granite_e32_pure_rl/checkpoints/step_02900.pt` & `step_02800.pt`
- `colab_mcl4_coe_v7_granite_e32_pure_rl/dream/dream_state.json` (`2,945` Dream-RSI cycles completed)
- `colab_mcl4_coe_v7_granite_e32_pure_rl/run_manifest.json`, `metrics.jsonl`, `samples.jsonl`, `train.log`
- Code bundle: `_bundles/mcl4_coe_v8_bundle.tar.gz`

### On Local Disk (`X:\MCL4-COE`)
- `runs/colab_mcl4_coe_v7_granite_e32_pure_rl/`: `run_manifest.json`, `metrics.jsonl`, `samples.jsonl`, `train.log`, `dream/dream_state.json`, and `.meta.json` checkpoints through `step_03000.meta.json`.
- `runs/colab_mcl4_coe_v7_granite_overnight_mitosis/checkpoints/step_06000.pt`: `2.06B` (`E = 48`) Mitotic RL Soup checkpoint ready for local testing in `scripts/chat_web.py`.
