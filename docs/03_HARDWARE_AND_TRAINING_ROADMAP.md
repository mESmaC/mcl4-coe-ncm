# 03. Hardware Physics & Staged Training Curriculum (`sm_86` -> Blackwell `sm_120`)

This document details the hardware memory hierarchy, Blackwell `NVFP4` execution path, staged training curriculum, and repository module layout for **`MCL4-COE`**.

---

## 1. Consumer Hardware Tiers & Memory Hierarchy

### Tier A: Rapid Ablation Target (`4 GB` Ampere Laptop GPU, `sm_86`)
* **Specs:** `4 GB` GDDR6 VRAM (native `BF16`), `32 GB` DDR5 RAM.
* **Role:** Fast `1M–15M` parameter architectural ablations, unit testing, data shard preparation, and rapid `MCL4-COE` verification runs.

### Tier B: Primary Single-GPU Training Target (`12 GB` Blackwell `RTX 5070` / Colab Blackwell `sm_120`)
* **Specs:**
  * **GPU:** NVIDIA GeForce RTX 5070 (`12 GB` GDDR7 VRAM, Blackwell `sm_120`, `~123 TFLOPS` BF16, `~246 TFLOPS` FP8, `~988 TOPS / ~1 PFLOP` 2:4 Sparse NVFP4) or Colab `RTX PRO 6000 Blackwell` (`sm_120`).
  * **System RAM:** `32 GB` DDR5 (backing `CPUOffloadAdamW` master weights and Adam states).
  * **Storage:** High-IOPS NVMe SSD storage (`~5–7 GB/s` sequential throughput) for memory-mapped token shards, replay buffers, and `MCL4-JIT` paged expert packs.

| Resource Tier | Physical Budget | What It Bounds | `MCL4-COE` Target Allocation |
| :--- | :--- | :--- | :--- |
| **1. GPU FLOPs / Step** | `~123 TFLOPS` BF16 / `~246 TFLOPS` FP8 | **Active Parameters per Token ($N_{\text{active}}$)** | **`50M – 600M` active parameters** (shared `MCL4-CED` trunk + `top-k` of `8–64` fine-grained experts) |
| **2. DDR5 RAM (`32 GB`)** | `~24 GB` usable for `CPUOffloadAdamW` | **Simultaneously Trainable Parameters ($N_{\text{trainable}}$)** | **`1B – 4B` total sparse parameters** (FP32/BF16 master weights + Adam states in RAM) |
| **3. GDDR7 VRAM (`12 GB`)** | `~8 GB` weights + `4 GB` activations/KV | **Resident Working Weights** | **`1B – 4B` working weights** in BF16 / FP8 / NVFP4 |
| **4. NVMe SSD** | High-IOPS local storage | **External Symbol Memory, Replay Buffers & Cold Experts** | Indexed AST/codebase stores (`A.TXY`), verified RL trajectories, and `MCL4-JIT` paged NVFP4 expert shards |

---

## 2. Unlocking Blackwell Sparse `NVFP4` During Self-Improvement

In `MCL4-COE`'s recursive self-improvement loop, **85% to 95% of wall-clock compute is spent generating candidate operator passes and `T$ -> A$` rollouts**:
1. **NVFP4 Rollout & Ensemble Search:**
   * Quantize working weights into **Blackwell NVFP4 (`E2M1` + `FP8 E4M3` block scales in [`src/mcl4_coe/nvfp4.py`](../src/mcl4_coe/nvfp4.py))** with optional 2:4 structured sparsity.
   * Run multi-pass `COE` ensembles (`Abduction -> Association -> Deduction -> Evaluation -> Metacognition -> Inference`) across candidate trajectories at hardware Tensor Core speed.
2. **Deterministic Verification on CPU:**
   * Verify candidate outputs against deterministic checkers (symbolic math evaluators, rule engines, AST checkers, and sandboxed unit-test harnesses) in parallel on CPU.
   * Store verified winning/contrastive episodes and error-correction traces in the replay buffer and `A.TXY` virtual memory.
3. **Selective BF16/NVFP4 Update via `CPUOffloadAdamW` / Fused `AdamW`:**
   * Run the backward pass and optimizer update strictly on the verified trajectories and active plastic experts.

---

## 3. Staged Training Curriculum for `MCL4-COE`

### Stage 1: Shared Trunk Pretraining (or Donor Grafting)
* **From-Scratch Path:** Deduplicated prose + curated code/math syntax packed into `uint16` streams, trained with causal cross-entropy + MoE `aux_loss` + `router_z_loss` with `MCL4` and `PLE`.
* **Pretrained Donor Path (`Granite 3.1 MoE` / `Llama-3.2-1B`):** Graft pretrained weights into the `MCL4-CED` encoder-decoder split (`donor_backbone_lr_scale = 0.002`) and fit the frozen-donor `DonorAnchoredMultiscaleVQVAE` concept hierarchy.

### Stage 2: Joint Cognitive Operator + `U$ / T$ / A$` Supervised Alignment
* **Data:** Block-aligned (`0x8000` prompt-masked) episodes covering all 9 `COE` operators:
  1. **`U$ -> T$ -> A$` Micro-Traces:** Step-by-step `T$` scratchpads followed by verifiable `A$` outputs.
  2. **`Decision` (`Choice` / `Noul` / `Score`) & `Association` Sets:** State + candidate descriptions encoded through `CED` (where candidates cross-attend to the shared state's lower-layer global `K/V`).
  3. **`Deduction`, `Induction`, `Abduction`, `Evaluation`, & `Metacognition` Tasks:** Rule-entailment triples (`entailed` / `contradicted` / `undetermined`), pattern completion sets, hypothesis ranking under evidence, and calibrated reliability labels (`correctness`, `evidence_sufficiency`, `context_benefit`).
* **Goal:** Train the compact readout projections (`d_proj = 32`) jointly with the shared `MCL4-CED` backbone and multiscale `NCM` latents so every operator sharpens the same representation.

### Stage 3: Multi-Pass Ensemble, Mitotic Expansion & Verifiable RL Self-Improvement
* **Mechanism:** Enable `RecurrentFeedback` (`anchor + sigmoid(gate) * LayerNorm(state)`), `Ornith` GRPO rollouts, `Dream-RSI` counterfactual policy search, and exact-ratio twin-gated MoE mitosis so the model chains operators across passes and expands expert capacity on verifiable tasks.

---

## 4. Repository Module Map

1. **Core Backbone & Sparse Routing ([`src/mcl4_coe/backbone.py`](../src/mcl4_coe/backbone.py), [`markov_local4.py`](../src/mcl4_coe/markov_local4.py), [`optim.py`](../src/mcl4_coe/optim.py), [`data.py`](../src/mcl4_coe/data.py)):**
   * Combines `CPUOffloadAdamW`, `HostBatchPrefetcher`, and `0x8000` block-aligned `TokenBlockDataset` with `MCL4` (`MarkovLocal4Router`), `PerLayerEmbedding` (`PLE`), and the `CED` lower-encoder / upper-decoder split.
2. **`CED`-Contextualized Cognitive Operators ([`src/mcl4_coe/operators.py`](../src/mcl4_coe/operators.py)):**
   * Implements the 9-operator `COE` readout suite (`Decision` [`Choice`, `Noul`, `Score`], `Association`, `Evaluation`, `Deduction`, `Induction`, `Abduction`, `Abstraction`, `Metacognition`, and `Inference`) where candidates cross-attend to the shared `CED` state prefix.
3. **Composable Multi-Pass Controller & Verification Loop ([`src/mcl4_coe/ensemble.py`](../src/mcl4_coe/ensemble.py), [`ornith_rl.py`](../src/mcl4_coe/ornith_rl.py), [`dream_rsi.py`](../src/mcl4_coe/dream_rsi.py)):**
   * Chains `RecurrentFeedback` across multi-operator schedules, verifier-grounded `Ornith` GRPO rollouts, and `Dream-RSI` discovery trees.
4. **Native Blackwell `NVFP4` & Donor-Anchored `NCM` ([`src/mcl4_coe/nvfp4.py`](../src/mcl4_coe/nvfp4.py), [`ncm.py`](../src/mcl4_coe/ncm.py), [`qwen_jit.py`](../src/mcl4_coe/qwen_jit.py)):**
   * Implements fused multi-expert Blackwell `NVFP4` grouped GEMM kernels (`nvfp4_grouped_linear`), `DonorAnchoredMultiscaleVQVAE` + `NCMGeneratorAdapter`, and `MCL4-JIT` working-set expert paging.
