# 03. Hardware Physics & Staged Training Roadmap (`RTX 2050` -> `RTX 5070`)

This document maps out the exact hardware limits, memory hierarchy design, and 4-milestone implementation roadmap for **`MCL4-COE`**.

---

## 1. Target Hardware Hierarchy

### Portable Validation Machine (`RTX 2050` Laptop)
* **Specs:** `4 GB` GDDR6 VRAM (Ampere `sm_86`, native BF16), `32 GB` DDR5 RAM.
* **Role:** Fast `1M–15M` parameter architectural ablations, unit testing, data preparation, and short `MCL4-COE` training runs (`3–15` minutes per run).

### Primary Flagship Workstation (`RTX 5070` Desktop)
* **Specs:**
  * **GPU:** NVIDIA GeForce RTX 5070 (`12 GB` GDDR7 VRAM, Blackwell `sm_120`, `~123 TFLOPS` BF16, `~246 TFLOPS` FP8, `~988 TOPS / ~1 PFLOP` 2:4 Sparse NVFP4).
  * **System RAM:** `32 GB` DDR5 (backing `CPUOffloadAdamW` master weights and Adam states).
  * **Storage:** `10 TB` NVMe SSD (`~5–7 GB/s` sequential throughput) + `2 TB` disk.

| Resource Tier | Physical Budget | What It Bounds | `MCL4-COE` Target Allocation |
| :--- | :--- | :--- | :--- |
| **1. GPU FLOPs / Step** | `~123 TFLOPS` BF16 / `~246 TFLOPS` FP8 | **Active Parameters per Token ($N_{\text{active}}$)** | **`50M – 300M` active parameters** (shared `MCL4-CED` trunk + `top-2` of `16–64` fine-grained experts) |
| **2. DDR5 RAM (`32 GB`)** | `~24 GB` usable for `CPUOffloadAdamW` | **Simultaneously Trainable Parameters ($N_{\text{trainable}}$)** | **`1B – 4B` total sparse parameters** (FP32/BF16 master weights + Adam states in RAM) |
| **3. GDDR7 VRAM (`12 GB`)** | `~8 GB` weights + `4 GB` activations/KV | **Resident Working Weights** | **`1B – 4B` working weights** in BF16 / FP8 / NVFP4 |
| **4. NVMe SSD (`10 TB`)** | `10,000 GB` high-IOPS storage | **External Symbol Memory, Replay Buffers & Cold Experts** | Indexed AST/codebase stores (`CDAS`), `100M+` verified RL/MCMC trajectories, and swappable domain expert packs |

---

## 2. Unlocking the RTX 5070's `~1 PFLOP` Sparse NVFP4 During Self-Improvement

In `MCL4-COE`'s recursive self-improvement loop, **85% to 95% of wall-clock compute is spent generating candidate operator passes and `T$ -> A$` rollouts**:
1. **NVFP4 Rollout & Ensemble Search:**
   * Before each rollout/search wave, quantize the current working weights into **Blackwell NVFP4 (`E2M1` + `FP8 E4M3` block scales, adapted from `oracle/nvfp4.py`)** with optional 2:4 structured sparsity.
   * Run multi-pass `COE` ensembles (`Abduction -> Association -> Deduction -> Evaluation -> Metacognition -> Inference`) across 16–64 candidates per problem at maximum Tensor Core speed.
2. **Deterministic Verification on CPU + 10TB NVMe:**
   * Verify candidate outputs against deterministic checkers (symbolic math evaluators, rule engines, `pytest`, `rustc`, AST checkers) in parallel on CPU.
   * Store verified winning/contrastive episodes and failed error-correction traces in the **10TB NVMe** replay buffer.
3. **Selective BF16/FP8 Update via `CPUOffloadAdamW`:**
   * Run the backward pass and `CPUOffloadAdamW` update strictly on the verified trajectories and active routed experts.

---

## 3. Staged Training Curriculum for `MCL4-COE`

### Stage 1: Shared Trunk Pretraining (`Inference` on Clean Prose + Code/Math Primitives)
* **Data:** Deduplicated, unwrapped paragraphs (`prose_clean`) + curated code/math syntax (`CDAS`), packed into `uint16` streams.
* **Objective:** Standard causal byte/token cross-entropy + MoE `aux_loss` + `router_z_loss` with `MCL4` (`4` local anchors + `12` Markov walkers) and `PLE`.
* **Goal:** Establish strong lexical, syntactic, and multi-hop graph routing representations in the shared `MCL4-CED` backbone.

### Stage 2: Joint Cognitive Operator + `U$ / T$ / A$` Supervised Warming
* **Data:** Block-aligned (`0x8000` prompt-masked) episodes covering all 9 `COE` operators:
  1. **`U$ -> T$ -> A$` Micro-Traces:** 100% step-by-step `T$` scratchpads (zero blind `U$ -> A$` guessing).
  2. **`Decision` (`Choice` / `Noul` / `Score`) & `Association` Sets:** State + candidate descriptions encoded through `CED` (where candidates cross-attend to the shared state's lower-layer global `K/V`).
  3. **`Deduction`, `Induction`, `Abduction`, `Evaluation`, & `Metacognition` Tasks:** Rule-entailment triples (`entailed` / `contradicted` / `undetermined`), pattern completion sets, hypothesis ranking under evidence, and calibrated reliability labels (`correctness`, `evidence_sufficiency`, `context_benefit`).
* **Goal:** Train the compact readout projections (`d_proj = 32`) jointly with the shared `MCL4-CED` backbone so every operator sharpens the same latent representation.

### Stage 3: Multi-Pass Ensemble & Verifiable RL Self-Improvement
* **Mechanism:** Enable `RecurrentFeedback` (`anchor + sigmoid(gate) * LayerNorm(state)`) so the model chains operators (`Abduction -> Association -> Deduction -> Metacognition -> Inference`) across multiple passes per prompt, rewarded by exact verifier outcomes and calibrated Brier/cross-entropy scores on `Metacognition`.

---

## 4. Implementation Milestones in This Repository

1. **Milestone 1 — Clean-Slate Core (`src/mcl4_coe/backbone.py`, `markov_local4.py`, `optim.py`, `data.py`):**
   * Merge `mini`'s high-throughput trainer (`CPUOffloadAdamW`, `HostBatchPrefetcher`, `0x8000` block-aligned `TokenBlockDataset`) with `mcattn`/`oracle`'s `MCL4` (`markov_local_tokens=4`), `PLE`, and `CED` lower-encoder / upper-decoder split.
2. **Milestone 2 — `CED`-Contextualized Cognitive Operators (`src/mcl4_coe/operators.py`):**
   * Implement the unified `COE` readout head supporting `Decision` (`Choice`, `Noul`, `Score`), `Association`, `Evaluation`, `Deduction`, `Induction`, `Abduction`, `Abstraction`, and `Metacognition` where candidates cross-attend to the `CED` state prefix—And prove on the `mcattn.decision` benchmark that `CED` cross-attention breaks past the `44.4%` bi-encoder ceiling to `>95%` accuracy in `<60` seconds on the RTX 2050.
3. **Milestone 3 — Composable Multi-Pass Controller & `U$ / T$ / A$` Verification Loop (`src/mcl4_coe/ensemble.py`):**
   * Wire `RecurrentFeedback` so a single prompt can execute any sequence of `COE` operators before emitting `T$ -> A$`, and benchmark multi-pass `MCL4-COE` against single-pass standard attention at matched parameter count (`1M–10M` on RTX 2050).
4. **Milestone 4 — RTX 5070 Scale-Up + Native `NVFP4` (`src/mcl4_coe/nvfp4.py`):**
   * Port `oracle/nvfp4.py` into the rollout/training path and scale to the flagship `RTX 5070` configuration for **Research Release `001`** on `yycore.ai`.
