# 07 — Proper Donor-Anchored RL/RSI Mitotic Expansion Protocol

> [!IMPORTANT]
> **Core Principle:** The entire reason to start from a multi-trillion-token donor MoE (`ibm-granite/granite-3.1-*`) is to **preserve 100% of its pretrained trunk and root-expert knowledge (`donor_backbone_lr_scale = 0.002` at `lr = 2e-5` $\implies$ `4e-8` donor LR)** while using **every-step RL + Dream-RSI (`ornith_every = 1`, `dream_every = 1`)** to drive gradual mitotic expert splitting and specialization.
>
> **Never use mitosis as bulk SFT capacity expansion on a high learning rate.**

---

## 1. Post-Mortem Comparison: Why Granite Succeeded vs. Why `v8-OLMoE-12B` Failed

| Axis | **`v7_granite_overnight_mitosis`** *(Correct Process — Beat Base Model)* | **`v8_olmoe_12b`** *(Incorrect Process — Degraded / Flat vs. Base)* | Root-Cause Lesson |
| :--- | :--- | :--- | :--- |
| **Phase 0: Step-0 Donor Graft** | `GraniteMoeForCausalLM` (`norm_topk_prob = True`, `embedding_multiplier = 12.0`, `residual_multiplier = 0.22`, `attention_multiplier = 0.015625`, `logits_scaling = 6.0`) | `OlmoeForCausalLM` (`norm_topk_prob = False`, un-normalized top-$k$ weights) | `norm_topk_prob = True` guarantees $\sum_{e \in \text{top-}k} w_e = 1$ across any expert count $E$, avoiding partition-function shrinkage when twins are released. |
| **Phase 1: Pre-Mitosis Slingshot (`enable_mitosis = False`)** | **`400` steps at fixed `E = 32`** (`ornith_every = 1`, `dream_every = 2`, `donor_backbone_lr_scale = 0.04`) to align the 9 COE heads and router before any mitosis | **Skipped.** Triggered mitosis (`64 -> 80 -> 96 -> 112`) directly inside a `2,800`-step SFT loop before COE/RL heads were calibrated | Always align COE heads and zero-init adapters at fixed donor $E_0$ before splitting a single expert. |
| **Phase 2: What Drives Mitosis** | **Pure RL/RSI loop:** `ornith_every = 1`, `dream_every = 1` across `5,600+` steps (`5,802` Dream-RSI cycles, `5.5M` counterfactual replays, `5,379` verified scaffolds) | **SFT curriculum loop:** `ornith_every = 8`, `dream_every = 8` (`2,800` steps, only `~239` Dream cycles total) | Mitosis must be driven by verifiable RL (`Ornith`) + `Dream-RSI` counterfactual rollouts on every step, not passive SFT cross-entropy. |
| **Donor Trunk & Root Expert Protection** | `lr = 2.0e-5`, `donor_backbone_lr_scale = 0.002` $\implies$ **donor LR = `4.0e-8`** (`99.8%` anchored to multi-trillion-token pretraining) | `lr = 1.5e-4`, `donor_backbone_lr_scale = 0.10–0.15` $\implies$ **donor LR = `1.5e-5`** (**`375x` higher**!) | High donor/adapter LR on a small SFT corpus (`~39M` tokens) overwrites pretrained routing and overfits `PLE` / `global_out_proj` to `#U$#` tags (`-7.60%` on natural prompts). |
| **Prefix Gist Conditioning (`enable_gist_conditioning`)** | **`False`** (clean token context at all times) | **`True` (`0.35–0.45`)** (replaced prefixes with compressed gist vectors on 35–45% of batches) | Keep `enable_gist_conditioning = False` so `decoder.global_out_proj` never distorts standard causal/cross-attention. |
| **Mitosis Granularity & Lineage Tracking** | Gradual **`+4` experts per mitosis event** (`mitosis_every = 200`), splitting distinct root experts with cumulative `mitotic_pairs` tracking | Bulk **`+16` experts per wave** (`64 -> 80 -> 96 -> 112`), overwriting `mitotic_pairs` each wave and re-splitting children into size-4 families | Split in small increments (`+4` or `+8`), accumulate `mitotic_pairs` across waves, and prioritize splitting unsplit root experts before 2nd-generation splits. |

---

## 2. The 6 Golden Rules of Proper Donor RL/RSI Expansion

1. **Rule 1 — Anchor the Pretrained Donor (`donor_lr <= 1e-7`):**
   - Set `lr = 2.0e-5` and `donor_backbone_lr_scale = 0.002` (`donor_lr = 4.0e-8`), with `freeze_donor_embeddings = True`.
   - The donor's attention projections, RMSNorms, embeddings, `lm_head`, and original root experts (`[0 .. E_0 - 1]`) must remain anchored to their multi-trillion-token pretrained manifold.
2. **Rule 2 — Keep Context Clean (`enable_gist_conditioning = False`):**
   - Never replace standard token prefixes with synthetic gist tokens during donor RL/RSI expansion.
   - Keep `PLE` and `decoder.global_out_proj` zero-initialized and regularized so natural-language prompts (`Question: ... Answer:`) suffer zero distribution shift.
3. **Rule 3 — Phase 1 Slingshot Alignment (`enable_mitosis = False` First):**
   - Before splitting any experts, run `300–400` steps at the native donor expert count $E_0$ with `enable_mitosis = False`, `ornith_every = 1`, `dream_every = 1..2` to warm up the 9 `COE` operator heads (`Decision`, `Deduction`, `Evaluation`, `Metacognition`, `Abstraction`) and populate the initial `ScaffoldCache` and `DreamState`.
4. **Rule 4 — Mitosis Inside 100% RL/RSI (`ornith_every = 1`, `dream_every = 1`):**
   - Mitosis only happens during active RL/RSI (`ornith_every = 1`, `dream_every = 1`), where `Ornith` verified rollouts and `Dream-RSI` counterfactual revisions create genuine functional divergence between parent and child experts (`orthogonal_mitosis = True`, chordal orthogonality loss on `mitotic_pairs`).
5. **Rule 5 — Cumulative `mitotic_pairs` & Breadth-First Root Splitting (`mitosis_every` + `mitosis_step_size`, NOT `mitosis_milestones`):**
   - In [`src/mcl4_coe/trainer.py`](../src/mcl4_coe/trainer.py#L1476), setting `mitosis_milestones` triggers `pairs_for_schedule = [] if milestones else existing_pairs`, which **wiped out `existing_pairs`** on each wave in `v8-OLMoE-12B` and allowed already-split experts to re-split into size-4 families while 24 root experts were never split.
   - Always use `mitosis_every = 200` + `mitosis_step_size = 4` (or `8`) with `mitosis_milestones = ()` (exactly as Granite did), or preserve `existing_pairs` across waves so `compute_mitotic_schedule` splits all unsplit root experts ($r \in [0 .. E_0 - 1]$) before any second-generation splits.
6. **Rule 6 — Benchmark Gate After Every Expansion Stage:**
   - Validate on held-out natural-language **ARC-Challenge**, **ARC-Easy**, and **MMLU** probes alongside internal `COE` / `Ornith` metrics to verify that every mitotic expansion strictly improves or preserves general reasoning over the base Granite donor.

---

## 3. Target Architecture Options for Scaling Granite to `~13B` (For Review Prior to Any Launch)

Both IBM Granite 3.1 MoE models use `GraniteMoeForCausalLM` (`vocab_size = 49,155`, `norm_topk_prob = True`, `top_k = 8`, `attention_multiplier = 0.015625`, `embedding_multiplier = 12.0`, `residual_multiplier = 0.22`, `logits_scaling = 6.0`) and share our existing `data_shards_granite49k` tokenizer:

| Option | Base Donor Model | Layers (`Enc + Dec`) | `d_model` / `d_ffn` | Start Experts ($E_0$) & Params | Params Added per Expert (Full Stack) | Target `~13B` Expert Count ($E_{\text{target}}$) | Active Params at `top_k = 8` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Option A (Recommended)** | **`ibm-granite/granite-3.1-3b-a800m-instruct`** | `32` (`16 + 16`) | `1536` / `512` | `E = 40` (`3.56B` total) | **`75.50M` / expert** | **`E = 160` (`12.62B`)** to **`E = 168` (`13.22B`)** (`4.0x–4.2x` base experts) | **`~890M`** |
| **Option B** | **`ibm-granite/granite-3.1-1b-a400m-instruct`** *(or resume `v7_granite_overnight_mitosis` @ `E=48, 2.06B`)* | `24` (`12 + 12`) | `1024` / `512` | `E = 32` (`1.45B` total) / `E = 48` (`2.06B`) | **`37.75M` / expert** | **`E = 336` (`12.93B`)** (`10.5x` base experts) | **`~470M`** |

---

## 4. Native Blackwell (`sm_120`, Colab `g4`) `NVFP4` + `2:4` Sparsity Protocol (`4 PFLOPS`)

Colab `g4` instances (`NVIDIA RTX PRO 6000 Blackwell Server Edition`) provide **`2 PFLOPS` dense `NVFP4`** and **`4 PFLOPS` `2:4` structured-sparse `NVFP4`** Tensor Core throughput (`torch.float4_e2m1fn_x2` with `1x16` `FP8 E4M3` block scales via `F.scaled_mm` in [`src/mcl4_coe/nvfp4.py`](../src/mcl4_coe/nvfp4.py)).

To unlock that `4 PFLOPS` compute and `~3.55x` memory compression at `13B+` scale **without** shocking the donor base:

1. **NVFP4 Donor Base (Pre-Quantized or In-House Block-Wise `E2M1` Quantization):**
   - Start from an `NVFP4`-quantized Granite donor checkpoint or quantize the donor directly via [`quantize_nvfp4(..., two_dimensional=True)`](../src/mcl4_coe/nvfp4.py#L204-L251) (`1x16` `FP8 E4M3` block scales + `FP32` global scale).
   - At `0.5625 bytes/param` (`0.5B` packed `E2M1` + `0.0625B` swizzled `FP8 E4M3` scale), a **`13B` parameter MoE occupies only `~7.3 GB` of VRAM** (vs. `26 GB` in `BF16` and `52 GB` in `FP32`).
   - Because the root donor experts are anchored (`donor_backbone_lr_scale = 0.002`) or frozen in older mitotic generations (`plastic_experts_per_layer`), their pre-quantized `(packed_fp4, swizzled_fp8_scales, global_scale)` buffers remain permanently resident in VRAM with **zero per-step re-quantization overhead**.
2. **Complementary `2:4` Sparsity via Mitosis (`1 Dense Parent -> 2 Sparse Twins = 100% Capacity Preserved`):**
   - Applying `2:4` structured sparsity cold to a single dense donor expert drops 50% of its weights.
   - Instead, when a parent expert splits via **mitosis (`E_parent -> E_parent + E_child`)**, the two twin `NVFP4Linear` experts together hold **`2x` the parameters (`2 * 50% = 100%` non-zero capacity)** and refine their `2to4_transposable` masks during RL/RSI with `SR-STE` (`sr_ste_decay = 1e-4`), unlocking the full **`4 PFLOPS` `2:4` sparse `NVFP4`** hardware path with zero information loss at birth.
3. **Batched / Compiled `NVFP4` Expert Dispatch:**
   - Replace per-expert Python loops with `torch.compile`-fused `quantize_nvfp4` + concatenated wide-subset `F.scaled_mm` dispatch so routed expert activations feed large contiguous matrices directly into the Blackwell `NVFP4` Tensor Cores.

