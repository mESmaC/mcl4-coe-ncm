# 01. `MCL4-COE` Unified Architecture Specification

This document specifies how **`MCL4-COE`** combines the proven mechanisms from `mini`, `mcattn`, and `oracle` into a single, cohesive from-scratch architecture.

---

## 1. Input Representation & Block-Aligned Protocol

### 1.1 Compact Turn & Role Framing (`U$ / T$ / A$` + Typed Items)
Instead of wasting 100+ byte tokens per turn on verbose ChatML headers (`<|im_start|>system...`), `MCL4-COE` uses ultra-compact 2-byte structural delimiters and typed item segments:
* **`U$`**: User / task / environment state input.
* **`T$`**: Explicit intermediate reasoning trace, equation step, or operator scratchpad.
* **`A$`**: Final verifiable answer, code patch, or structured output.
* **Typed Set Items (for `COE` operators):** When feeding structured inputs (observations, premises, candidate options, criteria, hypotheses), each item carries:
  * `role`: `observation` | `premise` | `candidate` | `hypothesis` | `criterion` | `goal` | `evidence`
  * `content`: UTF-8 bytes (or continuous patch projections).
  * **Order Invariance vs. Chronology:** Unordered candidate/observation sets do not embed arbitrary option indices (`0, 1, 2`), guaranteeing permutation equivariance; chronological sequences retain causal positional encoding.

### 1.2 Zero-Truncation Block Alignment & `0x8000` Prompt Masking
Inherited from [`mini/src/mini_moe/data.py`](https://github.com/mESmaC/mini):
* Every training example or cognitive episode is packed into an exact block boundary (`seq_len = 256` or `512` byte/subword tokens) padded with `pad_id = 256` so **no trace or decision set is ever severed mid-thought across blocks**.
* Conditioning tokens (`<bos>U$ ...\n` or shared `state` prefixes) carry the high bit `0x8000` in the `uint16` memory-mapped stream (`train.bin` / `val.bin`).
* The batch loader strips `0x8000` for `input_ids` (`raw & 0x7FFF`) and masks both `0x8000`-tagged prompt tokens and `pad_id` tokens to `-100` in `labels`, focusing 100% of gradient updates on the cognitive operators (`T$`, `A$`, and `COE` readouts).

---

## 2. The Shared Backbone: `MCL4` + `CED` + `PLE` + Sparse MoE

### 2.1 Per-Layer Token Embeddings (`PLE`)
Adapted from `mini` (`PerLayerTokenEmbedding`) and `oracle/ple.py`:
* Each transformer layer $l$ owns a narrow token lookup table $E^{(l)} \in \mathbb{R}^{V \times d_{\mathrm{ple}}}$ (`d_ple = 32` or `64`), a projection to `d_model`, an `FP32` RMSNorm, and a learned scalar gate.
* **Why it matters:** On byte or compact vocabularies, `PLE` injects fresh token identity directly into every layer's residual stream, adding parameter capacity with $O(1)$ lookup cost per token rather than widening the $O(d_{\mathrm{model}} \times d_{\mathrm{ffn}})$ expert matrices.

### 2.2 `MCL4`: Bounded Markov Chain Attention with 4 Mandatory Local Anchors
Adapted from `mcattn` (`tinystories_1m_markov_local4`) and `oracle/markov_local4.py`:
* **The Layer-0 Byte Starvation Discovery:** Frozen-checkpoint diagnostics in `mcattn` proved that standard attention places **83.77%** of its first-layer weight on `{t, t-1, t-2, t-3}` to assemble local bytes into words/numbers, whereas pure semantic Markov routing placed only **3.06%** there.
* **The `MCL4` Selection Policy:**
  1. **Bounded-Degree Causal Token Graph:** Each node $i$ has a self-edge, local edges at offsets $\lbrace \pm 1, \pm 2, \pm 4 \rbrace$ (clamped causally to $\le q$), and up to 4 backward semantic edges to earlier tokens with highest routing-key cosine similarity.
  2. **Query-Conditioned Multi-Hop Walk:** For query $q$, walkers start at the `top-W` (`W = 4`) seed nodes and take $S$ (`S = 3`) transitions using:

$$
P_q(i \to j) = \mathrm{softmax}_{j \in \mathcal{N}_q(i)} \left( \frac{\tanh(q_{\mathrm{route}}[q] + s_{\mathrm{route}}[i]) \cdot k_{\mathrm{route}}[j]}{\sqrt{r}} \right)
$$

  Bounded beam propagation (`beam_size = 8`) merges duplicate destinations differentiably (`_coalesce`) and tracks visitation mass.
  3. **Guaranteed `Local-4` Readout:** The hard attention subset $\mathcal{S}_q$ (`|S_q| <= 16`) always reserves the **4 local anchors** $\lbrace q, q-1, q-2, q-3 \rbrace$ plus up to **12 highest-visitation Markov-walked positions**.
  4. **Ordinary QKV Attention on $\mathcal{S}_q$ + Straight-Through Router Surrogate:** Forward values use exact scaled dot-product attention restricted to $\mathcal{S}_q$ (eliminating softmax dilution from the remaining context), while the backward pass supplies routing gradients via the visitation mass surrogate.

### 2.3 Causal Encoder-Decoder (`CED`) — Solving the Bi-Encoder Decision Bottleneck
Adapted from `oracle/causal.py` and `mcattn/association.py`:
* **What went wrong in `mcattn.decision`:** In `mcattn`, `context` (`state + question`) and candidate `options` (`"dog"`, `"rabbit"`, `"2"`) were encoded in completely separate `backbone.encode()` passes before pooling. Because options never attended to the state tokens inside attention, pooled 1-word options collapsed to generic byte averages (`44.4%` majority accuracy).
* **How `CED` Fixes It in `MCL4-COE`:**
  1. **Lower Causal Encoder Layers ($L_{\mathrm{enc}}$):** Process the shared conditioning prefix (`state`, `premises`, `instructions`) **once**, producing cached global keys and values $(K_{\mathrm{global}}, V_{\mathrm{global}})$.
  2. **Upper Decoder / Operator Layers ($L_{\mathrm{dec}}$):** Each candidate option, hypothesis, or `T$ / A$` continuation runs through the upper layers while **cross-attending directly to $(K_{\mathrm{global}}, V_{\mathrm{global}})$ from the state** alongside its own `MCL4` local/routed states!
  3. **Result:** Candidate $m$ directly compares its tokens against the exact evidence in `state` before pooling into its final feature vector, while still encoding the shared `state` only once per batch.

### 2.4 Fine-Grained Dropless MoE + `CPUOffloadAdamW` + Native `NVFP4`
* **Fine-Grained SwiGLU Experts:** Each layer routes tokens to `top-k` SwiGLU experts (plus an optional always-on shared expert), trained with FP32 load-balancing `aux_loss` and `router_z_loss`, and managed by `mini`'s split/prune/merge lifecycle (`expert_lifecycle.py`).
* **`CPUOffloadAdamW` (`mini/optim.py`):** Keeps working weights in `BF16` / `NVFP4` on GPU while storing FP32 master weights and Adam (`m, v`) states in **32 GB DDR5 system RAM**, transferring updates via pinned async host buffers.
* **Native Windows `NVFP4` (`oracle/nvfp4.py`):** Uses `cuBLASLt` packed `E2M1` operands with `FP8 E4M3` $16 \times 16$ block scales, Hadamard transforms, and stochastic rounding on Blackwell (`RTX 5070`).

---

## 3. The Cognitive Operator Ensemble (`COE`)

Instead of forcing all reasoning through autoregressive text generation, `MCL4-COE` exposes **9 composable operators** on top of the shared `MCL4-CED` backbone. Operators share backbone representations and can be executed **multiple times and in any order per prompt** via `RecurrentFeedback` (`anchor + sigmoid(gate) * LayerNorm(state)`).

### 3.1 `Association` (Relational Graph Operator)
* **Purpose:** Measure how typed inputs relate (`supports`, `contradicts`, `precedes`, `similar`) without forcing a winner-take-all choice.
* **Mechanism:** Maps `CED`-contextualized item states $h_i, h_j$ through learned projections $P_{\mathrm{left}}, P_{\mathrm{right}} \in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{proj}}}$ (`d_proj = 32`) and computes FP32 centered Pearson similarity:

$$
\tilde{u} = u - \bar{u}, \qquad \mathrm{sim}_{\mathrm{Pearson}}(u, v) = \frac{\tilde{u} \cdot \tilde{v}}{\lVert \tilde{u} \rVert_2 \lVert \tilde{v} \rVert_2 + \epsilon} \in [-1, 1]
$$

* **Output:** Pairwise or query-to-set relation scores $S_{ij}$ and independent calibrated link probabilities $\sigma(\alpha S_{ij} + \beta)$ (allowing multiple simultaneous links without mutual exclusion).

### 3.2 `Decision` (`Choice` / `Noul` / `Score` Single-Softmax Operator)
* **Purpose:** Permutation-invariant selection or expected-value estimation over a candidate set $\mathcal{C} = \lbrace c_1, \dots, c_M \rbrace$.
* **Mechanism:** Each candidate $c_m$ is contextualized against the shared `CED` state prefix, pooled over valid tokens, and scored via centered Pearson/cosine similarity $s_m = \mathrm{sim}(z_{\mathrm{state}}, z_{c_m})$ with learned log-clamped scale $\tau = \exp(\mathrm{clamp}(\log \tau, \ln 0.1, \ln 100))$:

$$
p_m = \frac{\exp(\tau s_m)}{\sum_{k=1}^M \exp(\tau s_k)}
$$

* **Sub-Modes:**
  * **`Choice`:** Returns $\arg\max_m p_m$ and the full ranked distribution.
  * **`Noul` (Boolean / Binary Event):** Returns $P(\mathrm{true})$ over `{false_description, true_description}`.
  * **`Score` (Expected Level Utility):** Given ordered numeric levels $v_1 < v_2 < \dots < v_M$, returns the expectation $\hat{v} = \sum_{m=1}^M p_m v_m$.

### 3.3 `Evaluation` (Criterion-Conditioned Assessment)
* **Purpose:** Score one or more candidate states/outputs against explicit criteria $K = \lbrace k_1, \dots, k_C \rbrace$ (e.g., `syntactic_validity`, `spec_compliance`, `minimal_diff`) without forcing candidates or criteria to compete in one softmax.
* **Mechanism:** Conditions the `CED` readout on `(candidate_m, criterion_c, evidence)` to produce normalized utility scores $U(m, c) \in [0, 1]$ per criterion.

### 3.4 `Deduction`, `Induction`, and `Abduction` (The Triadic Reasoning Operators)
Rather than treating "reasoning" as an amorphous blob of `<think>` text, `MCL4-COE` separates Peirce's three logical modes into distinct objectives on the shared core:
1. **`Deduction` (Rule + Premise $\to$ Necessary Conclusion):**
   * Given explicit premises $\mathcal{P}$, rules $\mathcal{R}$, and a candidate step/conclusion $q$, predicts the 3-way logical status in a single softmax: **`entailed` | `contradicted` | `undetermined`**, plus the supporting premise links via `Association`.
2. **`Induction` (Observed Cases $\to$ Pattern & Generalization):**
   * Given a set of observed input-output examples $\lbrace x_i \to y_i \rbrace$, pools the invariant relational structure across examples to predict the underlying rule or distribution over a held-out case $x_{\mathrm{new}}$, paired with sample-size uncertainty.
3. **`Abduction` (Observed Evidence $\to$ Ranked Explanatory Hypotheses):**
   * Given observations/symptoms $\mathcal{O}$ (e.g., a failing unit test or compiler traceback) and candidate hypotheses $\mathcal{H} = \lbrace h_1, \dots, h_M \rbrace$, uses `Association` + `Evaluation` to rank hypotheses by explanatory fit and identify the **most informative discriminating test/observation** to run next.

### 3.5 `Abstraction` (Multi-Scale Compression Operator)
* **Purpose:** Map a set of detailed items or trace steps to a requested level of abstraction (`brief`, `standard`, `detailed`)—either as a pooled latent summary vector injected into the next recurrence pass or as a compact `T$` text summary.

### 3.6 `Metacognition` & `Meta-Attention` (Epistemic Control Operators)
* **Purpose:** Prevent hallucination and control compute allocation during multi-step thinking.
* **Mechanism:** Evaluates three calibrated binary signals (`Noul` readouts) on the current working state:
  1. `correctness`: Estimated probability that the current candidate answer/patch satisfies the task.
  2. `evidence_sufficiency`: Whether the currently attended premises/context are sufficient to determine the answer.
  3. `context_benefit`: Whether retrieving additional symbols/files or running another reasoning pass will improve the result.
* **`Meta-Attention`:** Uses `Association` + `Metacognition` scores to re-weight which evidence items are anchored in the next `RecurrentFeedback` pass.

### 3.7 `Inference` (`U$ -> T$ -> A$` Autoregressive Synthesis)
* **Purpose:** Produce the final compact scratchpad (`T$`) and verifiable code/math/text answer (`A$`), conditioned on the latent states and selections produced by the upstream `COE` operators.

---

## 4. Dynamic Multi-Pass Ensemble Execution

Because all 9 operators share the `MCL4-CED` backbone, a single prompt can dynamically chain operators in a closed loop before emitting a final answer:

```text
Prompt (U$)
  ├─► Pass 1 [Abduction + Association]: Rank candidate root causes & link relevant premises
  ├─► Pass 2 [Deduction + Evaluation]: Check top candidates against rules/constraints
  ├─► Pass 3 [Metacognition]: Is evidence_sufficiency > threshold?
  │     ├─► If NO: Trigger Retrieval / focus Meta-Attention -> repeat Pass 2
  │     └─► If YES: Proceed
  └─► Pass 4 [Decision / Inference]: Emit verified choice or compact T$ -> A$ output
```
