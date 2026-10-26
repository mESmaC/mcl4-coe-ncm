"""Milestone 3 & 4 Live Validation Benchmark on the RTX 5070 (sm_120).

Verifies:
1. Milestone 3 — Joint Multi-Operator + `0x8000` `U$ -> T$ -> A$` Supervised Warming
   and Composable Multi-Pass Controller (`abduction -> association -> decision ->
   deduction -> evaluation -> metacognition -> inference`).
2. Ornith-1.0 / 1.5 Self-Improvement Loop + Bounded Speculative Expert Mitosis
   (`grow_experts` / `prune_experts`) with `CPUOffloadAdamW`.
3. Milestone 4 — Native Blackwell `sm_120` `NVFP4` (`E2M1` + `FP8 E4M3` swizzled block
   scales + `cuBLASLt` `F.scaled_mm` + Hadamard-16 backward outlier spreading) live
   training and rollout verification on the RTX 5070.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcl4_coe import (
    ByteTokenizer,
    COEConfig,
    CPUOffloadAdamW,
    CognitiveOperatorEnsemble,
    CognitiveSchedule,
    DecisionExample,
    MCL4CEDBackbone,
    MCL4CEDConfig,
    MultiPassController,
    NoveltyTaskBuffer,
    ScaffoldCache,
    collate_decisions,
    compute_harness_reward,
    compute_joint_coe_loss,
    compute_task_reward,
    convert_linear_to_nvfp4,
    get_gemm_counts,
    group_normalized_advantages,
    grpo_clipped_loss,
    grow_experts,
    parse_request,
    prepare_model,
    prune_experts,
    reset_gemm_counts,
)


def _encode_padded_pairs(
    tokenizer: ByteTokenizer,
    left_texts: list[str],
    right_texts: list[str],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    l_enc = [tokenizer.encode(t) for t in left_texts]
    r_enc = [tokenizer.encode(t) for t in right_texts]
    l_max = max(map(len, l_enc))
    r_max = max(map(len, r_enc))
    l_ids = torch.full((len(l_enc), l_max), tokenizer.pad_id, dtype=torch.long, device=device)
    r_ids = torch.full((len(r_enc), r_max), tokenizer.pad_id, dtype=torch.long, device=device)
    l_lens = torch.tensor([len(x) for x in l_enc], dtype=torch.long, device=device)
    r_lens = torch.tensor([len(x) for x in r_enc], dtype=torch.long, device=device)
    for i, row in enumerate(l_enc):
        l_ids[i, : len(row)] = torch.tensor(row, device=device)
    for i, row in enumerate(r_enc):
        r_ids[i, : len(row)] = torch.tensor(row, device=device)
    return l_ids, l_lens, r_ids, r_lens


def _build_uta_batch(
    tokenizer: ByteTokenizer,
    episodes: list[tuple[str, str, str]],
    block_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    inputs = []
    labels = []
    for u, t, a in episodes:
        padded = tokenizer.encode_masked_turn(u, t, a, seq_len=block_size)
        arr = torch.tensor(padded, dtype=torch.long, device=device)
        inp = torch.where(arr == tokenizer.pad_id, tokenizer.pad_id, arr & 0x7FFF)
        tgt = torch.full_like(inp, -100)
        unmasked_next = (arr[1:] != tokenizer.pad_id) & ((arr[1:] & 0x8000) == 0)
        tgt[:-1] = torch.where(unmasked_next, inp[1:], -100)
        inputs.append(inp)
        labels.append(tgt)
    return torch.stack(inputs, dim=0), torch.stack(labels, dim=0)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== MCL4-COE Milestone 3 & 4 Validation on {device} ===")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} (capability={torch.cuda.get_device_capability(0)})")

    tokenizer = ByteTokenizer()
    torch.manual_seed(42)

    cfg = MCL4CEDConfig(
        d_model=128,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        ple_dim=32,
        expert_hidden_dim=256,
        num_experts=4,
        top_k=2,
        max_seq_len=256,
    )
    model = CognitiveOperatorEnsemble(
        MCL4CEDBackbone(cfg),
        COEConfig(projection_dim=32, initial_scale=15.0, recurrence_steps=2),
    )
    prepare_model(model, device=device, bf16_working_weights=(device.type == "cuda"))
    base_lr = 2.5e-3
    optimizer = CPUOffloadAdamW(model.parameters(), lr=base_lr, weight_decay=0.01, transfer_mode="batched")

    # 1. Construct Stage-2 Multi-Operator + 0x8000 U$/T$/A$ Episodes
    uta_episodes = [
        ("A rabbit carries two red balls. Count?", "rabbit has 2 red balls -> 2.", "2"),
        ("A cat carries no balls. Count?", "cat has no balls -> 0.", "0"),
        ("A dog carries one blue ball. Count?", "dog has 1 blue ball -> 1.", "1"),
        ("A fox carries three green balls. Count?", "fox has 3 green balls -> 3.", "3"),
    ]
    lm_batch = _build_uta_batch(tokenizer, uta_episodes, block_size=96, device=device)

    decision_requests = [
        {
            "state": "A rabbit carries two red balls.",
            "questions": {
                "animal": {
                    "type": "choice",
                    "instructions": "Which animal is present?",
                    "options": [
                        {"id": "rabbit", "description": "A rabbit."},
                        {"id": "cat", "description": "A cat."},
                        {"id": "dog", "description": "A dog."},
                    ],
                },
                "has_ball": {
                    "type": "noul",
                    "instructions": "Does it carry a ball?",
                    "false_description": "No balls.",
                    "true_description": "Carries red balls.",
                },
            },
            "targets": {"animal": "rabbit", "has_ball": True},
        },
        {
            "state": "A cat carries no balls.",
            "questions": {
                "animal": {
                    "type": "choice",
                    "instructions": "Which animal is present?",
                    "options": [
                        {"id": "rabbit", "description": "A rabbit."},
                        {"id": "cat", "description": "A cat."},
                        {"id": "dog", "description": "A dog."},
                    ],
                },
                "has_ball": {
                    "type": "noul",
                    "instructions": "Does it carry a ball?",
                    "false_description": "No balls.",
                    "true_description": "Carries red balls.",
                },
            },
            "targets": {"animal": "cat", "has_ball": False},
        },
    ]
    dec_examples: list[DecisionExample] = []
    for req in decision_requests:
        dec_examples.extend(parse_request(req, require_targets=True))
    decision_batch = collate_decisions(dec_examples, cfg.max_seq_len, device=device)

    # Deduction 3-way triples: 0=entailed, 1=contradicted, 2=undetermined
    premises = [
        "#U$# A rabbit carries two red balls.",
        "#U$# A rabbit carries two red balls.",
        "#U$# A rabbit carries two red balls.",
        "#U$# A cat carries no balls.",
        "#U$# A cat carries no balls.",
        "#U$# A cat carries no balls.",
    ]
    conclusions = [
        "A rabbit carries red balls.",
        "A cat carries no balls.",
        "Tomorrow it will rain.",
        "A cat has no balls.",
        "A rabbit carries red balls.",
        "The room is upstairs.",
    ]
    ded_p, ded_pl, ded_c, ded_cl = _encode_padded_pairs(tokenizer, premises, conclusions, device)
    ded_targets = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long, device=device)
    deduction_batch = {
        "premise_ids": ded_p,
        "premise_lengths": ded_pl,
        "conclusion_ids": ded_c,
        "conclusion_lengths": ded_cl,
        "targets": ded_targets,
    }

    # Evaluation & Metacognition supervision on the same pairs
    eval_targets = torch.tensor([1.0, 0.0, 0.2, 1.0, 0.0, 0.2], dtype=torch.float32, device=device)
    evaluation_batch = {
        "criterion_ids": ded_p,
        "criterion_lengths": ded_pl,
        "candidate_ids": ded_c,
        "candidate_lengths": ded_cl,
        "targets": eval_targets,
    }
    meta_targets = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.5],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.5],
        ],
        dtype=torch.float32,
        device=device,
    )
    metacognition_batch = {
        "state_ids": ded_p,
        "state_lengths": ded_pl,
        "candidate_ids": ded_c,
        "candidate_lengths": ded_cl,
        "targets": meta_targets,
    }

    t0 = time.perf_counter()
    total_steps = 80
    for step in range(1, total_steps + 1):
        model.train()
        lr = base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * step / total_steps)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            joint = compute_joint_coe_loss(
                model,
                lm_batch=lm_batch,
                decision_batch=decision_batch,
                deduction_batch=deduction_batch,
                evaluation_batch=evaluation_batch,
                metacognition_batch=metacognition_batch,
            )
        joint.total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_joint = time.perf_counter() - t0

    # Evaluate MultiPassController after joint training
    model.eval()
    controller = MultiPassController(
        model,
        schedule=CognitiveSchedule(
            operators=("abduction", "association", "decision", "deduction", "evaluation", "metacognition"),
            recurrence_steps=2,
        ),
    )
    novelty_buf = NoveltyTaskBuffer(projection_dim=32, max_capacity=10_000, device=device)
    with torch.no_grad():
        mp_out = controller(
            decision_batch["context_ids"],
            decision_batch["context_lengths"],
            decision_batch["option_ids"],
            decision_batch["option_lengths"],
            targets=decision_batch["targets"],
            history_buffer= torch.randn(128, 32, device=device),
        )
        ded_eval = model.forward_deduction(**deduction_batch)
        dec_acc = int((mp_out.selected_indices == decision_batch["targets"]).sum().item())
        ded_acc = int((ded_eval.probabilities.argmax(-1) == ded_targets).sum().item())
        uta_sample = controller.generate_uta(
            "A rabbit carries two red balls. Count?", max_new_tokens=48
        )

    print(
        f"\n[1] Milestone 3 — Joint Multi-Operator + 0x8000 U$/T$/A$ Controller ({elapsed_joint:.2f}s):\n"
        f"    Multi-Pass Decision Accuracy:  {dec_acc}/{len(dec_examples)} ({100.0 * dec_acc / len(dec_examples):.1f}%)\n"
        f"    3-Way Deduction Accuracy:      {ded_acc}/{len(ded_targets)} ({100.0 * ded_acc / len(ded_targets):.1f}%)\n"
        f"    Final U$/T$/A$ Masked LM Loss: {joint.lm_loss.item():.4f}\n"
        f"    Generated U$ -> T$ -> A$ Trace: {uta_sample!r}"
    )

    # 2. Ornith-1.0 / 1.5 Self-Improvement & ExpertLifecycle Mitosis/Pruning
    novelty_buf.add(mp_out.decision.state_features)
    nov_scores = novelty_buf.compute_novelty(mp_out.decision.state_features)
    r_task_frontier = compute_task_reward(validity=1.0, pass_rate=0.20, novelty=0.85).item()
    r_task_invalid = compute_task_reward(validity=0.0, pass_rate=0.20, novelty=0.85).item()
    r_harness = compute_harness_reward(
        torch.tensor([0.95], device=device),
        torch.tensor([[0.0, 0.5, 1.0]], device=device),
        torch.tensor([[0.05, 0.5, 0.95]], device=device),
        negative_harness_scores=torch.tensor([[0.0, 0.0]], device=device),
    ).item()
    scaff_cache = ScaffoldCache()
    scaff_cache.update_if_better(
        "task_001",
        [
            "decision -> inference",
            "abduction -> association -> decision -> deduction -> evaluation -> metacognition",
        ],
        [0.45, 0.98],
        step=1,
    )

    optimizer.zero_grad(set_to_none=True)
    grow_res = grow_experts(model, model.backbone, optimizer, parent_index=0)
    prune_res = prune_experts(model, model.backbone, optimizer, expert_index=grow_res.new_num_experts - 1)
    print(
        f"\n[2] Ornith-1.0 / 1.5 Self-Improvement & ExpertLifecycle:\n"
        f"    Winning Cached Scaffold:       {scaff_cache.get('task_001')}\n"
        f"    Ornith-1.5 R_task (p*=0.20):   {r_task_frontier:.4f} (hard-gate V=0 -> {r_task_invalid:.4f})\n"
        f"    Ornith-1.5 R_harness Fidelity: {r_harness:.4f} | Self-Novelty after insert: {nov_scores.mean().item():.4f}\n"
        f"    Expert Mitosis & Pruning:      {grow_res.old_num_experts} -> {grow_res.new_num_experts} ({grow_res.new_parameters:,} params) -> {prune_res.new_num_experts} experts"
    )

    # 3. Milestone 4 — Native Blackwell sm_120 NVFP4 Training & Rollout Benchmark
    if device.type == "cuda":
        reset_gemm_counts()
        nvfp4_model = CognitiveOperatorEnsemble(
            MCL4CEDBackbone(cfg),
            COEConfig(projection_dim=32, initial_scale=15.0, recurrence_steps=1),
        ).to(device)
        converted = convert_linear_to_nvfp4(nvfp4_model, min_features=32)
        nvfp4_opt = torch.optim.AdamW(nvfp4_model.parameters(), lr=2e-3, weight_decay=0.01)

        t_fp4 = time.perf_counter()
        for step in range(1, 41):
            nvfp4_model.train()
            nvfp4_opt.zero_grad(set_to_none=True)
            out_fp4 = nvfp4_model.forward_decision(**decision_batch)
            assert out_fp4.loss is not None
            out_fp4.loss.backward()
            torch.nn.utils.clip_grad_norm_(nvfp4_model.parameters(), 1.0)
            nvfp4_opt.step()
        torch.cuda.synchronize()
        elapsed_fp4 = time.perf_counter() - t_fp4

        nvfp4_model.eval()
        with torch.no_grad():
            eval_fp4 = nvfp4_model.forward_decision(**decision_batch)
        fp4_acc = int((eval_fp4.probabilities.argmax(-1) == decision_batch["targets"]).sum().item())
        counts = get_gemm_counts()
        print(
            f"\n[3] Milestone 4 — Native Blackwell NVFP4 (`F.scaled_mm` E2M1 + FP8 E4M3 + Hadamard16):\n"
            f"    Converted NVFP4Linear Layers:  {converted}\n"
            f"    NVFP4 Decision Accuracy:       {fp4_acc}/{len(dec_examples)} ({100.0 * fp4_acc / len(dec_examples):.1f}%) | ce_loss={eval_fp4.ce_loss.item():.4f} | wall_time={elapsed_fp4:.2f}s\n"
            f"    Hardware NVFP4 GEMM Counts:    forward={counts['forward']:,}, dgrad={counts['dgrad']:,}, wgrad={counts['wgrad']:,}"
        )


if __name__ == "__main__":
    main()
