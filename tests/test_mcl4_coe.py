"""Unit and mathematical invariant tests for MCL4-COE."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from mcl4_coe import (
    ByteTokenizer,
    COEConfig,
    CPUOffloadAdamW,
    CognitiveOperatorEnsemble,
    CognitiveSchedule,
    GLiNER25CorpusGate,
    MCL4CEDBackbone,
    MCL4CEDConfig,
    MultiPassController,
    NVFP4Linear,
    NoveltyTaskBuffer,
    ScaffoldCache,
    TokenBlockDataset,
    collate_decisions,
    compute_harness_reward,
    compute_joint_coe_loss,
    compute_task_reward,
    convert_linear_to_nvfp4,
    evaluate_byte_sieve,
    get_gemm_counts,
    group_normalized_advantages,
    grpo_clipped_loss,
    grow_experts,
    normalize_prose_unwrapping,
    parse_request,
    pearson_similarity,
    prepare_model,
    prune_experts,
    reset_gemm_counts,
    staleness_weight,
)


def _tiny_model(device: str = "cpu") -> CognitiveOperatorEnsemble:
    torch.manual_seed(7)
    cfg = MCL4CEDConfig(
        d_model=64,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ple_dim=16,
        expert_hidden_dim=128,
        num_experts=4,
        top_k=2,
        max_seq_len=128,
    )
    backbone = MCL4CEDBackbone(cfg)
    coe = CognitiveOperatorEnsemble(backbone, COEConfig(projection_dim=16, recurrence_steps=2))
    return coe.to(device).eval()


def test_bitwise_permutation_invariance_across_candidates() -> None:
    """Reordering candidate options must permute logits and probabilities down to the last bit."""
    model = _tiny_model()
    request = {
        "state": "A rabbit carries two red balls.",
        "questions": {
            "animal": {
                "type": "choice",
                "instructions": "Which animal is here?",
                "options": [
                    {"id": "rabbit", "description": "A rabbit."},
                    {"id": "cat", "description": "A cat carries a bell."},
                    {"id": "dog", "description": "A large brown dog."},
                ],
            }
        },
    }
    examples = parse_request(request)
    batch = collate_decisions(examples, max_length=128)

    perm = torch.tensor([2, 0, 1], dtype=torch.long)
    batch_perm = {
        "context_ids": batch["context_ids"],
        "context_lengths": batch["context_lengths"],
        "option_ids": batch["option_ids"][:, perm, :],
        "option_lengths": batch["option_lengths"][:, perm],
    }

    with torch.no_grad():
        out_orig = model.forward_decision(**batch)
        out_perm = model.forward_decision(**batch_perm)

    assert torch.equal(out_orig.similarities[:, perm], out_perm.similarities)
    assert torch.equal(out_orig.logits[:, perm], out_perm.logits)
    assert torch.allclose(out_orig.probabilities[:, perm], out_perm.probabilities, atol=1e-7)


def test_absent_option_padding_invariance() -> None:
    """Adding a zero-length padded option slot must not alter valid option scores."""
    model = _tiny_model()
    request = {
        "state": "A cat carries one red ball.",
        "questions": {
            "has_ball": {
                "type": "noul",
                "instructions": "Is the animal carrying a ball?",
                "false_description": "The animal has no ball.",
                "true_description": "The animal carries a ball.",
            }
        },
    }
    examples = parse_request(request)
    batch = collate_decisions(examples, max_length=128)

    padded_opt_ids = torch.full(
        (1, 3, batch["option_ids"].shape[-1]),
        ByteTokenizer.pad_id,
        dtype=torch.long,
    )
    padded_opt_ids[:, :2, :] = batch["option_ids"]
    padded_opt_lengths = torch.tensor([[batch["option_lengths"][0, 0], batch["option_lengths"][0, 1], 0]])

    with torch.no_grad():
        out2 = model.forward_decision(**batch)
        out3 = model.forward_decision(
            context_ids=batch["context_ids"],
            context_lengths=batch["context_lengths"],
            option_ids=padded_opt_ids,
            option_lengths=padded_opt_lengths,
        )

    assert torch.equal(out2.similarities, out3.similarities[:, :2])
    assert torch.allclose(out2.probabilities, out3.probabilities[:, :2], atol=1e-7)
    assert out3.probabilities[0, 2].item() == 0.0


def test_0x8000_prompt_masking_and_mandatory_t_scratchpad(tmp_path: Path) -> None:
    """Verify 0x8000 high-bit prompt masking and mandatory #T$# scratchpad enforcement."""
    tok = ByteTokenizer()
    with pytest.raises(ValueError, match="mandatory"):
        tok.encode_masked_turn("What is 2+2?", "", "4", seq_len=64)

    block = tok.encode_masked_turn("What is 3*4?", "3 * 4 = 12.", "12", seq_len=64)
    assert len(block) == 64
    bin_path = tmp_path / "train.bin"
    # Write 2 blocks + 1 trailing token so len(dataset) == 2
    arr = np.asarray(block + block + [ByteTokenizer.pad_id], dtype="<u2")
    bin_path.write_bytes(arr.tobytes())

    ds = TokenBlockDataset(bin_path, seq_len=64)
    assert len(ds) == 2
    batch = ds.batch([0])
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

    # High bit must be stripped in input_ids
    assert (input_ids < 260).all()
    decoded_in = tok.decode(input_ids)
    assert decoded_in.startswith("#U$# What is 3*4?\n#T$# 3 * 4 = 12.\n#A$# 12")

    # Prompt tokens and pad tokens must be masked as -100 in labels
    unmasked_target_tokens = [t for t in labels.tolist() if t != -100]
    decoded_target = tok.decode(unmasked_target_tokens)
    assert decoded_target == "#T$# 3 * 4 = 12.\n#A$# 12"


def test_coe_operators_and_novelty_buffer() -> None:
    """Verify Association + Novelty N(q), Deduction, Metacognition, Abstraction, and Inference."""
    model = _tiny_model()
    tok = ByteTokenizer()
    texts = ["#U$# Write a binary search.", "#U$# Write a merge sort.", "#U$# Prove sqrt(2) is irrational."]
    encoded = [tok.encode(t) for t in texts]
    max_len = max(map(len, encoded))
    item_ids = torch.full((len(texts), max_len), tok.pad_id, dtype=torch.long)
    item_lengths = torch.tensor([len(row) for row in encoded], dtype=torch.long)
    for i, row in enumerate(encoded):
        item_ids[i, : len(row)] = torch.tensor(row)

    history = torch.randn(1024, 16)
    with torch.no_grad():
        assoc = model.forward_association_matrix(item_ids, item_lengths, history_buffer=history)
        assert assoc.similarities.shape == (3, 3)
        assert assoc.novelty is not None and assoc.novelty.shape == (3,)

        ded = model.forward_deduction(
            item_ids[:2],
            item_lengths[:2],
            item_ids[1:],
            item_lengths[1:],
            targets=torch.tensor([0, 2]),
        )
        assert ded.probabilities.shape == (2, 3)
        assert ded.loss is not None and torch.isfinite(ded.loss)

        meta = model.forward_metacognition(
            item_ids[:2],
            item_lengths[:2],
            item_ids[1:],
            item_lengths[1:],
            targets=torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.5, 1.0]]),
        )
        assert meta.probabilities.shape == (2, 3)
        assert meta.loss is not None and torch.isfinite(meta.loss)

        summary = model.compress_abstraction(item_ids, item_lengths, level="brief")
        assert summary.shape == (3, 64)

        lm_out = model.backbone(item_ids, targets=item_ids)
        assert lm_out.logits.shape == (3, max_len, tok.vocab_size)
        assert lm_out.loss is not None and torch.isfinite(lm_out.loss)


def test_cpu_offload_adamw_step() -> None:
    """Verify CPUOffloadAdamW updates working weights and preserves FP32 masters on CPU and CUDA."""
    model = _tiny_model()
    prepare_model(model, device="cpu", bf16_working_weights=False)
    opt = CPUOffloadAdamW(model.parameters(), lr=1e-2, transfer_mode="per_parameter")
    x = torch.randint(0, 256, (2, 16), dtype=torch.long)
    out = model.backbone(x, targets=x)
    assert out.loss is not None
    initial_loss = out.loss.item()
    out.loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    out2 = model.backbone(x, targets=x)
    assert out2.loss is not None and out2.loss.item() < initial_loss
    assert pearson_similarity(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 4.0, 6.0])).item() == pytest.approx(1.0, abs=1e-5)

    if torch.cuda.is_available():
        cuda_model = _tiny_model().train()
        prepare_model(cuda_model, device="cuda", bf16_working_weights=True)
        cuda_opt = CPUOffloadAdamW(cuda_model.parameters(), lr=5e-3, transfer_mode="batched")
        x_cuda = x.to("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            c_out1 = cuda_model.backbone(x_cuda, targets=x_cuda)
        assert c_out1.loss is not None
        c_out1.loss.backward()
        cuda_opt.step()
        cuda_opt.zero_grad(set_to_none=True)
        assert all(torch.isfinite(p).all() for p in cuda_model.parameters())
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            c_out2 = cuda_model.backbone(x_cuda, targets=x_cuda)
        assert c_out2.loss is not None and torch.isfinite(c_out2.loss)
        assert c_out2.loss.item() < c_out1.loss.item()


def test_multipass_controller_and_joint_coe_loss() -> None:
    """Verify Milestone 3 MultiPassController, generate_uta, and compute_joint_coe_loss."""
    model = _tiny_model()
    sched = CognitiveSchedule.from_scaffold_text(
        "abduction -> association -> decision -> deduction -> evaluation -> metacognition",
        recurrence_steps=2,
    )
    controller = MultiPassController(model, schedule=sched)
    request = {
        "state": "A rabbit carries two red balls.",
        "questions": {
            "animal": {
                "type": "choice",
                "instructions": "Which animal is here?",
                "options": [
                    {"id": "rabbit", "description": "A rabbit."},
                    {"id": "cat", "description": "A cat."},
                ],
            }
        },
        "targets": {"animal": "rabbit"},
    }
    batch = collate_decisions(parse_request(request, require_targets=True), max_length=128)
    history = torch.randn(32, 16)
    out = controller(
        batch["context_ids"],
        batch["context_lengths"],
        batch["option_ids"],
        batch["option_lengths"],
        targets=batch["targets"],
        history_buffer=history,
    )
    assert out.selected_indices.shape == (1,)
    assert out.deduction is not None and out.deduction.probabilities.shape == (1, 3)
    assert out.evaluation is not None and out.evaluation.utilities.shape == (1,)
    assert out.metacognition is not None and out.metacognition.probabilities.shape == (1, 3)
    assert out.novelty is not None and out.novelty.shape == (1,)

    uta_text = controller.generate_uta("What is 2+2?", max_new_tokens=12)
    assert "#U$# What is 2+2?\n#T$# " in uta_text
    assert "\n#A$# " in uta_text

    lm_x = torch.randint(0, 256, (2, 16), dtype=torch.long)
    joint = compute_joint_coe_loss(
        model,
        lm_batch=(lm_x, lm_x),
        decision_batch=batch,
    )
    assert torch.isfinite(joint.total_loss)
    assert joint.lm_loss is not None and joint.decision_loss is not None


def test_ornith_10_and_15_rewards_and_grpo() -> None:
    """Verify Ornith-1.0 ScaffoldCache and Ornith-1.5 R_task, R_harness, and staleness GRPO."""
    buf = NoveltyTaskBuffer(projection_dim=16, max_capacity=64)
    v1 = torch.arange(16, dtype=torch.float32)
    buf.add(v1)
    assert len(buf) == 1
    assert buf.compute_novelty(v1).item() == pytest.approx(0.0, abs=1e-5)
    assert buf.compute_novelty(-v1).item() == pytest.approx(2.0, abs=1e-5)

    # Hard gate: V=0 => R_task=0 even if pass_rate=0.2 and novelty=1.0
    assert compute_task_reward(0.0, 0.2, 1.0).item() == 0.0
    # Frontier peak at p* = 0.2
    r_frontier = compute_task_reward(1.0, 0.2, 1.0).item()
    r_trivial = compute_task_reward(1.0, 1.0, 1.0).item()
    assert r_frontier == pytest.approx(1.0, abs=1e-6)
    assert r_frontier > r_trivial

    # Harness reward rejects negative-rollout leakage
    c = torch.tensor([1.0])
    h_scores = torch.tensor([[0.0, 0.5, 1.0]])
    ref_util = torch.tensor([[0.0, 0.5, 1.0]])
    assert compute_harness_reward(c, h_scores, ref_util).item() == pytest.approx(1.0, abs=1e-5)
    assert (
        compute_harness_reward(
            c, h_scores, ref_util, negative_harness_scores=torch.tensor([[0.2]])
        ).item()
        == 0.0
    )

    # Staleness weighting and topology invalidation
    w = staleness_weight(torch.tensor([0, 1, 2, 5]), k1=1, k2=3, decay_lambda=0.5)
    assert w[0].item() == 1.0 and w[1].item() == 1.0
    assert 0.0 < w[2].item() < 1.0
    assert w[3].item() == 0.0
    assert staleness_weight(0, topology_changed=True).item() == 0.0

    # ScaffoldCache monotonic update
    cache = ScaffoldCache()
    assert cache.update_if_better("t1", ["s_low", "s_high"], [0.1, 0.8], step=1)
    assert cache.get("t1") == "s_high"
    assert not cache.update_if_better("t1", ["s_worse"], [0.5], step=2)
    assert cache.get("t1") == "s_high"

    # GRPO loss
    adv = group_normalized_advantages(torch.tensor([0.0, 1.0, 2.0]))
    logp = torch.zeros((3, 4), requires_grad=True)
    beh_logp = torch.zeros((3, 4))
    loss = grpo_clipped_loss(logp, beh_logp, adv, policy_age=0)
    loss.backward()
    assert logp.grad is not None and torch.isfinite(logp.grad).all()


def test_expert_mitosis_and_pruning() -> None:
    """Verify grow_experts and prune_experts across encoder and decoder layers with CPUOffloadAdamW."""
    model = _tiny_model().train()
    prepare_model(model, device="cpu", bf16_working_weights=False)
    opt = CPUOffloadAdamW(model.parameters(), lr=1e-3, transfer_mode="per_parameter")

    res_grow = grow_experts(model, model.backbone, opt, parent_index=0)
    assert res_grow.old_num_experts == 4 and res_grow.new_num_experts == 5
    assert model.backbone.config.num_experts == 5

    res_prune = prune_experts(model, model.backbone, opt, expert_index=4)
    assert res_prune.old_num_experts == 5 and res_prune.new_num_experts == 4
    assert model.backbone.config.num_experts == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU for native NVFP4")
def test_nvfp4_linear_conversion_and_gemm_on_cuda() -> None:
    """Verify native Blackwell NVFP4 GEMM, 2:4 transposable sparsity, SR-STE, and causal scaling on RTX 5070."""
    from mcl4_coe import compute_2to4_mask

    # 1. Verify 2:4 transposable mask is exact 2:4 along BOTH rows (W) and columns (W.T)
    w_test = torch.randn(64, 128, device="cuda")
    mask = compute_2to4_mask(w_test, mode="2to4_transposable")
    assert mask.shape == (64, 128)
    assert torch.all(mask.view(64, 32, 4).sum(dim=-1) == 2)
    assert torch.all(mask.t().contiguous().view(128, 16, 4).sum(dim=-1) == 2)

    # 2. Verify causal_activation_scale=True prevents future tokens from leaking into earlier rows
    layer = NVFP4Linear(64, 64, bias=False, device="cuda", sparsity_mode="2to4_transposable").eval()
    seq_a = torch.randn(2, 16, 64, device="cuda")
    seq_b = seq_a.clone()
    seq_b[:, 8:, :] = seq_b[:, 8:, :] * 25.0 - 10.0
    with torch.no_grad():
        out_a = layer(seq_a)
        out_b = layer(seq_b)
    torch.testing.assert_close(out_a[:, :8, :], out_b[:, :8, :], rtol=0, atol=0)

    # 3. Verify expert-only conversion, 2:4 sparse NVFP4 GEMM execution, and SR-STE master weight update
    reset_gemm_counts()
    model = _tiny_model(device="cuda").train()
    num_converted = convert_linear_to_nvfp4(
        model.backbone,
        min_features=32,
        experts_only=True,
        sparsity_mode="2to4_transposable",
        sr_ste_decay=1e-3,
    )
    assert num_converted > 0
    assert any(isinstance(m, NVFP4Linear) for m in model.backbone.modules())
    # Routers and attention projections must remain standard high-precision nn.Linear
    for enc in model.backbone.encoder_layers:
        assert not isinstance(enc.moe.router, NVFP4Linear)
        assert not isinstance(enc.attention.qkv, NVFP4Linear)
        assert isinstance(enc.moe.experts[0].up_proj, NVFP4Linear)
        assert isinstance(enc.moe.experts[0].down_proj, NVFP4Linear)

    opt = torch.optim.AdamW(model.backbone.parameters(), lr=1e-2)
    w_before = model.backbone.encoder_layers[0].moe.experts[0].up_proj.weight.detach().clone()
    x = torch.randint(0, 256, (2, 16), device="cuda", dtype=torch.long)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model.backbone(x, targets=x)
    assert out.loss is not None
    out.loss.backward()
    opt.step()

    w_after = model.backbone.encoder_layers[0].moe.experts[0].up_proj.weight.detach()
    assert not torch.equal(w_before, w_after)
    counts = get_gemm_counts()
    assert counts["forward"] > 0 and counts["dgrad"] > 0 and counts["wgrad"] > 0
    assert counts["sparse_2to4"] > 0


def test_curation_byte_sieve_and_gliner25_gate() -> None:
    """Verify Stage-0 CPU byte sieve, marker sanitization, and Stage-1 GLiNER2.5-Decide gate."""
    # Too short -> rejected in Stage 0
    short_res = evaluate_byte_sieve("Too short.")
    assert not short_res.keep and short_res.stage_rejected == "stage0_byte_sieve"

    # High-quality algorithmic prose with accidental #T$# marker in raw text
    good_raw = (
        "In digital signal processing, a matched filter correlates a known template,\n"
        "or reference signal, with an unknown signal to detect the presence of the\n"
        "template in the unknown signal. Note that raw literal #T$# markers are escaped."
    )

    def _mock_gliner25(window: str, schema: dict) -> dict:
        assert "domain" in schema and "reasoning_density" in schema and "defects" in schema
        if "Buy now" in window:
            return {
                "domain": {"label": "seo_marketing_or_boilerplate", "confidence": 0.94},
                "reasoning_density": {"label": "low_value_filler_or_stub", "confidence": 0.91},
                "defects": [{"label": "paywall_or_navigation_chrome", "confidence": 0.88}],
            }
        return {
            "domain": {"label": "expository_prose_or_textbook", "confidence": 0.96},
            "reasoning_density": {"label": "high_logical_or_explanatory_density", "confidence": 0.92},
            "defects": [],
        }

    gate = GLiNER25CorpusGate(classifier_fn=_mock_gliner25)
    dec_good = gate.evaluate_document(good_raw)
    assert dec_good.keep
    assert dec_good.sanitized_text is not None and "#T$#" not in dec_good.sanitized_text
    unwrapped = normalize_prose_unwrapping(dec_good.sanitized_text)
    assert "\nor reference" not in unwrapped

    seo_spam = (
        "Buy now and subscribe to our premium newsletter for exclusive deals on cloud\n"
        "hosting, enterprise SaaS solutions, and limited-time promotional discounts today!"
    )
    dec_bad = gate.evaluate_document(seo_spam)
    assert not dec_bad.keep and dec_bad.stage_rejected == "stage1_gliner25_decide"


def test_sharded_pipeline_and_index_receipt(tmp_path: Path) -> None:
    """Verify BlockAlignedShardWriter, per-shard receipts, IndexReceipt, and ShardedTokenBlockDataset."""
    from mcl4_coe import (
        BlockAlignedShardWriter,
        EpochBlockSampler,
        HostBatchPrefetcher,
        ShardedTokenBlockDataset,
        extract_typed_decision_uta,
        extract_uta_episodes,
    )

    writer = BlockAlignedShardWriter(
        tmp_path,
        track="uta_trace",
        split="train",
        seq_len=256,
        max_blocks_per_shard=2,
    )

    episodes = extract_uta_episodes(
        {
            "messages": [
                {"role": "user", "content": "What is 6 * 7?"},
                {"role": "assistant", "content": "<think>6 * 7 = 42.</think>\n42"},
            ]
        }
    )
    assert len(episodes) == 1
    dec_ep = extract_typed_decision_uta(
        {
            "state": "A rabbit carries two red balls.",
            "kind": "choice",
            "question": "Which animal is present?",
            "options": ["A rabbit.", "A cat."],
            "target": [1.0, 0.0],
        }
    )
    assert dec_ep is not None

    for i in range(10):
        u, t, a = episodes[0]
        assert writer.add_uta_episode(f"{u} ({i})", t, a)
    assert writer.add_uta_episode(*dec_ep)
    # Oversize trace (>256 bytes) must be rejected rather than severed mid-thought
    assert not writer.add_uta_episode("Q" * 200, "T" * 100, "A")
    writer.record_provenance(
        corpus="synthetic_test",
        source_file="test.jsonl",
        row_start=1,
        row_end=7,
        episodes_accepted=7,
        episodes_rejected_stage0=0,
    )
    writer.flush_shard()

    index_file = tmp_path / "index_receipt.json"
    assert index_file.exists()
    ds = ShardedTokenBlockDataset(index_file, track="uta_trace", split="train")
    assert len(ds) >= 2
    assert len(ds.shards) >= 1

    sampler = EpochBlockSampler(len(ds), seed=42)
    prefetcher = HostBatchPrefetcher(ds, micro_batch_size=2, pin_memory=False)
    try:
        batches = prefetcher.get(sampler.take(len(ds)))
        tok = ByteTokenizer()
        first_in = tok.decode(batches[0]["input_ids"][0])
        assert "#U$# " in first_in and "#T$# " in first_in and "#A$# " in first_in
        unmasked = [t for t in batches[0]["labels"][0].tolist() if t != -100]
        decoded_tgt = tok.decode(unmasked)
        # Prompt (#U$# ...) must be masked out of targets; only #T$# and #A$# are supervised
        assert decoded_tgt.startswith("#T$# ")
    finally:
        prefetcher.close()


def test_multitrack_mixture_and_curriculum_sampler(tmp_path: Path) -> None:
    """Verify MultiTrackMixtureDataset, CurriculumMixtureSchedule, and CurriculumMixtureSampler."""
    from mcl4_coe import (
        BlockAlignedShardWriter,
        CurriculumMixtureSampler,
        CurriculumMixtureSchedule,
        HostBatchPrefetcher,
        MultiTrackMixtureDataset,
    )

    track_receipts: dict[str, Path] = {}
    for t_name in ("pretrain_lm", "uta_trace", "typed_decision"):
        t_dir = tmp_path / t_name
        w = BlockAlignedShardWriter(
            t_dir,
            track=t_name,
            split="train",
            seq_len=128,
            max_blocks_per_shard=4,
        )
        for i in range(6):
            if t_name == "pretrain_lm":
                w.add_pretrain_document(f"Expository theorem {i}: energy is conserved in closed systems. " * 3)
            else:
                w.add_uta_episode(f"Compute {i}+1", f"{i} + 1 = {i+1}", str(i + 1))
        w.flush_shard()
        track_receipts[t_name] = t_dir / "index_receipt.json"

    mix_ds = MultiTrackMixtureDataset(track_receipts)
    assert len(mix_ds) > 0
    sched = CurriculumMixtureSchedule(
        start_weights={"pretrain_lm": 0.7, "uta_trace": 0.2, "typed_decision": 0.1},
        end_weights={"pretrain_lm": 0.1, "uta_trace": 0.5, "typed_decision": 0.4},
        mode="cosine",
    )
    sampler = CurriculumMixtureSampler(mix_ds, sched, seed=99)

    q_start = sampler.allocate_quotas(10, progress=0.0)
    assert q_start == {"pretrain_lm": 7, "uta_trace": 2, "typed_decision": 1}
    q_end = sampler.allocate_quotas(10, progress=1.0)
    assert q_end == {"pretrain_lm": 1, "uta_trace": 5, "typed_decision": 4}

    idx_batch = sampler.take(10, progress=0.5)
    assert idx_batch.shape == (10,)

    # Verify state_dict checkpoint round-trip
    saved_state = sampler.state_dict()
    next_expected = sampler.take(6, progress=0.8)
    sampler.load_state_dict(saved_state)
    next_restored = sampler.take(6, progress=0.8)
    assert torch.equal(next_expected, next_restored)

    prefetcher = HostBatchPrefetcher(mix_ds, micro_batch_size=5, pin_memory=False)
    try:
        batches = prefetcher.get(idx_batch)
        assert len(batches) == 2
        assert batches[0]["input_ids"].shape == (5, 128)
        assert batches[0]["track_ids"].shape == (5,)
    finally:
        prefetcher.close()


def test_trainer_checkpoints_and_webui_endpoints(tmp_path: Path) -> None:
    """Verify MCL4COETrainer, checkpoint receipts, and Web UI REST endpoints."""
    import json
    import threading
    from urllib.request import Request, urlopen

    from mcl4_coe import (
        BlockAlignedShardWriter,
        MCL4COETrainer,
        TrainingControllerState,
        TrainingRunConfig,
        list_checkpoints,
        list_runs,
        load_run_metrics,
        start_web_ui_server,
    )

    shards_dir = tmp_path / "shards"
    runs_dir = tmp_path / "runs"
    for t_name in ("pretrain_lm", "uta_trace", "typed_decision"):
        w = BlockAlignedShardWriter(
            shards_dir / t_name,
            track=t_name,
            split="train",
            seq_len=64,
            max_blocks_per_shard=4,
        )
        for i in range(4):
            if t_name == "pretrain_lm":
                w.add_pretrain_document(f"Theorem {i}: conservation of momentum in isolated frames.")
            else:
                w.add_uta_episode(f"Add {i}+2", f"{i}+2={i+2}", str(i + 2))
        w.flush_shard()

    cfg = TrainingRunConfig(
        run_id="unit_test_run",
        runs_root=str(runs_dir),
        shards_root=str(shards_dir),
        corpora_root=str(tmp_path / "no_corpora"),
        d_model=32,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ple_dim=8,
        expert_hidden_dim=64,
        num_experts=2,
        top_k=1,
        max_seq_len=64,
        projection_dim=8,
        total_steps=2,
        batch_size=3,
        micro_batch_size=3,
        checkpoint_every=2,
        sample_every=2,
    )
    trainer = MCL4COETrainer(cfg)
    final_m = trainer.train()
    assert final_m["step"] == 2
    assert "loss_pretrain_lm" in final_m

    runs = list_runs(runs_dir)
    assert len(runs) == 1 and runs[0]["run_id"] == "unit_test_run"
    run_data = load_run_metrics("unit_test_run", runs_dir)
    assert len(run_data["metrics"]) == 2
    ckpts = list_checkpoints(runs_dir)
    assert len(ckpts) == 1

    controller = TrainingControllerState(
        runs_root=runs_dir,
        shards_root=shards_dir,
        corpora_root=tmp_path / "no_corpora",
    )
    server = start_web_ui_server(controller, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{port}"
        html = urlopen(f"{base_url}/").read().decode("utf-8")
        assert "MCL4-COE Training" in html

        st = json.loads(urlopen(f"{base_url}/api/status").read().decode("utf-8"))
        assert "is_running" in st and "pretrain_lm" in st["shards"]

        eval_req = Request(
            f"{base_url}/api/checkpoints/eval",
            data=json.dumps({"checkpoints": [ckpts[0]["meta_file"]], "prompt": "Add 1+2"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        eval_res = json.loads(urlopen(eval_req).read().decode("utf-8"))
        assert len(eval_res["results"]) == 1
        assert "#U$# Add 1+2" in eval_res["results"][0]["trace"]
    finally:
        server.shutdown()
        server.server_close()

    # Verify seamless 2E -> 4E Expert Mitosis upcycling when resuming from checkpoint
    upcycle_cfg = TrainingRunConfig(
        run_id="unit_test_upcycled_4e",
        runs_root=str(runs_dir),
        shards_root=str(shards_dir),
        corpora_root=str(tmp_path / "no_corpora"),
        resume_checkpoint=ckpts[0]["meta_file"],
        d_model=32,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ple_dim=8,
        expert_hidden_dim=64,
        num_experts=4,
        top_k=2,
        max_seq_len=64,
        projection_dim=8,
        total_steps=3,
        batch_size=3,
        micro_batch_size=3,
        checkpoint_every=3,
        sample_every=3,
    )
    upcycled_trainer = MCL4COETrainer(upcycle_cfg)
    assert upcycled_trainer.model.backbone.config.num_experts == 4
    up_m = upcycled_trainer.train()
    assert up_m["step"] == 3
    assert up_m["num_experts"] == 4
    assert len(up_m["expert_load"]) == 4


def test_forward_all_operators_and_ornith_self_improver() -> None:
    """Verify 9-operator joint pass (`forward_all_operators`), `OrnithSelfImprover`, and `<think>` compaction."""
    from mcl4_coe import (
        DecisionExample,
        OrnithSelfImprover,
        collate_decisions,
        extract_uta_episodes,
    )

    model = _tiny_model().train()
    examples = [
        DecisionExample(
            question_id="q1",
            kind="choice",
            context="#U$# A rabbit carries two red balls.\n#Q$# Which animal is present?",
            descriptions=("#T$# Grounded in premise.\n#A$# rabbit", "#T$# Contradicts premise.\n#A$# cat"),
            values=("rabbit", "cat"),
            target=0,
        ),
        DecisionExample(
            question_id="q2",
            kind="choice",
            context="#U$# Three servers process 12 requests each.\n#Q$# Total requests?",
            descriptions=("#T$# 3 + 12 = 15.\n#A$# 15", "#T$# 3 * 12 = 36.\n#A$# 36"),
            values=("15", "36"),
            target=1,
        ),
    ]
    batch = collate_decisions(examples, max_length=96)
    model.zero_grad(set_to_none=True)
    op_out = model.forward_all_operators(**batch)
    assert torch.isfinite(op_out.total_loss)
    assert torch.isfinite(op_out.deduction_loss)
    assert torch.isfinite(op_out.evaluation_loss)
    assert torch.isfinite(op_out.metacognition_loss)
    assert torch.isfinite(op_out.abstraction_loss)
    op_out.total_loss.backward()
    assert model.deduction_query_proj.weight.grad is not None
    assert model.deduction_prototypes.grad is not None
    assert model.meta_query_proj.weight.grad is not None
    assert model.assoc_alpha.grad is not None
    assert model.abstraction_proj.weight.grad is not None

    # Verify OrnithSelfImprover 3-stage self-improvement loop
    improver = OrnithSelfImprover(projection_dim=model.config.projection_dim, max_seq_len=96)
    model.zero_grad(set_to_none=True)
    ornith_out = improver.step_self_improvement(model)
    assert torch.isfinite(ornith_out.loss)
    assert 0.0 <= ornith_out.task_reward <= 2.0
    assert 0.0 <= ornith_out.harness_reward <= 1.0
    assert ornith_out.novelty_buffer_size >= 1
    assert ornith_out.scaffolds_cached >= 1
    ornith_out.loss.backward()

    # Verify verbose <think> monologue AND oversized post-</think> answer compaction fit inside seq_len=2048
    verbose_think = (
        "<think>"
        + ("Step derivation: let x_i = i^2 + 1.\n" * 80)
        + "</think>\n"
        + ("Detailed verification paragraph with UTF-8 math symbols (∀x ∈ ℝ, x² ≥ 0).\n" * 40)
    )
    eps = extract_uta_episodes(
        {
            "messages": [
                {"role": "user", "content": "Compute the sequence limit."},
                {"role": "assistant", "content": verbose_think},
            ]
        }
    )
    assert len(eps) == 1
    u, t, a = eps[0]
    tok_ids = ByteTokenizer().encode_masked_turn(u, t, a, seq_len=2048)
    assert len(tok_ids) <= 2048
    assert "\n...\n" not in t and t.endswith(".")

    # Verify Nemotron-Math-Proofs-v2 and hh-rlhf extraction schemas
    nemotron_eps = extract_uta_episodes(
        {
            "problem": "Prove that n + 0 = n in Lean 4.",
            "lean4_formalization": "theorem add_zero (n : Nat) : n + 0 = n := by rfl",
            "messages": [
                {"role": "user", "content": "Formalize and prove n + 0 = n."},
                {
                    "role": "assistant",
                    "reasoning_content": "By definition of Nat.add on the second argument, n + 0 reduces judgmentally to n.",
                    "content": "```lean4\ntheorem add_zero (n : Nat) : n + 0 = n := by rfl\n```",
                },
            ],
        }
    )
    assert len(nemotron_eps) == 1
    assert "Nat.add" in nemotron_eps[0][1]

    hh_eps = extract_uta_episodes(
        {
            "chosen": "\n\nHuman: How do I reverse a list in Python?\n\nAssistant: First consider slicing vs in-place reversal. You can use `my_list[::-1]` to return a reversed copy."
        }
    )
    assert len(hh_eps) == 1
    assert "reverse a list" in hh_eps[0][0]


def test_dream_rsi_simulator_and_mcl4_coe_integration(tmp_path: Path) -> None:
    """Verify Dream-RSI prefix-observable trees, zero-execution replay, Eq. 1 Pareto scoring,
    monotonic policy improvement, and end-to-end integration with MCL4-COE & Ornith-1.5.
    """
    from mcl4_coe import (
        DreamNode,
        DreamOutcome,
        DreamProgram,
        DreamRSIEngine,
        DreamWorld,
        OrnithSelfImprover,
        dream_online_rollout,
        dream_replay,
        improve_dream_policy,
        score_dream_policy,
    )

    # 1. Verify Program DSL parsing, canonical serialization, andCOE operator rank features
    prog_text = (
        "WIDTH 2\n"
        "ROOT 1\n"
        "RANK SCORE GAIN UTILITY CONFIDENCE\n"
        "STOP 0.9\n"
        "PATIENCE 3\n"
        "ROUNDS 4"
    )
    prog = DreamProgram.from_text(prog_text)
    assert prog.to_text() == prog_text
    assert prog.width == 2
    assert prog.rank == ("SCORE", "GAIN", "UTILITY", "CONFIDENCE")

    # 2. Verify online discovery tree construction and zero-execution historical replay
    call_counter = 0

    def _mock_discover(view, parent_id: int, node_id: int) -> DreamOutcome:
        nonlocal call_counter
        call_counter += 1
        # Branch 1 (first root child) starts at 0.30 and improves to 0.95 on refinement;
        # Branch 2 starts at 0.45 but plateaus at 0.46.
        if parent_id == 0:
            root_kids = sum(1 for n in view.nodes[1:] if n.parent == 0)
            sc = 0.30 if root_kids == 0 else 0.45
            util = 0.95 if root_kids == 0 else 0.40
        else:
            parent_node = next(n for n in view.nodes if n.id == parent_id)
            sc = 0.95 if parent_node.score == 0.30 else 0.46
            util = 0.98 if parent_node.score == 0.30 else 0.42
        return DreamOutcome(
            score=sc,
            observation={"utility": util, "confidence": util, "novelty": 0.5},
        )

    broad_policy = DreamProgram(width=2, root_every=1, rank=("SCORE", "GAIN"), stop=None, patience=0, rounds=4)
    online = dream_online_rollout(
        broad_policy,
        _mock_discover,
        world_id="w_test",
        root_observation={"task": "synth"},
        metadata={"test": True},
        max_rounds=4,
        workers=2,
        max_calls=6,
        rho=1.0,
    )
    assert online.agent_calls == call_counter
    assert len(online.world.nodes) == online.agent_calls + 1

    # Historical replay must perform ZERO discover calls while traversing the stored tree
    before_calls = call_counter
    rep = dream_replay(
        online.world,
        prog,
        rho=1.0,
        max_rounds=4,
        workers=2,
        max_calls=6,
    )
    assert call_counter == before_calls  # Zero real executions during replay!
    assert rep.best_score == 0.95

    # 3. Verify Eq. 1 Pareto hypervolume scoring & monotonic policy improvement (pi^{t+1} >= pi^t)
    incumbent = DreamProgram(width=1, root_every=1, rank=("SHALLOW",), stop=None, patience=0, rounds=4)
    better_candidate = DreamProgram(width=2, root_every=2, rank=("UTILITY", "GAIN", "SCORE"), stop=0.9, patience=2, rounds=4)

    proposals = [
        {"text": "INVALID POLICY SYNTAX", "finished": True},
        {"text": better_candidate.to_text(), "finished": True},
    ]

    def _propose(_inc, _fb, rev: int):
        return proposals[rev - 1]

    imp = improve_dream_policy(
        [online.world],
        incumbent,
        _propose,
        revisions=2,
        max_rounds=4,
        workers=2,
        max_calls=6,
    )
    assert imp["final_score"] >= imp["initial_score"]
    assert imp["accepted_revision"] == 2
    assert imp["program_text"] == better_candidate.to_text()

    # 4. Verify end-to-end DreamRSIEngine + OrnithSelfImprover on MCL4-COE
    model = _tiny_model().train()
    dream_dir = tmp_path / "dream_out"
    engine = DreamRSIEngine(
        max_rounds=2,
        workers=2,
        max_calls=3,
        revisions_per_cycle=2,
        out_dir=dream_dir,
    )
    improver = OrnithSelfImprover(
        projection_dim=model.config.projection_dim,
        max_seq_len=96,
        enable_dream_rsi=True,
        dream_every=1,
        dream_engine=engine,
    )
    model.zero_grad(set_to_none=True)
    out1 = improver.step_self_improvement(model)
    assert out1.dream_cycle == 1
    assert out1.dream_worlds_archived == 1
    assert out1.dream_replays_dreamed > 0
    assert "WIDTH " in out1.dream_active_policy
    assert (dream_dir / "dream_state.json").is_file()
    assert (dream_dir / "dream_policy.json").is_file()
    assert (dream_dir / "dream_revisions.jsonl").is_file()
    out1.loss.backward()


def test_context_memory_hierarchy_and_atxy_router(tmp_path) -> None:
    """Verify Tier 2 (TurboQuant), Tier 3 (ContextLMCompressor), and Tier 4 (A.TXY + COEAddressRouter)."""
    from mcl4_coe import (
        ATXYAddress,
        ATXYFactorizedEmbedding,
        ATXYPageStore,
        ATXYPointerExecutor,
        ATXYRecord,
        COEAddressRouter,
        ContextLMCompressor,
        compress_encoded_state,
        index_python_repo_to_atxy,
        parse_atxy,
        turboquant_compress,
        turboquant_inner_product,
        unpack_atxy,
    )

    model = _tiny_model().train()
    x = torch.randint(0, 256, (2, 48), dtype=torch.long)
    valid = torch.ones_like(x, dtype=torch.bool)

    # 1. Tier 2: TurboQuant KV / EncodedState compression
    enc = model.backbone.encode_state(x, valid)
    comp_state = compress_encoded_state(enc)
    assert comp_state.compressed_hidden.compression_ratio > 3.5
    decomp_enc = comp_state.decompress()
    cos = torch.nn.functional.cosine_similarity(
        enc.hidden.flatten(1), decomp_enc.hidden.flatten(1), dim=-1
    )
    assert float(cos.mean().item()) > 0.85

    q = torch.randn(2, 6, 32)
    k = torch.randn(2, 16, 32)
    ck = turboquant_compress(k)
    approx_scores = turboquant_inner_product(q, ck)
    exact_scores = torch.matmul(q, k.transpose(-1, -2))
    score_corr = torch.nn.functional.cosine_similarity(
        approx_scores.flatten(1), exact_scores.flatten(1), dim=-1
    )
    assert float(score_corr.mean().item()) > 0.85

    # 2. Tier 3: ContextLMCompressor multi-chunk recurrent state compression
    ctx_lm = ContextLMCompressor(
        d_model=model.backbone.config.d_model,
        chunk_size=16,
        gist_tokens_per_chunk=4,
    )
    gist_enc = ctx_lm(model.backbone, x, valid)
    assert gist_enc.hidden.shape == (2, 12, model.backbone.config.d_model)
    assert gist_enc.valid.shape == (2, 12)

    # 3. Tier 4: A.TXY Virtual Memory PageStore, N5 WAL, Factorized Embedding, Pointer Executor & COE Router
    addr = parse_atxy("1.G0B")
    assert str(addr) == "1.G0B"
    assert unpack_atxy(addr.packed) == addr

    fact_emb = ATXYFactorizedEmbedding(d_model=model.backbone.config.d_model)
    emb_out = fact_emb.encode_addresses(["1.G0B", "2.010"])
    assert emb_out.shape == (2, model.backbone.config.d_model)

    repo_root = Path(__file__).resolve().parents[1]
    store = index_python_repo_to_atxy(repo_root, subsystems=("src",))
    assert len(store) > 4

    # Verify N5 append-only WAL persistence + CRC32 verification
    wal_path = tmp_path / "atxy_wal.log"
    first_rec = store.records()[0]
    wal_rec = store.commit_write(
        ATXYRecord(
            address=first_rec.address,
            kind=first_rec.kind,
            payload="wal_verified_node",
            summary="Updated via N5 WAL",
            up=first_rec.up,
            down=first_rec.down,
            fields={"lineno": 42},
        ),
        wal_path=wal_path,
    )
    assert wal_rec.payload == "wal_verified_node"
    assert wal_rec.version == first_rec.version + 1
    store_reloaded = ATXYPageStore()
    replayed = store_reloaded.load_wal(wal_path)
    assert replayed == 1
    assert store_reloaded.lookup(first_rec.address).payload == "wal_verified_node"

    # Verify N6 single-register pointer executor
    executor = ATXYPointerExecutor(store)
    kind_out, val_out = executor.run(
        [
            ("READ", first_rec.address, None),
            ("PROJECT", first_rec.address, "lineno"),
            ("FIND", None, {"plane": 1}),
            ("COUNT", "REG", None),
        ]
    )
    assert kind_out == "scalar"
    assert isinstance(val_out, int) and val_out > 0
    assert len(executor.audit) == 4

    # Verify N3b-primary natural-language -> A.TXY routing via COE Association + Decision
    router = COEAddressRouter(store, max_seq_len=128)
    route_out = router.route_query(
        model,
        "Locate wal_verified_node and module definitions in mcl4_coe",
        top_k_candidates=4,
    )
    assert route_out["address"] in route_out["candidates"]
    assert len(route_out["candidates"]) <= 4


def test_verify_python_rollout_and_portable_shard_healing(tmp_path) -> None:
    """Verify sandboxed Python code execution reward (`verify_python_rollout`) and portable shard self-healing."""
    import shutil
    from mcl4_coe import (
        MultiTrackMixtureDataset,
        ensure_sharded_tracks_ready,
        verify_python_rollout,
    )

    harness_h = "assert solve(3, 4) == 14 and solve(0, 5) == 2"

    good_rollout = (
        "#T$#We compute a * b + 2 and verify both assertions.#A$#"
        "def solve(a, b): return a * b + 2"
    )
    assert verify_python_rollout(good_rollout, harness_h) == 1.0

    # Missing #T$# scratchpad must return 0.0
    no_thought_rollout = "#A$#def solve(a, b): return a * b + 2"
    assert verify_python_rollout(no_thought_rollout, harness_h) == 0.0

    # Constant boundary-hack stub must be rejected by anti-shortcut validator or second assertion
    stub_rollout = "#T$#Constant stub return 14.#A$#def solve(a, b): return 14"
    assert verify_python_rollout(stub_rollout, harness_h) == 0.0

    # Verify ensure_sharded_tracks_ready rebuilds missing .bin shards automatically
    shards_copy = tmp_path / "data_shards"
    repo_shards = Path(__file__).resolve().parents[1] / "data_shards"
    for track in ("pretrain_lm", "uta_trace", "typed_decision"):
        (shards_copy / track).mkdir(parents=True, exist_ok=True)
        src_receipt = repo_shards / track / "index_receipt.json"
        if src_receipt.is_file():
            shutil.copyfile(src_receipt, shards_copy / track / "index_receipt.json")

    healed_receipts = ensure_sharded_tracks_ready(shards_copy, seq_len=128)
    assert set(healed_receipts.keys()) == {"pretrain_lm", "uta_trace", "typed_decision"}
    ds = MultiTrackMixtureDataset(healed_receipts)
    assert len(ds) > 0
    assert (shards_copy / "typed_decision" / "decision_examples_cache.pt").is_file()


def test_nemotron_core_schema_extractors() -> None:
    """Verify Nemotron-Core schema extractors across pretrain_lm, uta_trace, and typed_decision."""
    from mcl4_coe import (
        extract_pretrain_text,
        extract_typed_decision_uta,
        extract_uta_episodes,
        parse_mcqa_decision_fields,
    )

    # 1. Nemotron-MIND (`Text` capital-T column)
    mind_row = {
        "Document ID": "OWM-1",
        "Text": (
            "Participant 1: Let us derive the eigenvalues of a real symmetric 2x2 matrix step by step.\n"
            "Participant 2: Because the matrix is symmetric, the discriminant of its characteristic "
            "polynomial is guaranteed to be non-negative, yielding real eigenvalues."
        ),
    }
    text_out, rej = extract_pretrain_text(mind_row)
    assert not rej and text_out is not None and "eigenvalues" in text_out

    # 2. Redacted user prompt in Nemotron-Post-Training-Dataset-v1 (`content == "-"`) must be rejected
    redacted_row = {
        "messages": [
            {"role": "user", "content": "-"},
            {"role": "assistant", "content": "<think>Some thought</think>Some answer"},
        ]
    }
    assert extract_uta_episodes(redacted_row) == []

    # 3. Nemotron-SFT-Code (`text` formatted as `input: ... output: ...`)
    sft_row = {
        "text": (
            "input: Write a Python function to reverse a list.\n"
            "output: First, slice the list with step -1.\n\ndef rev(xs):\n    return xs[::-1]"
        )
    }
    eps = extract_uta_episodes(sft_row)
    assert len(eps) == 1
    assert "reverse a list" in eps[0][0]
    assert "xs[::-1]" in eps[0][2]

    # 4. Nemotron-RL-knowledge-mcqa
    rl_mcqa_row = {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": (
                        "Answer the following multiple choice question.\n\n"
                        "Which mechanism guarantees O(T) complexity in MCL4-COE?\n"
                        "A: Full quadratic attention\n"
                        "B: Markov Local-4 Router\n"
                        "C: Unbounded recursion\n"
                        "D: Dense all-to-all MLP"
                    ),
                }
            ]
        },
        "expected_answer": "B",
        "options": [
            {"A": "Full quadratic attention", "B": None, "C": None, "D": None},
            {"A": None, "B": "Markov Local-4 Router", "C": None, "D": None},
            {"A": None, "B": None, "C": "Unbounded recursion", "D": None},
            {"A": None, "B": None, "C": None, "D": "Dense all-to-all MLP"},
        ],
    }
    parsed_rl = parse_mcqa_decision_fields(rl_mcqa_row)
    assert parsed_rl is not None
    assert parsed_rl[4] == 1
    assert parsed_rl[3][1] == "Markov Local-4 Router"

    # 5. Nemotron-CrossThink
    crossthink_row = {
        "meta_data": {
            "question": (
                "Which transform precedes Lloyd-Max scalar quantization in TurboQuant?\n"
                "A: Fast Fourier Transform\n"
                "B: Fast Walsh-Hadamard Transform\n"
                "C: Discrete Cosine Transform"
            )
        },
        "reward_model": {"ground_truth": "B: Fast Walsh-Hadamard Transform"},
    }
    parsed_ct = parse_mcqa_decision_fields(crossthink_row)
    assert parsed_ct is not None
    assert parsed_ct[4] == 1

    # 6. Nemotron-Pretraining-Specialized-v1.1 (`Nemotron-Pretraining-Multiple-Choice`)
    spec_mc_row = {
        "text": (
            "In the A.TXY addressing system, how many total addresses exist across 6 planes of 6x6x6 cubes?\n"
            "A. 216\n"
            "B. 1296\n"
            "C. 4096\n"
            "D. 65536\n"
            "Answer: B\n"
            "Because each of the 6 planes contains 6*6*6 = 216 addresses, 6 * 216 = 1296 total addresses."
        )
    }
    uta_mc = extract_typed_decision_uta(spec_mc_row)
    assert uta_mc is not None
    assert "1296 total addresses" in uta_mc[1]
    assert "[1] 1296" in uta_mc[2]


def test_antidoom_ftpo_and_housekeeping_fixes() -> None:
    """Verify LiquidAI Antidoom FTPO loop detection & loss, <think> tag sanitization, and boilerplate removal."""
    from mcl4_coe import (
        compute_antidoom_ftpo_loss,
        detect_doom_loop,
        extract_typed_decision_uta,
        sanitize_trace_markers,
    )

    # 1. Strip <think> and </think> tags
    cleaned = sanitize_trace_markers("<think> First derive the recurrence. </think> Therefore x = 4.")
    assert "<think>" not in cleaned and "</think>" not in cleaned

    # 2. Verify no synthetic "state evidence" boilerplate in extract_typed_decision_uta
    no_rat_row = {
        "state": "Markov Local-4 routes each token to 4 local causal windows.",
        "kind": "choice",
        "question": "Which complexity does Markov Local-4 achieve?",
        "options": ["O(T^2)", "O(T)", "O(T^3)", "O(2^T)"],
        "target": 1,
    }
    uta_no_rat = extract_typed_decision_uta(no_rat_row, require_rationale=False)
    assert uta_no_rat is not None
    assert "state evidence" not in uta_no_rat[1]
    assert extract_typed_decision_uta(no_rat_row, require_rationale=True) is None

    # 3. Verify Antidoom inner repetition detection and FTPO preference loss
    repeated_bytes = list(b"Compute step:" + b" state evidence," * 5)
    loop_hit = detect_doom_loop(repeated_bytes, min_period=4, min_repeats=3)
    assert loop_hit is not None
    start_idx, rejected_tok, period = loop_hit
    assert period == len(b" state evidence,")
    assert rejected_tok == repeated_bytes[start_idx]

    logits = torch.zeros(1, 260, requires_grad=True)
    ref_logits = logits.detach().clone()
    rej_t = torch.tensor([rejected_tok], dtype=torch.long)
    chosen_t = torch.tensor([[ord("a"), ord("b"), ord("c"), ord("d")]], dtype=torch.long)
    ftpo_loss, chosen_win = compute_antidoom_ftpo_loss(
        logits,
        ref_logits,
        rej_t,
        chosen_t,
        beta=0.5,
        target_chosen_win=1.0,
    )
    assert ftpo_loss.requires_grad
    ftpo_loss.backward()
    assert logits.grad is not None
    assert float(logits.grad[0, rejected_tok].item()) > 0.0  # gradient pushes rejected token logit down
    assert 0.0 <= chosen_win <= 1.0

    # 4. Verify Exact-Ratio Twin-Gated Mitosis preserves forward CE loss & serializes mitotic_pairs
    from mcl4_coe.trainer import compute_mitotic_schedule, upcycle_state_dict_experts

    torch.manual_seed(7)
    cfg4 = MCL4CEDConfig(
        d_model=64,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ple_dim=16,
        expert_hidden_dim=128,
        num_experts=4,
        top_k=2,
        max_seq_len=64,
    )
    m4 = CognitiveOperatorEnsemble(MCL4CEDBackbone(cfg4), COEConfig(projection_dim=16))
    m4.eval()
    inp = torch.randint(4, 120, (2, 32))
    lbl = torch.randint(4, 120, (2, 32))
    with torch.no_grad():
        ce4 = float(m4.forward_inference(inp, targets=lbl).ce_loss.item())

    loads4 = [0.10, 0.45, 0.35, 0.10]
    pairs_4to6 = compute_mitotic_schedule(4, 6, expert_loads=loads4)
    assert pairs_4to6 == [(1, 4), (2, 5)]
    sd6 = upcycle_state_dict_experts(m4.state_dict(), 4, 6, expert_loads=loads4, seed=42)
    cfg6 = MCL4CEDConfig(
        d_model=64,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ple_dim=16,
        expert_hidden_dim=128,
        num_experts=6,
        top_k=2,
        max_seq_len=64,
    )
    m6 = CognitiveOperatorEnsemble(MCL4CEDBackbone(cfg6), COEConfig(projection_dim=16))
    m6.load_state_dict(sd6, strict=True)
    assert m6.backbone.encoder_layers[0].moe.mitotic_pairs == [(1, 4), (2, 5)]
    m6.eval()
    with torch.no_grad():
        ce6 = float(m6.forward_inference(inp, targets=lbl).ce_loss.item())
    assert abs(ce6 - ce4) < 2e-3


def test_morphcode_16k_tokenizer_and_backbone(tmp_path) -> None:
    """Verify MorphCode-16K (`vocab_size=16384`) lossless round-trip, morpheme splitting, factorized PLE, and shard transcoding."""
    from mcl4_coe import (
        MORPHCODE_16K_VOCAB_SIZE,
        MorphCodeTokenizer,
        MultiPassController,
        ShardedTokenBlockDataset,
        get_tokenizer,
        transcode_byte_shards_to_morphcode,
    )
    from mcl4_coe.pipeline import BlockAlignedShardWriter

    mt = get_tokenizer(MORPHCODE_16K_VOCAB_SIZE)
    assert isinstance(mt, MorphCodeTokenizer)
    assert mt.vocab_size == 16384
    assert (mt.pad_id, mt.bos_id, mt.eos_id, mt.unk_id) == (256, 257, 258, 259)

    # 1. Atomic control markers + morpheme decomposition + lossless round-trip
    uta_sample = (
        "#U$# How do plants use capillary action to transport water?\n"
        "#T$# Plants use adhesion and cohesion in narrow xylem tubes.\n"
        "#A$# Capillary action pulls water upward against gravity."
    )
    ids = mt.encode(uta_sample)
    assert mt.decode(ids) == uta_sample
    assert mt.compression_ratio(uta_sample) > 4.0
    tokens = mt.tokenize_to_strings(uta_sample)
    assert tokens[0] == "#U$#"
    assert "\n#T$#" in tokens and "\n#A$#" in tokens
    # Verify inflectional morpheme splitting (`plant + s`, `adhes + ion`, `cohes + ion`)
    assert " plant" in tokens and "s" in tokens
    assert " adhes" in tokens and "ion" in tokens

    # 2. Single-digit splitting for arithmetic and Python code operators
    math_sample = "Solve 3 * x + 5 = 20 for x == 5 using \\frac{15}{3}."
    math_ids = mt.encode(math_sample)
    assert mt.decode(math_ids) == math_sample
    math_toks = mt.tokenize_to_strings(math_sample)
    assert "2" in math_toks and "0" in math_toks
    assert " ==" in math_toks and "\\frac{" in math_toks

    # 3. Arbitrary UTF-8 byte fallback (emoji + accented characters)
    utf8_sample = "Café Δx ≈ 0.001 🚀"
    assert mt.decode(mt.encode(utf8_sample)) == utf8_sample

    # 4. Factorized PerLayerEmbedding + tied lm_head on 16,384 vocab
    cfg16k = MCL4CEDConfig(
        vocab_size=16384,
        d_model=64,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ple_dim=16,
        expert_hidden_dim=128,
        num_experts=4,
        top_k=2,
        max_seq_len=64,
    )
    m16k = CognitiveOperatorEnsemble(MCL4CEDBackbone(cfg16k), COEConfig(projection_dim=16))
    assert m16k.backbone.lm_head.weight is m16k.backbone.token_embedding.weight
    assert m16k.backbone.encoder_layers[0].ple.factorized_vocab

    inp16k = torch.tensor([ids[:32]], dtype=torch.long)
    lbl16k = torch.tensor([ids[1:33]], dtype=torch.long)
    out16k = m16k.forward_inference(inp16k, targets=lbl16k)
    assert out16k.logits.shape == (1, 32, 16384)
    assert out16k.ce_loss is not None and float(out16k.ce_loss.item()) > 0.0
    assert mt.count_utf8_bytes(lbl16k) == len(mt.decode(ids[1:33]).encode("utf-8"))

    ctrl16k = MultiPassController(m16k)
    coe_gen = ctrl16k.generate_uta_with_coe(
        "What is 2 + 2?",
        max_new_tokens=16,
        candidate_specs=((1, 0.0), (2, 0.35)),
    )
    assert (
        "#U$#" in coe_gen.text
        and "#T$#" in coe_gen.text
        and "#A$#" in coe_gen.text
    )

    # 5. Transcode uint16 byte shards with 0x8000 prompt masking to MorphCode-16K shards
    src_shards = tmp_path / "byte_shards"
    dst_shards = tmp_path / "morph16k_shards"
    writer = BlockAlignedShardWriter(
        src_shards / "uta_trace",
        track="uta_trace",
        seq_len=192,
        max_blocks_per_shard=64,
        vocab_size=260,
    )
    for i in range(6):
        assert writer.add_uta_episode(
            f"Question {i}: Why do objects fall?",
            "Gravity accelerates masses toward each other.",
            "Gravitational attraction pulls objects downward.",
        )
    writer.flush_shard()
    receipts_16k = transcode_byte_shards_to_morphcode(src_shards, dst_shards, seq_len=64)
    assert "uta_trace" in receipts_16k
    ds16k = ShardedTokenBlockDataset(
        dst_shards / "uta_trace" / "index_receipt.json",
        track="uta_trace",
        split="train",
    )
    assert len(ds16k) > 0
    item0 = ds16k[0]
    assert int(item0["input_ids"].max().item()) < 16384
    assert int((item0["labels"] == -100).sum().item()) > 0


def test_context_memory_turboquant_atxy_and_codebase_warmup(tmp_path: Path) -> None:
    """Verify TurboQuant 4-bit page compression, A.TXY virtual memory, and CodebaseWarmupEngine RL warmup + exact reset."""
    from mcl4_coe import (
        CodebaseWarmupEngine,
        compress_encoded_state,
    )

    # 1. Verify TurboQuant 4-bit PolarQuant + QJL compression and EncodedState round-trip
    model = _tiny_model()
    sample_ids = torch.randint(4, 120, (1, 48), dtype=torch.long)
    with torch.no_grad():
        enc = model.backbone.encode_state(sample_ids)
    comp = compress_encoded_state(enc)
    assert comp.compression_ratio > 5.0
    recon = comp.decompress()
    cos = float(
        torch.nn.functional.cosine_similarity(
            enc.hidden.reshape(-1, 64).float(),
            recon.hidden.reshape(-1, 64).float(),
            dim=-1,
        )
        .mean()
        .item()
    )
    assert cos > 0.95

    # 2. Create a mini Python codebase in tmp_path and run CodebaseWarmupEngine
    pkg = tmp_path / "mini_pkg"
    pkg.mkdir()
    (pkg / "math_ops.py").write_text(
        '"""Arithmetic and normalization utilities."""\n\n'
        "def normalize_scores(scores: list[float], eps: float = 1e-5) -> list[float]:\n"
        '    """Normalize a list of float scores by their L2 norm."""\n'
        "    norm_sq = sum(x * x for x in scores)\n"
        "    if norm_sq <= 0.0:\n"
        "        return [0.0 for _ in scores]\n"
        "    scale = (norm_sq + eps) ** 0.5\n"
        "    return [x / scale for x in scores]\n\n"
        "def discounted_return(rewards: list[float], gamma: float = 0.99) -> float:\n"
        '    """Compute discounted cumulative return across trajectory rewards."""\n'
        "    total = 0.0\n"
        "    for r in reversed(rewards):\n"
        "        if r != 0.0:\n"
        "            total = r + gamma * total\n"
        "    return total\n",
        encoding="utf-8",
    )
    (pkg / "router.py").write_text(
        '"""Query router and page cache."""\n\n'
        "from .math_ops import normalize_scores, discounted_return\n\n"
        "def select_best_page(raw_scores: list[float], min_prob: float = 0.1) -> int:\n"
        '    """Return index of highest normalized page score above threshold."""\n'
        "    probs = normalize_scores(raw_scores)\n"
        "    best_idx = max(range(len(probs)), key=lambda i: probs[i])\n"
        "    if probs[best_idx] < min_prob:\n"
        "        return 0\n"
        "    return best_idx\n\n"
        "def score_trajectory_pages(rewards: list[float], bonus: float = 1.0) -> float:\n"
        '    """Score trajectory pages with discounted cumulative return plus bonus."""\n'
        "    base_ret = discounted_return(rewards)\n"
        "    if base_ret > 0.0:\n"
        "        return base_ret + bonus\n"
        "    return base_ret\n",
        encoding="utf-8",
    )

    engine = CodebaseWarmupEngine(
        subsystems=("mini_pkg",),
        max_seq_len=64,
        num_route_options=4,
    )
    session = engine.index_repository(model, tmp_path, max_tasks_per_family=8)
    assert session.total_repo_files == 2
    assert session.total_repo_symbols >= 4
    assert session.compression_ratio > 5.0
    assert len(session.train_route_tasks) >= 4
    assert len(session.train_verify_tasks) >= 2

    report = engine.warmup_session(model, session, steps=4, lr=2e-3, batch_size=2)
    assert report.warmup_steps == 4
    assert report.step_losses[-1] <= report.step_losses[0]
    assert len(report.step_losses) == 4

    q_res = engine.query_codebase(
        model,
        session,
        "Where is normalize_scores defined to normalize float scores by L2 norm?",
        top_k_candidates=4,
    )
    assert "symbol" in q_res and "address" in q_res

    # 3. Verify Tier-3 ContextLMCompressor multi-file synthesis & end-to-end codebase-grounded generation
    mfile = engine.synthesize_multifile_context(
        model,
        session,
        "How does select_best_page call normalize_scores across modules?",
        max_files=2,
        top_k_candidates=4,
    )
    assert mfile.num_files_folded == 2
    assert mfile.compressed_gist_tokens == 12
    assert mfile.compressed_state.hidden.shape == (1, 12, 64)
    assert mfile.multifile_latent_prior.shape == (1, 64)

    ctrl = MultiPassController(model)
    rag_out = ctrl.generate_uta_with_codebase_session(
        "How does select_best_page call normalize_scores?",
        engine,
        session,
        max_files=2,
        top_k_candidates=4,
        max_new_tokens=20,
    )
    assert "#T$#" in rag_out.text and "#A$#" in rag_out.text
    assert rag_out.num_files_folded == 2
    assert rag_out.extra_prompt_tokens == 0
    assert len(rag_out.candidates) == 9

    # 4. Verify Incremental Dirty-Page Refresh (modifying an existing file + adding a new symbol + WAL framing)
    assert engine.detect_dirty_files(session) == []
    (pkg / "router.py").write_text(
        'from .math_ops import normalize_scores\n\n'
        'def select_best_page(scores: list[float]) -> int:\n'
        '    """Route query to the highest normalized page index with tie-breaking."""\n'
        '    normed = normalize_scores(scores)\n'
        '    return int(normed.index(max(normed)))\n\n'
        'def evict_stale_page(ages: list[int], max_age: int = 64) -> list[int]:\n'
        '    """Evict cached TurboQuant pages older than max_age ticks."""\n'
        '    return [idx for idx, a in enumerate(ages) if a > max_age]\n',
        encoding="utf-8",
    )
    detected_dirty = engine.detect_dirty_files(session)
    assert detected_dirty == ["mini_pkg/router.py"]

    wal_file = tmp_path / "session_dirty.wal"
    refresh_rep = engine.refresh_dirty_files(
        model,
        session,
        micro_warmup_steps=2,
        wal_path=wal_file,
    )
    assert refresh_rep.dirty_files == ("mini_pkg/router.py",)
    assert "mini_pkg/router.py:evict_stale_page" in refresh_rep.added_symbols
    assert "mini_pkg/router.py:select_best_page" in refresh_rep.modified_symbols
    assert refresh_rep.recompressed_pages == 1
    assert refresh_rep.wal_frames_appended >= 3
    assert wal_file.is_file() and wal_file.stat().st_size > 0
    assert engine.detect_dirty_files(session) == []
    assert session.store.resolve("mini_pkg/router.py:evict_stale_page") is not None

    # 5. Verify exact <1ms restoration of base weights (0.0 parameter drift)
    session.restore_base_weights(model)
    for name, param in model.named_parameters():
        if name in session.base_operator_state:
            drift = float((param.detach().cpu() - session.base_operator_state[name]).abs().max().item())
            assert drift == 0.0


def test_recurrent_depth_trajectory_pondering_and_adaptive_halting() -> None:
    """Verify Recurrent Depth trajectory pondering (`step_states`), `recurrent_kv`, `margin_halt_threshold`, 0-ULP candidate permutation invariance, and adaptive entropy halting."""
    model = _tiny_model().eval()
    x = torch.randint(4, 120, (2, 24), dtype=torch.long)

    # 1. Verify step_states trajectory length and entropy-gated residual recurrent depth + recurrent_kv
    out_r1 = model.forward_inference(x, recurrence_steps=1)
    assert len(out_r1.step_states) == 1
    out_r3 = model.forward_inference(
        x,
        recurrence_steps=3,
        recurrent_kv=True,
        depth_ponder=True,
    )
    assert len(out_r3.step_states) == 3
    assert out_r3.logits.shape == out_r1.logits.shape
    assert not torch.allclose(out_r3.logits, out_r1.logits)

    # 2. Verify 0-ULP bitwise candidate permutation invariance under depth_ponder and margin_halt_threshold
    ctx_ids = torch.randint(4, 120, (2, 20), dtype=torch.long)
    ctx_lens = torch.tensor([20, 16], dtype=torch.long)
    opt_ids = torch.randint(4, 120, (2, 4, 18), dtype=torch.long)
    opt_lens = torch.tensor([[18, 15, 12, 14], [14, 18, 11, 16]], dtype=torch.long)
    targets = torch.tensor([1, 2], dtype=torch.long)

    perm_idx = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    inv_idx = torch.argsort(perm_idx)
    opt_ids_perm = opt_ids[:, perm_idx, :]
    opt_lens_perm = opt_lens[:, perm_idx]

    dec_ponder = model.forward_decision(
        ctx_ids,
        ctx_lens,
        opt_ids,
        opt_lens,
        recurrence_steps=2,
        depth_ponder=True,
        margin_halt_threshold=0.35,
    )
    dec_ponder_perm = model.forward_decision(
        ctx_ids,
        ctx_lens,
        opt_ids_perm,
        opt_lens_perm,
        recurrence_steps=2,
        depth_ponder=True,
        margin_halt_threshold=0.35,
    )
    assert torch.equal(dec_ponder.similarities, dec_ponder_perm.similarities[:, inv_idx])
    assert torch.equal(dec_ponder.logits, dec_ponder_perm.logits[:, inv_idx])
    assert torch.allclose(
        dec_ponder.probabilities, dec_ponder_perm.probabilities[:, inv_idx], atol=1e-7
    )

    all_op_ponder = model.forward_all_operators(
        ctx_ids,
        ctx_lens,
        opt_ids,
        opt_lens,
        targets,
        recurrence_steps=2,
        depth_ponder=True,
        margin_halt_threshold=0.35,
    )
    assert torch.isfinite(all_op_ponder.total_loss)

    # 3. Verify adaptive_recurrent_depth in MultiPassController.generate_uta
    ctrl = MultiPassController(model)
    trace_adapt = ctrl.generate_uta(
        "Explain recurrent depth pondering.",
        max_new_tokens=20,
        adaptive_recurrent_depth=True,
        max_recurrence_steps=3,
        recurrent_entropy_threshold=1.5,
        recurrent_kv=True,
    )
    assert "#U$#" in trace_adapt and "#T$#" in trace_adapt and "#A$#" in trace_adapt


def test_pre_scaling_housekeeping_gist_conditioning_and_repo_uta_traces(tmp_path: Path) -> None:
    from mcl4_coe import TrainingRunConfig, synthesize_repository_uta_traces

    torch.manual_seed(1234)
    # 1. Verify projection_dim auto-scaling in TrainingRunConfig
    assert TrainingRunConfig(d_model=128).projection_dim == 32
    assert TrainingRunConfig(d_model=384).projection_dim == 96
    assert TrainingRunConfig(d_model=768).projection_dim == 192
    assert TrainingRunConfig(d_model=768, projection_dim=128).projection_dim == 128

    # 2. Verify end-to-end gradient flow into gist_queries, gist_norm, and abstraction_proj
    #    plus deep recurrent trajectory supervision
    b_cfg = MCL4CEDConfig(
        vocab_size=260,
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        ple_dim=16,
        expert_hidden_dim=128,
        num_experts=4,
        top_k=2,
        max_seq_len=128,
        dropout=0.0,
    )
    c_cfg = COEConfig(projection_dim=16, gist_tokens_per_chunk=4, gist_chunk_size=16)
    coe = CognitiveOperatorEnsemble(MCL4CEDBackbone(b_cfg), c_cfg)

    # Verify legacy state_dict without gist_queries / gist_norm loads cleanly under strict=True
    legacy_sd = {
        k: v for k, v in coe.state_dict().items() if not (k == "gist_queries" or k.startswith("gist_norm."))
    }
    coe.load_state_dict(legacy_sd, strict=True)

    coe.train()
    coe.zero_grad(set_to_none=True)
    x = torch.randint(4, 240, (2, 48), dtype=torch.long)
    loss_dict = compute_joint_coe_loss(
        coe,
        lm_batch=(x, x.clone()),
        recurrence_steps=2,
        recurrent_kv=True,
        depth_ponder=True,
        deep_supervision_weight=0.25,
        condition_on_prefix_gist=True,
        latent_prior_scale=0.35,
    )
    total_loss = loss_dict.total_loss
    assert torch.isfinite(total_loss)
    total_loss.backward()

    assert coe.gist_queries.grad is not None
    assert float(coe.gist_queries.grad.abs().sum().item()) > 0.0
    assert coe.gist_norm.weight.grad is not None
    assert float(coe.gist_norm.weight.grad.abs().sum().item()) > 0.0
    assert coe.abstraction_proj.weight.grad is not None
    assert float(coe.abstraction_proj.weight.grad.abs().sum().item()) > 0.0

    # 3. Verify strict prefix causality: mutating token (prefix_len - 1) must NOT leak backward
    #    into prefix positions 0 .. prefix_len - 2 even when condition_on_prefix_gist=True
    coe.eval()
    prefix_len = max(8, min(48 // 3, 48 - 8))  # 16
    x_a = x[:1].clone()
    x_b = x[:1].clone()
    x_b[0, prefix_len - 1] = (x_b[0, prefix_len - 1] + 17) % 200 + 10
    with torch.no_grad():
        out_a = coe.forward_inference(
            x_a,
            recurrence_steps=2,
            recurrent_kv=True,
            depth_ponder=True,
            condition_on_prefix_gist=True,
        )
        out_b = coe.forward_inference(
            x_b,
            recurrence_steps=2,
            recurrent_kv=True,
            depth_ponder=True,
            condition_on_prefix_gist=True,
        )
    # Prefix tokens 0 .. prefix_len - 2 must have 0-diff logits (zero lookahead leakage)
    max_prefix_leak = float((out_a.logits[:, : prefix_len - 1] - out_b.logits[:, : prefix_len - 1]).abs().max().item())
    assert max_prefix_leak < 1e-5, f"Lookahead leakage detected inside prefix: {max_prefix_leak}"
    # Continuation tokens >= prefix_len must differ because gist_state and latent_prior encode prefix_len - 1
    cont_diff = float((out_a.logits[:, prefix_len:] - out_b.logits[:, prefix_len:]).abs().max().item())
    assert cont_diff > 1e-4

    # 4. Verify AST-grounded repository UTA trace synthesis
    sample_repo = tmp_path / "repo"
    sample_repo.mkdir(parents=True, exist_ok=True)
    (sample_repo / "module_a.py").write_text(
        'class SampleRouter:\n'
        '    """Routes tokens across local context pages."""\n'
        '    def route_pages(self, query_vec: list[float], top_k: int = 2) -> list[int]:\n'
        '        """Return top-k page indices."""\n'
        '        if top_k <= 0:\n'
        '            return []\n'
        '        return list(range(min(len(query_vec), top_k)))\n',
        encoding="utf-8",
    )
    traces = synthesize_repository_uta_traces(sample_repo, max_episodes=16)
    assert len(traces) >= 2
    assert any("SampleRouter" in u or "route_pages" in u for u, _, _ in traces)

    # 5. Verify `codeparrot/github-code` streaming normalization, path sieve, and pretrain/UTA extraction
    from mcl4_coe import (
        extract_pretrain_text,
        extract_uta_episodes,
        infer_github_code_language,
        normalize_github_code_record,
    )

    assert infer_github_code_language("app/mods/mod190.js") == "JavaScript"
    assert infer_github_code_language("src/mcl4_coe/backbone.py") == "Python"
    raw_parquet_row = {
        "content": (
            "import mod189 from './mod189';\n"
            "export function computeNextValue(seed) {\n"
            "  const value = mod189 + seed + 1;\n"
            "  if (value <= 0) {\n"
            "    return 0;\n"
            "  }\n"
            "  return value;\n"
            "}\n"
            "export default computeNextValue;\n"
        ),
        "repo_name": "MirekSz/webpack-es6-ts",
        "path": "app/mods/mod190.js",
        "license": "isc",
        "size": "214",
    }
    norm_row = normalize_github_code_record(raw_parquet_row)
    assert norm_row["language"] == "JavaScript"
    assert norm_row["size"] == 214
    pretrain_txt, rej = extract_pretrain_text(norm_row)
    assert not rej and pretrain_txt is not None
    assert "# Repo: MirekSz/webpack-es6-ts | Path: app/mods/mod190.js | Language: JavaScript" in pretrain_txt
    # Ensure code newlines are never collapsed by prose unwrapping
    assert "import mod189 from './mod189';\nexport function computeNextValue" in pretrain_txt

    uta_eps = extract_uta_episodes(norm_row, max_seq_len=2048)
    assert len(uta_eps) == 1
    assert "MirekSz/webpack-es6-ts" in uta_eps[0][0]
    assert "computeNextValue" in uta_eps[0][2]

    # Vendored node_modules path should be rejected at Stage 0
    noisy_row = dict(norm_row, path="wp-content/node_modules/es6-set/polyfill.js")
    _, noisy_rej = extract_pretrain_text(noisy_row)
    assert noisy_rej is True
    assert extract_uta_episodes(noisy_row, max_seq_len=2048) == []

    # 6. Verify RoPE n_prefix alignment in project_global_kv and <eos> boundary isolation in packed blocks
    from mcl4_coe.pipeline import _is_clean_uta_episode

    dec_layer = coe.backbone.decoder_layers[0]
    dummy_seq = torch.randn(2, 16, b_cfg.d_model)
    dummy_prefix = torch.randn(2, 6, b_cfg.d_model)
    k_seq, _ = dec_layer.project_global_kv(dummy_seq, n_prefix=0)
    k_prefixed, _ = dec_layer.project_global_kv(
        torch.cat([dummy_prefix, dummy_seq], dim=1), n_prefix=6
    )
    assert torch.allclose(k_seq, k_prefixed[:, :, 6:, :], atol=1e-6)
    assert _is_clean_uta_episode("Explain X.", "Step 1: check X.", "X is valid.") is True
    assert _is_clean_uta_episode("Score Candidate 1 now.", "Evaluate step-by-step: [0]", "[0]") is False
    assert _is_clean_uta_episode("{'messages': []}", "TRUE", "{'messages': []}") is False

    # 7. Verify multi-generation progressive mitotic upcycling (4 -> 8 -> 16) keeps mitotic_pairs pairwise disjoint
    from mcl4_coe.trainer import upcycle_state_dict_experts

    sd4 = coe.state_dict()
    sd8 = upcycle_state_dict_experts(sd4, 4, 8, expert_loads=[0.35, 0.25, 0.22, 0.18])
    mp8 = sd8["backbone.encoder_layers.0.moe.mitotic_pairs_tensor"].tolist()
    assert len(mp8) == 4
    assert len({idx for pair in mp8 for idx in pair}) == 8

    sd16 = upcycle_state_dict_experts(
        sd8,
        8,
        16,
        expert_loads=[0.18, 0.14, 0.13, 0.12, 0.12, 0.11, 0.10, 0.10],
    )
    mp16 = sd16["backbone.encoder_layers.0.moe.mitotic_pairs_tensor"].tolist()
    assert len(mp16) == 8
    assert len({idx for pair in mp16 for idx in pair}) == 16


def test_ornith_v2_compositional_rlvr_and_dream_evolution(tmp_path: Path) -> None:
    """Verify Ornith-2.0 compositional environments, inline #T$# equation checker, IFEval verifier,
    OrnithEnvArchitect curriculum tiering, external rlvr_env_pool loading, and Dream-RSI scaffold/policy mutation.
    """
    from mcl4_coe import (
        COEConfig,
        CognitiveOperatorEnsemble,
        DreamProgram,
        MCL4CEDBackbone,
        MCL4CEDConfig,
        MCL4DreamDiscovery,
        MCL4PolicyProposer,
        OrnithEnvArchitect,
        OrnithSelfImprover,
        verify_python_rollout,
    )

    # 1. Verify inline #T$# arithmetic consistency penalty and single-function AST purity
    h_code = "assert solve([1, 4, 6], 3, 2) == [8, 12]"
    valid_comp = "#T$# Filter x >= 3 and multiply 4 * 2 = 8, 6 * 2 = 12.\n#A$# def solve(xs, t, m):\n    return [x * m for x in xs if x >= t]"
    assert verify_python_rollout(valid_comp, h_code) == 1.0

    # Hallucinated arithmetic in #T$# (`50 + 10 = 20`) must be penalized by 0.25x
    hallucinated_t = "#T$# First compute 50 + 10 = 20.\n#A$# def solve(xs, t, m):\n    return [x * m for x in xs if x >= t]"
    assert verify_python_rollout(hallucinated_t, h_code) == 0.25

    # Duplicate/shadowing `def solve` definitions in #A$# must be rejected (0.0)
    dup_fn = "#T$# Valid thought 2 + 2 = 4.\n#A$# def solve(xs, t, m):\n    return []\ndef solve(xs, t, m):\n    return [8, 12]"
    assert verify_python_rollout(dup_fn, h_code) == 0.0

    # 2. Verify IFEval constraint harness (`ifeval:lowercase;no_comma;keyword:shards;exact_int:31`)
    h_if = "ifeval:lowercase;no_comma;keyword:shards;exact_int:31"
    assert verify_python_rollout("#T$# 6 * 5 = 30, 30 + 1 = 31.\n#A$# cluster manages 31 total shards", h_if) == 1.0
    assert verify_python_rollout("#T$# 6 * 5 = 30, 30 + 1 = 31.\n#A$# Cluster, manages 31 total shards", h_if) == 0.5

    # 3. Verify all 24 procedural families + all 8x6x4 = 192 cross-template combinatorial pipelines pass gold == 1.0 and hack == 0.0
    improver = OrnithSelfImprover(projection_dim=16, max_seq_len=256)
    seen_domains: set[str] = set()
    for k in range(24):
        t = improver._generate_task(k, compact=False)
        seen_domains.add(t.domain)
        gold_cand = t.candidate_rollouts[t.gold_index]
        assert verify_python_rollout(gold_cand, t.harness_h, t.gold_answer) == 1.0, f"Family {k} ({t.task_id}) gold failed"
        assert min(verify_python_rollout(c, t.harness_h, t.gold_answer) for c in t.candidate_rollouts) == 0.0
    for c_idx in range(1, 193):
        ct = improver._synthesize_combinatorial_pipeline_task(c_idx, tier=(c_idx % 8) + 1)
        seen_domains.add(ct.domain)
        gold_cand = ct.candidate_rollouts[ct.gold_index]
        assert verify_python_rollout(gold_cand, ct.harness_h, ct.gold_answer) == 1.0, f"Combo {c_idx} ({ct.task_id}) gold failed"
        assert min(verify_python_rollout(c, ct.harness_h, ct.gold_answer) for c in ct.candidate_rollouts) == 0.0
    assert "compositional_code_math" in seen_domains
    assert "compositional_physics_math" in seen_domains
    assert "compositional_code_pm" in seen_domains
    assert "compositional_deduction_ifeval" in seen_domains

    # 4. Verify OrnithEnvArchitect promotes difficulty_tier up to Tier 8 when pass rate exceeds threshold
    arch = OrnithEnvArchitect()
    assert arch.difficulty_tier(0) == 1
    for _ in range(4):
        arch.record_outcome(0, pass_prob=0.95)
    assert arch.difficulty_tier(0) == 2

    # 5. Verify external rlvr_env_pool.pt loading and MorphCode-16K first-divergence FTPO
    pool_file = tmp_path / "rlvr_env_pool.pt"
    ext_t = improver._synthesize_combinatorial_pipeline_task(1, tier=2)
    torch.save({"tasks": [ext_t]}, pool_file)
    improver_ext = OrnithSelfImprover(
        projection_dim=16,
        max_seq_len=256,
        external_pool_path=pool_file,
    )
    assert len(improver_ext.external_tasks) == 1
    sampled_ext = improver_ext._generate_task(1, compact=False)
    assert sampled_ext.task_id == ext_t.task_id

    cfg16k = MCL4CEDConfig(
        vocab_size=16384,
        d_model=64,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ple_dim=16,
        expert_hidden_dim=128,
        num_experts=4,
        top_k=2,
        max_seq_len=96,
    )
    m16k = CognitiveOperatorEnsemble(MCL4CEDBackbone(cfg16k), COEConfig(projection_dim=16)).train()
    out16k = improver_ext.step_self_improvement(m16k, batch_size=2, max_seq_len=96)
    assert torch.isfinite(out16k.loss)
    assert out16k.antidoom_loss > 0.0  # MorphCode-16K first-divergence FTPO active!

    # 6. Verify Dream-RSI dynamic operator scaffold synthesis and policy mutation
    s_veto = MCL4DreamDiscovery.synthesize_operator_scaffold(
        "abduction -> decision -> metacognition",
        domain="compositional_code_math",
        depth=2,
        parent_vetoed=True,
        ded_entailed=0.2,
        novelty=0.5,
        utility=0.4,
        confidence=0.3,
        node_id=2,
    )
    assert "abstraction" in s_veto and "deduction" in s_veto and s_veto.endswith("metacognition")

    proposer = MCL4PolicyProposer(m16k, seed=42)
    muts = proposer._synthesize_mutated_programs(
        DreamProgram(),
        mean_hv=0.45,
        mean_penalty=0.20,
        max_w=3,
        max_r=4,
        revision=1,
    )
    assert len(muts) >= 2
    for mp in muts:
        parsed = DreamProgram.from_text(mp.to_text())
        assert 1 <= parsed.width <= 4


def test_dense_equivalent_moe_utilization() -> None:
    """Verify Recurrent Coverage / Tabu Routing across recurrent passes, Gram-Schmidt
    Orthogonal Mitosis at birth, Mitotic Orthogonalization loss, and Sequence Diversity loss.
    """
    from mcl4_coe.trainer import upcycle_state_dict_experts

    torch.manual_seed(17)
    cfg = MCL4CEDConfig(
        vocab_size=260,
        d_model=64,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ple_dim=16,
        expert_hidden_dim=128,
        num_experts=6,
        top_k=2,
        max_seq_len=64,
        recurrent_coverage_penalty=0.95,
        mitotic_ortho_coef=0.10,
        sequence_diversity_coef=0.05,
    )
    backbone = MCL4CEDBackbone(cfg).train()

    # 1. Verify Recurrent Coverage / Tabu Routing recruits fresh experts on Pass 2
    moe = backbone.decoder_layers[0].moe
    x_tok = torch.randn(2, 8, 64)
    val_tok = torch.ones(2, 8, dtype=torch.bool)
    _, _, _, _ = moe(x_tok, valid=val_tok, prior_expert_mask=None)
    mask_pass1 = moe.last_selected_mask
    assert mask_pass1 is not None and mask_pass1.shape == (2, 8, 6)
    _, _, _, _ = moe(x_tok, valid=val_tok, prior_expert_mask=mask_pass1)
    mask_pass2 = moe.last_selected_mask
    assert mask_pass2 is not None
    union_experts_per_token = ((mask_pass1 + mask_pass2) > 0).sum(dim=-1).float().mean().item()
    assert union_experts_per_token > float(cfg.top_k) + 1.5  # Strictly recruits new experts on Pass 2!

    # 2. Verify Gram-Schmidt Orthogonal Mitosis at birth (exact norm symmetry + zero radial dot product)
    base_cfg = MCL4CEDConfig(
        vocab_size=260,
        d_model=64,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ple_dim=16,
        expert_hidden_dim=128,
        num_experts=4,
        top_k=2,
        max_seq_len=64,
    )
    base_bb = MCL4CEDBackbone(base_cfg)
    up_sd = upcycle_state_dict_experts(
        base_bb.state_dict(),
        old_num_experts=4,
        new_num_experts=6,
        seed=99,
        orthogonal=True,
    )
    w_p_up = up_sd["decoder_layers.0.moe.experts.0.up_proj.weight"].float()
    w_c_up = up_sd["decoder_layers.0.moe.experts.4.up_proj.weight"].float()
    w_mean_up = 0.5 * (w_p_up + w_c_up)
    delta_up = 0.5 * (w_c_up - w_p_up)
    radial_cos = torch.nn.functional.cosine_similarity(w_mean_up, delta_up, dim=-1).abs().max().item()
    assert radial_cos < 5e-5
    assert torch.allclose(w_p_up.norm(dim=-1), w_c_up.norm(dim=-1), rtol=1e-5, atol=1e-6)

    w_p_dn = up_sd["decoder_layers.0.moe.experts.0.down_proj.weight"].float()
    w_c_dn = up_sd["decoder_layers.0.moe.experts.4.down_proj.weight"].float()
    w_mean_dn = 0.5 * (w_p_dn + w_c_dn)
    delta_dn = 0.5 * (w_c_dn - w_p_dn)
    radial_cos_dn = torch.nn.functional.cosine_similarity(w_mean_dn, delta_dn, dim=0).abs().max().item()
    assert radial_cos_dn < 5e-5
    assert torch.allclose(w_p_dn.norm(dim=0), w_c_dn.norm(dim=0), rtol=1e-5, atol=1e-6)

    # 3. Verify Mitotic Orthogonalization loss & multi-pass decode_candidates integration
    backbone.load_state_dict(up_sd, strict=False)
    ortho_before = float(backbone.decoder_layers[0].moe.compute_mitotic_ortho_loss(x_tok.device).item())
    assert ortho_before > 0.99
    inp_ids = torch.randint(0, 256, (2, 16), dtype=torch.long)
    lm_out = backbone(inp_ids, targets=inp_ids, recurrence_steps=2)
    assert lm_out.loss is not None and torch.isfinite(lm_out.loss)
    lm_out.loss.backward()
    with torch.no_grad():
        for p in backbone.decoder_layers[0].moe.experts.parameters():
            if p.grad is not None:
                p.sub_(0.5 * p.grad)
    ortho_after = float(backbone.decoder_layers[0].moe.compute_mitotic_ortho_loss(x_tok.device).item())
    assert ortho_after < ortho_before

    # 4. Verify fused wide-dense subset execution (forward_dense_subset, dense_expert_indices, dense_subset_k)
    subset_ids = [0, 2, 4]
    fused_dense = moe.forward_dense_subset(x_tok, subset_ids, valid=val_tok)
    manual_dense = (
        moe.experts[0](x_tok) + moe.experts[2](x_tok) + moe.experts[4](x_tok)
    )
    assert torch.allclose(fused_dense, manual_dense, rtol=1e-4, atol=1e-5)

    # Explicit pinned dense subset + hybrid sparse routing
    backbone.zero_grad(set_to_none=True)
    backbone.set_dense_expert_subset([1, 3, 5], combine_dense_and_sparse=True)
    lm_pinned = backbone(inp_ids, targets=inp_ids, recurrence_steps=1)
    assert lm_pinned.loss is not None and torch.isfinite(lm_pinned.loss)
    lm_pinned.loss.backward()
    for idx in (1, 3, 5):
        assert backbone.decoder_layers[0].moe.experts[idx].down_proj.weight.grad is not None
        assert backbone.decoder_layers[0].moe.experts[idx].down_proj.weight.grad.abs().sum() > 0

    # Dynamic sequence-level dense coalition (dense_subset_k=3) + Pass-2 coalition switching
    backbone.zero_grad(set_to_none=True)
    backbone.set_dense_expert_subset([], dense_subset_k=3, combine_dense_and_sparse=False)
    _, _, _, _ = moe(x_tok, valid=val_tok, prior_expert_mask=None)
    coal_mask_1 = moe.last_selected_mask
    assert coal_mask_1 is not None and int(coal_mask_1[0, 0].sum().item()) == 3
    _, _, _, _ = moe(x_tok, valid=val_tok, prior_expert_mask=coal_mask_1)
    coal_mask_2 = moe.last_selected_mask
    assert coal_mask_2 is not None and int(coal_mask_2[0, 0].sum().item()) == 3
    total_coalition_coverage = int(((coal_mask_1[0, 0] + coal_mask_2[0, 0]) > 0).sum().item())
    assert total_coalition_coverage >= 5  # Recruits almost the entire 6-expert bank across 2 dense passes!

    # 5. Verify Mitosis-Staged Corpus Curriculum (English prose foundation -> CoT reasoning -> decisions)
    from mcl4_coe.pipeline import CurriculumMixtureSchedule

    stages = (
        {"pretrain_lm": 0.90, "uta_trace": 0.08, "typed_decision": 0.02},
        {"pretrain_lm": 0.55, "uta_trace": 0.35, "typed_decision": 0.10},
        {"pretrain_lm": 0.30, "uta_trace": 0.50, "typed_decision": 0.20},
    )
    sched = CurriculumMixtureSchedule(
        start_weights=stages[0],
        end_weights=stages[-1],
        mode="mitotic_staged",
        stage_weights=stages,
    )
    w_s0 = sched.weights_at(0.0, stage_idx=0, stage_progress=0.0)
    w_s1 = sched.weights_at(0.4, stage_idx=1, stage_progress=0.0)
    w_s2 = sched.weights_at(0.9, stage_idx=2, stage_progress=1.0)
    assert abs(w_s0["pretrain_lm"] - 0.90) < 1e-6
    assert w_s0["pretrain_lm"] > w_s1["pretrain_lm"] > w_s2["pretrain_lm"]
    assert w_s0["uta_trace"] < w_s1["uta_trace"] < w_s2["uta_trace"]
    assert w_s0["typed_decision"] < w_s1["typed_decision"] < w_s2["typed_decision"]

    # 6. Verify Windowless Demand-Paged Context (InfiniteContextPager + WorkingStateLedger + forward_windowless)
    from mcl4_coe.context_memory import InfiniteContextPager, WorkingStateLedger

    backbone.set_dense_expert_subset([], dense_subset_k=0, combine_dense_and_sparse=False)
    ledger = WorkingStateLedger(
        objective="Refactor authentication middleware without breaking session tokens",
        active_step="Update JWT signature verifier in auth.py",
        pinned_facts={"token_ttl": "3600s", "issuer": "yycore-auth"},
    )
    assert "Refactor authentication middleware" in ledger.format_ledger_text()

    pager = InfiniteContextPager(
        page_size=16,
        top_k_pages=2,
        landmarks_per_page=2,
        working_state=ledger,
    )
    ledger_ids = torch.randint(1, 250, (1, 12), dtype=torch.long)
    pager.update_working_state(backbone, ledger_ids)
    assert pager.working_state.compressed_state is not None

    # Feed a 64-token stream through a model with active_window_len=16 (4x beyond active window!)
    long_stream = torch.randint(1, 250, (2, 64), dtype=torch.long)
    long_targets = long_stream.clone()
    long_targets[:, :8] = -100  # mask some targets to also exercise valid-target-only deep supervision
    out_win = backbone.forward_windowless(
        long_stream,
        targets=long_targets,
        pager=pager,
        active_window_len=16,
        page_size=16,
        top_k_pages=2,
        recurrence_steps=2,
        deep_supervision_weight=0.2,
    )
    # 48 prefix tokens -> 3 pages of 16 tokens compressed into TurboQuant pager; active tail is 16 tokens
    assert pager.num_pages == 3
    assert pager.total_ingested_tokens == 48
    assert out_win.logits.shape == (2, 16, cfg.vocab_size)
    assert out_win.loss is not None and torch.isfinite(out_win.loss)

    top_pages = pager.retrieve_top_k_pages(backbone, long_stream[:, -16:], top_k=2)
    assert len(top_pages) == 2
    gist_state = pager.build_gist_state(backbone, long_stream[:, -16:], top_k=2)
    # Gist state holds [12 ledger tokens + 2 * 16 retrieved page tokens] = 44 position-invariant prefix tokens
    assert gist_state is not None
    assert gist_state.hidden.shape == (2, 12 + 2 * 16, cfg.d_model)

    # 7. Verify COE-Gated Quarantined Fast-Weight Plasticity ("Einstein on a Whiteboard" Engine)
    from mcl4_coe.operators import COEConfig, CognitiveOperatorEnsemble

    coe = CognitiveOperatorEnsemble(backbone, COEConfig(projection_dim=16, recurrence_steps=2))
    slow_snapshot = {k: v.detach().clone() for k, v in coe.state_dict().items()}
    controller = coe.create_whiteboard_controller(
        whiteboard_slots=6,
        lora_rank=8,
        lora_alpha=4.0,
        lr=4e-2,
        inner_steps=2,
        max_prefix_norm=0.40,
        max_adapter_frobenius_norm=0.20,
        min_novelty=0.02,
        min_entropy_floor=0.20,
        max_anchor_drift=0.25,
    )
    anchor_ids = torch.randint(1, 250, (2, 16), dtype=torch.long)
    controller.register_anchor_invariants(anchor_ids)

    wb_state_ids = torch.randint(1, 250, (2, 20), dtype=torch.long)
    wb_hyp_ids = torch.randint(1, 250, (2, 16), dtype=torch.long)
    report1 = controller.step(wb_state_ids, wb_hyp_ids, recurrence_steps=2)
    assert report1.committed is True
    assert report1.status == "committed"
    assert report1.whiteboard_loss_after < report1.whiteboard_loss_before
    assert report1.adapter_frobenius_norm <= 0.20 + 1e-5
    assert report1.prefix_delta_norm <= 0.40 + 1e-5

    # Guarantee 100% of slow weights in COE + Backbone are bitwise untouched
    for k, v in coe.state_dict().items():
        assert torch.equal(v, slow_snapshot[k]), f"Slow weight {k} mutated during plasticity!"

    # Gate 1: Re-submitting identical hypothesis is filtered by COE Association Novelty
    report_dup = controller.step(wb_state_ids, wb_hyp_ids, recurrence_steps=2)
    assert report_dup.committed is False
    assert report_dup.status == "skipped_low_novelty"

    # Gate 3: If Correspondence Principle anchor drift threshold is violated, instant O(1) rollback occurs
    committed_lora_b = controller.whiteboard.fast_lora_B.detach().clone()
    controller.max_anchor_drift = -1.0  # Force anchor drift gate rejection
    wb_hyp_ids_2 = torch.randint(1, 250, (2, 16), dtype=torch.long)
    report_rb = controller.step(wb_state_ids, wb_hyp_ids_2, recurrence_steps=2)
    assert report_rb.committed is False
    assert report_rb.status == "rolled_back_anchor_invariant_drift"
    assert torch.equal(controller.whiteboard.fast_lora_B, committed_lora_b)

    # Inference conditioned on the committed QuarantinedWhiteboardState
    wb_out = coe.forward_with_whiteboard(
        wb_state_ids, targets=wb_state_ids, whiteboard=controller.whiteboard, recurrence_steps=2
    )
    assert wb_out.loss is not None and torch.isfinite(wb_out.loss)

    # 8. Verify Strengthened Inference-Time Learning:
    #    - Biorthogonal Hopfield-Kanerva Rule Readout & Surprise-Weighted Generative Fast Weights
    #    - Continual Multi-Chapter Episodic Absorption (`append_episodic_rules`)
    #    - Mitotic Whiteboard-to-Child Sleep Consolidation (`consolidate_whiteboard_to_mitotic_children`)
    controller.max_anchor_drift = 0.35
    controller.whiteboard.reset_fast_weights()

    ch1_ctx = torch.randint(4, 240, (2, 20), dtype=torch.long)
    ch1_hyp = torch.randint(4, 240, (2, 16), dtype=torch.long)
    ch1_doc = torch.cat([ch1_ctx, ch1_hyp], dim=1)
    ch2_ctx = torch.randint(4, 240, (2, 20), dtype=torch.long)
    ch2_hyp = torch.randint(4, 240, (2, 16), dtype=torch.long)
    ch2_doc = torch.cat([ch2_ctx, ch2_hyp], dim=1)

    # Base generative loss before absorption
    with torch.no_grad():
        base_gen_loss = float(
            coe.backbone(ch1_doc[:, :-1], targets=ch1_doc[:, 1:], recurrence_steps=2).loss.item()
        )

    abs_rep_1 = controller.absorb_context(
        ch1_doc,
        premise_ids=ch1_ctx,
        conclusion_ids=ch1_hyp,
        recurrence_steps=2,
    )
    assert abs_rep_1.committed is True
    assert abs_rep_1.whiteboard_loss_after < abs_rep_1.whiteboard_loss_before
    assert controller.whiteboard.episodic_rule_count == 2

    # Generative perplexity / loss on absorbed domain text must strictly improve
    with torch.no_grad():
        wb_gen_loss = float(
            coe.forward_with_whiteboard(
                ch1_doc[:, :-1],
                targets=ch1_doc[:, 1:],
                whiteboard=controller.whiteboard,
                recurrence_steps=2,
            ).loss.item()
        )
    assert wb_gen_loss < base_gen_loss

    # Continual Chapter 2 absorption appends without forgetting Chapter 1 rules
    abs_rep_2 = controller.absorb_context(
        ch2_doc,
        premise_ids=ch2_ctx,
        conclusion_ids=ch2_hyp,
        recurrence_steps=2,
    )
    assert abs_rep_2.committed is True
    assert controller.whiteboard.episodic_rule_count == 4

    # Verify solve_decision_on_whiteboard (Dense+WB) achieves 100% decision accuracy on absorbed rules
    all_ctx = torch.cat([ch1_ctx, ch2_ctx], dim=0)
    all_hyp = torch.cat([ch1_hyp, ch2_hyp], dim=0)
    ctx_lens_all = torch.full((4,), 20, dtype=torch.long)
    opt_ids_all = all_hyp.unsqueeze(0).expand(4, -1, -1).contiguous()
    opt_lens_all = torch.full((4, 4), 16, dtype=torch.long)
    targets_all = torch.arange(4, dtype=torch.long)

    dec_dense_wb, _ = controller.solve_decision_on_whiteboard(
        all_ctx,
        ctx_lens_all,
        opt_ids_all,
        opt_lens_all,
        targets=targets_all,
        recurrence_steps=2,
        adapt_on_context=False,
    )
    assert int((dec_dense_wb.logits.argmax(dim=-1) == targets_all).sum().item()) == 4

    # Verify Mitotic Whiteboard-to-Child Sleep Consolidation keeps parent experts at 0.0 delta
    cons_res = controller.consolidate_whiteboard_to_mitotic_children(
        consolidation_rate=0.25,
        reset_whiteboard_after=True,
    )
    assert cons_res["consolidated"] is True
    assert cons_res["num_child_experts_updated"] >= 1
    assert cons_res["parent_expert_max_delta"] == 0.0
    assert cons_res["child_expert_frobenius_delta"] > 0.0

    # 9. Verify Universal Multi-Modality Ingestion (Prose + Code AST + Web HTML) & 2-Hop Transitive Closure
    from mcl4_coe import ByteTokenizer, InfiniteContextPager

    tok_b = ByteTokenizer()
    pager_b = InfiniteContextPager(page_size=32, landmarks_per_page=2, top_k_pages=2)
    raw_paper_excerpt = (
        "# Section 1: Symplectic Routing\n"
        "When sparse experts are routed through the Krylov-7 manifold, the router enforces a symplectic gauge projection.\n"
        "When the router enforces a symplectic gauge projection, Hamiltonian energy drift stays below 1e-6.\n"
    )
    raw_code_excerpt = (
        "class LedgerNode:\n"
        "    default_cipher = 'chacha20_poly1305'\n"
        "    def check_crc(self, bad_crc: bool) -> str:\n"
        "        if bad_crc:\n"
        "            return 'quarantine_shard'\n"
        "        return 'commit_block'\n"
    )
    raw_web_excerpt = (
        "<html><body><table>"
        "<tr><th>Endpoint</th><th>Protocol</th></tr>"
        "<tr><td>/v4/sync</td><td>RDMA-RoCEv2</td></tr>"
        "</table></body></html>"
    )
    rep_p, props_p = controller.ingest_artifact(
        raw_paper_excerpt, tokenizer=tok_b, modality="prose", pager=pager_b, reset_whiteboard=True
    )
    rep_c, props_c = controller.ingest_artifact(
        raw_code_excerpt, tokenizer=tok_b, modality="code", pager=pager_b
    )
    rep_w, props_w = controller.ingest_artifact(
        raw_web_excerpt, tokenizer=tok_b, modality="web", pager=pager_b
    )
    assert len(props_p) >= 2
    assert len(props_c) >= 2
    assert len(props_w) >= 1
    assert rep_p.committed and rep_c.committed and rep_w.committed
    assert pager_b.num_pages >= 1

    # 10. Verify Frontier Whiteboard Capabilities:
    #     - Symbolic-Numerical & Boolean Predicate Extraction (`extract_symbolic_predicates`)
    #     - Algebraic Multi-Whiteboard Cartridge Fusion (`merge_whiteboards`)
    #     - Non-Monotonic AGM Belief Revision (`revision_history`)
    #     - Cross-Modal Contradiction & Bound Auditing (`audit_contradictions`)
    num_p, bool_p = controller.whiteboard.extract_symbolic_predicates(
        "in verify_shard_envelope, suppose crc_mismatch is false and epoch_lag = 27"
    )
    assert ("epoch_lag", "==", 27.0) in num_p
    assert ("crc_mismatch", False) in bool_p

    ctrl_cart1 = coe.create_whiteboard_controller(whiteboard_slots=6, lora_rank=8)
    ctrl_cart2 = coe.create_whiteboard_controller(whiteboard_slots=6, lora_rank=8)
    ctrl_cart1.ingest_artifact(raw_code_excerpt, tokenizer=tok_b, modality="code", reset_whiteboard=True)
    ctrl_cart2.ingest_artifact(
        "class LedgerNode:\n    default_cipher = 'aes_256_gcm_siv'\n",
        tokenizer=tok_b,
        modality="code",
        reset_whiteboard=True,
    )
    merged_wb = controller.merge_whiteboards(
        [ctrl_cart1.whiteboard, ctrl_cart2.whiteboard], allow_belief_revision=True
    )
    assert len(merged_wb.revision_history) >= 1
    assert "aes_256_gcm_siv" in merged_wb.revision_history[0]["new_conclusion"]

    conflicts = controller.audit_contradictions(
        raw_paper_excerpt,
        "When sparse experts are routed through the Krylov-7 manifold, Hamiltonian energy drift stays below 1e-2.\n",
        tokenizer=tok_b,
        modality_a="prose",
        modality_b="prose",
    )
    assert len(conflicts) >= 1
    assert conflicts[0]["type"] in ("transitive_2hop_bound_violation", "numeric_bound_violation")


def test_v7_shard_windowed_streaming_and_eviction(tmp_path) -> None:
    """Verify shard-windowed EpochBlockSampler streaming and consumed-shard disk eviction during mitosis."""
    from mcl4_coe.data import EpochBlockSampler
    from mcl4_coe.pipeline import (
        BlockAlignedShardWriter,
        CurriculumMixtureSchedule,
        CurriculumMixtureSampler,
        MultiTrackMixtureDataset,
        ShardedTokenBlockDataset,
    )

    # 1. Verify EpochBlockSampler with shard_boundaries visits each shard window sequentially
    sampler = EpochBlockSampler(12, seed=42, shard_boundaries=[4, 8, 12])
    s0 = set(sampler.take(4).tolist())
    s1 = set(sampler.take(4).tolist())
    s2 = set(sampler.take(4).tolist())
    assert s0 == {0, 1, 2, 3}
    assert s1 == {4, 5, 6, 7}
    assert s2 == {8, 9, 10, 11}

    # 2. Build a 3-shard track on disk (`max_blocks_per_shard=2`) and verify consumed-shard eviction
    track_dir = tmp_path / "pretrain_lm"
    writer = BlockAlignedShardWriter(
        track_dir,
        track="pretrain_lm",
        split="train",
        seq_len=16,
        max_blocks_per_shard=2,
        vocab_size=260,
    )
    for i in range(6):
        writer.add_pretrain_document(f"Document {i} " + ("alpha beta gamma " * 8))
    writer.flush_shard()

    receipt_path = track_dir / "index_receipt.json"
    ds = ShardedTokenBlockDataset(receipt_path, track="pretrain_lm", split="train")
    assert len(ds.shards) >= 3

    sched = CurriculumMixtureSchedule(start_weights={"pretrain_lm": 1.0}, mode="constant")
    mix_ds = MultiTrackMixtureDataset({"pretrain_lm": receipt_path}, split="train")
    mix_sampler = CurriculumMixtureSampler(mix_ds, sched, seed=7)

    # Consume exact blocks of Shard 0 (`2` blocks)
    idx_b0 = mix_sampler.take(2)
    batch0 = mix_ds.batch(idx_b0.tolist())
    assert batch0["input_ids"].shape == (2, 16)

    # Evict consumed shards from disk (`Shard 0` has cum_blocks=2 <= cursor=2)
    from mcl4_coe.pipeline import _track_shards_complete

    cleared = mix_sampler.evict_consumed_shards(delete_from_disk=True)
    assert "pretrain_lm" in cleared
    assert cleared["pretrain_lm"] == [ds.shards[0]["bin_file"]]
    assert not (track_dir / ds.shards[0]["bin_file"]).exists()
    assert (track_dir / ds.shards[-1]["bin_file"]).exists()
    assert _track_shards_complete(receipt_path)

    # Continue streaming Shard 1 & Shard 2 and verify fallback if an evicted index is queried
    idx_b1 = mix_sampler.take(2)
    batch1 = mix_ds.batch(idx_b1.tolist())
    assert batch1["input_ids"].shape == (2, 16)
    fallback_batch = mix_ds.batch([0])
    assert fallback_batch["input_ids"].shape == (1, 16)


def test_inference_ui_session_and_endpoints() -> None:
    """Verify InferenceSessionController checkpoint discovery, dynamic MoE/recurrence knobs, Whiteboard steering, and HTTP endpoints."""
    import json
    import threading
    import urllib.request
    from mcl4_coe.inference_ui import InferenceSessionController, start_inference_ui_server

    ctrl = InferenceSessionController()
    ckpts = ctrl.list_available_checkpoints()
    assert len(ckpts) >= 1
    ckpt_ids = {c["checkpoint_id"] for c in ckpts}
    assert "colab_mcl4_coe_v7_mitotic_3b_fast/step_05800" in ckpt_ids

    # 1. Initialize preset and verify dynamic MoE density & Tabu controls
    st0 = ctrl.load_preset({"preset": "tiny_morph16k", "device": "cpu", "dtype": "float32"})
    assert st0["loaded_info"]["loaded"] is True
    p_top2 = st0["parameter_stats"]["active_parameters"]

    st1 = ctrl.update_config(
        {
            "top_k": 4,
            "dense_subset_k": 6,
            "recurrent_coverage_penalty": 0.45,
            "recurrence_steps": 2,
        }
    )
    assert st1["parameter_stats"]["active_experts_per_token"] == 6
    assert st1["parameter_stats"]["active_parameters"] > p_top2
    assert ctrl.model is not None
    assert ctrl.model.backbone.encoder_layers[0].moe.dense_subset_k == 6
    assert ctrl.model.backbone.encoder_layers[0].moe.recurrent_coverage_penalty == 0.45

    # 2. Run an interactive generation turn and verify trace segments + MoE heatmap
    gen_res = ctrl.generate_turn(
        {
            "prompt": "Compute 3 + 4.",
            "max_new_tokens": 24,
            "dense_subset_k": 0,
            "top_k": 2,
        }
    )
    turn = gen_res["turn"]
    assert turn["turn_id"] == 1
    assert "thought" in turn["segments"] and "answer" in turn["segments"]
    assert len(turn["expert_routing"]["encoder_loads"]) == 2

    # 3. Ingest a rule into Quarantined Whiteboard and steer generation with Whiteboard active
    wb_res = ctrl.whiteboard_action(
        {
            "action": "ingest",
            "modality": "prose",
            "text": "When the telemetry frame arrives on channel 9, the controller emits ack_vector_9.",
        }
    )
    assert wb_res["status"]["whiteboard"]["episodic_rule_count"] >= 1
    gen_wb = ctrl.generate_turn(
        {
            "prompt": "What happens when the telemetry frame arrives on channel 9?",
            "max_new_tokens": 20,
            "use_whiteboard": True,
        }
    )
    assert gen_wb["turn"]["config_snapshot"]["use_whiteboard"] is True

    # 4. Run Typed-COE Decision Probe
    probe = ctrl.probe_decision(
        {
            "context": "Channel 9 received a telemetry frame.",
            "question": "Which ack vector is emitted?",
            "options": ["ack_vector_9", "ack_vector_0"],
        }
    )
    assert len(probe["candidates"]) == 2
    assert 0 <= probe["winner_index"] < 2

    # 5. Verify HTTP server endpoints
    server = start_inference_ui_server(ctrl, host="127.0.0.1", port=0)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        html = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode("utf-8")
        assert "opencode // mcl4-coe" in html
        st_http = json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status").read().decode("utf-8")
        )
        assert st_http["loaded_info"]["loaded"] is True
    finally:
        server.shutdown()
        server.server_close()


def test_olmoe_graft_equivalence_and_inplace_mitosis(tmp_path) -> None:
    """Verify full QK-Norm RoPE permutation equivalence, partition-invariant norm_topk_prob=False twin mitosis, and in-place mitosis with Generation-Scoped Plasticity."""
    from mcl4_coe.backbone import MCL4CEDConfig, SparseMoE, SwiGLUExpert
    from mcl4_coe.pipeline import BlockAlignedShardWriter
    from mcl4_coe.trainer import MCL4COETrainer, TrainingRunConfig

    torch.manual_seed(123)
    moe = SparseMoE(
        MCL4CEDConfig(
            d_model=32,
            n_heads=4,
            expert_hidden_dim=48,
            num_experts=4,
            top_k=2,
            norm_topk_prob=False,
        )
    )
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        y_pre = moe(x)[0]
        # Clone expert 0 into expert 4 with exact identical weights and router row
        child = SwiGLUExpert(32, 48)
        child.up_proj.weight.data.copy_(moe.experts[0].up_proj.weight.data)
        child.down_proj.weight.data.copy_(moe.experts[0].down_proj.weight.data)
        moe.experts.append(child)
        new_rw = torch.cat([moe.router.weight.data, moe.router.weight.data[0:1].clone()], dim=0)
        moe.router = torch.nn.Linear(32, 5, bias=False)
        moe.router.weight.data.copy_(new_rw)
        moe.num_experts = 5
        moe.mitotic_pairs = [(0, 4)]
        moe._grouped_weight_cache = None
        moe._mitotic_idx_cache = None
        y_post = moe(x)[0]
    assert torch.allclose(y_pre, y_post, atol=1e-5), "Partition-invariant twin mitosis must be exact at Step 0 when norm_topk_prob=False"

    # Verify full QK-Norm + untied embeddings + in-place mitosis in trainer
    shards_dir = tmp_path / "shards"
    runs_dir = tmp_path / "runs"
    for track_name in ("pretrain_lm", "uta_trace", "typed_decision"):
        w = BlockAlignedShardWriter(
            shards_dir / track_name,
            track=track_name,
            split="train",
            seq_len=16,
            max_blocks_per_shard=4,
            vocab_size=260,
        )
        for i in range(4):
            w.add_pretrain_document(f"Sample document {i} " + ("alpha beta gamma " * 8))
        w.flush_shard()
    cfg = TrainingRunConfig(
        run_id="test_olmoe_inplace_mitosis",
        runs_root=str(runs_dir),
        shards_root=str(shards_dir),
        corpora_root=str(tmp_path / "no_corpora"),
        vocab_size=260,
        max_seq_len=16,
        d_model=32,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ple_dim=8,
        expert_hidden_dim=48,
        num_experts=4,
        max_experts=6,
        top_k=2,
        use_qk_norm=True,
        qk_norm_mode="full",
        norm_topk_prob=False,
        tie_word_embeddings=False,
        preserve_residual_across_stacks=True,
        plastic_experts_per_layer=2,
        freeze_donor_embeddings=True,
        enable_mitosis=True,
        mitosis_milestones=((2, 6),),
        total_steps=3,
        batch_size=2,
        micro_batch_size=2,
        enable_ornith_rl=False,
        enable_dream_rsi=False,
        checkpoint_every=3,
        save_optimizer_state=False,
    )
    trainer = MCL4COETrainer(cfg)
    assert trainer.model.backbone.token_embedding.weight is not trainer.model.backbone.lm_head.weight
    assert not trainer.model.backbone.token_embedding.weight.requires_grad
    metrics = trainer.train()
    assert metrics["num_experts"] == 6
    plastic_count = sum(
        1 for e in trainer.model.backbone.encoder_layers[0].moe.experts if e.up_proj.weight.requires_grad
    )
    assert plastic_count == 4  # 2 added twin pairs = 4 active twin experts >= cap(2)


def test_ornith_pp_weighting_granular_skills_and_infinite_recursion() -> None:
    """Verify osu!-style Performance Point weighting (`p * 0.95^(n-1)`), 36 granular skills,
    unbounded recursive RL chains (`L -> inf`), and Dream-RSI `PP` / `RECURSION` rank features.
    """
    from mcl4_coe import (
        DreamProgram,
        DreamRSIEngine,
        OrnithEnvArchitect,
        OrnithPPLeaderboard,
        OrnithSelfImprover,
        SKILL_TAXONOMY,
        compute_inverse_star_miss_penalty,
        compute_total_pp,
        compute_weighted_pp_series,
        verify_python_rollout,
    )

    # 1. Verify exact osu! pp weighting formula: Total pp = sum p_n * 0.95^(n-1)
    # Top 5 scores [110, 100, 100, 90, 80] -> approx [110, 95, 90, 77, 65]
    top_five = [110.0, 100.0, 100.0, 90.0, 80.0]
    weighted = compute_weighted_pp_series(top_five, decay=0.95)
    assert [round(w) for w in weighted] == [110, 95, 90, 77, 65]
    assert compute_total_pp(top_five, decay=0.95) == pytest.approx(sum(weighted), abs=1e-3)

    # 2. Verify OrnithPPLeaderboard diminishing returns when farming easy scores vs top-1 play
    lb = OrnithPPLeaderboard(decay=0.95)
    first_sub = lb.submit_play("science.experimental_control_abduction", 120.0)
    assert first_sub.skill_rank == 1
    assert first_sub.skill_weight == pytest.approx(1.0)
    assert first_sub.weighted_pp == pytest.approx(120.0)

    # Repeated lower-score farming drops to deeper ranks n > 1 with exponentially decaying 0.95^(n-1) weight
    for _ in range(15):
        lb.submit_play("science.experimental_control_abduction", 100.0)
    farm_sub = lb.submit_play("science.experimental_control_abduction", 80.0)
    assert farm_sub.skill_rank == 17
    assert farm_sub.skill_weight == pytest.approx(0.95 ** 16, abs=1e-4)
    assert farm_sub.grad_scale < first_sub.grad_scale

    # Breakthrough higher-difficulty play (160pp) takes rank #1 and restores full weight 1.0
    breakthrough = lb.submit_play("science.experimental_control_abduction", 160.0)
    assert breakthrough.skill_rank == 1
    assert breakthrough.skill_weight == pytest.approx(1.0)
    assert breakthrough.delta_skill_pp > 40.0

    # 2b. Verify inverse-star miss penalty: failing 1-star is detrimental to PP & boosts grad_scale,
    # whereas failing a high-star (50-star) frontier task incurs near-zero PP loss
    assert compute_inverse_star_miss_penalty(1.0, 0.0) == pytest.approx(45.0)
    assert compute_inverse_star_miss_penalty(50.0, 0.0) < 0.40
    assert compute_inverse_star_miss_penalty(1.0, 0.75) == 0.0

    miss_low_star = lb.submit_play(
        "science.experimental_control_abduction",
        5.0,
        star_difficulty=1.0,
        verified_reward=0.0,
    )
    assert miss_low_star.miss_penalty_pp == pytest.approx(45.0)
    assert miss_low_star.delta_skill_pp < -40.0
    assert miss_low_star.grad_scale > 1.40

    miss_high_star = lb.submit_play(
        "science.experimental_control_abduction",
        5.0,
        star_difficulty=50.0,
        verified_reward=0.0,
    )
    assert miss_high_star.miss_penalty_pp < 0.40
    assert abs(miss_high_star.delta_skill_pp) < 0.40

    # 3. Verify all 56 granular skills & specialized/Sherlockian/Polymath/Agentic families (0..55) verify gold == 1.0 and hack == 0.0
    assert len(SKILL_TAXONOMY) == 64
    improver = OrnithSelfImprover(projection_dim=16, max_seq_len=384)
    seen_skills: set[str] = set()
    seen_domains: set[str] = set()
    for k in range(64):
        t = improver._generate_task(k, compact=False)
        seen_skills.add(t.skill)
        seen_domains.add(t.domain)
        gold_cand = t.candidate_rollouts[t.gold_index]
        assert verify_python_rollout(gold_cand, t.harness_h, t.gold_answer) == 1.0, (
            f"Family {k} ({t.skill} / {t.task_id}) gold failed verification"
        )
        assert min(verify_python_rollout(c, t.harness_h, t.gold_answer) for c in t.candidate_rollouts) == 0.0

    # Also verify multi-stage scaling across higher tiers (tier=1, 3, 6, 12) for Sherlockian (36..39), Polymath (40..47) & Agentic (48..55)
    for tier_val in (1, 3, 6, 12):
        for sh_m in range(4):
            t_sh = improver._synthesize_sherlock_reasoning_task(10 + sh_m, tier=tier_val, mode=sh_m)
            assert verify_python_rollout(
                t_sh.candidate_rollouts[t_sh.gold_index], t_sh.harness_h, t_sh.gold_answer
            ) == 1.0
        for pm_m in range(8):
            t_pm = improver._synthesize_polymath_challenge_task(20 + pm_m, tier=tier_val, mode=pm_m)
            assert verify_python_rollout(
                t_pm.candidate_rollouts[t_pm.gold_index], t_pm.harness_h, t_pm.gold_answer
            ) == 1.0
        for ag_m in range(8):
            t_ag = improver._synthesize_agentic_tool_terminal_task(30 + ag_m, tier=tier_val, mode=ag_m)
            assert t_ag.recursion_depth >= 3 + (tier_val - 1)
            assert verify_python_rollout(
                t_ag.candidate_rollouts[t_ag.gold_index], t_ag.harness_h, t_ag.gold_answer
            ) == 1.0
        for lc_m in range(8):
            t_lc = improver._synthesize_leetcode_recursive_coding_task(40 + lc_m, tier=tier_val, mode=lc_m)
            assert t_lc.recursion_depth >= 3 + (tier_val - 1)
            assert verify_python_rollout(
                t_lc.candidate_rollouts[t_lc.gold_index], t_lc.harness_h, t_lc.gold_answer
            ) == 1.0

    assert len(seen_skills) == 64
    for req_dom in (
        "arc_science",
        "arc_physics",
        "arc_chemistry",
        "arc_biology",
        "hellaswag_commonsense",
        "mmlu_economics",
        "mmlu_logic",
        "mmlu_probability",
        "mmlu_cs_systems",
        "recursive_code",
        "infinite_recursion",
        "sherlock_eliminative_deduction",
        "sherlock_diagnostic_abduction",
        "sherlock_forensic_timeline",
        "sherlock_pattern_induction",
        "hackthebox_security",
        "sherlock_word_math",
        "sherlock_cryptography",
        "frontier_science_math",
        "behavioral_analysis",
        "hard_algorithmic_code",
        "cross_skill_synthesis",
        "terminal_rl",
        "tool_use_rl",
        "computer_use_rl",
        "cybergym_rl",
        "cybersec_phd_rl",
        "infinite_agentic_recursion",
        "cross_skill_agentic",
        "leetcode_dp_rl",
        "leetcode_divconq_rl",
        "leetcode_backtracking_rl",
        "leetcode_graph_rl",
        "leetcode_array_window_rl",
        "leetcode_bsearch_rl",
        "leetcode_tree_interval_rl",
        "leetcode_grandmaster_rl",
    ):
        assert req_dom in seen_domains

    # 4. Verify unbounded tier promotion (beyond old Tier-8 cap) and deep L-hop infinite recursion
    arch = OrnithEnvArchitect()
    for _ in range(40):
        arch.record_outcome(34, pass_prob=0.95, pp_weight=0.20)
    assert arch.difficulty_tier(34) > 10
    assert arch.recursion_depth(34) > 12
    assert "INF" in arch.difficulty_mods(34)

    deep_chain = improver._synthesize_infinite_recursive_task(7, tier=12, mode="chain")
    assert deep_chain.recursion_depth == 14
    assert deep_chain.star_difficulty > 5.0
    assert verify_python_rollout(
        deep_chain.candidate_rollouts[deep_chain.gold_index],
        deep_chain.harness_h,
        deep_chain.gold_answer,
    ) == 1.0

    deep_code = improver._synthesize_infinite_recursive_task(8, tier=12, mode="code_recursion")
    assert deep_code.recursion_depth == 14
    assert verify_python_rollout(
        deep_code.candidate_rollouts[deep_code.gold_index],
        deep_code.harness_h,
        deep_code.gold_answer,
    ) == 1.0

    # 5. Verify Dream-RSI supports PP & RECURSION rank features and populates PP telemetry on OrnithStepOutput
    prog = DreamProgram.from_text(
        "WIDTH 2\nROOT 2\nRANK PP RECURSION UTILITY SCORE\nSTOP 0.88\nPATIENCE 2\nROUNDS 3"
    )
    assert prog.rank == ("PP", "RECURSION", "UTILITY", "SCORE")
    model = _tiny_model().train()
    imp_dream = OrnithSelfImprover(
        projection_dim=model.config.projection_dim,
        max_seq_len=128,
        enable_dream_rsi=True,
        dream_engine=DreamRSIEngine(initial_program=prog, max_rounds=2, workers=2, max_calls=4, revisions_per_cycle=2),
    )
    step_out = imp_dream.step_self_improvement(model, batch_size=2)
    assert step_out.raw_pp > 0.0
    assert step_out.weighted_pp > 0.0
    assert step_out.total_pp >= step_out.weighted_pp
    assert step_out.star_difficulty >= 1.0
    assert step_out.active_skill != ""


def test_multi_round_clone_only_mitosis_warmup_and_pure_rl(tmp_path: Path) -> None:
    """Verify multi-round clone-only mitosis (`4 -> 6 -> 8`):
    1. Freezes the original RL'd experts (`0..3`) and preserves their exact weights (`0.0` diff).
    2. Warms up only the newly spawned mitotic clones (`4, 5`) on tokens (`clone_token_warmup`) to shift
       their weights away from their donor parents while keeping COE heads frozen.
    3. Transitions automatically to Pure RL (`clone_pure_rl`) on the mitotic clones while keeping all
       non-clone experts frozen (`0.0` diff).
    4. When starting a second mitosis round (`6 -> 8`), releases the Round-1 pairs (`(0,4), (1,5)`),
       freezes both original experts and Round-1 clones (`0..5`), and repeats the warmup -> pure RL
       lifecycle exclusively on the new Round-2 mitotic clones (`6, 7`).
    """
    import json

    from mcl4_coe import BlockAlignedShardWriter, MCL4COETrainer, TrainingRunConfig

    shards_dir = tmp_path / "shards"
    runs_dir = tmp_path / "runs"
    for t_name in ("pretrain_lm", "uta_trace", "typed_decision"):
        w = BlockAlignedShardWriter(
            shards_dir / t_name,
            track=t_name,
            split="train",
            seq_len=64,
            max_blocks_per_shard=4,
        )
        for i in range(4):
            if t_name == "pretrain_lm":
                w.add_pretrain_document(f"Axiom {i}: capillary flow balances surface tension against gravity.")
            else:
                w.add_uta_episode(f"Solve {i}+3", f"{i}+3={i+3}", str(i + 3))
        w.flush_shard()

    # Base RL'd checkpoint with E=4 experts
    base_cfg = TrainingRunConfig(
        run_id="base_4e",
        runs_root=str(runs_dir),
        shards_root=str(shards_dir),
        corpora_root=str(tmp_path / "no_corpora"),
        d_model=32,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ple_dim=8,
        expert_hidden_dim=64,
        num_experts=4,
        max_experts=4,
        top_k=4,
        max_seq_len=64,
        projection_dim=8,
        total_steps=1,
        batch_size=2,
        micro_batch_size=2,
        checkpoint_every=1,
        sample_every=10,
        eval_every=0,
    )
    base_trainer = MCL4COETrainer(base_cfg)
    base_trainer.train()
    base_ckpt = runs_dir / "base_4e" / "checkpoints" / "step_00001.meta.json"
    base_exp_weights = {
        e: base_trainer.model.backbone.encoder_layers[0].moe.experts[e].up_proj.weight.detach().clone()
        for e in range(4)
    }

    # Mitosis Round 1: E=4 -> 6 (1 step token warmup on clones [4, 5], then 1 step pure RL on clones [4, 5])
    r1_cfg = TrainingRunConfig(
        run_id="mitosis_r1_6e",
        runs_root=str(runs_dir),
        shards_root=str(shards_dir),
        corpora_root=str(tmp_path / "no_corpora"),
        resume_checkpoint=str(base_ckpt),
        d_model=32,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ple_dim=8,
        expert_hidden_dim=64,
        num_experts=6,
        max_experts=6,
        top_k=4,
        max_seq_len=64,
        projection_dim=8,
        preserve_residual_across_stacks=True,
        donor_backbone_lr_scale=0.001,
        freeze_donor_embeddings=True,
        freeze_non_clone_experts=True,
        mitotic_perturbation_scale=5e-3,
        clone_expert_lr_scale=1.0,
        clone_warmup_steps=1,
        pure_rl_after_clone_warmup=True,
        release_mitotic_pairs=True,
        mitotic_ortho_coef=0.04,
        enable_ornith_rl=True,
        ornith_every=1,
        ornith_weight=0.25,
        lr=1e-3,
        total_steps=3,
        batch_size=2,
        micro_batch_size=2,
        checkpoint_every=2,
        sample_every=10,
        eval_every=0,
    )
    r1_trainer = MCL4COETrainer(r1_cfg)
    assert r1_trainer._get_model_mitotic_pairs() == [(0, 4), (1, 5)]
    assert r1_trainer._plastic_expert_indices == [4, 5]
    # Verify non-clone experts 0..3 preserved exact donor weights at birth and have requires_grad=False
    for e in range(4):
        exp_mod = r1_trainer.model.backbone.encoder_layers[0].moe.experts[e]
        assert not exp_mod.up_proj.weight.requires_grad
        assert torch.equal(exp_mod.up_proj.weight.detach(), base_exp_weights[e])
    for c in (4, 5):
        exp_mod = r1_trainer.model.backbone.encoder_layers[0].moe.experts[c]
        assert exp_mod.up_proj.weight.requires_grad
    clone_Birth_4 = r1_trainer.model.backbone.encoder_layers[0].moe.experts[4].up_proj.weight.detach().clone()

    r1_summary = r1_trainer.train()
    assert r1_summary["step"] == 3
    # Non-clone experts 0..3 must remain 100% frozen across both Phase 1 (token warmup) and Phase 2 (pure RL)
    for e in range(4):
        exp_mod = r1_trainer.model.backbone.encoder_layers[0].moe.experts[e]
        assert torch.equal(exp_mod.up_proj.weight.detach(), base_exp_weights[e])
    # Clone experts 4, 5 must have updated weights
    assert not torch.equal(
        r1_trainer.model.backbone.encoder_layers[0].moe.experts[4].up_proj.weight.detach(),
        clone_Birth_4,
    )
    r1_lines = [
        json.loads(line)
        for line in (runs_dir / "mitosis_r1_6e" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    step2_rec = next(r for r in r1_lines if r["step"] == 2)
    step3_rec = next(r for r in r1_lines if r["step"] == 3)
    assert step2_rec["mitosis_phase"] == "clone_token_warmup"
    assert step3_rec["mitosis_phase"] == "clone_pure_rl"

    r1_ckpt = runs_dir / "mitosis_r1_6e" / "checkpoints" / "step_00003.meta.json"
    r1_exp_weights = {
        e: r1_trainer.model.backbone.encoder_layers[0].moe.experts[e].up_proj.weight.detach().clone()
        for e in range(6)
    }

    # Mitosis Round 2: E=6 -> 8 (releases Round-1 pairs [(0,4), (1,5)], freezes 0..5, trains only new clones [6, 7])
    r2_cfg = TrainingRunConfig(
        run_id="mitosis_r2_8e",
        runs_root=str(runs_dir),
        shards_root=str(shards_dir),
        corpora_root=str(tmp_path / "no_corpora"),
        resume_checkpoint=str(r1_ckpt),
        d_model=32,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ple_dim=8,
        expert_hidden_dim=64,
        num_experts=8,
        max_experts=8,
        top_k=4,
        max_seq_len=64,
        projection_dim=8,
        preserve_residual_across_stacks=True,
        donor_backbone_lr_scale=0.001,
        freeze_donor_embeddings=True,
        freeze_non_clone_experts=True,
        mitotic_perturbation_scale=5e-3,
        clone_expert_lr_scale=1.0,
        clone_warmup_steps=1,
        pure_rl_after_clone_warmup=True,
        release_mitotic_pairs=True,
        mitotic_ortho_coef=0.04,
        enable_ornith_rl=True,
        ornith_every=1,
        ornith_weight=0.25,
        lr=1e-3,
        total_steps=5,
        batch_size=2,
        micro_batch_size=2,
        checkpoint_every=2,
        sample_every=10,
        eval_every=0,
    )
    r2_trainer = MCL4COETrainer(r2_cfg)
    assert r2_trainer._released_mitotic_pairs == [(0, 4), (1, 5)]
    assert r2_trainer._get_model_mitotic_pairs() == [(2, 6), (3, 7)]
    assert r2_trainer._plastic_expert_indices == [6, 7]
    for e in range(6):
        assert not r2_trainer.model.backbone.encoder_layers[0].moe.experts[e].up_proj.weight.requires_grad
    for c in (6, 7):
        assert r2_trainer.model.backbone.encoder_layers[0].moe.experts[c].up_proj.weight.requires_grad
    clone_Birth_6 = r2_trainer.model.backbone.encoder_layers[0].moe.experts[6].up_proj.weight.detach().clone()

    r2_summary = r2_trainer.train()
    assert r2_summary["step"] == 5
    for e in range(6):
        exp_mod = r2_trainer.model.backbone.encoder_layers[0].moe.experts[e]
        assert torch.equal(exp_mod.up_proj.weight.detach(), r1_exp_weights[e])
    assert not torch.equal(
        r2_trainer.model.backbone.encoder_layers[0].moe.experts[6].up_proj.weight.detach(),
        clone_Birth_6,
    )
    assert r2_summary["released_mitotic_pairs"] == [[0, 4], [1, 5]]
    assert r2_summary["mitotic_pairs"] == [[2, 6], [3, 7]]


