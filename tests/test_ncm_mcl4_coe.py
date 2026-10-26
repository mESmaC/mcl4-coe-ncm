"""Unit and integration tests for Next-Concept Modeling (NCM) on `MCL4-COE`."""

from __future__ import annotations

import torch

from mcl4_coe import (
    COEConfig,
    CognitiveOperatorEnsemble,
    DonorAnchoredMultiscaleVQVAE,
    InferenceSessionController,
    MCL4CEDBackbone,
    MCL4CEDConfig,
    MultiscaleResidualQuantizer,
    NCMConfig,
    NCMGeneratorAdapter,
    NCMSystem,
    OrnithSelfImprover,
    PhiResidualConv,
    TiedCodebookHead,
    compute_ncm_level_weights,
    compute_ncm_levels,
    get_tokenizer,
)


def _make_tiny_coe(num_experts: int = 8, top_k: int = 2) -> CognitiveOperatorEnsemble:
    b_cfg = MCL4CEDConfig(
        vocab_size=260,
        pad_id=256,
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        ple_dim=16,
        expert_hidden_dim=128,
        num_experts=num_experts,
        top_k=top_k,
        max_seq_len=128,
    )
    c_cfg = COEConfig(projection_dim=16, recurrence_steps=2)
    return CognitiveOperatorEnsemble(MCL4CEDBackbone(b_cfg), c_cfg)


def test_ncm_level_schedules_and_weights() -> None:
    for sched in ("quadratic", "linear", "sqrt", "log", "exp", "cosine"):
        lvls = compute_ncm_levels(64, num_levels=6, schedule=sched)
        assert len(lvls) == 6
        assert lvls[0] == 1
        assert lvls[-1] == 64
        assert all(lvls[i] < lvls[i + 1] for i in range(len(lvls) - 1))

    manual = compute_ncm_levels(32, manual_levels=(1, 4, 16, 32))
    assert manual == (1, 4, 16, 32)

    w_half = compute_ncm_level_weights(manual, alpha=0.5)
    assert w_half.shape == (4,)
    assert torch.all(w_half > 0)
    # Coarser levels have larger per-position weight when alpha > 0
    assert float(w_half[0].item()) > float(w_half[-1].item())

    # At alpha=1, w_k * ell_k is uniform (1/K) across all levels
    w_one = compute_ncm_level_weights(manual, alpha=1.0)
    ells = torch.tensor(manual, dtype=torch.float32)
    assert torch.allclose(w_one * ells, torch.full((4,), 0.25), atol=1e-5)


def test_multiscale_residual_quantizer_ema_and_kmeans() -> None:
    torch.manual_seed(0)
    cfg = NCMConfig(
        target_length=16,
        num_levels=4,
        level_schedule="quadratic",
        codebook_size=32,
        vq_embed_dim=24,
        fine_drop_prob=0.5,
    )
    vq = MultiscaleResidualQuantizer(cfg)
    vq.train()

    z = torch.randn(4, 16, 24, requires_grad=True)
    samples = z.detach().reshape(-1, 24)
    vq.kmeans_init_level(0, samples, n_iters=5)

    out = vq(z, force_cutoff_level=4)
    assert out.z_hat.shape == (4, 16, 24)
    assert len(out.code_indices) == 4
    assert [t.shape for t in out.code_indices] == [(4, l) for l in cfg.levels]
    assert out.vq_loss.requires_grad
    assert out.global_utilization > 0.0

    # Verify straight-through gradient reaches z
    (out.z_hat.sum() + out.vq_loss).backward()
    assert z.grad is not None and float(z.grad.abs().sum().item()) > 0.0

    # Verify teacher-forced corrupted input builder & single-level step reconstruction
    lvl_inputs, parent_codes, cum_z = vq.build_corrupted_ncm_inputs(
        out.code_indices, full_length=16, corruption_rate=0.25
    )
    assert len(lvl_inputs) == 4
    assert [t.shape for t in lvl_inputs] == [(4, l, 24) for l in cfg.levels]
    assert [p.shape for p in parent_codes] == [(4, l) for l in cfg.levels]
    assert len(cum_z) == 4

    z_next, next_in, next_parent = vq.step_next_level_reconstruction(
        0,
        torch.zeros(4, 16, 24),
        out.code_indices[0],
        full_length=16,
    )
    assert z_next.shape == (4, 16, 24)
    assert next_in is not None and next_in.shape == (4, cfg.levels[1], 24)
    assert next_parent is not None and next_parent.shape == (4, cfg.levels[1])


def test_donor_anchored_vqvae_preserves_donor_weights_and_learns() -> None:
    torch.manual_seed(7)
    coe = _make_tiny_coe()
    donor_emb_before = coe.backbone.token_embedding.weight.detach().clone()

    cfg = NCMConfig(
        target_length=16,
        num_levels=4,
        codebook_size=64,
        vq_embed_dim=32,
        fine_drop_prob=0.0,
    )
    vqvae = DonorAnchoredMultiscaleVQVAE(cfg, backbone=coe.backbone)
    opt = torch.optim.AdamW(vqvae.parameters(), lr=1e-2)

    tokens = torch.randint(32, 120, (3, 16), dtype=torch.long)
    out0 = vqvae(tokens)
    loss0 = float(out0.recon_loss.item())

    for _ in range(12):
        opt.zero_grad(set_to_none=True)
        step_out = vqvae(tokens)
        step_out.total_loss.backward()
        opt.step()

    out_final = vqvae(tokens)
    assert float(out_final.recon_loss.item()) < loss0
    # Donor token_embedding weight is untouched because _lookup_token_embeddings uses torch.no_grad()
    assert torch.equal(coe.backbone.token_embedding.weight, donor_emb_before)

    multiscale_preds = vqvae.reconstruct_multiscale_tokens(tokens)
    assert len(multiscale_preds) == 4
    assert all(p.shape == (3, 16) for p in multiscale_preds)


def test_block_diagonal_parallel_matches_single_level_and_prevents_leakage() -> None:
    torch.manual_seed(11)
    coe = _make_tiny_coe(num_experts=8, top_k=2)
    cfg = NCMConfig(
        target_length=16,
        num_levels=4,
        codebook_size=32,
        vq_embed_dim=24,
        scale_adaptive_top_k=False,
    )
    ncm_sys = coe.create_ncm_system(cfg)
    coe.eval()

    prompt_ids = torch.randint(32, 120, (2, 12), dtype=torch.long)
    target_ids = torch.randint(32, 120, (2, 16), dtype=torch.long)
    p_valid = torch.ones_like(prompt_ids, dtype=torch.bool)
    encoded_prompt = coe.backbone.encode_state(prompt_ids, p_valid, causal=True)

    codes = ncm_sys.vqvae.encode(target_ids)
    lvl_inputs, parent_codes, _ = ncm_sys.vqvae.vq.build_corrupted_ncm_inputs(
        codes, full_length=16, corruption_rate=0.0
    )

    logits_parallel, _, _, _ = ncm_sys.generator.forward_all_levels_parallel(
        encoded_prompt, lvl_inputs, parent_codes
    )

    # Perturb level 3 input and verify levels 0, 1, 2 logits in parallel pass are 100% unchanged (zero cross-level leakage)
    lvl_inputs_perturbed = [t.clone() for t in lvl_inputs]
    lvl_inputs_perturbed[3] = torch.randn_like(lvl_inputs_perturbed[3]) * 5.0
    logits_perturbed, _, _, _ = ncm_sys.generator.forward_all_levels_parallel(
        encoded_prompt, lvl_inputs_perturbed, parent_codes
    )
    for k in range(3):
        assert torch.allclose(logits_parallel[k], logits_perturbed[k], atol=1e-5)


def test_ncm_system_training_generation_early_exit_and_cfg() -> None:
    torch.manual_seed(21)
    coe = _make_tiny_coe(num_experts=8, top_k=2)
    cfg = NCMConfig(
        target_length=16,
        num_levels=4,
        codebook_size=48,
        vq_embed_dim=24,
        scale_adaptive_top_k=True,
        coarse_top_k=6,
        mid_top_k=4,
        fine_top_k=2,
        coarse_level_cutoff=1,
        mid_level_cutoff=3,
        coarse_branch_candidates=2,
        metacognition_halt_threshold=0.01,  # Trigger early exit when enable_early_exit=True
        min_early_exit_level=2,
    )
    ncm_sys = coe.create_ncm_system(cfg)
    assert coe.ncm_system is ncm_sys

    # Verify scale-adaptive MoE top_k schedule
    assert [ncm_sys.generator.get_scale_top_k(k) for k in range(4)] == [6, 4, 4, 2]

    prompt_ids = torch.randint(32, 120, (2, 10), dtype=torch.long)
    target_ids = torch.randint(32, 120, (2, 16), dtype=torch.long)

    opt = torch.optim.AdamW(coe.parameters(), lr=3e-3)
    loss_initial = None
    loss_final = None
    for step in range(6):
        opt.zero_grad(set_to_none=True)
        train_out = ncm_sys.forward_training_step(prompt_ids, target_ids)
        if step == 0:
            loss_initial = float(train_out.total_loss.item())
        loss_final = float(train_out.total_loss.item())
        train_out.total_loss.backward()
        opt.step()

    assert loss_initial is not None and loss_final is not None
    assert loss_final < loss_initial

    # Exact single-pass hierarchical log-probabilities (Eq. 13)
    hier_lp, per_lvl_lp = ncm_sys.compute_hierarchy_log_probs(prompt_ids, target_ids)
    assert hier_lp.shape == (2,)
    assert len(per_lvl_lp) == 4
    assert torch.isfinite(hier_lp).all()

    # Full K-level generation with mid-hierarchy COE + CFG
    coe.eval()
    tokenizer = get_tokenizer(260)
    gen_full = ncm_sys.generate(
        prompt_ids[:1],
        tokenizer=tokenizer,
        target_length=16,
        temperature=0.0,
        cfg_scale=1.2,
        enable_mid_hierarchy_coe=True,
        enable_early_exit=False,
        decode_all_levels=True,
    )
    assert gen_full.token_ids.shape == (1, 16)
    assert gen_full.levels_executed == 4
    assert not gen_full.halted_early
    assert gen_full.per_level_active_top_k == [6, 4, 4, 2]
    assert len(gen_full.level_decoded_texts) == 4
    assert len(gen_full.per_level_metacognition) == 4
    assert len(gen_full.per_level_deduction) == 4

    # Early-exit generation via Metacognition threshold
    gen_early = ncm_sys.generate(
        prompt_ids[:1],
        tokenizer=tokenizer,
        target_length=16,
        enable_mid_hierarchy_coe=True,
        enable_early_exit=True,
    )
    assert gen_early.halted_early
    assert gen_early.levels_executed == 2


def test_ornith_ncm_self_improvement_and_inference_ui_mode() -> None:
    torch.manual_seed(99)
    coe = _make_tiny_coe(num_experts=6, top_k=2)
    cfg = NCMConfig(
        target_length=16,
        num_levels=3,
        codebook_size=32,
        vq_embed_dim=16,
    )
    ncm_sys = coe.create_ncm_system(cfg)

    improver = OrnithSelfImprover(
        projection_dim=16,
        max_seq_len=96,
        seed=42,
        enable_dream_rsi=True,
        dream_every=1,
    )
    ornith_out = improver.step_ncm_self_improvement(ncm_sys, batch_size=2)
    assert ornith_out.loss.requires_grad
    assert ornith_out.raw_pp >= 0.0
    assert ornith_out.total_pp >= 0.0
    ornith_out.loss.backward()

    # Verify InferenceSessionController "ncm" mode
    ui_ctrl = InferenceSessionController()
    ui_ctrl.load_preset({"preset": "tiny_byte260", "device": "cpu", "dtype": "float32"})
    res = ui_ctrl.generate_turn(
        {
            "prompt": "Compute 3 + 4.",
            "mode": "ncm",
            "max_new_tokens": 32,
        }
    )
    turn = res["turn"]
    assert turn["config_snapshot"]["mode"] == "ncm"
    assert turn["mode_meta"]["levels_executed"] >= 1
    assert len(turn["mode_meta"]["level_decoded_texts"]) == turn["mode_meta"]["levels_executed"]


def test_ncm_coe_head_bridge_all_operators_and_permutation_invariance() -> None:
    torch.manual_seed(42)
    coe = _make_tiny_coe(num_experts=8, top_k=2)

    # Save legacy state_dict without coe_bridge keys to verify backward-compatible loading
    cfg = NCMConfig(
        target_length=16,
        num_levels=4,
        codebook_size=32,
        vq_embed_dim=24,
        enable_coe_head_bridge=True,
        coe_decision_prior_weight=0.10,
    )
    ncm_sys = coe.create_ncm_system(cfg)
    assert hasattr(ncm_sys, "coe_bridge")

    legacy_sd = {
        k: v.clone()
        for k, v in ncm_sys.state_dict().items()
        if not k.startswith("coe_bridge.")
    }
    ncm_sys.load_state_dict(legacy_sd, strict=True)

    # 1. Verify 0-ULP bitwise permutation invariance in forward_decision with NCM bridge active
    coe.eval()
    b, m, t_ctx, t_opt = 2, 4, 14, 12
    ctx_ids = torch.randint(32, 120, (b, t_ctx), dtype=torch.long)
    ctx_lens = torch.tensor([14, 11], dtype=torch.long)
    opt_ids = torch.randint(32, 120, (b, m, t_opt), dtype=torch.long)
    opt_lens = torch.tensor([[12, 10, 9, 11], [11, 8, 12, 10]], dtype=torch.long)

    dec_a = coe.forward_decision(ctx_ids, ctx_lens, opt_ids, opt_lens)
    perm = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    inv_perm = torch.argsort(perm)
    dec_b = coe.forward_decision(
        ctx_ids, ctx_lens, opt_ids[:, perm, :], opt_lens[:, perm]
    )
    assert torch.equal(dec_a.similarities, dec_b.similarities[:, inv_perm])
    assert torch.equal(dec_a.logits, dec_b.logits[:, inv_perm])

    # 2. Verify all individual COE heads (`Deduction`, `Metacognition`, `Evaluation`, `Abstraction`, `Association`, `Induction`, `Abduction`)
    p_ids = ctx_ids
    p_lens = ctx_lens
    c_ids = opt_ids[:, 0, :]
    c_lens = opt_lens[:, 0]

    ded_out = coe.forward_deduction(p_ids, p_lens, c_ids, c_lens)
    assert ded_out.probabilities.shape == (b, 3)
    assert torch.allclose(ded_out.probabilities.sum(dim=-1), torch.ones(b), atol=1e-5)

    meta_out = coe.forward_metacognition(p_ids, p_lens, c_ids, c_lens)
    assert meta_out.probabilities.shape == (b, 3)

    eval_out = coe.forward_evaluation(p_ids, p_lens, c_ids, c_lens)
    assert eval_out.utilities.shape == (b,)

    ind_out = coe.forward_induction(ctx_ids, ctx_lens, opt_ids, opt_lens)
    abd_out = coe.forward_abduction(ctx_ids, ctx_lens, opt_ids, opt_lens)
    assert ind_out.probabilities.shape == (b, m)
    assert abd_out.probabilities.shape == (b, m)

    abs_brief = coe.compress_abstraction(p_ids, p_lens, level="brief")
    abs_std = coe.compress_abstraction(p_ids, p_lens, level="standard")
    abs_det = coe.compress_abstraction(p_ids, p_lens, level="detailed")
    assert abs_brief.shape == (b, 64)
    assert not torch.allclose(abs_brief, abs_det)
    assert not torch.allclose(abs_brief, abs_std)

    assoc_out = coe.forward_association_matrix(p_ids, p_lens)
    assert assoc_out.similarities.shape == (b, b)

    # 3. Verify `forward_all_operators` trains `coe_bridge` parameters end-to-end
    coe.train()
    coe.zero_grad(set_to_none=True)
    ncm_sys.zero_grad(set_to_none=True)
    targets = torch.tensor([1, 2], dtype=torch.long)
    all_op_out = coe.forward_all_operators(ctx_ids, ctx_lens, opt_ids, opt_lens, targets)
    all_op_out.total_loss.backward()

    assert ncm_sys.coe_bridge.state_multi_proj.weight.grad is not None
    assert float(ncm_sys.coe_bridge.state_multi_proj.weight.grad.abs().sum().item()) > 0.0
    assert ncm_sys.coe_bridge.deduction_down.weight.grad is not None
    assert float(ncm_sys.coe_bridge.deduction_down.weight.grad.abs().sum().item()) > 0.0
    assert ncm_sys.coe_bridge.meta_down.weight.grad is not None
    assert float(ncm_sys.coe_bridge.meta_down.weight.grad.abs().sum().item()) > 0.0
    assert ncm_sys.coe_bridge.abstraction_vq_proj["brief"].weight.grad is not None
    assert float(ncm_sys.coe_bridge.abstraction_vq_proj["brief"].weight.grad.abs().sum().item()) > 0.0

