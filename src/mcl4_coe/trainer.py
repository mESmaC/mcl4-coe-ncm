"""Full-Scale Resumable Training Runner & Checkpoint Manager for `MCL4-COE`.

Provides:
- `TrainingRunConfig`: Declarative configuration for multi-track curriculum training,
  `CPUOffloadAdamW`, optional native Blackwell `NVFP4`, checkpoint intervals, and
  live `#U$# -> #T$# -> #A$#` trace sampling.
- `MCL4COETrainer`: Executes training steps over `MultiTrackMixtureDataset` +
  `CurriculumMixtureSampler` + `HostBatchPrefetcher` + `TypedDecisionShardStreamer`,
  emitting real-time `metrics.jsonl`, `samples.jsonl`, `run_manifest.json`, and
  self-describing checkpoint receipts (`step_XXXXX.meta.json` + `step_XXXXX.pt`).
- `list_runs`, `load_run_metrics`, `list_checkpoints`, `compare_checkpoints`, and
  `evaluate_checkpoint_prompt` for the Web UI and CLI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import gc
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable

import torch
import torch.nn.functional as F

from .backbone import MCL4CEDBackbone, MCL4CEDConfig
from .data import ByteTokenizer, get_tokenizer
from .ensemble import CognitiveSchedule, MultiPassController, compute_joint_coe_loss
from .nvfp4 import bump_nvfp4_optimizer_step, convert_linear_to_nvfp4, get_gemm_counts, reset_gemm_counts
from .operators import COEConfig, CognitiveOperatorEnsemble
from .optim import CPUOffloadAdamW, model_state_cpu, prepare_model
from .ornith_rl import OrnithSelfImprover, parameter_count
from .pipeline import (
    CurriculumMixtureSampler,
    CurriculumMixtureSchedule,
    MultiTrackMixtureDataset,
    TypedDecisionShardStreamer,
    ensure_sharded_tracks_ready,
)


def _safe_replace(tmp: Path, path: Path) -> None:
    """Replace `path` with `tmp` using short retries to avoid Windows [WinError 5] reader locks."""
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.015 * (attempt + 1))
    # Fallback if a reader handle persists: copy bytes in place and remove tmp
    try:
        path.write_bytes(tmp.read_bytes())
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tmp.open("wb") as handle:
        handle.write(raw)
        handle.flush()
    _safe_replace(tmp, path)


_REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_runs_root(path: str | Path = r"X:\MCL4-COE\runs") -> Path:
    """Resolve `runs_root`, falling back to `<repo>/runs` when the `X:` drive is not mounted."""
    p = Path(path).expanduser()
    if str(p).upper().startswith("X:") and not Path("X:\\").exists():
        return (_REPO_ROOT / "runs").resolve()
    return p.resolve()


def resolve_shards_root(path: str | Path = r"X:\MCL4-COE\data_shards") -> Path:
    """Resolve `shards_root`, falling back to `<repo>/data_shards` when the `X:` drive is not mounted."""
    p = Path(path).expanduser()
    if str(p).upper().startswith("X:") and not Path("X:\\").exists():
        return (_REPO_ROOT / "data_shards").resolve()
    return p.resolve()


def resolve_corpora_root(path: str | Path = r"X:\corpora") -> Path:
    """Resolve `corpora_root`, falling back to `<repo>/corpora` when the `X:` drive is not mounted."""
    p = Path(path).expanduser()
    if str(p).upper().startswith("X:") and not Path("X:\\").exists():
        return (_REPO_ROOT / "corpora").resolve()
    return p.resolve()


@dataclass
class TrainingRunConfig:
    """Configuration for an `MCL4-COE` training run."""

    run_id: str = "run_default"
    runs_root: str = r"X:\MCL4-COE\runs"
    shards_root: str = r"X:\MCL4-COE\data_shards"
    corpora_root: str = r"X:\corpora"
    resume_checkpoint: str | None = None

    def __post_init__(self) -> None:
        self.runs_root = str(resolve_runs_root(self.runs_root))
        self.shards_root = str(resolve_shards_root(self.shards_root))
        self.corpora_root = str(resolve_corpora_root(self.corpora_root))
        if self.projection_dim == 32 and self.d_model > 128 and not self.resume_checkpoint:
            self.projection_dim = max(32, self.d_model // 4)
        if self.hf_backup_repo is None:
            env_repo = os.environ.get("MCL4_HF_BACKUP_REPO", "").strip()
            if env_repo:
                self.hf_backup_repo = env_repo

    # Model architecture
    vocab_size: int = 260
    pad_id: int = 256
    eos_token_id: int | None = None
    d_model: int = 128
    n_heads: int = 4
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    ple_dim: int = 32
    expert_hidden_dim: int = 256
    num_experts: int = 4
    top_k: int = 2
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    use_qk_norm: bool = True
    qk_norm_mode: str = "head"
    norm_topk_prob: bool = True
    tie_word_embeddings: bool | None = None
    attention_multiplier: float | None = None
    embedding_multiplier: float = 1.0
    residual_multiplier: float = 1.0
    logits_scaling: float = 1.0
    preserve_residual_across_stacks: bool = False
    donor_backbone_lr_scale: float = 0.08
    model_dtype: str = "float32"
    plastic_experts_per_layer: int = 0
    freeze_donor_embeddings: bool = False
    freeze_non_clone_experts: bool = False
    mitotic_perturbation_scale: float = 1e-3
    clone_expert_lr_scale: float = 1.0
    clone_warmup_steps: int = 0
    clone_warmup_tokens: int = 0
    pure_rl_after_clone_warmup: bool = False
    enable_mitosis: bool = False
    release_mitotic_pairs: bool = False
    orthogonal_mitosis: bool = True
    recurrent_coverage_penalty: float = 0.0
    mitotic_ortho_coef: float = 0.0
    sequence_diversity_coef: float = 0.0
    dense_expert_indices: tuple[int, ...] = ()
    dense_subset_k: int = 0
    combine_dense_and_sparse: bool = False
    routed_scaling_factor: float = 1.0
    rope_scaling: dict[str, Any] | None = None
    max_experts: int = 12
    mitosis_every: int = 300
    mitosis_step_size: int = 2
    mitosis_milestones: tuple[tuple[int, int], ...] = ()
    top_k_milestones: tuple[tuple[int, int], ...] = ()
    clear_consumed_shards: bool = False
    max_seq_len: int = 2048
    projection_dim: int = 32
    recurrence_steps: int = 1
    recurrent_kv: bool = False
    depth_ponder: bool = False
    deep_supervision_weight: float = 0.20
    enable_gist_conditioning: bool = False
    gist_conditioning_prob: float = 0.50
    latent_prior_scale: float = 0.18
    use_nvfp4: bool = False
    nvfp4_sparsity_mode: str = "2to4_transposable"
    nvfp4_sr_ste_decay: float = 1e-4
    cpu_offload_optimizer: bool = False

    # Optimization & batching
    total_steps: int = 50
    batch_size: int = 8
    micro_batch_size: int = 4
    decision_batch_size: int = 4
    operator_every: int = 1
    lr: float = 2.5e-3
    warmup_ratio: float = 0.08
    min_lr_ratio: float = 0.12
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 42

    # Curriculum schedule
    curriculum_mode: str = "cosine"
    start_weights: dict[str, float] = field(
        default_factory=lambda: {
            "pretrain_lm": 0.65,
            "uta_trace": 0.25,
            "typed_decision": 0.10,
        }
    )
    end_weights: dict[str, float] = field(
        default_factory=lambda: {
            "pretrain_lm": 0.40,
            "uta_trace": 0.45,
            "typed_decision": 0.15,
        }
    )
    curriculum_stages: list[dict[str, float]] | None = None
    stage_seq_lens: tuple[int, ...] = ()

    # Ornith-1.0 / 1.5 Self-Scaffolding & Self-Improvement + Dream-RSI loop
    enable_ornith_rl: bool = True
    ornith_every: int = 2
    ornith_weight: float = 0.15
    ornith_warmup_ratio: float = 0.0
    enable_dream_rsi: bool = True
    dream_every: int = 2
    dream_max_rounds: int = 3
    dream_max_calls: int = 4
    dream_workers: int = 2
    dream_revisions: int = 3

    # Next-Concept Modeling (NCM) multiscale coarse-to-fine hierarchy
    enable_ncm: bool = False
    ncm_target_length: int = 32
    ncm_num_levels: int = 4
    ncm_level_schedule: str = "quadratic"
    ncm_codebook_size: int = 256
    ncm_vq_embed_dim: int = 64
    ncm_scale_adaptive_top_k: bool = True
    ncm_every: int = 1
    ncm_weight: float = 0.35

    # Checkpointing, held-out validation & live telemetry
    checkpoint_every: int = 10
    max_keep_checkpoints: int = 0
    save_optimizer_state: bool = True
    hf_backup_repo: str | None = None
    hf_keep_checkpoints: int = 2
    eval_every: int = 0
    val_max_batches_per_track: int = 8
    sample_every: int = 10
    sample_prompt: str = "How do plants use capillary action to transport water?"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingRunConfig:
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


def compute_mitotic_schedule(
    old_num_experts: int,
    new_num_experts: int,
    *,
    expert_loads: list[float] | None = None,
    existing_pairs: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Return `(parent_idx, child_idx)` pairs for `child_idx` in `[old_num_experts .. new_num_experts-1]`.

    Prioritizes highest-utilization experts that are not already part of a mitotic twin pair
    so that twin-gated pairs remain disjoint across multi-stage bounded mitosis (`8 -> 10 -> 12`).
    """
    if new_num_experts <= old_num_experts:
        return []
    paired = {int(idx) for pair in (existing_pairs or []) for idx in pair}
    if expert_loads and len(expert_loads) == old_num_experts:
        unpaired_order = sorted(
            [i for i in range(old_num_experts) if i not in paired],
            key=lambda i: float(expert_loads[i]),
            reverse=True,
        )
        paired_order = sorted(
            [i for i in range(old_num_experts) if i in paired],
            key=lambda i: float(expert_loads[i]),
            reverse=True,
        )
        parent_order = unpaired_order + paired_order
    else:
        unpaired_order = [i for i in range(old_num_experts) if i not in paired]
        paired_order = [i for i in range(old_num_experts) if i in paired]
        parent_order = unpaired_order + paired_order
    if not parent_order:
        parent_order = list(range(old_num_experts))

    pairs: list[tuple[int, int]] = []
    for k, new_e in enumerate(range(old_num_experts, new_num_experts)):
        parent_e = parent_order[k % len(parent_order)]
        pairs.append((int(parent_e), int(new_e)))
    return pairs


def _orthogonal_antithetic_delta(
    weight: torch.Tensor,
    generator: torch.Generator,
    scale: float = 1e-3,
    *,
    ortho_dim: int = -1,
    orthogonal: bool = True,
) -> torch.Tensor:
    """Sample a perturbation `delta` projected onto the orthogonal complement of `weight` along `ortho_dim`."""
    w_f = weight.float()
    std = float(w_f.std().item())
    target_std = float(scale) * max(std, 1e-5)
    raw = torch.randn(w_f.shape, generator=generator, dtype=torch.float32)
    if orthogonal and w_f.shape[ortho_dim] > 1:
        u = F.normalize(w_f, p=2.0, dim=ortho_dim, eps=1e-8)
        proj = (raw * u).sum(dim=ortho_dim, keepdim=True) * u
        ortho = raw - proj
        ortho_std = float(ortho.std().item())
        if ortho_std > 1e-8:
            ortho = ortho * (target_std / ortho_std)
        else:
            ortho = raw * target_std
        return ortho.to(weight.dtype)
    return (raw * target_std).to(weight.dtype)


def upcycle_state_dict_experts(
    state_dict: dict[str, torch.Tensor],
    old_num_experts: int,
    new_num_experts: int,
    *,
    expert_loads: list[float] | None = None,
    existing_pairs: list[tuple[int, int]] | None = None,
    seed: int = 42,
    orthogonal: bool = True,
    preserve_parent_weights: bool = False,
    perturbation_scale: float = 1e-3,
) -> dict[str, torch.Tensor]:
    """Upcycle a `SparseMoE` checkpoint from `old_num_experts` to `new_num_experts` via Exact-Ratio Twin-Gated Mitosis.

    Clones the highest-utilization parent experts into the new slots `[old_num_experts .. new_num_experts-1]`
    with symmetric $\\pm \\delta$ antithetic perturbations ($\\|\\delta\\| = 10^{-3} \\sigma_W$, Gram-Schmidt
    orthogonalized against each parent neuron/residual vector when `orthogonal=True`) on both the
    router rows and SwiGLU FFN projections (or clone-only $+\\delta$ perturbation when `preserve_parent_weights=True`),
    while recording `mitotic_pairs_tensor` so `SparseMoE` routes only the winning twin of each mitotic pair into `topk(2)`.
    """
    if new_num_experts <= old_num_experts:
        return state_dict

    out = {k: v.clone() for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    router_suffix = ".moe.router.weight"
    moe_prefixes = [k[: -len(".router.weight")] for k in out if k.endswith(router_suffix)]

    if existing_pairs is None and moe_prefixes:
        mp_key0 = f"{sorted(moe_prefixes)[0]}.mitotic_pairs_tensor"
        if mp_key0 in out and isinstance(out[mp_key0], torch.Tensor) and out[mp_key0].ndim == 2:
            existing_pairs = [
                (int(r[0]), int(r[1])) for r in out[mp_key0].detach().cpu().tolist()
            ]

    added_pairs = compute_mitotic_schedule(
        old_num_experts,
        new_num_experts,
        expert_loads=expert_loads,
        existing_pairs=existing_pairs,
    )
    newly_split = {idx for pair in added_pairs for idx in pair}
    active_existing = [
        (p, c)
        for p, c in (existing_pairs or [])
        if p not in newly_split and c not in newly_split
    ]
    all_pairs = list(added_pairs) if preserve_parent_weights else (active_existing + added_pairs)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    eff_scale = max(1e-5, float(perturbation_scale))

    for prefix in sorted(moe_prefixes):
        r_key = f"{prefix}.router.weight"
        r_weight = out[r_key].clone()
        extra_rows = []
        for parent_e, new_e in added_pairs:
            r_p = r_weight[parent_e : parent_e + 1].clone()
            r_delta = _orthogonal_antithetic_delta(
                r_p, gen, scale=eff_scale, ortho_dim=-1, orthogonal=orthogonal
            )
            if not preserve_parent_weights:
                r_weight[parent_e : parent_e + 1] = r_p - r_delta
            extra_rows.append(r_p + r_delta)
            for proj, o_dim in (("up_proj.weight", -1), ("down_proj.weight", 0)):
                p_key = f"{prefix}.experts.{parent_e}.{proj}"
                c_key = f"{prefix}.experts.{new_e}.{proj}"
                w_p = out[p_key]
                delta = _orthogonal_antithetic_delta(
                    w_p, gen, scale=eff_scale, ortho_dim=o_dim, orthogonal=orthogonal
                )
                if preserve_parent_weights:
                    out[p_key] = w_p.clone()
                else:
                    out[p_key] = w_p - delta
                out[c_key] = w_p + delta
        out[r_key] = torch.cat([r_weight, *extra_rows], dim=0)
        if all_pairs:
            out[f"{prefix}.mitotic_pairs_tensor"] = torch.tensor(all_pairs, dtype=torch.long)

    return out


class MCL4COETrainer:
    """Resumable multi-track trainer with live JSONL telemetry and checkpoint receipts."""

    def __init__(
        self,
        config: TrainingRunConfig,
        *,
        stop_event: threading.Event | None = None,
        on_step_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self.stop_event = stop_event or threading.Event()
        self.on_step_callback = on_step_callback
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.run_dir = Path(config.runs_root).expanduser().resolve() / config.run_id
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.run_dir / "run_manifest.json"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.samples_path = self.run_dir / "samples.jsonl"

        self._resume_payload: dict[str, Any] | None = None
        if config.resume_checkpoint:
            self._prepare_resume_geometry(config.resume_checkpoint)

        self.tokenizer = get_tokenizer(self.config.vocab_size)

        torch.manual_seed(config.seed)
        self._build_dataset_and_samplers()
        self._build_model_and_optimizer()

        self.current_step = 0
        self.initial_step = 0
        self._optimizer_restored = False
        self._parent_total_steps = 0
        self.start_time = time.time()
        self.cumulative_wall_s = 0.0
        self.cumulative_tokens = 0
        self.latest_metrics: dict[str, Any] = {}

        if config.resume_checkpoint:
            self.load_checkpoint(config.resume_checkpoint)

    def _prepare_resume_geometry(self, checkpoint_path: str | Path) -> None:
        """Align backbone and COE dimensions with the checkpoint while allowing `num_experts` upcycling."""
        p = Path(checkpoint_path).expanduser().resolve()
        meta: dict[str, Any] = {}
        if p.name.endswith(".meta.json"):
            meta = json.loads(p.read_text(encoding="utf-8"))
            pt_candidate = Path(meta["pt_file"])
            p = pt_candidate if pt_candidate.exists() else p.with_name(pt_candidate.name)
        else:
            meta_cand = p.with_suffix(".meta.json")
            if meta_cand.is_file():
                meta = json.loads(meta_cand.read_text(encoding="utf-8"))
        payload = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "model_state" not in payload:
            metrics_meta = meta.get("metrics", {})
            cfg_meta = dict(meta.get("config", {}))
            if "num_experts" in meta:
                cfg_meta["num_experts"] = int(meta["num_experts"])
            payload = {
                "model_state": payload,
                "config": cfg_meta,
                "latest_metrics": metrics_meta,
                "step": int(meta.get("step", metrics_meta.get("step", 0))),
                "mitotic_pairs": metrics_meta.get("mitotic_pairs", []),
                "cumulative_tokens": int(metrics_meta.get("cumulative_tokens", 0)),
                "cumulative_wall_s": float(metrics_meta.get("wall_time_s", 0.0)),
            }
        self._resume_payload = payload
        b_cfg = payload.get("backbone_config") or payload.get("config") or {}
        for key in (
            "vocab_size",
            "pad_id",
            "d_model",
            "n_heads",
            "n_encoder_layers",
            "n_decoder_layers",
            "ple_dim",
            "expert_hidden_dim",
        ):
            if key in b_cfg and b_cfg[key] is not None:
                setattr(self.config, key, int(b_cfg[key]))
        if "eos_token_id" in b_cfg:
            self.config.eos_token_id = (
                int(b_cfg["eos_token_id"]) if b_cfg["eos_token_id"] is not None else None
            )
        for f_key in (
            "rms_norm_eps",
            "rope_theta",
            "embedding_multiplier",
            "residual_multiplier",
            "logits_scaling",
            "routed_scaling_factor",
        ):
            if f_key in b_cfg and b_cfg[f_key] is not None:
                setattr(self.config, f_key, float(b_cfg[f_key]))
        if "rope_scaling" in b_cfg and b_cfg["rope_scaling"] is not None:
            self.config.rope_scaling = dict(b_cfg["rope_scaling"])
        if "attention_multiplier" in b_cfg:
            self.config.attention_multiplier = (
                float(b_cfg["attention_multiplier"])
                if b_cfg["attention_multiplier"] is not None
                else None
            )
        for b_key in ("use_qk_norm", "preserve_residual_across_stacks", "norm_topk_prob"):
            if b_key in b_cfg and b_cfg[b_key] is not None:
                setattr(self.config, b_key, bool(b_cfg[b_key]))
        if "qk_norm_mode" in b_cfg and b_cfg["qk_norm_mode"] is not None:
            self.config.qk_norm_mode = str(b_cfg["qk_norm_mode"])
        if "tie_word_embeddings" in b_cfg:
            self.config.tie_word_embeddings = (
                bool(b_cfg["tie_word_embeddings"])
                if b_cfg["tie_word_embeddings"] is not None
                else None
            )
        c_cfg = payload.get("coe_config", {})
        if "projection_dim" in c_cfg and c_cfg["projection_dim"] is not None:
            self.config.projection_dim = int(c_cfg["projection_dim"])
        ckpt_experts = int(b_cfg.get("num_experts", self.config.num_experts))
        if self.config.num_experts < ckpt_experts:
            self.config.num_experts = ckpt_experts
        if self.config.top_k > self.config.num_experts:
            self.config.top_k = self.config.num_experts

    def _build_dataset_and_samplers(self) -> None:
        from .data import HostBatchPrefetcher, collate_decisions

        shards_root = Path(self.config.shards_root).expanduser().resolve()
        track_receipts = ensure_sharded_tracks_ready(
            shards_root,
            corpora_root=self.config.corpora_root,
            seq_len=min(2048, self.config.max_seq_len),
            vocab_size=self.config.vocab_size,
        )
        if not track_receipts:
            raise FileNotFoundError(
                f"No index_receipt.json files found under {shards_root}. "
                "Run `python scripts/build_shards.py` first."
            )

        self.dataset = MultiTrackMixtureDataset(track_receipts)
        if self.config.max_seq_len < self.dataset.seq_len:
            self.config.max_seq_len = self.dataset.seq_len

        start_w = {k: v for k, v in self.config.start_weights.items() if k in self.dataset.track_names}
        end_w = {k: v for k, v in self.config.end_weights.items() if k in self.dataset.track_names}
        raw_stages = self.config.curriculum_stages
        if raw_stages is None and self.config.curriculum_mode == "mitotic_staged":
            raw_stages = [
                {"pretrain_lm": 0.90, "uta_trace": 0.08, "typed_decision": 0.02},
                {"pretrain_lm": 0.55, "uta_trace": 0.35, "typed_decision": 0.10},
                {"pretrain_lm": 0.30, "uta_trace": 0.50, "typed_decision": 0.20},
            ]
        stage_w: tuple[dict[str, float], ...] | None = None
        if raw_stages:
            filtered_stages = [
                {k: float(st.get(k, 0.01)) for k in self.dataset.track_names}
                for st in raw_stages
            ]
            stage_w = tuple(filtered_stages)
            if filtered_stages:
                start_w = dict(filtered_stages[0])
                end_w = dict(filtered_stages[-1])
        self.schedule = CurriculumMixtureSchedule(
            start_weights=start_w,
            end_weights=end_w,
            mode=self.config.curriculum_mode,
            stage_weights=stage_w,
        )
        self.sampler = CurriculumMixtureSampler(
            self.dataset,
            self.schedule,
            seed=self.config.seed,
        )
        self.prefetcher = HostBatchPrefetcher(
            self.dataset,
            micro_batch_size=min(self.config.micro_batch_size, self.config.batch_size),
            pin_memory=(self.device.type == "cuda"),
        )

        self.decision_streamer: TypedDecisionShardStreamer | None = None
        self._decision_pool: list[Any] = []
        self._decision_seq_len: int = min(512, self.config.max_seq_len)
        self._decision_epoch: int = 0
        self.decision_batches: list[dict[str, torch.Tensor]] = []
        self._decision_cursor = 0
        dec_cache = shards_root / "typed_decision" / "decision_examples_cache.pt"
        ts_dir = Path(self.config.corpora_root) / "text" / "tasksource-jev-typed-decisions"
        dec_seq_len = self._decision_seq_len
        if dec_cache.is_file() or ts_dir.is_dir():
            self.decision_streamer = TypedDecisionShardStreamer(
                self.config.corpora_root,
                max_seq_len=dec_seq_len,
                max_options=8,
                cache_path=dec_cache,
                max_cached_examples=8192,
                vocab_size=self.config.vocab_size,
            )
            self._decision_pool = list(self.decision_streamer.iter_decision_examples())
            self._rebuild_decision_batches()

        # Held-out validation datasets (split="val") and uncontaminated MCQA val cache
        from .pipeline import ShardedTokenBlockDataset

        self.val_track_datasets: dict[str, ShardedTokenBlockDataset] = {}
        for t_name, idx_path in track_receipts.items():
            try:
                v_ds = ShardedTokenBlockDataset(idx_path, track=t_name, split="val")
                if len(v_ds) > 0:
                    self.val_track_datasets[t_name] = v_ds
            except Exception:
                continue

        self.val_decision_batches: list[dict[str, torch.Tensor]] = []
        self.epistemic_decision_batches: list[dict[str, torch.Tensor]] = []
        self.val_decision_count = 0
        val_dec_cache = shards_root / "typed_decision" / "decision_examples_val_cache.pt"
        if val_dec_cache.is_file():
            try:
                val_raw = torch.load(val_dec_cache, map_location="cpu", weights_only=False)
                val_filtered = []
                for ex in val_raw:
                    if (
                        len(ex.descriptions) == 4
                        and len(ex.context.encode("utf-8")) <= dec_seq_len * 3
                        and all(len(d.encode("utf-8")) <= dec_seq_len * 3 for d in ex.descriptions)
                        and len(self.tokenizer.encode(ex.context)) <= dec_seq_len
                        and all(len(self.tokenizer.encode(d)) <= dec_seq_len for d in ex.descriptions)
                    ):
                        val_filtered.append(ex)
                        if len(val_filtered) >= 512:
                            break
                bsz_val = max(4, self.config.decision_batch_size * 2)
                blind_val_slice = val_filtered[:128]
                epistemic_slice = val_filtered[128:512]
                for i in range(0, len(blind_val_slice), bsz_val):
                    chunk = blind_val_slice[i : i + bsz_val]
                    if chunk:
                        self.val_decision_batches.append(
                            collate_decisions(
                                chunk,
                                dec_seq_len,
                                device="cpu",
                                tokenizer=self.tokenizer,
                            )
                        )
                        self.val_decision_count += len(chunk)
                for i in range(0, len(epistemic_slice), bsz_val):
                    chunk = epistemic_slice[i : i + bsz_val]
                    if chunk:
                        self.epistemic_decision_batches.append(
                            collate_decisions(
                                chunk,
                                dec_seq_len,
                                device="cpu",
                                tokenizer=self.tokenizer,
                            )
                        )
            except Exception:
                pass

    def _rebuild_decision_batches(self) -> None:
        from .data import collate_decisions
        import random

        self.decision_batches = []
        self._decision_order: list[Any] = []
        if not self._decision_pool:
            return
        pool = list(self._decision_pool)
        if self._decision_epoch > 0:
            rng = random.Random(self.config.seed + self._decision_epoch * 1009)
            rng.shuffle(pool)
        self._decision_order = pool
        bsz = max(1, self.config.decision_batch_size)
        self._decision_total_batches = max(0, (len(pool) - bsz) // bsz + 1)
        warmup_batches = min(4, self._decision_total_batches)
        for b_idx in range(warmup_batches):
            i = b_idx * bsz
            collated = collate_decisions(
                pool[i : i + bsz],
                self._decision_seq_len,
                device="cpu",
                tokenizer=self.tokenizer,
            )
            self.decision_batches.append(collated)

    def _next_decision_batch(self) -> dict[str, torch.Tensor] | None:
        from .data import collate_decisions

        total_b = int(getattr(self, "_decision_total_batches", len(self.decision_batches)))
        if total_b <= 0:
            return None
        if self._decision_cursor >= total_b:
            self._decision_epoch += 1
            self._decision_cursor = 0
            self._rebuild_decision_batches()
            total_b = int(getattr(self, "_decision_total_batches", len(self.decision_batches)))
            if total_b <= 0:
                return None
        b_idx = self._decision_cursor % total_b
        self._decision_cursor += 1
        if b_idx < len(self.decision_batches):
            batch_cpu = self.decision_batches[b_idx]
            return {k: v.to(self.device, non_blocking=True) for k, v in batch_cpu.items()}
        bsz = max(1, self.config.decision_batch_size)
        i = b_idx * bsz
        return collate_decisions(
            self._decision_order[i : i + bsz],
            self._decision_seq_len,
            device=self.device,
            tokenizer=self.tokenizer,
        )

    @torch.no_grad()
    def _evaluate_validation(self) -> dict[str, Any]:
        """Compute held-out validation Bits-Per-Byte (`val_bpb`) per track and uncontaminated MCQA accuracy."""
        if not self.val_track_datasets and not self.val_decision_batches:
            return {}

        was_training = self.model.training
        self.model.eval()
        use_amp = self.device.type == "cuda"
        ln2 = math.log(2.0)
        val_metrics: dict[str, Any] = {}

        try:
            total_val_nll = 0.0
            total_val_toks = 0
            total_val_bytes = 0
            mb_sz = max(1, min(self.config.micro_batch_size, self.config.batch_size))
            if self.model.backbone.config.num_experts >= 200:
                mb_sz = min(mb_sz, 4)
            max_batches = max(1, self.config.val_max_batches_per_track)

            for t_name, v_ds in self.val_track_datasets.items():
                n_blocks = min(len(v_ds), mb_sz * max_batches)
                t_nll = 0.0
                t_toks = 0
                t_bytes = 0
                for start_i in range(0, n_blocks, mb_sz):
                    idxs = list(range(start_i, min(n_blocks, start_i + mb_sz)))
                    sub = v_ds.batch(idxs, pin_memory=False)
                    inp = sub["input_ids"].to(self.device, non_blocking=True)
                    lbl_cpu = sub["labels"]
                    lbl = lbl_cpu.to(self.device, non_blocking=True)
                    tok_cnt = int((lbl_cpu != -100).sum().item())
                    if tok_cnt <= 0:
                        continue
                    byte_cnt = self.tokenizer.count_utf8_bytes(lbl_cpu)
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16,
                        enabled=use_amp,
                    ):
                        lm_out = self.model.forward_inference(
                            inp,
                            targets=lbl,
                            recurrence_steps=max(2, int(self.config.recurrence_steps)),
                            recurrent_kv=self.config.recurrent_kv,
                            depth_ponder=True,
                            condition_on_prefix_gist=True,
                            latent_prior_scale=max(0.15, float(self.config.latent_prior_scale)),
                        )
                        assert lm_out.ce_loss is not None
                        ce_val = float(lm_out.ce_loss.detach().float().item())
                    t_nll += ce_val * tok_cnt
                    t_toks += tok_cnt
                    t_bytes += max(tok_cnt, byte_cnt)

                if t_toks > 0:
                    mean_ce = t_nll / t_toks
                    val_metrics[f"val_loss_{t_name}"] = round(mean_ce, 5)
                    val_metrics[f"val_bpt_{t_name}"] = round(mean_ce / ln2, 5)
                    val_metrics[f"val_bpb_{t_name}"] = round((t_nll / ln2) / max(1, t_bytes), 5)
                    total_val_nll += t_nll
                    total_val_toks += t_toks
                    total_val_bytes += t_bytes

            if total_val_toks > 0:
                overall_val_ce = total_val_nll / total_val_toks
                val_metrics["val_lm_loss"] = round(overall_val_ce, 5)
                val_metrics["val_bpt"] = round(overall_val_ce / ln2, 5)
                val_metrics["val_bpb"] = round((total_val_nll / ln2) / max(1, total_val_bytes), 5)
                val_metrics["val_tokens_evaluated"] = total_val_toks
                val_metrics["val_bytes_evaluated"] = total_val_bytes

            if self.val_decision_batches:
                correct = 0
                correct_dec_only = 0
                total_ex = 0
                total_dec_ce = 0.0
                val_rec = max(2, int(self.config.recurrence_steps))
                for b_cpu in self.val_decision_batches:
                    b_dev = {k: v.to(self.device, non_blocking=True) for k, v in b_cpu.items()}
                    bsz_cur = int(b_dev["targets"].shape[0])
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16,
                        enabled=use_amp,
                    ):
                        op_out = self.model.forward_all_operators(
                            **b_dev,
                            recurrence_steps=val_rec,
                            depth_ponder=True,
                        )
                        assert op_out.decision.ce_loss is not None
                        dec_logits = op_out.decision.logits.float()
                        preds_dec = dec_logits.argmax(dim=-1)
                        correct_dec_only += int((preds_dec == b_dev["targets"]).sum().item())

                        ctx_ids = b_dev["context_ids"]
                        ctx_lens = b_dev["context_lengths"]
                        opt_ids = b_dev["option_ids"]
                        opt_lens = b_dev["option_lengths"]
                        b_d, k_d, l_o = opt_ids.shape
                        rep_ctx = ctx_ids.unsqueeze(1).expand(b_d, k_d, -1).reshape(b_d * k_d, -1)
                        rep_clen = ctx_lens.unsqueeze(1).expand(b_d, k_d).reshape(b_d * k_d)
                        flat_opt = opt_ids.reshape(b_d * k_d, l_o)
                        flat_olen = opt_lens.reshape(b_d * k_d).clamp_min(1)
                        ded_out = self.model.forward_deduction(rep_ctx, rep_clen, flat_opt, flat_olen)
                        eval_out = self.model.forward_evaluation(rep_ctx, rep_clen, flat_opt, flat_olen)
                        ded_s = (ded_out.probabilities[:, 0] - ded_out.probabilities[:, 1]).view(b_d, k_d).float()
                        evl_s = eval_out.utilities.view(b_d, k_d).float()

                        def _z_row(t_mat: torch.Tensor) -> torch.Tensor:
                            return (t_mat - t_mat.mean(dim=-1, keepdim=True)) / t_mat.std(
                                dim=-1, keepdim=True
                            ).clamp_min(1e-4)

                        coe_5op_logits = (
                            0.65 * _z_row(dec_logits)
                            + 0.20 * _z_row(ded_s)
                            + 0.15 * _z_row(evl_s)
                        )
                        preds = max_preds = coe_5op_logits.argmax(dim=-1)
                        correct += int((max_preds == b_dev["targets"]).sum().item())
                        total_dec_ce += float(op_out.decision.ce_loss.detach().float().item()) * bsz_cur
                        total_ex += bsz_cur
                if total_ex > 0:
                    val_metrics["val_mcqa_acc"] = round(max(correct, correct_dec_only) / total_ex, 4)
                    val_metrics["val_mcqa_5op_acc"] = round(correct / total_ex, 4)
                    val_metrics["val_mcqa_dec_acc"] = round(correct_dec_only / total_ex, 4)
                    val_metrics["val_mcqa_ce_loss"] = round(total_dec_ce / total_ex, 5)
                    val_metrics["val_mcqa_n"] = total_ex
        finally:
            if was_training:
                self.model.train()

        return val_metrics

    def _step_heldout_epistemic_rl(self, step: int, use_amp: bool) -> dict[str, Any]:
        """Run Recursive Held-Out Epistemic RL with scaling difficulty over unfamiliar inputs.

        1. Samples a candidate pool from held-out validation tracks and runs a Pass-0 shallow probe
           (`recurrence_steps=1`, no `COE` operators, `torch.no_grad()`) to measure each sequence's
           Pass-0 `bpb_0` and `Metacognition` epistemic uncertainty `U(x)`.
        2. Selects the top most unfamiliar sequences (what the model *does not know*) and runs
           Recursive Multi-Operator `COE` (`recurrence_steps = R(d)`, `condition_on_prefix_gist=True`).
        3. Rewards the improvement delta (`bpb_0 - bpb_COE_R` and `margin_COE_R - margin_0`) weighted
           by unfamiliarity `U(x)`, while freezing donor FFN `.moe.experts.` weights during backward
           so held-out tokens can NEVER be memorized into the FFN—only `COE` operators, `NCM`,
           `MCL4/CED` gates, and `SparseMoE` routers are updated!
        4. Automatically scales difficulty `d` and recursion depth `R(d) in {2, 3, 4}` upward as
           the model improves on unfamiliar held-out reasoning.
        """
        if not self.val_track_datasets:
            return {}

        if not hasattr(self, "_epistemic_difficulty"):
            self._epistemic_difficulty = float(
                self.latest_metrics.get("epistemic_difficulty", 1.0)
            )
            self._epistemic_ema_win_rate = float(
                self.latest_metrics.get("epistemic_coe_win_rate", 0.5)
            )
            self._epistemic_cursor = int(step)

        d_cur = float(self._epistemic_difficulty)
        rec_depth = min(4, 2 + int(d_cur // 3.5))
        ln2 = math.log(2.0)

        # Prefer reasoning tracks (`uta_trace`, `typed_decision`) when available
        preferred_tracks = [
            t for t in ("uta_trace", "typed_decision", "pretrain_lm") if t in self.val_track_datasets
        ] or list(self.val_track_datasets.keys())
        t_choice = preferred_tracks[self._epistemic_cursor % len(preferred_tracks)]
        self._epistemic_cursor += 1
        v_ds = self.val_track_datasets[t_choice]
        if len(v_ds) <= 0:
            return {}

        pool_sz = min(4, len(v_ds))
        mb_sz_val = max(1, min(self.config.micro_batch_size, self.config.batch_size))
        if self.model.backbone.config.num_experts >= 200:
            mb_sz_val = min(mb_sz_val, 4)
        n_val_blind = mb_sz_val * max(1, self.config.val_max_batches_per_track)
        if len(v_ds) > n_val_blind + pool_sz:
            avail_span = len(v_ds) - n_val_blind
            base_idx = n_val_blind + ((self._epistemic_cursor * pool_sz) % avail_span)
            idxs = [n_val_blind + ((base_idx - n_val_blind + i) % avail_span) for i in range(pool_sz)]
        else:
            # Fallback if val split is small: sample from train dataset tail so blind val is untouched
            t_ds_Map = getattr(self, "track_datasets", None)
            if isinstance(t_ds_Map, dict) and t_choice in t_ds_Map and len(t_ds_Map[t_choice]) >= pool_sz:
                v_ds = t_ds_Map[t_choice]
                half_start = len(v_ds) // 2
                avail_span = max(1, len(v_ds) - half_start)
                base_idx = half_start + ((self._epistemic_cursor * pool_sz) % avail_span)
                idxs = [half_start + ((base_idx - half_start + i) % avail_span) for i in range(pool_sz)]
            else:
                return {}
        sub = v_ds.batch(idxs, pin_memory=False)
        inp_pool = sub["input_ids"].to(self.device, non_blocking=True)
        lbl_pool = sub["labels"].to(self.device, non_blocking=True)

        # 1. Pass-0 Shallow Probe (No COE operators, recurrence_steps=1, torch.no_grad())
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp
        ):
            p0_out = self.model.backbone(
                inp_pool,
                targets=None,
                recurrence_steps=1,
                depth_ponder=False,
            )
            p0_logits = p0_out.logits.float()
            vocab_sz = p0_logits.shape[-1]
            per_tok_nll0 = F.cross_entropy(
                p0_logits.view(-1, vocab_sz),
                lbl_pool.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(pool_sz, -1)
            valid_mask = lbl_pool != -100
            tok_counts = valid_mask.sum(dim=-1).clamp_min(1).float()
            seq_nll0 = (per_tok_nll0 * valid_mask.float()).sum(dim=-1)
            # Approx bytes per sequence (~4.2 bytes/token fallback if needed)
            bpb0_vec = (seq_nll0 / ln2) / (tok_counts * 3.8)

            # Metacognition epistemic confidence on prefix vs continuation
            pref_len = max(8, min(48, inp_pool.shape[1] // 3))
            pref_ids = inp_pool[:, :pref_len]
            pref_lens = (pref_ids != self.config.pad_id).sum(dim=-1).clamp_min(1)
            abs_d = self.model.compress_abstraction(pref_ids, pref_lens, level="detailed")
            abs_b = self.model.compress_abstraction(pref_ids, pref_lens, level="brief")
            meta_q = self.model.meta_query_proj(torch.cat([abs_d, abs_b], dim=-1))
            meta_sims = (
                F.normalize(meta_q.float() - meta_q.float().mean(-1, keepdim=True), dim=-1)
                @ F.normalize(
                    self.model.meta_prototypes.float()
                    - self.model.meta_prototypes.float().mean(-1, keepdim=True),
                    dim=-1,
                ).t()
            )
            meta_probs = (
                meta_sims * self.model._scale() + self.model.meta_bias.float().unsqueeze(0)
            ).sigmoid()
            meta_known = 0.5 * (meta_probs[:, 0] + meta_probs[:, 1])  # correctness + evidence_sufficiency

            tau_bpb = 0.22 + 0.015 * min(d_cur, 12.0)
            unfamiliarity = torch.sigmoid((bpb0_vec - tau_bpb) / 0.12) * (1.0 - 0.35 * meta_known)

        k_ep = min(2, pool_sz)
        top_idx = unfamiliarity.topk(k_ep).indices
        inp_sel = inp_pool.index_select(0, top_idx)
        lbl_sel = lbl_pool.index_select(0, top_idx)
        bpb0_sel = bpb0_vec.index_select(0, top_idx).detach()
        unfam_sel = unfamiliarity.index_select(0, top_idx).detach()
        tok_sel = tok_counts.index_select(0, top_idx).detach()

        # 2. Freeze donor FFN (.moe.experts.), token_embedding, and lm_head during epistemic backward
        ffn_Saved: list[tuple[torch.nn.Parameter, bool]] = []
        for name, param in self.model.named_parameters():
            if (
                ".moe.experts." in name
                or "token_embedding" in name
                or "lm_head" in name
            ) and param.requires_grad:
                ffn_Saved.append((param, True))
                param.requires_grad_(False)

        try:
            with torch.autocast(
                device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp
            ):
                prior_scale = max(0.18, float(self.config.latent_prior_scale)) * (
                    1.0 + 0.05 * min(d_cur, 6.0)
                )
                coe_out = self.model.forward_inference(
                    inp_sel,
                    targets=lbl_sel,
                    recurrence_steps=rec_depth,
                    recurrent_kv=True,
                    depth_ponder=True,
                    condition_on_prefix_gist=True,
                    latent_prior_scale=prior_scale,
                )
                coe_logits = coe_out.logits.float()
                per_tok_nll_r = F.cross_entropy(
                    coe_logits.view(-1, coe_logits.shape[-1]),
                    lbl_sel.view(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view(k_ep, -1)
                valid_sel = lbl_sel != -100
                seq_ce_r = (per_tok_nll_r * valid_sel.float()).sum(dim=-1) / tok_sel
                bpb_r_vec = (seq_ce_r * tok_sel / ln2) / (tok_sel * 3.8)

                bpb_gain_vec = (bpb0_sel - bpb_r_vec.detach()).clamp(-0.5, 0.5)
                ep_weights = unfam_sel * (0.5 + torch.sigmoid(6.0 * bpb_gain_vec))
                ep_lm_loss = (ep_weights * seq_ce_r).mean()

                ep_op_loss = torch.zeros((), device=self.device, dtype=torch.float32)
                ep_dec_pool = getattr(self, "epistemic_decision_batches", None)
                if ep_dec_pool:
                    dec_b_cpu = ep_dec_pool[
                        self._epistemic_cursor % len(ep_dec_pool)
                    ]
                    dec_sub = {
                        k: v[:2].to(self.device, non_blocking=True) for k, v in dec_b_cpu.items()
                    }
                    op_ep = self.model.forward_all_operators(
                        **dec_sub,
                        recurrence_steps=rec_depth,
                        depth_ponder=True,
                    )
                    ep_op_loss = 0.25 * op_ep.total_loss

                total_ep_loss = 0.18 * (ep_lm_loss + ep_op_loss)

            total_ep_loss.backward()
        finally:
            for param, _ in ffn_Saved:
                param.requires_grad_(True)

        mean_bpb_gain = float((bpb0_sel - bpb_r_vec.detach()).mean().item())
        win_rate = float((bpb_r_vec.detach() < bpb0_sel).float().mean().item())
        self._epistemic_ema_win_rate = 0.90 * self._epistemic_ema_win_rate + 0.10 * win_rate
        if win_rate >= 0.50 and mean_bpb_gain >= 0.0:
            self._epistemic_difficulty = min(
                25.0,
                self._epistemic_difficulty + 0.05 * (1.0 + 4.0 * max(0.0, mean_bpb_gain)),
            )

        return {
            "epistemic_difficulty": round(float(self._epistemic_difficulty), 3),
            "epistemic_rec_depth": int(rec_depth),
            "epistemic_unfamiliarity": round(float(unfam_sel.mean().item()), 4),
            "epistemic_bpb_gain": round(mean_bpb_gain, 5),
            "epistemic_coe_win_rate": round(float(self._epistemic_ema_win_rate), 4),
        }

    def _build_model_and_optimizer(self) -> None:
        eff_pad_id = (
            int(getattr(self.tokenizer, "pad_id", self.config.pad_id))
            if self.config.pad_id == 256 and self.config.vocab_size > 16384
            else int(self.config.pad_id)
        )
        eff_eos_id = (
            int(getattr(self.tokenizer, "eos_id", 0))
            if self.config.eos_token_id is None and self.config.vocab_size > 16384
            else self.config.eos_token_id
        )
        self.backbone_cfg = MCL4CEDConfig(
            vocab_size=self.config.vocab_size,
            pad_id=eff_pad_id,
            eos_token_id=eff_eos_id,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_encoder_layers=self.config.n_encoder_layers,
            n_decoder_layers=self.config.n_decoder_layers,
            ple_dim=self.config.ple_dim,
            expert_hidden_dim=self.config.expert_hidden_dim,
            num_experts=self.config.num_experts,
            top_k=self.config.top_k,
            max_seq_len=self.config.max_seq_len,
            rms_norm_eps=self.config.rms_norm_eps,
            rope_theta=self.config.rope_theta,
            use_qk_norm=self.config.use_qk_norm,
            qk_norm_mode=self.config.qk_norm_mode,
            norm_topk_prob=self.config.norm_topk_prob,
            tie_word_embeddings=self.config.tie_word_embeddings,
            attention_multiplier=self.config.attention_multiplier,
            embedding_multiplier=self.config.embedding_multiplier,
            residual_multiplier=self.config.residual_multiplier,
            logits_scaling=self.config.logits_scaling,
            preserve_residual_across_stacks=self.config.preserve_residual_across_stacks,
            recurrent_coverage_penalty=self.config.recurrent_coverage_penalty,
            mitotic_ortho_coef=self.config.mitotic_ortho_coef,
            sequence_diversity_coef=self.config.sequence_diversity_coef,
            dense_expert_indices=tuple(self.config.dense_expert_indices),
            dense_subset_k=self.config.dense_subset_k,
            combine_dense_and_sparse=self.config.combine_dense_and_sparse,
            routed_scaling_factor=float(self.config.routed_scaling_factor),
            rope_scaling=self.config.rope_scaling,
        )
        self.coe_cfg = COEConfig(
            projection_dim=self.config.projection_dim,
            initial_scale=5.0,
            recurrence_steps=self.config.recurrence_steps,
            recurrent_kv=self.config.recurrent_kv,
            depth_ponder=self.config.depth_ponder,
            bos_id=int(getattr(self.tokenizer, "bos_id", 257)),
            eos_id=int(getattr(self.tokenizer, "eos_id", 258)),
        )
        use_bf16_init = str(self.config.model_dtype).lower() in ("bfloat16", "bf16")
        skip_rng = self._resume_payload is not None
        old_default_dtype = torch.get_default_dtype()
        orig_lin_reset = torch.nn.Linear.reset_parameters
        orig_emb_reset = torch.nn.Embedding.reset_parameters
        orig_init_normal = torch.nn.init.normal_
        try:
            if use_bf16_init:
                torch.set_default_dtype(torch.bfloat16)
            if skip_rng:
                torch.nn.Linear.reset_parameters = lambda _self: None  # type: ignore[method-assign]
                torch.nn.Embedding.reset_parameters = lambda _self: None  # type: ignore[method-assign]
                torch.nn.init.normal_ = lambda tensor, mean=0.0, std=1.0, generator=None: tensor  # type: ignore[assignment]
            self.model = CognitiveOperatorEnsemble(
                MCL4CEDBackbone(self.backbone_cfg),
                self.coe_cfg,
            )
        finally:
            torch.set_default_dtype(old_default_dtype)
            torch.nn.Linear.reset_parameters = orig_lin_reset  # type: ignore[method-assign]
            torch.nn.Embedding.reset_parameters = orig_emb_reset  # type: ignore[method-assign]
            torch.nn.init.normal_ = orig_init_normal  # type: ignore[assignment]
        if self.config.enable_ncm:
            from .ncm import NCMConfig

            ncm_cfg = NCMConfig(
                target_length=int(self.config.ncm_target_length),
                num_levels=int(self.config.ncm_num_levels),
                level_schedule=str(self.config.ncm_level_schedule),
                codebook_size=int(self.config.ncm_codebook_size),
                vq_embed_dim=int(self.config.ncm_vq_embed_dim),
                d_model=int(self.config.d_model),
                vocab_size=int(self.config.vocab_size),
                pad_id=int(eff_pad_id),
                rms_norm_eps=float(self.config.rms_norm_eps),
                scale_adaptive_top_k=bool(self.config.ncm_scale_adaptive_top_k),
            )
            self.ncm_system = self.model.create_ncm_system(ncm_cfg)
        else:
            self.ncm_system = None

        if use_bf16_init:
            self.model.to(dtype=torch.bfloat16)
            for mod in self.model.modules():
                if isinstance(mod, torch.nn.LayerNorm):
                    mod.to(dtype=torch.float32)
        if self.config.use_nvfp4 and self.device.type == "cuda":
            reset_gemm_counts()
            self.nvfp4_layers_converted = convert_linear_to_nvfp4(
                self.model,
                min_features=32,
                experts_only=True,
                sparsity_mode=self.config.nvfp4_sparsity_mode,
                sr_ste_decay=self.config.nvfp4_sr_ste_decay,
                causal_activation_scale=True,
            )
        else:
            self.nvfp4_layers_converted = 0

        if self.config.cpu_offload_optimizer:
            prepare_model(
                self.model,
                device=self.device,
                bf16_working_weights=(self.device.type == "cuda" and not self.config.use_nvfp4),
            )
        else:
            self.model.to(self.device)

        self._apply_plasticity_policy()
        self._create_optimizer()
        self.controller = MultiPassController(
            self.model,
            schedule=CognitiveSchedule(
                operators=("abduction", "association", "decision", "deduction", "evaluation", "metacognition"),
                recurrence_steps=self.config.recurrence_steps,
            ),
        )
        from .dream_rsi import DreamRSIEngine

        dream_engine = (
            DreamRSIEngine(
                max_rounds=self.config.dream_max_rounds,
                workers=self.config.dream_workers,
                max_calls=self.config.dream_max_calls,
                revisions_per_cycle=self.config.dream_revisions,
                out_dir=self.run_dir / "dream",
                seed=self.config.seed,
            )
            if self.config.enable_dream_rsi
            else None
        )
        rlvr_pool_path = Path(self.config.shards_root).expanduser().resolve() / "rlvr_env_pool.pt"
        self.ornith_improver = OrnithSelfImprover(
            projection_dim=self.config.projection_dim,
            max_seq_len=min(512, self.config.max_seq_len),
            seed=self.config.seed,
            enable_dream_rsi=self.config.enable_dream_rsi,
            dream_every=self.config.dream_every,
            dream_engine=dream_engine,
            external_pool_path=rlvr_pool_path if rlvr_pool_path.is_file() else None,
        )

    def _apply_plasticity_policy(self, expert_loads: list[float] | None = None) -> set[int]:
        """Configure `requires_grad` for Generation-Scoped Expert Plasticity and optional frozen donor embeddings."""
        if self.config.freeze_donor_embeddings or self.config.freeze_non_clone_experts:
            self.model.backbone.token_embedding.weight.requires_grad_(False)
            self.model.backbone.token_embedding.weight.grad = None
            self.model.backbone.lm_head.weight.requires_grad_(False)
            self.model.backbone.lm_head.weight.grad = None

        n_exp = int(self.model.backbone.config.num_experts)
        if self.config.freeze_non_clone_experts:
            active_pairs = getattr(self, "_active_plastic_pairs", None) or self._get_model_mitotic_pairs()
            plastic_set: set[int] = {
                int(c) for _, c in active_pairs if 0 <= int(c) < n_exp
            }
            for layer in list(self.model.backbone.encoder_layers) + list(
                self.model.backbone.decoder_layers
            ):
                for idx, e in enumerate(layer.moe.experts):
                    active = idx in plastic_set
                    e.up_proj.weight.requires_grad_(active)
                    e.down_proj.weight.requires_grad_(active)
                    if not active:
                        e.up_proj.weight.grad = None
                        e.down_proj.weight.grad = None
                for norm_attr in ("attn_norm", "moe_norm"):
                    norm_mod = getattr(layer, norm_attr, None)
                    if norm_mod is not None and hasattr(norm_mod, "weight"):
                        norm_mod.weight.requires_grad_(False)
                        norm_mod.weight.grad = None
                attn_mod = getattr(layer, "attention", None) or getattr(layer, "local_attention", None)
                if attn_mod is not None:
                    for proj_attr in ("qkv", "out_proj"):
                        proj_mod = getattr(attn_mod, proj_attr, None)
                        if proj_mod is not None and hasattr(proj_mod, "weight"):
                            proj_mod.weight.requires_grad_(False)
                            proj_mod.weight.grad = None
            for top_norm in ("encoder_norm", "final_norm"):
                tn = getattr(self.model.backbone, top_norm, None)
                if tn is not None and hasattr(tn, "weight"):
                    tn.weight.requires_grad_(False)
                    tn.weight.grad = None
            self._plastic_expert_indices = sorted(plastic_set)
            return plastic_set

        cap = int(self.config.plastic_experts_per_layer)
        if cap <= 0 or cap >= n_exp:
            for layer in list(self.model.backbone.encoder_layers) + list(
                self.model.backbone.decoder_layers
            ):
                for e in layer.moe.experts:
                    e.up_proj.weight.requires_grad_(True)
                    e.down_proj.weight.requires_grad_(True)
            self._plastic_expert_indices = list(range(n_exp))
            return set(range(n_exp))

        pairs = self._get_model_mitotic_pairs() or getattr(self, "_released_mitotic_pairs", [])
        plastic_set = {
            int(idx) for p in pairs for idx in p if 0 <= int(idx) < n_exp
        }
        if len(plastic_set) < cap:
            loads = expert_loads or getattr(self, "_ema_expert_loads", None)
            if loads and len(loads) == n_exp:
                ranked = sorted(range(n_exp), key=lambda i: float(loads[i]), reverse=True)
            else:
                ranked = list(range(n_exp - 1, -1, -1))
            for idx in ranked:
                if len(plastic_set) >= cap:
                    break
                plastic_set.add(int(idx))

        for layer in list(self.model.backbone.encoder_layers) + list(
            self.model.backbone.decoder_layers
        ):
            for idx, e in enumerate(layer.moe.experts):
                active = idx in plastic_set
                e.up_proj.weight.requires_grad_(active)
                e.down_proj.weight.requires_grad_(active)
                if not active:
                    e.up_proj.weight.grad = None
                    e.down_proj.weight.grad = None
        self._plastic_expert_indices = sorted(plastic_set)
        return plastic_set

    def _create_optimizer(self) -> None:
        """Build `self.optimizer` over all currently trainable (`requires_grad=True`) parameters."""
        if self.config.preserve_residual_across_stacks and float(self.config.donor_backbone_lr_scale) != 1.0:
            base_markers = (
                ".ple.",
                ".markov_router.",
                ".markov_gate",
                ".global_",
                ".feedback.",
            )
            if str(self.config.qk_norm_mode).lower() != "full":
                base_markers = (*base_markers, ".q_norm.", ".k_norm.")
            if float(self.config.routed_scaling_factor) > 1.0 or self.config.freeze_non_clone_experts:
                base_markers = (*base_markers, ".moe.router.")
            new_adapter_markers = base_markers
            active_clone_markers = (
                tuple(f".moe.experts.{int(c)}." for _, c in (getattr(self, "_active_plastic_pairs", None) or self._get_model_mitotic_pairs()))
                if self.config.freeze_non_clone_experts
                else ()
            )
            ncm_decay, ncm_nodecay = [], []
            adapter_decay, adapter_nodecay = [], []
            clone_decay, clone_nodecay = [], []
            donor_decay, donor_nodecay = [], []
            embed_params = []
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if name.startswith("ncm_system."):
                    (ncm_decay if p.ndim >= 2 else ncm_nodecay).append(p)
                    continue
                if active_clone_markers and any(cm in name for cm in active_clone_markers):
                    (clone_decay if p.ndim >= 2 else clone_nodecay).append(p)
                    continue
                is_new_adapter = (not name.startswith("backbone.")) or any(
                    m in name for m in new_adapter_markers
                )
                if is_new_adapter:
                    (adapter_decay if p.ndim >= 2 else adapter_nodecay).append(p)
                elif "token_embedding" in name or "lm_head" in name:
                    embed_params.append(p)
                else:
                    (donor_decay if p.ndim >= 2 else donor_nodecay).append(p)
            d_scale = float(self.config.donor_backbone_lr_scale)
            c_scale = float(self.config.clone_expert_lr_scale)
            ncm_scale = 5.0 if d_scale < 0.1 else 1.0
            emb_scale = d_scale / max(1.0, float(self.config.embedding_multiplier) * 0.5)
            param_groups = [
                {"params": ncm_decay, "weight_decay": self.config.weight_decay, "lr_scale": ncm_scale},
                {"params": ncm_nodecay, "weight_decay": 0.0, "lr_scale": ncm_scale},
                {"params": adapter_decay, "weight_decay": self.config.weight_decay, "lr_scale": 1.0},
                {"params": adapter_nodecay, "weight_decay": 0.0, "lr_scale": 1.0},
                {"params": clone_decay, "weight_decay": self.config.weight_decay, "lr_scale": c_scale},
                {"params": clone_nodecay, "weight_decay": 0.0, "lr_scale": c_scale},
                {"params": donor_decay, "weight_decay": self.config.weight_decay, "lr_scale": d_scale},
                {"params": donor_nodecay, "weight_decay": 0.0, "lr_scale": d_scale},
                {"params": embed_params, "weight_decay": self.config.weight_decay, "lr_scale": emb_scale},
            ]
            param_groups = [g for g in param_groups if g["params"]]
        else:
            decay_params = [p for p in self.model.parameters() if p.requires_grad and p.ndim >= 2]
            nodecay_params = [p for p in self.model.parameters() if p.requires_grad and p.ndim < 2]
            param_groups = [
                {"params": decay_params, "weight_decay": self.config.weight_decay, "lr_scale": 1.0},
                {"params": nodecay_params, "weight_decay": 0.0, "lr_scale": 1.0},
            ]

        if self.config.cpu_offload_optimizer:
            self.optimizer = CPUOffloadAdamW(
                param_groups,
                lr=self.config.lr,
                betas=(0.9, 0.95),
                weight_decay=self.config.weight_decay,
                transfer_mode="batched",
            )
        else:
            use_fused = self.device.type == "cuda"
            if use_fused:
                dtype_groups: list[dict[str, Any]] = []
                for g in param_groups:
                    by_dtype: dict[torch.dtype, list[torch.nn.Parameter]] = {}
                    for p in g["params"]:
                        by_dtype.setdefault(p.dtype, []).append(p)
                    for dt_params in by_dtype.values():
                        if dt_params:
                            ng = dict(g)
                            ng["params"] = dt_params
                            dtype_groups.append(ng)
                param_groups = dtype_groups
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.config.lr,
                betas=(0.9, 0.95),
                fused=use_fused,
            )

    def _write_manifest(self, status: str) -> None:
        payload = {
            "run_id": self.config.run_id,
            "status": status,
            "created_at": self.start_time,
            "updated_at": time.time(),
            "current_step": self.current_step,
            "total_steps": self.config.total_steps,
            "device": str(self.device),
            "gpu_name": torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU",
            "parameter_count": parameter_count(self.model),
            "num_experts": self.model.backbone.config.num_experts,
            "nvfp4_layers": self.nvfp4_layers_converted,
            "dataset_seq_len": self.dataset.seq_len,
            "dataset_total_blocks": len(self.dataset),
            "track_blocks": dict(zip(self.dataset.track_names, self.dataset.track_block_counts)),
            "config": self.config.to_dict(),
            "latest_metrics": self.latest_metrics,
        }
        _atomic_json(self.manifest_path, payload)

    @torch.no_grad()
    def _accumulate_per_track_losses_gpu(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        track_ids: torch.Tensor,
        accum_sums: dict[str, Tensor],
        accum_counts: dict[str, Tensor],
    ) -> None:
        """Accumulate unmasked token cross-entropy per track on GPU without host sync."""
        b, seq_len, vocab = logits.shape
        token_losses = (
            F.cross_entropy(
                logits.view(-1, vocab),
                labels.view(-1),
                ignore_index=-100,
                reduction="none",
            )
            .float()
            .view(b, seq_len)
        )
        valid_mask = (labels != -100).float()
        for t_idx, t_name in enumerate(self.dataset.track_names):
            row_mask = (track_ids == t_idx).unsqueeze(-1).float() * valid_mask
            key = f"loss_{t_name}"
            accum_sums[key] = accum_sums[key] + (token_losses * row_mask).sum()
            accum_counts[key] = accum_counts[key] + row_mask.sum()

    def save_checkpoint(
        self,
        tag: str | None = None,
        *,
        sample_uta: str | None = None,
    ) -> Path:
        """Save `.pt` checkpoint + lightweight `.meta.json` receipt for the Web UI."""
        step_name = tag or f"step_{self.current_step:05d}"
        pt_path = self.ckpt_dir / f"{step_name}.pt"
        meta_path = self.ckpt_dir / f"{step_name}.meta.json"

        if sample_uta is None:
            sample_uta = self.controller.generate_uta(
                self.config.sample_prompt,
                max_new_tokens=48,
            )
        meta = {
            "checkpoint_id": f"{self.config.run_id}/{step_name}",
            "run_id": self.config.run_id,
            "step": self.current_step,
            "total_steps": self.config.total_steps,
            "timestamp": time.time(),
            "pt_file": str(pt_path),
            "meta_file": str(meta_path),
            "parameter_count": parameter_count(self.model),
            "num_experts": self.model.backbone.config.num_experts,
            "use_nvfp4": self.config.use_nvfp4,
            "metrics": dict(self.latest_metrics),
            "sample_prompt": self.config.sample_prompt,
            "sample_uta": sample_uta,
            "config": self.config.to_dict(),
        }

        opt_state_cpu: dict[str, Any] = {}
        if self.config.save_optimizer_state:
            try:
                raw_opt = self.optimizer.state_dict()
                opt_state_cpu = {
                    "param_groups": raw_opt.get("param_groups", []),
                    "state": {
                        k: {
                            sk: (sv.detach().cpu() if isinstance(sv, torch.Tensor) else sv)
                            for sk, sv in v.items()
                        }
                        for k, v in raw_opt.get("state", {}).items()
                    },
                }
            except Exception:
                pass

        state_payload = {
            "step": self.current_step,
            "config": self.config.to_dict(),
            "backbone_config": asdict(self.backbone_cfg),
            "coe_config": asdict(self.coe_cfg),
            "mitotic_pairs": [list(p) for p in self._get_model_mitotic_pairs()],
            "released_mitotic_pairs": [
                list(p) for p in getattr(self, "_released_mitotic_pairs", [])
            ],
            "model_state": model_state_cpu(self.model),
            "optimizer_state": opt_state_cpu,
            "sampler_state": self.sampler.state_dict(),
            "cumulative_wall_s": self.cumulative_wall_s,
            "cumulative_tokens": self.cumulative_tokens,
            "latest_metrics": self.latest_metrics,
        }
        tmp_pt = pt_path.with_name(f"{pt_path.name}.{threading.get_ident()}.tmp")
        torch.save(state_payload, tmp_pt)
        del state_payload
        _safe_replace(tmp_pt, pt_path)
        _atomic_json(meta_path, meta)
        if getattr(self, "ornith_improver", None) is not None and getattr(self.ornith_improver, "dream_engine", None) is not None:
            try:
                self.ornith_improver.dream_engine.save_artifacts()
            except Exception:
                pass
        if self.config.hf_backup_repo:
            self._async_backup_to_hf(step_name, pt_path)
        if int(self.config.max_keep_checkpoints) > 0:
            try:
                pt_files = sorted(
                    self.ckpt_dir.glob("step_*.pt"),
                    key=lambda p: p.stat().st_mtime,
                )
                keep_n = max(1, int(self.config.max_keep_checkpoints))
                for old_pt in pt_files[:-keep_n]:
                    if old_pt != pt_path:
                        old_pt.unlink(missing_ok=True)
            except Exception:
                pass
        return meta_path

    def _async_backup_to_hf(self, step_name: str, pt_path: Path) -> None:
        """Asynchronously upload run telemetry and a compact `bfloat16` checkpoint to Hugging Face Hub."""
        repo_id = (self.config.hf_backup_repo or "").strip()
        hf_token = os.environ.get("HF_TOKEN", "").strip()
        if not repo_id or not hf_token:
            return
        run_dir = self.run_dir
        run_id = self.config.run_id
        keep_remote = max(1, int(self.config.hf_keep_checkpoints))
        already_compact = (not self.config.save_optimizer_state) and (
            str(self.config.model_dtype).lower() in ("bfloat16", "bf16")
        )
        tmp_cloud_pt = pt_path.with_name(f"hf_sync_{step_name}.pt")
        hardlinked = False
        if already_compact and pt_path.is_file():
            try:
                tmp_cloud_pt.unlink(missing_ok=True)
                os.link(pt_path, tmp_cloud_pt)
                hardlinked = True
            except Exception:
                hardlinked = False

        if not hasattr(self, "_hf_upload_lock"):
            self._hf_upload_lock = threading.Lock()

        def _worker() -> None:
            with self._hf_upload_lock:
                try:
                    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

                    api = HfApi(token=hf_token)
                    telemetry_ops: list[Any] = []
                    for rel_f in (
                        "run_manifest.json",
                        "metrics.jsonl",
                        "samples.jsonl",
                        "dream/dream_state.json",
                        f"checkpoints/{step_name}.meta.json",
                    ):
                        local_f = run_dir / rel_f
                        if local_f.is_file():
                            telemetry_ops.append(
                                CommitOperationAdd(
                                    path_in_repo=f"{run_id}/{rel_f}",
                                    path_or_fileobj=str(local_f),
                                )
                            )
                    ops: list[Any] = list(telemetry_ops)
                    if not getattr(self, "_hf_pt_quota_reached", False):
                        if not hardlinked and pt_path.is_file():
                            raw = torch.load(pt_path, map_location="cpu", weights_only=False)
                            raw.pop("optimizer_state", None)
                            if "model_state" in raw and isinstance(raw["model_state"], dict):
                                raw["model_state"] = {
                                    k: (
                                        v.to(dtype=torch.bfloat16)
                                        if torch.is_tensor(v) and torch.is_floating_point(v)
                                        else v
                                    )
                                    for k, v in raw["model_state"].items()
                                }
                            torch.save(raw, tmp_cloud_pt)
                            del raw
                            gc.collect()
                        if tmp_cloud_pt.is_file():
                            ops.append(
                                CommitOperationAdd(
                                    path_in_repo=f"{run_id}/checkpoints/{step_name}.pt",
                                    path_or_fileobj=str(tmp_cloud_pt),
                                )
                            )
                        try:
                            repo_files = api.list_repo_files(repo_id, repo_type="model")
                            remote_pts = sorted(
                                f
                                for f in repo_files
                                if f.startswith(f"{run_id}/checkpoints/step_")
                                and f.endswith(".pt")
                                and f != f"{run_id}/checkpoints/{step_name}.pt"
                            )
                            if len(remote_pts) >= keep_remote:
                                for old_remote in remote_pts[: len(remote_pts) - keep_remote + 1]:
                                    ops.append(CommitOperationDelete(path_in_repo=old_remote))
                        except Exception:
                            pass
                    if ops:
                        try:
                            api.create_commit(
                                repo_id=repo_id,
                                repo_type="model",
                                operations=ops,
                                commit_message=f"[{run_id}] Sync telemetry & checkpoint {step_name}",
                            )
                        except Exception as commit_exc:
                            if "storage" in str(commit_exc).lower() and telemetry_ops:
                                self._hf_pt_quota_reached = True
                                api.create_commit(
                                    repo_id=repo_id,
                                    repo_type="model",
                                    operations=telemetry_ops,
                                    commit_message=f"[{run_id}] Sync telemetry {step_name}",
                                )
                            else:
                                raise
                    tmp_cloud_pt.unlink(missing_ok=True)
                except Exception as exc:
                    print('hf_backup warning:', step_name, repo_id, exc, flush=True)
                    try:
                        tmp_cloud_pt.unlink(missing_ok=True)
                    except Exception:
                        pass

        t = threading.Thread(target=_worker, name=f"hf_backup_{step_name}", daemon=True)
        self._last_hf_backup_thread = t
        t.start()

    def _get_model_mitotic_pairs(self) -> list[tuple[int, int]]:
        if self.model.backbone.encoder_layers:
            return list(self.model.backbone.encoder_layers[0].moe.mitotic_pairs)
        return []

    def _set_model_mitotic_pairs(self, pairs: list[tuple[int, int]]) -> None:
        clean = [(int(p), int(c)) for p, c in pairs]
        for layer in list(self.model.backbone.encoder_layers) + list(
            self.model.backbone.decoder_layers
        ):
            layer.moe.mitotic_pairs = list(clean)

    def _ordered_trainable_param_names(self) -> list[str]:
        """Return trainable parameter names in the exact order they appear across `optimizer.param_groups`."""
        if self.config.preserve_residual_across_stacks and float(self.config.donor_backbone_lr_scale) != 1.0:
            base_markers = (
                ".ple.",
                ".markov_router.",
                ".markov_gate",
                ".global_",
                ".feedback.",
            )
            if str(self.config.qk_norm_mode).lower() != "full":
                base_markers = (*base_markers, ".q_norm.", ".k_norm.")
            if float(self.config.routed_scaling_factor) > 1.0 or self.config.freeze_non_clone_experts:
                base_markers = (*base_markers, ".moe.router.")
            new_adapter_markers = base_markers
            active_clone_markers = (
                tuple(f".moe.experts.{int(c)}." for _, c in (getattr(self, "_active_plastic_pairs", None) or self._get_model_mitotic_pairs()))
                if self.config.freeze_non_clone_experts
                else ()
            )
            ncm_decay, ncm_nodecay = [], []
            adapter_decay, adapter_nodecay = [], []
            clone_decay, clone_nodecay = [], []
            donor_decay, donor_nodecay = [], []
            embed_params = []
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if name.startswith("ncm_system."):
                    (ncm_decay if p.ndim >= 2 else ncm_nodecay).append(name)
                    continue
                if active_clone_markers and any(cm in name for cm in active_clone_markers):
                    (clone_decay if p.ndim >= 2 else clone_nodecay).append(name)
                    continue
                is_new_adapter = (not name.startswith("backbone.")) or any(
                    m in name for m in new_adapter_markers
                )
                if is_new_adapter:
                    (adapter_decay if p.ndim >= 2 else adapter_nodecay).append(name)
                elif "token_embedding" in name or "lm_head" in name:
                    embed_params.append(name)
                else:
                    (donor_decay if p.ndim >= 2 else donor_nodecay).append(name)
            return (
                ncm_decay
                + ncm_nodecay
                + adapter_decay
                + adapter_nodecay
                + clone_decay
                + clone_nodecay
                + donor_decay
                + donor_nodecay
                + embed_params
            )
        new_decay_names = [
            name for name, p in self.model.named_parameters() if p.requires_grad and p.ndim >= 2
        ]
        new_nodecay_names = [
            name for name, p in self.model.named_parameters() if p.requires_grad and p.ndim < 2
        ]
        return new_decay_names + new_nodecay_names

    def _upcycle_optimizer_state_dict(
        self,
        raw_opt_state: dict[str, Any],
        old_num_experts: int,
        new_num_experts: int,
        added_pairs: list[tuple[int, int]],
    ) -> bool:
        """Map and expand AdamW moments (`step`, `exp_avg`, `exp_avg_sq`) across an expert mitosis split."""
        if not raw_opt_state or not raw_opt_state.get("state"):
            return False
        try:
            new_ordered_names = self._ordered_trainable_param_names()
            new_expert_tokens = tuple(
                f".moe.experts.{e}." for e in range(old_num_experts, new_num_experts)
            )
            old_ordered_names = [
                name
                for name in new_ordered_names
                if not any(tok in name for tok in new_expert_tokens)
            ]
            old_param_ids = [
                pid
                for g in raw_opt_state.get("param_groups", [])
                for pid in g.get("params", [])
            ]
            if len(old_param_ids) != len(old_ordered_names):
                return False

            raw_state = raw_opt_state["state"]
            old_state_by_name = {
                name: raw_state[pid]
                for name, pid in zip(old_ordered_names, old_param_ids)
                if pid in raw_state
            }
            if not old_state_by_name:
                return False

            child_to_parent = {int(c): int(p) for p, c in added_pairs}
            new_state: dict[int, dict[str, Any]] = {}
            for new_idx, name in enumerate(new_ordered_names):
                if name.endswith(".moe.router.weight") and name in old_state_by_name:
                    st = old_state_by_name[name]
                    old_avg = st["exp_avg"].detach().cpu().clone()
                    old_sq = st["exp_avg_sq"].detach().cpu().clone()
                    extra_avg = [torch.zeros_like(old_avg[p_e : p_e + 1]) for p_e, _ in added_pairs]
                    extra_sq = [old_sq[p_e : p_e + 1].clone() for p_e, _ in added_pairs]
                    step_v = (
                        st["step"].detach().cpu().clone()
                        if isinstance(st["step"], torch.Tensor)
                        else st["step"]
                    )
                    new_state[new_idx] = {
                        "step": step_v,
                        "exp_avg": torch.cat([old_avg, *extra_avg], dim=0),
                        "exp_avg_sq": torch.cat([old_sq, *extra_sq], dim=0),
                    }
                    continue

                matched_child: int | None = None
                for c_e in child_to_parent:
                    if f".moe.experts.{c_e}." in name:
                        matched_child = c_e
                        break
                if matched_child is not None:
                    p_e = child_to_parent[matched_child]
                    parent_name = name.replace(
                        f".moe.experts.{matched_child}.", f".moe.experts.{p_e}."
                    )
                    if parent_name in old_state_by_name:
                        st = old_state_by_name[parent_name]
                        p_avg = st["exp_avg"].detach().cpu()
                        p_sq = st["exp_avg_sq"].detach().cpu().clone()
                        step_v = (
                            st["step"].detach().cpu().clone()
                            if isinstance(st["step"], torch.Tensor)
                            else st["step"]
                        )
                        new_state[new_idx] = {
                            "step": step_v,
                            "exp_avg": torch.zeros_like(p_avg),
                            "exp_avg_sq": p_sq,
                        }
                    continue

                if name in old_state_by_name:
                    st = old_state_by_name[name]
                    step_v = (
                        st["step"].detach().cpu().clone()
                        if isinstance(st["step"], torch.Tensor)
                        else st["step"]
                    )
                    new_state[new_idx] = {
                        "step": step_v,
                        "exp_avg": st["exp_avg"].detach().cpu().clone(),
                        "exp_avg_sq": st["exp_avg_sq"].detach().cpu().clone(),
                    }

            target_opt_dict = self.optimizer.state_dict()
            target_opt_dict["state"] = new_state
            self.optimizer.load_state_dict(target_opt_dict)
            return True
        except Exception:
            return False

    def perform_bounded_mitosis(self, step: int) -> dict[str, Any] | None:
        """Perform bounded in-flight expert mitosis (`old_e -> new_e <= max_experts`) or steady-state check."""
        if not self.config.enable_mitosis:
            return None
        base_step = getattr(self, "initial_step", 0)
        old_e = int(self.model.backbone.config.num_experts)
        max_e = max(old_e, int(self.config.max_experts))
        existing_pairs = self._get_model_mitotic_pairs()

        target_new_e: int | None = None
        milestones = [
            (int(ms[0]), int(ms[1]))
            for ms in (self.config.mitosis_milestones or ())
            if len(ms) == 2
        ]
        if milestones:
            due_targets = [t_e for m_step, t_e in milestones if step >= m_step and t_e > old_e]
            exact_hit = any(step == m_step for m_step, _ in milestones)
            if due_targets and (exact_hit or step == base_step + 1):
                target_new_e = min(max_e, max(due_targets))
            else:
                return None
        else:
            if self.config.mitosis_every <= 0:
                return None
            in_place_resume = bool(
                getattr(self, "_optimizer_restored", False)
                and getattr(self, "_parent_total_steps", 0) == self.config.total_steps
            )
            if in_place_resume:
                expected_splits = (base_step // self.config.mitosis_every) * max(
                    1, int(self.config.mitosis_step_size)
                )
                catch_up_due = (
                    step == base_step + 1
                    and base_step >= self.config.mitosis_every
                    and old_e < max_e
                    and len(existing_pairs) < expected_splits
                )
                if not catch_up_due and (step <= 1 or (step % self.config.mitosis_every != 0)):
                    return None
            else:
                local_step = step - base_step
                if local_step <= 1 or (local_step % self.config.mitosis_every != 0):
                    return None
            if old_e < max_e:
                target_new_e = min(max_e, old_e + max(1, int(self.config.mitosis_step_size)))

        loads_for_split = list(
            getattr(self, "_ema_expert_loads", None)
            or self.latest_metrics.get("expert_load")
            or [1.0 / max(1, old_e)] * old_e
        )

        if target_new_e is not None and old_e < target_new_e:
            new_e = target_new_e
            # When using staged milestones, prior-stage twin pairs have already differentiated over
            # hundreds of steps, so release prior pairs so any high-load expert can split cleanly.
            pairs_for_schedule = [] if milestones else existing_pairs
            added_pairs = compute_mitotic_schedule(
                old_e,
                new_e,
                expert_loads=loads_for_split,
                existing_pairs=pairs_for_schedule,
            )
            use_inplace = (not self.config.cpu_offload_optimizer) and (
                str(self.config.model_dtype).lower() in ("bfloat16", "bf16")
                or int(self.config.plastic_experts_per_layer) > 0
            )
            if use_inplace:
                self._perform_inplace_mitosis(
                    step=step,
                    old_e=old_e,
                    new_e=new_e,
                    loads_for_split=loads_for_split,
                    pairs_for_schedule=pairs_for_schedule,
                    added_pairs=added_pairs,
                )
            else:
                cur_model_state = model_state_cpu(self.model)
                raw_opt = self.optimizer.state_dict()
                cur_opt_state = {
                    "param_groups": raw_opt.get("param_groups", []),
                    "state": {
                        k: {
                            sk: (sv.detach().cpu().clone() if isinstance(sv, torch.Tensor) else sv)
                            for sk, sv in v.items()
                        }
                        for k, v in raw_opt.get("state", {}).items()
                    },
                }
                del raw_opt
                self.model.cpu()
                self.optimizer.state.clear()
                if hasattr(self, "controller") and self.controller is not None:
                    self.controller.model = None  # type: ignore[assignment]
                del self.model
                del self.optimizer
                if self.device.type == "cuda":
                    gc.collect()
                    torch.cuda.empty_cache()

                upcycled_state = upcycle_state_dict_experts(
                    cur_model_state,
                    old_e,
                    new_e,
                    expert_loads=loads_for_split,
                    existing_pairs=pairs_for_schedule,
                    seed=self.config.seed + step,
                    orthogonal=self.config.orthogonal_mitosis,
                    preserve_parent_weights=bool(self.config.freeze_non_clone_experts),
                    perturbation_scale=float(self.config.mitotic_perturbation_scale),
                )
                del cur_model_state
                saved_ornith = self.ornith_improver
                self.config.num_experts = new_e
                self._build_model_and_optimizer()
                self.ornith_improver = saved_ornith
                if hasattr(self, "controller") and self.controller is not None:
                    self.controller.model = self.model
                self.model.load_state_dict(upcycled_state, strict=False)
                del upcycled_state
                with torch.no_grad():
                    self.model.log_scale.data.clamp_(math.log(0.5), math.log(3.5))
                newly_split = {idx for pair in added_pairs for idx in pair}
                active_existing = [
                    (p, c)
                    for p, c in pairs_for_schedule
                    if p not in newly_split and c not in newly_split
                ]
                if self.config.freeze_non_clone_experts:
                    prev_rel = list(getattr(self, "_released_mitotic_pairs", []))
                    for pr in pairs_for_schedule:
                        if pr not in prev_rel:
                            prev_rel.append(pr)
                    self._released_mitotic_pairs = prev_rel
                    self._set_model_mitotic_pairs(list(added_pairs))
                    self._apply_plasticity_policy(loads_for_split)
                    self._create_optimizer()
                else:
                    self._set_model_mitotic_pairs(active_existing + added_pairs)
                if self.config.cpu_offload_optimizer:
                    prepare_model(
                        self.model,
                        device=self.device,
                        bf16_working_weights=(self.device.type == "cuda" and not self.config.use_nvfp4),
                        preserve_master_weights=False,
                    )
                else:
                    self.model.to(self.device)
                    self._optimizer_restored = self._upcycle_optimizer_state_dict(
                        cur_opt_state, old_e, new_e, added_pairs
                    )
                del cur_opt_state
                if self.device.type == "cuda":
                    gc.collect()
                    torch.cuda.empty_cache()

                new_ema = list(loads_for_split) + [0.0] * (new_e - old_e)
                for p_e, c_e in added_pairs:
                    half = 0.5 * float(new_ema[p_e])
                    new_ema[p_e] = half
                    new_ema[c_e] = half
                self._ema_expert_loads = new_ema

            all_pairs = self._get_model_mitotic_pairs()
            cleared_shards = (
                self.sampler.evict_consumed_shards(delete_from_disk=self.config.clear_consumed_shards)
                if hasattr(self.sampler, "evict_consumed_shards")
                else {}
            )
            return {
                "step": step,
                "action": "expand",
                "old_experts": old_e,
                "new_experts": new_e,
                "added_pairs": [list(p) for p in added_pairs],
                "mitotic_pairs": [list(p) for p in all_pairs],
                "parameter_count": parameter_count(self.model),
                **({"cleared_shards": cleared_shards} if cleared_shards else {}),
            }

        min_load = float(min(loads_for_split)) if loads_for_split else 0.0
        cleared_shards = (
            self.sampler.evict_consumed_shards(delete_from_disk=self.config.clear_consumed_shards)
            if hasattr(self.sampler, "evict_consumed_shards")
            else {}
        )
        return {
            "step": step,
            "action": "bounded_steady_state",
            "num_experts": old_e,
            "min_expert_load": round(min_load, 4),
            "mitotic_pairs": [list(p) for p in existing_pairs],
            "parameter_count": parameter_count(self.model),
            **({"cleared_shards": cleared_shards} if cleared_shards else {}),
        }

    def _perform_inplace_mitosis(
        self,
        *,
        step: int,
        old_e: int,
        new_e: int,
        loads_for_split: list[float],
        pairs_for_schedule: list[tuple[int, int]],
        added_pairs: list[tuple[int, int]],
    ) -> None:
        """Expand `SparseMoE` experts and routers directly on-device without full CPU state-dict cloning."""
        from .backbone import SwiGLUExpert

        old_state_by_name: dict[str, dict[str, Any]] = {}
        for name, p in self.model.named_parameters():
            if p in self.optimizer.state and self.optimizer.state[p]:
                old_state_by_name[name] = self.optimizer.state[p]
        self.optimizer.state.clear()

        gen = torch.Generator(device="cpu").manual_seed(self.config.seed + step)
        eff_scale = max(1e-5, float(self.config.mitotic_perturbation_scale))
        preserve_parent = bool(self.config.freeze_non_clone_experts)
        d_model = int(self.backbone_cfg.d_model)
        expert_hidden_dim = int(self.backbone_cfg.expert_hidden_dim)
        all_layers = list(self.model.backbone.encoder_layers) + list(
            self.model.backbone.decoder_layers
        )
        orig_lin_reset = torch.nn.Linear.reset_parameters
        old_default_dtype = torch.get_default_dtype()
        try:
            torch.nn.Linear.reset_parameters = lambda _self: None  # type: ignore[method-assign]
            with torch.no_grad():
                for layer in all_layers:
                    moe = layer.moe
                    for p_e, _ in added_pairs:
                        parent_exp = moe.experts[p_e]
                        p_up = parent_exp.up_proj.weight.data
                        p_down = parent_exp.down_proj.weight.data
                        torch.set_default_dtype(p_up.dtype)
                        with torch.device(p_up.device):
                            child_exp = SwiGLUExpert(d_model, expert_hidden_dim)
                        torch.set_default_dtype(old_default_dtype)
                        delta_up = _orthogonal_antithetic_delta(
                            p_up.detach().cpu().float(),
                            gen,
                            scale=eff_scale,
                            ortho_dim=1,
                            orthogonal=self.config.orthogonal_mitosis,
                        ).to(device=p_up.device, dtype=p_up.dtype)
                        if preserve_parent:
                            child_exp.up_proj.weight.data.copy_(p_up + delta_up)
                        else:
                            child_exp.up_proj.weight.data.copy_(p_up - delta_up)
                            p_up.add_(delta_up)

                        delta_down = _orthogonal_antithetic_delta(
                            p_down.detach().cpu().float(),
                            gen,
                            scale=eff_scale,
                            ortho_dim=0,
                            orthogonal=self.config.orthogonal_mitosis,
                        ).to(device=p_down.device, dtype=p_down.dtype)
                        if preserve_parent:
                            child_exp.down_proj.weight.data.copy_(p_down + delta_down)
                        else:
                            child_exp.down_proj.weight.data.copy_(p_down - delta_down)
                            p_down.add_(delta_down)
                        moe.experts.append(child_exp)

                    old_rw = moe.router.weight.data
                    extra_rows = []
                    for p_e, _ in added_pairs:
                        r_p = old_rw[p_e : p_e + 1].clone()
                        if preserve_parent:
                            r_delta = _orthogonal_antithetic_delta(
                                r_p.detach().cpu().float(),
                                gen,
                                scale=eff_scale,
                                ortho_dim=-1,
                                orthogonal=self.config.orthogonal_mitosis,
                            ).to(device=r_p.device, dtype=r_p.dtype)
                            extra_rows.append(r_p + r_delta)
                        else:
                            extra_rows.append(r_p)
                    new_rw = torch.cat([old_rw, *extra_rows], dim=0)
                    new_router = torch.nn.Linear(
                        d_model, new_e, bias=False, device=old_rw.device, dtype=old_rw.dtype
                    )
                    new_router.weight.data.copy_(new_rw)
                    moe.router = new_router
                    moe.num_experts = new_e
                    moe._grouped_weight_cache = None
                    moe._mitotic_idx_cache = None

                self.model.log_scale.data.clamp_(math.log(0.5), math.log(3.5))
        finally:
            torch.set_default_dtype(old_default_dtype)
            torch.nn.Linear.reset_parameters = orig_lin_reset  # type: ignore[method-assign]

        if self.config.use_nvfp4 and self.device.type == "cuda":
            for layer in all_layers:
                for idx in range(old_e, new_e):
                    self.nvfp4_layers_converted += convert_linear_to_nvfp4(
                        layer.moe.experts[idx],
                        min_features=32,
                        experts_only=False,
                        sparsity_mode=self.config.nvfp4_sparsity_mode,
                        sr_ste_decay=self.config.nvfp4_sr_ste_decay,
                        causal_activation_scale=True,
                    )

        self.config.num_experts = new_e
        self.backbone_cfg.num_experts = new_e
        self.model.backbone.config.num_experts = new_e
        newly_split = {idx for pair in added_pairs for idx in pair}
        active_existing = [
            (p, c)
            for p, c in pairs_for_schedule
            if p not in newly_split and c not in newly_split
        ]
        if preserve_parent:
            prev_rel = list(getattr(self, "_released_mitotic_pairs", []))
            for pr in pairs_for_schedule:
                if pr not in prev_rel:
                    prev_rel.append(pr)
            self._released_mitotic_pairs = prev_rel
            self._set_model_mitotic_pairs(list(added_pairs))
            self._clone_warmup_tokens_seen = 0
        else:
            self._set_model_mitotic_pairs(active_existing + added_pairs)

        new_ema = list(loads_for_split) + [0.0] * (new_e - old_e)
        for p_e, c_e in added_pairs:
            half = 0.5 * float(new_ema[p_e])
            new_ema[p_e] = half
            new_ema[c_e] = half
        self._ema_expert_loads = new_ema

        self._apply_plasticity_policy(new_ema)
        self._create_optimizer()

        child_to_parent = {int(c): int(p) for p, c in added_pairs}
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith(".moe.router.weight") and name in old_state_by_name:
                st = old_state_by_name[name]
                if "exp_avg" in st and "exp_avg_sq" in st:
                    old_avg = st["exp_avg"]
                    old_sq = st["exp_avg_sq"]
                    extra_avg = [torch.zeros_like(old_avg[p_e : p_e + 1]) for p_e, _ in added_pairs]
                    extra_sq = [old_sq[p_e : p_e + 1].clone() for p_e, _ in added_pairs]
                    step_v = (
                        st["step"].clone()
                        if isinstance(st.get("step"), torch.Tensor)
                        else st.get("step", 1)
                    )
                    self.optimizer.state[p] = {
                        "step": step_v,
                        "exp_avg": torch.cat([old_avg, *extra_avg], dim=0),
                        "exp_avg_sq": torch.cat([old_sq, *extra_sq], dim=0),
                    }
                continue

            matched_child: int | None = None
            for c_e in child_to_parent:
                if f".moe.experts.{c_e}." in name:
                    matched_child = c_e
                    break
            if matched_child is not None:
                p_e = child_to_parent[matched_child]
                parent_name = name.replace(
                    f".moe.experts.{matched_child}.", f".moe.experts.{p_e}."
                )
                if parent_name in old_state_by_name:
                    st = old_state_by_name[parent_name]
                    if "exp_avg_sq" in st:
                        p_sq = st["exp_avg_sq"].clone()
                        step_v = (
                            st["step"].clone()
                            if isinstance(st.get("step"), torch.Tensor)
                            else st.get("step", 1)
                        )
                        self.optimizer.state[p] = {
                            "step": step_v,
                            "exp_avg": torch.zeros_like(p_sq),
                            "exp_avg_sq": p_sq,
                        }
                continue

            if name in old_state_by_name:
                self.optimizer.state[p] = old_state_by_name[name]

        old_state_by_name.clear()
        self._optimizer_restored = True
        if self.device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Restore model weights (upcycling experts if `num_experts` increased), optimizer state, and sampler state."""
        ckpt_resolved = Path(checkpoint_path).expanduser().resolve()
        if ckpt_resolved.name.endswith(".meta.json") and ckpt_resolved.is_file():
            meta = json.loads(ckpt_resolved.read_text(encoding="utf-8"))
            pt_candidate = Path(meta["pt_file"])
            ckpt_pt_path = pt_candidate if pt_candidate.exists() else ckpt_resolved.with_name(pt_candidate.name)
        else:
            ckpt_pt_path = ckpt_resolved

        if self._resume_payload is not None:
            payload = self._resume_payload
            self._resume_payload = None
        else:
            meta_l: dict[str, Any] = {}
            meta_cand_l = ckpt_pt_path.with_suffix(".meta.json")
            if meta_cand_l.is_file():
                meta_l = json.loads(meta_cand_l.read_text(encoding="utf-8"))
            payload = torch.load(ckpt_pt_path, map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and "model_state" not in payload:
                metrics_meta = meta_l.get("metrics", {})
                cfg_meta = dict(meta_l.get("config", {}))
                if "num_experts" in meta_l:
                    cfg_meta["num_experts"] = int(meta_l["num_experts"])
                payload = {
                    "model_state": payload,
                    "config": cfg_meta,
                    "latest_metrics": metrics_meta,
                    "step": int(meta_l.get("step", metrics_meta.get("step", 0))),
                    "mitotic_pairs": metrics_meta.get("mitotic_pairs", []),
                    "cumulative_tokens": int(metrics_meta.get("cumulative_tokens", 0)),
                    "cumulative_wall_s": float(metrics_meta.get("wall_time_s", 0.0)),
                }

        b_cfg = payload.get("backbone_config") or payload.get("config") or {}
        ckpt_experts = int(b_cfg.get("num_experts", self.config.num_experts))
        model_state = payload["model_state"]
        raw_pairs = payload.get("mitotic_pairs") or payload.get("latest_metrics", {}).get("mitotic_pairs") or []
        raw_released = payload.get("released_mitotic_pairs") or []
        existing_pairs = [tuple(p) for p in raw_pairs]
        prev_released = [tuple(p) for p in raw_released]
        all_prior_pairs = prev_released + [p for p in existing_pairs if p not in prev_released]
        added_pairs: list[tuple[int, int]] = []
        if self.config.num_experts > ckpt_experts:
            prev_loads = payload.get("latest_metrics", {}).get("expert_load")
            added_pairs = compute_mitotic_schedule(
                ckpt_experts,
                self.config.num_experts,
                expert_loads=prev_loads,
                existing_pairs=all_prior_pairs,
            )
            model_state = upcycle_state_dict_experts(
                model_state,
                ckpt_experts,
                self.config.num_experts,
                expert_loads=prev_loads,
                existing_pairs=all_prior_pairs,
                seed=self.config.seed,
                orthogonal=self.config.orthogonal_mitosis,
                preserve_parent_weights=bool(self.config.freeze_non_clone_experts),
                perturbation_scale=float(self.config.mitotic_perturbation_scale),
            )

        self.model.load_state_dict(model_state, strict=False)
        del model_state
        with torch.no_grad():
            self.model.log_scale.data.clamp_(math.log(0.5), math.log(3.5))
        if self.config.release_mitotic_pairs and not (
            self.config.freeze_non_clone_experts and (added_pairs or existing_pairs)
        ):
            self._released_mitotic_pairs = list(all_prior_pairs) + list(added_pairs)
            self._set_model_mitotic_pairs([])
        elif added_pairs:
            # Release prior-generation mitotic pairs from plasticity/ortho-repulsion so they graduate,
            # keeping only the newly split pairs plastic for the current run.
            # In Co-Active Sliced Dense models (routed_scaling_factor > 1.0), keep all slice pairs
            # twin-gated in SparseMoE so each horizontal FFN slice routes 1 winning specialist.
            self._released_mitotic_pairs = list(all_prior_pairs)
            self._active_plastic_pairs = list(added_pairs)
            if float(self.config.routed_scaling_factor) > 1.0:
                self._set_model_mitotic_pairs(list(all_prior_pairs) + list(added_pairs))
            else:
                self._set_model_mitotic_pairs(list(added_pairs))
        elif existing_pairs:
            self._released_mitotic_pairs = list(prev_released)
            self._active_plastic_pairs = [
                p for p in existing_pairs if p not in prev_released
            ] or list(existing_pairs)
            if float(self.config.routed_scaling_factor) > 1.0 and prev_released:
                self._set_model_mitotic_pairs(list(all_prior_pairs))
            else:
                self._set_model_mitotic_pairs(list(existing_pairs))

        self.latest_metrics = dict(payload.get("latest_metrics", {}))
        if not added_pairs:
            self._clone_warmup_tokens_seen = int(
                self.latest_metrics.get("clone_warmup_tokens_seen", 0)
            )
            self._resumed_in_clone_pure_rl = bool(
                self.latest_metrics.get("mitosis_phase") == "clone_pure_rl"
            )
        init_loads = list(self.latest_metrics.get("expert_load", []))
        if init_loads and len(init_loads) == ckpt_experts and self.config.num_experts > ckpt_experts:
            init_loads = init_loads + [0.0] * (self.config.num_experts - ckpt_experts)
            for p_e, c_e in added_pairs:
                half = 0.5 * float(init_loads[p_e])
                init_loads[p_e] = half
                init_loads[c_e] = half
        if init_loads and len(init_loads) == self.config.num_experts:
            self._ema_expert_loads = [float(x) for x in init_loads]

        if (
            int(self.config.plastic_experts_per_layer) > 0
            or self.config.freeze_donor_embeddings
            or self.config.freeze_non_clone_experts
        ):
            self._apply_plasticity_policy(getattr(self, "_ema_expert_loads", None))
            self._create_optimizer()

        if self.config.cpu_offload_optimizer:
            prepare_model(
                self.model,
                device=self.device,
                bf16_working_weights=(self.device.type == "cuda" and not self.config.use_nvfp4),
                preserve_master_weights=False,
            )
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if p in self.optimizer.state and "master" in self.optimizer.state[p]:
                        self.optimizer.state[p]["master"].copy_(
                            p.detach().to(device="cpu", dtype=torch.float32)
                        )
        else:
            self.model.to(self.device)
            if payload.get("optimizer_state"):
                if self.config.num_experts == ckpt_experts:
                    try:
                        self.optimizer.load_state_dict(payload["optimizer_state"])
                        self._optimizer_restored = True
                    except Exception:
                        self._optimizer_restored = False
                elif self.config.num_experts > ckpt_experts:
                    self._optimizer_restored = self._upcycle_optimizer_state_dict(
                        payload["optimizer_state"],
                        ckpt_experts,
                        self.config.num_experts,
                        added_pairs,
                    )
        if "sampler_state" in payload:
            self.sampler.load_state_dict(payload["sampler_state"])
        self.current_step = int(payload.get("step", 0))
        self.initial_step = self.current_step
        self._parent_total_steps = int(payload.get("config", {}).get("total_steps", self.current_step))
        self.cumulative_wall_s = float(payload.get("cumulative_wall_s", 0.0))
        self.cumulative_tokens = int(payload.get("cumulative_tokens", 0))
        self._resume_lr = float(
            self.latest_metrics.get("lr", self.config.lr * self.config.min_lr_ratio)
        )
        del payload
        gc.collect()

        # Carry forward parent run's Dream-RSI world archive (`dream_state.json`) if present
        if (
            self.ornith_improver is not None
            and self.ornith_improver.dream_engine is not None
            and not self.ornith_improver.dream_engine.state.worlds
        ):
            parent_dream = ckpt_pt_path.parent.parent / "dream" / "dream_state.json"
            if parent_dream.is_file():
                try:
                    from .dream_rsi import DreamState

                    raw_d = json.loads(parent_dream.read_text(encoding="utf-8"))
                    self.ornith_improver.dream_engine.state = DreamState.from_dict(raw_d)
                    self.ornith_improver.dream_engine.save_artifacts()
                except Exception:
                    pass

        # If resuming in-place in an existing run dir, prune any steps > current_step
        for jsonl_file in (self.metrics_path, self.samples_path):
            if jsonl_file.is_file():
                try:
                    kept = []
                    for line in jsonl_file.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        if int(obj.get("step", 0)) <= self.current_step:
                            kept.append(line)
                    jsonl_file.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
                except Exception:
                    pass

        # Seed a fresh upcycled/resumed run's metrics.jsonl with the checkpoint anchor point
        if not self.metrics_path.is_file() and self.latest_metrics and self.current_step > 0:
            anchor = dict(self.latest_metrics)
            anchor["step"] = self.current_step
            anchor["total_steps"] = self.config.total_steps
            with self.metrics_path.open("w", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(anchor, sort_keys=True) + "\n")

        if self.ornith_improver is not None and self.metrics_path.is_file():
            try:
                self.ornith_improver.restore_from_metrics_jsonl(
                    self.metrics_path, max_step=self.current_step
                )
            except Exception:
                pass

    def _compute_lr(self, step: int) -> float:
        base_step = getattr(self, "initial_step", 0)
        stage_steps = max(1, self.config.total_steps - base_step)
        local_step = max(1, step - base_step)
        if getattr(self, "_optimizer_restored", False) and base_step > 0:
            start_lr = float(
                getattr(
                    self,
                    "_resume_lr",
                    self.latest_metrics.get("lr", self.config.lr * self.config.min_lr_ratio),
                )
            )
            # In-place continuation of an interrupted run: smoothly rejoin the global cosine curve
            if getattr(self, "_parent_total_steps", 0) == self.config.total_steps:
                g_warmup = max(1, int(round(self.config.total_steps * self.config.warmup_ratio)))
                if step <= g_warmup:
                    global_lr = self.config.lr * (step / g_warmup)
                else:
                    g_span = max(1, self.config.total_steps - g_warmup)
                    g_prog = min(1.0, max(0.0, (step - g_warmup) / g_span))
                    g_cos = 0.5 * (1.0 + math.cos(math.pi * g_prog))
                    global_lr = self.config.lr * (
                        self.config.min_lr_ratio + (1.0 - self.config.min_lr_ratio) * g_cos
                    )
                rejoin_steps = min(100, max(1, int(round(stage_steps * 0.05))))
                if local_step < rejoin_steps:
                    alpha = local_step / rejoin_steps
                    return start_lr + (global_lr - start_lr) * alpha
                return global_lr

            # Stage-2 continuation from a completed parent checkpoint: ramp gently from start_lr
            # to at most 1.75x min_lr and cosine-decay down to 0.65x min_lr.
            min_lr = self.config.lr * self.config.min_lr_ratio
            start_lr = max(min_lr * 0.5, min(self.config.lr, start_lr))
            peak_lr = min(self.config.lr, max(start_lr, min_lr * 1.75))
            end_lr = min_lr * 0.65
            warmup_steps = max(1, int(round(stage_steps * 0.05)))
            if local_step <= warmup_steps:
                alpha = local_step / warmup_steps
                return start_lr + (peak_lr - start_lr) * alpha
            decay_span = max(1, stage_steps - warmup_steps)
            decay_progress = min(1.0, max(0.0, (local_step - warmup_steps) / decay_span))
            cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return end_lr + (peak_lr - end_lr) * cosine

        # Cold start or expert-upcycled stage without optimizer moments: warm up over >= 12% of stage steps
        eff_warmup = max(self.config.warmup_ratio, 0.12 if base_step > 0 else 0.0)
        min_w = 48 if self.config.preserve_residual_across_stacks else 1
        warmup_steps = max(min_w, int(round(stage_steps * eff_warmup)))
        if local_step <= warmup_steps:
            return self.config.lr * (local_step / warmup_steps)
        decay_span = max(1, stage_steps - warmup_steps)
        decay_progress = min(1.0, max(0.0, (local_step - warmup_steps) / decay_span))
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return self.config.lr * (
            self.config.min_lr_ratio + (1.0 - self.config.min_lr_ratio) * cosine
        )

    def _mitotic_stage_progress(self, step: int) -> tuple[int | None, float | None]:
        """Return `(stage_idx, stage_progress)` when `mitotic_staged` curriculum or `curriculum_stages` is active."""
        if self.config.curriculum_mode != "mitotic_staged" and not self.config.curriculum_stages:
            return None, None
        if self.config.enable_mitosis and self.config.mitosis_every > 0:
            m_every = max(1, int(self.config.mitosis_every))
            n_pairs = len(self._get_model_mitotic_pairs())
            step_sz = max(1, int(self.config.mitosis_step_size))
            stage_from_pairs = n_pairs // step_sz
            stage_from_step = max(0, int(step) // m_every)
            stage_idx = max(stage_from_pairs, stage_from_step)
            stage_prog = min(1.0, max(0.0, float(int(step) % m_every) / float(m_every)))
            return stage_idx, stage_prog
        n_stages = len(self.schedule.stage_weights) if self.schedule.stage_weights else 3
        u = min(1.0, max(0.0, float(step) / float(max(1, self.config.total_steps))))
        scaled = u * max(1, n_stages - 1)
        stage_idx = int(math.floor(scaled))
        stage_prog = scaled - stage_idx
        return stage_idx, stage_prog

    def _active_stage_seq_len(self, stage_idx: int | None) -> int:
        """Return the active sequence length for the current mitotic stage."""
        stage_lens = tuple(int(x) for x in self.config.stage_seq_lens if int(x) > 0)
        if not stage_lens and self.config.curriculum_mode == "mitotic_staged" and self.dataset.seq_len >= 2048:
            stage_lens = (512, 1024, 2048)
        if stage_lens:
            idx = min(max(0, int(stage_idx or 0)), len(stage_lens) - 1)
            return max(16, min(int(self.dataset.seq_len), int(stage_lens[idx])))
        return int(self.dataset.seq_len)

    def _slice_micro_batch_to_seq_len(
        self, mb: dict[str, torch.Tensor], target_seq_len: int
    ) -> dict[str, torch.Tensor]:
        """Slice a micro-batch to `target_seq_len`, preserving valid target tokens (`labels != -100`) per row."""
        inp = mb["input_ids"]
        lbl = mb["labels"]
        if inp.ndim != 2 or inp.shape[1] <= target_seq_len:
            return mb
        head_lbl = lbl[:, :target_seq_len]
        if bool((head_lbl != -100).any(dim=-1).all()):
            return {
                "input_ids": inp[:, :target_seq_len],
                "labels": head_lbl,
                "track_ids": mb["track_ids"],
            }
        bsz, full_len = inp.shape
        out_inp = inp[:, :target_seq_len].clone()
        out_lbl = head_lbl.clone()
        for r in range(bsz):
            if not bool((out_lbl[r] != -100).any()):
                valid_pos = torch.nonzero(lbl[r] != -100, as_tuple=False).flatten()
                if valid_pos.numel() > 0:
                    end_pos = min(full_len, int(valid_pos[-1].item()) + 1)
                    start_pos = max(0, end_pos - target_seq_len)
                    out_inp[r] = inp[r, start_pos : start_pos + target_seq_len]
                    out_lbl[r] = lbl[r, start_pos : start_pos + target_seq_len]
        return {
            "input_ids": out_inp,
            "labels": out_lbl,
            "track_ids": mb["track_ids"],
        }

    def train(self) -> dict[str, Any]:
        """Run the training loop until `total_steps` or `stop_event` is set."""
        self._write_manifest("running")
        if self.device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        status = "completed"
        try:
            start_step = self.current_step + 1
            if start_step > self.config.total_steps:
                self._write_manifest("completed")
                return self.latest_metrics

            base_step = getattr(self, "initial_step", 0)
            stage_steps = max(1, self.config.total_steps - base_step)
            self._clone_warmup_tokens_seen = int(getattr(self, "_clone_warmup_tokens_seen", 0))
            self._logged_clone_pure_rl_transition = False
            warm_continuation = bool(getattr(self, "_optimizer_restored", False) and base_step > 0)
            if warm_continuation:
                progress0 = min(1.0, max(0.0, (start_step - 1) / max(1, self.config.total_steps)))
            else:
                progress0 = max(0.0, (start_step - 1 - base_step) / stage_steps)
            st_idx0, st_prog0 = self._mitotic_stage_progress(start_step - 1)
            next_indices = self.sampler.take(
                self.config.batch_size,
                progress=progress0,
                stage_idx=st_idx0,
                stage_progress=st_prog0,
            )
            self.prefetcher.schedule(next_indices)

            for step in range(start_step, self.config.total_steps + 1):
                if self.stop_event.is_set():
                    status = "stopped"
                    break

                t_step0 = time.perf_counter()
                self.current_step = step
                mitosis_event = self.perform_bounded_mitosis(step)
                if self.config.top_k_milestones:
                    desired_k = int(self.model.backbone.config.top_k)
                    for m_step, target_k in self.config.top_k_milestones:
                        if step >= int(m_step):
                            desired_k = min(int(target_k), int(self.model.backbone.config.num_experts))
                    cur_k_now = int(self.model.backbone.config.top_k)
                    if desired_k != cur_k_now:
                        self.config.top_k = desired_k
                        self.model.backbone.config.top_k = desired_k
                        for blk in list(self.model.backbone.encoder_layers) + list(
                            self.model.backbone.decoder_layers
                        ):
                            blk.moe.top_k = desired_k
                        print(
                            f"  [density-ramp @ step {step}] top_k={cur_k_now}->{desired_k} | "
                            f"experts={self.model.backbone.config.num_experts}",
                            flush=True,
                        )
                if warm_continuation:
                    progress = min(1.0, max(0.0, step / max(1, self.config.total_steps)))
                else:
                    progress = min(1.0, max(0.0, (step - base_step) / stage_steps))
                st_idx, st_prog = self._mitotic_stage_progress(step)
                cur_weights = self.schedule.weights_at(
                    progress, stage_idx=st_idx, stage_progress=st_prog
                )
                w_dec = float(cur_weights.get("typed_decision", 0.15))

                lr = self._compute_lr(step)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr * float(group.get("lr_scale", 1.0))

                cur_indices = next_indices
                quotas = self.sampler.allocate_quotas(
                    self.config.batch_size,
                    progress=progress,
                    stage_idx=st_idx,
                    stage_progress=st_prog,
                )
                if step < self.config.total_steps:
                    next_indices = self.sampler.take(
                        self.config.batch_size,
                        progress=progress,
                        stage_idx=st_idx,
                        stage_progress=st_prog,
                    )

                micro_batches = self.prefetcher.get(cur_indices)
                active_seq_len = self._active_stage_seq_len(st_idx)
                if active_seq_len < self.dataset.seq_len:
                    micro_batches = [
                        self._slice_micro_batch_to_seq_len(mb, active_seq_len)
                        for mb in micro_batches
                    ]
                cur_e = int(self.model.backbone.config.num_experts)
                cur_k = int(self.model.backbone.config.top_k)
                if cur_e >= 96:
                    max_mb_cap = (
                        1
                        if (
                            (cur_e >= 256 and cur_k >= 16)
                            or (cur_e >= 96 and int(self.backbone_cfg.d_model) >= 1536)
                        )
                        else (2 if cur_e >= 256 else (4 if cur_e >= 128 else 8))
                    )
                    split_mbs: list[dict[str, torch.Tensor]] = []
                    for mb in micro_batches:
                        n_rows = int(mb["input_ids"].shape[0])
                        if n_rows > max_mb_cap:
                            for s_i in range(0, n_rows, max_mb_cap):
                                split_mbs.append({k: v[s_i : s_i + max_mb_cap] for k, v in mb.items()})
                        else:
                            split_mbs.append(mb)
                    micro_batches = split_mbs
                if step < self.config.total_steps:
                    self.prefetcher.schedule(next_indices)

                local_step = max(1, step - base_step)
                has_clone_warmup_cfg = bool(
                    int(self.config.clone_warmup_steps) > 0 or int(self.config.clone_warmup_tokens) > 0
                )
                in_clone_warmup = bool(
                    not getattr(self, "_resumed_in_clone_pure_rl", False)
                    and (
                        (int(self.config.clone_warmup_steps) > 0 and local_step <= int(self.config.clone_warmup_steps))
                        or (
                            int(self.config.clone_warmup_tokens) > 0
                            and self._clone_warmup_tokens_seen < int(self.config.clone_warmup_tokens)
                        )
                    )
                )
                in_clone_pure_rl = bool(
                    not in_clone_warmup
                    and self.config.pure_rl_after_clone_warmup
                    and (has_clone_warmup_cfg or self.config.freeze_non_clone_experts)
                )
                if in_clone_pure_rl and not getattr(self, "_logged_clone_pure_rl_transition", False):
                    self._logged_clone_pure_rl_transition = True
                    for blk in list(self.model.backbone.encoder_layers) + list(
                        self.model.backbone.decoder_layers
                    ):
                        blk.moe.mitotic_ortho_coef = 0.0
                    if self.backbone_cfg.preserve_residual_across_stacks:
                        for name, param in self.model.named_parameters():
                            if any(
                                k in name
                                for k in ("global_out_proj", "ple.gate_scale", "markov_gate", "feedback.gate")
                            ):
                                param.requires_grad_(False)
                    print(
                        f"  [mitosis-phase2 @ step {step}] Clone token warmup complete "
                        f"({self._clone_warmup_tokens_seen:,} warmup tokens) -> "
                        f"Switching to Pure RL on mitotic clones {getattr(self, '_plastic_expert_indices', [])} "
                        f"(non-clones frozen)",
                        flush=True,
                    )
                if in_clone_pure_rl and len(micro_batches) > 1:
                    micro_batches = micro_batches[:1]

                run_operators = bool(
                    not in_clone_warmup
                    and (
                        step == 1
                        or step == start_step
                        or step == self.config.total_steps
                        or in_clone_pure_rl
                        or (step % max(1, int(self.config.operator_every)) == 0)
                    )
                )
                dec_batch = self._next_decision_batch() if run_operators else None
                if dec_batch is not None and (
                    cur_e >= 128 or (cur_e >= 80 and int(self.backbone_cfg.d_model) >= 1536)
                ):
                    dec_cap = (
                        1
                        if (cur_e >= 256 or (cur_e >= 96 and int(self.backbone_cfg.d_model) >= 1536))
                        else 2
                    )
                    dec_batch = {k: v[:dec_cap] for k, v in dec_batch.items()}
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                step_total_gpu = torch.zeros((), device=self.device, dtype=torch.float32)
                step_lm_gpu = torch.zeros((), device=self.device, dtype=torch.float32)
                step_dec_gpu = torch.zeros((), device=self.device, dtype=torch.float32)
                step_ded_gpu = torch.zeros((), device=self.device, dtype=torch.float32)
                step_eval_gpu = torch.zeros((), device=self.device, dtype=torch.float32)
                step_meta_gpu = torch.zeros((), device=self.device, dtype=torch.float32)
                step_abs_gpu = torch.zeros((), device=self.device, dtype=torch.float32)
                step_ornith_gpu = torch.zeros((), device=self.device, dtype=torch.float32)
                step_tokens_gpu = torch.zeros((), device=self.device, dtype=torch.long)
                step_expert_load_gpu = torch.zeros(
                    (self.model.backbone.config.num_experts,),
                    device=self.device,
                    dtype=torch.float32,
                )
                track_sums_gpu = {
                    f"loss_{t}": torch.zeros((), device=self.device, dtype=torch.float32)
                    for t in self.dataset.track_names
                }
                track_counts_gpu = {
                    f"loss_{t}": torch.zeros((), device=self.device, dtype=torch.float32)
                    for t in self.dataset.track_names
                }

                ornith_telemetry: dict[str, Any] = {}
                parent_reached_ornith = bool(
                    warm_continuation
                    and base_step >= int(max(1, self._parent_total_steps) * float(self.config.ornith_warmup_ratio))
                )
                if (
                    self.config.curriculum_mode == "mitotic_staged"
                    and self.config.enable_mitosis
                    and float(self.config.ornith_warmup_ratio) <= 0.0
                ):
                    ornith_active = bool((st_idx or 0) >= 1 or parent_reached_ornith)
                else:
                    ornith_active = (
                        progress >= float(self.config.ornith_warmup_ratio)
                    ) or parent_reached_ornith
                run_ornith = bool(
                    not in_clone_warmup
                    and self.config.enable_ornith_rl
                    and (ornith_active or in_clone_pure_rl)
                    and (
                        in_clone_pure_rl
                        or step % max(1, self.config.ornith_every) == 0
                        or (step == 1 and ornith_active and float(self.config.ornith_warmup_ratio) <= 0.0)
                    )
                )

                use_amp = self.device.type == "cuda"
                num_mb = len(micro_batches)
                # Token-weighted micro-batch scaling eliminates gradient/metric variance when
                # sparse `0x8000`-masked `typed_decision` rows mix with dense `uta_trace` rows.
                log_detail = bool(
                    step == 1
                    or step == start_step
                    or step == self.config.total_steps
                    or self.config.total_steps <= 32
                    or step % 10 == 0
                    or mitosis_event is not None
                )
                mb_token_counts = [
                    max(1, int((mb["labels"] != -100).sum().item())) for mb in micro_batches
                ]
                mb_valid_counts = [
                    max(1, int((mb["input_ids"] != self.tokenizer.pad_id).sum().item()))
                    for mb in micro_batches
                ]
                total_mb_tokens = max(1, sum(mb_token_counts))
                total_valid_tokens = max(total_mb_tokens, sum(mb_valid_counts))
                if in_clone_warmup:
                    self._clone_warmup_tokens_seen += total_valid_tokens
                if log_detail or not hasattr(self, "_cached_bpt_ratio"):
                    raw_bytes = sum(
                        self.tokenizer.count_utf8_bytes(mb["labels"]) for mb in micro_batches
                    )
                    total_mb_bytes = max(total_mb_tokens, raw_bytes)
                    self._cached_bpt_ratio = float(total_mb_bytes) / float(max(1, total_mb_tokens))
                else:
                    total_mb_bytes = max(
                        total_mb_tokens,
                        int(round(total_mb_tokens * float(self._cached_bpt_ratio))),
                    )

                fast_low_param_step = bool(
                    in_clone_warmup
                    or (
                        self.config.mitosis_milestones
                        and ((cur_e <= 32 and not run_operators) or cur_e >= 128)
                    )
                )
                step_rec = 1 if fast_low_param_step else self.config.recurrence_steps
                step_ponder = False if fast_low_param_step else self.config.depth_ponder

                saved_warmup_frozen: list[torch.nn.Parameter] = []
                if in_clone_warmup and self.config.freeze_non_clone_experts:
                    active_clone_tags = tuple(
                        f".moe.experts.{int(c)}." for _, c in self._get_model_mitotic_pairs()
                    )
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and not (
                            any(tag in name for tag in active_clone_tags) or ".moe.router." in name
                        ):
                            saved_warmup_frozen.append(param)
                            param.requires_grad_(False)

                try:
                    lm_phase_scale = 0.10 if in_clone_pure_rl else 1.0
                    for mb_idx, (mb, mb_toks) in enumerate(zip(micro_batches, mb_token_counts)):
                        inp = mb["input_ids"].to(self.device, non_blocking=True)
                        lbl = mb["labels"].to(self.device, non_blocking=True)
                        t_ids = mb["track_ids"].to(self.device, non_blocking=True)
                        step_tokens_gpu = step_tokens_gpu + mb_toks
                        mb_frac = float(mb_toks) / float(total_mb_tokens)
                        use_gist_mb = bool(
                            not fast_low_param_step
                            and self.config.enable_gist_conditioning
                            and self.config.gist_conditioning_prob > 0.0
                            and (((step + mb_idx) * 2654435761) % 1000)
                            < int(self.config.gist_conditioning_prob * 1000)
                        )

                        with torch.autocast(
                            device_type=self.device.type,
                            dtype=torch.bfloat16,
                            enabled=use_amp,
                        ):
                            lm_out = self.model.forward_inference(
                                inp,
                                targets=lbl,
                                recurrence_steps=step_rec,
                                recurrent_kv=self.config.recurrent_kv,
                                depth_ponder=step_ponder,
                                deep_supervision_weight=(
                                    0.0 if fast_low_param_step else self.config.deep_supervision_weight
                                ),
                                condition_on_prefix_gist=use_gist_mb,
                                latent_prior_scale=(
                                    self.config.latent_prior_scale if use_gist_mb else 0.0
                                ),
                            )
                            assert lm_out.loss is not None and lm_out.ce_loss is not None
                            lm_scale = mb_frac * lm_phase_scale
                            scaled_lm_loss = lm_out.loss * lm_scale

                        step_lm_gpu = step_lm_gpu + (lm_out.ce_loss.detach().float() * mb_frac)
                        step_total_gpu = step_total_gpu + scaled_lm_loss.detach().float()
                        step_expert_load_gpu = step_expert_load_gpu + (
                            lm_out.expert_load.detach().float().mean(dim=0) / num_mb
                        )
                        if log_detail:
                            self._accumulate_per_track_losses_gpu(
                                lm_out.logits.detach(),
                                lbl,
                                t_ids,
                                track_sums_gpu,
                                track_counts_gpu,
                            )
                        del lm_out
                        scaled_lm_loss.backward()
                        del scaled_lm_loss
                finally:
                    for param in saved_warmup_frozen:
                        param.requires_grad_(True)

                if dec_batch is not None:
                    eff_w_dec = w_dec if w_dec > 0.0 else 0.12
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16,
                        enabled=use_amp,
                    ):
                        op_out = self.model.forward_all_operators(
                            **dec_batch,
                            recurrence_steps=step_rec,
                            depth_ponder=step_ponder,
                        )
                        assert op_out.decision.ce_loss is not None
                        step_dec_gpu = op_out.decision.ce_loss.detach().float()
                        step_ded_gpu = op_out.deduction_loss.detach().float()
                        step_eval_gpu = op_out.evaluation_loss.detach().float()
                        step_meta_gpu = op_out.metacognition_loss.detach().float()
                        step_abs_gpu = op_out.abstraction_loss.detach().float()
                        scaled_op_loss = eff_w_dec * op_out.total_loss
                    step_total_gpu = step_total_gpu + scaled_op_loss.detach().float()
                    del op_out
                    scaled_op_loss.backward()
                    del scaled_op_loss

                ncm_telemetry: dict[str, Any] = {}
                if (
                    not in_clone_warmup
                    and self.config.enable_ncm
                    and self.ncm_system is not None
                    and micro_batches
                    and step % max(1, int(self.config.ncm_every)) == 0
                ):
                    mb0_inp = micro_batches[0]["input_ids"].to(self.device, non_blocking=True)
                    ncm_len = int(self.ncm_system.config.target_length)
                    p_len = max(4, min(ncm_len, mb0_inp.shape[1] // 2))
                    p_ids = mb0_inp[:, :p_len]
                    t_slice = mb0_inp[:, p_len : p_len + ncm_len]
                    if t_slice.shape[1] < ncm_len:
                        pad_fill = torch.full(
                            (t_slice.shape[0], ncm_len - t_slice.shape[1]),
                            int(self.config.pad_id),
                            dtype=t_slice.dtype,
                            device=self.device,
                        )
                        t_slice = torch.cat([t_slice, pad_fill], dim=1)
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16,
                        enabled=use_amp,
                    ):
                        ncm_out = self.ncm_system.forward_training_step(p_ids, t_slice)
                        scaled_ncm_loss = float(self.config.ncm_weight) * ncm_out.total_loss
                        ncm_telemetry = {
                            "ncm_loss": round(float(ncm_out.total_loss.detach().item()), 5),
                            "ncm_ce_loss": round(float(ncm_out.ncm_ce_loss.item()), 5),
                            "ncm_vq_recon_loss": round(float(ncm_out.vq_recon_loss.item()), 5),
                            "ncm_recon_accuracy": round(float(ncm_out.recon_accuracy), 4),
                            "ncm_codebook_utilization": round(float(ncm_out.codebook_utilization), 4),
                            "ncm_per_level_acc": list(ncm_out.per_level_acc),
                        }
                    step_total_gpu = step_total_gpu + scaled_ncm_loss.detach().float()
                    del ncm_out
                    scaled_ncm_loss.backward()
                    del scaled_ncm_loss

                if run_ornith:
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16,
                        enabled=use_amp,
                    ):
                        if self.config.enable_ncm and self.ncm_system is not None:
                            ornith_out = self.ornith_improver.step_ncm_self_improvement(self.ncm_system)
                        else:
                            ornith_out = self.ornith_improver.step_self_improvement(self.model)
                        step_ornith_gpu = ornith_out.loss.detach().float()
                        scaled_ornith_loss = self.config.ornith_weight * ornith_out.loss
                        ornith_telemetry = {
                            "ornith_loss": round(float(step_ornith_gpu.item()), 5),
                            "ornith_task_reward": round(ornith_out.task_reward, 4),
                            "ornith_harness_reward": round(ornith_out.harness_reward, 4),
                            "ornith_rollout_reward": round(ornith_out.mean_rollout_reward, 4),
                            "ornith_novelty": round(ornith_out.novelty, 4),
                            "ornith_pass_rate": round(ornith_out.pass_rate, 4),
                            "ornith_raw_pp": round(ornith_out.raw_pp, 4),
                            "ornith_weighted_pp": round(ornith_out.weighted_pp, 4),
                            "ornith_total_pp": round(ornith_out.total_pp, 4),
                            "ornith_pp_weight": round(ornith_out.pp_weight, 5),
                            "ornith_star_difficulty": round(ornith_out.star_difficulty, 4),
                            "ornith_recursion_depth": int(ornith_out.recursion_depth),
                            "ornith_active_skill": str(ornith_out.active_skill),
                            "scaffolds_cached": ornith_out.scaffolds_cached,
                            "novelty_buffer_size": ornith_out.novelty_buffer_size,
                            "dream_cycle": ornith_out.dream_cycle,
                            "dream_replay_score": round(ornith_out.dream_replay_score, 5),
                            "dream_initial_score": round(ornith_out.dream_initial_score, 5),
                            "dream_accepted_revisions": ornith_out.dream_accepted_revisions,
                            "dream_replays_dreamed": ornith_out.dream_replays_dreamed,
                            "dream_worlds_archived": ornith_out.dream_worlds_archived,
                            "dream_active_policy": ornith_out.dream_active_policy,
                        }
                    step_total_gpu = step_total_gpu + scaled_ornith_loss.detach().float()
                    del ornith_out
                    scaled_ornith_loss.backward()
                    del scaled_ornith_loss
                    ep_telemetry = self._step_heldout_epistemic_rl(step, use_amp)
                    if ep_telemetry:
                        ornith_telemetry.update(ep_telemetry)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
                if self.config.use_nvfp4:
                    bump_nvfp4_optimizer_step()

                # Single GPU-to-CPU sync point per step after optimizer.step()
                step_unmasked_tokens = int(step_tokens_gpu.item())
                step_lm_loss = float(step_lm_gpu.item())
                if dec_batch is not None:
                    step_dec_loss = float(step_dec_gpu.item())
                    step_ded_loss = float(step_ded_gpu.item())
                    step_eval_loss = float(step_eval_gpu.item())
                    step_meta_loss = float(step_meta_gpu.item())
                    step_abs_loss = float(step_abs_gpu.item())
                else:
                    step_dec_loss = float(self.latest_metrics.get("decision_loss", 0.0))
                    step_ded_loss = float(self.latest_metrics.get("deduction_loss", 0.0))
                    step_eval_loss = float(self.latest_metrics.get("evaluation_loss", 0.0))
                    step_meta_loss = float(self.latest_metrics.get("metacognition_loss", 0.0))
                    step_abs_loss = float(self.latest_metrics.get("abstraction_loss", 0.0))
                step_total_loss = float(step_total_gpu.item())
                expert_loads = [round(float(x), 4) for x in step_expert_load_gpu.cpu().tolist()]
                prev_ema = getattr(self, "_ema_expert_loads", None)
                if prev_ema is not None and len(prev_ema) == len(expert_loads):
                    self._ema_expert_loads = [
                        0.9 * float(p) + 0.1 * float(c) for p, c in zip(prev_ema, expert_loads)
                    ]
                else:
                    self._ema_expert_loads = [float(c) for c in expert_loads]
                step_elapsed = time.perf_counter() - t_step0
                self.cumulative_wall_s += step_elapsed
                self.cumulative_tokens += total_valid_tokens
                tok_per_s = total_valid_tokens / max(step_elapsed, 1e-6)

                avg_track_losses: dict[str, float] = {}
                for t_name in self.dataset.track_names:
                    key = f"loss_{t_name}"
                    if log_detail:
                        cnt = float(track_counts_gpu[key].item())
                        if cnt > 0:
                            avg_track_losses[key] = float(track_sums_gpu[key].item()) / cnt
                        elif key in self.latest_metrics:
                            avg_track_losses[key] = self.latest_metrics[key]
                    elif key in self.latest_metrics:
                        avg_track_losses[key] = self.latest_metrics[key]

                if not ornith_telemetry and self.latest_metrics:
                    for ok in (
                        "ornith_loss",
                        "ornith_task_reward",
                        "ornith_harness_reward",
                        "ornith_rollout_reward",
                        "ornith_novelty",
                        "ornith_pass_rate",
                        "scaffolds_cached",
                        "novelty_buffer_size",
                        "dream_cycle",
                        "dream_replay_score",
                        "dream_initial_score",
                        "dream_accepted_revisions",
                        "dream_replays_dreamed",
                        "dream_worlds_archived",
                        "dream_active_policy",
                    ):
                        if ok in self.latest_metrics:
                            ornith_telemetry[ok] = self.latest_metrics[ok]

                run_val = bool(
                    self.config.eval_every > 0
                    and (
                        step == 1
                        or step == start_step
                        or step % self.config.eval_every == 0
                        or step % self.config.checkpoint_every == 0
                        or step == self.config.total_steps
                        or (mitosis_event is not None and mitosis_event.get("action") == "expand")
                    )
                )
                val_telemetry: dict[str, Any] = {}
                if run_val:
                    val_telemetry = self._evaluate_validation()
                if not val_telemetry and self.latest_metrics:
                    for vk in list(self.latest_metrics.keys()):
                        if vk.startswith("val_"):
                            val_telemetry[vk] = self.latest_metrics[vk]

                vram_alloc_mb = (
                    torch.cuda.memory_allocated(0) / (1024**2) if self.device.type == "cuda" else 0.0
                )
                vram_res_mb = (
                    torch.cuda.memory_reserved(0) / (1024**2) if self.device.type == "cuda" else 0.0
                )
                gemm_counts = (
                    get_gemm_counts()
                    if self.config.use_nvfp4
                    else {"forward": 0, "dgrad": 0, "wgrad": 0, "sparse_2to4": 0}
                )

                active_pairs = self._get_model_mitotic_pairs()
                released_pairs = list(getattr(self, "_released_mitotic_pairs", []))
                plastic_experts = list(getattr(self, "_plastic_expert_indices", []))
                mitosis_phase = (
                    "clone_token_warmup"
                    if in_clone_warmup
                    else ("clone_pure_rl" if in_clone_pure_rl else "joint")
                )
                record: dict[str, Any] = {
                    "step": step,
                    "total_steps": self.config.total_steps,
                    "progress": round(progress, 4),
                    "lr": round(lr, 6),
                    "total_loss": round(step_total_loss, 5),
                    "lm_loss": round(step_lm_loss, 5),
                    "train_bpt": round(step_lm_loss / math.log(2.0), 5),
                    "train_bpb": round(
                        (step_lm_loss * step_unmasked_tokens / math.log(2.0))
                        / max(1, total_mb_bytes),
                        5,
                    ),
                    "decision_loss": round(step_dec_loss, 5),
                    "deduction_loss": round(step_ded_loss, 5),
                    "evaluation_loss": round(step_eval_loss, 5),
                    "metacognition_loss": round(step_meta_loss, 5),
                    "abstraction_loss": round(step_abs_loss, 5),
                    "tok_per_s": round(tok_per_s, 1),
                    "step_tokens": step_unmasked_tokens,
                    "cumulative_tokens": self.cumulative_tokens,
                    "wall_time_s": round(self.cumulative_wall_s, 2),
                    "step_time_s": round(step_elapsed, 3),
                    "vram_allocated_mb": round(vram_alloc_mb, 1),
                    "vram_reserved_mb": round(vram_res_mb, 1),
                    "num_experts": self.model.backbone.config.num_experts,
                    "top_k": int(self.model.backbone.config.top_k),
                    "expert_load": expert_loads,
                    "quotas": quotas,
                    "weights": {k: round(v, 4) for k, v in cur_weights.items()},
                    "nvfp4_gemms": gemm_counts["forward"] + gemm_counts["dgrad"] + gemm_counts["wgrad"],
                    "nvfp4_sparse_2to4_gemms": gemm_counts.get("sparse_2to4", 0),
                    "mitosis_phase": mitosis_phase,
                    "clone_warmup_tokens_seen": int(self._clone_warmup_tokens_seen),
                    **({"plastic_experts": plastic_experts} if plastic_experts else {}),
                    **({"mitotic_pairs": [list(p) for p in active_pairs]} if active_pairs else {}),
                    **(
                        {"released_mitotic_pairs": [list(p) for p in released_pairs]}
                        if released_pairs
                        else {}
                    ),
                    **({"mitosis_event": mitosis_event} if mitosis_event is not None else {}),
                    **ornith_telemetry,
                    **ncm_telemetry,
                    **val_telemetry,
                    **{k: round(v, 5) for k, v in avg_track_losses.items()},
                }
                self.latest_metrics = record
                with self.metrics_path.open("a", encoding="utf-8", newline="\n") as fh:
                    fh.write(json.dumps(record, sort_keys=True) + "\n")

                sample_trace: str | None = None
                if (
                    step % self.config.sample_every == 0
                    or step == 1
                    or step == start_step
                    or step == self.config.total_steps
                ):
                    sample_trace = self.controller.generate_uta(
                        self.config.sample_prompt,
                        max_new_tokens=256,
                    )
                    sample_entry: dict[str, Any] = {
                        "step": step,
                        "prompt": self.config.sample_prompt,
                        "trace": sample_trace,
                        "total_loss": record["total_loss"],
                        "timestamp": time.time(),
                    }
                    if self.config.enable_ncm and self.ncm_system is not None:
                        try:
                            tok = self.tokenizer
                            p_toks = tok.encode(
                                f"#U$# {self.config.sample_prompt}\n#T$# ",
                                add_bos=True,
                                add_eos=False,
                            )
                            p_ids_t = torch.tensor([p_toks], dtype=torch.long, device=self.device)
                            with torch.autocast(
                                device_type=self.device.type,
                                dtype=torch.bfloat16,
                                enabled=use_amp,
                            ):
                                ncm_gen = self.ncm_system.generate(
                                    p_ids_t,
                                    temperature=0.0,
                                    enable_mid_hierarchy_coe=True,
                                    enable_early_exit=False,
                                    tokenizer=tok,
                                )
                            sample_entry["ncm_trace"] = ncm_gen.text
                            sample_entry["ncm_levels_executed"] = ncm_gen.levels_executed
                        except Exception:
                            pass
                    with self.samples_path.open("a", encoding="utf-8", newline="\n") as sh:
                        sh.write(json.dumps(sample_entry, sort_keys=True) + "\n")

                if step % self.config.checkpoint_every == 0 or step == self.config.total_steps:
                    if self.config.clear_consumed_shards and hasattr(self.sampler, "evict_consumed_shards"):
                        ckpt_cleared = self.sampler.evict_consumed_shards(delete_from_disk=True)
                        if ckpt_cleared:
                            record["cleared_shards"] = ckpt_cleared
                            self.latest_metrics = record
                    if step == self.config.total_steps:
                        self._write_manifest("completed")
                    self.save_checkpoint(sample_uta=sample_trace)

                if log_detail and step < self.config.total_steps:
                    self._write_manifest("running")
                if self.on_step_callback is not None:
                    self.on_step_callback(record)

                stop_file = self.run_dir / ".stop_requested"
                if stop_file.exists():
                    try:
                        stop_file.unlink(missing_ok=True)
                    except OSError:
                        pass
                    self.stop_event.set()
                if self.stop_event.is_set():
                    status = "stopped"
                    break

            if status == "stopped":
                self._write_manifest("stopped")
                self.save_checkpoint(tag=f"step_{self.current_step:05d}_stopped")
        except Exception:
            status = "failed"
            raise
        finally:
            self.prefetcher.close()
            self._write_manifest(status)
            hf_t = getattr(self, "_last_hf_backup_thread", None)
            if hf_t is not None and hf_t.is_alive():
                hf_t.join()
            if self.device.type == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
        return self.latest_metrics


def list_runs(runs_root: str | Path = r"X:\MCL4-COE\runs") -> list[dict[str, Any]]:
    """Return all training runs sorted newest-first from `runs_root`."""
    root = resolve_runs_root(runs_root)
    if not root.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    for sub in root.iterdir():
        manifest = sub / "run_manifest.json"
        if manifest.is_file():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                runs.append(data)
            except Exception:
                continue
    runs.sort(key=lambda r: float(r.get("updated_at", 0.0)), reverse=True)
    return runs


def delete_run(run_id: str, runs_root: str | Path = r"X:\MCL4-COE\runs") -> bool:
    """Delete a specific run directory and its checkpoints from `runs_root`."""
    import shutil

    root = resolve_runs_root(runs_root)
    target = (root / run_id).resolve()
    if target.parent != root or not target.is_dir():
        return False
    shutil.rmtree(target, ignore_errors=True)
    return not target.exists()


def clear_all_runs(
    runs_root: str | Path = r"X:\MCL4-COE\runs",
    *,
    exclude_run_id: str | None = None,
) -> int:
    """Remove all completed/historical run directories under `runs_root`."""
    import shutil

    root = resolve_runs_root(runs_root)
    if not root.is_dir():
        return 0
    removed = 0
    for sub in list(root.iterdir()):
        if not sub.is_dir():
            continue
        if exclude_run_id and sub.name == exclude_run_id:
            continue
        shutil.rmtree(sub, ignore_errors=True)
        if not sub.exists():
            removed += 1
    return removed


def load_run_metrics(
    run_id: str,
    runs_root: str | Path = r"X:\MCL4-COE\runs",
) -> dict[str, Any]:
    """Load `run_manifest.json`, `metrics.jsonl`, and `samples.jsonl` for a specific `run_id`."""
    run_dir = resolve_runs_root(runs_root) / run_id
    manifest_file = run_dir / "run_manifest.json"
    metrics_file = run_dir / "metrics.jsonl"
    samples_file = run_dir / "samples.jsonl"

    manifest = (
        json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest_file.is_file()
        else {"run_id": run_id}
    )
    points: list[dict[str, Any]] = []
    if metrics_file.is_file():
        for line in metrics_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    points.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    samples: list[dict[str, Any]] = []
    if samples_file.is_file():
        for line in samples_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return {
        "manifest": manifest,
        "metrics": points,
        "samples": samples,
    }


def list_checkpoints(runs_root: str | Path = r"X:\MCL4-COE\runs") -> list[dict[str, Any]]:
    """List all saved checkpoint `.meta.json` receipts across all runs."""
    root = resolve_runs_root(runs_root)
    if not root.is_dir():
        return []
    ckpts: list[dict[str, Any]] = []
    for meta_file in root.rglob("*.meta.json"):
        try:
            entry = json.loads(meta_file.read_text(encoding="utf-8"))
            entry["meta_file"] = str(meta_file.resolve())
            pt_candidate = Path(entry.get("pt_file", ""))
            if not pt_candidate.exists():
                entry["pt_file"] = str(meta_file.with_name(pt_candidate.name).resolve())
            ckpts.append(entry)
        except Exception:
            continue
    ckpts.sort(key=lambda c: (float(c.get("timestamp", 0.0)), int(c.get("step", 0))), reverse=True)
    return ckpts


@torch.no_grad()
def evaluate_checkpoint_prompt(
    meta_or_pt_path: str | Path,
    prompt: str,
    *,
    max_new_tokens: int = 320,
) -> dict[str, Any]:
    """Load a saved checkpoint and generate a `#U$# -> #T$# -> #A$#` trace for interactive comparison."""
    p = Path(meta_or_pt_path).expanduser().resolve()
    if p.name.endswith(".meta.json"):
        meta = json.loads(p.read_text(encoding="utf-8"))
        pt_candidate = Path(meta["pt_file"])
        pt_path = pt_candidate if pt_candidate.exists() else p.with_name(pt_candidate.name)
    else:
        pt_path = p
        meta = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    b_cfg = MCL4CEDConfig(**payload["backbone_config"])
    c_cfg = COEConfig(**payload["coe_config"])
    from contextlib import ExitStack
    from unittest.mock import patch

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(target_dtype)
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(torch.nn.Linear, "reset_parameters", lambda self: None))
            stack.enter_context(patch.object(torch.nn.Embedding, "reset_parameters", lambda self: None))
            stack.enter_context(patch("torch.nn.init.normal_", lambda tensor, *a, **kw: tensor))
            model = CognitiveOperatorEnsemble(MCL4CEDBackbone(b_cfg), c_cfg)
    finally:
        torch.set_default_dtype(prev_dtype)
    model.load_state_dict(payload["model_state"], strict=False)
    mp_list = [tuple(pair) for pair in payload.get("mitotic_pairs", [])]
    if mp_list and model.backbone.encoder_layers and not model.backbone.encoder_layers[0].moe.mitotic_pairs:
        for layer in list(model.backbone.encoder_layers) + list(model.backbone.decoder_layers):
            layer.moe.mitotic_pairs = list(mp_list)
    model.to(device=device, dtype=target_dtype).eval()

    controller = MultiPassController(model)
    t0 = time.perf_counter()
    trace = controller.generate_uta(prompt, max_new_tokens=max_new_tokens)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "checkpoint_id": meta.get("checkpoint_id", pt_path.stem),
        "run_id": meta.get("run_id", ""),
        "step": payload.get("step", 0),
        "prompt": prompt,
        "trace": trace,
        "latency_ms": round(elapsed_ms, 2),
        "metrics": payload.get("latest_metrics", {}),
    }
