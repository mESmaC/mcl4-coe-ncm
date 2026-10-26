"""OpenCode-Style Interactive Inference & Testing Frontend for `MCL4-COE`.

Zero external web dependencies (uses Python's built-in `http.server.ThreadingHTTPServer`).
Provides a sleek, dark monospace developer interface inspired by OpenCode for interacting
with `MCL4-COE` checkpoints and testing inference-time capabilities:
- Local `.pt` / `.meta.json` checkpoint loading + on-demand Hugging Face Hub (`mesmac/mcl4-coe-runs`)
  checkpoint pulling and caching + instant preset model initialization.
- Interactive `#U$# -> collapsible #T$# -> #A$#` trace sessions across `Direct UTA`,
  `COE 9-Operator Verification Loop`, and `2-Round Inference-Time Dream-RL`.
- Live $O(1)$ inference-time controls: `recurrence_steps`, `adaptive_recurrent_depth`,
  `depth_ponder`, `recurrent_kv`, active MoE `top_k`, multi-active coalition `dense_subset_k`,
  `combine_dense_and_sparse`, Tabu `recurrent_coverage_penalty`, and `release_mitotic_pairs`.
- Live per-layer MoE expert routing heatmap & active parameter readout.
- Quarantined Whiteboard ("Einstein on a Whiteboard") fast-weight ingestion (`prose`/`code`/`web`),
  Hopfield-Kanerva rule ledger, 3-gate safety telemetry, and mitotic child sleep consolidation.
- Windowless Demand-Paged Context (`InfiniteContextPager` + `WorkingStateLedger`) & MCQA Decision Probe.
"""

from __future__ import annotations

from dataclasses import asdict, fields
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import gc
import json
import os
from pathlib import Path
import re
import threading
import time
import traceback
from typing import Any
from urllib.parse import urlparse

import torch

from .backbone import MCL4CEDBackbone, MCL4CEDConfig
from .context_memory import InfiniteContextPager, WorkingStateLedger
from .data import DecisionExample, collate_decisions, get_tokenizer
from .dream_rsi import DreamContextCache
from .ensemble import CognitiveSchedule, MultiPassController
from .operators import COEConfig, CognitiveOperatorEnsemble
from .ornith_rl import _verify_thought_equations, parameter_count, verify_python_rollout
from .trainer import list_checkpoints, resolve_runs_root


def _split_uta_segments(trace_text: str, fallback_prompt: str = "") -> dict[str, str]:
    """Split a `#U$# ... #T$# ... #A$# ...` trace into `(user, thought, answer)` strings."""
    raw = trace_text.strip()
    u_part = fallback_prompt.strip()
    t_part = ""
    a_part = ""

    if "#T$#" in raw:
        before_t, after_t = raw.split("#T$#", 1)
        if "#U$#" in before_t:
            u_part = before_t.split("#U$#", 1)[1].strip() or u_part
        if "#A$#" in after_t:
            t_raw, a_raw = after_t.split("#A$#", 1)
            t_part = t_raw.strip()
            a_part = a_raw.strip()
        else:
            t_part = after_t.strip()
    elif "#A$#" in raw:
        before_a, after_a = raw.split("#A$#", 1)
        if "#U$#" in before_a:
            u_part = before_a.split("#U$#", 1)[1].strip() or u_part
        a_part = after_a.strip()
    else:
        a_part = raw

    return {
        "user": u_part,
        "thought": t_part,
        "answer": a_part,
    }


def _compute_active_parameter_stats(model: CognitiveOperatorEnsemble) -> dict[str, Any]:
    """Compute total vs. active parameter counts per token under current `top_k` & `dense_subset_k`."""
    total_params = int(parameter_count(model))
    b_cfg = model.backbone.config
    num_e = max(1, int(b_cfg.num_experts))
    top_k = min(num_e, max(1, int(b_cfg.top_k)))
    dense_k = min(num_e, max(0, int(b_cfg.dense_subset_k)))
    if dense_k > 0:
        if b_cfg.combine_dense_and_sparse:
            eff_k = min(num_e, dense_k + top_k)
        else:
            eff_k = dense_k
    else:
        eff_k = top_k

    moe_layers = list(model.backbone.encoder_layers) + list(model.backbone.decoder_layers)
    expert_params_per_slot = 0
    for blk in moe_layers:
        if len(blk.moe.experts) > 0:
            expert_params_per_slot += sum(p.numel() for p in blk.moe.experts[0].parameters())

    total_expert_params = expert_params_per_slot * num_e
    shared_non_expert_params = max(0, total_params - total_expert_params)
    active_expert_params = expert_params_per_slot * eff_k
    active_params = shared_non_expert_params + active_expert_params
    active_pct = round(100.0 * active_params / max(1, total_params), 2)

    return {
        "total_parameters": total_params,
        "shared_parameters": shared_non_expert_params,
        "expert_parameters_per_column": expert_params_per_slot,
        "active_experts_per_token": eff_k,
        "active_parameters": active_params,
        "active_percent": active_pct,
    }


def _collect_layer_expert_loads(model: CognitiveOperatorEnsemble) -> dict[str, Any]:
    """Read `last_selected_mask` across all encoder & decoder MoE layers from the latest forward pass."""
    enc_loads: list[list[float]] = []
    dec_loads: list[list[float]] = []
    enc_top_experts: list[list[int]] = []
    dec_top_experts: list[list[int]] = []

    for blk in model.backbone.encoder_layers:
        mask = getattr(blk.moe, "last_selected_mask", None)
        if mask is not None and mask.numel() > 0:
            vec = mask.detach().float().reshape(-1, blk.moe.num_experts).mean(dim=0)
            loads = [round(float(x), 4) for x in vec.cpu().tolist()]
            top_idx = [int(i) for i in vec.topk(min(4, blk.moe.num_experts)).indices.cpu().tolist()]
        else:
            loads = [0.0] * blk.moe.num_experts
            top_idx = []
        enc_loads.append(loads)
        enc_top_experts.append(top_idx)

    for blk in model.backbone.decoder_layers:
        mask = getattr(blk.moe, "last_selected_mask", None)
        if mask is not None and mask.numel() > 0:
            vec = mask.detach().float().reshape(-1, blk.moe.num_experts).mean(dim=0)
            loads = [round(float(x), 4) for x in vec.cpu().tolist()]
            top_idx = [int(i) for i in vec.topk(min(4, blk.moe.num_experts)).indices.cpu().tolist()]
        else:
            loads = [0.0] * blk.moe.num_experts
            top_idx = []
        dec_loads.append(loads)
        dec_top_experts.append(top_idx)

    return {
        "encoder_loads": enc_loads,
        "decoder_loads": dec_loads,
        "encoder_top_experts": enc_top_experts,
        "decoder_top_experts": dec_top_experts,
    }


class InferenceSessionController:
    """Thread-safe controller managing the active `MCL4-COE` checkpoint, inference knobs, and session state."""

    DEFAULT_HF_REPO = "mesmac/mcl4-coe-runs"

    def __init__(
        self,
        runs_root: str | Path | None = None,
        hf_repo: str = DEFAULT_HF_REPO,
    ) -> None:
        if runs_root is None:
            repo_runs = Path(__file__).resolve().parents[2] / "runs"
            self.runs_root = repo_runs if repo_runs.is_dir() else resolve_runs_root()
        else:
            self.runs_root = Path(runs_root).expanduser().resolve()
        self.hf_repo = hf_repo

        self._lock = threading.RLock()
        self.model: CognitiveOperatorEnsemble | None = None
        self.controller: MultiPassController | None = None
        self.wb_controller: Any | None = None
        self.pager: InfiniteContextPager | None = None
        self.ledger: WorkingStateLedger | None = None
        self.dream_cache: DreamContextCache | None = None
        self.tokenizer: Any | None = None

        self.loaded_info: dict[str, Any] = {
            "loaded": False,
            "checkpoint_id": None,
            "run_id": None,
            "step": 0,
            "source": None,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "dtype": "bfloat16" if torch.cuda.is_available() else "float32",
        }
        self.inference_config: dict[str, Any] = {
            "mode": "uta",  # "uta" | "coe" | "dream_rl"
            "max_new_tokens": 256,
            "temperature": 0.0,
            "recurrence_steps": 2,
            "adaptive_recurrent_depth": False,
            "max_recurrence_steps": 3,
            "recurrent_entropy_threshold": 1.5,
            "depth_ponder": True,
            "recurrent_kv": True,
            "top_k": 4,
            "dense_subset_k": 0,
            "combine_dense_and_sparse": False,
            "recurrent_coverage_penalty": 0.35,
            "release_mitotic_pairs": False,
            "use_whiteboard": False,
            "use_windowless_pager": False,
        }
        self.history: list[dict[str, Any]] = []
        self._saved_mitotic_pairs: list[tuple[int, int]] = []

    def list_available_checkpoints(self) -> list[dict[str, Any]]:
        """Discover all local `.pt` and `.meta.json` checkpoints across `runs_root`."""
        roots: list[Path] = []
        if self.runs_root.is_dir():
            roots.append(self.runs_root)
        repo_runs = Path(__file__).resolve().parents[2] / "runs"
        if repo_runs.is_dir() and repo_runs.resolve() not in [r.resolve() for r in roots]:
            roots.append(repo_runs.resolve())
        x_runs = Path(r"X:\MCL4-COE\runs")
        if x_runs.is_dir() and x_runs.resolve() not in [r.resolve() for r in roots]:
            roots.append(x_runs.resolve())

        seen_ids: set[str] = set()
        items: list[dict[str, Any]] = []
        for root in roots:
            for entry in list_checkpoints(root):
                cid = str(entry.get("checkpoint_id") or "")
                if not cid or cid in seen_ids:
                    continue
                seen_ids.add(cid)
                pt_path = Path(str(entry.get("pt_file", "")))
                meta_path = Path(str(entry.get("meta_file", "")))
                if not pt_path.is_file() and meta_path.is_file():
                    alt_pt = meta_path.with_suffix("").with_suffix(".pt")
                    if alt_pt.is_file():
                        pt_path = alt_pt
                has_local_pt = pt_path.is_file()
                run_id = str(entry.get("run_id") or "")
                step_tag = meta_path.name.replace(".meta.json", "") if meta_path.name.endswith(".meta.json") else pt_path.stem
                hf_rel_path = f"{run_id}/checkpoints/{step_tag}.pt" if run_id else f"checkpoints/{step_tag}.pt"
                metrics = entry.get("metrics") or {}
                items.append(
                    {
                        "checkpoint_id": cid,
                        "run_id": run_id,
                        "step": int(entry.get("step", 0)),
                        "total_steps": int(entry.get("total_steps", 0)),
                        "parameter_count": int(entry.get("parameter_count", 0)),
                        "num_experts": int(entry.get("num_experts", 0)),
                        "has_local_pt": has_local_pt,
                        "pt_file": str(pt_path),
                        "meta_file": str(meta_path),
                        "hf_repo": self.hf_repo,
                        "hf_path": hf_rel_path,
                        "val_bpb": metrics.get("val_bpb"),
                        "val_mcqa_acc": metrics.get("val_mcqa_acc"),
                        "loss": metrics.get("total_loss"),
                        "timestamp": entry.get("timestamp", 0.0),
                    }
                )

            # Also scan any standalone `.pt` files without a `.meta.json`
            for pt_file in root.rglob("step_*.pt"):
                meta_sibling = pt_file.with_name(f"{pt_file.stem}.meta.json")
                if meta_sibling.is_file():
                    continue
                run_id = pt_file.parent.parent.name
                cid = f"{run_id}/{pt_file.stem}"
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                items.append(
                    {
                        "checkpoint_id": cid,
                        "run_id": run_id,
                        "step": int(re.sub(r"\D+", "", pt_file.stem) or 0),
                        "total_steps": 0,
                        "parameter_count": 0,
                        "num_experts": 0,
                        "has_local_pt": True,
                        "pt_file": str(pt_file.resolve()),
                        "meta_file": "",
                        "hf_repo": self.hf_repo,
                        "hf_path": f"{run_id}/checkpoints/{pt_file.name}",
                        "val_bpb": None,
                        "val_mcqa_acc": None,
                        "loss": None,
                        "timestamp": pt_file.stat().st_mtime,
                    }
                )

        items.sort(
            key=lambda c: (
                1 if c.get("has_local_pt") else 0,
                int(c.get("step", 0)),
                float(c.get("timestamp", 0.0)),
            ),
            reverse=True,
        )
        return items

    def _init_auxiliary_systems(self, run_dir: Path | None = None) -> None:
        """Initialize `MultiPassController`, `COEPlasticityController`, `InfiniteContextPager`, and `DreamContextCache`."""
        assert self.model is not None
        b_cfg = self.model.backbone.config
        self.controller = MultiPassController(
            self.model,
            schedule=CognitiveSchedule(
                operators=("abduction", "association", "decision", "deduction", "evaluation", "metacognition"),
                recurrence_steps=int(self.inference_config["recurrence_steps"]),
            ),
        )
        self.tokenizer = self.controller.tokenizer
        self.wb_controller = self.model.create_whiteboard_controller(
            whiteboard_slots=8,
            lora_rank=8,
        )
        self.pager = InfiniteContextPager(page_size=64, landmarks_per_page=4, top_k_pages=3)
        self.ledger = self.pager.ledger
        self.dream_cache = DreamContextCache()
        if run_dir is not None:
            for cand_rel in ("dream/dream_context_cache.json", "dream_out/dream_context_cache.json"):
                cand_p = run_dir / cand_rel
                if cand_p.is_file():
                    try:
                        self.dream_cache = DreamContextCache.load(cand_p)
                        break
                    except Exception:
                        pass

    def _sync_inference_knobs_to_model(self) -> None:
        """Apply `self.inference_config` routing/recurrence settings directly to `self.model` in $O(1)$."""
        if self.model is None:
            return
        b_cfg = self.model.backbone.config
        num_e = int(b_cfg.num_experts)
        top_k = min(num_e, max(1, int(self.inference_config.get("top_k", b_cfg.top_k))))
        dense_k = min(num_e, max(0, int(self.inference_config.get("dense_subset_k", 0))))
        comb = bool(self.inference_config.get("combine_dense_and_sparse", False))
        tabu = max(0.0, float(self.inference_config.get("recurrent_coverage_penalty", 0.0)))
        rel_pairs = bool(self.inference_config.get("release_mitotic_pairs", False))

        self.inference_config["top_k"] = top_k
        self.inference_config["dense_subset_k"] = dense_k

        b_cfg.top_k = top_k
        b_cfg.dense_subset_k = dense_k
        b_cfg.combine_dense_and_sparse = comb
        b_cfg.recurrent_coverage_penalty = tabu

        active_pairs = [] if rel_pairs else list(self._saved_mitotic_pairs)
        for blk in list(self.model.backbone.encoder_layers) + list(self.model.backbone.decoder_layers):
            blk.moe.top_k = top_k
            blk.moe.dense_subset_k = dense_k
            blk.moe.combine_dense_and_sparse = comb
            blk.moe.recurrent_coverage_penalty = tabu
            blk.moe.mitotic_pairs = list(active_pairs)
        if int(b_cfg.vocab_size) == 50304 and num_e > 64 and not active_pairs:
            self.model.backbone.set_lineage_family_routing(
                num_roots=64,
                top_families=min(64, max(8, top_k)),
                child_share=0.25,
            )
        elif hasattr(self.model.backbone, "set_lineage_family_routing"):
            self.model.backbone.set_lineage_family_routing(num_roots=0)

        rec_steps = max(1, int(self.inference_config.get("recurrence_steps", 2)))
        self.model.config.recurrence_steps = rec_steps
        self.model.config.depth_ponder = bool(self.inference_config.get("depth_ponder", True))
        self.model.config.recurrent_kv = bool(self.inference_config.get("recurrent_kv", True))
        if self.controller is not None:
            self.controller.schedule = CognitiveSchedule(
                operators=self.controller.schedule.operators,
                recurrence_steps=rec_steps,
            )

    def load_checkpoint(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Load a checkpoint from local `.pt`/`.meta.json` or download on-demand from Hugging Face Hub."""
        with self._lock:
            target_str = str(payload.get("checkpoint") or "").strip()
            hf_repo = str(payload.get("hf_repo") or self.hf_repo).strip() or self.DEFAULT_HF_REPO
            hf_token = str(payload.get("hf_token") or os.environ.get("HF_TOKEN") or "").strip() or None
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token

            req_device = str(payload.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")).lower()
            if req_device == "auto":
                req_device = "cuda" if torch.cuda.is_available() else "cpu"
            if req_device == "cuda" and not torch.cuda.is_available():
                req_device = "cpu"
            device = torch.device(req_device)

            req_dtype = str(payload.get("dtype") or ("bfloat16" if device.type == "cuda" else "float32")).lower()
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            target_dtype = dtype_map.get(req_dtype, torch.bfloat16 if device.type == "cuda" else torch.float32)

            if not target_str:
                raise ValueError("Specify a checkpoint path, checkpoint_id, or preset to load.")

            # Resolve target_str against discovered checkpoints or direct path
            discovered = self.list_available_checkpoints()
            matched_entry = next(
                (
                    c
                    for c in discovered
                    if c["checkpoint_id"] == target_str
                    or c["meta_file"] == target_str
                    or c["pt_file"] == target_str
                ),
                None,
            )

            meta: dict[str, Any] = {}
            run_dir: Path | None = None
            pt_path: Path | None = None
            source_label = "local"

            if matched_entry is not None:
                meta_file_str = matched_entry.get("meta_file") or ""
                if meta_file_str and Path(meta_file_str).is_file():
                    meta_p = Path(meta_file_str)
                    meta = json.loads(meta_p.read_text(encoding="utf-8"))
                    run_dir = meta_p.parent.parent
                pt_cand = Path(matched_entry["pt_file"]) if matched_entry.get("pt_file") else None
                if pt_cand and pt_cand.is_file():
                    pt_path = pt_cand
                else:
                    # Pull from Hugging Face Hub into the local run checkpoints directory
                    hf_rel = matched_entry["hf_path"]
                    local_dest_dir = (
                        Path(meta_file_str).parent.parent.parent
                        if (meta_file_str and Path(meta_file_str).is_file())
                        else self.runs_root
                    )
                    from huggingface_hub import hf_hub_download

                    downloaded = hf_hub_download(
                        repo_id=hf_repo,
                        filename=hf_rel,
                        repo_type="model",
                        token=hf_token,
                        local_dir=str(local_dest_dir),
                    )
                    pt_path = Path(downloaded).resolve()
                    source_label = f"hf://{hf_repo}/{hf_rel}"
            else:
                cand_p = Path(target_str).expanduser()
                if cand_p.is_file():
                    if cand_p.name.endswith(".meta.json"):
                        meta = json.loads(cand_p.read_text(encoding="utf-8"))
                        run_dir = cand_p.parent.parent
                        pt_in_meta = Path(str(meta.get("pt_file", "")))
                        local_sibling = cand_p.with_name(cand_p.name.replace(".meta.json", ".pt"))
                        if local_sibling.is_file():
                            pt_path = local_sibling
                        elif pt_in_meta.is_file():
                            pt_path = pt_in_meta
                        else:
                            run_id = str(meta.get("run_id") or cand_p.parent.parent.name)
                            step_tag = cand_p.name.replace(".meta.json", "")
                            hf_rel = f"{run_id}/checkpoints/{step_tag}.pt"
                            from huggingface_hub import hf_hub_download

                            downloaded = hf_hub_download(
                                repo_id=hf_repo,
                                filename=hf_rel,
                                repo_type="model",
                                token=hf_token,
                                local_dir=str(cand_p.parent.parent.parent),
                            )
                            pt_path = Path(downloaded).resolve()
                            source_label = f"hf://{hf_repo}/{hf_rel}"
                    else:
                        pt_path = cand_p.resolve()
                        run_dir = pt_path.parent.parent
                else:
                    # Treat target_str as an HF relative path e.g. "colab_mcl4_coe_v7_mitotic_3b_fast/checkpoints/step_05800.pt"
                    hf_rel = target_str if target_str.endswith(".pt") else f"{target_str}.pt"
                    if "/checkpoints/" not in hf_rel and "/" in hf_rel:
                        r_part, s_part = hf_rel.split("/", 1)
                        hf_rel = f"{r_part}/checkpoints/{s_part}"
                    from huggingface_hub import hf_hub_download

                    downloaded = hf_hub_download(
                        repo_id=hf_repo,
                        filename=hf_rel,
                        repo_type="model",
                        token=hf_token,
                        local_dir=str(self.runs_root),
                    )
                    pt_path = Path(downloaded).resolve()
                    run_dir = pt_path.parent.parent
                    source_label = f"hf://{hf_repo}/{hf_rel}"

            assert pt_path is not None and pt_path.is_file()

            # Free previous model memory before allocating new checkpoint
            if self.model is not None:
                self.model.cpu()
                del self.model
                self.model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            pt_bytes = pt_path.stat().st_size
            if device.type == "cuda":
                vram_total = torch.cuda.get_device_properties(device).total_memory
                if pt_bytes > int(vram_total * 0.90):
                    print(
                        f"[opencode // mcl4-coe] Checkpoint ({pt_bytes / 1e9:.2f} GB) exceeds GPU VRAM "
                        f"({vram_total / 1e9:.2f} GB); loading in bfloat16 on CPU with mmap=True.",
                        flush=True,
                    )
                    device = torch.device("cpu")
                    target_dtype = torch.bfloat16
                    req_dtype = "bfloat16"

            try:
                raw = torch.load(pt_path, map_location="cpu", weights_only=False, mmap=True)
            except TypeError:
                raw = torch.load(pt_path, map_location="cpu", weights_only=False)
            valid_b_keys = {f.name for f in fields(MCL4CEDConfig)}
            valid_c_keys = {f.name for f in fields(COEConfig)}
            src_b = raw.get("backbone_config") or raw.get("config") or meta.get("config") or {}
            src_c = raw.get("coe_config") or raw.get("config") or meta.get("config") or {}
            b_dict = {k: v for k, v in src_b.items() if k in valid_b_keys and v is not None}
            c_dict = {k: v for k, v in src_c.items() if k in valid_c_keys and v is not None}
            ms = raw.get("model_state") or {}
            rw0 = ms.get("backbone.encoder_layers.0.moe.router.weight")
            if isinstance(rw0, torch.Tensor) and rw0.ndim == 2:
                b_dict["num_experts"] = int(rw0.shape[0])
            mr0 = ms.get("backbone.encoder_layers.0.attention.markov_router.route_query.weight")
            n_h = int(b_dict.get("n_heads", 4))
            if isinstance(mr0, torch.Tensor) and mr0.ndim == 2 and mr0.shape[0] % n_h == 0:
                b_dict["markov_router_dim"] = int(mr0.shape[0] // n_h)
            if "dense_expert_indices" in b_dict and isinstance(b_dict["dense_expert_indices"], list):
                b_dict["dense_expert_indices"] = tuple(b_dict["dense_expert_indices"])

            b_cfg = MCL4CEDConfig(**b_dict)
            c_cfg = COEConfig(**c_dict)
            from contextlib import ExitStack
            from unittest.mock import patch

            prev_dtype = torch.get_default_dtype()
            torch.set_default_dtype(target_dtype)
            try:
                with ExitStack() as stack:
                    stack.enter_context(patch.object(torch.nn.Linear, "reset_parameters", lambda self: None))
                    stack.enter_context(patch.object(torch.nn.Embedding, "reset_parameters", lambda self: None))
                    stack.enter_context(patch("torch.nn.init.normal_", lambda tensor, *a, **kw: tensor))
                    with torch.device("meta" if device.type == "cpu" and pt_bytes > 8_000_000_000 else "cpu"):
                        model = CognitiveOperatorEnsemble(MCL4CEDBackbone(b_cfg), c_cfg)
            finally:
                torch.set_default_dtype(prev_dtype)
            use_assign = next(model.parameters()).is_meta
            model.load_state_dict(raw["model_state"], strict=False, assign=use_assign)
            if use_assign:
                for _, buf in model.named_buffers():
                    if buf.is_meta:
                        pass

            mp_list = [tuple(int(x) for x in pair) for pair in raw.get("mitotic_pairs", []) if len(pair) == 2]
            self._saved_mitotic_pairs = list(mp_list)
            for blk in list(model.backbone.encoder_layers) + list(model.backbone.decoder_layers):
                blk.moe.mitotic_pairs = list(mp_list)

            if not use_assign:
                model.to(device=device, dtype=target_dtype).eval()
            else:
                model.eval()
            for mod in model.modules():
                if isinstance(mod, torch.nn.LayerNorm):
                    mod.float()
            self.model = model

            # Adopt checkpoint's native top_k and recurrence_steps as initial UI defaults
            self.inference_config["top_k"] = int(b_cfg.top_k)
            self.inference_config["dense_subset_k"] = int(b_cfg.dense_subset_k)
            self.inference_config["combine_dense_and_sparse"] = bool(b_cfg.combine_dense_and_sparse)
            self.inference_config["recurrent_coverage_penalty"] = float(
                b_cfg.recurrent_coverage_penalty if b_cfg.recurrent_coverage_penalty > 0 else 0.35
            )
            self.inference_config["recurrence_steps"] = int(c_cfg.recurrence_steps)
            self.inference_config["depth_ponder"] = bool(c_cfg.depth_ponder)
            self.inference_config["recurrent_kv"] = bool(c_cfg.recurrent_kv)
            self.inference_config["release_mitotic_pairs"] = False

            self._init_auxiliary_systems(run_dir=run_dir)
            self._sync_inference_knobs_to_model()

            step_val = int(raw.get("step") or meta.get("step") or 0)
            run_id_val = str(
                meta.get("run_id")
                or (raw.get("config") or {}).get("run_id")
                or (run_dir.name if run_dir else pt_path.parent.name)
            )
            ckpt_id_val = str(meta.get("checkpoint_id") or f"{run_id_val}/{pt_path.stem}")

            self.loaded_info = {
                "loaded": True,
                "checkpoint_id": ckpt_id_val,
                "run_id": run_id_val,
                "step": step_val,
                "source": source_label if source_label != "local" else str(pt_path),
                "device": str(device),
                "dtype": req_dtype,
                "mitotic_pairs_count": len(self._saved_mitotic_pairs),
                "latest_metrics": raw.get("latest_metrics") or meta.get("metrics") or {},
            }
            del raw
            gc.collect()
            return self.status()

    def load_preset(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Initialize a fresh preset `MCL4-COE` model in memory for immediate inference testing."""
        with self._lock:
            preset = str(payload.get("preset") or "tiny_morph16k").strip()
            req_device = str(payload.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")).lower()
            if req_device == "cuda" and not torch.cuda.is_available():
                req_device = "cpu"
            device = torch.device(req_device)
            req_dtype = str(payload.get("dtype") or ("bfloat16" if device.type == "cuda" else "float32")).lower()
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            target_dtype = dtype_map.get(req_dtype, torch.float32)

            if self.model is not None:
                self.model.cpu()
                del self.model
                self.model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if preset == "v7_balanced_768_e8":
                b_cfg = MCL4CEDConfig(
                    vocab_size=16384,
                    d_model=768,
                    n_heads=8,
                    n_encoder_layers=4,
                    n_decoder_layers=4,
                    ple_dim=192,
                    expert_hidden_dim=384,
                    num_experts=8,
                    top_k=4,
                    max_seq_len=1024,
                    recurrent_coverage_penalty=0.35,
                )
                c_cfg = COEConfig(projection_dim=192, recurrence_steps=2, depth_ponder=True, recurrent_kv=True)
            elif preset == "v6_mitotic_e8":
                b_cfg = MCL4CEDConfig(
                    vocab_size=16384,
                    d_model=576,
                    n_heads=9,
                    n_encoder_layers=5,
                    n_decoder_layers=5,
                    ple_dim=144,
                    expert_hidden_dim=864,
                    num_experts=8,
                    top_k=2,
                    max_seq_len=1024,
                    recurrent_coverage_penalty=0.35,
                )
                c_cfg = COEConfig(projection_dim=144, recurrence_steps=2, depth_ponder=True, recurrent_kv=True)
            else:
                b_cfg = MCL4CEDConfig(
                    vocab_size=16384,
                    d_model=256,
                    n_heads=4,
                    n_encoder_layers=2,
                    n_decoder_layers=2,
                    ple_dim=64,
                    expert_hidden_dim=256,
                    num_experts=8,
                    top_k=2,
                    max_seq_len=512,
                    recurrent_coverage_penalty=0.35,
                )
                c_cfg = COEConfig(projection_dim=64, recurrence_steps=2, depth_ponder=True, recurrent_kv=True)

            model = CognitiveOperatorEnsemble(MCL4CEDBackbone(b_cfg), c_cfg)
            model.to(device=device, dtype=target_dtype).eval()
            for mod in model.modules():
                if isinstance(mod, torch.nn.LayerNorm):
                    mod.float()
            self.model = model
            self._saved_mitotic_pairs = [(0, 4), (1, 5)]
            self.inference_config["top_k"] = int(b_cfg.top_k)
            self.inference_config["dense_subset_k"] = 0
            self.inference_config["recurrence_steps"] = int(c_cfg.recurrence_steps)

            self._init_auxiliary_systems(run_dir=None)
            self._sync_inference_knobs_to_model()

            self.loaded_info = {
                "loaded": True,
                "checkpoint_id": f"preset/{preset}",
                "run_id": preset,
                "step": 0,
                "source": f"preset:{preset}",
                "device": str(device),
                "dtype": req_dtype,
                "mitotic_pairs_count": len(self._saved_mitotic_pairs),
                "latest_metrics": {},
            }
            return self.status()

    def update_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Update live inference-time routing, recurrence, and generation settings."""
        with self._lock:
            for k in (
                "mode",
                "max_new_tokens",
                "temperature",
                "recurrence_steps",
                "adaptive_recurrent_depth",
                "max_recurrence_steps",
                "recurrent_entropy_threshold",
                "depth_ponder",
                "recurrent_kv",
                "top_k",
                "dense_subset_k",
                "combine_dense_and_sparse",
                "recurrent_coverage_penalty",
                "release_mitotic_pairs",
                "use_whiteboard",
                "use_windowless_pager",
            ):
                if k in payload:
                    self.inference_config[k] = payload[k]
            self._sync_inference_knobs_to_model()
            return self.status()

    def status(self) -> dict[str, Any]:
        """Return full model, hardware, whiteboard, pager, and inference configuration status."""
        has_cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
        vram_alloc_mb = round(torch.cuda.memory_allocated(0) / (1024**2), 1) if has_cuda else 0.0
        vram_res_mb = round(torch.cuda.memory_reserved(0) / (1024**2), 1) if has_cuda else 0.0

        arch_info: dict[str, Any] = {}
        param_stats: dict[str, Any] = {}
        wb_info: dict[str, Any] = {
            "active": False,
            "episodic_rule_count": 0,
            "lora_frobenius_norm": 0.0,
            "prefix_delta_norm": 0.0,
            "rules": [],
            "revisions": [],
        }
        pager_info: dict[str, Any] = {
            "num_pages": 0,
            "total_tokens_paged": 0,
            "ledger_norm": 0.0,
        }

        if self.model is not None:
            b_cfg = self.model.backbone.config
            c_cfg = self.model.config
            arch_info = {
                "vocab_size": b_cfg.vocab_size,
                "tokenizer_type": (
                    "DonorHFTokenizer"
                    if getattr(self.tokenizer, "is_hf_donor", False)
                    else ("MorphCode-16K" if b_cfg.vocab_size > 1024 else "ByteTokenizer-260")
                ),
                "d_model": b_cfg.d_model,
                "n_heads": b_cfg.n_heads,
                "n_encoder_layers": b_cfg.n_encoder_layers,
                "n_decoder_layers": b_cfg.n_decoder_layers,
                "ple_dim": b_cfg.ple_dim,
                "expert_hidden_dim": b_cfg.expert_hidden_dim,
                "num_experts": b_cfg.num_experts,
                "top_k": b_cfg.top_k,
                "dense_subset_k": b_cfg.dense_subset_k,
                "combine_dense_and_sparse": b_cfg.combine_dense_and_sparse,
                "recurrent_coverage_penalty": b_cfg.recurrent_coverage_penalty,
                "max_seq_len": b_cfg.max_seq_len,
                "projection_dim": c_cfg.projection_dim,
                "active_mitotic_pairs": len(
                    self.model.backbone.encoder_layers[0].moe.mitotic_pairs
                    if self.model.backbone.encoder_layers
                    else []
                ),
                "saved_mitotic_pairs": len(self._saved_mitotic_pairs),
            }
            param_stats = _compute_active_parameter_stats(self.model)

        if self.wb_controller is not None:
            wb = self.wb_controller.whiteboard
            p_txts = getattr(wb, "_rule_premise_texts", []) or []
            c_txts = getattr(wb, "_rule_conc_texts", []) or []
            rules_list = [
                {"index": idx, "premise": p, "conclusion": c}
                for idx, (p, c) in enumerate(zip(p_txts, c_txts))
            ]
            wb_info = {
                "active": bool(self.inference_config.get("use_whiteboard", False)),
                "episodic_rule_count": int(wb.episodic_rule_count),
                "lora_frobenius_norm": round(float(wb.adapter_frobenius_norm()), 5),
                "prefix_delta_norm": round(float(wb.prefix_delta_norm()), 5),
                "rules": rules_list,
                "revisions": list(getattr(wb, "revision_history", [])),
            }

        if self.pager is not None:
            pager_info = {
                "num_pages": int(self.pager.num_pages),
                "page_size": int(self.pager.page_size),
                "landmarks_per_page": int(self.pager.landmarks_per_page),
                "total_tokens_paged": int(self.pager.total_ingested_tokens),
                "compressed_bytes": int(self.pager.total_compressed_bytes),
            }

        return {
            "loaded_info": self.loaded_info,
            "architecture": arch_info,
            "parameter_stats": param_stats,
            "inference_config": self.inference_config,
            "whiteboard": wb_info,
            "pager": pager_info,
            "dream_cache_entries": len(self.dream_cache) if self.dream_cache is not None else 0,
            "gpu_name": gpu_name,
            "vram_allocated_mb": vram_alloc_mb,
            "vram_reserved_mb": vram_res_mb,
            "history_count": len(self.history),
        }

    @torch.no_grad()
    def generate_turn(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute an interactive `#U$# -> #T$# -> #A$#` inference turn with full telemetry."""
        with self._lock:
            if self.model is None or self.controller is None or self.tokenizer is None:
                raise RuntimeError("No checkpoint loaded. Load a checkpoint or preset first.")

            # Apply any per-turn overrides
            self.update_config(payload)

            prompt = str(payload.get("prompt") or "").strip()
            if not prompt:
                raise ValueError("Prompt cannot be empty.")

            thought_prefix = payload.get("thought_prefix")
            if thought_prefix is not None:
                thought_prefix = str(thought_prefix).strip() or None
            answer_prefix = payload.get("answer_prefix")
            if answer_prefix is not None:
                answer_prefix = str(answer_prefix).strip() or None
            lock_thought = bool(payload.get("lock_thought", False))
            harness_h = str(payload.get("harness") or "").strip()

            cfg = self.inference_config
            mode = str(cfg.get("mode", "uta")).lower()
            max_new = max(16, min(1024, int(cfg.get("max_new_tokens", 256))))
            temp = max(0.0, float(cfg.get("temperature", 0.0)))
            rec_steps = max(1, int(cfg.get("recurrence_steps", 2)))
            use_wb = bool(cfg.get("use_whiteboard", False)) and self.wb_controller is not None
            wb_state = self.wb_controller.whiteboard if use_wb else None

            t0 = time.perf_counter()
            candidates_telemetry: list[dict[str, Any]] = []
            mode_meta: dict[str, Any] = {"mode": mode}

            if mode == "coe":
                coe_out = self.controller.generate_uta_with_coe(
                    prompt,
                    max_new_tokens=max_new,
                )
                trace_text = coe_out.text
                for c in coe_out.candidates:
                    candidates_telemetry.append(asdict(c))
                mode_meta.update(
                    {
                        "selected_index": coe_out.selected_index,
                        "improved_over_baseline": coe_out.improved_over_baseline,
                        "abstraction_norm": coe_out.abstraction_norm,
                        "num_candidates": len(coe_out.candidates),
                    }
                )
            elif mode == "dream_rl":
                dream_out = self.controller.generate_uta_with_dream_rl(
                    prompt,
                    dream_cache=self.dream_cache,
                    max_new_tokens=max_new,
                )
                trace_text = dream_out.text
                for c in dream_out.candidates:
                    candidates_telemetry.append(asdict(c))
                mode_meta.update(
                    {
                        "selected_index": dream_out.selected_index,
                        "selected_round": dream_out.selected_round,
                        "selected_strategy": dream_out.selected_strategy,
                        "improved_over_baseline": dream_out.improved_over_baseline,
                        "improved_in_round2": dream_out.improved_in_round2,
                        "round1_best_score": dream_out.round1_best_score,
                        "round2_best_score": dream_out.round2_best_score,
                        "final_best_score": dream_out.final_best_score,
                        "dream_cache_matched": dream_out.dream_cache_matched,
                        "dream_top_similarity": dream_out.dream_top_similarity,
                        "dream_recommended_scaffold": dream_out.dream_recommended_scaffold,
                        "round1_advantages": list(dream_out.round1_advantages),
                        "abstraction_norm": dream_out.abstraction_norm,
                        "num_candidates": len(dream_out.candidates),
                    }
                )
            elif mode == "ncm":
                ncm_sys = getattr(self.model, "ncm_system", None)
                if ncm_sys is None:
                    ncm_sys = self.model.create_ncm_system()
                p_ids = torch.tensor(
                    [self.tokenizer.encode(f"#U$# {prompt}", add_bos=True, add_eos=False)],
                    dtype=torch.long,
                    device=next(self.model.parameters()).device,
                )
                ncm_gen = ncm_sys.generate(
                    p_ids,
                    tokenizer=self.tokenizer,
                    target_length=int(ncm_sys.config.target_length),
                    temperature=temp,
                    enable_mid_hierarchy_coe=True,
                    enable_early_exit=bool(cfg.get("adaptive_recurrent_depth", False)),
                    decode_all_levels=True,
                )
                lvl_lines = [
                    f"Level {idx + 1} (ell={ncm_sys.config.levels[idx]}, top_k={ncm_gen.per_level_active_top_k[idx]}): {txt}"
                    for idx, txt in enumerate(ncm_gen.level_decoded_texts)
                ]
                thought_str = "\n".join(lvl_lines) if lvl_lines else "NCM coarse-to-fine hierarchy executed."
                trace_text = f"#U$# {prompt}\n#T$# {thought_str}\n#A$# {ncm_gen.text}"
                mode_meta.update(
                    {
                        "levels_executed": ncm_gen.levels_executed,
                        "total_levels": ncm_gen.total_levels,
                        "halted_early": ncm_gen.halted_early,
                        "lateral_refinements": ncm_gen.lateral_refinements,
                        "per_level_active_top_k": ncm_gen.per_level_active_top_k,
                        "per_level_metacognition": ncm_gen.per_level_metacognition,
                        "per_level_deduction": ncm_gen.per_level_deduction,
                        "level_decoded_texts": ncm_gen.level_decoded_texts,
                    }
                )
            else:
                trace_text = self.controller.generate_uta(
                    prompt,
                    max_new_tokens=max_new,
                    temperature=temp,
                    recurrence_steps=rec_steps,
                    thought_prefix=thought_prefix,
                    lock_thought=lock_thought,
                    answer_prefix=answer_prefix,
                    recurrent_kv=bool(cfg.get("recurrent_kv", True)),
                    depth_ponder=bool(cfg.get("depth_ponder", True)),
                    adaptive_recurrent_depth=bool(cfg.get("adaptive_recurrent_depth", False)),
                    max_recurrence_steps=int(cfg.get("max_recurrence_steps", 3)),
                    recurrent_entropy_threshold=float(cfg.get("recurrent_entropy_threshold", 1.5)),
                    whiteboard=wb_state,
                )

            elapsed_s = max(1e-6, time.perf_counter() - t0)
            segments = _split_uta_segments(trace_text, fallback_prompt=prompt)
            prompt_toks = self.tokenizer.encode(prompt, add_bos=False, add_eos=False)
            thought_toks = self.tokenizer.encode(segments["thought"], add_bos=False, add_eos=False) if segments["thought"] else []
            answer_toks = self.tokenizer.encode(segments["answer"], add_bos=False, add_eos=False) if segments["answer"] else []
            gen_tok_count = len(thought_toks) + len(answer_toks)
            tok_per_s = round(gen_tok_count / elapsed_s, 1)

            # Optional Windowless Context Pager update
            if bool(cfg.get("use_windowless_pager", False)) and self.pager is not None:
                full_ids = self.tokenizer.encode(trace_text, add_bos=True, add_eos=True)
                self.pager.ingest_tokens(self.model.backbone, full_ids, source_tag=f"turn_{len(self.history)+1}")

            # Inline verification checks (#T$# arithmetic equations + optional Python assertion harness)
            eq_consistency = _verify_thought_equations(segments["thought"]) if segments["thought"] else 1.0
            harness_score = (
                verify_python_rollout(trace_text, harness_h)
                if harness_h
                else None
            )

            expert_routing = _collect_layer_expert_loads(self.model)
            param_stats = _compute_active_parameter_stats(self.model)

            turn_record = {
                "turn_id": len(self.history) + 1,
                "timestamp": time.time(),
                "prompt": prompt,
                "thought_prefix": thought_prefix,
                "answer_prefix": answer_prefix,
                "lock_thought": lock_thought,
                "trace": trace_text,
                "segments": segments,
                "tokens": {
                    "prompt": len(prompt_toks),
                    "thought": len(thought_toks),
                    "answer": len(answer_toks),
                    "generated": gen_tok_count,
                },
                "latency_ms": round(elapsed_s * 1000.0, 1),
                "tok_per_s": tok_per_s,
                "config_snapshot": {
                    "mode": mode,
                    "recurrence_steps": rec_steps,
                    "top_k": int(cfg["top_k"]),
                    "dense_subset_k": int(cfg["dense_subset_k"]),
                    "combine_dense_and_sparse": bool(cfg["combine_dense_and_sparse"]),
                    "recurrent_coverage_penalty": float(cfg["recurrent_coverage_penalty"]),
                    "temperature": temp,
                    "use_whiteboard": use_wb,
                    "active_parameters": param_stats["active_parameters"],
                    "active_percent": param_stats["active_percent"],
                },
                "verification": {
                    "thought_equation_score": eq_consistency,
                    "harness": harness_h or None,
                    "harness_score": harness_score,
                },
                "mode_meta": mode_meta,
                "candidates": candidates_telemetry,
                "expert_routing": expert_routing,
            }
            self.history.append(turn_record)
            return {
                "turn": turn_record,
                "status": self.status(),
            }

    @torch.no_grad()
    def probe_decision(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run an interactive Typed-COE Decision + Deduction + Evaluation + Metacognition probe."""
        with self._lock:
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("No checkpoint loaded.")
            context_str = str(payload.get("context") or "").strip()
            question_str = str(payload.get("question") or "Select the best option.").strip()
            options_raw = payload.get("options") or []
            if not isinstance(options_raw, list) or len(options_raw) < 2:
                raise ValueError("Provide at least 2 candidate options.")

            clean_opts = [str(o).strip() for o in options_raw if str(o).strip()]
            if len(clean_opts) < 2:
                raise ValueError("Provide at least 2 non-empty candidate options.")

            device = next(self.model.parameters()).device
            max_seq = int(self.model.backbone.config.max_seq_len)
            ctx_full = (
                context_str
                if context_str.startswith("#U$#")
                else f"#U$# {context_str}\n#Q$# [choice] {question_str}"
            )
            descs = tuple(
                (o if o.startswith(("#A$#", "#T$#")) else f"#A$# {o}")
                for o in clean_opts
            )
            ex = DecisionExample(
                question_id="interactive_probe",
                kind="binary" if len(clean_opts) == 2 else "choice",
                context=ctx_full,
                descriptions=descs,
                values=tuple(f"opt_{i}" for i in range(len(clean_opts))),
                target=0,
            )
            batch = collate_decisions([ex], max_seq, device=device, tokenizer=self.tokenizer)
            ctx_ids = batch["context_ids"]
            ctx_lens = batch["context_lengths"]
            opt_ids = batch["option_ids"]
            opt_lens = batch["option_lengths"]

            rec_steps = int(self.inference_config.get("recurrence_steps", 2))
            use_wb = bool(self.inference_config.get("use_whiteboard", False)) and self.wb_controller is not None
            use_amp = device.type == "cuda"
            k_opts = len(clean_opts)
            rep_ctx_ids = ctx_ids.expand(k_opts, -1).contiguous()
            rep_ctx_lens = ctx_lens.expand(k_opts).contiguous()
            flat_opt_ids = opt_ids[0]
            flat_opt_lens = opt_lens[0]

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                if use_wb and self.wb_controller.whiteboard.episodic_rule_count > 0:
                    dec_out, _ = self.wb_controller.solve_decision_on_whiteboard(
                        ctx_ids,
                        ctx_lens,
                        opt_ids,
                        opt_lens,
                        recurrence_steps=rec_steps,
                        adapt_on_context=False,
                    )
                else:
                    dec_out = self.model.forward_decision(
                        ctx_ids,
                        ctx_lens,
                        opt_ids,
                        opt_lens,
                        recurrence_steps=rec_steps,
                        depth_ponder=bool(self.inference_config.get("depth_ponder", True)),
                    )

                ded_out = self.model.forward_deduction(rep_ctx_ids, rep_ctx_lens, flat_opt_ids, flat_opt_lens)
                eval_out = self.model.forward_evaluation(rep_ctx_ids, rep_ctx_lens, flat_opt_ids, flat_opt_lens)
                meta_out = self.model.forward_metacognition(rep_ctx_ids, rep_ctx_lens, flat_opt_ids, flat_opt_lens)

            probs = dec_out.probabilities[0].detach().float().cpu().tolist()
            sims = dec_out.similarities[0].detach().float().cpu().tolist()
            ded_p = ded_out.probabilities.detach().float().cpu().tolist()
            eval_u = eval_out.utilities.detach().float().cpu().tolist()
            meta_p = meta_out.probabilities.detach().float().cpu().tolist()
            winner_idx = int(dec_out.logits[0].argmax(dim=-1).item())

            rows = []
            for i, text_o in enumerate(clean_opts):
                rows.append(
                    {
                        "index": i,
                        "option": text_o,
                        "selected": i == winner_idx,
                        "probability": round(float(probs[i]), 4),
                        "pearson_similarity": round(float(sims[i]), 4),
                        "utility": round(float(eval_u[i]), 4),
                        "entailed": round(float(ded_p[i][0]), 4),
                        "contradicted": round(float(ded_p[i][1]), 4),
                        "correctness": round(float(meta_p[i][0]), 4),
                        "sufficiency": round(float(meta_p[i][1]), 4),
                    }
                )
            return {
                "winner_index": winner_idx,
                "winner_option": clean_opts[winner_idx],
                "used_whiteboard": use_wb,
                "candidates": rows,
            }

    def whiteboard_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Perform a Quarantined Whiteboard action (`ingest`, `reset`, `consolidate`, `audit`)."""
        with self._lock:
            if self.model is None or self.wb_controller is None or self.tokenizer is None:
                raise RuntimeError("No checkpoint loaded.")
            action = str(payload.get("action") or "ingest").lower()

            if action == "reset":
                self.wb_controller.whiteboard.reset_fast_weights()
                return {
                    "action": "reset",
                    "status": self.status(),
                }

            if action == "consolidate":
                rate = float(payload.get("consolidation_rate", 0.25))
                res = self.wb_controller.consolidate_whiteboard_to_mitotic_children(
                    consolidation_rate=rate,
                    reset_whiteboard_after=bool(payload.get("reset_after", False)),
                )
                return {
                    "action": "consolidate",
                    "consolidation": res,
                    "status": self.status(),
                }

            if action == "audit":
                doc_a = str(payload.get("doc_a") or "").strip()
                doc_b = str(payload.get("doc_b") or "").strip()
                mod_a = str(payload.get("modality_a") or "prose")
                mod_b = str(payload.get("modality_b") or "prose")
                conflicts = self.wb_controller.audit_contradictions(
                    doc_a,
                    doc_b,
                    tokenizer=self.tokenizer,
                    modality_a=mod_a,
                    modality_b=mod_b,
                )
                return {
                    "action": "audit",
                    "conflicts": conflicts,
                    "status": self.status(),
                }

            # Default: ingest artifact (prose, code, or web)
            raw_text = str(payload.get("text") or "").strip()
            if not raw_text:
                raise ValueError("Provide non-empty artifact text/code to ingest into the Whiteboard.")
            modality = str(payload.get("modality") or "prose").lower()
            reset_wb = bool(payload.get("reset_whiteboard", False))
            dev = next(self.model.parameters()).device
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                rep, props = self.wb_controller.ingest_artifact(
                    raw_text,
                    tokenizer=self.tokenizer,
                    modality=modality,
                    pager=self.pager,
                    reset_whiteboard=reset_wb,
                )
            self.inference_config["use_whiteboard"] = True
            prop_dicts = [
                (
                    {"premise": p[0], "conclusion": p[1], "source": p[2] if len(p) > 2 else ""}
                    if isinstance(p, (tuple, list))
                    else (p if isinstance(p, dict) else asdict(p))
                )
                for p in props
            ]
            return {
                "action": "ingest",
                "modality": modality,
                "report": asdict(rep),
                "propositions": prop_dicts,
                "status": self.status(),
            }

    def clear_session(self) -> dict[str, Any]:
        with self._lock:
            self.history.clear()
            if self.pager is not None:
                self.pager = InfiniteContextPager(page_size=64, landmarks_per_page=4, top_k_pages=3)
                self.ledger = self.pager.ledger
            return self.status()


OPENCODE_INFERENCE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>opencode // mcl4-coe inference terminal</title>
<style>
  :root {
    --bg: #09090b;
    --bg-elev: #101014;
    --bg-panel: #141419;
    --bg-hover: #1c1c24;
    --border: #23232b;
    --border-bright: #343442;
    --text: #f4f4f6;
    --muted: #9ca3af;
    --dim: #646775;
    --amber: #f59e0b;
    --emerald: #10b981;
    --cyan: #38bdf8;
    --violet: #a78bfa;
    --rose: #f43f5e;
    --mono: 'JetBrains Mono', 'Geist Mono', 'Fira Code', 'Cascadia Code', Consolas, ui-monospace, monospace;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0;
    padding: 0;
    height: 100vh;
    overflow: hidden;
    background: var(--bg);
    color: var(--text);
    font-family: var(--mono);
    font-size: 12.5px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }
  /* Top OpenCode Status Header */
  .topbar {
    height: 44px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 14px;
    background: var(--bg-elev);
    border-bottom: 1px solid var(--border);
    user-select: none;
  }
  .topbar-left, .topbar-right {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .brand-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 3px 9px;
    background: #18181f;
    border: 1px solid var(--border-bright);
    border-radius: 4px;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.3px;
    color: var(--text);
  }
  .brand-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--dim);
  }
  .brand-dot.loaded {
    background: var(--emerald);
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.65);
  }
  .ckpt-trigger {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 4px 10px;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text);
    cursor: pointer;
    font-family: var(--mono);
    font-size: 11.5px;
  }
  .ckpt-trigger:hover {
    border-color: var(--cyan);
    background: var(--bg-hover);
  }
  .meta-tag {
    padding: 2px 7px;
    border-radius: 3px;
    background: #16161d;
    border: 1px solid var(--border);
    color: var(--muted);
    font-size: 11px;
  }
  .meta-tag strong { color: var(--text); }
  .meta-tag.accent-cyan { border-color: rgba(56, 189, 248, 0.35); color: var(--cyan); }
  .meta-tag.accent-amber { border-color: rgba(245, 158, 11, 0.35); color: var(--amber); }
  .meta-tag.accent-violet { border-color: rgba(167, 139, 250, 0.35); color: var(--violet); }

  /* Mode Switcher Segmented Control */
  .mode-group {
    display: inline-flex;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
  }
  .mode-btn {
    background: transparent;
    border: none;
    color: var(--muted);
    padding: 4px 9px;
    font-family: var(--mono);
    font-size: 11px;
    cursor: pointer;
    border-right: 1px solid var(--border);
  }
  .mode-btn:last-child { border-right: none; }
  .mode-btn.active {
    background: rgba(56, 189, 248, 0.14);
    color: var(--cyan);
    font-weight: 600;
  }

  /* Main Split Workspace */
  .workspace {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 370px;
    height: calc(100vh - 44px - 26px);
  }
  .workspace.inspector-collapsed {
    grid-template-columns: minmax(0, 1fr);
  }
  .workspace.inspector-collapsed .inspector {
    display: none;
  }

  /* Center Terminal Session Pane */
  .session-pane {
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;
    background: var(--bg);
  }
  .transcript {
    flex: 1;
    overflow-y: auto;
    padding: 18px 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .empty-state {
    margin: auto;
    max-width: 600px;
    padding: 24px;
    border: 1px dashed var(--border-bright);
    border-radius: 6px;
    background: rgba(20, 20, 25, 0.55);
    color: var(--muted);
  }
  .empty-state h2 {
    margin: 0 0 8px 0;
    font-size: 14px;
    color: var(--text);
  }
  .shortcut-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 6px 14px;
    margin-top: 12px;
    font-size: 11.5px;
  }
  .kbd {
    display: inline-block;
    padding: 1px 5px;
    background: #1c1c24;
    border: 1px solid var(--border-bright);
    border-radius: 3px;
    color: var(--cyan);
    font-size: 11px;
  }

  /* Turn Card (OpenCode Block Style) */
  .turn-block {
    border: 1px solid var(--border);
    border-radius: 5px;
    background: var(--bg-elev);
    overflow: hidden;
  }
  .turn-user {
    padding: 9px 13px;
    background: #13131a;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
  }
  .turn-user-text {
    color: var(--text);
    font-weight: 600;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .turn-user-prefix {
    color: var(--amber);
    margin-right: 6px;
    font-weight: 700;
  }
  .turn-badges {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    flex-shrink: 0;
    font-size: 10.5px;
    color: var(--muted);
  }
  .pill {
    padding: 1px 6px;
    border-radius: 3px;
    background: var(--bg-panel);
    border: 1px solid var(--border);
  }
  .pill.ok { color: var(--emerald); border-color: rgba(16, 185, 129, 0.35); }
  .pill.warn { color: var(--amber); border-color: rgba(245, 158, 11, 0.35); }
  .pill.bad { color: var(--rose); border-color: rgba(244, 63, 94, 0.35); }

  /* Collapsible #T$# Thinking Trace */
  .thought-details {
    border-bottom: 1px solid var(--border);
    background: #0c0c10;
  }
  .thought-summary {
    padding: 6px 13px;
    cursor: pointer;
    color: var(--muted);
    font-size: 11.5px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    user-select: none;
    list-style: none;
  }
  .thought-summary::-webkit-details-marker { display: none; }
  .thought-summary:hover { background: #121218; color: var(--text); }
  .thought-tag {
    color: var(--violet);
    font-weight: 700;
    margin-right: 6px;
  }
  .thought-body {
    padding: 8px 14px 11px 24px;
    border-top: 1px dashed #1d1d26;
    color: #a5b4fc;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 12px;
  }

  /* #A$# Answer Output */
  .turn-answer {
    padding: 12px 14px;
    color: var(--text);
    white-space: pre-wrap;
    word-break: break-word;
    position: relative;
  }
  .answer-tag {
    color: var(--emerald);
    font-weight: 700;
    margin-right: 6px;
  }

  /* Turn Footer: Candidates & Routing Drawer */
  .turn-footer {
    padding: 5px 13px;
    background: #0d0d12;
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    font-size: 11px;
    color: var(--dim);
  }
  .cand-details {
    border-top: 1px solid var(--border);
    background: #0b0b0f;
  }
  .cand-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
  }
  .cand-table th, .cand-table td {
    padding: 5px 8px;
    border-bottom: 1px solid #1b1b22;
    text-align: left;
  }
  .cand-table th { color: var(--muted); font-weight: 600; }
  .cand-row-win { background: rgba(16, 185, 129, 0.08); }

  /* Prompt Composer at Bottom */
  .composer-wrap {
    border-top: 1px solid var(--border);
    background: var(--bg-elev);
    padding: 10px 16px 12px 16px;
    position: relative;
  }
  .prefix-drawer {
    display: none;
    grid-template-columns: 1fr 1fr auto;
    gap: 8px;
    margin-bottom: 8px;
    padding-bottom: 8px;
    border-bottom: 1px dashed var(--border);
  }
  .prefix-drawer.open { display: grid; }
  .composer-box {
    display: flex;
    align-items: flex-end;
    gap: 8px;
    background: var(--bg);
    border: 1px solid var(--border-bright);
    border-radius: 5px;
    padding: 7px 10px;
  }
  .composer-box:focus-within {
    border-color: var(--cyan);
  }
  .prompt-Sign {
    color: var(--cyan);
    font-weight: 700;
    padding-bottom: 3px;
    user-select: none;
  }
  #promptInput {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--text);
    font-family: var(--mono);
    font-size: 12.5px;
    resize: none;
    outline: none;
    max-height: 150px;
    min-height: 22px;
    line-height: 1.45;
  }
  .composer-actions {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .btn {
    background: var(--cyan);
    color: #051017;
    border: none;
    border-radius: 4px;
    padding: 5px 11px;
    font-family: var(--mono);
    font-size: 11.5px;
    font-weight: 700;
    cursor: pointer;
  }
  .btn:hover { opacity: 0.92; }
  .btn:disabled { opacity: 0.45; cursor: not-allowed; }
  .btn-ghost {
    background: var(--bg-panel);
    color: var(--muted);
    border: 1px solid var(--border);
    font-weight: 500;
  }
  .btn-ghost:hover { color: var(--text); border-color: var(--border-bright); }
  .btn-amber { background: var(--amber); color: #140c00; }
  .btn-emerald { background: var(--emerald); color: #02140d; }

  /* Slash Command Autocomplete Menu */
  .slash-menu {
    display: none;
    position: absolute;
    bottom: calc(100% + 4px);
    left: 16px;
    right: 16px;
    max-width: 560px;
    background: var(--bg-panel);
    border: 1px solid var(--border-bright);
    border-radius: 5px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.65);
    z-index: 40;
    overflow: hidden;
  }
  .slash-menu.open { display: block; }
  .slash-item {
    padding: 6px 11px;
    display: flex;
    justify-content: space-between;
    font-size: 11.5px;
    cursor: pointer;
    border-bottom: 1px solid var(--border);
  }
  .slash-item:last-child { border-bottom: none; }
  .slash-item:hover, .slash-item.selected {
    background: rgba(56, 189, 248, 0.14);
  }
  .slash-cmd { color: var(--cyan); font-weight: 700; }
  .slash-desc { color: var(--muted); }

  /* Right Inspector Sidebar */
  .inspector {
    background: var(--bg-elev);
    border-left: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;
  }
  .inspector-tabs {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border-bottom: 1px solid var(--border);
    background: #0d0d11;
  }
  .insp-tab {
    padding: 8px 4px;
    text-align: center;
    font-size: 11px;
    color: var(--muted);
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    font-family: var(--mono);
    cursor: pointer;
  }
  .insp-tab.active {
    color: var(--cyan);
    border-bottom-color: var(--cyan);
    background: var(--bg-elev);
    font-weight: 600;
  }
  .inspector-body {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .insp-section {
    border: 1px solid var(--border);
    border-radius: 5px;
    background: var(--bg-panel);
    padding: 10px;
  }
  .insp-title {
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--muted);
    margin: 0 0 8px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .field-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: 3px;
    font-size: 11px;
    color: var(--muted);
  }
  .field.full { grid-column: span 2; }
  input[type="text"], input[type="password"], input[type="number"], select, textarea {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text);
    padding: 5px 7px;
    font-family: var(--mono);
    font-size: 11.5px;
    outline: none;
    width: 100%;
  }
  input:focus, select:focus, textarea:focus {
    border-color: var(--cyan);
  }
  .check-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 4px 0;
    font-size: 11px;
    color: var(--text);
  }
  input[type="checkbox"] {
    accent-color: var(--cyan);
    cursor: pointer;
  }
  .stat-kv {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    padding: 2.5px 0;
    border-bottom: 1px dashed #1e1e26;
  }
  .stat-kv:last-child { border-bottom: none; }
  .stat-k { color: var(--muted); }
  .stat-v { color: var(--text); font-weight: 600; }

  /* Expert Heatmap Strip */
  .heatmap-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 4px;
    font-size: 10.5px;
  }
  .heatmap-label {
    width: 36px;
    color: var(--muted);
    flex-shrink: 0;
  }
  .heatmap-cells {
    flex: 1;
    display: flex;
    gap: 1px;
    height: 12px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 2px;
    overflow: hidden;
  }
  .heatmap-cell {
    flex: 1;
    height: 100%;
    background: rgba(56, 189, 248, 0.05);
  }

  /* Bottom OpenCode Status Bar */
  .statusbar {
    height: 26px;
    background: #0c0c10;
    border-top: 1px solid var(--border);
    padding: 0 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 11px;
    color: var(--muted);
    user-select: none;
  }
  .statusbar-left, .statusbar-right {
    display: flex;
    align-items: center;
    gap: 14px;
  }
</style>
</head>
<body>

<!-- Top OpenCode Header -->
<div class="topbar">
  <div class="topbar-left">
    <div class="brand-pill">
      <span id="statusDot" class="brand-dot"></span>
      <span>opencode // mcl4-coe</span>
    </div>
    <button class="ckpt-trigger" onclick="switchInspectorTab('weights')" title="Switch or Load Checkpoint (Alt+K)">
      <span id="activeCkptLabel">[no checkpoint loaded]</span>
      <span style="color:var(--dim)">▾</span>
    </button>
    <span class="meta-tag accent-cyan" id="activeParamBadge">active: -- / --</span>
    <span class="meta-tag accent-amber" id="moeBadge">E=-- · k=--</span>
    <span class="meta-tag accent-violet" id="wbBadge" style="display:none">WB: 0 rules</span>
  </div>

  <div class="topbar-right">
    <div class="mode-group" title="Generation Engine Mode">
      <button class="mode-btn active" data-mode="uta" onclick="setMode('uta')">Direct UTA</button>
      <button class="mode-btn" data-mode="coe" onclick="setMode('coe')">COE 9-Op Loop</button>
      <button class="mode-btn" data-mode="dream_rl" onclick="setMode('dream_rl')">2-Round Dream-RL</button>
    </div>
    <button class="btn btn-ghost" onclick="clearSession()">Clear</button>
    <button class="btn btn-ghost" onclick="toggleInspector()" title="Toggle Inspector Sidebar (Alt+I)">Inspector</button>
  </div>
</div>

<!-- Main Workspace -->
<div class="workspace" id="workspace">
  <!-- Center Transcript & Composer -->
  <div class="session-pane">
    <div class="transcript" id="transcript">
      <div class="empty-state" id="emptyState">
        <h2>mcl4-coe interactive inference & testing terminal</h2>
        <div>Load a checkpoint from the right-hand <strong>Weights</strong> panel (local <code>.pt</code>, Hugging Face <code>mesmac/mcl4-coe-runs</code>, or instant preset) and send a prompt or slash command below.</div>
        <div class="shortcut-grid">
          <div><span class="kbd">Enter</span> Send prompt</div>
          <div><span class="kbd">Shift+Enter</span> New line</div>
          <div><span class="kbd">/topk 16</span> Set active MoE top-k</div>
          <div><span class="kbd">/dense 32</span> Set multi-active coalition</div>
          <div><span class="kbd">/tabu 0.4</span> Set Tabu coverage penalty</div>
          <div><span class="kbd">/rec 3</span> Set recurrence steps</div>
          <div><span class="kbd">/mode coe</span> Switch to 9-Op COE loop</div>
          <div><span class="kbd">/ingest prose ...</span> Add to Whiteboard</div>
        </div>
      </div>
    </div>

    <!-- Bottom Prompt Composer -->
    <div class="composer-wrap">
      <div class="slash-menu" id="slashMenu"></div>

      <!-- Optional #T$# / #A$# Prefix Bar -->
      <div class="prefix-drawer" id="prefixDrawer">
        <div class="field">
          <label>#T$# Thought Prefix (optional)</label>
          <input type="text" id="thoughtPrefixInput" placeholder="Seed initial #T$# reasoning premise..."/>
        </div>
        <div class="field">
          <label>#A$# Answer Prefix / Harness</label>
          <input type="text" id="answerPrefixInput" placeholder="Optional #A$# prefix (e.g. def solve(n):)"/>
        </div>
        <div class="field" style="justify-content:flex-end;">
          <label style="display:flex;align-items:center;gap:5px;cursor:pointer;padding-bottom:5px;">
            <input type="checkbox" id="lockThoughtInput"/> Lock #T$#
          </label>
        </div>
      </div>

      <div class="composer-box">
        <span class="prompt-Sign">#U$# &gt;</span>
        <textarea id="promptInput" rows="1" placeholder="Ask a question, test reasoning, or type / for commands..."></textarea>
        <div class="composer-actions">
          <button class="btn btn-ghost" onclick="togglePrefixDrawer()" title="Seed #T$# or #A$# prefixes">#T$#/#A$#</button>
          <button class="btn" id="sendBtn" onclick="sendPrompt()">Run ↵</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Right Inspector Sidebar -->
  <aside class="inspector">
    <div class="inspector-tabs">
      <button class="insp-tab active" data-tab="knobs" onclick="switchInspectorTab('knobs')">Knobs</button>
      <button class="insp-tab" data-tab="weights" onclick="switchInspectorTab('weights')">Weights</button>
      <button class="insp-tab" data-tab="whiteboard" onclick="switchInspectorTab('whiteboard')">Whiteboard</button>
      <button class="insp-tab" data-tab="probe" onclick="switchInspectorTab('probe')">Probe</button>
    </div>

    <!-- TAB 1: INFERENCE & MOE KNOBS -->
    <div class="inspector-body" id="tab-knobs">
      <div class="insp-section">
        <div class="insp-title">
          <span>MoE Sparsity & Multi-Active Density</span>
          <span id="liveDensityRatio" style="color:var(--cyan)">--</span>
        </div>
        <div class="field-grid">
          <div class="field">
            <label>Active top_k (Sparse)</label>
            <input type="number" id="cfgTopK" min="1" max="944" value="4" onchange="pushKnobs()"/>
          </div>
          <div class="field">
            <label>dense_subset_k (Coalition)</label>
            <input type="number" id="cfgDenseK" min="0" max="944" value="0" onchange="pushKnobs()"/>
          </div>
          <div class="field full">
            <label>Recurrent Tabu Coverage Penalty (0.0 - 1.5)</label>
            <input type="number" id="cfgTabu" min="0" max="2.0" step="0.05" value="0.35" onchange="pushKnobs()"/>
          </div>
        </div>
        <div class="check-row">
          <span>Combine Dense Coalition + Sparse Top-K</span>
          <input type="checkbox" id="cfgCombineDenseSparse" onchange="pushKnobs()"/>
        </div>
        <div class="check-row">
          <span>Release Mitotic Twin Competition Pairs</span>
          <input type="checkbox" id="cfgReleasePairs" onchange="pushKnobs()"/>
        </div>
      </div>

      <div class="insp-section">
        <div class="insp-title">
          <span>Recurrence & Decoding</span>
        </div>
        <div class="field-grid">
          <div class="field">
            <label>recurrence_steps</label>
            <input type="number" id="cfgRecSteps" min="1" max="8" value="2" onchange="pushKnobs()"/>
          </div>
          <div class="field">
            <label>max_new_tokens</label>
            <input type="number" id="cfgMaxTokens" min="16" max="1024" step="16" value="256" onchange="pushKnobs()"/>
          </div>
          <div class="field">
            <label>temperature</label>
            <input type="number" id="cfgTemp" min="0" max="1.5" step="0.05" value="0.0" onchange="pushKnobs()"/>
          </div>
          <div class="field">
            <label>entropy_threshold</label>
            <input type="number" id="cfgEntThresh" min="0.2" max="5.0" step="0.1" value="1.5" onchange="pushKnobs()"/>
          </div>
        </div>
        <div class="check-row">
          <span>Depth Ponder (Residual Halting Gate)</span>
          <input type="checkbox" id="cfgDepthPonder" checked onchange="pushKnobs()"/>
        </div>
        <div class="check-row">
          <span>Recurrent Cross-Pass KV Memory</span>
          <input type="checkbox" id="cfgRecKV" checked onchange="pushKnobs()"/>
        </div>
        <div class="check-row">
          <span>Adaptive Recurrent Depth Escalation</span>
          <input type="checkbox" id="cfgAdaptiveRec" onchange="pushKnobs()"/>
        </div>
        <div class="check-row">
          <span>Steer with Quarantined Whiteboard</span>
          <input type="checkbox" id="cfgUseWhiteboard" onchange="pushKnobs()"/>
        </div>
        <div class="check-row">
          <span>Page into Windowless Context Ledger</span>
          <input type="checkbox" id="cfgUsePager" onchange="pushKnobs()"/>
        </div>
      </div>

      <div class="insp-section">
        <div class="insp-title">
          <span>Last Turn MoE Routing Heatmap</span>
          <span id="heatmapMeta" style="color:var(--dim)">--</span>
        </div>
        <div id="expertHeatmapBox">
          <div style="color:var(--dim);font-size:11px;">Run a turn to inspect per-layer specialist routing.</div>
        </div>
      </div>
    </div>

    <!-- TAB 2: CHECKPOINT & WEIGHTS LOADER -->
    <div class="inspector-body" id="tab-weights" style="display:none">
      <div class="insp-section">
        <div class="insp-title">
          <span>Discovered Checkpoints (Local & HF)</span>
          <button class="btn btn-ghost" style="padding:2px 6px;font-size:10px;" onclick="refreshCheckpoints()">Refresh</button>
        </div>
        <div class="field full" style="margin-bottom:8px;">
          <label>Select Checkpoint Receipt / Weights</label>
          <select id="ckptSelect"></select>
        </div>
        <div class="field full" style="margin-bottom:8px;">
          <label>Or Custom Path / HF Checkpoint ID</label>
          <input type="text" id="customCkptPath" placeholder="e.g. colab_mcl4_coe_v7_mitotic_3b_fast/step_05800"/>
        </div>
        <div class="field-grid" style="margin-bottom:8px;">
          <div class="field">
            <label>Device</label>
            <select id="loadDevice">
              <option value="auto">auto (cuda/cpu)</option>
              <option value="cuda">cuda</option>
              <option value="cpu">cpu</option>
            </select>
          </div>
          <div class="field">
            <label>Precision</label>
            <select id="loadDtype">
              <option value="bfloat16">bfloat16</option>
              <option value="float16">float16</option>
              <option value="float32">float32</option>
            </select>
          </div>
          <div class="field">
            <label>HF Hub Repo</label>
            <input type="text" id="hfRepoInput" value="mesmac/mcl4-coe-runs"/>
          </div>
          <div class="field">
            <label>HF_TOKEN (if private)</label>
            <input type="password" id="hfTokenInput" placeholder="hf_..."/>
          </div>
        </div>
        <button class="btn" style="width:100%" id="loadCkptBtn" onclick="loadSelectedCheckpoint()">Load Checkpoint Weights</button>
      </div>

      <div class="insp-section">
        <div class="insp-title">
          <span>Instant Architecture Presets (No Download)</span>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;">
          <button class="btn btn-ghost" style="flex:1" onclick="loadPreset('tiny_morph16k')">Tiny Morph16K</button>
          <button class="btn btn-ghost" style="flex:1" onclick="loadPreset('v6_mitotic_e8')">v6-Mitotic (E=8)</button>
          <button class="btn btn-ghost" style="flex:1" onclick="loadPreset('v7_balanced_768_e8')">v7-768 (E=8)</button>
        </div>
      </div>

      <div class="insp-section">
        <div class="insp-title">
          <span>Loaded Model Telemetry</span>
        </div>
        <div id="archStatsBox">
          <div style="color:var(--dim);font-size:11px;">No model currently loaded.</div>
        </div>
      </div>
    </div>

    <!-- TAB 3: QUARANTINED WHITEBOARD ("EINSTEIN ON A WHITEBOARD") -->
    <div class="inspector-body" id="tab-whiteboard" style="display:none">
      <div class="insp-section">
        <div class="insp-title">
          <span>Ingest Artifact into Fast Weights</span>
          <span id="wbRuleCountPill" style="color:var(--violet)">0 rules</span>
        </div>
        <div class="field-grid" style="margin-bottom:8px;">
          <div class="field">
            <label>Modality</label>
            <select id="wbModality">
              <option value="prose">prose (Paper / Spec)</option>
              <option value="code">code (Python AST)</option>
              <option value="web">web (HTML Table)</option>
            </select>
          </div>
          <div class="field" style="justify-content:flex-end;">
            <label style="display:flex;align-items:center;gap:5px;cursor:pointer;padding-bottom:4px;">
              <input type="checkbox" id="wbResetBefore"/> Reset before ingest
            </label>
          </div>
        </div>
        <div class="field full" style="margin-bottom:8px;">
          <textarea id="wbIngestText" rows="5" placeholder="Paste domain rules, code class/functions, or HTML table to absorb into QuarantinedWhiteboardState..."></textarea>
        </div>
        <div style="display:flex;gap:6px;">
          <button class="btn btn-emerald" style="flex:2" onclick="ingestWhiteboard()">Absorb into Whiteboard</button>
          <button class="btn btn-ghost" style="flex:1" onclick="consolidateWhiteboard()" title="Consolidate fast weights into mitotic child experts">Sleep Consolidate</button>
          <button class="btn btn-ghost" onclick="resetWhiteboard()" title="Zero out fast weights">Reset</button>
        </div>
        <div id="wbLastReport" style="margin-top:8px;font-size:11px;color:var(--muted);"></div>
      </div>

      <div class="insp-section">
        <div class="insp-title">
          <span>Hopfield-Kanerva Episodic Rule Ledger</span>
        </div>
        <div id="wbRulesList" style="display:flex;flex-direction:column;gap:6px;max-height:260px;overflow-y:auto;">
          <div style="color:var(--dim);font-size:11px;">No episodic rules absorbed yet.</div>
        </div>
      </div>
    </div>

    <!-- TAB 4: TYPED-COE DECISION & HARNESS PROBE -->
    <div class="inspector-body" id="tab-probe" style="display:none">
      <div class="insp-section">
        <div class="insp-title">
          <span>Typed-COE Multi-Option Decision Probe</span>
        </div>
        <div class="field full" style="margin-bottom:6px;">
          <label>#U$# Context / Premise</label>
          <textarea id="probeContext" rows="2" placeholder="State or problem context..."></textarea>
        </div>
        <div class="field full" style="margin-bottom:6px;">
          <label>#Q$# Question</label>
          <input type="text" id="probeQuestion" value="Select the most accurate and logically entailed option."/>
        </div>
        <div class="field full" style="margin-bottom:8px;">
          <label>Candidate Options (one per line)</label>
          <textarea id="probeOptions" rows="4" placeholder="Option A&#10;Option B&#10;Option C"></textarea>
        </div>
        <button class="btn btn-amber" style="width:100%" onclick="runDecisionProbe()">Score Candidates via COE</button>
      </div>

      <div class="insp-section">
        <div class="insp-title">
          <span>Operator Probe Breakdown</span>
        </div>
        <div id="probeResultBox">
          <div style="color:var(--dim);font-size:11px;">Run a probe above to inspect Decision, Deduction, Evaluation, and Metacognition heads.</div>
        </div>
      </div>
    </div>
  </aside>
</div>

<!-- Bottom OpenCode Status Bar -->
<div class="statusbar">
  <div class="statusbar-left">
    <span id="sbStatus">READY</span>
    <span id="sbDevice">GPU: --</span>
    <span id="sbVram">VRAM: -- MB</span>
    <span id="sbPager">Pager: 0 pages</span>
  </div>
  <div class="statusbar-right">
    <span><span class="kbd">/</span> commands</span>
    <span><span class="kbd">Alt+K</span> weights</span>
    <span><span class="kbd">Alt+I</span> inspector</span>
  </div>
</div>

<script>
const SLASH_COMMANDS = [
  { cmd: "/topk ", desc: "Set sparse active experts per token (e.g. /topk 16)" },
  { cmd: "/dense ", desc: "Set multi-active dense coalition size (e.g. /dense 32)" },
  { cmd: "/tabu ", desc: "Set recurrent Tabu coverage penalty (e.g. /tabu 0.45)" },
  { cmd: "/rec ", desc: "Set recurrence depth steps (e.g. /rec 3)" },
  { cmd: "/temp ", desc: "Set sampling temperature (e.g. /temp 0.25)" },
  { cmd: "/mode ", desc: "Switch generation mode: uta | coe | dream_rl" },
  { cmd: "/ingest prose ", desc: "Ingest prose rule into Quarantined Whiteboard" },
  { cmd: "/ingest code ", desc: "Ingest Python code into Quarantined Whiteboard" },
  { cmd: "/wbreset", desc: "Reset Quarantined Whiteboard fast weights to zero" },
  { cmd: "/clear", desc: "Clear conversation transcript & context pager" }
];

let appStatus = null;
let discoveredCheckpoints = [];

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts
  });
  const data = await res.json();
  if (!res.ok || data.error) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return data;
}

function fmtCount(n) {
  if (!n || n <= 0) return "--";
  if (n >= 1e9) return (n / 1e9).toFixed(3) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(n);
}

function escHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function setStatusText(msg) {
  document.getElementById("sbStatus").textContent = msg;
}

function switchInspectorTab(tabName) {
  document.getElementById("workspace").classList.remove("inspector-collapsed");
  document.querySelectorAll(".insp-tab").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.tab === tabName);
  });
  ["knobs", "weights", "whiteboard", "probe"].forEach(t => {
    const el = document.getElementById("tab-" + t);
    if (el) el.style.display = (t === tabName) ? "flex" : "none";
  });
}

function toggleInspector() {
  document.getElementById("workspace").classList.toggle("inspector-collapsed");
}

function togglePrefixDrawer() {
  document.getElementById("prefixDrawer").classList.toggle("open");
}

function renderStatus(st) {
  appStatus = st;
  const info = st.loaded_info || {};
  const arch = st.architecture || {};
  const pstats = st.parameter_stats || {};
  const cfg = st.inference_config || {};
  const wb = st.whiteboard || {};
  const pager = st.pager || {};

  const dot = document.getElementById("statusDot");
  dot.classList.toggle("loaded", Boolean(info.loaded));

  document.getElementById("activeCkptLabel").textContent = info.loaded
    ? `${info.checkpoint_id} (${info.device}/${info.dtype})`
    : "[no checkpoint loaded — click to load]";

  if (info.loaded && pstats.total_parameters) {
    document.getElementById("activeParamBadge").textContent =
      `active: ${fmtCount(pstats.active_parameters)} / ${fmtCount(pstats.total_parameters)} (${pstats.active_percent}%)`;
    document.getElementById("moeBadge").textContent =
      `E=${arch.num_experts} · k=${pstats.active_experts_per_token} · r=${cfg.recurrence_steps}`;
    document.getElementById("liveDensityRatio").textContent =
      `${fmtCount(pstats.active_parameters)} (${pstats.active_percent}%)`;
  } else {
    document.getElementById("activeParamBadge").textContent = "active: -- / --";
    document.getElementById("moeBadge").textContent = "E=-- · k=--";
    document.getElementById("liveDensityRatio").textContent = "--";
  }

  const wbBadge = document.getElementById("wbBadge");
  if (wb.episodic_rule_count > 0 || wb.active) {
    wbBadge.style.display = "inline-block";
    wbBadge.textContent = `WB: ${wb.episodic_rule_count} rules ${wb.active ? "(ON)" : "(OFF)"}`;
  } else {
    wbBadge.style.display = "none";
  }

  document.querySelectorAll(".mode-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.mode === cfg.mode);
  });

  // Sync knob inputs
  document.getElementById("cfgTopK").value = cfg.top_k ?? 4;
  document.getElementById("cfgDenseK").value = cfg.dense_subset_k ?? 0;
  document.getElementById("cfgTabu").value = cfg.recurrent_coverage_penalty ?? 0.35;
  document.getElementById("cfgCombineDenseSparse").checked = Boolean(cfg.combine_dense_and_sparse);
  document.getElementById("cfgReleasePairs").checked = Boolean(cfg.release_mitotic_pairs);
  document.getElementById("cfgRecSteps").value = cfg.recurrence_steps ?? 2;
  document.getElementById("cfgMaxTokens").value = cfg.max_new_tokens ?? 256;
  document.getElementById("cfgTemp").value = cfg.temperature ?? 0.0;
  document.getElementById("cfgEntThresh").value = cfg.recurrent_entropy_threshold ?? 1.5;
  document.getElementById("cfgDepthPonder").checked = Boolean(cfg.depth_ponder);
  document.getElementById("cfgRecKV").checked = Boolean(cfg.recurrent_kv);
  document.getElementById("cfgAdaptiveRec").checked = Boolean(cfg.adaptive_recurrent_depth);
  document.getElementById("cfgUseWhiteboard").checked = Boolean(cfg.use_whiteboard);
  document.getElementById("cfgUsePager").checked = Boolean(cfg.use_windowless_pager);

  // Sync Architecture Telemetry Box
  const archBox = document.getElementById("archStatsBox");
  if (info.loaded) {
    const m = info.latest_metrics || {};
    archBox.innerHTML = `
      <div class="stat-kv"><span class="stat-k">Checkpoint</span><span class="stat-v">${escHtml(info.checkpoint_id)}</span></div>
      <div class="stat-kv"><span class="stat-k">Step</span><span class="stat-v">${info.step}</span></div>
      <div class="stat-kv"><span class="stat-k">Total Params</span><span class="stat-v">${fmtCount(pstats.total_parameters)} (${(pstats.total_parameters||0).toLocaleString()})</span></div>
      <div class="stat-kv"><span class="stat-k">Active / Token</span><span class="stat-v" style="color:var(--cyan)">${fmtCount(pstats.active_parameters)} (${pstats.active_percent}%)</span></div>
      <div class="stat-kv"><span class="stat-k">Backbone</span><span class="stat-v">d=${arch.d_model} · L=${arch.n_encoder_layers}+${arch.n_decoder_layers} · H=${arch.n_heads}</span></div>
      <div class="stat-kv"><span class="stat-k">MoE Geometry</span><span class="stat-v">E=${arch.num_experts} · d_ff=${arch.expert_hidden_dim}</span></div>
      <div class="stat-kv"><span class="stat-k">Mitotic Pairs</span><span class="stat-v">${arch.active_mitotic_pairs} active / ${arch.saved_mitotic_pairs} saved</span></div>
      <div class="stat-kv"><span class="stat-k">Tokenizer</span><span class="stat-v">${escHtml(arch.tokenizer_type)} (V=${arch.vocab_size})</span></div>
      ${m.val_bpb ? `<div class="stat-kv"><span class="stat-k">Checkpoint val_bpb</span><span class="stat-v" style="color:var(--emerald)">${Number(m.val_bpb).toFixed(4)}</span></div>` : ""}
      ${m.val_mcqa_acc ? `<div class="stat-kv"><span class="stat-k">Checkpoint MCQA</span><span class="stat-v">${(Number(m.val_mcqa_acc)*100).toFixed(2)}%</span></div>` : ""}
    `;
  }

  // Sync Whiteboard Rules List
  document.getElementById("wbRuleCountPill").textContent =
    `${wb.episodic_rule_count || 0} rules · ||ΔW||=${wb.lora_frobenius_norm || 0}`;
  const wbList = document.getElementById("wbRulesList");
  if (wb.rules && wb.rules.length > 0) {
    wbList.innerHTML = wb.rules.map(r => `
      <div style="padding:6px 8px;background:var(--bg);border:1px solid var(--border);border-radius:4px;font-size:11px;">
        <div style="color:var(--cyan)"><strong>R${r.index} IF:</strong> ${escHtml(r.premise)}</div>
        <div style="color:var(--emerald)"><strong>THEN:</strong> ${escHtml(r.conclusion)}</div>
      </div>
    `).join("");
  } else {
    wbList.innerHTML = `<div style="color:var(--dim);font-size:11px;">No episodic rules absorbed yet.</div>`;
  }

  // Status bar
  document.getElementById("sbDevice").textContent = `GPU: ${st.gpu_name}`;
  document.getElementById("sbVram").textContent = `VRAM: ${st.vram_allocated_mb} / ${st.vram_reserved_mb} MB`;
  document.getElementById("sbPager").textContent = `Pager: ${pager.num_pages || 0} pages (||L||=${pager.ledger_norm || 0})`;
}

async function refreshStatus() {
  const st = await api("/api/status");
  renderStatus(st);
  return st;
}

async function refreshCheckpoints() {
  discoveredCheckpoints = await api("/api/checkpoints");
  const sel = document.getElementById("ckptSelect");
  if (!discoveredCheckpoints.length) {
    sel.innerHTML = `<option value="">-- No checkpoints found in runs/ --</option>`;
    return;
  }
  sel.innerHTML = discoveredCheckpoints.map(c => {
    const tag = c.has_local_pt ? "[LOCAL .pt]" : "[HF CLOUD]";
    const params = c.parameter_count ? ` · ${fmtCount(c.parameter_count)}` : "";
    const exp = c.num_experts ? ` · E=${c.num_experts}` : "";
    const bpb = c.val_bpb ? ` · bpb=${Number(c.val_bpb).toFixed(3)}` : "";
    return `<option value="${escHtml(c.checkpoint_id)}">${tag} ${escHtml(c.checkpoint_id)}${params}${exp}${bpb}</option>`;
  }).join("");
}

async function loadSelectedCheckpoint() {
  const custom = document.getElementById("customCkptPath").value.trim();
  const sel = document.getElementById("ckptSelect").value;
  const target = custom || sel;
  if (!target) {
    alert("Select a checkpoint or enter a path/ID first.");
    return;
  }
  const btn = document.getElementById("loadCkptBtn");
  btn.disabled = true;
  btn.textContent = "Loading / Pulling Weights...";
  setStatusText(`LOADING ${target}...`);
  try {
    const st = await api("/api/checkpoint/load", {
      method: "POST",
      body: JSON.stringify({
        checkpoint: target,
        device: document.getElementById("loadDevice").value,
        dtype: document.getElementById("loadDtype").value,
        hf_repo: document.getElementById("hfRepoInput").value.trim(),
        hf_token: document.getElementById("hfTokenInput").value.trim()
      })
    });
    renderStatus(st);
    await refreshCheckpoints();
    setStatusText(`LOADED ${st.loaded_info.checkpoint_id}`);
  } catch (err) {
    setStatusText(`LOAD ERROR: ${err.message}`);
    alert("Failed to load checkpoint: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Load Checkpoint Weights";
  }
}

async function loadPreset(presetName) {
  setStatusText(`INITIALIZING PRESET ${presetName}...`);
  try {
    const st = await api("/api/checkpoint/preset", {
      method: "POST",
      body: JSON.stringify({
        preset: presetName,
        device: document.getElementById("loadDevice").value,
        dtype: document.getElementById("loadDtype").value
      })
    });
    renderStatus(st);
    setStatusText(`READY (${presetName})`);
  } catch (err) {
    setStatusText(`ERROR: ${err.message}`);
    alert(err.message);
  }
}

async function setMode(mode) {
  const st = await api("/api/config", {
    method: "POST",
    body: JSON.stringify({ mode })
  });
  renderStatus(st);
}

async function pushKnobs() {
  const payload = {
    top_k: Number(document.getElementById("cfgTopK").value),
    dense_subset_k: Number(document.getElementById("cfgDenseK").value),
    recurrent_coverage_penalty: Number(document.getElementById("cfgTabu").value),
    combine_dense_and_sparse: document.getElementById("cfgCombineDenseSparse").checked,
    release_mitotic_pairs: document.getElementById("cfgReleasePairs").checked,
    recurrence_steps: Number(document.getElementById("cfgRecSteps").value),
    max_new_tokens: Number(document.getElementById("cfgMaxTokens").value),
    temperature: Number(document.getElementById("cfgTemp").value),
    recurrent_entropy_threshold: Number(document.getElementById("cfgEntThresh").value),
    depth_ponder: document.getElementById("cfgDepthPonder").checked,
    recurrent_kv: document.getElementById("cfgRecKV").checked,
    adaptive_recurrent_depth: document.getElementById("cfgAdaptiveRec").checked,
    use_whiteboard: document.getElementById("cfgUseWhiteboard").checked,
    use_windowless_pager: document.getElementById("cfgUsePager").checked
  };
  const st = await api("/api/config", {
    method: "POST",
    body: JSON.stringify(payload)
  });
  renderStatus(st);
}

function renderExpertHeatmap(routing) {
  const box = document.getElementById("expertHeatmapBox");
  if (!routing || (!routing.encoder_loads?.length && !routing.decoder_loads?.length)) {
    return;
  }
  const allLayers = [
    ...(routing.encoder_loads || []).map((loads, i) => ({
      label: `Enc ${i}`,
      loads,
      top: (routing.encoder_top_experts || [])[i] || []
    })),
    ...(routing.decoder_loads || []).map((loads, i) => ({
      label: `Dec ${i}`,
      loads,
      top: (routing.decoder_top_experts || [])[i] || []
    }))
  ];
  const numE = allLayers[0]?.loads?.length || 0;
  document.getElementById("heatmapMeta").textContent = `E=${numE}`;

  box.innerHTML = allLayers.map(layer => {
    const maxLoad = Math.max(0.01, ...layer.loads);
    // Bucket into at most 48 visual bars if E is huge (e.g. E=416)
    const buckets = Math.min(48, layer.loads.length);
    const step = layer.loads.length / buckets;
    let cellsHtml = "";
    for (let b = 0; b < buckets; b++) {
      const start = Math.floor(b * step);
      const end = Math.max(start + 1, Math.floor((b + 1) * step));
      let mVal = 0;
      for (let j = start; j < end && j < layer.loads.length; j++) {
        if (layer.loads[j] > mVal) mVal = layer.loads[j];
      }
      const alpha = Math.min(1.0, Math.max(0.06, mVal / maxLoad));
      cellsHtml += `<div class="heatmap-cell" style="background:rgba(56,189,248,${alpha.toFixed(2)})" title="E[${start}..${end-1}]: ${(mVal*100).toFixed(1)}%"></div>`;
    }
    const topStr = layer.top.slice(0, 3).map(idx => `e${idx}`).join(",");
    return `
      <div class="heatmap-row">
        <span class="heatmap-label">${layer.label}</span>
        <div class="heatmap-cells">${cellsHtml}</div>
        <span style="width:58px;text-align:right;color:var(--cyan);font-size:10px;">${topStr}</span>
      </div>
    `;
  }).join("");
}

function appendTurnToTranscript(turn) {
  const empty = document.getElementById("emptyState");
  if (empty) empty.style.display = "none";

  const container = document.getElementById("transcript");
  const snap = turn.config_snapshot || {};
  const ver = turn.verification || {};
  const meta = turn.mode_meta || {};

  const eqBadge = ver.thought_equation_score < 1.0
    ? `<span class="pill warn">#T$# eq mismatch</span>`
    : `<span class="pill ok">#T$# eq ok</span>`;

  let candsHtml = "";
  if (turn.candidates && turn.candidates.length > 0) {
    const rows = turn.candidates.map(c => {
      const isWin = c.index === meta.selected_index;
      return `
        <tr class="${isWin ? 'cand-row-win' : ''}">
          <td>${isWin ? '★ ' : ''}#${c.index} (${escHtml(c.strategy || ('r='+c.recurrence_steps))})</td>
          <td>${c.recurrence_steps}</td>
          <td>${c.temperature}</td>
          <td>${Number(c.decision_prob).toFixed(3)}</td>
          <td>${Number(c.evaluation_utility).toFixed(3)}</td>
          <td>${Number(c.deduction_entailed).toFixed(2)} / ${Number(c.deduction_contradicted).toFixed(2)}</td>
          <td>${Number(c.metacognition_correctness).toFixed(2)}</td>
          <td><strong>${Number(c.composite_score).toFixed(3)}</strong></td>
          <td>${c.vetoed ? '<span style="color:var(--rose)">VETO</span>' : '<span style="color:var(--emerald)">OK</span>'}</td>
        </tr>
      `;
    }).join("");
    candsHtml = `
      <details class="cand-details">
        <summary class="thought-summary">
          <span>▸ COE Multi-Candidate Verification Table (${turn.candidates.length} rollouts · winner #${meta.selected_index})</span>
          <span>||z_abs||=${meta.abstraction_norm ?? '--'}</span>
        </summary>
        <div style="padding:8px 12px;overflow-x:auto;">
          <table class="cand-table">
            <thead>
              <tr>
                <th>Candidate</th><th>r</th><th>temp</th><th>P(Dec)</th><th>Util</th><th>Ent/Con</th><th>Meta</th><th>Score</th><th>Gate</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </details>
    `;
  }

  const div = document.createElement("div");
  div.className = "turn-block";
  div.innerHTML = `
    <div class="turn-user">
      <div class="turn-user-text"><span class="turn-user-prefix">#U$#</span>${escHtml(turn.prompt)}</div>
      <div class="turn-badges">
        <span class="pill">${escHtml(snap.mode?.toUpperCase())}</span>
        <span class="pill">r=${snap.recurrence_steps}</span>
        <span class="pill">k=${snap.top_k}${snap.dense_subset_k > 0 ? `+dense(${snap.dense_subset_k})` : ''}</span>
        <span class="pill">${fmtCount(snap.active_parameters)} (${snap.active_percent}%)</span>
        ${snap.use_whiteboard ? `<span class="pill ok">WB ON</span>` : ''}
      </div>
    </div>

    <details class="thought-details" open>
      <summary class="thought-summary">
        <span><span class="thought-tag">#T$#</span>reasoning trace (${turn.tokens.thought} tokens)</span>
        <span>${turn.tok_per_s} tok/s · ${turn.latency_ms} ms</span>
      </summary>
      <div class="thought-body">${escHtml(turn.segments.thought || "(empty thought trace)")}</div>
    </details>

    <div class="turn-answer"><span class="answer-tag">#A$#</span>${escHtml(turn.segments.answer)}</div>

    ${candsHtml}

    <div class="turn-footer">
      <div>Turn #${turn.turn_id} · prompt=${turn.tokens.prompt}t · gen=${turn.tokens.generated}t · tabu=${snap.recurrent_coverage_penalty}</div>
      <div style="display:flex;gap:6px;">${eqBadge}</div>
    </div>
  `;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

async function handleSlashCommand(raw) {
  const parts = raw.trim().split(/\s+/);
  const cmd = parts[0].toLowerCase();
  const arg = parts.slice(1).join(" ").trim();

  if (cmd === "/clear") {
    await clearSession();
    return true;
  }
  if (cmd === "/topk") {
    document.getElementById("cfgTopK").value = Math.max(1, parseInt(arg || "4", 10));
    await pushKnobs();
    setStatusText(`SET top_k = ${document.getElementById("cfgTopK").value}`);
    return true;
  }
  if (cmd === "/dense") {
    document.getElementById("cfgDenseK").value = Math.max(0, parseInt(arg || "0", 10));
    await pushKnobs();
    setStatusText(`SET dense_subset_k = ${document.getElementById("cfgDenseK").value}`);
    return true;
  }
  if (cmd === "/tabu") {
    document.getElementById("cfgTabu").value = Math.max(0, parseFloat(arg || "0.35"));
    await pushKnobs();
    setStatusText(`SET recurrent_coverage_penalty = ${document.getElementById("cfgTabu").value}`);
    return true;
  }
  if (cmd === "/rec") {
    document.getElementById("cfgRecSteps").value = Math.max(1, parseInt(arg || "2", 10));
    await pushKnobs();
    setStatusText(`SET recurrence_steps = ${document.getElementById("cfgRecSteps").value}`);
    return true;
  }
  if (cmd === "/temp") {
    document.getElementById("cfgTemp").value = Math.max(0, parseFloat(arg || "0.0"));
    await pushKnobs();
    setStatusText(`SET temperature = ${document.getElementById("cfgTemp").value}`);
    return true;
  }
  if (cmd === "/mode") {
    const m = (arg || "uta").toLowerCase();
    if (["uta", "coe", "dream_rl"].includes(m)) {
      await setMode(m);
      setStatusText(`SET mode = ${m}`);
    }
    return true;
  }
  if (cmd === "/wbreset") {
    await resetWhiteboard();
    return true;
  }
  if (cmd === "/ingest") {
    const mod = (parts[1] || "prose").toLowerCase();
    const txt = parts.slice(2).join(" ");
    if (!txt) {
      alert("Usage: /ingest <prose|code|web> <text>");
      return true;
    }
    document.getElementById("wbModality").value = ["prose", "code", "web"].includes(mod) ? mod : "prose";
    document.getElementById("wbIngestText").value = txt;
    switchInspectorTab("whiteboard");
    await ingestWhiteboard();
    return true;
  }
  return false;
}

async function sendPrompt() {
  const input = document.getElementById("promptInput");
  const text = input.value.trim();
  if (!text) return;

  if (text.startsWith("/")) {
    const handled = await handleSlashCommand(text);
    if (handled) {
      input.value = "";
      document.getElementById("slashMenu").classList.remove("open");
      return;
    }
  }

  const sendBtn = document.getElementById("sendBtn");
  sendBtn.disabled = true;
  sendBtn.textContent = "Running...";
  setStatusText("SYNTHESIZING #U$# -> #T$# -> #A$#...");

  try {
    const res = await api("/api/generate", {
      method: "POST",
      body: JSON.stringify({
        prompt: text,
        thought_prefix: document.getElementById("thoughtPrefixInput").value,
        answer_prefix: document.getElementById("answerPrefixInput").value,
        lock_thought: document.getElementById("lockThoughtInput").checked
      })
    });
    input.value = "";
    appendTurnToTranscript(res.turn);
    renderExpertHeatmap(res.turn.expert_routing);
    renderStatus(res.status);
    setStatusText(`DONE (${res.turn.tok_per_s} tok/s · ${res.turn.latency_ms} ms)`);
  } catch (err) {
    setStatusText(`ERROR: ${err.message}`);
    alert(err.message);
  } finally {
    sendBtn.disabled = false;
    sendBtn.textContent = "Run ↵";
  }
}

async function ingestWhiteboard() {
  const text = document.getElementById("wbIngestText").value.trim();
  if (!text) {
    alert("Enter prose, code, or HTML table to absorb into the Whiteboard.");
    return;
  }
  setStatusText("ABSORBING INTO QUARANTINED WHITEBOARD...");
  try {
    const res = await api("/api/whiteboard", {
      method: "POST",
      body: JSON.stringify({
        action: "ingest",
        modality: document.getElementById("wbModality").value,
        text,
        reset_whiteboard: document.getElementById("wbResetBefore").checked
      })
    });
    const rep = res.report || {};
    document.getElementById("wbLastReport").innerHTML = `
      <span style="color:${rep.committed ? 'var(--emerald)' : 'var(--amber)'}">
        ${rep.committed ? '✓ COMMITTED' : '⚠ ' + escHtml(rep.status)}
      </span>
      · loss ${Number(rep.whiteboard_loss_before||0).toFixed(3)} → ${Number(rep.whiteboard_loss_after||0).toFixed(3)}
      · drift=${Number(rep.anchor_drift||0).toFixed(4)}
    `;
    renderStatus(res.status);
    setStatusText(`WHITEBOARD UPDATED (${res.status.whiteboard.episodic_rule_count} rules)`);
  } catch (err) {
    setStatusText(`WB ERROR: ${err.message}`);
    alert(err.message);
  }
}

async function resetWhiteboard() {
  const res = await api("/api/whiteboard", {
    method: "POST",
    body: JSON.stringify({ action: "reset" })
  });
  document.getElementById("wbLastReport").textContent = "Whiteboard fast weights reset to exact base model.";
  renderStatus(res.status);
  setStatusText("WHITEBOARD RESET");
}

async function consolidateWhiteboard() {
  const res = await api("/api/whiteboard", {
    method: "POST",
    body: JSON.stringify({ action: "consolidate", consolidation_rate: 0.25 })
  });
  const c = res.consolidation || {};
  document.getElementById("wbLastReport").innerHTML = `
    Sleep Consolidation: updated <strong>${c.num_child_experts_updated || 0}</strong> mitotic child experts
    (parent Δ=${c.parent_expert_max_delta}, child ||Δ||_F=${Number(c.child_expert_frobenius_delta||0).toFixed(5)})
  `;
  renderStatus(res.status);
}

async function runDecisionProbe() {
  const context = document.getElementById("probeContext").value.trim();
  const question = document.getElementById("probeQuestion").value.trim();
  const options = document.getElementById("probeOptions").value.split("\n").map(s => s.trim()).filter(Boolean);
  if (options.length < 2) {
    alert("Enter at least 2 candidate options (one per line).");
    return;
  }
  setStatusText("RUNNING TYPED-COE DECISION PROBE...");
  try {
    const res = await api("/api/probe/decision", {
      method: "POST",
      body: JSON.stringify({ context, question, options })
    });
    const box = document.getElementById("probeResultBox");
    box.innerHTML = res.candidates.map(c => `
      <div style="padding:7px 8px;margin-bottom:6px;border-radius:4px;border:1px solid ${c.selected ? 'var(--emerald)' : 'var(--border)'};background:${c.selected ? 'rgba(16,185,129,0.08)' : 'var(--bg)'}">
        <div style="display:flex;justify-content:space-between;font-weight:600;color:${c.selected ? 'var(--emerald)' : 'var(--text)'}">
          <span>${c.selected ? '★ ' : ''}[#${c.index}] ${escHtml(c.option)}</span>
          <span>P=${(c.probability*100).toFixed(1)}%</span>
        </div>
        <div style="font-size:10.5px;color:var(--muted);margin-top:3px;">
          sim=${c.pearson_similarity} · util=${c.utility} · ent=${c.entailed} · con=${c.contradicted} · meta=${c.correctness}
        </div>
      </div>
    `).join("");
    setStatusText(`PROBE WINNER: Option #${res.winner_index}`);
  } catch (err) {
    setStatusText(`PROBE ERROR: ${err.message}`);
    alert(err.message);
  }
}

async function clearSession() {
  const st = await api("/api/session/clear", { method: "POST" });
  document.getElementById("transcript").innerHTML = `
    <div class="empty-state" id="emptyState">
      <h2>Session cleared</h2>
      <div>Send a new prompt or slash command below.</div>
    </div>
  `;
  renderStatus(st);
  setStatusText("SESSION CLEARED");
}

// Keyboard & Slash Command Autocomplete Listeners
const promptInput = document.getElementById("promptInput");
const slashMenu = document.getElementById("slashMenu");

promptInput.addEventListener("input", () => {
  const val = promptInput.value;
  if (val.startsWith("/") && !val.includes("\n")) {
    const q = val.toLowerCase();
    const matches = SLASH_COMMANDS.filter(s => s.cmd.startsWith(q) || q === "/");
    if (matches.length > 0) {
      slashMenu.innerHTML = matches.map(m => `
        <div class="slash-item" onclick="applySlash('${m.cmd}')">
          <span class="slash-cmd">${escHtml(m.cmd)}</span>
          <span class="slash-desc">${escHtml(m.desc)}</span>
        </div>
      `).join("");
      slashMenu.classList.add("open");
      return;
    }
  }
  slashMenu.classList.remove("open");
});

function applySlash(cmd) {
  promptInput.value = cmd;
  slashMenu.classList.remove("open");
  promptInput.focus();
}

promptInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendPrompt();
  }
});

window.addEventListener("keydown", (e) => {
  if (e.altKey && e.key.toLowerCase() === "k") {
    e.preventDefault();
    switchInspectorTab("weights");
  } else if (e.altKey && e.key.toLowerCase() === "i") {
    e.preventDefault();
    toggleInspector();
  }
});

(async function init() {
  await refreshCheckpoints();
  await refreshStatus();
})();
</script>
</body>
</html>
"""


def create_inference_web_handler(controller: InferenceSessionController):
    """Create an HTTP handler serving the OpenCode-style `MCL4-COE` Inference Frontend."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
            raw = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _send_html(self, html: str) -> None:
            raw = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            if path in ("/", "/index.html", "/inference"):
                self._send_html(OPENCODE_INFERENCE_HTML)
                return
            if path == "/api/status":
                self._send_json(controller.status())
                return
            if path == "/api/checkpoints":
                self._send_json(controller.list_available_checkpoints())
                return
            if path == "/api/history":
                self._send_json(controller.history)
                return
            self._send_json({"error": f"Not found: {path}"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                body = self._read_json_body()
                if path == "/api/checkpoint/load":
                    self._send_json(controller.load_checkpoint(body))
                    return
                if path == "/api/checkpoint/preset":
                    self._send_json(controller.load_preset(body))
                    return
                if path == "/api/config":
                    self._send_json(controller.update_config(body))
                    return
                if path == "/api/generate":
                    self._send_json(controller.generate_turn(body))
                    return
                if path == "/api/whiteboard":
                    self._send_json(controller.whiteboard_action(body))
                    return
                if path == "/api/probe/decision":
                    self._send_json(controller.probe_decision(body))
                    return
                if path == "/api/session/clear":
                    self._send_json(controller.clear_session())
                    return
                self._send_json({"error": f"Not found: {path}"}, status=HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_json(
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    },
                    status=HTTPStatus.BAD_REQUEST,
                )

    return Handler


def start_inference_ui_server(
    controller: InferenceSessionController | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 7861,
) -> ThreadingHTTPServer:
    """Create and return a `ThreadingHTTPServer` serving the OpenCode-style Inference UI."""
    ctrl = controller or InferenceSessionController()
    handler_cls = create_inference_web_handler(ctrl)
    return ThreadingHTTPServer((host, port), handler_cls)
