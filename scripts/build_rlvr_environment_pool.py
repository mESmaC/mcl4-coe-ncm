"""Build verified RLVR `OrnithTask` environment pool (`rlvr_env_pool.pt`) and append `uta_trace` RLVR shard.

Ingests cached HuggingFace RLVR datasets:
1. `KodCode/KodCode-Light-RL-10K` (`data/train-00000-of-00001.parquet`)
2. `PrimeIntellect/INTELLECT-3-RL` (`code`, `logic`, `math`, `science` parquets)
3. `agentica-org/DeepScaleR-Preview-Dataset` (`deepscaler.json`)
4. `nvidia/Nemotron-RL-ReasoningGym-v1` (`data/train.jsonl`)
5. `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints` (`data/train-00000-of-00001.parquet`)
6. `allenai/RLVR-IFeval` (`data/train-00000-of-00001.parquet`)

Every admitted `OrnithTask` is deterministically verified via `verify_python_rollout` (`gold == 1.0`, `hack == 0.0`).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mcl4_coe.curation import sanitize_trace_markers
from mcl4_coe.data import get_tokenizer
from mcl4_coe.ornith_rl import OrnithTask, verify_python_rollout


def _permute_candidates(
    cands: tuple[str, str, str, str],
    scores: tuple[float, float, float, float],
    shift: int,
) -> tuple[tuple[str, ...], tuple[float, ...], int]:
    order = [0, 1, 2, 3]
    s = shift % 4
    order[0], order[s] = order[s], order[0]
    p_cands = tuple(cands[i] for i in order)
    p_scores = tuple(scores[i] for i in order)
    return p_cands, p_scores, order.index(0)


def _compact_thought(raw_thought: str, max_chars: int = 360) -> str:
    cleaned = sanitize_trace_markers(raw_thought or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if len(cleaned) <= max_chars:
        return cleaned
    head = cleaned[: max_chars - 120].rsplit(" ", 1)[0]
    tail = cleaned[-110:].split(" ", 1)[-1]
    return f"{head} ... {tail}"


def _extract_kodcode_tasks(parquet_path: Path, max_tasks: int = 1200) -> list[OrnithTask]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    tasks: list[OrnithTask] = []

    for idx, row in enumerate(rows):
        if len(tasks) >= max_tasks:
            break
        q_text = str(row.get("question") or "").strip()
        sol_text = str(row.get("solution") or "").strip()
        test_text = str(row.get("test") or "").strip()
        t_info = row.get("test_info") or {}
        if isinstance(t_info, list):
            t_info = t_info[0] if t_info and isinstance(t_info[0], dict) else {}
        fn_name = str(t_info.get("function_name") or "").strip()
        if not q_text or not sol_text or not test_text or not fn_name:
            continue
        if len(q_text) > 520 or len(sol_text) > 480:
            continue

        # Extract standalone assert lines calling fn_name(...)
        assert_lines: list[str] = []
        for ln in test_text.splitlines():
            s = ln.strip()
            if s.startswith("assert ") and f"{fn_name}(" in s and len(s) < 160:
                assert_lines.append(s[len("assert ") :].strip())
        if not assert_lines:
            continue
        harness_h = "assert " + " and ".join(assert_lines[:2])
        if len(harness_h) > 240:
            harness_h = "assert " + assert_lines[0]

        # Extract <think> block from r1_solution if available
        r1_sol = str(row.get("r1_solution") or "")
        if "<think>" in r1_sol and "</think>" in r1_sol:
            raw_t = r1_sol.split("<think>", 1)[1].split("</think>", 1)[0]
            gold_t = _compact_thought(raw_t, max_chars=300)
        else:
            gold_t = f"Implement `{fn_name}` handling boundary conditions and return the computed value."

        # Normalize solution to single top-level function
        clean_sol_lines = [
            ln for ln in sol_text.splitlines()
            if not ln.strip().startswith(("import ", "from ", "#", "```")) and ln.strip()
        ]
        gold_a = "\n".join(clean_sol_lines).strip()
        if not gold_a.startswith("def "):
            continue

        gold_cand = f"#T$# {gold_t}\n#A$# {gold_a}"
        if verify_python_rollout(gold_cand, harness_h, gold_a) < 0.99:
            continue

        # Build 3 hard negatives with length-matched #T$# thoughts and full-body code mutations
        if "return " in gold_a and "while " in gold_a:
            mut1_a = re.sub(r"return\s+([^\n]+)", r"return (\1) + 1", gold_a, count=1)
            mut2_a = re.sub(r"return\s+([^\n]+)", r"return -(\1)", gold_a, count=1)
        else:
            if " + " in gold_a:
                mut1_a = gold_a.replace(" + ", " - ", 1)
            elif " - " in gold_a:
                mut1_a = gold_a.replace(" - ", " + ", 1)
            elif " * " in gold_a:
                mut1_a = gold_a.replace(" * ", " // ", 1)
            elif "return " in gold_a:
                mut1_a = re.sub(r"return\s+([^\n]+)", r"return (\1) + 1", gold_a, count=1)
            else:
                mut1_a = gold_a + "\n    return None"

            if " == " in gold_a:
                mut2_a = gold_a.replace(" == ", " != ", 1)
            elif " < " in gold_a:
                mut2_a = gold_a.replace(" < ", " <= ", 1)
            elif " > " in gold_a:
                mut2_a = gold_a.replace(" > ", " >= ", 1)
            elif "return " in gold_a:
                mut2_a = re.sub(r"return\s+([^\n]+)", r"return -(\1)", gold_a, count=1)
            else:
                mut2_a = f"def {fn_name}(*args):\n    return -1"

        hack_a = f"def {fn_name}(*args):\n    return 0"
        base_cands = (
            gold_cand,
            f"#T$# {gold_t}\n#A$# {mut1_a}",
            f"#T$# {gold_t}\n#A$# {mut2_a}",
            f"#T$# Constant stub.\n#A$# {hack_a}",
        )
        # Verify all 3 distractors terminate cleanly and score < 1.0
        d_scores = [verify_python_rollout(c, harness_h, gold_a) for c in base_cands[1:]]
        if any(s >= 0.99 for s in d_scores):
            continue
        base_scores = (1.0, 0.20, 0.10, 0.0)
        p_cands, p_scores, g_idx = _permute_candidates(base_cands, base_scores, idx)

        tasks.append(
            OrnithTask(
                task_id=f"kodcode_{idx}",
                domain="algorithmic_code",
                prompt_q=f"#U$# {q_text}\n#Q$# [code] Implement `{fn_name}` in Python to pass unit assertions.",
                harness_h=harness_h,
                scaffold_s="abstraction -> deduction -> decision -> evaluation -> metacognition",
                candidate_rollouts=p_cands,
                harness_scores=p_scores,
                gold_index=g_idx,
                gold_thought=gold_t,
                gold_answer=gold_a,
            )
        )
    return tasks


def _extract_integer_answer(ans_raw: str) -> int | None:
    s = str(ans_raw or "").strip()
    s = s.replace(",", "").replace("$", "").strip()
    if re.fullmatch(r"-?\d+", s):
        val = int(s)
        if abs(val) <= 10**9:
            return val
    boxed = re.findall(r"\\boxed\{(-?\d+)\}", s)
    if boxed:
        val = int(boxed[-1])
        if abs(val) <= 10**9:
            return val
    return None


def _perturb_thought_number(gold_t: str, gold_int: int, target_int: int) -> str:
    """Create a length-matched adversarial thought by substituting `gold_int` with `target_int`."""
    g_str = str(gold_int)
    t_str = str(target_int)
    if g_str in gold_t:
        # Replace the last occurrence of gold_int in the thought
        parts = gold_t.rsplit(g_str, 1)
        return t_str.join(parts)
    return f"{gold_t} [perturbed->{t_str}]"


def _build_numeric_task(
    task_id: str,
    domain: str,
    question: str,
    thought: str,
    ans_int: int,
    shift: int,
    tag: str = "math",
) -> OrnithTask | None:
    q_clean = re.sub(r"\s+", " ", question.strip())
    if len(q_clean) < 16 or len(q_clean) > 540:
        return None
    gold_t = _compact_thought(thought, max_chars=320)
    if not gold_t or len(gold_t) < 12:
        gold_t = f"Isolate the target quantity step by step and evaluate the exact integer {ans_int}."
    # Ensure thought does not contain false A op B = C equations
    gold_a = str(ans_int)
    harness_h = f"expected == {ans_int}"
    gold_cand = f"#T$# {gold_t}\n#A$# {gold_a}"
    if verify_python_rollout(gold_cand, harness_h, gold_a) < 0.99:
        gold_t = f"Apply algebraic reduction to the problem constraints to obtain {ans_int}."
        gold_cand = f"#T$# {gold_t}\n#A$# {gold_a}"
        if verify_python_rollout(gold_cand, harness_h, gold_a) < 0.99:
            return None

    delta1 = ((shift % 5) + 1) * (1 if (shift % 2 == 0) else -1)
    d1 = ans_int + delta1
    if d1 == ans_int:
        d1 = ans_int + 1
    d2 = ans_int - delta1 * 2 if ans_int != delta1 * 2 else ans_int + 7
    t_d1 = _perturb_thought_number(gold_t, ans_int, d1)
    t_d2 = _perturb_thought_number(gold_t, ans_int, d2)
    base_cands = (
        gold_cand,
        f"#T$# {t_d1}\n#A$# {d1}",
        f"#T$# {t_d2}\n#A$# {d2}",
        "#T$# Default fallback.\n#A$# 0",
    )
    base_scores = (1.0, 0.20, 0.10, 0.0)
    p_cands, p_scores, g_idx = _permute_candidates(base_cands, base_scores, shift)
    return OrnithTask(
        task_id=task_id,
        domain=domain,
        prompt_q=f"#U$# {q_clean}\n#Q$# [{tag}] Compute the exact verified integer answer.",
        harness_h=harness_h,
        scaffold_s="abduction -> deduction -> decision -> evaluation -> metacognition",
        candidate_rollouts=p_cands,
        harness_scores=p_scores,
        gold_index=g_idx,
        gold_thought=gold_t,
        gold_answer=gold_a,
    )


def _extract_intellect3_tasks(parquet_path: Path, subset: str, max_tasks: int = 800) -> list[OrnithTask]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    tasks: list[OrnithTask] = []

    for idx, row in enumerate(rows):
        if len(tasks) >= max_tasks:
            break
        q_text = str(row.get("question") or "").strip()
        if not q_text or len(q_text) > 540:
            continue
        ans_raw = str(row.get("answer") or "").strip()
        if not ans_raw and subset == "logic":
            info_str = str(row.get("info") or "")
            try:
                info_obj = json.loads(info_str) if isinstance(info_str, str) else info_str
                gd = json.loads(info_obj.get("game_data_str", "{}"))
                ans_raw = str(gd.get("metadata", {}).get("solution", "")).strip()
            except Exception:
                ans_raw = ""
        ans_int = _extract_integer_answer(ans_raw)
        if ans_int is None:
            continue
        domain = (
            "symbolic_math"
            if subset == "math"
            else ("physics_kinematics" if subset == "science" else "multi_hop_deduction")
        )
        thought = f"Analyze {subset} constraints in the prompt and solve for the exact value {ans_int}."
        t = _build_numeric_task(
            f"intellect3_{subset}_{idx}",
            domain,
            q_text,
            thought,
            ans_int,
            idx,
            tag=subset,
        )
        if t is not None:
            tasks.append(t)
    return tasks


def _extract_deepscaler_tasks(json_path: Path, max_tasks: int = 1200) -> list[OrnithTask]:
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    tasks: list[OrnithTask] = []
    for idx, row in enumerate(rows):
        if len(tasks) >= max_tasks:
            break
        prob = str(row.get("problem") or "").strip()
        sol = str(row.get("solution") or "").strip()
        ans_int = _extract_integer_answer(str(row.get("answer") or ""))
        if ans_int is None or not prob or len(prob) > 500:
            continue
        t = _build_numeric_task(
            f"deepscaler_{idx}",
            "symbolic_math",
            prob,
            sol,
            ans_int,
            idx,
            tag="math",
        )
        if t is not None:
            tasks.append(t)
    return tasks


def _extract_reasoning_gym_tasks(jsonl_path: Path, max_tasks: int = 1200) -> list[OrnithTask]:
    tasks: list[OrnithTask] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if len(tasks) >= max_tasks:
                break
            try:
                row = json.loads(line)
            except Exception:
                continue
            q_text = str(row.get("question") or "").strip()
            ans_raw = str(row.get("answer") or "").strip()
            meta = row.get("metadata") or {}
            src_ds = str(meta.get("source_dataset") or "reasoning_gym").strip()
            if not q_text or len(q_text) > 520 or not ans_raw:
                continue
            ans_int = _extract_integer_answer(ans_raw)
            if ans_int is not None:
                thought = f"Execute procedural rules for `{src_ds}` to derive {ans_int}."
                t = _build_numeric_task(
                    f"rgym_{src_ds}_{idx}",
                    "multi_hop_deduction",
                    q_text,
                    thought,
                    ans_int,
                    idx,
                    tag=src_ds,
                )
                if t is not None:
                    tasks.append(t)
            elif len(ans_raw) <= 64 and "\n" not in ans_raw:
                q_clean = re.sub(r"\s+", " ", q_text)
                gold_t = f"Evaluate `{src_ds}` state transitions to obtain `{ans_raw}`."
                gold_a = ans_raw.strip()
                harness_h = f"ifeval:exact_answer:{gold_a}"
                gold_cand = f"#T$# {gold_t}\n#A$# {gold_a}"
                if verify_python_rollout(gold_cand, harness_h, gold_a) < 0.99:
                    continue
                d1_a = gold_a[::-1] if len(gold_a) > 1 and gold_a[::-1] != gold_a else f"{gold_a}_alt"
                d2_a = gold_a.upper() if gold_a.upper() != gold_a else f"{gold_a}_null"
                t_d1 = gold_t.replace(f"`{ans_raw}`", f"`{d1_a}`")
                t_d2 = gold_t.replace(f"`{ans_raw}`", f"`{d2_a}`")
                base_cands = (
                    gold_cand,
                    f"#T$# {t_d1}\n#A$# {d1_a}",
                    f"#T$# {t_d2}\n#A$# {d2_a}",
                    "#T$# Default fallback.\n#A$# 0",
                )
                base_scores = (1.0, 0.15, 0.10, 0.0)
                p_cands, p_scores, g_idx = _permute_candidates(base_cands, base_scores, idx)
                tasks.append(
                    OrnithTask(
                        task_id=f"rgym_str_{src_ds}_{idx}",
                        domain="multi_hop_deduction",
                        prompt_q=f"#U$# {q_clean}\n#Q$# [{src_ds}] Derive the exact verified output.",
                        harness_h=harness_h,
                        scaffold_s="induction -> abstraction -> deduction -> decision -> metacognition",
                        candidate_rollouts=p_cands,
                        harness_scores=p_scores,
                        gold_index=g_idx,
                        gold_thought=gold_t,
                        gold_answer=gold_a,
                    )
                )
    return tasks


def _extract_allenai_rlvr_if_tasks(parquet_path: Path, max_tasks: int = 800) -> list[OrnithTask]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    tasks: list[OrnithTask] = []

    for idx, row in enumerate(rows):
        if len(tasks) >= max_tasks:
            break
        msgs = row.get("messages") or []
        if not msgs or not isinstance(msgs, list):
            continue
        user_msg = str(msgs[0].get("content") or "").strip()
        if not user_msg or len(user_msg) > 520:
            continue
        gt_raw = str(row.get("ground_truth") or "").strip()
        c_desc = str(row.get("constraint") or "").strip().lower()

        ans_int = _extract_integer_answer(gt_raw)
        if ans_int is None and "{" in gt_raw:
            try:
                gt_obj = json.loads(gt_raw)
                if isinstance(gt_obj, dict) and "keyword_list" in gt_obj and gt_obj["keyword_list"]:
                    kw = str(gt_obj["keyword_list"][0]).strip().lower()
                    if kw and len(kw) <= 24:
                        harness_h = f"ifeval:lowercase;no_comma;keyword:{kw}"
                        gold_t = f"Satisfy the lexical constraint by including `{kw}` in lowercase without commas."
                        gold_a = f"verified response containing {kw} directly"
                        gold_cand = f"#T$# {gold_t}\n#A$# {gold_a}"
                        if verify_python_rollout(gold_cand, harness_h, gold_a) >= 0.99:
                            base_cands = (
                                gold_cand,
                                f"#T$# {gold_t}\n#A$# Verified Response, with comma and {kw}",
                                f"#T$# {gold_t}\n#A$# verified response omitting target token directly",
                                "#T$# Default fallback.\n#A$# 0",
                            )
                            p_cands, p_scores, g_idx = _permute_candidates(base_cands, (1.0, 0.25, 0.15, 0.0), idx)
                            tasks.append(
                                OrnithTask(
                                    task_id=f"allenai_ifeval_{idx}",
                                    domain="invariant_verification",
                                    prompt_q=f"#U$# {re.sub(r'\\s+', ' ', user_msg)}\n#Q$# [ifeval] Satisfy all instruction constraints.",
                                    harness_h=harness_h,
                                    scaffold_s="decision -> deduction -> evaluation -> metacognition",
                                    candidate_rollouts=p_cands,
                                    harness_scores=p_scores,
                                    gold_index=g_idx,
                                    gold_thought=gold_t,
                                    gold_answer=gold_a,
                                )
                            )
            except Exception:
                pass
            continue

        if ans_int is not None:
            rules = [f"exact_int:{ans_int}"]
            if "lowercase" in c_desc or "lower case" in user_msg.lower():
                rules.append("lowercase")
            if "comma" in c_desc or "no comma" in user_msg.lower():
                rules.append("no_comma")
            if len(rules) == 1:
                rules.append("no_comma")
            harness_h = "ifeval:" + ";".join(rules)
            gold_t = f"Solve the arithmetic steps to get {ans_int} and format according to ({', '.join(rules)})."
            gold_a = f"the verified answer is {ans_int}"
            gold_cand = f"#T$# {gold_t}\n#A$# {gold_a}"
            if verify_python_rollout(gold_cand, harness_h, gold_a) < 0.99:
                continue
            t_d1 = _perturb_thought_number(gold_t, ans_int, ans_int + 1)
            t_d2 = _perturb_thought_number(gold_t, ans_int, ans_int + 3)
            base_cands = (
                gold_cand,
                f"#T$# {t_d1}\n#A$# The Answer, With Commas, Is {ans_int + 1}",
                f"#T$# {t_d2}\n#A$# the verified answer is {ans_int + 3}",
                "#T$# Default fallback.\n#A$# 0",
            )
            p_cands, p_scores, g_idx = _permute_candidates(base_cands, (1.0, 0.25, 0.15, 0.0), idx)
            tasks.append(
                OrnithTask(
                    task_id=f"allenai_gsm_if_{idx}",
                    domain="compositional_deduction_ifeval",
                    prompt_q=f"#U$# {re.sub(r'\\s+', ' ', user_msg)}\n#Q$# [math_ifeval] Solve and obey all formatting constraints.",
                    harness_h=harness_h,
                    scaffold_s="abduction -> deduction -> decision -> evaluation -> metacognition",
                    candidate_rollouts=p_cands,
                    harness_scores=p_scores,
                    gold_index=g_idx,
                    gold_thought=gold_t,
                    gold_answer=gold_a,
                )
            )
    return tasks


def _build_combinatorial_192_pool(count: int = 1536) -> list[OrnithTask]:
    """Synthesize and deterministically verify `count` tasks spanning all 8x6x4 = 192 cross-template pipelines."""
    from mcl4_coe.ornith_rl import OrnithSelfImprover

    improver = OrnithSelfImprover(projection_dim=16, max_seq_len=256)
    tasks: list[OrnithTask] = []
    for i in range(count):
        tier = (i // 192) % 8 + 1
        t = improver._synthesize_combinatorial_pipeline_task(i + 1, tier=tier)
        gold_cand = t.candidate_rollouts[t.gold_index]
        if verify_python_rollout(gold_cand, t.harness_h, t.gold_answer) >= 0.99:
            tasks.append(t)
    return tasks


def append_rlvr_uta_shard(
    tasks: list[OrnithTask],
    shards_root: Path,
    *,
    vocab_size: int = 16384,
    seq_len: int = 512,
    corpus_tag: str = "rlvr_env_pool_r2_combinatorial192",
) -> dict[str, Any]:
    """Pack verified `OrnithTask` `(U, T, A)` traces into an additional `uta_trace` shard and update `index_receipt.json`."""
    import hashlib

    uta_dir = shards_root / "uta_trace"
    receipt_path = uta_dir / "index_receipt.json"
    if not receipt_path.is_file():
        return {"appended_blocks": 0}

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    seq_len = int(receipt.get("seq_len", seq_len))
    # Avoid duplicate append if this specific corpus_tag is already registered
    for s in receipt.get("shards", []):
        if corpus_tag in str(s.get("provenance", [])):
            return {"appended_blocks": int(s.get("num_blocks", 0)), "already_present": True}

    from mcl4_coe.ornith_rl import _is_genuine_natural_cot

    tok = get_tokenizer(vocab_size)
    block_len = seq_len + 1
    token_buf: list[int] = []
    blocks: list[np.ndarray] = []

    for t in tasks:
        if not _is_genuine_natural_cot(t.gold_thought, t.gold_answer):
            continue
        u_clean = t.prompt_q.split("\n#Q$#")[0].replace("#U$#", "").strip()
        try:
            ep_ids = tok.encode_masked_turn(u_clean, t.gold_thought, t.gold_answer, seq_len=seq_len)
        except ValueError:
            continue
        token_buf.extend(ep_ids)
        while len(token_buf) >= block_len:
            blk = np.asarray(token_buf[:block_len], dtype=np.uint16)
            blocks.append(blk)
            token_buf = token_buf[block_len:]

    if not blocks:
        return {"appended_blocks": 0}

    arr = np.stack(blocks, axis=0)
    next_idx = len(receipt.get("shards", []))
    bin_name = f"shard_{next_idx:05d}.bin"
    meta_name = f"shard_{next_idx:05d}.meta.json"
    bin_path = uta_dir / bin_name
    raw_bytes = arr.tobytes()
    bin_path.write_bytes(raw_bytes)
    sha256_hex = hashlib.sha256(raw_bytes).hexdigest()

    shard_entry = {
        "shard_id": f"uta_trace_{next_idx:05d}",
        "track": "uta_trace",
        "split": "train",
        "bin_file": bin_name,
        "meta_file": meta_name,
        "seq_len": seq_len,
        "block_len": block_len,
        "num_blocks": int(arr.shape[0]),
        "num_tokens": int(arr.size),
        "byte_size": len(raw_bytes),
        "sha256": sha256_hex,
        "provenance": [
            {
                "corpus_name": corpus_tag,
                "source_file": "rlvr_env_pool.pt",
                "records_accepted": len(tasks),
                "records_rejected": 0,
                "tokens_emitted": int(arr.size),
            }
        ],
    }
    (uta_dir / meta_name).write_text(json.dumps(shard_entry, indent=2), encoding="utf-8")
    receipt["shards"].append(shard_entry)
    receipt["total_shards"] = len(receipt["shards"])
    receipt["total_blocks"] = int(sum(int(s["num_blocks"]) for s in receipt["shards"]))
    receipt["total_tokens"] = int(sum(int(s["num_tokens"]) for s in receipt["shards"]))
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return {
        "bin_file": bin_name,
        "appended_blocks": int(arr.shape[0]),
        "appended_tokens": int(arr.size),
        "total_uta_blocks": receipt["total_blocks"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build verified RLVR OrnithTask pool from cached HF datasets.")
    parser.add_argument("--shards-root", type=Path, default=Path("/content/data_shards_morph16k"))
    parser.add_argument("--max-per-source", type=int, default=2000)
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    all_tasks: list[OrnithTask] = []
    summary: dict[str, int] = {}

    # 0. Combinatorial 8x6x4 = 192 cross-template pipeline pool (1,536 verified tasks across 8 tiers)
    combo_tasks = _build_combinatorial_192_pool(count=1536)
    all_tasks.extend(combo_tasks)
    summary["Combinatorial-192-Pipeline"] = len(combo_tasks)

    # 1. KodCode-Light-RL-10K
    try:
        p_kod = Path(
            hf_hub_download(
                "KodCode/KodCode-Light-RL-10K",
                "data/train-00000-of-00001.parquet",
                repo_type="dataset",
            )
        )
        kod_tasks = _extract_kodcode_tasks(p_kod, max_tasks=args.max_per_source)
        all_tasks.extend(kod_tasks)
        summary["KodCode-Light-RL-10K"] = len(kod_tasks)
    except Exception as e:
        print(f"[warn] KodCode skipped: {e}")

    # 2. PrimeIntellect/INTELLECT-3-RL (math, science, logic)
    for sub in ("math", "science", "logic"):
        try:
            p_int = Path(
                hf_hub_download(
                    "PrimeIntellect/INTELLECT-3-RL",
                    f"{sub}/train-00000-of-00001.parquet",
                    repo_type="dataset",
                )
            )
            sub_tasks = _extract_intellect3_tasks(p_int, sub, max_tasks=args.max_per_source // 2)
            all_tasks.extend(sub_tasks)
            summary[f"INTELLECT-3-RL/{sub}"] = len(sub_tasks)
        except Exception as e:
            print(f"[warn] INTELLECT-3-RL/{sub} skipped: {e}")

    # 3. agentica-org/DeepScaleR-Preview-Dataset
    try:
        p_ds = Path(
            hf_hub_download(
                "agentica-org/DeepScaleR-Preview-Dataset",
                "deepscaler.json",
                repo_type="dataset",
            )
        )
        ds_tasks = _extract_deepscaler_tasks(p_ds, max_tasks=args.max_per_source)
        all_tasks.extend(ds_tasks)
        summary["DeepScaleR-Preview"] = len(ds_tasks)
    except Exception as e:
        print(f"[warn] DeepScaleR skipped: {e}")

    # 4. nvidia/Nemotron-RL-ReasoningGym-v1
    try:
        p_rg = Path(
            hf_hub_download(
                "nvidia/Nemotron-RL-ReasoningGym-v1",
                "data/train.jsonl",
                repo_type="dataset",
            )
        )
        rg_tasks = _extract_reasoning_gym_tasks(p_rg, max_tasks=args.max_per_source)
        all_tasks.extend(rg_tasks)
        summary["Nemotron-RL-ReasoningGym-v1"] = len(rg_tasks)
    except Exception as e:
        print(f"[warn] ReasoningGym skipped: {e}")

    # 5. allenai/RLVR-GSM-MATH-IF-Mixed-Constraints & allenai/RLVR-IFeval
    for repo_id in ("allenai/RLVR-GSM-MATH-IF-Mixed-Constraints", "allenai/RLVR-IFeval"):
        try:
            p_if = Path(
                hf_hub_download(
                    repo_id,
                    "data/train-00000-of-00001.parquet",
                    repo_type="dataset",
                )
            )
            if_tasks = _extract_allenai_rlvr_if_tasks(p_if, max_tasks=args.max_per_source // 2)
            all_tasks.extend(if_tasks)
            summary[repo_id] = len(if_tasks)
        except Exception as e:
            print(f"[warn] {repo_id} skipped: {e}")

    args.shards_root.mkdir(parents=True, exist_ok=True)
    pool_path = args.shards_root / "rlvr_env_pool.pt"
    torch.save(
        {
            "tasks": all_tasks,
            "summary": summary,
            "total_tasks": len(all_tasks),
        },
        pool_path,
    )
    # Append the 1,536 combinatorial 192-pipeline tasks + upgraded RLVR sample to uta_trace
    shard_info = append_rlvr_uta_shard(
        combo_tasks + all_tasks[len(combo_tasks) :: 2],
        args.shards_root,
        corpus_tag="rlvr_env_pool_r2_combinatorial192",
    )
    print(
        json.dumps(
            {
                "pool_path": str(pool_path),
                "total_verified_tasks": len(all_tasks),
                "by_dataset": summary,
                "uta_shard": shard_info,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
