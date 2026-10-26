"""Lightweight Web UI & REST Controller for `MCL4-COE` Training, History & Checkpoint Comparison.

Zero external web dependencies (uses Python's built-in `http.server.ThreadingHTTPServer`).
Features:
- Live auto-refreshing SVG/Canvas charts for `total_loss`, `lm_loss`, `decision_loss`,
  per-track losses (`loss_pretrain_lm`, `loss_uta_trace`, `loss_typed_decision`),
  curriculum weight annealing, and token throughput (`tok/s`).
- Run History browser with multi-run curve overlay (check any previous run to overlay
  its loss curve against the current run).
- Checkpoint Browser & Side-by-Side Comparator with metric deltas (`Δ loss`) and
  interactive `#U$# -> #T$# -> #A$#` prompt evaluation across selected checkpoints.
- Start / Resume / Stop training runs directly from the browser.
"""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import gc
import json
from pathlib import Path
import threading
import time
import traceback
from typing import Any
from urllib.parse import parse_qs, urlparse

import torch

from .pipeline import IndexReceipt
from .trainer import (
    MCL4COETrainer,
    TrainingRunConfig,
    clear_all_runs,
    delete_run,
    evaluate_checkpoint_prompt,
    list_checkpoints,
    list_runs,
    load_run_metrics,
    resolve_corpora_root,
    resolve_runs_root,
    resolve_shards_root,
)


class TrainingControllerState:
    """Thread-safe state manager coordinating the background trainer thread and HTTP API."""

    def __init__(
        self,
        runs_root: str | Path = r"X:\MCL4-COE\runs",
        shards_root: str | Path = r"X:\MCL4-COE\data_shards",
        corpora_root: str | Path = r"X:\corpora",
    ) -> None:
        self.runs_root = resolve_runs_root(runs_root)
        self.shards_root = resolve_shards_root(shards_root)
        self.corpora_root = resolve_corpora_root(corpora_root)
        self.runs_root.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._trainer: MCL4COETrainer | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_error: str | None = None

    def shards_summary(self) -> dict[str, Any]:
        tracks: dict[str, Any] = {}
        for t_name in ("pretrain_lm", "uta_trace", "typed_decision"):
            idx_path = self.shards_root / t_name / "index_receipt.json"
            if idx_path.is_file():
                try:
                    rec = IndexReceipt.from_file(idx_path)
                    tracks[t_name] = {
                        "seq_len": rec.seq_len,
                        **rec.totals,
                    }
                except Exception:
                    continue
        return tracks

    def status(self) -> dict[str, Any]:
        with self._lock:
            is_running = self._thread is not None and self._thread.is_alive()
            trainer = self._trainer
            active_run_id = trainer.config.run_id if (is_running and trainer) else None
            current_step = trainer.current_step if trainer else 0
            total_steps = trainer.config.total_steps if trainer else 0
            latest_metrics = dict(trainer.latest_metrics) if trainer else {}
            last_error = self._last_error

        has_cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
        vram_alloc_mb = round(torch.cuda.memory_allocated(0) / (1024**2), 1) if has_cuda else 0.0
        vram_res_mb = round(torch.cuda.memory_reserved(0) / (1024**2), 1) if has_cuda else 0.0

        return {
            "is_running": is_running,
            "active_run_id": active_run_id,
            "current_step": current_step,
            "total_steps": total_steps,
            "latest_metrics": latest_metrics,
            "gpu_name": gpu_name,
            "vram_allocated_mb": vram_alloc_mb,
            "vram_reserved_mb": vram_res_mb,
            "last_error": last_error,
            "shards": self.shards_summary(),
        }

    def start_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError(f"Training run '{self._trainer.config.run_id if self._trainer else ''}' is already running.")

            run_id = str(payload.get("run_id") or f"run_{int(time.time())}").strip()
            cfg_dict = {
                **payload,
                "run_id": run_id,
                "runs_root": str(self.runs_root),
                "shards_root": str(self.shards_root),
                "corpora_root": str(self.corpora_root),
            }
            config = TrainingRunConfig.from_dict(cfg_dict)
            self._stop_event = threading.Event()
            self._last_error = None
            self._trainer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            trainer = MCL4COETrainer(config, stop_event=self._stop_event)
            self._trainer = trainer

            def _worker() -> None:
                try:
                    trainer.train()
                except Exception as exc:
                    with self._lock:
                        self._last_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

            thread = threading.Thread(target=_worker, name=f"mcl4-train-{run_id}", daemon=True)
            self._thread = thread
            thread.start()
            return {"status": "started", "run_id": run_id, "config": config.to_dict()}

    def stop_run(self) -> dict[str, Any]:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                return {"status": "idle"}
            self._stop_event.set()
            run_id = self._trainer.config.run_id if self._trainer else ""
            return {"status": "stopping", "run_id": run_id}


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>MCL4-COE Training & Checkpoint Studio</title>
<style>
  :root {
    --bg: #090d16;
    --panel: #111827;
    --panel-alt: #172033;
    --border: #26324d;
    --text: #e5e7eb;
    --muted: #94a3b8;
    --cyan: #38bdf8;
    --emerald: #34d399;
    --amber: #fbbf24;
    --rose: #fb7185;
    --violet: #a78bfa;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    font-size: 13px;
    line-height: 1.45;
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 20px;
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 20;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .brand h1 {
    margin: 0;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 0.3px;
    color: #f8fafc;
  }
  .badge {
    padding: 3px 9px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    background: var(--panel-alt);
    border: 1px solid var(--border);
  }
  .badge.running { background: rgba(52, 211, 153, 0.15); color: var(--emerald); border-color: var(--emerald); }
  .badge.idle { background: rgba(148, 163, 184, 0.12); color: var(--muted); }
  .status-pills {
    display: flex;
    gap: 14px;
    align-items: center;
    color: var(--muted);
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
  }
  .status-pills strong { color: var(--text); }
  .layout {
    display: grid;
    grid-template-columns: 320px minmax(0, 1fr);
    min-height: calc(100vh - 53px);
  }
  aside {
    background: var(--panel);
    border-right: 1px solid var(--border);
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    overflow-y: auto;
    overflow-x: hidden;
  }
  main {
    padding: 16px 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    overflow-y: auto;
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
  }
  .card h2 {
    margin: 0 0 10px 0;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
  }
  .charts-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
  }
  @media (max-width: 1300px) {
    .charts-grid { grid-template-columns: 1fr; }
    .layout { grid-template-columns: 1fr; }
  }
  canvas {
    width: 100%;
    height: 220px;
    background: #0b101b;
    border-radius: 6px;
    border: 1px solid #1e293b;
    display: block;
  }
  .form-grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 9px 8px;
  }
  .form-grid label {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 11px;
    color: var(--muted);
    min-width: 0;
  }
  .form-section-title {
    grid-column: span 2;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #64748b;
    margin-top: 4px;
    border-top: 1px solid #1e293b;
    padding-top: 7px;
  }
  input:not([type="checkbox"]), select {
    width: 100%;
    min-width: 0;
    background: #0b101b;
    border: 1px solid var(--border);
    color: var(--text);
    padding: 6px 8px;
    border-radius: 5px;
    font-size: 12px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    outline: none;
  }
  input:not([type="checkbox"]):focus, select:focus {
    border-color: var(--cyan);
  }
  input[type="checkbox"] {
    accent-color: var(--cyan);
    width: 14px;
    height: 14px;
    margin: 0;
    cursor: pointer;
  }
  button {
    background: var(--cyan);
    color: #06121d;
    border: none;
    padding: 8px 12px;
    border-radius: 5px;
    font-weight: 600;
    font-size: 12px;
    cursor: pointer;
    transition: opacity 0.15s;
  }
  button:hover { opacity: 0.9; }
  button.stop { background: var(--rose); color: #fff; }
  button.secondary {
    background: var(--panel-alt);
    color: var(--text);
    border: 1px solid var(--border);
  }
  .run-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
    max-height: 280px;
    overflow-y: auto;
  }
  .run-item {
    padding: 8px 10px;
    border-radius: 6px;
    background: #0c1322;
    border: 1px solid var(--border);
    cursor: pointer;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .run-item.active {
    border-color: var(--cyan);
    background: rgba(56, 189, 248, 0.08);
  }
  .run-item-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-weight: 600;
  }
  .run-item-meta {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: var(--muted);
    font-family: ui-monospace, monospace;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  th, td {
    text-align: left;
    padding: 7px 9px;
    border-bottom: 1px solid #1e293b;
  }
  th {
    color: var(--muted);
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
  }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  .trace-box {
    background: #070b12;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 10px 12px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .tag-u { color: var(--cyan); font-weight: 700; }
  .tag-t { color: var(--amber); font-weight: 700; }
  .tag-a { color: var(--emerald); font-weight: 700; }
  .tag-q { color: var(--violet); font-weight: 700; }
  .tag-h { color: var(--rose); font-weight: 700; }
  .compare-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 12px;
    margin-top: 10px;
  }
  .legend {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    font-size: 11px;
    color: var(--muted);
  }
  .legend span {
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }
  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
  }
</style>
</head>
<body>
<header>
  <div class="brand">
    <h1>MCL4-COE Training &amp; Checkpoint Studio</h1>
    <span id="statusBadge" class="badge idle">IDLE</span>
  </div>
  <div class="status-pills">
    <span>GPU: <strong id="gpuName">-</strong></span>
    <span>VRAM: <strong id="vramInfo">0 MB</strong></span>
    <span>Experts: <strong id="expertInfo">4E (top-2)</strong></span>
    <span>Step: <strong id="stepInfo">0 / 0</strong></span>
    <span>Loss: <strong id="lossInfo">-</strong></span>
    <span>Speed: <strong id="speedInfo">0 tok/s</strong></span>
  </div>
</header>

<div class="layout">
  <aside>
    <div class="card">
      <h2><span>Launch / Resume Run</span></h2>
      <div class="form-grid">
        <label style="grid-column: span 2;">Run ID
          <input id="inpRunId" type="text" value="rev1_mcl4_coe_4e"/>
        </label>
        <label>Total Steps
          <input id="inpSteps" type="number" value="100" min="1"/>
        </label>
        <label>Batch Size
          <input id="inpBatch" type="number" value="8" min="1"/>
        </label>
        <label>Learning Rate
          <input id="inpLr" type="number" step="0.0001" value="0.0025"/>
        </label>
        <label>Ckpt Interval
          <input id="inpCkptEvery" type="number" value="20" min="1"/>
        </label>

        <div class="form-section-title">Model &amp; Sparse MoE Architecture</div>
        <label>d_model
          <select id="inpDModel">
            <option value="128" selected>128 (Rev-1 Fast)</option>
            <option value="192">192 (Balanced)</option>
            <option value="256">256 (Rev-2 Deep)</option>
          </select>
        </label>
        <label>Layers (Enc+Dec)
          <select id="inpLayers">
            <option value="2" selected>2+2 (4 Layers)</option>
            <option value="3">3+3 (6 Layers)</option>
            <option value="4">4+4 (8 Layers)</option>
          </select>
        </label>
        <label>Sparse Experts
          <select id="inpExperts">
            <option value="4" selected>4E (Rev-1 Base)</option>
            <option value="8">8E (Mitosis Upcycle)</option>
            <option value="16">16E (Mitosis Upcycle)</option>
          </select>
        </label>
        <label>Active Top-K
          <select id="inpTopK">
            <option value="2" selected>Top-2 Active</option>
            <option value="4">Top-4 Active</option>
          </select>
        </label>

        <div class="form-section-title">Curriculum Annealing (Start &rarr; End)</div>
        <label>pretrain_lm (Start)
          <input id="inpStartPre" type="number" step="0.05" value="0.60"/>
        </label>
        <label>pretrain_lm (End)
          <input id="inpEndPre" type="number" step="0.05" value="0.15"/>
        </label>
        <label>uta_trace (Start)
          <input id="inpStartUta" type="number" step="0.05" value="0.25"/>
        </label>
        <label>uta_trace (End)
          <input id="inpEndUta" type="number" step="0.05" value="0.50"/>
        </label>

        <div class="form-section-title">Checkpoint, Ornith-1.5, Dream-RSI &amp; Precision</div>
        <label style="grid-column: span 2;">Resume / Upcycle From Checkpoint
          <select id="selResume" onchange="onResumeSelectChange(this.value)">
            <option value="">-- Fresh Run --</option>
          </select>
        </label>
        <label style="grid-column: span 2; flex-direction: row; align-items: center; gap: 8px; margin-top: 2px; color: var(--text); cursor: pointer;">
          <input id="inpOrnith" type="checkbox" checked/>
          <span>Enable <code class="mono" style="color:var(--emerald);">Ornith-1.5</code> Self-Improvement (<code class="mono">#Q$# &rarr; #H$# &rarr; #T$#</code>)</span>
        </label>
        <label style="grid-column: span 2; flex-direction: row; align-items: center; gap: 8px; margin-top: 2px; color: var(--text); cursor: pointer;">
          <input id="inpDreamRsi" type="checkbox" checked/>
          <span>Enable <code class="mono" style="color:var(--violet);">Dream-RSI</code> Evolving Worlds (<code class="mono">H_t &rarr; &pi;_{t+1}</code>)</span>
        </label>
        <label style="grid-column: span 2; flex-direction: row; align-items: center; gap: 8px; margin-top: 2px; color: var(--text); cursor: pointer;">
          <input id="inpNvfp4" type="checkbox"/>
          <span>Enable Blackwell <code class="mono" style="color:var(--cyan);">NVFP4</code> (<code class="mono">sm_120</code>)</span>
        </label>
      </div>
      <div style="display: grid; grid-template-columns: 1fr auto; gap: 8px; margin-top: 12px;">
        <button id="btnStart" onclick="startTraining()">Start Run</button>
        <button id="btnStop" class="stop" onclick="stopTraining()">Stop &amp; Save</button>
      </div>
      <div id="errBox" style="color: var(--rose); font-size: 11px; margin-top: 6px; display:none;"></div>
    </div>

    <div class="card">
      <h2>
        <span>9-Op, Ornith-1.5 &amp; Dream-RSI</span>
        <span id="ornithBadge" class="mono" style="font-size:11px; color:var(--emerald); text-transform:none;">ACTIVE</span>
      </h2>
      <div id="ornithTelemetryBox" class="mono" style="font-size: 11px; color: var(--muted);">
        Awaiting operator, Ornith-1.5 &amp; Dream-RSI telemetry...
      </div>
    </div>

    <div class="card">
      <h2>
        <span>Sparse MoE Expert Load</span>
        <span id="moeLoadBadge" class="mono" style="font-size:11px; color:var(--cyan); text-transform:none;">4E</span>
      </h2>
      <div id="expertLoadBox" class="mono" style="font-size: 11px; color: var(--muted);">
        Awaiting step telemetry...
      </div>
    </div>

    <div class="card">
      <h2>
        <span>Run History &amp; Overlay</span>
        <span style="display:flex; gap:5px;">
          <button class="secondary" style="padding: 3px 8px; font-size: 11px;" onclick="refreshAll()">Refresh</button>
          <button class="secondary" style="padding: 3px 8px; font-size: 11px; color: var(--rose);" onclick="clearHistory()">Clear</button>
        </span>
      </h2>
      <div style="font-size: 11px; color: var(--muted); margin-bottom: 6px;">
        Click a run to inspect; check box to overlay on Chart 1.
      </div>
      <div id="runList" class="run-list"></div>
    </div>

    <div class="card">
      <h2><span>Sharded Corpus Receipts</span></h2>
      <div id="shardSummary" class="mono" style="font-size: 11px; color: var(--muted);">Loading...</div>
    </div>
  </aside>

  <main>
    <div class="charts-grid">
      <div class="card">
        <h2>
          <span>1. Joint &amp; 9-Operator Loss</span>
          <div class="legend">
            <span><i class="dot" style="background:#38bdf8"></i>Total</span>
            <span><i class="dot" style="background:#34d399"></i>LM</span>
            <span><i class="dot" style="background:#fbbf24"></i>Decision</span>
            <span><i class="dot" style="background:#a78bfa"></i>Deduction</span>
          </div>
        </h2>
        <canvas id="chartLoss" width="460" height="220"></canvas>
      </div>

      <div class="card">
        <h2>
          <span>2. Per-Track Token Loss</span>
          <div class="legend">
            <span><i class="dot" style="background:#38bdf8"></i>pretrain_lm</span>
            <span><i class="dot" style="background:#fbbf24"></i>uta_trace</span>
            <span><i class="dot" style="background:#a78bfa"></i>typed_decision</span>
          </div>
        </h2>
        <canvas id="chartTracks" width="460" height="220"></canvas>
      </div>

      <div class="card">
        <h2>
          <span>3. Curriculum Weights &amp; Speed</span>
          <div class="legend">
            <span><i class="dot" style="background:#34d399"></i>tok/s</span>
            <span><i class="dot" style="background:#fb7185"></i>uta_weight</span>
          </div>
        </h2>
        <canvas id="chartSpeed" width="460" height="220"></canvas>
      </div>
    </div>

    <div class="card">
      <h2>
        <span>Checkpoint Browser &amp; Side-by-Side Comparator</span>
        <div style="display:flex; gap: 8px; align-items:center;">
          <input id="inpEvalPrompt" type="text" value="A rabbit carries two red balls. Count?" style="width: 260px;" placeholder="Prompt to evaluate on selected checkpoints..."/>
          <button onclick="evaluateSelectedCheckpoints()" style="white-space: nowrap;">Compare &amp; Run #U$# &rarr; #T$# &rarr; #A$#</button>
        </div>
      </h2>
      <div style="max-height: 230px; overflow-y: auto;">
        <table>
          <thead>
            <tr>
              <th>Select</th>
              <th>Checkpoint</th>
              <th>Step</th>
              <th>Total Loss</th>
              <th>LM Loss</th>
              <th>Decision Loss</th>
              <th>uta_trace Loss</th>
              <th>Speed</th>
              <th>Experts / Precision</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody id="ckptTableBody"></tbody>
        </table>
      </div>
      <div id="compareContainer" class="compare-grid"></div>
    </div>

    <div class="card">
      <h2>
        <span>Live #U$# &rarr; #T$# &rarr; #A$# Scratchpad Evolution</span>
        <span id="activeRunLabel" class="mono" style="color:var(--cyan); text-transform:none;">-</span>
      </h2>
      <div id="samplesContainer" class="compare-grid"></div>
    </div>
  </main>
</div>

<script>
let selectedRunId = null;
let overlayRunIds = new Set();
let runDataCache = {};
let allCheckpoints = [];
let selectedCkptMetaFiles = new Set();

function formatTraceHtml(text) {
  if (!text) return "";
  let escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  return escaped
    .replace(/#U\$#/g, '<span class="tag-u">#U$#</span>')
    .replace(/#T\$#/g, '<span class="tag-t">#T$#</span>')
    .replace(/#A\$#/g, '<span class="tag-a">#A$#</span>')
    .replace(/#Q\$#/g, '<span class="tag-q">#Q$#</span>')
    .replace(/#H\$#/g, '<span class="tag-h">#H$#</span>');
}

function drawMultiLineChart(canvasId, seriesList, yLabel) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const padL = 44, padR = 14, padT = 16, padB = 26;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  let allX = [];
  let allY = [];
  for (const s of seriesList) {
    for (const pt of s.points) {
      if (Number.isFinite(pt.x) && Number.isFinite(pt.y)) {
        allX.push(pt.x);
        allY.push(pt.y);
      }
    }
  }

  ctx.strokeStyle = "#1e293b";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padT + (plotH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(W - padR, y);
    ctx.stroke();
  }

  if (allX.length === 0) {
    ctx.fillStyle = "#64748b";
    ctx.font = "12px ui-monospace, monospace";
    ctx.fillText("No telemetry points yet", padL + 20, padT + plotH / 2);
    return;
  }

  const minX = Math.min(...allX);
  const maxX = Math.max(...allX, minX + 1);
  let minY = Math.min(...allY);
  let maxY = Math.max(...allY);
  if (Math.abs(maxY - minY) < 1e-6) {
    minY -= 0.5;
    maxY += 0.5;
  } else {
    const margin = 0.08 * (maxY - minY);
    minY = Math.max(0, minY - margin);
    maxY += margin;
  }

  ctx.fillStyle = "#94a3b8";
  ctx.font = "10px ui-monospace, monospace";
  for (let i = 0; i <= 4; i++) {
    const val = maxY - ((maxY - minY) * i) / 4;
    const y = padT + (plotH * i) / 4;
    ctx.fillText(val.toFixed(2), 4, y + 3);
  }
  ctx.fillText("step " + minX, padL, H - 6);
  ctx.fillText("step " + maxX, W - padR - 48, H - 6);

  for (const s of seriesList) {
    if (!s.points.length) continue;
    ctx.save();
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.dashed ? 1.5 : 2.2;
    if (s.dashed) ctx.setLineDash([4, 4]);
    ctx.beginPath();
    s.points.forEach((pt, idx) => {
      const px = padL + ((pt.x - minX) / (maxX - minX)) * plotW;
      const py = padT + (1.0 - (pt.y - minY) / (maxY - minY)) * plotH;
      if (idx === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
    ctx.restore();
  }
}

async function fetchJson(url, opts) {
  const res = await fetch(url, opts);
  return await res.json();
}

function renderExpertLoad(loads, numExperts) {
  const nE = numExperts || (loads ? loads.length : 4);
  document.getElementById("moeLoadBadge").textContent = `${nE}E`;
  document.getElementById("expertInfo").textContent = `${nE}E`;
  const box = document.getElementById("expertLoadBox");
  if (!loads || !loads.length) {
    box.innerHTML = "Awaiting step telemetry...";
    return;
  }
  const ideal = 100.0 / loads.length;
  const rows = loads.map((v, idx) => {
    const pct = (v * 100.0).toFixed(1);
    const barW = Math.min(100, Math.max(2, v * 100.0 * (loads.length / 2.2)));
    const dev = Math.abs(v * 100.0 - ideal);
    const col = dev > ideal * 0.65 ? "#fbbf24" : "#38bdf8";
    return `<div style="display:flex; align-items:center; gap:6px; margin-bottom:3px;">
      <span style="width:24px; color:var(--text);">E${idx}</span>
      <div style="flex:1; background:#0b101b; height:7px; border-radius:4px; overflow:hidden; border:1px solid #1e293b;">
        <div style="width:${barW}%; height:100%; background:${col};"></div>
      </div>
      <span style="width:42px; text-align:right;">${pct}%</span>
    </div>`;
  });
  box.innerHTML = rows.join("");
}

function renderOrnithTelemetry(m) {
  const box = document.getElementById("ornithTelemetryBox");
  const badge = document.getElementById("ornithBadge");
  if (!box) return;
  if (!m || (m.deduction_loss === undefined && m.ornith_task_reward === undefined)) {
    box.innerHTML = "Run with 9-Operator + Ornith-1.5 + Dream-RSI enabled to view live metrics.";
    if (badge) badge.textContent = "STAGE 1/2";
    return;
  }
  if (badge) {
    badge.textContent = m.dream_cycle ? `Dream t=${m.dream_cycle} | Scaffolds: ${m.scaffolds_cached ?? 0}` : (m.ornith_task_reward !== undefined ? `Scaffolds: ${m.scaffolds_cached ?? 0}` : "9-OP");
  }
  const fmt = (v, d=3) => (v !== undefined && v !== null ? Number(v).toFixed(d) : "-");
  const dreamBlock = m.dream_cycle ? `
    <div style="margin-top:6px; padding-top:6px; border-top:1px solid #1e293b;">
      <div style="color:var(--violet); font-weight:700; margin-bottom:3px;">Dream-RSI Evolving Worlds (0-Exec Replay)</div>
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:2px 10px;">
        <div>Worlds |H_t|: <strong style="color:var(--cyan);">${m.dream_worlds_archived ?? 0}</strong></div>
        <div>Replay J(&pi;): <strong style="color:var(--emerald);">${fmt(m.dream_replay_score, 4)}</strong></div>
        <div>Replays (0-Ex): <strong style="color:var(--amber);">${m.dream_replays_dreamed ?? 0}</strong></div>
        <div>Accepted &pi;: <strong style="color:var(--violet);">${m.dream_accepted_revisions ?? 0}</strong></div>
      </div>
      ${m.dream_active_policy ? `<pre style="margin:5px 0 0 0; padding:5px 7px; background:#070b12; border:1px solid #1e293b; border-radius:4px; color:#cbd5e1; font-size:10px; white-space:pre-wrap;">${m.dream_active_policy}</pre>` : ""}
    </div>
  ` : "";
  box.innerHTML = `
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:3px 10px;">
      <div>Decision CE: <strong style="color:var(--amber);">${fmt(m.decision_loss)}</strong></div>
      <div>Deduction CE: <strong style="color:var(--violet);">${fmt(m.deduction_loss)}</strong></div>
      <div>Eval/Assoc: <strong style="color:var(--cyan);">${fmt(m.evaluation_loss)}</strong></div>
      <div>Metacog BCE: <strong style="color:var(--emerald);">${fmt(m.metacognition_loss)}</strong></div>
      <div>R_task (#Q$#): <strong style="color:var(--emerald);">${fmt(m.ornith_task_reward)}</strong></div>
      <div>R_harness (#H$#): <strong style="color:var(--cyan);">${fmt(m.ornith_harness_reward)}</strong></div>
      <div>Pass Rate (p): <strong style="color:var(--amber);">${fmt(m.ornith_pass_rate, 2)}</strong></div>
      <div>Assoc Novelty: <strong style="color:var(--violet);">${fmt(m.ornith_novelty)}</strong></div>
    </div>
    ${dreamBlock}
  `;
}

async function refreshStatus() {
  const st = await fetchJson("/api/status");
  const badge = document.getElementById("statusBadge");
  if (st.is_running) {
    badge.textContent = "RUNNING: " + st.active_run_id;
    badge.className = "badge running";
    if (!selectedRunId) selectedRunId = st.active_run_id;
  } else {
    badge.textContent = "IDLE";
    badge.className = "badge idle";
  }
  document.getElementById("gpuName").textContent = st.gpu_name || "-";
  document.getElementById("vramInfo").textContent = `${st.vram_allocated_mb} / ${st.vram_reserved_mb} MB`;
  document.getElementById("stepInfo").textContent = `${st.current_step} / ${st.total_steps}`;
  const m = st.latest_metrics || {};
  document.getElementById("lossInfo").textContent = m.total_loss !== undefined ? m.total_loss.toFixed(4) : "-";
  document.getElementById("speedInfo").textContent = m.tok_per_s !== undefined ? `${m.tok_per_s} tok/s` : "0 tok/s";
  if (m.expert_load) {
    renderExpertLoad(m.expert_load, m.num_experts);
  }
  if (st.is_running && Object.keys(m).length > 0) {
    renderOrnithTelemetry(m);
  }

  const errBox = document.getElementById("errBox");
  if (st.last_error) {
    errBox.style.display = "block";
    errBox.textContent = st.last_error;
  } else {
    errBox.style.display = "none";
  }

  const shardLines = [];
  for (const [tName, info] of Object.entries(st.shards || {})) {
    shardLines.push(`${tName.padEnd(15)}: ${info.num_blocks} blks (${info.num_shards} shards, ${(info.num_tokens/1e6).toFixed(2)}M tok)`);
  }
  document.getElementById("shardSummary").innerHTML = shardLines.join("<br/>") || "No shards found";
  return st;
}

async function refreshRunsAndCharts() {
  const runs = await fetchJson("/api/runs");
  const validIds = new Set(runs.map(r => r.run_id));
  if (selectedRunId && !validIds.has(selectedRunId)) {
    selectedRunId = null;
  }
  if (!selectedRunId && runs.length > 0) {
    selectedRunId = runs[0].run_id;
  }
  const listEl = document.getElementById("runList");
  listEl.innerHTML = "";
  for (const r of runs) {
    const div = document.createElement("div");
    div.className = "run-item" + (r.run_id === selectedRunId ? " active" : "");
    const lm = r.latest_metrics || {};
    const checked = overlayRunIds.has(r.run_id) ? "checked" : "";
    div.innerHTML = `
      <div class="run-item-top">
        <span onclick="selectRun('${r.run_id}')" style="flex:1;">${r.run_id}</span>
        <label style="font-size:10px; color:var(--muted); display:flex; align-items:center; gap:3px;">
          <input type="checkbox" ${checked} onchange="toggleOverlay('${r.run_id}', this.checked)"/> overlay
        </label>
      </div>
      <div class="run-item-meta" onclick="selectRun('${r.run_id}')">
        <span>${r.status} (${r.current_step}/${r.total_steps})</span>
        <span>loss=${lm.total_loss !== undefined ? lm.total_loss.toFixed(3) : "-"}</span>
      </div>
    `;
    listEl.appendChild(div);
  }

  if (selectedRunId) {
    document.getElementById("activeRunLabel").textContent = selectedRunId;
    const primary = await fetchJson(`/api/runs/${encodeURIComponent(selectedRunId)}`);
    runDataCache[selectedRunId] = primary;
    const pts = primary.metrics || [];
    if (pts.length > 0) {
      const lastPt = pts[pts.length - 1];
      if (lastPt.expert_load) {
        renderExpertLoad(lastPt.expert_load, lastPt.num_experts);
      }
      renderOrnithTelemetry(lastPt);
    }

    const lossSeries = [
      { color: "#38bdf8", points: pts.map(p => ({ x: p.step, y: p.total_loss })) },
      { color: "#34d399", points: pts.map(p => ({ x: p.step, y: p.lm_loss })) },
      { color: "#fbbf24", points: pts.map(p => ({ x: p.step, y: p.decision_loss })) },
      { color: "#a78bfa", points: pts.filter(p => p.deduction_loss !== undefined).map(p => ({ x: p.step, y: p.deduction_loss })) },
    ];

    const overlayColors = ["#fb7185", "#a78bfa", "#f472b6", "#60a5fa"];
    let cIdx = 0;
    for (const ovId of overlayRunIds) {
      if (ovId === selectedRunId) continue;
      const ovData = await fetchJson(`/api/runs/${encodeURIComponent(ovId)}`);
      const ovPts = ovData.metrics || [];
      lossSeries.push({
        color: overlayColors[cIdx % overlayColors.length],
        dashed: true,
        points: ovPts.map(p => ({ x: p.step, y: p.total_loss })),
      });
      cIdx++;
    }
    drawMultiLineChart("chartLoss", lossSeries, "Loss");

    const trackSeries = [
      { color: "#38bdf8", points: pts.filter(p => p.loss_pretrain_lm !== undefined).map(p => ({ x: p.step, y: p.loss_pretrain_lm })) },
      { color: "#fbbf24", points: pts.filter(p => p.loss_uta_trace !== undefined).map(p => ({ x: p.step, y: p.loss_uta_trace })) },
      { color: "#a78bfa", points: pts.filter(p => p.loss_typed_decision !== undefined).map(p => ({ x: p.step, y: p.loss_typed_decision })) },
    ];
    drawMultiLineChart("chartTracks", trackSeries, "Track Loss");

    const speedSeries = [
      { color: "#34d399", points: pts.map(p => ({ x: p.step, y: p.tok_per_s })) },
      { color: "#fb7185", points: pts.map(p => ({ x: p.step, y: ((p.weights || {}).uta_trace || 0) * 5000 })) },
    ];
    drawMultiLineChart("chartSpeed", speedSeries, "tok/s");

    const samplesEl = document.getElementById("samplesContainer");
    samplesEl.innerHTML = "";
    const samples = (primary.samples || []).slice(-4).reverse();
    for (const s of samples) {
      const card = document.createElement("div");
      card.className = "trace-box";
      card.innerHTML = `<div style="color:var(--muted); font-size:11px; margin-bottom:4px;">Step ${s.step} | total_loss=${s.total_loss}</div>` + formatTraceHtml(s.trace);
      samplesEl.appendChild(card);
    }
  } else {
    document.getElementById("activeRunLabel").textContent = "-";
    drawMultiLineChart("chartLoss", [], "Loss");
    drawMultiLineChart("chartTracks", [], "Track Loss");
    drawMultiLineChart("chartSpeed", [], "tok/s");
    document.getElementById("samplesContainer").innerHTML = "";
    renderExpertLoad(null, parseInt(document.getElementById("inpExperts").value, 10));
    renderOrnithTelemetry(null);
  }
}

async function refreshCheckpoints() {
  allCheckpoints = await fetchJson("/api/checkpoints");
  const tbody = document.getElementById("ckptTableBody");
  const selResume = document.getElementById("selResume");
  const currentResume = selResume.value;

  tbody.innerHTML = "";
  selResume.innerHTML = '<option value="">-- Fresh Run --</option>';

  for (const c of allCheckpoints) {
    const opt = document.createElement("option");
    opt.value = c.meta_file;
    opt.textContent = `${c.checkpoint_id} (${c.num_experts}E | loss=${(c.metrics || {}).total_loss ?? "-"})`;
    selResume.appendChild(opt);

    const m = c.metrics || {};
    const checked = selectedCkptMetaFiles.has(c.meta_file) ? "checked" : "";
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><input type="checkbox" ${checked} onchange="toggleCkpt('${c.meta_file.replace(/\\/g, "\\\\")}', this.checked)"/></td>
      <td class="mono" style="color:var(--cyan);">${c.checkpoint_id}</td>
      <td class="mono">${c.step} / ${c.total_steps}</td>
      <td class="mono">${m.total_loss !== undefined ? m.total_loss.toFixed(4) : "-"}</td>
      <td class="mono">${m.lm_loss !== undefined ? m.lm_loss.toFixed(4) : "-"}</td>
      <td class="mono">${m.decision_loss !== undefined ? m.decision_loss.toFixed(4) : "-"}</td>
      <td class="mono">${m.loss_uta_trace !== undefined ? m.loss_uta_trace.toFixed(4) : "-"}</td>
      <td class="mono">${m.tok_per_s !== undefined ? m.tok_per_s + " tok/s" : "-"}</td>
      <td class="mono">${c.num_experts}E / ${c.use_nvfp4 ? "NVFP4" : "BF16"}</td>
      <td>
        <button class="secondary" style="padding:2px 7px; font-size:11px;" onclick="selectForResume('${c.meta_file.replace(/\\/g, "\\\\")}')">Use to Resume</button>
      </td>
    `;
    tbody.appendChild(tr);
  }
  selResume.value = currentResume;
}

function selectRun(runId) {
  selectedRunId = runId;
  refreshRunsAndCharts();
}

function toggleOverlay(runId, enabled) {
  if (enabled) overlayRunIds.add(runId);
  else overlayRunIds.delete(runId);
  refreshRunsAndCharts();
}

function toggleCkpt(metaFile, enabled) {
  if (enabled) selectedCkptMetaFiles.add(metaFile);
  else selectedCkptMetaFiles.delete(metaFile);
  renderStaticCheckpointComparison();
}

function onResumeSelectChange(metaFile) {
  if (!metaFile) return;
  const ckpt = allCheckpoints.find(c => c.meta_file === metaFile);
  if (!ckpt || !ckpt.config) return;
  if (ckpt.config.d_model) {
    document.getElementById("inpDModel").value = String(ckpt.config.d_model);
  }
  if (ckpt.config.n_encoder_layers) {
    document.getElementById("inpLayers").value = String(ckpt.config.n_encoder_layers);
  }
}

function selectForResume(metaFile) {
  document.getElementById("selResume").value = metaFile;
  onResumeSelectChange(metaFile);
}

function renderStaticCheckpointComparison() {
  const container = document.getElementById("compareContainer");
  container.innerHTML = "";
  const chosen = allCheckpoints.filter(c => selectedCkptMetaFiles.has(c.meta_file));
  if (chosen.length === 0) return;

  const baseLoss = (chosen[0].metrics || {}).total_loss;
  for (const c of chosen) {
    const m = c.metrics || {};
    const delta = (baseLoss !== undefined && m.total_loss !== undefined)
      ? (m.total_loss - baseLoss).toFixed(4)
      : "0.0000";
    const box = document.createElement("div");
    box.className = "trace-box";
    box.innerHTML = `
      <div style="display:flex; justify-content:space-between; margin-bottom:6px; border-bottom:1px solid #1e293b; padding-bottom:4px;">
        <strong style="color:var(--cyan);">${c.checkpoint_id}</strong>
        <span style="color:var(--muted);">step=${c.step} | Δloss=${delta}</span>
      </div>
      <div style="font-size:11px; color:var(--muted); margin-bottom:6px;">
        total=${m.total_loss ?? "-"} | lm=${m.lm_loss ?? "-"} | dec=${m.decision_loss ?? "-"} | uta=${m.loss_uta_trace ?? "-"}
      </div>
      <div>${formatTraceHtml(c.sample_uta || "")}</div>
    `;
    container.appendChild(box);
  }
}

async function evaluateSelectedCheckpoints() {
  const prompt = document.getElementById("inpEvalPrompt").value.trim();
  const files = Array.from(selectedCkptMetaFiles);
  if (files.length === 0) {
    alert("Select at least one checkpoint checkbox in the table first.");
    return;
  }
  const res = await fetchJson("/api/checkpoints/eval", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ checkpoints: files, prompt }),
  });
  const container = document.getElementById("compareContainer");
  container.innerHTML = "";
  for (const item of res.results || []) {
    const m = item.metrics || {};
    const box = document.createElement("div");
    box.className = "trace-box";
    box.innerHTML = `
      <div style="display:flex; justify-content:space-between; margin-bottom:6px; border-bottom:1px solid #1e293b; padding-bottom:4px;">
        <strong style="color:var(--emerald);">${item.checkpoint_id}</strong>
        <span style="color:var(--muted);">step=${item.step} | ${item.latency_ms} ms</span>
      </div>
      <div style="font-size:11px; color:var(--muted); margin-bottom:6px;">
        total_loss=${m.total_loss ?? "-"} | lm_loss=${m.lm_loss ?? "-"} | decision_loss=${m.decision_loss ?? "-"}
      </div>
      <div>${formatTraceHtml(item.trace)}</div>
    `;
    container.appendChild(box);
  }
}

async function startTraining() {
  const preStart = parseFloat(document.getElementById("inpStartPre").value);
  const utaStart = parseFloat(document.getElementById("inpStartUta").value);
  const decStart = Math.max(0.0, +(1.0 - preStart - utaStart).toFixed(4));

  const preEnd = parseFloat(document.getElementById("inpEndPre").value);
  const utaEnd = parseFloat(document.getElementById("inpEndUta").value);
  const decEnd = Math.max(0.0, +(1.0 - preEnd - utaEnd).toFixed(4));

  const dModel = parseInt(document.getElementById("inpDModel").value, 10);
  const nLayers = parseInt(document.getElementById("inpLayers").value, 10);
  const numExperts = parseInt(document.getElementById("inpExperts").value, 10);
  const topK = parseInt(document.getElementById("inpTopK").value, 10);

  const payload = {
    run_id: document.getElementById("inpRunId").value.trim(),
    total_steps: parseInt(document.getElementById("inpSteps").value, 10),
    batch_size: parseInt(document.getElementById("inpBatch").value, 10),
    lr: parseFloat(document.getElementById("inpLr").value),
    d_model: dModel,
    n_heads: dModel >= 256 ? 8 : 4,
    n_encoder_layers: nLayers,
    n_decoder_layers: nLayers,
    expert_hidden_dim: dModel * 2,
    ple_dim: Math.max(32, Math.floor(dModel / 4)),
    num_experts: numExperts,
    top_k: topK,
    checkpoint_every: parseInt(document.getElementById("inpCkptEvery").value, 10),
    enable_ornith_rl: document.getElementById("inpOrnith").checked,
    enable_dream_rsi: document.getElementById("inpDreamRsi").checked,
    use_nvfp4: document.getElementById("inpNvfp4").checked,
    resume_checkpoint: document.getElementById("selResume").value || null,
    start_weights: { pretrain_lm: preStart, uta_trace: utaStart, typed_decision: decStart },
    end_weights: { pretrain_lm: preEnd, uta_trace: utaEnd, typed_decision: decEnd },
  };
  const res = await fetchJson("/api/train/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (res.run_id) {
    selectedRunId = res.run_id;
  }
  await refreshAll();
}

async function stopTraining() {
  await fetchJson("/api/train/stop", { method: "POST" });
  await refreshAll();
}

async function clearHistory() {
  await fetchJson("/api/runs/clear", { method: "POST" });
  selectedRunId = null;
  overlayRunIds.clear();
  selectedCkptMetaFiles.clear();
  document.getElementById("compareContainer").innerHTML = "";
  await refreshAll();
}

async function refreshAll() {
  await refreshStatus();
  await refreshRunsAndCharts();
  await refreshCheckpoints();
}

refreshAll();
setInterval(async () => {
  const st = await refreshStatus();
  if (st.is_running) {
    await refreshRunsAndCharts();
    await refreshCheckpoints();
  }
}, 2000);
</script>
</body>
</html>
"""


def create_web_handler(controller: TrainingControllerState):
    """Create a request handler bound to `controller`."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            # Keep console output clean during live polling
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
            if path in ("/", "/index.html"):
                self._send_html(DASHBOARD_HTML)
                return
            if path == "/api/status":
                self._send_json(controller.status())
                return
            if path == "/api/runs":
                self._send_json(list_runs(controller.runs_root))
                return
            if path.startswith("/api/runs/"):
                run_id = path[len("/api/runs/") :].strip("/")
                self._send_json(load_run_metrics(run_id, controller.runs_root))
                return
            if path == "/api/checkpoints":
                self._send_json(list_checkpoints(controller.runs_root))
                return
            self._send_json({"error": f"Not found: {path}"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                body = self._read_json_body()
                if path == "/api/train/start":
                    self._send_json(controller.start_run(body))
                    return
                if path == "/api/train/stop":
                    self._send_json(controller.stop_run())
                    return
                if path == "/api/runs/clear":
                    st = controller.status()
                    removed = clear_all_runs(
                        controller.runs_root,
                        exclude_run_id=st.get("active_run_id"),
                    )
                    self._send_json({"status": "cleared", "removed": removed})
                    return
                if path == "/api/runs/delete":
                    rid = str(body.get("run_id") or "")
                    ok = delete_run(rid, controller.runs_root)
                    self._send_json({"status": "deleted" if ok else "not_found", "run_id": rid})
                    return
                if path == "/api/checkpoints/eval":
                    ckpts = body.get("checkpoints", [])
                    prompt = str(body.get("prompt") or "What is 2+2?")
                    results = [
                        evaluate_checkpoint_prompt(c, prompt, max_new_tokens=48)
                        for c in ckpts
                    ]
                    self._send_json({"results": results})
                    return
                self._send_json({"error": f"Not found: {path}"}, status=HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_json(
                    {"error": f"{type(exc).__name__}: {exc}"},
                    status=HTTPStatus.BAD_REQUEST,
                )

    return Handler


def start_web_ui_server(
    controller: TrainingControllerState,
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
) -> ThreadingHTTPServer:
    """Create and return a `ThreadingHTTPServer` serving the `MCL4-COE` Studio UI."""
    handler_cls = create_web_handler(controller)
    server = ThreadingHTTPServer((host, port), handler_cls)
    return server
