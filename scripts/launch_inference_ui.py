"""Launch the OpenCode-Style Interactive Inference & Testing Frontend for `MCL4-COE`.

Usage:
    py -3.11 scripts/launch_inference_ui.py --open-browser
    py -3.11 scripts/launch_inference_ui.py --checkpoint colab_mcl4_coe_v7_mitotic_3b_fast/step_05800 --open-browser
    py -3.11 scripts/launch_inference_ui.py --preset v7_balanced_768_e8 --open-browser
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import webbrowser

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcl4_coe.inference_ui import InferenceSessionController, start_inference_ui_server


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the OpenCode-style MCL4-COE Inference & Testing Frontend."
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7861, help="Bind port (default: 7861)")
    parser.add_argument(
        "--runs-root",
        type=str,
        default=str(REPO_ROOT / "runs"),
        help="Root directory containing run folders and checkpoint receipts",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="mesmac/mcl4-coe-runs",
        help="Hugging Face Hub repository for on-demand checkpoint weight downloads",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default="",
        help="Optional Hugging Face token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional checkpoint_id or .pt/.meta.json path to load on startup",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="",
        choices=["", "tiny_morph16k", "v6_mitotic_e8", "v7_balanced_768_e8"],
        help="Optional architecture preset to initialize on startup",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Target device for model execution",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model floating-point precision",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Automatically open the UI in the default web browser",
    )
    args = parser.parse_args()

    controller = InferenceSessionController(
        runs_root=args.runs_root,
        hf_repo=args.hf_repo,
    )

    if args.checkpoint:
        print(f"[opencode // mcl4-coe] Loading checkpoint: {args.checkpoint} ...")
        controller.load_checkpoint(
            {
                "checkpoint": args.checkpoint,
                "hf_repo": args.hf_repo,
                "hf_token": args.hf_token,
                "device": args.device,
                "dtype": args.dtype,
            }
        )
    elif args.preset:
        print(f"[opencode // mcl4-coe] Initializing preset: {args.preset} ...")
        controller.load_preset(
            {
                "preset": args.preset,
                "device": args.device,
                "dtype": args.dtype,
            }
        )

    server = start_inference_ui_server(controller, host=args.host, port=args.port)
    url = f"http://{args.host}:{args.port}"
    print(f"[opencode // mcl4-coe] Inference terminal running at {url}")
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[opencode // mcl4-coe] Shutting down server.")
        server.server_close()


if __name__ == "__main__":
    main()
