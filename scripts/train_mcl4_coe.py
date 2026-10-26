"""Full-Scale `MCL4-COE` Training Runner & Lightweight Web Studio Launcher.

Usage Examples:
1. Launch the interactive Web UI Dashboard on `http://127.0.0.1:7860`:
   python scripts/train_mcl4_coe.py --serve --port 7860

2. Launch the Web UI Dashboard AND immediately start a training run in the background:
   python scripts/train_mcl4_coe.py --serve --auto-start --run-id run_stage2_warm --steps 40

3. Run a headless training job (populates `X:\\MCL4-COE\\runs\\<run_id>` for later review in the UI):
   python scripts/train_mcl4_coe.py --headless --run-id run_bf16_baseline --steps 15 --checkpoint-every 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import webbrowser

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcl4_coe import (
    MCL4COETrainer,
    TrainingControllerState,
    TrainingRunConfig,
    start_web_ui_server,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="MCL4-COE Full-Scale Training Runner & Web UI.")
    parser.add_argument("--runs-root", type=str, default=r"X:\MCL4-COE\runs", help="Directory for run histories and checkpoints")
    parser.add_argument("--shards-root", type=str, default=r"X:\MCL4-COE\data_shards", help="Directory containing sharded .bin + index_receipt.json tracks")
    parser.add_argument("--corpora-root", type=str, default=r"X:\corpora", help="Root corpora directory")

    parser.add_argument("--run-id", type=str, default="run_stage2_coe", help="Run identifier")
    parser.add_argument("--steps", type=int, default=20, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Global batch size in blocks")
    parser.add_argument("--micro-batch-size", type=int, default=4, help="Micro-batch size per forward/backward")
    parser.add_argument("--lr", type=float, default=2.5e-3, help="Peak learning rate")
    parser.add_argument("--d-model", type=int, default=128, help="Backbone model dimension (e.g. 128, 192, 256)")
    parser.add_argument("--layers", type=int, default=2, help="Number of encoder and decoder layers each (e.g. 2, 3, 4)")
    parser.add_argument("--num-experts", type=int, default=4, help="Number of Sparse MoE experts (e.g. 4, 8, 16)")
    parser.add_argument("--top-k", type=int, default=2, help="Number of active experts routed per token")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Checkpoint interval in steps")
    parser.add_argument("--sample-every", type=int, default=5, help="Sample #U$#->#T$#->#A$# interval in steps")
    parser.add_argument("--use-nvfp4", action="store_true", help="Enable native Blackwell sm_120 NVFP4 linear layers")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Optional checkpoint .meta.json or .pt path to resume from")

    parser.add_argument("--serve", action="store_true", help="Launch the lightweight Web UI server")
    parser.add_argument("--headless", action="store_true", help="Run training in CLI mode without starting the Web server")
    parser.add_argument("--auto-start", action="store_true", help="Automatically start the configured run when launching with --serve")
    parser.add_argument("--open-browser", action="store_true", help="Automatically open the Web UI in the default browser")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web UI bind host")
    parser.add_argument("--port", type=int, default=7860, help="Web UI bind port")
    args = parser.parse_args()

    cfg = TrainingRunConfig(
        run_id=args.run_id,
        runs_root=args.runs_root,
        shards_root=args.shards_root,
        corpora_root=args.corpora_root,
        resume_checkpoint=args.resume_checkpoint,
        total_steps=args.steps,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        lr=args.lr,
        d_model=args.d_model,
        n_heads=8 if args.d_model >= 256 else 4,
        n_encoder_layers=args.layers,
        n_decoder_layers=args.layers,
        expert_hidden_dim=args.d_model * 2,
        ple_dim=max(32, args.d_model // 4),
        num_experts=args.num_experts,
        top_k=args.top_k,
        checkpoint_every=args.checkpoint_every,
        sample_every=args.sample_every,
        use_nvfp4=args.use_nvfp4,
    )

    if args.headless and not args.serve:
        print(f"=== Starting Headless MCL4-COE Training Run: {cfg.run_id} ({cfg.total_steps} steps, {cfg.num_experts}E top-{cfg.top_k}) ===")

        def _on_step(rec: dict) -> None:
            print(
                f"  [step {rec['step']:>4}/{rec['total_steps']}] "
                f"total_loss={rec['total_loss']:.4f} | lm={rec['lm_loss']:.4f} | dec={rec['decision_loss']:.4f} | "
                f"speed={rec['tok_per_s']:,.0f} tok/s | experts={rec.get('num_experts', cfg.num_experts)}E"
            )

        trainer = MCL4COETrainer(cfg, on_step_callback=_on_step)
        final_metrics = trainer.train()
        print(f"=== Completed Run {cfg.run_id} | Final Loss: {final_metrics.get('total_loss')} ===")
        print(f"Run artifacts & checkpoints saved to: {Path(cfg.runs_root) / cfg.run_id}")
        return

    controller = TrainingControllerState(
        runs_root=args.runs_root,
        shards_root=args.shards_root,
        corpora_root=args.corpora_root,
    )
    if args.auto_start:
        controller.start_run(cfg.to_dict())

    server = start_web_ui_server(controller, host=args.host, port=args.port)
    url = f"http://{args.host}:{args.port}"
    print(f"=== MCL4-COE Training & Checkpoint Studio Web UI running at {url} ===")
    print(f"    Runs & Checkpoints Directory: {Path(args.runs_root).resolve()}")
    print(f"    Sharded Data Directory:       {Path(args.shards_root).resolve()}")
    print("    Press Ctrl+C to stop the server.")
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Web UI server...")
        controller.stop_run()
        server.server_close()


if __name__ == "__main__":
    main()
