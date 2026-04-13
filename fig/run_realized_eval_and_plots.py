from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = Path(__file__).resolve().parent
DREAMER_DIR = ROOT / "dreamer4-main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run realized-time evaluation and plotting pipeline for AtomWorld-Twins"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--print-samples", type=int, default=5)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--title-name", type=str, default="AtomWorld-Twins")
    parser.add_argument("--model-label", type=str, default="AtomWorld-Twins")
    parser.add_argument("--ppo-label", type=str, default="SwarmThinkers PPO")
    parser.add_argument(
        "--ppo-eval",
        type=str,
        default=str(ROOT / "results" / "ppo_v9_results" / "ppo_macro_eval_val.json"),
    )
    parser.add_argument("--eval-output", type=str, default=None)
    parser.add_argument(
        "--comparison-eval-output",
        type=str,
        default=str(FIG_DIR / "macro_edit_eval_comparison.png"),
    )
    parser.add_argument(
        "--comparison-time-output",
        type=str,
        default=str(FIG_DIR / "macro_edit_time_alignment.png"),
    )
    parser.add_argument(
        "--realized-output",
        type=str,
        default=str(FIG_DIR / "macro_edit_realized_time.png"),
    )
    return parser.parse_args()


def run_step(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    cache = Path(args.cache).resolve()
    model_dir = Path(args.model_dir).resolve() if args.model_dir else checkpoint.parent.resolve()
    eval_output = Path(args.eval_output).resolve() if args.eval_output else (model_dir / "eval_time_alignment_realized.json").resolve()

    eval_command = [
        sys.executable,
        str(DREAMER_DIR / "eval_macro_time_alignment.py"),
        "--checkpoint",
        str(checkpoint),
        "--cache",
        str(cache),
        "--split",
        args.split,
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--print_samples",
        str(args.print_samples),
        "--output",
        str(eval_output),
        "--save_all_samples",
    ]
    if args.limit > 0:
        eval_command.extend(["--limit", str(args.limit)])

    comparison_command = [
        sys.executable,
        str(FIG_DIR / "plot_atomworld_twins_comparison.py"),
        "--model-eval",
        str(eval_output),
        "--model-dir",
        str(model_dir),
        "--ppo-eval",
        args.ppo_eval,
        "--title-name",
        args.title_name,
        "--model-label",
        args.model_label,
        "--ppo-label",
        args.ppo_label,
        "--eval-output",
        str(Path(args.comparison_eval_output).resolve()),
        "--time-output",
        str(Path(args.comparison_time_output).resolve()),
    ]

    realized_command = [
        sys.executable,
        str(FIG_DIR / "plot_realized_time_diagnostics.py"),
        "--model-eval",
        str(eval_output),
        "--model-dir",
        str(model_dir),
        "--title-name",
        args.title_name,
        "--model-label",
        args.model_label,
        "--output",
        str(Path(args.realized_output).resolve()),
    ]

    run_step(eval_command)
    run_step(comparison_command)
    run_step(realized_command)

    print("Finished realized-time evaluation pipeline")
    print("Eval JSON:", eval_output)
    print("Comparison figure:", Path(args.comparison_eval_output).resolve())
    print("Time alignment figure:", Path(args.comparison_time_output).resolve())
    print("Realized-time figure:", Path(args.realized_output).resolve())


if __name__ == "__main__":
    main()