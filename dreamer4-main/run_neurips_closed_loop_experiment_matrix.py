from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT_ROOT = ROOT / "results" / "neurips_closed_loop_matrix"
DEFAULT_FIXEDK_ROOT = ROOT / "results" / "neurips_fixedk_matrix"


@dataclass(frozen=True)
class ClosedLoopRun:
    name: str
    group: str
    checkpoint: str
    constraint_mode: str = "full"
    segments: int = 200
    seed: int = 0
    save_snapshots: bool = False
    snapshot_every: int = 25
    notes: str = ""


def _runs(fixedk_root: Path, segments: int) -> list[ClosedLoopRun]:
    return [
        ClosedLoopRun(
            name="full_seed0",
            group="full",
            checkpoint=str(fixedk_root / "full_seed0" / "best_model.pt"),
            segments=segments,
            seed=0,
            save_snapshots=True,
            notes="Primary full model with snapshots for structure-evolution visualization.",
        ),
        ClosedLoopRun(
            name="full_seed1",
            group="full",
            checkpoint=str(fixedk_root / "full_seed1" / "best_model.pt"),
            segments=segments,
            seed=1,
        ),
        ClosedLoopRun(
            name="full_seed2",
            group="full",
            checkpoint=str(fixedk_root / "full_seed2" / "best_model.pt"),
            segments=segments,
            seed=2,
        ),
        ClosedLoopRun(
            name="strict_full_seed0",
            group="strict_constraints",
            checkpoint=str(fixedk_root / "full_seed0" / "best_model.pt"),
            constraint_mode="full",
            segments=segments,
            seed=0,
            save_snapshots=True,
            notes="Strict inference constraint baseline: all three constraints enabled.",
        ),
        ClosedLoopRun(
            name="strict_no_inventory_seed0",
            group="strict_constraints",
            checkpoint=str(fixedk_root / "full_seed0" / "best_model.pt"),
            constraint_mode="no_inventory",
            segments=segments,
            seed=0,
            notes="Inference-time ablation: raw edits on reachable support, no paired inventory projection.",
        ),
        ClosedLoopRun(
            name="strict_no_reachability_seed0",
            group="strict_constraints",
            checkpoint=str(fixedk_root / "full_seed0" / "best_model.pt"),
            constraint_mode="no_reachability",
            segments=segments,
            seed=0,
            notes="Inference-time ablation: unrestricted global candidate support, raw edits.",
        ),
        ClosedLoopRun(
            name="strict_no_continuous_time_seed0",
            group="strict_constraints",
            checkpoint=str(fixedk_root / "full_seed0" / "best_model.pt"),
            constraint_mode="no_continuous_time",
            segments=segments,
            seed=0,
            notes="Inference-time ablation: learned duration replaced by CTMC start-state baseline.",
        ),
        ClosedLoopRun(
            name="strict_no_constraints_seed0",
            group="strict_constraints",
            checkpoint=str(fixedk_root / "full_seed0" / "best_model.pt"),
            constraint_mode="no_constraints",
            segments=segments,
            seed=0,
            notes="Inference-time ablation: no reachable support, no inventory projection, no learned duration.",
        ),
        ClosedLoopRun(
            name="baseline_no_change_seed0",
            group="baseline",
            checkpoint=str(fixedk_root / "full_seed0" / "best_model.pt"),
            constraint_mode="no_change",
            segments=segments,
            seed=0,
            notes="Simple copy-state baseline with CTMC start-state duration.",
        ),
        ClosedLoopRun(
            name="abl_no_duration_seed0",
            group="trained_ablation",
            checkpoint=str(fixedk_root / "abl_no_duration_seed0" / "best_model.pt"),
            segments=segments,
            seed=0,
            notes="Train-time ablation: no duration supervision.",
        ),
        ClosedLoopRun(
            name="abl_no_tau_exp_seed0",
            group="trained_ablation",
            checkpoint=str(fixedk_root / "abl_no_tau_exp_seed0" / "best_model.pt"),
            segments=segments,
            seed=0,
            notes="Train-time ablation: no expected-time main target.",
        ),
        ClosedLoopRun(
            name="abl_no_realized_aux_seed0",
            group="trained_ablation",
            checkpoint=str(fixedk_root / "abl_no_realized_aux_seed0" / "best_model.pt"),
            segments=segments,
            seed=0,
            notes="Train-time ablation: no realized-time auxiliary head loss.",
        ),
        ClosedLoopRun(
            name="abl_no_future_candidate_aug_seed0",
            group="trained_ablation",
            checkpoint=str(fixedk_root / "abl_no_future_candidate_aug_seed0" / "best_model.pt"),
            segments=segments,
            seed=0,
            notes="Train-time ablation/proxy: candidate support misses teacher path augmentation.",
        ),
        ClosedLoopRun(
            name="abl_no_proj_loss_seed0",
            group="trained_ablation",
            checkpoint=str(fixedk_root / "abl_no_proj_loss_seed0" / "best_model.pt"),
            segments=segments,
            seed=0,
            notes="Train-time ablation: no projected-state consistency loss.",
        ),
        ClosedLoopRun(
            name="abl_no_prior_rollout_seed0",
            group="trained_ablation",
            checkpoint=str(fixedk_root / "abl_no_prior_rollout_seed0" / "best_model.pt"),
            segments=segments,
            seed=0,
            notes="Train-time ablation: no prior-side rollout closure.",
        ),
        ClosedLoopRun(
            name="abl_posterior_only_tau_seed0",
            group="trained_ablation",
            checkpoint=str(fixedk_root / "abl_posterior_only_tau_seed0" / "best_model.pt"),
            segments=segments,
            seed=0,
            notes="Train-time ablation: posterior-only duration pathway.",
        ),
        ClosedLoopRun(
            name="diag_k2_seed0",
            group="macro_efficiency",
            checkpoint=str(fixedk_root / "diag_k2_seed0" / "best_model.pt"),
            segments=segments,
            seed=0,
            notes="Fixed-k=2 diagnostic for macro-step efficiency/performance tradeoff.",
        ),
        ClosedLoopRun(
            name="diag_k8_seed0",
            group="macro_efficiency",
            checkpoint=str(fixedk_root / "diag_k8_seed0" / "best_model.pt"),
            segments=segments,
            seed=0,
            notes="Fixed-k=8 diagnostic for macro-step efficiency/performance tradeoff.",
        ),
        ClosedLoopRun(
            name="diag_multik_248_seed0",
            group="macro_efficiency",
            checkpoint=str(fixedk_root / "diag_multik_248_seed0" / "best_model.pt"),
            segments=segments,
            seed=0,
            notes="Optional multi-k diagnostic, not promoted as solved planner unless closed-loop metrics support it.",
        ),
    ]


def _select_shard(runs: list[ClosedLoopRun], shard: str) -> list[ClosedLoopRun]:
    if shard == "all":
        return runs
    if shard == "full":
        return [run for run in runs if run.group == "full"]
    if shard == "strict":
        return [run for run in runs if run.group in {"strict_constraints", "baseline"}]
    if shard == "ablation_a":
        names = {"abl_no_duration_seed0", "abl_no_tau_exp_seed0", "abl_no_realized_aux_seed0", "abl_no_future_candidate_aug_seed0"}
        return [run for run in runs if run.name in names]
    if shard == "ablation_b":
        names = {"abl_no_proj_loss_seed0", "abl_no_prior_rollout_seed0", "abl_posterior_only_tau_seed0"}
        return [run for run in runs if run.name in names]
    if shard == "efficiency":
        return [run for run in runs if run.group == "macro_efficiency"]
    raise ValueError(f"Unknown shard: {shard}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the NeurIPS closed-loop autonomous rollout matrix")
    parser.add_argument("--result_root", type=str, default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--fixedk_root", type=str, default=str(DEFAULT_FIXEDK_ROOT))
    parser.add_argument("--segments", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--shard", type=str, default="all", choices=["all", "full", "strict", "ablation_a", "ablation_b", "efficiency"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--max_episode_steps_override", type=int, default=5000)
    parser.add_argument(
        "--reference_mode",
        type=str,
        default="independent_teacher",
        choices=["independent_teacher", "on_policy_teacher_probe"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_root)
    fixedk_root = Path(args.fixedk_root)
    result_root.mkdir(parents=True, exist_ok=True)
    runs = _select_shard(_runs(fixedk_root, int(args.segments)), args.shard)
    manifest_path = result_root / f"manifest_{args.shard}.json"
    manifest_path.write_text(
        json.dumps(
            {
                "shard": args.shard,
                "segments": int(args.segments),
                "device": args.device,
                "reference_mode": args.reference_mode,
                "fixedk_root": str(fixedk_root),
                "runs": [asdict(run) for run in runs],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    for index, run in enumerate(runs, start=1):
        run_dir = result_root / run.name
        run_dir.mkdir(parents=True, exist_ok=True)
        output = run_dir / "eval_closed_loop.json"
        if output.exists() and not args.overwrite:
            print(f"[{index}/{len(runs)}] skip existing {run.name}: {output}", flush=True)
            continue
        cmd = [
            sys.executable,
            str(ROOT / "eval_macro_closed_loop_rollout.py"),
            "--checkpoint",
            run.checkpoint,
            "--output",
            str(output),
            "--rollout_segments",
            str(run.segments),
            "--seed",
            str(run.seed),
            "--device",
            str(args.device),
            "--constraint_mode",
            run.constraint_mode,
            "--reference_mode",
            str(args.reference_mode),
            "--progress_every",
            str(args.progress_every),
            "--print_segments",
            "5",
        ]
        if args.max_episode_steps_override is not None:
            cmd.extend(["--max_episode_steps_override", str(args.max_episode_steps_override)])
        if run.save_snapshots:
            cmd.extend(["--save_snapshots", "--snapshot_every", str(run.snapshot_every)])
        print(f"[{index}/{len(runs)}] run {run.name}", flush=True)
        print(" ".join(cmd), flush=True)
        with (run_dir / "closed_loop.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            print(f"[{index}/{len(runs)}] FAILED {run.name} returncode={proc.returncode}", flush=True)
            raise SystemExit(proc.returncode)
        print(f"[{index}/{len(runs)}] done {run.name}", flush=True)
    print(f"Closed-loop shard complete: {args.shard}", flush=True)


if __name__ == "__main__":
    main()
