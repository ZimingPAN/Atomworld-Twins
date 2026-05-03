from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = Path("results/neurips_fixedk_matrix")
PYTHON = "/home/likun/panziming/AtomWorld-Twins/conda/bin/python3"
PYTHONPATH = (
    "/home/likun/panziming/pydeps:"
    "/home/likun/panziming/AtomWorld-Twins/kmcteacher_backend:"
    "/home/likun/panziming/AtomWorld-Twins/LightZero-main:"
    "/home/likun/panziming/AtomWorld-Twins/dreamer4-main"
)


@dataclass
class RunSpec:
    name: str
    group: str
    seed: int = 0
    segment_k: int = 4
    segment_ks: list[int] | None = None
    train_segments: int = 2000
    val_segments: int = 400
    epochs: int = 80
    batch_size: int = 32
    max_candidate_sites: int = 128
    max_seed_vacancies: int = 8
    teacher_path_summary_mode: str = "stepwise"
    tau_supervision_mode: str = "prior_main"
    tau_weight: float = 1.0
    realized_tau_weight: float = 0.25
    proj_weight: float = 0.5
    path_weight: float = 0.05
    prior_edit_weight: float = 0.25
    prior_latent_weight: float = 0.25
    prior_reward_weight: float = 0.5
    reward_weight: float = 0.5
    reward_prediction_source: str = "raw"
    cu_density: float = 0.0134
    v_density: float = 0.0002
    disable_teacher_candidate_augmentation: bool = False
    long_segments: int = 200
    long_max_episode_steps: int = 2000
    eval_only_checkpoint: str | None = None
    notes: str = ""
    extra_train_args: list[str] = field(default_factory=list)

    @property
    def run_dir(self) -> Path:
        return RESULT_ROOT / self.name

    @property
    def cache_path(self) -> Path:
        horizon_tag = "k" + "-".join(str(k) for k in self.segment_ks) if self.segment_ks else f"k{self.segment_k}"
        cache_name = (
            f"seed{self.seed}_{horizon_tag}_c{self.max_candidate_sites}_"
            f"sv{self.max_seed_vacancies}_cu{self.cu_density:g}_"
            f"{self.teacher_path_summary_mode}"
        )
        if self.disable_teacher_candidate_augmentation:
            cache_name += "_noaug"
        return RESULT_ROOT / "caches" / f"{cache_name}.pt"


def _base_matrix(args: argparse.Namespace) -> list[RunSpec]:
    full_epochs = int(args.full_epochs)
    ablation_epochs = int(args.ablation_epochs)
    train_segments = int(args.train_segments)
    val_segments = int(args.val_segments)
    specs: list[RunSpec] = []

    for seed in args.full_seeds:
        specs.append(
            RunSpec(
                name=f"full_seed{seed}",
                group="full",
                seed=seed,
                epochs=full_epochs,
                train_segments=train_segments,
                val_segments=val_segments,
                long_segments=args.long_segments_full,
                notes="Full fixed-k AtomWorld-Twins run.",
            )
        )

    specs.extend(
        [
            RunSpec(
                name="abl_legacy_path_seed0",
                group="ablation",
                teacher_path_summary_mode="legacy",
                epochs=ablation_epochs,
                train_segments=train_segments,
                val_segments=val_segments,
                notes="Ablation: legacy compressed teacher path summary.",
            ),
            RunSpec(
                name="abl_posterior_only_tau_seed0",
                group="ablation",
                tau_supervision_mode="posterior_only",
                epochs=ablation_epochs,
                train_segments=train_segments,
                val_segments=val_segments,
                notes="Ablation: posterior-only tau supervision.",
            ),
            RunSpec(
                name="abl_no_tau_exp_seed0",
                group="ablation",
                tau_weight=0.0,
                epochs=ablation_epochs,
                train_segments=train_segments,
                val_segments=val_segments,
                notes="Ablation: remove main tau_exp loss.",
            ),
            RunSpec(
                name="abl_no_duration_seed0",
                group="ablation",
                tau_weight=0.0,
                realized_tau_weight=0.0,
                epochs=ablation_epochs,
                train_segments=train_segments,
                val_segments=val_segments,
                notes="Ablation: remove both tau_exp and realized-time duration losses.",
            ),
            RunSpec(
                name="abl_no_realized_aux_seed0",
                group="ablation",
                realized_tau_weight=0.0,
                epochs=ablation_epochs,
                train_segments=train_segments,
                val_segments=val_segments,
                notes="Ablation: remove realized-time auxiliary distribution loss.",
            ),
            RunSpec(
                name="abl_no_proj_loss_seed0",
                group="ablation",
                proj_weight=0.0,
                epochs=ablation_epochs,
                train_segments=train_segments,
                val_segments=val_segments,
                notes="Ablation: remove projected-state consistency loss.",
            ),
            RunSpec(
                name="abl_no_prior_edit_seed0",
                group="ablation",
                prior_edit_weight=0.0,
                epochs=ablation_epochs,
                train_segments=train_segments,
                val_segments=val_segments,
                notes="Ablation: remove prior-side sparse edit loss.",
            ),
            RunSpec(
                name="abl_no_prior_latent_seed0",
                group="ablation",
                prior_latent_weight=0.0,
                epochs=ablation_epochs,
                train_segments=train_segments,
                val_segments=val_segments,
                notes="Ablation: remove prior-side latent/projection loss.",
            ),
            RunSpec(
                name="abl_no_path_kl_seed0",
                group="ablation",
                path_weight=0.0,
                epochs=ablation_epochs,
                train_segments=train_segments,
                val_segments=val_segments,
                notes="Ablation: remove posterior-prior path KL loss.",
            ),
            RunSpec(
                name="abl_no_prior_rollout_seed0",
                group="ablation",
                tau_supervision_mode="posterior_only",
                path_weight=0.0,
                prior_edit_weight=0.0,
                prior_latent_weight=0.0,
                prior_reward_weight=0.0,
                epochs=ablation_epochs,
                train_segments=train_segments,
                val_segments=val_segments,
                notes="Ablation: remove prior-side edit/latent/reward and path KL losses.",
            ),
            RunSpec(
                name="abl_no_future_candidate_aug_seed0",
                group="ablation",
                disable_teacher_candidate_augmentation=True,
                max_candidate_sites=1024,
                max_seed_vacancies=16,
                batch_size=4,
                epochs=ablation_epochs,
                train_segments=max(1000, train_segments // 2),
                val_segments=max(200, val_segments // 2),
                notes="Ablation: train without future teacher candidate augmentation.",
            ),
        ]
    )

    specs.extend(
        [
            RunSpec(
                name="sens_candidate64_seed0",
                group="sensitivity",
                max_candidate_sites=64,
                epochs=ablation_epochs,
                train_segments=train_segments,
                val_segments=val_segments,
                notes="Candidate support sensitivity: 64 sites.",
            ),
            RunSpec(
                name="sens_candidate256_seed0",
                group="sensitivity",
                max_candidate_sites=256,
                epochs=ablation_epochs,
                train_segments=train_segments,
                val_segments=val_segments,
                notes="Candidate support sensitivity: 256 sites.",
            ),
            RunSpec(
                name="sens_seedvac16_seed0",
                group="sensitivity",
                max_seed_vacancies=16,
                epochs=ablation_epochs,
                train_segments=train_segments,
                val_segments=val_segments,
                notes="Seed-vacancy sensitivity: 16 seed vacancies.",
            ),
        ]
    )

    if args.include_k_diagnostics:
        specs.extend(
            [
                RunSpec(
                    name="diag_k2_seed0",
                    group="k_diagnostic",
                    segment_k=2,
                    epochs=ablation_epochs,
                    train_segments=train_segments,
                    val_segments=val_segments,
                    long_segments=args.long_segments_ablation,
                    notes="Fixed-k diagnostic: k=2.",
                ),
                RunSpec(
                    name="diag_k8_seed0",
                    group="k_diagnostic",
                    segment_k=8,
                    max_candidate_sites=256,
                    epochs=ablation_epochs,
                    train_segments=train_segments,
                    val_segments=val_segments,
                    long_segments=args.long_segments_ablation,
                    notes="Fixed-k diagnostic: k=8.",
                ),
                RunSpec(
                    name="diag_multik_248_seed0",
                    group="multi_k_optional",
                    segment_k=8,
                    segment_ks=[2, 4, 8],
                    max_candidate_sites=256,
                    epochs=ablation_epochs,
                    train_segments=train_segments,
                    val_segments=val_segments,
                    long_segments=args.long_segments_ablation,
                    notes="Optional multi-k diagnostic only; not a main fixed-k result.",
                ),
            ]
        )

    if args.include_ood_eval:
        specs.append(
            RunSpec(
                name="ood_low_density_eval_full_seed0",
                group="ood_eval",
                seed=100,
                train_segments=1,
                val_segments=val_segments,
                epochs=1,
                cu_density=0.005,
                eval_only_checkpoint=str(RESULT_ROOT / "full_seed0" / "final_model.pt"),
                long_segments=0,
                notes="Low-density OOD paired evaluation using full_seed0 checkpoint.",
            )
        )
    return specs


def _train_args(spec: RunSpec, gpu: str) -> list[str]:
    args = [
        PYTHON,
        "train_dreamer_macro_edit.py",
        "--save_dir",
        str(spec.run_dir),
        "--dataset_cache",
        str(spec.cache_path),
        "--seed",
        str(spec.seed),
        "--teacher_path_summary_mode",
        spec.teacher_path_summary_mode,
        "--tau_supervision_mode",
        spec.tau_supervision_mode,
        "--tau_weight",
        str(spec.tau_weight),
        "--realized_tau_weight",
        str(spec.realized_tau_weight),
        "--proj_weight",
        str(spec.proj_weight),
        "--path_weight",
        str(spec.path_weight),
        "--prior_edit_weight",
        str(spec.prior_edit_weight),
        "--prior_latent_weight",
        str(spec.prior_latent_weight),
        "--prior_reward_weight",
        str(spec.prior_reward_weight),
        "--reward_weight",
        str(spec.reward_weight),
        "--reward_prediction_source",
        spec.reward_prediction_source,
        "--train_segments",
        str(spec.train_segments),
        "--val_segments",
        str(spec.val_segments),
        "--epochs",
        str(spec.epochs),
        "--batch_size",
        str(spec.batch_size),
        "--lr",
        "1e-4",
        "--eval_freq",
        "5",
        "--save_freq",
        "20",
        "--max_candidate_sites",
        str(spec.max_candidate_sites),
        "--max_seed_vacancies",
        str(spec.max_seed_vacancies),
        "--cu_density",
        str(spec.cu_density),
        "--v_density",
        str(spec.v_density),
        "--device",
        f"cuda:{gpu}",
    ]
    if spec.segment_ks:
        args.extend(["--segment_ks", *[str(item) for item in spec.segment_ks]])
    else:
        args.extend(["--segment_k", str(spec.segment_k)])
    if spec.disable_teacher_candidate_augmentation:
        args.append("--disable_teacher_candidate_augmentation")
    if spec.eval_only_checkpoint:
        args.extend(["--eval_only", "--resume", spec.eval_only_checkpoint])
    args.extend(spec.extra_train_args)
    return args


def _eval_args(spec: RunSpec, gpu: str, output_name: str = "eval_time_alignment.json") -> list[str]:
    checkpoint = spec.eval_only_checkpoint or str(spec.run_dir / "final_model.pt")
    return [
        PYTHON,
        "eval_macro_time_alignment.py",
        "--checkpoint",
        checkpoint,
        "--cache",
        str(spec.cache_path),
        "--split",
        "val",
        "--output",
        str(spec.run_dir / output_name),
        "--save_all_samples",
        "--device",
        f"cuda:{gpu}",
    ]


def _long_args(spec: RunSpec, gpu: str) -> list[str]:
    return [
        PYTHON,
        "eval_macro_long_trajectory.py",
        "--checkpoint",
        str(spec.run_dir / "final_model.pt"),
        "--rollout_segments",
        str(spec.long_segments),
        "--max_episode_steps_override",
        str(spec.long_max_episode_steps),
        "--output",
        str(spec.run_dir / f"eval_long_trajectory_{spec.long_segments}.json"),
        "--device",
        f"cuda:{gpu}",
    ]


def _ppo_eval_args(cache_path: Path, gpu: str) -> list[str]:
    return [
        PYTHON,
        "../eval_ppo_macro_segments.py",
        "--checkpoint",
        "../results/ppo_v9_results/best_model.pt",
        "--config",
        "../results/ppo_v9_results/config.json",
        "--cache",
        str(cache_path),
        "--split",
        "val",
        "--output",
        str(RESULT_ROOT / "baselines" / "ppo_v9_eval_val.json"),
        "--save_all_samples",
        "--device",
        f"cuda:{gpu}",
    ]


def _shell_command(cmd: list[str], log_path: Path) -> str:
    quoted = shlex.join(cmd)
    return f"{quoted} 2>&1 | tee {shlex.quote(str(log_path))}\nstatus=${{PIPESTATUS[0]}}\nif [ \"$status\" -ne 0 ]; then echo FAILED:$status; exit $status; fi"


def _write_run_config(spec: RunSpec) -> None:
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(spec)
    payload["run_dir"] = str(spec.run_dir)
    payload["cache_path"] = str(spec.cache_path)
    (spec.run_dir / "run_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_queues(specs: list[RunSpec], gpus: list[str], session_prefix: str, include_monitor: bool) -> list[Path]:
    out_dir = ROOT / RESULT_ROOT / "queues"
    out_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / RESULT_ROOT / "caches").mkdir(parents=True, exist_ok=True)
    queue_paths: list[Path] = []
    queues: dict[str, list[RunSpec]] = {gpu: [] for gpu in gpus}
    cache_owner: dict[str, str] = {}
    for idx, spec in enumerate(specs):
        if spec.group == "ood_eval":
            queues[gpus[0]].append(spec)
        elif str(spec.cache_path) in cache_owner:
            queues[cache_owner[str(spec.cache_path)]].append(spec)
        else:
            gpu = min(gpus, key=lambda item: len(queues[item]))
            cache_owner[str(spec.cache_path)] = gpu
            queues[gpu].append(spec)

    for gpu, gpu_specs in queues.items():
        path = out_dir / f"{session_prefix}_gpu{gpu}.sh"
        lines = [
            "#!/usr/bin/env bash",
            "set -u",
            f"export PYTHONPATH={shlex.quote(PYTHONPATH)}",
            "export PYTHONUNBUFFERED=1",
            "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            f"cd {shlex.quote(str(ROOT))}",
            f"echo queue_start gpu={gpu} date=$(date -Is)",
            "echo 'Stop a specific process with: kill <PID>. Do not use pkill or killall.'",
        ]
        for spec in gpu_specs:
            _write_run_config(spec)
            lines.extend(
                [
                    f"echo RUN_START {shlex.quote(spec.name)} date=$(date -Is)",
                    f"mkdir -p {shlex.quote(str(spec.run_dir))}",
                    _shell_command(_train_args(spec, gpu), spec.run_dir / "train.log"),
                    _shell_command(_eval_args(spec, gpu), spec.run_dir / "eval_time_alignment.log"),
                ]
            )
            if spec.long_segments > 0 and not spec.eval_only_checkpoint:
                lines.append(_shell_command(_long_args(spec, gpu), spec.run_dir / "eval_long.log"))
            if spec.name == "full_seed0":
                baseline_dir = RESULT_ROOT / "baselines"
                lines.append(f"mkdir -p {shlex.quote(str(baseline_dir))}")
                lines.append(_shell_command(_ppo_eval_args(spec.cache_path, gpu), baseline_dir / "ppo_v9_eval.log"))
            lines.append(f"echo RUN_DONE {shlex.quote(spec.name)} date=$(date -Is)")
        lines.append(f"echo queue_done gpu={gpu} date=$(date -Is)")
        path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
        path.chmod(0o755)
        queue_paths.append(path)

    if include_monitor:
        monitor = out_dir / f"{session_prefix}_monitor.sh"
        session_names = [f"{session_prefix}_gpu{gpu}" for gpu in gpus]
        monitor_lines = [
            "#!/usr/bin/env bash",
            "set -u",
            f"export PYTHONPATH={shlex.quote(PYTHONPATH)}",
            "export PYTHONUNBUFFERED=1",
            "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            f"cd {shlex.quote(str(ROOT))}",
            "echo monitor_start date=$(date -Is)",
            "while true; do",
            "  alive=0",
        ]
        for name in session_names:
            monitor_lines.append(f"  if tmux has-session -t {shlex.quote(name)} 2>/dev/null; then alive=1; fi")
        monitor_lines.extend(
            [
                "  if [ \"$alive\" -eq 0 ]; then break; fi",
                "  sleep 300",
                "done",
                f"{shlex.quote(PYTHON)} review_neurips_fixedk_experiment_matrix.py --result_root {shlex.quote(str(RESULT_ROOT))} 2>&1 | tee {shlex.quote(str(RESULT_ROOT / 'review.log'))}",
                "if [ -s results/neurips_fixedk_matrix/rerun_commands.sh ]; then",
                "  echo rerun_start date=$(date -Is)",
                "  bash results/neurips_fixedk_matrix/rerun_commands.sh 2>&1 | tee results/neurips_fixedk_matrix/rerun.log",
                f"  {shlex.quote(PYTHON)} review_neurips_fixedk_experiment_matrix.py --result_root {shlex.quote(str(RESULT_ROOT))} 2>&1 | tee results/neurips_fixedk_matrix/review_after_rerun.log",
                "fi",
                "echo monitor_done date=$(date -Is)",
            ]
        )
        monitor.write_text("\n".join(monitor_lines) + "\n", encoding="utf-8")
        monitor.chmod(0o755)
        queue_paths.append(monitor)
    return queue_paths


def launch_queues(queue_paths: list[Path], gpus: list[str], session_prefix: str, include_monitor: bool) -> None:
    for gpu in gpus:
        path = ROOT / RESULT_ROOT / "queues" / f"{session_prefix}_gpu{gpu}.sh"
        session = f"{session_prefix}_gpu{gpu}"
        subprocess.run(["tmux", "new-session", "-d", "-s", session, "bash", str(path)], check=True)
        print(f"launched {session}: {path}")
    if include_monitor:
        monitor_path = ROOT / RESULT_ROOT / "queues" / f"{session_prefix}_monitor.sh"
        monitor_session = f"{session_prefix}_monitor"
        subprocess.run(["tmux", "new-session", "-d", "-s", monitor_session, "bash", str(monitor_path)], check=True)
        print(f"launched {monitor_session}: {monitor_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch NeurIPS fixed-k experiment matrix.")
    parser.add_argument("--gpus", type=str, default="3,4,5")
    parser.add_argument("--session_prefix", type=str, default="nfx")
    parser.add_argument("--full_seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--full_epochs", type=int, default=120)
    parser.add_argument("--ablation_epochs", type=int, default=80)
    parser.add_argument("--train_segments", type=int, default=2000)
    parser.add_argument("--val_segments", type=int, default=400)
    parser.add_argument("--long_segments_full", type=int, default=500)
    parser.add_argument("--long_segments_ablation", type=int, default=200)
    parser.add_argument("--include_k_diagnostics", action="store_true", default=True)
    parser.add_argument("--no_include_k_diagnostics", dest="include_k_diagnostics", action="store_false")
    parser.add_argument("--include_ood_eval", action="store_true", default=True)
    parser.add_argument("--no_include_ood_eval", dest="include_ood_eval", action="store_false")
    parser.add_argument("--write_only", action="store_true")
    parser.add_argument("--no_monitor", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU id is required")
    specs = _base_matrix(args)
    queue_paths = write_queues(specs, gpus, args.session_prefix, include_monitor=not args.no_monitor)
    manifest = {
        "result_root": str(RESULT_ROOT),
        "gpus": gpus,
        "session_prefix": args.session_prefix,
        "runs": [asdict(spec) | {"run_dir": str(spec.run_dir), "cache_path": str(spec.cache_path)} for spec in specs],
    }
    manifest_path = ROOT / RESULT_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote manifest: {manifest_path}")
    for path in queue_paths:
        print(f"wrote queue: {path}")
    if not args.write_only:
        launch_queues(queue_paths, gpus, args.session_prefix, include_monitor=not args.no_monitor)


if __name__ == "__main__":
    main()
