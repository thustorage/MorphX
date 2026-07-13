#!/usr/bin/env python3
"""Run SOSP 2026 AE part 1: multi-task co-location.

The script regenerates the log/table inputs for Figures 10, 11, and 12 from the
built-in workload points. It intentionally skips the smsched-88 columns and uses
only the plain smsched configuration.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fnmatch
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
COLOCATE_DIR = REPO_ROOT / "scripts" / "colocate"
SUMMARY_SCRIPT = REPO_ROOT / "scripts" / "analysis" / "colocate_summary.py"

DEFAULT_DURATION_SECONDS = 180.0
DEFAULT_MODEL = "llama-3-8b"
DEFAULT_DATASET = "sharegpt"
DEFAULT_INPUT_LENGTH = 2048
DEFAULT_OUTPUT_LENGTH = 2048
DEFAULT_GGNN_BASE_SIZE = 1_000_000
DEFAULT_GGNN_QUERY_SIZE = 10_000
DEFAULT_MM_SIZE = 8192

SCHEDULERS = (
    "mps-30",
    "mps-50",
    "orion",
    "smsched",
    "stream",
    "tgs",
    "timeslice",
)
SYSTEM_NAME_RE = re.compile("smsched", re.IGNORECASE)
SMSCHED_CONDA_ENV_RE = re.compile(r"(?<=/envs/)smsched(?=/)", re.IGNORECASE)
SCHEDULER_LABELS = {"smsched": "MorphX"}

COMPACT_METRICS = (
    "llm_latency_ms",
    "llm_p99_ms",
    "completed",
)

OUTPUT_TABLES = {
    "llm-1-ggnn": "Figure10,12(a).txt",
    "llm-4-ggnn": "Figure10,12(b).txt",
    "llm-1-mm": "Figure11(a).txt",
    "llm-4-mm": "Figure11(b).txt",
}

TABLE_FIGURE_LABELS = {
    "llm-1-ggnn": "Figures10,12(a)",
    "llm-4-ggnn": "Figures10,12(b)",
    "llm-1-mm": "Figure11(a)",
    "llm-4-mm": "Figure11(b)",
}

MODEL_PATHS = {
    "llama-3-8b": Path(
        "/huggingface-cache/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/"
        "snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
    ),
}

DATASET_PATHS = {
    "sharegpt": Path(
        "/huggingface-cache/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/"
        "snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/"
        "ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
    ),
}


@dataclass(frozen=True)
class Experiment:
    scenario_dir: str
    table_key: str
    experiment_name: str
    secondary_task: str
    llm_rps: float
    secondary_rps_raw: float
    secondary_rps_scaled: float


@dataclass(frozen=True)
class RunSpec:
    experiment: Experiment
    scheduler: str
    split_hint: str
    command: list[str]
    env_updates: dict[str, str]
    env_removals: tuple[str, ...]
    log_path: Path
    err_path: Path


def parse_args() -> argparse.Namespace:
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    default_output = REPO_ROOT / "ae-results" / "colocate-part1" / timestamp
    default_python = os.environ.get("PYTHON")
    if not default_python:
        conda_python = Path("/home/rtx/miniconda3/envs/smsched/bin/python")
        default_python = str(conda_python if conda_python.exists() else Path(sys.executable))

    parser = argparse.ArgumentParser(
        description="Run and summarize SOSP 2026 AE part 1 multi-task co-location experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--python", dest="python_bin", default=default_python)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--input-length", type=int, default=DEFAULT_INPUT_LENGTH)
    parser.add_argument("--output-length", type=int, default=DEFAULT_OUTPUT_LENGTH)
    parser.add_argument("--ggnn-base-size", type=int, default=DEFAULT_GGNN_BASE_SIZE)
    parser.add_argument("--ggnn-query-size", type=int, default=DEFAULT_GGNN_QUERY_SIZE)
    parser.add_argument("--mm-size", type=int, default=DEFAULT_MM_SIZE)
    parser.add_argument(
        "--smsched-env",
        default=os.environ.get("SMSCHED"),
        help=(
            "Shell-style environment assignments for smsched. If omitted, "
            "LD_PRELOAD=<repo>/runtime/build/libcuda.so:<repo>/runtime/build/libpreload.so is used."
        ),
    )
    parser.add_argument(
        "--tgs-high-env",
        default=os.environ.get("TGS_HIGH"),
        help=(
            "Shell-style environment assignments injected into the LLM child for TGS. "
            "If omitted, the script tries TGS/hijack/high-priority-lib."
        ),
    )
    parser.add_argument(
        "--tgs-low-env",
        default=os.environ.get("TGS_LOW"),
        help=(
            "Shell-style environment assignments injected into background children for TGS. "
            "If omitted, the script tries TGS/hijack/low-priority-lib."
        ),
    )
    parser.add_argument(
        "--mps-control",
        default=os.environ.get("MPS_CONTROL", "nvidia-cuda-mps-control"),
        help="Command used to control CUDA MPS. The script appends -d for start.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default="0",
        help=(
            "CUDA_VISIBLE_DEVICES for workload children. Defaults to GPU 0. "
            "Use 'auto' to pick the GPU with the lowest current memory use; "
            "use an empty string to leave unset."
        ),
    )
    parser.add_argument(
        "--only-scheduler",
        action="append",
        choices=SCHEDULERS,
        help="Limit to one scheduler. Repeat to select multiple schedulers.",
    )
    parser.add_argument(
        "--only-table",
        action="append",
        choices=tuple(OUTPUT_TABLES.keys()),
        help="Limit to one table/scenario. Repeat to select multiple tables.",
    )
    parser.add_argument(
        "--only-experiment",
        action="append",
        help=(
            "Limit to matching workload points. Accepts experiment names such as "
            "llm-1-ggnn-8, table keys, scenario names, or table-prefixed patterns; "
            "shell-style wildcards are supported. Repeat to select multiple groups."
        ),
    )
    parser.add_argument(
        "--smsched-split-hint",
        help=(
            "Set SMSCHED_SPLIT_HINT for every smsched run. If omitted, the runner "
            "preserves the original plain-smsched behavior and removes the variable."
        ),
    )
    parser.add_argument(
        "--smsched-split-hint-for",
        action="append",
        default=[],
        metavar="SELECTOR=SPLIT",
        help=(
            "Override SMSCHED_SPLIT_HINT for matching smsched workload points. "
            "SELECTOR may match the table key, scenario, experiment name, or "
            "table-prefixed experiment, e.g. llm-1-ggnn:llm-1-ggnn-*=76. "
            "Shell-style wildcards are supported; later rules win."
        ),
    )
    parser.add_argument(
        "--smsched-debug",
        action="store_true",
        help="Set SMSCHED_DEBUG=1 for smsched runs. Intended only for short diagnostic runs.",
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Stop after this many scheduler runs.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip logs that already contain a results block.")
    parser.add_argument("--fail-fast", action="store_true", help="Abort on the first failed scheduler run.")
    parser.add_argument(
        "--no-mpi-cxx-stub",
        action="store_true",
        help=(
            "Do not build an empty libmpi_cxx.so.40 shim when this PyTorch build "
            "declares but does not use the removed OpenMPI C++ binding library."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print and record commands without running them.")
    parser.add_argument("--preflight-only", action="store_true", help="Run preflight checks and exit.")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze existing logs in --output-dir.")
    parser.add_argument("--no-analyze", action="store_true", help="Do not build tables after experiments.")
    return parser.parse_args()


def format_rps_for_name(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def format_table_number(value: Optional[float], precision: int = 3) -> str:
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.{precision}f}"


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def reviewer_text(value: object) -> str:
    text = str(value)
    kept: list[tuple[str, str]] = []

    def keep(match: re.Match[str]) -> str:
        marker = f"__MXKEEP{len(kept)}__"
        kept.append((marker, match.group(0)))
        return marker

    updated = SYSTEM_NAME_RE.sub("MorphX", SMSCHED_CONDA_ENV_RE.sub(keep, text))
    for marker, original in kept:
        updated = updated.replace(marker, original)
    return updated


def scheduler_label(scheduler: str) -> str:
    return SCHEDULER_LABELS.get(scheduler, scheduler)


def parse_float(value: str) -> Optional[float]:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        parsed = float(stripped)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def build_experiments() -> list[Experiment]:
    experiments: list[Experiment] = []
    for llm_rps in (1.0, 4.0):
        for ggnn_rps in (1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0):
            llm_tag = format_rps_for_name(llm_rps)
            other_tag = format_rps_for_name(ggnn_rps)
            experiments.append(
                Experiment(
                    scenario_dir="llm-ggnn",
                    table_key=f"llm-{llm_tag}-ggnn",
                    experiment_name=f"llm-{llm_tag}-ggnn-{other_tag}",
                    secondary_task="ggnn",
                    llm_rps=llm_rps,
                    secondary_rps_raw=ggnn_rps,
                    secondary_rps_scaled=ggnn_rps * 0.1,
                )
            )

    mm_points = {
        1.0: (1.0, 2.0, 4.0, 6.0, 8.0, 10.0),
        4.0: (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    }
    for llm_rps, values in mm_points.items():
        for mm_rps in values:
            llm_tag = format_rps_for_name(llm_rps)
            other_tag = format_rps_for_name(mm_rps)
            experiments.append(
                Experiment(
                    scenario_dir="llm-mm",
                    table_key=f"llm-{llm_tag}-mm",
                    experiment_name=f"llm-{llm_tag}-mm-{other_tag}",
                    secondary_task="mm",
                    llm_rps=llm_rps,
                    secondary_rps_raw=mm_rps,
                    secondary_rps_scaled=mm_rps * 0.01,
                )
            )
    return experiments


def parse_smsched_split_hint_overrides(values: Iterable[str]) -> tuple[tuple[str, str], ...]:
    rules: list[tuple[str, str]] = []
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"--smsched-split-hint-for must look like SELECTOR=SPLIT: {raw!r}")
        selector, split_hint = raw.split("=", 1)
        selector = selector.strip()
        split_hint = split_hint.strip()
        if not selector or not split_hint:
            raise ValueError(f"--smsched-split-hint-for must include both selector and split: {raw!r}")
        rules.append((selector, split_hint))
    return tuple(rules)


def experiment_selector_candidates(experiment: Experiment) -> tuple[str, ...]:
    return (
        experiment.table_key,
        experiment.scenario_dir,
        experiment.experiment_name,
        f"{experiment.table_key}:{experiment.experiment_name}",
        f"{experiment.scenario_dir}:{experiment.experiment_name}",
    )


def experiment_matches_selector(experiment: Experiment, selector: str) -> bool:
    return any(fnmatch.fnmatchcase(candidate, selector) for candidate in experiment_selector_candidates(experiment))


def smsched_split_hint_for_experiment(args: argparse.Namespace, experiment: Experiment) -> str:
    split_hint = args.smsched_split_hint or ""
    for selector, override in args.smsched_split_hint_overrides:
        if experiment_matches_selector(experiment, selector):
            split_hint = override
    return split_hint


def unmatched_smsched_split_hint_selectors(args: argparse.Namespace, experiments: Iterable[Experiment]) -> list[str]:
    experiments = list(experiments)
    unmatched: list[str] = []
    for selector, _ in args.smsched_split_hint_overrides:
        if not any(experiment_matches_selector(experiment, selector) for experiment in experiments):
            unmatched.append(selector)
    return unmatched


def split_env_assignments(assignments: Optional[str]) -> dict[str, str]:
    if not assignments:
        return {}
    try:
        parts = shlex.split(assignments)
    except ValueError:
        parts = assignments.split()

    env_updates: dict[str, str] = {}
    for part in parts:
        if part == "env":
            continue
        if "=" not in part:
            raise ValueError(f"Invalid env assignment in {assignments!r}: {part!r}")
        key, value = part.split("=", 1)
        if not key:
            raise ValueError(f"Invalid empty env key in {assignments!r}")
        env_updates[key] = value
    return env_updates


def prepend_path_value(path: Path, current: Optional[str]) -> str:
    path_str = str(path)
    parts = [part for part in (current or "").split(":") if part]
    parts = [part for part in parts if part != path_str]
    return ":".join([path_str] + parts)


def python_lib_dir(python_bin: str) -> Optional[Path]:
    resolved = shutil.which(python_bin) if os.sep not in python_bin else python_bin
    if not resolved:
        return None
    path = Path(resolved).resolve()
    if path.parent.name != "bin":
        return None
    lib_dir = path.parent.parent / "lib"
    if lib_dir.is_dir():
        return lib_dir
    return None


def known_runtime_library_dirs(args: argparse.Namespace) -> list[Path]:
    dirs: list[Path] = []

    candidates = (
        Path("/mnt/data/rtx/llvm-project/install/lib/x86_64-unknown-linux-gnu"),
        Path("/mnt/data/rtx/llvm-project/build/lib/x86_64-unknown-linux-gnu"),
        Path("/home/rtx/llvm-project/install/lib/x86_64-unknown-linux-gnu"),
        Path("/home/rtx/llvm-project/build/lib/x86_64-unknown-linux-gnu"),
        python_lib_dir(args.python_bin),
        Path("/usr/mpi/gcc/openmpi-4.1.7rc1/lib"),
        Path("/mnt/data/rtx/miniconda3/envs/nanoflow/lib"),
        Path("/home/rtx/miniconda3/envs/nanoflow/lib"),
    )
    for candidate in candidates:
        if candidate is not None and candidate.is_dir():
            dirs.append(candidate)

    unique: list[Path] = []
    seen: set[str] = set()
    for directory in dirs:
        key = str(directory.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(directory)
    return unique


def library_exists_in_dirs(name: str, dirs: Iterable[Path]) -> bool:
    return any((directory / name).exists() for directory in dirs)


def build_empty_mpi_cxx_stub(stub_dir: Path) -> Optional[Path]:
    stub_dir.mkdir(parents=True, exist_ok=True)
    output = stub_dir / "libmpi_cxx.so.40"
    if output.exists():
        return output
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        return None
    command = [
        compiler,
        "-shared",
        "-fPIC",
        "-Wl,-soname,libmpi_cxx.so.40",
        "-o",
        str(output),
        "-xc",
        "/dev/null",
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode != 0:
        return None
    symlink = stub_dir / "libmpi_cxx.so"
    if not symlink.exists():
        try:
            symlink.symlink_to(output.name)
        except OSError:
            shutil.copy2(output, symlink)
    return output


def prepare_runtime_library_dirs(args: argparse.Namespace, output_dir: Path) -> list[Path]:
    dirs = known_runtime_library_dirs(args)
    env_dirs = [Path(part) for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part]
    if not args.no_mpi_cxx_stub and not library_exists_in_dirs("libmpi_cxx.so.40", [*dirs, *env_dirs]):
        stub = build_empty_mpi_cxx_stub(output_dir / "runtime-libs")
        if stub is not None:
            dirs.insert(0, stub.parent)
    return dirs


def choose_least_used_gpu() -> Optional[str]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    command = [
        nvidia_smi,
        "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20, check=False)
    except subprocess.SubprocessError:
        return None
    if result.returncode != 0:
        return None
    best_index: Optional[str] = None
    best_used: Optional[int] = None
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            used = int(parts[1])
        except ValueError:
            continue
        if best_used is None or used < best_used:
            best_index = parts[0]
            best_used = used
    return best_index


def resolve_cuda_visible_devices(args: argparse.Namespace) -> Optional[str]:
    raw = args.cuda_visible_devices
    if raw is None:
        return None
    if raw == "":
        return None
    if raw.lower() == "auto":
        return choose_least_used_gpu()
    return raw


def short_mps_pipe_base(output_dir: Path) -> Path:
    digest = hashlib.sha1(str(output_dir.resolve()).encode("utf-8")).hexdigest()[:12]
    return Path("/tmp") / f"morphx-ae-mps-{os.getuid()}-{digest}" / "part1"


def prepare_mps_dirs(output_dir: Path) -> tuple[Path, Path, Path, Path]:
    log_base = output_dir / "mps"
    pipe_base = short_mps_pipe_base(output_dir)
    plain_pipe = pipe_base / "plain"
    plain_log = log_base / "plain-log"
    mps_pipe = pipe_base / "mps"
    mps_log = log_base / "mps-log"
    for directory in (plain_pipe, plain_log, mps_pipe, mps_log):
        directory.mkdir(parents=True, exist_ok=True)
        try:
            directory.chmod(0o777)
        except OSError:
            pass
    return plain_pipe, plain_log, mps_pipe, mps_log


def common_env_updates(args: argparse.Namespace) -> dict[str, str]:
    updates: dict[str, str] = {}
    library_dirs = list(getattr(args, "runtime_library_dirs", []))
    if not library_dirs:
        library_dirs = known_runtime_library_dirs(args)
    ld_library_path = os.environ.get("LD_LIBRARY_PATH")
    for lib_dir in reversed(library_dirs):
        ld_library_path = prepend_path_value(lib_dir, ld_library_path)
    if ld_library_path:
        updates["LD_LIBRARY_PATH"] = ld_library_path
    cuda_visible_devices = getattr(args, "effective_cuda_visible_devices", None)
    if cuda_visible_devices:
        updates["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    plain_pipe_dir = getattr(args, "cuda_plain_mps_pipe_dir", None)
    plain_log_dir = getattr(args, "cuda_plain_mps_log_dir", None)
    if plain_pipe_dir:
        updates["CUDA_MPS_PIPE_DIRECTORY"] = str(plain_pipe_dir)
    if plain_log_dir:
        updates["CUDA_MPS_LOG_DIRECTORY"] = str(plain_log_dir)
    return updates


def mps_client_cuda_visible_devices(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    devices = [part.strip() for part in value.split(",") if part.strip()]
    if len(devices) == 1 and devices[0].isdigit():
        return "0"
    return value


def default_smsched_env(user_value: Optional[str]) -> str:
    if user_value:
        return user_value
    cuda_hook = REPO_ROOT / "runtime" / "build" / "libcuda.so"
    preload = REPO_ROOT / "runtime" / "build" / "libpreload.so"
    return f"LD_PRELOAD={cuda_hook}:{preload}"


def build_tgs_preload_string(priority: str) -> Optional[str]:
    lib_dir = REPO_ROOT / "TGS" / "hijack" / f"{priority}-priority-lib"
    lib_names = (
        "libcontroller.so",
        "libcuda.so",
        "libcuda.so.1",
        "libnvidia-ml.so",
        "libnvidia-ml.so.1",
    )
    paths = [lib_dir / name for name in lib_names]
    if not all(path.exists() for path in paths):
        return None
    return "LD_PRELOAD=" + ":".join(str(path) for path in paths)


def default_tgs_high_env(args: argparse.Namespace) -> Optional[str]:
    return args.tgs_high_env or build_tgs_preload_string("high")


def default_tgs_low_env(args: argparse.Namespace) -> Optional[str]:
    return args.tgs_low_env or build_tgs_preload_string("low")


def build_base_command(args: argparse.Namespace, experiment: Experiment) -> list[str]:
    cmd = [
        args.python_bin,
        "run.py",
        "--enable-llm",
        "--model",
        args.model,
        "--dataset",
        args.dataset,
        "--input-length",
        str(args.input_length),
        "--output-length",
        str(args.output_length),
        "--rps",
        format_rps_for_name(experiment.llm_rps),
        "--time",
        str(args.duration),
    ]
    if experiment.secondary_task == "ggnn":
        cmd.extend(
            [
                "--enable-ggnn",
                "--ggnn-rps",
                format_rps_for_name(experiment.secondary_rps_raw),
                "--ggnn-base-size",
                str(args.ggnn_base_size),
                "--ggnn-query-size",
                str(args.ggnn_query_size),
            ]
        )
    elif experiment.secondary_task == "mm":
        cmd.extend(
            [
                "--enable-mm",
                "--mm-rps",
                format_rps_for_name(experiment.secondary_rps_raw),
                "--mm-size",
                str(args.mm_size),
            ]
        )
    else:
        raise ValueError(f"Unsupported secondary task: {experiment.secondary_task}")
    return cmd


def build_run_spec(
    args: argparse.Namespace,
    logs_root: Path,
    experiment: Experiment,
    scheduler: str,
) -> RunSpec:
    cmd = build_base_command(args, experiment)
    env_updates: dict[str, str] = common_env_updates(args)
    env_removals: tuple[str, ...] = ()
    split_hint = ""

    if scheduler == "timeslice":
        cmd.insert(2, "--use-multiprocess")
    elif scheduler == "tgs":
        cmd.insert(2, "--use-tgs")
        cmd.insert(2, "--use-multiprocess")
        tgs_high = default_tgs_high_env(args)
        tgs_low = default_tgs_low_env(args)
        if tgs_high:
            env_updates["TGS_HIGH"] = tgs_high
        if tgs_low:
            env_updates["TGS_LOW"] = tgs_low
    elif scheduler == "orion":
        cmd.insert(2, "--use-orion")
    elif scheduler.startswith("mps-"):
        percentage = scheduler.split("-", 1)[1]
        cmd.insert(2, percentage)
        cmd.insert(2, "--mps-percentage")
        cmd.insert(2, "--use-mps")
        cmd.insert(2, "--use-multiprocess")
        client_cuda_visible_devices = mps_client_cuda_visible_devices(env_updates.get("CUDA_VISIBLE_DEVICES"))
        if client_cuda_visible_devices:
            env_updates["CUDA_VISIBLE_DEVICES"] = client_cuda_visible_devices
        mps_pipe_dir = getattr(args, "cuda_mps_pipe_dir", None)
        mps_log_dir = getattr(args, "cuda_mps_log_dir", None)
        if mps_pipe_dir:
            env_updates["CUDA_MPS_PIPE_DIRECTORY"] = str(mps_pipe_dir)
        if mps_log_dir:
            env_updates["CUDA_MPS_LOG_DIRECTORY"] = str(mps_log_dir)
    elif scheduler == "smsched":
        env_updates.update(split_env_assignments(default_smsched_env(args.smsched_env)))
        env_updates["SMSCHED"] = default_smsched_env(args.smsched_env)
        if args.smsched_debug:
            env_updates["SMSCHED_DEBUG"] = "1"
        split_hint = smsched_split_hint_for_experiment(args, experiment)
        if split_hint:
            env_updates["SMSCHED_SPLIT_HINT"] = split_hint
        else:
            env_removals = ("SMSCHED_SPLIT_HINT",)
    elif scheduler != "stream":
        raise ValueError(f"Unsupported scheduler: {scheduler}")

    log_dir = logs_root / experiment.scenario_dir / experiment.experiment_name
    return RunSpec(
        experiment=experiment,
        scheduler=scheduler,
        split_hint=split_hint,
        command=cmd,
        env_updates=env_updates,
        env_removals=env_removals,
        log_path=log_dir / f"{scheduler}.log",
        err_path=log_dir / f"{scheduler}.err",
    )


def selected_experiments(args: argparse.Namespace) -> list[Experiment]:
    experiments = build_experiments()
    if args.only_table:
        allowed = set(args.only_table)
        experiments = [exp for exp in experiments if exp.table_key in allowed]
    if args.only_experiment:
        selectors = tuple(args.only_experiment)
        experiments = [
            exp
            for exp in experiments
            if any(experiment_matches_selector(exp, selector) for selector in selectors)
        ]
        if not experiments:
            raise ValueError("--only-experiment matched no workload points: " + ", ".join(selectors))
    return experiments


def selected_schedulers(args: argparse.Namespace) -> tuple[str, ...]:
    if not args.only_scheduler:
        return SCHEDULERS
    allowed = set(args.only_scheduler)
    return tuple(scheduler for scheduler in SCHEDULERS if scheduler in allowed)


def has_complete_log(path: Path) -> bool:
    try:
        return "=== Results ===" in path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


def remove_incomplete_log_pair(log_path: Path, err_path: Path) -> None:
    if not log_path.exists() or has_complete_log(log_path):
        return
    for path in (log_path, err_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def quote_cmd(command: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: Optional[dict[str, str]] = None,
    stdout_path: Optional[Path] = None,
    stderr_path: Optional[Path] = None,
    input_text: Optional[str] = None,
    timeout: Optional[float] = None,
) -> subprocess.CompletedProcess[str]:
    stdout_target = subprocess.PIPE
    stderr_target = subprocess.PIPE
    stdout_handle = None
    stderr_handle = None
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_handle = stdout_path.open("ab")
        stdout_target = stdout_handle
    if stderr_path is not None:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_handle = stderr_path.open("ab")
        stderr_target = stderr_handle
    try:
        return subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            input=input_text,
            text=True,
            stdout=stdout_target,
            stderr=stderr_target,
            timeout=timeout,
            check=False,
        )
    finally:
        if stdout_handle is not None:
            stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def mps_control_command(args: argparse.Namespace) -> list[str]:
    return shlex.split(args.mps_control)


def mps_control_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.update(mps_control_assignments(args))
    return env


def mps_control_assignments(args: argparse.Namespace) -> dict[str, str]:
    assignments: dict[str, str] = {}
    cuda_visible_devices = getattr(args, "effective_cuda_visible_devices", None)
    if cuda_visible_devices:
        assignments["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    pipe_dir = getattr(args, "cuda_mps_pipe_dir", None)
    log_dir = getattr(args, "cuda_mps_log_dir", None)
    if pipe_dir:
        assignments["CUDA_MPS_PIPE_DIRECTORY"] = str(pipe_dir)
    if log_dir:
        assignments["CUDA_MPS_LOG_DIRECTORY"] = str(log_dir)
    return assignments


def mps_control_invocation(args: argparse.Namespace) -> list[str]:
    return mps_control_command(args)


def clear_private_mps_dir(path: Optional[Path]) -> None:
    if path is None or not path.is_dir():
        return
    for child in path.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError:
            pass


def private_mps_pid(args: argparse.Namespace) -> Optional[int]:
    pipe_dir = getattr(args, "cuda_mps_pipe_dir", None)
    if not pipe_dir:
        return None
    pid_path = pipe_dir / "nvidia-cuda-mps-control.pid"
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def private_mps_control_exists(args: argparse.Namespace) -> bool:
    pipe_dir = getattr(args, "cuda_mps_pipe_dir", None)
    return bool(pipe_dir and (pipe_dir / "control").exists())


def stop_mps(args: argparse.Namespace, err_path: Path) -> bool:
    if not private_mps_control_exists(args):
        append_text(err_path, "\n[AE] private MPS control socket not present; skip stop\n")
        return True
    command = mps_control_invocation(args)
    result = run_mps_stop_command(args, command, err_path)
    if result is None:
        return False
    return result.returncode == 0 or not private_mps_control_exists(args)


def run_mps_stop_command(args: argparse.Namespace, command: list[str], err_path: Path) -> Optional[subprocess.CompletedProcess[str]]:
    append_text(err_path, f"\n[AE] stopping MPS: {quote_cmd(command)}\n")
    try:
        result = run_command(
            command,
            cwd=REPO_ROOT,
            env=mps_control_env(args),
            stderr_path=err_path,
            input_text="quit\n",
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        append_text(err_path, "[AE] timed out while stopping private MPS\n")
        return None
    except OSError as exc:
        append_text(err_path, f"[AE] failed to stop private MPS: {exc}\n")
        return None
    if result.stdout:
        append_text(err_path, result.stdout)
    append_text(err_path, f"[AE] MPS stop command returned {result.returncode}\n")
    return result


def start_mps(args: argparse.Namespace, err_path: Path) -> bool:
    if private_mps_control_exists(args):
        stop_ok = stop_mps(args, err_path)
        time.sleep(1)
        if not stop_ok and private_mps_control_exists(args):
            pid = private_mps_pid(args)
            if pid is not None and process_exists(pid):
                append_text(err_path, f"[AE] private MPS daemon {pid} is still alive; refusing to clear pipe dir\n")
                return False
            append_text(err_path, "[AE] clearing stale private MPS pipe dir after failed stop\n")
    clear_private_mps_dir(getattr(args, "cuda_mps_pipe_dir", None))
    command = mps_control_invocation(args) + ["-d"]
    result = run_mps_start_command(args, command, err_path)
    if result is None:
        return False
    time.sleep(2)
    if result.returncode != 0:
        append_text(err_path, f"[AE] MPS start command failed with return code {result.returncode}\n")
        return False
    if not private_mps_control_exists(args):
        append_text(err_path, "[AE] private MPS control socket was not detected after start\n")
        return False
    return True


def run_mps_start_command(args: argparse.Namespace, command: list[str], err_path: Path) -> Optional[subprocess.CompletedProcess[str]]:
    append_text(err_path, f"\n[AE] starting MPS: {quote_cmd(command)}\n")
    try:
        result = run_command(
            command,
            cwd=REPO_ROOT,
            env=mps_control_env(args),
            stderr_path=err_path,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        append_text(err_path, "[AE] timed out while starting private MPS\n")
        time.sleep(2)
        if private_mps_control_exists(args):
            append_text(err_path, "[AE] private MPS control socket exists after timeout; continuing\n")
            return subprocess.CompletedProcess(command, 0)
        return None
    except OSError as exc:
        append_text(err_path, f"[AE] failed to start private MPS: {exc}\n")
        return None
    if result.stdout:
        append_text(err_path, result.stdout)
    append_text(err_path, f"[AE] MPS start command returned {result.returncode}\n")
    return result


def build_env(spec: RunSpec) -> dict[str, str]:
    env = os.environ.copy()
    for key in spec.env_removals:
        env.pop(key, None)
    env.update(spec.env_updates)
    return env


def run_spec(args: argparse.Namespace, spec: RunSpec) -> dict[str, object]:
    started_at = dt.datetime.now().isoformat(timespec="seconds")
    start = time.time()
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    status = "ok"
    returncode: Optional[int] = None
    skipped = False

    if args.skip_existing and has_complete_log(spec.log_path):
        skipped = True
        status = "skipped"
        elapsed = 0.0
        return {
            "status": status,
            "returncode": returncode,
            "elapsed_seconds": elapsed,
            "started_at": started_at,
            "skipped": skipped,
        }

    remove_incomplete_log_pair(spec.log_path, spec.err_path)

    if args.dry_run:
        append_text(spec.log_path, f"[AE dry-run] {quote_cmd(spec.command)}\n")
        status = "dry-run"
        elapsed = time.time() - start
        return {
            "status": status,
            "returncode": returncode,
            "elapsed_seconds": elapsed,
            "started_at": started_at,
            "skipped": skipped,
        }

    if spec.scheduler.startswith("mps-") and not start_mps(args, spec.err_path):
        status = "mps-start-failed"
        elapsed = time.time() - start
        return {
            "status": status,
            "returncode": returncode,
            "elapsed_seconds": elapsed,
            "started_at": started_at,
            "skipped": skipped,
        }

    env = build_env(spec)
    append_text(spec.err_path, f"\n[AE] command: {quote_cmd(spec.command)}\n")
    append_text(spec.err_path, f"[AE] cwd: {COLOCATE_DIR}\n")
    if spec.env_updates:
        visible_env = {key: spec.env_updates[key] for key in sorted(spec.env_updates)}
        append_text(spec.err_path, f"[AE] env updates: {json.dumps(visible_env, sort_keys=True)}\n")
    if spec.env_removals:
        append_text(spec.err_path, f"[AE] env removals: {', '.join(spec.env_removals)}\n")

    try:
        result = run_command(
            spec.command,
            cwd=COLOCATE_DIR,
            env=env,
            stdout_path=spec.log_path,
            stderr_path=spec.err_path,
        )
        returncode = result.returncode
        if returncode != 0:
            status = "failed"
    finally:
        if spec.scheduler.startswith("mps-"):
            stop_mps(args, spec.err_path)
            time.sleep(2)

    elapsed = time.time() - start
    return {
        "status": status,
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "started_at": started_at,
        "skipped": skipped,
    }


def write_manifest_header(path: Path, args: argparse.Namespace, logs_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "logs_root": str(logs_root),
        "duration_seconds": args.duration,
        "schedulers": [scheduler_label(scheduler) for scheduler in selected_schedulers(args)],
        "tables": args.only_table or list(OUTPUT_TABLES),
        "morphx_note": (
            "The extra MorphX comparison columns from earlier development "
            "runs are intentionally omitted from AE tables."
        ),
    }
    path.write_text(reviewer_text(json.dumps(payload, indent=2, sort_keys=True)) + "\n", encoding="utf-8")


def append_command_row(path: Path, spec: RunSpec, result: Optional[dict[str, object]] = None) -> None:
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "table_key",
            "scenario",
            "experiment",
            "scheduler",
            "split_hint",
            "llm_rps",
            "other_rps_raw",
            "other_rps_scaled",
            "status",
            "returncode",
            "elapsed_seconds",
            "env_updates",
            "env_removals",
            "command",
            "log",
            "err",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        if write_header:
            writer.writeheader()
        result = result or {}
        writer.writerow(
            {
                "table_key": spec.experiment.table_key,
                "scenario": spec.experiment.scenario_dir,
                "experiment": spec.experiment.experiment_name,
                "scheduler": scheduler_label(spec.scheduler),
                "split_hint": spec.split_hint,
                "llm_rps": format_table_number(spec.experiment.llm_rps),
                "other_rps_raw": format_table_number(spec.experiment.secondary_rps_raw),
                "other_rps_scaled": format_table_number(spec.experiment.secondary_rps_scaled),
                "status": result.get("status", "planned"),
                "returncode": result.get("returncode", ""),
                "elapsed_seconds": format_table_number(result.get("elapsed_seconds"), precision=2)
                if isinstance(result.get("elapsed_seconds"), (float, int))
                else "",
                "env_updates": reviewer_text(json.dumps(spec.env_updates, sort_keys=True)),
                "env_removals": reviewer_text(",".join(spec.env_removals)),
                "command": reviewer_text(quote_cmd(spec.command)),
                "log": reviewer_text(spec.log_path),
                "err": reviewer_text(spec.err_path),
            }
        )


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        rows = list(reader)
    if not rows:
        return [], []
    headers = [cell.strip() for cell in rows[0]]
    data_rows: list[dict[str, str]] = []
    for row in rows[1:]:
        if not any(cell.strip() for cell in row):
            continue
        padded = row + [""] * (len(headers) - len(row))
        data_rows.append({headers[idx]: padded[idx].strip() for idx in range(len(headers))})
    return headers, data_rows


def write_tsv(path: Path, headers: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def figure_label_for_experiment(experiment: Experiment) -> str:
    return TABLE_FIGURE_LABELS.get(experiment.table_key, "Figures10-12")


def figure_scope_for_experiments(experiments: Iterable[Experiment]) -> str:
    table_keys = {experiment.table_key for experiment in experiments}
    if table_keys and all(key.endswith("-ggnn") for key in table_keys):
        return "Figures10,12"
    if table_keys and all(key.endswith("-mm") for key in table_keys):
        return "Figure11"
    return "Figures10-12"


def run_analysis(args: argparse.Namespace, logs_root: Path, output_dir: Path) -> int:
    full_dir = output_dir / "tables" / "full"
    stdout_path = output_dir / "analysis" / "colocate_summary.stdout.txt"
    stderr_path = output_dir / "analysis" / "colocate_summary.stderr.txt"
    command = [
        args.python_bin,
        str(SUMMARY_SCRIPT),
        "--logs-root",
        str(logs_root),
        "--output-dir",
        str(full_dir),
    ]
    print(f"[AE][part1][Figures10-12][analysis] START logs={logs_root}", flush=True)
    result = run_command(
        command,
        cwd=REPO_ROOT,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    if result.returncode != 0:
        print(
            f"[AE][part1][Figures10-12][analysis] FAILED rc={result.returncode} stderr={stderr_path}",
            flush=True,
        )
        return result.returncode
    build_compact_tables(full_dir, output_dir / "tables" / "compact")
    print(f"[AE][part1][Figures10-12][analysis] DONE tables={output_dir / 'tables' / 'compact'}", flush=True)
    return 0


def build_compact_tables(full_dir: Path, compact_dir: Path) -> None:
    compact_dir.mkdir(parents=True, exist_ok=True)
    for table_key in OUTPUT_TABLES:
        source = full_dir / f"{table_key}.tsv"
        if not source.exists():
            continue
        _, rows = read_tsv(source)
        headers = ["other_rps"]
        for scheduler in SCHEDULERS:
            for metric in COMPACT_METRICS:
                headers.append(f"{scheduler_label(scheduler)}_{metric}")

        compact_rows: list[dict[str, object]] = []
        for row in rows:
            compact: dict[str, object] = {"other_rps": row.get("other_rps", "")}
            for scheduler in SCHEDULERS:
                for metric in COMPACT_METRICS:
                    col = f"{scheduler}_{metric}"
                    compact[f"{scheduler_label(scheduler)}_{metric}"] = row.get(col, "")
            compact_rows.append(compact)
        write_tsv(compact_dir / f"{table_key}.tsv", headers, compact_rows)


def preflight(args: argparse.Namespace, output_dir: Path) -> tuple[bool, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    python_path = shutil.which(args.python_bin) if os.sep not in args.python_bin else args.python_bin
    if not python_path or not Path(python_path).exists():
        errors.append(f"Python not found: {args.python_bin}")

    required_paths = [
        COLOCATE_DIR / "run.py",
        SUMMARY_SCRIPT,
    ]
    for path in required_paths:
        if not path.exists():
            errors.append(f"Missing required path: {path}")

    model_path = MODEL_PATHS.get(args.model)
    if model_path and not model_path.exists():
        warnings.append(f"Model path from run.py table is missing: {model_path}")
    dataset_path = DATASET_PATHS.get(args.dataset)
    if dataset_path and not dataset_path.exists():
        warnings.append(f"Dataset path from run.py table is missing: {dataset_path}")

    smsched_env = default_smsched_env(args.smsched_env)
    try:
        smsched_updates = split_env_assignments(smsched_env)
        ld_preload = smsched_updates.get("LD_PRELOAD")
        if ld_preload:
            for preload_path in ld_preload.split(":"):
                if preload_path and not Path(preload_path).exists():
                    warnings.append(f"MorphX LD_PRELOAD target is missing: {preload_path}")
    except ValueError as exc:
        errors.append(str(exc))

    if "tgs" in selected_schedulers(args):
        tgs_high = default_tgs_high_env(args)
        tgs_low = default_tgs_low_env(args)
        if not tgs_high:
            warnings.append("No TGS_HIGH value or high-priority fallback libs found; tgs runs will not inject TGS.")
        else:
            try:
                for preload_path in split_env_assignments(tgs_high).get("LD_PRELOAD", "").split(":"):
                    if preload_path and not Path(preload_path).exists():
                        warnings.append(f"TGS high-priority LD_PRELOAD target is missing: {preload_path}")
            except ValueError as exc:
                errors.append(str(exc))
        if not tgs_low:
            warnings.append("No TGS_LOW value or low-priority fallback libs found; tgs runs will not inject TGS.")
        else:
            try:
                for preload_path in split_env_assignments(tgs_low).get("LD_PRELOAD", "").split(":"):
                    if preload_path and not Path(preload_path).exists():
                        warnings.append(f"TGS low-priority LD_PRELOAD target is missing: {preload_path}")
            except ValueError as exc:
                errors.append(str(exc))

    if any(scheduler.startswith("mps-") for scheduler in selected_schedulers(args)):
        mps_cmd = mps_control_command(args)
        if not mps_cmd:
            errors.append("Empty --mps-control command")
        elif shutil.which(mps_cmd[-1]) is None and shutil.which(mps_cmd[0]) is None:
            warnings.append(f"MPS control command may not be available: {args.mps_control}")

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = subprocess.run([nvidia_smi, "-L"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20)
        if result.returncode != 0:
            warnings.append(f"nvidia-smi -L failed: {result.stderr.strip()}")
    else:
        warnings.append("nvidia-smi was not found in PATH.")

    output_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = output_dir / "preflight.txt"
    with preflight_path.open("w", encoding="utf-8") as handle:
        handle.write("errors:\n")
        for item in errors:
            handle.write(f"- {item}\n")
        handle.write("warnings:\n")
        for item in warnings:
            handle.write(f"- {item}\n")
    return not errors, errors, warnings


def main() -> int:
    args = parse_args()
    args.smsched_split_hint_overrides = parse_smsched_split_hint_overrides(args.smsched_split_hint_for)
    output_dir = args.output_dir.resolve()
    logs_root = output_dir / "logs" / "colocate"
    manifest_path = output_dir / "manifest.json"
    commands_path = output_dir / "commands.tsv"

    print(f"[AE][part1][Figures10-12][setup] output_dir={output_dir}", flush=True)
    print("[AE][part1][Figures10-12][setup] MorphX extra comparison columns are omitted.", flush=True)

    args.runtime_library_dirs = prepare_runtime_library_dirs(args, output_dir)
    args.effective_cuda_visible_devices = resolve_cuda_visible_devices(args)
    (
        args.cuda_plain_mps_pipe_dir,
        args.cuda_plain_mps_log_dir,
        args.cuda_mps_pipe_dir,
        args.cuda_mps_log_dir,
    ) = prepare_mps_dirs(output_dir)
    if args.runtime_library_dirs:
        print(
            "[AE][part1][Figures10-12][setup] runtime_library_dirs="
            + reviewer_text(":".join(str(path) for path in args.runtime_library_dirs)),
            flush=True,
        )
    if args.effective_cuda_visible_devices:
        print(f"[AE][part1][Figures10-12][setup] CUDA_VISIBLE_DEVICES={args.effective_cuda_visible_devices}", flush=True)
    print(f"[AE][part1][Figures10-12][setup] non_mps_cuda_pipe_dir={args.cuda_plain_mps_pipe_dir}", flush=True)
    print(f"[AE][part1][Figures10-12][setup] private_mps_pipe_dir={args.cuda_mps_pipe_dir}", flush=True)

    ok, errors, warnings = preflight(args, output_dir)
    for warning in warnings:
        print(f"[AE][part1][Figures10-12][warning] {warning}", flush=True)
    for error in errors:
        print(f"[AE][part1][Figures10-12][error] {error}", flush=True)
    if not ok:
        return 2
    if args.preflight_only:
        print("[AE][part1][Figures10-12][setup] preflight complete.", flush=True)
        return 0

    if args.analyze_only:
        analysis_rc = run_analysis(args, logs_root, output_dir)
        if analysis_rc != 0:
            return analysis_rc
        print(f"[AE][part1][Figures10-12][analysis] analyze_only_complete={output_dir / 'tables' / 'compact'}", flush=True)
        return 0

    write_manifest_header(manifest_path, args, logs_root)

    experiments = selected_experiments(args)
    for selector in unmatched_smsched_split_hint_selectors(args, experiments):
        print(f"[AE][part1][{figure_scope_for_experiments(experiments)}][warning] unmatched MorphX tuning selector", flush=True)
    schedulers = selected_schedulers(args)
    specs: list[RunSpec] = []
    for experiment in experiments:
        for scheduler in schedulers:
            specs.append(build_run_spec(args, logs_root, experiment, scheduler))
    if args.max_runs is not None:
        specs = specs[: args.max_runs]

    figure_scope = figure_scope_for_experiments(experiments)
    print(
        f"[AE][part1][{figure_scope}][plan] planned_runs={len(specs)} "
        f"({len(experiments)} workload points x {len(schedulers)} schedulers, truncated={args.max_runs is not None})",
        flush=True,
    )

    failures = 0
    wall_start = time.time()
    for idx, spec in enumerate(specs, start=1):
        run_figure = figure_label_for_experiment(spec.experiment)
        print(
            f"[AE][part1][{run_figure}][run {idx}/{len(specs)}] START "
            f"{spec.experiment.experiment_name} scheduler={scheduler_label(spec.scheduler)}",
            flush=True,
        )
        result = run_spec(args, spec)
        append_command_row(commands_path, spec, result)
        status = result.get("status")
        elapsed = result.get("elapsed_seconds")
        elapsed_text = f"{elapsed:.1f}s" if isinstance(elapsed, (float, int)) else "n/a"
        total_elapsed = time.time() - wall_start
        avg_elapsed = total_elapsed / idx if idx else 0.0
        remaining = avg_elapsed * (len(specs) - idx)
        print(
            f"[AE][part1][{run_figure}][run {idx}/{len(specs)}] DONE "
            f"status={status} run_elapsed={elapsed_text} "
            f"part_elapsed={format_duration(total_elapsed)} "
            f"part_eta={format_duration(remaining)}",
            flush=True,
        )
        if status not in {"ok", "skipped", "dry-run"}:
            failures += 1
            if args.fail_fast:
                print(f"[AE][part1][{run_figure}][run {idx}/{len(specs)}] stopping because --fail-fast is set.", flush=True)
                break

    if args.no_analyze:
        return 1 if failures else 0

    analysis_rc = run_analysis(args, logs_root, output_dir)
    if analysis_rc != 0:
        return analysis_rc

    print(f"[AE][part1][Figures10-12][analysis] compact_tables={output_dir / 'tables' / 'compact'}", flush=True)
    if failures:
        print(f"[AE][part1][Figures10-12] completed with {failures} failed scheduler runs.", flush=True)
        return 1
    print("[AE][part1][Figures10-12] completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
