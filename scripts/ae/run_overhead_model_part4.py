#!/usr/bin/env python3
"""Run SOSP 2026 AE part 4: overhead and model-accuracy experiments.

The runner regenerates the log/table inputs for Figure 15 and Figure 16.
It writes fresh artifacts under ae-results/overhead-model-part4/<timestamp>.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
OVERHEAD_DIR = REPO_ROOT / "scripts" / "overhead"
SINGLE_REQUEST = OVERHEAD_DIR / "single_request.py"
MICROBENCH_DIR = REPO_ROOT / "microbench"
MICROBENCH_BUILD = MICROBENCH_DIR / "build"
BENCH_BIN = MICROBENCH_BUILD / "bench"
MICROBENCH_MODEL_BUILD = MICROBENCH_DIR / "build-model"
BENCH_BIN_MODEL = MICROBENCH_MODEL_BUILD / "bench"

OVERHEAD_TASKS = ("DNN", "GEMM", "GEMV", "GGNN", "LLM")
OVERHEAD_METHODS = ("base", "ncu", "neutrino", "nvbit", "smsched")
MODEL_SERIES = ("16", "128", "512", "8192", "ggnn")
SYSTEM_NAME_RE = re.compile("smsched", re.IGNORECASE)
SMSCHED_CONDA_ENV_RE = re.compile(r"(?<=/envs/)smsched(?=/)", re.IGNORECASE)
METHOD_LABELS = {"smsched": "MorphX"}
SPLIT_NAME_RE = re.compile("split", re.IGNORECASE)
MODEL_SMS_POINTS = {
    "16": (108, 72, 36, 4),
    "128": (108, 72, 36, 4),
    "512": (108, 96, 84, 72, 60, 48, 36, 24, 12, 8),
    "8192": (108, 96, 84, 72, 60, 48, 36, 24, 12, 4),
    "ggnn": (108, 96, 84, 72, 60, 48, 36, 24, 12),
}

AVG_MS_RE = re.compile(r"\bavg_ms=([0-9.+\-eE]+)")
PROFILE_KERNEL_RE = re.compile(
    r"^\[(?:smsched|MorphX)\] kernel (?P<name>.*?): blocks: (?P<blocks>\d+), occup: (?P<occup>\d+), "
    r"waves: (?P<waves>\d+), CI: (?P<ci>[0-9.+\-eE]+), WaveTime: (?P<wavetime>[0-9.+\-eE]+)us"
)
PROFILE_KERNEL_SHORT_RE = re.compile(
    r"^\[(?:smsched|MorphX)\] kernel (?P<name>.*?): occup: (?P<occup>\d+), "
    r"CI: (?P<ci>[0-9.+\-eE]+), WaveTime: (?P<wavetime>[0-9.+\-eE]+)us"
)
PROFILE_WAVE_RE = re.compile(r"^\[(?:smsched|MorphX)\] AvgWave \d+: (?P<sms>\d+), (?P<lat>[0-9.+\-eE]+)us")
BENCH_RESULT_RE = re.compile(r"^M=\s*(?P<m>\d+),\s*N=\s*(?P<n>\d+),\s*K=\s*(?P<k>\d+)\s*\|")


@dataclass(frozen=True)
class OverheadSpec:
    task: str
    base_args: tuple[str, ...]
    profiler_args: tuple[str, ...]


@dataclass(frozen=True)
class ProfileBlock:
    name: str
    blocks: int
    occup: int
    waves_count: int
    ci: float
    wavetime_us: float
    waves: tuple[tuple[int, float], ...]
    source: str = "new"


OVERHEAD_SPECS: dict[str, OverheadSpec] = {
    "DNN": OverheadSpec("DNN", ("--workload", "dnn", "--runs", "5"), ("--workload", "dnn", "--runs", "1")),
    "GEMM": OverheadSpec(
        "GEMM",
        ("--workload", "gemm", "--mm-size", "8192", "--runs", "5"),
        ("--workload", "gemm", "--mm-size", "8192", "--runs", "1"),
    ),
    "GEMV": OverheadSpec(
        "GEMV",
        ("--workload", "gemm", "--mm-m", "8192", "--mm-n", "16", "--mm-k", "8192", "--runs", "5"),
        ("--workload", "gemm", "--mm-m", "8192", "--mm-n", "16", "--mm-k", "8192", "--runs", "1"),
    ),
    "GGNN": OverheadSpec(
        "GGNN",
        ("--workload", "ggnn", "--ggnn-base-size", "1000000", "--ggnn-query-size", "10000", "--runs", "5"),
        ("--workload", "ggnn", "--ggnn-base-size", "1000000", "--ggnn-query-size", "10000", "--runs", "1"),
    ),
    "LLM": OverheadSpec(
        "LLM",
        ("--workload", "llm", "--model", "llama-3-8b", "--input-length", "4096", "--output-length", "3", "--runs", "1"),
        ("--workload", "llm", "--model", "llama-3-8b", "--input-length", "4096", "--output-length", "3", "--runs", "1"),
    ),
}


def parse_args() -> argparse.Namespace:
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    default_output = REPO_ROOT / "ae-results" / "overhead-model-part4" / timestamp
    conda_python = Path("/home/rtx/miniconda3/envs/smsched/bin/python")
    default_python = os.environ.get("PYTHON") or str(conda_python if conda_python.exists() else Path(sys.executable))

    parser = argparse.ArgumentParser(
        description="Run and summarize SOSP 2026 AE Figure 15/16 experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--python", dest="python_bin", default=default_python)
    parser.add_argument(
        "--smsched-env",
        default=os.environ.get("SMSCHED"),
        help="Shell-style environment assignments for SMSched. Defaults to runtime/build/libcuda.so:libpreload.so.",
    )
    parser.add_argument(
        "--smsched-model-env",
        default=os.environ.get("SMSCHED_MODEL"),
        help="Shell-style environment assignments for Figure16. Defaults to runtime/build/libcuda.so:libpreload.so.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default="0",
        help="CUDA_VISIBLE_DEVICES for child runs. Defaults to GPU 0; use 'auto' to pick the least-used GPU; empty string leaves it unset.",
    )
    parser.add_argument("--only-section", action="append", choices=("overhead", "model"), help="Run only one section.")
    parser.add_argument("--only-task", action="append", choices=OVERHEAD_TASKS, help="Limit Figure15 to selected tasks.")
    parser.add_argument("--only-method", action="append", choices=OVERHEAD_METHODS, help="Limit Figure15 to selected methods.")
    parser.add_argument("--only-series", action="append", choices=MODEL_SERIES, help="Limit Figure16 to selected series.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse complete logs in the output directory.")
    parser.add_argument("--fail-fast", action="store_true", help="Abort after the first failed child run.")
    parser.add_argument("--dry-run", action="store_true", help="Write commands without executing them.")
    parser.add_argument("--preflight-only", action="store_true", help="Run preflight checks and exit.")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze existing logs in --output-dir.")
    parser.add_argument("--no-build", action="store_true", help="Do not rebuild runtime/microbench before running.")
    parser.add_argument("--allow-legacy-model-fallback", action="store_true", help="Use legacy Figure16 profile logs when a new log is missing.")
    parser.add_argument("--no-mpi-cxx-stub", action="store_true", help="Do not create a local libmpi_cxx.so.40 shim.")
    parser.add_argument("--no-analyze", action="store_true", help="Skip table generation.")
    parser.add_argument("--no-update-latest-link", action="store_true", help="Do not update latest result/table symlinks.")
    return parser.parse_args()


def split_env_assignments(assignments: Optional[str]) -> dict[str, str]:
    if not assignments:
        return {}
    try:
        parts = shlex.split(assignments)
    except ValueError:
        parts = assignments.split()

    updates: dict[str, str] = {}
    for part in parts:
        if part == "env":
            continue
        if "=" not in part:
            raise ValueError(f"Invalid environment assignment: {part!r}")
        key, value = part.split("=", 1)
        if not key:
            raise ValueError("Environment assignment has an empty key")
        updates[key] = value
    return updates


def default_smsched_env(user_value: Optional[str]) -> str:
    if user_value:
        return user_value
    cuda_hook = REPO_ROOT / "runtime" / "build" / "libcuda.so"
    preload = REPO_ROOT / "runtime" / "build" / "libpreload.so"
    return f"LD_PRELOAD={cuda_hook}:{preload}"


def default_smsched_model_env(user_value: Optional[str]) -> str:
    if user_value:
        return user_value
    return default_smsched_env(None)


def python_lib_dir(python_bin: str) -> Optional[Path]:
    resolved = shutil.which(python_bin) if os.sep not in python_bin else python_bin
    if not resolved:
        return None
    path = Path(resolved).resolve()
    if path.parent.name != "bin":
        return None
    lib_dir = path.parent.parent / "lib"
    return lib_dir if lib_dir.is_dir() else None


def prepend_path(path: Path, current: Optional[str]) -> str:
    path_str = str(path)
    parts = [part for part in (current or "").split(":") if part and part != path_str]
    return ":".join([path_str] + parts)


def common_env(args: argparse.Namespace, include_mpi_stub: bool = True) -> dict[str, str]:
    updates: dict[str, str] = {}
    ld_library_path = os.environ.get("LD_LIBRARY_PATH")
    runtime_dirs = tuple(getattr(args, "runtime_library_dirs", []))
    if not include_mpi_stub:
        mpi_stub_dir = getattr(args, "mpi_cxx_stub_dir", None)
        if mpi_stub_dir is not None:
            runtime_dirs = tuple(directory for directory in runtime_dirs if directory != mpi_stub_dir)
    candidates = runtime_dirs or (
        Path("/mnt/data/rtx/llvm-project/install/lib/x86_64-unknown-linux-gnu"),
        Path("/mnt/data/rtx/llvm-project/build/lib/x86_64-unknown-linux-gnu"),
        python_lib_dir(args.python_bin),
        Path("/usr/mpi/gcc/openmpi-4.1.7rc1/lib"),
    )
    for candidate in reversed(candidates):
        if candidate is not None and candidate.is_dir():
            ld_library_path = prepend_path(candidate, ld_library_path)
    if ld_library_path:
        updates["LD_LIBRARY_PATH"] = ld_library_path
    cuda_visible_devices = getattr(args, "effective_cuda_visible_devices", None)
    if cuda_visible_devices:
        updates["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return updates


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


def prepare_runtime_library_dirs(args: argparse.Namespace) -> list[Path]:
    dirs: list[Path] = []
    args.mpi_cxx_stub_dir = None
    for candidate in (
        Path("/mnt/data/rtx/llvm-project/install/lib/x86_64-unknown-linux-gnu"),
        Path("/mnt/data/rtx/llvm-project/build/lib/x86_64-unknown-linux-gnu"),
        python_lib_dir(args.python_bin),
        Path("/usr/mpi/gcc/openmpi-4.1.7rc1/lib"),
    ):
        if candidate is not None and candidate.is_dir():
            dirs.append(candidate)
    env_dirs = [Path(part) for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part]
    if not args.no_mpi_cxx_stub and not library_exists_in_dirs("libmpi_cxx.so.40", [*dirs, *env_dirs]):
        stub = build_empty_mpi_cxx_stub(args.output_dir / "runtime-libs")
        if stub is not None:
            args.mpi_cxx_stub_dir = stub.parent
            dirs.insert(0, stub.parent)
    unique: list[Path] = []
    seen: set[str] = set()
    for directory in dirs:
        key = str(directory.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(directory)
    return unique


def choose_least_used_gpu() -> Optional[str]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    command = [nvidia_smi, "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]
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
    if raw is None or raw == "":
        return None
    if raw.lower() == "auto":
        return choose_least_used_gpu()
    return raw


def build_child_env(
    args: argparse.Namespace,
    updates: Optional[dict[str, str]] = None,
    removals: Iterable[str] = (),
    include_mpi_stub: bool = True,
    clean_base: bool = False,
) -> dict[str, str]:
    if clean_base:
        env = {
            key: os.environ[key]
            for key in ("PATH", "HOME", "USER", "SHELL", "TERM", "CUDA_HOME", "CUDAToolkit_ROOT")
            if key in os.environ
        }
    else:
        env = os.environ.copy()
    for key in (
        "LD_PRELOAD",
        "CUDA_INJECTION64_PATH",
        "NCU_PROFILING",
        "SMSCHED",
        "SMSCHED_SPLIT_HINT",
        "SMSCHED_PROFILE",
        "SMSCHED_PROFILE_LOG",
        "SMSCHED_ONLY_WHITELIST",
        "SMSCHED_TARGET_PATTERNS",
        "NEUTRINO_HOOK_DRIVER",
        "NEUTRINO_REAL_DRIVER",
        "NEUTRINO_DRIVER_NAME",
    ):
        env.pop(key, None)
    for key in removals:
        env.pop(key, None)
    env.update(common_env(args, include_mpi_stub=include_mpi_stub))
    if updates:
        env.update(updates)
    return env


def run_command(
    label: str,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    err_path: Path,
    dry_run: bool,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_line = " ".join(shlex.quote(part) for part in command)
    header = reviewer_text(
        f"# label: {label}\n# cwd: {cwd}\n# command: {command_line}\n# started: {dt.datetime.now().isoformat()}\n"
    )
    with log_path.open("w", encoding="utf-8") as stdout_handle, err_path.open("w", encoding="utf-8") as stderr_handle:
        stdout_handle.write(header)
        stdout_handle.flush()
        if dry_run:
            stdout_handle.write("# dry-run: command not executed\n")
            return 0
        start = time.monotonic()
        process = subprocess.Popen(command, cwd=str(cwd), env=env, stdout=stdout_handle, stderr=stderr_handle, text=True)
        return_code = process.wait()
        elapsed = time.monotonic() - start
        stdout_handle.write(f"# finished: {dt.datetime.now().isoformat()}\n# returncode: {return_code}\n# elapsed_seconds: {elapsed:.3f}\n")
    return return_code


def has_avg_log(path: Path) -> bool:
    try:
        return AVG_MS_RE.search(path.read_text(encoding="utf-8", errors="ignore")) is not None
    except OSError:
        return False


def has_profile_log(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return "[smsched] AvgWave" in text or "[MorphX] AvgWave" in text
    except OSError:
        return False


def selected_sections(args: argparse.Namespace) -> tuple[str, ...]:
    if not args.only_section:
        return ("overhead", "model")
    allowed = set(args.only_section)
    return tuple(section for section in ("overhead", "model") if section in allowed)


def section_prefix(args: argparse.Namespace) -> str:
    sections = selected_sections(args)
    if sections == ("overhead",):
        return "[AE][part3][Figure15]"
    if sections == ("model",):
        return "[AE][part4][Figure16]"
    return "[AE][part3-4][Figures15-16]"


def selected_tasks(args: argparse.Namespace) -> tuple[str, ...]:
    if not args.only_task:
        return OVERHEAD_TASKS
    allowed = set(args.only_task)
    return tuple(task for task in OVERHEAD_TASKS if task in allowed)


def selected_methods(args: argparse.Namespace) -> tuple[str, ...]:
    if not args.only_method:
        return OVERHEAD_METHODS
    allowed = set(args.only_method)
    return tuple(method for method in OVERHEAD_METHODS if method in allowed)


def selected_model_series(args: argparse.Namespace) -> tuple[str, ...]:
    if not args.only_series:
        return MODEL_SERIES
    allowed = set(args.only_series)
    return tuple(series for series in MODEL_SERIES if series in allowed)


def overhead_command(args: argparse.Namespace, task: str, method: str, log_base: Path) -> tuple[list[str], dict[str, str], tuple[str, ...], Optional[str]]:
    spec = OVERHEAD_SPECS[task]
    bench_args = spec.profiler_args if method in {"ncu", "nvbit", "neutrino"} else spec.base_args
    command = [args.python_bin, "single_request.py", *bench_args]
    env_updates: dict[str, str] = {}
    removals: tuple[str, ...] = ()

    if method == "base":
        return command, env_updates, removals, None
    if method == "smsched":
        env_updates.update(split_env_assignments(default_smsched_env(args.smsched_env)))
        env_updates.update(
            {
                "SMSCHED": default_smsched_env(args.smsched_env),
                "SMSCHED_PROFILE": "0" if task == "GGNN" else "1",
                "SMSCHED_PROFILE_LOG": "0",
                "SMSCHED_DEBUG": "0",
                "SMSCHED_SPLIT_HINT": "108",
            }
        )
        if task == "GGNN":
            env_updates["SMSCHED_ONLY_WHITELIST"] = "1"
        return command, env_updates, removals, None
    if method == "ncu":
        if shutil.which("ncu") is None:
            return command, env_updates, removals, "ncu not found"
        ncu_output = log_base / "ncu_output"
        env_updates["NCU_PROFILING"] = "1"
        ncu_flags = [
            "ncu",
            "-o",
            str(ncu_output),
            "-f",
            "--profile-from-start",
            "off",
            "--metrics",
            "dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum",
            "--target-processes",
            "all",
        ]
        command = [
            *ncu_flags,
            args.python_bin,
            "single_request.py",
            *bench_args,
        ]
        return command, env_updates, removals, None
    if method == "nvbit":
        tool = REPO_ROOT / "nvbit-tutorial" / "tools" / "mem_trace" / "mem_trace.so"
        if not tool.exists():
            return command, env_updates, removals, f"NvBit mem_trace tool not found: {tool}"
        env_updates["CUDA_INJECTION64_PATH"] = str(tool)
        return command, env_updates, removals, None
    if method == "neutrino":
        neutrino = shutil.which("neutrino")
        if neutrino is None:
            return command, env_updates, removals, "neutrino not found"
        command = [neutrino, "-p", "dmat", *command]
        return command, env_updates, removals, None
    raise ValueError(f"Unknown overhead method: {method}")


def run_overhead(args: argparse.Namespace) -> bool:
    logs_root = args.output_dir / "logs" / "overhead"
    ok = True
    tasks = selected_tasks(args)
    methods = selected_methods(args)
    total = len(tasks) * len(methods)
    done = 0
    wall_start = time.monotonic()
    for task in tasks:
        for method in methods:
            done += 1
            log_dir = logs_root / task.lower()
            log_path = log_dir / f"{method}.log"
            err_path = log_dir / f"{method}.err"
            if args.skip_existing and has_avg_log(log_path):
                total_elapsed = time.monotonic() - wall_start
                remaining = (total_elapsed / done) * (total - done) if done else 0.0
                print(
                    f"[AE][part3][Figure15][run {done}/{total}] SKIP existing task={task} method={method_label(method)} "
                    f"part_elapsed={format_duration(total_elapsed)} part_eta={format_duration(remaining)}",
                    flush=True,
                )
                continue
            command, env_updates, removals, skip_reason = overhead_command(args, task, method, log_dir)
            if skip_reason:
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path.write_text(f"# skipped: {skip_reason}\n", encoding="utf-8")
                err_path.write_text("", encoding="utf-8")
                print(
                    f"[AE][part3][Figure15][run {done}/{total}] SKIP task={task} method={method_label(method)} reason={reviewer_text(skip_reason)}",
                    flush=True,
                )
                continue
            print(f"[AE][part3][Figure15][run {done}/{total}] START task={task} method={method_label(method)}", flush=True)
            run_start = time.monotonic()
            env = build_child_env(args, env_updates, removals)
            rc = run_command(f"Figure15 {task}/{method_label(method)}", command, OVERHEAD_DIR, env, log_path, err_path, args.dry_run)
            run_elapsed = time.monotonic() - run_start
            total_elapsed = time.monotonic() - wall_start
            remaining = (total_elapsed / done) * (total - done) if done else 0.0
            print(
                f"[AE][part3][Figure15][run {done}/{total}] DONE task={task} method={method_label(method)} rc={rc} "
                f"run_elapsed={format_duration(run_elapsed)} "
                f"part_elapsed={format_duration(total_elapsed)} part_eta={format_duration(remaining)}",
                flush=True,
            )
            if rc != 0:
                ok = False
                print(f"[AE][part3][Figure15][run {done}/{total}] FAILED task={task} method={method_label(method)} rc={rc}", flush=True)
                if args.fail_fast:
                    return False
    return ok


def model_command(args: argparse.Namespace, series: str) -> tuple[list[str], Path, dict[str, str]]:
    model_env = default_smsched_model_env(args.smsched_model_env)
    env_updates = split_env_assignments(model_env)
    env_updates.update(
        {
            "SMSCHED": model_env,
            "SMSCHED_PROFILE": "1",
            "SMSCHED_PROFILE_LOG": "1",
            "SMSCHED_DEBUG": "0",
            "SMSCHED_TARGET_PATTERNS": "__smsched_no_target__",
            "SMSCHED_BENCH_WARMUP_ITERS": "2",
            "SMSCHED_BENCH_TIMED_ITERS": "5",
        }
    )
    if series == "ggnn":
        command = [
            args.python_bin,
            "single_request.py",
            "--workload",
            "ggnn",
            "--ggnn-base-size",
            "1000000",
            "--ggnn-query-size",
            "10000",
            "--runs",
            "1",
        ]
        return command, OVERHEAD_DIR, env_updates
    env_updates["SMSCHED_BENCH_M_SIZES"] = series
    env_updates["SMSCHED_BENCH_N_SIZES"] = "65536"
    env_updates["SMSCHED_BENCH_K_SIZES"] = "65536"
    return [str(BENCH_BIN_MODEL)], MICROBENCH_DIR, env_updates


def run_model(args: argparse.Namespace) -> bool:
    logs_root = args.output_dir / "logs" / "model"
    ok = True
    series_list = selected_model_series(args)
    wall_start = time.monotonic()
    for idx, series in enumerate(series_list, start=1):
        log_path = logs_root / f"{series}.log"
        err_path = logs_root / f"{series}.err"
        if args.skip_existing and has_profile_log(log_path):
            total_elapsed = time.monotonic() - wall_start
            remaining = (total_elapsed / idx) * (len(series_list) - idx) if idx else 0.0
            print(
                f"[AE][part4][Figure16][run {idx}/{len(series_list)}] SKIP existing series={series} "
                f"part_elapsed={format_duration(total_elapsed)} part_eta={format_duration(remaining)}",
                flush=True,
            )
            continue
        command, cwd, env_updates = model_command(args, series)
        print(f"[AE][part4][Figure16][run {idx}/{len(series_list)}] START series={series}", flush=True)
        run_start = time.monotonic()
        env = build_child_env(args, env_updates, include_mpi_stub=(series == "ggnn"), clean_base=True)
        rc = run_command(f"Figure16 {series}", command, cwd, env, log_path, err_path, args.dry_run)
        run_elapsed = time.monotonic() - run_start
        total_elapsed = time.monotonic() - wall_start
        remaining = (total_elapsed / idx) * (len(series_list) - idx) if idx else 0.0
        print(
            f"[AE][part4][Figure16][run {idx}/{len(series_list)}] DONE series={series} rc={rc} "
            f"run_elapsed={format_duration(run_elapsed)} "
            f"part_elapsed={format_duration(total_elapsed)} part_eta={format_duration(remaining)}",
            flush=True,
        )
        if rc != 0:
            if has_profile_log(log_path):
                print(
                    f"[AE][part4][Figure16][run {idx}/{len(series_list)}] WARNING series={series} "
                    f"rc={rc} but profile output is complete; continuing",
                    flush=True,
                )
            else:
                ok = False
                print(f"[AE][part4][Figure16][run {idx}/{len(series_list)}] FAILED series={series} rc={rc}", flush=True)
                if args.fail_fast:
                    return False
    return ok


def parse_avg_ms(path: Path) -> Optional[float]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    matches = AVG_MS_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def write_tsv(path: Path, headers: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: Optional[float], precision: int = 3) -> str:
    if value is None or math.isnan(value) or math.isinf(value):
        return "N/A"
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


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def analyze_overhead(args: argparse.Namespace) -> None:
    logs_root = args.output_dir / "logs" / "overhead"
    table_dir = args.output_dir / "tables"
    rows: list[dict[str, object]] = []
    for task in OVERHEAD_TASKS:
        values = {method: parse_avg_ms(logs_root / task.lower() / f"{method}.log") for method in OVERHEAD_METHODS}
        base = values["base"]
        row: dict[str, object] = {"Task": task}
        for method in OVERHEAD_METHODS:
            row[method_label(method)] = fmt(values[method])
        for method in OVERHEAD_METHODS:
            rel_key = f"{method_label(method)}(rel)"
            value = values[method]
            row[rel_key] = fmt(value / base if value is not None and base and base > 0 else None)
        rows.append(row)

    headers = ["Task", *(method_label(method) for method in OVERHEAD_METHODS), *(f"{method_label(method)}(rel)" for method in OVERHEAD_METHODS)]
    write_tsv(table_dir / "Figure15.tsv", headers, rows)


def parse_profile_blocks(path: Path) -> list[ProfileBlock]:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []
    blocks: list[ProfileBlock] = []
    current: Optional[dict[str, object]] = None
    waves: list[tuple[int, float]] = []
    for line in lines:
        kernel_match = PROFILE_KERNEL_RE.match(line)
        if kernel_match is None:
            kernel_match = PROFILE_KERNEL_SHORT_RE.match(line)
        if kernel_match:
            if current is not None:
                blocks.append(
                    ProfileBlock(
                        name=str(current["name"]),
                        blocks=int(current.get("blocks", 0)),
                        occup=int(current["occup"]),
                        waves_count=int(current.get("waves", len(waves))),
                        ci=float(current["ci"]),
                        wavetime_us=float(current["wavetime"]),
                        waves=tuple(waves),
                    )
                )
            current = kernel_match.groupdict()
            waves = []
            continue
        wave_match = PROFILE_WAVE_RE.match(line)
        if wave_match and current is not None:
            waves.append((int(wave_match.group("sms")), float(wave_match.group("lat"))))
            continue
        if current is not None and (line.startswith("[smsched] kernel") or line.startswith("[MorphX] kernel")):
            continue
    if current is not None:
        blocks.append(
            ProfileBlock(
                name=str(current["name"]),
                blocks=int(current.get("blocks", 0)),
                occup=int(current["occup"]),
                waves_count=int(current.get("waves", len(waves))),
                ci=float(current["ci"]),
                wavetime_us=float(current["wavetime"]),
                waves=tuple(waves),
            )
        )
    return blocks


def with_profile_source(block: ProfileBlock, source: str) -> ProfileBlock:
    return ProfileBlock(
        name=block.name,
        blocks=block.blocks,
        occup=block.occup,
        waves_count=block.waves_count,
        ci=block.ci,
        wavetime_us=block.wavetime_us,
        waves=block.waves,
        source=source,
    )


def parse_legacy_bench_profile(path: Path) -> dict[str, ProfileBlock]:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return {}

    mapped: dict[str, ProfileBlock] = {}
    current: Optional[dict[str, object]] = None
    waves: list[tuple[int, float]] = []
    pending_cutlass: Optional[ProfileBlock] = None

    def finish_current() -> None:
        nonlocal current, waves, pending_cutlass
        if current is None:
            return
        block = ProfileBlock(
            name=str(current["name"]),
            blocks=int(current["blocks"]),
            occup=int(current["occup"]),
            waves_count=int(current["waves"]),
            ci=float(current["ci"]),
            wavetime_us=float(current["wavetime"]),
            waves=tuple(waves),
            source=str(path),
        )
        if "cutlass" in block.name.lower() and block.waves:
            pending_cutlass = block
        current = None
        waves = []

    for line in lines:
        kernel_match = PROFILE_KERNEL_RE.match(line)
        if kernel_match:
            finish_current()
            current = kernel_match.groupdict()
            waves = []
            continue
        wave_match = PROFILE_WAVE_RE.match(line)
        if wave_match and current is not None:
            waves.append((int(wave_match.group("sms")), float(wave_match.group("lat"))))
            continue
        result_match = BENCH_RESULT_RE.match(line)
        if result_match:
            finish_current()
            m_value = result_match.group("m")
            if m_value in {"16", "128", "512", "8192"} and pending_cutlass is not None:
                mapped[m_value] = with_profile_source(pending_cutlass, str(path))
                pending_cutlass = None
    finish_current()
    return mapped


def fallback_model_block(series: str) -> Optional[ProfileBlock]:
    if series == "ggnn":
        path = REPO_ROOT / "scripts" / "logs" / "overheads" / "ggnn" / "smsched.log"
        selected = select_model_block(series, parse_profile_blocks(path))
        return with_profile_source(selected, str(path)) if selected is not None else None
    return parse_legacy_bench_profile(REPO_ROOT / "microbench" / "bench.log").get(series)


def expected_model_sms() -> dict[str, list[int]]:
    return {series: list(MODEL_SMS_POINTS[series]) for series in MODEL_SERIES}


def select_model_block(series: str, blocks: list[ProfileBlock]) -> Optional[ProfileBlock]:
    if series == "ggnn":
        candidates = [block for block in blocks if "ggnn3sym" in block.name or "ggnn::sym" in block.name]
        if candidates:
            return max(candidates, key=lambda block: len(block.waves))
        ggnn_blocks = [block for block in blocks if "ggnn" in block.name]
        if ggnn_blocks:
            return max(ggnn_blocks, key=lambda block: len(block.waves))
        return None
    cutlass_blocks = [block for block in blocks if "cutlass" in block.name.lower() and block.waves]
    if cutlass_blocks:
        return max(cutlass_blocks, key=lambda block: block.blocks)
    return None


def analyze_model(args: argparse.Namespace) -> None:
    table_dir = args.output_dir / "tables"
    logs_root = args.output_dir / "logs" / "model"
    expected = expected_model_sms()
    series_waves: dict[str, dict[int, float]] = {}
    metadata_rows: list[dict[str, object]] = []
    for series in MODEL_SERIES:
        blocks = parse_profile_blocks(logs_root / f"{series}.log")
        selected = select_model_block(series, blocks)
        if selected is None and args.allow_legacy_model_fallback:
            selected = fallback_model_block(series)
        if selected is None:
            series_waves[series] = {}
            metadata_rows.append({"series": series, "kernel": "N/A", "blocks": "N/A", "occup": "N/A", "ci": "N/A", "waves": 0, "source": "N/A"})
            continue
        series_waves[series] = {sms: lat for sms, lat in selected.waves}
        metadata_rows.append(
            {
                "series": series,
                "kernel": selected.name,
                "blocks": selected.blocks,
                "occup": selected.occup,
                "ci": fmt(selected.ci, 6),
                "waves": len(selected.waves),
                "source": selected.source,
            }
        )

    max_rows = max((len(expected[series]) for series in MODEL_SERIES), default=0)
    headers: list[str] = []
    for series in MODEL_SERIES:
        headers.extend([f"sm-{series}", f"lat-{series}", f"lat-{series}-ms"])
    rows: list[dict[str, object]] = []
    for idx in range(max_rows):
        row: dict[str, object] = {}
        for series in MODEL_SERIES:
            sms_values = expected[series]
            if idx >= len(sms_values):
                row[f"sm-{series}"] = ""
                row[f"lat-{series}"] = ""
                row[f"lat-{series}-ms"] = ""
                continue
            sms = sms_values[idx]
            lat_us = series_waves.get(series, {}).get(sms)
            row[f"sm-{series}"] = sms
            row[f"lat-{series}"] = fmt(lat_us, 6) if lat_us is not None else "N/A"
            row[f"lat-{series}-ms"] = fmt(lat_us / 1000.0, 9) if lat_us is not None else "N/A"
        rows.append(row)
    write_tsv(table_dir / "Figure16.tsv", headers, rows)
    write_tsv(table_dir / "Figure16-profile-metadata.tsv", ["series", "kernel", "blocks", "occup", "ci", "waves", "source"], metadata_rows)


def analyze(args: argparse.Namespace) -> None:
    sections = selected_sections(args)
    if "overhead" in sections:
        analyze_overhead(args)
    if "model" in sections:
        analyze_model(args)


def run_build(label: str, command: list[str], cwd: Path) -> bool:
    print(f"[AE][build] {label}", flush=True)
    result = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        combined = (result.stdout or "") + (result.stderr or "")
        tail = "\n".join(SPLIT_NAME_RE.sub("tuning", reviewer_text(combined)).splitlines()[-40:])
        if tail:
            print(f"[AE][build][tail]\n{tail}", file=sys.stderr, flush=True)
        print(f"[AE][build] FAILED {label}: return code {result.returncode}", flush=True)
        return False
    print(f"[AE][build] DONE {label}", flush=True)
    return True


def build_binaries(args: argparse.Namespace) -> bool:
    if args.no_build or args.dry_run:
        return True
    ok = True
    sections = selected_sections(args)
    runtime_build = REPO_ROOT / "runtime" / "build"
    if runtime_build.is_dir():
        ok = run_build("runtime cuda/preload", ["cmake", "--build", str(runtime_build), "--target", "cuda", "preload", "-j"], REPO_ROOT) and ok
    else:
        print(f"[AE][build] skip runtime: build directory not found: {runtime_build}", flush=True)
    if "model" in sections:
        smsched_pass_build = REPO_ROOT / "smsched-pass" / "build"
        if smsched_pass_build.is_dir():
            ok = run_build("MorphX compiler pass", ["cmake", "--build", str(smsched_pass_build), "-j"], REPO_ROOT) and ok
        else:
            print(f"[AE][build] skip MorphX compiler pass: build directory not found", flush=True)
        if not MICROBENCH_MODEL_BUILD.is_dir():
            ok = run_build(
                "microbench model configure",
                ["cmake", "-S", str(MICROBENCH_DIR), "-B", str(MICROBENCH_MODEL_BUILD), "-DSMSCHED_ENABLE_PASS=ON"],
                REPO_ROOT,
            ) and ok
        if MICROBENCH_MODEL_BUILD.is_dir():
            ok = run_build("microbench model bench", ["cmake", "--build", str(MICROBENCH_MODEL_BUILD), "--target", "bench", "-j"], REPO_ROOT) and ok
        else:
            print(f"[AE][build] skip microbench model: build directory not found: {MICROBENCH_MODEL_BUILD}", flush=True)
    if MICROBENCH_BUILD.is_dir():
        ok = run_build("microbench bench", ["cmake", "--build", str(MICROBENCH_BUILD), "--target", "bench", "-j"], REPO_ROOT) and ok
    else:
        print(f"[AE][build] skip microbench: build directory not found: {MICROBENCH_BUILD}", flush=True)
    return ok


def preflight(args: argparse.Namespace) -> list[str]:
    warnings: list[str] = []
    if not SINGLE_REQUEST.exists():
        warnings.append(f"Missing overhead entry script: {SINGLE_REQUEST}")
    if "model" in selected_sections(args) and args.no_build and not BENCH_BIN_MODEL.exists():
        warnings.append(f"Missing model microbench bench binary: {BENCH_BIN_MODEL}")
    smsched_envs = [split_env_assignments(default_smsched_env(args.smsched_env))]
    if "model" in selected_sections(args):
        smsched_envs.append(split_env_assignments(default_smsched_model_env(args.smsched_model_env)))
    for smsched_env in smsched_envs:
        for preload_path in smsched_env.get("LD_PRELOAD", "").split(":"):
            if preload_path and not Path(preload_path).exists():
                warnings.append(f"MorphX LD_PRELOAD target is missing: {preload_path}")
        hook_driver = smsched_env.get("NEUTRINO_HOOK_DRIVER", "")
        if hook_driver and not Path(hook_driver).exists():
            warnings.append(f"MorphX NEUTRINO_HOOK_DRIVER target is missing: {hook_driver}")
    if shutil.which("ncu") is None:
        warnings.append("ncu not found; Figure15 ncu values will be N/A")
    if not (REPO_ROOT / "nvbit-tutorial" / "tools" / "mem_trace" / "mem_trace.so").exists():
        warnings.append("NvBit mem_trace.so not found; Figure15 nvbit values will be N/A")
    if shutil.which("neutrino") is None:
        warnings.append("neutrino not found; Figure15 neutrino values will be N/A")
    return warnings


def update_latest_links(args: argparse.Namespace) -> None:
    if args.no_update_latest_link:
        return
    result_root = REPO_ROOT / "ae-results" / "overhead-model-part4"
    result_root.mkdir(parents=True, exist_ok=True)
    latest = result_root / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(args.output_dir, target_is_directory=True)

    tables_root = REPO_ROOT / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    tables_latest = tables_root / "overhead-model-part4-latest"
    if tables_latest.exists() or tables_latest.is_symlink():
        tables_latest.unlink()
    tables_latest.symlink_to(args.output_dir / "tables", target_is_directory=True)


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.effective_cuda_visible_devices = resolve_cuda_visible_devices(args)
    args.runtime_library_dirs = prepare_runtime_library_dirs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = section_prefix(args)

    print(f"{prefix}[setup] output_dir={args.output_dir}", flush=True)
    if args.effective_cuda_visible_devices:
        print(f"{prefix}[setup] CUDA_VISIBLE_DEVICES={args.effective_cuda_visible_devices}", flush=True)
    for warning in preflight(args):
        print(f"{prefix}[preflight] WARNING: {warning}", flush=True)
    if args.preflight_only:
        return 0

    if args.analyze_only:
        analyze(args)
        update_latest_links(args)
        print(f"{prefix}[analysis] analyze_only_tables={args.output_dir / 'tables'}", flush=True)
        return 0

    if not build_binaries(args):
        return 1

    ok = True
    sections = selected_sections(args)
    if "overhead" in sections:
        ok = run_overhead(args) and ok
    if "model" in sections:
        ok = run_model(args) and ok

    if not args.no_analyze:
        analyze(args)
        update_latest_links(args)
        print(f"{prefix}[analysis] tables={args.output_dir / 'tables'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
