#!/usr/bin/env python3
"""Run SOSP 2026 AE part 2: large-model PD co-location.

The runner uses built-in PD-online workload points for Figures 13 and 14,
rewrites them into temporary workload configs, runs test-pd-online.py for
baseline/smsched/stream/chunked, and emits fresh tables from the resulting logs.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PD_SCRIPT = REPO_ROOT / "scripts" / "test-pd-online.py"

DEFAULT_DURATION_SECONDS = 180.0
DEFAULT_MODEL = "llama-3-8b"
DEFAULT_REQUEST_TYPE = "poisson"

VARIANTS = ("baseline", "smsched", "stream", "chunked")
METRICS = ("avg", "p99", "ttft_avg", "ttft_p99")
SYSTEM_NAME_RE = re.compile("smsched", re.IGNORECASE)
SMSCHED_CONDA_ENV_RE = re.compile(r"(?<=/envs/)smsched(?=/)", re.IGNORECASE)
VARIANT_LABELS = {"smsched": "MorphX"}

WORKLOAD_CONFIGS = {
    "Figure13": (
        "in-2048-out-256-rps-1-split-76",
        "in-2048-out-256-rps-2-split-76",
        "in-2048-out-256-rps-3-split-76",
        "in-2048-out-256-rps-4-split-76",
        "in-2048-out-256-rps-5-split-76",
        "in-4096-out-128-rps-1-split-88",
        "in-4096-out-128-rps-1.5-split-88",
        "in-4096-out-128-rps-2-split-88",
        "in-4096-out-128-rps-2.5-split-88",
        "in-4096-out-128-rps-3-split-88",
        "in-4096-out-256-rps-1-split-88",
        "in-4096-out-256-rps-1.5-split-88",
        "in-4096-out-256-rps-2-split-88",
        "in-4096-out-256-rps-2.5-split-88",
        "in-4096-out-256-rps-3-split-88",
    ),
    "Figure14": (
        "in-2048-out-256-rps-1-split-76",
        "in-2048-out-256-rps-1.5-split-76",
        "in-2048-out-256-rps-2-split-76",
        "in-2048-out-256-rps-2.5-split-76",
        "in-2048-out-256-rps-3-split-76",
        "in-4096-out-128-rps-0.4-split-76",
        "in-4096-out-128-rps-0.8-split-76",
        "in-4096-out-128-rps-1.2-split-76",
        "in-4096-out-128-rps-1.6-split-76",
        "in-4096-out-128-rps-2-split-76",
        "in-4096-out-256-rps-0.4-split-76",
        "in-4096-out-256-rps-0.8-split-76",
        "in-4096-out-256-rps-1.2-split-76",
        "in-4096-out-256-rps-1.6-split-76",
        "in-4096-out-256-rps-2-split-76",
    ),
}

CONFIG_PATTERN = re.compile(r"^in-(\d+)-out-(\d+)-rps-([0-9.]+)-split-([\w.-]+)$")
LATENCY_PATTERN = re.compile(r"Latency:\s*([0-9.+-eE]+)\s*([a-zA-Z]*)", re.IGNORECASE)
TTFT_PATTERN = re.compile(r"TTFT:\s*([0-9.+-eE]+)\s*([a-zA-Z]*)", re.IGNORECASE)
TOKENS_PATTERN = re.compile(r"Total generated tokens:\s*(\d+)", re.IGNORECASE)

MODEL_PATHS = {
    "llama-3-8b": Path(
        "/huggingface-cache/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/"
        "snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
    ),
}


@dataclass(frozen=True)
class WorkloadRow:
    table: str
    order: int
    row_id: str
    config: str
    output_config: str
    input_length: int
    output_length: int
    rps: float
    rps_text: str
    split_hint: str


@dataclass(frozen=True)
class RunSpec:
    row: WorkloadRow
    variant: str
    split_hint: str
    config_path: Path
    command: list[str]
    env_updates: dict[str, str]
    env_removals: tuple[str, ...]
    log_path: Path
    err_path: Path


MetricData = dict[str, float | int]


def parse_args() -> argparse.Namespace:
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    default_output = REPO_ROOT / "ae-results" / "pd-colocate-part2" / timestamp
    default_python = os.environ.get("PYTHON")
    if not default_python:
        conda_python = Path("/home/rtx/miniconda3/envs/smsched/bin/python")
        default_python = str(conda_python if conda_python.exists() else Path(sys.executable))

    parser = argparse.ArgumentParser(
        description="Run and summarize SOSP 2026 AE part 2 PD co-location experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--python", dest="python_bin", default=default_python)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--request-type", default=DEFAULT_REQUEST_TYPE)
    parser.add_argument(
        "--smsched-env",
        default=os.environ.get("SMSCHED"),
        help=(
            "Shell-style environment assignments for smsched. If omitted, "
            "LD_PRELOAD=<repo>/runtime/build/libcuda.so:<repo>/runtime/build/libpreload.so is used."
        ),
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
        "--only-table",
        action="append",
        choices=tuple(WORKLOAD_CONFIGS.keys()),
        help="Limit to one output table. Repeat to select multiple tables.",
    )
    parser.add_argument(
        "--only-variant",
        action="append",
        choices=VARIANTS,
        help="Limit to one variant. Repeat to select multiple variants.",
    )
    parser.add_argument(
        "--only-config",
        action="append",
        help=(
            "Limit to selected table configs. Repeat to select multiple rows. "
            "Accepts either the config with split-* or the output config without split-*; "
            "prefix with the table name to disambiguate, e.g. Figure14:in-2048-out-256-rps-3."
        ),
    )
    parser.add_argument(
        "--override-split-hint",
        help="Use this split hint for all smsched runs instead of each workload's default split.",
    )
    parser.add_argument(
        "--split-hint-for",
        action="append",
        default=[],
        metavar="SELECTOR=SPLIT",
        help=(
            "Override the smsched split hint for matching configs. SELECTOR may be "
            "a config with split-*, an output config without split-*, or either prefixed "
            "with the table name, e.g. Figure14:in-4096-out-128-rps-*=60. "
            "Shell-style wildcards are supported; later rules win. "
            "--override-split-hint, when set, still overrides all rows."
        ),
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Stop after this many child runs.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip logs that already contain final output.")
    parser.add_argument("--fail-fast", action="store_true", help="Abort on the first failed child run.")
    parser.add_argument(
        "--preserve-ld-preload",
        action="store_true",
        help="Keep an inherited LD_PRELOAD for non-smsched runs.",
    )
    parser.add_argument(
        "--no-mpi-cxx-stub",
        action="store_true",
        help=(
            "Do not build an empty libmpi_cxx.so.40 shim when this PyTorch build "
            "declares but does not use the removed OpenMPI C++ binding library."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Record commands without running them.")
    parser.add_argument("--preflight-only", action="store_true", help="Run preflight checks and exit.")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze existing logs in --output-dir.")
    parser.add_argument("--no-analyze", action="store_true", help="Do not build tables after experiments.")
    parser.add_argument(
        "--no-update-latest-link",
        action="store_true",
        help="Do not update ae-results/pd-colocate-part2/latest and tables/pd-colocate-part2-latest symlinks.",
    )
    return parser.parse_args()


def format_rps_for_name(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def format_table_number(value: Optional[float], precision: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return f"{float(value):.{precision}f}"


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


def variant_label(variant: str) -> str:
    return VARIANT_LABELS.get(variant, variant)


def parse_float(value: str) -> Optional[float]:
    stripped = value.strip()
    if not stripped or stripped == "N/A":
        return None
    try:
        parsed = float(stripped)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def write_tsv(path: Path, headers: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def figure_scope_for_rows(rows: Sequence[WorkloadRow]) -> str:
    tables = {row.table for row in rows}
    if tables == {"Figure13"}:
        return "Figure13"
    if tables == {"Figure14"}:
        return "Figure14"
    return "Figures13-14"


def parse_workload_rows(table: str, configs: Sequence[str]) -> list[WorkloadRow]:
    rows: list[WorkloadRow] = []
    for order, config in enumerate(configs, start=1):
        match = CONFIG_PATTERN.match(config)
        if not match:
            raise ValueError(f"{table}: invalid config string: {config!r}")
        input_length = int(match.group(1))
        output_length = int(match.group(2))
        rps_text = match.group(3)
        rps = float(rps_text)
        split_hint = match.group(4)
        normalized_rps_text = format_rps_for_name(rps)
        output_config = f"in-{input_length}-out-{output_length}-rps-{normalized_rps_text}"
        rows.append(
            WorkloadRow(
                table=table,
                order=order,
                row_id=str(order),
                config=config,
                output_config=output_config,
                input_length=input_length,
                output_length=output_length,
                rps=rps,
                rps_text=normalized_rps_text,
                split_hint=split_hint,
            )
        )
    return rows


def load_selected_rows(args: argparse.Namespace) -> list[WorkloadRow]:
    selected_tables = args.only_table or list(WORKLOAD_CONFIGS)
    rows: list[WorkloadRow] = []
    for table in selected_tables:
        rows.extend(parse_workload_rows(table, WORKLOAD_CONFIGS[table]))
    if args.only_config:
        selected_configs = tuple(args.only_config)
        rows = [
            row
            for row in rows
            if any(row_matches_selector(row, selector) for selector in selected_configs)
        ]
        if not rows:
            raise ValueError(
                "--only-config matched no rows: "
                + ", ".join(selected_configs)
            )
    return rows


def parse_split_hint_overrides(values: Sequence[str]) -> tuple[tuple[str, str], ...]:
    rules: list[tuple[str, str]] = []
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"--split-hint-for must look like SELECTOR=SPLIT: {raw!r}")
        selector, split_hint = raw.split("=", 1)
        selector = selector.strip()
        split_hint = split_hint.strip()
        if not selector or not split_hint:
            raise ValueError(f"--split-hint-for must include both selector and split: {raw!r}")
        rules.append((selector, split_hint))
    return tuple(rules)


def split_hint_candidates(row: WorkloadRow) -> tuple[str, ...]:
    return (
        row.config,
        row.output_config,
        f"{row.table}:{row.config}",
        f"{row.table}:{row.output_config}",
    )


def row_matches_selector(row: WorkloadRow, selector: str) -> bool:
    return any(fnmatch.fnmatchcase(candidate, selector) for candidate in split_hint_candidates(row))


def split_hint_for_row(args: argparse.Namespace, row: WorkloadRow) -> str:
    split_hint = row.split_hint
    for selector, override in args.split_hint_overrides:
        if row_matches_selector(row, selector):
            split_hint = override
    if args.override_split_hint:
        split_hint = args.override_split_hint
    return split_hint


def unmatched_split_hint_selectors(args: argparse.Namespace, rows: Sequence[WorkloadRow]) -> list[str]:
    unmatched: list[str] = []
    for selector, _ in args.split_hint_overrides:
        if not any(row_matches_selector(row, selector) for row in rows):
            unmatched.append(selector)
    return unmatched


def selected_variants(args: argparse.Namespace) -> tuple[str, ...]:
    if not args.only_variant:
        return VARIANTS
    allowed = set(args.only_variant)
    return tuple(variant for variant in VARIANTS if variant in allowed)


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
    if raw is None or raw == "":
        return None
    if raw.lower() == "auto":
        return choose_least_used_gpu()
    return raw


def short_mps_pipe_base(output_dir: Path) -> Path:
    digest = hashlib.sha1(str(output_dir.resolve()).encode("utf-8")).hexdigest()[:12]
    return Path("/tmp") / f"morphx-ae-mps-{os.getuid()}-{digest}" / "part2"


def prepare_plain_mps_dirs(output_dir: Path) -> tuple[Path, Path]:
    pipe_dir = short_mps_pipe_base(output_dir) / "plain"
    log_dir = output_dir / "mps" / "plain-log"
    for directory in (pipe_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
        try:
            directory.chmod(0o777)
        except OSError:
            pass
    return pipe_dir, log_dir


def common_env_updates(args: argparse.Namespace) -> dict[str, str]:
    updates: dict[str, str] = {}
    ld_library_path = os.environ.get("LD_LIBRARY_PATH")
    for lib_dir in reversed(list(getattr(args, "runtime_library_dirs", []))):
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


def default_smsched_env(user_value: Optional[str]) -> str:
    if user_value:
        return user_value
    cuda_hook = REPO_ROOT / "runtime" / "build" / "libcuda.so"
    preload = REPO_ROOT / "runtime" / "build" / "libpreload.so"
    return f"LD_PRELOAD={cuda_hook}:{preload}"


def write_workload_configs(args: argparse.Namespace, rows: Sequence[WorkloadRow], output_dir: Path) -> dict[tuple[str, str], Path]:
    config_paths: dict[tuple[str, str], Path] = {}
    for row in rows:
        config_path = output_dir / "configs" / row.table / f"{row.config}.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "time": args.duration,
            "model": args.model,
            "type": args.request_type,
            "rps": row.rps,
            "input_length": row.input_length,
            "output_length": row.output_length,
        }
        config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        config_paths[(row.table, row.config)] = config_path
    return config_paths


def build_run_spec(
    args: argparse.Namespace,
    logs_root: Path,
    row: WorkloadRow,
    variant: str,
    config_path: Path,
) -> RunSpec:
    command = [args.python_bin, str(PD_SCRIPT), str(config_path)]
    env_updates = common_env_updates(args)
    env_removals: list[str] = ["SMSCHED", "SMSCHED_SPLIT_HINT"]
    split_hint = split_hint_for_row(args, row)

    if variant == "baseline":
        env_removals.append("VLLM_PREFILL_THREAD")
    elif variant == "stream":
        env_updates["VLLM_PREFILL_THREAD"] = "1"
    elif variant == "smsched":
        env_updates.update(split_env_assignments(default_smsched_env(args.smsched_env)))
        env_updates["SMSCHED"] = default_smsched_env(args.smsched_env)
        env_updates["SMSCHED_SPLIT_HINT"] = split_hint
        env_updates["VLLM_PREFILL_THREAD"] = "1"
    elif variant == "chunked":
        command.append("--enable-chunked-prefill")
        env_removals.append("VLLM_PREFILL_THREAD")
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    if variant != "smsched" and not args.preserve_ld_preload:
        env_removals.append("LD_PRELOAD")

    log_dir = logs_root / row.table / row.config
    return RunSpec(
        row=row,
        variant=variant,
        split_hint=split_hint,
        config_path=config_path,
        command=command,
        env_updates=env_updates,
        env_removals=tuple(dict.fromkeys(env_removals)),
        log_path=log_dir / f"{variant}.log",
        err_path=log_dir / f"{variant}.err",
    )


def has_complete_log(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "[final] Total requests served:" in text and "[final] Throughput:" in text


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


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: Optional[dict[str, str]] = None,
    stdout_path: Optional[Path] = None,
    stderr_path: Optional[Path] = None,
    timeout: Optional[float] = None,
) -> subprocess.CompletedProcess[str]:
    stdout_target = subprocess.PIPE
    stderr_target = subprocess.PIPE
    stdout_handle = None
    stderr_handle = None
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_handle = stdout_path.open("a", encoding="utf-8")
        stdout_target = stdout_handle
    if stderr_path is not None:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_handle = stderr_path.open("a", encoding="utf-8")
        stderr_target = stderr_handle
    try:
        return subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=stdout_target,
            stderr=stderr_target,
            text=True,
            timeout=timeout,
            check=False,
        )
    finally:
        if stdout_handle is not None:
            stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()


def build_env(spec: RunSpec) -> dict[str, str]:
    env = os.environ.copy()
    for key in spec.env_removals:
        env.pop(key, None)
    env.update(spec.env_updates)
    return env


def run_spec(args: argparse.Namespace, spec: RunSpec) -> dict[str, object]:
    started_at = dt.datetime.now().isoformat(timespec="seconds")
    start = time.time()
    status = "ok"
    returncode: Optional[int] = None
    skipped = False

    if args.skip_existing and has_complete_log(spec.log_path):
        return {
            "status": "skipped",
            "returncode": returncode,
            "elapsed_seconds": 0.0,
            "started_at": started_at,
            "skipped": True,
        }

    remove_incomplete_log_pair(spec.log_path, spec.err_path)

    if args.dry_run:
        append_text(spec.log_path, f"[AE dry-run] {quote_cmd(spec.command)}\n")
        return {
            "status": "dry-run",
            "returncode": returncode,
            "elapsed_seconds": time.time() - start,
            "started_at": started_at,
            "skipped": skipped,
        }

    env = build_env(spec)
    append_text(spec.err_path, f"\n[AE] command: {quote_cmd(spec.command)}\n")
    append_text(spec.err_path, f"[AE] cwd: {PD_SCRIPT.parent}\n")
    append_text(spec.err_path, f"[AE] config: {spec.config_path}\n")
    if spec.env_updates:
        visible_env = {key: spec.env_updates[key] for key in sorted(spec.env_updates)}
        append_text(spec.err_path, f"[AE] env updates: {json.dumps(visible_env, sort_keys=True)}\n")
    if spec.env_removals:
        append_text(spec.err_path, f"[AE] env removals: {', '.join(spec.env_removals)}\n")

    result = run_command(
        spec.command,
        cwd=PD_SCRIPT.parent,
        env=env,
        stdout_path=spec.log_path,
        stderr_path=spec.err_path,
    )
    returncode = result.returncode
    if returncode != 0:
        status = "failed"
    elif not has_complete_log(spec.log_path):
        status = "incomplete"

    return {
        "status": status,
        "returncode": returncode,
        "elapsed_seconds": time.time() - start,
        "started_at": started_at,
        "skipped": skipped,
    }


def write_manifest_header(path: Path, args: argparse.Namespace, rows: Sequence[WorkloadRow], logs_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "logs_root": str(logs_root),
        "duration_seconds": args.duration,
        "model": args.model,
        "request_type": args.request_type,
        "tables": args.only_table or list(WORKLOAD_CONFIGS),
        "variants": [variant_label(variant) for variant in selected_variants(args)],
        "workload_rows": len(rows),
        "note": "output config names strip the split-* suffix as requested",
    }
    path.write_text(reviewer_text(json.dumps(payload, indent=2, sort_keys=True)) + "\n", encoding="utf-8")


def append_command_row(path: Path, spec: RunSpec, result: Optional[dict[str, object]] = None) -> None:
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "table",
            "row_id",
            "config",
            "output_config",
            "variant",
            "rps",
            "input_length",
            "output_length",
            "split_hint",
            "status",
            "returncode",
            "elapsed_seconds",
            "env_updates",
            "env_removals",
            "command",
            "config_path",
            "log",
            "err",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        if write_header:
            writer.writeheader()
        result = result or {}
        elapsed = result.get("elapsed_seconds")
        writer.writerow(
            {
                "table": spec.row.table,
                "row_id": spec.row.row_id,
                "config": spec.row.config,
                "output_config": spec.row.output_config,
                "variant": variant_label(spec.variant),
                "rps": format_table_number(spec.row.rps),
                "input_length": spec.row.input_length,
                "output_length": spec.row.output_length,
                "split_hint": spec.split_hint,
                "status": result.get("status", "planned"),
                "returncode": result.get("returncode", ""),
                "elapsed_seconds": format_table_number(float(elapsed), precision=2)
                if isinstance(elapsed, (float, int))
                else "",
                "env_updates": reviewer_text(json.dumps(spec.env_updates, sort_keys=True)),
                "env_removals": reviewer_text(",".join(spec.env_removals)),
                "command": reviewer_text(quote_cmd(spec.command)),
                "config_path": reviewer_text(spec.config_path),
                "log": reviewer_text(spec.log_path),
                "err": reviewer_text(spec.err_path),
            }
        )


def convert_time_to_seconds(raw_value: float, unit: str) -> float:
    normalized = unit.lower()
    if normalized in ("ms", "millisecond", "milliseconds"):
        return raw_value / 1000.0
    if normalized in ("us", "microsecond", "microseconds"):
        return raw_value / 1_000_000.0
    if normalized in ("", "s", "sec", "secs", "second", "seconds"):
        return raw_value
    if normalized in ("m", "min", "mins", "minute", "minutes"):
        return raw_value * 60.0
    raise ValueError(f"Unsupported time unit: {unit}")


def extract_metrics(log_path: Path) -> list[MetricData]:
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    metrics: list[MetricData] = []
    current_ttft: Optional[float] = None
    current_latency: Optional[float] = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "TTFT:" in line:
                match = TTFT_PATTERN.search(line)
                if match:
                    try:
                        current_ttft = convert_time_to_seconds(float(match.group(1)), match.group(2) or "")
                    except ValueError:
                        current_ttft = None
            if "Latency:" in line:
                match = LATENCY_PATTERN.search(line)
                if match:
                    try:
                        current_latency = convert_time_to_seconds(float(match.group(1)), match.group(2) or "")
                    except ValueError:
                        current_latency = None
            if "Total generated tokens:" in line:
                match = TOKENS_PATTERN.search(line)
                if not match:
                    continue
                tokens = int(match.group(1))
                if current_ttft is not None and current_latency is not None and tokens > 0:
                    metrics.append(
                        {
                            "tbt": current_latency / tokens,
                            "tokens": tokens,
                            "ttft": current_ttft,
                        }
                    )
                current_ttft = None
                current_latency = None

    if not metrics:
        raise ValueError(f"No valid metrics found in {log_path}")
    return metrics


def compute_weighted_average(values: Sequence[tuple[float, int]]) -> float:
    total_count = sum(count for _, count in values)
    if total_count == 0:
        return 0.0
    return sum(value * count for value, count in values) / total_count


def compute_weighted_percentile(values: Sequence[tuple[float, int]], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of an empty sequence")
    total_count = sum(count for _, count in values)
    if total_count == 0:
        return 0.0
    target = total_count * percentile
    running = 0
    for value, count in sorted(values, key=lambda item: item[0]):
        running += count
        if running >= target:
            return value
    return sorted(values, key=lambda item: item[0])[-1][0]


def compute_percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of an empty sequence")
    sorted_values = sorted(values)
    if len(sorted_values) == 1 or percentile == 0.0:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * percentile
    lower_idx = math.floor(index)
    upper_idx = math.ceil(index)
    if lower_idx == upper_idx:
        return sorted_values[lower_idx]
    lower_val = sorted_values[lower_idx]
    upper_val = sorted_values[upper_idx]
    weight = index - lower_idx
    return lower_val + (upper_val - lower_val) * weight


def compute_metrics(log_path: Path) -> dict[str, float]:
    metrics = extract_metrics(log_path)
    tbt_data = [(float(item["tbt"]), int(item["tokens"])) for item in metrics]
    ttft_data = [float(item["ttft"]) for item in metrics]
    return {
        "avg": compute_weighted_average(tbt_data),
        "p99": compute_weighted_percentile(tbt_data, 0.99),
        "ttft_avg": sum(ttft_data) / len(ttft_data),
        "ttft_p99": compute_percentile(ttft_data, 0.99),
    }


def metric_column(variant: str, metric: str) -> str:
    label = variant_label(variant)
    if metric == "avg":
        return f"{label}(avg)"
    if metric == "p99":
        return f"{label}(p99)"
    if metric == "ttft_avg":
        return f"{label}(ttft_avg)"
    if metric == "ttft_p99":
        return f"{label}(ttft_p99)"
    raise ValueError(metric)


def table_headers() -> list[str]:
    headers = ["config", "rps"]
    for metric in METRICS:
        for variant in VARIANTS:
            headers.append(metric_column(variant, metric))
    for other in ("baseline", "stream", "chunked"):
        headers.append(f"vs_{other}_avg(%)")
        headers.append(f"vs_{other}_p99(%)")
    return headers


def build_result_tables(rows: Sequence[WorkloadRow], logs_root: Path, tables_dir: Path) -> dict[str, list[dict[str, object]]]:
    by_table: dict[str, list[WorkloadRow]] = {}
    for row in rows:
        by_table.setdefault(row.table, []).append(row)

    outputs: dict[str, list[dict[str, object]]] = {}
    headers = table_headers()
    for table, table_rows in by_table.items():
        rendered: list[dict[str, object]] = []
        for row in sorted(table_rows, key=lambda item: item.order):
            out: dict[str, object] = {
                "config": row.output_config,
                "rps": format_table_number(row.rps),
            }
            metrics_by_variant: dict[str, dict[str, float]] = {}
            for variant in VARIANTS:
                log_path = logs_root / row.table / row.config / f"{variant}.log"
                try:
                    metrics_by_variant[variant] = compute_metrics(log_path)
                except (OSError, ValueError) as exc:
                    out[f"{variant}_parse_error"] = str(exc)

            for metric in METRICS:
                for variant in VARIANTS:
                    value = metrics_by_variant.get(variant, {}).get(metric)
                    out[metric_column(variant, metric)] = format_table_number(value)

            smsched = metrics_by_variant.get("smsched")
            for other in ("baseline", "stream", "chunked"):
                other_metrics = metrics_by_variant.get(other)
                for metric in ("avg", "p99"):
                    column = f"vs_{other}_{metric}(%)"
                    value: Optional[float] = None
                    if smsched and other_metrics and other_metrics.get(metric, 0.0) > 0:
                        value = (other_metrics[metric] - smsched[metric]) / other_metrics[metric] * 100.0
                    out[column] = format_table_number(value)
            rendered.append(out)
        write_tsv(tables_dir / f"{table}.tsv", headers, rendered)
        outputs[table] = rendered
    return outputs


def preflight(args: argparse.Namespace, output_dir: Path) -> tuple[bool, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    python_path = shutil.which(args.python_bin) if os.sep not in args.python_bin else args.python_bin
    if not python_path or not Path(python_path).exists():
        errors.append(f"Python not found: {args.python_bin}")

    required_paths = [PD_SCRIPT]
    for path in required_paths:
        if not path.exists():
            errors.append(f"Missing required path: {path}")

    model_path = MODEL_PATHS.get(args.model)
    if model_path and not model_path.exists():
        warnings.append(f"Model path from test-pd-online.py is missing: {model_path}")

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

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run([nvidia_smi, "-L"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20)
            if result.returncode != 0:
                warnings.append(f"nvidia-smi -L failed: {result.stderr.strip()}")
        except subprocess.SubprocessError as exc:
            warnings.append(f"nvidia-smi -L failed: {exc}")
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


def safe_update_symlink(link: Path, target: Path) -> Optional[str]:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        if link.is_symlink() or link.is_file():
            link.unlink()
        else:
            return f"Refusing to replace existing directory: {link}"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        return f"Failed to update symlink {link}: {exc}"
    return None


def update_latest_links(args: argparse.Namespace, output_dir: Path) -> list[str]:
    if args.no_update_latest_link:
        return []
    warnings: list[str] = []
    result_link = REPO_ROOT / "ae-results" / "pd-colocate-part2" / "latest"
    tables_link = REPO_ROOT / "tables" / "pd-colocate-part2-latest"
    for link, target in ((result_link, output_dir), (tables_link, output_dir / "tables")):
        warning = safe_update_symlink(link, target)
        if warning:
            warnings.append(warning)
    return warnings


def main() -> int:
    args = parse_args()
    args.split_hint_overrides = parse_split_hint_overrides(args.split_hint_for)
    output_dir = args.output_dir.resolve()
    logs_root = output_dir / "logs" / "pd-online"
    manifest_path = output_dir / "manifest.json"
    commands_path = output_dir / "commands.tsv"

    print(f"[AE][part2][Figures13-14][setup] output_dir={output_dir}", flush=True)
    print("[AE][part2][Figures13-14][setup] PD-online workload duration is set by --duration; default is 180s.", flush=True)

    args.runtime_library_dirs = prepare_runtime_library_dirs(args, output_dir)
    args.effective_cuda_visible_devices = resolve_cuda_visible_devices(args)
    args.cuda_plain_mps_pipe_dir, args.cuda_plain_mps_log_dir = prepare_plain_mps_dirs(output_dir)

    if args.runtime_library_dirs:
        print(
            "[AE][part2][Figures13-14][setup] runtime_library_dirs="
            + reviewer_text(":".join(str(path) for path in args.runtime_library_dirs)),
            flush=True,
        )
    if args.effective_cuda_visible_devices:
        print(f"[AE][part2][Figures13-14][setup] CUDA_VISIBLE_DEVICES={args.effective_cuda_visible_devices}", flush=True)
    print(f"[AE][part2][Figures13-14][setup] cuda_pipe_dir={args.cuda_plain_mps_pipe_dir}", flush=True)

    ok, errors, warnings = preflight(args, output_dir)
    for warning in warnings:
        print(f"[AE][part2][Figures13-14][warning] {warning}", flush=True)
    for error in errors:
        print(f"[AE][part2][Figures13-14][error] {error}", flush=True)
    if not ok:
        return 2
    if args.preflight_only:
        print("[AE][part2][Figures13-14][setup] preflight complete.", flush=True)
        return 0

    rows = load_selected_rows(args)
    figure_scope = figure_scope_for_rows(rows)
    if args.analyze_only:
        tables_dir = output_dir / "tables"
        build_result_tables(rows, logs_root, tables_dir)
        for warning in update_latest_links(args, output_dir):
            print(f"[AE][part2][{figure_scope}][warning] {warning}", flush=True)
        print(f"[AE][part2][{figure_scope}][analysis] analyze_only_generated_tables={tables_dir}", flush=True)
        return 0

    for selector in unmatched_split_hint_selectors(args, rows):
        print(f"[AE][part2][{figure_scope}][warning] unmatched MorphX tuning selector", flush=True)
    config_paths = write_workload_configs(args, rows, output_dir)
    write_manifest_header(manifest_path, args, rows, logs_root)

    variants = selected_variants(args)
    specs: list[RunSpec] = []
    for row in rows:
        for variant in variants:
            config_path = config_paths[(row.table, row.config)]
            specs.append(build_run_spec(args, logs_root, row, variant, config_path))
    if args.max_runs is not None:
        specs = specs[: args.max_runs]

    print(
        f"[AE][part2][{figure_scope}][plan] planned_runs={len(specs)} "
        f"({len(rows)} workload rows x {len(variants)} variants, truncated={args.max_runs is not None})",
        flush=True,
    )
    estimated_hours = len(specs) * max(args.duration, 0.0) / 3600.0
    print(f"[AE][part2][{figure_scope}][plan] injection_window_lower_bound={estimated_hours:.2f}h", flush=True)

    failures = 0
    wall_start = time.time()
    for idx, spec in enumerate(specs, start=1):
        print(
            f"[AE][part2][{spec.row.table}][run {idx}/{len(specs)}] START "
            f"{spec.row.output_config} variant={variant_label(spec.variant)}",
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
            f"[AE][part2][{spec.row.table}][run {idx}/{len(specs)}] DONE "
            f"status={status} run_elapsed={elapsed_text} "
            f"part_elapsed={format_duration(total_elapsed)} "
            f"part_eta={format_duration(remaining)}",
            flush=True,
        )
        if status not in {"ok", "skipped", "dry-run"}:
            failures += 1
            if args.fail_fast:
                print(f"[AE][part2][{spec.row.table}][run {idx}/{len(specs)}] stopping because --fail-fast is set.", flush=True)
                break

    if not args.no_analyze:
        tables_dir = output_dir / "tables"
        build_result_tables(rows, logs_root, tables_dir)
        for warning in update_latest_links(args, output_dir):
            print(f"[AE][part2][{figure_scope}][warning] {warning}", flush=True)
        print(f"[AE][part2][{figure_scope}][analysis] generated_tables={tables_dir}", flush=True)

    if failures:
        print(f"[AE][part2][{figure_scope}] completed with {failures} failed child runs.", flush=True)
        return 1
    print(f"[AE][part2][{figure_scope}] completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
