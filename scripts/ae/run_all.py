#!/usr/bin/env python3
"""Unified SOSP 2026 AE runner.

Public output layout:

  logs/<timestamp>/
    part1-colocate/
    part2-pd-colocate/
    part3-overhead/
    part4-model/

The lower-level part runners still own workload-specific command generation and
parsing. This wrapper gives reviewers one stable entry point, consistent part
names, resume/analyze modes, repository-local builds, and final tables/figures.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS_ROOT = REPO_ROOT / "logs"
TABLES_DIR = REPO_ROOT / "tables"
FIGURES_DIR = REPO_ROOT / "figures"
SYSTEM_NAME_RE = re.compile("smsched", re.IGNORECASE)
SMSCHED_CONDA_ENV_RE = re.compile(r"(?<=/envs/)smsched(?=/)", re.IGNORECASE)


@dataclass(frozen=True)
class Part:
    name: str
    title: str


PARTS = (
    Part("part1-colocate", "multi-task co-location: Figures 10-12"),
    Part("part2-pd-colocate", "LLM PD co-location: Figures 13-14"),
    Part("part3-overhead", "overhead analysis: Figure 15"),
    Part("part4-model", "GEMM performance model accuracy: Figure 16"),
)

PART_FIGURES = {
    "part1-colocate": "Figures10-12",
    "part2-pd-colocate": "Figures13-14",
    "part3-overhead": "Figure15",
    "part4-model": "Figure16",
}

HIDDEN_OPTION_PREFIXES = (
    "--smsched-env",
    "--smsched-model-env",
    "--smsched-split-hint",
    "--smsched-split-hint-for",
    "--override-split-hint",
    "--split-hint-for",
)

SPLIT_NAME_RE = re.compile("split", re.IGNORECASE)


def default_python() -> str:
    candidates = (
        os.environ.get("PYTHON"),
        "/home/rtx/miniconda3/envs/smsched/bin/python",
        "/mnt/data/rtx/miniconda3/envs/smsched/bin/python",
    )
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete MorphX SOSP 2026 AE workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes:\n"
            "  ./run.sh                    Run all experiments from scratch into logs/<timestamp>.\n"
            "  ./run.sh short              Run a shorter subset: Figure10/12(a), Figure13(a), Figure15, Figure16.\n"
            "  ./run.sh resume             Continue the newest logs/<timestamp> run at run granularity.\n"
            "  ./run.sh analyse            Regenerate tables/figures from the newest logs/<timestamp>.\n"
            "  ./run.sh analyse-reference  Regenerate tables/figures from logs/reference.\n"
            "  ./run.sh smoke              Short dry-run smoke test of the workflow wiring.\n"
        ),
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="run",
        choices=("run", "short", "resume", "analyse", "analyse-reference", "smoke"),
        help="Workflow mode. Default: run.",
    )
    parser.add_argument("--python", dest="python_bin", default=default_python(), help="Python interpreter to use.")
    parser.add_argument("--no-build", action="store_true", help="Skip repository-local builds.")
    return parser.parse_args()


def now_timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


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


def quote_cmd(command: Iterable[str]) -> str:
    return " ".join(shlex_quote(part) for part in command)


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(str(value))


def display_cmd(command: Iterable[str]) -> str:
    """Return a reviewer-facing command line without exposing tuning internals."""
    display: list[str] = []
    skip_next = False
    for raw in command:
        token = str(raw)
        if skip_next:
            skip_next = False
            continue
        if any(token == option for option in HIDDEN_OPTION_PREFIXES):
            skip_next = True
            continue
        if any(token.startswith(f"{option}=") for option in HIDDEN_OPTION_PREFIXES):
            continue
        if token.endswith("/bin/python") and "/envs/" in token:
            display.append("$PYTHON")
        elif "split" in token.lower():
            display.append("[hidden]")
        else:
            display.append(reviewer_system_text(token))
    return quote_cmd(display)


def reviewer_system_text(value: object) -> str:
    """Use the paper system name without rewriting the smsched conda path."""
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


def sanitize_user_text(text: str) -> str:
    """Keep reviewer-facing stdout/stderr free of internal tuning names."""
    return SPLIT_NAME_RE.sub("tuning", reviewer_system_text(text))


def stream_pipe(pipe, target) -> None:
    try:
        for line in pipe:
            target.write(sanitize_user_text(line))
            target.flush()
    finally:
        pipe.close()


def part_prefix(part: Part, index: int) -> str:
    return f"[AE][part {index}/{len(PARTS)}][{PART_FIGURES[part.name]}]"


def run_command(
    command: list[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    prefix: str = "[AE]",
    quiet: bool = False,
) -> None:
    print(f"{prefix}[command] $ {display_cmd(command)}", flush=True)
    if quiet:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            combined = (result.stdout or "") + (result.stderr or "")
            tail = "\n".join(sanitize_user_text(combined).splitlines()[-40:])
            if tail:
                print(f"{prefix}[command][tail]\n{tail}", file=sys.stderr, flush=True)
        else:
            print(f"{prefix}[command] DONE", flush=True)
    else:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_thread = threading.Thread(target=stream_pipe, args=(process.stdout, sys.stdout))
        stderr_thread = threading.Thread(target=stream_pipe, args=(process.stderr, sys.stderr))
        stdout_thread.start()
        stderr_thread.start()
        result_returncode = process.wait()
        stdout_thread.join()
        stderr_thread.join()
        result = subprocess.CompletedProcess(command, result_returncode)

    if result.returncode != 0:
        raise RuntimeError(f"command failed with return code {result.returncode}: {display_cmd(command)}")


def latest_run_dir() -> Path:
    latest = LOGS_ROOT / "latest"
    if latest.is_symlink() and latest.exists():
        return latest.resolve()
    candidates = [
        path
        for path in LOGS_ROOT.iterdir()
        if path.is_dir() and path.name not in {"reference"} and not path.name.startswith("smoke-")
    ] if LOGS_ROOT.exists() else []
    if not candidates:
        raise RuntimeError("No previous logs/<timestamp> directory found. Run ./run.sh first.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def create_run_dir(prefix: str = "", *, update_latest: bool = True) -> Path:
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    name = f"{prefix}{now_timestamp()}"
    run_dir = LOGS_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=False)
    if update_latest:
        latest = LOGS_ROOT / "latest"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(run_dir.name, target_is_directory=True)
    (run_dir / "manifest.txt").write_text(
        f"created_at={dt.datetime.now().isoformat(timespec='seconds')}\n"
        f"repo_root={REPO_ROOT}\n",
        encoding="utf-8",
    )
    return run_dir


def append_manifest(run_dir: Path, **items: str) -> None:
    with (run_dir / "manifest.txt").open("a", encoding="utf-8") as handle:
        for key, value in items.items():
            handle.write(f"{key}={value}\n")


def manifest_value(run_dir: Path, key: str) -> str | None:
    try:
        lines = (run_dir / "manifest.txt").read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    prefix = f"{key}="
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix) :]
    return None


def is_short_run(run_dir: Path) -> bool:
    return manifest_value(run_dir, "mode") == "short"


def part_dir(run_dir: Path, part_name: str) -> Path:
    return run_dir / part_name


def configure_runtime(use_cuda_llm: bool, *, prefix: str = "[AE]") -> None:
    if not (REPO_ROOT / "runtime" / "CMakeLists.txt").exists():
        print(f"{prefix}[build] runtime CMakeLists.txt not found; skipping runtime configure", flush=True)
        return
    cmake_args = [
        "cmake",
        "-S",
        str(REPO_ROOT / "runtime"),
        "-B",
        str(REPO_ROOT / "runtime" / "build"),
        f"-DUSE_CUDA_LLM={'ON' if use_cuda_llm else 'OFF'}",
        "-DUSE_CUDA_LOG=OFF",
        "-DUSE_CUDA_ORION=OFF",
        "-DUSE_CUDA_LITHOS=OFF",
    ]
    run_command(cmake_args, prefix=prefix, quiet=True)
    run_command(
        ["cmake", "--build", str(REPO_ROOT / "runtime" / "build"), "--target", "cuda", "cuda_hook", "preload", "api", "-j"],
        prefix=prefix,
        quiet=True,
    )


def build_common_tools(include_model: bool, *, python_bin: str, prefix: str = "[AE]") -> None:
    smsched_pass_build = REPO_ROOT / "smsched-pass" / "build"
    if not smsched_pass_build.is_dir():
        llvm_candidates = [
            os.environ.get("LLVM_DIR"),
            "/home/rtx/llvm-project/install/lib/cmake/llvm",
            "/mnt/data/rtx/llvm-project/install/lib/cmake/llvm",
        ]
        llvm_dir = next((candidate for candidate in llvm_candidates if candidate and Path(candidate).exists()), None)
        if llvm_dir is None:
            raise RuntimeError("LLVM_DIR was not found; expected the MorphX compiler pass toolchain on this machine.")
        run_command(
            [
                "cmake",
                "-S",
                str(REPO_ROOT / "smsched-pass"),
                "-B",
                str(smsched_pass_build),
                f"-DLLVM_DIR={llvm_dir}",
            ],
            prefix=prefix,
            quiet=True,
        )
    run_command(["cmake", "--build", str(smsched_pass_build), "-j"], prefix=prefix, quiet=True)

    tgs_hijack = REPO_ROOT / "TGS" / "hijack"
    if tgs_hijack.is_dir():
        run_command(["bash", str(tgs_hijack / "build.sh")], prefix=prefix, quiet=True)
    else:
        print(f"{prefix}[build] skip TGS hijack: source directory not found", flush=True)

    microbench = REPO_ROOT / "microbench"
    if not (microbench / "build").is_dir():
        run_command(["cmake", "-S", str(microbench), "-B", str(microbench / "build")], prefix=prefix, quiet=True)
    if (microbench / "build").is_dir():
        run_command(["cmake", "--build", str(microbench / "build"), "--target", "bench", "-j"], prefix=prefix, quiet=True)
    if include_model:
        model_build = microbench / "build-model"
        if not model_build.is_dir():
            run_command(["cmake", "-S", str(microbench), "-B", str(model_build), "-DSMSCHED_ENABLE_PASS=ON"], prefix=prefix, quiet=True)
        run_command(["cmake", "--build", str(model_build), "--target", "bench", "-j"], prefix=prefix, quiet=True)

    nvbit_mem_trace = REPO_ROOT / "nvbit-tutorial" / "tools" / "mem_trace"
    if nvbit_mem_trace.is_dir():
        run_command(["make", "-C", str(nvbit_mem_trace)], prefix=prefix, quiet=True)

    ensure_local_python_package("ggnn", REPO_ROOT / "ggnn", python_bin=python_bin, prefix=prefix)


def ensure_local_python_package(package: str, source_dir: Path, *, python_bin: str, prefix: str) -> None:
    if not source_dir.exists():
        print(f"{prefix}[build] skip Python package {package}: source directory not found", flush=True)
        return
    probe = (
        "import importlib.util, pathlib; "
        f"spec=importlib.util.find_spec({package!r}); "
        "print(pathlib.Path(spec.origin).resolve() if spec and spec.origin else '')"
    )
    result = subprocess.run([python_bin, "-c", probe], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    origin = Path(result.stdout.strip()) if result.returncode == 0 and result.stdout.strip() else None
    try:
        if origin is not None:
            if source_dir.resolve() in origin.parents:
                print(f"{prefix}[build] Python package {package} already points to this checkout", flush=True)
            else:
                print(f"{prefix}[build] Python package {package} provided by the conda environment", flush=True)
            return
    except OSError:
        if origin is not None:
            print(f"{prefix}[build] Python package {package} provided by the conda environment", flush=True)
            return
    print(f"{prefix}[command] $ {display_cmd([python_bin, '-m', 'pip', 'install', '-e', str(source_dir), '--no-build-isolation'])}", flush=True)
    install = subprocess.run(
        [python_bin, "-m", "pip", "install", "-e", str(source_dir), "--no-build-isolation"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if install.returncode == 0:
        print(f"{prefix}[command] DONE", flush=True)
        return
    if origin is not None:
        print(
            f"{prefix}[build] WARNING: editable install for Python package {package} failed; "
            f"using existing environment package at {sanitize_user_text(str(origin))}",
            flush=True,
        )
        return
    combined = (install.stdout or "") + (install.stderr or "")
    tail = "\n".join(sanitize_user_text(combined).splitlines()[-40:])
    if tail:
        print(f"{prefix}[command][tail]\n{tail}", file=sys.stderr, flush=True)
    raise RuntimeError(f"failed to install Python package {package} from {source_dir}")


def build_for_part(part_name: str, args: argparse.Namespace, *, prefix: str = "[AE]") -> None:
    if args.no_build:
        print(f"{prefix}[build] skipped for {part_name} because --no-build is set", flush=True)
        return
    print(f"{prefix}[build] preparing binaries for {part_name}", flush=True)
    if part_name == "part2-pd-colocate":
        configure_runtime(use_cuda_llm=True, prefix=prefix)
    else:
        configure_runtime(use_cuda_llm=False, prefix=prefix)
    if part_name in {"part1-colocate", "part3-overhead", "part4-model"}:
        build_common_tools(include_model=(part_name == "part4-model"), python_bin=args.python_bin, prefix=prefix)


def pd_smsched_env() -> str:
    return (
        f"LD_PRELOAD={REPO_ROOT / 'runtime' / 'build' / 'libcuda.so'}:"
        f"{REPO_ROOT / 'runtime' / 'build' / 'libpreload.so'} "
        "SMSCHED_LLM_TRICK_MODE=0 SMSCHED_LLM_TRACE_LIMIT=0"
    )


def part1_mm_smsched_env() -> str:
    return (
        f"LD_PRELOAD={REPO_ROOT / 'runtime' / 'build' / 'libcuda.so'}:"
        f"{REPO_ROOT / 'runtime' / 'build' / 'libpreload.so'} "
        "SMSCHED_TARGET_PATTERNS=half_t SMSCHED_ONLY_WHITELIST=1"
    )


def run_part(
    part: Part,
    run_dir: Path,
    args: argparse.Namespace,
    *,
    resume: bool = False,
    smoke: bool = False,
    short: bool = False,
    prefix: str = "[AE]",
) -> None:
    output_dir = part_dir(run_dir, part.name)
    command: list[str]
    common = ["--output-dir", str(output_dir), "--python", args.python_bin]
    if part.name == "part1-colocate":
        if smoke:
            command = [
                args.python_bin,
                str(REPO_ROOT / "scripts" / "ae" / "run_colocate_part1.py"),
                *common,
                "--duration",
                "180",
                "--smsched-split-hint",
                "76",
                "--fail-fast",
            ]
            command.extend(["--only-scheduler", "stream", "--only-experiment", "llm-1-ggnn-1", "--duration", "1", "--dry-run"])
            if resume:
                command.append("--skip-existing")
            run_command(command, prefix=prefix)
            return

        common_part1 = [
            args.python_bin,
            str(REPO_ROOT / "scripts" / "ae" / "run_colocate_part1.py"),
            *common,
            "--duration",
            "180",
            "--smsched-split-hint",
            "76",
            "--fail-fast",
        ]
        ggnn_command = [
            *common_part1,
            "--only-table",
            "llm-1-ggnn",
        ]
        if not short:
            ggnn_command.extend(["--only-table", "llm-4-ggnn"])
        mm_command = [
            *common_part1,
            "--only-table",
            "llm-1-mm",
            "--only-table",
            "llm-4-mm",
            "--smsched-env",
            part1_mm_smsched_env(),
        ]
        if resume:
            ggnn_command.append("--skip-existing")
            mm_command.append("--skip-existing")
        run_command(ggnn_command, prefix=prefix)
        if not short:
            run_command(mm_command, prefix=prefix)
        return
    elif part.name == "part2-pd-colocate":
        command = [
            args.python_bin,
            str(REPO_ROOT / "scripts" / "ae" / "run_pd_colocate_part2.py"),
            *common,
            "--duration",
            "180",
            "--smsched-env",
            pd_smsched_env(),
            "--no-update-latest-link",
            "--fail-fast",
        ]
        if smoke:
            command.extend(["--only-config", "Figure13:in-2048-out-256-rps-1", "--only-variant", "baseline", "--duration", "1", "--dry-run"])
        elif short:
            command.extend(["--only-table", "Figure13", "--only-config", "Figure13:in-2048-out-256-rps-*"])
    elif part.name == "part3-overhead":
        command = [
            args.python_bin,
            str(REPO_ROOT / "scripts" / "ae" / "run_overhead_model_part4.py"),
            *common,
            "--only-section",
            "overhead",
            "--no-build",
            "--no-update-latest-link",
            "--fail-fast",
        ]
        if smoke:
            command.extend(["--only-task", "DNN", "--only-method", "base", "--dry-run"])
    elif part.name == "part4-model":
        command = [
            args.python_bin,
            str(REPO_ROOT / "scripts" / "ae" / "run_overhead_model_part4.py"),
            *common,
            "--only-section",
            "model",
            "--no-build",
            "--no-update-latest-link",
            "--fail-fast",
        ]
        if smoke:
            command.extend(["--only-series", "16", "--dry-run"])
    else:
        raise ValueError(part.name)
    if resume:
        command.append("--skip-existing")
    run_command(command, prefix=prefix)


def analyze_part(part: Part, run_dir: Path, args: argparse.Namespace, *, short: bool = False, prefix: str = "[AE]") -> None:
    output_dir = part_dir(run_dir, part.name)
    common = ["--output-dir", str(output_dir), "--python", args.python_bin, "--analyze-only"]
    if part.name == "part1-colocate":
        command = [
            args.python_bin,
            str(REPO_ROOT / "scripts" / "ae" / "run_colocate_part1.py"),
            *common,
            "--smsched-split-hint",
            "76",
        ]
        if short:
            command.extend(["--only-table", "llm-1-ggnn"])
    elif part.name == "part2-pd-colocate":
        command = [
            args.python_bin,
            str(REPO_ROOT / "scripts" / "ae" / "run_pd_colocate_part2.py"),
            *common,
            "--smsched-env",
            pd_smsched_env(),
            "--no-update-latest-link",
        ]
        if short:
            command.extend(["--only-table", "Figure13", "--only-config", "Figure13:in-2048-out-256-rps-*"])
    elif part.name == "part3-overhead":
        command = [
            args.python_bin,
            str(REPO_ROOT / "scripts" / "ae" / "run_overhead_model_part4.py"),
            *common,
            "--only-section",
            "overhead",
            "--no-build",
            "--no-update-latest-link",
        ]
    elif part.name == "part4-model":
        command = [
            args.python_bin,
            str(REPO_ROOT / "scripts" / "ae" / "run_overhead_model_part4.py"),
            *common,
            "--only-section",
            "model",
            "--no-build",
            "--no-update-latest-link",
        ]
    else:
        raise ValueError(part.name)
    run_command(command, prefix=prefix)


def copy_required(src: Path, dst: Path) -> None:
    if not src.exists():
        raise RuntimeError(f"required generated file is missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def collect_tables_and_figures(run_dir: Path, args: argparse.Namespace, *, short: bool = False) -> None:
    for generated_dir in (TABLES_DIR, FIGURES_DIR):
        if generated_dir.is_symlink() or generated_dir.is_file():
            generated_dir.unlink()
        elif generated_dir.exists():
            shutil.rmtree(generated_dir)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    generated_tables: list[Path] = []

    part1 = part_dir(run_dir, "part1-colocate") / "tables" / "compact"
    copy_required(part1 / "llm-1-ggnn.tsv", TABLES_DIR / "Figure10,12(a).txt")
    generated_tables.append(TABLES_DIR / "Figure10,12(a).txt")
    if not short:
        copy_required(part1 / "llm-4-ggnn.tsv", TABLES_DIR / "Figure10,12(b).txt")
        generated_tables.append(TABLES_DIR / "Figure10,12(b).txt")
        copy_required(part1 / "llm-1-mm.tsv", TABLES_DIR / "Figure11(a).txt")
        generated_tables.append(TABLES_DIR / "Figure11(a).txt")
        copy_required(part1 / "llm-4-mm.tsv", TABLES_DIR / "Figure11(b).txt")
        generated_tables.append(TABLES_DIR / "Figure11(b).txt")

    part2 = part_dir(run_dir, "part2-pd-colocate") / "tables"
    copy_required(part2 / "Figure13.tsv", TABLES_DIR / "Figure13.txt")
    generated_tables.append(TABLES_DIR / "Figure13.txt")
    if not short:
        copy_required(part2 / "Figure14.tsv", TABLES_DIR / "Figure14.txt")
        generated_tables.append(TABLES_DIR / "Figure14.txt")

    part3 = part_dir(run_dir, "part3-overhead") / "tables"
    copy_required(part3 / "Figure15.tsv", TABLES_DIR / "Figure15.txt")
    generated_tables.append(TABLES_DIR / "Figure15.txt")

    part4 = part_dir(run_dir, "part4-model") / "tables"
    copy_required(part4 / "Figure16.tsv", TABLES_DIR / "Figure16.txt")
    generated_tables.append(TABLES_DIR / "Figure16.txt")
    if (part4 / "Figure16-profile-metadata.tsv").exists():
        copy_required(part4 / "Figure16-profile-metadata.tsv", TABLES_DIR / "Figure16-profile-metadata.txt")
        generated_tables.append(TABLES_DIR / "Figure16-profile-metadata.txt")

    readme_lines = [
        "# Aggregated AE tables",
        f"# Generated from {run_dir}",
        "",
        *[path.name for path in sorted(generated_tables, key=lambda item: item.name)],
        "",
    ]
    (TABLES_DIR / "README.txt").write_text("\n".join(readme_lines), encoding="utf-8")

    run_command(
        [
            args.python_bin,
            str(REPO_ROOT / "plot_figures.py"),
            "--tables-dir",
            str(TABLES_DIR),
            "--output-dir",
            str(FIGURES_DIR),
            "--formats",
            "png,pdf",
            *(["--allow-missing"] if short else []),
        ],
        prefix="[AE][final][Figures10-16]",
    )
    print(f"[AE][final][Figures10-16] tables={TABLES_DIR}", flush=True)
    print(f"[AE][final][Figures10-16] figures={FIGURES_DIR}", flush=True)


def looks_like_text(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:4096]
    except OSError:
        return False
    if not chunk:
        return True
    if b"\x00" in chunk:
        return False
    return True


def sanitize_reviewer_text(root: Path) -> None:
    if not root.exists():
        return
    files = [root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()]
    for path in files:
        if path.suffix.lower() in {".png", ".pdf", ".so", ".a", ".o", ".ncu-rep", ".nsys-rep", ".sqlite"}:
            continue
        if not looks_like_text(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        updated = reviewer_system_text(text)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def reference_run_dir() -> Path:
    reference = LOGS_ROOT / "reference"
    if not reference.is_dir():
        raise RuntimeError("logs/reference was not found. Run the full workflow first or place reference logs there.")
    missing = [part.name for part in PARTS if not (reference / part.name).is_dir()]
    if missing:
        raise RuntimeError(f"logs/reference is incomplete; missing: {', '.join(missing)}")
    return reference


def run_or_analyze(run_dir: Path, args: argparse.Namespace, *, resume: bool, analyze_only: bool, smoke: bool, short: bool = False) -> None:
    total_start = time.monotonic()
    for idx, part in enumerate(PARTS, start=1):
        prefix = part_prefix(part, idx)
        part_start = time.monotonic()
        print(f"{prefix} BEGIN {part.name}: {part.title}", flush=True)
        if analyze_only:
            analyze_part(part, run_dir, args, short=short, prefix=prefix)
        else:
            build_for_part(part.name, args, prefix=prefix)
            run_part(part, run_dir, args, resume=resume, smoke=smoke, short=short, prefix=prefix)
        part_elapsed = time.monotonic() - part_start
        total_elapsed = time.monotonic() - total_start
        avg_part = total_elapsed / idx
        remaining = avg_part * (len(PARTS) - idx)
        print(
            f"{prefix} END {part.name}: "
            f"part_elapsed={format_duration(part_elapsed)} total_elapsed={format_duration(total_elapsed)} "
            f"workflow_remaining_estimate={format_duration(remaining)}",
            flush=True,
        )


def main() -> int:
    args = parse_args()
    try:
        if args.mode == "analyse-reference":
            run_dir = reference_run_dir()
            run_or_analyze(run_dir, args, resume=False, analyze_only=True, smoke=False)
            sanitize_reviewer_text(run_dir)
            collect_tables_and_figures(run_dir, args)
            sanitize_reviewer_text(TABLES_DIR)
            return 0
        if args.mode == "analyse":
            run_dir = latest_run_dir()
            short = is_short_run(run_dir)
            print(f"[AE][workflow][Figures10-16] analyze_latest_run={run_dir}", flush=True)
            run_or_analyze(run_dir, args, resume=False, analyze_only=True, smoke=False, short=short)
            sanitize_reviewer_text(run_dir)
            collect_tables_and_figures(run_dir, args, short=short)
            sanitize_reviewer_text(TABLES_DIR)
            return 0
        if args.mode == "resume":
            run_dir = latest_run_dir()
            short = is_short_run(run_dir)
            print(f"[AE][workflow][Figures10-16] resume_latest_run={run_dir}", flush=True)
            run_or_analyze(run_dir, args, resume=True, analyze_only=False, smoke=False, short=short)
            sanitize_reviewer_text(run_dir)
            collect_tables_and_figures(run_dir, args, short=short)
            sanitize_reviewer_text(TABLES_DIR)
            return 0
        if args.mode == "smoke":
            run_dir = create_run_dir("smoke-", update_latest=False)
            print(f"[AE][workflow][Figures10-16] smoke_logs={run_dir}", flush=True)
            run_or_analyze(run_dir, args, resume=False, analyze_only=False, smoke=True)
            print("[AE][workflow][Figures10-16] smoke_complete=true dry_run_tables_figures=false", flush=True)
            return 0
        if args.mode == "short":
            run_dir = create_run_dir()
            append_manifest(
                run_dir,
                mode="short",
                scope="Figure10/12(a), Figure13(a), Figure15, Figure16",
            )
            print(f"[AE][workflow][Figures10-16] short_logs={run_dir}", flush=True)
            run_or_analyze(run_dir, args, resume=False, analyze_only=False, smoke=False, short=True)
            sanitize_reviewer_text(run_dir)
            collect_tables_and_figures(run_dir, args, short=True)
            sanitize_reviewer_text(TABLES_DIR)
            return 0

        run_dir = create_run_dir()
        append_manifest(run_dir, mode="full")
        print(f"[AE][workflow][Figures10-16] logs={run_dir}", flush=True)
        run_or_analyze(run_dir, args, resume=False, analyze_only=False, smoke=False)
        sanitize_reviewer_text(run_dir)
        collect_tables_and_figures(run_dir, args)
        sanitize_reviewer_text(TABLES_DIR)
        return 0
    except RuntimeError as exc:
        print(f"[AE][error] {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
