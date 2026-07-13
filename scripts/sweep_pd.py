#!/usr/bin/env python3
"""Batch runner for pd configs.

This script enumerates input/output length combinations and optionally rps
values, updates `configs/pd/T.json`, and invokes `run.py` to collect logs.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


# INPUT_LENGTHS = (1024, 2048, 4096)
# OUTPUT_LENGTHS = (64, 128, 256, 512)
# ONLINE_RPS = (1, 3, 5, 7, 9, 11)

# INPUT_LENGTHS = (4096,)
# OUTPUT_LENGTHS = (256,)
# ONLINE_RPS = (0.5, 1)

# llama-2-7b
# 1024, 128, (3, 4, 5, 6, 7)
# 2048, 128, ()
# llama-3-8b
# 1024, 128, (5, 5.5, 6, 6.5, 7)
# 2048, 128, (2.5, 3, 3.5, 4, 4.5)
# 2048, 256, (2.5, 3, 3.5, 4, 4.5)

# Configs: (input_length, output_length): (rps1, rps2, ...)
CONFIGS = {
    # (1024, 256): (3, 4, 5, 6, 7),
    # (1024, 128): (5, 6, 7, 8, 9),
    # (2048, 128): (3, 4, 5, 6, 7),
    (2048, 256): (1.5, 2.5),
    (4096, 128): (0.4, 0.8, 1.2, 1.6), 
    (4096, 256): (0.4, 0.8, 1.2, 1.6),
    # (2048, 128): (1,)
}

# CONFIGS = {}
# for input_length in (4096,):
#     for output_length in (128, 256):
#         CONFIGS[(input_length, output_length)] = (1, 1.5, 2, 2.5, 3, 3.5, 4)

def load_template(template_cfg: Path) -> dict:
    try:
        with open(template_cfg, "r", encoding="utf-8") as src:
            return json.load(src)
    except FileNotFoundError as exc:
        raise SystemExit(f"Template config not found: {template_cfg}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse JSON template {template_cfg}: {exc}") from exc


def write_config(
    base_cfg: dict,
    dest: Path,
    input_length: int,
    output_length: int,
    rps: Optional[int] = None,
) -> None:
    cfg = dict(base_cfg)
    cfg["input_length"] = input_length
    cfg["output_length"] = output_length
    if rps is not None:
        cfg["rps"] = rps
    with open(dest, "w", encoding="utf-8") as out:
        json.dump(cfg, out, indent=4)
        out.write("\n")


def run_one(run_py: Path, load: str, config_name: str, workload: str, split_hint: int) -> int:
    cmd: List[str] = [sys.executable, str(run_py), load, config_name, workload, str(split_hint)]
    proc = subprocess.run(cmd, cwd=run_py.parent)
    return proc.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep pd configs using run.py")
    parser.add_argument(
        "--load",
        "--mode",
        dest="load",
        default="pd-offline",
        choices=("pd-offline", "pd-online"),
        help="Load type to use when invoking run.py (default: %(default)s)",
    )
    parser.add_argument(
        "--workload",
        default="smsched",
        choices=("baseline", "stream", "smsched", "chunked", "all"),
        help="Workload for run.py (default: %(default)s)",
    )
    parser.add_argument(
        "--template",
        default="E",
        help="Base config name to clone (default: %(default)s)",
    )
    parser.add_argument(
        "--config-name",
        default="T",
        help="Config name (without extension) to write before each run (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the commands without executing run.py",
    )
    parser.add_argument(
        "--split-hint",
        type=int,
        default=76,
        help="Split hint value to pass to run.py (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    configs_dir = repo_root / "configs" / "pd"
    template_cfg = configs_dir / f"{args.template}.json"
    dest_cfg = configs_dir / f"{args.config_name}.json"
    run_py = Path(__file__).resolve().parent / "run.py"

    base_cfg = load_template(template_cfg)

    failures = 0
    for (input_length, output_length), rps_list in CONFIGS.items():
        if args.load == "pd-online":
            for rps in rps_list:
                write_config(base_cfg, dest_cfg, input_length, output_length, rps=rps)
                if args.dry_run:
                    print(
                        f"DRY RUN: would execute {run_py.name} {args.load} {args.config_name} {args.workload} {args.split_hint} (rps={rps})"
                    )
                    continue

                status_msg = (
                    f"Running input={input_length} output={output_length} rps={rps} split={args.split_hint}"
                )
                print(status_msg.ljust(80, " "), end="")
                ret = run_one(run_py, args.load, args.config_name, args.workload, args.split_hint)
                if ret == 0:
                    print("[OK]")
                else:
                    print(f"[FAIL {ret}]")
                    failures += 1
        else:
            write_config(base_cfg, dest_cfg, input_length, output_length)
            if args.dry_run:
                print(
                    f"DRY RUN: would execute {run_py.name} {args.load} {args.config_name} {args.workload} {args.split_hint}"
                )
                continue

            status_msg = f"Running input={input_length} output={output_length} split={args.split_hint}"
            print(status_msg.ljust(80, " "), end="")
            ret = run_one(run_py, args.load, args.config_name, args.workload, args.split_hint)
            if ret == 0:
                print("[OK]")
            else:
                print(f"[FAIL {ret}]")
                failures += 1

    if failures:
        raise SystemExit(f"Completed with {failures} failures")


if __name__ == "__main__":
    main()
