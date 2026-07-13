#!/usr/bin/env python3
"""Merge fresh multi-task SMSched columns into existing co-location tables."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Optional


DEFAULT_TABLES = (
    "llm-1-ggnn.tsv",
    "llm-4-ggnn.tsv",
    "llm-1-mm.tsv",
    "llm-4-mm.tsv",
)
SMSCHED_COLUMNS = (
    "smsched_llm_latency_ms",
    "smsched_llm_p99_ms",
    "smsched_completed",
)
BASELINE_SCHEDULERS = ("mps-30", "mps-50", "orion", "stream", "tgs", "timeslice")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge smsched columns from a fresh multi-task co-location run with "
            "baseline scheduler columns from an existing compact table directory."
        )
    )
    parser.add_argument("--baseline-compact-dir", type=Path, required=True)
    parser.add_argument("--smsched-compact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--table", action="append", default=[], help="Table filename, e.g. llm-1-ggnn.tsv")
    return parser.parse_args()


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        rows = list(reader)
    if not rows:
        return [], []
    headers = [cell.strip() for cell in rows[0]]
    out: list[dict[str, str]] = []
    for row in rows[1:]:
        if not any(cell.strip() for cell in row):
            continue
        padded = row + [""] * (len(headers) - len(row))
        out.append({headers[idx]: padded[idx].strip() for idx in range(len(headers))})
    return headers, out


def write_tsv(path: Path, headers: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_float(value: str) -> Optional[float]:
    value = value.strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def fmt(value: Optional[float], precision: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{precision}f}"


def row_key(row: dict[str, str]) -> Optional[float]:
    value = parse_float(row.get("other_rps", ""))
    if value is None:
        return None
    return round(value, 6)


def same_p99_frontier(row: dict[str, str]) -> tuple[Optional[str], Optional[float]]:
    smsched_p99 = parse_float(row.get("smsched_llm_p99_ms", ""))
    if smsched_p99 is None:
        return None, None
    best_scheduler: Optional[str] = None
    best_completed: Optional[float] = None
    for scheduler in BASELINE_SCHEDULERS:
        p99 = parse_float(row.get(f"{scheduler}_llm_p99_ms", ""))
        completed = parse_float(row.get(f"{scheduler}_completed", ""))
        if p99 is None or completed is None or p99 > smsched_p99:
            continue
        if best_completed is None or completed > best_completed:
            best_scheduler = scheduler
            best_completed = completed
    return best_scheduler, best_completed


def merge_table(baseline_path: Path, smsched_path: Path, output_path: Path) -> list[dict[str, object]]:
    headers, baseline_rows = read_tsv(baseline_path)
    _, smsched_rows = read_tsv(smsched_path)
    smsched_by_key = {key: row for row in smsched_rows if (key := row_key(row)) is not None}
    merged: list[dict[str, str]] = []
    summary: list[dict[str, object]] = []

    for old_row in baseline_rows:
        key = row_key(old_row)
        row = dict(old_row)
        new_row = smsched_by_key.get(key, {})
        for column in SMSCHED_COLUMNS:
            if new_row.get(column):
                row[column] = new_row[column]
        merged.append(row)

        frontier_scheduler, frontier_completed = same_p99_frontier(row)
        smsched_completed = parse_float(row.get("smsched_completed", ""))
        summary.append(
            {
                "other_rps": row.get("other_rps", ""),
                "smsched_llm_p99_ms": row.get("smsched_llm_p99_ms", ""),
                "smsched_completed": row.get("smsched_completed", ""),
                "best_baseline_at_le_smsched_p99": frontier_scheduler or "",
                "best_baseline_completed_at_le_smsched_p99": fmt(frontier_completed),
                "smsched_completed_wins_at_same_p99": int(
                    smsched_completed is not None
                    and frontier_completed is not None
                    and smsched_completed > frontier_completed
                ),
            }
        )

    write_tsv(output_path, headers, merged)
    return summary


def main() -> int:
    args = parse_args()
    tables = tuple(args.table) if args.table else DEFAULT_TABLES
    all_summary: list[dict[str, object]] = []
    for table_name in tables:
        baseline_path = args.baseline_compact_dir / table_name
        smsched_path = args.smsched_compact_dir / table_name
        output_path = args.output_dir / table_name
        if not baseline_path.exists():
            raise FileNotFoundError(baseline_path)
        if not smsched_path.exists():
            raise FileNotFoundError(smsched_path)
        for row in merge_table(baseline_path, smsched_path, output_path):
            row["table"] = table_name.removesuffix(".tsv")
            all_summary.append(row)

    write_tsv(
        args.output_dir / "smsched-frontier-summary.tsv",
        [
            "table",
            "other_rps",
            "smsched_llm_p99_ms",
            "smsched_completed",
            "best_baseline_at_le_smsched_p99",
            "best_baseline_completed_at_le_smsched_p99",
            "smsched_completed_wins_at_same_p99",
        ],
        all_summary,
    )
    print(f"merged tables: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
