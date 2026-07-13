#!/usr/bin/env python3
"""Plot regenerated AE tables into figures.

By default this script reads ./tables and writes ./figures.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


PART1_METHODS = [
    (("MorphX", "smsched"), "MorphX", "#d62728", "o"),
    (("stream",), "Stream", "#1f77b4", "s"),
    (("mps-30",), "MPS-30", "#2ca02c", "^"),
    (("mps-50",), "MPS-50", "#ff7f0e", "v"),
    (("orion",), "Orion", "#9467bd", "D"),
    (("tgs",), "TGS", "#8c564b", "P"),
    (("timeslice",), "TimeSlice", "#7f7f7f", "X"),
]

PD_METHODS = [
    (("MorphX", "smsched"), "MorphX", "#d62728", "o"),
    (("baseline",), "Baseline", "#2ca02c", "^"),
    (("stream",), "Stream", "#1f77b4", "s"),
    (("chunked",), "Chunked", "#ff7f0e", "v"),
]

BAR_METHODS = [
    (("base(rel)",), "Base", "#4c78a8"),
    (("ncu(rel)",), "NCU", "#f58518"),
    (("neutrino(rel)",), "Neutrino", "#54a24b"),
    (("nvbit(rel)",), "NVBit", "#b279a2"),
    (("MorphX(rel)", "smsched(rel)"), "MorphX", "#e45756"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SOSP AE figures from regenerated tables.")
    parser.add_argument("--tables-dir", type=Path, default=Path("tables"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated output formats, e.g. png,pdf or png.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip figures whose input tables are absent. Used by the short AE mode.",
    )
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def fnum(value: str | None) -> float:
    if value is None:
        return math.nan
    value = value.strip()
    if not value or value.upper() == "N/A":
        return math.nan
    return float(value)


def first_value(row: dict[str, str], keys: Iterable[str]) -> str | None:
    for key in keys:
        if key in row:
            return row.get(key)
    return None


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: Iterable[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt = fmt.strip().lower()
        if not fmt:
            continue
        fig.savefig(output_dir / f"{stem}.{fmt}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def require_tables(tables_dir: Path, table_names: Iterable[str], *, allow_missing: bool) -> list[Path]:
    paths = [tables_dir / name for name in table_names]
    existing = [path for path in paths if path.exists()]
    missing = [path.name for path in paths if not path.exists()]
    if missing and not allow_missing:
        raise FileNotFoundError(f"missing required table(s): {', '.join(missing)}")
    return existing


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.tick_params(axis="both", labelsize=9)


def sorted_xy(rows: list[dict[str, str]], x_keys: Iterable[str], y_keys: Iterable[str]) -> tuple[list[float], list[float]]:
    x_keys = tuple(x_keys)
    y_keys = tuple(y_keys)
    points: list[tuple[float, float]] = []
    for row in rows:
        x = fnum(first_value(row, x_keys))
        y = fnum(first_value(row, y_keys))
        if not (math.isnan(x) or math.isnan(y)):
            points.append((x, y))
    points.sort(key=lambda pair: pair[0])
    return [p[0] for p in points], [p[1] for p in points]


def plot_latency_throughput(
    tables_dir: Path,
    output_dir: Path,
    formats: Iterable[str],
    table_names: tuple[str, str],
    figure_name: str,
    y_label: str,
    allow_missing: bool = False,
) -> None:
    all_titles = ["LLM Request Rate = 1 req/s", "LLM Request Rate = 4 req/s"]
    panels = [
        (panel_idx, name, all_titles[panel_idx])
        for panel_idx, name in enumerate(table_names)
        if (tables_dir / name).exists()
    ]
    if not panels:
        require_tables(tables_dir, table_names, allow_missing=allow_missing)
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.0), sharey=True)
    axes_list = np.atleast_1d(axes)
    for ax, (panel_idx, table_name, title) in zip(axes_list, panels):
        rows = read_tsv(tables_dir / table_name)
        for methods, label, color, marker in PART1_METHODS:
            x, y = sorted_xy(rows, (f"{method}_llm_p99_ms" for method in methods), (f"{method}_completed" for method in methods))
            if x:
                ax.plot(x, y, label=label, color=color, marker=marker, linewidth=1.8, markersize=4.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("LLM P99 Token Latency (ms)", fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_xlim((20, 60) if panel_idx == 0 else (20, 130))
        style_axes(ax)

    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9, frameon=False)
    fig.subplots_adjust(top=0.80, wspace=0.18)
    save_figure(fig, output_dir, figure_name, formats)


def plot_query_latency(
    tables_dir: Path,
    output_dir: Path,
    formats: Iterable[str],
    allow_missing: bool = False,
) -> None:
    table_names = ("Figure10,12(a).txt", "Figure10,12(b).txt")
    all_titles = ["LLM Request Rate = 1 req/s", "LLM Request Rate = 4 req/s"]
    panels = [
        (panel_idx, name, all_titles[panel_idx])
        for panel_idx, name in enumerate(table_names)
        if (tables_dir / name).exists()
    ]
    if not panels:
        require_tables(tables_dir, table_names, allow_missing=allow_missing)
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.0), sharey=True)
    axes_list = np.atleast_1d(axes)
    for ax, (_panel_idx, table_name, title) in zip(axes_list, panels):
        rows = read_tsv(tables_dir / table_name)
        for methods, label, color, marker in PART1_METHODS:
            x, y = sorted_xy(rows, (f"{method}_completed" for method in methods), (f"{method}_llm_p99_ms" for method in methods))
            if x:
                ax.plot(x, y, label=label, color=color, marker=marker, linewidth=1.8, markersize=4.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("ANNS Query Rate ($10^5$ Op/s)", fontsize=10)
        ax.set_ylabel("LLM P99 Latency (ms)", fontsize=10)
        style_axes(ax)

    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9, frameon=False)
    fig.subplots_adjust(top=0.80, wspace=0.18)
    save_figure(fig, output_dir, "Figure12", formats)


CONFIG_RE = re.compile(r"^in-(?P<input>\d+)-out-(?P<output>\d+)-rps-.+$")


def config_label(config_prefix: str) -> str:
    match = re.match(r"^in-(?P<input>\d+)-out-(?P<output>\d+)$", config_prefix)
    if not match:
        return config_prefix
    return f"Input={match.group('input')}, Output={match.group('output')}"


def group_pd_rows(rows: list[dict[str, str]]) -> list[tuple[str, list[dict[str, str]]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    order: list[str] = []
    for row in rows:
        config = row.get("config", "")
        match = CONFIG_RE.match(config)
        prefix = f"in-{match.group('input')}-out-{match.group('output')}" if match else config
        if prefix not in groups:
            groups[prefix] = []
            order.append(prefix)
        groups[prefix].append(row)
    return [(prefix, groups[prefix]) for prefix in order]


def plot_pd_figure(
    tables_dir: Path,
    output_dir: Path,
    formats: Iterable[str],
    table_name: str,
    figure_name: str,
    allow_missing: bool = False,
) -> None:
    if not (tables_dir / table_name).exists():
        require_tables(tables_dir, (table_name,), allow_missing=allow_missing)
        return
    rows = read_tsv(tables_dir / table_name)
    grouped = group_pd_rows(rows)
    if allow_missing:
        grouped = grouped[: max(1, len(grouped))]
        if not grouped:
            grouped = [("missing-0", [])]
    elif len(grouped) < 3:
        grouped.extend((f"missing-{i}", []) for i in range(3 - len(grouped)))
        grouped = grouped[:3]
    else:
        grouped = grouped[:3]

    fig_height = 3.4 * len(grouped)
    fig, axes = plt.subplots(len(grouped), 2, figsize=(10.8, fig_height), sharex=False)
    axes_grid = np.atleast_2d(axes)
    metric_cols = [("avg", "Avg Norm. Req. Latency (s)"), ("p99", "P99 Norm. Req. Latency (s)")]
    for row_idx, (config, config_rows) in enumerate(grouped):
        for col_idx, (metric, ylabel) in enumerate(metric_cols):
            ax = axes_grid[row_idx][col_idx]
            if not config_rows:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            for methods, label, color, marker in PD_METHODS:
                points: list[tuple[float, float]] = []
                for row in config_rows:
                    x = fnum(row.get("rps"))
                    y = fnum(first_value(row, (f"{method}({metric})" for method in methods)))
                    if not (math.isnan(x) or math.isnan(y)):
                        points.append((x, y))
                points.sort(key=lambda pair: pair[0])
                if points:
                    ax.plot(
                        [p[0] for p in points],
                        [p[1] for p in points],
                        label=label,
                        color=color,
                        marker=marker,
                        linewidth=1.8,
                        markersize=4.5,
                    )
            ax.set_title(f"{config_label(config)} - {metric.upper()}", fontsize=10)
            ax.set_xlabel("Request Rate (req/s)", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            style_axes(ax)

    handles, labels = axes_grid[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9, frameon=False)
    fig.subplots_adjust(top=0.92, hspace=0.48, wspace=0.25)
    save_figure(fig, output_dir, figure_name, formats)


def plot_overhead(tables_dir: Path, output_dir: Path, formats: Iterable[str], allow_missing: bool = False) -> None:
    if not (tables_dir / "Figure15.txt").exists():
        require_tables(tables_dir, ("Figure15.txt",), allow_missing=allow_missing)
        return
    rows = read_tsv(tables_dir / "Figure15.txt")
    tasks = [row["Task"] for row in rows]
    active_methods = [
        method
        for method in BAR_METHODS
        if any(not math.isnan(fnum(first_value(row, method[0]))) for row in rows)
    ]

    x = np.arange(len(tasks))
    width = min(0.16, 0.75 / max(1, len(active_methods)))
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    for idx, (cols, label, color) in enumerate(active_methods):
        values = [fnum(first_value(row, cols)) for row in rows]
        offset = (idx - (len(active_methods) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_yscale("log")
    ax.set_ylabel("Normalized Runtime (x Base)", fontsize=10)
    ax.set_xlabel("Task", fontsize=10)
    ax.set_title("Profiling / Scheduling Overhead", fontsize=11)
    ax.legend(ncol=len(active_methods), fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    style_axes(ax)
    fig.subplots_adjust(top=0.80)
    save_figure(fig, output_dir, "Figure15", formats)


def plot_model_accuracy(tables_dir: Path, output_dir: Path, formats: Iterable[str], allow_missing: bool = False) -> None:
    if not (tables_dir / "Figure16.txt").exists():
        require_tables(tables_dir, ("Figure16.txt",), allow_missing=allow_missing)
        return
    rows = read_tsv(tables_dir / "Figure16.txt")
    series = ["16", "512", "8192"]
    outlier_indices = {
        "512": {1, 2},
        "8192": {-4, -3},
    }

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=False)
    for ax, m_value in zip(axes, series):
        points: list[tuple[float, float]] = []
        for row in rows:
            x = fnum(row.get(f"sm-{m_value}"))
            y = fnum(row.get(f"lat-{m_value}-ms"))
            if not (math.isnan(x) or math.isnan(y)):
                points.append((x, y))
        points.sort(key=lambda pair: pair[0])
        drops = outlier_indices.get(m_value, set())
        normalized_drops = {idx if idx >= 0 else len(points) + idx for idx in drops}
        points = [point for idx, point in enumerate(points) if idx not in normalized_drops]
        ax.scatter(
            [p[0] for p in points],
            [p[1] for p in points],
            s=42,
            color="#d62728",
            edgecolor="black",
            linewidth=0.5,
            label="Measured",
        )
        ax.set_title(f"M={m_value}, N=K=65536", fontsize=10)
        ax.set_xlabel("#SMs", fontsize=10)
        ax.set_ylabel("Latency (ms)", fontsize=10)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        style_axes(ax)
    fig.subplots_adjust(wspace=0.30)
    save_figure(fig, output_dir, "Figure16", formats)


def main() -> int:
    args = parse_args()
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )

    plot_latency_throughput(
        args.tables_dir,
        args.output_dir,
        formats,
        ("Figure10,12(a).txt", "Figure10,12(b).txt"),
        "Figure10",
        "ANNS Throughput ($10^5$ Op/s)",
        allow_missing=args.allow_missing,
    )
    plot_latency_throughput(
        args.tables_dir,
        args.output_dir,
        formats,
        ("Figure11(a).txt", "Figure11(b).txt"),
        "Figure11",
        "MM Throughput ($10^5$ Op/s)",
        allow_missing=args.allow_missing,
    )
    plot_query_latency(args.tables_dir, args.output_dir, formats, allow_missing=args.allow_missing)
    plot_pd_figure(args.tables_dir, args.output_dir, formats, "Figure13.txt", "Figure13", allow_missing=args.allow_missing)
    plot_pd_figure(args.tables_dir, args.output_dir, formats, "Figure14.txt", "Figure14", allow_missing=args.allow_missing)
    plot_overhead(args.tables_dir, args.output_dir, formats, allow_missing=args.allow_missing)
    plot_model_accuracy(args.tables_dir, args.output_dir, formats, allow_missing=args.allow_missing)

    print(f"Wrote figures to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
