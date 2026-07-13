#!/usr/bin/env python3
"""Summarize colocated workload logs into a table.

The script scans `scripts/logs/colocate` for experiment folders such as
`llm-1-mm-12` or `rps-1-llama-3-8b-20000`, parses the scheduler logs inside,
and prints a table containing:

* Scenario / experiment identifiers
* Scheduler name (`smsched`, `mps-30`, `timeslice`, ...)
* RPS for the LLM task and the colocated task
* LLM average finished latency (ms)
* Completion count for the colocated task (matmuls, GGNN queries, ...)

Run with `--help` for the full list of options. In addition to printing the
summary table, the script writes per-scenario, tab-separated datasets where each
row corresponds to a unique `other_rps` value and every scheduler contributes its
own `llm_latency_ms` and `completed` columns (stored under
`scripts/analysis/colocate` by default).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

RESULT_HEADER = "=== Results ==="
COMPLETED_PATTERN = re.compile(r"Completed[^:]*:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
COMMAND_TIME_PATTERN = re.compile(r"\s--time\s+([0-9]+(?:\.[0-9]+)?)")
AVG_FIRST_PATTERN = re.compile(r"Avg first token \(ms\):\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
AVG_FINISHED_PATTERN = re.compile(r"Avg finished latency \(ms\):\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
TIME_PER_TOKEN_PATTERN = re.compile(r"Time per token:\s*([0-9]+(?:\.[0-9]+)?)\s*ms/token", re.IGNORECASE)
REQUEST_LATENCY_PATTERN = re.compile(r"Request latency:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE)
GENERATED_TOKENS_PATTERN = re.compile(r"Generated tokens:\s*([0-9]+)", re.IGNORECASE)
TIME_TO_FIRST_TOKEN_PATTERN = re.compile(r"Time to first token:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE)
EXPERIMENT_PATTERN = re.compile(
    r"(?P<task1>[a-zA-Z0-9_]+)-(?P<rps1>\d+(?:\.\d+)?)-(?P<task2>[a-zA-Z0-9_]+)-(?P<rps2>\d+(?:\.\d+)?)$"
)
REPO_ROOT = Path(__file__).resolve().parents[2]
COLUMN_SEPARATOR = "\t"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[0] / "colocate"
SLO_TABLE_FILENAME = "slo_summary.tsv"

SLO_configs = {
    "llm-1-ggnn": [25, 27, 29, 31, 33], 
    "llm-4-ggnn": [40, 50, 60, 70, 80],
    "llm-1-mm":   [25, 30, 35, 40, 45], 
    "llm-4-mm":   [40, 50, 60, 70, 80],
}

GGNN_TASK_TOKEN = "ggnn"
GGNN_SCALE_FACTOR = 0.1
MM_TASK_TOKEN = "mm"
MM_SCALE_FACTOR = 0.01


def scale_ggnn_metric(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return value * GGNN_SCALE_FACTOR


def scale_mm_metric(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return value * MM_SCALE_FACTOR


def _format_slo_label_ms(value_ms: float) -> str:
    if abs(value_ms - round(value_ms)) < 1e-9:
        value_str = str(int(round(value_ms)))
    else:
        value_str = f"{value_ms:.3f}".rstrip("0").rstrip(".")
    return f"p99<={value_str}ms"


def build_slo_targets_from_config(values: Sequence[float]) -> List[Tuple[str, float, str]]:
    targets: List[Tuple[str, float, str]] = []
    for raw in values:
        try:
            threshold_ms = float(raw)
        except (TypeError, ValueError):
            continue
        label = _format_slo_label_ms(threshold_ms)
        targets.append(("llm_p99_latency_ms", threshold_ms, label))
    return targets


def get_slo_targets_for_key(config_key: Optional[str]) -> List[Tuple[str, float, str]]:
    if not config_key:
        return []
    values = SLO_configs.get(config_key)
    if not values:
        return []
    return build_slo_targets_from_config(values)


def collect_slo_targets_from_rows(rows: Sequence[SummaryRow]) -> List[Tuple[str, float, str]]:
    targets_by_threshold: Dict[float, Tuple[str, float, str]] = {}
    for row in rows:
        key = derive_slo_config_key(row.scenario, row.llm_rps)
        for field_name, threshold_ms, label in get_slo_targets_for_key(key):
            if threshold_ms not in targets_by_threshold:
                targets_by_threshold[threshold_ms] = (field_name, threshold_ms, label)
    ordered_thresholds = sorted(targets_by_threshold.keys())
    return [targets_by_threshold[t] for t in ordered_thresholds]


def derive_slo_config_key(scenario: str, llm_rps: Optional[float]) -> str:
    if llm_rps is None:
        return scenario
    llm_rps_tag = format_rps_for_filename(llm_rps)
    return build_scenario_name_with_llm_rps(scenario, llm_rps_tag)

@dataclass
class SummaryRow:
    scenario: str
    experiment: str
    scheduler: str
    llm_rps: Optional[float]
    secondary_rps: Optional[float]
    llm_avg_first_token_ms: Optional[float]
    llm_avg_finished_ms: Optional[float]
    llm_p95_latency_ms: Optional[float]
    llm_p99_latency_ms: Optional[float]
    secondary_task: Optional[str]
    secondary_completed_label: Optional[str]
    secondary_completed_value: Optional[float]
    llm_relative_throughput: Optional[float]
    secondary_relative_throughput: Optional[float]
    log_path: Path

    @property
    def secondary_completed_display(self) -> str:
        if self.secondary_completed_label is None:
            return "-"
        if self.secondary_completed_value is None:
            return f"{self.secondary_completed_label}=n/a"
        value = self.secondary_completed_value
        if abs(value - round(value)) < 1e-6:
            value_str = str(int(round(value)))
        else:
            value_str = f"{value:.2f}"
        return f"{self.secondary_completed_label}={value_str}"


@dataclass
class StandaloneBaselines:
    llm_latency_ms: Optional[float]
    secondary_completed: Dict[Tuple[str, Optional[float]], Optional[float]]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize colocated workload logs into a table.")
    default_root = Path(__file__).resolve().parents[1] / "logs" / "colocate"
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=default_root,
        help="Path to scripts/logs/colocate (defaults to that location relative to this script).",
    )
    parser.add_argument(
        "--show-first-token",
        action="store_true",
        help="Include the LLM avg first token latency column in the output table.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="If set, limit the number of rows that get printed (useful for quick inspections).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store per-scenario tables and plots (defaults to scripts/analysis/colocate).",
    )
    parser.add_argument(
        "--slo-secondary-task",
        type=str,
        default="ggnn",
        help="Only consider rows whose secondary task matches this token when building the SLO table (defaults to 'ggnn').",
    )
    return parser.parse_args(argv)


def extract_results_sections(content: str) -> dict[str, List[str]]:
    idx = content.rfind(RESULT_HEADER)
    if idx == -1:
        return {}

    lines = content[idx:].splitlines()[1:]
    sections: dict[str, List[str]] = {}
    current_section: Optional[str] = None

    for line in lines:
        stripped = line.rstrip("\n")
        if not stripped.strip():
            continue
        if not stripped.startswith(" ") and stripped.endswith("stats:"):
            current_section = stripped[:-1]
            sections[current_section] = []
            continue
        if stripped.startswith("  ") and current_section:
            sections[current_section].append(stripped.strip())
            continue
        # As soon as we hit an unrelated line we stop parsing the summary block.
        break

    return sections


def parse_llm_metrics(section_lines: List[str]) -> Tuple[Optional[float], Optional[float]]:
    if not section_lines:
        return None, None
    block = "\n".join(section_lines)
    first_token = _extract_float(block, AVG_FIRST_PATTERN)
    finished = _extract_float(block, AVG_FINISHED_PATTERN)
    return first_token, finished


def parse_secondary_completion(section_lines: List[str], duration_seconds: float = 180.0) -> Tuple[Optional[str], Optional[float]]:
    if not section_lines:
        return None, None
    if duration_seconds <= 0:
        duration_seconds = 180.0
    for line in section_lines:
        match = COMPLETED_PATTERN.search(line)
        if match:
            try:
                return line.split(":")[0], float(match.group(1)) / duration_seconds
            except ValueError:
                return line.split(":")[0], None
    return None, None


def _extract_float(text: str, pattern: re.Pattern[str]) -> Optional[float]:
    match = pattern.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def compute_weighted_average(values: Sequence[float], weights: Sequence[float]) -> Optional[float]:
    if not values or not weights or len(values) != len(weights):
        return None
    weighted_sum = 0.0
    total_weight = 0.0
    for value, weight in zip(values, weights):
        if weight is None or weight <= 0:
            continue
        weighted_sum += value * weight
        total_weight += weight
    if total_weight <= 0:
        return None
    return weighted_sum / total_weight


def extract_llm_token_latency_stats(content: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    latencies: List[float] = []
    weights: List[float] = []
    current_tokens: Optional[float] = None

    for line in content.splitlines():
        time_to_first_token_match = TIME_TO_FIRST_TOKEN_PATTERN.search(line)
        if time_to_first_token_match:
            time_to_first_token = float(time_to_first_token_match.group(1))
        request_latency_match = REQUEST_LATENCY_PATTERN.search(line)
        if request_latency_match:
            request_latency = float(request_latency_match.group(1))
        tokens_match = GENERATED_TOKENS_PATTERN.search(line)
        if tokens_match:
            current_tokens = float(tokens_match.group(1)) - 1  # exclude first token
            latency = (request_latency - time_to_first_token) / current_tokens if current_tokens and current_tokens > 0 else 0.0
            latencies.append(latency)
            weight = current_tokens if current_tokens is not None and current_tokens > 0 else 0.0
            weights.append(weight)
            current_tokens = None

    avg_latency = compute_weighted_average(latencies, weights)
    p95 = compute_percentile(latencies, 0.95, weights)
    p99 = compute_percentile(latencies, 0.99, weights)
    return avg_latency, p95, p99

def _value_at_weight_rank(pairs: Sequence[Tuple[float, float]], rank_index: float) -> float:
    cumulative = 0.0
    for value, weight in pairs:
        if weight <= 0:
            continue
        cumulative += weight
        if rank_index < cumulative:
            return value
    return pairs[-1][0]


def _compute_unweighted_percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    rank = percentile * (len(sorted_values) - 1)
    lower_idx = int(rank)
    upper_idx = min(lower_idx + 1, len(sorted_values) - 1)
    fraction = rank - lower_idx
    lower_value = sorted_values[lower_idx]
    upper_value = sorted_values[upper_idx]
    return lower_value + (upper_value - lower_value) * fraction


def compute_percentile(
    values: Sequence[float],
    percentile: float,
    weights: Optional[Sequence[float]] = None,
) -> Optional[float]:
    if not values:
        return None
    if weights is None:
        return _compute_unweighted_percentile(values, percentile)
    if len(values) != len(weights):
        return None

    sanitized_pairs: List[Tuple[float, float]] = []
    for value, weight in zip(values, weights):
        w = 0.0 if weight is None else float(weight)
        if w <= 0:
            sanitized_pairs.append((value, 0.0))
            continue
        sanitized_pairs.append((value, w))

    positive_pairs = [(value, w) for value, w in sanitized_pairs if w > 0]
    if not positive_pairs:
        return _compute_unweighted_percentile(values, percentile)

    positive_pairs.sort(key=lambda pair: pair[0])
    total_weight = sum(weight for _, weight in positive_pairs)
    if total_weight <= 0:
        return _compute_unweighted_percentile([value for value, _ in positive_pairs], percentile)
    if len(positive_pairs) == 1:
        return positive_pairs[0][0]

    max_rank = max(total_weight - 1.0, 0.0)
    rank = percentile * max_rank
    lower_rank = math.floor(rank)
    upper_rank = math.ceil(rank)
    max_index = math.floor(max_rank)
    if upper_rank > max_index:
        upper_rank = max_index
    fraction = rank - lower_rank
    lower_value = _value_at_weight_rank(positive_pairs, lower_rank)
    upper_value = _value_at_weight_rank(positive_pairs, upper_rank)
    return lower_value + (upper_value - lower_value) * fraction


def normalize_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", token.lower())


def infer_tasks(scenario: str) -> Tuple[str, Optional[str]]:
    parts = [p for p in scenario.split("-") if p]
    if not parts:
        return scenario, None
    primary = parts[0]
    secondary = parts[1] if len(parts) > 1 else None
    return primary, secondary


def parse_rps_tokens(name: str) -> Tuple[Optional[float], Optional[float]]:
    tokens = [tok for tok in name.replace("_", "-").split("-") if tok]
    numeric_tokens: List[float] = []
    for tok in tokens:
        try:
            numeric_tokens.append(float(tok))
        except ValueError:
            continue
    if len(numeric_tokens) >= 2:
        return numeric_tokens[0], numeric_tokens[-1]
    if len(numeric_tokens) == 1:
        return numeric_tokens[0], None
    return None, None


def discover_log_files(logs_root: Path) -> List[Tuple[str, str, Path]]:
    rows: List[Tuple[str, str, Path]] = []
    for scenario_dir in sorted(logs_root.iterdir()):
        if not scenario_dir.is_dir():
            continue
        for experiment_dir in sorted(scenario_dir.iterdir()):
            if not experiment_dir.is_dir():
                continue
            if not EXPERIMENT_PATTERN.match(experiment_dir.name):
                continue
            for log_path in sorted(experiment_dir.glob("*.log")):
                rows.append((scenario_dir.name, experiment_dir.name, log_path))
    return rows


def infer_duration_from_manifest(log_path: Path) -> Optional[float]:
    for parent in log_path.parents:
        manifest_path = parent / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            duration = float(payload.get("duration_seconds", 180.0))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            break
        return duration if duration > 0 else None
    return None


def infer_duration_seconds(log_path: Path) -> float:
    err_path = log_path.with_suffix(".err")
    try:
        content = err_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        manifest_duration = infer_duration_from_manifest(log_path)
        return manifest_duration if manifest_duration is not None else 180.0
    match = COMMAND_TIME_PATTERN.search(content)
    if not match:
        manifest_duration = infer_duration_from_manifest(log_path)
        return manifest_duration if manifest_duration is not None else 180.0
    try:
        duration = float(match.group(1))
    except ValueError:
        manifest_duration = infer_duration_from_manifest(log_path)
        return manifest_duration if manifest_duration is not None else 180.0
    return duration if duration > 0 else 180.0


def collect_rows(logs_root: Path, baselines: StandaloneBaselines) -> List[SummaryRow]:
    rows: List[SummaryRow] = []
    for scenario_name, experiment_name, log_path in discover_log_files(logs_root):
        try:
            content = log_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as err:
            print(f"Warning: failed to read {log_path}: {err}", file=sys.stderr)
            continue

        sections = extract_results_sections(content)
        llm_section = sections.get("LLM stats")
        if not llm_section:
            continue
        llm_first_token, legacy_llm_finished = parse_llm_metrics(llm_section)
        llm_avg_latency_ms, llm_p95_latency, llm_p99_latency = extract_llm_token_latency_stats(content)
        if llm_avg_latency_ms is None:
            llm_avg_latency_ms = legacy_llm_finished

        primary_task, scenario_secondary = infer_tasks(scenario_name)
        normalized_secondary = normalize_token(scenario_secondary) if scenario_secondary else None

        secondary_section_name = None
        if normalized_secondary:
            for section_name in sections:
                if normalize_token(section_name.replace(" stats", "")) == normalized_secondary:
                    secondary_section_name = section_name
                    break
        if secondary_section_name is None:
            # fall back to the first non-LLM section, if any
            for section_name in sections:
                if section_name != "LLM stats":
                    secondary_section_name = section_name
                    break
        secondary_lines = sections.get(secondary_section_name, []) if secondary_section_name else []
        duration_seconds = infer_duration_seconds(log_path)
        secondary_label, secondary_value = parse_secondary_completion(secondary_lines, duration_seconds)

        llm_rps, secondary_rps = parse_rps_tokens(experiment_name)

        secondary_task_name = (
            scenario_secondary
            or (secondary_section_name.replace(" stats", "") if secondary_section_name else None)
        )

        normalized_secondary_task = normalize_token(secondary_task_name) if secondary_task_name else None
        normalized_secondary_rps = normalize_rps_value(secondary_rps)
        secondary_baseline_value: Optional[float] = None
        if normalized_secondary_task:
            key_exact = (normalized_secondary_task, normalized_secondary_rps)
            fallback_key = (normalized_secondary_task, None)
            if key_exact in baselines.secondary_completed:
                secondary_baseline_value = baselines.secondary_completed[key_exact]
            elif fallback_key in baselines.secondary_completed:
                secondary_baseline_value = baselines.secondary_completed[fallback_key]
        is_ggnn_secondary = normalized_secondary_task == GGNN_TASK_TOKEN
        if is_ggnn_secondary:
            secondary_value = scale_ggnn_metric(secondary_value)
            secondary_baseline_value = scale_ggnn_metric(secondary_baseline_value)
            secondary_rps = scale_ggnn_metric(secondary_rps)
        is_mm_secondary = normalized_secondary_task == MM_TASK_TOKEN
        if is_mm_secondary:
            secondary_value = scale_mm_metric(secondary_value)
            secondary_baseline_value = scale_mm_metric(secondary_baseline_value)
            secondary_rps = scale_mm_metric(secondary_rps)
        llm_relative_throughput = safe_divide(baselines.llm_latency_ms, llm_avg_latency_ms)
        secondary_relative_throughput = safe_divide(
            secondary_value,
            secondary_baseline_value,
        )

        rows.append(
            SummaryRow(
                scenario=scenario_name,
                experiment=experiment_name,
                scheduler=log_path.stem,
                llm_rps=llm_rps,
                secondary_rps=secondary_rps,
                llm_avg_first_token_ms=llm_first_token,
                llm_avg_finished_ms=llm_avg_latency_ms,
                llm_p95_latency_ms=llm_p95_latency,
                llm_p99_latency_ms=llm_p99_latency,
                secondary_task=secondary_task_name,
                secondary_completed_label=secondary_label,
                secondary_completed_value=secondary_value,
                llm_relative_throughput=llm_relative_throughput,
                secondary_relative_throughput=secondary_relative_throughput,
                log_path=log_path,
            )
        )
    return rows


def group_rows_by_scenario(rows: List[SummaryRow]) -> Dict[str, List[SummaryRow]]:
    grouped: Dict[str, List[SummaryRow]] = defaultdict(list)
    for row in rows:
        grouped[row.scenario].append(row)
    for scenario_rows in grouped.values():
        scenario_rows.sort(key=lambda r: (sort_other_rps_value(r), r.experiment, r.scheduler))
    return grouped


def group_rows_by_llm_rps(rows: List[SummaryRow]) -> Dict[Optional[float], List[SummaryRow]]:
    grouped: Dict[Optional[float], List[SummaryRow]] = defaultdict(list)
    for row in rows:
        grouped[normalize_rps_value(row.llm_rps)].append(row)
    return grouped


def sort_other_rps_value(row: SummaryRow) -> float:
    return row.secondary_rps if row.secondary_rps is not None else float("inf")


def format_table(rows: List[SummaryRow], include_first_token: bool) -> str:
    if not rows:
        return "No results found under the provided logs directory."

    columns = [
        ("scenario", lambda r: r.scenario),
        ("experiment", lambda r: r.experiment),
        ("scheduler", lambda r: r.scheduler),
        ("llm_rps", lambda r: format_number(r.llm_rps)),
        ("other_rps", lambda r: format_number(r.secondary_rps)),
    ]
    if include_first_token:
        columns.append(("llm_first_token_ms", lambda r: format_number(r.llm_avg_first_token_ms, precision=2)))
    columns.append(("llm_finished_ms", lambda r: format_number(r.llm_avg_finished_ms, precision=2)))
    columns.append(("llm_p95_ms", lambda r: format_number(r.llm_p95_latency_ms, precision=2)))
    columns.append(("llm_p99_ms", lambda r: format_number(r.llm_p99_latency_ms, precision=2)))
    columns.append(("llm_rel_throughput", lambda r: format_number(r.llm_relative_throughput, precision=3)))
    columns.append(("other_task", lambda r: r.secondary_task or "-"))
    columns.append(("other_completed", lambda r: r.secondary_completed_display))
    columns.append(("other_rel_throughput", lambda r: format_number(r.secondary_relative_throughput, precision=3)))
    columns.append(("log", lambda r: str(r.log_path.relative_to(REPO_ROOT)) if REPO_ROOT in r.log_path.parents else str(r.log_path)))

    table_rows: List[List[str]] = []
    header = [name for name, _ in columns]
    for row in rows:
        table_rows.append([formatter(row) for _, formatter in columns])

    widths = [len(h) for h in header]
    for row in table_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def format_row(cells: List[str]) -> str:
        return COLUMN_SEPARATOR.join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells))

    lines = [format_row(header), format_row(["-" * w for w in widths])]
    lines.extend(format_row(row) for row in table_rows)
    return "\n".join(lines)


def format_number(value: Optional[float], precision: int = 0) -> str:
    if value is None:
        return "-"
    if precision == 0 and abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.{precision}f}"


def format_plain_number(value: Optional[float], precision: int = 3) -> str:
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.{precision}f}"


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def safe_divide(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None:
        return None
    if abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def normalize_rps_value(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(value, 6)


def extract_rps_from_name(name: str) -> Optional[float]:
    first, last = parse_rps_tokens(name)
    if last is not None:
        return last
    return first


def load_standalone_baselines(logs_root: Path) -> StandaloneBaselines:
    llm_latency_ms: Optional[float] = None
    secondary_completed: Dict[Tuple[str, Optional[float]], Optional[float]] = {}

    top_level_dirs = sorted([path for path in logs_root.iterdir() if path.is_dir()])
    for top_dir in top_level_dirs:
        normalized_dir = normalize_token(top_dir.name)
        if not normalized_dir:
            continue
        standalone_logs = sorted(top_dir.rglob("standalone.log"))
        for standalone_log in standalone_logs:
            try:
                content = standalone_log.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            sections = extract_results_sections(content)
            if not sections:
                continue
            if normalized_dir == "llm":
                if llm_latency_ms is not None:
                    break
                llm_token_avg, _, _ = extract_llm_token_latency_stats(content)
                if llm_token_avg is None:
                    llm_section = sections.get("LLM stats")
                    _, legacy_finished = parse_llm_metrics(llm_section or [])
                    llm_token_avg = legacy_finished
                if llm_token_avg is not None:
                    llm_latency_ms = llm_token_avg
                    break
            else:
                parent_name = standalone_log.parent.name
                rps_value = normalize_rps_value(extract_rps_from_name(parent_name))
                key = (normalized_dir, rps_value)
                if key in secondary_completed:
                    continue
                baseline_value = None
                for section_name, lines in sections.items():
                    if section_name == "LLM stats":
                        continue
                    _, value = parse_secondary_completion(lines)
                    if value is not None:
                        baseline_value = value
                        break
                if baseline_value is not None:
                    secondary_completed[key] = baseline_value
        # ensure we also break outer loop if we've collected needed llm baseline
        if normalized_dir == "llm" and llm_latency_ms is not None:
            continue

    return StandaloneBaselines(llm_latency_ms=llm_latency_ms, secondary_completed=secondary_completed)


def write_scenario_table(scenario: str, rows: List[SummaryRow], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / f"{sanitize_filename(scenario)}.tsv"

    schedulers = sorted({row.scheduler for row in rows})
    other_rps_values = sorted({row.secondary_rps for row in rows if row.secondary_rps is not None})

    cell_map: Dict[Tuple[float, str], SummaryRow] = {}
    for row in rows:
        if row.secondary_rps is None:
            continue
        cell_map[(row.secondary_rps, row.scheduler)] = row

    headers = ["other_rps"]
    for scheduler in schedulers:
        headers.append(f"{scheduler}_llm_latency_ms")
        headers.append(f"{scheduler}_llm_p95_ms")
        headers.append(f"{scheduler}_llm_p99_ms")
        headers.append(f"{scheduler}_completed")
        headers.append(f"{scheduler}_llm_rel_throughput")
        headers.append(f"{scheduler}_other_rel_throughput")

    table_cells: List[List[str]] = []
    for other_rps in other_rps_values:
        row_cells: List[str] = [format_plain_number(other_rps)]
        for scheduler in schedulers:
            data = cell_map.get((other_rps, scheduler))
            if data is None:
                row_cells.extend(["", "", "", "", "", ""])
            else:
                row_cells.append(format_plain_number(data.llm_avg_finished_ms, precision=2))
                row_cells.append(format_plain_number(data.llm_p95_latency_ms, precision=2))
                row_cells.append(format_plain_number(data.llm_p99_latency_ms, precision=2))
                row_cells.append(format_plain_number(data.secondary_completed_value))
                row_cells.append(format_plain_number(data.llm_relative_throughput, precision=3))
                row_cells.append(format_plain_number(data.secondary_relative_throughput, precision=3))
        table_cells.append(row_cells)

    widths = [len(h) for h in headers]
    for cells in table_cells:
        for idx, cell in enumerate(cells):
            widths[idx] = max(widths[idx], len(cell))

    def format_row(cells: List[str]) -> str:
        padded = [cell.ljust(widths[idx]) for idx, cell in enumerate(cells)]
        return COLUMN_SEPARATOR.join(padded)

    with table_path.open("w", encoding="utf-8") as handle:
        handle.write(format_row(headers) + "\n")
        for cells in table_cells:
            handle.write(format_row(cells) + "\n")
    return table_path


def build_slo_summary(
    rows: List[SummaryRow],
    secondary_task_filter: Optional[str],
    slo_targets: Sequence[Tuple[str, float, str]],
) -> Tuple[List[str], List[List[str]], Dict[str, float]]:
    normalized_filter = normalize_token(secondary_task_filter) if secondary_task_filter else None
    schedulers = sorted({row.scheduler for row in rows})
    headers = ["scheduler"]
    targets = list(slo_targets)
    metric_thresholds: Dict[str, float] = {}
    for _, threshold_ms, label in targets:
        completed_header = f"{label}_completed"
        other_rps_header = f"{label}_other_rps"
        headers.append(completed_header)
        headers.append(other_rps_header)
        metric_thresholds[completed_header] = threshold_ms
        metric_thresholds[other_rps_header] = threshold_ms
    table_rows: List[List[str]] = []

    for scheduler in schedulers:
        scheduler_rows = [row for row in rows if row.scheduler == scheduler]
        cells: List[str] = [scheduler]
        for field_name, threshold_ms, _ in targets:
            best_row: Optional[SummaryRow] = None
            best_completed: Optional[float] = None
            for row in scheduler_rows:
                if normalized_filter is not None:
                    row_secondary = normalize_token(row.secondary_task or "")
                    if row_secondary != normalized_filter:
                        continue
                metric_value = getattr(row, field_name)
                completed_value = row.secondary_completed_value
                if metric_value is None or completed_value is None:
                    continue
                if metric_value <= threshold_ms:
                    if best_completed is None or completed_value > best_completed:
                        best_completed = completed_value
                        best_row = row
            if best_row is None or best_completed is None:
                cells.extend(["0", "0"])
            else:
                completed_str = format_plain_number(best_completed)
                other_rps_str = format_plain_number(best_row.secondary_rps)
                cells.extend([
                    completed_str if completed_str else "0",
                    other_rps_str if other_rps_str else "0",
                ])
        table_rows.append(cells)

    completed_indices = [idx for idx, name in enumerate(headers) if name.endswith("_completed")]
    other_rps_indices = [idx for idx, name in enumerate(headers) if name.endswith("_other_rps")]
    reordered_headers = [headers[0]]
    reordered_headers.extend(headers[idx] for idx in completed_indices)
    reordered_headers.extend(headers[idx] for idx in other_rps_indices)

    reordered_rows: List[List[str]] = []
    for row in table_rows:
        new_row = [row[0]]
        new_row.extend(row[idx] for idx in completed_indices)
        new_row.extend(row[idx] for idx in other_rps_indices)
        reordered_rows.append(new_row)

    return reordered_headers, reordered_rows, metric_thresholds


def format_generic_table(headers: List[str], rows: List[List[str]], separator: str = COLUMN_SEPARATOR) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def format_row(cells: List[str]) -> str:
        padded = [cells[idx].ljust(widths[idx]) for idx in range(len(headers))]
        return separator.join(padded)

    # lines = [format_row(headers), format_row(["-" * w for w in widths])]
    lines = [format_row(headers)]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def format_rps_for_filename(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    formatted = f"{value:.3f}".rstrip("0").rstrip(".")
    return formatted.replace(".", "_")


def build_scenario_name_with_llm_rps(scenario: str, llm_rps_tag: Optional[str]) -> str:
    if not llm_rps_tag:
        return scenario
    parts = scenario.split("-", 1)
    first = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    if rest:
        return f"{first}-{llm_rps_tag}-{rest}"
    return f"{first}-{llm_rps_tag}"


def transpose_table(
    headers: List[str],
    rows: List[List[str]],
    row_label_name: str = "metric",
    value_column_name: Optional[str] = None,
    metric_values: Optional[Dict[str, float]] = None,
) -> Tuple[List[str], List[List[str]]]:
    if not rows:
        transposed_header = [row_label_name]
        if value_column_name and metric_values is not None:
            transposed_header.append(value_column_name)
        return transposed_header, rows
    column_headers = [row[0] for row in rows]
    transposed_header = [row_label_name]
    include_value_column = value_column_name is not None and metric_values is not None
    if include_value_column:
        transposed_header.append(value_column_name or "value")
    transposed_header.extend(column_headers)
    transposed_rows: List[List[str]] = []
    for col_idx in range(1, len(headers)):
        label = headers[col_idx]
        new_row = [label]
        if include_value_column:
            value = metric_values.get(label)
            if value is None:
                new_row.append("")
            else:
                new_row.append(format_plain_number(value, precision=3))
        for row in rows:
            if col_idx < len(row):
                new_row.append(row[col_idx])
            else:
                new_row.append("")
        transposed_rows.append(new_row)
    return transposed_header, transposed_rows


def determine_slo_filter_for_scenario(scenario: str, default_filter: Optional[str]) -> Optional[str]:
    primary, secondary = infer_tasks(scenario)
    _ = primary  # unused but captured for clarity
    if secondary:
        return secondary
    return default_filter


def write_slo_summary_table(
    rows: List[SummaryRow],
    output_dir: Path,
    secondary_task_filter: Optional[str],
    filename: str = SLO_TABLE_FILENAME,
    slo_targets: Optional[Sequence[Tuple[str, float, str]]] = None,
    slo_config_key: Optional[str] = None,
) -> Tuple[Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / filename
    targets: Sequence[Tuple[str, float, str]]
    if slo_targets is not None:
        targets = slo_targets
    else:
        targets = get_slo_targets_for_key(slo_config_key)
    headers, table_rows, metric_thresholds = build_slo_summary(rows, secondary_task_filter, targets)

    value_column_name = "value_ms" if metric_thresholds else None
    metric_values = metric_thresholds if metric_thresholds else None
    transposed_headers, transposed_rows = transpose_table(
        headers,
        table_rows,
        row_label_name="metric",
        value_column_name=value_column_name,
        metric_values=metric_values,
    )
    pretty_table = format_generic_table(transposed_headers, transposed_rows, separator="\t")

    with table_path.open("w", encoding="utf-8") as handle:
        handle.write(pretty_table + "\n")

    return table_path, pretty_table




def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    logs_root = args.logs_root

    if not logs_root.is_dir():
        print(f"Error: logs root does not exist or is not a directory: {logs_root}", file=sys.stderr)
        return 1

    baselines = load_standalone_baselines(logs_root)
    rows = collect_rows(logs_root, baselines)
    rows.sort(key=lambda r: (sort_other_rps_value(r), r.scenario, r.experiment, r.scheduler))

    if not rows:
        return 0

    grouped = group_rows_by_scenario(rows)
    tables_dir = args.output_dir
    generated_tables: List[Path] = []
    scenario_slo_tables: List[Path] = []

    for scenario, scenario_rows in grouped.items():
        llm_groups = group_rows_by_llm_rps(scenario_rows)
        sorted_llm_keys = sorted(
            llm_groups.keys(),
            key=lambda v: float("inf") if v is None else v,
        )
        for llm_key in sorted_llm_keys:
            rows_for_llm = llm_groups[llm_key]
            llm_rps_tag = format_rps_for_filename(llm_key) if llm_key is not None else None
            scenario_output_name = build_scenario_name_with_llm_rps(scenario, llm_rps_tag)
            sanitized_base = sanitize_filename(scenario_output_name)

            table_path = write_scenario_table(scenario_output_name, rows_for_llm, tables_dir)
            generated_tables.append(table_path)
            scenario_filter = determine_slo_filter_for_scenario(scenario, args.slo_secondary_task)
            scenario_slo_filename = f"{sanitized_base}-slo.tsv"
            scenario_slo_path, _ = write_slo_summary_table(
                rows_for_llm,
                tables_dir,
                scenario_filter,
                filename=scenario_slo_filename,
                slo_config_key=scenario_output_name,
            )
            scenario_slo_tables.append(scenario_slo_path)

    aggregate_slo_targets = collect_slo_targets_from_rows(rows)
    slo_table_path, slo_table_text = write_slo_summary_table(
        rows,
        tables_dir,
        args.slo_secondary_task,
        slo_targets=aggregate_slo_targets,
    )

    print()
    print("SLO summary across schedulers:")
    print(slo_table_text)

    print()
    print(f"Generated {len(generated_tables)} per-scenario tables under {tables_dir}.")
    print(f"Generated {len(scenario_slo_tables)} per-scenario SLO tables under {tables_dir}.")
    print(f"Aggregate SLO summary saved to {slo_table_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
