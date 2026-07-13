#!/usr/bin/env python3
"""Aggregate prefill and decode step times from a log file."""

import argparse
import re
from typing import List

_PREFILL_PATTERN = re.compile(r"^prefill step time:\s*([0-9.+-eE]+)\s+seconds")
_DECODE_PATTERN = re.compile(r"^decode step time:\s*([0-9.+-eE]+)\s+seconds")


def _parse_times(lines: List[str]) -> tuple[List[float], List[float]]:
    prefill: List[float] = []
    decode: List[float] = []
    for line in lines:
        if (match := _PREFILL_PATTERN.search(line)):
            prefill.append(float(match.group(1)))
            continue
        if (match := _DECODE_PATTERN.search(line)):
            decode.append(float(match.group(1)))
    return prefill, decode


def _summarize(values: List[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    total = sum(values)
    avg = total / len(values)
    mx = max(values)
    return total, avg, mx


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize prefill and decode step times.")
    parser.add_argument("log_file", help="Path to the log file to parse")
    args = parser.parse_args()

    with open(args.log_file, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    prefill_times, decode_times = _parse_times(lines)

    prefill_total, prefill_avg, prefill_max = _summarize(prefill_times)
    decode_total, decode_avg, decode_max = _summarize(decode_times)

    print(f"prefill_sum_seconds: {prefill_total:.6f}")
    if prefill_times:
        print(f"prefill_avg_seconds: {prefill_avg:.6f} (count={len(prefill_times)})")
    else:
        print("prefill_avg_seconds: N/A (count=0)")
    print(f"prefill_max_seconds: {prefill_max:.6f}")

    print(f"decode_sum_seconds: {decode_total:.6f}")
    if decode_times:
        print(f"decode_avg_seconds: {decode_avg:.6f} (count={len(decode_times)})")
    else:
        print("decode_avg_seconds: N/A (count=0)")
    print(f"decode_max_seconds: {decode_max:.6f}")


if __name__ == "__main__":
    main()
