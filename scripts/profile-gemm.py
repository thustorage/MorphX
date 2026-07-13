import argparse
import json
import os
import shlex
import subprocess
import sys

import torch
import torch.nn as nn


n_ci = 10
n_mi = 10


def parse_smsched_assignments(raw_value):
    assignments = {}
    if not raw_value:
        return assignments
    for token in shlex.split(raw_value):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        assignments[key] = value
    return assignments


def create_context(log_stream=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run profile-gemm.")

    device = torch.device("cuda")
    event_start = torch.cuda.Event(enable_timing=True)
    event_ci_end = torch.cuda.Event(enable_timing=True)
    event_mi_end = torch.cuda.Event(enable_timing=True)
    s0 = torch.cuda.Stream()
    s1 = torch.cuda.Stream()

    with torch.cuda.stream(s0):
        x_ci = torch.full((n_ci, 4096, 4096), 0.3).bfloat16().to(device)
        w_ci = torch.full((n_ci, 4096, 4096), 3.0).bfloat16().to(device)
        x_mi = torch.full((n_mi, 4, 4096), 0.3).bfloat16().to(device)
        w_mi = torch.full((n_mi, 4096, 4096), 3.0).bfloat16().to(device)
        for i in range(n_ci):
            nn.functional.linear(x_ci[i], w_ci[i])
        for i in range(n_mi):
            nn.functional.linear(x_mi[i], w_mi[i])

    s0.synchronize()

    if log_stream is not None:
        print("Warm up done", file=log_stream, flush=True)

    return {
        "event_start": event_start,
        "event_ci_end": event_ci_end,
        "event_mi_end": event_mi_end,
        "s0": s0,
        "s1": s1,
        "x_ci": x_ci,
        "w_ci": w_ci,
        "x_mi": x_mi,
        "w_mi": w_mi,
    }


def measure_ci_standalone(ctx):
    event_start = ctx["event_start"]
    event_ci_end = ctx["event_ci_end"]
    s0 = ctx["s0"]
    x_ci = ctx["x_ci"]
    w_ci = ctx["w_ci"]

    event_start.record(s0)
    with torch.cuda.stream(s0):
        for i in range(n_ci):
            nn.functional.linear(x_ci[i], w_ci[i])
    event_ci_end.record(s0)
    event_ci_end.synchronize()
    return n_ci / event_start.elapsed_time(event_ci_end) * 1000


def measure_mi_standalone(ctx):
    event_start = ctx["event_start"]
    event_mi_end = ctx["event_mi_end"]
    s0 = ctx["s0"]
    x_mi = ctx["x_mi"]
    w_mi = ctx["w_mi"]

    event_start.record(s0)
    with torch.cuda.stream(s0):
        for i in range(n_mi):
            nn.functional.linear(x_mi[i], w_mi[i])
    event_mi_end.record(s0)
    event_mi_end.synchronize()
    return n_mi / event_start.elapsed_time(event_mi_end) * 1000


def measure_ci_colocate(ctx):
    event_start = ctx["event_start"]
    event_ci_end = ctx["event_ci_end"]
    s0 = ctx["s0"]
    s1 = ctx["s1"]
    x_ci = ctx["x_ci"]
    w_ci = ctx["w_ci"]
    x_mi = ctx["x_mi"]
    w_mi = ctx["w_mi"]

    with torch.cuda.stream(s1):
        for i in range(100):
            nn.functional.linear(x_mi[i % n_mi], w_mi[i % n_mi])

    event_start.record(s0)
    with torch.cuda.stream(s0):
        for i in range(n_ci):
            nn.functional.linear(x_ci[i], w_ci[i])

    event_ci_end.record(s0)
    event_ci_end.synchronize()
    s1.synchronize()

    return n_ci / event_start.elapsed_time(event_ci_end) * 1000


def measure_mi_colocate(ctx):
    event_start = ctx["event_start"]
    event_mi_end = ctx["event_mi_end"]
    s0 = ctx["s0"]
    s1 = ctx["s1"]
    x_ci = ctx["x_ci"]
    w_ci = ctx["w_ci"]
    x_mi = ctx["x_mi"]
    w_mi = ctx["w_mi"]

    with torch.cuda.stream(s1):
        for i in range(100):
            nn.functional.linear(x_ci[i % n_ci], w_ci[i % n_ci])

    event_start.record(s0)
    with torch.cuda.stream(s0):
        for i in range(n_mi):
            nn.functional.linear(x_mi[i], w_mi[i])

    event_mi_end.record(s0)
    event_mi_end.synchronize()
    s1.synchronize()

    return n_mi / event_start.elapsed_time(event_mi_end) * 1000


def compute_baseline():
    ctx = create_context(log_stream=sys.stdout)
    ci_full = measure_ci_standalone(ctx)
    mi_full = measure_mi_standalone(ctx)
    torch.cuda.synchronize()
    del ctx
    torch.cuda.empty_cache()
    return ci_full, mi_full


def run_measurements_for_div():
    ctx = create_context()
    ci_value = measure_ci_standalone(ctx)
    mi_value = measure_mi_standalone(ctx)
    ci_co_locate = measure_ci_colocate(ctx)
    mi_co_locate = measure_mi_colocate(ctx)
    torch.cuda.synchronize()
    del ctx
    torch.cuda.empty_cache()
    return {
        "ci_standalone": ci_value,
        "mi_standalone": mi_value,
        "ci_co_locate": ci_co_locate,
        "mi_co_locate": mi_co_locate,
    }


def run_child_process(div):
    env = os.environ.copy()
    env["SMSCHED_SPLIT_HINT"] = str(div)
    env["SMSCHED_PROFILE"] = "0"
    smsched_value = os.environ.get("SMSCHED")
    if smsched_value is not None:
        env["SMSCHED"] = smsched_value
        for key, value in parse_smsched_assignments(smsched_value).items():
            env[key] = value

    cmd = [sys.executable, os.path.abspath(__file__), "--div", str(div)]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    output = completed.stdout.strip()
    if not output:
        raise RuntimeError(f"Child process for div {div} returned empty output")
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Child process for div {div} produced no parsable output")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse child output for div {div}: {lines[-1]}\nstdout:\n{output}\nstderr:\n{completed.stderr}"
        ) from exc


def parent_main():
    smsched_value = os.environ.get("SMSCHED")
    if smsched_value is not None:
        for key, value in parse_smsched_assignments(smsched_value).items():
            os.environ[key] = value

    ci_full, mi_full = compute_baseline()

    ci_standalone = [[0, 0]]
    mi_standalone = [[0, 0]]
    ci_co_locate = [[0, 0]]
    mi_co_locate = [[0, 0]]

    for div in range(4, 108, 4):
        child_result = run_child_process(div)
        ci_standalone.append([div, child_result["ci_standalone"]])
        mi_standalone.append([div, child_result["mi_standalone"]])
        ci_co_locate.append([div, child_result["ci_co_locate"]])
        mi_co_locate.append([div, child_result["mi_co_locate"]])

    ci_standalone.append([108, ci_full])
    mi_standalone.append([108, mi_full])
    ci_co_locate.append([108, ci_full])
    mi_co_locate.append([108, mi_full])

    sum_co_locate = []
    for i in range(len(ci_co_locate)):
        for j in range(len(mi_co_locate)):
            if ci_co_locate[i][0] + mi_co_locate[j][0] == 108:
                sum_co_locate.append([
                    ci_co_locate[i][0],
                    ci_co_locate[i][1] / ci_full + mi_co_locate[j][1] / mi_full,
                ])

    print("", flush=True)
    print("CI standalone Throughput: ", ci_full, flush=True)
    print("MI standalone Throughput: ", mi_full, flush=True)
    print("CI standalone: ", ci_standalone, flush=True)
    print("MI standalone: ", mi_standalone, flush=True)
    print("CI co-locate: ", ci_co_locate, flush=True)
    print("MI co-locate: ", mi_co_locate, flush=True)
    print("co-locate sum: ", sum_co_locate, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Profile GEMM with SMSched split hints.")
    parser.add_argument("--div", type=int, default=None, help="Division point for SMSCHED split hint")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.div is not None:
        os.environ["SMSCHED_SPLIT_HINT"] = str(args.div)
        os.environ["SMSCHED_PROFILE"] = "0"
        smsched_value = os.environ.get("SMSCHED")
        if smsched_value is not None:
            for key, value in parse_smsched_assignments(smsched_value).items():
                os.environ[key] = value
        results = run_measurements_for_div()
        print(json.dumps(results), flush=True)
    else:
        parent_main()


if __name__ == "__main__":
    main()


 