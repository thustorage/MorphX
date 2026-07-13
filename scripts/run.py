#!/usr/bin/env python3
"""Run the test-pd-online.py with and without SMSCHED env, creating a structured logs dir.

Usage: run.py <mode: pd-online|pd-offline> <config_basename_without_ext> <workload: baseline|smsched|stream|all> <split_hint> [--no-err]
e.g. run.py pd-online E all 80  -> reads ../configs/pd/E.json

This mirrors the previous run.sh behavior.
"""
import os
import sys
import json
import subprocess
import shlex
from pathlib import Path
from contextlib import nullcontext


def die(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    sys.exit(code)


def open_err_sink(path: Path, discard: bool):
    """Return a context manager yielding the stderr target."""
    return nullcontext(subprocess.DEVNULL) if discard else open(path, 'wb')


def main(argv):
    script = argv[0]
    args = argv[1:]

    discard_stderr = False
    if '--no-err' in args:
        discard_stderr = True
        args = [a for a in args if a != '--no-err']

    if len(args) < 4:
        die(
            f"Usage: {script} <mode: pd-online|pd-offline> <config_basename_without_ext> "
            "<workload: baseline|smsched|stream|all> <split_hint> [--no-err] "
            "(e.g. pd-online E all 80)",
            2,
        )
    if len(args) > 4:
        die(f"Unexpected extra arguments: {' '.join(args[4:])}")

    mode, name, workload, split_hint = args
    if mode not in ('pd-online', 'pd-offline'):
        die(f"Invalid mode: {mode}. Expected 'pd-online' or 'pd-offline'")

    if workload not in ('baseline', 'smsched', 'stream', 'chunked', 'all'):
        die(f"Invalid workload: {workload}. Expected one of baseline, smsched, stream, chunked, all")

    if not split_hint:
        die("split_hint must be a non-empty string")

    run_baseline = workload in ('baseline', 'all')
    run_stream = workload in ('stream', 'all')
    run_smsched = workload in ('smsched', 'all')
    run_chunked = workload in ('chunked', 'all')
    cfg = Path(__file__).resolve().parents[1] / 'configs' / 'pd' / f"{name}.json"

    if not cfg.is_file():
        die(f"Config file not found: {cfg}")

    try:
        with open(cfg, 'r') as f:
            j = json.load(f)
    except Exception as e:
        die(f"Failed to read/parse JSON config {cfg}: {e}")

    rps = j.get('rps')
    input_length = j.get('input_length')
    output_length = j.get('output_length')

    if rps is None or input_length is None or output_length is None:
        die(f"One of rps/input_length/output_length is missing in {cfg} (rps={rps}, input_length={input_length}, output_length={output_length})")

    dir_stub = f"in-{input_length}-out-{output_length}"
    if mode == 'pd-online':
        dir_stub += f"-rps-{rps}"
    dir_stub += f"-split-{split_hint}"
    outdir = Path.cwd() / 'logs' / f"{mode}" / dir_stub
    outdir.mkdir(parents=True, exist_ok=True)

    # Baseline run
    baseline_out = outdir / 'baseline.log'
    baseline_err = outdir / 'baseline.err'

    # Construct command for baseline run based on mode
    test_script = f"test-{mode}.py"
    cmd = [sys.executable, test_script, str(cfg)]
    chunked_cmd = cmd + ["--enable-chunked-prefill"]

    if run_baseline:
        with open(baseline_out, 'wb') as fout, open_err_sink(baseline_err, discard_stderr) as ferr:
            print(f"Running baseline: {' '.join(cmd)}")
            res = subprocess.run(cmd, stdout=fout, stderr=ferr)
            if res.returncode != 0:
                print(f"Baseline exited with code {res.returncode}")
    else:
        print(f"Skipping baseline run (workload=\"{workload}\")")

    # Stream run mirrors baseline but forces prefill threads to 1.
    stream_out = outdir / 'stream.log'
    stream_err = outdir / 'stream.err'

    if run_stream:
        env = os.environ.copy()
        env['VLLM_PREFILL_THREAD'] = '1'
        with open(stream_out, 'wb') as fout, open_err_sink(stream_err, discard_stderr) as ferr:
            print(f"Running stream: {' '.join(cmd)} (VLLM_PREFILL_THREAD=1)")
            res = subprocess.run(cmd, stdout=fout, stderr=ferr, env=env)
            if res.returncode != 0:
                print(f"Stream exited with code {res.returncode}")
    else:
        print(f"Skipping stream run (workload=\"{workload}\")")

    # SMSched run: mirror the original `env $SMSCHED python ...` behaviour.
    # Original used `env $SMSCHED python ...` where $SMSCHED could be something like "VAR=VAL"
    smsched_val = os.environ.get('SMSCHED', '')

    smsched_out = outdir / 'smsched.log'
    smsched_err = outdir / 'smsched.err'

    # If SMSCHED is empty, just run python normally; otherwise use a shell command to allow
    # the string in SMSCHED (e.g. "VAR=VALUE") to be interpreted like the original script.
    shell_parts = ['env']
    smsched_val = smsched_val.strip()
    if smsched_val:
        shell_parts.append(smsched_val)
    shell_parts.extend([
        f"SMSCHED_SPLIT_HINT={split_hint}",
        "VLLM_PREFILL_THREAD=1",
        # "SMSCHED_PROFILE=0",
        shlex.quote(sys.executable),
        shlex.quote(test_script),
        shlex.quote(str(cfg)),
    ])
    shell_cmd = ' '.join(shell_parts)
    if run_smsched:
        with open(smsched_out, 'wb') as fout, open_err_sink(smsched_err, discard_stderr) as ferr:
            print(f"Running smsched (shell): {shell_cmd}")
            res = subprocess.run(shell_cmd, shell=True, stdout=fout, stderr=ferr, executable='/bin/bash')
            if res.returncode != 0:
                print(f"SMSched run exited with code {res.returncode}")
    else:
        print(f"Skipping smsched run (workload=\"{workload}\")")

    chunked_out = outdir / 'chunked.log'
    chunked_err = outdir / 'chunked.err'

    if run_chunked:
        with open(chunked_out, 'wb') as fout, open_err_sink(chunked_err, discard_stderr) as ferr:
            print(f"Running chunked: {' '.join(chunked_cmd)}")
            res = subprocess.run(chunked_cmd, stdout=fout, stderr=ferr)
            if res.returncode != 0:
                print(f"Chunked exited with code {res.returncode}")
    else:
        print(f"Skipping chunked run (workload=\"{workload}\")")


if __name__ == '__main__':
    main(sys.argv)
