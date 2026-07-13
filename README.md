# MorphX SOSP 2026 Artifact Evaluation

This repository contains the artifact evaluation workflow for the MorphX SOSP
2026 submission. It is intended to be cloned on the prepared evaluation machine
and run with the existing `smsched` conda environment.

## Table of Contents

- [Evaluating the Artifact](#evaluating-the-artifact)
- [Artifact Overview](#artifact-overview)
- [Environment Setup](#environment-setup)
- [Dependencies](#dependencies)
- [Output Layout](#output-layout)
- [Notes](#notes)

## Evaluating the Artifact

Activate the prepared environment and run the artifact driver:

```bash
conda activate smsched
./run.sh > run.log 2> run.err
```

The complete run can take more than 20 hours. We recommend running the command
inside `tmux` or another terminal multiplexer. The driver executes every
experiment point sequentially, writes per-run logs under `logs/<timestamp>/`,
then regenerates `tables/` and `figures/`.

`run.sh` prints detailed progress updates. A typical progress line looks like:

```text
[AE][part 1/4][Figures10-12][run 12/280] DONE status=ok run_elapsed=180.4s part_elapsed=36m12s part_eta=13h28m
```

For a shorter run that still exercises all major components, use:

```bash
./run.sh short > run.log 2> run.err
```

The short mode runs Figure 10/12(a), Figure 13(a), Figure 15, and Figure 16.
It produces a subset of the final tables and figures. To generate all tables
and figures immediately from the committed reference logs, use:

```bash
./run.sh analyse-reference > run.log 2> run.err
```

`run.sh` modes and options:

- `./run.sh`: run all four experiment parts from scratch.
- `./run.sh short`: run the shorter subset described above.
- `./run.sh resume`: continue the newest `logs/<timestamp>/` run. Resume is at
  run granularity. If the newest run was started with `short`, resume keeps the
  same short scope.
- `./run.sh analyse`: regenerate `tables/` and `figures/` from the newest
  `logs/<timestamp>/` directory without rerunning experiments.
- `./run.sh analyse-reference`: regenerate `tables/` and `figures/` from the
  committed `logs/reference/` logs.
- `./run.sh smoke`: dry-run one representative point per part to check paths,
  builds, and output formatting.
- `./run.sh --help`: show the command-line help.
- `./run.sh --no-build ...`: skip repo-local builds, mainly for repeated smoke
  checks after binaries have already been built.

## Artifact Overview

This artifact reproduces the evaluation logs, tables, and figures for four
experiment groups:

- **Figures 10-12:** multi-task co-location with LLM plus ANNS/GEMM workloads.
- **Figures 13-14:** large-model PD co-location.
- **Figure 15:** runtime/profiler overhead analysis.
- **Figure 16:** GEMM performance-model accuracy analysis.

Important repository components:

- `run.sh`: top-level artifact driver.
- `scripts/ae/`: orchestration scripts called by `run.sh`.
- `scripts/colocate/`: multi-task co-location workload driver for Figures 10-12.
- `scripts/overhead/`: single-request overhead workload driver for Figure 15.
- `runtime/`: MorphX CUDA runtime interception layer. The AE runner builds it
  with `cuda.cpp` for co-location/overhead/model experiments and `cuda-llm.cpp`
  for the PD co-location experiment.
- `smsched-pass/`: historical directory name for the MorphX LLVM compiler pass.
  The AE runner builds this pass before compiling repo-local CUDA workloads that
  need patched kernels.
- `microbench/`: GEMM microbenchmark and model-accuracy workload for Figure 16.
- `ggnn/`: vendored GGNN source needed by the ANNS workloads.
- `TGS/hijack/`: vendored TGS hijack source used by the TGS baseline.
- `nvbit-tutorial/core/` and `nvbit-tutorial/tools/mem_trace/`: minimal NVBit
  components needed by the Figure 15 NVBit baseline.

## Environment Setup

The artifact assumes the prepared machine already provides:

- A CUDA-capable NVIDIA GPU matching the prepared MorphX evaluation setup.
- CUDA 12.x toolchain and driver libraries.
- The conda environment named `smsched`.
- Local model and dataset caches at the paths used by the workload scripts.

The AE scripts pin all experiment processes to GPU 0 by default.

Use:

```bash
conda activate smsched
```

The PyTorch/vLLM stack in this environment was built with the MorphX compiler
pass, so the required patched kernel variants are already available. Repo-local
CUDA components are rebuilt by `run.sh` before the relevant experiment part.

## Dependencies

The prepared `smsched` environment contains PyTorch, vLLM, FlashInfer, Neutrino,
and Python dependencies needed by the workloads.

- `requirements.txt`: pip-style `package==version` snapshot of the `smsched`
  environment for inspection.
- `docs/environment-packages.txt`: longer environment record with import origins
  for key packages.

The artifact does not expect reviewers to recreate the environment with
`pip install -r requirements.txt`; the prepared environment should be used
directly.

## Output Layout

- `logs/<timestamp>/`: logs for a new full run, split into `part1-colocate`,
  `part2-pd-colocate`, `part3-overhead`, and `part4-model`.
- `logs/reference/`: reference logs from a completed run on the prepared machine.
  This directory is committed so reviewers can immediately run
  `./run.sh analyse-reference`.
- `tables/`: aggregated tables generated from the latest analysed logs. This is
  a generated directory and is not committed to the repository.
- `figures/`: figures generated from `tables/` in PNG and PDF formats. This is
  a generated directory and is not committed to the repository.

## Notes

The paper system name is MorphX. Some internal file names, environment
variables, and legacy directory names still contain historical strings for
compatibility with the existing code, but reviewer-facing logs, tables, figures,
and `run.sh` progress output use MorphX.
