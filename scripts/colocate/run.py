#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import sys
import threading
import time
from contextlib import contextmanager
# 导入 multiprocessing 模块
import multiprocessing
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import torch
import shlex

try:
    import ggnn
except Exception:
    ggnn = None

try:
    # vLLM is optional. If not present, script will exit with helpful message.
    from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
    from vllm.inputs import TokensPrompt
except Exception:
    LLMEngine = None
    EngineArgs = None
    RequestOutput = None
    SamplingParams = None
    TokensPrompt = None

import torchvision
from collections import deque
from torch.amp import autocast

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ["VLLM_USE_V1"] = "0"

# When set to True (via --use-orion), the script will run in multithread
# mode and create CUDA streams with different priorities: LLM gets
# high-priority streams, other workloads get low-priority streams.
ORION_SCHEDULER = False


@contextmanager
def without_child_ld_preload():
    """Prevent vLLM inspection subprocesses from inheriting MorphX LD_PRELOAD.

    The MorphX runtime is already loaded into this process by the dynamic loader.
    Temporarily removing LD_PRELOAD from os.environ only affects subprocesses
    started during vLLM engine initialization.
    """
    old_ld_preload = os.environ.pop("LD_PRELOAD", None)
    try:
        yield
    finally:
        if old_ld_preload is not None:
            os.environ["LD_PRELOAD"] = old_ld_preload


def exit_after_flush_if_morphx_preloaded():
    preload = os.environ.get("LD_PRELOAD", "")
    if "libcuda.so" not in preload or "libpreload.so" not in preload:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

# Copy of model path table from test-pd-online.py (user can edit paths as needed)
model_paths = {
    "llama-2-7b": "/huggingface-cache/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590",
    "llama-3-8b": "/huggingface-cache/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
    "qwen3-4b": "/huggingface-cache/hub/models--Qwen--Qwen3-4B/snapshots/82d62bb073771e7a1ea59435f548908540217d1f",
    "qwen3-8b": "/huggingface-cache/hub/models--Qwen--Qwen3-8B/snapshots/2069b3fae1114555f3c020c81410e51fa0f656f2",
    "gpt2-l": "/huggingface-cache/hub/models--openai-community--gpt2-large/snapshots/32b71b12589c2f8d625668d2335a01cac3249519"
}

dataset_paths = {
    "sharegpt": "/huggingface-cache/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
}


def allocate_cpu_affinities(worker_labels):
    """Return a mapping from worker label to a dedicated CPU set."""
    affinities = {}
    if not worker_labels:
        return affinities

    available_cpus = None
    if hasattr(os, "sched_getaffinity"):
        try:
            available_cpus = sorted(os.sched_getaffinity(0))
        except Exception:
            available_cpus = None

    if not available_cpus:
        cpu_count = os.cpu_count()
        if cpu_count:
            available_cpus = list(range(cpu_count))

    if not available_cpus:
        for label in worker_labels:
            affinities[label] = None
        return affinities

    for idx, label in enumerate(worker_labels):
        cpu_id = available_cpus[idx % len(available_cpus)]
        affinities[label] = {cpu_id}

    return affinities


def apply_cpu_affinity(cpu_ids, worker_name, thread_id):
    if not cpu_ids:
        return
    try:
        os.sched_setaffinity(0, cpu_ids)
        print(
            f"[{worker_name}] Worker {thread_id} (PID {os.getpid()}): bound to CPUs {sorted(cpu_ids)}",
            flush=True,
        )
    except AttributeError:
        print(f"[{worker_name}] Worker {thread_id} (PID {os.getpid()}): CPU affinity not supported", flush=True)
    except Exception as exc:
        print(
            f"[{worker_name}] Worker {thread_id} (PID {os.getpid()}): failed to set CPU affinity ({exc})",
            flush=True,
        )


def _apply_tgs_env_from_string(tgs_string):
    """Parse a shell-like env assignment string (e.g. "LD_PRELOAD=/foo/lib.so VAR=1") and apply
    each KEY=VAL to os.environ in the current process. Uses shlex.split to respect quoting.
    """
    if not tgs_string:
        return
    try:
        parts = shlex.split(tgs_string)
    except Exception:
        parts = tgs_string.split()
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            # Only set if key is non-empty
            if k:
                os.environ[k] = v


def _apply_tgs_env_and_record(tgs_string):
    """Apply assignments from a TGS string into os.environ, returning a dict
    mapping keys to their previous values (or None if not present). This lets
    the caller restore the original environment after spawning a child.

    Example tgs_string: "LD_PRELOAD=/foo/lib.so VAR=1".
    """
    prev = {}
    if not tgs_string:
        return prev
    try:
        parts = shlex.split(tgs_string)
    except Exception:
        parts = tgs_string.split()
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            if not k:
                continue
            prev[k] = os.environ.get(k)
            os.environ[k] = v
    return prev


def _restore_env(prev_map):
    """Restore environment variables recorded in prev_map. If previous value
    was None, the variable is deleted from os.environ.
    """
    for k, old in prev_map.items():
        if old is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = old


def create_cuda_stream(label: str = "", is_llm: bool = False):
    if not ORION_SCHEDULER:
        return torch.cuda.Stream()
    prio = -5 if is_llm else 0
    return torch.cuda.Stream(priority=prio)

def create_prompts(vocab_size, num_requests, input_len, output_len):
    input_lens = torch.full((num_requests,), input_len, dtype=torch.int32, device="cuda")
    output_lens = torch.full((num_requests,), output_len, dtype=torch.int32, device="cuda")
    requests = []
    for i in range(num_requests):
        token_ids = [(i + j) % vocab_size for j in range(int(input_lens[i]))]
        prompt = TokensPrompt(prompt_token_ids=token_ids)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=int(output_lens[i]), min_tokens=int(output_lens[i]))
        requests.append((prompt, sampling_params))
    return requests


def load_sharegpt_samples(dataset_path):
    if not os.path.exists(dataset_path):
        print(f"[LLM] ShareGPT dataset '{dataset_path}' not found")
        return []
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                raw_data = json.load(f)
            else:
                raw_data = [json.loads(line) for line in f if line.strip()]
    except Exception as exc:
        print(f"[LLM] Failed to load ShareGPT dataset '{dataset_path}': {exc}")
        return []

    samples = []
    maximum_entries = 1000
    num_entries = 0
    for entry in raw_data:
        if not isinstance(entry, dict):
            continue
        conversations = entry.get("conversations") or entry.get("conversation")
        if not isinstance(conversations, list):
            continue
        prompt_parts = []
        response_text = None
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            role = turn.get("from") or turn.get("role")
            text = turn.get("value") or turn.get("content")
            if not isinstance(text, str):
                continue
            role_lower = role.lower() if isinstance(role, str) else ""
            if role_lower in ("assistant", "gpt", "bot"):
                response_text = text
                break
            prompt_parts.append(text)
        prompt_text = "\n\n".join(part.strip() for part in prompt_parts if isinstance(part, str) and part.strip())
        if prompt_text:
            samples.append({"prompt": prompt_text, "response": response_text})
            num_entries += 1
            if num_entries >= maximum_entries:
                break
    if not samples:
        print(f"[LLM] No usable entries found in ShareGPT dataset '{dataset_path}'")
    return samples


def build_sharegpt_specs(tokenizer, samples, default_output_len, max_context_length=4096):
    specs = []
    for sample in samples:
        prompt_text = sample.get("prompt")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            continue
        prompt_tokens = None
        try:
            prompt_tokens = tokenizer.encode(prompt_text)
        except Exception:
            try:
                prompt_tokens = tokenizer(prompt_text).input_ids  # type: ignore[attr-defined]
            except Exception as exc:
                print(f"[LLM] Failed to tokenize ShareGPT prompt: {exc}")
                continue
        if isinstance(prompt_tokens, torch.Tensor):
            prompt_tokens = prompt_tokens.tolist()
        if not isinstance(prompt_tokens, (list, tuple)) or not prompt_tokens:
            continue
        response_text = sample.get("response")
        max_tokens = default_output_len
        if isinstance(response_text, str) and response_text.strip():
            try:
                response_tokens = tokenizer.encode(response_text)
                if isinstance(response_tokens, torch.Tensor):
                    response_tokens = response_tokens.tolist()
                if isinstance(response_tokens, (list, tuple)) and len(response_tokens) > 0:
                    max_tokens = len(response_tokens)
            except Exception:
                pass
        try:
            max_tokens = max(int(max_tokens), 1)
        except Exception:
            max_tokens = max(default_output_len, 1)
        if len(prompt_tokens) + max_tokens > max_context_length:
            continue
        specs.append({"prompt_ids": list(prompt_tokens), "max_tokens": max_tokens})
    if not specs:
        print("[LLM] Tokenization produced no valid ShareGPT samples; falling back to synthetic prompts")
    return specs


def initialize_engine(model, input_length, output_length, enable_chunked_prefill: bool):
    if LLMEngine is None:
        raise RuntimeError("vLLM (vllm) is not available in this Python environment")
    context_length = int(input_length) + int(output_length)
    engine_args = EngineArgs(
        model=model_paths.get(model, model),
        enforce_eager=True,
        max_model_len=context_length,
        max_num_batched_tokens=context_length,
        enable_chunked_prefill=enable_chunked_prefill,
        gpu_memory_utilization=0.9,
    )
    with without_child_ld_preload():
        return LLMEngine.from_engine_args(engine_args)


def llm_worker(thread_id, cpu_affinity, config, total_time, result_dict, warmup_lock, ready_barrier, start_barrier):
    """Run Poisson-arrival LLM requests on provided CUDA stream (created internally).

    This function initializes the vLLM engine, then sets engine_ready_event to signal
    initialization completion. It then waits on start_event before actually scheduling
    and processing requests; this allows the main thread to start the GEMM worker so
    both workloads begin at the same time.

    result_dict will be populated with keys: served, avg_first_token_ms, avg_finished_ms
    """
    apply_cpu_affinity(cpu_affinity, "LLM", thread_id)

    # 在进程内部创建 CUDA stream（若启用 ORION，则为 LLM 使用高优先级）
    stream = create_cuda_stream("llm", is_llm=True)
    
    rate = config.get("rps")
    # 为了避免不同进程间的随机数冲突，使用进程/线程 ID 作为种子
    rng = np.random.default_rng(seed=thread_id) 
    intervals = rng.exponential(1 / rate, int(rate * total_time * 2))
    timestamps = np.cumsum(intervals)
    timestamps = timestamps[timestamps < total_time]
    num_requests = len(timestamps)
    input_len = int(config.get("input_length"))
    output_len = int(config.get("output_length"))

    print(f"[LLM] Worker {thread_id} (PID {os.getpid()}): initializing engine (will schedule {num_requests} requests)")

    # initialize engine while holding the stream
    context_length = int(input_len) + int(output_len)
    sharegpt_specs = []

    with warmup_lock:
        with torch.cuda.stream(stream):
            engine = initialize_engine(config.get("model"), input_len, output_len, enable_chunked_prefill=config.get("enable_chunked_prefill", False))
            print(f"[LLM] Worker {thread_id} (PID {os.getpid()}): engine initialized", flush=True)
            dataset_path = dataset_paths.get(config.get("dataset"))
            if dataset_path:
                samples = load_sharegpt_samples(dataset_path)
                if samples:
                    sharegpt_specs = build_sharegpt_specs(engine.get_tokenizer(), samples, output_len, context_length)
                    if sharegpt_specs:
                        max_required = max(len(spec["prompt_ids"]) + spec["max_tokens"] for spec in sharegpt_specs)
                        if max_required > context_length:
                            print(
                                f"[LLM] Warning: ShareGPT sample requires context {max_required}, exceeds configured {context_length}. Generation may fail.",
                                flush=True,
                            )
            # warmup: execute one request
            if sharegpt_specs:
                warmup_spec = sharegpt_specs[0]
                warmup_prompt = TokensPrompt(prompt_token_ids=warmup_spec["prompt_ids"])
                warmup_sampling = SamplingParams(temperature=0.0, max_tokens=warmup_spec["max_tokens"], min_tokens=warmup_spec["max_tokens"])
            else:
                warmup_prompt = TokensPrompt(prompt_token_ids=[0] * input_len)
                warmup_sampling = SamplingParams(temperature=0.0, max_tokens=output_len, min_tokens=output_len)
            engine.add_request("warmup", warmup_prompt, warmup_sampling)
            while engine.has_unfinished_requests():
                request_outputs = engine.step()
                if request_outputs:
                    for ro in request_outputs:
                        if ro.finished:
                            break
        stream.synchronize()

    # Wait for coordinator to allow start of actual workload
    print(f"[LLM] Worker {thread_id} (PID {os.getpid()}): warmup completed, waiting for start", flush=True)
    if ready_barrier is not None:
        ready_barrier.wait( timeout=total_time + 5)

    print(f"[LLM] Worker {thread_id} (PID {os.getpid()}): started")
    start_time = time.time()
    if start_barrier is not None:
        start_barrier.wait(timeout=total_time + 5)
    # Create prompts and run the scheduling loop on the stream
    with torch.cuda.stream(stream):
        if sharegpt_specs:
            prompts = []
            for idx in range(num_requests):
                spec = sharegpt_specs[idx % len(sharegpt_specs)]
                prompt = TokensPrompt(prompt_token_ids=spec["prompt_ids"])
                sampling_params = SamplingParams(temperature=0.0, max_tokens=spec["max_tokens"], min_tokens=spec["max_tokens"])
                prompts.append((prompt, sampling_params))
        else:
            prompts = create_prompts(engine.get_tokenizer().vocab_size, num_requests, input_len, output_len)
        i = 0
        served = 0
        avg_first_token = 0.0
        avg_finished = 0.0

        while i < num_requests or engine.has_unfinished_requests():
            now = time.time() - start_time
            if i < num_requests and now >= timestamps[i]:
                prompt, sampling_params = prompts[i]
                engine.add_request(str(i), prompt, sampling_params)
                i += 1
            if time.time() - start_time > total_time:
                break
            if engine.has_unfinished_requests():
                # print(f"[LLM] Worker {thread_id} (PID {os.getpid()}): stepping engine at time {now:.2f}s", flush=True)
                request_outputs = engine.step()
                stream.synchronize()
                if request_outputs is None:
                    continue
                elif len(request_outputs) == 0:
                    request_outputs = engine.get_prefill_result(isBlock=True)

                for request_output in request_outputs:
                    if request_output.finished:
                        arrival_time = request_output.metrics.arrival_time
                        first_sched_time = request_output.metrics.first_scheduled_time
                        first_token_time = request_output.metrics.first_token_time
                        finished_time = request_output.metrics.finished_time
                        num_generated = len(request_output.outputs[0].token_ids)
                        if first_token_time is not None:
                            ttft = (first_token_time - arrival_time) * 1000.0
                        else:
                            ttft = None
                        latency = (finished_time - arrival_time) * 1000.0
                        print(f"[LLM] Worker {thread_id} (PID {os.getpid()}): Request {request_output.request_id} served.")
                        print(f"    Arrival time: {arrival_time:.4f} s")
                        print(f"    Time in queue: {first_sched_time - arrival_time:.4f} s")
                        print(f"    Time to first token: {ttft:.4f} ms" if ttft is not None else "    Time to first token: N/A")
                        print(f"    Request latency: {latency:.4f} ms")
                        print(f"    Generated tokens: {num_generated}")
                        print(f"    Time per token: {((latency - ttft) / (num_generated - 1)) if num_generated > 1 and first_token_time is not None else 0:.4f} ms/token")
                        print(f"    Current batch size: {len(request_outputs)}")
                        if time.time() - start_time <= total_time:
                            served += 1
                            if ttft is not None:
                                avg_first_token += ttft
                            avg_finished += latency

        # final stats
        if served > 0:
            result_dict["served"] = served
            result_dict["avg_first_token_ms"] = (avg_first_token / served) if avg_first_token > 0 else None
            result_dict["avg_finished_ms"] = (avg_finished / served)
        else:
            result_dict["served"] = 0
            result_dict["avg_first_token_ms"] = None
            result_dict["avg_finished_ms"] = None

        engine.wait_prefill_shutdown()


def gemm_worker(thread_id, cpu_affinity, config, total_time, result_dict, warmup_lock, ready_barrier, start_barrier):
    """Continuously execute matrix multiplies (A @ B) on given stream until total_time elapses.

    We compute synchronous matmuls to measure completion count reliably. Results populated in result_dict:
      - ops_count (number of matmuls executed)
      - time_s (wall time elapsed during measured loop)
      - gflops (computed as 2*N^3 * ops / time / 1e9)
    """
    apply_cpu_affinity(cpu_affinity, "GEMM", thread_id)

    # 在进程内部创建 CUDA stream（若启用 ORION，则为 GEMM 使用低优先级）
    stream = create_cuda_stream("mm", is_llm=False)
    
    device = torch.device("cuda")

    # read parameters from config dict for consistency
    try:
        size = int(config.get("mm_size", config.get("size", 8192)))
    except Exception:
        size = 8192
    dtype_name = config.get("mm_dtype", config.get("dtype", "float16"))
    dtype_t = getattr(torch, dtype_name)
    try:
        rps = float(config.get("mm_rps", config.get("rps", 0.0)))
    except Exception:
        rps = 0.0

    # Perform warmup before waiting
    with warmup_lock:
        with torch.cuda.stream(stream):
            # allocate matrices once on GPU
            A = torch.randn((size, size), dtype=dtype_t, device=device)
            B = torch.randn((size, size), dtype=dtype_t, device=device)
            # warm up
            C = torch.matmul(A, B)
        torch.cuda.synchronize()

    print(f"[GEMM] Worker {thread_id} (PID {os.getpid()}): warmup completed, waiting for start")
    if ready_barrier is not None:
        ready_barrier.wait(timeout=total_time + 5)
    print(f"[GEMM] Worker {thread_id} (PID {os.getpid()}): started")
    start = time.time()
    if start_barrier is not None:
        start_barrier.wait(timeout=total_time + 5)

    if rps > 0:
        rng = np.random.default_rng(seed=thread_id)
        intervals = rng.exponential(1 / rps, int(rps * total_time * 2))
        timestamps = np.cumsum(intervals)
        timestamps = timestamps[timestamps < total_time]
        num_requests = len(timestamps)
        print(f"num_requests: {num_requests}")
        req_idx = 0
    else: # rps <= 0 means run continuously
        num_requests = -1
        req_idx = 0
        timestamps = []

    with torch.cuda.stream(stream):
        ops = 0
        # run synchronous matmuls and count completions
        while True:
            now = time.time() - start
            if now >= total_time:
                break

            if rps > 0: # Poisson arrival
                if req_idx >= num_requests or now < timestamps[req_idx]:
                    time.sleep(0.001)
                    continue
                req_idx += 1
            # if rps <= 0, runs continuously

            for i in range(10):
                A = torch.randn((size, size), dtype=dtype_t, device=device)
                B = torch.randn((size, size), dtype=dtype_t, device=device)
                C = torch.matmul(A, B)
            # ensure completion to count definitively
            torch.cuda.synchronize()
            ops += 10
            
        elapsed = time.time() - start
        total_time_s = elapsed
        # number of floating point ops per matmul = 2*N^3
        flops_per = 2 * (size ** 3)
        gflops = (ops * flops_per) / total_time_s / 1e9 if total_time_s > 0 else 0

        result_dict["ops_count"] = ops
        result_dict["time_s"] = total_time_s
        result_dict["gflops"] = gflops


def ggnn_worker(thread_id, cpu_affinity, config, rps, total_time, result_dict, warmup_lock, ready_barrier, start_barrier):
    """Continuously execute GGNN queries of fixed size until total_time elapses."""
    if ggnn is None:
        print(f"[GGNN] Worker {thread_id} (PID {os.getpid()}): ggnn module not available; skipping worker")
        result_dict["completed_ggnn_queries"] = 0
        result_dict["avg_latency_ms"] = None
        return

    apply_cpu_affinity(cpu_affinity, "GGNN", thread_id)

    # 在进程内部创建 CUDA stream（若启用 ORION，则为 GGNN 使用低优先级）
    stream = create_cuda_stream("ggnn", is_llm=False)

    base_size = int(config.get("base_size", 100000))
    query_size = int(config.get("query_size", base_size))
    feature_dim = int(config.get("feature_dim", 128))
    k_build = int(config.get("k_build", 24))
    tau_build = float(config.get("tau_build", 0.5))
    k_query = int(config.get("k_query", 10))
    tau_query = float(config.get("tau_query", 0.64))
    max_iterations = int(config.get("max_iterations", 400))
    dtype_name = str(config.get("dtype", "float32"))
    randomize_query = config.get("randomize_query", True)
    if isinstance(randomize_query, str):
        randomize_query = randomize_query.lower() not in ("false", "0", "no")
    randomize_query = bool(randomize_query)

    dtype_t = getattr(torch, dtype_name, None)
    if dtype_t is None:
        print(f"[GGNN] Worker {thread_id} (PID {os.getpid()}): Unknown dtype '{dtype_name}', falling back to float32")
        dtype_t = torch.float32

    distance_measure = None
    if hasattr(ggnn, "DistanceMeasure"):
        target_name = str(config.get("measure", "euclidean")).upper()
        for attr in dir(ggnn.DistanceMeasure):
            if attr.upper() == target_name:
                distance_measure = getattr(ggnn.DistanceMeasure, attr)
                break
    if distance_measure is None:
        distance_measure = ggnn.DistanceMeasure.Euclidean

    device = torch.device("cuda")

    warmup_ms = None
    with warmup_lock:
        with torch.cuda.stream(stream):
            base = torch.rand((base_size, feature_dim), dtype=dtype_t, device=device)
            query = torch.rand((query_size, feature_dim), dtype=dtype_t, device=device)

            index = ggnn.GGNN()
            index.set_base(base)
            index.set_return_results_on_gpu(True)
            index.build(k_build=k_build, tau_build=tau_build, measure=distance_measure)
            torch.cuda.synchronize()

            warmup_start = time.time()
            index.query(query, k_query, tau_query, max_iterations, distance_measure)
            torch.cuda.synchronize()
        warmup_ms = (time.time() - warmup_start) * 1000.0

    if warmup_ms is not None:
        print(f"[GGNN] Worker {thread_id} (PID {os.getpid()}): Warmup query completed in {warmup_ms:.2f} ms")

    print(f"[GGNN] Worker {thread_id} (PID {os.getpid()}): warmup completed, waiting for start")
    if ready_barrier is not None:
        ready_barrier.wait(timeout=total_time + 5)
    print(f"[GGNN] Worker {thread_id} (PID {os.getpid()}): started")
    run_start = time.time()
    if start_barrier is not None:
        start_barrier.wait(timeout=total_time + 5)

    if rps > 0:
        rng = np.random.default_rng(seed=thread_id)
        intervals = rng.exponential(1 / rps, int(rps * total_time * 2))
        timestamps = np.cumsum(intervals)
        timestamps = timestamps[timestamps < total_time]
        num_requests = len(timestamps)
        req_idx = 0
    else: # rps <= 0 means run continuously
        num_requests = -1
        req_idx = 0
        timestamps = []

    try:
        latencies = []
        completed = 0
        while True:
            now = time.time() - run_start
            if now >= total_time:
                break

            if rps > 0: # Poisson arrival
                if req_idx >= num_requests or now < timestamps[req_idx]:
                    time.sleep(0.001)
                    continue
                req_idx += 1
            # if rps <= 0, runs continuously

            op_start = time.time()
            with torch.cuda.stream(stream):
                if randomize_query:
                    query.uniform_(0.0, 1.0)
                # print(f"[GGNN] Worker {thread_id} (PID {os.getpid()}): Starting GGNN query time: {time.time() - run_start:.2f} s", flush=True)
                indices, dists = index.query(query, k_query, tau_query, max_iterations, distance_measure)
                # print('indices:', indices[:5], '\n squared dists:',  dists[:5], '\n')
            op_end = time.time()
            latency_ms = (op_end - op_start) * 1000.0
            # print(f"[GGNN] Worker {thread_id} (PID {os.getpid()}): Completed GGNN query in {latency_ms:.2f} ms", flush=True)
            elapsed = op_end - run_start
            if elapsed <= total_time:
                latencies.append(latency_ms)
                completed += 1

        if completed > 0:
            result_dict["completed_ggnn_queries"] = completed
            result_dict["avg_latency_ms"] = sum(latencies) / len(latencies)
        else:
            result_dict["completed_ggnn_queries"] = 0
            result_dict["avg_latency_ms"] = None
    except Exception as exc:
        print(f"[GGNN] Worker {thread_id} (PID {os.getpid()}): Execution failed: {exc}")
        result_dict["completed_ggnn_queries"] = 0
        result_dict["avg_latency_ms"] = None

def dnn_worker(thread_id, cpu_affinity, config, total_time, result_dict, warmup_lock, ready_barrier, start_barrier):
    max_batch_size = config.get("max_batch_size", 256)
    rps = config.get("rps", 100)
    batch_window = config.get("batch_window", 0.01)
    if not torch.cuda.is_bf16_supported():
        print("Warning: Your GPU does not support BF16 natively.")

    apply_cpu_affinity(cpu_affinity, "DNN", thread_id)

    stream = create_cuda_stream("dnn", is_llm=False)

    with warmup_lock:
        with torch.cuda.stream(stream):
            print("Loading model...", flush=True)
            model = torchvision.models.resnet50().cuda()
            model = model.to(memory_format=torch.channels_last)
            model.eval()

            print("Warming up...", flush=True)
            dummy_input = torch.randn(max_batch_size, 3, 256, 256).cuda().to(memory_format=torch.channels_last)
            for _ in range(3):
                with torch.no_grad():
                    with autocast(device_type='cuda', dtype=torch.bfloat16):
                        _ = model(dummy_input)
            torch.cuda.synchronize()

            num_requests_estimated = int(rps * total_time * 1.2) + 100
            inter_arrival_times = np.random.exponential(1.0 / rps, num_requests_estimated)
            arrival_times = np.cumsum(inter_arrival_times)
            arrival_times = arrival_times[arrival_times <= total_time]
            print(f"Total requests generated: {len(arrival_times)}", flush=True)

            max_input_data = torch.randn(max_batch_size, 3, 256, 256).cuda()
            max_input_data = max_input_data.to(memory_format=torch.channels_last)

    print(f"[DNN] Worker {thread_id} (PID {os.getpid()}): warmup completed, waiting for start")
    if ready_barrier is not None:
        ready_barrier.wait(timeout=total_time + 5)
    print(f"[DNN] Worker {thread_id} (PID {os.getpid()}): started")

    request_queue = deque()
    next_request_idx = 0
    completed_requests = 0
    inference_count = 0

    start_time = time.time()
    if start_barrier is not None:
        start_barrier.wait(timeout=total_time + 5)

    while True:
        current_time = time.time() - start_time
        
        while next_request_idx < len(arrival_times) and arrival_times[next_request_idx] <= current_time:
            request_queue.append(arrival_times[next_request_idx])
            next_request_idx += 1
        
        if current_time > total_time and not request_queue:
            break
            
        should_infer = False
        if request_queue:
            if len(request_queue) >= max_batch_size:
                should_infer = True
            elif (current_time - request_queue[0]) >= batch_window:
                should_infer = True

        if should_infer:
            batch_size = min(len(request_queue), max_batch_size)
            
            for _ in range(batch_size):
                request_queue.popleft()
            batch_input = max_input_data[:batch_size]
            
            inf_start = time.time()
            with torch.no_grad():
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    output = model(batch_input)
            torch.cuda.synchronize()
            inf_end = time.time()
            
            inf_time = inf_end - inf_start
            completed_requests += batch_size
            inference_count += 1
            
            print(f"[DNN] Inference {inference_count}: Batch Size: {batch_size}, Time: {inf_time:.4f}s, Pending: {len(request_queue)}", flush=True)
        else:
            # 避免空转占用 CPU
            if rps < 1000:
                time.sleep(0.0001) 

    result_dict["completed_requests"] = completed_requests
    result_dict["inference_count"] = inference_count


def llm_worker_wrapper(mps_percentage, thread_id, cpu_affinity, use_tgs, *args):
    # This wrapper runs inside the child process (when using multiprocessing).
    # Optionally set CUDA MPS percentage and TGS env for the LLM worker process.
    if mps_percentage is not None:
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps_percentage)
    if use_tgs:
        # Apply the parent-provided TGS_HIGH assignment string (if present) so that
        # environment variables like LD_PRELOAD=... are set in this child process.
        tgs_high = os.environ.get("TGS_HIGH")
        if tgs_high:
            _apply_tgs_env_from_string(tgs_high)
            # optional debug print
            print(f"[LLM wrapper] Applied TGS_HIGH assignments: {tgs_high}", flush=True)
    llm_worker(thread_id, cpu_affinity, *args)

def other_worker_wrapper(worker_func, mps_percentage, thread_id, cpu_affinity, use_tgs, *args):
    # This wrapper runs inside the child process (when using multiprocessing).
    # Optionally set CUDA MPS percentage and TGS env for background worker processes.
    if mps_percentage is not None:
        other_percentage = 100 - mps_percentage
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(other_percentage)
    if use_tgs:
        # Apply the parent-provided TGS_LOW assignment string (if present) so that
        # environment variables like LD_PRELOAD=... are set in this child process.
        tgs_low = os.environ.get("TGS_LOW")
        if tgs_low:
            _apply_tgs_env_from_string(tgs_low)
            print(f"[worker wrapper] Applied TGS_LOW assignments: {tgs_low}", flush=True)
    worker_func(thread_id, cpu_affinity, *args)

def parse_args():
    parser = argparse.ArgumentParser(description="Run colocated LLM + big-matrix workload")
    # workload parameters (override config file when provided)
    parser.add_argument("--model", help="Model name or path to use for LLM")
    parser.add_argument("--input-length", type=int, help="LLM input length (tokens)")
    parser.add_argument("--output-length", type=int, help="LLM output length (tokens)")
    parser.add_argument("--rps", type=float, help="LLM requests per second (per-thread)")
    parser.add_argument("--time", type=float, help="Total runtime in seconds for the workload")
    parser.add_argument("--mm-size", type=int, help="Square matrix size N for GEMM (NxN)")
    parser.add_argument("--mm-dtype", default=None, help="Matrix dtype for GEMM (e.g. float16, float32)")
    parser.add_argument("--mm-rps", type=float, help="GEMM requests per second (0 for continuous)")
    # enable/disable workers
    parser.add_argument("--enable-llm", action="store_true", help="Launch the LLM worker thread")
    parser.add_argument("--enable-mm", action="store_true", help="Launch the GEMM worker thread")
    parser.add_argument("--enable-ggnn", action="store_true", help="Launch the GGNN worker thread")
    parser.add_argument("--enable-dnn", action="store_true", help="Launch the DNN worker thread")
    parser.add_argument("--enable-chunked-prefill", action="store_true", help="Enable vLLM chunked prefill")
    parser.add_argument("--ggnn-base-size", type=int, help="Number of vectors in GGNN base tensors (default 1,000,000)")
    parser.add_argument("--ggnn-query-size", type=int, help="Number of vectors in GGNN query tensors (default same as base size)")
    parser.add_argument("--ggnn-dim", type=int, help="Feature dimension for GGNN tensors (default 128)")
    parser.add_argument("--ggnn-rps", type=float, help="GGNN requests per second (0 for continuous)")
    parser.add_argument("--ggnn-k-build", type=int, help="k parameter for GGNN graph build (default 24)")
    parser.add_argument("--ggnn-tau-build", type=float, help="tau parameter for GGNN graph build (default 0.5)")
    parser.add_argument("--ggnn-k-query", type=int, help="k parameter for GGNN query (default 10)")
    parser.add_argument("--ggnn-tau-query", type=float, help="tau parameter for GGNN query (default 0.64)")
    parser.add_argument("--ggnn-max-iter", type=int, help="Maximum iterations for GGNN query (default 400)")
    parser.add_argument("--ggnn-measure", help="Distance measure for GGNN (default euclidean)")
    parser.add_argument("--ggnn-dtype", help="Tensor dtype for GGNN data (default float32)")
    parser.add_argument("--ggnn-no-randomize", action="store_true", help="Reuse warmup query instead of randomizing each GGNN iteration")
    parser.add_argument("--dataset", help="type of dataset to use (e.g., 'sharegpt') for LLM worker")
    parser.add_argument("--dnn-max-batch-size", type=int, default=256, help="Maximum batch size for DNN worker (default 256)")
    parser.add_argument("--dnn-rps", type=float, default=100.0, help="DNN worker requests per second (default 100.0)")
    parser.add_argument("--dnn-batch-window", type=float, default=0.01, help="Batching window in seconds for DNN worker (default 0.01)")
    # 新增参数，用于切换多进程模式
    parser.add_argument("--use-multiprocess", action="store_true", help="Execute each task in a separate process instead of a thread")
    parser.add_argument("--use-mps", action="store_true", help="Enable CUDA MPS with percentage-based resource allocation for workers.")
    parser.add_argument("--mps-percentage", type=int, help="Percentage of MPS resources for the LLM worker (0-100). Required if --use-mps is set.")
    parser.add_argument("--use-tgs", action="store_true", help="When set (multi-process only), set TGS_HIGH for the LLM worker process and TGS_LOW for background worker processes.")
    parser.add_argument("--use-orion", action="store_true", help="Use ORION scheduler: force multithread mode and set LLM stream to high priority and other streams to low priority.")
    return parser.parse_args()


def main():
    args = parse_args()

    # ORION scheduler forces multithread mode and special stream priorities
    global ORION_SCHEDULER
    if getattr(args, "use_orion", False):
        ORION_SCHEDULER = True
        print("Using ORION scheduler: forcing multithread mode and setting CUDA stream priorities (LLM high, others low)")

    if args.use_mps:
        if ORION_SCHEDULER:
            print("Warning: --use-mps ignored because --use-orion forces multithread mode.")
        else:
            if not args.use_multiprocess:
                print("Error: --use-mps requires --use-multiprocess to be enabled.", file=sys.stderr)
                sys.exit(1)
            if args.mps_percentage is None:
                print("Error: --mps-percentage must be provided when using --use-mps.", file=sys.stderr)
                sys.exit(1)
            if not (0 <= args.mps_percentage <= 100):
                print(f"Error: --mps-percentage must be between 0 and 100, but got {args.mps_percentage}.", file=sys.stderr)
                sys.exit(1)

    # 确定是使用线程还是进程
    # If ORION is enabled we force multithread mode, otherwise use CLI
    if ORION_SCHEDULER:
        use_multiprocess = False
    else:
        use_multiprocess = args.use_multiprocess
    if use_multiprocess:
        print("Running in multi-process mode.")
        ProcessClass = multiprocessing.Process
        LockClass = multiprocessing.Lock
        BarrierClass = multiprocessing.Barrier
        # 使用 Manager 来创建可跨进程共享的字典
        manager = multiprocessing.Manager()
        DictClass = manager.dict
    else:
        print("Running in multi-thread mode.")
        ProcessClass = threading.Thread
        LockClass = threading.Lock
        BarrierClass = threading.Barrier
        DictClass = dict # 线程可以直接使用普通字典

    # If requested, ensure the parent process exposes TGS assignment strings so
    # child processes inherit them. Do NOT read these values from CLI; the
    # user manages them in their shell (e.g. ~/.bashrc) and they should already
    # be present in os.environ when this script starts.
    if args.use_tgs:
        if os.environ.get("TGS_HIGH"):
            print(f"Using TGS_HIGH from parent environment: {os.environ.get('TGS_HIGH')}")
        else:
            print("Warning: --use-tgs set but TGS_HIGH not found in parent environment; LLM worker will see no TGS_HIGH assignment.")

        if os.environ.get("TGS_LOW"):
            print(f"Using TGS_LOW from parent environment: {os.environ.get('TGS_LOW')}")
        else:
            print("Warning: --use-tgs set but TGS_LOW not found in parent environment; background workers will see no TGS_LOW assignment.")

    # Merge CLI args (when provided) over default values
    total_time = float(args.time if args.time is not None else 30.0)
    mm_size = int(args.mm_size if args.mm_size is not None else 8192)
    mm_dtype = args.mm_dtype if args.mm_dtype is not None else "float16"
    mm_rps = float(args.mm_rps if args.mm_rps is not None else 0.0)
    dnn_max_batch_size = int(args.dnn_max_batch_size if args.dnn_max_batch_size is not None else 256)
    dnn_rps = float(args.dnn_rps if args.dnn_rps is not None else 100.0)
    dnn_batch_window = float(args.dnn_batch_window if args.dnn_batch_window is not None else 0.01)

    model = args.model
    input_length = int(args.input_length if args.input_length is not None else 8)
    output_length = int(args.output_length if args.output_length is not None else 8)
    rps = float(args.rps if args.rps is not None else 1.0)
    dataset = args.dataset

    # which workers to run
    enable_llm = bool(args.enable_llm)
    enable_mm = bool(args.enable_mm)
    enable_ggnn = bool(args.enable_ggnn)
    enable_dnn = bool(args.enable_dnn)

    # pass-through for vLLM option
    enable_chunked = bool(args.enable_chunked_prefill)

    # prepare a config dict for llm_worker (keeps same keys as earlier JSON based flow)
    llm_cfg = {
        "model": model,
        "input_length": input_length,
        "output_length": output_length,
        "rps": rps,
        "enable_chunked_prefill": enable_chunked,
        "dataset": dataset,
    }

    # Build ggnn_cfg from CLI args (consistent with other workload cfgs)
    ggnn_cfg = {
        "base_size": int(args.ggnn_base_size) if args.ggnn_base_size is not None else 100000,
        "query_size": int(args.ggnn_query_size) if args.ggnn_query_size is not None else None,
        "feature_dim": int(args.ggnn_dim) if args.ggnn_dim is not None else 128,
        "k_build": int(args.ggnn_k_build) if args.ggnn_k_build is not None else 24,
        "tau_build": float(args.ggnn_tau_build) if args.ggnn_tau_build is not None else 0.5,
        "k_query": int(args.ggnn_k_query) if args.ggnn_k_query is not None else 10,
        "tau_query": float(args.ggnn_tau_query) if args.ggnn_tau_query is not None else 0.64,
        "max_iterations": int(args.ggnn_max_iter) if args.ggnn_max_iter is not None else 400,
        "measure": args.ggnn_measure if args.ggnn_measure is not None else "euclidean",
        "dtype": args.ggnn_dtype if args.ggnn_dtype is not None else "float32",
        "randomize_query": False if args.ggnn_no_randomize else True,
    }
    # If query_size not provided, default to base_size
    if ggnn_cfg["query_size"] is None:
        ggnn_cfg["query_size"] = ggnn_cfg["base_size"]

    ggnn_rps = float(args.ggnn_rps if args.ggnn_rps is not None else 0.0)
    ggnn_cfg["rps"] = ggnn_rps

    # 使用 DictClass (共享字典) 替换普通字典
    llm_stats = DictClass()
    mm_stats = DictClass()
    ggnn_stats = DictClass()
    dnn_stats = DictClass()

    # coordination events
    # 使用 LockClass (线程或进程锁)
    warmup_lock = LockClass()  # serialize warmup sections across workers

    t_llm = None
    t_mm = None
    t_ggnn = None
    t_dnn = None

    if not (enable_llm or enable_mm or enable_ggnn or enable_dnn):
        print("No workers enabled. Use --enable-llm, --enable-mm, --enable-ggnn, or --enable-dnn. Exiting.")
        return

    active_workers = sum(1 for flag in (enable_llm, enable_mm, enable_ggnn, enable_dnn) if flag)
    # 使用 BarrierClass (线程或进程屏障)
    ready_barrier = BarrierClass(active_workers) if active_workers > 1 else None
    start_barrier = BarrierClass(active_workers) if active_workers > 1 else None

    worker_sequence = []
    if enable_llm:
        worker_sequence.append("llm")
    if enable_mm:
        worker_sequence.append("mm")
    if enable_ggnn:
        worker_sequence.append("ggnn")
    if enable_dnn:
        worker_sequence.append("dnn")

    cpu_affinities = allocate_cpu_affinities(worker_sequence)
    if worker_sequence:
        print("CPU affinity assignments:")
        for label in worker_sequence:
            assigned = cpu_affinities.get(label)
            label_display = label.upper()
            if assigned:
                print(f"  {label_display}: {sorted(assigned)}")
            else:
                print(f"  {label_display}: not set (unsupported on this platform)")

    # If TGS is enabled and parent has a TGS_HIGH assignment string,
    # temporarily apply those assignments (e.g. LD_PRELOAD=...) to the
    # parent environment so the dynamic loader sees them when the child
    # process is spawned. Then restore the parent's environment.
    tgs_high_prev_env = None
    if args.use_tgs and os.environ.get("TGS_HIGH"):
        tgs_high_prev_env = _apply_tgs_env_and_record(os.environ.get("TGS_HIGH"))
        print(f"Applied TGS_HIGH assignments to parent environment for LLM spawn: {os.environ.get('TGS_HIGH')}")


    # Launch LLM worker
    if enable_llm:
        print(f"Starting engine initialization on LLM worker")
        # Use wrapper when either MPS or TGS env should be set in the child process
        if args.use_mps or use_multiprocess and args.use_tgs:
            target = llm_worker_wrapper
            # mps_percentage may be None when not using MPS; wrapper handles that
            worker_args = (
                args.mps_percentage if args.use_mps else None,
                0,
                cpu_affinities.get("llm"),
                args.use_tgs,
                llm_cfg,
                total_time,
                llm_stats,
                warmup_lock,
                ready_barrier,
                start_barrier,
            )
        else:
            target = llm_worker
            worker_args = (
                0,
                cpu_affinities.get("llm"),
                llm_cfg,
                total_time,
                llm_stats,
                warmup_lock,
                ready_barrier,
                start_barrier,
            )

        t_llm = ProcessClass(
            target=target,
            args=worker_args,
            daemon=False,
        )
        t_llm.start()

    if tgs_high_prev_env is not None:
        _restore_env(tgs_high_prev_env)
        print("Restored parent environment after spawning LLM worker")

    # prepare per-worker config dicts for consistent passing
    mm_cfg = {"mm_size": mm_size, "mm_dtype": mm_dtype, "mm_rps": mm_rps}
    dnn_cfg = {"max_batch_size": dnn_max_batch_size, "rps": dnn_rps, "batch_window": dnn_batch_window}

    # Temporarily apply TGS_LOW before spawning background worker so LD_PRELOAD
    # and other assignments are present for the new process. Restore after.
    tgs_low_prev_env = None
    if args.use_tgs and os.environ.get("TGS_LOW"):
        tgs_low_prev_env = _apply_tgs_env_and_record(os.environ.get("TGS_LOW"))
        print(f"Applied TGS_LOW assignments to parent environment for MM spawn: {os.environ.get('TGS_LOW')}")


    # Launch MM worker
    if enable_mm:
        # Use wrapper when either MPS or TGS env should be set in the child process
        if args.use_mps or use_multiprocess and args.use_tgs:
            target = other_worker_wrapper
            worker_args = (
                gemm_worker,
                args.mps_percentage if args.use_mps else None,
                1,
                cpu_affinities.get("mm"),
                args.use_tgs,
                mm_cfg,
                total_time,
                mm_stats,
                warmup_lock,
                ready_barrier,
                start_barrier,
            )
        else:
            target = gemm_worker
            worker_args = (
                1,
                cpu_affinities.get("mm"),
                mm_cfg,
                total_time,
                mm_stats,
                warmup_lock,
                ready_barrier,
                start_barrier,
            )

        t_mm = ProcessClass(
            target=target,
            args=worker_args,
            daemon=False,
        )
        t_mm.start()

    if enable_ggnn:
        print(
            "Starting GGNN worker (base_size={}, query_size={}, dim={}, k_query={})".format(
                ggnn_cfg.get("base_size"),
                ggnn_cfg.get("query_size"),
                ggnn_cfg.get("feature_dim"),
                ggnn_cfg.get("k_query"),
            )
        )
        # Use wrapper when either MPS or TGS env should be set in the child process
        if args.use_mps or use_multiprocess and args.use_tgs:
            target = other_worker_wrapper
            worker_args = (
                ggnn_worker,
                args.mps_percentage if args.use_mps else None,
                3,
                cpu_affinities.get("ggnn"),
                args.use_tgs,
                ggnn_cfg,
                ggnn_rps,
                total_time,
                ggnn_stats,
                warmup_lock,
                ready_barrier,
                start_barrier,
            )
        else:
            target = ggnn_worker
            worker_args = (
                3,
                cpu_affinities.get("ggnn"),
                ggnn_cfg,
                ggnn_rps,
                total_time,
                ggnn_stats,
                warmup_lock,
                ready_barrier,
                start_barrier,
            )

        t_ggnn = ProcessClass(
            target=target,
            args=worker_args,
            daemon=False,
        )
        t_ggnn.start()

    if enable_dnn:
        print("Starting DNN worker (max_batch_size={}, rps={})".format(
            dnn_cfg.get("max_batch_size"),
            dnn_cfg.get("rps"),
        ))
        # Use wrapper when either MPS or TGS env should be set in the child process
        if args.use_mps or use_multiprocess and args.use_tgs:
            target = other_worker_wrapper
            worker_args = (
                dnn_worker,
                args.mps_percentage if args.use_mps else None,
                4,
                cpu_affinities.get("dnn"),
                args.use_tgs,
                dnn_cfg,
                total_time,
                dnn_stats,
                warmup_lock,
                ready_barrier,
                start_barrier,
            )
        else:
            target = dnn_worker
            worker_args = (
                4,
                cpu_affinities.get("dnn"),
                dnn_cfg,
                total_time,
                dnn_stats,
                warmup_lock,
                ready_barrier,
                start_barrier,
            )
        t_dnn = ProcessClass(
            target=target,
            args=worker_args,
            daemon=False,
        )
        t_dnn.start()

    if tgs_low_prev_env is not None:
        _restore_env(tgs_low_prev_env)
        print("Restored parent environment after spawning GGNN worker")

    # join any workers we started
    if t_llm is not None:
        t_llm.join()
    if t_mm is not None:
        t_mm.join()
    if t_ggnn is not None:
        t_ggnn.join()
    if t_dnn is not None:
        t_dnn.join()

    print("\n=== Results ===")
    print("LLM stats:")
    # 将共享字典 (DictProxy) 转换为普通字典以便于打印
    llm_stats_print = dict(llm_stats)
    if llm_stats_print.get("served", 0) > 0:
        print(f"  Served requests: {llm_stats_print['served']}")
        print(f"  Avg first token (ms): {llm_stats_print['avg_first_token_ms']}")
        print(f"  Avg finished latency (ms): {llm_stats_print['avg_finished_ms']}")
    else:
        print("  No LLM requests served or vLLM not available.")

    print("MM stats:")
    mm_stats_print = dict(mm_stats)
    if mm_stats_print:
        print(f"  Completed matmuls: {mm_stats_print.get('ops_count')}")
        print(f"  Elapsed time (s): {mm_stats_print.get('time_s'):.6f}")
        print(f"  Approx GFLOPS: {mm_stats_print.get('gflops'):.2f}")
    else:
        print("  GEMM stats not available.")

    print("GGNN stats:")
    ggnn_stats_print = dict(ggnn_stats)
    if ggnn_stats_print:
        print(f"  Completed GGNN queries: {ggnn_stats_print.get('completed_ggnn_queries')}")
        avg_latency = ggnn_stats_print.get("avg_latency_ms")
        if avg_latency is not None:
            print(f"  Avg latency (ms): {avg_latency:.2f}")
        else:
            print("  Avg latency (ms): N/A")
    else:
        print("  GGNN stats not available.")
    
    print("DNN stats:")
    dnn_stats_print = dict(dnn_stats)
    if dnn_stats_print:
        print(f"  Completed DNN requests: {dnn_stats_print.get('completed_requests')}")
        print(f"  Inference count: {dnn_stats_print.get('inference_count')}")
    else:
        print("  DNN stats not available.")
    print()

    if use_multiprocess:
        manager.shutdown() # 关闭 Manager

if __name__ == '__main__':
    # 启用 multiprocessing.Process 时，如果导入了 CUDA，可能需要设置 start method
    if sys.platform.startswith('linux'):
        multiprocessing.set_start_method('spawn', force=True) # 通常用于避免 CUDA 相关问题
    main()
    exit_after_flush_if_morphx_preloaded()
