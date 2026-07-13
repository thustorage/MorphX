# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import pdb
import time
import torch
import torch.nn as nn
import sys
import numpy as np
from contextlib import contextmanager
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils import FlexibleArgumentParser
from vllm.inputs import TokensPrompt
import threading
import json
import smsched_api

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ["VLLM_USE_V1"] = "0"
# os.environ["VLLM_PREFILL_THREAD"] = "1"


@contextmanager
def without_child_ld_preload():
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

model_paths = {
    "qwen2.5-7b": "/huggingface-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
    "glm-9b": "/huggingface-cache/hub/models--zai-org--GLM-Z1-9B-0414/snapshots/b221b06fefb23ca320922cf6e68ab5f2fb82de81",
    "llama-2-7b": "/huggingface-cache/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590",
    "llama-3-8b": "/huggingface-cache/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
    "qwen3-4b": "/huggingface-cache/hub/models--Qwen--Qwen3-4B/snapshots/82d62bb073771e7a1ea59435f548908540217d1f",
    "qwen3-8b": "/huggingface-cache/hub/models--Qwen--Qwen3-8B/snapshots/2069b3fae1114555f3c020c81410e51fa0f656f2", 
    "gpt2-l": "/huggingface-cache/hub/models--openai-community--gpt2-large/snapshots/32b71b12589c2f8d625668d2335a01cac3249519"
}

def create_prompts(vocab_size, num_requests, input_len, output_len) -> list[tuple[TokensPrompt, SamplingParams]]:
    input_lens = torch.full((num_requests,), input_len, dtype=torch.int32, device="cuda")
    output_lens = torch.full((num_requests,), output_len, dtype=torch.int32, device="cuda")
    requests = []
    for i in range(num_requests):
        token_ids = [(i + j) % vocab_size for j in range(input_lens[i])]
        prompt = TokensPrompt(prompt_token_ids=token_ids)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=output_lens[i], min_tokens=output_lens[i])
        requests.append((prompt, sampling_params))
    return requests

lock = threading.Lock()
barrier = threading.Barrier(2)

def process_in_thread(tid, engine, stream, total_time, task):
    print(f"process in thread {tid} started.", flush=True)
    rng = np.random.default_rng(seed=tid)
    rate = task.get("rps")
    intervals = rng.exponential(1 / rate, int(rate * total_time * 2))
    timestamps = np.cumsum(intervals)
    timestamps = timestamps[timestamps < total_time]
    # timestamps = [0 for i in range(50)] # All come before start
    # timestamps = np.linspace(0 if tid == 0 else 1.5, total_time, int(rate * total_time), endpoint=False) # Uniform arrival
    num_requests = len(timestamps)
    input_len = task.get("input_length")
    output_len = task.get("output_length")

    print(f"[internal] Tid: {tid}, Request timestamps: {timestamps}, input_len: {input_len}, output_len: {output_len}", flush=True)
    i = 0
    average_first_token_time = 0
    average_finished_time = 0
    total_avg_tbt = 0
    total_throughput = 0
    served = 0
    # barrier.wait()

    with torch.cuda.stream(stream):
        prompts = create_prompts(engine.get_tokenizer().vocab_size, num_requests, input_len, output_len)
        start_time = time.time()
        while i < num_requests or engine.has_unfinished_requests():
            if i < num_requests and time.time() - start_time >= timestamps[i]:
                prompt, sampling_params = prompts[i]
                engine.add_request(str(i), prompt, sampling_params)
                i += 1
            if time.time() - start_time > total_time:
                break
            if engine.has_unfinished_requests():
                # print(f"Step time: {time.time() - start_time:.2f}s", flush=True)
                # print("step...", flush=True)
                request_outputs: list[RequestOutput] = engine.step()
                if request_outputs is None:
                    continue
                elif len(request_outputs) == 0:
                    request_outputs = engine.get_prefill_result(isBlock=True)

                for request_output in request_outputs:
                    if request_output.finished:
                        request_id = request_output.request_id
                        arrival_time_abs = request_output.metrics.arrival_time
                        # timings from metrics (seconds, epoch-based)
                        arrival_time_abs = request_output.metrics.arrival_time
                        first_token_time_abs = request_output.metrics.first_token_time
                        finished_time_abs = request_output.metrics.finished_time

                        # time in queue (if provided directly use it, else compute)
                        time_in_queue = None
                        if hasattr(request_output.metrics, 'time_in_queue') and request_output.metrics.time_in_queue is not None:
                            # vllm may provide time_in_queue in seconds
                            time_in_queue = request_output.metrics.time_in_queue * 1000
                        else:
                            # fallback: difference between first_scheduled_time and arrival_time
                            if hasattr(request_output.metrics, 'first_scheduled_time') and request_output.metrics.first_scheduled_time is not None:
                                time_in_queue = (request_output.metrics.first_scheduled_time - arrival_time_abs) * 1000

                        # prefill time: time between first_scheduled_time and first_token_time (best effort)
                        prefill_time_ms = None
                        if hasattr(request_output.metrics, 'first_scheduled_time') and request_output.metrics.first_scheduled_time is not None and first_token_time_abs is not None:
                            prefill_time_ms = (first_token_time_abs - request_output.metrics.first_scheduled_time) * 1000
                        else:
                            # fallback to scheduler_time if available (scheduler_time is local scheduling overhead)
                            if hasattr(request_output.metrics, 'scheduler_time') and request_output.metrics.scheduler_time is not None:
                                prefill_time_ms = request_output.metrics.scheduler_time * 1000
                            # else try model forward/execute times sum
                            elif (hasattr(request_output.metrics, 'model_forward_time') and request_output.metrics.model_forward_time) or (hasattr(request_output.metrics, 'model_execute_time') and request_output.metrics.model_execute_time):
                                mf = request_output.metrics.model_forward_time or 0
                                me = request_output.metrics.model_execute_time or 0
                                prefill_time_ms = (mf + me) * 1000

                        ttft = (first_token_time_abs - arrival_time_abs) * 1000  # ms
                        latency = (finished_time_abs - arrival_time_abs) * 1000 # ms
                        
                        total_tokens = len(request_output.outputs[0].token_ids)

                        print(f"Tid {tid} Request {request_id} finished.")
                        # print queue wait and prefill times when available
                        if time_in_queue is not None:
                            print(f"  Time in queue before prefill: {time_in_queue:.4f} ms")
                        else:
                            print(f"  Time in queue before prefill: (n/a)")

                        if prefill_time_ms is not None:
                            print(f"  Prefill time (approx): {prefill_time_ms:.4f} ms")
                        else:
                            print(f"  Prefill time (approx): (n/a)")

                        print(f"  TTFT: {ttft:.4f} ms")
                        print(f"  Latency: {latency:.4f} ms")

                        if total_tokens > 1:
                            generation_time_s = finished_time_abs - first_token_time_abs
                            avg_tbt = (generation_time_s / (total_tokens - 1)) * 1000 # ms
                            throughput = (total_tokens - 1) / generation_time_s if generation_time_s > 0 else float('inf')
                            print(f"  Total generated tokens: {total_tokens}")
                            print(f"  Average TBT: {avg_tbt:.4f} ms/token")
                            print(f"  Throughput (tokens/s): {throughput:.2f}")
                            if time.time() - start_time <= total_time:
                                total_avg_tbt += avg_tbt
                                total_throughput += throughput
                        elif total_tokens == 1:
                            print(f"  Total generated tokens: {total_tokens}")
                            print(f"  Only one token generated, no TBT.")
                        else:
                            print(f"  No tokens were generated.")

                        if time.time() - start_time <= total_time:
                            served += 1
                            average_first_token_time += ttft
                            average_finished_time += latency
        print(f"[final] Tid {tid} finished.", flush = True)
        if served > 0:
            print(f"[final] Average first token time: {average_first_token_time / served}ms", flush = True)
            print(f"[final] Average finished time: {average_finished_time / served}ms", flush = True)
            print(f"[final] Average TBT: {total_avg_tbt / served} ms/token", flush=True)
            print(f"[final] Average throughput: {total_throughput / served} tokens/s", flush=True)
        print(f"[final] Total requests served: {served}, total time: {total_time:.2f}s, generated: {len(timestamps)}", flush = True)
        print(f"[final] Throughput: {served / total_time:.6f} requests/s", flush = True) 

        engine.wait_prefill_shutdown()

def initialize_engine(model, input_length, output_length, enable_chunked_prefill: bool) -> LLMEngine:
    context_length = input_length + output_length
    engine_args = EngineArgs(
        model=model_paths[model],
        enforce_eager=True,
        max_model_len=context_length,
        max_num_batched_tokens=context_length,
        max_num_seqs=128,
        enable_chunked_prefill=enable_chunked_prefill,
        gpu_memory_utilization=0.9,
    )
    with without_child_ld_preload():
        return LLMEngine.from_engine_args(engine_args)


def main(data, enable_chunked_prefill: bool = False):
    total_time = data.get("time")

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        engine = initialize_engine(
            data.get("model"),
            int(data.get("input_length")),
            int(data.get("output_length")),
            enable_chunked_prefill,
        )
    stream.synchronize()

    process_in_thread(0, engine, stream, total_time, data)


def parse_args():
    parser = argparse.ArgumentParser(description="Run pd-online workload")
    parser.add_argument("config_path", help="Path to workload configuration JSON")
    parser.add_argument(
        "--enable-chunked-prefill",
        action="store_true",
        dest="enable_chunked_prefill",
        help="Enable vLLM chunked prefill",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.config_path):
        print(f"Error: File '{args.config_path}' not found.", file=sys.stderr)
        sys.exit(1)
    with open(args.config_path, 'r') as f:
        data = json.load(f, parse_int=float, parse_float=float)
        main(data, enable_chunked_prefill=args.enable_chunked_prefill)
    exit_after_flush_if_morphx_preloaded()
