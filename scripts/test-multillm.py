# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import pdb
import time
import torch
import torch.nn as nn
import sys
import numpy as np
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils import FlexibleArgumentParser
from vllm.inputs import TokensPrompt
import threading
import json
import smsched_api

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
# os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ["VLLM_USE_V1"] = "0"

model_paths = {
    "llama-2-7b": "/huggingface-cache/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590",
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
    # timestamps = np.linspace(0 if tid == 0 else 1.5, total_time, int(rate * total_time), endpoint=False)
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
    start_time = time.time()
    # barrier.wait()

    with torch.cuda.stream(stream):
        prompts = create_prompts(engine.get_tokenizer().vocab_size, num_requests, input_len, output_len)
        while i < num_requests or engine.has_unfinished_requests():
            if i < num_requests and time.time() - start_time >= timestamps[i]:
                prompt, sampling_params = prompts[i]
                engine.add_request(str(i), prompt, sampling_params)
                i += 1
            if engine.has_unfinished_requests():
                # print(f"Step time: {time.time() - start_time:.2f}s", flush=True)
                # print("step...", flush=True)
                request_outputs: list[RequestOutput] = engine.step()
                if not request_outputs:
                    continue
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
        print(f"[final] Tid {tid} all requests finished.", flush = True)
        if served > 0:
            print(f"[final] Average first token time: {average_first_token_time / served}ms", flush = True)
            print(f"[final] Average finished time: {average_finished_time / served}ms", flush = True)
            print(f"[final] Average TBT: {total_avg_tbt / served} ms/token", flush=True)
            print(f"[final] Average throughput: {total_throughput / served} tokens/s", flush=True)
        print(f"[final] Total requests served: {served}, total time: {total_time:.2f}s, generated: {len(timestamps)}", flush = True)
        print(f"[final] Throughput: {served / total_time:.6f} requests/s", flush = True) 


def initialize_engine(model) -> LLMEngine:
    engine_args = EngineArgs(model=model_paths[model], 
                             enforce_eager=True, max_model_len=2048, gpu_memory_utilization=0.48)
    return LLMEngine.from_engine_args(engine_args)

def main(data, enable_task0, enable_task1, mps_div):
    total_time = data.get("time")
    tasks = data.get("tasks")

    s0 = torch.cuda.Stream()
    s1 = torch.cuda.Stream()
    with torch.cuda.stream(s0):
        if enable_task0:
            engine0 = initialize_engine(tasks[0].get("model"))
    s0.synchronize()
    with torch.cuda.stream(s1):
        if enable_task1:
            print(f"Initialize engine for task 1: {tasks[1]}", flush=True)
            engine1 = initialize_engine(tasks[1].get("model"))
    s1.synchronize()

    if mps_div is not None:
        smsched_api.fix_sm_for_stream(s0, 0, mps_div)
        smsched_api.fix_sm_for_stream(s1, mps_div, 108)

    if enable_task0:
        thread_s0 = threading.Thread(target=process_in_thread, args=(0, engine0, s0, total_time, tasks[0]))
    if enable_task1:
        thread_s1 = threading.Thread(target=process_in_thread, args=(1, engine1, s1, total_time, tasks[1]))
    if enable_task0:
        thread_s0.start()
    if enable_task1:
        thread_s1.start()
    if enable_task0:
        thread_s0.join()
    if enable_task1:
        thread_s1.join()


if __name__ == '__main__':
    if len(sys.argv) < 4:
        # to stderr
        print("Usage: python test-llmserve.py <json_path> <enable_task0> <enable_task1> (<mps_div>)", file=sys.stderr)
        sys.exit(1)
    json_path = sys.argv[1]
    enable_task0 = int(sys.argv[2])
    enable_task1 = int(sys.argv[3])
    if len(sys.argv) == 5:
        mps_div = int(sys.argv[4])
    else:
        mps_div = None
    if not os.path.exists(json_path):
        print(f"Error: File '{json_path}' not found.", file=sys.stderr)
        sys.exit(1)
    with open(json_path, 'r') as f:
        data = json.load(f, parse_int=float, parse_float=float)
        main(data, enable_task0, enable_task1, mps_div)