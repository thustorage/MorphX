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
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ["VLLM_USE_V1"] = "0"
# os.environ["VLLM_PREFILL_THREAD"] = "1"

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
    input_len = task.get("input_length")
    output_len = task.get("output_length")
    print(f"[internal] Tid: {tid}, input_len: {input_len}, output_len: {output_len}", flush=True)
    i = 0
    num_requests = 500

    with torch.cuda.stream(stream):
        prompts = create_prompts(engine.get_tokenizer().vocab_size, num_requests, input_len, output_len)
        # Enqueue all requests
        for i in range(num_requests):
            prompt, sampling_params = prompts[i]
            engine.add_request(str(i), prompt, sampling_params)

        start = time.time()
        per_request_throughputs = []
        served = 0

        # Drain engine and collect per-request metrics from RequestOutput objects
        while engine.has_unfinished_requests():
            request_outputs: list[RequestOutput] = engine.step()
            if request_outputs is None:
                continue
            elif len(request_outputs) == 0:
                request_outputs = engine.get_prefill_result(isBlock=True)

            for request_output in request_outputs:
                # Only consider finished requests
                if not getattr(request_output, 'finished', False):
                    continue

                metrics = getattr(request_output, 'metrics', None)
                if metrics is None:
                    continue

                # Prefer timestamps provided by vllm metrics
                first_token_time_abs = getattr(metrics, 'first_scheduled_time', None)
                finished_time_abs = getattr(metrics, 'finished_time', None)

                # Determine token count — follow test-pd-online.py: outputs[0].token_ids length
                try:
                    total_tokens = len(request_output.outputs[0].token_ids)
                except Exception:
                    total_tokens = int(output_len)

                # Compute per-request throughput using generation phase (first token -> finished)
                if first_token_time_abs is not None and finished_time_abs is not None and total_tokens > 1:
                    generation_time_s = finished_time_abs - first_token_time_abs
                    if generation_time_s <= 0:
                        tps = float('inf')
                    else:
                        tps = (total_tokens - 1) / generation_time_s
                    per_request_throughputs.append(tps)
                    served += 1

        end = time.time()
        print(f"[final] Tid {tid} finished.", flush=True)
        print(f"[final] served {num_requests} requests in {end - start:.2f}s", flush=True)
        print(f"[final] request throughput: {num_requests / (end - start):.2f} req/s", flush=True)
        print(f"[final] token throughput: {num_requests * output_len / (end - start):.2f} tokens/s", flush=True)

        if served > 0 and per_request_throughputs:
            avg_tps = sum(per_request_throughputs) / len(per_request_throughputs)
            print(f"[final] average per-request token throughput: {avg_tps:.2f} tokens/s over {len(per_request_throughputs)} requests", flush=True)
        else:
            print("[final] No per-request throughput samples were recorded.", flush=True)

        engine.wait_prefill_shutdown()

def initialize_engine(model, input_length, output_length, enable_chunked_prefill: bool) -> LLMEngine:
    context_length = input_length + output_length
    engine_args = EngineArgs(
        model=model_paths[model],
        enforce_eager=True,
        max_model_len=context_length,
        max_num_batched_tokens=context_length,
        max_num_seqs=30,
        enable_chunked_prefill=enable_chunked_prefill,
        gpu_memory_utilization=0.9,
    )
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
    parser = argparse.ArgumentParser(description="Run pd-offline workload")
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