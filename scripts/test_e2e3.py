# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import threading # Import the threading module
import time # Import time for potential delays if needed, and for metrics
import torch
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils import FlexibleArgumentParser
from vllm.inputs import TokensPrompt
import multiprocessing

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

MODEL_PATH = "/modelscope-cache/models/LLM-Research/Meta-Llama-3-8B-Instruct"
MAX_MODEL_LEN = 1024


GPU_MEM_UTIL = 0.95
def create_prompts(vocab_size) -> list[tuple[TokensPrompt, SamplingParams]]:
    input_len = 32
    output_len = 16
    num_requests = 4
    input_lens = torch.full((num_requests,), input_len, dtype=torch.int32, device="cuda")
    output_lens = torch.full((num_requests,), output_len, dtype=torch.int32, device="cuda")
    requests = []
    for i in range(num_requests):
        token_ids = [(i + j) % vocab_size for j in range(input_lens[i])]
        prompt = TokensPrompt(prompt_token_ids=token_ids)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=output_lens[i])
        requests.append((prompt, sampling_params))
    return requests


def process_requests(engine: LLMEngine,
                     prompts: list[tuple[TokensPrompt, SamplingParams]],
                     engine_id: int):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    num_initial_prompts = len(prompts)
    print(f"[Engine {engine_id}] Starting processing for {num_initial_prompts} prompts.")
    print(f"[Engine {engine_id}] Vocab size: {engine.get_tokenizer().vocab_size}")
    while prompts or engine.has_unfinished_requests():
        if prompts:
            prompt, sampling_params = prompts.pop(0)
            request_id_str = f"eng{engine_id}_req{request_id}"
            engine.add_request(request_id_str, prompt, sampling_params)
            request_id += 1

        request_outputs: list[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                arrival_time = request_output.metrics.arrival_time / 1e6
                first_scheduled_time = request_output.metrics.first_scheduled_time / 1e6 - arrival_time
                time_in_queue = request_output.metrics.time_in_queue / 1e6
                first_token_time = request_output.metrics.first_token_time / 1e6 - arrival_time
                finished_time = request_output.metrics.finished_time / 1e6 - arrival_time
                
                print(f"[Engine {engine_id}] Request {request_output.request_id} Finished. "
                      f"Time in Queue: {time_in_queue:.4f}s, "
                      f"First Token Time: {first_token_time:.4f}s, "
                      f"Finished Time: {finished_time:.4f}s")

def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    engine_args = EngineArgs(model=MODEL_PATH,
                             enforce_eager=True,
                             max_model_len=MAX_MODEL_LEN,
                             gpu_memory_utilization=GPU_MEM_UTIL,
                             )
    return LLMEngine.from_engine_args(engine_args)


def run_engine_thread(engine_id: int, args):
    engine = initialize_engine(args)
    prompts = create_prompts(engine.get_tokenizer().vocab_size)
    s_thread = torch.cuda.Stream()
    with torch.cuda.stream(s_thread):
        process_requests(engine, prompts, engine_id)
    s_thread.synchronize()

def main(args: argparse.Namespace):
    process1 = multiprocessing.Process(target=run_engine_thread, args=(1, args))
    process2 = multiprocessing.Process(target=run_engine_thread, args=(2, args))
    process1.start()
    process2.start()
    process1.join()
    process2.join()

if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Demo on using the LLMEngine class directly with two engines in parallel threads')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    
    main(args)