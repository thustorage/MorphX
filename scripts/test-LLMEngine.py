# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import pdb
import torch
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils import FlexibleArgumentParser
from vllm.inputs import TokensPrompt
import threading
import smsched_api
from pyinstrument import Profiler

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ["VLLM_USE_V1"] = "0"

def create_example_prompts() -> list[tuple[str, SamplingParams]]:
    """Create a list of test prompts with their sampling parameters."""
    return [
        ("A robot may not injure a human being",
         SamplingParams(temperature=0.0, logprobs=1, prompt_logprobs=1)),
        ("To be or not to be,",
         SamplingParams(temperature=0.0, logprobs=1, prompt_logprobs=1)),
        ("What is the meaning of life?",
         SamplingParams(temperature=0.0, logprobs=1, prompt_logprobs=1))
    ]
def create_randome_prompts(vocab_size, num_requests, input_len, output_len) -> list[tuple[TokensPrompt, SamplingParams]]:
    input_lens = torch.full((num_requests,), input_len, dtype=torch.int32, device="cuda")
    output_lens = torch.full((num_requests,), output_len, dtype=torch.int32, device="cuda")
    requests = []
    for i in range(num_requests):
        token_ids = [(i + j) % vocab_size for j in range(input_lens[i])]
        prompt = TokensPrompt(prompt_token_ids=token_ids)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=output_lens[i], min_tokens=output_lens[i])
        requests.append((prompt, sampling_params))
    return requests

import time

def process_requests(engine: LLMEngine,
                     test_prompts: list[tuple[str, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    request_states = {}
    # profiler = Profiler()
    # profiler.start()
    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            req_id_str = str(request_id)
            engine.add_request(req_id_str, prompt, sampling_params)
            request_states[req_id_str] = {
                "add_time": time.time(),
                "first_token_time": None,
                "last_token_time": None,
                "token_count": 0
            }
            request_id += 1

        # print("before step()...", flush=True)
        request_outputs: list[RequestOutput] = engine.step()
        if not request_outputs:
            # request_outputs = engine.get_prefill_result(isBlock=True)
            continue
        for request_output in request_outputs:
            req_id = request_output.request_id
            state = request_states[req_id]
            
            new_token_count = len(request_output.outputs[0].token_ids)
            
            # First token
            if state["first_token_time"] is None and new_token_count > 0:
                now = time.time()
                state["first_token_time"] = now
                state["last_token_time"] = now
                state["token_count"] = new_token_count
                ttft = now - state["add_time"]
                print(f"Request {req_id}: TTFT: {ttft:.4f} s")

            # Subsequent tokens
            elif new_token_count > state["token_count"]:
                now = time.time()
                state["last_token_time"] = now
                state["token_count"] = new_token_count

            if request_output.finished:
                finish_time = time.time()
                latency = finish_time - state["add_time"]
                total_tokens = state["token_count"]
                print(f"Request {req_id} finished.")
                print(f"  Latency: {latency:.4f} s")
                if total_tokens > 1 and state["first_token_time"] is not None:
                    # Exclude TTFT from throughput calculation
                    generation_time = finish_time - state["first_token_time"]
                    # The number of intervals between tokens is total_tokens - 1
                    avg_tbt = generation_time / (total_tokens - 1)
                    throughput = (total_tokens - 1) / generation_time if generation_time > 0 else float('inf')
                    print(f"  Total generated tokens: {total_tokens}")
                    print(f"  Average TBT: {avg_tbt:.4f} s/token")
                    print(f"  Throughput (tokens/s): {throughput:.2f}")
                elif total_tokens == 1:
                     print(f"  Total generated tokens: {total_tokens}")
                     print(f"  Only one token generated, no TBT.")
                else:
                    print(f"  No tokens were generated.")
    # profiler.stop()
    # html = profiler.output_html()
    # with open("LLMEngine.html", "w") as f:
    #     f.write(html)
    # print("Profile saved to profile.html")
    # profiler.open_in_browser()

model_paths = {
    "llama-2-7b": "/huggingface-cache/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590",
    "qwen3-4b": "/huggingface-cache/hub/models--Qwen--Qwen3-4B/snapshots/82d62bb073771e7a1ea59435f548908540217d1f",
    "qwen3-8b": "/huggingface-cache/hub/models--Qwen--Qwen3-8B/snapshots/2069b3fae1114555f3c020c81410e51fa0f656f2", 
    "gpt2-l": "/huggingface-cache/hub/models--openai-community--gpt2-large/snapshots/32b71b12589c2f8d625668d2335a01cac3249519"
}

def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs(model=model_paths['llama-2-7b'], 
                             enforce_eager=True, 
                             max_model_len=1155+211, 
                             max_num_batched_tokens=1155+211, 
                             gpu_memory_utilization=0.9)
    return LLMEngine.from_engine_args(engine_args)

import copy

def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    s0 = torch.cuda.Stream()
    with torch.cuda.stream(s0):
        engine0 = initialize_engine(args)
        prompts0 = create_randome_prompts(
            vocab_size=engine0.model_config.get_vocab_size(), 
            num_requests=30, input_len=1155, output_len=211)

        # Run one warmup with the same configuration before timed profiling.
        # The warmup will run the same prompts but will not record timing metrics.
        def run_warmup(engine: LLMEngine, prompts: list[tuple[TokensPrompt, SamplingParams]]):
            # Make a deep copy of prompts so we don't mutate the original list
            warmup_prompts = copy.deepcopy(prompts)
            # Use a simple loop: add requests and step until all finished. Do not record times.
            req_id = 0
            while warmup_prompts or engine.has_unfinished_requests():
                if warmup_prompts:
                    prompt, sampling_params = warmup_prompts.pop(0)
                    engine.add_request(str(req_id), prompt, sampling_params)
                    req_id += 1
                # Step the engine; discard outputs
                _ = engine.step()

        # Perform warmup before launching the timed processing thread (inside the stream)
        # try:
        #     run_warmup(engine0, prompts0)
        # except Exception as e:
        #     print(f"Warmup failed: {e}")
    s0.synchronize()

    # smsched_api.fix_sm_for_stream(s0, 0, 108)

    def process_in_thread(engine, prompts, stream):
        with torch.cuda.stream(stream):
            process_requests(engine, prompts)
    thread_s0 = threading.Thread(target=process_in_thread, args=(engine0, prompts0, s0))
    # thread_s1 = threading.Thread(target=process_in_thread, args=(engine1, prompts1, s1))
    thread_s0.start()
    # thread_s1.start()
    thread_s0.join()
    # thread_s1.join()

if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)