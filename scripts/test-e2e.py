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

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

def create_prompts(vocab_size) -> list[tuple[TokensPrompt, SamplingParams]]:
    input_len = 2048
    output_len = 512
    num_requests = 8
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
                     prompts: list[tuple[str, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    # print vocab size
    print("Vocab size: ", engine.get_tokenizer().vocab_size)
    average_first_token_time = 0
    average_finished_time = 0
    request_num = 0
    while prompts or engine.has_unfinished_requests():
        if prompts:
            prompt, sampling_params = prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1

        request_outputs: list[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                arrival_time = request_output.metrics.arrival_time * 1e6
                first_scheduled_time = request_output.metrics.first_scheduled_time * 1e6 - arrival_time
                time_in_queue = request_output.metrics.time_in_queue * 1e6
                first_token_time = request_output.metrics.first_token_time * 1e6 - arrival_time
                finished_time = request_output.metrics.finished_time * 1e6 - arrival_time
                print("time in queue: ", time_in_queue, "first token time: ", first_token_time, "finished time: ", finished_time)
                average_first_token_time += first_token_time
                average_finished_time += finished_time
                request_num += 1
    print("average first token time: ", average_first_token_time / request_num)
    print("average finished time: ", average_finished_time / request_num)


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    engine_args = EngineArgs(model="/modelscope-cache/models/LLM-Research/llama-2-7b", 
                             enforce_eager=True, max_model_len=4096, gpu_memory_utilization=0.48)
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    s0 = torch.cuda.Stream()
    s1 = torch.cuda.Stream()
    with torch.cuda.stream(s0):
        engine0 = initialize_engine(args)
        prompts0 = create_prompts(engine0.get_tokenizer().vocab_size)
    s0.synchronize()
    with torch.cuda.stream(s1):
        engine1 = initialize_engine(args)
        prompts1 = create_prompts(engine0.get_tokenizer().vocab_size)
    s1.synchronize()

    smsched_api.fix_sm_for_stream(s0, 0, 108)
    smsched_api.fix_sm_for_stream(s1, 0, 108)

    # with torch.cuda.stream(s1):
    #     process_requests(engine1, prompts1)

    def process_in_thread(engine, prompts, stream):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record(stream)
        with torch.cuda.stream(stream):
            process_requests(engine, prompts)
        e1.record(stream)
        e1.synchronize()
        elapsed_time = e0.elapsed_time(e1)
        print("E2E elapsed time: ", elapsed_time)
    thread_s0 = threading.Thread(target=process_in_thread, args=(engine0, prompts0, s0))
    thread_s1 = threading.Thread(target=process_in_thread, args=(engine1, prompts1, s1))
    thread_s0.start()
    thread_s1.start()
    thread_s0.join()
    thread_s1.join()


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    
    main(args)