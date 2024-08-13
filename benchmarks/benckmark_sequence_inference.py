import argparse
import csv
import json
import random
import time
from typing import List, Optional, Tuple

from transformers import PreTrainedTokenizerBase

from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int],
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        if fixed_output_len is None:
            output_len = len(completion_token_ids[i])
        else:
            output_len = fixed_output_len
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 8192:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))
    # print(f"data_len:{data_len!r}")
    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


def run_vllm(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    trust_remote_code: bool,
    dtype: str,
    max_num_batched_tokens: Optional[int],
    max_num_seqs: Optional[int],
    pipeline_parallel_size: int,
    sequence_parallel_size: int,
    max_model_len: Optional[int],
    enable_long_sequence: bool,
) -> float:
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        pipeline_parallel_size=pipeline_parallel_size,
        sequence_parallel_size=sequence_parallel_size,
        max_model_len=max_model_len,
        enable_long_sequence=enable_long_sequence,
    )

    # Add the requests to the engine.
    for prompt, _, output_len in requests:
        sampling_params = SamplingParams(
            n=n,
            temperature=0.0 if use_beam_search else 1.0,
            top_p=1.0,
            use_beam_search=use_beam_search,
            ignore_eos=True,
            max_tokens=output_len,
        )
        # FIXME(woosuk): Do not use internal method.
        llm._add_request(
            prompt=prompt,
            prompt_token_ids=None,
            sampling_params=sampling_params,
        )

    start = time.perf_counter()
    # FIXME(woosuk): Do use internal method.
    llm._run_engine(use_tqdm=True)
    end = time.perf_counter()
    return end - start


# def run_hf(
#     requests: List[Tuple[str, int, int]],
#     model: str,
#     tokenizer: PreTrainedTokenizerBase,
#     n: int,
#     use_beam_search: bool,
#     max_batch_size: int,
#     trust_remote_code: bool,
# ) -> float:
#     assert not use_beam_search
#     llm = AutoModelForCausalLM.from_pretrained(
#         model, torch_dtype=torch.float16, trust_remote_code=trust_remote_code)
#     if llm.config.model_type == "llama":
#         # To enable padding in the HF backend.
#         tokenizer.pad_token = tokenizer.eos_token
#     llm = llm.cuda()

#     pbar = tqdm(total=len(requests))
#     start = time.perf_counter()
#     batch: List[str] = []
#     max_prompt_len = 0
#     max_output_len = 0
#     for i in range(len(requests)):
#         prompt, prompt_len, output_len = requests[i]
#         # Add the prompt to the batch.
#         batch.append(prompt)
#         max_prompt_len = max(max_prompt_len, prompt_len)
#         max_output_len = max(max_output_len, output_len)
#         if len(batch) < max_batch_size and i != len(requests) - 1:
#             # Check if we can add more requests to the batch.
#             _, next_prompt_len, next_output_len = requests[i + 1]
#            if (max(max_prompt_len, next_prompt_len) +
#                     max(max_output_len, next_output_len)) <= 2048:
#                 # We can add more requests to the batch.
#                 continue

#         # Generate the sequences.
#         input_ids = tokenizer(batch, return_tensors="pt",
#                               padding=True).input_ids
#         llm_outputs = llm.generate(
#             input_ids=input_ids.cuda(),
#             do_sample=not use_beam_search,
#             num_return_sequences=n,
#             temperature=1.0,
#             top_p=1.0,
#             use_cache=True,
#             max_new_tokens=max_output_len,
#         )
#         # Include the decoding time.
#         tokenizer.batch_decode(llm_outputs, skip_special_tokens=True)
#         pbar.update(len(batch))

#         # Clear the batch.
#         batch = []
#         max_prompt_len = 0
#         max_output_len = 0
#     end = time.perf_counter()
#     return end - start


def main(args: argparse.Namespace):
    # print(args)
    random.seed(args.seed)

    # Sample the requests.
    tokenizer = get_tokenizer(args.tokenizer,
                              trust_remote_code=args.trust_remote_code)
    requests = sample_requests(args.dataset, args.num_prompts, tokenizer,
                               args.output_len)

    if args.backend == "vllm":
        elapsed_time = run_vllm(
            requests,
            args.model,
            args.tokenizer,
            args.quantization,
            args.tensor_parallel_size,
            args.seed,
            args.n,
            args.use_beam_search,
            args.trust_remote_code,
            args.dtype,
            args.max_num_batched_tokens,
            args.max_num_seqs,
            args.pipeline_parallel_size,
            args.sequence_parallel_size,
            args.max_model_len,
            args.enable_long_sequence,
        )
    # elif args.backend == "hf":
    #     assert args.tensor_parallel_size == 1
    #     elapsed_time = run_hf(requests, args.model, tokenizer, args.n,
    #                           args.use_beam_search, args.hf_max_batch_size,
    #                           args.trust_remote_code)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    total_num_tokens = sum(prompt_len + output_len
                           for _, prompt_len, output_len in requests)
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s")
    if args.result is not None:
        throughput_requests = float(len(requests) / elapsed_time)
        throughput_tokens = float(total_num_tokens / elapsed_time)
        row = []
        row.append(args.tensor_parallel_size)
        row.append(args.pipeline_parallel_size)
        row.append(args.num_prompts)
        row.append(args.max_num_seqs)
        row.append(args.output_len)
        row.append(throughput_requests)
        row.append(throughput_tokens)
        with open(args.result, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm", "hf"],
                        default="vllm")
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="Path to the dataset.")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=['awq', 'squeezellm', None],
                        default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-max-batch-size",
                        type=int,
                        default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument('--max-num-batched-tokens',
                        type=int,
                        default=None,
                        help='maximum number of batched tokens per '
                        'iteration')
    parser.add_argument('--result',
                        type=str,
                        default="../result/result.csv",
                        help='Path of result csv file')
    parser.add_argument('--max-num-seqs',
                        type=int,
                        default=256,
                        help='maximum number of batched seqs per '
                        'iteration')
    parser.add_argument('--pipeline-parallel-size',
                        type=int,
                        default=1,
                        help='pipeline-parallel-size')
    parser.add_argument('--sequence-parallel-size',
                        type=int,
                        default=1,
                        help='sequence-parallel-size')
    parser.add_argument('--max-model-len',
                        type=int,
                        default=None,
                        help='enbale the long sequence for the'
                        ' distributed inference.')
    parser.add_argument('--enable-long-sequence',
                        type=bool,
                        default=False,
                        help='enbale the long sequence for the '
                        'distributed inference.')
    args = parser.parse_args()

    if args.backend == "vllm":
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
    elif args.backend == "hf":
        if args.hf_max_batch_size is None:
            raise ValueError("HF max batch size is required for HF backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
    if args.tokenizer is None:
        args.tokenizer = args.model

    main(args)