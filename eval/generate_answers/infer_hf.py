import argparse
import copy
import json
import os
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from rewindkv import enable_kv_compression, set_compression_config
from eval.generate_answers.utils_hf import count_completed_samples, batched_generate


def main():
    parser = argparse.ArgumentParser(description="Sample answers for the prompts of a jsonl file")
    parser.add_argument("--input_file", type=str, required=True, help="Input jsonl file (one {'prompt', ...} per line)")
    parser.add_argument("--output_file", type=str, required=True, help="Output jsonl file; existing samples are resumed")
    parser.add_argument("--model_path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--n_samples", type=int, default=8, help="Number of samples per prompt")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of samples generated together")
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=None)
    # KV-cache compression
    parser.add_argument("--method", type=str, default="rewindkv", choices=("fullkv", "rpc", "rewindkv"))
    parser.add_argument("--P", type=int, default=1024, help="Compression period")
    parser.add_argument("--c", type=int, default=4, help="Compression ratio (P / c tokens kept per round)")
    parser.add_argument("--R", type=int, default=None, help="[rpc] number of recent selector queries (default 32)")
    parser.add_argument("--K", type=int, default=None, help="[rewindkv] number of selector queries (default 128)")
    parser.add_argument("--block_size", type=int, default=None, help="[rewindkv] queries per block (default 32)")
    parser.add_argument("--alpha", type=float, default=None, help="[rewindkv] recency weight vs. diversity (default 0.5)")
    parser.add_argument("--half_life", type=float, default=None, help="[rewindkv] recency half-life in blocks (default 4)")
    parser.add_argument("--scoring", type=str, default=None, choices=("attention_sum", "attention_max"),
                        help="Default: attention_sum for rpc, attention_max for rewindkv")
    parser.add_argument("--aggregation", type=str, default=None, choices=("all", "group"),
                        help="Default: all for rpc, group for rewindkv")
    parser.add_argument("--pooling", type=str, default=None, choices=("avgpool", "maxpool"),
                        help="Local pooling for attention_sum (default avgpool)")
    parser.add_argument("--kernel_size", type=int, default=7, help="Local pooling kernel size")
    args = parser.parse_args()

    if os.path.exists(args.output_file):
        completed_counts = count_completed_samples(args.output_file)
        print(f"Found {sum(completed_counts.values())} completed samples from previous run")
    else:
        os.makedirs(Path(args.output_file).parent, exist_ok=True)
        open(args.output_file, 'w', encoding='utf-8').close()
        completed_counts = dict()

    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(l) for l in f]

    expanded_data = []
    for item in data:
        remaining = max(args.n_samples - completed_counts.get(item['prompt'], 0), 0)
        for _ in range(remaining):
            expanded_data.append(copy.deepcopy(item))

    total_tasks = len(expanded_data)
    print(f"Total remaining samples to process: {total_tasks}")

    if args.method != "fullkv":
        enable_kv_compression()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = 'left'

    if args.method != "fullkv":
        set_compression_config(
            model,
            method=args.method,
            P=args.P,
            c=args.c,
            R=args.R,
            K=args.K,
            block_size=args.block_size,
            alpha=args.alpha,
            half_life=args.half_life,
            scoring=args.scoring,
            aggregation=args.aggregation,
            pooling=args.pooling,
            kernel_size=args.kernel_size,
        )
    else:
        print("[KV compression] method=fullkv (no compression)")

    total_generation_seconds = 0.0
    total_output_tokens = 0
    if torch.cuda.is_available():
        for dev in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device=dev)

    for i in tqdm(range(0, len(expanded_data), args.batch_size)):
        batch_dicts = expanded_data[i : i + args.batch_size]
        print(f"[Timestamp: {datetime.now()}][{total_tasks} samples remaining, {len(batch_dicts)} in this batch]")

        batch_seconds, batch_output_tokens = batched_generate(
            model=model,
            tokenizer=tokenizer,
            output_file=args.output_file,
            batch_dicts=batch_dicts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        total_generation_seconds += batch_seconds
        total_output_tokens += batch_output_tokens
        total_tasks -= len(batch_dicts)

    if total_generation_seconds > 0:
        peak_memory_gb = 0.0
        if torch.cuda.is_available():
            peak_memory_gb = sum(
                torch.cuda.max_memory_allocated(device=dev)
                for dev in range(torch.cuda.device_count())
            ) / 1024**3
        print(
            f"[Throughput Summary] output_tokens={total_output_tokens} "
            f"generation_seconds={total_generation_seconds:.2f} "
            f"tokens_per_second={total_output_tokens / total_generation_seconds:.2f} "
            f"peak_memory_gb={peak_memory_gb:.2f} "
            f"batch_size={args.batch_size}"
        )


if __name__ == "__main__":
    main()
