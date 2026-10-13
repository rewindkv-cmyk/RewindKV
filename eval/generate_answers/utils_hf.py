import os
import json
import time
import collections

import torch

from utils.apply_chat_template import apply_chat_template


def repair_truncated_jsonl_tail(output_file):
    """Remove only an incomplete final JSONL record left by an interrupted write."""
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        return False

    with open(output_file, 'rb+') as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        f.seek(end - 1)
        if f.read(1) == b'\n':
            return False

        pos = end
        record_start = 0
        while pos > 0:
            chunk_start = max(0, pos - 65536)
            f.seek(chunk_start)
            chunk = f.read(pos - chunk_start)
            newline = chunk.rfind(b'\n')
            if newline >= 0:
                record_start = chunk_start + newline + 1
                break
            pos = chunk_start

        f.seek(record_start)
        tail = f.read(end - record_start)
        try:
            json.loads(tail.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            f.truncate(record_start)
            print(f"Removed truncated final JSONL record from {output_file}")
            return True

        f.seek(0, os.SEEK_END)
        f.write(b'\n')
        return False


def count_completed_samples(output_file):
    prompt_counts = collections.defaultdict(int)
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        repair_truncated_jsonl_tail(output_file)
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    prompt_counts[item['prompt']] += len(item.get('gen', []))
                except json.JSONDecodeError:
                    continue
    return prompt_counts


def count_generated_tokens(token_ids, eos_token_id):
    """Count generated tokens through the first EOS."""
    if eos_token_id is None:
        return token_ids.numel()

    eos_positions = (token_ids == eos_token_id).nonzero(as_tuple=True)[0]
    if eos_positions.numel() == 0:
        return token_ids.numel()
    return eos_positions[0].item() + 1


def batched_generate(
    model,
    tokenizer,
    output_file: str,
    batch_dicts: list,
    max_new_tokens: int = 32768,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = None,
):
    template_prompts = [apply_chat_template(tokenizer, d['prompt']) for d in batch_dicts]
    inputs = tokenizer(template_prompts, padding=True, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[-1]

    torch.cuda.synchronize()
    gen_start = time.perf_counter()
    outs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id,
    )
    torch.cuda.synchronize()
    # Wall time of the whole batch (shared by all of its samples).
    batch_generation_seconds = time.perf_counter() - gen_start

    generated_token_ids = [o[input_length:] for o in outs]
    responses = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_token_ids]
    batch_output_tokens = 0

    with open(output_file, 'a', encoding='utf-8') as g:
        for i, response in enumerate(responses):
            result = dict(batch_dicts[i])
            result["gen"] = [response]
            result["num_input_tokens"] = int(inputs.attention_mask[i].sum().item())
            result["num_output_tokens"] = count_generated_tokens(generated_token_ids[i], tokenizer.eos_token_id)
            result["batch_generation_seconds"] = batch_generation_seconds
            result["batch_size"] = len(batch_dicts)
            batch_output_tokens += result["num_output_tokens"]

            g.write(json.dumps(result, ensure_ascii=False) + '\n')
            g.flush()

    return batch_generation_seconds, batch_output_tokens
