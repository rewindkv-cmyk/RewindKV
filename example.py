"""Interactive demo: generate an answer to one prompt with a compressed KV cache.

    python example.py --method rewindkv
    python example.py --method rpc
    python example.py --method fullkv
"""

import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from rewindkv import enable_kv_compression, set_compression_config
from utils.apply_chat_template import apply_chat_template


def main(model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
         method: str = "rewindkv",
         max_new_tokens: int = 32768,
         P: int = 1024,
         c: int = 4,
         **method_kwargs):
    """``method_kwargs`` are forwarded to ``set_compression_config`` (e.g. --K 128 --alpha 0.5)."""
    if method != "fullkv":
        enable_kv_compression()

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    streamer = TextStreamer(tokenizer)

    if method != "fullkv":
        set_compression_config(model, method=method, P=P, c=c, **method_kwargs)

    prompt = apply_chat_template(tokenizer, input("Prompt: "))
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    context_length = inputs.input_ids.shape[-1]

    with torch.no_grad():
        outputs = model.generate(**inputs,
                                 max_new_tokens=max_new_tokens,
                                 pad_token_id=tokenizer.eos_token_id,
                                 do_sample=True,
                                 temperature=0.6,
                                 top_p=0.95,
                                 streamer=streamer)

    print(f"\nContext Length: {context_length}")
    print(f"Output Length: {outputs[0][context_length:].shape[-1]}\n")


if __name__ == '__main__':
    fire.Fire(main)
