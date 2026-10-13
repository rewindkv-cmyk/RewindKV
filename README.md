# RewindKV

Official implementation of **RewindKV**, a training-free KV-cache compression method for reasoning LLMs.
This repository also includes **RPC** (Reasoning Path Compression, [arXiv:2505.13866](https://arxiv.org/abs/2505.13866)) as a baseline, run through the same code path.

> This repository is anonymized for double-blind review.

## Overview

Reasoning models generate long chains of thought, so the KV cache of *generated* tokens dominates memory and decoding cost.
RewindKV and RPC both compress that cache periodically during decoding:

* Every **P** generated tokens, each layer scores its past generated tokens with a set of **selector queries** and keeps only the top-scoring ones.
* The first compression happens after P + R (RPC) or P + K (RewindKV) generated tokens, then every P tokens.
* After the *n*-th compression, the generated part of the cache keeps **n · P / c** tokens (plus the selector tokens). The prompt is never compressed.

The two methods differ in how they choose the selector queries and how they score the tokens.

| | RPC | RewindKV |
|---|---|---|
| Selector queries | Last **R** queries | **K** queries picked from *all* queries generated since the last compression, by block-wise selection |
| Token score | Sum over selectors, mean over all heads, local average pooling | Max over selectors, max over the query heads of each KV head (one score per KV head) |
| Always kept | The last R tokens | The K selector tokens |
| Default hyper-parameters | P=1024, c=4, R=32 | P=1024, c=4, K=128, block size 32, α=0.5, half-life 4 |

### RewindKV selector

At each compression, the *N* queries generated since the previous compression are split into *M* blocks of `block_size` consecutive queries (m = 0 is the oldest block).

1. **Recency.** $r_m = 2^{-\Delta_m / h}$ with $\Delta_m = M - 1 - m$ (distance to the latest block, in blocks) and half-life *h*.
2. **Diversity.** $d_m$ = mean pairwise cosine distance between the (head-averaged, L2-normalized) queries of block *m*.
3. **Utility.** Each signal is normalized to sum to one over blocks: $u_m = \alpha\,\hat r_m + (1-\alpha)\,\hat d_m$.
4. **Budget.** Every block first gets one selector; the remaining $K - M$ are split proportionally to $u_m$ with the largest-remainder method (floor, then one more slot to the blocks with the largest fractional parts). If $K < M$, all *K* are split proportionally.
5. **Within a block.** The first selector is the block medoid; the rest are chosen by farthest-point sampling in cosine distance.

The original (per-head) query vectors at the selected positions are then used as selectors to score all generated tokens.
Selection and compression are done independently for every layer, while all layers compress at the same decoding step and keep the same number of tokens.

## Installation

```bash
conda create -n rewindkv python=3.11
conda activate rewindkv
pip install -r requirements.txt
# flash-attn must match your CUDA / torch build
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

The code targets **transformers 4.45** and Qwen2-architecture models (e.g. `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, `nvidia/OpenReasoning-Nemotron-14B`).

## Usage

### Python API

```python
import torch
from transformers import AutoModelForCausalLM
from rewindkv import enable_kv_compression, set_compression_config

enable_kv_compression()  # call before loading the model
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
set_compression_config(model, method="rewindkv", P=1024, c=4, K=128, alpha=0.5, half_life=4)
# or: set_compression_config(model, method="rpc", P=1024, c=4, R=32)
# then call model.generate(...) as usual
```

### Interactive demo

```bash
python example.py --method rewindkv     # or rpc / fullkv
```

### AIME 2024 / 2025 evaluation

`scripts/run_aime.sh` samples answers and scores them. Outputs go to `outputs/<model>/`, scores to `results/<model>/`.

```bash
METHOD=rewindkv bash scripts/run_aime.sh
METHOD=rpc      bash scripts/run_aime.sh
METHOD=fullkv   bash scripts/run_aime.sh

# other settings
METHOD=rewindkv ALPHA=0.7 K=64 DATASETS="aime25" CUDA_VISIBLE_DEVICES=1 bash scripts/run_aime.sh
MODEL=nvidia/OpenReasoning-Nemotron-14B METHOD=rewindkv bash scripts/run_aime.sh
```

Sampling uses temperature 0.6, top-p 0.95, up to 32,768 new tokens, and 8 samples per problem; the reported accuracy is averaged over all samples.
Generation resumes from an existing output file, so an interrupted run can be restarted with the same command.

The two steps can also be run separately:

```bash
python -m eval.generate_answers.infer_hf \
    --input_file eval/data/aime24.jsonl \
    --output_file outputs/r1-7b/aime24_rewindkv.jsonl \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --n_samples 8 --batch_size 8 \
    --method rewindkv --P 1024 --c 4 --K 128 --alpha 0.5 --half_life 4

python eval/eval/eval.py \
    --input_path outputs/r1-7b/aime24_rewindkv.jsonl \
    --cache_path results/r1-7b/aime24_rewindkv.jsonl \
    --task_name math_opensource/aime24
```

### Arguments

| Argument | Method | Default | Description |
|---|---|---|---|
| `--method` | | `rewindkv` | `fullkv`, `rpc` or `rewindkv` |
| `--P` | both | 1024 | Compression period (generated tokens between compressions) |
| `--c` | both | 4 | Compression ratio; each round keeps P / c more tokens |
| `--R` | rpc | 32 | Number of most recent selector queries |
| `--K` | rewindkv | 128 | Number of selector queries |
| `--block_size` | rewindkv | 32 | Queries per block |
| `--alpha` | rewindkv | 0.5 | Recency weight (1 − α weights diversity) |
| `--half_life` | rewindkv | 4 | Recency half-life, in blocks |
| `--scoring` | both | rpc: `attention_sum`, rewindkv: `attention_max` | Token scoring |
| `--aggregation` | both | rpc: `all`, rewindkv: `group` | One score shared by all KV heads (`all`) or one per KV head (`group`) |
| `--pooling`, `--kernel_size` | both | `avgpool`, 7 | Local pooling, used with `attention_sum` only |

## Repository structure

```
rewindkv/
  cluster.py          # selector selection, token scoring and cache compression (RPC and RewindKV)
  qwen2.py            # Qwen2 flash-attention layer with periodic compression
eval/
  generate_answers/   # batched sampling (infer_hf.py)
  eval/               # AIME answer extraction and scoring
  data/               # AIME 2024 / 2025 problems
scripts/run_aime.sh   # end-to-end AIME evaluation
example.py            # interactive demo
```

## Acknowledgements

The code is built on the official implementation of RPC ([jiwonsong-dev/ReasoningPathCompression](https://github.com/jiwonsong-dev/ReasoningPathCompression), MIT License). The math answer checker reuses open-source evaluation utilities credited in `eval/eval/math_opensource_utils/grader.py`.

## License

MIT License. See [LICENSE](LICENSE).
