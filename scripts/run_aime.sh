#!/bin/bash
# Sample and score AIME24 / AIME25 with FullKV, RPC or RewindKV.
#
#   METHOD=rewindkv bash scripts/run_aime.sh
#   METHOD=rpc      bash scripts/run_aime.sh
#   METHOD=fullkv   bash scripts/run_aime.sh
#   METHOD=rewindkv ALPHA=0.7 DATASETS="aime25" CUDA_VISIBLE_DEVICES=1 bash scripts/run_aime.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}
MODEL_NICKNAME=${MODEL_NICKNAME:-$(basename "$MODEL")}
METHOD=${METHOD:-rewindkv}          # fullkv | rpc | rewindkv
DATASETS=${DATASETS:-"aime24 aime25"}
N_SAMPLES=${N_SAMPLES:-8}
BSZ=${BSZ:-8}

P=${P:-1024}; C=${C:-4}
R=${R:-32}                          # rpc
K=${K:-128}; BLOCK_SIZE=${BLOCK_SIZE:-32}; ALPHA=${ALPHA:-0.5}; HALF_LIFE=${HALF_LIFE:-4}   # rewindkv

case "$METHOD" in
    fullkv)   TAG="fullkv";              ARGS=(--method fullkv) ;;
    rpc)      TAG="rpc-P$P-R$R-c$C";     ARGS=(--method rpc --P "$P" --c "$C" --R "$R") ;;
    rewindkv) TAG="rewindkv-P$P-K$K-c$C-a$ALPHA-hl$HALF_LIFE"
              ARGS=(--method rewindkv --P "$P" --c "$C" --K "$K" --block_size "$BLOCK_SIZE" \
                    --alpha "$ALPHA" --half_life "$HALF_LIFE") ;;
    *) echo "unknown METHOD=$METHOD" >&2; exit 1 ;;
esac

for DATASET in $DATASETS; do
    NAME="${DATASET}_n${N_SAMPLES}_${TAG}"
    OUT="outputs/$MODEL_NICKNAME/$NAME.jsonl"
    RES="results/$MODEL_NICKNAME/$NAME.jsonl"
    mkdir -p "outputs/$MODEL_NICKNAME" "results/$MODEL_NICKNAME"

    python -m eval.generate_answers.infer_hf \
        --input_file "eval/data/$DATASET.jsonl" \
        --output_file "$OUT" \
        --model_path "$MODEL" \
        --n_samples "$N_SAMPLES" \
        --batch_size "$BSZ" \
        "${ARGS[@]}"

    python eval/eval/eval.py \
        --input_path "$OUT" \
        --cache_path "$RES" \
        --task_name "math_opensource/$DATASET" | tee "results/$MODEL_NICKNAME/$NAME.txt"
done
