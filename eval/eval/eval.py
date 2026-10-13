import argparse
import json
import os

from math_opensource import compute_scores


def get_after_think(text):
    parts = text.split("\n</think>\n\n", 1)
    if len(parts) > 1:
        return parts[1]
    return text


def print_token_statistics(data):
    if not data:
        return
    input_lengths = [item.get("num_input_tokens") for item in data]
    output_lengths = [item.get("num_output_tokens") for item in data]
    if all(length is not None for length in input_lengths):
        print(f"Avg Number of Input Tokens: {sum(input_lengths) / len(input_lengths):.2f}")
    if all(length is not None for length in output_lengths):
        print(f"Avg Number of Output Tokens: {sum(output_lengths) / len(output_lengths):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Score sampled answers")
    parser.add_argument("--input_path", type=str, required=True, help="jsonl written by infer_hf")
    parser.add_argument("--cache_path", type=str, required=True, help="jsonl to save per-sample scores")
    parser.add_argument("--task_name", type=str, required=True,
                        choices=("math_opensource/aime24", "math_opensource/aime25"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.cache_path)), exist_ok=True)

    with open(args.input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    for item in data:
        item["task"] = args.task_name
        item['gen'][0] = get_after_think(item['gen'][0])

    acc = compute_scores(data, args.cache_path)
    print(f"Task: {args.task_name}, Accuracy: {acc}")
    print_token_statistics(data)


if __name__ == "__main__":
    main()
