import argparse
import json
import os
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []

    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, list):
            return obj

        if isinstance(obj, dict):
            # 兼容 {"data": [...]} 这种格式
            for key in ["data", "train", "examples", "samples"]:
                if key in obj and isinstance(obj[key], list):
                    return obj[key]

            # 单条样本
            return [obj]

    raise ValueError(f"不支持的数据格式: {path}，请使用 .json 或 .jsonl")


def get_question_text(sample: Dict[str, Any]) -> str:
    """
    自动兼容常见数据格式：
    1. Alpaca: instruction + input + output
    2. 简单问答: question / answer
    3. prompt / response
    4. ShareGPT: conversations / messages
    """

    # 最常见：question 字段
    if "question" in sample:
        return str(sample.get("question") or "")

    # Alpaca 格式
    if "instruction" in sample:
        instruction = str(sample.get("instruction") or "")
        input_text = str(sample.get("input") or "")
        if input_text.strip():
            return instruction + "\n" + input_text
        return instruction

    # prompt 格式
    if "prompt" in sample:
        return str(sample.get("prompt") or "")

    # query 格式
    if "query" in sample:
        return str(sample.get("query") or "")

    # ShareGPT / OpenAI messages 格式
    if "conversations" in sample and isinstance(sample["conversations"], list):
        return extract_first_user_message(sample["conversations"])

    if "messages" in sample and isinstance(sample["messages"], list):
        return extract_first_user_message(sample["messages"])

    # 如果字段不匹配，返回空
    return ""


def extract_first_user_message(messages: List[Dict[str, Any]]) -> str:
    for msg in messages:
        role = msg.get("role") or msg.get("from")
        content = msg.get("content") or msg.get("value") or ""

        if role in ["user", "human"]:
            return str(content)

    return ""


def percentile(values: List[int], p: float) -> int:
    if not values:
        return 0

    values = sorted(values)
    idx = int(len(values) * p)
    idx = min(idx, len(values) - 1)
    return values[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型路径或 Hugging Face 模型名，例如 Qwen/Qwen3.6-27B 或 /path/to/model",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="数据集路径，支持 .json / .jsonl",
    )
    parser.add_argument(
        "--save_detail",
        type=str,
        default=None,
        help="可选：保存每条样本的 token 统计到 jsonl",
    )

    args = parser.parse_args()

    print(f"Loading tokenizer from: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=False,
    )

    print(f"Loading dataset from: {args.data}")
    dataset = load_json_or_jsonl(args.data)

    lengths = []
    max_tokens = -1
    max_index = -1
    max_text = ""

    detail_writer = None
    if args.save_detail:
        os.makedirs(os.path.dirname(args.save_detail) or ".", exist_ok=True)
        detail_writer = open(args.save_detail, "w", encoding="utf-8")

    empty_count = 0

    for idx, sample in enumerate(dataset):
        question_text = get_question_text(sample)

        if not question_text.strip():
            empty_count += 1

        token_ids = tokenizer.encode(
            question_text,
            add_special_tokens=False,
        )

        token_len = len(token_ids)
        lengths.append(token_len)

        if token_len > max_tokens:
            max_tokens = token_len
            max_index = idx
            max_text = question_text

        if detail_writer:
            detail = {
                "index": idx,
                "question_tokens": token_len,
                "question_preview": question_text[:300],
            }
            detail_writer.write(json.dumps(detail, ensure_ascii=False) + "\n")

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} samples... current max tokens = {max_tokens}")

    if detail_writer:
        detail_writer.close()

    total = len(lengths)
    avg_len = sum(lengths) / total if total > 0 else 0

    print("\n========== Token Length Summary ==========")
    print(f"Total samples: {total}")
    print(f"Empty question samples: {empty_count}")
    print(f"Max question tokens: {max_tokens}")
    print(f"Max sample index: {max_index}")
    print(f"Average question tokens: {avg_len:.2f}")
    print(f"P50: {percentile(lengths, 0.50)}")
    print(f"P90: {percentile(lengths, 0.90)}")
    print(f"P95: {percentile(lengths, 0.95)}")
    print(f"P99: {percentile(lengths, 0.99)}")

    print("\n========== Max Token Sample Preview ==========")
    print(max_text[:2000])

    if args.save_detail:
        print(f"\nDetail saved to: {args.save_detail}")


if __name__ == "__main__":
    main()
