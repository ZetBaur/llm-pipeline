import argparse
import json
import logging
from pathlib import Path

from llm_pipeline.llm_client import OpenAIChatClient
from llm_pipeline.pipeline import run_plain
from llm_pipeline.utils import setup_logging

logger = logging.getLogger("main")


def load_inputs(path: Path, limit: int | None) -> list[dict]:
    items = json.loads(path.read_text(encoding="utf-8"))
    return items[:limit] if limit else items


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM pipeline for customer messages")
    parser.add_argument("--input", type=Path, default=Path("examples/inputs.json"))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("outputs/day1.json"))
    args = parser.parse_args()

    setup_logging()
    client = OpenAIChatClient()
    results = []
    for item in load_inputs(args.input, args.limit):
        result = run_plain(item["text"], client)
        results.append({"id": item["id"], "text": item["text"], **result})
        print(f"\n[{item['id']}] {item['text']}")
        print(f"  summary: {result['summary']}")
        for point in result["key_points"]:
            print(f"  - {point}")
        print(f"  answer: {result['final_answer']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {args.output}; tokens: {client.prompt_tokens} in, {client.completion_tokens} out")


if __name__ == "__main__":
    main()
