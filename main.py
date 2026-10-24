import argparse
import json
import logging
from pathlib import Path

from llm_pipeline.llm_client import OpenAIChatClient
from llm_pipeline.pipeline import analyze_json, run_plain
from llm_pipeline.schemas import OutputValidationError
from llm_pipeline.utils import setup_logging

logger = logging.getLogger("main")


def load_inputs(path: Path, limit: int | None) -> list[dict]:
    items = json.loads(path.read_text(encoding="utf-8"))
    return items[:limit] if limit else items


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM pipeline for customer messages")
    parser.add_argument("--input", type=Path, default=Path("examples/inputs.json"))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--mode", choices=["plain", "json"], default="json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    setup_logging()
    client = OpenAIChatClient()
    output = args.output or Path(f"outputs/{args.mode}.json")
    results = []
    for item in load_inputs(args.input, args.limit):
        print(f"\n[{item['id']}] {item['text']}")
        try:
            if args.mode == "plain":
                result = run_plain(item["text"], client)
                result.pop("raw", None)
            else:
                result = analyze_json(item["text"], client).model_dump()
        except OutputValidationError as exc:
            print(f"  INVALID RESPONSE: {exc}")
            results.append({"id": item["id"], "error": exc.problems})
            continue
        results.append({"id": item["id"], "text": item["text"], **result})
        for key, value in result.items():
            print(f"  {key}: {value}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {output}; tokens: {client.prompt_tokens} in, {client.completion_tokens} out")


if __name__ == "__main__":
    main()
