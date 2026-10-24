import argparse
import json
import logging
from pathlib import Path

from llm_pipeline.llm_client import LLMError, OpenAIChatClient
from llm_pipeline.process import process_message
from llm_pipeline.utils import setup_logging

logger = logging.getLogger("main")


def load_inputs(path: Path, limit: int | None) -> list[dict]:
    items = json.loads(path.read_text(encoding="utf-8"))
    return items[:limit] if limit else items


def print_outcome(outcome) -> None:
    print(f"  status:   {outcome.status}")
    if outcome.result:
        r = outcome.result
        print(f"  category: {r.category} | intent: {r.intent} | sentiment: {r.sentiment}")
        print(f"  summary:  {r.summary}")
        for point in r.key_points:
            print(f"    - {point}")
    print(f"  answer:   {outcome.final_answer}")
    if outcome.error:
        print(f"  error:    {outcome.error}")
    for warning in outcome.warnings:
        print(f"  warning:  {warning}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM pipeline for customer messages")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="one message to process")
    source.add_argument("--input", type=Path, default=Path("examples/demo_scenarios.json"),
                        help="JSON file with a list of {id, text}")
    parser.add_argument("--limit", type=int, help="process only the first N items of the file")
    parser.add_argument("--output", type=Path, default=Path("outputs/result.json"))
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(args.log_level)
    logging.getLogger("llm_pipeline").setLevel(args.log_level.upper())
    try:
        client = OpenAIChatClient()
    except LLMError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    items = [{"id": "text", "text": args.text}] if args.text else load_inputs(args.input, args.limit)
    results = []
    for item in items:
        print(f"\n[{item['id']}] {item.get('title', '')}\n  input: {item['text'].strip()[:200]}")
        outcome = process_message(item["text"], client)
        print_outcome(outcome)
        results.append({"id": item["id"], **outcome.model_dump()})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {args.output}; {client.calls} API calls, "
          f"{client.prompt_tokens} prompt + {client.completion_tokens} completion tokens")


if __name__ == "__main__":
    main()
