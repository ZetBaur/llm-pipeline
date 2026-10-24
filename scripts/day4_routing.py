import json
import re
from pathlib import Path

from llm_pipeline.llm_client import OpenAIChatClient
from llm_pipeline.pipeline import classify, respond, run_routed
from llm_pipeline.routing import ROUTES
from llm_pipeline.schemas import Classification
from llm_pipeline.utils import count_words, setup_logging

MARKERS = {
    "apology": r"извин|прости|сожале|sorry|apolog",
    "numbered_steps": r"(?m)^\s*\d+[.)]",
    "call_to_action": r"демо|demo|quote|коммерческ|записа|book a",
    "thanks": r"спасибо|благодар|thank",
}


def main() -> None:
    setup_logging("WARNING")
    client = OpenAIChatClient()
    items = json.loads(Path("examples/inputs.json").read_text(encoding="utf-8"))

    print("== classification on 10 texts ==")
    correct = 0
    rows = []
    for item in items:
        result = run_routed(item["text"], client)
        ok = result["category"] == item["expected_category"]
        correct += ok
        rows.append({"id": item["id"], "expected": item["expected_category"], **result})
        print(f"[{item['id']}] expected {item['expected_category']:16s} got {result['category']:16s} "
              f"{'OK' if ok else 'MISS'}  intent: {result['intent']}")
        print(f"      answer ({count_words(result['final_answer'])} words): {result['final_answer']}")
    print(f"accuracy: {correct}/{len(items)}")

    print("\n== does the category change the answer? same 3 texts, all 5 routes forced ==")
    by_route: dict[str, list[str]] = {name: [] for name in ROUTES}
    for item in items[:3]:
        base = classify(item["text"], client)
        for name in ROUTES:
            forced = Classification(category=name, intent=base.intent, sentiment=base.sentiment)
            by_route[name].append(respond(item["text"], forced, client))
    summary = {}
    for name, answers in by_route.items():
        words = [count_words(a) for a in answers]
        stats = {
            "mean_words": round(sum(words) / len(words), 1),
            "limit": ROUTES[name].max_words,
        }
        for marker, pattern in MARKERS.items():
            stats[marker] = f"{sum(bool(re.search(pattern, a, re.IGNORECASE)) for a in answers)}/{len(answers)}"
        summary[name] = stats
        print(f"{name:17s} {stats}")

    Path("outputs/day4_routing.json").write_text(
        json.dumps({"classification": rows, "accuracy": f"{correct}/{len(items)}",
                    "forced_routes": by_route, "marker_stats": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"tokens: {client.prompt_tokens} in, {client.completion_tokens} out, calls {client.calls}")


if __name__ == "__main__":
    main()
