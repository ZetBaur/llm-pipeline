import json
import logging
from pathlib import Path

from llm_pipeline.llm_client import OpenAIChatClient
from llm_pipeline.pipeline import run_plain
from llm_pipeline.prompts import PLAIN_VARIANTS
from llm_pipeline.utils import count_words, setup_logging

MAX_SUMMARY_WORDS = 25
MAX_POINT_WORDS = 12
MAX_ANSWER_WORDS = 40


def check(result: dict) -> dict:
    return {
        "summary_ok": count_words(result["summary"]) <= MAX_SUMMARY_WORDS,
        "three_points": len(result["key_points"]) == 3,
        "points_short": all(count_words(p) <= MAX_POINT_WORDS for p in result["key_points"]),
        "answer_ok": count_words(result["final_answer"]) <= MAX_ANSWER_WORDS,
    }


def main() -> None:
    setup_logging("WARNING")
    logging.getLogger("llm_pipeline").setLevel(logging.WARNING)
    client = OpenAIChatClient()
    items = json.loads(Path("examples/inputs.json").read_text(encoding="utf-8"))[:5]
    report = {}
    for name in PLAIN_VARIANTS:
        rows = []
        for item in items:
            try:
                result = run_plain(item["text"], client, variant=name)
                flags = check(result)
                flags["parsed"] = True
                flags["words"] = {
                    "summary": count_words(result["summary"]),
                    "answer": count_words(result["final_answer"]),
                    "points": len(result["key_points"]),
                }
            except ValueError as exc:
                flags = {"parsed": False, "error": str(exc)}
            rows.append({"id": item["id"], **flags})
        passed = sum(all(v for k, v in r.items() if k in {"parsed", "summary_ok", "three_points", "points_short", "answer_ok"}) for r in rows)
        report[name] = {"fully_compliant": f"{passed}/{len(rows)}", "rows": rows}
        print(f"{name}: fully compliant {passed}/{len(rows)}")
        for r in rows:
            print("   ", r)
    out = Path("outputs/day2_prompt_comparison.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"tokens: {client.prompt_tokens} in, {client.completion_tokens} out")


if __name__ == "__main__":
    main()
