import json
import logging
from pathlib import Path

from llm_pipeline.llm_client import Completion, OpenAIChatClient
from llm_pipeline.process import process_message
from llm_pipeline.utils import setup_logging


class Chaos:
    def __init__(self, client, mode: str) -> None:
        self.client = client
        self.mode = mode
        self.seen: set[str] = set()
        self.damaged = 0

    def complete(self, system, user, **kwargs):
        completion = self.client.complete(system, user, **kwargs)
        key = system[:40]
        if kwargs.get("json_mode") and key not in self.seen and "rejected" not in user:
            self.seen.add(key)
            self.damaged += 1
            if self.mode == "text_instead_of_json":
                text = "Конечно! Вот результат анализа: категория определена, детали ниже."
            else:
                data = json.loads(completion.text)
                text = json.dumps({k: v for k, v in list(data.items())[:-1]})
            return Completion(text, completion.prompt_tokens, completion.completion_tokens, completion.latency_s, completion.model)
        return completion


def main() -> None:
    setup_logging("WARNING")
    logging.getLogger("llm_pipeline").setLevel(logging.INFO)
    items = json.loads(Path("examples/inputs.json").read_text(encoding="utf-8"))
    report = {}

    for mode in ("text_instead_of_json", "missing_keys"):
        print(f"\n== scenario: {mode} (first JSON answer of every step is damaged) ==")
        chaos = Chaos(OpenAIChatClient(), mode)
        outcome = process_message(items[1]["text"], chaos)
        report[mode] = {"status": outcome.status, "damaged_answers": chaos.damaged, "answer": outcome.final_answer}
        print(f"status={outcome.status}, damaged answers={chaos.damaged}, answer: {outcome.final_answer}")

    print("\n== scenario: API unavailable (connection refused, 2 retries) ==")
    down = OpenAIChatClient(api_key="sk-test", base_url="http://127.0.0.1:9/v1", max_retries=2, backoff=0.3, timeout=2)
    outcome = process_message(items[0]["text"], down)
    report["api_down"] = outcome.model_dump(exclude={"result"})
    print(f"status={outcome.status}, error={outcome.error}\nanswer: {outcome.final_answer}")

    print("\n== scenario: invalid API key (must not be retried) ==")
    outcome = process_message(items[0]["text"], OpenAIChatClient(api_key="sk-invalid", max_retries=2, backoff=0.3))
    report["bad_key"] = outcome.model_dump(exclude={"result"})
    print(f"status={outcome.status}, error={outcome.error[:150]}")

    print("\n== scenario: empty text and huge text ==")
    client = OpenAIChatClient()
    print("empty:", process_message("   ", client).model_dump(exclude={"result"}))
    outcome = process_message("Не могу войти в аккаунт. " * 400, client)
    print("huge:", outcome.status, outcome.warnings[:1])

    print("\n== all 10 texts through process_message ==")
    statuses = {}
    for item in items:
        outcome = process_message(item["text"], client)
        statuses[item["id"]] = outcome.status
        print(f"[{item['id']}] {outcome.status:9s} warnings={len(outcome.warnings)} answer: {outcome.final_answer[:120]}")
    report["statuses"] = statuses
    print("statuses:", {s: list(statuses.values()).count(s) for s in set(statuses.values())})
    print(f"tokens: {client.prompt_tokens} in, {client.completion_tokens} out, calls {client.calls}")
    Path("outputs/day6_negative.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
