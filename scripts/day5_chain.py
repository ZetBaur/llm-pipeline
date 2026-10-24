import json
from pathlib import Path

from llm_pipeline.chain import run_chain, self_check_step
from llm_pipeline.llm_client import OpenAIChatClient
from llm_pipeline.utils import setup_logging

PLANTED = [
    (
        "Подскажите, сколько стоит корпоративный тариф на 50 сотрудников и можно ли получить скидку при оплате на год вперёд?",
        "Корпоративный тариф на 50 сотрудников стоит 1200 долларов в месяц, а при оплате за год скидка составляет 20%.",
        "invented price and discount",
    ),
    (
        "Заказ №48213 привезли на неделю позже, коробка была помята, а курьер ещё и нагрубил. Требую вернуть деньги за доставку.",
        "Спасибо за обращение! Мы рады, что заказ дошёл до вас, и хотим предложить подписку на нашу рассылку.",
        "ignores the complaint and the refund demand",
    ),
    (
        "Hi, the export to CSV fails with error 500 whenever the file has more than 10,000 rows. Is there a workaround?",
        "Exporting works fine for files of any size, so no workaround is needed. Please check that you are using the latest browser.",
        "contradicts the message",
    ),
]


def main() -> None:
    setup_logging("INFO")
    client = OpenAIChatClient()
    items = json.loads(Path("examples/inputs.json").read_text(encoding="utf-8"))

    print("== chain on 10 texts ==")
    rows = []
    for item in items:
        result = run_chain(item["text"], client)
        rows.append({"id": item["id"], "expected": item["expected_category"], **result.model_dump()})
        flag = "revised" if result.revised else "first draft ok"
        print(f"[{item['id']}] {result.category:16s} {flag}; issues: {result.issues_found}")
        print(f"      final ({len(result.final_answer.split())} words): {result.final_answer[:230]}")
    revised = sum(r["revised"] for r in rows)
    passed = sum(r["self_check_passed"] for r in rows)
    print(f"revised {revised}/{len(rows)}, final check passed {passed}/{len(rows)}")

    print("\n== does the self-check catch planted bad replies? ==")
    detected = 0
    planted_rows = []
    for text, bad_answer, why in PLANTED:
        check, _ = self_check_step(text, bad_answer, client)
        detected += not check.ok
        planted_rows.append({"why": why, "ok": check.ok, "issues": check.issues()})
        print(f"{why}: flagged={not check.ok} {check.issues()}")
    print(f"detected {detected}/{len(PLANTED)}")

    Path("outputs/day5_chain.json").write_text(
        json.dumps({"chain": rows, "planted": planted_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"tokens: {client.prompt_tokens} in, {client.completion_tokens} out, calls {client.calls}")


if __name__ == "__main__":
    main()
