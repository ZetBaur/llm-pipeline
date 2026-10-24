import json

import pytest

from llm_pipeline.pipeline import analyze_json, validate_analysis
from llm_pipeline.schemas import OutputValidationError

GOOD = {
    "summary": "Customer cannot log in.",
    "category": "support",
    "sentiment": "negative",
    "key_points": ["Login fails", "Error shown", "Tried reinstall"],
    "final_answer": "We will check it.",
}


def raw(**changes) -> str:
    data = {**GOOD, **changes}
    return json.dumps({k: v for k, v in data.items() if v is not ...})


def test_valid_response_is_parsed() -> None:
    analysis = validate_analysis(raw())

    assert analysis.category == "support"
    assert analysis.key_points[0] == "Login fails"


def test_fenced_json_is_accepted() -> None:
    assert validate_analysis("```json\n" + raw() + "\n```").sentiment == "negative"


def test_extra_fields_are_ignored() -> None:
    assert validate_analysis(raw(extra="x")).summary


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("", "empty"),
        ("Sorry, I cannot do that", "not JSON"),
        ('{"summary": "x",}', "broken JSON"),
        ("[1, 2, 3]", "must be an object"),
        (raw(summary=...), "summary: field is missing"),
        (raw(category="spam"), "category"),
        (raw(sentiment=5), "sentiment"),
        (raw(key_points=["a", "b"]), "key_points"),
        (raw(key_points="one, two, three"), "key_points"),
        (raw(key_points=["a", " ", "c"]), "must not be empty"),
        (raw(final_answer=""), "final_answer"),
    ],
)
def test_bad_responses_give_readable_error(response: str, expected: str) -> None:
    with pytest.raises(OutputValidationError) as info:
        validate_analysis(response)

    assert expected in str(info.value)
    assert info.value.problems


def test_error_lists_all_problems_at_once() -> None:
    with pytest.raises(OutputValidationError) as info:
        validate_analysis(raw(category="spam", key_points=["a"], summary=...))

    assert len(info.value.problems) == 3


def test_analyze_json_uses_json_mode(fake_client) -> None:
    fake_client.responses = [raw()]

    analysis = analyze_json("I cannot log in", fake_client)

    assert analysis.category == "support"
    assert fake_client.calls[0]["json_mode"] is True


def test_analyze_json_empty_input(fake_client) -> None:
    with pytest.raises(ValueError):
        analyze_json("", fake_client)
