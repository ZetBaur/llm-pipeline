import pytest

from llm_pipeline.utils import count_words, extract_json, parse_plain_output, truncate_words

PLAIN = """SUMMARY: Customer cannot log in.
KEY POINTS:
- Login fails
1. Error shown
* iOS 18
ANSWER: We will check it."""


def test_parse_plain_output() -> None:
    parsed = parse_plain_output(PLAIN)

    assert parsed["summary"] == "Customer cannot log in."
    assert parsed["key_points"] == ["Login fails", "Error shown", "iOS 18"]
    assert parsed["final_answer"] == "We will check it."


@pytest.mark.parametrize("raw", ["", "   ", "just some text", "SUMMARY: only summary"])
def test_parse_plain_output_rejects_bad_format(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_plain_output(raw)


def test_extract_json_plain_fenced_and_wrapped() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": 1} hope it helps') == {"a": 1}


@pytest.mark.parametrize("raw", ["", "no json here", '{"a": 1,}', "[1, 2]", None])
def test_extract_json_rejects_garbage(raw) -> None:
    with pytest.raises(ValueError):
        extract_json(raw)


def test_words() -> None:
    assert count_words("a b  c") == 3
    assert truncate_words("one two three four", 2) == "one two..."
    assert truncate_words("one two", 5) == "one two"
