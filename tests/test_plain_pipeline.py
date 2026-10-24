import pytest

from llm_pipeline.pipeline import run_plain
from tests.test_utils import PLAIN


def test_run_plain_returns_parsed_fields(fake_client) -> None:
    fake_client.responses = [PLAIN]

    result = run_plain("I cannot log in", fake_client)

    assert result["summary"] == "Customer cannot log in."
    assert len(result["key_points"]) == 3
    assert "I cannot log in" in fake_client.calls[0]["user"]
    assert "exactly 3" in fake_client.calls[0]["system"]


def test_run_plain_rejects_empty_text(fake_client) -> None:
    with pytest.raises(ValueError):
        run_plain("   ", fake_client)
    assert fake_client.calls == []


def test_run_plain_rejects_unparseable_answer(fake_client) -> None:
    fake_client.responses = ["I'm sorry, I can't help with that."]

    with pytest.raises(ValueError):
        run_plain("hello", fake_client)
