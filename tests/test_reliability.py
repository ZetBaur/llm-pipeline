import json

import httpx2 as httpx
import openai
import pytest

from llm_pipeline.guardrails import (
    MAX_INPUT_CHARS,
    guess_language,
    limit_reply,
    prepare_input,
    safe_reply,
)
from llm_pipeline.llm_client import LLMError, OpenAIChatClient
from llm_pipeline.process import process_message
from llm_pipeline.reliability import llm_json
from llm_pipeline.schemas import Classification

EXTRACT = json.dumps(
    {
        "summary": "Customer cannot log in.",
        "key_points": ["Login fails", "Error shown", "Tried reinstall"],
        "customer_requests": ["fix login"],
        "language": "English",
    }
)
CLASSIFY = json.dumps({"category": "support", "intent": "fix login", "sentiment": "negative"})
CHECK_OK = json.dumps({"unsupported_claims": [], "missed_details": [], "contradicts_message": False})
CHECK_CLAIM = json.dumps({"unsupported_claims": ["says it is possible"], "missed_details": []})
CHECK_MISSED = json.dumps({"unsupported_claims": [], "missed_details": ["asks for refund"]})


def test_text_instead_of_json_is_repaired(fake_client) -> None:
    fake_client.responses = ["Sure! The category is support.", CLASSIFY]

    result, _ = llm_json(fake_client, "sys", "user", Classification, max_tokens=100)

    assert result.category == "support"
    assert "rejected" in fake_client.calls[1]["user"] and "The category is support" in fake_client.calls[1]["user"]


def test_missing_keys_are_repaired(fake_client) -> None:
    fake_client.responses = [json.dumps({"category": "sales"}), CLASSIFY]

    result, _ = llm_json(fake_client, "sys", "user", Classification)

    assert result.intent == "fix login"
    assert "intent: field is missing" in fake_client.calls[1]["user"]


def test_pipeline_survives_broken_json_in_middle_step(fake_client) -> None:
    fake_client.responses = ["garbage", EXTRACT, CLASSIFY, "Please try again.", CHECK_OK]

    outcome = process_message("I cannot log in", fake_client)

    assert outcome.status == "ok"
    assert outcome.final_answer == "Please try again."


def test_persistently_broken_output_gives_safe_reply(fake_client) -> None:
    fake_client.responses = ["garbage", "more garbage"]

    outcome = process_message("I cannot log in", fake_client)

    assert outcome.status == "failed" and outcome.result is None
    assert outcome.final_answer.startswith("Thank you for your message")
    assert "invalid model output" in outcome.error


def test_api_failure_gives_safe_reply_in_customer_language(fake_client) -> None:
    fake_client.responses = [LLMError("API unavailable after 4 attempts")]

    outcome = process_message("Не могу войти в аккаунт", fake_client)

    assert outcome.status == "failed"
    assert outcome.final_answer.startswith("Спасибо за обращение")
    assert "unavailable" in outcome.error


def test_unsupported_claim_after_revision_is_replaced(fake_client) -> None:
    fake_client.responses = [EXTRACT, CLASSIFY, "Switch is possible.", CHECK_CLAIM, "It is possible.", CHECK_CLAIM]

    outcome = process_message("Can I switch plans?", fake_client)

    assert outcome.status == "degraded"
    assert outcome.final_answer.startswith("Thank you for your message")
    assert any("safe reply" in w for w in outcome.warnings)
    assert outcome.result is not None and outcome.result.revised


def test_missed_detail_keeps_reply_but_marks_degraded(fake_client) -> None:
    fake_client.responses = [EXTRACT, CLASSIFY, "A", CHECK_MISSED, "B", CHECK_MISSED]

    outcome = process_message("I want a refund", fake_client)

    assert outcome.status == "degraded"
    assert outcome.final_answer == "B"


@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_input_is_rejected_without_api_calls(fake_client, text) -> None:
    outcome = process_message(text, fake_client)

    assert outcome.status == "failed" and "empty" in outcome.error
    assert fake_client.calls == []


def test_long_input_is_truncated_with_warning() -> None:
    cleaned, warnings = prepare_input("word " * 5000)

    assert len(cleaned) == MAX_INPUT_CHARS
    assert warnings


def test_limit_reply_cuts_at_sentence_boundary() -> None:
    answer = "First sentence is here. " * 30

    clipped, was_clipped = limit_reply(answer, 20)

    assert was_clipped and clipped.endswith(".") and len(clipped.split()) <= 26
    assert limit_reply("Short reply.", 20) == ("Short reply.", False)


def test_language_guess() -> None:
    assert guess_language("привет") == "ru"
    assert guess_language("hello") == "en"
    assert guess_language("hello", "Russian") == "ru"
    assert safe_reply("hello", "German").startswith("Thank you")


class Stub:
    def __init__(self, behaviours):
        self.behaviours = list(behaviours)
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        item = self.behaviours.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class Response:
    class Choice:
        class Message:
            content = '{"ok": true}'

        message = Message()

    class Usage:
        prompt_tokens = 7
        completion_tokens = 3

    choices = [Choice()]
    usage = Usage()
    model = "stub-model"


def make_client(behaviours, max_retries=3) -> tuple[OpenAIChatClient, list[float]]:
    sleeps: list[float] = []
    client = OpenAIChatClient(api_key="test", max_retries=max_retries, backoff=1.0, sleep=sleeps.append)
    client._client = Stub(behaviours)
    return client, sleeps


def connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx.Request("POST", "http://x"))


def test_client_retries_temporary_errors_with_backoff() -> None:
    client, sleeps = make_client([connection_error(), connection_error(), Response()])

    completion = client.complete("s", "u")

    assert completion.text == '{"ok": true}'
    assert sleeps == [1.0, 2.0]
    assert client.calls == 1 and client.prompt_tokens == 7


def test_client_gives_up_after_max_retries() -> None:
    client, sleeps = make_client([connection_error()] * 3, max_retries=2)

    with pytest.raises(LLMError, match="after 3 attempts"):
        client.complete("s", "u")

    assert sleeps == [1.0, 2.0]


def test_client_does_not_retry_bad_requests() -> None:
    request = httpx.Request("POST", "http://x")
    error = openai.BadRequestError("bad", response=httpx.Response(400, request=request), body=None)
    client, sleeps = make_client([error, Response()])

    with pytest.raises(LLMError, match="BadRequestError"):
        client.complete("s", "u")

    assert sleeps == []


def test_missing_api_key_is_a_clear_error(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        OpenAIChatClient()
