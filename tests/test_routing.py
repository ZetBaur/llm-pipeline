import json

import pytest

from llm_pipeline.pipeline import classify, respond, run_routed
from llm_pipeline.routing import ROUTES, get_route
from llm_pipeline.schemas import CATEGORIES, Classification, OutputValidationError


def cls(category: str = "complaint") -> str:
    return json.dumps({"category": category, "intent": "get a refund", "sentiment": "negative"})


def test_every_category_has_a_route() -> None:
    assert set(ROUTES) == set(CATEGORIES)
    assert all(route.instruction and route.max_words > 0 for route in ROUTES.values())


def test_unknown_category_has_no_route() -> None:
    with pytest.raises(ValueError):
        get_route("spam")


def test_classify_parses_result(fake_client) -> None:
    fake_client.responses = [cls("sales")]

    result = classify("how much is the plan", fake_client)

    assert result == Classification(category="sales", intent="get a refund", sentiment="negative")
    assert fake_client.calls[0]["json_mode"] is True


@pytest.mark.parametrize(
    "response", ["", "nonsense", cls("spam"), json.dumps({"category": "sales"})]
)
def test_classify_rejects_bad_output(fake_client, response: str) -> None:
    fake_client.responses = [response, response]

    with pytest.raises(OutputValidationError):
        classify("hello", fake_client)


@pytest.mark.parametrize("category", CATEGORIES)
def test_respond_uses_instruction_of_category(fake_client, category: str) -> None:
    fake_client.responses = ["Reply."]
    classification = Classification(category=category, intent="x", sentiment="neutral")

    assert respond("text", classification, fake_client) == "Reply."

    system = fake_client.calls[0]["system"]
    assert ROUTES[category].instruction in system
    assert f"at most {ROUTES[category].max_words} words" in system
    for other, route in ROUTES.items():
        if other != category:
            assert route.instruction not in system


def test_respond_rejects_empty_reply(fake_client) -> None:
    fake_client.responses = ["   "]

    with pytest.raises(OutputValidationError):
        respond("text", Classification(category="sales", intent="x", sentiment="neutral"), fake_client)


def test_run_routed_chains_two_calls(fake_client) -> None:
    fake_client.responses = [cls("complaint"), "We are sorry."]

    result = run_routed("my order is late", fake_client)

    assert result["category"] == "complaint"
    assert result["final_answer"] == "We are sorry."
    assert len(fake_client.calls) == 2
    assert ROUTES["complaint"].instruction in fake_client.calls[1]["system"]
