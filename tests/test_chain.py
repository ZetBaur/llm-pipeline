import json

import pytest

from llm_pipeline.chain import run_chain, self_check_step
from llm_pipeline.schemas import OutputValidationError, SelfCheck

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
CHECK_BAD = json.dumps(
    {
        "unsupported_claims": ["says the fix ships tomorrow"],
        "missed_details": [],
        "contradicts_message": False,
    }
)


def test_chain_without_issues_runs_four_llm_calls(fake_client) -> None:
    fake_client.responses = [EXTRACT, CLASSIFY, "1. Try again.", CHECK_OK]

    result = run_chain("I cannot log in", fake_client)

    assert [s.name for s in result.steps] == ["extract", "classify", "structure", "respond", "self_check"]
    assert result.self_check_passed and not result.revised
    assert result.category == "support" and result.final_answer == "1. Try again."
    assert len(fake_client.calls) == 4


def test_each_step_uses_result_of_previous_step(fake_client) -> None:
    fake_client.responses = [EXTRACT, CLASSIFY, "Reply", CHECK_OK]

    run_chain("I cannot log in", fake_client)

    respond_system = fake_client.calls[2]["system"]
    assert "Customer cannot log in." in respond_system
    assert "fix login" in respond_system
    assert "category: support" in respond_system
    assert "Reply" in fake_client.calls[3]["user"]


def test_failed_check_triggers_one_revision(fake_client) -> None:
    fake_client.responses = [EXTRACT, CLASSIFY, "Fix ships tomorrow.", CHECK_BAD, "We will check.", CHECK_OK]

    result = run_chain("I cannot log in", fake_client)

    assert result.revised and result.self_check_passed
    assert result.final_answer == "We will check."
    assert result.issues_found == ["unsupported claim: says the fix ships tomorrow"]
    revise_system = fake_client.calls[4]["system"]
    assert "says the fix ships tomorrow" in revise_system and "Fix ships tomorrow." in revise_system
    assert [s.name for s in result.steps][-2:] == ["revise", "self_check_2"]


def test_revision_that_still_fails_is_reported(fake_client) -> None:
    fake_client.responses = [EXTRACT, CLASSIFY, "A", CHECK_BAD, "B", CHECK_BAD]

    result = run_chain("I cannot log in", fake_client)

    assert result.revised and not result.self_check_passed
    assert len(fake_client.calls) == 6


def test_bad_extract_output_stops_the_chain(fake_client) -> None:
    fake_client.responses = ["not json"]

    with pytest.raises(OutputValidationError):
        run_chain("hello", fake_client)


def test_empty_input_makes_no_calls(fake_client) -> None:
    with pytest.raises(ValueError):
        run_chain("  ", fake_client)
    assert fake_client.calls == []


def test_self_check_properties() -> None:
    assert SelfCheck().ok
    assert not SelfCheck(contradicts_message=True).ok
    check = SelfCheck(unsupported_claims=["a"], missed_details=["b"], contradicts_message=True)
    assert len(check.issues()) == 3


def test_self_check_step_rejects_garbage(fake_client) -> None:
    fake_client.responses = ["looks fine to me"]

    with pytest.raises(OutputValidationError):
        self_check_step("text", "answer", fake_client)
