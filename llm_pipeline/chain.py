import logging
import time

from pydantic import ValidationError

from llm_pipeline.llm_client import ChatClient, Completion
from llm_pipeline.pipeline import classify
from llm_pipeline.prompts import (
    CHAIN_CONTEXT,
    EXTRACT_SYSTEM,
    NO_INVENTION_RULE,
    RESPOND_SYSTEM,
    REVISE_SUFFIX,
    SELF_CHECK_SYSTEM,
    build_user_prompt,
)
from llm_pipeline.routing import get_route
from llm_pipeline.schemas import (
    ChainResult,
    Classification,
    ExtractResult,
    OutputValidationError,
    SelfCheck,
    StepRecord,
    describe_errors,
)
from llm_pipeline.utils import extract_json

logger = logging.getLogger(__name__)
TOTAL_STEPS = 5


def _parse(model_cls, completion: Completion):
    try:
        return model_cls.model_validate(extract_json(completion.text))
    except ValidationError as exc:
        raise OutputValidationError(describe_errors(exc), raw=completion.text) from exc
    except ValueError as exc:
        raise OutputValidationError([str(exc)], raw=completion.text) from exc


def _record(name: str, completion: Completion | None, started: float, note: str = "") -> StepRecord:
    return StepRecord(
        name=name,
        seconds=round(time.time() - started, 2),
        prompt_tokens=completion.prompt_tokens if completion else 0,
        completion_tokens=completion.completion_tokens if completion else 0,
        note=note,
    )


def extract_step(text: str, client: ChatClient) -> tuple[ExtractResult, Completion]:
    completion = client.complete(EXTRACT_SYSTEM, build_user_prompt(text), json_mode=True, max_tokens=400)
    return _parse(ExtractResult, completion), completion


def build_context(extract: ExtractResult) -> str:
    return CHAIN_CONTEXT.format(
        summary=extract.summary,
        requests="; ".join(extract.customer_requests),
        points="; ".join(extract.key_points),
    )


def respond_step(
    text: str,
    classification: Classification,
    extract: ExtractResult,
    client: ChatClient,
    problems: list[str] | None = None,
    draft: str = "",
) -> tuple[str, Completion]:
    route = get_route(classification.category)
    system = RESPOND_SYSTEM.format(
        rule=NO_INVENTION_RULE,
        category=classification.category,
        intent=classification.intent,
        instruction=route.instruction,
        max_words=route.max_words,
    )
    system += "\n" + build_context(extract)
    if problems:
        system += REVISE_SUFFIX.format(issues="\n".join(f"- {p}" for p in problems), draft=draft)
    completion = client.complete(system, build_user_prompt(text), max_tokens=300)
    answer = completion.text.strip()
    if not answer:
        raise OutputValidationError(["final_answer: model returned an empty reply"])
    return answer, completion


def self_check_step(text: str, answer: str, client: ChatClient) -> tuple[SelfCheck, Completion]:
    user = f"Customer message:\n\"\"\"\n{text.strip()}\n\"\"\"\n\nDraft reply:\n\"\"\"\n{answer}\n\"\"\""
    completion = client.complete(SELF_CHECK_SYSTEM, user, json_mode=True, max_tokens=400)
    return _parse(SelfCheck, completion), completion


def run_chain(text: str, client: ChatClient) -> ChainResult:
    if not text or not text.strip():
        raise ValueError("Input text is empty")
    steps: list[StepRecord] = []

    started = time.time()
    extract, completion = extract_step(text, client)
    steps.append(_record("extract", completion, started, f"{len(extract.customer_requests)} requests"))
    logger.info("step 1/%d extract: %s", TOTAL_STEPS, extract.summary)

    started = time.time()
    classification = classify(text, client)
    steps.append(_record("classify", None, started, f"{classification.category} / {classification.intent}"))
    logger.info("step 2/%d classify: %s, intent=%r", TOTAL_STEPS, classification.category, classification.intent)

    started = time.time()
    structured = {
        "summary": extract.summary,
        "key_points": extract.key_points,
        "category": classification.category,
        "sentiment": classification.sentiment,
    }
    steps.append(_record("structure", None, started, "merged extract and classification"))
    logger.info("step 3/%d structure: %s", TOTAL_STEPS, structured["category"])

    started = time.time()
    answer, completion = respond_step(text, classification, extract, client)
    steps.append(_record("respond", completion, started, f"{len(answer.split())} words"))
    logger.info("step 4/%d respond[%s]: %d words", TOTAL_STEPS, classification.category, len(answer.split()))

    started = time.time()
    check, completion = self_check_step(text, answer, client)
    steps.append(_record("self_check", completion, started, "ok" if check.ok else f"{len(check.issues())} issue(s)"))
    logger.info("step 5/%d self-check: %s", TOTAL_STEPS, "ok" if check.ok else check.issues())

    issues_found = check.issues()
    revised = False
    if not check.ok:
        started = time.time()
        answer, completion = respond_step(text, classification, extract, client, issues_found, answer)
        steps.append(_record("revise", completion, started, f"{len(answer.split())} words"))
        check, completion = self_check_step(text, answer, client)
        steps.append(_record("self_check_2", completion, started, "ok" if check.ok else "still issues"))
        revised = True
        logger.info("revision done, second check: %s", "ok" if check.ok else check.issues())

    return ChainResult(
        **structured,
        final_answer=answer,
        intent=classification.intent,
        language=extract.language,
        customer_requests=extract.customer_requests,
        self_check_passed=check.ok,
        revised=revised,
        issues_found=issues_found,
        steps=steps,
    )
