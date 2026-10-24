import logging
import time

from llm_pipeline.guardrails import limit_reply
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
from llm_pipeline.reliability import llm_json
from llm_pipeline.routing import get_route
from llm_pipeline.schemas import (
    ChainResult,
    Classification,
    ExtractResult,
    OutputValidationError,
    SelfCheck,
    StepRecord,
)

logger = logging.getLogger(__name__)
TOTAL_STEPS = 5


def _record(name: str, completion: Completion | None, started: float, note: str = "") -> StepRecord:
    return StepRecord(
        name=name,
        seconds=round(time.time() - started, 2),
        prompt_tokens=completion.prompt_tokens if completion else 0,
        completion_tokens=completion.completion_tokens if completion else 0,
        note=note,
    )


def extract_step(text: str, client: ChatClient) -> tuple[ExtractResult, Completion]:
    return llm_json(client, EXTRACT_SYSTEM, build_user_prompt(text), ExtractResult, max_tokens=400)


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
    return llm_json(client, SELF_CHECK_SYSTEM, user, SelfCheck, max_tokens=400)


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
    answer, clipped = limit_reply(answer, get_route(classification.category).max_words)
    steps.append(_record("respond", completion, started, f"{len(answer.split())} words" + (", clipped" if clipped else "")))
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
        answer, _ = limit_reply(answer, get_route(classification.category).max_words)
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
        remaining_issues=[] if check.ok else check.issues(),
        blocking_issues=bool(check.unsupported_claims or check.contradicts_message),
        steps=steps,
    )
