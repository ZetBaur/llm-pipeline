import logging

from pydantic import ValidationError

from llm_pipeline.llm_client import ChatClient
from llm_pipeline.prompts import (
    BEST_PLAIN_VARIANT,
    CLASSIFY_SYSTEM,
    JSON_SYSTEM,
    NO_INVENTION_RULE,
    PLAIN_VARIANTS,
    RESPOND_SYSTEM,
    build_user_prompt,
)
from llm_pipeline.routing import get_route
from llm_pipeline.schemas import (
    Analysis,
    Classification,
    OutputValidationError,
    describe_errors,
)
from llm_pipeline.utils import extract_json, parse_plain_output

logger = logging.getLogger(__name__)


def run_plain(text: str, client: ChatClient, variant: str = BEST_PLAIN_VARIANT) -> dict:
    if not text or not text.strip():
        raise ValueError("Input text is empty")
    system = PLAIN_VARIANTS[variant]
    completion = client.complete(system, build_user_prompt(text), max_tokens=400)
    logger.info(
        "plain/%s: %d prompt + %d completion tokens, %.1fs",
        variant,
        completion.prompt_tokens,
        completion.completion_tokens,
        completion.latency_s,
    )
    result = parse_plain_output(completion.text)
    result["raw"] = completion.text
    return result


def validate_analysis(raw: str) -> Analysis:
    try:
        data = extract_json(raw)
    except ValueError as exc:
        raise OutputValidationError([str(exc)], raw=raw or "") from exc
    try:
        return Analysis.model_validate(data)
    except ValidationError as exc:
        raise OutputValidationError(describe_errors(exc), raw=raw) from exc


def analyze_json(text: str, client: ChatClient) -> Analysis:
    if not text or not text.strip():
        raise ValueError("Input text is empty")
    completion = client.complete(JSON_SYSTEM, build_user_prompt(text), json_mode=True, max_tokens=500)
    logger.info(
        "json: %d prompt + %d completion tokens, %.1fs",
        completion.prompt_tokens,
        completion.completion_tokens,
        completion.latency_s,
    )
    analysis = validate_analysis(completion.text)
    logger.info("json: valid, category=%s sentiment=%s", analysis.category, analysis.sentiment)
    return analysis


def classify(text: str, client: ChatClient) -> Classification:
    if not text or not text.strip():
        raise ValueError("Input text is empty")
    completion = client.complete(CLASSIFY_SYSTEM, build_user_prompt(text), json_mode=True, max_tokens=150)
    try:
        data = extract_json(completion.text)
        result = Classification.model_validate(data)
    except ValueError as exc:
        if isinstance(exc, ValidationError):
            raise OutputValidationError(describe_errors(exc), raw=completion.text) from exc
        raise OutputValidationError([str(exc)], raw=completion.text) from exc
    logger.info("classify: category=%s intent=%r sentiment=%s", result.category, result.intent, result.sentiment)
    return result


def respond(text: str, classification: Classification, client: ChatClient) -> str:
    route = get_route(classification.category)
    system = RESPOND_SYSTEM.format(
        rule=NO_INVENTION_RULE,
        category=classification.category,
        intent=classification.intent,
        instruction=route.instruction,
        max_words=route.max_words,
    )
    completion = client.complete(system, build_user_prompt(text), temperature=0.0, max_tokens=300)
    answer = completion.text.strip()
    if not answer:
        raise OutputValidationError(["final_answer: model returned an empty reply"], raw="")
    logger.info("respond[%s]: %d words", route.category, len(answer.split()))
    return answer


def run_routed(text: str, client: ChatClient) -> dict:
    classification = classify(text, client)
    answer = respond(text, classification, client)
    return {**classification.model_dump(), "final_answer": answer}
