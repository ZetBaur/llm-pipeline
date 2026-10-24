import logging

from llm_pipeline.chain import run_chain
from llm_pipeline.guardrails import prepare_input, safe_reply
from llm_pipeline.llm_client import ChatClient, LLMError
from llm_pipeline.schemas import Outcome, OutputValidationError

logger = logging.getLogger(__name__)


def process_message(text: str | None, client: ChatClient) -> Outcome:
    try:
        cleaned, warnings = prepare_input(text)
    except ValueError as exc:
        logger.warning("rejected input: %s", exc)
        return Outcome(status="failed", final_answer="", error=str(exc))
    for warning in warnings:
        logger.warning(warning)

    try:
        result = run_chain(cleaned, client)
    except LLMError as exc:
        logger.error("API failure, using safe reply: %s", exc)
        return Outcome(
            status="failed",
            final_answer=safe_reply(cleaned),
            error=f"LLM unavailable: {exc}",
            warnings=warnings,
        )
    except OutputValidationError as exc:
        logger.error("model output stayed invalid after repair, using safe reply: %s", exc)
        return Outcome(
            status="failed",
            final_answer=safe_reply(cleaned),
            error=f"invalid model output: {exc}",
            warnings=warnings,
        )

    if result.blocking_issues:
        logger.warning("reply still contains unsupported claims after revision, using safe reply")
        warnings.append("draft replaced by the safe reply: " + "; ".join(result.remaining_issues))
        result = result.model_copy(update={"final_answer": safe_reply(cleaned, result.language)})
        return Outcome(status="degraded", final_answer=result.final_answer, result=result, warnings=warnings)

    if not result.self_check_passed:
        warnings.append("self-check still has remarks: " + "; ".join(result.remaining_issues))
        return Outcome(status="degraded", final_answer=result.final_answer, result=result, warnings=warnings)
    return Outcome(status="ok", final_answer=result.final_answer, result=result, warnings=warnings)
