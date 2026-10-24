import logging

from pydantic import BaseModel, ValidationError

from llm_pipeline.llm_client import ChatClient, Completion
from llm_pipeline.schemas import OutputValidationError, describe_errors
from llm_pipeline.utils import extract_json

logger = logging.getLogger(__name__)

REPAIR_SUFFIX = (
    "\n\nYour previous reply was rejected: {problems}\nPrevious reply:\n{raw}\n"
    "Return ONE corrected JSON object only, with all required fields and valid values."
)


def parse_model(model_cls: type[BaseModel], completion: Completion):
    try:
        return model_cls.model_validate(extract_json(completion.text))
    except ValidationError as exc:
        raise OutputValidationError(describe_errors(exc), raw=completion.text) from exc
    except ValueError as exc:
        raise OutputValidationError([str(exc)], raw=completion.text) from exc


def llm_json(
    client: ChatClient,
    system: str,
    user: str,
    model_cls: type[BaseModel],
    max_tokens: int = 400,
    repairs: int = 1,
):
    completion = client.complete(system, user, json_mode=True, max_tokens=max_tokens)
    for attempt in range(repairs + 1):
        try:
            return parse_model(model_cls, completion), completion
        except OutputValidationError as exc:
            if attempt == repairs:
                raise
            logger.warning("invalid model output (%s); asking the model to repair it", exc)
            repair = user + REPAIR_SUFFIX.format(problems="; ".join(exc.problems), raw=exc.raw[:600])
            completion = client.complete(system, repair, json_mode=True, max_tokens=max_tokens)
    raise AssertionError("unreachable")
