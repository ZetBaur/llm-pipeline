import logging

from llm_pipeline.llm_client import ChatClient
from llm_pipeline.prompts import BEST_PLAIN_VARIANT, PLAIN_VARIANTS, build_user_prompt
from llm_pipeline.utils import parse_plain_output

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
