import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import openai
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4.1-mini"
RETRYABLE = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


class LLMError(Exception):
    pass


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    model: str


class ChatClient(Protocol):
    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ) -> Completion: ...


class OpenAIChatClient:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        timeout: float = 30.0,
        backoff: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        load_dotenv(".env")
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise LLMError("OPENAI_API_KEY is not set: put it into .env (see .env.example)")
        self.model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        self.max_retries = max_retries
        self.backoff = backoff
        self.sleep = sleep
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._client = openai.OpenAI(api_key=key, timeout=timeout, max_retries=0)

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ) -> Completion:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "seed": 42,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(1, self.max_retries + 2):
            started = time.time()
            try:
                response = self._client.chat.completions.create(**kwargs)
            except RETRYABLE as exc:
                if attempt > self.max_retries:
                    raise LLMError(f"API unavailable after {attempt} attempts: {type(exc).__name__}") from exc
                delay = self.backoff * 2 ** (attempt - 1)
                logger.warning(
                    "API error %s (attempt %d/%d), retry in %.1fs",
                    type(exc).__name__,
                    attempt,
                    self.max_retries + 1,
                    delay,
                )
                self.sleep(delay)
                continue
            except openai.OpenAIError as exc:
                raise LLMError(f"API call failed: {type(exc).__name__}: {exc}") from exc

            usage = response.usage
            self.calls += 1
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
            text = response.choices[0].message.content or ""
            return Completion(
                text=text,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                latency_s=time.time() - started,
                model=response.model,
            )
        raise LLMError("unreachable")
