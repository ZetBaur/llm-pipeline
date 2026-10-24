from dataclasses import dataclass, field

import pytest

from llm_pipeline.llm_client import Completion, LLMError


@dataclass
class FakeClient:
    responses: list = field(default_factory=list)
    calls: list = field(default_factory=list)

    def complete(self, system, user, *, json_mode=False, temperature=0.0, max_tokens=700):
        self.calls.append({"system": system, "user": user, "json_mode": json_mode})
        if not self.responses:
            raise LLMError("no more fake responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return Completion(text=item, prompt_tokens=10, completion_tokens=5, latency_s=0.01, model="fake")


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()
