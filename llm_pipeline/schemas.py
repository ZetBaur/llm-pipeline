from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

Category = Literal["support", "feedback", "complaint", "sales", "general_question"]
Sentiment = Literal["positive", "neutral", "negative"]
CATEGORIES: tuple[str, ...] = ("support", "feedback", "complaint", "sales", "general_question")


class OutputValidationError(Exception):
    def __init__(self, problems: list[str], raw: str = "") -> None:
        super().__init__("; ".join(problems))
        self.problems = problems
        self.raw = raw


class Analysis(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    summary: str = Field(min_length=1, max_length=400)
    category: Category
    sentiment: Sentiment
    key_points: list[str] = Field(min_length=3, max_length=3)
    final_answer: str = Field(min_length=1, max_length=800)

    @field_validator("key_points")
    @classmethod
    def points_not_empty(cls, points: list[str]) -> list[str]:
        cleaned = [p.strip() for p in points]
        if any(not p for p in cleaned):
            raise ValueError("key points must not be empty")
        return cleaned


def describe_errors(error: ValidationError) -> list[str]:
    problems = []
    for item in error.errors():
        where = ".".join(str(part) for part in item["loc"]) or "response"
        if item["type"] == "missing":
            problems.append(f"{where}: field is missing")
        else:
            problems.append(f"{where}: {item['msg']}")
    return problems


class Classification(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    category: Category
    intent: str = Field(min_length=1, max_length=120)
    sentiment: Sentiment


class ExtractResult(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    summary: str = Field(min_length=1, max_length=400)
    key_points: list[str] = Field(min_length=3, max_length=3)
    customer_requests: list[str] = Field(min_length=1, max_length=5)
    language: str = Field(min_length=2, max_length=30)

    @field_validator("key_points", "customer_requests")
    @classmethod
    def items_not_empty(cls, items: list[str]) -> list[str]:
        cleaned = [item.strip() for item in items]
        if any(not item for item in cleaned):
            raise ValueError("items must not be empty")
        return cleaned


class SelfCheck(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    unsupported_claims: list[str] = Field(default_factory=list, max_length=10)
    missed_details: list[str] = Field(default_factory=list, max_length=10)
    contradicts_message: bool = False

    @property
    def ok(self) -> bool:
        return not (self.unsupported_claims or self.missed_details or self.contradicts_message)

    def issues(self) -> list[str]:
        found = [f"unsupported claim: {c}" for c in self.unsupported_claims]
        found += [f"missed detail: {d}" for d in self.missed_details]
        if self.contradicts_message:
            found.append("the reply contradicts the customer's message")
        return found


class StepRecord(BaseModel):
    name: str
    ok: bool = True
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    note: str = ""


class ChainResult(Analysis):
    intent: str
    language: str
    customer_requests: list[str]
    self_check_passed: bool
    revised: bool
    issues_found: list[str] = Field(default_factory=list)
    steps: list[StepRecord] = Field(default_factory=list)
