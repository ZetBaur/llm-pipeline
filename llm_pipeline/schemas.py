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
