import json
import logging
import re

logger = logging.getLogger(__name__)

FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True
    )
    for noisy in ("httpx", "httpx2", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def count_words(text: str) -> int:
    return len(text.split())


def truncate_words(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text.strip()
    return " ".join(words[:limit]).rstrip(",;:") + "..."


def extract_json(raw: str) -> dict:
    if raw is None or not raw.strip():
        raise ValueError("Model returned an empty response")
    candidate = raw.strip()
    fenced = FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"Response is not JSON: {raw[:120]!r}") from None
        try:
            data = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Response contains broken JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError("JSON response must be an object")
    return data


def parse_plain_output(raw: str) -> dict:
    if not raw or not raw.strip():
        raise ValueError("Model returned an empty response")
    summary = re.search(r"SUMMARY:\s*(.*?)(?=\n\s*KEY POINTS:|\Z)", raw, re.DOTALL | re.IGNORECASE)
    points = re.search(r"KEY POINTS:\s*(.*?)(?=\n\s*ANSWER:|\Z)", raw, re.DOTALL | re.IGNORECASE)
    answer = re.search(r"ANSWER:\s*(.*)\Z", raw, re.DOTALL | re.IGNORECASE)
    if not (summary and points and answer):
        raise ValueError("Output does not contain SUMMARY, KEY POINTS and ANSWER sections")
    items = [
        re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        for line in points.group(1).splitlines()
        if line.strip()
    ]
    return {
        "summary": summary.group(1).strip(),
        "key_points": items,
        "final_answer": answer.group(1).strip(),
    }
