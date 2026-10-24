import re

MAX_INPUT_CHARS = 6000
CYRILLIC = re.compile(r"[а-яёА-ЯЁ]")
SENTENCE_END = re.compile(r"[.!?…](?=\s|$)")

SAFE_REPLIES = {
    "ru": "Спасибо за обращение! Мы получили ваше сообщение, проверим детали и вернёмся к вам с ответом.",
    "en": "Thank you for your message. We have received it, will check the details and get back to you.",
}


def prepare_input(text: str | None) -> tuple[str, list[str]]:
    if text is None or not text.strip():
        raise ValueError("Input text is empty")
    cleaned = " ".join(text.split()) if "\n" not in text else text.strip()
    warnings = []
    if len(cleaned) > MAX_INPUT_CHARS:
        cleaned = cleaned[:MAX_INPUT_CHARS]
        warnings.append(f"input truncated to {MAX_INPUT_CHARS} characters")
    return cleaned, warnings


def limit_reply(answer: str, max_words: int, tolerance: float = 1.3) -> tuple[str, bool]:
    words = answer.split()
    hard_limit = int(max_words * tolerance)
    if len(words) <= hard_limit:
        return answer.strip(), False
    clipped = " ".join(words[:hard_limit])
    ends = [m.end() for m in SENTENCE_END.finditer(clipped)]
    if ends and ends[-1] > len(clipped) * 0.4:
        return clipped[: ends[-1]].strip(), True
    return clipped.rstrip(",;:") + "...", True


def guess_language(text: str, language_hint: str = "") -> str:
    hint = language_hint.lower()
    if "russian" in hint or "русск" in hint:
        return "ru"
    if hint and "russian" not in hint:
        return "en"
    return "ru" if CYRILLIC.search(text) else "en"


def safe_reply(text: str, language_hint: str = "") -> str:
    return SAFE_REPLIES[guess_language(text, language_hint)]
