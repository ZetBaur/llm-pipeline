from dataclasses import dataclass

from llm_pipeline.schemas import CATEGORIES


@dataclass(frozen=True)
class Route:
    category: str
    instruction: str
    max_words: int


ROUTES: dict[str, Route] = {
    "support": Route(
        "support",
        "Structured and technical. Give at most 3 short numbered steps to try, then ask for the "
        "one detail that is still missing (app version, error text, logs). Do not promise a fix date.",
        70,
    ),
    "complaint": Route(
        "complaint",
        "Careful and empathic. Start with a sincere apology that names the concrete problem, "
        "do not make excuses, say what happens next (the case is escalated). No sales language.",
        50,
    ),
    "sales": Route(
        "sales",
        "Short and selling. One sentence with the benefit, one concrete call to action "
        "(book a demo, get a quote). Do not invent prices or discounts.",
        40,
    ),
    "feedback": Route(
        "feedback",
        "Warm and brief. Thank the customer, mention exactly what they praised or suggested, say "
        "it is passed to the product team. Do not promise that a suggestion will be built.",
        40,
    ),
    "general_question": Route(
        "general_question",
        "Neutral and direct. Answer only what can be answered from the message itself; "
        "otherwise say you will check and get back, and name what you will check.",
        40,
    ),
}

assert set(ROUTES) == set(CATEGORIES)


def get_route(category: str) -> Route:
    try:
        return ROUTES[category]
    except KeyError:
        raise ValueError(f"No route for category {category!r}") from None
