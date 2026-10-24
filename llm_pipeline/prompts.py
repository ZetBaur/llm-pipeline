PLAIN_FORMAT = (
    "Reply strictly in this format, with these three labels and nothing else:\n"
    "SUMMARY: <text>\n"
    "KEY POINTS:\n- <point 1>\n- <point 2>\n- <point 3>\n"
    "ANSWER: <text>"
)

PLAIN_SYSTEM_V1 = (
    "You analyse customer messages. Give a short summary, 3 key points and a short helpful reply. "
    + PLAIN_FORMAT
)

PLAIN_SYSTEM_V2 = (
    "You are an assistant of a customer-support team. Analyse the customer's message.\n"
    "Rules:\n"
    "- Write in the same language as the message.\n"
    "- SUMMARY: one sentence, at most 25 words.\n"
    "- KEY POINTS: exactly 3 points, each at most 12 words.\n"
    "- ANSWER: a polite reply of at most 40 words, addressed to the customer.\n"
    + PLAIN_FORMAT
)

PLAIN_SYSTEM_V3 = (
    "You are an assistant of a customer-support team. Analyse the customer's message.\n"
    "Hard limits (count the words before answering): SUMMARY at most 25 words, exactly 3 KEY "
    "POINTS of at most 12 words each, ANSWER at most 40 words. Write in the language of the "
    "message. Never invent facts that are not in the message; if something is unknown, say "
    "that you will check it.\n"
    "Example of the expected shape (for an unrelated message):\n"
    "SUMMARY: The customer cannot download an invoice for March and asks for a new copy.\n"
    "KEY POINTS:\n- Invoice for March is not downloadable\n- Customer needs it for accounting\n"
    "- Wants a new copy\n"
    "ANSWER: Thank you for letting us know. We will send you a fresh copy of the March invoice "
    "today and check why the download failed.\n"
    "Now do the same for the real message.\n" + PLAIN_FORMAT
)

PLAIN_VARIANTS = {"v1_minimal": PLAIN_SYSTEM_V1, "v2_rules": PLAIN_SYSTEM_V2, "v3_strict": PLAIN_SYSTEM_V3}
BEST_PLAIN_VARIANT = "v2_rules"

USER_TEMPLATE = "Customer message:\n\"\"\"\n{text}\n\"\"\""


def build_user_prompt(text: str) -> str:
    return USER_TEMPLATE.format(text=text.strip())


JSON_SYSTEM = (
    "You are an assistant of a customer-support team. Analyse the customer's message and "
    "return ONE JSON object and nothing else, with exactly these fields:\n"
    '- "summary": string, one sentence, at most 25 words\n'
    '- "category": one of "support", "feedback", "complaint", "sales", "general_question"\n'
    '- "sentiment": one of "positive", "neutral", "negative"\n'
    '- "key_points": array of exactly 3 strings, each at most 12 words\n'
    '- "final_answer": string, a polite reply to the customer of at most 40 words\n'
    "Write summary, key_points and final_answer in the same language as the message. "
    "Never invent facts that are not in the message."
)


CLASSIFY_SYSTEM = (
    "You classify customer messages. Return ONE JSON object with exactly these fields:\n"
    '- "category": one of "support" (technical problem, something does not work), '
    '"feedback" (opinion, thanks, suggestion), "complaint" (dissatisfaction, demand, threat to '
    'leave), "sales" (price, plans, demo, purchase), "general_question" (information about the '
    "company or product that is not a purchase or a failure)\n"
    '- "intent": a short phrase in English, at most 8 words, what the customer wants\n'
    '- "sentiment": one of "positive", "neutral", "negative"\n'
    "If a message fits several categories, choose the one that needs the most urgent action."
)

NO_INVENTION_RULE = (
    "Use only facts from the customer's message. Never state hours, prices, addresses, "
    "deadlines or policies that are not in the message; if the customer asks for them, say that "
    "you will check and get back."
)

RESPOND_SYSTEM = (
    "You write the reply of a customer-support team to a customer. Write in the same language "
    "as the customer's message. {rule}\n"
    "Reply style for this message (category: {category}, the customer wants: {intent}):\n"
    "{instruction}\n"
    "Length: at most {max_words} words in total. Return only the text of the reply."
)


EXTRACT_SYSTEM = (
    "You extract the meaning of a customer's message. Return ONE JSON object with exactly:\n"
    '- "summary": string, one sentence, at most 25 words\n'
    '- "key_points": array of exactly 3 short strings (at most 12 words each)\n'
    '- "customer_requests": array of 1 to 5 strings, everything the customer asks or demands, '
    "each as a short phrase (include every concrete detail such as order numbers, amounts, "
    "device models)\n"
    '- "language": the language of the message in English, for example "Russian"\n'
    "Use only what is in the message."
)

SELF_CHECK_SYSTEM = (
    "You are a reviewer of customer-support replies. You get the customer's message and a "
    "draft reply. Return ONE JSON object with exactly:\n"
    '- "unsupported_claims": array of strings. Each is a concrete statement in the reply that is '
    "NOT in the message and is presented as a fact or a decision: prices, hours, deadlines, "
    "policies, statements that something is possible or impossible, guarantees, technical causes. "
    "These are NEVER unsupported claims: polite phrases; promises to check, escalate, review or "
    "get back; offering something to the customer (a demo, a quote, a callback, an escalation "
    "of the case); instructions or steps the customer may try (clearing a cache, reinstalling, "
    "splitting a file); thanks and apologies. Only list a statement if the reply asserts it as a "
    "fact about the company, the product or the situation and the message gives no basis for it. "
    "Example of a real unsupported claim: reply says \"switching plans mid-month is possible\" "
    "while the message only asks whether it is possible.\n"
    '- "missed_details": array of strings. Each is an explicit question, request or demand of the '
    "customer that the reply does not address in any way. A reply addresses a request if it "
    "acknowledges it, even when it only promises to check or to get back. NOT restating details "
    "(device model, order number, error text, amounts) is NOT a problem, unless the reply gets "
    "the detail wrong.\n"
    '- "contradicts_message": true only if the reply says the opposite of the message or '
    "misstates it, otherwise false.\n"
    "Empty arrays and false when the reply is acceptable. Do not judge style, length or tone."
)

REVISE_SUFFIX = (
    "\nA reviewer found problems in your previous draft. Write a corrected reply that fixes ALL of "
    "them and keeps the style and length limit above.\nProblems:\n{issues}\nPrevious draft:\n{draft}"
)

CHAIN_CONTEXT = (
    "Structured analysis of the message (use it, do not repeat it):\n"
    "- summary: {summary}\n- customer requests: {requests}\n- key points: {points}"
)
