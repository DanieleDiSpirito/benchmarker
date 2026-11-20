"""Prompt templates and structured-output schemas (identical for every backend)."""
from __future__ import annotations

PREAMBLE = (
    "You are taking part in a benchmark. Answer directly from your own knowledge and reasoning. "
    "Do not use any tools: do not run commands or code, do not read or write files, do not search the web.\n\n"
)

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

CODE_SCHEMA = {
    "type": "object",
    "properties": {"code": {"type": "string"}},
    "required": ["code"],
    "additionalProperties": False,
}

LETTERS = "ABCDEFGHIJ"


def build(bench: str, item: dict) -> tuple[str, dict | None]:
    """Return (prompt, json_schema or None)."""
    if bench == "humaneval_plus":
        return (
            PREAMBLE
            + "Complete the following Python function. Return in the `code` field the complete, self-contained "
            "Python code: the needed imports and the full function definition with the same name and signature. "
            "No markdown fences, no example usage, no tests.\n\n"
            + item["prompt"],
            CODE_SCHEMA,
        )
    if bench == "mbpp_plus":
        return (
            PREAMBLE
            + "Write a Python function for the task below. Return in the `code` field the complete, self-contained "
            "Python code (imports + function). No markdown fences, no example usage, no tests.\n\n"
            f"Task: {item['prompt']}\n"
            f"Your code must pass this test:\n{item['test_list'][0]}\n",
            CODE_SCHEMA,
        )
    if bench in ("aime_2026", "math500_l5"):
        extra = (
            "The answer is an integer between 0 and 999. "
            if bench == "aime_2026"
            else "Write the answer in simplest exact form (LaTeX allowed, e.g. \\frac{3}{4}, \\sqrt{2}, intervals). "
        )
        return (
            PREAMBLE
            + "Solve the following math problem. "
            + extra
            + "Put only the final answer in the `answer` field, without words or units.\n\n"
            + item["problem"],
            ANSWER_SCHEMA,
        )
    if bench == "mmlu_pro":
        opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(item["options"]))
        return (
            PREAMBLE
            + "Answer the following multiple-choice question. Put only the letter of the correct option "
            "in the `answer` field.\n\n"
            f"Question: {item['question']}\n\nOptions:\n{opts}\n",
            ANSWER_SCHEMA,
        )
    if bench == "ifeval":
        # Free-form answer: the format itself is what gets evaluated, so no schema and no preamble
        # beyond the tool restriction.
        return (
            "Do not use any tools. Reply to the request below with the requested text only.\n\n" + item["prompt"],
            None,
        )
    raise KeyError(bench)
