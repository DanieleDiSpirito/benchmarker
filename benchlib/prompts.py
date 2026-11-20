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
    if bench == "livecodebench_hard":
        if item["starter_code"]:
            how = ("Complete the class below. Return in the `code` field the complete Python 3 code "
                   "(imports + the full `class Solution`), keeping the given method signature.\n\n"
                   f"```python\n{item['starter_code']}\n```\n")
        else:
            how = ("Write a complete Python 3 program that reads the input from standard input and writes the "
                   "answer to standard output. Return the whole program in the `code` field.\n")
        return (
            PREAMBLE + "Solve the following competitive-programming problem. Efficiency matters: the hidden tests "
            "include the largest inputs allowed by the constraints. No markdown fences, no explanations.\n\n"
            f"### Problem: {item['title']}\n\n{item['content']}\n\n" + how,
            CODE_SCHEMA,
        )
    if bench in ("hmmt_feb_2026", "beyond_aime"):
        extra = ("The answer is an integer. " if bench == "beyond_aime" else
                 "Write the answer in simplest exact form (LaTeX allowed, e.g. \\frac{3}{4}, \\sqrt{2}). ")
        return (
            PREAMBLE + "Solve the following math problem. " + extra
            + "Put only the final answer in the `answer` field, without words or units.\n\n" + item["problem"],
            ANSWER_SCHEMA,
        )
    if bench == "supergpqa_hard":
        opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(item["options"]))
        return (
            PREAMBLE + "Answer the following multiple-choice question. Put only the letter of the correct option "
            f"in the `answer` field.\n\nQuestion: {item['question']}\n\nOptions:\n{opts}\n",
            ANSWER_SCHEMA,
        )
    if bench == "bbeh_mini":
        return (
            PREAMBLE + "Solve the following task. Put only the final answer in the `answer` field, in the exact "
            "format the task asks for, without explanations.\n\n" + item["input"],
            ANSWER_SCHEMA,
        )
    if bench == "ifbench":
        # Free-form answer: the format itself is what gets evaluated.
        return ("Do not use any tools. Reply to the request below with the requested text only.\n\n" + item["prompt"],
                None)
    raise KeyError(bench)
