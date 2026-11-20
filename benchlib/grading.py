"""Deterministic graders. No LLM is involved anywhere in scoring.

Call `grade()` from the main thread: math-verify uses signal-based timeouts.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

from .config import NLTK_DIR

CODE_TIMEOUT_S = 60
FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def _structured(result: dict, field: str) -> str | None:
    """Field from the structured output, or from the last JSON object in the text."""
    s = result.get("structured")
    if isinstance(s, dict) and isinstance(s.get(field), str):
        return s[field]
    text = result.get("text") or ""
    for m in reversed(list(re.finditer(r"\{.*?\}", text, re.S))):
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get(field), str):
            return obj[field]
    return None


def extract_code(result: dict) -> str:
    code = _structured(result, "code")
    if code is None:
        code = result.get("text") or ""
    m = FENCE.search(code)
    return m.group(1) if m else code


def extract_answer(result: dict) -> str:
    ans = _structured(result, "answer")
    if ans is None:
        text = result.get("text") or ""
        boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
        ans = boxed[-1] if boxed else text.strip().splitlines()[-1] if text.strip() else ""
    return ans.strip()


def run_python(program: str, timeout: int = CODE_TIMEOUT_S) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="bench_exec_") as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(program)
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "HOME": d, "TMPDIR": d}
        try:
            p = subprocess.run([sys.executable, "-I", path], cwd=d, env=env, capture_output=True,
                               text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if p.returncode == 0:
            return True, ""
        return False, (p.stderr or p.stdout)[-600:]


def _prompt_imports(prompt: str) -> str:
    return "\n".join(l for l in prompt.splitlines() if l.startswith(("import ", "from ")))


def grade_humaneval(item: dict, code: str) -> tuple[bool, str]:
    program = f"{_prompt_imports(item['prompt'])}\n{code}\n\n{item['test']}\n\ncheck({item['entry_point']})\n"
    return run_python(program)


def grade_mbpp(item: dict, code: str) -> tuple[bool, str]:
    program = "\n".join(item["test_imports"]) + f"\n{code}\n\n{item['test']}\n"
    return run_python(program)


def _int_answer(s: str) -> int | None:
    s = s.replace("\\boxed", "").replace("$", "").replace(",", "").strip().strip("{}").strip()
    m = re.fullmatch(r"0*(\d{1,4})(?:\.0+)?", s)
    return int(m.group(1)) if m else None


def grade_aime(item: dict, ans: str) -> bool:
    v = _int_answer(ans)
    return v is not None and v == int(item["gold"])


def _math_norm(s: str) -> str:
    s = s.replace("\\$", "").replace("\\boxed", "").strip()
    s = s.strip("$").strip()
    return f"${s}$"


def grade_math(item: dict, ans: str) -> bool:
    from math_verify import parse, verify

    gold = parse(_math_norm(item["gold"]))
    pred = parse(_math_norm(ans))
    if not pred:
        return False
    return bool(verify(gold, pred))


def grade_mmlu(item: dict, ans: str) -> bool:
    m = re.search(r"\b([A-J])\b", ans.upper())
    return bool(m) and m.group(1) == item["gold"].upper()


_ifeval_ready = False


def grade_ifeval(item: dict, text: str) -> tuple[bool, float, bool]:
    """(strict prompt-level pass, strict instruction-level fraction, loose prompt-level pass)."""
    global _ifeval_ready
    if not _ifeval_ready:
        from .datasets import ensure_nltk

        ensure_nltk(log=lambda *_: None)
        _ifeval_ready = True
    from .vendor.ifeval import evaluation_lib as ev

    inp = ev.InputExample(key=0, instruction_id_list=item["instruction_id_list"], prompt=item["prompt"],
                          kwargs=item["kwargs"])
    strict = ev.test_instruction_following_strict(inp, {item["prompt"]: text})
    loose = ev.test_instruction_following_loose(inp, {item["prompt"]: text})
    frac = sum(strict.follow_instruction_list) / max(1, len(strict.follow_instruction_list))
    return strict.follow_all_instructions, frac, loose.follow_all_instructions


def grade(bench: str, item: dict, result: dict) -> dict:
    """Return {correct, score, extracted, detail}. `score` is the primary metric in [0,1]."""
    if bench in ("humaneval_plus", "mbpp_plus"):
        code = extract_code(result)
        ok, err = (grade_humaneval if bench == "humaneval_plus" else grade_mbpp)(item, code)
        return {"correct": ok, "score": float(ok), "extracted": code[:4000], "detail": err}
    if bench == "ifeval":
        strict, frac, loose = grade_ifeval(item, result.get("text") or "")
        return {"correct": strict, "score": float(strict), "extracted": None,
                "detail": {"instruction_strict": frac, "loose": loose}}
    ans = extract_answer(result)
    try:
        ok = {"aime_2026": grade_aime, "math500_l5": grade_math, "mmlu_pro": grade_mmlu}[bench](item, ans)
    except Exception as e:  # grader failure counts as wrong, but is recorded
        return {"correct": False, "score": 0.0, "extracted": ans[:500], "detail": f"grader error: {e}"}
    return {"correct": ok, "score": float(ok), "extracted": ans[:500], "detail": ""}


def gold_result(bench: str, item: dict) -> dict:
    """A fake model result carrying the reference answer (for `bench.py selftest`)."""
    if bench in ("humaneval_plus",):
        return {"structured": {"code": item["prompt"] + item["gold"]}}
    if bench == "mbpp_plus":
        return {"structured": {"code": item["gold"]}}
    if bench == "mmlu_pro":
        return {"structured": {"answer": item["gold"]}}
    if bench in ("aime_2026", "math500_l5"):
        return {"structured": {"answer": item["gold"]}}
    raise KeyError(bench)
