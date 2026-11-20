"""Deterministic graders. No LLM is involved anywhere in scoring.

Call `grade()` from the main thread: math-verify uses signal-based timeouts.

Generated code runs in a separate `python -I` process inside a temporary directory, with
CPU/file-size limits and, on macOS, a `sandbox-exec` profile that denies network access
and any file write outside that directory.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from decimal import Decimal, InvalidOperation

TEST_TIMEOUT_S = 10          # per hidden test (LiveCodeBench uses 6 s on its reference machine)
FENCE = re.compile(r"```(?:python|py|python3)?\s*\n(.*?)```", re.S)
SANDBOX_EXEC = shutil.which("sandbox-exec") if sys.platform == "darwin" else None

LEETCODE_HEADER = """\
from typing import *
from functools import *
from collections import *
from itertools import *
from heapq import *
from bisect import *
from math import *
import math, sys, re, heapq, bisect, itertools, collections, functools, string, random, operator
sys.setrecursionlimit(1 << 20)
"""

FUNCTIONAL_HARNESS = r"""
import json, signal, sys
def _alarm(*_): raise TimeoutError()
signal.signal(signal.SIGALRM, _alarm)
ns = {}
src = open("solution.py", encoding="utf-8").read()
try:
    exec(compile(src, "solution.py", "exec"), ns)
    method = getattr(ns["Solution"](), sys.argv[1])
except BaseException as e:
    print(json.dumps({"ok": False, "why": "load: " + repr(e)[:300]})); sys.exit(0)
tests = json.load(open("tests.json", encoding="utf-8"))
for i, t in enumerate(tests):
    try:
        args = [json.loads(line) for line in t["input"].split("\n")]
        expected = json.loads(t["output"])
        signal.alarm(int(sys.argv[2]))
        got = method(*args)
        signal.alarm(0)
        got = json.loads(json.dumps(got))  # tuples -> lists, as LiveCodeBench does
    except BaseException as e:
        signal.alarm(0)
        print(json.dumps({"ok": False, "why": f"test {i}: " + repr(e)[:300]})); sys.exit(0)
    if got != expected:
        print(json.dumps({"ok": False, "why": f"test {i}: wrong answer"})); sys.exit(0)
print(json.dumps({"ok": True, "why": ""}))
"""


# ---------------------------------------------------------------- answer extraction
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
        text = (result.get("text") or "").strip()
        boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
        ans = boxed[-1] if boxed else (text.splitlines()[-1] if text else "")
    return ans.strip()


# ---------------------------------------------------------------- sandboxed execution
def _limits():
    import resource

    resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
    if sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))


def _cmd(workdir: str, argv: list[str]) -> list[str]:
    if not SANDBOX_EXEC:
        return argv
    real = os.path.realpath(workdir)
    profile = ("(version 1)(allow default)(deny network*)(deny file-write*)"
               f'(allow file-write* (subpath "{real}") (literal "/dev/null") (literal "/dev/stdout") '
               '(literal "/dev/stderr") (literal "/dev/tty") (literal "/dev/dtracehelper"))')
    return [SANDBOX_EXEC, "-p", profile] + argv


def _run(workdir: str, argv: list[str], stdin: str | None, timeout: float) -> tuple[int | None, str, str]:
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "HOME": workdir, "TMPDIR": workdir,
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
    try:
        p = subprocess.run(_cmd(workdir, argv), cwd=workdir, env=env, input=stdin, capture_output=True,
                           text=True, timeout=timeout, preexec_fn=_limits,
                           stdin=None if stdin is not None else subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return None, "", "timeout"
    return p.returncode, p.stdout, p.stderr


def _same_output(got: str, expected: str) -> bool:
    """LiveCodeBench stdout comparison: line by line after stripping; tokens may also
    match numerically (Decimal), so 1.0 == 1.00."""
    g = [l.strip() for l in got.strip().split("\n")]
    e = [l.strip() for l in expected.strip().split("\n")]
    if g == e:
        return True
    if len(g) != len(e):
        return False
    for gl, el in zip(g, e):
        if gl == el:
            continue
        gt, et = gl.split(), el.split()
        if len(gt) != len(et):
            return False
        try:
            if any(Decimal(a) != Decimal(b) for a, b in zip(gt, et)):
                return False
        except InvalidOperation:
            return False
    return True


def run_lcb(item: dict, code: str, tests: list[dict]) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="bench_exec_") as d:
        if item.get("func_name"):
            with open(os.path.join(d, "solution.py"), "w", encoding="utf-8") as f:
                f.write(LEETCODE_HEADER + "\n" + code)
            with open(os.path.join(d, "tests.json"), "w", encoding="utf-8") as f:
                json.dump(tests, f)
            with open(os.path.join(d, "harness.py"), "w", encoding="utf-8") as f:
                f.write(FUNCTIONAL_HARNESS)
            rc, out, err = _run(d, [sys.executable, "-I", "harness.py", item["func_name"], str(TEST_TIMEOUT_S)],
                                None, TEST_TIMEOUT_S * len(tests) + 30)
            if rc is None:
                return False, "timeout"
            try:
                v = json.loads(out.strip().splitlines()[-1])
                return bool(v["ok"]), v["why"]
            except (ValueError, IndexError, KeyError):
                return False, (err or out)[-400:]
        with open(os.path.join(d, "solution.py"), "w", encoding="utf-8") as f:
            f.write(code)
        for i, t in enumerate(tests):
            rc, out, err = _run(d, [sys.executable, "-I", "solution.py"], t["input"], TEST_TIMEOUT_S)
            if rc is None:
                return False, f"test {i}: timeout"
            if rc != 0:
                return False, f"test {i}: exit {rc}: {err[-300:]}"
            if not _same_output(out, t["output"]):
                return False, f"test {i}: wrong answer"
        return True, ""


# ---------------------------------------------------------------- answer graders
def _int_answer(s: str) -> int | None:
    s = s.replace("\\boxed", "").replace("$", "").replace(",", "").replace("\\!", "").strip().strip("{}").strip()
    m = re.fullmatch(r"([+-]?\d+)(?:\.0+)?", s)
    return int(m.group(1)) if m else None


def grade_integer(item: dict, ans: str) -> bool:
    v = _int_answer(ans)
    return v is not None and v == _int_answer(item["gold"])


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


def mc_letter(ans: str, n_options: int) -> str | None:
    """Option letter from an answer: a bare letter, '(C)', 'C.', 'Answer: C', or 'The answer is C'."""
    valid = "ABCDEFGHIJ"[:n_options]
    a = ans.strip().upper()
    m = re.fullmatch(r"\(?\s*([A-J])\s*\)?[.:]?", a)
    if m and m.group(1) in valid:
        return m.group(1)
    m = re.search(r"ANSWER(?: IS)?\s*[:\-]?\s*\(?([A-J])\b", a)
    if m and m.group(1) in valid:
        return m.group(1)
    return None


def grade_mc(item: dict, ans: str) -> bool:
    return mc_letter(ans, len(item["options"])) == item["gold"].strip().upper()


def grade_bbeh(item: dict, ans: str) -> bool:
    from .vendor.bbeh.evaluate import evaluate_correctness

    return bool(evaluate_correctness(ans, item["gold"]))


_nltk_ready = False


def grade_ifbench(item: dict, text: str) -> tuple[bool, float, bool]:
    """(strict prompt-level pass, strict instruction-level fraction, loose prompt-level pass)."""
    global _nltk_ready
    if not _nltk_ready:
        from .datasets import ensure_nltk

        ensure_nltk(log=lambda *_: None)
        _nltk_ready = True
    from .vendor.ifbench import evaluation_lib as ev

    inp = ev.InputExample(key=0, instruction_id_list=item["instruction_id_list"], prompt=item["prompt"],
                          kwargs=item["kwargs"])
    strict = ev.test_instruction_following_strict(inp, {item["prompt"]: text})
    loose = ev.test_instruction_following_loose(inp, {item["prompt"]: text})
    frac = sum(strict.follow_instruction_list) / max(1, len(strict.follow_instruction_list))
    return strict.follow_all_instructions, frac, loose.follow_all_instructions


def grade(bench: str, item: dict, result: dict) -> dict:
    """Return {correct, score, extracted, detail}. `score` is the primary metric in [0, 1]."""
    if bench == "livecodebench_hard":
        from .datasets import lcb_hidden_tests

        code = extract_code(result)
        ok, why = run_lcb(item, code, lcb_hidden_tests(item))
        return {"correct": ok, "score": float(ok), "extracted": code[:6000], "detail": why}
    if bench == "ifbench":
        strict, frac, loose = grade_ifbench(item, result.get("text") or "")
        return {"correct": strict, "score": float(strict), "extracted": None,
                "detail": {"instruction_strict": frac, "loose": loose}}
    ans = extract_answer(result)
    try:
        ok = {"hmmt_feb_2026": grade_math, "beyond_aime": grade_integer, "supergpqa_hard": grade_mc,
              "bbeh_mini": grade_bbeh}[bench](item, ans)
    except Exception as e:  # a grader failure counts as wrong, but is recorded
        return {"correct": False, "score": 0.0, "extracted": ans[:500], "detail": f"grader error: {e}"}
    return {"correct": ok, "score": float(ok), "extracted": ans[:500], "detail": ""}


def gold_result(bench: str, item: dict) -> dict:
    """A fake model result carrying the reference answer (for `bench.py selftest`)."""
    if bench in ("hmmt_feb_2026", "beyond_aime", "supergpqa_hard", "bbeh_mini"):
        return {"structured": {"answer": item["gold"]}}
    raise KeyError(bench)
