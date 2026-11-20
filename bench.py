#!/usr/bin/env python3
"""Benchmarker: entry point.

  uv run bench.py prepare                 download and freeze the datasets (already done if data/frozen exists)
  uv run bench.py selftest                check every grader against reference answers and synthetic cases
  uv run bench.py run --backend agy       run the suite on Antigravity (Gemini)
  uv run bench.py run --backend claude    run the suite on Claude Code (Opus, Sonnet, Haiku)
  uv run bench.py run --backend codex     run the suite on Codex (GPT) and extend the results
  uv run bench.py escalate --backend agy   re-run at high effort every task a model got wrong at medium
  uv run bench.py status                  progress, tokens, cost and quota per model
  uv run bench.py regrade                 re-score stored replies with the current graders
  uv run bench.py paper                   rebuild paper/benchmark_paper.pdf from every result present
  uv run bench.py paper --suite archive/X report an archived suite into archive/X/benchmark_paper.pdf
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bench.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="download and freeze the datasets")
    p.add_argument("--force", action="store_true")

    sub.add_parser("selftest", help="validate the graders (everything must pass)")

    p = sub.add_parser("run", help="run the suite on one backend")
    p.add_argument("--backend", required=True, choices=["agy", "claude", "codex"])
    p.add_argument("--models", nargs="*", help="subset of model keys from config/models.toml")
    p.add_argument("--benchmarks", nargs="*", help="subset of benchmarks")
    p.add_argument("--preset", help="smoke | mini | small | standard | full (first run only)")
    p.add_argument("--relock", action="store_true", help="recreate data/suite.lock.json with --preset")
    p.add_argument("--workers", type=int)
    p.add_argument("--timeout", type=int, default=1800, help="seconds per request")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--reserve-5h", type=float, default=3.0, help="%% of the 5-hour window to always keep free")
    p.add_argument("--reserve-weekly", type=float, default=3.0, help="%% of the weekly limit to always keep free")
    p.add_argument("--max-weekly-per-model", type=float, help="cap on the weekly %% each model may consume")
    p.add_argument("--no-wait", action="store_true", help="stop instead of waiting for the 5-hour window to reset")
    p.add_argument("--no-paper", action="store_true", help="do not rebuild the paper at the end")

    p = sub.add_parser("escalate", help="re-run at a higher effort every task a model got wrong")
    p.add_argument("--backend", required=True, choices=["agy", "claude", "codex"])
    p.add_argument("--models", nargs="*", help="subset of model keys from config/models.toml")
    p.add_argument("--effort", default="high")
    p.add_argument("--workers", type=int)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--reserve-5h", type=float, default=3.0)
    p.add_argument("--reserve-weekly", type=float, default=3.0)
    p.add_argument("--max-weekly-per-model", type=float)
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--no-paper", action="store_true")

    sub.add_parser("status", help="summary of the results present")
    sub.add_parser("regrade", help="re-score every stored reply with the current graders (no model calls)")

    p = sub.add_parser("paper", help="build the PDF")
    p.add_argument("--out", default=None)
    p.add_argument("--suite", default=None,
                   help="archived suite directory (e.g. archive/v1-mini-2026-11-20): report it into that directory")

    a = ap.parse_args(argv)

    if a.cmd == "prepare":
        from benchlib.datasets import prepare
        prepare(force=a.force)
        return 0

    if a.cmd == "selftest":
        return selftest()

    if a.cmd == "run":
        from benchlib.runner import run
        try:
            run(a.backend, a.models, a.benchmarks, a.preset, relock=a.relock, workers=a.workers,
                timeout=a.timeout, retries=a.retries, reserve_5h=a.reserve_5h / 100,
                reserve_weekly=a.reserve_weekly / 100,
                max_weekly=None if a.max_weekly_per_model is None else a.max_weekly_per_model / 100,
                wait=not a.no_wait)
        except KeyboardInterrupt:
            print("\nInterrupted: saved results stay valid, run the same command again to resume.")
            return 130
        if not a.no_paper:
            from benchlib.paper import build_paper
            print(f"Paper: {build_paper()}")
        return 0

    if a.cmd == "escalate":
        from benchlib.runner import escalate
        try:
            escalate(a.backend, a.models, effort=a.effort, workers=a.workers, timeout=a.timeout,
                     reserve_5h=a.reserve_5h / 100, reserve_weekly=a.reserve_weekly / 100,
                     max_weekly=None if a.max_weekly_per_model is None else a.max_weekly_per_model / 100,
                     wait=not a.no_wait)
        except KeyboardInterrupt:
            print("\nInterrupted: saved results stay valid, run the same command again to resume.")
            return 130
        if not a.no_paper:
            from benchlib.paper import build_paper
            print(f"Paper: {build_paper()}")
        return 0

    if a.cmd == "regrade":
        from benchlib.runner import regrade
        regrade()
        return 0

    if a.cmd == "status":
        from benchlib.analysis import print_status
        print_status()
        return 0

    if a.cmd == "paper":
        if a.suite:
            from benchlib.config import use_suite_dir
            use_suite_dir(a.suite)
        from benchlib.paper import build_paper
        print(f"Paper: {build_paper(a.out)}")
        return 0
    return 1


# ---------------------------------------------------------------- self-test
LCB_STDIN = {"id": "selftest/stdin", "func_name": None}
LCB_STDIN_TESTS = [{"input": "3\n1 2 3\n", "output": "6\n", "testtype": "stdin"},
                   {"input": "2\n0.5 0.25\n", "output": "0.750", "testtype": "stdin"}]
LCB_FUNC = {"id": "selftest/functional", "func_name": "pairSums"}
LCB_FUNC_TESTS = [{"input": "[1, 2, 3]\n4", "output": "[[1, 3]]", "testtype": "functional"},
                  {"input": "[5]\n5", "output": "[]", "testtype": "functional"}]


def _check(label: str, ok: bool, failures: list) -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {label}")
    if not ok:
        failures.append(label)


def selftest() -> int:
    from benchlib.datasets import BENCHMARKS, ensure_nltk, load_pool
    from benchlib.grading import gold_result, grade, grade_bbeh, grade_mc, grade_math, run_lcb

    ensure_nltk()
    failures: list[str] = []

    print("Reference answers (must all be accepted):")
    for bench in ("hmmt_feb_2026", "beyond_aime", "supergpqa_hard", "bbeh_mini"):
        items = load_pool(bench)
        bad = [it["id"] for it in items if not grade(bench, it, gold_result(bench, it))["correct"]]
        _check(f"{bench:<16} {len(items) - len(bad)}/{len(items)}" + (f"  rejected: {bad[:5]}" if bad else ""),
               not bad, failures)

    print("Wrong answers (must be rejected):")
    _check("math: 1/2 vs \\frac{1}{3}", not grade_math({"gold": "\\frac{1}{3}"}, "1/2"), failures)
    _check("math: 0.5 vs \\frac{1}{2} accepted", grade_math({"gold": "\\frac{1}{2}"}, "0.5"), failures)
    _check("multiple choice: 'I think B' is not read as option I",
           not grade_mc({"gold": "I", "options": list("ABCDEFGHIJ")}, "I think B"), failures)
    _check("multiple choice: '(C)' read as C", grade_mc({"gold": "C", "options": list("ABCD")}, "(C)"), failures)
    _check("BBEH: 2,3,4 vs 2,3,5", not grade_bbeh({"gold": "2, 3, 5"}, "2, 3, 4"), failures)

    print("LiveCodeBench runner (synthetic problems):")
    good_stdin = "n=int(input()); print(sum(map(float, input().split())) if n==2 else sum(map(int, input().split())))"
    _check("stdin: correct program passes (numeric tolerance 0.75 == 0.750)",
           run_lcb(LCB_STDIN, good_stdin, LCB_STDIN_TESTS)[0], failures)
    _check("stdin: wrong program fails", not run_lcb(LCB_STDIN, "print(0)", LCB_STDIN_TESTS)[0], failures)
    _check("stdin: infinite loop times out",
           not run_lcb(LCB_STDIN, "while True: pass", LCB_STDIN_TESTS[:1])[0], failures)
    good_func = ("class Solution:\n    def pairSums(self, nums: List[int], t: int) -> List[List[int]]:\n"
                 "        return [[a, b] for i, a in enumerate(nums) for b in nums[i+1:] if a + b == t]\n")
    _check("functional: correct method passes", run_lcb(LCB_FUNC, good_func, LCB_FUNC_TESTS)[0], failures)
    _check("functional: wrong method fails",
           not run_lcb(LCB_FUNC, good_func.replace("a + b == t", "a + b != t"), LCB_FUNC_TESTS)[0], failures)
    _check("sandbox: writing outside the work directory is blocked or harmless",
           _sandbox_probe(run_lcb), failures)

    print("IFBench verifiers:")
    items = load_pool("ifbench")
    accepted = [grade("ifbench", it, {"text": ""})["correct"] for it in items[:40]]
    _check("empty reply rejected by every prompt", not any(accepted), failures)

    print(f"\n{'All checks passed.' if not failures else f'{len(failures)} check(s) FAILED.'}")
    return 1 if failures else 0


def _sandbox_probe(run_lcb) -> bool:
    import os
    import tempfile

    target = os.path.join(tempfile.gettempdir(), "benchmarker_sandbox_probe.txt")
    if os.path.exists(target):
        os.remove(target)
    code = f"open({target!r}, 'w').write('x'); print(6)"
    run_lcb(LCB_STDIN, code, LCB_STDIN_TESTS[:1])
    from benchlib.grading import SANDBOX_EXEC
    escaped = os.path.exists(target)
    if escaped:
        os.remove(target)
    return (not escaped) if SANDBOX_EXEC else True


if __name__ == "__main__":
    sys.exit(main())
