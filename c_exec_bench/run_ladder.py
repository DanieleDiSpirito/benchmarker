#!/usr/bin/env python3
"""
Effort ladder for the C-Exec-Equiv benchmark (separate from run_benchmark.py).

Every (problem, variant) pair is first run at LOW effort. Only the pairs a model got wrong are
re-run at MEDIUM, and only those still wrong at HIGH. Nothing goes beyond high.

The CLIs are called exactly as in the main benchmark (bench.py): same backends (benchlib/backends),
same isolation (no tools, no MCP, no user settings), same models (config/models.toml), same token
accounting and API prices. Only the task and the effort change. The answer comes back as structured
output ({"stdout": ...}), like the main benchmark's answers.

The levels run phase by phase (all low, then medium on the failures, then high on the remaining
failures), so a run stopped half-way still gives complete numbers for the finished levels.
Every answer is checkpointed to results/ladder/<cli>__<model>.json (one file per model, so
different CLIs can run at the same time): run the same command again to resume.

Usage (from the repository root, like bench.py):
  uv run c_exec_bench/run_ladder.py --backend agy          # every agy model of the main benchmark
  uv run c_exec_bench/run_ladder.py --backend all          # agy, then claude, then codex
  uv run c_exec_bench/run_ladder.py --models claude-haiku-5-5 gpt-6-luna --tiers 3 4
  uv run c_exec_bench/run_ladder.py --report               # only rebuild report/ladder_report.md

Per model and level the report gives the single accuracy (solved / attempted at that level) and the
cumulative accuracy (solved by that level or an earlier one / all pairs), plus time, tokens and API cost.
"""

import argparse
import json
import pathlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))          # benchlib, from the main benchmark

from benchlib.backends import get_backend                                   # noqa: E402
from benchlib.config import BACKENDS, all_models, cost_usd, load_pricing, price_for  # noqa: E402
from benchlib.prompts import PREAMBLE                                       # noqa: E402
from benchlib.runner import effort_variant                                  # noqa: E402
from run_benchmark import DATASET_FILE, VARIANTS, extract_predicted_stdout, normalize_stdout  # noqa: E402

LADDER_DIR = ROOT / "results" / "ladder"          # one file per model: parallel runs never overwrite each other
REPORT_FILE = ROOT / "report" / "ladder_report.md"
LEVELS = ("low", "medium", "high")   # never beyond high

STDOUT_SCHEMA = {
    "type": "object",
    "properties": {"stdout": {"type": "string"}},
    "required": ["stdout"],
    "additionalProperties": False,
}


def build_prompt(c_code: str) -> str:
    return (PREAMBLE + "Trace the following C11 program by hand and predict the exact text it prints to standard "
            "output when compiled and run. Put that text, and nothing else, in the `stdout` field (lines separated "
            "by \\n, no trailing newline).\n\n```c\n" + c_code.strip() + "\n```\n")


# ---------------------------------------------------------------- models
def resolve_models(names: list[str], backends: list[str]) -> list[dict]:
    """Models of the main benchmark, by key / cli_model (optionally prefixed with 'cli:') or by backend."""
    cfg = all_models()
    out = [m for (b, _), m in cfg.items() if b in backends]
    for name in names:
        cli, _, key = name.rpartition(":")
        hits = [m for (b, k), m in cfg.items() if (not cli or b == cli) and key in (k, m["cli_model"])]
        if not hits:
            raise SystemExit(f"Unknown model '{name}': use a key from config/models.toml, e.g. "
                             + ", ".join(f"{b}:{k}" for b, k in cfg))
        out += hits
    seen, uniq = set(), []
    for m in out:
        if (m["backend"], m["key"]) not in seen:
            seen.add((m["backend"], m["key"]))
            uniq.append(m)
    return uniq


def label_of(m: dict) -> str:
    return f"{m['backend']}:{m['key']}"


# ---------------------------------------------------------------- store
def _model_file(label: str) -> pathlib.Path:
    return LADDER_DIR / (label.replace(":", "__").replace("/", "_") + ".json")


def load_store() -> dict:
    runs = {}
    if LADDER_DIR.exists():
        for f in sorted(LADDER_DIR.glob("*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            runs[d["label"]] = d["records"]
    return {"runs": runs}


_save_lock = threading.Lock()


def save_store(store: dict, label: str) -> None:
    with _save_lock:
        LADDER_DIR.mkdir(parents=True, exist_ok=True)
        f = _model_file(label)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps({"label": label, "levels": list(LEVELS), "records": store["runs"][label]},
                                  indent=2), encoding="utf-8")
        tmp.replace(f)


# ---------------------------------------------------------------- ladder
def needs(steps: list[dict], level: str, retry_errors: bool) -> bool:
    """Should this pair be run at `level`? Only if every lower level was run and failed."""
    idx = LEVELS.index(level)
    done = {s["effort"]: s for s in steps}
    if any(done.get(lv, {}).get("correct") for lv in LEVELS[:idx]):
        return False                                   # already solved at a lower effort
    if any(lv not in done for lv in LEVELS[:idx]):
        return False                                   # lower level not run yet
    cur = done.get(level)
    if cur is None:
        return True
    return retry_errors and cur["status"] in ("cli_error", "timeout")


def one_call(backend, model, level, code, gt, timeout, pricing) -> dict:
    t0 = time.perf_counter()
    try:
        res = backend.call(model, build_prompt(code), STDOUT_SCHEMA, timeout)
    except Exception as e:  # never kill the pool
        from benchlib.backends.base import CallResult
        res = CallResult(ok=False, error=f"exception: {e}")
    elapsed = time.perf_counter() - t0
    if isinstance(res.structured, dict) and "stdout" in res.structured:
        pred = normalize_stdout(res.structured["stdout"])
    else:
        pred = extract_predicted_stdout(res.text or "")[0] if res.text else ""
    status = "ok" if res.ok else ("timeout" if res.error == "timeout" else "cli_error")
    price = price_for(model["pricing"], time.strftime("%Y-%m-%d"), pricing) if model.get("pricing") else None
    return {"effort": level, "cli_model": model["cli_model"], "status": status, "predicted": pred,
            "correct": status == "ok" and pred == gt, "elapsed_sec": round(elapsed, 3), "usage": res.tokens,
            "cost_usd": cost_usd(res.tokens, price) if price else None, "turns": res.turns,
            "error": res.error}


def run_level(backend, base_model, level, problems, store, timeout, retry_errors, workers, pricing) -> None:
    label = label_of(base_model)
    model = effort_variant(base_model, level)
    recs = store["runs"].setdefault(label, {})
    todo = []
    for prob in problems:
        rec = recs.setdefault(prob["id"], {"tier": prob["tier"], "algorithm_concept": prob["algorithm_concept"],
                                           "ground_truth": normalize_stdout(prob["ground_truth_stdout"]),
                                           "variants": {}})
        for var in VARIANTS:
            steps = rec["variants"].setdefault(var, {"steps": []})["steps"]
            if needs(steps, level, retry_errors):
                todo.append((prob, var, rec, steps))
    print(f"\n=== {label} @ {level} ({model['cli_model']}): {len(todo)} pair(s) to run ===", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(one_call, backend, model, level, prob[var], rec["ground_truth"], timeout, pricing):
                (prob, var, rec, steps) for prob, var, rec, steps in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            prob, var, rec, steps = futs[fut]
            step = fut.result()
            with _save_lock:
                steps[:] = [s for s in steps if s["effort"] != level] + [step]
            save_store(store, label)
            gt = rec["ground_truth"]
            mark = "PASS" if step["correct"] else \
                f"FAIL ({step['status']}: got {step['predicted']!r} vs {gt!r})" + \
                (f" [{step['error'][:160]}]" if step["error"] else "")
            print(f"  [{i}/{len(todo)}] {prob['id']} [{var:12s}] {level:6s} -> {mark} ({step['elapsed_sec']}s)",
                  flush=True)


# ---------------------------------------------------------------- report
def ladder_stats(recs: dict, tiers=None) -> dict:
    pairs = [(r, v) for r in recs.values() if tiers is None or r["tier"] in tiers for v in r["variants"].values()]
    n = len(pairs)
    out = {"n": n, "levels": {}}
    solved_before = 0
    for lv in LEVELS:
        steps = [s for _, v in pairs for s in v["steps"] if s["effort"] == lv]
        ok = sum(1 for s in steps if s["correct"])
        times = sorted(s["elapsed_sec"] for s in steps)
        tok = {k: sum(s.get("usage", {}).get(k, 0) for s in steps) for k in ("input", "output", "reasoning")}
        solved_before += ok
        out["levels"][lv] = {
            "attempted": len(steps), "solved": ok,
            "single": ok / len(steps) if steps else None,
            "cumulative": solved_before / n if n else None,
            "median_s": times[len(times) // 2] if times else None,
            "total_s": sum(times),
            "tokens": tok,
            "cost": sum(s.get("cost_usd") or 0 for s in steps),
        }
    return out


def _p(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def build_report(store: dict) -> str:
    L = ["# C-Exec-Equiv: effort ladder (low → medium → high)", "",
         "Every (problem, variant) pair runs at **low** effort; pairs still wrong are re-run at **medium**, then at "
         "**high** (never beyond). *Single* = solved / attempted at that level; *cumulative* = solved by that level "
         "or an earlier one / all pairs. Errors and timeouts count as wrong. CLIs are called as in the main benchmark (no tools).", ""]
    L += ["| Model | Pairs | Low | Medium (single) | Medium (cumul.) | High (single) | High (cumul.) "
          "| Median time L/M/H | Output tok L/M/H | API $ per pair L/M/H |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for label, recs in store["runs"].items():
        st = ladder_stats(recs)
        lv = st["levels"]
        med = " / ".join("–" if lv[x]["median_s"] is None else f"{lv[x]['median_s']:.0f}s" for x in LEVELS)
        outt = " / ".join(f"{lv[x]['tokens']['output'] / max(1, lv[x]['attempted']):.0f}" if lv[x]["attempted"]
                          else "–" for x in LEVELS)
        cost = " / ".join(f"{lv[x]['cost'] / lv[x]['attempted']:.4f}" if lv[x]["attempted"] else "–" for x in LEVELS)
        L.append(f"| {label} | {st['n']} | {_p(lv['low']['single'])} "
                 f"| {_p(lv['medium']['single'])} ({lv['medium']['solved']}/{lv['medium']['attempted']}) "
                 f"| {_p(lv['medium']['cumulative'])} "
                 f"| {_p(lv['high']['single'])} ({lv['high']['solved']}/{lv['high']['attempted']}) "
                 f"| {_p(lv['high']['cumulative'])} | {med} | {outt} | {cost} |")
    L += ["", "Output tokens are the mean per attempted pair at that level (agy includes thinking in output).", "",
          "## Cumulative accuracy per tier", "",
          "| Model | Tier | Pairs | Low | ≤ Medium | ≤ High |", "|---|---|---|---|---|---|"]
    for label, recs in store["runs"].items():
        for t in sorted({r["tier"] for r in recs.values()}):
            st = ladder_stats(recs, {t})
            lv = st["levels"]
            L.append(f"| {label} | {t} | {st['n']} | {_p(lv['low']['cumulative'])} "
                     f"| {_p(lv['medium']['cumulative'])} | {_p(lv['high']['cumulative'])} |")
    L += ["", "## Cumulative accuracy per variant", "",
          "| Model | Variant | Low | ≤ Medium | ≤ High |", "|---|---|---|---|---|"]
    for label, recs in store["runs"].items():
        for var in VARIANTS:
            sub = {pid: dict(r, variants={var: r["variants"][var]}) for pid, r in recs.items()
                   if var in r["variants"]}
            lv = ladder_stats(sub)["levels"]
            L.append(f"| {label} | {var} | {_p(lv['low']['cumulative'])} | {_p(lv['medium']['cumulative'])} "
                     f"| {_p(lv['high']['cumulative'])} |")
    return "\n".join(L) + "\n"


def write_report(store: dict) -> None:
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(build_report(store), encoding="utf-8")
    print(f"Report: {REPORT_FILE}")


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", nargs="+", choices=BACKENDS + ("all",),
                    help="every model of the main benchmark (config/models.toml) for these CLIs; 'all' = agy claude codex")
    ap.add_argument("--models", nargs="+", default=[],
                    help="model keys from config/models.toml, optionally as cli:key (e.g. claude-haiku-5-5 codex:gpt-6-luna)")
    ap.add_argument("--tiers", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--levels", nargs="+", choices=LEVELS, default=list(LEVELS),
                    help="stop the ladder early, e.g. --levels low medium (a level needs the ones below it)")
    ap.add_argument("--workers", type=int, help="parallel requests (default: the backend's workers in config/models.toml)")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per request, as in bench.py")
    ap.add_argument("--retry-errors", action="store_true", help="re-run CLI errors and timeouts at the same level")
    ap.add_argument("--report", action="store_true", help="only rebuild report/ladder_report.md")
    a = ap.parse_args()

    store = load_store()
    if a.report:
        write_report(store)
        return 0
    backends = list(BACKENDS) if a.backend and "all" in a.backend else (a.backend or [])
    models = resolve_models(a.models, backends)
    if not models:
        ap.error("--backend or --models is required (or use --report)")
    if not DATASET_FILE.exists():
        print(f"[ERROR] {DATASET_FILE} not found. Run validate_dataset.py first.", file=sys.stderr)
        return 1
    problems = [p for p in json.loads(DATASET_FILE.read_text(encoding="utf-8"))["problems"] if p["tier"] in a.tiers]
    levels = [lv for lv in LEVELS if lv in a.levels]
    pricing = load_pricing()

    backend_cache = {}
    try:
        for m in models:
            b = m["backend"]
            if b not in backend_cache:
                backend_cache[b] = get_backend(b)
            workers = a.workers or backend_cache[b].settings.get("workers", 1)
            for lv in levels:
                run_level(backend_cache[b], m, lv, problems, store, a.timeout, a.retry_errors, workers, pricing)
    except KeyboardInterrupt:
        print("\nInterrupted: every answer so far is saved, run the same command again to resume.")
        return 130
    finally:
        write_report(store)
    return 0


if __name__ == "__main__":
    sys.exit(main())
