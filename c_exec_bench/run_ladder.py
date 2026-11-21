#!/usr/bin/env python3
"""
Effort ladder for the C-Exec-Equiv benchmark (separate from run_benchmark.py).

Every (problem, variant) pair is first run at LOW effort. Only the pairs a model got wrong are
re-run at MEDIUM, and only those still wrong at HIGH. Nothing goes beyond high.

The levels run phase by phase (all low, then medium on the failures, then high on the remaining
failures), so a run stopped half-way still gives complete numbers for the finished levels.
Every answer is checkpointed to results/ladder/<cli>__<model>.json (one file per model, so
different CLIs can run at the same time): run the same command again to resume.

Usage:
  python3 run_ladder.py --models codex:gpt-6.1-terra codex:gpt-6-luna
  python3 run_ladder.py --models agy:gemini-4-neon claude:claude-haiku-5-5 --tiers 3 4
  python3 run_ladder.py --backend codex     # every Codex model of the main benchmark
  python3 run_ladder.py --backend all       # agy, then claude, then codex
  python3 run_ladder.py --report            # only rebuild report/ladder_report.md

Per model and level the report gives the single accuracy (solved / attempted at that level) and the
cumulative accuracy (solved by that level or an earlier one / all pairs), plus time and tokens.
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any

from run_benchmark import (DATASET_FILE, VARIANTS, build_prompt, extract_predicted_stdout, make_isolated_workspace,
                           normalize_stdout)

ROOT = pathlib.Path(__file__).parent
LADDER_DIR = ROOT / "results" / "ladder"          # one file per model: parallel runs never overwrite each other
LEGACY_FILE = ROOT / "results" / "ladder.json"    # single-file store of the first version
REPORT_FILE = ROOT / "report" / "ladder_report.md"
LEVELS = ("low", "medium", "high")   # never beyond high
MODELS_TOML = ROOT.parent / "config" / "models.toml"   # models of the main benchmark
BACKENDS = ("agy", "claude", "codex")


def benchmarked_models(backends: list[str]) -> list[str]:
    """cli:model specs of every model in the main benchmark (config/models.toml) for these CLIs."""
    import tomllib

    cfg = tomllib.loads(MODELS_TOML.read_text(encoding="utf-8"))
    return [f"{b}:{m['cli_model']}" for b in backends for m in cfg.get(b, {}).get("models", [])]


# ---------------------------------------------------------------- CLI runners (effort as a parameter)
def _usage(d: dict | None) -> dict[str, int]:
    """Normalise a usage block to input / cached_input / output / reasoning tokens."""
    d = d or {}
    pick = lambda *keys: next((int(d[k]) for k in keys if isinstance(d.get(k), (int, float))), 0)
    return {
        "input": pick("input_tokens", "prompt_tokens", "inputTokens"),
        "cached_input": pick("cached_input_tokens", "cache_read_input_tokens", "cacheReadInputTokens"),
        "output": pick("output_tokens", "completion_tokens", "outputTokens"),
        "reasoning": pick("reasoning_output_tokens", "reasoning_tokens", "thinking_tokens"),
    }


def run_claude(prompt: str, model: str | None, effort: str, ws: pathlib.Path, timeout: int) -> dict[str, Any]:
    # Same isolation as the main benchmark (benchlib/backends/claude.py): no tools, no MCP servers, no user or
    # project settings (hooks, plugins, output styles), no slash commands, nothing saved.
    cmd = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose", "--tools", "",
           "--strict-mcp-config", "--setting-sources", "", "--disable-slash-commands", "--no-session-persistence",
           "--effort", effort]
    if model:
        cmd += ["--model", model]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ws, capture_output=True, text=True, timeout=timeout)
    elapsed = time.perf_counter() - t0
    tool_used, result = False, None
    for line in proc.stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "assistant":
            content = (ev.get("message") or {}).get("content") or []
            if any(isinstance(c, dict) and c.get("type") in ("tool_use", "server_tool_use") for c in content):
                tool_used = True
        elif ev.get("type") == "result":
            result = ev
    ok = bool(result) and result.get("subtype") == "success" and not result.get("is_error")
    return {"returncode": proc.returncode if ok else (proc.returncode or 1), "elapsed_sec": round(elapsed, 3),
            "tool_used": tool_used, "raw_response": (result or {}).get("result") or "",
            "usage": _usage((result or {}).get("usage")),
            "stderr": (proc.stderr.strip() or ("" if ok else str((result or {}).get("result"))))[-2000:]}


def run_codex(prompt: str, model: str | None, effort: str, ws: pathlib.Path, timeout: int) -> dict[str, Any]:
    out_msg = ws / "last_msg.txt"
    if out_msg.exists():
        out_msg.unlink()
    cmd = ["codex", "exec", "--json", "--ephemeral", "--skip-git-repo-check",
           "-c", f"model_reasoning_effort={effort}", "-s", "read-only", "-o", str(out_msg)]
    if model:
        cmd += ["-m", model]
    cmd.append(prompt)
    env = dict(os.environ, CODEX_HOME=str(ws / ".codex_home"))
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ws, env=env, capture_output=True, text=True, timeout=timeout)
    elapsed = time.perf_counter() - t0
    tool_used, usage = False, {}
    text = out_msg.read_text(encoding="utf-8") if out_msg.exists() else ""
    for line in proc.stdout.splitlines():
        if not line.strip().startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = ev.get("item") or {}
        if item.get("type") in ("command_execution", "mcp_tool_call", "file_change", "web_search"):
            tool_used = True
        if not text and item.get("type") == "agent_message":
            text = item.get("text", "")
        if ev.get("type") == "turn.completed":
            usage = _usage(ev.get("usage"))
    return {"returncode": proc.returncode, "elapsed_sec": round(elapsed, 3), "tool_used": tool_used,
            "raw_response": text or proc.stdout, "usage": usage, "stderr": proc.stderr.strip()[-2000:]}


def agy_model_id(model: str | None, effort: str) -> str | None:
    """agy encodes the effort in the model id (gemini-4-neon-low / -medium / -high)."""
    if not model:
        return None
    for lv in ("-low", "-medium", "-high", "-xhigh", "-max"):
        if model.endswith(lv):
            model = model[: -len(lv)]
            break
    return f"{model}-{effort}"


def run_agy(prompt: str, model: str | None, effort: str, ws: pathlib.Path, timeout: int) -> dict[str, Any]:
    cmd = ["agy", "-p", prompt, "--agent", "no-tools", "--effort", effort, "--output-format", "json"]
    mid = agy_model_id(model, effort)
    if mid:
        cmd += ["--model", mid]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ws, capture_output=True, text=True, timeout=timeout)
    elapsed = time.perf_counter() - t0
    tool_used, text, usage = False, proc.stdout, {}
    try:
        env = json.loads(proc.stdout)
        text = env.get("result", env.get("response", proc.stdout))
        usage = _usage(env.get("usage"))
        # agy counts thinking inside output_tokens: keep output as the billed total
        tool_used = env.get("num_turns", 1) > 1 or '"tool_use"' in proc.stdout or '"functionCall"' in proc.stdout
    except json.JSONDecodeError:
        pass
    return {"returncode": proc.returncode, "elapsed_sec": round(elapsed, 3), "tool_used": tool_used,
            "raw_response": text, "usage": usage, "stderr": proc.stderr.strip()[-2000:]}


RUNNERS = {"claude": run_claude, "codex": run_codex, "agy": run_agy}


# ---------------------------------------------------------------- ladder
def parse_spec(spec: str) -> tuple[str, str | None, str]:
    cli, _, model = spec.partition(":")
    cli, model = cli.strip(), (model.strip() or None)
    if cli == "agy" and model:       # one label per model, whatever effort suffix was typed
        model = agy_model_id(model, "x")[:-2]
    return cli, model, f"{cli}:{model or 'default'}"


def _model_file(label: str) -> pathlib.Path:
    return LADDER_DIR / (label.replace(":", "__").replace("/", "_") + ".json")


def load_store() -> dict:
    runs = {}
    if LADDER_DIR.exists():
        for f in sorted(LADDER_DIR.glob("*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            runs[d["label"]] = d["records"]
    if LEGACY_FILE.exists():   # models only in the old single file are imported as they are
        for label, recs in json.loads(LEGACY_FILE.read_text(encoding="utf-8")).get("runs", {}).items():
            if not label.startswith("claude:"):   # the first Claude runner was not isolated: those answers are void
                runs.setdefault(label, recs)
    return {"runs": runs}


def save_store(store: dict, label: str) -> None:
    LADDER_DIR.mkdir(parents=True, exist_ok=True)
    f = _model_file(label)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps({"label": label, "levels": list(LEVELS), "records": store["runs"][label]}, indent=2),
                   encoding="utf-8")
    tmp.replace(f)


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


def run_level(cli, model, label, level, problems, store, ws, timeout, retry_errors) -> None:
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
    print(f"\n=== {label} @ {level}: {len(todo)} pair(s) to run ===")
    for i, (prob, var, rec, steps) in enumerate(todo, 1):
        gt = rec["ground_truth"]
        try:
            res = RUNNERS[cli](build_prompt(prob[var]), model, level, ws, timeout)
            pred, summary = extract_predicted_stdout(res["raw_response"])
            status = "tool_violation" if res["tool_used"] else ("ok" if res["returncode"] == 0 else "cli_error")
            step = {"effort": level, "status": status, "predicted": pred, "correct": status == "ok" and pred == gt,
                    "elapsed_sec": res["elapsed_sec"], "usage": res["usage"], "reasoning_summary": summary,
                    "stderr": res["stderr"] if status != "ok" else ""}
        except subprocess.TimeoutExpired:
            step = {"effort": level, "status": "timeout", "predicted": "", "correct": False, "elapsed_sec": timeout,
                    "usage": {}, "reasoning_summary": "", "stderr": "TimeoutExpired"}
        steps[:] = [s for s in steps if s["effort"] != level] + [step]
        save_store(store, label)
        mark = "PASS" if step["correct"] else f"FAIL ({step['status']}: got {step['predicted']!r} vs {gt!r})"
        print(f"  [{i}/{len(todo)}] {prob['id']} [{var:12s}] {level:6s} -> {mark} ({step['elapsed_sec']}s)")


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
        }
    return out


def _p(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def build_report(store: dict) -> str:
    L = ["# C-Exec-Equiv: effort ladder (low → medium → high)", "",
         "Every (problem, variant) pair runs at **low** effort; pairs still wrong are re-run at **medium**, then at "
         "**high** (never beyond). *Single* = solved / attempted at that level; *cumulative* = solved by that level "
         "or an earlier one / all pairs. Errors, timeouts and tool violations count as wrong.", ""]
    L += ["| Model | Pairs | Low | Medium (single) | Medium (cumul.) | High (single) | High (cumul.) "
          "| Median time L/M/H | Output tok L/M/H |",
          "|---|---|---|---|---|---|---|---|---|"]
    for label, recs in store["runs"].items():
        st = ladder_stats(recs)
        lv = st["levels"]
        med = " / ".join("–" if lv[x]["median_s"] is None else f"{lv[x]['median_s']:.0f}s" for x in LEVELS)
        outt = " / ".join(f"{lv[x]['tokens']['output'] / max(1, lv[x]['attempted']):.0f}" if lv[x]["attempted"]
                          else "–" for x in LEVELS)
        L.append(f"| {label} | {st['n']} | {_p(lv['low']['single'])} "
                 f"| {_p(lv['medium']['single'])} ({lv['medium']['solved']}/{lv['medium']['attempted']}) "
                 f"| {_p(lv['medium']['cumulative'])} "
                 f"| {_p(lv['high']['single'])} ({lv['high']['solved']}/{lv['high']['attempted']}) "
                 f"| {_p(lv['high']['cumulative'])} | {med} | {outt} |")
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
    ap.add_argument("--models", nargs="+", help="cli:model targets, e.g. codex:gpt-6.1-terra agy:gemini-4-neon")
    ap.add_argument("--backend", nargs="+", choices=BACKENDS + ("all",),
                    help="every model of the main benchmark (config/models.toml) for these CLIs; 'all' = agy claude codex")
    ap.add_argument("--tiers", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--levels", nargs="+", choices=LEVELS, default=list(LEVELS),
                    help="stop the ladder early, e.g. --levels low medium (a level needs the ones below it)")
    ap.add_argument("--timeout", type=int, default=600, help="seconds per request (high effort can be slow)")
    ap.add_argument("--retry-errors", action="store_true", help="re-run CLI errors and timeouts at the same level")
    ap.add_argument("--report", action="store_true", help="only rebuild report/ladder_report.md")
    a = ap.parse_args()

    store = load_store()
    if a.report:
        write_report(store)
        return 0
    if a.backend:
        a.models = (a.models or []) + benchmarked_models(list(BACKENDS) if "all" in a.backend else a.backend)
    if not a.models:
        ap.error("--models or --backend is required (or use --report)")
    if not DATASET_FILE.exists():
        print(f"[ERROR] {DATASET_FILE} not found. Run validate_dataset.py first.", file=sys.stderr)
        return 1
    problems = [p for p in json.loads(DATASET_FILE.read_text(encoding="utf-8"))["problems"] if p["tier"] in a.tiers]
    levels = [lv for lv in LEVELS if lv in a.levels]

    ws = make_isolated_workspace()
    try:
        for spec in a.models:
            cli, model, label = parse_spec(spec)
            if cli not in RUNNERS:
                print(f"[WARN] Unknown CLI '{cli}', skipping.")
                continue
            if shutil.which(cli) is None:
                print(f"[WARN] '{cli}' not found in PATH, skipping {label}.")
                continue
            for lv in levels:
                run_level(cli, model, label, lv, problems, store, ws, a.timeout, a.retry_errors)
    except KeyboardInterrupt:
        print("\nInterrupted: every answer so far is saved, run the same command again to resume.")
        return 130
    finally:
        shutil.rmtree(ws, ignore_errors=True)
        write_report(store)
    return 0


if __name__ == "__main__":
    sys.exit(main())
