#!/usr/bin/env python3
"""
Step 2: Tool-Free Benchmark Evaluation Harness
Runs validated C triplets across:
  - Claude Code CLI (`claude`)
  - OpenAI Codex CLI (`codex`)
  - Google Antigravity CLI (`agy`)

Each CLI is invoked in an isolated empty workspace with tools disabled via CLI flags,
restricted agent configs, and post-hoc JSON trace verification (any tool call
automatically disqualifies that item as a tool-violation).
Supports incremental checkpointing/resuming so rate limits never lose progress.
"""

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

ROOT = pathlib.Path(__file__).parent
DATASET_FILE = ROOT / "results" / "validated_dataset.json"
PREDICTIONS_FILE = ROOT / "results" / "predictions.json"
VARIANTS = ("classic_c", "randomized_c", "strange_c")

SYSTEM_INSTRUCTIONS = (
    "You are participating in a tool-free program execution benchmark. "
    "You MUST NOT call any tools, shell commands, compilers, interpreters, or file readers. "
    "Trace the C program mentally/textually and return ONLY a valid JSON object with two keys: "
    '"reasoning_summary" (brief trace notes) and "stdout" (the exact string printed to standard output, '
    "without trailing newline unless multiple lines are printed)."
)


def build_prompt(c_code: str) -> str:
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        "Predict the exact standard output (`stdout`) produced when compiling and running this C11 program:\n\n"
        f"```c\n{c_code.strip()}\n```\n\n"
        'Respond ONLY with raw JSON: {"reasoning_summary": "...", "stdout": "..."}'
    )


def normalize_stdout(s: str) -> str:
    if s is None:
        return ""
    lines = [line.strip() for line in str(s).strip().splitlines()]
    return "\n".join(lines).strip()


def extract_predicted_stdout(raw_text: str) -> tuple[str, str]:
    """Returns (predicted_stdout, reasoning_summary)."""
    text = raw_text.strip()
    # Try parsing JSON block inside response
    json_match = re.search(r'\{[\s\S]*?"stdout"[\s\S]*?\}', text)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            return normalize_stdout(obj.get("stdout", "")), str(obj.get("reasoning_summary", ""))
        except json.JSONDecodeError:
            pass
    # Fallback: look for "stdout": "..." regex
    m = re.search(r'"stdout"\s*:\s*"([^"]*)"', text)
    if m:
        return normalize_stdout(m.group(1)), text[:300]
    # Last-resort fallback: last non-empty line stripped of backticks
    lines = [ln.strip(" `\"'") for ln in text.splitlines() if ln.strip()]
    return normalize_stdout(lines[-1] if lines else ""), text[:300]


def make_isolated_workspace() -> pathlib.Path:
    ws = pathlib.Path(tempfile.mkdtemp(prefix="c_bench_ws_"))
    # Pre-create zero-tool custom agent for Antigravity CLI (`agy`)
    agent_dir = ws / ".agent" / "agents"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "no-tools.md").write_text(
        "---\n"
        "name: no-tools\n"
        "description: Tool-free benchmark evaluator\n"
        "tools: []\n"
        "---\n"
        "You have zero tools available. Do not attempt to invoke any tools.\n",
        encoding="utf-8",
    )
    # Create locked-down CODEX_HOME for Codex CLI
    codex_home = ws / ".codex_home"
    codex_home.mkdir(parents=True, exist_ok=True)
    (codex_home / "config.toml").write_text(
        'approval_policy = "never"\n'
        'sandbox_mode = "read-only"\n'
        'model_reasoning_effort = "medium"\n'
        "[features]\n"
        "shell_tool = false\n"
        "apply_patch_freeform = false\n"
        "multi_agent = false\n",
        encoding="utf-8",
    )
    # Symlink auth.json if user already logged into ~/.codex
    user_auth = pathlib.Path.home() / ".codex" / "auth.json"
    if user_auth.exists():
        os.symlink(user_auth, codex_home / "auth.json")
    return ws


def run_claude(prompt: str, model: str | None, ws: pathlib.Path, timeout: int) -> dict[str, Any]:
    cmd = [
        "claude",
        "--restricted",
        "--tools",
        "",
        "--effort",
        "medium",
        "--no-session-persistence",
        "--output-format",
        "json",
        "-p",
        prompt,
    ]
    if model:
        cmd.extend(["--model", model])

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ws, capture_output=True, text=True, timeout=timeout)
    elapsed = time.perf_counter() - t0

    tool_used = False
    raw_result = proc.stdout
    try:
        envelope = json.loads(proc.stdout)
        raw_result = envelope.get("result", proc.stdout)
        # Check for any tool usage in turns or messages
        if envelope.get("num_turns", 1) > 1:
            tool_used = True
        if '"tool_use"' in proc.stdout or '"tool_result"' in proc.stdout:
            tool_used = True
    except json.JSONDecodeError:
        pass

    return {
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 3),
        "tool_used": tool_used,
        "raw_response": raw_result,
        "stderr": proc.stderr.strip(),
    }


def run_codex(prompt: str, model: str | None, ws: pathlib.Path, timeout: int) -> dict[str, Any]:
    out_msg = ws / "last_msg.txt"
    if out_msg.exists():
        out_msg.unlink()

    cmd = [
        "codex",
        "exec",
        "--json",
        "--ephemeral",
        "--skip-git-repo-check",
        "-c",
        "model_reasoning_effort=medium",
        "-s",
        "read-only",
        "-o",
        str(out_msg),
    ]
    if model:
        cmd.extend(["-m", model])
    cmd.append(prompt)

    env = os.environ.copy()
    env["CODEX_HOME"] = str(ws / ".codex_home")

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ws, env=env, capture_output=True, text=True, timeout=timeout)
    elapsed = time.perf_counter() - t0

    tool_used = False
    final_text = out_msg.read_text(encoding="utf-8") if out_msg.exists() else ""
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
            ev_type = ev.get("type", "")
            item = ev.get("item", {})
            item_type = item.get("type", "")
            if item_type in ("command_execution", "mcp_tool_call", "file_change", "web_search"):
                tool_used = True
            if not final_text and item_type == "agent_message":
                final_text = item.get("text", "")
        except json.JSONDecodeError:
            continue

    return {
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 3),
        "tool_used": tool_used,
        "raw_response": final_text or proc.stdout,
        "stderr": proc.stderr.strip(),
    }


def run_agy(prompt: str, model: str | None, ws: pathlib.Path, timeout: int) -> dict[str, Any]:
    cmd = [
        "agy",
        "-p",
        prompt,
        "--agent",
        "no-tools",
        "--effort",
        "medium",
        "--output-format",
        "json",
    ]
    if model:
        cmd.extend(["--model", model])

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ws, capture_output=True, text=True, timeout=timeout)
    elapsed = time.perf_counter() - t0

    tool_used = False
    raw_result = proc.stdout
    try:
        envelope = json.loads(proc.stdout)
        raw_result = envelope.get("result", envelope.get("response", proc.stdout))
        if envelope.get("num_turns", 1) > 1:
            tool_used = True
        if '"tool_use"' in proc.stdout or '"functionCall"' in proc.stdout:
            tool_used = True
    except json.JSONDecodeError:
        pass

    return {
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 3),
        "tool_used": tool_used,
        "raw_response": raw_result,
        "stderr": proc.stderr.strip(),
    }


RUNNERS = {
    "claude": run_claude,
    "codex": run_codex,
    "agy": run_agy,
}


def parse_model_spec(spec: str) -> tuple[str, str | None, str]:
    if ":" in spec:
        cli, model = spec.split(":", 1)
        return cli.strip(), model.strip(), f"{cli.strip()}:{model.strip()}"
    return spec.strip(), None, f"{spec.strip()}:default"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tool-free C execution benchmark.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["claude", "codex", "agy"],
        help="List of cli[:model] targets (e.g. claude:claude-opus-5-5 codex:gpt-6.2-sol agy:gemini-4-neon)",
    )
    parser.add_argument("--tiers", nargs="+", type=int, default=[1, 2, 3, 4], help="Tiers to run")
    parser.add_argument("--timeout", type=int, default=180, help="Per-problem timeout in seconds")
    parser.add_argument("--retry-errors", action="store_true", help="Re-run previously failed/errored items")
    args = parser.parse_args()

    if not DATASET_FILE.exists():
        print(f"[ERROR] {DATASET_FILE} not found. Run validate_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    dataset = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
    problems = [p for p in dataset["problems"] if p["tier"] in args.tiers]

    # Load existing checkpoint if available
    if PREDICTIONS_FILE.exists():
        store = json.loads(PREDICTIONS_FILE.read_text(encoding="utf-8"))
    else:
        store = {"runs": {}}

    ws = make_isolated_workspace()
    try:
        for spec in args.models:
            cli, model_name, label = parse_model_spec(spec)
            if cli not in RUNNERS:
                print(f"[WARN] Unknown CLI '{cli}', skipping.")
                continue
            if shutil.which(cli) is None:
                print(f"[WARN] Executable '{cli}' not found in PATH, skipping '{label}'.")
                continue

            model_records = store["runs"].setdefault(label, {})
            print(f"\n=== Evaluating {label} ({len(problems)} triplets x 3 variants) ===")

            for prob in problems:
                pid = prob["id"]
                gt = normalize_stdout(prob["ground_truth_stdout"])
                prob_rec = model_records.setdefault(
                    pid,
                    {
                        "tier": prob["tier"],
                        "algorithm_concept": prob["algorithm_concept"],
                        "ground_truth": gt,
                        "variants": {},
                    },
                )

                for var in VARIANTS:
                    existing = prob_rec["variants"].get(var)
                    if existing and not (args.retry_errors and existing.get("status") != "ok"):
                        continue

                    prompt = build_prompt(prob[var])
                    try:
                        res = RUNNERS[cli](prompt, model_name, ws, args.timeout)
                        pred, summary = extract_predicted_stdout(res["raw_response"])
                        status = "tool_violation" if res["tool_used"] else ("ok" if res["returncode"] == 0 else "cli_error")
                        correct = (status == "ok") and (pred == gt)
                        prob_rec["variants"][var] = {
                            "status": status,
                            "predicted": pred,
                            "correct": correct,
                            "elapsed_sec": res["elapsed_sec"],
                            "reasoning_summary": summary,
                            "stderr": res["stderr"],
                        }
                        mark = "PASS" if correct else f"FAIL ({status}: got '{pred}' vs '{gt}')"
                        print(f"  {pid} [{var:12s}] -> {mark} ({res['elapsed_sec']}s)")
                    except subprocess.TimeoutExpired:
                        prob_rec["variants"][var] = {
                            "status": "timeout",
                            "predicted": "",
                            "correct": False,
                            "elapsed_sec": args.timeout,
                            "reasoning_summary": "",
                            "stderr": "TimeoutExpired",
                        }
                        print(f"  {pid} [{var:12s}] -> TIMEOUT ({args.timeout}s)")

                    # Checkpoint after every single evaluation
                    PREDICTIONS_FILE.write_text(json.dumps(store, indent=2), encoding="utf-8")
    finally:
        shutil.rmtree(ws, ignore_errors=True)

    print(f"\nAll runs saved to {PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()
