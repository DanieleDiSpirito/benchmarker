"""OpenAI Codex CLI backend (`codex exec`).

Token usage and rate limits are read from the session rollout file that Codex writes in
$CODEX_HOME/sessions/**/rollout-*-<thread_id>.jsonl. Each `token_count` event carries
  info.total_token_usage = {input_tokens, cached_input_tokens, output_tokens, reasoning_output_tokens, ...}
  rate_limits = {primary: {used_percent, window_minutes=300, resets_at}, secondary: {..., window_minutes=10080}}
output_tokens already include reasoning_output_tokens.
The `--json` event stream on stdout is used for the thread id, errors and as a fallback for usage.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import threading
import time
from pathlib import Path

from .base import Backend, CallResult


def _codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))


def _iso(epoch) -> str | None:
    try:
        return dt.datetime.fromtimestamp(float(epoch), dt.timezone.utc).isoformat(timespec="seconds")
    except Exception:
        return None


def _window(minutes) -> str | None:
    if minutes == 300:
        return "5h"
    if minutes == 10080:
        return "weekly"
    return f"{minutes}m" if minutes else None


class CodexBackend(Backend):
    name = "codex"

    def __init__(self, settings: dict):
        super().__init__(settings)
        self.bin = self.binary("CODEX_BIN", "codex")
        self._lock = threading.Lock()
        self._latest: dict = {}   # bucket_id -> obs, most recent by observation time

    def version(self) -> str:
        try:
            return subprocess.run([self.bin, "--version"], capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            return "sconosciuta"

    # --- rollout parsing ---------------------------------------------------
    @staticmethod
    def _find_rollout(thread_id: str, started: float) -> Path | None:
        root = _codex_home() / "sessions"
        if not root.exists():
            return None
        for _ in range(10):  # the file may be flushed slightly after exit
            hits = [p for p in root.rglob(f"rollout-*{thread_id}*.jsonl")]
            if hits:
                return max(hits, key=lambda p: p.stat().st_mtime)
            time.sleep(0.5)
        return None

    @staticmethod
    def _parse_rollout(path: Path) -> tuple[dict | None, int, dict | None]:
        total, requests, limits, prev = None, 0, None, None
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                pl = ev.get("payload") if isinstance(ev, dict) else None
                if not isinstance(pl, dict) or pl.get("type") != "token_count":
                    continue
                info = pl.get("info")
                if isinstance(info, dict) and isinstance(info.get("total_token_usage"), dict):
                    t = info["total_token_usage"]
                    if t != prev:
                        requests += 1
                        prev = t
                    total = t
                if isinstance(pl.get("rate_limits"), dict):
                    limits = pl["rate_limits"]
        return total, requests, limits

    def _limits_to_obs(self, limits: dict) -> dict:
        obs = {}
        lid = limits.get("limit_id") or "codex"
        for slot in ("primary", "secondary"):
            b = limits.get(slot)
            if not isinstance(b, dict) or b.get("used_percent") is None:
                continue
            win = _window(b.get("window_minutes"))
            obs[f"{lid}:{win}"] = {"group": lid, "group_models": "", "window": win,
                                   "used": float(b["used_percent"]) / 100.0, "reset": _iso(b.get("resets_at")),
                                   "observed": time.time()}
        return obs

    def _remember(self, obs: dict) -> None:
        with self._lock:
            for k, v in obs.items():
                cur = self._latest.get(k)
                # keep the newest observation; within the same window usage only grows
                if cur is None or v["reset"] != cur["reset"] or v["used"] >= cur["used"]:
                    self._latest[k] = v

    # --- interface ---------------------------------------------------------
    def call(self, model: dict, prompt: str, schema: dict | None, timeout: int) -> CallResult:
        with self.workdir() as d:
            last = os.path.join(d, ".bench_last_message.txt")
            cmd = [self.bin, "exec", "--json", "--skip-git-repo-check", "--sandbox", "read-only",
                   "-C", d, "-m", model["cli_model"], "-o", last]
            if model.get("effort"):
                cmd += ["-c", f'model_reasoning_effort="{model["effort"]}"']
            if schema is not None:
                cmd += ["--output-schema", self.write_schema(d, schema)]
            cmd.append(prompt)
            started = time.time()
            try:
                p = self.run(cmd, d, timeout)
            except subprocess.TimeoutExpired:
                return CallResult(ok=False, error="timeout")
            text = ""
            if os.path.exists(last):
                with open(last, encoding="utf-8", errors="replace") as f:
                    text = f.read()

        thread_id, usage, errors, agent_msgs = None, None, [], []
        for line in p.stdout.splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            t = ev.get("type")
            if t == "thread.started":
                thread_id = ev.get("thread_id")
            elif t == "turn.completed" and isinstance(ev.get("usage"), dict):
                usage = ev["usage"]
            elif t in ("error", "turn.failed"):
                errors.append(json.dumps(ev)[:300])
            elif t == "item.completed" and isinstance(ev.get("item"), dict):
                it = ev["item"]
                if it.get("type") in ("agent_message", "assistant_message") and it.get("text"):
                    agent_msgs.append(it["text"])
        if not text and agent_msgs:
            text = agent_msgs[-1]

        total, requests, limits = (None, None, None)
        if thread_id:
            path = self._find_rollout(thread_id, started)
            if path:
                total, requests, limits = self._parse_rollout(path)
        src = total or usage or {}
        tokens = {
            "input": int(src.get("input_tokens") or 0),
            "cached_input": int(src.get("cached_input_tokens") or 0),
            "output": int(src.get("output_tokens") or 0),
            "reasoning": int(src.get("reasoning_output_tokens") or 0),
        }
        tokens["cached_input"] = min(tokens["cached_input"], tokens["input"])
        obs = self._limits_to_obs(limits) if limits else None
        if obs:
            self._remember(obs)

        structured = None
        if schema is not None and text.strip():
            try:
                structured = json.loads(text.strip())
            except ValueError:
                structured = None
        ok = p.returncode == 0 and bool(text.strip())
        return CallResult(
            ok=ok, text=text, structured=structured if isinstance(structured, dict) else None,
            tokens=tokens, turns=requests or None, cli_duration_s=time.time() - started,
            error=None if ok else f"exit {p.returncode}: {' | '.join(errors) or p.stderr[-400:]}",
            quota_obs=obs,
        )

    def quota_snapshot(self) -> dict | None:
        with self._lock:
            return {k: dict(v) for k, v in self._latest.items()} or None

    def warmup(self, model: dict) -> None:
        """Tiny request to learn the current rate-limit usage before measuring a block."""
        self.call(model, "Reply with the single word OK.", None, 300)

    def model_buckets(self, model: dict, snapshot: dict) -> dict[str, str | None]:
        res = {"5h": None, "weekly": None, "group": None}
        for bid, b in snapshot.items():
            if b["window"] in ("5h", "weekly") and res[b["window"]] is None:
                res[b["window"]] = bid
                res["group"] = b["group"]
        return res
