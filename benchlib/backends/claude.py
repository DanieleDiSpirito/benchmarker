"""Claude Code CLI backend (`claude -p`), subscription login.

* `--output-format stream-json --verbose` emits, besides the messages, a `rate_limit_event`
  whose `rate_limit_info.unifiedWindows` carries `five_hour` and `seven_day` utilization (0..1,
  two decimals) and reset times; some plans add `seven_day_opus` / `seven_day_sonnet`.
* The final `result` event carries `modelUsage` (per model, summed over turns): inputTokens,
  cacheReadInputTokens, cacheCreationInputTokens, outputTokens, thinkingTokens, costUSD.
* The call is isolated: no tools (`--tools ""`, only the StructuredOutput tool when a schema is
  given), no MCP servers, no user/project settings or CLAUDE.md, no slash commands, no session file.
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import threading
import time

from .base import Backend, CallResult

WINDOWS = {"five_hour": "5h", "seven_day": "weekly", "seven_day_opus": "weekly", "seven_day_sonnet": "weekly"}


def _iso(epoch) -> str | None:
    try:
        return dt.datetime.fromtimestamp(float(epoch), dt.timezone.utc).isoformat(timespec="seconds")
    except Exception:
        return None


class ClaudeBackend(Backend):
    name = "claude"

    def __init__(self, settings: dict):
        super().__init__(settings)
        self.bin = self.binary("CLAUDE_BIN", "claude")
        self._lock = threading.Lock()
        self._latest: dict = {}

    def version(self) -> str:
        try:
            return subprocess.run([self.bin, "--version"], capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            return "sconosciuta"

    def _remember(self, obs: dict) -> None:
        with self._lock:
            for k, v in obs.items():
                cur = self._latest.get(k)
                if cur is None or v["reset"] != cur["reset"] or v["used"] >= cur["used"]:
                    self._latest[k] = v

    def call(self, model: dict, prompt: str, schema: dict | None, timeout: int) -> CallResult:
        with self.workdir() as d:
            cmd = [self.bin, "-p", prompt, "--model", model["cli_model"], "--output-format", "stream-json",
                   "--verbose", "--tools", "", "--strict-mcp-config", "--setting-sources", "",
                   "--disable-slash-commands", "--no-session-persistence"]
            if model.get("effort"):
                cmd += ["--effort", model["effort"]]
            if schema is not None:
                cmd += ["--json-schema", json.dumps(schema)]
            started = time.time()
            try:
                p = self.run(cmd, d, timeout)
            except subprocess.TimeoutExpired:
                return CallResult(ok=False, error="timeout")

        result, obs = None, {}
        for line in p.stdout.splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            t = ev.get("type")
            if t == "result":
                result = ev
            elif t == "rate_limit_event":
                win = (ev.get("rate_limit_info") or {}).get("unifiedWindows") or {}
                for name, w in win.items():
                    if name in WINDOWS and isinstance(w, dict) and w.get("utilization") is not None:
                        obs[f"claude:{name}"] = {"group": "claude", "group_models": name, "window": WINDOWS[name],
                                                 "used": float(w["utilization"]), "reset": _iso(w.get("resetsAt")),
                                                 "observed": time.time()}
        if obs:
            self._remember(obs)
        if not result:
            return CallResult(ok=False, error=f"exit {p.returncode}: {(p.stderr or p.stdout)[-400:]}", quota_obs=obs or None)

        tok = {"input": 0, "cached_input": 0, "cache_write": 0, "output": 0, "reasoning": 0}
        mu = result.get("modelUsage") or {}
        if mu:
            for u in mu.values():
                tok["cached_input"] += int(u.get("cacheReadInputTokens") or 0)
                tok["cache_write"] += int(u.get("cacheCreationInputTokens") or 0)
                tok["input"] += int(u.get("inputTokens") or 0) + int(u.get("cacheReadInputTokens") or 0) \
                    + int(u.get("cacheCreationInputTokens") or 0)
                tok["output"] += int(u.get("outputTokens") or 0)
                tok["reasoning"] += int(u.get("thinkingTokens") or 0)
        else:
            u = result.get("usage") or {}
            tok["cached_input"] = int(u.get("cache_read_input_tokens") or 0)
            tok["cache_write"] = int(u.get("cache_creation_input_tokens") or 0)
            tok["input"] = int(u.get("input_tokens") or 0) + tok["cached_input"] + tok["cache_write"]
            tok["output"] = int(u.get("output_tokens") or 0)
            tok["reasoning"] = int((u.get("output_tokens_details") or {}).get("thinking_tokens") or 0)

        ok = result.get("subtype") == "success" and not result.get("is_error")
        structured = result.get("structured_output")
        return CallResult(
            ok=ok, text=result.get("result") or "", structured=structured if isinstance(structured, dict) else None,
            tokens=tok, turns=result.get("num_turns"), cli_duration_s=(result.get("duration_ms") or 0) / 1000 or None,
            error=None if ok else f"{result.get('subtype')}: {str(result.get('result'))[:300]}",
            quota_obs=obs or None, extra={"cli_cost_usd": result.get("total_cost_usd")},
        )

    def quota_snapshot(self) -> dict | None:
        with self._lock:
            return {k: dict(v) for k, v in self._latest.items()} or None

    def warmup(self, model: dict) -> None:
        self.call(model, "Reply with the single word OK.", None, 300)

    def model_buckets(self, model: dict, snapshot: dict) -> dict[str, str | None]:
        """Model-specific weekly window (seven_day_opus / _sonnet) wins over the shared one."""
        fam = "opus" if "opus" in model["cli_model"] else "sonnet" if "sonnet" in model["cli_model"] else None
        res = {"5h": None, "weekly": None, "group": "claude"}
        if "claude:five_hour" in snapshot:
            res["5h"] = "claude:five_hour"
        if fam and f"claude:seven_day_{fam}" in snapshot:
            res["weekly"] = f"claude:seven_day_{fam}"
        elif "claude:seven_day" in snapshot:
            res["weekly"] = "claude:seven_day"
        return res
