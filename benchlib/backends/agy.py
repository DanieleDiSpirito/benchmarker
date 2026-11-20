"""Antigravity CLI (`agy`) backend.

* `agy -p ... --output-format json` returns `usage` with input/output/thinking/cache_read tokens.
  thinking_tokens are already part of output_tokens (total_tokens = input + output).
* `agy -p /quota --output-format json` returns, per model group, the remaining fraction of the
  5-hour and weekly limits with their reset time.
"""
from __future__ import annotations

import json
import re
import subprocess

from .base import Backend, CallResult


class AgyBackend(Backend):
    name = "agy"

    def __init__(self, settings: dict):
        super().__init__(settings)
        self.bin = self.binary("AGY_BIN", "agy")
        self._display: dict[str, str] | None = None

    def version(self) -> str:
        try:
            p = subprocess.run([self.bin, "changelog"], capture_output=True, text=True, timeout=30)
            m = re.search(r"\b(\d+\.\d+\.\d+)\b", p.stdout)
            return m.group(1) if m else "sconosciuta"
        except Exception:
            return "sconosciuta"

    def call(self, model: dict, prompt: str, schema: dict | None, timeout: int) -> CallResult:
        with self.workdir() as d:
            cmd = [self.bin, "-p", prompt, "--model", model["cli_model"], "--output-format", "json",
                   "--sandbox", "--disable-slash-commands", "--print-timeout", f"{timeout}s"]
            if schema is not None:
                cmd += ["--json-schema", self.write_schema(d, schema)]
            try:
                p = self.run(cmd, d, timeout + 60)
            except subprocess.TimeoutExpired:
                return CallResult(ok=False, error="timeout")
        out = p.stdout.strip()
        data = None
        for line in reversed(out.splitlines()):  # the JSON document is the last line
            try:
                data = json.loads(line)
                break
            except ValueError:
                continue
        if not isinstance(data, dict):
            return CallResult(ok=False, error=f"exit {p.returncode}: {(p.stderr or out)[-400:]}")
        u = data.get("usage") or {}
        inp = int(u.get("input_tokens") or 0)
        tokens = {
            "input": inp,
            "cached_input": min(int(u.get("cache_read_tokens") or 0), inp),
            "output": int(u.get("output_tokens") or 0),
            "reasoning": int(u.get("thinking_tokens") or 0),
        }
        status = data.get("status")
        return CallResult(
            ok=status == "SUCCESS",
            text=data.get("response") or "",
            structured=data.get("structured_output"),
            tokens=tokens,
            turns=data.get("num_turns"),
            cli_duration_s=data.get("duration_seconds"),
            error=None if status == "SUCCESS" else f"status {status}: {(p.stderr or '')[-300:]}",
        )

    def quota_snapshot(self) -> dict | None:
        try:
            p = subprocess.run([self.bin, "-p", "/quota", "--output-format", "json"],
                               capture_output=True, text=True, timeout=90, stdin=subprocess.DEVNULL)
            data = json.loads(p.stdout.strip().splitlines()[-1])
            groups = data["command"]["data"]["groups"]
        except Exception:
            return None
        snap = {}
        for g in groups:
            for b in g.get("buckets", []):
                snap[b["id"]] = {
                    "group": g["name"],
                    "group_models": g.get("description", ""),
                    "window": b.get("window"),
                    "used": round(1.0 - float(b["remaining_fraction"]), 8),
                    "reset": b.get("reset_time"),
                }
        return snap

    def _display_names(self) -> dict[str, str]:
        if self._display is None:
            self._display = {}
            try:
                p = subprocess.run([self.bin, "models"], capture_output=True, text=True, timeout=90,
                                   stdin=subprocess.DEVNULL)
                for line in p.stdout.splitlines():
                    if "\t" in line:
                        mid, name = line.split("\t", 1)
                        self._display[mid.strip()] = re.sub(r"\s*\([^)]*\)\s*$", "", name.strip())
            except Exception:
                pass
        return self._display

    def model_group(self, model: dict, snapshot: dict) -> str | None:
        name = self._display_names().get(model["cli_model"], model.get("label", ""))
        groups = {}
        for b in snapshot.values():
            groups.setdefault(b["group"], b["group_models"])
        # A group named after the model (e.g. "Gemini 4 Argon") wins over a shared group.
        for g in groups:
            if g == name:
                return g
        for g, desc in groups.items():
            listed = [x.strip() for x in desc.split(":", 1)[-1].split(",")]
            if name in listed:
                return g
        return None

    def model_buckets(self, model: dict, snapshot: dict) -> dict[str, str | None]:
        group = self.model_group(model, snapshot)
        res = {"5h": None, "weekly": None, "group": group}
        for bid, b in snapshot.items():
            if b["group"] == group and b["window"] in ("5h", "weekly"):
                res[b["window"]] = bid
        return res
