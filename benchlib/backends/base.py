"""Common backend interface.

Token fields are normalised as:
  input         all prompt tokens of the turn, cached ones included
  cached_input  prompt tokens served from cache (subset of input)
  output        generated tokens, reasoning included (this is how both CLIs report it)
  reasoning     reasoning/thinking tokens (subset of output; 0 when the CLI does not expose them)
  cache_write   prompt tokens written to cache (subset of input; Claude Code only, billed at a premium)

Quota snapshots: {bucket_id: {"group", "window" ("5h"|"weekly"), "used" (0..1), "reset" (ISO UTC)}}
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import tempfile


@dataclasses.dataclass
class CallResult:
    ok: bool
    text: str = ""
    structured: dict | None = None
    tokens: dict = dataclasses.field(default_factory=lambda: {"input": 0, "cached_input": 0, "output": 0, "reasoning": 0})
    turns: int | None = None
    cli_duration_s: float | None = None
    error: str | None = None
    quota_obs: dict | None = None   # quota observed during the call (claude, codex)
    extra: dict = dataclasses.field(default_factory=dict)  # backend-specific data kept in the record

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


class Backend:
    name = "base"

    def __init__(self, settings: dict):
        self.settings = settings

    # --- helpers -----------------------------------------------------------
    def binary(self, env_var: str, default: str) -> str:
        b = os.environ.get(env_var) or self.settings.get("binary") or default
        path = shutil.which(b)
        if not path:
            raise SystemExit(f"CLI '{b}' not found in PATH (set {env_var})")
        return path

    @staticmethod
    def workdir() -> tempfile.TemporaryDirectory:
        return tempfile.TemporaryDirectory(prefix="bench_run_")

    @staticmethod
    def write_schema(d: str, schema: dict) -> str:
        p = os.path.join(d, ".bench_schema.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(schema, f)
        return p

    @staticmethod
    def run(cmd: list[str], cwd: str, timeout: int) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL)

    # --- interface ---------------------------------------------------------
    def version(self) -> str:
        raise NotImplementedError

    def call(self, model: dict, prompt: str, schema: dict | None, timeout: int) -> CallResult:
        raise NotImplementedError

    def quota_snapshot(self) -> dict | None:
        raise NotImplementedError

    def model_buckets(self, model: dict, snapshot: dict) -> dict[str, str | None]:
        """{"5h": bucket_id|None, "weekly": bucket_id|None} relevant to `model`."""
        raise NotImplementedError
