"""Suite presets and the suite lock that keeps every backend on the same items."""
from __future__ import annotations

import datetime as dt
import json

from . import config as cfg
from .datasets import BENCHMARKS, load_pool, pool_digest

# Items per benchmark. Each preset takes the first N items of the frozen pools.
_B = ("livecodebench_hard", "hmmt_feb_2026", "beyond_aime", "supergpqa_hard", "bbeh_mini", "ifbench")
PRESETS: dict[str, dict[str, int]] = {
    "smoke":    dict(zip(_B, (2, 2, 2, 2, 2, 2))),
    "mini":     dict(zip(_B, (6, 6, 6, 8, 8, 8))),
    "small":    dict(zip(_B, (12, 12, 12, 20, 23, 20))),
    "standard": dict(zip(_B, (30, 33, 30, 60, 69, 60))),
    "full":     dict(zip(_B, (80, 33, 100, 300, 460, 300))),
}


def read_lock() -> dict | None:
    if cfg.SUITE_LOCK.exists():
        return json.loads(cfg.SUITE_LOCK.read_text(encoding="utf-8"))
    return None


def write_lock(preset: str, benches: list[str]) -> dict:
    lock = {
        "preset": preset,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "benchmarks": {},
    }
    for b in benches:
        items = load_pool(b)[: PRESETS[preset][b]]
        lock["benchmarks"][b] = {"n": len(items), "pool_sha256_16": pool_digest(b), "ids": [x["id"] for x in items]}
    cfg.SUITE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    cfg.SUITE_LOCK.write_text(json.dumps(lock, indent=1), encoding="utf-8")
    return lock


def resolve_suite(preset: str | None, benches: list[str] | None, relock: bool) -> dict:
    """Return the lock to use. The first run creates it; later runs (also on other
    machines) reuse it so that all models answer the same items."""
    lock = read_lock()
    if lock and not relock:
        if preset and preset != lock["preset"]:
            raise SystemExit(
                f"The suite is locked to preset '{lock['preset']}' (data/suite.lock.json) but you asked for "
                f"'{preset}'. Models are only comparable on the same items: omit --preset, or use --relock "
                "(only if nobody has results yet)."
            )
        return lock
    return write_lock(preset or "mini", benches or list(BENCHMARKS))


def suite_items(lock: dict, bench: str) -> list[dict]:
    wanted = lock["benchmarks"][bench]["ids"]
    by_id = {x["id"]: x for x in load_pool(bench)}
    missing = [i for i in wanted if i not in by_id]
    if missing:
        raise SystemExit(f"{bench}: {len(missing)} locked items are missing from the frozen data")
    return [by_id[i] for i in wanted]
