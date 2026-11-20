"""Suite presets and the suite lock that keeps every backend on the same items."""
from __future__ import annotations

import datetime as dt
import json

from .config import SUITE_LOCK
from .datasets import BENCHMARKS, load_pool, pool_digest

# Items per benchmark. Each preset takes the first N items of the frozen pools.
PRESETS: dict[str, dict[str, int]] = {
    "smoke":    {"humaneval_plus": 2,  "mbpp_plus": 2,  "aime_2026": 2,  "math500_l5": 2,  "mmlu_pro": 2,  "ifeval": 2},
    "mini":     {"humaneval_plus": 6,  "mbpp_plus": 6,  "aime_2026": 6,  "math500_l5": 6,  "mmlu_pro": 8,  "ifeval": 8},
    "small":    {"humaneval_plus": 12, "mbpp_plus": 12, "aime_2026": 12, "math500_l5": 12, "mmlu_pro": 20, "ifeval": 20},
    "standard": {"humaneval_plus": 30, "mbpp_plus": 30, "aime_2026": 30, "math500_l5": 30, "mmlu_pro": 60, "ifeval": 60},
    "full":     {"humaneval_plus": 163, "mbpp_plus": 376, "aime_2026": 30, "math500_l5": 134, "mmlu_pro": 300, "ifeval": 541},
}


def read_lock() -> dict | None:
    if SUITE_LOCK.exists():
        return json.loads(SUITE_LOCK.read_text(encoding="utf-8"))
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
    SUITE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    SUITE_LOCK.write_text(json.dumps(lock, indent=1), encoding="utf-8")
    return lock


def resolve_suite(preset: str | None, benches: list[str] | None, relock: bool) -> dict:
    """Return the lock to use. The first run creates it; later runs (also on other
    machines) reuse it so all models answer the same items."""
    lock = read_lock()
    if lock and not relock:
        if preset and preset != lock["preset"]:
            raise SystemExit(
                f"La suite è bloccata sul preset '{lock['preset']}' (data/suite.lock.json): "
                f"hai chiesto '{preset}'. Per confrontare i modelli servono gli stessi item. "
                "Ometti --preset, oppure usa --relock (solo se nessuno ha ancora risultati)."
            )
        return lock
    return write_lock(preset or "mini", benches or list(BENCHMARKS))


def suite_items(lock: dict, bench: str) -> list[dict]:
    wanted = lock["benchmarks"][bench]["ids"]
    by_id = {x["id"]: x for x in load_pool(bench)}
    missing = [i for i in wanted if i not in by_id]
    if missing:
        raise SystemExit(f"{bench}: {len(missing)} item del lock non presenti nei dati congelati")
    return [by_id[i] for i in wanted]
