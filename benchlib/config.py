"""Paths, model configuration and API price lookup."""
from __future__ import annotations

import datetime as dt
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
FROZEN_DIR = DATA_DIR / "frozen"
NLTK_DIR = DATA_DIR / "nltk"
RESULTS_DIR = ROOT / "results"
PAPER_DIR = ROOT / "paper"
SUITE_LOCK = DATA_DIR / "suite.lock.json"

BACKENDS = ("agy", "claude", "codex")


def load_models() -> dict:
    return tomllib.loads((CONFIG_DIR / "models.toml").read_text(encoding="utf-8"))


def backend_models(backend: str) -> list[dict]:
    cfg = load_models().get(backend, {})
    return [dict(m, backend=backend) for m in cfg.get("models", [])]


def backend_settings(backend: str) -> dict:
    cfg = load_models().get(backend, {})
    return {k: v for k, v in cfg.items() if k != "models"}


def all_models() -> dict[tuple[str, str], dict]:
    """(backend, key) -> model config, for every configured model."""
    return {(m["backend"], m["key"]): m for b in BACKENDS for m in backend_models(b)}


def load_pricing() -> dict:
    return tomllib.loads((CONFIG_DIR / "pricing.toml").read_text(encoding="utf-8"))


def price_for(key: str, day: str | dt.date | None = None, pricing: dict | None = None) -> dict | None:
    """Price entry valid on `day` (ISO date or date). None if the model has no price."""
    pricing = pricing or load_pricing()
    if day is None:
        day = dt.date.today()
    if isinstance(day, str):
        day = dt.date.fromisoformat(day[:10])
    for entry in pricing.get("price", []):
        if entry["key"] != key:
            continue
        lo = dt.date.fromisoformat(entry["valid_from"]) if "valid_from" in entry else dt.date.min
        hi = dt.date.fromisoformat(entry["valid_until"]) if "valid_until" in entry else dt.date.max
        if lo <= day <= hi:
            return entry
    return None


def cost_usd(tokens: dict, price: dict | None) -> float | None:
    """API-equivalent cost of one request. Output already includes reasoning tokens."""
    if price is None:
        return None
    cached = min(tokens.get("cached_input", 0), tokens.get("input", 0))
    written = min(tokens.get("cache_write", 0), tokens.get("input", 0) - cached)
    uncached = tokens.get("input", 0) - cached - written
    return (
        uncached * price["input"]
        + cached * price.get("cached_input", price["input"])
        + written * price.get("cache_write", price["input"])
        + tokens.get("output", 0) * price["output"]
    ) / 1_000_000
