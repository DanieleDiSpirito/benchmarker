"""Aggregates every results file present (any backend, any machine) into metrics."""
from __future__ import annotations

import datetime as dt
import math
import statistics
from collections import defaultdict

from .config import BACKENDS, RESULTS_DIR, all_models, cost_usd, load_pricing, price_for
from .datasets import BENCHMARKS
from .runner import _parse_iso, load_records
from .suite import read_lock

RESET_TOLERANCE_S = 20 * 60


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    # clamp: floating-point rounding can put a bound a hair on the wrong side of p (e.g. p = 1)
    return (max(0.0, min(p, c - h)), min(1.0, max(p, c + h)))


def _pct(values, q):
    if not values:
        return None
    v = sorted(values)
    i = min(len(v) - 1, max(0, int(math.ceil(q * len(v))) - 1))
    return v[i]


def _block_delta(block: dict, window: str) -> tuple[float | None, bool]:
    """(delta used fraction, valid). Invalid when the window rolled over inside the block."""
    bid = (block.get("buckets") or {}).get(window)
    b, a = (block.get("before") or {}).get(bid), (block.get("after") or {}).get(bid)
    if not bid or not b or not a:
        return None, False
    d = a["used"] - b["used"]
    if b["used"] <= 1e-9 and d >= 0:
        return d, True     # window started fresh during the block: delta still exact
    tb, ta = _parse_iso(b.get("reset")), _parse_iso(a.get("reset"))
    if tb and ta and abs((ta - tb).total_seconds()) > RESET_TOLERANCE_S:
        return None, False
    if d < -1e-6:
        return None, False
    return max(0.0, d), True


def collect() -> dict:
    lock = read_lock()
    pricing = load_pricing()
    models_cfg = all_models()
    out = {"lock": lock, "pricing": pricing, "cells": {}, "models": [], "blocks": defaultdict(list)}
    if not lock:
        return out
    for backend in BACKENDS:
        bdir = RESULTS_DIR / backend
        if not bdir.exists():
            continue
        cfg_order = [k for (b, k) in models_cfg if b == backend]
        present = sorted([p.name for p in bdir.iterdir() if p.is_dir()],
                         key=lambda k: cfg_order.index(k) if k in cfg_order else 999)
        for mkey in present:
            mcfg = models_cfg.get((backend, mkey), {"key": mkey, "label": mkey, "backend": backend,
                                                    "cli_model": mkey, "pricing": None})
            blocks = load_records(bdir / mkey / "quota.jsonl")
            any_cell = False
            for bench in lock["benchmarks"]:
                recs = load_records(bdir / mkey / f"{bench}.jsonl")
                if not recs:
                    continue
                any_cell = True
                out["cells"][(backend, mkey, bench)] = _cell(bench, lock, recs, mcfg, pricing,
                                                             [b for b in blocks if b.get("bench") == bench])
            if any_cell:
                out["models"].append(dict(mcfg, backend=backend))
    _apply_quota(out)
    return out


def _cell(bench, lock, recs, mcfg, pricing, blocks) -> dict:
    ids = set(lock["benchmarks"][bench]["ids"])
    recs = [r for r in recs if r["id"] in ids]
    final = {}
    for r in recs:
        final[r["id"]] = r
    fin = list(final.values())
    n_target = len(ids)
    n_done = len(fin)
    correct = sum(1 for r in fin if r["grade"]["correct"])
    errors = sum(1 for r in fin if not r["call"]["ok"])
    tok = defaultdict(int)
    cost, cost_known = 0.0, True
    block_cost = defaultdict(float)
    for r in recs:  # every attempt consumes tokens and quota
        t = r["call"]["tokens"]
        for k in ("input", "cached_input", "cache_write", "output", "reasoning"):
            tok[k] += int(t.get(k, 0))
        c = cost_usd(t, price_for(r["price_key"], r["started"], pricing)) if r.get("price_key") else None
        if c is None:
            cost_known = False
        else:
            cost += c
            block_cost[r.get("block")] += c
    ok_fin = [r for r in fin if r["call"]["ok"]]
    lat = [r["latency_s"] for r in ok_fin]
    tps = [r["call"]["tokens"]["output"] / r["latency_s"] for r in ok_fin if r["latency_s"] > 0]
    turns = [r["call"]["turns"] for r in ok_fin if r["call"].get("turns")]
    lo, hi = wilson(correct, n_done)
    cell = {
        "bench": bench, "n_target": n_target, "n_done": n_done, "attempts": len(recs),
        "correct": correct, "errors": errors, "acc": correct / n_done if n_done else None, "ci": (lo, hi),
        "tokens": dict(tok), "cost": cost if cost_known else None,
        "lat_mean": statistics.mean(lat) if lat else None, "lat_p50": _pct(lat, 0.5), "lat_p90": _pct(lat, 0.9),
        "tps": statistics.median(tps) if tps else None,
        "turns": statistics.mean(turns) if turns else None,
        "first": min((r["started"] for r in recs), default=None),
        "last": max((r["finished"] for r in recs), default=None),
        "blocks": [dict(b, _cost=block_cost.get(b["block"], 0.0)) for b in blocks],
    }
    if bench == "ifeval":
        cell["ifeval_loose"] = sum(1 for r in fin if (r["grade"].get("detail") or {}).get("loose")) / max(1, n_done)
        cell["ifeval_inst"] = sum((r["grade"].get("detail") or {}).get("instruction_strict", 0) for r in fin) / max(1, n_done)
    return cell


def _apply_quota(out: dict) -> None:
    """Quota % per cell: exact block deltas where valid, otherwise estimated from the
    model's own %-per-USD ratio measured on the valid blocks (quota is proportional to
    token cost, as both providers document)."""
    ratio: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for (backend, mkey, bench), cell in out["cells"].items():
        for b in cell["blocks"]:
            for w in ("5h", "weekly"):
                d, ok = _block_delta(b, w)
                if ok and b["_cost"] > 0:
                    ratio[(backend, mkey, w)][0] += d
                    ratio[(backend, mkey, w)][1] += b["_cost"]
    for (backend, mkey, bench), cell in out["cells"].items():
        cell["quota"] = {}
        for w in ("5h", "weekly"):
            total, estimated, measured_any = 0.0, False, False
            for b in cell["blocks"]:
                d, ok = _block_delta(b, w)
                if ok:
                    total += d
                    measured_any = True
                else:
                    num, den = ratio[(backend, mkey, w)]
                    if den > 0 and b["_cost"] > 0:
                        total += num / den * b["_cost"]
                        estimated = True
            have = measured_any or estimated
            cell["quota"][w] = {"used": total if have else None, "estimated": estimated}


def model_summary(data: dict, model: dict) -> dict:
    cells = [c for (b, k, _), c in data["cells"].items() if b == model["backend"] and k == model["key"]]
    s = {"n_done": sum(c["n_done"] for c in cells), "n_target": sum(c["n_target"] for c in cells),
         "attempts": sum(c["attempts"] for c in cells), "correct": sum(c["correct"] for c in cells),
         "errors": sum(c["errors"] for c in cells)}
    accs = [c["acc"] for c in cells if c["acc"] is not None]
    s["acc_macro"] = statistics.mean(accs) if accs else None
    s["acc_micro"] = s["correct"] / s["n_done"] if s["n_done"] else None
    s["benches"] = len(cells)
    tok = defaultdict(int)
    for c in cells:
        for k, v in c["tokens"].items():
            tok[k] += v
    s["tokens"] = dict(tok)
    costs = [c["cost"] for c in cells]
    s["cost"] = None if any(x is None for x in costs) or not costs else sum(costs)
    s["cost_per_correct"] = s["cost"] / s["correct"] if s["cost"] is not None and s["correct"] else None
    lat = [c["lat_p50"] for c in cells if c["lat_p50"] is not None]
    s["lat_p50"] = statistics.median(lat) if lat else None
    tps = [c["tps"] for c in cells if c["tps"] is not None]
    s["tps"] = statistics.median(tps) if tps else None
    for w in ("5h", "weekly"):
        vals = [c["quota"][w] for c in cells]
        used = [v["used"] for v in vals if v["used"] is not None]
        s[f"q_{w}"] = sum(used) if used else None
        s[f"q_{w}_est"] = any(v["estimated"] for v in vals)
        s[f"cap_{w}"] = s["attempts"] / s[f"q_{w}"] if s[f"q_{w}"] else None
    return s


def print_status() -> None:
    data = collect()
    if not data["lock"]:
        print("Nessuna suite: esegui `bench.py run --backend ... --preset ...`.")
        return
    print(f"Suite '{data['lock']['preset']}': " +
          ", ".join(f"{b}={v['n']}" for b, v in data["lock"]["benchmarks"].items()))
    if not data["models"]:
        print("Nessun risultato.")
        return
    hdr = f"{'backend':<6} {'modello':<20} {'fatti':>9} {'acc':>6} {'in tok':>10} {'out tok':>9} {'reason':>9} {'API $':>8} {'5h %':>7} {'sett %':>7}"
    print(hdr)
    print("-" * len(hdr))
    for m in data["models"]:
        s = model_summary(data, m)
        f = lambda v, fmt: (fmt.format(v) if v is not None else "-")
        print(f"{m['backend']:<6} {m['key']:<20} {s['n_done']:>4}/{s['n_target']:<4} "
              f"{f(s['acc_macro'] and 100 * s['acc_macro'], '{:5.1f}'):>6} {s['tokens'].get('input', 0):>10} "
              f"{s['tokens'].get('output', 0):>9} {s['tokens'].get('reasoning', 0):>9} {f(s['cost'], '{:8.3f}'):>8} "
              f"{f(s['q_5h'] and 100 * s['q_5h'], '{:6.2f}'):>7} {f(s['q_weekly'] and 100 * s['q_weekly'], '{:6.2f}'):>7}")
