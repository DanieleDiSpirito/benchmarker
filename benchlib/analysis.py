"""Aggregates every results file present (any backend, any machine) into metrics."""
from __future__ import annotations

import math
import statistics
from collections import defaultdict

from . import config as cfg
from .config import BACKENDS, all_models, cost_usd, load_pricing, price_for
from .runner import _parse_iso, load_records
from .suite import read_lock

RESET_TOLERANCE_S = 20 * 60
# A request is "stalled" when it took long but produced almost nothing: the time went into
# server-side queueing or rate-limit retries, not into generation. Such requests keep their
# tokens, cost, quota and score, but are left out of the time statistics.
STALL_MIN_S = 120
STALL_MAX_TPS = 5.0


def is_stalled(r: dict) -> bool:
    lat = r.get("latency_s") or 0
    return lat > STALL_MIN_S and r["call"]["tokens"].get("output", 0) / lat < STALL_MAX_TPS


# A quantised measurement (1-point resolution) is used as-is only when it spans at least
# this many resolution steps, i.e. when the quantisation error is at most ~10%.
MIN_STEPS = 10


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
    return v[min(len(v) - 1, max(0, int(math.ceil(q * len(v))) - 1))]


def _block_delta(block: dict, window: str) -> tuple[float | None, bool]:
    """(delta of used fraction, valid). Invalid when the window rolled over inside the block."""
    bid = (block.get("buckets") or {}).get(window)
    b, a = (block.get("before") or {}).get(bid), (block.get("after") or {}).get(bid)
    if not bid or not b or not a:
        return None, False
    d = a["used"] - b["used"]
    if b["used"] <= 1e-9 and d >= 0:
        return d, True     # the window started fresh during the block: the delta is still exact
    tb, ta = _parse_iso(b.get("reset")), _parse_iso(a.get("reset"))
    if tb and ta and abs((ta - tb).total_seconds()) > RESET_TOLERANCE_S:
        return None, False
    if d < -1e-6:
        return None, False
    return max(0.0, d), True


def _resolution(blocks: list[dict]) -> float:
    """0.01 when every reading is a whole percentage point (Claude Code, Codex), else 0."""
    vals = [s["used"] for b in blocks for side in ("before", "after") for s in (b.get(side) or {}).values()
            if isinstance(s, dict) and "used" in s]
    if vals and all(abs(v * 100 - round(v * 100)) < 1e-6 for v in vals):
        return 0.01
    return 0.0


def collect() -> dict:
    lock = read_lock()
    pricing = load_pricing()
    models_cfg = all_models()
    out = {"lock": lock, "pricing": pricing, "cells": {}, "models": [], "resolution": {}}
    if not lock:
        return out
    for backend in BACKENDS:
        bdir = cfg.RESULTS_DIR / backend
        if not bdir.exists():
            continue
        order = [k for (b, k) in models_cfg if b == backend]
        present = sorted([p.name for p in bdir.iterdir() if p.is_dir()],
                         key=lambda k: order.index(k) if k in order else 999)
        all_blocks = []
        for mkey in present:
            mcfg = models_cfg.get((backend, mkey), {"key": mkey, "label": mkey, "backend": backend,
                                                    "cli_model": mkey, "pricing": None})
            blocks = load_records(bdir / mkey / "quota.jsonl")
            all_blocks += blocks
            any_cell, observed = False, {}
            for bench in lock["benchmarks"]:
                recs = load_records(bdir / mkey / f"{bench}.jsonl")
                if not recs:
                    continue
                any_cell = True
                observed = {"cli_model": recs[-1].get("cli_model"), "effort": recs[-1].get("effort")}
                out["cells"][(backend, mkey, bench)] = _cell(bench, lock, recs, pricing,
                                                             [b for b in blocks if b.get("bench") == bench])
            if any_cell:
                # what was actually run (an archived suite may predate the current config)
                out["models"].append(dict(mcfg, backend=backend, **{k: v for k, v in observed.items() if v}))
        out["resolution"][backend] = _resolution(all_blocks)
    _apply_quota(out)
    return out


def _cell(bench, lock, recs, pricing, blocks) -> dict:
    ids = set(lock["benchmarks"][bench]["ids"])
    recs = [r for r in recs if r["id"] in ids]
    final = {}
    for r in recs:
        final[r["id"]] = r       # the last attempt for an item is the one that counts
    fin = list(final.values())
    n_done = len(fin)
    correct = sum(1 for r in fin if r["grade"]["correct"])
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
    timed = [r for r in ok_fin if not is_stalled(r)]
    lat = [r["latency_s"] for r in timed]
    tps = [r["call"]["tokens"]["output"] / r["latency_s"] for r in timed if r["latency_s"] > 0]
    turns = [r["call"]["turns"] for r in ok_fin if r["call"].get("turns")]
    lo, hi = wilson(correct, n_done)
    cell = {
        "bench": bench, "n_target": len(ids), "n_done": n_done, "attempts": len(recs),
        "correct": correct, "errors": sum(1 for r in fin if not r["call"]["ok"]),
        "acc": correct / n_done if n_done else None, "ci": (lo, hi),
        "tokens": dict(tok), "cost": cost if cost_known else None,
        "lat_mean": statistics.mean(lat) if lat else None, "lat_p50": _pct(lat, 0.5), "lat_p90": _pct(lat, 0.9),
        "tps": statistics.median(tps) if tps else None,
        "turns": statistics.mean(turns) if turns else None,
        "stalled": len(ok_fin) - len(timed),
        "thinking_rate": (sum(1 for r in ok_fin if r["call"]["tokens"].get("reasoning", 0) > 0) / len(ok_fin)
                          if ok_fin else None),
        "first": min((r["started"] for r in recs), default=None),
        "last": max((r["finished"] for r in recs), default=None),
        "blocks": [dict(b, _cost=block_cost.get(b["block"], 0.0)) for b in blocks],
    }
    if bench in ("ifbench", "ifeval"):
        cell["if_loose"] = sum(1 for r in fin if (r["grade"].get("detail") or {}).get("loose")) / max(1, n_done)
        cell["if_inst"] = sum((r["grade"].get("detail") or {}).get("instruction_strict", 0) for r in fin) / max(1, n_done)
    return cell


def _apply_quota(out: dict) -> None:
    """Quota used per cell and window. Each cell gets {"used", "method", "bound"}:

    method "measured"  exact block deltas (agy: 8-decimal readings), or a quantised total spanning
                       at least MIN_STEPS points, split across benchmarks by API cost;
    method "ratio"     quantised total too small and the model's other window moved by more steps:
                       derived from it with the account-wide weekly/5-hour ratio (both windows count
                       the same usage, so the ratio is a property of the account, not of the model);
    method "cost"      window rolled over inside a block: block cost x the model's %-per-USD;
    method "bound"     nothing moved: only an upper bound (< resolution) is known.
    """
    cells = out["cells"]
    by_model = defaultdict(list)
    for (backend, mkey, bench), c in cells.items():
        by_model[(backend, mkey)].append(c)

    # per model and window: measured total, cost of the valid blocks, and %-per-USD
    tot = {}
    for (backend, mkey), cs in by_model.items():
        for w in ("5h", "weekly"):
            t, cost_valid, have = 0.0, 0.0, False
            for c in cs:
                for b in c["blocks"]:
                    d, ok = _block_delta(b, w)
                    if ok:
                        t, cost_valid, have = t + d, cost_valid + b["_cost"], True
            tot[(backend, mkey, w)] = (t if have else None, cost_valid)

    # account-wide ratio weekly/5h for quantised backends
    k_ratio = {}
    for backend, res in out["resolution"].items():
        if not res:
            continue
        s5 = sum((tot.get((backend, m, "5h"))[0] or 0.0) for (b, m) in by_model if b == backend)
        sw = sum((tot.get((backend, m, "weekly"))[0] or 0.0) for (b, m) in by_model if b == backend)
        k_ratio[backend] = (sw / s5 if s5 else None, s5, sw)
    out["k_ratio"] = k_ratio

    for (backend, mkey), cs in by_model.items():
        res = out["resolution"].get(backend, 0.0)
        model_cost = sum(c["cost"] or 0.0 for c in cs)
        for w in ("5h", "weekly"):
            t, cost_valid = tot[(backend, mkey, w)]
            if not res:  # fine-grained readings: per-block values, cost-based fill-in for rolled windows
                rate = t / cost_valid if t and cost_valid else None
                for c in cs:
                    used, method, have = 0.0, "measured", False
                    for b in c["blocks"]:
                        d, ok = _block_delta(b, w)
                        if ok:
                            used, have = used + d, True
                        elif rate and b["_cost"]:
                            used, method, have = used + rate * b["_cost"], "cost", True
                    c.setdefault("quota", {})[w] = {"used": used if have else None, "method": method, "bound": None}
                continue
            # quantised readings (1 point): decide at model level, then split by API cost
            other = "weekly" if w == "5h" else "5h"
            t_other = tot[(backend, mkey, other)][0]
            value, method, bound = None, "measured", None
            if t is not None and t >= MIN_STEPS * res:
                value = t
            elif t is not None and t_other and t_other > t + res / 2 and k_ratio.get(backend, (None,))[0]:
                # the other window moved by more steps, so it pins this model's usage down better
                k = k_ratio[backend][0]
                value, method = t_other * (k if w == "weekly" else 1 / k), "ratio"
            elif t is not None and t > 0:
                value = t                       # small but non-zero: report it, flagged as coarse
                method = "coarse"
            elif t is not None:
                method, bound = "bound", res
            for c in cs:
                share = (c["cost"] or 0.0) / model_cost if model_cost else 1 / len(cs)
                c.setdefault("quota", {})[w] = {
                    "used": None if value is None else value * share,
                    "method": method if value is None or len(cs) == 1 else (method if method != "measured" else "split"),
                    "bound": bound,
                }


def model_summary(data: dict, model: dict) -> dict:
    cells = [c for (b, k, _), c in data["cells"].items() if b == model["backend"] and k == model["key"]]
    s = {"n_done": sum(c["n_done"] for c in cells), "n_target": sum(c["n_target"] for c in cells),
         "attempts": sum(c["attempts"] for c in cells), "correct": sum(c["correct"] for c in cells),
         "errors": sum(c["errors"] for c in cells), "stalled": sum(c.get("stalled", 0) for c in cells)}
    ok = [(c["thinking_rate"], c["n_done"] - c["errors"]) for c in cells if c.get("thinking_rate") is not None]
    s["thinking_rate"] = (sum(t * n for t, n in ok) / sum(n for _, n in ok)) if ok and sum(n for _, n in ok) else None
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
        qs = [c.get("quota", {}).get(w, {}) for c in cells]
        used = [q["used"] for q in qs if q.get("used") is not None]
        s[f"q_{w}"] = sum(used) if used else None
        methods = {q.get("method") for q in qs if q}
        s[f"q_{w}_method"] = ("ratio" if "ratio" in methods else "cost" if "cost" in methods else
                              "coarse" if "coarse" in methods else "bound" if "bound" in methods else "measured")
        bounds = [q["bound"] for q in qs if q.get("bound")]
        s[f"q_{w}_bound"] = bounds[0] if bounds and not used else None
        s[f"cap_{w}"] = s["attempts"] / s[f"q_{w}"] if s[f"q_{w}"] else None
    return s


def quota_text(value: float | None, method: str, bound: float | None, digits: int = 2) -> str:
    """Human-readable quota value: '7.00%', '~0.54%' (derived), '<1%' (below resolution)."""
    if value is None:
        if bound:
            return f"<{bound * 100:.0f}%"
        return "n/a"
    mark = {"ratio": "~", "cost": "~", "coarse": "≈"}.get(method, "")
    return f"{mark}{100 * value:.{digits}f}%"


def print_status() -> None:
    data = collect()
    if not data["lock"]:
        print("No suite yet: run `bench.py run --backend ... --preset ...` first.")
        return
    print(f"Suite '{data['lock']['preset']}': " +
          ", ".join(f"{b}={v['n']}" for b, v in data["lock"]["benchmarks"].items()))
    if not data["models"]:
        print("No results.")
        return
    hdr = (f"{'backend':<7} {'model':<20} {'done':>9} {'acc':>6} {'in tok':>10} {'out tok':>9} {'reason':>9} "
           f"{'API $':>8} {'5h':>8} {'weekly':>8}")
    print(hdr)
    print("-" * len(hdr))
    for m in data["models"]:
        s = model_summary(data, m)
        acc = "-" if s["acc_macro"] is None else f"{100 * s['acc_macro']:.1f}"
        cost = "-" if s["cost"] is None else f"{s['cost']:.3f}"
        print(f"{m['backend']:<7} {m['key']:<20} {s['n_done']:>4}/{s['n_target']:<4} {acc:>6} "
              f"{s['tokens'].get('input', 0):>10} {s['tokens'].get('output', 0):>9} {s['tokens'].get('reasoning', 0):>9} "
              f"{cost:>8} {quota_text(s['q_5h'], s['q_5h_method'], s['q_5h_bound']):>8} "
              f"{quota_text(s['q_weekly'], s['q_weekly_method'], s['q_weekly_bound']):>8}")
    print("~ derived from the other window (account-wide weekly/5h ratio); ≈ below 10 resolution steps; "
          "<1% nothing moved at 1-point resolution")


# ---------------------------------------------------------------- escalation (wrong at medium -> high)
def escalation(data: dict, model: dict, effort: str = "high") -> dict | None:
    """Recovery of a model's wrong answers when re-run at a higher effort. None if not run."""
    lock = data["lock"]
    base_dir = cfg.RESULTS_DIR / model["backend"] / model["key"]
    hi_dir = base_dir / effort
    if not hi_dir.exists():
        return None
    pricing = data["pricing"]
    res = data["resolution"].get(model["backend"], 0.0)
    out = {"effort": effort, "benches": {}, "wrong": 0, "rerun": 0, "recovered": 0, "n_done": 0, "correct": 0,
           "tokens": defaultdict(int), "cost": 0.0, "attempts": 0, "lat": [], "reasoned": 0, "ok": 0,
           "quota": {}}
    blocks = load_records(hi_dir / "quota.jsonl")
    accs_med, accs_cum = [], []
    for bench, spec in lock["benchmarks"].items():
        ids = set(spec["ids"])
        med = {}
        for r in load_records(base_dir / f"{bench}.jsonl"):
            if r["id"] in ids:
                med[r["id"]] = r
        if not med:
            continue
        wrong = {i for i, r in med.items() if not r["grade"]["correct"]}
        recs = [r for r in load_records(hi_dir / f"{bench}.jsonl") if r["id"] in wrong]
        hi = {}
        for r in recs:
            hi[r["id"]] = r
        rec = sum(1 for r in hi.values() if r["grade"]["correct"])
        n, c = len(med), len(med) - len(wrong)
        out["benches"][bench] = {"n": n, "wrong": len(wrong), "rerun": len(hi), "recovered": rec}
        out["wrong"] += len(wrong); out["rerun"] += len(hi); out["recovered"] += rec
        out["n_done"] += n; out["correct"] += c
        accs_med.append(c / n); accs_cum.append((c + rec) / n)
        for r in recs:
            t = r["call"]["tokens"]
            for kk in ("input", "cached_input", "cache_write", "output", "reasoning"):
                out["tokens"][kk] += int(t.get(kk, 0))
            out["cost"] += cost_usd(t, price_for(r["price_key"], r["started"], pricing)) or 0.0
            out["attempts"] += 1
        for r in hi.values():
            if r["call"]["ok"]:
                out["ok"] += 1
                out["reasoned"] += r["call"]["tokens"].get("reasoning", 0) > 0
                if not is_stalled(r):
                    out["lat"].append(r["latency_s"])
    out["acc_medium"] = statistics.mean(accs_med) if accs_med else None
    out["acc_cumulative"] = statistics.mean(accs_cum) if accs_cum else None
    out["recovery_rate"] = out["recovered"] / out["rerun"] if out["rerun"] else None
    out["lat_p50"] = statistics.median(out["lat"]) if out["lat"] else None
    out["thinking_rate"] = out["reasoned"] / out["ok"] if out["ok"] else None
    out["complete"] = out["rerun"] == out["wrong"]
    for w in ("5h", "weekly"):
        vals = [d for d, ok in (_block_delta(b, w) for b in blocks) if ok]
        used = sum(vals) if vals else None
        method = "coarse" if res and used else ("bound" if res and used == 0 else "measured")
        out["quota"][w] = {"used": None if method == "bound" else used, "method": method,
                           "bound": res if method == "bound" else None}
    return out


def item_times(model: dict, lock: dict | None = None) -> list[dict]:
    """Final attempt of every task a model answered, in suite order (benchmark by benchmark):
    {"bench", "id", "latency", "correct", "stalled"}. A failed call counts as wrong."""
    lock = lock or read_lock()
    mdir = cfg.RESULTS_DIR / model["backend"] / model["key"]
    out = []
    for bench, spec in (lock or {}).get("benchmarks", {}).items():
        final = {r["id"]: r for r in load_records(mdir / f"{bench}.jsonl")}
        for iid in spec["ids"]:
            r = final.get(iid)
            if r is None or r.get("latency_s") is None:
                continue
            out.append({"bench": bench, "id": iid, "latency": r["latency_s"],
                        "correct": bool(r["call"]["ok"] and r["grade"]["correct"]), "stalled": is_stalled(r)})
    return out
