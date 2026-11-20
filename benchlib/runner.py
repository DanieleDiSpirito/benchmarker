"""Runs the suite for one backend: parallel requests, resume, quota blocks and quota guard.

Results layout (append-only, safe to copy between machines):
  results/<backend>/<model>/<bench>.jsonl   one record per attempt (last attempt per item wins)
  results/<backend>/<model>/quota.jsonl     one record per measurement block (quota before/after)
  results/<backend>/meta.json               CLI version and run dates
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import json
import platform
import time
import uuid

from .backends import get_backend
from .config import RESULTS_DIR, backend_models, backend_settings, cost_usd, price_for
from .datasets import BENCHMARKS
from .grading import grade
from .prompts import build
from .suite import resolve_suite, suite_items

TEXT_LIMIT = 200_000   # long enough that `bench.py regrade` sees the whole reply


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _parse_iso(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_records(path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def append(path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


class QuotaGuard:
    def __init__(self, backend, model, reserve_5h, reserve_weekly, max_weekly, wait, log):
        self.b, self.m = backend, model
        self.reserve_5h, self.reserve_weekly, self.max_weekly = reserve_5h, reserve_weekly, max_weekly
        self.wait, self.log = wait, log
        self.weekly_spent = 0.0   # weekly fraction used by this model during this run

    def snapshot(self) -> dict | None:
        return self.b.quota_snapshot()

    def verdict(self, snap: dict | None, block_before: dict | None) -> tuple[str, str]:
        if not snap:
            return "ok", ""
        bk = self.b.model_buckets(self.m, snap)
        w, h = bk.get("weekly"), bk.get("5h")
        if w and w in snap:
            if 1 - snap[w]["used"] < self.reserve_weekly:
                return "stop", f"weekly limit almost exhausted ({100 * (1 - snap[w]['used']):.1f}% left)"
            if self.max_weekly is not None and block_before and w in block_before:
                spent = self.weekly_spent + max(0.0, snap[w]["used"] - block_before[w]["used"])
                if spent >= self.max_weekly:
                    return "stop", f"per-model weekly budget reached ({100 * spent:.1f}%)"
        if h and h in snap and 1 - snap[h]["used"] < self.reserve_5h:
            return "wait", snap[h].get("reset") or ""
        return "ok", ""

    def sleep_until(self, reset_iso: str) -> bool:
        if not self.wait:
            return False
        t = _parse_iso(reset_iso)
        secs = 15 * 60 if t is None else max(60.0, (t - dt.datetime.now(dt.timezone.utc)).total_seconds() + 90)
        self.log(f"    5-hour window almost exhausted: waiting {secs / 60:.0f} min (reset {reset_iso})")
        time.sleep(secs)
        return True


def run(backend_name: str, models: list[str] | None, benches: list[str] | None, preset: str | None,
        relock: bool = False, workers: int | None = None, timeout: int = 1800, retries: int = 2,
        reserve_5h: float = 0.03, reserve_weekly: float = 0.03, max_weekly: float | None = None,
        wait: bool = True, check_every: int = 5, log=print) -> None:
    lock = resolve_suite(preset, None, relock)
    backend = get_backend(backend_name)
    settings = backend_settings(backend_name)
    workers = workers or int(settings.get("workers", 3))
    cfg_models = backend_models(backend_name)
    if models:
        unknown = set(models) - {m["key"] for m in cfg_models}
        if unknown:
            raise SystemExit(f"models not configured for {backend_name}: {', '.join(sorted(unknown))}")
        cfg_models = [m for m in cfg_models if m["key"] in models]
    benches = benches or list(lock["benchmarks"])
    for b in benches:
        if b not in lock["benchmarks"]:
            raise SystemExit(f"benchmark '{b}' is not part of the locked suite")

    meta_path = RESULTS_DIR / backend_name / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"runs": []}
    meta["runs"].append({"started": now_iso(), "cli_version": backend.version(), "preset": lock["preset"],
                         "os": f"{platform.system()} {platform.machine()}", "python": platform.python_version()})
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=1))

    log(f"Backend {backend_name} | preset '{lock['preset']}' | {len(cfg_models)} models | "
        f"{len(benches)} benchmarks | {workers} parallel requests")
    for model in cfg_models:
        stopped = _run_model(backend, model, lock, benches, workers, timeout, retries,
                             QuotaGuard(backend, model, reserve_5h, reserve_weekly, max_weekly, wait, log),
                             check_every, log)
        if stopped:
            log(f"  [{model['key']}] stopped: {stopped}. Run the same command again to resume.")


def _run_model(backend, model, lock, benches, workers, timeout, retries, guard, check_every, log,
               mdir=None, only: dict[str, set] | None = None):
    """Run `model` on the locked suite. `mdir` overrides the output folder and `only` restricts
    each benchmark to a set of item ids (used by the high-effort escalation phase)."""
    mdir = mdir or RESULTS_DIR / backend.name / model["key"]
    qpath = mdir / "quota.jsonl"
    for bench in benches:
        items = suite_items(lock, bench)
        if only is not None:
            items = [it for it in items if it["id"] in only.get(bench, set())]
            if not items:
                continue
        rpath = mdir / f"{bench}.jsonl"
        past = load_records(rpath)
        attempts: dict[str, int] = {}
        done: set[str] = set()
        for r in past:
            attempts[r["id"]] = attempts.get(r["id"], 0) + 1
            if r["call"]["ok"]:
                done.add(r["id"])
        pending = [it for it in items if it["id"] not in done and attempts.get(it["id"], 0) < 1 + retries]
        if not pending:
            log(f"  [{model['key']}] {bench}: complete ({len(done)}/{len(items)})")
            continue
        log(f"  [{model['key']}] {bench}: {len(pending)} items to run ({len(done)} already done)")

        while pending:
            if hasattr(backend, "warmup") and backend.quota_snapshot() is None:
                backend.warmup(model)
            before = guard.snapshot()
            v, info = guard.verdict(before, None)
            if v == "stop":
                return info
            if v == "wait":
                if not guard.sleep_until(info):
                    return "5-hour window exhausted (--no-wait)"
                continue
            block = {"block": uuid.uuid4().hex[:12], "bench": bench, "model": model["key"],
                     "backend": backend.name, "started": now_iso(), "before": before, "items": []}
            halt = None
            with cf.ThreadPoolExecutor(max_workers=workers) as pool:
                inflight: dict[cf.Future, dict] = {}
                completed = 0
                while (pending and halt is None) or inflight:
                    while pending and halt is None and len(inflight) < workers:
                        it = pending.pop(0)
                        prompt, schema = build(bench, it)
                        fut = pool.submit(_timed_call, backend, model, prompt, schema, timeout)
                        inflight[fut] = it
                    finished, _ = cf.wait(list(inflight), return_when=cf.FIRST_COMPLETED)
                    for fut in finished:
                        it = inflight.pop(fut)
                        res, started, ended, elapsed = fut.result()
                        g = safe_grade(bench, it, res.as_dict(), res.ok)
                        rec = {
                            "bench": bench, "id": it["id"], "backend": backend.name, "model": model["key"],
                            "cli_model": model["cli_model"], "effort": model.get("effort"),
                            "price_key": model.get("pricing"), "block": block["block"],
                            "started": started, "finished": ended,
                            "latency_s": round(elapsed, 2),
                            "call": {k: v for k, v in res.as_dict().items() if k != "quota_obs"},
                            "grade": g,
                        }
                        rec["call"]["text"] = (rec["call"]["text"] or "")[:TEXT_LIMIT]
                        append(mdir / f"{bench}.jsonl", rec)
                        block["items"].append(it["id"])
                        attempts[it["id"]] = attempts.get(it["id"], 0) + 1
                        if not res.ok and attempts[it["id"]] < 1 + retries:
                            pending.append(it)
                        completed += 1
                        price = price_for(model["pricing"], started) if model.get("pricing") else None
                        c = cost_usd(res.tokens, price)
                        mark = "OK " if g["correct"] else ("ERR" if not res.ok else "NO ")
                        log(f"    {mark} {it['id']:<28} {rec['latency_s']:>7.1f}s  in={res.tokens['input']:>6} "
                            f"out={res.tokens['output']:>6} (reason {res.tokens['reasoning']:>6})"
                            + (f"  ${c:.4f}" if c is not None else "")
                            + ("" if res.ok else f"  [{(res.error or '')[:80]}]"))
                        if completed % check_every == 0 and halt is None and pending:
                            vv, ii = guard.verdict(guard.snapshot(), before)
                            if vv != "ok":
                                halt = (vv, ii)
            block["finished"] = now_iso()
            block["after"] = guard.snapshot()
            block["buckets"] = backend.model_buckets(model, block["after"] or before or {})
            append(qpath, block)
            w = block["buckets"].get("weekly")
            if w and before and block["after"] and w in before and w in block["after"]:
                guard.weekly_spent += max(0.0, block["after"][w]["used"] - before[w]["used"])
            if halt:
                v, info = halt
                if v == "stop":
                    return info
                if not guard.sleep_until(info):
                    return "5-hour window exhausted (--no-wait)"
    return None


def safe_grade(bench: str, item: dict, call: dict, ok: bool) -> dict:
    """Grade a reply; a grader crash is recorded (and fixable with `bench.py regrade`) instead
    of stopping the run and losing the reply."""
    if not ok:
        return {"correct": False, "score": 0.0, "extracted": None, "detail": "call failed"}
    try:
        return grade(bench, item, call)
    except Exception as e:
        return {"correct": False, "score": 0.0, "extracted": None, "detail": f"grader exception: {e!r}"[:500],
                "grader_error": True}


def regrade(log=print) -> None:
    """Re-score every stored reply with the current graders (no model is called)."""
    from .suite import read_lock

    lock = read_lock()
    if not lock:
        raise SystemExit("No suite lock: nothing to regrade.")
    pools = {b: {it["id"]: it for it in suite_items(lock, b)} for b in lock["benchmarks"]}
    for path in sorted(list(RESULTS_DIR.glob("*/*/*.jsonl")) + list(RESULTS_DIR.glob("*/*/*/*.jsonl"))):
        bench = path.stem
        if bench not in pools:
            continue
        recs, changed = load_records(path), 0
        for r in recs:
            it = pools[bench].get(r["id"])
            if it is None:
                continue
            g = safe_grade(bench, it, r["call"], r["call"]["ok"])
            changed += g["correct"] != r["grade"]["correct"]
            r["grade"] = g
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
        tmp.replace(path)
        log(f"  {path.relative_to(RESULTS_DIR)}: {len(recs)} records, {changed} verdicts changed")


def _timed_call(backend, model, prompt, schema, timeout):
    started, t0 = now_iso(), time.perf_counter()
    try:
        res = backend.call(model, prompt, schema, timeout)
    except Exception as e:  # never kill the pool
        from .backends.base import CallResult
        res = CallResult(ok=False, error=f"exception: {e}")
    return res, started, now_iso(), time.perf_counter() - t0


# ---------------------------------------------------------------- escalation phase
def wrong_ids(backend: str, key: str, lock: dict) -> dict[str, set]:
    """Items a model got wrong in its main (medium) run: last attempt per item not correct."""
    out = {}
    for bench, spec in lock["benchmarks"].items():
        final = {}
        for r in load_records(RESULTS_DIR / backend / key / f"{bench}.jsonl"):
            if r["id"] in spec["ids"]:
                final[r["id"]] = r
        out[bench] = {i for i, r in final.items() if not r["grade"]["correct"]}
    return out


def effort_variant(model: dict, effort: str) -> dict:
    """The same model at another effort: agy encodes it in the model id, the others take a flag."""
    m = dict(model)
    old = model.get("effort")
    if old and m["cli_model"].endswith(f"-{old}"):
        m["cli_model"] = m["cli_model"][: -len(old)] + effort
    m["effort"] = effort
    return m


def escalate(backend_name: str, models: list[str] | None, effort: str = "high", workers: int | None = None,
             timeout: int = 1800, retries: int = 2, reserve_5h: float = 0.03, reserve_weekly: float = 0.03,
             max_weekly: float | None = None, wait: bool = True, check_every: int = 5, log=print) -> None:
    """Re-run, at `effort`, every task each model got wrong in its main run. Results go to
    results/<cli>/<model>/<effort>/ and never alter the main (medium) results."""
    from .suite import read_lock

    lock = read_lock()
    if not lock:
        raise SystemExit("No suite lock: run the main benchmark first.")
    backend = get_backend(backend_name)
    workers = workers or int(backend_settings(backend_name).get("workers", 3))
    cfg_models = backend_models(backend_name)
    if models:
        cfg_models = [m for m in cfg_models if m["key"] in models]
    for model in cfg_models:
        wrong = wrong_ids(backend_name, model["key"], lock)
        n = sum(len(v) for v in wrong.values())
        if not n:
            log(f"  [{model['key']}] nothing to escalate (no main results or no wrong answers)")
            continue
        hi = effort_variant(model, effort)
        log(f"  [{model['key']}] escalating {n} wrong answers to effort '{effort}' ({hi['cli_model']})")
        stopped = _run_model(backend, hi, lock, list(lock["benchmarks"]), workers, timeout, retries,
                             QuotaGuard(backend, hi, reserve_5h, reserve_weekly, max_weekly, wait, log),
                             check_every, log, mdir=RESULTS_DIR / backend_name / model["key"] / effort, only=wrong)
        if stopped:
            log(f"  [{model['key']}] stopped: {stopped}. Run the same command again to resume.")
