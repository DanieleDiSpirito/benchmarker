#!/usr/bin/env python3
"""Benchmarker: entry point.

  uv run bench.py prepare                     scarica e congela i dataset (già fatto se data/frozen esiste)
  uv run bench.py selftest                    verifica i valutatori con le soluzioni di riferimento
  uv run bench.py run --backend agy           esegue la suite con Antigravity (Gemini)
  uv run bench.py run --backend claude        esegue la suite con Claude Code (Opus, Sonnet)
  uv run bench.py run --backend codex         esegue la suite con Codex (GPT) ed estende i risultati
  uv run bench.py status                      avanzamento e quota consumata per modello
  uv run bench.py paper                       rigenera paper/benchmark_paper.pdf da tutti i risultati
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bench.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="scarica e congela i dataset")
    p.add_argument("--force", action="store_true")

    sub.add_parser("selftest", help="valuta le soluzioni di riferimento (devono passare tutte)")

    p = sub.add_parser("run", help="esegue la suite su un backend")
    p.add_argument("--backend", required=True, choices=["agy", "claude", "codex"])
    p.add_argument("--models", nargs="*", help="sottoinsieme di chiavi da config/models.toml")
    p.add_argument("--benchmarks", nargs="*", help="sottoinsieme di benchmark")
    p.add_argument("--preset", help="smoke | mini | small | standard | full (solo alla prima esecuzione)")
    p.add_argument("--relock", action="store_true", help="ricrea data/suite.lock.json con --preset")
    p.add_argument("--workers", type=int)
    p.add_argument("--timeout", type=int, default=1800, help="secondi per richiesta")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--reserve-5h", type=float, default=3.0, help="%% della finestra 5h da lasciare libero")
    p.add_argument("--reserve-weekly", type=float, default=3.0, help="%% settimanale da lasciare libero")
    p.add_argument("--max-weekly-per-model", type=float, help="tetto %% settimanale consumabile da ciascun modello")
    p.add_argument("--no-wait", action="store_true", help="non attendere il reset della finestra 5h: fermati")
    p.add_argument("--no-paper", action="store_true", help="non rigenerare il paper alla fine")

    sub.add_parser("status", help="riepilogo dei risultati presenti")

    p = sub.add_parser("paper", help="genera il PDF")
    p.add_argument("--out", default=None)

    a = ap.parse_args(argv)

    if a.cmd == "prepare":
        from benchlib.datasets import prepare
        prepare(force=a.force)
        return 0

    if a.cmd == "selftest":
        return selftest()

    if a.cmd == "run":
        from benchlib.runner import run
        try:
            run(a.backend, a.models, a.benchmarks, a.preset, relock=a.relock, workers=a.workers,
                timeout=a.timeout, retries=a.retries, reserve_5h=a.reserve_5h / 100,
                reserve_weekly=a.reserve_weekly / 100,
                max_weekly=None if a.max_weekly_per_model is None else a.max_weekly_per_model / 100,
                wait=not a.no_wait)
        except KeyboardInterrupt:
            print("\nInterrotto: i risultati salvati restano validi, rilancia per riprendere.")
            return 130
        if not a.no_paper:
            from benchlib.paper import build_paper
            print(f"Paper: {build_paper()}")
        return 0

    if a.cmd == "status":
        from benchlib.analysis import print_status
        print_status()
        return 0

    if a.cmd == "paper":
        from benchlib.paper import build_paper
        print(f"Paper: {build_paper(a.out)}")
        return 0
    return 1


def selftest() -> int:
    from benchlib.datasets import BENCHMARKS, ensure_nltk, load_pool
    from benchlib.grading import gold_result, grade

    ensure_nltk()
    bad = 0
    for bench in BENCHMARKS:
        if bench == "ifeval":
            items = load_pool(bench)
            # IFEval has no reference answers: check the checker runs and rejects an empty reply.
            r = [grade(bench, it, {"text": ""})["correct"] for it in items[:30]]
            ok = not any(r)
            print(f"  {bench:<16} verificatori caricati, risposta vuota respinta: {'OK' if ok else 'ERRORE'}")
            bad += 0 if ok else 1
            continue
        items = load_pool(bench)
        fails = [it["id"] for it in items if not grade(bench, it, gold_result(bench, it))["correct"]]
        print(f"  {bench:<16} soluzioni di riferimento corrette: {len(items) - len(fails)}/{len(items)}"
              + (f"  falliti: {fails[:5]}" if fails else ""))
        bad += len(fails)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
