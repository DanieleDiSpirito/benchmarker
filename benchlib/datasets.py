"""Benchmark definitions, dataset download and the frozen, shuffled item pools.

`bench.py prepare` downloads every dataset once, shuffles it with a fixed seed and
stores it in data/frozen/<bench>.jsonl.gz (versioned). Presets take the first N
items of each pool, so every machine evaluates exactly the same problems.

LiveCodeBench hidden tests are large (~57 MB compressed): they are not versioned but
downloaded into data/cache/ the first time they are needed, and checked against the
SHA-256 recorded in the frozen pool.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import pickle
import random
import urllib.request
import zlib

from .config import DATA_DIR, FROZEN_DIR, NLTK_DIR

SEED = 2026
UA = {"User-Agent": "benchmarker/2.0"}
CACHE_DIR = DATA_DIR / "cache"
LCB_URL = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test6.jsonl"

BENCHMARKS: dict[str, dict] = {
    "livecodebench_hard": {
        "label": "LiveCodeBench v6 (hard)",
        "area": "Coding",
        "source": "livecodebench/code_generation_lite (test6, Jan-Apr 2025)",
        "metric": "pass@1 on all hidden tests",
        "verification": "program executed on every hidden test (stdin/stdout or LeetCode-style function calls)",
        "pool": None,
    },
    "hmmt_feb_2026": {
        "label": "HMMT Feb 2026",
        "area": "Math",
        "source": "MathArena/hmmt_feb_2026",
        "metric": "accuracy (symbolic equivalence)",
        "verification": "math-verify (LaTeX parsing + SymPy equivalence)",
        "pool": None,
    },
    "beyond_aime": {
        "label": "BeyondAIME",
        "area": "Math",
        "source": "ByteDance-Seed/BeyondAIME",
        "metric": "accuracy (exact integer)",
        "verification": "exact integer comparison",
        "pool": None,
    },
    "supergpqa_hard": {
        "label": "SuperGPQA (hard)",
        "area": "Graduate-level knowledge",
        "source": "m-a-p/SuperGPQA (difficulty = hard)",
        "metric": "accuracy (multiple choice, up to 10 options)",
        "verification": "exact match of the option letter",
        "pool": 300,
    },
    "bbeh_mini": {
        "label": "BIG-Bench Extra Hard (mini)",
        "area": "Reasoning",
        "source": "BBEH/bbeh (official 460-item mini split, 23 tasks)",
        "metric": "accuracy",
        "verification": "official BBEH answer normalisation and fuzzy match",
        "pool": None,
    },
    "ifbench": {
        "label": "IFBench",
        "area": "Instruction following",
        "source": "allenai/IFBench_test",
        "metric": "prompt-level strict accuracy (loose as secondary)",
        "verification": "official AllenAI programmatic verifiers (58 out-of-distribution constraints)",
        "pool": None,
    },
}

# First version of the suite (v1, 2026-11-20), kept so that archived runs can still be reported.
LEGACY_BENCHMARKS: dict[str, dict] = {
    "humaneval_plus": {"label": "HumanEval+", "area": "Coding", "verification": "EvalPlus unit tests (base + extended)"},
    "mbpp_plus": {"label": "MBPP+", "area": "Coding", "verification": "EvalPlus unit tests (base + extended)"},
    "aime_2026": {"label": "AIME 2026", "area": "Math", "verification": "exact integer 0-999"},
    "math500_l5": {"label": "MATH-500 (level 5)", "area": "Math", "verification": "math-verify symbolic equivalence"},
    "mmlu_pro": {"label": "MMLU-Pro", "area": "Knowledge", "verification": "exact match of the option letter (A-J)"},
    "ifeval": {"label": "IFEval", "area": "Instruction following", "verification": "official Google IFEval verifiers"},
}


def bench_info(bench: str) -> dict:
    """Metadata of a current or legacy benchmark."""
    return BENCHMARKS.get(bench) or LEGACY_BENCHMARKS[bench]


def is_legacy_suite(lock: dict) -> bool:
    return any(b in LEGACY_BENCHMARKS for b in lock["benchmarks"])


# Items whose own reference data is unusable (none known for the current suite).
EXCLUDED: dict[str, str] = {}


# ---------------------------------------------------------------- download helpers
def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def _load_hf(name: str, split: str) -> list[dict]:
    import pyarrow.parquet as pq

    meta = json.loads(_get(f"https://datasets-server.huggingface.co/parquet?dataset={name}"))
    files = [f for f in meta["parquet_files"] if f["split"] == split]
    if not files:
        raise RuntimeError(f"no parquet files for {name}:{split}")
    config = files[0]["config"]
    rows: list[dict] = []
    for f in files:
        if f["config"] == config:
            rows.extend(pq.read_table(io.BytesIO(_get(f["url"]))).to_pylist())
    return rows


class _NoGlobalsUnpickler(pickle.Unpickler):
    """LiveCodeBench stores hidden tests as base64(zlib(pickle(json string))).
    Only plain values may be loaded: any class or function reference is refused."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refused global {module}.{name}")


def decode_lcb_tests(blob: str) -> list[dict]:
    try:
        return json.loads(blob)
    except ValueError:
        pass
    raw = _NoGlobalsUnpickler(io.BytesIO(zlib.decompress(base64.b64decode(blob)))).load()
    if not isinstance(raw, str):
        raise ValueError("unexpected LiveCodeBench test payload")
    return json.loads(raw)


def _lcb_rows() -> tuple[list[dict], str]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / "lcb_test6.jsonl"
    if not path.exists():
        tmp = path.with_suffix(".part")
        tmp.write_bytes(_get(LCB_URL))
        tmp.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f], digest


# ---------------------------------------------------------------- normalisation
def _normalize(bench: str, row: dict) -> dict | None:
    if bench == "hmmt_feb_2026":
        return {"id": f"HMMT2026/{row['problem_idx']}", "problem": row["problem"], "gold": str(row["answer"])}
    if bench == "beyond_aime":
        return {"id": None, "problem": row["problem"], "gold": str(row["answer"]).strip()}
    if bench == "supergpqa_hard":
        if row.get("difficulty") != "hard":
            return None
        opts = row["options"]
        if isinstance(opts, str):
            import ast
            opts = ast.literal_eval(opts)
        return {"id": f"SuperGPQA/{row['uuid']}", "question": row["question"], "options": list(opts),
                "gold": row["answer_letter"], "field": row.get("field"), "discipline": row.get("discipline")}
    if bench == "bbeh_mini":
        if str(row.get("mini")) not in ("1", "True", "true"):
            return None
        return {"id": None, "task": row["task"], "input": row["input"], "gold": str(row["target"])}
    if bench == "ifbench":
        kwargs = row["kwargs"]
        ids = row["instruction_id_list"]
        if isinstance(kwargs, str):
            import ast
            kwargs, ids = ast.literal_eval(kwargs), ast.literal_eval(ids) if isinstance(ids, str) else ids
        kwargs = [{k: v for k, v in (kw or {}).items() if v is not None} for kw in kwargs]
        return {"id": f"IFBench/{row['key']}", "prompt": row["prompt"], "instruction_id_list": list(ids),
                "kwargs": kwargs}
    raise KeyError(bench)


def _assign_ids(bench: str, items: list[dict]) -> None:
    """Datasets without an id column get a content hash, stable across machines."""
    for it in items:
        if it["id"] is None:
            text = it.get("problem") or it.get("input")
            it["id"] = f"{bench}/{hashlib.sha256(text.encode()).hexdigest()[:12]}"


def frozen_path(bench: str):
    return FROZEN_DIR / f"{bench}.jsonl.gz"


def _write_pool(bench: str, items: list[dict]) -> None:
    items.sort(key=lambda x: x["id"])
    random.Random(f"{SEED}:{bench}").shuffle(items)
    if BENCHMARKS[bench]["pool"]:
        items[:] = items[: BENCHMARKS[bench]["pool"]]
    # mtime=0: identical bytes on every machine, so the pool digest is reproducible
    with open(frozen_path(bench), "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        for it in items:
            gz.write((json.dumps(it, ensure_ascii=False) + "\n").encode("utf-8"))


def prepare(benches: list[str] | None = None, force: bool = False, log=print) -> None:
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    for bench in benches or list(BENCHMARKS):
        if frozen_path(bench).exists() and not force:
            log(f"  {bench}: already frozen ({frozen_path(bench).name})")
            continue
        log(f"  {bench}: downloading {BENCHMARKS[bench]['source']} ...")
        if bench == "livecodebench_hard":
            items = _prepare_lcb(log)
        else:
            name, split = {"hmmt_feb_2026": ("MathArena/hmmt_feb_2026", "train"),
                           "beyond_aime": ("ByteDance-Seed/BeyondAIME", "test"),
                           "supergpqa_hard": ("m-a-p/SuperGPQA", "train"),
                           "bbeh_mini": ("BBEH/bbeh", "train"),
                           "ifbench": ("allenai/IFBench_test", "train")}[bench]
            items = [x for x in (_normalize(bench, r) for r in _load_hf(name, split)) if x]
            _assign_ids(bench, items)
        _write_pool(bench, items)
        log(f"  {bench}: {len(items)} items frozen")
    ensure_nltk(log)


def _prepare_lcb(log) -> list[dict]:
    rows, digest = _lcb_rows()
    items = []
    for r in rows:
        if r["difficulty"] != "hard":
            continue
        meta = json.loads(r.get("metadata") or "{}")
        items.append({
            "id": f"LCB/{r['question_id']}", "title": r["question_title"], "platform": r["platform"],
            "contest_date": r["contest_date"][:10], "content": r["question_content"],
            "starter_code": r["starter_code"], "func_name": meta.get("func_name"),
            "public_tests": json.loads(r["public_test_cases"]), "tests_sha256": digest,
        })
    log(f"  livecodebench_hard: {len(items)} hard problems out of {len(rows)}")
    return items


def lcb_hidden_tests(item: dict) -> list[dict]:
    """Hidden + public tests of one LiveCodeBench problem (downloads the source once)."""
    tests_dir = CACHE_DIR / "lcb_tests"
    path = tests_dir / (item["id"].replace("/", "_") + ".json.gz")
    if not path.exists():
        rows, digest = _lcb_rows()
        if digest != item["tests_sha256"]:
            raise RuntimeError("LiveCodeBench source file changed upstream (SHA-256 mismatch): "
                               "results would not be comparable")
        tests_dir.mkdir(parents=True, exist_ok=True)
        for r in rows:
            if r["difficulty"] != "hard":
                continue
            p = tests_dir / (f"LCB_{r['question_id']}.json.gz")
            tests = json.loads(r["public_test_cases"]) + decode_lcb_tests(r["private_test_cases"])
            with gzip.open(p, "wt", encoding="utf-8") as f:
                json.dump(tests, f)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


NLTK_RESOURCES = [("tokenizers/punkt_tab/english/", "punkt_tab"), ("corpora/stopwords", "stopwords"),
                  ("taggers/averaged_perceptron_tagger_eng/", "averaged_perceptron_tagger_eng")]


def ensure_nltk(log=print) -> None:
    import nltk

    NLTK_DIR.mkdir(parents=True, exist_ok=True)
    if str(NLTK_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DIR))
    for probe, pkg in NLTK_RESOURCES:
        try:
            nltk.data.find(probe)
        except LookupError:
            log(f"  nltk: downloading {pkg} ...")
            nltk.download(pkg, download_dir=str(NLTK_DIR), quiet=True)


def load_pool(bench: str) -> list[dict]:
    path = frozen_path(bench)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run `bench.py prepare` first")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [x for x in (json.loads(line) for line in f) if x["id"] not in EXCLUDED]


def pool_digest(bench: str) -> str:
    return hashlib.sha256(frozen_path(bench).read_bytes()).hexdigest()[:16]
