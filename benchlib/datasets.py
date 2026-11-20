"""Benchmark definitions, dataset download and the frozen, shuffled item pools.

Items are downloaded once (`bench.py prepare`), shuffled with a fixed seed and
stored in data/frozen/<bench>.jsonl.gz. Presets take the first N items of each
pool, so every machine that ships with the same data folder evaluates exactly
the same problems.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import random
import urllib.request

from .config import FROZEN_DIR, NLTK_DIR

SEED = 2026

# Items whose own reference solution fails the official tests (dataset defects): never evaluated.
EXCLUDED = {
    "HumanEval/32": "il test EvalPlus chiama _poly(*float): errore anche con la soluzione canonica",
    "Mbpp/590": "confronto esatto tra float: la soluzione canonica differisce all'ultima cifra",
    "Mbpp/596": "sys.getsizeof dipende dalla versione di Python",
}
UA = {"User-Agent": "benchmarker/1.0"}

BENCHMARKS: dict[str, dict] = {
    "humaneval_plus": {
        "label": "HumanEval+",
        "area": "Coding",
        "hf": ("evalplus/humanevalplus", "test"),
        "metric": "pass@1 sui test EvalPlus (base + estesi)",
        "verification": "esecuzione dei test unitari in un processo Python isolato",
        "pool": None,
    },
    "mbpp_plus": {
        "label": "MBPP+",
        "area": "Coding",
        "hf": ("evalplus/mbppplus", "test"),
        "metric": "pass@1 sui test EvalPlus (base + estesi)",
        "verification": "esecuzione dei test unitari in un processo Python isolato",
        "pool": None,
    },
    "aime_2026": {
        "label": "AIME 2026",
        "area": "Matematica",
        "hf": ("MathArena/aime_2026", "train"),
        "metric": "accuratezza (risposta intera esatta)",
        "verification": "confronto esatto tra interi 0-999",
        "pool": None,
    },
    "math500_l5": {
        "label": "MATH-500 (livello 5)",
        "area": "Matematica",
        "hf": ("HuggingFaceH4/MATH-500", "test"),
        "metric": "accuratezza (equivalenza simbolica)",
        "verification": "math-verify (parsing LaTeX + equivalenza SymPy)",
        "pool": None,
    },
    "mmlu_pro": {
        "label": "MMLU-Pro",
        "area": "Conoscenza e ragionamento",
        "hf": ("TIGER-Lab/MMLU-Pro", "test"),
        "metric": "accuratezza (scelta multipla, 10 opzioni)",
        "verification": "confronto esatto della lettera A-J",
        "pool": 300,
    },
    "ifeval": {
        "label": "IFEval",
        "area": "Instruction following",
        "hf": ("google/IFEval", "train"),
        "metric": "prompt-level strict accuracy (loose come metrica secondaria)",
        "verification": "verificatori programmatici ufficiali Google (formato, lunghezza, parole chiave...)",
        "pool": None,
    },
}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _load_hf(name: str, split: str) -> list[dict]:
    import pyarrow.parquet as pq

    meta = json.loads(_get(f"https://datasets-server.huggingface.co/parquet?dataset={name}"))
    files = [f for f in meta["parquet_files"] if f["split"] == split]
    if not files:
        raise RuntimeError(f"nessun file parquet per {name}:{split}")
    config = files[0]["config"]
    rows: list[dict] = []
    for f in files:
        if f["config"] != config:
            continue
        rows.extend(pq.read_table(io.BytesIO(_get(f["url"]))).to_pylist())
    return rows


def _normalize(bench: str, row: dict) -> dict | None:
    if bench == "humaneval_plus":
        return {"id": row["task_id"], "prompt": row["prompt"], "entry_point": row["entry_point"],
                "test": row["test"], "gold": row["canonical_solution"]}
    if bench == "mbpp_plus":
        return {"id": f"Mbpp/{row['task_id']}", "prompt": row["prompt"], "test_list": list(row["test_list"]),
                "test_imports": list(row.get("test_imports") or []), "test": row["test"], "gold": row["code"]}
    if bench == "aime_2026":
        return {"id": f"AIME2026/{row['problem_idx']}", "problem": row["problem"], "gold": str(row["answer"])}
    if bench == "math500_l5":
        if int(row["level"]) != 5:
            return None
        return {"id": row["unique_id"], "problem": row["problem"], "gold": row["answer"], "subject": row["subject"]}
    if bench == "mmlu_pro":
        return {"id": f"MMLUPro/{row['question_id']}", "question": row["question"], "options": list(row["options"]),
                "gold": row["answer"], "category": row["category"]}
    if bench == "ifeval":
        kwargs = [{k: v for k, v in (kw or {}).items() if v is not None} for kw in row["kwargs"]]
        return {"id": f"IFEval/{row['key']}", "prompt": row["prompt"],
                "instruction_id_list": list(row["instruction_id_list"]), "kwargs": kwargs}
    raise KeyError(bench)


def frozen_path(bench: str):
    return FROZEN_DIR / f"{bench}.jsonl.gz"


def prepare(benches: list[str] | None = None, force: bool = False, log=print) -> None:
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    for bench in benches or list(BENCHMARKS):
        path = frozen_path(bench)
        if path.exists() and not force:
            log(f"  {bench}: già presente ({path.name})")
            continue
        spec = BENCHMARKS[bench]
        log(f"  {bench}: scarico {spec['hf'][0]} [{spec['hf'][1]}] ...")
        items = [x for x in (_normalize(bench, r) for r in _load_hf(*spec["hf"])) if x]
        items.sort(key=lambda x: x["id"])
        random.Random(f"{SEED}:{bench}").shuffle(items)
        if spec["pool"]:
            items = items[: spec["pool"]]
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        log(f"  {bench}: {len(items)} item salvati")
    ensure_nltk(log)


def ensure_nltk(log=print) -> None:
    import nltk

    NLTK_DIR.mkdir(parents=True, exist_ok=True)
    if str(NLTK_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DIR))
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        log("  nltk: scarico punkt_tab ...")
        nltk.download("punkt_tab", download_dir=str(NLTK_DIR), quiet=True)


def load_pool(bench: str) -> list[dict]:
    path = frozen_path(bench)
    if not path.exists():
        raise FileNotFoundError(f"{path} mancante: esegui prima `bench.py prepare`")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [x for x in (json.loads(line) for line in f) if x["id"] not in EXCLUDED]


def pool_digest(bench: str) -> str:
    return hashlib.sha256(frozen_path(bench).read_bytes()).hexdigest()[:16]
