"""Groups models by consumption and speed (k-means on standardised log features).

Features per model (all per answer, log10, then z-scored):
  cost       API-equivalent cost
  tokens     output tokens (reasoning included)
  time       median wall-clock time
k is chosen in 2..min(5, n-1) by the silhouette coefficient; k-means++ initialisation with a
fixed seed and 50 restarts keeps the result deterministic.
"""
from __future__ import annotations

import math

import numpy as np

FEATURES = ("cost", "tokens", "time")
SEED = 2026


def features(models: list[dict], summaries: dict) -> tuple[list[dict], np.ndarray]:
    rows, keep = [], []
    for m in models:
        s = summaries[(m["backend"], m["key"])]
        n = max(1, s["attempts"])
        vals = (s["cost"] / n if s["cost"] else None, s["tokens"].get("output", 0) / n or None, s["lat_p50"])
        if any(v is None or v <= 0 for v in vals):
            continue
        keep.append(m)
        rows.append([math.log10(v) for v in vals])
    return keep, np.array(rows, dtype=float)


def _kmeans(x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 100) -> tuple[np.ndarray, float]:
    centers = [x[rng.integers(len(x))]]
    for _ in range(1, k):  # k-means++
        d = np.min([((x - c) ** 2).sum(1) for c in centers], axis=0)
        centers.append(x[rng.choice(len(x), p=d / d.sum())] if d.sum() > 0 else x[rng.integers(len(x))])
    c = np.array(centers)
    for _ in range(iters):
        lab = ((x[:, None, :] - c[None]) ** 2).sum(2).argmin(1)
        new = np.array([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    inertia = float(((x - c[lab]) ** 2).sum())
    return lab, inertia


def _silhouette(x: np.ndarray, lab: np.ndarray) -> float:
    d = np.sqrt(((x[:, None, :] - x[None]) ** 2).sum(2))
    scores = []
    for i in range(len(x)):
        same = lab == lab[i]
        if same.sum() <= 1:
            scores.append(0.0)
            continue
        a = d[i, same].sum() / (same.sum() - 1)
        b = min(d[i, lab == j].mean() for j in set(lab) if j != lab[i])
        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(scores))


def cluster(models: list[dict], summaries: dict) -> dict:
    """{"k", "silhouette", "groups": [{"name", "members": [model], "center": {...}}], "skipped": [model]}"""
    keep, raw = features(models, summaries)
    skipped = [m for m in models if m not in keep]
    if len(keep) < 4:
        return {"k": 1, "silhouette": None, "groups": [{"name": "All models", "members": keep, "center": {}}],
                "skipped": skipped}
    z = (raw - raw.mean(0)) / np.where(raw.std(0) > 0, raw.std(0), 1)
    best = None
    for k in range(2, min(5, len(keep) - 1) + 1):
        rng = np.random.default_rng(SEED + k)
        lab, inertia = min((_kmeans(z, k, rng) for _ in range(50)), key=lambda t: t[1])
        if len(set(lab)) < k:
            continue
        sil = _silhouette(z, lab)
        if best is None or sil > best[0] + 1e-9:
            best = (sil, k, lab)
    sil, k, lab = best
    med = np.median(raw, 0)
    groups = []
    for j in sorted(set(lab), key=lambda j: raw[lab == j, 0].mean()):   # cheapest group first
        center = raw[lab == j].mean(0)
        speed = "Quick" if center[2] < med[2] else "Slow"
        toks = "few tokens" if center[1] < med[1] else "many tokens"
        groups.append({"name": f"{speed}, {toks}",
                       "members": [m for m, l in zip(keep, lab) if l == j],
                       "center": {f: 10 ** v for f, v in zip(FEATURES, center)}})
    for i, g in enumerate(groups, 1):   # the C-number keeps two groups with the same profile apart
        g["name"] = f"C{i} · {g['name']}"
    return {"k": k, "silhouette": sil, "groups": groups, "skipped": skipped}
