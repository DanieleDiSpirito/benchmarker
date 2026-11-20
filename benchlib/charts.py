"""Static figures for the PDF (matplotlib, Agg).

Colour encodes the subscription plan (three validated hues), never a rank; every model is
named directly next to its mark, so identity never relies on colour."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import numpy as np  # noqa: E402

from .config import backend_settings  # noqa: E402
from .datasets import bench_info  # noqa: E402

# Validated categorical palette (light mode), fixed slot order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#ffffff"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.dpi": 220,
})


PLAN_COLORS = {"agy": SERIES[0], "claude": SERIES[1], "codex": SERIES[2]}


def plan_name(backend: str) -> str:
    return backend_settings(backend).get("plan", backend).split(" (")[0]


class _ByPlan(dict):
    def get(self, key, default=None):
        return PLAN_COLORS.get(key[0], default)


def model_colors():
    """(backend, key) -> colour of the model's plan."""
    return _ByPlan()


def plan_legend(fig, models, y=1.0):
    present = [b for b in PLAN_COLORS if any(m["backend"] == b for m in models)]
    fig.legend(handles=[Patch(color=PLAN_COLORS[b], label=plan_name(b)) for b in present], ncol=len(present),
               loc="upper center", bbox_to_anchor=(0.5, y), frameon=False, fontsize=7.5)


def _mk(m):
    return (m["backend"], m["key"])


def fig_accuracy(data, models, path):
    colors = model_colors()
    benches = list(data["lock"]["benchmarks"])
    cols = 3
    rows = (len(benches) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 1.0 + rows * (0.32 * len(models) + 0.9)), squeeze=False)
    for ax, bench in zip(axes.flat, benches):
        labels, vals, errs, cs = [], [], [], []
        for m in models:
            c = data["cells"].get((m["backend"], m["key"], bench))
            labels.append(m["label"])
            if c and c["acc"] is not None:
                vals.append(100 * c["acc"])
                errs.append((max(0.0, 100 * (c["acc"] - c["ci"][0])), max(0.0, 100 * (c["ci"][1] - c["acc"]))))
            else:
                vals.append(0)
                errs.append((0, 0))
            cs.append(colors.get(_mk(m), MUTED))
        y = list(range(len(models)))[::-1]
        ax.barh(y, vals, color=cs, height=0.62, edgecolor=SURFACE, linewidth=1)
        ax.errorbar(vals, y, xerr=list(zip(*errs)), fmt="none", ecolor=INK2, elinewidth=0.7, capsize=1.5)
        for yy, v, e, m in zip(y, vals, errs, models):
            c = data["cells"].get((m["backend"], m["key"], bench))
            txt = f"{v:.0f}%" if c and c["acc"] is not None else "n/a"
            ax.text(v + e[1] + 3, yy, txt, va="center", fontsize=7, color=INK)
        ax.set_yticks(y, labels, fontsize=7)
        ax.set_xlim(0, 125)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", visible=False)
        ax.set_title(bench_info(bench)["label"], loc="left")
    for ax in list(axes.flat)[len(benches):]:
        ax.axis("off")
    fig.supxlabel("Accuracy (%) with 95% Wilson confidence interval", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plan_legend(fig, models)
    fig.savefig(path)
    plt.close(fig)


def fig_value(data, models, summaries, path):
    """Accuracy against cost per answer (left) and against weekly capacity (right)."""
    colors = model_colors()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    panels = ((axes[0], lambda s: s["cost"] / s["attempts"] if s["cost"] is not None and s["attempts"] else None,
               "API-equivalent cost per answer (USD, log)", "Quality vs cost"),
              (axes[1], lambda s: s["cap_weekly"], "Tasks per full weekly limit (log)", "Quality vs usable quota"))
    pending = []
    for ax, fx, xlabel, title in panels:
        plotted, labels = 0, []
        for m in models:
            s = summaries[_mk(m)]
            x = fx(s)
            if x is None or s["acc_macro"] is None:
                continue
            yv = 100 * s["acc_macro"]
            ax.scatter([x], [yv], s=56, color=colors.get(_mk(m), MUTED), edgecolor=SURFACE, linewidth=2, zorder=3)
            labels.append((x, yv, m["label"]))
            plotted += 1
        if plotted:
            ax.set_xscale("log")
            ax.margins(x=0.4, y=0.15)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
        pending.append((ax, labels))
    axes[0].set_ylabel("Mean accuracy (%)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plan_legend(fig, models)
    for ax, labels in pending:
        place_labels(ax, labels)
    fig.savefig(path)
    plt.close(fig)


def fig_tokens(data, models, summaries, path):
    parts = [("Uncached input", SERIES[0]), ("Cached input", SERIES[1]),
             ("Visible output", SERIES[2]), ("Reasoning", SERIES[3])]
    fig, ax = plt.subplots(figsize=(6.6, 0.9 + 0.36 * len(models)))
    y = list(range(len(models)))[::-1]
    for yy, m in zip(y, models):
        s = summaries[_mk(m)]
        n = max(1, s["attempts"])
        t = s["tokens"]
        vals = [(t.get("input", 0) - t.get("cached_input", 0)) / n, t.get("cached_input", 0) / n,
                (t.get("output", 0) - t.get("reasoning", 0)) / n, t.get("reasoning", 0) / n]
        left = 0
        for v, (_, col) in zip(vals, parts):
            ax.barh(yy, v, left=left, color=col, height=0.6, edgecolor=SURFACE, linewidth=1.5)
            left += v
        ax.text(left * 1.01 + 200, yy, f"{left / 1000:.1f}k", va="center", fontsize=7, color=INK)
    ax.set_yticks(y, [m["label"] for m in models], fontsize=7.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean tokens per request")
    ax.legend(handles=[Patch(color=c, label=l) for l, c in parts], ncol=4, fontsize=7, frameon=False,
              loc="lower left", bbox_to_anchor=(0.0, 1.02), borderaxespad=0)
    ax.set_title("Token composition per request", loc="left", pad=22)
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_quota(data, models, summaries, path):
    colors = model_colors()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.9 + 0.36 * len(models)), sharey=True)
    y = list(range(len(models)))[::-1]
    for ax, w, title in ((axes[0], "5h", "5-hour window"), (axes[1], "weekly", "Weekly limit")):
        for yy, m in zip(y, models):
            s = summaries[_mk(m)]
            q = s[f"q_{w}"]
            if q is None or not s["attempts"]:
                label = f" <{100 * s[f'q_{w}_bound']:.0f}% in total" if s[f"q_{w}_bound"] else " n/a"
                ax.text(0, yy, label, va="center", fontsize=7, color=MUTED)
                continue
            v = 100 * q / s["attempts"] * 10
            derived = s[f"q_{w}_method"] in ("ratio", "cost", "coarse")
            ax.barh(yy, v, color=colors.get(_mk(m), MUTED), height=0.6, edgecolor=SURFACE,
                    hatch="////" if derived else None, linewidth=1)
            ax.text(v, yy, f"  {'~' if derived else ''}{v:.2f}", va="center", fontsize=7, color=INK)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.25)
        ax.set_xlabel("% of quota per 10 tasks")
    axes[0].set_yticks(y, [m["label"] for m in models], fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plan_legend(fig, models)
    fig.savefig(path)
    plt.close(fig)


def fig_clusters(clusters: dict, summaries, path):
    """Cost per answer vs median time; points coloured by plan, clusters drawn as shaded regions."""
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for g in clusters["groups"]:
        pts = []
        for m in g["members"]:
            s = summaries[_mk(m)]
            pts.append((np.log10(s["cost"] / max(1, s["attempts"])), np.log10(s["lat_p50"])))
        pts = np.array(pts)
        # rounded region: convex hull of small circles around every member (log space)
        ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        cloud = np.concatenate([p + 0.07 * np.c_[np.cos(ang) * 1.6, np.sin(ang)] for p in pts])
        hull = _hull(cloud)
        ax.fill(10 ** hull[:, 0], 10 ** hull[:, 1], color=GRID, alpha=0.55, zorder=1, linewidth=0)
        ax.plot(10 ** np.r_[hull[:, 0], hull[:1, 0]], 10 ** np.r_[hull[:, 1], hull[:1, 1]], color=MUTED,
                linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
        top = hull[hull[:, 1].argmax()]
        ax.text(10 ** top[0], 10 ** (top[1] + 0.03), g["name"], ha="center", va="bottom", fontsize=7.5,
                color=INK2, fontweight="bold", zorder=4)
    models = [m for g in clusters["groups"] for m in g["members"]]
    labels = []
    for m in models:
        s = summaries[_mk(m)]
        x, yv = s["cost"] / max(1, s["attempts"]), s["lat_p50"]
        ax.scatter([x], [yv], s=52, color=PLAN_COLORS.get(m["backend"], MUTED), edgecolor=SURFACE, linewidth=2, zorder=3)
        labels.append((x, yv, m["label"]))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.margins(0.18)
    ax.set_xlabel("API-equivalent cost per answer (USD, log)")
    ax.set_ylabel("Median time per answer (s, log)")
    ax.set_title("Consumption and speed clusters", loc="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plan_legend(fig, models)
    place_labels(ax, labels)
    fig.savefig(path)
    plt.close(fig)


def _hull(points: np.ndarray) -> np.ndarray:
    """Convex hull (monotone chain), counter-clockwise."""
    pts = sorted(map(tuple, points))
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def place_labels(ax, items, fontsize=6.8):
    """Direct labels that avoid each other and the points: for every (x, y, text) try a few
    offsets around the point and keep the first that collides with nothing placed so far."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    taken = [ax.transData.transform((x, y)) for x, y, _ in items]
    inside = ax.get_window_extent(r)
    boxes = []
    offsets = [(5, 3, "left", "bottom"), (5, -3, "left", "top"), (-5, 3, "right", "bottom"),
               (-5, -3, "right", "top"), (0, 7, "center", "bottom"), (0, -7, "center", "top"),
               (9, 0, "left", "center"), (-9, 0, "right", "center")]
    for x, y, text in items:
        best = None
        for dx, dy, ha, va in offsets:
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=fontsize, color=INK, zorder=5)
            bb = t.get_window_extent(r).expanded(1.04, 1.15)
            out = bb.x0 < inside.x0 or bb.x1 > inside.x1 or bb.y0 < inside.y0 or bb.y1 > inside.y1
            hit = out or any(bb.overlaps(b) for b in boxes) or any(
                bb.contains(px, py) for (px, py) in taken if (px, py) != tuple(ax.transData.transform((x, y))))
            if not hit:
                best = (t, bb)
                break
            t.remove()
        if best is None:  # everything collides: keep the first position
            dx, dy, ha, va = offsets[0]
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=fontsize, color=INK, zorder=5)
            best = (t, t.get_window_extent(r))
        boxes.append(best[1])


def fig_escalation(rows, path):
    """rows: [(model, esc)]. Medium accuracy (solid) plus what high effort recovered (hatched)."""
    fig, ax = plt.subplots(figsize=(7.0, 0.9 + 0.36 * len(rows)))
    y = list(range(len(rows)))[::-1]
    for yy, (m, e) in zip(y, rows):
        col = PLAN_COLORS.get(m["backend"], MUTED)
        base, cum = 100 * e["acc_medium"], 100 * e["acc_cumulative"]
        ax.barh(yy, base, color=col, height=0.6, edgecolor=SURFACE, linewidth=1)
        if cum > base:
            ax.barh(yy, cum - base, left=base, color=SURFACE, height=0.6, edgecolor=col, hatch="////", linewidth=1)
        ax.text(cum + 1.5, yy, f"{base:.0f}% → {cum:.0f}%  (+{cum - base:.0f})", va="center", fontsize=7, color=INK)
    ax.set_yticks(y, [m["label"] for m, _ in rows], fontsize=7.5)
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean accuracy (%): medium effort (solid) + wrong answers recovered at high effort (hatched)")
    ax.set_title("Escalating wrong answers to high effort", loc="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plan_legend(fig, [m for m, _ in rows])
    fig.savefig(path)
    plt.close(fig)


OK_COLOR, WRONG_COLOR = SERIES[0], SERIES[7]   # blue = correct, red = wrong
BENCH_SHORT = {"livecodebench_hard": "LCB", "hmmt_feb_2026": "HMMT", "beyond_aime": "BAIME",
               "supergpqa_hard": "SGPQA", "bbeh_mini": "BBEH", "ifbench": "IFB", "humaneval_plus": "HE+",
               "mbpp_plus": "MBPP+", "aime_2026": "AIME", "math500_l5": "MATH", "mmlu_pro": "MMLU-P", "ifeval": "IFEval"}


def fig_item_times(rows, path, ncols=2):
    """rows: [(model, items)] with items from analysis.item_times. One panel per model: a bar per task in
    suite order, grouped by benchmark; blue = correct, red = wrong. Each panel has its own time scale."""
    import math
    import statistics

    n = len(rows)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 0.4 + 1.45 * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    width = max(len(it) for _, it in rows)   # same bar width in every panel, even for a partial run
    for ax, (m, items) in zip(axes.flat, rows):
        x = np.arange(len(items))
        lat = np.array([it["latency"] for it in items])
        cols = [OK_COLOR if it["correct"] else WRONG_COLOR for it in items]
        ax.bar(x, [0 if it["stalled"] else v for it, v in zip(items, lat)], width=0.8, color=cols, linewidth=0)
        stalled = [i for i, it in enumerate(items) if it["stalled"]]
        normal = [v for it, v in zip(items, lat) if not it["stalled"]]
        top = max(normal or [1.0]) * 1.12
        for i in stalled:   # provider stalls: clipped to the scale of the panel and marked
            ax.bar(i, top, width=0.8, color=SURFACE, edgecolor=cols[i], hatch="////", linewidth=0.6)
            ax.text(i + 0.8, top, f"{lat[i]:.0f} s", ha="left", va="top", fontsize=5.5, color=INK2)
        ax.set_ylim(0, top)
        # benchmark separators and labels
        start = 0
        for j in range(1, len(items) + 1):
            if j == len(items) or items[j]["bench"] != items[start]["bench"]:
                if j < len(items):
                    ax.axvline(j - 0.5, color=MUTED, linewidth=0.5, zorder=0)
                ax.text((start + j - 1) / 2, -0.04, BENCH_SHORT.get(items[start]["bench"], items[start]["bench"]),
                        transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.5, color=INK2)
                start = j
        ax.set_xticks([])
        ax.set_xlim(-0.6, width - 0.4)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="y", labelsize=6)
        med = lambda ok: statistics.median(v for it, v in zip(items, lat) if it["correct"] == ok and not it["stalled"]) \
            if any(it["correct"] == ok and not it["stalled"] for it in items) else None
        mo, mw = med(True), med(False)
        sub = "  ".join(s for s in (f"median correct {mo:.0f} s" if mo is not None else "",
                                    f"wrong {mw:.0f} s" if mw is not None else "") if s)
        ax.set_title(f"{m['label']}", loc="left", fontsize=7.5)
        ax.set_title(sub, loc="right", fontsize=6, fontweight="normal", color=INK2)
        ax.set_ylabel("s", fontsize=6, rotation=0, labelpad=6)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / (0.4 + 1.45 * nrows)), h_pad=1.6)
    fig.legend(handles=[Patch(color=OK_COLOR, label="Correct"), Patch(color=WRONG_COLOR, label="Wrong"),
                        Patch(facecolor=SURFACE, edgecolor=MUTED, hatch="////", label="Provider stall (clipped)")],
               ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=7)
    fig.savefig(path)
    plt.close(fig)
