"""Static figures for the PDF (matplotlib, Agg). Colours follow the model, never its rank."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .config import all_models  # noqa: E402
from .datasets import BENCHMARKS  # noqa: E402

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


def model_colors() -> dict[tuple[str, str], str]:
    keys = list(all_models())
    return {k: SERIES[i] if i < len(SERIES) else MUTED for i, k in enumerate(keys)}


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
            txt = f"{v:.0f}%" if c and c["acc"] is not None else "n/d"
            ax.text(v + e[1] + 3, yy, txt, va="center", fontsize=7, color=INK)
        ax.set_yticks(y, labels, fontsize=7)
        ax.set_xlim(0, 125)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", visible=False)
        ax.set_title(BENCHMARKS[bench]["label"], loc="left")
    for ax in list(axes.flat)[len(benches):]:
        ax.axis("off")
    fig.supxlabel("Accuratezza (%) con intervallo di confidenza al 95% (Wilson)", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_cost_accuracy(data, models, summaries, path):
    colors = model_colors()
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for m in models:
        s = summaries[_mk(m)]
        if s["cost"] is None or s["acc_macro"] is None or not s["attempts"]:
            continue
        x = s["cost"] / s["attempts"]
        yv = 100 * s["acc_macro"]
        ax.scatter([x], [yv], s=64, color=colors.get(_mk(m), MUTED), edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(m["label"], (x, yv), xytext=(6, 4), textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("Costo API equivalente per task (USD, scala logaritmica)")
    ax.set_ylabel("Accuratezza media (%)")
    ax.set_title("Accuratezza vs costo per task", loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_tokens(data, models, summaries, path):
    parts = [("Input non in cache", SERIES[0]), ("Input da cache", SERIES[1]),
             ("Output visibile", SERIES[2]), ("Reasoning", SERIES[3])]
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
        ax.text(left * 1.01 + 200, yy, f"{left / 1000:.1f}k".replace(".", ","), va="center", fontsize=7, color=INK)
    ax.set_yticks(y, [m["label"] for m in models], fontsize=7.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Token medi per richiesta")
    ax.legend(handles=[Patch(color=c, label=l) for l, c in parts], ncol=4, fontsize=7, frameon=False,
              loc="lower left", bbox_to_anchor=(0.0, 1.02), borderaxespad=0)
    ax.set_title("Composizione dei token per richiesta", loc="left", pad=22)
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_quota(data, models, summaries, path):
    colors = model_colors()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.9 + 0.36 * len(models)), sharey=True)
    y = list(range(len(models)))[::-1]
    for ax, w, title in ((axes[0], "5h", "Finestra di 5 ore"), (axes[1], "weekly", "Limite settimanale")):
        for yy, m in zip(y, models):
            s = summaries[_mk(m)]
            q = s[f"q_{w}"]
            if q is None or not s["attempts"]:
                ax.text(0, yy, " n/d", va="center", fontsize=7, color=MUTED)
                continue
            v = 100 * q / s["attempts"] * 10
            ax.barh(yy, v, color=colors.get(_mk(m), MUTED), height=0.6, edgecolor=SURFACE,
                    hatch="////" if s[f"q_{w}_est"] else None, linewidth=1)
            ax.text(v, yy, f"  {v:.2f}".replace(".", ","), va="center", fontsize=7, color=INK)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.25)
        ax.set_xlabel("% di quota ogni 10 task")
    axes[0].set_yticks(y, [m["label"] for m in models], fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
