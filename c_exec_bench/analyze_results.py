#!/usr/bin/env python3
"""
Step 3: Statistical Analysis, Plot Generation & Markdown Report Builder
Reads results/predictions.json and generates:
  1. report/plots/accuracy_by_tier.png        (Grouped bar chart across Tiers 1-4)
  2. report/plots/style_degradation.png       (Classic vs Randomized vs Strange C)
  3. report/plots/consistency_heatmap.png     (Triplet semantic consistency & tool violations)
  4. report/plots/latency_vs_accuracy.png     (Reasoning time vs accuracy scatter)
  5. report/benchmark_report.md               (Publication-style summary report)
"""

import json
import pathlib
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).parent
PREDICTIONS_FILE = ROOT / "results" / "predictions.json"
REPORT_DIR = ROOT / "report"
PLOTS_DIR = REPORT_DIR / "plots"
VARIANTS = ("classic_c", "randomized_c", "strange_c")
VARIANT_LABELS = {
    "classic_c": "Classic C",
    "randomized_c": "Randomized C",
    "strange_c": "Strange / Obfuscated C",
}


def compute_metrics(runs: dict) -> dict:
    summary = {}
    for model, probs in runs.items():
        tiers = sorted({p["tier"] for p in probs.values()})
        tier_acc = {t: [] for t in tiers}
        var_acc = {v: [] for v in VARIANTS}
        consistent_triplets = 0
        all_correct_triplets = 0
        total_triplets = len(probs)
        latencies = []
        violations = 0

        for pid, pdata in probs.items():
            t = pdata["tier"]
            vmap = pdata.get("variants", {})
            preds = []
            corrects = []
            for v in VARIANTS:
                rec = vmap.get(v, {})
                if rec.get("status") == "tool_violation":
                    violations += 1
                is_ok = bool(rec.get("correct", False))
                tier_acc[t].append(1.0 if is_ok else 0.0)
                var_acc[v].append(1.0 if is_ok else 0.0)
                corrects.append(is_ok)
                if rec.get("status") == "ok":
                    preds.append(rec.get("predicted", ""))
                    latencies.append(float(rec.get("elapsed_sec", 0.0)))

            if len(preds) == 3 and len(set(preds)) == 1:
                consistent_triplets += 1
            if len(corrects) == 3 and all(corrects):
                all_correct_triplets += 1

        all_scores = [x for vals in var_acc.values() for x in vals]
        summary[model] = {
            "overall_acc": 100.0 * np.mean(all_scores) if all_scores else 0.0,
            "tier_acc": {t: (100.0 * np.mean(vals) if vals else 0.0) for t, vals in tier_acc.items()},
            "var_acc": {v: (100.0 * np.mean(vals) if vals else 0.0) for v, vals in var_acc.items()},
            "consistency_rate": 100.0 * consistent_triplets / max(total_triplets, 1),
            "strict_triplet_acc": 100.0 * all_correct_triplets / max(total_triplets, 1),
            "mean_latency": float(np.mean(latencies)) if latencies else 0.0,
            "tool_violations": violations,
            "total_triplets": total_triplets,
        }
    return summary


def plot_accuracy_by_tier(summary: dict) -> None:
    models = list(summary.keys())
    tiers = sorted({t for m in summary.values() for t in m["tier_acc"].keys()})
    x = np.arange(len(tiers))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    palette = ["#2b5c8f", "#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e"]
    for idx, m in enumerate(models):
        vals = [summary[m]["tier_acc"].get(t, 0.0) for t in tiers]
        offset = (idx - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width * 0.92, label=m, color=palette[idx % len(palette)])
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.0f}%", (b.get_x() + b.get_width() / 2, h + 1.2),
                        ha="center", va="bottom", fontsize=8)

    ax.set_title("Tool-Free C Execution Accuracy by Difficulty Tier", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Tier {t}" for t in tiers], fontsize=10)
    ax.set_ylabel("Exact stdout Accuracy (%)", fontsize=10)
    ax.set_ylim(0, 112)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "accuracy_by_tier.png")
    plt.close(fig)


def plot_style_degradation(summary: dict) -> None:
    models = list(summary.keys())
    x = np.arange(len(VARIANTS))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    palette = ["#2b5c8f", "#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e"]
    for idx, m in enumerate(models):
        vals = [summary[m]["var_acc"][v] for v in VARIANTS]
        offset = (idx - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width * 0.92, label=m, color=palette[idx % len(palette)])
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.1f}%", (b.get_x() + b.get_width() / 2, h + 1.2),
                        ha="center", va="bottom", fontsize=8)

    ax.set_title("Memorization vs. Semantic Reasoning: Classic vs. Obfuscated C", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS], fontsize=10)
    ax.set_ylabel("Exact stdout Accuracy (%)", fontsize=10)
    ax.set_ylim(0, 112)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "style_degradation.png")
    plt.close(fig)


def plot_consistency_and_latency(summary: dict) -> None:
    models = list(summary.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=160)

    # Left: Semantic Consistency vs Strict All-3-Correct Triplet Rate
    x = np.arange(len(models))
    w = 0.36
    cons = [summary[m]["consistency_rate"] for m in models]
    strict = [summary[m]["strict_triplet_acc"] for m in models]
    ax1.bar(x - w / 2, cons, w, label="Triplet Self-Consistency", color="#4c78a8")
    ax1.bar(x + w / 2, strict, w, label="All-3-Variants Correct", color="#54a24b")
    ax1.set_title("Equivalence Triplet Consistency", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=20, ha="right", fontsize=8.5)
    ax1.set_ylabel("Triplets (%)", fontsize=9.5)
    ax1.set_ylim(0, 112)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(fontsize=8)

    # Right: Mean Latency vs Overall Accuracy
    palette = ["#2b5c8f", "#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e"]
    for idx, m in enumerate(models):
        lat = summary[m]["mean_latency"]
        acc = summary[m]["overall_acc"]
        ax2.scatter(lat, acc, s=110, color=palette[idx % len(palette)], zorder=4)
        ax2.annotate(m, (lat, acc), xytext=(6, 4), textcoords="offset points", fontsize=8.5)
    ax2.set_title("Inference Latency vs. Accuracy", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Mean Seconds per Program", fontsize=9.5)
    ax2.set_ylabel("Overall Accuracy (%)", fontsize=9.5)
    ax2.set_ylim(0, 108)
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "consistency_and_latency.png")
    plt.close(fig)


def write_markdown_report(summary: dict) -> None:
    lines = [
        "# Tool-Free C Program Execution Benchmark Report",
        "",
        "## 1. Methodology Overview",
        "- **Task:** Predict exact `stdout` of self-contained ANSI C11 programs without compiler, interpreter, or shell access.",
        "- **Semantic Equivalence Triplets:** Every problem is evaluated across three provably equivalent implementations (`classic_c`, `randomized_c`, `strange_c`) verified under `gcc -O0` and `clang -O2` with `-fsanitize=undefined,address`.",
        "",
        "## 2. Summary Leaderboard",
        "",
        "| Model | Overall Acc | Classic C | Randomized C | Strange C | Style Drop (Classic - Strange) | Triplet Consistency | All-3 Correct | Mean Latency | Tool Violations |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    ranked = sorted(summary.items(), key=lambda kv: kv[1]["overall_acc"], reverse=True)
    for m, s in ranked:
        drop = s["var_acc"]["classic_c"] - s["var_acc"]["strange_c"]
        lines.append(
            f"| **{m}** | {s['overall_acc']:.1f}% | {s['var_acc']['classic_c']:.1f}% | "
            f"{s['var_acc']['randomized_c']:.1f}% | {s['var_acc']['strange_c']:.1f}% | "
            f"{drop:+.1f}% | {s['consistency_rate']:.1f}% | {s['strict_triplet_acc']:.1f}% | "
            f"{s['mean_latency']:.2f}s | {s['tool_violations']} |"
        )

    lines.extend(
        [
            "",
            "## 3. Accuracy Across Difficulty Tiers",
            "",
            "| Model | Tier 1 (Scalar) | Tier 2 (Arrays/Nested) | Tier 3 (Pointers/Bitwise) | Tier 4 (Invariants/Cycles) |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
    )
    for m, s in ranked:
        t_vals = [f"{s['tier_acc'].get(t, 0.0):.1f}%" for t in (1, 2, 3, 4)]
        lines.append(f"| **{m}** | " + " | ".join(t_vals) + " |")

    lines.extend(
        [
            "",
            "## 4. Visual Analysis",
            "",
            "### Accuracy by Difficulty Tier",
            "![Accuracy by Difficulty Tier](plots/accuracy_by_tier.png)",
            "",
            "### Surface Pattern Matching vs. Deep Semantic Tracking",
            "![Style Degradation](plots/style_degradation.png)",
            "",
            "### Triplet Self-Consistency & Reasoning Latency",
            "![Consistency and Latency](plots/consistency_and_latency.png)",
            "",
        ]
    )

    report_path = REPORT_DIR / "benchmark_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report generated at {report_path}")


def main() -> None:
    if not PREDICTIONS_FILE.exists():
        print(f"[ERROR] {PREDICTIONS_FILE} not found. Run run_benchmark.py first.", file=sys.stderr)
        sys.exit(1)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(PREDICTIONS_FILE.read_text(encoding="utf-8"))
    runs = data.get("runs", {})
    if not runs:
        print("[ERROR] No model runs found in predictions.json.", file=sys.stderr)
        sys.exit(1)

    summary = compute_metrics(runs)
    plot_accuracy_by_tier(summary)
    plot_style_degradation(summary)
    plot_consistency_and_latency(summary)
    write_markdown_report(summary)


if __name__ == "__main__":
    main()
