"""Builds paper/benchmark_paper.pdf from every result present (reportlab, no TeX needed)."""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

from . import charts
from .clusters import cluster
from .analysis import MIN_STEPS, STALL_MAX_TPS, STALL_MIN_S, collect, escalation, item_times, model_summary, quota_text
from . import config as cfg
from .config import all_models, backend_settings, price_for
from .datasets import bench_info, is_legacy_suite

# ---------------------------------------------------------------- fonts & styles
_FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, file in [("Serif", "DejaVuSerif.ttf"), ("Serif-B", "DejaVuSerif-Bold.ttf"),
                   ("Serif-I", "DejaVuSerif-Italic.ttf"), ("Sans", "DejaVuSans.ttf"),
                   ("Sans-B", "DejaVuSans-Bold.ttf"), ("Mono", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(_FONT_DIR / file)))
pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-B", italic="Serif-I", boldItalic="Serif-B")
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-B", italic="Sans", boldItalic="Sans-B")

INK = colors.HexColor("#0b0b0b")
INK2 = colors.HexColor("#52514e")
ZEBRA = colors.HexColor("#f4f4f1")

ST = {
    "title": ParagraphStyle("title", fontName="Serif-B", fontSize=17, leading=21, alignment=1, spaceAfter=6, textColor=INK),
    "sub": ParagraphStyle("sub", fontName="Serif", fontSize=9.5, leading=13, alignment=1, textColor=INK2, spaceAfter=14),
    "h1": ParagraphStyle("h1", fontName="Serif-B", fontSize=12.5, leading=16, spaceBefore=12, spaceAfter=5, textColor=INK),
    "h2": ParagraphStyle("h2", fontName="Serif-B", fontSize=10.5, leading=14, spaceBefore=8, spaceAfter=3, textColor=INK),
    "body": ParagraphStyle("body", fontName="Serif", fontSize=9.3, leading=13.2, alignment=TA_JUSTIFY, spaceAfter=5, textColor=INK),
    "abs": ParagraphStyle("abs", fontName="Serif", fontSize=9, leading=12.8, alignment=TA_JUSTIFY, leftIndent=18,
                          rightIndent=18, spaceAfter=8, textColor=INK),
    "bullet": ParagraphStyle("bullet", fontName="Serif", fontSize=9.3, leading=13, leftIndent=14, bulletIndent=4,
                             spaceAfter=2, textColor=INK, alignment=TA_JUSTIFY),
    "cap": ParagraphStyle("cap", fontName="Serif-I", fontSize=8.2, leading=11, textColor=INK2, spaceBefore=3, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=7, leading=8.6, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Sans-B", fontSize=7, leading=8.6, textColor=INK),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=7.6, leading=10, leftIndent=10, textColor=INK, spaceAfter=2),
}

TITLE = "Which $20 AI Subscription Gives a Student the Most?"
SUBTITLE = ("Model quality versus usable quota on OpenAI, Anthropic and Google plans: cost per answer, "
            "token usage, response time and 5-hour / weekly limits")
AUTHORS = ["Daniele Di Spirito", "Maurizio Camposeo"]

CLI_NAMES = {"agy": "Antigravity CLI (<i>agy</i>)", "claude": "Claude Code", "codex": "OpenAI Codex CLI"}


# ---------------------------------------------------------------- number formatting
def num(v, d=1, dash="–"):
    return dash if v is None else f"{v:,.{d}f}"


def pct(v, d=1, dash="–"):
    return dash if v is None else f"{100 * v:,.{d}f}%"


def usd(v, dash="–"):
    if v is None:
        return dash
    return "$" + num(v, 4 if abs(v) < 0.1 else 3 if abs(v) < 10 else 2)


def ktok(v):
    if v is None:
        return "–"
    return f"{v / 1000:,.1f}k" if v >= 1000 else f"{v:,.0f}"


def P(t, s="body"):
    return Paragraph(t, ST[s])


def B(t):
    return Paragraph(t, ST["bullet"], bulletText="•")


def table(rows, widths, header_rows=1, bold_first_col=False):
    data = []
    for i, r in enumerate(rows):
        data.append([c if not isinstance(c, str) else Paragraph(
            c, ST["cellb"] if (i < header_rows or (bold_first_col and j == 0)) else ST["cell"]) for j, c in enumerate(r)])
    t = Table(data, colWidths=widths, repeatRows=header_rows)
    style = [
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, INK),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.6, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    for i in range(header_rows, len(rows)):
        if (i - header_rows) % 2 == 1:
            style.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(style))
    return t


class Doc(BaseDocTemplate):
    def __init__(self, path, **kw):
        super().__init__(str(path), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm,
                         bottomMargin=18 * mm, title=TITLE,
                         author=", ".join(AUTHORS), **kw)
        pw, ph = A4
        lw, lh = landscape(A4)
        self.addPageTemplates([
            PageTemplate("portrait", [Frame(20 * mm, 18 * mm, pw - 40 * mm, ph - 36 * mm, id="p")],
                         onPage=self._footer, pagesize=A4),
            PageTemplate("landscape", [Frame(14 * mm, 14 * mm, lw - 28 * mm, lh - 28 * mm, id="l")],
                         onPage=self._footer, pagesize=landscape(A4)),
        ])

    @staticmethod
    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Sans", 7)
        canvas.setFillColor(INK2)
        w, _ = canvas._pagesize
        canvas.drawCentredString(w / 2, 9 * mm, f"— {doc.page} —")
        canvas.restoreState()


def _meta(backend):
    p = cfg.RESULTS_DIR / backend / "meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def build_paper(out: str | None = None) -> Path:
    data = collect()
    cfg.PAPER_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(out) if out else cfg.PAPER_DIR / "benchmark_paper.pdf"
    if not data["lock"] or not data["models"]:
        raise SystemExit("No results to report: run `bench.py run` first.")
    models = data["models"]
    summ = {(m["backend"], m["key"]): model_summary(data, m) for m in models}
    pending = [] if is_legacy_suite(data["lock"]) else \
        [m for (b, k), m in all_models().items() if not any(x["backend"] == b and x["key"] == k for x in models)]
    with tempfile.TemporaryDirectory(prefix="bench_fig_") as figdir:
        figs = {n: os.path.join(figdir, f"{n}.png") for n in ("acc", "value", "tok", "quota", "times", "clusters", "esc")}
        charts.fig_accuracy(data, models, figs["acc"])
        charts.fig_value(data, models, summ, figs["value"])
        charts.fig_tokens(data, models, summ, figs["tok"])
        charts.fig_quota(data, models, summ, figs["quota"])
        times = [(m, it) for m in models if (it := item_times(m, data["lock"]))]
        if times:
            charts.fig_item_times(times, figs["times"])
        data["has_times"] = bool(times)
        data["clusters"] = cluster(models, summ)
        charts.fig_clusters(data["clusters"], summ, figs["clusters"])
        data["escalation"] = [(m, e) for m in models if (e := escalation(data, m)) and e["rerun"]]
        if data["escalation"]:
            charts.fig_escalation(data["escalation"], figs["esc"])
        Doc(out_path).build(_story(data, models, summ, pending, figs))
    return out_path


MAX_FIG_H = 190 * mm   # every figure, with its caption, fits on one portrait page


def _img(path, width):
    from reportlab.lib.utils import ImageReader

    iw, ih = ImageReader(path).getSize()
    h = width * ih / iw
    if h > MAX_FIG_H:
        width, h = width * MAX_FIG_H / h, MAX_FIG_H
    return Image(path, width=width, height=h)


def _fig(path, width, caption):
    return KeepTogether([_img(path, width), P(caption, "cap")])


def _q(s, w):
    return quota_text(s[f"q_{w}"], s[f"q_{w}_method"], s[f"q_{w}_bound"])


def _story(data, models, summ, pending, figs):
    k = lambda m: (m["backend"], m["key"])
    lock = data["lock"]
    legacy = is_legacy_suite(lock)
    efforts = sorted({m.get("effort") for m in models if m.get("effort")})
    effort_txt = " / ".join(efforts) or "default"
    benches = list(lock["benchmarks"])
    settings = {b: backend_settings(b) for b in ("agy", "claude", "codex")}
    metas = {b: _meta(b) for b in settings}
    present = [b for b in ("agy", "claude", "codex") if any(m["backend"] == b for m in models)]
    first = min((c["first"] for c in data["cells"].values() if c["first"]), default="")[:10]
    last = max((c["last"] for c in data["cells"].values() if c["last"]), default="")[:10]
    n_items = sum(v["n"] for v in lock["benchmarks"].values())
    pw = A4[0] - 40 * mm
    lw = landscape(A4)[0] - 28 * mm
    ranked = sorted([m for m in models if summ[k(m)]["acc_macro"] is not None], key=lambda m: -summ[k(m)]["acc_macro"])
    cheap = sorted([m for m in models if summ[k(m)]["cost_per_correct"]], key=lambda m: summ[k(m)]["cost_per_correct"])
    roomy = sorted([m for m in models if summ[k(m)]["cap_weekly"]], key=lambda m: -summ[k(m)]["cap_weekly"])

    S = [P(TITLE, "title"), P(SUBTITLE, "sub"), P("<b>" + " · ".join(AUTHORS) + "</b>", "sub"),
         P(f"Runs from {first} to {last} · generated {dt.date.today().isoformat()}", "sub")]

    abs_txt = (
        "A student choosing a $20 subscription can find endless comparisons of model quality, but quality is only "
        "half of the answer: a very strong model that can be used for five prompts a week is worth less than a weaker "
        "one that can be used all day. We compare the flagship and the fast models of the $20 plans of OpenAI, "
        f"Anthropic and Google through their own CLIs, on {len(benches)} {'' if legacy else 'hard '}benchmarks scored "
        "without any LLM judge "
        f"({n_items} tasks per model), all at <i>{effort_txt}</i> reasoning effort. For every answer we measure the cost it "
        "would have through the API, the input, output and reasoning tokens, the response time and, above all, the "
        "share of the 5-hour and weekly limits it consumes, from which we derive how many tasks each plan really allows.")
    if ranked:
        abs_txt += f" The most accurate model is <b>{ranked[0]['label']}</b> ({pct(summ[k(ranked[0])]['acc_macro'])})"
        if roomy:
            abs_txt += (f"; the one allowing the most tasks per week is <b>{roomy[0]['label']}</b> "
                        f"(about {num(summ[k(roomy[0])]['cap_weekly'], 0)})")
        abs_txt += "."
    S.append(P("<b>Abstract.</b> " + abs_txt, "abs"))
    if pending:
        S.append(P("<b>Note.</b> Not yet run: " + ", ".join(m["label"] for m in pending) + ". Running "
                   "<font face='Mono'>uv run bench.py run --backend " + "|".join(sorted({m['backend'] for m in pending}))
                   + "</font> on a machine with that subscription adds them; every table and figure is rebuilt.", "abs"))

    # 1
    S += [P("1. Introduction", "h1"),
          P("Each provider sells a $20/month plan (ChatGPT Plus, Claude Pro, Google AI Pro) that gives access to both a "
            "flagship model and cheaper, faster ones, each with a 5-hour and a weekly usage limit. Which plan gets the "
            "most done depends on three things: how often a model answers correctly, how much of the limit each answer "
            "consumes, and how long it takes. We measure all three on the same tasks, at the same reasoning effort "
            f"(<i>{effort_txt}</i>); how the picture changes with the effort level is left to future work.")]

    # 2
    S += [P("2. Setup", "h1"), P("2.1 Plans and models", "h2"),
          P("Every model is called non-interactively through its plan's CLI, one question per request, in an empty "
            "temporary directory. Claude Code runs with every tool disabled; the agy and Codex agents keep their tools, "
            "but the prompt forbids them and their sandbox restricts them.")]
    rows = [["Plan", "CLI", "Model", "Model id", "Effort", "API price $/M tokens<br/>in · cache · out", "Status"]]
    table_models = {(m["backend"], m["key"]): m for m in models} if legacy else \
        {bk: next((x for x in models if (x["backend"], x["key"]) == bk), m) for bk, m in all_models().items()}
    for (b, key), m in table_models.items():
        pr = price_for(m.get("pricing", ""), last or None)
        done = any(x["backend"] == b and x["key"] == key for x in models)
        rows.append([settings[b].get("plan", "–"), b, m["label"], f"<font face='Mono'>{m['cli_model']}</font>",
                     m.get("effort", "–"),
                     (f"{num(pr['input'], 2)} · {num(pr.get('cached_input', pr['input']), 3)} · {num(pr['output'], 2)}"
                      + ("" if pr.get("confidence") == "official" else "†")) if pr else "–",
                     "run" if done else "<i>pending</i>"])
    S += [table(rows, [33 * mm, 11 * mm, 26 * mm, 41 * mm, 13 * mm, 32 * mm, 14 * mm]),
          P("Table 1. Plans and models. API prices only serve to express each answer's cost in dollars; "
            "† = third-party source, to be checked against the provider's pricing page.", "cap")]

    S += [P("2.2 Benchmarks", "h2"),
          P(("This run used the first version of the suite: widely used benchmarks whose answers can be checked by code. "
             "Several of them proved close to saturation, which is why later runs use harder ones."
             if legacy else
             "Classic benchmarks (HumanEval, GSM8K, MATH, AIME 2024) are saturated, so we use recent or deliberately hard "
             "ones whose answers can be checked by code.") + " Items are frozen with a fixed seed and pinned in "
            "<font face='Mono'>data/suite.lock.json</font>, so results produced on different machines cover the same tasks.")]
    rows = [["Benchmark", "Area", "Tasks", "Scoring"]]
    for b in benches:
        sp = bench_info(b)
        rows.append([sp["label"], sp["area"], str(lock["benchmarks"][b]["n"]), sp["verification"]])
    S += [table(rows, [42 * mm, 36 * mm, 13 * mm, 79 * mm]),
          P(("Table 2. Benchmarks (first suite version). Generated code ran in a separate <font face='Mono'>python -I</font> "
             "process with a 60 s limit per program. " if legacy else
             "Table 2. Benchmarks. LiveCodeBench problems date from January–April 2025 (hard only), HMMT from February "
             "2026. Code runs in an isolated process (on macOS also sandboxed: no network, no writes outside a temporary "
             "directory), 10 s per hidden test. ") + "All graders are validated by <font face='Mono'>bench.py selftest</font>.",
            "cap")]

    # 3
    S += [P("3. Metrics", "h1")]
    for t in [
        "<b>Accuracy</b>: pass@1 with a 95% Wilson interval; the overall score is the unweighted mean over benchmarks.",
        "<b>Cost per answer</b>: what the request would cost through the API (input, cache reads and writes, output; "
        "reasoning is billed as output). Also reported per correct answer.",
        "<b>Tokens</b>: input (with the cached part), output and reasoning as reported by the CLI, agent system "
        "prompt included. Reasoning is a subset of output, never added twice.",
        "<b>Time per answer</b>: wall-clock latency of the CLI call (median and 90th percentile). Requests that "
        f"took over {STALL_MIN_S} s while producing under {STALL_MAX_TPS:.0f} output tokens/s were waiting on the "
        "provider (queue or rate-limit retries), not generating: they are excluded from time figures only.",
        "<b>Quota</b>: change of the 5-hour and weekly counters read from the CLI before and after each block of "
        "requests, and the resulting <b>capacity</b> (tasks per full 5-hour window or week).",
        "<b>Escalation</b>: every task a model got wrong at the main effort is re-run once at <i>high</i> effort, "
        "kept apart from the main results. We report how many are recovered and the cumulative accuracy "
        "(main effort plus recovered tasks), with the extra cost, time and quota they took.",
        "<b>Clusters</b>: models are grouped by consumption (cost and output tokens per answer) and speed (median "
        "time) with k-means on standardised log values; the number of groups is the one with the best silhouette.",
    ]:
        S.append(B(t))
    S.append(P("Claude Code and Codex report quota in whole percentage points. A model's total is exact to ±1 point; it "
               "is split across benchmarks by API cost. Totals spanning fewer than "
               f"{MIN_STEPS} points are marked ≈; a window that barely moved is derived from the other one through the "
               "account's weekly/5-hour ratio (marked ~); &lt;1% means the counter never moved. Quota is not "
               "proportional to list prices, so it is never estimated from them. Gemini 4 Argon has only a weekly limit "
               "(n/a for 5 hours). The account must not be used for anything else during a run."))

    # 4 Results
    S += [NextPageTemplate("landscape"), PageBreak(), P("4. Results", "h1"),
          P("Table 3 is the main result: for each model, quality, cost, tokens, time and quota over the whole suite.")]
    hdr = ["Model", "Plan", "Mean<br/>acc.", "Tasks", "Cost per<br/>answer", "Cost per<br/>correct", "Input<br/>tokens",
           "Output<br/>tokens", "Reason.<br/>tokens", "With<br/>reasoning", "Median<br/>time", "p90<br/>time", "5-hour<br/>quota",
           "Weekly<br/>quota", "Tasks per<br/>5h", "Tasks per<br/>week"]
    rows = [hdr]
    for m in models:
        s = summ[k(m)]
        t = s["tokens"]
        n = max(1, s["attempts"])
        p90 = [c["lat_p90"] for (b, kk, _), c in data["cells"].items() if (b, kk) == k(m) and c["lat_p90"]]
        rows.append([
            m["label"], settings[m["backend"]].get("plan", m["backend"]).split(" (")[0], pct(s["acc_macro"]),
            f"{s['n_done']}/{s['n_target']}", usd(s["cost"] / n if s["cost"] is not None else None),
            usd(s["cost_per_correct"]), ktok(t.get("input", 0) / n), ktok(t.get("output", 0) / n),
            ktok(t.get("reasoning", 0) / n), pct(s["thinking_rate"], 0),
            (num(s["lat_p50"], 1) + " s") if s["lat_p50"] else "–",
            (num(max(p90), 0) + " s") if p90 else "–", _q(s, "5h"), _q(s, "weekly"),
            num(s["cap_5h"], 0) if s["cap_5h"] else "–", num(s["cap_weekly"], 0) if s["cap_weekly"] else "–",
        ])
    w = [30, 24, 12, 12, 15, 15, 16, 16, 15, 16, 13, 12, 14, 14, 16, 14]
    scale = lw / (sum(w) * mm)
    S += [table(rows, [x * mm * scale for x in w], bold_first_col=True),
          P("Table 3. Per-model summary. Token and cost figures are means per answer (retries included); quota is the "
            "total for the suite within the model's plan; capacity = tasks ÷ quota share. ≈ coarse, ~ derived, "
            "&lt;1% below resolution, n/a no such limit (section 3). With reasoning = share of answers with at "
            "least one reasoning token: at a given effort, adaptive models may skip reasoning altogether."
            + (f" Times exclude {sum(summ[k(m)]['stalled'] for m in models)} stalled request(s) "
               f"(over {STALL_MIN_S} s at under {STALL_MAX_TPS:.0f} output tokens/s, i.e. queueing or rate-limit retries, "
               "not generation)." if any(summ[k(m)]['stalled'] for m in models) else ""), "cap")]

    rows = [["Model"] + [bench_info(b)["label"] for b in benches] + ["Mean"]]
    for m in models:
        r = [m["label"]]
        for b in benches:
            c = data["cells"].get((m["backend"], m["key"], b))
            r.append("–" if not c or c["acc"] is None else
                     f"<b>{pct(c['acc'], 0)}</b> <font size=6 color='#52514e'>[{100 * c['ci'][0]:.0f}–"
                     f"{100 * c['ci'][1]:.0f}] n={c['n_done']}</font>")
        r.append(f"<b>{pct(summ[k(m)]['acc_macro'])}</b>")
        rows.append(r)
    cw = (lw - 38 * mm - 18 * mm) / len(benches)
    S += [KeepTogether([table(rows, [38 * mm] + [cw] * len(benches) + [18 * mm], bold_first_col=True),
                        P("Table 4. Accuracy per benchmark with 95% confidence interval (instruction following: prompt-level "
                          "strict accuracy).",
                          "cap")])]

    S += [NextPageTemplate("portrait"), PageBreak(),
          _fig(figs["value"], pw, "Figure 1. The student's trade-off: mean accuracy against API cost per answer (left) "
               "and against the number of tasks a full weekly limit allows (right). Top right is best in the right panel."),
          _fig(figs["quota"], pw, "Figure 2. Quota consumed per 10 tasks in the model's plan (hatched: ≈ or ~ values)."),
          _fig(figs["acc"], pw, "Figure 3. Accuracy per benchmark with 95% confidence intervals."),
          _fig(figs["tok"], pw * 0.92, "Figure 4. Mean tokens per answer. Input includes the CLI's system prompt; "
               "reasoning and visible output form the billed output.")]
    if data.get("has_times"):
        S.append(_fig(figs["times"], pw, "Figure 5. Time per task, one bar per task in suite order and grouped by "
                      "benchmark: blue = correct, red = wrong (a failed call counts as wrong). Each panel has its own "
                      "time scale. Provider stalls are clipped and labelled with their real time; medians exclude them."))

    # 4.1 clusters
    cl = data["clusters"]
    S += [PageBreak(), P("4.1 Consumption and speed clusters", "h2")]
    if cl["k"] > 1:
        S.append(P(f"k-means finds <b>{cl['k']} groups</b> (silhouette {cl['silhouette']:.2f}; above 0.5 the groups are "
                   "well separated, below 0.25 they are weak). Each group is named after its centre relative to the median "
                   "model; the tables compare models only with their peers."))
    S.append(_fig(figs["clusters"], pw, "Figure 6. Consumption and speed clusters (shaded regions). Colour shows the plan; "
                  "the grouping also uses output tokens, which this projection does not show."))
    hdr = ["Model", "Plan", "Mean<br/>acc.", "Cost per<br/>answer", "Input<br/>tokens", "Output<br/>tokens",
           "Reasoning<br/>tokens", "Median<br/>time", "Output<br/>tok/s", "5h quota<br/>/10 tasks", "Weekly<br/>/10 tasks"]
    w = [27, 19, 11, 14, 12, 12, 18, 13, 12, 15, 15]
    scale = pw / (sum(w) * mm)
    for g in cl["groups"]:
        c = g["center"]
        rows = [hdr]
        for m in sorted(g["members"], key=lambda m: -(summ[k(m)]["acc_macro"] or 0)):
            s_ = summ[k(m)]
            n = max(1, s_["attempts"])
            t = s_["tokens"]
            per10 = lambda wdw: ("–" if s_[f"q_{wdw}"] is None else
                                 quota_text(s_[f"q_{wdw}"] / n * 10, s_[f"q_{wdw}_method"], None))
            rows.append([m["label"], charts.plan_name(m["backend"]), pct(s_["acc_macro"]),
                         usd(s_["cost"] / n if s_["cost"] is not None else None), ktok(t.get("input", 0) / n),
                         ktok(t.get("output", 0) / n), ktok(t.get("reasoning", 0) / n),
                         (num(s_["lat_p50"], 1) + " s") if s_["lat_p50"] else "–", num(s_["tps"], 0) if s_["tps"] else "–",
                         per10("5h"), per10("weekly")])
        cap = (f"<b>{g['name']}</b>" + (f" — centre: {usd(c['cost'])} and {ktok(c['tokens'])} output tokens per answer, "
                                        f"{num(c['time'], 0)} s median time." if c else "."))
        S.append(KeepTogether([Spacer(1, 6), P(cap, "body"), table(rows, [x * mm * scale for x in w], bold_first_col=True)]))
    if cl["skipped"]:
        S.append(P("Not clustered (missing cost or time): " + ", ".join(m["label"] for m in cl["skipped"]) + ".", "cap"))

    esc = data.get("escalation") or []
    if esc:
        S += [PageBreak(), P("4.2 Escalating wrong answers to high effort", "h2"),
              P("Each task a model got wrong at the main effort was re-run once at <i>high</i> effort. A large recovery "
                "means the model knew how to solve the task and simply did not think enough; a small one means the "
                "error is a real limit of the model. The cost of escalation is what a student pays in quota for the "
                "second attempt.")]
        S.append(_fig(figs["esc"], pw, "Figure 7. Accuracy at the main effort and after re-running the wrong answers at "
                      "high effort (cumulative)."))
        hdr = ["Model", "Recovered<br/>/ wrong", "Recovery<br/>rate", "Acc.<br/>main", "Acc.<br/>cumul.",
               "Cost per<br/>re-run", "Output<br/>per re-run", "Median<br/>time", "Extra<br/>5h quota", "Extra<br/>weekly"]
        rows = [hdr]
        for m, e in esc:
            n = max(1, e["attempts"])
            q5, qw = e["quota"]["5h"], e["quota"]["weekly"]
            rows.append([m["label"] + ("" if e["complete"] else " *"), f"{e['recovered']} / {e['wrong']}",
                         pct(e["recovery_rate"], 0), pct(e["acc_medium"]), pct(e["acc_cumulative"]),
                         usd(e["cost"] / n), ktok(e["tokens"].get("output", 0) / n),
                         (num(e["lat_p50"], 0) + " s") if e["lat_p50"] else "–",
                         quota_text(q5["used"], q5["method"], q5["bound"]), quota_text(qw["used"], qw["method"], qw["bound"])])
        w = [30, 17, 15, 13, 14, 15, 15, 14, 15, 15]
        scale = pw / (sum(w) * mm)
        S += [table(rows, [x * mm * scale for x in w], bold_first_col=True),
              P("Table 6. Escalation of wrong answers to high effort. Accuracies are means over benchmarks; quota is the "
                "extra share used by the re-runs (≈ coarse, &lt;1% below resolution). * = escalation not finished.",
                "cap")]
        bcols = [b for b in benches if any(b in e["benches"] for _, e in esc)]
        rows = [["Model"] + [bench_info(b)["label"] for b in bcols]]
        for m, e in esc:
            rows.append([m["label"]] + [
                (f"{e['benches'][b]['recovered']} / {e['benches'][b]['wrong']}" if e["benches"].get(b, {}).get("wrong")
                 else "–") for b in bcols])
        cw = (pw - 30 * mm) / max(1, len(bcols))
        S += [KeepTogether([table(rows, [30 * mm] + [cw] * len(bcols), bold_first_col=True),
                            P("Table 7. Recovered / wrong answers per benchmark (– = nothing wrong at the main effort).",
                              "cap")])]

    S += [NextPageTemplate("landscape"), PageBreak(), P("4.3 Per-benchmark detail", "h2")]
    hdr = ["Benchmark", "Model", "Acc.", "95% CI", "n", "Err.", "Input", "Output", "Reasoning", "Cost per<br/>answer",
           "Median<br/>time", "p90<br/>time", "5-hour<br/>quota", "Weekly<br/>quota"]
    rows = [hdr]
    for b in benches:
        for m in models:
            c = data["cells"].get((m["backend"], m["key"], b))
            if not c:
                continue
            n = max(1, c["attempts"])
            t = c["tokens"]
            q5, qw = c["quota"]["5h"], c["quota"]["weekly"]
            rows.append([bench_info(b)["label"], m["label"], pct(c["acc"]),
                         f"{100 * c['ci'][0]:.0f}–{100 * c['ci'][1]:.0f}%", str(c["n_done"]), str(c["errors"]),
                         ktok(t.get("input", 0) / n), ktok(t.get("output", 0) / n), ktok(t.get("reasoning", 0) / n),
                         usd(c["cost"] / n if c["cost"] is not None else None),
                         (num(c["lat_p50"], 1) + " s") if c["lat_p50"] else "–",
                         (num(c["lat_p90"], 1) + " s") if c["lat_p90"] else "–",
                         quota_text(q5["used"], q5["method"], q5["bound"]), quota_text(qw["used"], qw["method"], qw["bound"])])
    w = [32, 30, 13, 17, 8, 8, 15, 15, 16, 17, 15, 15, 15, 15]
    scale = lw / (sum(w) * mm)
    S += [table(rows, [x * mm * scale for x in w]),
          P("Table 5. Metrics per benchmark (means per answer). Err. = requests that failed after retries, counted wrong.",
            "cap")]

    # 5 Discussion
    S += [NextPageTemplate("portrait"), PageBreak(), P("5. Discussion", "h1")]
    plans = {}
    for m in models:
        plans.setdefault(m["backend"], []).append(m)
    for b, ms in plans.items():
        best = max(ms, key=lambda m: summ[k(m)]["acc_macro"] or 0)
        most = max(ms, key=lambda m: summ[k(m)]["cap_weekly"] or 0)
        txt = (f"<b>{settings[b].get('plan', b)}</b>: most accurate {best['label']} ({pct(summ[k(best)]['acc_macro'])}, "
               f"{usd(summ[k(best)]['cost_per_correct'])} per correct answer")
        if summ[k(most)]["cap_weekly"]:
            txt += f"); most tasks per week {most['label']} (about {num(summ[k(most)]['cap_weekly'], 0)})"
        else:
            txt += ")"
        S.append(B(txt + "."))
    for g in data["clusters"]["groups"]:
        if len(g["members"]) > 1:
            top = max(g["members"], key=lambda m: summ[k(m)]["acc_macro"] or 0)
            S.append(B(f"Within <b>{g['name']}</b> the most accurate model is {top['label']} "
                       f"({pct(summ[k(top)]['acc_macro'])})."))
    S.append(P("Read together, Table 3 and Figure 1 answer the student's question: the best plan is the one whose "
               "models stay accurate on the tasks that matter while leaving enough weekly room. The conclusions are "
               "indicative: samples are small (see the intervals), quotas depend on each account's plan and state, and "
               "plans run by different people are comparable only in relative terms."))
    S += [P("Limitations and future work", "h2")]
    for t in ["Small samples per benchmark to preserve the quota; a larger preset (e.g. <i>standard</i>) gives firmer numbers.",
              "Public datasets may appear in training data" + ("." if legacy else
                                                                 "; HMMT February 2026 is the most recent set."),
              f"The main runs use <i>{effort_txt}</i> effort; only wrong answers are retried at high. A full effort "
              "ladder (low for every task, medium for the failures, then high), reporting single and cumulative "
              "accuracy per step, would show the cheapest effort that solves each task.",
              "The benchmark suite was selected by Claude (Anthropic), one of the model families under test, so it "
              "may favour tasks on which Claude models do well. In the near future we will repeat the study on "
              "suites selected by models outside the comparison, such as DeepSeek and Kimi.",
              "Some API prices come from third-party sources (†)."]:
        S.append(B(t))

    S += [P("Appendix: reproducing and extending", "h1"),
          P("Code, frozen data and raw results (full reply, tokens and score of every task) are in the repository:")]
    for c in ["uv run bench.py selftest", "uv run bench.py run --backend agy|claude|codex", "uv run bench.py paper"]:
        S.append(P(c, "code"))
    vers = [f"{b} {metas[b]['runs'][-1]['cli_version']}" for b in present if metas.get(b)]
    S.append(P(("CLI versions: " + "; ".join(vers) + ". " if vers else "") + "Pool fingerprints (SHA-256/16): "
               + "; ".join(f"{bench_info(b)['label']} {lock['benchmarks'][b]['pool_sha256_16']}" for b in benches)
               + f". Prices checked on {data['pricing'].get('checked_on', '?')}."))
    return S
