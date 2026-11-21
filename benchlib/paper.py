"""Builds paper/benchmark_paper.pdf from every result present (reportlab, no TeX needed)."""
from __future__ import annotations

import datetime as dt
import os
import tempfile
from pathlib import Path

import matplotlib
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

rl_config.useA85 = 0   # binary compressed streams: git and GitHub then treat the PDF as binary

from . import charts
from .clusters import cluster
from .analysis import MIN_STEPS, STALL_MAX_TPS, STALL_MIN_S, cache_stats, collect, escalation, item_times, model_summary, quota_text
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
GROUP_RULE = colors.HexColor("#8a8984")   # rule between the models of different plans

ST = {
    "title": ParagraphStyle("title", fontName="Serif-B", fontSize=17, leading=21, alignment=1, spaceAfter=6, textColor=INK),
    "sub": ParagraphStyle("sub", fontName="Serif", fontSize=9.5, leading=13, alignment=1, textColor=INK2, spaceAfter=14),
    "h1": ParagraphStyle("h1", keepWithNext=1, fontName="Serif-B", fontSize=12.5, leading=16, spaceBefore=12, spaceAfter=5, textColor=INK),
    "h2": ParagraphStyle("h2", keepWithNext=1, fontName="Serif-B", fontSize=10.5, leading=14, spaceBefore=8, spaceAfter=3, textColor=INK),
    "body": ParagraphStyle("body", fontName="Serif", fontSize=9.3, leading=13.2, alignment=TA_JUSTIFY, spaceAfter=5, textColor=INK),
    "abs": ParagraphStyle("abs", fontName="Serif", fontSize=9, leading=12.8, alignment=TA_JUSTIFY, leftIndent=0,
                          rightIndent=0, spaceAfter=8, textColor=INK),
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


def table(rows, widths, header_rows=1, bold_first_col=False, groups=None):
    """groups: one key per data row (e.g. the model's CLI); a rule separates rows whose key changes.
    With tuple keys a change in the first element (e.g. the benchmark) gets a stronger rule."""
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
    if groups:
        for j in range(1, len(groups)):
            a, b = groups[j - 1], groups[j]
            if a == b:
                continue
            strong = isinstance(a, tuple) and a[0] != b[0]
            style.append(("LINEABOVE", (0, header_rows + j), (-1, header_rows + j), 0.8 if strong else 0.5,
                          INK if strong else GROUP_RULE))
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


def build_paper(out: str | None = None, verbose: bool = False) -> Path:
    data = collect()
    cfg.PAPER_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(out) if out else cfg.PAPER_DIR / ("benchmark_paper_verbose.pdf" if verbose else "benchmark_paper.pdf")
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
        data["times"] = times
        data["clusters"] = cluster(models, summ)
        charts.fig_clusters(data["clusters"], summ, figs["clusters"])
        data["escalation"] = [(m, e) for m in models if (e := escalation(data, m)) and e["rerun"]]
        if data["escalation"]:
            charts.fig_escalation(data["escalation"], figs["esc"])
        Doc(out_path).build(_story(data, models, summ, pending, figs, compact=not verbose))
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


def _findings(data, models, summ) -> list[tuple[str, str]]:
    """Authors' reading of the results; every number is taken from the data, and a finding whose models
    are missing is left out."""
    by = {m["key"]: m for m in models}
    S = lambda key: summ[(by[key]["backend"], key)]
    esc = {m["key"]: e for m, e in data.get("escalation") or []}
    plan = lambda b: [m for m in models if m["backend"] == b]
    span = lambda vals, f: f"{f(min(vals))}–{f(max(vals))}"
    tok = lambda v: f"{v / 1000:.1f}k" if v >= 1000 else f"{v:.0f}"
    per = lambda s, key: s["tokens"].get(key, 0) / max(1, s["attempts"])
    out = []

    def add(fn):
        try:
            t = fn()
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return
        if t:
            out.append((fn.__name__, t))

    def agy_quota():
        cap_g = [S(m["key"])["cap_weekly"] for m in plan("agy")]
        cap_c = [S(m["key"])["cap_weekly"] for m in plan("claude")]
        slow = [m for m in plan("agy") if S(m["key"])["lat_p50"] > 25]
        rest = [S(m["key"])["lat_p50"] for m in models if m["backend"] != "agy"]
        return (f"<b>Google AI Pro has the smallest weekly allowance and the slowest models.</b> Its models give "
                f"{span(cap_g, lambda v: num(v, 0))} correct answers per week (Claude Pro {span(cap_c, lambda v: num(v, 0))}), and "
                + " and ".join(m["label"] for m in slow) + f" take a median {span([S(m['key'])['lat_p50'] for m in slow], lambda v: num(v, 0))} s "
                f"per answer, against {span(rest, lambda v: num(v, 0))} s for every Claude and GPT model.")

    def argon():
        s, e = S("gemini-4-argon"), esc["gemini-4-argon"]
        best = max(esc.values(), key=lambda x: x["acc_cumulative"])
        top = " — the highest cumulative accuracy of all models" if best is e else ""
        return (f"<b>Gemini 4 Argon skips reasoning at medium effort.</b> It reasons in only {pct(s['thinking_rate'], 0)} "
                f"of its answers ({tok(per(s, 'reasoning'))} reasoning tokens on average) and scores {pct(s['acc_macro'])}: "
                f"its adaptive reasoning judges the tasks easier than they are and saves tokens at the price of wrong "
                f"answers. At high effort it recovers {e['recovered']} of its {e['wrong']} errors and reaches "
                f"{pct(e['acc_cumulative'])}{top}, but those {e['rerun']} re-runs (half of the suite) took "
                f"{pct(e['quota']['weekly']['used'])} of the weekly limit.")

    def agy_reasoning():
        r = lambda b: [per(S(m["key"]), "reasoning") for m in plan(b) if S(m["key"])["thinking_rate"] > 0.5]
        return (f"<b>Gemini models reason the most:</b> {span(r('agy'), tok)} reasoning tokens per answer, against "
                f"{span(r('claude'), tok)} for Claude and {span(r('codex'), tok)} for GPT (counting the models that usually "
                f"reason). "
                f"This buys accuracy (Gemini 4 Neon is the most accurate model) but costs quota and time.")

    def windows():
        w = {b: 1 / data["k_ratio"][b][0] for b in ("agy", "claude", "codex") if (data["k_ratio"].get(b) or (None,))[0]}
        pairs = [("claude-opus-5-5", "gpt-6.2-astra"), ("claude-sonnet-5-5", "gpt-6.2-sol"),
                 ("claude-haiku-5-5", "gpt-6.1-terra")]
        txt = "; ".join(f"{by[a]['label'].split()[1]} {num(S(a)['cap_5h'], 0)} vs {by[b]['label'].split()[1]} "
                        f"{num(S(b)['cap_5h'], 0)}" for a, b in pairs)
        return (f"<b>Claude Pro's week is far larger than its 5-hour window.</b> One weekly limit holds about "
                f"{num(w['claude'], 0)} full 5-hour windows on Claude Pro, against {num(w['codex'], 0)} on ChatGPT Plus"
                + (f" and {num(w['agy'], 1)} on Google AI Pro" if "agy" in w else "") +
                f". Correct answers per 5-hour window, pairing the models by tier: {txt}. Weekly figures for both "
                f"plans are derived (~, section 3).")

    def luna():
        s, e = S("gpt-6-luna"), esc["gpt-6-luna"]
        return (f"<b>GPT-6 Luna is behind the other models</b> ({pct(s['acc_macro'])}, reasoning in "
                f"{pct(s['thinking_rate'], 0)} of answers). It gives about {num(s['cap_weekly'], 0)} correct answers a week, "
                f"but at its accuracy it suits simple tasks, e.g. as a sub-agent, rather than hard problems. At high effort it recovers "
                f"{e['recovered']} of {e['wrong']} errors ({pct(e['acc_cumulative'])}): as with Argon, much of the gap is "
                f"skipped reasoning.")

    def astra():
        s = S("gpt-6.2-astra")
        caps = [x["cap_weekly"] for x in summ.values() if x["cap_weekly"]]
        low = " — the lowest of all models" if s["cap_weekly"] <= min(caps) * 1.01 else ""
        return (f"<b>GPT-6.2 Astra consumes too much for everyday use.</b> {usd(s['cost'] / s['attempts'])} per answer at "
                f"API prices and {pct(10 * s['q_5h'] / s['attempts'], 1)} of the 5-hour window per 10 tasks: about "
                f"{num(s['cap_5h'], 0)} correct answers per window and {num(s['cap_weekly'], 0)} per week{low}. It is accurate "
                f"({pct(s['acc_macro'])}) but sustainable only for occasional use, e.g. as an orchestrator that "
                f"delegates to cheaper models.")

    def verbose():
        vis = {m["key"]: per(S(m["key"]), "output") - per(S(m["key"]), "reasoning") for m in models}
        cl = [vis[m["key"]] for m in plan("claude") if m["key"] != "claude-haiku-5-5"]
        others = [v for k, v in vis.items() if by[k]["backend"] != "claude"]
        return (f"<b>Claude models are verbose.</b> Opus and Sonnet write {span(cl, tok)} visible output tokens per "
                f"answer (output minus reasoning), against at most {tok(max(others))} for every Gemini and GPT model.")

    def cache():
        st = {m["key"]: cache_stats(m) for m in models}
        g = lambda b, f: [st[m["key"]][f] for m in plan(b) if st.get(m["key"]) and st[m["key"]][f] is not None]
        p = lambda v: f"{100 * v:.0f}%"
        sav = lambda b: span(sorted(abs(100 * v) for v in g(b, "saving")), lambda v: f"{v:.0f}%")
        return (f"<b>Caching is part of what a plan delivers.</b> Codex serves {span(g('codex', 'cached_share'), p)} of "
                f"its input from cache, which cuts the API cost per answer by {sav('codex')}. Claude Code hits the cache "
                f"on {span(g('claude', 'hit_rate'), p)} of answers ({span(g('claude', 'cached_share'), p)} of the input), "
                f"but in one-question sessions it also writes a new cache entry every time, billed at a premium, so on "
                f"API prices caching raises its cost by {sav('claude')}. Antigravity caches almost nothing "
                f"({span(g('agy', 'cached_share'), p)}). Since students use a model in long contiguous sessions, an "
                f"ecosystem that caches well is worth more than these one-question runs show.")

    for fn in (agy_quota, argon, agy_reasoning, windows, luna, astra, verbose, cache):
        add(fn)
    return out


def _times_comment(times, summary_ref="Table 3") -> str:
    """Figure 5 in words: models whose wrong answers come much faster or much slower than their correct ones."""
    import statistics as st

    fast, slow, same = [], [], []
    for m, items in times:
        ok = [i["latency"] for i in items if i["correct"] and not i["stalled"]]
        ko = [i["latency"] for i in items if not i["correct"] and not i["stalled"]]
        if len(ok) < 3 or len(ko) < 3:
            continue
        a, b = st.median(ok), st.median(ko)
        txt = f"{m['label']} ({b:.0f} s wrong vs {a:.0f} s correct)"
        (fast if b < a / 1.5 else slow if b > a * 1.5 else same).append(txt)
    parts = []
    if fast:
        parts.append("Some models fail <b>fast</b>: " + ", ".join(fast) + ". They give up on hard tasks after little "
                     "reasoning, so a wrong answer costs them little time; this matches the skipped reasoning seen in "
                     f"{summary_ref}.")
    if slow:
        parts.append("Others fail <b>slowly</b>: " + ", ".join(slow) + ". They reason at length and still miss, "
                     "the worst case for a student, who waits and pays quota for a wrong answer.")
    if same:
        parts.append("For " + ", ".join(t.split(" (")[0] for t in same) + " the time does not tell a wrong answer "
                     "from a right one.")
    from collections import defaultdict

    by_bench = defaultdict(list)
    for _, items in times:
        for i in items:
            if not i["stalled"]:
                by_bench[i["bench"]].append(i["latency"])
    slowest = sorted(by_bench, key=lambda b: -st.mean(by_bench[b]))[:2]
    if len(slowest) == 2:
        parts.append(f"Across all models the longest bars fall on {bench_info(slowest[0])['label']} and "
                     f"{bench_info(slowest[1])['label']} (mean {st.mean(by_bench[slowest[0]]):.0f} s and "
                     f"{st.mean(by_bench[slowest[1]]):.0f} s per task), the tasks that need the longest reasoning.")
    return " ".join(parts)


def _separation_note(data, models) -> str:
    """How poorly 6-8 questions per benchmark separate the models (IFBench as the example)."""
    from collections import Counter

    cells = [data["cells"].get((m["backend"], m["key"], "ifbench")) for m in models]
    accs = Counter(round(c["acc"], 3) for c in cells if c and c["acc"] is not None)
    if not accs:
        return ""
    acc, n_same = accs.most_common(1)[0]
    fails = Counter(i["id"] for _, items in (data.get("times") or []) for i in items
                    if i["bench"] == "ifbench" and not i["correct"])
    hard = fails.most_common(1)[0] if fails else None
    return (f"With 6–8 questions per benchmark the models are barely separated: on IFBench {n_same} of {len(cells)} "
            f"models score exactly {pct(acc, 0)}" + (f", mostly because one task ({hard[0]}) defeats {hard[1]} of "
                                                     f"{len(cells)} models" if hard and hard[1] > 1 else "") + ".")


def _q(s, w):
    return quota_text(s[f"q_{w}"], s[f"q_{w}_method"], s[f"q_{w}_bound"])


VERBOSE_LABELS = {"t_models": "1", "t_benches": "2", "t_summary": "3", "t_acc": "4", "t_detail": "5", "t_esc": "6",
                  "t_escb": "7", "t_correct": "8", "t_cache": "9", "f_value": "1", "f_quota": "2", "f_acc": "3", "f_tok": "4",
                  "f_times": "5", "f_clusters": "6", "f_esc": "7"}
COMPACT_LABELS = {"t_benches": "1", "t_summary": "2", "t_esc": "3", "t_correct": "4", "f_value": "1", "f_tok": "2",
                  "t_models": "A1", "t_sources": "A2", "t_cache": "A3", "t_escb": "A4", "t_acc": "A5", "t_detail": "A6",
                  "f_quota": "A1", "f_acc": "A2", "f_times": "A3", "f_clusters": "A4", "f_esc": "A5"}


def _story(data, models, summ, pending, figs, compact=True):
    """compact: the considerations and the tables/figures that support them, everything else in an appendix.
    verbose: every table and figure in the body (the earlier layout)."""
    L = COMPACT_LABELS if compact else VERBOSE_LABELS
    T = lambda key: f"Table {L[key]}"
    F = lambda key: f"Figure {L[key]}"
    k = lambda m: (m["backend"], m["key"])
    lock = data["lock"]
    legacy = is_legacy_suite(lock)
    efforts = sorted({m.get("effort") for m in models if m.get("effort")})
    effort_txt = " / ".join(efforts) or "default"
    benches = list(lock["benchmarks"])
    settings = {b: backend_settings(b) for b in ("agy", "claude", "codex")}
    plan_of = lambda b: settings[b].get("plan", b).split(" (")[0]
    first = min((c["first"] for c in data["cells"].values() if c["first"]), default="")[:10]
    last = max((c["last"] for c in data["cells"].values() if c["last"]), default="")[:10]
    n_items = sum(v["n"] for v in lock["benchmarks"].values())
    pw = A4[0] - 40 * mm
    lw = landscape(A4)[0] - 28 * mm
    ranked = sorted([m for m in models if summ[k(m)]["acc_macro"] is not None], key=lambda m: -summ[k(m)]["acc_macro"])
    roomy = sorted([m for m in models if summ[k(m)]["cap_weekly"]], key=lambda m: -summ[k(m)]["cap_weekly"])
    esc = data.get("escalation") or []
    findings = _findings(data, models, summ)

    # ------------------------------------------------------------ blocks
    def models_table():
        rows = [["Plan", "CLI", "Model", "Model id", "Effort", "API price $/M tokens<br/>in · cache · out", "Status"]]
        table_models = {(m["backend"], m["key"]): m for m in models} if legacy else \
            {bk: next((x for x in models if (x["backend"], x["key"]) == bk), m) for bk, m in all_models().items()}
        for (b, key), m in table_models.items():
            pr = price_for(m.get("pricing", ""), last or None)
            done = any(x["backend"] == b and x["key"] == key for x in models)
            rows.append([plan_of(b), b, m["label"], f"<font face='Mono'>{m['cli_model']}</font>", m.get("effort", "–"),
                         (f"{num(pr['input'], 2)} · {num(pr.get('cached_input', pr['input']), 3)} · {num(pr['output'], 2)}"
                          + ("" if pr.get("confidence") == "official" else "†")) if pr else "–",
                         "run" if done else "<i>pending</i>"])
        return [table(rows, [24 * mm, 12 * mm, 33 * mm, 42 * mm, 14 * mm, 32 * mm, 13 * mm],
                      groups=[b for b, _ in table_models]),
                P(f"{T('t_models')}. Plans and models. API prices only serve to express each answer's cost in dollars; "
                  "† = third-party source, to be checked against the provider's pricing page.", "cap")]

    def sources_table():
        subset = {"livecodebench_hard": "file test6.jsonl; hard problems of January–April 2025",
                  "hmmt_feb_2026": "all problems", "beyond_aime": "all problems",
                  "supergpqa_hard": "questions with difficulty = hard",
                  "bbeh_mini": "official mini split (460 items, 23 tasks)", "ifbench": "all prompts"}
        rows = [["Benchmark", "Hugging Face dataset", "Items used", "Pool SHA-256/16"]]
        for b in benches:
            sp = bench_info(b)
            name = sp.get("source", "").split(" (")[0]
            rows.append([sp["label"], f"<font face='Mono'>{name}</font>", subset.get(b, sp.get("source", "")),
                         f"<font face='Mono'>{lock['benchmarks'][b]['pool_sha256_16']}</font>"])
        return [KeepTogether([table(rows, [36 * mm, 62 * mm, 44 * mm, 28 * mm]),
                              P(f"{T('t_sources')}. Dataset sources, all public on Hugging Face "
                                "(huggingface.co/datasets/&lt;name&gt;), downloaded by <font face='Mono'>bench.py "
                                "prepare</font>. Items are normalised, shuffled with a fixed seed (2026) and frozen in "
                                "<font face='Mono'>data/frozen/</font>; the fingerprint of each frozen pool shows whether "
                                "a later download still matches. Graders: the official IFBench (AllenAI) and BBEH "
                                "(Google) scorers, vendored under Apache 2.0, and math-verify for math.", "cap")])]

    def cache_table():
        rows = [["Model", "Plan", "Input read<br/>from cache", "Answers with<br/>a cache hit", "Cache writes<br/>per answer",
                 "Cost per<br/>answer", "Without<br/>cache", "Saving"]]
        keys = []
        for m in models:
            c = cache_stats(m)
            if not c:
                continue
            keys.append(m["backend"])
            rows.append([m["label"], plan_of(m["backend"]), pct(c["cached_share"], 0), pct(c["hit_rate"], 0),
                         ktok(c["writes_per_answer"]) if c["writes_per_answer"] else "–", usd(c["cost"]),
                         usd(c["cost_no_cache"]),
                         "–" if c["saving"] is None else
                         ("0%" if round(100 * c["saving"]) == 0 else f"{100 * c['saving']:+.0f}%".replace("-", "−"))])
        return [Spacer(1, 6), KeepTogether([table(rows, [32 * mm, 24 * mm, 20 * mm, 22 * mm, 20 * mm, 18 * mm, 18 * mm,
                                                         16 * mm], bold_first_col=True, groups=keys),
                P(f"{T('t_cache')}. Caching as delivered by each ecosystem: share of input tokens served from cache, "
                  "share of answers with any cache hit, tokens written to cache (billed at a premium by Anthropic), and "
                  "the API cost per answer as measured and with no caching at all. Saving = how much cheaper caching "
                  "made each answer (negative: the cache writes cost more than the reads saved).", "cap")])]

    def benches_table():
        rows = [["Benchmark", "Area", "Tasks", "Scoring"]]
        for b in benches:
            sp = bench_info(b)
            rows.append([sp["label"], sp["area"], str(lock["benchmarks"][b]["n"]), sp["verification"]])
        return [KeepTogether([table(rows, [42 * mm, 36 * mm, 13 * mm, 79 * mm]),
                P((f"{T('t_benches')}. Benchmarks (first suite version). Generated code ran in a separate "
                   "<font face='Mono'>python -I</font> process with a 60 s limit per program." if legacy else
                   f"{T('t_benches')}. Benchmarks. LiveCodeBench problems date from January–April 2025 (hard only), HMMT "
                   "from February 2026. Code runs in an isolated process (on macOS also sandboxed: no network, no writes "
                   "outside a temporary directory), 10 s per hidden test."), "cap")])]

    bench_intro = P(("This run used the first version of the suite: widely used benchmarks whose answers can be checked "
                     "by code. Several of them proved close to saturation, which is why later runs use harder ones."
                     if legacy else
                     "Classic benchmarks (HumanEval, GSM8K, MATH, AIME 2024) are saturated, so we use recent or "
                     "deliberately hard ones whose answers can be checked by code.") + " Items are frozen with a fixed seed "
                    "and pinned in <font face='Mono'>data/suite.lock.json</font>, so results produced on different machines "
                    "cover the same tasks.")
    calling = ("Every model is called non-interactively through its plan's CLI, one question per request, in an empty "
               "temporary directory. Claude Code runs with every tool disabled; the agy and Codex agents keep their tools, "
               "but the prompt forbids them and their sandbox restricts them."
               + ("" if legacy else " The agy and Claude Code runs used one machine and account, the Codex runs another "
                  "machine, account and network: times and quotas are comparable across these two groups only in "
                  "relative terms."))

    def summary_table():
        hdr = ["Model", "Plan", "Mean<br/>acc.", "Tasks", "Cost per<br/>answer", "Cost per<br/>correct", "Input<br/>tokens",
               "Output<br/>tokens", "Reason.<br/>tokens", "With<br/>reasoning", "Median<br/>time", "p90<br/>time",
               "5-hour<br/>quota", "Weekly<br/>quota", "Correct<br/>per 5h", "Correct<br/>per week"]
        rows = [hdr]
        for m in models:
            s = summ[k(m)]
            t = s["tokens"]
            n = max(1, s["attempts"])
            p90 = [c["lat_p90"] for (b, kk, _), c in data["cells"].items() if (b, kk) == k(m) and c["lat_p90"]]
            rows.append([
                m["label"], plan_of(m["backend"]), pct(s["acc_macro"]),
                f"{s['n_done']}/{s['n_target']}", usd(s["cost"] / n if s["cost"] is not None else None),
                usd(s["cost_per_correct"]), ktok(t.get("input", 0) / n), ktok(t.get("output", 0) / n),
                ktok(t.get("reasoning", 0) / n), pct(s["thinking_rate"], 0),
                (num(s["lat_p50"], 1) + " s") if s["lat_p50"] else "–",
                (num(max(p90), 0) + " s") if p90 else "–", _q(s, "5h"), _q(s, "weekly"),
                num(s["cap_5h"], 0) if s["cap_5h"] else "–", num(s["cap_weekly"], 0) if s["cap_weekly"] else "–",
            ])
        w = [30, 24, 12, 12, 15, 15, 16, 16, 15, 16, 13, 12, 14, 14, 16, 14]
        scale = lw / (sum(w) * mm)
        return [table(rows, [x * mm * scale for x in w], bold_first_col=True, groups=[m["backend"] for m in models]),
                P(f"{T('t_summary')}. Per-model summary. Token and cost figures are means per answer (retries included); "
                  "quota is the total for the suite within the model's plan; capacity = correct answers ÷ quota share. "
                  "≈ coarse, ~ derived, &lt;1% below resolution, n/a no such limit (section 3). With reasoning = share of "
                  "answers with at least one reasoning token: at a given effort, adaptive models may skip reasoning "
                  "altogether."
                  + (f" Times exclude {sum(summ[k(m)]['stalled'] for m in models)} stalled request(s) "
                     f"(over {STALL_MIN_S} s at under {STALL_MAX_TPS:.0f} output tokens/s, i.e. queueing or rate-limit "
                     "retries, not generation)." if any(summ[k(m)]['stalled'] for m in models) else ""), "cap")]

    def acc_table():
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
        return [KeepTogether([table(rows, [38 * mm] + [cw] * len(benches) + [18 * mm], bold_first_col=True,
                                    groups=[m["backend"] for m in models]),
                              P(f"{T('t_acc')}. Accuracy per benchmark with 95% confidence interval (instruction "
                                "following: prompt-level strict accuracy).", "cap")])]

    fig_value = _fig(figs["value"], pw, f"{F('f_value')}. The student's trade-off: mean accuracy against API cost per "
                     "answer (top) and against the share of the weekly limit the whole suite used (bottom). Top left is "
                     "best in both panels. Weekly shares of Claude Pro and ChatGPT Plus are derived (~, section 3).")
    fig_quota = _fig(figs["quota"], pw, f"{F('f_quota')}. Quota consumed per 10 tasks in the model's plan (hatched: ≈ or "
                     "~ values).")
    fig_acc = _fig(figs["acc"], pw, f"{F('f_acc')}. Accuracy per benchmark with 95% confidence intervals.")
    fig_tok = KeepTogether([_img(figs["tok"], pw * 0.92),
                            P(f"{F('f_tok')}. Mean tokens per answer. Input includes the CLI's system prompt; reasoning "
                              "and visible output form the billed output.", "cap")]
                           + [B(t) for name, t in findings if name in ("agy_reasoning", "verbose")])

    def times_block():
        if not data.get("has_times"):
            return []
        return [_fig(figs["times"], pw, f"{F('f_times')}. Time per task, one bar per task in suite order and grouped by "
                     "benchmark: blue = correct, red = wrong (a failed call counts as wrong). Each panel has its own time "
                     "scale. Provider stalls are clipped and labelled with their real time; medians exclude them."),
                P(_times_comment(data["times"], T("t_summary")))]

    def clusters_block(title):
        cl = data["clusters"]
        out = [P(title, "h2")]
        if cl["k"] > 1:
            out.append(P(f"Models are grouped by consumption (cost and output tokens per answer) and speed (median time) "
                         f"with k-means on standardised log values. k-means finds <b>{cl['k']} groups</b> (silhouette "
                         f"{cl['silhouette']:.2f}; above 0.5 the groups are well separated, below 0.25 they are weak). Each "
                         "group is named after its centre relative to the median model; the tables compare models only "
                         "with their peers."))
        out.append(_fig(figs["clusters"], pw, f"{F('f_clusters')}. Consumption and speed clusters (shaded regions). "
                        "Colour shows the plan; the grouping also uses output tokens, which this projection does not show."))
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
                             (num(s_["lat_p50"], 1) + " s") if s_["lat_p50"] else "–",
                             num(s_["tps"], 0) if s_["tps"] else "–", per10("5h"), per10("weekly")])
            cap = (f"<b>{g['name']}</b>" + (f" — centre: {usd(c['cost'])} and {ktok(c['tokens'])} output tokens per "
                                            f"answer, {num(c['time'], 0)} s median time." if c else "."))
            out.append(KeepTogether([Spacer(1, 6), P(cap, "body"),
                                     table(rows, [x * mm * scale for x in w], bold_first_col=True)]))
        if cl["skipped"]:
            out.append(P("Not clustered (missing cost or time): " + ", ".join(m["label"] for m in cl["skipped"]) + ".",
                         "cap"))
        return out

    esc_intro = P("Each task a model got wrong at the main effort was re-run once at <i>high</i> effort. A large recovery "
                  "means the model knew how to solve the task and simply did not think enough; a small one means the "
                  "error is a real limit of the model. The cost of escalation is what a student pays in quota for the "
                  "second attempt.")
    fig_esc = _fig(figs["esc"], pw, f"{F('f_esc')}. Accuracy at the main effort and after re-running the wrong answers "
                   "at high effort (cumulative).") if esc else None

    def esc_table():
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
        return [KeepTogether([table(rows, [x * mm * scale for x in w], bold_first_col=True,
                                    groups=[m["backend"] for m, _ in esc]),
                              P(f"{T('t_esc')}. Escalation of wrong answers to high effort. Accuracies are means over "
                                "benchmarks; quota is the extra share used by the re-runs (≈ coarse, &lt;1% below "
                                "resolution). * = escalation not finished.", "cap")])]

    def escb_table():
        bcols = [b for b in benches if any(b in e["benches"] for _, e in esc)]
        rows = [["Model"] + [bench_info(b)["label"] for b in bcols]]
        for m, e in esc:
            rows.append([m["label"]] + [
                (f"{e['benches'][b]['recovered']} / {e['benches'][b]['wrong']}" if e["benches"].get(b, {}).get("wrong")
                 else "–") for b in bcols])
        cw = (pw - 30 * mm) / max(1, len(bcols))
        return [KeepTogether([table(rows, [30 * mm] + [cw] * len(bcols), bold_first_col=True,
                                    groups=[m["backend"] for m, _ in esc]),
                              P(f"{T('t_escb')}. Recovered / wrong answers per benchmark (– = nothing wrong at the main "
                                "effort).", "cap")])]

    def detail_table():
        hdr = ["Benchmark", "Model", "Acc.", "95% CI", "n", "Err.", "Input", "Output", "Reasoning", "Cost per<br/>answer",
               "Median<br/>time", "p90<br/>time", "5-hour<br/>quota", "Weekly<br/>quota"]
        rows, keys = [hdr], []
        for b in benches:
            for m in models:
                c = data["cells"].get((m["backend"], m["key"], b))
                if not c:
                    continue
                keys.append((b, m["backend"]))
                n = max(1, c["attempts"])
                t = c["tokens"]
                q5, qw = c["quota"]["5h"], c["quota"]["weekly"]
                rows.append([bench_info(b)["label"], m["label"], pct(c["acc"]),
                             f"{100 * c['ci'][0]:.0f}–{100 * c['ci'][1]:.0f}%", str(c["n_done"]), str(c["errors"]),
                             ktok(t.get("input", 0) / n), ktok(t.get("output", 0) / n), ktok(t.get("reasoning", 0) / n),
                             usd(c["cost"] / n if c["cost"] is not None else None),
                             (num(c["lat_p50"], 1) + " s") if c["lat_p50"] else "–",
                             (num(c["lat_p90"], 1) + " s") if c["lat_p90"] else "–",
                             quota_text(q5["used"], q5["method"], q5["bound"]),
                             quota_text(qw["used"], qw["method"], qw["bound"])])
        w = [32, 30, 13, 17, 8, 8, 15, 15, 16, 17, 15, 15, 15, 15]
        scale = lw / (sum(w) * mm)
        return [table(rows, [x * mm * scale for x in w], groups=keys),
                P(f"{T('t_detail')}. Metrics per benchmark (means per answer). Err. = requests that failed after retries, "
                  "counted wrong.", "cap")]

    def correct_table():
        good = sorted([m for m in models if summ[k(m)]["cap_weekly"]], key=lambda m: -summ[k(m)]["cap_weekly"])
        if not good:
            return []
        rows = [["Model", "Plan", "Mean<br/>acc.", "Correct per<br/>5-hour window", "Correct per<br/>week"]]
        for m in good:
            s_ = summ[k(m)]
            rows.append([m["label"], plan_of(m["backend"]), pct(s_["acc_macro"]),
                         num(s_["cap_5h"], 0) if s_["cap_5h"] else "–", num(s_["cap_weekly"], 0)])
        return [Spacer(1, 4), KeepTogether([table(rows, [38 * mm, 32 * mm, 20 * mm, 36 * mm, 30 * mm], bold_first_col=True),
                                            P(f"{T('t_correct')}. Correct answers a full limit gives, best first: quality "
                                              f"and quota in one number. ~ and ≈ values of {T('t_summary')} carry over.",
                                              "cap")])]

    def metrics(short):
        items = [
            "<b>Accuracy</b>: pass@1 with a 95% Wilson interval; the overall score is the unweighted mean over benchmarks.",
            "<b>Cost per answer</b>: what the request would cost through the API (input, cache reads and writes, output; "
            "reasoning is billed as output)." + ("" if short else " Also reported per correct answer."),
            "<b>Tokens</b>: input (with the cached part), output and reasoning as reported by the CLI, agent system "
            "prompt included. Reasoning is a subset of output, never added twice.",
            "<b>Time per answer</b>: wall-clock latency of the CLI call (median and 90th percentile). Requests that "
            f"took over {STALL_MIN_S} s while producing under {STALL_MAX_TPS:.0f} output tokens/s were waiting on the "
            "provider, not generating: they are excluded from time figures only.",
            "<b>Quota</b>: change of the 5-hour and weekly counters read from the CLI before and after each block of "
            "requests, and the resulting <b>capacity</b>: correct answers per full 5-hour window or week (wrong answers "
            "consume quota but do not count).",
            "<b>Escalation</b>: every task a model got wrong at the main effort is re-run once at <i>high</i> effort, "
            "kept apart from the main results. We report how many are recovered and the cumulative accuracy "
            "(main effort plus recovered tasks), with the extra cost, time and quota they took.",
        ]
        if not short:
            items.append("<b>Clusters</b>: models are grouped by consumption (cost and output tokens per answer) and "
                         "speed (median time) with k-means on standardised log values; the number of groups is the one "
                         "with the best silhouette.")
        out = [P("3. Metrics", "h1")] + [B(t) for t in items]
        out.append(P("Claude Code and Codex report quota in whole percentage points. A model's total is exact to ±1 "
                     "point; it is split across benchmarks by API cost. Totals spanning fewer than "
                     f"{MIN_STEPS} points are marked ≈; a window that barely moved is derived from the other one through "
                     "the account's weekly/5-hour ratio (marked ~); &lt;1% means the counter never moved. Quota is not "
                     "proportional to list prices, so it is never estimated from them. Gemini 4 Argon has only a weekly "
                     "limit (n/a for 5 hours)." + ("" if short else " The account must not be used for anything else "
                                                                     "during a run.")))
        return out

    def limitations():
        out = [P("Limitations and future work", "h2")]
        sep = _separation_note(data, models)
        if sep and compact:
            sep += f" Confidence intervals per benchmark are in {T('t_acc')}."
        for t in ["Small samples per benchmark to preserve the quota; a larger preset (e.g. <i>standard</i>) gives firmer "
                  "numbers. " + sep,
                  "Public datasets may appear in training data" + ("." if legacy else
                                                                     "; HMMT February 2026 is the most recent set."),
                  f"The main runs use <i>{effort_txt}</i> effort; only wrong answers are retried at high. A full effort "
                  "ladder (low for every task, medium for the failures, then high), reporting single and cumulative "
                  "accuracy per step, would show the cheapest effort that solves each task.",
                  "The benchmark suite was selected by Claude (Anthropic), one of the model families under test, so it "
                  "may favour tasks on which Claude models do well. In the near future we will repeat the study on "
                  "suites selected by models outside the comparison, such as DeepSeek and Kimi.",
                  "Escalation re-runs only the answers that were wrong at medium, once: the cumulative accuracy assumes "
                  "that an answer correct at medium would stay correct at high, and a single re-run cannot separate the "
                  "effect of the higher effort from run-to-run variance (re-running the same tasks at medium would).",
                  "Every task is a fresh one-question session, so only the CLI's fixed prompt can be served from cache; "
                  "in a real multi-turn session the conversation itself is cached too, and the ecosystems that cache "
                  "well gain more than measured here. In Codex the cache warms up during a run (the first requests "
                  "miss), so its share varies from run to run. Whether the plans count cached tokens less against "
                  "the quota is not published.",
                  "Some API prices come from third-party sources (†)."]:
            out.append(B(t))
        return out

    closing = P(f"Read together, {T('t_summary')}, {T('t_correct')} and {F('f_value')} answer the student's question: "
                "the best plan is the one whose models stay accurate on the tasks that matter while leaving enough weekly "
                "room. The conclusions are indicative: samples are small, quotas depend on each account's plan and state, "
                "and plans run by different people are comparable only in relative terms.")
    key_findings = [P("Key findings", "h2")] + [B(t) for name, t in findings if name not in ("agy_reasoning", "verbose")]

    # ------------------------------------------------------------ front matter
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
        "share of the 5-hour and weekly limits it consumes, from which we derive how many correct answers each plan "
        "really allows.")
    if ranked:
        abs_txt += f" The most accurate model is <b>{ranked[0]['label']}</b> ({pct(summ[k(ranked[0])]['acc_macro'])})"
        if roomy:
            abs_txt += (f"; the one giving the most correct answers per week is <b>{roomy[0]['label']}</b> "
                        f"(about {num(summ[k(roomy[0])]['cap_weekly'], 0)})")
        abs_txt += "."
    S.append(P("<b>Abstract.</b> " + abs_txt, "abs"))
    if pending:
        S.append(P("<b>Note.</b> Not yet run: " + ", ".join(m["label"] for m in pending) + ". Running "
                   "<font face='Mono'>uv run bench.py run --backend " + "|".join(sorted({m['backend'] for m in pending}))
                   + "</font> on a machine with that subscription adds them; every table and figure is rebuilt.", "abs"))
    S += [P("1. Introduction", "h1"),
          P("Each provider sells a $20/month plan (ChatGPT Plus, Claude Pro, Google AI Pro) that gives access to both a "
            "flagship model and cheaper, faster ones, each with a 5-hour and a weekly usage limit. Which plan gets the "
            "most done depends on three things: how often a model answers correctly, how much of the limit each answer "
            "consumes, and how long it takes. We measure all three on the same tasks, at the same reasoning effort "
            f"(<i>{effort_txt}</i>); how the picture changes with the effort level is left to future work.")]

    if compact:
        # ------------------------------------------------------------ compact body
        listing = "; ".join(f"<b>{plan_of(b)}</b> ({b}): " + ", ".join(m["label"] for m in models if m["backend"] == b)
                            for b in ("agy", "claude", "codex") if any(m["backend"] == b for m in models))
        S += [P("2. Setup", "h1"),
              P(f"Models under test, at <i>{effort_txt}</i> effort: {listing}. Model ids and the API prices used for "
                f"the cost figures are in {T('t_models')}. " + calling),
              bench_intro] + benches_table()
        S += metrics(short=True)
        S += [NextPageTemplate("landscape"), PageBreak(), P("4. Results", "h1"),
              P(f"{T('t_summary')} is the main result: for each model, quality, cost, tokens, time and quota over the "
                "whole suite. The appendix holds the per-benchmark results, the quota and accuracy charts, the time of "
                "every task and the consumption clusters.")] + summary_table()
        S += [NextPageTemplate("portrait"), PageBreak(), fig_value, fig_tok]
        if esc:
            S += [P("4.1 Escalating wrong answers to high effort", "h2"), esc_intro] + esc_table()
        S += [P("5. Discussion", "h1")] + key_findings + correct_table() + [closing] + limitations()

        # ------------------------------------------------------------ appendix
        S += [PageBreak(), P("Appendix", "h1"),
              P("Supporting tables and figures: models and prices, dataset sources, accuracy per benchmark, quota per "
                "task, the time of every task, consumption clusters and the per-benchmark detail of escalation and "
                "metrics.")]
        S += models_table() + [Spacer(1, 8)] + sources_table() + cache_table() + [fig_quota, fig_acc] + times_block()
        S += [PageBreak()] + clusters_block("A.1 Consumption and speed clusters")
        if esc:
            S += [PageBreak(), P("A.2 Escalation per benchmark", "h2"), fig_esc] + escb_table()
        S += [NextPageTemplate("landscape"), PageBreak(), P("A.3 Per-benchmark detail", "h2")] + acc_table() \
            + [Spacer(1, 8)] + detail_table()
        return S

    # ------------------------------------------------------------ verbose body (every table and figure inline)
    S += [P("2. Setup", "h1"), P("2.1 Plans and models", "h2"), P(calling)] + models_table()
    S += [PageBreak(), P("2.2 Benchmarks", "h2"), bench_intro] + benches_table()
    S += metrics(short=False)
    S += [NextPageTemplate("landscape"), PageBreak(), P("4. Results", "h1"),
          P(f"{T('t_summary')} is the main result: for each model, quality, cost, tokens, time and quota over the whole "
            "suite.")] + summary_table() + acc_table()
    S += [NextPageTemplate("portrait"), PageBreak(), fig_value, fig_quota, fig_acc, fig_tok] + times_block()
    S += [PageBreak()] + clusters_block("4.1 Consumption and speed clusters")
    if esc:
        S += [PageBreak(), P("4.2 Escalating wrong answers to high effort", "h2"), esc_intro, fig_esc] + esc_table() \
            + escb_table()
    S += [NextPageTemplate("landscape"), PageBreak(), P("4.3 Per-benchmark detail", "h2")] + detail_table()
    S += [NextPageTemplate("portrait"), PageBreak(), P("5. Discussion", "h1")]
    plans = {}
    for m in models:
        plans.setdefault(m["backend"], []).append(m)
    for b, ms in plans.items():
        best = max(ms, key=lambda m: summ[k(m)]["acc_macro"] or 0)
        most = max(ms, key=lambda m: summ[k(m)]["cap_weekly"] or 0)
        txt = (f"<b>{plan_of(b)}</b>: most accurate {best['label']} ({pct(summ[k(best)]['acc_macro'])}, "
               f"{usd(summ[k(best)]['cost_per_correct'])} per correct answer")
        if summ[k(most)]["cap_weekly"]:
            txt += f"); most correct answers per week {most['label']} (about {num(summ[k(most)]['cap_weekly'], 0)})"
        else:
            txt += ")"
        S.append(B(txt + "."))
    for g in data["clusters"]["groups"]:
        if len(g["members"]) > 1:
            top = max(g["members"], key=lambda m: summ[k(m)]["acc_macro"] or 0)
            S.append(B(f"Within <b>{g['name']}</b> the most accurate model is {top['label']} "
                       f"({pct(summ[k(top)]['acc_macro'])})."))
    S += key_findings + correct_table() + cache_table() + [closing] + limitations()
    return S
