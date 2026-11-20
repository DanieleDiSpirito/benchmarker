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
from .analysis import collect, model_summary
from .config import PAPER_DIR, RESULTS_DIR, all_models, price_for
from .datasets import BENCHMARKS, EXCLUDED

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
RULE = colors.HexColor("#b8b7b0")
ZEBRA = colors.HexColor("#f4f4f1")
ACCENT = colors.HexColor("#2a78d6")

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


# ---------------------------------------------------------------- number formatting (Italian)
def num(v, d=1, dash="–"):
    if v is None:
        return dash
    s = f"{v:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(v, d=1, dash="–"):
    return dash if v is None else num(100 * v, d) + "%"


def usd(v, dash="–"):
    if v is None:
        return dash
    d = 4 if abs(v) < 0.1 else 3 if abs(v) < 10 else 2
    return "$" + num(v, d)


def ktok(v):
    if v is None:
        return "–"
    return num(v / 1000, 1) + "k" if v >= 1000 else num(v, 0)


def P(t, s="body"):
    return Paragraph(t, ST[s])


def table(rows, widths, header_rows=1, align_right_from=1, zebra=True, bold_first_col=False):
    data = []
    for i, r in enumerate(rows):
        st = "cellb" if i < header_rows else "cell"
        data.append([c if not isinstance(c, str) else Paragraph(
            c, ST["cellb"] if (i < header_rows or (bold_first_col and j == 0)) else ST[st]) for j, c in enumerate(r)])
    t = Table(data, colWidths=widths, repeatRows=header_rows)
    style = [
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, INK),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.6, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    if zebra:
        for i in range(header_rows, len(rows)):
            if (i - header_rows) % 2 == 1:
                style.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(style))
    return t


# ---------------------------------------------------------------- document
class Doc(BaseDocTemplate):
    def __init__(self, path, **kw):
        super().__init__(str(path), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm,
                         bottomMargin=18 * mm, title="Benchmark LLM in modalità abbonamento",
                         author="benchmarker", **kw)
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
    p = RESULTS_DIR / backend / "meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def build_paper(out: str | None = None) -> Path:
    data = collect()
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(out) if out else PAPER_DIR / "benchmark_paper.pdf"
    if not data["lock"] or not data["models"]:
        raise SystemExit("Nessun risultato da riportare: esegui prima `bench.py run`.")
    models = data["models"]
    summ = {(m["backend"], m["key"]): model_summary(data, m) for m in models}
    configured = all_models()
    pending = [m for (b, k), m in configured.items() if not any(x["backend"] == b and x["key"] == k for x in models)]
    lock = data["lock"]
    benches = list(lock["benchmarks"])

    with tempfile.TemporaryDirectory(prefix="bench_fig_") as figdir:
        figs = {
            "acc": os.path.join(figdir, "acc.png"), "cost": os.path.join(figdir, "cost.png"),
            "tok": os.path.join(figdir, "tok.png"), "quota": os.path.join(figdir, "quota.png"),
        }
        charts.fig_accuracy(data, models, figs["acc"])
        charts.fig_cost_accuracy(data, models, summ, figs["cost"])
        charts.fig_tokens(data, models, summ, figs["tok"])
        charts.fig_quota(data, models, summ, figs["quota"])
        story = _story(data, models, summ, pending, lock, benches, figs)
        Doc(out_path).build(story)
    return out_path


def _img(path, width):
    from reportlab.lib.utils import ImageReader

    iw, ih = ImageReader(path).getSize()
    return Image(path, width=width, height=width * ih / iw)


def _story(data, models, summ, pending, lock, benches, figs):
    k = lambda m: (m["backend"], m["key"])
    today = dt.date.today().isoformat()
    metas = {b: _meta(b) for b in ("agy", "claude", "codex")}
    cli_names = {"agy": "Antigravity <i>agy</i>", "claude": "Claude Code", "codex": "OpenAI <i>Codex</i>"}
    backends_present = sorted({m["backend"] for m in models})
    first = min((c["first"] for c in data["cells"].values() if c["first"]), default="")[:10]
    last = max((c["last"] for c in data["cells"].values() if c["last"]), default="")[:10]
    n_items = sum(v["n"] for v in lock["benchmarks"].values())
    pw = A4[0] - 40 * mm
    lw = landscape(A4)[0] - 28 * mm

    ranked = sorted([m for m in models if summ[k(m)]["acc_macro"] is not None],
                    key=lambda m: -summ[k(m)]["acc_macro"])
    best = ranked[0] if ranked else None
    cheap = sorted([m for m in models if summ[k(m)]["cost_per_correct"]], key=lambda m: summ[k(m)]["cost_per_correct"])

    S = []
    S += [P("Benchmark deterministico di modelli linguistici in modalità abbonamento", "title"),
          P("Accuratezza su coding, matematica, conoscenza e instruction following, con consumo di token, "
            "costo API equivalente e quota delle finestre di 5 ore e settimanale", "sub"),
          P(f"Generato il {today} · esecuzioni dal {first} al {last} · backend: {', '.join(backends_present)}", "sub")]

    # Abstract
    abs_txt = (
        f"Valutiamo {len(models)} modelli attraverso le CLI in abbonamento "
        f"({', '.join(cli_names[b] for b in backends_present)}) "
        f"su {len(benches)} benchmark pubblici ({n_items} item per modello, preset <i>{lock['preset']}</i>), "
        "tutti verificabili in modo deterministico: test unitari eseguiti, confronto esatto o simbolico delle "
        "risposte, verificatori programmatici. Nessun punteggio dipende da un LLM-as-a-Judge. Per ogni richiesta "
        "registriamo i token di input (con la parte servita da cache), di output e di reasoning, stimiamo il costo "
        "che lo stesso traffico avrebbe via API e misuriamo la frazione consumata della quota di 5 ore e settimanale "
        "dell'abbonamento, leggendola dalla CLI prima e dopo ogni blocco di richieste.")
    if best:
        sb = summ[k(best)]
        abs_txt += (f" Il modello con l'accuratezza media più alta è <b>{best['label']}</b> ({pct(sb['acc_macro'])}); ")
        if cheap:
            sc = summ[k(cheap[0])]
            abs_txt += f"il costo API più basso per risposta corretta è di <b>{cheap[0]['label']}</b> ({usd(sc['cost_per_correct'])})."
    S += [P("<b>Sommario.</b> " + abs_txt, "abs")]
    if pending:
        S += [P("<b>Nota.</b> Modelli configurati ma non ancora eseguiti: " + ", ".join(m["label"] for m in pending)
                + ". Per aggiungerli basta eseguire <font face='Mono'>uv run bench.py run --backend "
                + "/".join(sorted({m['backend'] for m in pending})) + "</font> su una macchina che ha accesso a quella "
                "CLI: le tabelle e i grafici di questo documento vengono rigenerati includendo anche i nuovi risultati.", "abs")]

    # 1. Introduzione
    S += [P("1. Introduzione", "h1"),
          P("Le classifiche pubbliche misurano i modelli via API, con i prompt e le impostazioni scelti da chi le "
            "compila. Chi usa i modelli tramite un abbonamento si pone però domande diverse: quanto rende ciascun "
            "modello sulle stesse attività, quanti token consuma davvero passando per l'agente della CLI, quanto "
            "costerebbe lo stesso lavoro via API e, soprattutto, quanta parte della quota a disposizione si esaurisce. "
            "Questo documento risponde a tutte e quattro le domande con un'unica procedura riproducibile. Le sezioni "
            "2 e 3 descrivono setup e metriche; i risultati sono nella sezione 5, dopo una discussione dei limiti.")]

    # 2. Setup
    S += [P("2. Setup sperimentale", "h1"), P("2.1 Modelli e interfacce", "h2"),
          P("Ogni modello viene invocato in modalità non interattiva attraverso la CLI del proprio abbonamento, con "
            "lo stesso livello di ragionamento (<i>effort</i>) per tutti. Ogni richiesta parte in una directory "
            "temporanea vuota: un'unica domanda, nessuno storico di conversazione, nessun file del progetto visibile.")]
    S += [P("<b>Antigravity (agy).</b> <font face='Mono'>agy -p &lt;prompt&gt; --model &lt;id&gt; --output-format json "
            "--sandbox --disable-slash-commands [--json-schema]</font>. Il JSON restituito contiene la risposta, l'output "
            "strutturato e il campo <i>usage</i> (input, output, thinking, cache read).", "bullet", ),
          P("<b>Claude Code.</b> <font face='Mono'>claude -p &lt;prompt&gt; --model &lt;id&gt; --effort &lt;effort&gt; "
            "--output-format stream-json --verbose --tools \"\" --strict-mcp-config --setting-sources \"\" "
            "--disable-slash-commands --no-session-persistence [--json-schema]</font>. Nessuno strumento, nessun server "
            "MCP, nessuna impostazione o CLAUDE.md dell'utente: il modello riceve solo il prompt di sistema minimo di "
            "Claude Code. L'evento finale <i>result</i> contiene i token per modello (input, lettura e scrittura della "
            "cache, output, thinking), l'evento <i>rate_limit_event</i> l'utilizzo delle finestre di 5 ore e di 7 giorni.",
            "bullet"),
          P("<b>Codex.</b> <font face='Mono'>codex exec --json --sandbox read-only --skip-git-repo-check -m &lt;id&gt; "
            "-c model_reasoning_effort=&lt;effort&gt; [--output-schema]</font>. Token e rate limit vengono letti "
            "dagli eventi <i>token_count</i> del file di sessione che Codex scrive in "
            "<font face='Mono'>~/.codex/sessions</font>.", "bullet")]
    rows = [["Modello", "Backend", "ID nella CLI", "Effort", "Prezzo API $/M token<br/>input · cache · output",
             "Fonte prezzo", "Stato"]]
    for (b, key), m in all_models().items():
        pr = price_for(m.get("pricing", ""), last or None)
        done = any(x["backend"] == b and x["key"] == key for x in models)
        rows.append([m["label"], b, f"<font face='Mono'>{m['cli_model']}</font>", m.get("effort", "–"),
                     f"{num(pr['input'], 2)} · {num(pr.get('cached_input', pr['input']), 3)} · {num(pr['output'], 2)}" if pr else "–",
                     ("ufficiale" if pr and pr.get("confidence") == "official" else "fonti terze") if pr else "–",
                     "eseguito" if done else "<i>in attesa</i>"])
    S += [table(rows, [30 * mm, 14 * mm, 37 * mm, 12 * mm, 34 * mm, 21 * mm, 17 * mm]),
          P("Tabella 1. Modelli valutati. I prezzi servono solo a stimare il costo equivalente via API: "
            "l'abbonamento non li addebita. Se un prezzo cambia nel periodo di esecuzione, ogni richiesta usa quello "
            "valido nel giorno in cui è stata eseguita.", "cap")]
    vers = [f"{b}: {metas[b]['runs'][-1]['cli_version']}" for b in backends_present if metas.get(b)]
    if vers:
        S += [P("Versioni delle CLI usate nell'ultima esecuzione: " + "; ".join(vers) + ".")]

    S += [P("2.2 Benchmark", "h2"),
          P("Abbiamo scelto benchmark diffusi in letteratura i cui punteggi si calcolano senza alcun giudizio "
            "soggettivo. I dataset vengono scaricati una sola volta da Hugging Face, mescolati con un seme fisso e "
            "congelati in <font face='Mono'>data/frozen</font>. Il preset prende i primi N item di ogni pool, e il file "
            "<font face='Mono'>data/suite.lock.json</font> fissa gli identificativi esatti: chi estende i risultati "
            "su un altro computer valuta quindi gli stessi problemi.")]
    rows = [["Benchmark", "Area", "Item", "Sorgente", "Metrica", "Verifica"]]
    for b in benches:
        sp = BENCHMARKS[b]
        rows.append([sp["label"], sp["area"], str(lock["benchmarks"][b]["n"]),
                     f"<font face='Mono'>{sp['hf'][0]}</font>", sp["metric"], sp["verification"]])
    S += [table(rows, [24 * mm, 23 * mm, 10 * mm, 37 * mm, 37 * mm, 39 * mm]),
          P("Tabella 2. Benchmark e metodo di verifica. AIME 2026 (febbraio 2026) e il sottoinsieme di livello 5 di "
            "MATH-500 servono a distinguere modelli che su GSM8K sarebbero tutti vicini al 100%. "
            f"Item esclusi perché la loro soluzione di riferimento non supera i test ufficiali: "
            f"{', '.join(EXCLUDED)}.", "cap")]

    S += [P("2.3 Protocollo", "h2"),
          P("Il prompt è identico per tutti i modelli. Chiede di rispondere senza usare strumenti (niente esecuzione di "
            "codice, file o ricerche) e, dove serve, impone uno schema JSON di output: il campo <i>answer</i> per "
            "matematica e scelta multipla, il campo <i>code</i> per il coding. Lo schema elimina gli errori di "
            "estrazione della risposta senza aiutare il modello. In IFEval la risposta è testo libero, perché è proprio "
            "il formato a essere valutato. Il codice generato viene eseguito in un processo Python isolato "
            "(<font face='Mono'>python -I</font>, directory temporanea, ambiente ridotto, timeout di 60 s) con le "
            "suite di test estese di EvalPlus. Le risposte matematiche sono confrontate come interi (AIME) o per "
            "equivalenza simbolica con <i>math-verify</i> (MATH). IFEval usa i verificatori ufficiali di Google "
            "Research inclusi nel progetto. Prima di ogni campagna, <font face='Mono'>bench.py selftest</font> "
            "controlla che tutte le soluzioni di riferimento ottengano il punteggio pieno.")]
    S += [P("Le richieste di un modello vengono eseguite in parallelo (3 per volta), un modello alla volta. Una "
            "richiesta fallita (errore della CLI o timeout) viene ripetuta fino a due volte; se fallisce sempre, conta "
            "come risposta errata. Tutti i tentativi, ripetizioni comprese, entrano nel conteggio di token, costi e "
            "quota, perché li consumano davvero. Ogni risposta viene salvata appena arriva, quindi un'esecuzione "
            "interrotta riprende esattamente da dove si era fermata.")]

    # 3. Metriche
    S += [P("3. Metriche", "h1"), P("3.1 Accuratezza e statistiche", "h2"),
          P("Per ogni coppia modello-benchmark riportiamo l'accuratezza pass@1 (un solo tentativo, nessun campionamento "
            "multiplo) con un intervallo di confidenza al 95% di Wilson. Con campioni piccoli l'intervallo è ampio: "
            "due modelli i cui intervalli si sovrappongono molto non vanno considerati diversi. L'accuratezza "
            "complessiva è la media non pesata dei benchmark (<i>macro</i>), così ogni area pesa allo stesso modo. "
            "Riportiamo anche la latenza (mediana e 90° percentile, misurata lato client e comprensiva "
            "dell'avvio della CLI), la velocità di generazione (token di output al secondo, mediana) e il numero "
            "di errori.")]
    S += [P("3.2 Token", "h2"),
          P("I token sono quelli dichiarati dalla CLI per l'intera richiesta, quindi includono il prompt di sistema e "
            "le definizioni degli strumenti dell'agente: è ciò che l'abbonamento conta davvero, non solo il testo del "
            "benchmark. <b>Input</b> comprende i token letti dalla cache, riportati anche a parte. <b>Output</b> "
            "comprende i token di <b>reasoning</b>, che entrambe le CLI includono nel conteggio di output e che i "
            "provider fatturano come output. Il reasoning viene quindi mostrato come sottoinsieme e mai sommato due "
            "volte. Se una CLI non espone i token di thinking (Claude Code li riporta solo in alcuni casi), il "
            "reasoning risulta 0 ma è comunque compreso nell'output. Claude Code scrive in cache il proprio prompt di "
            "sistema (cache con durata di 1 ora): questi token di <i>scrittura</i> sono parte dell'input e hanno un "
            "prezzo maggiorato.")]
    S += [P("3.3 Costo API equivalente", "h2"),
          P("Per ogni richiesta: <i>costo</i> = (input − cache − scrittura) × p<sub>in</sub> + cache × "
            "p<sub>cache</sub> + scrittura × p<sub>scrittura</sub> + output × p<sub>out</sub>, con i prezzi della Tabella 1 per milione di token (tariffa Standard, contesto "
            "breve, senza sconti batch). È il costo che lo stesso traffico avrebbe sostenuto via API. Riportiamo il "
            "totale, la media per task e il <b>costo per risposta corretta</b> (totale diviso per il numero di "
            "risposte corrette), che tiene conto insieme del prezzo e dell'accuratezza.")]
    S += [P("3.4 Quota dell'abbonamento (5 ore e settimanale)", "h2"),
          P("Prima e dopo ogni blocco di richieste dello stesso modello sullo stesso benchmark leggiamo lo stato della "
            "quota. Con agy usiamo il comando <font face='Mono'>/quota</font>, che restituisce per ogni gruppo di "
            "modelli la frazione rimanente (con otto cifre decimali) e l'orario di reset. Con Claude Code usiamo "
            "l'utilizzo delle finestre <i>five_hour</i> e <i>seven_day</i> riportato a ogni richiesta; se il piano ha "
            "un limite settimanale specifico per Opus o Sonnet, si usa quello. Con Codex usiamo i "
            "<i>rate_limits</i> allegati a ogni evento token_count (finestra primaria di 300 minuti e secondaria "
            "di 10.080 minuti, cioè 7 giorni). La quota consumata è la differenza tra le due letture. La quota è "
            "condivisa all'interno di un gruppo: in agy Gemini 4 Neon e Hydrogen condividono un budget, mentre Gemini 4 "
            "Argon ha un limite settimanale separato; in Claude Code e in Codex tutti i modelli condividono il budget "
            "dell'account. Le percentuali di gruppi e CLI diversi si riferiscono quindi a budget diversi. Durante una "
            "misura l'account non va usato per altro: per esempio, nessuna sessione di Claude Code aperta mentre "
            "girano i modelli Claude.")]
    S += [P("Se una finestra si azzera durante un blocco, la differenza non è più valida. In quel caso la "
            "percentuale viene stimata moltiplicando il costo API del blocco per il rapporto %-per-dollaro misurato "
            "sugli altri blocchi dello stesso modello: secondo i provider, la quota si consuma in proporzione al costo "
            "dei token. I valori stimati sono segnati con un asterisco nelle tabelle e tratteggiati nei grafici. "
            "Claude Code e Codex espongono la percentuale usata con risoluzione dell'1%: un blocco breve può risultare "
            "0% o 1%, quindi per questi modelli conviene leggere i totali e non i singoli benchmark. Al primo blocco di "
            "ogni modello viene inviata una richiesta minima di riscaldamento per ottenere la lettura iniziale. Dalla quota ricaviamo anche la "
            "<b>capacità</b>, cioè quanti task dello stesso tipo entrerebbero in una finestra di 5 ore o in una "
            "settimana piena.")]

    # 4. Limiti
    S += [P("4. Limiti e minacce alla validità", "h1")]
    for t in [
        "<b>Campione.</b> Il preset usato contiene pochi item per benchmark per non esaurire la quota: le differenze "
        "piccole non sono statisticamente significative (si vedano gli intervalli di confidenza).",
        "<b>Agente, non API pura.</b> Le CLI aggiungono un prompt di sistema e strumenti. Il prompt chiede di non "
        "usarli, ma un modello potrebbe comunque provarci (la sandbox lo limita); il numero di turni è registrato "
        "nei risultati grezzi. L'overhead di input è parte integrante del costo reale.",
        "<b>Contaminazione.</b> HumanEval, MBPP, MATH e MMLU-Pro sono pubblici da tempo e possono comparire nei dati "
        "di addestramento. AIME 2026 è più recente, ma non è garantito che sia escluso.",
        "<b>Prezzi.</b> Alcuni prezzi provengono da fonti secondarie (Tabella 1) e vanno verificati sulle pagine "
        "ufficiali; si aggiornano modificando <font face='Mono'>config/pricing.toml</font> e rigenerando il paper, "
        "senza ripetere le esecuzioni.",
        "<b>Quota.</b> Le percentuali dipendono dal piano dell'account e dal suo stato al momento della misura; "
        "risultati ottenuti su account e date diversi (per esempio agy e Codex eseguiti da persone diverse) non "
        "sono confrontabili in senso assoluto tra provider, ma sono validi all'interno dello stesso account.",
    ]:
        S.append(P(t, "bullet", ) if False else Paragraph(t, ST["bullet"], bulletText="•"))

    # 5. Risultati
    S += [NextPageTemplate("landscape"), PageBreak(), P("5. Risultati", "h1"),
          P("La Tabella 3 riassume, per ogni modello, tutte le metriche richieste: accuratezza, token di input, output e "
            "reasoning, costo API equivalente e percentuale di quota consumata nella finestra di 5 ore e in quella "
            "settimanale. I totali si riferiscono all'intera suite. I valori segnati con * sono stimati (sezione 3.4).")]
    hdr = ["Modello", "CLI", "Acc.<br/>media", "Task", "Token<br/>input", "di cui<br/>cache", "Token<br/>output",
           "di cui<br/>reason.", "Costo API<br/>totale", "Costo<br/>per task", "Costo per<br/>risp. corretta",
           "Quota<br/>5h", "Quota<br/>sett.", "Task per<br/>fin. 5h", "Task per<br/>sett.",
           "Latenza<br/>mediana", "Output<br/>tok/s"]
    rows = [hdr]
    for m in models:
        s = summ[k(m)]
        t = s["tokens"]
        star = lambda w: "*" if s[f"q_{w}_est"] else ""
        rows.append([
            m["label"], m["backend"], pct(s["acc_macro"]), f"{s['n_done']}/{s['n_target']}",
            num(t.get("input", 0), 0), pct(t.get("cached_input", 0) / t["input"]) if t.get("input") else "–",
            num(t.get("output", 0), 0), num(t.get("reasoning", 0), 0), usd(s["cost"]),
            usd(s["cost"] / s["attempts"] if s["cost"] is not None and s["attempts"] else None),
            usd(s["cost_per_correct"]),
            (pct(s["q_5h"], 2) + star("5h")) if s["q_5h"] is not None else "n/d",
            (pct(s["q_weekly"], 2) + star("weekly")) if s["q_weekly"] is not None else "n/d",
            num(s["cap_5h"], 0) if s["cap_5h"] else "–", num(s["cap_weekly"], 0) if s["cap_weekly"] else "–",
            (num(s["lat_p50"], 1) + " s") if s["lat_p50"] else "–", num(s["tps"], 0) if s["tps"] else "–",
        ])
    w = [32, 11, 13, 13, 19, 12, 18, 17, 17, 15, 18, 14, 14, 15, 15, 15, 13]
    scale = lw / (sum(w) * mm)
    S += [table(rows, [x * mm * scale for x in w], bold_first_col=True),
          P("Tabella 3. Riepilogo per modello sull'intera suite. Acc. media = media non pesata dei benchmark. "
            "La percentuale di cache è calcolata sui token di input. Capacità = task eseguiti ÷ frazione di quota "
            "consumata. Quota del gruppo del modello (sezione 3.4). n/d = limite non previsto per quel modello "
            "(per esempio Gemini 4 Argon ha solo un limite settimanale dedicato, senza finestra di 5 ore).", "cap")]

    # accuracy matrix
    rows = [["Modello"] + [BENCHMARKS[b]["label"] for b in benches] + ["Media"]]
    for m in models:
        r = [m["label"]]
        for b in benches:
            c = data["cells"].get((m["backend"], m["key"], b))
            if not c or c["acc"] is None:
                r.append("–")
            else:
                r.append(f"<b>{pct(c['acc'], 0)}</b> <font size=6 color='#52514e'>[{num(100 * c['ci'][0], 0)}–"
                         f"{num(100 * c['ci'][1], 0)}] n={c['n_done']}</font>")
        r.append(f"<b>{pct(summ[k(m)]['acc_macro'])}</b>")
        rows.append(r)
    cw = (lw - 38 * mm - 20 * mm) / len(benches)
    S += [KeepTogether([table(rows, [38 * mm] + [cw] * len(benches) + [20 * mm], bold_first_col=True),
                        P("Tabella 4. Accuratezza pass@1 per benchmark, con intervallo di confidenza al 95% (Wilson) "
                          "e numero di item valutati. Per IFEval è la <i>prompt-level strict accuracy</i>.", "cap")])]
    ife = [m for m in models if data["cells"].get((m["backend"], m["key"], "ifeval"))]
    if ife:
        S += [P("IFEval, metriche secondarie: " + "; ".join(
            f"{m['label']}: loose {pct(data['cells'][(m['backend'], m['key'], 'ifeval')]['ifeval_loose'])}, "
            f"instruction-level strict {pct(data['cells'][(m['backend'], m['key'], 'ifeval')]['ifeval_inst'])}"
            for m in ife) + ".")]

    S += [NextPageTemplate("portrait"), PageBreak(),
          P("5.1 Grafici", "h2"), _img(figs["acc"], pw),
          P("Figura 1. Accuratezza per benchmark. Le barre sottili indicano l'intervallo di confidenza al 95%.", "cap"),
          KeepTogether([_img(figs["cost"], pw * 0.92),
                        P("Figura 2. Accuratezza media in funzione del costo API medio per task (asse logaritmico). "
                          "I modelli in alto a sinistra offrono il compromesso migliore.", "cap")]),
          KeepTogether([_img(figs["tok"], pw * 0.92),
                        P("Figura 3. Composizione media dei token per richiesta. L'input comprende il prompt di sistema "
                          "dell'agente; reasoning e output visibile insieme formano gli output token fatturati.", "cap")]),
          KeepTogether([_img(figs["quota"], pw),
                        P("Figura 4. Quota consumata ogni 10 task, nel gruppo di quota del modello. Le barre "
                          "tratteggiate sono stime (sezione 3.4).", "cap")])]

    # per-benchmark detail (landscape)
    S += [NextPageTemplate("landscape"), PageBreak(), P("5.2 Dettaglio per benchmark", "h2"),
          P("La Tabella 5 riporta le stesse metriche per ogni coppia benchmark-modello. Le medie di token e costo "
            "sono per tentativo; i valori di quota sono quelli del singolo benchmark.")]
    hdr = ["Benchmark", "Modello", "Acc.", "IC 95%", "n", "Err.", "Input<br/>medio", "Cache<br/>media", "Output<br/>medio",
           "Reason.<br/>medio", "Costo<br/>totale", "Costo<br/>per task", "Quota<br/>5h", "Quota<br/>sett.",
           "Latenza<br/>p50", "Latenza<br/>p90", "Output<br/>tok/s"]
    rows = [hdr]
    for b in benches:
        for m in models:
            c = data["cells"].get((m["backend"], m["key"], b))
            if not c:
                continue
            n = max(1, c["attempts"])
            t = c["tokens"]
            q5, qw = c["quota"]["5h"], c["quota"]["weekly"]
            rows.append([
                BENCHMARKS[b]["label"], m["label"], pct(c["acc"]),
                f"{num(100 * c['ci'][0], 0)}–{num(100 * c['ci'][1], 0)}%", str(c["n_done"]), str(c["errors"]),
                ktok(t.get("input", 0) / n), ktok(t.get("cached_input", 0) / n), ktok(t.get("output", 0) / n),
                ktok(t.get("reasoning", 0) / n), usd(c["cost"]), usd(c["cost"] / n if c["cost"] is not None else None),
                (pct(q5["used"], 2) + ("*" if q5["estimated"] else "")) if q5["used"] is not None else "n/d",
                (pct(qw["used"], 2) + ("*" if qw["estimated"] else "")) if qw["used"] is not None else "n/d",
                (num(c["lat_p50"], 1) + " s") if c["lat_p50"] else "–",
                (num(c["lat_p90"], 1) + " s") if c["lat_p90"] else "–", num(c["tps"], 0) if c["tps"] else "–",
            ])
    w = [27, 30, 13, 18, 8, 8, 14, 14, 14, 16, 15, 15, 14, 14, 15, 15, 13]
    scale = lw / (sum(w) * mm)
    S += [table(rows, [x * mm * scale for x in w]),
          P("Tabella 5. Metriche per benchmark e modello. Err. = richieste fallite anche dopo i tentativi di ripetizione "
            "(contano come errate). * = quota stimata. n/d = limite non previsto per quel modello.", "cap")]

    # 6. Conclusioni
    S += [NextPageTemplate("portrait"), PageBreak(), P("6. Discussione e conclusioni", "h1")]
    concl = []
    for b in benches:
        cands = [(m, data["cells"][(m["backend"], m["key"], b)]) for m in models
                 if (m["backend"], m["key"], b) in data["cells"] and data["cells"][(m["backend"], m["key"], b)]["acc"] is not None]
        if not cands:
            continue
        top = max(c["acc"] for _, c in cands)
        leaders = [m["label"] for m, c in cands if c["acc"] == top]
        concl.append(f"<b>{BENCHMARKS[b]['label']}</b>: miglior risultato {pct(top, 0)} ({', '.join(leaders)}).")
    for t in concl:
        S.append(Paragraph(t, ST["bullet"], bulletText="•"))
    if cheap:
        S.append(P("In rapporto qualità/prezzo, il costo API per risposta corretta va da "
                   f"{usd(summ[k(cheap[0])]['cost_per_correct'])} ({cheap[0]['label']}) a "
                   f"{usd(summ[k(cheap[-1])]['cost_per_correct'])} ({cheap[-1]['label']})."))
    capm = [m for m in models if summ[k(m)]["cap_weekly"]]
    if capm:
        capm.sort(key=lambda m: -summ[k(m)]["cap_weekly"])
        S.append(P("Rispetto alla quota, con un intero limite settimanale si eseguirebbero circa "
                   + "; ".join(f"{num(summ[k(m)]['cap_weekly'], 0)} task con {m['label']}" for m in capm)
                   + ". I gruppi di quota sono diversi, quindi il confronto vale per l'account misurato."))
    S.append(P("Con il preset usato questi risultati sono indicativi. Per conclusioni più solide conviene ripetere la "
               "campagna con un preset più ampio (per esempio <i>standard</i>), distribuendo l'esecuzione su più "
               "finestre: il runner aspetta da solo il reset della quota e riprende."))

    # Appendix
    S += [P("Appendice A. Riproducibilità ed estensione", "h1"),
          P("Il codice, i dati congelati e i risultati grezzi (un file JSONL per modello e benchmark, con la risposta "
            "completa, i token e il punteggio di ogni item) stanno nella stessa cartella. Per riprodurre o estendere:")]
    for c in ["uv run bench.py selftest", "uv run bench.py run --backend agy", "uv run bench.py run --backend claude",
              "uv run bench.py run --backend codex",
              "uv run bench.py status", "uv run bench.py paper"]:
        S.append(P(c, "code"))
    S.append(P("Il comando <font face='Mono'>run</font> usa sempre la suite fissata in "
               "<font face='Mono'>data/suite.lock.json</font>, salta gli item già completati e, alla fine, rigenera "
               "questo PDF con tutti i risultati presenti in <font face='Mono'>results/</font>, indipendentemente dal "
               "computer che li ha prodotti. Per unire i risultati di due macchine basta copiare le rispettive "
               "sottocartelle <font face='Mono'>results/&lt;backend&gt;/</font>. Impronte dei pool di dati (SHA-256, "
               "16 caratteri): " + "; ".join(f"{BENCHMARKS[b]['label']} {lock['benchmarks'][b]['pool_sha256_16']}"
                                             for b in benches) + "."))
    pr = data["pricing"]
    S.append(P(f"Prezzi verificati il {pr.get('checked_on', '?')}. Fonti: " + "; ".join(
        f"{e['key']}: {e.get('source', '')}" for e in pr.get("price", [])) + "."))
    return S
