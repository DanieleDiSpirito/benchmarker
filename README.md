# Benchmarker

Benchmark deterministico di LLM usati **in abbonamento** tramite CLI:

- **Antigravity (`agy`)**: Gemini 4 Neon, Gemini 4 Argon, Gemini 4 Hydrogen
- **Claude Code (`claude`)**: Claude Opus 5.5, Claude Sonnet 5.5, Claude Haiku 5.5
- **Codex (`codex`)**: GPT-6.2 Astra, GPT-6.2 Sol, GPT-6.1 Terra

Per ogni modello misura l'accuratezza su coding (HumanEval+, MBPP+), matematica (AIME 2026, MATH-500 livello 5),
conoscenza (MMLU-Pro) e instruction following (IFEval). Nessun punteggio usa un LLM-as-a-Judge. Misura anche:

- token di **input** (con la parte da cache), **output** e **reasoning**;
- **costo stimato via API** (prezzi in `config/pricing.toml`);
- **percentuale consumata** della finestra di **5 ore** e del limite **settimanale**, letta dalla CLI prima e dopo
  ogni blocco di richieste.

Il risultato è un paper PDF in `paper/benchmark_paper.pdf`: prima descrive metodo e metriche, poi riporta le tabelle e i grafici.

## Requisiti

- [uv](https://docs.astral.sh/uv/). Installa da solo Python 3.14 e le dipendenze al primo `uv run`.
- La CLI del proprio abbonamento, già autenticata: `agy`, `claude` oppure `codex`.

Non servono LaTeX né altri programmi: il PDF viene generato in Python.

## Uso

```bash
uv run bench.py selftest              # verifica i valutatori: tutte le soluzioni di riferimento devono passare
uv run bench.py run --backend agy     # esegue i modelli Gemini (Antigravity) e rigenera il paper
uv run bench.py run --backend claude  # esegue Opus, Sonnet e Haiku (Claude Code) e rigenera il paper
uv run bench.py run --backend claude --models claude-haiku-5-5   # solo Haiku 5.5
uv run bench.py status                # avanzamento, token, costo e quota per modello
uv run bench.py paper                 # rigenera solo il PDF
```

Opzioni utili di `run`:

| Opzione | Effetto |
|---|---|
| `--models k1 k2` | solo alcuni modelli (chiavi di `config/models.toml`) |
| `--benchmarks b1 b2` | solo alcuni benchmark |
| `--preset mini` | solo alla prima esecuzione: dimensione della suite (`smoke` 12, `mini` 40, `small` 88, `standard` 240 task per modello) |
| `--max-weekly-per-model 30` | ferma un modello quando ha consumato il 30% del limite settimanale |
| `--reserve-5h 3` / `--reserve-weekly 3` | quanta quota lasciare sempre libera (in %) |
| `--no-wait` | se la finestra di 5 ore si esaurisce, fermati invece di attendere il reset |
| `--workers 3` | richieste in parallelo |

Le esecuzioni sono **riprendibili**: ogni risposta viene salvata appena arriva. Se interrompi lo script, se la quota
finisce o se il PC si spegne, basta rilanciare lo stesso comando. Quando la finestra di 5 ore è quasi esaurita, lo
script aspetta il reset e poi riparte da solo.

## Estendere il paper con Codex (su un altro computer)

Il progetto contiene già i risultati Gemini e Claude in `results/agy/` e `results/claude/`. Chi ha Codex deve:

1. avere `codex` installato e autenticato (`codex login`);
2. dalla cartella del progetto, eseguire:
   ```bash
   uv run bench.py selftest
   uv run bench.py run --backend codex
   ```
3. restituire la cartella `results/codex/` (oppure l'intero progetto) e il nuovo `paper/benchmark_paper.pdf`.

`run --backend codex` usa la stessa suite fissata in `data/suite.lock.json`: stessi problemi, stesso ordine, stessi
prompt. Aggiunge solo i dati Codex; quelli di Gemini e Claude non vengono toccati. Il paper viene rigenerato con tutti i
risultati presenti in `results/`. Se gli ID dei modelli nella tua installazione di Codex sono diversi, modifica
`cli_model` in `config/models.toml`. Puoi impostare anche `CODEX_BIN` (percorso della CLI) e `CODEX_HOME`.

Durante la misura non usare lo stesso account per altro (per esempio una sessione di Claude Code aperta mentre gira
`--backend claude`): la percentuale di quota è la differenza tra due letture dello stesso account.

## Struttura

```
bench.py                 comandi (prepare, selftest, run, status, paper)
config/models.toml       modelli per backend (ID CLI, effort, chiave prezzo)
config/pricing.toml      prezzi API $/M token con fonte e periodo di validità
data/frozen/             item dei dataset, congelati e mescolati con seme fisso
data/suite.lock.json     ID esatti degli item della suite in uso
results/<backend>/<modello>/<benchmark>.jsonl   una riga per tentativo (risposta, token, punteggio)
results/<backend>/<modello>/quota.jsonl         letture della quota prima/dopo ogni blocco
benchlib/                runner, backend, valutatori, analisi, grafici, paper
benchlib/vendor/ifeval/  verificatori IFEval di Google Research (Apache 2.0)
```

Se un prezzo cambia, correggi `config/pricing.toml` e lancia `uv run bench.py paper`: i costi vengono ricalcolati
dai token salvati, senza ripetere le esecuzioni.

Per passare il progetto a un'altra persona, copia l'intera cartella tranne `.venv/` (uv la ricrea da solo).
