# Which $20 AI Subscription Gives a Student the Most?

*Daniele Di Spirito, Maurizio Camposeo*

A deterministic benchmark of the models included in the $20/month plans of Google (Google AI Pro),
Anthropic (Claude Pro) and OpenAI (ChatGPT Plus), run **through each plan's own CLI**. Model quality is only
half of the answer: a very strong model you can use for five prompts a week is worth less than a weaker one
you can use all day. So for every answer we measure:

- accuracy on hard tasks, scored **without any LLM judge**;
- **cost per answer** if the same request went through the API (`config/pricing.toml`);
- **input / output / reasoning tokens**;
- **time per answer**;
- the share of the **5-hour** and **weekly** limits consumed, read from the CLI's own quota counters, and the
  resulting **capacity**: correct answers per full 5-hour window / week;
- **consumption/speed clusters**: k-means on cost, output tokens and median time per answer (number of groups
  chosen by silhouette), with one comparison table per cluster.

All models run at **medium** reasoning effort. The output is `paper/benchmark_paper.pdf`.

| Plan | CLI | Models |
|---|---|---|
| Google AI Pro | `agy` (Antigravity) | Gemini 4 Neon, Gemini 4 Argon, Gemini 4 Hydrogen, Gemini 3.8 Flash |
| Claude Pro | `claude` (Claude Code) | Claude Opus 5.5, Claude Sonnet 5.5, Claude Haiku 5.5 |
| ChatGPT Plus | `codex` (Codex CLI) | GPT-6.2 Astra, GPT-6.2 Sol, GPT-6.1 Terra, GPT-6 Luna |

## Benchmarks

| Benchmark | Area | Scoring |
|---|---|---|
| LiveCodeBench v6, hard (Jan–Apr 2025) | Coding | program run on every hidden test |
| HMMT February 2026 | Math | symbolic equivalence (math-verify) |
| BeyondAIME | Math | exact integer |
| SuperGPQA, hard | Graduate-level knowledge | option letter |
| BIG-Bench Extra Hard, mini | Reasoning | official BBEH matching |
| IFBench | Instruction following | official AllenAI verifiers |

## Requirements

- [uv](https://docs.astral.sh/uv/): it installs Python 3.14 and the dependencies on the first `uv run`.
- Your plan's CLI, already logged in: `agy`, `claude` or `codex`.

No LaTeX is needed: the PDF is generated in Python. On first use the scripts download the NLTK tokenizers and
the LiveCodeBench hidden tests (~134 MB, into `data/cache/`, not versioned).

## Usage

```bash
uv run bench.py selftest                       # validate every grader: all checks must pass
uv run bench.py run --backend agy --preset mini   # first run only: --preset fixes the suite size
uv run bench.py run --backend claude
uv run bench.py run --backend claude --models claude-haiku-5-5   # a single model
uv run bench.py escalate --backend agy         # re-run at high effort every task a model got wrong at medium
uv run bench.py status                         # progress, tokens, cost and quota per model
uv run bench.py paper                          # rebuild the PDF only
uv run bench.py regrade                        # re-score stored replies with the current graders
```

| Option | Effect |
|---|---|
| `--preset` | first run only: suite size per model (`smoke` 12, `mini` 42, `small` 99, `standard` 282 tasks) |
| `--models k1 k2` | only some models (keys in `config/models.toml`) |
| `--benchmarks b1 b2` | only some benchmarks |
| `--max-weekly-per-model 30` | stop a model once it has used 30% of the weekly limit |
| `--reserve-5h 3` / `--reserve-weekly 3` | quota (%) always kept free |
| `--no-wait` | stop instead of waiting for the 5-hour window to reset |
| `--workers 3` | parallel requests |

Runs **resume**: every reply is saved as soon as it arrives. If the run stops (Ctrl-C, quota, shutdown), run the
same command again. When the 5-hour window is nearly used up the runner waits for the reset and continues.

**While a backend is running, do not use the same account for anything else** (for example no other Claude Code
session during `--backend claude`): quota is measured as the difference between two readings of the account.

## Escalation to high effort

The main runs use **medium** effort. `bench.py escalate --backend <cli>` then re-runs, once at **high** effort, every
task each model got wrong; results go to `results/<cli>/<model>/high/` and never change the main results. The
paper reports, per model, how many wrong answers high effort recovers, the cumulative accuracy and the extra cost,
time and quota. Run it after the main run of that CLI has finished (it resumes like `run`; if it is launched
earlier it only covers the answers already scored). A full low → medium → high ladder is left as future work.

## Extending the paper with Codex (on another machine)

1. Install and log in to the Codex CLI (`codex login`).
2. From the repository:
   ```bash
   uv run bench.py selftest
   uv run bench.py run --backend codex
   git add results/codex paper && git commit -m "Codex results"
   ```
3. Send the commit back (push, or `git bundle create codex.bundle --all`).

`run` always uses the suite pinned in `data/suite.lock.json` (same tasks, same order, same prompts) and only adds
`results/codex/`; the other results are untouched and the paper is rebuilt from everything in `results/`. If your
Codex model ids differ, edit `cli_model` in `config/models.toml`. `CODEX_BIN` / `CODEX_HOME` are honoured. If two
people commit a regenerated PDF, resolve the conflict by running `uv run bench.py paper` again.

## Quota measurement

- **agy**: `/quota` returns the remaining fraction of each group's 5-hour and weekly limits with 8 decimals.
- **Claude Code** and **Codex**: usage is reported in whole percentage points. Totals per model are exact to ±1
  point and are split across benchmarks by API cost; small totals are marked ≈, values derived from the other
  window through the account's weekly/5-hour ratio are marked ~, and `<1%` means the counter never moved.
  Quota is not proportional to list prices, so it is never estimated from them.

## Layout

```
bench.py                 commands (prepare, selftest, run, status, paper, regrade)
config/models.toml       plans and models per CLI (model id, effort, price key)
config/pricing.toml      API prices $/M tokens, with source and validity dates
data/frozen/             frozen, shuffled benchmark items
data/suite.lock.json     exact item ids of the suite in use
results/<cli>/<model>/<benchmark>.jsonl   one line per attempt (reply, tokens, score)
results/<cli>/<model>/quota.jsonl         quota readings before/after each block
benchlib/                runner, CLI backends, graders, analysis, clustering, charts, paper
benchlib/vendor/         IFBench (AllenAI) and BBEH (Google) official scorers, Apache 2.0
archive/                 results of earlier suite versions
```

Generated code is executed in a separate `python -I` process in a temporary directory with CPU and file-size
limits; on macOS a `sandbox-exec` profile also blocks network access and writes outside that directory. On
Linux there is no such sandbox: run the benchmark in a container or VM if that matters to you.

If a price changes, edit `config/pricing.toml` and run `uv run bench.py paper`: costs are recomputed from the
stored tokens without running anything again.

To hand the project over, share the git repository (`.venv/` and `data/cache/` are not versioned).
