# C-Exec-Equiv Benchmark Harness

Tool-free C program execution benchmark comparing **Claude Code (`claude`)**, **OpenAI Codex CLI (`codex`)**, and **Google Antigravity CLI (`agy`)** across four difficulty tiers and three semantic equivalence styles (`classic_c`, `randomized_c`, `strange_c`).

## Directory Layout
```text
c_exec_bench/
├── data/                  # Place your generated tier1.json .. tier4.json here
├── results/               # Auto-populated ground truth & model predictions
├── report/                # Auto-generated Markdown report & PNG charts
├── validate_dataset.py    # 1. Compiles with gcc/clang + UBSan/ASan & computes ground truth
├── run_benchmark.py       # 2. Runs Claude, Codex, and Antigravity with tools disabled
├── analyze_results.py     # 3. Computes metrics, plots charts & builds benchmark_report.md
└── run_all.sh             # One-command pipeline runner
```

## Quick Start
1. Drop your 4 JSON files (`tier1.json`, `tier2.json`, `tier3.json`, `tier4.json`) into `c_exec_bench/data/`.
2. Install Python plotting dependencies (standard library is used for everything else):
   ```bash
   pip install matplotlib numpy
   ```
3. Run the entire pipeline (or run individual steps):
   ```bash
   chmod +x run_all.sh
   ./run_all.sh
   ```

## How Tool-Free Enforcement Works
Each runner adapter enforces zero code execution at **two layers** (CLI flags + isolated empty temp working directory + strict JSON trace auditing):
- **Claude Code:** Invoked with `claude --restricted --tools "" --no-session-persistence --output-format json -p "<prompt>"`.
- **OpenAI Codex CLI:** Invoked inside an isolated temp directory with a locked-down `CODEX_HOME` (`config.toml` disabling shell/MCP/features), `codex exec --json --ephemeral --skip-git-repo-check`, and strict event-stream auditing (`command_execution` / `mcp_tool_call` events trigger automatic disqualification).
- **Antigravity CLI (`agy`):** Invoked inside an isolated workspace containing `.agent/agents/no-tools.md` (`tools: []`), `agy -p "<prompt>" --agent no-tools --output-format json`, plus JSON trace auditing to disqualify any turn with `num_turns > 1` or tool-call events.

## Customizing Models
Pass `--models` to `run_benchmark.py` using `provider:model_id` syntax:
```bash
python3 run_benchmark.py \
  --models "claude:claude-opus-5-5" "claude:claude-sonnet-5-5" \
           "codex:gpt-6.2-sol" "codex:gpt-6.1-terra" \
           "agy:gemini-4-neon" "agy:gemini-4-argon"
```
(Or omit `:model_id`—e.g. `--models claude codex agy`—to use each CLI's currently configured default model.)

## Effort ladder (low → medium → high)
`run_ladder.py` is a separate run that does not touch `results/predictions.json`. It calls the CLIs **exactly as the
main benchmark does** (`benchlib/backends`: same isolation, same models from `config/models.toml`, same token
accounting and API prices, structured `{"stdout": ...}` answer); only the task and the effort change.

Every (problem, variant) pair runs at **low** effort; only the pairs a model got wrong are re-run at **medium**, and
only those still wrong at **high** (never beyond high). Levels run phase by phase and every answer is checkpointed to
`results/ladder/<cli>__<model>.json` (one file per model, so different CLIs can run at the same time); the same
command resumes after a stop. Run it from the repository root:
```bash
uv run c_exec_bench/run_ladder.py --backend codex        # every Codex model of the main benchmark
uv run c_exec_bench/run_ladder.py --backend agy claude   # several CLIs; --backend all = agy, claude, codex
uv run c_exec_bench/run_ladder.py --models claude-haiku-5-5 gpt-6-luna --tiers 3 4
uv run c_exec_bench/run_ladder.py --report               # rebuild report/ladder_report.md only
```
`report/ladder_report.md` gives, per model, the single accuracy at each level (solved / attempted there) and the
cumulative accuracy (solved by that level or an earlier one / all pairs), with time, tokens and API cost per level,
plus cumulative accuracy per tier and per variant. As with `bench.py`, do not use the same account for anything
else while it runs (no Claude Code session open during `--backend claude`).
