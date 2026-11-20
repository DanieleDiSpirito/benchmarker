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
