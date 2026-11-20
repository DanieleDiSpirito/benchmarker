# C-Exec-Equiv Benchmark Guide

A tool-free C11 program execution benchmark designed to evaluate **Claude Code (`claude`)**, **OpenAI Codex CLI (`codex`)**, and **Google Antigravity CLI (`agy`)** across four difficulty tiers and three semantic equivalence styles (`classic_c`, `randomized_c`, `strange_c`).

## 1. Directory Structure

After unzipping `c_exec_bench.zip` and adding `run_models.sh`, your folder should look like this:

```text
c_exec_bench/
├── data/                  # Place tier1.json, tier2.json, tier3.json, tier4.json here
├── results/               # Auto-generated ground truth & checkpointed model predictions
├── report/                # Auto-generated Markdown report & PNG charts
├── validate_dataset.py    # Step 1: Compiles & verifies C triplets with UBSan/ASan
├── run_benchmark.py       # Step 2: Core evaluation engine with tool-free enforcement
├── analyze_results.py     # Step 3: Generates plots & benchmark_report.md
├── run_models.sh          # Recommended launcher for your 9 target models & CLI flags
├── run_all.sh             # Runs default CLI models end-to-end
└── README.md              # This guide
```

## 2. Prerequisites & Installation

1. **Compilers & Sanitizer Runtimes (for Step 1 ground-truth validation):**
   Ensure `gcc` (and optionally `clang`) plus ASan/UBSan runtimes are installed:
   * **Fedora / RHEL:**
     ```bash
     sudo dnf install gcc clang libasan libubsan
     ```
   * **Ubuntu / Debian:**
     ```bash
     sudo apt install gcc clang libasan8 libubsan1
     ```

2. **Python Packages (for Step 3 plotting):**
   ```bash
   pip install matplotlib numpy
   ```

3. **Make Scripts Executable & Authenticate Each CLI Once:**
   ```bash
   chmod +x run_models.sh run_all.sh
   ```
   Before running the benchmark, ensure you are logged into each CLI tool with your Pro plan:
   * **Claude Code:** Run `claude` once in your terminal and complete login.
   * **Codex CLI:** Run `codex` once and complete login (creates `~/.codex/auth.json`, which the runner automatically symlinks into its isolated environment).
   * **Antigravity CLI:** Run `agy` once and complete login.

## 3. Step-by-Step Execution

### Step 1: Validate Your Generated JSON Files

Copy your 4 DeepSeek-generated JSON files into `c_exec_bench/data/` and run validation:

```bash
cp /path/to/tier1.json /path/to/tier2.json /path/to/tier3.json /path/to/tier4.json data/
python3 validate_dataset.py
```

* **What this does:** Strips any stray copy-paste quotes/fences, compiles every `classic_c`, `randomized_c`, and `strange_c` program under `gcc -O0` and `clang -O2` with `-fsanitize=undefined,address`, discards any triplet with Undefined Behavior or mismatched outputs, and writes the verified ground truth to `results/validated_dataset.json`.

### Step 2: Run the Benchmark via `run_models.sh`

`run_models.sh` automatically validates the dataset (if `results/validated_dataset.json` does not exist yet), runs the selected model(s) with **external tools disabled**, saves checkpoints to `results/predictions.json` after every single problem, and regenerates all plots and `report/benchmark_report.md` upon completion.

#### Configured Model Targets & Shortcuts

| Shortcut | Full CLI Target (`cli:model_id`) | Provider CLI |
| :--- | :--- | :--- |
| `astra` | `codex:gpt-6.2-astra` | OpenAI Codex CLI |
| `sol` | `codex:gpt-6.2-sol` | OpenAI Codex CLI |
| `terra` | `codex:gpt-6.1-terra` | OpenAI Codex CLI |
| `neon` | `agy:gemini-4-neon` | Google Antigravity CLI |
| `hydrogen` | `agy:gemini-4-hydrogen` | Google Antigravity CLI |
| `argon` | `agy:gemini-4-argon` | Google Antigravity CLI |
| `opus` | `claude:claude-opus-5-5` | Claude Code CLI |
| `sonnet` | `claude:claude-sonnet-5-5` | Claude Code CLI |
| `haiku` | `claude:claude-haiku-5-5` | Claude Code CLI |

#### Usage Examples

* **Run all 9 models sequentially:**
  ```bash
  ./run_models.sh all
  ```

* **Run all 3 models of a single provider family:**
  ```bash
  ./run_models.sh codex
  ./run_models.sh agy
  ./run_models.sh claude
  ```

* **Run just one specific model (using its shortcut or full model name):**
  ```bash
  ./run_models.sh opus
  ./run_models.sh astra
  ./run_models.sh neon
  ```

* **Run a quick smoke test on specific difficulty tiers (`--tiers`):**
  ```bash
  ./run_models.sh opus --tiers 1
  ./run_models.sh sol --tiers 1 2
  ./run_models.sh all --tiers 1
  ```

* **Increase per-problem timeout (`--timeout`, default `180` seconds) for heavy Tier 4 reasoning:**
  ```bash
  ./run_models.sh astra --tiers 4 --timeout 300
  ./run_models.sh opus --tiers 3 4 --timeout 300
  ```

* **Retry only timed-out or rate-limited items (`--retry-errors`):**
  ```bash
  ./run_models.sh claude --retry-errors
  ./run_models.sh all --retry-errors
  ```

* **Run any custom `cli:model_id` directly:**
  ```bash
  ./run_models.sh codex:gpt-6.2-sol --tiers 1 2 3 4
  ```

### Step 3: View Results & Charts (or Regenerate Manually)

`run_models.sh` automatically runs `analyze_results.py` at the end of every execution, but you can also regenerate the report and charts manually at any time:

```bash
python3 analyze_results.py
```

Generated outputs:
* `report/benchmark_report.md` — Full leaderboard table (Overall Accuracy, Classic vs. Randomized vs. Strange C accuracy, Style Degradation delta, Triplet Self-Consistency, Mean Latency, and Tool Violations).
* `report/plots/accuracy_by_tier.png` — Grouped bar chart comparing models across Tiers 1–4.
* `report/plots/style_degradation.png` — Bar chart showing accuracy drop from `Classic C` to `Strange / Obfuscated C`.
* `report/plots/consistency_and_latency.png` — Side-by-side chart of triplet consistency and inference time vs. accuracy.

## 4. How Tool-Free Enforcement Works

Each runner adapter inside `run_benchmark.py` enforces zero code execution at **two layers** (preventative CLI/environment restrictions + strict JSON trace auditing):

* **Claude Code (`claude`):** Invoked with `claude --restricted --tools "" --no-session-persistence --output-format json -p "<prompt>"` inside an isolated empty workspace, and audited to ensure `num_turns == 1` with zero `tool_use` blocks.
* **OpenAI Codex CLI (`codex`):** Invoked inside an isolated temp directory with a locked-down `CODEX_HOME` (`approval_policy = "never"`, `sandbox_mode = "read-only"`, `shell_tool = false`, `apply_patch_freeform = false`, `multi_agent = false`), `codex exec --json --ephemeral --skip-git-repo-check`, and event-stream auditing (`command_execution`, `mcp_tool_call`, `file_change`, or `web_search` events trigger automatic `tool_violation` disqualification).
* **Antigravity CLI (`agy`):** Invoked inside an isolated workspace containing `.agent/agents/no-tools.md` (`tools: []`), `agy -p "<prompt>" --agent no-tools --output-format json`, and audited to disqualify any run with `num_turns > 1` or `functionCall` / `tool_use` blocks.

## 5. Troubleshooting

* **Missing `libasan.so` / `libubsan.so` on Fedora:** Install `sudo dnf install libasan libubsan` before running `python3 validate_dataset.py`.
* **Resuming after an interruption or quota limit:** Simply re-run the exact same `./run_models.sh ...` command. Because `run_benchmark.py` checkpoints after every single problem, already-completed items are skipped automatically.