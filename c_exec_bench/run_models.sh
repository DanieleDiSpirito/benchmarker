#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# All 9 exact model identifiers mapped to their respective CLIs:
#   - Codex CLI (`codex`): gpt-6.2-astra, gpt-6.2-sol, gpt-6.1-terra
#   - Antigravity CLI (`agy`): gemini-4-neon, gemini-4-hydrogen, gemini-4-argon
#   - Claude Code (`claude`): claude-opus-5-5, claude-sonnet-5-5, claude-haiku-5-5

CODEX_MODELS=("codex:gpt-6.2-astra" "codex:gpt-6.2-sol" "codex:gpt-6.1-terra")
AGY_MODELS=("agy:gemini-4-neon" "agy:gemini-4-hydrogen" "agy:gemini-4-argon")
CLAUDE_MODELS=("claude:claude-opus-5-5" "claude:claude-sonnet-5-5" "claude:claude-haiku-5-5")

ALL_MODELS=("${CODEX_MODELS[@]}" "${AGY_MODELS[@]}" "${CLAUDE_MODELS[@]}")

usage() {
  cat <<EOF
Usage: ./run_models.sh [TARGET] [--tiers 1 2 3 4] [--retry-errors]

Targets (run all 9 models, a single family, or one specific model):
  all           Run all 9 models across Codex, Antigravity, and Claude (default)
  codex         Run only Codex models (gpt-6.2-astra, gpt-6.2-sol, gpt-6.1-terra)
  agy           Run only Antigravity models (gemini-4-neon, gemini-4-hydrogen, gemini-4-argon)
  claude        Run only Claude models (claude-opus-5-5, claude-sonnet-5-5, claude-haiku-5-5)

Individual model shortcuts:
  astra         codex:gpt-6.2-astra
  sol           codex:gpt-6.2-sol
  terra         codex:gpt-6.1-terra
  neon          agy:gemini-4-neon
  hydrogen      agy:gemini-4-hydrogen
  argon         agy:gemini-4-argon
  opus          claude:claude-opus-5-5
  sonnet        claude:claude-sonnet-5-5
  haiku         claude:claude-haiku-5-5

You can also pass a full cli:model string directly (e.g. ./run_models.sh codex:gpt-6.2-sol).
Any extra flags after TARGET (such as --tiers 1 or --retry-errors) are forwarded to run_benchmark.py.
EOF
}

TARGET="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$TARGET" in
  -h|--help|help)
    usage
    exit 0
    ;;
  all)
    SELECTED=("${ALL_MODELS[@]}")
    ;;
  codex)
    SELECTED=("${CODEX_MODELS[@]}")
    ;;
  agy)
    SELECTED=("${AGY_MODELS[@]}")
    ;;
  claude)
    SELECTED=("${CLAUDE_MODELS[@]}")
    ;;
  astra|gpt-6.2-astra)
    SELECTED=("codex:gpt-6.2-astra")
    ;;
  sol|gpt-6.2-sol)
    SELECTED=("codex:gpt-6.2-sol")
    ;;
  terra|gpt-6.1-terra)
    SELECTED=("codex:gpt-6.1-terra")
    ;;
  neon|gemini-4-neon)
    SELECTED=("agy:gemini-4-neon")
    ;;
  hydrogen|gemini-4-hydrogen)
    SELECTED=("agy:gemini-4-hydrogen")
    ;;
  argon|gemini-4-argon)
    SELECTED=("agy:gemini-4-argon")
    ;;
  opus|claude-opus-5-5)
    SELECTED=("claude:claude-opus-5-5")
    ;;
  sonnet|claude-sonnet-5-5)
    SELECTED=("claude:claude-sonnet-5-5")
    ;;
  haiku|claude-haiku-5-5)
    SELECTED=("claude:claude-haiku-5-5")
    ;;
  *:*)
    SELECTED=("$TARGET")
    ;;
  *)
    echo "[ERROR] Unknown target: $TARGET" >&2
    usage
    exit 1
    ;;
esac

if [[ ! -f "results/validated_dataset.json" ]]; then
  echo "[1/3] Validating dataset..."
  python3 validate_dataset.py
fi

echo "[2/3] Running benchmark for: ${SELECTED[*]}"
python3 run_benchmark.py --models "${SELECTED[@]}" "$@"

echo "[3/3] Updating plots and Markdown report..."
python3 analyze_results.py

echo "Done! Results and charts updated in report/benchmark_report.md"
