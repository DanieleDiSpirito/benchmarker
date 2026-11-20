#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[1/3] Validating C triplets with gcc/clang & UBSan/ASan..."
python3 validate_dataset.py

echo "[2/3] Running tool-free evaluation across Claude Code, Codex CLI, and Antigravity CLI..."
# Customize --models below if you want to test specific model versions (e.g. claude:claude-opus-5-5)
python3 run_benchmark.py --models claude codex agy

echo "[3/3] Generating plots and Markdown benchmark report..."
python3 analyze_results.py

echo "Done! Open report/benchmark_report.md to view results and charts."
