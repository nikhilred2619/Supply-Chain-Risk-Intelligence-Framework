#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/run_full_experiment.sh
# One-command full experiment reproduction script.
#
# Runs all paper experiments in sequence and saves results + figures.
# No DataCo dataset required for the simulation benchmark.
#
# Usage:
#   bash scripts/run_full_experiment.sh
#   bash scripts/run_full_experiment.sh --with-dataco  # if dataset available
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$ROOT_DIR/results"

cd "$ROOT_DIR"

echo "═══════════════════════════════════════════════════════════════════"
echo "  LLM-FMEA SCRA — Full Experiment Suite"
echo "  Paper: LLM-Augmented Decision Intelligence for Supply Chain Risk"
echo "  Author: Nikhil Reddy Donapati"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# ── 1. Synthetic simulation benchmark ────────────────────────────────────────
echo "[ Step 1/4 ] Running 120-scenario simulation benchmark ..."
python experiments/run_simulation.py
echo "  ✓ Simulation complete"
echo ""

# ── 2. Figure generation ─────────────────────────────────────────────────────
echo "[ Step 2/4 ] Generating publication-quality figures ..."
python experiments/generate_figures.py
echo "  ✓ Figures saved to $RESULTS_DIR/figures/"
echo ""

# ── 3. Unit tests ─────────────────────────────────────────────────────────────
echo "[ Step 3/4 ] Running test suite ..."
python -m pytest tests/ -v --tb=short 2>&1 | tail -20
echo "  ✓ Tests complete"
echo ""

# ── 4. DataCo validation (optional) ──────────────────────────────────────────
if [[ "${1:-}" == "--with-dataco" ]]; then
    DATACO_PATH="data/raw/dataco_supply_chain.csv"
    if [[ -f "$DATACO_PATH" ]]; then
        echo "[ Step 4/4 ] Running DataCo real-world validation ..."
        python experiments/run_dataco_eval.py
        echo "  ✓ DataCo validation complete"
    else
        echo "[ Step 4/4 ] DataCo dataset not found at $DATACO_PATH"
        echo "  Download: https://doi.org/10.17632/8gx2fvg2k6.5"
    fi
else
    echo "[ Step 4/4 ] DataCo validation skipped (pass --with-dataco to enable)"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  All experiments complete. Results saved to: $RESULTS_DIR/"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
ls -lh "$RESULTS_DIR/" 2>/dev/null || true
ls -lh "$RESULTS_DIR/figures/" 2>/dev/null || true
