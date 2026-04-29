"""
experiments/run_simulation.py
──────────────────────────────
Full synthetic benchmark experiment.

Reproduces all paper results on the 120-scenario parameterized dataset:
  - RPN–cost correlation (r = 0.89)
  - Risk level distribution (Table I / Table IV)
  - Model comparison (Table VI)
  - Ablation study (Table VII)
  - McNemar's test (Section VI-E)
  - Non-linear threshold analysis (Fig. 3)

Run:
    python experiments/run_simulation.py
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

from utils.data_loader import generate_synthetic_scenarios
from utils.config import EVAL_CFG, RESULTS_DIR, SIM_CFG
from utils.logger import get_logger
from services.pipeline import LLMFMEAPipeline
from models.rule_based_baseline import RuleBasedSystem
from models.decision_tree_model import DecisionTreeBaseline
from models.fmea_scorer import FMEAScorer, ScenarioInput
from evaluation.metrics import (
    compute_metrics, mcnemar_test, cross_validate, compute_roc, confusion_matrix_report
)

log = get_logger("simulation_experiment")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_simulation_experiment():
    """Full reproduction of paper simulation benchmark results."""
    log.info("=" * 70)
    log.info("  LLM-FMEA Simulation Benchmark Experiment")
    log.info("  Paper: Sections IV, VI — 120-Scenario Parameterized Benchmark")
    log.info("=" * 70)

    # ── 1. Generate synthetic dataset ─────────────────────────────────────
    log.info("\n[1/7] Generating 120-scenario synthetic benchmark ...")
    scenarios, labels, df = generate_synthetic_scenarios(
        n=SIM_CFG.n_scenarios,
        seed=SIM_CFG.random_seed,
        save_path=Path("data/synthetic"),
    )

    tier_counts = {t: labels.count(t) for t in ["Critical","High","Medium","Low"]}
    log.info("  Tier distribution: %s", tier_counts)

    # Convert string labels to binary (Critical/High = 1, Medium/Low = 0)
    binary_labels = [1 if l in ("Critical","High") else 0 for l in labels]

    # ── 2. Train/test split ───────────────────────────────────────────────
    log.info("\n[2/7] Splitting 80/20 stratified ...")
    sc_train, sc_test, y_train, y_test, l_train, l_test = train_test_split(
        scenarios, binary_labels, labels,
        test_size=EVAL_CFG.test_split if hasattr(EVAL_CFG,'test_split') else 0.20,
        stratify=binary_labels,
        random_state=EVAL_CFG.random_seed,
    )
    log.info("  Train: %d  Test: %d", len(sc_train), len(sc_test))

    # ── 3. FMEA scoring on full set (for RPN-cost correlation) ────────────
    log.info("\n[3/7] Computing FMEA scores and RPN-Cost correlation ...")
    scorer   = FMEAScorer()
    pipeline = LLMFMEAPipeline(mock_llm=True)

    rpn_scores   = []
    cost_impacts = []
    for sc in scenarios:
        fmea = scorer.score(sc)
        rpn_scores.append(fmea.rpn)
        cost_impacts.append(fmea.cost_impact)

    r, p_val = pearsonr(rpn_scores, cost_impacts)
    log.info("  RPN–Cost Pearson r = %.3f (p = %.4f)  [paper: r=0.89, p<0.001]", r, p_val)

    # ── 4. Model predictions on test set ─────────────────────────────────
    log.info("\n[4/7] Running model comparisons on test set (n=%d) ...", len(sc_test))

    # Rule-Based System
    rbs = RuleBasedSystem()
    y_pred_rbs   = rbs.predict_batch(sc_test)
    y_proba_rbs  = rbs.predict_proba_batch(sc_test)

    # Decision Tree
    dt = DecisionTreeBaseline()
    dt.fit(sc_train, y_train)
    y_pred_dt    = dt.predict_batch(sc_test)
    y_proba_dt   = dt.predict_proba_batch(sc_test)

    # LLM-FMEA
    y_pred_llm   = [pipeline.predict_binary(sc) for sc in sc_test]
    y_proba_llm  = [pipeline.predict_proba(sc)  for sc in sc_test]

    # Metrics
    metrics_rbs = compute_metrics(y_test, y_pred_rbs, y_proba_rbs, "Rule-Based")
    metrics_dt  = compute_metrics(y_test, y_pred_dt,  y_proba_dt,  "Decision Tree")
    metrics_llm = compute_metrics(y_test, y_pred_llm, y_proba_llm, "LLM-FMEA")

    print("\n" + "─"*65)
    print("  TABLE VI.  Classification Performance — Simulation Benchmark")
    print("─"*65)
    header = f"{'Model':<20} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}"
    print(header)
    print("─"*65)
    for m in [metrics_rbs, metrics_dt, metrics_llm]:
        print(f"  {m.model_name:<18} {m.accuracy*100:>7.1f}%"
              f" {m.precision*100:>7.1f}% {m.recall*100:>7.1f}%"
              f" {m.f1*100:>7.1f}%")
    print("─"*65)

    # ── 5. McNemar's test ─────────────────────────────────────────────────
    log.info("\n[5/7] McNemar's significance test (LLM-FMEA vs RBS) ...")
    mcnemar = mcnemar_test(y_test, y_pred_llm, y_pred_rbs, "LLM-FMEA", "Rule-Based")
    print(f"\n  McNemar's test: χ²={mcnemar.chi2:.1f}, p={mcnemar.p_value:.4f}")
    print(f"  → {mcnemar.interpretation}")

    # ── 6. Ablation study ────────────────────────────────────────────────
    log.info("\n[6/7] Ablation study — component contributions ...")
    _run_ablation_simulation(sc_train, y_train, sc_test, y_test, scenarios, binary_labels)

    # ── 7. Non-linear threshold analysis ────────────────────────────────
    log.info("\n[7/7] Non-linear threshold analysis (I<10%% + d>5 days) ...")
    _nonlinear_threshold_analysis(scenarios)

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        "rpn_cost_correlation": {"r": round(r, 4), "p_value": round(p_val, 6)},
        "performance": {
            "rule_based": metrics_rbs.as_row(),
            "decision_tree": metrics_dt.as_row(),
            "llm_fmea": metrics_llm.as_row(),
        },
        "mcnemar": {
            "b": mcnemar.b, "d": mcnemar.d,
            "chi2": round(mcnemar.chi2, 2),
            "p_value": round(mcnemar.p_value, 6),
            "significant": mcnemar.significant,
        },
        "tier_distribution": tier_counts,
        "confusion_matrix_llm_fmea": confusion_matrix_report(y_test, y_pred_llm),
    }

    out_path = RESULTS_DIR / "simulation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("\nResults saved → %s", out_path)
    log.info("Simulation experiment complete.")
    return results


def _run_ablation_simulation(sc_train, y_train, sc_test, y_test, scenarios, binary_labels):
    """Ablation study: FMEA-Only vs LLM-Only vs LLM-FMEA (Full)."""
    from models.fmea_scorer import FMEAScorer
    scorer = FMEAScorer()

    # FMEA-Only: predict based on RPN with no LLM adjustment
    def fmea_only_predict(sc):
        fmea = scorer.score(sc)
        return 1 if fmea.rpn >= 150 else 0

    # LLM-Only: LLM confidence signal only (no FMEA RPN thresholding)
    from models.llm_reasoner import LLMReasoner
    from models.fmea_scorer import FMEAScores as FS
    llm = LLMReasoner(mock_mode=True)

    def llm_only_predict(sc):
        fmea = scorer.score(sc)
        out  = llm.reason(sc, fmea)
        return 1 if out.confidence >= 0.72 else 0  # threshold for recall parity

    pipeline = LLMFMEAPipeline(mock_llm=True)

    y_fmea_only = [fmea_only_predict(sc) for sc in sc_test]
    y_llm_only  = [llm_only_predict(sc)  for sc in sc_test]
    y_full      = [pipeline.predict_binary(sc) for sc in sc_test]

    m_fmea = compute_metrics(y_test, y_fmea_only, model_name="FMEA Only")
    m_llm  = compute_metrics(y_test, y_llm_only,  model_name="LLM Only")
    m_full = compute_metrics(y_test, y_full,       model_name="LLM-FMEA (Full)")

    print("\n" + "─"*65)
    print("  TABLE VII.  Ablation Study — Component Contributions")
    print("─"*65)
    for m in [m_fmea, m_llm, m_full]:
        print(f"  {m.model_name:<20} {m.accuracy*100:>7.1f}%"
              f" {m.precision*100:>7.1f}% {m.recall*100:>7.1f}%"
              f" {m.f1*100:>7.1f}%")
    print("─"*65)
    print(f"  Full vs FMEA-Only: ΔF1 = +{(m_full.f1 - m_fmea.f1)*100:.1f}pp "
          f"  [paper: +15.7pp]")
    print(f"  Full vs LLM-Only:  ΔF1 = +{(m_full.f1 - m_llm.f1)*100:.1f}pp "
          f"  [paper: +14.5pp]")


def _nonlinear_threshold_analysis(scenarios):
    """Identify non-linear escalation at I<10%% + d>5 days."""
    scorer = FMEAScorer()
    critical_zone = []
    normal_zone   = []

    for sc in scenarios:
        fmea = scorer.score(sc)
        if sc.inventory_buffer < 0.10 and sc.delay_duration > 5:
            critical_zone.append(fmea.stockout_probability)
        else:
            normal_zone.append(fmea.stockout_probability)

    if critical_zone:
        log.info("  Critical zone (I<10%% + d>5d): n=%d, mean_Ps=%.3f  [paper: Ps→1.0]",
                 len(critical_zone), np.mean(critical_zone))
    if normal_zone:
        log.info("  Normal zone: n=%d, mean_Ps=%.3f", len(normal_zone), np.mean(normal_zone))


if __name__ == "__main__":
    run_simulation_experiment()
