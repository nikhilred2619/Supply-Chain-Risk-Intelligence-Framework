"""
experiments/run_dataco_eval.py
───────────────────────────────
DataCo Smart Supply Chain dataset validation experiment.

Reproduces paper primary empirical results (Section V):
  - Classification performance (Table II)
  - McNemar's test on 36,104 test records (Section V-B)
  - 5-fold cross-validation (Table III)
  - ROC-AUC analysis (Section V-D / Fig 4)

Requires DataCo dataset: https://doi.org/10.17632/8gx2fvg2k6.5
Place at: data/raw/dataco_supply_chain.csv

Run:
    python experiments/run_dataco_eval.py
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from utils.data_loader import load_dataco
from utils.config import EVAL_CFG, RESULTS_DIR
from utils.logger import get_logger
from services.pipeline import LLMFMEAPipeline
from models.rule_based_baseline import RuleBasedSystem
from models.decision_tree_model import DecisionTreeBaseline
from evaluation.metrics import (
    compute_metrics, mcnemar_test, cross_validate, compute_roc, confusion_matrix_report
)

log = get_logger("dataco_experiment")


def run_dataco_experiment():
    log.info("=" * 70)
    log.info("  LLM-FMEA DataCo External Validation Experiment")
    log.info("  Paper Section V — Primary Empirical Results (n=180,519)")
    log.info("=" * 70)

    # ── Load dataset ──────────────────────────────────────────────────────
    log.info("\n[1/5] Loading DataCo dataset ...")
    try:
        scenarios, labels, df = load_dataco()
    except FileNotFoundError as e:
        log.error(str(e))
        return

    pos_rate = sum(labels) / len(labels)
    log.info("  Records: %d | High-risk: %.1f%% (paper: 59.1%%)",
             len(labels), pos_rate * 100)

    # ── Train/test split (80/20 stratified) ──────────────────────────────
    log.info("\n[2/5] Stratified 80/20 split ...")
    sc_train, sc_test, y_train, y_test = train_test_split(
        scenarios, labels,
        test_size=0.20,
        stratify=labels,
        random_state=EVAL_CFG.random_seed,
    )
    log.info("  Train: %d | Test: %d", len(sc_train), len(sc_test))

    # ── Model predictions ─────────────────────────────────────────────────
    log.info("\n[3/5] Running model predictions on test set (n=%d) ...", len(sc_test))

    rbs = RuleBasedSystem()
    log.info("  Rule-Based System ...")
    y_pred_rbs  = rbs.predict_batch(sc_test)
    y_proba_rbs = rbs.predict_proba_batch(sc_test)

    dt = DecisionTreeBaseline()
    dt.fit(sc_train, y_train)
    log.info("  Decision Tree ...")
    y_pred_dt   = dt.predict_batch(sc_test)
    y_proba_dt  = dt.predict_proba_batch(sc_test)

    pipeline = LLMFMEAPipeline(mock_llm=True)
    log.info("  LLM-FMEA (processing %d records — may take a moment) ...", len(sc_test))
    y_pred_llm  = [pipeline.predict_binary(sc) for sc in sc_test]
    y_proba_llm = [pipeline.predict_proba(sc)  for sc in sc_test]

    # ── Performance metrics ───────────────────────────────────────────────
    m_rbs = compute_metrics(y_test, y_pred_rbs, y_proba_rbs, "Rule-Based")
    m_dt  = compute_metrics(y_test, y_pred_dt,  y_proba_dt,  "Decision Tree")
    m_llm = compute_metrics(y_test, y_pred_llm, y_proba_llm, "LLM-FMEA")

    print("\n" + "─"*68)
    print("  TABLE II.  Classification Performance — DataCo (n=%d)" % len(sc_test))
    print("─"*68)
    print(f"  {'Model':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
    print("─"*68)
    for m in [m_rbs, m_dt, m_llm]:
        auc_str = f"{m.auc_roc:.3f}" if m.auc_roc else "  —  "
        print(f"  {m.model_name:<22} {m.accuracy*100:>6.1f}%"
              f" {m.precision*100:>6.1f}% {m.recall*100:>6.1f}%"
              f" {m.f1*100:>6.1f}% {auc_str:>7}")
    print("─"*68)
    print("  Paper targets: LLM-FMEA Recall=96.8%, F1=73.8%, AUC=0.760")

    # ── McNemar's test ────────────────────────────────────────────────────
    log.info("\n[4/5] McNemar's test on DataCo test set ...")
    mcnemar = mcnemar_test(y_test, y_pred_llm, y_pred_rbs)
    print(f"\n  McNemar's: b={mcnemar.b}, d={mcnemar.d}, "
          f"χ²={mcnemar.chi2:.1f}, p={mcnemar.p_value:.6f}")
    print(f"  [paper: χ²≈156.4, p<0.001]")

    # ── 5-fold CV (on a 20k sample for efficiency) ─────────────────────
    log.info("\n[5/5] 5-fold stratified cross-validation (sample n=20,000) ...")
    sample_idx = np.random.default_rng(42).choice(
        len(scenarios), size=min(20000, len(scenarios)), replace=False)
    sc_sample = [scenarios[i] for i in sample_idx]
    y_sample  = [labels[i]    for i in sample_idx]

    cv_rbs = cross_validate(rbs.predict_binary, sc_sample, y_sample,
                             "Rule-Based", n_folds=5)
    cv_llm = cross_validate(pipeline.predict_binary, sc_sample, y_sample,
                             "LLM-FMEA", n_folds=5)

    print("\n" + "─"*68)
    print("  TABLE III.  5-Fold Cross-Validation — DataCo")
    print("─"*68)
    for cv in [cv_rbs, cv_llm]:
        print(f"  {cv.model_name:<22} "
              f"Recall: {cv.recall_mean*100:.1f}%±{cv.recall_std*100:.1f}%  "
              f"F1: {cv.f1_mean*100:.1f}%±{cv.f1_std*100:.1f}%")
    print("─"*68)

    # ── Save ──────────────────────────────────────────────────────────────
    results = {
        "dataset": {"n_total": len(labels), "n_test": len(y_test),
                    "positive_rate": round(pos_rate, 4)},
        "performance": {
            "rule_based": m_rbs.as_row(),
            "decision_tree": m_dt.as_row(),
            "llm_fmea": m_llm.as_row(),
        },
        "mcnemar": {"b": mcnemar.b, "d": mcnemar.d,
                    "chi2": round(mcnemar.chi2, 2),
                    "p_value": round(mcnemar.p_value, 8),
                    "significant": mcnemar.significant},
        "cross_validation": {
            "rule_based": cv_rbs.as_row(),
            "llm_fmea": cv_llm.as_row(),
        },
    }
    out = RESULTS_DIR / "dataco_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved → %s", out)
    return results


if __name__ == "__main__":
    run_dataco_experiment()
