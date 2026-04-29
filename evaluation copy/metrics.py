"""
evaluation/metrics.py
──────────────────────
Classification evaluation metrics, McNemar's test, cross-validation,
ROC-AUC analysis, and ablation study utilities.

Paper: Section VII (Model Evaluation) — Tables II–IX
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from utils.config import EVAL_CFG
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ClassificationReport:
    """Full classification performance report for one model."""
    model_name:  str
    accuracy:    float
    precision:   float
    recall:      float
    f1:          float
    auc_roc:     Optional[float]
    n_samples:   int

    def as_row(self, pct: bool = True) -> dict:
        mult = 100 if pct else 1
        return {
            "Model":      self.model_name,
            "Accuracy":   f"{self.accuracy*mult:.1f}%",
            "Precision":  f"{self.precision*mult:.1f}%",
            "Recall":     f"{self.recall*mult:.1f}%",
            "F1":         f"{self.f1*mult:.1f}%",
            "AUC-ROC":    f"{self.auc_roc:.3f}" if self.auc_roc else "—",
        }


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_proba: Optional[list[float]] = None,
    model_name: str = "Model",
) -> ClassificationReport:
    """Compute full classification performance metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    auc = None
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            pass

    return ClassificationReport(
        model_name=model_name,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        auc_roc=auc,
        n_samples=len(y_true),
    )


# ══════════════════════════════════════════════════════════════════════════════
# McNemar's Test
# Paper: Sections V-B and VI-E; References [26]
# Q. McNemar (1947) Psychometrika 12(2):153–157
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class McNemarResult:
    """McNemar's test result for paired model comparison."""
    b: int           # Model A correct, Model B incorrect
    d: int           # Model B correct, Model A incorrect
    chi2: float      # McNemar's χ² statistic (continuity-corrected)
    p_value: float
    significant: bool
    interpretation: str


def mcnemar_test(
    y_true:   list[int],
    y_pred_a: list[int],   # Proposed model (LLM-FMEA)
    y_pred_b: list[int],   # Baseline model (RBS)
    model_a:  str = "LLM-FMEA",
    model_b:  str = "Rule-Based",
    alpha:    float = 0.05,
) -> McNemarResult:
    """
    McNemar's test for paired binary classifications.

    Continuity-corrected statistic (paper Equation 3):
        χ² = (|b − d| − 1)² / (b + d)

    b = A correct, B incorrect (A-favourable discordant pairs)
    d = B correct, A incorrect (B-favourable discordant pairs)
    """
    y_true   = np.array(y_true)
    y_pred_a = np.array(y_pred_a)
    y_pred_b = np.array(y_pred_b)

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    b = int(np.sum(correct_a & ~correct_b))   # A correct, B wrong
    d = int(np.sum(~correct_a & correct_b))   # B correct, A wrong

    if (b + d) == 0:
        chi2, p_value = 0.0, 1.0
    else:
        # Continuity-corrected McNemar's χ²
        chi2 = (abs(b - d) - 1) ** 2 / (b + d)
        p_value = float(stats.chi2.sf(chi2, df=1))

    significant = p_value < alpha

    interp = (
        f"{model_a} significantly outperforms {model_b} "
        f"(b={b}, d={d}, χ²={chi2:.1f}, p={p_value:.4f})"
        if significant else
        f"No statistically significant difference between {model_a} and {model_b} "
        f"(b={b}, d={d}, χ²={chi2:.1f}, p={p_value:.4f})"
    )

    log.info("McNemar's test [%s vs %s]: b=%d, d=%d, χ²=%.2f, p=%.4f → %s",
             model_a, model_b, b, d, chi2,
             p_value, "SIGNIFICANT" if significant else "not significant")

    return McNemarResult(
        b=b, d=d, chi2=chi2, p_value=p_value,
        significant=significant, interpretation=interp,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5-Fold Stratified Cross-Validation
# Paper: Section V-C (Table III / Table IX)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CVResult:
    """5-fold stratified cross-validation result."""
    model_name: str
    accuracy_mean:   float;  accuracy_std:   float
    precision_mean:  float;  precision_std:  float
    recall_mean:     float;  recall_std:     float
    f1_mean:         float;  f1_std:         float

    def as_row(self) -> dict:
        def fmt(m, s): return f"{m*100:.1f}±{s*100:.1f}%"
        return {
            "Model":     self.model_name,
            "Accuracy":  fmt(self.accuracy_mean,  self.accuracy_std),
            "Precision": fmt(self.precision_mean, self.precision_std),
            "Recall":    fmt(self.recall_mean,    self.recall_std),
            "F1":        fmt(self.f1_mean,        self.f1_std),
        }


def cross_validate(
    predict_fn,       # Callable(ScenarioInput) → int  (already trained externally)
    scenarios: list,
    labels:    list[int],
    model_name: str = "Model",
    n_folds:    int  = 5,
    seed:       int  = 42,
) -> CVResult:
    """
    5-fold stratified cross-validation.

    For models that don't require training (RBS, LLM-FMEA), predict_fn
    is called directly. For Decision Tree, a fit+predict callback is used.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y = np.array(labels)

    fold_metrics = {"acc":[], "prec":[], "rec":[], "f1":[]}

    for fold, (_, test_idx) in enumerate(skf.split(np.zeros(len(labels)), y)):
        test_sc  = [scenarios[i] for i in test_idx]
        y_test   = y[test_idx]
        y_pred   = np.array([predict_fn(sc) for sc in test_sc])

        fold_metrics["acc"].append(accuracy_score(y_test, y_pred))
        fold_metrics["prec"].append(precision_score(y_test, y_pred, zero_division=0))
        fold_metrics["rec"].append(recall_score(y_test, y_pred, zero_division=0))
        fold_metrics["f1"].append(f1_score(y_test, y_pred, zero_division=0))

    res = CVResult(
        model_name=model_name,
        accuracy_mean=np.mean(fold_metrics["acc"]),
        accuracy_std=np.std(fold_metrics["acc"]),
        precision_mean=np.mean(fold_metrics["prec"]),
        precision_std=np.std(fold_metrics["prec"]),
        recall_mean=np.mean(fold_metrics["rec"]),
        recall_std=np.std(fold_metrics["rec"]),
        f1_mean=np.mean(fold_metrics["f1"]),
        f1_std=np.std(fold_metrics["f1"]),
    )
    log.info("CV [%s]: Recall=%.3f±%.3f  F1=%.3f±%.3f",
             model_name, res.recall_mean, res.recall_std,
             res.f1_mean, res.f1_std)
    return res


# ══════════════════════════════════════════════════════════════════════════════
# ROC-AUC Analysis
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ROCResult:
    model_name: str
    fpr:        np.ndarray
    tpr:        np.ndarray
    auc:        float
    thresholds: np.ndarray


def compute_roc(
    y_true:  list[int],
    y_proba: list[float],
    model_name: str = "Model",
) -> ROCResult:
    y_true  = np.array(y_true)
    y_proba = np.array(y_proba)
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    log.info("ROC-AUC [%s]: %.3f", model_name, auc)
    return ROCResult(model_name=model_name, fpr=fpr, tpr=tpr,
                     auc=auc, thresholds=thresholds)


# ══════════════════════════════════════════════════════════════════════════════
# Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════

def confusion_matrix_report(
    y_true: list[int],
    y_pred: list[int],
) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}
    return {"confusion_matrix": cm.tolist()}
