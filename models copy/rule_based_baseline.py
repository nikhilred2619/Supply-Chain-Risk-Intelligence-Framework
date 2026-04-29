"""
models/rule_based_baseline.py
──────────────────────────────
Rule-Based System (RBS) baseline — fixed RPN thresholds, no training.

Represents current industry practice: manually defined threshold rules
applied to FMEA scores without machine learning or LLM augmentation.
Paper: comparison baseline in Tables II, VI, VII.
"""

from __future__ import annotations
from models.fmea_scorer import FMEAScorer, ScenarioInput
from utils.config import FMEA_CFG
import numpy as np


class RuleBasedSystem:
    """
    Rule-Based System baseline.

    Applies fixed RPN thresholds without training.
    Binary classification threshold calibrated for max F1 on training split.
    """

    def __init__(self):
        self.scorer   = FMEAScorer()
        self.cfg      = FMEA_CFG
        self._threshold = self.cfg.rpn_high   # 150 — threshold for "high risk"

    def predict_binary(self, scenario: ScenarioInput) -> int:
        fmea = self.scorer.score(scenario)
        return 1 if fmea.rpn >= self._threshold else 0

    def predict_proba(self, scenario: ScenarioInput) -> float:
        fmea = self.scorer.score(scenario)
        return float(np.clip(fmea.rpn / 800, 0.0, 1.0))

    def predict_batch(self, scenarios: list[ScenarioInput]) -> list[int]:
        return [self.predict_binary(sc) for sc in scenarios]

    def predict_proba_batch(self, scenarios: list[ScenarioInput]) -> list[float]:
        return [self.predict_proba(sc) for sc in scenarios]
