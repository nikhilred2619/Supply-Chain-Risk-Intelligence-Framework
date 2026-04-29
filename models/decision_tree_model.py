"""
models/decision_tree_model.py
──────────────────────────────
Decision Tree baseline — scikit-learn, max_depth=8, information gain.

Trained on six raw input features (no FMEA-derived features).
Paper: Section IV-F (Baseline Systems and Experimental Protocol).
"""

from __future__ import annotations
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

from models.fmea_scorer import ScenarioInput
from utils.config import TRANSPORT_RISK_MAP, EVAL_CFG
from utils.logger import get_logger

log = get_logger(__name__)


class DecisionTreeBaseline:
    """
    Decision Tree baseline (scikit-learn).

    Features: delay_duration, demand_deviation, inventory_buffer,
              supplier_reliability, transport_risk_encoded, fraud_indicator
    max_depth=8, criterion='entropy' (information gain), random_state=42
    """

    def __init__(self):
        self.model = DecisionTreeClassifier(
            max_depth=8,
            criterion="entropy",
            random_state=EVAL_CFG.random_seed,
            class_weight="balanced",
        )
        self._trained = False

    # ──────────────────────────────────────────────────────────────────────────

    def _featurise(self, scenarios: list[ScenarioInput]) -> np.ndarray:
        """Convert ScenarioInput list to feature matrix."""
        rows = []
        for sc in scenarios:
            rows.append([
                sc.delay_duration,
                sc.demand_deviation,
                sc.inventory_buffer,
                sc.supplier_reliability,
                TRANSPORT_RISK_MAP.get(sc.transport_risk, 2),
                int(sc.fraud_indicator),
            ])
        return np.array(rows, dtype=np.float64)

    def fit(self, scenarios: list[ScenarioInput], labels: list[int]) -> None:
        X = self._featurise(scenarios)
        y = np.array(labels)
        self.model.fit(X, y)
        self._trained = True
        train_acc = self.model.score(X, y)
        log.info("DecisionTree trained: train_acc=%.3f, depth=%d",
                 train_acc, self.model.get_depth())

    def predict_binary(self, scenario: ScenarioInput) -> int:
        if not self._trained:
            raise RuntimeError("Call fit() before predict_binary()")
        X = self._featurise([scenario])
        return int(self.model.predict(X)[0])

    def predict_proba(self, scenario: ScenarioInput) -> float:
        if not self._trained:
            raise RuntimeError("Call fit() before predict_proba()")
        X = self._featurise([scenario])
        return float(self.model.predict_proba(X)[0, 1])

    def predict_batch(self, scenarios: list[ScenarioInput]) -> list[int]:
        if not self._trained:
            raise RuntimeError("Call fit() before predict_batch()")
        X = self._featurise(scenarios)
        return self.model.predict(X).tolist()

    def predict_proba_batch(self, scenarios: list[ScenarioInput]) -> list[float]:
        if not self._trained:
            raise RuntimeError("Call fit() before predict_proba_batch()")
        X = self._featurise(scenarios)
        return self.model.predict_proba(X)[:, 1].tolist()
