"""
models/fmea_scorer.py
──────────────────────
Core FMEA Risk Priority Number (RPN) scoring engine.

Implements the deterministic risk quantification layer (Layer 3) of the
LLM-FMEA framework, exactly as specified in the research paper:

    RPN  = S × O × D                              (Equation 1)
    Ps   = min(1, (d × ΔD) / I)                  (Equation 2)

References
----------
- Sharma et al. (2022) IEEE Trans. Engineering Management 69(4):1345-1356
- Scarf (1958) newsvendor framework
- Silver, Pyke & Thomas (2017) Inventory and Production Management
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from utils.config import FMEA_CFG, TRANSPORT_RISK_MAP
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ScenarioInput:
    """
    Structured input for a single supply chain risk scenario.

    Parameters align with paper Section IV-A variable domains:
      d       ∈ {0,1,3,5,10,15} days
      delta_d ∈ {-0.30,-0.10,0.10,0.30,0.60}
      inv_buf ∈ {0.05,0.10,0.20,0.40}  (fraction, NOT percent)
      rel     ∈ [0.40,0.95]
      tr      ∈ {"Low","Medium","High"}
      fraud   ∈ {True, False}
    """
    delay_duration:      float           # d  — days
    demand_deviation:    float           # ΔD — fractional (+0.30 = +30%)
    inventory_buffer:    float           # I  — fractional (0.10 = 10%)
    supplier_reliability: float          # R  — [0, 1]
    transport_risk:      str             # TR — "Low"/"Medium"/"High"
    fraud_indicator:     bool = False    # Binary fraud flag

    def validate(self) -> None:
        assert self.delay_duration >= 0,               "delay_duration must be ≥ 0"
        assert -1.0 <= self.demand_deviation <= 2.0,   "demand_deviation out of range"
        assert 0 < self.inventory_buffer <= 1.0,       "inventory_buffer must be in (0,1]"
        assert 0.0 <= self.supplier_reliability <= 1.0,"supplier_reliability must be in [0,1]"
        assert self.transport_risk in TRANSPORT_RISK_MAP, \
            f"transport_risk must be one of {list(TRANSPORT_RISK_MAP)}"


@dataclass
class FMEAScores:
    """All FMEA-derived scores for a single scenario."""
    severity:            int     # S ∈ [1,10]
    occurrence:          int     # O ∈ [1,10]
    detection:           int     # D ∈ [1,10]
    rpn:                 int     # RPN = S × O × D ∈ [1,1000]
    stockout_probability: float  # Ps = min(1,(d×ΔD)/I)
    service_level_drop:  float   # ΔSL (derived)
    cost_impact:         float   # CI  (derived)
    risk_tier:           str     # Critical/High/Medium/Low
    rpn_contribution:    dict    # breakdown for explainability


class FMEAScorer:
    """
    FMEA Risk Priority Number scoring engine.

    Implements deterministic, auditable risk quantification as described
    in the paper's Layer 3. All scoring functions are calibrated to
    industry benchmarks (Chopra & Sodhi 2004; Sharma et al. 2022).
    """

    def __init__(self):
        self.cfg = FMEA_CFG
        log.info("FMEAScorer initialized [RPN thresholds: Critical≥%d, High≥%d, Medium≥%d]",
                 self.cfg.rpn_critical, self.cfg.rpn_high, self.cfg.rpn_medium)

    # ──────────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────────

    def score(self, scenario: ScenarioInput, llm_detection_boost: Optional[int] = None) -> FMEAScores:
        """
        Compute all FMEA scores for a supply chain scenario.

        Parameters
        ----------
        scenario : ScenarioInput
        llm_detection_boost : int, optional
            Override for Detection score from LLM reasoning layer (Layer 4).
            Enables the hybrid augmentation studied in the ablation analysis.

        Returns
        -------
        FMEAScores
        """
        scenario.validate()

        s = self._severity(scenario)
        o = self._occurrence(scenario)
        d = llm_detection_boost if llm_detection_boost is not None else self._detection(scenario)

        # Clamp to [1,10]
        s, o, d = (max(1, min(10, v)) for v in (s, o, d))

        rpn = s * o * d

        ps  = self._stockout_probability(scenario)
        sl  = self._service_level_drop(scenario, ps)
        ci  = self._cost_impact(scenario, rpn, ps)

        tier = self._classify_tier(rpn, ps)

        return FMEAScores(
            severity=s,
            occurrence=o,
            detection=d,
            rpn=rpn,
            stockout_probability=ps,
            service_level_drop=sl,
            cost_impact=ci,
            risk_tier=tier,
            rpn_contribution={"S": s, "O": o, "D": d,
                               "S×O": s * o, "RPN": rpn},
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Severity — S ∈ [1,10]
    # Calibrated to Cost Impact and Service Level Drop (paper Section IV-B)
    # ──────────────────────────────────────────────────────────────────────────

    def _severity(self, sc: ScenarioInput) -> int:
        base = 1

        # Delay contribution (non-linear escalation; paper Fig 3)
        if sc.delay_duration >= 10:
            base += 5
        elif sc.delay_duration >= 5:
            base += 3
        elif sc.delay_duration >= 3:
            base += 2
        elif sc.delay_duration >= 1:
            base += 1

        # Demand deviation contribution
        dd = abs(sc.demand_deviation)
        if dd >= 0.50:
            base += 3
        elif dd >= 0.30:
            base += 2
        elif dd >= 0.10:
            base += 1

        # Inventory buffer depletion risk
        if sc.inventory_buffer < 0.05:
            base += 2
        elif sc.inventory_buffer < 0.10:
            base += 1

        # Stockout probability amplifier
        ps = self._stockout_probability(sc)
        if ps >= 0.80:
            base += 1

        return min(10, base)

    # ──────────────────────────────────────────────────────────────────────────
    # Occurrence — O ∈ [1,10]
    # Supplier reliability and transport risk contribute to disruption frequency
    # ──────────────────────────────────────────────────────────────────────────

    def _occurrence(self, sc: ScenarioInput) -> int:
        # Supplier reliability: low R → high occurrence
        r = sc.supplier_reliability
        if r < 0.50:
            o = 8
        elif r < 0.65:
            o = 6
        elif r < 0.75:
            o = 5
        elif r < 0.85:
            o = 4
        elif r < 0.92:
            o = 3
        else:
            o = 2

        # Transport risk additive
        tr_map = {"Low": 0, "Medium": 1, "High": 2}
        o += tr_map.get(sc.transport_risk, 0)

        # Fraud elevates structural occurrence (paper: fraud scenarios systematically elevated)
        if sc.fraud_indicator:
            o += 1

        return min(10, max(1, o))

    # ──────────────────────────────────────────────────────────────────────────
    # Detection — D ∈ [1,10]
    # Difficulty of early identification; fraud raises D significantly
    # Paper: mean D = 7.3 (fraud) vs 4.2 (non-fraud)
    # ──────────────────────────────────────────────────────────────────────────

    def _detection(self, sc: ScenarioInput) -> int:
        if sc.fraud_indicator:
            # Integrity-based failures are inherently difficult to detect early
            d = 7
            # Low transport visibility raises detection difficulty further
            if sc.transport_risk == "High":
                d += 1
        else:
            d = 4
            # Long delays with low inventory → easier to detect (metrics visible)
            if sc.delay_duration > 5 and sc.inventory_buffer < 0.10:
                d -= 1
            # High demand spike → harder to distinguish from organic demand
            if abs(sc.demand_deviation) > 0.40:
                d += 1

        return min(10, max(1, d))

    # ──────────────────────────────────────────────────────────────────────────
    # Stockout Probability — Ps = min(1, (d × ΔD) / I)   [Equation 2]
    # Paper Section IV-C; newsvendor framework (Scarf 1958)
    # ──────────────────────────────────────────────────────────────────────────

    def _stockout_probability(self, sc: ScenarioInput) -> float:
        delta_d = max(0.0, sc.demand_deviation)   # only positive demand surges drive stockout
        if sc.inventory_buffer <= 0:
            return 1.0
        ps = (sc.delay_duration * delta_d) / sc.inventory_buffer
        return float(np.clip(ps, 0.0, 1.0))

    # ──────────────────────────────────────────────────────────────────────────
    # Service Level Drop — ΔSL (%)
    # Deterministic weighted aggregation calibrated to industry benchmarks
    # ──────────────────────────────────────────────────────────────────────────

    def _service_level_drop(self, sc: ScenarioInput, ps: float) -> float:
        base_sl_drop = ps * 0.50   # maximum 50% SL drop at Ps=1
        reliability_penalty = (1 - sc.supplier_reliability) * 0.15
        transport_penalty   = (TRANSPORT_RISK_MAP[sc.transport_risk] - 1) * 0.04
        return float(np.clip(base_sl_drop + reliability_penalty + transport_penalty, 0.0, 0.50))

    # ──────────────────────────────────────────────────────────────────────────
    # Cost Impact — CI (fractional cost increase)
    # Paper Table II: range +4% to +75%, mean +22.8%
    # ──────────────────────────────────────────────────────────────────────────

    def _cost_impact(self, sc: ScenarioInput, rpn: int, ps: float) -> float:
        # RPN-to-cost mapping (paper: r=0.89 correlation)
        rpn_component = (rpn / 800) * 0.55    # normalised RPN contribution
        ps_component  = ps * 0.25
        fraud_penalty = 0.08 if sc.fraud_indicator else 0.0
        transport_mod = (TRANSPORT_RISK_MAP[sc.transport_risk] - 1) * 0.05
        ci = 0.04 + rpn_component + ps_component + fraud_penalty + transport_mod
        return float(np.clip(ci, 0.04, 0.75))

    # ──────────────────────────────────────────────────────────────────────────
    # Risk Tier Classification
    # Four-tier classification: Critical / High / Medium / Low
    # Primary driver: RPN; secondary signal: stockout probability
    # ──────────────────────────────────────────────────────────────────────────

    def _classify_tier(self, rpn: int, ps: float) -> str:
        # Non-linear threshold override (paper Section VI-C critical zone)
        if ps >= self.cfg.ps_critical_threshold and rpn >= self.cfg.rpn_high:
            return "Critical"

        if rpn >= self.cfg.rpn_critical:
            return "Critical"
        elif rpn >= self.cfg.rpn_high:
            return "High"
        elif rpn >= self.cfg.rpn_medium:
            return "Medium"
        else:
            return "Low"


# ── Convenience function ──────────────────────────────────────────────────────

def compute_rpn(severity: int, occurrence: int, detection: int) -> int:
    """Direct RPN computation: S × O × D."""
    return severity * occurrence * detection


def stockout_probability(delay_days: float, demand_delta: float, inv_buffer: float) -> float:
    """
    Stockout Probability formula (Equation 2).

    Parameters
    ----------
    delay_days   : float — delay duration d
    demand_delta : float — demand deviation ΔD (fractional)
    inv_buffer   : float — inventory buffer I (fractional)
    """
    if inv_buffer <= 0:
        return 1.0
    delta_d = max(0.0, demand_delta)
    return float(np.clip((delay_days * delta_d) / inv_buffer, 0.0, 1.0))
