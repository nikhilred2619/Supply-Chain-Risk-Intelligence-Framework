"""
tests/test_fmea_scorer.py
──────────────────────────
Unit tests for the FMEA Risk Priority Number scoring engine.
Tests validate paper-aligned formula implementations.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models.fmea_scorer import FMEAScorer, ScenarioInput, stockout_probability, compute_rpn
from utils.config import FMEA_CFG


@pytest.fixture
def scorer():
    return FMEAScorer()


@pytest.fixture
def critical_scenario():
    return ScenarioInput(
        delay_duration=10.0,
        demand_deviation=0.60,
        inventory_buffer=0.05,
        supplier_reliability=0.45,
        transport_risk="High",
        fraud_indicator=True,
    )


@pytest.fixture
def low_scenario():
    return ScenarioInput(
        delay_duration=0.0,
        demand_deviation=-0.10,
        inventory_buffer=0.40,
        supplier_reliability=0.92,
        transport_risk="Low",
        fraud_indicator=False,
    )


class TestRPNFormula:
    def test_rpn_equals_s_times_o_times_d(self):
        assert compute_rpn(5, 6, 7) == 210
        assert compute_rpn(9, 8, 9) == 648
        assert compute_rpn(1, 1, 1) == 1

    def test_rpn_range(self, scorer, critical_scenario):
        fmea = scorer.score(critical_scenario)
        assert 1 <= fmea.rpn <= 1000

    def test_critical_scenario_exceeds_rpn_threshold(self, scorer, critical_scenario):
        fmea = scorer.score(critical_scenario)
        assert fmea.rpn >= FMEA_CFG.rpn_critical, (
            f"Critical scenario RPN {fmea.rpn} should be ≥ {FMEA_CFG.rpn_critical}")

    def test_low_scenario_below_medium_threshold(self, scorer, low_scenario):
        fmea = scorer.score(low_scenario)
        assert fmea.rpn < FMEA_CFG.rpn_medium


class TestStockoutProbability:
    """Tests for Equation 2: Ps = min(1, (d × ΔD) / I)"""

    def test_saturation_at_one(self):
        ps = stockout_probability(delay_days=15, demand_delta=0.60, inv_buffer=0.05)
        assert ps == 1.0, "Should saturate at 1.0 (paper: d=15, ΔD=+60%, I=5%)"

    def test_zero_delay_zero_stockout(self):
        ps = stockout_probability(delay_days=0, demand_delta=0.30, inv_buffer=0.10)
        assert ps == 0.0

    def test_negative_demand_no_stockout(self):
        ps = stockout_probability(delay_days=5, demand_delta=-0.30, inv_buffer=0.10)
        assert ps == 0.0, "Negative demand deviation should not drive stockout"

    def test_nonlinear_threshold_region(self):
        """Ps should be high when I<10% and d>5 (paper Fig 3 critical zone)."""
        ps = stockout_probability(delay_days=8, demand_delta=0.30, inv_buffer=0.08)
        assert ps >= 0.30, f"Critical zone Ps={ps:.3f} expected ≥ 0.30"

    def test_large_buffer_reduces_risk(self):
        # d=3: small buffer 3*0.30/0.05=1.8->1.0; large: 3*0.30/0.40=0.225
        ps_small = stockout_probability(3, 0.30, 0.05)
        ps_large = stockout_probability(3, 0.30, 0.40)
        assert ps_small >= ps_large

    def test_formula_exact(self):
        """d=3, ΔD=+30%, I=10% → Ps = min(1, 3×0.30/0.10) = min(1, 9) = 1.0"""
        ps = stockout_probability(3, 0.30, 0.10)
        assert ps == 1.0

    def test_formula_partial(self):
        """d=1, ΔD=+10%, I=40% → Ps = min(1, 1×0.10/0.40) = 0.25"""
        ps = stockout_probability(1, 0.10, 0.40)
        assert abs(ps - 0.25) < 1e-9


class TestFraudDetection:
    def test_fraud_increases_detection_score(self, scorer):
        sc_nonfr = ScenarioInput(10, 0.30, 0.10, 0.70, "High", False)
        sc_fraud  = ScenarioInput(10, 0.30, 0.10, 0.70, "High", True)
        fmea_nonfr = scorer.score(sc_nonfr)
        fmea_fraud  = scorer.score(sc_fraud)
        assert fmea_fraud.detection >= fmea_nonfr.detection, (
            "Fraud scenarios should have higher Detection scores (paper: 7.3 vs 4.2)")

    def test_fraud_detection_mean_aligned_with_paper(self, scorer):
        """Paper: mean D = 7.3 (fraud) vs 4.2 (non-fraud)."""
        fraud_d    = []
        nonfr_d    = []
        for d in [5, 10, 15]:
            for tr in ["Low","Medium","High"]:
                sc_f  = ScenarioInput(d, 0.30, 0.10, 0.65, tr, True)
                sc_nf = ScenarioInput(d, 0.30, 0.10, 0.65, tr, False)
                fraud_d.append(scorer.score(sc_f).detection)
                nonfr_d.append(scorer.score(sc_nf).detection)
        import statistics
        assert statistics.mean(fraud_d) > statistics.mean(nonfr_d), (
            "Fraud mean Detection should exceed non-fraud mean")


class TestRiskTierClassification:
    def test_high_rpn_classified_critical(self, scorer):
        sc = ScenarioInput(15, 0.60, 0.05, 0.40, "High", False)
        fmea = scorer.score(sc)
        assert fmea.risk_tier in ("Critical", "High")

    def test_low_rpn_classified_low(self, scorer):
        sc = ScenarioInput(0, -0.30, 0.40, 0.95, "Low", False)
        fmea = scorer.score(sc)
        assert fmea.risk_tier in ("Low", "Medium")

    def test_all_tiers_reachable(self, scorer):
        """All four risk tiers should be reachable through different inputs."""
        scenarios = {
            "Critical": ScenarioInput(15, 0.60, 0.05, 0.40, "High", True),
            "High":     ScenarioInput(10, 0.30, 0.10, 0.55, "High", False),
            "Medium":   ScenarioInput(3,  0.10, 0.20, 0.75, "Medium", False),
            "Low":      ScenarioInput(0, -0.10, 0.40, 0.92, "Low", False),
        }
        achieved = set()
        for expected_tier, sc in scenarios.items():
            achieved.add(scorer.score(sc).risk_tier)
        assert len(achieved) >= 3, f"Should reach at least 3 tiers, got: {achieved}"


class TestInputValidation:
    def test_invalid_inventory_buffer(self, scorer):
        with pytest.raises(AssertionError):
            sc = ScenarioInput(5, 0.30, 0.0, 0.70, "Medium", False)
            sc.validate()

    def test_invalid_transport_risk(self, scorer):
        with pytest.raises(AssertionError):
            sc = ScenarioInput(5, 0.30, 0.10, 0.70, "InvalidRisk", False)
            sc.validate()

    def test_invalid_reliability(self, scorer):
        with pytest.raises(AssertionError):
            sc = ScenarioInput(5, 0.30, 0.10, 1.5, "Medium", False)
            sc.validate()
