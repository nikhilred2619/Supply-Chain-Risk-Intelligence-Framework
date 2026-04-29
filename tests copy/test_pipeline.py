"""
tests/test_pipeline.py
───────────────────────
Integration tests for the six-layer LLM-FMEA pipeline.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from services.pipeline import LLMFMEAPipeline, PipelineResult
from models.fmea_scorer import ScenarioInput


@pytest.fixture(scope="module")
def pipeline():
    return LLMFMEAPipeline(mock_llm=True)


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


class TestPipelineOutput:
    def test_returns_pipeline_result(self, pipeline, critical_scenario):
        result = pipeline.assess(critical_scenario)
        assert isinstance(result, PipelineResult)

    def test_all_fields_populated(self, pipeline, critical_scenario):
        r = pipeline.assess(critical_scenario)
        assert r.risk_tier in ("Critical", "High", "Medium", "Low")
        assert 1 <= r.rpn_score <= 1000
        assert 0.0 <= r.stockout_probability <= 1.0
        assert 0.0 <= r.confidence_score <= 1.0
        assert len(r.explanation) > 10
        assert r.processing_time_ms > 0
        assert r.pipeline_version == "1.0.0"

    def test_critical_scenario_is_high_priority(self, pipeline, critical_scenario):
        r = pipeline.assess(critical_scenario)
        assert r.is_high_priority, f"Critical scenario returned tier={r.risk_tier}"

    def test_binary_predict_returns_0_or_1(self, pipeline, critical_scenario):
        pred = pipeline.predict_binary(critical_scenario)
        assert pred in (0, 1)

    def test_proba_in_unit_interval(self, pipeline, critical_scenario):
        proba = pipeline.predict_proba(critical_scenario)
        assert 0.0 <= proba <= 1.0

    def test_batch_returns_correct_count(self, pipeline):
        scenarios = [
            ScenarioInput(float(d), 0.30, 0.10, 0.70, "Medium", False)
            for d in [0, 1, 3, 5, 10, 15]
        ]
        results = pipeline.assess_batch(scenarios, verbose=False)
        assert len(results) == len(scenarios)

    def test_llm_detection_adjustment_applied(self, pipeline):
        """Fraud scenario should trigger LLM detection boost."""
        sc_fraud  = ScenarioInput(8.0, 0.30, 0.08, 0.60, "High", True)
        sc_nonfr  = ScenarioInput(8.0, 0.30, 0.08, 0.60, "High", False)
        r_fraud   = pipeline.assess(sc_fraud)
        r_nonfr   = pipeline.assess(sc_nonfr)
        assert r_fraud.detection_adjustment >= r_nonfr.detection_adjustment, (
            "Fraud scenario should receive ≥ detection adjustment vs non-fraud")

    def test_explanation_references_risk_tier(self, pipeline, critical_scenario):
        r = pipeline.assess(critical_scenario)
        tier_word = r.risk_tier.upper()
        assert tier_word in r.explanation.upper(), (
            f"Explanation should reference tier '{tier_word}'")
