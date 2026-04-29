"""
models/llm_reasoner.py
──────────────────────
LLM Contextual Reasoning Layer (Layer 4 of the six-layer pipeline).

In production mode, this module calls a GPT-4-class enterprise model with:
  - temperature = 0.2
  - top_p = 0.9
  - max_tokens = 512
  - Fixed system prompt v1.0

In mock mode (default; no API key required), a deterministic rule-based
simulation produces outputs statistically consistent with the paper's
reported LLM behaviour — including fraud detection boost (+68% FN reduction)
and borderline scenario confidence stabilisation.

Paper Reference: Section IV-E (LLM Configuration)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import json
import re

from models.fmea_scorer import ScenarioInput, FMEAScores
from utils.config import LLM_CFG, TRANSPORT_RISK_MAP
from utils.logger import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT_V1 = """You are a supply chain risk analyst embedded in an enterprise decision intelligence system.

Your task: analyse the provided scenario variables and preliminary FMEA scores, then produce:
1. A revised Detection score adjustment (integer in [-2, +3]) if the preliminary score underestimates observability difficulty
2. A confidence score for the FMEA risk tier [0.0, 1.0]
3. A natural language risk explanation (≤200 tokens) suitable for operations managers

Risk tier logic:
- Critical: RPN ≥ 300 OR (Ps ≥ 50% AND RPN ≥ 150)
- High: RPN 150–299
- Medium: RPN 60–149
- Low: RPN < 60

Respond ONLY with valid JSON:
{
  "detection_adjustment": <int>,
  "confidence": <float>,
  "explanation": "<string>"
}"""


@dataclass
class LLMOutput:
    """Structured output from the LLM reasoning layer."""
    detection_adjustment:  int    # Adjustment to FMEA Detection score [-2, +3]
    confidence:            float  # Confidence in risk tier classification [0,1]
    explanation:           str    # Natural language explanation (≤200 tokens)
    model_used:            str    # Model identifier
    mock_mode:             bool   # Whether mock simulation was used


class LLMReasoner:
    """
    LLM Contextual Reasoning Layer.

    Wraps the LLM call with structured prompt engineering, output parsing,
    and fallback to deterministic mock simulation for reproducible research.
    """

    def __init__(self, mock_mode: Optional[bool] = None):
        self.cfg = LLM_CFG
        self.mock_mode = mock_mode if mock_mode is not None else self.cfg.mock_mode
        if not self.mock_mode:
            self._init_openai_client()
        log.info("LLMReasoner initialised [mode=%s, model=%s]",
                 "mock" if self.mock_mode else "live", self.cfg.model)

    def _init_openai_client(self) -> None:
        try:
            import openai
            self._client = openai.OpenAI(api_key=self.cfg.api_key)
        except ImportError:
            log.warning("openai package not installed — falling back to mock mode")
            self.mock_mode = True

    # ──────────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────────

    def reason(self, scenario: ScenarioInput, fmea: FMEAScores) -> LLMOutput:
        """
        Run LLM contextual reasoning over a scenario + FMEA scores.

        Returns structured LLMOutput including detection adjustment,
        confidence score, and natural language explanation.
        """
        if self.mock_mode:
            return self._mock_reason(scenario, fmea)
        return self._live_reason(scenario, fmea)

    # ──────────────────────────────────────────────────────────────────────────
    # Live LLM call
    # ──────────────────────────────────────────────────────────────────────────

    def _live_reason(self, scenario: ScenarioInput, fmea: FMEAScores) -> LLMOutput:
        payload = self._build_payload(scenario, fmea)
        try:
            response = self._client.chat.completions.create(
                model=self.cfg.model,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                max_tokens=self.cfg.max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_V1},
                    {"role": "user",   "content": json.dumps(payload)},
                ],
            )
            raw = response.choices[0].message.content
            return self._parse_response(raw, live=True)
        except Exception as exc:
            log.error("LLM call failed: %s — falling back to mock", exc)
            return self._mock_reason(scenario, fmea)

    # ──────────────────────────────────────────────────────────────────────────
    # Mock / deterministic simulation
    # Statistically calibrated to paper-reported LLM behaviour
    # ──────────────────────────────────────────────────────────────────────────

    def _mock_reason(self, scenario: ScenarioInput, fmea: FMEAScores) -> LLMOutput:
        """
        Deterministic LLM simulation.

        Calibration targets (from paper ablation study):
        - Fraud scenarios: detection boost reduces FN by ~68% vs FMEA-only
        - Borderline RPN [140-160]: confidence stabilisation
        - High demand + low inventory: additional severity signal
        """
        adj        = self._compute_detection_adjustment(scenario, fmea)
        confidence = self._compute_confidence(scenario, fmea, adj)
        explanation = self._generate_explanation(scenario, fmea, confidence)

        return LLMOutput(
            detection_adjustment=adj,
            confidence=confidence,
            explanation=explanation,
            model_used="mock-gpt4-v1.0",
            mock_mode=True,
        )

    def _compute_detection_adjustment(self, scenario: ScenarioInput, fmea: FMEAScores) -> int:
        """
        Compute Detection score adjustment based on contextual signals.

        Fraud scenarios receive positive adjustment (harder to detect early)
        consistent with paper: fraud_detection_mean = 7.3 vs 4.2.
        """
        adj = 0
        if scenario.fraud_indicator:
            # Integrity failures are systematically underestimated by FMEA alone
            adj += 2
            if scenario.supplier_reliability < 0.60:
                adj += 1
        if scenario.transport_risk == "High" and scenario.delay_duration > 5:
            adj += 1
        if scenario.inventory_buffer < 0.05 and fmea.stockout_probability > 0.80:
            # Extreme stockout → signal is obvious; reduce detection difficulty
            adj -= 1
        return max(-2, min(3, adj))

    def _compute_confidence(self, scenario: ScenarioInput, fmea: FMEAScores, adj: int) -> float:
        """
        Confidence in risk tier classification.

        Borderline cases [140–160 RPN] receive lower confidence;
        extreme cases receive high confidence (>0.90).
        """
        rpn = fmea.rpn
        # Clear cases
        if rpn >= 400 or rpn < 40:
            base = 0.95
        elif rpn >= 300 or rpn < 60:
            base = 0.88
        elif 140 <= rpn <= 160:
            # Borderline — paper error analysis Section VII-D
            base = 0.68
        else:
            base = 0.80

        # Adjustment factors
        if fmea.stockout_probability >= 0.80:
            base = min(1.0, base + 0.05)
        if scenario.fraud_indicator and adj > 0:
            base = min(1.0, base + 0.04)

        return round(float(base), 3)

    def _generate_explanation(self, scenario: ScenarioInput,
                               fmea: FMEAScores, confidence: float) -> str:
        """Generate structured natural language risk explanation (≤200 tokens)."""
        tier = fmea.risk_tier
        ps_pct = fmea.stockout_probability * 100
        ci_pct = fmea.cost_impact * 100
        sl_pct = fmea.service_level_drop * 100

        # Tier-specific lead sentence
        if tier == "Critical":
            lead = (f"CRITICAL risk detected (RPN={fmea.rpn}, confidence={confidence:.0%}). "
                    f"Immediate operational intervention required.")
        elif tier == "High":
            lead = (f"HIGH risk identified (RPN={fmea.rpn}, confidence={confidence:.0%}). "
                    f"Escalation to procurement leadership recommended within 24 hours.")
        elif tier == "Medium":
            lead = (f"MEDIUM risk flagged (RPN={fmea.rpn}, confidence={confidence:.0%}). "
                    f"Enhanced monitoring and contingency review advised.")
        else:
            lead = (f"LOW risk assessed (RPN={fmea.rpn}, confidence={confidence:.0%}). "
                    f"Standard monitoring protocols apply.")

        # Contributing factors
        factors = []
        if scenario.delay_duration > 5:
            factors.append(f"delay of {scenario.delay_duration:.0f} days exceeds critical threshold")
        if scenario.inventory_buffer < 0.10:
            factors.append(f"low inventory buffer ({scenario.inventory_buffer*100:.0f}%) amplifies disruption exposure")
        if scenario.demand_deviation > 0.20:
            factors.append(f"demand surge of +{scenario.demand_deviation*100:.0f}% strains replenishment capacity")
        if scenario.supplier_reliability < 0.70:
            factors.append(f"supplier reliability of {scenario.supplier_reliability:.0%} elevates occurrence risk")
        if scenario.fraud_indicator:
            factors.append("fraud indicator active — integrity failure mode requires immediate audit")
        if fmea.stockout_probability > 0.30:
            factors.append(f"stockout probability {ps_pct:.1f}% exceeds operational threshold")

        factor_text = "; ".join(factors[:4]) + "." if factors else "Risk driven by compound parameter interactions."

        recommendation = ""
        if tier == "Critical":
            recommendation = " Activate emergency procurement protocol and safety-stock replenishment."
        elif tier == "High":
            recommendation = " Initiate supplier review and demand hedging strategy."

        return f"{lead} Contributing factors: {factor_text}{recommendation} Projected cost exposure: +{ci_pct:.1f}%; service level impact: -{sl_pct:.1f}%."

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_payload(self, scenario: ScenarioInput, fmea: FMEAScores) -> dict:
        return {
            "scenario": {
                "delay_duration_days":    scenario.delay_duration,
                "demand_deviation_pct":   round(scenario.demand_deviation * 100, 1),
                "inventory_buffer_pct":   round(scenario.inventory_buffer * 100, 1),
                "supplier_reliability":   scenario.supplier_reliability,
                "transport_risk":         scenario.transport_risk,
                "fraud_indicator":        scenario.fraud_indicator,
            },
            "fmea_scores": {
                "severity":    fmea.severity,
                "occurrence":  fmea.occurrence,
                "detection":   fmea.detection,
                "rpn":         fmea.rpn,
                "stockout_probability_pct": round(fmea.stockout_probability * 100, 1),
                "preliminary_risk_tier": fmea.risk_tier,
            },
        }

    def _parse_response(self, raw: str, live: bool = True) -> LLMOutput:
        """Parse and validate LLM JSON response."""
        try:
            clean = re.sub(r"```json|```", "", raw).strip()
            data = json.loads(clean)
            return LLMOutput(
                detection_adjustment=int(data.get("detection_adjustment", 0)),
                confidence=float(data.get("confidence", 0.70)),
                explanation=str(data.get("explanation", "")),
                model_used=self.cfg.model if live else "mock",
                mock_mode=not live,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            log.warning("LLM response parse error: %s — using defaults", exc)
            return LLMOutput(
                detection_adjustment=0,
                confidence=0.65,
                explanation="Risk assessment completed. Manual review recommended for borderline cases.",
                model_used="parse-error-fallback",
                mock_mode=True,
            )
