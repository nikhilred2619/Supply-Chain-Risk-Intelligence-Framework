"""
utils/config.py
───────────────
Central configuration for the LLM-FMEA Supply Chain Risk Assessment framework.
All thresholds, model parameters, and environment settings are defined here
to ensure reproducibility and paper-alignment.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "results" / "figures"

for _d in [RESULTS_DIR, FIGURES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ── FMEA Parameters (paper-aligned) ──────────────────────────────────────────
@dataclass(frozen=True)
class FMEAConfig:
    """FMEA scoring constants aligned with paper Section IV-B."""
    score_min: int = 1
    score_max: int = 10

    # RPN tier thresholds
    rpn_critical:  int = 300   # ≥ 300 → Critical
    rpn_high:      int = 150   # 150–299 → High
    rpn_medium:    int = 60    # 60–149 → Medium
    # < 60 → Low

    # Detection score boost for fraud scenarios (paper: mean D=7.3 vs 4.2)
    fraud_detection_mean:     float = 7.3
    nonfraud_detection_mean:  float = 4.2

    # Stockout probability thresholds
    ps_critical_threshold:  float = 0.50   # Ps ≥ 50% → Critical signal
    ps_high_threshold:      float = 0.25

    # Non-linear compounding threshold (paper Section VI-C)
    delay_critical_days:    float = 5.0    # d > 5 days
    inventory_critical_pct: float = 0.10   # I < 10%


# ── Simulation Parameters ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class SimulationConfig:
    """120-scenario parameterized benchmark configuration (paper Section IV-A)."""
    n_scenarios: int = 120
    random_seed: int = 42
    test_split:  float = 0.20   # 80/20 stratified split

    # Variable domains
    delay_values:   tuple = (0, 1, 3, 5, 10, 15)          # days
    demand_deltas:  tuple = (-0.30, -0.10, 0.10, 0.30, 0.60)  # demand deviation
    inventory_lvls: tuple = (0.05, 0.10, 0.20, 0.40)      # buffer fraction
    reliability_range: tuple = (0.40, 0.95)                # supplier reliability
    transport_risks: tuple = ("Low", "Medium", "High")

    # Risk tier distribution (paper Table I)
    tier_distribution: dict = field(default_factory=lambda: {
        "Critical": 0.20,
        "High":     0.30,
        "Medium":   0.35,
        "Low":      0.15,
    })


# ── LLM Configuration ────────────────────────────────────────────────────────
@dataclass
class LLMConfig:
    """LLM component settings (paper Section IV-E)."""
    model:          str   = "gpt-4"
    temperature:    float = 0.2
    top_p:          float = 0.9
    max_tokens:     int   = 512
    system_prompt_version: str = "v1.0"
    mock_mode:      bool  = True    # True → deterministic mock (no API key needed)

    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))


# ── Model Evaluation ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class EvalConfig:
    """Evaluation and cross-validation settings."""
    cv_folds:        int   = 5
    random_seed:     int   = 42
    primary_metric:  str   = "recall"    # Recall-optimized (Proposition 1)
    significance_level: float = 0.05

    # Decision threshold for binary classification
    # Calibrated for recall maximization per Proposition 1
    decision_threshold: float = 0.35


# ── API Configuration ────────────────────────────────────────────────────────
@dataclass
class APIConfig:
    host:    str = "0.0.0.0"
    port:    int = 8000
    title:   str = "LLM-FMEA Supply Chain Risk Assessment API"
    version: str = "1.0.0"
    description: str = (
        "Enterprise REST API for real-time supply chain risk assessment "
        "using the LLM-augmented FMEA decision intelligence framework."
    )


# ── Singleton instances ───────────────────────────────────────────────────────
FMEA_CFG       = FMEAConfig()
SIM_CFG        = SimulationConfig()
LLM_CFG        = LLMConfig()
EVAL_CFG       = EvalConfig()
API_CFG        = APIConfig()

# Transport risk numerical encoding
TRANSPORT_RISK_MAP = {"Low": 1, "Medium": 2, "High": 3}

# Risk tier ordering (for ordinal comparisons)
RISK_TIER_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
