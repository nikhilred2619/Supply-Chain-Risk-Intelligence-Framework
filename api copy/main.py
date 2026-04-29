"""
api/main.py
────────────
FastAPI REST API for the LLM-FMEA Supply Chain Risk Assessment framework.

Layer 6 — Enterprise Output Interface
Exposes OpenAPI 3.0-compatible endpoints for ERP/CRM integration:
  POST /predict       — Single scenario risk assessment
  POST /predict/batch — Batch scenario processing
  GET  /health        — Health check
  GET  /docs          — Auto-generated Swagger UI

Paper: Section III, Layer 6 (REST API, OpenAPI 3.0, SAP/Oracle/Salesforce)

Run:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import uvicorn

from models.fmea_scorer import ScenarioInput
from services.pipeline import LLMFMEAPipeline
from utils.config import API_CFG
from utils.logger import get_logger

log = get_logger("api")

# ── FastAPI application ───────────────────────────────────────────────────────

app = FastAPI(
    title=API_CFG.title,
    version=API_CFG.version,
    description=API_CFG.description,
    contact={
        "name":  "Nikhil Reddy Donapati",
        "email": "nikhil.donapati@myemail.indwes.edu",
        "url":   "https://orcid.org/0009-0006-7699-3928",
    },
    license_info={"name": "MIT"},
    openapi_tags=[
        {"name": "Risk Assessment", "description": "Real-time supply chain risk scoring"},
        {"name": "Health",          "description": "Service health monitoring"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Pipeline singleton ────────────────────────────────────────────────────────
_pipeline: Optional[LLMFMEAPipeline] = None


def get_pipeline() -> LLMFMEAPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = LLMFMEAPipeline(mock_llm=True)
    return _pipeline


# ══════════════════════════════════════════════════════════════════════════════
# Request / Response schemas (Pydantic v2)
# ══════════════════════════════════════════════════════════════════════════════

class RiskRequest(BaseModel):
    """
    Single-scenario risk assessment request.

    All fields correspond to the paper's six input variables (Section IV-A).
    """
    delay_duration: float = Field(
        ..., ge=0, le=30,
        description="Delay duration in days (d). Paper domain: {0,1,3,5,10,15}",
        example=8.0,
    )
    demand_deviation: float = Field(
        ..., ge=-1.0, le=2.0,
        description="Demand deviation as fraction (ΔD). E.g. 0.30 = +30%. Paper domain: {-0.30,-0.10,0.10,0.30,0.60}",
        example=0.30,
    )
    inventory_buffer: float = Field(
        ..., gt=0, le=1.0,
        description="Inventory buffer as fraction (I). E.g. 0.07 = 7%. Paper domain: {0.05,0.10,0.20,0.40}",
        example=0.07,
    )
    supplier_reliability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Supplier reliability score (R). Paper domain: [0.40,0.95]",
        example=0.72,
    )
    transport_risk: str = Field(
        ...,
        description='Transport risk class (TR): "Low", "Medium", or "High"',
        example="High",
    )
    fraud_indicator: bool = Field(
        False,
        description="Binary fraud / integrity anomaly flag",
        example=False,
    )

    @field_validator("transport_risk")
    @classmethod
    def validate_transport_risk(cls, v: str) -> str:
        allowed = {"Low", "Medium", "High"}
        if v not in allowed:
            raise ValueError(f"transport_risk must be one of {sorted(allowed)}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "delay_duration": 8.0,
                "demand_deviation": 0.30,
                "inventory_buffer": 0.07,
                "supplier_reliability": 0.72,
                "transport_risk": "High",
                "fraud_indicator": False,
            }
        }


class ComponentScores(BaseModel):
    severity:   int
    occurrence: int
    detection:  int
    detection_raw:        int
    detection_adjustment: int


class RiskResponse(BaseModel):
    """
    Risk assessment result from the six-layer pipeline.
    Compatible with SAP BTP OData, Oracle IC, and Salesforce Platform Events.
    """
    risk_tier:            str   = Field(..., description="Critical / High / Medium / Low")
    rpn_score:            int   = Field(..., description="Risk Priority Number (S×O×D)")
    stockout_probability: float = Field(..., description="Stockout Probability Pₛ ∈ [0,1]")
    confidence_score:     float = Field(..., description="LLM classification confidence ∈ [0,1]")
    explanation:          str   = Field(..., description="Natural language risk explanation")
    component_scores:     ComponentScores
    service_level_drop:   float = Field(..., description="Projected service level drop ∈ [0,0.5]")
    cost_impact:          float = Field(..., description="Projected cost impact fraction ∈ [0,0.75]")
    processing_time_ms:   float
    pipeline_version:     str
    is_high_priority:     bool

    class Config:
        json_schema_extra = {
            "example": {
                "risk_tier": "Critical",
                "rpn_score": 576,
                "stockout_probability": 0.343,
                "confidence_score": 0.91,
                "explanation": "CRITICAL risk detected (RPN=576, confidence=91%). Delay of 8 days combined with 7% inventory buffer and +30% demand surge drives stockout probability to 34.3%...",
                "component_scores": {"severity": 9, "occurrence": 8, "detection": 8,
                                     "detection_raw": 6, "detection_adjustment": 2},
                "service_level_drop": 0.228,
                "cost_impact": 0.512,
                "processing_time_ms": 87.3,
                "pipeline_version": "1.0.0",
                "is_high_priority": True,
            }
        }


class BatchRiskRequest(BaseModel):
    scenarios: List[RiskRequest] = Field(..., min_length=1, max_length=500)


class BatchRiskResponse(BaseModel):
    total:      int
    results:    List[RiskResponse]
    summary: dict


class HealthResponse(BaseModel):
    status:           str
    version:          str
    pipeline_ready:   bool
    uptime_seconds:   float


# ══════════════════════════════════════════════════════════════════════════════
# Startup / Shutdown
# ══════════════════════════════════════════════════════════════════════════════

_start_time = time.time()


@app.on_event("startup")
async def startup_event():
    log.info("LLM-FMEA API starting up ...")
    get_pipeline()   # warm up
    log.info("Pipeline ready. Serving at :%d", API_CFG.port)


# ══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint — suitable for Kubernetes liveness probes."""
    return HealthResponse(
        status="healthy",
        version=API_CFG.version,
        pipeline_ready=_pipeline is not None,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post("/predict", response_model=RiskResponse, tags=["Risk Assessment"])
async def predict(request: RiskRequest):
    """
    Real-time supply chain risk assessment.

    Runs the complete six-layer LLM-FMEA pipeline and returns:
    - Risk tier classification (Critical / High / Medium / Low)
    - FMEA Risk Priority Number (RPN)
    - Stockout Probability (Pₛ)
    - LLM confidence score
    - Natural language explanation
    - Component FMEA scores (S, O, D)
    - Projected operational impact (service level drop, cost impact)

    **Enterprise integration**: This endpoint is designed for consumption by
    SAP Integration Suite, Oracle Integration Cloud, and Salesforce Platform Events.
    """
    pipeline = get_pipeline()

    try:
        scenario = ScenarioInput(
            delay_duration=request.delay_duration,
            demand_deviation=request.demand_deviation,
            inventory_buffer=request.inventory_buffer,
            supplier_reliability=request.supplier_reliability,
            transport_risk=request.transport_risk,
            fraud_indicator=request.fraud_indicator,
        )
        result = pipeline.assess(scenario)
    except (ValueError, AssertionError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        log.error("Pipeline error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal pipeline error")

    return RiskResponse(
        risk_tier=result.risk_tier,
        rpn_score=result.rpn_score,
        stockout_probability=round(result.stockout_probability, 4),
        confidence_score=round(result.confidence_score, 3),
        explanation=result.explanation,
        component_scores=ComponentScores(
            severity=result.severity,
            occurrence=result.occurrence,
            detection=result.detection,
            detection_raw=result.detection_raw,
            detection_adjustment=result.detection_adjustment,
        ),
        service_level_drop=round(result.service_level_drop, 4),
        cost_impact=round(result.cost_impact, 4),
        processing_time_ms=result.processing_time_ms,
        pipeline_version=result.pipeline_version,
        is_high_priority=result.is_high_priority,
    )


@app.post("/predict/batch", response_model=BatchRiskResponse, tags=["Risk Assessment"])
async def predict_batch(request: BatchRiskRequest):
    """
    Batch risk assessment for multiple scenarios.

    Processes up to 500 scenarios in a single request.
    Returns individual results plus an aggregate summary.
    """
    pipeline = get_pipeline()
    results  = []
    tier_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}

    for req in request.scenarios:
        scenario = ScenarioInput(
            delay_duration=req.delay_duration,
            demand_deviation=req.demand_deviation,
            inventory_buffer=req.inventory_buffer,
            supplier_reliability=req.supplier_reliability,
            transport_risk=req.transport_risk,
            fraud_indicator=req.fraud_indicator,
        )
        result = pipeline.assess(scenario)
        tier_counts[result.risk_tier] = tier_counts.get(result.risk_tier, 0) + 1
        results.append(RiskResponse(
            risk_tier=result.risk_tier,
            rpn_score=result.rpn_score,
            stockout_probability=round(result.stockout_probability, 4),
            confidence_score=round(result.confidence_score, 3),
            explanation=result.explanation,
            component_scores=ComponentScores(
                severity=result.severity,
                occurrence=result.occurrence,
                detection=result.detection,
                detection_raw=result.detection_raw,
                detection_adjustment=result.detection_adjustment,
            ),
            service_level_drop=round(result.service_level_drop, 4),
            cost_impact=round(result.cost_impact, 4),
            processing_time_ms=result.processing_time_ms,
            pipeline_version=result.pipeline_version,
            is_high_priority=result.is_high_priority,
        ))

    high_priority_count = tier_counts.get("Critical", 0) + tier_counts.get("High", 0)
    return BatchRiskResponse(
        total=len(results),
        results=results,
        summary={
            "tier_distribution": tier_counts,
            "high_priority_count": high_priority_count,
            "high_priority_pct": round(high_priority_count / len(results) * 100, 1),
            "avg_rpn": round(sum(r.rpn_score for r in results) / len(results), 1),
            "avg_confidence": round(sum(r.confidence_score for r in results) / len(results), 3),
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=API_CFG.host,
        port=API_CFG.port,
        reload=False,
        log_level="info",
    )
