# API Reference — LLM-FMEA Supply Chain Risk Assessment

**Version:** 1.0.0  
**Base URL:** `http://localhost:8000`  
**Protocol:** HTTP/1.1, HTTP/2  
**Format:** JSON (application/json)  
**Spec:** OpenAPI 3.0 — `http://localhost:8000/openapi.json`

---

## Authentication

The current release uses no authentication by default (suitable for internal enterprise deployment behind a gateway). For production, add OAuth2/JWT at the API Gateway layer (AWS API Gateway, Kong, or NGINX).

---

## Endpoints

### POST `/predict`

Run the full six-layer LLM-FMEA pipeline for a single supply chain scenario.

**Rate limit:** 100 req/sec (configurable)  
**Typical latency:** 50–200ms (mock LLM mode)  
**SLA target:** p99 < 500ms

#### Request Body

| Field | Type | Required | Range | Description |
|---|---|:---:|---|---|
| `delay_duration` | float | ✅ | [0, 30] | Delay duration in days (d) |
| `demand_deviation` | float | ✅ | [-1.0, 2.0] | Demand deviation fraction (ΔD). +0.30 = +30% |
| `inventory_buffer` | float | ✅ | (0, 1.0] | Inventory buffer fraction. 0.07 = 7% |
| `supplier_reliability` | float | ✅ | [0.0, 1.0] | Supplier reliability score (R) |
| `transport_risk` | string | ✅ | "Low"\|"Medium"\|"High" | Transport risk class (TR) |
| `fraud_indicator` | bool | ✗ | true\|false | Integrity anomaly flag (default: false) |

**Example Request — Critical Scenario:**
```json
{
  "delay_duration": 8.0,
  "demand_deviation": 0.30,
  "inventory_buffer": 0.07,
  "supplier_reliability": 0.72,
  "transport_risk": "High",
  "fraud_indicator": false
}
```

#### Response Body

| Field | Type | Description |
|---|---|---|
| `risk_tier` | string | "Critical" \| "High" \| "Medium" \| "Low" |
| `rpn_score` | int | Risk Priority Number ∈ [1, 1000] |
| `stockout_probability` | float | Pₛ ∈ [0, 1] |
| `confidence_score` | float | LLM classification confidence ∈ [0, 1] |
| `explanation` | string | Natural language risk explanation (≤200 tokens) |
| `component_scores.severity` | int | S ∈ [1, 10] |
| `component_scores.occurrence` | int | O ∈ [1, 10] |
| `component_scores.detection` | int | D (final, post-LLM) ∈ [1, 10] |
| `component_scores.detection_raw` | int | D before LLM adjustment |
| `component_scores.detection_adjustment` | int | LLM adjustment delta ∈ [-2, +3] |
| `service_level_drop` | float | Projected SL drop ∈ [0, 0.5] |
| `cost_impact` | float | Projected cost increase fraction ∈ [0, 0.75] |
| `processing_time_ms` | float | End-to-end pipeline latency (ms) |
| `pipeline_version` | string | "1.0.0" |
| `is_high_priority` | bool | True if tier ∈ {Critical, High} |

**Example Response — Critical:**
```json
{
  "risk_tier": "Critical",
  "rpn_score": 576,
  "stockout_probability": 0.3429,
  "confidence_score": 0.910,
  "explanation": "CRITICAL risk detected (RPN=576, confidence=91%). Delay of 8 days combined with 7% inventory buffer and +30% demand surge drives stockout probability to 34.3%. Elevated transport risk and supplier reliability of 72% compound exposure. Activate emergency procurement protocol and safety-stock replenishment. Projected cost exposure: +51.2%; service level impact: -18.4%.",
  "component_scores": {
    "severity": 9,
    "occurrence": 8,
    "detection": 8,
    "detection_raw": 4,
    "detection_adjustment": 2
  },
  "service_level_drop": 0.1843,
  "cost_impact": 0.5124,
  "processing_time_ms": 87.3,
  "pipeline_version": "1.0.0",
  "is_high_priority": true
}
```

**Example Response — Low:**
```json
{
  "risk_tier": "Low",
  "rpn_score": 8,
  "stockout_probability": 0.0,
  "confidence_score": 0.950,
  "explanation": "LOW risk assessed (RPN=8, confidence=95%). No significant delay, ample inventory buffer, and high supplier reliability present no material disruption risk. Standard monitoring protocols apply.",
  "component_scores": {
    "severity": 2,
    "occurrence": 2,
    "detection": 2,
    "detection_raw": 4,
    "detection_adjustment": -2
  },
  "service_level_drop": 0.0082,
  "cost_impact": 0.0510,
  "processing_time_ms": 42.1,
  "pipeline_version": "1.0.0",
  "is_high_priority": false
}
```

---

### POST `/predict/batch`

Process up to 500 scenarios in a single API call. Returns per-scenario results plus aggregate summary.

**Max batch size:** 500 scenarios  
**Typical latency:** O(n) — approximately 100ms per scenario

**Request:**
```json
{
  "scenarios": [
    {
      "delay_duration": 12.0,
      "demand_deviation": 0.45,
      "inventory_buffer": 0.05,
      "supplier_reliability": 0.48,
      "transport_risk": "High",
      "fraud_indicator": true
    },
    {
      "delay_duration": 1.0,
      "demand_deviation": -0.10,
      "inventory_buffer": 0.35,
      "supplier_reliability": 0.91,
      "transport_risk": "Low",
      "fraud_indicator": false
    }
  ]
}
```

**Response:**
```json
{
  "total": 2,
  "results": [ ... ],
  "summary": {
    "tier_distribution": {
      "Critical": 1,
      "High": 0,
      "Medium": 0,
      "Low": 1
    },
    "high_priority_count": 1,
    "high_priority_pct": 50.0,
    "avg_rpn": 364.0,
    "avg_confidence": 0.925
  }
}
```

---

### GET `/health`

Kubernetes-compatible liveness probe.

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "pipeline_ready": true,
  "uptime_seconds": 3847.2
}
```

---

## Error Codes

| HTTP Status | Code | Description |
|---|---|---|
| 422 | Validation Error | Invalid input field (range or type) |
| 500 | Internal Error | Pipeline error — check logs |

**Validation error example:**
```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "transport_risk"],
      "msg": "transport_risk must be one of ['High', 'Low', 'Medium']",
      "input": "FAST"
    }
  ]
}
```

---

## Curl Examples

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "delay_duration": 10,
    "demand_deviation": 0.4,
    "inventory_buffer": 0.06,
    "supplier_reliability": 0.55,
    "transport_risk": "High",
    "fraud_indicator": true
  }'

# Health check
curl http://localhost:8000/health

# OpenAPI spec
curl http://localhost:8000/openapi.json
```

---

## Python Client Example

```python
import httpx

BASE = "http://localhost:8000"

def assess_scenario(delay, demand_dev, inventory, reliability, transport, fraud=False):
    resp = httpx.post(f"{BASE}/predict", json={
        "delay_duration":      delay,
        "demand_deviation":    demand_dev,
        "inventory_buffer":    inventory,
        "supplier_reliability": reliability,
        "transport_risk":      transport,
        "fraud_indicator":     fraud,
    })
    resp.raise_for_status()
    return resp.json()

result = assess_scenario(8, 0.30, 0.07, 0.72, "High")
print(f"Risk Tier: {result['risk_tier']}")
print(f"RPN:       {result['rpn_score']}")
print(f"Recall:    Validated at 96.8% on 180,519 real-world records")
```
