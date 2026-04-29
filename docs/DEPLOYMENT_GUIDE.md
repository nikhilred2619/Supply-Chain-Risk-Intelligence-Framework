# Enterprise Deployment Guide

**LLM-FMEA Supply Chain Risk Assessment Framework v1.0.0**

---

## Overview

This guide covers production deployment of the LLM-FMEA framework across three enterprise integration patterns: standalone REST API, cloud-native containerized deployment, and native ERP/CRM platform integration.

---

## Deployment Architectures

### Architecture A — Standalone REST API

```
Supply Chain Events
        │
        ▼
┌───────────────────┐
│  LLM-FMEA API     │  ← uvicorn / gunicorn
│  Port 8000        │
│  Docker Container │
└───────────────────┘
        │
        ├── Response: risk_tier, rpn, explanation
        └── Webhook: Platform Events (if Critical/High)
```

**Best for:** Pilot deployment, proof-of-value, single-team adoption

```bash
# Launch
docker compose up -d
curl http://localhost:8000/health
```

---

### Architecture B — Cloud-Native (AWS / Azure / GCP)

```
ERP / WMS / TMS
      │
      ▼
API Gateway (AWS API GW / Azure APIM)
      │  auth + rate limiting
      ▼
Load Balancer
      │
   ┌──┴──┐
   │     │     ← 2+ ECS/AKS replicas
   ▼     ▼
 SCRA  SCRA
 Pod   Pod
   │     │
   └──┬──┘
      │
   Results DB (RDS / CosmosDB)
      │
   BI Dashboard (QuickSight / Power BI)
```

**Best for:** Enterprise-wide deployment, high throughput (>1,000 assessments/hour)

```yaml
# Kubernetes deployment excerpt
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-fmea-scra
spec:
  replicas: 3
  template:
    spec:
      containers:
        - name: scra-api
          image: nikhildonapati/llm-fmea-scra:1.0.0
          ports: [{containerPort: 8000}]
          env:
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: scra-secrets
                  key: openai-api-key
          resources:
            requests: {cpu: "500m", memory: "512Mi"}
            limits:   {cpu: "2000m", memory: "2Gi"}
          livenessProbe:
            httpGet: {path: /health, port: 8000}
            initialDelaySeconds: 10
            periodSeconds: 30
```

---

### Architecture C — SAP BTP Integration

```
SAP S/4HANA (Order Created Event)
        │
        ▼
SAP Integration Suite
        │  OData Adapter
        ▼
LLM-FMEA REST API (/predict)
        │
        ▼
Risk Assessment Result
        │
   ┌────┴─────────────────────────┐
   │                              │
   ▼                              ▼
SAP SCM Cloud               SAP Alert Notification
(auto-PO if Critical)       (MM Purchasing team)
```

**Integration steps:**

1. Create Integration Flow in SAP Integration Suite
2. Add REST Adapter pointing to `http://llm-fmea-api:8000/predict`
3. Map S/4HANA delivery fields to FMEA input schema
4. Configure OData write-back for `SupplyChainRiskSet`

```python
# SAP BTP connector usage
from services.integrations import SAPBTPConnector

connector = SAPBTPConnector(
    api_host="https://your-iflow.cfapps.eu10.hana.ondemand.com",
    client_id=os.getenv("SAP_CLIENT_ID"),
    client_secret=os.getenv("SAP_CLIENT_SECRET"),
)
connector.push_risk_assessment(
    result,
    material_number="MAT-00847231",
    plant_code="DE01"
)
```

---

### Architecture D — Salesforce + Agentforce

```
Salesforce Order Object (trigger on status change)
        │  Apex Trigger
        ▼
Salesforce Flow (REST Callout)
        │  POST /predict
        ▼
LLM-FMEA API
        │
        ▼
SupplyChainRiskAlert__e (Platform Event)
        │
   ┌────┴──────────────────────────────┐
   │                │                  │
   ▼                ▼                  ▼
Agentforce       Create Case        Einstein
Agent            (Critical only)    Dashboard
(Investigation)  + Notify Manager   (Risk KPIs)
```

**Salesforce setup:**

```apex
// Apex Trigger — fires on Order status change
trigger OrderRiskAssessment on Order (after update) {
    for (Order o : Trigger.new) {
        if (o.Status == 'Delayed') {
            SCRACallout.assess(o.Id,
                o.Delay_Days__c,
                o.Demand_Deviation__c,
                o.Inventory_Buffer__c,
                o.Supplier_Reliability__c,
                o.Transport_Risk__c,
                o.Fraud_Flag__c);
        }
    }
}
```

```python
# Salesforce connector usage
from services.integrations import SalesforceConnector

connector = SalesforceConnector(
    instance_url="https://your-org.salesforce.com",
    access_token=os.getenv("SF_ACCESS_TOKEN"),
)
connector.publish_risk_event(result)
connector.create_case_if_critical(result, order_id="ORD-2025-00847")
```

---

### Architecture E — Oracle Integration Cloud

```
Oracle SCM Cloud (Shipment Delay Event)
        │
        ▼
Oracle Integration Cloud (OIC)
        │  REST Trigger
        ▼
LLM-FMEA API (/predict)
        │
        ▼
Risk Result
        │
   ┌────┴──────────────────────────┐
   │                               │
   ▼                               ▼
Oracle SCM (auto-PO)        Oracle HCM Notification
(Critical threshold)        (Supply chain planner)
```

---

## Staged Adoption Roadmap

### Phase 1 — FMEA-Only Mode (Week 1–4)
```
Deploy:   Layers 1–3 only
Cost:     Zero LLM API cost
Goal:     Establish baseline RPN scoring, operator trust
Metric:   RPN-Cost correlation validation (target r > 0.80)
```

### Phase 2 — Selective LLM (Week 5–8)
```
Deploy:   All layers, LLM enabled for RPN ≥ 150 only
Cost:     ~20% of orders require LLM call @ $0.01–0.05/call
Goal:     Validate detection improvement vs Phase 1
Metric:   Recall uplift target > +15pp vs FMEA-only baseline
```

### Phase 3 — Full Pipeline (Week 9+)
```
Deploy:   All six layers active, REST API → ERP/CRM
Cost:     Full LLM coverage, async batch for throughput
Goal:     Production alerting integrated with operations workflow
Metric:   Critical event false negative rate < 5%
```

---

## Environment Variables

```bash
# Required for live LLM mode
OPENAI_API_KEY=sk-...

# Optional Salesforce integration
SF_INSTANCE_URL=https://your-org.salesforce.com
SF_ACCESS_TOKEN=...

# Optional SAP BTP
SAP_API_HOST=https://your-iflow.cfapps.eu10.hana.ondemand.com
SAP_CLIENT_ID=...
SAP_CLIENT_SECRET=...

# Optional Oracle IC
ORACLE_OIC_HOST=https://your-instance.integration.ocp.oraclecloud.com
ORACLE_USERNAME=...
ORACLE_PASSWORD=...

# API configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=info
```

---

## Performance Benchmarks

| Deployment | Throughput | p50 Latency | p99 Latency |
|---|:---:|:---:|:---:|
| Local (mock LLM) | 500 req/s | 12ms | 45ms |
| Local (live GPT-4) | 8 req/s | 1.8s | 3.9s |
| Docker single | 200 req/s | 28ms | 95ms |
| K8s (3 replicas) | 600 req/s | 22ms | 78ms |
| Async batch mode | 1,200/hr | — | — |

---

## Security Checklist

- [ ] API Gateway authentication (OAuth2 / API key)
- [ ] TLS/HTTPS termination at load balancer
- [ ] Secret rotation for OPENAI_API_KEY
- [ ] Network policy: restrict pod-to-pod communication
- [ ] Audit logging enabled (all `/predict` calls logged)
- [ ] PII: no customer-identifiable data in LLM payloads
- [ ] Rate limiting: 100 req/s per client

---

## Monitoring

```yaml
# Prometheus metrics exposed at /metrics (add prometheus-fastapi-instrumentator)
scra_predictions_total{tier="Critical"} 247
scra_predictions_total{tier="High"}     891
scra_predictions_total{tier="Medium"}  2341
scra_predictions_total{tier="Low"}     5821
scra_pipeline_latency_seconds{quantile="0.99"} 0.089
scra_rpn_score_histogram_bucket{le="300"} 7102
```

**Alerting rules:**
- Critical event rate > 15% of hourly volume → PagerDuty
- p99 latency > 500ms → scale-out trigger
- Pipeline error rate > 0.1% → on-call alert
