<div align="center">

# 🔴 LLM-FMEA Supply Chain Risk Assessment

### *LLM-Augmented Decision Intelligence for Real-Time Supply Chain Risk*

**A Six-Layer Enterprise AI Framework Integrating FMEA Quantification with Large Language Model Contextual Reasoning**

---

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-27%20Passing-brightgreen?style=for-the-badge)](tests/)
[![Paper](https://img.shields.io/badge/Paper-Under%20Review-red?style=for-the-badge)](https://orcid.org/0009-0006-7699-3928)

---

**Author:** Nikhil Reddy Donapati · Agentforce AI Specialist & Senior Salesforce Developer · Texas, USA  
**ORCID:** [0009-0006-7699-3928](https://orcid.org/0009-0006-7699-3928) · **Paper:** *SN Computer Science / Cluster Computing (Under Review)*

</div>

---

## Table of Contents

- [Problem Statement](#-problem-statement)
- [Business Value](#-business-value)
- [What Makes This Original](#-what-makes-this-original)
- [Architecture](#-six-layer-architecture)
- [Live Demo](#-live-demo)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Enterprise Integration](#-enterprise-integration)
- [Experimental Results](#-experimental-results)
- [Project Structure](#-project-structure)
- [Deployment](#-deployment)
- [Research Contribution](#-research-contribution)
- [Citation](#-citation)

---

## 🔴 Problem Statement

Global supply chains lose an estimated **$182 billion annually** to disruption events. Traditional risk management systems fail on five fronts:

| Limitation | Operational Impact |
|---|---|
| **Static rule engines** | Cannot adapt to compound, non-linear disruptions |
| **Threshold-only alerts** | Miss 30%+ of high-risk events (catastrophic false negatives) |
| **No contextual reasoning** | Cannot explain *why* risk is elevated — unusable for operations |
| **Manual FMEA processes** | Days to score — too slow for real-time supply chain decisions |
| **Black-box ML predictions** | Not auditable for regulatory compliance or procurement review |

> *"A missed supply chain disruption costs 3×–8× more than a false alarm. Yet most enterprise systems are still optimized for accuracy, not recall."*

This framework solves all five simultaneously.

---

## 💼 Business Value

### The Numbers

```
BEFORE — Traditional Approaches
─────────────────────────────────────────────
Rule-Based System     → 70.4% Recall   → 1 in 3 disruptions missed
Decision Tree         → 58.3% Recall   → nearly half of crises undetected
Manual FMEA scoring   → 3–5 days       → operationally useless at real-time scale

AFTER — LLM-FMEA Framework
─────────────────────────────────────────────
Hybrid LLM-FMEA       → 96.8% Recall   → validated on 180,519 records
Processing latency     → < 100ms        → enterprise real-time grade
Explanation           → Auto-generated → actionable for operations teams
Integration           → REST API        → SAP / Oracle / Salesforce ready
```

### ROI Scenario (Mid-Size Retail Distributor)

```
Annual disruption exposure:    $12.4M
Missed risks @ 29.6% FN rate:  $3.7M annual loss
─────────────────────────────────────────────
After LLM-FMEA deployment:
  FN rate drops from 29.6% → 3.2%
  Estimated annual savings:   $3.2M
  API infrastructure cost:    ~$48K/year
  Net ROI:                    66×
```

---

## 🔬 What Makes This Original

This is not an academic toy. It is the **first enterprise-deployable framework** combining FMEA determinism with LLM reasoning — validated at scale.

### Three Contributions No Prior Work Has Combined

```
┌─────────────────────────────────────────────────────────────────────┐
│  CONTRIBUTION 1: Algorithmic Novelty                                │
│                                                                     │
│  Formal hybrid of FMEA deterministic RPN scoring + LLM contextual  │
│  reasoning — validated on 180,519 real records. No prior work       │
│  combines both within one unified enterprise pipeline.              │
├─────────────────────────────────────────────────────────────────────┤
│  CONTRIBUTION 2: Empirical Rigor                                    │
│                                                                     │
│  Dual-dataset: controlled 120-scenario benchmark + DataCo 180K.     │
│  McNemar's significance test. 5-fold CV. Ablation study isolating   │
│  each component's marginal contribution. AUC = 0.760.               │
├─────────────────────────────────────────────────────────────────────┤
│  CONTRIBUTION 3: Enterprise Architecture                            │
│                                                                     │
│  OpenAPI 3.0 REST API. SAP BTP, Oracle IC, Salesforce connectors.  │
│  Streamlit demo dashboard. Dockerfile. Staged deployment roadmap.   │
│  Not a proof-of-concept — a production-deployable system.           │
└─────────────────────────────────────────────────────────────────────┘
```

### Literature Comparison

| Capability | Zhang et al. | Wang et al. | Sharma et al. | **This Work** |
|---|:---:|:---:|:---:|:---:|
| LLM Integration | ✗ | Partial | ✗ | ✅ |
| FMEA / RPN Scoring | ✗ | ✗ | ✅ | ✅ |
| Real-World Dataset > 100K | ✅ | ✗ | ✗ | ✅ |
| Enterprise REST API | ✗ | ✗ | ✗ | ✅ |
| SAP / Oracle / Salesforce | ✗ | ✗ | ✗ | ✅ |
| McNemar Statistical Test | ✗ | ✗ | ✗ | ✅ |
| Fraud Detection Layer | ✗ | ✗ | ✗ | ✅ |
| Ablation Study | ✗ | ✗ | ✗ | ✅ |

---

## 🏗️ Six-Layer Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                 LLM-FMEA Decision Intelligence Pipeline              │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  L1  DATA INGESTION                                          │   │
│  │      ERP Records · CRM Signals · Logistics · Scenarios       │   │
│  └──────────────────────────┬─────────────────────────────────┘   │
│                              │ validated schema                      │
│  ┌──────────────────────────▼─────────────────────────────────┐   │
│  │  L2  FEATURE ENGINEERING                                    │   │
│  │      Min-Max Norm · Derived Signals · Schema Validation     │   │
│  └──────────────────────────┬─────────────────────────────────┘   │
│                              │ normalized 6-feature vector           │
│  ┌──────────────────────────▼─────────────────────────────────┐   │
│  │  L3  FMEA RISK QUANTIFICATION                   [AUDITABLE] │   │
│  │      RPN = S × O × D                                        │   │
│  │      Pₛ  = min(1, d·ΔD / I)                                │   │
│  └──────────────────────────┬─────────────────────────────────┘   │
│                              │ RPN + Pₛ scores                      │
│  ┌──────────────────────────▼─────────────────────────────────┐   │
│  │  L4  LLM CONTEXTUAL REASONING                [EXPLAINABLE]  │   │
│  │      GPT-4-class · Detection Boost · NL Explanation         │   │
│  └──────────────────────────┬─────────────────────────────────┘   │
│                              │ adjusted scores + explanation         │
│  ┌──────────────────────────▼─────────────────────────────────┐   │
│  │  L5  DECISION SYNTHESIS                                     │   │
│  │      Weighted Fusion: FMEA(65%) + LLM(35%)                 │   │
│  │      → Critical / High / Medium / Low                       │   │
│  └──────────────────────────┬─────────────────────────────────┘   │
│                              │ risk tier + full metadata             │
│  ┌──────────────────────────▼─────────────────────────────────┐   │
│  │  L6  ENTERPRISE OUTPUT              [REST API / OpenAPI 3.0] │   │
│  │      SAP BTP · Oracle IC · Salesforce Platform Events       │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

**Core Equations:**

```python
# Equation 1: Risk Priority Number
RPN = S × O × D      # S, O, D ∈ [1,10] → RPN ∈ [1,1000]

# Equation 2: Stockout Probability (Scarf 1958 newsvendor)
Pₛ = min(1.0, (d × ΔD) / I)

# Critical Zone (empirically identified):
# d > 5 days AND I < 10%  →  Pₛ → 1.0  (catastrophic threshold)
```

---

## 🖥️ Live Demo

```bash
streamlit run demo/dashboard.py
```

The interactive Streamlit dashboard provides real-time risk assessment with:
- Input panel: all six scenario variables with sliders
- Live RPN gauge chart with tier color coding
- Confidence score indicator
- Natural language explanation panel
- Scenario history table with trend visualization

---

## ⚡ Quick Start

### Python

```bash
git clone https://github.com/nikhildonapati/llm-fmea-scra.git
cd llm-fmea-scra
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Live demo dashboard
streamlit run demo/dashboard.py

# REST API
uvicorn api.main:app --reload --port 8000
# → http://localhost:8000/docs

# Reproduce all paper experiments
bash scripts/run_full_experiment.sh
```

### Docker (Full Stack)

```bash
docker compose up --build
# API  → http://localhost:8000
# Docs → http://localhost:8000/docs
# Demo → http://localhost:8501
```

---

## 📡 API Reference

### POST `/predict`

**Request:**
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

**Response:**
```json
{
  "risk_tier": "Critical",
  "rpn_score": 576,
  "stockout_probability": 0.3429,
  "confidence_score": 0.910,
  "explanation": "CRITICAL risk detected (RPN=576, confidence=91%). Delay of 8 days combined with 7% inventory buffer and +30% demand surge drives stockout probability to 34.3%. Activate emergency procurement protocol.",
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

### POST `/predict/batch`

Process up to 500 scenarios in one call. Returns per-scenario results plus aggregate summary with tier distribution, average RPN, and high-priority count.

### GET `/health`

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "pipeline_ready": true,
  "uptime_seconds": 3847.2
}
```

**Interactive Docs:** `http://localhost:8000/docs` (Swagger UI)

---

## 🏢 Enterprise Integration

### Salesforce Platform Events + Agentforce

```python
from services.integrations import SalesforceConnector
connector = SalesforceConnector(instance_url="...", access_token="...")
connector.publish_risk_event(result)          # → SupplyChainRiskAlert__e
connector.create_case_if_critical(result)     # → Salesforce Case for ops team
```

Integration chain: `LLM-FMEA API → Platform Event → Agentforce Agent → Procurement Escalation`

### SAP Business Technology Platform

```python
from services.integrations import SAPBTPConnector
connector = SAPBTPConnector(api_host="...", client_id="...")
connector.push_risk_assessment(result, material_number="MAT-001", plant_code="US01")
```

Integration chain: `LLM-FMEA API → SAP Integration Suite → OData Adapter → S/4HANA SCM`

### Oracle Integration Cloud

```python
from services.integrations import OracleICConnector
connector = OracleICConnector(oic_host="...")
connector.trigger_risk_workflow(result, po_number="PO-2025-003847")
```

Integration chain: `LLM-FMEA API → OIC REST Trigger → Oracle SCM Cloud → Auto-PO Creation`

---

## 📊 Experimental Results

### DataCo Real-World Validation (n = 180,519)

| Model | Accuracy | Precision | **Recall** | F1 | AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| Rule-Based | 44.2% | 52.1% | 70.4% | 59.9% | 0.320 |
| Decision Tree | 71.1% | 88.9% | 58.3% | 70.5% | 0.762 |
| **LLM-FMEA** | 59.3% | 59.6% | **96.8%** | **73.8%** | **0.760** |

> McNemar's χ² = 156.4, p < 0.001 · 5-Fold CV: 96.9% ± 0.4%

### Ablation Study

| Config | F1 | Δ vs Full |
|---|:---:|:---:|
| FMEA Only | 76.9% | −15.7pp |
| LLM Only | 78.1% | −14.5pp |
| **LLM-FMEA (Full)** | **92.6%** | — |

---

## 📁 Project Structure

```
llm-fmea-scra/
├── demo/dashboard.py              ← Streamlit live demo
├── api/main.py                    ← FastAPI REST API (OpenAPI 3.0)
├── models/
│   ├── fmea_scorer.py             ← Core FMEA RPN engine
│   ├── llm_reasoner.py            ← LLM contextual reasoning
│   ├── rule_based_baseline.py     ← Rule-based baseline
│   └── decision_tree_model.py     ← Decision tree baseline
├── services/
│   ├── pipeline.py                ← Six-layer orchestrator
│   └── integrations.py            ← SAP / Oracle / Salesforce connectors
├── evaluation/metrics.py          ← McNemar, CV, ROC-AUC
├── experiments/
│   ├── run_simulation.py          ← 120-scenario benchmark
│   ├── run_dataco_eval.py         ← DataCo 180K validation
│   └── generate_figures.py        ← 6 publication figures
├── utils/{config,data_loader,logger}.py
├── tests/{test_fmea_scorer,test_pipeline}.py   ← 27 passing tests
├── docs/{API_REFERENCE,DEPLOYMENT_GUIDE,METHODOLOGY}.md
├── .github/workflows/ci.yml       ← GitHub Actions CI
├── Dockerfile + docker-compose.yml
└── scripts/run_full_experiment.sh ← One-command reproduction
```

---

## 🚀 Deployment

### Staged Enterprise Adoption

```
Phase 1 — FMEA-Only Mode
  Layers 1–3 only. Zero LLM cost. Establish baseline.

Phase 2 — Selective LLM Augmentation
  Layer 4 enabled for RPN ≥ 150 events only (~20% of orders).
  Cost: ~$0.01–0.05/invocation. Validate detection uplift.

Phase 3 — Full Hybrid Pipeline
  All layers active. REST API feeds SAP/Oracle/Salesforce.
  Target throughput: 1,000+ assessments/hour (async mode).
```

---

## 🎓 Research Contribution

This repository constitutes a citable, reproducible original contribution:

1. **Algorithmic** — First formal hybrid of FMEA RPN + LLM reasoning for supply chain risk
2. **Empirical** — Dual-dataset validation with McNemar's significance test at scale
3. **Architectural** — Production REST API with native SAP/Oracle/Salesforce connectors

---

## 📖 Citation

```bibtex
@article{donapati2025llmfmea,
  title  = {LLM-Augmented Decision Intelligence for Real-Time Supply Chain
            Risk Assessment: A Parameterized Simulation and FMEA-Based Framework},
  author = {Donapati, Nikhil Reddy},
  year   = {2025},
  note   = {Under Review — SN Computer Science},
  url    = {https://github.com/nikhildonapati/llm-fmea-scra}
}
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for terms.

---

<div align="center">

*Built on 6+ years of enterprise Salesforce and Agentforce AI specialization*

**Nikhil Reddy Donapati · Texas, USA · [ORCID 0009-0006-7699-3928](https://orcid.org/0009-0006-7699-3928)**

</div>
