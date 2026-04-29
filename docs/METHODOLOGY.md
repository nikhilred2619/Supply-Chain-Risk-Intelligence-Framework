# Technical Methodology

**LLM-FMEA Supply Chain Risk Assessment Framework**  
*Paper: "LLM-Augmented Decision Intelligence for Real-Time Supply Chain Risk Assessment"*  
*Author: Nikhil Reddy Donapati · ORCID: 0009-0006-7699-3928*

---

## 1. Theoretical Foundation

### 1.1 FMEA Risk Priority Number

Failure Mode and Effects Analysis (FMEA) provides deterministic risk quantification through the Risk Priority Number:

```
RPN = S × O × D

where:
  S  ∈ [1,10]  Severity    — impact magnitude on operations and finance
  O  ∈ [1,10]  Occurrence  — probability of disruption event materializing
  D  ∈ [1,10]  Detection   — difficulty of early identification
```

**Tier thresholds (paper Section III):**

| RPN Range | Risk Tier |
|---|---|
| ≥ 300 | Critical |
| 150 – 299 | High |
| 60 – 149 | Medium |
| < 60 | Low |

### 1.2 Stockout Probability Model

Grounded in the classical newsvendor framework (Scarf, 1958):

```
Pₛ = min(1, (d × ΔD) / I)

where:
  d   = delay duration (days)
  ΔD  = demand deviation (fractional; only positive surge drives stockout)
  I   = inventory buffer (fractional)
```

**Non-linear threshold (paper Section VI-C):**

At `d > 5 days` concurrent with `I < 10%`, `Pₛ → 1.0` — an empirically identified catastrophic compounding zone that linear models cannot capture.

### 1.3 Asymmetric Cost Proposition

For supply chain risk detection where false negative cost C_fn >> false positive cost C_fp:

```
Optimal threshold satisfies:
  C_fn / C_fp  >  (1 − Recall) / FPR

Industry estimates: C_fn / C_fp ∈ [3×, 8×]
→ Recall-maximizing calibration is theoretically optimal
→ Explains 59.3% accuracy / 96.8% recall operating point on DataCo
```

---

## 2. Six-Layer Pipeline Specification

### Layer 1: Data Ingestion
- Accepts heterogeneous enterprise streams: ERP transactions, CRM events, logistics feeds, scenario parameters
- Input schema validation, null-handling, type enforcement
- Parameterized scenario input: `(d, ΔD, I, R, TR, Fraud)`

### Layer 2: Feature Engineering
- Min-max normalization of continuous variables to `[0,1]`
- Transport risk categorical encoding: Low→1, Medium→2, High→3
- Fraud indicator binary encoding
- Derived signals: demand magnitude `|ΔD|`, inventory depletion risk `(d×ΔD)/I`

### Layer 3: FMEA Risk Quantification
- Severity scoring: calibrated to delay duration, demand deviation, inventory depletion, stockout probability
- Occurrence scoring: supplier reliability (inverse relationship) + transport risk additive
- Detection scoring: fraud scenarios receive elevated D (paper mean D=7.3 vs 4.2 non-fraud)
- RPN computation + stockout probability + service level drop + cost impact

### Layer 4: LLM Contextual Reasoning
- Model: GPT-4-class, temperature=0.2, top-p=0.9, max_tokens=512
- Input: structured JSON with all six scenario variables + preliminary FMEA scores
- Output: Detection score adjustment `∈ [-2,+3]`, confidence score `∈ [0,1]`, NL explanation
- Chain-of-thought prompt design (Wei et al., 2022): per-dimension assessment before synthesis
- Mock mode: deterministic simulation calibrated to paper-reported LLM behavior

### Layer 5: Decision Intelligence Synthesis
- Weighted fusion: FMEA tier (65%) + LLM confidence signal (35%)
- LLM escalation rules: high-confidence (≥0.88) + Pₛ ≥ 50% → one-step tier escalation
- Fraud + detection adjustment ≥ 2 → escalation if borderline (RPN ≥ 60)
- Conservative: never demotes tier by more than one level

### Layer 6: Enterprise Output Interface
- REST API: OpenAPI 3.0, Pydantic v2 validation, async FastAPI
- Output: `risk_tier, rpn_score, stockout_probability, confidence_score, explanation, component_scores, service_level_drop, cost_impact, processing_time_ms`
- Batch endpoint: up to 500 scenarios with aggregate summary
- Health endpoint: Kubernetes-compatible liveness probe

---

## 3. Experimental Methodology

### 3.1 Datasets

**Primary: DataCo Smart Supply Chain (doi: 10.17632/8gx2fvg2k6.5)**
- 180,519 transaction-level records from retail distribution enterprise
- Ground truth: `Delivery Status` field (Late delivery / Shipping canceled = high-risk)
- Positive class (high-risk): 59.1% — class imbalance consistent with paper
- Leakage-free design: input features restricted to order-time observables only

**Secondary: Parameterized Synthetic Benchmark**
- 120 discrete scenarios, 6-variable parameter space
- Tier distribution: Critical 20%, High 30%, Medium 35%, Low 15%
- Purpose: controlled stress-testing, ablation isolation, threshold analysis

### 3.2 Feature Mapping (DataCo → FMEA Schema)

| DataCo Field | FMEA Variable | Mapping Logic |
|---|---|---|
| `Days for shipment (scheduled)` | d (delay) | Direct, clipped to [0,15] |
| `Order Item Quantity` (normalized) | ΔD (demand dev) | z-score × 0.4, clipped [-1,1] |
| `Order Item Profit Ratio` | I (inventory) | Clipped to [0.01, 0.50] |
| Per-category on-time rate | R (reliability) | 1 − category FN rate |
| `Shipping Mode` | TR (transport) | First Class/Same Day→Low, Standard→Medium |
| `Order Status` (SUSPECTED_FRAUD) | Fraud | Binary anomaly flag |

### 3.3 Model Configurations

| System | Configuration |
|---|---|
| Rule-Based (RBS) | Fixed RPN threshold ≥ 150 for high-risk; no training |
| Decision Tree | max_depth=8, criterion='entropy', class_weight='balanced', seed=42 |
| LLM-FMEA | Full six-layer pipeline, mock LLM (calibrated deterministic simulation) |

### 3.4 Evaluation Protocol

- **Train/test split:** 80/20 stratified by binary label
- **Primary metric:** Recall (paper Proposition 1 — asymmetric cost justification)
- **Cross-validation:** 5-fold stratified (seed=42)
- **Statistical significance:** McNemar's test (continuity-corrected)
- **Discriminative capability:** ROC-AUC
- **Component contribution:** Ablation study (FMEA-Only, LLM-Only, Full)

---

## 4. Statistical Analysis

### McNemar's Test

Continuity-corrected McNemar's statistic (paper Equation 3):

```
χ² = (|b − d| − 1)² / (b + d)    df = 1

where:
  b = #(LLM-FMEA correct, RBS incorrect)  — A-favourable discordant pairs
  d = #(RBS correct, LLM-FMEA incorrect)  — B-favourable discordant pairs
```

Results:
- Simulation benchmark: b=38, d=7, χ²=20.0, p<0.001
- DataCo (36,104 test records): χ²=156.4, p<0.001

### RPN–Cost Correlation

Pearson r = 0.89 (paper), r = 0.860 (implementation, n=120, p<0.001)

Validates FMEA RPN as a financially grounded, auditable risk proxy — consistent with Sharma et al. (2022) reporting r=0.84 in logistics contexts.

---

## 5. Ablation Study Design

Three configurations evaluated under identical experimental conditions:

| Configuration | FMEA Layer | LLM Layer | Fusion |
|---|:---:|:---:|:---:|
| FMEA-Only | ✅ | ✗ | Fixed threshold |
| LLM-Only | Partial (feature input) | ✅ | Confidence threshold |
| LLM-FMEA (Full) | ✅ | ✅ | Weighted fusion |

**Key findings:**
- Full hybrid surpasses FMEA-Only by +15.7 F1 points
- Full hybrid surpasses LLM-Only by +14.5 F1 points
- LLM augmentation concentrated in: fraud scenarios (−68% FN) and borderline RPN [140–160]
- FMEA provides deterministic precision foundation; LLM provides contextual recall uplift

---

## 6. References

1. Sharma et al. (2022). IEEE Trans. Engineering Management, 69(4):1345-1356
2. Scarf (1958). Newsvendor framework. Stanford University Press
3. Wei et al. (2022). Chain-of-thought prompting. NeurIPS 35:24824-24837
4. Chopra & Sodhi (2004). MIT Sloan Management Review, 46(1):53-61
5. Ivanov (2017). Simulation-based ripple effect. IJPR, 55(7):2083-2101
6. McNemar (1947). Psychometrika, 12(2):153-157
7. Constante et al. (2019). DataCo dataset. Mendeley Data v5. doi:10.17632/8gx2fvg2k6.5
