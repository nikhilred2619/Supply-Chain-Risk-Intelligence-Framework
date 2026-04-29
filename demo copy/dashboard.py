"""
demo/dashboard.py
──────────────────
LLM-FMEA Supply Chain Risk Assessment — Interactive Streamlit Dashboard

Enterprise-grade live demonstration of the six-layer pipeline.
Designed for GitHub visibility, investor demos, and O-1 visa evidence.

Run:
    streamlit run demo/dashboard.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
from datetime import datetime

from models.fmea_scorer import ScenarioInput, FMEAScorer
from services.pipeline import LLMFMEAPipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM-FMEA | Supply Chain Risk",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }

  .main { background: #0a0e1a; }

  /* Header strip */
  .header-strip {
    background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0a0e1a 100%);
    border-bottom: 2px solid #1e3a5f;
    padding: 20px 32px;
    margin: -1rem -1rem 2rem -1rem;
  }
  .header-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #e2e8f0;
    letter-spacing: -0.5px;
  }
  .header-sub {
    font-size: 0.82rem;
    color: #64748b;
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
  }

  /* Risk tier badges */
  .badge-critical {
    background: #7f1d1d; color: #fca5a5;
    border: 1px solid #ef4444;
    padding: 6px 18px; border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.3rem; font-weight: 700; letter-spacing: 2px;
    display: inline-block;
  }
  .badge-high {
    background: #7c2d12; color: #fdba74;
    border: 1px solid #f97316;
    padding: 6px 18px; border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.3rem; font-weight: 700; letter-spacing: 2px;
    display: inline-block;
  }
  .badge-medium {
    background: #713f12; color: #fde68a;
    border: 1px solid #f59e0b;
    padding: 6px 18px; border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.3rem; font-weight: 700; letter-spacing: 2px;
    display: inline-block;
  }
  .badge-low {
    background: #14532d; color: #86efac;
    border: 1px solid #22c55e;
    padding: 6px 18px; border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.3rem; font-weight: 700; letter-spacing: 2px;
    display: inline-block;
  }

  /* Metric cards */
  .metric-card {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 12px;
  }
  .metric-label {
    font-size: 0.72rem;
    color: #64748b;
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
  }
  .metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
    color: #e2e8f0;
  }

  /* Explanation box */
  .explanation-box {
    background: #0f172a;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 18px 20px;
    font-size: 0.9rem;
    color: #cbd5e1;
    line-height: 1.7;
    font-family: 'IBM Plex Sans', sans-serif;
    margin-top: 8px;
  }

  /* Section headers */
  .section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 8px;
    margin-bottom: 16px;
  }

  /* Stmetric override */
  [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 2rem !important;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1e293b;
  }

  /* Buttons */
  .stButton > button {
    background: #1e3a5f;
    color: #93c5fd;
    border: 1px solid #2563eb;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 1px;
    border-radius: 6px;
    width: 100%;
    padding: 12px;
    font-size: 0.9rem;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    background: #2563eb;
    border-color: #60a5fa;
    color: white;
  }

  /* Pipeline layer cards */
  .layer-card {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .layer-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #3b82f6;
    font-weight: 700;
    min-width: 24px;
  }
  .layer-name { font-weight: 600; color: #cbd5e1; font-size: 0.85rem; }
  .layer-desc { font-size: 0.75rem; color: #64748b; }

  /* History table */
  .history-row-critical { border-left: 3px solid #ef4444; }
  .history-row-high { border-left: 3px solid #f97316; }

  div.stAlert { background: #0f172a; border-color: #1e3a5f; }

  .formula-box {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 16px 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #7dd3fc;
    margin-top: 12px;
  }
</style>
""", unsafe_allow_html=True)


# ── Singleton pipeline ────────────────────────────────────────────────────────
@st.cache_resource
def get_pipeline():
    return LLMFMEAPipeline(mock_llm=True)


@st.cache_resource
def get_scorer():
    return FMEAScorer()


# ── Session state init ────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "assess_count" not in st.session_state:
    st.session_state.assess_count = 0


# ── Colour helpers ────────────────────────────────────────────────────────────
TIER_COLOR   = {"Critical": "#ef4444", "High": "#f97316", "Medium": "#f59e0b", "Low": "#22c55e"}
TIER_BG      = {"Critical": "#7f1d1d", "High": "#7c2d12", "Medium": "#713f12", "Low": "#14532d"}
TIER_BADGE   = {"Critical": "badge-critical", "High": "badge-high",
                "Medium": "badge-medium", "Low": "badge-low"}


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Input Panel
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="section-header">SCENARIO PARAMETERS</div>', unsafe_allow_html=True)

    delay = st.slider("Delay Duration (days)", 0, 15, 8, 1,
                       help="d — days behind scheduled delivery")

    demand_pct = st.slider("Demand Deviation (%)", -30, 60, 30, 5,
                            help="ΔD — demand surge or drop vs forecast")

    inv_pct = st.slider("Inventory Buffer (%)", 1, 40, 7, 1,
                          help="I — current inventory as % of demand")

    reliability = st.slider("Supplier Reliability", 0.40, 0.95, 0.72, 0.01,
                             help="R — historical on-time fulfillment rate")

    transport_risk = st.selectbox("Transport Risk Class", ["Low", "Medium", "High"], index=2,
                                   help="TR — logistics exposure level")

    fraud = st.checkbox("Fraud / Integrity Anomaly", value=False,
                         help="Integrity-based failure mode detected")

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("⚡  ASSESS RISK", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">PRESETS</div>', unsafe_allow_html=True)

    preset_cols = st.columns(2)
    with preset_cols[0]:
        if st.button("🔴 Crisis", use_container_width=True):
            st.session_state["preset"] = "crisis"
            st.rerun()
        if st.button("🟡 Moderate", use_container_width=True):
            st.session_state["preset"] = "moderate"
            st.rerun()
    with preset_cols[1]:
        if st.button("🟠 High Risk", use_container_width=True):
            st.session_state["preset"] = "high"
            st.rerun()
        if st.button("🟢 Normal", use_container_width=True):
            st.session_state["preset"] = "normal"
            st.rerun()

    # Apply presets
    PRESETS = {
        "crisis":   dict(delay=15, demand_pct=60, inv_pct=5,  reliability=0.42, transport_risk="High",   fraud=True),
        "high":     dict(delay=10, demand_pct=30, inv_pct=8,  reliability=0.58, transport_risk="High",   fraud=False),
        "moderate": dict(delay=5,  demand_pct=20, inv_pct=15, reliability=0.75, transport_risk="Medium", fraud=False),
        "normal":   dict(delay=1,  demand_pct=-5, inv_pct=35, reliability=0.91, transport_risk="Low",    fraud=False),
    }
    if "preset" in st.session_state:
        p = PRESETS[st.session_state.pop("preset")]
        delay = p["delay"]; demand_pct = p["demand_pct"]; inv_pct = p["inv_pct"]
        reliability = p["reliability"]; transport_risk = p["transport_risk"]; fraud = p["fraud"]
        run_btn = True   # auto-run

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">PIPELINE STATUS</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="layer-card">
  <span class="layer-num">L1</span>
  <div><div class="layer-name">Data Ingestion</div></div>
</div>
<div class="layer-card">
  <span class="layer-num">L2</span>
  <div><div class="layer-name">Feature Engineering</div></div>
</div>
<div class="layer-card">
  <span class="layer-num">L3</span>
  <div><div class="layer-name">FMEA Quantification</div></div>
</div>
<div class="layer-card">
  <span class="layer-num">L4</span>
  <div><div class="layer-name">LLM Reasoning</div></div>
</div>
<div class="layer-card">
  <span class="layer-num">L5</span>
  <div><div class="layer-name">Decision Synthesis</div></div>
</div>
<div class="layer-card">
  <span class="layer-num">L6</span>
  <div><div class="layer-name">Enterprise Output</div></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="header-strip">
  <div class="header-title">🔴 LLM-FMEA · Supply Chain Risk Assessment</div>
  <div class="header-sub">
    Six-Layer Decision Intelligence Pipeline · FMEA + LLM Contextual Reasoning ·
    Validated on 180,519 Records · 96.8% Recall
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# Build scenario on every interaction
scenario = ScenarioInput(
    delay_duration=float(delay),
    demand_deviation=demand_pct / 100.0,
    inventory_buffer=inv_pct / 100.0,
    supplier_reliability=reliability,
    transport_risk=transport_risk,
    fraud_indicator=fraud,
)

if run_btn or st.session_state.assess_count == 0:
    pipeline = get_pipeline()
    with st.spinner("Running six-layer pipeline..."):
        result = pipeline.assess(scenario)
    st.session_state.last_result = result
    st.session_state.assess_count += 1

    # Append to history
    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "tier":      result.risk_tier,
        "rpn":       result.rpn_score,
        "confidence": result.confidence_score,
        "ps":        round(result.stockout_probability, 3),
        "delay":     delay,
        "inv":       inv_pct,
        "fraud":     "Yes" if fraud else "No",
        "latency_ms": result.processing_time_ms,
    })

result = st.session_state.get("last_result")

if result is None:
    st.info("Set parameters in the sidebar and click **ASSESS RISK**.")
    st.stop()

tier  = result.risk_tier
color = TIER_COLOR[tier]


# ── Row 1: Risk Badge + Key Metrics ──────────────────────────────────────────
badge_col, m1, m2, m3, m4 = st.columns([2, 1.3, 1.3, 1.3, 1.3])

with badge_col:
    st.markdown(
        f'<div style="padding:8px 0;">'
        f'<div class="metric-label">RISK CLASSIFICATION</div>'
        f'<span class="{TIER_BADGE[tier]}">{tier.upper()}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

with m1:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">RPN SCORE</div>'
        f'<div class="metric-value" style="color:{color};">{result.rpn_score}</div>'
        f'<div style="font-size:0.7rem;color:#64748b;font-family:IBM Plex Mono,monospace;">/ 1000</div>'
        f'</div>', unsafe_allow_html=True)

with m2:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">CONFIDENCE</div>'
        f'<div class="metric-value">{result.confidence_score:.0%}</div>'
        f'<div style="font-size:0.7rem;color:#64748b;font-family:IBM Plex Mono,monospace;">LLM confidence</div>'
        f'</div>', unsafe_allow_html=True)

with m3:
    ps_pct = result.stockout_probability * 100
    ps_color = "#ef4444" if ps_pct >= 50 else "#f59e0b" if ps_pct >= 25 else "#22c55e"
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">STOCKOUT Pₛ</div>'
        f'<div class="metric-value" style="color:{ps_color};">{ps_pct:.1f}%</div>'
        f'<div style="font-size:0.7rem;color:#64748b;font-family:IBM Plex Mono,monospace;">probability</div>'
        f'</div>', unsafe_allow_html=True)

with m4:
    ci_pct = result.cost_impact * 100
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">COST IMPACT</div>'
        f'<div class="metric-value">+{ci_pct:.1f}%</div>'
        f'<div style="font-size:0.7rem;color:#64748b;font-family:IBM Plex Mono,monospace;">projected</div>'
        f'</div>', unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)

# ── Row 2: RPN Gauge + FMEA Breakdown + Explanation ──────────────────────────
gauge_col, breakdown_col, explain_col = st.columns([1.4, 1.3, 2.3])

with gauge_col:
    st.markdown('<div class="section-header">RPN GAUGE</div>', unsafe_allow_html=True)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=result.rpn_score,
        number={"font": {"size": 40, "family": "IBM Plex Mono", "color": color},
                "suffix": ""},
        gauge={
            "axis": {"range": [0, 1000],
                     "tickwidth": 1,
                     "tickcolor": "#334155",
                     "tickfont": {"color": "#475569", "size": 10}},
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": "#0f172a",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 60],   "color": "#14532d"},
                {"range": [60, 150], "color": "#713f12"},
                {"range": [150, 300],"color": "#7c2d12"},
                {"range": [300, 1000],"color": "#450a0a"},
            ],
            "threshold": {
                "line": {"color": "#fbbf24", "width": 2},
                "thickness": 0.75,
                "value": result.rpn_score,
            },
        },
        title={"text": f"<b>{tier}</b>",
               "font": {"size": 14, "family": "IBM Plex Mono", "color": color}},
    ))
    fig_gauge.update_layout(
        height=240,
        margin=dict(t=30, b=10, l=20, r=20),
        paper_bgcolor="#0a0e1a",
        font={"family": "IBM Plex Mono"},
    )
    st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        f'<div class="formula-box">'
        f'RPN = S × O × D<br>'
        f'    = {result.severity} × {result.occurrence} × {result.detection}<br>'
        f'    = <b style="color:{color}">{result.rpn_score}</b>'
        f'</div>', unsafe_allow_html=True)


with breakdown_col:
    st.markdown('<div class="section-header">FMEA COMPONENT SCORES</div>', unsafe_allow_html=True)

    components = {
        "Severity (S)":   result.severity,
        "Occurrence (O)": result.occurrence,
        "Detection (D)":  result.detection,
    }
    comp_colors = ["#f87171", "#fb923c", "#a78bfa"]

    fig_bar = go.Figure()
    for i, (name, val) in enumerate(components.items()):
        fig_bar.add_trace(go.Bar(
            x=[val],
            y=[name],
            orientation="h",
            marker=dict(color=comp_colors[i], opacity=0.85),
            text=[f"{val}/10"],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=13, color=comp_colors[i]),
        ))

    fig_bar.update_layout(
        showlegend=False,
        height=190,
        margin=dict(t=5, b=5, l=5, r=60),
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        xaxis=dict(range=[0, 12], showgrid=False,
                   tickfont=dict(color="#475569", family="IBM Plex Mono"),
                   showticklabels=False),
        yaxis=dict(tickfont=dict(color="#94a3b8", family="IBM Plex Mono", size=12)),
        bargap=0.35,
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # LLM adjustment callout
    adj = result.detection_adjustment
    adj_color = "#f97316" if adj > 0 else "#22c55e" if adj < 0 else "#64748b"
    adj_label = f"+{adj}" if adj >= 0 else str(adj)
    st.markdown(
        f'<div class="metric-card" style="padding:12px 14px;">'
        f'<div class="metric-label">LLM DETECTION ADJUSTMENT</div>'
        f'<div style="font-family:IBM Plex Mono;font-size:1.2rem;color:{adj_color};font-weight:700;">'
        f'{adj_label} pts &nbsp;·&nbsp; Raw D: {result.detection_raw} → Final: {result.detection}</div>'
        f'<div style="font-size:0.72rem;color:#64748b;margin-top:4px;">'
        f'{"Fraud/integrity boost applied" if fraud else "Standard detection scoring"}'
        f'</div></div>',
        unsafe_allow_html=True
    )

    # Service level drop
    sl_pct = result.service_level_drop * 100
    sl_color = "#ef4444" if sl_pct >= 25 else "#f59e0b" if sl_pct >= 15 else "#22c55e"
    st.markdown(
        f'<div class="metric-card" style="padding:12px 14px;">'
        f'<div class="metric-label">SERVICE LEVEL DROP</div>'
        f'<div style="font-family:IBM Plex Mono;font-size:1.2rem;color:{sl_color};font-weight:700;">'
        f'−{sl_pct:.1f}%</div>'
        f'</div>', unsafe_allow_html=True)


with explain_col:
    st.markdown('<div class="section-header">AI RISK EXPLANATION — LAYER 4 OUTPUT</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="explanation-box">{result.explanation}</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">STOCKOUT PROBABILITY · FORMULA</div>',
                unsafe_allow_html=True)

    # Stockout formula visualization
    d_cont = np.linspace(0, 15, 200)
    fig_ps = go.Figure()

    inv_levels = [(5, "#ef4444"), (10, "#f97316"), (20, "#f59e0b"), (40, "#22c55e")]
    dd = max(0, demand_pct / 100.0)
    for inv, lc in inv_levels:
        ps_line = np.minimum(1.0, (d_cont * dd) / (inv / 100))
        fig_ps.add_trace(go.Scatter(
            x=d_cont, y=ps_line * 100,
            mode="lines", name=f"I={inv}%",
            line=dict(color=lc, width=2),
        ))

    # Current scenario point
    ps_curr = result.stockout_probability * 100
    fig_ps.add_trace(go.Scatter(
        x=[delay], y=[ps_curr],
        mode="markers", name="Current",
        marker=dict(color=color, size=12, symbol="diamond",
                    line=dict(color="white", width=2)),
        showlegend=True,
    ))
    fig_ps.add_hline(y=50, line=dict(color="#475569", dash="dot", width=1))
    fig_ps.add_vline(x=5,  line=dict(color="#475569", dash="dot", width=1))

    fig_ps.update_layout(
        height=200,
        margin=dict(t=10, b=30, l=40, r=10),
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0f172a",
        legend=dict(font=dict(color="#94a3b8", size=10, family="IBM Plex Mono"),
                    bgcolor="rgba(0,0,0,0)", x=0.01, y=0.98),
        xaxis=dict(title="Delay d (days)", titlefont=dict(color="#64748b", size=11),
                   tickfont=dict(color="#475569"), gridcolor="#1e293b"),
        yaxis=dict(title="Pₛ (%)", titlefont=dict(color="#64748b", size=11),
                   tickfont=dict(color="#475569"), gridcolor="#1e293b",
                   range=[0, 105]),
        font=dict(family="IBM Plex Mono"),
    )
    st.plotly_chart(fig_ps, use_container_width=True, config={"displayModeBar": False})


# ── Row 3: Scenario History ───────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">SCENARIO HISTORY · SESSION</div>', unsafe_allow_html=True)

if st.session_state.history:
    df_hist = pd.DataFrame(st.session_state.history[::-1])   # newest first

    # RPN trend sparkline
    if len(df_hist) > 1:
        rpn_col, table_col = st.columns([1.2, 2.8])

        with rpn_col:
            rpn_series = [row["rpn"] for row in st.session_state.history]
            tier_series = [row["tier"] for row in st.session_state.history]
            colors_series = [TIER_COLOR[t] for t in tier_series]

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=list(range(len(rpn_series))),
                y=rpn_series,
                mode="lines+markers",
                line=dict(color="#3b82f6", width=2),
                marker=dict(color=colors_series, size=9,
                            line=dict(color="white", width=1.5)),
            ))
            fig_trend.add_hline(y=300, line=dict(color="#ef4444", dash="dot", width=1))
            fig_trend.add_hline(y=150, line=dict(color="#f97316", dash="dot", width=1))
            fig_trend.update_layout(
                title=dict(text="RPN History", font=dict(color="#94a3b8", size=12,
                                                           family="IBM Plex Mono")),
                height=200,
                margin=dict(t=35, b=30, l=40, r=10),
                paper_bgcolor="#0a0e1a",
                plot_bgcolor="#0f172a",
                xaxis=dict(showticklabels=False, gridcolor="#1e293b"),
                yaxis=dict(range=[0, 1050], gridcolor="#1e293b",
                           tickfont=dict(color="#475569")),
            )
            st.plotly_chart(fig_trend, use_container_width=True,
                            config={"displayModeBar": False})

        with table_col:
            display_cols = ["timestamp", "tier", "rpn", "confidence", "ps",
                            "delay", "inv", "fraud", "latency_ms"]
            rename_map = {
                "timestamp": "Time", "tier": "Tier", "rpn": "RPN",
                "confidence": "Conf.", "ps": "Pₛ",
                "delay": "d(days)", "inv": "I(%)",
                "fraud": "Fraud", "latency_ms": "ms"
            }
            df_display = df_hist[display_cols].rename(columns=rename_map)
            df_display["Conf."] = df_display["Conf."].apply(lambda x: f"{x:.0%}")
            st.dataframe(
                df_display,
                use_container_width=True,
                height=180,
                hide_index=True,
            )
    else:
        st.markdown(f"*Run more scenarios to see trend analysis.*")
        st.json(st.session_state.history[0])


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="border-top:1px solid #1e293b; padding-top:16px; 
     font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:#334155;
     display:flex; justify-content:space-between;">
  <span>LLM-FMEA SCRA Framework v1.0.0 · Nikhil Reddy Donapati · ORCID 0009-0006-7699-3928</span>
  <span>Validated on 180,519 DataCo records · 96.8% Recall · McNemar χ²=156.4 p&lt;0.001</span>
</div>
""", unsafe_allow_html=True)
