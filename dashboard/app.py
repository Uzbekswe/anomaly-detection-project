"""Manufacturing Anomaly Detection — Interactive Dashboard.

A self-contained Streamlit application that loads the trained LSTM-AE model
directly from disk and simulates a fleet of turbofan engines degrading over
time.  No external API server required — the model runs in-process.

Real-world analogy:
    In production the flow is  IoT Sensors → Kafka/MQTT → Feature Pipeline
    → Model API → Dashboard → Alerts.   This demo collapses the pipeline
    into a single app for easy sharing and demonstration.

Usage (local):
    streamlit run dashboard/app.py

Deployment:
    Push to GitHub → connect repo to Streamlit Community Cloud.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ARTIFACT_DIR = _PROJECT_ROOT / "data" / "artifacts"
_SCALER_PATH = _ARTIFACT_DIR / "scaler.json"


# ──────────────────────────────────────────────
# Model loading (cached — runs once)
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading anomaly detection model…")
def load_predictor():
    """Load the LSTM-AE model + scaler from local artifacts.

    Uses Streamlit's ``cache_resource`` so the model stays in memory
    across reruns (no reload on every button click).
    """
    import sys
    sys.path.insert(0, str(_PROJECT_ROOT))
    from src.serving.predictor import AnomalyPredictor, load_model_from_local
    mc = load_model_from_local("lstm_autoencoder")
    return AnomalyPredictor(mc)


@st.cache_data(show_spinner=False)
def load_scaler_ranges():
    """Load feature ranges from scaler.json for realistic data simulation."""
    if not _SCALER_PATH.exists():
        return None, None, []
    with open(_SCALER_PATH) as f:
        s = json.load(f)
    if "min_vals" not in s:
        return None, None, []
    names = list(s["min_vals"].keys())
    fmin = np.array(list(s["min_vals"].values()), dtype=np.float64)
    fmax = np.array(list(s["max_vals"].values()), dtype=np.float64)
    return fmin, fmax, names


# ──────────────────────────────────────────────
# Data simulation helpers
# ──────────────────────────────────────────────

FEAT_MIN, FEAT_MAX, FEAT_NAMES = load_scaler_ranges()
NUM_FEATURES = len(FEAT_NAMES) if FEAT_NAMES else 101
WINDOW_SIZE = 30
THRESHOLD = 0.23  # from training


def simulate_engine_window(
    cycle: int,
    max_cycle: int,
    noise_std: float = 0.12,
) -> np.ndarray:
    """Simulate one sensor window for an engine at a given cycle.

    Real-world context
    ──────────────────
    In a factory, each engine/machine has IoT sensors streaming readings
    every second.  Early in the engine's life the readings are stable.
    As the engine degrades toward failure, sensor values drift and become
    noisier — the model should detect this as anomalous.

    This function reproduces that behavior:
      - cycle / max_cycle = 0 → brand-new engine, values at centre of range
      - cycle / max_cycle ≈ 1 → near failure, values drift + noise spikes
    """
    if FEAT_MIN is None or FEAT_MAX is None:
        return np.random.uniform(0.3, 0.7, (WINDOW_SIZE, NUM_FEATURES))

    progress = min(cycle / max_cycle, 1.0)       # 0 → 1  (new → failing)
    feat_range = FEAT_MAX - FEAT_MIN
    centre = (FEAT_MIN + FEAT_MAX) / 2.0

    # Base: normal readings with small jitter
    jitter = np.random.normal(0, noise_std, (WINDOW_SIZE, NUM_FEATURES))
    base = centre + jitter * feat_range

    # Degradation: gradually shift a growing subset of sensors
    if progress > 0.4:
        degradation_strength = (progress - 0.4) / 0.6          # 0 → 1
        n_affected = int(NUM_FEATURES * 0.15 * degradation_strength)
        if n_affected > 0:
            rng = np.random.RandomState(cycle)                  # deterministic per cycle
            affected = rng.choice(NUM_FEATURES, size=n_affected, replace=False)
            drift = degradation_strength * feat_range[affected] * rng.uniform(0.3, 1.5, n_affected)
            base[:, affected] += drift

    # Near-failure: sudden spikes in the last 20% of window
    if progress > 0.85:
        spike_channels = np.random.choice(NUM_FEATURES, size=max(3, NUM_FEATURES // 15), replace=False)
        spike_start = int(WINDOW_SIZE * 0.6)
        for ch in spike_channels:
            base[spike_start:, ch] = FEAT_MAX[ch] + feat_range[ch] * np.random.uniform(0.5, 2.0)

    # Clip normal readings to range; allow spikes for near-failure
    if progress < 0.85:
        base = np.clip(base, FEAT_MIN, FEAT_MAX)

    return base


def predict_window(predictor, window: np.ndarray) -> dict:
    """Run prediction and add timing info."""
    t0 = time.perf_counter()
    result = predictor.predict(window)
    result["processing_time_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result


# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    page_icon="🏭",
    layout="wide",
)


# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────

def _defaults():
    return dict(
        history=[],
        anomaly_events=[],
        engines={},          # engine_id → {cycle, max_cycle, status}
        sim_running=False,
        sim_step=0,
    )

for k, v in _defaults().items():
    if k not in st.session_state:
        st.session_state[k] = v


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/factory.png", width=64)
    st.title("Controls")

    st.divider()
    st.subheader("🏭 Engine Fleet")
    num_engines = st.slider("Number of engines", 1, 10, 4)
    max_lifecycle = st.slider("Engine lifecycle (cycles)", 50, 300, 150,
                              help="How many cycles until an engine reaches end-of-life")
    sim_speed = st.select_slider("Simulation speed",
                                 options=[0.5, 1.0, 2.0, 5.0, 10.0],
                                 value=2.0,
                                 help="Cycles per click in Run Simulation")

    st.divider()
    st.subheader("⚙️ Model Info")
    st.caption(f"**Model:** LSTM Autoencoder")
    st.caption(f"**Features:** {NUM_FEATURES}")
    st.caption(f"**Window size:** {WINDOW_SIZE}")
    st.caption(f"**Threshold:** {THRESHOLD}")

    st.divider()
    if st.button("🔄 Reset Everything", use_container_width=True):
        for k, v in _defaults().items():
            st.session_state[k] = v
        st.rerun()


# ──────────────────────────────────────────────
# Initialise engines
# ──────────────────────────────────────────────

if not st.session_state.engines or len(st.session_state.engines) != num_engines:
    st.session_state.engines = {}
    for i in range(num_engines):
        eid = f"engine_{i + 1:03d}"
        # Stagger engines at different lifecycle points for variety
        start_cycle = int(max_lifecycle * (i / (num_engines + 1)) * 0.5)
        st.session_state.engines[eid] = {
            "cycle": start_cycle,
            "max_cycle": max_lifecycle + np.random.randint(-20, 20),
            "status": "healthy",
        }


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────

st.title("🏭 Manufacturing Anomaly Detection")
st.caption("Real-time IoT sensor monitoring — NASA CMAPSS Turbofan Engine Degradation Simulation")

st.info(
    "**How it works in the real world:**  IoT sensors on each engine stream "
    "readings every second → a feature pipeline computes rolling statistics → "
    "the LSTM Autoencoder reconstructs the sensor window → high reconstruction "
    "error = anomaly (engine degrading).  This demo simulates that entire flow "
    "in-browser — click **▶ Run Simulation** to watch engines degrade over time.",
    icon="💡",
)


# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────

tab_live, tab_fleet, tab_analysis, tab_arch = st.tabs([
    "📡 Live Monitor", "🏗️ Fleet Overview", "📊 Analysis", "🔧 Architecture"
])


# ══════════════════════════════════════════════
# TAB 1: Live Monitor
# ══════════════════════════════════════════════

with tab_live:
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        run_sim = st.button("▶ Run Simulation Step", type="primary", use_container_width=True)
    with col_btn2:
        run_burst = st.button("⏩ Run 10 Steps", use_container_width=True)
    with col_btn3:
        run_50 = st.button("⏭️ Run 50 Steps", use_container_width=True)

    steps_to_run = 0
    if run_sim:
        steps_to_run = int(sim_speed)
    elif run_burst:
        steps_to_run = 10
    elif run_50:
        steps_to_run = 50

    if steps_to_run > 0:
        predictor = load_predictor()
        progress_bar = st.progress(0, text="Simulating…")

        for step in range(steps_to_run):
            for eid, eng in st.session_state.engines.items():
                if eng["status"] == "failed":
                    continue

                eng["cycle"] += 1
                window = simulate_engine_window(eng["cycle"], eng["max_cycle"])
                result = predict_window(predictor, window)

                ts = (datetime.now(timezone.utc) + timedelta(seconds=step)).isoformat()
                progress = eng["cycle"] / eng["max_cycle"]

                entry = {
                    "timestamp": ts,
                    "engine_id": eid,
                    "cycle": eng["cycle"],
                    "lifecycle_pct": round(progress * 100, 1),
                    "anomaly_score": result["anomaly_score"],
                    "is_anomaly": result["is_anomaly"],
                    "confidence": result["confidence"],
                    "processing_time_ms": result["processing_time_ms"],
                }
                st.session_state.history.append(entry)

                if result["is_anomaly"]:
                    st.session_state.anomaly_events.append(entry)
                    if progress > 0.9:
                        eng["status"] = "critical"
                    else:
                        eng["status"] = "warning"
                else:
                    eng["status"] = "healthy"

                if eng["cycle"] >= eng["max_cycle"]:
                    eng["status"] = "failed"

            progress_bar.progress((step + 1) / steps_to_run, text=f"Step {step + 1}/{steps_to_run}")

        progress_bar.empty()

        # Trim history
        if len(st.session_state.history) > 2000:
            st.session_state.history = st.session_state.history[-2000:]
        if len(st.session_state.anomaly_events) > 200:
            st.session_state.anomaly_events = st.session_state.anomaly_events[-200:]

    # ── Latest status banner ──
    if st.session_state.history:
        latest = st.session_state.history[-1]
        if latest["is_anomaly"]:
            st.error(
                f"⚠️ ANOMALY — **{latest['engine_id']}** | "
                f"Score: {latest['anomaly_score']:.4f} | "
                f"Cycle: {latest['cycle']} ({latest['lifecycle_pct']}% life)"
            )
        else:
            st.success(
                f"✅ Normal — **{latest['engine_id']}** | "
                f"Score: {latest['anomaly_score']:.4f} | "
                f"Cycle: {latest['cycle']} ({latest['lifecycle_pct']}% life)"
            )

    # ── KPI Row ──
    if st.session_state.history:
        m1, m2, m3, m4, m5 = st.columns(5)

        total = len(st.session_state.history)
        anomalies_list = [h for h in st.session_state.history if h["is_anomaly"]]
        scores_all = [h["anomaly_score"] for h in st.session_state.history]
        latencies = [h["processing_time_ms"] for h in st.session_state.history]
        critical_count = sum(1 for e in st.session_state.engines.values() if e["status"] in ("critical", "failed"))

        m1.metric("Total Readings", f"{total:,}")
        m2.metric("Anomalies", len(anomalies_list))
        m3.metric("Anomaly Rate", f"{100 * len(anomalies_list) / total:.1f}%")
        m4.metric("Avg Latency", f"{np.mean(latencies):.1f} ms")
        m5.metric("Critical Engines", critical_count, delta=None)

    # ── Anomaly score chart ──
    if st.session_state.history:
        st.subheader("Anomaly Score Timeline")

        fig = go.Figure()

        for eid in st.session_state.engines:
            eng_history = [h for h in st.session_state.history if h["engine_id"] == eid]
            if not eng_history:
                continue
            cycles = [h["cycle"] for h in eng_history]
            scores = [h["anomaly_score"] for h in eng_history]
            fig.add_trace(go.Scatter(
                x=cycles, y=scores,
                mode="lines+markers",
                name=eid,
                marker=dict(size=3),
                line=dict(width=2),
            ))

        # Threshold line
        all_cycles = [h["cycle"] for h in st.session_state.history]
        if all_cycles:
            fig.add_hline(
                y=THRESHOLD, line_dash="dash", line_color="red",
                annotation_text=f"Threshold ({THRESHOLD})",
                annotation_position="top right",
            )

        fig.update_layout(
            xaxis_title="Engine Cycle",
            yaxis_title="Anomaly Score (reconstruction error)",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2: Fleet Overview
# ══════════════════════════════════════════════

with tab_fleet:
    st.subheader("Engine Fleet Status")

    if not st.session_state.engines:
        st.info("Click **▶ Run Simulation Step** to start.")
    else:
        cols = st.columns(min(len(st.session_state.engines), 4))
        for i, (eid, eng) in enumerate(st.session_state.engines.items()):
            col = cols[i % len(cols)]
            with col:
                progress = eng["cycle"] / eng["max_cycle"]
                status = eng["status"]

                status_emoji = {"healthy": "🟢", "warning": "🟡", "critical": "🔴", "failed": "⚫"}
                status_label = {"healthy": "Healthy", "warning": "Warning", "critical": "Critical", "failed": "Failed"}

                st.markdown(f"### {status_emoji.get(status, '⚪')} {eid}")
                st.progress(min(progress, 1.0))
                st.caption(f"Cycle {eng['cycle']} / {eng['max_cycle']}  •  {status_label.get(status, status)}")

                # Latest score for this engine
                eng_readings = [h for h in st.session_state.history if h["engine_id"] == eid]
                if eng_readings:
                    last = eng_readings[-1]
                    st.metric("Last Score", f"{last['anomaly_score']:.4f}",
                              delta=f"{'ANOMALY' if last['is_anomaly'] else 'normal'}",
                              delta_color="inverse" if last["is_anomaly"] else "off")
                else:
                    st.caption("No readings yet")

        # Fleet summary table
        st.divider()
        st.subheader("Fleet Summary")

        fleet_data = []
        for eid, eng in st.session_state.engines.items():
            eng_h = [h for h in st.session_state.history if h["engine_id"] == eid]
            eng_anomalies = [h for h in eng_h if h["is_anomaly"]]
            fleet_data.append({
                "Engine": eid,
                "Status": eng["status"].title(),
                "Cycle": eng["cycle"],
                "Lifecycle %": f"{eng['cycle'] / eng['max_cycle'] * 100:.0f}%",
                "Readings": len(eng_h),
                "Anomalies": len(eng_anomalies),
                "Anomaly Rate": f"{100 * len(eng_anomalies) / max(len(eng_h), 1):.1f}%",
                "Last Score": f"{eng_h[-1]['anomaly_score']:.4f}" if eng_h else "—",
            })

        st.dataframe(fleet_data, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 3: Analysis
# ══════════════════════════════════════════════

with tab_analysis:
    if not st.session_state.history:
        st.info("Run some simulation steps first to generate data for analysis.")
    else:
        col_a1, col_a2 = st.columns(2)

        all_scores = [h["anomaly_score"] for h in st.session_state.history]
        all_anomaly_flags = [h["is_anomaly"] for h in st.session_state.history]

        # Score distribution
        with col_a1:
            st.subheader("Score Distribution")
            normal_scores = [s for s, a in zip(all_scores, all_anomaly_flags) if not a]
            anomaly_scores = [s for s, a in zip(all_scores, all_anomaly_flags) if a]

            fig_dist = go.Figure()
            if normal_scores:
                fig_dist.add_trace(go.Histogram(x=normal_scores, name="Normal",
                                                marker_color="#2ecc71", opacity=0.7, nbinsx=30))
            if anomaly_scores:
                fig_dist.add_trace(go.Histogram(x=anomaly_scores, name="Anomaly",
                                                marker_color="#e74c3c", opacity=0.7, nbinsx=30))
            fig_dist.add_vline(x=THRESHOLD, line_dash="dash", line_color="red",
                               annotation_text="Threshold")
            fig_dist.update_layout(barmode="overlay", height=350,
                                   xaxis_title="Anomaly Score", yaxis_title="Count",
                                   margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_dist, use_container_width=True)

        # Latency distribution
        with col_a2:
            st.subheader("Inference Latency")
            latencies = [h["processing_time_ms"] for h in st.session_state.history]
            fig_lat = go.Figure()
            fig_lat.add_trace(go.Histogram(x=latencies, name="Latency",
                                           marker_color="#3498db", nbinsx=30))
            fig_lat.update_layout(height=350,
                                  xaxis_title="Latency (ms)", yaxis_title="Count",
                                  margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_lat, use_container_width=True)

        # Per-engine degradation
        st.subheader("Engine Degradation Curves")
        st.caption("Watch how anomaly scores increase as engines age — the model detects degradation before failure.")

        fig_deg = go.Figure()
        for eid in st.session_state.engines:
            eng_h = [h for h in st.session_state.history if h["engine_id"] == eid]
            if not eng_h:
                continue
            pcts = [h["lifecycle_pct"] for h in eng_h]
            scores = [h["anomaly_score"] for h in eng_h]
            fig_deg.add_trace(go.Scatter(x=pcts, y=scores, mode="lines",
                                         name=eid, line=dict(width=2)))
        fig_deg.add_hline(y=THRESHOLD, line_dash="dash", line_color="red",
                          annotation_text="Threshold")
        fig_deg.update_layout(
            xaxis_title="Engine Lifecycle (%)",
            yaxis_title="Anomaly Score",
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_deg, use_container_width=True)

        # Anomaly events log
        if st.session_state.anomaly_events:
            st.subheader("Anomaly Event Log")
            events = list(reversed(st.session_state.anomaly_events[-50:]))
            st.dataframe(
                events,
                column_config={
                    "timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
                    "engine_id": st.column_config.TextColumn("Engine"),
                    "cycle": st.column_config.NumberColumn("Cycle"),
                    "lifecycle_pct": st.column_config.NumberColumn("Life %", format="%.1f%%"),
                    "anomaly_score": st.column_config.NumberColumn("Score", format="%.4f"),
                    "confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
                    "processing_time_ms": st.column_config.NumberColumn("Latency", format="%.1f ms"),
                },
                use_container_width=True,
                hide_index=True,
            )


# ══════════════════════════════════════════════
# TAB 4: Architecture
# ══════════════════════════════════════════════

with tab_arch:
    st.subheader("System Architecture")
    st.markdown(
        """
    ### How This Works in the Real World

    ```
    ┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
    │  IoT Sensors  │────▶│  Message Queue   │────▶│  Feature Engine  │────▶│  ML Model   │
    │  (per engine) │     │  (Kafka / MQTT)  │     │  (rolling stats) │     │  (LSTM-AE)  │
    └──────────────┘     └─────────────────┘     └──────────────────┘     └──────┬──────┘
                                                                                  │
                              ┌──────────────┐     ┌──────────────────┐           │
                              │  Alert System │◀────│   Dashboard      │◀──────────┘
                              │  (PagerDuty)  │     │   (this app)     │
                              └──────────────┘     └──────────────────┘
    ```

    ### Pipeline Stages

    | Stage | What Happens | This Demo |
    |-------|-------------|-----------|
    | **1. Data Collection** | Sensors on each turbofan engine stream temperature, pressure, speed readings at 1 Hz | Simulated with realistic value ranges from CMAPSS dataset |
    | **2. Feature Engineering** | Raw readings → sliding windows (30 cycles) + rolling mean/std statistics → 101 features | Built into simulation using trained scaler ranges |
    | **3. Anomaly Detection** | LSTM Autoencoder reconstructs the window; high reconstruction error = anomaly | Model loaded from `data/artifacts/lstm_autoencoder.pt` |
    | **4. Alerting** | Score > threshold (0.23) triggers alert with confidence level | Shown as red/green banners + anomaly event log |
    | **5. Fleet Monitoring** | Track all engines simultaneously, flag degradation trends | Fleet Overview tab with status cards |

    ### Model Details

    **LSTM Autoencoder** learns the "normal" pattern of healthy engine sensor readings.
    When an engine starts degrading, the reconstruction error increases because the
    model can't reproduce the abnormal patterns — this is the anomaly score.

    - **Input:** 30-cycle window × 101 features (sensors + rolling statistics)
    - **Architecture:** Encoder (LSTM → latent) → Decoder (latent → reconstructed window)
    - **Threshold:** """
        + str(THRESHOLD)
        + """ (optimized on validation set via F1 score search)
    - **Training F1:** 0.79 on held-out test set

    ### What Would Change in Production?

    1. **Streaming ingestion** — Kafka/Flink instead of batch windows
    2. **Model serving** — FastAPI behind a load balancer (included in `src/serving/`)
    3. **Database** — PostgreSQL for anomaly event history (included in `src/serving/db.py`)
    4. **Monitoring** — Prometheus metrics for model drift detection (included in `src/monitoring/`)
    5. **CI/CD** — Automated retraining when drift exceeds threshold
    6. **Multi-model** — Ensemble of LSTM-AE + Isolation Forest + PatchTST (all trained)
    """
    )
