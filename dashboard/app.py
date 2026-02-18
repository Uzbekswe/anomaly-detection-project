"""Streamlit dashboard for real-time anomaly detection monitoring.

Features:
    - Service health status indicator
    - Live sensor readings chart with anomaly markers
    - Anomaly score / reconstruction error time series
    - Recent anomaly events table
    - Key metrics summary (anomaly rate, avg score, avg latency)

Usage:
    streamlit run dashboard/app.py

Environment:
    STREAMLIT_API_URL              — FastAPI base URL (default: http://localhost:8000)
    STREAMLIT_REFRESH_INTERVAL_SEC — Auto-refresh interval (default: 5)
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

API_URL = os.environ.get("STREAMLIT_API_URL", "http://localhost:8000")
REFRESH_INTERVAL = int(os.environ.get("STREAMLIT_REFRESH_INTERVAL_SEC", "5"))


# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 Manufacturing Anomaly Detection")
st.caption("Real-time IoT sensor monitoring — NASA CMAPSS Turbofan Engine Data")


# ──────────────────────────────────────────────
# Session state initialization
# ──────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []

if "anomaly_events" not in st.session_state:
    st.session_state.anomaly_events = []


# ──────────────────────────────────────────────
# API helpers
# ──────────────────────────────────────────────


def check_health() -> dict | None:
    """Check API health status."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def send_detection_request(sensor_id: str, window: list[list[float]]) -> dict | None:
    """Send a single detection request to the API."""
    try:
        payload = {
            "sensor_id": sensor_id,
            "window": window,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        resp = requests.post(f"{API_URL}/detect", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        st.error(f"API request failed: {e}")
        return None


def generate_simulated_window(
    num_sensors: int = 14,
    window_size: int = 30,
    inject_anomaly: bool = False,
) -> list[list[float]]:
    """Generate a simulated sensor window for demo purposes.

    When the API is live, this simulates streaming sensor data.
    """
    base = np.random.normal(loc=0.5, scale=0.1, size=(window_size, num_sensors))

    if inject_anomaly:
        # Inject a spike in a few sensors to simulate degradation
        anomaly_sensors = np.random.choice(num_sensors, size=3, replace=False)
        anomaly_start = np.random.randint(window_size // 2, window_size - 5)
        for s in anomaly_sensors:
            base[anomaly_start:, s] += np.random.uniform(0.5, 1.5)

    return base.clip(0, 1).tolist()


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    sensor_id = st.text_input("Sensor ID", value="engine_001")
    num_sensors = st.number_input("Number of sensors", min_value=1, max_value=50, value=14)
    window_size = st.number_input("Window size", min_value=10, max_value=100, value=30)
    inject_anomaly = st.checkbox("Inject anomaly in next reading", value=False)
    auto_refresh = st.checkbox("Auto-refresh", value=False)

    st.divider()

    # Health check
    st.subheader("Service Status")
    health = check_health()
    if health:
        status = health.get("status", "unknown")
        if status == "healthy":
            st.success(f"Status: {status}")
        else:
            st.warning(f"Status: {status}")
        st.text(f"Model loaded: {health.get('model_loaded', False)}")
        st.text(f"Version: {health.get('model_version', 'N/A')}")
    else:
        st.error("API unreachable")
        st.caption(f"Trying: {API_URL}")

    st.divider()

    if st.button("Clear history"):
        st.session_state.history = []
        st.session_state.anomaly_events = []
        st.rerun()


# ──────────────────────────────────────────────
# Main: Send reading & update
# ──────────────────────────────────────────────

col_send, col_status = st.columns([1, 3])

with col_send:
    send_clicked = st.button("Send Reading", type="primary", use_container_width=True)

if send_clicked or (auto_refresh and health and health.get("model_loaded")):
    window = generate_simulated_window(
        num_sensors=num_sensors,
        window_size=window_size,
        inject_anomaly=inject_anomaly,
    )
    result = send_detection_request(sensor_id, window)

    if result:
        entry = {
            "timestamp": result["timestamp"],
            "anomaly_score": result["anomaly_score"],
            "is_anomaly": result["is_anomaly"],
            "confidence": result["confidence"],
            "model_version": result["model_version"],
            "processing_time_ms": result["processing_time_ms"],
            "sensor_id": result["sensor_id"],
        }
        st.session_state.history.append(entry)

        if result["is_anomaly"]:
            st.session_state.anomaly_events.append(entry)

        # Keep last 200 readings
        if len(st.session_state.history) > 200:
            st.session_state.history = st.session_state.history[-200:]
        if len(st.session_state.anomaly_events) > 50:
            st.session_state.anomaly_events = st.session_state.anomaly_events[-50:]

    with col_status:
        if result:
            if result["is_anomaly"]:
                st.error(
                    f"ANOMALY DETECTED — Score: {result['anomaly_score']:.4f} "
                    f"| Confidence: {result['confidence']:.2f}"
                )
            else:
                st.success(
                    f"Normal — Score: {result['anomaly_score']:.4f} "
                    f"| Confidence: {result['confidence']:.2f}"
                )


# ──────────────────────────────────────────────
# Metrics row
# ──────────────────────────────────────────────

if st.session_state.history:
    st.divider()

    m1, m2, m3, m4 = st.columns(4)

    scores = [h["anomaly_score"] for h in st.session_state.history]
    anomalies = [h for h in st.session_state.history if h["is_anomaly"]]
    latencies = [h["processing_time_ms"] for h in st.session_state.history]

    m1.metric("Total Readings", len(st.session_state.history))
    m2.metric("Anomalies Detected", len(anomalies))
    m3.metric(
        "Anomaly Rate",
        f"{100 * len(anomalies) / len(st.session_state.history):.1f}%",
    )
    m4.metric("Avg Latency", f"{np.mean(latencies):.1f}ms")


# ──────────────────────────────────────────────
# Charts
# ──────────────────────────────────────────────

if st.session_state.history:
    st.divider()

    chart_col1, chart_col2 = st.columns(2)

    timestamps = [h["timestamp"] for h in st.session_state.history]
    scores = [h["anomaly_score"] for h in st.session_state.history]
    is_anomaly = [h["is_anomaly"] for h in st.session_state.history]

    # ── Anomaly score time series ──
    with chart_col1:
        st.subheader("Anomaly Score Over Time")

        fig_score = go.Figure()

        # Score line
        fig_score.add_trace(go.Scatter(
            x=timestamps,
            y=scores,
            mode="lines+markers",
            name="Anomaly Score",
            line=dict(color="royalblue", width=2),
            marker=dict(size=4),
        ))

        # Anomaly markers
        anomaly_ts = [t for t, a in zip(timestamps, is_anomaly) if a]
        anomaly_sc = [s for s, a in zip(scores, is_anomaly) if a]
        if anomaly_ts:
            fig_score.add_trace(go.Scatter(
                x=anomaly_ts,
                y=anomaly_sc,
                mode="markers",
                name="Anomaly",
                marker=dict(color="red", size=10, symbol="x"),
            ))

        fig_score.update_layout(
            xaxis_title="Time",
            yaxis_title="Anomaly Score",
            height=400,
            showlegend=True,
            margin=dict(l=20, r=20, t=20, b=20),
        )

        st.plotly_chart(fig_score, use_container_width=True)

    # ── Confidence over time ──
    with chart_col2:
        st.subheader("Model Confidence Over Time")

        confidences = [h["confidence"] for h in st.session_state.history]

        fig_conf = go.Figure()
        fig_conf.add_trace(go.Scatter(
            x=timestamps,
            y=confidences,
            mode="lines+markers",
            name="Confidence",
            line=dict(color="green", width=2),
            marker=dict(size=4),
        ))

        fig_conf.update_layout(
            xaxis_title="Time",
            yaxis_title="Confidence",
            yaxis=dict(range=[0, 1.05]),
            height=400,
            showlegend=True,
            margin=dict(l=20, r=20, t=20, b=20),
        )

        st.plotly_chart(fig_conf, use_container_width=True)

    # ── Score distribution histogram ──
    st.subheader("Anomaly Score Distribution")

    normal_scores = [s for s, a in zip(scores, is_anomaly) if not a]
    anomaly_scores_list = [s for s, a in zip(scores, is_anomaly) if a]

    fig_dist = go.Figure()
    if normal_scores:
        fig_dist.add_trace(go.Histogram(
            x=normal_scores,
            name="Normal",
            marker_color="royalblue",
            opacity=0.7,
        ))
    if anomaly_scores_list:
        fig_dist.add_trace(go.Histogram(
            x=anomaly_scores_list,
            name="Anomaly",
            marker_color="red",
            opacity=0.7,
        ))

    fig_dist.update_layout(
        xaxis_title="Anomaly Score",
        yaxis_title="Count",
        barmode="overlay",
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
    )

    st.plotly_chart(fig_dist, use_container_width=True)


# ──────────────────────────────────────────────
# Recent anomaly events table
# ──────────────────────────────────────────────

if st.session_state.anomaly_events:
    st.divider()
    st.subheader("Recent Anomaly Events")

    # Show most recent first
    events_display = list(reversed(st.session_state.anomaly_events[-20:]))
    st.dataframe(
        events_display,
        column_config={
            "timestamp": st.column_config.TextColumn("Timestamp"),
            "sensor_id": st.column_config.TextColumn("Sensor ID"),
            "anomaly_score": st.column_config.NumberColumn("Score", format="%.4f"),
            "confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
            "processing_time_ms": st.column_config.NumberColumn("Latency (ms)", format="%.1f"),
            "model_version": st.column_config.TextColumn("Model"),
            "is_anomaly": st.column_config.CheckboxColumn("Anomaly"),
        },
        use_container_width=True,
        hide_index=True,
    )


# ──────────────────────────────────────────────
# Auto-refresh
# ──────────────────────────────────────────────

if auto_refresh:
    time.sleep(REFRESH_INTERVAL)
    st.rerun()
