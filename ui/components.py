"""
Reusable Streamlit UI components for the Deepfake Forensic Detection System.

Components: score gauges, confidence bars, boundary timelines, and more.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional


def render_classification_badge(classification: str, confidence: float):
    """Render a large classification badge with confidence."""
    color = "#2ecc71" if classification == "REAL" else "#e74c3c"
    bg = "#1a4a2e" if classification == "REAL" else "#4a1a1a"
    emoji = "✅" if classification == "REAL" else "🚨"

    st.markdown(f"""
    <div style="text-align: center; padding: 2rem; background: {bg};
                border-radius: 16px; border: 2px solid {color}40;
                margin-bottom: 1.5rem;">
        <div style="font-size: 3rem;">{emoji}</div>
        <div style="display: inline-block; padding: 0.5rem 2rem;
                    background: {color}; color: white; font-size: 2rem;
                    font-weight: 800; border-radius: 10px;
                    letter-spacing: 3px; margin: 0.5rem 0;">
            {classification}
        </div>
        <div style="font-size: 3rem; font-weight: 800; color: {color};
                    margin-top: 0.5rem;">{confidence:.1f}%</div>
        <div style="color: #8b949e; text-transform: uppercase;
                    letter-spacing: 1px; font-size: 0.85rem;">
            Confidence Score
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_score_gauges(scores: Dict[str, float]):
    """Render four score gauges using Plotly."""
    score_config = [
        ("Lip Sync", scores.get("lip_sync", 0), "#3498db"),
        ("Identity", scores.get("identity", 0), "#9b59b6"),
        ("Temporal", scores.get("temporal", 0), "#e67e22"),
        ("AV Sync", scores.get("av_sync", 0), "#1abc9c"),
    ]

    cols = st.columns(4)
    for col, (name, value, color) in zip(cols, score_config):
        with col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value * 100,
                number={"suffix": "%", "font": {"size": 24, "color": color}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#333"},
                    "bar": {"color": color, "thickness": 0.6},
                    "bgcolor": "#1e1e2e",
                    "bordercolor": "#333",
                    "steps": [
                        {"range": [0, 30], "color": "#162447"},
                        {"range": [30, 70], "color": "#1f4068"},
                        {"range": [70, 100], "color": "#1b1b2f"},
                    ],
                    "threshold": {
                        "line": {"color": "#e74c3c", "width": 2},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
            ))
            fig.update_layout(
                height=180,
                margin=dict(l=20, r=20, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e6e6e6"},
                title={"text": name, "font": {"size": 13, "color": "#8b949e"}},
            )
            st.plotly_chart(fig, use_container_width=True)


def render_boundary_timeline(boundaries: List[dict], duration: float):
    """Render the temporal forgery boundary timeline."""
    if not boundaries or duration <= 0:
        st.info("No temporal boundaries detected.")
        return

    fig = go.Figure()

    colors = {"REAL": "#2ecc71", "FAKE": "#e74c3c", "BOUNDARY": "#f39c12"}

    for b in boundaries:
        color = colors.get(b.get("tag", ""), "#888888")
        fig.add_trace(go.Bar(
            x=[b["end_time"] - b["start_time"]],
            y=["Timeline"],
            orientation="h",
            base=b["start_time"],
            marker_color=color,
            name=b["tag"],
            hovertemplate=(
                f"{b['tag']}<br>"
                f"Start: {b['start_time']:.1f}s<br>"
                f"End: {b['end_time']:.1f}s<br>"
                f"Duration: {b['end_time'] - b['start_time']:.1f}s"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        height=100,
        margin=dict(l=0, r=0, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Time (seconds)",
            range=[0, duration],
            showgrid=True,
            gridcolor="#333",
            tickfont=dict(color="#8b949e"),
        ),
        yaxis=dict(visible=False),
        barmode="stack",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Legend
    st.markdown("""
    <div style="display: flex; gap: 1.5rem; justify-content: center; margin-top: -0.5rem;">
        <span style="color: #2ecc71;">● Real</span>
        <span style="color: #e74c3c;">● Fake</span>
        <span style="color: #f39c12;">● Boundary</span>
    </div>
    """, unsafe_allow_html=True)


def render_channel_weights(weights: Dict[str, float]):
    """Render evidence channel contribution weights."""
    if not weights:
        return

    st.markdown("#### Evidence Channel Weights")
    st.caption("How much each evidence type contributed to the decision")

    colors = {
        "lip_sync": "#3498db",
        "identity": "#9b59b6",
        "temporal": "#e67e22",
        "av_sync": "#1abc9c",
    }
    labels = {
        "lip_sync": "Lip Sync",
        "identity": "Identity",
        "temporal": "Temporal",
        "av_sync": "AV Sync",
    }

    for key, value in weights.items():
        label = labels.get(key, key)
        color = colors.get(key, "#888")
        pct = value * 100
        st.markdown(f"""
        <div style="margin-bottom: 0.5rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                <span style="font-size: 0.85rem; color: #c9d1d9;">{label}</span>
                <span style="font-size: 0.85rem; color: {color};">{pct:.1f}%</span>
            </div>
            <div style="background: #21262d; border-radius: 4px; height: 8px;">
                <div style="width: {pct}%; background: {color}; height: 100%;
                            border-radius: 4px; transition: width 0.5s;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_metadata(report_dict: dict):
    """Render analysis metadata."""
    meta = report_dict.get("metadata", {})
    cols = st.columns(3)
    with cols[0]:
        st.metric("Duration", f"{meta.get('duration', 0):.1f}s")
    with cols[1]:
        st.metric("Frames", meta.get("num_frames", 0))
    with cols[2]:
        st.metric("Processing", f"{meta.get('processing_time', 0):.1f}s")
