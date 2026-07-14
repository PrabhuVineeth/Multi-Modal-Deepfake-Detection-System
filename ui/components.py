"""
Reusable Streamlit UI components for the Deepfake Forensic Detection System.

Theme: NEON BRUTALISM — hard edges, neon accents, monospace typography.

Components: score gauges, confidence bars, boundary timelines, and more.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional


# Neon Brutalism palette
NEON = {
    "cyan": "#00f0ff",
    "magenta": "#ff00aa",
    "lime": "#39ff14",
    "yellow": "#ffe600",
    "red": "#ff3333",
    "bg": "#0a0a0a",
    "surface": "#111111",
    "surface2": "#1a1a1a",
    "border": "#2a2a2a",
    "text": "#e0e0e0",
    "muted": "#888888",
}


def render_classification_badge(classification: str, confidence: float):
    """Render a large classification badge with confidence — neon brutalism style."""
    is_real = classification == "REAL"
    color = NEON["lime"] if is_real else NEON["red"]
    emoji = "✓" if is_real else "✗"

    st.markdown(f"""
    <div style="text-align: center; padding: 2rem; background: {NEON['surface']};
                border: 3px solid {color}; margin-bottom: 1.5rem;
                box-shadow: 8px 8px 0px {color}40;">
        <div style="font-size: 3rem; color: {color}; font-family: 'Outfit', sans-serif; font-weight: 900;">{emoji}</div>
        <div style="display: inline-block; padding: 0.5rem 2rem;
                    background: {color}; color: #000000; font-size: 2rem;
                    font-weight: 900; font-family: 'Outfit', sans-serif;
                    letter-spacing: 4px; margin: 0.5rem 0;
                    text-transform: uppercase;">
            {classification}
        </div>
        <div style="font-size: 3rem; font-weight: 900; color: {color};
                    font-family: 'Outfit', sans-serif;
                    margin-top: 0.5rem; text-shadow: 0 0 20px {color}40;">{confidence:.1f}%</div>
        <div style="color: {NEON['muted']}; text-transform: uppercase;
                    letter-spacing: 2px; font-size: 0.75rem;
                    font-family: 'JetBrains Mono', monospace;">
            Confidence Score
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_score_gauges(scores: Dict[str, float]):
    """Render four score gauges using Plotly — neon brutalism colors."""
    score_config = [
        ("Lip Sync", scores.get("lip_sync", 0), NEON["cyan"]),
        ("Identity", scores.get("identity", 0), NEON["magenta"]),
        ("Temporal", scores.get("temporal", 0), NEON["yellow"]),
        ("AV Sync", scores.get("av_sync", 0), NEON["lime"]),
    ]

    cols = st.columns(4)
    for col, (name, value, color) in zip(cols, score_config):
        with col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value * 100,
                number={"suffix": "%", "font": {"size": 24, "color": color, "family": "JetBrains Mono"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": NEON["border"]},
                    "bar": {"color": color, "thickness": 0.6},
                    "bgcolor": NEON["surface"],
                    "bordercolor": NEON["border"],
                    "borderwidth": 2,
                    "steps": [
                        {"range": [0, 30], "color": NEON["surface"]},
                        {"range": [30, 70], "color": NEON["surface2"]},
                        {"range": [70, 100], "color": NEON["surface"]},
                    ],
                    "threshold": {
                        "line": {"color": NEON["red"], "width": 2},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
            ))
            fig.update_layout(
                height=180,
                margin=dict(l=20, r=20, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": NEON["text"], "family": "JetBrains Mono"},
                title={"text": name, "font": {"size": 13, "color": NEON["muted"], "family": "JetBrains Mono"}},
            )
            st.plotly_chart(fig, use_container_width=True)


def render_boundary_timeline(boundaries: List[dict], duration: float):
    """Render the temporal forgery boundary timeline — neon brutalism style."""
    if not boundaries or duration <= 0:
        st.info("No temporal boundaries detected.")
        return

    fig = go.Figure()

    colors = {"REAL": NEON["lime"], "FAKE": NEON["red"], "BOUNDARY": NEON["yellow"]}

    for b in boundaries:
        color = colors.get(b.get("tag", ""), NEON["muted"])
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
            gridcolor=NEON["border"],
            tickfont=dict(color=NEON["muted"], family="JetBrains Mono"),
        ),
        yaxis=dict(visible=False),
        barmode="stack",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Legend
    st.markdown(f"""
    <div style="display: flex; gap: 1.5rem; justify-content: center; margin-top: -0.5rem; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;">
        <span style="color: {NEON['lime']};">■ Real</span>
        <span style="color: {NEON['red']};">■ Fake</span>
        <span style="color: {NEON['yellow']};">■ Boundary</span>
    </div>
    """, unsafe_allow_html=True)


def render_channel_weights(weights: Dict[str, float]):
    """Render evidence channel contribution weights — neon brutalism bars."""
    if not weights:
        return

    st.markdown(
        f"""<div style="font-family:'Outfit',sans-serif; font-weight:800; font-size:1.2rem; color:#ffffff; margin-bottom:0.8rem; padding-left:0.8rem; border-left:4px solid {NEON['cyan']};">
            Evidence Channel Weights
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div style="font-size:0.72rem; color:{NEON['muted']}; margin-bottom:0.8rem; font-family:'JetBrains Mono',monospace;">
            How much each evidence type contributed to the decision
        </div>""",
        unsafe_allow_html=True,
    )

    colors = {
        "lip_sync": NEON["cyan"],
        "identity": NEON["magenta"],
        "temporal": NEON["yellow"],
        "av_sync": NEON["lime"],
    }
    labels = {
        "lip_sync": "Lip Sync",
        "identity": "Identity",
        "temporal": "Temporal",
        "av_sync": "AV Sync",
    }

    for key, value in weights.items():
        label = labels.get(key, key)
        color = colors.get(key, NEON["muted"])
        pct = value * 100
        st.markdown(f"""
        <div style="margin-bottom: 0.6rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                <span style="font-size: 0.78rem; color: {NEON['text']}; font-family: 'JetBrains Mono', monospace;">{label}</span>
                <span style="font-size: 0.78rem; color: {color}; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{pct:.1f}%</span>
            </div>
            <div style="background: {NEON['surface']}; border: 2px solid {NEON['border']}; height: 10px;">
                <div style="width: {pct}%; background: {color}; height: 100%;
                            transition: width 0.5s;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_metadata(report_dict: dict):
    """Render analysis metadata — neon brutalism cards."""
    meta = report_dict.get("metadata", {})
    cols = st.columns(3)
    items = [
        ("Duration", f"{meta.get('duration', 0):.1f}s", NEON["cyan"]),
        ("Frames", str(meta.get("num_frames", 0)), NEON["magenta"]),
        ("Processing", f"{meta.get('processing_time', 0):.1f}s", NEON["yellow"]),
    ]
    for col, (label, value, color) in zip(cols, items):
        with col:
            st.markdown(f"""
            <div style="background:{NEON['surface2']}; border:2px solid {NEON['border']};
                        border-top:3px solid {color}; padding:1rem;
                        box-shadow:4px 4px 0px {NEON['border']};">
                <div style="font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:{NEON['muted']}; font-family:'JetBrains Mono',monospace;">{label}</div>
                <div style="font-size:1.5rem; font-weight:900; color:{color}; font-family:'Outfit',sans-serif; margin-top:0.15rem;">{value}</div>
            </div>
            """, unsafe_allow_html=True)
