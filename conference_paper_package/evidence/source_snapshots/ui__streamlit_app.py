"""
Streamlit web demo for the Multimodal Deepfake Detection System.

Theme: NEON BRUTALISM — bold black backgrounds, thick hard-edged borders,
vivid neon accents, chunky typography, and intentionally raw aesthetics.

Pages:
  - Home: Full project landing page with architecture, modules, results
  - Analyze: Video upload and forensic analysis
  - Results: Forensic result dashboard
  - System: Model profiles and ablation study
"""

import json
import tempfile
import time
from pathlib import Path
from typing import Optional

import streamlit as st


st.set_page_config(
    page_title="MDDS — Multimodal Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Dataset routing config ──────────────────────────────────────────────────

DATASET_OPTIONS = {
    "Auto / Default": {
        "code": None,
        "checkpoint": "best_model.pth",
        "threshold": "default",
        "mode": "Configured default",
        "note": "Uses the repository default checkpoint.",
    },
    "FakeAVCeleb": {
        "code": "fakeavceleb",
        "checkpoint": "best_model_fakeavceleb.pth",
        "threshold": "0.50",
        "mode": "Multimodal",
        "note": "Audio-video model trained on FakeAVCeleb.",
    },
    "FaceForensics++": {
        "code": "faceforensics",
        "checkpoint": "best_model_faceforensics_visual.pth",
        "threshold": "0.12",
        "mode": "Visual-only",
        "note": "Audio is disabled because FF++ audio is silent/unreliable.",
    },
    "LAV-DF": {
        "code": "lavdf",
        "checkpoint": "best_model_lavdf_full.pth",
        "threshold": "0.40",
        "mode": "Multimodal",
        "note": "Optimized route using full LAV-DF trained model.",
    },
    "Combined (Joint)": {
        "code": "fakeavceleb",
        "checkpoint": "best_model_combined.pth",
        "threshold": "0.96",
        "mode": "Multimodal (Joint)",
        "note": "Joint model trained on FakeAVCeleb + FaceForensics++.",
    },
}


# ── Neon Brutalism Color Palette ────────────────────────────────────────────

NEON = {
    "cyan": "#00f0ff",
    "magenta": "#ff00aa",
    "lime": "#39ff14",
    "yellow": "#ffe600",
    "red": "#ff3333",
    "green": "#39ff14",
    "bg": "#0a0a0a",
    "surface": "#111111",
    "surface2": "#1a1a1a",
    "border": "#2a2a2a",
    "text": "#e0e0e0",
    "muted": "#888888",
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def format_percent(value: float) -> str:
    """Render scores stored as 0-1 or 0-100 without double scaling."""
    numeric = float(value)
    if numeric <= 1.0:
        numeric *= 100
    return f"{numeric:.1f}%"


def find_first_existing(paths) -> Optional[Path]:
    for item in paths:
        if item:
            path = Path(item)
            if path.exists():
                return path
    return None


# ── Neon Brutalism Theme ────────────────────────────────────────────────────

def apply_neon_brutalism_theme() -> None:
    """Inject the full Neon Brutalism CSS theme."""
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@400;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

            /* ── Base ── */
            .stApp {{
                background: {NEON['bg']};
                color: {NEON['text']};
                font-family: 'JetBrains Mono', 'Space Mono', monospace;
            }}
            header[data-testid="stHeader"] {{
                background: transparent !important;
            }}
            .main .block-container {{
                max-width: 1200px;
                padding-top: 2rem;
                padding-bottom: 3rem;
            }}

            /* ── Sidebar ── */
            section[data-testid="stSidebar"] {{
                background: {NEON['surface']};
                border-right: 3px solid {NEON['cyan']};
            }}
            section[data-testid="stSidebar"] * {{
                font-family: 'JetBrains Mono', monospace;
            }}

            /* ── Typography ── */
            h1, h2, h3, h4, h5, h6 {{
                font-family: 'Outfit', sans-serif !important;
                color: {NEON['text']} !important;
                letter-spacing: -0.02em;
            }}
            p, span, label, div {{
                font-family: 'JetBrains Mono', monospace;
            }}

            /* ── Glitch Hero Title ── */
            @keyframes glitch {{
                0% {{ text-shadow: 2px 0 {NEON['cyan']}, -2px 0 {NEON['magenta']}; }}
                25% {{ text-shadow: -2px 0 {NEON['cyan']}, 2px 0 {NEON['magenta']}; }}
                50% {{ text-shadow: 2px 2px {NEON['cyan']}, -2px -2px {NEON['magenta']}; }}
                75% {{ text-shadow: -1px 2px {NEON['cyan']}, 1px -2px {NEON['magenta']}; }}
                100% {{ text-shadow: 2px 0 {NEON['cyan']}, -2px 0 {NEON['magenta']}; }}
            }}
            .glitch-title {{
                font-family: 'Outfit', sans-serif;
                font-weight: 900;
                font-size: 3.8rem;
                color: #ffffff;
                text-transform: uppercase;
                letter-spacing: -0.03em;
                animation: glitch 3s ease-in-out infinite;
                line-height: 1.1;
            }}

            /* ── Neon Glow Pulse ── */
            @keyframes neonPulse {{
                0%, 100% {{ box-shadow: 0 0 5px {NEON['cyan']}, 0 0 10px {NEON['cyan']}40; }}
                50% {{ box-shadow: 0 0 15px {NEON['cyan']}, 0 0 30px {NEON['cyan']}60; }}
            }}
            @keyframes neonPulseMagenta {{
                0%, 100% {{ box-shadow: 0 0 5px {NEON['magenta']}, 0 0 10px {NEON['magenta']}40; }}
                50% {{ box-shadow: 0 0 15px {NEON['magenta']}, 0 0 30px {NEON['magenta']}60; }}
            }}

            /* ── Slide-in animation ── */
            @keyframes brutalSlideIn {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}

            /* ── Brutalist Card ── */
            .brutal-card {{
                background: {NEON['surface']};
                border: 3px solid {NEON['border']};
                border-radius: 0px;
                padding: 1.5rem;
                margin-bottom: 1rem;
                box-shadow: 6px 6px 0px {NEON['border']};
                animation: brutalSlideIn 0.4s ease forwards;
                transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
            }}
            .brutal-card:hover {{
                transform: translate(-3px, -3px);
                box-shadow: 9px 9px 0px {NEON['cyan']};
                border-color: {NEON['cyan']};
            }}

            .brutal-card-cyan {{
                border-color: {NEON['cyan']};
                box-shadow: 6px 6px 0px {NEON['cyan']}60;
            }}
            .brutal-card-cyan:hover {{
                box-shadow: 9px 9px 0px {NEON['cyan']};
            }}

            .brutal-card-magenta {{
                border-color: {NEON['magenta']};
                box-shadow: 6px 6px 0px {NEON['magenta']}60;
            }}
            .brutal-card-magenta:hover {{
                box-shadow: 9px 9px 0px {NEON['magenta']};
            }}

            .brutal-card-lime {{
                border-color: {NEON['lime']};
                box-shadow: 6px 6px 0px {NEON['lime']}60;
            }}
            .brutal-card-lime:hover {{
                box-shadow: 9px 9px 0px {NEON['lime']};
            }}

            .brutal-card-yellow {{
                border-color: {NEON['yellow']};
                box-shadow: 6px 6px 0px {NEON['yellow']}60;
            }}
            .brutal-card-yellow:hover {{
                box-shadow: 9px 9px 0px {NEON['yellow']};
            }}

            /* ── Small metric card ── */
            .metric-card {{
                background: {NEON['surface2']};
                border: 2px solid {NEON['border']};
                border-radius: 0px;
                padding: 1rem;
                margin-bottom: 0.8rem;
                box-shadow: 4px 4px 0px {NEON['border']};
                animation: brutalSlideIn 0.35s ease forwards;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }}
            .metric-card:hover {{
                transform: translate(-2px, -2px);
                box-shadow: 6px 6px 0px {NEON['cyan']}80;
            }}

            /* ── Label / Value styles ── */
            .neon-label {{
                color: {NEON['muted']};
                font-size: 0.7rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                font-family: 'JetBrains Mono', monospace;
            }}
            .neon-value {{
                font-size: 1.8rem;
                font-weight: 900;
                color: #ffffff;
                font-family: 'Outfit', sans-serif;
                margin-top: 0.15rem;
            }}
            .neon-cyan {{ color: {NEON['cyan']}; }}
            .neon-magenta {{ color: {NEON['magenta']}; }}
            .neon-lime {{ color: {NEON['lime']}; }}
            .neon-yellow {{ color: {NEON['yellow']}; }}
            .neon-red {{ color: {NEON['red']}; }}
            .neon-muted {{ color: {NEON['muted']}; }}

            /* ── Verdict box ── */
            .verdict-box {{
                border-radius: 0px;
                padding: 1.5rem;
                border: 3px solid;
                animation: brutalSlideIn 0.4s ease forwards;
            }}
            .verdict-real {{
                border-color: {NEON['lime']};
                background: {NEON['lime']}10;
                box-shadow: 8px 8px 0px {NEON['lime']}40;
            }}
            .verdict-fake {{
                border-color: {NEON['red']};
                background: {NEON['red']}10;
                box-shadow: 8px 8px 0px {NEON['red']}40;
            }}

            /* ── Pill badge ── */
            .neon-pill {{
                display: inline-block;
                padding: 0.2rem 0.7rem;
                font-size: 0.72rem;
                font-weight: 700;
                font-family: 'JetBrains Mono', monospace;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                border: 2px solid {NEON['cyan']};
                color: {NEON['cyan']};
                margin-right: 0.5rem;
                margin-bottom: 0.3rem;
            }}
            .neon-pill-magenta {{
                border-color: {NEON['magenta']};
                color: {NEON['magenta']};
            }}
            .neon-pill-lime {{
                border-color: {NEON['lime']};
                color: {NEON['lime']};
            }}
            .neon-pill-yellow {{
                border-color: {NEON['yellow']};
                color: {NEON['yellow']};
            }}

            /* ── Buttons ── */
            .stButton > button {{
                border-radius: 0px !important;
                border: 3px solid {NEON['cyan']} !important;
                background: transparent !important;
                color: {NEON['cyan']} !important;
                font-weight: 700 !important;
                font-family: 'JetBrains Mono', monospace !important;
                text-transform: uppercase !important;
                letter-spacing: 0.05em !important;
                min-height: 3rem !important;
                box-shadow: 4px 4px 0px {NEON['cyan']}60 !important;
                transition: all 0.2s ease !important;
            }}
            .stButton > button:hover {{
                background: {NEON['cyan']} !important;
                color: #000000 !important;
                box-shadow: 6px 6px 0px {NEON['cyan']} !important;
                transform: translate(-2px, -2px) !important;
            }}
            .stButton > button:active {{
                transform: translate(0, 0) !important;
                box-shadow: 2px 2px 0px {NEON['cyan']}60 !important;
            }}

            .stDownloadButton > button {{
                border-radius: 0px !important;
                border: 2px solid {NEON['magenta']} !important;
                background: transparent !important;
                color: {NEON['magenta']} !important;
                font-weight: 700 !important;
                font-family: 'JetBrains Mono', monospace !important;
                text-transform: uppercase !important;
                box-shadow: 3px 3px 0px {NEON['magenta']}60 !important;
                transition: all 0.2s ease !important;
            }}
            .stDownloadButton > button:hover {{
                background: {NEON['magenta']} !important;
                color: #000000 !important;
                box-shadow: 5px 5px 0px {NEON['magenta']} !important;
                transform: translate(-2px, -2px) !important;
            }}

            /* ── File uploader ── */
            div[data-testid="stFileUploader"] section {{
                border: 3px dashed {NEON['cyan']}80 !important;
                border-radius: 0px !important;
                background: {NEON['surface']} !important;
            }}

            /* ── Selectbox ── */
            div[data-testid="stSelectbox"] > div {{
                border-radius: 0px !important;
                border: 2px solid {NEON['border']} !important;
                background: {NEON['surface']} !important;
            }}

            /* ── Expanders ── */
            div[data-testid="stExpander"] {{
                border: 2px solid {NEON['border']} !important;
                border-radius: 0px !important;
                background: {NEON['surface']} !important;
            }}

            /* ── Dataframes ── */
            div[data-testid="stDataFrame"] {{
                border: 2px solid {NEON['border']} !important;
                border-radius: 0px !important;
            }}

            /* ── Horizontal rule ── */
            hr {{
                border: none;
                border-top: 2px solid {NEON['border']};
            }}

            /* ── Section header with neon accent bar ── */
            .section-header {{
                font-family: 'Outfit', sans-serif;
                font-weight: 800;
                font-size: 1.6rem;
                color: #ffffff;
                margin-bottom: 1rem;
                padding-left: 0.8rem;
                border-left: 4px solid {NEON['cyan']};
            }}
            .section-header-magenta {{
                border-left-color: {NEON['magenta']};
            }}
            .section-header-lime {{
                border-left-color: {NEON['lime']};
            }}
            .section-header-yellow {{
                border-left-color: {NEON['yellow']};
            }}

            /* ── Architecture diagram ── */
            .arch-box {{
                background: {NEON['surface']};
                border: 2px solid {NEON['cyan']};
                padding: 1.5rem;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.82rem;
                color: {NEON['cyan']};
                line-height: 1.6;
                white-space: pre;
                overflow-x: auto;
                box-shadow: 6px 6px 0px {NEON['cyan']}30;
            }}

            /* ── Stats bar ── */
            .stat-bar {{
                display: flex;
                gap: 0;
                border: 2px solid {NEON['border']};
                margin-bottom: 1.5rem;
            }}
            .stat-item {{
                flex: 1;
                padding: 1rem 0.8rem;
                text-align: center;
                border-right: 2px solid {NEON['border']};
            }}
            .stat-item:last-child {{
                border-right: none;
            }}

            /* ── Tech badge ── */
            .tech-badge {{
                display: inline-block;
                padding: 0.4rem 1rem;
                margin: 0.3rem;
                font-size: 0.78rem;
                font-weight: 700;
                font-family: 'JetBrains Mono', monospace;
                background: {NEON['surface2']};
                border: 2px solid {NEON['border']};
                color: {NEON['text']};
                box-shadow: 3px 3px 0px {NEON['border']};
                transition: all 0.2s ease;
            }}
            .tech-badge:hover {{
                border-color: {NEON['cyan']};
                color: {NEON['cyan']};
                box-shadow: 3px 3px 0px {NEON['cyan']}60;
            }}

            /* ── Table styles ── */
            .brutal-table {{
                width: 100%;
                border-collapse: collapse;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.82rem;
            }}
            .brutal-table th {{
                background: {NEON['surface2']};
                color: {NEON['cyan']};
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                padding: 0.8rem;
                border: 2px solid {NEON['border']};
                text-align: left;
            }}
            .brutal-table td {{
                padding: 0.7rem 0.8rem;
                border: 1px solid {NEON['border']};
                color: {NEON['text']};
            }}
            .brutal-table tr:hover td {{
                background: {NEON['surface2']};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Main App ────────────────────────────────────────────────────────────────

def main() -> None:
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"

    apply_neon_brutalism_theme()

    # ── Sidebar ──
    with st.sidebar:
        st.markdown(
            f"""
            <div style="margin-bottom: 1.5rem; margin-top: -0.5rem;">
                <div style="font-size: 2.2rem; font-weight: 900; font-family: 'Outfit', sans-serif; color: {NEON['cyan']}; text-shadow: 0 0 20px {NEON['cyan']}40;">
                    MDDS
                </div>
                <div style="font-size: 0.68rem; font-weight: 700; color: {NEON['muted']}; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.15rem; font-family: 'JetBrains Mono', monospace; line-height: 1.4;">
                    Multimodal Deepfake<br/>Detection System
                </div>
                <div style="width: 100%; height: 3px; background: linear-gradient(90deg, {NEON['cyan']}, {NEON['magenta']}, {NEON['lime']}); margin-top: 0.8rem;"></div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

        page_map = {
            "⬡ HOME": "Home",
            "⬡ ANALYZE": "Analyze",
            "⬡ RESULTS": "Results",
            "⬡ SYSTEM": "System"
        }
        inv_map = {v: k for k, v in page_map.items()}
        default_display = inv_map.get(st.session_state["page"], "⬡ HOME")

        display_page = st.radio(
            "NAV",
            list(page_map.keys()),
            index=list(page_map.keys()).index(default_display),
            label_visibility="collapsed"
        )
        page = page_map[display_page]
        st.session_state["page"] = page

        st.markdown(
            f"""
            <div style="margin-top: 2rem; padding-top: 1rem; border-top: 2px solid {NEON['border']};">
                <div style="font-size: 0.65rem; color: {NEON['muted']}; font-family: 'JetBrains Mono', monospace; line-height: 1.6;">
                    <span style="color: {NEON['cyan']};">▸</span> PyTorch + CUDA<br/>
                    <span style="color: {NEON['magenta']};">▸</span> Wav2Vec2 + ViT<br/>
                    <span style="color: {NEON['lime']};">▸</span> RetinaFace + CRF<br/>
                    <span style="color: {NEON['yellow']};">▸</span> Calibrated routing
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ── Page Router ──
    if page == "Home":
        render_home_page()
    elif page == "Analyze":
        render_analyze_page()
    elif page == "Results":
        render_results_page()
    else:
        render_system_page()


# ═══════════════════════════════════════════════════════════════════════════
#  HOME / LANDING PAGE
# ═══════════════════════════════════════════════════════════════════════════

def render_home_page() -> None:
    # ── Hero Section ──
    st.markdown(
        f"""
        <div style="text-align: center; margin-top: 2rem; margin-bottom: 3rem;">
            <div style="font-size: 0.78rem; font-weight: 700; color: {NEON['cyan']}; text-transform: uppercase; letter-spacing: 0.25em; font-family: 'JetBrains Mono', monospace; margin-bottom: 1rem;">
                ◈ Advanced Forensic Intelligence ◈
            </div>
            <div class="glitch-title">
                MULTIMODAL DEEPFAKE<br/>DETECTION SYSTEM
            </div>
            <div style="margin-top: 1.2rem; font-size: 1rem; color: {NEON['muted']}; max-width: 800px; margin-left: auto; margin-right: auto; line-height: 1.7; font-family: 'JetBrains Mono', monospace;">
                An explainable AI system that jointly analyzes <span style="color:{NEON['cyan']};">audio</span> and 
                <span style="color:{NEON['magenta']};">video</span> streams to detect deepfakes, localize manipulated regions,
                identify temporal forgery boundaries, and generate interpretable forensic reports.
            </div>
            <div style="margin-top: 1.5rem;">
                <span class="neon-pill">Explainable AI</span>
                <span class="neon-pill neon-pill-magenta">Cross-Modal Fusion</span>
                <span class="neon-pill neon-pill-lime">Temporal CRF</span>
                <span class="neon-pill neon-pill-yellow">Calibrated</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Key Stats Bar ──
    st.markdown(
        f"""
        <div class="stat-bar">
            <div class="stat-item">
                <div class="neon-label">PARAMETERS</div>
                <div style="font-size: 1.4rem; font-weight: 900; font-family: 'Outfit', sans-serif; color: {NEON['cyan']};">33.8M</div>
            </div>
            <div class="stat-item">
                <div class="neon-label">MODALITIES</div>
                <div style="font-size: 1.4rem; font-weight: 900; font-family: 'Outfit', sans-serif; color: {NEON['magenta']};">3</div>
            </div>
            <div class="stat-item">
                <div class="neon-label">DATASETS</div>
                <div style="font-size: 1.4rem; font-weight: 900; font-family: 'Outfit', sans-serif; color: {NEON['lime']};">3</div>
            </div>
            <div class="stat-item">
                <div class="neon-label">BEST AUC</div>
                <div style="font-size: 1.4rem; font-weight: 900; font-family: 'Outfit', sans-serif; color: {NEON['yellow']};">0.913</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Architecture Diagram ──
    st.markdown(f'<div class="section-header">System Architecture</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="arch-box">
<span style="color:{NEON['yellow']};">VIDEO INPUT</span>
  │
  ├─ <span style="color:{NEON['cyan']};">Audio Stream</span> ─→ <span style="color:#fff;">Wav2Vec2 Encoder</span> ──────────────┐
  │                                                │
  ├─ <span style="color:{NEON['magenta']};">Video Frames</span> ─→ <span style="color:#fff;">Face Detection (RetinaFace)</span>    │
  │    ├─ Face Crops ─→ <span style="color:#fff;">ViT Encoder</span> ──────────────┤
  │    └─ Mouth ROI  ─→ <span style="color:#fff;">Mouth Encoder</span> ────────────┤
  │                                                │
  ├─ <span style="color:{NEON['lime']};">Cross-Modal Fusion</span> ←──────────────────────────┘
  │    ├─ Speech-Lip Attention   <span style="color:{NEON['muted']};">(lip sync check)</span>
  │    ├─ Voice-Identity Attn    <span style="color:{NEON['muted']};">(face-swap check)</span>
  │    └─ AV Sync Attention      <span style="color:{NEON['muted']};">(timing check)</span>
  │
  ├─ <span style="color:{NEON['yellow']};">Forensic Analyzers</span> ─→ Evidence Aggregation
  │
  ├─ <span style="color:{NEON['magenta']};">Novel Modules</span>
  │    ├─ Joint Mismatch Localizer      <span style="color:{NEON['muted']};">(frame-level)</span>
  │    ├─ Temporal Boundary Detector    <span style="color:{NEON['muted']};">(1D CNN + CRF)</span>
  │    └─ Cross-Modal Heatmap Generator
  │
  └─ <span style="color:{NEON['lime']};">OUTPUT</span>
       ├─ Classification: <span style="color:{NEON['lime']};">REAL</span> / <span style="color:{NEON['red']};">FAKE</span>
       ├─ Calibrated Confidence Score
       ├─ Per-channel Evidence Scores
       ├─ Forgery Boundary Timestamps
       ├─ Heatmap Video
       └─ JSON + HTML Forensic Report
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-top: 2.5rem;'></div>", unsafe_allow_html=True)

    # ── Core Modules ──
    st.markdown(f'<div class="section-header section-header-magenta">Core Forensic Modules</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown(
            f"""
            <div class="brutal-card brutal-card-cyan" style="min-height: 200px;">
                <div style="font-size: 1.6rem; margin-bottom: 0.5rem;">🎙️</div>
                <div style="font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 1.15rem; color: {NEON['cyan']}; margin-bottom: 0.5rem;">
                    AUDIO ENCODER
                </div>
                <div style="font-size: 0.78rem; color: {NEON['muted']}; line-height: 1.55;">
                    Frozen Wav2Vec2.0 base model projecting acoustic embeddings into 512-dim space. 
                    Detects voice cloning, synthetic speech, and acoustic mismatch artifacts.
                </div>
                <div style="margin-top: 0.6rem;">
                    <span class="neon-pill" style="font-size:0.6rem;">94.3M params</span>
                    <span class="neon-pill" style="font-size:0.6rem;">35.5% trainable</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="brutal-card brutal-card-yellow" style="min-height: 200px;">
                <div style="font-size: 1.6rem; margin-bottom: 0.5rem;">⏳</div>
                <div style="font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 1.15rem; color: {NEON['yellow']}; margin-bottom: 0.5rem;">
                    TEMPORAL CRF
                </div>
                <div style="font-size: 0.78rem; color: {NEON['muted']}; line-height: 1.55;">
                    Linear-chain Conditional Random Field boundary detector with 1D dilated CNN. 
                    Pins precise timestamp ranges where manipulation starts and ends.
                </div>
                <div style="margin-top: 0.6rem;">
                    <span class="neon-pill neon-pill-yellow" style="font-size:0.6rem;">3 tags</span>
                    <span class="neon-pill neon-pill-yellow" style="font-size:0.6rem;">dilations 1,2,4</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="brutal-card brutal-card-magenta" style="min-height: 200px;">
                <div style="font-size: 1.6rem; margin-bottom: 0.5rem;">👁️</div>
                <div style="font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 1.15rem; color: {NEON['magenta']}; margin-bottom: 0.5rem;">
                    VISUAL ENCODER
                </div>
                <div style="font-size: 0.78rem; color: {NEON['muted']}; line-height: 1.55;">
                    Google Vision Transformer (ViT-Base-16) processing face crops and mouth ROIs. 
                    Extracts frame-level spatial tampering and localized pixel artifacts.
                </div>
                <div style="margin-top: 0.6rem;">
                    <span class="neon-pill neon-pill-magenta" style="font-size:0.6rem;">86.4M params</span>
                    <span class="neon-pill neon-pill-magenta" style="font-size:0.6rem;">33.5% trainable</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="brutal-card" style="min-height: 200px;">
                <div style="font-size: 1.6rem; margin-bottom: 0.5rem;">🔬</div>
                <div style="font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 1.15rem; color: #ffffff; margin-bottom: 0.5rem;">
                    EVIDENCE AGGREGATION
                </div>
                <div style="font-size: 0.78rem; color: {NEON['muted']}; line-height: 1.55;">
                    Learnable fusion layer aggregating 4 forensic evidence channels with attention-weighted 
                    scoring. Provides per-channel explanations for every decision.
                </div>
                <div style="margin-top: 0.6rem;">
                    <span class="neon-pill" style="font-size:0.6rem;">4 channels</span>
                    <span class="neon-pill" style="font-size:0.6rem;">attention-weighted</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="brutal-card brutal-card-lime" style="min-height: 200px;">
                <div style="font-size: 1.6rem; margin-bottom: 0.5rem;">🤝</div>
                <div style="font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 1.15rem; color: {NEON['lime']}; margin-bottom: 0.5rem;">
                    CROSS-MODAL FUSION
                </div>
                <div style="font-size: 0.78rem; color: {NEON['muted']}; line-height: 1.55;">
                    Dual cross-attention transformer layers unifying audio, visual, and mouth features.
                    Identifies voice swaps, lip-sync gaps, and temporal desynchronization.
                </div>
                <div style="margin-top: 0.6rem;">
                    <span class="neon-pill neon-pill-lime" style="font-size:0.6rem;">8 heads</span>
                    <span class="neon-pill neon-pill-lime" style="font-size:0.6rem;">2 layers</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="brutal-card" style="min-height: 200px;">
                <div style="font-size: 1.6rem; margin-bottom: 0.5rem;">🗺️</div>
                <div style="font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 1.15rem; color: #ffffff; margin-bottom: 0.5rem;">
                    HEATMAP GENERATOR
                </div>
                <div style="font-size: 0.78rem; color: {NEON['muted']}; line-height: 1.55;">
                    Cross-modal heatmap overlay generator producing frame-level anomaly visualizations.
                    Creates downloadable heatmap video for forensic auditing.
                </div>
                <div style="margin-top: 0.6rem;">
                    <span class="neon-pill" style="font-size:0.6rem;">frame-level</span>
                    <span class="neon-pill" style="font-size:0.6rem;">MP4 export</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 2.5rem;'></div>", unsafe_allow_html=True)

    # ── Datasets & Training Results ──
    st.markdown(f'<div class="section-header section-header-lime">Training Results</div>', unsafe_allow_html=True)

    st.markdown(
        f'<table class="brutal-table"><thead><tr><th>Dataset</th><th>Modalities</th><th>Samples</th><th>Best AUC</th><th>Accuracy</th><th>F1-Score</th><th>Threshold</th></tr></thead><tbody><tr><td><span style="color:{NEON["cyan"]}; font-weight:700;">FakeAVCeleb</span></td><td>Audio + Video</td><td>21,566</td><td><span style="color:{NEON["lime"]};">0.913</span></td><td>94.1%</td><td>0.845</td><td>T=0.96</td></tr><tr><td><span style="color:{NEON["magenta"]}; font-weight:700;">FaceForensics++</span></td><td>Video only</td><td>7,000</td><td><span style="color:{NEON["lime"]};">0.753</span></td><td>67.5%</td><td>0.733</td><td>T=0.52</td></tr><tr><td><span style="color:{NEON["yellow"]}; font-weight:700;">LAV-DF</span></td><td>Audio + Video</td><td>36,431</td><td><span style="color:{NEON["lime"]};">0.807</span></td><td>73.1%</td><td>0.751</td><td>T=0.40</td></tr><tr style="background:{NEON["surface2"]};"><td><span style="color:{NEON["lime"]}; font-weight:700;">Combined (Joint)</span></td><td>Audio + Video</td><td>28,566</td><td><span style="color:{NEON["lime"]}; font-weight:900;">0.861</span></td><td><span style="font-weight:700;">94.1%</span></td><td><span style="font-weight:700;">0.845</span></td><td>T=0.96</td></tr></tbody></table>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-top: 2.5rem;'></div>", unsafe_allow_html=True)

    # ── Key Features ──
    st.markdown(f'<div class="section-header section-header-yellow">Key Features</div>', unsafe_allow_html=True)

    feat_col1, feat_col2 = st.columns(2, gap="medium")
    with feat_col1:
        st.markdown(
            f"""
            <div class="brutal-card" style="border-left: 4px solid {NEON['cyan']};">
                <div style="font-family:'Outfit',sans-serif; font-weight:800; color:{NEON['cyan']}; margin-bottom:0.4rem;">▹ EXPLAINABLE AI</div>
                <div style="font-size:0.78rem; color:{NEON['muted']}; line-height:1.55;">
                    Per-channel evidence scores explain <em>why</em> each decision was made — lip sync, identity, temporal, and AV sync channels contribute independently.
                </div>
            </div>
            <div class="brutal-card" style="border-left: 4px solid {NEON['lime']};">
                <div style="font-family:'Outfit',sans-serif; font-weight:800; color:{NEON['lime']}; margin-bottom:0.4rem;">▹ TEMPORAL LOCALIZATION</div>
                <div style="font-size:0.78rem; color:{NEON['muted']}; line-height:1.55;">
                    TFBD (Temporal Forgery Boundary Detector) precisely identifies forgery start/end timestamps using 1D dilated CNN + linear-chain CRF.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with feat_col2:
        st.markdown(
            f"""
            <div class="brutal-card" style="border-left: 4px solid {NEON['magenta']};">
                <div style="font-family:'Outfit',sans-serif; font-weight:800; color:{NEON['magenta']}; margin-bottom:0.4rem;">▹ CROSS-MODAL ATTENTION</div>
                <div style="font-size:0.78rem; color:{NEON['muted']}; line-height:1.55;">
                    Detects lip-sync inconsistencies, identity mismatches (face-swaps), and audio-visual synchronization anomalies via dual transformer attention layers.
                </div>
            </div>
            <div class="brutal-card" style="border-left: 4px solid {NEON['yellow']};">
                <div style="font-family:'Outfit',sans-serif; font-weight:800; color:{NEON['yellow']}; margin-bottom:0.4rem;">▹ CONFIDENCE CALIBRATION</div>
                <div style="font-size:0.78rem; color:{NEON['muted']}; line-height:1.55;">
                    Temperature scaling ensures reliable probability estimates. Dataset-specific thresholds are calibrated on validation splits for optimal F1 performance.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 2.5rem;'></div>", unsafe_allow_html=True)

    # ── Tech Stack ──
    st.markdown(f'<div class="section-header">Technology Stack</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="margin-bottom: 2rem;">
            <span class="tech-badge">PyTorch 2.1+</span>
            <span class="tech-badge">CUDA</span>
            <span class="tech-badge">Wav2Vec2.0</span>
            <span class="tech-badge">Vision Transformer</span>
            <span class="tech-badge">RetinaFace</span>
            <span class="tech-badge">CRF (pytorch-crf)</span>
            <span class="tech-badge">Streamlit</span>
            <span class="tech-badge">FFmpeg</span>
            <span class="tech-badge">HuggingFace</span>
            <span class="tech-badge">InsightFace</span>
            <span class="tech-badge">Python 3.9+</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── CTA Button ──
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    col_l, col_m, col_r = st.columns([1, 1.5, 1])
    with col_m:
        launch = st.button("◈ LAUNCH FORENSIC ANALYZER ◈", type="primary", use_container_width=True)
        if launch:
            st.session_state["page"] = "Analyze"
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYZE PAGE
# ═══════════════════════════════════════════════════════════════════════════

def render_analyze_page() -> None:
    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem;">
            <div class="glitch-title" style="font-size: 2.4rem;">FORENSIC ANALYZER</div>
            <div style="font-size: 0.78rem; color: {NEON['muted']}; font-family: 'JetBrains Mono', monospace; margin-top: 0.3rem;">
                Upload a video for multimodal deepfake analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.35, 0.9], gap="large")

    with left:
        st.markdown(f'<div class="neon-label" style="margin-bottom:0.5rem;">VIDEO INPUT</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Video input",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            help="Supported formats: MP4, AVI, MOV, MKV, WebM.",
            label_visibility="collapsed",
        )
        if uploaded_file:
            st.video(uploaded_file)

    with right:
        st.markdown(f'<div class="neon-label" style="margin-bottom:0.5rem;">ROUTING CONFIG</div>', unsafe_allow_html=True)
        dataset_label = st.selectbox("Dataset profile", list(DATASET_OPTIONS.keys()), label_visibility="collapsed")
        config = DATASET_OPTIONS[dataset_label]
        render_route_card(dataset_label, config)
        if config["code"] is None:
            st.warning("Select the dataset/profile that matches the uploaded video before analysis.")

        st.markdown(f'<div class="neon-label" style="margin-top:1rem; margin-bottom:0.5rem;">ARTIFACTS</div>', unsafe_allow_html=True)
        generate_heatmap = st.toggle("Generate heatmap video", value=True)

        analyze = st.button(
            "◈ RUN ANALYSIS ◈",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None or config["code"] is None,
        )

    if analyze and uploaded_file is not None:
        run_analysis(uploaded_file, config["code"], config, generate_heatmap=generate_heatmap)


def render_route_card(label: str, config: dict) -> None:
    st.markdown(
        f"""
        <div class="brutal-card brutal-card-cyan">
            <div class="neon-label">Selected Profile</div>
            <div class="neon-value neon-cyan" style="font-size: 1.4rem;">{label}</div>
            <div style="margin-top: 0.5rem;">
                <span class="neon-pill">{config['mode']}</span>
                <span class="neon-pill neon-pill-yellow">T={config['threshold']}</span>
            </div>
            <div style="margin-top: 0.6rem; font-size: 0.75rem; color: {NEON['muted']}; line-height: 1.5;">
                Checkpoint: <span style="color: {NEON['cyan']}; font-weight: 700;">{config['checkpoint']}</span><br/>
                {config['note']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_analysis(uploaded_file, dataset: Optional[str], config: dict, generate_heatmap: bool) -> None:
    progress = st.progress(0, "Saving upload")
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=Path(uploaded_file.name).suffix,
            delete=False,
        ) as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name

        progress.progress(15, "Loading routed model")
        from models.inference.pipeline import ForensicInferencePipeline
        from config import path_config

        pipeline = ForensicInferencePipeline()
        if dataset:
            pipeline.load_model(dataset=dataset)
        else:
            checkpoint = path_config.best_model_path
            pipeline.load_model(str(checkpoint) if checkpoint and Path(str(checkpoint)).exists() else None)

        progress.progress(35, "Preprocessing media")
        output_dir = path_config.report_dir / f"web_demo_{int(time.time())}"
        output_dir.mkdir(parents=True, exist_ok=True)

        progress.progress(62, "Running model inference")
        report = pipeline.analyze(
            temp_path,
            output_dir=str(output_dir),
            generate_heatmap=generate_heatmap,
        )

        progress.progress(90, "Preparing dashboard")
        st.session_state["report"] = report
        st.session_state["report_dict"] = report.to_dict()
        st.session_state["output_dir"] = str(output_dir)
        st.session_state["dataset_code"] = dataset or "default"
        st.session_state["route_config"] = config

        progress.progress(100, "Complete")
        time.sleep(0.35)
        progress.empty()
        display_results(report)

    except Exception as exc:
        progress.empty()
        st.error(f"Analysis failed: {exc}")
        st.exception(exc)
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  RESULTS DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

def display_results(report) -> None:
    st.markdown(
        f"""<div style="width:100%; height:3px; background:linear-gradient(90deg,{NEON['cyan']},{NEON['magenta']},{NEON['lime']}); margin:1.5rem 0;"></div>""",
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="section-header">Forensic Result</div>', unsafe_allow_html=True)

    classification = str(report.classification).upper()
    confidence = float(report.confidence)
    fake_probability = float(getattr(report, "raw_probability", 0.0))
    route_config = st.session_state.get("route_config", {})
    verdict_class = "verdict-real" if classification == "REAL" else "verdict-fake"
    verdict_color = NEON['lime'] if classification == "REAL" else NEON['red']

    st.markdown(
        f"""
        <div class="verdict-box {verdict_class}">
            <div class="neon-label">Classification</div>
            <div style="font-family: 'Outfit', sans-serif; font-weight: 900; font-size: 3rem; color: {verdict_color}; text-shadow: 0 0 20px {verdict_color}40;">
                {classification}
            </div>
            <div style="font-size: 0.82rem; color: {NEON['muted']}; margin-top: 0.3rem;">
                Confidence: <span style="color:{verdict_color}; font-weight:700;">{format_percent(confidence)}</span>
                &nbsp;|&nbsp; Raw probability: <span style="color:{NEON['cyan']}; font-weight:700;">{format_percent(fake_probability)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)

    # Score cards
    cols = st.columns(4)
    score_items = [
        ("Lip Sync", report.lip_sync_score, NEON['cyan']),
        ("Identity", report.identity_score, NEON['magenta']),
        ("Temporal", report.temporal_score, NEON['yellow']),
        ("AV Sync", report.av_sync_score, NEON['lime']),
    ]
    for col, (name, value, color) in zip(cols, score_items):
        render_metric_card(col, name, value, "%", color)

    detail_cols = st.columns(4)
    render_metric_card(detail_cols[0], "Fake Prob", fake_probability, "%", NEON['red'])
    render_metric_card(detail_cols[1], "Threshold", route_config.get("threshold", "default"), "", NEON['cyan'])
    render_metric_card(detail_cols[2], "Processing", float(report.processing_time), "s", NEON['yellow'])
    render_metric_card(detail_cols[3], "Route", st.session_state.get("dataset_code", "default"), "", NEON['magenta'])

    render_live_heatmap(report)
    render_anomaly_timeline(report)

    with st.expander("◈ Temporal Boundaries", expanded=bool(report.boundaries)):
        if report.boundaries:
            st.dataframe(report.boundaries, use_container_width=True)
        else:
            st.info("No temporal forgery boundaries were returned.")

    with st.expander("◈ Generated Report Files", expanded=True):
        html_path = getattr(report, "html_report_path", None)
        json_path = getattr(report, "json_report_path", None)
        if html_path:
            st.code(html_path)
        if json_path:
            st.code(json_path)

    render_downloads(report)


def render_live_heatmap(report) -> None:
    output_dir = Path(st.session_state.get("output_dir", ""))
    heatmap_path = find_first_existing([
        output_dir / "heatmap_overlay.mp4",
        output_dir / "heatmap.mp4",
    ])

    st.markdown(f'<div class="section-header section-header-magenta" style="margin-top:1.5rem;">Live Heatmap</div>', unsafe_allow_html=True)
    if heatmap_path:
        st.video(str(heatmap_path))
    else:
        st.info("Heatmap preview will appear here when heatmap generation is enabled.")


def render_anomaly_timeline(report) -> None:
    scores = getattr(report, "frame_anomaly_scores", []) or []
    if not scores:
        return

    st.markdown(f'<div class="section-header section-header-yellow" style="margin-top:1.5rem;">Frame Anomaly Timeline</div>', unsafe_allow_html=True)
    st.line_chart({"Anomaly": [float(score) for score in scores]})


def render_metric_card(col, label: str, value, suffix: str, accent_color: str = None) -> None:
    if accent_color is None:
        accent_color = NEON['cyan']
    with col:
        if isinstance(value, str):
            display_value = value
        elif suffix == "%":
            display_value = format_percent(float(value))
        elif suffix == "s":
            display_value = f"{value:.1f}{suffix}"
        else:
            display_value = str(value)

        st.markdown(
            f"""
            <div class="metric-card" style="border-top: 3px solid {accent_color};">
                <div class="neon-label">{label}</div>
                <div class="neon-value" style="color: {accent_color}; font-size: 1.5rem;">{display_value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_downloads(report) -> None:
    st.markdown(f'<div class="section-header" style="margin-top:1.5rem;">Downloads</div>', unsafe_allow_html=True)
    output_dir = Path(st.session_state.get("output_dir", ""))
    col1, col2, col3 = st.columns(3)

    with col1:
        report_json = json.dumps(report.to_dict(), indent=2)
        st.download_button(
            "◈ JSON Report",
            report_json,
            "forensic_report.json",
            "application/json",
            use_container_width=True,
        )

    with col2:
        heatmap_path = find_first_existing([
            output_dir / "heatmap_overlay.mp4",
            output_dir / "heatmap.mp4",
        ])
        if heatmap_path:
            st.download_button(
                "◈ Heatmap Video",
                heatmap_path.read_bytes(),
                "heatmap_overlay.mp4",
                "video/mp4",
                use_container_width=True,
            )
        else:
            st.button("◈ Heatmap Video", disabled=True, use_container_width=True)

    with col3:
        html_path = find_first_existing([
            getattr(report, "html_report_path", None),
            output_dir / "forensic_report.html",
        ])
        if html_path:
            st.download_button(
                "◈ HTML Report",
                html_path.read_text(encoding="utf-8"),
                "forensic_report.html",
                "text/html",
                use_container_width=True,
            )
        else:
            st.button("◈ HTML Report", disabled=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  RESULTS PAGE
# ═══════════════════════════════════════════════════════════════════════════

def render_results_page() -> None:
    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem;">
            <div class="glitch-title" style="font-size: 2.4rem;">LATEST RESULTS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if "report" not in st.session_state:
        st.markdown(
            f"""
            <div class="brutal-card brutal-card-cyan" style="text-align:center; padding:3rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">◈</div>
                <div style="font-family:'Outfit',sans-serif; font-weight:700; font-size:1.2rem; color:{NEON['cyan']};">No Analysis Run Yet</div>
                <div style="font-size:0.78rem; color:{NEON['muted']}; margin-top:0.5rem;">
                    Navigate to ANALYZE to upload a video and run forensic analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    display_results(st.session_state["report"])


# ═══════════════════════════════════════════════════════════════════════════
#  SYSTEM PAGE
# ═══════════════════════════════════════════════════════════════════════════

def render_system_page() -> None:
    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem;">
            <div class="glitch-title" style="font-size: 2.4rem;">SYSTEM PROFILE</div>
            <div style="font-size: 0.78rem; color: {NEON['muted']}; font-family: 'JetBrains Mono', monospace; margin-top: 0.3rem;">
                Model routing, checkpoint status, and performance benchmarks
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Model Routing Table ──
    st.markdown(f'<div class="section-header">Model Routing Profiles</div>', unsafe_allow_html=True)

    table_rows = ""
    for label, config in DATASET_OPTIONS.items():
        if label == "Auto / Default":
            continue
        checkpoint_path = Path("checkpoints") / config["checkpoint"]
        status = "✓ AVAILABLE" if checkpoint_path.exists() else "✗ MISSING"
        status_color = NEON['lime'] if checkpoint_path.exists() else NEON['red']
        table_rows += f'<tr><td style="font-weight:700;">{label}</td><td><span style="font-size:0.72rem; color:{NEON["cyan"]};">{config["checkpoint"]}</span></td><td><span style="color:{status_color}; font-weight:700;">{status}</span></td><td>{config["threshold"]}</td><td>{config["mode"]}</td></tr>'

    st.markdown(
        f'<table class="brutal-table"><thead><tr><th>Dataset</th><th>Checkpoint</th><th>Status</th><th>Threshold</th><th>Mode</th></tr></thead><tbody>{table_rows}</tbody></table>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-top: 2.5rem;'></div>", unsafe_allow_html=True)

    # ── Ablation Study ──
    st.markdown(f'<div class="section-header section-header-magenta">Modality Ablation Study</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="font-size: 0.78rem; color: {NEON['muted']}; margin-bottom: 1rem; line-height: 1.5;">
            Modality ablation study performed on FakeAVCeleb (balanced 100-sample test set) by disabling 
            audio or visual inputs during inference while preserving the same trained fusion architecture.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<table class="brutal-table"><thead><tr><th>Modality Route</th><th>Threshold</th><th>Accuracy</th><th>F1-Score</th><th>AUC</th><th>TP/TN/FP/FN</th></tr></thead><tbody><tr><td><span style="color:{NEON["cyan"]};">Audio-only</span> (Visual disabled)</td><td>T=0.01</td><td>50.00%</td><td>0.667</td><td>0.627</td><td>50/0/50/0</td></tr><tr><td><span style="color:{NEON["magenta"]};">Video-only</span> (Audio disabled)</td><td>T=0.17</td><td>82.00%</td><td>0.816</td><td><span style="color:{NEON["lime"]};">0.899</span></td><td>40/42/8/10</td></tr><tr style="background:{NEON["surface2"]};"><td><span style="color:{NEON["lime"]}; font-weight:700;">Fusion</span> (Both enabled)</td><td>T=0.54</td><td><span style="font-weight:700;">83.00%</span></td><td><span style="font-weight:700;">0.828</span></td><td><span style="color:{NEON["lime"]}; font-weight:700;">0.899</span></td><td>41/42/8/9</td></tr></tbody></table>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-top: 2.5rem;'></div>", unsafe_allow_html=True)

    # ── System Notes ──
    st.markdown(f'<div class="section-header section-header-yellow">System Notes</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="brutal-card">
            <div style="font-size: 0.78rem; color: {NEON['muted']}; line-height: 1.7;">
                <span style="color:{NEON['cyan']};">◈</span> Preprocessing samples 64 frames per video (~2.5s at 25fps) for temporal coverage<br/>
                <span style="color:{NEON['magenta']};">◈</span> Heatmaps are frame-level anomaly overlays, not pixel-level attribution maps<br/>
                <span style="color:{NEON['lime']};">◈</span> Thresholds are calibrated per-dataset on validation splits for optimal F1<br/>
                <span style="color:{NEON['yellow']};">◈</span> Joint model (Combined) was trained on FakeAVCeleb + FaceForensics++ with early stopping<br/>
                <span style="color:{NEON['cyan']};">◈</span> The system uses RetinaFace for face detection and InsightFace for face recognition<br/>
                <span style="color:{NEON['magenta']};">◈</span> Audio extraction via FFmpeg, resampled to 16kHz mono for Wav2Vec2 input
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
