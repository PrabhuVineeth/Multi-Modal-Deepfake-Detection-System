"""
Streamlit frontend for the Deepfake Forensic Detection System.

Pages:
  1. Upload — Drag-and-drop video upload
  2. Analysis — Full forensic results dashboard
  3. About — Architecture and methodology
"""

import json
import tempfile
import time
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Deepfake Forensic Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
    }
    .main .block-container {
        max-width: 1100px;
        padding: 2rem 1rem;
    }
    h1, h2, h3, h4 {
        color: #e6e6e6 !important;
    }
    .stMetricValue {
        font-size: 1.5rem !important;
    }
    .sidebar .sidebar-content {
        background-color: #161b22;
    }
    .upload-section {
        border: 2px dashed #30363d;
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        background: #161b22;
        margin: 2rem 0;
    }
    .stDownloadButton > button {
        background: #238636 !important;
        color: white !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main Streamlit app."""
    # Sidebar navigation
    st.sidebar.title("🔍 Forensic Analyzer")
    page = st.sidebar.radio(
        "Navigation",
        ["🎬 Upload & Analyze", "📊 Results", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Deepfake Forensic Detection System** v1.0\n\n"
        "Multimodal analysis using audio-visual "
        "cross-attention for explainable deepfake detection."
    )

    if page == "🎬 Upload & Analyze":
        render_upload_page()
    elif page == "📊 Results":
        render_results_page()
    elif page == "ℹ️ About":
        render_about_page()


def render_upload_page():
    """Upload and analyze a video."""
    st.title("🎬 Upload Video for Analysis")
    st.markdown(
        "Upload a video file to analyze for potential deepfake manipulation. "
        "Supported formats: MP4, AVI, MOV, MKV, WebM."
    )

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help="Maximum file size: 500MB",
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.video(uploaded_file)
        with col2:
            st.markdown("**File Info**")
            st.text(f"Name: {uploaded_file.name}")
            size_mb = uploaded_file.size / (1024 * 1024)
            st.text(f"Size: {size_mb:.1f} MB")

        if st.button("🔬 Analyze Video", type="primary", use_container_width=True):
            _run_analysis(uploaded_file)


def _run_analysis(uploaded_file):
    """Run the forensic analysis pipeline."""
    progress = st.progress(0, "Initializing...")

    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(
        suffix=Path(uploaded_file.name).suffix,
        delete=False,
    ) as tmp:
        tmp.write(uploaded_file.getvalue())
        video_path = tmp.name

    try:
        # Initialize pipeline
        progress.progress(10, "Loading model...")

        from inference.pipeline import ForensicInferencePipeline
        pipeline = ForensicInferencePipeline()

        # Check for checkpoint
        from config import path_config
        checkpoint = path_config.best_model_path
        if checkpoint and Path(str(checkpoint)).exists():
            pipeline.load_model(str(checkpoint))
        else:
            st.warning(
                "⚠️ No trained model checkpoint found. "
                "Using untrained model — results will be random. "
                "Train a model first or set `path_config.best_model_path`."
            )
            pipeline.load_model(None)

        progress.progress(30, "Preprocessing video...")
        time.sleep(0.5)

        progress.progress(50, "Running forensic analysis...")
        output_dir = tempfile.mkdtemp()
        report = pipeline.analyze(
            video_path,
            output_dir=output_dir,
            generate_heatmap=True,
        )

        progress.progress(90, "Generating report...")
        time.sleep(0.3)

        # Store results in session state
        st.session_state["report"] = report
        st.session_state["report_dict"] = report.to_dict()
        st.session_state["output_dir"] = output_dir

        progress.progress(100, "Analysis complete!")
        time.sleep(0.5)
        progress.empty()

        # Display results inline
        _display_results(report)

    except Exception as e:
        progress.empty()
        st.error(f"❌ Analysis failed: {str(e)}")
        st.exception(e)

    finally:
        # Cleanup temp video
        Path(video_path).unlink(missing_ok=True)


def _display_results(report):
    """Display forensic analysis results."""
    from ui.components import (
        render_classification_badge,
        render_score_gauges,
        render_boundary_timeline,
        render_channel_weights,
        render_metadata,
    )

    st.markdown("---")
    st.title("📊 Forensic Analysis Results")

    # Classification badge
    render_classification_badge(report.classification, report.confidence)

    # Score gauges
    st.markdown("### Evidence Scores")
    render_score_gauges({
        "lip_sync": report.lip_sync_score,
        "identity": report.identity_score,
        "temporal": report.temporal_score,
        "av_sync": report.av_sync_score,
    })

    # Boundary timeline
    if report.boundaries:
        st.markdown("### Temporal Forgery Boundaries")
        render_boundary_timeline(report.boundaries, report.duration)

    # Channel weights
    if report.channel_weights:
        render_channel_weights(report.channel_weights)

    # Metadata
    st.markdown("### Analysis Details")
    render_metadata(report.to_dict())

    # Downloads
    st.markdown("### Downloads")
    col1, col2, col3 = st.columns(3)
    with col1:
        report_json = json.dumps(report.to_dict(), indent=2)
        st.download_button(
            "📄 JSON Report",
            report_json,
            "forensic_report.json",
            "application/json",
        )
    with col2:
        output_dir = st.session_state.get("output_dir", "")
        heatmap_path = Path(output_dir) / "heatmap_overlay.mp4"
        if heatmap_path.exists():
            with open(heatmap_path, "rb") as f:
                st.download_button(
                    "🎥 Heatmap Video",
                    f.read(),
                    "heatmap_overlay.mp4",
                    "video/mp4",
                )
        else:
            st.button("🎥 Heatmap Video", disabled=True, help="Not generated")
    with col3:
        html_path = Path(output_dir) / "forensic_report.html" if output_dir else None
        if html_path and html_path.exists():
            with open(html_path, "r") as f:
                st.download_button(
                    "🌐 HTML Report",
                    f.read(),
                    "forensic_report.html",
                    "text/html",
                )


def render_results_page():
    """Display stored results from the last analysis."""
    st.title("📊 Analysis Results")

    if "report" not in st.session_state:
        st.info(
            "No analysis results yet. "
            "Go to **Upload & Analyze** to analyze a video."
        )
        return

    _display_results(st.session_state["report"])


def render_about_page():
    """Display system architecture and methodology."""
    st.title("ℹ️ About the System")

    st.markdown("""
    ## Multimodal Deepfake Forensic Detection System

    An explainable multimodal deepfake detection system that analyzes both
    audio and video streams to detect forged media, localize manipulated
    regions, identify temporal forgery boundaries, and generate
    interpretable forensic reports.

    ### Architecture

    ```
    Video Input
      ├─ Audio → Wav2Vec2 Encoder ──────────┐
      │                                      │
      ├─ Frames → Face Detection             │
      │    ├─ Face Crops → ViT Encoder ──────┤
      │    └─ Mouth ROI → Mouth Encoder ─────┤
      │                                      │
      ├─ Cross-Modal Fusion ←────────────────┘
      │    ├─ Speech-Lip Attention
      │    ├─ Voice-Identity Attention
      │    └─ AV Sync Attention
      │
      ├─ Forensic Analyzers
      │    ├─ Lip Sync Analyzer
      │    ├─ Identity Analyzer
      │    ├─ Temporal Analyzer
      │    └─ AV Sync Analyzer
      │
      ├─ Evidence Aggregation Engine
      │    └─ Attention-weighted channel fusion
      │
      ├─ Novel Modules
      │    ├─ Joint Mismatch Localizer
      │    ├─ Temporal Forgery Boundary Detector (1D CNN + CRF)
      │    └─ Cross-Modal Heatmap Generator
      │
      └─ Output
           ├─ Classification: REAL / FAKE
           ├─ Confidence Score (calibrated)
           ├─ Forgery Boundaries
           ├─ Heatmap Video
           └─ Forensic Report
    ```

    ### Key Features

    - **Multimodal Analysis**: Examines both audio and visual streams
    - **Explainable AI**: Per-channel evidence scores explain the decision
    - **Temporal Localization**: TFBD identifies exact forgery start/end frames
    - **Cross-Modal Attention**: Detects lip-sync, identity, and AV sync inconsistencies
    - **Confidence Calibration**: Temperature scaling ensures reliable probability estimates

    ### Supported Datasets

    | Dataset | Type | Modalities |
    |---------|------|-----------|
    | FaceForensics++ | Face manipulation | Video |
    | DFDC | Various deepfakes | Audio + Video |
    | Celeb-DF v2 | Celebrity deepfakes | Video |
    | FakeAVCeleb | Audio-visual forgery | Audio + Video |
    | ForgeryNet | Temporal boundaries | Video + Labels |

    ### Technology Stack

    - **Deep Learning**: PyTorch, HuggingFace Transformers
    - **Audio**: Wav2Vec2 (Facebook)
    - **Visual**: Vision Transformer (Google ViT)
    - **Face Detection**: InsightFace / RetinaFace
    - **Sequence Labeling**: pytorch-crf
    - **API**: FastAPI + Uvicorn
    - **UI**: Streamlit + Plotly
    """)


if __name__ == "__main__":
    main()
