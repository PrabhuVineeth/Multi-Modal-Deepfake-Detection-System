"""
Forensic report generator.

Produces structured JSON and formatted HTML reports from forensic
analysis results. HTML reports use Jinja2 templates with embedded
visualizations.
"""

import base64
import io
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from loguru import logger

from models.inference.postprocessing import ForensicReport
from utils.io_utils import save_json


class ReportGenerator:
    """
    Generates forensic analysis reports in JSON and HTML formats.
    """

    def __init__(
        self,
        template_dir: Optional[str] = None,
        output_dir: str = "output/reports",
    ):
        """
        Args:
            template_dir: Path to Jinja2 template directory.
            output_dir: Default output directory.
        """
        self.template_dir = Path(template_dir) if template_dir else (
            Path(__file__).parent / "templates"
        )
        self.output_dir = Path(output_dir)

    def generate(
        self,
        report: ForensicReport,
        key_frames: Optional[List[np.ndarray]] = None,
        output_dir: Optional[str] = None,
    ) -> dict:
        """
        Generate both JSON and HTML reports.

        Args:
            report: ForensicReport from post-processing.
            key_frames: Optional list of heatmap key frame images.
            output_dir: Override output directory.

        Returns:
            Dict with paths to generated files.
        """
        out_dir = Path(output_dir) if output_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # JSON report
        json_path = str(out_dir / "forensic_report.json")
        save_json(report.to_dict(), json_path)
        paths["json"] = json_path

        # HTML report
        html_path = str(out_dir / "forensic_report.html")
        self._generate_html(report, key_frames, html_path)
        paths["html"] = html_path

        logger.info(f"Reports generated in: {out_dir}")
        return paths

    def _generate_html(
        self,
        report: ForensicReport,
        key_frames: Optional[List[np.ndarray]],
        output_path: str,
    ) -> None:
        """Generate HTML report using Jinja2 template."""
        try:
            from jinja2 import Environment, FileSystemLoader
            env = Environment(loader=FileSystemLoader(str(self.template_dir)))
            env.filters['basename'] = lambda p: Path(p).name
            template = env.get_template("report.html")
        except Exception:
            # Fallback: generate HTML without Jinja2 template
            logger.warning("Jinja2 template not found, using inline HTML")
            html = self._generate_inline_html(report, key_frames)
            Path(output_path).write_text(html, encoding="utf-8")
            return

        # Encode key frames as base64 for embedding
        frame_data = []
        if key_frames:
            for i, frame in enumerate(key_frames):
                b64 = self._frame_to_base64(frame)
                frame_data.append({"index": i, "data": b64})

        html = template.render(
            report=report,
            report_dict=report.to_dict(),
            key_frames=frame_data,
            cls_color="#2ecc71" if report.classification == "REAL" else "#e74c3c",
        )

        Path(output_path).write_text(html, encoding="utf-8")
        logger.info(f"HTML report saved: {output_path}")

    def _generate_inline_html(
        self,
        report: ForensicReport,
        key_frames: Optional[List[np.ndarray]],
    ) -> str:
        """Generate a self-contained HTML report without external templates."""
        cls_color = "#2ecc71" if report.classification == "REAL" else "#e74c3c"
        cls_bg = "#1a4a2e" if report.classification == "REAL" else "#4a1a1a"

        # Encode key frames
        frames_html = ""
        if key_frames:
            for i, frame in enumerate(key_frames):
                b64 = self._frame_to_base64(frame)
                frames_html += f'''
                <div class="frame-card">
                    <img src="data:image/jpeg;base64,{b64}" alt="Key Frame {i}">
                </div>'''

        # Boundary timeline
        boundary_html = ""
        if report.boundaries:
            for b in report.boundaries:
                b_color = {"REAL": "#2ecc71", "FAKE": "#e74c3c", "BOUNDARY": "#f39c12"}.get(b["tag"], "#888")
                width = max(2, (b["end_time"] - b["start_time"]) / max(report.duration, 1) * 100)
                boundary_html += f'<div class="boundary-seg" style="width:{width}%;background:{b_color};" title="{b["tag"]}: {b["start_time"]:.1f}s - {b["end_time"]:.1f}s"></div>'

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forensic Analysis Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0d1117; color: #e6e6e6; padding: 2rem; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        .header {{ text-align: center; padding: 2rem; background: {cls_bg}; border-radius: 16px; margin-bottom: 2rem; border: 1px solid {cls_color}33; }}
        .badge {{ display: inline-block; padding: 0.5rem 2rem; background: {cls_color}; color: white; font-size: 1.8rem; font-weight: bold; border-radius: 8px; letter-spacing: 2px; }}
        .confidence {{ font-size: 3rem; font-weight: bold; color: {cls_color}; margin-top: 1rem; }}
        .scores {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 2rem 0; }}
        .score-card {{ background: #161b22; border-radius: 12px; padding: 1.5rem; text-align: center; border: 1px solid #30363d; }}
        .score-value {{ font-size: 2rem; font-weight: bold; margin: 0.5rem 0; }}
        .score-label {{ color: #8b949e; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }}
        .gauge {{ width: 80px; height: 80px; border-radius: 50%; margin: 0 auto 0.5rem; position: relative; }}
        .gauge::after {{ content: ''; position: absolute; inset: 8px; background: #161b22; border-radius: 50%; }}
        .section {{ background: #161b22; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid #30363d; }}
        .section h3 {{ color: #58a6ff; margin-bottom: 1rem; font-size: 1.1rem; }}
        .boundary-timeline {{ display: flex; height: 30px; border-radius: 6px; overflow: hidden; margin: 0.5rem 0; }}
        .boundary-seg {{ height: 100%; transition: opacity 0.2s; cursor: pointer; }}
        .boundary-seg:hover {{ opacity: 0.8; }}
        .frames-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 1rem; }}
        .frame-card {{ border-radius: 8px; overflow: hidden; border: 1px solid #30363d; }}
        .frame-card img {{ width: 100%; display: block; }}
        .meta {{ color: #8b949e; font-size: 0.85rem; }}
        .meta-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; }}
        .legend {{ display: flex; gap: 1.5rem; margin-top: 0.5rem; }}
        .legend-item {{ display: flex; align-items: center; gap: 0.4rem; font-size: 0.85rem; }}
        .legend-dot {{ width: 12px; height: 12px; border-radius: 3px; }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <div class="badge">{report.classification}</div>
        <div class="confidence">{report.confidence:.1f}%</div>
        <div class="meta">Confidence Score</div>
    </div>

    <div class="scores">
        <div class="score-card">
            <div class="score-label">Lip Sync</div>
            <div class="score-value" style="color: #3498db">{report.lip_sync_score:.1%}</div>
        </div>
        <div class="score-card">
            <div class="score-label">Identity</div>
            <div class="score-value" style="color: #9b59b6">{report.identity_score:.1%}</div>
        </div>
        <div class="score-card">
            <div class="score-label">Temporal</div>
            <div class="score-value" style="color: #e67e22">{report.temporal_score:.1%}</div>
        </div>
        <div class="score-card">
            <div class="score-label">AV Sync</div>
            <div class="score-value" style="color: #1abc9c">{report.av_sync_score:.1%}</div>
        </div>
    </div>

    <div class="section">
        <h3>Temporal Forgery Boundaries</h3>
        <div class="boundary-timeline">{boundary_html if boundary_html else '<div style="width:100%;background:#2ecc71;"></div>'}</div>
        <div class="legend">
            <div class="legend-item"><div class="legend-dot" style="background:#2ecc71"></div> Real</div>
            <div class="legend-item"><div class="legend-dot" style="background:#e74c3c"></div> Fake</div>
            <div class="legend-item"><div class="legend-dot" style="background:#f39c12"></div> Boundary</div>
        </div>
    </div>

    {f'<div class="section"><h3>Key Heatmap Frames</h3><div class="frames-grid">{frames_html}</div></div>' if frames_html else ''}

    <div class="section">
        <h3>Analysis Details</h3>
        <div class="meta-grid">
            <div class="meta">Video: {Path(report.video_path).name}</div>
            <div class="meta">Duration: {report.duration:.1f}s</div>
            <div class="meta">Frames: {report.num_frames}</div>
            <div class="meta">Processing: {report.processing_time:.1f}s</div>
            <div class="meta">Model: v{report.model_version}</div>
            <div class="meta">Time: {report.timestamp}</div>
        </div>
    </div>
</div>
</body>
</html>'''

    @staticmethod
    def _frame_to_base64(frame: np.ndarray) -> str:
        """Encode a BGR frame as base64 JPEG string."""
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode("utf-8")
