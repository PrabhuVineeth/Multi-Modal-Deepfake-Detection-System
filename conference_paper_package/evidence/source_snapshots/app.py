"""
FastAPI backend for the Deepfake Forensic Detection System.

Endpoints:
  POST /analyze       — Full forensic analysis (upload video)
  POST /analyze/quick — Quick classification (classification + confidence only)
  GET  /report/{id}   — Retrieve a previously generated report
  GET  /heatmap/{id}  — Stream heatmap video
  GET  /health        — Health check
"""

import asyncio
import time
import uuid
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel

from config import get_device, inference_config, path_config

# ── App Setup ──
app = FastAPI(
    title="Deepfake Forensic Detection API",
    description="Multimodal deepfake detection with explainable forensic analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State ──
_pipeline = None
_reports = {}  # report_id → report data

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB


# ── Models ──
class AnalysisResponse(BaseModel):
    """Response for full analysis."""
    report_id: str
    classification: str
    confidence: float
    scores: dict
    boundaries: list
    has_forgery: bool
    processing_time: float
    json_report_path: Optional[str] = None
    html_report_path: Optional[str] = None


class QuickResponse(BaseModel):
    """Response for quick classification."""
    classification: str
    confidence: float
    processing_time: float


class HealthResponse(BaseModel):
    """Response for health check."""
    status: str
    device: str
    model_loaded: bool


# ── Startup ──
@app.on_event("startup")
async def startup():
    """Load the forensic model on startup."""
    global _pipeline
    logger.info("Starting Deepfake Forensic API...")

    try:
        from models.inference.pipeline import ForensicInferencePipeline
        _pipeline = ForensicInferencePipeline()

        # Check for checkpoint
        checkpoint = path_config.best_model_path
        if checkpoint and Path(checkpoint).exists():
            _pipeline.load_model(str(checkpoint))
        else:
            logger.warning(
                "No checkpoint found. Loading fresh model (untrained). "
                "Set path_config.best_model_path to use a trained model."
            )
            _pipeline.load_model(None)

        logger.info("API ready for requests")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        _pipeline = None


# ── Endpoints ──
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        device=str(get_device()),
        model_loaded=_pipeline is not None and _pipeline._is_loaded,
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    video: UploadFile = File(...),
    dataset: Optional[str] = Query(None, description="Dataset name for routing: 'fakeavceleb', 'faceforensics', or 'lavdf'")
):
    """
    Full forensic analysis of an uploaded video.

    Returns classification, scores, boundaries, and generates
    a full report with heatmap video.
    """
    if _pipeline is None:
        raise HTTPException(503, "Model not loaded")

    if dataset is not None:
        try:
            await asyncio.to_thread(_pipeline.load_model, dataset=dataset)
        except Exception as e:
            logger.error(f"Failed to load routed model for dataset '{dataset}': {e}")
            raise HTTPException(500, f"Model routing failed: {str(e)}")

    if not _pipeline._is_loaded:
        raise HTTPException(503, "Model not loaded")

    # Validate file
    _validate_upload(video)

    # Save upload
    report_id = str(uuid.uuid4())[:8]
    upload_dir = path_config.output_dir / "uploads" / report_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    video_path = upload_dir / video.filename

    async with aiofiles.open(str(video_path), "wb") as f:
        content = await video.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(413, "File too large (max 500MB)")
        await f.write(content)

    # Run analysis
    try:
        output_dir = str(path_config.report_dir / report_id)
        report = await asyncio.to_thread(
            _pipeline.analyze,
            str(video_path),
            output_dir=output_dir,
            generate_heatmap=True,
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")

    # Store report
    _reports[report_id] = report.to_dict()
    _reports[report_id]["output_dir"] = output_dir

    return AnalysisResponse(
        report_id=report_id,
        classification=report.classification,
        confidence=report.confidence,
        scores={
            "lip_sync": report.lip_sync_score,
            "identity": report.identity_score,
            "temporal": report.temporal_score,
            "av_sync": report.av_sync_score,
        },
        boundaries=report.boundaries,
        has_forgery=report.has_forgery_boundaries,
        processing_time=report.processing_time,
        json_report_path=report.json_report_path,
        html_report_path=report.html_report_path,
    )


@app.post("/analyze/quick", response_model=QuickResponse)
async def analyze_quick(
    video: UploadFile = File(...),
    dataset: Optional[str] = Query(None, description="Dataset name for routing: 'fakeavceleb', 'faceforensics', or 'lavdf'")
):
    """Quick classification — returns only verdict and confidence."""
    if _pipeline is None:
        raise HTTPException(503, "Model not loaded")

    if dataset is not None:
        try:
            await asyncio.to_thread(_pipeline.load_model, dataset=dataset)
        except Exception as e:
            logger.error(f"Failed to load routed model for dataset '{dataset}': {e}")
            raise HTTPException(500, f"Model routing failed: {str(e)}")

    if not _pipeline._is_loaded:
        raise HTTPException(503, "Model not loaded")

    _validate_upload(video)

    # Save to temp
    upload_dir = path_config.output_dir / "uploads" / "quick"
    upload_dir.mkdir(parents=True, exist_ok=True)
    video_path = upload_dir / video.filename

    async with aiofiles.open(str(video_path), "wb") as f:
        await f.write(await video.read())

    try:
        report = await asyncio.to_thread(
            _pipeline.analyze,
            str(video_path),
            generate_heatmap=False,
        )
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")
    finally:
        # Cleanup temp file
        video_path.unlink(missing_ok=True)

    return QuickResponse(
        classification=report.classification,
        confidence=report.confidence,
        processing_time=report.processing_time,
    )


@app.get("/report/{report_id}")
async def get_report(report_id: str):
    """Retrieve a previously generated forensic report."""
    if report_id not in _reports:
        raise HTTPException(404, "Report not found")
    return JSONResponse(_reports[report_id])


@app.get("/heatmap/{report_id}")
async def get_heatmap(report_id: str):
    """Stream the heatmap overlay video for a report."""
    if report_id not in _reports:
        raise HTTPException(404, "Report not found")

    output_dir = _reports[report_id].get("output_dir", "")
    heatmap_path = Path(output_dir) / "heatmap_overlay.mp4"

    if not heatmap_path.exists():
        raise HTTPException(404, "Heatmap video not available")

    return FileResponse(
        str(heatmap_path),
        media_type="video/mp4",
        filename=f"heatmap_{report_id}.mp4",
    )


def _validate_upload(video: UploadFile):
    """Validate an uploaded video file."""
    if not video.filename:
        raise HTTPException(400, "No filename provided")

    ext = Path(video.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported format: {ext}. Allowed: {ALLOWED_EXTENSIONS}",
        )


# ── Run ──
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
