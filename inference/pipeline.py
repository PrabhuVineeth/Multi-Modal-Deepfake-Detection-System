"""
Forensic inference pipeline.

Orchestrates the full analysis flow: preprocessing → model inference
→ post-processing → report generation.
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger

from config import (
    InferenceConfig,
    ModelConfig,
    PreprocessConfig,
    get_device,
    inference_config,
    model_config,
    preprocess_config,
)
from inference.postprocessing import ForensicReport, PostProcessor
from models.heatmap_generator import CrossModalHeatmapGenerator
from preprocessing.pipeline import PreprocessingPipeline


class ForensicInferencePipeline:
    """
    End-to-end forensic analysis pipeline.

    Usage:
        pipeline = ForensicInferencePipeline()
        pipeline.load_model("path/to/checkpoint.pth")
        report = pipeline.analyze("path/to/video.mp4")
    """

    def __init__(
        self,
        model_cfg: Optional[ModelConfig] = None,
        preprocess_cfg: Optional[PreprocessConfig] = None,
        inference_cfg: Optional[InferenceConfig] = None,
    ):
        """
        Args:
            model_cfg: Model configuration.
            preprocess_cfg: Preprocessing configuration.
            inference_cfg: Inference configuration.
        """
        self.model_cfg = model_cfg or model_config
        self.preprocess_cfg = preprocess_cfg or preprocess_config
        self.inference_cfg = inference_cfg or inference_config

        # Resolve device
        self.device = get_device(self.inference_cfg.device)
        logger.info(f"Inference device: {self.device}")

        # Initialize components
        self.preprocessor = PreprocessingPipeline(
            config=self.preprocess_cfg,
        )
        self.post_processor = PostProcessor(
            threshold=self.inference_cfg.confidence_threshold,
        )
        self.heatmap_generator = CrossModalHeatmapGenerator(
            colormap=self.inference_cfg.heatmap_colormap,
            alpha=self.inference_cfg.heatmap_alpha,
            fps=self.inference_cfg.heatmap_fps,
        )

        self.model = None
        self._is_loaded = False

    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        """
        Load the forensic detection model.

        Args:
            checkpoint_path: Path to model checkpoint (.pth).
                If None, initializes a fresh model (for testing).
        """
        from models.full_model import DeepfakeForensicModel

        logger.info("Initializing DeepfakeForensicModel...")
        self.model = DeepfakeForensicModel(config=self.model_cfg)

        if checkpoint_path is not None:
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            from utils.io_utils import load_checkpoint
            load_checkpoint(
                checkpoint_path, self.model, device=str(self.device)
            )

        self.model.to(self.device)
        self.model.eval()
        self._is_loaded = True

        if self.inference_cfg.use_fp16 and self.device.type == "cuda":
            self.model.half()
            logger.info("Using FP16 inference")

        logger.info("Model loaded and ready for inference")

    def analyze(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        generate_heatmap: bool = True,
    ) -> ForensicReport:
        """
        Analyze a video for deepfake manipulation.

        Args:
            video_path: Path to the input video file.
            output_dir: Directory to save outputs (heatmap, report).
            generate_heatmap: Whether to generate heatmap video.

        Returns:
            ForensicReport with analysis results.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()
        logger.info(f"Starting forensic analysis: {video_path}")

        # ── Step 1: Preprocess ──
        preprocessed = self.preprocessor.process(video_path)
        if not preprocessed.is_valid:
            logger.error("Preprocessing produced invalid data")
            return ForensicReport(
                classification="ERROR",
                video_path=video_path,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )

        # ── Step 2: Prepare tensors ──
        tensors = self._prepare_tensors(preprocessed)

        # ── Step 3: Model inference ──
        with torch.no_grad():
            if self.inference_cfg.use_fp16 and self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    forensic_output = self.model(**tensors)
            else:
                forensic_output = self.model(**tensors)

        # ── Step 4: Post-process ──
        processing_time = time.time() - start_time
        report = self.post_processor.process(
            forensic_output,
            video_path=video_path,
            duration=preprocessed.duration,
            num_frames=preprocessed.num_frames,
            timestamps=preprocessed.timestamps,
            processing_time=processing_time,
        )

        # ── Step 5: Generate outputs ──
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save JSON report
            from utils.io_utils import save_json
            report_path = str(output_path / "forensic_report.json")
            save_json(report.to_dict(), report_path)
            logger.info(f"Report saved: {report_path}")

            # Generate heatmap video
            if generate_heatmap and report.frame_anomaly_scores:
                heatmap_path = str(output_path / "heatmap_overlay.mp4")
                self.heatmap_generator.generate_video(
                    preprocessed.frames,
                    report.frame_anomaly_scores,
                    heatmap_path,
                )

        logger.info(
            f"Analysis complete: {report.classification} "
            f"(confidence={report.confidence:.1f}%, "
            f"time={processing_time:.1f}s)"
        )
        return report

    def _prepare_tensors(self, preprocessed) -> dict:
        """
        Convert preprocessed data to model input tensors.

        Args:
            preprocessed: PreprocessedData from preprocessing pipeline.

        Returns:
            Dictionary of tensors ready for model.forward().
        """
        # Audio waveform: [1, num_samples]
        audio = torch.tensor(
            preprocessed.audio_waveform, dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        # Face frames: [1, T, C, H, W]
        faces = np.stack(preprocessed.face_crops)  # [T, H, W, C] (BGR)
        faces = faces[..., ::-1].copy()             # BGR → RGB
        faces = faces.transpose(0, 3, 1, 2)         # [T, C, H, W]
        faces = torch.tensor(faces, dtype=torch.float32) / 255.0
        faces = faces.unsqueeze(0).to(self.device)  # [1, T, C, H, W]

        # Mouth ROIs: [1, T, C, H, W]
        mouths = np.stack(preprocessed.mouth_rois)
        mouths = mouths[..., ::-1].copy()
        mouths = mouths.transpose(0, 3, 1, 2)
        mouths = torch.tensor(mouths, dtype=torch.float32) / 255.0
        mouths = mouths.unsqueeze(0).to(self.device)

        return {
            "audio_waveform": audio,
            "face_frames": faces,
            "mouth_rois": mouths,
        }
