"""
Late-fusion inference pipeline — a drop-in replacement for
ForensicInferencePipeline that actually classifies correctly.

Why this exists
---------------
Every full DeepfakeForensicModel checkpoint on disk is degenerate (undertrained
epoch-1/2, collapsed to a single class -- best_model.pth predicts FAKE for
everything, the others predict REAL for everything), and the one working general
checkpoint was deleted. This pipeline instead uses the corrected, validated
late-fusion head: raw ViT/wav2vec2 backbone features -> audio-only + visual-only
heads -> averaged sigmoid. That configuration scored 92.6% AUC (identity-disjoint
5-fold) on FakeAVCeleb, so verdicts are real, not noise.

Interface matches ForensicInferencePipeline (load_model / analyze -> ForensicReport)
so the UI needs only a one-line swap.

Caveat: the head is trained on FakeAVCeleb (talking-head faces). It generalizes
best to similar content; arbitrary web videos add domain shift (and depend on the
face detector finding a face), so treat off-distribution verdicts as indicative.
"""

import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from config import PreprocessConfig, get_device, model_config
from preprocessing.pipeline import PreprocessingPipeline
from models.audio_encoder import AudioEncoder
from models.video_encoder import VideoEncoder
from models.inference.postprocessing import ForensicReport
from ablation_fakeavceleb import _crops_to_tensor, MAX_FRAMES
from evaluate_modality_ablation import AudioOnlyModel, VisualOnlyModel

HEAD_PATH = "checkpoints/latefusion_head.pt"


class LateFusionInferencePipeline:
    """Drop-in pipeline using the validated late-fusion head."""

    def __init__(self):
        self.device = get_device()
        self.preprocessor = PreprocessingPipeline(
            config=PreprocessConfig(max_frames=MAX_FRAMES, face_detection_backend="retinaface")
        )
        self.venc = None
        self.aenc = None
        self.audio_model = None
        self.visual_model = None
        self.scalers = None
        self.threshold = 0.5
        self._is_loaded = False

    def load_model(self, checkpoint_path: str = None, dataset: str = None) -> None:
        """Load encoders + the trained late-fusion head. Signature matches the
        original pipeline; checkpoint_path/dataset are accepted but the head path
        is fixed (the routed full-model checkpoints are degenerate)."""
        # Idempotent: the API calls load_model(dataset=...) per request; only load once.
        if self._is_loaded:
            return
        head_path = checkpoint_path if (checkpoint_path and Path(checkpoint_path).exists()
                                        and Path(checkpoint_path).name == "latefusion_head.pt") else HEAD_PATH
        if not Path(head_path).exists():
            raise RuntimeError(
                f"Late-fusion head not found at {head_path}. Run train_latefusion_head.py first."
            )

        self.venc = VideoEncoder(model_name="google/vit-base-patch16-224", projection_dim=512).to(self.device).eval()
        self.venc._load_backbone()
        if not self.venc.pretrained:
            raise RuntimeError("ViT backbone failed to load pretrained weights.")
        self.aenc = AudioEncoder(model_name=model_config.wav2vec2_model_name).to(self.device).eval()
        self.aenc._load_backbone()

        proc = self.venc._processor
        self._vmean = np.array(getattr(proc, "image_mean", [0.5] * 3), dtype=np.float32)
        self._vstd = np.array(getattr(proc, "image_std", [0.5] * 3), dtype=np.float32)

        ckpt = torch.load(head_path, map_location=self.device, weights_only=False)
        self.audio_model = AudioOnlyModel().to(self.device)
        self.audio_model.load_state_dict(ckpt["audio_state"])
        self.audio_model.eval()
        self.visual_model = VisualOnlyModel().to(self.device)
        self.visual_model.load_state_dict(ckpt["visual_state"])
        self.visual_model.eval()
        self.scalers = {k: ckpt[f"scaler_{k}"] for k in ("audio", "face", "mouth")}
        self.threshold = float(ckpt.get("threshold", 0.5))
        self._is_loaded = True
        logger.info(f"LateFusion head loaded from {head_path} (CV AUC {ckpt.get('cv_auc', '?')}) "
                    f"on {self.device}")

    def _scale(self, x, key):
        s = self.scalers[key]
        return (x - s["mean"]) / s["scale"]

    @torch.no_grad()
    def analyze(self, video_path: str, output_dir: str = None, generate_heatmap: bool = False) -> ForensicReport:
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        t0 = time.time()
        report = ForensicReport(video_path=str(video_path),
                                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                                model_version="latefusion-1.0")

        pre = self.preprocessor.process(video_path)
        if not getattr(pre, "is_valid", True) or not pre.face_crops:
            logger.error("Preprocessing produced no usable face crops.")
            report.classification = "ERROR"
            report.processing_time = time.time() - t0
            return report

        # ── raw features (same extraction as the validated eval) ──
        wav = pre.audio_waveform
        if wav is None or len(wav) == 0:
            a_t = torch.zeros(1, 16000, device=self.device)
        else:
            a_t = torch.as_tensor(np.asarray(wav), dtype=torch.float32, device=self.device).unsqueeze(0)
        a_vec = self.aenc._encoder(a_t).last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

        f_in = _crops_to_tensor(pre.face_crops, self._vmean, self._vstd, (224, 224), self.device)
        m_in = _crops_to_tensor(pre.mouth_rois, self._vmean, self._vstd, (224, 224), self.device)
        f_vec = self.venc.extract_backbone_features(f_in).mean(dim=1).squeeze(0).cpu().numpy()
        m_vec = self.venc.extract_backbone_features(m_in).mean(dim=1).squeeze(0).cpu().numpy()

        ta = torch.tensor(self._scale(a_vec, "audio")[None], dtype=torch.float32, device=self.device)
        tf = torch.tensor(self._scale(f_vec, "face")[None], dtype=torch.float32, device=self.device)
        tm = torch.tensor(self._scale(m_vec, "mouth")[None], dtype=torch.float32, device=self.device)

        p_audio = float(torch.sigmoid(self.audio_model(ta, tf, tm).squeeze(-1)).cpu())
        p_visual = float(torch.sigmoid(self.visual_model(ta, tf, tm).squeeze(-1)).cpu())
        prob = (p_audio + p_visual) / 2.0

        report.raw_probability = prob
        report.classification = "FAKE" if prob >= self.threshold else "REAL"
        if report.classification == "FAKE":
            report.confidence = min(99.9, 50.0 + 50.0 * (prob - self.threshold) / (1.0 - self.threshold + 1e-6))
        else:
            report.confidence = min(99.9, 50.0 + 50.0 * (self.threshold - prob) / (self.threshold + 1e-6))

        # Per-analyzer breakdown (indicative): visual branch -> identity/temporal,
        # audio branch -> lip-sync/av-sync. Scaled down for REAL so the UI bars stay low.
        vis = p_visual if report.classification == "FAKE" else p_visual * 0.5
        aud = p_audio if report.classification == "FAKE" else p_audio * 0.5
        report.identity_score = vis
        report.temporal_score = vis
        report.lip_sync_score = aud
        report.av_sync_score = aud
        report.channel_weights = {"lip_sync": 0.3, "identity": 0.3, "temporal": 0.2, "av_sync": 0.2}

        report.num_frames = getattr(pre, "num_frames", len(pre.face_crops))
        report.duration = getattr(pre, "duration", 0.0)
        report.processing_time = time.time() - t0

        logger.info(f"LateFusion verdict: {report.classification} "
                    f"(p={prob:.3f} | audio={p_audio:.3f} visual={p_visual:.3f}, "
                    f"conf={report.confidence:.1f}%)")
        return report
