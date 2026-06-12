"""Model architecture modules for the Deepfake Forensic Detection System."""

from models.audio_encoder import AudioEncoder
from models.video_encoder import VideoEncoder, MouthEncoder
from models.cross_attention import (
    CrossModalFusion,
    MultimodalRepresentation,
    SpeechLipAttention,
    VoiceIdentityAttention,
    AVSyncAttention,
)
from models.forensic_analyzers import (
    ForensicAnalyzerBundle,
    LipSyncAnalyzer,
    IdentityAnalyzer,
    TemporalAnalyzer,
    AVSyncAnalyzer,
)
from models.evidence_aggregation import ForensicEvidenceAggregator
from models.mismatch_localization import JointMismatchLocalizer
from models.tfbd import TemporalForgeryBoundaryDetector, ForgeryBoundary
from models.heatmap_generator import CrossModalHeatmapGenerator
from models.calibration import (
    ConfidenceCalibrator,
    TemperatureScaler,
    PlattScaler,
)
from models.full_model import DeepfakeForensicModel, ForensicOutput

__all__ = [
    "AudioEncoder",
    "VideoEncoder",
    "MouthEncoder",
    "CrossModalFusion",
    "MultimodalRepresentation",
    "SpeechLipAttention",
    "VoiceIdentityAttention",
    "AVSyncAttention",
    "ForensicAnalyzerBundle",
    "LipSyncAnalyzer",
    "IdentityAnalyzer",
    "TemporalAnalyzer",
    "AVSyncAnalyzer",
    "ForensicEvidenceAggregator",
    "JointMismatchLocalizer",
    "TemporalForgeryBoundaryDetector",
    "ForgeryBoundary",
    "CrossModalHeatmapGenerator",
    "ConfidenceCalibrator",
    "TemperatureScaler",
    "PlattScaler",
    "DeepfakeForensicModel",
    "ForensicOutput",
]
