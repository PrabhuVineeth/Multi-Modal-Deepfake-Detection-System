"""Dataset modules for the Deepfake Forensic Detection System."""

from datasets.base_dataset import BaseDeepfakeDataset, SampleMetadata
from datasets.faceforensics import FaceForensicsDataset
from datasets.fakeavceleb import FakeAVCelebDataset
from datasets.lavdf import LAVDFDataset
from datasets.forgerynet import ForgeryNetDataset

__all__ = [
    "BaseDeepfakeDataset",
    "SampleMetadata",
    "FaceForensicsDataset",
    "FakeAVCelebDataset",
    "LAVDFDataset",
    "ForgeryNetDataset",
]
