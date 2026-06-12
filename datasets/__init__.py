"""Dataset modules for the Deepfake Forensic Detection System."""

from datasets.base_dataset import BaseDeepfakeDataset, SampleMetadata
from datasets.faceforensics import FaceForensicsDataset
from datasets.dfdc import DFDCDataset
from datasets.celebdf import CelebDFDataset
from datasets.fakeavceleb import FakeAVCelebDataset
from datasets.forgerynet import ForgeryNetDataset

__all__ = [
    "BaseDeepfakeDataset",
    "SampleMetadata",
    "FaceForensicsDataset",
    "DFDCDataset",
    "CelebDFDataset",
    "FakeAVCelebDataset",
    "ForgeryNetDataset",
]
