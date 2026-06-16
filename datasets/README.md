# Dataset Download Instructions

## Supported Datasets

The Deepfake Forensic Detection System supports the following benchmark datasets.
Each requires separate download and academic access agreements.

---

### 1. FaceForensics++ (FF++)

- **Paper**: [FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/abs/1901.08971)
- **Access**: Request access at https://github.com/ondyari/FaceForensics
- **Manipulation types**: DeepFakes, Face2Face, FaceSwap, NeuralTextures
- **Compression levels**: c0 (raw), c23 (HQ), c40 (LQ)
- **Config key**: `path_config.faceforensics_root`

```python
from datasets import FaceForensicsDataset

dataset = FaceForensicsDataset(
    root_dir="/path/to/FaceForensics++",
    split="train",
    compression="c23",
    manipulation_types=["Deepfakes", "Face2Face"],
)
```

---

### 2. FakeAVCeleb

- **Paper**: [FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset](https://arxiv.org/abs/2108.05080)
- **Access**: Request at https://github.com/DASH-Lab/FakeAVCeleb
- **Forgery types**: face-swap, lip-sync, both (audio+video)
- **Config key**: `path_config.fakeavceleb_root`

```python
from datasets import FakeAVCelebDataset

dataset = FakeAVCelebDataset(
    root_dir="/path/to/FakeAVCeleb",
    split="train",
    forgery_types=["lip-sync", "face-swap"],
)
```

---

### 3. LAV-DF

- **Paper**: LAV-DF: A Large-scale Audio-Visual Deepfake Dataset
- **Access**: Request from the official LAV-DF project release
- **Special**: Realistic audiovisual manipulation with temporal forgery segments
- **Config key**: `path_config.lavdf_root`

```python
from datasets import LAVDFDataset

dataset = LAVDFDataset(
    root_dir="/path/to/LAV-DF",
    split="train",
)
# Includes boundary_tags when temporal annotations are available
```

---

### 4. ForgeryNet

- **Paper**: [ForgeryNet: A Versatile Benchmark for Comprehensive Forgery Analysis](https://arxiv.org/abs/2103.05630)
- **Access**: Request at https://github.com/yinanhe/ForgeryNet
- **Special**: Provides per-frame temporal boundary labels for TFBD training
- **Config key**: `path_config.forgerynet_root`

```python
from datasets import ForgeryNetDataset

dataset = ForgeryNetDataset(
    root_dir="/path/to/ForgeryNet",
    split="train",
)
# Includes boundary_tags in each sample for TFBD training
```

---

## Configuration

Set dataset paths in `config.py`:

```python
from config import path_config

path_config.faceforensics_root = Path("/path/to/FaceForensics++")
path_config.fakeavceleb_root = Path("/path/to/FakeAVCeleb")
path_config.lavdf_root = Path("/path/to/LAV-DF")
path_config.forgerynet_root = Path("/path/to/ForgeryNet")
```
