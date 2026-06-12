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

### 2. DFDC (Deepfake Detection Challenge)

- **Paper**: [The Deepfake Detection Challenge](https://arxiv.org/abs/2006.07397)
- **Access**: Download from https://www.kaggle.com/c/deepfake-detection-challenge
- **Size**: ~470 GB (full dataset)
- **Config key**: `path_config.dfdc_root`

```python
from datasets import DFDCDataset

dataset = DFDCDataset(
    root_dir="/path/to/dfdc",
    split="train",
    num_chunks=10,  # Load only first 10 chunks
)
```

---

### 3. Celeb-DF v2

- **Paper**: [Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics](https://arxiv.org/abs/1909.12962)
- **Access**: Request at https://github.com/yuezunli/celeb-deepfakeforensics
- **Config key**: `path_config.celebdf_root`

```python
from datasets import CelebDFDataset

dataset = CelebDFDataset(
    root_dir="/path/to/Celeb-DF-v2",
    split="test",
)
```

---

### 4. FakeAVCeleb

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

### 5. ForgeryNet

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
path_config.dfdc_root = Path("/path/to/dfdc")
path_config.celebdf_root = Path("/path/to/Celeb-DF-v2")
path_config.fakeavceleb_root = Path("/path/to/FakeAVCeleb")
path_config.forgerynet_root = Path("/path/to/ForgeryNet")
```
