# Dataset Download Instructions

## Supported Datasets

The Deepfake Forensic Detection System uses the following three benchmark datasets.

---

### 1. FakeAVCeleb (Local)

- **Paper**: [FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset](https://arxiv.org/abs/2108.05080)
- **Status**: ✅ **Already downloaded** at `c:\Users\Nitte\Desktop\NNM24AD071\FakeAVCeleb_v1.2`
- **Forgery types**: face-swap, lip-sync, both (audio+video)
- **Config key**: `path_config.fakeavceleb_root`

```python
from datasets import FakeAVCelebDataset

dataset = FakeAVCelebDataset(
    root_dir="c:/Users/Nitte/Desktop/NNM24AD071/FakeAVCeleb_v1.2",
    split="train",
    forgery_types=["lip-sync", "face-swap"],
)
```

---

### 2. FaceForensics++ (Kaggle)

- **Paper**: [FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/abs/1901.08971)
- **Source**: Kaggle (`sophatvathana/faceforensics`)
- **Manipulation types**: DeepFakes, Face2Face, FaceSwap, NeuralTextures
- **Compression levels**: c0 (raw), c23 (HQ), c40 (LQ)
- **Config key**: `path_config.faceforensics_root`

```python
from datasets import FaceForensicsDataset

dataset = FaceForensicsDataset(
    root_dir="c:/Users/Nitte/Desktop/NNM24AD071/FaceForensics++",
    split="train",
    compression="c23",
    manipulation_types=["Deepfakes", "Face2Face"],
)
```

---

### 3. LAV-DF (Kaggle)

- **Paper**: LAV-DF: A Large-scale Audio-Visual Deepfake Dataset
- **Source**: Kaggle (`bibek777/lavdf-localized-audio-visual-deepfake-dataset`)
- **Special**: Realistic audiovisual manipulation with temporal forgery segments
- **Config key**: `path_config.lavdf_root`

```python
from datasets import LAVDFDataset

dataset = LAVDFDataset(
    root_dir="c:/Users/Nitte/Desktop/NNM24AD071/LAV-DF",
    split="train",
)
# Includes boundary_tags when temporal annotations are available
```

---

## Downloading from Kaggle

FaceForensics++ and LAV-DF are available on Kaggle. Use the provided helper script:

### Setup Kaggle API Credentials

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings)
2. Scroll to the **API** section → click **Create New Token**
3. Save the downloaded `kaggle.json` to `C:\Users\<your-username>\.kaggle\kaggle.json`

### Download Datasets

```bash
# Install kaggle (if not already)
pip install kaggle

# Download both datasets at once
python download_kaggle_datasets.py

# Download only FaceForensics++
python download_kaggle_datasets.py --dataset ff++

# Download only LAV-DF
python download_kaggle_datasets.py --dataset lavdf

# Custom output directory
python download_kaggle_datasets.py --output-dir D:\datasets
```

---

## Configuration

Dataset paths are auto-configured in `config.py` with smart defaults.
To override, edit `config.py` or set them at runtime:

```python
from config import path_config
from pathlib import Path

path_config.faceforensics_root = Path("c:/Users/Nitte/Desktop/NNM24AD071/FaceForensics++")
path_config.fakeavceleb_root = Path("c:/Users/Nitte/Desktop/NNM24AD071/FakeAVCeleb_v1.2")
path_config.lavdf_root = Path("c:/Users/Nitte/Desktop/NNM24AD071/LAV-DF")
```
