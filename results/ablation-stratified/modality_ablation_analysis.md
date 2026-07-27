# Modality Ablation Analysis Report

We evaluated the performance of three classifier heads trained under the identical 5-fold cross-validation protocol using the pre-extracted, cached features from the RetinaFace + ViT-16 run. This analysis isolates the information content in the audio, visual, and fused channels to understand how the multimodal fusion operates.

## Modality Benchmark Results

### 1. Overall Metrics
| Configuration | Mean AUC% | Mean EER% | Mean F1% |
| :--- | :---: | :---: | :---: |
| **Audio-Only** | 69.97% ± 1.41% | 35.00% ± 1.69% | 65.71% ± 1.95% |
| **Visual-Only** | **84.15% ± 2.46%** | **23.81% ± 3.05%** | **75.27% ± 3.10%** |
| **Full Fusion** | 81.11% ± 4.99% | 25.90% ± 4.82% | 72.44% ± 5.41% |

### 2. Per-Manipulation-Type AUC Breakdown (Reals vs One Fake Type)
| Configuration | Face-Swap (Video-only) | Lip-Sync (Audio-only) | Both (AV) |
| :--- | :---: | :---: | :---: |
| **Audio-Only** | 47.54% | **77.99%** | 84.32% |
| **Visual-Only** | **89.23%** | 65.93% | **97.59%** |
| **Full Fusion** | 86.03% | 61.54% | 94.26% |

---

## Critical Findings & Scientific Insights

### 1. The Drowning Modality Phenomenon
The most important scientific finding is the performance on **Lip-Sync (Audio-only) deepfakes**:
* **Audio-Only Model:** **77.99% AUC**
* **Visual-Only Model:** **65.93% AUC**
* **Full Fusion Model:** **61.54% AUC**

**Audio-Only beats Full Fusion on Lip-Sync by 16.45% AUC.** 

This is a classic case of **modality drowning**:
- Visual deepfakes (Face-Swap and Both) are easier for the visual backbones to detect (yielding 89.23% and 97.59% AUC).
- During joint training, the optimizer quickly minimizes loss by relying on the highly dominant visual features.
- Consequently, the fusion classifier learns to ignore the audio channel. When presented with a `lip-sync` fake (which has a real face but synthesized audio), the dominant visual branch signals that the face is real, completely overriding the audio branch's detection signal. The final fused prediction is drowned out, dropping AUC to **61.54%** (worse than both single-modality classifiers!).

### 2. The Audio Features Do Work
The **77.99% AUC** achieved by the Audio-Only model on Lip-Sync (and **84.32% AUC** on Both) proves that the Wav2Vec2 features do carry strong, discriminative forgery information. The feature extraction pipeline is mathematically sound.

### 3. Visual Artifacts in Lip-Sync
The Visual-Only model scores **65.93% AUC** on Lip-Sync. Since the video source is real, this detection is driven by visual artifacts in the modified mouth region (introduced by Wav2Lip or similar lip-generation tools used to sync the face with the fake audio). 

---

## Recommended Architectural Fixes

To prevent modality drowning and allow the system to exceed single-modality performance, the following fusion modifications are recommended:
1. **Modality Gating (Gated Multimodal Unit - GMU):** Implement gate weights $\mathbf{g}_a, \mathbf{g}_v \in [0, 1]$ computed via sigmoid activations on the incoming modalities. This allows the model to dynamically choose which modality is more informative for a given sample, letting the audio branch take control when visual features are ambiguous.
2. **Feature-Level Dropout (Modality Dropout):** Randomly zero out entire modalities (e.g. dropping visual features 20% of the time, or dropping audio features 20% of the time) during training. This forces the fusion head to learn robust classifiers for both paths independently rather than over-indexing on visual patterns.
3. **Explicit Multi-Task Modality Losses:** Train auxiliary single-modality classification projections alongside the fused head, preventing representation collapse in the weaker modality.
