# Modality Masking (Dropout) Analysis Report

We implemented a **modality masking (modality dropout)** training protocol and benchmarked it on the cached stratified features (RetinaFace + ViT-16) to see if forcing the optimizer to train on single-modality subsets would resolve the **modality drowning** effect.

During training of the Full Fusion model, we applied a **20% masking rate** per sample:
- With 20% probability, the audio feature was zeroed out (`audio = 0`).
- With 20% probability, the visual features were zeroed out (`face = 0`, `mouth = 0`).
- With 60% probability, both modalities remained active.

---

## Modality Masking Benchmark Results

### 1. Overall Metrics
| Configuration | Mean AUC% | Mean EER% | Mean F1% |
| :--- | :---: | :---: | :---: |
| **Audio-Only** | 69.97% ± 1.41% | 35.00% ± 1.69% | 65.71% ± 1.95% |
| **Visual-Only** | **84.15% ± 2.46%** | **23.81% ± 3.05%** | **75.27% ± 3.10%** |
| **Full Fusion (Baseline)** | 81.11% ± 4.99% | 25.90% ± 4.82% | 72.44% ± 5.41% |
| **Full Fusion (Gated/Masked)** | 79.89% ± 7.47% | 28.31% ± 6.45% | 72.75% ± 5.39% |

### 2. Per-Manipulation-Type AUC Breakdown (Reals vs One Fake Type)
| Configuration | Face-Swap (Video-only) | Lip-Sync (Audio-only) | Both (AV) |
| :--- | :---: | :---: | :---: |
| **Audio-Only** | 47.54% | **77.99%** | 84.32% |
| **Visual-Only** | **89.23%** | 65.93% | **97.59%** |
| **Full Fusion (Baseline)** | 86.03% | 61.54% | 94.26% |
| **Full Fusion (Gated/Masked)** | 84.99% | 60.46% | 91.91% |

---

## Critical Findings & Analysis

### 1. Why Simple Modality Masking Failed to Resolve Drowning
The results show that **modality masking (input dropout) did not lift the Lip-Sync AUC** (it went from 61.54% down to 60.46%) and caused minor degradation across all categories. 

The scientific explanation lies in the **structural design of the cross-modal attention layers**:
- **Cross-Attention Degeneracy:** Our fusion layer (`CrossModalFusion`) relies on query-key-value cross-attention (e.g. audio queries attending to face keys/values).
- When visual features are zeroed out ($K=0, V=0$), the attention weight calculation yields uniform weights, but the weighted sum of values becomes exactly $0.0$.
- Because of layer normalization and residual connections:
  $$\text{Output} = \text{LayerNorm}(\text{Query} + \text{Dropout}(\text{Attention}(\text{Query}, 0, 0)))$$
  the output degenerates to a normalized version of the audio alone.
- However, during inference (when all inputs are non-zero), the statistics of these outputs are completely different. Forcing the model to train on zeroed-out inputs disrupts the learning of the projection heads and downstream evidence aggregators, leading to overall generalization loss.

### 2. The Structural Solution
Since simple input-level masking is incompatible with cross-attention alignment, the drowning issue must be solved structurally:
1. **Multi-Task Modality Heads:** Instead of masking input features, we should attach auxiliary, independent classification heads for **Audio-only** and **Visual-only** paths during training. Summing their single-modality losses with the main fusion head loss forces the backbones and projection layers to retain separate modality representations without forcing zeros through the cross-attention blocks.
2. **Post-Attention Gating (Gated Multimodal Unit):** Place a trainable gate (sigmoid layer) at the output of the visual and audio streams before the evidence aggregator to dynamically regulate their influence.
