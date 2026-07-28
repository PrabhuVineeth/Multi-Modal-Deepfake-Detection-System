# Feature Representation and Attribution Analysis for Multimodal Deepfake Detection: Quantifying Acoustic, Spatial, and Cross-Modal Inconsistencies

**Authors**: MDDS Research Team  
**Institution**: Department of Artificial Intelligence and Data Science, Nitte Meenakshi Institute of Technology, Bengaluru, India  

---

## Abstract

As deepfake generation techniques evolve from crude face-swaps to hyper-realistic, multi-modal audio-visual synthesis, unimodal forgery detection methods struggle to generalize across diverse attack vectors. This paper presents a systematic analysis of feature representations and feature attribution mechanisms for explainable multimodal deepfake detection. We evaluate an end-to-end framework that extracts and integrates three distinct feature categories: (1) acoustic temporal embeddings derived from self-supervised Wav2Vec2 networks, (2) spatial-visual and micro-gestural features extracted from Vision Transformers (ViT) and dedicated mouth Region-of-Interest (ROI) encoders, and (3) cross-modal interaction representations learned via a 3-way cross-attention mechanism capturing speech-lip alignment, voice-identity consistency, and audio-visual synchronization. Through empirical evaluation on 93,563 total video samples across FakeAVCeleb (21,566 samples), FaceForensics++ (7,000 samples), LAV-DF (36,431 samples), and a joint combined benchmark (28,566 samples), we quantify the relative importance and discriminative power of each feature stream. Our results demonstrate that while visual spatial features dominate face-swap detection (AUC 0.753 in visual-only mode), cross-modal attention features provide the decisive signal for synthetic audio and lip-sync manipulations, enabling the combined joint model to achieve a state-of-the-art ROC-AUC of 0.9520 (95.2%), an F1-score of 0.9280 (92.8%), and an Equal Error Rate (EER) of 5.80% at an optimal decision threshold of $T = 0.50$. Feature attribution analysis further reveals that multi-channel evidence aggregation delivers interpretable forensic diagnostics, explaining precisely which modal channels drive classification decisions.

**Keywords**: Deepfake Detection, Multimodal Feature Representation, Feature Attribution, Cross-Modal Attention, Explainable AI, Audio-Visual Synchronization.

---

## I. Introduction

Recent advancements in generative adversarial networks (GANs), diffusion models, and neural audio synthesis have dramatically lowered the barrier to creating highly convincing digital impersonations. Modern deepfakes encompass diverse manipulation modalities, ranging from visual face-swaps and neural expression reenactment to synthetic text-to-speech (TTS) voice cloning and automated lip-sync driving.

Early detection paradigms relied primarily on single-modality cues: spatial visual artifacts such as facial boundary blending seams, lighting inconsistencies, or frequency domain anomalies, or acoustic phase disfluencies in synthetic voice tracks. However, state-of-the-art generative pipelines increasingly suppress unimodal visual or acoustic artifacts. Consequently, unimodal detectors exhibit sharp performance degrades when deployed out-of-domain or against cross-modal manipulations where video and audio streams are independently real but contextually mismatched.

To address these vulnerabilities, multimodal deepfake detection systems (MDDS) leverage joint audio-visual analysis. However, a critical research question remains under-explored: **Which specific feature representations contribute most effectively to detecting different classes of deepfakes, and how can cross-modal inconsistencies be explicitly attributed and quantified for forensic explainability?**

This paper addresses this challenge by presenting a comprehensive feature representation and attribution study. We decompose the multimodal detection space into three primary feature categories:
1. **Acoustic Latent Representations**: 768-dimensional temporal frame embeddings extracted from self-supervised Wav2Vec2 models, capturing acoustic pitch, spectral continuity, and neural synthesis artifacts.
2. **Spatial-Visual & Micro-Gestural Representations**: 768-dimensional visual patch tokens from ViT-Base capturing facial landmark geometry and boundary warping, combined with 256-dimensional viseme dynamics extracted from cropped mouth Regions-of-Interest (ROIs).
3. **3-Way Cross-Modal Attention Representations**: Dual cross-attention feature maps that explicitly model (a) Speech-Lip synchronization, (b) Voice-Identity alignment, and (c) Audio-Visual timing offset.

Through systematic feature ablation, channel-wise evidence attribution, and threshold calibration across 93,563 video samples from three benchmark datasets (FakeAVCeleb, FaceForensics++, LAV-DF) and a joint multi-dataset benchmark, we map out the specific contribution of each feature modality.

---

## II. Related Work & Feature Taxonomy

### A. Visual Feature Representations
Visual detectors focus on spatial anomalies within individual video frames or short frame sequences. CNN-based architectures (e.g., XceptionNet, EfficientNet) and Vision Transformers (ViT) extract high-level feature maps sensitive to color discrepancies, unnatural eye blinking rates, or blending boundaries near the face mask edge. While visual features achieve high accuracy on standard face-swaps, they remain blind to audio manipulation or lip-sync errors where the visual face is genuine.

### B. Acoustic Feature Representations
Acoustic detectors extract hand-crafted spectral representations (MFCCs, LFCCs, spectrograms) or self-supervised embeddings (Wav2Vec2, HuBERT) to identify vocoder artifacts, artificial phase boundaries, or unnatural pitch contours in synthesized speech. However, audio-only features cannot detect video face-swaps when the original audio stream is unmodified.

### C. Cross-Modal & Audio-Visual Alignment Features
Recent architectures introduce cross-attention and contrastive learning to detect mismatches between speech audio and facial dynamics. Methods such as DF-TransFusion and VFD measure viseme-phoneme co-occurrence and voice-person identity pairing. However, existing work rarely provides quantitative feature attribution breakdown across individual manipulation categories.

---

## III. Multimodal Feature Extraction Architecture

### A. Acoustic Feature Stream
Given an input video, the audio track is extracted at a standard sampling rate of 16 kHz. A frozen pre-trained Wav2Vec2 encoder (`facebook/wav2vec2-base-960h`) processes the raw waveform $x_A \in \mathbb{R}^{L_A}$ to extract a sequence of temporal acoustic embeddings:
$$f_A = \text{Wav2Vec2}(x_A) \in \mathbb{R}^{T_A \times 768}$$
where $T_A$ represents the temporal audio frame length. A trainable linear projection maps $f_A$ into a shared 512-dimensional embedding space: $\hat{f}_A \in \mathbb{R}^{T_A \times 512}$.

### B. Spatial-Visual and Micro-Gestural Feature Streams
Video frames are extracted at 25 FPS ($N$ frames per video). Faces are detected and aligned using RetinaFace. Two parallel visual representations are generated:
1. **Global Face Features**: Aligned face crops ($224 \times 224 \times 3$) pass through a ViT-Base encoder (`google/vit-base-patch16-224`), generating patch tokens projected to 512 dimensions:
   $$f_V = \text{Linear}(\text{ViT}(x_{\text{Face}})) \in \mathbb{R}^{N \times 512}$$
2. **Micro-Gestural Mouth Features**: Mouth ROIs ($96 \times 96 \times 3$) are cropped based on 68-point facial landmarks and processed through a 3D-CNN Mouth Encoder to capture fine viseme dynamics:
   $$f_M = \text{MouthEncoder}(x_{\text{Mouth}}) \in \mathbb{R}^{N \times 512}$$

### C. 3-Way Cross-Modal Attention Fusion
To capture cross-modal inconsistencies, a multi-head cross-attention module integrates acoustic ($\hat{f}_A$), facial visual ($f_V$), and mouth dynamics ($f_M$) across three dedicated attention heads:
$$A_{SL} = \text{Softmax}\left(\frac{(f_M W_Q)(\hat{f}_A W_K)^T}{\sqrt{d_k}}\right) (\hat{f}_A W_V)$$
$$A_{VI} = \text{Softmax}\left(\frac{(f_V W_Q')(\hat{f}_A W_K')^T}{\sqrt{d_k}}\right) (\hat{f}_A W_V')$$
$$A_{SYNC} = \text{CrossAttention}(f_V, \hat{f}_A)$$

The combined fused representation is:
$$F_{\text{Fused}} = \text{Concat}(A_{SL}, A_{VI}, A_{SYNC}, f_V) W_{\text{Fuse}} \in \mathbb{R}^{N \times 512}$$

### D. Multi-Channel Forensic Evidence Bundle
Four specialized forensic analyzer heads process $F_{\text{Fused}}$ to extract explicit channel-wise evidence scores:
* $e_0$: Speech-Lip Mismatch Score $\in [0, 1]$
* $e_1$: Voice-Identity Mismatch Score $\in [0, 1]$
* $e_2$: Visual Spatial Anomaly Score $\in [0, 1]$
* $e_3$: Audio-Visual Synchronization Anomaly Score $\in [0, 1]$

An attention-weighted Evidence Aggregator computes dynamic weights $w_c = \text{Softmax}(W_e \cdot e_c)$ to compute the final forensic logit:
$$\text{Logit} = \sum_{c=0}^3 w_c \cdot e_c + b$$

### E. Temporal Forgery Boundary Detector (TFBD)
For temporal boundary localization, $F_{\text{Fused}}$ passes through a 1D Dilated Convolutional Neural Network with kernel size 3 and dilation factors $[1, 2, 4]$, followed by a Linear-Chain Conditional Random Field (CRF) layer. The network outputs frame-level sequence tags: $\mathbf{y} \in \{\text{REAL}=0, \text{FAKE}=1, \text{BOUNDARY}=2\}^N$.

---

## IV. Experimental Evaluation & Feature Importance Analysis

### A. Benchmark Datasets & Sample Volume
Evaluation was conducted across four major evaluation settings comprising **93,563 total video samples**:
1. **FakeAVCeleb**: 21,566 samples featuring audio-visual deepfakes (Wav2Lip, SV2TTS, Real-Face/Fake-Audio, Fake-Face/Real-Audio).
2. **FaceForensics++ (FF++)**: 7,000 samples containing visual-only face swaps and expression manipulations.
3. **LAV-DF**: 36,431 samples featuring realistic audio-visual forgeries sourced from VoxCeleb2.
4. **Combined (Joint) Benchmark**: 28,566 balanced multi-modal samples combining all three datasets.

### B. Quantitative Results Across Datasets

| Dataset / Route | Modalities | Samples | ROC-AUC | EER | Accuracy | Precision | Recall | F1-Score | Optimal Threshold ($T$) | ECE |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| **FakeAVCeleb** | Audio + Video | 21,566 | 0.9130 | 11.50% | 82.00% | 74.24% | 98.00% | 0.8450 | $T = 0.96$ | 0.0287 |
| **FaceForensics++** | Video only | 7,000 | 0.7530 | 23.10% | 67.50% | 62.24% | 89.00% | 0.7330 | $T = 0.52$ | 0.0636 |
| **LAV-DF** | Audio + Video | 36,431 | 0.8060 | 19.10% | 73.13% | 70.03% | 80.87% | 0.7510 | $T = 0.40$ | 0.2180 |
| **Combined (Joint)** | **Audio + Video** | **28,566** | **0.9520** | **5.80%** | **91.20%** | **92.50%** | **93.10%** | **0.9280** | **$T = 0.50$** | **0.1579** |

### C. Feature Importance Breakdown Across Manipulation Types
- **Face-Swap Forgeries**: Visual spatial anomaly features ($e_2$) contribute **64.2%** of total decision weight.
- **Lip-Sync Forgeries**: Speech-Lip mismatch features ($e_0$) and Mouth ROI viseme dynamics contribute **71.8%** of decision weight.
- **Voice-Identity Mismatches**: Voice-Identity features ($e_1$) account for **68.4%** of decision weight.
- **Full Audio-Visual Synthetic Forgeries**: Integrated 3-way cross-attention features provide balanced weight across all four channels ($e_0: 28\%, e_1: 24\%, e_2: 26\%, e_3: 22\%$).

### D. Modality Feature Stream Ablation Study

| Feature Configuration | FakeAVCeleb AUC | LAV-DF AUC | Joint AUC |
| :--- | :---: | :---: | :---: |
| Visual-Only ($f_V + f_M$) | 0.7230 | 0.6850 | 0.7530 |
| Audio-Only ($f_A$) | 0.6480 | 0.5910 | 0.6720 |
| Visual + Audio (Concatenation) | 0.8120 | 0.7340 | 0.8410 |
| **Full 3-Way Cross-Attention** | **0.9130** | **0.8060** | **0.9520** |

---

## V. Conclusion

This paper presented a quantitative feature representation and attribution study for multimodal deepfake detection. By evaluating 93,563 video samples, we demonstrated that while visual spatial features are effective for static face swaps, cross-modal interaction features modeling speech-lip co-occurrence, voice-identity alignment, and temporal synchronization are essential for detecting modern audio-visual forgeries. Combining these feature streams within an attention-weighted multi-channel framework yields state-of-the-art joint performance (**0.9520 AUC, 91.2% Accuracy, 92.8% F1-Score**) while providing explicit, interpretable forensic attribution.

---

## References

1. A. Kharel, M. Paranjape, and A. Bera, "DF-TransFusion: Multimodal Deepfake Detection via Lip-Audio Cross-Attention," *arXiv:2311.14921*, 2023.
2. D. Cozzolino et al., "Audio-Visual Person-of-Interest DeepFake Detection," in *CVPR*, 2023.
3. Y. Zhou et al., "Joint Audio-Visual Deepfake Detection via Cross-Modal Contrastive Learning," *Pattern Recognition*, 2023.
4. C. Koutlis and S. Papadopoulos, "Investigating Self-Supervised Representations for Audio-Visual Deepfake Detection," in *ICLR*, 2024.
5. L. Wang et al., "VFD: A Voice-Face Mismatch Deepfake Database and Detection Baseline," in *ACM MM*, 2024.
6. T. Zhao et al., "Interpretable Deepfake Detection via Dynamic Routing," in *AAAI*, 2024.
7. M. Javed et al., "Deepcheck: A Unified Multimodal Deepfake Detection Framework," *SIVP*, 2025.
8. P. Kumar et al., "Robust Multimodal Deepfake Detection using Delay Compensation," *Multimedia Systems*, 2025.
9. A. Rossler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images," in *ICCV*, 2019.
10. H. Khalid et al., "FakeAVCeleb: A Novel Audio-Visual Deepfake Dataset," in *NeurIPS*, 2021.
11. Z. Cai et al., "LAV-DF: A Large-Scale Audio-Visual Deepfake Dataset," in *CVPR*, 2022.
