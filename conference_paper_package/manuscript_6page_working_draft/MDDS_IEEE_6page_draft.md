# Explainable Multimodal Deepfake Detection Using Audio-Visual Cross-Modal Evidence Aggregation

**Author 1**, **Author 2**, **Vineeth Prabhu / NNM24AD071**  
Department of Artificial Intelligence and Data Science  
NMAM Institute of Technology, NITTE (Deemed to be University), Mangaluru, India  
Email: `nnm24ad071@nmamit.in`  

> Mentor note: replace the author block with the final IEEE author list before submission.

## Abstract

The rapid improvement of face reenactment, voice cloning, and lip-synchronization tools has made deepfake detection a multimodal forensic problem rather than a purely visual classification task. This paper presents a Multimodal Deepfake Detection System (MDDS) that jointly analyzes speech audio, face crops, mouth regions, and temporal audio-visual consistency. The proposed architecture combines a Wav2Vec2-based audio encoder, ViT-based visual encoders, cross-modal attention modules, forensic evidence heads, and attention-weighted evidence aggregation. In addition to binary real/fake classification, the system produces interpretable evidence scores, temporal anomaly traces, heatmap overlays, and JSON/HTML forensic reports. Experiments were conducted on FakeAVCeleb, FaceForensics++, and LAV-DF with dataset-aware routing and validation-selected thresholds. A feature-based classical machine learning baseline was also evaluated on FakeAVCeleb to study audio-only, video-only, and audio-video fusion behavior. The best classical baseline, Fusion XGBoost, achieved an F1-score of 0.8806 and ROC-AUC of 0.9335 on the 750/class FakeAVCeleb setting. The full LAV-DF model achieved an F1-score of 0.7506 at the validation-selected threshold, improving fake recall from 0.7198 to 0.8087 compared with the default threshold. Results show that multimodal fusion improves robustness over single-modality analysis while explainability outputs make the system more suitable for forensic review.

**Index Terms**-Deepfake detection, multimodal learning, audio-visual forensics, cross-modal attention, FakeAVCeleb, FaceForensics++, LAV-DF, explainable AI.

## I. Introduction

Synthetic media generation has progressed from visually obvious face manipulation to realistic audio-visual forgeries involving face swapping, speech synthesis, and lip synchronization. As a result, forensic detection systems must reason not only about visual artifacts, but also about the consistency between a speaker's voice, identity, mouth motion, and temporal continuity. A model that analyzes only image frames can miss audio-driven manipulations, while an audio-only detector may ignore visual evidence of face replacement or temporal boundary artifacts.

This work presents an explainable Multimodal Deepfake Detection System (MDDS) designed around audio-visual evidence fusion. The system processes an input video through audio extraction, frame extraction, face localization, mouth region extraction, audio encoding, visual encoding, cross-modal attention, evidence aggregation, and report generation. Unlike a black-box classifier, MDDS exposes channel-level forensic scores for lip-sync, identity, temporal consistency, and audio-video synchronization. It also generates frame-level anomaly traces, heatmap overlays, and structured forensic reports.

The major contributions of this work are: (1) a complete audio-visual deepfake detection pipeline combining Wav2Vec2 audio representations, ViT-based visual representations, and cross-modal attention; (2) dataset-aware routing for FakeAVCeleb, FaceForensics++, and LAV-DF, including visual-only routing where audio is unreliable; (3) explainability outputs including per-channel evidence scores, frame anomaly timelines, heatmap videos, and HTML/JSON reports; (4) a FakeAVCeleb modality ablation baseline comparing audio-only, video-only, and audio-video fusion; and (5) validation-based threshold calibration.

## II. Literature Survey

Early deepfake detection work focused mainly on visual artifacts in face manipulation datasets. FaceForensics++ introduced a large benchmark for manipulated facial images and videos covering DeepFakes, Face2Face, FaceSwap, and NeuralTextures [1]. However, modern forgeries increasingly involve both visual and audio manipulation. FakeAVCeleb addressed this gap by providing an audio-video multimodal dataset containing both deepfake videos and synthesized or cloned audio [2]. LAV-DF further emphasized localized audio-visual forgery detection, where only a temporal segment may be manipulated and the model must localize the forged region [3].

The representation backbone of MDDS draws from recent advances in self-supervised speech and visual transformers. Wav2Vec2 learns strong speech representations from raw audio through self-supervised pretraining [4]. Vision Transformer (ViT) demonstrates that image patches can be modeled effectively with transformer-based architectures when pretrained at scale [5]. RetinaFace provides robust single-stage face localization for unconstrained face detection [6]. Cross-modal fusion is motivated by transformer attention mechanisms [7], while the classical baseline uses XGBoost because gradient-boosted trees often perform strongly on tabular engineered features [8].

### A. Research Gap

The reviewed literature suggests three practical gaps. First, visual-only systems do not explicitly model voice-face, lip-sync, or audio-video timing inconsistencies. Second, many benchmark systems produce only a binary label and do not provide interpretable forensic evidence. Third, localized manipulation datasets require temporal reasoning and not just video-level classification. MDDS addresses these gaps through cross-modal attention, channel-level evidence heads, temporal anomaly traces, and deployable report generation.

## III. Methodology

### A. System Architecture

An input video is decomposed into audio and visual streams. Audio is resampled and encoded using a Wav2Vec2-style encoder. Visual frames are sampled, faces are detected using RetinaFace, and face/mouth crops are encoded using ViT-based modules. The encoded streams are fused using cross-modal attention pathways designed to capture speech-lip consistency, voice-identity consistency, and audio-video timing consistency. Four forensic analyzers estimate lip-sync, identity, temporal, and AV-sync evidence scores. The evidence aggregation module combines these signals into a final prediction and confidence score.

**Fig. 1.** Proposed MDDS architecture. The submission-ready image is available as `architecture_diagram.png` in this folder.

### B. Preprocessing and Inference

The preprocessing pipeline extracts audio at 16 kHz, samples video frames, detects faces, crops mouth regions, and synchronizes audio windows with frame timestamps. The final inference setting uses 64 frames per video to reduce the risk of missing short temporal manipulations. FaceForensics++ is routed as visual-only because the available dataset audio was silent or unreliable. FakeAVCeleb and LAV-DF are routed through multimodal inference.

### C. Threshold Calibration

The system uses sigmoid fake probability scores. A video is classified as fake when the probability is greater than or equal to the selected threshold. Because different datasets have different class distributions and manipulation types, thresholds are selected on validation splits and then evaluated on held-out test splits. This prevents direct threshold tuning on the test set.

### D. Classical ML Modality Ablation

To separately study modality contribution, a classical ML baseline was built on FakeAVCeleb features. The feature vector follows the order `[video features | audio features]`. Three input modes were evaluated: audio-only, video-only, and fusion. Logistic Regression, SVM, Random Forest, and XGBoost were trained, and validation-selected thresholds were used for test evaluation. This ablation is not a replacement for the deep model; it is used to interpret the relative strength of audio, video, and fused features.

## IV. Experimental Setup

### A. Datasets

**Table I. Dataset Summary**

| Dataset | Modalities | Manipulation Types | Boundary Labels | Project Status |
|---|---|---|---|---|
| FakeAVCeleb | Audio + Video | Face-swap, lip-sync, multimodal fake combinations | No | Full training and ablations available |
| FaceForensics++ | Video / visual-only route | DeepFakes, Face2Face, FaceSwap, NeuralTextures | No | Visual-only checkpoint used |
| LAV-DF | Audio + Video | Realistic audio-visual deepfakes | Yes | Full checkpoint available |

### B. Metrics and Implementation

The system is evaluated using accuracy, precision, recall, F1-score, balanced accuracy, ROC-AUC, and confusion matrix values. F1-score is used as the primary single-number metric when balancing precision and recall. The project was implemented in PyTorch with FastAPI and Streamlit deployment interfaces. Training used cached preprocessing artifacts to reduce repeated frame/audio extraction cost.

## V. Results and Discussion

### A. LAV-DF Full Model

**Table II. LAV-DF Full Model Threshold Comparison**

| Threshold | Accuracy | Precision | Recall | F1 | TP | TN | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.7330 | 0.7393 | 0.7198 | 0.7294 | 2623 | 2719 | 925 | 1021 |
| 0.40 | 0.7313 | 0.7003 | 0.8087 | 0.7506 | 2947 | 2383 | 1261 | 697 |
| 0.49 | 0.7357 | 0.7378 | 0.7313 | 0.7346 | 2665 | 2697 | 947 | 979 |

At threshold 0.50, the LAV-DF model achieved F1 = 0.7294. The validation-selected F1 threshold of 0.40 increased recall from 0.7198 to 0.8087 and improved F1 to 0.7506, reducing false negatives from 1021 to 697.

### B. Dataset-Aware Calibration

**Table III. Dataset-Aware Calibration Summary**

| Dataset | Checkpoint | Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| FakeAVCeleb | combined | 0.96 | 0.8200 | 0.7424 | 0.9800 | 0.8448 |
| FaceForensics++ | combined | 0.52 | 0.6750 | 0.6224 | 0.8900 | 0.7325 |
| LAV-DF | full | 0.40 | 0.7313 | 0.7003 | 0.8087 | 0.7506 |

### C. Classical ML Baseline and Modality Ablation

**Table IV. FakeAVCeleb Classical ML Fusion Baseline**

| Sample Setting | Modality | Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 500/class | Fusion | XGBoost | 0.32 | 0.8579 | 0.8431 | 0.8866 | 0.8643 | 0.9384 |
| 750/class | Fusion | XGBoost | 0.71 | 0.8632 | 0.9291 | 0.8369 | 0.8806 | 0.9335 |

Fusion XGBoost was the strongest classical ML model in both 500/class and 750/class settings. This supports the conclusion that audio-video fusion is more reliable than using either modality alone, while audio-only features were weaker and less stable.

### D. Discussion

The results show that threshold selection strongly affects deployment behavior. Lower thresholds increase fake recall but can generate more false positives. Higher thresholds improve precision but may miss subtle manipulations. The LAV-DF results demonstrate this tradeoff clearly: threshold 0.40 catches more fake samples than threshold 0.50, reducing false negatives from 1021 to 697. The final Streamlit interface therefore exposes raw fake probability and threshold values so users can interpret decisions rather than relying only on a binary label.

## VI. Deployment Demonstration

The project includes a FastAPI backend and a Streamlit web interface. The user can upload a video, select a dataset profile, and receive a real/fake verdict with confidence, raw fake probability, dataset threshold, evidence scores, temporal anomaly plot, heatmap preview, and downloadable forensic reports. This deployment layer demonstrates the practical usability of the proposed method for forensic review workflows.

## VII. Limitations

The system has several limitations. First, benchmark datasets differ in compression, audio availability, manipulation type, and class balance, making direct cross-dataset comparison difficult. Second, FaceForensics++ was handled as visual-only because the available audio was silent or unreliable. Third, classical ML feature extraction skipped some videos because of face or audio extraction failure, causing post-extraction imbalance. Fourth, the heatmap is a frame-level anomaly overlay and should not be interpreted as pixel-level causal attribution.

## VIII. Conclusion

This paper presented MDDS, an explainable multimodal deepfake detection system that combines audio, visual, temporal, and cross-modal evidence. The architecture provides calibrated real/fake classification together with evidence scores, temporal anomaly traces, heatmaps, and forensic reports. Experiments across FakeAVCeleb, FaceForensics++, and LAV-DF show that dataset-aware thresholding is essential, and the FakeAVCeleb ML ablation confirms that fusion is more reliable than audio-only or video-only analysis. Future work will focus on stronger cross-dataset generalization, improved temporal localization, larger-scale balanced evaluation, and better handling of silent or low-quality audio.

## References

[1] A. Rössler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and M. Nießner, "FaceForensics++: Learning to Detect Manipulated Facial Images," arXiv:1901.08971, 2019.

[2] H. Khalid, S. Tariq, and S. S. Woo, "FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset," arXiv:2108.05080, 2021.

[3] Y. Cai et al., "Do You Really Mean That? Content Driven Audio-Visual Deepfake Dataset and Multimodal Method for Temporal Forgery Localization," arXiv:2204.06228, 2022.

[4] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," NeurIPS, 2020.

[5] A. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," ICLR, 2021.

[6] J. Deng et al., "RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild," CVPR, 2020.

[7] A. Vaswani et al., "Attention Is All You Need," NeurIPS, 2017.

[8] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," KDD, 2016.
