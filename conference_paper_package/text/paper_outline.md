# IEEE Conference Paper Draft Outline

## Title
Explainable Multimodal Deepfake Detection Using Audio-Visual Cross-Modal Evidence Aggregation

## Abstract
Briefly state the problem, proposed multimodal system, datasets, explainability outputs, and main results.

## I. Introduction
- Deepfake threat and need for audio-visual forensic systems.
- Limitations of single-modality detection.
- Contributions: multimodal fusion, temporal localization, heatmap/report generation, dataset-aware deployment, modality ablation.

## II. Related Work
- Visual deepfake detection.
- Audio deepfake/spoofing detection.
- Audio-visual/lip-sync consistency detection.
- Explainable forensic reporting and temporal localization.

## III. Proposed Method
Use `tables/architecture_components.csv` and `figures/architecture_mermaid.md`.

## IV. Experimental Setup
- Datasets: FakeAVCeleb, FaceForensics++, LAV-DF.
- Training details: batch size, epochs, learning rates, checkpoint routing.
- Metrics: Accuracy, Precision, Recall, F1, AUC, Balanced Accuracy, confusion matrix, ECE where available.

## V. Results and Discussion
- Final deep-model evaluation and calibration tables.
- Classical ML baseline.
- Modality ablation: Audio-only vs Video-only vs Fusion.
- Discussion of threshold tradeoffs and domain shift.

## VI. Deployment Demonstration
- Streamlit/FastAPI upload system.
- JSON/HTML reports and heatmap outputs.

## VII. Limitations
Use `text/limitations.md`.

## VIII. Conclusion
Summarize multimodal benefit and future work.
