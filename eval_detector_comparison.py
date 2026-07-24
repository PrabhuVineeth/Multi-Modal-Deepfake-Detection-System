"""
Comparative Evaluation: RetinaFace vs MTCNN Face Detectors

This script runs side-by-side benchmarking of RetinaFace and MTCNN face detectors
on test videos, measuring:
1. Face Detection Rate (%)
2. Processing Speed / Latency (ms/frame)
3. Model Classification Probability & Verdict
"""

import time
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

from config import PreprocessConfig
from preprocessing.pipeline import PreprocessingPipeline
from models.inference.pipeline import ForensicInferencePipeline


def find_test_videos(repo_root: Path) -> List[Path]:
    """Find all .mp4 videos in sample directories."""
    paths_to_check = [
        repo_root / "test_videos",
        repo_root / "output" / "uploads",
        repo_root / "output" / "reports",
        repo_root / "conference_paper_package" / "evidence",
    ]
    videos = []
    for p in paths_to_check:
        if p.exists():
            for f in p.rglob("*.mp4"):
                if "heatmap" not in f.name.lower():
                    videos.append(f)
    return videos[:5]


def run_detector_benchmark(video_path: Path) -> Dict[str, Any]:
    """Run comparative benchmark on a single video for both RetinaFace and MTCNN."""
    results = {"video": video_path.name}

    # Initialize inference pipeline
    pipeline = ForensicInferencePipeline()
    pipeline.load_model()

    for backend in ["retinaface", "mtcnn"]:
        logger.info(f"Testing {backend.upper()} on {video_path.name}...")
        prep_cfg = PreprocessConfig(
            face_detection_backend=backend,
            max_frames=64,  # Standard sample batch for fast evaluation
        )
        preprocessor = PreprocessingPipeline(config=prep_cfg)

        start_time = time.time()
        preprocessed = preprocessor.process(video_path)
        det_time = time.time() - start_time

        num_frames = preprocessed.num_frames
        face_det_rate = getattr(preprocessed, 'face_detection_rate', 0.0) * 100.0
        ms_per_frame = (det_time / max(1, num_frames)) * 1000.0

        # Run model inference on preprocessed crops
        tensors = pipeline._prepare_tensors(preprocessed)
        import torch
        with torch.no_grad():
            output = pipeline.model(**tensors)
            prob = float(output.probability.item())
            verdict = "FAKE" if prob >= 0.50 else "REAL"

        results[backend] = {
            "det_rate": face_det_rate,
            "ms_per_frame": ms_per_frame,
            "prob": prob,
            "verdict": verdict,
            "frames": num_frames,
        }

    return results


def main():
    print("=" * 80)
    print("      FACE DETECTOR COMPARATIVE EVALUATION: RETINAFACE vs MTCNN")
    print("=" * 80)

    # Search for sample videos
    repo_root = Path(__file__).parent
    sample_videos = find_test_videos(repo_root)

    if not sample_videos:
        print("No sample test videos found in test_videos/ or output/uploads/.")
        return

    print(f"Found {len(sample_videos)} test video(s) for evaluation.\n")

    all_results = []
    for video in sample_videos:
        try:
            res = run_detector_benchmark(video)
            all_results.append(res)
        except Exception as e:
            logger.error(f"Error benchmarking {video.name}: {e}")

    # Display comparison table
    print("\n" + "=" * 90)
    print(f"{'Video Name':<22} | {'RetinaFace Det%':<15} | {'MTCNN Det%':<12} | {'Retina Lat (ms)':<15} | {'MTCNN Lat (ms)':<15} | {'Retina Verdict':<14} | {'MTCNN Verdict':<13}")
    print("-" * 90)

    retina_rates, mtcnn_rates = [], []
    retina_lats, mtcnn_lats = [], []

    for r in all_results:
        rf = r["retinaface"]
        mt = r["mtcnn"]
        retina_rates.append(rf["det_rate"])
        mtcnn_rates.append(mt["det_rate"])
        retina_lats.append(rf["ms_per_frame"])
        mtcnn_lats.append(mt["ms_per_frame"])

        print(
            f"{r['video'][:20]:<22} | "
            f"{rf['det_rate']:6.1f}%          | "
            f"{mt['det_rate']:6.1f}%       | "
            f"{rf['ms_per_frame']:8.1f} ms      | "
            f"{mt['ms_per_frame']:8.1f} ms     | "
            f"{rf['verdict']} ({rf['prob']:.2f})    | "
            f"{mt['verdict']} ({mt['prob']:.2f})"
        )

    print("=" * 90)
    if retina_rates:
        avg_rf_det = sum(retina_rates) / len(retina_rates)
        avg_mt_det = sum(mtcnn_rates) / len(mtcnn_rates)
        avg_rf_lat = sum(retina_lats) / len(retina_lats)
        avg_mt_lat = sum(mtcnn_lats) / len(mtcnn_lats)

        print("\nAVERAGE SUMMARY:")
        print(f"  • Face Detection Rate : RetinaFace = {avg_rf_det:.1f}%  |  MTCNN = {avg_mt_det:.1f}%")
        print(f"  • Frame Preprocess Latency: RetinaFace = {avg_rf_lat:.1f} ms/frame  |  MTCNN = {avg_mt_lat:.1f} ms/frame")
    print("=" * 90)


if __name__ == "__main__":
    main()
