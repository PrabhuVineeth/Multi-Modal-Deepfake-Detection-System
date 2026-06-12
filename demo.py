"""
CLI demo script for the Deepfake Forensic Detection System.

Usage:
    python demo.py --video path/to/video.mp4 --output results/
    python demo.py --video path/to/video.mp4 --device cuda --no-heatmap
"""

import argparse
import sys
import time
from pathlib import Path

from loguru import logger

from config import get_device, path_config
from utils.logger import setup_logger


def main():
    """Run forensic analysis on a single video via CLI."""
    parser = argparse.ArgumentParser(
        description="Deepfake Forensic Detection — CLI Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py --video sample.mp4
  python demo.py --video sample.mp4 --output results/ --device cuda
  python demo.py --video sample.mp4 --checkpoint best_model.pth --no-heatmap
        """,
    )
    parser.add_argument(
        "--video", type=str, required=True,
        help="Path to the input video file",
    )
    parser.add_argument(
        "--output", type=str, default="output/demo",
        help="Output directory for reports and heatmaps (default: output/demo)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (default: auto-detect)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'auto', 'cuda', 'cpu' (default: auto)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Classification threshold (default: 0.5)",
    )
    parser.add_argument(
        "--no-heatmap", action="store_true",
        help="Skip heatmap video generation",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save debug outputs (intermediate frames, audio)",
    )
    args = parser.parse_args()

    # Validate input
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)

    # Setup
    setup_logger(log_dir=str(Path(args.output) / "logs"))
    device = get_device(args.device)

    print()
    print("=" * 60)
    print("  DEEPFAKE FORENSIC DETECTION SYSTEM")
    print("=" * 60)
    print(f"  Video:  {video_path.name}")
    print(f"  Device: {device}")
    print(f"  Output: {args.output}")
    print("=" * 60)
    print()

    start_time = time.time()

    # Initialize pipeline
    from inference.pipeline import ForensicInferencePipeline
    from config import InferenceConfig

    inf_config = InferenceConfig(
        confidence_threshold=args.threshold,
        device=args.device,
    )
    pipeline = ForensicInferencePipeline(inference_cfg=inf_config)

    # Load model
    checkpoint = args.checkpoint
    if checkpoint is None:
        # Auto-detect
        best_path = path_config.checkpoint_dir / "best_model.pth"
        if best_path.exists():
            checkpoint = str(best_path)
            print(f"  Using checkpoint: {checkpoint}")
        else:
            print("  ⚠ No checkpoint found — using untrained model")
            print("    Results will be random. Train a model first.")
            print()

    pipeline.load_model(checkpoint)

    # Run analysis
    print("  Analyzing video...")
    report = pipeline.analyze(
        str(video_path),
        output_dir=args.output,
        generate_heatmap=not args.no_heatmap,
    )

    total_time = time.time() - start_time

    # Print results
    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)

    cls_icon = "✅" if report.classification == "REAL" else "🚨"
    print(f"  {cls_icon} Classification: {report.classification}")
    print(f"  📊 Confidence:      {report.confidence:.1f}%")
    print()
    print("  Evidence Scores:")
    print(f"    Lip Sync:   {report.lip_sync_score:.4f}")
    print(f"    Identity:   {report.identity_score:.4f}")
    print(f"    Temporal:   {report.temporal_score:.4f}")
    print(f"    AV Sync:    {report.av_sync_score:.4f}")

    if report.boundaries:
        print()
        print("  Temporal Boundaries:")
        for b in report.boundaries:
            print(f"    [{b['tag']:8s}] {b['start_time']:.1f}s — {b['end_time']:.1f}s")

    if report.channel_weights:
        print()
        print("  Evidence Weights:")
        for k, v in report.channel_weights.items():
            bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
            print(f"    {k:10s} |{bar}| {v:.3f}")

    print()
    print(f"  ⏱ Processing time: {total_time:.1f}s")
    print(f"  📁 Output saved to: {args.output}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
