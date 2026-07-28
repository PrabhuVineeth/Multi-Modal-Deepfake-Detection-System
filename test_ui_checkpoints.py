"""
Quick degeneracy test for the UI's full-model checkpoints.

The UI loads checkpoints/best_model.pth, which is byte-identical to
best_model_combined.pth -- the degenerate all-fake model. This script runs the
real inference pipeline on one KNOWN-REAL and one KNOWN-FAKE video for each
surviving full checkpoint, so we can pick a non-degenerate one for the UI.

A healthy checkpoint => REAL video -> low prob (REAL), FAKE video -> high prob.
A degenerate all-fake checkpoint => both classified FAKE (prob ~ high for real too).
"""

import sys
from pathlib import Path

from loguru import logger

from models.inference.pipeline import ForensicInferencePipeline

REAL_VIDEO = "G:/NNM24AD071/FakeAVCeleb_v1.2/RealVideo-RealAudio/African/men/id00076/00109.mp4"
FAKE_VIDEO = "G:/NNM24AD071/FakeAVCeleb_v1.2/FakeVideo-RealAudio/African/men/id00076/00109_1.mp4"

CANDIDATES = [
    "checkpoints/best_model.pth",                     # == combined (expected degenerate)
    "checkpoints/best_model_lavdf_full_tuned.pth",
    "checkpoints/best_model_faceforensics_visual.pth",
]


def test_ckpt(ckpt):
    if not Path(ckpt).exists():
        return f"{ckpt}: MISSING"
    try:
        pipe = ForensicInferencePipeline()
        pipe.load_model(ckpt)
        rr = pipe.analyze(REAL_VIDEO, generate_heatmap=False)
        fr = pipe.analyze(FAKE_VIDEO, generate_heatmap=False)
        healthy = (rr.classification == "REAL" and fr.classification == "FAKE")
        return (f"{Path(ckpt).name}:\n"
                f"    REAL video -> {rr.classification:<5} (p={rr.raw_probability:.3f})\n"
                f"    FAKE video -> {fr.classification:<5} (p={fr.raw_probability:.3f})\n"
                f"    => {'HEALTHY (use this)' if healthy else 'DEGENERATE / not usable'}")
    except Exception as e:
        return f"{Path(ckpt).name}: ERROR {type(e).__name__}: {e}"


def main():
    ckpts = sys.argv[1:] or CANDIDATES
    print("\n" + "=" * 70)
    print("UI CHECKPOINT DEGENERACY TEST (threshold 0.5)")
    print("=" * 70)
    for c in ckpts:
        print(test_ckpt(c))
        print("-" * 70)


if __name__ == "__main__":
    main()
