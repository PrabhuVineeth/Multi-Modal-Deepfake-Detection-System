"""
Visualization utilities for the Deepfake Forensic Detection System.

Produces: score distribution plots, attention heatmaps,
forgery boundary timelines, and calibration reliability diagrams.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ──────────────────────────────────────────────
# Style defaults
# ──────────────────────────────────────────────

COLORS = {
    "real": "#2ecc71",
    "fake": "#e74c3c",
    "boundary": "#f39c12",
    "lip_sync": "#3498db",
    "identity": "#9b59b6",
    "temporal": "#e67e22",
    "av_sync": "#1abc9c",
    "confidence": "#2c3e50",
}

plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "text.color": "#eaeaea",
    "axes.labelcolor": "#eaeaea",
    "xtick.color": "#aaaaaa",
    "ytick.color": "#aaaaaa",
    "axes.edgecolor": "#333333",
    "grid.color": "#333333",
    "figure.dpi": 150,
    "font.family": "sans-serif",
})


def plot_score_distribution(
    real_scores: np.ndarray,
    fake_scores: np.ndarray,
    title: str = "Score Distribution",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot overlapping histograms of scores for real vs. fake samples.

    Args:
        real_scores: Array of scores for real samples.
        fake_scores: Array of scores for fake samples.
        title: Plot title.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(real_scores, bins=50, alpha=0.6, color=COLORS["real"], label="Real", density=True)
    ax.hist(fake_scores, bins=50, alpha=0.6, color=COLORS["fake"], label="Fake", density=True)
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    title: str = "Cross-Modal Attention",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a 2D attention weight heatmap.

    Args:
        attention_weights: 2D array of attention weights [query_len, key_len].
        x_labels: Labels for key positions.
        y_labels: Labels for query positions.
        title: Plot title.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(attention_weights, cmap="magma", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Key (Audio)")
    ax.set_ylabel("Query (Visual)")

    if x_labels is not None:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=6)
    if y_labels is not None:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=6)

    fig.colorbar(im, ax=ax, label="Attention Weight")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_boundary_timeline(
    timestamps: np.ndarray,
    tags: np.ndarray,
    scores: Optional[np.ndarray] = None,
    title: str = "Temporal Forgery Boundary",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a timeline showing REAL/FAKE/BOUNDARY segments.

    Args:
        timestamps: Array of frame timestamps in seconds.
        tags: Array of tag predictions (0=REAL, 1=FAKE, 2=BOUNDARY).
        scores: Optional per-frame confidence scores.
        title: Plot title.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), gridspec_kw={"height_ratios": [1, 3]})

    # Top: color-coded timeline bar
    ax_bar = axes[0]
    tag_colors = {0: COLORS["real"], 1: COLORS["fake"], 2: COLORS["boundary"]}
    for i in range(len(timestamps) - 1):
        ax_bar.axvspan(
            timestamps[i], timestamps[i + 1],
            color=tag_colors.get(int(tags[i]), "#888888"),
            alpha=0.8,
        )
    ax_bar.set_xlim(timestamps[0], timestamps[-1])
    ax_bar.set_yticks([])
    ax_bar.set_title(title)

    # Legend
    patches = [
        mpatches.Patch(color=COLORS["real"], label="REAL"),
        mpatches.Patch(color=COLORS["fake"], label="FAKE"),
        mpatches.Patch(color=COLORS["boundary"], label="BOUNDARY"),
    ]
    ax_bar.legend(handles=patches, loc="upper right", fontsize=8)

    # Bottom: confidence score line
    ax_score = axes[1]
    if scores is not None:
        ax_score.plot(timestamps, scores, color=COLORS["confidence"], linewidth=1.5)
        ax_score.fill_between(timestamps, scores, alpha=0.2, color=COLORS["confidence"])
        ax_score.axhline(y=0.5, color="#888888", linestyle="--", alpha=0.5, label="Threshold")
        ax_score.set_ylabel("Confidence")
        ax_score.legend(fontsize=8)
    ax_score.set_xlabel("Time (seconds)")
    ax_score.set_xlim(timestamps[0], timestamps[-1])
    ax_score.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_reliability_diagram(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a calibration reliability diagram.

    Args:
        confidences: Predicted confidence values.
        accuracies: Binary accuracy labels (1=correct, 0=incorrect).
        n_bins: Number of bins for the histogram.
        title: Plot title.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_confs = []
    bin_accs = []
    bin_counts = []

    for lower, upper in zip(bin_lowers, bin_uppers):
        mask = (confidences > lower) & (confidences <= upper)
        if mask.sum() > 0:
            bin_confs.append(confidences[mask].mean())
            bin_accs.append(accuracies[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_confs.append((lower + upper) / 2)
            bin_accs.append(0.0)
            bin_counts.append(0)

    bin_confs = np.array(bin_confs)
    bin_accs = np.array(bin_accs)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "--", color="#888888", label="Perfect Calibration")

    # Bar plot
    bar_width = 1.0 / n_bins
    ax.bar(
        bin_confs, bin_accs, width=bar_width * 0.9,
        color=COLORS["lip_sync"], alpha=0.7, edgecolor="white", linewidth=0.5,
        label="Model",
    )

    # Gap visualization
    for bc, ba in zip(bin_confs, bin_accs):
        if ba > 0:
            ax.plot([bc, bc], [bc, ba], color=COLORS["fake"], linewidth=2, alpha=0.5)

    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction of Positives (Accuracy)")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_forensic_dashboard(
    scores: Dict[str, float],
    classification: str,
    confidence: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a compact forensic dashboard showing all scores.

    Args:
        scores: Dict with keys 'lip_sync', 'identity', 'temporal', 'av_sync'.
        classification: "REAL" or "FAKE".
        confidence: Overall confidence percentage (0-100).
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 5, figsize=(16, 3))

    score_items = [
        ("Lip Sync", scores.get("lip_sync", 0), COLORS["lip_sync"]),
        ("Identity", scores.get("identity", 0), COLORS["identity"]),
        ("Temporal", scores.get("temporal", 0), COLORS["temporal"]),
        ("AV Sync", scores.get("av_sync", 0), COLORS["av_sync"]),
        ("Confidence", confidence / 100, COLORS["confidence"]),
    ]

    for ax, (label, value, color) in zip(axes, score_items):
        # Circular gauge
        theta = np.linspace(0, 2 * np.pi * value, 100)
        theta_bg = np.linspace(0, 2 * np.pi, 100)

        ax.plot(np.cos(theta_bg), np.sin(theta_bg), color="#333333", linewidth=8)
        if len(theta) > 0:
            ax.plot(np.cos(theta), np.sin(theta), color=color, linewidth=8)

        ax.text(0, 0, f"{value:.0%}", ha="center", va="center",
                fontsize=14, fontweight="bold", color=color)
        ax.text(0, -1.5, label, ha="center", va="center", fontsize=10, color="#aaaaaa")
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-2, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")

    # Overall classification color
    cls_color = COLORS["real"] if classification == "REAL" else COLORS["fake"]
    fig.suptitle(
        f"Classification: {classification}",
        fontsize=18,
        fontweight="bold",
        color=cls_color,
        y=1.05,
    )

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig
