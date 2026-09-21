"""Generates the comparison plots for docs/tracking_algo_report.docx from the
ground-truth scoring in score_trackers.py. Pure post-processing on already
computed numbers - no model inference.

Usage: python3 scripts/tracking_algo/visualize.py [--results-dir DIR] [--out-dir DIR]
Writes PNGs to docs/tracking_algo_plots/ by default.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from score_trackers import CLIPS, RESULTS, TRACKERS, VERSIONS, score_all  # noqa: E402

OUT_DIR = Path(__file__).parent.parent.parent / "docs" / "tracking_algo_plots"

# Distinct marker per version, distinct color per tracker - standard paper-scatter idiom.
MARKERS = {"v1": "o", "v2": "s", "v3": "^"}
COLORS = {t: c for t, c in zip(TRACKERS, plt.cm.tab10.colors)}
DISPLAY = {"bytetrack": "ByteTrack", "botsort": "BoT-SORT", "ocsort": "OC-SORT",
           "deepocsort": "DeepOCSORT", "mcbyteplusplus": "McByte++"}

plt.rcParams.update({
    "axes.titleweight": "bold", "axes.titlesize": 13, "axes.labelsize": 11,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": ":",
})


def bar_pairs_ranking(scored, out_dir=OUT_DIR):
    """Standard ranked horizontal bar chart, one bar per (tracker, version)
    pair, sorted best (shortest bar) to worst. Bar length = total count error
    - the same number score_trackers.py ranks by - so ties are just equal-
    length bars with equal printed numbers: no axis trade-off to interpret,
    the way the earlier signed-error scatter required."""
    pts = sorted(((err, t, v) for (t, v), (err, _) in scored.items()), key=lambda p: p[0])
    best = pts[0][0]
    labels = [f"{DISPLAY[t]} / {v}" for _, t, v in pts]
    errors = [e for e, _, _ in pts]
    colors = [COLORS[t] for _, t, _ in pts]
    edgecolors = ["gold" if e == best else "black" for e in errors]
    linewidths = [2.5 if e == best else 0.6 for e in errors]

    fig, ax = plt.subplots(figsize=(8, 7))
    y = np.arange(len(pts))[::-1]  # rank 1 at the top
    bars = ax.barh(y, errors, color=colors, edgecolor=edgecolors, linewidth=linewidths)
    ax.bar_label(bars, fmt="%d", padding=4, fontsize=9)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Total count error (|IN error| + |OUT error|, both clips) — lower is better")
    ax.set_title("Tracker × Version, ranked by total error")
    ax.margins(x=0.1)
    fig.tight_layout()
    fig.savefig(out_dir / "pair_ranking.png", dpi=150)
    plt.close(fig)


def heatmap_error(scored, out_dir=OUT_DIR):
    grid = np.array([[scored[(t, v)][0] for v in VERSIONS] for t in TRACKERS])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.grid(False)  # grid lines make no sense over a cell heatmap
    im = ax.imshow(grid, cmap="RdYlGn_r")
    ax.set_xticks(range(len(VERSIONS)), VERSIONS)
    ax.set_yticks(range(len(TRACKERS)), [DISPLAY[t] for t in TRACKERS])
    for i in range(len(TRACKERS)):
        for j in range(len(VERSIONS)):
            ax.text(j, i, grid[i, j], ha="center", va="center", fontsize=11)
    ax.set_title("Total error (both clips) — lower is better")
    fig.colorbar(im, ax=ax, label="Total absolute error")
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_error.png", dpi=150)
    plt.close(fig)


def bar_ranking(scored, out_dir=OUT_DIR):
    tracker_avg = {t: np.mean([scored[(t, v)][0] for v in VERSIONS]) for t in TRACKERS}
    order = sorted(tracker_avg, key=tracker_avg.get)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar([DISPLAY[t] for t in order], [tracker_avg[t] for t in order],
                  color=[COLORS[t] for t in order], edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_ylabel("Avg total error (v1/v2/v3)")
    ax.set_title("Overall tracker ranking — lower is better")
    ax.margins(y=0.12)
    fig.tight_layout()
    fig.savefig(out_dir / "bar_tracker_ranking.png", dpi=150)
    plt.close(fig)


def predicted_vs_actual(gt, scored, out_dir=OUT_DIR):
    """Classic ML-paper predicted-vs-actual scatter: total (IN+OUT) crossings per
    clip per pair, against the y=x line."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    truths, preds = [], []
    for (t, v), (_, per_clip) in scored.items():
        for c in CLIPS:
            true_total = sum(gt[c])
            pred_total = per_clip[c][0] + per_clip[c][1]
            truths.append(true_total)
            preds.append(pred_total)
            ax.scatter(true_total, pred_total, color=COLORS[t], marker=MARKERS[v],
                       s=90, edgecolor="black", linewidth=0.5, alpha=0.85)
    lo, hi = min(truths) - 2, max(truths + preds) + 2
    ax.plot([lo, hi], [lo, hi], "--", color="grey", linewidth=1, label="perfect prediction")
    ax.set_xlabel("True total crossings (ground truth)")
    ax.set_ylabel("Predicted total crossings")
    ax.set_title("Predicted vs. actual crossings, per clip per pair")
    tracker_handles = [plt.Line2D([], [], marker="o", color=COLORS[t], linestyle="",
                                   markersize=9, label=DISPLAY[t]) for t in TRACKERS]
    ax.legend(handles=tracker_handles + [plt.Line2D([], [], linestyle="--", color="grey",
              label="perfect prediction")], fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=RESULTS)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt, scored = score_all(args.results_dir)
    bar_pairs_ranking(scored, args.out_dir)
    heatmap_error(scored, args.out_dir)
    bar_ranking(scored, args.out_dir)
    predicted_vs_actual(gt, scored, args.out_dir)
    print(f"wrote 4 plots to {args.out_dir}")


if __name__ == "__main__":
    main()
