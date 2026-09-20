"""Generates the comparison plots for docs/tracking_algo_report.docx from the
ground-truth scoring in score_trackers.py. Pure post-processing on already
computed numbers - no model inference.

Usage: python3 scripts/tracking_algo/visualize.py
Writes PNGs to docs/tracking_algo_plots/.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from score_trackers import CLIPS, TRACKERS, VERSIONS, score_all  # noqa: E402

OUT_DIR = Path(__file__).parent.parent.parent / "docs" / "tracking_algo_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Distinct marker per version, distinct color per tracker - standard paper-scatter idiom.
MARKERS = {"v1": "o", "v2": "s", "v3": "^"}
COLORS = {t: c for t, c in zip(TRACKERS, plt.cm.tab10.colors)}
DISPLAY = {"bytetrack": "ByteTrack", "botsort": "BoT-SORT", "ocsort": "OC-SORT",
           "deepocsort": "DeepOCSORT", "mcbyteplusplus": "McByte++"}


def scatter_pairs(gt, scored):
    """The requested paper-style scatter: one point per (tracker, version) pair,
    axes = signed count error (predicted - true) per direction, summed over both
    clips. The origin is the ground truth itself - the closer to (0,0), the better."""
    fig, ax = plt.subplots(figsize=(9, 7))
    # ponytail: no label-collision library on hand - sort by position and stagger
    # offsets deterministically so nearby points don't print on top of each other.
    points = []
    for (t, v), (_, per_clip) in scored.items():
        in_err = sum(per_clip[c][0] for c in CLIPS) - sum(gt[c][0] for c in CLIPS)
        out_err = sum(per_clip[c][1] for c in CLIPS) - sum(gt[c][1] for c in CLIPS)
        points.append((in_err, out_err, t, v))
    points.sort(key=lambda p: (p[0], p[1]))
    for i, (in_err, out_err, t, v) in enumerate(points):
        ax.scatter(in_err, out_err, marker=MARKERS[v], color=COLORS[t], s=140,
                   edgecolor="black", linewidth=0.6, zorder=3)
        dx = 6 if i % 2 == 0 else -70
        dy = [10, -4, -18][i % 3]
        ax.annotate(f"{DISPLAY[t]}/{v}", (in_err, out_err), fontsize=7,
                    xytext=(dx, dy), textcoords="offset points")
    ax.axhline(0, color="grey", linewidth=0.8, zorder=1)
    ax.axvline(0, color="grey", linewidth=0.8, zorder=1)
    ax.scatter(0, 0, marker="*", color="gold", s=400, edgecolor="black",
               linewidth=1, zorder=4, label="Ground truth")
    ax.set_xlabel("IN count error (predicted − true, summed over both clips)")
    ax.set_ylabel("OUT count error (predicted − true, summed over both clips)")
    ax.set_title("Tracker × logic-version accuracy — distance from origin = worse")

    tracker_handles = [plt.Line2D([], [], marker="o", color=COLORS[t], linestyle="",
                                   markersize=9, label=DISPLAY[t]) for t in TRACKERS]
    version_handles = [plt.Line2D([], [], marker=MARKERS[v], color="grey", linestyle="",
                                   markersize=9, label=v) for v in VERSIONS]
    leg1 = ax.legend(handles=tracker_handles, title="Tracker", loc="upper left", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=version_handles, title="Version", loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scatter_pairs.png", dpi=150)
    plt.close(fig)


def heatmap_error(scored):
    grid = np.array([[scored[(t, v)][0] for v in VERSIONS] for t in TRACKERS])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(grid, cmap="RdYlGn_r")
    ax.set_xticks(range(len(VERSIONS)), VERSIONS)
    ax.set_yticks(range(len(TRACKERS)), [DISPLAY[t] for t in TRACKERS])
    for i in range(len(TRACKERS)):
        for j in range(len(VERSIONS)):
            ax.text(j, i, grid[i, j], ha="center", va="center", fontsize=11)
    ax.set_title("Total error (both clips) — lower is better")
    fig.colorbar(im, ax=ax, label="Total absolute error")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "heatmap_error.png", dpi=150)
    plt.close(fig)


def bar_ranking(scored):
    tracker_avg = {t: np.mean([scored[(t, v)][0] for v in VERSIONS]) for t in TRACKERS}
    order = sorted(tracker_avg, key=tracker_avg.get)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([DISPLAY[t] for t in order], [tracker_avg[t] for t in order],
           color=[COLORS[t] for t in order])
    ax.set_ylabel("Avg total error (v1/v2/v3)")
    ax.set_title("Overall tracker ranking — lower is better")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bar_tracker_ranking.png", dpi=150)
    plt.close(fig)


def predicted_vs_actual(gt, scored):
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
    fig.savefig(OUT_DIR / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)


def main():
    gt, scored = score_all()
    scatter_pairs(gt, scored)
    heatmap_error(scored)
    bar_ranking(scored)
    predicted_vs_actual(gt, scored)
    print(f"wrote 4 plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
