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
DPI = 160

# Distinct marker per version, distinct color per tracker - standard paper-scatter idiom.
MARKERS = {"v1": "o", "v2": "s", "v3": "^"}
COLORS = {t: c for t, c in zip(TRACKERS, plt.cm.tab10.colors)}
DISPLAY = {"bytetrack": "ByteTrack", "botsort": "BoT-SORT", "ocsort": "OC-SORT",
           "deepocsort": "DeepOCSORT", "mcbyteplusplus": "McByte++"}

# seaborn-v0_8-* styles ship inside matplotlib itself (no seaborn dependency) -
# cleaner default look (whitegrid, softer spines) than stock matplotlib.
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.titleweight": "bold", "axes.titlesize": 14, "axes.titlepad": 12,
    "axes.labelsize": 11.5,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.9,
    "grid.alpha": 0.35, "grid.linestyle": "--", "grid.linewidth": 0.6,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9.5, "legend.frameon": True, "legend.framealpha": 0.92,
    "legend.edgecolor": "#cccccc", "font.family": "DejaVu Sans",
})


def _tracker_legend(ax, **kwargs):
    handles = [plt.Line2D([], [], marker="o", color=COLORS[t], linestyle="",
                           markersize=9, markeredgecolor="black", markeredgewidth=0.5,
                           label=DISPLAY[t]) for t in TRACKERS]
    return ax.legend(handles=handles, title="Tracker", **kwargs)


def bar_pairs_ranking(scored, out_dir=OUT_DIR):
    """Standard ranked horizontal bar chart, one bar per (tracker, version)
    pair, sorted best (shortest bar) to worst. Bar length = total count error
    - the same number score_trackers.py ranks by - so ties are just equal-
    length bars with equal printed numbers: no axis trade-off to interpret,
    the way a signed-error scatter alone would require."""
    pts = sorted(((err, t, v) for (t, v), (err, _) in scored.items()), key=lambda p: p[0])
    best = pts[0][0]
    labels = [f"{DISPLAY[t]} / {v}" for _, t, v in pts]
    errors = [e for e, _, _ in pts]
    colors = [COLORS[t] for _, t, _ in pts]
    edgecolors = ["gold" if e == best else "black" for e in errors]
    linewidths = [2.5 if e == best else 0.6 for e in errors]

    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    y = np.arange(len(pts))[::-1]  # rank 1 at the top
    bars = ax.barh(y, errors, color=colors, edgecolor=edgecolors, linewidth=linewidths,
                   height=0.72, zorder=3)
    ax.bar_label(bars, fmt="%d", padding=5, fontsize=9.5, fontweight="bold")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Total count error (|IN error| + |OUT error|, both clips) — lower is better")
    ax.set_title("Tracker × Version, ranked by total error")
    ax.margins(x=0.09)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out_dir / "pair_ranking.png", dpi=DPI)
    plt.close(fig)


def scatter_pairs(gt, scored, out_dir=OUT_DIR):
    """Classic research-paper scatter: signed IN-count error (x) vs. signed
    OUT-count error (y) per tracker×version pair, color = tracker, marker
    shape = version. The origin is ground truth - the closer to (0, 0), the
    better. Color+marker together uniquely identify every point, the usual
    paper idiom, so no per-point text labels are needed (that's what caused
    the clutter/overlap in an earlier version of this plot)."""
    fig, ax = plt.subplots(figsize=(7.2, 6.8))
    xs, ys = [], []
    for (t, v), (_, per_clip) in scored.items():
        in_err = sum(per_clip[c][0] - gt[c][0] for c in CLIPS)
        out_err = sum(per_clip[c][1] - gt[c][1] for c in CLIPS)
        xs.append(in_err)
        ys.append(out_err)
        ax.scatter(in_err, out_err, color=COLORS[t], marker=MARKERS[v], s=140,
                   edgecolor="black", linewidth=0.7, alpha=0.9, zorder=3)

    xpad, ypad = (max(xs) - min(xs)) * 0.15 + 1.5, (max(ys) - min(ys)) * 0.15 + 1.5
    ax.axhline(0, color="#999999", linewidth=1, zorder=1)
    ax.axvline(0, color="#999999", linewidth=1, zorder=1)
    ax.scatter(0, 0, marker="*", s=420, color="gold", edgecolor="black",
               linewidth=1, zorder=4, label="ground truth")
    ax.set_xlim(min(0, min(xs)) - xpad, max(0, max(xs)) + xpad)
    ax.set_ylim(min(0, min(ys)) - ypad, max(0, max(ys)) + ypad)
    ax.set_xlabel("IN count error (predicted − true, summed over both clips)")
    ax.set_ylabel("OUT count error (predicted − true, summed over both clips)")
    ax.set_title("Signed IN vs. OUT error, per tracker × version pair")

    tracker_leg = _tracker_legend(ax, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.add_artist(tracker_leg)
    version_handles = [plt.Line2D([], [], marker=MARKERS[v], color="grey", linestyle="",
                                   markersize=9, markeredgecolor="black", markeredgewidth=0.5,
                                   label=v) for v in VERSIONS] + \
        [plt.Line2D([], [], marker="*", color="gold", linestyle="", markersize=14,
                    markeredgecolor="black", label="ground truth")]
    ax.legend(handles=version_handles, title="Version", loc="upper left",
              bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_pairs.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def heatmap_error(scored, out_dir=OUT_DIR):
    grid = np.array([[scored[(t, v)][0] for v in VERSIONS] for t in TRACKERS])
    fig, ax = plt.subplots(figsize=(5.8, 5.3))
    ax.grid(False)  # grid lines make no sense over a cell heatmap
    im = ax.imshow(grid, cmap="RdYlGn_r")
    ax.set_xticks(range(len(VERSIONS)), VERSIONS)
    ax.set_yticks(range(len(TRACKERS)), [DISPLAY[t] for t in TRACKERS])
    for i in range(len(TRACKERS)):
        for j in range(len(VERSIONS)):
            # Light text on the darkest (worst) cells stays readable on dark red.
            txt_color = "white" if grid[i, j] >= grid.max() * 0.75 else "black"
            ax.text(j, i, grid[i, j], ha="center", va="center", fontsize=12,
                    fontweight="bold", color=txt_color)
    ax.set_title("Total error (both clips) — lower is better")
    fig.colorbar(im, ax=ax, label="Total absolute error", shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_error.png", dpi=DPI)
    plt.close(fig)


def bar_ranking(scored, out_dir=OUT_DIR):
    tracker_avg = {t: np.mean([scored[(t, v)][0] for v in VERSIONS]) for t in TRACKERS}
    order = sorted(tracker_avg, key=tracker_avg.get)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.bar([DISPLAY[t] for t in order], [tracker_avg[t] for t in order],
                  color=[COLORS[t] for t in order], edgecolor="black", linewidth=0.7,
                  width=0.62, zorder=3)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Avg total error (v1/v2/v3)")
    ax.set_title("Overall tracker ranking — lower is better")
    ax.margins(y=0.14)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out_dir / "bar_tracker_ranking.png", dpi=DPI)
    plt.close(fig)


def predicted_vs_actual(gt, scored, out_dir=OUT_DIR):
    """Classic ML-paper predicted-vs-actual scatter: total (IN+OUT) crossings per
    clip per pair, against the y=x line."""
    fig, ax = plt.subplots(figsize=(6.8, 6.3))
    truths, preds = [], []
    for (t, v), (_, per_clip) in scored.items():
        for c in CLIPS:
            true_total = sum(gt[c])
            pred_total = per_clip[c][0] + per_clip[c][1]
            truths.append(true_total)
            preds.append(pred_total)
            ax.scatter(true_total, pred_total, color=COLORS[t], marker=MARKERS[v],
                       s=110, edgecolor="black", linewidth=0.6, alpha=0.88, zorder=3)
    lo, hi = min(truths) - 2, max(truths + preds) + 2
    ax.plot([lo, hi], [lo, hi], "--", color="#888888", linewidth=1.3, zorder=2,
            label="perfect prediction")
    ax.set_xlabel("True total crossings (ground truth)")
    ax.set_ylabel("Predicted total crossings")
    ax.set_title("Predicted vs. actual crossings, per clip per pair")
    tracker_handles = [plt.Line2D([], [], marker="o", color=COLORS[t], linestyle="",
                                   markersize=9, markeredgecolor="black", markeredgewidth=0.5,
                                   label=DISPLAY[t]) for t in TRACKERS]
    ax.legend(handles=tracker_handles + [plt.Line2D([], [], linestyle="--", color="#888888",
              label="perfect prediction")], loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "predicted_vs_actual.png", dpi=DPI)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=RESULTS)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt, scored = score_all(args.results_dir)
    bar_pairs_ranking(scored, args.out_dir)
    scatter_pairs(gt, scored, args.out_dir)
    heatmap_error(scored, args.out_dir)
    bar_ranking(scored, args.out_dir)
    predicted_vs_actual(gt, scored, args.out_dir)
    print(f"wrote 5 plots to {args.out_dir}")


if __name__ == "__main__":
    main()
