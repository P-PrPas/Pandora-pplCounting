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


def _diamond_rings(ax, lim, step, fontsize=7.5, labels=True):
    """Dashed L1-ball rings (|x|+|y|=r) centered on the origin - the project's
    actual scoring metric, so two points on the same ring are exactly tied."""
    for r in range(step, int(np.ceil(lim / step)) * step + 1, step):
        ax.plot([r, 0, -r, 0, r], [0, r, 0, -r, 0], "--", color="lightgrey",
                 linewidth=0.9, zorder=1)
        if labels:
            ax.annotate(f"error={r}", (0, r), xytext=(3, 2), textcoords="offset points",
                        fontsize=fontsize, color="grey")


def _plot_points(ax, pts, rank_start, best_l1, numbered):
    """Draws each (in_err, out_err, l1, tracker, version) point. numbered=True
    badges each point with its global rank (1=best); numbered=False draws a
    plain marker only (used for the crowded cluster in the main view, which
    gets its numbers in the zoomed inset instead)."""
    for i, (in_err, out_err, l1, t, v) in enumerate(pts):
        rank = rank_start + i
        is_best = l1 == best_l1
        ax.scatter(in_err, out_err, marker=MARKERS[v], color=COLORS[t],
                   s=210 if is_best else 150,
                   edgecolor="gold" if is_best else "black",
                   linewidth=2.5 if is_best else 0.7, zorder=4)
        if numbered:
            ax.annotate(str(rank), (in_err, out_err), ha="center", va="center",
                        fontsize=7, fontweight="bold", zorder=5,
                        xytext=(9, 9), textcoords="offset points",
                        bbox=dict(boxstyle="circle,pad=0.18", fc="white",
                                  ec=COLORS[t], lw=1.1, alpha=0.95))


def scatter_pairs(gt, scored, out_dir=OUT_DIR):
    """One point per (tracker, version) pair, axes = signed count error
    (predicted - true) per direction, summed over both clips. The origin is
    ground truth itself.

    The project's actual ranking metric is L1 (|IN error| + |OUT error|), not
    Euclidean distance - a plain circular "closer = better" reading disagrees
    with that metric near-ties (e.g. one point closer on X, another closer on
    Y, but equal total error). So distance-from-star is drawn as diamond-
    shaped L1 iso-error rings instead of a Euclidean circle: two points on the
    same ring are *tied*, unambiguously, no eyeballing needed.

    Most pairs cluster tightly near the star while a couple of failure cases
    (McByte++) sit far off - a single fixed-scale view would either crush the
    competitive cluster into an unreadable knot or waste most of the canvas
    on empty space. So the crowded cluster gets its own zoomed inset with
    numbered points (cross-referenced to the ranking list on the right);
    isolated outliers get a direct text label instead, since there's room.
    """
    pts = []
    for (t, v), (_, per_clip) in scored.items():
        in_err = sum(per_clip[c][0] for c in CLIPS) - sum(gt[c][0] for c in CLIPS)
        out_err = sum(per_clip[c][1] for c in CLIPS) - sum(gt[c][1] for c in CLIPS)
        pts.append([in_err, out_err, abs(in_err) + abs(out_err), t, v])
    pts.sort(key=lambda p: p[2])  # rank 1 = lowest L1 error = best

    # ponytail: a couple of pairs land on the exact same (in_err, out_err) -
    # nudge duplicates apart on a tiny circle so both markers stay visible.
    seen = {}
    for p in pts:
        key = (p[0], p[1])
        n = seen.setdefault(key, 0)
        if n:
            angle = n * 2.4
            p[0] += 0.4 * np.cos(angle)
            p[1] += 0.4 * np.sin(angle)
        seen[key] += 1

    best_l1 = pts[0][2]

    # ponytail: split cluster vs. outliers at the single biggest gap in sorted
    # L1 error - a simple elbow heuristic, good enough for a handful of
    # points; revisit with real clustering if the pair count grows a lot.
    l1s = [p[2] for p in pts]
    gaps = [l1s[i + 1] - l1s[i] for i in range(len(l1s) - 1)]
    split = gaps.index(max(gaps)) if gaps and max(gaps) > 2 * (np.median(gaps) or 1) else len(pts) - 1
    cluster, outliers = pts[:split + 1], pts[split + 1:]

    fig, ax = plt.subplots(figsize=(11, 7.5))
    fig.subplots_adjust(right=0.7)

    lim = max(abs(p[0]) for p in pts + [[0, 0]]) or 1
    lim = max(lim, max(abs(p[1]) for p in pts)) * 1.2
    _diamond_rings(ax, lim, step=4)
    ax.axhline(0, color="grey", linewidth=0.8, zorder=1)
    ax.axvline(0, color="grey", linewidth=0.8, zorder=1)
    ax.scatter(0, 0, marker="*", color="gold", s=450, edgecolor="black",
               linewidth=1, zorder=6)
    _plot_points(ax, cluster, 1, best_l1, numbered=False)
    _plot_points(ax, outliers, split + 2, best_l1, numbered=False)
    for i, (in_err, out_err, l1, t, v) in enumerate(outliers):
        ax.annotate(f"{DISPLAY[t]}/{v} ({l1})", (in_err, out_err), fontsize=8.5,
                    xytext=(-6, 14 if i % 2 == 0 else -18), textcoords="offset points",
                    ha="center")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("IN count error (predicted − true, summed over both clips)")
    ax.set_ylabel("OUT count error (predicted − true, summed over both clips)")
    ax.set_title("Tracker × logic-version accuracy\n"
                 "closer to ★ = better · same dashed ring = tied total error")

    if outliers:
        pad = 1.5
        xs, ys = [p[0] for p in cluster], [p[1] for p in cluster]
        x0, x1, y0, y1 = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
        axins = ax.inset_axes([0.03, 0.5, 0.44, 0.44])
        axins.set_xlim(x0, x1)
        axins.set_ylim(y0, y1)
        axins.set_aspect("equal", adjustable="box")
        axins.grid(True, alpha=0.25, linestyle=":")
        inset_lim = max(abs(x0), abs(x1), abs(y0), abs(y1))
        _diamond_rings(axins, inset_lim, step=2, fontsize=6.5)
        if x0 <= 0 <= x1 and y0 <= 0 <= y1:
            axins.scatter(0, 0, marker="*", color="gold", s=200, edgecolor="black", zorder=6)
        _plot_points(axins, cluster, 1, best_l1, numbered=True)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.set_title("cluster detail", fontsize=8.5)
        ax.indicate_inset_zoom(axins, edgecolor="black")

    tracker_handles = [plt.Line2D([], [], marker="o", color=COLORS[t], linestyle="",
                                   markersize=9, label=DISPLAY[t]) for t in TRACKERS]
    version_handles = [plt.Line2D([], [], marker=MARKERS[v], color="grey", linestyle="",
                                   markersize=9, label=v) for v in VERSIONS]
    leg1 = ax.legend(handles=tracker_handles, title="Tracker", loc="lower left", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=version_handles, title="Version", loc="upper right", fontsize=8)

    lines = ["Ranking (total error)", "─" * 24]
    for rank, (_, _, l1, t, v) in enumerate(pts, start=1):
        lines.append(f"{rank:>2}. {DISPLAY[t]}/{v}  ({l1})")
    fig.text(0.72, 0.9, "\n".join(lines), va="top", ha="left",
              family="monospace", fontsize=8.5,
              bbox=dict(boxstyle="round,pad=0.5", fc="#f7f7f7", ec="grey", lw=0.8))
    fig.savefig(out_dir / "scatter_pairs.png", dpi=150)
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
    scatter_pairs(gt, scored, args.out_dir)
    heatmap_error(scored, args.out_dir)
    bar_ranking(scored, args.out_dir)
    predicted_vs_actual(gt, scored, args.out_dir)
    print(f"wrote 4 plots to {args.out_dir}")


if __name__ == "__main__":
    main()
