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
VERSION_COLORS = {v: c for v, c in zip(VERSIONS, plt.cm.Set2.colors)}
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
    labels = [f"{i}.  {DISPLAY[t]} / {v}" for i, (_, t, v) in enumerate(pts, start=1)]
    errors = [e for e, _, _ in pts]
    colors = [COLORS[t] for _, t, _ in pts]
    edgecolors = ["gold" if e == best else "black" for e in errors]
    linewidths = [2.5 if e == best else 0.6 for e in errors]

    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    y = np.arange(len(pts))[::-1]  # rank 1 at the top
    for row in y[::2]:  # zebra striping - easier to trace a row across 15 of them
        ax.axhspan(row - 0.5, row + 0.5, color="#f2f2f2", zorder=0)
    bars = ax.barh(y, errors, color=colors, edgecolor=edgecolors, linewidth=linewidths,
                   height=0.72, zorder=3)
    ax.bar_label(bars, fmt="%d", padding=5, fontsize=9.5, fontweight="bold")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Total count error (|IN error| + |OUT error|, both clips) — lower is better")
    ax.set_title("Tracker × Version, ranked by total error")
    ax.margins(x=0.09)
    ax.set_ylim(y.min() - 0.5, y.max() + 0.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out_dir / "pair_ranking.png", dpi=DPI)
    plt.close(fig)


def _diamond_rings(ax, lim, step, fontsize=7.5, labels=True):
    """Dashed L1-ball rings (|x|+|y|=r) centered on the origin - the project's
    actual scoring metric (see docs/tracking_algo_ground_truth.md), so two
    points on the same ring are exactly tied, no eyeballing required. Drawn
    as a diamond rather than a circle because that's the shape of "equal
    total count error" under this metric - a circle would visually favor
    points that are closer on one axis even when their total error is worse."""
    for r in range(step, int(np.ceil(lim / step)) * step + 1, step):
        ax.plot([r, 0, -r, 0, r], [0, r, 0, -r, 0], "--", color="#bbbbbb",
                 linewidth=1, zorder=1)
        if labels:
            # Labeled on the empty top-right diagonal (+x, +y) - every data
            # point in this dataset sits in the lower-right/upper-left area,
            # so this is the one spot on the ring guaranteed not to collide
            # with a marker, the outlier text labels, or the legends.
            ax.annotate(f"error={r}", (r / 2, r / 2), xytext=(4, 3),
                        textcoords="offset points",
                        fontsize=fontsize, color="#777777")


def _plot_points(ax, pts, rank_start, best_l1, numbered):
    """Draws each (in_err, out_err, l1, tracker, version) point. numbered=True
    badges each point with its global rank (1=best); numbered=False draws a
    plain marker only (used for the crowded cluster in the main view, which
    gets its numbers in the zoomed inset instead)."""
    for i, (in_err, out_err, l1, t, v) in enumerate(pts):
        rank = rank_start + i
        is_best = l1 == best_l1
        ax.scatter(in_err, out_err, marker=MARKERS[v], color=COLORS[t],
                   s=220 if is_best else 160,
                   edgecolor="gold" if is_best else "black",
                   linewidth=2.5 if is_best else 0.8, zorder=4)
        if numbered:
            ax.annotate(str(rank), (in_err, out_err), ha="center", va="center",
                        fontsize=7.5, fontweight="bold", zorder=5,
                        xytext=(9, 9), textcoords="offset points",
                        bbox=dict(boxstyle="circle,pad=0.2", fc="white",
                                  ec=COLORS[t], lw=1.2, alpha=0.97))


def scatter_pairs(gt, scored, out_dir=OUT_DIR):
    """One point per (tracker, version) pair, axes = signed count error
    (predicted - true) per direction, summed over both clips. The origin is
    ground truth itself.

    The project's actual ranking metric is L1 (|IN error| + |OUT error|), not
    Euclidean distance - a plain circular "closer = better" reading disagrees
    with that metric on near-ties (one point closer on X, another closer on
    Y, but equal total error - or, worse, a point that looks close because
    it's near-zero on one axis while actually having a larger total error
    than a point that looks farther away). So distance-from-star is drawn as
    diamond-shaped L1 iso-error rings instead of a Euclidean circle: two
    points on the same ring are *tied*, unambiguously, no eyeballing needed -
    and a point one ring further out is worse, full stop, regardless of
    which axis its error sits on.

    Most pairs cluster tightly near the star while a couple of failure cases
    (McByte++ v1/v2) sit far off - a single fixed-scale view would either
    crush the competitive cluster into an unreadable knot or waste most of
    the canvas on empty space. So the crowded cluster gets its own zoomed
    inset with numbered points (cross-referenced to the ranking list on the
    right); isolated outliers get a direct text label instead, since there's
    room out there for one.
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

    fig, ax = plt.subplots(figsize=(11.5, 7.8))
    fig.subplots_adjust(right=0.7)

    lim = max(abs(p[0]) for p in pts + [[0, 0]]) or 1
    lim = max(lim, max(abs(p[1]) for p in pts)) * 1.2
    _diamond_rings(ax, lim, step=4)
    ax.axhline(0, color="#999999", linewidth=0.9, zorder=1)
    ax.axvline(0, color="#999999", linewidth=0.9, zorder=1)
    ax.scatter(0, 0, marker="*", color="gold", s=480, edgecolor="black",
               linewidth=1, zorder=6)
    _plot_points(ax, cluster, 1, best_l1, numbered=False)
    _plot_points(ax, outliers, split + 2, best_l1, numbered=False)
    for i, (in_err, out_err, l1, t, v) in enumerate(outliers):
        ax.annotate(f"{DISPLAY[t]}/{v}  (error={l1})", (in_err, out_err), fontsize=9,
                    fontweight="bold", xytext=(-8, 16 if i % 2 == 0 else -20),
                    textcoords="offset points", ha="center")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("IN count error (predicted − true, summed over both clips)")
    ax.set_ylabel("OUT count error (predicted − true, summed over both clips)")
    ax.set_title("Tracker × logic-version accuracy\n"
                 "closer to ★ = better  ·  same dashed diamond = tied total error")

    if outliers:
        pad = 1.5
        xs, ys = [p[0] for p in cluster], [p[1] for p in cluster]
        x0, x1, y0, y1 = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
        axins = ax.inset_axes([0.03, 0.5, 0.46, 0.46])
        axins.set_xlim(x0, x1)
        axins.set_ylim(y0, y1)
        axins.set_aspect("equal", adjustable="box")
        axins.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
        inset_lim = max(abs(x0), abs(x1), abs(y0), abs(y1))
        _diamond_rings(axins, inset_lim, step=2, fontsize=7)
        if x0 <= 0 <= x1 and y0 <= 0 <= y1:
            axins.scatter(0, 0, marker="*", color="gold", s=220, edgecolor="black", zorder=6)
        _plot_points(axins, cluster, 1, best_l1, numbered=True)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.set_title("cluster detail (zoomed)", fontsize=9, fontweight="bold")
        ax.indicate_inset_zoom(axins, edgecolor="black")

    tracker_leg = _tracker_legend(ax, loc="lower left")
    ax.add_artist(tracker_leg)
    version_handles = [plt.Line2D([], [], marker=MARKERS[v], color="grey", linestyle="",
                                   markersize=9, markeredgecolor="black", markeredgewidth=0.5,
                                   label=v) for v in VERSIONS]
    ax.legend(handles=version_handles, title="Version", loc="lower right")

    lines = ["Ranking (total error)", "─" * 26]
    for rank, (_, _, l1, t, v) in enumerate(pts, start=1):
        lines.append(f"{rank:>2}. {DISPLAY[t]}/{v}  ({l1})")
    fig.text(0.72, 0.9, "\n".join(lines), va="top", ha="left",
              family="monospace", fontsize=9,
              bbox=dict(boxstyle="round,pad=0.5", fc="#f7f7f7", ec="#999999", lw=0.8))
    fig.text(0.72, 0.06,
              "Note: distance from ★ is total count error\n"
              "(|IN error| + |OUT error|) - the same number\n"
              "used to rank every pair, not straight-line\n"
              "distance. That's why the rings are diamonds,\n"
              "not circles.",
              va="bottom", ha="left", fontsize=8.5, color="#555555", style="italic")
    fig.savefig(out_dir / "scatter_pairs.png", dpi=DPI)
    plt.close(fig)



def heatmap_error(scored, out_dir=OUT_DIR):
    """Rows/columns sorted best-average-first (not alphabetical/fixed order)
    so the best tracker and best version are simply top-left - the reader
    doesn't have to cross-reference a separate ranking table to see it."""
    trackers = sorted(TRACKERS, key=lambda t: np.mean([scored[(t, v)][0] for v in VERSIONS]))
    versions = sorted(VERSIONS, key=lambda v: np.mean([scored[(t, v)][0] for t in TRACKERS]))
    grid = np.array([[scored[(t, v)][0] for v in versions] for t in trackers])
    best = grid.min()

    fig, ax = plt.subplots(figsize=(5.8, 5.3))
    ax.grid(False)  # grid lines make no sense over a cell heatmap
    im = ax.imshow(grid, cmap="RdYlGn_r")
    ax.set_xticks(range(len(versions)), versions)
    ax.set_yticks(range(len(trackers)), [DISPLAY[t] for t in trackers])
    # Thin white separators between cells - without them, same-ish shades of
    # green blend into their neighbors and the grid reads as one blob.
    ax.set_xticks(np.arange(-0.5, len(versions)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(trackers)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i in range(len(trackers)):
        for j in range(len(versions)):
            # Light text on the darkest (worst) cells stays readable on dark red.
            txt_color = "white" if grid[i, j] >= grid.max() * 0.75 else "black"
            ax.text(j, i, grid[i, j], ha="center", va="center", fontsize=12,
                    fontweight="bold", color=txt_color)
            if grid[i, j] == best:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="gold", linewidth=3, zorder=5))
    ax.set_title("Total error (both clips) — lower is better\n"
                 "sorted best → worst, top-left to bottom-right")
    fig.colorbar(im, ax=ax, label="Total absolute error", shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_error.png", dpi=DPI)
    plt.close(fig)


def bar_ranking(scored, out_dir=OUT_DIR):
    """Overall tracker ranking, averaged across v1/v2/v3. Plain bars only -
    the number in the label is the whole story, no per-version breakdown
    needed here (see pair_ranking.png for that level of detail)."""
    tracker_avg = {t: np.mean([scored[(t, v)][0] for v in VERSIONS]) for t in TRACKERS}
    order = sorted(tracker_avg, key=tracker_avg.get)
    best = min(tracker_avg.values())
    edgecolors = ["gold" if tracker_avg[t] == best else "black" for t in order]
    linewidths = [2.5 if tracker_avg[t] == best else 0.7 for t in order]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.bar([DISPLAY[t] for t in order], [tracker_avg[t] for t in order],
                  color=[COLORS[t] for t in order], edgecolor=edgecolors, linewidth=linewidths,
                  width=0.6, zorder=3)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Avg total error (v1/v2/v3)")
    ax.set_title("Overall tracker ranking — lower is better")
    ax.margins(y=0.14)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out_dir / "bar_tracker_ranking.png", dpi=DPI)
    plt.close(fig)


def bar_version_ranking(scored, out_dir=OUT_DIR):
    """Same idea as bar_tracker_ranking.png but averaged the other way:
    one bar per logic version (v1/v2/v3), averaged across all 5 trackers."""
    version_avg = {v: np.mean([scored[(t, v)][0] for t in TRACKERS]) for v in VERSIONS}
    order = sorted(version_avg, key=version_avg.get)
    best = min(version_avg.values())
    edgecolors = ["gold" if version_avg[v] == best else "black" for v in order]
    linewidths = [2.5 if version_avg[v] == best else 0.7 for v in order]

    fig, ax = plt.subplots(figsize=(6, 4.8))
    bars = ax.bar(order, [version_avg[v] for v in order],
                  color=[VERSION_COLORS[v] for v in order], edgecolor=edgecolors,
                  linewidth=linewidths, width=0.5, zorder=3)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Avg total error (across all 5 trackers)")
    ax.set_title("Overall logic-version ranking — lower is better")
    ax.margins(y=0.14)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out_dir / "bar_version_ranking.png", dpi=DPI)
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
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")  # equal axes = the y=x line is visually 45°, not skewed
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
    bar_version_ranking(scored, args.out_dir)
    predicted_vs_actual(gt, scored, args.out_dir)
    print(f"wrote 6 plots to {args.out_dir}")


if __name__ == "__main__":
    main()
