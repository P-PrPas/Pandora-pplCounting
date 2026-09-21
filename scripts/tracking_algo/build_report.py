"""Assembles docs/tracking_algo_report.docx from the ground-truth scoring
(score_trackers.py) and the plots in docs/tracking_algo_plots/ (visualize.py).
Run visualize.py first if the plots don't exist yet.

Usage: python3 scripts/tracking_algo/build_report.py
"""
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

sys.path.insert(0, str(Path(__file__).parent))
from score_trackers import CLIPS, TRACKERS, VERSIONS, score_all  # noqa: E402

DOCS = Path(__file__).parent.parent.parent / "docs"
PLOTS = DOCS / "tracking_algo_plots"
DISPLAY = {"bytetrack": "ByteTrack", "botsort": "BoT-SORT", "ocsort": "OC-SORT",
           "deepocsort": "DeepOCSORT", "mcbyteplusplus": "McByte++"}


def add_table(doc, headers, rows):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    for cell, h in zip(table.rows[0].cells, headers):
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
    for row in rows:
        cells = table.add_row().cells
        for cell, v in zip(cells, row):
            cell.text = str(v)
    return table


def add_figure(doc, filename, caption):
    doc.add_picture(str(PLOTS / filename), width=Inches(6))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph(caption)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.runs[0].italic = True
    p.runs[0].font.size = Pt(10)


def main():
    gt, scored = score_all()
    doc = Document()

    doc.add_heading("Tracking-Algorithm Ground Truth & Comparison Report", level=0)
    doc.add_paragraph(
        "Comparison of 5 trackers (ByteTrack, BoT-SORT, OC-SORT, DeepOCSORT, "
        "McByte++) across 3 counting-logic versions (v1 foot-point, v2 pose/"
        "neck, v3 head-detector), scored against manually reviewed ground "
        "truth for the two dataset clips."
    )

    doc.add_heading("1. Methodology", level=1)
    doc.add_paragraph(
        "Ground truth was built by visually reviewing both clips frame-by-frame "
        "against data/results/tracking_algo/bytetrack/v1/clip{1,2}.mp4 (the same "
        "raw footage every tracker ran on; this render simply has the A/B zone "
        "polygons burned in for reference). Candidate moments to check were found "
        "by unioning all 5 trackers' own event frames and clustering nearby ones "
        "— used only to bound review scope, never trusted as truth. Each "
        "cluster was independently judged by sustained body motion through the "
        "doorway (growing + approaching camera = OUT, shrinking + receding = IN), "
        "ignoring track IDs (they churn under occlusion) and each tracker's own "
        "flashed direction tag (that's exactly what's being verified). Full "
        "per-cluster detail: scripts/tracking_algo/ground_truth.csv."
    )

    doc.add_heading("2. Ground Truth", level=1)
    add_table(doc, ["Clip", "Genuine IN", "Genuine OUT", "Total"],
              [[c, gt[c][0], gt[c][1], sum(gt[c])] for c in CLIPS])
    doc.add_paragraph(
        "Confidence note: clip1's opening ~8 seconds (frames 64-224) is the "
        "least certain window — a dense, fully-occluded cluster of "
        "identically-uniformed people; the true OUT count there is a best "
        "estimate with a plausible range of 3-5. All other windows are "
        "medium-high to high confidence."
    )

    doc.add_heading("3. Key Finding: Debounce-Dwell Artifacts", level=1)
    doc.add_paragraph(
        "The dominant remaining source of overcounting is boundary-dwell "
        "jitter: a person standing still exactly on the A/B line (scanning a "
        "badge, adjusting a mask, hanging a jacket on a wall hook) has their "
        "crossing point jitter back and forth across the line and re-fires a "
        "“crossing” with no real movement. ZoneCounter's DEBOUNCE_FRAMES=15 "
        "(0.6s) absorbs most of this, but it still shows up worst for "
        "trackers that hold an ID alive longest through the dwell (McByte++'s "
        "Re-ID), since they get more chances to re-trigger it."
    )

    doc.add_heading("4. Visualizations", level=1)
    add_figure(doc, "pair_ranking.png",
               "Figure 1. Every tracker×version pair ranked by total count "
               "error (both clips combined), shortest bar = lowest error = "
               "best. Gold-edged bars mark the best pair(s); equal-length "
               "bars with equal printed numbers are exact ties.")
    add_figure(doc, "scatter_pairs.png",
               "Figure 2. Signed IN vs. OUT count error per pair (color = "
               "tracker, shape = version); the gold star is ground truth. "
               "Distance from the star is drawn as dashed diamonds, not "
               "circles, because the ranking metric is total count error "
               "(|IN error| + |OUT error|) — points on the same diamond are "
               "exact ties regardless of which axis their error sits on. The "
               "13 competitive pairs are zoomed into the inset (numbered by "
               "rank, cross-referenced to the list on the right); McByte++ "
               "v1/v2 are the two labeled outliers. Nearly every pair "
               "overcounts IN (boundary-dwell jitter, Section 3) but "
               "undercounts OUT (lost tracks in clip1's dense, occluded "
               "opening cluster) — McByte++ is the exception, its stronger "
               "Re-ID keeps IDs alive through the dwell so it also "
               "overcounts OUT slightly instead of losing them.")
    add_figure(doc, "predicted_vs_actual.png",
               "Figure 3. Predicted vs. true total crossings per clip per pair, "
               "against the y=x “perfect prediction” line. Points above the "
               "line overcount; McByte++ overcounts most on the busy clip.")
    add_figure(doc, "heatmap_error.png",
               "Figure 4. Total absolute error (both clips combined) for every "
               "tracker×version combination; rows and columns sorted best-average"
               "-first, so the best combinations cluster top-left. Greener = "
               "closer to ground truth; the gold outline marks the overall best "
               "cell(s).")
    add_figure(doc, "bar_tracker_ranking.png",
               "Figure 5. Overall tracker ranking: bar height is the average "
               "total error across v1/v2/v3, gold-outlined for the best; the "
               "three markers on top of each bar are that version's own error, "
               "showing whether the average hides a spread (as for McByte++'s "
               "v1/v2 vs. v3) or reflects three similar results.")

    doc.add_heading("5. Rankings", level=1)
    doc.add_heading("5.1 Overall tracker ranking (avg error across v1/v2/v3)", level=2)
    tracker_avg = sorted(
        ((t, sum(scored[(t, v)][0] for v in VERSIONS) / len(VERSIONS)) for t in TRACKERS),
        key=lambda kv: kv[1])
    add_table(doc, ["Rank", "Tracker", "Avg error"],
              [[i + 1, DISPLAY[t], f"{avg:.2f}"] for i, (t, avg) in enumerate(tracker_avg)])

    doc.add_heading("5.2 Overall logic-version ranking (avg error across trackers)", level=2)
    version_avg = sorted(
        ((v, sum(scored[(t, v)][0] for t in TRACKERS) / len(TRACKERS)) for v in VERSIONS),
        key=lambda kv: kv[1])
    add_table(doc, ["Rank", "Version", "Avg error"],
              [[i + 1, v, f"{avg:.2f}"] for i, (v, avg) in enumerate(version_avg)])

    doc.add_heading("5.3 Best pair (tracker + logic version)", level=2)
    pairs = sorted(scored.items(), key=lambda kv: kv[1][0])
    add_table(doc, ["Rank", "Tracker", "Version", "Total error"],
              [[i + 1, DISPLAY[t], v, err] for i, ((t, v), (err, _)) in enumerate(pairs)])

    doc.add_heading("6. Conclusion", level=1)
    best_tracker, best_tracker_avg = tracker_avg[0]
    best_version, best_version_avg = version_avg[0]
    best_pair = pairs[0][0]
    doc.add_paragraph(
        f"Overall best tracker: {DISPLAY[best_tracker]} (avg error "
        f"{best_tracker_avg:.2f}). Overall best logic version: {best_version} "
        f"(avg error {best_version_avg:.2f}). Best pair: "
        f"{DISPLAY[best_pair[0]]} + {best_pair[1]} (error {pairs[0][1][0]}). "
        "McByte++ ranks last despite being the most sophisticated tracker here "
        "(Re-ID, mask-aware) — its stronger ID retention keeps a dwelling "
        "person's identity alive longer, giving the shared counting logic's "
        "boundary-jitter weakness (Section 3) more chances to re-trigger; its "
        "tracking quality isn't necessarily worse, it's just better at exposing "
        "that bug. The gap between the top four trackers is small relative to "
        "clip-to-clip noise — don't over-read fine differences among them."
    )

    out_path = DOCS / "tracking_algo_report.docx"
    doc.save(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
