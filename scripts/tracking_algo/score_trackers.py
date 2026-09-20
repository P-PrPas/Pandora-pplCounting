"""Scores every tracker x logic-version combo against the manually-reviewed
ground truth (scripts/tracking_algo/ground_truth.csv), and ranks them overall
and as tracker+version pairs. See docs/tracking_algo_ground_truth.md for the
full methodology writeup.

Usage: python3 scripts/tracking_algo/score_trackers.py [--results-dir DIR]
(default results dir = data/results/tracking_algo, ZoneCounter debounce_frames=15)
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE.parent.parent / "data" / "results" / "tracking_algo"
TRACKERS = ["bytetrack", "botsort", "ocsort", "deepocsort", "mcbyteplusplus"]
VERSIONS = ["v1", "v2", "v3"]
CLIPS = ["clip1", "clip2"]


def ground_truth():
    totals = defaultdict(lambda: [0, 0])  # clip -> [in, out]
    for row in csv.DictReader(open(HERE / "ground_truth.csv")):
        if row["direction"] == "none":
            continue
        totals[row["clip"]][0 if row["direction"] == "in" else 1] += int(row["count"])
    return totals


def load_counts(tracker, version, clip, results_dir=RESULTS):
    rows = list(csv.DictReader(open(results_dir / tracker / version / f"{clip}.csv")))
    ins = sum(1 for r in rows if r["direction"] == "in")
    outs = sum(1 for r in rows if r["direction"] == "out")
    return ins, outs


def score_all(results_dir=RESULTS):
    """Returns (gt, scored) where scored[(tracker, version)] = (total_err, per_clip)
    and per_clip[clip] = (pred_in, pred_out, err). Shared with visualize.py.
    results_dir lets callers score any run_comparison.py output tree (e.g. a
    non-default --debounce sweep) against the same ground truth."""
    gt = ground_truth()
    scored = {}
    for t in TRACKERS:
        for v in VERSIONS:
            err = 0
            per_clip = {}
            for c in CLIPS:
                ins, outs = load_counts(t, v, c, results_dir)
                gi, go = gt[c]
                e = abs(ins - gi) + abs(outs - go)
                err += e
                per_clip[c] = (ins, outs, e)
            scored[(t, v)] = (err, per_clip)
    return gt, scored


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=RESULTS)
    args = p.parse_args()
    gt, scored = score_all(args.results_dir)

    print(f"Ground truth: clip1 IN={gt['clip1'][0]} OUT={gt['clip1'][1]}   "
          f"clip2 IN={gt['clip2'][0]} OUT={gt['clip2'][1]}\n")

    print("=== Best pair (tracker + logic version), lower error = better ===")
    for (t, v), (err, per_clip) in sorted(scored.items(), key=lambda kv: kv[1][0]):
        c1, c2 = per_clip["clip1"], per_clip["clip2"]
        print(f"{err:2d}  {t:15s} {v}   clip1 IN/OUT={c1[0]}/{c1[1]} (err {c1[2]})   "
              f"clip2 IN/OUT={c2[0]}/{c2[1]} (err {c2[2]})")

    print("\n=== Overall tracker ranking (avg error across v1/v2/v3) ===")
    tracker_avg = {t: sum(scored[(t, v)][0] for v in VERSIONS) / len(VERSIONS) for t in TRACKERS}
    for t, avg in sorted(tracker_avg.items(), key=lambda kv: kv[1]):
        print(f"{avg:5.2f}  {t}")

    print("\n=== Overall logic-version ranking (avg error across trackers) ===")
    version_avg = {v: sum(scored[(t, v)][0] for t in TRACKERS) / len(TRACKERS) for v in VERSIONS}
    for v, avg in sorted(version_avg.items(), key=lambda kv: kv[1]):
        print(f"{avg:5.2f}  {v}")


if __name__ == "__main__":
    main()
