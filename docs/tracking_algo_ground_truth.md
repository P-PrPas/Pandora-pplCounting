# Tracking-Algorithm Ground Truth & Ranking

Manually-reviewed ground truth for how many people genuinely crossed the A/B
doorway boundary in each dataset clip, built to score the 5-tracker x
3-logic-version comparison (see `docs/REPORT.md` for the original v1 POC,
`scripts/tracking_algo/` for the comparison harness).

## Methodology

Reviewed frame-by-frame using `data/results/tracking_algo/bytetrack/v1/clip{1,2}.mp4`
as the visual reference (same raw footage all 5 trackers ran on; this render
just has the A/B zone polygons burned in). Candidate moments to check were
found by unioning all 5 trackers' own event frame numbers and clustering
consecutive events within ~30 frames — used only as a scope-bounding index,
never as ground truth itself, to avoid circularity. Each cluster was then
independently judged from `scripts/tracking_algo/cluster_view.py` filmstrips
(cropped/tiled frame sequences) by actual sustained body motion through the
doorway (growing+approaching camera = **out**, shrinking+receding = **in**),
ignoring track ID labels (they churn/switch constantly under occlusion) and
ignoring each tracker's own flashed direction tag (that's exactly what's
being verified).

Full per-cluster breakdown: `scripts/tracking_algo/ground_truth.csv`.
Reproducible scoring: `scripts/tracking_algo/score_trackers.py`.

### Biggest source of tracker overcounting: debounce-dwell artifacts

`ZoneCounter`'s `DEBOUNCE_FRAMES=4` (0.16s) is short enough that a person
standing still exactly on the A/B boundary line — scanning a badge, adjusting
a mask, hanging a jacket on a wall hook — gets their foot-point jitter across
the line and re-fires a "crossing" repeatedly with no real movement. This
pattern recurred many times across both clips (e.g. clip1 frames 416, 466,
1039, 1074, 2070-2102; clip2 frames 60-130) and is the dominant reason every
tracker's raw count runs higher than the true count, especially for trackers
that hold onto an ID longer (McByte++'s Re-ID keeps the dwelling person's ID
alive, so it re-fires the debounce bug more than trackers that lose the ID to
occlusion first).

## Ground truth

| Clip | Genuine IN | Genuine OUT | Total |
|---|---|---|---|
| clip1 (~90s, busy) | 8 | 21 | 29 |
| clip2 (~20s, quiet) | 2 | 6 | 8 |

Confidence notes: clip1's opening 8 seconds (frames 64-224) is the least
certain window — a dense, fully-occluded cluster of same-uniform people;
the true OUT count there is a best estimate of 3, plausible range 3-5. All
other windows are medium-high to high confidence. See `ground_truth.csv` for
per-cluster confidence and rationale.

## Scoring

Error = `|predicted_IN - true_IN| + |predicted_OUT - true_OUT|`, summed
across both clips. Lower is better. Ground truth is the same for all 3 logic
versions (it's the same physical people crossing the same physical doorway;
only the automated counting method differs).

### Overall tracker ranking (averaged across v1/v2/v3)

| Rank | Tracker | Avg error |
|---|---|---|
| 1 | OC-SORT | 14.3 |
| 1 | DeepOCSORT | 14.3 |
| 3 | BoT-SORT | 15.7 |
| 4 | ByteTrack | 16.7 |
| 5 | McByte++ | 30.0 |

McByte++ ranks last despite being the most sophisticated tracker (Re-ID,
mask-aware) — because it keeps IDs alive through the exact scenario
(stationary dwell at the boundary) that the counting logic's debounce weakness
punishes hardest. Its tracking quality isn't necessarily worse than the
others'; it's just better at exposing an existing bug in `ZoneCounter`, not at
avoiding it. OC-SORT and DeepOCSORT come out slightly ahead of BoT-SORT and
ByteTrack, but the gap between the top 4 is small relative to clip-to-clip
noise — don't over-read fine differences among them.

### Overall logic-version ranking (averaged across trackers)

| Rank | Version | Avg error |
|---|---|---|
| 1 | v1 (foot point) | 16.6 |
| 2 | v3 (head detector) | 18.4 |
| 3 | v2 (pose/neck) | 19.6 |

### Best pair (tracker x logic version), full ranking

| Rank | Tracker | Version | Error |
|---|---|---|---|
| 1 | BoT-SORT | v1 | 12 |
| 1 | OC-SORT | v1 | 12 |
| 3 | DeepOCSORT | v1 | 12 |
| 4 | ByteTrack | v1 | 14 |
| 5 | OC-SORT | v3 | 14 |
| 6 | DeepOCSORT | v3 | 15 |
| 7 | ByteTrack | v2 | 16 |
| 7 | BoT-SORT | v2 | 16 |
| 7 | DeepOCSORT | v2 | 16 |
| 10 | OC-SORT | v2 | 17 |
| 11 | BoT-SORT | v3 | 19 |
| 12 | ByteTrack | v3 | 20 |
| 13 | McByte++ | v3 | 24 |
| 14 | McByte++ | v1 | 33 |
| 15 | McByte++ | v2 | 33 |

**Best pair: BoT-SORT + v1** (tied with OC-SORT + v1 and close to DeepOCSORT + v1) —
v1 (simple foot-point zone crossing) wins across the board because it's the
version least exposed to the debounce-dwell bug: v2/v3's head/pose-based
points sit closer to the doorway's vertical midline where dwelling people
linger, so they jitter across the boundary line more than a foot point does.
Fixing `DEBOUNCE_FRAMES` (raise from 4 to ~12-15 frames) is a separate,
not-yet-implemented task that would likely close most of this gap for v2/v3
and for McByte++ specifically, since the issue is in the shared counting
logic, not in any one tracker.

## Debounce experiment: DEBOUNCE_FRAMES 4 → 15

Re-ran the full 5-tracker x 3-version comparison with `DEBOUNCE_FRAMES=15`
(0.6s, long enough to absorb a stationary dwell at the boundary), keeping
results in a separate `data/results/tracking_algo_debounce15/` tree so the
debounce=4 baseline above is untouched. Reproduce with:
`python3 scripts/tracking_algo/run_comparison.py --debounce 15`, score with
`python3 scripts/tracking_algo/score_trackers.py --results-dir data/results/tracking_algo_debounce15`.

| Tracker | Avg error (debounce=4) | Avg error (debounce=15) | Change |
|---|---|---|---|
| OC-SORT | 14.33 | 9.33 | -5.00 |
| BoT-SORT | 15.67 | 10.00 | -5.67 |
| ByteTrack | 16.67 | 10.67 | -6.00 |
| DeepOCSORT | 14.33 | 10.67 | -3.67 |
| McByte++ | 30.00 | 19.67 | -10.33 |

The fix confirms the hypothesis: every tracker's error dropped, McByte++ most
of all in absolute terms (though it still ranks last — its stronger Re-ID
still keeps dwelling IDs alive across a 15-frame window better than the
others, so it still re-triggers the bug more often than trackers that lose
the ID sooner). The version ranking also flips: v3 (head detector) is now
best on average (10.40) instead of v1 (12.00) — with the debounce bug mostly
fixed, v3's more accurate head-based crossing point starts to pay off instead
of being swamped by dwell jitter. New best pair: **OC-SORT + v1** (error 8,
down from 12), with OC-SORT + v3 tied close behind (error 8 as well). Full
before/after per-pair table and chart: `docs/tracking_algo_report.docx`
section 7.

See `docs/tracking_algo_report.docx` for the full visual report (scatter
plot, heatmap, per-tracker bar charts, predicted-vs-actual, and the
debounce before/after comparison) generated from this same scoring pipeline
via `scripts/tracking_algo/visualize.py` + `scripts/tracking_algo/build_report.py`.

