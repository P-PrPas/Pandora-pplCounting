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

### Biggest source of remaining tracker overcounting: boundary-dwell jitter

A person standing still exactly on the A/B boundary line — scanning a badge,
adjusting a mask, hanging a jacket on a wall hook — can have their crossing
point jitter back and forth across the line and re-fire a "crossing" with no
real movement. `ZoneCounter`'s `DEBOUNCE_FRAMES=15` (0.6s) absorbs most of
this, but it's still the dominant reason any tracker's raw count runs above
the true count, especially for trackers that hold onto an ID longest through
the dwell (McByte++'s Re-ID keeps the dwelling person's ID alive longer than
the others, so it re-fires the bug more).

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
| 1 | OC-SORT | 9.33 |
| 2 | BoT-SORT | 10.00 |
| 3 | ByteTrack | 10.67 |
| 3 | DeepOCSORT | 10.67 |
| 5 | McByte++ | 19.67 |

McByte++ ranks last despite being the most sophisticated tracker (Re-ID,
mask-aware) — because it keeps IDs alive through the exact scenario
(stationary dwell at the boundary) that the counting logic's debounce
punishes hardest. Its tracking quality isn't necessarily worse than the
others'; it's just better at exposing this weakness, not at avoiding it.
The gap between the top four trackers is small relative to clip-to-clip
noise — don't over-read fine differences among them.

### Overall logic-version ranking (averaged across trackers)

| Rank | Version | Avg error |
|---|---|---|
| 1 | v3 (head detector) | 10.40 |
| 2 | v1 (foot point) | 12.00 |
| 3 | v2 (pose/neck) | 13.80 |

### Best pair (tracker x logic version), full ranking

| Rank | Tracker | Version | Error |
|---|---|---|---|
| 1 | OC-SORT | v1 | 8 |
| 1 | OC-SORT | v3 | 8 |
| 3 | BoT-SORT | v1 | 9 |
| 3 | DeepOCSORT | v1 | 9 |
| 5 | ByteTrack | v1 | 10 |
| 5 | ByteTrack | v3 | 10 |
| 5 | BoT-SORT | v2 | 10 |
| 8 | BoT-SORT | v3 | 11 |
| 8 | DeepOCSORT | v2 | 11 |
| 8 | McByte++ | v3 | 11 |
| 11 | ByteTrack | v2 | 12 |
| 11 | OC-SORT | v2 | 12 |
| 11 | DeepOCSORT | v3 | 12 |
| 14 | McByte++ | v1 | 24 |
| 14 | McByte++ | v2 | 24 |

**Best pair: OC-SORT + v1**, tied with OC-SORT + v3. Unlike the earlier v1
foot-point sweep, v3's head-detector point is now competitive (best on
average) once boundary jitter isn't dominating — a more accurate crossing
point starts to pay off instead of being swamped by dwell noise. McByte++'s
own v3 pairing (error 11) is close behind and far ahead of its v1/v2 pairings,
suggesting v3's cleaner crossing point also helps offset its stronger,
bug-exposing ID retention.

See `docs/tracking_algo_report.docx` for the full visual report (ranked bar
chart, heatmap, per-tracker bar chart, predicted-vs-actual) generated from
this same scoring pipeline via `scripts/tracking_algo/visualize.py` +
`scripts/tracking_algo/build_report.py`. An earlier report comparing
`DEBOUNCE_FRAMES=4` vs `15` (the tuning work behind the value used above) is
kept for reference at `docs/tracking_algo_debounce_experiment_report.docx`.
