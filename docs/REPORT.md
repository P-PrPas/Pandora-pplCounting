# Doorway In/Out Counter — POC Results & Evaluation

Scene: overhead fisheye camera above a checkpoint with two walk-through metal
detectors. Zones use the customer-supplied, perspective-fitted quads sitting
directly at the checkpoint chokepoint: Zone A ("outside", cyan) covers the
detector lane, Zone B ("inside", magenta) covers the corridor threshold just
beyond it. A→B = **in**, B→A = **out**.

## Results

| Clip | Duration | IN | OUT | Events |
|---|---|---|---|---|
| `vlc-record-...11h38m09s` (11:30–13:00) | 90s | 14 | 28 | `data/clip1_counted.csv` |
| `vlc-record-...13h47m04s` (01:45–02:05) | 20s | 8 | 5 | `data/clip2_counted.csv` |

Both counts were spot-checked against the annotated video frame-by-frame at
each event timestamp — in every case the crowd's actual direction of travel
matched the logged direction, including the fast (<0.5s) transitions: with
the zones now sitting flush against each other at the chokepoint, a person
walking straight through registers as a crossing within a handful of frames,
and a person who hesitates or half-turns right at the boundary produces a
genuine out→in→out sequence rather than a bogus one (confirmed visually for
track 221 in clip 1 and track 122 in clip 2). Traffic is bidirectional at
this location, unlike the previous (incorrect) zone placement.

## Edge cases observed

- **Fast crossings at a flush zone boundary.** With the corrected zones
  butted directly against each other at the chokepoint, a person walking
  straight through registers a confirmed transition within a few hundred ms
  (e.g. clip 1 track 72, clip 2 track 122) — verified frame-by-frame to be
  real, continuous walk-throughs, not noise. `DEBOUNCE_FRAMES=4` is low
  enough to catch these without misfiring on jitter (see
  `test_zone_counter.py`).
- **Genuine hesitation produces genuine oscillation.** Track 221 in clip 1
  registers out→in→out within ~2.3s; the frames show the person actually
  stopping and turning right at the boundary line before finally leaving —
  the debounce logic correctly reports this as three real events rather than
  collapsing or dropping them.
- **Occlusion can still break a track mid-crossing, mitigated (not eliminated).**
  Dense, identically-uniformed clusters fully blocking each other's boxes was
  the main open risk after the first zone revision. Fixed two params, both
  in `people_counter.py`/`bytetrack_custom.yaml`, no new dependency:
  - `conf=0.1` instead of `0.25` on `model.track()` — ByteTrack is designed
    to associate low-score detections in a second matching pass, but at
    `conf=0.25` those boxes were filtered out before ever reaching it.
  - `track_buffer: 60` instead of the default `30` — a fully-hidden track
    now survives ~2.4s of total occlusion (was ~1.2s) before ByteTrack gives
    up and would assign a new ID on reappearance.
  Residual risk: a person hidden longer than that, or an ID *swap* between
  two adjacent tracks (no appearance signal), can still happen — full
  elimination needs ReID or a doorway-scoped re-acquisition heuristic (still
  proposed below, not implemented).
- **Lingering near the doorway.** The security guard standing beside the
  checkpoint for the full clip 1 duration correctly never triggers a count —
  they stay classified in one zone the whole time.
- **Reverse-direction-midway.** Covered by the hesitation case above in real
  footage, and by construction in the debounce logic for the case where a
  track never reaches the threshold at all (verified in
  `test_zone_counter.py`).

## Proposed architectural enhancements (not implemented — POC scope)

1. **Tripwire line instead of polygon pair.** The doorway is physically
   narrow; a single line-crossing test (displacement vector sign change
   across a line) is less sensitive to zone-depth tuning than two polygons
   and is the more standard approach when the chokepoint itself is this
   thin. Worth it if false debounce-timing sensitivity shows up at scale.
2. **Fisheye undistortion / homography calibration.** Would straighten the
   chokepoint region, shrink the barrel distortion that's worst at exactly
   the area we care about, and make foot-point-to-real-world distance
   consistent — a real accuracy lever, skipped here because it adds a
   calibration step for a single fixed camera in a POC.
3. **Doorway-scoped re-acquisition.** Rather than a global appearance ReID
   (which uniforms defeat anyway), a cheap heuristic scoped *only* to the
   small chokepoint ROI — bridge a lost track to a new one appearing within a
   few frames and a few pixels — would recover the occlusion/ID-switch case
   above with much lower mismatch risk than a full ReID system.
4. **Higher-resolution inference crop at the doorway.** Running a second,
   cropped/upscaled inference pass on just the doorway ROI would improve
   recall on the small/backlit figures there, at the cost of one extra
   (cheap, small-crop) forward pass per frame.
