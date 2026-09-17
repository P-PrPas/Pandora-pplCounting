# Doorway People Counter — POC

Automated in/out people counting at a checkpoint doorway: YOLO11 + ByteTrack,
foot-point tracking, dual-zone crossing detection, and a customer-facing
annotated video with live counts.

## Layout

- `scripts/v1/` — the original batch pipeline. `people_counter.py` (detection, tracking,
  counting), `presentation.py` (video rendering), and `counting.py`
  (`ZoneCounter` state machine + zone classification — the shared core also
  used by `app/`), plus local config/weights (`bytetrack_custom.yaml`,
  `yolo11s.pt`). Tracks people by foot point (bbox bottom-center).
- `scripts/v2/` — head-tracked variant: same detection+tracking, but classifies
  by neck point (midpoint of the shoulder keypoints from a YOLO-pose model)
  instead of the foot point. The zone polygons are still drawn by the user at
  foot level; a one-time offline calibration step (`calibrate_zones.py`)
  measures this camera's real foot→neck offset from reference footage and
  shifts the zones to head level, caching the result to a JSON file. The
  runtime script (`people_counter_v2.py`) just loads that file and runs a
  single tracking pass — no whole-clip pre-scan, so it also works against a
  live/RTSP source, not just a finished recording. The annotated output
  shows both the user-drawn (foot) and calibrated (head) polygons.
- `scripts/v3/` — head-detected variant: instead of inferring a head-level
  point from a foot detection (v1) or a pose model's neck point (v2), this
  runs a dedicated head detector (`head_detector.pt`, a pretrained YOLOv8n
  checkpoint trained on SCUT-HEAD by a third party — no training done here)
  and tracks its boxes directly. Same "draw zones at foot level, calibrate
  once, run against head-level zones" split as v2 (`calibrate_zones_v3.py` /
  `people_counter_v3.py`), except the foot→head offset is measured by
  pairing v1's person detector with this head detector on the same frames
  (`head_calibration_v3.py`), since a plain head detector never observes a
  foot and a head together in one detection the way a pose model does.
  Reuses v2's per-vertex IDW interpolation unchanged — same fix that keeps
  adjacent zones from overlapping after the shift. Also runs v1's person
  detector every frame at runtime as a filter: a detected "head" only counts
  if a person box sits under it, which is what actually excludes a known
  false-positive hotspot (a static wall sign this model confidently misreads
  as a head) instead of a hardcoded per-camera pixel region — see
  `head_tracker_v3.py`.
- `app/` — live desktop POC (PySide6). Draw the Enter/Exit zones on a camera
  snapshot, then run real-time detection/tracking/counting against an RTSP
  stream. See "Live desktop app" below.
- `tests/` — self-checks for the counting state machine and the renderer.
  No video or model download required; run directly with `python3`.
- `docs/` — `REPORT.md` (results + edge-case evaluation + proposed
  enhancements) and `PRESENTATION.md` (render/output notes).
- `assets/` — bundled fonts used by the renderer.
- `data/` — not tracked upstream; local working directory.
  - `dataset/` — raw source clips.
  - `results/v<N>/` — rendered outputs for a given pipeline version (current: `v1`).

## Quickstart

Run everything from the repo root:

```bash
python3 tests/test_zone_counter.py
python3 tests/test_presentation.py
python3 tests/test_v2_head_calibration.py
python3 scripts/v1/people_counter.py data/dataset/<clip>.mp4 data/results/v1/<output_prefix>

# v2 (head-tracked): calibrate once per camera, then run
python3 scripts/v2/calibrate_zones.py scripts/v2/zones_head.json data/dataset/<clip1>.mp4 [<clip2>.mp4 ...]
python3 scripts/v2/people_counter_v2.py scripts/v2/zones_head.json data/dataset/<clip>.mp4 data/results/v2/<output_prefix>

# v3 (head-detected): calibrate once per camera, then run
python3 scripts/v3/calibrate_zones_v3.py scripts/v3/zones_head_v3.json data/dataset/<clip1>.mp4 [<clip2>.mp4 ...]
python3 scripts/v3/people_counter_v3.py scripts/v3/zones_head_v3.json data/dataset/<clip>.mp4 data/results/v3/<output_prefix>
```

See `docs/REPORT.md` for results and `docs/PRESENTATION.md` for render/export
details.

### Live desktop app

```bash
pip install --break-system-packages -r app/requirements.txt
cp .env.example .env   # set RTSP_URL to your camera stream
python3 app/main.py
```

Draw the Enter zone (click points, "Finish zone"), then the Exit zone, then
"Continue" and "Start Live Count". Zones are cached to `app/zones.json` per
RTSP source so a relaunch against the same camera skips redrawing.

On the run page, pick a **Logic** preset — V1 (foot tracking), V2 (head
tracking), or V3 (head detection) — before starting. V2/V3 each need a
one-time-per-camera calibration: click "Calibrate" and let it sample
~20-30s of the live stream (people actually walking through frame), then
"Start Live Count" unlocks. Calibration samples are cached to
`app/calibration_samples.json` per RTSP source and logic (V2's neck offset
and V3's head offset aren't interchangeable, so switching presets on the
same camera doesn't require redoing the other's calibration), so redrawing
zones or relaunching against the same camera doesn't require recalibrating
— only a new camera or an explicit "Recalibrate" does. V3 also runs a
second model (the person detector) every frame, both during calibration and
live counting, for its overlap filter — see `scripts/v3/`.
