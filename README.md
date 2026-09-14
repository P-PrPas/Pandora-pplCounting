# Doorway People Counter — POC

Automated in/out people counting at a checkpoint doorway: YOLO11 + ByteTrack,
foot-point tracking, dual-zone crossing detection, and a customer-facing
annotated video with live counts.

## Layout

- `scripts/` — the batch pipeline. `people_counter.py` (detection, tracking,
  counting), `presentation.py` (video rendering), and `counting.py`
  (`ZoneCounter` state machine + zone classification — the shared core also
  used by `app/`), plus local config/weights (`bytetrack_custom.yaml`,
  `yolo11s.pt`).
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
python3 scripts/people_counter.py data/dataset/<clip>.mp4 data/results/v1/<output_prefix>
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
