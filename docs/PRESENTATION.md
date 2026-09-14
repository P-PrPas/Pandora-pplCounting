# People flow — customer presentation

The presentation uses a 2560 × 1280 canvas with the complete 1920 × 1080 camera
image, no top header, a separate analytics rail, and an unobstructed crossing area. DejaVu Sans
regular/bold is bundled with its license so rendering is consistent across machines.

- Mint: inside / entered. Amber: outside / exited. Bright cyan: tracked person.
- High-contrast bounding boxes with dark casing and corner accents, dark ID badges that move apart when overlapping, short motion trails,
  and a direction-colored highlight immediately after a confirmed crossing.
- Cumulative entered/exited totals, signed net flow, the five latest crossings,
  active tracks, and elapsed playback time. All values come from the tracker.
- Net flow is **not occupancy**; totals represent crossing events, not unique people.
- The footage is explicitly labeled as a recorded proof of concept.

## Render

Use the Python environment that contains OpenCV, NumPy, Pillow, PyTorch and
Ultralytics (in this workspace, `/usr/bin/python3`, not the default `python`).
Run from the repo root:

```bash
/usr/bin/python3 scripts/people_counter.py \
  data/dataset/vlc-record-2026-09-04-11h38m09s-trimmed.mp4 \
  data/results/v1/clip1_presentation
/usr/bin/python3 scripts/people_counter.py \
  data/dataset/vlc-record-2026-09-04-13h47m04s-trimmed.mp4 \
  data/results/v1/clip2_presentation
/usr/bin/python3 tests/test_zone_counter.py
/usr/bin/python3 tests/test_presentation.py
```

The counting zones, detector settings, tracking settings and counting state machine
are unchanged. Coordinates remain calibrated for this particular camera framing.
The original `data/results/v1/clip1_counted.mp4` remains available for comparison.

For browser/presentation compatibility, convert the rendered MPEG-4 file to H.264:

```bash
ffmpeg -i data/results/v1/clip1_presentation.mp4 -c:v libx264 -crf 18 \
  -preset veryfast -threads 4 -pix_fmt yuv420p -movflags +faststart -an \
  data/results/v1/clip1_showcase.mp4
```

## Delivered verification

Both `data/results/v1/clip1_showcase.mp4` and `data/results/v1/clip2_showcase.mp4` are H.264 presentation
exports. Their matching `_preview.jpg` files show actual frames at 60 and 10 seconds.
Each event CSV is checked byte for byte against its original `_counted.csv`:
clip 1 has 14 entered / 28 exited; clip 2 has 8 entered / 5 exited.
Both the counting and presentation self-checks pass, including bounding-box contrast
and non-overlapping badge placement. Use the same H.264 conversion command with
`clip2` to export the second clip after rendering.
