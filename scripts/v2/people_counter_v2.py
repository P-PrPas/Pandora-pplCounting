"""V2: track by neck point (shoulder-keypoint midpoint) instead of foot point.

Same detection+tracking+debounce pipeline as v1, but classification uses a
YOLO-pose model's neck point rather than the bbox-bottom foot point — the
idea being it's less sensitive to stride/pose than the foot.

The zone polygons are still drawn by the user at foot level (same numbers as
v1's ZONE_A/ZONE_B — same physical camera, same doorway). The head-level
polygons actually used for classification come from a *pre-computed*
calibration file (see calibrate_zones.py) rather than being derived from
this run's own footage — the offset is a property of the fixed camera, not
of whatever clip/stream happens to be running, and a live RTSP feed has no
"whole clip" to pre-scan the way a batch calibration pass would need. Run
calibrate_zones.py once per camera (or whenever it moves) to produce that
file. The output video draws both: the user's foot-level polygons (thin
white outline) and the calibrated head-level polygons actually used for
counting (filled, as in v1).

Usage (from repo root):
    python3 scripts/v2/calibrate_zones.py <calibration.json> <ref_clip> [more_clips...]
    python3 scripts/v2/people_counter_v2.py <calibration.json> <input_video> <output_prefix>
"""
import sys
import csv
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

V1_DIR = Path(__file__).parent.parent / "v1"
sys.path.insert(0, str(V1_DIR))
sys.path.insert(0, str(Path(__file__).parent))
from presentation import Presentation                                   # noqa: E402
from counting import ZoneCounter, best_device                           # noqa: E402
from people_counter import ZONE_A as ZONE_A_FOOT, ZONE_B as ZONE_B_FOOT  # noqa: E402
from head_tracker import render_v2_frame                                # noqa: E402

HERE = Path(__file__).parent
MODEL_PATH = HERE / "yolo11s-pose.pt"
TRACKER_CONFIG = V1_DIR / "bytetrack_custom.yaml"

TRAIL_LEN = 30
HIGHLIGHT_FRAMES = 20


def load_head_zones(path):
    data = json.loads(Path(path).read_text())
    return (np.array(data["zone_a_head"], dtype=np.int32),
            np.array(data["zone_b_head"], dtype=np.int32))


def main():
    if len(sys.argv) != 4:
        print(f"usage: python3 {sys.argv[0]} <calibration.json> <input_video> <output_prefix>")
        sys.exit(1)
    calibration_path, src, out_prefix = sys.argv[1], sys.argv[2], sys.argv[3]

    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    cap.release()
    if not w or not h:
        raise ValueError(f"Cannot open video: {src}")

    device = best_device()
    print(f"people_counter_v2: running inference on device={device}")

    zone_a_head, zone_b_head = load_head_zones(calibration_path)
    print(f"people_counter_v2: loaded head-level zones from {calibration_path}")
    print(f"  zone A (outside) vertices shifted by avg {(zone_a_head - ZONE_A_FOOT).mean(axis=0).round(1)} px")
    print(f"  zone B (inside)  vertices shifted by avg {(zone_b_head - ZONE_B_FOOT).mean(axis=0).round(1)} px")

    presentation = Presentation(duration, footer='Recorded footage · AI-assisted counting (head-tracked, v2)')
    writer = cv2.VideoWriter(f"{out_prefix}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, presentation.size)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {out_prefix}.mp4")

    model = YOLO(MODEL_PATH)
    counter = ZoneCounter()
    trails = defaultdict(lambda: deque(maxlen=TRAIL_LEN))
    recent_events = {}
    events = []
    in_count = out_count = 0

    print("people_counter_v2: tracking + counting by neck point...")
    results = model.track(src, classes=[0], conf=0.1, tracker=str(TRACKER_CONFIG),
                           stream=True, verbose=False, device=device)
    for frame_idx, r in enumerate(results):
        frame, active, fired = render_v2_frame(
            r, frame_idx, zone_a_head, zone_b_head, ZONE_A_FOOT, ZONE_B_FOOT,
            counter, trails, recent_events, HIGHLIGHT_FRAMES, mark_foot=True)
        for tid, event in fired:
            in_count += event == "in"
            out_count += event == "out"
            events.append((round(frame_idx / fps, 2), frame_idx, tid, event))

        writer.write(presentation.render(frame, in_count, out_count, events,
                                         (frame_idx + 1) / fps, active))

    writer.release()

    with open(f"{out_prefix}.csv", "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["time_s", "frame", "track_id", "direction"])
        cw.writerows(events)

    print(f"{src}: IN={in_count} OUT={out_count} ({len(events)} events) "
          f"-> {out_prefix}.mp4 / {out_prefix}.csv")


if __name__ == "__main__":
    main()
