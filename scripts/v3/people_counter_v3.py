"""V3: detect heads directly, instead of inferring a head-level point from a
foot detection (v1) or a pose model's neck point (v2).

Why: v2's zero-shot head-detection idea (ask an open-vocab detector like
YOLO-World for "head") turned out not to work - see the chat write-up. Text-
prompted open-vocab detectors are trained on whole-image/whole-object
captions and just aren't good at localizing a *sub-part* of an object; asking
for "head" or "person head" returns full-body boxes with a confidence <0.3.

Instead this uses `head_detector.pt` - a YOLOv8n checkpoint someone else
already trained on SCUT-HEAD (classroom CCTV, single "head" class) and
published on GitHub (github.com/Abcfsa/YOLOv8_head_detector). No training
budget spent, and unlike the open-vocab attempt it actually gives tight
head-only boxes at 0.4-0.9 confidence on our footage (spike-tested).

Like v2, users still draw zones at foot level (easy to eyeball - asking
someone to draw a polygon around where heads will be is confusing) and a
one-time offline calibration step converts that into the head-level zones
actually used for classification. Unlike v2, the offset being calibrated
here is foot->head (this model's own box center), not foot->neck, and it's
measured by pairing v1's person detector with this head detector on the same
frames (see calibrate_zones_v3.py / head_calibration_v3.py) since a plain
head detector - unlike a pose model - never observes foot and head together
in one detection.

Per-vertex IDW calibration (reused unchanged from v2) is also what keeps
ZONE_A/ZONE_B from overlapping after the shift: a shared foot-level vertex
between the two zones always maps to the same head-level point, so they stay
flush instead of drifting apart.

This model also has a domain-shift problem on this footage: it confidently
(~0.74 conf, same as real heads - confidence thresholding can't separate it)
hallucinates a "head" on a static wall sign/closed-door area that never
moves. Rather than hardcoding a pixel region to exclude (camera-specific,
needs re-deriving per scene), this runs v1's person detector every frame
too and drops any head with no person box under it (head_tracker_v3.py) -
the wall sign never gets a person box, verified empirically, so it's
filtered out by construction. Costs a second model's inference per frame.

Usage (from repo root):
    python3 scripts/v3/calibrate_zones_v3.py <calibration.json> <ref_clip> [more_clips...]
    python3 scripts/v3/people_counter_v3.py <calibration.json> <input_video> <output_prefix>
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
from head_tracker_v3 import render_v3_frame                             # noqa: E402

HERE = Path(__file__).parent
MODEL_PATH = HERE / "head_detector.pt"  # ponytail: SCUT-HEAD nano - swap in medium.pt if accuracy needs it
PERSON_MODEL_PATH = V1_DIR / "yolo11s.pt"  # only for the person-overlap filter, not tracked
TRACKER_CONFIG = HERE / "bytetrack_v3.yaml"  # v1's config with match_thresh loosened for small head boxes

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
    print(f"people_counter_v3: running inference on device={device}")

    zone_a_head, zone_b_head = load_head_zones(calibration_path)
    print(f"people_counter_v3: loaded head-level zones from {calibration_path}")
    print(f"  zone A (outside) vertices shifted by avg {(zone_a_head - ZONE_A_FOOT).mean(axis=0).round(1)} px")
    print(f"  zone B (inside)  vertices shifted by avg {(zone_b_head - ZONE_B_FOOT).mean(axis=0).round(1)} px")

    presentation = Presentation(duration, footer='Recorded footage · AI-assisted counting (head-detected, v3)')
    writer = cv2.VideoWriter(f"{out_prefix}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, presentation.size)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {out_prefix}.mp4")

    model = YOLO(MODEL_PATH)
    person_model = YOLO(PERSON_MODEL_PATH)  # per-frame overlap filter only, not tracked
    counter = ZoneCounter()
    trails = defaultdict(lambda: deque(maxlen=TRAIL_LEN))
    recent_events = {}
    events = []
    in_count = out_count = 0

    print("people_counter_v3: tracking + counting by head point...")
    # single-class model (just "head") - no classes=[0] filter needed, conf=0.1
    # for the same occlusion mitigation as v1/v2 (let ByteTrack's low-score
    # matching stage do its job instead of filtering boxes out beforehand).
    results = model.track(src, conf=0.1, tracker=str(TRACKER_CONFIG),
                           stream=True, verbose=False, device=device)
    for frame_idx, r in enumerate(results):
        pr = person_model.predict(r.orig_img, classes=[0], conf=0.1, verbose=False, device=device)[0]
        person_boxes = pr.boxes.xyxy.cpu().numpy() if pr.boxes is not None else np.zeros((0, 4))
        frame, active, fired = render_v3_frame(
            r, frame_idx, zone_a_head, zone_b_head, ZONE_A_FOOT, ZONE_B_FOOT,
            counter, trails, recent_events, HIGHLIGHT_FRAMES, person_boxes)
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
