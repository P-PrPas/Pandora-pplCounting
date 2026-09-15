"""V2: track by neck point (shoulder-keypoint midpoint) instead of foot point.

Same detection+tracking+debounce pipeline as v1, but classification uses a
YOLO-pose model's neck point rather than the bbox-bottom foot point — the
idea being it's less sensitive to stride/pose than the foot.

The zone polygons are still drawn by the user at foot level (same numbers as
v1's ZONE_A/ZONE_B — same physical camera, same doorway). Pass 1 measures
this clip's own real foot->neck offset per zone and shifts the polygons up
to head level for pass 2's actual classification. The output video draws
both: the user's foot-level polygons (thin white outline) and the calibrated
head-level polygons actually used for counting (filled, as in v1).

Usage (from repo root):
    python3 scripts/v2/people_counter_v2.py <input_video> <output_prefix>
"""
import sys
import csv
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

V1_DIR = Path(__file__).parent.parent / "v1"
sys.path.insert(0, str(V1_DIR))
sys.path.insert(0, str(Path(__file__).parent))
from presentation import Presentation, WHITE, chip, draw_track, draw_zones  # noqa: E402
from counting import ZoneCounter, best_device, classify_point              # noqa: E402
from people_counter import ZONE_A as ZONE_A_FOOT, ZONE_B as ZONE_B_FOOT    # noqa: E402
from head_calibration import head_level_zones, neck_point                  # noqa: E402

HERE = Path(__file__).parent
MODEL_PATH = HERE / "yolo11s-pose.pt"
TRACKER_CONFIG = V1_DIR / "bytetrack_custom.yaml"

TRAIL_LEN = 30
HIGHLIGHT_FRAMES = 20
# ponytail: calibration is a one-off statistical average, not part of the
# actual counting inference — skipping frames here doesn't touch counting
# accuracy (pass 2 below still runs every frame at full res).
CALIBRATION_STRIDE = 3


def draw_user_zone_outline(frame, polygon, label, occupied=None):
    """Thin, unfilled outline for the user-drawn (foot-level) reference zone,
    kept visually distinct from the filled head-level zone actually used.
    """
    cv2.polylines(frame, [polygon], True, WHITE, 1, cv2.LINE_AA)
    x, y = polygon[np.argmin(polygon[:, 1])]
    chip(frame, label, int(x), int(y) - 40, WHITE, occupied)


def calibrate(src, device):
    """Pass 1: collect (foot_position, foot->neck offset) pairs from every
    confident pose detection in the clip - used to interpolate a
    per-vertex offset for each zone (see head_level_zones).
    """
    model = YOLO(MODEL_PATH)
    positions, offsets = [], []
    results = model(src, classes=[0], conf=0.1, stream=True, verbose=False,
                     device=device, vid_stride=CALIBRATION_STRIDE)
    for r in results:
        if r.boxes is None or r.keypoints is None:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        kpts = r.keypoints.data.cpu().numpy()
        for box, kp in zip(boxes, kpts):
            x1, y1, x2, y2 = box
            foot = ((x1 + x2) / 2, y2)
            neck = neck_point(kp)
            if neck is None:
                continue
            positions.append(foot)
            offsets.append((neck[0] - foot[0], neck[1] - foot[1]))
    print(f"  {len(positions)} calibration samples collected")
    return head_level_zones(ZONE_A_FOOT, ZONE_B_FOOT, positions, offsets)


def main():
    if len(sys.argv) != 3:
        print(f"usage: python3 {sys.argv[0]} <input_video> <output_prefix>")
        sys.exit(1)
    src, out_prefix = sys.argv[1], sys.argv[2]

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

    print("people_counter_v2: pass 1/2 - calibrating head-level zones from this clip's own poses...")
    zone_a_head, zone_b_head = calibrate(src, device)
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

    print("people_counter_v2: pass 2/2 - tracking + counting by neck point...")
    results = model.track(src, classes=[0], conf=0.1, tracker=str(TRACKER_CONFIG),
                           stream=True, verbose=False, device=device)
    for frame_idx, r in enumerate(results):
        frame = r.orig_img
        occupied = []
        draw_zones(frame, zone_a_head, zone_b_head, occupied,
                   label_a='A / OUTSIDE (head, active)', label_b='B / INSIDE (head, active)')
        draw_user_zone_outline(frame, ZONE_A_FOOT, 'A · foot (user-drawn)', occupied)
        draw_user_zone_outline(frame, ZONE_B_FOOT, 'B · foot (user-drawn)', occupied)
        active = 0

        if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            kpts = r.keypoints.data.cpu().numpy()
            active = len(ids)
            for box, tid, kp in zip(boxes, ids, kpts):
                x1, y1, x2, y2 = box
                foot_raw = (int((x1 + x2) / 2), int(y2))
                neck = neck_point(kp)
                # ponytail: shoulders not confidently visible this frame (occlusion,
                # side-on pose) -> classify as no zone, same as v1's off-polygon case.
                # Simpler than falling back to the foot point, and never miscounts.
                zone = classify_point(neck, zone_a_head, zone_b_head) if neck else None
                event = counter.update(tid, zone)
                if event:
                    in_count += event == "in"
                    out_count += event == "out"
                    events.append((round(frame_idx / fps, 2), frame_idx, tid, event))
                    recent_events[tid] = (event, frame_idx)

                if neck is not None:
                    point = (int(neck[0]), int(neck[1]))
                    trails[tid].append(point)
                    last = recent_events.get(tid)
                    highlighted = bool(last) and frame_idx - last[1] < HIGHLIGHT_FRAMES
                    draw_track(frame, box, tid, point, trails[tid], highlighted,
                               last[0] if last else None, occupied)
                    cv2.circle(frame, foot_raw, 3, WHITE, -1, cv2.LINE_AA)  # v1's tracked point, for comparison

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
