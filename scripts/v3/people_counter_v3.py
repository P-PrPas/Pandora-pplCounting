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

Because the detector's own boxes are already at head level, there's no
foot->head offset to calibrate (that whole problem was specific to v2's
neck-from-pose approach) - ZONE_A/ZONE_B below are just hand-placed at head
height directly, same "hardcoded for this fixed camera" approach as v1's
foot-level zones. Everything else (tracking, debounce, rendering) is
untouched - same ZoneCounter/classify_point/best_device/Presentation as v1.

v2's calibrated head zones turned out to be the wrong seed: they're fit to
neck points, not head-bbox centers, and scattering every head-center this
model produced across a reference frame showed most real crossings landing
in ZONE_B and almost none in ZONE_A. That scatter also exposed a domain-shift
problem: the SCUT-HEAD-trained detector confidently (~0.74 conf, same as
real heads) hallucinates a "head" on a static wall sign/closed-door area
(x<850 in this framing) that never moves - classic fixed-camera background
clutter. Confidence thresholding can't separate it (it's not lower-confidence
than real detections), so HEAD_ROI below hard-excludes that region before
any detection reaches the tracker, and ZONE_A/ZONE_B were redrawn from the
cleaned point cloud instead of copied from v2.

Usage (from repo root):
    python3 scripts/v3/people_counter_v3.py <input_video> <output_prefix>
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
from presentation import Presentation, draw_track, draw_zones  # noqa: E402
from counting import ZoneCounter, best_device, classify_point  # noqa: E402

HERE = Path(__file__).parent
MODEL_PATH = HERE / "head_detector.pt"  # ponytail: SCUT-HEAD nano - swap in medium.pt if accuracy needs it
TRACKER_CONFIG = HERE / "bytetrack_v3.yaml"  # v1's config with match_thresh loosened for small head boxes

# Hardcoded for this fixed camera framing, same "eyeball it" approach as v1's
# foot-level zones - but derived from a scatter of this model's actual head
# centers over the reference clip (see chat write-up), not v2's neck-fit zones.
# x<850 is excluded up front (HEAD_ROI) because that's a static false-positive
# hotspot, not because real heads don't appear there.
HEAD_ROI = (850, 150, 1300, 450)          # (x1, y1, x2, y2) - ignore detections outside this box entirely
ZONE_B = np.array([[850, 150], [1300, 150], [1300, 235], [850, 235]])  # inside / corridor (far side of doorway)
ZONE_A = np.array([[850, 210], [1300, 210], [1300, 400], [850, 400]])  # outside / checkpoint (near side)

TRAIL_LEN = 30
HIGHLIGHT_FRAMES = 20


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
    presentation = Presentation(duration, label_a="OUTSIDE (head)", label_b="INSIDE (head)",
                                 footer="Recorded footage · AI-assisted counting (head-detected, v3)")

    writer = cv2.VideoWriter(f"{out_prefix}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, presentation.size)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {out_prefix}.mp4")
    model = YOLO(MODEL_PATH)
    counter = ZoneCounter()
    trails = defaultdict(lambda: deque(maxlen=TRAIL_LEN))
    recent_events = {}
    events = []
    in_count = out_count = 0

    device = best_device()
    print(f"people_counter_v3: running inference on device={device}")
    # single-class model (just "head") - no classes=[0] filter needed, conf=0.1
    # for the same occlusion mitigation as v1/v2 (let ByteTrack's low-score
    # matching stage do its job instead of filtering boxes out beforehand).
    results = model.track(src, conf=0.1, tracker=str(TRACKER_CONFIG),
                           stream=True, verbose=False, device=device)
    for frame_idx, r in enumerate(results):
        frame = r.orig_img
        occupied = []
        draw_zones(frame, ZONE_A, ZONE_B, occupied, label_a="OUTSIDE (head)", label_b="INSIDE (head)")
        active = 0

        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            rx1, ry1, rx2, ry2 = HEAD_ROI
            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = box
                head = (int((x1 + x2) / 2), int((y1 + y2) / 2))  # head bbox center
                if not (rx1 <= head[0] <= rx2 and ry1 <= head[1] <= ry2):
                    continue  # outside HEAD_ROI - known false-positive area or irrelevant background
                active += 1
                trails[tid].append(head)

                event = counter.update(tid, classify_point(head, ZONE_A, ZONE_B))
                if event:
                    in_count += event == "in"
                    out_count += event == "out"
                    events.append((round(frame_idx / fps, 2), frame_idx, tid, event))
                    recent_events[tid] = (event, frame_idx)

                last = recent_events.get(tid)
                highlighted = bool(last) and frame_idx - last[1] < HIGHLIGHT_FRAMES
                draw_track(frame, box, tid, head, trails[tid], highlighted,
                           last[0] if last else None, occupied)

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
