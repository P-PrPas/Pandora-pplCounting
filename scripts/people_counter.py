"""People in/out counter for the checkpoint doorway.

Camera is fixed (overhead fisheye above two walk-through metal detectors).
Zone A = outside / checkpoint lane, Zone B = inside / corridor threshold,
using customer-supplied perspective-fitted quads sitting flush against each
other at the chokepoint. A->B crossing = "in", B->A = "out".

Usage (from repo root):
    python3 scripts/people_counter.py <input_video> <output_prefix>

Produces <output_prefix>.mp4 (annotated) and <output_prefix>.csv (event log).
"""
import sys
import csv
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from presentation import Presentation, draw_track, draw_zones
from counting import ZoneCounter, best_device, classify_point  # noqa: F401 (ZoneCounter re-exported for tests)

HERE = Path(__file__).parent
MODEL_PATH = HERE / "yolo11s.pt"
TRACKER_CONFIG = HERE / "bytetrack_custom.yaml"

# Hardcoded for this fixed camera framing (see data/ clips) — re-eyeball if the camera moves.
# Client-supplied, perspective-fitted quads (replaces the earlier rough rectangles).
ZONE_A = np.array([[780, 550], [685, 697], [1133, 771], [1164, 605]])  # outside / checkpoint
ZONE_B = np.array([[797, 535], [974, 313], [1185, 328], [1163, 590]])  # inside / corridor

TRAIL_LEN = 30         # frames of trail drawn per track
HIGHLIGHT_FRAMES = 20  # how long a track highlights its crossing direction


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
    presentation = Presentation(duration)

    writer = cv2.VideoWriter(f"{out_prefix}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, presentation.size)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {out_prefix}.mp4")
    model = YOLO(MODEL_PATH)
    counter = ZoneCounter()
    trails = defaultdict(lambda: deque(maxlen=TRAIL_LEN))
    recent_events = {}  # track_id -> (event_label, frame_idx) for the direction highlight
    events = []
    in_count = out_count = 0

    # ponytail: device auto-picked (cuda > mps > cpu) — same weights/precision on any
    # backend, so this is a free speedup, not an accuracy tradeoff. Override with the
    # DEVICE env var if a given machine's backend hits an unimplemented op.
    # conf=0.1 (not 0.25) so partially-blocked people still get a low-score detection
    # that ByteTrack's second-stage matching can use — the whole point of its two-stage
    # design, wasted if we filter those boxes out before they reach it. Paired with
    # bytetrack_custom.yaml's longer track_buffer, this is the fix for boxes getting
    # lost when the doorway is crowded and people fully block each other.
    device = best_device()
    print(f"people_counter: running inference on device={device}")
    results = model.track(src, classes=[0], conf=0.1, tracker=str(TRACKER_CONFIG),
                           stream=True, verbose=False, device=device)
    for frame_idx, r in enumerate(results):
        frame = r.orig_img
        occupied = []
        draw_zones(frame, ZONE_A, ZONE_B, occupied)
        active = 0

        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            active = len(ids)
            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = box
                foot = (int((x1 + x2) / 2), int(y2))
                trails[tid].append(foot)

                event = counter.update(tid, classify_point(foot, ZONE_A, ZONE_B))
                if event:
                    in_count += event == "in"
                    out_count += event == "out"
                    events.append((round(frame_idx / fps, 2), frame_idx, tid, event))
                    recent_events[tid] = (event, frame_idx)

                last = recent_events.get(tid)
                highlighted = bool(last) and frame_idx - last[1] < HIGHLIGHT_FRAMES
                draw_track(frame, box, tid, foot, trails[tid], highlighted,
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
