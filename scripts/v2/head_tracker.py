"""Shared per-frame v2 (head-tracked) logic: classify by neck point against
calibrated head-level zones, feed the counter, draw both zone layers + track.

Used by both the batch pipeline (people_counter_v2.py) and the live app
(app/video_worker.py) so the two can't drift apart - this is the exact logic
that had the zone-overlap bug fixed in it once already; one implementation,
not two.
"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "v1"))
from counting import classify_point                                          # noqa: E402
from presentation import WHITE, draw_track, draw_user_zone_outline, draw_zones  # noqa: E402

from head_calibration import neck_point


def render_v2_frame(r, frame_idx, zone_a_head, zone_b_head, zone_a_foot, zone_b_foot,
                     counter, trails, recent_events, highlight_frames,
                     label_a='A / OUTSIDE (head, active)', label_b='B / INSIDE (head, active)',
                     label_a_foot='A · foot (user-drawn)', label_b_foot='B · foot (user-drawn)',
                     mark_foot=False):
    """Runs one v2 tracking+counting+draw step against a single ultralytics
    Result `r` (from a pose model's .track(..., stream=True), so it has both
    r.boxes and r.keypoints). Mutates `trails`/`recent_events` in place.

    `mark_foot`: also plot v1's raw foot point as a small white dot, for
    visual side-by-side comparison against the neck point actually used
    (batch pipeline uses this; the live app doesn't need it).

    Returns (frame, active_track_count, fired_events) where fired_events is
    a list of (track_id, "in"|"out") for crossings confirmed this frame.
    """
    frame = r.orig_img
    occupied = []
    draw_zones(frame, zone_a_head, zone_b_head, occupied, label_a=label_a, label_b=label_b)
    draw_user_zone_outline(frame, zone_a_foot, label_a_foot, occupied)
    draw_user_zone_outline(frame, zone_b_foot, label_b_foot, occupied)
    active = 0
    fired = []

    if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        kpts = r.keypoints.data.cpu().numpy()
        active = len(ids)
        for box, tid, kp in zip(boxes, ids, kpts):
            neck = neck_point(kp)
            # ponytail: shoulders not confidently visible this frame (occlusion,
            # side-on pose) -> classify as no zone, same as v1's off-polygon case.
            zone = classify_point(neck, zone_a_head, zone_b_head) if neck else None
            event = counter.update(tid, zone)
            if event:
                fired.append((int(tid), event))
                recent_events[tid] = (event, frame_idx)

            if neck is not None:
                point = (int(neck[0]), int(neck[1]))
                trails[tid].append(point)
                last = recent_events.get(tid)
                highlighted = bool(last) and frame_idx - last[1] < highlight_frames
                draw_track(frame, box, tid, point, trails[tid], highlighted,
                           last[0] if last else None, occupied)
                if mark_foot:
                    x1, y1, x2, y2 = box
                    cv2.circle(frame, (int((x1 + x2) / 2), int(y2)), 3, WHITE, -1, cv2.LINE_AA)

    return frame, active, fired

