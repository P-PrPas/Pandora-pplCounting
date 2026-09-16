"""Shared per-frame v3 (head-detected) logic: classify by head-box center
against calibrated head-level zones, feed the counter, draw both zone layers
+ track.

Same role as scripts/v2/head_tracker.py, adapted for a dedicated head
detector's plain boxes instead of a pose model's keypoints - no neck-point
extraction needed, but adds HEAD_ROI: this model confidently (~0.74 conf,
indistinguishable from real heads) hallucinates a "head" on a static wall
sign in this camera's frame (see people_counter_v3.py's docstring), so
anything outside HEAD_ROI is dropped before it ever reaches the tracker.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "v1"))
from counting import classify_point                                     # noqa: E402
from presentation import draw_track, draw_user_zone_outline, draw_zones  # noqa: E402

# Hardcoded for this fixed camera framing - a static wall sign at x<850 that
# this model confidently misreads as a head. Re-derive per camera if this
# model is ever pointed at a different scene (see chat write-up for how).
HEAD_ROI = (850, 150, 1300, 450)  # (x1, y1, x2, y2)


def render_v3_frame(r, frame_idx, zone_a_head, zone_b_head, zone_a_foot, zone_b_foot,
                     counter, trails, recent_events, highlight_frames,
                     label_a='A / OUTSIDE (head, active)', label_b='B / INSIDE (head, active)',
                     label_a_foot='A · foot (user-drawn)', label_b_foot='B · foot (user-drawn)'):
    """Runs one v3 tracking+counting+draw step against a single ultralytics
    Result `r` (from the head detector's .track(..., stream=True)). Mutates
    `trails`/`recent_events` in place.

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

            zone = classify_point(head, zone_a_head, zone_b_head)
            event = counter.update(tid, zone)
            if event:
                fired.append((int(tid), event))
                recent_events[tid] = (event, frame_idx)

            last = recent_events.get(tid)
            highlighted = bool(last) and frame_idx - last[1] < highlight_frames
            draw_track(frame, box, tid, head, trails[tid], highlighted,
                       last[0] if last else None, occupied)

    return frame, active, fired
