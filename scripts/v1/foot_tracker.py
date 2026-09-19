"""Shared per-frame v1 (foot-tracked) logic: classify by foot point against
the user-drawn zones, feed the counter, draw the track.

Extracted from people_counter.py's main loop so the tracking-algorithm
comparison harness (scripts/tracking_algo/) can drive v1 the same uniform way
it already drives v2/v3 (see head_tracker.py / head_tracker_v3.py) - one
implementation, not one per caller.
"""
from counting import classify_point
from presentation import draw_track, draw_zones


def render_v1_frame(r, frame_idx, zone_a, zone_b, counter, trails, recent_events, highlight_frames,
                     label_a='A / OUTSIDE', label_b='B / INSIDE'):
    """Runs one v1 tracking+counting+draw step against a single ultralytics
    Result `r` (from .track(..., stream=True)). Mutates `trails`/`recent_events`
    in place.

    Returns (frame, active_track_count, fired_events) where fired_events is
    a list of (track_id, "in"|"out") for crossings confirmed this frame.
    """
    frame = r.orig_img
    occupied = []
    draw_zones(frame, zone_a, zone_b, occupied, label_a=label_a, label_b=label_b)
    active = 0
    fired = []

    if r.boxes is not None and r.boxes.id is not None:
        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        active = len(ids)
        for box, tid in zip(boxes, ids):
            x1, y1, x2, y2 = box
            foot = (int((x1 + x2) / 2), int(y2))
            trails[tid].append(foot)

            event = counter.update(tid, classify_point(foot, zone_a, zone_b))
            if event:
                fired.append((int(tid), event))
                recent_events[tid] = (event, frame_idx)

            last = recent_events.get(tid)
            highlighted = bool(last) and frame_idx - last[1] < highlight_frames
            draw_track(frame, box, tid, foot, trails[tid], highlighted,
                       last[0] if last else None, occupied)

    return frame, active, fired
