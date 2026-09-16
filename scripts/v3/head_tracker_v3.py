"""Shared per-frame v3 (head-detected) logic: classify by head-box center
against calibrated head-level zones, feed the counter, draw both zone layers
+ track.

Same role as scripts/v2/head_tracker.py, adapted for a dedicated head
detector's plain boxes instead of a pose model's keypoints - no neck-point
extraction needed, but adds a person-overlap filter: this model confidently
(~0.74 conf, indistinguishable from real heads) hallucinates a "head" on a
static wall sign in this camera's frame (see people_counter_v3.py's
docstring). A hardcoded pixel ROI would have "fixed" that for this one
camera framing only; instead a head only reaches the tracker if a
same-frame person-detector box sits under it - the wall sign never gets a
person box (verified empirically), so it's filtered out by construction,
on any camera, with no per-scene constant to re-derive.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "v1"))
from counting import classify_point                                     # noqa: E402
from presentation import draw_track, draw_user_zone_outline, draw_zones  # noqa: E402

# A real head sits at the very top of a full-body detector box, not just
# "somewhere in the upper half" - shared with head_calibration_v3.py's
# person<->head pairing so both use the same containment rule.
TOP_FRACTION = 0.25


def head_near_person(head_xy, person_boxes):
    """True if `head_xy` falls in the top TOP_FRACTION of any given person
    box's height and within its x-span. `person_boxes` is this frame's
    person-detector boxes (xyxy) - typically a handful, so a plain loop
    beats setting up numpy for it.
    """
    hx, hy = head_xy
    for x1, y1, x2, y2 in person_boxes:
        if x1 <= hx <= x2 and hy <= y1 + TOP_FRACTION * (y2 - y1):
            return True
    return False


def render_v3_frame(r, frame_idx, zone_a_head, zone_b_head, zone_a_foot, zone_b_foot,
                     counter, trails, recent_events, highlight_frames, person_boxes,
                     label_a='A / OUTSIDE (head, active)', label_b='B / INSIDE (head, active)',
                     label_a_foot='A · foot (user-drawn)', label_b_foot='B · foot (user-drawn)'):
    """Runs one v3 tracking+counting+draw step against a single ultralytics
    Result `r` (from the head detector's .track(..., stream=True)). Mutates
    `trails`/`recent_events` in place. `person_boxes` is this same frame's
    person-detector boxes (xyxy), used to drop heads with no person under
    them (see head_near_person).

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
        for box, tid in zip(boxes, ids):
            x1, y1, x2, y2 = box
            head = (int((x1 + x2) / 2), int((y1 + y2) / 2))  # head bbox center
            if not head_near_person(head, person_boxes):
                continue  # no person under this "head" - background/false-positive
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
