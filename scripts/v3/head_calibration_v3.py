"""Foot->head zone calibration for the v3 (head-detected) pipeline.

Same physical idea as scripts/v2/head_calibration.py - measure this camera's
real foot->head offset from reference footage and interpolate it per zone
vertex, so users still draw zones at foot level (easy to eyeball) while
counting classifies by head position (robust to foot occlusion). See that
module's docstring for the full reasoning; head_level_zones() is reused
unchanged from there - it's a pure function of (position, offset) samples,
agnostic to how they were collected.

The difference: v2's pose model detects one person and gives both their foot
box and neck keypoint in a single detection, so pairing them is free. v3's
head model only detects heads - there's no matching "foot" in the same
detection. So this runs v1's person detector alongside v3's head detector on
the same frames and pairs each person box with the head detection above it.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent / "v2"))
sys.path.insert(0, str(Path(__file__).parent))
from head_calibration import head_level_zones  # noqa: E402,F401  (re-exported for callers)
from head_tracker_v3 import TOP_FRACTION       # noqa: E402


def collect_head_samples(source, person_model_path, head_model_path, device,
                          stride=1, max_frames=None, on_frame=None, should_stop=None):
    """Returns parallel lists of every matched (foot_position, foot->head offset).

    Matching: greedy one-to-one over all (person, head) pairs where the head
    center falls inside the person box's x-span and in its top TOP_FRACTION
    (head_tracker_v3.head_near_person's containment rule), closest pairs
    assigned first. One-to-one matters here more than it did for a single
    pose detection - two people standing close together at the doorway pinch
    point (the exact crowded case this whole project cares about) can each
    have a candidate head in the other's x-span, and matching each person
    independently let both grab the same nearest head, poisoning the offset
    samples right where the zone boundary sits. A person or head with no
    match this frame contributes no sample - no match means no guess.

    This also naturally excludes the wall-sign false-positive hotspot (no
    person ever stands inside it) as a side effect of the containment rule
    itself - no separate ROI filter needed, same as at runtime.

    `max_frames`/`on_frame`/`should_stop`: same live-calibration hooks as
    collect_pose_samples() in head_calibration.py, kept for signature parity
    in case v3 ever grows a live in-app calibration step like v2's.
    """
    person_model = YOLO(person_model_path)
    head_model = YOLO(head_model_path)
    positions, offsets = [], []

    cap = cv2.VideoCapture(source)
    frame_i = processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_i % stride:
            frame_i += 1
            continue
        frame_i += 1
        if max_frames is not None and processed >= max_frames:
            break
        if should_stop is not None and should_stop():
            break

        pr = person_model.predict(frame, classes=[0], conf=0.1, verbose=False, device=device)[0]
        hr = head_model.predict(frame, conf=0.1, verbose=False, device=device)[0]
        person_boxes = pr.boxes.xyxy.cpu().numpy() if pr.boxes is not None else np.zeros((0, 4))
        head_boxes = hr.boxes.xyxy.cpu().numpy() if hr.boxes is not None else np.zeros((0, 4))
        heads = np.array([[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in head_boxes]) \
            if len(head_boxes) else np.zeros((0, 2))

        # all (person, head) pairs allowed by the containment rule, nearest first
        pairs = []
        for pi, (x1, y1, x2, y2) in enumerate(person_boxes):
            if len(heads) == 0:
                continue
            in_x = (heads[:, 0] >= x1) & (heads[:, 0] <= x2)
            in_y = heads[:, 1] <= y1 + TOP_FRACTION * (y2 - y1)
            top_center = np.array([(x1 + x2) / 2, y1])
            for hi in np.where(in_x & in_y)[0]:
                d2 = np.sum((heads[hi] - top_center) ** 2)
                pairs.append((d2, pi, hi))
        pairs.sort(key=lambda p: p[0])

        used_p, used_h = set(), set()
        for d2, pi, hi in pairs:
            if pi in used_p or hi in used_h:
                continue
            used_p.add(pi)
            used_h.add(hi)
            x1, y1, x2, y2 = person_boxes[pi]
            foot = ((x1 + x2) / 2, y2)
            head = heads[hi]
            positions.append(foot)
            offsets.append((head[0] - foot[0], head[1] - foot[1]))

        processed += 1
        if on_frame:
            on_frame(processed, len(positions))

    cap.release()
    return positions, offsets
