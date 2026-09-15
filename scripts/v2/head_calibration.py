"""Foot->head zone calibration + neck-point extraction for the v2 pipeline.

Physical idea: with an overhead/fisheye camera, a person's head and foot land
at different pixel positions (parallax), and the size/direction of that
offset varies smoothly across the frame (closer to the camera's nadir ->
smaller offset). Rather than hand-calibrating camera height/focal length, we
measure the real foot->neck offset from this video's own pose detections and
interpolate it at each zone vertex from the nearest real observations.
"""
import numpy as np
from ultralytics import YOLO

# Standard COCO-pose keypoint order (all YOLO*-pose weights use this).
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
KPT_CONF_MIN = 0.5  # ponytail: fixed threshold; expose as a param if a camera needs tuning


def collect_pose_samples(source, model_path, device, stride=1, max_frames=None,
                          on_frame=None, should_stop=None):
    """Run a YOLO-pose model over `source`, returning parallel lists of every
    confident detection's (foot_position, foot->neck offset).

    `max_frames` stops early after that many *processed* frames (i.e. after
    `stride`-many source frames each) - a finished clip ends on its own, but a
    live/RTSP source doesn't, so callers sampling a live stream for
    calibration need a way to say "that's enough".

    `on_frame(frame_index, sample_count)`, if given, is called after every
    processed frame - lets a caller show live progress. `should_stop()`, if
    given, is checked every frame and breaks the loop when truthy - lets a
    caller cancel a live calibration in progress.
    """
    model = YOLO(model_path)
    positions, offsets = [], []
    results = model(source, classes=[0], conf=0.1, stream=True, verbose=False,
                     device=device, vid_stride=stride)
    for i, r in enumerate(results):
        if max_frames is not None and i >= max_frames:
            break
        if should_stop is not None and should_stop():
            break
        if r.boxes is None or r.keypoints is None:
            if on_frame:
                on_frame(i, len(positions))
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        kpts = r.keypoints.data.cpu().numpy()
        for box, kp in zip(boxes, kpts):
            neck = neck_point(kp)
            if neck is None:
                continue
            x1, y1, x2, y2 = box
            foot = ((x1 + x2) / 2, y2)
            positions.append(foot)
            offsets.append((neck[0] - foot[0], neck[1] - foot[1]))
        if on_frame:
            on_frame(i, len(positions))
    return positions, offsets


def neck_point(keypoints):
    """keypoints: (17, 3) array of (x, y, conf). "Neck" = shoulder midpoint,
    used as the head-level stand-in (pose models don't have a head keypoint).
    Returns (x, y), or None if either shoulder isn't confidently visible.
    """
    left, right = keypoints[LEFT_SHOULDER], keypoints[RIGHT_SHOULDER]
    if left[2] < KPT_CONF_MIN or right[2] < KPT_CONF_MIN:
        return None
    return ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)


def _idw_offset(query, positions, offsets, k=12, power=2):
    """Inverse-distance-weighted offset at `query`, from the k nearest
    (position, offset) samples (Shepard's method). A pure function of image
    position - NOT of which zone `query` happens to belong to - so two
    zones that share a vertex/edge (flush against each other, as these are)
    still land on the same point after the shift instead of drifting apart.
    """
    if len(positions) == 0:
        return np.zeros(2)
    d2 = np.sum((positions - query) ** 2, axis=1)
    nearest = np.argsort(d2)[:min(k, len(d2))]
    d2 = d2[nearest]
    if d2[0] < 1.0:  # query ~coincides with a sample - use it directly, skip the divide
        return offsets[nearest[0]]
    weights = 1.0 / d2 ** (power / 2)
    return (weights[:, None] * offsets[nearest]).sum(axis=0) / weights.sum()


def head_level_zones(zone_a_foot, zone_b_foot, positions, offsets, k=12):
    """positions/offsets: parallel lists of observed (foot_x, foot_y) and its
    foot->neck offset, pooled from every confident pose detection in the
    clip (not split by zone - see calibrate() in people_counter_v2.py).
    Each vertex of each zone gets its own IDW-interpolated offset from the
    nearest real observations, so the correction varies smoothly across the
    frame instead of being one rigid per-zone shift.

    An earlier version averaged each zone's samples into a single shift and
    translated the whole polygon by it - two zones that touch at a shared
    edge got shifted by different amounts and pulled apart into an overlap.
    Per-vertex interpolation fixes that: a shared vertex is the same input
    point for both zones, so it always maps to the same output point.

    ponytail: plain Shepard's-method IDW over k nearest neighbors, no new
    dependency. Upgrade to a fitted parametric camera model (height/focal
    length) if a future zone needs to extrapolate far past where people
    were actually observed walking.
    """
    positions = np.asarray(positions, dtype=float) if len(positions) else np.zeros((0, 2))
    offsets = np.asarray(offsets, dtype=float) if len(offsets) else np.zeros((0, 2))

    def shift(zone):
        deltas = np.array([_idw_offset(v, positions, offsets, k) for v in zone])
        return (zone + deltas).astype(np.int32)

    return shift(zone_a_foot), shift(zone_b_foot)
