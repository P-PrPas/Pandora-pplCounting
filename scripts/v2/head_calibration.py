"""Foot->head zone calibration + neck-point extraction for the v2 pipeline.

Physical idea: with an overhead/fisheye camera, a person's head and foot land
at different pixel positions (parallax), and the size of that offset depends
on where in the frame the person is. Rather than hand-calibrating camera
height/focal length, we measure the real foot->neck offset from this video's
own pose detections and use each zone's average as a uniform shift — no
camera model needed, and it self-calibrates to whatever clip is given.
"""
import numpy as np

# Standard COCO-pose keypoint order (all YOLO*-pose weights use this).
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
KPT_CONF_MIN = 0.5  # ponytail: fixed threshold; expose as a param if a camera needs tuning


def neck_point(keypoints):
    """keypoints: (17, 3) array of (x, y, conf). "Neck" = shoulder midpoint,
    used as the head-level stand-in (pose models don't have a head keypoint).
    Returns (x, y), or None if either shoulder isn't confidently visible.
    """
    left, right = keypoints[LEFT_SHOULDER], keypoints[RIGHT_SHOULDER]
    if left[2] < KPT_CONF_MIN or right[2] < KPT_CONF_MIN:
        return None
    return ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)


def head_level_zones(zone_a_foot, zone_b_foot, samples_a, samples_b):
    """samples_* are lists of (dx, dy) foot->neck offsets observed with the
    foot inside that zone (see calibrate() in people_counter_v2.py). Returns
    (zone_a_head, zone_b_head): the same polygons, each uniformly translated
    by that zone's average observed offset.

    ponytail: a single per-zone average, not per-vertex interpolation — these
    doorway zones are small/local, so parallax is roughly uniform across each
    one. Upgrade to IDW-per-vertex if a future zone spans a much wider area.
    """
    all_samples = samples_a + samples_b
    fallback = np.mean(all_samples, axis=0) if all_samples else np.zeros(2)
    offset_a = np.mean(samples_a, axis=0) if len(samples_a) >= 3 else fallback
    offset_b = np.mean(samples_b, axis=0) if len(samples_b) >= 3 else fallback
    return (zone_a_foot + offset_a).astype(np.int32), (zone_b_foot + offset_b).astype(np.int32)
