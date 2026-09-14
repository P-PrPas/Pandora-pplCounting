"""Self-check for v2's foot->head zone calibration (no video/model needed)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "v2"))

import numpy as np
from head_calibration import head_level_zones, neck_point


def demo():
    # neck_point: midpoint of confident shoulders; None if either is low-confidence
    kpts = np.zeros((17, 3))
    kpts[5] = [100, 200, 0.9]  # left shoulder
    kpts[6] = [140, 210, 0.9]  # right shoulder
    assert neck_point(kpts) == (120, 205)
    kpts[6, 2] = 0.1  # right shoulder not confidently visible
    assert neck_point(kpts) is None

    # head_level_zones: uniform per-zone shift by the average observed offset
    zone_a = np.array([[0, 100], [50, 100], [50, 150], [0, 150]])
    zone_b = np.array([[100, 100], [150, 100], [150, 150], [100, 150]])
    samples_a = [(2, -40), (-2, -60), (0, -50)]    # mean (0, -50)
    samples_b = [(10, -30), (10, -30), (10, -30)]  # mean (10, -30)
    head_a, head_b = head_level_zones(zone_a, zone_b, samples_a, samples_b)
    assert np.array_equal(head_a, zone_a + np.array([0, -50]))
    assert np.array_equal(head_b, zone_b + np.array([10, -30]))

    # fallback: a zone with too few samples borrows the combined average of both
    head_a2, _ = head_level_zones(zone_a, zone_b, [], samples_b)
    assert np.array_equal(head_a2, zone_a + np.array([10, -30]))

    # fallback: no samples anywhere -> zero shift (head zone == foot zone), never crashes
    head_a3, head_b3 = head_level_zones(zone_a, zone_b, [], [])
    assert np.array_equal(head_a3, zone_a) and np.array_equal(head_b3, zone_b)

    print("v2 head-calibration self-check passed")


if __name__ == "__main__":
    demo()
