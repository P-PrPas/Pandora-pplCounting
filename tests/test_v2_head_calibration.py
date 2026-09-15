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

    # head_level_zones: per-vertex IDW offset, not one shift for the whole zone -
    # a vertex near a cluster of samples should land close to that cluster's offset.
    positions = [(0, 500), (10, 500), (0, 510), (500, 500), (510, 500), (500, 510)]
    offsets = [(0, -30), (0, -30), (0, -30), (10, -70), (10, -70), (10, -70)]
    zone_a = np.array([[0, 500], [50, 500], [50, 550], [0, 550]])       # near the first cluster
    zone_b = np.array([[500, 500], [550, 500], [550, 550], [500, 550]])  # near the second cluster
    head_a, head_b = head_level_zones(zone_a, zone_b, positions, offsets, k=3)
    assert np.allclose(head_a - zone_a, [0, -30], atol=1)
    assert np.allclose(head_b - zone_b, [10, -70], atol=1)

    # The bug this replaced: averaging each zone into one shift pulled two
    # zones that share a boundary apart (different zones, different shifts).
    # Per-vertex IDW is a pure function of position, so a vertex shared by
    # two adjacent ("flush") zones must map to the exact same point either way.
    shared_vertex = (50, 500)
    zone_a2 = np.array([[0, 500], shared_vertex, [50, 550], [0, 550]])
    zone_b2 = np.array([shared_vertex, [500, 500], [500, 550], [50, 550]])
    head_a2, head_b2 = head_level_zones(zone_a2, zone_b2, positions, offsets, k=3)
    assert tuple(head_a2[1]) == tuple(head_b2[0]), "shared vertex must map identically for both zones"

    # fallback: no samples anywhere -> zero shift (head zone == foot zone), never crashes
    head_a3, head_b3 = head_level_zones(zone_a, zone_b, [], [])
    assert np.array_equal(head_a3, zone_a) and np.array_equal(head_b3, zone_b)

    print("v2 head-calibration self-check passed")


if __name__ == "__main__":
    demo()
