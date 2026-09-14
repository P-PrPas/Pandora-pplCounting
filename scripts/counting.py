"""Source-agnostic counting core: shared by the batch CLI (scripts/people_counter.py)
and the live desktop app (app/), so both run the exact same algorithm.
"""
import os

import cv2
import torch

DEBOUNCE_FRAMES = 4  # consecutive frames required in a zone before a state change is confirmed


def best_device():
    """Fastest available inference device, same weights/precision either way — no
    accuracy tradeoff, just hardware acceleration. Honors a DEVICE env override
    (e.g. force "cpu" if a backend hits an unimplemented op on a given machine).

    torch.cuda.is_available() can lie: it returns True even when the installed
    torch build has no compiled kernels for the GPU's compute capability (seen on
    this dev box's V100 + a CUDA 13 wheel) — actually running an op is the only
    reliable check, so probe with a throwaway op before trusting either backend.
    """
    if os.environ.get("DEVICE"):
        return os.environ["DEVICE"]
    checks = [("cuda", torch.cuda.is_available), ("mps", torch.backends.mps.is_available)]
    for device, is_available in checks:
        if not is_available():
            continue
        try:
            (torch.zeros(1, device=device) + 1).item()
            return device
        except RuntimeError:
            pass  # e.g. no kernel image for this GPU's compute capability — fall through
    return "cpu"


def classify_point(pt, zone_a, zone_b):
    """Classify a point as zone 'A', 'B', or None (outside both)."""
    if cv2.pointPolygonTest(zone_b, pt, False) >= 0:
        return "B"
    if cv2.pointPolygonTest(zone_a, pt, False) >= 0:
        return "A"
    return None


class ZoneCounter:
    """Per-track A/B state machine with anti-jitter debounce and duplicate-count prevention.

    A->B = "in", B->A = "out". A transition only fires once the track has spent
    `debounce_frames` consecutive frames in the new zone, and only advances the
    confirmed state (so lingering in a zone never re-fires, and jitter that never
    reaches the threshold never counts).
    """

    def __init__(self, debounce_frames=DEBOUNCE_FRAMES):
        self.debounce_frames = debounce_frames
        self.confirmed = {}  # track_id -> "A" | "B"
        self.pending = {}    # track_id -> [zone, consecutive_count]

    def update(self, track_id, zone):
        """Feed the current raw zone for a track this frame. Returns "in", "out", or None."""
        if zone is None:
            self.pending.pop(track_id, None)
            return None

        if self.confirmed.get(track_id) == zone:
            self.pending.pop(track_id, None)
            return None

        pend = self.pending.get(track_id)
        pend = [zone, pend[1] + 1] if pend and pend[0] == zone else [zone, 1]
        self.pending[track_id] = pend

        if pend[1] < self.debounce_frames:
            return None

        prev = self.confirmed.get(track_id)
        self.confirmed[track_id] = zone
        self.pending.pop(track_id, None)
        if prev == "A" and zone == "B":
            return "in"
        if prev == "B" and zone == "A":
            return "out"
        return None  # first-ever sighting just establishes a baseline zone, doesn't count
