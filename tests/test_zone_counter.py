"""Self-check for the ZoneCounter debounce/state-machine logic (no video/model needed)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from people_counter import ZoneCounter


def demo():
    c = ZoneCounter(debounce_frames=3)

    # first sighting (in A) just establishes a baseline — must not count
    for _ in range(3):
        assert c.update(1, "A") is None

    # clean A -> B walk-through: only the 3rd confirmed frame in B fires "in"
    assert [c.update(1, "B") for _ in range(3)] == [None, None, "in"]

    # lingering in B afterwards must not re-fire
    assert c.update(1, "B") is None

    # walking back out: B -> A fires exactly one "out"
    assert [c.update(1, "A") for _ in range(3)] == [None, None, "out"]

    # anti-jitter: bouncing back before reaching the debounce threshold must not count
    c2 = ZoneCounter(debounce_frames=4)
    assert c2.update(2, "A") is None
    assert c2.update(2, "B") is None  # 1 frame in B
    assert c2.update(2, "B") is None  # 2 frames in B
    assert c2.update(2, "A") is None  # jitters back to A before reaching threshold -> resets
    assert c2.update(2, "A") is None

    # a track first seen outside both zones, then entering B, must not falsely count as "in"
    c3 = ZoneCounter(debounce_frames=2)
    assert c3.update(3, None) is None
    assert c3.update(3, "B") is None
    assert c3.update(3, "B") is None  # baseline set to B, no prior zone -> no event

    print("all zone-counter self-checks passed")


if __name__ == "__main__":
    demo()
