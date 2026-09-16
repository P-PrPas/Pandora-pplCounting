"""One-time offline calibration: derive this camera's head-level zone
polygons for the v3 (head-detected) pipeline from reference footage, and
cache them to JSON. Same role/usage pattern as scripts/v2/calibrate_zones.py
- see that module's docstring for why this is a separate offline step (the
short version: the offset is a property of the fixed camera, not of
whatever clip/stream happens to be running, and a live RTSP feed has no
"whole clip" to pre-scan the way this batch pass does).

Usage (from repo root):
    python3 scripts/v3/calibrate_zones_v3.py <output.json> <clip1> [clip2 ...]

Feed it every reference clip you have for this camera - more people/paths
covered means a better-sampled interpolation (see head_calibration_v3.py).
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "v1"))
sys.path.insert(0, str(Path(__file__).parent))
from counting import best_device                                          # noqa: E402
from people_counter import ZONE_A as ZONE_A_FOOT, ZONE_B as ZONE_B_FOOT    # noqa: E402
from head_calibration_v3 import collect_head_samples, head_level_zones    # noqa: E402

HERE = Path(__file__).parent
PERSON_MODEL_PATH = HERE.parent / "v1" / "yolo11s.pt"
HEAD_MODEL_PATH = HERE / "head_detector.pt"
CALIBRATION_STRIDE = 3  # ponytail: one-off statistical pass, subsampling doesn't hurt it


def main():
    if len(sys.argv) < 3:
        print(f"usage: python3 {sys.argv[0]} <output.json> <clip1> [clip2 ...]")
        sys.exit(1)
    out_path, clips = sys.argv[1], sys.argv[2:]

    device = best_device()
    positions, offsets = [], []
    for clip in clips:
        print(f"calibrate_zones_v3: sampling {clip} (device={device})...")
        p, o = collect_head_samples(clip, PERSON_MODEL_PATH, HEAD_MODEL_PATH, device,
                                     stride=CALIBRATION_STRIDE)
        print(f"  {len(p)} samples")
        positions += p
        offsets += o

    # ponytail: v2's default k=12 left visible zone overlap here (~27% of the
    # smaller zone) - v3's samples are noisier than v2's (cross-model foot/head
    # pairing vs. one pose model giving both per detection), so two nearby
    # query vertices can draw from disjoint, inconsistently-offset neighbor
    # sets. A bigger k averages that out; swept 12->150 against actual overlap
    # and k=80 already matches v2's near-zero overlap without over-smoothing.
    zone_a_head, zone_b_head = head_level_zones(ZONE_A_FOOT, ZONE_B_FOOT, positions, offsets, k=80)
    Path(out_path).write_text(json.dumps({
        "source_clips": clips,
        "sample_count": len(positions),
        "zone_a_head": zone_a_head.tolist(),
        "zone_b_head": zone_b_head.tolist(),
    }, indent=2))
    print(f"calibrate_zones_v3: {len(positions)} total samples across {len(clips)} clip(s) -> {out_path}")


if __name__ == "__main__":
    main()
