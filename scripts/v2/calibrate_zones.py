"""One-time offline calibration: derive this camera's head-level zone polygons
from reference footage, and cache them to JSON.

Why this is a separate, offline step: the foot->neck offset is a property of
the fixed camera geometry (position/angle), not of whatever happens to be
live right now. people_counter_v2.py used to re-derive it by scanning the
*entire* clip before counting a single frame - fine for a batch clip, but
impossible against an open-ended RTSP stream. So calibration now runs once
(or whenever the camera moves) against reference footage with people walking
through the zones, and the result is just loaded at runtime (see
people_counter_v2.py) - one pass, real-time-compatible.

Usage (from repo root):
    python3 scripts/v2/calibrate_zones.py <output.json> <clip1> [clip2 ...]

Feed it every reference clip you have for this camera - more people/paths
covered means a better-sampled interpolation (see head_calibration.py).
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "v1"))
sys.path.insert(0, str(Path(__file__).parent))
from counting import best_device                                        # noqa: E402
from people_counter import ZONE_A as ZONE_A_FOOT, ZONE_B as ZONE_B_FOOT  # noqa: E402
from head_calibration import collect_pose_samples, head_level_zones      # noqa: E402

HERE = Path(__file__).parent
MODEL_PATH = HERE / "yolo11s-pose.pt"
CALIBRATION_STRIDE = 3  # ponytail: one-off statistical pass, subsampling doesn't hurt it


def main():
    if len(sys.argv) < 3:
        print(f"usage: python3 {sys.argv[0]} <output.json> <clip1> [clip2 ...]")
        sys.exit(1)
    out_path, clips = sys.argv[1], sys.argv[2:]

    device = best_device()
    positions, offsets = [], []
    for clip in clips:
        print(f"calibrate_zones: sampling {clip} (device={device})...")
        p, o = collect_pose_samples(clip, MODEL_PATH, device, stride=CALIBRATION_STRIDE)
        print(f"  {len(p)} samples")
        positions += p
        offsets += o

    zone_a_head, zone_b_head = head_level_zones(ZONE_A_FOOT, ZONE_B_FOOT, positions, offsets)
    Path(out_path).write_text(json.dumps({
        "source_clips": clips,
        "sample_count": len(positions),
        "zone_a_head": zone_a_head.tolist(),
        "zone_b_head": zone_b_head.tolist(),
    }, indent=2))
    print(f"calibrate_zones: {len(positions)} total samples across {len(clips)} clip(s) -> {out_path}")


if __name__ == "__main__":
    main()
