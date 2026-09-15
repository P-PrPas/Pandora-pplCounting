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
from head_calibration import head_level_zones, neck_point                # noqa: E402
from ultralytics import YOLO

HERE = Path(__file__).parent
MODEL_PATH = HERE / "yolo11s-pose.pt"
CALIBRATION_STRIDE = 3  # ponytail: one-off statistical pass, subsampling doesn't hurt it


def collect_samples(src, device):
    model = YOLO(MODEL_PATH)
    positions, offsets = [], []
    results = model(src, classes=[0], conf=0.1, stream=True, verbose=False,
                     device=device, vid_stride=CALIBRATION_STRIDE)
    for r in results:
        if r.boxes is None or r.keypoints is None:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        kpts = r.keypoints.data.cpu().numpy()
        for box, kp in zip(boxes, kpts):
            neck = neck_point(kp)
            if neck is None:
                continue
            x1, y1, x2, y2 = box
            positions.append(((x1 + x2) / 2, y2))
            offsets.append((neck[0] - positions[-1][0], neck[1] - positions[-1][1]))
    return positions, offsets


def main():
    if len(sys.argv) < 3:
        print(f"usage: python3 {sys.argv[0]} <output.json> <clip1> [clip2 ...]")
        sys.exit(1)
    out_path, clips = sys.argv[1], sys.argv[2:]

    device = best_device()
    positions, offsets = [], []
    for clip in clips:
        print(f"calibrate_zones: sampling {clip} (device={device})...")
        p, o = collect_samples(clip, device)
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
