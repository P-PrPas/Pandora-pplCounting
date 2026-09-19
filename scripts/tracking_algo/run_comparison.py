"""Runs every tracker (trackers.py) against every logic version (v1/v2/v3) on
every dataset clip, writing a lightweight (no dashboard - just the existing
zone/track overlay plus a small burned-in label) annotated video + event-log
csv per combo to data/results/tracking_algo/<tracker>/<version>/<clip>.{mp4,csv}.

Usage: python3 scripts/tracking_algo/run_comparison.py [--tracker NAME] [--version v1|v2|v3] [--clip N]
(no flags = full sweep: every tracker x every version x every clip)
"""
import argparse
import csv
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch

# ponytail: osnet Re-ID runs many small per-box forward passes per frame;
# torch's default multi-threaded intra-op pool causes severe contention on
# these tiny ops (observed: 600%+ CPU, minutes per hundred frames). Single
# thread is faster here since there's no big matmul to parallelize.
torch.set_num_threads(1)

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "v1"))
sys.path.insert(0, str(HERE.parent / "v2"))
sys.path.insert(0, str(HERE.parent / "v3"))

from counting import ZoneCounter, best_device                            # noqa: E402
from presentation import WHITE, chip                                     # noqa: E402
from foot_tracker import render_v1_frame                                 # noqa: E402
from head_tracker import render_v2_frame                                 # noqa: E402
from head_tracker_v3 import render_v3_frame                              # noqa: E402
from people_counter import ZONE_A as ZONE_A_FOOT, ZONE_B as ZONE_B_FOOT   # noqa: E402
from people_counter_v2 import load_head_zones as load_zones_v2           # noqa: E402
from people_counter_v3 import load_head_zones as load_zones_v3           # noqa: E402

import detect_cache
from trackers import TRACKER_NAMES, build_tracker
from shim import FakeResult

REPO_ROOT = HERE.parent.parent
DATASET_DIR = REPO_ROOT / "data" / "dataset"
RESULTS_DIR = REPO_ROOT / "data" / "results" / "tracking_algo"
CLIPS = sorted(DATASET_DIR.glob("*.mp4"))
ZONES_V2 = HERE.parent / "v2" / "zones_head.json"
ZONES_V3 = HERE.parent / "v3" / "zones_head_v3.json"
TRAIL_LEN, HIGHLIGHT_FRAMES = 30, 20


def _v2_keypoints(tracked, cached_kpts):
    """Re-attach each surviving tracked box's pose keypoints by det_idx (col
    7). No match (occluded/dropped this frame) -> all-zero keypoints, which
    neck_point() already treats as "not confidently visible" (same as any
    other low-confidence pose frame) - see head_calibration.py.
    """
    out = np.zeros((len(tracked), 17, 3))
    for i, det_idx in enumerate(tracked[:, 7].astype(int)):
        if 0 <= det_idx < len(cached_kpts):
            out[i] = cached_kpts[det_idx]
    return out


def run_one(tracker_name, version, clip_path, device):
    clip_name = f"clip{CLIPS.index(clip_path) + 1}"
    out_dir = RESULTS_DIR / tracker_name / version
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = out_dir / clip_name

    frames_meta = detect_cache.build(version, clip_path, device)
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(f"{out_prefix}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracker = build_tracker(tracker_name, device)
    counter = ZoneCounter()
    trails = defaultdict(lambda: deque(maxlen=TRAIL_LEN))
    recent_events = {}
    events = []
    in_count = out_count = 0

    zone_a_head = zone_b_head = None
    if version == "v2":
        zone_a_head, zone_b_head = load_zones_v2(ZONES_V2)
    elif version == "v3":
        zone_a_head, zone_b_head = load_zones_v3(ZONES_V3)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        meta = frames_meta[frame_idx]
        tracked = tracker.update(meta["boxes"], frame)  # [x1,y1,x2,y2,id,conf,cls,det_idx]

        if version == "v1":
            r = FakeResult(frame, tracked[:, :4], tracked[:, 4])
            out_frame, active, fired = render_v1_frame(
                r, frame_idx, ZONE_A_FOOT, ZONE_B_FOOT, counter, trails, recent_events, HIGHLIGHT_FRAMES)
        elif version == "v2":
            kpts = _v2_keypoints(tracked, meta["keypoints"])
            r = FakeResult(frame, tracked[:, :4], tracked[:, 4], keypoints=kpts)
            out_frame, active, fired = render_v2_frame(
                r, frame_idx, zone_a_head, zone_b_head, ZONE_A_FOOT, ZONE_B_FOOT,
                counter, trails, recent_events, HIGHLIGHT_FRAMES)
        else:
            r = FakeResult(frame, tracked[:, :4], tracked[:, 4])
            out_frame, active, fired = render_v3_frame(
                r, frame_idx, zone_a_head, zone_b_head, ZONE_A_FOOT, ZONE_B_FOOT,
                counter, trails, recent_events, HIGHLIGHT_FRAMES, meta["person_boxes"])

        for tid, event in fired:
            in_count += event == "in"
            out_count += event == "out"
            events.append((round(frame_idx / fps, 2), frame_idx, tid, event))

        chip(out_frame, f"{tracker_name}  IN={in_count:02d} OUT={out_count:02d}", 20, 20, WHITE)
        writer.write(out_frame)
        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"  ...{frame_idx} frames")

    writer.release()
    cap.release()
    with open(f"{out_prefix}.csv", "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["time_s", "frame", "track_id", "direction"])
        cw.writerows(events)
    print(f"{tracker_name}/{version}/{clip_name}: IN={in_count} OUT={out_count} ({len(events)} events) "
          f"-> {out_prefix}.mp4")
    return in_count, out_count, len(events)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracker", choices=TRACKER_NAMES)
    p.add_argument("--version", choices=["v1", "v2", "v3"])
    p.add_argument("--clip", type=int, help="1-based clip index")
    args = p.parse_args()

    device = best_device()
    print(f"run_comparison: device={device}")
    trackers = [args.tracker] if args.tracker else TRACKER_NAMES
    versions = [args.version] if args.version else ["v1", "v2", "v3"]
    clips = [CLIPS[args.clip - 1]] if args.clip else CLIPS

    summary = {}
    for version in versions:
        for clip_path in clips:
            for tracker_name in trackers:
                summary[(tracker_name, version, clip_path.stem)] = run_one(tracker_name, version, clip_path, device)

    print("\n=== summary ===")
    for key, (i, o, e) in summary.items():
        print(*key, "IN", i, "OUT", o, "events", e)


if __name__ == "__main__":
    main()
