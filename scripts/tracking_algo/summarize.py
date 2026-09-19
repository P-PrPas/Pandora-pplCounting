"""Tiles every tracker's already-rendered comparison clip for a given
(version, clip) into one grid video for direct side-by-side viewing, with a
dashboard-style overlay (big tracker name + live IN/OUT, reusing this
project's existing presentation palette) replacing the tiny in-frame chip,
plus a live leaderboard panel in the 6th cell. Pure post-processing on
cv2.VideoCapture-decoded frames - no model inference, so it can't drift from
what each individual tracker cell actually showed.

Usage: python3 scripts/tracking_algo/summarize.py [--version v1|v2|v3] [--clip N]
(no flags = every version x every clip)
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "v1"))
from presentation import BG, MINT, AMBER, MUTED, WHITE, PANEL, panel, text  # noqa: E402

from trackers import TRACKER_NAMES

REPO_ROOT = HERE.parent.parent
RESULTS_DIR = REPO_ROOT / "data" / "results" / "tracking_algo"
GRID_COLS, GRID_ROWS = 3, 2   # 5 trackers + 1 leaderboard panel
GUTTER = 10
VIDEO_W, VIDEO_H = 720, 405   # each tracker's frame downscaled to this (16:9)
HEADER_H = 64
CELL_W, CELL_H = VIDEO_W, HEADER_H + VIDEO_H

DISPLAY_NAMES = {
    "bytetrack": "ByteTrack", "botsort": "BoT-SORT", "ocsort": "OC-SORT",
    "deepocsort": "DeepOCSORT", "mcbyteplusplus": "McByte++",
}
ACCENTS = {  # BGR, distinct per tracker, drawn from this project's own palette
    "bytetrack": MUTED, "botsort": (255, 210, 40), "ocsort": MINT,
    "deepocsort": AMBER, "mcbyteplusplus": (196, 114, 255),
}


def load_events(tracker, version, clip_name):
    path = RESULTS_DIR / tracker / version / f"{clip_name}.csv"
    with open(path) as f:
        events = [(int(r["frame"]), r["direction"]) for r in csv.DictReader(f)]
    return sorted(events, key=lambda e: e[0])


def draw_tile(frame, name, in_count, out_count):
    accent = ACCENTS[name]
    cell = np.full((CELL_H, CELL_W, 3), BG, dtype=np.uint8)
    panel(cell, 0, 0, CELL_W, HEADER_H, PANEL, radius=10)
    cv2.rectangle(cell, (0, 0), (6, HEADER_H), accent, -1)  # accent stripe = at-a-glance tracker ID
    text(cell, DISPLAY_NAMES[name], 20, 12, 26, accent, True)
    text(cell, f"IN {in_count}", CELL_W - 190, 16, 22, MINT, True)
    text(cell, f"OUT {out_count}", CELL_W - 95, 16, 22, AMBER, True)
    cell[HEADER_H:, :] = cv2.resize(frame, (VIDEO_W, VIDEO_H))
    cv2.rectangle(cell, (0, HEADER_H), (CELL_W - 1, CELL_H - 1), accent, 2)
    return cell


def draw_leaderboard(counts, version, clip_name):
    cell = np.full((CELL_H, CELL_W, 3), PANEL, dtype=np.uint8)
    text(cell, "LIVE COMPARISON", 20, 14, 24, WHITE, True)
    text(cell, f"{version} - {clip_name}", 20, 44, 16, MUTED)
    row_h = (CELL_H - 80) // len(TRACKER_NAMES)
    for i, name in enumerate(TRACKER_NAMES):
        y = 80 + i * row_h
        cv2.rectangle(cell, (20, y + 6), (36, y + 22), ACCENTS[name], -1)
        text(cell, DISPLAY_NAMES[name], 46, y, 19, WHITE, True)
        in_c, out_c = counts[name]
        text(cell, f"IN {in_c}", CELL_W - 190, y, 19, MINT, True)
        text(cell, f"OUT {out_c}", CELL_W - 95, y, 19, AMBER, True)
    return cell


def build_grid(version, clip_name):
    caps = []
    for name in TRACKER_NAMES:
        f = RESULTS_DIR / name / version / f"{clip_name}.mp4"
        if not f.exists():
            print(f"summarize: skipping {version}/{clip_name} - missing {f}")
            return
        caps.append(cv2.VideoCapture(str(f)))
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 25.0
    events = {name: load_events(name, version, clip_name) for name in TRACKER_NAMES}
    ptr = {name: 0 for name in TRACKER_NAMES}
    counts = {name: [0, 0] for name in TRACKER_NAMES}

    out_dir = RESULTS_DIR / "summarize" / version
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip_name}.mp4"
    canvas_w = GUTTER + GRID_COLS * (CELL_W + GUTTER)
    canvas_h = GUTTER + GRID_ROWS * (CELL_H + GUTTER)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas_w, canvas_h))

    n_frames = 0
    while True:
        frames = []
        for cap in caps:
            ok, frame = cap.read()
            if not ok:
                frames = None
                break
            frames.append(frame)
        if frames is None:
            break

        for name in TRACKER_NAMES:
            evs, i = events[name], ptr[name]
            while i < len(evs) and evs[i][0] <= n_frames:
                counts[name][0 if evs[i][1] == "in" else 1] += 1
                i += 1
            ptr[name] = i

        tiles = [draw_tile(f, name, *counts[name]) for f, name in zip(frames, TRACKER_NAMES)]
        tiles.append(draw_leaderboard(counts, version, clip_name))

        canvas = np.full((canvas_h, canvas_w, 3), BG, dtype=np.uint8)
        for i, tile in enumerate(tiles):
            row, col = divmod(i, GRID_COLS)
            y = GUTTER + row * (CELL_H + GUTTER)
            x = GUTTER + col * (CELL_W + GUTTER)
            canvas[y:y + CELL_H, x:x + CELL_W] = tile
        writer.write(canvas)
        n_frames += 1

    writer.release()
    for cap in caps:
        cap.release()
    print(f"summarize: {version}/{clip_name} -> {out_path} ({n_frames} frames)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", choices=["v1", "v2", "v3"])
    p.add_argument("--clip", type=int, help="1-based clip index")
    args = p.parse_args()

    versions = [args.version] if args.version else ["v1", "v2", "v3"]
    clips = [f"clip{args.clip}"] if args.clip else [f"clip{i}" for i in (1, 2)]
    for version in versions:
        for clip_name in clips:
            build_grid(version, clip_name)


if __name__ == "__main__":
    main()
