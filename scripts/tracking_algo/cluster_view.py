"""One-off tool: for ground-truth review, extracts a filmstrip of frames
around a given frame-number window from a rendered clip, cropped tight to the
doorway/zone ROI so crossing details are legible when tiled.

Usage: python3 scripts/tracking_algo/cluster_view.py <video> <out_png> <start_frame> <end_frame> [--stride 3]
"""
import argparse

import cv2
import numpy as np

ROI = (650, 50, 1700, 700)  # x0, y0, x1, y1 - doorway/zone area, both clips share this framing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("out_png")
    p.add_argument("start", type=int)
    p.add_argument("end", type=int)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--cols", type=int, default=4)
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    x0, y0, x1, y1 = ROI
    cw, ch = x1 - x0, y1 - y0

    frames_wanted = list(range(args.start, args.end + 1, args.stride))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames_wanted[0])
    cells = []
    idx = frames_wanted[0]
    want_set = set(frames_wanted)
    last_wanted = frames_wanted[-1]
    while idx <= last_wanted:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want_set:
            crop = frame[y0:y1, x0:x1].copy()
            label = f"f{idx} t={idx/fps:.2f}s"
            cv2.rectangle(crop, (0, 0), (170, 22), (0, 0, 0), -1)
            cv2.putText(crop, label, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(crop)
        idx += 1

    cols = args.cols
    rows = (len(cells) + cols - 1) // cols
    sheet = np.zeros((rows * ch, cols * cw, 3), dtype=np.uint8)
    for i, c in enumerate(cells):
        r, cc = divmod(i, cols)
        sheet[r*ch:(r+1)*ch, cc*cw:(cc+1)*cw] = c
    cv2.imwrite(args.out_png, sheet)
    print(f"wrote {len(cells)} frames ({frames_wanted[0]}-{frames_wanted[-1]}) to {args.out_png}")


if __name__ == "__main__":
    main()
