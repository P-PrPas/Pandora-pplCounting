"""One-off tool: tiles sampled frames from a rendered clip into contact-sheet
grids (frame number burned into each cell) for fast manual ground-truth
review, instead of extracting/viewing hundreds of single frames.

Usage: python3 scripts/tracking_algo/contact_sheet.py <video> <out_dir> [--stride 8] [--grid 4x4]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("out_dir")
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--grid", default="4x4")
    args = p.parse_args()
    rows, cols = map(int, args.grid.split("x"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cell_w, cell_h = 480, 270
    per_sheet = rows * cols

    sheet_idx = 0
    cells = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.stride == 0:
            cell = cv2.resize(frame, (cell_w, cell_h))
            label = f"f{frame_idx} t={frame_idx/fps:.2f}s"
            cv2.rectangle(cell, (0, 0), (150, 20), (0, 0, 0), -1)
            cv2.putText(cell, label, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(cell)
            if len(cells) == per_sheet:
                sheet = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
                for i, c in enumerate(cells):
                    r, cc = divmod(i, cols)
                    sheet[r*cell_h:(r+1)*cell_h, cc*cell_w:(cc+1)*cell_w] = c
                cv2.imwrite(str(out_dir / f"sheet_{sheet_idx:03d}.png"), sheet)
                sheet_idx += 1
                cells = []
        frame_idx += 1
    if cells:
        sheet = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
        for i, c in enumerate(cells):
            r, cc = divmod(i, cols)
            sheet[r*cell_h:(r+1)*cell_h, cc*cell_w:(cc+1)*cell_w] = c
        cv2.imwrite(str(out_dir / f"sheet_{sheet_idx:03d}.png"), sheet)
        sheet_idx += 1
    print(f"wrote {sheet_idx} sheets to {out_dir}")


if __name__ == "__main__":
    main()
