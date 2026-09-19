"""Adapter: wraps the vendored McBytePlusPlusTracker (with-reid, no mask —
see the user's chosen integration scope) behind the same
`.update(dets, frame) -> [x1,y1,x2,y2,id,conf,cls,det_idx]` shape boxmot's
trackers already return, so run_comparison.py never has to special-case it.

McByte++'s core tracker doesn't preserve the input row index of a matched
detection (only tlwh/score survive into its STrack), unlike boxmot. det_idx
is recovered here by IoU-matching each output track's box back against this
frame's input dets - good enough to re-attach v2's pose keypoints by index;
no match above IOU_MATCH_MIN just means no keypoint this frame (handled the
same as any other occluded/off-pose frame by render_v2_frame).
"""
from types import SimpleNamespace

import numpy as np

from .tracker import McBytePlusPlusTracker

IOU_MATCH_MIN = 0.3


def _iou(a, b):
    """a: (N,4), b: (M,4) xyxy -> (N,M) IoU matrix."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1, iy1 = np.maximum(ax1, bx1), np.maximum(ay1, by1)
    ix2, iy2 = np.minimum(ax2, bx2), np.minimum(ay2, by2)
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = ((ax2 - ax1) * (ay2 - ay1))
    area_b = ((bx2 - bx1) * (by2 - by1))
    return inter / np.clip(area_a + area_b - inter, 1e-6, None)


class McByteAdapter:
    def __init__(self, device="cpu", reid_weights="osnet_x1_0_market1501.pt"):
        # ponytail: resolve/download through boxmot's own catalog+downloader
        # (already vetted, already a dependency) instead of re-implementing
        # gdown handling here - see trackers.py for why this checkpoint
        # (general-pedestrian, not McByte++'s own sports-tuned default).
        from boxmot import BotSort
        from boxmot.reid.core.runtime import WEIGHTS
        reid_path = WEIGHTS / reid_weights
        if not reid_path.exists():
            BotSort(reid_weights=reid_weights, device=device)  # side effect: downloads it

        # Values match this project's own tuned bytetrack_custom.yaml (crowded-
        # doorway, ~60-frame occlusion buffer at 25fps) so all trackers here see
        # the same domain-appropriate thresholds, not each one's stock default.
        args = SimpleNamespace(
            track_thresh=0.25, track_buffer=60,
            cmc_method="none", cmc_downscale=2, cmc_interval=1,  # static camera - CMC buys nothing
            reid_model_path=str(reid_path), reid_device=device,
        )
        self.tracker = McBytePlusPlusTracker(args, save_folder=None, frame_rate=25, enable_logging=False)

    def update(self, dets, frame):
        h, w = frame.shape[:2]
        if len(dets) == 0:
            output_stracks, *_ = self.tracker.update(
                np.zeros((0, 5)), (h, w), (h, w), None, {}, frame, "none", dets_from_file=True)
            return np.zeros((0, 8))

        output_results = np.asarray(dets)[:, :5]  # x1,y1,x2,y2,conf (drop cls - single-class per version)
        output_stracks, *_ = self.tracker.update(
            output_results, (h, w), (h, w), None, {}, frame, "none", dets_from_file=True)
        if not output_stracks:
            return np.zeros((0, 8))

        track_boxes = np.array([t.tlbr for t in output_stracks])
        det_boxes = np.asarray(dets)[:, :4]
        ious = _iou(track_boxes, det_boxes)
        best_idx = ious.argmax(axis=1)
        best_iou = ious.max(axis=1)

        rows = []
        for t, idx, iou in zip(output_stracks, best_idx, best_iou):
            x1, y1, x2, y2 = t.tlbr
            det_idx = int(idx) if iou >= IOU_MATCH_MIN else -1
            cls = dets[det_idx][5] if det_idx >= 0 else 0
            rows.append([x1, y1, x2, y2, t.export_id, t.score, cls, det_idx])
        return np.array(rows)
