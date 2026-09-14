"""Background thread running live detect+track+count on the RTSP stream,
reusing the exact same algorithm/rendering as the batch pipeline.
"""
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "v1"))
from counting import ZoneCounter, best_device, classify_point
from presentation import Presentation, draw_track, draw_zones

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts" / "v1"
MODEL_PATH = SCRIPTS_DIR / "yolo11s.pt"
TRACKER_CONFIG = SCRIPTS_DIR / "bytetrack_custom.yaml"

TRAIL_LEN = 30
HIGHLIGHT_FRAMES = 20


class VideoWorker(QThread):
    """Runs model.track() against a live source; emits a rendered frame + counts per detection."""

    frame_ready = Signal(QImage)                      # the full rendered dashboard, ready to display
    event_fired = Signal(float, int, str)            # elapsed_s, track_id, "in"|"out"
    error = Signal(str)
    finished_clean = Signal()

    def __init__(self, source, enter_zone, exit_zone, parent=None):
        super().__init__(parent)
        self.source = source
        # User's "Enter" -> zone_b (MINT/"inside"), "Exit" -> zone_a (AMBER/"outside"),
        # matching classify_point's A/B convention with no changes to ZoneCounter itself.
        self.zone_a = np.array(exit_zone)
        self.zone_b = np.array(enter_zone)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            model = YOLO(MODEL_PATH)
            counter = ZoneCounter()
            trails = defaultdict(lambda: deque(maxlen=TRAIL_LEN))
            recent_events = {}
            events = []  # (elapsed_s, frame_idx, track_id, direction) — feeds the "recent crossings" panel
            in_count = out_count = 0
            start = last_tick = time.monotonic()
            fps = 0.0
            # duration is unknown for a live/indefinite stream — same dashboard, no fixed end.
            presentation = Presentation(0, label_a="EXIT", label_b="ENTER",
                                         footer="Live RTSP feed · AI-assisted counting")

            # ponytail: same call shape as scripts/v1/people_counter.py. Device auto-picked
            # (cuda > mps > cpu) — same weights/precision either way, so this is a free
            # speedup, not an accuracy tradeoff. conf=0.1 for the same occlusion mitigation.
            device = best_device()
            print(f"video_worker: running inference on device={device}")
            results = model.track(self.source, classes=[0], conf=0.1, tracker=str(TRACKER_CONFIG),
                                   stream=True, verbose=False, device=device)
            for frame_idx, r in enumerate(results):
                if self._stop:
                    break
                frame = r.orig_img
                occupied = []
                draw_zones(frame, self.zone_a, self.zone_b, occupied, label_a="EXIT", label_b="ENTER")
                elapsed = time.monotonic() - start
                active = 0

                # EMA smoothing so the readout doesn't flicker frame to frame.
                now = time.monotonic()
                inst_fps = 1 / max(1e-6, now - last_tick)
                fps = inst_fps if frame_idx == 0 else fps * 0.9 + inst_fps * 0.1
                last_tick = now

                if r.boxes is not None and r.boxes.id is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    ids = r.boxes.id.cpu().numpy().astype(int)
                    active = len(ids)
                    for box, tid in zip(boxes, ids):
                        x1, y1, x2, y2 = box
                        foot = (int((x1 + x2) / 2), int(y2))
                        trails[tid].append(foot)

                        event = counter.update(tid, classify_point(foot, self.zone_a, self.zone_b))
                        if event:
                            in_count += event == "in"
                            out_count += event == "out"
                            recent_events[tid] = (event, frame_idx)
                            events.append((round(elapsed, 2), frame_idx, int(tid), event))
                            self.event_fired.emit(elapsed, int(tid), event)

                        last = recent_events.get(tid)
                        highlighted = bool(last) and frame_idx - last[1] < HIGHLIGHT_FRAMES
                        draw_track(frame, box, tid, foot, trails[tid], highlighted,
                                   last[0] if last else None, occupied)

                # duration tracks elapsed itself (no fixed end for a live session) so the
                # dashboard's time readout and progress bar read as "session running time".
                presentation.duration = elapsed
                canvas = presentation.render(frame, in_count, out_count, events, elapsed, active, fps=fps)
                rgb = canvas[:, :, ::-1].copy()
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
                self.frame_ready.emit(qimg)

            self.finished_clean.emit()
        except Exception as e:  # trust boundary: RTSP can die anytime, surface it, don't crash the app
            self.error.emit(str(e))
