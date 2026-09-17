"""Background thread: sample the live RTSP stream to measure this camera's
real foot->head-level offset - foot->neck for v2 (see
scripts/v2/head_calibration.py), foot->head for v3 (see
scripts/v3/head_calibration_v3.py). One-time-per-camera step for the V2/V3
(head-tracking) presets - run once, cached, so a live session afterwards is
a single real-time pass, not a prescan.
"""
import sys
from pathlib import Path

from PySide6.QtCore import QThread, Signal

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "v1"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "v2"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "v3"))
from counting import best_device                          # noqa: E402
from head_calibration import collect_pose_samples         # noqa: E402
from head_calibration_v3 import collect_head_samples       # noqa: E402

V1_DIR = Path(__file__).parent.parent / "scripts" / "v1"
POSE_MODEL_PATH = Path(__file__).parent.parent / "scripts" / "v2" / "yolo11s-pose.pt"
PERSON_MODEL_PATH = V1_DIR / "yolo11s.pt"
HEAD_MODEL_PATH = Path(__file__).parent.parent / "scripts" / "v3" / "head_detector.pt"
CALIBRATION_FRAMES = 500  # ~20-30s of live footage at a typical 15-25 fps stream


class CalibrationWorker(QThread):
    progress = Signal(int)
    finished_ok = Signal(list, list)
    error = Signal(str)

    def __init__(self, source, logic="v2", parent=None):
        super().__init__(parent)
        self.source = source
        self.logic = logic
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            device = best_device()
            on_frame = lambda i, n: self.progress.emit(n)     # noqa: E731
            should_stop = lambda: self._stop                  # noqa: E731
            if self.logic == "v3":
                positions, offsets = collect_head_samples(
                    self.source, PERSON_MODEL_PATH, HEAD_MODEL_PATH, device,
                    max_frames=CALIBRATION_FRAMES, on_frame=on_frame, should_stop=should_stop)
            else:
                positions, offsets = collect_pose_samples(
                    self.source, POSE_MODEL_PATH, device, max_frames=CALIBRATION_FRAMES,
                    on_frame=on_frame, should_stop=should_stop)
            self.finished_ok.emit(positions, offsets)
        except Exception as e:
            self.error.emit(str(e))
