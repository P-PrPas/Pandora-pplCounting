"""Background thread: sample the live RTSP stream with a pose model to
measure this camera's real foot->neck offset (see scripts/v2/head_calibration.py).
One-time-per-camera step for the V2 (head-tracking) preset - run once,
cached, so a live V2 session afterwards is a single real-time pass, not a
prescan.
"""
import sys
from pathlib import Path

from PySide6.QtCore import QThread, Signal

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "v1"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "v2"))
from counting import best_device            # noqa: E402
from head_calibration import collect_pose_samples  # noqa: E402

POSE_MODEL_PATH = Path(__file__).parent.parent / "scripts" / "v2" / "yolo11s-pose.pt"
# ponytail: fixed sample budget rather than a wall-clock timer - simpler, and
# "enough frames" is what actually matters for calibration quality, not
# elapsed time (which varies with the stream's real framerate).
CALIBRATION_FRAMES = 500  # ~20-30s of live footage at a typical 15-25 fps stream


class CalibrationWorker(QThread):
    progress = Signal(int)              # running sample count
    finished_ok = Signal(list, list)    # positions, offsets
    error = Signal(str)

    def __init__(self, source, parent=None):
        super().__init__(parent)
        self.source = source
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            device = best_device()
            positions, offsets = collect_pose_samples(
                self.source, POSE_MODEL_PATH, device, max_frames=CALIBRATION_FRAMES,
                on_frame=lambda i, n: self.progress.emit(n),
                should_stop=lambda: self._stop)
            self.finished_ok.emit(positions, offsets)
        except Exception as e:  # trust boundary: RTSP can die anytime mid-calibration
            self.error.emit(str(e))
