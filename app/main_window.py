"""Main window: zone-drawing step, then live video + HUD once running."""
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
    QRadioButton, QStackedWidget, QVBoxLayout, QWidget,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "v1"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "v2"))
from presentation import AMBER, BG, LINE, MINT, MUTED, PANEL, WHITE
from head_calibration import head_level_zones

import calibration_store
from calibration_worker import CalibrationWorker
from video_worker import VideoWorker
from zone_canvas import ZoneCanvas


def _hex(bgr):
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


STYLE = f"""
QWidget {{ background: {_hex(BG)}; color: {_hex(WHITE)}; font-family: 'DejaVu Sans'; font-size: 13px; }}
QLabel#title {{ font-size: 20px; font-weight: bold; }}
QLabel#subtitle {{ color: {_hex(MUTED)}; }}
QFrame, .panel {{ background: {_hex(PANEL)}; border-radius: 10px; }}
QPushButton {{
    background: {_hex(PANEL)}; border: 1px solid {_hex(LINE)}; border-radius: 8px;
    padding: 10px 20px; font-weight: bold;
}}
QPushButton:hover {{ border-color: {_hex(MINT)}; }}
QPushButton:disabled {{ color: {_hex(MUTED)}; }}
QPushButton#primary {{ background: {_hex(MINT)}; color: {_hex(BG)}; border: none; }}
QPushButton#danger {{ background: {_hex(AMBER)}; color: {_hex(BG)}; border: none; }}
QPushButton#primary:disabled, QPushButton#danger:disabled {{
    background: {_hex(PANEL)}; color: {_hex(MUTED)};
}}
"""


class MainWindow(QMainWindow):
    def __init__(self, source, snapshot_bgr):
        super().__init__()
        self.source = source
        self.worker = None
        self.calibration_worker = None
        # keyed by logic - v2's neck offset and v3's head offset aren't interchangeable
        self.calib_cache = {"v2": calibration_store.load(source, "v2"),
                             "v3": calibration_store.load(source, "v3")}
        self.setWindowTitle("Doorway People Counter — Live POC")
        self.resize(1400, 860)
        self.setStyleSheet(STYLE)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # -- step 1: zone drawing --
        self.canvas = ZoneCanvas(snapshot_bgr, source)
        self.canvas.changed.connect(self._sync_zone_controls)
        draw_page = QWidget()
        dlay = QVBoxLayout(draw_page)
        header = QLabel("Step 1 — Draw the Enter and Exit zones")
        header.setObjectName("title")
        sub = QLabel("Click points to trace a zone (3+ points), then press Enter ↵ or click Finish. "
                      "Draw the Enter zone first, then the Exit zone.")
        sub.setObjectName("subtitle")
        dlay.addWidget(header)
        dlay.addWidget(sub)
        dlay.addWidget(self.canvas, 1)

        controls = QHBoxLayout()
        self.undo_btn = QPushButton("Undo point")
        self.finish_btn = QPushButton("Finish zone")
        self.finish_btn.setObjectName("primary")
        self.reset_btn = QPushButton("Reset")
        self.to_run_btn = QPushButton("Continue →")
        self.to_run_btn.setObjectName("primary")
        self.undo_btn.clicked.connect(self.canvas.undo_point)
        self.finish_btn.clicked.connect(self.canvas.finish_current)
        self.reset_btn.clicked.connect(self.canvas.reset)
        self.to_run_btn.clicked.connect(self._go_to_run_page)
        for b in (self.undo_btn, self.finish_btn, self.reset_btn):
            controls.addWidget(b)
        controls.addStretch()
        controls.addWidget(self.to_run_btn)
        dlay.addLayout(controls)
        self.stack.addWidget(draw_page)

        # Enter finishes the current zone — no need to reach for the mouse after the last click.
        for key in (Qt.Key_Return, Qt.Key_Enter):
            QShortcut(QKeySequence(key), self, activated=self.canvas.finish_current)

        # -- step 2: live run --
        run_page = QWidget()
        rlay = QVBoxLayout(run_page)
        self.video_label = QLabel("Press Start to begin live inference")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(1280, 640)
        self.video_label.setStyleSheet(f"background:{_hex(BG)}; border:1px solid {_hex(LINE)}; border-radius:10px;")
        rlay.addWidget(self.video_label, 1)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Logic:"))
        self.v1_radio = QRadioButton("V1 · Foot tracking")
        self.v2_radio = QRadioButton("V2 · Head tracking")
        self.v3_radio = QRadioButton("V3 · Head detection")
        self.logic_group = QButtonGroup(self)
        self.logic_group.addButton(self.v1_radio)
        self.logic_group.addButton(self.v2_radio)
        self.logic_group.addButton(self.v3_radio)
        self.calibrate_btn = QPushButton("Calibrate (required for V2/V3)")
        self.calib_status = QLabel("Not calibrated")
        self.calib_status.setObjectName("subtitle")
        self.v1_radio.toggled.connect(self._sync_preset_controls)
        self.v2_radio.toggled.connect(self._sync_preset_controls)
        self.v3_radio.toggled.connect(self._sync_preset_controls)
        self.calibrate_btn.clicked.connect(self._toggle_calibration)
        preset_row.addWidget(self.v1_radio)
        preset_row.addWidget(self.v2_radio)
        preset_row.addWidget(self.v3_radio)
        preset_row.addWidget(self.calibrate_btn)
        preset_row.addWidget(self.calib_status)
        preset_row.addStretch()
        rlay.addLayout(preset_row)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("▶  Start Live Count")
        self.start_btn.setObjectName("primary")
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        self.redraw_btn = QPushButton("Redraw Zones")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        self.redraw_btn.clicked.connect(self._redraw)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.redraw_btn)
        btn_row.addStretch()
        self.status = QLabel("Idle")
        self.status.setObjectName("subtitle")
        btn_row.addWidget(self.status)
        rlay.addLayout(btn_row)
        self.stack.addWidget(run_page)

        self._sync_zone_controls()
        self.v1_radio.setChecked(True)
        self._sync_preset_controls()

    def _sync_zone_controls(self):
        self.finish_btn.setEnabled(self.canvas.can_finish_current)
        zone = "Exit" if self.canvas.is_last_zone else "Enter"
        self.finish_btn.setText(f"Finish {zone} zone  (Enter ↵)")
        self.to_run_btn.setEnabled(self.canvas.done)

    def _go_to_run_page(self):
        self.stack.setCurrentIndex(1)

    def _redraw(self):
        self._stop()
        self.canvas.reset()
        self.stack.setCurrentIndex(0)

    def _current_logic(self):
        if self.v3_radio.isChecked():
            return "v3"
        if self.v2_radio.isChecked():
            return "v2"
        return "v1"

    def _sync_preset_controls(self):
        logic = self._current_logic()
        needs_calib = logic in ("v2", "v3")
        self.calibrate_btn.setVisible(needs_calib)
        self.calib_status.setVisible(needs_calib)
        if needs_calib:
            positions, _ = self.calib_cache[logic]
            if positions:
                self.calibrate_btn.setText("Recalibrate")
                self.calib_status.setText(f"Calibrated ({len(positions)} samples)")
            else:
                self.calibrate_btn.setText(f"Calibrate (required for {logic.upper()})")
                self.calib_status.setText("Not calibrated")
        if self.worker is None and self.calibration_worker is None:
            self.start_btn.setEnabled(not needs_calib or bool(self.calib_cache[logic][0]))

    def _toggle_calibration(self):
        if self.calibration_worker:
            self.calibration_worker.stop()
            return
        logic = self._current_logic()
        self.calibration_worker = CalibrationWorker(self.source, logic=logic)
        self.calibration_worker.progress.connect(self._on_calibration_progress)
        self.calibration_worker.finished_ok.connect(self._on_calibration_done)
        self.calibration_worker.error.connect(self._on_calibration_error)
        self.calibration_worker.start()
        self.calibrate_btn.setText("Cancel calibrating…")
        self.start_btn.setEnabled(False)
        self.v1_radio.setEnabled(False)
        self.v2_radio.setEnabled(False)
        self.v3_radio.setEnabled(False)
        self.calib_status.setText("Calibrating… 0 samples so far")

    def _on_calibration_progress(self, n):
        self.calib_status.setText(f"Calibrating… {n} samples so far")

    def _on_calibration_done(self, positions, offsets):
        logic = self._current_logic()
        if not positions:
            self._finish_calibration("Not calibrated (no samples collected)")
            return
        self.calib_cache[logic] = (positions, offsets)
        calibration_store.save(self.source, logic, positions, offsets)
        self._finish_calibration(f"Calibrated ({len(positions)} samples)")

    def _on_calibration_error(self, message):
        self._finish_calibration("Not calibrated")
        QMessageBox.critical(self, "Calibration failed", f"Live calibration stopped:\n{message}")

    def _finish_calibration(self, status_text):
        self.calibration_worker = None
        self.v1_radio.setEnabled(True)
        self.v2_radio.setEnabled(True)
        self.v3_radio.setEnabled(True)
        self.calib_status.setText(status_text)
        self._sync_preset_controls()

    def _start(self):
        logic = self._current_logic()
        kwargs = {}
        if logic in ("v2", "v3"):
            positions, offsets = self.calib_cache[logic]
            zone_a_head, zone_b_head = head_level_zones(
                self.canvas.exit_points, self.canvas.enter_points, positions, offsets)
            kwargs = dict(logic=logic, exit_zone_head=zone_a_head, enter_zone_head=zone_b_head)
        self.worker = VideoWorker(self.source, self.canvas.enter_points, self.canvas.exit_points, **kwargs)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.error.connect(self._on_error)
        self.worker.finished_clean.connect(self._on_finished)
        self.worker.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.v1_radio.setEnabled(False)
        self.v2_radio.setEnabled(False)
        self.v3_radio.setEnabled(False)
        self.calibrate_btn.setEnabled(False)
        self.status.setText("Running…")

    def _stop(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(3000)
            self.worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.v1_radio.setEnabled(True)
        self.v2_radio.setEnabled(True)
        self.v3_radio.setEnabled(True)
        self.calibrate_btn.setEnabled(True)
        self.status.setText("Idle")
        self._sync_preset_controls()

    def _on_frame(self, qimg):
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)

    def _on_error(self, message):
        self._stop()
        QMessageBox.critical(self, "Stream error", f"Live inference stopped:\n{message}")

    def _on_finished(self):
        self.status.setText("Stream ended")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.v1_radio.setEnabled(True)
        self.v2_radio.setEnabled(True)
        self.v3_radio.setEnabled(True)
        self.calibrate_btn.setEnabled(True)
        self._sync_preset_controls()

    def closeEvent(self, event):
        self._stop()
        if self.calibration_worker:
            self.calibration_worker.stop()
            self.calibration_worker.wait(3000)
        super().closeEvent(event)
