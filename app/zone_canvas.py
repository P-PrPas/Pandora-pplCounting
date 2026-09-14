"""Click-to-draw zone editor: shows a camera snapshot, captures two polygons
(Enter, then Exit) by mouse click, and persists them per-RTSP-source so the
customer doesn't have to redraw on every launch.
"""
import json
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QPointF, QSize, Signal
from PySide6.QtGui import QColor, QFontMetrics, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "v1"))
from presentation import AMBER, MINT, WHITE  # BGR tuples, single source of truth for the palette

ZONES_FILE = Path(__file__).parent / "zones.json"


def _qcolor(bgr, alpha=255):
    b, g, r = bgr
    return QColor(r, g, b, alpha)


ENTER_COLOR = _qcolor(MINT)   # matches presentation.py: "inside" / entered
EXIT_COLOR = _qcolor(AMBER)   # matches presentation.py: "outside" / exited
WHITE_COLOR = _qcolor(WHITE)


class ZoneCanvas(QWidget):
    """Two-polygon editor over a static snapshot. Points are stored in original
    image (camera) coordinates, independent of how the widget is scaled/resized.
    """

    changed = Signal()  # emitted after any edit; MainWindow reads canvas state to update its own UI

    def __init__(self, snapshot_bgr, rtsp_url, parent=None):
        super().__init__(parent)
        self.rtsp_url = rtsp_url
        self.img_h, self.img_w = snapshot_bgr.shape[:2]
        self.image = self._to_qimage(snapshot_bgr)
        self.enter_points = []   # list[(x, y)] in image coords
        self.exit_points = []
        self.drawing_exit = False
        self.done = False
        self.hover_pos = None
        self.setMouseTracking(True)
        self.setMinimumSize(480, 270)
        self._load_saved()

    # -- geometry: image <-> widget coordinate mapping (letterboxed, aspect-preserved) --

    def _fit(self):
        avail_w, avail_h = self.width(), self.height()
        scale = min(avail_w / self.img_w, avail_h / self.img_h) if self.img_w and self.img_h else 1
        draw_w, draw_h = self.img_w * scale, self.img_h * scale
        x0, y0 = (avail_w - draw_w) / 2, (avail_h - draw_h) / 2
        return x0, y0, scale

    def _to_widget(self, pt):
        x0, y0, scale = self._fit()
        return QPointF(x0 + pt[0] * scale, y0 + pt[1] * scale)

    def _to_image(self, pos):
        x0, y0, scale = self._fit()
        if scale == 0:
            return None
        x, y = (pos.x() - x0) / scale, (pos.y() - y0) / scale
        if 0 <= x <= self.img_w and 0 <= y <= self.img_h:
            return int(x), int(y)
        return None

    @staticmethod
    def _to_qimage(frame_bgr):
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()

    # -- state --

    @property
    def current_points(self):
        return self.exit_points if self.drawing_exit else self.enter_points

    @property
    def current_zone_name(self):
        return "Exit" if self.drawing_exit else "Enter"

    @property
    def can_finish_current(self):
        return not self.done and len(self.current_points) >= 3

    @property
    def is_last_zone(self):
        return self.drawing_exit

    def undo_point(self):
        if self.done or not self.current_points:
            return
        self.current_points.pop()
        self.changed.emit()
        self.update()

    def finish_current(self):
        if not self.can_finish_current:
            return
        if not self.drawing_exit:
            self.drawing_exit = True
        else:
            self.done = True
            self._save()
        self.changed.emit()
        self.update()

    def reset(self):
        self.enter_points, self.exit_points = [], []
        self.drawing_exit = False
        self.done = False
        self.changed.emit()
        self.update()

    def _save(self):
        ZONES_FILE.write_text(json.dumps({
            "rtsp_url": self.rtsp_url,
            "enter": self.enter_points,
            "exit": self.exit_points,
        }))

    def _load_saved(self):
        # ponytail: only ever caches the single most-recently-drawn source; add a
        # keyed-by-source dict in zones.json if multiple cameras need remembering.
        if not ZONES_FILE.exists():
            return
        try:
            data = json.loads(ZONES_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return
        if data.get("rtsp_url") != self.rtsp_url:
            return
        enter, exit_ = data.get("enter", []), data.get("exit", [])
        if len(enter) >= 3 and len(exit_) >= 3:
            self.enter_points = [tuple(p) for p in enter]
            self.exit_points = [tuple(p) for p in exit_]
            self.drawing_exit = True
            self.done = True

    # -- events --

    def mousePressEvent(self, event):
        if self.done or event.button() != Qt.LeftButton:
            return
        pt = self._to_image(event.position())
        if pt is not None:
            self.current_points.append(pt)
            self.changed.emit()
            self.update()

    def mouseMoveEvent(self, event):
        self.hover_pos = event.position()
        if not self.done:
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        x0, y0, scale = self._fit()
        target = self.image.rect()
        target.moveTo(int(x0), int(y0))
        target.setSize(QSize(int(self.img_w * scale), int(self.img_h * scale)))
        painter.drawImage(target, self.image)

        self._draw_polygon(painter, self.enter_points, ENTER_COLOR, closed=True)
        self._draw_polygon(painter, self.exit_points, EXIT_COLOR, closed=True)

        if not self.done and self.hover_pos is not None and self.current_points:
            pen = QPen(self.current_zone_color, 1, Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(self._to_widget(self.current_points[-1]), self.hover_pos)

        self._draw_banner(painter)

    def _draw_banner(self, painter):
        # Always-visible status of which zone is being drawn — the #1 UX complaint fix:
        # without this the user has no idea Enter vs Exit is currently active.
        if self.done:
            label, color = "✓  Both zones set — click Continue", WHITE_COLOR
        else:
            n = len(self.current_points)
            label = f"✏  Drawing: {self.current_zone_name.upper()} ZONE   ({n} point{'s' if n != 1 else ''})"
            color = self.current_zone_color
        font = painter.font()
        font.setBold(True)
        font.setPointSize(13)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        pad_x, pad_y = 16, 10
        rect_w = metrics.horizontalAdvance(label) + pad_x * 2
        rect_h = metrics.height() + pad_y * 2
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 190))
        painter.drawRoundedRect(16, 16, rect_w, rect_h, 8, 8)
        painter.setPen(QPen(color))
        painter.drawText(16 + pad_x, 16 + pad_y + metrics.ascent(), label)

    @property
    def current_zone_color(self):
        return EXIT_COLOR if self.drawing_exit else ENTER_COLOR

    def _draw_polygon(self, painter, points, color, closed):
        if not points:
            return
        widget_pts = [self._to_widget(p) for p in points]
        pen = QPen(color, 2)
        painter.setPen(pen)
        if len(widget_pts) > 1:
            poly = QPolygonF(widget_pts)
            if closed and (points is self.enter_points and self.drawing_exit or points is self.exit_points and self.done):
                painter.setBrush(QColor(color.red(), color.green(), color.blue(), 40))
                painter.drawPolygon(poly)
                painter.setBrush(Qt.NoBrush)
            else:
                painter.drawPolyline(poly)
        for p in widget_pts:
            painter.setBrush(color)
            painter.drawEllipse(p, 4, 4)
            painter.setBrush(Qt.NoBrush)
