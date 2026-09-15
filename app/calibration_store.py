"""Tiny per-RTSP-source cache for v2 calibration samples (raw foot->neck
observations, not a baked zone polygon - see head_calibration.py for why:
the offset is a pure function of pixel position, so any zone shape drawn
later can derive its head-level polygon from these same samples without
re-running pose detection).

ponytail: single-most-recent-source cache, same pattern as zone_canvas.py's
zones.json. Add a keyed-by-source dict if multiple cameras need remembering.
"""
import json
from pathlib import Path

CALIBRATION_FILE = Path(__file__).parent / "calibration_samples.json"


def load(rtsp_url):
    """Returns (positions, offsets), both [] if nothing cached for this source."""
    if not CALIBRATION_FILE.exists():
        return [], []
    try:
        data = json.loads(CALIBRATION_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return [], []
    if data.get("rtsp_url") != rtsp_url:
        return [], []
    return data.get("positions", []), data.get("offsets", [])


def save(rtsp_url, positions, offsets):
    # positions/offsets come straight out of numpy box math (float32) - plain
    # float() so json.dumps doesn't choke on non-native numeric types.
    CALIBRATION_FILE.write_text(json.dumps({
        "rtsp_url": rtsp_url,
        "positions": [[float(x), float(y)] for x, y in positions],
        "offsets": [[float(x), float(y)] for x, y in offsets],
    }))
