"""Tiny per-RTSP-source cache for v2/v3 calibration samples (raw
foot->neck or foot->head observations, not a baked zone polygon - see
head_calibration.py for why: the offset is a pure function of pixel
position, so any zone shape drawn later can derive its head-level polygon
from these same samples without re-running detection).

Keyed by (rtsp_url, logic) - v2's neck offset and v3's head offset aren't
interchangeable even for the same camera, so switching presets shouldn't
clobber or misuse the other's cached samples.

ponytail: single-most-recent-source cache, same pattern as zone_canvas.py's
zones.json. Add a keyed-by-source dict if multiple cameras need remembering.
"""
import json
from pathlib import Path

CALIBRATION_FILE = Path(__file__).parent / "calibration_samples.json"


def load(rtsp_url, logic):
    """Returns (positions, offsets), both [] if nothing cached for this source+logic."""
    if not CALIBRATION_FILE.exists():
        return [], []
    try:
        data = json.loads(CALIBRATION_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return [], []
    if data.get("rtsp_url") != rtsp_url:
        return [], []
    bucket = data.get(logic, {})
    return bucket.get("positions", []), bucket.get("offsets", [])


def save(rtsp_url, logic, positions, offsets):
    # positions/offsets come straight out of numpy box math (float32) - plain
    # float() so json.dumps doesn't choke on non-native numeric types.
    if not CALIBRATION_FILE.exists() or json.loads(CALIBRATION_FILE.read_text() or "{}").get("rtsp_url") != rtsp_url:
        data = {"rtsp_url": rtsp_url}
    else:
        data = json.loads(CALIBRATION_FILE.read_text())
    data[logic] = {
        "positions": [[float(x), float(y)] for x, y in positions],
        "offsets": [[float(x), float(y)] for x, y in offsets],
    }
    CALIBRATION_FILE.write_text(json.dumps(data))
