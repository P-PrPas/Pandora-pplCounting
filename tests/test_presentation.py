"""Run with python3 test_presentation.py; no inference or video required."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "v1"))

import numpy as np
from presentation import Presentation, BG, MINT, TRACK, chip, draw_track, draw_zones
from people_counter import ZONE_A, ZONE_B


def demo():
    frame = np.full((1080, 1920, 3), 90, dtype=np.uint8)
    original = frame.copy()
    draw_zones(frame, ZONE_A, ZONE_B)
    assert not np.array_equal(frame, original)
    for box in [(-10, 0, 60, 150), (1880, 1040, 1950, 1100)]:
        draw_track(frame, box, 999, (20, 100), [(0, 0), (20, 100)], True, 'in')
    contrast = np.full((300, 300, 3), 255, dtype=np.uint8)
    draw_track(contrast, (50, 60, 250, 250), 1, (150, 250), [], False, None)
    assert tuple(contrast[150, 50]) == TRACK
    assert tuple(contrast[150, 47]) == BG
    occupied = []
    for tid in range(3):
        chip(frame, str(tid), 100, 100, occupied=occupied)
    for i, (left, top, right, bottom) in enumerate(occupied):
        assert 0 <= top < bottom < frame.shape[0]
        for l, t, r, b in occupied[:i]:
            assert right <= l or left >= r or bottom <= t or top >= b
    ui = Presentation(90)
    empty = ui.render(frame, 0, 0, [], 0, 0)
    events = [(n, n*25, n+1, 'in' if n%2 else 'out') for n in range(7)]
    full = ui.render(frame, 999, 1000, events, 90, 12)
    assert full.shape == (1280, 2560, 3) and full.dtype == np.uint8
    assert np.array_equal(full[40:1120, 40:1960], frame)
    assert not np.array_equal(empty[704:1072, 2000:], full[704:1072, 2000:])
    assert tuple(full[1208, 2520]) == MINT
    assert tuple(ui.base[1208, 100]) != MINT  # rendering does not mutate the template
    assert tuple(full[0, 0]) == BG
    print('presentation self-check passed')


if __name__ == '__main__':
    demo()
