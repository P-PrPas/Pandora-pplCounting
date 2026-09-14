"""Presentation overlay. BGR palette, bundled type, original camera coordinates."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = (22, 17, 12)
PANEL = (34, 28, 21)
LINE = (62, 53, 42)
WHITE = (242, 241, 233)
MUTED = (167, 153, 134)
MINT = (184, 225, 104)
AMBER = (112, 193, 247)
TRACK = (255, 210, 40)  # bright cyan, distinct from mint IN and amber OUT
FONT_DIR = Path(__file__).parent.parent / 'assets' / 'fonts'


@lru_cache(maxsize=32)
def font(size, bold=False):
    return ImageFont.truetype(str(FONT_DIR / ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')), size)


@lru_cache(maxsize=512)
def text_sprite(value, size, color, bold):
    face = font(size, bold)
    bounds = face.getbbox(value)
    im = Image.new('RGBA', (max(1, bounds[2] + 2), size + 8))
    ImageDraw.Draw(im).text((0, -bounds[1]), value, font=face, fill=(*color[::-1], 255))
    rgba = np.array(im)
    return rgba[:, :, 2::-1], rgba[:, :, 3:4].astype(np.float32) / 255


def text(frame, value, x, y, size=22, color=WHITE, bold=False):
    pixels, alpha = text_sprite(str(value), size, color, bold)
    h, w = pixels.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    if x0 < x1 and y0 < y1:
        p = pixels[y0-y:y1-y, x0-x:x1-x]
        a = alpha[y0-y:y1-y, x0-x:x1-x]
        roi = frame[y0:y1, x0:x1]
        roi[:] = p * a + roi * (1 - a)


def panel(frame, x, y, w, h, fill=PANEL, radius=16):
    cv2.rectangle(frame, (x+radius, y), (x+w-radius, y+h), fill, -1)
    cv2.rectangle(frame, (x, y+radius), (x+w, y+h-radius), fill, -1)
    for cx, cy in [(x+radius, y+radius), (x+w-radius, y+radius),
                   (x+radius, y+h-radius), (x+w-radius, y+h-radius)]:
        cv2.circle(frame, (cx, cy), radius, fill, -1, cv2.LINE_AA)


def chip(frame, label, x, y, color=TRACK, occupied=None):
    width = text_sprite(label, 18, color, True)[0].shape[1] + 24
    x = max(0, min(x, frame.shape[1] - width - 1))
    y = max(0, min(y, frame.shape[0] - 34))
    if occupied is not None:
        original_y = y
        # ponytail: linear scan per badge; use spatial indexing for hundreds of tracks.
        candidates = list(range(y, -1, -36)) + list(range(y+36, frame.shape[0]-33, 36))
        for candidate in candidates:
            if all(x+width+4 <= left or x >= right+4 or candidate+36 <= top or candidate >= bottom+4
                   for left, top, right, bottom in occupied):
                y = candidate
                break
        occupied.append((x, y, x+width, y+32))
        if y != original_y:
            cv2.line(frame, (x+width//2, original_y+32), (x+width//2, y+32), color, 1, cv2.LINE_AA)
    panel(frame, x, y, width, 32, BG, 6)
    text(frame, label, x+12, y+6, 18, color, True)


def draw_zones(frame, outside, inside, occupied=None, label_a='A / OUTSIDE', label_b='B / INSIDE'):
    overlay = frame.copy()
    for polygon, color in [(outside, AMBER), (inside, MINT)]:
        cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, .08, frame, .92, 0, dst=frame)
    for polygon, color in [(outside, AMBER), (inside, MINT)]:
        cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)
    # Labels sit beside the calibrated polygons, leaving the crossing visible.
    for label, polygon, color in [(label_a, outside, AMBER), (label_b, inside, MINT)]:
        x, y = polygon[np.argmax(polygon[:, 0])]
        cv2.line(frame, (int(x), int(y)), (int(x)+24, int(y)), color, 1, cv2.LINE_AA)
        chip(frame, label, int(x)+24, int(y)-16, color, occupied)


def draw_track(frame, box, tid, foot, trail, highlighted, event_label, occupied=None):
    x1, y1, x2, y2 = (int(v) for v in box)
    color = (MINT if event_label == 'in' else AMBER) if highlighted else TRACK
    # Dark casing preserves contrast on both bright floors and dark clothing.
    cv2.rectangle(frame, (x1, y1), (x2, y2), BG, 5, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    length = max(3, min(18, (x2-x1)//3, (y2-y1)//3))
    for x, y, dx, dy in [(x1,y1,1,1), (x2,y1,-1,1), (x1,y2,1,-1), (x2,y2,-1,-1)]:
        cv2.line(frame, (x, y), (x+dx*length, y), color, 4 if highlighted else 3, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x, y+dy*length), color, 4 if highlighted else 3, cv2.LINE_AA)
    label = f'{tid:03d}' + (f' / {event_label.upper()}' if highlighted and event_label else '')
    chip(frame, label, x1, y1-36, color, occupied)
    pts = list(trail)
    for i in range(1, len(pts)):
        shade = tuple(int(c * (.25 + .65*i/max(1, len(pts)-1))) for c in color)
        cv2.line(frame, pts[i-1], pts[i], shade, 2, cv2.LINE_AA)
    cv2.circle(frame, foot, 5, BG, -1, cv2.LINE_AA)
    cv2.circle(frame, foot, 3, color, -1, cv2.LINE_AA)
    if highlighted:
        cv2.circle(frame, foot, 9, color, 2, cv2.LINE_AA)


def clock(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds//60:02d}:{seconds%60:02d}'


class Presentation:
    """2560 x 1280 canvas; the full 16:9 feed sits beside clip-level analytics."""
    size = (2560, 1280)

    def __init__(self, duration):
        self.duration = duration
        self.base = np.full((1280, 2560, 3), BG, dtype=np.uint8)
        f = self.base
        panel(f, 2000, 40, 520, 354)
        text(f, 'DIRECTIONAL TOTALS', 2028, 67, 18, MUTED, True)
        text(f, 'Cumulative crossings in this clip', 2028, 100, 18, MUTED)
        cv2.line(f, (2260, 152), (2260, 288), LINE, 1)
        text(f, 'ENTERED', 2028, 157, 19, MINT, True)
        text(f, 'A  →  B', 2028, 296, 19, MINT)
        text(f, 'EXITED', 2290, 157, 19, AMBER, True)
        text(f, 'B  →  A', 2290, 296, 19, AMBER)
        text(f, 'Crossing events, not unique visitors', 2028, 354, 17, MUTED)
        panel(f, 2000, 410, 520, 174)
        text(f, 'NET FLOW', 2028, 436, 18, MUTED, True)
        text(f, 'Entered minus exited · not occupancy', 2028, 546, 17, MUTED)
        panel(f, 2000, 600, 520, 520)
        text(f, 'RECENT CROSSINGS', 2028, 628, 18, MUTED, True)
        text(f, 'TIME', 2028, 676, 14, MUTED)
        text(f, 'TRACK', 2150, 676, 14, MUTED)
        text(f, 'DIRECTION', 2310, 676, 14, MUTED)
        cv2.line(f, (2028, 704), (2492, 704), LINE, 1)
        text(f, 'HOW TO READ', 40, 1162, 16, MUTED, True)
        for x, color, title in [(235, TRACK, 'Tracked person'), (540, AMBER, 'A / Outside'), (820, MINT, 'B / Inside')]:
            cv2.rectangle(f, (x, 1161), (x+19, 1180), color, 2, cv2.LINE_AA)
            text(f, title, x+32, 1160, 20, WHITE)
        text(f, 'Highlighted track = new crossing', 1090, 1160, 20, MUTED)
        text(f, 'PROOF OF CONCEPT', 40, 1231, 14, MUTED, True)
        text(f, 'Recorded footage · AI-assisted counting', 2000, 1232, 16, MUTED)

    def render(self, frame, in_count, out_count, events, elapsed, active):
        canvas = self.base.copy()
        canvas[40:1120, 40:1960] = cv2.resize(frame, (1920,1080)) if frame.shape[:2] != (1080,1920) else frame
        text(canvas, f'{in_count:02d}', 2023, 198, 76, MINT, True)
        text(canvas, f'{out_count:02d}', 2285, 198, 76, AMBER, True)
        text(canvas, f'{in_count-out_count:+d}', 2025, 471, 49, WHITE, True)
        if not events:
            text(canvas, 'Waiting for a crossing', 2028, 746, 24, WHITE)
            text(canvas, 'Events appear after a zone transition.', 2028, 791, 17, MUTED)
        for i, (seconds, _, tid, direction) in enumerate(reversed(events[-5:])):
            y = 729+i*69
            color = MINT if direction == 'in' else AMBER
            if i == 0:
                panel(canvas, 2016, y-8, 488, 55, (45, 39, 29), 8)
            text(canvas, clock(seconds), 2028, y+5, 22, WHITE)
            text(canvas, f'{tid:03d}', 2150, y+5, 22, MUTED)
            text(canvas, '↗  IN' if direction == 'in' else '↙  OUT', 2310, y+5, 22, color, True)
        text(canvas, f'{active:02d} active tracks in frame', 2028, 1084, 18, MUTED)
        text(canvas, f'{clock(elapsed)}  /  {clock(self.duration)}', 1700, 1160, 22, WHITE)
        cv2.line(canvas, (40, 1208), (2520, 1208), LINE, 3)
        progress = min(1, max(0, elapsed/self.duration)) if self.duration > 0 else 0
        if progress:
            cv2.line(canvas, (40, 1208), (40+int(2480*progress), 1208), MINT, 3)
        return canvas
