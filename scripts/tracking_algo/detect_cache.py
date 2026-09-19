"""Per-(version, clip) detection cache. The detector CNN is the expensive
part of this whole comparison and is identical across trackers for a given
(version, clip) - running it once and caching beats 5x redundant inference
(5 trackers x 3 versions x 2 clips would otherwise mean 30 full detection
passes instead of 6).
"""
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
CACHE_DIR = HERE / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(HERE.parent / "v1"))


def _boxes_array(r):
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 6))
    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy()
    return np.concatenate([xyxy, conf[:, None], cls[:, None]], axis=1)


def build(version, clip_path, device):
    """Returns a list of per-frame dicts (one per video frame, in order):
    {"boxes": Nx6 [x1,y1,x2,y2,conf,cls], "keypoints": Nx17x3|None (v2 only),
    "person_boxes": Mx4|None (v3 only)}. Cached to disk after the first build.
    """
    from ultralytics import YOLO
    clip_path = Path(clip_path)
    cache_file = CACHE_DIR / f"{version}_{clip_path.stem}.pkl"
    if cache_file.exists():
        return pickle.loads(cache_file.read_bytes())

    if version == "v1":
        model, person_model = YOLO(HERE.parent / "v1" / "yolo11s.pt"), None
    elif version == "v2":
        model, person_model = YOLO(HERE.parent / "v2" / "yolo11s-pose.pt"), None
    else:
        model = YOLO(HERE.parent / "v3" / "head_detector.pt")
        person_model = YOLO(HERE.parent / "v1" / "yolo11s.pt")

    cap = cv2.VideoCapture(str(clip_path))
    frames = []
    print(f"detect_cache: building {version}/{clip_path.stem}...")
    kwargs = dict(conf=0.1, verbose=False, device=device)
    if version != "v3":
        kwargs["classes"] = [0]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, **kwargs)[0]
        entry = {"boxes": _boxes_array(r), "keypoints": None, "person_boxes": None}
        if version == "v2":
            kp = r.keypoints
            entry["keypoints"] = kp.data.cpu().numpy() if kp is not None and len(kp) else np.zeros((0, 17, 3))
        if version == "v3":
            pr = person_model.predict(frame, classes=[0], conf=0.1, verbose=False, device=device)[0]
            entry["person_boxes"] = pr.boxes.xyxy.cpu().numpy() if pr.boxes is not None else np.zeros((0, 4))
        frames.append(entry)
        if len(frames) % 200 == 0:
            print(f"  ...{len(frames)} frames")
    cap.release()
    cache_file.write_bytes(pickle.dumps(frames))
    print(f"detect_cache: cached {len(frames)} frames -> {cache_file}")
    return frames
