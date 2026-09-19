"""Duck-types just the ultralytics Result attributes render_v{1,2,3}_frame
read (r.orig_img, r.boxes.xyxy/.id, r.keypoints.data) so a frame tracked by
an external tracker (boxmot, McByte++) can be fed straight into those
existing render functions, unchanged.
"""
import torch


class _Boxes:
    def __init__(self, xyxy, ids):
        self.xyxy = torch.as_tensor(xyxy, dtype=torch.float32)
        self.id = torch.as_tensor(ids) if ids is not None and len(ids) else None


class _Keypoints:
    def __init__(self, data):
        self.data = torch.as_tensor(data, dtype=torch.float32)


class FakeResult:
    def __init__(self, frame, xyxy, ids, keypoints=None):
        self.orig_img = frame
        self.boxes = _Boxes(xyxy, ids)
        self.keypoints = _Keypoints(keypoints) if keypoints is not None else None
