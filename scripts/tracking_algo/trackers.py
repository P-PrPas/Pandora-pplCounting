"""Uniform tracker factory: build_tracker(name, device) -> obj exposing
.update(dets, frame) -> np.ndarray of [x1,y1,x2,y2,id,conf,cls,det_idx] rows,
`det_idx` being the input `dets` row a track came from (needed to re-attach
v2's pre-tracking pose keypoints to the surviving post-tracking boxes).

boxmot already gives bytetrack/botsort/ocsort/deepocsort that exact shape
behind one API - no per-tracker adapter needed. McByte++ is vendored
separately (not a pip package); mcbyte_tracker.adapter reformats its output
into the same shape so run_comparison.py never has to special-case it.
"""
from boxmot import BotSort, ByteTrack, DeepOcSort, OcSort

# ponytail: one general-pedestrian ReID checkpoint (Market1501-trained
# osnet_x1_0) shared by every ReID-using tracker below, including McByte++'s
# adapter - not McByte++'s own documented checkpoint (GTA-link's sports-tuned
# sports_model.pth.tar-60), since our footage is a doorway, not a sports
# court. boxmot auto-downloads this one from its own catalog on first use.
REID_WEIGHTS = "osnet_x1_0_market1501.pt"

# Camera is fixed (see README) - camera-motion-compensation buys nothing here
# and costs real per-frame time, so it's off for BotSort in this harness.
BUILDERS = {
    "bytetrack": lambda device: ByteTrack(),
    "botsort": lambda device: BotSort(reid_weights=REID_WEIGHTS, device=device, use_cmc=False),
    "ocsort": lambda device: OcSort(),
    "deepocsort": lambda device: DeepOcSort(reid_weights=REID_WEIGHTS, device=device),
}

TRACKER_NAMES = [*BUILDERS, "mcbyteplusplus"]


def build_tracker(name, device="cpu"):
    if name == "mcbyteplusplus":
        from mcbyte_tracker.adapter import McByteAdapter
        return McByteAdapter(device=device, reid_weights=REID_WEIGHTS)
    return BUILDERS[name](device)
