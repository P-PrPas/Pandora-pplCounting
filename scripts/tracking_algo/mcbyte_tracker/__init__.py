"""Vendored from McBytePlusPlus (github.com/tstanczyk95/McBytePlusPlus,
Apache-2.0), tracker-core files only: STrack/McBytePlusPlusTracker
(tracker.py, "with-reid" variant), plus its basetrack/kalman_filter/matching/
gmc dependencies. Not vendored: EdgeTAM mask-propagation, the bundled YOLOX
detector, deep-person-reid's training code - see adapter.py for what this
project actually wires up (with-reid, no mask).

Only fix applied to the upstream files: `from yolox.tracker import ...`
absolute imports rewritten to relative (`from . import ...`) since this
package isn't installed inside a `yolox` tree here.
"""
