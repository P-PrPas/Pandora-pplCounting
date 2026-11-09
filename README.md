<div align="center">

![Pandora: computer vision for doorway people counting](assets/readme/banner.svg)

# Doorway People Counter

**A proof of concept for counting people entering and leaving through a doorway.**

`LIVE RTSP` &nbsp; · &nbsp; `OFFLINE VIDEO` &nbsp; · &nbsp; `THREE TRACKING APPROACHES`

[Get started](#getting-started) · [Live app](#live-desktop-app) · [Offline processing](#offline-video-processing) · [Documentation](#documentation)

</div>

---

## At a glance

Pandora detects and tracks people crossing two configured doorway zones. A transition from the outside zone (A) to the inside zone (B) counts as **IN**; the reverse counts as **OUT**. The desktop app displays live counts, while the offline pipelines produce annotated video and a CSV event log.

**Choose a tracking approach**

| Logic | Tracking point | Models | Calibration |
| --- | --- | --- | --- |
| V1 | Person bounding box, bottom center (foot) | YOLO11 person detector | Draw foot-level zones |
| V2 | Midpoint of shoulder keypoints (neck) | YOLO11 pose | Foot-to-neck offset |
| V3 | Head detection | Head detector and YOLO11 person detector | Foot-to-head offset |

V3 uses person detections to filter head detections. All three use tracking and a shared zone-crossing counter.

> [!IMPORTANT]
> This is a camera-specific proof of concept. Validate accuracy on your own footage and conditions before operational use.

## Getting started

### Requirements

- Python 3, a graphical desktop for the PySide6 app, and access to an RTSP camera for live use.
- The model checkpoints at the paths below. Checkpoint availability and licensing should be verified before deployment.
- A virtual environment is recommended. GPU acceleration is optional; inference can run on CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r app/requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. Run the following commands from the repository root.

### Model files

| File | Used by |
| --- | --- |
| `scripts/v1/yolo11s.pt` | V1 and V3 person detection |
| `scripts/v2/yolo11s-pose.pt` | V2 pose detection and calibration |
| `scripts/v3/head_detector.pt` | V3 head detection and calibration |

The V3 checkpoint is a third-party YOLOv8n model trained on SCUT-HEAD. Obtain the appropriate checkpoints and place them at these paths if they are absent. The application does not train models.

## Live desktop app

**Configure → draw zones → calibrate → count**

1. Copy the example configuration and set your camera URL:

   ```bash
   cp .env.example .env
   # Edit .env and set RTSP_URL=rtsp://user:password@host:554/path
   python app/main.py
   ```

2. On the camera snapshot, draw the Enter zone and select **Finish zone**; repeat for the Exit zone. Select **Continue**.
3. Choose a **Logic** preset. For V2 or V3, select **Calibrate** and let people walk through the frame while samples are collected (approximately 20–30 seconds).
4. Select **Start Live Count**. Recalibrate after moving the camera or changing its view.

The app stores zones in `app/zones.json` and calibration samples in `app/calibration_samples.json`. Calibration is associated with the camera source and logic; V2 and V3 samples are separate. These files and `.env` are ignored by Git. Treat them as sensitive: camera URLs can contain credentials. A changed source or camera geometry requires checking the saved setup.

## Offline video processing

> [!NOTE]
> The offline scripts use the fixed-camera `ZONE_A` and `ZONE_B` coordinates in `scripts/v1/people_counter.py`, **not** the zones drawn in the desktop app. Adjust those polygons for the camera before processing, and recalibrate V2/V3 after changing the camera view or zones. > Use representative reference clips with people walking through the doorway.

```bash
mkdir -p data/results/v1 data/results/v2 data/results/v3

# V1: foot tracking
python scripts/v1/people_counter.py data/dataset/clip.mp4 data/results/v1/output

# V2: calibrate, then count
python scripts/v2/calibrate_zones.py scripts/v2/zones_head.json data/dataset/clip.mp4
python scripts/v2/people_counter_v2.py scripts/v2/zones_head.json data/dataset/clip.mp4 data/results/v2/output

# V3: calibrate, then count
python scripts/v3/calibrate_zones_v3.py scripts/v3/zones_head_v3.json data/dataset/clip.mp4
python scripts/v3/people_counter_v3.py scripts/v3/zones_head_v3.json data/dataset/clip.mp4 data/results/v3/output
```

Calibration accepts additional reference clips after the first. Each counting run writes `<output_prefix>.mp4` and `<output_prefix>.csv`; the CSV columns are `time_s`, `frame`, `track_id`, and `direction`. Input footage and generated results under `data/` are local and ignored by Git.

## Verification

Run the lightweight checks from the repository root:

```bash
python tests/test_zone_counter.py
python tests/test_presentation.py
python tests/test_v2_head_calibration.py
```

These checks do not replace validation with the intended camera, lighting, traffic, and hardware. Inference selects an available device automatically; set `DEVICE` to override it if needed.

## Project layout

```text
app/                    PySide6 live application and calibration storage
scripts/v1/             Foot tracking, counting, and presentation
scripts/v2/             Pose-based neck tracking and calibration
scripts/v3/             Head detection and calibration
scripts/tracking_algo/  Offline tracker comparison and reporting tools
tests/                  Counting, presentation, and calibration checks
docs/                   Evaluation, presentation notes, and reports
assets/fonts/           Fonts used by the video renderer
```

The tracker comparison tools are research tooling, separate from the live app. Their additional dependencies are listed in `scripts/tracking_algo/requirements.txt`.

## Troubleshooting

- **Missing configuration or unreachable camera:** Check `RTSP_URL`, camera connectivity, and credentials; restart the app after updating `.env`.
- **Missing model:** Check the checkpoint paths above and obtain the required files.
- **Poor counts:** Check the zone placement, camera framing, occlusion, lighting, and calibration samples. Recalibrate after camera changes and compare against manually counted footage.
- **Cannot create output video:** Create the output directory and check write permissions and codec support.

## Go deeper

| Document | What you will find |
| --- | --- |
| [Evaluation report](docs/REPORT.md) | Results, edge cases, and proposed improvements |
| [Presentation notes](docs/PRESENTATION.md) | Video rendering and export details |
| [Tracker comparison](docs/tracking_algo_ground_truth.md) | Ground truth for the research tools |

---

<div align="center"><sub>Built for understanding movement through a doorway.</sub></div>
