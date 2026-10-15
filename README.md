# Fall Detection App

## Overview

This project provides a real-time fall detection system using video analytics and MQTT messaging from [Intel® SceneScape](https://github.com/open-edge-platform/scenescape). The system processes camera feeds, detects people, computes features (such as aspect ratio, velocity, and bounding box area), and determines the state of each person (e.g., standing, walking, running, fallen). Results are published via MQTT and can be visualized in Node-RED dashboards.

<p align="center">
  <img src="images/dashboard.gif" alt="Fall Detection Dashboard Preview">
</p>

---

## How it Works

<p align="center">
  <img src="images/FallDetection.png" alt="Fall Detection Bounding Box Comparison">
</p>

The fall detection system leverages SceneScape’s multi-camera tracking and 3D scene understanding to robustly determine a person’s state, regardless of camera angle or position.

- **Bounding Box Comparison:**  
  For each detected person, the system computes the 2D bounding box from the camera’s perspective (the "detected bounding box") and also projects a 3D canonical bounding box for a standing person at that location into the camera view (the "projected bounding box"). By comparing the aspect ratio and area of these two boxes, the system can infer whether the person is upright or has fallen.

- **Detection and Motion Approach:**  
  This system relies solely on the detection bounding boxes provided by the person detector and the velocity of the person tracked across the scene. As long as people are detected in the scene, the fall detection logic will operate. It does **not** attempt to classify whether a person has fallen using image-based classification or deep learning on the image itself. Such classification techniques are often prone to errors due to varying camera angles, occlusions, and scene complexity, and do not account for how people have been moving. By using geometric reasoning based on bounding boxes, scene calibration, and velocities, this approach remains robust across different viewpoints and camera placements.

- **Camera-Angle Robustness:**  
  Because the projected bounding box is calculated using the camera’s intrinsic and extrinsic parameters, the comparison is robust to different camera angles, heights, and lens distortions. This means the system does not rely on a fixed camera placement or a specific viewpoint.

- **Multi-Camera Tracking:**  
  SceneScape provides consistent person IDs across all cameras in the scene. The fall detection logic aggregates features (such as aspect ratio, velocity, and bounding box area) for each person across all visible cameras. This enables reliable detection even if a fall is only visible from certain perspectives, or if a person moves between camera views.

- **State Determination:**  
  The system uses a combination of aspect ratio ratio (detected/projected), velocity, and bounding box area change to classify each person’s state as standing, walking, running, or fallen. The logic is designed to minimize false positives due to occlusions or partial views.

---

## Prerequisites

- Linux with Docker Engine, Docker Compose v2, Python 3, and Git LFS.
- A SceneScape `2026.3.x` checkout with its images and secrets initialized.
- SceneScape must be running with the **Controller** profile. This profile starts both the Scene Controller and Analytics service that publishes regulated scene data.

If the SceneScape web certificate uses a different hostname, export `SCENESCAPE_CERT_HOSTNAME` before running `setup.py` or `uninstall.py` (for example, `export SCENESCAPE_CERT_HOSTNAME=web.example.com`). The default is `web.scenescape.intel.com`; the configured name must match the certificate.

Fetch the model and video assets before setup. Git LFS pointer files are not playable media:

```sh
git lfs install
git lfs pull
```

- **API Key (Token) Required:**  
  1. Log in to the SceneScape web UI.
  2. Go to the **Admin** panel.
  3. Locate the API key for the `scenectrl` user and copy it.
  4. You will be prompted to provide this key during setup.

---

## Dataset

The included dataset features three people falling in various ways, captured from two different camera views. The scene was mapped using Polycam and an iPhone 16 Pro with lidar, and the scene map image was extracted from an orthographic view of the resulting 3D reconstruction. AprilTags are visible in the scene, but were **not** used for camera calibration.

Two variations of the synchronized video feeds are provided, both suitable for looping:
- **Variation 1:** Shows a single person falling, with two other people walking and standing.
- **Variation 2:** A longer 5-minute version featuring all three people walking, running, standing, and falling in various ways.

These videos are intended for testing and demonstration of the fall detection system’s robustness across multiple camera angles and activity types.

---

## Model

The person detection model used in this application is trained in Intel® Geti™ with a single class: **person**. The model is specifically trained on the provided dataset, which means it is optimized for detecting people in the included test scenes.

**Note:**  
If you plan to use this fall detection system in different environments or with new video data, you may need to retrain the model with additional data to ensure reliable person detection across a variety of scenes and conditions.

---

## Quick Start

### 1. **Prepare SceneScape**

Start the current SceneScape DL Streamer example with the Controller profile, following the SceneScape documentation. Create the scene requested by this application and obtain the `scenectrl` API token from the admin UI.

### 2. **Run Setup**

```sh
python3 setup.py
```

The setup script will:
- Prompt you to create a scene in SceneScape using your dataset image.
- Copy model and video files to the SceneScape project volumes `vol-models` and `vol-videos`.
- Upsert both cameras with their complete calibration.
- Build a dedicated fall-detection image derived from the current controller image.
- Configure and start Node-RED.
- Install required Node-RED modules.
- Import and configure Node-RED flows.
- Write its environment and Compose override under `.generated/` in this repository.

The setup does not modify files in the SceneScape checkout and does not copy SceneScape credentials into this repository.

---

### 3. **Start Services Manually**

Setup starts all services automatically. To start them manually later, set `SCENESCAPE_DIR` to the checkout used during setup and run:

```sh
docker compose \
  --project-directory "$SCENESCAPE_DIR" \
  --env-file .generated/fall-detection.env \
  -f "$SCENESCAPE_DIR/sample_data/compose/docker-compose-dl-streamer-example.yml" \
  -f .generated/docker-compose.override.yml \
  --profile controller up -d --build
```

---

### 4. **View the Dashboards**

- **Node-RED Dashboard:**  
  [http://localhost:1880/ui](http://localhost:1880/ui)

- **SceneScape UI:**  
  [https://localhost](https://localhost)

---

### 5. **Uninstall / Clean Up**

To stop and remove only this application's services and generated state, run:

```sh
python3 uninstall.py
```

The script optionally removes cameras recorded as created by this app and app-owned Node-RED data. Cameras that existed before setup are left intact. It restores the running Scene Controller to the base Compose configuration without deleting SceneScape files, environment settings, or shared Docker volumes. If an earlier setup did not record camera ownership, uninstall leaves those cameras untouched.

---

## Directory Structure

```
fall_detection_app/
├── dataset/
├── model/
├── Dockerfile.fall-detection
├── docker-compose.override.template.yml
├── detect_falls.py
├── flows.json
├── setup.py
├── uninstall.py
└── ...
```

---

## Notes

- Configuration is handled by `setup.py`; no manual editing of SceneScape files or Node-RED flows is required.
- The fall detector uses a standalone Alpine Python image and does not depend on the SceneScape controller image.
- Node-RED state persists in the app-owned `node_red_data/` directory.
- Detections are consumed from `scenescape/data/scene/<scene-id>/person` and fall results are published to `scenescape/fall-detection/<scene-id>`.
- Only detector-observed camera bounds (`projected: false`) are used as posture evidence.

---

## Troubleshooting

- Generated Compose environment values are in `.generated/fall-detection.env`. This file contains the API token and is created with mode `0600`.
- The actual shared volume names use the Compose project prefix, normally `scenescape_vol-models` and `scenescape_vol-videos`.
- Check service logs with the same base and override files shown above, followed by:
  ```sh
  --profile controller logs <service>
  ```

---

## License

See [LICENSE](LICENSE) for details.
