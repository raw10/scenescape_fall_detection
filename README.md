# Fall Detection App

## Overview

This project provides a real-time fall detection system using video analytics and MQTT messaging from Intel® SceneScape. The system processes camera feeds, detects people, computes features (such as aspect ratio, velocity, and bounding box area), and determines the state of each person (e.g., standing, walking, running, fallen). Results are published via MQTT and can be visualized in Node-RED dashboards.

---

## How it Works

<p align="center">
  <img src="refs/heads/main/images/FallDetection.png" alt="Fall Detection Bounding Box Comparison" width="600"/>
</p>

The fall detection system leverages SceneScape’s multi-camera tracking and 3D scene understanding to robustly determine a person’s state, regardless of camera angle or position.

- **Bounding Box Comparison:**  
  For each detected person, the system computes the 2D bounding box from the camera’s perspective (the "detected bounding box") and also projects a 3D canonical bounding box for a standing person at that location into the camera view (the "projected bounding box"). By comparing the aspect ratio and area of these two boxes, the system can infer whether the person is upright or has fallen.

- **Detection-Only Approach:**  
  This system relies solely on the detection bounding boxes provided by the person detector. As long as people are detected in the scene, the fall detection logic will operate. It does **not** attempt to classify whether a person has fallen using image-based classification or deep learning on the image itself. Such classification techniques are often prone to errors due to varying camera angles, occlusions, and scene complexity. By using geometric reasoning based on bounding boxes and scene calibration, this approach remains robust across different viewpoints and camera placements.

- **Camera-Angle Robustness:**  
  Because the projected bounding box is calculated using the camera’s intrinsic and extrinsic parameters, the comparison is robust to different camera angles, heights, and lens distortions. This means the system does not rely on a fixed camera placement or a specific viewpoint.

- **Multi-Camera Tracking:**  
  SceneScape provides consistent person IDs across all cameras in the scene. The fall detection logic aggregates features (such as aspect ratio, velocity, and bounding box area) for each person across all visible cameras. This enables reliable detection even if a fall is only visible from certain perspectives, or if a person moves between camera views.

- **State Determination:**  
  The system uses a combination of aspect ratio ratio (detected/projected), velocity, and bounding box area change to classify each person’s state as standing, walking, running, or fallen. The logic is designed to minimize false positives due to occlusions or partial views.

---

## Prerequisites

- **SceneScape must be running.**  
  Follow the official SceneScape instructions to launch the out-of-box demo scenes before proceeding with this app.

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

### 1. **Extract the Files**

Download and extract the provided `.zip` archive containing the fall detection app files, model, and dataset:

```sh
unzip fall_detection_app.zip
cd fall_detection_app
```

---

### 2. **Run the Setup Script**

The setup script will:
- Prompt you to create a scene in SceneScape using your dataset image.
- Copy model and video files to the correct locations.
- Configure and start Node-RED.
- Install required Node-RED modules.
- Import and configure Node-RED flows.
- Set up all necessary Docker Compose overrides.

Run:

```sh
./setup.py
```

Follow the prompts to complete the setup, including entering your SceneScape API key when requested.

---

### 3. **Start All Services**

After setup completes, all services will be started automatically.  
If you need to start them manually later, run:

```sh
docker compose up -d
```

---

### 4. **View the Dashboards**

- **Node-RED Dashboard:**  
  [http://localhost:1880/ui](http://localhost:1880/ui)

- **SceneScape UI:**  
  [https://localhost](https://localhost)

---

### 5. **Uninstall / Clean Up**

To remove all files, cameras, and configuration created by the setup, run:

```sh
./uninstall.py
```

Follow the prompts to select which data to remove.

---

## Directory Structure

```
fall_detection_app/
├── dataset/
├── model/
├── docker-compose.override.template.yml
├── detect_falls.py
├── flows.json
├── setup.py
├── uninstall.py
└── ...
```

---

## Notes

- All configuration is now handled by `setup.py`—no manual editing of Docker Compose files or Node-RED flows is required.
- The Node-RED data directory is now a bind mount (`./node_red_data`), so all flows and dashboard settings are persistent.
- The uninstall script will prompt before removing any persistent data.

---

## Troubleshooting

- If you encounter issues with secrets or permissions, ensure the `SECRETSDIR` environment variable is set to `secrets` before running Docker Compose commands.
- For advanced debugging, check the logs of each service with:
  ```sh
  docker compose logs <service>
  ```

---

## License

See [LICENSE](LICENSE) for details.
