import argparse
import os
import json
import requests
import sys
import paho.mqtt.client as mqtt
import ssl
import numpy as np
from scene_common import transform
from scipy.spatial.transform import Rotation as R
import time
from collections import defaultdict, deque

def parse_args():
    parser = argparse.ArgumentParser(description="Fall Detection App")
    parser.add_argument('--controller-auth', type=str, default="/app/controller.auth",
                        help='Path to controller.auth JSON file')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 1883)),
                        help='Port for both MQTT and API (default: 1883)')
    parser.add_argument('--scene-uuid', type=str, required=True,
                        help='Scene UUID to subscribe/query')
    parser.add_argument('--insecure', action='store_true', default=True,
                        help='Run in insecure mode (ignore SSL certs)')
    parser.add_argument('--broker', type=str, required=True,
                        help='MQTT broker hostname or alias')
    parser.add_argument('--resturl', type=str,
                        required=True, help='Base REST API URL')
    parser.add_argument('--root-cert', type=str, default="/run/secrets/certs/scenescape-ca.pem",
                        help='Path to root CA certificate for MQTT TLS (Docker secret)')
    parser.add_argument('--window-seconds', type=float,
                        default=0.5, help='Rolling window size in seconds')
    parser.add_argument('--walk-velocity-threshold', type=float,
                        default=0.2, help='Velocity threshold for walking')
    parser.add_argument('--run-velocity-threshold', type=float,
                        default=1.3, help='Velocity threshold for running')
    parser.add_argument('--fallen-arr-threshold', type=float,
                        default=0.6, help='ARR threshold for fallen')
    parser.add_argument('--area-rate-threshold', type=float, default=5000.0,
                        help='Area rate threshold for fallen state logic')
    return parser.parse_args()

def get_cameras(api_url, api_key, insecure, retries=5, delay=5):
    headers = {"Authorization": f"Token {api_key}"}
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                api_url,
                headers=headers,
                verify=not insecure,
                timeout=10
            )
            response.raise_for_status()
            cameras = response.json()
            if isinstance(cameras, dict) and "results" in cameras:
                camera_count = len(cameras["results"])
            elif isinstance(cameras, list):
                camera_count = len(cameras)
            else:
                camera_count = 0
            print(f"Retrieved {camera_count} cameras from API.")
            return cameras
        except requests.RequestException as e:
            print(
                f"Error retrieving cameras from API (attempt {attempt}/{retries}): {e}", file=sys.stderr)
            if attempt < retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Giving up.", file=sys.stderr)
    return None

def project_point(pt3d, intrinsics, distortion):
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    x, y, z = pt3d
    if z == 0:
        z = 1e-6
    u = fx * x / z + cx
    v = fy * y / z + cy
    return [u, v]

def get_canonical_bbox(obj, intrinsics, distortion, cam_extrinsics):
    cx, cy, cz = obj["translation"]
    w, d, h = obj["size"]
    r = (w + d) / 8
    h = h * 0.85
    base_z = cz
    top_z = cz + h
    offsets = [(-r, -r), (-r, r), (r, -r), (r, r)]
    corners_3d_world = []
    for ox, oy in offsets:
        corners_3d_world.append([cx + ox, cy + oy, base_z])
        corners_3d_world.append([cx + ox, cy + oy, top_z])
    corners_3d_cam = [world_to_camera(pt, cam_extrinsics)
                      for pt in corners_3d_world]
    filtered_corners_2d = []
    for pt in corners_3d_cam:
        if pt[2] <= 1e-3:
            continue
        filtered_corners_2d.append(project_point(pt, intrinsics, distortion))
    if not filtered_corners_2d:
        print("No valid projected 2D corners for canonical bbox (all points behind camera or invalid).")
        return None
    xs = [pt[0] for pt in filtered_corners_2d]
    ys = [pt[1] for pt in filtered_corners_2d]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bbox = {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}
    return bbox

def on_connect(client, userdata, flags, reason_code, properties):
    print(f"on_connect called with reason_code={reason_code}")
    if reason_code == 0:
        print("Connected to MQTT broker.")
        topic = userdata['mqtt_topic']
        print(f"Subscribing to topic: {topic}")
        client.subscribe(topic)
    else:
        print(f"Failed to connect to MQTT broker, reason code {reason_code}")

# {uuid: {cam_id: deque of (timestamp, feature_vector)}}
feature_history = defaultdict(lambda: defaultdict(deque))
tracked_people = {}
# {uuid: {cam_id: deque of (timestamp, area)}}
bb_area_history = defaultdict(lambda: defaultdict(deque))

def compute_smoothed_area_and_rate(area_hist):
    if area_hist:
        times = np.array([t for t, _ in area_hist])
        areas = np.array([a for _, a in area_hist])
        if len(times) > 1:
            weights = np.linspace(1, 2, len(times))
        else:
            weights = np.array([1.0])
        weights /= weights.sum()
        smoothed_area = float(np.average(areas, weights=weights))
        if len(times) > 1:
            slope, _ = np.polyfit(times, areas, 1)
            area_rate = float(slope)
        else:
            area_rate = 0.0
    else:
        smoothed_area = 0.0
        area_rate = 0.0
    return smoothed_area, area_rate

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode('utf-8')
        data = json.loads(payload)
        timestamp = data.get("timestamp")

        camera_calibrations = userdata.get("camera_calibrations", {})
        args = userdata.get("args")
        window_seconds = args.window_seconds if args else 2.0

        # {uuid: [(feature_vector, cam_id)]}
        person_features = defaultdict(list)
        canonical_bboxes = {}
        metrics_by_uuid = defaultdict(dict)

        for obj in data.get("objects", []):
            uuid = obj.get("id")
            if not uuid or obj.get("category") != "person":
                continue

            velocity = obj.get("velocity", [0, 0, 0])
            v_mag = float(np.linalg.norm(velocity))

            if "bounding_box_px" in obj and "bounding_box_camera_id" in obj:
                cam_id = obj["bounding_box_camera_id"]
                detected_bbox = obj["bounding_box_px"]
                detected_bbox_xyxy = {
                    "x_min": detected_bbox["x"],
                    "y_min": detected_bbox["y"],
                    "x_max": detected_bbox["x"] + detected_bbox["width"],
                    "y_max": detected_bbox["y"] + detected_bbox["height"],
                }
                cam_extrinsics = camera_calibrations.get(
                    cam_id, {}).get("extrinsics")
                intrinsics = camera_calibrations.get(
                    cam_id, {}).get("intrinsics")
                distortion = camera_calibrations.get(
                    cam_id, {}).get("distortion")

                canonical_bbox = None
                if intrinsics and distortion and cam_extrinsics:
                    canonical_bbox = get_canonical_bbox(
                        obj, intrinsics, distortion, cam_extrinsics)
                    if canonical_bbox is not None:
                        canonical_bboxes[cam_id] = canonical_bbox

                # Aspect ratio ratio
                def bbox_ar(b):
                    w = b["x_max"] - b["x_min"]
                    h = b["y_max"] - b["y_min"]
                    return h / w if w > 0 else 0

                aspect_ratio_detected = bbox_ar(detected_bbox_xyxy)
                aspect_ratio_canonical = bbox_ar(
                    canonical_bbox) if canonical_bbox else 1
                aspect_ratio_ratio = aspect_ratio_detected / \
                    aspect_ratio_canonical if aspect_ratio_canonical > 0 else 0

                # Calculate area and update area history
                area = detected_bbox["width"] * detected_bbox["height"]
                now = time.time()
                area_hist = bb_area_history[uuid][cam_id]
                area_hist.append((now, area))
                while area_hist and now - area_hist[0][0] > window_seconds:
                    area_hist.popleft()
                smoothed_area, area_rate = compute_smoothed_area_and_rate(
                    area_hist)

                # Clip flags
                resolution = camera_calibrations.get(
                    cam_id, {}).get("resolution")
                clip_flags = bbox_clip_flags(
                    detected_bbox, resolution) if detected_bbox and resolution else [0, 0, 0, 0]

                # Compose feature vector
                feature_vector = [
                    aspect_ratio_ratio,
                    v_mag,
                    smoothed_area,
                    area_rate,
                    *clip_flags
                ]

                # Store in rolling window for weighted average
                fhist = feature_history[uuid][cam_id]
                fhist.append((now, feature_vector))
                while fhist and now - fhist[0][0] > window_seconds:
                    fhist.popleft()

                # Weighted average: newer samples weighted higher
                if fhist:
                    times = np.array([t for t, _ in fhist])
                    features = np.array([fv for _, fv in fhist])
                    if len(times) > 1:
                        weights = np.linspace(1, 2, len(times))
                    else:
                        weights = np.array([1.0])
                    weights /= weights.sum()
                    feature_vector_smoothed = np.average(
                        features, axis=0, weights=weights).tolist()
                else:
                    feature_vector_smoothed = feature_vector

                person_features[uuid].append(
                    (feature_vector_smoothed, cam_id))

                # State logic (unpack features)
                aspect_ratio_ratio, v_mag, smoothed_area, area_rate, clip_left, clip_right, clip_top, clip_bottom = feature_vector_smoothed
                arr_thresh = args.fallen_arr_threshold
                area_rate_threshold = args.area_rate_threshold

                if v_mag >= args.run_velocity_threshold:
                    state = "running"
                elif v_mag >= args.walk_velocity_threshold:
                    state = "walking"
                elif (
                    aspect_ratio_ratio < arr_thresh
                    and not (clip_bottom and abs(area_rate) > area_rate_threshold)
                ):
                    state = "fallen"
                else:
                    state = "standing"

                bb_canonical_xyxy = canonical_bboxes.get(cam_id)
                bb_canonical = xyxy_to_xywh(
                    bb_canonical_xyxy) if bb_canonical_xyxy else None

                metrics_by_uuid[uuid][cam_id] = {
                    "bounding_box_px": detected_bbox,
                    "bb_canonical": bb_canonical,
                    "feature_vector": feature_vector,
                    "feature_vector_smoothed": feature_vector_smoothed,
                    "state": state
                }

        # 2. Aggregate and determine state per person, and update tracked_people
        now = time.time()
        state_priority = ["fallen", "falling",
                          "running", "walking", "standing", "unknown"]

        for uuid, feats in person_features.items():
            metrics = metrics_by_uuid[uuid]
            # Consensus: pick the "most severe" state by priority
            final_state = sorted(
                metrics.values(), key=lambda x: state_priority.index(x["state"]))[0]["state"]
            cams_seen = set(metrics.keys())
            prev = tracked_people.get(uuid)
            if prev and prev["state"] == final_state:
                state_start_time = prev.get("state_start_time", now)
            else:
                state_start_time = now
            state_duration = now - state_start_time

            tracked_people[uuid] = {
                "uuid": uuid,
                "state": final_state,
                "state_duration": state_duration,
                "state_start_time": state_start_time,
                "camera_ids": list(cams_seen),
                "last_seen": now,
                "metrics": metrics
            }

        # 3. Gather all people seen within the rolling window
        active_people = [
            {k: v for k, v in person.items() if k not in (
                "last_seen", "state_start_time")}
            for person in tracked_people.values()
            if now - person["last_seen"] < window_seconds
        ]

        # Count people in each state
        state_counts = {"fallen": 0, "standing": 0,
                        "walking": 0, "running": 0, "falling": 0, "unknown": 0}
        for person in active_people:
            state = person.get("state", "unknown")
            if state in state_counts:
                state_counts[state] += 1
            else:
                state_counts["unknown"] += 1

        scene_id = userdata.get("scene_id", "unknown")
        publish_topic = f"scenescape/fall-detection/{scene_id}"
        message = {
            "timestamp": timestamp,
            "state_counts": state_counts,
            "scene_id": scene_id,
            "people": active_people
        }
        client.publish(publish_topic, json.dumps(message))

    except Exception as e:
        print(f"Error decoding MQTT message: {e}")

def initialize_mqtt_client(**kwargs):
    if hasattr(mqtt, 'CallbackAPIVersion'):
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, **kwargs)
    else:
        return mqtt.Client(**kwargs)

def world_to_camera(pt_world, cam_extrinsics):
    t = np.array(cam_extrinsics["translation"])
    q = cam_extrinsics["rotation"]
    pose_mat = transform.CameraPose.poseToPoseMat(t, q, [1, 1, 1])
    pt_world_h = np.array([*pt_world, 1.0])
    world_to_cam = np.linalg.inv(pose_mat)
    pt_cam_h = world_to_cam @ pt_world_h
    pt_cam = pt_cam_h[:3]
    return pt_cam.tolist()

def bbox_from_pose(pose):
    points = [pt for pt in pose if pt and len(pt) == 2]
    if not points:
        return None
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}

def xyxy_to_xywh(box):
    if not box:
        return None
    x = round(box["x_min"], 3)
    y = round(box["y_min"], 3)
    width = round(box["x_max"] - box["x_min"], 3)
    height = round(box["y_max"] - box["y_min"], 3)
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height
    }

def bbox_clip_flags(bbox, resolution, margin=2):
    """Returns a list of 0/1 flags for [clip_left, clip_right, clip_top, clip_bottom]."""
    if not bbox or not resolution:
        return [0, 0, 0, 0]
    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
    img_w, img_h = resolution
    return [
        int(x <= margin),                      # clip_left
        int((x + w) >= (img_w - margin)),      # clip_right
        int(y <= margin),                      # clip_top
        int((y + h) >= (img_h - margin)),      # clip_bottom
    ]

def main():
    args = parse_args()
    print(f"Looking for controller.auth at: {args.controller_auth}")
    print(f"Current working directory: {os.getcwd()}")

    api_key = os.environ.get("SCENESCAPE_API_KEY")
    print(f"Using API key: {api_key[:6]}...")

    print(f"Scene controller: {args.broker}")
    print(f"Scene UUID: {args.scene_uuid}")
    print(f"Insecure mode: {args.insecure}")

    mqtt_topic = f"scenescape/regulated/scene/{args.scene_uuid}"
    api_url = f"{args.resturl}/cameras?scene={args.scene_uuid}"
    print(f"MQTT topic: {mqtt_topic}")
    print(f"API URL: {api_url}")

    cameras = get_cameras(api_url, api_key, args.insecure)
    if cameras is None:
        print(
            "Failed to retrieve cameras. Will keep running for debugging.", file=sys.stderr)
        # Instead of exiting, enter a wait loop for debugging
        try:
            while True:
                print("Waiting for debugging... (press Ctrl+C to exit)")
                time.sleep(60)
        except KeyboardInterrupt:
            print("Exiting on user request.")
            sys.exit(1)

    camera_calibrations = {}
    if isinstance(cameras, dict) and "results" in cameras:
        camera_list = cameras["results"]
    else:
        camera_list = cameras

    print("Retrieved camera names:")
    for cam in camera_list:
        name = cam.get('name', cam.get('uid', 'unknown'))
        intrinsics = cam.get("intrinsics") or {}
        cx = intrinsics.get("cx")
        cy = intrinsics.get("cy")
        resolution = [2 * cx, 2 * cy] if cx and cy else cam.get("resolution")
        camera_calibrations[name] = {
            "extrinsics": {
                "translation": cam.get("translation"),
                "rotation": cam.get("rotation"),
                "scale": cam.get("scale"),
            },
            "intrinsics": intrinsics,
            "distortion": cam.get("distortion"),
            "resolution": resolution,
        }

    # Example: print calibration for each camera
    for cam_name, calib in camera_calibrations.items():
        print(f"\nCalibration for {cam_name}:")
        print(json.dumps(calib, indent=2))

    sys.stdout.flush()

    userdata = {
        "mqtt_topic": mqtt_topic,
        "camera_calibrations": camera_calibrations,
        "scene_id": args.scene_uuid,
        "args": args
    }
    mqtt_client = initialize_mqtt_client(userdata=userdata)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    try:
        with open(args.controller_auth, "r") as f:
            print(f"Successfully opened {args.controller_auth}")
            auth = json.load(f)
        mqtt_client.username_pw_set(auth["user"], auth["password"])

        mqtt_client.tls_set(cert_reqs=ssl.CERT_NONE)
        mqtt_client.tls_insecure_set(True)

        print(f"Connecting to MQTT broker at {args.broker}:{args.port} ...")
        mqtt_client.connect(args.broker, args.port, 60)
        mqtt_client.loop_forever()
    except Exception as e:
        print(f"Error during MQTT setup or main loop: {e}", file=sys.stderr)
        print("Entering wait loop for debugging. (press Ctrl+C to exit)")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("Exiting on user request.")
            sys.exit(1)

if __name__ == "__main__":
    main()
