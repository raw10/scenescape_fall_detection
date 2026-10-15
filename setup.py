#!/usr/bin/env python3

import requests
import getpass
import sys
import os
from pathlib import Path
import re
import json
import subprocess
import time

SCENESCAPE_CERT_HOSTNAME = os.environ.get(
    "SCENESCAPE_CERT_HOSTNAME", "web.scenescape.intel.com")

class SceneScapeHTTPSAdapter(requests.adapters.HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        pool_kwargs["assert_hostname"] = SCENESCAPE_CERT_HOSTNAME
        pool_kwargs["server_hostname"] = SCENESCAPE_CERT_HOSTNAME
        return super().init_poolmanager(
            connections, maxsize, block=block, **pool_kwargs)

session = requests.Session()
session.mount("https://localhost", SceneScapeHTTPSAdapter())

def read_env_file(env_path):
    values = {}
    if not os.path.isfile(env_path):
        return values
    with open(env_path, "r") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value
    return values

def update_env_file(env_path, values):
    lines = []
    if os.path.isfile(env_path):
        with open(env_path, "r") as env_file:
            lines = env_file.readlines()
    remaining = dict(values)
    for index, line in enumerate(lines):
        key = line.split("=", 1)[0].strip() if "=" in line else None
        if key in remaining:
            lines[index] = f"{key}={remaining.pop(key)}\n"
    lines.extend(f"{key}={value}\n" for key, value in remaining.items())
    with open(env_path, "w") as env_file:
        env_file.writelines(lines)

def prompt_for_api_key(env_path):
    api_key = os.environ.get("SCENESCAPE_API_KEY") \
        or read_env_file(env_path).get("SCENESCAPE_API_KEY")
    if api_key:
        print("Using the existing SceneScape API key.")
    else:
        print("Please enter your SceneScape API key (you can find this in the admin panel):")
        api_key = getpass.getpass("API Key: ")
        os.environ["SCENESCAPE_API_KEY"] = api_key  # Set for this process

    update_env_file(env_path, {"SCENESCAPE_API_KEY": api_key})
    os.chmod(env_path, 0o600)
    print(f"API key written to app-owned environment file {env_path}.")
    return api_key

def prompt_for_scenescape_path():
    default_path = os.path.expanduser("~/scenescape")
    user_input = input(f"Enter the path to your SceneScape install [{default_path}]: ").strip()
    scenescape_path = user_input if user_input else default_path
    if not os.path.isdir(scenescape_path):
        print(f"Directory '{scenescape_path}' does not exist. Please check the path and try again.")
        sys.exit(1)
    print(f"Using SceneScape install at: {scenescape_path}")
    return scenescape_path

def ensure_dir_exists(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def copy_into_volume(src_dir, volume_name):
    """Copy the contents of src_dir into a named Docker volume using a temporary container."""
    src_dir = os.path.abspath(src_dir)
    subprocess.run(
        [
            "docker", "run", "--rm",
            "-v", f"{src_dir}:/src:ro",
            "-v", f"{volume_name}:/dst",
            "alpine", "sh", "-c", "cp -r /src/. /dst/"
        ],
        check=True
    )

def copy_model_and_videos(project_dir, compose_project_name):
    volume_prefix = compose_project_name
    src_models = os.path.join(project_dir, "model")
    if os.path.isdir(src_models):
        model_volume = f"{volume_prefix}_vol-models"
        print(f"Copying model files into Docker volume {model_volume}...")
        copy_into_volume(src_models, model_volume)
    else:
        print(f"No model directory found at {src_models}")

    src_videos = os.path.join(project_dir, "dataset")
    video_volume = f"{volume_prefix}_vol-videos"
    print(f"Copying dataset files into Docker volume {video_volume}...")
    copy_into_volume(src_videos, video_volume)

def compose_command(scenescape_path, override_path, env_path, *args):
    base_compose = os.path.join(
        scenescape_path, "sample_data", "compose", "docker-compose-dl-streamer-example.yml")
    return [
        "docker", "compose",
        "--project-directory", scenescape_path,
        "--env-file", env_path,
        "-f", base_compose,
        "-f", override_path,
        "--profile", "controller",
        *args,
    ]

def start_manager(scenescape_path, override_path, env_path):
    print("Applying SceneScape manager thread limits...")
    subprocess.run(
        compose_command(
            scenescape_path, override_path, env_path,
            "up", "-d", "web"),
        check=True,
    )
    readiness_url = "https://localhost:443/api/v1/database-ready"
    for _ in range(30):
        try:
            response = session.get(readiness_url, timeout=5)
            if response.ok and response.json().get("databaseReady") is True:
                print("SceneScape manager is ready.")
                return
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2)
    raise RuntimeError("SceneScape manager did not become ready in time.")

def get_scenes(api_key, scenescape_path):
    # Use the local API endpoint for scenes
    api_url = "https://localhost/api/v1/"
    headers = {"Authorization": f"Token {api_key}"}
    try:
        resp = session.get(f"{api_url}scenes", headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("results", [])
        print("Failed to fetch scenes from API.")
        sys.exit(1)
    except Exception as e:
        print(f"Could not connect to SceneScape API: {e}")
        sys.exit(1)

def get_image_version(image_name):
    try:
        output = subprocess.check_output(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            universal_newlines=True
        )
        versions = [
            line.split(":")[1]
            for line in output.splitlines()
            if line.startswith(f"{image_name}:")
        ]
        for v in versions:
            if v != "latest":
                return v
        if versions:
            return versions[0]
    except Exception as e:
        print(f"Could not determine {image_name} version from docker images: {e}")
    return "latest"

def add_camera(api_url, api_key, scene_uid, camera):
    headers = {"Authorization": f"Token {api_key}", "Content-Type": "application/json"}
    distortion = camera.get("distortion")
    if distortion and any(value != 0 for value in distortion.values()):
        raise ValueError(
            f"Camera '{camera.get('name')}' has nonzero distortion, which the "
            "SceneScape 2026.3 camera API cannot store safely."
        )
    # The 2026.3 resolution field injects an invalid `cam` model argument.
    payload = {
        "sensor_id": camera.get("uid"),
        "name": camera.get("name"),
        "scene": scene_uid,
        "translation": camera.get("extrinsics", {}).get("translation", [0, 0, 0]),
        "rotation": camera.get("extrinsics", {}).get("rotation", [0, 0, 0]),
        "scale": camera.get("extrinsics", {}).get("scale", [1, 1, 1]),
        "intrinsics": camera.get("intrinsics"),
        "transform_type": "euler"
    }
    response = session.get(
        f"{api_url}cameras?scene={scene_uid}", headers=headers, timeout=10)
    response.raise_for_status()
    camera_data = response.json()
    existing_cameras = camera_data.get("results", []) \
        if isinstance(camera_data, dict) else camera_data
    existing = next(
        (item for item in existing_cameras
         if item.get("uid") == payload["sensor_id"]
         or item.get("name") == payload["name"]),
        None,
    )
    if existing:
        camera_uid = existing.get("uid") or payload["sensor_id"]
        if existing.get("scene") == scene_uid:
            payload.pop("scene")
        response = session.put(
            f"{api_url}camera/{camera_uid}", headers=headers,
            json=payload, timeout=10)
        action = "updated"
    else:
        response = session.post(
            f"{api_url}camera", headers=headers, json=payload, timeout=10)
        action = "created"
    if not response.ok:
        raise requests.HTTPError(
            f"Camera '{payload['name']}' {action} failed: "
            f"HTTP {response.status_code}: {response.text}",
            response=response,
        )
    print(f"Camera '{payload['name']}' {action}.")
    if action == "created":
        return payload["sensor_id"]
    return None

def record_created_camera(generated_dir, scene_uid, camera_uid):
    path = os.path.join(generated_dir, "created-cameras.json")
    with open(path, "r") as registry_file:
        created_cameras = json.load(registry_file)
    entry = {"scene": scene_uid, "uid": camera_uid}
    if entry not in created_cameras:
        created_cameras.append(entry)
        with open(path, "w") as registry_file:
            json.dump(created_cameras, registry_file)

def load_cameras_from_file(cameras_file):
    with open(cameras_file, "r") as f:
        data = json.load(f)
    # If the file is a dict with a "cameras" key, return that
    if isinstance(data, dict) and "cameras" in data:
        return data["cameras"]
    # If it's already a list, return as is
    if isinstance(data, list):
        return data
    raise ValueError("cameras.json format not recognized (should be a list or have a 'cameras' key)")

def select_scene(api_url, api_key):
    headers = {"Authorization": f"Token {api_key}"}
    try:
        resp = session.get(f"{api_url}/scenes", headers=headers, timeout=10)
        resp.raise_for_status()
        scenes = resp.json().get("results", []) if isinstance(resp.json(), dict) else resp.json()
    except Exception as e:
        print(f"Failed to retrieve scenes from API: {e}")
        sys.exit(1)

    if not scenes:
        print("No scenes found. Please create a scene in SceneScape before continuing.")
        sys.exit(1)
    elif len(scenes) == 1:
        scene = scenes[0]
        print(f"Only one scene found: {scene.get('name', scene.get('uid', 'unknown'))} (UUID: {scene.get('uid', scene.get('uuid', ''))})")
        return scene.get('uid') or scene.get('uuid')
    else:
        print("Available scenes:")
        for idx, scene in enumerate(scenes, 1):
            print(f"{idx}. {scene.get('name', scene.get('uid', 'unknown'))} (UUID: {scene.get('uid', scene.get('uuid', ''))})")
        while True:
            try:
                choice = int(input(f"Select a scene [1-{len(scenes)}]: "))
                if 1 <= choice <= len(scenes):
                    return scenes[choice - 1].get('uid') or scenes[choice - 1].get('uuid')
            except Exception:
                pass
            print("Invalid selection. Please try again.")

def start_node_red(scenescape_path, override_path, env_path):
    print("Starting Node-RED container...")
    subprocess.run(
        compose_command(
            scenescape_path, override_path, env_path,
            "up", "-d", "node-red"),
        check=True,
    )
    # Wait for Node-RED to be ready
    for _ in range(30):
        try:
            r = requests.get("http://localhost:1880")
            if r.status_code == 200:
                print("Node-RED is up!")
                return
        except Exception:
            pass
        print("Waiting for Node-RED to be ready...")
        time.sleep(2)
    print("Node-RED did not start in time.")
    exit(1)

def install_npm_modules(modules):
    """Install a list of npm modules in Node-RED via its admin API."""
    for module in modules:
        print(f"Installing {module}...")
        resp = requests.post("http://localhost:1880/nodes", json={"module": module})
        if resp.status_code == 200:
            print(f"{module} installed.")
        else:
            print(f"Failed to install {module}: {resp.text}")

# Install required npm modules for Node-RED
modules_to_install = [
    "node-red-dashboard"
]

def setup_flows(flows_path, scene_uuid, auth_path):
    print("Importing flows with updated scene UUID and MQTT credentials...")

    # Load credentials from controller.auth
    with open(auth_path) as f:
        auth = json.load(f)
    mqtt_user = auth.get("user", "")
    mqtt_pass = auth.get("password", "")

    # Load and update flows
    with open(flows_path) as f:
        flows = json.load(f)
    for node in flows:
        # Update MQTT topics
        if node.get("type") in ("mqtt in", "mqtt out") and "topic" in node:
            node["topic"] = node["topic"].replace("SCENE-UUID", scene_uuid)
        # Update MQTT broker credentials
        if node.get("type") == "mqtt-broker":
            node["credentials"] = {"user": mqtt_user, "password": mqtt_pass}

    # Import updated flows into Node-RED
    resp = requests.post("http://localhost:1880/flows", json=flows)
    if 200 <= resp.status_code < 300:
        print("Flows imported successfully.")
    else:
        print(f"Failed to import flows: {resp.status_code} {resp.text}")

def prompt_create_scene(dataset_dir):
    # Look for a .png file in the dataset directory
    png_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith('.png')]
    if not png_files:
        print(f"No .png file found in {dataset_dir}. Please add a scene image before continuing.")
        sys.exit(1)
    scene_image = png_files[0]
    print(f"Found scene image: {scene_image}")

    # Parse pixels per meter from filename (e.g., 73p76ppm means 73.76 pixels per meter)
    match = re.search(r'(\d+)p(\d+)ppm', scene_image)
    if match:
        ppm = float(f"{match.group(1)}.{match.group(2)}")
        print(f"Parsed pixels per meter from filename: {ppm}")
    else:
        print("Could not parse pixels per meter from filename. Please ensure the filename contains the ppm (e.g., 73p76ppm).")
        sys.exit(1)

    # Prompt user to create the scene in SceneScape UI
    print("\nBefore continuing, please create a scene in the SceneScape UI:")
    print(f"  - Use the image: {os.path.abspath(os.path.join(dataset_dir, scene_image))}")
    print(f"  - Set pixels per meter: {ppm}")
    print("Once the scene is created, press Enter to continue...")
    input()

def ensure_secretsdir_env(scenescape_path=None):
    if "SECRETSDIR" not in os.environ or not os.environ["SECRETSDIR"]:
        if scenescape_path:
            secretsdir = os.path.join(os.path.abspath(scenescape_path), "manager","secrets")
        else:
            secretsdir = "./secrets"
        os.environ["SECRETSDIR"] = secretsdir
        print(f'Set environment variable: SECRETSDIR={secretsdir}')

def main():
    project_dir = os.path.abspath(os.getcwd())
    dataset_dir = os.path.join(project_dir, "dataset")
    prompt_create_scene(dataset_dir)

    scenescape_path = os.path.abspath(prompt_for_scenescape_path())
    ca_cert_path = os.path.join(scenescape_path, "manager/secrets/certs/scenescape-ca.pem")
    if not os.path.isfile(ca_cert_path):
        print(f"CA certificate not found at {ca_cert_path}. Please check your SceneScape install.")
        sys.exit(1)
    session.verify = ca_cert_path

    default_app_path = project_dir
    app_path = input(
        f"Enter the path to your fall_detection_app [{default_app_path}]: ").strip()
    fall_detection_app_path = os.path.abspath(app_path or default_app_path)
    generated_dir = os.path.join(fall_detection_app_path, ".generated")
    ensure_dir_exists(generated_dir)
    created_cameras_path = os.path.join(generated_dir, "created-cameras.json")
    if not os.path.exists(created_cameras_path):
        with open(created_cameras_path, "w") as registry_file:
            json.dump([], registry_file)
    env_path = os.path.join(generated_dir, "fall-detection.env")

    # Ensure node_red_data exists and is owned by the current user
    node_red_data_path = os.path.join(fall_detection_app_path, "node_red_data")
    if not os.path.isdir(node_red_data_path):
        os.makedirs(node_red_data_path, exist_ok=True)
        print(f"Created node_red_data directory at {node_red_data_path}")
    # Set ownership to current user
    try:
        uid = os.getuid()
        gid = os.getgid()
        os.chown(node_red_data_path, uid, gid)
        print(f"Set ownership of {node_red_data_path} to UID:{uid} GID:{gid}")
    except Exception as e:
        print(f"Warning: Could not set ownership of {node_red_data_path}: {e}")

    api_key = prompt_for_api_key(env_path)

    # Prompt for API URL (or set default)
    api_url = "https://localhost:443/api/v1"

    # Select scene
    scene_uid = select_scene(api_url, api_key)

    # Prompt for proxy usage
    use_proxy = input("Do you want to use a proxy server for node-red? [y/N]: ").strip().lower()
    http_proxy = ""
    https_proxy = ""
    if use_proxy in ("y", "yes"):
        http_proxy = input("Enter the HTTP proxy URL (or leave blank): ").strip()
        # If the user enters an HTTP proxy, use it as the default for HTTPS proxy
        https_proxy = input(f"Enter the HTTPS proxy URL (or leave blank, default: {http_proxy}): ").strip()
        if not https_proxy and http_proxy:
            https_proxy = http_proxy

    version_path = os.path.join(scenescape_path, "version.txt")
    with open(version_path, "r") as version_file:
        scenescape_version = version_file.read().strip()
    compose_project_name = os.environ.get("COMPOSE_PROJECT_NAME", "scenescape")
    secrets_dir = os.path.join(scenescape_path, "manager", "secrets")
    scenescape_env = read_env_file(os.path.join(scenescape_path, ".env"))
    update_env_file(env_path, {
        **scenescape_env,
        "SCENESCAPE_API_KEY": api_key,
        "SECRETSDIR": secrets_dir,
        "VERSION": scenescape_version,
        "COMPOSE_PROJECT_NAME": compose_project_name,
        "HTTP_PROXY": http_proxy,
        "HTTPS_PROXY": https_proxy,
        "http_proxy": http_proxy,
        "https_proxy": https_proxy,
        "NO_PROXY": os.environ.get("NO_PROXY", ""),
        "no_proxy": os.environ.get("no_proxy", ""),
    })
    os.chmod(env_path, 0o600)

    template_path = os.path.join(
        fall_detection_app_path, "docker-compose.override.template.yml")
    with open(template_path) as f:
        template = f.read()
    override = template.replace("{{SCENESCAPE_VERSION}}", scenescape_version)
    override = override.replace("{{SCENE_UUID}}", scene_uid)
    override = override.replace("{{FALL_DETECTION_APP_PATH}}", fall_detection_app_path)
    override = override.replace("{{SCENESCAPE_PATH}}", scenescape_path)
    override_path = os.path.join(generated_dir, "docker-compose.override.yml")
    with open(override_path, "w") as f:
        f.write(override)
    print(f"App-owned Docker Compose override written to {override_path}")

    copy_model_and_videos(project_dir, compose_project_name)
    start_manager(scenescape_path, override_path, env_path)

    # Add cameras from calibration file
    api_url = "https://localhost/api/v1/"
    dataset_dir = os.path.join(os.getcwd(), "dataset")
    cameras_file = os.path.join(dataset_dir, "cameras.json")
    if not os.path.isfile(cameras_file):
        print(f"Camera calibration file not found at {cameras_file}")
        sys.exit(1)
    cameras = load_cameras_from_file(cameras_file)
    for cam in cameras:
        created_uid = add_camera(api_url, api_key, scene_uid, cam)
        if created_uid:
            record_created_camera(generated_dir, scene_uid, created_uid)

    # Start node-red service
    start_node_red(scenescape_path, override_path, env_path)

    # Now it's safe to install modules and import flows
    install_npm_modules(modules_to_install)
    setup_flows(
        flows_path=os.path.join(fall_detection_app_path, "flows.json"),
        scene_uuid=scene_uid,
        auth_path=os.path.join(
            scenescape_path, "manager", "secrets", "controller.auth")
    )

    print("\nSetup complete!")
    print("Starting all services...")
    subprocess.run(
        compose_command(
            scenescape_path, override_path, env_path,
            "up", "-d", "--build"),
        check=True,
    )

    print("\nYou can now view the Node-RED dashboard UI at:  http://<host>:1880/ui")
    print("And the SceneScape UI at:                       https://<host>\n")

if __name__ == "__main__":
    main()