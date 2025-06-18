#!/usr/bin/python3

import requests
import getpass
import sys
import os
from pathlib import Path
import urllib3
import ssl
import re
import json
import shutil
import glob
import subprocess
import stat
import time
from PIL import Image

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

session = requests.Session()
session.verify = False

from requests.packages.urllib3.util.ssl_ import create_urllib3_context

class HostNameIgnoreAdapter(requests.adapters.HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)

session.mount('https://', HostNameIgnoreAdapter())

def prompt_for_api_key(scenescape_path):
    api_key = os.environ.get("SCENESCAPE_API_KEY")
    if api_key:
        print("Using SceneScape API key from environment variable SCENESCAPE_API_KEY.")
    else:
        print("Please enter your SceneScape API key (you can find this in the admin panel):")
        api_key = getpass.getpass("API Key: ")
        os.environ["SCENESCAPE_API_KEY"] = api_key  # Set for this process

    # Write to .env file in the scenescape directory for docker-compose
    env_path = os.path.join(scenescape_path, ".env")
    with open(env_path, "w") as envf:
        envf.write(f'SCENESCAPE_API_KEY={api_key}\n')
    print(f"API key written to {env_path} for docker-compose.")
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

def copy_model_and_videos(project_dir, scenescape_dir):
    # Copy all .mp4 files from <project_dir>/dataset to scenescape/sample_data
    dataset_dir = os.path.join(project_dir, "dataset")
    sample_data_dst = os.path.join(scenescape_dir, "sample_data")
    ensure_dir_exists(sample_data_dst)
    video_files = glob.glob(os.path.join(dataset_dir, "*.mp4"))
    if not video_files:
        print(f"No .mp4 files found in {dataset_dir}")
    for video in video_files:
        dst = os.path.join(sample_data_dst, os.path.basename(video))
        if not os.path.isfile(dst):
            print(f"Copying video: {video} -> {dst}")
            shutil.copy2(video, dst)
        else:
            print(f"Video already exists, skipping: {dst}")

    # Copy model directory (recursively)
    src_models = os.path.join(project_dir, "model")
    dst_models = os.path.join(scenescape_dir, "models")
    if os.path.isdir(src_models):
        print(f"Copying model files from {src_models} to {dst_models}")
        for root, dirs, files in os.walk(src_models):
            rel_path = os.path.relpath(root, src_models)
            dst_root = os.path.join(dst_models, rel_path)
            ensure_dir_exists(dst_root)
            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst_root, file)
                if not os.path.isfile(dst_file):
                    print(f"Copying model file: {src_file} -> {dst_file}")
                    shutil.copy2(src_file, dst_file)
                else:
                    print(f"Model file already exists, skipping: {dst_file}")
    else:
        print(f"No model directory found at {src_models}")

def get_scenes(api_key, scenescape_path):
    # Use the local API endpoint for scenes
    api_url = "https://localhost/api/v1/"
    headers = {"Authorization": f"Token {api_key}"}
    try:
        resp = session.get(f"{api_url}scenes", headers=headers, timeout=10, verify=False)
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
    # Only send supported fields
    payload = {
        "name": camera.get("name"),
        "scene": scene_uid,
        "translation": camera.get("extrinsics", {}).get("translation", [0, 0, 0]),
        "rotation": camera.get("extrinsics", {}).get("rotation", [0, 0, 0]),
        "scale": camera.get("extrinsics", {}).get("scale", [1, 1, 1]),
        "transform_type": "euler"
    }
    resp = session.post(f"{api_url}camera", headers=headers, json=payload, timeout=10, verify=False)
    if resp.status_code == 201:
        print(f"Camera '{payload['name']}' created.")
    elif resp.status_code == 400 and "name" in resp.text:
        print(f"Camera '{payload['name']}' already exists, skipping.")
    else:
        print(f"Failed to create camera '{payload['name']}'': {resp.text}")

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
        resp = requests.get(f"{api_url}/scenes", headers=headers, verify=False, timeout=10)
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

def copy_controller_auth(scenescape_path, app_path):
    src = os.path.join(scenescape_path, "secrets", "controller.auth")
    dst = os.path.join(app_path, "controller.auth")
    if not os.path.isfile(src):
        print(f"controller.auth not found at {src}")
        sys.exit(1)
    print(f"Copying {src} to {dst}")
    shutil.copy2(src, dst)
    os.chmod(dst, 0o644)  # <-- Add this line to set permissions to rw-r--r--
    print(f"Set permissions of {dst} to 644 (rw-r--r--)")

def start_node_red(scenescape_path):
    print("Starting Node-RED container...")
    subprocess.run(["docker", "compose", "up", "-d", "node-red"], cwd=scenescape_path)
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

def ensure_secretsdir_env():
    if "SECRETSDIR" not in os.environ or not os.environ["SECRETSDIR"]:
        os.environ["SECRETSDIR"] = "secrets"
        print('Set environment variable: SECRETSDIR=secrets')

def main():
    ensure_secretsdir_env()

    dataset_dir = os.path.join(os.getcwd(), "dataset")
    prompt_create_scene(dataset_dir)

    scenescape_path = prompt_for_scenescape_path()
    ca_cert_path = os.path.join(scenescape_path, "secrets/certs/scenescape-ca.pem")
    if not os.path.isfile(ca_cert_path):
        print(f"CA certificate not found at {ca_cert_path}. Please check your SceneScape install.")
        sys.exit(1)

    # Ensure node_red_data exists and is owned by the current user
    node_red_data_path = os.path.join(scenescape_path, "node_red_data")
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

    api_key = prompt_for_api_key(scenescape_path)

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

    # Prompt for fall_detection_app path to mount
    default_app_path = os.getcwd()
    app_path = input(f"Enter the path to your fall_detection_app [{default_app_path}]: ").strip()
    fall_detection_app_path = app_path if app_path else default_app_path

    percebro_version = get_image_version("scenescape-percebro")
    scenescape_version = get_image_version("scenescape")

    with open("docker-compose.override.template.yml") as f:
        template = f.read()
    override = template.replace("{{PERCEBRO_VERSION}}", percebro_version)
    override = override.replace("{{SCENESCAPE_VERSION}}", scenescape_version)
    override = override.replace("{{SCENE_UUID}}", scene_uid)
    override = override.replace("{{FALL_DETECTION_APP_PATH}}", fall_detection_app_path)
    override = override.replace("{{HTTP_PROXY}}", http_proxy)
    override = override.replace("{{HTTPS_PROXY}}", https_proxy)
    override_path = os.path.join(scenescape_path, "docker-compose.override.yml")
    with open(override_path, "w") as f:
        f.write(override)
    print(f"Docker-compose override written to {override_path}")

    project_dir = os.getcwd()
    copy_model_and_videos(project_dir, scenescape_path)

    # Add cameras from calibration file
    api_url = "https://localhost/api/v1/"
    dataset_dir = os.path.join(os.getcwd(), "dataset")
    cameras_file = os.path.join(dataset_dir, "cameras.json")
    if not os.path.isfile(cameras_file):
        print(f"Camera calibration file not found at {cameras_file}")
        sys.exit(1)
    cameras = load_cameras_from_file(cameras_file)
    for cam in cameras:
        add_camera(api_url, api_key, scene_uid, cam)

    # Copy controller.auth to app path
    copy_controller_auth(scenescape_path, fall_detection_app_path)

    # Start node-red service
    start_node_red(scenescape_path)

    # Now it's safe to install modules and import flows
    install_npm_modules(modules_to_install)
    setup_flows(
        flows_path=os.path.join(fall_detection_app_path, "flows.json"),
        scene_uuid=scene_uid,
        auth_path=os.path.join(fall_detection_app_path, "controller.auth")
    )

    print("\nSetup complete!")
    print("Starting all services...")
    subprocess.run(["docker", "compose", "up", "-d"], cwd=scenescape_path)

    print("\nYou can now view the Node-RED dashboard UI at:  http://<host>:1880/ui")
    print("And the SceneScape UI at:                       https://<host>\n")

if __name__ == "__main__":
    main()