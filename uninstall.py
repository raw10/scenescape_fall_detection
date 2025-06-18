#!/usr/bin/python3

import os
import sys
import shutil
import glob
import requests
import getpass
import json
import subprocess
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def prompt_for_scenescape_path():
    default_path = os.path.expanduser("~/scenescape")
    user_input = input(f"Enter the path to your SceneScape install [{default_path}]: ").strip()
    scenescape_path = user_input if user_input else default_path
    if not os.path.isdir(scenescape_path):
        print(f"Directory '{scenescape_path}' does not exist. Please check the path and try again.")
        sys.exit(1)
    print(f"Using SceneScape install at: {scenescape_path}")
    return scenescape_path

def prompt_for_app_path():
    default_app_path = os.getcwd()
    app_path = input(f"Enter the path to your fall_detection_app [{default_app_path}]: ").strip()
    return app_path if app_path else default_app_path

def prompt_for_api_key(scenescape_path):
    api_key = os.environ.get("SCENESCAPE_API_KEY")
    if not api_key:
        env_path = os.path.join(scenescape_path, ".env")
        if os.path.isfile(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("SCENESCAPE_API_KEY="):
                        api_key = line.strip().split("=", 1)[1]
                        break
    if not api_key:
        api_key = getpass.getpass("Enter your SceneScape API key: ")
    return api_key

def delete_cameras(api_url, api_key, scene_uid):
    headers = {"Authorization": f"Token {api_key}"}
    resp = requests.get(f"{api_url}/cameras?scene={scene_uid}", headers=headers, verify=False)
    print("Status code:", resp.status_code)
    if resp.status_code != 200:
        print(f"Failed to fetch cameras: {resp.text}")
        return
    try:
        data = resp.json()
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        return
    cameras = data.get("results", []) if isinstance(data, dict) else data
    print("Cameras found:", cameras)
    for cam in cameras:
        cam_id = cam.get("uid") or cam.get("id")
        if not cam_id:
            continue
        del_resp = requests.delete(f"{api_url}/camera/{cam_id}", headers=headers, verify=False)
        print(f"Delete status: {del_resp.status_code}, response: {del_resp.text}")
        if del_resp.status_code in (200, 202, 204):
            print(f"Deleted camera: {cam.get('name', cam_id)}")
        else:
            print(f"Failed to delete camera {cam.get('name', cam_id)}: {del_resp.text}")

def remove_copied_models(model_src, dst_models):
    if not os.path.isdir(model_src) or not os.path.isdir(dst_models):
        return

    # Track directories from the source model structure
    src_dirs = set()
    for root, dirs, files in os.walk(model_src):
        rel_dir = os.path.relpath(root, model_src)
        src_dirs.add(rel_dir)

        # Remove matching files
        dst_dir = os.path.join(dst_models, rel_dir) if rel_dir != '.' else dst_models
        for file in files:
            dst_file = os.path.join(dst_dir, file)
            if os.path.isfile(dst_file):
                print(f"Removing model file: {dst_file}")
                os.remove(dst_file)

    # Only remove empty directories that match the source structure
    for rel_dir in sorted(src_dirs, reverse=True):  # deepest first
        dst_dir = os.path.join(dst_models, rel_dir) if rel_dir != '.' else dst_models
        if os.path.isdir(dst_dir) and not os.listdir(dst_dir):
            print(f"Removing empty model directory: {dst_dir}")
            os.rmdir(dst_dir)

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
            print(f"{idx}: {scene.get('name', scene.get('uid', scene.get('uuid', '')))} (UUID: {scene.get('uid', scene.get('uuid', ''))})")
        while True:
            try:
                choice = int(input(f"Select a scene [1-{len(scenes)}]: "))
                if 1 <= choice <= len(scenes):
                    return scenes[choice - 1].get('uid') or scenes[choice - 1].get('uuid')
            except Exception:
                pass
            print("Invalid selection. Please try again.")

def main():
    scenescape_path = prompt_for_scenescape_path()
    app_path = prompt_for_app_path()

    # Remove only .mp4 files that exist in both dataset and sample_data
    dataset_videos = set(f for f in os.listdir(os.path.join(app_path, "dataset")) if f.lower().endswith('.mp4'))
    sample_data_dst = os.path.join(scenescape_path, "sample_data")
    for video in dataset_videos:
        video_path = os.path.join(sample_data_dst, video)
        if os.path.isfile(video_path):
            print(f"Removing video: {video_path}")
            os.remove(video_path)

    # Remove only model files that exist in both model and scenescape/models
    model_src = os.path.join(app_path, "model")
    dst_models = os.path.join(scenescape_path, "models")
    remove_copied_models(model_src, dst_models)

    # Remove .env file
    env_path = os.path.join(scenescape_path, ".env")
    if os.path.isfile(env_path):
        print(f"Removing {env_path}")
        os.remove(env_path)

    # Remove controller.auth from app directory
    auth_path = os.path.join(app_path, "controller.auth")
    if os.path.isfile(auth_path):
        print(f"Removing {auth_path}")
        os.remove(auth_path)

    # Remove node_red_data directory (optional)
    node_red_data_path = os.path.join(scenescape_path, "node_red_data")
    if os.path.isdir(node_red_data_path):
        resp = input(
            f"\nThe directory {node_red_data_path} contains all Node-RED flows and data, "
            "including flows that may not be affiliated with this project.\n"
            "Do you want to remove this folder? [y/N]: "
        ).strip().lower()
        if resp in ("y", "yes"):
            print(f"Removing {node_red_data_path}")
            shutil.rmtree(node_red_data_path)
        else:
            print(f"Keeping {node_red_data_path}")

    # Delete cameras from the scene
    api_key = prompt_for_api_key(scenescape_path)
    api_url = "https://localhost/api/v1"
    scene_uid = select_scene(api_url, api_key)
    if scene_uid:
        delete_cameras(api_url, api_key, scene_uid)

    # Remove docker-compose.override.yml
    override_path = os.path.join(scenescape_path, "docker-compose.override.yml")
    if os.path.isfile(override_path):
        print(f"Removing {override_path}")
        os.remove(override_path)
        # Ensure SECRETSDIR is set before running docker compose up
        if "SECRETSDIR" not in os.environ or not os.environ["SECRETSDIR"]:
            os.environ["SECRETSDIR"] = "secrets"
            print('Set environment variable: SECRETSDIR=secrets')
        # After removing the override, bring down any orphaned containers
        print("Running: docker compose up -d --remove-orphans")
        subprocess.run(
            ["docker", "compose", "up", "-d", "--remove-orphans"],
            cwd=scenescape_path,
            env=os.environ
        )

    print("Uninstall complete.")

if __name__ == "__main__":
    main()