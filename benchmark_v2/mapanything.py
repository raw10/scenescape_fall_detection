#!/usr/bin/env python3

"""Generate aligned MapAnything scenes using frames without detected people when available."""

import argparse
import base64
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

import requests
import trimesh

if __package__:
  from benchmark_v2.alignment import align_geometry, transform_camera, validate_camera
  from benchmark_v2.cameras import camera_name, select_videos
else:
  from alignment import align_geometry, transform_camera, validate_camera
  from cameras import camera_name, select_videos


ROOT = Path(__file__).resolve().parent


def select_mapping_frame(detections):
  if not detections:
    raise ValueError("No frames detected")
  start = None
  previous = -math.inf
  timestamps = []
  for index, record in enumerate(detections):
    timestamp = record["timestamp"] / 1e9
    if not math.isfinite(timestamp) or timestamp < 0 or timestamp <= previous:
      raise ValueError("Detection timestamps must be finite, nonnegative and strictly increasing")
    previous = timestamp
    timestamps.append(timestamp)
    if record["objects"]:
      start = None
    else:
      if start is None:
        start = index
      if timestamp - timestamps[start] >= 0.5:
        midpoint = (timestamp + timestamps[start]) / 2
        return min(range(start, index + 1), key=lambda candidate: abs(timestamps[candidate] - midpoint))
  return None


def extract_mapping_frame(video_path: Path, frame_path: Path, model=None, confidence=0.15):
  if not 0 <= confidence <= 1:
    raise ValueError("Confidence must be between 0 and 1")
  with tempfile.TemporaryDirectory() as detections_dir:
    command = [
        sys.executable, str(ROOT / "detect_videos.py"),
        "--input", str(video_path.parent.parent), "--chute", video_path.parent.name,
        "--cameras", video_path.stem, "--output", detections_dir,
        "--confidence", str(confidence),
    ]
    if model is not None:
      command.extend(["--model", str(model)])
    subprocess.run(command, check=True)
    detections = json.loads(
        (Path(detections_dir) / video_path.parent.name / f"{video_path.stem}.json").read_text())
  frame_index = select_mapping_frame(detections)
  fallback = frame_index is None
  if frame_index is None:
    frame_index = 0
    print(f"{video_path}: no 0.5-second window without a detected person; using first frame", flush=True)
  else:
    print(f"{video_path}: using frame {frame_index} from a 0.5-second detection-free window", flush=True)
  subprocess.run([
      "ffmpeg", "-v", "error", "-i", str(video_path),
      "-vf", f"select=eq(n\\,{frame_index})", "-frames:v", "1", "-y", str(frame_path),
  ], check=True)
  return dict(frame_index=frame_index, source_time_s=detections[frame_index]["timestamp"] / 1e9,
              fallback=fallback, confidence=confidence, person_free_window_s=0.5,
              model=str(model) if model is not None else "yolo11s-pose")


def generate_scene(session: requests.Session, api_url: str, video_path: Path,
                   timeout: int, camera_id: str) -> None:
  generate_multiview(session, api_url, [video_path], [camera_id],
                     video_path.parent, timeout)


def video_resolution(video_path):
  probe = subprocess.run(
      ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
       "stream=width,height", "-of", "json", str(video_path)],
      check=True, capture_output=True, text=True)
  stream = json.loads(probe.stdout)["streams"][0]
  width, height = stream["width"], stream["height"]
  if not isinstance(width, int) or not isinstance(height, int) or min(width, height) <= 0:
    raise ValueError(f"Invalid video dimensions: {video_path}")
  return width, height


def mapping_cameras(result, camera_ids):
  indexed = []
  for field in ("camera_poses", "intrinsics"):
    items = result[field]
    if not isinstance(items, list) or len(items) != len(camera_ids):
      raise ValueError(f"Expected exactly one {field} entry per requested camera")
    by_id = {}
    for item in items:
      identifier = item["camera_id"]
      if identifier in by_id or identifier not in camera_ids:
        raise ValueError(f"Duplicate or unexpected {field} camera ID: {identifier}")
      by_id[identifier] = item
    indexed.append(by_id)
  return indexed[0], indexed[1]


def generate_multiview(session, api_url, videos, camera_ids, directory, timeout):
  if not videos or len(videos) != len(camera_ids) or len(set(camera_ids)) != len(camera_ids):
    raise ValueError("Expected one distinct camera ID per input video")
  if len({video.resolve() for video in videos}) != len(videos):
    raise ValueError("Input videos must be distinct")
  resolutions = [video_resolution(video) for video in videos]

  with tempfile.TemporaryDirectory() as temp_dir, ExitStack() as stack:
    files = []
    for index, video in enumerate(videos):
      frame_path = Path(temp_dir) / f"frame-{index}.jpg"
      extract_mapping_frame(video, frame_path)
      if not frame_path.is_file() or frame_path.stat().st_size == 0:
        raise ValueError(f"No mapping frame extracted from {video}")
      frame = stack.enter_context(frame_path.open("rb"))
      files.append(("images", (frame_path.name, frame, "image/jpeg")))
    data = [("output_format", "glb"), ("mesh_type", "mesh")]
    data.extend(("camera_ids", identifier) for identifier in camera_ids)
    response = session.post(
        f"{api_url}/reconstruction", files=files, data=data, timeout=60)
    response.raise_for_status()
    request_id = response.json()["request_id"]

  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    response = session.get(
        f"{api_url}/reconstruction/status/{request_id}", timeout=30)
    response.raise_for_status()
    status = response.json()

    if status["state"] == "complete":
      result = dict(status["result"])
      poses, intrinsics = mapping_cameras(result, camera_ids)
      for identifier, (width, height) in zip(camera_ids, resolutions):
        validate_camera(poses[identifier], intrinsics[identifier]["K"], width, height)
      glb_data = base64.b64decode(result.pop("glb_data"), validate=True)
      if not glb_data.startswith(b"glTF"):
        raise ValueError("Mapping service returned an invalid GLB file")
      directory.mkdir(parents=True, exist_ok=True)
      (directory / "scene.raw.glb").write_bytes(glb_data)
      (directory / "mapping.json").write_text(
          json.dumps(result, indent=2, allow_nan=False) + "\n")
      return
    if status["state"] == "failed":
      raise RuntimeError(status.get("error", "Mapping reconstruction failed"))

    time.sleep(2)

  raise TimeoutError(f"Reconstruction timed out after {timeout} seconds")


def align_scene(video_path: Path, camera_id: str, meters_per_unit: float,
                floor_plane) -> None:
  directory = video_path.parent
  result = json.loads((directory / "mapping.json").read_text())
  poses = [pose for pose in result["camera_poses"] if pose["camera_id"] == camera_id]
  intrinsics = [item for item in result["intrinsics"] if item["camera_id"] == camera_id]
  if len(poses) != 1 or len(intrinsics) != 1:
    raise ValueError(f"Expected one pose and intrinsics for {camera_id}")
  probe = subprocess.run(
      ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
       "stream=width,height", "-of", "json", str(video_path)],
      check=True, capture_output=True, text=True)
  stream = json.loads(probe.stdout)["streams"][0]
  scene = trimesh.load_scene(directory / "scene.raw.glb", process=False)
  mesh, camera, alignment = align_geometry(
      scene, poses[0], intrinsics[0]["K"], stream["width"], stream["height"],
      meters_per_unit, floor_plane)
  camera.update(uid=camera_id, name=camera_id, transform_type="quaternion")
  mesh.export(directory / "scene.glb", file_type="glb")
  (directory / "camera.json").write_text(json.dumps(camera, indent=2, allow_nan=False) + "\n")
  (directory / "alignment.json").write_text(json.dumps(alignment, indent=2, allow_nan=False) + "\n")


def align_multiview(videos, camera_ids, directory, meters_per_unit, floor_plane):
  result = json.loads((directory / "mapping.json").read_text())
  poses, intrinsics = mapping_cameras(result, camera_ids)
  resolutions = [video_resolution(video) for video in videos]
  for identifier, (width, height) in zip(camera_ids, resolutions):
    validate_camera(poses[identifier], intrinsics[identifier]["K"], width, height)
  reference = next((index for index, video in enumerate(videos) if video.stem == "cam2"), 0)
  identifier = camera_ids[reference]
  scene = trimesh.load_scene(directory / "scene.raw.glb", process=False)
  mesh, _, alignment = align_geometry(
      scene, poses[identifier], intrinsics[identifier]["K"], *resolutions[reference],
      meters_per_unit, floor_plane)
  cameras = {}
  for video, identifier, (width, height) in zip(videos, camera_ids, resolutions):
    camera = transform_camera(poses[identifier], intrinsics[identifier]["K"],
                              width, height, alignment["mesh_world_to_scene"])
    camera.update(uid=identifier, name=video.stem, transform_type="quaternion",
                  source_video=str(video.resolve()))
    cameras[video.stem] = camera
  alignment["reference_camera"] = videos[reference].stem
  mesh.export(directory / "scene.glb", file_type="glb")
  (directory / "cameras.json").write_text(json.dumps(cameras, indent=2, allow_nan=False) + "\n")
  (directory / "alignment.json").write_text(json.dumps(alignment, indent=2, allow_nan=False) + "\n")


def selected_chutes(data_dir, chute):
  if chute is not None:
    if not chute.startswith("chute") or Path(chute).name != chute:
      raise ValueError("--chute must name a chute directory, not a path")
    directory = data_dir / chute
    if not directory.is_dir():
      raise FileNotFoundError(directory)
    return [directory]
  directories = sorted(path for path in data_dir.glob("chute*") if path.is_dir())
  if not directories:
    raise FileNotFoundError(f"No chute* directories found in {data_dir}")
  return directories


def camera_selection(values):
  if values == ["all"]:
    return None
  if len(set(values)) != len(values):
    raise ValueError("Duplicate camera selection")
  return [camera_name(value) for value in values]


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Generate aligned scene.glb and camera.json for each chute video")
  parser.add_argument(
      "--url", default="https://localhost:8444/v1",
      help="Mapping service API URL (default: %(default)s)")
  parser.add_argument(
      "--data-dir", type=Path, default=ROOT / "data",
      help="Directory containing chute folders (default: %(default)s)")
  parser.add_argument(
      "--timeout", type=int, default=900,
      help="Timeout per reconstruction in seconds (default: %(default)s)")
  parser.add_argument("--realign", action="store_true",
                      help="Reuse scene.raw.glb and mapping.json without calling MapAnything")
  parser.add_argument("--meters-per-unit", type=float, default=1.0)
  parser.add_argument("--floor-plane", nargs=4, type=float, metavar=("NX", "NY", "NZ", "D"))
  parser.add_argument("--chute", help="Select one chute directory")
  parser.add_argument("--cameras", nargs="+", help="Joint views: cam1 cam2 ... or all alone")
  parser.add_argument("--output", type=Path, help="Required separate mapping root for joint views")
  args = parser.parse_args()

  multiview = args.cameras is not None
  if multiview and args.output is None:
    parser.error("--output is required with --cameras")
  if not multiview and args.output is not None:
    parser.error("--output requires --cameras; omit both for legacy cam2 mapping")
  names = camera_selection(args.cameras) if multiview else ["cam2"]
  directories = selected_chutes(args.data_dir, args.chute)
  selections = []
  for directory in directories:
    if not multiview and args.chute is None and not (directory / "cam2.avi").is_file():
      continue
    videos = select_videos(directory, names)
    if multiview:
      destination = (args.output / directory.name).resolve()
      source = args.data_dir.resolve()
      if destination == source or source in destination.parents:
        parser.error("--output must be outside --data-dir to protect legacy assets")
    else:
      destination = directory
    selections.append((videos, destination))
  if not selections:
    raise FileNotFoundError(f"No chute*/cam2.avi files found in {args.data_dir}")

  api_url = args.url.rstrip("/")
  session = requests.Session()
  session.verify = False

  if not args.realign:
    health_response = session.get(f"{api_url}/health", timeout=30)
    health_response.raise_for_status()
    health = health_response.json()
    if health.get("model") != "mapanything" or health.get("device") != "cpu":
      raise RuntimeError("Mapping service must run MapAnything in CPU mode")

  for videos, destination in selections:
    camera_ids = []
    for video in videos:
      identity = f"{video.parent.name}/{video.stem}"
      if multiview:
        identity = f"multiview/{destination}/{identity}"
      camera_ids.append("bench-" + hashlib.sha256(identity.encode()).hexdigest()[:12])
    if not args.realign:
      print(f"Generating scene for {videos[0].parent.name}...", flush=True)
      if multiview:
        generate_multiview(session, api_url, videos, camera_ids, destination, args.timeout)
      else:
        generate_scene(session, api_url, videos[0], args.timeout, camera_ids[0])
    if multiview:
      align_multiview(videos, camera_ids, destination, args.meters_per_unit, args.floor_plane)
    else:
      align_scene(videos[0], camera_ids[0], args.meters_per_unit, args.floor_plane)
    print(f"Saved aligned mapping in {destination}", flush=True)


if __name__ == "__main__":
  main()
