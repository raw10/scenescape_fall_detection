#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
from pathlib import Path

if __package__:
  from benchmark_v2.cameras import camera_name, select_videos
else:
  from cameras import camera_name, select_videos


DEFAULT_IMAGE = "docker.io/intel/dlstreamer-pipeline-server:2026.3.0-20260915-weekly-ubuntu24"
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models/yolo11s-pose/FP16/yolo11s-pose.xml"
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".m4v", ".ts", ".webm"}


def main():
  parser = argparse.ArgumentParser(description="Run DL Streamer person detection on every video")
  parser.add_argument("--input", type=Path, default=Path(__file__).parent / "data")
  parser.add_argument("--output", type=Path, default=Path(__file__).parent / "detections-yolo11s-pose")
  parser.add_argument("--image", default=DEFAULT_IMAGE)
  parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
  parser.add_argument("--confidence", type=float, help="Override detection confidence threshold (0 to 1)")
  parser.add_argument("--chute", help="Select one chute directory")
  parser.add_argument("--cameras", nargs="+", help="Select cam1 cam2 ... or all alone")
  parser.add_argument("--overwrite", action="store_true", help="Replace existing selected detection files")
  args = parser.parse_args()
  if args.confidence is not None and not 0 <= args.confidence <= 1:
    parser.error("--confidence must be between 0 and 1")

  input_dir = args.input.resolve()
  output_dir = args.output.resolve()
  if args.chute is not None or args.cameras is not None:
    names = None
    if args.cameras is not None and args.cameras != ["all"]:
      names = [camera_name(value) for value in args.cameras]
      if len(set(names)) != len(names):
        parser.error("Duplicate camera selection")
    if args.chute is not None:
      if not args.chute.startswith("chute") or Path(args.chute).name != args.chute:
        parser.error("--chute must name a chute directory, not a path")
      directories = [input_dir / args.chute]
    else:
      directories = sorted(path for path in input_dir.glob("chute*") if path.is_dir())
    videos = []
    for directory in directories:
      if not directory.is_dir():
        raise FileNotFoundError(directory)
      videos.extend(select_videos(directory, names))
  else:
    videos = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
  if not videos:
    raise SystemExit(f"No videos found in {input_dir}")
  outputs = [output_dir / video.relative_to(input_dir).with_suffix(".json") for video in videos]
  if len(set(outputs)) != len(outputs):
    raise ValueError("Multiple input videos would write the same detection output")
  for output in outputs:
    if output.exists() and not args.overwrite:
      raise FileExistsError(f"Detection output exists: {output}; use fresh --output or explicit --overwrite")
  model = args.model.resolve(strict=True)

  for video, output in zip(videos, outputs):
    relative = video.relative_to(input_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output.exists():
      output.unlink()
    container_input = Path("/data") / relative
    container_output = Path("/output") / output.relative_to(output_dir)

    pipeline = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--entrypoint", "gst-launch-1.0",
        "-v", f"{input_dir}:/data:ro",
        "-v", f"{output_dir}:/output",
        "-v", f"{model.parent}:/model:ro",
        args.image, "-q",
        "filesrc", f"location={container_input}", "!",
        "decodebin", "!", "videoconvert", "!", "video/x-raw,format=BGR", "!",
        "gvadetect", f"model=/model/{model.name}", "device=CPU",
        "pre-process-backend=opencv", "!",
        "gvametaconvert", "add-tensor-data=true", "add-empty-results=true", "!",
        "gvametapublish", "method=file", "file-format=json",
        f"file-path={container_output}", "!", "fakesink", "sync=false",
    ]
    if args.confidence is not None:
      pipeline.insert(pipeline.index("pre-process-backend=opencv"), f"threshold={args.confidence}")
    print(f"Detecting {relative} -> {output.relative_to(output_dir)}", flush=True)
    subprocess.run(pipeline, check=True)
    records = json.loads(output.read_text())
    for record in records:
      record["objects"] = [
          obj for obj in record.get("objects", [])
          if obj.get("detection", {}).get("label") == "person"
      ]
    output.write_text(json.dumps(records))

  print(f"Wrote {len(videos)} detection files to {output_dir}")


if __name__ == "__main__":
  main()