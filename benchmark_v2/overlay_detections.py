#!/usr/bin/env python3

import argparse
import json
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

from detect_videos import VIDEO_EXTENSIONS


def main():
  parser = argparse.ArgumentParser(description="Draw saved DL Streamer boxes on videos using FFmpeg")
  parser.add_argument("--input", type=Path, default=Path(__file__).parent / "data")
  parser.add_argument("--detections", type=Path, default=Path(__file__).parent / "detections-yolo11s-pose")
  parser.add_argument("--output", type=Path, default=Path(__file__).parent / "overlays-yolo11s-pose")
  parser.add_argument("--format", choices=("webp", "mp4"), default="webp")
  args = parser.parse_args()

  for video in sorted(args.input.rglob("*")):
    if video.suffix.lower() not in VIDEO_EXTENSIONS or not video.is_file():
      continue
    relative = video.relative_to(args.input)
    records = json.loads((args.detections / relative.with_suffix(".json")).read_text())
    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate", "-of", "json", str(video),
    ], text=True)
    frame_rate = float(Fraction(json.loads(probe)["streams"][0]["r_frame_rate"]))
    filters = []
    for record in records:
      timestamp = record["timestamp"] / 1e9
      for person in record.get("objects", []):
        if person.get("detection", {}).get("label") != "person":
          continue
        filters.append(
            f"drawbox=x={person['x']}:y={person['y']}:w={person['w']}:h={person['h']}:"
            f"color=lime:t=2:enable='lt(abs(t-{timestamp:.9f}),{0.5 / frame_rate:.9f})'"
        )
    if args.format == "webp":
      filters.append("fps=20")
      encoding = ["-c:v", "libwebp_anim", "-quality", "75", "-loop", "0"]
    else:
      encoding = ["-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    output = args.output / relative.with_suffix(f".{args.format}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as graph:
      graph.write(",".join(filters) or "null")
      graph.flush()
      print(f"Overlaying {relative} -> {output}", flush=True)
      subprocess.run([
          "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
          "-filter_script:v", graph.name, "-an", *encoding, str(output),
      ], check=True)


if __name__ == "__main__":
  main()