"""Run the YOLO fall baseline and compare source-time alerts with SceneScape."""

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import cv2
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from ultralytics import YOLO

if __package__ in (None, ""):
  sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark_v2 import evaluate as evaluation
from benchmark_v2.cameras import camera_name


BASE = Path(__file__).resolve().parents[1]
BASELINE = BASE / "results-yolo11s-pose-quarter-speed-20261026T114844Z"


@dataclass
class FallAlerts:
  gap_s: float = 0.25
  armed: bool = True
  fall_free_since: float | None = None
  previous_time: float | None = None

  def __post_init__(self):
    if not math.isfinite(self.gap_s) or self.gap_s <= 0:
      raise ValueError("fall-free gap must be positive and finite")

  def update(self, timestamp, confidence):
    if not math.isfinite(timestamp) or timestamp < 0:
      raise ValueError("source timestamps must be nonnegative and finite")
    if self.previous_time is not None and timestamp <= self.previous_time:
      raise ValueError("source timestamps must be strictly increasing")
    first_frame = self.previous_time is None
    self.previous_time = timestamp
    if self.fall_free_since is not None:
      elapsed = timestamp - self.fall_free_since
      if elapsed >= self.gap_s or math.isclose(elapsed, self.gap_s, abs_tol=1e-9):
        self.armed = True
    if confidence is None:
      if self.fall_free_since is None:
        self.fall_free_since = timestamp
      return None
    self.fall_free_since = None
    if not self.armed:
      return None
    self.armed = False
    return {"video_time_s": timestamp, "confidence": confidence,
        "first_observation": first_frame}


def retained_detections(boxes, threshold):
  return [{"class_id": int(box[5]), "confidence": float(box[4]),
       "box_xyxy": [float(coordinate) for coordinate in box[:4]]}
      for box in boxes if int(box[5]) in (0, 1) and box[4] >= threshold]


def write_json(path, value):
  path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def frame_times(video):
  probe = subprocess.run([
    "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
    "-show_entries", "frame=best_effort_timestamp_time:stream=start_time,nb_frames:format=start_time,duration",
    "-of", "json", str(video)], check=True, capture_output=True, text=True)
  metadata = json.loads(probe.stdout)
  stream = metadata["streams"][0]
  origin = float(metadata["format"].get("start_time", stream.get("start_time", "nan")))
  duration = float(metadata["format"]["duration"])
  if not math.isfinite(origin) or not math.isfinite(duration) or duration <= 0:
    raise ValueError("video time origin or duration is invalid")
  timestamps = [float(frame["best_effort_timestamp_time"]) - origin
          for frame in metadata["frames"]]
  if not timestamps:
    raise ValueError("video has no timestamped frames")
  previous = -1.0
  for timestamp in timestamps:
    if not math.isfinite(timestamp) or not 0 <= timestamp < duration or timestamp <= previous:
      raise ValueError("frame timestamps must increase within the source duration")
    previous = timestamp
  declared = stream.get("nb_frames", "N/A")
  if declared != "N/A" and int(declared) != len(timestamps):
    raise ValueError("ffprobe frame count differs from the declared video frame count")
  return timestamps, origin, probe.stderr.strip()


def comparison_speed(results, directories, camera="cam2"):
  camera = camera_name(camera)
  speeds = []
  for directory in directories:
    path = results / f"{directory.name}.json"
    if not path.is_file():
      continue
    result = evaluation.load_json(path)
    if len(evaluation.result_cameras(result, camera)) != 1:
      raise ValueError("YOLO comparison requires a single-camera SceneScape result")
    if result.get("chute") != directory.name:
      raise ValueError(f"{path}: chute does not match directory")
    speed = evaluation.number(result.get("playback_speed"), "baseline playback_speed")
    if not 0 < speed < 1:
      raise ValueError(f"{path}: baseline playback_speed must be between 0 and 1")
    speeds.append(speed)
  if not speeds:
    raise ValueError("no matching SceneScape results from which to read playback speed")
  if any(not math.isclose(speed, speeds[0], rel_tol=1e-6, abs_tol=1e-9) for speed in speeds):
    raise ValueError("SceneScape baseline has mixed playback speeds")
  return speeds[0]


def process_clip(directory, output, model, video_backend, settings, camera="cam2"):
  camera = camera_name(camera)
  if settings.get("camera", camera) != camera:
    raise ValueError("YOLO settings camera does not match selected camera")
  video = directory / f"{camera}.avi"
  result = {"chute": directory.name, "status": "failed", "events": [],
        "camera": camera, "reference_camera": camera,
        "playback_speed": settings["playback_speed"], "detector": "yolo11-best",
        "source_video": str(video.resolve()), "settings": settings,
        "processed_frames": 0,
        "timing": "Frame PTS minus file playback origin, in source seconds. playback_speed "
        "is the comparison timeline only; inference is offline without replay sleeps."}
  capture = None
  try:
    timestamps, origin, warnings = frame_times(video)
    result.update(expected_frames=len(timestamps), source_time_origin_s=origin,
        decoder_warnings=warnings)
    trace_dir = output / directory.name
    trace_dir.mkdir()
    alerts = FallAlerts(settings["fall_free_gap_s"])
    capture = video_backend.VideoCapture(str(video))
    if not capture.isOpened():
      raise ValueError(f"cannot open {video}")
    with (trace_dir / "predictions.jsonl").open("x") as trace:
      for timestamp in timestamps:
        available, frame = capture.read()
        if not available:
          raise ValueError(f"video ended after {result['processed_frames']} of {len(timestamps)} frames")
        prediction = model.predict(frame, conf=settings["confidence"], imgsz=settings["image_size"],
                 device="cpu", verbose=False)[0]
        detections = retained_detections(prediction.boxes.data.cpu().tolist(), settings["confidence"])
        fall_confidence = max((box["confidence"] for box in detections if box["class_id"] == 1),
                   default=None)
        event = alerts.update(timestamp, fall_confidence)
        if event is not None:
          result["events"].append(event)
        trace.write(json.dumps({"video_time_s": timestamp, "detections": detections},
                     allow_nan=False) + "\n")
        result["processed_frames"] += 1
      if capture.read()[0]:
        raise ValueError("decoder produced more frames than ffprobe timestamps")
    effective = vars(model.predictor.args)
    result["inference_settings"] = {key: effective.get(key)
        for key in ("imgsz", "conf", "iou", "max_det", "agnostic_nms", "half", "rect", "classes")}
    result["inference_settings"]["device"] = str(model.predictor.device)
    result["status"] = "complete"
  except Exception as error:
    result["error"] = f"{type(error).__name__}: {error}"
  finally:
    if capture is not None:
      capture.release()
  write_json(output / f"{directory.name}.json", result)
  print(f"{directory.name}: {result['status']}; {result['processed_frames']} frames; "
      f"{len(result['events'])} alerts" + (f"; {result['error']}" if "error" in result else ""), flush=True)
  return result


def write_comparison(yolo_rows, scene_rows, output, settings):
  output.mkdir(parents=True, exist_ok=True)
  yolo = {row["chute"]: row for row in yolo_rows}
  scene = {row["chute"]: row for row in scene_rows}
  if yolo.keys() != scene.keys():
    raise ValueError("comparison methods must cover the same requested clip names")
  common = sorted(name for name in yolo if yolo[name]["status"] == scene[name]["status"] == "scored")
  excluded = sorted(set(yolo) - set(common))
  columns = ["gt", "tp", "fp", "fn", "duplicate_fp", "outside_window_fp", "precision", "recall", "f1",
       "median_delay_source_s", "median_delay_replay_s"]
  records = []
  if common:
    for label, rows in (("YOLO11", yolo), ("SceneScape", scene)):
      selected = [rows[name] for name in common]
      summary = evaluation.aggregate(selected)
      delays = [detail["delay_source_s"] for row in selected for detail in row["details"]
            if detail["kind"] == "TP"]
      summary["median_delay_source_s"] = statistics.median(delays) if delays else None
      records.append({"method": label, "common_clips": len(common),
              **{key: summary[key] for key in columns}})
  environment = Environment(
    loader=FileSystemLoader(Path(__file__).resolve().parent),
    undefined=StrictUndefined, autoescape=False, keep_trailing_newline=True,
    trim_blocks=True, lstrip_blocks=True)
  environment.filters["display"] = evaluation.display
  report = environment.get_template("comparison.md.j2").render(
    settings=settings, yolo=yolo, scene=scene, common=common, excluded=excluded,
    columns=columns, records=records,
    yolo_coverage=sum(row["status"] == "scored" for row in yolo_rows),
    scene_coverage=sum(row["status"] == "scored" for row in scene_rows))
  (output / "comparison.md").write_text(report)
  with (output / "comparison.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["method", "common_clips"] + columns)
    writer.writeheader()
    for record in records:
      writer.writerow({key: evaluation.display(value) for key, value in record.items()})
  return common


def reports(args, settings, camera=None):
  camera = camera_name(camera if camera is not None else getattr(args, "camera", "cam2"))
  evaluation.result_cameras(settings, camera)
  slowdown = 1 / settings["playback_speed"]
  coverage_end_tolerance_s = getattr(args, "coverage_end_tolerance_s", 0)
  evaluation.validate_settings(slowdown, 1, coverage_end_tolerance_s)
  output = args.output / "reports"
  chutes = getattr(args, "chutes", None) or [args.chute]
  yolo_rows = []
  scene_rows = []
  for chute in chutes:
    yolo_rows.extend(evaluation.evaluate(args.data_dir, args.output, slowdown, 1, chute, camera,
      coverage_end_tolerance_s=coverage_end_tolerance_s))
    scene_rows.extend(evaluation.evaluate(args.data_dir, args.scenescape_results, slowdown, 1, chute, camera,
      coverage_end_tolerance_s=coverage_end_tolerance_s))
  if any(len(row.get("cameras", [camera])) != 1 for row in scene_rows):
    raise ValueError("YOLO comparison requires single-camera SceneScape results")
  settings = {**settings, "camera": camera, "reference_camera": camera, "decoder_warning_clips": [],
      "coverage_end_tolerance_s": coverage_end_tolerance_s}
  for row in yolo_rows:
    path = args.output / f"{row['chute']}.json"
    if path.is_file():
      result = evaluation.load_json(path)
      if result.get("decoder_warnings"):
        settings["decoder_warning_clips"].append(row["chute"])
      if row["status"] == "failed" and result.get("error"):
        row["reason"] = result["error"]
  evaluation.write_report(yolo_rows, args.output, output / "yolo", slowdown, 1,
      coverage_end_tolerance_s=coverage_end_tolerance_s)
  evaluation.write_report(scene_rows, args.scenescape_results, output / "scenescape", slowdown, 1,
      coverage_end_tolerance_s=coverage_end_tolerance_s)
  common = write_comparison(yolo_rows, scene_rows, output, settings)
  print(f"Comparison: {output / 'comparison.md'} ({len(common)}/{len(yolo_rows)} common clips)")
  return 0 if len(common) == len(yolo_rows) and common else 1


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--weights", type=Path, default=Path(__file__).resolve().with_name("best.pt"))
  parser.add_argument("--data-dir", type=Path, default=BASE / "data")
  parser.add_argument("--output", type=Path, default=BASE / "results-yolo11-best-conf04-gap025")
  parser.add_argument("--scenescape-results", type=Path, default=BASELINE)
  selection = parser.add_mutually_exclusive_group()
  selection.add_argument("--chute")
  selection.add_argument("--chutes", nargs="+")
  parser.add_argument("--camera", type=camera_name, default="cam2")
  parser.add_argument("--conf", type=float, default=0.4)
  parser.add_argument("--fall-free-gap-s", type=float, default=0.25)
  parser.add_argument("--coverage-end-tolerance-s", type=float, default=0)
  parser.add_argument("--evaluate-only", action="store_true", help="Rebuild reports without inference")
  args = parser.parse_args(argv)
  try:
    evaluation.validate_settings(4, 1, args.coverage_end_tolerance_s)
    if not math.isfinite(args.conf) or not 0 < args.conf <= 1:
      raise ValueError("confidence must be in (0, 1]")
    FallAlerts(args.fall_free_gap_s)
    settings = evaluation.load_json(args.output / "run.json") if args.evaluate_only else None
    requested = args.chutes or ([args.chute] if args.chute else None)
    if settings is not None and settings.get("clips"):
      if requested is None:
        args.chutes = requested = settings["clips"]
      elif any(name not in settings["clips"] for name in requested):
        raise ValueError("selected chutes are not in saved run clips")
    if requested is not None:
      if len(set(requested)) != len(requested):
        raise ValueError("selected chute names must be unique")
      for name in requested:
        if Path(name).name != name or not name.startswith("chute"):
          raise ValueError("--chute/--chutes must contain chute directory names")
        if not (args.data_dir / name).is_dir():
          raise ValueError(f"selected chute directory does not exist: {args.data_dir / name}")
      directories = [args.data_dir / name for name in requested]
    else:
      directories = sorted(directory for directory in args.data_dir.glob("chute*") if directory.is_dir())
    if not directories:
      raise ValueError("no selected chute directories found")
    speed = comparison_speed(args.scenescape_results, directories, args.camera)
    if args.evaluate_only:
      evaluation.result_cameras(settings, args.camera)
      if not math.isclose(settings["playback_speed"], speed, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError("saved YOLO speed differs from selected baseline")
      settings.update(scenescape_results=str(args.scenescape_results.resolve()),
              yolo_results=str(args.output.resolve()),
              coverage_end_tolerance_s=args.coverage_end_tolerance_s)
      return reports(args, settings)
    if args.output.exists():
      raise ValueError(f"output already exists: {args.output}; choose a fresh --output or --evaluate-only")
    if not args.weights.is_file():
      raise ValueError(f"local weights not found: {args.weights}")
    model = YOLO(str(args.weights))
    if model.task != "detect" or model.names != {0: "non-fall", 1: "fall"}:
      raise ValueError(f"unsupported model contract: task={model.task}, names={model.names}; "
               "agree an input strategy before proceeding")
    with args.weights.open("rb") as handle:
      fingerprint = hashlib.file_digest(handle, "sha256").hexdigest()
    settings = {"weights": str(args.weights.resolve()), "weights_sha256": fingerprint,
          "camera": args.camera, "reference_camera": args.camera,
          "task": model.task, "classes": model.names, "confidence": args.conf,
          "fall_free_gap_s": args.fall_free_gap_s, "playback_speed": speed,
          "coverage_end_tolerance_s": args.coverage_end_tolerance_s,
          "image_size": model.overrides.get("imgsz", 640), "device": "cpu",
          "versions": {name: version(name) for name in ("ultralytics", "torch", "opencv-python")},
          "scenescape_results": str(args.scenescape_results.resolve()),
          "yolo_results": str(args.output.resolve()),
          "clips": [directory.name for directory in directories]}
    args.output.mkdir(parents=True)
    write_json(args.output / "run.json", settings)
    failed = False
    for directory in directories:
      result = process_clip(directory, args.output, model, cv2, settings, args.camera)
      failed |= result["status"] != "complete"
    return max(int(failed), reports(args, settings))
  except (ValueError, OSError, RuntimeError, ImportError, KeyError, TypeError, subprocess.SubprocessError) as error:
    parser.exit(2, f"YOLO benchmark failed: {error}\n")


if __name__ == "__main__":
  raise SystemExit(main())