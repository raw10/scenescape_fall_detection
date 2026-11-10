"""Evaluate fall alerts against reviewed source-video intervals."""

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_v2.cameras import camera_name


def number(value, name):
  if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
    raise ValueError(f"{name} must be a finite number")
  return float(value)


def validate_settings(slowdown, tolerance, coverage_end_tolerance_s=0):
  if number(slowdown, "slowdown factor") <= 1:
    raise ValueError("slowdown factor must be greater than 1")
  if number(tolerance, "tolerance") < 0:
    raise ValueError("tolerance must be nonnegative")
  if number(coverage_end_tolerance_s, "coverage end tolerance") < 0:
    raise ValueError("coverage end tolerance must be nonnegative")


def label_duration(duration, label_end, coverage_end_tolerance_s):
  return max(duration, label_end) if label_end - duration < coverage_end_tolerance_s else duration


def metrics(tp, fp, fn):
  def ratio(numerator, denominator):
    return numerator / denominator if denominator else None

  return {"precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
      "f1": ratio(2 * tp, 2 * tp + fp + fn)}


def score(falls, events, duration, slowdown, tolerance=1.0, coverage_end_tolerance_s=0):
  """Match in source seconds; apply slowdown only when reporting replay delay."""
  validate_settings(slowdown, tolerance, coverage_end_tolerance_s)
  if number(duration, "video duration") <= 0:
    raise ValueError("video duration must be positive")
  if not isinstance(falls, list) or not isinstance(events, list):
    raise ValueError("falls and events must be lists")
  for fall in falls:
    start = number(fall.get("start_s"), "start_s")
    end = number(fall.get("end_s"), "end_s")
    if not 0 <= start < end <= label_duration(duration, end, coverage_end_tolerance_s):
      raise ValueError(f"invalid interval {start}-{end}; video duration is {duration}s")
  ordered_falls = sorted(falls, key=lambda fall: fall["start_s"])
  for previous, current in zip(ordered_falls, ordered_falls[1:]):
    if current["start_s"] < previous["end_s"]:
      raise ValueError("overlapping fall intervals require person-aware evaluation")
  for event in events:
    timestamp = number(event.get("video_time_s"), "video_time_s")
    if not 0 <= timestamp <= duration:
      raise ValueError(f"event at {timestamp}s is outside video duration {duration}s")

  matched = set()
  details = []
  for event in sorted(events, key=lambda event: event["video_time_s"]):
    timestamp = event["video_time_s"]
    eligible = [index for index, fall in enumerate(ordered_falls)
          if max(0, fall["start_s"] - tolerance) <= timestamp
          <= min(duration, fall["end_s"] + tolerance)]
    available = [index for index in eligible if index not in matched]
    detail = {"video_time_s": timestamp, "person_id": event.get("person_id", ""),
         "first_observation": event.get("first_observation", False),
         "ambiguous": len(eligible) > 1}
    if available:
      index = available[0]
      matched.add(index)
      delay = timestamp - ordered_falls[index]["start_s"]
      detail.update(kind="TP", fall=ordered_falls[index], delay_source_s=delay,
             delay_replay_s=delay * slowdown)
    else:
      detail["kind"] = "duplicate FP" if eligible else "outside-window FP"
    details.append(detail)
  missed = [fall for index, fall in enumerate(ordered_falls) if index not in matched]
  tp = len(matched)
  fp = len(events) - tp
  duplicates = sum(detail["kind"] == "duplicate FP" for detail in details)
  delays = [detail["delay_replay_s"] for detail in details if detail["kind"] == "TP"]
  return {"gt": len(falls), "tp": tp, "fp": fp, "fn": len(missed),
      "duplicate_fp": duplicates, "outside_window_fp": fp - duplicates,
      **metrics(tp, fp, len(missed)), "details": details, "missed": missed,
      "median_delay_replay_s": statistics.median(delays) if delays else None}


def load_json(path):
  value = json.loads(path.read_text())
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected a JSON object")
  return value


def video_duration(path):
  result = subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
    check=True, capture_output=True, text=True)
  duration = float(json.loads(result.stdout)["format"]["duration"])
  if not math.isfinite(duration) or duration <= 0:
    raise ValueError(f"{path}: invalid video duration")
  return duration


def result_cameras(result, camera="cam2"):
  camera = camera_name(camera)
  selected = result.get("cameras", result.get("selected_cameras"))
  if "cameras" in result and "selected_cameras" in result and result["cameras"] != result["selected_cameras"]:
    raise ValueError("conflicting result camera selections")
  declared = result.get("camera")
  reference = result.get("reference_camera")
  if selected is not None:
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected):
      raise ValueError("result cameras must be a nonempty unique list")
    selected = [camera_name(name) for name in selected]
    if reference is None:
      if len(selected) != 1:
        raise ValueError("combined result requires reference_camera")
      reference = selected[0]
    if declared is not None and declared not in (reference, "combined"):
      raise ValueError("result camera does not match reference_camera")
  else:
    reference = reference if reference is not None else declared
    reference = "cam2" if reference is None else reference
    selected = [camera_name(reference)]
    if declared is not None and declared != reference:
      raise ValueError("result camera does not match reference_camera")
  if camera_name(reference) != camera or camera not in selected:
    raise ValueError(f"result camera selection does not match {camera}")
  return selected


def video_origin(path):
  probe = subprocess.run([
    "ffprobe", "-v", "error", "-show_entries", "format=start_time", "-of", "json", str(path)],
    check=True, capture_output=True, text=True)
  return number(float(json.loads(probe.stdout)["format"]["start_time"]), "video source origin")


def validate_coverage(directory, result, selected, falls, duration, slowdown, tolerance,
           coverage_end_tolerance_s=0):
  durations = {}
  origins = []
  for name in selected:
    video = directory / f"{name}.avi"
    source_duration = duration if name == result.get("reference_camera", selected[0]) else video_duration(video)
    score(falls, [], source_duration, slowdown, tolerance, coverage_end_tolerance_s)
    durations[name] = source_duration
    if len(selected) > 1:
      origins.append(video_origin(video))
  if origins and any(not math.isclose(origin, origins[0], abs_tol=1e-6, rel_tol=0)
             for origin in origins):
    raise ValueError("selected videos do not share a common source timeline origin")
  windows = result.get("source_windows", result.get("per_camera"))
  if windows is not None:
    if not isinstance(windows, dict) or set(windows) != set(selected):
      raise ValueError("source_windows must cover exactly the selected cameras")
  elif "source_start_s" in result or "source_end_s" in result:
    if len(selected) > 1:
      raise ValueError("combined result requires per-camera source windows")
    windows = {name: result for name in selected}
  if windows is not None:
    for name in selected:
      window = windows[name]
      start = number(window.get("source_start_s"), f"{name} source_start_s")
      end = number(window.get("source_end_s"), f"{name} source_end_s")
      if not 0 <= start <= end <= durations[name]:
        raise ValueError(f"{name}: invalid source window")
      if any(fall["start_s"] < start or fall["end_s"] > label_duration(
          end, fall["end_s"], coverage_end_tolerance_s) for fall in falls):
        raise ValueError(f"{name}: source window does not fully cover reviewed falls")
  return durations


def evaluate_clip(directory, results, slowdown, tolerance, camera="cam2", coverage_end_tolerance_s=0):
  camera = camera_name(camera)
  row = {"chute": directory.name, "status": "unreviewed", "gt": None,
     "camera": camera, "reference_camera": camera,
     "slowdown_factor": slowdown, "playback_speed": 1 / slowdown,
    "tolerance_source_s": tolerance, "coverage_end_tolerance_s": coverage_end_tolerance_s}
  result_path = results / f"{directory.name}.json"
  result = load_json(result_path) if result_path.is_file() else None
  if result is not None:
    if result.get("chute") != directory.name:
      raise ValueError("result chute does not match directory")
    selected = result_cameras(result, camera)
    row.update(cameras=selected, source_video=str(directory / f"{camera}.avi"))
    for key in ("timing", "source_windows", "per_camera", "source_start_s", "source_end_s",
          "source_time_origin_s", "replay_video_zero_utc"):
      if key in result:
        row[key] = result[key]
  annotation_path = directory / "ground_truth.json"
  if not annotation_path.is_file():
    return row
  annotation = load_json(annotation_path)
  reviewed = annotation.get("reviewed", False)
  if not isinstance(reviewed, bool):
    raise ValueError("reviewed must be true or false")
  if not reviewed:
    return row
  duration = video_duration(directory / f"{camera}.avi")
  falls = annotation.get("falls")
  validated = score(falls, [], duration, slowdown, tolerance, coverage_end_tolerance_s)
  label_end = max((fall["end_s"] for fall in falls), default=0)
  row.update(gt=validated["gt"], source_duration_s=duration,
      replay_duration_s=duration * slowdown, label_end_s=label_end,
      coverage_end_shortfall_s=max(0, label_end - duration))
  if result is None:
    row["status"] = "missing"
    return row
  speed = number(result.get("playback_speed"), "result playback_speed")
  if not math.isclose(speed, 1 / slowdown, rel_tol=1e-6, abs_tol=1e-9):
    raise ValueError(f"result playback_speed={speed} does not match slowdown {slowdown}")
  if result.get("status") != "complete":
    row.update(status="failed", reason=result.get("status", "missing result status"))
    return row
  row["source_durations_s"] = validate_coverage(
    directory, result, selected, falls, duration, slowdown, tolerance, coverage_end_tolerance_s)
  row["source_end_shortfalls_s"] = {name: max(0, label_end - source_duration)
      for name, source_duration in row["source_durations_s"].items()}
  windows = result.get("source_windows", result.get("per_camera"))
  if windows is None and "source_end_s" in result:
    windows = {name: result for name in selected}
  if windows is not None:
    row["source_window_end_shortfalls_s"] = {
        name: max(0, label_end - windows[name]["source_end_s"]) for name in selected}
  row.update(score(falls, result.get("events"), duration, slowdown, tolerance,
           coverage_end_tolerance_s), status="scored")
  return row


def evaluate(data_dir, results, slowdown, tolerance=1.0, chute=None, camera="cam2",
  coverage_end_tolerance_s=0):
  camera = camera_name(camera)
  validate_settings(slowdown, tolerance, coverage_end_tolerance_s)
  if chute and (Path(chute).name != chute or not chute.startswith("chute")):
    raise ValueError("chute must be a chute directory name")
  if not results.is_dir():
    raise ValueError(f"results directory does not exist: {results}")
  directories = ([data_dir / chute] if chute else sorted(data_dir.glob("chute*")))
  directories = [directory for directory in directories if directory.is_dir()]
  if not directories:
    raise ValueError(f"no chute directories found in {data_dir}")
  rows = []
  for directory in directories:
    try:
      rows.append(evaluate_clip(directory, results, slowdown, tolerance, camera,
            coverage_end_tolerance_s))
    except (ValueError, TypeError, AttributeError, KeyError) as error:
      raise ValueError(f"{directory.name}: {error}") from error
  return rows


def aggregate(rows):
  scored = [row for row in rows if row["status"] == "scored"]
  counts = {key: sum(row[key] for row in scored)
       for key in ("gt", "tp", "fp", "fn", "duplicate_fp", "outside_window_fp")}
  delays = [detail["delay_replay_s"] for row in scored for detail in row["details"]
       if detail["kind"] == "TP"]
  return {**counts, **metrics(counts["tp"], counts["fp"], counts["fn"]),
      "scored": len(scored), "expected": len(rows),
      "median_delay_replay_s": statistics.median(delays) if delays else None}


def display(value):
  return "N/A" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def write_report(rows, results, output, slowdown, tolerance, coverage_end_tolerance_s=0):
  summary = aggregate(rows)
  complete = summary["scored"] == summary["expected"]
  lines = ["# Fall Detection Evaluation", "",
      f"Results: `{results}`", "",
      f"Coverage: {summary['scored']}/{summary['expected']} clips scored. "
      + ("Complete." if complete else "INCOMPLETE: metrics describe the scored subset only."),
      f"Slowdown: {slowdown:g}x; playback_speed: {1 / slowdown:g}.",
      f"Matching tolerance: +/-{tolerance:g} source-video seconds.", "",
      f"Coverage end allowance: {coverage_end_tolerance_s:g} source-video seconds "
      "(shortfalls must be strictly smaller; zero is strict).",
      "One alert per episode is TP; extra alerts are FP (duplicates counted separately).",
      "Labels and event timestamps use source-video seconds; they are not rescaled for matching.",
      "Time-only matching assumes at most one fall at a time and cannot verify person identity.",
      "Delays are approximate; replay seconds are not verified real-world seconds.", "",
      "## Coverage", "", "| Status | Clips |", "| --- | ---: |"]
  for status in ("scored", "missing", "failed", "unreviewed"):
    lines.append(f"| {status} | {sum(row['status'] == status for row in rows)} |")
  lines.extend(["", "## Scored Subset", ""])
  for key in ("gt", "tp", "fp", "fn", "duplicate_fp", "outside_window_fp",
        "precision", "recall", "f1", "median_delay_replay_s"):
    lines.append(f"- {key}: {display(summary[key])}")
  columns = ["chute", "status", "gt", "tp", "fp", "fn", "duplicate_fp", "outside_window_fp",
       "precision", "recall", "f1", "median_delay_replay_s"]
  lines.extend(["", "## Per Video", "", "| " + " | ".join(columns) + " |",
         "| " + " | ".join(["---"] * len(columns)) + " |"])
  for row in rows:
    lines.append("| " + " | ".join(display(row.get(key)) for key in columns) + " |")
  lines.extend(["", "## Metric Definitions", "",
      "| Metric | Meaning |", "| --- | --- |",
      "| `gt` | Ground truth: number of manually labeled fall episodes. |",
      "| `tp` | True positives: fall episodes with a matching alert, at most one per episode. |",
      "| `fp` | False positives: unmatched extra alerts, including duplicates. "
      "Equals duplicate_fp + outside_window_fp. |",
      "| `fn` | False negatives: labeled fall episodes with no matching alert. |",
      "| `duplicate_fp` | Extra alerts inside an already matched episode's tolerance window, "
      "with no eligible unmatched episode remaining. Included in fp. |",
      "| `outside_window_fp` | Alerts outside every labeled episode's tolerance window. Included in fp. |",
      "| `precision` | Fraction of alerts credited as detections: tp / (tp + fp). |",
      "| `recall` | Fraction of labeled falls detected: tp / (tp + fn). |",
      "| `f1` | Balance of precision and recall: 2 * tp / (2 * tp + fp + fn). |",
      "| `median_delay_replay_s` | Median delay from labeled fall onset to its matched alert, "
      "in slowed-replay seconds: (alert source time - fall start_s) * slowdown factor. "
      "Uses TP matches only; negative values mean alerts before labeled onset within tolerance. |",
      "", "Precision, recall and F1 range from 0 to 1; higher is better. "
      "N/A means a zero denominator or no matched alerts for delay. "
      "Overall metrics use summed scored-clip counts; overall median delay uses all TP matches."])
  lines.extend(["", "## Review Details", ""])
  for row in rows:
    lines.extend([f"### {row['chute']}", ""])
    provenance = {key: row[key] for key in ("camera", "reference_camera", "cameras",
        "source_video", "source_durations_s", "timing", "source_windows", "per_camera", "source_start_s",
        "source_end_s", "source_time_origin_s", "replay_video_zero_utc", "source_duration_s",
        "replay_duration_s", "label_end_s", "coverage_end_tolerance_s", "coverage_end_shortfall_s",
        "source_end_shortfalls_s", "source_window_end_shortfalls_s") if key in row}
    lines.extend([f"Provenance: `{json.dumps(provenance, sort_keys=True)}`", ""])
    if row["status"] != "scored":
      lines.extend([f"Not scored: {row['status']}. {row.get('reason', '')}", ""])
      continue
    for fall in row["missed"]:
      lines.append(f"- FN: source interval {fall['start_s']:g}-{fall['end_s']:g}s")
    for detail in row["details"]:
      text = (f"- {detail['kind']}: source {detail['video_time_s']:g}s; "
          f"person_id={detail['person_id']}; first_observation={detail['first_observation']}")
      if detail["kind"] == "TP":
        text += (f"; delay_source_s={detail['delay_source_s']:.3f}; "
            f"delay_replay_s={detail['delay_replay_s']:.3f}")
      if detail["ambiguous"]:
        text += "; overlapping tolerance windows: earliest unmatched episode selected"
      lines.append(text)
    if not row["details"] and not row["missed"]:
      lines.append("No falls and no alerts.")
    lines.append("")
  output.mkdir(parents=True, exist_ok=True)
  (output / "report.md").write_text("\n".join(lines) + "\n")
  fields = columns + ["slowdown_factor", "playback_speed", "tolerance_source_s",
            "coverage_end_tolerance_s", "label_end_s", "coverage_end_shortfall_s",
            "source_end_shortfalls_s", "source_window_end_shortfalls_s",
            "source_duration_s", "replay_duration_s", "camera", "reference_camera",
            "cameras", "source_video", "source_durations_s", "timing", "source_windows", "per_camera",
            "source_start_s", "source_end_s", "source_time_origin_s", "replay_video_zero_utc"]
  with (output / "per_video.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
      writer.writerow({key: "N/A" if row.get(key) is None else
          json.dumps(row[key], sort_keys=True) if isinstance(row[key], (dict, list)) else row[key]
          for key in fields})
  return summary


def main():
  base = Path(__file__).resolve().parent
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data-dir", type=Path, default=base / "data")
  parser.add_argument("--results", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--slowdown-factor", type=float, required=True)
  parser.add_argument("--tolerance-s", type=float, default=1.0, help="Source-video seconds (default: 1)")
  parser.add_argument("--coverage-end-tolerance-s", type=float, default=0,
            help="Allow strictly smaller label-end coverage shortfalls in source seconds (default: 0)")
  parser.add_argument("--chute")
  parser.add_argument("--camera", type=camera_name, default="cam2")
  args = parser.parse_args()
  try:
    rows = evaluate(args.data_dir, args.results, args.slowdown_factor, args.tolerance_s, args.chute,
                args.camera, args.coverage_end_tolerance_s)
    summary = write_report(rows, args.results, args.output, args.slowdown_factor, args.tolerance_s,
                       args.coverage_end_tolerance_s)
  except (ValueError, OSError, subprocess.SubprocessError) as error:
    parser.exit(2, f"Evaluation failed: {error}\n")
  print(f"{summary['scored']}/{summary['expected']} clips scored; "
     f"TP={summary['tp']} FP={summary['fp']} FN={summary['fn']}")
  print(f"Report: {args.output / 'report.md'}")
  return 0 if summary["scored"] == summary["expected"] else 1


if __name__ == "__main__":
  raise SystemExit(main())