"""Create a human-readable visual report from multi-view summary records."""

import argparse
import csv
import math
from pathlib import Path

from jinja2 import Environment, StrictUndefined
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch


SCENE = "#007f86"
YOLO = "#bc4a36"
COMBINED = "#35424a"
DETECTED = "#16856b"
MISSED = "#dce3e6"
DUPLICATE = "#dfa72c"
OUTSIDE = "#bc4a36"

TEMPLATE = """# Benchmark At A Glance

{% if not comparable %}
**NOT COMPARABLE:** No shared scored chutes. No performance charts or rankings
are shown; missing and failed conditions are not zero-performance results.

See the [detailed coverage report](summary.md).
{% else %}
## Main Takeaway

{% if gt == 0 %}
There are no labeled falls in the shared subset. Fall-detection recall and
camera rankings cannot be interpreted from this subset.
{% elif combined.fn == fewest_fn %}
**With missed falls as the primary metric, combined cameras have the fewest
observed misses: {{ combined.fn }} of {{ gt }} labeled falls. This comes with
{{ combined.fp }} false alerts, so alert burden remains a separate tradeoff.**
{% else %}
**Missed falls are the primary metric: the lowest observed FN count is
{{ fewest_fn }} of {{ gt }} labeled falls.** Compare all views below; false
alerts and F1 provide secondary context rather than replacing the FN comparison.
{% endif %}

All comparisons below use **{{ scored }} shared chutes out of {{ expected }}
requested, containing {{ gt }} labeled falls**. Each physical fall is counted
once per condition, not once per camera.

## Missed Falls: Primary Metric

![Missed falls by camera and method, with combined cameras shown separately](missed-falls.png)

Lower FN is better. Each label shows missed falls / labeled falls and the
percentage missed (FN / labeled falls, equivalent to 1 minus recall when labels
are present). All bars use the same shared subset. Failed or unscored conditions
are excluded, not treated as zero misses. The combined result is not an average
of the individual cameras.

{% if gt %}
- **Missed-fall comparison:** Combined cameras miss {{ combined.fn }}/{{ gt }}; the lowest observed single-camera FN count is {{ fewest_single_fn }}/{{ gt }}. These are results on this run, not independently validated camera selections.
- **Single-camera comparison:** SceneScape has higher F1 in {{ wins }}/{{ cameras | length }} matched camera views ({{ ties }} ties). Its highest observed single-camera F1 is {{ best_scene.label }} at {{ '%.3f' | format(best_scene.f1) }}; YOLO's is {{ best_yolo.label }} at {{ '%.3f' | format(best_yolo.f1) }}.
- **Combined-camera tradeoff:** {{ combined.tp }}/{{ gt }} falls detected, {{ combined.fn }} missed, and {{ combined.fp }} false alerts. Precision (the share of alerts counted as true positives): {{ combined.precision | percent }}.
{% if combined.fp %}
- **Repeat alerts matter:** {{ combined.duplicate_fp }}/{{ combined.fp }} combined false alerts ({{ '%.1f' | format(100 * combined.duplicate_fp / combined.fp) }}%) are duplicates under the scoring rule; {{ combined.outside_window_fp }} are outside the matching windows. This identifies an investigation priority, not the underlying cause or the benefit of a proposed fix.
{% endif %}
{% endif %}

## Falls Found And Alert Burden

![Falls detected and missed beside the false-alert breakdown](outcomes.png)

The left panel counts physical falls; the right counts false alerts and uses a
different scale. Duplicate alerts repeat a detection within an already-matched
fall window; they are not additional falls. Outside-window alerts did not match
a labeled fall under the timing tolerance.

| Condition | Detected / Labeled | Missed | False Alerts | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{% for record in selected %}
| {{ record.label }} | {{ record.tp }} / {{ record.gt }} | {{ record.fn }} | {{ record.fp }} | {{ record.precision | percent }} | {{ record.recall | percent }} | {{ record.f1 | score }} |
{% endfor %}

Best camera views are selected **retrospectively from this run**, not on an
independent validation set. The same-view YOLO comparison is included alongside
YOLO's best observed view when those differ.

## Camera-To-Camera Comparison

![Paired SceneScape and YOLO F1 scores for every camera](camera-f1.png)

Higher F1 means a better balance between detecting labeled falls and avoiding
false alerts. It is not accuracy or a measure of processing speed. The combined
condition is shown separately and is not an average of the single-camera scores.

## Coverage And Limits

{{ expected - scored }} requested chutes are outside this shared comparison.
An unscored condition may represent a failed run or a label/video-coverage
exclusion; it is **not** counted as an ordinary missed fall. Individual rows in
the detailed report may use more chutes and are therefore not directly
interchangeable with these shared-subset results.

The report uses the saved scoring results unchanged. False-alert counts are not
false alerts per hour, and camera conditions must never be pooled as independent
physical falls. This is a pipeline evaluation, not deployment-ready reliability
evidence.

See [detailed metrics, exclusions, and provenance](summary.md) and
[machine-readable results](summary.csv){% if has_ledger %}, plus the
[conditions queued for retry](../benchmark-failures.json){% endif %}.
{% endif %}
"""


def label(condition):
  method, camera = condition.split("/")
  return f"{'SceneScape' if method == 'scenescape' else 'YOLO'} / {camera}"


def save_figure(figure, path):
  FigureCanvasAgg(figure)
  figure.savefig(path, dpi=160, facecolor="white")
  figure.clear()


def style_axis(axis):
  axis.spines[["top", "right", "left"]].set_visible(False)
  axis.spines["bottom"].set_color("#aebac0")
  axis.tick_params(axis="both", length=0, labelsize=10, pad=8)
  axis.set_axisbelow(True)
  axis.grid(axis="x", color="#e5eaed", linewidth=0.8)


def plot_outcomes(records, output):
  figure = Figure(figsize=(13, 4.8), layout="constrained")
  found, alerts = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.2]})
  positions = list(range(len(records)))
  for axis in (found, alerts):
    style_axis(axis)
    axis.set_ylim(len(records) - 0.4, -0.7)
    axis.set_yticks(positions)
  found.set_yticklabels([record["label"] for record in records])
  alerts.set_yticklabels([])
  found.set_title("Falls detected / missed", loc="left", pad=40, fontweight="bold")
  alerts.set_title("False alerts: repeated / outside window", loc="left", pad=40, fontweight="bold")
  found.set_xlim(0, max(1, records[0]["gt"]) * 1.14)
  alerts.set_xlim(0, max(1, max(record["fp"] for record in records)) * 1.19)
  for position, record in zip(positions, records):
    for axis, segments in ((found, [("tp", DETECTED), ("fn", MISSED)]),
                           (alerts, [("duplicate_fp", DUPLICATE), ("outside_window_fp", OUTSIDE)])):
      start = 0
      for key, color in segments:
        value = record[key]
        axis.barh(position, value, left=start, color=color, height=0.54)
        if value:
          axis.text(start + value / 2, position, str(value), ha="center", va="center",
                    color="white" if color in (DETECTED, OUTSIDE) else "#172b34", fontsize=10)
        start += value
      if axis is alerts:
        axis.text(start + alerts.get_xlim()[1] * 0.025, position, str(start), va="center", fontweight="bold")
  found.set_xlabel("Labeled falls (one count per physical fall)", labelpad=12)
  alerts.set_xlabel("False alerts (different scale from falls)", labelpad=12)
  found.legend(handles=[Patch(color=DETECTED, label="Detected"), Patch(color=MISSED, label="Missed")],
               loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=2)
  alerts.legend(handles=[Patch(color=DUPLICATE, label="Duplicate"), Patch(color=OUTSIDE, label="Outside window")],
                loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=2)
  save_figure(figure, output / "outcomes.png")


def plot_camera_scores(records, cameras, output):
  figure = Figure(figsize=(12, 7), layout="constrained")
  axis = figure.add_subplot()
  style_axis(axis)
  lookup = {record["condition"]: record for record in records}
  for position, camera in enumerate(cameras):
    for method, offset, color in (("scenescape", -0.16, SCENE), ("yolo", 0.16, YOLO)):
      score = lookup[f"{method}/{camera}"]["f1"]
      axis.barh(position + offset, score or 0, height=0.27, color=color)
      axis.text((score or 0) + 0.015, position + offset, "N/A" if score is None else f"{score:.3f}",
                va="center", fontsize=10, color=color)
  combined = lookup["scenescape/combined"]["f1"]
  axis.barh(len(cameras) + 0.4, combined or 0, height=0.45, color=COMBINED)
  axis.text((combined or 0) + 0.015, len(cameras) + 0.4,
            "N/A" if combined is None else f"{combined:.3f}", va="center", fontsize=10)
  axis.axhline(len(cameras) - 0.25, color="#c9d2d7", linewidth=1)
  axis.set_yticks([*range(len(cameras)), len(cameras) + 0.4], [*cameras, "All cameras\nSceneScape only"])
  axis.set_ylim(len(cameras) + 1, -0.9)
  axis.set_xlim(0, 1.08)
  axis.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
  axis.set_xlabel("F1 score on the same scored chutes (higher is better)", labelpad=12)
  axis.set_title("Viewpoint matters: compare methods camera by camera", loc="left", pad=40, fontweight="bold")
  axis.legend(handles=[Patch(color=SCENE, label="SceneScape"), Patch(color=YOLO, label="YOLO")],
              loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False)
  save_figure(figure, output / "camera-f1.png")


def plot_missed_falls(records, cameras, output):
  figure = Figure(figsize=(12, 7), layout="constrained")
  axis = figure.add_subplot()
  style_axis(axis)
  lookup = {record["condition"]: record for record in records}
  gt = records[0]["gt"]
  scale = max(1, gt)
  for position, camera in enumerate([*cameras, "combined"]):
    methods = (("scenescape", 0, COMBINED),) if camera == "combined" else (
        ("scenescape", -0.16, SCENE), ("yolo", 0.16, YOLO))
    for method, offset, color in methods:
      record = lookup[f"{method}/{camera}"]
      row = position + (0.4 if camera == "combined" else offset)
      axis.barh(row, record["fn"], height=0.45 if camera == "combined" else 0.27, color=color)
      rate = f"{100 * record['fn'] / gt:.1f}%" if gt else "N/A"
      axis.text(record["fn"] + scale * 0.015, row, f"{record['fn']} / {gt} ({rate})",
                va="center", fontsize=10, color=color)
  axis.axhline(len(cameras) - 0.25, color="#c9d2d7", linewidth=1)
  axis.set_yticks([*range(len(cameras)), len(cameras) + 0.4], [*cameras, "All cameras\nSceneScape only"])
  axis.set_ylim(len(cameras) + 1, -0.9)
  axis.set_xlim(0, scale * 1.4)
  step = max(1, math.ceil(scale / 6))
  axis.set_xticks(list(range(0, gt + 1, step)))
  axis.set_xlabel("Missed labeled falls (FN): lower is better", labelpad=12)
  axis.set_title("Missed falls: compare every camera on the same labels", loc="left", pad=40, fontweight="bold")
  axis.legend(handles=[Patch(color=SCENE, label="SceneScape"), Patch(color=YOLO, label="YOLO")],
              loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False)
  save_figure(figure, output / "missed-falls.png")


def write_visual_report(records, output):
  output.mkdir(parents=True, exist_ok=True)
  common = [dict(record, label=label(record["condition"])) for record in records if record["subset"] == "common"]
  context = {"comparable": bool(common and common[0]["scored"]),
             "has_ledger": (output.parent / "benchmark-failures.json").is_file()}
  if context["comparable"]:
    for record in common:
      if (record["gt"], record["scored"], record["chutes"]) != (common[0]["gt"], common[0]["scored"], common[0]["chutes"]):
        raise ValueError("Visual comparison requires identical scored chutes and labels")
      if record["tp"] + record["fn"] != record["gt"] or record["duplicate_fp"] + record["outside_window_fp"] != record["fp"]:
        raise ValueError("Inconsistent fall or false-alert counts")
    scene = [record for record in common if record["condition"].startswith("scenescape/cam")]
    yolo = [record for record in common if record["condition"].startswith("yolo/cam")]
    cameras = [record["condition"].split("/")[1] for record in scene]
    best_scene = max(scene, key=lambda record: record["f1"] if record["f1"] is not None else -1)
    best_yolo = max(yolo, key=lambda record: record["f1"] if record["f1"] is not None else -1)
    lookup = {record["condition"]: record for record in common}
    combined = lookup["scenescape/combined"]
    selected_names = dict.fromkeys([best_scene["condition"], best_scene["condition"].replace("scenescape/", "yolo/"),
                                   best_yolo["condition"], combined["condition"]])
    selected = [lookup[name] for name in selected_names]
    wins = sum(lookup[f"scenescape/{camera}"]["f1"] is not None and lookup[f"yolo/{camera}"]["f1"] is not None
               and lookup[f"scenescape/{camera}"]["f1"] > lookup[f"yolo/{camera}"]["f1"] for camera in cameras)
    ties = sum(lookup[f"scenescape/{camera}"]["f1"] is not None
               and lookup[f"scenescape/{camera}"]["f1"] == lookup[f"yolo/{camera}"]["f1"] for camera in cameras)
    context.update(gt=common[0]["gt"], scored=common[0]["scored"],
                   expected=max(record["expected"] for record in records if record["subset"] == "individual"),
                   cameras=cameras, best_scene=best_scene, best_yolo=best_yolo, combined=combined,
                   selected=selected, wins=wins, ties=ties,
                   fewest_fn=min(record["fn"] for record in common),
                   fewest_single_fn=min(record["fn"] for record in scene + yolo))
    plot_missed_falls(common, cameras, output)
    plot_outcomes(selected, output)
    plot_camera_scores(common, cameras, output)
  environment = Environment(undefined=StrictUndefined, autoescape=False,
      trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
  environment.filters["percent"] = lambda value: "N/A" if value is None else f"{100 * value:.1f}%"
  environment.filters["score"] = lambda value: "N/A" if value is None else f"{value:.3f}"
  (output / "overview.md").write_text(environment.from_string(TEMPLATE).render(**context))


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--summary-csv", type=Path, required=True)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args(argv)
  try:
    with args.summary_csv.open(newline="") as handle:
      records = list(csv.DictReader(handle))
    for record in records:
      for key in ("gt", "tp", "fp", "fn", "duplicate_fp", "outside_window_fp", "scored", "expected"):
        record[key] = None if record[key] == "N/A" else int(record[key])
      for key in ("precision", "recall", "f1"):
        record[key] = None if record[key] == "N/A" else float(record[key])
        if record[key] is not None and not math.isfinite(record[key]):
          raise ValueError(f"Non-finite {key}")
    output = args.output or args.summary_csv.parent
    write_visual_report(records, output)
  except (ValueError, KeyError, OSError) as error:
    parser.exit(2, f"Visual report failed: {error}\n")
  print(f"Visual report: {output / 'overview.md'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())