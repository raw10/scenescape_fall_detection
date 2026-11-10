"""Summarize saved per-view and combined results without running inference."""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from jinja2 import Environment, StrictUndefined

if __package__ in (None, ""):
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_v2 import evaluate as evaluation
from benchmark_v2.cameras import camera_name
from benchmark_v2.visualize_views import write_visual_report


TEMPLATE = """# Multi-View Evaluation

Start with the [visual overview and main takeaways](overview.md).

Results: `{{ results }}`
Requested chutes: {{ chutes | join(', ') }}.
Selected cameras: {{ cameras | join(', ') }}.
Common scored chutes: {{ common | join(', ') or 'none; NOT COMPARABLE' }}.

Each condition counts each physical chute label once. Views are descriptive conditions,
not independent labels; counts are never pooled across views or methods.
Matching uses source seconds; replay delays use each result's recorded playback speed.

## Coverage And Metrics

| Subset | Condition | Scored/Expected | Missing | Failed | Unreviewed | GT | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{% for record in records %}
| {{ record.subset }} | {{ record.condition }} | {{ record.scored }}/{{ record.expected }} | {{ record.missing }} | {{ record.failed }} | {{ record.unreviewed }} | {{ record.gt | display }} | {{ record.tp | display }} | {{ record.fp | display }} | {{ record.fn | display }} | {{ record.precision | display }} | {{ record.recall | display }} | {{ record.f1 | display }} |
{% endfor %}

## Common-Set Exclusions

{% for chute in excluded %}
- {{ chute }}:{% for condition, rows in conditions.items() %}{% if rows[chute].status != 'scored' %} {{ condition }}={{ rows[chute].status }} ({{ rows[chute].get('reason', '') }});{% endif %}{% endfor %}
{% else %}
None.
{% endfor %}

## Per-Chute Provenance

{% for condition, rows in conditions.items() %}
{% for chute, row in rows.items() %}
- {{ condition }}/{{ chute }}: {{ row.status }}; reference={{ row.get('reference_camera', 'not recorded') }}; cameras={{ row.get('cameras', []) | join(', ') }}; playback_speed={{ row.get('playback_speed') | display }}.
  Provenance: `{{ row | tojson }}`
{% endfor %}
{% endfor %}
"""


def summarize(data_dir, results, chutes, cameras=None, coverage_end_tolerance_s=0):
  evaluation.validate_settings(4, 1, coverage_end_tolerance_s)
  if not chutes or len(set(chutes)) != len(chutes):
    raise ValueError("chutes must be a nonempty unique list")
  if any(Path(chute).name != chute or not chute.startswith("chute") for chute in chutes):
    raise ValueError("chutes must be chute directory names")
  if not results.is_dir():
    raise ValueError(f"results root does not exist: {results}")
  if cameras is None:
    cameras = sorted({path.name for method in ("scenescape", "yolo")
        for path in (results / method).glob("cam*") if path.is_dir()},
        key=lambda name: int(camera_name(name)[3:]))
  if not cameras or len(set(cameras)) != len(cameras):
    raise ValueError("cameras must be a nonempty unique selection")
  cameras = [camera_name(name) for name in cameras]
  conditions = {}
  selections = [(method, camera) for method in ("scenescape", "yolo") for camera in cameras]
  selections.append(("scenescape", "combined"))
  for method, view in selections:
    directory = results / method / view
    rows = {}
    for chute in chutes:
      row = {"chute": chute, "status": "missing", "reason": "result file not found"}
      path = directory / f"{chute}.json"
      try:
        if path.is_file():
          result = evaluation.load_json(path)
          reference = result.get("reference_camera") if view == "combined" else view
          selected = evaluation.result_cameras(result, reference)
          if view == "combined":
            if set(selected) != set(cameras):
              raise ValueError("combined result cameras differ from selected cameras")
          elif selected != [view]:
            raise ValueError("individual condition contains combined-camera results")
          speed = evaluation.number(result.get("playback_speed"), "result playback_speed")
          if not 0 < speed < 1:
            raise ValueError("result playback_speed must be between 0 and 1")
          row = evaluation.evaluate(data_dir, directory, 1 / speed, chute=chute,
                                    camera=reference,
                                    coverage_end_tolerance_s=coverage_end_tolerance_s)[0]
        elif directory.is_dir() and view != "combined":
          row = evaluation.evaluate(data_dir, directory, 4, chute=chute, camera=view,
                                    coverage_end_tolerance_s=coverage_end_tolerance_s)[0]
          row.pop("playback_speed", None)
          row.pop("slowdown_factor", None)
          row.pop("replay_duration_s", None)
      except (ValueError, TypeError, AttributeError, KeyError, OSError,
              subprocess.SubprocessError) as error:
        row.update(status="failed", reason=str(error))
      rows[chute] = row
    conditions[f"{method}/{view}"] = rows
  common = sorted(chute for chute in chutes
          if all(rows[chute]["status"] == "scored" for rows in conditions.values()))
  records = []
  for condition, rows in conditions.items():
    for subset, names in (("individual", chutes), ("common", common)):
      selected_rows = [rows[name] for name in names]
      summary = evaluation.aggregate(selected_rows)
      if not summary["scored"]:
        for key in ("gt", "tp", "fp", "fn", "duplicate_fp", "outside_window_fp"):
          summary[key] = None
      records.append({"condition": condition, "subset": subset, **summary,
          **{status: sum(row["status"] == status for row in selected_rows)
             for status in ("missing", "failed", "unreviewed")},
          "chutes": ",".join(name for name in names if rows[name]["status"] == "scored")})
  return {"results": str(results), "chutes": chutes, "cameras": cameras,
      "conditions": conditions, "common": common,
      "excluded": sorted(set(chutes) - set(common)), "records": records}


def write_summary(summary, output):
  output.mkdir(parents=True, exist_ok=True)
  environment = Environment(undefined=StrictUndefined, autoescape=False,
      trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
  environment.filters["display"] = evaluation.display
  (output / "summary.md").write_text(environment.from_string(TEMPLATE).render(**summary))
  fields = list(summary["records"][0])
  with (output / "summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for record in summary["records"]:
      writer.writerow({key: "N/A" if value is None else value for key, value in record.items()})
  write_visual_report(summary["records"], output)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data-dir", type=Path, required=True)
  parser.add_argument("--results", type=Path, required=True)
  parser.add_argument("--chutes", nargs="+", default=["chute01", "chute02", "chute03"])
  parser.add_argument("--cameras", nargs="+", type=camera_name)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--coverage-end-tolerance-s", type=float, default=0)
  args = parser.parse_args(argv)
  try:
    summary = summarize(args.data_dir, args.results, args.chutes, args.cameras,
                        coverage_end_tolerance_s=args.coverage_end_tolerance_s)
    output = args.output or args.results / "reports"
    write_summary(summary, output)
  except (ValueError, OSError, subprocess.SubprocessError) as error:
    parser.exit(2, f"Summary failed: {error}\n")
  print(f"Common scored chutes: {', '.join(summary['common']) or 'none'}; {output / 'summary.md'}")
  return 0 if len(summary["common"]) == len(args.chutes) else 1


if __name__ == "__main__":
  raise SystemExit(main())