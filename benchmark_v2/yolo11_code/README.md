# YOLO11 Fall Baseline 

https://github.com/SyedBurhanAhmed/Real-Time-Fall-Detection-using-YOLO

Run the local `best.pt` on the original benchmark videos and compare its alerts
against the saved SceneScape experiment. No Docker or SceneScape replay is needed.

## Run

From the repository root, with Python 3.10+ and FFmpeg/ffprobe installed:

```sh
python3 -m pip install -r benchmark_v2/yolo11_code/requirements.txt
python3 benchmark_v2/yolo11_code/main.py --chute chute01 \
    --output benchmark_v2/results-yolo11-best-smoke
python3 benchmark_v2/yolo11_code/main.py
```

The default full-run output is `benchmark_v2/results-yolo11-best-conf04-gap025`.
Inference requires a fresh output directory and never overwrites an experiment.
Use `--output` to keep multiple runs. Defaults resolve relative to the script,
so running from another directory also works.

Defaults: confidence **0.4**, fall-free gap **0.25 source seconds**, CPU inference,
and checkpoint image size **640**. Override with `--conf`,
`--fall-free-gap-s`, `--weights`, or `--data-dir`. Inference always uses CPU.
The local model must be a detector with classes `{0: 'non-fall', 1: 'fall'}`;
unsupported tasks stop instead of guessing how to crop inputs.

The selected SceneScape baseline is
`benchmark_v2/results-yolo11s-pose-quarter-speed-20261026T114844Z`.
Use `--scenescape-results` to compare with a different saved experiment. Its
selected results must have one consistent playback speed between zero and one.
The evaluator's matching tolerance remains +/-1 source-video second.

To rebuild reports without running the model again:

```sh
python3 benchmark_v2/yolo11_code/main.py --evaluate-only \
    --output benchmark_v2/results-yolo11-best-conf04-gap025
```

Include `--chute chute01` when re-evaluating a single-clip smoke directory. You
can also use the existing `benchmark_v2/evaluate.py` CLI on the saved clip JSONs.
Re-evaluation uses saved confidence/gap settings, not new inference CLI values.
The runner's dependencies must be installed for both inference and re-evaluation.

## Outputs

- `run.json`: weights SHA-256, class mapping, software versions, inference settings
  and baseline provenance. Effective predictor settings also appear per clip.
- `<chute>.json`: completion/failure status, source-time alerts, frame counts,
  settings and decoder warnings/errors.
- `<chute>/predictions.jsonl`: source timestamp and retained class/confidence/box
  predictions for every processed frame, including empty frames.
- `reports/yolo/` and `reports/scenescape/`: existing evaluator's `report.md` and
  `per_video.csv`, including coverage, missed episodes and alert details.
- `reports/comparison.md` and `comparison.csv`: totals on identical commonly
  scored clips, with per-clip differences and exclusions in the Markdown report.
  The Markdown layout is defined in `comparison.md.j2` using Jinja2.

Exit status is **0** for complete paired coverage, **1** for incomplete/failed
coverage, and **2** for invalid input/setup. The chosen baseline has failed runs
on chute18, chute20 and chute22, so a full comparison normally exits **1** even
when all YOLO clips succeed. Inspect the reports, not the exit code alone.

## Alert And Timing Rules

Every original frame is processed independently; no saved pose/person detections,
tracking, frame skipping, video rewriting or real-time waits are used. A frame is
fallen if at least one class-1 box survives model NMS and confidence filtering.
Non-fall boxes do not override a fall box. Emit immediately when armed; latch
until a continuous fall-free gap reaches 0.25 seconds, measured from the first
frame without an accepted fall. Empty detections also count as fall-free. A
returning fall exactly at the boundary can emit another alert. A fall in the
first decoded frame gets `first_observation: true`. No person ID is invented.

Timestamps come from ffprobe presentation timestamps minus the source file's
playback origin and are paired with sequential decoded frames. Missing frames,
non-increasing timestamps or inconsistent frame counts fail the clip. As agreed,
decoder-concealed damaged frames are accepted when counts/timestamps match;
ffprobe warnings remain visible in results and comparison reports. Inputs are
never repaired or modified. Fully processed clips with zero boxes are valid
zero-alert results; unreadable or partially processed clips are not.

Saved `playback_speed: 0.25` describes the selected comparison's **virtual 4x
slowed timeline**, not measured inference speed. Alerts remain in **source
seconds** and are never scaled twice. The 0.25-source-second gap equals 1 second
on that slowed timeline. Derived replay delays are not inference latency.

## Comparison Limits

Individual reports expose all failed, missing and unreviewed clips. Headline
paired metrics use only the intersection of successfully scored clips and are
computed from summed TP/FP/FN counts, not average clip scores. Failures are not
converted into successful zero-alert runs. No common clips means not comparable.

This compares complete pipelines, not isolated fall-classifier accuracy or
runtime performance. YOLO emits clip-global alerts while SceneScape emits
person transitions; nearby or simultaneous people can be merged by this simple
baseline. Existing evaluation cannot associate simultaneous falls with people.
Thresholds are fixed, not tuned against the evaluation annotations.

## Tests

```sh
python3 -m unittest discover -s benchmark_v2/tests
```

YOLO tests mock video/model dependencies and do not load weights or require a GPU.