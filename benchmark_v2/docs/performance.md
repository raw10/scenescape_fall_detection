# Fall Detection Performance

For the newer eight-camera experiment, see the
[full multiview benchmark report](performance-multiview.md), which prioritizes
missed falls and compares all views on 20 shared chutes containing 23 labeled
falls. The report below preserves the earlier 21-clip, 26-fall experiment;
its scores must not be mixed with the newer run.

## Summary

SceneScape's geometry- and motion-based fall detection achieved **higher precision,
recall, and F1 than the YOLO11 fall-model baseline** on the 21 commonly scored
clips. It detected **23 of 26 labeled falls**, with **only 3 false negatives**,
and generated **55% fewer false alerts** than YOLO11 (42 versus 94).

These results show the value of combining person detections with calibrated
geometry and tracking, rather than relying on a frame-level fall label alone.
SceneScape achieved the stronger detection-versus-false-alert balance in this
experiment; YOLO11 produced earlier matched alerts.

## Results

Both methods were scored on the same 21 clips and 26 labeled fall episodes.
Counts and ratios below come from the [paired comparison](reports/comparison.md)
and its [CSV summary](reports/comparison.csv).

| Metric | SceneScape Fall Detection | YOLO11 Fall Model |
| --- | ---: | ---: |
| Detected falls (TP) | **23** | 22 |
| Missed falls (FN) | **3** | 4 |
| False alerts (FP) | **42** | 94 |
| Precision | **35.4%** | 19.0% |
| Recall | **88.5%** | 84.6% |
| F1 | **0.505** | 0.310 |
| Median matched-alert delay, source seconds | 1.233 | **0.200** |

SceneScape missed one fewer fall while producing 52 fewer false alerts. Of its
42 false alerts, 18 were additional alerts within already detected fall windows
and 24 were outside all labeled fall windows. YOLO11 produced 25 duplicate and
69 outside-window false alerts. All additional alerts count against precision;
duplicate alerts are not discarded to improve the reported score.

The three SceneScape misses occurred in `chute13`, `chute15`, and `chute21`.
Despite the stronger relative result, 42 false alerts remain a meaningful
limitation: the current configuration is a promising benchmark result, not a
claim of deployment-ready alarm reliability.

See the [SceneScape report](reports/scenescape/report.md) and
[YOLO11 report](reports/yolo/report.md) for individual coverage, per-video counts,
missed intervals, and alert timestamps. The YOLO-only report covers all 24 clips;
use the paired comparison above, not unequal individual subsets, for headline
comparisons.

## Evaluation Setup

- **SceneScape pipeline:** YOLO11s-pose person detections supply bounding boxes
	to SceneScape tracking and the geometry/motion-based fall detector. The fall
	logic uses observed versus projected standing-person boxes and motion
	features; this experiment does not use pose keypoints as fall evidence.
- **YOLO11 baseline:** the supplied `best.pt` predicts `non-fall` and `fall`
	directly from original video frames. Confidence is fixed at 0.4, following
	the model creators' recommended inference setting. One clip-level alert is
	emitted when a fall appears, with rearming after 0.25
	source seconds without a fall detection. No person tracking is used.
- **Timing:** SceneScape replay uses quarter speed (4x slowdown). Labels and
	alerts are matched in original source-video seconds. The dataset videos are
	sped up by a factor of four, so quarter-speed replay compensates for that
	speedup and restores realistic motion timing for the kinematic detector.
	Without this correction, people would appear to move four times faster,
	distorting velocity and rate-of-change features and temporal smoothing.
	The slowdown changes replay intervals and timestamps, not the source videos
	or annotation times. YOLO11 processes frames independently and does not need
	real-time waits; its alerts are evaluated against the same source timeline.
	Reported source delays
	are not inference latency; derived slowed-replay delays are not verified
	real-world response times. Each method's median uses its own matched falls.
- **Scoring:** reviewed fall intervals allow +/-1 source second at both ends.
	Each interval can receive one TP; unmatched intervals are FN, and extra alerts
	are FP. Overall precision, recall, and F1 use summed counts, not averages of
	per-video scores.
- **Coverage:** YOLO11 completed 24 clips. SceneScape had no observed scene
	people in `chute18`, `chute20`, and `chute22` (`no_observed_people`). These
	pipeline failures are excluded from both sides of the paired comparison,
	not treated as classifier misses or successful zero-alert runs. Their root
	cause is not established by the saved reports. They do not change the
	21-clip comparison, but limit its coverage to 21 of 24 clips.
- **Video decoding:** the sources produced decoding warnings. As agreed for
	this experiment, recovered frames were accepted only with matching frame
	counts and timestamps; warnings remain disclosed in the comparison report.

This is an end-to-end comparison of the two configured pipelines, not an
isolated classifier comparison or a CPU/GPU performance benchmark. SceneScape
emits person-level transitions, whereas the simple YOLO baseline groups falls
at clip level. The scoring assumes non-overlapping fall episodes and cannot
verify person identity or evaluate simultaneous falls separately.

## Improvement Opportunities

1. **Review the three misses first.** Inspect person detections, tracking,
	 calibration, and feature traces around each missed episode to distinguish
	 missing input evidence from fall-decision thresholds.
2. **Evaluate motion-adaptive thresholds.** A useful hypothesis is that
	 rate-of-change thresholds could account for estimated motion magnitude,
	 helping slower falls without increasing nuisance alerts. Test this on
	 separate tuning clips and then evaluate with fixed settings; the present
	 results do not establish that velocity adaptation will improve accuracy.
3. **Reduce repeated alerts and nuisance triggers.** Review state stability,
	 rearming, and posture thresholds. The 18 duplicate and 24 outside-window
	 alerts identify distinct opportunities to improve precision while preserving
	 the current high recall.
4. **Expand viewpoint and no-fall coverage.** Test oblique views, occlusion,
	 lying-down people, and everyday activities to assess where the geometry
	 remains discriminative and how often nuisance alerts occur.

## Limitations And Detector Choice

The kinematic detector depends on reliable person boxes and tracking. If the
upstream detector misses a lying-down person, the fall logic has no observed
posture box to evaluate. Calibration quality and tracking errors can also
affect the final result.

At some oblique viewpoints, including views around 45 degrees, standing and
fallen bounding boxes may be too similar for the current posture thresholds.
This is a potential geometric failure mode, not a demonstrated universal angle
limit or a verified explanation for all three misses. Multi-camera evidence
may help, but this single-camera-per-clip benchmark does not establish that
benefit.

For this workflow, **YOLO pose models are a practical starting point for person
detection**, including evaluation of non-upright poses. Validate the selected
model on standing, falling, and lying-down people in the target scene. Retail
person detectors or other general detectors may miss non-upright poses, but
that behavior is model- and scene-dependent, not a property of all non-pose
YOLO models. Using a pose-trained detector here does not mean the fall logic
uses its keypoints.