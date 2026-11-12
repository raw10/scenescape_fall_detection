# YOLO11 Versus SceneScape

YOLO results: `/home/labrat/scenescape_fall_detection/benchmark_v2/results-yolo11-best-conf04-gap025`
SceneScape baseline: `/home/labrat/scenescape_fall_detection/benchmark_v2/results-yolo11s-pose-quarter-speed-20261026T114844Z`

Confidence: 0.400; fall-free gap: 0.250 source seconds from the first fall-free frame; no person tracking.
Comparison slowdown: 4.000x; tolerance: +/-1 source second.
Source timestamps are not rescaled. Replay delays are derived source delays times slowdown, not measured inference latency.

YOLO coverage: 24/24.
SceneScape coverage: 21/24.
Common scored clips: 21/24.

Common clips: chute01, chute02, chute03, chute04, chute05, chute06, chute07, chute08, chute09, chute10, chute11, chute12, chute13, chute14, chute15, chute16, chute17, chute19, chute21, chute23, chute24

## Exclusions

- chute18: YOLO: scored ; SceneScape: failed no_observed_people
- chute20: YOLO: scored ; SceneScape: failed no_observed_people
- chute22: YOLO: scored ; SceneScape: failed no_observed_people

## Common-Clip Totals

| Method | gt | tp | fp | fn | duplicate_fp | outside_window_fp | precision | recall | f1 | median_delay_source_s | median_delay_replay_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| YOLO11 | 26 | 22 | 94 | 4 | 25 | 69 | 0.190 | 0.846 | 0.310 | 0.200 | 0.800 | 
| SceneScape | 26 | 23 | 42 | 3 | 18 | 24 | 0.354 | 0.885 | 0.505 | 1.233 | 4.933 | 

## Paired Clips

| Clip | YOLO TP/FP/FN | SceneScape TP/FP/FN |
| --- | --- | --- |
| chute01 | 1/3/0 | 1/2/0 |
| chute02 | 1/1/0 | 1/0/0 |
| chute03 | 1/2/0 | 1/0/0 |
| chute04 | 1/2/0 | 1/1/0 |
| chute05 | 1/2/0 | 1/2/0 |
| chute06 | 1/3/0 | 1/0/0 |
| chute07 | 2/2/0 | 2/2/0 |
| chute08 | 1/1/0 | 1/0/0 |
| chute09 | 1/2/0 | 1/0/0 |
| chute10 | 1/2/0 | 1/3/0 |
| chute11 | 1/1/0 | 1/0/0 |
| chute12 | 1/2/0 | 1/2/0 |
| chute13 | 2/3/0 | 1/0/1 |
| chute14 | 2/2/0 | 2/1/0 |
| chute15 | 1/3/0 | 0/0/1 |
| chute16 | 1/4/0 | 1/2/0 |
| chute17 | 1/5/0 | 1/8/0 |
| chute19 | 1/5/0 | 1/0/0 |
| chute21 | 0/5/1 | 0/0/1 |
| chute23 | 1/17/3 | 4/7/0 |
| chute24 | 0/27/0 | 0/12/0 |

## Decoder Warnings

Recovered frames were allowed only with matching timestamps and full frame counts. Warnings are saved in each clip's JSON result.
Clips with warnings: chute01, chute02, chute03, chute04, chute05, chute06, chute07, chute08, chute09, chute10, chute11, chute12, chute13, chute14, chute15, chute16, chute17, chute18, chute19, chute20, chute21, chute22, chute23, chute24

## Limits

These are complete-pipeline results, not an isolated classifier or runtime benchmark. YOLO uses clip-global alerts; SceneScape uses person transitions. Clip-global grouping can merge nearby or simultaneous people. Failed runs are not successful zero-alert runs.
Individual report.md files contain source alert timestamps, missed intervals and delay details.