# Fall Detection Evaluation

Results: `/home/labrat/scenescape_fall_detection/benchmark_v2/results-yolo11-best-conf04-gap025`

Coverage: 24/24 clips scored. Complete.
Slowdown: 4x; playback_speed: 0.25.
Matching tolerance: +/-1 source-video seconds.

One alert per episode is TP; extra alerts are FP (duplicates counted separately).
Labels and event timestamps use source-video seconds; they are not rescaled for matching.
Time-only matching assumes at most one fall at a time and cannot verify person identity.
Delays are approximate; replay seconds are not verified real-world seconds.

## Coverage

| Status | Clips |
| --- | ---: |
| scored | 24 |
| missing | 0 |
| failed | 0 |
| unreviewed | 0 |

## Scored Subset

- gt: 29
- tp: 25
- fp: 110
- fn: 4
- duplicate_fp: 27
- outside_window_fp: 83
- precision: 0.185
- recall: 0.862
- f1: 0.305
- median_delay_replay_s: 0.767

## Per Video

| chute | status | gt | tp | fp | fn | duplicate_fp | outside_window_fp | precision | recall | f1 | median_delay_replay_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chute01 | scored | 1 | 1 | 3 | 0 | 2 | 1 | 0.250 | 1.000 | 0.400 | 0.767 |
| chute02 | scored | 1 | 1 | 1 | 0 | 1 | 0 | 0.500 | 1.000 | 0.667 | 0.100 |
| chute03 | scored | 1 | 1 | 2 | 0 | 0 | 2 | 0.333 | 1.000 | 0.500 | 0.967 |
| chute04 | scored | 1 | 1 | 2 | 0 | 2 | 0 | 0.333 | 1.000 | 0.500 | -1.200 |
| chute05 | scored | 1 | 1 | 2 | 0 | 2 | 0 | 0.333 | 1.000 | 0.500 | 3.700 |
| chute06 | scored | 1 | 1 | 3 | 0 | 0 | 3 | 0.250 | 1.000 | 0.400 | 4.033 |
| chute07 | scored | 2 | 2 | 2 | 0 | 2 | 0 | 0.500 | 1.000 | 0.667 | 0.533 |
| chute08 | scored | 1 | 1 | 1 | 0 | 1 | 0 | 0.500 | 1.000 | 0.667 | -3.967 |
| chute09 | scored | 1 | 1 | 2 | 0 | 0 | 2 | 0.333 | 1.000 | 0.500 | 1.633 |
| chute10 | scored | 1 | 1 | 2 | 0 | 1 | 1 | 0.333 | 1.000 | 0.500 | 7.833 |
| chute11 | scored | 1 | 1 | 1 | 0 | 1 | 0 | 0.500 | 1.000 | 0.667 | 1.967 |
| chute12 | scored | 1 | 1 | 2 | 0 | 0 | 2 | 0.333 | 1.000 | 0.500 | 2.000 |
| chute13 | scored | 2 | 2 | 3 | 0 | 3 | 0 | 0.400 | 1.000 | 0.571 | 0.483 |
| chute14 | scored | 2 | 2 | 2 | 0 | 0 | 2 | 0.500 | 1.000 | 0.667 | 0.783 |
| chute15 | scored | 1 | 1 | 3 | 0 | 1 | 2 | 0.250 | 1.000 | 0.400 | -2.267 |
| chute16 | scored | 1 | 1 | 4 | 0 | 1 | 3 | 0.200 | 1.000 | 0.333 | -0.267 |
| chute17 | scored | 1 | 1 | 5 | 0 | 2 | 3 | 0.167 | 1.000 | 0.286 | 1.500 |
| chute18 | scored | 1 | 1 | 2 | 0 | 0 | 2 | 0.333 | 1.000 | 0.500 | 3.633 |
| chute19 | scored | 1 | 1 | 5 | 0 | 2 | 3 | 0.167 | 1.000 | 0.286 | -2.433 |
| chute20 | scored | 1 | 1 | 7 | 0 | 2 | 5 | 0.125 | 1.000 | 0.222 | -0.633 |
| chute21 | scored | 1 | 0 | 5 | 1 | 0 | 5 | 0.000 | 0.000 | 0.000 | N/A |
| chute22 | scored | 1 | 1 | 7 | 0 | 0 | 7 | 0.125 | 1.000 | 0.222 | -1.600 |
| chute23 | scored | 4 | 1 | 17 | 3 | 4 | 13 | 0.056 | 0.250 | 0.091 | -3.900 |
| chute24 | scored | 0 | 0 | 27 | 0 | 0 | 27 | 0.000 | N/A | 0.000 | N/A |

## Metric Definitions

| Metric | Meaning |
| --- | --- |
| `gt` | Ground truth: number of manually labeled fall episodes. |
| `tp` | True positives: fall episodes with a matching alert, at most one per episode. |
| `fp` | False positives: unmatched extra alerts, including duplicates. Equals duplicate_fp + outside_window_fp. |
| `fn` | False negatives: labeled fall episodes with no matching alert. |
| `duplicate_fp` | Extra alerts inside an already matched episode's tolerance window, with no eligible unmatched episode remaining. Included in fp. |
| `outside_window_fp` | Alerts outside every labeled episode's tolerance window. Included in fp. |
| `precision` | Fraction of alerts credited as detections: tp / (tp + fp). |
| `recall` | Fraction of labeled falls detected: tp / (tp + fn). |
| `f1` | Balance of precision and recall: 2 * tp / (2 * tp + fp + fn). |
| `median_delay_replay_s` | Median delay from labeled fall onset to its matched alert, in slowed-replay seconds: (alert source time - fall start_s) * slowdown factor. Uses TP matches only; negative values mean alerts before labeled onset within tolerance. |

Precision, recall and F1 range from 0 to 1; higher is better. N/A means a zero denominator or no matched alerts for delay. Overall metrics use summed scored-clip counts; overall median delay uses all TP matches.

## Review Details

### chute01

- outside-window FP: source 7.44167s; person_id=; first_observation=False
- TP: source 9.19167s; person_id=; first_observation=False; delay_source_s=0.192; delay_replay_s=0.767
- duplicate FP: source 12.4667s; person_id=; first_observation=False
- duplicate FP: source 12.95s; person_id=; first_observation=False

### chute02

- TP: source 3.025s; person_id=; first_observation=False; delay_source_s=0.025; delay_replay_s=0.100
- duplicate FP: source 7.01667s; person_id=; first_observation=False

### chute03

- outside-window FP: source 0.05s; person_id=; first_observation=False
- outside-window FP: source 3.575s; person_id=; first_observation=False
- TP: source 5.24167s; person_id=; first_observation=False; delay_source_s=0.242; delay_replay_s=0.967

### chute04

- TP: source 2.7s; person_id=; first_observation=False; delay_source_s=-0.300; delay_replay_s=-1.200
- duplicate FP: source 3.575s; person_id=; first_observation=False
- duplicate FP: source 8.20833s; person_id=; first_observation=False

### chute05

- TP: source 2.925s; person_id=; first_observation=False; delay_source_s=0.925; delay_replay_s=3.700
- duplicate FP: source 3.26667s; person_id=; first_observation=False
- duplicate FP: source 5.19167s; person_id=; first_observation=False

### chute06

- outside-window FP: source 0.066667s; person_id=; first_observation=False
- outside-window FP: source 0.358333s; person_id=; first_observation=False
- outside-window FP: source 3.75s; person_id=; first_observation=False
- TP: source 6.00833s; person_id=; first_observation=False; delay_source_s=1.008; delay_replay_s=4.033

### chute07

- TP: source 3.05833s; person_id=; first_observation=False; delay_source_s=0.058; delay_replay_s=0.233
- duplicate FP: source 3.6s; person_id=; first_observation=False
- TP: source 6.20833s; person_id=; first_observation=False; delay_source_s=0.208; delay_replay_s=0.833; overlapping tolerance windows: earliest unmatched episode selected
- duplicate FP: source 7.49167s; person_id=; first_observation=False

### chute08

- TP: source 2.00833s; person_id=; first_observation=False; delay_source_s=-0.992; delay_replay_s=-3.967
- duplicate FP: source 5.55833s; person_id=; first_observation=False

### chute09

- outside-window FP: source 0.025s; person_id=; first_observation=False
- outside-window FP: source 3.00833s; person_id=; first_observation=False
- TP: source 5.40833s; person_id=; first_observation=False; delay_source_s=0.408; delay_replay_s=1.633

### chute10

- outside-window FP: source 2.9s; person_id=; first_observation=False
- TP: source 5.95833s; person_id=; first_observation=False; delay_source_s=1.958; delay_replay_s=7.833
- duplicate FP: source 6.64167s; person_id=; first_observation=False

### chute11

- TP: source 3.49167s; person_id=; first_observation=False; delay_source_s=0.492; delay_replay_s=1.967
- duplicate FP: source 6.65833s; person_id=; first_observation=False

### chute12

- outside-window FP: source 3.40833s; person_id=; first_observation=False
- TP: source 5.5s; person_id=; first_observation=False; delay_source_s=0.500; delay_replay_s=2.000
- outside-window FP: source 7.21667s; person_id=; first_observation=False

### chute13

- TP: source 2.83333s; person_id=; first_observation=False; delay_source_s=-0.167; delay_replay_s=-0.667
- duplicate FP: source 4.29167s; person_id=; first_observation=False
- TP: source 7.40833s; person_id=; first_observation=False; delay_source_s=0.408; delay_replay_s=1.633
- duplicate FP: source 9.31667s; person_id=; first_observation=False
- duplicate FP: source 9.91667s; person_id=; first_observation=False

### chute14

- TP: source 3.65833s; person_id=; first_observation=False; delay_source_s=-0.342; delay_replay_s=-1.367
- TP: source 8.73333s; person_id=; first_observation=False; delay_source_s=0.733; delay_replay_s=2.933; overlapping tolerance windows: earliest unmatched episode selected
- outside-window FP: source 11.0833s; person_id=; first_observation=False
- outside-window FP: source 12.3s; person_id=; first_observation=False

### chute15

- outside-window FP: source 3.33333s; person_id=; first_observation=False
- outside-window FP: source 4.375s; person_id=; first_observation=False
- TP: source 5.43333s; person_id=; first_observation=False; delay_source_s=-0.567; delay_replay_s=-2.267
- duplicate FP: source 8.41667s; person_id=; first_observation=False

### chute16

- outside-window FP: source 3.55s; person_id=; first_observation=False
- outside-window FP: source 5.15833s; person_id=; first_observation=False
- outside-window FP: source 6.7s; person_id=; first_observation=False
- TP: source 7.93333s; person_id=; first_observation=False; delay_source_s=-0.067; delay_replay_s=-0.267
- duplicate FP: source 9.33333s; person_id=; first_observation=False

### chute17

- outside-window FP: source 2.3s; person_id=; first_observation=False
- outside-window FP: source 2.89167s; person_id=; first_observation=False
- TP: source 6.375s; person_id=; first_observation=False; delay_source_s=0.375; delay_replay_s=1.500
- duplicate FP: source 7.55833s; person_id=; first_observation=False
- duplicate FP: source 8.16667s; person_id=; first_observation=False
- outside-window FP: source 9.125s; person_id=; first_observation=False

### chute18

- outside-window FP: source 3.125s; person_id=; first_observation=False
- outside-window FP: source 3.59167s; person_id=; first_observation=False
- TP: source 5.90833s; person_id=; first_observation=False; delay_source_s=0.908; delay_replay_s=3.633

### chute19

- outside-window FP: source 0.316667s; person_id=; first_observation=False
- outside-window FP: source 1.21667s; person_id=; first_observation=False
- outside-window FP: source 2.425s; person_id=; first_observation=False
- TP: source 3.39167s; person_id=; first_observation=False; delay_source_s=-0.608; delay_replay_s=-2.433
- duplicate FP: source 4s; person_id=; first_observation=False
- duplicate FP: source 4.26667s; person_id=; first_observation=False

### chute20

- outside-window FP: source 0.033333s; person_id=; first_observation=False
- outside-window FP: source 0.591667s; person_id=; first_observation=False
- outside-window FP: source 1.275s; person_id=; first_observation=False
- outside-window FP: source 1.875s; person_id=; first_observation=False
- outside-window FP: source 2.58333s; person_id=; first_observation=False
- TP: source 3.84167s; person_id=; first_observation=False; delay_source_s=-0.158; delay_replay_s=-0.633
- duplicate FP: source 4.38333s; person_id=; first_observation=False
- duplicate FP: source 4.85833s; person_id=; first_observation=False

### chute21

- FN: source interval 7-9s
- outside-window FP: source 0.466667s; person_id=; first_observation=False
- outside-window FP: source 2.65s; person_id=; first_observation=False
- outside-window FP: source 3.13333s; person_id=; first_observation=False
- outside-window FP: source 3.41667s; person_id=; first_observation=False
- outside-window FP: source 3.79167s; person_id=; first_observation=False

### chute22

- outside-window FP: source 0.133333s; person_id=; first_observation=False
- outside-window FP: source 0.883333s; person_id=; first_observation=False
- outside-window FP: source 1.38333s; person_id=; first_observation=False
- outside-window FP: source 1.96667s; person_id=; first_observation=False
- outside-window FP: source 3.3s; person_id=; first_observation=False
- outside-window FP: source 4s; person_id=; first_observation=False
- outside-window FP: source 4.5s; person_id=; first_observation=False
- TP: source 6.6s; person_id=; first_observation=False; delay_source_s=-0.400; delay_replay_s=-1.600

### chute23

- FN: source interval 22-25s
- FN: source interval 26-28s
- FN: source interval 29-31s
- outside-window FP: source 1.6s; person_id=; first_observation=False
- outside-window FP: source 3.5s; person_id=; first_observation=False
- outside-window FP: source 4.025s; person_id=; first_observation=False
- outside-window FP: source 4.39167s; person_id=; first_observation=False
- outside-window FP: source 8.64167s; person_id=; first_observation=False
- outside-window FP: source 9.40833s; person_id=; first_observation=False
- outside-window FP: source 10.3417s; person_id=; first_observation=False
- TP: source 11.025s; person_id=; first_observation=False; delay_source_s=-0.975; delay_replay_s=-3.900
- duplicate FP: source 12.0667s; person_id=; first_observation=False
- duplicate FP: source 12.4667s; person_id=; first_observation=False
- duplicate FP: source 12.9417s; person_id=; first_observation=False
- duplicate FP: source 13.75s; person_id=; first_observation=False
- outside-window FP: source 15.7417s; person_id=; first_observation=False
- outside-window FP: source 18.175s; person_id=; first_observation=False
- outside-window FP: source 39.6s; person_id=; first_observation=False
- outside-window FP: source 40.95s; person_id=; first_observation=False
- outside-window FP: source 41.4583s; person_id=; first_observation=False
- outside-window FP: source 42.6s; person_id=; first_observation=False

### chute24

- outside-window FP: source 0.008333s; person_id=; first_observation=False
- outside-window FP: source 0.7s; person_id=; first_observation=False
- outside-window FP: source 3.15833s; person_id=; first_observation=False
- outside-window FP: source 4.90833s; person_id=; first_observation=False
- outside-window FP: source 5.575s; person_id=; first_observation=False
- outside-window FP: source 6.425s; person_id=; first_observation=False
- outside-window FP: source 6.725s; person_id=; first_observation=False
- outside-window FP: source 7.49167s; person_id=; first_observation=False
- outside-window FP: source 8.00833s; person_id=; first_observation=False
- outside-window FP: source 9.6s; person_id=; first_observation=False
- outside-window FP: source 10.1167s; person_id=; first_observation=False
- outside-window FP: source 10.875s; person_id=; first_observation=False
- outside-window FP: source 11.4167s; person_id=; first_observation=False
- outside-window FP: source 11.9917s; person_id=; first_observation=False
- outside-window FP: source 12.7417s; person_id=; first_observation=False
- outside-window FP: source 13.0583s; person_id=; first_observation=False
- outside-window FP: source 14.675s; person_id=; first_observation=False
- outside-window FP: source 17.3667s; person_id=; first_observation=False
- outside-window FP: source 17.7417s; person_id=; first_observation=False
- outside-window FP: source 19.4s; person_id=; first_observation=False
- outside-window FP: source 20.4333s; person_id=; first_observation=False
- outside-window FP: source 20.85s; person_id=; first_observation=False
- outside-window FP: source 21.5833s; person_id=; first_observation=False
- outside-window FP: source 24.125s; person_id=; first_observation=False
- outside-window FP: source 26.1667s; person_id=; first_observation=False
- outside-window FP: source 28.1583s; person_id=; first_observation=False
- outside-window FP: source 29.6917s; person_id=; first_observation=False

