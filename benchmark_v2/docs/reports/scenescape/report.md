# Fall Detection Evaluation

Results: `/home/labrat/scenescape_fall_detection/benchmark_v2/results-yolo11s-pose-quarter-speed-20261026T114844Z`

Coverage: 21/24 clips scored. INCOMPLETE: metrics describe the scored subset only.
Slowdown: 4x; playback_speed: 0.25.
Matching tolerance: +/-1 source-video seconds.

One alert per episode is TP; extra alerts are FP (duplicates counted separately).
Labels and event timestamps use source-video seconds; they are not rescaled for matching.
Time-only matching assumes at most one fall at a time and cannot verify person identity.
Delays are approximate; replay seconds are not verified real-world seconds.

## Coverage

| Status | Clips |
| --- | ---: |
| scored | 21 |
| missing | 0 |
| failed | 3 |
| unreviewed | 0 |

## Scored Subset

- gt: 26
- tp: 23
- fp: 42
- fn: 3
- duplicate_fp: 18
- outside_window_fp: 24
- precision: 0.354
- recall: 0.885
- f1: 0.505
- median_delay_replay_s: 4.933

## Per Video

| chute | status | gt | tp | fp | fn | duplicate_fp | outside_window_fp | precision | recall | f1 | median_delay_replay_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chute01 | scored | 1 | 1 | 2 | 0 | 2 | 0 | 0.333 | 1.000 | 0.500 | 0.833 |
| chute02 | scored | 1 | 1 | 0 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 6.167 |
| chute03 | scored | 1 | 1 | 0 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 5.233 |
| chute04 | scored | 1 | 1 | 1 | 0 | 1 | 0 | 0.500 | 1.000 | 0.667 | 4.133 |
| chute05 | scored | 1 | 1 | 2 | 0 | 1 | 1 | 0.333 | 1.000 | 0.500 | 9.533 |
| chute06 | scored | 1 | 1 | 0 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 7.533 |
| chute07 | scored | 2 | 2 | 2 | 0 | 2 | 0 | 0.500 | 1.000 | 0.667 | 4.533 |
| chute08 | scored | 1 | 1 | 0 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 8.700 |
| chute09 | scored | 1 | 1 | 0 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 4.700 |
| chute10 | scored | 1 | 1 | 3 | 0 | 2 | 1 | 0.250 | 1.000 | 0.400 | 2.833 |
| chute11 | scored | 1 | 1 | 0 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 10.567 |
| chute12 | scored | 1 | 1 | 2 | 0 | 2 | 0 | 0.333 | 1.000 | 0.500 | 1.733 |
| chute13 | scored | 2 | 1 | 0 | 1 | 0 | 0 | 1.000 | 0.500 | 0.667 | 4.367 |
| chute14 | scored | 2 | 2 | 1 | 0 | 1 | 0 | 0.667 | 1.000 | 0.800 | 9.650 |
| chute15 | scored | 1 | 0 | 0 | 1 | 0 | 0 | N/A | 0.000 | 0.000 | N/A |
| chute16 | scored | 1 | 1 | 2 | 0 | 2 | 0 | 0.333 | 1.000 | 0.500 | 3.133 |
| chute17 | scored | 1 | 1 | 8 | 0 | 3 | 5 | 0.111 | 1.000 | 0.200 | 7.733 |
| chute18 | failed | 1 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| chute19 | scored | 1 | 1 | 0 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 5.300 |
| chute20 | failed | 1 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| chute21 | scored | 1 | 0 | 0 | 1 | 0 | 0 | N/A | 0.000 | 0.000 | N/A |
| chute22 | failed | 1 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| chute23 | scored | 4 | 4 | 7 | 0 | 2 | 5 | 0.364 | 1.000 | 0.533 | 3.733 |
| chute24 | scored | 0 | 0 | 12 | 0 | 0 | 12 | 0.000 | N/A | 0.000 | N/A |

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

- TP: source 9.20833s; person_id=a0a2c656-0bae-4b9f-83d9-5bae80365757; first_observation=False; delay_source_s=0.208; delay_replay_s=0.833
- duplicate FP: source 10.1667s; person_id=a0a2c656-0bae-4b9f-83d9-5bae80365757; first_observation=False
- duplicate FP: source 12.6333s; person_id=a0a2c656-0bae-4b9f-83d9-5bae80365757; first_observation=False

### chute02

- TP: source 4.54167s; person_id=724bd5b6-911f-4159-b2da-e03b4fa04072; first_observation=False; delay_source_s=1.542; delay_replay_s=6.167

### chute03

- TP: source 6.30833s; person_id=c36d2d35-e8e1-4976-8935-ce019a920965; first_observation=False; delay_source_s=1.308; delay_replay_s=5.233

### chute04

- TP: source 4.03333s; person_id=53f09ea9-e9b8-44df-928f-05bc168ba14c; first_observation=True; delay_source_s=1.033; delay_replay_s=4.133
- duplicate FP: source 6.55s; person_id=53f09ea9-e9b8-44df-928f-05bc168ba14c; first_observation=False

### chute05

- TP: source 4.38333s; person_id=c609bf54-84c2-4dec-8b27-18e18c7b57f4; first_observation=True; delay_source_s=2.383; delay_replay_s=9.533
- duplicate FP: source 5.53333s; person_id=c609bf54-84c2-4dec-8b27-18e18c7b57f4; first_observation=False
- outside-window FP: source 6.19167s; person_id=c609bf54-84c2-4dec-8b27-18e18c7b57f4; first_observation=False

### chute06

- TP: source 6.88333s; person_id=ff5280eb-675f-452e-a424-f7c2d21034b3; first_observation=False; delay_source_s=1.883; delay_replay_s=7.533

### chute07

- TP: source 4.23333s; person_id=2b60c488-f54c-49ba-930e-c84ba2e60979; first_observation=False; delay_source_s=1.233; delay_replay_s=4.933
- duplicate FP: source 4.85833s; person_id=2b60c488-f54c-49ba-930e-c84ba2e60979; first_observation=False
- TP: source 7.03333s; person_id=2b60c488-f54c-49ba-930e-c84ba2e60979; first_observation=False; delay_source_s=1.033; delay_replay_s=4.133
- duplicate FP: source 7.45833s; person_id=ea0aed5d-cec0-4623-898d-eca3e9052890; first_observation=False

### chute08

- TP: source 5.175s; person_id=05eb9190-e12b-4154-9c3a-fced6845ed2e; first_observation=True; delay_source_s=2.175; delay_replay_s=8.700

### chute09

- TP: source 6.175s; person_id=8e7a71a2-1a01-4066-ae32-b119c4338ca4; first_observation=True; delay_source_s=1.175; delay_replay_s=4.700

### chute10

- TP: source 4.70833s; person_id=ce57e7ce-94e3-432b-976d-5c7128ef0c56; first_observation=False; delay_source_s=0.708; delay_replay_s=2.833
- duplicate FP: source 6.54167s; person_id=ce57e7ce-94e3-432b-976d-5c7128ef0c56; first_observation=False
- duplicate FP: source 6.975s; person_id=ce57e7ce-94e3-432b-976d-5c7128ef0c56; first_observation=False
- outside-window FP: source 7.78333s; person_id=ca4ea779-81b2-4dc5-8dc4-13c5d1f70a7e; first_observation=False

### chute11

- TP: source 5.64167s; person_id=5895fceb-d261-4e0d-a611-76ace95b357c; first_observation=True; delay_source_s=2.642; delay_replay_s=10.567

### chute12

- TP: source 5.43333s; person_id=2bd1a27a-70e4-486a-b9d2-b967075015b9; first_observation=False; delay_source_s=0.433; delay_replay_s=1.733
- duplicate FP: source 6.35s; person_id=2bd1a27a-70e4-486a-b9d2-b967075015b9; first_observation=False
- duplicate FP: source 6.68333s; person_id=2bd1a27a-70e4-486a-b9d2-b967075015b9; first_observation=False

### chute13

- FN: source interval 3-6s
- TP: source 8.09167s; person_id=bd087859-2a55-4443-b1de-6a0f244e81e8; first_observation=False; delay_source_s=1.092; delay_replay_s=4.367

### chute14

- TP: source 8.04167s; person_id=ce0e712e-56b3-4e49-b5ce-d66e1308f64f; first_observation=False; delay_source_s=4.042; delay_replay_s=16.167; overlapping tolerance windows: earliest unmatched episode selected
- TP: source 8.78333s; person_id=ce0e712e-56b3-4e49-b5ce-d66e1308f64f; first_observation=False; delay_source_s=0.783; delay_replay_s=3.133; overlapping tolerance windows: earliest unmatched episode selected
- duplicate FP: source 9.56667s; person_id=ce0e712e-56b3-4e49-b5ce-d66e1308f64f; first_observation=False

### chute15

- FN: source interval 6-8s

### chute16

- TP: source 8.78333s; person_id=aeac46aa-85cf-4bb6-b701-9218083bdc89; first_observation=True; delay_source_s=0.783; delay_replay_s=3.133
- duplicate FP: source 9.28333s; person_id=aeac46aa-85cf-4bb6-b701-9218083bdc89; first_observation=False
- duplicate FP: source 9.93333s; person_id=aeac46aa-85cf-4bb6-b701-9218083bdc89; first_observation=False

### chute17

- TP: source 7.93333s; person_id=1c4e400d-7b64-460a-8832-5b7b07a7801e; first_observation=True; delay_source_s=1.933; delay_replay_s=7.733
- duplicate FP: source 8.40833s; person_id=1c4e400d-7b64-460a-8832-5b7b07a7801e; first_observation=False
- duplicate FP: source 8.65s; person_id=756ae4d8-1fe6-4b5a-8b97-922535f87a5c; first_observation=False
- duplicate FP: source 8.90833s; person_id=1c4e400d-7b64-460a-8832-5b7b07a7801e; first_observation=False
- outside-window FP: source 9.31667s; person_id=152f9b39-1533-411e-9e05-91c087a40764; first_observation=False
- outside-window FP: source 9.68333s; person_id=152f9b39-1533-411e-9e05-91c087a40764; first_observation=False
- outside-window FP: source 11.0167s; person_id=9825a2ef-d4c1-4e99-9d96-a61feb3fbebe; first_observation=True
- outside-window FP: source 11.7333s; person_id=152f9b39-1533-411e-9e05-91c087a40764; first_observation=False
- outside-window FP: source 12.7083s; person_id=756ae4d8-1fe6-4b5a-8b97-922535f87a5c; first_observation=False

### chute18

Not scored: failed. no_observed_people

### chute19

- TP: source 5.325s; person_id=437cd0a3-7242-49bf-bfeb-446d5d7dec3f; first_observation=True; delay_source_s=1.325; delay_replay_s=5.300

### chute20

Not scored: failed. no_observed_people

### chute21

- FN: source interval 7-9s

### chute22

Not scored: failed. no_observed_people

### chute23

- TP: source 13.85s; person_id=6a0ecb39-4a50-421f-b5e4-dd1d9fc9aa36; first_observation=False; delay_source_s=1.850; delay_replay_s=7.400
- outside-window FP: source 15.6417s; person_id=6a0ecb39-4a50-421f-b5e4-dd1d9fc9aa36; first_observation=False
- TP: source 21.6917s; person_id=6a0ecb39-4a50-421f-b5e4-dd1d9fc9aa36; first_observation=False; delay_source_s=-0.308; delay_replay_s=-1.233
- duplicate FP: source 22.8417s; person_id=6a0ecb39-4a50-421f-b5e4-dd1d9fc9aa36; first_observation=False
- TP: source 26.0167s; person_id=6a0ecb39-4a50-421f-b5e4-dd1d9fc9aa36; first_observation=False; delay_source_s=0.017; delay_replay_s=0.067
- duplicate FP: source 27.65s; person_id=6a0ecb39-4a50-421f-b5e4-dd1d9fc9aa36; first_observation=False
- TP: source 31.675s; person_id=6a0ecb39-4a50-421f-b5e4-dd1d9fc9aa36; first_observation=False; delay_source_s=2.675; delay_replay_s=10.700
- outside-window FP: source 37.7917s; person_id=6a0ecb39-4a50-421f-b5e4-dd1d9fc9aa36; first_observation=False
- outside-window FP: source 38.4667s; person_id=6a0ecb39-4a50-421f-b5e4-dd1d9fc9aa36; first_observation=False
- outside-window FP: source 43.825s; person_id=6a0ecb39-4a50-421f-b5e4-dd1d9fc9aa36; first_observation=False
- outside-window FP: source 44.0167s; person_id=6a0ecb39-4a50-421f-b5e4-dd1d9fc9aa36; first_observation=False

### chute24

- outside-window FP: source 8.60833s; person_id=d36f3f8c-11e4-46eb-911f-31007972dbd3; first_observation=False
- outside-window FP: source 9.16667s; person_id=d36f3f8c-11e4-46eb-911f-31007972dbd3; first_observation=False
- outside-window FP: source 9.7s; person_id=d36f3f8c-11e4-46eb-911f-31007972dbd3; first_observation=False
- outside-window FP: source 11.15s; person_id=d36f3f8c-11e4-46eb-911f-31007972dbd3; first_observation=False
- outside-window FP: source 12.95s; person_id=d36f3f8c-11e4-46eb-911f-31007972dbd3; first_observation=False
- outside-window FP: source 15.325s; person_id=d36f3f8c-11e4-46eb-911f-31007972dbd3; first_observation=False
- outside-window FP: source 17.275s; person_id=d36f3f8c-11e4-46eb-911f-31007972dbd3; first_observation=False
- outside-window FP: source 18.5417s; person_id=d36f3f8c-11e4-46eb-911f-31007972dbd3; first_observation=False
- outside-window FP: source 19.6417s; person_id=d36f3f8c-11e4-46eb-911f-31007972dbd3; first_observation=False
- outside-window FP: source 20.5167s; person_id=d36f3f8c-11e4-46eb-911f-31007972dbd3; first_observation=False
- outside-window FP: source 20.9333s; person_id=d36f3f8c-11e4-46eb-911f-31007972dbd3; first_observation=False
- outside-window FP: source 29.0833s; person_id=d36f3f8c-11e4-46eb-911f-31007972dbd3; first_observation=False

