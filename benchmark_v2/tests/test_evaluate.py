import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_v2.evaluate import aggregate, evaluate, metrics, score, validate_settings, write_report


class MatchingTests(unittest.TestCase):
    def test_coverage_end_allowance_is_strict_and_does_not_extend_events(self):
        falls = [{"start_s": 9, "end_s": 20.49}]
        with self.assertRaises(ValueError):
            score(falls, [], 20, 4)
        result = score(falls, [{"video_time_s": 20}], 20, 4,
                       coverage_end_tolerance_s=0.5)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["details"][0]["fall"], falls[0])
        with self.assertRaises(ValueError):
            score([{"start_s": 9, "end_s": 20.5}], [], 20, 4,
                  coverage_end_tolerance_s=0.5)
        with self.assertRaisesRegex(ValueError, "outside video duration"):
            score(falls, [{"video_time_s": 20.1}], 20, 4,
                  coverage_end_tolerance_s=0.5)

    def evaluate(self, times, falls=None, slowdown=4, tolerance=1):
        if falls is None:
            falls = [{"start_s": 9, "end_s": 12}]
        events = [{"video_time_s": timestamp, "person_id": str(index)}
                  for index, timestamp in enumerate(times)]
        return score(falls, events, 20, slowdown, tolerance)

    def test_no_double_scaling(self):
        for slowdown in (2, 4):
            with self.subTest(slowdown=slowdown):
                result = self.evaluate([10], slowdown=slowdown)
                self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 0))
                self.assertEqual(result["details"][0]["delay_source_s"], 1)
                self.assertEqual(result["median_delay_replay_s"], slowdown)

    def test_source_tolerance_boundaries(self):
        for timestamp in (8, 12, 13):
            self.assertEqual(self.evaluate([timestamp])["tp"], 1)
        for timestamp in (7.999, 13.001):
            result = self.evaluate([timestamp])
            self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 1))
        self.assertEqual(self.evaluate([8])["details"][0]["delay_source_s"], -1)

    def test_duplicates_and_outside_alert(self):
        result = self.evaluate([10, 11, 19])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 2, 0))
        self.assertEqual(result["duplicate_fp"], 1)
        self.assertEqual(result["outside_window_fp"], 1)

    def test_miss(self):
        self.assertEqual(self.evaluate([])["fn"], 1)

    def test_negative_video(self):
        self.assertEqual(self.evaluate([], falls=[])["fp"], 0)
        self.assertIsNone(self.evaluate([], falls=[])["precision"])
        self.assertEqual(self.evaluate([3], falls=[])["fp"], 1)

    def test_video_edges(self):
        falls = [{"start_s": 0, "end_s": 2}, {"start_s": 18, "end_s": 20}]
        self.assertEqual(self.evaluate([0, 20], falls=falls)["tp"], 2)

    def test_ambiguous_windows_are_deterministic(self):
        falls = [{"start_s": 2, "end_s": 4}, {"start_s": 5, "end_s": 7}]
        result = self.evaluate([4.5, 5], falls=falls)
        self.assertEqual(result["tp"], 2)
        self.assertTrue(result["details"][0]["ambiguous"])
        self.assertEqual(result["details"][0]["fall"], falls[0])

    def test_invalid_intervals(self):
        for start, end in ((-1, 3), (4, 4), (4, 3), (2, 21), (float("nan"), 3)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                self.evaluate([], falls=[{"start_s": start, "end_s": end}])
        with self.assertRaises(ValueError):
            self.evaluate([], falls=[{"start_s": 1, "end_s": 4},
                                     {"start_s": 3, "end_s": 5}])

    def test_invalid_events(self):
        for timestamp in (-1, 21, float("inf"), True):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                self.evaluate([timestamp])

    def test_invalid_settings(self):
        for slowdown in (0, 1, -2, float("inf"), float("nan"), True):
            with self.subTest(slowdown=slowdown), self.assertRaises(ValueError):
                validate_settings(slowdown, 1)
        with self.assertRaises(ValueError):
            validate_settings(4, -1)

    def test_metrics(self):
        self.assertEqual(metrics(2, 1, 2), {"precision": 2 / 3, "recall": 0.5, "f1": 4 / 7})
        self.assertEqual(metrics(0, 0, 0), {"precision": None, "recall": None, "f1": None})


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "data"
        self.results = self.root / "results"
        self.results.mkdir()
        self.directory = self.data / "chute01"
        self.directory.mkdir(parents=True)
        self.annotation = {"reviewed": True, "falls": [{"start_s": 9, "end_s": 12}]}
        self.result = {"chute": "chute01", "playback_speed": 0.25, "status": "complete",
                       "events": [{"video_time_s": 10, "person_id": "person", "first_observation": True}]}
        duration_patch = patch("benchmark_v2.evaluate.video_duration", return_value=20)
        duration_patch.start()
        self.addCleanup(duration_patch.stop)
        self.save()

    def save(self):
        (self.directory / "ground_truth.json").write_text(json.dumps(self.annotation))
        (self.results / "chute01.json").write_text(json.dumps(self.result))

    def run_evaluation(self):
        return evaluate(self.data, self.results, 4)

    def test_report_and_csv(self):
        rows = self.run_evaluation()
        output = self.root / "report"
        summary = write_report(rows, self.results, output, 4, 1)
        self.assertEqual(summary["tp"], 1)
        report = (output / "report.md").read_text()
        self.assertIn("1/1 clips scored. Complete.", report)
        self.assertIn("delay_source_s=1.000; delay_replay_s=4.000", report)
        self.assertIn("first_observation=True", report)
        self.assertNotIn("FP/hour", report)
        self.assertIn("## Metric Definitions", report)
        for metric in ("gt", "tp", "fp", "fn", "duplicate_fp", "outside_window_fp",
                       "precision", "recall", "f1", "median_delay_replay_s"):
            self.assertIn(f"| `{metric}` |", report)
        self.assertIn("in slowed-replay seconds", report)
        self.assertIn("Equals duplicate_fp + outside_window_fp", report)
        with (output / "per_video.csv").open() as handle:
            record = next(csv.DictReader(handle))
        self.assertEqual(record["slowdown_factor"], "4")
        self.assertEqual(record["median_delay_replay_s"], "4")
        self.assertEqual(record["tp"], "1")

    def test_missing_result_is_not_scored(self):
        (self.results / "chute01.json").unlink()
        rows = self.run_evaluation()
        self.assertEqual(rows[0]["status"], "missing")
        self.assertEqual(rows[0]["gt"], 1)
        self.assertNotIn("tp", rows[0])
        output = self.root / "report"
        summary = write_report(rows, self.results, output, 4, 1)
        self.assertEqual(summary["scored"], 0)
        self.assertIsNone(summary["recall"])
        self.assertIn("INCOMPLETE", (output / "report.md").read_text())

    def test_failed_result(self):
        self.result["status"] = "no_observed_people"
        self.save()
        row = self.run_evaluation()[0]
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["reason"], "no_observed_people")
        self.assertNotIn("fn", row)

    def test_unreviewed_is_not_negative(self):
        for annotation in ({"falls": []}, {"reviewed": False, "falls": []}):
            self.annotation = annotation
            self.save()
            self.assertEqual(self.run_evaluation()[0]["status"], "unreviewed")
        (self.directory / "ground_truth.json").unlink()
        self.assertEqual(self.run_evaluation()[0]["status"], "unreviewed")

    def test_reviewed_negative(self):
        self.annotation["falls"] = []
        self.save()
        row = self.run_evaluation()[0]
        self.assertEqual((row["tp"], row["fp"], row["fn"]), (0, 1, 0))

    def test_wrong_or_missing_speed(self):
        for speed in (1, 0.5, None, float("nan")):
            self.result["playback_speed"] = speed
            self.save()
            with self.subTest(speed=speed), self.assertRaisesRegex(ValueError, "playback_speed"):
                self.run_evaluation()
        del self.result["playback_speed"]
        self.save()
        with self.assertRaises(ValueError):
            self.run_evaluation()

    def test_mixed_speeds_rejected(self):
        second = self.data / "chute02"
        second.mkdir()
        (second / "ground_truth.json").write_text(json.dumps(self.annotation))
        (self.results / "chute02.json").write_text(json.dumps(
            {**self.result, "chute": "chute02", "playback_speed": 0.5}))
        with self.assertRaisesRegex(ValueError, "chute02.*playback_speed"):
            self.run_evaluation()

    def test_malformed_inputs(self):
        for annotation in ({"reviewed": "true", "falls": []},
                           {"reviewed": True}, {"reviewed": True, "falls": [None]}):
            self.annotation = annotation
            self.save()
            with self.subTest(annotation=annotation), self.assertRaises(ValueError):
                self.run_evaluation()

    def test_micro_aggregation(self):
        first = {**score([{"start_s": 9, "end_s": 12}], [{"video_time_s": 10}], 20, 4),
                 "status": "scored"}
        second = {**score([], [{"video_time_s": 1}, {"video_time_s": 2}], 20, 4),
                  "status": "scored"}
        result = aggregate([first, second, {"status": "failed", "gt": 5}])
        self.assertEqual(result["precision"], 1 / 3)
        self.assertEqual(result["recall"], 1)
        self.assertEqual((result["scored"], result["expected"]), (2, 3))

    def test_missing_inputs(self):
        with self.assertRaises(ValueError):
            evaluate(self.data, self.root / "absent", 4)
        with self.assertRaises(ValueError):
            evaluate(self.data, self.results, 4, chute="chute99")

    def test_selected_camera_duration(self):
        self.result.update(camera="cam1", reference_camera="cam1")
        self.save()
        with patch("benchmark_v2.evaluate.video_duration", return_value=20) as duration:
            row = evaluate(self.data, self.results, 4, camera="cam1")[0]
        duration.assert_called_once_with(self.directory / "cam1.avi")
        self.assertEqual(row["reference_camera"], "cam1")

    def test_invalid_camera_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(self.data, self.results, 4, camera="../cam1")

    def test_metadata_mismatch_and_legacy_cam2_only(self):
        for metadata in ({}, {"camera": "cam2"}, {"camera": "cam1", "reference_camera": "cam2"},
                         {"cameras": ["cam1", "cam2"], "reference_camera": "cam2"}):
            self.result.update(metadata)
            self.save()
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                evaluate(self.data, self.results, 4, camera="cam1")

    def test_unreviewed_still_validates_and_preserves_camera_metadata(self):
        self.annotation["reviewed"] = False
        self.result.update(camera="cam1", reference_camera="cam1", timing="source seconds")
        self.save()
        with self.assertRaises(ValueError):
            self.run_evaluation()
        row = evaluate(self.data, self.results, 4, camera="cam1")[0]
        self.assertEqual(row["status"], "unreviewed")
        self.assertEqual(row["timing"], "source seconds")

    def test_combined_coverage_and_provenance(self):
        self.result.update(cameras=["cam1", "cam2"], reference_camera="cam2",
                           source_windows={name: {"source_start_s": 0, "source_end_s": 19}
                                           for name in ("cam1", "cam2")})
        self.save()
        with patch("benchmark_v2.evaluate.video_origin", return_value=0):
            row = self.run_evaluation()[0]
        self.assertEqual(row["cameras"], ["cam1", "cam2"])
        self.assertEqual(row["source_windows"], self.result["source_windows"])
        with patch("benchmark_v2.evaluate.video_duration", side_effect=[20, 10]), self.assertRaises(ValueError):
            self.run_evaluation()
        with patch("benchmark_v2.evaluate.video_origin", side_effect=[0, 1]), self.assertRaisesRegex(
                ValueError, "timeline"):
            self.run_evaluation()
        self.result["source_windows"]["cam1"]["source_start_s"] = 10
        self.save()
        with patch("benchmark_v2.evaluate.video_origin", return_value=0), self.assertRaisesRegex(
                ValueError, "fully cover"):
            self.run_evaluation()

    def test_replay_metadata_per_camera_coverage(self):
        self.result.update(selected_cameras=["cam1", "cam2"], reference_camera="cam2",
                           source_start_s=0, source_end_s=19,
                           per_camera={name: {"source_start_s": 0, "source_end_s": 19}
                                       for name in ("cam1", "cam2")})
        self.save()
        with patch("benchmark_v2.evaluate.video_origin", return_value=0):
            row = self.run_evaluation()[0]
        self.assertEqual(row["cameras"], ["cam1", "cam2"])
        self.assertEqual(row["per_camera"], self.result["per_camera"])
        self.result["per_camera"]["cam1"]["source_end_s"] = 10
        self.save()
        with patch("benchmark_v2.evaluate.video_origin", return_value=0), self.assertRaisesRegex(
                ValueError, "fully cover"):
            self.run_evaluation()


if __name__ == "__main__":
    unittest.main()