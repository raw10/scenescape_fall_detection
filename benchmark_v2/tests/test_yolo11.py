import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from benchmark_v2 import evaluate as evaluation
from benchmark_v2.yolo11_code.main import (
    FallAlerts, comparison_speed, frame_times, main, process_clip,
    reports, retained_detections, write_comparison,
)


class AlertTests(unittest.TestCase):
    def test_continuous_fall_and_first_frame(self):
        alerts = FallAlerts()
        self.assertEqual(alerts.update(0, 0.8), {
            "video_time_s": 0, "confidence": 0.8, "first_observation": True})
        self.assertIsNone(alerts.update(1, 0.9))
        self.assertIsNone(alerts.update(2, 0.9))

    def test_gap_boundaries_start_at_first_fall_free_frame(self):
        for return_time, emits in ((1.249, False), (1.25, True), (1.251, True)):
            with self.subTest(return_time=return_time):
                alerts = FallAlerts()
                alerts.update(0, 0.8)
                self.assertIsNone(alerts.update(1, None))
                event = alerts.update(return_time, 0.7)
                self.assertEqual(event is not None, emits)
                if event:
                    self.assertEqual(event["video_time_s"], return_time)
                    self.assertFalse(event["first_observation"])

    def test_empty_frames_and_later_first_fall(self):
        alerts = FallAlerts()
        self.assertIsNone(alerts.update(0, None))
        self.assertIsNone(alerts.update(0.1, None))
        self.assertFalse(alerts.update(2, 0.5)["first_observation"])
        self.assertIsNone(alerts.update(2.1, None))
        self.assertIsNone(alerts.update(2.35, None))
        self.assertIsNotNone(alerts.update(2.4, 0.5))

    def test_short_gaps_reset_and_new_clip_resets(self):
        alerts = FallAlerts()
        alerts.update(0, 0.8)
        for timestamp, confidence in ((0.1, None), (0.2, 0.8), (0.3, None), (0.4, 0.9)):
            self.assertIsNone(alerts.update(timestamp, confidence))
        self.assertIsNotNone(FallAlerts().update(0, 0.8))

    def test_bad_timestamps_and_gap(self):
        for gap in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                FallAlerts(gap)
        for timestamp in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                FallAlerts().update(timestamp, None)
        alerts = FallAlerts()
        alerts.update(1, None)
        with self.assertRaises(ValueError):
            alerts.update(1, None)

    def test_confidence_boundary_and_class_filter(self):
        boxes = [[1, 2, 3, 4, 0.399, 1], [1, 2, 3, 4, 0.4, 1],
                 [1, 2, 3, 4, 0.9, 0], [1, 2, 3, 4, 0.9, 2]]
        retained = retained_detections(boxes, 0.4)
        self.assertEqual([box["class_id"] for box in retained], [1, 0])
        self.assertEqual(retained[0]["confidence"], 0.4)
        self.assertEqual(retained_detections([], 0.4), [])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "data" / "chute01"
        self.directory.mkdir(parents=True)
        self.output = self.root / "results"
        self.output.mkdir()
        self.baseline = self.root / "baseline"
        self.baseline.mkdir()
        self.settings = {"confidence": 0.4, "fall_free_gap_s": 0.25,
                         "image_size": 640, "device": "cpu", "playback_speed": 0.25,
                         "yolo_results": str(self.output), "scenescape_results": str(self.baseline)}

    def fake_model(self, rows):
        prediction = MagicMock()
        prediction.boxes.data.cpu.return_value.tolist.return_value = rows
        model = MagicMock()
        model.predict.return_value = [prediction]
        model.predictor.args = SimpleNamespace(imgsz=640, conf=0.4)
        return model

    def run_clip(self, reads, rows=None, failure=None):
        cv2 = MagicMock()
        capture = cv2.VideoCapture.return_value
        capture.read.side_effect = [(available, object()) for available in reads]
        model = self.fake_model(rows or [])
        if failure:
            model.predict.side_effect = failure
        with patch("benchmark_v2.yolo11_code.main.frame_times", return_value=([0, 0.1], 0, "recovered frame")):
            result = process_clip(self.directory, self.output, model, cv2, self.settings)
        capture.release.assert_called_once()
        for prediction_call in model.predict.call_args_list:
            self.assertEqual(prediction_call.kwargs["device"], "cpu")
        return result

    def save_baseline(self, name="chute01", speed=0.25, status="complete"):
        (self.baseline / f"{name}.json").write_text(json.dumps({
            "chute": name, "playback_speed": speed, "status": status, "events": []}))

    def test_zero_boxes_complete_and_trace_has_every_frame(self):
        self.settings["device"] = "cuda"
        result = self.run_clip([True, True, False])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["events"], [])
        self.assertEqual(result["processed_frames"], 2)
        self.assertEqual(result["decoder_warnings"], "recovered frame")
        trace = (self.output / "chute01" / "predictions.jsonl").read_text().splitlines()
        self.assertEqual([json.loads(line)["video_time_s"] for line in trace], [0, 0.1])
        saved = json.loads((self.output / "chute01.json").read_text())
        self.assertEqual(saved, result)

    def test_multiple_boxes_one_alert_and_evaluator_accepts_schema(self):
        rows = [[0, 0, 10, 10, 0.8, 1], [1, 1, 11, 11, 0.9, 1], [0, 0, 5, 5, 0.99, 0]]
        result = self.run_clip([True, True, False], rows)
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(result["events"][0]["confidence"], 0.9)
        self.assertNotIn("person_id", result["events"][0])
        (self.directory / "ground_truth.json").write_text(json.dumps({
            "reviewed": True, "falls": [{"start_s": 0, "end_s": 1}]}))
        with patch("benchmark_v2.evaluate.video_duration", return_value=2):
            row = evaluation.evaluate_clip(self.directory, self.output, 4, 1)
        self.assertEqual((row["status"], row["tp"], row["fp"]), ("scored", 1, 0))

    def test_partial_decode_failed(self):
        result = self.run_clip([True, False])
        self.assertEqual(result["status"], "failed")
        self.assertIn("1 of 2", result["error"])

    def test_extra_frame_failed(self):
        result = self.run_clip([True, True, True])
        self.assertEqual(result["status"], "failed")
        self.assertIn("more frames", result["error"])

    def test_inference_failure_failed(self):
        result = self.run_clip([True], failure=RuntimeError("bad inference"))
        self.assertEqual(result["status"], "failed")
        self.assertIn("bad inference", result["error"])

    def test_missing_video_failed(self):
        with patch("benchmark_v2.yolo11_code.main.frame_times", side_effect=ValueError("missing video")):
            result = process_clip(self.directory, self.output, None, None, self.settings)
        self.assertEqual(result["status"], "failed")
        self.assertIn("missing video", result["error"])

    def test_selected_camera_decode_and_metadata(self):
        backend = MagicMock()
        backend.VideoCapture.return_value.read.side_effect = [(True, object()), (False, None)]
        with patch("benchmark_v2.yolo11_code.main.frame_times", return_value=([0], 5, "")) as probe:
            result = process_clip(self.directory, self.output, self.fake_model([]), backend,
                                  self.settings, camera="cam3")
        video = self.directory / "cam3.avi"
        probe.assert_called_once_with(video)
        backend.VideoCapture.assert_called_once_with(str(video))
        self.assertEqual(result["camera"], "cam3")
        self.assertEqual(result["reference_camera"], "cam3")
        self.assertEqual(result["source_time_origin_s"], 5)
        self.assertEqual(result["status"], "complete")

    def test_baseline_camera_mismatch_and_combined_rejected(self):
        self.save_baseline()
        with self.assertRaises(ValueError):
            comparison_speed(self.baseline, [self.directory], "cam1")
        path = self.baseline / "chute01.json"
        result = json.loads(path.read_text())
        result.update(cameras=["cam1", "cam2"], reference_camera="cam2")
        path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "single-camera"):
            comparison_speed(self.baseline, [self.directory])

    def test_reports_forward_camera(self):
        args = SimpleNamespace(output=self.output, data_dir=self.directory.parent,
                               scenescape_results=self.baseline, chute="chute01", camera="cam1")
        settings = {**self.settings, "camera": "cam1", "reference_camera": "cam1"}
        with patch("benchmark_v2.yolo11_code.main.evaluation.evaluate", return_value=[]) as evaluate:
            self.assertEqual(reports(args, settings), 1)
        self.assertEqual([call.args[-1] for call in evaluate.call_args_list], ["cam1", "cam1"])

    def test_chutes_process_only_explicit_list_in_one_output(self):
        names = ["chute01", "chute02", "chute03"]
        for name in names[1:] + ["chute24"]:
            self.directory.with_name(name).mkdir()
        weights = self.root / "best.pt"
        weights.write_bytes(b"test weights")
        output = self.root / "fresh"
        model = MagicMock(task="detect", names={0: "non-fall", 1: "fall"})
        model.overrides = {}
        with patch("benchmark_v2.yolo11_code.main.YOLO", return_value=model), patch(
                "benchmark_v2.yolo11_code.main.comparison_speed", return_value=0.25) as speed, patch(
                "benchmark_v2.yolo11_code.main.process_clip", return_value={"status": "complete"}) as process, patch(
                "benchmark_v2.yolo11_code.main.reports", return_value=0) as report:
            self.assertEqual(main([
                "--chutes", *names, "--camera", "cam3", "--weights", str(weights),
                "--coverage-end-tolerance-s", "0.5",
                "--data-dir", str(self.directory.parent), "--output", str(output),
                "--scenescape-results", str(self.baseline)]), 0)
        self.assertEqual([call.args[0].name for call in process.call_args_list], names)
        self.assertTrue(all(call.args[1] == output and call.args[-1] == "cam3"
                            for call in process.call_args_list))
        self.assertEqual([directory.name for directory in speed.call_args.args[1]], names)
        self.assertEqual(report.call_args.args[0].chutes, names)
        saved = json.loads((output / "run.json").read_text())
        self.assertEqual(saved["clips"], names)
        self.assertEqual(saved["device"], "cpu")
        self.assertEqual(saved["coverage_end_tolerance_s"], 0.5)
        self.assertTrue(all(call.args[4]["coverage_end_tolerance_s"] == 0.5
                            for call in process.call_args_list))

    def test_reports_forward_coverage_end_tolerance_to_both_methods(self):
        for tolerance in (0, 0.5):
            args = SimpleNamespace(output=self.output, data_dir=self.directory.parent,
                                   scenescape_results=self.baseline, chute="chute01")
            if tolerance:
                args.coverage_end_tolerance_s = tolerance
            with self.subTest(tolerance=tolerance), patch(
                    "benchmark_v2.yolo11_code.main.evaluation.evaluate", return_value=[]) as evaluate, patch(
                    "benchmark_v2.yolo11_code.main.evaluation.write_report") as write_report, patch(
                    "benchmark_v2.yolo11_code.main.write_comparison", return_value=[]) as comparison:
                self.assertEqual(reports(args, self.settings), 1)
            self.assertEqual([call.args[1] for call in evaluate.call_args_list],
                             [self.output, self.baseline])
            self.assertEqual([call.args[1] for call in write_report.call_args_list],
                             [self.output, self.baseline])
            for mocked in (evaluate, write_report):
                self.assertEqual([call.kwargs["coverage_end_tolerance_s"]
                                  for call in mocked.call_args_list], [tolerance, tolerance])
            self.assertEqual(comparison.call_args.args[3]["coverage_end_tolerance_s"], tolerance)

    def test_cli_rejects_invalid_coverage_end_tolerance_before_model_loading(self):
        for tolerance in ("-0.1", "nan", "inf", "-inf"):
            with self.subTest(tolerance=tolerance), patch("sys.stderr"), patch(
                    "benchmark_v2.yolo11_code.main.YOLO") as model, patch(
                    "benchmark_v2.yolo11_code.main.evaluation.load_json") as load, self.assertRaises(SystemExit) as raised:
                main(["--coverage-end-tolerance-s=" + tolerance, "--evaluate-only"])
            self.assertEqual(raised.exception.code, 2)
            model.assert_not_called()
            load.assert_not_called()

    def test_chutes_reject_invalid_duplicate_missing_and_conflicting_selection(self):
        selections = [(["--chutes", "chute01", "chute01"], "unique"),
                      (["--chutes", "chute01", "../chute02"], "directory names"),
                      (["--chutes", "chute01", "chute02"], "does not exist"),
                      (["--chutes", "chute01", "other"], "directory names"),
                      (["--chutes"], "expected at least one argument"),
                      (["--chute", "chute01", "--chutes", "chute01"], "not allowed")]
        for selection, message in selections:
            with self.subTest(selection=selection), patch(
                    "sys.stderr") as stderr, patch(
                    "benchmark_v2.yolo11_code.main.YOLO") as model, self.assertRaises(SystemExit) as raised:
                main(["--data-dir", str(self.directory.parent), *selection])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn(message, "".join(call.args[0] for call in stderr.write.call_args_list))
            model.assert_not_called()

    def test_reports_evaluate_only_selected_chutes(self):
        names = ["chute01", "chute02", "chute03"]
        args = SimpleNamespace(output=self.output, data_dir=self.directory.parent,
                               scenescape_results=self.baseline, chute=None, chutes=names)
        with patch("benchmark_v2.yolo11_code.main.evaluation.evaluate", return_value=[]) as evaluate:
            self.assertEqual(reports(args, self.settings), 1)
        self.assertEqual([call.args[4] for call in evaluate.call_args_list],
                         [name for name in names for _ in range(2)])
        self.assertEqual([call.args[1] for call in evaluate.call_args_list],
                         [self.output, self.baseline] * 3)

    def test_evaluate_only_uses_saved_clips_and_rejects_mismatched_selection(self):
        names = ["chute01", "chute02"]
        self.directory.with_name("chute02").mkdir()
        self.directory.with_name("chute03").mkdir()
        settings = {**self.settings, "clips": names}
        (self.output / "run.json").write_text(json.dumps(settings))
        before = (self.output / "run.json").read_bytes()
        arguments = ["--evaluate-only", "--data-dir", str(self.directory.parent),
                     "--output", str(self.output), "--scenescape-results", str(self.baseline)]
        with patch("benchmark_v2.yolo11_code.main.YOLO") as model, patch(
                "benchmark_v2.yolo11_code.main.comparison_speed", return_value=0.25) as speed, patch(
                "benchmark_v2.yolo11_code.main.reports", return_value=0) as report:
            self.assertEqual(main(arguments), 0)
            self.assertEqual(report.call_args.args[0].chutes, names)
            self.assertEqual([directory.name for directory in speed.call_args.args[1]], names)
            self.assertEqual(main([*arguments, "--chutes", "chute02"]), 0)
            self.assertEqual(report.call_args.args[0].chutes, ["chute02"])
            for selection in (["--chutes", "chute01", "chute03"], ["--chute", "chute03"]):
                with self.subTest(selection=selection), patch("sys.stderr") as stderr, self.assertRaises(SystemExit) as raised:
                    main([*arguments, *selection])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("saved run clips", "".join(call.args[0] for call in stderr.write.call_args_list))
            model.assert_not_called()
        self.assertEqual((self.output / "run.json").read_bytes(), before)

    def test_evaluate_only_legacy_settings_rejected_for_other_camera(self):
        self.save_baseline()
        path = self.baseline / "chute01.json"
        result = json.loads(path.read_text())
        result.update(camera="cam1", reference_camera="cam1")
        path.write_text(json.dumps(result))
        (self.output / "run.json").write_text(json.dumps(self.settings))
        with patch("benchmark_v2.yolo11_code.main.YOLO") as model, self.assertRaises(SystemExit) as raised:
            main(["--evaluate-only", "--camera", "cam1", "--chute", "chute01",
                  "--data-dir", str(self.directory.parent), "--output", str(self.output),
                  "--scenescape-results", str(self.baseline)])
        self.assertEqual(raised.exception.code, 2)
        model.assert_not_called()

    def test_baseline_speeds_include_failed_runs_and_are_not_modified(self):
        self.save_baseline()
        self.save_baseline("chute02", status="no_observed_people")
        paths = [self.baseline / "chute01.json", self.baseline / "chute02.json"]
        original = [path.read_bytes() for path in paths]
        directories = [self.directory, self.directory.with_name("chute02"), self.directory.with_name("chute03")]
        self.assertEqual(comparison_speed(self.baseline, directories), 0.25)
        self.assertEqual([path.read_bytes() for path in paths], original)
        self.save_baseline("chute02", speed=0.5)
        with self.assertRaisesRegex(ValueError, "mixed"):
            comparison_speed(self.baseline, directories)

    def test_baseline_missing_or_invalid_speed(self):
        with self.assertRaisesRegex(ValueError, "no matching"):
            comparison_speed(self.baseline, [self.directory])
        for speed in (0, 1, None, True):
            self.save_baseline(speed=speed)
            with self.assertRaises(ValueError):
                comparison_speed(self.baseline, [self.directory])

    def test_output_collision_leaves_existing_files_untouched(self):
        self.save_baseline()
        sentinel = self.output / "existing.txt"
        sentinel.write_text("keep me")
        with self.assertRaises(SystemExit) as raised:
            main(["--data-dir", str(self.directory.parent), "--output", str(self.output),
                  "--scenescape-results", str(self.baseline)])
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(sentinel.read_text(), "keep me")

    def probe(self, frames=None, declared="2", origin="5", warnings=""):
        metadata = {"streams": [{"start_time": origin, "nb_frames": declared}],
                    "format": {"start_time": origin, "duration": "2"},
                    "frames": [{"best_effort_timestamp_time": timestamp}
                               for timestamp in (frames if frames is not None else ("5", "5.1"))]}
        return subprocess.CompletedProcess([], 0, json.dumps(metadata), warnings)

    def test_frame_timestamps_normalize_origin_and_keep_warnings(self):
        with patch("benchmark_v2.yolo11_code.main.subprocess.run", return_value=self.probe(warnings="damaged block")):
            timestamps, origin, warnings = frame_times(Path("video.avi"))
        self.assertEqual(timestamps[0], 0)
        self.assertAlmostEqual(timestamps[1], 0.1)
        self.assertEqual(origin, 5)
        self.assertEqual(warnings, "damaged block")

    def test_bad_frame_timestamps_and_count_rejected(self):
        for metadata in (self.probe(frames=[]), self.probe(frames=("5", "5")),
                         self.probe(frames=("5", "nan")), self.probe(declared="3"),
                         self.probe(origin="nan"), self.probe(frames=("4", "5"))):
            with self.subTest(metadata=metadata), patch(
                    "benchmark_v2.yolo11_code.main.subprocess.run", return_value=metadata):
                with self.assertRaises(ValueError):
                    frame_times(Path("video.avi"))

    def scored(self, name, events):
        return {"chute": name, "status": "scored", **evaluation.score(
            [{"start_s": 1, "end_s": 2}], [{"video_time_s": time} for time in events], 10, 4)}

    def test_paired_totals_use_common_set_and_report_exclusions(self):
        yolo = [self.scored("chute01", [1, 1.5]), self.scored("chute02", [1]),
                {"chute": "chute03", "status": "unreviewed"}]
        scene = [self.scored("chute01", [1]),
                 {"chute": "chute02", "status": "failed", "reason": "no_observed_people"},
                 {"chute": "chute03", "status": "unreviewed"}]
        common = write_comparison(yolo, scene, self.output, self.settings)
        self.assertEqual(common, ["chute01"])
        with (self.output / "comparison.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["common_clips"] for row in rows], ["1", "1"])
        self.assertEqual([(row["tp"], row["fp"]) for row in rows], [("1", "1"), ("1", "0")])
        text = (self.output / "comparison.md").read_text()
        self.assertIn("no_observed_people", text)
        self.assertIn("unreviewed", text)
        self.assertIn("median_delay_source_s", text)
        self.assertIn("| YOLO11 | 1 | 1 | 1 | 0 |", text)
        self.assertIn("| chute01 | 1/1/0 | 1/0/0 |", text)
        self.assertNotIn("training-data", text)
        self.assertNotIn("Training-set overlap", text)

    def test_empty_common_set_is_not_a_zero_score(self):
        rows = [{"chute": "chute01", "status": "failed"}]
        self.assertEqual(write_comparison(rows, rows, self.output, self.settings), [])
        self.assertIn("NOT COMPARABLE", (self.output / "comparison.md").read_text())
        with (self.output / "comparison.csv").open() as handle:
            self.assertEqual(list(csv.DictReader(handle)), [])
        with self.assertRaises(ValueError):
            write_comparison(rows, [], self.output, self.settings)

    def test_evaluate_only_needs_no_model_and_preserves_results(self):
        self.save_baseline()
        (self.output / "run.json").write_text(json.dumps(self.settings))
        before = (self.output / "run.json").read_bytes()
        with patch("benchmark_v2.yolo11_code.main.reports", return_value=0) as report:
            code = main(["--evaluate-only", "--data-dir", str(self.directory.parent),
                         "--output", str(self.output), "--scenescape-results", str(self.baseline),
                         "--coverage-end-tolerance-s", "0.5"])
        self.assertEqual(code, 0)
        report.assert_called_once()
        self.assertEqual(report.call_args.args[0].coverage_end_tolerance_s, 0.5)
        self.assertEqual(report.call_args.args[1]["coverage_end_tolerance_s"], 0.5)
        self.assertEqual((self.output / "run.json").read_bytes(), before)
        self.save_baseline(speed=0.5)
        with self.assertRaises(SystemExit) as raised:
            main(["--evaluate-only", "--data-dir", str(self.directory.parent),
                  "--output", str(self.output), "--scenescape-results", str(self.baseline)])
        self.assertEqual(raised.exception.code, 2)

    def test_reports_mark_incomplete_coverage_with_nonzero_exit(self):
        self.save_baseline(status="no_observed_people")
        (self.directory / "ground_truth.json").write_text(json.dumps({"reviewed": True, "falls": []}))
        (self.output / "chute01.json").write_text(json.dumps({
            "chute": "chute01", "playback_speed": 0.25, "status": "complete", "events": []}))
        args = SimpleNamespace(output=self.output, data_dir=self.directory.parent,
                               scenescape_results=self.baseline, chute=None)
        with patch("benchmark_v2.evaluate.video_duration", return_value=2):
            self.assertEqual(reports(args, self.settings), 1)
        self.assertTrue((self.output / "reports" / "yolo" / "report.md").is_file())
        self.assertTrue((self.output / "reports" / "scenescape" / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()