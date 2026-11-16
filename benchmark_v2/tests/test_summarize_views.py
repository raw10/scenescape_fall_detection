import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_v2 import evaluate as evaluation
from benchmark_v2.summarize_views import main, summarize, write_summary
from benchmark_v2.visualize_views import main as visual_main, plot_missed_falls, write_visual_report


class SummaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "data"
        self.results = self.root / "results"
        self.chutes = ["chute01", "chute02", "chute03"]
        for chute in self.chutes:
            directory = self.data / chute
            directory.mkdir(parents=True)
            (directory / "ground_truth.json").write_text(json.dumps({
                "reviewed": True, "falls": [{"start_s": 1, "end_s": 2}]}))
            for condition in ("scenescape/cam1", "scenescape/cam2", "yolo/cam1", "yolo/cam2",
                              "scenescape/combined"):
                directory = self.results / condition
                directory.mkdir(parents=True, exist_ok=True)
                camera = condition.split("/")[1]
                metadata = ({"cameras": ["cam1", "cam2"], "reference_camera": "cam2"}
                            if camera == "combined" else {"camera": camera, "reference_camera": camera})
                (directory / f"{chute}.json").write_text(json.dumps({
                    "chute": chute, "status": "complete", "playback_speed": 0.25,
                    "events": [{"video_time_s": 1.5}], "timing": "source seconds", **metadata}))
        duration = patch("benchmark_v2.evaluate.video_duration", return_value=10)
        origin = patch("benchmark_v2.evaluate.video_origin", return_value=0)
        duration.start()
        origin.start()
        self.addCleanup(duration.stop)
        self.addCleanup(origin.stop)

    def test_common_set_counts_do_not_pool_repeated_labels(self):
        (self.results / "yolo/cam1/chute02.json").unlink()
        path = self.results / "scenescape/cam2/chute03.json"
        result = json.loads(path.read_text())
        result["status"] = "no_observed_people"
        path.write_text(json.dumps(result))
        summary = summarize(self.data, self.results, self.chutes)
        self.assertEqual(summary["cameras"], ["cam1", "cam2"])
        self.assertEqual(summary["common"], ["chute01"])
        self.assertEqual(summary["excluded"], ["chute02", "chute03"])
        self.assertEqual(len(summary["records"]), 10)
        common = [record for record in summary["records"] if record["subset"] == "common"]
        self.assertTrue(all(record["gt"] == record["tp"] == 1 for record in common))
        individual = next(record for record in summary["records"]
                          if record["condition"] == "yolo/cam1" and record["subset"] == "individual")
        self.assertEqual((individual["scored"], individual["expected"], individual["missing"]), (2, 3, 1))
        output = self.root / "report"
        write_summary(summary, output)
        text = (output / "summary.md").read_text()
        self.assertIn("Common scored chutes: chute01", text)
        self.assertIn("yolo/cam1=missing", text)
        self.assertIn("no_observed_people", text)
        self.assertIn("source seconds", text)
        with (output / "summary.csv").open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 10)
        overview = (output / "overview.md").read_text()
        self.assertIn("1 shared chutes out of 3", overview)
        self.assertIn("containing 1 labeled falls", overview)
        self.assertIn("Missed Falls: Primary Metric", overview)
        self.assertLess(overview.index("missed-falls.png"), overview.index("outcomes.png"))
        for name in ("outcomes.png", "camera-f1.png", "missed-falls.png"):
            self.assertTrue((output / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_empty_common_set_is_not_zero_metrics(self):
        for path in (self.results / "yolo/cam1").glob("*.json"):
            path.unlink()
        summary = summarize(self.data, self.results, self.chutes, ["cam1", "cam2"])
        self.assertEqual(summary["common"], [])
        for record in summary["records"]:
            if record["subset"] == "common" or record["condition"] == "yolo/cam1":
                self.assertIsNone(record["tp"])
                self.assertIsNone(record["f1"])
        output = self.root / "report"
        write_summary(summary, output)
        self.assertIn("NOT COMPARABLE", (output / "summary.md").read_text())
        self.assertIn("NOT COMPARABLE", (output / "overview.md").read_text())
        self.assertFalse((output / "outcomes.png").exists())
        self.assertFalse((output / "missed-falls.png").exists())

    def test_missed_falls_chart_uses_fn_counts_and_shared_denominator(self):
        records = [{"condition": condition, "fn": count, "gt": 3} for condition, count in (
            ("scenescape/cam1", 0), ("yolo/cam1", 1), ("scenescape/cam2", 2),
            ("yolo/cam2", 3), ("scenescape/combined", 1))]
        with patch("benchmark_v2.visualize_views.save_figure") as save:
            plot_missed_falls(records, ["cam1", "cam2"], self.root)
        figure, path = save.call_args.args
        axis = figure.axes[0]
        self.assertEqual(path.name, "missed-falls.png")
        self.assertEqual([bar.get_width() for bar in axis.patches], [0, 1, 2, 3, 1])
        self.assertEqual([text.get_text() for text in axis.texts], [
            "0 / 3 (0.0%)", "1 / 3 (33.3%)", "2 / 3 (66.7%)", "3 / 3 (100.0%)", "1 / 3 (33.3%)"])
        self.assertIn("lower is better", axis.get_xlabel())
        figure.clear()

    def test_visual_report_rejects_mismatched_comparison_and_alert_counts(self):
        for field, value in (("chutes", "chute99"), ("gt", 99), ("fp", 99)):
            summary = summarize(self.data, self.results, self.chutes)
            common = next(record for record in summary["records"] if record["subset"] == "common")
            common[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                write_visual_report(summary["records"], self.root / "invalid")

    def test_visual_report_handles_no_alerts_and_no_labeled_falls(self):
        for falls in ([{"start_s": 1, "end_s": 2}], []):
            for chute in self.chutes:
                (self.data / chute / "ground_truth.json").write_text(json.dumps({
                    "reviewed": True, "falls": falls}))
            for path in self.results.rglob("*.json"):
                result = json.loads(path.read_text())
                result["events"] = []
                path.write_text(json.dumps(result))
            summary = summarize(self.data, self.results, self.chutes)
            output = self.root / "no-alerts"
            with self.subTest(falls=falls):
                write_summary(summary, output)
                overview = (output / "overview.md").read_text()
                self.assertIn("N/A", overview)
                if falls:
                    self.assertIn("true positives): N/A", overview)
                else:
                    self.assertIn("no labeled falls", overview)

    def test_visual_cli_uses_saved_csv_without_rescoring(self):
        summary = summarize(self.data, self.results, self.chutes)
        output = self.root / "report"
        write_summary(summary, output)
        original = (output / "summary.csv").read_bytes()
        with patch("benchmark_v2.evaluate.evaluate") as evaluate:
            self.assertEqual(visual_main(["--summary-csv", str(output / "summary.csv")]), 0)
        evaluate.assert_not_called()
        self.assertEqual((output / "summary.csv").read_bytes(), original)

    def test_missing_condition_and_mismatch_are_explicit_coverage(self):
        summary = summarize(self.data, self.results, self.chutes, ["cam1", "cam2", "cam3"])
        self.assertEqual(summary["common"], [])
        self.assertEqual(summary["conditions"]["yolo/cam3"]["chute01"]["status"], "missing")
        self.assertEqual(summary["conditions"]["scenescape/combined"]["chute01"]["status"], "failed")

    def test_cli_and_invalid_selections(self):
        self.assertEqual(main(["--data-dir", str(self.data), "--results", str(self.results),
                               "--chutes", *self.chutes]), 0)
        for cameras in ([], ["cam1", "cam1"], ["combined"]):
            with self.assertRaises(ValueError):
                summarize(self.data, self.results, self.chutes, cameras)

    def test_coverage_end_tolerance_forwarded_for_all_conditions_and_missing_results(self):
        (self.results / "yolo/cam1/chute02.json").unlink()
        for tolerance in (0, 0.5):
            options = {"coverage_end_tolerance_s": tolerance} if tolerance else {}
            with self.subTest(tolerance=tolerance), patch(
                    "benchmark_v2.summarize_views.evaluation.evaluate",
                    wraps=evaluation.evaluate) as evaluate:
                summary = summarize(self.data, self.results, self.chutes, **options)
            self.assertEqual(evaluate.call_count, 15)
            self.assertTrue(all(call.kwargs["coverage_end_tolerance_s"] == tolerance
                                for call in evaluate.call_args_list))
            self.assertEqual(summary["conditions"]["yolo/cam1"]["chute02"]["status"], "missing")

    def test_cli_forwards_coverage_end_tolerance(self):
        for tolerance in (0, 0.5):
            options = ["--coverage-end-tolerance-s", str(tolerance)] if tolerance else []
            with self.subTest(tolerance=tolerance), patch(
                    "benchmark_v2.summarize_views.summarize", wraps=summarize) as summary:
                self.assertEqual(main(["--data-dir", str(self.data), "--results", str(self.results),
                                       "--chutes", *self.chutes, *options]), 0)
            self.assertEqual(summary.call_args.kwargs["coverage_end_tolerance_s"], tolerance)

    def test_invalid_coverage_end_tolerance_rejected_before_evaluation(self):
        for tolerance in ("-0.1", "nan", "inf", "-inf"):
            with self.subTest(tolerance=tolerance), patch(
                    "benchmark_v2.summarize_views.evaluation.evaluate") as evaluate:
                with self.assertRaises(ValueError):
                    summarize(self.data, self.results, self.chutes,
                              coverage_end_tolerance_s=float(tolerance))
                with patch("sys.stderr"), self.assertRaises(SystemExit) as raised:
                    main(["--data-dir", str(self.data), "--results", str(self.results),
                          "--coverage-end-tolerance-s=" + tolerance])
                self.assertEqual(raised.exception.code, 2)
                evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()