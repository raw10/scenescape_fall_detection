import argparse
import importlib
import sys
import unittest
from unittest.mock import patch

from benchmark_v2 import detect_videos, run_fall_detection


class DefaultsTests(unittest.TestCase):
    def defaults(self, module):
        original = argparse.ArgumentParser.parse_args
        captured = []

        def capture(parser, *_args, **_kwargs):
            captured.append(original(parser, []))
            raise SystemExit

        with patch.object(argparse.ArgumentParser, "parse_args", capture):
            with self.assertRaises(SystemExit):
                module.main()
        return captured[0]

    def test_detection_defaults(self):
        args = self.defaults(detect_videos)
        self.assertEqual(args.model.name, "yolo11s-pose.xml")
        self.assertTrue(args.model.is_file())
        self.assertEqual(args.output.name, "detections-yolo11s-pose")

    def test_replay_defaults(self):
        args = self.defaults(run_fall_detection)
        self.assertEqual(args.detections.name, "detections-yolo11s-pose")
        self.assertEqual(args.output.name, "results-yolo11s-pose")

    def test_overlay_defaults(self):
        with patch.dict(sys.modules, {"detect_videos": detect_videos}):
            module = importlib.import_module("benchmark_v2.overlay_detections")
        args = self.defaults(module)
        self.assertEqual(args.detections.name, "detections-yolo11s-pose")
        self.assertEqual(args.output.name, "overlays-yolo11s-pose")


if __name__ == "__main__":
    unittest.main()