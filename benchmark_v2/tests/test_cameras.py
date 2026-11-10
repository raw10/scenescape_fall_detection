import tempfile
import unittest
from pathlib import Path

from benchmark_v2.cameras import camera_name, select_videos


class CameraSelectionTests(unittest.TestCase):
    def test_explicit_and_discovered_views(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ("cam1", "cam2", "cam10"):
                (directory / f"{name}.avi").touch()
            self.assertEqual([path.stem for path in select_videos(directory)], ["cam1", "cam2", "cam10"])
            self.assertEqual([path.stem for path in select_videos(directory, ["cam2"])], ["cam2"])
            with self.assertRaises(FileNotFoundError):
                select_videos(directory, ["cam3"])
            with self.assertRaises(ValueError):
                select_videos(directory, ["cam2", "cam2"])

    def test_names_are_not_paths(self):
        for value in ("../cam1", "cam0", "cam2.avi", "camx"):
            with self.assertRaises(ValueError):
                camera_name(value)