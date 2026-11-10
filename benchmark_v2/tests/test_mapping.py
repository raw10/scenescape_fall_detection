import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from benchmark_v2.alignment import align_geometry, transform_camera
from benchmark_v2 import detect_videos, mapanything


INTRINSICS = [[500, 0, 320], [0, 510, 240], [0, 0, 1]]


def mapping_result(identifiers):
  return {
    "camera_poses": [{"camera_id": identifier, "translation": [index, 0, 3],
                      "rotation": [0, 0, 0, 1]} for index, identifier in enumerate(identifiers)],
    "intrinsics": [{"camera_id": identifier, "K": INTRINSICS} for identifier in identifiers],
  }


class SharedTransformTests(unittest.TestCase):
  def test_shared_scaled_rotation(self):
    mesh = trimesh.creation.box(extents=[4, 6, 0.1])
    intrinsics = [[500, 0, 320], [0, 510, 240], [0, 0, 1]]
    reference = {"translation": [0, 3, 0], "rotation": [0, 0, 0, 1]}
    other = {"translation": [2, 4, 1],
             "rotation": Rotation.from_euler("z", 35, degrees=True).as_quat()}
    aligned, camera, alignment = align_geometry(
        mesh, reference, intrinsics, 640, 480, 2, [0, 1, 0, 0])
    transform = np.asarray(alignment["mesh_world_to_scene"])
    second = transform_camera(other, intrinsics, 640, 480, transform)
    np.testing.assert_allclose(second["translation"],
                               transform[:3, :3] @ other["translation"] + transform[:3, 3])
    np.testing.assert_allclose(
        Rotation.from_quat(second["rotation"]).as_matrix(),
        transform[:3, :3] / 2 @ Rotation.from_quat(other["rotation"]).as_matrix(),
        atol=1e-12)
    np.testing.assert_allclose(aligned.vertices,
                               trimesh.transform_points(mesh.vertices, transform))
    self.assertEqual(camera, transform_camera(reference, intrinsics, 640, 480, transform))
    self.assertEqual(second["scale"], [1, 1, 1])

  def test_invalid_camera_and_transform(self):
    pose = {"translation": [0, 0, 3], "rotation": [0, 0, 0, 1]}
    for transform in (np.diag([-1, 1, 1, 1]), np.diag([1, 2, 1, 1]), np.zeros((4, 4))):
      with self.subTest(transform=transform), self.assertRaises(ValueError):
        transform_camera(pose, INTRINSICS, 640, 480, transform)
    for invalid in ({**pose, "rotation": [0, 0, 0, 2]},
                    {**pose, "translation": [0, float("nan"), 3]}):
      with self.assertRaises(ValueError):
        transform_camera(invalid, INTRINSICS, 640, 480, np.eye(4))


class MappingTests(unittest.TestCase):
  def test_window_uses_source_time_not_frame_count(self):
    short = [{"timestamp": index * 8333333, "objects": []} for index in range(10)]
    self.assertIsNone(mapanything.select_mapping_frame(short))
    records = [{"timestamp": timestamp, "objects": []}
               for timestamp in (0, 100000000, 250000000, 499999999)]
    self.assertIsNone(mapanything.select_mapping_frame(records))
    records.append({"timestamp": 500000000, "objects": []})
    self.assertEqual(mapanything.select_mapping_frame(records), 2)

  def test_window_rejects_invalid_timestamps(self):
    for timestamps in ([0, 0], [1, 0], [-1], [float("nan")], [float("inf")]):
      with self.subTest(timestamps=timestamps), self.assertRaises(ValueError):
        mapanything.select_mapping_frame([{"timestamp": value, "objects": []}
                                          for value in timestamps])

  def test_mapping_frame_selection(self):
    person = {"detection": {"label": "person"}}
    for objects, expected in (([[], [], [], [person]], 1),
                  ([[person], [], [person], [], [], []], 4),
                  ([[], [person], []], 0),
                              ([[person], [person]], 0)):
      with self.subTest(objects=objects), tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        video = root / "data/chute01/cam2.avi"
        frame = root / "selected.jpg"
        detection_json = json.dumps([{"objects": entries, "timestamp": index * 250000000}
                   for index, entries in enumerate(objects)])

        def run(command, video=video, frame=frame, detection_json=detection_json,
          expected=expected, **_kwargs):
          if command[0] == mapanything.sys.executable:
            self.assertEqual(command[1], str(mapanything.ROOT / "detect_videos.py"))
            self.assertEqual(command[command.index("--input") + 1], str(video.parent.parent))
            self.assertEqual(command[command.index("--chute") + 1], "chute01")
            self.assertEqual(command[command.index("--cameras") + 1], "cam2")
            self.assertEqual(command[command.index("--confidence") + 1], "0.15")
            output = Path(command[command.index("--output") + 1]) / "chute01/cam2.json"
            output.parent.mkdir()
            output.write_text(detection_json)
          else:
            self.assertEqual(command[0], "ffmpeg")
            self.assertEqual(command[command.index("-vf") + 1], f"select=eq(n\\,{expected})")
            self.assertEqual(command[-1], str(frame))
            frame.write_bytes(b"selected-frame")

        with patch.object(mapanything.subprocess, "run", side_effect=run) as commands:
          mapanything.extract_mapping_frame(video, frame)
        self.assertEqual(commands.call_count, 2)
        self.assertEqual(frame.read_bytes(), b"selected-frame")

  def test_empty_detection_output_is_not_a_fallback(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)

      def run(command, **_kwargs):
        output = Path(command[command.index("--output") + 1]) / "chute01/cam2.json"
        output.parent.mkdir()
        output.write_text("[]")

      with patch.object(mapanything.subprocess, "run", side_effect=run) as commands, \
          self.assertRaisesRegex(ValueError, "No frames detected"):
        mapanything.extract_mapping_frame(root / "chute01/cam2.avi", root / "selected.jpg")
      commands.assert_called_once()

  def test_detector_failure_does_not_extract_first_frame(self):
    with patch.object(mapanything.subprocess, "run", side_effect=
                      mapanything.subprocess.CalledProcessError(1, "detector")) as commands, \
        self.assertRaises(mapanything.subprocess.CalledProcessError):
      mapanything.extract_mapping_frame(Path("data/chute01/cam2.avi"), Path("selected.jpg"))
    commands.assert_called_once()

  def test_mapping_model_override_is_forwarded_and_recorded(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      model = root / "yolo11x-pose.xml"

      def run(command, **_kwargs):
        if command[0] == mapanything.sys.executable:
          self.assertEqual(command[command.index("--model") + 1], str(model))
          self.assertEqual(command[command.index("--confidence") + 1], "0.3")
          output = Path(command[command.index("--output") + 1]) / "chute01/cam2.json"
          output.parent.mkdir()
          output.write_text(json.dumps([
              {"timestamp": timestamp, "objects": []} for timestamp in (0, 250000000, 500000000)]))

      with patch.object(mapanything.subprocess, "run", side_effect=run):
        metadata = mapanything.extract_mapping_frame(
            root / "chute01/cam2.avi", root / "selected.jpg", model=model, confidence=0.3)
      self.assertEqual(metadata["model"], str(model))
      self.assertEqual(metadata.get("confidence"), 0.3)
      self.assertEqual(metadata["frame_index"], 1)
      self.assertFalse(metadata["fallback"])

  def test_joint_request_and_saved_shared_mapping(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      source = root / "data" / "chute10"
      source.mkdir(parents=True)
      videos = [source / f"cam{number}.avi" for number in (1, 2)]
      for video in videos:
        video.touch()
      destination = root / "mapped" / source.name
      identifiers = ["id-one", "id-two"]
      result = mapping_result(identifiers)
      result["camera_poses"].reverse()
      result["intrinsics"].reverse()
      raw = trimesh.creation.box(extents=[4, 6, 0.1]).export(file_type="glb")
      response = {**result, "glb_data": base64.b64encode(raw).decode()}
      session = Mock()
      session.post.return_value.json.return_value = {"request_id": "joint"}
      session.get.return_value.json.return_value = {"state": "complete", "result": response}

      def extract(video, frame):
        frame.write_bytes(video.stem.encode())

      def post(*_args, **kwargs):
        self.assertEqual([field for field, _ in kwargs["files"]], ["images", "images"])
        self.assertEqual([part[1].read() for _, part in kwargs["files"]], [b"cam1", b"cam2"])
        self.assertEqual([value for field, value in kwargs["data"] if field == "camera_ids"], identifiers)
        return session.post.return_value

      session.post.side_effect = post
      with patch.object(mapanything, "video_resolution", return_value=(640, 480)), \
          patch.object(mapanything, "extract_mapping_frame", side_effect=extract):
        mapanything.generate_multiview(session, "http://mapping", videos, identifiers, destination, 30)
        with patch.object(mapanything, "align_geometry", wraps=align_geometry) as align:
          mapanything.align_multiview(videos, identifiers, destination, 2, [0, 0, 1, 0])
        self.assertEqual(align.call_count, 1)
        self.assertEqual(align.call_args.args[1]["camera_id"], "id-two")
        cameras = json.loads((destination / "cameras.json").read_text())
        self.assertEqual(list(cameras), ["cam1", "cam2"])
        self.assertEqual(cameras["cam1"]["source_video"], str(videos[0]))
        self.assertEqual(cameras["cam2"]["uid"], "id-two")
        first = (destination / "cameras.json").read_text()
        mapanything.align_multiview(videos, identifiers, destination, 2, [0, 0, 1, 0])
        self.assertEqual(first, (destination / "cameras.json").read_text())
      session.post.assert_called_once()
      self.assertEqual({path.name for path in destination.iterdir()},
                       {"scene.raw.glb", "mapping.json", "scene.glb", "alignment.json", "cameras.json"})
      self.assertEqual({path.name for path in source.iterdir()}, {"cam1.avi", "cam2.avi"})

  def test_ambiguous_or_missing_metadata(self):
    for field in ("camera_poses", "intrinsics"):
      for mutation in ("missing", "duplicate", "unexpected", "extra-frame"):
        result = mapping_result(["one", "two"])
        if mutation == "missing":
          result[field].pop()
        elif mutation == "duplicate":
          result[field][1]["camera_id"] = "one"
        elif mutation == "unexpected":
          result[field][1]["camera_id"] = "other"
        else:
          result[field].append(result[field][0])
        with self.subTest(field=field, mutation=mutation), self.assertRaises(ValueError):
          mapanything.mapping_cameras(result, ["one", "two"])

  def test_invalid_joint_inputs(self):
    session = Mock()
    for videos, identifiers in (([], []), ([Path("one")], []),
                                ([Path("one"), Path("two")], ["same", "same"]),
                                ([Path("one"), Path("one")], ["one", "two"])):
      with self.assertRaises(ValueError):
        mapanything.generate_multiview(session, "unused", videos, identifiers, Path("unused"), 1)
    session.post.assert_not_called()

  def test_invalid_pose_or_empty_frame_does_not_save_mapping(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      video = root / "cam1.avi"
      destination = root / "mapped"
      session = Mock()
      session.post.return_value.json.return_value = {"request_id": "invalid"}
      result = mapping_result(["one"])
      result["camera_poses"][0]["translation"] = [0, float("nan"), 3]
      result["glb_data"] = base64.b64encode(b"glTF").decode()
      session.get.return_value.json.return_value = {"state": "complete", "result": result}
      with patch.object(mapanything, "video_resolution", return_value=(640, 480)), \
          patch.object(mapanything, "extract_mapping_frame", side_effect=lambda _, frame: frame.write_bytes(b"image")):
        with self.assertRaises(ValueError):
          mapanything.generate_multiview(session, "unused", [video], ["one"], destination, 30)
      self.assertFalse(destination.exists())
      session.reset_mock()
      with patch.object(mapanything, "video_resolution", return_value=(640, 480)), \
          patch.object(mapanything, "extract_mapping_frame", side_effect=lambda _, frame: frame.touch()):
        with self.assertRaises(ValueError):
          mapanything.generate_multiview(session, "unused", [video], ["one"], destination, 30)
      session.post.assert_not_called()

  def test_reference_falls_back_to_first_selected_camera(self):
    with tempfile.TemporaryDirectory() as temporary:
      directory = Path(temporary)
      videos = [directory / "cam1.avi", directory / "cam3.avi"]
      identifiers = ["first", "third"]
      (directory / "mapping.json").write_text(json.dumps(mapping_result(identifiers)))
      trimesh.creation.box().export(directory / "scene.raw.glb")
      with patch.object(mapanything, "video_resolution", return_value=(640, 480)), \
          patch.object(mapanything, "align_geometry", wraps=align_geometry) as align:
        mapanything.align_multiview(videos, identifiers, directory, 1, [0, 0, 1, 0])
      self.assertEqual(align.call_args.args[1]["camera_id"], "first")
      self.assertEqual(json.loads((directory / "alignment.json").read_text())["reference_camera"], "cam1")

  def test_cli_legacy_and_multiview_realign(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      source = root / "data" / "chute10"
      source.mkdir(parents=True)
      for name in ("cam1", "cam2"):
        (source / f"{name}.avi").touch()
      base = ["mapanything", "--data-dir", str(root / "data"), "--realign"]
      with patch.object(sys, "argv", base), patch.object(mapanything, "align_scene") as align:
        mapanything.main()
      self.assertEqual(align.call_args.args[0], source / "cam2.avi")
      expected = "bench-" + mapanything.hashlib.sha256(b"chute10/cam2").hexdigest()[:12]
      self.assertEqual(align.call_args.args[1], expected)
      joint = base + ["--chute", "chute10", "--cameras", "all", "--output", str(root / "mapped")]
      with patch.object(sys, "argv", joint), patch.object(mapanything, "align_multiview") as align, \
          patch.object(mapanything.requests, "Session") as session:
        mapanything.main()
      self.assertEqual([video.stem for video in align.call_args.args[0]], ["cam1", "cam2"])
      self.assertEqual(align.call_args.args[2], root / "mapped" / "chute10")
      session.return_value.get.assert_not_called()
      session.return_value.post.assert_not_called()
      for extra in (["--cameras", "cam1"],
                    ["--cameras", "all", "--output", str(root / "data")]):
        with patch.object(sys, "argv", base + extra), self.assertRaises(SystemExit):
          mapanything.main()
      with self.assertRaises(ValueError):
        mapanything.camera_selection(["all", "cam1"])
      with patch.object(sys, "argv", base + ["--cameras", "cam3", "--output", str(root / "mapped")]), \
          patch.object(mapanything.requests, "Session") as session:
        with self.assertRaises(FileNotFoundError):
          mapanything.main()
      session.assert_not_called()


class DetectionSelectionTests(unittest.TestCase):
  def test_selection_outputs_and_explicit_overwrite(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      source = root / "data" / "chute10"
      source.mkdir(parents=True)
      for name in ("cam1", "cam2"):
        (source / f"{name}.avi").touch()
      model = root / "model.xml"
      model.touch()
      output = root / "detections"
      argv = ["detect_videos", "--input", str(root / "data"), "--output", str(output),
              "--model", str(model), "--chute", "chute10", "--cameras", "cam2"]

      def detect(command, **_kwargs):
        self.assertIn("device=CPU", command)
        self.assertIn("location=/data/chute10/cam2.avi", command)
        target = output / "chute10" / "cam2.json"
        self.assertFalse(target.exists())
        target.write_text(json.dumps([{"objects": [
            {"detection": {"label": "person"}}, {"detection": {"label": "chair"}}]}]))

      with patch.object(sys, "argv", argv), patch.object(detect_videos.subprocess, "run", side_effect=detect) as run:
        detect_videos.main()
      self.assertIn("add-empty-results=true", run.call_args.args[0])
      run.assert_called_once()
      self.assertFalse((output / "chute10" / "cam1.json").exists())
      self.assertEqual(len(json.loads((output / "chute10" / "cam2.json").read_text())[0]["objects"]), 1)
      with patch.object(sys, "argv", argv), patch.object(detect_videos.subprocess, "run") as run:
        with self.assertRaises(FileExistsError):
          detect_videos.main()
      run.assert_not_called()
      with patch.object(sys, "argv", argv + ["--overwrite"]), \
          patch.object(detect_videos.subprocess, "run", side_effect=detect) as overwrite:
        detect_videos.main()
      self.assertNotIn("threshold=0.15", overwrite.call_args.args[0])
      with patch.object(sys, "argv", argv + ["--overwrite", "--confidence", "0.15"]), \
          patch.object(detect_videos.subprocess, "run", side_effect=detect) as sensitive:
        detect_videos.main()
      self.assertIn("threshold=0.15", sensitive.call_args.args[0])

  def test_missing_explicit_view_in_any_chute_fails_before_detection(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      for name in ("chute01", "chute10"):
        directory = root / name
        directory.mkdir()
        (directory / "cam1.avi").touch()
      (root / "chute01" / "cam2.avi").touch()
      argv = ["detect_videos", "--input", str(root), "--cameras", "cam1", "cam2"]
      with patch.object(sys, "argv", argv), patch.object(detect_videos.subprocess, "run") as run:
        with self.assertRaises(FileNotFoundError):
          detect_videos.main()
      run.assert_not_called()


if __name__ == "__main__":
  unittest.main()