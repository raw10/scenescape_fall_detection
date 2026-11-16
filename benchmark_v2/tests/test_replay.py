import argparse
import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchmark_v2 import run_fall_detection as runner
from benchmark_v2.run_fall_detection import merge_frames


class CleanupTests(unittest.TestCase):
  def test_cleanup_is_scoped_and_deletes_cameras_first(self):
    scenes = [{"uid": "old", "name": "benchmark-v2-chute01"},
              {"uid": "fresh", "name": "benchmark-v2-chute01-" + "a" * 20},
              {"uid": "other", "name": "benchmark-v2-chute02"},
              {"uid": "manual", "name": "fall"}]
    cameras = [{"uid": "v2-one", "scene": "old"},
               {"uid": "v2-two", "scene": "fresh"},
               {"uid": "manual", "scene": "manual"}]
    session = Mock()
    with tempfile.TemporaryDirectory() as temporary, \
         patch.object(runner, "list_api", side_effect=[scenes, cameras, scenes[2:], cameras[2:]]):
      runner.cleanup_chute(session, "api", "chute01", Path(temporary))
      evidence = json.loads((Path(temporary) / "cleanup.json").read_text())
    self.assertEqual(evidence["status"], "complete")
    self.assertEqual([call.args[0] for call in session.delete.call_args_list],
                     ["api/camera/v2-one", "api/camera/v2-two", "api/scene/old", "api/scene/fresh"])

  def test_unexpected_camera_blocks_all_deletion(self):
    session = Mock()
    with tempfile.TemporaryDirectory() as temporary, \
         patch.object(runner, "list_api", side_effect=[
             [{"uid": "old", "name": "benchmark-v2-chute01"}],
             [{"uid": "manual", "scene": "old"}]]), \
         self.assertRaisesRegex(RuntimeError, "Unexpected camera"):
      runner.cleanup_chute(session, "api", "chute01", Path(temporary))
    session.delete.assert_not_called()

  def test_delete_failure_is_audited(self):
    session = Mock()
    session.delete.return_value.raise_for_status.side_effect = RuntimeError("delete rejected")
    with tempfile.TemporaryDirectory() as temporary, \
         patch.object(runner, "list_api", side_effect=[
             [{"uid": "old", "name": "benchmark-v2-chute01"}], []]):
      with self.assertRaisesRegex(RuntimeError, "delete rejected"):
        runner.cleanup_chute(session, "api", "chute01", Path(temporary))
      evidence = json.loads((Path(temporary) / "cleanup.json").read_text())
    self.assertEqual(evidence["status"], "failed")
    self.assertEqual(evidence["deleted"], [])


class ChuteLifecycleTests(unittest.TestCase):
  def test_restart_between_chutes(self):
    calls = Mock()
    with patch.object(runner.subprocess, "run") as worker, \
         patch.object(runner, "restart_controller") as restart:
      calls.attach_mock(worker, "worker")
      calls.attach_mock(restart, "restart")
      runner.run_chutes(["docker", "run"], [Path("chute01"), Path("chute02")],
                        Path("results"), "controller", False)
    self.assertEqual([call[0] for call in calls.mock_calls],
                     ["worker", "restart", "restart", "worker", "restart", "restart"])
    self.assertEqual(worker.call_args_list[0].args[0][-2:], ["--chute", "chute01"])
    self.assertEqual(restart.call_args_list[0].args, ("scenescape-web-1", Path("results/chute01")))
    self.assertEqual(restart.call_args_list[0].kwargs, {"service": "web"})
    restart.assert_any_call("controller", Path("results/chute01"))

  def test_failed_worker_restarts_and_stops_before_next_chute(self):
    with patch.object(runner.subprocess, "run", side_effect=runner.subprocess.CalledProcessError(1, "worker")) as worker, \
         patch.object(runner, "restart_controller") as restart, \
         self.assertRaises(runner.subprocess.CalledProcessError):
      runner.run_chutes(["docker", "run"], [Path("chute01"), Path("chute02")],
                        Path("results"), "controller", False)
    worker.assert_called_once()
    self.assertEqual(restart.call_count, 2)
    restart.assert_any_call("scenescape-web-1", Path("results/chute01"), service="web")
    restart.assert_any_call("controller", Path("results/chute01"))

  def test_keep_scenes_skips_restart(self):
    with patch.object(runner.subprocess, "run"), patch.object(runner, "restart_controller") as restart:
      runner.run_chutes(["docker", "run"], [Path("chute01")], Path("results"), "controller", True)
    restart.assert_not_called()

  def test_restart_waits_for_health_and_saves_evidence(self):
    with tempfile.TemporaryDirectory() as temporary, \
         patch.object(runner.subprocess, "run") as restart, \
         patch.object(runner.subprocess, "check_output", side_effect=[
             json.dumps({"Running": True, "Health": {"Status": "starting"}}),
             json.dumps({"Running": True, "Health": {"Status": "healthy"}})]), \
         patch.object(runner.threading.Event, "wait") as wait:
      runner.restart_controller("controller", Path(temporary))
      evidence = json.loads((Path(temporary) / "controller-restart.json").read_text())
    restart.assert_called_once_with(["docker", "restart", "controller"], check=True)
    wait.assert_called_once_with(1)
    self.assertEqual(evidence["status"], "complete")

  def test_web_restart_allows_health_after_two_minutes(self):
    with tempfile.TemporaryDirectory() as temporary, \
         patch.object(runner.subprocess, "run"), \
         patch.object(runner.subprocess, "check_output", side_effect=[
             json.dumps({"Running": True, "Health": {"Status": "starting"}}),
             json.dumps({"Running": True, "Health": {"Status": "healthy"}})]), \
         patch.object(runner.time, "monotonic", side_effect=[0, 180]), \
         patch.object(runner.threading.Event, "wait") as wait:
      runner.restart_controller("web", Path(temporary), service="web")
      evidence = json.loads((Path(temporary) / "web-restart.json").read_text())
    self.assertEqual(evidence["status"], "complete")
    self.assertEqual(evidence["health_timeout_s"], 300)
    wait.assert_called_once_with(1)

  def test_restart_deadlines_and_stopped_container_remain_fatal(self):
    for service, elapsed, running, timeout in [
        ("web", 300, True, 300), ("controller", 120, True, 120),
        ("web", 0, False, 300)]:
      with self.subTest(service=service, running=running), \
           tempfile.TemporaryDirectory() as temporary, \
           patch.object(runner.subprocess, "run"), \
           patch.object(runner.subprocess, "check_output", return_value=json.dumps(
               {"Running": running, "Health": {"Status": "starting"}})), \
           patch.object(runner.time, "monotonic", side_effect=[0, elapsed]), \
           patch.object(runner.threading.Event, "wait") as wait:
        with self.assertRaisesRegex(RuntimeError, "did not become healthy"):
          runner.restart_controller(service, Path(temporary), service=service)
        evidence = json.loads((Path(temporary) / f"{service}-restart.json").read_text())
      self.assertEqual(evidence["status"], "failed")
      self.assertEqual(evidence["health_timeout_s"], timeout)
      wait.assert_not_called()

  def test_restart_failure_stops_next_chute(self):
    with patch.object(runner.subprocess, "run") as worker, \
         patch.object(runner, "restart_controller", side_effect=RuntimeError("unhealthy")), \
         self.assertRaisesRegex(RuntimeError, "unhealthy"):
      runner.run_chutes(["docker", "run"], [Path("chute01"), Path("chute02")],
                        Path("results"), "controller", False)
    worker.assert_called_once()

  def test_custom_web_restart_saves_separate_evidence(self):
    with tempfile.TemporaryDirectory() as temporary, \
         patch.object(runner.subprocess, "run") as restart, \
         patch.object(runner.subprocess, "check_output", return_value=json.dumps(
             {"Running": True, "Health": {"Status": "healthy"}})):
      runner.run_chutes(["docker", "run"], [Path("chute01")], Path(temporary),
                        "controller", False, web="custom-web")
      directory = Path(temporary) / "chute01"
      web = json.loads((directory / "web-restart.json").read_text())
      controller = json.loads((directory / "controller-restart.json").read_text())
    self.assertEqual(web, {"container": "custom-web", "status": "complete", "health_timeout_s": 300})
    self.assertEqual(controller, {"container": "controller", "status": "complete", "health_timeout_s": 120})
    self.assertEqual([call.args[0] for call in restart.call_args_list][1:],
                     [["docker", "restart", "custom-web"], ["docker", "restart", "controller"]])


class MergeTests(unittest.TestCase):
  def test_unequal_fps_delayed_start_empty_frames_and_stable_ties(self):
    streams = {
        "cam2": [{"timestamp": value, "objects": []} for value in (0, 500000000, 1000000000)],
        "cam1": [{"timestamp": value, "objects": []} for value in (500000000, 750000000)],
    }
    merged = merge_frames(streams)
    self.assertEqual([(pts, camera) for pts, camera, _frame in merged], [
        (0, "cam2"), (0.5, "cam1"), (0.5, "cam2"), (0.75, "cam1"), (1, "cam2")])
    self.assertTrue(all(frame["objects"] == [] for _pts, _camera, frame in merged))

  def test_reject_invalid_pts(self):
    for values in ([0, 0], [1, 0], [float("nan")], [float("inf")], [-1], []):
      with self.subTest(values=values), self.assertRaises(ValueError):
        merge_frames({"cam1": [{"timestamp": value} for value in values]})


def camera(name):
  return {"uid": f"mapping-{name}", "translation": [0, 0, 2], "rotation": [0, 0, 0, 1],
          "scale": [1, 1, 1], "intrinsics": {"fx": 100, "fy": 100, "cx": 50, "cy": 50},
          "distortion": {"k1": 0, "k2": 0, "p1": 0, "p2": 0, "k3": 0},
          "resolution": [100, 100], "transform_type": "quaternion",
          "source_video": f"/original/chute01/{name}.avi"}


class ConditionTests(unittest.TestCase):
  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.root = Path(self.temporary.name)
    self.directory = self.root / "data/chute01"
    self.directory.mkdir(parents=True)
    self.mapping = self.root / "mapping/chute01"
    self.mapping.mkdir(parents=True)
    (self.mapping / "scene.glb").write_bytes(b"glTF-test")
    self.cameras = {name: camera(name) for name in ("cam1", "cam2", "cam3")}
    self.cameras["cam1"]["rotation"] = [0, 0, 2 ** -0.5, 2 ** -0.5]
    runner.save_json(self.mapping / "cameras.json", self.cameras)
    self.args = argparse.Namespace(cameras=["all"], mapping_dir=self.mapping.parent,
                                   detections=self.root / "detections", output=self.root / "results")
    cleanup = patch.object(runner, "cleanup_chute")
    self.cleanup_chute = cleanup.start()
    self.addCleanup(cleanup.stop)
    for name in self.cameras:
      (self.directory / f"{name}.avi").touch()
      runner.save_json(self.args.detections / "chute01" / f"{name}.json", [
          {"timestamp": 0, "objects": []}, {"timestamp": 500000000, "objects": []}])
    self.source = {"start_s": 0, "duration_s": 1, "end_s": 1, "frame_count": 2,
                   "resolution": [100, 100], "pts": [0, 0.5]}
    self.probe = patch.object(runner, "probe_video", side_effect=lambda _video: copy.deepcopy(self.source))
    self.probe.start()
    self.addCleanup(self.probe.stop)

  def test_selection_missing_video_detection_and_calibration(self):
    for missing in ("video", "detection", "calibration"):
      with self.subTest(missing=missing):
        self.args.cameras = ["cam1"]
        if missing == "video":
          self.args.cameras = ["cam4"]
        elif missing == "detection":
          (self.args.detections / "chute01/cam1.json").unlink()
        else:
          runner.save_json(self.args.detections / "chute01/cam1.json", [{"timestamp": 0}])
          runner.save_json(self.mapping / "cameras.json", {"cam2": camera("cam2")})
        with self.assertRaises((ValueError, FileNotFoundError)):
          runner.load_condition(self.args, self.directory)

  def test_inactive_cameras_not_registered_fresh_ids_and_shared_calibration(self):
    self.args.cameras = ["cam1"]
    condition = runner.load_condition(self.args, self.directory)
    self.assertEqual(condition["per_camera"]["cam1"]["source_end_s"], 1)
    self.assertEqual(condition["per_camera"]["cam1"]["last_detection_pts_s"], 0.5)
    calls = []
    registered = {}

    def manager(_session, method, url, **kwargs):
      calls.append((method, url, kwargs))
      if method == "POST" and url.endswith("/scene"):
        return {"uid": f"scene-{len(calls)}"}
      if method == "POST" and url.endswith("/camera"):
        payload = kwargs["json"]
        registered[payload["sensor_id"]] = payload
        return payload
      if method == "GET" and "/camera/" in url:
        return {**registered[url.rsplit("/", 1)[1]], "rotation": [0, 0, 90], "transform_type": "euler"}
      return {}

    with patch.object(runner, "api", side_effect=manager):
      first = runner.register_condition(Mock(), "api", self.directory, condition,
                                        self.args.output / "first", "run1")
      second = runner.register_condition(Mock(), "api", self.directory, condition,
                                         self.args.output / "second", "run2")
    self.assertEqual(list(first["camera_ids"]), ["cam1"])
    self.assertEqual(first["reference_camera"], "cam1")
    self.assertNotEqual(first["scene_name"], second["scene_name"])
    self.assertNotEqual(first["camera_id"], second["camera_id"])
    self.assertLessEqual(len(first["camera_id"]), 20)
    self.assertFalse((self.directory / "registration.json").exists())
    payload = registered[first["camera_id"]]
    for key in ("translation", "rotation", "scale", "intrinsics", "distortion", "resolution"):
      self.assertEqual(payload[key], self.cameras["cam1"][key])
    transforms = [kwargs["json"] for method, url, kwargs in calls
                  if method == "POST" and "/scene/" in url]
    self.assertEqual(transforms[0], {"mesh_translation": [0, 0, 0],
                                     "mesh_rotation": [0, 0, 0], "mesh_scale": [1, 1, 1]})
    self.args.cameras = ["all"]
    combined = runner.load_condition(self.args, self.directory)
    self.assertEqual(combined["reference_camera"], "cam2")
    self.assertEqual(combined["mapping_fingerprint"], condition["mapping_fingerprint"])

  def test_duplicate_calibration_uids_fail(self):
    self.cameras["cam2"]["uid"] = self.cameras["cam1"]["uid"]
    runner.save_json(self.mapping / "cameras.json", self.cameras)
    with self.assertRaisesRegex(ValueError, "duplicate"):
      runner.load_condition(self.args, self.directory)

  def test_invalid_calibration_and_wrong_video_fail(self):
    self.args.cameras = ["cam1"]
    for key, value in (("translation", [0, 0]), ("rotation", [0, 0, 0, 0]),
                       ("intrinsics", {}), ("resolution", [200, 200]),
                       ("source_video", "/original/chute01/cam2.avi")):
      invalid = camera("cam1")
      invalid[key] = value
      runner.save_json(self.mapping / "cameras.json", {"cam1": invalid})
      with self.subTest(key=key), self.assertRaises(ValueError):
        runner.load_condition(self.args, self.directory)

  def test_camera_readback_mismatch_fails(self):
    self.args.cameras = ["cam1"]
    condition = runner.load_condition(self.args, self.directory)
    actual = {**camera("cam1"), "scene": "fresh", "translation": [9, 9, 9]}
    with patch.object(runner, "api", side_effect=[{"uid": "fresh"}, {}, {}, actual]), \
         self.assertRaisesRegex(RuntimeError, "readback mismatch for translation"):
      runner.register_condition(Mock(), "api", self.directory, condition, self.args.output, "new")

  def test_incomplete_camera_pose_readback_retries(self):
    self.args.cameras = ["cam1"]
    condition = runner.load_condition(self.args, self.directory)
    actual = {**self.cameras["cam1"], "scene": "fresh"}
    for key in ("translation", "rotation", "scale"):
      with self.subTest(key=key), \
           patch.object(runner, "api", side_effect=[
               {"uid": "fresh"}, {}, {}, {**actual, key: None}, actual]) as manager, \
           patch.object(runner.threading, "Event") as event:
        result = runner.register_condition(Mock(), "api", self.directory, condition,
                                           self.args.output, "new")
        self.assertIn("cam1", result["camera_ids"])
        self.assertEqual(manager.call_count, 5)
        self.assertEqual(event.return_value.wait.call_count, 3)
        self.assertTrue(all(call.args == (1,) for call in event.return_value.wait.call_args_list))
        evidence = json.loads((self.args.output / "cam1-readback.json").read_text())
        self.assertIsNone(evidence["response"][key])

  def test_incomplete_camera_pose_readback_is_bounded(self):
    self.args.cameras = ["cam1"]
    condition = runner.load_condition(self.args, self.directory)
    actual = {**self.cameras["cam1"], "scene": "fresh", "translation": None}
    with patch.object(runner, "api", side_effect=[{"uid": "fresh"}, {}, {}] + [actual] * 10), \
         patch.object(runner.threading, "Event") as event, \
         self.assertRaisesRegex(RuntimeError, "pose still incomplete after 10 readbacks"):
      runner.register_condition(Mock(), "api", self.directory, condition, self.args.output, "new")
    self.assertEqual(event.return_value.wait.call_count, 11)
    evidence = json.loads((self.args.output / "cam1-readback.json").read_text())
    self.assertEqual(evidence["attempt"], 10)

  def test_camera_creation_and_readback_have_settle_delays(self):
    self.args.cameras = ["cam1"]
    condition = runner.load_condition(self.args, self.directory)
    actual = {**self.cameras["cam1"], "scene": "fresh"}
    operations = []

    def manager(_session, method, url, **_kwargs):
      operations.append((method, url))
      if method == "GET":
        return actual
      return {"uid": "fresh"} if url == "api/scene" else {}

    with patch.object(runner, "api", side_effect=manager), \
         patch.object(runner.threading, "Event") as event:
      event.return_value.wait.side_effect = lambda seconds: operations.append(("wait", seconds))
      result = runner.register_condition(Mock(), "api", self.directory, condition,
                                         self.args.output, "new")
    camera_url = f"api/camera/{result['camera_id']}"
    self.assertEqual(operations, [
        ("POST", "api/scene"), ("POST", "api/scene/fresh"), ("wait", 1),
        ("POST", "api/camera"), ("wait", 1), ("GET", camera_url)])

  def test_source_mismatch_fails(self):
    for frames in ([{"timestamp": 0}], [{"timestamp": 100000000}, {"timestamp": 600000000}]):
      runner.save_json(self.args.detections / "chute01/cam1.json", frames)
      with self.subTest(frames=frames), self.assertRaisesRegex(ValueError, "frame counts|PTS differ"):
        runner.load_condition(self.args, self.directory)

  def test_host_source_metadata_used_without_worker_ffprobe(self):
    source_metadata = self.root / "source-metadata.json"
    self.args.source_metadata = source_metadata
    runner.save_json(source_metadata, {"chute01": {name: self.source for name in self.cameras}})
    with patch.object(runner, "probe_video", side_effect=AssertionError("No ffprobe in runtime image")):
      condition = runner.load_condition(self.args, self.directory)
    self.assertEqual(condition["selected_cameras"], ["cam1", "cam2", "cam3"])

  def test_mapping_change_changes_condition_identity(self):
    self.args.cameras = ["cam1"]
    first = runner.load_condition(self.args, self.directory)
    (self.mapping / "scene.glb").write_bytes(b"glTF-changed")
    second = runner.load_condition(self.args, self.directory)
    self.assertNotEqual(first["mapping_fingerprint"], second["mapping_fingerprint"])

  def test_worker_continues_failed_chutes_and_persists_results(self):
    second = self.directory.parent / "chute02"
    second.mkdir()
    self.args.data_dir = self.directory.parent
    self.args.chute = None
    self.args.ca = self.root / "ca"
    self.args.url = "api"
    self.args.speed = 0.25
    self.args.tracking_radius_m = None
    self.args.exclusive_stack = False
    condition = {"selected_cameras": ["cam1"], "reference_camera": "cam1",
                 "mapping_fingerprint": "hash", "per_camera": {}}

    def replay(args, directory, _registration, _token, _active, _radius):
      runner.save_json(args.output / f"{directory.name}.json", {"status": "complete", "events": []})

    with patch.object(runner, "load_condition", side_effect=[ValueError("missing camera"), condition]), \
         patch.object(runner, "register_condition", return_value={"scene_id": "new"}) as register, \
         patch.object(runner, "replay", side_effect=replay), \
         patch.object(runner.requests, "Session"), \
         patch.dict(runner.os.environ, {"SCENESCAPE_API_KEY": "test"}):
      self.assertEqual(runner.worker(self.args), 1)
    register.assert_called_once()
    self.assertEqual(self.cleanup_chute.call_count, 2)
    self.assertEqual(json.loads((self.args.output / "chute01.json").read_text())["status"], "failed")
    self.assertEqual(json.loads((self.args.output / "chute02.json").read_text())["status"], "complete")

  def test_registration_settle_delay_before_radius_and_replay(self):
    self.args.data_dir = self.directory.parent
    self.args.chute = "chute01"
    self.args.ca = self.root / "ca"
    self.args.url = "api"
    self.args.speed = 1
    self.args.tracking_radius_m = None
    self.args.exclusive_stack = False
    for name in ("scene.glb", "camera.json"):
      (self.directory / name).touch()
    condition = {"selected_cameras": ["cam1"], "reference_camera": "cam1",
                 "mapping_fingerprint": "hash", "per_camera": {}}

    def replay(args, directory, registration, _token, _condition, _radius):
      runner.save_json(args.output / f"{directory.name}.json",
                       {**registration, "status": "complete", "events": []})

    for joint in (False, True):
      for settle in (None, 0.0, 1.25):
        with self.subTest(joint=joint, settle=settle):
          self.args.mapping_dir = self.mapping.parent if joint else None
          self.args.output = self.root / f"settle-{joint}-{settle}"
          if settle is not None:
            self.args.registration_settle_s = settle
          elif hasattr(self.args, "registration_settle_s"):
            del self.args.registration_settle_s
          calls = Mock()
          register_name = "register_condition" if joint else "register_scene"
          with patch.object(runner, "load_condition", return_value=condition), \
               patch.object(runner, register_name, return_value={"scene_id": "new"}) as register, \
               patch.object(runner.threading.Event, "wait") as wait, \
               patch.object(runner, "tracking_radius", wraps=runner.tracking_radius) as radius, \
               patch.object(runner, "replay", side_effect=replay) as replay_mock, \
               patch.object(runner.requests, "Session"), \
               patch.dict(runner.os.environ, {"SCENESCAPE_API_KEY": "test"}):
            for name, mock in (("register", register), ("wait", wait),
                               ("radius", radius), ("replay", replay_mock)):
              calls.attach_mock(mock, name)
            self.assertEqual(runner.worker(self.args), 0)
          expected = ["register", "wait", "radius", "replay"] if settle else ["register", "radius", "replay"]
          self.assertEqual([call[0] for call in calls.mock_calls], expected)
          if settle:
            wait.assert_called_once_with(settle)
          for path in (self.args.output / "chute01.json",
                       self.args.output / "chute01/registration.json"):
            self.assertEqual(json.loads(path.read_text())["registration_settle_s"], settle or 0)

  def test_worker_preserves_existing_joint_results(self):
    self.args.data_dir = self.directory.parent
    self.args.chute = "chute01"
    result_path = self.args.output / "chute01.json"
    runner.save_json(result_path, {"status": "complete", "evidence": "original"})
    with patch.object(runner.requests, "Session") as session:
      with self.assertRaises(FileExistsError):
        runner.worker(self.args)
    session.assert_not_called()
    self.assertEqual(json.loads(result_path.read_text())["evidence"], "original")

  def test_worker_cleanup_failure_is_fatal(self):
    self.args.data_dir = self.directory.parent
    self.args.chute = "chute01"
    self.args.ca = self.root / "ca"
    self.args.url = "api"
    self.args.speed = 0.25
    self.args.tracking_radius_m = None
    self.args.exclusive_stack = False
    self.cleanup_chute.side_effect = RuntimeError("delete failed")
    with patch.object(runner, "load_condition", side_effect=ValueError("invalid input")), \
         patch.object(runner.requests, "Session"), \
         patch.dict(runner.os.environ, {"SCENESCAPE_API_KEY": "test"}), \
         self.assertRaisesRegex(RuntimeError, "delete failed"):
      runner.worker(self.args)
    result = json.loads((self.args.output / "chute01.json").read_text())
    self.assertEqual(result["status"], "failed")
    self.assertEqual(result["cleanup_error"], "delete failed")

  def test_worker_keep_scenes_skips_cleanup(self):
    self.args.data_dir = self.directory.parent
    self.args.chute = "chute01"
    self.args.ca = self.root / "ca"
    self.args.url = "api"
    self.args.speed = 0.25
    self.args.tracking_radius_m = None
    self.args.exclusive_stack = False
    self.args.keep_scenes = True
    with patch.object(runner, "load_condition", side_effect=ValueError("invalid input")), \
         patch.object(runner.requests, "Session"), \
         patch.dict(runner.os.environ, {"SCENESCAPE_API_KEY": "test"}):
      self.assertEqual(runner.worker(self.args), 1)
    self.cleanup_chute.assert_not_called()

  def test_worker_refuses_later_replays_after_unsafe_radius_restore(self):
    (self.directory.parent / "chute02").mkdir()
    self.args.data_dir = self.directory.parent
    self.args.chute = None
    self.args.ca = self.root / "ca"
    self.args.url = "api"
    self.args.speed = 0.25
    self.args.tracking_radius_m = 3
    self.args.exclusive_stack = True
    condition = {"selected_cameras": ["cam1"], "reference_camera": "cam1",
                 "mapping_fingerprint": "hash", "per_camera": {}}

    def unsafe(_session, _base, _radius, _exclusive, path, _registration):
      runner.save_json(path, {"status": "restore_failed", "restore_error": "REST restore failed"})
      raise RuntimeError("restore failed")

    with patch.object(runner, "load_condition", return_value=condition), \
         patch.object(runner, "register_condition", return_value={"scene_id": "new"}) as register, \
         patch.object(runner, "tracking_radius", side_effect=unsafe), \
         patch.object(runner, "replay") as replay, patch.object(runner.requests, "Session"), \
         patch.dict(runner.os.environ, {"SCENESCAPE_API_KEY": "test"}):
      self.assertEqual(runner.worker(self.args), 1)
    register.assert_called_once()
    replay.assert_not_called()
    result = json.loads((self.args.output / "chute02.json").read_text())
    self.assertEqual(result["status"], "failed")
    self.assertIn("manual recovery", result["error"])


class ProbeTests(unittest.TestCase):
  def probe(self, start="0", duration="1", pts=("0", "0.5")):
    document = {"streams": [{"start_time": start, "duration": duration, "width": 100, "height": 100}],
                "frames": [{"best_effort_timestamp_time": value} for value in pts]}
    with patch.object(runner.subprocess, "check_output", return_value=json.dumps(document)):
      return runner.probe_video(Path("cam1.avi"))

  def test_common_source_zero_and_unequal_frame_rates(self):
    self.assertEqual(self.probe()["pts"], [0, 0.5])
    self.assertEqual(self.probe(pts=("0", "0.25", "0.5"))["frame_count"], 3)

  def test_invalid_source_pts_duration_and_origin(self):
    for kwargs in ({"start": "1"}, {"duration": "nan"}, {"duration": "0"},
                   {"pts": ("0", "0")}, {"pts": ("0", "nan")},
                   {"pts": ("0", "2")}, {"pts": ()}, {"pts": ("0.1", "0.5")}):
      with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
        self.probe(**kwargs)

  def test_orientation_equivalence_matches_manager_xyz_convention(self):
    for rotation in ([30, 45, 80], [-90, 0, 0], [0, 90, 0]):
      matrix = runner.orientation_matrix(rotation)
      self.assertTrue(runner.np.allclose(matrix.T @ matrix, runner.np.eye(3)))
    self.assertTrue(runner.np.allclose(runner.orientation_matrix([0, 0, 90]),
                                     runner.orientation_matrix([0, 0, 1, 1])))


class RadiusTests(unittest.TestCase):
  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.path = Path(self.temporary.name) / "radius.json"
    self.asset = {"uid": "42", "name": "person", "tracking_radius": 2.0}
    self.puts = []
    self.fail_update = False
    self.fail_restore = False

    def manager(_session, method, url, **kwargs):
      if url == "api/assets":
        return {"results": [self.asset.copy(), {"uid": "99", "name": "vehicle"}]}
      if method == "PUT":
        value = kwargs["json"]["tracking_radius"]
        self.puts.append(value)
        if value == 2 and self.fail_restore:
          raise RuntimeError("restore HTTP failure")
        self.asset["tracking_radius"] = value
        if value == 3 and self.fail_update:
          raise RuntimeError("ambiguous PUT failure")
      return self.asset.copy()

    self.manager = patch.object(runner, "api", side_effect=manager)
    self.api = self.manager.start()
    self.addCleanup(self.manager.stop)

  def context(self, requested=3.0, exclusive=True):
    return runner.tracking_radius(Mock(), "api", requested, exclusive, self.path, {"scene_id": "fresh"})

  def test_no_override_leaves_assets_alone(self):
    with self.context(None) as evidence:
      self.assertFalse(evidence["changed"])
    self.api.assert_not_called()

  def test_unchanged_value_needs_no_exclusive_confirmation(self):
    with self.context(2, False) as evidence:
      self.assertEqual(evidence["effective_m"], 2)
    self.assertEqual(self.puts, [])

  def test_change_requires_explicit_exclusive_stack(self):
    with self.assertRaisesRegex(ValueError, "exclusive-stack"), self.context(exclusive=False):
      pass
    self.assertEqual(self.puts, [])

  def test_invalid_radius_and_ambiguous_category(self):
    for requested in (0, -1, float("nan"), float("inf")):
      with self.subTest(requested=requested), self.assertRaises(ValueError), self.context(requested):
        pass
    self.asset["name"] = "Person"
    with self.assertRaisesRegex(RuntimeError, "exactly one"), self.context():
      pass

  def test_update_readback_and_restore_after_success(self):
    with patch.object(runner, "verify_radius_activation") as activation:
      with self.context() as evidence:
        before = json.loads(self.path.read_text())
        self.assertEqual(before["asset_uid"], "42")
        self.assertEqual(before["original_m"], 2)
        self.assertEqual(self.asset["tracking_radius"], 3)
        self.assertTrue(evidence["activation_verified"])
    activation.assert_called_once()
    self.assertEqual(self.puts, [3, 2])
    self.assertEqual(json.loads(self.path.read_text())["status"], "restored")

  def test_restore_on_exception_and_keyboard_interrupt(self):
    for error in (ValueError("replay error"), KeyboardInterrupt()):
      with self.subTest(error=type(error).__name__), \
           patch.object(runner, "verify_radius_activation"), \
           self.assertRaises(type(error)), self.context():
        raise error
      self.assertEqual(self.asset["tracking_radius"], 2)

  def test_real_activation_check_fails_closed_and_restores(self):
    with self.assertRaisesRegex(RuntimeError, "Cannot verify Controller"), self.context():
      self.fail("Unverified radius must never reach replay")
    self.assertEqual(self.asset["tracking_radius"], 2)
    self.assertIn("hard_kill_caveat", json.loads(self.path.read_text()))

  def test_ambiguous_put_failure_still_restores(self):
    self.fail_update = True
    with self.assertRaisesRegex(RuntimeError, "ambiguous PUT"), self.context():
      pass
    self.assertEqual(self.asset["tracking_radius"], 2)

  def test_external_change_not_clobbered(self):
    with patch.object(runner, "verify_radius_activation"), \
         self.assertRaisesRegex(RuntimeError, "externally"), self.context():
      self.asset["tracking_radius"] = 9
    self.assertEqual(self.asset["tracking_radius"], 9)
    self.assertEqual(self.puts, [3])
    self.assertEqual(json.loads(self.path.read_text())["status"], "restore_conflict")

  def test_restore_failure_is_prominent(self):
    self.fail_restore = True
    with patch.object(runner, "verify_radius_activation"), \
         self.assertRaisesRegex(RuntimeError, "restore HTTP"), self.context():
      pass
    self.assertEqual(json.loads(self.path.read_text())["status"], "restore_failed")

  def test_update_readback_mismatch_does_not_claim_activation(self):
    original = self.api.side_effect
    updated = False

    def manager(session, method, url, **kwargs):
      nonlocal updated
      response = original(session, method, url, **kwargs)
      if method == "PUT" and kwargs["json"]["tracking_radius"] == 3:
        updated = True
      elif method == "GET" and updated and url == "api/asset/42":
        updated = False
        return {**response, "tracking_radius": 2}
      return response

    self.api.side_effect = manager
    with patch.object(runner, "verify_radius_activation") as activation, \
         self.assertRaisesRegex(RuntimeError, "update readback failed"), self.context():
      pass
    activation.assert_not_called()
    self.assertEqual(self.asset["tracking_radius"], 2)


class ReplayTests(unittest.TestCase):
  def run_replay(self, observed=True, legacy=False, publish_ok=True):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    root = Path(temporary.name)
    args = argparse.Namespace(output=root / "output", detections=root / "detections",
                              auth=root / "auth", ca=root / "ca", broker="mock", url="api",
                              speed=0.25, drain=0)
    runner.save_json(args.auth, {"user": "test", "password": "not-a-real-secret"})
    person = {"x": 1, "y": 2, "w": 3, "h": 4,
              "detection": {"label": "person", "confidence": 0.9}}
    streams = {"cam2": [{"timestamp": value, "objects": [] if value == 0 else [person]}
                        for value in (0, 500000000, 1000000000)],
               "cam1": [{"timestamp": value, "objects": []} for value in (0, 750000000)]}
    registration = {"scene_id": "fresh", "camera_id": "id2", "camera_ids": {"cam1": "id1", "cam2": "id2"},
            "registration_settle_s": 1.25}
    condition = {"streams": streams, "per_camera": {
        name: {"input_records": len(frames), "published_records": 0, "published": False}
        for name, frames in streams.items()}}
    if legacy:
      registration.pop("camera_ids")
      runner.save_json(args.detections / "chute01/cam2.json", streams["cam2"])
      condition = None
    sent = []
    client = Mock()
    process = Mock()
    process.poll.return_value = None

    def incoming(topic, payload):
      client.on_message(client, None, SimpleNamespace(topic=topic, retain=False,
                                                    payload=json.dumps(payload).encode()))

    def publish(topic, payload, qos):
      self.assertEqual(qos, 1)
      document = json.loads(payload)
      if topic.startswith("scenescape/data/scene/"):
        incoming("scenescape/fall-detection/fresh", document)
      else:
        sent.append((topic, document))
        objects = [{"camera_bounds": {document["id"]: {"projected": False}}}] if observed else []
        incoming("scenescape/data/scene/fresh/person", {"objects": objects})
        incoming("scenescape/fall-detection/fresh", {
            "timestamp": document["timestamp"], "people": [{"uuid": "shared", "state": "fallen"}]})
      result = Mock(rc=0)
      result.is_published.return_value = publish_ok
      return result

    client.publish.side_effect = publish
    client.subscribe.side_effect = lambda _topics: client.on_subscribe(client, None, 1, [], None)
    client.loop_start.side_effect = lambda: client.on_connect(client, None, None, 0, None)
    mqtt = SimpleNamespace(Client=Mock(return_value=client), CallbackAPIVersion=SimpleNamespace(VERSION2=2),
                           MQTT_ERR_SUCCESS=0)
    with ExitStack() as stack:
      stack.enter_context(patch.object(runner, "mqtt", mqtt))
      stack.enter_context(patch.object(runner.subprocess, "Popen", return_value=process))
      stack.enter_context(patch.object(runner.time, "time", return_value=1000))
      stack.enter_context(patch.object(runner.time, "monotonic", return_value=500))
      stack.enter_context(patch.object(runner.threading.Event, "wait", return_value=True))
      error = None
      try:
        runner.replay(args, Path("chute01"), registration, "token", condition)
      except RuntimeError as caught:
        error = caught
    return json.loads((args.output / "chute01.json").read_text()), sent, mqtt, client, process, error

  def test_one_client_shared_origin_empty_frames_and_single_transition(self):
    result, sent, mqtt, client, process, error = self.run_replay()
    self.assertIsNone(error)
    mqtt.Client.assert_called_once()
    self.assertEqual([document["id"] for _topic, document in sent], ["id1", "id2", "id2", "id1", "id2"])
    pts = [(datetime.fromisoformat(document["timestamp"].replace("Z", "+00:00")).timestamp() - 1000) * 0.25
           for _topic, document in sent]
    self.assertEqual(pts, [0, 0, 0.5, 0.75, 1])
    self.assertEqual(sent[0][1]["objects"], {"person": []})
    self.assertEqual(len(result["events"]), 1)
    self.assertEqual(result["reference_camera"], "cam2")
    self.assertEqual(result["registration_settle_s"], 1.25)
    self.assertEqual(result["observed_camera_bounds"], {"cam1": 2, "cam2": 3})
    self.assertTrue(all(value["published"] for value in result["per_camera"].values()))
    client.disconnect.assert_called_once()
    process.terminate.assert_called_once()

  def test_no_observed_people_fails(self):
    result, _sent, _mqtt, _client, _process, error = self.run_replay(observed=False)
    self.assertIsNotNone(error)
    self.assertEqual(result["status"], "failed")

  def test_publication_failure_persists_partial_evidence(self):
    result, _sent, _mqtt, _client, _process, error = self.run_replay(publish_ok=False)
    self.assertIn("publication failed", str(error))
    self.assertEqual(result["status"], "failed")
    self.assertFalse(result["per_camera"]["cam1"]["published"])

  def test_legacy_cam2_without_new_arguments(self):
    result, sent, _mqtt, _client, _process, error = self.run_replay(legacy=True)
    self.assertIsNone(error)
    self.assertEqual(result["selected_cameras"], ["cam2"])
    self.assertEqual(len(sent), 3)


class CliTests(unittest.TestCase):
  def test_worker_forwarding_and_readonly_mounts(self):
    metadata = [{"NetworkSettings": {"Networks": {"network": {"Aliases": ["web.scenescape.intel.com"]}}},
                 "Mounts": [{"Source": "/tmp/secrets/browser.auth", "Destination": "/run/secrets/browser.auth"}]}]
    with tempfile.TemporaryDirectory() as output, \
         patch.object(runner.sys, "argv", ["replay", "--output", output, "--mapping-dir", "/tmp/mapping",
                                           "--data-dir", output + "/data",
                                           "--cameras", "cam1", "cam2", "--tracking-radius-m", "3",
                                           "--exclusive-stack", "--speed", "0.25", "--chute", "chute01",
                                           "--registration-settle-s", "1.25"]), \
                           patch.object(runner.subprocess, "check_output", side_effect=[
                             json.dumps(metadata), json.dumps({"Running": True, "Health": {"Status": "healthy"}}),
                             json.dumps({"Running": True, "Health": {"Status": "healthy"}})]), \
        patch.object(runner, "probe_video", return_value={"pts": [0], "duration_s": 1}) as probe, \
         patch.object(runner.subprocess, "run") as docker:
      directory = Path(output) / "data/chute01"
      directory.mkdir(parents=True)
      (directory / "cam1.avi").touch()
      (directory / "cam2.avi").touch()
      runner.main()
      self.assertEqual(probe.call_count, 2)
      self.assertTrue((Path(output) / "source-metadata.json").is_file())
    command = docker.call_args_list[0].args[0]
    self.assertIn("/tmp/mapping:/mapping:ro", command)
    self.assertTrue(any(value.endswith(":/data:ro") for value in command))
    self.assertEqual(command[command.index("--cameras") + 1:command.index("--cameras") + 3], ["cam1", "cam2"])
    self.assertEqual(command[command.index("--mapping-dir") + 1], "/mapping")
    self.assertEqual(command[command.index("--tracking-radius-m") + 1], "3.0")
    self.assertEqual(command[command.index("--registration-settle-s") + 1], "1.25")
    self.assertIn("--exclusive-stack", command)
    self.assertEqual(command[command.index("--source-metadata") + 1], "/results/source-metadata.json")

  def test_invalid_options_fail_before_docker(self):
    for options in (["--tracking-radius-m", "nan"], ["--tracking-radius-m", "0"],
                    ["--registration-settle-s", "nan"], ["--registration-settle-s", "inf"],
                    ["--registration-settle-s=-inf"], ["--registration-settle-s", "-1"],
                    ["--cameras", "cam1"], ["--mapping-dir", "/tmp", "--cameras", "all", "cam1"]):
      with self.subTest(options=options), patch.object(runner.sys, "argv", ["replay", *options]), \
           patch.object(runner.subprocess, "check_output") as docker, \
           patch("sys.stderr"), self.assertRaises(SystemExit):
        runner.main()
      docker.assert_not_called()


if __name__ == "__main__":
  unittest.main()