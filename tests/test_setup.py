import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def load_setup():
    spec = importlib.util.spec_from_file_location("setup_under_test", ROOT / "setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAMERA = {
    "uid": "lawn-cam1",
    "name": "lawn-cam1",
    "extrinsics": {
        "translation": [1, 2, 3],
        "rotation": [4, 5, 6],
        "scale": [1, 1, 1],
    },
    "intrinsics": {"fx": 100, "fy": 100, "cx": 50, "cy": 40},
    "distortion": {"k1": 0, "k2": 0, "p1": 0, "p2": 0, "k3": 0},
    "resolution": [100, 80],
}


class SetupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_setup()

    def response(self, data=None):
        response = Mock()
        response.json.return_value = data
        response.raise_for_status.return_value = None
        return response

    def test_compose_command_uses_current_example_before_app_override(self):
        self.assertEqual(self.module.compose_command(
            "/scenescape", "/app/override.yml", "/app/app.env", "up", "-d"), [
                "docker", "compose",
                "--project-directory", "/scenescape",
                "--env-file", "/app/app.env",
                "-f", "/scenescape/sample_data/compose/compose.dl-streamer-example.yml",
                "-f", "/app/override.yml",
                "--profile", "controller", "up", "-d",
            ])

    def test_start_manager_waits_for_docker_health_without_host_http_probe(self):
        with patch.object(self.module.subprocess, "run") as run, \
                patch.object(self.module, "session") as session:
            self.module.start_manager("/scenescape", "/app/override.yml", "/app/app.env")

        run.assert_called_once_with(self.module.compose_command(
            "/scenescape", "/app/override.yml", "/app/app.env",
            "up", "-d", "--wait", "--wait-timeout", "180", "web"), check=True)
        session.get.assert_not_called()

    def test_start_manager_propagates_health_wait_failure(self):
        error = self.module.subprocess.CalledProcessError(1, ["docker", "compose"])
        with patch.object(self.module.subprocess, "run", side_effect=error), \
                patch.object(self.module, "session") as session:
            with self.assertRaises(self.module.subprocess.CalledProcessError):
                self.module.start_manager("/scenescape", "/app/override.yml", "/app/app.env")

        session.get.assert_not_called()

    def test_certificate_hostname_uses_environment_and_default(self):
        for env, expected in (({"SCENESCAPE_CERT_HOSTNAME": "custom.example.net"},
                               "custom.example.net"), ({}, "web.scenescape.intel.com")):
            with self.subTest(expected=expected), patch.dict(os.environ, env, clear=True):
                module = load_setup()
                with patch.object(module.requests.adapters.HTTPAdapter, "init_poolmanager") as init:
                    module.SceneScapeHTTPSAdapter().init_poolmanager(1, 1)
                self.assertEqual(init.call_args.kwargs["assert_hostname"], expected)
                self.assertEqual(init.call_args.kwargs["server_hostname"], expected)

    def test_add_camera_creates_with_full_calibration(self):
        with patch.object(self.module, "session") as session:
            session.get.return_value = self.response({"results": []})
            session.post.return_value = self.response()
            created_uid = self.module.add_camera(
                "https://manager/api/v1/", "key", "scene", CAMERA)

        self.assertEqual(created_uid, "lawn-cam1")
        payload = session.post.call_args.kwargs["json"]
        self.assertEqual(payload["sensor_id"], "lawn-cam1")
        self.assertEqual(payload["intrinsics"], CAMERA["intrinsics"])
        self.assertNotIn("distortion", payload)
        self.assertNotIn("resolution", payload)
        session.put.assert_not_called()

    def test_add_camera_rejects_nonzero_distortion(self):
        camera = {**CAMERA, "distortion": {**CAMERA["distortion"], "k1": 0.1}}
        with patch.object(self.module, "session") as session:
            with self.assertRaisesRegex(ValueError, "nonzero distortion"):
                self.module.add_camera(
                    "https://manager/api/v1/", "key", "scene", camera)

        session.get.assert_not_called()

    def test_add_camera_updates_matching_uid(self):
        with patch.object(self.module, "session") as session:
            session.get.return_value = self.response({
                "results": [{"uid": "lawn-cam1", "name": "lawn-cam1", "scene": "scene"}],
            })
            session.put.return_value = self.response()
            created_uid = self.module.add_camera(
                "https://manager/api/v1/", "key", "scene", CAMERA)

        self.assertIsNone(created_uid)
        self.assertTrue(session.put.call_args.args[0].endswith("camera/lawn-cam1"))
        self.assertNotIn("scene", session.put.call_args.kwargs["json"])
        session.post.assert_not_called()

    def test_record_created_camera_preserves_other_scenes_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as generated_dir:
            registry = Path(generated_dir) / "created-cameras.json"
            registry.write_text('[{"scene": "other", "uid": "other-cam"}]')

            self.module.record_created_camera(generated_dir, "scene", "lawn-cam1")
            self.module.record_created_camera(generated_dir, "scene", "lawn-cam1")

            self.assertEqual(json.loads(registry.read_text()), [
                {"scene": "other", "uid": "other-cam"},
                {"scene": "scene", "uid": "lawn-cam1"},
            ])


if __name__ == "__main__":
    unittest.main()
