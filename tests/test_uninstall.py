import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_uninstall():
    spec = importlib.util.spec_from_file_location("uninstall_under_test", ROOT / "uninstall.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class UninstallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_uninstall()

    def test_compose_command_uses_current_example_with_optional_override(self):
        for override in (None, "/app/override.yml"):
            with self.subTest(override=override):
                expected = [
                    "docker", "compose",
                    "--project-directory", "/scenescape",
                    "--env-file", "/app/app.env",
                    "-f", "/scenescape/sample_data/compose/compose.dl-streamer-example.yml",
                ]
                if override:
                    expected.extend(["-f", override])
                expected.extend(["--profile", "controller", "ps"])
                self.assertEqual(self.module.compose_command(
                    "/scenescape", override, "/app/app.env", "ps"), expected)

    def test_certificate_hostname_uses_environment_and_default(self):
        for env, expected in (({"SCENESCAPE_CERT_HOSTNAME": "custom.example.net"},
                               "custom.example.net"), ({}, "web.scenescape.intel.com")):
            with self.subTest(expected=expected), patch.dict(os.environ, env, clear=True):
                module = load_uninstall()
                with patch.object(module.requests.adapters.HTTPAdapter, "init_poolmanager") as init:
                    module.SceneScapeHTTPSAdapter().init_poolmanager(1, 1)
                self.assertEqual(init.call_args.kwargs["assert_hostname"], expected)
                self.assertEqual(init.call_args.kwargs["server_hostname"], expected)

    def test_delete_only_cameras_recorded_in_matching_scene(self):
        with tempfile.TemporaryDirectory() as generated_dir:
            registry = Path(generated_dir) / "created-cameras.json"
            registry.write_text(json.dumps([
                {"scene": "scene-a", "uid": "created"},
            ]))
            with patch.object(self.module, "session") as session:
                session.get.return_value.json.return_value = {"results": [
                    {"scene": "scene-a", "uid": "created"},
                    {"scene": "scene-b", "uid": "created"},
                    {"scene": "scene-a", "uid": "preexisting"},
                ]}
                self.module.delete_app_cameras(generated_dir, "key", "ca.pem")

            session.delete.assert_called_once()
            self.assertTrue(session.delete.call_args.args[0].endswith("camera/created"))

    def test_missing_registry_never_deletes_existing_cameras(self):
        with tempfile.TemporaryDirectory() as generated_dir:
            with patch.object(self.module, "session") as session:
                self.module.delete_app_cameras(generated_dir, "key", "ca.pem")
            session.get.assert_not_called()
            session.delete.assert_not_called()

    def test_uninstall_restores_running_scene_with_base_compose_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            app_path = Path(temp_dir) / "app"
            generated_dir = app_path / ".generated"
            generated_dir.mkdir(parents=True)
            (generated_dir / "fall-detection.env").write_text("SCENESCAPE_API_KEY=key\n")
            (generated_dir / "docker-compose.override.yml").write_text("services: {}\n")
            with patch.object(self.module, "prompt_for_path", side_effect=[
                str(app_path), str(Path(temp_dir) / "scenescape")
            ]), patch("builtins.input", side_effect=["", ""]), \
                    patch.object(self.module.subprocess, "run") as run:
                run.return_value = SimpleNamespace(stdout="scene-container\n")
                self.assertEqual(self.module.main(), 0)

            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(len(commands), 3)
            self.assertEqual(commands[0][-8:], [
                "rm", "--stop", "--force", "falling-cams", "falling-video",
                "fall-detection", "node-red", "falling-mediaserver",
            ])
            self.assertEqual(commands[1][-5:], ["ps", "--status", "running", "-q", "scene"])
            self.assertEqual(commands[2][-4:], ["up", "-d", "--no-deps", "scene"])
            self.assertEqual(commands[2].count("-f"), 1)
            self.assertNotIn(str(generated_dir / "docker-compose.override.yml"), commands[2])

    def test_uninstall_does_not_start_stopped_scene(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            app_path = Path(temp_dir) / "app"
            generated_dir = app_path / ".generated"
            generated_dir.mkdir(parents=True)
            (generated_dir / "fall-detection.env").write_text("SCENESCAPE_API_KEY=key\n")
            (generated_dir / "docker-compose.override.yml").write_text("services: {}\n")
            with patch.object(self.module, "prompt_for_path", side_effect=[
                str(app_path), str(Path(temp_dir) / "scenescape")
            ]), patch("builtins.input", side_effect=["", ""]), \
                    patch.object(self.module.subprocess, "run") as run:
                run.return_value = SimpleNamespace(stdout="")
                self.assertEqual(self.module.main(), 0)

            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args_list[-1].args[0][-5:], [
                "ps", "--status", "running", "-q", "scene",
            ])


if __name__ == "__main__":
    unittest.main()