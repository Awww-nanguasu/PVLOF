"""Offline launcher regression checks run during image build; no ES requests."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wizard import prepare_wizard


class WizardEnvironmentTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.env_file = Path(temporary.name) / ".env"
        self.env_file.write_text(
            'PVLOF_API_KEY="example-key$with-dollar"\n'
            'PVLOF_MODEL_ROOT=../native-model\n'
            'PVLOF_RUNTIME_ROOT=./native-runtime\n'
            'PVLOF_API_HOST=127.0.0.1\n',
            encoding="utf-8",
        )

    def test_service_key_is_reused_without_command_line_exposure(self):
        args = prepare_wizard([], self.env_file)
        self.assertEqual(os.environ["PVLOF_CLIENT_API_KEY"], "example-key$with-dollar")
        self.assertNotIn("example-key$with-dollar", " ".join(args))
        self.assertNotIn("--api-key", args)

    def test_explicit_client_key_is_preserved(self):
        os.environ["PVLOF_CLIENT_API_KEY"] = "explicit-client-key"
        prepare_wizard([], self.env_file)
        self.assertEqual(os.environ["PVLOF_CLIENT_API_KEY"], "explicit-client-key")

    def test_blank_client_key_falls_back_to_service_key(self):
        os.environ["PVLOF_CLIENT_API_KEY"] = "  "
        prepare_wizard([], self.env_file)
        self.assertEqual(os.environ["PVLOF_CLIENT_API_KEY"], "example-key$with-dollar")

    def test_docker_paths_override_mounted_windows_configuration(self):
        expected = {
            "PVLOF_MODEL_ROOT": "/app/PVLOF-V3-deploy",
            "PVLOF_RUNTIME_ROOT": "/app/runtime",
            "PVLOF_API_HOST": "0.0.0.0",
        }
        os.environ.update(expected)
        prepare_wizard([], self.env_file)
        for key, value in expected.items():
            self.assertEqual(os.environ[key], value)

    def test_default_output_is_on_host_mount_and_arguments_not_mutated(self):
        original = ["--poll-seconds", "2"]
        args = prepare_wizard(original, self.env_file)
        self.assertEqual(original, ["--poll-seconds", "2"])
        self.assertEqual(args[-2:], ["--output-directory", "/app/outputs/pvlof_v3_cli"])

    def test_both_explicit_output_argument_forms_are_preserved(self):
        for args in (
            ["--output-directory", "/app/outputs/custom"],
            ["--output-directory=/app/outputs/custom"],
        ):
            with self.subTest(args=args):
                self.assertEqual(prepare_wizard(args, self.env_file), args)

    def test_no_auth_configuration_is_supported(self):
        self.env_file.write_text("PVLOF_API_KEY=\n", encoding="utf-8")
        prepare_wizard([], self.env_file)
        self.assertEqual(os.environ["PVLOF_CLIENT_API_KEY"], "")


if __name__ == "__main__":
    unittest.main()
