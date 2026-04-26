import unittest
from unittest.mock import patch

from vlm_evals.backends.vllm_backend import ManagedVLLMServer, build_vllm_serve_command


class VLLMBackendTests(unittest.TestCase):
    def test_builds_vllm_serve_command_without_shell(self):
        command = build_vllm_serve_command(
            {
                "backend": "vllm",
                "model_name": "org/model",
                "serve": {
                    "host": "127.0.0.1",
                    "port": 9001,
                    "args": ["--dtype", "auto", "--generation-config", "vllm"],
                },
            }
        )
        self.assertEqual(command[:6], ["vllm", "serve", "org/model", "--host", "127.0.0.1", "--port"])
        self.assertIn("9001", command)
        self.assertIn("--generation-config", command)

    @patch("vlm_evals.backends.vllm_backend.urllib.request.urlopen")
    def test_readiness_check_uses_health_endpoint(self, urlopen):
        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

        urlopen.return_value = Response()
        server = ManagedVLLMServer(
            {
                "backend": "vllm",
                "model_name": "org/model",
                "serve": {"host": "127.0.0.1", "port": 9001, "health_path": "/health"},
            }
        )
        self.assertTrue(server._is_ready())
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:9001/health")


if __name__ == "__main__":
    unittest.main()

