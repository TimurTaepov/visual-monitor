import unittest
from unittest.mock import patch

from vlm_evals.serving.backend_manager import ServingBackendManager


class FakeBackend:
    starts = 0
    closes = 0

    def start(self):
        FakeBackend.starts += 1

    def close(self):
        FakeBackend.closes += 1


class ServingBackendManagerTests(unittest.TestCase):
    def setUp(self):
        FakeBackend.starts = 0
        FakeBackend.closes = 0

    @patch("vlm_evals.serving.backend_manager.create_backend")
    def test_reuses_backend_until_close_all(self, create_backend):
        create_backend.return_value = FakeBackend()
        manager = ServingBackendManager()
        config = {"backend": "mock", "model_name": "fake"}

        with manager.backend(config) as first:
            with manager.backend(config) as second:
                self.assertIs(first, second)

        self.assertEqual(FakeBackend.starts, 1)
        self.assertEqual(FakeBackend.closes, 0)

        manager.close_all()
        self.assertEqual(FakeBackend.closes, 1)


if __name__ == "__main__":
    unittest.main()
