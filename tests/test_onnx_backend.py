import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vlm_evals.backends.onnx_backend import ONNXBackend
from vlm_evals.tasks.schemas import EvalTask


class FakeImages:
    opened_paths = None

    @classmethod
    def open(cls, *paths):
        cls.opened_paths = paths
        return SimpleNamespace(paths=paths)


class FakeTokenizer:
    applied_messages = None

    def __init__(self, model):
        self.model = model

    def apply_chat_template(self, messages, add_generation_prompt):
        self.applied_messages = messages
        return "templated prompt"


class FakeStream:
    def decode(self, token):
        return {1: '{"answer": ', 2: '"first image"}'}.get(token, "")


class FakeProcessor:
    def __init__(self):
        self.calls = []

    def create_stream(self):
        return FakeStream()

    def __call__(self, prompt, images=None):
        self.calls.append((prompt, images))
        return {"prompt": prompt, "images": images}


class FakeModel:
    type = "phi3v"

    def create_multimodal_processor(self):
        return FakeProcessor()


class FakeGeneratorParams:
    search_options = None

    def __init__(self, model):
        self.model = model

    def set_search_options(self, **options):
        self.search_options = options


class FakeGenerator:
    inputs = None

    def __init__(self, model, params):
        self.tokens = [1, 2]

    def set_inputs(self, inputs):
        self.inputs = inputs

    def is_done(self):
        return not self.tokens

    def generate_next_token(self):
        return None

    def get_next_tokens(self):
        return [self.tokens.pop(0)]


class FakeOG:
    Images = FakeImages
    Tokenizer = FakeTokenizer
    GeneratorParams = FakeGeneratorParams
    Generator = FakeGenerator

    @staticmethod
    def Model(config):
        return FakeModel()


class ONNXBackendTests(unittest.TestCase):
    def test_requires_model_path(self):
        with self.assertRaisesRegex(ValueError, "model_path"):
            ONNXBackend({"backend": "onnx", "model_name": "x"})

    @patch("vlm_evals.backends.onnx_backend._load_onnxruntime_genai", return_value=FakeOG)
    def test_generates_with_multimodal_processor(self, _load):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()
            image_path = Path(tmpdir) / "image.png"
            image_path.write_bytes(b"image")

            backend = ONNXBackend(
                {
                    "backend": "onnx",
                    "model_name": "local/onnx-model",
                    "model_path": str(model_path),
                    "max_length": 128,
                }
            )
            task = EvalTask(
                task_id="t1",
                image_path=str(image_path),
                task_type="subtlebench_multiple_choice",
                prompt_template="subtlebench_multiple_choice_v1",
                expected_schema="subtlebench_multiple_choice_schema",
                labels={"answer": "first image"},
            )

            prediction = backend.predict(task, "Which image changed?", {})

        self.assertEqual(prediction["backend"], "onnx")
        self.assertEqual(prediction["model_name"], "local/onnx-model")
        self.assertEqual(prediction["raw_output"], '{"answer": "first image"}')
        self.assertEqual(FakeImages.opened_paths, (str(image_path),))


if __name__ == "__main__":
    unittest.main()
