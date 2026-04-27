import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from vlm_evals.backends.openai_compatible import OpenAICompatibleVisionClient
from vlm_evals.tasks.schemas import EvalTask


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer": true}'))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        )


class FakeClient:
    def __init__(self):
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class OpenAICompatibleVisionClientTests(unittest.TestCase):
    def test_sends_prompt_image_and_records_usage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            image_path.write_bytes(b"abc")
            task = EvalTask(
                task_id="t1",
                image_path=str(image_path),
                task_type="object_presence",
                prompt_template="object_presence_v1",
                expected_schema="object_presence_schema",
                labels={"answer": True},
            )

            fake_client = FakeClient()
            client = OpenAICompatibleVisionClient(
                client=fake_client,
                config={"temperature": 0.2, "max_tokens": 42, "timeout_seconds": 3},
                request_model_name="served-model",
                backend_name="vllm",
                reported_model_name="org/model",
            )
            prediction = client.predict(task, "is the object visible?", {})

        request = fake_client.completions.kwargs
        self.assertEqual(request["model"], "served-model")
        self.assertEqual(request["temperature"], 0.2)
        self.assertEqual(request["max_tokens"], 42)
        self.assertEqual(request["timeout"], 3.0)
        self.assertEqual(request["messages"][0]["content"][0]["text"], "is the object visible?")
        self.assertTrue(request["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(prediction["backend"], "vllm")
        self.assertEqual(prediction["model_name"], "org/model")
        self.assertEqual(prediction["raw_output"], '{"answer": true}')
        self.assertEqual(prediction["tokens_in"], 11)
        self.assertEqual(prediction["tokens_out"], 7)


if __name__ == "__main__":
    unittest.main()
