from __future__ import annotations

from pathlib import Path

from vlm_evals.tasks.schemas import EvalTask


class PromptBuilder:
    def __init__(self, prompt_dir: str | Path = "prompts", default_prompt_id: str | None = None):
        self.prompt_dir = Path(prompt_dir)
        self.default_prompt_id = default_prompt_id
        self._cache: dict[str, str] = {}

    def load(self, prompt_id: str) -> str:
        prompt_id = prompt_id.removesuffix(".txt")
        if prompt_id not in self._cache:
            path = self.prompt_dir / f"{prompt_id}.txt"
            self._cache[prompt_id] = path.read_text(encoding="utf-8")
        return self._cache[prompt_id]

    def build(self, task: EvalTask) -> str:
        prompt_id = task.prompt_template or self.default_prompt_id
        if not prompt_id:
            raise ValueError(f"Task {task.task_id} does not define a prompt template")
        prompt = self.load(prompt_id)
        metadata = "\n".join(f"- {k}: {v}" for k, v in sorted(task.metadata.items()))
        if len(task.image_paths) == 1:
            image_lines = f"Image path: {task.image_path}"
        else:
            image_lines = "Image paths:\n" + "\n".join(
                f"- image {idx}: {path}" for idx, path in enumerate(task.image_paths, start=1)
            )
        return (
            f"{prompt.rstrip()}\n\n"
            f"Task id: {task.task_id}\n"
            f"Task type: {task.task_type}\n"
            f"{image_lines}\n"
            f"Metadata:\n{metadata if metadata else '- none'}\n"
        )
