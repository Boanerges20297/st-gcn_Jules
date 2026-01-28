"""
Project-root stub for `google.generativeai` to satisfy imports during tests.
See `src/google/generativeai.py` for the source-of-truth stub.
"""
from typing import Any, Dict


def configure(api_key: str = None) -> None:
    return None


class _StubResponse:
    def __init__(self, text: str):
        self.text = text


class GenerativeModel:
    def __init__(self, model_name: str = "stub-model"):
        self.model_name = model_name

    def generate_content(self, prompt: str, generation_config: Dict[str, Any] = None) -> _StubResponse:
        raise RuntimeError("google.generativeai stub: no backend available in tests")
