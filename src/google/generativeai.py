"""
Minimal stub for `google.generativeai` used in tests.

This file provides `configure()` and a `GenerativeModel` class with a
`generate_content()` method that raises a RuntimeError so callers will
fall back to the deterministic/mock parsing paths during tests.

Do NOT use this stub in production. Install the real `google-genai`
SDK and remove this file when running against the real API.
"""
from typing import Any, Dict


def configure(api_key: str = None) -> None:
    # no-op stub
    return None


class _StubResponse:
    def __init__(self, text: str):
        self.text = text


class GenerativeModel:
    def __init__(self, model_name: str = "stub-model"):
        self.model_name = model_name

    def generate_content(self, prompt: str, generation_config: Dict[str, Any] = None) -> _StubResponse:
        # Simulate an SDK failure so the calling code will fall back to
        # REST or the local mock parser. Raising an exception here ensures
        # that production code paths are not accidentally used during tests.
        raise RuntimeError("google.generativeai stub: no backend available in tests")
