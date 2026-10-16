"""Builds the engines named in the settings.

Engine modules are imported only when their name is configured, so the fakes
never pull in faster-whisper or piper. A new engine is a new module plus an
entry in ENGINES; every class is constructed as `Cls(settings)`.
"""

from __future__ import annotations

import importlib
import logging

from app.config import Settings

log = logging.getLogger(__name__)

KINDS = ("stt", "tts", "llm", "acceptability")

ENGINES: dict[str, dict[str, str]] = {
    "stt": {
        "fake": "app.engines.fake:FakeSTT",
        "faster_whisper": "app.engines.stt_faster_whisper:FasterWhisperSTT",
    },
    "tts": {
        "fake": "app.engines.fake:FakeTTS",
        "piper": "app.engines.tts_piper:PiperTTS",
    },
    "llm": {
        "fake": "app.engines.fake:FakeLLM",
        "claude_cli": "app.engines.llm_claude_cli:ClaudeCliLLM",
        "provider": "app.engines.llm_provider:ProviderLLM",
    },
    "acceptability": {
        "none": "app.engines.fake:NoAcceptability",
        "fake": "app.engines.fake:FakeAcceptability",
    },
}


class UnavailableEngine:
    """Stands in for an engine that could not be built, so the app still starts."""

    def __init__(self, name: str, detail: str) -> None:
        self.name = name
        self.detail = detail

    def ready(self) -> bool:
        return False

    def warmup(self) -> None:
        pass

    def _unavailable(self, *args, **kwargs):
        raise RuntimeError(f"{self.name} engine is unavailable: {self.detail}")

    transcribe = synthesize = reply = judge = wrap_up = score = _unavailable


def build_engine(kind: str, settings: Settings):
    """Import, construct and warm up the configured engine of `kind`, one of KINDS."""
    name = getattr(settings, f"{kind}_engine")
    try:
        known = ENGINES[kind]
        if name not in known:
            raise ValueError(f"unknown {kind} engine {name!r}; known: {', '.join(sorted(known))}")
        module_name, class_name = known[name].split(":")
        engine = getattr(importlib.import_module(module_name), class_name)(settings)
        engine.warmup()
    except Exception as exc:
        log.exception("%s engine %r is unavailable", kind, name)
        return UnavailableEngine(name, f"{type(exc).__name__}: {exc}")
    return engine
