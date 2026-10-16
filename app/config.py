"""Settings read from LOCI_* environment variables and an optional .env file."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
PREFIX = "LOCI_"


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8770
    # A relative LOCI_MODELS_DIR / LOCI_DATA_DIR is resolved against ROOT, not the cwd.
    models_dir: Path = ROOT / "models"
    data_dir: Path = ROOT / "data"
    stt_engine: str = "faster_whisper"  # or "fake"
    stt_model: str = "ZLSCompLing/Whisper-Large-V3-Turbo-Luxembourgish"
    stt_device: str = "auto"  # or "cuda", "cpu"
    tts_engine: str = "piper"  # or "fake"
    tts_voice_lb: str = "lb_LU-marylux-medium"
    tts_voice_en: str = "en_GB-cori-medium"
    # Per language: speaking-rate multiplier, pitch in semitones, linear gain.
    tts_rate_lb: float = 0.7
    tts_rate_en: float = 1.0
    tts_pitch_lb: float = 0.0
    tts_pitch_en: float = -1.5
    tts_volume_lb: float = 1.0
    tts_volume_en: float = 1.0
    llm_engine: str = "claude_cli"  # or "fake"
    llm_model: str = "sonnet"
    llm_timeout_s: float = 90.0
    # `claude --effort` level per mode: low, medium, high, xhigh, max; "" or "default" (any case)
    # passes no flag and leaves the CLI default.
    llm_effort_chat: str = "low"
    llm_effort_teacher: str = "low"
    acceptability_engine: str = "none"  # or "fake"; scores how acceptable a Luxembourgish text is
    # How much Luxembourgish a chat reply mixes into its English, 0 to 3, when a turn names no level.
    chat_mix: int = 0

    @classmethod
    def from_env(cls, env: Mapping[str, str | None] | None = None) -> Settings:
        """Build settings from `env`; by default the process environment over ROOT/.env."""
        if env is None:
            env = {**dotenv_values(ROOT / ".env"), **os.environ}
        values = {}
        for field in fields(cls):
            raw = env.get(PREFIX + field.name.upper())
            if not raw:
                continue
            if isinstance(field.default, Path):
                values[field.name] = ROOT / raw
            else:
                values[field.name] = type(field.default)(raw)
        return cls(**values)
