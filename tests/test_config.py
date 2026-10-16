from pathlib import Path

from app import config
from app.config import ROOT, Settings


def test_defaults():
    settings = Settings.from_env({})

    assert settings == Settings()
    assert settings.host == "127.0.0.1"
    assert settings.port == 8770
    assert settings.models_dir == ROOT / "models"
    assert settings.data_dir == ROOT / "data"
    assert settings.stt_engine == "faster_whisper"
    assert settings.stt_model == "ZLSCompLing/Whisper-Large-V3-Turbo-Luxembourgish"
    assert settings.stt_device == "auto"
    assert settings.tts_engine == "piper"
    assert settings.tts_voice_lb == "lb_LU-marylux-medium"
    assert settings.tts_voice_en == "en_GB-cori-medium"
    assert settings.llm_engine == "claude_cli"
    assert settings.llm_model == "sonnet"
    assert settings.llm_timeout_s == 90
    assert settings.llm_effort_chat == "low"
    assert settings.llm_effort_teacher == "low"
    assert settings.chat_mix == 0


def test_root_is_the_repo_root_not_the_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert (ROOT / "app" / "config.py").is_file()
    assert Settings.from_env({}).models_dir == ROOT / "models"


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCI_HOST", "0.0.0.0")
    monkeypatch.setenv("LOCI_PORT", "9001")
    monkeypatch.setenv("LOCI_STT_DEVICE", "cpu")
    monkeypatch.setenv("LOCI_TTS_VOICE_LB", "other-voice")
    monkeypatch.setenv("LOCI_LLM_MODEL", "opus")
    monkeypatch.setenv("LOCI_LLM_TIMEOUT_S", "12.5")
    monkeypatch.setenv("LOCI_LLM_EFFORT_CHAT", "medium")
    monkeypatch.setenv("LOCI_LLM_EFFORT_TEACHER", "high")
    monkeypatch.setenv("LOCI_CHAT_MIX", "2")
    monkeypatch.setenv("LOCI_MODELS_DIR", str(tmp_path / "shared-models"))
    monkeypatch.setenv("LOCI_DATA_DIR", "other-data")

    settings = Settings.from_env()

    assert settings.host == "0.0.0.0"
    assert settings.port == 9001
    assert settings.stt_device == "cpu"
    assert settings.tts_voice_lb == "other-voice"
    assert settings.llm_model == "opus"
    assert settings.llm_timeout_s == 12.5
    assert (settings.llm_effort_chat, settings.llm_effort_teacher) == ("medium", "high")
    assert settings.chat_mix == 2 and type(settings.chat_mix) is int
    assert settings.models_dir == tmp_path / "shared-models"
    assert settings.data_dir == ROOT / "other-data"
    assert isinstance(settings.models_dir, Path)
    # conftest selects the fake engines through the environment.
    assert (settings.stt_engine, settings.tts_engine, settings.llm_engine) == ("fake",) * 3


def test_dotenv_file_is_read_and_the_environment_wins(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("LOCI_PORT=9100\nLOCI_LLM_MODEL=opus\n", encoding="utf-8")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.delenv("LOCI_PORT", raising=False)
    monkeypatch.setenv("LOCI_LLM_MODEL", "haiku")

    settings = Settings.from_env()

    assert settings.port == 9100
    assert settings.llm_model == "haiku"


def test_voice_tuning_defaults():
    settings = Settings.from_env({})

    assert (settings.tts_rate_lb, settings.tts_rate_en) == (0.7, 1.0)
    assert (settings.tts_pitch_lb, settings.tts_pitch_en) == (0.0, -1.5)
    assert (settings.tts_volume_lb, settings.tts_volume_en) == (1.0, 1.0)


def test_voice_tuning_is_read_from_the_environment_as_floats():
    env = {
        "LOCI_TTS_RATE_LB": "0.6",
        "LOCI_TTS_RATE_EN": "1",
        "LOCI_TTS_PITCH_LB": "2",
        "LOCI_TTS_PITCH_EN": "-3.5",
        "LOCI_TTS_VOLUME_LB": "0.5",
        "LOCI_TTS_VOLUME_EN": "",
    }

    settings = Settings.from_env(env)

    assert (settings.tts_rate_lb, settings.tts_rate_en) == (0.6, 1.0)
    assert (settings.tts_pitch_lb, settings.tts_pitch_en) == (2.0, -3.5)
    assert (settings.tts_volume_lb, settings.tts_volume_en) == (0.5, 1.0)
    names = [f"tts_{key}_{lang}" for key in ("rate", "pitch", "volume") for lang in ("lb", "en")]
    assert all(type(getattr(settings, name)) is float for name in names)
