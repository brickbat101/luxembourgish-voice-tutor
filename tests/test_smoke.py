import io
import wave
from dataclasses import replace

import httpx
import pytest

from app.config import Settings
from app.engines.base import LLMError
from app.engines.fake import FakeSTT
from scripts import smoke
from tests.audio import make_noise_wav, make_silence_wav


@pytest.fixture
def settings(tmp_path):
    return replace(Settings.from_env(), data_dir=tmp_path / "data")


@pytest.fixture
def recording(tmp_path):
    path = tmp_path / "speech.wav"
    path.write_bytes(make_noise_wav(1.0))
    return path


def test_smoke_sends_a_recording_prints_the_turn_and_saves_the_reply(client, recording, tmp_path, capsys):
    out_path = tmp_path / "out" / "last_reply.wav"

    smoke.smoke(client, recording, None, out_path)

    lines = capsys.readouterr().out.splitlines()
    assert lines[:5] == [
        "health: ok",
        "  stt: fake ready",
        "  tts: fake ready",
        "  llm: fake ready",
        "  acceptability: none ready no encoder installed",
    ]
    # The temporary data folder has no LOD store.
    assert lines[5].startswith("  lod store: NOT READY ") and "download_lod.py" in lines[5]
    assert f"transcript: {FakeSTT.TEXT}" in lines
    assert "mode: chat" in lines
    assert "[lb] Moien!" in lines and "[en] Hello! How was your day?" in lines
    assert "claims: none" in lines
    assert any(line.startswith("timings: stt_ms ") and "total_ms" in line for line in lines)
    with wave.open(io.BytesIO(out_path.read_bytes())) as wav:
        assert wav.getnframes() > 0
    assert lines[-1].startswith(f"reply audio: {out_path}")


def test_smoke_sends_text_and_prints_the_checked_claims(client, recording, tmp_path, capsys):
    smoke.smoke(client, recording, "How do I conjugate goen?", tmp_path / "last_reply.wav")

    lines = capsys.readouterr().out.splitlines()
    assert "sending the text: How do I conjugate goen?" in lines
    assert "transcript: How do I conjugate goen?" in lines
    assert "mode: teacher" in lines


def test_smoke_exits_with_the_error_and_fix_of_the_server(client, recording, tmp_path):
    client.app.state.llm.queue.append(LLMError("auth", "Failed to authenticate"))
    out_path = tmp_path / "last_reply.wav"

    with pytest.raises(SystemExit) as exit_info:
        smoke.smoke(client, recording, None, out_path)

    message = str(exit_info.value)
    assert "HTTP 503" in message
    assert "error: Failed to authenticate" in message
    assert "fix: run `claude auth login` once in a terminal" in message
    assert not out_path.exists()


def test_smoke_exits_when_nothing_was_heard_or_the_recording_is_missing(client, tmp_path):
    silence = tmp_path / "silence.wav"
    silence.write_bytes(make_silence_wav(1.0))

    with pytest.raises(SystemExit, match="heard no speech"):
        smoke.smoke(client, silence, None, tmp_path / "last_reply.wav")
    with pytest.raises(SystemExit, match="download_models.py --lod-audio"):
        smoke.smoke(client, tmp_path / "absent.m4a", None, tmp_path / "last_reply.wav")
    assert not (tmp_path / "last_reply.wav").exists()


def test_smoke_exits_when_the_server_is_not_reachable(tmp_path):
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    with httpx.Client(base_url="http://127.0.0.1:8770", transport=httpx.MockTransport(refuse)) as client:
        with pytest.raises(SystemExit) as exit_info:
            smoke.smoke(client, tmp_path / "absent.m4a", None, tmp_path / "last_reply.wav")

    message = str(exit_info.value)
    assert "GET http://127.0.0.1:8770/api/health failed: ConnectError" in message
    assert "run.ps1" in message
