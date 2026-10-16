"""A recorded sentence through the HTTP routes with the real speech engines.

    pytest -m models -s   real speech-to-text and text-to-speech, fake tutor
    pytest -m live -s     all three real engines; uses the Claude subscription

The turns run on a temporary data folder that holds a copy of the LOD store, so the
phrase memory of the real data folder is never written.
"""

import base64
import io
import json
import shutil
import subprocess
import sys
import wave
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.audio import cer, make_silence_wav
from tests.realdata import REAL_DATA

# Fixed in advance: an example sentence of the LOD entry MOIEN1, spoken by a person.
RECORDING = "moien.m4a"
MAX_CER = 0.35
SAMPLE_RATE = 22050
TEACHER_QUESTION = "Can you explain how to conjugate goen?"
MAX_CLI_CALLS = 3
TIMING_KEYS = {"stt_ms", "llm_ms", "tts_ms", "total_ms"}


def show(label: str, value) -> None:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    encoding = sys.stdout.encoding or "utf-8"
    print(f"{label}: {text}".encode(encoding, "backslashreplace").decode(encoding))


@pytest.fixture
def real_files(tmp_path):
    """(settings on the real models and a temporary data folder, the recording, its text)."""
    real = Settings.from_env()
    piper = real.models_dir / "piper"
    lod_audio = REAL_DATA / "lod_audio"
    store = REAL_DATA / "lod" / "lod.sqlite"
    needed = {
        real.models_dir / "whisper-lb" / "model.bin": "python scripts/download_models.py --stt",
        store: "python scripts/download_lod.py",
        lod_audio / RECORDING: "python scripts/download_models.py --lod-audio",
        lod_audio / "texts.json": "python scripts/download_models.py --lod-audio",
    }
    for voice in (real.tts_voice_lb, real.tts_voice_en):
        for suffix in (".onnx", ".onnx.json"):
            needed[piper / f"{voice}{suffix}"] = "python scripts/download_models.py --tts"
    for path, command in needed.items():
        if not path.is_file():
            pytest.skip(f"no {path}; run: {command}")

    data_dir = tmp_path / "data"
    (data_dir / "lod").mkdir(parents=True)
    shutil.copyfile(store, data_dir / "lod" / "lod.sqlite")
    settings = replace(real, data_dir=data_dir, stt_engine="faster_whisper", tts_engine="piper")
    texts = json.loads((lod_audio / "texts.json").read_text(encoding="utf-8"))
    return settings, (lod_audio / RECORDING).read_bytes(), texts[RECORDING]


def started(client) -> str:
    """Print the health of the started app, check it and open a session."""
    health = client.get("/api/health").json()
    print()
    for kind, engine in health["engines"].items():
        show(f"{kind} engine", f"{engine['name']} ready={engine['ready']} {engine['detail']}")
    show("stt device", client.app.state.stt.device)
    show("LOD store", health["lod"])
    assert health["engines"]["stt"]["ready"] and health["engines"]["tts"]["ready"]
    assert health["lod"]["ready"] is True
    return client.post("/api/session").json()["session_id"]


def spoken_turn(client, session_id: str, recording: bytes, text: str) -> dict:
    """Send the recording through /api/turn and check the transcript, the reply and its audio."""
    response = client.post(
        "/api/turn",
        data={"session_id": session_id},
        files={"audio": (RECORDING, recording, "audio/mp4")},
    )
    body = response.json()
    assert response.status_code == 200, body
    error_rate = cer(text, body["transcript"])
    show("transcript", f"{body['transcript']!r}, CER {error_rate:.3f} against {text!r}")
    show("mode", body["mode"])
    show("segments", [[segment["lang"], segment["text"]] for segment in body["segments"]])
    show("claims", body["claims"])
    show("shape_warning", body["shape_warning"] or "none")
    show("timings", body["timings"])

    assert body["heard"] is True
    assert body["transcript"] and error_rate <= MAX_CER
    assert body["segments"][0]["lang"] == "lb"
    assert any(segment["lang"] == "en" for segment in body["segments"])
    assert set(body["timings"]) == TIMING_KEYS
    assert body["timings"]["stt_ms"] > 0 and body["timings"]["tts_ms"] > 0

    with wave.open(io.BytesIO(base64.b64decode(body["audio_wav_base64"]))) as wav:
        shape = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate())
        duration_s = wav.getnframes() / wav.getframerate()
    show("reply audio", f"{duration_s:.2f} s")
    assert shape == (1, 2, SAMPLE_RATE)
    assert body["sample_rate"] == SAMPLE_RATE
    assert duration_s > 1
    offsets = body["segment_offsets"]
    assert [offset["lang"] for offset in offsets] == [segment["lang"] for segment in body["segments"]]
    for lang in ("lb", "en"):
        assert any(
            offset["lang"] == lang and offset["end_s"] > offset["start_s"] for offset in offsets
        )
    assert offsets[-1]["end_s"] == pytest.approx(duration_s)
    return body


@pytest.mark.models
def test_recording_through_real_speech_engines_with_a_fake_tutor(real_files):
    settings, recording, text = real_files

    with TestClient(create_app(replace(settings, llm_engine="fake"))) as client:
        llm = client.app.state.llm
        session_id = started(client)

        turn = spoken_turn(client, session_id, recording, text)

        assert len(llm.calls) == 1
        assert llm.calls[0]["history"][-1].text == turn["transcript"]
        saved = json.loads((settings.data_dir / "phrases.json").read_text(encoding="utf-8"))
        assert [entry["lb"] for entry in saved] == ["Moien!"]

        silence = client.post(
            "/api/turn",
            data={"session_id": session_id},
            files={"audio": ("silence.wav", make_silence_wav(5), "audio/wav")},
        )

        body = silence.json()
        show("5 s silence", body)
        assert silence.status_code == 200
        assert set(body) == {"heard", "transcript", "timings"}
        assert (body["heard"], body["transcript"]) == (False, "")
        assert len(llm.calls) == 1


class CountedRunner:
    """subprocess.run that refuses more than `limit` `claude -p` calls."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.calls = 0

    def __call__(self, argv, **kwargs):
        if argv[1] == "-p":
            if self.calls == self.limit:
                raise RuntimeError(f"this test run may make {self.limit} claude -p calls, not more")
            self.calls += 1
        return subprocess.run(argv, **kwargs)


@pytest.mark.live
def test_live_recording_then_teacher_turn_through_all_real_engines(request):
    if "live" not in request.config.getoption("markexpr"):
        pytest.skip("calls the real claude CLI on the user's subscription; run with -m live")
    settings, recording, text = request.getfixturevalue("real_files")

    with TestClient(create_app(replace(settings, llm_engine="claude_cli"))) as client:
        llm = client.app.state.llm
        if not llm.ready():
            pytest.skip(llm.detail)
        llm.runner = runner = CountedRunner(MAX_CLI_CALLS)
        session_id = started(client)

        chat = spoken_turn(client, session_id, recording, text)

        assert chat["timings"]["llm_ms"] > 0

        response = client.post("/api/turn", data={"session_id": session_id, "text": TEACHER_QUESTION})

        teacher = response.json()
        assert response.status_code == 200, teacher
        show("teacher question", TEACHER_QUESTION)
        show("teacher mode", teacher["mode"])
        show("teacher segments", [[segment["lang"], segment["text"]] for segment in teacher["segments"]])
        show("teacher claims", teacher["claims"])
        show("teacher shape_warning", teacher["shape_warning"] or "none")
        show("teacher timings", teacher["timings"])
        show("claude -p calls", runner.calls)
        assert teacher["mode"] == "teacher"
        assert teacher["claims"]
        assert any(claim["verified"] is True for claim in teacher["claims"])
        assert teacher["timings"]["stt_ms"] == 0 and teacher["timings"]["llm_ms"] > 0
        assert len(teacher["segment_offsets"]) == len(teacher["segments"])
