import base64
import inspect
import io
import json
import re
import wave
from dataclasses import replace

import pytest

from fastapi.testclient import TestClient

from app import api
from app.config import ROOT, Settings
from app.engines.base import LLMError, Segment, TutorReply
from app.engines.fake import SAMPLE_RATE, FakeSTT
from app.main import create_app
from tests.audio import make_noise_wav, make_silence_wav
from tests.conftest import plain_page

TURN_KEYS = {
    "heard",
    "transcript",
    "mode",
    "mix",
    "segments",
    "new_phrases",
    "claims",
    "phrases_added",
    "timings",
    "shape_warning",
}
AUDIO_KEYS = {"audio_wav_base64", "sample_rate", "segment_offsets"}
TIMING_KEYS = {"stt_ms", "llm_ms", "tts_ms", "total_ms"}
OFFSET_KEYS = {"index", "lang", "start_s", "end_s"}
TEACHER_REPLY = TutorReply(
    segments=[Segment("lb", "Ech ginn."), Segment("en", "That means I go."), Segment("lb", "du gees")],
    new_phrases=[{"lb": "Ech ginn.", "en": "I go."}],
    mode="teacher",
    claims=[{"form": "gees", "lemma": "goen"}],
)
# A chat reply at mix level 2: Luxembourgish phrases after the opening one.
MIXED_REPLY = TutorReply(
    segments=[
        Segment("lb", "Dat ass schéin!"),
        Segment("lb", "E Kaffi mat denger Schwëster,"),
        Segment("en", "that is a good start to the day. I hope you had"),
        Segment("lb", "vill Zäit fir ze schwätzen."),
        Segment("en", "What did you talk about?"),
    ],
    new_phrases=[{"lb": "e Kaffi mat denger Schwëster", "en": "a coffee with your sister"}],
)
MIX_FIX = "send `mix` as a whole number from 0 to 3, or leave it out"


@pytest.fixture
def settings(tmp_path):
    # The phrase memory of these tests lives in a temporary folder, never in the real data/.
    return replace(Settings.from_env(), data_dir=tmp_path / "data")


@pytest.fixture
def state(client):
    return client.app.state


@pytest.fixture
def session_id(client):
    return client.post("/api/session").json()["session_id"]


@pytest.fixture
def no_stt(state, monkeypatch):
    """Fails the test when speech-to-text runs."""

    def transcribe(audio_bytes, mime):
        raise AssertionError("speech-to-text must not run")

    monkeypatch.setattr(state.stt, "transcribe", transcribe)


def wav_upload(wav_bytes: bytes) -> dict:
    return {"audio": ("speech.wav", wav_bytes, "audio/wav")}


def opened_wav(wav_bytes: bytes) -> tuple[int, int]:
    """(sample rate, frames) of a WAV that `wave` can open."""
    with wave.open(io.BytesIO(wav_bytes)) as wav:
        return wav.getframerate(), wav.getnframes()


# --- session ---


def test_session_returns_a_new_known_session_each_time(client, state):
    first, second = client.post("/api/session"), client.post("/api/session")

    assert first.status_code == second.status_code == 200
    assert set(first.json()) == {"session_id"}
    assert first.json()["session_id"] != second.json()["session_id"]
    assert state.sessions.get(first.json()["session_id"]) is not None


# --- stt ---


def test_stt_returns_the_transcript_and_its_milliseconds(client):
    response = client.post("/api/stt", files=wav_upload(make_noise_wav(1.0)))

    body = response.json()
    assert response.status_code == 200
    assert set(body) == {"text", "language", "duration_s", "segments", "stt_ms"}
    assert (body["text"], body["language"], body["duration_s"], body["segments"]) == (FakeSTT.TEXT, "lb", 1.0, [])
    assert type(body["stt_ms"]) is int and body["stt_ms"] >= 0


def test_stt_of_silence_is_an_empty_transcript(client):
    body = client.post("/api/stt", files=wav_upload(make_silence_wav(1.0))).json()

    assert body["text"] == ""


def test_stt_passes_the_content_type_of_the_upload_on_and_defaults_to_webm(client, state, monkeypatch):
    mimes = []
    transcribe = state.stt.transcribe
    monkeypatch.setattr(state.stt, "transcribe", lambda audio, mime: mimes.append(mime) or transcribe(audio, mime))
    bare_part = b'--x\r\nContent-Disposition: form-data; name="audio"; filename="speech"\r\n\r\nabc\r\n--x--\r\n'

    client.post("/api/stt", files={"audio": ("speech", b"abc", "audio/webm;codecs=opus")})
    response = client.post(
        "/api/stt", content=bare_part, headers={"Content-Type": "multipart/form-data; boundary=x"}
    )

    assert response.status_code == 200
    assert mimes == ["audio/webm;codecs=opus", "audio/webm"]


def test_stt_without_a_file_is_a_422(client):
    assert client.post("/api/stt").status_code == 422


@pytest.mark.parametrize("path", ["/api/stt", "/api/turn"])
def test_an_upload_over_the_limit_is_a_413(client, session_id, monkeypatch, no_stt, path):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 10)

    response = client.post(path, data={"session_id": session_id}, files=wav_upload(b"x" * 11))

    assert response.status_code == 413
    assert set(response.json()) == {"error", "fix"}


def test_the_upload_limit_is_25_megabytes_and_an_upload_at_the_limit_passes(client, monkeypatch):
    assert api.MAX_UPLOAD_BYTES == 25 * 1024 * 1024
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 10)

    assert client.post("/api/stt", files=wav_upload(b"x" * 10)).status_code == 200


# --- chat ---


def test_chat_returns_the_turn_without_audio(client, session_id, state):
    response = client.post("/api/chat", json={"session_id": session_id, "text": " I had coffee. ", "stt_ms": 40})

    body = response.json()
    assert response.status_code == 200
    assert set(body) == TURN_KEYS
    assert body["heard"] is True
    assert body["transcript"] == "I had coffee."
    assert (body["mode"], body["mix"]) == ("chat", 0)
    assert body["segments"] == [
        {"lang": "lb", "text": "Moien!", "gloss": "Hello!"},
        {"lang": "en", "text": "Hello! How was your day?"},
    ]
    assert body["new_phrases"] == [{"lb": "Moien!", "en": "Hello!"}]
    assert body["claims"] == []
    assert body["phrases_added"] == ["Moien!"]
    assert set(body["timings"]) == TIMING_KEYS
    assert body["timings"]["stt_ms"] == 40
    assert body["shape_warning"] == ""
    assert [turn.role for turn in state.sessions.get(session_id).turns] == ["user", "tutor"]


def test_chat_in_teacher_mode_returns_the_checked_claims(client, session_id, state):
    state.llm.queue.append(TEACHER_REPLY)

    body = client.post("/api/chat", json={"session_id": session_id, "text": "I had coffee", "mode": "teacher"}).json()

    assert body["mode"] == "teacher"
    assert [segment["lang"] for segment in body["segments"]] == ["lb", "en", "lb"]
    assert [set(claim) for claim in body["claims"]] == [{"form", "lemma", "verified"}]
    assert (body["claims"][0]["form"], body["claims"][0]["lemma"]) == ("gees", "goen")
    assert type(body["claims"][0]["verified"]) is bool
    assert state.llm.calls[0]["mode"] == "teacher"


def test_chat_with_an_unknown_session_is_a_404(client, state):
    response = client.post("/api/chat", json={"session_id": "no-such-session", "text": "Hello"})

    assert response.status_code == 404
    assert response.json() == {"error": "unknown session", "fix": "reload the page"}
    assert state.llm.calls == []


def test_an_auth_error_is_a_503_with_the_fix(client, session_id, state):
    state.llm.queue.append(LLMError("auth", "Failed to authenticate: OAuth session expired"))

    response = client.post("/api/chat", json={"session_id": session_id, "text": "Hello"})

    assert response.status_code == 503
    assert response.json()["error"] == "Failed to authenticate: OAuth session expired"
    assert "claude auth login" in response.json()["fix"]


def test_another_llm_error_is_a_502(client, session_id, state):
    state.llm.queue.append(LLMError("timeout", "claude gave no reply within 90 s"))

    response = client.post("/api/chat", json={"session_id": session_id, "text": "Hello"})

    assert response.status_code == 502
    assert response.json() == {"error": "claude gave no reply within 90 s", "fix": ""}


@pytest.mark.parametrize("fields", [{"text": "  "}, {"text": "Hello", "mode": "lecture"}])
def test_chat_with_blank_text_or_an_unknown_mode_is_a_400(client, session_id, state, fields):
    response = client.post("/api/chat", json={"session_id": session_id, **fields})

    assert response.status_code == 400
    assert set(response.json()) == {"error", "fix"}
    assert state.llm.calls == []


def test_chat_at_a_mix_level_returns_the_level_and_the_glosses_of_the_lb_segments(client, session_id, state):
    state.llm.queue.append(MIXED_REPLY)

    response = client.post("/api/chat", json={"session_id": session_id, "text": "I had coffee.", "mix": 2})

    body = response.json()
    assert response.status_code == 200
    assert set(body) == TURN_KEYS
    assert (body["mode"], body["mix"], body["shape_warning"]) == ("chat", 2, "")
    assert state.llm.calls[0]["grounding"].startswith("mix level: 2\n")
    assert body["segments"] == [
        {"lang": "lb", "text": "Dat ass schéin!"},
        {"lang": "lb", "text": "E Kaffi mat denger Schwëster,", "gloss": "a coffee with your sister"},
        {"lang": "en", "text": "that is a good start to the day. I hope you had"},
        {"lang": "lb", "text": "vill Zäit fir ze schwätzen."},
        {"lang": "en", "text": "What did you talk about?"},
    ]


def test_chat_in_teacher_mode_ignores_the_mix_level(client, session_id, state):
    state.llm.queue.append(TEACHER_REPLY)
    fields = {"session_id": session_id, "text": "I had coffee", "mode": "teacher", "mix": 3}

    body = client.post("/api/chat", json=fields).json()

    assert (body["mode"], body["mix"], body["shape_warning"]) == ("teacher", 0, "")
    assert "mix level" not in state.llm.calls[0]["grounding"]
    assert body["segments"][0] == {"lang": "lb", "text": "Ech ginn.", "gloss": "I go."}


@pytest.mark.parametrize("mix", [4, -1, 1.5, "two", True, [2]])
def test_chat_with_a_mix_that_is_no_level_is_a_400(client, session_id, state, mix):
    response = client.post("/api/chat", json={"session_id": session_id, "text": "Hello", "mix": mix})

    assert response.status_code == 400
    assert response.json() == {"error": f"unknown mix level {mix!r}; known: 0 to 3", "fix": MIX_FIX}
    assert state.llm.calls == []


# --- tts ---


def test_tts_returns_a_wav_with_the_segment_offsets_in_a_header(client):
    segments = [{"lang": "lb", "text": "Moien!"}, {"lang": "en", "text": "Hello!"}]

    response = client.post("/api/tts", json={"segments": segments})

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    rate, frames = opened_wav(response.content)
    assert rate == SAMPLE_RATE
    raw = response.headers["x-segment-offsets"]
    offsets = json.loads(raw)
    assert " " not in raw
    assert [set(offset) for offset in offsets] == [OFFSET_KEYS, OFFSET_KEYS]
    assert [(offset["index"], offset["lang"]) for offset in offsets] == [(0, "lb"), (1, "en")]
    assert offsets[0]["start_s"] == 0
    assert offsets[0]["end_s"] < offsets[1]["start_s"] < offsets[1]["end_s"]
    assert offsets[1]["end_s"] == pytest.approx(frames / rate)


def test_tts_without_segments_is_a_400(client):
    response = client.post("/api/tts", json={"segments": []})

    assert response.status_code == 400
    assert set(response.json()) == {"error", "fix"}


def test_a_failing_tts_engine_is_a_500_naming_the_engine(client, state, monkeypatch):
    monkeypatch.setattr(state.tts, "synthesize", lambda segments, voice=None: 1 / 0)

    response = client.post("/api/tts", json={"segments": [{"lang": "lb", "text": "Moien!"}]})

    assert response.status_code == 500
    assert response.json() == {"error": "tts engine (fake) failed: ZeroDivisionError: division by zero", "fix": ""}


def test_tts_passes_the_voice_to_the_engine(client, state):
    segments = [{"lang": "lb", "text": "Moien!"}]
    voice = {"lb": {"rate": 0.7, "pitch": -2, "volume": 1.5}, "en": {"pitch": 1}}

    response = client.post("/api/tts", json={"segments": segments, "voice": voice})

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert state.tts.voice == voice


def test_tts_without_a_voice_passes_none(client, state):
    state.tts.voice = "untouched"

    response = client.post("/api/tts", json={"segments": [{"lang": "lb", "text": "Moien!"}]})

    assert response.status_code == 200
    assert state.tts.voice is None


def test_tts_passes_a_trial_respelling_to_the_engine_and_writes_nothing(client, state, settings):
    segments = [{"lang": "lb", "text": "Moien, Stefan!"}]
    before = sorted(settings.data_dir.rglob("*"))

    response = client.post(
        "/api/tts", json={"segments": segments, "respell": {"Stefan": " [[ ʃtɛfan ]] ", "moien": "mojen"}}
    )

    assert response.status_code == 200
    assert state.tts.respell == {"Stefan": "[[ ʃtɛfan ]]", "moien": "mojen"}
    assert sorted(settings.data_dir.rglob("*")) == before


def test_tts_without_a_respelling_calls_the_engine_with_two_arguments(client, state, monkeypatch):
    calls = []
    fake = state.tts.synthesize

    def synthesize(segments, voice=None):
        calls.append((segments, voice))
        return fake(segments, voice)

    monkeypatch.setattr(state.tts, "synthesize", synthesize)

    response = client.post("/api/tts", json={"segments": [{"lang": "lb", "text": "Moien!"}], "respell": {}})

    assert response.status_code == 200
    assert len(calls) == 1


@pytest.mark.parametrize(
    "respell",
    [{"a": "x"}, {"moien": ""}, {"moien": "[[ a"}, {"moien": "x" * 61}, {"moien": "a\nb"}, {f"wo{n}rd": "x" for n in "abcdef"}],
)
def test_tts_with_a_malformed_respelling_is_a_400_and_nothing_is_spoken(client, state, respell):
    state.tts.respell = "untouched"

    response = client.post("/api/tts", json={"segments": [{"lang": "lb", "text": "Moien!"}], "respell": respell})

    assert response.status_code == 400
    body = response.json()
    assert set(body) == {"error", "fix"} and body["error"].startswith("bad respelling") and body["fix"]
    assert state.tts.respell == "untouched"


@pytest.mark.parametrize("voice", [{"lb": {"rate": "fast"}}, {"en": {"pitch": None}}, {"lb": 1.0}])
def test_tts_with_a_malformed_voice_is_a_400_and_nothing_is_spoken(client, state, voice):
    state.tts.voice = "untouched"

    response = client.post("/api/tts", json={"segments": [{"lang": "lb", "text": "Moien!"}], "voice": voice})

    assert response.status_code == 400
    assert set(response.json()) == {"error", "fix"}
    assert response.json()["error"].startswith("bad voice settings: voice.")
    assert state.tts.voice == "untouched"


def test_tts_with_a_voice_that_is_no_object_is_a_422(client):
    response = client.post("/api/tts", json={"segments": [{"lang": "lb", "text": "Moien!"}], "voice": "slow"})

    assert response.status_code == 422


# --- voice ---


def test_voice_returns_the_defaults_the_ranges_and_the_sample_sentences(client):
    response = client.get("/api/voice")

    neutral = {"rate": 1.0, "pitch": 0.0, "volume": 1.0}
    assert response.status_code == 200
    assert response.json() == {
        "defaults": {"lb": {**neutral, "rate": 0.7}, "en": {**neutral, "pitch": -1.5}},
        "ranges": {"rate": [0.5, 1.5], "pitch": [-6, 6], "volume": [0.2, 2.0]},
        "samples": {"lb": "Gudde Moien! Wéi geet et dir haut?", "en": "Good morning! How are you today?"},
    }


def test_voice_defaults_follow_the_settings(tmp_path):
    settings = replace(Settings.from_env(), data_dir=tmp_path / "data", tts_rate_lb=0.6, tts_volume_en=9.0)

    with TestClient(create_app(settings)) as other:
        defaults = other.get("/api/voice").json()["defaults"]

    assert defaults["lb"] == {"rate": 0.6, "pitch": 0.0, "volume": 1.0}
    assert defaults["en"] == {"rate": 1.0, "pitch": -1.5, "volume": 2.0}


# --- mix ---


def test_mix_returns_the_default_and_the_four_levels(client):
    response = client.get("/api/mix")

    body = response.json()
    assert response.status_code == 200
    assert set(body) == {"default", "levels"}
    assert body["default"] == 0
    assert [level["value"] for level in body["levels"]] == [0, 1, 2, 3]
    assert [level["name"] for level in body["levels"]] == [
        "Opening phrase only",
        "A phrase here and there",
        "Phrases and sentences",
        "Mostly Luxembourgish",
    ]
    assert all(set(level) == {"value", "name", "hint"} and level["hint"] for level in body["levels"])


@pytest.mark.parametrize(("chat_mix", "default"), [(2, 2), (7, 3), (-1, 0)])
def test_mix_default_follows_the_setting_within_the_levels(tmp_path, chat_mix, default):
    settings = replace(Settings.from_env(), data_dir=tmp_path / "data", chat_mix=chat_mix)

    with TestClient(create_app(settings)) as other:
        assert other.get("/api/mix").json()["default"] == default


# --- turn ---


def test_turn_with_speech_returns_transcript_reply_audio_and_phrases(client, session_id):
    response = client.post("/api/turn", data={"session_id": session_id}, files=wav_upload(make_noise_wav(1.0)))

    body = response.json()
    assert response.status_code == 200
    assert set(body) == TURN_KEYS | AUDIO_KEYS
    assert body["heard"] is True
    assert body["transcript"] == FakeSTT.TEXT
    assert (body["mode"], body["mix"]) == ("chat", 0)
    assert body["segments"][0] == {"lang": "lb", "text": "Moien!", "gloss": "Hello!"}
    assert body["phrases_added"] == ["Moien!"]
    assert set(body["timings"]) == TIMING_KEYS
    rate, frames = opened_wav(base64.b64decode(body["audio_wav_base64"]))
    assert rate == body["sample_rate"] == SAMPLE_RATE
    assert [set(offset) for offset in body["segment_offsets"]] == [OFFSET_KEYS, OFFSET_KEYS]
    assert [offset["lang"] for offset in body["segment_offsets"]] == ["lb", "en"]
    assert body["segment_offsets"][-1]["end_s"] == pytest.approx(frames / rate)


def test_turn_with_silence_hears_nothing_and_never_calls_the_llm(client, session_id, state):
    response = client.post("/api/turn", data={"session_id": session_id}, files=wav_upload(make_silence_wav(1.0)))

    body = response.json()
    assert response.status_code == 200
    assert set(body) == {"heard", "transcript", "timings"}
    assert (body["heard"], body["transcript"]) == (False, "")
    assert set(body["timings"]) == {"stt_ms"}
    assert type(body["timings"]["stt_ms"]) is int
    assert state.llm.calls == []
    assert state.sessions.get(session_id).turns == []
    assert state.phrases.list() == []


def test_turn_with_text_runs_no_speech_to_text(client, session_id, no_stt):
    response = client.post(
        "/api/turn", data={"session_id": session_id, "text": "I had coffee.", "mode": "teacher", "stt_ms": "120"}
    )

    body = response.json()
    assert response.status_code == 200
    assert set(body) == TURN_KEYS | AUDIO_KEYS
    assert (body["transcript"], body["mode"]) == ("I had coffee.", "teacher")
    assert body["timings"]["stt_ms"] == 120
    assert opened_wav(base64.b64decode(body["audio_wav_base64"]))[0] == SAMPLE_RATE


def test_turn_with_text_and_audio_uses_the_text(client, session_id, no_stt):
    response = client.post(
        "/api/turn",
        data={"session_id": session_id, "text": "I had coffee."},
        files=wav_upload(make_noise_wav(0.5)),
    )

    assert response.status_code == 200
    assert response.json()["transcript"] == "I had coffee."


def test_turn_with_a_voice_speaks_the_reply_again_with_it(client, session_id, state, monkeypatch):
    calls = []
    synthesize = state.tts.synthesize

    def recording(segments, voice=None):
        audio = synthesize(segments, voice)
        if voice is not None:
            # Stands for audio that differs from the default one: the route must return this.
            audio.segment_offsets = [{**offset, "end_s": offset["end_s"] + 1.0} for offset in audio.segment_offsets]
        calls.append((list(segments), voice, audio))
        return audio

    monkeypatch.setattr(state.tts, "synthesize", recording)
    voice = {"lb": {"rate": 0.6, "pitch": 0, "volume": 1}, "en": {"rate": 1, "pitch": -3, "volume": 0.8}}

    response = client.post(
        "/api/turn", data={"session_id": session_id, "text": "I had coffee.", "voice": json.dumps(voice)}
    )

    body = response.json()
    reply = [Segment("lb", "Moien!"), Segment("en", "Hello! How was your day?")]
    assert response.status_code == 200
    assert set(body) == TURN_KEYS | AUDIO_KEYS
    assert [(segments, given) for segments, given, _ in calls] == [(reply, None), (reply, voice)]
    assert body["segment_offsets"] == calls[1][2].segment_offsets != calls[0][2].segment_offsets
    assert base64.b64decode(body["audio_wav_base64"]) == calls[1][2].wav_bytes
    assert body["sample_rate"] == SAMPLE_RATE


@pytest.mark.parametrize("fields", [{}, {"voice": ""}, {"voice": "{}"}, {"voice": "null"}])
def test_turn_without_a_voice_speaks_the_reply_once(client, session_id, state, monkeypatch, fields):
    calls = []
    synthesize = state.tts.synthesize
    monkeypatch.setattr(state.tts, "synthesize", lambda *args: calls.append(args) or synthesize(*args))

    response = client.post("/api/turn", data={"session_id": session_id, "text": "I had coffee.", **fields})

    assert response.status_code == 200
    assert set(response.json()) == TURN_KEYS | AUDIO_KEYS
    assert len(calls) == 1 and len(calls[0]) == 1


@pytest.mark.parametrize("voice", ["{not json", "[1]", '"slow"', '{"lb": {"rate": "fast"}}', '{"en": 3}'])
def test_turn_with_a_malformed_voice_is_a_400_before_any_engine_runs(client, session_id, state, no_stt, voice):
    response = client.post(
        "/api/turn",
        data={"session_id": session_id, "voice": voice},
        files=wav_upload(make_noise_wav(0.5)),
    )

    assert response.status_code == 400
    assert set(response.json()) == {"error", "fix"}
    assert response.json()["error"].startswith("bad voice settings: ")
    assert state.llm.calls == []
    assert state.sessions.get(session_id).turns == []


def test_turn_at_a_mix_level_passes_it_on_and_speaks_every_segment(client, session_id, state):
    state.llm.queue.append(MIXED_REPLY)

    response = client.post("/api/turn", data={"session_id": session_id, "text": "I had coffee.", "mix": "2"})

    body = response.json()
    assert response.status_code == 200
    assert set(body) == TURN_KEYS | AUDIO_KEYS
    assert (body["mode"], body["mix"], body["shape_warning"]) == ("chat", 2, "")
    assert state.llm.calls[0]["grounding"].startswith("mix level: 2\n")
    glosses = [segment.get("gloss") for segment in body["segments"]]
    assert glosses == [None, "a coffee with your sister", None, None, None]
    assert [offset["lang"] for offset in body["segment_offsets"]] == ["lb", "lb", "en", "lb", "en"]


@pytest.mark.parametrize("fields", [{}, {"mix": ""}])
def test_turn_without_a_mix_takes_the_setting(tmp_path, fields):
    settings = replace(Settings.from_env(), data_dir=tmp_path / "data", chat_mix=2)

    with TestClient(create_app(settings)) as other:
        other.app.state.llm.queue.append(MIXED_REPLY)
        session_id = other.post("/api/session").json()["session_id"]
        body = other.post("/api/turn", data={"session_id": session_id, "text": "I had coffee.", **fields}).json()

        assert (body["mix"], body["shape_warning"]) == (2, "")
        assert other.app.state.llm.calls[0]["grounding"].startswith("mix level: 2\n")


@pytest.mark.parametrize("mix", ["4", "-1", "1.5", "two", "true"])
def test_turn_with_a_mix_that_is_no_level_is_a_400_before_any_engine_runs(client, session_id, state, no_stt, mix):
    response = client.post(
        "/api/turn",
        data={"session_id": session_id, "mix": mix},
        files=wav_upload(make_noise_wav(0.5)),
    )

    assert response.status_code == 400
    assert response.json() == {"error": f"unknown mix level {mix!r}; known: 0 to 3", "fix": MIX_FIX}
    assert state.llm.calls == []
    assert state.sessions.get(session_id).turns == []


@pytest.mark.parametrize("fields", [{}, {"text": "   "}])
def test_turn_with_neither_audio_nor_text_is_a_400(client, session_id, state, fields):
    response = client.post("/api/turn", data={"session_id": session_id, **fields})

    assert response.status_code == 400
    assert set(response.json()) == {"error", "fix"}
    assert state.llm.calls == []


def test_turn_with_an_unknown_session_is_a_404_before_any_engine_runs(client, state, no_stt):
    response = client.post(
        "/api/turn", data={"session_id": "no-such-session"}, files=wav_upload(make_noise_wav(0.5))
    )

    assert response.status_code == 404
    assert response.json() == {"error": "unknown session", "fix": "reload the page"}
    assert state.llm.calls == []


def test_turn_with_an_auth_error_is_a_503_with_the_fix(client, session_id, state):
    state.llm.queue.append(LLMError("auth", "Not logged in"))

    response = client.post("/api/turn", data={"session_id": session_id, "text": "Hello"})

    assert response.status_code == 503
    assert response.json()["error"] == "Not logged in"
    assert "claude auth login" in response.json()["fix"]


# --- phrases ---


def test_phrases_total_grows_after_a_turn_in_the_temporary_data_folder(client, session_id, state, tmp_path):
    before = client.get("/api/phrases")

    client.post("/api/turn", data={"session_id": session_id, "text": "Hello there"})
    after = client.get("/api/phrases").json()

    assert before.status_code == 200
    assert before.json() == {"total": 0, "phrases": []}
    assert after["total"] == 1
    (entry,) = after["phrases"]
    assert (entry["lb"], entry["en"], entry["times_seen"], entry["sessions"]) == ("Moien!", "Hello!", 1, [session_id])
    assert state.phrases.path == tmp_path / "data" / "phrases.json"
    assert state.phrases.path.is_file()


def test_phrases_come_newest_last_seen_first(client, state):
    entry = {"en": "", "first_seen": "2026-01-01T00:00:00+00:00", "times_seen": 1, "sessions": []}
    stamps = {"old": "2026-01-01T00:00:00+00:00", "new": "2026-03-01T00:00:00+00:00", "mid": "2026-02-01T00:00:00+00:00"}
    state.phrases.path.parent.mkdir(parents=True, exist_ok=True)
    state.phrases.path.write_text(
        json.dumps([{**entry, "lb": lb, "last_seen": stamp} for lb, stamp in stamps.items()]), encoding="utf-8"
    )

    body = client.get("/api/phrases").json()

    assert body["total"] == 3
    assert [phrase["lb"] for phrase in body["phrases"]] == ["new", "mid", "old"]


# --- the page and the wiring ---


def test_the_page_is_served_at_the_root_with_its_script_and_style(client):
    page = client.get("/")

    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    for element_id in ("talk", "transcript", "send", "teacher", "player", "status"):
        assert f'id="{element_id}"' in page.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/style.css").status_code == 200


def test_the_page_loads_nothing_from_other_origins_and_builds_no_html_from_text():
    static = ROOT / "static"
    html = (static / "index.html").read_text(encoding="utf-8")
    script = (static / "app.js").read_text(encoding="utf-8")
    style = (static / "style.css").read_text(encoding="utf-8")

    assert sorted(re.findall(r'(?:src|href)="([^"]*)"', html)) == ["app.js", "style.css"]
    assert "//" not in html
    assert "url(" not in style and "@import" not in style
    assert not re.search(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write|\beval\(", script)
    assert set(re.findall(r"""["'](/[^"']*)["']""", script)) == {
        "/api/session",
        "/api/health",
        "/api/stt",
        "/api/turn",
        "/api/voice",
        "/api/tts",
        "/api/mix",
    }
    assert script.isascii()


def test_the_page_banner_also_reports_a_missing_lod_store():
    script = (ROOT / "static" / "app.js").read_text(encoding="utf-8")

    assert "health.lod && !health.lod.ready" in script
    english = (ROOT / "static" / "i18n" / "en.json").read_text(encoding="utf-8")
    assert 't("health.no_lod")' in script
    assert "teacher answers cannot be checked against LOD" in english
    assert "Run: python scripts/download_lod.py" in english


def test_the_page_has_a_closed_voice_panel_with_its_controls(client):
    html = plain_page(client.get("/").text)
    sliders = [f"voice-{lang}-{part}" for lang in ("lb", "en") for part in ("rate", "pitch", "volume")]

    assert re.search(r'<details id="voice"[^>]*>\s*<summary>Voice</summary>', html)
    assert not re.search(r'<details id="voice"[^>]*\bopen\b', html)
    for slider in sliders:
        assert re.search(rf'<input id="{slider}" type="range" step="[0-9.]+" disabled>', html)
        assert f'id="{slider}-value"' in html
    for button in ("voice-lb-sample", "voice-en-sample", "voice-reset"):
        assert re.search(rf'<button id="{button}" type="button" disabled>', html)


def test_the_page_script_sends_the_voice_and_guards_local_storage():
    script = (ROOT / "static" / "app.js").read_text(encoding="utf-8")

    assert 'form.append("voice", JSON.stringify(voice))' in script
    assert "voice: voiceValues()" in script
    # Every use of localStorage sits in a try block of one of the ten store functions.
    stores = ("storedVoice()", "storeVoice(values)", "storedMix()", "storeMix(value)")
    stores += ("storedSetting()", "storeSetting(value)", "storedTalk()", "storeTalk(press)")
    stores += ("storedLang()", "storeLang(code)")
    for use in re.finditer(r"localStorage\.", script):
        function = script[: use.start()].rsplit("\nfunction ", 1)[1]
        assert function.startswith(stores)
        assert "try {" in function and "catch" not in function
    assert len(re.findall(r"localStorage\.", script)) == 11


def test_the_page_has_a_mix_slider_next_to_the_teacher_button(client):
    html = plain_page(client.get("/").text)

    assert re.search(
        r'<button id="teacher"[^>]*>Teacher</button>\s*<label class="mix" for="mix">Luxembourgish mix\s*'
        r'<input id="mix" type="range" min="0" max="3" step="1" value="0" disabled>\s*'
        r'<output id="mix-value" for="mix"></output>',
        html,
    )


def test_the_page_script_keeps_the_mix_level_sends_it_and_marks_switched_segments_by_style():
    static = ROOT / "static"
    script = (static / "app.js").read_text(encoding="utf-8")
    style = (static / "style.css").read_text(encoding="utf-8")

    assert 'const MIX_STORE = "loci.mix";' in script
    assert 'request("/api/mix")' in script
    assert 'if (mixLevels) form.append("mix", mixInput.value);' in script
    assert "mixValue.textContent = level ? level.name" in script
    # The LB tag stays on the opening phrase and on teacher segments only.
    assert 'const mixed = turn.mode === "chat" && turn.mix > 0;' in script
    assert 'const switched = mixed && index > 0 && segment.lang === "lb";' in script
    assert 'if (segment.lang === "lb" && !switched) node.append(element("span", "tag", t("log.lb_tag")));' in script
    assert 'node.title = t("log.gloss_play", { gloss: segment.gloss });' in script
    assert ".segment.switched {" in style and ".segment.glossed {" in style


def test_health_is_still_reachable_next_to_the_page(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_an_unknown_api_path_is_a_404(client):
    assert client.get("/api/no-such-route").status_code == 404


def test_every_route_is_a_plain_function_so_a_slow_turn_runs_in_the_thread_pool():
    endpoints = [route.endpoint for route in api.router.routes]

    assert len(endpoints) == 8
    assert not any(inspect.iscoroutinefunction(endpoint) for endpoint in endpoints)
