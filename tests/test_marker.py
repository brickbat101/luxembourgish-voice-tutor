import base64
import io
import wave
from dataclasses import replace

import pytest

from app.engines.base import SYNTHETIC_COMMENT, Segment, mark_synthetic
from app.engines.factory import ENGINES, build_engine
from tests.audio import make_silence_wav


def info_comment(wav: bytes) -> str:
    """The ICMT text of the LIST/INFO chunk of a RIFF/WAVE file, "" when there is none."""
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    position = 12
    while position + 8 <= len(wav):
        chunk_id = wav[position : position + 4]
        size = int.from_bytes(wav[position + 4 : position + 8], "little")
        body = wav[position + 8 : position + 8 + size]
        if chunk_id == b"LIST" and body[:4] == b"INFO":
            inner = 4
            while inner + 8 <= len(body):
                tag = body[inner : inner + 4]
                length = int.from_bytes(body[inner + 4 : inner + 8], "little")
                if tag == b"ICMT":
                    return body[inner + 8 : inner + 8 + length].rstrip(b"\0").decode("utf-8")
                inner += 8 + length + (length & 1)
        position += 8 + size + (size & 1)
    return ""


def test_the_marker_adds_a_comment_and_keeps_the_audio_readable():
    plain = make_silence_wav(0.1)

    marked = mark_synthetic(plain)

    assert "Synthetic speech" in info_comment(marked)
    assert info_comment(marked) == SYNTHETIC_COMMENT
    assert b"Luxembourgish voice tutor (Piper TTS)" in marked
    with wave.open(io.BytesIO(plain)) as before, wave.open(io.BytesIO(marked)) as after:
        assert after.getnframes() == before.getnframes()
        assert after.getframerate() == before.getframerate()
        assert after.readframes(after.getnframes()) == before.readframes(before.getnframes())
    assert int.from_bytes(marked[4:8], "little") == len(marked) - 8
    assert len(marked) % 2 == 0


def test_the_marker_is_idempotent_and_leaves_other_bytes_alone():
    plain = make_silence_wav(0.1)

    assert mark_synthetic(mark_synthetic(plain)) == mark_synthetic(plain)
    for other in (b"not a wav", b"", b"RIFF", b"RIFF\0\0\0\0WAVE", plain[:30]):
        assert mark_synthetic(other) == other


def test_the_marker_pads_a_chunk_of_odd_size():
    plain = bytearray(make_silence_wav(0.1))
    plain += b"junk" + (3).to_bytes(4, "little") + b"abc\0"
    plain[4:8] = (len(plain) - 8).to_bytes(4, "little")

    marked = mark_synthetic(bytes(plain))

    assert info_comment(marked) == SYNTHETIC_COMMENT
    assert int.from_bytes(marked[4:8], "little") == len(marked) - 8


def test_every_served_wav_holds_the_comment(client):
    session_id = client.post("/api/session").json()["session_id"]

    tts = client.post("/api/tts", json={"segments": [{"lang": "lb", "text": "Moien."}]})
    turn = client.post("/api/turn", data={"session_id": session_id, "text": "Moien"}).json()
    topic = client.get("/api/practice/topics").json()["topics"][0]["id"]
    practice = client.post("/api/practice/start", data={"session_id": session_id, "topic_id": topic}).json()
    play = client.post("/api/listening/play", data={"item_id": "gemeng-adress", "speed": "normal"})

    assert tts.status_code == 200 and info_comment(tts.content) == SYNTHETIC_COMMENT
    assert info_comment(base64.b64decode(turn["audio_wav_base64"])) == SYNTHETIC_COMMENT
    assert info_comment(base64.b64decode(practice["audio_wav_base64"])) == SYNTHETIC_COMMENT
    assert play.status_code == 200 and info_comment(base64.b64decode(play.json()["audio_wav_base64"])) == SYNTHETIC_COMMENT


@pytest.mark.parametrize("name", sorted(ENGINES["tts"]))
def test_every_tts_engine_marks_its_audio(settings, name):
    engine = build_engine("tts", replace(settings, tts_engine=name))
    if not engine.ready():
        pytest.skip(f"the {name} engine is not ready: {getattr(engine, 'detail', '')}")

    audio = engine.synthesize([Segment("lb", "Moien.")])

    assert info_comment(audio.wav_bytes) == SYNTHETIC_COMMENT
