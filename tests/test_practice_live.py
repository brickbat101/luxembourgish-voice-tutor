"""A planted wrong verb form through the real voice, the real recogniser and the real tutor.

    pytest -m "live and models" -s tests/test_practice_live.py

The Luxembourgish voice says an answer, speech-to-text hears it, and the claude CLI judges
what was heard as the practice routes do it: two requests on the Claude subscription. The
talk runs on a temporary data folder that holds a copy of the LOD store, so the answers it
keeps are never in the real data folder.
"""

import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.content import WARMUPS_ASKED
from app.main import create_app
from app.practice_store import ANSWERS_FILE
from tests.realdata import REAL_DATA

pytestmark = [pytest.mark.live, pytest.mark.models]

TOPIC = "wunnen"
QUESTION = "Wou wunnt Dir?"  # the first question of the topic
PLANTED = "Ech wunnt zu Esch."  # "wunnt" is the form of du and hien; with ech it is "wunnen"
RIGHT = "Ech wunnen zu Esch."
MAX_CLI_CALLS = 4  # two judgements, each asked for once more at most
WORD = re.compile(r"[^\W\d_]+")


def show(label: str, value) -> None:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    encoding = sys.stdout.encoding or "utf-8"
    print(f"{label}: {text}".encode(encoding, "backslashreplace").decode(encoding))


def has(text: str, word: str) -> bool:
    """Whether `word` is a whole word of `text`, whatever its case."""
    return word in {found.casefold() for found in WORD.findall(text)}


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


@pytest.fixture
def real_settings(tmp_path):
    """Settings on the real models and voices, and a temporary data folder with a copy of the LOD store."""
    real = Settings.from_env()
    store = REAL_DATA / "lod" / "lod.sqlite"
    needed = {
        real.models_dir / "whisper-lb" / "model.bin": "python scripts/download_models.py --stt",
        store: "python scripts/download_lod.py",
    }
    for voice in (real.tts_voice_lb, real.tts_voice_en):
        for suffix in (".onnx", ".onnx.json"):
            needed[real.models_dir / "piper" / f"{voice}{suffix}"] = "python scripts/download_models.py --tts"
    for path, command in needed.items():
        if not path.is_file():
            pytest.skip(f"no {path}; run: {command}")

    data_dir = tmp_path / "data"
    (data_dir / "lod").mkdir(parents=True)
    shutil.copyfile(store, data_dir / "lod" / "lod.sqlite")
    return replace(real, data_dir=data_dir, stt_engine="faster_whisper", tts_engine="piper", llm_engine="claude_cli")


def said_and_heard(client, sentence: str) -> tuple[bytes, str]:
    """(`sentence` as the Luxembourgish voice says it, what speech-to-text makes of that)."""
    started = time.perf_counter()
    spoken = client.post("/api/tts", json={"segments": [{"lang": "lb", "text": sentence}]})
    tts_s = time.perf_counter() - started
    assert spoken.status_code == 200, spoken.text
    heard = client.post("/api/stt", files={"audio": ("speech.wav", spoken.content, "audio/wav")})
    assert heard.status_code == 200, heard.text
    transcript = heard.json()["text"]
    show("said", sentence)
    show("   heard", repr(transcript))
    show("   seconds", f"voice {tts_s:.2f}, speech-to-text {heard.json()['stt_ms'] / 1000:.2f}")
    return spoken.content, transcript


def judged(client, recording: bytes, transcript: str) -> dict:
    """The practice turn that follows `transcript`, confirmed as it was heard, as the first answer
    to the topic of a talk; the warm-up questions are let go, two skips each."""
    session_id = client.post("/api/session").json()["session_id"]
    opened = client.post("/api/practice/start", data={"session_id": session_id, "topic_id": TOPIC})
    assert opened.status_code == 200, opened.text
    for _ in range(2 * WARMUPS_ASKED):
        skipped = client.post("/api/practice/answer", data={"session_id": session_id, "text": "skip"})
        assert skipped.status_code == 200, skipped.text
    assert skipped.json()["question"]["lb"] == QUESTION
    started = time.perf_counter()
    response = client.post(
        "/api/practice/answer",
        data={"session_id": session_id, "text": transcript, "raw_text": transcript},
        files={"audio": ("speech.wav", recording, "audio/wav")},
    )
    took_s = time.perf_counter() - started
    turn = response.json()
    assert response.status_code == 200, turn
    show("   judged", turn["correction"] or f"no judgement: {turn['feedback_error']}")
    show("   tutor", [[segment["lang"], segment["text"]] for segment in turn["segments"]])
    show("   seconds", f"answer route {took_s:.1f}")

    # The answer is kept with its recording, in the temporary folder.
    store = client.app.state.practice_store
    (folder,) = store.path.glob(f"*-{session_id[:8]}")
    (line,) = (json.loads(text) for text in (folder / ANSWERS_FILE).read_text(encoding="utf-8").splitlines())
    assert (line["text"], line["raw_transcript"], line["audio_file"]) == (transcript, transcript, "1.wav")
    assert (folder / "1.wav").read_bytes() == recording
    assert (turn["save_error"], line["verdict"]) == ("", turn["correction"] and turn["correction"]["verdict"])
    return turn


def test_live_a_planted_wrong_verb_form_is_put_right_and_the_right_form_is_left_alone(request, tmp_path):
    if "live" not in request.config.getoption("markexpr"):
        pytest.skip("calls the real claude CLI on the user's subscription; run with -m live")
    settings = request.getfixturevalue("real_settings")

    with TestClient(create_app(settings)) as client:
        state = client.app.state
        if not state.llm.ready():
            pytest.skip(state.llm.detail)
        state.llm.runner = runner = CountedRunner(MAX_CLI_CALLS)
        health = client.get("/api/health").json()
        assert health["engines"]["stt"]["ready"] and health["engines"]["tts"]["ready"], health
        assert health["lod"]["ready"] is True
        assert tmp_path in state.practice_store.path.parents
        notes = []

        print()
        show("stt device", state.stt.device)
        recording, transcript = said_and_heard(client, PLANTED)
        if not transcript.strip():
            notes.append(f"{PLANTED} -> speech-to-text heard nothing; not judged")
        else:
            correction = judged(client, recording, transcript)["correction"]
            assert correction is not None
            if has(transcript, "wunnt"):
                # What was heard still holds the planted form: the tutor quotes it and puts it right.
                assert correction["verdict"] == "fix", correction
                assert has(correction["quote"], "wunnt"), correction
                assert not has(correction["better_lb"], "wunnt"), correction
                notes.append(f"{PLANTED} -> heard with 'wunnt': fix, better version {correction['better_lb']!r}")
            else:
                notes.append(
                    f"{PLANTED} -> heard as {transcript!r}, without 'wunnt': the recogniser changed the planted"
                    f" form, so nothing is checked; the verdict was {correction['verdict']}"
                )

        recording, transcript = said_and_heard(client, RIGHT)
        if not transcript.strip():
            notes.append(f"{RIGHT} -> speech-to-text heard nothing; not judged")
        else:
            correction = judged(client, recording, transcript)["correction"]
            assert correction is not None
            if has(transcript, "wunnen"):
                # The right form stays: no fix takes "wunnen" out of what was said.
                changed = (
                    correction["verdict"] == "fix"
                    and has(correction["quote"], "wunnen")
                    and not has(correction["better_lb"], "wunnen")
                )
                assert not changed, correction
                notes.append(f"{RIGHT} -> heard with 'wunnen': {correction['verdict']}, 'wunnen' is left as it is")
            else:
                notes.append(
                    f"{RIGHT} -> heard as {transcript!r}, without 'wunnen': the recogniser changed the form,"
                    f" so nothing is checked; the verdict was {correction['verdict']}"
                )

        for note in notes:
            show("result", note)
        show("claude -p calls", runner.calls)
