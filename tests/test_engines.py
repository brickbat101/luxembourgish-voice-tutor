import io
import subprocess
import sys
import wave

import pytest

from app.config import ROOT, Settings
from app.engines.base import (
    VOICE_RANGES,
    Judgement,
    LLMEngine,
    LLMError,
    Segment,
    STTEngine,
    TTSEngine,
    Turn,
    TutorReply,
    WrapUpText,
    ends_sentence,
    voice_tuning,
)
from app.engines.factory import UnavailableEngine
from app.engines.fake import FakeLLM, FakeSTT, FakeTTS

# 250 ms is not a whole number of samples at 22050 Hz.
ONE_SAMPLE_S = 1 / 22050

def make_wav(samples: bytes, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples)
    return buffer.getvalue()


def test_fakes_satisfy_the_protocols():
    assert isinstance(FakeSTT(), STTEngine)
    assert isinstance(FakeTTS(), TTSEngine)
    assert isinstance(FakeLLM(), LLMEngine)
    assert not isinstance(FakeSTT(), TTSEngine)


@pytest.mark.parametrize("engine_class", [FakeSTT, FakeTTS, FakeLLM])
def test_fakes_are_named_and_ready(engine_class):
    engine = engine_class()
    engine.warmup()

    assert engine.name == "fake"
    assert engine.ready() is True


def test_fake_stt_returns_nothing_for_silence():
    transcript = FakeSTT().transcribe(make_wav(b"\x00\x00" * 16000), "audio/wav")

    assert transcript.text == ""
    assert transcript.duration_s == 1.0


def test_fake_stt_returns_a_luxembourgish_sentence_for_sound():
    transcript = FakeSTT().transcribe(make_wav(b"\x10\x20" * 8000), "audio/wav")

    assert transcript.text == FakeSTT.TEXT
    assert transcript.language == "lb"
    assert transcript.duration_s == 0.5


def test_fake_stt_accepts_audio_that_is_not_wav():
    assert FakeSTT().transcribe(b"\x1aE\xdf\xa3webm", "audio/webm").text == FakeSTT.TEXT


def test_fake_tts_writes_16_bit_mono_22050_hz_wav():
    audio = FakeTTS().synthesize([Segment("lb", "Moien!"), Segment("en", "Hello!")])

    with wave.open(io.BytesIO(audio.wav_bytes)) as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 22050
        duration_s = wav.getnframes() / wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    assert audio.sample_rate == 22050
    assert duration_s == pytest.approx(0.4 + 0.25 + 0.4, abs=ONE_SAMPLE_S)
    assert any(frames)


def test_fake_tts_fills_segment_offsets():
    audio = FakeTTS().synthesize([Segment("lb", "Moien!"), Segment("en", "Hello!")])

    first, second = audio.segment_offsets
    assert (first["index"], first["lang"]) == (0, "lb")
    assert (second["index"], second["lang"]) == (1, "en")
    assert first["start_s"] == 0
    assert first["end_s"] == pytest.approx(0.4, abs=ONE_SAMPLE_S)
    assert second["start_s"] == pytest.approx(0.65, abs=ONE_SAMPLE_S)
    assert second["end_s"] == pytest.approx(1.05, abs=ONE_SAMPLE_S)


def test_fake_tts_records_the_voice_and_speaks_the_same_tones():
    tts = FakeTTS()
    segments = [Segment("lb", "Moien!"), Segment("en", "Hello!")]
    voice = {"lb": {"rate": 0.5, "volume": 0.2}}

    plain = tts.synthesize(segments)
    assert tts.voice is None
    tuned = tts.synthesize(segments, voice)

    assert tts.voice == voice
    assert tuned == plain


def test_fake_llm_replies_bilingually_and_records_calls():
    llm = FakeLLM()
    history = [Turn("user", "I had coffee with my sister.")]

    reply = llm.reply(history, "chat", "")

    assert reply.segments == [Segment("lb", "Moien!"), Segment("en", "Hello! How was your day?")]
    assert reply.new_phrases == [{"lb": "Moien!", "en": "Hello!"}]
    assert reply.mode == "chat"
    assert reply.claims == []
    assert llm.calls == [{"history": history, "mode": "chat", "grounding": ""}]


def test_fake_llm_plays_back_queued_outcomes_first():
    llm = FakeLLM()
    queued = TutorReply([Segment("en", "Hello!"), Segment("lb", "Moien!")])
    llm.queue += [queued, LLMError("auth", "not logged in")]
    history = [Turn("user", "Hello")]

    assert llm.reply(history, "chat", "") is queued
    with pytest.raises(LLMError, match="not logged in"):
        llm.reply(history, "teacher", "LOD facts")
    assert llm.reply(history, "chat", "").segments[0] == Segment("lb", "Moien!")
    assert [(call["mode"], call["grounding"]) for call in llm.calls] == [
        ("chat", ""),
        ("teacher", "LOD facts"),
        ("chat", ""),
    ]


def test_fake_llm_judges_an_answer_good_and_records_the_call_apart_from_the_replies():
    llm = FakeLLM()

    judgement = llm.judge("Wou wunnt Dir?", "Where do you live?", "Zu Esch.", "LOD facts")
    llm.reply([Turn("user", "Hello")], "chat", "")

    assert judgement == Judgement("good", FakeLLM.REACTION)
    assert (judgement.quote, judgement.better_lb, judgement.why_en, judgement.claims) == ("", "", "", [])
    assert llm.calls[0] == {
        "method": "judge",
        "question_lb": "Wou wunnt Dir?",
        "question_en": "Where do you live?",
        "answer": "Zu Esch.",
        "grounding": "LOD facts",
        "extra": "",
    }
    assert [call.get("method") for call in llm.calls] == ["judge", None]


def test_fake_llm_plays_back_queued_judgements_apart_from_queued_replies():
    llm = FakeLLM()
    queued = Judgement("unclear", "Sorry, I didn't catch that one.")
    reply = TutorReply([Segment("lb", "Moien!")])
    llm.judge_queue += [queued, LLMError("timeout", "no reply")]
    llm.queue.append(reply)

    assert llm.judge("Wat?", "What?", "dir op", "", "help used: yes") is queued
    with pytest.raises(LLMError, match="no reply"):
        llm.judge("Wat?", "What?", "Jo.", "")
    assert llm.judge("Wat?", "What?", "Jo.", "").verdict == "good"
    assert llm.reply([Turn("user", "Hello")], "chat", "") is reply
    assert [call.get("extra") for call in llm.calls] == ["help used: yes", "", "", None]


def test_an_engine_that_could_not_be_built_judges_nothing_and_says_why():
    engine = UnavailableEngine("claude_cli", "FileNotFoundError: practice.md")

    assert isinstance(engine, LLMEngine)
    with pytest.raises(RuntimeError, match="claude_cli engine is unavailable: FileNotFoundError: practice.md"):
        engine.judge("Wat?", "What?", "Jo.", "")
    with pytest.raises(RuntimeError, match="claude_cli engine is unavailable: FileNotFoundError: practice.md"):
        engine.wrap_up("Topic: Work (Aarbecht)")


def test_fake_llm_looks_back_on_a_talk_with_three_sentences_and_records_the_call():
    llm = FakeLLM()

    text = llm.wrap_up("Topic: Work (Aarbecht)")
    llm.judge("Wat?", "What?", "Jo.", "")

    assert text == FakeLLM.WRAP_UP
    assert all((text.words_en, text.grammar_en, text.task_en))
    assert llm.calls[0] == {"method": "wrap_up", "facts": "Topic: Work (Aarbecht)"}
    assert [call["method"] for call in llm.calls] == ["wrap_up", "judge"]


def test_fake_llm_plays_back_queued_wrap_ups_apart_from_queued_judgements():
    llm = FakeLLM()
    queued = WrapUpText("You had the words.", "Whole sentences.", "I could follow.")
    llm.wrap_up_queue += [queued, LLMError("timeout", "no reply")]
    llm.judge_queue.append(Judgement("unclear", "Sorry, I didn't catch that one."))

    assert llm.wrap_up("facts") is queued
    with pytest.raises(LLMError, match="no reply"):
        llm.wrap_up("facts")
    assert llm.wrap_up("facts") == FakeLLM.WRAP_UP
    assert llm.judge("Wat?", "What?", "dir op", "").verdict == "unclear"


def test_llm_error_carries_kind_and_detail():
    error = LLMError("auth", "not logged in")

    assert error.kind == "auth"
    assert error.detail == "not logged in"
    assert "not logged in" in str(error)


def test_fake_engines_do_not_import_the_real_engine_packages():
    code = (
        "import sys\n"
        "from app.config import Settings\n"
        "from app.engines.factory import KINDS, build_engine\n"
        "import app.main\n"
        "settings = Settings.from_env()\n"
        "assert [build_engine(kind, settings).name for kind in KINDS] == ['fake'] * 3 + ['none']\n"
        "loaded = {'faster_whisper', 'ctranslate2', 'piper', 'onnxruntime'} & set(sys.modules)\n"
        "assert not loaded, loaded\n"
    )

    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


# --- the end of a sentence ---


@pytest.mark.parametrize("text", ["Moien.", "Wéi geet et?", "Merci!", "Moien…", 'Si seet "Moien!"', "«Jo.»", "“Jo.” \n"])
def test_a_segment_that_ends_in_sentence_punctuation_ends_a_sentence(text):
    assert ends_sentence(text) is True


@pytest.mark.parametrize("text", ["Moien", "Moien,", "e laangen Dag", "Dat heescht:", "“Moien”", "U.S.A", "", " "])
def test_a_segment_that_ends_otherwise_stops_inside_a_sentence(text):
    assert ends_sentence(text) is False


# --- voice tuning ---


def test_voice_ranges():
    assert VOICE_RANGES == {"rate": (0.5, 1.5), "pitch": (-6, 6), "volume": (0.2, 2.0)}


def test_voice_tuning_without_a_voice_is_the_settings():
    neutral = {"rate": 1.0, "pitch": 0.0, "volume": 1.0}

    assert voice_tuning(Settings()) == {"lb": {**neutral, "rate": 0.7}, "en": {**neutral, "pitch": -1.5}}
    assert voice_tuning(Settings(), {}) == voice_tuning(Settings(), None) == voice_tuning(Settings())


def test_voice_tuning_lays_the_voice_over_the_settings_key_by_key():
    settings = Settings(tts_rate_lb=0.7, tts_volume_en=0.5)

    tuning = voice_tuning(settings, {"lb": {"pitch": 2}, "en": {"rate": 1.2, "pitch": 0}})

    assert tuning == {
        "lb": {"rate": 0.7, "pitch": 2.0, "volume": 1.0},
        "en": {"rate": 1.2, "pitch": 0.0, "volume": 0.5},
    }
    assert all(type(value) is float for values in tuning.values() for value in values.values())


def test_voice_tuning_clamps_the_voice_and_the_settings_to_the_ranges():
    settings = Settings(tts_rate_en=3.0, tts_pitch_en=-9.0, tts_volume_en=0.0)

    tuning = voice_tuning(settings, {"lb": {"rate": 0.1, "pitch": 6.5, "volume": 2.5}})

    assert tuning == {
        "lb": {"rate": 0.5, "pitch": 6.0, "volume": 2.0},
        "en": {"rate": 1.5, "pitch": -6.0, "volume": 0.2},
    }


def test_voice_tuning_ignores_unknown_languages_and_keys():
    voice = {"fr": {"rate": "vite"}, "de": 3, "lb": {"speed": 0.5, "rate": 1.0}}

    assert voice_tuning(Settings(), voice)["lb"] == {"rate": 1.0, "pitch": 0.0, "volume": 1.0}
    assert set(voice_tuning(Settings(), voice)) == {"lb", "en"}


@pytest.mark.parametrize(
    ("voice", "named"),
    [
        ("slow", "voice is not an object"),
        (["lb"], "voice is not an object"),
        ({"lb": 0.8}, "voice.lb is not an object"),
        ({"lb": {"rate": "0.8"}}, "voice.lb.rate is not a number"),
        ({"en": {"pitch": None}}, "voice.en.pitch is not a number"),
        ({"en": {"volume": True}}, "voice.en.volume is not a number"),
        ({"lb": {"rate": float("nan")}}, "voice.lb.rate is not a number"),
        ({"lb": {"pitch": float("inf")}}, "voice.lb.pitch is not a number"),
    ],
)
def test_voice_tuning_rejects_what_is_not_a_number(voice, named):
    with pytest.raises(ValueError, match=named):
        voice_tuning(Settings(), voice)
