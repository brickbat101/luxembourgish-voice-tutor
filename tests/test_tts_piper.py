import dataclasses
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.config import ROOT, Settings
from app.engines.base import Segment, TTSEngine
from app.engines.factory import build_engine
from app.engines.stt_faster_whisper import FasterWhisperSTT
from app.api import VOICE_SAMPLES
from app.pronunciation_store import checked_respell, checked_respelling, checked_word
from app.engines.tts_piper import (
    GAP_S,
    SAMPLE_RATE,
    SHORT_GAP_S,
    PiperTTS,
    amplify,
    clean_text,
    pronounce,
    resample,
)
from scripts import download_models
from tests.audio import cer

# 250 ms is not a whole number of samples at 22050 Hz.
ONE_SAMPLE_S = 1 / SAMPLE_RATE
VOICE_FILES = [
    "lb_LU-marylux-medium.onnx",
    "lb_LU-marylux-medium.onnx.json",
    "en_GB-cori-medium.onnx",
    "en_GB-cori-medium.onnx.json",
]
REPLY = [Segment("lb", "Moien, wéi geet et?"), Segment("en", "Hello, how are you today?")]
ROUND_TRIP_RECORDING = "moien.m4a"
ROUND_TRIP_MAX_CER = 0.30
ENGLISH_SENTENCE = "Hello, how are you today?"
RMS_FLOOR = 500  # of 32767
NEUTRAL = {"rate": 1.0, "pitch": 0.0, "volume": 1.0}
# The real voices are not deterministic: the duration of a sentence varies by about 5 % from
# run to run, so their tuning tests compare means over this many runs.
TUNING_RUNS = 20


def read_wav(wav_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav_bytes)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, SAMPLE_RATE)
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")


def window(samples: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    return samples[round(start_s * SAMPLE_RATE) : round(end_s * SAMPLE_RATE)]


def tone_hz(samples: np.ndarray) -> float:
    """The strongest frequency in `samples`."""
    spectrum = np.abs(np.fft.rfft(samples.astype(np.float64)))
    return float(np.argmax(spectrum)) * SAMPLE_RATE / len(samples)


# --- rules that need no voice --------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "cleaned"),
    [
        ("Moien, wéi geet et?", "Moien, wéi geet et?"),
        ("**Moien**, wéi geet _et_?", "Moien, wéi geet et?"),
        ("## Gudde `Moien`", "Gudde Moien"),
        ("  Moien,\n\twéi   geet et? ", "Moien, wéi geet et?"),
        ('"Moien!"', "Moien!"),
        ("„Gudde Moien“", "Gudde Moien"),
        ("“Good morning”", "Good morning"),
        ("«Bonjour»", "Bonjour"),
        ("'Hello'", "Hello"),
        ('*"Moien!"*', "Moien!"),
        ('He said "Moien" to me.', 'He said "Moien" to me.'),
        ("d'Kanner sinn doheem", "d'Kanner sinn doheem"),
        ("Ech sinn de Marc", "Ech sinn de Marc"),
        ("Say [[ wɜːd ]] now", "Say [ wɜːd ] now"),
        ("a ]] b [[ c [[[d]]]]", "a ] b [ c [d]"),
        ("[*[x]_]", "[x]"),
        ("[x] [] ][", "[x] [] ]["),
        ("", ""),
        (" ** ", ""),
        ('""', ""),
    ],
)
def test_text_is_cleaned_for_speech(raw, cleaned):
    assert clean_text(raw) == cleaned


def test_resample_keeps_audio_that_is_already_at_22050_hz():
    samples = np.arange(100, dtype=np.int16)

    assert resample(samples, SAMPLE_RATE) is samples
    assert len(resample(np.zeros(0, dtype=np.int16), 16000)) == 0


@pytest.mark.parametrize("rate", [16000, 24000, 44100])
def test_resample_keeps_duration_and_shape(rate):
    ramp = np.linspace(0, 20000, rate, dtype=np.int16)  # one second

    out = resample(ramp, rate)

    assert out.dtype == np.int16
    assert len(out) == SAMPLE_RATE
    assert np.abs(out - np.linspace(0, 20000, SAMPLE_RATE)).max() <= 3
    assert out[0] == 0 and out[-1] >= 19998


def test_resample_at_half_the_rate_halves_a_tone_and_doubles_its_length():
    tone = np.round(10000 * np.sin(2 * np.pi * 1000 / SAMPLE_RATE * np.arange(4410))).astype(np.int16)

    out = resample(tone, SAMPLE_RATE / 2)

    assert len(out) == 8820
    assert tone_hz(out) == pytest.approx(500, abs=3)


def test_amplify_scales_and_clips_without_wrapping():
    samples = np.array([-32768, -20000, -1, 0, 3, 20000, 32767], dtype=np.int16)

    assert amplify(samples, 1.0) is samples
    assert amplify(samples, 0.5).tolist() == [-16384, -10000, 0, 0, 2, 10000, 16384]
    assert amplify(samples, 2.0).tolist() == [-32767, -32767, -2, 0, 6, 32767, 32767]
    assert amplify(samples, 2.0).dtype == np.int16


# --- the engine around stub voices ---------------------------------------------------------


class StubVoice:
    """Stands in for piper.PiperVoice: one chunk per sentence, as long as the length scale asks.

    The audio is a constant level per voice, or a tone of `tone_hz` at that level.
    """

    SECONDS = 0.2
    LEVELS = {"lb_LU-marylux-medium": 1000, "en_GB-cori-medium": 2000}
    rates: dict = {}
    length_scales: dict = {}
    tone_hz = 0
    loaded: list = []
    active = 0
    most_active = 0

    def __init__(self, model_path):
        self.name = Path(model_path).stem
        self.config = SimpleNamespace(
            sample_rate=self.rates.get(self.name, SAMPLE_RATE),
            length_scale=self.length_scales.get(self.name, 1.0),
        )
        self.texts = []
        self.phonemized = []
        self.scales = []
        self.loaded.append(self)

    @classmethod
    def load(cls, model_path):
        return cls(model_path)

    def audio(self, length_scale: float) -> np.ndarray:
        count = round(self.SECONDS * self.config.sample_rate * length_scale)
        level = self.LEVELS[self.name]
        if not self.tone_hz:
            return np.full(count, level, dtype=np.int16)
        step = 2 * np.pi * self.tone_hz / self.config.sample_rate
        return np.round(level * np.sin(step * np.arange(count))).astype(np.int16)

    def phonemize(self, text):
        """The characters of each sentence: the stub has no phoneme set of its own."""
        self.phonemized.append(text)
        return [list(sentence) for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence]

    def synthesize(self, text, syn_config):
        self.texts.append(text)
        self.scales.append(syn_config.length_scale)
        cls = type(self)
        cls.active += 1
        cls.most_active = max(cls.most_active, cls.active)
        time.sleep(0.01)
        cls.active -= 1
        samples = self.audio(syn_config.length_scale)
        count = len(samples)
        for chunk in (samples[: count // 2], samples[count // 2 :]):
            yield SimpleNamespace(audio_int16_bytes=chunk.tobytes())


@pytest.fixture
def stub(monkeypatch):
    class Stub(StubVoice):
        rates = {}
        length_scales = {}
        loaded = []

    monkeypatch.setattr("piper.PiperVoice", Stub)
    return Stub


@pytest.fixture
def stub_settings(tmp_path):
    folder = tmp_path / "models" / "piper"
    folder.mkdir(parents=True)
    for name in VOICE_FILES:
        (folder / name).write_bytes(b"")
    # Neutral tuning: the tests of joining and gaps count samples of the untouched stub audio.
    # A data folder of its own: no pronunciation file of the user reaches these tests.
    return Settings(
        models_dir=tmp_path / "models",
        data_dir=tmp_path / "data",
        tts_engine="piper",
        tts_rate_lb=1.0,
        tts_pitch_en=0.0,
    )


@pytest.fixture
def engine(stub, stub_settings):
    tts = PiperTTS(stub_settings)
    tts.warmup()
    return tts


def voices(stub):
    return {voice.name.split("_")[0]: voice for voice in stub.loaded}


def test_engine_is_a_tts_engine_and_not_ready_before_warmup(stub_settings):
    tts = PiperTTS(stub_settings)

    assert isinstance(tts, TTSEngine)
    assert tts.name == "piper"
    assert tts.ready() is False
    assert tts.detail == "not loaded"
    with pytest.raises(RuntimeError, match="warmup"):
        tts.synthesize([Segment("lb", "Moien")])


@pytest.mark.parametrize("missing", VOICE_FILES)
def test_warmup_without_a_voice_file_names_the_download_script(stub, stub_settings, missing):
    (stub_settings.models_dir / "piper" / missing).unlink()
    tts = PiperTTS(stub_settings)

    with pytest.raises(FileNotFoundError, match="download_models.py --tts") as error:
        tts.warmup()
    assert missing in str(error.value)
    assert tts.ready() is False
    assert stub.loaded == []


def test_warmup_loads_both_configured_voices_once(stub, stub_settings):
    settings = dataclasses.replace(stub_settings, tts_voice_en="en_GB-alan-low")
    folder = settings.models_dir / "piper"
    (folder / "en_GB-alan-low.onnx").write_bytes(b"")
    (folder / "en_GB-alan-low.onnx.json").write_bytes(b"")
    stub.rates = {"en_GB-alan-low": 16000}
    tts = PiperTTS(settings)

    tts.warmup()
    tts.warmup()

    assert [voice.name for voice in stub.loaded] == ["lb_LU-marylux-medium", "en_GB-alan-low"]
    assert tts.ready() is True
    assert "lb=lb_LU-marylux-medium (22050 Hz)" in tts.detail
    assert "en=en_GB-alan-low (16000 Hz)" in tts.detail
    assert "cpu" in tts.detail


def test_each_segment_goes_to_the_voice_of_its_language(stub, engine):
    engine.synthesize(
        [
            Segment("lb", "Moien!"),
            Segment("en", "Hello!"),
            Segment("fr", "Bonjour!"),
            Segment("", "No language."),
            Segment("lb", "Äddi!"),
        ]
    )

    assert voices(stub)["lb"].texts == ["Moien!", "Äddi!"]
    assert voices(stub)["en"].texts == ["Hello!", "Bonjour!", "No language."]


def test_the_voice_gets_the_cleaned_text(stub, engine):
    engine.synthesize([Segment("lb", '**"Moien,   wéi geet et?"**')])

    assert voices(stub)["lb"].texts == ["Moien, wéi geet et?"]


def test_segments_are_joined_with_a_gap_between_them_only(stub, engine):
    audio = engine.synthesize(
        [Segment("lb", "Moien!"), Segment("en", "Hello!"), Segment("lb", "Äddi!")]
    )

    samples = read_wav(audio.wav_bytes)
    assert audio.sample_rate == SAMPLE_RATE
    assert len(samples) / SAMPLE_RATE == pytest.approx(3 * 0.2 + 2 * GAP_S, abs=2 * ONE_SAMPLE_S)
    assert samples[0] == 1000 and samples[-1] == 1000
    assert [(offset["index"], offset["lang"]) for offset in audio.segment_offsets] == [
        (0, "lb"),
        (1, "en"),
        (2, "lb"),
    ]
    first, second, third = audio.segment_offsets
    assert first["start_s"] == 0
    assert third["end_s"] == len(samples) / SAMPLE_RATE
    for offset, level in zip(audio.segment_offsets, (1000, 2000, 1000), strict=True):
        assert offset["end_s"] - offset["start_s"] == pytest.approx(0.2, abs=ONE_SAMPLE_S)
        assert set(window(samples, offset["start_s"], offset["end_s"])) == {level}
    for before, after in ((first, second), (second, third)):
        assert after["start_s"] - before["end_s"] == pytest.approx(GAP_S, abs=ONE_SAMPLE_S)
        assert set(window(samples, before["end_s"], after["start_s"])) == {0}


def test_one_segment_has_no_gap(engine):
    audio = engine.synthesize([Segment("en", "Hello!")])

    samples = read_wav(audio.wav_bytes)
    assert len(samples) == 4410 and set(samples) == {2000}
    assert audio.segment_offsets == [{"index": 0, "lang": "en", "start_s": 0.0, "end_s": 0.2}]


def test_the_gap_is_short_after_a_segment_that_stops_inside_a_sentence(engine):
    audio = engine.synthesize(
        [
            Segment("lb", "Dat ass schéin!"),
            Segment("en", "Starting the day with"),
            Segment("lb", "e Kaffi mat denger Schwëster"),
            Segment("en", "sounds lovely."),
            Segment("en", "What did you talk about?"),
        ]
    )

    samples = read_wav(audio.wav_bytes)
    offsets = audio.segment_offsets
    assert (GAP_S, SHORT_GAP_S) == (0.25, 0.08)
    gaps = [after["start_s"] - before["end_s"] for before, after in zip(offsets, offsets[1:])]
    assert gaps == [pytest.approx(gap, abs=ONE_SAMPLE_S) for gap in (GAP_S, SHORT_GAP_S, SHORT_GAP_S, GAP_S)]
    assert len(samples) / SAMPLE_RATE == pytest.approx(5 * 0.2 + 2 * GAP_S + 2 * SHORT_GAP_S, abs=4 * ONE_SAMPLE_S)
    assert offsets[0]["start_s"] == 0
    assert offsets[-1]["end_s"] == len(samples) / SAMPLE_RATE
    for offset, level in zip(offsets, (1000, 2000, 1000, 2000, 2000), strict=True):
        assert offset["end_s"] - offset["start_s"] == pytest.approx(0.2, abs=ONE_SAMPLE_S)
        assert set(window(samples, offset["start_s"], offset["end_s"])) == {level}
    for before, after in zip(offsets, offsets[1:]):
        assert set(window(samples, before["end_s"], after["start_s"])) == {0}


@pytest.mark.parametrize(
    ("text", "gap"),
    [
        ("Moien.", GAP_S),
        ("Wéi geet et?", GAP_S),
        ("Moien…", GAP_S),
        ('Si seet "Moien!"', GAP_S),
        ("Si seet «Moien.»", GAP_S),
        ("**Moien!**", GAP_S),
        ("“Moien!”", GAP_S),
        ("Moien", SHORT_GAP_S),
        ("Moien,", SHORT_GAP_S),
        ("e laangen Dag", SHORT_GAP_S),
        ("Dat heescht:", SHORT_GAP_S),
        ("“Moien”", SHORT_GAP_S),
    ],
)
def test_the_gap_follows_how_the_segment_before_it_ends(engine, text, gap):
    audio = engine.synthesize([Segment("lb", text), Segment("en", "Hello!")])

    first, second = audio.segment_offsets
    assert second["start_s"] - first["end_s"] == pytest.approx(gap, abs=ONE_SAMPLE_S)
    assert second["end_s"] == pytest.approx(0.2 + gap + 0.2, abs=2 * ONE_SAMPLE_S)


def test_an_empty_segment_does_not_change_the_gap_after_the_segment_before_it(engine):
    blank = Segment("en", " ")

    inside = engine.synthesize([Segment("lb", "e laangen Dag"), blank, Segment("en", "sounds good.")])
    closed = engine.synthesize([Segment("lb", "Dat ass schéin!"), blank, Segment("en", "Hello!")])

    for audio, gap in ((inside, SHORT_GAP_S), (closed, GAP_S)):
        first, _, last = audio.segment_offsets
        assert last["start_s"] - first["end_s"] == pytest.approx(gap, abs=ONE_SAMPLE_S)


def test_a_voice_at_another_rate_is_resampled_to_22050_hz(stub, stub_settings):
    stub.rates = {"en_GB-cori-medium": 16000}
    tts = PiperTTS(stub_settings)
    tts.warmup()

    audio = tts.synthesize([Segment("lb", "Moien!"), Segment("en", "Hello!")])

    samples = read_wav(audio.wav_bytes)
    first, second = audio.segment_offsets
    assert len(samples) / SAMPLE_RATE == pytest.approx(0.2 + GAP_S + 0.2, abs=2 * ONE_SAMPLE_S)
    assert second["end_s"] - second["start_s"] == pytest.approx(0.2, abs=ONE_SAMPLE_S)
    assert set(window(samples, second["start_s"], second["end_s"])) == {2000}
    assert set(window(samples, first["start_s"], first["end_s"])) == {1000}


@pytest.mark.parametrize("empty", ["", "   ", "**", '""'])
def test_an_empty_segment_adds_no_audio_and_no_gap(stub, engine, empty):
    blank = Segment("en", empty)

    audio = engine.synthesize([blank, Segment("lb", "Moien!"), blank, Segment("en", "Hello!")])

    samples = read_wav(audio.wav_bytes)
    assert len(samples) / SAMPLE_RATE == pytest.approx(0.2 + GAP_S + 0.2, abs=2 * ONE_SAMPLE_S)
    assert samples[0] == 1000 and samples[-1] == 2000
    leading, first, middle, last = audio.segment_offsets
    assert leading == {"index": 0, "lang": "en", "start_s": 0.0, "end_s": 0.0}
    assert first["start_s"] == 0
    assert middle["start_s"] == middle["end_s"] == first["end_s"]
    assert (middle["index"], middle["lang"]) == (2, "en")
    assert last["start_s"] - first["end_s"] == pytest.approx(GAP_S, abs=ONE_SAMPLE_S)
    assert voices(stub)["en"].texts == ["Hello!"]


def test_no_speech_at_all_gives_an_empty_wav(engine):
    for segments in ([], [Segment("lb", "")]):
        audio = engine.synthesize(segments)

        assert len(read_wav(audio.wav_bytes)) == 0
        assert len(audio.segment_offsets) == len(segments)


def test_synthesis_runs_one_call_at_a_time(stub, engine):
    lengths = []

    def work():
        audio = engine.synthesize([Segment("lb", "Moien!"), Segment("en", "Hello!")])
        lengths.append(len(read_wav(audio.wav_bytes)))

    threads = [threading.Thread(target=work) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(set(lengths)) == 1 and len(lengths) == 6
    assert stub.most_active == 1


# --- rate, pitch and volume around stub voices ---------------------------------------------


def warmed(settings) -> PiperTTS:
    tts = PiperTTS(settings)
    tts.warmup()
    return tts


def speak(tts, lang, **tuning) -> np.ndarray:
    """The samples of one segment spoken with `tuning` over the neutral values."""
    audio = tts.synthesize([Segment(lang, "Moien!")], {lang: {**NEUTRAL, **tuning}})
    samples = read_wav(audio.wav_bytes)
    assert audio.segment_offsets == [
        {"index": 0, "lang": lang, "start_s": 0.0, "end_s": len(samples) / SAMPLE_RATE}
    ]
    return samples


@pytest.mark.parametrize(("rate", "seconds"), [(0.5, 0.4), (0.8, 0.25), (1.0, 0.2), (1.25, 0.16)])
def test_rate_divides_the_duration(stub, engine, rate, seconds):
    samples = speak(engine, "lb", rate=rate)

    assert voices(stub)["lb"].scales == [pytest.approx(1 / rate)]
    assert len(samples) / SAMPLE_RATE == pytest.approx(seconds, abs=2 * ONE_SAMPLE_S)
    assert set(samples) == {1000}


def test_the_length_scale_of_the_voice_itself_is_multiplied_not_replaced(stub, stub_settings):
    stub.length_scales = {"lb_LU-marylux-medium": 1.2}
    tts = warmed(stub_settings)

    samples = speak(tts, "lb", rate=0.5)

    assert voices(stub)["lb"].scales == [pytest.approx(2.4)]
    assert len(samples) / SAMPLE_RATE == pytest.approx(0.48, abs=2 * ONE_SAMPLE_S)


def test_neutral_tuning_leaves_the_audio_of_the_voice_bit_for_bit(stub, engine):
    stub.tone_hz = 1000

    samples = speak(engine, "en")

    assert voices(stub)["en"].scales == [1.0]
    assert np.array_equal(samples, voices(stub)["en"].audio(1.0))
    assert tone_hz(samples) == pytest.approx(1000, abs=3)


@pytest.mark.parametrize(("pitch", "rate"), [(-6.0, 1.0), (6.0, 1.0), (-1.5, 1.0), (-6.0, 0.5), (3.0, 1.25)])
def test_pitch_moves_the_frequency_and_keeps_the_duration_the_rate_gives(stub, engine, pitch, rate):
    stub.tone_hz = 1000

    samples = speak(engine, "en", pitch=pitch, rate=rate)

    assert tone_hz(samples) == pytest.approx(1000 * 2 ** (pitch / 12), abs=6)
    assert len(samples) / SAMPLE_RATE == pytest.approx(0.2 / rate, abs=2 * ONE_SAMPLE_S)
    assert np.abs(samples).max() == pytest.approx(2000, abs=20)


def test_volume_scales_the_samples(engine):
    assert set(speak(engine, "en", volume=0.5)) == {1000}
    assert set(speak(engine, "lb", volume=2.0)) == {2000}


def test_volume_clips_at_the_16_bit_range_without_wrapping(stub, engine):
    stub.LEVELS = {"lb_LU-marylux-medium": 20000, "en_GB-cori-medium": -20000}

    assert set(speak(engine, "lb", volume=2.0)) == {32767}
    assert set(speak(engine, "en", volume=2.0)) == {-32767}


def test_values_outside_the_ranges_are_clamped(stub, engine):
    slow_and_loud = speak(engine, "lb", rate=0.1, volume=9.0)
    fast_low_and_quiet = speak(engine, "en", rate=4.0, pitch=-40.0, volume=0.0)

    assert voices(stub)["lb"].scales == [2.0]
    assert set(slow_and_loud) == {2000}
    assert voices(stub)["en"].scales == [pytest.approx(2**-0.5 / 1.5)]
    assert set(fast_low_and_quiet) == {400}


def test_unknown_languages_and_keys_are_ignored(stub, engine):
    voice = {"fr": {"rate": "vite"}, "de": 3, "lb": {"speed": 0.5, "volume": 0.5}}

    audio = engine.synthesize([Segment("lb", "Moien!")], voice)

    assert voices(stub)["lb"].scales == [1.0]
    assert set(read_wav(audio.wav_bytes)) == {500}


@pytest.mark.parametrize(
    "voice",
    [
        "slow",
        ["lb"],
        {"lb": 0.8},
        {"lb": {"rate": "0.8"}},
        {"en": {"pitch": None}},
        {"en": {"volume": True}},
        {"lb": {"rate": float("nan")}},
    ],
)
def test_a_voice_that_is_not_numbers_is_a_value_error_and_nothing_is_spoken(stub, engine, voice):
    with pytest.raises(ValueError, match="voice"):
        engine.synthesize([Segment("lb", "Moien!"), Segment("en", "Hello!")], voice)

    assert [loaded.texts for loaded in stub.loaded] == [[], []]


def test_without_a_voice_the_tuning_comes_from_the_settings(stub, stub_settings):
    tts = warmed(dataclasses.replace(stub_settings, tts_rate_lb=0.5, tts_volume_lb=0.5, tts_pitch_en=-6.0))

    audio = tts.synthesize([Segment("lb", "Moien!"), Segment("en", "Hello!")])

    samples = read_wav(audio.wav_bytes)
    first, second = audio.segment_offsets
    assert voices(stub)["lb"].scales == [2.0]
    assert voices(stub)["en"].scales == [pytest.approx(2**-0.5)]
    assert first["start_s"] == 0
    assert first["end_s"] == pytest.approx(0.4, abs=ONE_SAMPLE_S)
    assert set(window(samples, first["start_s"], first["end_s"])) == {500}
    assert second["start_s"] - first["end_s"] == pytest.approx(GAP_S, abs=ONE_SAMPLE_S)
    assert set(window(samples, first["end_s"], second["start_s"])) == {0}
    assert second["end_s"] - second["start_s"] == pytest.approx(0.2, abs=2 * ONE_SAMPLE_S)
    assert set(window(samples, second["start_s"], second["end_s"])) == {2000}
    assert second["end_s"] == len(samples) / SAMPLE_RATE


def test_the_default_settings_slow_luxembourgish_and_lower_english(stub, stub_settings):
    folders = {"models_dir": stub_settings.models_dir, "data_dir": stub_settings.data_dir}
    tts = warmed(dataclasses.replace(Settings(), **folders))

    tts.synthesize([Segment("lb", "Moien!"), Segment("en", "Hello!")])

    assert voices(stub)["lb"].scales == [pytest.approx(1 / 0.7)]
    assert voices(stub)["en"].scales == [pytest.approx(2 ** (-1.5 / 12))]


def test_a_voice_overrides_the_settings_key_by_key(stub, stub_settings):
    tts = warmed(dataclasses.replace(stub_settings, tts_rate_lb=0.5, tts_volume_lb=0.5))

    audio = tts.synthesize([Segment("lb", "Moien!")], {"lb": {"rate": 1.0}})

    assert voices(stub)["lb"].scales == [1.0]
    assert set(read_wav(audio.wav_bytes)) == {500}


# --- the pronunciation table around stub voices --------------------------------------------

PHONEMES = "[[ lˈʌksəmbˌɜːɡɪʃ ]]"
# Regular-expression escapes in a replacement are text like any other.
LITERAL = r"\1 \g<0> $& \\"
WORDS = {
    "en": {
        "Moien": "MOY-en",
        "Gudde  Moien": "goody MOY-en",
        "Äddi": "eddy",
        "Marc": PHONEMES,
        "tricky": LITERAL,
    }
}


def write_table(settings, table) -> Path:
    """The user's pronunciation file: `table` as JSON, or as it is when it is text."""
    path = settings.data_dir / "pronunciations.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(table if isinstance(table, str) else json.dumps(table), encoding="utf-8")
    return path


def a_second_later(path: Path, stamp: int) -> None:
    """Set the modification time of `path` one second after `stamp` (in ns)."""
    os.utime(path, ns=(stamp + 10**9, stamp + 10**9))


def given(tts, stub, lang, text) -> str:
    """What the voice is asked to say for one segment of `text` in `lang`."""
    tts.synthesize([Segment(lang, text)])
    return voices(stub)["lb" if lang == "lb" else "en"].texts[-1]


def test_the_shipped_table_respells_luxembourgish_for_the_english_voice_only(stub, engine):
    assert given(engine, stub, "en", "How does that work in Luxembourgish?") == (
        "How does that work in Luxembourghish?"
    )
    assert given(engine, stub, "lb", "Luxembourgish") == "Luxembourgish"
    # In an English sentence the language's own name is spoken as its English name.
    assert given(engine, stub, "en", "That is how you say it in Lëtzebuergesch.") == (
        "That is how you say it in Luxembourghish."
    )
    assert given(engine, stub, "lb", "Ech schwätze Lëtzebuergesch.") == "Ech schwätze Lëtzebuergesch."
    assert engine.detail == (
        "device=cpu, lb=lb_LU-marylux-medium (22050 Hz), en=en_GB-cori-medium (22050 Hz)"
    )


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("Moien!", "MOY-en!"),
        ("moien, MOIEN and mOiEn", "MOY-en, MOY-en and MOY-en"),
        ("ÄDDI an äddi", "eddy an eddy"),
        ("Moiens are moienish, émoien too", "Moiens are moienish, émoien too"),
        ("Bäddi, Äddis and Äddiö", "Bäddi, Äddis and Äddiö"),
        ("Luxembourgishness", "Luxembourgishness"),
        ("Moien's sound, Luxembourgish's too", "MOY-en's sound, Luxembourghish's too"),
        ("(moien) moien-moien 2moien3 «Moien»", "(MOY-en) MOY-en-MOY-en 2MOY-en3 «MOY-en»"),
        ("Gudde Moien, gudde   moien!", "goody MOY-en, goody MOY-en!"),
        ("Gudde Moiens", "Gudde Moiens"),
        ("Say **Gudde** _Moien_", "Say goody MOY-en"),
        ("So tricky.", f"So {LITERAL}."),
        ("Hello there", "Hello there"),
    ],
)
def test_whole_words_of_the_table_are_replaced_in_any_case(stub, stub_settings, text, spoken):
    write_table(stub_settings, WORDS)
    tts = warmed(stub_settings)

    assert given(tts, stub, "en", text) == spoken


def test_each_language_has_its_own_table(stub, stub_settings):
    write_table(stub_settings, {"lb": {"Moien": "Mojen"}, "en": {"hello": "hullo"}})
    tts = warmed(stub_settings)

    assert given(tts, stub, "lb", "Moien, hello, Luxembourgish") == "Mojen, hello, Luxembourgish"
    assert given(tts, stub, "en", "Moien, hello, Luxembourgish") == "Moien, hullo, Luxembourghish"
    # A segment in any other language is spoken by the English voice, with the English table.
    assert given(tts, stub, "fr", "Moien, hello") == "Moien, hullo"


def test_the_users_entry_wins_over_the_shipped_one_and_an_empty_one_removes_it(stub, stub_settings):
    path = write_table(stub_settings, {"en": {"luxembourgish": "Luxembourg-ish"}})
    tts = warmed(stub_settings)
    assert given(tts, stub, "en", "Luxembourgish") == "Luxembourg-ish"

    stamp = path.stat().st_mtime_ns
    write_table(stub_settings, {"en": {"LUXEMBOURGISH": ""}})
    a_second_later(path, stamp)

    assert given(tts, stub, "en", "Luxembourgish") == "Luxembourgish"
    assert "pronunciations" not in tts.detail


def test_an_edit_of_the_users_file_applies_to_the_next_request(stub, engine, stub_settings):
    assert given(engine, stub, "en", "hello") == "hello"

    path = write_table(stub_settings, {"en": {"hello": "hallo"}})
    assert given(engine, stub, "en", "hello, Luxembourgish") == "hallo, Luxembourghish"

    # The same size as before: only the modification time tells the two files apart.
    stamp = path.stat().st_mtime_ns
    write_table(stub_settings, {"en": {"hello": "hullo"}})
    a_second_later(path, stamp)
    assert given(engine, stub, "en", "hello") == "hullo"

    path.unlink()
    assert given(engine, stub, "en", "hello, Luxembourgish") == "hello, Luxembourghish"
    assert len(stub.loaded) == 2


def test_a_users_file_with_a_byte_order_mark_is_read(stub, engine, stub_settings):
    path = write_table(stub_settings, "")
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"en": {"hello": "hallo"}}).encode())

    assert given(engine, stub, "en", "hello") == "hallo"
    assert "pronunciations" not in engine.detail


@pytest.mark.parametrize(
    ("content", "problem"),
    [
        ('{"en": {"hello": "hallo"}', "JSONDecodeError"),
        ("", "JSONDecodeError"),
        ('["hello"]', "not an object of languages"),
        ('{"en": ["hello"]}', "'en' is not an object of words"),
        ('{"en": {"hello": 3}}', "en.hello is not a string: 3"),
        ('{"en": {"hello": "hallo"}, "fr": {}}', "unknown language 'fr'"),
        ('{"en": {" ": "hallo"}}', "'en' has an empty word"),
    ],
)
def test_a_broken_users_file_is_ignored_and_named_in_the_detail(
    stub, engine, stub_settings, content, problem
):
    path = write_table(stub_settings, content)

    assert given(engine, stub, "en", "hello, Luxembourgish") == "hello, Luxembourghish"
    assert engine.ready() is True
    assert engine.detail.startswith("device=cpu, lb=lb_LU-marylux-medium (22050 Hz)")
    assert str(path) in engine.detail and problem in engine.detail

    stamp = path.stat().st_mtime_ns
    write_table(stub_settings, {"en": {"hello": "hallo"}})
    a_second_later(path, stamp)

    assert "pronunciations" not in engine.detail
    assert given(engine, stub, "en", "hello") == "hallo"


def test_a_broken_users_file_shows_in_the_detail_before_anything_is_spoken(stub_settings):
    path = write_table(stub_settings, "{")

    tts = PiperTTS(stub_settings)

    assert tts.detail.startswith("not loaded; pronunciations: ")
    assert str(path) in tts.detail


def test_phonemes_in_the_text_of_a_segment_are_not_passed_on_as_phonemes(stub, stub_settings):
    write_table(stub_settings, WORDS)
    tts = warmed(stub_settings)

    assert given(tts, stub, "en", "Say [[ wɜːd ]] to [[[Moien]]]] and [*[Äddi") == (
        "Say [ wɜːd ] to [MOY-en] and [eddy"
    )
    assert given(tts, stub, "lb", "So [[ moien ]]") == "So [ moien ]"


def test_punctuation_after_phonemes_from_the_table_moves_into_them(stub, stub_settings):
    write_table(stub_settings, WORDS)
    tts = warmed(stub_settings)

    assert given(tts, stub, "en", "Ask Marc, then Marc and Marc?! Marc's here: Marc.") == (
        "Ask [[ lˈʌksəmbˌɜːɡɪʃ, ]] then [[ lˈʌksəmbˌɜːɡɪʃ ]] and [[ lˈʌksəmbˌɜːɡɪʃ?! ]] "
        "[[ lˈʌksəmbˌɜːɡɪʃ ]]'s here: [[ lˈʌksəmbˌɜːɡɪʃ. ]]"
    )


def test_text_inside_phonemes_is_not_replaced_again():
    words = {"moien": "[[ mˈojən ]]", "mˈojən": "wrong"}

    once = pronounce("Moien [[ moien ]] moien.", words)

    assert once == "[[ mˈojən ]] [[ moien ]] [[ mˈojən. ]]"
    assert pronounce(once, words) == once
    assert pronounce("Moien", {}) == "Moien"


# --- a respelling for one call, and the phonemes of a text --------------------------------


@pytest.mark.parametrize("word", ["Lëtzebuergesch", "moien", "ab", "x" * 40])
def test_a_word_is_one_run_of_letters(word):
    assert checked_word(word) == word


@pytest.mark.parametrize("word", ["a", "d'", "3", "x y", "ab3", "a_b", "", " moien", "x" * 41, None, 3])
def test_anything_else_is_not_a_word(word):
    with pytest.raises(ValueError):
        checked_word(word)


@pytest.mark.parametrize(
    ("value", "kept"),
    [
        ("moy-en", "moy-en"),
        ("  moy-en  ", "moy-en"),
        ("[[ ʃtɛfan ]]", "[[ ʃtɛfan ]]"),
        ("say [[ ʃtɛfan ]] now", "say [[ ʃtɛfan ]] now"),
        ("x" * 60, "x" * 60),
        ("a", "a"),
    ],
)
def test_a_respelling_is_short_text_with_no_or_one_phoneme_block(value, kept):
    assert checked_respelling(value) == kept


@pytest.mark.parametrize(
    "value",
    ["", "   ", "x" * 61, "a\nb", "a\tb", "a\x00b", "[[ a", "a ]]", "[a]", "[[ a ]] [[ b ]]", "[[ ]]", "[[ a ]]]", None, 3],
)
def test_a_respelling_that_breaks_a_rule_is_refused(value):
    with pytest.raises(ValueError):
        checked_respelling(value)


def test_a_respell_object_has_few_words_and_each_follows_the_rules():
    assert checked_respell({"moien": " MOY-en ", "Stefan": "[[ ʃtɛfan ]]"}) == {
        "moien": "MOY-en",
        "Stefan": "[[ ʃtɛfan ]]",
    }
    assert checked_respell({}) == {}
    for bad in ({"a": "x"}, {"moien": ""}, ["moien"], "moien", {f"wo{n}rd": "x" for n in "abcdef"}):
        with pytest.raises(ValueError):
            checked_respell(bad)


def test_a_respelling_for_one_call_wins_over_the_table_and_the_next_call_is_back_to_it(stub, stub_settings):
    write_table(stub_settings, WORDS)
    tts = warmed(stub_settings)
    segments = [Segment("en", "Moien and Luxembourgish and Äddi")]

    tts.synthesize(segments, respell={"MOIEN": "mo-yen", "luxembourgish": "lux"})
    tts.synthesize(segments)

    assert voices(stub)["en"].texts == ["mo-yen and lux and eddy", "MOY-en and Luxembourghish and eddy"]


def test_a_respelling_for_one_call_reaches_the_voice_with_its_phoneme_block_intact(stub, engine):
    engine.synthesize([Segment("lb", "Moien, Stefan!")], respell={"Stefan": "[[ ʃtɛfan ]]"})

    assert voices(stub)["lb"].texts == ["Moien, [[ ʃtɛfan! ]]"]


def test_a_respelling_applies_to_every_segment_of_the_call(stub, engine):
    engine.synthesize(
        [Segment("lb", "Stefan an Marc"), Segment("en", "Stefan and Marc")], respell={"stefan": "stef"}
    )

    assert voices(stub)["lb"].texts == ["stef an Marc"]
    assert voices(stub)["en"].texts == ["stef and Marc"]


def test_phonemes_apply_the_table_and_join_the_sentences_with_a_space(stub, stub_settings):
    write_table(stub_settings, WORDS)
    tts = warmed(stub_settings)

    assert tts.phonemes("en", "**Moien**. Äddi! Done") == "MOY-en. eddy! Done"
    assert voices(stub)["en"].phonemized == ["MOY-en. eddy! Done"]
    assert tts.phonemes("lb", "Moien") == "Moien"
    assert tts.phonemes("fr", "Äddi") == "eddy"
    assert tts.phonemes("lb", "") == ""


def test_phonemes_are_empty_before_the_voices_are_loaded(stub, stub_settings):
    assert PiperTTS(stub_settings).phonemes("lb", "Moien") == ""
    assert stub.loaded == []


def test_a_saved_table_applies_to_the_phonemes_and_the_next_speech_with_no_restart(stub, engine, stub_settings):
    assert engine.phonemes("en", "hello") == "hello"

    write_table(stub_settings, {"en": {"hello": "hallo"}})

    assert engine.phonemes("en", "hello") == "hallo"
    assert given(engine, stub, "en", "hello") == "hallo"
    assert len(stub.loaded) == 2


def test_factory_builds_the_piper_engine(stub, stub_settings):
    tts = build_engine("tts", stub_settings)

    assert isinstance(tts, PiperTTS)
    assert tts.ready() is True


def test_factory_reports_missing_voices_as_unavailable(stub_settings):
    (stub_settings.models_dir / "piper" / VOICE_FILES[0]).unlink()

    tts = build_engine("tts", stub_settings)

    assert tts.name == "piper"
    assert tts.ready() is False
    assert "FileNotFoundError" in tts.detail and "download_models.py --tts" in tts.detail


def test_importing_the_engine_module_does_not_import_piper():
    code = (
        "import sys\n"
        "import app.engines.tts_piper\n"
        "loaded = {'piper', 'onnxruntime'} & set(sys.modules)\n"
        "assert not loaded, loaded\n"
    )

    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


# --- the download script -------------------------------------------------------------------


def test_download_fetches_only_the_voices_that_are_missing(tmp_path, monkeypatch, capsys):
    folder = tmp_path / "piper"
    folder.mkdir()
    (folder / "lb_LU-marylux-medium.onnx").write_bytes(b"model")
    (folder / "lb_LU-marylux-medium.onnx.json").write_bytes(b"{}")
    (folder / "en_GB-cori-medium.onnx").write_bytes(b"")
    calls = []
    monkeypatch.setattr(
        "piper.download_voices.download_voice", lambda voice, target: calls.append((voice, target))
    )

    download_models.download_tts(Settings(models_dir=tmp_path))

    assert calls == [("en_GB-cori-medium", folder)]
    assert "lb_LU-marylux-medium is present" in capsys.readouterr().out


def test_download_creates_the_piper_folder_and_has_a_tts_flag(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "piper.download_voices.download_voice", lambda voice, target: calls.append((voice, target))
    )
    monkeypatch.setenv("LOCI_MODELS_DIR", str(tmp_path / "models"))

    download_models.main(["--tts"])

    folder = tmp_path / "models" / "piper"
    assert folder.is_dir()
    assert calls == [("lb_LU-marylux-medium", folder), ("en_GB-cori-medium", folder)]


# --- the real voices (pytest -m models -s) -------------------------------------------------


def timed(tts, segments, label):
    start = time.perf_counter()
    audio = tts.synthesize(segments)
    seconds = time.perf_counter() - start
    duration_s = len(read_wav(audio.wav_bytes)) / SAMPLE_RATE
    print(f"\n[tts] {label}: synthesized in {seconds:.2f} s, {duration_s:.2f} s of audio")
    return audio


def rms(samples) -> float:
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


@pytest.fixture(scope="module")
def real_settings():
    settings = Settings.from_env()
    folder = settings.models_dir / "piper"
    for voice in (settings.tts_voice_lb, settings.tts_voice_en):
        for path in (folder / f"{voice}.onnx", folder / f"{voice}.onnx.json"):
            if not path.is_file():
                pytest.skip(f"no voice file {path}; run: python scripts/download_models.py --tts")
    return settings


@pytest.fixture(scope="module")
def tts_data_dir(tmp_path_factory) -> Path:
    """The data folder of the real engine: no pronunciation file of the user reaches the tests."""
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="module")
def tts(real_settings, tts_data_dir):
    piper_tts = PiperTTS(dataclasses.replace(real_settings, data_dir=tts_data_dir))
    start = time.perf_counter()
    piper_tts.warmup()
    print(f"\n[tts] loaded in {time.perf_counter() - start:.2f} s: {piper_tts.detail}")
    return piper_tts


@pytest.fixture(scope="module")
def stt(real_settings):
    model_dir = real_settings.models_dir / "whisper-lb"
    if not (model_dir / "model.bin").is_file():
        pytest.skip(f"no model in {model_dir}; run: python scripts/download_models.py --stt")
    whisper = FasterWhisperSTT(real_settings)
    whisper.warmup()
    print(f"\n[stt] {whisper.detail}")
    return whisper


@pytest.fixture(scope="module")
def lod_texts(real_settings):
    path = real_settings.data_dir / "lod_audio" / "texts.json"
    if not path.is_file():
        pytest.skip(f"no {path}; run: python scripts/download_models.py --lod-audio")
    return json.loads(path.read_text(encoding="utf-8"))


def round_trip(tts, stt, lang, text):
    audio = timed(tts, [Segment(lang, text)], f"{lang} {text!r}")
    transcript = stt.transcribe(audio.wav_bytes, "audio/wav")
    error_rate = cer(text, transcript.text)
    print(f"[tts -> stt {stt.device}] {lang} voice: CER {error_rate:.3f} -> {transcript.text!r}")
    return error_rate


@pytest.mark.models
def test_real_voices_load_on_cpu_at_22050_hz(tts, real_settings):
    assert tts.ready() is True
    assert f"lb={real_settings.tts_voice_lb} (22050 Hz)" in tts.detail
    assert f"en={real_settings.tts_voice_en} (22050 Hz)" in tts.detail
    assert [type(voice).__name__ for voice in tts._voices.values()] == ["PiperVoice"] * 2
    for voice in tts._voices.values():
        assert voice.session.get_providers() == ["CPUExecutionProvider"]


@pytest.mark.models
def test_bilingual_reply_is_one_wav_with_two_spoken_segments(tts):
    for segment in REPLY:
        timed(tts, [segment], f"{segment.lang} {segment.text!r}")
    audio = timed(tts, REPLY, "both segments joined")

    samples = read_wav(audio.wav_bytes)
    duration_s = len(samples) / SAMPLE_RATE
    assert audio.sample_rate == SAMPLE_RATE
    assert duration_s > 1.5
    first, second = audio.segment_offsets
    assert (first["index"], first["lang"]) == (0, "lb")
    assert (second["index"], second["lang"]) == (1, "en")
    assert 0 == first["start_s"] < first["end_s"] < second["start_s"] < second["end_s"]
    assert second["start_s"] - first["end_s"] == pytest.approx(GAP_S, abs=ONE_SAMPLE_S)
    assert second["end_s"] == pytest.approx(duration_s)
    levels = [
        rms(window(samples, first["start_s"], first["end_s"])),
        rms(window(samples, first["end_s"], second["start_s"])),
        rms(window(samples, second["start_s"], second["end_s"])),
    ]
    print(f"[tts] RMS lb segment {levels[0]:.0f}, gap {levels[1]:.0f}, en segment {levels[2]:.0f}")
    assert levels[0] > RMS_FLOOR
    assert levels[1] < RMS_FLOOR
    assert levels[2] > RMS_FLOOR


@pytest.mark.models
def test_luxembourgish_speech_survives_the_round_trip_through_stt(tts, stt, lod_texts):
    error_rate = round_trip(tts, stt, "lb", lod_texts[ROUND_TRIP_RECORDING])

    assert error_rate <= ROUND_TRIP_MAX_CER


@pytest.mark.models
def test_other_round_trips_are_reported(tts, stt, lod_texts):
    # Printed, not asserted: the STT model is forced to Luxembourgish.
    for name, text in lod_texts.items():
        if name != ROUND_TRIP_RECORDING:
            round_trip(tts, stt, "lb", text)
    round_trip(tts, stt, "en", ENGLISH_SENTENCE)


QUESTION = "How does that work in Luxembourgish?"
PRONUNCIATION_RUNS = 5


def spoken_phonemes(tts, lang, text) -> str:
    """The phonemes of what the real voice of `lang` is given for one segment of `text`."""
    voice, texts = tts._voices[lang], []
    synthesize = voice.synthesize

    def record(text, config):
        texts.append(text)
        return synthesize(text, config)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(voice, "synthesize", record)
        audio = tts.synthesize([Segment(lang, text)])
    assert rms(read_wav(audio.wav_bytes)) > RMS_FLOOR
    return " | ".join("".join(sentence) for sentence in voice.phonemize(texts[0]))


@pytest.mark.models
def test_the_english_voice_says_luxembourgish_with_a_hard_g(tts):
    untouched = "".join(tts._voices["en"].phonemize(QUESTION)[0])

    phonemes = spoken_phonemes(tts, "en", QUESTION)

    print(f"\n[tts] en {QUESTION!r}: {untouched} -> {phonemes}")
    assert "ɡɪʃ" in phonemes
    assert "dʒ" not in phonemes
    assert phonemes.endswith("?")


@pytest.mark.models
def test_the_respelled_word_takes_about_as_long_as_luxembourg(tts):
    def seconds(text):
        takes = [tts.synthesize([Segment("en", text)]) for _ in range(PRONUNCIATION_RUNS)]
        return float(np.mean([len(read_wav(take.wav_bytes)) for take in takes])) / SAMPLE_RATE

    respelled, plain = seconds(QUESTION), seconds(QUESTION.replace("Luxembourgish", "Luxembourg"))

    print(
        f"\n[tts] en, mean of {PRONUNCIATION_RUNS}: {QUESTION!r} {respelled:.3f} s, "
        f"with 'Luxembourg' {plain:.3f} s (x{respelled / plain:.3f})"
    )
    assert respelled / plain == pytest.approx(1.0, abs=0.25)


@pytest.mark.models
def test_phonemes_from_the_users_file_keep_the_punctuation_after_them(tts, tts_data_dir):
    path = tts_data_dir / "pronunciations.json"
    path.write_text(json.dumps({"en": {"Luxembourgish": PHONEMES}}), encoding="utf-8")
    try:
        phonemes = spoken_phonemes(tts, "en", "In Luxembourgish, we say hello. Is it Luxembourgish?")
    finally:
        path.unlink()

    print(f"\n[tts] en with {PHONEMES}: {phonemes}")
    assert "lˈʌksəmbˌɜːɡɪʃ, wiː sˈeɪ həlˈəʊ. | ɪz ɪt lˈʌksəmbˌɜːɡɪʃ?" in phonemes
    assert spoken_phonemes(tts, "en", QUESTION).endswith("lˈʌksɪmbˌɜːɡɪʃ?")


def f0_values(samples: np.ndarray, low_hz: float = 75.0, high_hz: float = 400.0) -> list[float]:
    """The fundamental of each voiced 40 ms frame, from the peak of its autocorrelation."""
    frame, hop = round(0.04 * SAMPLE_RATE), round(0.01 * SAMPLE_RATE)
    low, high = int(SAMPLE_RATE / high_hz), int(SAMPLE_RATE / low_hz)
    signal = samples.astype(np.float64)
    loud = 0.1 * rms(samples)
    found = []
    for start in range(0, len(signal) - frame, hop):
        voiced = signal[start : start + frame]
        voiced = voiced - voiced.mean()
        energy = float(np.dot(voiced, voiced))
        if np.sqrt(energy / frame) < loud:
            continue
        corr = np.correlate(voiced, voiced, "full")[frame - 1 :] / energy
        lag = low + int(np.argmax(corr[low:high]))
        if corr[lag] > 0.5:
            # A parabola through the peak and its neighbours places it between two samples.
            before, peak, after = corr[lag - 1 : lag + 2]
            bend = before - 2 * peak + after
            found.append(SAMPLE_RATE / (lag + (0.5 * (before - after) / bend if bend else 0.0)))
    return found


def spoken_sample(tts, lang, **tuning) -> list[np.ndarray]:
    """TUNING_RUNS takes of the page's sample sentence with `tuning` over the neutral values."""
    segments, voice = [Segment(lang, VOICE_SAMPLES[lang])], {lang: {**NEUTRAL, **tuning}}
    return [read_wav(tts.synthesize(segments, voice).wav_bytes) for _ in range(TUNING_RUNS)]


def mean_seconds(takes: list[np.ndarray]) -> float:
    return float(np.mean([len(samples) for samples in takes])) / SAMPLE_RATE


def median_f0(takes: list[np.ndarray]) -> float:
    return float(np.median([value for samples in takes for value in f0_values(samples)]))


@pytest.mark.models
def test_a_lower_rate_makes_the_luxembourgish_voice_slower(tts):
    normal, slower, slowest = (mean_seconds(spoken_sample(tts, "lb", rate=rate)) for rate in (1.0, 0.85, 0.5))

    print(
        f"\n[tts] lb sample, mean of {TUNING_RUNS}: rate 1.0 {normal:.3f} s, rate 0.85 {slower:.3f} s "
        f"(x{slower / normal:.3f}, 1/0.85 = {1 / 0.85:.3f}), rate 0.5 {slowest:.3f} s (x{slowest / normal:.3f})"
    )
    # The voice follows Piper's length scale only in part: the audio grows by less than 1 / rate.
    assert 1.0 < slower / normal <= 1.1 / 0.85
    assert 1.25 < slowest / normal <= 2.2


@pytest.mark.models
def test_a_lower_pitch_lowers_the_english_voice_and_keeps_its_duration(tts):
    plain, lower = spoken_sample(tts, "en", pitch=0.0), spoken_sample(tts, "en", pitch=-1.5)

    seconds = mean_seconds(plain), mean_seconds(lower)
    f0 = median_f0(plain), median_f0(lower)
    print(
        f"\n[tts] en sample, mean of {TUNING_RUNS}: pitch 0 {seconds[0]:.3f} s, median f0 {f0[0]:.1f} Hz; "
        f"pitch -1.5 {seconds[1]:.3f} s (x{seconds[1] / seconds[0]:.3f}), median f0 {f0[1]:.1f} Hz "
        f"(x{f0[1] / f0[0]:.3f}, 2**(-1.5/12) = {2 ** (-1.5 / 12):.3f})"
    )
    assert 0.85 < f0[1] / f0[0] < 0.97
    assert seconds[1] / seconds[0] == pytest.approx(1.0, abs=0.1)
