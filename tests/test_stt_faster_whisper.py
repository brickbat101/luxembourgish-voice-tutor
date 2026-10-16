import dataclasses
import io
import json
import tempfile
import threading
import time
import wave
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.engines import stt_faster_whisper as stt
from app.engines.base import STTEngine
from app.engines.stt_faster_whisper import FasterWhisperSTT, build_transcript, suffix_for
from tests.audio import cer, make_noise_wav, make_silence_wav

TRANSCRIBE_KWARGS = {
    "language": "lb",
    "task": "transcribe",
    "beam_size": 5,
    "vad_filter": True,
    "vad_parameters": {"min_silence_duration_ms": 500},
    "condition_on_previous_text": False,
    "no_speech_threshold": 0.6,
}
MAX_CER = 0.35
LOD_RECORDINGS = ["benotzen.m4a", "goen.m4a", "moien.m4a"]


# --- helpers -------------------------------------------------------------------------------


def test_silence_and_noise_wavs_have_the_asked_length_and_level():
    with wave.open(io.BytesIO(make_silence_wav(0.5))) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        assert wav.getnframes() == 8000
        assert not any(wav.readframes(8000))
    with wave.open(io.BytesIO(make_noise_wav(0.5, amplitude=0.01))) as wav:
        assert wav.getnframes() == 8000
        samples = memoryview(wav.readframes(8000)).cast("h")
    assert 300 <= max(samples) <= 328 and -328 <= min(samples) <= -300


def test_cer_ignores_case_and_punctuation():
    assert cer("gudde Moien, Madamm!", "Gudde Moien Madamm.") == 0
    assert cer("abcd", "abxd") == 0.25
    assert cer("abcd", "abcde") == 0.25
    assert cer("abcd", "") == 1


# --- rules that need no model --------------------------------------------------------------


@pytest.mark.parametrize(
    ("mime", "suffix"),
    [
        ("audio/webm;codecs=opus", ".webm"),
        ("audio/webm", ".webm"),
        ("audio/ogg; codecs=opus", ".ogg"),
        ("audio/mp4", ".m4a"),
        ("audio/x-m4a", ".m4a"),
        ("audio/wav", ".wav"),
        ("audio/x-wav", ".wav"),
        ("AUDIO/WAVE", ".wav"),
        ("application/octet-stream", ".audio"),
        ("", ".audio"),
    ],
)
def test_suffix_follows_the_mime_type(mime, suffix):
    assert suffix_for(mime) == suffix


def segment(start, end, text):
    return {"start": start, "end": end, "text": text}


def test_no_segment_gives_an_empty_transcript():
    transcript = build_transcript([], 5.0)

    assert (transcript.text, transcript.language, transcript.duration_s) == ("", "lb", 5.0)
    assert transcript.segments == []


def test_less_than_300_ms_of_speech_gives_an_empty_transcript():
    assert build_transcript([segment(1.0, 1.2, "Jo.")], 5.0).text == ""
    assert build_transcript([segment(1.0, 1.1, "Jo"), segment(3.0, 3.1, "nee")], 5.0).segments == []
    assert build_transcript([segment(1.0, 1.4, "Jo.")], 5.0).text == "Jo."


@pytest.mark.parametrize("text", ["jo jo jo", "Jo, jo, jo, jo.", "Merci! merci! MERCI!"])
def test_one_token_repeated_three_times_gives_an_empty_transcript(text):
    assert build_transcript([segment(0.0, 2.0, text)], 2.0).text == ""


@pytest.mark.parametrize("text", ["jo jo", "jo jo jo nee", "Moien"])
def test_other_short_transcripts_are_kept(text):
    assert build_transcript([segment(0.0, 2.0, text)], 2.0).text == text


def test_a_token_repeated_across_segments_gives_an_empty_transcript():
    segments = [segment(0.0, 1.0, "Jo."), segment(1.5, 2.5, "Jo."), segment(3.0, 4.0, "Jo.")]

    assert build_transcript(segments, 4.0).text == ""


def test_segment_texts_are_joined_with_single_spaces():
    segments = [segment(0.0, 1.5, "Moien,"), segment(2.0, 3.0, "wéi geet et?")]

    transcript = build_transcript(segments, 3.2)

    assert transcript.text == "Moien, wéi geet et?"
    assert transcript.segments == segments
    assert transcript.duration_s == 3.2


# --- the engine around a stub model --------------------------------------------------------


class StubModel:
    """Stands in for faster_whisper.WhisperModel."""

    cuda_works = True
    segments = [(0.0, 1.5, " Moien,"), (2.0, 3.0, " wéi geet et? ")]
    created = []

    def __init__(self, path, device, compute_type):
        self.path, self.device, self.compute_type = path, device, compute_type
        self.calls = []
        self.audio = []
        self.active = 0
        self.most_active = 0
        self.threads = {threading.get_ident()}
        self.created.append(self)

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        self.audio.append(audio)
        return self._segments(), SimpleNamespace(duration=len(audio) / 16000)

    def _segments(self):
        # Like the real model, the work (and a cuBLAS failure) happens while iterating.
        if self.device == "cuda" and not self.cuda_works:
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
        self.threads.add(threading.get_ident())
        self.active += 1
        self.most_active = max(self.most_active, self.active)
        time.sleep(0.02)
        self.active -= 1
        for start, end, text in self.segments:
            yield SimpleNamespace(start=start, end=end, text=text)


@pytest.fixture
def stub(monkeypatch):
    class Stub(StubModel):
        created = []

    monkeypatch.setattr("faster_whisper.WhisperModel", Stub)
    return Stub


@pytest.fixture
def stub_settings(tmp_path):
    model_dir = tmp_path / "models" / "whisper-lb"
    model_dir.mkdir(parents=True)
    (model_dir / "model.bin").write_bytes(b"")
    return Settings(models_dir=tmp_path / "models", stt_device="cpu")


@pytest.fixture
def temp_dir(tmp_path, monkeypatch):
    path = tmp_path / "temp"
    path.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(path))
    return path


def test_engine_is_an_stt_engine_and_not_ready_before_warmup(stub_settings):
    engine = FasterWhisperSTT(stub_settings)

    assert isinstance(engine, STTEngine)
    assert engine.name == "faster_whisper"
    assert engine.ready() is False
    with pytest.raises(RuntimeError, match="warmup"):
        engine.transcribe(make_silence_wav(0.1), "audio/wav")


def test_warmup_without_the_model_names_the_download_script(tmp_path):
    engine = FasterWhisperSTT(Settings(models_dir=tmp_path))

    with pytest.raises(FileNotFoundError, match="download_models.py --stt"):
        engine.warmup()
    assert engine.ready() is False


def test_cpu_loads_int8_without_a_trial_transcribe(stub, stub_settings):
    engine = FasterWhisperSTT(stub_settings)
    engine.warmup()

    (model,) = stub.created
    assert (model.device, model.compute_type) == ("cpu", "int8")
    assert model.path == str(stub_settings.models_dir / "whisper-lb")
    assert model.calls == []
    assert engine.ready() is True
    assert engine.device == "cpu"
    assert "cpu" in engine.detail and "int8" in engine.detail and model.path in engine.detail


def test_auto_uses_cuda_float16_after_a_trial_transcribe_without_vad(stub, stub_settings):
    engine = FasterWhisperSTT(dataclasses.replace(stub_settings, stt_device="auto"))
    engine.warmup()

    (model,) = stub.created
    assert (model.device, model.compute_type) == ("cuda", "float16")
    assert len(model.audio[0]) == 16000 and not model.audio[0].any()
    assert model.calls[0]["vad_filter"] is False
    assert engine.device == "cuda"
    assert "cuda" in engine.detail and "float16" in engine.detail


def test_auto_falls_back_to_cpu_int8_when_cuda_fails(stub, stub_settings):
    stub.cuda_works = False
    engine = FasterWhisperSTT(dataclasses.replace(stub_settings, stt_device="auto"))
    engine.warmup()

    assert [(model.device, model.compute_type) for model in stub.created] == [
        ("cuda", "float16"),
        ("cpu", "int8"),
    ]
    assert engine.ready() is True
    assert engine.device == "cpu"
    assert "device=cpu" in engine.detail and "cublas64_12.dll" in engine.detail


def test_cuda_does_not_fall_back(stub, stub_settings):
    stub.cuda_works = False
    engine = FasterWhisperSTT(dataclasses.replace(stub_settings, stt_device="cuda"))

    with pytest.raises(RuntimeError, match="cublas64_12.dll"):
        engine.warmup()
    assert engine.ready() is False
    assert [model.device for model in stub.created] == ["cuda"]


def test_unknown_device_is_rejected(stub, stub_settings):
    engine = FasterWhisperSTT(dataclasses.replace(stub_settings, stt_device="gpu"))

    with pytest.raises(ValueError, match="stt_device"):
        engine.warmup()


def test_transcribe_decodes_the_wav_and_calls_the_model_as_specified(stub, stub_settings, temp_dir):
    engine = FasterWhisperSTT(stub_settings)
    engine.warmup()

    transcript = engine.transcribe(make_silence_wav(1.0), "audio/wav")

    (model,) = stub.created
    assert model.calls == [TRANSCRIBE_KWARGS]
    assert model.audio[0].dtype == "float32" and len(model.audio[0]) == 16000
    assert transcript.text == "Moien, wéi geet et?"
    assert transcript.language == "lb"
    assert transcript.duration_s == 1.0
    assert transcript.segments == [
        {"start": 0.0, "end": 1.5, "text": "Moien,"},
        {"start": 2.0, "end": 3.0, "text": "wéi geet et?"},
    ]
    assert list(temp_dir.iterdir()) == []


def test_transcribe_applies_the_empty_transcript_rules(stub, stub_settings):
    engine = FasterWhisperSTT(stub_settings)
    engine.warmup()

    for segments in ([], [(0.0, 0.2, " Jo.")], [(0.0, 2.0, " Jo, jo, jo.")]):
        stub.segments = segments
        transcript = engine.transcribe(make_silence_wav(2.0), "audio/wav")
        assert (transcript.text, transcript.segments, transcript.duration_s) == ("", [], 2.0)


def test_temp_file_gets_the_mime_suffix_and_is_deleted(stub, stub_settings, temp_dir, monkeypatch):
    seen = []

    def fake_decode(path):
        seen.append((path, list(temp_dir.iterdir())))
        return [0.0] * 16000

    monkeypatch.setattr(stt, "decode_audio", fake_decode)
    engine = FasterWhisperSTT(stub_settings)
    engine.warmup()

    engine.transcribe(b"\x1aE\xdf\xa3 webm bytes", "audio/webm;codecs=opus")

    ((path, files_during_decode),) = seen
    assert path.endswith(".webm")
    assert [str(file) for file in files_during_decode] == [path]
    assert list(temp_dir.iterdir()) == []


def test_temp_file_is_deleted_when_the_audio_cannot_be_decoded(stub, stub_settings, temp_dir):
    import av

    engine = FasterWhisperSTT(stub_settings)
    engine.warmup()

    with pytest.raises(av.error.InvalidDataError):
        engine.transcribe(b"this is not audio", "audio/webm")
    assert list(temp_dir.iterdir()) == []


def test_nvidia_wheel_folders_are_put_on_path_once(tmp_path, monkeypatch):
    bin_dir = tmp_path / "nvidia" / "cublas" / "bin"
    bin_dir.mkdir(parents=True)
    spec = SimpleNamespace(submodule_search_locations=[str(tmp_path / "nvidia")])
    monkeypatch.setattr(stt.importlib.util, "find_spec", lambda name: spec)
    monkeypatch.setattr(stt.sys, "platform", "win32")
    monkeypatch.setenv("PATH", "C:\\somewhere")

    stt.add_nvidia_dll_dirs()
    stt.add_nvidia_dll_dirs()

    assert stt.os.environ["PATH"] == f"{bin_dir}{stt.os.pathsep}C:\\somewhere"


def test_the_model_is_loaded_and_run_on_one_thread_whoever_calls(stub, stub_settings):
    engine = FasterWhisperSTT(stub_settings)
    engine.warmup()
    wav = make_silence_wav(0.2)
    texts = []

    def work():
        texts.append(engine.transcribe(wav, "audio/wav").text)

    threads = [threading.Thread(target=work) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    work()

    (model,) = stub.created
    assert texts == ["Moien, wéi geet et?"] * 7
    assert model.most_active == 1
    assert len(model.threads) == 1
    assert threading.get_ident() not in model.threads


# --- the real model (pytest -m models -s) --------------------------------------------------


def timed(engine, audio_bytes, mime, label):
    start = time.perf_counter()
    transcript = engine.transcribe(audio_bytes, mime)
    seconds = time.perf_counter() - start
    print(f"\n[stt {engine.device}] {label}: {seconds:.2f} s -> {transcript.text!r}")
    return transcript


def load(settings):
    engine = FasterWhisperSTT(settings)
    start = time.perf_counter()
    engine.warmup()
    print(f"\n[stt] loaded in {time.perf_counter() - start:.2f} s: {engine.detail}")
    return engine


@pytest.fixture(scope="module")
def real_settings():
    settings = Settings.from_env()
    model_dir = settings.models_dir / "whisper-lb"
    if not (model_dir / "model.bin").is_file():
        pytest.skip(f"no model in {model_dir}; run: python scripts/download_models.py --stt")
    return settings


@pytest.fixture(scope="module")
def engine(real_settings):
    return load(real_settings)


@pytest.fixture(scope="module")
def lod_audio(real_settings):
    folder = real_settings.data_dir / "lod_audio"
    wanted = [folder / name for name in LOD_RECORDINGS] + [folder / "texts.json"]
    if not all(path.is_file() for path in wanted):
        pytest.skip(
            f"no LOD recordings in {folder}; run: python scripts/download_models.py --lod-audio"
        )
    texts = json.loads((folder / "texts.json").read_text(encoding="utf-8"))
    return {name: ((folder / name).read_bytes(), texts[name]) for name in LOD_RECORDINGS}


def to_webm_opus(audio_bytes: bytes) -> bytes:
    """Re-encode a recording the way the browser's MediaRecorder sends it."""
    import av

    buffer = io.BytesIO()
    with av.open(io.BytesIO(audio_bytes)) as source, av.open(buffer, "w", format="webm") as target:
        stream = target.add_stream("libopus", rate=48000)
        resampler = av.AudioResampler(
            format=stream.format, layout="mono", rate=48000, frame_size=960
        )
        for frame in source.decode(audio=0):
            for resampled in resampler.resample(frame):
                target.mux(stream.encode(resampled))
        for resampled in resampler.resample(None):
            target.mux(stream.encode(resampled))
        target.mux(stream.encode(None))
    return buffer.getvalue()


@pytest.mark.models
def test_five_seconds_of_silence_give_an_empty_transcript(engine):
    transcript = timed(engine, make_silence_wav(5), "audio/wav", "5 s silence")

    assert transcript.text == ""
    assert transcript.segments == []
    assert transcript.duration_s == pytest.approx(5.0)


@pytest.mark.models
def test_five_seconds_of_low_white_noise_give_an_empty_transcript(engine):
    transcript = timed(engine, make_noise_wav(5, amplitude=0.01), "audio/wav", "5 s noise 0.01")

    assert transcript.text == ""
    assert transcript.segments == []


@pytest.mark.models
@pytest.mark.parametrize("name", LOD_RECORDINGS)
def test_lod_recording_is_transcribed(engine, lod_audio, name):
    audio_bytes, text = lod_audio[name]

    transcript = timed(engine, audio_bytes, "audio/mp4", name)
    error_rate = cer(text, transcript.text)
    print(f"[stt {engine.device}] {name}: CER {error_rate:.3f} against {text!r}")

    assert error_rate <= MAX_CER
    assert transcript.language == "lb"
    assert transcript.duration_s > 1
    assert transcript.segments and set(transcript.segments[0]) == {"start", "end", "text"}


@pytest.mark.models
def test_browser_webm_opus_is_decoded_and_transcribed(engine, lod_audio):
    audio_bytes, text = lod_audio["benotzen.m4a"]
    webm = to_webm_opus(audio_bytes)
    assert webm[:4] == b"\x1aE\xdf\xa3"

    transcript = timed(engine, webm, "audio/webm;codecs=opus", "benotzen as webm/opus")
    error_rate = cer(text, transcript.text)
    print(f"[stt {engine.device}] benotzen as webm/opus: CER {error_rate:.3f}")

    assert error_rate <= MAX_CER


@pytest.mark.models
def test_cpu_device_loads_and_transcribes(real_settings, lod_audio):
    cpu_engine = load(dataclasses.replace(real_settings, stt_device="cpu"))
    audio_bytes, text = lod_audio["benotzen.m4a"]

    assert cpu_engine.ready() is True
    assert cpu_engine.device == "cpu"
    assert "int8" in cpu_engine.detail
    transcript = timed(cpu_engine, audio_bytes, "audio/mp4", "benotzen.m4a on cpu")
    assert cer(text, transcript.text) <= MAX_CER
