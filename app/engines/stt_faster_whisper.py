"""Speech to text: faster-whisper with the Luxembourgish Whisper model in <models_dir>/whisper-lb.

faster_whisper, ctranslate2, av and numpy are imported on first use, not with this module.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import string
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.config import Settings
from app.engines.base import Transcript

log = logging.getLogger(__name__)

LANGUAGE = "lb"
SAMPLE_RATE = 16000
MIN_SPEECH_S = 0.3
COMPUTE_TYPES = {"cuda": "float16", "cpu": "int8"}
SUFFIXES = {
    "webm": ".webm",
    "ogg": ".ogg",
    "mp4": ".m4a",
    "m4a": ".m4a",
    "wav": ".wav",
    "wave": ".wav",
}


def suffix_for(mime: str) -> str:
    """File suffix for a MIME type such as "audio/webm;codecs=opus"."""
    subtype = mime.split(";")[0].strip().lower().rpartition("/")[2]
    return SUFFIXES.get(subtype.removeprefix("x-"), ".audio")


def is_repeated_token(text: str) -> bool:
    """True for one word said three or more times and nothing else, a common Whisper artefact."""
    tokens = [token.strip(string.punctuation).lower() for token in text.split()]
    tokens = [token for token in tokens if token]
    return len(tokens) >= 3 and len(set(tokens)) == 1


def build_transcript(segments: list[dict], duration_s: float) -> Transcript:
    """Join the segment texts; too little speech or a repeated token gives an empty transcript."""
    text = " ".join(segment["text"] for segment in segments if segment["text"])
    speech_s = sum(segment["end"] - segment["start"] for segment in segments)
    if not text or speech_s < MIN_SPEECH_S or is_repeated_token(text):
        return Transcript("", LANGUAGE, duration_s, [])
    return Transcript(text, LANGUAGE, duration_s, segments)


def decode_audio(path: str):
    """Decode an audio file with PyAV to 16 kHz mono float32 samples.

    faster-whisper 1.2.1 cannot do this itself here: its decoder passes `metadata_errors`
    to av.open, which av 19 rejects.
    """
    import av
    import numpy as np

    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    chunks = []
    with av.open(path) as container:
        for frame in container.decode(audio=0):
            chunks.extend(out.to_ndarray() for out in resampler.resample(frame))
        chunks.extend(out.to_ndarray() for out in resampler.resample(None))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks, axis=1).reshape(-1).astype(np.float32) / 32768.0


def add_nvidia_dll_dirs() -> None:
    """Windows: put the cuBLAS/cuDNN folders of the nvidia-*-cu12 wheels on PATH.

    ctranslate2 loads those DLLs by name on first GPU use and finds them only through
    PATH; os.add_dll_directory has no effect on that lookup.
    """
    if sys.platform != "win32":
        return
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return
    path = os.environ.get("PATH", "")
    present = path.split(os.pathsep)
    bin_dirs = [
        str(bin_dir)
        for root in spec.submodule_search_locations
        for bin_dir in sorted(Path(root).glob("*/bin"))
    ]
    missing = [bin_dir for bin_dir in bin_dirs if bin_dir not in present]
    if missing:
        os.environ["PATH"] = os.pathsep.join([*missing, path])


class FasterWhisperSTT:
    name = "faster_whisper"

    def __init__(self, settings: Settings) -> None:
        self.model_path = settings.models_dir / "whisper-lb"
        self.requested_device = settings.stt_device
        self.device = ""
        self.detail = "not loaded"
        self._model = None
        # The model is loaded and run on this one thread only. That serialises requests, and
        # it avoids a crash at process exit seen with ctranslate2 on CUDA when a model first
        # run on one thread was later run on the main thread.
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")

    def ready(self) -> bool:
        return self._model is not None

    def warmup(self) -> None:
        self._worker.submit(self._warmup).result()

    def transcribe(self, audio_bytes: bytes, mime: str) -> Transcript:
        if self._model is None:
            raise RuntimeError("faster_whisper engine is not loaded; call warmup() first")
        handle = tempfile.NamedTemporaryFile(suffix=suffix_for(mime), delete=False)
        try:
            with handle:
                handle.write(audio_bytes)
            audio = decode_audio(handle.name)
        finally:
            os.unlink(handle.name)
        segments, duration_s = self._worker.submit(self._run, audio).result()
        return build_transcript(segments, duration_s)

    def _warmup(self) -> None:
        if self._model is not None:
            return
        if self.requested_device not in ("auto", "cuda", "cpu"):
            raise ValueError(f"stt_device must be auto, cuda or cpu, not {self.requested_device!r}")
        if not (self.model_path / "model.bin").is_file():
            raise FileNotFoundError(
                f"no model in {self.model_path}; run: python scripts/download_models.py --stt"
            )
        cuda_error = ""
        if self.requested_device in ("auto", "cuda"):
            try:
                self._load("cuda")
            except Exception as exc:
                if self.requested_device == "cuda":
                    raise
                log.warning("faster-whisper on cuda failed, falling back to cpu", exc_info=True)
                cuda_error = f"; cuda failed: {type(exc).__name__}: {exc}"
        if self._model is None:
            self._load("cpu")
        self.detail += cuda_error
        log.info("faster-whisper ready: %s", self.detail)

    def _load(self, device: str) -> None:
        if device == "cuda":
            add_nvidia_dll_dirs()
        from faster_whisper import WhisperModel

        compute_type = COMPUTE_TYPES[device]
        model = WhisperModel(str(self.model_path), device=device, compute_type=compute_type)
        if device == "cuda":
            # The model loads without cuBLAS and cuDNN; only running it shows they work.
            # VAD stays off here: it would drop the silence before the model sees it.
            import numpy as np

            segments, _ = model.transcribe(
                np.zeros(SAMPLE_RATE, dtype=np.float32),
                language=LANGUAGE,
                beam_size=1,
                temperature=0.0,
                max_new_tokens=8,
                vad_filter=False,
            )
            list(segments)
        self._model = model
        self.device = device
        self.detail = f"device={device}, compute_type={compute_type}, model={self.model_path}"

    def _run(self, audio) -> tuple[list[dict], float]:
        found, info = self._model.transcribe(
            audio,
            language=LANGUAGE,
            task="transcribe",
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )
        # `found` is a generator; the model runs while it is consumed.
        segments = [
            {"start": segment.start, "end": segment.end, "text": segment.text.strip()}
            for segment in found
        ]
        return segments, info.duration
