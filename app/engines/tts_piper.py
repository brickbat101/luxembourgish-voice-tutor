"""Text to speech: two Piper voices in <models_dir>/piper, Luxembourgish and English, on CPU.

Before a segment is spoken, the words in the pronunciation table of its language are replaced:
the table shipped next to this module with <data_dir>/pronunciations.json laid over it.

piper and onnxruntime are imported on first use, not with this module.
"""

from __future__ import annotations

import io
import json
import logging
import re
import threading
import wave
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from app.config import Settings
from app.engines.base import VOICE_LANGS, Segment, WavAudio, ends_sentence, mark_synthetic, voice_tuning

log = logging.getLogger(__name__)

SAMPLE_RATE = 22050
GAP_S = 0.25  # of silence after a segment that ends a sentence
SHORT_GAP_S = 0.08  # after a segment that stops inside a sentence
MARKDOWN = re.compile(r"[*_`#]")
# Opening quote -> the quotes that may close it.
QUOTES = {'"': '"', "'": "'", "“": "”", "„": "“”", "«": "»", "»": "«", "‘": "’", "‚": "‘’"}
# Piper speaks what stands in [[ ]] as raw phonemes.
DOUBLED_BRACKET = re.compile(r"([\[\]])\1+")
PHONEME_BLOCK = re.compile(r"(\[\[.*?\]\])")
# Piper drops or reads out ("colon") punctuation that directly follows a phoneme block.
AFTER_BLOCK = re.compile(r"\s*\]\]([,.;:!?]+)")
LETTER = r"[^\W\d_]"
DEFAULT_PRONUNCIATIONS = Path(__file__).with_name("pronunciations.json")


def clean_text(text: str) -> str:
    """Drop markdown markers and quotes around the whole phrase, collapse whitespace.

    Doubled square brackets become single ones: only the pronunciation table writes phonemes.
    """
    text = " ".join(MARKDOWN.sub("", text).split())
    text = DOUBLED_BRACKET.sub(r"\1", text)
    if len(text) >= 2 and text[-1] in QUOTES.get(text[0], ""):
        text = text[1:-1].strip()
    return text


def word_key(word: str) -> str:
    """A table key as it is matched: single spaces, lower case."""
    return " ".join(word.split()).lower()


def read_pronunciations(path: Path) -> dict[str, dict[str, str]]:
    """{"lb": {word: replacement}, "en": {...}} from a JSON file; ValueError for another shape."""
    # utf-8-sig: Windows editors may put a byte order mark first.
    table = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(table, dict):
        raise ValueError("the file is not an object of languages")
    for lang, words in table.items():
        if lang not in VOICE_LANGS:
            raise ValueError(f"unknown language {lang!r}; known: {', '.join(VOICE_LANGS)}")
        if not isinstance(words, dict):
            raise ValueError(f"{lang!r} is not an object of words")
        for word, replacement in words.items():
            if not word_key(word):
                raise ValueError(f"{lang!r} has an empty word")
            if not isinstance(replacement, str):
                raise ValueError(f"{lang}.{word} is not a string: {replacement!r}")
    return table


def pronounce(text: str, words: dict[str, str]) -> str:
    """`text` with each whole word that is a key of `words` (lower case) replaced by its value.

    Any case matches and the longest key wins. A word ends where a character that is not a
    letter follows, an apostrophe included. Text inside [[ ]] stays as it is, and punctuation
    right after a [[ ]] block moves into it.
    """
    if not words:
        return text
    longest_first = sorted(words, key=len, reverse=True)
    pattern = re.compile(
        f"(?<!{LETTER})(?:{'|'.join(map(re.escape, longest_first))})(?!{LETTER})", re.IGNORECASE
    )
    parts = PHONEME_BLOCK.split(text)
    # The even parts are outside [[ ]].
    parts[::2] = [
        pattern.sub(lambda match: words.get(match.group().lower(), match.group()), part)
        for part in parts[::2]
    ]
    return AFTER_BLOCK.sub(r"\1 ]]", "".join(parts))


class Pronunciations:
    """The shipped table with the user's file laid over it; that file is re-read when it changes."""

    def __init__(self, user_path: Path) -> None:
        self.user_path = user_path
        self._defaults = read_pronunciations(DEFAULT_PRONUNCIATIONS)
        # (stamp of the user file, words per language, what is wrong with the file). It is
        # replaced as one value, so a reader in another thread never sees half of a new table.
        self._state = (object(), {}, "")

    def _current(self) -> tuple:
        try:
            stat = self.user_path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            stamp = None
        if stamp != self._state[0]:
            user, problem = {}, ""
            if stamp is not None:
                try:
                    user = read_pronunciations(self.user_path)
                except (OSError, ValueError) as exc:
                    problem = (
                        f"pronunciations: {self.user_path} is ignored, the shipped table is used "
                        f"({type(exc).__name__}: {exc})"
                    )
                    log.warning(problem)
            words = {}
            for lang in VOICE_LANGS:
                merged = {
                    word_key(word): replacement
                    for table in (self._defaults, user)
                    for word, replacement in table.get(lang, {}).items()
                }
                # An empty replacement takes the word out of the table.
                words[lang] = {word: value for word, value in merged.items() if value}
            self._state = (stamp, words, problem)
        return self._state

    def problem(self) -> str:
        """What is wrong with the user's file; "" when it is fine or absent."""
        return self._current()[2]

    def lookup(self, lang: str, word: str) -> str:
        """The respelling the table has for `word` in `lang` now; "" when it has none."""
        return self._current()[1].get(lang, {}).get(word_key(word), "")

    def apply(self, lang: str, text: str, extra: Mapping[str, str] | None = None) -> str:
        """`text` with the table of `lang`; `extra` ({word: respelling}) is laid over it for this call only."""
        words = self._current()[1][lang]
        if extra:
            merged = {**words, **{word_key(word): value for word, value in extra.items()}}
            words = {word: value for word, value in merged.items() if value}
        return pronounce(text, words)


def resample(samples: np.ndarray, rate: float) -> np.ndarray:
    """int16 samples at `rate` Hz, linearly interpolated to SAMPLE_RATE."""
    if rate == SAMPLE_RATE or not len(samples):
        return samples
    positions = np.arange(round(len(samples) * SAMPLE_RATE / rate)) * (rate / SAMPLE_RATE)
    return np.round(np.interp(positions, np.arange(len(samples)), samples)).astype(np.int16)


def amplify(samples: np.ndarray, volume: float) -> np.ndarray:
    """int16 samples times `volume`; what leaves the 16-bit range is clipped, not wrapped."""
    if volume == 1.0:
        return samples
    return np.clip(np.round(samples * volume), -32767, 32767).astype(np.int16)


class PiperTTS:
    name = "piper"

    def __init__(self, settings: Settings) -> None:
        folder = settings.models_dir / "piper"
        # Segments in any language other than "lb" are spoken by the "en" voice.
        self.model_paths = {
            "lb": folder / f"{settings.tts_voice_lb}.onnx",
            "en": folder / f"{settings.tts_voice_en}.onnx",
        }
        self._detail = "not loaded"
        self._settings = settings
        self._voices: dict = {}
        self._lock = threading.Lock()
        self._pronunciations = Pronunciations(settings.data_dir / "pronunciations.json")

    @property
    def detail(self) -> str:
        """The loaded voices, then what is wrong with the user's pronunciation file, if anything."""
        problem = self._pronunciations.problem()
        return f"{self._detail}; {problem}" if problem else self._detail

    def ready(self) -> bool:
        return bool(self._voices)

    def warmup(self) -> None:
        with self._lock:
            if self._voices:
                return
            for model_path in self.model_paths.values():
                for path in (model_path, model_path.with_name(model_path.name + ".json")):
                    if not path.is_file():
                        raise FileNotFoundError(
                            f"no voice file {path}; run: python scripts/download_models.py --tts"
                        )
            from piper import PiperVoice

            voices = {
                lang: PiperVoice.load(model_path) for lang, model_path in self.model_paths.items()
            }
            self._detail = "device=cpu, " + ", ".join(
                f"{lang}={self.model_paths[lang].stem} ({voice.config.sample_rate} Hz)"
                for lang, voice in voices.items()
            )
            self._voices = voices
        log.info("piper ready: %s", self.detail)

    def synthesize(
        self, segments: list[Segment], voice=None, respell: Mapping[str, str] | None = None
    ) -> WavAudio:
        """`voice` overrides the rate, pitch and volume settings: see `voice_tuning`.

        `respell` ({word: respelling}) is laid over the pronunciation table for this call only.
        """
        if not self._voices:
            raise RuntimeError("piper engine is not loaded; call warmup() first")
        tuning = voice_tuning(self._settings, voice)
        # Silence after a segment, by whether that segment ended a sentence.
        gaps = {
            closed: np.zeros(round(seconds * SAMPLE_RATE), dtype=np.int16)
            for closed, seconds in ((True, GAP_S), (False, SHORT_GAP_S))
        }
        parts: list[np.ndarray] = []
        offsets = []
        length = 0
        closed = True  # of the last segment that was spoken
        with self._lock:
            for index, segment in enumerate(segments):
                samples = self._speak(segment, tuning, respell)
                # The gap goes between two stretches of speech, never around an empty segment.
                if length and len(samples):
                    parts.append(gaps[closed])
                    length += len(gaps[closed])
                if len(samples):
                    closed = ends_sentence(clean_text(segment.text))
                parts.append(samples)
                offsets.append(
                    {
                        "index": index,
                        "lang": segment.lang,
                        "start_s": length / SAMPLE_RATE,
                        "end_s": (length + len(samples)) / SAMPLE_RATE,
                    }
                )
                length += len(samples)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(b"".join(part.astype("<i2").tobytes() for part in parts))
        # AI Act Art. 50(2): the audio carries a machine-readable mark that it is synthetic.
        return WavAudio(mark_synthetic(buffer.getvalue()), SAMPLE_RATE, offsets)

    def phonemes(self, lang: str, text: str) -> str:
        """The phonemes the voice of `lang` is given for `text`, sentences apart by a space.

        The text goes through the same steps as in `synthesize`. "" when the voices are not loaded.
        """
        with self._lock:
            if not self._voices:
                return ""
            lang = "lb" if lang == "lb" else "en"
            text = self._pronunciations.apply(lang, clean_text(text))
            sentences = self._voices[lang].phonemize(text)
        return " ".join("".join(sentence) for sentence in sentences)

    def _speak(self, segment: Segment, tuning: dict, respell: Mapping[str, str] | None = None) -> np.ndarray:
        text = clean_text(segment.text)
        if not text:
            return np.zeros(0, dtype=np.int16)
        from piper import SynthesisConfig

        lang = "lb" if segment.lang == "lb" else "en"
        text = self._pronunciations.apply(lang, text, respell)
        voice = self._voices[lang]
        rate, pitch, volume = (tuning[lang][key] for key in ("rate", "pitch", "volume"))
        # Piper has no pitch control. Reading the samples as if recorded at `shift` times their
        # rate moves the pitch by `shift` and divides the duration by it, so the voice is asked
        # for a duration `shift` times longer: the result lasts what the rate alone gives.
        # The real voices follow the length scale only in part, so durations change by less
        # than it asks and a lower pitch leaves the speech a little slower.
        shift = 2 ** (pitch / 12)
        config = SynthesisConfig(length_scale=voice.config.length_scale * shift / rate)
        # Piper yields one chunk of 16-bit mono audio per sentence.
        pcm = b"".join(chunk.audio_int16_bytes for chunk in voice.synthesize(text, config))
        samples = resample(np.frombuffer(pcm, dtype=np.int16), voice.config.sample_rate * shift)
        return amplify(samples, volume)
