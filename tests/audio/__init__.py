"""Test helpers: generated WAV clips and the character error rate. Standard library only."""

from __future__ import annotations

import io
import random
import struct
import unicodedata
import wave

SAMPLE_RATE = 16000


def _wav(samples: list[int]) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buffer.getvalue()


def make_silence_wav(seconds: float) -> bytes:
    """Digital silence as a 16 kHz mono 16-bit WAV."""
    return _wav([0] * round(seconds * SAMPLE_RATE))


def make_noise_wav(seconds: float, amplitude: float = 0.01) -> bytes:
    """White noise as a 16 kHz mono 16-bit WAV; `amplitude` is the peak, 1.0 being full scale."""
    rng = random.Random(0)
    peak = amplitude * 32767
    return _wav([round(rng.uniform(-peak, peak)) for _ in range(round(seconds * SAMPLE_RATE))])


def _normalise(text: str) -> str:
    kept = "".join(char for char in text.lower() if not unicodedata.category(char).startswith("P"))
    return " ".join(kept.split())


def cer(ref: str, hyp: str) -> float:
    """Character error rate: Levenshtein distance / len(ref), lower-cased, punctuation stripped."""
    ref, hyp = _normalise(ref), _normalise(hyp)
    row = list(range(len(hyp) + 1))
    for i, ref_char in enumerate(ref, start=1):
        previous, row[0] = row[0], i
        for j, hyp_char in enumerate(hyp, start=1):
            previous, row[j] = row[j], min(
                row[j] + 1, row[j - 1] + 1, previous + (ref_char != hyp_char)
            )
    return row[-1] / len(ref)
