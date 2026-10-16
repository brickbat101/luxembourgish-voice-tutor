"""In-memory engines for tests and for running the app without models. Standard library only."""

from __future__ import annotations

import io
import math
import struct
import wave

from app.engines.base import (
    TEMPLATE_WRAP_UP,
    Judgement,
    Segment,
    Transcript,
    TutorReply,
    Turn,
    WavAudio,
    WrapUpText,
    mark_synthetic,
)

SAMPLE_RATE = 22050


class FakeSTT:
    name = "fake"
    TEXT = "Moien, wéi geet et?"

    def __init__(self, settings=None) -> None:
        pass

    def ready(self) -> bool:
        return True

    def warmup(self) -> None:
        pass

    def transcribe(self, audio_bytes: bytes, mime: str = "audio/wav") -> Transcript:
        try:
            with wave.open(io.BytesIO(audio_bytes)) as wav:
                frames = wav.readframes(wav.getnframes())
                duration_s = wav.getnframes() / wav.getframerate()
        except (wave.Error, EOFError):
            frames, duration_s = audio_bytes, 0.0
        text = self.TEXT if any(frames) else ""
        return Transcript(text=text, language="lb", duration_s=duration_s)


class FakeTTS:
    name = "fake"
    TONE_S = 0.4
    GAP_S = 0.25
    TONE_HZ = 440.0

    def __init__(self, settings=None) -> None:
        self.voice = None  # the `voice` of the last synthesize call
        self.respell = None  # and its `respell`

    def ready(self) -> bool:
        return True

    def warmup(self) -> None:
        pass

    def synthesize(self, segments: list[Segment], voice=None, respell=None) -> WavAudio:
        self.voice = voice
        self.respell = respell
        tone_n = round(self.TONE_S * SAMPLE_RATE)
        gap_n = round(self.GAP_S * SAMPLE_RATE)
        step = 2 * math.pi * self.TONE_HZ / SAMPLE_RATE
        tone = [round(12000 * math.sin(step * i)) for i in range(tone_n)]

        samples: list[int] = []
        offsets = []
        for index, segment in enumerate(segments):
            if index:
                samples.extend([0] * gap_n)
            start = len(samples)
            samples.extend(tone)
            offsets.append(
                {
                    "index": index,
                    "lang": segment.lang,
                    "start_s": start / SAMPLE_RATE,
                    "end_s": len(samples) / SAMPLE_RATE,
                }
            )

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        # AI Act Art. 50(2): the audio carries a machine-readable mark that it is synthetic.
        return WavAudio(mark_synthetic(buffer.getvalue()), SAMPLE_RATE, offsets)


class FakeLLM:
    name = "fake"
    REACTION = "Nice, that came across."  # of a judgement when none is queued
    # Of a look back on a talk when none is queued.
    WRAP_UP = TEMPLATE_WRAP_UP

    def __init__(self, settings=None) -> None:
        # Every call in order; a `judge` call is the one with "method": "judge", a `wrap_up` call
        # the one with "method": "wrap_up".
        self.calls: list[dict] = []
        # Outcomes of the next calls, first in first out: a reply to return or an exception to raise.
        self.queue: list[TutorReply | Exception] = []
        self.judge_queue: list[Judgement | Exception] = []
        self.wrap_up_queue: list[WrapUpText | Exception] = []

    def ready(self) -> bool:
        return True

    def warmup(self) -> None:
        pass

    def reply(self, history: list[Turn], mode: str, grounding: str) -> TutorReply:
        self.calls.append({"history": list(history), "mode": mode, "grounding": grounding})
        if self.queue:
            outcome = self.queue.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return TutorReply(
            segments=[Segment("lb", "Moien!"), Segment("en", "Hello! How was your day?")],
            new_phrases=[{"lb": "Moien!", "en": "Hello!"}],
            mode=mode,
            claims=[],
        )

    def judge(self, question_lb: str, question_en: str, answer: str, grounding: str, extra: str = "") -> Judgement:
        self.calls.append(
            {
                "method": "judge",
                "question_lb": question_lb,
                "question_en": question_en,
                "answer": answer,
                "grounding": grounding,
                "extra": extra,
            }
        )
        if self.judge_queue:
            outcome = self.judge_queue.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return Judgement("good", self.REACTION)

    def wrap_up(self, facts: str) -> WrapUpText:
        self.calls.append({"method": "wrap_up", "facts": facts})
        if self.wrap_up_queue:
            outcome = self.wrap_up_queue.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return self.WRAP_UP


def _score_key(text: str) -> str:
    return " ".join(text.split()).casefold()


class NoAcceptability:
    """No acceptability encoder: no text gets a score."""

    name = "none"
    detail = "no encoder installed"

    def __init__(self, settings=None) -> None:
        pass

    def ready(self) -> bool:
        return True

    def warmup(self) -> None:
        pass

    def score(self, text: str) -> float | None:
        return None


class FakeAcceptability:
    name = "fake"

    def __init__(self, settings=None, scores: dict[str, float] | None = None) -> None:
        # Keyed by the text with single spaces, case-folded; a text without a score gets None.
        self.scores = {_score_key(text): score for text, score in (scores or {}).items()}
        self.calls: list[str] = []  # every text scored, in order

    def ready(self) -> bool:
        return True

    def warmup(self) -> None:
        pass

    def score(self, text: str) -> float | None:
        self.calls.append(text)
        return self.scores.get(_score_key(text))
