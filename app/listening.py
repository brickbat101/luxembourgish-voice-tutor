"""The listening part: an item read aloud at one of two speeds, its speakers told apart by pitch,
and the learner's choices checked against its questions."""

from __future__ import annotations

import io
import wave
from collections.abc import Mapping

from app.content.listening import ListeningItem, ListeningQuestion
from app.engines.base import Segment, WavAudio, mark_synthetic

SPEEDS = {"normal": 1.0, "slow": 0.6}  # the lb rate; 0.6 measured as 0.75x the speed of 1.0
SPEED_LABELS = {"normal": "1x", "slow": "0.75x"}
TURN_GAP_S = 0.6  # silence between two turns


def turn_voice(item: ListeningItem, speaker: str, speed: str, voice: Mapping | None = None) -> dict:
    """The voice override of one turn: the learner's lb volume stays, while the rate is the speed's
    and the pitch the speaker's, so an item sounds the same for everyone."""
    own = dict((voice or {}).get("lb", {}))
    return {"lb": {**own, "rate": SPEEDS[speed], "pitch": item.speaker(speaker).pitch}}


def render_item(tts, item: ListeningItem, speed: str, voice: Mapping | None = None) -> WavAudio:
    """The whole item as one WAV: one synthesize call per turn, the turns joined with TURN_GAP_S of
    silence. `segment_offsets` has one entry per turn, with its `speaker`."""
    frames: list[bytes] = []
    offsets: list[dict] = []
    params = None
    length = 0  # in frames
    for index, turn in enumerate(item.turns):
        audio = tts.synthesize([Segment("lb", turn.lb)], turn_voice(item, turn.speaker, speed, voice))
        with wave.open(io.BytesIO(audio.wav_bytes)) as wav:
            shape = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate())
            pcm = wav.readframes(wav.getnframes())
        if params is None:
            params = shape
        elif shape != params:
            raise ValueError(f"turn {index + 1} is {shape} (channels, sample width, rate), the first turn {params}")
        channels, width, rate = params
        if index:
            gap = round(TURN_GAP_S * rate)
            frames.append(b"\0" * (gap * channels * width))
            length += gap
        start = length
        length += len(pcm) // (channels * width)
        frames.append(pcm)
        offsets.append(
            {"index": index, "lang": "lb", "start_s": start / rate, "end_s": length / rate, "speaker": turn.speaker}
        )
    if params is None:
        raise ValueError(f"item {item.id!r} has no turns")
    channels, width, rate = params
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        wav.writeframes(b"".join(frames))
    return WavAudio(mark_synthetic(buffer.getvalue()), rate, offsets)


def chosen_option(question: ListeningQuestion, answers: Mapping) -> int | None:
    """The index the learner chose for the question, None when it is missing or no index of its options."""
    choice = answers.get(question.id)
    if isinstance(choice, bool) or not isinstance(choice, int) or not 0 <= choice < len(question.options):
        return None
    return choice


def check_answers(item: ListeningItem, answers: Mapping) -> dict:
    """Which of the learner's choices are right, how many of how many, and the transcript.

    `answers` maps a question id to the index of the chosen option; a missing or out-of-range
    choice is not right. No other figure is computed.
    """
    questions = []
    for question in item.questions:
        chosen = chosen_option(question, answers)
        questions.append(
            {
                "id": question.id,
                "chosen": chosen,
                "answer": question.answer,
                "right": chosen == question.answer,
                "why_en": question.why_en,
            }
        )
    return {
        "questions": questions,
        "right": sum(entry["right"] for entry in questions),
        "total": len(questions),
        "transcript": [{"speaker": turn.speaker, "lb": turn.lb} for turn in item.turns],
    }
