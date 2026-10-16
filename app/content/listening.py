"""The fixed content of the listening part: B1 recordings, each a script of turns read by the
Luxembourgish voice, with its multiple-choice questions, from listening.toml."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from app.content import PHOTO_ID, _tables, _text
from app.engines.base import VOICE_RANGES

LISTENING_PATH = Path(__file__).resolve().parent / "listening.toml"
GENRES = ("news", "conversation", "discussion")  # news item, everyday conversation, longer discussion
QUESTIONS_PER_ITEM = (4, 7)  # the fewest and the most questions of an item
OPTIONS_PER_QUESTION = (2, 3)  # shown as A/B or A/B/C
ITEMS_PER_GENRE = 2  # the fewest items of each genre
LEVEL = "B1"


@dataclass(frozen=True)
class Speaker:
    name: str  # shown before each of its turns
    pitch: float  # in semitones, within VOICE_RANGES["pitch"]


@dataclass(frozen=True)
class Turn:
    speaker: str  # the name of one of the item's speakers
    lb: str


@dataclass(frozen=True)
class ListeningQuestion:
    id: str
    lb: str
    en: str
    options: tuple[str, ...]  # in Luxembourgish
    answer: int  # the index of the right option
    why_en: str  # the words of the script that give the answer away


@dataclass(frozen=True)
class ListeningItem:
    id: str
    genre: str  # one of GENRES
    title_en: str
    level: str
    speakers: tuple[Speaker, ...]
    turns: tuple[Turn, ...]
    questions: tuple[ListeningQuestion, ...]  # in the order of the recording

    @property
    def words(self) -> int:
        return sum(len(turn.lb.split()) for turn in self.turns)

    @property
    def text(self) -> str:
        """The whole script, a paragraph per turn."""
        return "\n\n".join(turn.lb for turn in self.turns)

    def speaker(self, name: str) -> Speaker:
        return next(speaker for speaker in self.speakers if speaker.name == name)


def _line(table: dict, key: str, where: str) -> str:
    """The text `key` of `table` as one line: a text may run over several lines in the file."""
    return " ".join(_text(table, key, where).split())


def _speaker(raw: dict, item: str, number: int) -> Speaker:
    name = _line(raw, "name", f"{item}, speaker number {number}")
    where = f"{item}, speaker {name!r}"
    pitch = raw.get("pitch")
    if isinstance(pitch, bool) or not isinstance(pitch, (int, float)) or not math.isfinite(pitch):
        raise ValueError(f"{where}: `pitch` is not a number")
    low, high = VOICE_RANGES["pitch"]
    if not low <= pitch <= high:
        raise ValueError(f"{where}: the pitch {pitch} is outside {low} to {high}")
    return Speaker(name, float(pitch))


def _question(raw: dict, item: str, number: int) -> ListeningQuestion:
    question_id = _text(raw, "id", f"{item}, question number {number}")
    where = f"{item}, question {question_id!r}"
    if not PHOTO_ID.fullmatch(question_id):
        raise ValueError(f"{where}: the id may hold only a-z, 0-9 and single hyphens between them")
    options = raw.get("options")
    fewest, most = OPTIONS_PER_QUESTION
    if not isinstance(options, list) or not fewest <= len(options) <= most:
        raise ValueError(f"{where}: `options` is not a list of {fewest} to {most} texts")
    if not all(isinstance(option, str) and option.strip() for option in options):
        raise ValueError(f"{where}: an option is empty or not a text")
    answer = raw.get("answer")
    if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < len(options):
        raise ValueError(f"{where}: `answer` is not an index of `options` (0 to {len(options) - 1})")
    return ListeningQuestion(
        id=question_id,
        lb=_line(raw, "lb", where),
        en=_line(raw, "en", where),
        options=tuple(" ".join(option.split()) for option in options),
        answer=answer,
        why_en=_line(raw, "why_en", where),
    )


def _item(raw: dict, where: str) -> ListeningItem:
    item_id = _text(raw, "id", where)
    where = f"item {item_id!r}"
    if not PHOTO_ID.fullmatch(item_id):
        raise ValueError(f"{where}: the id may hold only a-z, 0-9 and single hyphens between them")
    genre = _text(raw, "genre", where)
    if genre not in GENRES:
        raise ValueError(f"{where}: unknown genre {genre!r}; known: {', '.join(GENRES)}")
    level = _text(raw, "level", where)
    if level != LEVEL:
        raise ValueError(f"{where}: the level is {level!r}, not {LEVEL!r}")
    speakers: list[Speaker] = []
    for number, entry in enumerate(_tables(raw, "speaker", where), start=1):
        speaker = _speaker(entry, where, number)
        if speaker.name in {known.name for known in speakers}:
            raise ValueError(f"{where}, speaker {speaker.name!r}: the name is used twice")
        speakers.append(speaker)
    names = {speaker.name for speaker in speakers}
    turns: list[Turn] = []
    for number, entry in enumerate(_tables(raw, "turn", where), start=1):
        turn_where = f"{where}, turn {number}"
        turn = Turn(_line(entry, "speaker", turn_where), _line(entry, "lb", turn_where))
        if turn.speaker not in names:
            raise ValueError(f"{turn_where}: the speaker {turn.speaker!r} is not one of the item's speakers")
        turns.append(turn)
    for speaker in speakers:
        if speaker.name not in {turn.speaker for turn in turns}:
            raise ValueError(f"{where}, speaker {speaker.name!r}: no turn is spoken by it")
    tables = _tables(raw, "question", where)
    fewest, most = QUESTIONS_PER_ITEM
    if not fewest <= len(tables) <= most:
        raise ValueError(f"{where}: {len(tables)} questions, an item has {fewest} to {most}")
    questions = tuple(_question(entry, where, number) for number, entry in enumerate(tables, start=1))
    return ListeningItem(item_id, genre, _line(raw, "title_en", where), level, tuple(speakers), tuple(turns), questions)


def load_listening(path: Path = LISTENING_PATH) -> tuple[ListeningItem, ...]:
    """The items of the file at `path`; a ValueError says which entry of it is wrong."""
    try:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"not valid TOML: {exc}") from exc
    items: list[ListeningItem] = []
    seen: set[str] = set()
    for number, entry in enumerate(_tables(raw, "item", "the file"), start=1):
        item = _item(entry, f"item number {number}")
        if item.id in {known.id for known in items}:
            raise ValueError(f"item {item.id!r}: the id is used twice")
        for question in item.questions:
            if question.id in seen:
                raise ValueError(f"item {item.id!r}, question {question.id!r}: the id is used twice")
            seen.add(question.id)
        items.append(item)
    for genre in GENRES:
        count = sum(item.genre == genre for item in items)
        if count < ITEMS_PER_GENRE:
            raise ValueError(f"the file: {ITEMS_PER_GENRE} items of genre {genre!r} are needed, it has {count}")
    return tuple(items)
