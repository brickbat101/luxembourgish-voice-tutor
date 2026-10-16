"""The fixed content of practice mode: the tutor's lines, the warm-up questions and the
questions per topic, from practice.toml, and the photos of the photo part, from photos.toml."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

PRACTICE_PATH = Path(__file__).resolve().parent / "practice.toml"
PHOTOS_PATH = Path(__file__).resolve().parent / "photos.toml"
PHOTO_FOLDER = "photos"  # in static/: the image of a photo is <id>.jpg in it
PHOTO_LICENCES = ("Public domain", "CC0")  # as Wikimedia Commons names them; no other is taken
PHOTOS_OFFERED = 3  # the learner picks one of this many
PHOTO_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")  # an id is part of a file name and of a URL
STAGES = ("general", "past", "present", "plans")  # in the order a topic goes through them
WARMUP_STAGE = "warmup"  # the stage of the warm-up questions; no topic question has it
WARMUPS_ASKED = 2  # warm-up questions a talk opens with
TUTOR_LINES = (
    "open_lb",
    "open_en",
    "ack_en",
    "said_en",
    "better_en",
    "could_say_en",
    "more_lb",
    "more_en",
    "again_en",
    "starter_en",
    "photo_intro_lb",
    "photo_intro_en",
    "photo_lb",
    "photo_en",
    "photo_last_lb",
    "photo_last_en",
    "end_en",
    "once_more_en",
    "no_answers_en",
    "eval_lb",
    "eval_en",
    "warmup_lb",
    "warmup_en",
)
# The placeholders a tutor line must hold.
PLACEHOLDERS = {"open_lb": "{topic}", "open_en": "{topic}", "again_en": "{question_en}"}


@dataclass(frozen=True)
class Question:
    id: str
    stage: str
    lb: str
    en: str
    starter: str = ""  # how an answer could begin, in Luxembourgish


@dataclass(frozen=True)
class Topic:
    id: str
    name_lb: str
    name_en: str
    questions: tuple[Question, ...]


@dataclass(frozen=True)
class Photo:
    id: str
    file: str  # its title on Wikimedia Commons, without "File:"
    source_url: str  # its page there
    licence: str  # one of PHOTO_LICENCES
    author: str
    description_en: str  # what it shows: all the judgement of an answer knows of the picture
    question_lb: str  # the one question that is about this scene
    question_en: str
    topic: str = ""  # the id of the practice topic it belongs to; "" for none

    @property
    def image(self) -> str:
        """Where the image is: under static/ as a path, from the page as a URL."""
        return f"{PHOTO_FOLDER}/{self.id}.jpg"


@dataclass(frozen=True)
class PracticeContent:
    tutor: dict[str, str]  # one text per name of TUTOR_LINES
    stage_lines: dict[str, tuple[str, str]]  # (lb, en) per stage after the first
    topics: tuple[Topic, ...]
    photos: tuple[Photo, ...] = ()  # empty: a talk has no photo part
    warmup: tuple[Question, ...] = ()  # a talk opens with WARMUPS_ASKED of them; empty: with its topic

    def topic(self, topic_id: str) -> Topic | None:
        return next((topic for topic in self.topics if topic.id == topic_id), None)


def _text(table: dict, key: str, where: str, required: bool = True) -> str:
    value = table.get(key, "")
    if not isinstance(value, str):
        raise ValueError(f"{where}: `{key}` is not a text")
    if required and not value.strip():
        raise ValueError(f"{where}: `{key}` is missing")
    return value.strip()


def _tables(parent: dict, key: str, where: str) -> list[dict]:
    """The `[[key]]` tables of `parent`; at least one."""
    tables = parent.get(key)
    if not isinstance(tables, list) or not tables or not all(isinstance(table, dict) for table in tables):
        raise ValueError(f"{where}: no [[{key}]] tables")
    return tables


def _question(raw: dict, where: str, last_stage: str) -> Question:
    stage = _text(raw, "stage", where)
    if stage not in STAGES:
        raise ValueError(f"{where}: unknown stage {stage!r}; known: {', '.join(STAGES)}")
    if last_stage and STAGES.index(stage) < STAGES.index(last_stage):
        raise ValueError(f"{where}: stage {stage!r} comes after {last_stage!r}; the order is {', '.join(STAGES)}")
    return Question(
        id=_text(raw, "id", where),
        stage=stage,
        lb=_text(raw, "lb", where),
        en=_text(raw, "en", where),
        starter=_text(raw, "starter", where, required=False),
    )


def _topic(raw: dict, where: str) -> Topic:
    topic_id = _text(raw, "id", where)
    where = f"topic {topic_id!r}"
    questions: list[Question] = []
    for number, entry in enumerate(_tables(raw, "question", where), start=1):
        last_stage = questions[-1].stage if questions else ""
        # A question is named by its id when it has one, else by its place in the topic.
        given = entry.get("id")
        named = repr(given) if isinstance(given, str) and given.strip() else f"number {number}"
        questions.append(_question(entry, f"{where}, question {named}", last_stage))
    return Topic(topic_id, _text(raw, "name_lb", where), _text(raw, "name_en", where), tuple(questions))


def _warmup(raw: dict, where: str) -> Question:
    question_id = _text(raw, "id", where)
    where = f"warm-up question {question_id!r}"
    return Question(
        id=question_id,
        stage=WARMUP_STAGE,
        lb=_text(raw, "lb", where),
        en=_text(raw, "en", where),
        starter=_text(raw, "starter", where, required=False),
    )


def load_practice(path: Path = PRACTICE_PATH) -> PracticeContent:
    """The content of the file at `path`; a ValueError says which entry of it is wrong."""
    try:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"not valid TOML: {exc}") from exc
    lines = raw.get("tutor")
    if not isinstance(lines, dict):
        raise ValueError("no [tutor] table")
    tutor = {name: _text(lines, name, "[tutor]") for name in TUTOR_LINES}
    for name, placeholder in PLACEHOLDERS.items():
        if placeholder not in tutor[name]:
            raise ValueError(f"[tutor]: `{name}` has no {placeholder}")
    stages = lines.get("stage")
    if not isinstance(stages, dict):
        raise ValueError("no [tutor.stage] table")
    stage_lines = {
        stage: (_text(stages, f"{stage}_lb", "[tutor.stage]"), _text(stages, f"{stage}_en", "[tutor.stage]"))
        for stage in STAGES[1:]
    }
    topics: list[Topic] = []
    seen: set[str] = set()
    for number, entry in enumerate(_tables(raw, "topic", "the file"), start=1):
        topic = _topic(entry, f"topic number {number}")
        if topic.id in {known.id for known in topics}:
            raise ValueError(f"topic {topic.id!r}: the id is used twice")
        for question in topic.questions:
            if question.id in seen:
                raise ValueError(f"topic {topic.id!r}, question {question.id!r}: the id is used twice")
            seen.add(question.id)
        topics.append(topic)
    warmup: list[Question] = []
    for number, entry in enumerate(_tables(raw, "warmup", "the file"), start=1):
        question = _warmup(entry, f"warm-up question number {number}")
        if question.id in seen:
            raise ValueError(f"warm-up question {question.id!r}: the id is used twice")
        seen.add(question.id)
        warmup.append(question)
    if len(warmup) < WARMUPS_ASKED:
        raise ValueError(f"the file: {len(warmup)} [[warmup]] tables, a talk opens with {WARMUPS_ASKED}")
    return PracticeContent(tutor, stage_lines, tuple(topics), warmup=tuple(warmup))


def _photo(raw: dict, where: str) -> Photo:
    photo_id = _text(raw, "id", where)
    where = f"photo {photo_id!r}"
    if not PHOTO_ID.fullmatch(photo_id):
        raise ValueError(f"{where}: the id may hold only a-z, 0-9 and single hyphens between them")
    licence = _text(raw, "licence", where)
    if licence not in PHOTO_LICENCES:
        raise ValueError(f"{where}: the licence {licence!r} is not one of {', '.join(PHOTO_LICENCES)}")
    texts = ("file", "source_url", "author", "description_en", "question_lb", "question_en")
    # One line each: the description runs over several lines in the file.
    fields = {key: " ".join(_text(raw, key, where).split()) for key in texts}
    return Photo(id=photo_id, licence=licence, topic=_text(raw, "topic", where, required=False), **fields)


def load_photos(path: Path = PHOTOS_PATH) -> tuple[Photo, ...]:
    """The photos of the file at `path`, PHOTOS_OFFERED or more; a ValueError says which entry of it is wrong."""
    try:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"not valid TOML: {exc}") from exc
    photos: list[Photo] = []
    for number, entry in enumerate(_tables(raw, "photo", "the file"), start=1):
        photo = _photo(entry, f"photo number {number}")
        if photo.id in {known.id for known in photos}:
            raise ValueError(f"photo {photo.id!r}: the id is used twice")
        photos.append(photo)
    if len(photos) < PHOTOS_OFFERED:
        raise ValueError(f"the file: {len(photos)} [[photo]] tables, the learner picks from {PHOTOS_OFFERED}")
    return tuple(photos)
