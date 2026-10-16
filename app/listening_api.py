"""The routes under /api/listening: the items, one item read aloud at a speed, the learner's answers checked.

No route needs a session: `play` renders the item afresh on every call, so replays are free, and
`check` takes the item id with every answer. `play` gives the questions without their answers;
`check` gives which were right, why, and the transcript.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Form, Request

from app.api import audio_json, voice_override
from app.content.listening import ListeningItem
from app.listening import SPEED_LABELS, SPEEDS, check_answers, render_item
from app.orchestrator import TurnError

LISTENING_FILE = "app/content/listening.toml"

router = APIRouter(prefix="/api/listening")


def listening_items(state) -> tuple[ListeningItem, ...]:
    if state.listening is None:
        raise TurnError(
            503,
            f"the listening content ({LISTENING_FILE}) cannot be used: {state.listening_detail}",
            "fix the file and restart",
        )
    return state.listening


def known_item(state, item_id: str) -> ListeningItem:
    items = listening_items(state)
    for item in items:
        if item.id == item_id:
            return item
    raise TurnError(400, f"unknown item {item_id!r}; known: {', '.join(item.id for item in items)}")


def checked_speed(speed: str) -> str:
    if speed not in SPEEDS:
        raise TurnError(400, f"unknown speed {speed!r}; known: {', '.join(SPEEDS)}")
    return speed


def checked_answers(answers: str) -> dict:
    try:
        parsed = json.loads(answers)
    except ValueError as exc:
        raise TurnError(400, f"bad answers: not JSON: {exc}", "send `answers` as a JSON object") from exc
    if not isinstance(parsed, dict):
        raise TurnError(400, "bad answers: not a JSON object", "send question id -> option index, from 0")
    return parsed


@router.get("/items")
def listening_list(request: Request) -> dict:
    return {
        "items": [
            {
                "id": item.id,
                "genre": item.genre,
                "title_en": item.title_en,
                "level": item.level,
                "questions": len(item.questions),
                "words": item.words,
            }
            for item in listening_items(request.app.state)
        ]
    }


@router.post("/play")
def listening_play(
    request: Request,
    item_id: str = Form(...),
    speed: str = Form(...),
    voice: str | None = Form(None),
) -> dict:
    """The item read aloud at `speed`, with its questions; no answer, reason or transcript."""
    state = request.app.state
    item = known_item(state, item_id)
    speed = checked_speed(speed)
    override = voice_override(state.settings, voice)
    try:
        audio = render_item(state.tts, item, speed, override)
    except Exception as exc:
        raise TurnError(500, f"tts engine ({state.tts.name}) failed: {type(exc).__name__}: {exc}") from exc
    return {
        "item": {
            "id": item.id,
            "genre": item.genre,
            "title_en": item.title_en,
            "level": item.level,
            "speed": speed,
            "speed_label": SPEED_LABELS[speed],
            "questions": [
                {"id": question.id, "lb": question.lb, "en": question.en, "options": list(question.options)}
                for question in item.questions
            ],
        },
        **audio_json(audio),
    }


@router.post("/check")
def listening_check(request: Request, item_id: str = Form(...), answers: str = Form(...)) -> dict:
    """`answers` is a JSON object of question id -> the index of the chosen option."""
    item = known_item(request.app.state, item_id)
    return {"item_id": item.id, **check_answers(item, checked_answers(answers))}
