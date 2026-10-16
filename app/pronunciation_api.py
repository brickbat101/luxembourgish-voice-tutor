"""The routes under /api/pronunciation: the words the learner flagged as mispronounced, and the
respelling that fixes one.

A flag is kept in data/pronunciation_flags.json with the Piper phonemes of the word as they were
when it was flagged. Saving a respelling writes data/pronunciations.json, which the voice reads
again on its next synthesis, and takes the flag away. Both files are the app's, not the learner's.
A trial of a respelling is `respell` on POST /api/tts; it writes nothing.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.orchestrator import TurnError
from app.pronunciation_store import (
    TABLE_FILE,
    TableFileError,
    checked_respelling,
    checked_word,
    flag_lang,
    save_pronunciation,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pronunciation")

WORD_FIX = "send one word of 2 to 40 letters"
RESPELLING_FIX = "send a respelling of 1 to 60 characters, with no square bracket or one whole [[ phonemes ]] block"


class FlagIn(BaseModel):
    lang: str
    word: str
    text: str


class WordIn(BaseModel):
    lang: str
    word: str
    respelling: str


def phonemes_of(tts, lang: str, word: str) -> str:
    """The phonemes the voice makes of `word`; "" when the engine has none or fails."""
    phonemes = getattr(tts, "phonemes", None)
    if phonemes is None:
        return ""
    try:
        return phonemes(lang, word)
    except Exception as exc:
        log.warning("pronunciation: no phonemes for %r: %s: %s", word, type(exc).__name__, exc)
        return ""


@router.get("/flags")
def list_flags(request: Request) -> dict:
    """The flags, the latest first, each with the respelling the table has for its word now."""
    state = request.app.state
    return {
        "flags": [
            {**flag, "respelling": state.pronunciations.lookup(flag["lang"], flag["word"])}
            for flag in state.flags.list()
        ]
    }


@router.post("/flags")
def add_flag(request: Request, body: FlagIn) -> dict:
    """Flag a word of a line; flagging it again refreshes the entry and counts one more."""
    state = request.app.state
    try:
        word = checked_word(body.word)
    except ValueError as exc:
        raise TurnError(400, f"bad word: {exc}", WORD_FIX) from exc
    lang = flag_lang(body.lang)
    return state.flags.flag(lang, word, body.text, phonemes_of(state.tts, lang, word))


@router.delete("/flags/{flag_id}")
def dismiss_flag(request: Request, flag_id: str) -> dict:
    if not request.app.state.flags.dismiss(flag_id):
        raise TurnError(404, f"there is no flag {flag_id!r}", "reload the list of flags")
    return {"dismissed": flag_id}


@router.post("/words")
def save_word(request: Request, body: WordIn) -> dict:
    """Keep the respelling of a word in the user's pronunciation table; the voice uses it from now on."""
    state = request.app.state
    try:
        word = checked_word(body.word)
    except ValueError as exc:
        raise TurnError(400, f"bad word: {exc}", WORD_FIX) from exc
    try:
        respelling = checked_respelling(body.respelling)
    except ValueError as exc:
        raise TurnError(400, f"bad respelling: {exc}", RESPELLING_FIX) from exc
    lang = flag_lang(body.lang)
    path = state.settings.data_dir / TABLE_FILE
    try:
        save_pronunciation(path, lang, word, respelling)
    except TableFileError as exc:
        fix = f"fix or delete {path}, then save again; nothing was changed"
        raise TurnError(409, f"the pronunciation file cannot be used: {exc}", fix) from exc
    except OSError as exc:
        raise TurnError(500, f"the respelling could not be saved: {type(exc).__name__}: {exc}") from exc
    state.flags.forget_word(lang, word)
    return {"lang": lang, "word": word, "respelling": respelling}
