"""Pronunciation flags and saved respellings: the rules for a word and its respelling, the flags
file and the write into the user's pronunciation table.

A word is one run of letters, the same letters the pronunciation table matches
(`app.engines.tts_piper.LETTER`), so every flagged word can become a table key. Both files are
the app's, not the learner's: `data/pronunciation_flags.json` and `data/pronunciations.json`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import threading
import unicodedata
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from app.engines.tts_piper import read_pronunciations, word_key

log = logging.getLogger(__name__)

WORD = re.compile(r"[^\W\d_]{2,40}")
# Text without square brackets, or text around exactly one whole [[ phonemes ]] block.
BRACKETS = re.compile(r"[^\[\]]*(?:\[\[\s*[^\[\]\s][^\[\]]*\]\][^\[\]]*)?")
MAX_RESPELLING = 60
MAX_RESPELL = 5
MAX_FLAGS = 200
MAX_TEXT = 500
FLAGS_FILE = "pronunciation_flags.json"
TABLE_FILE = "pronunciations.json"
FLAG_FIELDS = {"id", "lang", "word", "text", "flagged_at", "phonemes", "count"}


def checked_word(word) -> str:
    """`word` when it is one run of 2 to 40 letters; ValueError otherwise."""
    if not isinstance(word, str) or not WORD.fullmatch(word):
        raise ValueError(f"{word!r} is not one word of 2 to 40 letters")
    return word


def checked_respelling(value) -> str:
    """The stripped `value` when it is a respelling of 1 to 60 characters; ValueError otherwise.

    It holds no control character and either no square bracket or one whole [[ phonemes ]] block.
    """
    if not isinstance(value, str):
        raise ValueError(f"the respelling is not text: {value!r}")
    value = value.strip()
    if not 1 <= len(value) <= MAX_RESPELLING:
        raise ValueError(f"the respelling has {len(value)} characters; it needs 1 to {MAX_RESPELLING}")
    if any(unicodedata.category(char) in ("Cc", "Zl", "Zp") for char in value):
        raise ValueError("the respelling holds a control character")
    if not BRACKETS.fullmatch(value):
        raise ValueError("square brackets are only allowed as one whole [[ phonemes ]] block")
    return value


def checked_respell(mapping) -> dict[str, str]:
    """{word: respelling} of at most 5 words, each through the two rules above; ValueError otherwise."""
    if not isinstance(mapping, Mapping):
        raise ValueError("the respelling is not an object of words")
    if len(mapping) > MAX_RESPELL:
        raise ValueError(f"{len(mapping)} words; at most {MAX_RESPELL}")
    return {checked_word(word): checked_respelling(value) for word, value in mapping.items()}


def flag_lang(lang: str) -> str:
    """The language of the voice that speaks a segment: "lb", else "en"."""
    return "lb" if lang == "lb" else "en"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, data) -> None:
    """`data` as UTF-8 JSON (no byte order mark, indent 2) in `path`, through a `.tmp` next to it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))
    os.replace(temp, path)


class FlagStore:
    """The flagged words as {"flags": [...]} in one file: one entry per (language, word)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def list(self) -> list[dict]:
        """The flags, the one flagged last first."""
        with self._lock:
            return list(reversed(self._read()))

    def flag(self, lang: str, word: str, text: str, phonemes: str = "") -> dict:
        """Flag `word` as it was spoken in `text`; a repeat refreshes the entry and counts one more."""
        lang, word = flag_lang(lang), checked_word(word)
        key = word_key(word)
        with self._lock:
            flags = self._read()
            known = next(
                (flag for flag in flags if flag["lang"] == lang and word_key(flag["word"]) == key), None
            )
            if known is None:
                known = {"id": secrets.token_hex(4), "lang": lang, "word": word, "count": 0}
            else:
                flags.remove(known)
            known.update(text=text.strip()[:MAX_TEXT], phonemes=phonemes, flagged_at=_now())
            known["count"] += 1
            flags.append(known)
            self._write(flags[-MAX_FLAGS:])
        return dict(known)

    def dismiss(self, flag_id: str) -> bool:
        """Remove the flag with this id; False when there is none."""
        with self._lock:
            flags = self._read()
            kept = [flag for flag in flags if flag["id"] != flag_id]
            if len(kept) == len(flags):
                return False
            self._write(kept)
        return True

    def forget_word(self, lang: str, word: str) -> int:
        """Remove the flag of this word in this language; how many were removed (0 or 1)."""
        lang, key = flag_lang(lang), word_key(word)
        with self._lock:
            flags = self._read()
            kept = [flag for flag in flags if not (flag["lang"] == lang and word_key(flag["word"]) == key)]
            if len(kept) != len(flags):
                self._write(kept)
        return len(flags) - len(kept)

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8-sig"))
            flags = data["flags"] if isinstance(data, dict) else None
            if not isinstance(flags, list) or not all(
                isinstance(flag, dict) and FLAG_FIELDS <= set(flag) and isinstance(flag["word"], str)
                for flag in flags
            ):
                raise ValueError("not a list of flags")
        except (OSError, ValueError, KeyError) as exc:
            corrupt = self.path.with_name(self.path.name + ".corrupt")
            log.warning(
                "pronunciation flags: %s cannot be read (%s: %s); it is renamed %s and the list starts empty",
                self.path,
                type(exc).__name__,
                exc,
                corrupt.name,
            )
            os.replace(self.path, corrupt)
            return []
        return flags

    def _write(self, flags: list[dict]) -> None:
        _write_json(self.path, {"flags": flags})


class TableFileError(Exception):
    """The user's pronunciation table cannot be read, so it is not written over."""


_table_lock = threading.Lock()


def save_pronunciation(path: Path, lang: str, word: str, respelling: str) -> str:
    """Set the respelling of `word` in the user's table `path`; the respelling as it was kept.

    Raises ValueError for a word or respelling that breaks the rules, and TableFileError, leaving the
    file as it is, when the file exists and is not a valid table.
    """
    lang, word, respelling = flag_lang(lang), checked_word(word), checked_respelling(respelling)
    path = Path(path)
    key = word_key(word)
    with _table_lock:
        table = {}
        if path.exists():
            try:
                table = read_pronunciations(path)
            except (OSError, ValueError) as exc:
                raise TableFileError(f"{path} cannot be read ({type(exc).__name__}: {exc})") from exc
        words = {name: value for name, value in table.get(lang, {}).items() if word_key(name) != key}
        words[key] = respelling
        table[lang] = words
        _write_json(path, table)
    return respelling
