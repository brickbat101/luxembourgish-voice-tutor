"""The confirmed answers of practice talks, kept on this PC: their text and their recordings.

Everything is in one folder directly inside the data folder. A talk has a folder of its own
there, named by the day and the session: `answers.jsonl` with one JSON line per answer, and
next to it the recordings as they were uploaded. The store reads, writes and deletes nothing
outside its folder, and does nothing at all when that folder is not really where it is named:
a link at its place, or a path that leads out of the data folder, is refused.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from app.engines.stt_faster_whisper import suffix_for

ANSWERS_FILE = "answers.jsonl"
SESSION_CHARS = 8  # of a session id, in the name of the folder of its talks
_NOT_IN_A_NAME = re.compile(r"[^0-9A-Za-z]")
_NOT_IN_A_FOLDER_NAME = re.compile(r"[^0-9A-Za-z-]")
_FOLDER_NAME = re.compile(r"\d{4}-\d\d-\d\d-(.+)")
NOT_KEPT = "not kept: "  # the start of the text of a StoreError for an answer that is not there
NOTE_CHARS = 200  # of the learner's note on a dispute


class StoreError(Exception):
    """The store could not be read, written or deleted; the text names the path."""


def session_part(session_id: str) -> str:
    """What of `session_id` names the folders of its talks, after their day."""
    return _NOT_IN_A_NAME.sub("", session_id)[:SESSION_CHARS] or "session"


def is_session_folder(folder: str, session_id: str) -> bool:
    """Whether `folder` is the name of the folder of a talk of `session_id`, of any day."""
    match = _FOLDER_NAME.fullmatch(folder)
    return bool(match) and match.group(1) == session_part(session_id)


def _local_now() -> datetime:
    return datetime.now().astimezone()


def _replace(path: Path, data: bytes) -> None:
    """Writes `data` as the file `path`, which is the old file or the new one, never half of it."""
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


class PracticeStore:
    """The answers under `path`, which must be a folder directly inside `data_dir`.

    `clock` gives the local time an answer is saved or disputed at.
    """

    def __init__(self, path: Path, data_dir: Path, clock: Callable[[], datetime] = _local_now) -> None:
        self.path = Path(path)
        self.data_dir = Path(data_dir)
        self._clock = clock
        self._lock = threading.Lock()

    def save(self, session_id: str, answer: dict, audio: bytes | None = None, mime: str = "") -> dict:
        """Keeps one answer of the session; where: `folder`, relative to the store, and `line`, from 1.

        `answer` is the line without its `time` and `audio_file`, which are added here.
        `audio` is the recording of the answer as it was uploaded, of the type `mime`; without
        one `audio_file` is null. The recording is `<n><suffix>` next to the lines, n being
        the number of the line.
        """
        with self._lock:
            now = self._clock()
            folder = self._folder() / f"{now.date().isoformat()}-{session_part(session_id)}"
            lines = folder / ANSWERS_FILE
            recording = None
            try:
                folder.mkdir(parents=True, exist_ok=True)
                kept = lines.read_text(encoding="utf-8") if lines.exists() else ""
                number = len(kept.splitlines()) + 1
                if audio:
                    recording = folder / f"{number}{suffix_for(mime)}"
                    _replace(recording, audio)
                line = {
                    "time": now.isoformat(timespec="seconds"),
                    **answer,
                    "audio_file": recording.name if recording else None,
                }
                _replace(lines, f"{kept}{json.dumps(line, ensure_ascii=False)}\n".encode())
            except OSError as exc:
                if recording is not None:
                    # No recording stays without its line.
                    with contextlib.suppress(OSError):
                        recording.unlink()
                raise StoreError(f"this answer was not saved in {folder}: {type(exc).__name__}: {exc}") from exc
            return {"folder": folder.name, "line": number}

    def dispute(self, folder: str, line: int, note: str = "") -> dict:
        """Marks answer `line` (from 1) in `folder` as one the learner thinks was right; the line as it is then.

        `disputed` is the local time and `note`, cut to NOTE_CHARS. A line that is disputed
        already keeps its first mark, and the file is not written. `folder` is the name of a
        folder directly inside the store; a folder or a line that is not there is a StoreError
        whose text starts with NOT_KEPT.
        """
        with self._lock:
            root = self._folder()
            if not folder or _NOT_IN_A_FOLDER_NAME.search(folder):
                raise StoreError(f"{NOT_KEPT}{folder!r} is not the name of a folder in {root}")
            place = root / folder
            lines = place / ANSWERS_FILE
            try:
                if place.is_symlink() or place.is_junction() or (place.exists() and place.resolve().parent != root):
                    raise StoreError(f"{place} is a link, not a folder of the saved answers; nothing is written there")
                if lines.is_symlink() or not lines.is_file():
                    raise StoreError(f"{NOT_KEPT}there are no answers in {place}")
                kept = lines.read_text(encoding="utf-8").splitlines(keepends=True)
            except OSError as exc:
                raise StoreError(f"the answers in {place} cannot be read: {type(exc).__name__}: {exc}") from exc
            if not 1 <= line <= len(kept):
                raise StoreError(f"{NOT_KEPT}there is no answer {line} in {lines}")
            try:
                answer = json.loads(kept[line - 1])
            except ValueError as exc:
                raise StoreError(f"line {line} of {lines} is not JSON: {exc}") from exc
            if not isinstance(answer, dict):
                raise StoreError(f"line {line} of {lines} is not an answer")
            if answer.get("disputed"):
                return answer
            answer["disputed"] = {"time": self._clock().isoformat(timespec="seconds"), "note": note[:NOTE_CHARS]}
            kept[line - 1] = json.dumps(answer, ensure_ascii=False) + "\n"
            try:
                _replace(lines, "".join(kept).encode())
            except OSError as exc:
                raise StoreError(f"answer {line} in {lines} was not marked: {type(exc).__name__}: {exc}") from exc
            return answer

    def summary(self) -> dict:
        """What is kept: `answers` (lines), `sessions` (folders with lines), `bytes` (of every
        file under the folder) and `folder`, where it all is."""
        with self._lock:
            root = self._folder()
            answers = sessions = size = 0
            try:
                # os.walk does not go into a link to a folder, and a link is not counted as its target.
                for folder, _, names in os.walk(root):
                    for name in names:
                        file = Path(folder) / name
                        size += file.lstat().st_size
                        if name == ANSWERS_FILE:
                            lines = file.read_text(encoding="utf-8").splitlines()
                            sessions += 1
                            answers += sum(1 for line in lines if line.strip())
            except OSError as exc:
                raise StoreError(f"the saved answers in {root} cannot be counted: {type(exc).__name__}: {exc}") from exc
            return {"answers": answers, "bytes": size, "sessions": sessions, "folder": str(self.path)}

    def delete_all(self) -> None:
        """Removes the folder of the store with everything in it, and nothing else. A folder
        that is not there is fine; none is made."""
        with self._lock:
            root = self._folder()
            try:
                shutil.rmtree(root)
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise StoreError(
                    f"the saved answers in {root} could not all be deleted: {type(exc).__name__}: {exc}"
                ) from exc

    def _folder(self) -> Path:
        """The folder of the store as it is on disk; a StoreError when that is not the folder
        of its name directly inside the data folder."""
        try:
            data_dir = self.data_dir.resolve()
            folder = self.path.resolve()
        except OSError as exc:
            raise StoreError(f"the folder {self.path} cannot be looked at: {type(exc).__name__}: {exc}") from exc
        if folder.parent != data_dir or os.path.normcase(folder.name) != os.path.normcase(self.path.name):
            raise StoreError(
                f"the folder for saved answers, {self.path}, is not a folder of its own inside {data_dir}"
                f" (it leads to {folder}); nothing is saved in it or deleted from it"
            )
        return folder
