"""The learner's history in one SQLite database on this PC, with its daily backups.

Everything is in one folder directly inside the data folder, `learner/`: the database
`learner.sqlite`, `backups/learner-YYYY-MM-DD.sqlite` (the last 7 days the app was started)
and `pre-migration/`, a copy of phrases.json and practice/ as they were before the first
import. A link at the place of that folder, or a path that leads out of the data folder, is
refused. A database that fails `PRAGMA quick_check` is left exactly as it is: the store does
not open, and the error names the newest backup to restore by hand.

phrases.json and practice/ are only read here, never changed; their own stores keep writing them.
After the import the app records here as it goes: sessions, chat and teacher turns, the practice
answers the practice store keeps, and the phrases met.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import threading
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from app import fsrs
from app.lod.checks import lemmas
from app.orchestrator import PHRASE_FIELDS, phrase_key
from app.practice_store import ANSWERS_FILE, StoreError

if TYPE_CHECKING:
    from app.lod.store import LodStore

T = TypeVar("T")

LEARNER_FOLDER = "learner"  # in the data folder: the database, its backups, the pre-migration copy
DATABASE = "learner.sqlite"
BACKUPS = "backups"
PRE_MIGRATION = "pre-migration"
KEEP_BACKUPS = 7
PHRASES_FILE = "phrases.json"
PRACTICE_FOLDER = "practice"
PRACTICE_FIELDS = "practice-fields"  # the pass that copies category, label and disputed of answers imported before
CORRECTED_VERDICTS = ("fix", "english")  # a correction of these verdicts becomes a review item
QUEUE_LIMIT = 20
NEW_PER_DAY = 10  # items seen for the first time per UTC day
REVIEW_MODES = ("typed", "spoken")
RATED_BY = ("auto", "learner")
CATEGORIES = ("words", "grammar", "task")  # of a correction, as the progress counts them
PROGRESS_WEEKS = 4
_ITEM_FIELDS = "id, source, form, gloss, hint, state, due"
_STRIPPED = re.compile(r"[.,!?;:…\"“”„«»]")

_SCHEMA_V1 = (
    "CREATE TABLE learner (id INTEGER PRIMARY KEY, created TEXT NOT NULL, settings_json TEXT NOT NULL)",
    "CREATE TABLE session (id TEXT PRIMARY KEY, started TEXT NOT NULL, last_seen TEXT NOT NULL,"
    " modes TEXT NOT NULL DEFAULT '')",
    "CREATE TABLE turn (id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL,"
    " mode TEXT NOT NULL, text TEXT NOT NULL, duration_ms INTEGER NULL, at TEXT NOT NULL)",
    "CREATE TABLE phrase (id INTEGER PRIMARY KEY, phrase_key TEXT NOT NULL UNIQUE, lb TEXT NOT NULL,"
    " en TEXT NOT NULL, first_seen TEXT NOT NULL, last_seen TEXT NOT NULL, times_seen INTEGER NOT NULL,"
    " sessions_json TEXT NOT NULL)",
    "CREATE TABLE answer (id INTEGER PRIMARY KEY, folder TEXT, line INTEGER, session_id TEXT NULL,"
    " topic_id TEXT, question_id TEXT, question_lb TEXT, at TEXT NOT NULL, raw_transcript TEXT, text TEXT,"
    " help_used INTEGER, duration_ms INTEGER NULL, verdict TEXT NULL, judge_model TEXT NULL,"
    " audio_file TEXT NULL, UNIQUE (folder, line))",
    "CREATE TABLE correction (id INTEGER PRIMARY KEY, answer_id INTEGER NOT NULL UNIQUE REFERENCES answer(id),"
    " verdict TEXT, reaction_en TEXT, quote TEXT, better_lb TEXT, why_en TEXT,"
    " claims_json TEXT NOT NULL DEFAULT '[]', category TEXT NULL, label TEXT NULL,"
    " disputed INTEGER NOT NULL DEFAULT 0)",
    "CREATE TABLE item (id INTEGER PRIMARY KEY, source TEXT NOT NULL CHECK (source IN ('phrase', 'correction')),"
    " source_id INTEGER NOT NULL, form TEXT NOT NULL, gloss TEXT NOT NULL, hint TEXT NULL, created TEXT NOT NULL,"
    " state TEXT NOT NULL CHECK (state IN ('new', 'learning', 'review', 'relearning')), step INTEGER NULL,"
    " stability REAL NULL, difficulty REAL NULL, due TEXT NULL, last_review TEXT NULL,"
    " reps INTEGER NOT NULL DEFAULT 0, lapses INTEGER NOT NULL DEFAULT 0, UNIQUE (source, source_id))",
    "CREATE TABLE review_log (id INTEGER PRIMARY KEY, item_id INTEGER NOT NULL REFERENCES item(id),"
    " at TEXT NOT NULL, rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 4), mode TEXT, heard TEXT,"
    " confirmed TEXT, matched INTEGER, rated_by TEXT, superseded INTEGER NOT NULL DEFAULT 0)",
    "CREATE TABLE migrated (source TEXT PRIMARY KEY, at TEXT NOT NULL, rows INTEGER NOT NULL)",
    "CREATE INDEX item_due ON item (due)",
    "CREATE INDEX review_log_item_at ON review_log (item_id, at)",
    "CREATE INDEX answer_at ON answer (at)",
    "CREATE INDEX turn_session ON turn (session_id)",
)
TABLES = ("learner", "session", "turn", "phrase", "answer", "correction", "item", "review_log", "migrated")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def _moment(text: str) -> datetime:
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def _parsed(text: str | None) -> datetime | None:
    """A stored time in UTC; None for one that is missing, not a time or without a time zone."""
    try:
        moment = datetime.fromisoformat(str(text))
    except ValueError:
        return None
    return moment.astimezone(timezone.utc) if moment.tzinfo else None


def _week(moment: datetime | None) -> str | None:
    """The ISO week of `moment` as `YYYY-Www`; None for None."""
    if moment is None:
        return None
    year, week, _ = moment.isocalendar()
    return f"{year}-W{week:02d}"


def _minutes(ms: int) -> float:
    return round(ms / 60000, 1)


def _second(moment: datetime) -> datetime:
    """`moment` in UTC as a row stores it, to the second."""
    if moment.tzinfo is None:
        raise ValueError(f"the time {moment.isoformat()} has no time zone")
    return moment.astimezone(timezone.utc).replace(microsecond=0)


def _normalised(text: str) -> str:
    return " ".join(_STRIPPED.sub(" ", text.casefold().replace("’", "'")).split())


def _text_or_none(value) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _correction_fields(line: dict) -> tuple[str | None, str | None, int]:
    """(category, label, disputed) of the correction of an answer line: None for a missing or empty
    category or label, disputed 1 when the line holds the learner's dispute."""
    return _text_or_none(line.get("category")), _text_or_none(line.get("label")), int(bool(line.get("disputed")))


def matches(answer: str, expected: str, lod: LodStore | None) -> bool:
    """Whether `answer` says `expected`: the same once normalised (case, ’, punctuation, spaces),
    or the same number of words with each word equal or a form of a LOD lemma of the expected one."""
    said, wanted = _normalised(answer), _normalised(expected)
    if said == wanted:
        return True
    if lod is None or not said:
        return False
    said_words, wanted_words = said.split(), wanted.split()
    if len(said_words) != len(wanted_words):
        return False
    return all(a == b or lemmas(lod, a) & lemmas(lod, b) for a, b in zip(said_words, wanted_words, strict=True))


def _card(item: sqlite3.Row) -> fsrs.Card:
    """The FSRS card of the cached columns of `item`."""
    if item["state"] == "new":
        return fsrs.new_card(_moment(item["created"]))
    return fsrs.Card(
        item["state"],
        item["step"],
        item["stability"],
        item["difficulty"],
        _moment(item["due"]),
        _moment(item["last_review"]) if item["last_review"] else None,
    )


def _schema_v1(connection: sqlite3.Connection, now: str) -> None:
    for statement in _SCHEMA_V1:
        connection.execute(statement)
    connection.execute("INSERT INTO learner (id, created, settings_json) VALUES (1, ?, '{}')", (now,))


# Index n takes user_version n to n + 1; a step gets the connection and the time it runs at.
MIGRATIONS: list[Callable[[sqlite3.Connection, str], None]] = [_schema_v1]


def _is_link(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _skipped(path: Path) -> bool:
    """What the import neither reads nor copies: links, junctions and half-written `*.tmp` files."""
    return path.name.endswith(".tmp") or _is_link(path)


def _tree_size(root: Path) -> int:
    size = 0
    for folder, _, names in os.walk(root):
        size += sum((Path(folder) / name).lstat().st_size for name in names)
    return size


class LearnerStore:
    """The database under `folder`, which must be a folder directly inside `data_dir`.

    `clock` gives the time rows are written at and the UTC day of the backup. Every public
    method holds one lock, so the one connection is shared safely by the server's threads.
    """

    def __init__(self, folder: Path, data_dir: Path, clock: Callable[[], datetime] = _utc_now) -> None:
        self.folder = Path(folder)
        self.data_dir = Path(data_dir)
        self._clock = clock
        self._lock = threading.Lock()
        self._connection: sqlite3.Connection | None = None

    @property
    def path(self) -> Path:
        return self.folder / DATABASE

    def open(self) -> None:
        """Connects, checks the file and brings its schema to the newest version."""
        with self._lock:
            if self._connection is not None:
                return
            folder = self._folder()
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise StoreError(f"the folder {folder} cannot be made: {type(exc).__name__}: {exc}") from exc
            folder = self._folder()
            path = folder / DATABASE
            connection = None
            try:
                connection = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
                connection.row_factory = sqlite3.Row
                for pragma in ("synchronous=NORMAL", "foreign_keys=ON", "busy_timeout=5000", "secure_delete=ON"):
                    connection.execute(f"PRAGMA {pragma}")
                # Before WAL, which writes to the file: a file that fails the check is not touched.
                result = [row[0] for row in connection.execute("PRAGMA quick_check")]
                if result != ["ok"]:
                    raise sqlite3.DatabaseError("; ".join(str(line) for line in result[:5]))
                connection.execute("PRAGMA journal_mode=WAL")
            except sqlite3.Error as exc:
                if connection is not None:
                    connection.close()
                raise StoreError(
                    f"the learner database {path} fails its check ({type(exc).__name__}: {exc}) and is left as it"
                    f" is; {self._newest_backup_text()}"
                ) from exc
            try:
                self._migrate(connection, path)
            except BaseException:
                connection.close()
                raise
            self._connection = connection

    def close(self) -> None:
        with self._lock:
            connection, self._connection = self._connection, None
            if connection is None:
                return
            try:
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                connection.close()

    def counts(self) -> dict[str, int]:
        """Rows per table."""
        with self._lock:
            connection = self._open_connection()
            return {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in TABLES}

    def export(self, copy: Path) -> dict:
        """Every row of every table and a `VACUUM INTO` copy of the database at `copy`, a path that
        does not exist yet, taken in one hold of the lock so that both show the same rows:
        {exported_at, schema_version, tables: {table: [row, ...]}}."""
        with self._lock:
            connection = self._open_connection()
            try:
                tables = {
                    table: [dict(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY rowid")]
                    for table in TABLES
                }
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                connection.execute("VACUUM INTO ?", (str(copy),))
            except sqlite3.Error as exc:
                raise StoreError(
                    f"the learner database {self.path} was not exported: {type(exc).__name__}: {exc}"
                ) from exc
            return {"exported_at": _iso(self._clock()), "schema_version": version, "tables": tables}

    def kept_files(self) -> list[tuple[str, Path]]:
        """(name relative to the data folder, with "/", path) of phrases.json and phrases.json.corrupt
        and of every file in the folders of practice/: each answers.jsonl and the recordings next to
        it. Links, junctions and `*.tmp` files are left out, as the import leaves them out."""
        found = []
        for name in (PHRASES_FILE, PHRASES_FILE + ".corrupt"):
            path = self.data_dir / name
            if not _is_link(path) and path.is_file():
                found.append((name, path))
        for folder in self._answer_folders():
            for child in sorted(folder.iterdir()):
                if not _skipped(child) and child.is_file():
                    found.append((f"{PRACTICE_FOLDER}/{folder.name}/{child.name}", child))
        return found

    def delete_all(self) -> None:
        """Empties every table (secure_delete overwrites the rows), truncates the WAL, closes the
        store and removes its folder with everything in it: the database with its -wal and -shm,
        the backups and the pre-migration copy. A folder that is not there is fine. A folder that
        is a link, or leads out of the data folder, is refused and nothing is deleted. The store
        stays closed."""
        with self._lock:
            folder = self._folder()
            connection, self._connection = self._connection, None
            if connection is not None:
                try:
                    for table in reversed(TABLES):  # the referring rows before the rows they refer to
                        connection.execute(f"DELETE FROM {table}")
                    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.Error:
                    pass  # the files go next in any case
                finally:
                    connection.close()
            try:
                shutil.rmtree(folder)
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise StoreError(
                    f"the learner store in {folder} could not all be deleted: {type(exc).__name__}: {exc}"
                ) from exc

    def migrate_files(self, questions: dict[str, str] | None = None) -> dict[str, int]:
        """Copies phrases.json and the practice answers in, once each; the rows added per source.

        `questions` maps a question id to its English, the gloss of a correction item; without
        it the item's gloss is the question in Luxembourgish. The originals are only read.
        A database whose practice answers were imported before category, label and disputed were
        copied gets one more pass, PRACTICE_FIELDS: the answer lines it lacks are added and the
        empty fields of its corrections are filled from their lines.
        """
        questions = questions or {}
        with self._lock:
            connection = self._open_connection()
            done = {row["source"] for row in connection.execute("SELECT source FROM migrated")}
            phrases = None if PHRASES_FILE in done else self._phrase_entries()
            answers = None if PRACTICE_FOLDER in done else self._answer_lines()
            again = None if PRACTICE_FOLDER not in done or PRACTICE_FIELDS in done else self._answer_lines()
            if phrases or answers:
                self._pre_migration_copy()
            added = {}
            now = _iso(self._clock())
            if phrases is not None:
                added[PHRASES_FILE] = self._transaction(lambda: self._insert_phrases(phrases, now))
            if answers is not None:
                added[PRACTICE_FOLDER] = self._transaction(
                    lambda: self._insert_answers(answers, questions, now, (PRACTICE_FOLDER, PRACTICE_FIELDS))
                )
            if again is not None:
                added[PRACTICE_FIELDS] = self._transaction(
                    lambda: self._insert_answers(again, questions, now, (PRACTICE_FIELDS,))
                )
            return added

    def backup_daily(self) -> Path | None:
        """Today's backup, made if missing; the oldest beyond KEEP_BACKUPS are removed. The new file or None."""
        with self._lock:
            connection = self._open_connection()
            backups = self._child(BACKUPS)
            target = backups / f"learner-{self._clock().astimezone(timezone.utc).date().isoformat()}.sqlite"
            made = None
            try:
                backups.mkdir(exist_ok=True)
                if not target.exists():
                    connection.execute("VACUUM INTO ?", (str(target),))
                    made = target
                for old in sorted(backups.glob("learner-*.sqlite"))[:-KEEP_BACKUPS]:
                    old.unlink()
            except (OSError, sqlite3.Error) as exc:
                raise StoreError(f"the backup {target} was not made: {type(exc).__name__}: {exc}") from exc
            return made

    def item(self, item_id: int) -> dict | None:
        """Every column of the review item, or None when there is none of that id."""
        with self._lock:
            row = self._item_row(item_id)
            return dict(row) if row is not None else None

    def queue(
        self, now: datetime | None = None, limit: int = QUEUE_LIMIT, new_per_day: int = NEW_PER_DAY
    ) -> list[dict]:
        """The items to review at `now`: those seen before and due, by due time, then new items by
        creation, as many as `new_per_day` leaves after the new items seen today (UTC). At most `limit`."""
        now, limit = _second(now or self._clock()), max(limit, 0)
        with self._lock:
            connection = self._open_connection()
            due = connection.execute(
                f"SELECT {_ITEM_FIELDS} FROM item WHERE state != 'new' AND due <= ? ORDER BY due, id LIMIT ?",
                (_iso(now), limit),
            ).fetchall()
            room = min(limit - len(due), self._new_left(now, new_per_day))
            new = connection.execute(
                f"SELECT {_ITEM_FIELDS} FROM item WHERE state = 'new' ORDER BY created, id LIMIT ?", (max(room, 0),)
            ).fetchall()
            return [dict(row) for row in (*due, *new)]

    def review_counts(self, now: datetime | None = None, new_per_day: int = NEW_PER_DAY) -> dict[str, int]:
        """due: seen before and due at `now`; new_available: new items the queue may still show today;
        in_review: seen before; total: all items."""
        now = _second(now or self._clock())
        with self._lock:
            connection = self._open_connection()
            row = connection.execute(
                "SELECT COUNT(*) AS total, COALESCE(SUM(state = 'new'), 0) AS new,"
                " COALESCE(SUM(state != 'new' AND due <= ?), 0) AS due FROM item",
                (_iso(now),),
            ).fetchone()
            return {
                "due": row["due"],
                "new_available": min(row["new"], self._new_left(now, new_per_day)),
                "in_review": row["total"] - row["new"],
                "total": row["total"],
            }

    def review(
        self,
        item_id: int,
        rating: int,
        now: datetime | None = None,
        mode: str = "typed",
        heard: str = "",
        confirmed: str = "",
        matched: bool = False,
        rated_by: str = "learner",
    ) -> dict:
        """Logs `rating` of the item at `now` and schedules it from its cached state, in one
        transaction: {log_id, item_id, rating, state, due}. A ValueError for an unknown item or a bad rating."""
        now = _second(now or self._clock())
        with self._lock:
            return self._write(lambda: self._reviewed(item_id, rating, now, mode, heard, confirmed, matched, rated_by))

    def override(self, log_id: int, rating: int, now: datetime | None = None, item_id: int | None = None) -> dict:
        """Replaces the automatic rating of row `log_id` with the learner's `rating`: the row is marked
        superseded, the learner's row is added with the same answer, and the item is rebuilt from its log.
        {log_id of the new row, item_id, rating, state, due, superseded: `log_id`}. A ValueError when the
        row is not the latest valid row of its item (of `item_id`, when given) or was not rated automatically."""
        now = _second(now or self._clock())
        with self._lock:
            return self._write(lambda: self._overridden(log_id, rating, now, item_id))

    def rebuild(self, item_id: int) -> dict:
        """Rewrites the item's cached state, reps and lapses from its log rows that are not superseded;
        the item as it is then. A ValueError for an unknown item."""
        with self._lock:
            return self._write(lambda: self._rebuilt(item_id))

    def mark_disputed(self, folder: str, line: int) -> bool:
        """Marks the correction of the answer kept at line `line` of `folder` as disputed;
        whether there is such a correction."""
        with self._lock:
            connection = self._open_connection()
            try:
                marked = connection.execute(
                    "UPDATE correction SET disputed = 1"
                    " WHERE answer_id = (SELECT id FROM answer WHERE folder = ? AND line = ?)",
                    (folder, line),
                ).rowcount
            except sqlite3.Error as exc:
                raise StoreError(
                    f"the dispute was not written to {self.path}: {type(exc).__name__}: {exc}"
                ) from exc
            return bool(marked)

    def session_seen(self, session_id: str, mode: str = "") -> None:
        """Records that the session is in use now, in `mode` ("chat", "teacher", "practice"; "" for none)."""
        with self._lock:
            now = _iso(self._clock())
            self._record(lambda: self._seen(session_id, mode, now))

    def turn_said(self, session_id: str, mode: str, user_text: str, duration_ms: int, tutor_text: str) -> None:
        """Records one chat or teacher turn of the session: the learner's text with the milliseconds
        they spoke it (0 when typed), then the tutor's reply."""
        with self._lock:
            now = _iso(self._clock())

            def write() -> None:
                connection = self._open_connection()
                said = max(int(duration_ms), 0)
                for role, text, duration in (("user", user_text, said), ("tutor", tutor_text, None)):
                    connection.execute(
                        "INSERT INTO turn (session_id, role, mode, text, duration_ms, at) VALUES (?, ?, ?, ?, ?, ?)",
                        (session_id, role, mode, text, duration, now),
                    )
                self._seen(session_id, mode, now)

            self._record(write)

    def phrase_met(self, lb: str, en: str, session_id: str) -> bool:
        """Records one meeting with a phrase as PhraseMemory.add does; a new phrase also becomes a
        review item. True when the phrase is new to the store."""
        key = phrase_key(lb)
        if not key:
            return False
        with self._lock:
            now = _iso(self._clock())

            def write() -> bool:
                connection = self._open_connection()
                known = connection.execute("SELECT * FROM phrase WHERE phrase_key = ?", (key,)).fetchone()
                if known is None:
                    phrase_id = connection.execute(
                        "INSERT INTO phrase (phrase_key, lb, en, first_seen, last_seen, times_seen, sessions_json)"
                        " VALUES (?, ?, ?, ?, ?, 1, ?)",
                        (key, lb.strip(), en.strip(), now, now, json.dumps([session_id], ensure_ascii=False)),
                    ).lastrowid
                    self._insert_item("phrase", phrase_id, lb.strip(), en.strip(), None, now)
                    return True
                sessions = json.loads(known["sessions_json"])
                if session_id not in sessions:
                    sessions.append(session_id)
                connection.execute(
                    "UPDATE phrase SET last_seen = ?, times_seen = times_seen + 1, sessions_json = ? WHERE id = ?",
                    (now, json.dumps(sessions, ensure_ascii=False), known["id"]),
                )
                return False

            return self._record(write)

    def answer_kept(
        self,
        session_id: str,
        folder: str,
        line: int,
        record: dict,
        audio_file: str | None,
        duration_ms: int,
        judge_model: str | None,
        question_en: str | None,
    ) -> bool:
        """Records the practice answer the practice store has just kept at line `line` of `folder`.

        `record` is the line it kept, without its `time` and `audio_file`; `audio_file` is the name
        of the recording next to it, None without one. Its correction keeps the line's category and
        label; a corrected verdict with a better version becomes a review item glossed with
        `question_en`. False when an answer is recorded at that place already.
        """
        with self._lock:
            now = _iso(self._clock())
            audio = f"{PRACTICE_FOLDER}/{folder}/{audio_file}" if audio_file else None
            duration = max(int(duration_ms), 0)

            def write() -> bool:
                self._seen(session_id, "practice", now)
                return self._insert_answer(
                    folder, line, record, now, session_id, audio, duration, judge_model, question_en, now
                )

            return self._record(write)

    def next_due(self, now: datetime | None = None, new_per_day: int = NEW_PER_DAY) -> str | None:
        """When the queue next has an item after `now`: the earliest due of the items seen before, or
        the next UTC midnight when new items wait for tomorrow; None when no item would ever come."""
        now = _second(now or self._clock())
        with self._lock:
            connection = self._open_connection()
            seen = connection.execute(
                "SELECT MIN(due) FROM item WHERE state != 'new' AND due > ?", (_iso(now),)
            ).fetchone()[0]
            waiting = connection.execute("SELECT COUNT(*) FROM item WHERE state = 'new'").fetchone()[0]
            times = [_moment(seen)] if seen else []
            if waiting and not self._new_left(now, new_per_day):
                times.append(now.replace(hour=0, minute=0, second=0) + timedelta(days=1))
            return _iso(min(times)) if times else None

    def progress(self, now: datetime | None = None, topics: Iterable[tuple[str, str, Sequence[str]]] = ()) -> dict:
        """What the learner has done, without streaks: minutes spoken this ISO week (UTC) and in all,
        from the durations of the learner's chat turns and practice answers; phrases met; review
        items seen before, in all and due at `now`; corrections by category in the last four ISO
        weeks, newest first; and per topic `(id, name_en, question ids)` its questions answered."""
        now = _second(now or self._clock())
        topics = list(topics)
        week = _week(now)
        with self._lock:
            connection = self._open_connection()
            spoken = connection.execute(
                "SELECT at, duration_ms FROM turn WHERE role = 'user' AND duration_ms > 0"
                " UNION ALL SELECT at, duration_ms FROM answer WHERE duration_ms > 0"
            ).fetchall()
            phrases = connection.execute("SELECT COUNT(*) FROM phrase").fetchone()[0]
            items = connection.execute(
                "SELECT COUNT(*) AS total, COALESCE(SUM(state != 'new'), 0) AS in_review,"
                " COALESCE(SUM(state != 'new' AND due <= ?), 0) AS due FROM item",
                (_iso(now),),
            ).fetchone()
            corrected = connection.execute(
                "SELECT answer.at, correction.category FROM correction JOIN answer ON answer.id = correction.answer_id"
                " WHERE correction.category IN (" + ", ".join("?" * len(CATEGORIES)) + ")",
                CATEGORIES,
            ).fetchall()
            answered = connection.execute(
                "SELECT DISTINCT topic_id, question_id FROM answer WHERE question_id IS NOT NULL"
            ).fetchall()
        total_ms = week_ms = 0
        for row in spoken:
            total_ms += row["duration_ms"]
            if _week(_parsed(row["at"])) == week:
                week_ms += row["duration_ms"]
        weeks = [_week(now - timedelta(weeks=back)) for back in range(PROGRESS_WEEKS)]
        by_week = {key: dict.fromkeys(CATEGORIES, 0) for key in weeks}
        for row in corrected:
            key = _week(_parsed(row["at"]))
            if key in by_week:
                by_week[key][row["category"]] += 1
        done = {(row["topic_id"], row["question_id"]) for row in answered}
        return {
            "minutes": {"week": _minutes(week_ms), "total": _minutes(total_ms)},
            "phrases": phrases,
            "items": {"in_review": items["in_review"], "total": items["total"], "due": items["due"]},
            "corrections_by_week": [{"week": key, **by_week[key]} for key in weeks],
            "topics": [
                {
                    "id": topic_id,
                    "name_en": name_en,
                    "answered": sum((topic_id, question) in done for question in set(questions)),
                    "questions": len(questions),
                }
                for topic_id, name_en, questions in topics
            ],
        }

    # Everything below runs under self._lock.

    def _item_row(self, item_id: int) -> sqlite3.Row | None:
        return self._open_connection().execute("SELECT * FROM item WHERE id = ?", (item_id,)).fetchone()

    def _known_item(self, item_id: int) -> sqlite3.Row:
        row = self._item_row(item_id)
        if row is None:
            raise ValueError(f"there is no review item {item_id}")
        return row

    def _new_left(self, now: datetime, new_per_day: int) -> int:
        """How many new items may still be seen on the UTC day of `now`."""
        start = now.replace(hour=0, minute=0, second=0)
        seen = self._open_connection().execute(
            "SELECT COUNT(*) FROM (SELECT MIN(at) AS first FROM review_log GROUP BY item_id)"
            " WHERE first >= ? AND first < ?",
            (_iso(start), _iso(start + timedelta(days=1))),
        ).fetchone()[0]
        return max(new_per_day - seen, 0)

    def _atomic(self, write: Callable[[], T], failure: str) -> T:
        """What `write` returns, run in one transaction; a database error is a StoreError of
        `failure`, in which `{path}` names the database."""
        connection = self._open_connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            result = write()
            connection.execute("COMMIT")
        except BaseException as exc:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            if isinstance(exc, sqlite3.Error):
                raise StoreError(f"{failure.format(path=self.path)}: {type(exc).__name__}: {exc}") from exc
            raise
        return result

    def _write(self, write: Callable[[], dict]) -> dict:
        return self._atomic(write, "the review was not written to {path}")

    def _record(self, write: Callable[[], T]) -> T:
        return self._atomic(write, "this was not recorded in {path}")

    def _seen(self, session_id: str, mode: str, now: str) -> None:
        """The session's row, made when missing, with `last_seen` now and `mode` among its comma-separated `modes`."""
        connection = self._open_connection()
        connection.execute(
            "INSERT OR IGNORE INTO session (id, started, last_seen, modes) VALUES (?, ?, ?, '')",
            (session_id, now, now),
        )
        modes = connection.execute("SELECT modes FROM session WHERE id = ?", (session_id,)).fetchone()["modes"]
        known = modes.split(",") if modes else []
        if mode and mode not in known:
            known.append(mode)
        connection.execute(
            "UPDATE session SET last_seen = ?, modes = ? WHERE id = ?", (now, ",".join(known), session_id)
        )

    def _log(self, item_id: int, rating: int, now: datetime, mode, heard, confirmed, matched, rated_by) -> int:
        if mode not in REVIEW_MODES:
            raise ValueError(f"a review mode is {' or '.join(REVIEW_MODES)}, not {mode!r}")
        if rated_by not in RATED_BY:
            raise ValueError(f"a rating is by {' or '.join(RATED_BY)}, not {rated_by!r}")
        return self._open_connection().execute(
            "INSERT INTO review_log (item_id, at, rating, mode, heard, confirmed, matched, rated_by)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (item_id, _iso(now), rating, mode, heard, confirmed, int(bool(matched)), rated_by),
        ).lastrowid

    def _cache(self, item_id: int, card: fsrs.Card | None, reps: int, lapses: int, created: str) -> None:
        """Writes the cached columns: `card`'s, or those of an item never reviewed when it is None."""
        if card is None:
            values = ("new", None, None, None, created, None)
        else:
            last = _iso(card.last_review) if card.last_review else None
            values = (card.state, card.step, card.stability, card.difficulty, _iso(card.due), last)
        self._open_connection().execute(
            "UPDATE item SET state = ?, step = ?, stability = ?, difficulty = ?, due = ?, last_review = ?,"
            " reps = ?, lapses = ? WHERE id = ?",
            (*values, reps, lapses, item_id),
        )

    def _reviewed(self, item_id, rating, now, mode, heard, confirmed, matched, rated_by) -> dict:
        item = self._known_item(item_id)
        before = _card(item)
        # Never before the last review, so that the log in time order is the order of the reviews.
        now = max(now, before.last_review or now)
        card = fsrs.review(before, rating, now)
        log_id = self._log(item_id, rating, now, mode, heard, confirmed, matched, rated_by)
        lapses = item["lapses"] + (rating == fsrs.AGAIN and item["state"] == "review")
        self._cache(item_id, card, item["reps"] + 1, lapses, item["created"])
        return {"log_id": log_id, "item_id": item_id, "rating": rating, "state": card.state, "due": _iso(card.due)}

    def _overridden(self, log_id: int, rating: int, now: datetime, item_id: int | None) -> dict:
        connection = self._open_connection()
        row = connection.execute("SELECT * FROM review_log WHERE id = ?", (log_id,)).fetchone()
        if row is None or (item_id is not None and row["item_id"] != item_id):
            raise ValueError(f"there is no review {log_id}" + (f" of item {item_id}" if item_id is not None else ""))
        latest = connection.execute(
            "SELECT id FROM review_log WHERE item_id = ? AND superseded = 0 ORDER BY at DESC, id DESC LIMIT 1",
            (row["item_id"],),
        ).fetchone()
        if row["rated_by"] != "auto" or row["superseded"] or latest["id"] != log_id:
            raise ValueError(f"review {log_id} is not the latest automatic rating of item {row['item_id']}")
        rating = fsrs.checked_rating(rating)
        now = max(now, _moment(row["at"]))
        connection.execute("UPDATE review_log SET superseded = 1 WHERE id = ?", (log_id,))
        new_id = self._log(
            row["item_id"], rating, now, row["mode"], row["heard"], row["confirmed"], row["matched"], "learner"
        )
        item = self._rebuilt(row["item_id"])
        return {
            "log_id": new_id,
            "item_id": row["item_id"],
            "rating": rating,
            "state": item["state"],
            "due": item["due"],
            "superseded": log_id,
        }

    def _rebuilt(self, item_id: int) -> dict:
        item = self._known_item(item_id)
        rows = self._open_connection().execute(
            "SELECT rating, at FROM review_log WHERE item_id = ? AND superseded = 0 ORDER BY at, id", (item_id,)
        ).fetchall()
        card, lapses = None, 0
        for row in rows:
            before = card or fsrs.new_card(_moment(item["created"]))
            lapses += row["rating"] == fsrs.AGAIN and before.state == "review"
            card = fsrs.review(before, row["rating"], _moment(row["at"]))
        self._cache(item_id, card, len(rows), lapses, item["created"])
        return dict(self._item_row(item_id))

    def _open_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise StoreError(f"the learner database {self.path} is not open")
        return self._connection

    def _migrate(self, connection: sqlite3.Connection, path: Path) -> None:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version > len(MIGRATIONS):
            raise StoreError(
                f"the learner database {path} has schema version {version}; this app knows up to {len(MIGRATIONS)}"
            )
        for n in range(version, len(MIGRATIONS)):
            try:
                connection.execute("BEGIN IMMEDIATE")
                MIGRATIONS[n](connection, _iso(self._clock()))
                connection.execute(f"PRAGMA user_version = {int(n + 1)}")
                connection.execute("COMMIT")
            except sqlite3.Error as exc:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise StoreError(
                    f"the learner database {path} could not be brought to schema version {n + 1}:"
                    f" {type(exc).__name__}: {exc}"
                ) from exc

    def _transaction(self, write: Callable[[], int]) -> int:
        return self._atomic(write, "the import into {path} failed")

    def _phrase_entries(self) -> list[dict]:
        """The entries of phrases.json, read directly; [] when it is missing or not a list of phrase entries."""
        path = self.data_dir / PHRASES_FILE
        if _is_link(path) or not path.is_file():
            return []
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(entries, list) or not all(
            isinstance(entry, dict) and PHRASE_FIELDS <= set(entry) for entry in entries
        ):
            return []
        return entries

    def _practice_root(self) -> Path | None:
        """data_dir/practice as it is on disk, or None when it is missing or not a folder of its own there."""
        practice = self.data_dir / PRACTICE_FOLDER
        if _is_link(practice) or not practice.is_dir():
            return None
        try:
            root = practice.resolve()
            if root.parent != self.data_dir.resolve():
                return None
        except OSError:
            return None
        return root

    def _answer_folders(self) -> list[Path]:
        root = self._practice_root()
        if root is None:
            return []
        folders = []
        for child in sorted(root.iterdir()):
            if _skipped(child) or not child.is_dir():
                continue
            resolved = child.resolve()
            if resolved.parent == root and os.path.normcase(resolved.name) == os.path.normcase(child.name):
                folders.append(child)
        return folders

    def _answer_lines(self) -> list[tuple[str, int, dict]]:
        """(folder name, line number from 1, line) of every answer line under practice/."""
        found = []
        for folder in self._answer_folders():
            lines = folder / ANSWERS_FILE
            if _is_link(lines) or not lines.is_file():
                continue
            try:
                text = lines.read_text(encoding="utf-8")
            except (OSError, ValueError):
                continue
            for number, raw in enumerate(text.splitlines(), start=1):
                try:
                    line = json.loads(raw) if raw.strip() else None
                except ValueError:
                    line = None
                if isinstance(line, dict):
                    found.append((folder.name, number, line))
        return found

    def _pre_migration_copy(self) -> None:
        """Copies phrases.json and practice/ to learner/pre-migration once, checked by size."""
        target = self._child(PRE_MIGRATION)
        if target.exists():
            return
        temp = self._child(PRE_MIGRATION + ".tmp")
        read = 0

        def copy(source: str, destination: str) -> str:
            nonlocal read
            read += os.stat(source).st_size
            return shutil.copy2(source, destination)

        def ignore(folder: str, names: list[str]) -> set[str]:
            return {name for name in names if _skipped(Path(folder) / name)}

        try:
            shutil.rmtree(temp, ignore_errors=True)
            temp.mkdir()
            phrases = self.data_dir / PHRASES_FILE
            if phrases.is_file() and not _is_link(phrases):
                copy(str(phrases), str(temp / PHRASES_FILE))
            root = self._practice_root()
            if root is not None:
                shutil.copytree(root, temp / PRACTICE_FOLDER, symlinks=False, ignore=ignore, copy_function=copy)
            copied = _tree_size(temp)
            if copied != read:
                raise StoreError(
                    f"the copy in {temp} holds {copied} bytes of the {read} read; nothing was imported"
                )
            os.replace(temp, target)
        except OSError as exc:
            shutil.rmtree(temp, ignore_errors=True)
            raise StoreError(f"the copy {target} was not made: {type(exc).__name__}: {exc}") from exc
        except StoreError:
            shutil.rmtree(temp, ignore_errors=True)
            raise

    def _insert_phrases(self, entries: list[dict], now: str) -> int:
        connection = self._open_connection()
        rows = 0
        for entry in entries:
            lb = str(entry["lb"]).strip()
            key = phrase_key(lb)
            try:
                values = (
                    key,
                    lb,
                    str(entry["en"]).strip(),
                    str(entry["first_seen"]),
                    str(entry["last_seen"]),
                    int(entry["times_seen"]),
                    json.dumps(entry["sessions"], ensure_ascii=False),
                )
            except (TypeError, ValueError):
                continue
            if not key:
                continue
            added = connection.execute(
                "INSERT OR IGNORE INTO phrase (phrase_key, lb, en, first_seen, last_seen, times_seen, sessions_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                values,
            ).rowcount
            if not added:
                continue
            rows += 1
            phrase = connection.execute("SELECT id, lb, en FROM phrase WHERE phrase_key = ?", (key,)).fetchone()
            self._insert_item("phrase", phrase["id"], phrase["lb"], phrase["en"], None, now)
        connection.execute("INSERT INTO migrated (source, at, rows) VALUES (?, ?, ?)", (PHRASES_FILE, now, rows))
        return rows

    def _insert_answers(
        self, lines: list[tuple[str, int, dict]], questions: dict[str, str], now: str, sources: tuple[str, ...]
    ) -> int:
        """Adds the answer lines the database lacks and fills the empty fields of the corrections it
        has; how many answers were added or filled. Each of `sources` is then recorded as done."""
        connection = self._open_connection()
        rows = 0
        for folder, number, line in lines:
            audio = line.get("audio_file")
            duration = line.get("duration_ms")
            added = self._insert_answer(
                folder,
                number,
                line,
                str(line.get("time") or now),
                None,
                f"{PRACTICE_FOLDER}/{folder}/{audio}" if isinstance(audio, str) and audio else None,
                duration if isinstance(duration, int) and not isinstance(duration, bool) else None,
                line.get("judge_model"),
                questions.get(str(line.get("question_id"))),
                now,
            )
            rows += added or self._fill_correction(folder, number, line)
        for source in sources:
            connection.execute("INSERT INTO migrated (source, at, rows) VALUES (?, ?, ?)", (source, now, rows))
        return rows

    def _insert_answer(
        self,
        folder: str,
        number: int,
        line: dict,
        at: str,
        session_id: str | None,
        audio_file: str | None,
        duration_ms: int | None,
        judge_model: str | None,
        question_en: str | None,
        now: str,
    ) -> bool:
        """Adds the answer `line` kept at line `number` of `folder`, with its correction and, for a
        corrected verdict with a better version, its review item, glossed with `question_en` or else
        the question in Luxembourgish; False when an answer is at that place already."""
        connection = self._open_connection()
        cursor = connection.execute(
            "INSERT OR IGNORE INTO answer (folder, line, session_id, topic_id, question_id, question_lb, at,"
            " raw_transcript, text, help_used, duration_ms, verdict, judge_model, audio_file)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                folder,
                number,
                session_id,
                line.get("topic_id"),
                line.get("question_id"),
                line.get("question_lb"),
                at,
                line.get("raw_transcript"),
                line.get("text"),
                int(bool(line.get("help_used"))),
                duration_ms,
                line.get("verdict"),
                judge_model,
                audio_file,
            ),
        )
        if not cursor.rowcount:
            return False
        correction = line.get("correction")
        if not isinstance(correction, dict):
            return True
        correction_id = connection.execute(
            "INSERT INTO correction (answer_id, verdict, reaction_en, quote, better_lb, why_en, claims_json,"
            " category, label, disputed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                cursor.lastrowid,
                line.get("verdict"),
                correction.get("reaction_en"),
                correction.get("quote"),
                correction.get("better_lb"),
                correction.get("why_en"),
                json.dumps(line.get("claims") or [], ensure_ascii=False),
                *_correction_fields(line),
            ),
        ).lastrowid
        better = str(correction.get("better_lb") or "").strip()
        if line.get("verdict") in CORRECTED_VERDICTS and better:
            gloss = question_en or str(line.get("question_lb") or "")
            quote = str(correction.get("quote") or "").strip() or None
            self._insert_item("correction", correction_id, better, gloss, quote, now)
        return True

    def _fill_correction(self, folder: str, number: int, line: dict) -> bool:
        """Fills the category and label the correction of the answer at that place lacks, and marks it
        disputed, from `line`; whether anything was filled."""
        category, label, disputed = _correction_fields(line)
        return bool(
            self._open_connection()
            .execute(
                "UPDATE correction SET category = COALESCE(category, ?), label = COALESCE(label, ?),"
                " disputed = MAX(disputed, ?)"
                " WHERE answer_id = (SELECT id FROM answer WHERE folder = ? AND line = ?)"
                " AND ((category IS NULL AND ? IS NOT NULL) OR (label IS NULL AND ? IS NOT NULL) OR disputed < ?)",
                (category, label, disputed, folder, number, category, label, disputed),
            )
            .rowcount
        )

    def _insert_item(self, source: str, source_id: int, form: str, gloss: str, hint: str | None, now: str) -> None:
        self._open_connection().execute(
            "INSERT OR IGNORE INTO item (source, source_id, form, gloss, hint, created, state, due)"
            " VALUES (?, ?, ?, ?, ?, ?, 'new', ?)",
            (source, source_id, form, gloss, hint, now, now),
        )

    def _newest_backup_text(self) -> str:
        backups = self.folder / BACKUPS
        try:
            found = sorted(backups.glob("learner-*.sqlite"))
        except OSError:
            found = []
        if not found:
            return f"there is no backup in {backups}"
        return (
            f"the newest backup is {found[-1]}; to restore it, stop the app, copy it over {self.path}"
            f" and delete {DATABASE}-wal and {DATABASE}-shm next to it if they are there"
        )

    def _child(self, name: str) -> Path:
        """A folder of the store's folder, refused when a link or junction is at its place."""
        child = self._folder() / name
        if _is_link(child) or (child.exists() and child.resolve().parent != self._folder()):
            raise StoreError(f"{child} is a link, not a folder of the learner store; nothing is written there")
        return child

    def _folder(self) -> Path:
        """The folder of the store as it is on disk; a StoreError when that is not the folder
        of its name directly inside the data folder."""
        try:
            data_dir = self.data_dir.resolve()
            folder = self.folder.resolve()
        except OSError as exc:
            raise StoreError(f"the folder {self.folder} cannot be looked at: {type(exc).__name__}: {exc}") from exc
        if folder.parent != data_dir or os.path.normcase(folder.name) != os.path.normcase(self.folder.name):
            raise StoreError(
                f"the folder of the learner store, {self.folder}, is not a folder of its own inside {data_dir}"
                f" (it leads to {folder}); nothing is read or written there"
            )
        return folder
