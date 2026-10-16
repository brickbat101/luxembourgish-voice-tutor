"""The learner store and its first-launch import, on temporary data folders only."""

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import learner_store
from app.learner_store import LearnerStore
from app.main import create_app
from app.practice_store import StoreError
from tests.test_practice_store import OFFICE, WUNNT, files_under, link

NOW = datetime(2026, 10, 14, 9, 30, 0, tzinfo=timezone.utc)
RECORDED = "2026-10-13-0a1b2c3d"
TYPED = "2026-10-13-ffeeddcc"
PHRASES = [
    {
        "lb": "Moien!",
        "en": "Hello!",
        "first_seen": "2026-10-12T08:00:00+00:00",
        "last_seen": "2026-10-13T08:00:00+00:00",
        "times_seen": 3,
        "sessions": ["s1", "s2"],
    },
    {
        "lb": "Wéi geet et?",
        "en": "How are you?",
        "first_seen": "2026-10-12T08:01:00+00:00",
        "last_seen": "2026-10-12T08:01:00+00:00",
        "times_seen": 1,
        "sessions": ["s1"],
    },
    {
        "lb": "Merci villmools",
        "en": "Thank you very much",
        "first_seen": "2026-10-12T08:02:00+00:00",
        "last_seen": "2026-10-12T08:02:00+00:00",
        "times_seen": 2,
        "sessions": ["s1"],
    },
]
QUESTIONS = {"wunnen-1": "Where do you live?", "wunnen-2": "What is your home like?"}


class Clock:
    def __init__(self, now: datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def answer_line(answer: dict, time: str, audio_file: str | None) -> str:
    return json.dumps({"time": time, **answer, "audio_file": audio_file}, ensure_ascii=False) + "\n"


@pytest.fixture
def data_dir(tmp_path):
    """Three phrases and two talks: one answer with a recording and a correction, one typed without."""
    folder = tmp_path / "data"
    (folder / "practice" / RECORDED).mkdir(parents=True)
    (folder / "practice" / TYPED).mkdir()
    (folder / "phrases.json").write_text(json.dumps(PHRASES, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "practice" / RECORDED / "answers.jsonl").write_text(
        answer_line(WUNNT, "2026-10-13T12:00:05+02:00", "1.webm"), encoding="utf-8"
    )
    (folder / "practice" / RECORDED / "1.webm").write_bytes(b"\x1aE\xdf\xa3 a recording")
    (folder / "practice" / TYPED / "answers.jsonl").write_text(
        answer_line(OFFICE, "2026-10-13T12:10:00+02:00", None), encoding="utf-8"
    )
    return folder


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(data_dir, clock):
    store = LearnerStore(data_dir / "learner", data_dir, clock)
    yield store
    store.close()


def originals(data_dir: Path) -> dict[str, bytes]:
    files = {"phrases.json": (data_dir / "phrases.json").read_bytes()}
    for name in files_under(data_dir / "practice"):
        files[f"practice/{name}"] = (data_dir / "practice" / name).read_bytes()
    return files


def rows(store: LearnerStore, sql: str) -> list[dict]:
    with closing(sqlite3.connect(store.path)) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(sql)]


def test_the_first_open_makes_schema_version_1_in_wal_with_the_learner_row(store):
    store.open()

    with closing(sqlite3.connect(store.path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        columns = {row[1] for row in connection.execute("PRAGMA table_info(review_log)")}
        correction = {row[1]: row for row in connection.execute("PRAGMA table_info(correction)")}
    assert store.path == store.folder / "learner.sqlite"
    assert set(store.counts()) == set(learner_store.TABLES)
    assert rows(store, "SELECT id, created, settings_json FROM learner") == [
        {"id": 1, "created": "2026-10-14T09:30:00+00:00", "settings_json": "{}"}
    ]
    assert "superseded" in columns
    assert {"category", "label", "disputed"} <= set(correction)
    assert correction["disputed"][3] == 1  # NOT NULL


def test_the_import_copies_the_phrases_and_answers_into_rows_and_items(store, data_dir):
    store.open()

    added = store.migrate_files(QUESTIONS)

    counts = store.counts()
    assert added == {"phrases.json": 3, "practice": 2}
    assert {table: counts[table] for table in ("phrase", "answer", "correction", "item", "migrated")} == {
        "phrase": 3,
        "answer": 2,
        "correction": 1,
        "item": 4,
        "migrated": 3,  # phrases.json, practice and practice-fields, which a first import needs no more
    }
    assert rows(store, "SELECT source, COUNT(*) AS n FROM item GROUP BY source ORDER BY source") == [
        {"source": "correction", "n": 1},
        {"source": "phrase", "n": 3},
    ]
    moien = rows(store, "SELECT * FROM phrase WHERE phrase_key = 'moien'")[0]
    assert (moien["lb"], moien["en"], moien["times_seen"]) == ("Moien!", "Hello!", 3)
    assert json.loads(moien["sessions_json"]) == ["s1", "s2"]
    answers = rows(store, "SELECT folder, line, session_id, audio_file, help_used, verdict FROM answer ORDER BY folder")
    assert answers == [
        {"folder": RECORDED, "line": 1, "session_id": None, "audio_file": f"practice/{RECORDED}/1.webm",
         "help_used": 0, "verdict": "fix"},
        {"folder": TYPED, "line": 1, "session_id": None, "audio_file": None, "help_used": 1, "verdict": None},
    ]
    correction = rows(store, "SELECT * FROM correction")[0]
    assert (correction["better_lb"], correction["category"], correction["label"], correction["disputed"]) == (
        "Ech wunnen zu Esch.",
        None,
        None,
        0,
    )
    item = rows(store, "SELECT * FROM item WHERE source = 'correction'")[0]
    assert (item["form"], item["gloss"], item["hint"], item["state"]) == (
        "Ech wunnen zu Esch.",
        "Where do you live?",
        "Ech wunnt",
        "new",
    )
    assert item["due"] == item["created"] == "2026-10-14T09:30:00+00:00"


def test_without_the_practice_content_a_correction_item_is_glossed_with_the_question(store):
    store.open()
    store.migrate_files()

    assert rows(store, "SELECT gloss FROM item WHERE source = 'correction'") == [{"gloss": "Wou wunnt Dir?"}]


def test_a_dispute_marks_the_correction_of_the_answer_at_its_folder_and_line_and_nothing_else(store, data_dir):
    store.open()
    store.migrate_files(QUESTIONS)
    before = originals(data_dir)

    marked = [store.mark_disputed(folder, line) for folder, line in ((TYPED, 1), (RECORDED, 2), ("nowhere", 1))]
    first = store.mark_disputed(RECORDED, 1)
    again = store.mark_disputed(RECORDED, 1)

    # The typed answer has no correction; there is no second line and no such folder.
    assert (marked, first, again) == ([False, False, False], True, True)
    assert rows(store, "SELECT disputed FROM correction") == [{"disputed": 1}]
    assert originals(data_dir) == before


def test_a_closed_store_refuses_a_dispute(store):
    with pytest.raises(StoreError, match="is not open"):
        store.mark_disputed(RECORDED, 1)


def test_a_second_start_adds_nothing_and_makes_no_second_copy(data_dir, clock):
    first = LearnerStore(data_dir / "learner", data_dir, clock)
    first.open()
    first.migrate_files(QUESTIONS)
    counts = first.counts()
    first.close()
    copy_stat = (data_dir / "learner" / "pre-migration" / "phrases.json").stat().st_mtime_ns

    second = LearnerStore(data_dir / "learner", data_dir, clock)
    second.open()
    try:
        assert second.migrate_files(QUESTIONS) == {}
        assert second.counts() == counts
    finally:
        second.close()
    assert (data_dir / "learner" / "pre-migration" / "phrases.json").stat().st_mtime_ns == copy_stat


def test_the_originals_stay_byte_identical_and_the_pre_migration_copy_holds_the_same_files(store, data_dir):
    before = originals(data_dir)
    store.open()

    store.migrate_files(QUESTIONS)

    copy = data_dir / "learner" / "pre-migration"
    assert originals(data_dir) == before
    assert originals(copy) == before
    assert not (data_dir / "learner" / "pre-migration.tmp").exists()


def test_a_copy_that_does_not_hold_what_was_read_stops_the_import(store, data_dir, monkeypatch):
    monkeypatch.setattr(learner_store, "_tree_size", lambda root: 1)
    store.open()

    with pytest.raises(StoreError, match="nothing was imported"):
        store.migrate_files(QUESTIONS)

    assert store.counts()["phrase"] == store.counts()["answer"] == store.counts()["migrated"] == 0
    assert sorted(path.name for path in (data_dir / "learner").iterdir()) == ["learner.sqlite"] + [
        name for name in ("learner.sqlite-shm", "learner.sqlite-wal") if (data_dir / "learner" / name).exists()
    ]


def test_nothing_to_import_records_zero_rows_and_makes_no_copy(tmp_path, clock):
    data_dir = tmp_path / "data"
    store = LearnerStore(data_dir / "learner", data_dir, clock)
    store.open()
    try:
        assert store.migrate_files() == {"phrases.json": 0, "practice": 0}
        assert rows(store, "SELECT source, rows FROM migrated ORDER BY source") == [
            {"source": "phrases.json", "rows": 0},
            {"source": "practice", "rows": 0},
            {"source": "practice-fields", "rows": 0},
        ]
    finally:
        store.close()
    assert not (data_dir / "learner" / "pre-migration").exists()


def test_a_phrases_file_that_is_not_json_is_left_in_place_and_records_zero(store, data_dir):
    (data_dir / "phrases.json").write_text("[{not json", encoding="utf-8")
    store.open()

    added = store.migrate_files(QUESTIONS)

    assert added == {"phrases.json": 0, "practice": 2}
    assert (data_dir / "phrases.json").read_text(encoding="utf-8") == "[{not json"
    assert not (data_dir / "phrases.json.corrupt").exists()
    assert rows(store, "SELECT rows FROM migrated WHERE source = 'phrases.json'") == [{"rows": 0}]


def test_a_link_at_the_place_of_the_learner_folder_is_refused(data_dir, tmp_path, clock):
    outside = tmp_path / "outside"
    outside.mkdir()
    link(data_dir / "learner", outside)
    store = LearnerStore(data_dir / "learner", data_dir, clock)

    with pytest.raises(StoreError, match="not a folder of its own"):
        store.open()

    assert list(outside.iterdir()) == []


def test_a_link_inside_practice_is_neither_imported_nor_copied(store, data_dir, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "answers.jsonl").write_text(answer_line(OFFICE, "2026-10-13T13:00:00+02:00", None), encoding="utf-8")
    link(data_dir / "practice" / "2026-10-13-elsewhere", outside)
    (data_dir / "practice" / TYPED / "answers.jsonl.tmp").write_text("half", encoding="utf-8")
    store.open()

    store.migrate_files(QUESTIONS)

    assert store.counts()["answer"] == 2
    copied = files_under(data_dir / "learner" / "pre-migration" / "practice")
    assert copied == [f"{RECORDED}/1.webm", f"{RECORDED}/answers.jsonl", f"{TYPED}/answers.jsonl"]


def test_the_daily_backup_is_made_once_a_day_and_7_are_kept(store, clock):
    store.open()
    store.migrate_files(QUESTIONS)
    backups = store.folder / "backups"

    made = store.backup_daily()
    assert made == backups / "learner-2026-10-14.sqlite"
    assert store.backup_daily() is None
    with closing(sqlite3.connect(made)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM phrase").fetchone()[0] == 3
    for day in range(1, 9):
        clock.now = NOW + timedelta(days=day)
        store.backup_daily()

    assert sorted(path.name for path in backups.iterdir()) == [
        f"learner-2026-10-{day}.sqlite" for day in range(16, 23)
    ]


def test_a_file_that_is_not_a_database_is_refused_left_alone_and_the_newest_backup_named(data_dir, clock):
    folder = data_dir / "learner"
    (folder / "backups").mkdir(parents=True)
    (folder / "backups" / "learner-2026-10-12.sqlite").write_bytes(b"older")
    (folder / "backups" / "learner-2026-10-13.sqlite").write_bytes(b"newer")
    junk = b"this is not a database, " * 200
    (folder / "learner.sqlite").write_bytes(junk)
    store = LearnerStore(folder, data_dir, clock)

    with pytest.raises(StoreError) as caught:
        store.open()

    assert str(folder / "learner.sqlite") in str(caught.value)
    assert str(folder / "backups" / "learner-2026-10-13.sqlite") in str(caught.value)
    assert (folder / "learner.sqlite").read_bytes() == junk
    assert sorted(path.name for path in folder.iterdir()) == ["backups", "learner.sqlite"]
    store.close()


def test_a_database_of_a_newer_schema_is_refused(data_dir, clock):
    (data_dir / "learner").mkdir()
    with closing(sqlite3.connect(data_dir / "learner" / "learner.sqlite")) as connection:
        connection.execute("PRAGMA user_version = 9")
    store = LearnerStore(data_dir / "learner", data_dir, clock)

    with pytest.raises(StoreError, match="schema version 9"):
        store.open()


def test_close_is_idempotent_and_a_closed_store_refuses_work(store):
    store.open()
    store.close()
    store.close()

    with pytest.raises(StoreError, match="not open"):
        store.counts()


def test_the_app_on_an_empty_data_folder_reports_the_learner_store_ready_and_closes_it(settings):
    with TestClient(create_app(settings)) as client:
        learner = client.get("/api/health").json()["learner"]

    path = settings.data_dir / "learner" / "learner.sqlite"
    assert learner == {"ready": True, "detail": "0 phrases, 0 answers, 0 review items", "path": str(path)}
    assert (settings.data_dir / "learner" / "backups").is_dir()
    path.unlink()  # on Windows only a closed database can be removed


def test_the_app_imports_the_data_folder_on_start(settings):
    settings.data_dir.mkdir(parents=True)
    (settings.data_dir / "phrases.json").write_text(json.dumps(PHRASES), encoding="utf-8")

    with TestClient(create_app(settings)) as client:
        learner = client.get("/api/health").json()["learner"]
        counts = client.app.state.learner.counts()

    assert learner["ready"] is True
    assert learner["detail"] == "3 phrases, 0 answers, 3 review items"
    assert counts["phrase"] == 3


def test_the_app_runs_without_a_learner_store_that_fails_its_check(settings):
    folder = settings.data_dir / "learner"
    folder.mkdir(parents=True)
    junk = b"not a database" * 100
    (folder / "learner.sqlite").write_bytes(junk)

    with TestClient(create_app(settings)) as client:
        learner = client.get("/api/health").json()["learner"]
        session_id = client.post("/api/session").json()["session_id"]
        chat = client.post("/api/chat", json={"session_id": session_id, "text": "Hello"})
        assert client.app.state.learner is None

    assert learner["ready"] is False
    assert str(folder / "learner.sqlite") in learner["detail"]
    assert "there is no backup" in learner["detail"]
    assert learner["path"] == str(folder / "learner.sqlite")
    assert chat.status_code == 200
    assert (folder / "learner.sqlite").read_bytes() == junk
