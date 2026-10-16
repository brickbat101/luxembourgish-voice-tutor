"""The Data block's routes: the summary, the export ZIP and the delete of everything the app keeps
about the learner; on temporary data folders only, with the fake engines."""

import inspect
import io
import json
import re
import sqlite3
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import learner_api, review_api
from app.config import ROOT
from app.learner_data import kept_bytes, learner_paths
from app.learner_store import TABLES
from app.main import create_app
from tests.test_learner_store import PHRASES, RECORDED, TYPED, answer_line
from tests.test_practice_store import OFFICE, WUNNT, link

RECORDING = ("speech", b"\x1aE\xdf\xa3 a new recording", "audio/webm")
# In the data folder and not the learner's: the dictionary, its recordings, a settings file.
APP_FILES = {
    "lod/note.txt": b"the dictionary would be here",
    "lod_audio/moien.m4a": b"a dictionary recording",
    "pronunciations.json": b"{}",
    "pronunciation_flags.json": b'{"flags": []}',
}
# What the learner store holds that is the learner's, beyond the rows.
CONTENT_TABLES = ("session", "turn", "phrase", "answer", "correction", "item", "review_log")
OLD_BACKUPS = [f"learner-2026-09-{day:02d}.sqlite" for day in range(1, 10)]


def files_under(folder: Path) -> dict[str, bytes]:
    """Every file under `folder` by its path relative to it, links not followed."""
    found = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and not path.is_symlink():
            found[path.relative_to(folder).as_posix()] = path.read_bytes()
    return found


def content_counts(path: Path) -> dict[str, int]:
    with closing(sqlite3.connect(path)) as connection:
        return {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in CONTENT_TABLES}


def seed(data_dir: Path) -> None:
    """Phrases and two older talks to import, a corrupt and a half-written phrases file, a
    half-written recording, nine old backups, and the app's own files."""
    (data_dir / "practice" / RECORDED).mkdir(parents=True)
    (data_dir / "practice" / TYPED).mkdir()
    (data_dir / "phrases.json").write_text(json.dumps(PHRASES, ensure_ascii=False, indent=2), encoding="utf-8")
    (data_dir / "phrases.json.corrupt").write_text("[{not json", encoding="utf-8")
    (data_dir / "phrases.json.tmp").write_text("[", encoding="utf-8")
    (data_dir / "practice" / RECORDED / "answers.jsonl").write_text(
        answer_line(WUNNT, "2026-10-13T12:00:05+02:00", "1.webm"), encoding="utf-8"
    )
    (data_dir / "practice" / RECORDED / "1.webm").write_bytes(b"\x1aE\xdf\xa3 a recording")
    (data_dir / "practice" / RECORDED / "2.webm.tmp").write_bytes(b"half")
    (data_dir / "practice" / TYPED / "answers.jsonl").write_text(
        answer_line(OFFICE, "2026-10-13T12:10:00+02:00", None), encoding="utf-8"
    )
    backups = data_dir / "learner" / "backups"
    backups.mkdir(parents=True)
    for name in OLD_BACKUPS:
        with closing(sqlite3.connect(backups / name)) as connection:
            connection.execute("CREATE TABLE phrase (lb TEXT)")
            connection.execute("INSERT INTO phrase VALUES ('Moien!')")
            connection.commit()
    for name, data in APP_FILES.items():
        (data_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (data_dir / name).write_bytes(data)


@pytest.fixture
def data_dir(settings):
    seed(settings.data_dir)
    return settings.data_dir


@pytest.fixture
def client(settings, data_dir):
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def state(client):
    return client.app.state


@pytest.fixture
def used(client, state):
    """The app after a chat turn, a spoken and a typed practice answer and one review: the session id."""
    session_id = client.post("/api/session").json()["session_id"]
    assert client.post("/api/chat", json={"session_id": session_id, "text": "Hello"}).status_code == 200
    client.post("/api/practice/start", data={"session_id": session_id, "topic_id": "wunnen"})
    data = {"session_id": session_id, "text": "Ech wunnen zu Esch.", "duration_ms": "2100"}
    assert client.post("/api/practice/answer", data=data, files={"audio": RECORDING}).status_code == 200
    data = {"session_id": session_id, "text": "Ech hunn e Gaart."}
    assert client.post("/api/practice/answer", data=data).status_code == 200
    item = client.get("/api/review/next").json()["item"]
    rated = client.post("/api/review/rate", data={"item_id": item["id"], "rating": "3", "text": "Moien"})
    assert rated.status_code == 200
    return session_id


def exported(client) -> tuple[zipfile.ZipFile, str]:
    response = client.get("/api/learner/export")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    return zipfile.ZipFile(io.BytesIO(response.content)), response.headers["content-disposition"]


def exported_files(names: list[str]) -> set[str]:
    """The data-folder paths the ZIP entries `names` stand for; learner.sqlite stands for the database."""
    paths = set()
    for name in names:
        if name.startswith(("originals/", "audio/")):
            paths.add(name.split("/", 1)[1])
        elif name == "learner.sqlite":
            paths.update(f"learner/learner.sqlite{end}" for end in ("", "-wal", "-shm"))
    return paths


# --- export ---


def test_the_export_holds_every_row_a_checked_copy_of_the_database_the_recordings_and_the_originals(
    client, state, used, data_dir, tmp_path
):
    counts = state.learner.counts()
    archive, disposition = exported(client)
    names = archive.namelist()
    document = json.loads(archive.read("learner.json"))

    today = document["exported_at"][:10]
    assert today == datetime.now(timezone.utc).date().isoformat()
    assert disposition == f'attachment; filename="luxembourgish-tutor-export-{today}.zip"'
    assert document["schema_version"] == 1
    assert {table: len(rows) for table, rows in document["tables"].items()} == counts
    assert set(document["tables"]) == set(TABLES)
    assert counts["answer"] == 4 and counts["review_log"] == 1 and counts["turn"] == 2
    # The raw and the confirmed text of every answer, and every review.
    assert {row["text"] for row in document["tables"]["answer"]} >= {"Ech wunnen zu Esch.", "Ech hunn e Gaart."}
    assert document["tables"]["review_log"][0]["confirmed"] == "Moien"

    copy = tmp_path / "copy.sqlite"
    copy.write_bytes(archive.read("learner.sqlite"))
    with closing(sqlite3.connect(copy)) as connection:
        assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert {t: connection.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES} == counts

    recordings = [row["audio_file"] for row in document["tables"]["answer"] if row["audio_file"]]
    assert len(recordings) == 2
    for audio_file in recordings:
        assert archive.read(f"audio/{audio_file}") == (data_dir / audio_file).read_bytes()
    folders = sorted(path.name for path in (data_dir / "practice").iterdir())
    assert len(folders) == 3
    for folder in folders:
        name = f"practice/{folder}/answers.jsonl"
        assert archive.read(f"originals/{name}") == (data_dir / name).read_bytes()
    for name in ("phrases.json", "phrases.json.corrupt"):
        assert archive.read(f"originals/{name}") == (data_dir / name).read_bytes()
    # Copies of these and half-written files are not in it.
    assert not [name for name in names if "backups" in name or "pre-migration" in name or name.endswith(".tmp")]
    assert archive.getinfo("learner.sqlite").compress_type == zipfile.ZIP_STORED
    assert archive.getinfo(f"audio/{recordings[0]}").compress_type == zipfile.ZIP_STORED
    assert archive.getinfo("learner.json").compress_type == zipfile.ZIP_DEFLATED


def test_a_recording_the_store_did_not_record_is_exported_too(client, state, data_dir):
    stray = data_dir / "practice" / TYPED / "7.webm"
    stray.write_bytes(b"a recording without its row")

    archive, _ = exported(client)

    assert archive.read(f"audio/practice/{TYPED}/7.webm") == stray.read_bytes()


def test_without_the_learner_store_summary_and_export_answer_503(client, state, monkeypatch):
    monkeypatch.setattr(state, "learner", None)
    monkeypatch.setattr(state, "learner_detail", "it fails its check")

    for path in ("/api/learner/summary", "/api/learner/export"):
        response = client.get(path)
        assert response.status_code == 503
        assert "the learner store is not open: it fails its check" in response.json()["error"]


# --- the summary ---


def test_the_summary_counts_phrases_answers_recordings_and_items_and_the_bytes_of_every_learner_file(
    client, state, used, data_dir
):
    counts = state.learner.counts()

    summary = client.get("/api/learner/summary").json()

    assert summary == {
        "phrases": counts["phrase"],
        "answers": 4,
        "recordings": 2,
        "items": counts["item"],
        "bytes": kept_bytes(data_dir),
        "folders": [str(data_dir / "learner"), str(data_dir / "practice"), str(data_dir / "phrases.json")],
    }
    learner_files = {name: data for name, data in files_under(data_dir).items() if name not in APP_FILES}
    assert summary["bytes"] == sum(len(data) for data in learner_files.values())


# --- delete ---


def test_every_learner_file_is_exported_or_a_known_copy_and_after_delete_none_is_left(client, state, used, data_dir):
    before = files_under(data_dir)
    archive, _ = exported(client)
    in_export = exported_files(archive.namelist())

    assert (data_dir / "learner" / "learner.sqlite-wal").is_file()
    assert any(name.startswith("learner/pre-migration/") for name in before)
    learner_files = sorted(name for name in before if name not in APP_FILES)
    for name in learner_files:
        copy = name.startswith(("learner/backups/", "learner/pre-migration/")) or name.endswith(".tmp")
        assert name in in_export or copy, f"{name} is the learner's and is not in the export"

    response = client.delete("/api/learner")

    assert (response.status_code, response.json()) == (200, {"deleted": True, "left": []})
    after = files_under(data_dir)
    # Decision 0780dd2a95e9: what is there now is the app's files and the empty store opened again.
    today = datetime.now(timezone.utc).date().isoformat()
    fresh = {f"learner/learner.sqlite{end}" for end in ("", "-wal", "-shm")} | {
        f"learner/backups/learner-{today}.sqlite"
    }
    assert set(after) - set(APP_FILES) <= fresh
    for name in learner_files:
        assert name not in after or name in fresh, f"{name} was neither deleted nor made afresh"
    for name, data in APP_FILES.items():
        assert after[name] == data
    assert not (data_dir / "practice").exists()
    assert not [path for path in data_dir.iterdir() if path.name.startswith("phrases.json")]
    assert content_counts(state.learner.path) == dict.fromkeys(CONTENT_TABLES, 0)
    assert content_counts(data_dir / "learner" / "backups" / f"learner-{today}.sqlite") == dict.fromkeys(
        CONTENT_TABLES, 0
    )


def test_after_delete_the_app_records_again_in_an_empty_store_and_the_routes_answer(
    client, state, used, data_dir
):
    old = state.learner

    assert client.delete("/api/learner").status_code == 200

    assert state.learner is not old and state.orchestrator.learner is state.learner
    summary = client.get("/api/learner/summary").json()
    assert (summary["phrases"], summary["answers"], summary["recordings"], summary["items"]) == (0, 0, 0, 0)
    review = client.get("/api/review/next")
    assert (review.status_code, review.json()["item"]) == (200, None)
    assert client.get("/api/learner/progress").status_code == 200
    # The open talk goes on: its next answer is line 1 of a new folder, with no old row in the way.
    data = {"session_id": used, "text": "Ech wunnen zu Esch."}
    practice = client.post("/api/practice/answer", data=data, files={"audio": RECORDING})
    assert practice.status_code == 200 and practice.json()["kept"]["line"] == 1
    chat = client.post("/api/chat", json={"session_id": used, "text": "Hello"})
    assert chat.status_code == 200
    counts = state.learner.counts()
    assert (counts["answer"], counts["phrase"], counts["turn"]) == (1, 1, 2)
    assert json.loads((data_dir / "phrases.json").read_text(encoding="utf-8"))[0]["times_seen"] == 1


def test_a_second_delete_is_fine(client, used):
    first = client.delete("/api/learner")
    second = client.delete("/api/learner")

    assert (first.status_code, second.status_code) == (200, 200)
    assert second.json() == {"deleted": True, "left": []}


def test_a_learner_folder_that_is_a_link_is_refused_and_what_it_leads_to_is_untouched(settings, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("mine", encoding="utf-8")
    settings.data_dir.mkdir(parents=True)
    (settings.data_dir / "phrases.json").write_text(json.dumps(PHRASES), encoding="utf-8")
    link(settings.data_dir / "learner", outside)

    with TestClient(create_app(settings)) as client:
        assert client.app.state.learner is None
        response = client.delete("/api/learner")

    assert response.status_code == 500
    error = response.json()["error"]
    assert error.startswith(f"not everything was deleted; still on this PC: {settings.data_dir / 'learner'}.")
    assert "is not a folder of its own inside" in error
    assert sorted(path.name for path in outside.iterdir()) == ["keep.txt"]
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "mine"
    # The other parts are deleted all the same.
    assert not (settings.data_dir / "phrases.json").exists()


def test_a_file_that_cannot_be_deleted_is_named_and_the_store_is_opened_again(client, state, used, monkeypatch):
    def refuse() -> None:
        raise PermissionError(13, "Permission denied", str(state.phrases.path))

    monkeypatch.setattr(state.phrases, "delete", refuse)

    response = client.delete("/api/learner")

    assert response.status_code == 500
    body = response.json()
    assert str(state.phrases.path) in body["error"] and "PermissionError" in body["error"]
    assert body["fix"] == "close any program that has these files open, then press Delete everything again"
    assert state.learner is not None and client.get("/api/review/next").status_code == 200


def test_the_learner_paths_are_the_store_folder_the_practice_folder_and_the_phrases_files(tmp_path):
    assert [path.relative_to(tmp_path).as_posix() for path in learner_paths(tmp_path)] == [
        "learner",
        "practice",
        "phrases.json",
        "phrases.json.corrupt",
        "phrases.json.tmp",
    ]


def test_the_readme_documents_the_store_export_and_delete_and_every_learner_and_review_route():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    text = " ".join(readme.split())
    section = " ".join(readme.split("\n## Export and delete\n")[1].split("\n## ")[0].split())

    for router in (learner_api.router, review_api.router):
        for route in router.routes:
            (method,) = route.methods
            assert f"| `{method} {route.path}` |" in readme, route.path
    assert re.findall(r"`(\w+) /api/practice/saved`", readme) == ["GET", "GET"]
    for said in (
        "`learner.sqlite-wal`, `learner.sqlite-shm`, the 7 backups and the pre-migration copy",
        "It is the last copy.",
        "Copies outside the app are not deleted",
        "OneDrive, Windows File History",
        "The backups and the pre-migration copy are not in the ZIP",
        "`audio/practice/<folder>/<n>.webm`",
        "the app opens a new, empty learner store",
    ):
        assert said in section, said
    for said in ("FSRS-6 with its 21 default weights", "py-fsrs 6.3.2", "no fuzz", "The category is a heuristic"):
        assert said in text, said


def test_the_learner_routes_are_plain_functions():
    routes = [(route.path, sorted(route.methods)) for route in learner_api.router.routes]

    assert routes == [
        ("/api/learner/progress", ["GET"]),
        ("/api/learner/summary", ["GET"]),
        ("/api/learner/export", ["GET"]),
        ("/api/learner", ["DELETE"]),
    ]
    assert not any(inspect.iscoroutinefunction(route.endpoint) for route in learner_api.router.routes)
