"""The store of confirmed practice answers, on temporary folders only."""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app import practice_store
from app.practice_store import ANSWERS_FILE, PracticeStore, StoreError
from tests.audio import make_noise_wav

SESSION = "0a1b2c3d4e5f60718293a4b5c6d7e8f9"
OTHER_SESSION = "ffeeddccbbaa99887766554433221100"
NOON = datetime(2026, 10, 13, 12, 0, 5, tzinfo=timezone(timedelta(hours=2)))
FOLDER = "2026-10-13-0a1b2c3d"
WUNNT = {
    "topic_id": "wunnen",
    "question_id": "wunnen-1",
    "question_lb": "Wou wunnt Dir?",
    "raw_transcript": "Ech wunnt zu Äsch.",
    "text": "Ech wunnt zu Esch.",
    "help_used": False,
    "verdict": "fix",
    "correction": {
        "reaction_en": "Esch, good.",
        "quote": "Ech wunnt",
        "better_lb": "Ech wunnen zu Esch.",
        "why_en": "with ech the verb is wunnen, so:",
    },
    "claims": [{"form": "wunnen", "lemma": "wunnen", "verified": True}],
}
OFFICE = {
    "topic_id": "wunnen",
    "question_id": "wunnen-2",
    "question_lb": "Wéi ass Är Wunneng?",
    "raw_transcript": "",
    "text": "Ech schaffen an engem Büro.",
    "help_used": True,
    "verdict": None,
    "correction": None,
    "claims": [],
}


class Clock:
    def __init__(self, now: datetime = NOON) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def data_dir(tmp_path):
    """A data folder that holds what the store must leave alone."""
    folder = tmp_path / "data"
    (folder / "lod").mkdir(parents=True)
    (folder / "lod" / "lod.sqlite").write_bytes(b"the dictionary")
    (folder / "phrases.json").write_text("[]", encoding="utf-8")
    return folder


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(data_dir, clock):
    return PracticeStore(data_dir / "practice", data_dir, clock)


@pytest.fixture
def outside(tmp_path):
    """A folder beside the data folder, with a file nobody may touch."""
    folder = tmp_path / "outside"
    folder.mkdir()
    (folder / "keep.txt").write_text("mine", encoding="utf-8")
    return folder


def link(at: Path, target: Path) -> None:
    """Makes `at` lead to the folder `target`: a junction on Windows, which needs no rights, else a symlink."""
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(target), str(at))
    else:
        at.symlink_to(target, target_is_directory=True)


def files_under(folder: Path) -> list[str]:
    return sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file())


def lines_of(folder: Path) -> list[dict]:
    return [json.loads(line) for line in (folder / ANSWERS_FILE).read_text(encoding="utf-8").splitlines()]


def untouched(data_dir: Path) -> bool:
    return files_under(data_dir / "lod") == ["lod.sqlite"] and (data_dir / "phrases.json").read_text() == "[]"


# --- save ---


def test_two_answers_with_a_recording_are_two_lines_and_two_wav_files(store, data_dir, clock):
    first, second = make_noise_wav(0.5), make_noise_wav(1.0)

    where = store.save(SESSION, WUNNT, first, "audio/wav")
    clock.now = NOON + timedelta(seconds=40)
    then = store.save(SESSION, OFFICE, second, "audio/x-wav")

    folder = data_dir / "practice" / FOLDER
    assert (where, then) == ({"folder": FOLDER, "line": 1}, {"folder": FOLDER, "line": 2})
    assert files_under(data_dir / "practice") == [f"{FOLDER}/1.wav", f"{FOLDER}/2.wav", f"{FOLDER}/{ANSWERS_FILE}"]
    assert lines_of(folder) == [
        {"time": "2026-10-13T12:00:05+02:00", **WUNNT, "audio_file": "1.wav"},
        {"time": "2026-10-13T12:00:45+02:00", **OFFICE, "audio_file": "2.wav"},
    ]
    assert list(lines_of(folder)[0]) == ["time", *WUNNT, "audio_file"]
    assert ((folder / "1.wav").read_bytes(), (folder / "2.wav").read_bytes()) == (first, second)
    # As written, not escaped: the file is read by a person too.
    assert "Äsch" in (folder / ANSWERS_FILE).read_text(encoding="utf-8")
    assert untouched(data_dir)


@pytest.mark.parametrize("audio", [None, b""])
def test_an_answer_without_a_recording_has_no_audio_file(store, data_dir, audio):
    store.save(SESSION, OFFICE, audio, "audio/webm")

    assert files_under(data_dir / "practice") == [f"{FOLDER}/{ANSWERS_FILE}"]
    assert lines_of(data_dir / "practice" / FOLDER) == [
        {"time": "2026-10-13T12:00:05+02:00", **OFFICE, "audio_file": None}
    ]


@pytest.mark.parametrize(
    ("mime", "suffix"),
    [("audio/webm;codecs=opus", ".webm"), ("audio/mp4", ".m4a"), ("audio/ogg", ".ogg"), ("application/zip", ".audio")],
)
def test_a_recording_is_kept_as_it_came_under_the_suffix_of_its_type(store, data_dir, mime, suffix):
    store.save(SESSION, WUNNT, b"\x1aE\xdf\xa3 as it came", mime)

    folder = data_dir / "practice" / FOLDER
    assert (folder / f"1{suffix}").read_bytes() == b"\x1aE\xdf\xa3 as it came"
    assert lines_of(folder)[0]["audio_file"] == f"1{suffix}"


def test_the_number_of_a_recording_is_the_number_of_its_line(store, data_dir):
    store.save(SESSION, OFFICE)
    store.save(SESSION, WUNNT, b"second", "audio/webm")
    store.save(SESSION, OFFICE)
    store.save(SESSION, WUNNT, b"fourth", "audio/webm")

    folder = data_dir / "practice" / FOLDER
    assert [line["audio_file"] for line in lines_of(folder)] == [None, "2.webm", None, "4.webm"]
    assert files_under(folder) == ["2.webm", "4.webm", ANSWERS_FILE]


def test_a_session_and_a_day_have_a_folder_of_their_own(store, data_dir, clock):
    store.save(SESSION, WUNNT)
    store.save(OTHER_SESSION, OFFICE)
    clock.now = NOON + timedelta(days=1)
    store.save(SESSION, OFFICE)

    assert sorted(path.name for path in (data_dir / "practice").iterdir()) == [
        "2026-10-13-0a1b2c3d",
        "2026-10-13-ffeeddcc",
        "2026-10-14-0a1b2c3d",
    ]


@pytest.mark.parametrize(
    ("session_id", "folder"),
    [("../../outside", "2026-10-13-outside"), ("..\\..\\x:y", "2026-10-13-xy"), ("../..", "2026-10-13-session")],
)
def test_a_session_id_cannot_name_a_folder_outside_the_store(store, data_dir, outside, session_id, folder):
    assert store.save(session_id, WUNNT, b"audio", "audio/wav") == {"folder": folder, "line": 1}

    assert files_under(data_dir / "practice") == [f"{folder}/1.wav", f"{folder}/{ANSWERS_FILE}"]
    assert files_under(outside) == ["keep.txt"]


def test_a_write_that_fails_is_a_store_error_that_names_the_folder_and_leaves_what_was_kept(
    store, data_dir, monkeypatch
):
    store.save(SESSION, WUNNT, b"first", "audio/wav")
    replace = os.replace

    def lines_cannot_be_written(source, target):
        if Path(target).name == ANSWERS_FILE:
            raise PermissionError(13, "Permission denied")
        replace(source, target)

    monkeypatch.setattr(practice_store.os, "replace", lines_cannot_be_written)

    with pytest.raises(StoreError) as raised:
        store.save(SESSION, OFFICE, b"second", "audio/wav")

    folder = data_dir / "practice" / FOLDER
    assert str(raised.value).startswith(f"this answer was not saved in {folder}: PermissionError: ")
    # The line of the first answer is whole, and no recording is there without a line.
    assert lines_of(folder) == [{"time": "2026-10-13T12:00:05+02:00", **WUNNT, "audio_file": "1.wav"}]
    assert not (folder / "2.wav").exists()

    monkeypatch.setattr(practice_store.os, "replace", replace)
    store.save(SESSION, OFFICE, b"second", "audio/wav")

    assert [line["audio_file"] for line in lines_of(folder)] == ["1.wav", "2.wav"]
    assert files_under(folder) == ["1.wav", "2.wav", ANSWERS_FILE]


def test_a_file_where_the_folder_should_be_is_a_store_error(store, data_dir):
    (data_dir / "practice").write_text("not a folder", encoding="utf-8")

    with pytest.raises(StoreError, match="this answer was not saved in "):
        store.save(SESSION, WUNNT)

    assert (data_dir / "practice").read_text(encoding="utf-8") == "not a folder"


# --- dispute ---


def test_a_dispute_marks_its_line_only_with_the_time_and_leaves_the_rest_as_it_was(store, data_dir, clock):
    store.save(SESSION, WUNNT, b"first", "audio/webm")
    store.save(SESSION, WUNNT, b"second", "audio/webm")
    folder = data_dir / "practice" / FOLDER
    before = lines_of(folder)
    first_line = (folder / ANSWERS_FILE).read_text(encoding="utf-8").splitlines(keepends=True)[0]
    clock.now = NOON + timedelta(minutes=3)

    marked = store.dispute(FOLDER, 2, "wunnt is what I meant")

    disputed = {"time": "2026-10-13T12:03:05+02:00", "note": "wunnt is what I meant"}
    assert marked == {**before[1], "disputed": disputed}
    assert lines_of(folder) == [before[0], {**before[1], "disputed": disputed}]
    assert (folder / ANSWERS_FILE).read_text(encoding="utf-8").startswith(first_line)
    assert files_under(data_dir / "practice") == [f"{FOLDER}/1.webm", f"{FOLDER}/2.webm", f"{FOLDER}/{ANSWERS_FILE}"]
    assert ((folder / "1.webm").read_bytes(), (folder / "2.webm").read_bytes()) == (b"first", b"second")
    assert untouched(data_dir)


def test_a_second_dispute_keeps_the_first_mark_and_writes_nothing(store, data_dir, clock):
    store.save(SESSION, WUNNT)
    first = store.dispute(FOLDER, 1, "first")
    lines = data_dir / "practice" / FOLDER / ANSWERS_FILE
    written = (lines.read_bytes(), lines.stat().st_mtime_ns)
    clock.now = NOON + timedelta(hours=1)

    again = store.dispute(FOLDER, 1, "second")

    assert again["disputed"] == first["disputed"] == {"time": "2026-10-13T12:00:05+02:00", "note": "first"}
    assert (lines.read_bytes(), lines.stat().st_mtime_ns) == written


def test_the_note_of_a_dispute_is_cut_to_200_characters(store, data_dir):
    store.save(SESSION, WUNNT)

    store.dispute(FOLDER, 1, "x" * 500)

    assert lines_of(data_dir / "practice" / FOLDER)[0]["disputed"]["note"] == "x" * 200


def test_a_dispute_survives_a_new_store_on_the_same_folder(store, data_dir, clock):
    store.save(SESSION, WUNNT)
    store.save(SESSION, OFFICE)
    store.dispute(FOLDER, 1)

    reloaded = PracticeStore(data_dir / "practice", data_dir, Clock(NOON + timedelta(days=2)))

    assert reloaded.dispute(FOLDER, 1)["disputed"] == {"time": "2026-10-13T12:00:05+02:00", "note": ""}
    assert [line.get("disputed") for line in lines_of(data_dir / "practice" / FOLDER)] == [
        {"time": "2026-10-13T12:00:05+02:00", "note": ""},
        None,
    ]
    assert reloaded.summary()["answers"] == 2


def test_a_line_saved_without_the_disputed_key_is_marked(store, data_dir):
    # As a line saved before disputes were kept: no `disputed` key at all.
    store.save(SESSION, OFFICE)
    assert "disputed" not in lines_of(data_dir / "practice" / FOLDER)[0]

    store.dispute(FOLDER, 1)

    assert lines_of(data_dir / "practice" / FOLDER)[0]["disputed"]["time"] == "2026-10-13T12:00:05+02:00"


def test_a_line_saved_with_disputed_null_is_marked(store, data_dir):
    store.save(SESSION, {**WUNNT, "disputed": None})

    marked = store.dispute(FOLDER, 1)

    assert marked["disputed"]["time"] == "2026-10-13T12:00:05+02:00"
    assert list(marked) == ["time", *WUNNT, "disputed", "audio_file"]


@pytest.mark.parametrize(
    ("folder", "line", "said"),
    [
        ("2026-10-12-0a1b2c3d", 1, "not kept: there are no answers in "),
        (FOLDER, 3, "not kept: there is no answer 3 in "),
        (FOLDER, 0, "not kept: there is no answer 0 in "),
        (FOLDER, -1, "not kept: there is no answer -1 in "),
        ("a/b", 1, "not kept: 'a/b' is not the name of a folder in "),
        ("a\\b", 1, "not kept: 'a\\\\b' is not the name of a folder in "),
        ("..", 1, "not kept: '..' is not the name of a folder in "),
        (f"{FOLDER}/..", 1, "not kept: "),
        ("", 1, "not kept: '' is not the name of a folder in "),
    ],
)
def test_a_dispute_of_an_answer_that_is_not_there_is_a_store_error_and_writes_nothing(
    store, data_dir, outside, folder, line, said
):
    store.save(SESSION, WUNNT)
    store.save(SESSION, OFFICE)
    lines = data_dir / "practice" / FOLDER / ANSWERS_FILE
    kept = lines.read_bytes()

    with pytest.raises(StoreError) as raised:
        store.dispute(folder, line)

    assert str(raised.value).startswith(said)
    assert lines.read_bytes() == kept
    assert files_under(data_dir / "practice") == [f"{FOLDER}/{ANSWERS_FILE}"]
    assert files_under(outside) == ["keep.txt"]


def test_a_dispute_through_a_link_at_the_place_of_a_folder_is_refused(store, data_dir, outside):
    store.save(SESSION, WUNNT)
    (outside / ANSWERS_FILE).write_text(json.dumps({"text": "not the store's"}) + "\n", encoding="utf-8")
    link(data_dir / "practice" / "2026-10-13-ffeeddcc", outside)

    with pytest.raises(StoreError) as raised:
        store.dispute("2026-10-13-ffeeddcc", 1)

    assert "is a link, not a folder of the saved answers" in str(raised.value)
    assert not str(raised.value).startswith("not kept: ")
    assert (outside / ANSWERS_FILE).read_text(encoding="utf-8") == json.dumps({"text": "not the store's"}) + "\n"
    assert sorted(path.name for path in outside.iterdir()) == [ANSWERS_FILE, "keep.txt"]


def test_a_dispute_in_a_store_folder_that_is_not_its_own_is_refused(data_dir, outside, clock):
    link(data_dir / "practice", outside)
    store = PracticeStore(data_dir / "practice", data_dir, clock)

    with pytest.raises(StoreError, match="is not a folder of its own inside"):
        store.dispute(FOLDER, 1)

    assert files_under(outside) == ["keep.txt"]


def test_a_dispute_that_cannot_be_written_is_a_store_error_and_leaves_the_line_as_it_was(
    store, data_dir, monkeypatch
):
    store.save(SESSION, WUNNT)
    lines = data_dir / "practice" / FOLDER / ANSWERS_FILE
    kept = lines.read_bytes()

    def read_only(source, target):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(practice_store.os, "replace", read_only)

    with pytest.raises(StoreError) as raised:
        store.dispute(FOLDER, 1)

    assert str(raised.value).startswith(f"answer 1 in {lines} was not marked: PermissionError: ")
    assert lines.read_bytes() == kept


@pytest.mark.parametrize(
    ("folder", "session_id", "own"),
    [
        (FOLDER, SESSION, True),
        ("2026-10-14-0a1b2c3d", SESSION, True),  # after midnight, the same talk
        ("2026-10-13-ffeeddcc", SESSION, False),
        (FOLDER, OTHER_SESSION, False),
        ("0a1b2c3d", SESSION, False),
        ("x2026-10-13-0a1b2c3d", SESSION, False),
        ("2026-10-13-0a1b2c3d-x", SESSION, False),
        ("2026-10-13-session", "../..", True),
    ],
)
def test_a_folder_is_a_sessions_own_by_the_session_part_of_its_name_whatever_its_day(folder, session_id, own):
    assert practice_store.is_session_folder(folder, session_id) is own


# --- summary ---


def test_summary_counts_the_lines_the_sessions_and_the_bytes_of_every_file(store, data_dir):
    store.save(SESSION, WUNNT, make_noise_wav(0.5), "audio/wav")
    store.save(SESSION, OFFICE)
    store.save(OTHER_SESSION, OFFICE, b"\x00" * 1000, "audio/webm")

    files = [path for path in (data_dir / "practice").rglob("*") if path.is_file()]
    assert len(files) == 4
    assert store.summary() == {
        "answers": 3,
        "bytes": sum(path.stat().st_size for path in files),
        "sessions": 2,
        "folder": str(data_dir / "practice"),
    }


def test_summary_of_nothing_is_zero_and_makes_no_folder(store, data_dir):
    assert store.summary() == {"answers": 0, "bytes": 0, "sessions": 0, "folder": str(data_dir / "practice")}
    assert not (data_dir / "practice").exists()


# --- delete ---


def test_delete_all_removes_the_folder_and_nothing_else_and_a_second_call_is_fine(store, data_dir):
    store.save(SESSION, WUNNT, make_noise_wav(0.5), "audio/wav")
    store.save(OTHER_SESSION, OFFICE)

    store.delete_all()

    assert not (data_dir / "practice").exists()
    assert store.summary()["answers"] == 0
    assert sorted(path.name for path in data_dir.iterdir()) == ["lod", "phrases.json"]
    assert untouched(data_dir)

    store.delete_all()

    assert not (data_dir / "practice").exists()
    assert untouched(data_dir)


def test_an_answer_after_a_delete_starts_the_folder_again(store, data_dir):
    store.save(SESSION, WUNNT, b"first", "audio/wav")
    store.delete_all()

    store.save(SESSION, OFFICE, b"again", "audio/wav")

    assert [line["text"] for line in lines_of(data_dir / "practice" / FOLDER)] == [OFFICE["text"]]
    assert store.summary()["answers"] == 1


def test_delete_all_removes_a_link_inside_the_folder_and_not_what_it_leads_to(store, data_dir, outside):
    store.save(SESSION, WUNNT)
    link(data_dir / "practice" / "elsewhere", outside)

    assert store.summary()["answers"] == 1
    store.delete_all()

    assert not (data_dir / "practice").exists()
    assert files_under(outside) == ["keep.txt"]


# --- a folder that is not the store's own ---


def refused(store: PracticeStore) -> list[str]:
    """What the store says to each of its three calls; none of them may work."""
    said = []
    for call in (lambda: store.save(SESSION, WUNNT, b"audio", "audio/wav"), store.summary, store.delete_all):
        with pytest.raises(StoreError) as raised:
            call()
        said.append(str(raised.value))
    return said


def test_a_link_at_the_place_of_the_folder_that_leads_out_of_the_data_folder_is_refused(data_dir, outside, clock):
    link(data_dir / "practice", outside)
    store = PracticeStore(data_dir / "practice", data_dir, clock)

    said = refused(store)

    assert all(f"is not a folder of its own inside {data_dir.resolve()}" in text for text in said)
    assert all(f"(it leads to {outside.resolve()})" in text for text in said)
    assert files_under(outside) == ["keep.txt"]
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "mine"
    assert untouched(data_dir)


def test_a_link_at_the_place_of_the_folder_that_leads_to_another_folder_of_the_data_folder_is_refused(
    data_dir, clock
):
    link(data_dir / "practice", data_dir / "lod")
    store = PracticeStore(data_dir / "practice", data_dir, clock)

    refused(store)

    assert untouched(data_dir)


@pytest.mark.parametrize(
    "path",
    ["..", "../outside", "practice/../../outside", ".", "practice/..", "lod/practice", "practice/deeper"],
)
def test_a_path_that_is_not_a_folder_directly_inside_the_data_folder_is_refused(data_dir, outside, clock, path):
    store = PracticeStore(data_dir / path, data_dir, clock)

    refused(store)

    assert files_under(outside) == ["keep.txt"]
    assert untouched(data_dir)
    assert not (data_dir / "practice").exists()


def test_a_data_folder_behind_a_link_is_the_data_folder(tmp_path, data_dir, clock):
    link(tmp_path / "shortcut", data_dir)
    store = PracticeStore(tmp_path / "shortcut" / "practice", tmp_path / "shortcut", clock)

    store.save(SESSION, WUNNT)

    assert store.summary()["answers"] == 1
    assert files_under(data_dir / "practice") == [f"{FOLDER}/{ANSWERS_FILE}"]
    store.delete_all()
    assert not (data_dir / "practice").exists()
    assert untouched(data_dir)
