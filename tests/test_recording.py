"""What the app records in the learner store as it goes, with the fake engines and temporary data folders."""

import json
import logging
import sqlite3
from contextlib import closing
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app import api
from app.engines.base import Judgement, LLMError
from app.engines.providers import RoleConfig, default_config
from app.learner_store import PRACTICE_FIELDS, LearnerStore
from app.practice_store import StoreError
from tests.test_learner_store import QUESTIONS, RECORDED, TYPED, Clock, answer_line
from tests.test_learner_store import data_dir as data_dir  # noqa: F401 (a fixture)
from tests.test_practice_store import OFFICE, WUNNT

FIX = Judgement(
    verdict="fix",
    reaction_en="Esch, good.",
    quote="Ech wunnt",
    better_lb="Ech wunnen zu Esch.",
    why_en="with ech the verb is wunnen, so:",
    claims=[{"form": "wunnen", "lemma": "wunnen"}],
)
RECORDING = ("speech", b"\x1aE\xdf\xa3 a recording", "audio/webm")


def rows(path, sql: str, *args) -> list[dict]:
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(sql, args)]


@pytest.fixture
def state(client):
    return client.app.state


@pytest.fixture
def db(state):
    return lambda sql, *args: rows(state.learner.path, sql, *args)


@pytest.fixture
def session_id(client):
    return client.post("/api/session").json()["session_id"]


@pytest.fixture
def practice(client, session_id, state):
    """Starts the topic wunnen; posts answers to its questions, the judgements queued first."""

    def answer(text, judgement=None, files=None, **more):
        if judgement is not None:
            state.llm.judge_queue.append(judgement)
        data = {"session_id": session_id, "text": text, **more}
        return client.post("/api/practice/answer", data=data, files=files)

    opening = client.post("/api/practice/start", data={"session_id": session_id, "topic_id": "wunnen"}).json()
    answer.question = opening["question"]
    return answer


def test_a_new_session_is_a_session_row_without_modes(state, session_id, db):
    (row,) = db("SELECT * FROM session")
    assert (row["id"], row["modes"]) == (session_id, "")
    assert row["started"] == row["last_seen"]


def test_a_typed_chat_turn_is_two_turns_and_the_phrase_it_names_with_its_review_item(client, session_id, db):
    response = client.post("/api/chat", json={"session_id": session_id, "text": "Hello there"})

    assert response.status_code == 200
    turns = db("SELECT session_id, role, mode, text, duration_ms FROM turn ORDER BY id")
    assert turns == [
        {"session_id": session_id, "role": "user", "mode": "chat", "text": "Hello there", "duration_ms": 0},
        {"session_id": session_id, "role": "tutor", "mode": "chat", "text": "Moien! Hello! How was your day?",
         "duration_ms": None},
    ]
    (phrase,) = db("SELECT * FROM phrase")
    assert (phrase["phrase_key"], phrase["lb"], phrase["en"], phrase["times_seen"]) == ("moien", "Moien!", "Hello!", 1)
    assert json.loads(phrase["sessions_json"]) == [session_id]
    (item,) = db("SELECT * FROM item")
    assert (item["source"], item["source_id"], item["form"], item["gloss"], item["state"]) == (
        "phrase", phrase["id"], "Moien!", "Hello!", "new"
    )
    assert db("SELECT modes FROM session") == [{"modes": "chat"}]


def test_a_phrase_met_again_counts_up_without_a_second_row_or_item(client, session_id, db, settings):
    other = client.post("/api/session").json()["session_id"]
    for session in (session_id, session_id, other):
        assert client.post("/api/chat", json={"session_id": session, "text": "Hi"}).status_code == 200

    (phrase,) = db("SELECT * FROM phrase")
    assert phrase["times_seen"] == 3
    assert json.loads(phrase["sessions_json"]) == [session_id, other]
    assert db("SELECT COUNT(*) AS n FROM item") == [{"n": 1}]
    # phrases.json is written as before, and agrees.
    (entry,) = json.loads((settings.data_dir / "phrases.json").read_text(encoding="utf-8"))
    assert (entry["lb"], entry["times_seen"], entry["sessions"]) == ("Moien!", 3, [session_id, other])


def test_a_spoken_turn_keeps_how_long_the_learner_spoke(client, session_id, db):
    data = {"session_id": session_id, "text": "How do you say thank you?", "duration_ms": "4200", "stt_ms": "800"}

    response = client.post("/api/turn", data=data)

    assert response.status_code == 200
    assert db("SELECT role, mode, duration_ms FROM turn ORDER BY id") == [
        {"role": "user", "mode": "teacher", "duration_ms": 4200},
        {"role": "tutor", "mode": "teacher", "duration_ms": None},
    ]
    assert db("SELECT modes FROM session") == [{"modes": "teacher"}]


def test_a_json_chat_turn_takes_a_duration_too_and_a_negative_one_is_zero(client, session_id, db):
    client.post("/api/chat", json={"session_id": session_id, "text": "Hi", "duration_ms": 1500})
    client.post("/api/chat", json={"session_id": session_id, "text": "Hi", "duration_ms": -3})

    assert db("SELECT duration_ms FROM turn WHERE role = 'user' ORDER BY id") == [
        {"duration_ms": 1500},
        {"duration_ms": 0},
    ]


def test_a_corrected_answer_is_an_answer_its_correction_and_a_review_item(practice, state, session_id, db, settings):
    body = practice(WUNNT["text"], FIX, files={"audio": RECORDING}, raw_text="Ech wunnt zu Äsch.", duration_ms="3100")

    assert body.status_code == 200
    turn = body.json()
    kept = turn["kept"]
    (answer,) = db("SELECT * FROM answer")
    assert (answer["folder"], answer["line"], answer["session_id"]) == (kept["folder"], 1, session_id)
    assert answer["audio_file"] == f"practice/{kept['folder']}/1.webm"
    assert (settings.data_dir / answer["audio_file"]).read_bytes() == RECORDING[1]
    assert (answer["topic_id"], answer["question_id"], answer["text"], answer["raw_transcript"]) == (
        "wunnen", practice.question["id"], WUNNT["text"], "Ech wunnt zu Äsch."
    )
    assert (answer["duration_ms"], answer["verdict"], answer["help_used"]) == (3100, "fix", 0)
    assert answer["judge_model"] is None  # the fake engine names no model
    (correction,) = db("SELECT * FROM correction")
    assert correction["answer_id"] == answer["id"]
    assert (correction["quote"], correction["better_lb"], correction["disputed"]) == (FIX.quote, FIX.better_lb, 0)
    assert turn["correction"]["category"] and turn["correction"]["label"]
    assert (correction["category"], correction["label"]) == (turn["correction"]["category"], turn["correction"]["label"])
    assert json.loads(correction["claims_json"]) == turn["correction"]["claims"]
    (item,) = db("SELECT * FROM item WHERE source = 'correction'")
    assert (item["source_id"], item["form"], item["gloss"], item["hint"]) == (
        correction["id"], FIX.better_lb, practice.question["en"], FIX.quote
    )
    assert db("SELECT modes FROM session") == [{"modes": "practice"}]
    # The saved line is as before: the learner store adds nothing to it.
    (line,) = [json.loads(text) for text in (settings.data_dir / "practice" / kept["folder"] / "answers.jsonl")
               .read_text(encoding="utf-8").splitlines()]
    assert "duration_ms" not in line and line["audio_file"] == "1.webm"


def test_a_good_typed_answer_has_a_correction_row_and_no_item(practice, db):
    practice(OFFICE["text"])

    (answer,) = db("SELECT audio_file, duration_ms, verdict FROM answer")
    assert answer == {"audio_file": None, "duration_ms": 0, "verdict": "good"}
    assert db("SELECT category, label FROM correction") == [{"category": None, "label": None}]
    assert db("SELECT COUNT(*) AS n FROM item WHERE source = 'correction'") == [{"n": 0}]


def test_an_answer_the_judge_did_not_judge_is_kept_without_a_verdict_or_correction(practice, db):
    response = practice(OFFICE["text"], LLMError("timeout", "the judge took too long"))

    assert response.status_code == 200
    assert db("SELECT verdict, judge_model FROM answer") == [{"verdict": None, "judge_model": None}]
    assert db("SELECT COUNT(*) AS n FROM correction") == [{"n": 0}]


def test_a_dispute_marks_the_recorded_correction(client, practice, session_id, db):
    kept = practice(WUNNT["text"], FIX).json()["kept"]

    response = client.post(
        "/api/practice/dispute", data={"session_id": session_id, "folder": kept["folder"], "line": kept["line"]}
    )

    assert response.status_code == 200
    assert db("SELECT disputed FROM correction") == [{"disputed": 1}]


def test_without_a_learner_store_every_route_answers_and_nothing_is_recorded(client, state, monkeypatch, settings):
    path = state.learner.path
    monkeypatch.setattr(state, "learner", None)
    monkeypatch.setattr(state.orchestrator, "learner", None)

    session_id = client.post("/api/session").json()["session_id"]
    chat = client.post("/api/chat", json={"session_id": session_id, "text": "Hi", "duration_ms": 900})
    turn = client.post("/api/turn", data={"session_id": session_id, "text": "Hi", "duration_ms": "900"})
    client.post("/api/practice/start", data={"session_id": session_id, "topic_id": "wunnen"})
    state.llm.judge_queue.append(FIX)
    answer = client.post(
        "/api/practice/answer", data={"session_id": session_id, "text": WUNNT["text"]}, files={"audio": RECORDING}
    )

    assert [response.status_code for response in (chat, turn, answer)] == [200, 200, 200]
    assert answer.json()["kept"]["line"] == 1
    counts = rows(path, "SELECT (SELECT COUNT(*) FROM session) + (SELECT COUNT(*) FROM turn)"
                       " + (SELECT COUNT(*) FROM answer) + (SELECT COUNT(*) FROM phrase) AS n")
    assert counts == [{"n": 0}]
    assert (settings.data_dir / "phrases.json").is_file()


def test_a_store_that_fails_is_logged_and_the_turn_and_the_answer_go_on(
    client, state, session_id, practice, monkeypatch, caplog
):
    def fails(*args):
        raise StoreError("this was not recorded in learner.sqlite: OperationalError: database is locked")

    for method in ("turn_said", "phrase_met", "answer_kept", "session_seen"):
        monkeypatch.setattr(state.learner, method, fails)

    with caplog.at_level(logging.WARNING):
        chat = client.post("/api/chat", json={"session_id": session_id, "text": "Hi"})
        answer = practice(WUNNT["text"], FIX)
        session = client.post("/api/session")

    assert [response.status_code for response in (chat, answer, session)] == [200, 200, 200]
    assert chat.json()["phrases_added"] == ["Moien!"]
    assert answer.json()["kept"]["line"] == 1
    for method in ("turn_said", "phrase_met", "answer_kept", "session_seen"):
        assert f"the learner store has not recorded {method}: StoreError: " in caplog.text


def test_the_judge_model_is_the_one_the_engine_names(settings):
    cli = SimpleNamespace(llm=SimpleNamespace(name="claude_cli"), settings=replace(settings, llm_model="opus"))
    config = default_config(settings)
    routed = replace(config, roles={**config.roles, "judge": RoleConfig("claude_cli", model="haiku")})
    provider = SimpleNamespace(llm=SimpleNamespace(name="provider", config=routed), settings=settings)
    unrouted = SimpleNamespace(llm=SimpleNamespace(name="provider", config=config), settings=settings)
    fake = SimpleNamespace(llm=SimpleNamespace(name="fake"), settings=settings)

    assert [api.judge_model(state) for state in (cli, provider, unrouted, fake)] == [
        "opus", "haiku", settings.llm_model, None
    ]


# The store's own methods.


@pytest.fixture
def store(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    store = LearnerStore(folder / "learner", folder, Clock())
    store.open()
    yield store
    store.close()


def test_a_session_gathers_its_modes_once_each_and_its_last_seen(store):
    store.session_seen("s1", "")
    store._clock.now = store._clock.now.replace(minute=45)
    for mode in ("chat", "teacher", "chat", "practice"):
        store.session_seen("s1", mode)

    (row,) = rows(store.path, "SELECT * FROM session")
    assert (row["modes"], row["started"], row["last_seen"]) == (
        "chat,teacher,practice", "2026-10-14T09:30:00+00:00", "2026-10-14T09:45:00+00:00"
    )


def test_a_turn_of_a_session_never_seen_makes_its_session(store):
    store.turn_said("s2", "chat", "Hi", 0, "Moien!")

    assert rows(store.path, "SELECT id, modes FROM session") == [{"id": "s2", "modes": "chat"}]
    assert store.counts()["turn"] == 2


def test_phrase_met_is_new_once_and_ignores_an_empty_phrase(store):
    assert [store.phrase_met(lb, "Hello!", "s1") for lb in ("Moien!", "moien", " Moien. ", "...")] == [
        True, False, False, False
    ]
    assert rows(store.path, "SELECT lb, times_seen FROM phrase") == [{"lb": "Moien!", "times_seen": 3}]
    assert store.counts()["item"] == 1


def test_an_answer_at_a_place_already_recorded_is_not_recorded_twice(store):
    record = {**WUNNT, "category": "words", "label": "verified", "disputed": None}
    assert store.answer_kept("s1", RECORDED, 1, record, "1.webm", 2000, "sonnet", "Where do you live?") is True
    assert store.answer_kept("s1", RECORDED, 1, record, "1.webm", 2000, "sonnet", "Where do you live?") is False

    counts = store.counts()
    assert (counts["answer"], counts["correction"], counts["item"]) == (1, 1, 1)
    assert rows(store.path, "SELECT category, label, disputed FROM correction") == [
        {"category": "words", "label": "verified", "disputed": 0}
    ]
    assert rows(store.path, "SELECT judge_model, audio_file FROM answer") == [
        {"judge_model": "sonnet", "audio_file": f"practice/{RECORDED}/1.webm"}
    ]


def test_phrases_and_answers_imported_on_the_first_start_are_not_recorded_again(data_dir):
    store = LearnerStore(data_dir / "learner", data_dir, Clock())
    store.open()
    try:
        store.migrate_files(QUESTIONS)
        before = store.counts()
        assert store.phrase_met("Moien", "Hello!", "s9") is False
        assert store.answer_kept("s9", RECORDED, 1, WUNNT, "1.webm", 0, None, None) is False
        after = store.counts()
        (moien,) = rows(store.path, "SELECT times_seen, sessions_json FROM phrase WHERE phrase_key = 'moien'")
    finally:
        store.close()
    assert {table: after[table] for table in ("phrase", "answer", "correction", "item")} == {
        table: before[table] for table in ("phrase", "answer", "correction", "item")
    }
    assert (moien["times_seen"], json.loads(moien["sessions_json"])) == (4, ["s1", "s2", "s9"])


# The import of category, label and disputed.


def test_the_first_import_keeps_the_category_label_and_dispute_of_a_line(tmp_path):
    folder = tmp_path / "data"
    (folder / "practice" / RECORDED).mkdir(parents=True)
    line = {**WUNNT, "category": "grammar", "label": "likely", "disputed": {"time": "2026-10-13T12:01:00+02:00",
                                                                           "note": ""}}
    (folder / "practice" / RECORDED / "answers.jsonl").write_text(
        answer_line(line, "2026-10-13T12:00:05+02:00", None), encoding="utf-8"
    )
    store = LearnerStore(folder / "learner", folder, Clock())
    store.open()
    try:
        store.migrate_files(QUESTIONS)
        assert rows(store.path, "SELECT category, label, disputed FROM correction") == [
            {"category": "grammar", "label": "likely", "disputed": 1}
        ]
    finally:
        store.close()


def test_a_database_imported_before_gets_the_fields_and_the_missed_answers_once(data_dir):
    store = LearnerStore(data_dir / "learner", data_dir, Clock())
    store.open()
    store.migrate_files(QUESTIONS)
    # As a database imported before the fields were copied: no practice-fields row.
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute("DELETE FROM migrated WHERE source = ?", (PRACTICE_FIELDS,))
    store.close()
    lines = data_dir / "practice" / RECORDED / "answers.jsonl"
    labelled = {**WUNNT, "category": "words", "label": "unverified", "disputed": {"time": "x", "note": ""}}
    lines.write_text(
        answer_line(labelled, "2026-10-13T12:00:05+02:00", "1.webm")
        + answer_line({**OFFICE, "category": "", "label": ""}, "2026-10-13T12:02:00+02:00", None),
        encoding="utf-8",
    )
    (data_dir / "practice" / TYPED / "answers.jsonl").unlink()

    second = LearnerStore(data_dir / "learner", data_dir, Clock())
    second.open()
    try:
        assert second.migrate_files(QUESTIONS) == {PRACTICE_FIELDS: 2}
        assert second.migrate_files(QUESTIONS) == {}
        correction = rows(second.path, "SELECT category, label, disputed FROM correction")
        answers = rows(second.path, "SELECT folder, line FROM answer ORDER BY folder, line")
        counts = second.counts()
    finally:
        second.close()
    assert correction == [{"category": "words", "label": "unverified", "disputed": 1}]
    assert answers == [{"folder": RECORDED, "line": 1}, {"folder": RECORDED, "line": 2}, {"folder": TYPED, "line": 1}]
    assert counts["item"] == 4  # the three phrases and the one correction, none twice
