"""The progress of the learner store, /api/learner/progress, when the review queue has an item
again, and the page's Review mode and Progress block; on temporary data folders only."""

import json
import re
import shutil
import sqlite3
import subprocess
from contextlib import closing
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app import fsrs, learner_api
from app.config import ROOT
from app.learner_store import LearnerStore
from app.main import create_app
from tests.conftest import plain_page
from tests.test_learner_store import PHRASES, Clock, answer_line

NOW = datetime(2026, 10, 15, 9, 0, tzinfo=timezone.utc)  # a Thursday of ISO week 2026-W42
FOLDER = "2026-10-13-0a1b2c3d"
TOPICS = [
    ("wunnen", "Home", ("wunnen-1", "wunnen-2", "wunnen-3")),
    ("famill", "Family", ("famill-1", "famill-2")),
    ("iessen", "Food", ("iessen-1",)),
]
EMPTY_WEEKS = [
    {"week": week, "words": 0, "grammar": 0, "task": 0} for week in ("2026-W42", "2026-W41", "2026-W40", "2026-W39")
]


def answer(question_id: str, time: str, duration_ms: int | None) -> str:
    """A judged answer with a correction, which becomes a review item."""
    line = {
        "topic_id": question_id.split("-")[0],
        "question_id": question_id,
        "question_lb": "Wou wunnt Dir?",
        "raw_transcript": "Ech wunnt zu Esch.",
        "text": "Ech wunnt zu Esch.",
        "help_used": False,
        "verdict": "fix",
        "correction": {
            "reaction_en": "Esch, good.",
            "quote": "Ech wunnt",
            "better_lb": f"Ech wunnen zu Esch ({time}).",
            "why_en": "with ech the verb is wunnen, so:",
        },
        "claims": [],
    }
    if duration_ms is not None:
        line["duration_ms"] = duration_ms
    return answer_line(line, time, None)


# (question id, time, duration in ms, category of its correction)
ANSWERS = [
    ("wunnen-1", "2026-10-14T12:00:00+02:00", 90_000, "grammar"),
    ("wunnen-1", "2026-10-13T08:00:00+02:00", 30_000, "words"),
    ("wunnen-2", "2026-10-06T10:00:00+02:00", 60_000, "task"),
    ("famill-1", "2026-09-01T10:00:00+02:00", None, "words"),  # older than four weeks
    ("famill-foto-1", "2026-10-14T13:00:00+02:00", None, None),  # a photo question, of no topic's list
]
# (role, time, duration in ms): the tutor's speech is not the learner's.
TURNS = [
    ("user", "2026-10-15T08:00:00+00:00", 12_000),
    ("tutor", "2026-10-15T08:00:05+00:00", 50_000),
    ("user", "2026-10-01T08:00:00+00:00", 6_000),
]


def seed(folder) -> None:
    (folder / "practice" / FOLDER).mkdir(parents=True)
    (folder / "phrases.json").write_text(json.dumps(PHRASES, ensure_ascii=False), encoding="utf-8")
    lines = "".join(answer(question, time, duration) for question, time, duration, _ in ANSWERS)
    (folder / "practice" / FOLDER / "answers.jsonl").write_text(lines, encoding="utf-8")


def categorise_and_talk(store: LearnerStore) -> None:
    """Sets the category of each correction and adds the chat turns, as the app is to record them."""
    with closing(sqlite3.connect(store.path)) as connection, connection:
        for _, time, _, category in ANSWERS:
            connection.execute(
                "UPDATE correction SET category = ? WHERE answer_id = (SELECT id FROM answer WHERE at = ?)",
                (category, time),
            )
        for role, time, duration in TURNS:
            connection.execute(
                "INSERT INTO turn (session_id, role, mode, text, duration_ms, at) VALUES ('s1', ?, 'chat', 'x', ?, ?)",
                (role, duration, time),
            )


def item_ids(store: LearnerStore) -> list[int]:
    with closing(sqlite3.connect(store.path)) as connection:
        return [row[0] for row in connection.execute("SELECT id FROM item ORDER BY id")]


@pytest.fixture
def clock():
    return Clock(NOW)


@pytest.fixture
def empty(tmp_path, clock):
    folder = tmp_path / "data"
    folder.mkdir()
    store = LearnerStore(folder / "learner", folder, clock)
    store.open()
    yield store
    store.close()


@pytest.fixture
def store(tmp_path, clock):
    folder = tmp_path / "data"
    seed(folder)
    store = LearnerStore(folder / "learner", folder, clock)
    store.open()
    store.migrate_files()
    categorise_and_talk(store)
    yield store
    store.close()


# --- the store ---


def test_progress_gives_the_five_figures_of_a_seeded_store(store):
    first, second = item_ids(store)[:2]
    store.review(first, fsrs.AGAIN, now=NOW - timedelta(days=3))  # learning, due again three days ago
    store.review(second, fsrs.GOOD, now=NOW)  # learning, due in ten minutes

    progress = store.progress(NOW, TOPICS)

    # This week: 90 s + 30 s of answers and 12 s of a chat turn; in all also 60 s and 6 s.
    assert progress["minutes"] == {"week": 2.2, "total": 3.3}
    assert progress["phrases"] == len(PHRASES)
    assert progress["items"] == {"in_review": 2, "total": len(PHRASES) + len(ANSWERS), "due": 1}
    assert progress["corrections_by_week"] == [
        {"week": "2026-W42", "words": 1, "grammar": 1, "task": 0},
        {"week": "2026-W41", "words": 0, "grammar": 0, "task": 1},
        *EMPTY_WEEKS[2:],
    ]
    assert progress["topics"] == [
        {"id": "wunnen", "name_en": "Home", "answered": 2, "questions": 3},
        {"id": "famill", "name_en": "Family", "answered": 1, "questions": 2},
        {"id": "iessen", "name_en": "Food", "answered": 0, "questions": 1},
    ]


def test_progress_of_an_empty_store_is_zeros_and_four_weeks_of_zeros(empty):
    assert empty.progress(NOW, TOPICS) == {
        "minutes": {"week": 0, "total": 0},
        "phrases": 0,
        "items": {"in_review": 0, "total": 0, "due": 0},
        "corrections_by_week": EMPTY_WEEKS,
        "topics": [{"id": id_, "name_en": name, "answered": 0, "questions": len(ids)} for id_, name, ids in TOPICS],
    }


def test_the_weeks_are_iso_weeks_keyed_by_their_iso_year(empty):
    # 2027-01-02 is a Saturday of ISO week 53 of 2026; the Monday after begins 2027-W01.
    weeks = empty.progress(datetime(2027, 1, 2, 23, 0, tzinfo=timezone.utc))["corrections_by_week"]
    monday = empty.progress(datetime(2027, 1, 4, 0, 0, tzinfo=timezone.utc))["corrections_by_week"]

    assert [week["week"] for week in weeks] == ["2026-W53", "2026-W52", "2026-W51", "2026-W50"]
    assert monday[0]["week"] == "2027-W01"


def test_next_due_of_an_empty_store_is_none(empty):
    assert empty.next_due(NOW) is None


def test_next_due_is_the_earliest_due_of_the_items_seen_or_tomorrow_when_new_ones_wait(store):
    first, second = item_ids(store)[:2]

    store.review(first, fsrs.GOOD, now=NOW)
    assert store.next_due(NOW) == "2026-10-15T09:10:00+00:00"
    # With no new item left for today, the new ones come at midnight (UTC), before a review in days.
    store.review(first, fsrs.GOOD, now=NOW + timedelta(minutes=10))
    store.review(second, fsrs.EASY, now=NOW)
    assert store.next_due(NOW, new_per_day=2) == "2026-10-16T00:00:00+00:00"
    assert store.next_due(NOW, new_per_day=10) == store.item(first)["due"]


# --- the routes ---


@pytest.fixture
def seeded(settings):
    seed(settings.data_dir)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_the_progress_route_has_the_shape_of_the_block_and_counts_the_practice_topics(seeded):
    body = seeded.get("/api/learner/progress").json()

    assert set(body) == {"minutes", "phrases", "items", "corrections_by_week", "topics"}
    # 90 s, 30 s and 60 s of answers; this week's share depends on the day the test runs.
    assert body["minutes"]["total"] == 3.0 and 0 <= body["minutes"]["week"] <= 3.0
    assert body["phrases"] == len(PHRASES)
    assert set(body["items"]) == {"in_review", "total", "due"}
    assert [len(week) for week in body["corrections_by_week"]] == [4] * 4
    assert len(body["topics"]) == 12
    assert {"id": "wunnen", "name_en": "Home", "answered": 2, "questions": 8} in body["topics"]
    assert {"id": "famill", "name_en": "Family", "answered": 1, "questions": 8} in body["topics"]
    assert all(set(topic) == {"id", "name_en", "answered", "questions"} for topic in body["topics"])


def test_items_in_review_rise_by_one_per_rated_item(seeded):
    def in_review():
        return seeded.get("/api/learner/progress").json()["items"]["in_review"]

    before = in_review()
    item = seeded.get("/api/review/next").json()["item"]
    seeded.post("/api/review/rate", data={"item_id": item["id"], "rating": "2", "text": "eppes"})
    item = seeded.get("/api/review/next").json()["item"]
    seeded.post("/api/review/rate", data={"item_id": item["id"], "rating": "4", "text": "eppes"})

    assert [before, in_review()] == [0, 2]


def test_without_practice_content_the_topics_are_empty():
    assert learner_api.practice_topics(None) == []


def test_without_the_learner_store_the_progress_route_is_a_503(seeded):
    state = seeded.app.state
    state.learner.close()
    state.learner, state.learner_detail = None, "the database fails its check"

    response = seeded.get("/api/learner/progress")

    assert response.status_code == 503
    assert response.json()["error"] == "the learner store is not open: the database fails its check"


def test_next_without_an_item_says_when_the_next_one_is_due(settings):
    (settings.data_dir).mkdir(parents=True)
    (settings.data_dir / "phrases.json").write_text(json.dumps(PHRASES[:1]), encoding="utf-8")
    with TestClient(create_app(settings)) as client:
        item = client.get("/api/review/next").json()["item"]
        answered = client.post("/api/review/answer", data={"item_id": item["id"], "text": PHRASES[0]["lb"]}).json()
        body = client.get("/api/review/next").json()

    assert answered["matched"] is True
    assert (body["item"], body["next_due"]) == (None, answered["due"])


# --- the page ---


def test_the_page_has_the_review_toggle_beside_practice_and_the_review_card_in_the_dock(client):
    html = plain_page(client.get("/").text)

    assert re.search(
        r'<button id="practice" type="button" aria-pressed="false"\s[^>]*>Practice</button>\s*'
        r'<button id="review" type="button" aria-pressed="false"\s[^>]*>Review</button>\s*<button id="teacher"',
        html,
    )
    assert re.search(r'<div id="review-panel" class="review" hidden>\s*<div id="review-card" class="card">', html)
    for element_id in (
        "review-counts", "review-gloss", "review-hint", "review-result", "review-expected", "review-said",
        "review-ratings", "review-override", "review-next", "review-leave",
    ):
        assert f'id="{element_id}"' in html
    ratings = re.findall(r'<button type="button" data-rating="(\d)">(\w+)</button>', html)
    assert ratings == [("1", "Again"), ("2", "Hard"), ("3", "Good"), ("4", "Easy")]
    assert '<button id="review-override" type="button" hidden>I did not know it</button>' in html


def test_the_page_has_the_progress_block_above_the_voice_block(client):
    html = plain_page(client.get("/").text)

    assert re.search(r'<details id="progress" class="progress">\s*<summary>Progress</summary>', html)
    assert html.index('<details id="progress"') < html.index('<details id="voice"')
    for element_id in ("progress-figures", "progress-corrections", "progress-topics", "progress-problem"):
        assert f'id="{element_id}"' in html
    assert "0 min" in html


def test_the_page_script_runs_review_and_progress_through_their_routes():
    static = ROOT / "static"
    script = (static / "app.js").read_text(encoding="utf-8")
    style = (static / "style.css").read_text(encoding="utf-8")

    assert sorted(re.findall(r'"(api/(?:review|learner)/[^"]*)"', script)) == [
        "api/learner/export",
        "api/learner/progress",
        "api/learner/summary",
        "api/review/answer",
        "api/review/next",
        "api/review/rate",
    ]
    # Decision 6a4c454e07c9: the one override is Again.
    assert "const OVERRIDE_RATING = 1;" in script
    assert "rateReview(OVERRIDE_RATING, review.answer.logId)" in script
    # Decision 9c4aceeee60b: review and practice share the dock and never are on together.
    assert "async function enterReview() {\n  if (review || busy) return;\n  leavePractice();" in script
    assert "leaveReview();\n    enterPractice();" in script
    assert 'mode: review.heard ? "spoken" : "typed"' in script
    assert "if (review) {\n    sendReviewAnswer();\n    return;\n  }" in script
    assert "if (review) {\n    review.heard = heard.text;" in script
    assert script.count("refreshProgress()") >= 3
    assert re.search(r"streak|goal", script, re.IGNORECASE) is None
    for rule in ("#review[aria-pressed=\"true\"]", ".review .result {", ".progress table {"):
        assert rule in style
    added = style.split("/* --- review --- */")[1].split(".progress ul {")[0]
    # The review and progress rules carry no colour of an error or a warning.
    assert ".progress table {" in added
    assert not re.search(r"--error|--warn|red\b", added)


def test_the_page_script_parses():
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    subprocess.run(["node", "--check", str(ROOT / "static" / "app.js")], check=True, timeout=30)
