"""The review queue, log, override and rebuild of the learner store, the answer match and the
/api/review routes, on temporary data folders only."""

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app import fsrs
from app.learner_store import LearnerStore, matches
from app.main import create_app
from tests.test_fsrs import PINNED
from tests.test_learner_store import Clock, answer_line
from tests.test_practice_store import WUNNT

T0 = datetime(2026, 10, 15, 9, 0, tzinfo=timezone.utc)
CACHED = ("state", "step", "stability", "difficulty", "due", "last_review", "reps", "lapses")


def phrase(n: int) -> dict:
    return {
        "lb": f"Saz {n}",
        "en": f"Sentence {n}",
        "first_seen": "2026-10-12T08:00:00+00:00",
        "last_seen": "2026-10-12T08:00:00+00:00",
        "times_seen": 1,
        "sessions": ["s1"],
    }


def seed(folder, phrases: int = 12) -> None:
    """`phrases` phrase items and one correction item, "Ech wunnen zu Esch."."""
    (folder / "practice" / "2026-10-13-0a1b2c3d").mkdir(parents=True)
    (folder / "phrases.json").write_text(json.dumps([phrase(n) for n in range(phrases)]), encoding="utf-8")
    (folder / "practice" / "2026-10-13-0a1b2c3d" / "answers.jsonl").write_text(
        answer_line(WUNNT, "2026-10-13T12:00:05+02:00", None), encoding="utf-8"
    )


class FakeLod:
    """LOD with two forms of `hunn` and nothing else."""

    FORMS = {"hunn": ["hunn"], "hat": ["hunn"], "hues": ["hunn"]}

    def lookup_form(self, form: str) -> list[str]:
        return self.FORMS.get(form.casefold(), [])

    def close(self) -> None:
        pass


@pytest.fixture
def clock():
    return Clock(T0)


@pytest.fixture
def store(tmp_path, clock):
    folder = tmp_path / "data"
    seed(folder)
    store = LearnerStore(folder / "learner", folder, clock)
    store.open()
    store.migrate_files({"wunnen-1": "Where do you live?"})
    yield store
    store.close()


def item_id(store: LearnerStore, form: str) -> int:
    with closing(sqlite3.connect(store.path)) as connection:
        return connection.execute("SELECT id FROM item WHERE form = ?", (form,)).fetchone()[0]


def log_rows(store: LearnerStore) -> list[dict]:
    with closing(sqlite3.connect(store.path)) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute("SELECT * FROM review_log ORDER BY id")]


def cached(store: LearnerStore, item: int) -> dict:
    row = store.item(item)
    return {key: row[key] for key in CACHED}


# --- queue ---


def test_on_day_one_the_queue_holds_10_new_items_in_order_of_creation(store):
    queue = store.queue()

    assert len(queue) == 10
    assert [item["state"] for item in queue] == ["new"] * 10
    assert [item["form"] for item in queue] == [f"Saz {n}" for n in range(10)]
    assert set(queue[0]) == {"id", "source", "form", "gloss", "hint", "state", "due"}
    assert store.review_counts() == {"due": 0, "new_available": 10, "in_review": 0, "total": 13}


def test_new_items_seen_today_use_up_the_cap_and_the_next_utc_day_has_10_again(store, clock):
    for n in range(4):
        store.review(item_id(store, f"Saz {n}"), fsrs.EASY)

    assert [item["form"] for item in store.queue()] == [f"Saz {n}" for n in range(4, 10)]
    assert store.review_counts()["new_available"] == 6
    clock.now = T0.replace(hour=23, minute=59, second=59)
    assert len(store.queue()) == 6
    clock.now = T0 + timedelta(days=1)
    assert store.review_counts()["new_available"] == 9  # 13 items, 4 seen
    assert len([item for item in store.queue() if item["state"] == "new"]) == 9


def test_due_items_come_first_by_due_time_then_new_items(store, clock):
    review = item_id(store, "Saz 0")
    learning = item_id(store, "Saz 1")
    store.review(review, fsrs.EASY)  # review, due in 8 days
    clock.now = T0 + timedelta(days=8, minutes=5)
    store.review(learning, fsrs.GOOD)  # learning step 1, due in 10 minutes
    clock.now = T0 + timedelta(days=8, minutes=20)

    queue = store.queue()

    assert [item["id"] for item in queue[:2]] == [review, learning]
    assert [item["state"] for item in queue[:2]] == ["review", "learning"]
    assert all(item["state"] == "new" for item in queue[2:])
    assert store.review_counts()["due"] == 2
    assert len(store.queue(limit=3)) == 3


def test_an_item_not_yet_due_is_not_in_the_queue(store, clock):
    seen = item_id(store, "Saz 0")
    store.review(seen, fsrs.EASY)
    clock.now = T0 + timedelta(days=7)

    assert seen not in [item["id"] for item in store.queue()]
    assert store.review_counts()["due"] == 0


# --- review, rebuild, override ---


def test_the_pinned_sequence_through_the_store_caches_the_scheduler_values_and_rebuild_keeps_them(store, clock):
    item = item_id(store, "Saz 0")
    assert store.item(item)["created"] == T0.isoformat()

    for rating, moment, state, step, stability, difficulty, due in PINNED:
        clock.now = moment
        done = store.review(item, rating, mode="spoken", heard="h", confirmed="c", matched=False, rated_by="learner")
        assert (done["state"], done["due"]) == (state, due.isoformat())
    before = cached(store, item)

    assert before["stability"] == pytest.approx(33.07711761294955, abs=1e-6)
    assert before["difficulty"] == pytest.approx(7.630706295391642, abs=1e-6)
    assert (before["reps"], before["lapses"]) == (8, 1)
    assert store.rebuild(item) == store.item(item)
    assert cached(store, item) == before


def test_a_review_logs_its_row_with_the_answer(store):
    item = item_id(store, "Saz 0")

    done = store.review(item, fsrs.GOOD, mode="spoken", heard="saz null", confirmed="Saz 0", matched=True,
                        rated_by="auto")  # fmt: skip

    [row] = log_rows(store)
    assert done == {"log_id": row["id"], "item_id": item, "rating": 3, "state": "learning",
                    "due": "2026-10-15T09:10:00+00:00"}  # fmt: skip
    assert {key: row[key] for key in ("item_id", "at", "rating", "mode", "heard", "confirmed", "matched",
                                      "rated_by", "superseded")} == {
        "item_id": item, "at": "2026-10-15T09:00:00+00:00", "rating": 3, "mode": "spoken", "heard": "saz null",
        "confirmed": "Saz 0", "matched": 1, "rated_by": "auto", "superseded": 0,
    }  # fmt: skip


def test_a_review_with_a_bad_rating_or_item_writes_nothing(store):
    item = item_id(store, "Saz 0")

    for bad in (0, 5, True):
        with pytest.raises(ValueError, match="rating"):
            store.review(item, bad)
    with pytest.raises(ValueError, match="no review item"):
        store.review(99999, fsrs.GOOD)
    with pytest.raises(ValueError, match="mode"):
        store.review(item, fsrs.GOOD, mode="sung")

    assert log_rows(store) == []
    assert store.item(item)["state"] == "new"


def test_override_supersedes_the_auto_row_and_the_replay_gives_the_learners_rating(store, clock):
    item = item_id(store, "Saz 0")
    auto = store.review(item, fsrs.GOOD, heard="h", confirmed="Saz 0", matched=True, rated_by="auto")
    clock.now = T0 + timedelta(seconds=4)

    done = store.override(auto["log_id"], fsrs.AGAIN)

    first, second = log_rows(store)
    assert (first["superseded"], first["rated_by"], first["rating"]) == (1, "auto", 3)
    assert (second["superseded"], second["rated_by"], second["rating"]) == (0, "learner", 1)
    assert (second["heard"], second["confirmed"], second["matched"]) == ("h", "Saz 0", 1)
    expected = fsrs.review(fsrs.new_card(T0), fsrs.AGAIN, T0 + timedelta(seconds=4))
    row = store.item(item)
    assert (row["state"], row["step"], row["stability"], row["reps"]) == ("learning", 0, expected.stability, 1)
    assert done == {"log_id": second["id"], "item_id": item, "rating": 1, "state": "learning",
                    "due": "2026-10-15T09:01:04+00:00", "superseded": first["id"]}  # fmt: skip
    assert cached(store, item) == {key: store.rebuild(item)[key] for key in CACHED}


def test_only_the_latest_auto_row_of_its_item_can_be_overridden(store, clock):
    item, other = item_id(store, "Saz 0"), item_id(store, "Saz 1")
    auto = store.review(item, fsrs.GOOD, rated_by="auto")
    clock.now = T0 + timedelta(minutes=10)
    learner_row = store.review(item, fsrs.GOOD, rated_by="learner")

    with pytest.raises(ValueError, match="latest automatic"):
        store.override(auto["log_id"], fsrs.HARD)
    with pytest.raises(ValueError, match="latest automatic"):
        store.override(learner_row["log_id"], fsrs.HARD)
    with pytest.raises(ValueError, match="no review"):
        store.override(99999, fsrs.HARD)
    later = store.review(item, fsrs.GOOD, rated_by="auto")
    with pytest.raises(ValueError, match=f"of item {other}"):
        store.override(later["log_id"], fsrs.HARD, item_id=other)
    with pytest.raises(ValueError, match="rating"):
        store.override(later["log_id"], 7)
    store.override(later["log_id"], fsrs.HARD)
    with pytest.raises(ValueError, match="latest automatic"):
        store.override(later["log_id"], fsrs.EASY)

    assert [row["superseded"] for row in log_rows(store)] == [0, 0, 1, 0]


def test_rebuild_of_an_item_without_valid_rows_makes_it_new_again(store):
    item = item_id(store, "Saz 0")
    store.review(item, fsrs.GOOD)
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute("UPDATE review_log SET superseded = 1")

    row = store.rebuild(item)

    assert (row["state"], row["step"], row["stability"], row["due"], row["reps"]) == (
        "new", None, None, row["created"], 0
    )  # fmt: skip
    with pytest.raises(ValueError):
        store.rebuild(99999)


def test_a_clock_that_goes_back_still_logs_in_review_order(store, clock):
    item = item_id(store, "Saz 0")
    store.review(item, fsrs.GOOD)
    clock.now = T0 - timedelta(hours=1)
    store.review(item, fsrs.GOOD)
    before = cached(store, item)

    store.rebuild(item)

    assert cached(store, item) == before
    assert [row["at"] for row in log_rows(store)] == ["2026-10-15T09:00:00+00:00"] * 2


# --- match ---


@pytest.mark.parametrize(
    ("answer", "expected", "lod", "result"),
    [
        ("Ech wunnen zu Esch.", "ech wunnen zu esch", None, True),
        ("  Ech   wunnen zu Esch!  ", "Ech wunnen zu Esch.", None, True),
        ("D’Fra ass do", "d'Fra ass do.", None, True),
        ("Ech hunn Honger", "Ech hat Honger", FakeLod(), True),
        ("Ech hunn Honger", "Ech hat Honger", None, False),
        ("Ech hunn Honger", "Ech hat Honger elo", FakeLod(), False),
        ("Ech hunn Duuscht", "Ech hat Honger", FakeLod(), False),
        ("", "Ech hat Honger", FakeLod(), False),
    ],
)
def test_matches(answer, expected, lod, result):
    assert matches(answer, expected, lod) is result


# --- routes ---


@pytest.fixture
def seeded(settings):
    seed(settings.data_dir)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def first_item(client) -> dict:
    return client.get("/api/review/next").json()["item"]


def form_of(client, item: dict) -> str:
    return client.app.state.learner.item(item["id"])["form"]


def test_next_on_the_migrated_data_shows_one_item_without_its_form_and_10_new_on_day_one(seeded):
    body = seeded.get("/api/review/next").json()

    assert set(body["item"]) == {"id", "gloss", "hint", "source", "state"}
    assert "form" not in body["item"] and "Saz 0" not in json.dumps(body)
    assert body["item"]["gloss"] == "Sentence 0"
    assert body["counts"] == {"due": 0, "new_available": 10, "in_review": 0, "total": 13}


def test_next_with_nothing_to_review_has_no_item(client):
    assert client.get("/api/review/next").json() == {
        "item": None,
        "counts": {"due": 0, "new_available": 0, "in_review": 0, "total": 0},
        "next_due": None,
    }


def test_a_typed_match_is_rated_good_on_its_own(seeded):
    item = first_item(seeded)

    body = seeded.post("/api/review/answer", data={"item_id": item["id"], "text": "saz 0!", "raw_text": "sas 0"}).json()

    assert (body["matched"], body["expected"], body["rating"], body["state"]) == (True, "Saz 0", 3, "learning")
    assert isinstance(body["log_id"], int) and body["due"]
    [row] = log_rows(seeded.app.state.learner)
    assert (row["rated_by"], row["matched"], row["mode"], row["heard"], row["confirmed"]) == (
        "auto", 1, "typed", "sas 0", "saz 0!"
    )  # fmt: skip


def test_a_match_through_lod_uses_the_apps_lod(seeded):
    seeded.app.state.lod = FakeLod()
    store = seeded.app.state.learner
    item = item_id(store, "Ech wunnen zu Esch.")
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute("UPDATE item SET form = 'Ech hat Honger' WHERE id = ?", (item,))

    body = seeded.post("/api/review/answer", data={"item_id": item, "text": "Ech hunn Honger", "mode": "spoken"}).json()

    assert body["matched"] is True


def test_a_mismatch_writes_nothing_and_the_learners_rating_is_then_logged(seeded):
    item = first_item(seeded)

    body = seeded.post("/api/review/answer", data={"item_id": item["id"], "text": "eppes anescht"}).json()

    assert body == {"matched": False, "expected": "Saz 0", "log_id": None}
    assert log_rows(seeded.app.state.learner) == []
    rated = seeded.post(
        "/api/review/rate", data={"item_id": item["id"], "rating": "2", "text": "eppes anescht", "mode": "spoken"}
    ).json()
    [row] = log_rows(seeded.app.state.learner)
    assert (rated["rating"], rated["log_id"], rated["superseded"]) == (2, row["id"], None)
    assert (row["rated_by"], row["matched"], row["rating"], row["mode"]) == ("learner", 0, 2, "spoken")


def test_rate_with_the_log_id_of_the_auto_row_supersedes_it(seeded):
    item = first_item(seeded)
    auto = seeded.post("/api/review/answer", data={"item_id": item["id"], "text": "Saz 0"}).json()

    body = seeded.post(
        "/api/review/rate", data={"item_id": item["id"], "rating": "1", "text": "Saz 0", "log_id": auto["log_id"]}
    ).json()

    first, second = log_rows(seeded.app.state.learner)
    assert (first["id"], first["superseded"]) == (auto["log_id"], 1)
    assert (second["rated_by"], second["rating"], second["superseded"]) == ("learner", 1, 0)
    assert (body["superseded"], body["log_id"], body["state"]) == (auto["log_id"], second["id"], "learning")


@pytest.mark.parametrize("rating", ["0", "5", "-1", "x", "2.5"])
def test_a_bad_rating_is_a_400_and_writes_nothing(seeded, rating):
    item = first_item(seeded)

    response = seeded.post("/api/review/rate", data={"item_id": item["id"], "rating": rating})

    assert response.status_code == 400
    assert log_rows(seeded.app.state.learner) == []


def test_an_unknown_item_a_bad_mode_an_empty_answer_or_a_foreign_log_id_is_a_400(seeded):
    item = first_item(seeded)
    other = item_id(seeded.app.state.learner, "Saz 5")
    auto = seeded.post("/api/review/answer", data={"item_id": item["id"], "text": "Saz 0"}).json()

    for path, data in [
        ("/api/review/answer", {"item_id": 99999, "text": "Saz 0"}),
        ("/api/review/answer", {"item_id": item["id"], "text": "  "}),
        ("/api/review/answer", {"item_id": item["id"], "text": "Saz 0", "mode": "sung"}),
        ("/api/review/rate", {"item_id": 99999, "rating": "3"}),
        ("/api/review/rate", {"item_id": other, "rating": "3", "log_id": auto["log_id"]}),
        ("/api/review/rate", {"item_id": item["id"], "rating": "3", "log_id": 99999}),
    ]:
        response = seeded.post(path, data=data)
        assert response.status_code == 400, (path, data, response.text)
        assert response.json()["error"]

    assert len(log_rows(seeded.app.state.learner)) == 1


def test_without_the_learner_store_the_routes_are_a_503(seeded):
    state = seeded.app.state
    state.learner.close()
    state.learner, state.learner_detail = None, "the database fails its check"

    for response in (
        seeded.get("/api/review/next"),
        seeded.post("/api/review/answer", data={"item_id": 1, "text": "Saz 0"}),
        seeded.post("/api/review/rate", data={"item_id": 1, "rating": "3"}),
    ):
        assert response.status_code == 503
        assert response.json()["error"] == "the learner store is not open: the database fails its check"
