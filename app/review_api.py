"""The routes under /api/review: the next item to say, the learner's answer, the learner's rating.

The learner sees the English gloss and says or types the Luxembourgish; `text` is the answer
the learner confirmed, `raw_text` what speech-to-text heard. A confirmed answer that matches
the form is rated Good on its own; after a mismatch nothing is logged until the learner rates.
"""

from __future__ import annotations

from fastapi import APIRouter, Form, Request

from app.fsrs import GOOD, checked_rating
from app.learner_store import REVIEW_MODES, LearnerStore, matches
from app.orchestrator import TurnError
from app.practice_store import StoreError

router = APIRouter(prefix="/api/review")


def learner(state) -> LearnerStore:
    if state.learner is None:
        raise TurnError(
            503, f"the learner store is not open: {state.learner_detail}", "look at /api/health and restart the app"
        )
    return state.learner


def known_item(store: LearnerStore, item_id: int) -> dict:
    item = store.item(item_id)
    if item is None:
        raise TurnError(400, f"there is no review item {item_id}", "ask /api/review/next for an item")
    return item


def checked_mode(mode: str) -> str:
    if mode not in REVIEW_MODES:
        raise TurnError(400, f"unknown mode {mode!r}; known: {', '.join(REVIEW_MODES)}")
    return mode


def written(write) -> dict:
    try:
        return write()
    except ValueError as exc:
        raise TurnError(400, str(exc)) from exc
    except StoreError as exc:
        raise TurnError(500, str(exc), "look at /api/health; the review can be sent again") from exc


@router.get("/next")
def review_next(request: Request) -> dict:
    """The first item of the queue without its form, which the learner is to say, and the counts;
    without an item, `next_due` says when the queue has one again (null: never, as it stands)."""
    store = learner(request.app.state)
    try:
        queue = store.queue(limit=1)
        counts = store.review_counts()
        next_due = None if queue else store.next_due()
    except StoreError as exc:
        raise TurnError(500, str(exc), "look at /api/health") from exc
    item = queue[0] if queue else None
    return {
        "item": item and {key: item[key] for key in ("id", "gloss", "hint", "source", "state")},
        "counts": counts,
        "next_due": next_due,
    }


@router.post("/answer")
def review_answer(
    request: Request,
    item_id: int = Form(...),
    text: str = Form(""),
    raw_text: str = Form(""),
    mode: str = Form("typed"),
) -> dict:
    """Rates a confirmed answer that matches the form Good; a mismatch writes nothing."""
    state = request.app.state
    store = learner(state)
    mode = checked_mode(mode)
    if not text.strip():
        raise TurnError(400, "no answer", "confirm or type your answer")
    item = known_item(store, item_id)
    if not matches(text, item["form"], state.lod):
        return {"matched": False, "expected": item["form"], "log_id": None}
    done = written(
        lambda: store.review(
            item_id, GOOD, mode=mode, heard=raw_text.strip(), confirmed=text.strip(), matched=True, rated_by="auto"
        )
    )
    return {
        "matched": True,
        "expected": item["form"],
        "rating": GOOD,
        "log_id": done["log_id"],
        "due": done["due"],
        "state": done["state"],
    }


@router.post("/rate")
def review_rate(
    request: Request,
    item_id: int = Form(...),
    rating: str = Form(...),
    text: str = Form(""),
    raw_text: str = Form(""),
    mode: str = Form("typed"),
    log_id: int | None = Form(None),
) -> dict:
    """The learner's rating 1..4: with `log_id`, in place of that automatic Good of the item;
    without, of the answer that did not match."""
    store = learner(request.app.state)
    mode = checked_mode(mode)
    try:
        value = checked_rating(int(rating))
    except ValueError as exc:
        fix = "send 1 Again, 2 Hard, 3 Good or 4 Easy"
        raise TurnError(400, f"a rating is 1, 2, 3 or 4, not {rating!r}", fix) from exc
    known_item(store, item_id)
    if log_id is not None:
        done = written(lambda: store.override(log_id, value, item_id=item_id))
    else:
        done = written(
            lambda: store.review(
                item_id, value, mode=mode, heard=raw_text.strip(), confirmed=text.strip(), matched=False,
                rated_by="learner",
            )
        )
    return {
        "rating": value,
        "log_id": done["log_id"],
        "due": done["due"],
        "state": done["state"],
        "superseded": done.get("superseded"),
    }
