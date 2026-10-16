"""The routes under /api/learner: what the learner has done so far, as the Progress block shows it,
and, for the Data block, what is kept on this PC, the export of all of it and its delete.

No streaks, goals or comparisons: totals, the corrections of the last four weeks and the
practice topics with how many of their questions were answered.
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.learner_data import chunks, delete_everything, export_name, export_zip, kept_bytes
from app.learner_store import LEARNER_FOLDER, PHRASES_FILE, PRACTICE_FOLDER, LearnerStore
from app.orchestrator import TurnError
from app.practice_store import ANSWERS_FILE, StoreError
from app.review_api import learner

router = APIRouter(prefix="/api/learner")
_deleting = threading.Lock()  # one delete at a time, so that one store is opened after it


def practice_topics(practice) -> list[tuple[str, str, tuple[str, ...]]]:
    """(id, English name, question ids) of every practice topic; [] without the practice content."""
    if practice is None:
        return []
    return [(topic.id, topic.name_en, tuple(question.id for question in topic.questions)) for topic in practice.topics]


@router.get("/progress")
def learner_progress(request: Request) -> dict:
    """Minutes spoken, phrases met, review items, corrections per week by category, topics touched."""
    state = request.app.state
    store = learner(state)
    try:
        return store.progress(topics=practice_topics(state.practice))
    except StoreError as exc:
        raise TurnError(500, str(exc), "look at /api/health") from exc


@router.get("/summary")
def learner_summary(request: Request) -> dict:
    """What is kept on this PC: phrases, answers, recordings and review items, the bytes of every
    learner file, and the folders and file they are in."""
    store = learner(request.app.state)
    try:
        counts = store.counts()
        files = store.kept_files()
    except (StoreError, OSError) as exc:
        raise TurnError(500, str(exc), "look at /api/health") from exc
    recordings = sum(name.startswith(PRACTICE_FOLDER + "/") and path.name != ANSWERS_FILE for name, path in files)
    data_dir = store.data_dir
    return {
        "phrases": counts["phrase"],
        "answers": counts["answer"],
        "recordings": recordings,
        "items": counts["item"],
        "bytes": kept_bytes(data_dir),
        "folders": [str(data_dir / LEARNER_FOLDER), str(data_dir / PRACTICE_FOLDER), str(data_dir / PHRASES_FILE)],
    }


@router.get("/export")
def learner_export(request: Request) -> StreamingResponse:
    """Everything kept on this PC as one ZIP: learner.json, learner.sqlite, originals/ and audio/."""
    store = learner(request.app.state)
    try:
        archive, exported_at = export_zip(store)
    except StoreError as exc:
        raise TurnError(500, str(exc), "look at /api/health, then export again") from exc
    return StreamingResponse(
        chunks(archive),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{export_name(exported_at)}"'},
    )


@router.delete("")
def learner_delete(request: Request) -> dict:
    """Deletes every learner file, checks by looking that none is left, and opens an empty learner
    store, so the app keeps recording. A talk that is open goes on."""
    state = request.app.state
    data_dir = state.settings.data_dir
    with _deleting:
        store = state.learner or LearnerStore(data_dir / LEARNER_FOLDER, data_dir)
        left, problems = delete_everything(store, state.practice_store, state.phrases)
        # A store whose folder was refused is still open; it is not used after this.
        with contextlib.suppress(sqlite3.Error, StoreError):
            store.close()
        state.learner, state.learner_detail = state.open_learner()
        state.orchestrator.learner = state.learner
    if left or problems:
        still = ", ".join(left) or "nothing"
        raise TurnError(
            500,
            f"not everything was deleted; still on this PC: {still}. " + " ".join(problems),
            "close any program that has these files open, then press Delete everything again",
        )
    return {"deleted": True, "left": []}
