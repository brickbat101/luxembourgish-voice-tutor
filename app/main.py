"""FastAPI application: engines, LOD store, orchestrator, practice and listening content, the
store of practice answers and the learner store built in the lifespan, the routes, the page."""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import replace
from html import escape
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app import api, learner_api, listening_api, pronunciation_api, review_api
from app.config import ROOT, Settings
from app.content import PHOTOS_PATH, PRACTICE_PATH, Photo, PracticeContent, load_photos, load_practice
from app.content.listening import LISTENING_PATH, ListeningItem, load_listening
from app.engines.factory import KINDS, build_engine
from app.engines.tts_piper import Pronunciations
from app.guard import AccessGuard, load_token
from app.learner_store import DATABASE, LEARNER_FOLDER, PRACTICE_FOLDER, LearnerStore
from app.lod.store import LodStore
from app.orchestrator import Orchestrator, SessionStore, TurnError
from app.practice_store import PracticeStore, StoreError
from app.pronunciation_store import FLAGS_FILE, TABLE_FILE, FlagStore

log = logging.getLogger(__name__)

STATIC_DIR = ROOT / "static"  # the page, and under photos/ the images of practice mode
TOKEN_PLACEHOLDER = '<meta name="loci-token" content="">'


def engine_health(engine) -> dict:
    try:
        ready = bool(engine.ready())
        detail = str(getattr(engine, "detail", ""))
    except Exception as exc:
        ready, detail = False, f"{type(exc).__name__}: {exc}"
    return {"name": engine.name, "ready": ready, "detail": detail}


def open_lod(path: Path) -> tuple[LodStore | None, str]:
    """(the store, "") or (None, why not); the app runs without the store."""
    store = None
    try:
        store = LodStore(path)
        store.known("goen")  # a file that is not a store fails here, not in the first turn
    except (OSError, sqlite3.Error) as exc:
        if store is not None:
            store.close()
        return None, f"{type(exc).__name__}: {exc}"
    return store, ""


def open_learner(data_dir: Path, questions: dict[str, str]) -> tuple[LearnerStore | None, str]:
    """(the store, opened, imported and backed up today, "") or (None, why not); the app runs without it.

    A store that fails its check is left as it is on disk; the reason names the newest backup.
    """
    store = LearnerStore(data_dir / LEARNER_FOLDER, data_dir)
    try:
        store.open()
        store.migrate_files(questions)
        store.backup_daily()
    except (OSError, sqlite3.Error, StoreError) as exc:
        with contextlib.suppress(sqlite3.Error):
            store.close()
        detail = str(exc) if isinstance(exc, StoreError) else f"{type(exc).__name__}: {exc}"
        log.error("the learner store is not used: %s", detail)
        return None, detail
    return store, ""


def learner_health(store: LearnerStore | None, detail: str, data_dir: Path) -> dict:
    if store is None:
        return {"ready": False, "detail": detail, "path": str(data_dir / LEARNER_FOLDER / DATABASE)}
    try:
        counts = store.counts()
        detail = f"{counts['phrase']} phrases, {counts['answer']} answers, {counts['item']} review items"
    except (sqlite3.Error, StoreError) as exc:
        return {"ready": False, "detail": f"{type(exc).__name__}: {exc}", "path": str(store.path)}
    return {"ready": True, "detail": detail, "path": str(store.path)}


def practice_questions(practice: PracticeContent | None) -> dict[str, str]:
    """The English of every practice question by its id; {} without the practice content."""
    if practice is None:
        return {}
    questions = [*practice.warmup, *(question for topic in practice.topics for question in topic.questions)]
    return {question.id: question.en for question in questions}


def open_practice(path: Path) -> tuple[PracticeContent | None, str]:
    """(the practice content, "") or (None, what is wrong in the file); the app runs without it."""
    try:
        return load_practice(path), ""
    except ValueError as exc:
        return None, str(exc)
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"


def open_listening(path: Path) -> tuple[tuple[ListeningItem, ...] | None, str]:
    """(the listening items, "") or (None, what is wrong in the file); the app runs without them."""
    try:
        return load_listening(path), ""
    except ValueError as exc:
        return None, str(exc)
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"


def open_photos(path: Path, static_dir: Path) -> tuple[tuple[Photo, ...], str]:
    """(the photos of practice mode, "") or ((), why there are none); a talk then ends after its topic.

    A photo counts only with its image under `static_dir`, so one entry that is wrong or
    one image that is missing leaves the talks without the photo part.
    """
    try:
        photos = load_photos(path)
    except ValueError as exc:
        return (), str(exc)
    except OSError as exc:
        return (), f"{type(exc).__name__}: {exc}"
    missing = [photo.image for photo in photos if not (static_dir / photo.image).is_file()]
    if missing:
        return (), f"no image {', '.join(missing)} in {static_dir}; run: python scripts/download_photos.py"
    return photos, ""


def unknown_topics(photos: tuple[Photo, ...], practice: PracticeContent) -> list[Photo]:
    """The photos whose `topic` is set and is not the id of a topic of `practice`."""
    return [photo for photo in photos if photo.topic and practice.topic(photo.topic) is None]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = app.state
        for kind in KINDS:
            setattr(state, kind, build_engine(kind, settings))
        state.lod, state.lod_detail = open_lod(settings.data_dir / "lod" / "lod.sqlite")
        state.sessions = SessionStore()
        state.practice, state.practice_detail = open_practice(PRACTICE_PATH)
        photos, state.photos_detail = open_photos(PHOTOS_PATH, STATIC_DIR)
        if state.photos_detail:
            log.warning("practice talks have no photo part: %s", state.photos_detail)
        if state.practice is not None:
            state.practice = replace(state.practice, photos=photos)
            for photo in unknown_topics(photos, state.practice):
                log.warning("photo %r has the topic %r, which is not a practice topic", photo.id, photo.topic)
        state.listening, state.listening_detail = open_listening(LISTENING_PATH)
        if state.listening_detail:
            log.warning("the listening part is not used: %s", state.listening_detail)
        state.practice_store = PracticeStore(settings.data_dir / PRACTICE_FOLDER, settings.data_dir)
        # DELETE /api/learner opens the store again this way once everything is deleted.
        state.open_learner = lambda: open_learner(settings.data_dir, practice_questions(state.practice))
        state.learner, state.learner_detail = state.open_learner()
        state.orchestrator = Orchestrator(state.stt, state.tts, state.llm, state.lod, settings, learner=state.learner)
        state.phrases = state.orchestrator.phrases
        state.flags = FlagStore(settings.data_dir / FLAGS_FILE)
        state.pronunciations = Pronunciations(settings.data_dir / TABLE_FILE)
        yield
        if state.learner is not None:
            state.learner.close()
        if state.lod is not None:
            state.lod.close()

    app = FastAPI(title="Luxembourgish voice tutor", lifespan=lifespan)
    app.state.settings = settings
    token, warning = load_token(settings.data_dir, os.environ.get("LOCI_ACCESS_TOKEN"))
    app.state.access_token = token
    app.state.access_detail = warning

    @app.get("/api/health")
    def health() -> dict:
        engines = {kind: engine_health(getattr(app.state, kind)) for kind in KINDS}
        status = "ok" if all(engine["ready"] for engine in engines.values()) else "degraded"
        lod = {"ready": app.state.lod is not None, "detail": app.state.lod_detail}
        learner = learner_health(app.state.learner, app.state.learner_detail, settings.data_dir)
        access = {"persisted": not app.state.access_detail, "detail": app.state.access_detail}
        return {"status": status, "engines": engines, "lod": lod, "learner": learner, "access": access}

    app.include_router(api.router)
    app.include_router(api.practice_router)
    app.include_router(review_api.router)
    app.include_router(listening_api.router)
    app.include_router(learner_api.router)
    app.include_router(pronunciation_api.router)
    app.add_exception_handler(TurnError, api.turn_error_response)

    # Keep this mount last: routes registered after a mount at "/" are never reached.
    if STATIC_DIR.is_dir():

        @app.get("/", response_class=HTMLResponse)
        def page() -> HTMLResponse:
            html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
            if TOKEN_PLACEHOLDER not in html:
                raise RuntimeError(f"static/index.html has no {TOKEN_PLACEHOLDER}")
            meta = f'<meta name="loci-token" content="{escape(app.state.access_token)}">'
            html = html.replace(TOKEN_PLACEHOLDER, meta)
            return HTMLResponse(html, headers={"Cache-Control": "no-store"})

        @app.get("/index.html", include_in_schema=False)
        def page_without_token() -> RedirectResponse:
            return RedirectResponse("/", 307)

        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    # Added last, it wraps everything, the mount included.
    app.add_middleware(AccessGuard, token=token, port=settings.port)

    return app


app = create_app()
