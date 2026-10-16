"""The HTTP routes under /api: session, stt, chat, tts, turn, voice, mix, phrases, practice.

`/api/turn` is the one-call path of the page; the fine-grained routes let a client run
speech-to-text or text-to-speech on its own. The routes under `/api/practice` run the
practice talk, whose state is kept on the session, and look back on it when it ends or the
learner stops; each confirmed answer is kept on this PC, `/api/practice/saved` counts what
is kept (`DELETE /api/learner` deletes it with everything else), and `/api/practice/dispute`
marks a kept answer whose correction the learner thinks was wrong. The routes are plain
functions, so FastAPI runs them in its thread pool and a slow turn does not block the event loop.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, File, Form, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.content import Photo, Question
from app.engines.base import VOICE_RANGES, LLMError, Segment, WavAudio, voice_tuning
from app.engines.stt_faster_whisper import suffix_for
from app.orchestrator import AUTH_FIX, MIX_LEVELS, Session, TurnError, TurnResult, checked_mix
from app.practice import (
    TALK_SETTINGS,
    Correction,
    PracticeFlow,
    PracticeState,
    PracticeTurn,
    WrapUp,
    add_sentences,
    answer_record,
    facts_text,
    is_skip,
    judge_answer,
    wrap_up_facts,
    wrap_up_lines,
)
from app.practice_store import NOT_KEPT, StoreError, is_session_folder
from app.pronunciation_store import MAX_RESPELL, checked_respell

log =logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
DEFAULT_MIME = "audio/webm"
# What the page's "Play sample" buttons speak.
VOICE_SAMPLES = {"lb": "Gudde Moien! Wéi geet et dir haut?", "en": "Good morning! How are you today?"}
PRACTICE_FILE = "app/content/practice.toml"

router = APIRouter(prefix="/api")
practice_router = APIRouter(prefix="/api/practice")


class ChatIn(BaseModel):
    session_id: str
    text: str
    mode: str = "auto"
    stt_ms: int = 0
    duration_ms: int = 0  # how long the learner spoke the message; 0 when it was typed
    # Any type, so that a level that is no whole number is the 400 of `checked_mix`, not a 422.
    mix: Any = None


class SegmentIn(BaseModel):
    lang: str
    text: str


class TtsIn(BaseModel):
    segments: list[SegmentIn]
    voice: dict | None = None
    # {word: respelling} spoken instead of the pronunciation table for this request; nothing is saved.
    respell: dict[str, str] | None = None


def turn_error_response(request: Request, error: TurnError) -> JSONResponse:
    """The exception handler `app.main` registers for TurnError."""
    return JSONResponse({"error": error.error, "fix": error.fix}, status_code=error.status)


def find_session(state, session_id: str) -> Session:
    session = state.sessions.get(session_id)
    if session is None:
        raise TurnError(404, "unknown session", "reload the page")
    return session


def read_upload(audio: UploadFile) -> bytes:
    data = audio.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        megabytes = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise TurnError(413, f"the recording is larger than {megabytes} MB", "record a shorter message")
    return data


def turn_json(result: TurnResult) -> dict:
    """A turn without its audio. A segment has a `gloss` only when the turn found one for it."""
    return {
        "heard": True,
        "transcript": result.transcript,
        "mode": result.mode,
        "mix": result.mix,
        "segments": [
            {**asdict(segment), **({"gloss": gloss} if gloss else {})}
            for segment, gloss in zip(result.reply.segments, result.glosses, strict=True)
        ],
        "new_phrases": [
            {"lb": phrase.get("lb", ""), "en": phrase.get("en", "")} for phrase in result.reply.new_phrases
        ],
        "claims": result.claims,
        "phrases_added": result.phrases_added,
        "timings": result.timings,
        "shape_warning": result.shape_warning,
    }


def checked_voice(settings, voice):
    """`voice` as the engines take it; a malformed one is a 400 whatever the engine."""
    try:
        voice_tuning(settings, voice)
    except ValueError as exc:
        fix = "send rate, pitch and volume as numbers per language (lb, en), or leave `voice` out"
        raise TurnError(400, f"bad voice settings: {exc}", fix) from exc
    return voice


def checked_respell_in(respell):
    """`respell` as the engines take it; a malformed one is a 400 whatever the engine."""
    if respell is None:
        return None
    try:
        return checked_respell(respell)
    except ValueError as exc:
        fix = f"send `respell` as {{word: respelling}} with at most {MAX_RESPELL} words, or leave it out"
        raise TurnError(400, f"bad respelling: {exc}", fix) from exc


def voice_override(settings, voice: str | None):
    """The `voice` form field as the engines take it; None when the field is empty."""
    if not voice:
        return None
    try:
        override = json.loads(voice)
    except ValueError as exc:
        raise TurnError(400, f"bad voice settings: not JSON: {exc}", "send `voice` as a JSON object") from exc
    return checked_voice(settings, override)


def synthesize(tts, segments: list[Segment], voice=None, respell=None) -> WavAudio:
    try:
        if respell:
            return tts.synthesize(segments, voice, respell=respell)
        return tts.synthesize(segments, voice)
    except Exception as exc:
        raise TurnError(500, f"tts engine ({tts.name}) failed: {type(exc).__name__}: {exc}") from exc


def audio_json(audio: WavAudio) -> dict:
    return {
        "audio_wav_base64": base64.b64encode(audio.wav_bytes).decode("ascii"),
        "sample_rate": audio.sample_rate,
        "segment_offsets": audio.segment_offsets,
    }


def practice_flow(state) -> PracticeFlow:
    if state.practice is None:
        raise TurnError(
            500,
            f"the practice questions ({PRACTICE_FILE}) cannot be used: {state.practice_detail}",
            "correct the file and restart the app",
        )
    return PracticeFlow(state.practice)


def judged(
    state,
    talk: PracticeState,
    question: Question,
    text: str,
    help_used: bool,
    photo: Photo | None = None,
    raw_text: str = "",
) -> tuple[Correction | None, str]:
    """(what the tutor makes of the answer `text` to `question`, "") or (None, why there is no judgement).

    `photo` is the photo the question is about; the judgement gets its description. `raw_text`
    is what was heard, "" for a typed answer: the label of the correction takes it, and the
    acceptability encoder of the app. A missing
    login is the 503 of a chat turn, since every answer would fail on it. Any other failure
    leaves the answer without a judgement, so that the talk goes on.
    """
    try:
        rules = state.orchestrator.rules
        picture = photo.description_en if photo else ""
        correction = judge_answer(
            state.llm, state.lod, rules, talk, question, text, help_used, picture, state.acceptability, raw_text
        )
        return correction, ""
    except LLMError as error:
        problem = error.detail or str(error)
        if error.kind == "auth":
            raise TurnError(503, problem, fix=AUTH_FIX) from error
    except Exception as exc:
        problem = f"llm engine ({state.llm.name}) failed: {type(exc).__name__}: {exc}"
    log.warning("practice: no judgement of the answer to %s: %s", question.id, problem)
    return None, problem


def recorded(state, method: str, *args):
    """What `method` of the learner store returns for `args`; None without a store or when it fails,
    which is only logged: recording never stops a route."""
    if state.learner is None:
        return None
    try:
        return getattr(state.learner, method)(*args)
    except Exception as exc:
        log.warning("the learner store has not recorded %s: %s: %s", method, type(exc).__name__, exc)
        return None


def judge_model(state) -> str | None:
    """The model that judges practice answers when the engine says which: the Claude CLI's
    LOCI_LLM_MODEL, or the model the provider engine's judge role is routed to; None otherwise."""
    llm = state.llm
    if llm.name == "claude_cli":
        return state.settings.llm_model
    config = getattr(llm, "config", None)
    if llm.name != "provider" or config is None:
        return None
    route = config.roles["judge"]
    entry = config.providers.get(route.provider)
    return route.model or (entry.model if entry else "") or None


def kept(
    state, session: Session, question: Question, recording: bytes, mime: str, duration_ms: int = 0
) -> tuple[dict | None, dict | None, str]:
    """(what is kept on this PC, the last answer of the session's talk now with it, where that
    answer is: `folder` and `line`, "") or (None, None, why that answer is not kept).

    The answer is the one to `question`; `recording` is its upload of the type `mime`, empty
    when there is none, and `duration_ms` how long it lasts. A kept answer is recorded in the
    learner store too. A store that fails does not stop the talk.
    """
    talk = session.practice
    record = answer_record(talk, question, talk.answers[-1])
    try:
        where = state.practice_store.save(session.id, record, recording, mime)
    except StoreError as error:
        log.warning("practice: %s", error)
        return None, None, str(error)
    folder, line = where["folder"], where["line"]
    audio_file = f"{line}{suffix_for(mime)}" if recording else None
    judge = judge_model(state) if record["verdict"] else None
    recorded(state, "answer_kept", session.id, folder, line, record, audio_file, duration_ms, judge, question.en)
    try:
        return state.practice_store.summary(), where, ""
    except StoreError as error:
        log.warning("practice: %s", error)
        return None, None, str(error)


def store_error(error: StoreError) -> TurnError:
    return TurnError(500, str(error), "look at that folder, or point LOCI_DATA_DIR at another data folder")


def looked_back(state, flow: PracticeFlow, talk: PracticeState) -> WrapUp:
    """The look back on `talk`: what code knows of it and, when it has an answer, the tutor's
    sentences from one LLM call.

    A missing login is the 503 of a chat turn, and the look back can be asked for again. Any
    other failure leaves the look back without the sentences.
    """
    wrap_up = wrap_up_facts(flow, talk, state.lod)
    if not wrap_up.answered:
        return wrap_up
    text = None
    try:
        text = state.llm.wrap_up(facts_text(flow, talk, wrap_up))
    except LLMError as error:
        problem = error.detail or str(error)
        if error.kind == "auth":
            raise TurnError(503, problem, fix=AUTH_FIX) from error
        log.warning("practice: the look back has no sentences: %s", problem)
    except Exception as exc:
        log.warning(
            "practice: the look back has no sentences: llm engine (%s) failed: %s: %s",
            state.llm.name,
            type(exc).__name__,
            exc,
        )
    add_sentences(wrap_up, flow, talk, text)
    return wrap_up


def spoken_wrap_up(state, wrap_up: WrapUp, lines: list[tuple[Segment, str]], voice) -> dict:
    """A look back with its audio. A segment has a `role`: the name of the sentence it is a
    piece of, "once_more", "better" for a better version said again, "end" for the goodbye,
    or "line" for the one line of a talk without an answer."""
    return {
        "part": "done",
        "wrapup": {
            "answered": wrap_up.answered,
            "skipped": wrap_up.skipped,
            "help_questions": wrap_up.help_questions,
            "corrections": wrap_up.corrections,
            "unknown_words": wrap_up.unknown_words,
            "photo_described": wrap_up.photo_described,
            "sentences": wrap_up.sentences,
            "llm_failed": wrap_up.llm_failed,
        },
        "segments": [{**asdict(segment), "role": role} for segment, role in lines],
        **audio_json(synthesize(state.tts, [segment for segment, _ in lines], voice)),
    }


def photo_json(photo: Photo) -> dict:
    """A photo as the page shows it. `url` is relative to the page; `alt` says what the photo shows."""
    return {
        "id": photo.id,
        "url": photo.image,
        "alt": photo.description_en,
        "caption": f"{photo.author}, {photo.licence}",
    }


def spoken_practice_turn(
    state,
    turn: PracticeTurn,
    voice,
    feedback_error: str = "",
    saved: dict | None = None,
    save_error: str = "",
    where: dict | None = None,
) -> dict:
    """A practice turn with its audio. A segment has a `gloss` when the content gives its English.

    `correction` is what the tutor made of the answer the turn follows, None when it was not
    judged; `feedback_error` then says why, "" when no judgement was due. Its `label`,
    `category`, `source` and `checks` are set on a "fix" or an "english" only; `alternative_lb`
    is another way to say a "good" answer, shown and not spoken. `saved` is what is
    kept on this PC once that answer is with it, None when the turn kept none; `save_error`
    then says why, "" when there was no answer to keep. `kept` is where that answer is kept,
    `folder` and `line`, for /api/practice/dispute; None as `saved` is. `photos` are the photos to pick
    from, on a "choose" turn only; `photo` is the picked one while its questions are open.
    """
    question = turn.question
    correction = turn.correction
    return {
        "part": turn.part,
        "kind": turn.kind,
        "question": question
        and {
            "id": question.id,
            "lb": question.lb,
            "en": question.en,
            "stage": question.stage,
            "index": turn.index,
            "total": turn.total,
        },
        "segments": [
            {**asdict(segment), **({"gloss": gloss} if gloss else {})}
            for segment, gloss in zip(turn.segments, turn.glosses, strict=True)
        ],
        "correction": correction
        and {
            "verdict": correction.verdict,
            "quote": correction.quote,
            "better_lb": correction.better_lb,
            "why_en": correction.why_en,
            "claims": correction.claims,
            "label": correction.label,
            "category": correction.category,
            "source": correction.source,
            "alternative_lb": correction.alternative_lb,
            "checks": correction.checks,
        },
        "feedback_error": feedback_error,
        "saved": saved,
        "save_error": save_error,
        "kept": where,
        "photos": [photo_json(photo) for photo in turn.photos],
        "photo": turn.photo and photo_json(turn.photo),
        **audio_json(synthesize(state.tts, turn.segments, voice)),
    }


@router.post("/session")
def new_session(request: Request) -> dict:
    state = request.app.state
    session = state.sessions.new()
    recorded(state, "session_seen", session.id, "")
    return {"session_id": session.id}


@router.post("/stt")
def stt(request: Request, audio: UploadFile = File(...)) -> dict:
    transcript, stt_ms = request.app.state.orchestrator.transcribe(
        read_upload(audio), audio.content_type or DEFAULT_MIME
    )
    return {**asdict(transcript), "stt_ms": stt_ms}


@router.post("/chat")
def chat(request: Request, body: ChatIn) -> dict:
    state = request.app.state
    session = find_session(state, body.session_id)
    return turn_json(state.orchestrator.turn(session, body.text, body.mode, body.stt_ms, body.mix, body.duration_ms))


@router.post("/tts")
def tts(request: Request, body: TtsIn) -> Response:
    if not body.segments:
        raise TurnError(400, "no segments to speak")
    state = request.app.state
    voice = checked_voice(state.settings, body.voice)
    respell = checked_respell_in(body.respell)
    audio = synthesize(state.tts, [Segment(item.lang, item.text) for item in body.segments], voice, respell)
    offsets = json.dumps(audio.segment_offsets, separators=(",", ":"))
    return Response(audio.wav_bytes, media_type="audio/wav", headers={"X-Segment-Offsets": offsets})


@router.post("/turn")
def turn(
    request: Request,
    session_id: str = Form(...),
    mode: str = Form("auto"),
    text: str | None = Form(None),
    stt_ms: int = Form(0),
    audio: UploadFile | None = File(None),
    voice: str | None = Form(None),
    mix: str | None = Form(None),
    duration_ms: int = Form(0),
) -> dict:
    """`duration_ms` is how long the learner spoke the message, as the page knows it from /api/stt;
    0 for a typed message. A recording sent here is not measured."""
    state = request.app.state
    session = find_session(state, session_id)
    level = checked_mix(mix, state.settings.chat_mix)
    override = voice_override(state.settings, voice)
    if text is None or not text.strip():
        if audio is None:
            raise TurnError(400, "no message", "send a recording as `audio` or a message as `text`")
        transcript, stt_ms = state.orchestrator.transcribe(
            read_upload(audio), audio.content_type or DEFAULT_MIME
        )
        text = transcript.text
        if not text.strip():
            return {"heard": False, "transcript": "", "timings": {"stt_ms": stt_ms}}
    result = state.orchestrator.turn(session, text, mode, stt_ms, level, duration_ms)
    spoken = result.audio
    if override:
        # The orchestrator has spoken the reply with the settings; it is spoken again with the
        # override. Synthesis takes about 0.05 s per segment, so the second pass is not felt.
        spoken = synthesize(state.tts, result.reply.segments, override)
    return {**turn_json(result), **audio_json(spoken)}


@router.get("/voice")
def voice_info(request: Request) -> dict:
    return {
        "defaults": voice_tuning(request.app.state.settings),
        "ranges": {key: list(bounds) for key, bounds in VOICE_RANGES.items()},
        "samples": VOICE_SAMPLES,
    }


@router.get("/mix")
def mix_info(request: Request) -> dict:
    return {
        "default": checked_mix(None, request.app.state.settings.chat_mix),
        "levels": [dict(level) for level in MIX_LEVELS],
    }


@router.get("/phrases")
def phrases(request: Request) -> dict:
    entries = request.app.state.phrases.latest()
    return {"total": len(entries), "phrases": entries}


@practice_router.get("/topics")
def practice_topics(request: Request) -> dict:
    topics = practice_flow(request.app.state).content.topics
    return {
        "topics": [
            {"id": topic.id, "name_lb": topic.name_lb, "name_en": topic.name_en, "questions": len(topic.questions)}
            for topic in topics
        ]
    }


@practice_router.post("/start")
def practice_start(
    request: Request,
    session_id: str = Form(...),
    topic_id: str = Form(...),
    voice: str | None = Form(None),
    setting: str = Form("relaxed"),
) -> dict:
    """Opens a topic on the session, whatever talk it held: the tutor speaks first.

    The new talk knows the warm-up questions the session's talks have asked and the photos
    they have described, and draws others first. `setting` is kept with each of its answers.
    """
    state = request.app.state
    session = find_session(state, session_id)
    flow = practice_flow(state)
    override = voice_override(state.settings, voice)
    if flow.content.topic(topic_id) is None:
        known = ", ".join(topic.id for topic in flow.content.topics)
        raise TurnError(400, f"unknown topic {topic_id!r}; known: {known}")
    if setting not in TALK_SETTINGS:
        raise TurnError(400, f"unknown setting {setting!r}; known: {', '.join(TALK_SETTINGS)}")
    last = session.practice
    talk = PracticeState(
        setting=setting,
        photos_described=list(last.photos_described) if last else [],
        warmups_asked=list(last.warmups_asked) if last else [],
    )
    turn = flow.start(talk, topic_id)
    session.practice = talk
    return spoken_practice_turn(state, turn, override)


@practice_router.post("/answer")
def practice_answer(
    request: Request,
    session_id: str = Form(...),
    text: str = Form(""),
    raw_text: str = Form(""),
    help_used: bool = Form(False),
    audio: UploadFile | None = File(None),
    voice: str | None = Form(None),
    duration_ms: int = Form(0),
) -> dict:
    """Takes the answer the learner confirmed: `text`. `raw_text` is what speech-to-text heard.

    `audio` is the recording of the answer, `duration_ms` its length as /api/stt gave it (0 for
    a typed answer); this route does not transcribe it. An answer that
    is no skip is judged by the LLM before the talk moves on, and is then kept on this PC with
    its recording and what the tutor made of it.
    """
    state = request.app.state
    session = find_session(state, session_id)
    flow = practice_flow(state)
    override = voice_override(state.settings, voice)
    talk = session.practice
    if talk is None:
        raise TurnError(409, "no practice topic is open", "start a topic first")
    if talk.part == "done":
        raise TurnError(409, "this practice talk is over", "start a topic to talk again")
    if talk.part == "photo" and not talk.photo_id:
        raise TurnError(409, "no picture is chosen", "pick one of the pictures first")
    if not text.strip():
        raise TurnError(400, "no answer", "confirm or type your answer")
    text = text.strip()
    if is_skip(text):
        return spoken_practice_turn(state, flow.step(talk, text, help_used, raw_text.strip()), override)
    recording = read_upload(audio) if audio is not None else b""
    mime = (audio.content_type if audio is not None else "") or DEFAULT_MIME
    question = flow.open_question(talk)
    correction, problem = judged(state, talk, question, text, help_used, flow.photo(talk), raw_text.strip())
    turn = flow.step(talk, text, help_used, raw_text.strip(), correction)
    saved, where, save_problem = kept(state, session, question, recording, mime, duration_ms)
    return spoken_practice_turn(state, turn, override, problem, saved, save_problem, where)


@practice_router.post("/photo")
def practice_photo(
    request: Request,
    session_id: str = Form(...),
    photo_id: str = Form(...),
    voice: str | None = Form(None),
) -> dict:
    """Takes the picture the learner picked of those on offer: the tutor asks the first question about it."""
    state = request.app.state
    session = find_session(state, session_id)
    flow = practice_flow(state)
    override = voice_override(state.settings, voice)
    talk = session.practice
    if talk is None:
        raise TurnError(409, "no practice topic is open", "start a topic first")
    if talk.part == "done":
        raise TurnError(409, "this practice talk is over", "start a topic to talk again")
    if talk.part != "photo":
        raise TurnError(409, "no picture is on offer yet", "answer the questions of the topic first")
    if talk.photo_id:
        raise TurnError(409, "a picture is already chosen", "answer the question about it")
    if photo_id not in talk.photos_offered:
        raise TurnError(400, f"picture {photo_id!r} is not on offer; on offer: {', '.join(talk.photos_offered)}")
    return spoken_practice_turn(state, flow.choose(talk, photo_id), override)


@practice_router.post("/stop")
def practice_stop(
    request: Request,
    session_id: str = Form(...),
    voice: str | None = Form(None),
) -> dict:
    """Ends the talk where it stands and looks back on it. A talk that ran to its end is
    looked back on too; its goodbye is said, so the look back has none. Once per talk."""
    state = request.app.state
    session = find_session(state, session_id)
    flow = practice_flow(state)
    override = voice_override(state.settings, voice)
    talk = session.practice
    if talk is None:
        raise TurnError(409, "no practice topic is open", "start a topic first")
    if talk.wrapped:
        raise TurnError(409, "this practice talk is over", "start a topic to talk again")
    wrap_up = looked_back(state, flow, talk)
    lines = wrap_up_lines(flow.content.tutor, wrap_up, goodbye=talk.part != "done")
    body = spoken_wrap_up(state, wrap_up, lines, override)
    talk.part, talk.wrapped = "done", True
    return body


@practice_router.post("/dispute")
def practice_dispute(
    request: Request,
    session_id: str = Form(...),
    folder: str = Form(...),
    line: int = Form(...),
    note: str = Form(""),
) -> dict:
    """Marks an answer of the session kept on this PC, at `folder` and `line` as its turn's `kept`
    says, as one whose correction the learner thinks was wrong; a second time changes nothing.
    The learner store's copy of that correction, when it has one, is marked too."""
    state = request.app.state
    session = find_session(state, session_id)
    if not is_session_folder(folder, session.id):
        raise TurnError(403, "not your answer", "dispute an answer of this talk")
    try:
        answer = state.practice_store.dispute(folder, line, note)
    except StoreError as error:
        if str(error).startswith(NOT_KEPT):
            raise TurnError(404, str(error), "this answer is not kept on this PC") from error
        raise store_error(error) from error
    if state.learner is not None:
        try:
            state.learner.mark_disputed(folder, line)
        except Exception as exc:
            log.warning("practice: the learner store has not marked the dispute: %s: %s", type(exc).__name__, exc)
    return {"folder": folder, "line": line, "disputed": answer["disputed"]}


@practice_router.get("/saved")
def practice_saved(request: Request) -> dict:
    """How much of the practice answers is kept on this PC, and where."""
    try:
        return request.app.state.practice_store.summary()
    except StoreError as error:
        raise store_error(error) from error
