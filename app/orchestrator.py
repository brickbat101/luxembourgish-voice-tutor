"""One conversation turn: mode, mix level, grounding, tutor reply, shape check, claims, speech, phrases.

The HTTP routes own the transport; this module owns the order of the stages, their timings
and the mapping of every failure to a `TurnError`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from app.config import ROOT
from app.engines.base import LLMError, Segment, Transcript, Turn, TutorReply, WavAudio, ends_sentence
from app.lod.store import LodStore

if TYPE_CHECKING:
    from app.learner_store import LearnerStore
    from app.practice import PracticeState

log = logging.getLogger(__name__)

MAX_HISTORY = 30
MAX_GROUNDING_WORDS = 12
MAX_WORDS = {"chat": 90, "teacher": 220}  # a reply stays under this many words
MODES = ("auto", "chat", "teacher")
# How much Luxembourgish a chat reply mixes into its English; `value` is the level.
MIX_LEVELS = (
    {"value": 0, "name": "Opening phrase only", "hint": "One Luxembourgish phrase to open, then English."},
    {"value": 1, "name": "A phrase here and there", "hint": "One or two short Luxembourgish phrases in the reply."},
    {"value": 2, "name": "Phrases and sentences", "hint": "About a third of the reply is Luxembourgish."},
    {"value": 3, "name": "Mostly Luxembourgish", "hint": "Simple Luxembourgish sentences; English where it gets hard."},
)
MIX_FIX = f"send `mix` as a whole number from 0 to {len(MIX_LEVELS) - 1}, or leave it out"
# What a chat reply at a mix level must hold after segment 0: per level the fewest "lb"
# segments, and the smallest and the largest share of its words that are in "lb" segments.
MIX_MIN_LB_SEGMENTS = {1: 1, 2: 1, 3: 1}
MIX_MIN_LB_SHARE = {1: 0.0, 2: 0.2, 3: 0.4}
MIX_MAX_LB_SHARE = {1: 0.3, 2: 1.0, 3: 1.0}
MAX_MIX_PHRASES = 40  # of the phrase memory, given to the model at a mix level
# The bare words "teacher" and "explain" also occur in day chat, so only these phrasings count.
TEACHER_TRIGGERS = re.compile(
    r"teacher mode"
    r"|\bhow (?:do|would|can) (?:you|i|we) (?:say|conjugate|form|spell|pronounce)\b"
    r"|\bhow (?:(?:would|do|can|could|should) (?:i|you|we|one)|does one) (?:say|describe|ask|tell|call|order"
    r"|greet|introduce|write|spell|pronounce|translate|express|answer|reply|respond|talk about|thank"
    r"|apologi[sz]e|count)\b"
    r"|\bhow (?:would|do|can|could|should) (?:i|you|we) \b[^.?!]{1,60}\bin luxembourgish\b"
    r"|conjugat|grammar|why is it|what does\b[^.?!]+\bmean"
    r"|\b(?:can|could|would|will) you explain|please explain|\A\s*explain\b"
    r"|\bhow (?:does|do|is|are) (?:the |this |that )?(?:word|verb|noun|adjective|sentence|phrase)\b"
    r"|\b(?:what|how)(?:['’]s| is| are| do you call)\b[^.?!]{0,60}\bin luxembourgish\b"
    r"|\b(?:word|verb) for\b|\bplural of\b|\bpast tense\b|\bdifference between\b",
    re.IGNORECASE,
)
# Runs of 3 or more letters, diacritics included; an apostrophe ends a token ("d'Bäckerei").
WORD = re.compile(r"[^\W\d_]{3,}")
ENGLISH_WORD = re.compile(r"[^\W\d_]+")
# The frame of a question, not what it asks about: articles, pronouns, conjunctions, question
# words, auxiliaries, prepositions and the words a learner asks with.
ENGLISH_STOP = frozenset(
    "a an the"
    " i me my you your he him his she her it its we us our they them their this that these those"
    " and or but how what which who why"
    " do does did can could would should will is are am"
    " in on at to of for with from by about as"
    " luxembourgish english say mean conjugate explain please word verb".split()
)
# Between the words that ask and the word asked about: "the word for dog", "the verb to be".
_LEAD = r"(?:(?:to|for|the|word|verb)\s+)*"
_ONE_WORD = r"""["'“‘«]?([^\W\d_]+)"""
# In each pattern group 1 is what the question asks about.
ASKED = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"""(?<![^\W\d_])["'“‘«]([^"'“”‘’«»]+)["'”’»](?![^\W\d_])""",  # a word or phrase in quotes
        rf"\b(?:say|conjugate|the\s+word|the\s+verb)\s+{_LEAD}{_ONE_WORD}",
        rf"""\bhow\s+(?:does|do)\s+{_LEAD}{_ONE_WORD}["'”’»]?\s+works?\b""",
        rf"\b(?:what\s+is|what['’]s|how\s+do\s+you\s+say)\s+{_LEAD}([^.?!]*?)\s+in\s+luxembourgish\b",
    )
)
# Group 1 is a word asked about as a verb.
VERB_ASKED = re.compile(rf"\b(?:conjugate|the\s+verb)\s+{_LEAD}{_ONE_WORD}", re.IGNORECASE)
# An article or pronoun in front of a form: "d'Waasser", "du gees".
LEADING = re.compile(
    r"\A(?:d['’]\s*|(?:de|den|dem|der|déi|dat|en|eng|ech|du|hien|hatt|si|et|mir|dir|een)\s+)(?=\S)",
    re.IGNORECASE,
)
# A word of a spoken segment; an apostrophe inside it does not end it ("d'Waasser").
SPOKEN_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
RULES_PATH = ROOT / "app" / "prompts" / "rules_lb.md"
NO_STORE = "LOD facts: the dictionary store is not installed; treat every form as not verified in LOD"
REASK = "Your last reply broke the format: "
AUTH_FIX = "run `claude auth login` once in a terminal"
PHRASE_FIELDS = {"lb", "en", "first_seen", "last_seen", "times_seen", "sessions"}


class TurnError(Exception):
    """A failed turn: the HTTP status for the route, the error text and how to fix it."""

    def __init__(self, status: int, error: str, fix: str = "") -> None:
        super().__init__(error)
        self.status = status
        self.error = error
        self.fix = fix


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


@dataclass
class Session:
    id: str
    turns: list[Turn] = field(default_factory=list)
    created: str = field(default_factory=_now)
    practice: PracticeState | None = None  # where a practice talk stands, once one was started


class SessionStore:
    """The sessions of the running server, in memory."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def new(self) -> Session:
        session = Session(uuid.uuid4().hex)
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)


def phrase_key(lb: str) -> str:
    """What makes two phrases the same one: case and trailing punctuation do not count."""
    return " ".join(lb.split()).rstrip(" .,!?;:…").casefold()


class PhraseMemory:
    """The Luxembourgish phrases met so far, as a JSON list in one file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def list(self) -> list[dict]:
        with self._lock:
            return self._read()

    def latest(self, limit: int | None = None) -> list[dict]:
        """The phrases, last seen first; at most `limit` of them."""
        # Reversed first, so that of two entries of the same second the later one in the file leads.
        entries = sorted(reversed(self.list()), key=lambda entry: entry["last_seen"], reverse=True)
        return entries[:limit]

    def add(self, lb: str, en: str, session_id: str) -> bool:
        """Record one meeting with a phrase; True when the phrase is new to the memory."""
        key = phrase_key(lb)
        with self._lock:
            entries = self._read()
            now = _now()
            known = next((entry for entry in entries if phrase_key(entry["lb"]) == key), None)
            if known is None:
                entries.append(
                    {
                        "lb": lb.strip(),
                        "en": en.strip(),
                        "first_seen": now,
                        "last_seen": now,
                        "times_seen": 1,
                        "sessions": [session_id],
                    }
                )
            else:
                known["last_seen"] = now
                known["times_seen"] += 1
                if session_id not in known["sessions"]:
                    known["sessions"].append(session_id)
            self._write(entries)
        return known is None

    def delete(self) -> None:
        """Removes the file and the `.corrupt` and `.tmp` next to it; one that is not there is fine.
        The memory then starts empty on its next add."""
        with self._lock:
            for path in (self.path, *(self.path.with_name(self.path.name + end) for end in (".corrupt", ".tmp"))):
                path.unlink(missing_ok=True)

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            entries = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(entries, list) or not all(
                isinstance(entry, dict) and PHRASE_FIELDS <= set(entry) for entry in entries
            ):
                raise ValueError("not a list of phrase entries")
        except (OSError, ValueError):
            os.replace(self.path, self.path.with_name(self.path.name + ".corrupt"))
            return []
        return entries

    def _write(self, entries: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(self.path.name + ".tmp")
        temp.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, self.path)


@dataclass
class TurnResult:
    transcript: str  # the user text of the turn
    mode: str  # "chat" or "teacher"
    reply: TutorReply
    audio: WavAudio
    claims: list[dict]  # {"form", "lemma", "verified"}
    phrases_added: list[str]  # the lb phrases that were new to the memory
    timings: dict  # stt_ms, llm_ms, tts_ms, total_ms
    shape_warning: str  # what the accepted reply still breaks, or ""
    mix: int  # the mix level the reply was made at; 0 in teacher mode
    glosses: list[str]  # per segment: the English of an "lb" segment that is a known phrase, or ""


def pick_mode(user_text: str, mode: str = "auto") -> str:
    if mode != "auto":
        return mode
    return "teacher" if TEACHER_TRIGGERS.search(user_text) else "chat"


def checked_mix(mix, default: int = 0) -> int:
    """`mix` as a level of MIX_LEVELS; None and "" stand for `default`. A form sends the level as text."""
    if mix is None or mix == "":
        return min(max(default, 0), len(MIX_LEVELS) - 1)
    level = int(mix) if isinstance(mix, str) and mix.isascii() and mix.isdigit() else mix
    if type(level) is not int or not 0 <= level < len(MIX_LEVELS):
        raise TurnError(400, f"unknown mix level {mix!r}; known: 0 to {len(MIX_LEVELS) - 1}", MIX_FIX)
    return level


def candidate_words(*texts: str, limit: int = MAX_GROUNDING_WORDS) -> list[str]:
    """The first `limit` distinct tokens of the texts, in order."""
    words: dict[str, str] = {}
    for text in texts:
        for word in WORD.findall(text):
            words.setdefault(word.casefold(), word)
    return list(words.values())[:limit]


def english_terms(text: str) -> list[str]:
    """The English words and two-word phrases of `text` to look up in the LOD glosses, in order.

    A phrase comes before its first word. A word needs three letters and must not be in
    ENGLISH_STOP; a phrase is left out when both of its words are in it. What the question
    asks about (ASKED) is kept whatever it is: a word in quotes, "say do", "the word for
    you", "how does be work", "what is to go in Luxembourgish". A word asked about as a verb
    (VERB_ASKED) is given as "to " + the word, which puts the verbs first in the lookup.
    """
    asked = [match.span(1) for pattern in ASKED for match in pattern.finditer(text)]
    verbs = {match.start(1) for match in VERB_ASKED.finditer(text)}
    found = list(ENGLISH_WORD.finditer(text))
    words = [match.group().casefold() for match in found]
    kept = [any(start <= match.start() < end for start, end in asked) for match in found]
    terms: dict[str, None] = {}
    for index, word in enumerate(words):
        # "go in Luxembourgish" asks about "go", not about "go in".
        frame = words[index + 1 : index + 3] == ["in", "luxembourgish"]
        if not frame and index + 1 < len(words) and text[found[index].end() : found[index + 1].start()].isspace():
            if kept[index] or kept[index + 1] or not {word, words[index + 1]} <= ENGLISH_STOP:
                terms[f"{word} {words[index + 1]}"] = None
        if kept[index] or (len(word) >= 3 and word not in ENGLISH_STOP):
            terms[f"to {word}" if found[index].start() in verbs else word] = None
    return list(terms)


def shape_violations(reply: TutorReply, mode: str, mix: int = 0) -> list[str]:
    """What `reply` breaks of the bilingual shape of `mode`; empty when it fits.

    `mix` is the mix level of a chat reply; teacher mode has none.
    """
    segments = reply.segments
    words = [len(segment.text.split()) for segment in segments]
    found = [f"segment {index} has no text" for index, count in enumerate(words) if not count]
    if not segments or segments[0].lang != "lb":
        found.append("segment 0 must have lang lb")
    elif not 1 <= words[0] <= 12:
        found.append(f"segment 0 must be one to twelve words, it has {words[0]}")
    if not any(segment.lang == "en" for segment in segments):
        found.append("at least one segment must have lang en")
    if mode == "chat" and mix:
        found += _mix_violations(segments, words, mix)
    elif mode == "chat" and any(segment.lang != "en" for segment in segments[1:]):
        found.append("in chat mode every segment after segment 0 must have lang en")
    if sum(words) >= MAX_WORDS[mode]:
        found.append(f"a {mode} reply must be under {MAX_WORDS[mode]} words, it has {sum(words)}")
    return found


def _mix_violations(segments: list[Segment], words: list[int], mix: int) -> list[str]:
    """What a chat reply breaks of mix level `mix` (1 or more); `words` is the word count per segment.

    What is switched is a phrase or a sentence: a one-word "lb" segment after segment 0 must
    be a sentence of its own ("Merci!"), after a segment that ends a sentence.
    """
    found = []
    if len(segments) > 1 and segments[-1].lang != "en":
        found.append("the last segment must have lang en")
    for index in range(1, len(segments)):
        text = segments[index].text
        whole = ends_sentence(segments[index - 1].text) and ends_sentence(text)
        if segments[index].lang == "lb" and words[index] == 1 and not whole:
            found.append(
                f"segment {index} is a lone Luxembourgish word inside a sentence ({text.strip()}):"
                " switch a whole phrase or sentence, or keep the word in English"
            )
    switched = [count for segment, count in zip(segments[1:], words[1:]) if segment.lang == "lb"]
    least = MIX_MIN_LB_SEGMENTS[mix]
    if len(switched) < least:
        found.append(
            f"at mix level {mix} at least {least} of the segments after segment 0 must have lang lb,"
            f" {len(switched)} have"
        )
    percent = round(100 * sum(switched) / max(sum(words[1:]), 1))
    low, high = round(100 * MIX_MIN_LB_SHARE[mix]), round(100 * MIX_MAX_LB_SHARE[mix])
    if percent < low:
        found.append(
            f"at mix level {mix} at least {low} percent of the words after segment 0 must be in lb segments,"
            f" {percent} percent are"
        )
    if percent > high:
        found.append(
            f"at mix level {mix} at most {high} percent of the words after segment 0 may be in lb segments,"
            f" {percent} percent are"
        )
    return found


def bare_form(form: str) -> str:
    """`form` without closing punctuation and without an article or pronoun in front of it."""
    return LEADING.sub("", form.strip().rstrip(".!?,").rstrip())


def checked_claim(lod: LodStore | None, claim: dict) -> dict:
    """`claim` with `verified`: whether LOD lists its form, as a form of its lemma when it names one."""
    form, lemma = claim.get("form", ""), claim.get("lemma", "")
    tried = dict.fromkeys((form, form.strip().rstrip(".!?,"), bare_form(form)))
    verified = lod is not None and any(lod.verify(candidate, lemma or None) for candidate in tried)
    return {"form": form, "lemma": lemma, "verified": verified}


def stated_forms(reply: TutorReply) -> set[str]:
    """The forms and headwords `reply` claims, each as its bare form, case-folded."""
    forms = [claim.get(key, "") for claim in reply.claims for key in ("form", "lemma")]
    return {bare_form(form).casefold() for form in forms} - {""}


def _spoken_words(text: str) -> list[tuple[str, int, int]]:
    """(bare form case-folded, start, end) of every word of `text`."""
    return [(bare_form(match.group()).casefold(), *match.span()) for match in SPOKEN_WORD.finditer(text)]


def _foreign_spans(
    text: str, forms: set[str], lb_words: set[str], english_words: frozenset[str]
) -> list[tuple[int, int]]:
    """Where `text` has Luxembourgish words.

    A word of `forms` or `lb_words` that is not in `english_words` is Luxembourgish, and so
    is a word of `lb_words` with only spaces between it and a Luxembourgish word. Words with
    no other word between them share one span. A span next to the start or the end of `text`
    reaches it when only punctuation lies between.
    """
    words = _spoken_words(text)
    known = forms | lb_words
    foreign = [word in known and word not in english_words for word, _, _ in words]
    # joined[i]: only spaces between word i and word i + 1.
    joined = [text[left[2] : right[1]].isspace() for left, right in zip(words, words[1:])]
    for index in range(1, len(words)):
        if foreign[index - 1] and joined[index - 1] and words[index][0] in lb_words:
            foreign[index] = True
    for index in range(len(words) - 2, -1, -1):
        if foreign[index + 1] and joined[index] and words[index][0] in lb_words:
            foreign[index] = True
    spans: list[tuple[int, int]] = []
    for (_, start, end), is_foreign in zip(words, foreign):
        if not is_foreign:
            continue
        if spans and not re.search(r"\w", text[spans[-1][1] : start]):
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))
    if spans and not re.search(r"\w", text[: spans[0][0]]):
        spans[0] = (0, spans[0][1])
    if spans and not re.search(r"\w", text[spans[-1][1] :]):
        spans[-1] = (spans[-1][0], len(text))
    return spans


def split_foreign(
    segments: list[Segment], forms: set[str], english_words: frozenset[str], phrases: Iterable[str] = ()
) -> list[Segment]:
    """`segments` with the Luxembourgish words of each "en" segment in "lb" segments of their own.

    A word is Luxembourgish when its bare form is in `forms` or occurs in an "lb" segment and
    is not in `english_words`, all case-folded. A run of words from the "lb" segments around
    such a word goes with it, English look-alikes included ("mäin Dag war"); on its own a
    look-alike stays English. Punctuation stays with the English next to it. A word that
    ends the last segment stays in it, so the reply still ends in English. The words of
    `phrases`, Luxembourgish texts, count like the words of the "lb" segments.
    """
    lb_texts = [segment.text for segment in segments if segment.lang == "lb"] + list(phrases)
    lb_words = {word for text in lb_texts for word, _, _ in _spoken_words(text)}
    result = []
    for index, segment in enumerate(segments):
        text = segment.text
        spans = _foreign_spans(text, forms, lb_words, english_words) if segment.lang == "en" else []
        if spans and index == len(segments) - 1 and spans[-1][1] == len(text):
            spans.pop()
        if not spans:
            result.append(segment)
            continue
        cursor = 0
        pieces = []
        for start, end in spans:
            pieces += [Segment("en", text[cursor:start].strip()), Segment("lb", text[start:end].strip())]
            cursor = end
        pieces.append(Segment("en", text[cursor:].strip()))
        result += [piece for piece in pieces if piece.text]
    return result


class Orchestrator:
    """`learner` is the learner store that turns and phrases are also recorded in; None records nothing.
    A store that fails is logged, and the turn goes on."""

    def __init__(self, stt, tts, llm, lod: LodStore | None, settings, learner: LearnerStore | None = None) -> None:
        self.stt = stt
        self.tts = tts
        self.llm = llm
        self.lod = lod
        self.settings = settings
        self.learner = learner
        self.phrases = PhraseMemory(settings.data_dir / "phrases.json")
        self.rules = RULES_PATH.read_text(encoding="utf-8").strip()

    def transcribe(self, audio_bytes: bytes, mime: str) -> tuple[Transcript, int]:
        """The transcript and the milliseconds it took, to pass on to `turn` as `stt_ms`."""
        started = time.perf_counter()
        transcript = self._engine("stt", self.stt, "transcribe", audio_bytes, mime)
        return transcript, _ms(started)

    def turn(
        self,
        session: Session,
        user_text: str,
        mode: str = "auto",
        stt_ms: int = 0,
        mix: int | None = None,
        duration_ms: int = 0,
    ) -> TurnResult:
        """`mix` is the mix level of a chat reply; None takes the `chat_mix` setting. `duration_ms`
        is how long the learner spoke the message, 0 when it was typed; it is only recorded."""
        started = time.perf_counter()
        user_text = user_text.strip()
        if not user_text:
            raise TurnError(400, "empty message")
        if mode not in MODES:
            raise TurnError(400, f"unknown mode {mode!r}; known: {', '.join(MODES)}")
        mix = checked_mix(mix, self.settings.chat_mix)
        mode = pick_mode(user_text, mode)
        if mode != "chat":
            mix = 0
        history = session.turns + [Turn("user", user_text, mode=mode)]
        grounding = self._grounding(session, user_text) if mode == "teacher" else self._mix_block(mix)

        llm_started = time.perf_counter()
        reply = self._reply(history, mode, mix, grounding)
        violations = shape_violations(reply, mode, mix)
        if violations:
            note = REASK + "; ".join(violations)
            reply = self._reply(history, mode, mix, f"{grounding}\n\n{note}".strip())
            violations = shape_violations(reply, mode, mix)
        llm_ms = _ms(llm_started)
        if mode == "teacher" and self.lod is not None:
            segments = split_foreign(reply.segments, stated_forms(reply), self.lod.english_words())
            reply = replace(reply, segments=segments)

        claims = [checked_claim(self.lod, claim) for claim in reply.claims]
        tts_started = time.perf_counter()
        audio = self._engine("tts", self.tts, "synthesize", reply.segments)
        tts_ms = _ms(tts_started)
        phrases_added = self._remember(reply.new_phrases, session.id)
        glosses = self._glosses(reply)

        text = " ".join(segment.text.strip() for segment in reply.segments)
        session.turns += [history[-1], Turn("tutor", text, list(reply.segments), mode)]
        del session.turns[:-MAX_HISTORY]
        self._recorded("turn_said", session.id, mode, user_text, duration_ms, text)
        return TurnResult(
            transcript=user_text,
            mode=mode,
            reply=reply,
            audio=audio,
            claims=claims,
            phrases_added=phrases_added,
            timings={
                "stt_ms": stt_ms,
                "llm_ms": llm_ms,
                "tts_ms": tts_ms,
                "total_ms": stt_ms + _ms(started),
            },
            shape_warning="; ".join(violations),
            mix=mix,
            glosses=glosses,
        )

    def _reply(self, history: list[Turn], mode: str, mix: int, grounding: str) -> TutorReply:
        """One reply of the tutor.

        A chat reply at a mix level comes back with the Luxembourgish words of its "en"
        segments in "lb" segments of their own, so the shape check counts what is spoken.
        """
        reply = self._engine("llm", self.llm, "reply", history, mode, grounding)
        if mode == "chat" and mix and self.lod is not None:
            # Chat has no claims: what is Luxembourgish is known from the "lb" segments and the new phrases.
            phrases = [phrase.get("lb", "") for phrase in reply.new_phrases]
            segments = split_foreign(reply.segments, set(), self.lod.english_words(), phrases)
            reply = replace(reply, segments=segments)
        return reply

    def _mix_block(self, mix: int) -> str:
        """The chat block: the mix level and the phrases the learner has met; "" at level 0."""
        if not mix:
            return ""
        lines = [f"mix level: {mix}"]
        for entry in self._known_phrases(MAX_MIX_PHRASES):
            # One line per phrase, so text inside a phrase cannot pose as another line of the block.
            lb, en = (" ".join(entry[key].split()) for key in ("lb", "en"))
            if lb:
                lines.append(f"{lb} = {en}" if en else lb)
        if len(lines) == 1:
            return f"{lines[0]}\nThe learner has met no phrases yet."
        lines.insert(1, "Phrases the learner has met, as Luxembourgish = English, the latest first:")
        return "\n".join(lines)

    def _known_phrases(self, limit: int | None = None) -> list[dict]:
        try:
            return self.phrases.latest(limit)
        except OSError as exc:
            raise TurnError(500, f"phrase memory ({self.phrases.path}) failed: {type(exc).__name__}: {exc}") from exc

    def _glosses(self, reply: TutorReply) -> list[str]:
        """Per segment of `reply` the English of its phrase, "" when it has none.

        An "lb" segment has one when its text is a phrase of the memory or of the reply's
        `new_phrases`; the reply's own gloss wins.
        """
        known = [(entry["lb"], entry["en"]) for entry in self._known_phrases()]
        known += [(phrase.get("lb", ""), phrase.get("en", "")) for phrase in reply.new_phrases]
        glosses = {phrase_key(lb): en.strip() for lb, en in known if phrase_key(lb) and en.strip()}
        return [
            glosses.get(phrase_key(segment.text), "") if segment.lang == "lb" else "" for segment in reply.segments
        ]

    def _grounding(self, session: Session, user_text: str) -> str:
        """The teacher block: LOD facts for the words of this exchange, then the rules sheet.

        The user text and the last tutor turn are looked up as Luxembourgish forms, the user
        text also as English.
        """
        if self.lod is None:
            return f"{NO_STORE}\n\n{self.rules}"
        last_tutor = next((turn.text for turn in reversed(session.turns) if turn.role == "tutor"), "")
        facts = self.lod.grounding_for(candidate_words(user_text, last_tutor), english_terms(user_text))
        return f"{facts}\n\n{self.rules}"

    def _remember(self, new_phrases: list[dict], session_id: str) -> list[str]:
        """The phrases new to the phrase memory, which says what is new; the learner store records each meeting too."""
        added = []
        for phrase in new_phrases:
            lb, en = phrase.get("lb", ""), phrase.get("en", "")
            if not phrase_key(lb):
                continue
            try:
                if self.phrases.add(lb, en, session_id):
                    added.append(lb)
            except OSError as exc:
                raise TurnError(
                    500, f"phrase memory ({self.phrases.path}) failed: {type(exc).__name__}: {exc}"
                ) from exc
            self._recorded("phrase_met", lb, en, session_id)
        return added

    def _recorded(self, method: str, *args) -> None:
        """Calls `method` of the learner store; without a store nothing, and a failure is only logged."""
        if self.learner is None:
            return
        try:
            getattr(self.learner, method)(*args)
        except Exception as exc:
            log.warning("the learner store has not recorded %s: %s: %s", method, type(exc).__name__, exc)

    @staticmethod
    def _engine(kind: str, engine, method: str, *args):
        """Call an engine; whatever it raises becomes a TurnError."""
        try:
            return getattr(engine, method)(*args)
        except LLMError as error:
            detail = error.detail or str(error)
            if error.kind == "auth":
                raise TurnError(503, detail, fix=AUTH_FIX) from error
            raise TurnError(502, detail) from error
        except Exception as exc:
            raise TurnError(500, f"{kind} engine ({engine.name}) failed: {type(exc).__name__}: {exc}") from exc
