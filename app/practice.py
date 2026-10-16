"""Practice mode: a talk that the tutor leads from the fixed content of `app.content`: two
warm-up questions, the questions of one topic, then three questions about a photo the learner picks.

`PracticeFlow` is a state machine over a `PracticeState`; it calls no engine. `judge_answer`
asks the LLM what the tutor makes of one answer and checks what comes back against the answer
and LOD. The routes call both and speak what the flow returns. A talk ends in a look back, a
`WrapUp`: what code knows of the talk, and the tutor's sentences about it once they are checked.
"""

from __future__ import annotations

import logging
import random
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field

from app.content import PHOTOS_OFFERED, WARMUPS_ASKED, Photo, PracticeContent, Question
from app.corrections import HEARD_UNEDITED, label_correction
from app.engines.base import AcceptabilityEngine, Judgement, Segment, WrapUpText
from app.lod import checks
from app.lod.checks import WORD as ANSWER_WORD
from app.lod.store import LodStore
from app.orchestrator import NO_STORE, SPOKEN_WORD, bare_form, candidate_words, checked_claim, split_foreign

log = logging.getLogger(__name__)

MORE_BELOW_WORDS = 6  # a confirmed answer with fewer words is asked for more, once per question
GROUNDING_WORDS = 30  # words of an answer that are looked up in LOD for its judgement
HELP_USED = "help used: yes"  # the line that tells the model the learner saw the question in English
EARLIER_REACTIONS = 4  # of the talk, given to the model so that it does not say the same again
REACTIONS_SO_FAR = "Your reactions so far in this chat, oldest first:"
PICTURE_SHOWS = "What the picture shows:"  # heads the description of the photo an answer is about
PHOTO_STAGE = "photo"  # the stage of the questions about a photo
# How a talk is held, as the learner picks it; the flow is the same in both, the page shows a clock in the second.
TALK_SETTINGS = ("relaxed", "realistic")
_AUXILIARIES = frozenset({"hunn", "sinn", "ginn", "wäerten"})  # LOD lemmas; a swap between them is grammar
# Of a look back, in the order they are said: whether the message got across comes first.
SENTENCES = ("task_en", "words_en", "grammar_en")
SPOKEN_BETTER = 3  # better versions a look back says once more; the page lists every one
# What the sentences of a look back must not hold: a word of a test, a mark, a certificate
# that a sentence is flawless, a teacher's praise, a duty.
NOT_SAID = re.compile(
    r"\b(?:exams?|candidates?|marks?|scores?|grades?|levels?|pass(?:es|ed)?|fail\w*|percent\w*|timers?|a[12]"
    r"|perfect\w*|correct\w*|flawless\w*|no mistakes?|no errors?|well done|good job|excellent"
    r"|you should|you need to|you must|remember to|next time)\b|%|\d\s*/\s*\d",
    re.IGNORECASE,
)
# Where one sentence of a text ends and the next begins.
SENTENCE_GAP = re.compile(r"[.!?…]+\s+")
# How the reason of a fix leads into the better version: ", so:" or only its last punctuation.
LEAD_OUT = re.compile(r"(?:[,\s]+so)?[\s.,;:]*\Z", re.IGNORECASE)
# A text in quotation marks; a single one counts only where it is no apostrophe inside a word.
QUOTED = re.compile(r"""["“„«]([^"“”„«»]+)["”»]|(?<!\w)['‘]([^'‘’]+)['’](?!\w)""")
_NOT_LETTER_OR_SPACE = re.compile(r"[^\w\s]|[\d_]")


def _plain(text: str) -> str:
    """`text` case-folded, without punctuation, with single spaces."""
    return " ".join(_NOT_LETTER_OR_SPACE.sub("", text).split()).casefold()


# A confirmed text that is one of these, and nothing else, skips the question.
SKIP_PHRASES = frozenset(
    _plain(phrase)
    for phrase in ("I don't know", "no idea", "skip", "next", "pass", "keng Ahnung", "ech weess net")
)


@dataclass
class Correction:
    """What the tutor makes of an answer, after the checks of `checked_judgement`.

    Only a "fix" has `quote`, `why_en` and `why_spoken`, only a "fix" and an "english" have
    `better_lb`, `claims`, `label`, `category` and `source`. A "good" may have `alternative_lb`:
    the better version of a fix between two forms LOD accepts, with the `checks` that say so.
    """

    verdict: str  # "good", "fix", "english" or "unclear"
    reaction_en: str
    quote: str = ""  # the learner's own words
    better_lb: str = ""
    why_en: str = ""
    claims: list[dict] = field(default_factory=list)  # {"form", "lemma", "verified"} for words of better_lb
    # `why_en` as it is spoken, its Luxembourgish words in "lb" segments; empty: as one English segment.
    why_spoken: list[Segment] = field(default_factory=list)
    label: str = ""  # "verified", "likely" or "unverified"
    category: str = ""  # "words" or "grammar"
    source: str = ""  # of the answer: "typed", "heard-edited" or "heard-unedited"
    alternative_lb: str = ""
    checks: list[str] = field(default_factory=list)  # the facts the label (or the alternative) rests on


@dataclass
class Answer:
    question_id: str
    raw_transcript: str  # what speech-to-text heard; "" for an answer that was typed
    text: str  # what the learner confirmed
    help_used: bool  # the learner saw or heard the English of the question
    correction: Correction | None = None  # None when the answer was not judged


@dataclass
class PracticeState:
    part: str = "warmup"  # then "interview", "photo", "done"
    topic_id: str = ""
    setting: str = "relaxed"  # one of TALK_SETTINGS
    warmup_ids: list[str] = field(default_factory=list)  # the warm-up questions of this talk, in order
    # Warm-up ids drawn in the talks of the session, this one too: the next talk draws others first.
    warmups_asked: list[str] = field(default_factory=list)
    question_index: int = 0  # among the questions of the part
    asked_more: bool = False  # the open question has had its cue for more
    skips: int = 0  # of the open question
    answers: list[Answer] = field(default_factory=list)
    photos_offered: list[str] = field(default_factory=list)  # ids the learner picks from in the photo part
    photo_id: str = ""  # the one they picked; "" until then
    # Ids picked in the talks of the session, this one too: the next talk offers others first.
    photos_described: list[str] = field(default_factory=list)
    wrapped: bool = False  # the look back on the talk was given; it is given once


@dataclass
class WrapUp:
    """The look back on a talk: what code knows of it, then the tutor's sentences about it.

    A question is named by its number in the talk, from 1: the warm-up questions, then those
    of the topic, then those about the photo.
    """

    answered: int  # questions the learner answered
    skipped: int  # questions they let go without an answer
    help_questions: list[int]  # the questions they had in English
    # Of every "fix" and "english" in order: {"question_number", "question_lb", "answer",
    # "verdict", "quote", "better_lb", "why_en", "claims", "label", "category", "source",
    # "checks"}; `answer` is the confirmed text.
    corrections: list[dict]
    unknown_words: list[str]  # words of their answers that LOD does not list, as they wrote them
    photo_described: bool  # they answered a question about a photo
    # Per name of SENTENCES what the tutor says under that heading; "" when it did not arrive.
    sentences: dict[str, str] = field(default_factory=lambda: dict.fromkeys(SENTENCES, ""))
    llm_failed: bool = False  # one of `sentences` is missing
    # Per name of SENTENCES the sentence as it is spoken, what it quotes in "lb" segments.
    spoken: dict[str, list[Segment]] = field(default_factory=dict)


@dataclass
class PracticeTurn:
    """What the tutor says next.

    `question` is the open question; None while the learner picks a photo and when the talk
    is over.
    """

    segments: list[Segment]
    question: Question | None
    part: str
    kind: str  # "question", "more", "again", "choose" or "end"
    glosses: list[str] = field(default_factory=list)  # per segment its English, or ""
    index: int = 0  # of `question` among the questions of the part, from 0
    total: int = 0  # questions of the part
    correction: Correction | None = None  # of the answer this turn follows
    photos: list[Photo] = field(default_factory=list)  # to pick from, on a "choose" turn only
    photo: Photo | None = None  # the picked one, while its questions are open


def is_skip(text: str) -> bool:
    return _plain(text) in SKIP_PHRASES


def word_count(text: str) -> int:
    return len(ANSWER_WORD.findall(text))


def had_help(state: PracticeState, help_used: bool) -> bool:
    """Whether the learner has the open question in English: they asked for it, or a skip brought it."""
    return help_used or state.skips > 0


def answer_grounding(lod: LodStore | None, rules: str, answer: str) -> str:
    """The block a judgement works from: LOD facts for the words of `answer`, then the rules sheet."""
    if lod is None:
        return f"{NO_STORE}\n\n{rules}"
    return f"{lod.grounding_for(candidate_words(answer, limit=GROUNDING_WORDS), [])}\n\n{rules}"


def labelled_claims(lod: LodStore | None, better_lb: str, claims: list[dict]) -> list[dict]:
    """The LOD labels under `better_lb`: each of `claims` as checked, then every word of
    `better_lb` that no claim covers and LOD does not list, as a claim without lemma."""
    labelled = []
    covered: set[str] = set()
    for claim in claims:
        form = " ".join(claim.get("form", "").split())
        words = {bare_form(match.group()).casefold() for match in SPOKEN_WORD.finditer(form)}
        if words and not words <= covered:
            covered |= words
            labelled.append(checked_claim(lod, {"form": form, "lemma": claim.get("lemma", "").strip()}))
    for match in SPOKEN_WORD.finditer(better_lb):
        word = bare_form(match.group())
        if word.casefold() not in covered:
            covered.add(word.casefold())
            if lod is None or not lod.known(word):
                labelled.append({"form": word, "lemma": "", "verified": False})
    return labelled


def lead_in(why: str) -> str:
    """The reason `why` as it leads into the better version, whatever the model ended it with; "" for none."""
    why = why.rstrip(" .,;:")
    return f"{why}:" if why else ""


def spoken_reason(lod: LodStore | None, why: str, quote: str, better_lb: str, claims: list[dict]) -> list[Segment]:
    """The reason `why` of a fix as it is spoken, its Luxembourgish words in "lb" segments of their own.

    A word is Luxembourgish as `split_foreign` has it, from the forms and lemmas of `claims`
    and the words of `quote` and `better_lb`. Empty without a store and without a reason.
    """
    reason = lead_in(why)
    if lod is None or not reason:
        return []
    forms = {bare_form(claim.get(key, "")).casefold() for claim in claims for key in ("form", "lemma")} - {""}
    # The better version follows the reason, so a Luxembourgish word that ends the reason is split off too.
    segments = [Segment("en", reason), Segment("lb", better_lb)]
    return split_foreign(segments, forms, lod.english_words(), [quote])[:-1]


def answer_source(text: str, raw_transcript: str) -> str:
    """Where the confirmed answer `text` came from: "typed" without `raw_transcript`, else
    "heard-unedited" when it is what was heard, whatever its case and punctuation, else "heard-edited"."""
    if not raw_transcript.strip():
        return "typed"
    return HEARD_UNEDITED if _plain(raw_transcript) == _plain(text) else "heard-edited"


def correction_category(lod: LodStore | None, verdict: str, quote: str, better_lb: str) -> str:
    """The category of a "fix" or an "english": "words" or "grammar"; "" for the others.

    A fix is grammar when a changed word and a new word are forms of one LOD lemma, when
    it changes no word of the quote (words were added or moved), or when it swaps forms of the
    auxiliaries hunn, sinn, ginn and wäerten; any other fix is words, and so is an answer in
    English.
    """
    if verdict == "english":
        return "words"
    if verdict != "fix":
        return ""
    old, new = checks.changed_words(quote, better_lb)
    if not old or checks.same_lemma(lod, quote, better_lb):
        return "grammar"
    if all(checks.lemmas(lod, word) & _AUXILIARIES for word in old) and any(
        checks.lemmas(lod, word) & _AUXILIARIES for word in new
    ):
        return "grammar"
    return "words"


def checked_judgement(
    judgement: Judgement,
    text: str,
    lod: LodStore | None,
    encoder: AcceptabilityEngine | None = None,
    raw_transcript: str = "",
) -> Correction:
    """`judgement` of the confirmed answer `text` as the tutor may say it; `raw_transcript`
    is what was heard, "" for a typed answer.

    A "fix" must quote whole words of `text` and give a better version that differs from
    them; an "english" must give the Luxembourgish. One that does not becomes "good" with
    its reaction kept, so the tutor never corrects what the learner did not say. A fix or an
    english that stands then gets its label from `label_correction`; a fix that the label
    downgrades becomes "good" too, and one between two forms LOD accepts keeps its better
    version as `alternative_lb`.
    """
    verdict = judgement.verdict
    reaction, quote, better, why = (
        " ".join(part.split())
        for part in (judgement.reaction_en, judgement.quote, judgement.better_lb, judgement.why_en)
    )
    wrong = ""
    if verdict == "fix":
        if not _plain(quote) or f" {_plain(quote)} " not in f" {_plain(text)} ":
            wrong = f"the quote {quote!r} is not in the answer"
        elif not _plain(better):
            wrong = "there is no better version"
        elif _plain(better) == _plain(quote):
            wrong = f"the better version {better!r} is the quote"
    elif verdict == "english" and not _plain(better):
        wrong = "there is no Luxembourgish version"
    if wrong:
        log.warning("practice: a judgement %r is taken as good: %s", verdict, wrong)
        verdict = "good"
    if verdict not in ("fix", "english"):
        return Correction(verdict, reaction)
    source = answer_source(text, raw_transcript)
    label = label_correction(lod, encoder, text, quote if verdict == "fix" else "", better, source)
    if label.downgrade:
        log.warning("practice: a fix is taken as good: %s: %s", label.downgrade, "; ".join(label.checks))
        if label.downgrade == "optional variant":
            return Correction("good", reaction, alternative_lb=better, checks=label.checks)
        return Correction("good", reaction)
    category = correction_category(lod, verdict, quote, better)
    claims = labelled_claims(lod, better, judgement.claims)
    labelled = {"label": label.tier, "category": category, "source": source, "checks": label.checks}
    if verdict == "fix":
        spoken = spoken_reason(lod, why, quote, better, claims)
        return Correction(verdict, reaction, quote, better, why, claims, spoken, **labelled)
    return Correction(verdict, reaction, better_lb=better, claims=claims, **labelled)


def judge_answer(
    llm,
    lod: LodStore | None,
    rules: str,
    state: PracticeState,
    question: Question,
    text: str,
    help_used: bool = False,
    picture: str = "",
    encoder: AcceptabilityEngine | None = None,
    raw_text: str = "",
) -> Correction:
    """What the tutor makes of the confirmed answer `text` to `question`, the open question
    of `state`: one LLM call, then the checks. `state` is left as it is.

    `encoder` and `raw_text` (what was heard, "" for a typed answer) go to the label of the
    correction. `rules` is the rules sheet. The model is told when the learner had the question in
    English, what `picture` shows when the question is about a photo (its description: the
    model does not get the image), and what the tutor said to the last answers, so that it
    finds other words. Whatever the engine raises is passed on.
    """
    notes = [HELP_USED] if had_help(state, help_used) else []
    if picture.strip():
        notes += [PICTURE_SHOWS, " ".join(picture.split())]
    said = [answer.correction.reaction_en for answer in state.answers if answer.correction]
    earlier = [reaction for reaction in said if reaction][-EARLIER_REACTIONS:]
    if earlier:
        notes += [REACTIONS_SO_FAR, *(f"- {reaction}" for reaction in earlier)]
    grounding = answer_grounding(lod, rules, text)
    judgement = llm.judge(question.lb, question.en, text, grounding, "\n".join(notes))
    return checked_judgement(judgement, text, lod, encoder, raw_text)


def correction_lines(tutor: dict[str, str], correction: Correction) -> list[tuple[Segment, str]]:
    """What the tutor says about an answer, each line with its gloss: the reaction, then the correction."""
    spoken = [("en", correction.reaction_en)]
    if correction.verdict == "fix":
        reason = [(piece.lang, piece.text) for piece in correction.why_spoken]
        spoken += [
            ("en", tutor["said_en"]),
            ("lb", correction.quote),
            *(reason or [("en", lead_in(correction.why_en) or tutor["better_en"])]),
            ("lb", correction.better_lb),
        ]
    elif correction.verdict == "english":
        spoken += [("en", tutor["could_say_en"]), ("lb", correction.better_lb)]
    return [(Segment(lang, text), "") for lang, text in spoken if text]


def photo_questions(tutor: dict[str, str], photo: Photo) -> tuple[Question, ...]:
    """The questions about `photo`, in order: what it shows, its own question, how the learner likes it."""
    lines = (
        (tutor["photo_lb"], tutor["photo_en"]),
        (photo.question_lb, photo.question_en),
        (tutor["photo_last_lb"], tutor["photo_last_en"]),
    )
    return tuple(
        Question(f"photo-{photo.id}-{number}", PHOTO_STAGE, lb, en) for number, (lb, en) in enumerate(lines, start=1)
    )


def _drawn(entries: Sequence, used: Collection[str], count: int, rng: random.Random) -> list:
    """`count` of `entries` drawn with `rng`: those whose id is not in `used`, and others only
    when these are too few."""
    fresh = [entry for entry in entries if entry.id not in used]
    known = [entry for entry in entries if entry.id in used]
    drawn = rng.sample(fresh, min(count, len(fresh)))
    return drawn + rng.sample(known, min(count - len(drawn), len(known)))


def drawn_photos(
    photos: Sequence[Photo], described: Collection[str], rng: random.Random, topic_id: str = ""
) -> list[Photo]:
    """PHOTOS_OFFERED of `photos` drawn with `rng`: first those of `topic_id` whose id is not in
    `described`, then other photos not in `described`, and those in `described` only when
    these are too few."""
    fresh = [photo for photo in photos if topic_id and photo.topic == topic_id and photo.id not in described]
    drawn = rng.sample(fresh, min(PHOTOS_OFFERED, len(fresh)))
    rest = [photo for photo in photos if photo not in drawn]
    return drawn + _drawn(rest, described, PHOTOS_OFFERED - len(drawn), rng)


def drawn_warmups(questions: Sequence[Question], asked: Collection[str], rng: random.Random) -> list[Question]:
    """WARMUPS_ASKED of the warm-up `questions` drawn with `rng`, in the order they are asked:
    those whose id is not in `asked`, and others only when these are too few."""
    return _drawn(questions, asked, WARMUPS_ASKED, rng)


class PracticeFlow:
    """The talk as a state machine. `rng` draws the warm-up questions and the photos on offer;
    one with a seed repeats its draws."""

    def __init__(self, content: PracticeContent, rng: random.Random | None = None) -> None:
        self.content = content
        self.rng = rng or random.Random()

    def start(self, state: PracticeState, topic_id: str) -> PracticeTurn:
        """Opens the topic on `state`, whatever talk it held: the opening lines, the line of the
        warm-up and its first question. Content without warm-up questions opens with the topic.

        Of that talk only `setting`, `photos_described` and `warmups_asked` are kept.
        """
        topic = self.content.topic(topic_id)
        if topic is None:
            raise ValueError(f"unknown topic {topic_id!r}")
        warmups = drawn_warmups(self.content.warmup, state.warmups_asked, self.rng)
        state.part = "warmup" if warmups else "interview"
        state.topic_id, state.question_index = topic.id, 0
        state.asked_more, state.skips, state.answers = False, 0, []
        state.photos_offered, state.photo_id, state.wrapped = [], "", False
        state.warmup_ids = [question.id for question in warmups]
        state.warmups_asked += [drawn for drawn in state.warmup_ids if drawn not in state.warmups_asked]
        tutor = self.content.tutor
        lines = [
            (
                Segment("lb", tutor["open_lb"].replace("{topic}", topic.name_lb)),
                tutor["open_en"].replace("{topic}", topic.name_en),
            )
        ]
        if warmups:
            lines.append((Segment("lb", tutor["warmup_lb"]), tutor["warmup_en"]))
        lines.append((Segment("lb", self.open_question(state).lb), ""))
        return self._turn(state, "question", lines)

    def photo(self, state: PracticeState) -> Photo | None:
        """The photo the questions of `state` are about; None outside the photo part and before the pick."""
        if state.part != "photo":
            return None
        return next((photo for photo in self.content.photos if photo.id == state.photo_id), None)

    def _warmups(self, state: PracticeState) -> tuple[Question, ...]:
        """The warm-up questions of the talk of `state`, in the order they are asked."""
        by_id = {question.id: question for question in self.content.warmup}
        return tuple(by_id[question_id] for question_id in state.warmup_ids if question_id in by_id)

    def _questions(self, state: PracticeState) -> tuple[Question, ...]:
        """The questions of the part `state` is in; none while a photo is picked and when the talk is over."""
        if state.part == "warmup":
            return self._warmups(state)
        if state.part == "interview":
            topic = self.content.topic(state.topic_id)
            return topic.questions if topic else ()
        photo = self.photo(state)
        return photo_questions(self.content.tutor, photo) if photo else ()

    def asked(self, state: PracticeState) -> list[Question]:
        """The questions of the talk of `state` in the order they are asked: the warm-up, those
        of the topic, then those about the photo once one is picked."""
        topic = self.content.topic(state.topic_id)
        photo = next((photo for photo in self.content.photos if photo.id == state.photo_id), None)
        return [
            *self._warmups(state),
            *(topic.questions if topic else ()),
            *(photo_questions(self.content.tutor, photo) if photo else ()),
        ]

    def behind(self, state: PracticeState) -> int:
        """How many of the questions of `asked` the talk has left behind, answered or not.

        A talk that ran to its end has left all of them behind.
        """
        if state.part == "done":
            return len(self.asked(state))
        if state.part == "warmup":
            return state.question_index
        topic = self.content.topic(state.topic_id)
        before = len(self._warmups(state)) + (len(topic.questions) if topic and state.part == "photo" else 0)
        return before + state.question_index

    def open_question(self, state: PracticeState) -> Question:
        """The question `state` waits for an answer to."""
        questions = self._questions(state)
        if not questions:
            raise ValueError("no question is open")
        return questions[state.question_index]

    def step(
        self,
        state: PracticeState,
        text: str,
        help_used: bool = False,
        raw_text: str = "",
        correction: Correction | None = None,
    ) -> PracticeTurn:
        """Takes the confirmed answer `text` to the open question; `raw_text` is what was heard.

        `correction` is what the tutor makes of the answer and is said first; without one the
        tutor only acknowledges the answer. A skip has none. After the last warm-up question
        the first question of the topic follows. After the last question of the
        topic the learner is offered photos to pick from, when the content has them; after
        the last question about the photo, or without photos, the talk is over.
        """
        question = self.open_question(state)
        questions = self._questions(state)
        tutor = self.content.tutor
        said: list[tuple[Segment, str]] = []
        if is_skip(text):
            correction = None
            if state.skips == 0:
                state.skips = 1
                again = tutor["again_en"].replace("{question_en}", question.en)
                if not question.starter:
                    return self._turn(state, "again", [(Segment("en", again), "")])
                lines = [(Segment("en", f"{again} {tutor['starter_en']}"), ""), (Segment("lb", question.starter), "")]
                return self._turn(state, "again", lines)
        else:
            said = correction_lines(tutor, correction) if correction else []
            state.answers.append(Answer(question.id, raw_text, text, had_help(state, help_used), correction))
            if word_count(text) < MORE_BELOW_WORDS and not state.asked_more:
                state.asked_more = True
                more = (Segment("lb", tutor["more_lb"]), tutor["more_en"])
                return self._turn(state, "more", [*self._heard(said), more], correction)

        state.question_index += 1
        state.asked_more, state.skips = False, 0
        if state.question_index < len(questions):
            following = questions[state.question_index]
            lines = self._heard(said)
            if following.stage != question.stage:
                stage_lb, stage_en = self.content.stage_lines[following.stage]
                lines.append((Segment("lb", stage_lb), stage_en))
            lines.append((Segment("lb", following.lb), ""))
            return self._turn(state, "question", lines, correction)
        if state.part == "warmup":
            state.part, state.question_index = "interview", 0
            first = (Segment("lb", self.open_question(state).lb), "")
            return self._turn(state, "question", [*self._heard(said), first], correction)
        if state.part == "interview" and len(self.content.photos) >= PHOTOS_OFFERED:
            state.part, state.question_index = "photo", 0
            offered = drawn_photos(self.content.photos, state.photos_described, self.rng, state.topic_id)
            state.photos_offered = [photo.id for photo in offered]
            intro = (Segment("lb", tutor["photo_intro_lb"]), tutor["photo_intro_en"])
            return self._turn(state, "choose", [*self._heard(said), intro], correction)
        state.part = "done"
        return self._turn(state, "end", [*said, (Segment("en", tutor["end_en"]), "")], correction)

    def choose(self, state: PracticeState, photo_id: str) -> PracticeTurn:
        """Takes the learner's pick among the photos on offer: the first question about it."""
        if state.part != "photo" or state.photo_id:
            raise ValueError("no photo is on offer")
        if photo_id not in state.photos_offered:
            raise ValueError(f"photo {photo_id!r} is not on offer")
        state.photo_id, state.question_index = photo_id, 0
        state.asked_more, state.skips = False, 0
        if photo_id not in state.photos_described:
            state.photos_described.append(photo_id)
        return self._turn(state, "question", [(Segment("lb", self.open_question(state).lb), "")])

    def _heard(self, said: list[tuple[Segment, str]]) -> list[tuple[Segment, str]]:
        """`said`, or the fixed acknowledgement when the tutor has nothing of its own to say."""
        return list(said) or [(Segment("en", self.content.tutor["ack_en"]), "")]

    def _turn(
        self,
        state: PracticeState,
        kind: str,
        lines: list[tuple[Segment, str]],
        correction: Correction | None = None,
    ) -> PracticeTurn:
        questions = self._questions(state)
        picking = state.part == "photo" and not state.photo_id
        by_id = {photo.id: photo for photo in self.content.photos}
        return PracticeTurn(
            segments=[segment for segment, _ in lines],
            question=questions[state.question_index] if questions else None,
            part=state.part,
            kind=kind,
            glosses=[gloss for _, gloss in lines],
            index=state.question_index,
            total=len(questions),
            correction=correction,
            photos=[by_id[photo_id] for photo_id in state.photos_offered if picking and photo_id in by_id],
            photo=self.photo(state),
        )


def answer_record(state: PracticeState, question: Question, answer: Answer) -> dict:
    """`answer` to `question` in the talk of `state`, as the store of confirmed answers keeps it.

    `verdict` and `correction` are null for an answer that was not judged; `claims` are the
    LOD labels of the better version, `label`, `category`, `source`, `alternative_lb` and
    `checks` those of the correction. `disputed` is null: the learner has not disputed it.
    `setting` is how the talk was held, one of TALK_SETTINGS.
    """
    correction = answer.correction
    return {
        "topic_id": state.topic_id,
        "setting": state.setting,
        "question_id": question.id,
        "question_lb": question.lb,
        "raw_transcript": answer.raw_transcript,
        "text": answer.text,
        "help_used": answer.help_used,
        "verdict": correction.verdict if correction else None,
        "correction": correction
        and {
            "reaction_en": correction.reaction_en,
            "quote": correction.quote,
            "better_lb": correction.better_lb,
            "why_en": correction.why_en,
        },
        "claims": list(correction.claims) if correction else [],
        "label": correction.label if correction else "",
        "category": correction.category if correction else "",
        "source": correction.source if correction else "",
        "alternative_lb": correction.alternative_lb if correction else "",
        "checks": list(correction.checks) if correction else [],
        "disputed": None,
    }


def unknown_words(lod: LodStore | None, texts: Sequence[str], asked: Sequence[str]) -> list[str]:
    """The words of `texts` that LOD does not list, each once as it is first written, without
    the words of `asked`. None without a store: no word was looked up."""
    if lod is None:
        return []
    seen = {bare_form(match.group()).casefold() for text in asked for match in SPOKEN_WORD.finditer(text)}
    unknown = []
    for text in texts:
        for match in SPOKEN_WORD.finditer(text):
            word = bare_form(match.group())
            if word.casefold() not in seen:
                seen.add(word.casefold())
                if not lod.known(word):
                    unknown.append(word)
    return unknown


def _in_english(answer: Answer) -> bool:
    """Whether the tutor took `answer` as English."""
    return answer.correction is not None and answer.correction.verdict == "english"


def wrap_up_facts(flow: PracticeFlow, state: PracticeState, lod: LodStore | None) -> WrapUp:
    """What code knows of the talk of `state`, without the tutor's sentences: no LLM.

    `state` is as the talk left it: stopped where it stands, or run to its end. The words of
    an answer the tutor took as English are not looked up in LOD.
    """
    asked = flow.asked(state)
    number = {question.id: index for index, question in enumerate(asked, start=1)}
    asked_lb = {question.id: question.lb for question in asked}
    answered = {answer.question_id for answer in state.answers}
    corrections = []
    for answer in state.answers:
        correction = answer.correction
        if correction and correction.verdict in ("fix", "english"):
            corrections.append(
                {
                    "question_number": number.get(answer.question_id, 0),
                    "question_lb": asked_lb.get(answer.question_id, ""),
                    "answer": answer.text,
                    "verdict": correction.verdict,
                    "quote": correction.quote,
                    "better_lb": correction.better_lb,
                    "why_en": correction.why_en,
                    "claims": correction.claims,
                    "label": correction.label,
                    "category": correction.category,
                    "source": correction.source,
                    "checks": correction.checks,
                }
            )
    luxembourgish = [answer.text for answer in state.answers if not _in_english(answer)]
    helped ={number.get(answer.question_id, 0) for answer in state.answers if answer.help_used}
    return WrapUp(
        answered=len(answered),
        skipped=sum(question.id not in answered for question in asked[: flow.behind(state)]),
        help_questions=sorted(helped - {0}),
        corrections=corrections,
        unknown_words=unknown_words(lod, luxembourgish, list(asked_lb.values())),
        photo_described=any(question.stage == PHOTO_STAGE and question.id in answered for question in asked),
    )


def _made_of(correction: Correction | None) -> str:
    """What the tutor made of an answer, as the model is told it."""
    if correction is None:
        return "No feedback was given on this one."
    if correction.verdict == "fix":
        why = LEAD_OUT.sub("", correction.why_en)
        told =f'You put one thing right: "{correction.quote}" became "{correction.better_lb}"'
        return f"{told} Reason: {why}" if why else told
    if correction.verdict == "english":
        return f'That was English. In Luxembourgish: "{correction.better_lb}"'
    if correction.verdict == "unclear":
        return "You did not catch this one."
    return "Nothing to put right."


def facts_text(flow: PracticeFlow, state: PracticeState, wrap_up: WrapUp) -> str:
    """What the model is told of the talk of `state` for the sentences of `wrap_up`: the
    counts, then every answer with what the tutor made of it.

    One line each, so text inside an answer cannot pose as another line. The words LOD does
    not list are left out: not being listed does not make a word wrong.
    """

    def line(text: str) -> str:
        return " ".join(text.split())

    topic = flow.content.topic(state.topic_id)
    helped = ", ".join(str(number) for number in wrap_up.help_questions)
    lines = [
        f"Topic: {topic.name_en} ({topic.name_lb})" if topic else "Topic: none",
        f"Questions the learner answered: {wrap_up.answered}",
        f"Questions the learner let go: {wrap_up.skipped}",
        f"Questions the learner had in English: {helped or 'none'}",
        f"The learner described a photo: {'yes' if wrap_up.photo_described else 'no'}",
        "",
        "The chat, in order:",
    ]
    for number, question in enumerate(flow.asked(state), start=1):
        answers = [answer for answer in state.answers if answer.question_id == question.id]
        if answers:
            lines.append(f"Question {number}: {line(question.lb)} ({line(question.en)})")
        for answer in answers:
            lines += [f"- The learner said: {line(answer.text)}", f"  {line(_made_of(answer.correction))}"]
    return "\n".join(lines)


def _said_in(sources: Sequence[str], quoted: str) -> bool:
    """Whether `quoted` is whole words of one of `sources`, whatever its case and punctuation."""
    wanted = _plain(quoted)
    return bool(wanted) and any(f" {wanted} " in f" {_plain(source)} " for source in sources)


def spoken_sentence(text: str, luxembourgish: Sequence[str], english: Sequence[str]) -> list[Segment]:
    """The sentence `text` of a look back as the tutor may say it; empty when it may not be said.

    What it quotes from `luxembourgish` is an "lb" segment of its own, without the quotation
    marks. It may not be said when it holds a word of NOT_SAID, or quotes what is in neither
    `luxembourgish` nor `english`: those are the words of the talk.
    """
    text = " ".join(text.split())
    if NOT_SAID.search(text):
        return []
    pieces: list[Segment] = []
    last = 0
    for match in QUOTED.finditer(text):
        quoted = (match.group(1) or match.group(2)).strip()
        if _said_in(luxembourgish, quoted):
            pieces += [Segment("en", text[last : match.start()].strip()), Segment("lb", quoted)]
            last = match.end()
        elif not _said_in(english, quoted):
            return []
    rest = text[last:].strip()
    if pieces and rest and not any(character.isalnum() for character in rest):
        # The full stop of the sentence stays with the Luxembourgish that ends it.
        pieces[-1] = Segment("lb", pieces[-1].text + rest)
    else:
        pieces.append(Segment("en", rest))
    return [piece for piece in pieces if piece.text]


def _sentences(text: str) -> list[str]:
    """The sentences of `text`; punctuation inside quotation marks ends none."""
    hidden = QUOTED.sub(lambda match: "x" * len(match.group()), text)
    ends = [match.end() for match in SENTENCE_GAP.finditer(hidden)]
    return [text[start:end].strip() for start, end in zip([0, *ends], [*ends, len(text)]) if text[start:end].strip()]


def add_sentences(wrap_up: WrapUp, flow: PracticeFlow, state: PracticeState, text: WrapUpText | None) -> None:
    """Puts on `wrap_up` what the tutor may say of `text` under each heading; None: none arrived.

    A sentence that `spoken_sentence` does not let through is dropped with a log line, and
    the others under its heading are kept. The words of the talk are its questions, the
    learner's answers, and of each correction the quote and the better version.
    """
    corrections = [answer.correction for answer in state.answers if answer.correction]
    english = [answer.text for answer in state.answers if _in_english(answer)]
    luxembourgish = [
        *(question.lb for question in flow.asked(state)),
        *(answer.text for answer in state.answers if not _in_english(answer)),
        *(part for correction in corrections for part in (correction.quote, correction.better_lb)),
    ]
    for name in SENTENCES:
        kept: list[str] = []
        pieces: list[Segment] = []
        for sentence in _sentences(" ".join(getattr(text, name).split()) if text else ""):
            spoken = spoken_sentence(sentence, luxembourgish, english)
            if not spoken:
                log.warning("practice: a sentence of the %s of the look back is dropped: %r", name, sentence)
                continue
            kept.append(sentence)
            if pieces and pieces[-1].lang == spoken[0].lang == "en":
                # One stretch of English is one segment, whatever its sentences.
                pieces[-1] = Segment("en", f"{pieces[-1].text} {spoken[0].text}")
                spoken = spoken[1:]
            pieces += spoken
        wrap_up.sentences[name] = " ".join(kept)
        wrap_up.spoken[name] = pieces
    wrap_up.llm_failed = not all(wrap_up.sentences.values())


def wrap_up_lines(tutor: dict[str, str], wrap_up: WrapUp, goodbye: bool) -> list[tuple[Segment, str]]:
    """What the tutor says looking back, each line with its role.

    The sentences, each piece with the name of its sentence; "once_more" and then the first
    SPOKEN_BETTER better versions as "better"; with `goodbye` the goodbye as "end". A talk
    without an answer gets one "line" and nothing else.
    """
    if not wrap_up.answered:
        return [(Segment("en", tutor["no_answers_en"]), "line")]
    lines = [(piece, name) for name in SENTENCES for piece in wrap_up.spoken.get(name, [])]
    again = [correction["better_lb"] for correction in wrap_up.corrections[:SPOKEN_BETTER]]
    if again:
        lines.append((Segment("en", tutor["once_more_en"]), "once_more"))
        lines += [(Segment("lb", better), "better") for better in again]
    if goodbye:
        lines.append((Segment("en", tutor["end_en"]), "end"))
    return lines
