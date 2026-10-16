import json
import shutil
import subprocess
import sys
import threading
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.engines.base import LLMError, Segment, Transcript, Turn, TutorReply
from app.engines.fake import FakeLLM, FakeSTT, FakeTTS
from app.engines.llm_claude_cli import ClaudeCliLLM, build_prompt
from app.lod.build import build
from app.lod.store import LodStore
from app.main import create_app
from app.orchestrator import (
    MAX_MIX_PHRASES,
    MIX_LEVELS,
    TEACHER_TRIGGERS,
    Orchestrator,
    PhraseMemory,
    SessionStore,
    TurnError,
    bare_form,
    candidate_words,
    checked_claim,
    checked_mix,
    english_terms,
    phrase_key,
    pick_mode,
    shape_violations,
    split_foreign,
    stated_forms,
)
from tests.realdata import REAL_DATA

FIXTURES = Path(__file__).parent / "fixtures"
GOEN_PRESENT = "  present: ech ginn, du gees, hien/si/et geet, mir ginn, dir gitt, si ginn"
HUNN_PRESENT = "  present: ech hunn, du hues, hien/si/et huet, mir hunn, dir hutt, si hunn"
# What the user typed, with Teacher on, when teacher mode found no dictionary facts.
ENGLISH_QUESTION = "I have a dog. How does the word have work in Luxembourgish?"
MAX_LIVE_CALLS = 3
EN_FIRST = TutorReply([Segment("en", "Hello! How was your day?"), Segment("lb", "Moien!")])
TEACHER_REPLY = TutorReply(
    segments=[
        Segment("lb", "Ech ginn."),
        Segment("en", "That means I go. With you, the verb becomes"),
        Segment("lb", "du gees"),
        Segment("en", "and for he or she it is"),
        Segment("lb", "hien geet"),
        Segment("en", "Where do you go today?"),
    ],
    new_phrases=[{"lb": "Ech ginn.", "en": "I go."}],
    mode="teacher",
    claims=[{"form": "gees", "lemma": "goen"}, {"form": "goest", "lemma": "goen"}],
)
DAY_QUESTION = "How would I describe my day in Luxembourgish?"
# A teacher reply whose English names three Luxembourgish forms.
PATTERN = "the pattern is ginn for I, we, and they, but gees and geet with a vowel change."
MIXED_REPLY = TutorReply(
    segments=[Segment("lb", "Ech ginn."), Segment("en", f"That means I go, and {PATTERN} Where do you go today?")],
    mode="teacher",
    claims=[{"form": form, "lemma": "goen"} for form in ("ginn", "gees", "geet")],
)
MIXED_SPLIT = [
    Segment("lb", "Ech ginn."),
    Segment("en", "That means I go, and the pattern is"),
    Segment("lb", "ginn"),
    Segment("en", "for I, we, and they, but"),
    Segment("lb", "gees"),
    Segment("en", "and"),
    Segment("lb", "geet"),
    Segment("en", "with a vowel change. Where do you go today?"),
]
GOEN_FORMS = {"ginn", "gees", "geet"}
DAY_SENTENCES = [Segment("lb", text) for text in ("Mäin Dag war gutt.", "Mäin Dag war laang.", "Ech sinn midd.")]
ENGLISH = frozenset({"war", "go", "the", "word"})
# Chat replies at a mix level: after the opening phrase, Luxembourgish phrases and sentences in the English.
OPENING = Segment("lb", "Dat ass schéin!")
SISTER = "e Kaffi mat denger Schwëster"
NEW_PHRASES = [{"lb": "Dat ass schéin!", "en": "That is nice!"}, {"lb": SISTER, "en": "a coffee with your sister"}]
LEVEL_1_REPLY = TutorReply(
    segments=[
        OPENING,
        Segment("en", "Starting the day with"),
        Segment("lb", SISTER),
        Segment("en", "sounds lovely. I hope you two had plenty of time. What did you talk about?"),
    ],
    new_phrases=NEW_PHRASES,
)
LEVEL_2_REPLY = TutorReply(
    segments=[
        OPENING,
        Segment("lb", "E Kaffi mat denger Schwëster,"),
        Segment("en", "that is a good start to the day. I hope you had"),
        Segment("lb", "vill Zäit fir ze schwätzen."),
        Segment("en", "What did you talk about?"),
    ],
    new_phrases=NEW_PHRASES,
)
LEVEL_3_REPLY = TutorReply(
    segments=[
        OPENING,
        Segment("lb", "Du hues Kaffi mat denger Schwëster gedronk, dat ass schéin!"),
        Segment("en", "Coffee with your sister, nice."),
        Segment("lb", "Wéi geet et hir?"),
        Segment("en", "How is she doing?"),
    ],
    new_phrases=NEW_PHRASES,
)
# The phrase of LEVEL_1_REPLY left inside the English.
INLINE_REPLY = replace(
    LEVEL_1_REPLY,
    segments=[OPENING, Segment("en", " ".join(segment.text for segment in LEVEL_1_REPLY.segments[1:]))],
)
NO_LB_AFTER_OPENING = "at mix level 1 at least 1 of the segments after segment 0 must have lang lb, 0 have"
MIX_HEADER = "Phrases the learner has met, as Luxembourgish = English, the latest first:"


def words(count: int) -> str:
    return " ".join(["word"] * count)


def reply_of(*segments: tuple[str, str]) -> TutorReply:
    return TutorReply([Segment(lang, text) for lang, text in segments])


@pytest.fixture(scope="module")
def small_path(tmp_path_factory):
    target = tmp_path_factory.mktemp("lod") / "lod.sqlite"
    build(FIXTURES / "lod_tab.xml", FIXTURES / "lod_art.xml", target)
    return target


@pytest.fixture
def small(small_path):
    store = LodStore(small_path)
    yield store
    store.close()


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data")


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def orchestrator(settings, llm, small):
    return Orchestrator(FakeSTT(), FakeTTS(), llm, small, settings)


@pytest.fixture
def session():
    return SessionStore().new()


@pytest.fixture
def real_lod(tmp_path):
    """The built LOD store, copied into the temporary data folder of `settings`."""
    source = REAL_DATA / "lod" / "lod.sqlite"
    if not source.is_file():
        pytest.skip(f"{source} is missing; run scripts/download_lod.py")
    target = tmp_path / "data" / "lod" / "lod.sqlite"
    target.parent.mkdir(parents=True)
    shutil.copyfile(source, target)
    store = LodStore(target)
    yield store
    store.close()


def lod_facts(grounding: str) -> list[str]:
    """The lines of the LOD facts block, without the rules sheet after it."""
    return grounding.split("\n\n")[0].splitlines()


def turn_error(orchestrator, session, text="Hello there", **kwargs) -> TurnError:
    with pytest.raises(TurnError) as raised:
        orchestrator.turn(session, text, **kwargs)
    return raised.value


def stamp(minute: int) -> str:
    return f"2026-03-01T00:{minute:02d}:00+00:00"


def remember(orchestrator, *phrases: tuple[str, str, str]) -> None:
    """Make (lb, en, last_seen) phrases the phrase memory, in this order."""
    entry = {"first_seen": stamp(0), "times_seen": 1, "sessions": []}
    entries = [{**entry, "lb": lb, "en": en, "last_seen": last_seen} for lb, en, last_seen in phrases]
    path = orchestrator.phrases.path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")


# --- a turn ---


def test_text_turn_replies_bilingually_speaks_and_remembers_the_phrase(orchestrator, session, settings):
    result = orchestrator.turn(session, "  I had coffee with my sister.  ")

    assert result.transcript == "I had coffee with my sister."
    assert result.mode == "chat"
    assert [segment.lang for segment in result.reply.segments] == ["lb", "en"]
    assert [offset["lang"] for offset in result.audio.segment_offsets] == ["lb", "en"]
    assert result.audio.wav_bytes[:4] == b"RIFF"
    assert result.claims == []
    assert result.phrases_added == ["Moien!"]
    assert result.shape_warning == ""
    (entry,) = json.loads((settings.data_dir / "phrases.json").read_text(encoding="utf-8"))
    assert (entry["lb"], entry["en"], entry["times_seen"]) == ("Moien!", "Hello!", 1)
    assert entry["sessions"] == [session.id]
    assert entry["first_seen"] == entry["last_seen"]


def test_second_identical_turn_increments_times_seen_and_adds_nothing(orchestrator, session):
    orchestrator.turn(session, "Hello there")

    result = orchestrator.turn(session, "Hello there")

    assert result.phrases_added == []
    (entry,) = orchestrator.phrases.list()
    assert entry["times_seen"] == 2
    assert entry["sessions"] == [session.id]


def test_timings_have_the_four_stages_in_whole_milliseconds(orchestrator, session):
    result = orchestrator.turn(session, "Hello there", stt_ms=40)

    assert set(result.timings) == {"stt_ms", "llm_ms", "tts_ms", "total_ms"}
    assert all(type(value) is int and value >= 0 for value in result.timings.values())
    assert result.timings["stt_ms"] == 40
    assert result.timings["total_ms"] >= 40 + result.timings["llm_ms"] + result.timings["tts_ms"]


def test_transcribe_returns_the_transcript_and_its_milliseconds(orchestrator):
    transcript, stt_ms = orchestrator.transcribe(b"\x1aE\xdf\xa3webm", "audio/webm")

    assert isinstance(transcript, Transcript)
    assert transcript.text == FakeSTT.TEXT
    assert type(stt_ms) is int and stt_ms >= 0


# --- history ---


def test_the_llm_sees_the_new_user_turn_last_and_the_session_gets_both_turns(orchestrator, session, llm):
    orchestrator.turn(session, "First message")
    orchestrator.turn(session, "Second message")

    history = llm.calls[1]["history"]
    assert [(turn.role, turn.text) for turn in history] == [
        ("user", "First message"),
        ("tutor", "Moien! Hello! How was your day?"),
        ("user", "Second message"),
    ]
    assert [turn.role for turn in session.turns] == ["user", "tutor", "user", "tutor"]
    assert session.turns[-1].segments == [Segment("lb", "Moien!"), Segment("en", "Hello! How was your day?")]
    assert session.turns[-1].mode == "chat"


def test_stored_history_is_capped_at_30_turns(orchestrator, session):
    for number in range(17):
        orchestrator.turn(session, f"message {number}")

    assert len(session.turns) == 30
    assert session.turns[0].text == "message 2"
    assert (session.turns[-2].text, session.turns[-1].role) == ("message 16", "tutor")


@pytest.mark.parametrize("failure", ["llm", "tts", "phrases"])
def test_a_failed_turn_leaves_the_session_history_unchanged(orchestrator, session, llm, monkeypatch, failure):
    orchestrator.turn(session, "First message")
    before = list(session.turns)
    if failure == "llm":
        llm.queue.append(LLMError("timeout", "claude gave no reply within 90 s"))
    elif failure == "tts":
        monkeypatch.setattr(orchestrator.tts, "synthesize", lambda segments: 1 / 0)
    else:
        monkeypatch.setattr(orchestrator.phrases, "_write", lambda entries: (_ for _ in ()).throw(OSError("disk full")))

    error = turn_error(orchestrator, session, "Second message")

    assert error.status in (500, 502)
    assert session.turns == before


def test_blank_text_is_a_400_without_an_llm_call(orchestrator, session, llm):
    for text in ("", "   \n"):
        error = turn_error(orchestrator, session, text)
        assert (error.status, error.error, error.fix) == (400, "empty message", "")

    assert llm.calls == []
    assert session.turns == []


def test_unknown_mode_is_a_400_without_an_llm_call(orchestrator, session, llm):
    assert turn_error(orchestrator, session, mode="lecture").status == 400
    assert llm.calls == []


# --- mode ---


@pytest.mark.parametrize(
    "text",
    [
        "Can you EXPLAIN that?",
        "Can you explain how to conjugate goen?",
        "Could you explain the word order?",
        "Please explain that word",
        "Explain the n-rule",
        "How do you say good morning?",
        "How would I say that politely?",
        "how can we pronounce this",
        "Why is it de and not den?",
        "how do I conjugate goen",
        "What is the conjugation of to be?",
        "I want some grammar now",
        "Teacher mode please",
        "What does moien mean?",
        ENGLISH_QUESTION,
        "How does this verb work?",
        "How is the sentence built?",
        "What is dog in Luxembourgish?",
        "What's the word for bread?",
        "How do you call a cat in Luxembourgish?",
        "Is there a verb for that?",
        "What is the plural of Hond?",
        "What is the past tense of goen?",
        "What is the difference between kennen and wëssen?",
        DAY_QUESTION,
        "How do I ask for the bill in a restaurant?",
        "How can I greet my neighbour?",
        "How do I get home in Luxembourgish?",
        "How should we apologise for being late?",
        "And how does one say cheers?",
    ],
)
def test_teacher_triggers_switch_auto_to_teacher(text):
    assert TEACHER_TRIGGERS.search(text)
    assert pick_mode(text) == "teacher"


@pytest.mark.parametrize(
    "text",
    [
        "I had coffee with my sister.",
        "What does she do? I mean, nothing.",
        "Why not?",
        "My son's teacher called this morning.",
        "I had to explain the delay to my boss.",
        "Explaining it took a while.",
        "How do you feel today?",
        "I had a great day at work in Luxembourg.",
        "My son's teacher called today.",
        "We talked about the difference in price.",
        "What is that? We heard it in Luxembourgish on the radio.",
        "How do I get home from here?",
        "I asked my boss how I could describe the problem.",
        "I went for a walk in the park and then cooked dinner for my family.",
        "How do I get there? They only spoke in Luxembourgish.",
    ],
)
def test_other_text_stays_in_chat(text):
    assert pick_mode(text) == "chat"


def test_the_mode_of_the_caller_wins(orchestrator, session, llm):
    assert orchestrator.turn(session, "Please explain the grammar", mode="chat").mode == "chat"
    assert orchestrator.turn(session, "I had coffee", mode="teacher").mode == "teacher"
    assert [call["mode"] for call in llm.calls] == ["chat", "teacher"]
    assert llm.calls[0]["grounding"] == ""


# --- teacher grounding ---


def test_candidate_words_keep_diacritics_split_apostrophes_and_stop_at_12():
    assert candidate_words("Ech ginn an d'Bäckerei, ech GINN!", "wéi geet et? 123 abc_def") == [
        "Ech",
        "ginn",
        "Bäckerei",
        "wéi",
        "geet",
        "abc",
        "def",
    ]
    many = " ".join(f"{letter}{letter}{letter}" for letter in "abcdefghijklmnop")
    assert candidate_words(many) == many.split()[:12]


def test_candidate_words_stop_at_the_limit_they_are_given():
    many = " ".join(f"{letter}{letter}{letter}" for letter in "abcdefghijklmnop")

    assert candidate_words(many, limit=14) == many.split()[:14]
    assert candidate_words("aaa bbb", many, limit=3) == ["aaa", "bbb", "ccc"]
    assert candidate_words(many, limit=30) == many.split()


def test_checked_claim_verifies_a_form_against_the_store_it_is_given(small):
    assert checked_claim(small, {"form": "du gees.", "lemma": "goen"}) == {
        "form": "du gees.",
        "lemma": "goen",
        "verified": True,
    }
    assert checked_claim(small, {"form": "gees"}) == {"form": "gees", "lemma": "", "verified": True}
    assert checked_claim(small, {"form": "goest", "lemma": "goen"})["verified"] is False
    assert checked_claim(None, {"form": "gees", "lemma": "goen"})["verified"] is False


def test_conjugate_goen_goes_to_teacher_mode_with_the_goen_table_and_the_rules(orchestrator, session, llm):
    result = orchestrator.turn(session, "how do I conjugate goen")

    assert result.mode == "teacher"
    (call,) = llm.calls
    assert call["mode"] == "teacher"
    grounding = call["grounding"]
    assert grounding.startswith("LOD facts (")
    assert '- goen [GOEN1] VRB: "to go"; n-rule form goe' in grounding
    assert GOEN_PRESENT in grounding
    assert "not found in LOD: how, conjugate" in grounding
    assert "\n\n# Luxembourgish rules sheet" in grounding
    assert "Eifeler Regel" in grounding
    assert session.turns[-1].mode == "teacher"


def test_grounding_also_looks_up_the_words_of_the_previous_tutor_turn(orchestrator, session, llm):
    llm.queue.append(reply_of(("lb", "Kacka!"), ("en", "A word for children. What happened?")))
    orchestrator.turn(session, "My toddler said something funny")

    orchestrator.turn(session, "Can you explain that word?")

    assert llm.calls[0]["grounding"] == ""
    assert '- Kacka [KACKA1] SUBST: "poo"' in llm.calls[1]["grounding"]


def test_english_terms_are_the_content_words_and_the_two_word_phrases_in_order():
    assert english_terms(ENGLISH_QUESTION) == [
        "i have",
        "have a",
        "have",
        "a dog",
        "dog",
        "word have",
        "have work",
        "work",
    ]
    assert english_terms("how do I say thank you") == ["say thank", "thank you", "thank"]
    # No phrase across punctuation, and an apostrophe ends a word.
    assert english_terms("Tired? What's the word for dog, please?") == ["tired", "s the", "for dog", "dog"]
    assert english_terms("Do you have the time?") == ["you have", "have the", "have", "the time", "time"]
    assert english_terms("") == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("How do you say 'do' in Luxembourgish?", ["do"]),
        ('What does "be" mean?', ["be"]),
        ("What does “thank you” mean?", ["thank you", "thank", "you"]),
        ("how do I say do", ["say do", "do"]),
        ("How does the word go work?", ["word go", "go work", "go", "work"]),
        ("What is the word for you?", ["for you", "you"]),
        ("How does be work?", ["does be", "be work", "be", "work"]),
        ("What is to go in Luxembourgish?", ["to go", "go"]),
        ("What is the word for water in Luxembourgish?", ["for water", "water"]),
        ("How do you say I am tired in Luxembourgish?", ["say i", "i am", "i", "am tired", "am", "tired"]),
    ],
    ids=[
        "quotes",
        "double-quotes",
        "quoted-phrase",
        "say",
        "the-word",
        "the-word-for",
        "how-does-work",
        "what-is-to",
        "what-is-the-word-for",
        "how-do-you-say-in",
    ],
)
def test_a_quoted_word_or_the_word_asked_about_is_kept_whatever_it_is(text, expected):
    assert english_terms(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("How do I conjugate work?", ["conjugate work", "to work"]),
        ("How do I conjugate do?", ["conjugate do", "to do"]),
        ("How do I conjugate the verb to be?", ["to be"]),
        ("Is water the verb water?", ["is water", "water the", "water", "verb water", "to water"]),
        ("I want to drink water.", ["i want", "want to", "want", "to drink", "drink water", "drink", "water"]),
    ],
    ids=["conjugate", "conjugate-stop-word", "the-verb-to", "the-verb", "after-to"],
)
def test_a_word_asked_about_as_a_verb_is_given_with_to(text, expected):
    assert english_terms(text) == expected


def test_a_question_asked_in_english_is_grounded_through_the_english_gloss(orchestrator, session, llm):
    result = orchestrator.turn(session, "How do I say April? And how does the word go work?")

    assert result.mode == "teacher"
    assert lod_facts(llm.calls[0]["grounding"])[1:] == [
        'English "april" -> Abrëll',
        '- Abrëll [ABRELL1] SUBST: "April"; gender M',
        'English "go" -> goen',
        '- goen [GOEN1] VRB: "to go"; n-rule form goe',
        "  auxiliary sinn; past participle gaangen / gaang",
        GOEN_PRESENT,
        "  past simple: ech goung, du goungs, hien/si/et goung, mir goungen, dir goungt, si goungen",
        "  imperative: géi! (du), gitt! (dir)",
        "not found in LOD: How, say, And, does, the, word, work",
    ]


def test_english_words_are_taken_from_the_user_text_not_from_the_tutor_turn(orchestrator, session, llm):
    llm.queue.append(reply_of(("lb", "Moien!"), ("en", "Did the baby do a poo in April?")))
    orchestrator.turn(session, "My toddler said something funny")

    orchestrator.turn(session, "Can you explain that word?")

    assert lod_facts(llm.calls[1]["grounding"])[1:] == [
        "not found in LOD: Can, you, explain, that, word, Moien, Did, the, baby, poo, April"
    ]


@pytest.mark.models
def test_the_english_question_about_have_gets_the_hunn_table_and_the_hond_entry(settings, session, llm, real_lod):
    orchestrator = Orchestrator(FakeSTT(), FakeTTS(), llm, real_lod, settings)

    result = orchestrator.turn(session, ENGLISH_QUESTION)

    assert result.mode == "teacher"
    facts = lod_facts(llm.calls[0]["grounding"])
    assert facts[1:3] == ['English "have" -> hunn, disposéieren', '- hunn [HUNN1] VRB: "to have"; n-rule form hu']
    assert HUNN_PRESENT in facts
    assert 'English "dog" -> Hond, Ridd' in facts
    assert 'English "work" -> Aarbecht, Tätegkeet, déngen, wierken, schaffen' in facts
    assert '- Hond [HOND1] SUBST: "dog"; gender M; plural Hënn / Honn' in facts
    assert facts[-1] == "not found in LOD: How, does, the, word, Luxembourgish"


@pytest.mark.models
def test_say_thank_you_is_grounded_on_merci_and_conjugate_goen_as_before(settings, session, llm, real_lod):
    orchestrator = Orchestrator(FakeSTT(), FakeTTS(), llm, real_lod, settings)

    orchestrator.turn(session, "how do I say thank you")
    orchestrator.turn(SessionStore().new(), "how do I conjugate goen")

    thank_you, goen = (lod_facts(call["grounding"]) for call in llm.calls)
    assert thank_you[1:] == [
        'English "thank you" -> merci, Dank',
        '- merci [MERCI2] INTERJ: "thank you"',
        '- Dank [DANK2] SUBST: "thank you"; gender M; register VEREELZT',
        "not found in LOD: how, say",
    ]
    assert goen[1:] == [
        '- goen [GOEN1] VRB: "to go"; n-rule form goe',
        "  auxiliary sinn; past participle gaangen / gaang",
        GOEN_PRESENT,
        "  past simple: ech goung, du goungs, hien/si/et goung, mir goungen, dir goungt, si goungen",
        "  imperative: géi! (du), gitt! (dir)",
        "not found in LOD: how, conjugate",
    ]


@pytest.mark.models
def test_the_facts_block_of_a_long_question_stays_within_the_caps(settings, session, llm, real_lod):
    orchestrator = Orchestrator(FakeSTT(), FakeTTS(), llm, real_lod, settings)
    question = "Please explain these verbs: eat, drink, come, give, take, make, get, run and see."

    orchestrator.turn(session, question)

    facts = lod_facts(llm.calls[0]["grounding"])
    english = [index for index, line in enumerate(facts) if line.startswith("English ")]
    # Nine verbs with entries; "give" brings the fifteenth entry, so the block ends there.
    assert [facts[index] for index in english] == [
        'English "eat" -> iessen, friessen, nammen',
        'English "drink" -> saufen, drénken, becheren',
        'English "come" -> kommen, matfueren, matkommen, Jitz',
        'English "give" -> ginn, matginn, schenken',
    ]
    assert len([line for line in facts[english[0] :] if line.startswith("- ")]) == 15
    # The verbs the caps left out are in LOD, so they are not reported as missing.
    assert facts[-1] == "not found in LOD: Please, explain, these, verbs"


@pytest.mark.models
def test_questions_about_short_words_and_nouns_are_grounded_in_auto_mode(settings, llm, real_lod):
    orchestrator = Orchestrator(FakeSTT(), FakeTTS(), llm, real_lod, settings)
    questions = (
        "What is the word for water in Luxembourgish?",
        "What is to go in Luxembourgish?",
        "How does the verb be work?",
    )

    modes = [orchestrator.turn(SessionStore().new(), question).mode for question in questions]

    assert modes == ["teacher"] * 3
    water, go, be = (lod_facts(call["grounding"]) for call in llm.calls)
    assert water[1:3] == [
        'English "water" -> Waasser, Gewässer, nätzen, dränken, tréinen',
        '- Waasser [WAASSER2] SUBST: "water"; gender N; plural Waasser',
    ]
    assert water[-1] == "not found in LOD: What, the, word, for, Luxembourgish"
    assert [line for line in go if line.startswith("English")] == ['English "to go" -> goen, higoen, dohigoen']
    assert 'English "to be" -> ginn, sinn, befannen' in be
    assert "  present: ech sinn, du bass, hien/si/et ass, mir sinn, dir sidd, si sinn" in be


def test_teacher_mode_without_a_store_says_so_and_verifies_nothing(settings, session, llm):
    orchestrator = Orchestrator(FakeSTT(), FakeTTS(), llm, None, settings)
    llm.queue.append(TEACHER_REPLY)

    result = orchestrator.turn(session, "how do I conjugate goen")

    grounding = llm.calls[0]["grounding"]
    assert "the dictionary store is not installed" in grounding
    assert "not verified in LOD" in grounding
    assert "# Luxembourgish rules sheet" in grounding
    assert [claim["verified"] for claim in result.claims] == [False, False]


# --- claims ---


def test_claims_are_verified_against_the_store_in_a_new_list(orchestrator, session, llm):
    llm.queue.append(TEACHER_REPLY)

    result = orchestrator.turn(session, "how do I conjugate goen")

    assert result.claims == [
        {"form": "gees", "lemma": "goen", "verified": True},
        {"form": "goest", "lemma": "goen", "verified": False},
    ]
    assert result.reply.claims == [{"form": "gees", "lemma": "goen"}, {"form": "goest", "lemma": "goen"}]
    assert result.shape_warning == ""
    assert len(result.audio.segment_offsets) == 6


def test_a_claim_without_a_lemma_is_checked_as_a_form(orchestrator, session, llm):
    claims = [{"form": "geet", "lemma": ""}, {"form": "gees", "lemma": "ausfléien"}]
    llm.queue.append(replace(TEACHER_REPLY, claims=claims))

    result = orchestrator.turn(session, "how do I conjugate goen")

    assert [claim["verified"] for claim in result.claims] == [True, False]


@pytest.mark.parametrize(
    ("form", "expected"),
    [
        ("d'Waasser", "Waasser"),
        ("D’Waasser.", "Waasser"),
        ("du gees", "gees"),
        ("Hien geet!", "geet"),
        ("den Abrëll,", "Abrëll"),
        (" gees? ", "gees"),
        ("gees", "gees"),
        ("du", "du"),
        ("d'", "d'"),
        ("dugees", "dugees"),
        ("", ""),
    ],
)
def test_bare_form_drops_closing_punctuation_and_one_article_or_pronoun(form, expected):
    assert bare_form(form) == expected


def test_a_claim_with_an_article_or_pronoun_in_front_is_checked_without_it(orchestrator, session, llm):
    claims = [
        {"form": "du gees", "lemma": "goen"},
        {"form": "d'Kacka", "lemma": "Kacka"},
        {"form": "den Abrëll.", "lemma": "Abrëll"},
        {"form": "geet,", "lemma": ""},
        {"form": "d'Xyzzy", "lemma": "Xyzzy"},
        {"form": "du goest", "lemma": "goen"},
        {"form": "du gees", "lemma": "Kacka"},
    ]
    llm.queue.append(replace(TEACHER_REPLY, claims=claims))

    result = orchestrator.turn(session, "how do I conjugate goen")

    assert [claim["verified"] for claim in result.claims] == [True, True, True, True, False, False, False]
    assert [(claim["form"], claim["lemma"]) for claim in result.claims] == [
        (claim["form"], claim["lemma"]) for claim in claims
    ]
    assert all(set(claim) == {"form", "lemma", "verified"} for claim in result.claims)


@pytest.mark.models
def test_a_noun_claimed_with_its_article_is_verified_in_the_real_store(settings, session, llm, real_lod):
    orchestrator = Orchestrator(FakeSTT(), FakeTTS(), llm, real_lod, settings)
    claims = [{"form": "d'Waasser", "lemma": "Waasser"}, {"form": "d'Xyzzy", "lemma": "Xyzzy"}]
    llm.queue.append(replace(TEACHER_REPLY, claims=claims))

    result = orchestrator.turn(session, "What is the word for water in Luxembourgish?")

    assert result.claims == [
        {"form": "d'Waasser", "lemma": "Waasser", "verified": True},
        {"form": "d'Xyzzy", "lemma": "Xyzzy", "verified": False},
    ]


# --- Luxembourgish words in English segments ---


def test_split_foreign_gives_each_form_in_an_english_segment_its_own_lb_segment():
    assert split_foreign(MIXED_REPLY.segments, GOEN_FORMS, ENGLISH) == MIXED_SPLIT


def test_split_foreign_leaves_a_form_that_is_also_an_english_word():
    segments = [Segment("lb", "Ech war."), Segment("en", "The word war means was, not a war. And you?")]

    assert split_foreign(segments, {"war"}, ENGLISH) == segments


def test_split_foreign_returns_a_segment_without_a_form_as_it_is():
    segments = [Segment("lb", "Ech ginn."), Segment("en", "That means I go."), Segment("en", "Say geet now. Ready?")]

    result = split_foreign(segments, GOEN_FORMS, ENGLISH)

    assert result[0] is segments[0] and result[1] is segments[1]
    assert result[2:] == [Segment("en", "Say"), Segment("lb", "geet"), Segment("en", "now. Ready?")]


def test_split_foreign_merges_forms_with_no_english_between_them():
    segments = [Segment("en", "The forms gees, geet and Ginn differ."), Segment("en", "Shall we go on?")]

    assert split_foreign(segments, GOEN_FORMS, ENGLISH) == [
        Segment("en", "The forms"),
        Segment("lb", "gees, geet"),
        Segment("en", "and"),
        Segment("lb", "Ginn"),
        Segment("en", "differ."),
        Segment("en", "Shall we go on?"),
    ]


def test_split_foreign_leaves_no_segment_of_punctuation_and_ends_the_reply_in_english():
    segments = [Segment("en", "The word is d'Waasser."), Segment("en", "Can you say d'Waasser?")]

    assert split_foreign(segments, {"waasser"}, ENGLISH) == [
        Segment("en", "The word is"),
        Segment("lb", "d'Waasser."),
        Segment("en", "Can you say d'Waasser?"),
    ]


def test_split_foreign_leaves_lb_segments_alone_and_changes_nothing_the_second_time():
    segments = [Segment("lb", "du gees"), *MIXED_REPLY.segments, Segment("en", "Now say gees, geet?")]

    once = split_foreign(segments, GOEN_FORMS, ENGLISH)

    assert once == [segments[0], *MIXED_SPLIT, segments[-1]]
    assert once[0] is segments[0]
    assert split_foreign(once, GOEN_FORMS, ENGLISH) == once


def test_split_foreign_takes_a_word_of_the_lb_segments_and_the_run_of_their_words_around_it():
    said = "That means I am tired. Notice the pattern mäin Dag war plus a word"
    segments = [*DAY_SENTENCES, Segment("en", f"{said}, would you like to add more describing words?")]

    once = split_foreign(segments, set(), ENGLISH)

    assert once == [
        *DAY_SENTENCES,
        Segment("en", "That means I am tired. Notice the pattern"),
        Segment("lb", "mäin Dag war"),
        Segment("en", "plus a word, would you like to add more describing words?"),
    ]
    assert split_foreign(once, set(), ENGLISH) == once


def test_split_foreign_takes_a_whole_sentence_quoted_in_english_in_the_casing_it_has_there():
    segments = [*DAY_SENTENCES, Segment("en", "say mäin DAG war gutt for a good day"), Segment("en", "Ready?")]

    once = split_foreign(segments, set(), ENGLISH)

    assert once[3:] == [
        Segment("en", "say"),
        Segment("lb", "mäin DAG war gutt"),
        Segment("en", "for a good day"),
        Segment("en", "Ready?"),
    ]
    assert split_foreign(once, set(), ENGLISH) == once


def test_split_foreign_takes_an_unclaimed_word_of_an_lb_segment_and_leaves_a_word_found_nowhere():
    english = "This verb uses sinn, not hunn, for the past"
    segments = [Segment("lb", "Ech sinn gaangen."), Segment("en", english), Segment("en", "Shall we go on?")]

    once = split_foreign(segments, set(), ENGLISH)

    assert once == [
        segments[0],
        Segment("en", "This verb uses"),
        Segment("lb", "sinn"),
        Segment("en", ", not hunn, for the past"),
        segments[2],
    ]
    assert split_foreign(once, set(), ENGLISH) == once


def test_split_foreign_leaves_an_english_look_alike_that_is_not_next_to_a_luxembourgish_word():
    segments = [*DAY_SENTENCES, Segment("en", "the war was long"), Segment("en", "The war, Dag war, and more. Ready?")]

    assert split_foreign(segments, {"war"}, ENGLISH) == [
        *segments[:4],
        Segment("en", "The war,"),
        Segment("lb", "Dag war"),
        Segment("en", ", and more. Ready?"),
    ]


def test_split_foreign_does_not_extend_a_run_over_a_word_that_is_in_no_lb_segment():
    segments = [*DAY_SENTENCES, Segment("en", "Then say war ass mäin Dag for it."), Segment("en", "Ready?")]

    assert split_foreign(segments, {"hunn"}, ENGLISH)[3:] == [
        Segment("en", "Then say war ass"),
        Segment("lb", "mäin Dag"),
        Segment("en", "for it."),
        Segment("en", "Ready?"),
    ]


def test_stated_forms_are_the_claimed_forms_and_headwords_as_bare_forms():
    reply = TutorReply(
        segments=[Segment("lb", "D'Waasser."), Segment("en", "Water."), Segment("lb", "Ech drénken Waasser.")],
        claims=[{"form": "du gees", "lemma": "goen"}, {"form": "Geet", "lemma": "Goen"}, {"form": " ", "lemma": ""}],
    )

    assert stated_forms(reply) == {"gees", "geet", "goen"}


@pytest.mark.models
def test_an_unclaimed_run_of_luxembourgish_in_english_is_split_out_in_a_turn(settings, session, llm, real_lod):
    orchestrator = Orchestrator(FakeSTT(), FakeTTS(), llm, real_lod, settings)
    said = "That means I am tired. Notice the pattern mäin Dag war plus a word"
    english = Segment("en", f"{said}, would you like to add more describing words?")
    llm.queue.append(TutorReply([*DAY_SENTENCES, Segment("en", "The war was long."), english], mode="teacher"))

    result = orchestrator.turn(session, DAY_QUESTION)

    assert result.reply.segments == [
        *DAY_SENTENCES,
        Segment("en", "The war was long."),
        Segment("en", "That means I am tired. Notice the pattern"),
        Segment("lb", "mäin Dag war"),
        Segment("en", "plus a word, would you like to add more describing words?"),
    ]


def test_the_headword_of_a_claim_is_split_out_of_an_english_segment_too(orchestrator, session, llm):
    english = "In the present tense goen changes its vowel. Where do you go today?"
    segments = [Segment("lb", "Ech ginn."), Segment("en", english)]
    llm.queue.append(replace(MIXED_REPLY, segments=segments, claims=[{"form": "ginn", "lemma": "goen"}]))

    result = orchestrator.turn(session, "how do I conjugate goen")

    assert result.reply.segments == [
        Segment("lb", "Ech ginn."),
        Segment("en", "In the present tense"),
        Segment("lb", "goen"),
        Segment("en", "changes its vowel. Where do you go today?"),
    ]


def test_a_teacher_reply_is_split_after_the_shape_check_for_the_voice_and_the_history(orchestrator, session, llm):
    llm.queue.append(MIXED_REPLY)

    result = orchestrator.turn(session, "how do I conjugate goen")

    assert len(llm.calls) == 1
    assert result.shape_warning == ""
    assert result.reply.segments == MIXED_SPLIT
    assert [offset["lang"] for offset in result.audio.segment_offsets] == [segment.lang for segment in MIXED_SPLIT]
    assert session.turns[-1].segments == MIXED_SPLIT
    assert session.turns[-1].text == " ".join(segment.text for segment in MIXED_REPLY.segments)
    assert [claim["verified"] for claim in result.claims] == [True, True, True]
    assert len(MIXED_REPLY.segments) == 2


def test_a_chat_reply_and_a_reply_without_a_store_are_not_split(orchestrator, session, settings, llm):
    llm.queue += [MIXED_REPLY, MIXED_REPLY]

    chat = orchestrator.turn(session, "how do I conjugate goen", mode="chat")
    no_store = Orchestrator(FakeSTT(), FakeTTS(), llm, None, settings).turn(session, "how do I conjugate goen")

    assert (chat.mode, no_store.mode) == ("chat", "teacher")
    assert chat.reply.segments == no_store.reply.segments == MIXED_REPLY.segments


# --- shape check ---


def test_an_en_first_reply_is_asked_again_once_and_the_second_reply_is_used(orchestrator, session, llm):
    llm.queue.append(EN_FIRST)

    result = orchestrator.turn(session, "Hello there")

    assert len(llm.calls) == 2
    assert "broke the format" not in llm.calls[0]["grounding"]
    assert llm.calls[1]["grounding"] == (
        "Your last reply broke the format: segment 0 must have lang lb;"
        " in chat mode every segment after segment 0 must have lang en"
    )
    assert llm.calls[1]["history"] == llm.calls[0]["history"]
    assert result.reply.segments[0] == Segment("lb", "Moien!")
    assert result.shape_warning == ""
    assert session.turns[-1].text == "Moien! Hello! How was your day?"


def test_a_second_broken_reply_is_accepted_with_a_shape_warning(orchestrator, session, llm):
    llm.queue += [EN_FIRST, reply_of(("lb", "Moien!"))]

    result = orchestrator.turn(session, "Hello there")

    assert len(llm.calls) == 2
    assert result.reply.segments == [Segment("lb", "Moien!")]
    assert result.shape_warning == "at least one segment must have lang en"


def test_the_reask_note_follows_the_teacher_grounding(orchestrator, session, llm):
    llm.queue.append(EN_FIRST)

    orchestrator.turn(session, "how do I conjugate goen")

    first, second = (call["grounding"] for call in llm.calls)
    assert second.startswith(first + "\n\nYour last reply broke the format: segment 0 must have lang lb")


def test_teacher_shape_accepts_alternating_segments_and_chat_shape_rejects_them():
    assert shape_violations(TEACHER_REPLY, "teacher") == []
    assert shape_violations(TEACHER_REPLY, "chat") == ["in chat mode every segment after segment 0 must have lang en"]


@pytest.mark.parametrize(
    ("reply", "mode", "expected"),
    [
        (reply_of(("lb", "Moien!"), ("en", "Hello!")), "chat", []),
        (reply_of(("lb", words(12)), ("en", words(77))), "chat", []),
        (reply_of(("lb", words(13)), ("en", "Hello!")), "chat", ["segment 0 must be one to twelve words, it has 13"]),
        (reply_of(("lb", "Moien!"), ("en", words(89))), "chat", ["a chat reply must be under 90 words, it has 90"]),
        (reply_of(("lb", "Moien!"), ("en", words(89))), "teacher", []),
        (reply_of(("lb", "Moien!"), ("en", words(218))), "teacher", []),
        (
            reply_of(("lb", "Moien!"), ("en", words(219))),
            "teacher",
            ["a teacher reply must be under 220 words, it has 220"],
        ),
        (reply_of(("lb", "Moien!"), ("en", " \n")), "chat", ["segment 1 has no text"]),
        (reply_of(("lb", "Moien!"), ("lb", "Äddi!")), "teacher", ["at least one segment must have lang en"]),
        (
            reply_of(("lb", " "), ("en", "Hello!")),
            "chat",
            ["segment 0 has no text", "segment 0 must be one to twelve words, it has 0"],
        ),
    ],
    ids=[
        "ok",
        "89-words",
        "long-phrase",
        "chat-90",
        "teacher-90",
        "teacher-219",
        "teacher-220",
        "empty-segment",
        "no-en",
        "empty-phrase",
    ],
)
def test_shape_violations(reply, mode, expected):
    assert shape_violations(reply, mode) == expected


# --- mix level ---


def test_the_four_mix_levels_are_numbered_from_0_and_named():
    assert [level["value"] for level in MIX_LEVELS] == [0, 1, 2, 3]
    assert [level["name"] for level in MIX_LEVELS] == [
        "Opening phrase only",
        "A phrase here and there",
        "Phrases and sentences",
        "Mostly Luxembourgish",
    ]
    assert all(set(level) == {"value", "name", "hint"} and level["hint"] for level in MIX_LEVELS)


@pytest.mark.parametrize(
    ("mix", "default", "expected"),
    [(None, 0, 0), (None, 2, 2), ("", 3, 3), (0, 3, 0), (3, 0, 3), ("2", 0, 2), (None, 9, 3), (None, -1, 0)],
)
def test_checked_mix_is_the_level_or_the_default_within_the_levels(mix, default, expected):
    assert checked_mix(mix, default) == expected


@pytest.mark.parametrize("mix", [4, -1, 1.0, True, "4", "-1", "two", "1.5", " 1", [1]])
def test_checked_mix_refuses_what_is_no_level(mix):
    with pytest.raises(TurnError) as raised:
        checked_mix(mix, 2)

    assert raised.value.status == 400
    assert raised.value.error == f"unknown mix level {mix!r}; known: 0 to 3"
    assert raised.value.fix == "send `mix` as a whole number from 0 to 3, or leave it out"


@pytest.mark.parametrize(
    ("reply", "mix", "expected"),
    [
        (LEVEL_1_REPLY, 1, []),
        (LEVEL_2_REPLY, 2, []),
        (LEVEL_3_REPLY, 3, []),
        (LEVEL_1_REPLY, 0, ["in chat mode every segment after segment 0 must have lang en"]),
        (reply_of(("lb", "Moien!"), ("en", "Hello! How was your day?")), 1, [NO_LB_AFTER_OPENING]),
        (
            LEVEL_2_REPLY,
            1,
            ["at mix level 1 at most 30 percent of the words after segment 0 may be in lb segments, 37 percent are"],
        ),
        (
            reply_of(
                ("lb", "Dat ass schéin!"),
                ("en", "Starting the day with"),
                ("lb", "e Kaffi"),
                ("en", "sounds lovely. I hope you two had plenty of time. What did you talk about?"),
            ),
            2,
            ["at mix level 2 at least 20 percent of the words after segment 0 must be in lb segments, 10 percent are"],
        ),
        (
            reply_of(
                ("lb", "Dat ass schéin!"),
                ("lb", "Du hues Kaffi mat denger Schwëster gedronk, dat ass schéin!"),
                ("en", "I hope you two had plenty of time. What did you talk about?"),
            ),
            2,
            [],
        ),
        (
            LEVEL_2_REPLY,
            3,
            ["at mix level 3 at least 40 percent of the words after segment 0 must be in lb segments, 37 percent are"],
        ),
        (
            replace(LEVEL_2_REPLY, segments=[*LEVEL_2_REPLY.segments, Segment("lb", "Wéi geet et hir?")]),
            2,
            ["the last segment must have lang en"],
        ),
        (
            reply_of(("lb", "Moien!"), ("lb", words(45)), ("en", words(44))),
            3,
            ["a chat reply must be under 90 words, it has 90"],
        ),
    ],
    ids=[
        "level-1",
        "level-2",
        "level-3",
        "level-0",
        "no-lb-after-opening",
        "level-1-share",
        "level-2-share",
        "level-2-one-sentence",
        "level-3-share",
        "ends-in-lb",
        "90-words",
    ],
)
def test_shape_violations_of_a_chat_reply_at_a_mix_level(reply, mix, expected):
    assert shape_violations(reply, "chat", mix) == expected


@pytest.mark.parametrize(
    ("before", "word", "after", "lone"),
    [
        ("I hope the", "Kaffi", "was good. What did you two talk about today?", True),
        ("That sounds like a good start to the", "Dag.", "What did you two talk about?", True),
        ("That sounds lovely.", "Kaffi", "is always good. What did you two talk about?", True),
        ("That sounds lovely.", "Merci!", "What did you two talk about?", False),
        ("Is that right?", "“Jo.”", "Then tell me, what did you two talk about?", False),
    ],
    ids=["inside", "ends-the-sentence", "starts-the-sentence", "own-sentence", "own-sentence-in-quotes"],
)
def test_a_one_word_lb_segment_of_a_mixed_reply_must_be_a_sentence_of_its_own(before, word, after, lone):
    reply = reply_of(("lb", "Moien!"), ("en", before), ("lb", word), ("en", after))
    note = (
        f"segment 2 is a lone Luxembourgish word inside a sentence ({word}):"
        " switch a whole phrase or sentence, or keep the word in English"
    )

    for mix in (1, 3):
        assert (note in shape_violations(reply, "chat", mix)) is lone
    assert shape_violations(reply, "chat", 1) == ([note] if lone else [])
    assert shape_violations(reply_of(("lb", "Moien!"), ("lb", "Merci!"), ("en", after)), "chat", 1) == []


def test_teacher_shape_has_no_mix_level():
    assert shape_violations(TEACHER_REPLY, "teacher", 3) == shape_violations(TEACHER_REPLY, "teacher") == []


def test_level_0_gives_the_model_the_prompt_of_a_plain_chat_turn(orchestrator, session, llm):
    remember(orchestrator, ("Äddi!", "Bye!", stamp(1)))

    result = orchestrator.turn(session, "Hello there", mix=0)

    (call,) = llm.calls
    assert call["grounding"] == ""
    assert build_prompt(call["history"], call["mode"], call["grounding"]) == (
        "Conversation so far:\nUser: Hello there\n\nmode: chat\n\nReply to the last User line with the JSON object."
    )
    assert (result.mix, result.shape_warning) == (0, "")


def test_a_mixed_turn_gives_the_model_the_level_and_the_phrases_met_latest_first(orchestrator, session, llm):
    remember(
        orchestrator,
        ("Moien!", "Hello!", stamp(1)),
        ("Äddi!", "Bye!", stamp(3)),
        ("Wéi\ngeet  et?", "How are\nyou?", stamp(2)),
        ("Kaffi", " ", stamp(4)),
    )
    llm.queue.append(LEVEL_1_REPLY)

    result = orchestrator.turn(session, "I had coffee with my sister.", mix=1)

    (call,) = llm.calls
    assert call["grounding"].splitlines() == [
        "mix level: 1",
        MIX_HEADER,
        "Kaffi",
        "Äddi! = Bye!",
        "Wéi geet et? = How are you?",
        "Moien! = Hello!",
    ]
    assert (result.mode, result.mix, result.shape_warning) == ("chat", 1, "")
    assert result.reply.segments == LEVEL_1_REPLY.segments


def test_the_model_gets_at_most_40_phrases_and_is_told_when_there_are_none(orchestrator, session, llm):
    llm.queue += [LEVEL_2_REPLY, LEVEL_2_REPLY]

    orchestrator.turn(session, "I had coffee with my sister.", mix=2)
    remember(orchestrator, *((f"Saz {number}", f"sentence {number}", stamp(number)) for number in range(1, 46)))
    orchestrator.turn(session, "Then we went for a walk.", mix=2)

    assert llm.calls[0]["grounding"] == "mix level: 2\nThe learner has met no phrases yet."
    lines = llm.calls[1]["grounding"].splitlines()
    assert lines[:3] == ["mix level: 2", MIX_HEADER, "Saz 45 = sentence 45"]
    assert lines[-1] == "Saz 6 = sentence 6"
    assert len(lines) == 2 + MAX_MIX_PHRASES == 42


def test_without_a_level_a_turn_takes_the_chat_mix_setting(settings, session, llm, small):
    orchestrator = Orchestrator(FakeSTT(), FakeTTS(), llm, small, replace(settings, chat_mix=2))
    llm.queue += [LEVEL_2_REPLY, LEVEL_1_REPLY]

    by_setting = orchestrator.turn(session, "I had coffee with my sister.")
    by_turn = orchestrator.turn(session, "Then we went for a walk.", mix=1)

    assert (by_setting.mix, by_turn.mix) == (2, 1)
    assert [call["grounding"].splitlines()[0] for call in llm.calls] == ["mix level: 2", "mix level: 1"]
    assert (by_setting.shape_warning, by_turn.shape_warning) == ("", "")


@pytest.mark.parametrize("mix", [4, -1, "two", 1.5, True])
def test_a_mix_that_is_no_level_is_a_400_without_an_llm_call(orchestrator, session, llm, mix):
    for text in ("Hello there", "how do I conjugate goen"):
        error = turn_error(orchestrator, session, text, mix=mix)

        assert (error.status, error.error) == (400, f"unknown mix level {mix!r}; known: 0 to 3")
        assert "whole number from 0 to 3" in error.fix
    assert llm.calls == []
    assert session.turns == []


def test_teacher_mode_ignores_the_mix_level(orchestrator, session, llm):
    remember(orchestrator, ("Äddi!", "Bye!", stamp(1)))
    llm.queue += [TEACHER_REPLY, TEACHER_REPLY]

    plain = orchestrator.turn(SessionStore().new(), "how do I conjugate goen")
    mixed = orchestrator.turn(session, "how do I conjugate goen", mix=3)

    assert len(llm.calls) == 2
    assert llm.calls[1]["grounding"] == llm.calls[0]["grounding"]
    assert "mix level" not in llm.calls[1]["grounding"]
    assert (mixed.mode, mixed.mix, mixed.shape_warning) == ("teacher", 0, "")
    assert mixed.reply.segments == plain.reply.segments == TEACHER_REPLY.segments
    assert mixed.claims == plain.claims


def test_a_mixed_reply_that_is_english_after_the_opening_is_asked_again_once(orchestrator, session, llm):
    llm.queue += [reply_of(("lb", "Moien!"), ("en", "Hello! How was your day?")), LEVEL_1_REPLY]

    result = orchestrator.turn(session, "I had coffee with my sister.", mix=1)

    first, second = (call["grounding"] for call in llm.calls)
    assert first == "mix level: 1\nThe learner has met no phrases yet."
    assert second == f"{first}\n\nYour last reply broke the format: {NO_LB_AFTER_OPENING}"
    assert result.reply.segments == LEVEL_1_REPLY.segments
    assert result.shape_warning == ""


def test_a_second_reply_that_breaks_the_mix_level_is_accepted_with_a_shape_warning(orchestrator, session, llm):
    result = orchestrator.turn(session, "Hello there", mix=1)

    assert len(llm.calls) == 2
    assert result.reply.segments == [Segment("lb", "Moien!"), Segment("en", "Hello! How was your day?")]
    assert (result.mix, result.shape_warning) == (1, NO_LB_AFTER_OPENING)


def test_split_foreign_takes_the_words_of_the_phrases_like_those_of_the_lb_segments():
    english = frozenset({"an", "war", "man"})
    said = "An old man spoke of the war, and de Mann war an der Stad was all I got."
    segments = [Segment("lb", "Moien!"), Segment("en", said), Segment("en", "What did you do?")]

    once = split_foreign(segments, set(), english, ["De Mann war an der Stad."])

    assert split_foreign(segments, set(), english) == segments
    assert once == [
        segments[0],
        Segment("en", "An old man spoke of the war, and"),
        Segment("lb", "de Mann war an der Stad"),
        Segment("en", "was all I got."),
        segments[2],
    ]
    assert split_foreign(once, set(), english, ["De Mann war an der Stad."]) == once


def test_a_phrase_left_in_the_english_of_a_mixed_reply_is_split_out_before_the_shape_check(orchestrator, session, llm):
    llm.queue.append(INLINE_REPLY)

    result = orchestrator.turn(session, "I had coffee with my sister.", mix=1)

    assert len(llm.calls) == 1
    assert result.reply.segments == LEVEL_1_REPLY.segments
    assert result.shape_warning == ""
    assert [offset["lang"] for offset in result.audio.segment_offsets] == ["lb", "en", "lb", "en"]
    assert session.turns[-1].segments == LEVEL_1_REPLY.segments
    assert len(INLINE_REPLY.segments) == 2


def test_a_lone_word_split_out_of_a_mixed_reply_is_asked_again(orchestrator, session, llm):
    english = Segment("en", "I hope the Kaffi was good. What did you two talk about?")
    llm.queue += [TutorReply([OPENING, english], new_phrases=[{"lb": "Kaffi", "en": "coffee"}]), LEVEL_1_REPLY]

    result = orchestrator.turn(session, "I had coffee with my sister.", mix=1)

    assert llm.calls[1]["grounding"].endswith(
        "\n\nYour last reply broke the format: segment 2 is a lone Luxembourgish word inside a sentence (Kaffi):"
        " switch a whole phrase or sentence, or keep the word in English"
    )
    assert result.reply.segments == LEVEL_1_REPLY.segments
    assert result.shape_warning == ""


def test_a_mixed_reply_without_a_store_is_not_split(settings, session, llm):
    orchestrator = Orchestrator(FakeSTT(), FakeTTS(), llm, None, settings)
    llm.queue += [INLINE_REPLY, INLINE_REPLY]

    result = orchestrator.turn(session, "I had coffee with my sister.", mix=1)

    assert len(llm.calls) == 2
    assert result.reply.segments == INLINE_REPLY.segments
    assert result.shape_warning == NO_LB_AFTER_OPENING


@pytest.mark.models
def test_a_mixed_turn_on_the_real_store_splits_out_the_phrase_and_leaves_english_look_alikes(
    settings, session, llm, real_lod
):
    orchestrator = Orchestrator(FakeSTT(), FakeTTS(), llm, real_lod, settings)
    english = "An old man told me about the war, and I think de Mann war an der Stad says it well. What did he say?"
    phrase = {"lb": "De Mann war an der Stad.", "en": "The man was in town."}
    llm.queue.append(TutorReply([OPENING, Segment("en", english)], new_phrases=[phrase]))

    result = orchestrator.turn(session, "An old man told me about the war today.", mix=1)

    assert {"an", "war", "man"} <= real_lod.english_words()
    assert len(llm.calls) == 1
    assert result.reply.segments == [
        OPENING,
        Segment("en", "An old man told me about the war, and I think"),
        Segment("lb", "de Mann war an der Stad"),
        Segment("en", "says it well. What did he say?"),
    ]
    assert result.shape_warning == ""
    assert result.glosses == ["", "", "The man was in town.", ""]


def test_an_lb_segment_has_the_gloss_of_its_phrase_from_the_reply_or_the_memory(orchestrator, session, llm):
    remember(
        orchestrator,
        ("vill Zäit fir ze schwätzen", "plenty of time to talk", stamp(1)),
        ("Dat ass schéin", "That is lovely", stamp(2)),
    )
    llm.queue += [LEVEL_2_REPLY, LEVEL_3_REPLY]

    level_2 = orchestrator.turn(session, "I had coffee with my sister.", mix=2)
    level_3 = orchestrator.turn(session, "She is doing well.", mix=3)
    plain = orchestrator.turn(session, "Hello there")

    assert level_2.glosses == ["That is nice!", "a coffee with your sister", "", "plenty of time to talk", ""]
    assert level_3.glosses == ["That is nice!", "", "", "", ""]
    assert plain.glosses == ["Hello!", ""]


# --- errors ---


def test_auth_error_is_a_503_with_the_fix(orchestrator, session, llm):
    llm.queue.append(LLMError("auth", "Failed to authenticate: OAuth session expired"))

    error = turn_error(orchestrator, session)

    assert error.status == 503
    assert error.error == "Failed to authenticate: OAuth session expired"
    assert "claude auth login" in error.fix
    assert str(error) == error.error
    assert len(llm.calls) == 1


def test_auth_error_without_detail_still_has_a_message(orchestrator, session, llm):
    llm.queue.append(LLMError("auth"))

    error = turn_error(orchestrator, session)

    assert (error.status, error.error) == (503, "auth")
    assert "claude auth login" in error.fix


@pytest.mark.parametrize("kind", ["timeout", "schema", "other"])
def test_other_llm_errors_are_a_502(orchestrator, session, llm, kind):
    llm.queue.append(LLMError(kind, "what went wrong"))

    error = turn_error(orchestrator, session)

    assert (error.status, error.error, error.fix) == (502, "what went wrong", "")


def test_an_llm_error_on_the_reask_fails_the_turn(orchestrator, session, llm):
    llm.queue += [EN_FIRST, LLMError("timeout", "too slow")]

    assert turn_error(orchestrator, session).status == 502
    assert session.turns == []
    assert orchestrator.phrases.list() == []


def test_any_other_engine_exception_is_a_500_naming_the_engine(orchestrator, session, llm, monkeypatch):
    llm.queue.append(RuntimeError("fake engine is unavailable"))
    assert turn_error(orchestrator, session).error == (
        "llm engine (fake) failed: RuntimeError: fake engine is unavailable"
    )

    monkeypatch.setattr(orchestrator.tts, "name", "broken")
    monkeypatch.setattr(orchestrator.tts, "synthesize", lambda segments: 1 / 0)
    error = turn_error(orchestrator, session)
    assert (error.status, error.error) == (500, "tts engine (broken) failed: ZeroDivisionError: division by zero")

    monkeypatch.setattr(orchestrator.stt, "transcribe", lambda audio, mime: [][0])
    with pytest.raises(TurnError) as raised:
        orchestrator.transcribe(b"audio", "audio/wav")
    assert (raised.value.status, raised.value.error) == (
        500,
        "stt engine (fake) failed: IndexError: list index out of range",
    )


# --- sessions ---


def test_session_store_hands_out_distinct_sessions_and_finds_them():
    store = SessionStore()

    first, second = store.new(), store.new()

    assert first.id != second.id
    assert first.turns == [] and first.turns is not second.turns
    assert datetime.fromisoformat(first.created).utcoffset() == timedelta(0)
    assert store.get(first.id) is first
    assert store.get("no-such-session") is None


# --- phrase memory ---


def test_phrase_key_ignores_case_spacing_and_trailing_punctuation():
    assert phrase_key("  Wéi  geet et?! ") == phrase_key("wéi geet et") == "wéi geet et"
    assert phrase_key("Moien") != phrase_key("Moien, Anna")


def test_phrase_memory_adds_then_counts_repeats(tmp_path):
    memory = PhraseMemory(tmp_path / "data" / "phrases.json")
    assert memory.list() == []
    assert not (tmp_path / "data").exists()

    assert memory.add("Wéi geet et?", "How are you?", "session-a") is True
    assert memory.add("wéi geet et", "how is it going", "session-b") is False
    assert memory.add("WÉI GEET ET?!", "", "session-a") is False
    assert memory.add("Äddi!", "Bye!", "session-b") is True

    first, second = memory.list()
    assert (first["lb"], first["en"], first["times_seen"]) == ("Wéi geet et?", "How are you?", 3)
    assert first["sessions"] == ["session-a", "session-b"]
    assert datetime.fromisoformat(first["first_seen"]).utcoffset() == timedelta(0)
    assert first["first_seen"] <= first["last_seen"]
    assert set(first) == {"lb", "en", "first_seen", "last_seen", "times_seen", "sessions"}
    assert (second["lb"], second["times_seen"], second["sessions"]) == ("Äddi!", 1, ["session-b"])
    assert [path.name for path in (tmp_path / "data").iterdir()] == ["phrases.json"]
    assert "Äddi!" in memory.path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "content",
    [b"{not json", b"\xff\xfe\x00", b'{"lb": "Moien"}', b'[{"lb": "Moien"}]', b""],
    ids=["not-json", "not-utf8", "not-a-list", "missing-fields", "empty-file"],
)
def test_a_corrupt_phrase_file_is_moved_aside_and_a_fresh_memory_started(tmp_path, content):
    path = tmp_path / "phrases.json"
    path.write_bytes(content)
    memory = PhraseMemory(path)

    assert memory.add("Moien!", "Hello!", "session-a") is True

    assert (tmp_path / "phrases.json.corrupt").read_bytes() == content
    assert [entry["lb"] for entry in memory.list()] == ["Moien!"]


def test_phrase_memory_counts_every_add_from_several_threads(tmp_path):
    memory = PhraseMemory(tmp_path / "phrases.json")
    threads = [
        threading.Thread(target=lambda: [memory.add("Moien!", "Hello!", "session-a") for _ in range(5)])
        for _ in range(4)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    (entry,) = memory.list()
    assert entry["times_seen"] == 20


def test_every_new_phrase_of_a_reply_is_saved_and_only_new_ones_reported(orchestrator, session, llm):
    orchestrator.turn(session, "Hello there")
    phrases = [{"lb": "moien", "en": "hello"}, {"lb": "Äddi!", "en": "Bye!"}, {"lb": " ", "en": "nothing"}]
    llm.queue.append(replace(TEACHER_REPLY, new_phrases=phrases))

    result = orchestrator.turn(session, "how do I conjugate goen")

    assert result.phrases_added == ["Äddi!"]
    assert [(entry["lb"], entry["times_seen"]) for entry in orchestrator.phrases.list()] == [
        ("Moien!", 2),
        ("Äddi!", 1),
    ]


# --- wiring in the app ---


def test_lifespan_without_a_store_runs_and_health_reports_it(tmp_path):
    settings = replace(Settings.from_env(), data_dir=tmp_path / "data")

    with TestClient(create_app(settings)) as client:
        response = client.get("/api/health")
        state = client.app.state

        assert state.lod is None
        assert state.orchestrator.lod is None
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["lod"]["ready"] is False
        assert "run scripts/download_lod.py" in body["lod"]["detail"]
        result = state.orchestrator.turn(state.sessions.new(), "how do I conjugate goen")
        assert result.claims == []
        assert state.phrases.list()[0]["lb"] == "Moien!"


def test_lifespan_with_a_file_that_is_not_a_store_runs_without_it(tmp_path):
    settings = replace(Settings.from_env(), data_dir=tmp_path / "data")
    (settings.data_dir / "lod").mkdir(parents=True)
    (settings.data_dir / "lod" / "lod.sqlite").write_bytes(b"this is not a database, only some bytes. " * 200)

    with TestClient(create_app(settings)) as client:
        body = client.get("/api/health").json()

        assert client.app.state.lod is None
        assert body["status"] == "ok"
        assert body["lod"]["ready"] is False
        assert "DatabaseError" in body["lod"]["detail"]


def test_lifespan_wires_the_store_sessions_phrases_and_orchestrator(tmp_path, small_path):
    settings = replace(Settings.from_env(), data_dir=tmp_path / "data")
    (settings.data_dir / "lod").mkdir(parents=True)
    (settings.data_dir / "lod" / "lod.sqlite").write_bytes(small_path.read_bytes())

    with TestClient(create_app(settings)) as client:
        state = client.app.state

        assert client.get("/api/health").json()["lod"] == {"ready": True, "detail": ""}
        assert isinstance(state.lod, LodStore)
        assert isinstance(state.sessions, SessionStore)
        assert isinstance(state.orchestrator, Orchestrator)
        assert state.orchestrator.lod is state.lod
        assert (state.orchestrator.stt, state.orchestrator.tts, state.orchestrator.llm) == (
            state.stt,
            state.tts,
            state.llm,
        )
        assert state.phrases is state.orchestrator.phrases
        assert state.phrases.path == settings.data_dir / "phrases.json"
        session = state.sessions.new()
        assert state.orchestrator.turn(session, "how do I conjugate goen").mode == "teacher"
        assert GOEN_PRESENT in state.llm.calls[0]["grounding"]


# --- the real CLI ---


def show(label: str, value) -> None:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    encoding = sys.stdout.encoding or "utf-8"
    print(f"{label}: {text}".encode(encoding, "backslashreplace").decode(encoding))


class RecordingLLM:
    """Passes every call on to `engine` and keeps what went in and what came back."""

    def __init__(self, engine) -> None:
        self.engine = engine
        self.name = engine.name
        self.calls: list[dict] = []

    def reply(self, history: list[Turn], mode: str, grounding: str) -> TutorReply:
        reply = self.engine.reply(history, mode, grounding)
        self.calls.append({"mode": mode, "grounding": grounding, "reply": reply})
        return reply


@pytest.mark.live
def test_live_chat_turn_then_teacher_turn(request, tmp_path):
    if "live" not in request.config.getoption("markexpr"):
        pytest.skip("calls the real claude CLI on the user's subscription; run with -m live")
    cli_calls = []

    def runner(argv, **kwargs):
        cli_calls.append(argv[1])
        return subprocess.run(argv, **kwargs)

    settings = Settings(data_dir=tmp_path / "data")
    engine = ClaudeCliLLM(settings, runner=runner)
    engine.warmup()
    if not engine.ready():
        pytest.skip(engine.detail)
    lod_path = REAL_DATA / "lod" / "lod.sqlite"
    lod = LodStore(lod_path) if lod_path.is_file() else None
    llm = RecordingLLM(engine)
    orchestrator = Orchestrator(FakeSTT(), FakeTTS(), llm, lod, settings)
    session = SessionStore().new()

    print()
    show("LOD store", str(lod_path) if lod else "not installed")
    turns = (
        ("I walked to the bakery this morning and bought fresh bread.", "chat"),
        ("Can you explain how to conjugate goen?", "teacher"),
    )
    for text, mode in turns:
        seen = len(llm.calls)
        result = orchestrator.turn(session, text)
        calls = llm.calls[seen:]

        show(f"{mode} user", text)
        for number, call in enumerate(calls, start=1):
            segments = [[segment.lang, segment.text] for segment in call["reply"].segments]
            show(f"{mode} reply {number} segments", segments)
            show(f"{mode} reply {number} shape", shape_violations(call["reply"], mode) or "ok")
        show(f"{mode} new_phrases", result.reply.new_phrases)
        show(f"{mode} phrases_added", result.phrases_added)
        show(f"{mode} claims", result.claims)
        show(f"{mode} shape_warning", result.shape_warning or "none")
        show(f"{mode} timings", result.timings)

        assert result.mode == mode == result.reply.mode
        assert 1 <= len(calls) <= 2
        assert set(result.timings) == {"stt_ms", "llm_ms", "tts_ms", "total_ms"}
        assert result.timings["llm_ms"] > 0
        assert len(result.audio.segment_offsets) == len(result.reply.segments)
        assert all(type(claim["verified"]) is bool for claim in result.claims)
    show("claude -p calls", len([first for first in cli_calls if first == "-p"]))

    chat, teacher = llm.calls[0], llm.calls[-1]
    assert chat["grounding"] == ""
    assert "# Luxembourgish rules sheet" in teacher["grounding"]
    if lod:
        assert '- goen [GOEN1] VRB: "to go"' in teacher["grounding"]
        lod.close()
    assert len(session.turns) == 4


@pytest.mark.live
def test_live_chat_turn_at_mix_level_2_switches_phrases_and_ends_in_english(request, settings, session):
    if "live" not in request.config.getoption("markexpr"):
        pytest.skip("calls the real claude CLI on the user's subscription; run with -m live")
    engine = ClaudeCliLLM(settings)
    engine.warmup()
    if not engine.ready():
        pytest.skip(engine.detail)
    lod_path = REAL_DATA / "lod" / "lod.sqlite"
    lod = LodStore(lod_path) if lod_path.is_file() else None
    llm = RecordingLLM(engine)
    orchestrator = Orchestrator(FakeSTT(), FakeTTS(), llm, lod, settings)
    remember(
        orchestrator,
        ("Mäin Dag war gutt.", "My day was good.", stamp(1)),
        ("ech drénken Kaffi", "I drink coffee", stamp(2)),
        ("Moien, wéi geet et?", "Hello, how are you?", stamp(3)),
    )
    text = "I had a long day at work and then I cooked dinner for my family."

    result = orchestrator.turn(session, text, mix=2)

    print()
    show("LOD store", str(lod_path) if lod else "not installed")
    show("user", text)
    for number, call in enumerate(llm.calls, start=1):
        show(f"reply {number} segments", [[segment.lang, segment.text] for segment in call["reply"].segments])
    show("spoken segments", [[segment.lang, segment.text] for segment in result.reply.segments])
    show("new_phrases", result.reply.new_phrases)
    show("glosses", result.glosses)
    show("shape_warning", result.shape_warning or "none")
    show("timings", result.timings)
    if lod:
        lod.close()

    later = result.reply.segments[1:]
    assert (result.mode, result.mix) == ("chat", 2)
    assert 1 <= len(llm.calls) <= 2
    assert llm.calls[0]["grounding"].startswith("mix level: 2\n" + MIX_HEADER)
    assert any(segment.lang == "lb" for segment in later)
    assert result.reply.segments[-1].lang == "en"
    assert result.shape_warning == ""


def live_turn(request, settings, session, question: str, mode: str):
    """One turn through the real CLI and the real LOD store, printed: (result, LOD facts lines)."""
    if "live" not in request.config.getoption("markexpr"):
        pytest.skip("calls the real claude CLI on the user's subscription; run with -m live")
    cli_calls = []

    def runner(argv, **kwargs):
        if argv[1] == "-p":
            if len(cli_calls) == MAX_LIVE_CALLS:
                raise RuntimeError(f"this test may make {MAX_LIVE_CALLS} claude -p calls, not more")
            cli_calls.append(argv)
        return subprocess.run(argv, **kwargs)

    engine = ClaudeCliLLM(settings, runner=runner)
    engine.warmup()
    if not engine.ready():
        pytest.skip(engine.detail)
    llm = RecordingLLM(engine)
    orchestrator = Orchestrator(FakeSTT(), FakeTTS(), llm, request.getfixturevalue("real_lod"), settings)

    result = orchestrator.turn(session, question, mode)

    facts = lod_facts(llm.calls[0]["grounding"])
    print()
    show("user", question)
    show("mode", f"{mode} -> {result.mode}")
    show("effort", cli_calls[0][cli_calls[0].index("--effort") + 1])
    show("LOD facts", "\n" + "\n".join(facts))
    for number, call in enumerate(llm.calls, start=1):
        show(f"reply {number} segments", [[segment.lang, segment.text] for segment in call["reply"].segments])
        show(f"reply {number} shape", shape_violations(call["reply"], "teacher") or "ok")
    show("spoken segments", [[segment.lang, segment.text] for segment in result.reply.segments])
    show("claims", result.claims)
    show("shape_warning", result.shape_warning or "none")
    show("timings", result.timings)
    show("claude -p calls", len(cli_calls))

    assert result.mode == "teacher" == result.reply.mode
    assert cli_calls[0][-2:] == ["--effort", "low"]
    assert result.timings["llm_ms"] > 0
    return result, facts


@pytest.mark.live
def test_live_teacher_turn_asked_in_english_states_verified_forms_of_hunn(request, settings, session):
    result, facts = live_turn(request, settings, session, ENGLISH_QUESTION, "teacher")

    assert 'English "have" -> hunn, disposéieren' in facts
    assert any(claim["verified"] and claim["lemma"] == "hunn" for claim in result.claims)


@pytest.mark.live
def test_live_auto_turn_asking_for_a_noun_states_the_verified_noun(request, settings, session):
    result, facts = live_turn(request, settings, session, "What is the word for water in Luxembourgish?", "auto")

    assert 'English "water" -> Waasser, Gewässer, nätzen, dränken, tréinen' in facts
    assert any(claim["verified"] and claim["lemma"] == "Waasser" for claim in result.claims)


@pytest.mark.live
def test_live_auto_turn_asking_how_to_describe_a_day_gets_sentences_not_a_conjugation(request, settings, session):
    result, _ = live_turn(request, settings, session, DAY_QUESTION, "auto")

    assert len([segment for segment in result.reply.segments if segment.lang == "lb"]) >= 2
    assert not any(claim["lemma"].casefold() == "beschreiwen" for claim in result.claims)
