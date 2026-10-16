"""Stage-1 LOD checks, the acceptability engines and the shared label of a correction."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app import practice
from app.config import Settings
from app.corrections import ACCEPTABLE, OBJECTS, Label, label_correction
from app.engines.base import AcceptabilityEngine
from app.engines.factory import ENGINES, KINDS, UnavailableEngine, build_engine
from app.engines.fake import FakeAcceptability, NoAcceptability
from app.lod import checks
from app.lod.store import LodStore, VerbTable
from tests.realdata import REAL_DATA


@dataclass(frozen=True)
class Head:
    pos: str


class Lexicon:
    """Stands in for the LOD store: each form with its lemmas; `names` are proper nouns; `verbs` maps a
    lemma to the LOD tables_json block of its conjugation (indicative present p1..p6 at least)."""

    def __init__(
        self, forms: dict[str, list[str]], names: frozenset[str] = frozenset(), verbs: dict[str, dict] | None = None
    ) -> None:
        self.forms = {form.casefold(): lemmas for form, lemmas in forms.items()}
        self.names = {name.casefold() for name in names}
        self.verbs = {lemma.casefold(): tables for lemma, tables in (verbs or {}).items()}

    def lookup_form(self, form: str) -> list[str]:
        return list(self.forms.get(form.casefold(), []))

    def conjugation(self, lemma: str) -> VerbTable | None:
        tables = self.verbs.get(lemma.casefold())
        if tables is None:
            return None
        present = tables["indicative"]["present"]
        cells = [present.get(f"p{person}", "") for person in range(1, 7)]
        return VerbTable(lemma, "", "hunn", "", cells, "", "", {"conjugations": [tables]})

    def known(self, form: str) -> bool:
        return bool(self.lookup_form(form))

    def headword(self, word: str) -> Head | None:
        if word.casefold() not in self.forms:
            return None
        return Head("NP" if word.casefold() in self.names else "SUBST")


# As the real LOD store has them, apart from "wunne", which only the
# "+ n" fallback finds here.
LOD = Lexicon(
    {
        "ech": ["ech"],
        "zu": ["zu"],
        "wunnen": ["wunnen"],
        "wunnt": ["wunnen"],
        "Esch": ["Esch"],
        "Diddenuewen": ["Diddenuewen"],
        "d'": ["d'"],
        "Fra": ["Fra"],
        "mengen": ["mengen"],
        "Frënn": ["Frënd"],
        "Frënd": ["Frënd"],
        "de": ["de", "den"],
        "den": ["den"],
        "Mann": ["Mann", "mann"],
        "Problem": ["Problem"],
        "Probleem": ["Probleem"],
        "hunn": ["Hunn", "hunn"],
        "sinn": ["sinn"],
        "ass": ["Ass", "sinn"],
        "gaangen": ["goen"],
        "gär": ["gär"],
        "liesen": ["Lies", "liesen"],
        "Bicher": ["Buch"],
        "e": ["E", "en"],
        "Becher": ["Becher", "becheren"],
        "Kaffi": ["Kaffi"],
        "drénken": ["drénken"],
        "ëëë": ["ëëë"],
        "hm": ["hm"],
        "Aarbecht": ["Aarbecht"],
        "schaffen": ["schaffen"],
        "Problemer": ["Problem"],
        "brauch": ["Brauch", "brauchen"],
        "bräicht": ["brauchen"],
        "Stull": ["Stull"],
        "si": ["si"],
        "haut": ["haut"],
        "dat": ["dat"],
        "gutt": ["gutt"],
        "gudde": ["gutt"],
    },
    names=frozenset({"Esch", "Diddenuewen"}),
    verbs={
        "wunnen": {
            "indicative": {"present": {"p1": "wunnen", "p2": "wunns", "p3": "wunnt", "p4": "wunnen", "p5": "wunnt", "p6": "wunnen"}},
            "conditional": {"presentPerfect": {"p1": "géif / géing wunnen", "p3": "géif / géing wunnen"}},
        },
        "brauchen": {
            "indicative": {"present": {"p1": "brauch", "p2": "brauchs", "p3": "brauch", "p4": "brauchen", "p5": "braucht", "p6": "brauchen"}},
            "conditional": {"presentSimple": {"p1": "bräicht / bréicht", "p3": "bräicht / bréicht", "p4": "bräichten / bréichten"}},
        },
    },
)

WUNNT = ("Ech wunnt zu Esch.", "Ech wunnt", "Ech wunnen zu Esch.")
WUNNT_CHECKS = [
    "wunnt and wunnen are forms of wunnen (LOD)",
    "ech takes wunnen, not wunnt (LOD)",
    "every word the tutor added is in LOD",
    "no encoder installed",
]
BRAICHT = ("Ech bräicht e Stull.", "Ech bräicht e Stull", "Ech brauch e Stull.")


def label(lod, text, quote, better, source="typed", encoder=None) -> Label:
    return label_correction(lod, encoder or NoAcceptability(), text, quote, better, source)


# --- words, clitics, lookups ---


def test_practice_counts_words_with_the_word_pattern_of_the_checks():
    assert practice.ANSWER_WORD is checks.WORD
    assert checks.WORD.findall("D'Aarbecht ass schwéier, ech schaffen 8 Stonnen.") == [
        "D'Aarbecht", "ass", "schwéier", "ech", "schaffen", "8", "Stonnen",
    ]


@pytest.mark.parametrize(
    ("word", "parts"),
    [
        ("d'Fra", ["d'", "Fra"]),
        ("D’Aarbecht", ["D'", "Aarbecht"]),
        ("geet's", ["geet", "'s"]),
        ("wéi't", ["wéi", "'t"]),
        ("Kaffi", ["Kaffi"]),
        ("d'", ["d'"]),
    ],
)
def test_split_clitic_splits_off_d_and_the_trailing_t_s_n(word, parts):
    assert checks.split_clitic(word) == parts


def test_words_split_the_clitics_of_every_word():
    assert checks.words("D'Fra geet's.") == ["D'", "Fra", "geet", "'s"]


def test_known_through_the_word_the_clitic_and_the_n_fallback():
    assert checks.known(LOD, "wunnt")
    assert checks.known(LOD, "wunne")  # + n: wunnen
    assert checks.known(LOD, "d'Fra")  # Fra
    assert checks.known(LOD, "'s")
    assert not checks.known(LOD, "Thionville")
    assert checks.lemmas(LOD, "wunne") == {"wunnen"}
    assert checks.lemmas(LOD, "d'Fra") == {"fra"}
    assert checks.lemmas(LOD, "liesen") == {"lies", "liesen"}


def test_without_a_store_nothing_is_known():
    assert not checks.known(None, "wunnen")
    assert not checks.known(None, "d'")
    assert checks.lemmas(None, "wunnen") == set()
    assert checks.same_lemma(None, "Ech wunnt", "Ech wunnen") == ""
    assert not checks.optional_variant(None, "Problem", "Probleem")


def test_all_known_needs_every_word():
    assert checks.all_known(LOD, ["ech", "wunnen"])
    assert checks.all_known(LOD, [])
    assert not checks.all_known(LOD, ["ech", "Kaffisbecher"])


def test_a_common_word_is_a_known_word_that_is_no_proper_noun():
    assert checks.common_word(LOD, "Frënn")
    assert not checks.common_word(LOD, "Diddenuewen")
    assert not checks.common_word(LOD, "Monique")
    assert not checks.common_word(None, "Frënn")


@pytest.mark.parametrize(
    ("word", "first", "reason"),
    [
        ("ëëë", False, "filler"),
        ("Ëëë", True, "filler"),
        ("ähm", False, "filler"),
        ("hm", False, "filler"),
        ("aaa", False, "filler"),
        ("ee", False, None),
        ("2024", False, "number"),
        ("3e", False, "number"),
        ("Thionville", False, "name"),
        ("Thionville", True, None),
        ("Esch", False, None),
        ("Wunnt", True, None),
        ("wunnt", False, None),
        ("kaffisbecher", False, None),
    ],
)
def test_skipped_says_why_stage_1_judges_no_word(word, first, reason):
    assert checks.skipped(word, LOD, first) == reason


def test_changed_words_are_case_folded_in_order_with_clitics_split():
    assert checks.changed_words("Ech wunnt", "Ech wunnen zu Esch.") == (["wunnt"], ["wunnen", "zu", "esch"])
    assert checks.changed_words("Ech gär liesen Bicher", "Ech liesen gär Bicher") == ([], [])
    assert checks.changed_words("d'Fra", "der Fra") == (["d'"], ["der"])


def test_same_lemma_finds_the_swap_of_two_forms_of_one_lemma():
    assert checks.same_lemma(LOD, "Ech wunnt", "Ech wunnen zu Esch.") == "wunnen"
    assert checks.same_lemma(LOD, "mengen Freunden", "mengen Frënn") == ""
    assert checks.same_lemma(LOD, "Ech hunn gaangen", "Ech sinn gaangen") == ""


def test_an_n_rule_pair_is_no_same_lemma_evidence():
    assert checks.same_lemma(LOD, "de Mann", "den Mann") == ""
    assert checks.n_only(LOD, "de", "den")
    assert checks.n_only(LOD, "wann", "wa")
    assert checks.n_only(LOD, "wunne", "Wunnen")
    assert not checks.n_only(LOD, "wunnt", "wunnen")
    assert not checks.n_only(LOD, "de", "de")


def test_optional_variants_are_n_rule_pairs_and_lemmas_that_differ_in_doubled_vowels():
    assert checks.optional_variant(LOD, "de", "den")
    assert checks.optional_variant(LOD, "Problem", "Probleem")
    assert not checks.optional_variant(LOD, "hunn", "sinn")
    assert not checks.optional_variant(LOD, "Problemer", "Probleemer")  # forms, not lemmas
    assert not checks.optional_variant(LOD, "Problem", "Problem")


# --- the label ---


def test_a_same_lemma_fix_whose_new_words_are_in_lod_is_verified():
    assert label(LOD, *WUNNT) == Label("verified", "", WUNNT_CHECKS)


def test_person_forms_hold_the_finite_verb_of_every_tense_per_person():
    cells = checks.person_forms(LOD, "brauchen")

    assert cells[1] == {"brauch", "bräicht", "bréicht"}
    assert cells[4] == {"brauchen", "bräichten", "bréichten"}
    assert cells[2] == {"brauchs"}
    assert checks.person_forms(LOD, "wunnen")[1] == {"wunnen", "géif", "géing"}  # "géif / géing wunnen": the finite word
    assert checks.person_forms(LOD, "Mann") == {}
    assert checks.person_forms(None, "wunnen") == {}


@pytest.mark.parametrize(
    ("text", "quote", "form", "pronoun"),
    [
        ("Ech wunnt zu Esch.", "Ech wunnt", "wunnt", "ech"),
        ("Ech wunnt zu Esch.", "wunnt", "wunnt", "ech"),  # the word before the quoted form
        ("Haut wunnt ech zu Esch.", "wunnt", "wunnt", "ech"),  # inverted: the word after it
        ("Si wunnt zu Esch.", "wunnt zu Esch", "wunnt", "si"),
        ("Hie wunnt zu Esch.", "wunnt", "wunnt", "hie"),
        ("De Mann wunnt zu Esch.", "wunnt", "wunnt", ""),
        ("Ech mengen, de Mann wunnt zu Esch.", "wunnt", "wunnt", ""),  # a pronoun elsewhere is not the subject
    ],
)
def test_the_subject_pronoun_comes_from_the_quote_or_beside_the_quoted_form(text, quote, form, pronoun):
    assert checks.subject_pronoun(text, quote, form) == pronoun


def test_a_same_lemma_swap_whose_quoted_form_also_fits_the_subject_is_likely():
    result = label(LOD, *BRAICHT)

    assert result == Label(
        "likely",
        "",
        [
            "bräicht and brauch are forms of brauchen (LOD)",
            "bräicht also fits ech in LOD: at most Likely",
            "every word the tutor added is in LOD",
            "no encoder installed",
        ],
    )


def test_a_one_word_quote_is_verified_through_the_pronoun_beside_it():
    assert label(LOD, "Ech wunnt zu Esch.", "wunnt", "wunnen").tier == "verified"
    assert label(LOD, "Haut wunnt ech zu Esch.", "wunnt", "Haut wunnen ech zu Esch.").tier == "verified"


def test_an_ambiguous_pronoun_gives_no_agreement_evidence():
    result = label(LOD, "Si wunnt zu Esch.", "Si wunnt", "Si wunnen zu Esch.")

    assert result.tier == "likely"
    assert "wunnt also fits si in LOD: at most Likely" in result.checks


def test_a_same_lemma_swap_without_a_subject_pronoun_is_likely():
    adjective = label(LOD, "Dat ass e gutt Mann.", "e gutt Mann", "e gudde Mann")
    noun_subject = label(LOD, "De Mann wunnen zu Esch.", "De Mann wunnen", "De Mann wunnt zu Esch.")

    assert adjective.tier == "likely"
    assert "no subject pronoun beside gutt: at most Likely" in adjective.checks
    assert noun_subject.tier == "likely"
    assert "no subject pronoun beside wunnen: at most Likely" in noun_subject.checks


def test_a_same_lemma_swap_on_a_lemma_without_a_verb_table_is_likely():
    result = label(LOD, "Ech hunn e gutt Mann.", "Ech hunn e gutt Mann", "Ech hunn e gudde Mann.")

    assert result.tier == "likely"
    assert "gutt has no verb table in LOD: at most Likely" in result.checks


@pytest.mark.parametrize("source", ["typed", "heard-edited", ""])
def test_typed_and_edited_answers_can_be_verified(source):
    assert label(LOD, *WUNNT, source=source).tier == "verified"


def test_a_heard_answer_that_was_not_edited_tops_out_at_likely():
    result = label(LOD, *WUNNT, source="heard-unedited")

    assert (result.tier, result.downgrade) == ("likely", "")
    assert "heard and not edited: at most Likely" in result.checks


def test_a_fix_without_same_lemma_evidence_whose_new_words_are_known_is_likely():
    result = label(LOD, "Ech hunn mengen Freunden geschriwwen.", "mengen Freunden", "mengen Frënn")

    assert result == Label(
        "likely",
        "",
        ["no same-lemma swap in LOD: at most Likely", "every word the tutor added is in LOD", "no encoder installed"],
    )


def test_a_better_version_with_a_word_lod_lacks_is_unverified():
    result = label(LOD, "Ech drénken e Becher Kaffi.", "e Becher Kaffi", "e Kaffisbecher")

    assert result.tier == "unverified"
    assert "not in LOD: kaffisbecher" in result.checks


def test_an_unknown_word_the_learner_said_and_the_tutor_kept_is_not_checked():
    result = label(LOD, "Ech wunnt zu Thionville.", "Ech wunnt zu Thionville", "Ech wunnen zu Thionville.")

    assert result.tier == "verified"
    assert "every word the tutor added is in LOD" in result.checks


def test_a_fix_on_known_words_without_a_swap_stays_a_fix_labelled_likely():
    order = label(LOD, "Ech gär liesen Bicher.", "Ech gär liesen Bicher", "Ech liesen gär Bicher")
    auxiliary = label(LOD, "Ech hunn gaangen.", "Ech hunn gaangen", "Ech sinn gaangen.")

    assert (order.tier, order.downgrade) == ("likely", "")
    assert "the tutor added no word" in order.checks
    assert (auxiliary.tier, auxiliary.downgrade) == ("likely", "")


@pytest.mark.parametrize(
    ("text", "quote", "better"),
    [
        ("Ech wunnen zu Thionville.", "Thionville", "Diddenuewen"),
        ("Ech wunnen mat Monika.", "Monika", "Monique"),
        ("Ëëë ech wunnen zu Esch.", "Ëëë ech wunnen", "Ech wunnen"),
        ("Ech wunnen hei zënter 2024.", "zënter 2024", "zënter 2023"),
    ],
)
def test_a_fix_on_names_fillers_or_numbers_only_is_downgraded(text, quote, better):
    result = label(LOD, text, quote, better)

    assert (result.tier, result.downgrade) == ("", "skipped words")
    assert result.checks


def test_the_skip_says_what_the_word_is():
    assert label(LOD, "Ech wunnen zu Thionville.", "Thionville", "Diddenuewen").checks == [
        "Thionville is a name: not a learner error"
    ]


def test_a_name_at_the_start_of_a_sentence_is_no_name():
    result = label(LOD, "Esch ass schéin. Thionville ass grouss.", "Thionville ass", "Diddenuewen ass")

    assert result.downgrade == ""


def test_a_capitalised_foreign_word_the_tutor_replaces_with_a_common_word_stays_a_fix():
    result = label(LOD, "Meng Arbeit ass schwéier.", "Meng Arbeit", "Meng Aarbecht")

    assert (result.tier, result.downgrade) == ("likely", "")


@pytest.mark.parametrize(
    ("text", "quote", "better", "line"),
    [
        ("Ech gesinn de Mann.", "de Mann", "den Mann", "de and den differ only by the n-rule: LOD accepts both"),
        ("Dat ass e Problem.", "e Problem", "e Probleem", "problem and probleem are both spellings LOD lists"),
    ],
)
def test_a_fix_between_optional_variants_is_downgraded(text, quote, better, line):
    assert label(LOD, text, quote, better) == Label("", "optional variant", [line])


def test_an_n_rule_change_beside_a_same_lemma_swap_tops_out_at_likely():
    result = label(LOD, "Ech wunnt bei de Mann.", "Ech wunnt bei de Mann", "Ech wunnen bei den Mann.")

    assert result.tier == "likely"
    assert "ech takes wunnen, not wunnt (LOD)" in result.checks
    assert "de and den differ only by the n-rule: at most Likely" in result.checks


def test_an_encoder_that_accepts_the_answer_downgrades_the_fix():
    encoder = FakeAcceptability(scores={WUNNT[0]: ACCEPTABLE + 0.05})

    result = label(LOD, *WUNNT, encoder=encoder)

    assert result == Label("", "encoder accepts", ["the encoder scores the answer 0.95: acceptable"])
    assert encoder.calls == [WUNNT[0]]


def test_an_encoder_that_objects_to_the_better_version_keeps_a_swap_from_verified():
    encoder = FakeAcceptability(scores={WUNNT[2]: OBJECTS - 0.3})

    result = label(LOD, *WUNNT, encoder=encoder)

    assert result.tier == "likely"
    assert "the encoder objects to the better version: not Verified" in result.checks
    assert "the encoder scores the better version 0.20" in result.checks
    assert encoder.calls == [WUNNT[0], WUNNT[2]]


def test_an_encoder_that_prefers_the_better_version_makes_an_unknown_word_likely():
    text, quote, better = "Ech drénken e Becher Kaffi.", "e Becher Kaffi", "e Kaffisbecher"
    encoder = FakeAcceptability(scores={text: 0.3, better: 0.8})

    assert label(LOD, text, quote, better, encoder=encoder).tier == "likely"


def test_an_encoder_that_fails_counts_as_none():
    result = label(LOD, *WUNNT, encoder=UnavailableEngine("onnx", "no model"))

    assert result == Label("verified", "", WUNNT_CHECKS)


def test_an_english_answer_is_likely_when_the_new_words_are_known_and_never_verified():
    known = label(LOD, "I live in Esch.", "", "Ech wunnen zu Esch.")
    unknown = label(LOD, "I drink a coffee mug.", "", "Ech drénken e Kaffisbecher.")

    assert (known.tier, known.downgrade) == ("likely", "")
    assert "an English answer has no form of the learner's to compare: at most Likely" in known.checks
    assert (unknown.tier, unknown.downgrade) == ("unverified", "")


@pytest.mark.parametrize(
    ("text", "quote", "better"),
    [WUNNT, ("Ech wunnen zu Thionville.", "Thionville", "Diddenuewen"), ("Ech gesinn de Mann.", "de Mann", "den Mann")],
)
def test_without_a_store_every_correction_is_unverified_and_none_is_downgraded(text, quote, better):
    result = label(None, text, quote, better)

    assert result == Label("unverified", "", ["no LOD store: nothing is verified", "no encoder installed"])


# --- the acceptability engines ---


def test_acceptability_is_the_fourth_engine_kind():
    assert KINDS == ("stt", "tts", "llm", "acceptability")
    assert set(ENGINES["acceptability"]) == {"none", "fake"}
    assert Settings().acceptability_engine == "none"
    assert Settings.from_env({"LOCI_ACCEPTABILITY_ENGINE": "fake"}).acceptability_engine == "fake"


def test_the_acceptability_engines_follow_the_protocol():
    assert isinstance(NoAcceptability(), AcceptabilityEngine)
    assert isinstance(FakeAcceptability(), AcceptabilityEngine)
    assert NoAcceptability().ready() and FakeAcceptability().ready()


def test_no_acceptability_has_no_opinion():
    engine = NoAcceptability()

    assert (engine.name, engine.score("Ech wunnen zu Esch.")) == ("none", None)
    assert engine.detail == "no encoder installed"


def test_the_fake_encoder_returns_the_scripted_score_and_logs_the_call():
    engine = FakeAcceptability(scores={"Ech wunnen zu Esch.": 0.97})

    assert engine.score("  ech WUNNEN   zu Esch. ") == 0.97
    assert engine.score("Ech wunnt zu Esch.") is None
    assert engine.calls == ["  ech WUNNEN   zu Esch. ", "Ech wunnt zu Esch."]


def test_build_engine_gives_no_encoder_by_default_and_the_fake_by_name():
    assert build_engine("acceptability", Settings()).name == "none"
    assert isinstance(build_engine("acceptability", Settings(acceptability_engine="fake")), FakeAcceptability)


def test_an_unknown_encoder_is_unavailable_and_its_score_raises():
    engine = build_engine("acceptability", Settings(acceptability_engine="onnx_encoder"))

    assert isinstance(engine, UnavailableEngine)
    assert not engine.ready()
    with pytest.raises(RuntimeError, match="unavailable"):
        engine.score("Moien")


def test_health_lists_the_fake_encoder_when_it_is_configured(monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import create_app

    monkeypatch.setenv("LOCI_ACCEPTABILITY_ENGINE", "fake")
    with TestClient(create_app(Settings.from_env())) as client:
        body = client.get("/api/health").json()

    assert body["engines"]["acceptability"] == {"name": "fake", "ready": True, "detail": ""}
    assert isinstance(client.app.state.acceptability, FakeAcceptability)


# --- on the real store ---


@pytest.fixture(scope="module")
def real():
    path = REAL_DATA / "lod" / "lod.sqlite"
    if not path.is_file():
        pytest.skip(f"{path} is missing; run scripts/download_lod.py")
    store = LodStore(path)
    yield store
    store.close()


@pytest.mark.models
def test_real_store_lookups(real):
    assert checks.known(real, "wunne")
    assert checks.known(real, "d'Fra")
    assert checks.known(real, "gaange")  # + n: gaangen
    assert not checks.known(real, "Thionville")
    assert checks.skipped("ëëë", real) == "filler"
    assert checks.skipped("Thionville", real) == "name"
    assert checks.skipped("Esch", real) is None


@pytest.mark.models
def test_real_store_swaps_and_variants(real):
    assert checks.same_lemma(real, "Ech wunnt", "Ech wunnen zu Esch.") == "wunnen"
    assert checks.same_lemma(real, "mengen Freunden", "mengen Frënn") == ""
    assert checks.same_lemma(real, "de Mann", "den Mann") == ""
    assert checks.n_only(real, "de", "den")
    assert checks.optional_variant(real, "Problem", "Probleem")
    assert not checks.optional_variant(real, "hunn", "sinn")


@pytest.mark.models
@pytest.mark.parametrize(
    ("text", "quote", "better", "tier", "downgrade"),
    [
        (*WUNNT, "verified", ""),
        ("Ech wunnt zu Esch.", "wunnt", "Ech wunnen zu Esch.", "verified", ""),
        (*BRAICHT, "likely", ""),  # conditional and present are both correct: live LB-JUDGE false correction
        ("Si wunnt zu Esch.", "Si wunnt", "Si wunnen zu Esch.", "likely", ""),
        ("Hie verpassen keng Molzecht.", "verpassen", "Hie verpasst keng Molzecht.", "verified", ""),
        # wëllt and wëllen are forms of the adjective wëll too: the verb table of wëllen still decides
        ("Mir wëllt eis adaptéieren.", "wëllt", "Mir wëllen eis adaptéieren.", "verified", ""),
        ("Ech hunn e gutt Mann gesinn.", "e gutt Mann", "e gudde Mann", "likely", ""),
        ("Ech hunn mengen Freunden geschriwwen.", "mengen Freunden", "mengen Frënn", "likely", ""),
        ("Ech drénken e Becher Kaffi.", "e Becher Kaffi", "e Kaffisbecher", "unverified", ""),
        ("Ech wunnt zu Thionville.", "Ech wunnt zu Thionville", "Ech wunnen zu Thionville.", "verified", ""),
        ("Ech hunn gaangen.", "Ech hunn gaangen", "Ech sinn gaangen.", "likely", ""),
        ("Meng Arbeit ass schwéier.", "Meng Arbeit", "Meng Aarbecht", "likely", ""),
        ("Ech wunnen zu Thionville.", "Thionville", "Diddenuewen", "", "skipped words"),
        ("Ech gesinn de Mann.", "de Mann", "den Mann", "", "optional variant"),
        ("Dat ass e Problem.", "e Problem", "e Probleem", "", "optional variant"),
    ],
)
def test_real_store_labels(real, text, quote, better, tier, downgrade):
    result = label(real, text, quote, better)

    assert (result.tier, result.downgrade) == (tier, downgrade), result.checks
