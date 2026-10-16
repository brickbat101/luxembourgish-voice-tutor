"""LB-JUDGE: the set builder on the fixture dump and a tiny LOD store, the four perturbations,
the runner and its report, and scripts/lb_judge.py build and run."""

import json
import random
import sqlite3
import time
from pathlib import Path

import pytest

from app.config import Settings
from app.content import Question, load_practice
from app.engines.base import Judgement, LLMError
from app.engines.fake import NoAcceptability
from app.eval import lb_judge
from app.eval.report import (
    OracleJudge,
    Outcome,
    binomial_cdf,
    format_report,
    headline,
    judge_rows,
    oracle_judgement,
    summarise,
    upper_95,
)
from app.eval.lb_judge import (
    CLASSES,
    Example,
    LodFacts,
    Token,
    adjective_ending,
    auxiliary,
    correct_rows,
    detokenise,
    error_rows,
    examples,
    noun_gender,
    read_set,
    verb_person,
    write_set,
)
from app.lod import checks
from app.lod.build import SCHEMA, variants
from app.lod.store import LodStore
from scripts import lb_judge as script
from tests.realdata import REAL_DATA
from tests.test_practice import KnownForms

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "lod_examples.xml"
SEEDS = range(20)

VERBS = [
    ("wunnen", "WUNNEN1", "hunn", "gewunnt", "wunnen", "wunns", "wunnt", "wunnen", "wunnt", "wunnen"),
    ("goen", "GOEN1", "sinn", "gaangen / gaang", "ginn", "gees", "geet", "ginn", "gitt", "ginn"),
    ("kucken", "KUCKEN1", "hunn", "gekuckt", "kucken", "kucks", "kuckt", "kucken", "kuckt", "kucken"),
    ("sinn", "SINN1", "sinn", "gewiescht", "sinn", "bass", "ass", "sinn", "sidd", "sinn"),
    ("hunn", "HUNN1", "hunn", "gehat", "hunn (mech)", "hues (dech)", "huet (sech)", "hunn (eis)", "hutt", "hunn"),
    ("fueren", "FUEREN1", "hunn / sinn", "gefuer", "fueren", "fiers", "fiert", "fueren", "fuert", "fueren"),
    ("maachen", "MAACHEN1", "hunn", "gemaach", "maachen", "méchs", "mécht", "maachen", "maacht", "maachen"),
]
CASES = ("nominative", "accusative", "dative", "genitive")


def declension(masculine, feminine, neutral, plural):
    cells = lambda forms: dict(zip(CASES, forms))  # noqa: E731
    return {
        "declension": {
            "positive": {
                "singular": {"masculine": cells(masculine), "feminine": cells(feminine), "neutral": cells(neutral)},
                "plural": {"masculineFeminineNeutral": cells(plural)},
            }
        }
    }


ADJECTIVES = [
    ("gutt", "GUTT1", declension(["gudden"] * 4, ["gutt", "gutt", "gudder", "gudder"],
                                 ["gutt", "gutt", "gudden", "gudden"], ["gutt", "gutt", "gudden", "gudder"])),
    # LOD declines the pronoun een like an adjective; its feminine cell is "eng".
    ("een", "EEN1", declension(["een"] * 4, ["eng", "eng", "enger", "enger"],
                               ["eent", "eent", "engem", "enges"], ["eng"] * 4)),
]
HEADWORDS = [
    ("Mann", "M"), ("Fra", "F"), ("Kand", "N"), ("Dier", "F"), ("Esch", ""),
]
OTHER_POS = [("gutt", "ADJ"), ("eng", "ART"), ("een", "PRON"), ("Esch", "NP")]
# Every other word of the fixture and of the hand-made examples, so that LOD knows them all.
WORDS = (
    "ech du hien hie et mir dir si zu an all do haut heem op gëschter steet dat wéi kuck d' e en de den eng"
    " Kanner Dag Gaart wunns wunnt"
).split()


def lod_store(folder: Path, verbs=VERBS, adj_forms=()) -> Path:
    """A LOD store in `folder` with the verbs, adjectives and headwords above."""
    path = folder / "lod.sqlite"
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    db.executemany(
        "INSERT INTO verbs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', '', '{\"conjugations\": []}')", verbs
    )
    db.executemany(
        "INSERT INTO adjectives VALUES (?, ?, ?)",
        [(lemma, entry_id, json.dumps(forms)) for lemma, entry_id, forms in ADJECTIVES],
    )
    nouns = [(lemma, lemma.upper() + "1", "SUBST" if gender else "NP", gender) for lemma, gender in HEADWORDS]
    others = [(lemma, lemma.upper() + "1", pos, "") for lemma, pos in OTHER_POS]
    db.executemany("INSERT INTO headwords VALUES (?, ?, ?, ?, '', '', '', '')", nouns + others)
    forms = [(word.casefold(), word, "lemma") for word in WORDS]
    forms += [(lemma.casefold(), lemma, "lemma") for lemma, *_ in HEADWORDS + OTHER_POS]
    for lemma, _, _, participle, *cells in verbs:
        forms += [(form, lemma, "verb_form") for cell in [lemma, participle, *cells] for form in variants(cell)]
    for lemma, _, table in ADJECTIVES:
        positive = table["declension"]["positive"]
        forms += [(cell, lemma, "adj_form") for genders in positive.values() for cases in genders.values()
                  for cell in cases.values()]
    forms += [(form, form, "adj_form") for form in adj_forms]
    db.executemany("INSERT OR IGNORE INTO wordforms VALUES (?, ?, ?)", forms)
    db.commit()
    db.close()
    return path


@pytest.fixture
def store(tmp_path):
    return lod_store(tmp_path)


@pytest.fixture
def facts(store):
    return LodFacts(store)


def example(text: str, example_id: str = "x1") -> Example:
    return Example(example_id, tuple(Token(word) for word in text.split()), False)


def outcomes(perturbation, text, facts):
    """The rows a perturbation makes of `text` over SEEDS (None when it makes none)."""
    return [perturbation(example(text), facts, random.Random(seed)) for seed in SEEDS]


def texts(perturbation, text, facts):
    return {row["text"] for row in outcomes(perturbation, text, facts)}


NOTHING = [None] * len(SEEDS)


def assert_error_row(row, error_type, base_id="x1"):
    assert row["label"] == "error"
    assert row["source"] == "perturbation"
    assert row["error_type"] == error_type
    assert row["base_id"] == base_id
    assert row["review"] == "unreviewed"
    assert row["wrong"] and row["gold"] and row["wrong"] != row["gold"]
    assert row["wrong"] in row["text"]


# --- examples and detokenise ---


def test_examples_reads_the_twelve_fixture_examples():
    found = {item.id: item for item in examples(FIXTURE)}

    assert sorted(found) == [f"ex{k:02d}" for k in range(1, 13)]
    assert [key for key, item in found.items() if item.labelled] == ["ex01", "ex02"]
    # The gloss words are not tokens of the example.
    assert [token.text for token in found["ex05"].tokens] == ["ech", "wunnen", "zu", "Esch"]
    assert [token.is_headword for token in found["ex05"].tokens] == [False, True, False, False]
    assert len(found["ex04"].words) == 12
    assert found["ex10"].text == "Ass hien heem gaangen?"
    assert found["ex09"].text == "Kuck, de Mann ass do!"


def test_detokenise():
    assert detokenise(["ech", "wunnen", "zu", "Esch"]) == "Ech wunnen zu Esch."
    assert detokenise(["mir", ",", "d'", "Kanner", "!"]) == "Mir, d'Kanner!"
    assert detokenise(["d'", "Kand", "ass", "do!"]) == "D'Kand ass do!"
    assert detokenise(["wéi", "gutt,", "du", "wunns", "?"]) == "Wéi gutt, du wunns?"
    assert detokenise(["hie", "seet", "(", "lues", ")"]) == "Hie seet (lues)"


# --- the correct arm ---


def test_correct_rows_takes_statements_first_and_never_labelled_digit_or_long_examples():
    pool = list(examples(FIXTURE))
    statements = {"ex05", "ex06", "ex07", "ex08", "ex12"}

    for seed in SEEDS:
        rows = correct_rows(pool, 4, random.Random(seed))
        assert len(rows) == 4
        assert {row["base_id"] for row in rows} <= statements

    rows = correct_rows(pool, 7, random.Random(1))
    assert {row["base_id"] for row in rows} >= statements
    assert not {row["base_id"] for row in rows} & {"ex01", "ex02", "ex03", "ex04"}
    assert len(correct_rows(pool, 50, random.Random(1))) == 8


def test_correct_rows_are_reproducible_and_have_the_set_fields():
    pool = list(examples(FIXTURE))

    rows = correct_rows(pool, 4, random.Random(7))

    assert rows == correct_rows(pool, 4, random.Random(7))
    assert [row["id"] for row in rows] == ["c-1", "c-2", "c-3", "c-4"]
    for row in rows:
        assert list(row) == ["id", "text", "label", "source", "base_id", "error_type", "review"]
        assert (row["label"], row["source"], row["error_type"], row["review"]) == (
            "correct", "lod_example", "", "unreviewed",
        )
        assert row["text"][0].isupper() and row["text"].endswith((".", "!", "?"))


# --- LodFacts ---


def test_lod_facts_reads_the_store(facts):
    assert facts.present["wunnen"] == [(0, 1), (0, 4), (0, 6)]
    assert {"gaangen", "gaang"} <= set(facts.participle_aux)
    assert facts.participle_aux["gefuer"] == [frozenset({"hunn", "sinn"})]
    assert "hunn" in facts.person_forms[1] and "hues" in facts.person_forms[2]
    assert facts.noun_gender("Mann") == "M" and facts.noun_gender("Kand") == "N"
    assert facts.noun_gender("mann") == "" and facts.noun_gender("Esch") == ""
    assert "eng" in facts.determiners and "gutt" in facts.adjective_lemmas
    assert facts.knows_all(example("ech wunnen zu Esch"))
    assert not facts.knows_all(example("ech wunnen zu Bartreng"))


def test_lod_facts_names_download_lod_when_the_store_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_lod.py"):
        LodFacts(tmp_path / "lod.sqlite")


# --- the four perturbations ---


def test_verb_person_swaps_the_form_for_one_no_verb_lists_for_the_person(facts):
    rows = outcomes(verb_person, "ech wunnen zu Esch", facts)

    assert {row["wrong"] for row in rows} == {"wunns", "wunnt"}
    for row in rows:
        assert_error_row(row, "verb_person")
        assert row["gold"] == "wunnen"
        assert row["text"] == f"Ech {row['wrong']} zu Esch."


def test_verb_person_skips_forms_another_verb_lists_and_n_rule_pairs(tmp_path):
    # wunne (an n-rule pair of wunnen) and wunns (p1 of another verb) are no error after ech.
    verbs = [
        ("wunnen", "WUNNEN1", "hunn", "gewunnt", "wunnen", "wunns", "wunnt", "wunnen", "wunne", "wunnen"),
        ("fuddelen", "FUDDELEN1", "hunn", "gefuddelt", "wunns", "fudds", "fuddelt", "fuddelen", "fuddelt", "fuddelen"),
    ]
    facts = LodFacts(lod_store(tmp_path, verbs=verbs))

    rows = outcomes(verb_person, "ech wunnen zu Esch", facts)

    assert {row["wrong"] for row in rows} == {"wunnt"}


def test_verb_person_lets_expletive_et_take_a_plural_form(facts):
    rows = outcomes(verb_person, "et geet haut", facts)

    assert {row["wrong"] for row in rows} == {"gees", "gitt"}


def test_verb_person_needs_a_pronoun_directly_before_a_present_form(facts):
    assert outcomes(verb_person, "de Mann wunnt zu Esch", facts) == NOTHING
    assert outcomes(verb_person, "ech , wunnen zu Esch", facts) == NOTHING
    assert outcomes(verb_person, "ech kuckt", facts) == NOTHING


def test_auxiliary_swaps_the_auxiliary_of_a_participle_that_takes_it_alone(facts):
    rows = outcomes(auxiliary, "ech sinn haut heem gaangen", facts)
    for row in rows:
        assert_error_row(row, "auxiliary")
        assert (row["gold"], row["wrong"], row["text"]) == ("sinn", "hunn", "Ech hunn haut heem gaangen.")

    assert texts(auxiliary, "hien huet de Mann gekuckt", facts) == {"Hien ass de Mann gekuckt."}
    assert texts(auxiliary, "mir si gëschter gaang", facts) == {"Mir hu gëschter gaang."}
    # The pronoun si is no auxiliary.
    assert texts(auxiliary, "si hunn de Mann gekuckt", facts) == {"Si sinn de Mann gekuckt."}


def test_auxiliary_skips_a_swap_that_could_be_legal(tmp_path, facts):
    # The base would already be wrong: goen takes sinn.
    assert outcomes(auxiliary, "ech hunn haut heem gaangen", facts) == NOTHING
    # fueren takes hunn and sinn.
    assert outcomes(auxiliary, "ech sinn op Esch gefuer", facts) == NOTHING
    # The participle of the first clause is not looked for in the second.
    assert {row["gold"] for row in outcomes(auxiliary, "ech hunn Honger, ech sinn gaangen", facts)} == {"sinn"}
    # A participle that is also an adjective form ("ass gemaach" is a state).
    folder = tmp_path / "adjective"
    folder.mkdir()
    with_adjective = LodFacts(lod_store(folder, adj_forms=["gemaach"]))
    assert outcomes(auxiliary, "et huet de Mann gemaach", with_adjective) == NOTHING
    assert texts(auxiliary, "et huet de Mann gemaach", facts) == {"Et ass de Mann gemaach."}


def test_adjective_ending_swaps_to_another_gender_cell_of_the_same_case(facts):
    rows = outcomes(adjective_ending, "de gudde Mann wunnt zu Esch", facts)

    assert {row["wrong"] for row in rows} == {"gutt", "gudder"}
    for row in rows:
        assert_error_row(row, "adjective_ending")
        assert row["gold"] == "gudde"

    rows = outcomes(adjective_ending, "eng gutt Fra kuckt", facts)
    # gudden loses its n before F by the n-rule.
    assert {row["wrong"] for row in rows} == {"gudde"}


def test_adjective_ending_never_treats_an_article_as_an_adjective(facts):
    assert outcomes(adjective_ending, "do steet eng Fra", facts) == NOTHING
    # No noun of one gender after the adjective.
    assert outcomes(adjective_ending, "de gudde Esch", facts) == NOTHING
    assert outcomes(adjective_ending, "dat ass gutt", facts) == NOTHING


def test_noun_gender_swaps_the_article_to_the_other_gender(facts):
    rows = outcomes(noun_gender, "do steet de Mann", facts)
    assert {row["text"] for row in rows} == {"Do steet eng Mann."}
    assert_error_row(rows[0], "noun_gender")
    assert (rows[0]["gold"], rows[0]["wrong"]) == ("de", "eng")

    # en or e by the n-rule: e before F, en before D.
    assert texts(noun_gender, "do steet eng Fra", facts) == {"Do steet e Fra."}
    assert texts(noun_gender, "do steet eng Dier", facts) == {"Do steet en Dier."}
    assert texts(noun_gender, "en Mann", facts) == {"Eng Mann."}


def test_noun_gender_skips_neuter_nouns_and_d(facts):
    assert outcomes(noun_gender, "do steet d' Kand", facts) == NOTHING
    assert outcomes(noun_gender, "do steet en Kand", facts) == NOTHING
    assert outcomes(noun_gender, "do steet de Fra", facts) == NOTHING


# --- the error arm ---

ONE_CLASS_EACH = {
    "verb_person": ["ech wunnen zu Esch", "mir kucken haut"],
    "auxiliary": ["d' Kand ass haut gaangen", "d' Fra huet gëschter gekuckt"],
    "adjective_ending": ["d' Kand kuckt e gudde Mann", "do steet eng gutt Fra"],
    "noun_gender": ["do steet de Mann", "do steet eng Fra"],
}


def class_examples(sentences=ONE_CLASS_EACH):
    texts = [text for texts in sentences.values() for text in texts]
    return [example(text, f"b{k}") for k, text in enumerate(texts, 1)]


def test_error_rows_gives_n_over_four_per_class_from_distinct_bases(facts):
    pool = class_examples()

    for seed in SEEDS:
        rows = error_rows(pool, facts, 8, random.Random(seed))
        assert [row["error_type"] for row in rows] == [name for name in CLASSES for _ in range(2)]
        assert len({row["base_id"] for row in rows}) == 8
        assert [row["id"] for row in rows] == [f"e-{k}" for k in range(1, 9)]
        for row in rows:
            assert_error_row(row, row["error_type"], row["base_id"])
            assert list(row) == [
                "id", "text", "label", "source", "base_id", "error_type", "wrong", "gold", "review",
            ]


def test_error_rows_fills_a_short_class_from_the_others(facts):
    sentences = dict(ONE_CLASS_EACH)
    sentences["verb_person"] = ["ech wunnen zu Esch", "mir kucken haut", "du kucks haut", "dir kuckt haut"]
    sentences["adjective_ending"] = ["d' Kand kuckt e gudde Mann"]

    rows = error_rows(class_examples(sentences), facts, 8, random.Random(3))

    counts = {name: sum(row["error_type"] == name for row in rows) for name in CLASSES}
    assert counts == {"verb_person": 3, "auxiliary": 2, "adjective_ending": 1, "noun_gender": 2}
    assert len(error_rows(class_examples(sentences), facts, 40, random.Random(3))) == 9


def test_error_rows_are_reproducible(facts):
    pool = list(examples(FIXTURE))

    assert error_rows(pool, facts, 4, random.Random(7)) == error_rows(pool, facts, 4, random.Random(7))


# --- the set file ---


def test_write_set_and_read_set_round_trip_with_the_header(tmp_path, facts):
    pool = list(examples(FIXTURE))
    rows = correct_rows(pool, 2, random.Random(1)) + error_rows(pool, facts, 4, random.Random(1))
    rows.append(rows[0] | {"id": "c-3", "text": "Wéi gutt!"})
    path = tmp_path / "eval" / "lb_judge.jsonl"

    write_set(rows, path, seed=1, lod_dump="lod_examples.xml")

    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    head = json.loads(lines[0])
    assert head["set"] == "lb_judge" and head["seed"] == 1 and head["lod_dump"] == "lod_examples.xml"
    assert (head["review"], head["reviewed_by"]) == ("unreviewed", "none")
    assert len(lines) == 1 + len(rows)
    assert "Wéi gutt!" in raw
    assert read_set(path) == rows
    assert [item.name for item in path.parent.iterdir()] == ["lb_judge.jsonl"]


# --- scripts/lb_judge.py build ---


def test_build_writes_n_correct_and_n_error_rows_and_prints_the_counts(tmp_path, store, capsys):
    out = tmp_path / "out" / "set.jsonl"

    code = script.main(["build", "--n", "4", "--dump", str(FIXTURE), "--store", str(store), "--out", str(out)])

    assert code == 0
    rows = read_set(out)
    assert [row["label"] for row in rows] == ["correct"] * 4 + ["error"] * 4
    assert {row["error_type"] for row in rows[4:]} == set(CLASSES)
    printed = capsys.readouterr().out
    assert "correct 4" in printed
    assert "error 4: verb_person 1, auxiliary 1, adjective_ending 1, noun_gender 1" in printed
    assert str(out) in printed
    assert script.main(["build", "--n", "4", "--dump", str(FIXTURE), "--store", str(store), "--out", str(out)]) == 0
    assert read_set(out) == rows


def test_build_does_not_write_a_set_short_of_n(tmp_path, store, capsys):
    out = tmp_path / "set.jsonl"

    code = script.main(["build", "--n", "50", "--dump", str(FIXTURE), "--store", str(store), "--out", str(out)])

    assert code == 1
    assert not out.exists()
    assert "not written" in capsys.readouterr().err


def test_build_names_download_lod_when_the_dump_or_store_is_missing(tmp_path, store):
    out = str(tmp_path / "set.jsonl")
    with pytest.raises(SystemExit, match="none.xml is missing; run scripts/download_lod.py"):
        script.main(["build", "--dump", str(tmp_path / "none.xml"), "--store", str(store), "--out", out])
    with pytest.raises(SystemExit, match="download_lod.py"):
        script.main(["build", "--dump", str(FIXTURE), "--store", str(tmp_path / "none.sqlite"), "--out", out])
    assert not (tmp_path / "set.jsonl").exists()


# --- the real dump and store ---


@pytest.mark.models
def test_build_on_the_real_dump_and_store(tmp_path):
    dump = REAL_DATA / "lod" / "raw" / "new_lod-art.xml"
    store_path = REAL_DATA / "lod" / "lod.sqlite"
    if not dump.is_file() or not store_path.is_file():
        pytest.skip("the LOD dump or store is missing; run scripts/download_lod.py")
    out = tmp_path / "lb_judge.jsonl"

    started = time.monotonic()
    code = script.main(["build", "--n", "20", "--dump", str(dump), "--store", str(store_path), "--out", str(out)])

    assert code == 0
    assert time.monotonic() - started < 90
    rows = read_set(out)
    correct = [row for row in rows if row["label"] == "correct"]
    errors = [row for row in rows if row["label"] == "error"]
    assert len(correct) == 20 and len(errors) == 20
    lod = LodStore(store_path)
    try:
        for row in correct:
            assert all(checks.known(lod, word) for word in checks.words(row["text"])), row["text"]
    finally:
        lod.close()
    assert all(row["wrong"] != row["gold"] for row in errors)
    assert {row["error_type"] for row in errors} == set(lb_judge.CLASSES)


# --- the report ---


def test_upper_95_is_the_one_sided_exact_bound():
    assert round(upper_95(0, 100), 4) == 0.0295
    assert round(upper_95(25, 500), 4) == 0.0691
    assert round(upper_95(5, 200), 4) == 0.0518
    assert round(upper_95(0, 500), 3) == 0.006
    # 16 of 500 is the most false corrections that keep the bound under 5%.
    assert round(upper_95(16, 500), 3) == 0.048
    assert upper_95(16, 500) < 0.05 < upper_95(17, 500)
    # At the bound the chance of seeing at most x is 5%.
    assert binomial_cdf(16, 500, upper_95(16, 500)) == pytest.approx(0.05, abs=1e-9)
    assert upper_95(3, 3) == 1.0 and upper_95(0, 0) == 1.0


def outcome(label, verdict="good", tier="", **fields):
    judged = fields.pop("judged", verdict)
    return Outcome(f"{label[0]}-{random.random()}", label, judged=judged, verdict=verdict, tier=tier, **fields)


def test_summarise_counts_the_correct_arm():
    outcomes = [
        outcome("correct", "fix", "verified", category="grammar"),
        outcome("correct", "english", "likely", category="words"),
        outcome("correct", "fix", "unverified", category="words"),
        outcome("correct", "good", judged="fix", downgraded=True, alternative_lb="e Probleem"),
        outcome("correct", "good"),
        outcome("correct", "", judged="", error="timeout"),
    ]

    report = summarise(outcomes, engine="fake", model="oracle", date="2026-10-16", seed=7)

    arm = report.correct
    assert (arm["rows"], arm["n"], arm["errors"]) == (6, 5, 1)
    assert (arm["false_corrections"], arm["unverified_corrections"], arm["downgraded"]) == (2, 1, 1)
    assert arm["fpr"] == 2 / 5
    assert arm["upper_95"] == upper_95(2, 5)
    assert arm["tiers"] == {"verified": 1, "likely": 1, "unverified": 1, "downgraded": 1, "none": 1}
    assert arm["verdicts"] == {"fix": 2, "english": 1, "good": 2}
    assert report.errors == {"timeout": 1}
    assert report.labels["categories"] == {"grammar": 1, "words": 2}
    assert (report.review, report.reviewed_by, report.seed) == ("unreviewed", "none", 7)


def test_summarise_counts_recall_on_hits_and_any_fix_apart():
    outcomes = [
        outcome("error", "fix", "verified", error_type="verb_person", hit=True),
        outcome("error", "fix", "likely", error_type="verb_person", hit=False),
        outcome("error", "good", error_type="verb_person"),
        outcome("error", "fix", "likely", error_type="noun_gender", hit=True),
        outcome("error", "", judged="", error_type="noun_gender", error="schema"),
    ]

    arm = summarise(outcomes).error

    assert (arm["rows"], arm["n"], arm["errors"], arm["hit"], arm["caught"]) == (5, 4, 1, 2, 3)
    assert (arm["recall"], arm["recall_any"]) == (2 / 4, 3 / 4)
    verb = arm["by_class"]["verb_person"]
    assert (verb["n"], verb["caught"], verb["hit"], verb["recall"], verb["recall_any"]) == (3, 2, 1, 1 / 3, 2 / 3)
    assert verb["tiers"] == {"verified": 1, "likely": 1, "unverified": 0, "downgraded": 0, "none": 1}
    assert arm["by_class"]["noun_gender"]["n"] == 1
    assert arm["tiers"] == {"verified": 1, "likely": 2, "unverified": 0, "downgraded": 0, "none": 1}


FORBIDDEN = ("published", "native-validated", "fpr <=", "fpr ≤")


@pytest.mark.parametrize("false_corrections", [0, 3])
def test_headline_is_unreviewed_with_its_upper_bound_and_never_claims_more(false_corrections):
    outcomes = [outcome("correct", "fix", "likely") for _ in range(false_corrections)]
    outcomes += [outcome("correct") for _ in range(100 - false_corrections)]
    report = summarise(outcomes, engine="claude_cli", model="sonnet", date="2026-10-16")

    line = headline(report)

    assert line.startswith("Unreviewed false-correction rate:")
    assert f"{false_corrections} of 100 correct sentences" in line
    assert "upper bound" in line and "unreviewed" in line.casefold()
    assert "model sonnet" in line and "2026-10-16" in line
    assert not any(word in line.casefold() for word in FORBIDDEN)
    text = format_report(report)
    assert text.splitlines()[0] == line
    assert not any(word in text.casefold() for word in FORBIDDEN)


def test_headline_of_an_empty_run_still_says_unreviewed_and_upper_bound():
    line = headline(summarise([]))

    assert "0 of 0" in line and "upper bound" in line and "Unreviewed" in line


# --- the runner ---

QUESTION = Question("lb-judge", "general", "Sot mir eppes.", "Tell me something.")
FOUR_ROWS = [
    {"id": "c-1", "text": "Ech wunnen zu Esch.", "label": "correct"},
    {"id": "c-2", "text": "De Mann kuckt.", "label": "correct"},
    {"id": "e-1", "text": "Ech wunnt zu Esch.", "label": "error", "error_type": "verb_person",
     "wrong": "wunnt", "gold": "wunnen"},
    {"id": "e-2", "text": "Eng Mann kuckt.", "label": "error", "error_type": "noun_gender",
     "wrong": "eng", "gold": "de"},
]


def four_row_store():
    return KnownForms(
        ech="ech", wunnen="wunnen", wunnt="wunnen", zu="zu", esch="esch", de="den", mann="Mann",
        kuckt="kucken", eng="een",
    )


def test_the_content_has_the_one_question_of_the_set():
    tutor = load_practice().tutor

    assert (tutor["eval_lb"], tutor["eval_en"]) == ("Sot mir eppes.", "Tell me something.")


def test_judge_rows_sends_each_row_through_the_pipeline_with_the_oracle():
    llm = OracleJudge(FOUR_ROWS)
    clock = iter(range(100)).__next__

    outcomes = judge_rows(FOUR_ROWS, llm, four_row_store(), NoAcceptability(), "RULES", QUESTION, clock=clock)

    assert [(item.id, item.verdict) for item in outcomes] == [
        ("c-1", "good"), ("c-2", "good"), ("e-1", "fix"), ("e-2", "fix"),
    ]
    verb, gender = outcomes[2], outcomes[3]
    # A same-lemma swap is Verified; an article of another lemma is at most Likely.
    assert (verb.quote, verb.better_lb, verb.tier, verb.category, verb.hit) == (
        "wunnt", "Ech wunnen zu Esch.", "verified", "grammar", True,
    )
    assert (gender.quote, gender.better_lb, gender.tier, gender.hit) == ("Eng", "De Mann kuckt.", "likely", True)
    assert all(item.seconds == 1 and not item.error and not item.downgraded for item in outcomes)
    calls = [call for call in llm.calls if call.get("method") == "judge"]
    assert [call["answer"] for call in calls] == [row["text"] for row in FOUR_ROWS]
    assert {(call["question_lb"], call["question_en"], call["extra"]) for call in calls} == {
        ("Sot mir eppes.", "Tell me something.", ""),
    }
    assert all(call["grounding"].endswith("\n\nRULES") for call in calls)


def test_judge_rows_puts_an_llm_error_on_its_row_and_goes_on():
    llm = OracleJudge(FOUR_ROWS)
    llm.judge_queue.append(LLMError("timeout", "90 s"))
    done = []

    outcomes = judge_rows(
        FOUR_ROWS, llm, four_row_store(), NoAcceptability(), "", QUESTION,
        progress=lambda count, total, item: done.append((count, total, item.id)),
    )

    assert (outcomes[0].error, outcomes[0].verdict, outcomes[0].judged) == ("timeout", "", "")
    assert [item.verdict for item in outcomes[1:]] == ["good", "fix", "fix"]
    assert done == [(1, 4, "c-1"), (2, 4, "c-2"), (3, 4, "e-1"), (4, 4, "e-2")]
    assert summarise(outcomes).correct["n"] == 1


def test_judge_rows_marks_a_fix_the_checks_drop_as_downgraded():
    llm = OracleJudge([])
    # A quote that is not in the answer: the tutor never corrects what was not said.
    llm.judge_queue.append(Judgement("fix", "Hm.", quote="Fra", better_lb="Mann"))

    (item,) = judge_rows(FOUR_ROWS[:1], llm, four_row_store(), NoAcceptability(), "", QUESTION)

    assert (item.judged, item.verdict, item.downgraded, item.tier) == ("fix", "good", True, "")


def test_the_oracle_quotes_the_word_with_its_clitic_and_keeps_the_case():
    fix = oracle_judgement("Ech fueren d'nächste Woch.", {"wrong": "nächste", "gold": "nächst"})
    assert (fix.verdict, fix.quote, fix.better_lb) == ("fix", "d'nächste", "Ech fueren d'nächst Woch.")
    assert fix.claims == [{"form": "nächst", "lemma": ""}]

    fix = oracle_judgement("Hunn ech gaangen?", {"wrong": "hunn", "gold": "sinn"})
    assert (fix.quote, fix.better_lb) == ("Hunn", "Sinn ech gaangen?")
    # "e" is a word of its own, not a letter of another word.
    fix = oracle_judgement("Do steet e Fra.", {"wrong": "e", "gold": "eng"})
    assert (fix.quote, fix.better_lb) == ("e", "Do steet eng Fra.")
    assert oracle_judgement("Do steet eng Fra.", {"wrong": "e", "gold": "eng"}) is None


def test_the_oracle_says_good_to_a_correct_row_and_to_an_unknown_answer():
    llm = OracleJudge(FOUR_ROWS)

    assert llm.judge("q", "q", "Ech wunnen zu Esch.", "").verdict == "good"
    assert llm.judge("q", "q", "Eppes anescht.", "").verdict == "good"
    assert llm.judge("q", "q", "ech  wunnt zu Esch.", "").quote == "wunnt"


# --- scripts/lb_judge.py run ---


@pytest.fixture
def fixture_set(tmp_path, store):
    path = tmp_path / "set" / "lb_judge.jsonl"
    assert script.main(["build", "--n", "4", "--dump", str(FIXTURE), "--store", str(store), "--out", str(path)]) == 0
    return path


def run_args(set_path, store, report, *more):
    return ["run", *more, "--set", str(set_path), "--store", str(store), "--report", str(report)]


def test_run_fake_on_the_fixture_set_shows_no_false_correction_and_catches_the_errors(
    tmp_path, fixture_set, store, capsys
):
    report_path = tmp_path / "report" / "lb_judge_report.json"
    out = tmp_path / "printed.txt"

    code = script.main(run_args(fixture_set, store, report_path, "--fake", "--json", "--out", str(out)))

    assert code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert (report["reviewed_by"], report["review"], report["engine"], report["model"]) == (
        "none", "unreviewed", "fake", "oracle",
    )
    assert (report["correct"]["n"], report["correct"]["false_corrections"]) == (4, 0)
    assert report["correct"]["tiers"]["none"] == 4
    error = report["error"]
    assert error["n"] == 4 and error["hit"] == error["caught"]
    # Stage 1 labels a verb-person and an adjective-ending swap; both are caught.
    for name in ("verb_person", "adjective_ending"):
        assert error["by_class"][name]["recall"] == 1.0
    assert len(report["outcomes"]) == 8
    assert report["headline"].startswith("Unreviewed false-correction rate: 0 of 4 correct sentences")
    printed = capsys.readouterr().out
    assert report["headline"] in printed
    assert "recall per class:" in printed and "label mix:" in printed
    assert '"reviewed_by": "none"' in printed
    assert out.read_text(encoding="utf-8").startswith(report["headline"])


def test_run_draws_the_rows_with_the_seed(tmp_path, fixture_set, store):
    first, second = tmp_path / "first.json", tmp_path / "second.json"

    for path in (first, second):
        assert script.main(run_args(fixture_set, store, path, "--fake", "--correct", "2", "--errors", "1")) == 0

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in (first, second)]
    assert (reports[0]["correct"]["n"], reports[0]["error"]["n"]) == (2, 1)
    assert [item["id"] for item in reports[0]["outcomes"]] == [item["id"] for item in reports[1]["outcomes"]]


def test_run_fake_exits_1_when_the_pipeline_shows_a_false_correction(tmp_path, fixture_set, store, monkeypatch):
    class FixesEverything(OracleJudge):
        def judge(self, question_lb, question_en, answer, grounding, extra=""):
            first = answer.split()[0]
            return Judgement("fix", "Hm.", quote=first, better_lb=answer.replace(first, "Mir", 1))

    monkeypatch.setattr(script, "OracleJudge", FixesEverything)

    assert script.main(run_args(fixture_set, store, tmp_path / "r.json", "--fake")) == 1


def test_run_live_refuses_an_engine_that_is_not_the_claude_cli(tmp_path, fixture_set, store, capsys):
    report = tmp_path / "r.json"

    assert script.main(run_args(fixture_set, store, report, "--live")) == 2

    assert "claude_cli" in capsys.readouterr().err
    assert not report.exists()


def write_providers(data_dir, judge_kind):
    entries = {"local": {"kind": "fake", "model": "tiny"}, "cli": {"kind": "claude_cli", "model": "opus"}}
    roles = {role: {"provider": "local"} for role in ("tutor_chat", "teacher_explain", "wrapup")}
    roles["judge"] = {"provider": "cli" if judge_kind == "claude_cli" else "local"}
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "providers.json").write_text(json.dumps({"providers": entries, "roles": roles}))


def test_live_engine_takes_the_claude_cli_or_a_provider_whose_judge_is_on_it(tmp_path):
    data_dir = tmp_path / "data"
    assert "claude_cli only" in script.live_engine(Settings(llm_engine="fake", data_dir=data_dir))[2]

    cli = Settings(llm_engine="claude_cli", llm_model="sonnet", data_dir=data_dir)
    assert script.live_engine(cli) == ("claude_cli", "sonnet", "")

    provider = Settings(llm_engine="provider", data_dir=data_dir)
    write_providers(data_dir, "fake")
    assert "judge role" in script.live_engine(provider)[2]
    write_providers(data_dir, "claude_cli")
    assert script.live_engine(provider) == ("provider (cli)", "opus", "")


def test_run_live_needs_yes_above_the_budget(tmp_path, store, monkeypatch, capsys):
    monkeypatch.setenv("LOCI_LLM_ENGINE", "claude_cli")
    rows = [{"id": f"c-{k}", "text": f"Ech wunnen {k}.", "label": "correct"} for k in range(script.LIVE_ROWS + 1)]
    path = tmp_path / "big.jsonl"
    write_set(rows, path, seed=1, lod_dump="none")
    report = tmp_path / "r.json"

    assert script.main(run_args(path, store, report, "--live")) == 2

    assert "--yes" in capsys.readouterr().err
    assert not report.exists()


def test_run_names_build_when_the_set_is_missing(tmp_path, store):
    with pytest.raises(SystemExit, match="run scripts/lb_judge.py build"):
        script.main(run_args(tmp_path / "none.jsonl", store, tmp_path / "r.json", "--fake"))


@pytest.mark.live
def test_run_live_on_a_sample_of_the_real_set(request, tmp_path, monkeypatch, capsys):
    if "live" not in request.config.getoption("markexpr"):
        pytest.skip("calls the real claude CLI on the user's subscription; run with -m live")
    set_path, store_path = REAL_DATA / "eval" / "lb_judge.jsonl", REAL_DATA / "lod" / "lod.sqlite"
    if not set_path.is_file() or not store_path.is_file():
        pytest.skip("the LB-JUDGE set or the LOD store is missing; run scripts/lb_judge.py build")
    monkeypatch.setenv("LOCI_LLM_ENGINE", "claude_cli")
    report_path = tmp_path / "report.json"

    code = script.main(run_args(set_path, store_path, report_path, "--live", "--correct", "3", "--errors", "2"))

    printed = capsys.readouterr()
    if code == 2 and "not ready" in printed.err:
        pytest.skip(printed.err.strip())
    assert code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(report["outcomes"]) == 5
    assert (report["correct"]["rows"], report["reviewed_by"], report["engine"]) == (3, "none", "claude_cli")
    assert report["headline"] in printed.out
    assert f"of {report['correct']['n']} correct sentences" in report["headline"]
