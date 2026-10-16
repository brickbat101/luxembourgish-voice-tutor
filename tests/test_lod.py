"""LOD store: parser and lookups on the XML fixtures; golden set and counts on the built store."""

import json
import sqlite3
import time
import zipfile
from pathlib import Path

import httpx
import pytest

from app.lod import form_key
from app.lod.build import SCHEMA, build, variants
from app.lod.store import MAX_ENGLISH_ENTRIES, MAX_ENGLISH_TERMS, Headword, LodStore
from scripts import download_lod
from tests.realdata import REAL_DATA

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = json.loads((Path(__file__).parent / "golden_verbs.json").read_text(encoding="utf-8"))
FIXED_GOLDEN = (
    "sinn hunn goen kommen maachen kënnen wëllen ginn benotzen huelen "
    "schwätzen iessen drénken schlofen kafen gesinn wëssen soen fueren bleiwen"
).split()
API = "https://lod.lu/api/lb/entry/"
# lod.lu answers 403 to the bare python-httpx agent, so the test names itself.
USER_AGENT = "loci-language-app/0.1 (golden verb cross-check; python-httpx)"

GOEN_FORMS = (
    "goen gaangen gaang ginn gees geet gitt goung goungs goungen goungt "
    "géing géings géingen géingt géi"
).split()


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
def rows(small_path):
    db = sqlite3.connect(small_path)
    yield lambda sql: db.execute(sql).fetchall()
    db.close()


@pytest.fixture
def gloss_store(tmp_path):
    """Makes a store of (lemma, entry_id, pos, en_gloss, register) rows; `tables`: the ids with a verb table."""
    stores = []

    def make(headwords: list[tuple], tables: tuple[str, ...] = ()) -> LodStore:
        path = tmp_path / f"lod{len(stores)}.sqlite"
        db = sqlite3.connect(path)
        db.executescript(SCHEMA)
        db.executemany("INSERT INTO headwords VALUES (?, ?, ?, '', '', '', ?, ?)", headwords)
        db.executemany("INSERT INTO verbs (lemma, entry_id) VALUES ('', ?)", [(entry_id,) for entry_id in tables])
        db.commit()
        db.close()
        stores.append(LodStore(path))
        return stores[-1]

    yield make
    for store in stores:
        store.close()


@pytest.fixture(scope="module")
def real_path():
    path = REAL_DATA / "lod" / "lod.sqlite"
    if not path.is_file():
        pytest.skip(f"{path} is missing; run scripts/download_lod.py")
    return path


@pytest.fixture
def real(real_path):
    store = LodStore(real_path)
    yield store
    store.close()


# --- cell text ---


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("gees", ["gees"]),
        ("géi!", ["géi"]),
        ("gaangen / gaang", ["gaangen", "gaang"]),
        ("maachen (mech)", ["maachen"]),
        ("abonéiert (iech) ", ["abonéiert"]),
        ("hieft / hutt (iech)!", ["hieft", "hutt"]),
        ("kënns un", ["kënns un"]),
        ("dinn / doen of", ["dinn of", "doen of"]),
        ("paaks / pécks / peeks (dech) an", ["paaks an", "pécks an", "peeks an"]),
        ("sëfs / säifs dech dout", ["sëfs dech dout", "säifs dech dout"]),
        ("bekannt gemaach / bekannt gemaacht", ["bekannt gemaach", "bekannt gemaacht"]),
        ("erstinn op / operstinn", ["erstinn op", "operstinn"]),
        ("Kackaen", ["kackaen"]),
        ("", []),
    ],
)
def test_variants_of_a_cell(cell, expected):
    assert variants(cell) == expected


def test_form_key_folds_case_beyond_ascii():
    assert form_key("  ËMWELT ") == form_key("Ëmwelt") == "ëmwelt"
    assert form_key("Maach (dech)!") == "maach"


# --- parser on the 3-entry fixtures ---


def test_build_returns_the_row_counts(tmp_path):
    counts = build(FIXTURES / "lod_tab.xml", FIXTURES / "lod_art.xml", tmp_path / "lod.sqlite")

    # wordforms: goen 18, ausfléien 7, al 6, Kacka 3, Abrëll 2.
    assert counts == {"verbs": 2, "adjectives": 1, "headwords": 3, "wordforms": 36}
    assert [path.name for path in tmp_path.iterdir()] == ["lod.sqlite"]


def test_verb_rows(rows):
    verbs = rows("SELECT * FROM verbs")

    assert [row[:4] for row in verbs] == [
        ("goen", "GOEN1", "sinn", "gaangen / gaang"),
        ("ausfléien", "AUSFLEIEN1", "sinn / hunn", "ausgeflunn"),
    ]
    assert verbs[0][4:12] == ("ginn", "gees", "geet", "ginn", "gitt", "ginn", "géi!", "gitt!")
    # The present comes from the block with all six persons, here the second one.
    assert verbs[1][4:10] == ("fléien aus", "flitts aus", "flitt aus", "fléien aus", "flitt aus", "fléien aus")
    assert verbs[1][10:12] == ("fléi aus!", "flitt aus!")
    goen, ausfleien = (json.loads(row[12]) for row in verbs)
    assert (goen["model"], goen["separable"]) == ("GOEN1", False)
    assert goen["conjugations"][0]["indicative"]["pastSimple"]["p1"] == "goung"
    assert goen["conjugations"][0]["conditional"]["presentPerfect"]["p4"] == "géifen / géinge goen"
    assert (ausfleien["model"], ausfleien["separable"]) == ("ZEIEN2", True)
    assert [block["auxiliaryVerb"] for block in ausfleien["conjugations"]] == ["sinn", "hunn"]
    assert ausfleien["conjugations"][0]["indicative"]["present"] == {"p3": "flitt aus", "p6": "fléien aus"}


def test_adjective_row(rows):
    ((lemma, entry_id, forms_json),) = rows("SELECT * FROM adjectives")

    forms = json.loads(forms_json)
    assert (lemma, entry_id) == ("al", "AL1")
    assert forms["basicForms"] == {"positive": "al", "comparative": "méi al", "superlative": "am eelsten"}
    assert forms["declension"]["positive"]["singular"]["neutral"]["nominative"] == "aalt"
    assert forms["declension"]["positive"]["plural"]["masculineFeminineNeutral"]["dative"] == "alen"


def test_headword_rows(rows):
    assert rows("SELECT * FROM headwords") == [
        # EGS sits on 2 of 3 meanings only, so it is not the register of goen.
        ("goen", "GOEN1", "VRB", "", "", "goe", "to go", ""),
        ("Kacka", "KACKA1", "SUBST", "M", "Kackaen", "", "poo", "KANNERSPROOCH / EGS"),
        # A plural marked rare="true" is a word form but not the plural to teach.
        ("Abrëll", "ABRELL1", "SUBST", "M", "", "", "April", ""),
    ]


def test_wordform_rows(rows):
    def forms(lemma):
        return set(rows(f"SELECT form, kind FROM wordforms WHERE lemma = '{lemma}'"))

    assert forms("goen") == {("goen", "lemma"), ("goe", "n_rule")} | {(form, "verb_form") for form in GOEN_FORMS}
    assert forms("Kacka") == {("kacka", "lemma"), ("kackaen", "plural"), ("kackae", "n_rule")}
    assert forms("Abrëll") == {("abrëll", "lemma"), ("abrëller", "plural")}
    assert forms("al") == {
        (form, "adj_form") for form in ("al", "méi al", "am eelsten", "alen", "aler", "aalt")
    }
    assert ("flitts aus", "verb_form") in forms("ausfléien")
    assert ("fléi aus", "verb_form") in forms("ausfléien")
    # Compound tenses are auxiliary + participle, not forms of the verb.
    assert not rows("SELECT * FROM wordforms WHERE form IN ('si gaangen', 'ass ausgeflunn', 'ass', 'hunn')")


def test_a_failed_build_keeps_the_old_store(tmp_path):
    target = tmp_path / "lod.sqlite"
    build(FIXTURES / "lod_tab.xml", FIXTURES / "lod_art.xml", target)

    with pytest.raises(FileNotFoundError):
        build(FIXTURES / "lod_tab.xml", tmp_path / "missing.xml", target)

    store = LodStore(target)
    assert store.known("Abrëll")
    store.close()


# --- store on the fixtures ---


def test_conjugation(small):
    table = small.conjugation("GOEN")

    assert (table.lemma, table.entry_id, table.aux, table.participle) == ("goen", "GOEN1", "sinn", "gaangen / gaang")
    assert table.present == ["ginn", "gees", "geet", "ginn", "gitt", "ginn"]
    assert (table.imp_sg, table.imp_pl) == ("géi!", "gitt!")
    assert table.tables["conjugations"][0]["indicative"]["presentPerfect"]["p3"] == "ass gaangen / gaang"
    assert small.conjugation("gees") is None
    assert small.conjugation("xyzzy") is None


def test_headword(small):
    assert small.headword("kacka") == Headword(
        lemma="Kacka",
        entry_id="KACKA1",
        pos="SUBST",
        gender="M",
        plural="Kackaen",
        n_rule_form="",
        en_gloss="poo",
        register="KANNERSPROOCH / EGS",
    )
    assert small.headword("ABRËLL").lemma == "Abrëll"
    assert small.headword("Kackaen") is None
    assert small.headword("xyzzy") is None


def test_lookup_form_and_known(small):
    assert small.lookup_form("Gees") == ["goen"]
    assert small.lookup_form("goe") == ["goen"]
    assert small.lookup_form("ABRËLLER") == ["Abrëll"]
    assert small.lookup_form("Flitt  aus") == ["ausfléien"]
    assert small.lookup_form("géi!") == ["goen"]
    assert small.lookup_form("al") == ["al"]
    assert small.lookup_form("xyzzy") == []
    assert small.lookup_form("") == []
    assert small.known("gaang") and small.known("Kackae") and small.known("am eelsten")
    assert not small.known("xyzzy")


def test_verify(small):
    assert small.verify("gees")
    assert small.verify("gees", "goen")
    assert small.verify("Gees", "GOEN")
    assert not small.verify("gees", "ausfléien")
    assert not small.verify("xyzzy")
    assert not small.verify("xyzzy", "goen")


def test_grounding_block(small):
    assert small.grounding_for(["gees", "Kacka", "xyzzy", "ausfléien", "xyzzy"]) == "\n".join(
        [
            "LOD facts (forms as LOD prints them: 'a / b' are variants,"
            " '(mech)' is an optional reflexive pronoun)",
            '- goen [GOEN1] VRB: "to go"; n-rule form goe',
            "  auxiliary sinn; past participle gaangen / gaang",
            "  present: ech ginn, du gees, hien/si/et geet, mir ginn, dir gitt, si ginn",
            "  past simple: ech goung, du goungs, hien/si/et goung, mir goungen, dir goungt, si goungen",
            "  imperative: géi! (du), gitt! (dir)",
            '- Kacka [KACKA1] SUBST: "poo"; gender M; plural Kackaen; register KANNERSPROOCH / EGS',
            # No article entry in the fixture: the verb table alone.
            "- ausfléien [AUSFLEIEN1] VRB",
            "  auxiliary sinn / hunn; past participle ausgeflunn",
            "  present: ech fléien aus, du flitts aus, hien/si/et flitt aus,"
            " mir fléien aus, dir flitt aus, si fléien aus",
            "  imperative: fléi aus! (du), flitt aus! (dir)",
            "not found in LOD: xyzzy",
        ]
    )
    assert small.grounding_for(["Abrëll"]).splitlines()[1:] == [
        '- Abrëll [ABRELL1] SUBST: "April"; gender M',
        "not found in LOD: none",
    ]


def test_lookup_english_matches_the_gloss_with_or_without_to(small):
    assert small.lookup_english("go") == ["goen"]
    assert small.lookup_english("  To   GO ") == ["goen"]
    assert small.lookup_english("april") == ["Abrëll"]
    assert small.lookup_english("poo") == ["Kacka"]
    # Only the first translation of the first meaning is the gloss.
    assert small.lookup_english("walk") == []
    assert small.lookup_english("xyzzy") == []
    assert small.lookup_english("") == []


def test_english_words_are_the_words_of_the_glosses_case_folded(small, gloss_store):
    assert small.english_words() == {"to", "go", "poo", "april"}
    assert small.english_words() is small.english_words()
    store = gloss_store([("Pätter", "P1", "SUBST", "Godfather (of a child)", ""), ("eidel", "E1", "ADJ", "", "")])
    assert store.english_words() == {"godfather", "of", "a", "child"}


def test_lookup_english_keeps_three_verbs_and_two_other_words_each_ranked_on_its_own(gloss_store):
    store = gloss_store(
        [
            ("disposéieren", "DISPOSEIEREN1", "VRB", "to have", ""),
            ("hunn", "HUNN1", "VRB", "to have", ""),
            ("hunn", "HUNN3", "VRB", "to have", ""),
            ("Mupp", "MUPP1", "SUBST", "dog", "EGS"),
            ("Ridd", "RIDD1", "SUBST", "dog", ""),
            ("Hond", "HOND1", "SUBST", "Dog", ""),
            ("Aarbechtsplaz", "AARBECHTSPLAZ1", "SUBST", "work", ""),
            ("Aarbecht", "AARBECHT1", "SUBST", "work", ""),
            ("Tätegkeet", "TATEGKEET1", "SUBST", "work", ""),
            ("wierken", "WIERKEN1", "VRB", "to work", ""),
            ("matschaffen", "MATSCHAFFEN1", "VRB", "to work", ""),
            ("schaffen", "SCHAFFEN3", "VRB", "to work", ""),
            ("déngen", "DENGEN3", "VRB", "to work", ""),
            ("Jitz", "JITZ1", "SUBST", "come", "VULG"),
            ("kommen", "KOMMEN1", "VRB", "to come", ""),
            ("nammen", "NAMMEN1", "VRB", "to eat", "KANNERSPROOCH"),
            ("friessen", "FRIESSEN1", "VRB", "to eat", ""),
            ("iessen", "IESSEN3", "VRB", "to eat", ""),
            ("opiessen", "OPIESSEN1", "VRB", "to eat", ""),
        ],
        tables=(
            "DISPOSEIEREN1",
            "HUNN1",
            "HUNN3",
            "MATSCHAFFEN1",
            "SCHAFFEN3",
            "DENGEN3",
            "KOMMEN1",
            "NAMMEN1",
            "FRIESSEN1",
            "IESSEN3",
            "OPIESSEN1",
        ),
    )

    # The shorter lemma first, and a lemma with two entries once.
    assert store.lookup_english("have") == ["hunn", "disposéieren"]
    # Two other words: the one with a register label is left out; equally long ones in dump order.
    assert store.lookup_english("dog") == ["Ridd", "Hond"]
    # The two shortest nouns, then three verbs: those with a verb table before the shorter wierken.
    assert store.lookup_english("work") == ["Aarbecht", "Tätegkeet", "déngen", "schaffen", "matschaffen"]
    # Asked as a verb, the verbs come first.
    assert store.lookup_english("to work") == ["déngen", "schaffen", "matschaffen", "Aarbecht", "Tätegkeet"]
    # A register label goes last, also across the two kinds.
    assert store.lookup_english("come") == ["kommen", "Jitz"]
    assert store.lookup_english("eat") == ["iessen", "friessen", "opiessen"]


def test_grounding_block_with_english_words(small):
    block = small.grounding_for(["xyzzy", "poo", "have"], ["to go", "go", "poo", "xyzzy", "have a"])

    assert block.splitlines() == [
        "LOD facts (forms as LOD prints them: 'a / b' are variants,"
        " '(mech)' is an optional reflexive pronoun)",
        # "go" is a word of the phrase that matched, so it is not looked up again.
        'English "to go" -> goen',
        '- goen [GOEN1] VRB: "to go"; n-rule form goe',
        "  auxiliary sinn; past participle gaangen / gaang",
        "  present: ech ginn, du gees, hien/si/et geet, mir ginn, dir gitt, si ginn",
        "  past simple: ech goung, du goungs, hien/si/et goung, mir goungen, dir goungt, si goungen",
        "  imperative: géi! (du), gitt! (dir)",
        'English "poo" -> Kacka',
        '- Kacka [KACKA1] SUBST: "poo"; gender M; plural Kackaen; register KANNERSPROOCH / EGS',
        # "poo" matched as English; "xyzzy" and "have" matched neither way.
        "not found in LOD: xyzzy, have",
    ]


def test_grounding_prints_an_entry_once_when_a_form_and_an_english_word_lead_to_it(small):
    assert small.grounding_for(["gees", "Go"], ["Go"]).splitlines()[1:] == [
        '- goen [GOEN1] VRB: "to go"; n-rule form goe',
        "  auxiliary sinn; past participle gaangen / gaang",
        "  present: ech ginn, du gees, hien/si/et geet, mir ginn, dir gitt, si ginn",
        "  past simple: ech goung, du goungs, hien/si/et goung, mir goungen, dir goungt, si goungen",
        "  imperative: géi! (du), gitt! (dir)",
        'English "go" -> goen',
        "not found in LOD: none",
    ]


def test_grounding_shows_six_english_words_and_drops_the_later_ones(gloss_store):
    numbers = "one two three four five six seven eight".split()
    store = gloss_store([(f"lemma{index}", f"ID{index}", "SUBST", gloss, "") for index, gloss in enumerate(numbers)])

    lines = store.grounding_for(numbers + ["nine"], numbers + ["nine"]).splitlines()

    assert MAX_ENGLISH_TERMS == 6
    assert [line for line in lines if line.startswith("English")] == [
        f'English "{gloss}" -> lemma{index}' for index, gloss in enumerate(numbers[:6])
    ]
    assert [line for line in lines if line.startswith("- ")] == [
        f'- lemma{index} [ID{index}] SUBST: "{gloss}"' for index, gloss in enumerate(numbers[:6])
    ]
    # "seven" and "eight" have entries that the cap left out; only "nine" is not in LOD.
    assert lines[-1] == "not found in LOD: nine"


def test_grounding_prints_fifteen_entries_for_the_english_words_and_drops_the_rest(gloss_store):
    numbers = "one two three four five".split()
    rows = []
    for index, number in enumerate(numbers):
        rows += [(f"n{index}{letter}", f"N{index}{letter}", "SUBST", number, "") for letter in "ab"]
        rows += [(f"v{index}{letter}", f"V{index}{letter}", "VRB", f"to {number}", "") for letter in "ab"]
    store = gloss_store(rows)

    lines = store.grounding_for(numbers, numbers).splitlines()

    assert MAX_ENGLISH_ENTRIES == 15
    assert [line for line in lines if line.startswith("English")] == [
        f'English "{number}" -> n{index}a, n{index}b, v{index}a, v{index}b' for index, number in enumerate(numbers[:4])
    ]
    # Four entries for each of "one", "two" and "three", and the first three of "four".
    assert [line.split()[1] for line in lines if line.startswith("- ")] == [
        f"{kind}{index}{letter}" for index in range(4) for kind in "nv" for letter in "ab"
    ][:15]
    assert lines[-1] == "not found in LOD: none"


def test_a_missing_store_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_lod"):
        LodStore(tmp_path / "lod.sqlite")


# --- download script, without the network ---


def test_extract_keeps_xml_inside_the_raw_folder(tmp_path):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    with zipfile.ZipFile(tmp_path / "dump.zip", "w") as archive:
        archive.writestr("../../evil.xml", "<x/>")
        archive.writestr("sub/ok.xml", "<y/>")
        archive.writestr("run.bat", "echo no")

    extracted = download_lod.extract(tmp_path / "dump.zip", raw)

    assert sorted(path.name for path in extracted) == ["evil.xml", "ok.xml"]
    assert sorted(path.name for path in raw.iterdir()) == ["evil.xml", "ok.xml"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["data", "dump.zip"]


def test_extract_refuses_an_oversized_archive(tmp_path, monkeypatch):
    with zipfile.ZipFile(tmp_path / "dump.zip", "w") as archive:
        archive.writestr("big.xml", "<x>" + "a" * 100 + "</x>")
    monkeypatch.setattr(download_lod, "MAX_UNPACKED_BYTES", 50)

    with pytest.raises(ValueError, match="unpacks to more than"):
        download_lod.extract(tmp_path / "dump.zip", tmp_path)


def test_fetch_skips_a_file_that_is_present(tmp_path, monkeypatch):
    present = tmp_path / "260727-new-lod-tab.zip"
    present.write_bytes(b"zip")

    def no_network(*args, **kwargs):
        raise AssertionError("network used")

    monkeypatch.setattr(httpx, "stream", no_network)

    assert download_lod.fetch(download_lod.URLS[0], tmp_path) == present
    assert present.read_bytes() == b"zip"


# --- golden set ---


def test_golden_file_has_40_verbs_with_the_fixed_20():
    lemmas = [verb["lemma"] for verb in GOLDEN["verbs"]]

    assert len(lemmas) == len(set(lemmas)) == 40
    assert lemmas[:20] == FIXED_GOLDEN
    assert all(len(verb["present"]) == 6 and verb["aux"] and verb["participle"] for verb in GOLDEN["verbs"])
    assert set(GOLDEN["api_checked"]) <= set(lemmas) and len(GOLDEN["api_checked"]) == 5


@pytest.mark.models
@pytest.mark.parametrize("golden", GOLDEN["verbs"], ids=lambda verb: verb["entry_id"])
def test_golden_verb_equals_the_store(real, golden):
    table = real.conjugation(golden["lemma"])

    assert table is not None
    assert table.lemma == golden["lemma"]
    assert table.entry_id == golden["entry_id"]
    assert table.aux == golden["aux"]
    assert table.participle == golden["participle"]
    assert table.present == golden["present"]


@pytest.mark.models
def test_real_store_counts(real_path):
    db = sqlite3.connect(real_path)
    verbs, headwords = (db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in ("verbs", "headwords"))
    db.close()

    assert verbs > 2_000
    assert headwords > 30_000


@pytest.mark.models
def test_real_store_knows_forms(real):
    assert real.known("benotzen")
    assert real.known("benotze")
    assert real.known("geholl")
    assert not real.known("xyzzy")
    assert real.lookup_form("geholl") == ["huelen"]
    assert real.headword("ëmwelt").lemma == "Ëmwelt"


@pytest.mark.models
def test_real_grounding_has_the_goen_table_and_the_misses(real):
    block = real.grounding_for(["goen", "xyzzy"])

    assert block.splitlines()[0].startswith("LOD facts")
    assert '- goen [GOEN1] VRB: "to go"; n-rule form goe' in block
    assert "  auxiliary sinn; past participle gaangen / gaang" in block
    assert "  present: ech ginn, du gees, hien/si/et geet, mir ginn, dir gitt, si ginn" in block
    assert block.splitlines()[-1] == "not found in LOD: xyzzy"


@pytest.mark.models
def test_real_lookup_english_puts_the_everyday_word_first(real):
    assert real.lookup_english("have") == ["hunn", "disposéieren"]
    assert real.lookup_english("go")[0] == "goen"
    assert real.lookup_english("dog") == ["Hond", "Ridd"]
    # nammen is KANNERSPROOCH.
    assert real.lookup_english("eat") == ["iessen", "friessen", "nammen"]
    # Dank is VEREELZT.
    assert real.lookup_english("thank you") == ["merci", "Dank"]
    assert real.lookup_english("good morning") == ["moien"]
    assert real.lookup_english("tired") == ["midd"]
    assert "sinn" in real.lookup_english("be")
    assert real.lookup_english("xyzzy") == []


@pytest.mark.models
def test_real_lookup_english_keeps_the_noun_next_to_the_verbs(real):
    assert real.lookup_english("water") == ["Waasser", "Gewässer", "nätzen", "dränken", "tréinen"]
    assert real.lookup_english("to water") == ["nätzen", "dränken", "tréinen", "Waasser", "Gewässer"]
    assert real.lookup_english("work") == ["Aarbecht", "Tätegkeet", "déngen", "wierken", "schaffen"]
    assert real.lookup_english("book") == ["Buch", "buchen", "protokolléieren"]
    # Jitz and jitzen are VULG.
    assert real.lookup_english("come") == ["kommen", "matfueren", "matkommen", "Jitz"]


@pytest.mark.models
def test_real_english_words_have_the_english_look_alikes_and_no_luxembourgish_forms(real):
    words = real.english_words()

    assert {"war", "hat", "an", "all", "am", "do"} <= words
    assert not {"ginn", "gees", "geet", "waasser", "gutt", "laang", "midd"} & words


@pytest.mark.models
def test_real_grounding_for_english_words_has_the_hunn_table_and_the_hond_entry(real):
    words = ["have", "dog", "How", "does", "the", "word", "work", "Luxembourgish"]

    block = real.grounding_for(words, ["i have", "have a", "have", "a dog", "dog"])

    assert 'English "have" -> hunn, disposéieren' in block
    assert '- hunn [HUNN1] VRB: "to have"; n-rule form hu' in block
    assert "  present: ech hunn, du hues, hien/si/et huet, mir hunn, dir hutt, si hunn" in block
    assert 'English "dog" -> Hond, Ridd' in block
    assert '- Hond [HOND1] SUBST: "dog"; gender M; plural Hënn / Honn' in block
    assert block.splitlines()[-1] == "not found in LOD: How, does, the, word, work, Luxembourgish"


@pytest.mark.live
def test_golden_verbs_equal_the_lod_api(request):
    if "live" not in request.config.getoption("markexpr"):
        pytest.skip("calls lod.lu five times; run with -m live")
    verbs = {verb["lemma"]: verb for verb in GOLDEN["verbs"]}
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=20) as client:
        for lemma in GOLDEN["api_checked"]:
            golden = verbs[lemma]
            try:
                response = client.get(API + golden["entry_id"])
            except httpx.TransportError as error:
                pytest.skip(f"lod.lu is not reachable: {error!r}")
            response.raise_for_status()
            table = response.json()["entry"]["tables"]["verbConjugation"]
            present = table["indicative"]["present"]
            assert table["auxiliaryVerb"] == golden["aux"], lemma
            assert table["pastParticiple"] == golden["participle"], lemma
            assert [present.get(f"p{person}", "") for person in range(1, 7)] == golden["present"], lemma
            time.sleep(0.1)
