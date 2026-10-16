"""Parse the two LOD XML dumps into one SQLite file. XML names and table columns: SCHEMA.md."""

from __future__ import annotations

import json
import os
import sqlite3
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

from app.lod import form_key

TAB_XML = "new_lod-tab.xml"  # Flexiounstabellen: verb conjugation and adjective declension
ART_XML = "new_lod-art.xml"  # linguistic data: one <entry> per headword

TABLES = ("verbs", "adjectives", "headwords", "wordforms")
SCHEMA = """
CREATE TABLE verbs (lemma TEXT, entry_id TEXT, aux TEXT, participle TEXT,
    p1 TEXT, p2 TEXT, p3 TEXT, p4 TEXT, p5 TEXT, p6 TEXT, imp_sg TEXT, imp_pl TEXT, tables_json TEXT);
CREATE TABLE adjectives (lemma TEXT, entry_id TEXT, forms_json TEXT);
CREATE TABLE headwords (lemma TEXT, entry_id TEXT, pos TEXT, gender TEXT, plural TEXT,
    n_rule_form TEXT, en_gloss TEXT, register TEXT);
CREATE TABLE wordforms (form TEXT, lemma TEXT, kind TEXT, PRIMARY KEY (form, lemma, kind)) WITHOUT ROWID;
CREATE INDEX verbs_lemma ON verbs (lemma);
CREATE INDEX headwords_lemma ON headwords (lemma);
"""

# The tenses whose cells are forms of the verb itself; the others are auxiliary + participle.
SIMPLE_TENSES = (
    ("indicative", "present"),
    ("indicative", "pastSimple"),
    ("conditional", "presentSimple"),
    ("imperative", "present"),
)


def variants(cell: str) -> list[str]:
    """The lookup keys of the forms in one table cell.

    " / " separates alternatives. When only the last one has several words, the others
    share its tail: "dinn / doen of" is "dinn of" and "doen of". Otherwise each alternative
    is complete: "bekannt gemaach / bekannt gemaacht", "erstinn op / operstinn".
    """
    parts = [part.split() for part in form_key(cell).split("/")]
    parts = [part for part in parts if part]
    if all(len(part) == 1 for part in parts[:-1]):
        parts = [part[:1] + parts[-1][1:] for part in parts]
    return [" ".join(part) for part in parts]


def _records(path: Path, tag: str) -> Iterator[ET.Element]:
    """Yield every `tag` element of a dump, emptying each one after use."""
    for _, element in ET.iterparse(path):
        if element.tag == tag:
            yield element
            element.clear()


def _text(element: ET.Element | None) -> str:
    return (element.text or "").strip() if element is not None else ""


def _tree(element: ET.Element) -> dict | str:
    """An element as nested dicts keyed by child tag; a leaf is its trimmed text."""
    if len(element) == 0:
        return _text(element)
    return {child.tag: _tree(child) for child in element}


def _cells(block: dict, mood: str, tense: str) -> dict:
    """The {"p1": ..} cells of one tense of a conjugation block, {} if absent."""
    cells = block.get(mood)
    cells = cells.get(tense) if isinstance(cells, dict) else None
    return cells if isinstance(cells, dict) else {}


def _verb(table: ET.Element) -> tuple[tuple, list[tuple[str, str]]]:
    """The verbs row of one <verbConjugation> and its (form, kind) pairs."""
    lemma = _text(table.find("infinitive"))
    participle = _text(table.find("pastParticiple"))
    # A table repeats auxiliaryVerb, indicative, conditional, imperative once per use of the verb.
    blocks: list[dict] = []
    for child in table:
        if child.tag in ("infinitive", "pastParticiple"):
            continue
        if child.tag == "auxiliaryVerb" or not blocks:
            blocks.append({})
        blocks[-1][child.tag] = _tree(child)
    main = max(blocks, key=lambda block: len(_cells(block, "indicative", "present")), default={})
    present = _cells(main, "indicative", "present")
    imperative = _cells(main, "imperative", "present")
    aux = dict.fromkeys(block["auxiliaryVerb"] for block in blocks if block.get("auxiliaryVerb"))
    tables = {
        "model": table.get("model", ""),
        "separable": table.get("separableVerb") == "yes",
        "conjugations": blocks,
    }
    row = (
        lemma,
        table.get("id", ""),
        " / ".join(aux),
        participle,
        *(present.get(f"p{person}", "") for person in range(1, 7)),
        imperative.get("p2", ""),
        imperative.get("p5", ""),
        json.dumps(tables, ensure_ascii=False),
    )
    cells = [lemma, participle]
    for block in blocks:
        for mood, tense in SIMPLE_TENSES:
            cells += _cells(block, mood, tense).values()
    forms = [(form, "verb_form") for cell in cells for form in variants(cell)]
    return row, [(lemma, "lemma"), *forms]


def _adjective(table: ET.Element) -> tuple[tuple, list[tuple[str, str]]]:
    """The adjectives row of one <adjDeclension> and its (form, kind) pairs."""
    lemma = _text(table.find("basicForms/positive"))
    cells = [_text(leaf) for leaf in table.iter() if len(leaf) == 0]
    forms = [(form, "adj_form") for cell in cells for form in variants(cell)]
    return (lemma, table.get("id", ""), json.dumps(_tree(table), ensure_ascii=False)), forms


def _headword(entry: ET.Element) -> tuple[tuple, list[tuple[str, str]]]:
    """The headwords row of one <entry> and its (form, kind) pairs."""
    lemma = entry.find("lemma")
    n_rule_form = lemma.get("nRuleForm", "") if lemma is not None else ""
    pos = entry.find("microStructure/partOfSpeech")
    plurals = entry.findall(".//inflection[@type='plural']/form")
    labels = [_text(label) for label in entry.findall("microStructure/attribute")]
    per_meaning = [
        [_text(label) for label in meaning.findall("attribute")]
        for meaning in entry.findall("microStructure/grammaticalUnit/meaning")
    ]
    # Labels on single meanings describe the entry only when no meaning is unlabelled.
    if per_meaning and all(per_meaning):
        labels += [label for meaning in per_meaning for label in meaning]
    row = (
        _text(lemma),
        entry.get("id", ""),
        _text(pos),
        pos.get("gen", "") if pos is not None else "",
        " / ".join(dict.fromkeys(_text(form) for form in plurals if form.get("rare") != "true")),
        n_rule_form,
        _text(entry.find("microStructure/grammaticalUnit/meaning/targetLanguage[@lang='en']/translation")),
        " / ".join(dict.fromkeys(labels)),
    )
    forms = [(_text(lemma), "lemma"), (n_rule_form, "n_rule")]
    for form in plurals:
        forms += [(_text(form), "plural"), (form.get("nRuleForm", ""), "n_rule")]
    return row, forms


def _store(db: sqlite3.Connection, table: str, row: tuple, forms: list[tuple[str, str]]) -> None:
    """Insert one row and its word forms; a record without a lemma is dropped."""
    lemma = row[0]
    if not lemma:
        return
    db.execute(f"INSERT INTO {table} VALUES ({', '.join('?' * len(row))})", row)
    keyed = [(key, lemma, kind) for form, kind in forms if (key := form_key(form))]
    db.executemany("INSERT OR IGNORE INTO wordforms VALUES (?, ?, ?)", keyed)


def build(tab_xml: Path, art_xml: Path, target: Path) -> dict[str, int]:
    """Build the store in a temp file next to `target`, then move it into place.

    Returns the row count of each table.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    temp.unlink(missing_ok=True)
    db = sqlite3.connect(temp)
    try:
        db.executescript(SCHEMA)
        for inflection in _records(tab_xml, "inflection"):
            for table in inflection.findall("verbConjugation"):
                _store(db, "verbs", *_verb(table))
            for table in inflection.findall("adjDeclension"):
                _store(db, "adjectives", *_adjective(table))
        for entry in _records(art_xml, "entry"):
            _store(db, "headwords", *_headword(entry))
        db.commit()
        counts = {table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in TABLES}
    finally:
        db.close()
    os.replace(temp, target)
    return counts
