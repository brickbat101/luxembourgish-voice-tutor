"""Read-only lookups in the LOD store built by app/lod/build.py.

Lookups ignore case; lemmas and forms come back in LOD's own casing. A miss is None, False
or an empty list, never a guess.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from app.lod import form_key

# p1..p6 of a verb table, in this order.
PERSONS = ("ech", "du", "hien/si/et", "mir", "dir", "si")
HEADER = (
    "LOD facts (forms as LOD prints them: 'a / b' are variants,"
    " '(mech)' is an optional reflexive pronoun)"
)
MAX_ENGLISH_TERMS = 6  # English words or phrases that bring entries into one grounding block
MAX_ENGLISH_ENTRIES = 15  # entries that all of them together bring into it
MAX_VERB_LEMMAS = 3  # lemmas per English word that LOD glosses as "to " + the word
MAX_OTHER_LEMMAS = 2  # lemmas per English word that LOD glosses as the word itself
_GLOSS_WORD = re.compile(r"[^\W\d_]+")


@dataclass(frozen=True)
class VerbTable:
    lemma: str
    entry_id: str
    aux: str  # "hunn", "sinn" or "hunn / sinn"
    participle: str
    present: list[str]  # p1..p6 as LOD prints them, "" for a person the verb lacks
    imp_sg: str
    imp_pl: str
    tables: dict  # {"model", "separable", "conjugations": [one block per use of the verb]}


@dataclass(frozen=True)
class Headword:
    lemma: str
    entry_id: str
    pos: str
    gender: str
    plural: str
    n_rule_form: str
    en_gloss: str
    register: str


def _verb_table(row: sqlite3.Row) -> VerbTable:
    return VerbTable(
        lemma=row["lemma"],
        entry_id=row["entry_id"],
        aux=row["aux"],
        participle=row["participle"],
        present=[row[f"p{person}"] for person in range(1, 7)],
        imp_sg=row["imp_sg"],
        imp_pl=row["imp_pl"],
        tables=json.loads(row["tables_json"]),
    )


def _persons(cells: list[str]) -> str:
    return ", ".join(f"{person} {cell}" for person, cell in zip(PERSONS, cells) if cell)


def _gloss_key(text: str) -> str:
    return " ".join(text.split()).casefold()


def _first(ranked: list[tuple], limit: int) -> list[tuple[bool, str]]:
    """(labelled, lemma) of the first `limit` distinct lemmas of ranked gloss-index entries."""
    picked: dict[str, bool] = {}
    for labelled, *_, lemma in ranked:
        picked.setdefault(lemma, labelled)
    return [(labelled, lemma) for lemma, labelled in picked.items()][:limit]


class LodStore:
    def __init__(self, path: Path) -> None:
        if not Path(path).is_file():
            raise FileNotFoundError(f"{path} is missing; run scripts/download_lod.py")
        # Read-only, and usable from the server's worker threads.
        uri = Path(path).resolve().as_uri() + "?mode=ro"
        self._db = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._glosses: dict[str, list[tuple]] | None = None
        self._english: frozenset[str] | None = None

    def close(self) -> None:
        self._db.close()

    def _entries(self, table: str, word: str) -> list[sqlite3.Row]:
        """The rows of `table` whose lemma is `word`: exact casing first, then dump order."""
        return self._db.execute(
            f"SELECT * FROM {table} WHERE lemma IN"
            " (SELECT lemma FROM wordforms WHERE form = ? AND kind = 'lemma')"
            " ORDER BY lemma != ?, rowid",
            (form_key(word), word),
        ).fetchall()

    def conjugation(self, lemma: str) -> VerbTable | None:
        """The verb table of `lemma`; of several entries with that lemma, the first in the dump."""
        rows = self._entries("verbs", lemma)
        return _verb_table(rows[0]) if rows else None

    def headword(self, word: str) -> Headword | None:
        """The headword entry whose lemma is `word`; of several, the first in the dump."""
        rows = self._entries("headwords", word)
        return Headword(*rows[0]) if rows else None

    def lookup_form(self, form: str) -> list[str]:
        """The lemmas that `form` is a form of: lemma, plural, n-rule form, verb or adjective form."""
        rows = self._db.execute(
            "SELECT DISTINCT lemma FROM wordforms WHERE form = ? ORDER BY lemma", (form_key(form),)
        )
        return [row["lemma"] for row in rows]

    def known(self, form: str) -> bool:
        return bool(self.lookup_form(form))

    def verify(self, form: str, lemma: str | None = None) -> bool:
        """True when LOD lists `form`, and as a form of `lemma` when one is given."""
        lemmas = self.lookup_form(form)
        if lemma is None:
            return bool(lemmas)
        return form_key(lemma) in {form_key(found) for found in lemmas}

    def lookup_english(self, term: str) -> list[str]:
        """The lemmas LOD glosses as `term`: verbs glossed "to " + term and words glossed as the term.

        At most MAX_VERB_LEMMAS verbs and MAX_OTHER_LEMMAS other words, so neither kind pushes
        the other out. In each kind entries without a register label come first, then, of the
        verbs, those with a verb table, then the shorter lemma, then dump order. The other
        words come before the verbs; the verbs come first when `term` itself starts with
        "to ". A lemma with a register label never comes before one without.
        """
        word = _gloss_key(term)
        verb_asked = word.startswith("to ")
        word = word.removeprefix("to ")
        if not word:
            return []
        glosses = self._gloss_index()
        verbs = _first(sorted(glosses.get("to " + word, [])), MAX_VERB_LEMMAS)
        others = sorted(glosses.get(word, []), key=lambda rank: (rank[0], rank[2], rank[3]))
        others = _first(others, MAX_OTHER_LEMMAS)
        picked = sorted(verbs + others if verb_asked else others + verbs, key=lambda pick: pick[0])
        return list(dict.fromkeys(lemma for _, lemma in picked))

    def _gloss_index(self) -> dict[str, list[tuple]]:
        """Gloss key -> (labelled, no verb table, lemma length, dump position, lemma) per entry.

        The store has no index on en_gloss, so the glosses are read once, on first use.
        """
        if self._glosses is None:
            with_table = {row["entry_id"] for row in self._db.execute("SELECT entry_id FROM verbs")}
            glosses: dict[str, list[tuple]] = {}
            for position, lemma, entry_id, gloss, register in self._db.execute(
                "SELECT rowid, lemma, entry_id, en_gloss, register FROM headwords WHERE en_gloss != ''"
            ):
                rank = (bool(register), entry_id not in with_table, len(lemma), position, lemma)
                glosses.setdefault(_gloss_key(gloss), []).append(rank)
            self._glosses = glosses
        return self._glosses

    def english_words(self) -> frozenset[str]:
        """Every word of the English glosses, case-folded; collected once, on first use."""
        if self._english is None:
            self._english = frozenset(word for gloss in self._gloss_index() for word in _GLOSS_WORD.findall(gloss))
        return self._english

    def grounding_for(self, words: list[str], english: list[str] | None = None) -> str:
        """The plain-text "LOD facts" block for a prompt.

        First every entry the `words` belong to as Luxembourgish forms. Then, for each of
        the first MAX_ENGLISH_TERMS `english` words or phrases that LOD has as a gloss, a
        line naming its lemmas, and their entries until MAX_ENGLISH_ENTRIES are printed. A
        word of a phrase that matched is not looked up on its own. No entry is printed twice.
        """
        lines = [HEADER]
        seen: set[str] = set()
        missing: list[str] = []
        for word in words:
            lemmas = self.lookup_form(word)
            if not lemmas and word not in missing:
                missing.append(word)
            for lemma in lemmas:
                if lemma not in seen:
                    seen.add(lemma)
                    lines += [line for entry in self._facts(lemma) for line in entry]
        matched: set[str] = set()  # the English terms that have entries, and their words
        shown = printed = 0
        for term in english or []:
            key = _gloss_key(term)
            lemmas = [] if key in matched else self.lookup_english(key)
            if not lemmas:
                continue
            matched.update((key, *key.split()))
            if shown == MAX_ENGLISH_TERMS or printed == MAX_ENGLISH_ENTRIES:
                continue
            shown += 1
            lines.append(f'English "{key}" -> {", ".join(lemmas)}')
            for lemma in lemmas:
                if lemma not in seen:
                    seen.add(lemma)
                    entries = self._facts(lemma)[: MAX_ENGLISH_ENTRIES - printed]
                    printed += len(entries)
                    lines += [line for entry in entries for line in entry]
        missing = [word for word in missing if _gloss_key(word) not in matched]
        lines.append("not found in LOD: " + (", ".join(missing) or "none"))
        return "\n".join(lines)

    def _facts(self, lemma: str) -> list[list[str]]:
        """The grounding lines of every entry spelled exactly `lemma`, one list per entry."""
        verbs = {
            row["entry_id"]: _verb_table(row)
            for row in self._db.execute("SELECT * FROM verbs WHERE lemma = ? ORDER BY rowid", (lemma,))
        }
        entries = []
        for row in self._db.execute("SELECT * FROM headwords WHERE lemma = ? ORDER BY rowid", (lemma,)):
            head = Headword(*row)
            facts = [f'"{head.en_gloss}"'] if head.en_gloss else []
            facts += [
                f"{label} {value}"
                for label, value in (
                    ("gender", head.gender),
                    ("plural", head.plural),
                    ("n-rule form", head.n_rule_form),
                    ("register", head.register),
                )
                if value
            ]
            title = f"- {head.lemma} [{head.entry_id}] {head.pos}".rstrip()
            entries.append([f"{title}: {'; '.join(facts)}" if facts else title])
            if head.entry_id in verbs:
                entries[-1] += self._verb_lines(verbs.pop(head.entry_id))
        for verb in verbs.values():
            entries.append([f"- {verb.lemma} [{verb.entry_id}] VRB", *self._verb_lines(verb)])
        return entries

    @staticmethod
    def _verb_lines(verb: VerbTable) -> list[str]:
        lines = [
            f"  auxiliary {verb.aux}; past participle {verb.participle}",
            f"  present: {_persons(verb.present)}",
        ]
        for block in verb.tables["conjugations"]:
            past = block.get("indicative", {}).get("pastSimple")
            if past:
                lines.append("  past simple: " + _persons([past.get(f"p{person}", "") for person in range(1, 7)]))
                break
        imperative = [f"{cell} ({who})" for cell, who in ((verb.imp_sg, "du"), (verb.imp_pl, "dir")) if cell]
        if imperative:
            lines.append("  imperative: " + ", ".join(imperative))
        return lines
