"""The LB-JUDGE set: LOD example sentences (correct arm) and single-error perturbations of them.

Every error row changes exactly one token for a LOD form that is not legal in its place, and
keeps the new token as `wrong` and the original as `gold`. Rows are dicts in the order of
ROW_FIELDS / ERROR_FIELDS; the set file is JSONL with a header line first.
"""

from __future__ import annotations

import datetime
import json
import random
import re
import sqlite3
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.lod import form_key
from app.lod.build import SIMPLE_TENSES, variants
from app.lod import checks

SET_NAME = "lb_judge"
REVIEW = "unreviewed"
MIN_WORDS = 3
MAX_WORDS = 10
CLASSES = ("verb_person", "auxiliary", "adjective_ending", "noun_gender")

Row = dict[str, str]

# Subject pronouns and the person (p1..p6 of a verb table) they take; "si" is p3 or p6.
PRONOUN_PERSON = {"ech": 1, "du": 2, "hien": 3, "hie": 3, "hatt": 3, "et": 3, "mir": 4, "dir": 5}
# Persons whose forms are also legal after the pronoun: expletive "et" takes a plural verb
# ("et ginn vill Leit").
ALSO_LEGAL = {"et": (6,)}
# The present forms of the two auxiliaries, paired by person; "hu" and "si" are n-rule forms.
_HUNN_SINN = {"hunn": "sinn", "hues": "bass", "huet": "ass", "hutt": "sidd", "hu": "si"}
AUX_SWAP = {**_HUNN_SINN, **{sinn: hunn for hunn, sinn in _HUNN_SINN.items()}}
AUX_LEMMA = {**dict.fromkeys(_HUNN_SINN, "hunn"), **dict.fromkeys(_HUNN_SINN.values(), "sinn")}
# "si" is an auxiliary only next to a subject that takes it ("mir si gaang", "haut si mir gaang").
_SI_BEFORE = frozenset({"mir", "si"})
_SI_AFTER = frozenset({"mir"})
MASCULINE_ARTICLES = frozenset({"de", "den", "en", "e"})
FEMININE_ARTICLE = "eng"
GENDERS = {"M": "masculine", "F": "feminine", "N": "neutral"}
ADJECTIVE = "ADJ"
NOUN = "SUBST"
DETERMINER_POS = frozenset({"ART", "PRON"})
# The n of the n-rule stays before a vowel and before d, h, n, t, z (ZLS 2019, ch. 6).
N_KEEPERS = frozenset(checks.VOWELS | set("dhntz"))
_CLAUSE_END = frozenset(",;:.!?")
_OPENING = ("(", "[", "„", "«")
_DASHES = frozenset({"-", "–", "—"})
_PARTS = re.compile(r"\A(\W*)(.*?)(\W*)\Z", re.DOTALL)


@dataclass(frozen=True)
class Token:
    text: str
    is_headword: bool = False


@dataclass(frozen=True)
class Example:
    id: str
    tokens: tuple[Token, ...]
    labelled: bool  # the example carries a register label (colloquial, ironic, ..)

    @property
    def words(self) -> list[str]:
        return [token.text for token in self.tokens if _has_alnum(token.text)]

    @property
    def text(self) -> str:
        return detokenise([token.text for token in self.tokens])


def _has_alnum(text: str) -> bool:
    return any(char.isalnum() for char in text)


def _text(element: ET.Element) -> str:
    return (element.text or "").strip()


def examples(xml_path: Path) -> Iterator[Example]:
    """Every <example> of a LOD article dump, in dump order, streamed."""
    for _, element in ET.iterparse(xml_path):
        if element.tag == "example":
            text = element.find("text")
            if text is not None:
                tokens = tuple(
                    Token(_text(child), child.tag == "inflectedHeadword")
                    for child in text
                    if child.tag in ("word", "inflectedHeadword") and _text(child)
                )
                yield Example(element.get("id", ""), tokens, text.find("attribute") is not None)
            element.clear()
        elif element.tag == "entry":
            element.clear()


def _is_clitic(token: str) -> bool:
    return token.casefold() in ("d'", "d’")


def detokenise(tokens: Sequence[str]) -> str:
    """Tokens as a sentence: punctuation on the word before it, d' on the word after it,
    the first letter upper-case and a final "." when the text ends in a letter or digit."""
    text = ""
    glue = False
    for token in tokens:
        closing = not _has_alnum(token) and not token.startswith(_OPENING) and token not in _DASHES
        text += token if not text or glue or closing else " " + token
        glue = _is_clitic(token) or (not _has_alnum(token) and token.startswith(_OPENING))
    for index, char in enumerate(text):
        if char.isalpha():
            text = text[:index] + char.upper() + text[index + 1 :]
            break
    if text and text[-1].isalnum():
        text += "."
    return text


def eligible(example: Example) -> bool:
    """Whether an example may be a row or a base: no register label, MIN_WORDS to MAX_WORDS
    words, no token with a digit."""
    return (
        not example.labelled
        and MIN_WORDS <= len(example.words) <= MAX_WORDS
        and not any(char.isdigit() for token in example.tokens for char in token.text)
    )


def _is_question_or_exclamation(text: str) -> bool:
    return text.rstrip("“”\"»)").endswith(("!", "?"))


def correct_rows(examples: Iterable[Example], n: int, rng: random.Random) -> list[Row]:
    """n correct rows drawn from the eligible examples: statements first, then exclamations and
    questions. Fewer when there are not enough."""
    pool = [example for example in examples if eligible(example)]
    statements = [example for example in pool if not _is_question_or_exclamation(example.text)]
    rest = [example for example in pool if _is_question_or_exclamation(example.text)]
    picked = rng.sample(statements, min(n, len(statements)))
    picked += rng.sample(rest, min(n - len(picked), len(rest)))
    return [
        {
            "id": f"c-{k}",
            "text": example.text,
            "label": "correct",
            "source": "lod_example",
            "base_id": example.id,
            "error_type": "",
            "review": REVIEW,
        }
        for k, example in enumerate(picked, 1)
    ]


def _single_words(cell: str) -> list[str]:
    return [form for form in variants(cell) if form and " " not in form]


class LodFacts:
    """What the perturbations need from the LOD store, read once into dicts. Forms are form_key()s."""

    def __init__(self, store_path: Path) -> None:
        if not Path(store_path).is_file():
            raise FileNotFoundError(f"{store_path} is missing; run scripts/download_lod.py")
        db = sqlite3.connect(Path(store_path).resolve().as_uri() + "?mode=ro", uri=True)
        try:
            self._read_verbs(db)
            self._read_adjectives(db)
            self._read_headwords(db)
            self._read_wordforms(db)
        finally:
            db.close()

    def _read_verbs(self, db: sqlite3.Connection) -> None:
        # tables[i][p - 1]: the single-word present forms of person p of verb table i.
        self.tables: list[tuple[tuple[str, ...], ...]] = []
        self.present: dict[str, list[tuple[int, int]]] = {}  # form -> (table, person)
        self.person_forms: dict[int, set[str]] = {person: set() for person in range(1, 7)}
        self.participle_aux: dict[str, list[frozenset[str]]] = {}
        self.verb_lemmas: set[str] = set()
        rows = db.execute("SELECT lemma, aux, participle, p1, p2, p3, p4, p5, p6, tables_json FROM verbs")
        for lemma, aux, participle, *cells, tables_json in rows:
            self.verb_lemmas.add(form_key(lemma))
            table = tuple(tuple(_single_words(cell)) for cell in cells)
            for person, forms in enumerate(table, 1):
                for form in forms:
                    self.present.setdefault(form, []).append((len(self.tables), person))
            self.tables.append(table)
            auxiliaries = frozenset(part.strip() for part in aux.split("/") if part.strip())
            for form in _single_words(participle):
                self.participle_aux.setdefault(form, []).append(auxiliaries)
            # Every simple-tense cell of a person is legal for it, the finite word of a separable form too.
            person_cells = list(enumerate(cells, 1))
            for block in json.loads(tables_json or "{}").get("conjugations", []):
                for mood, tense in SIMPLE_TENSES:
                    tense_cells = (block.get(mood) or {}).get(tense)
                    if isinstance(tense_cells, dict):
                        person_cells += [(person, tense_cells.get(f"p{person}")) for person in range(1, 7)]
            for person, cell in person_cells:
                if isinstance(cell, str):
                    self.person_forms[person].update(form.split()[0] for form in variants(cell) if form)

    def _read_adjectives(self, db: sqlite3.Connection) -> None:
        # adjectives[i] = (lemma, {(gender, case): forms}) of the positive singular; plural cells
        # count only for `adjective_tables`.
        self.adjectives: list[tuple[str, dict[tuple[str, str], tuple[str, ...]]]] = []
        self.adjective_tables: dict[str, set[int]] = {}  # positive-declension form -> tables
        for lemma, forms_json in db.execute("SELECT lemma, forms_json FROM adjectives"):
            positive = json.loads(forms_json).get("declension", {}).get("positive", {})
            if not isinstance(positive, dict):
                continue
            index = len(self.adjectives)
            singular: dict[tuple[str, str], tuple[str, ...]] = {}
            for number, genders in positive.items():
                if not isinstance(genders, dict):
                    continue
                for gender, cases in genders.items():
                    if not isinstance(cases, dict):
                        continue
                    for case, cell in cases.items():
                        if not isinstance(cell, str):
                            continue
                        forms = tuple(_single_words(cell))
                        if number == "singular":
                            singular[(gender, case)] = forms
                        for form in forms:
                            self.adjective_tables.setdefault(form, set()).add(index)
            self.adjectives.append((lemma, singular))

    def _read_headwords(self, db: sqlite3.Connection) -> None:
        self.noun_genders: dict[str, set[str]] = {}  # noun lemma as LOD spells it -> genders
        self.adjective_lemmas: set[str] = set()
        self.determiners: set[str] = set()  # article and pronoun lemmas
        for lemma, pos, gender in db.execute("SELECT lemma, pos, gender FROM headwords"):
            if pos == NOUN:
                self.noun_genders.setdefault(lemma, set()).add(gender)
            elif pos == ADJECTIVE:
                self.adjective_lemmas.add(form_key(lemma))
            elif pos in DETERMINER_POS:
                self.determiners.add(form_key(lemma))

    def _read_wordforms(self, db: sqlite3.Connection) -> None:
        self.form_kinds: dict[str, set[str]] = {}
        self.form_lemmas: dict[str, set[str]] = {}
        for form, lemma, kind in db.execute("SELECT form, lemma, kind FROM wordforms"):
            self.form_kinds.setdefault(form, set()).add(kind)
            self.form_lemmas.setdefault(form, set()).add(form_key(lemma))

    def lookup_form(self, form: str) -> list[str]:
        """The case-folded lemmas of `form`, as LodStore.lookup_form gives them, for app.lod.checks."""
        return sorted(self.form_lemmas.get(form_key(form), ()))

    def knows_all(self, example: Example) -> bool:
        """Whether LOD knows every word of the example, with the fallbacks of app.lod.checks.known."""
        return all(checks.known(self, word) for word in checks.words(example.text))

    def noun_gender(self, word: str) -> str:
        """"M", "F" or "N" when `word` is spelled as a noun lemma whose entries all have that one gender."""
        genders = self.noun_genders.get(word, set())
        return next(iter(genders)) if len(genders) == 1 and genders <= set(GENDERS) else ""


def _parts(token: str) -> tuple[str, str, str]:
    """(leading punctuation, the word, trailing punctuation) of a token."""
    match = _PARTS.match(token)
    return match.group(1), match.group(2), match.group(3)


def _word(token: Token) -> str:
    return _parts(token.text)[1]


def _cased(new: str, old: str) -> str:
    return new[:1].upper() + new[1:] if old[:1].isupper() else new


def _keeps_n(next_word: str) -> bool:
    return not next_word[:1].isalpha() or next_word[:1].casefold() in N_KEEPERS


def _n_pair(a: str, b: str) -> bool:
    short, long = sorted((a, b), key=len)
    return long in (short + "n", short + "nn")


def _legal(form: str, legal: set[str]) -> bool:
    """Whether `form`, or a form differing from it only by a final n or nn, is in `legal`."""
    near = {form, form + "n", form + "nn"}
    near |= {form[:-1]} if form.endswith("n") else set()
    near |= {form[:-2]} if form.endswith("nn") else set()
    return bool(near & legal)


def _row(example: Example, index: int, new: str, error_type: str) -> Row | None:
    """The error row of `example` with token `index` replaced by `new`; None when nothing changes."""
    tokens = [token.text for token in example.tokens]
    lead, old, trail = _parts(tokens[index])
    new = _cased(new, old)
    if new.casefold() == old.casefold():
        return None
    tokens[index] = lead + new + trail
    return {
        "id": "",
        "text": detokenise(tokens),
        "label": "error",
        "source": "perturbation",
        "base_id": example.id,
        "error_type": error_type,
        "wrong": new,
        "gold": old,
        "review": REVIEW,
    }


def _bare(token: Token) -> bool:
    """A single word with no punctuation around it."""
    lead, word, trail = _parts(token.text)
    return bool(word) and not lead and not trail and " " not in word


def _clause_ends(token: Token) -> bool:
    lead, word, trail = _parts(token.text)
    return bool(set(trail) & _CLAUSE_END) or not word


def verb_person(example: Example, facts: LodFacts, rng: random.Random) -> Row | None:
    """A subject pronoun directly before a present form of its person: the form becomes another
    person's form of the same verb table that no verb lists for the pronoun's person."""
    tokens = example.tokens
    slots = []
    for index in range(1, len(tokens)):
        pronoun, verb = tokens[index - 1], tokens[index]
        subject = _word(pronoun).casefold()
        person = PRONOUN_PERSON.get(subject)
        if person is None or not _bare(pronoun) or " " in _word(verb):
            continue
        form = form_key(_word(verb))
        legal = set().union(*(facts.person_forms[each] for each in (person, *ALSO_LEGAL.get(subject, ()))))
        for table, cell_person in facts.present.get(form, []):
            if cell_person != person:
                continue
            others = {
                other
                for other_person, forms in enumerate(facts.tables[table], 1)
                if other_person != person
                for other in forms
                if other != form and not _legal(other, legal)
            }
            slots += [(index, other) for other in sorted(others)]
    if not slots:
        return None
    index, new = rng.choice(sorted(set(slots)))
    return _row(example, index, new, "verb_person")


def _is_aux(tokens: tuple[Token, ...], index: int) -> bool:
    word = _word(tokens[index]).casefold()
    if word not in AUX_SWAP or " " in word:
        return False
    if word != "si":
        return True
    before = _word(tokens[index - 1]).casefold() if index > 0 else ""
    after = _word(tokens[index + 1]).casefold() if index + 1 < len(tokens) else ""
    return before in _SI_BEFORE or after in _SI_AFTER


def _participle_of_only(facts: LodFacts, word: str, aux: str) -> bool:
    """Whether `word` is a participle whose every verb entry takes `aux` alone, and no adjective,
    present form or infinitive."""
    form = form_key(word)
    entries = facts.participle_aux.get(form)
    if not entries or any(entry != {aux} for entry in entries):
        return False
    kinds = facts.form_kinds.get(form, set())
    return "adj_form" not in kinds and form not in facts.present and form not in facts.verb_lemmas


def auxiliary(example: Example, facts: LodFacts, rng: random.Random) -> Row | None:
    """A form of hunn or sinn and, later in its clause, a participle whose every verb takes that
    auxiliary alone: the auxiliary becomes the same person of the other one."""
    tokens = example.tokens
    slots = []
    for index in range(len(tokens)):
        if not _is_aux(tokens, index):
            continue
        aux = AUX_LEMMA[_word(tokens[index]).casefold()]
        for later in range(index + 1, len(tokens)):
            if _clause_ends(tokens[later - 1]) or _is_aux(tokens, later):
                break
            if _participle_of_only(facts, _word(tokens[later]), aux):
                slots.append(index)
                break
    if not slots:
        return None
    index = rng.choice(slots)
    return _row(example, index, AUX_SWAP[_word(tokens[index]).casefold()], "auxiliary")


def _adjective_table(facts: LodFacts, form: str) -> int | None:
    """The one adjective table that has `form` (or `form` + n) as a positive cell, when that
    table's lemma is an adjective and neither it nor the form is an article or pronoun."""
    if facts.form_lemmas.get(form, set()) & facts.determiners:
        return None
    tables = facts.adjective_tables.get(form) or facts.adjective_tables.get(form + "n") or set()
    if len(tables) != 1:
        return None
    (table,) = tables
    lemma = form_key(facts.adjectives[table][0])
    return table if lemma in facts.adjective_lemmas and lemma not in facts.determiners else None


def adjective_ending(example: Example, facts: LodFacts, rng: random.Random) -> Row | None:
    """An adjective directly before a noun of one gender: the adjective becomes the cell of the
    same case for another gender, when that spelling is not one of the noun's gender."""
    tokens = example.tokens
    slots = []
    for index in range(len(tokens) - 1):
        adjective, noun = tokens[index], tokens[index + 1]
        word = _word(adjective)
        if not _bare(adjective) or not word[:1].islower():
            continue
        gender = facts.noun_gender(_word(noun)) if _word(noun)[:1].isupper() else ""
        if not gender:
            continue
        form = form_key(word)
        table = _adjective_table(facts, form)
        if table is None:
            continue
        cells = facts.adjectives[table][1]
        own = GENDERS[gender]
        own_forms = {cell for (cell_gender, _), forms in cells.items() if cell_gender == own for cell in forms}
        cases = sorted(
            case for (cell_gender, case), forms in cells.items() if cell_gender == own and _legal(form, set(forms))
        )
        for case in cases:
            for other in sorted(set(GENDERS.values()) - {own}):
                for new in cells.get((other, case), ()):
                    if new.endswith("en") and not _keeps_n(_word(noun)):
                        new = new[:-1]
                    if new != form and not _n_pair(new, form) and not _legal(new, own_forms):
                        slots.append((index, new))
    if not slots:
        return None
    index, new = rng.choice(sorted(set(slots)))
    return _row(example, index, new, "adjective_ending")


def noun_gender(example: Example, facts: LodFacts, rng: random.Random) -> Row | None:
    """de / den / en / e before a masculine noun becomes eng; eng before a feminine noun becomes
    en or e by the n-rule. Neuter nouns, nouns of several genders and d' are left alone."""
    tokens = example.tokens
    slots = []
    for index in range(len(tokens) - 1):
        article, noun = tokens[index], tokens[index + 1]
        word, noun_word = _word(article).casefold(), _word(noun)
        if not _bare(article) or not noun_word[:1].isupper():
            continue
        gender = facts.noun_gender(noun_word)
        if word in MASCULINE_ARTICLES and gender == "M":
            slots.append((index, FEMININE_ARTICLE))
        elif word == FEMININE_ARTICLE and gender == "F":
            slots.append((index, "en" if _keeps_n(noun_word) else "e"))
    if not slots:
        return None
    index, new = rng.choice(slots)
    return _row(example, index, new, "noun_gender")


Perturbation = Callable[[Example, LodFacts, random.Random], Row | None]
PERTURBATIONS: dict[str, Perturbation] = {
    "verb_person": verb_person,
    "auxiliary": auxiliary,
    "adjective_ending": adjective_ending,
    "noun_gender": noun_gender,
}


def error_rows(examples: Iterable[Example], facts: LodFacts, n: int, rng: random.Random) -> list[Row]:
    """n error rows, n // 4 per class, each from a distinct eligible example; a class short of
    its share is made up by the others in turn. Fewer when the examples run out."""
    pool = [example for example in examples if eligible(example)]
    order = rng.sample(pool, len(pool))
    used: set[str] = set()

    def draws(error_type: str) -> Iterator[Row]:
        for example in order:
            if example.id not in used:
                row = PERTURBATIONS[error_type](example, facts, rng)
                if row is not None:
                    yield row

    sources = {error_type: draws(error_type) for error_type in CLASSES}
    rows: dict[str, list[Row]] = {error_type: [] for error_type in CLASSES}

    def take(error_type: str) -> bool:
        row = next(sources[error_type], None)
        if row is None:
            return False
        used.add(row["base_id"])
        rows[error_type].append(row)
        return True

    for error_type in CLASSES:
        for _ in range(n // 4):
            if not take(error_type):
                break
    live = list(CLASSES)
    while live and sum(map(len, rows.values())) < n:
        for error_type in list(live):
            if sum(map(len, rows.values())) == n:
                break
            if not take(error_type):
                live.remove(error_type)
    flat = [row for error_type in CLASSES for row in rows[error_type]]
    for k, row in enumerate(flat, 1):
        row["id"] = f"e-{k}"
    return flat


def header(seed: int, lod_dump: str, built: datetime.date | None = None) -> dict:
    return {
        "set": SET_NAME,
        "built": (built or datetime.date.today()).isoformat(),
        "lod_dump": lod_dump,
        "seed": seed,
        "review": REVIEW,
        "reviewed_by": "none",
    }


def write_set(rows: Iterable[Row], path: Path, seed: int, lod_dump: str) -> None:
    """The set as JSONL (UTF-8, header line first), written to a temp file and moved into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [header(seed, lod_dump), *rows]
    temp = path.with_name(path.name + ".tmp")
    temp.write_text("".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines), encoding="utf-8")
    temp.replace(path)


def read_set(path: Path) -> list[Row]:
    """The rows of a set file, without its header."""
    lines = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [line for line in lines if line.get("set") != SET_NAME]


def build_set(dump: Path, facts: LodFacts, n: int, seed: int) -> tuple[list[Row], list[Row]]:
    """(correct rows, error rows), n of each when the dump has enough, drawn with one seeded rng
    from the examples whose every word LOD knows."""
    rng = random.Random(seed)
    pool = [example for example in examples(dump) if facts.knows_all(example)]
    return correct_rows(pool, n, rng), error_rows(pool, facts, n, rng)
