"""Stage-1 checks of a correction on the LOD store: which words LOD knows, which it says nothing
about, and whether a fix swaps two forms of one lemma.

Every function takes `lod: LodStore | None`; without a store nothing is known. Words are
case-folded for matching; an apostrophe ’ counts as '.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.lod.store import LodStore

# A word; an apostrophe or a hyphen inside it does not end it ("d'Aarbecht").
WORD = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*")
_LEADING_CLITIC = re.compile(r"\A(d')(?=[^\W_])", re.IGNORECASE)
_TRAILING_CLITIC = re.compile(r"(?<=[^\W_])('[tsn])\Z", re.IGNORECASE)
CLITICS = frozenset({"d'", "'t", "'s", "'n"})
FILLERS = frozenset({"ëëë", "äh", "ähm", "hm", "mh", "mmh", "euh", "eh"})
VOWELS = frozenset("aeiouyäëéèêöüâîôû")
# A run of one vowel letter this long or longer is a filler ("aaa"); "ee" is a word.
FILLER_RUN = 3
_DOUBLED_VOWEL = re.compile(r"([aeiouäëéöü])\1")
PROPER_NOUN = "NP"  # the LOD part of speech of names and places
# Subject pronouns, case-folded, with the persons (p1..p6) of a LOD verb table they take; "si" is she or they.
PRONOUNS = {
    "ech": (1,),
    "du": (2,),
    "hien": (3,),
    "hie": (3,),
    "hatt": (3,),
    "et": (3,),
    "si": (3, 6),
    "mir": (4,),
    "dir": (5,),
}
_PERSON_CELL = re.compile(r"\Ap[1-6]\Z")


def _plain(word: str) -> str:
    return word.replace("’", "'")


def split_clitic(word: str) -> list[str]:
    """`word` with a leading d' and a trailing 't, 's or 'n split off: "d'Fra" -> ["d'", "Fra"]."""
    word = _plain(word)
    parts = []
    head = _LEADING_CLITIC.match(word)
    if head:
        parts.append(head.group(1))
        word = word[head.end() :]
    tail = _TRAILING_CLITIC.search(word)
    if tail:
        return [*parts, word[: tail.start()], tail.group(1)]
    return [*parts, word]


def words(text: str) -> list[str]:
    """The words of `text` as written, clitics split off."""
    return [part for word in WORD.findall(text) for part in split_clitic(word)]


def _lookup(lod: LodStore, word: str) -> set[str]:
    return {lemma.casefold() for lemma in lod.lookup_form(_plain(word))}


def lemmas(lod: LodStore | None, word: str) -> set[str]:
    """The case-folded lemmas of `word`: as written, else without its clitic, else with an "n" added."""
    if lod is None:
        return set()
    found = _lookup(lod, word)
    if found:
        return found
    core = [part for part in split_clitic(word) if part.casefold() not in CLITICS]
    for candidate in dict.fromkeys(core[:1] + [word]):
        found = _lookup(lod, candidate) or _lookup(lod, candidate + "n")
        if found:
            return found
    return set()


def known(lod: LodStore | None, word: str) -> bool:
    """Whether LOD lists `word`, with the fallbacks of `lemmas`; a clitic on its own counts as known."""
    if lod is None:
        return False
    return _plain(word).casefold() in CLITICS or bool(lemmas(lod, word))


def all_known(lod: LodStore | None, words: list[str]) -> bool:
    return all(known(lod, word) for word in words)


def common_word(lod: LodStore | None, word: str) -> bool:
    """Whether `word` is a form of a LOD lemma that is not a proper noun."""
    if lod is None:
        return False
    for lemma in lemmas(lod, word):
        head = lod.headword(lemma)
        if head is None or head.pos != PROPER_NOUN:
            return True
    return False


def skipped(word: str, lod: LodStore | None, first: bool = False) -> str | None:
    """Why stage 1 says nothing about `word`: "filler", "number" or "name"; None when it may judge it.

    A name starts upper-case, is not the `first` word of a sentence and is unknown to LOD.
    """
    folded = _plain(word).casefold()
    if folded in FILLERS or (len(folded) >= FILLER_RUN and len(set(folded)) == 1 and folded[0] in VOWELS):
        return "filler"
    if any(char.isdigit() for char in folded):
        return "number"
    if word[:1].isupper() and not first and not known(lod, word):
        return "name"
    return None


def changed_words(quote: str, better: str) -> tuple[list[str], list[str]]:
    """(the words of `quote` not in `better`, the words of `better` not in `quote`), case-folded, in order."""
    old = [word.casefold() for word in words(quote)]
    new = [word.casefold() for word in words(better)]
    return (
        list(dict.fromkeys(word for word in old if word not in new)),
        list(dict.fromkeys(word for word in new if word not in old)),
    )


def n_only(lod: LodStore | None, a: str, b: str) -> bool:
    """Whether `a` and `b` differ only by a final "n" or "nn" (the n-rule, optional in places)."""
    a, b = _plain(a).casefold(), _plain(b).casefold()
    short, long = sorted((a, b), key=len)
    return bool(short) and long in (short + "n", short + "nn")


def optional_variant(lod: LodStore | None, a: str, b: str) -> bool:
    """Whether `a` and `b` are both accepted: an n-rule pair, or two LOD lemmas that differ only by
    doubled vowels (Problem / Probleem)."""
    if n_only(lod, a, b):
        return True
    if lod is None or a.casefold() == b.casefold():
        return False
    if _DOUBLED_VOWEL.sub(r"\1", a.casefold()) != _DOUBLED_VOWEL.sub(r"\1", b.casefold()):
        return False
    return all(word.casefold() in _lookup(lod, word) for word in (a, b))


def swap(lod: LodStore | None, quote: str, better: str) -> tuple[str, str, str] | None:
    """(quoted word, better word, their lemma) of the first changed pair that shares a LOD lemma.

    An n-rule pair is no evidence: "de" and "den" are both forms of "den".
    """
    old, new = changed_words(quote, better)
    for a in old:
        for b in new:
            if n_only(lod, a, b):
                continue
            shared = lemmas(lod, a) & lemmas(lod, b)
            if shared:
                return a, b, sorted(shared)[0]
    return None


def _add_finite(forms: set[str], cell: str) -> None:
    """The first word of each variant of a LOD cell: the finite verb ("hu gebraucht" -> "hu")."""
    for variant in cell.split("/"):
        first = WORD.search(variant)
        if first:
            forms.add(first.group().casefold())


def person_forms(lod: LodStore | None, lemma: str) -> dict[int, set[str]]:
    """The finite forms LOD lists for each person (1..6) of the verb `lemma`, in every tense and
    mood it tables; {} when `lemma` has no verb table."""
    if lod is None:
        return {}
    table = lod.conjugation(lemma)
    if table is None:
        return {}
    cells: dict[int, set[str]] = {person: set() for person in range(1, 7)}
    for person, cell in enumerate(table.present, 1):
        _add_finite(cells[person], cell)

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if node and all(_PERSON_CELL.match(key) for key in node):
                for key, cell in node.items():
                    _add_finite(cells[int(key[1])], str(cell))
            else:
                for value in node.values():
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(table.tables)
    return cells


def subject_pronoun(text: str, quote: str, form: str) -> str:
    """The case-folded subject pronoun of the quoted verb `form`: the first pronoun of `quote`, else
    the word right before `form` in `text`, else the word right after it ("Haut wunnt ech"); ""."""
    quoted = [word.casefold() for word in words(quote)]
    for word in quoted:
        if word in PRONOUNS:
            return word
    said = [word.casefold() for word in words(text)]
    form = form.casefold()
    for at in (index for index, word in enumerate(said) if word == form):
        for neighbour in (at - 1, at + 1):
            if 0 <= neighbour < len(said) and said[neighbour] in PRONOUNS:
                return said[neighbour]
    return ""


def agreement(lod: LodStore | None, pronoun: str, quoted: str, better: str, lemma: str) -> str:
    """Why LOD's verb tables do or do not show that `pronoun` takes `better` and not `quoted`: ""
    when they do, else the reason the swap is at most Likely.

    Every lemma the two forms share is read (wëllt / wëllen are forms of the adjective wëll and of
    the verb wëllen); `lemma` names the swap in the reason.
    """
    cells: dict[int, set[str]] = {}
    for shared in sorted(lemmas(lod, quoted) & lemmas(lod, better)):
        for person, forms in person_forms(lod, shared).items():
            cells.setdefault(person, set()).update(forms)
    if not cells:
        return f"{lemma} has no verb table in LOD: at most Likely"
    persons = PRONOUNS[pronoun]
    if not any(better.casefold() in cells[person] for person in persons):
        return f"LOD lists no {better} for {pronoun}: at most Likely"
    if any(quoted.casefold() in cells[person] for person in persons):
        return f"{quoted} also fits {pronoun} in LOD: at most Likely"
    return ""


def same_lemma(lod: LodStore | None, quote: str, better: str) -> str:
    """The lemma a changed word of `quote` and a new word of `better` share; "" when none does."""
    found = swap(lod, quote, better)
    return found[2] if found else ""
