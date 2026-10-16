"""LOD (Lëtzebuerger Online Dictionnaire, CC0) grammar store.

build.py parses the two XML dumps into lod.sqlite, store.py reads it. SCHEMA.md lists the
XML names and the tables. The app makes no LOD network calls.
"""

from __future__ import annotations

import re

_NOT_PART_OF_THE_FORM = re.compile(r"\([^)]*\)|!")


def form_key(text: str) -> str:
    """The lookup key of a form, as stored in wordforms.form.

    LOD prints an optional reflexive pronoun in parentheses and "!" after imperatives
    ("maach (dech)!"); both are dropped, spaces are collapsed and case is folded.
    """
    return " ".join(_NOT_PART_OF_THE_FORM.sub(" ", text).split()).casefold()
