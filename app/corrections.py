"""The one label every correction gets, from stage-1 LOD checks and the acceptability encoder.

`label_correction` gives a tier (Verified, Likely, Unverified) or a downgrade (the fix is no
correction). The words LOD must list for Verified and Likely are the NEW words of the better
version: the words the learner's whole text does not hold. A word the learner said and the
tutor kept (a town, a brand) is no claim of the tutor's and is not checked.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.lod import checks

if TYPE_CHECKING:
    from app.engines.base import AcceptabilityEngine
    from app.lod.store import LodStore

log = logging.getLogger(__name__)

ACCEPTABLE = 0.9  # the encoder scores the learner's text at least this: the fix is no correction
OBJECTS = 0.5  # the encoder scores the better version below this: no Verified
TIERS = ("verified", "likely", "unverified")
SOURCES = ("typed", "heard-edited", "heard-unedited")
HEARD_UNEDITED = "heard-unedited"
_SENTENCE_GAP = re.compile(r"[.!?…]+[\"'“”«»‘’]*\s+")


@dataclass
class Label:
    tier: str  # one of TIERS, or "" when the fix is downgraded
    downgrade: str = ""  # "", "skipped words", "optional variant" or "encoder accepts"
    checks: list[str] = field(default_factory=list)  # one plain English line per fact used


def _starts(text: str) -> set[str]:
    """The case-folded words that open a sentence of `text`."""
    starts = set()
    for sentence in _SENTENCE_GAP.split(text):
        first = checks.WORD.search(sentence)
        if first:
            starts.update(part.casefold() for part in checks.split_clitic(first.group()))
    return starts


def _score(encoder: AcceptabilityEngine | None, text: str) -> float | None:
    if encoder is None:
        return None
    try:
        return encoder.score(text)
    except Exception:
        log.exception("the acceptability encoder failed on %r", text)
        return None


def _skipped_words(lod: LodStore, text: str, quote: str, old: list[str], new: list[str]) -> list[str] | None:
    """The check lines of rule 1 when every changed word of the quote is skipped, else None.

    A "name" the tutor replaced with a common LOD word was a foreign word, not a name.
    """
    said = {word.casefold() for word in checks.words(text)}
    starts = _starts(text) if said else set()
    as_written = {word.casefold(): word for word in reversed(checks.words(quote))}
    lines = []
    for folded in old:
        first = folded in starts if folded in said else folded in _starts(quote)
        reason = checks.skipped(as_written.get(folded, folded), lod, first)
        if reason is None:
            return None
        if reason == "name" and any(checks.common_word(lod, word) for word in new):
            return None
        lines.append(f"{as_written.get(folded, folded)} is a {reason}: not a learner error")
    return lines


def _variants(lod: LodStore, old: list[str], new: list[str]) -> list[str] | None:
    """The check lines of rule 2 when every changed word pairs with an optional variant, else None."""
    pairs = [(a, b) for a in old for b in new if checks.optional_variant(lod, a, b)]
    if not pairs or {a for a, _ in pairs} != set(old) or {b for _, b in pairs} != set(new):
        return None
    return [
        f"{a} and {b} differ only by the n-rule: LOD accepts both"
        if checks.n_only(lod, a, b)
        else f"{a} and {b} are both spellings LOD lists"
        for a, b in pairs
    ]


def label_correction(
    lod: LodStore | None,
    encoder: AcceptabilityEngine | None,
    text: str,
    quote: str,
    better_lb: str,
    source: str = "typed",
) -> Label:
    """The label of a "fix" (with `quote`) or an "english" (no quote) on the learner's `text`.

    `source` is one of SOURCES ("" counts as typed); a heard answer the learner did not edit
    tops out at Likely. Without a store every correction is Unverified and only the encoder
    can downgrade it.
    """
    old, changed = checks.changed_words(quote, better_lb) if quote else ([], [])
    said = {word.casefold() for word in checks.words(text)}
    new = list(dict.fromkeys(word.casefold() for word in checks.words(better_lb) if word.casefold() not in said))
    # A whole better sentence repeats words around the quote: those the learner said are not changed.
    changed = [word for word in changed if word not in said]

    if lod is not None and quote:
        if old:
            lines = _skipped_words(lod, text, quote, old, new)
            if lines is not None:
                return Label("", "skipped words", lines)
        if old and changed:
            lines = _variants(lod, old, changed)
            if lines is not None:
                return Label("", "optional variant", lines)

    text_score = _score(encoder, text)
    if text_score is not None and text_score >= ACCEPTABLE:
        return Label("", "encoder accepts", [f"the encoder scores the answer {text_score:.2f}: acceptable"])
    better_score = _score(encoder, better_lb)
    if text_score is None and better_score is None:
        encoder_lines = ["no encoder installed"]
    else:
        encoder_lines = [
            f"the encoder scores {what} {score:.2f}"
            for what, score in (("the answer", text_score), ("the better version", better_score))
            if score is not None
        ]
    objects = better_score is not None and better_score < OBJECTS
    agrees = text_score is not None and better_score is not None and better_score > text_score

    if lod is None:
        return Label("unverified", "", ["no LOD store: nothing is verified", *encoder_lines])

    lines = []
    unknown = [word for word in new if not checks.known(lod, word)]
    better_ok = not unknown
    if unknown:
        lines.append("not in LOD: " + ", ".join(unknown))
    elif new:
        lines.append("every word the tutor added is in LOD")
    else:
        lines.append("the tutor added no word")

    if not quote:
        lines.append("an English answer has no form of the learner's to compare: at most Likely")
        return Label("likely" if better_ok else "unverified", "", lines + encoder_lines)

    found = checks.swap(lod, quote, better_lb)
    agreed = False
    if found:
        a, b, lemma = found
        lines.insert(0, f"{a} and {b} are forms of {lemma} (LOD)")
        # Both forms may be correct (bräicht / brauch): Verified needs the subject to take one and not the other.
        pronoun = checks.subject_pronoun(text, quote, a)
        if pronoun:
            disagreement = checks.agreement(lod, pronoun, a, b, lemma)
        else:
            disagreement = f"no subject pronoun beside {a}: at most Likely"
        agreed = not disagreement
        lines.insert(1, f"{pronoun} takes {b}, not {a} (LOD)" if agreed else disagreement)
    else:
        lines.insert(0, "no same-lemma swap in LOD: at most Likely")
    if objects:
        lines.append("the encoder objects to the better version: not Verified")

    if agreed and better_ok and not objects:
        tier = "verified"
        n_pairs = [(a, b) for a in old for b in changed if checks.n_only(lod, a, b)]
        if n_pairs:
            tier = "likely"
            lines.append(f"{n_pairs[0][0]} and {n_pairs[0][1]} differ only by the n-rule: at most Likely")
        if source == HEARD_UNEDITED and tier == "verified":
            tier = "likely"
            lines.append("heard and not edited: at most Likely")
    elif better_ok or agrees:
        tier = "likely"
    else:
        tier = "unverified"
    return Label(tier, "", lines + encoder_lines)
