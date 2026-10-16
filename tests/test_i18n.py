"""The interface strings.

The rule for strings: A string the page shows is a key in en.json and fr.json, applied with
data-i18n in HTML or t() in JS.
"""

import json
import re
from html.parser import HTMLParser

import pytest

from app.config import ROOT

STATIC = ROOT / "static"
TABLES = STATIC / "i18n"
LANGS = ("en", "fr")
VOID = {"meta", "link", "input", "br", "hr", "img", "source", "area", "base", "col", "embed", "track", "wbr"}
SKIPPED_TAGS = {"script", "style", "option"}
PLACEHOLDER = re.compile(r"\{(\w+)\}")


def collapse(text):
    return " ".join(text.split())


class PageStrings(HTMLParser):
    """The `data-i18n` elements of a page as {key: text}, the `data-i18n-attr` attributes as
    {key: value}, and the text with letters that no `data-i18n` element holds."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.texts = {}
        self.attrs = {}
        self.bare = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        for pair in filter(None, (values.get("data-i18n-attr") or "").split(";")):
            name, key = pair.split(":", 1)
            self.attrs[key] = collapse(values.get(name) or "")
        if tag in VOID:
            return
        parent_skips = bool(self.stack) and self.stack[-1]["skip"]
        skip = parent_skips or tag in SKIPPED_TAGS or values.get("lang") == "lb" or values.get("id") == "lang"
        self.stack.append({"tag": tag, "key": values.get("data-i18n"), "skip": skip, "text": []})

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        while self.stack:
            element = self.stack.pop()
            if element["key"]:
                assert element["key"] not in self.texts or self.texts[element["key"]] == collapse(
                    "".join(element["text"])
                ), f"{element['key']} is used for two different texts"
                self.texts[element["key"]] = collapse("".join(element["text"]))
            if element["tag"] == tag:
                break

    def handle_data(self, data):
        if not self.stack:
            return
        holders = [element for element in self.stack if element["key"]]
        for element in holders:
            element["text"].append(data)
        if not holders and not self.stack[-1]["skip"] and re.search(r"[^\W\d_]", data):
            self.bare.append(collapse(data))


def page_strings():
    parser = PageStrings()
    parser.feed((STATIC / "index.html").read_text(encoding="utf-8"))
    parser.close()
    return parser


REGEX_AFTER = set("(,=:[!&|?{};") | {"return"}
STRING = re.compile(r"\"(?:[^\"\\\n]|\\.)*\"|'(?:[^'\\\n]|\\.)*'")


def script_literals(source):
    """[(text, start, is_key)] for every string literal of the script, comments and regular
    expressions skipped; `text` is as written between the quotes, `is_key` whether the literal is
    the first argument of t( or tp(."""
    found = []
    position = 0
    previous = ""  # the last token that is not a space
    while position < len(source):
        char = source[position]
        if char.isspace():
            position += 1
        elif source.startswith("//", position):
            position = source.find("\n", position)
            position = len(source) if position < 0 else position
        elif source.startswith("/*", position):
            position = source.index("*/", position) + 2
        elif char in "\"'":
            match = STRING.match(source, position)
            assert match, f"unterminated string at {position}"
            before = source[max(0, position - 6) : position]
            found.append((match.group()[1:-1], position, re.search(r"\bt(?:p)?\(\s*$", before) is not None))
            position, previous = match.end(), '"'
        elif char == "/" and previous in REGEX_AFTER:
            end, in_class = position + 1, False
            while source[end] != "/" or in_class:
                if source[end] == "\\":
                    end += 1
                elif source[end] == "[":
                    in_class = True
                elif source[end] == "]":
                    in_class = False
                end += 1
            position, previous = end + 1, "/"
        elif char.isalpha() or char == "_" or char == "$":
            end = position
            while end < len(source) and (source[end].isalnum() or source[end] in "_$"):
                end += 1
            previous = source[position:end]
            position = end
        else:
            previous = char
            position += 1
    return found


@pytest.fixture(scope="module")
def tables():
    return {code: json.loads((TABLES / f"{code}.json").read_text(encoding="utf-8")) for code in LANGS}


@pytest.fixture(scope="module")
def page():
    return page_strings()


def placeholders(text):
    return sorted(PLACEHOLDER.findall(text))


def test_both_tables_have_the_same_keys_and_placeholders(tables):
    english, french = tables["en"], tables["fr"]

    assert sorted(english) == sorted(french)
    assert english["_note"] != french["_note"]
    assert french["_note"].startswith("Traduction automatique, non relue.")
    for key, text in english.items():
        for code in LANGS:
            value = tables[code][key]
            assert isinstance(value, str) and value.strip(), f"{code}.json {key}"
        assert placeholders(text) == placeholders(french[key]), key
    for key in english:
        if key.endswith("_one"):
            assert key[: -len("_one")] + "_other" in english, key
        if key.endswith("_other"):
            assert key[: -len("_other")] + "_one" in english, key


def test_every_key_of_the_page_is_in_both_tables(page, tables):
    for code in LANGS:
        missing = sorted((set(page.texts) | set(page.attrs)) - set(tables[code]))
        assert missing == [], f"{code}.json lacks {missing}"


def test_the_english_of_the_page_is_the_english_table(page, tables):
    for key, text in page.texts.items():
        assert text == collapse(tables["en"][key]), key
    for key, text in page.attrs.items():
        assert text == collapse(tables["en"][key]), key


def test_every_text_of_the_page_with_letters_is_a_key(page):
    """A string the page shows is a key in en.json and fr.json, applied with data-i18n in HTML or t() in JS."""
    assert page.bare == []


@pytest.fixture(scope="module")
def literals():
    return script_literals((STATIC / "app.js").read_text(encoding="utf-8"))


def key_forms(key):
    """The table keys a key of the script stands for: itself, or its _one and _other forms."""
    return [key, key + "_one", key + "_other"]


def test_every_key_the_script_asks_for_is_in_both_tables(literals, tables):
    asked = sorted({text for text, _, is_key in literals if is_key})

    assert asked, "the script calls t() somewhere"
    for key in asked:
        for code in LANGS:
            assert any(form in tables[code] for form in key_forms(key)), f"{code}.json lacks {key}"
    plural = [key for key in asked if key + "_one" in tables["en"]]
    for key in plural:
        assert key + "_other" in tables["fr"] and key + "_one" in tables["fr"]


def test_the_script_has_no_shown_string_outside_t(literals):
    """A string literal with a letter and a space in app.js outside t(...) is a line of tests/i18n_allow.txt."""
    allowed = {
        line.strip()
        for line in (ROOT / "tests" / "i18n_allow.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    stray = sorted(
        {
            text.strip()
            for text, _, is_key in literals
            if not is_key and re.search(r"[^\W\d_]", text) and " " in text and text.strip() not in allowed
        }
    )

    assert stray == [], "give these a key in en.json and fr.json and call t(): " + "; ".join(stray)
    # A line of the list that no literal needs any more is removed.
    used = {text.strip() for text, _, _ in literals}
    assert sorted(allowed - used) == []


def test_every_key_of_the_tables_is_used(page, literals, tables):
    used = set(page.texts) | set(page.attrs) | {text for text, _, _ in literals}
    dead = []
    for key in tables["en"]:
        if key == "_note":
            continue
        base = re.sub(r"_(?:one|other)$", "", key)
        if key not in used and base not in used:
            dead.append(key)

    assert dead == []


def test_the_lint_reads_strings_comments_and_expressions():
    source = (
        "// a comment with \"quotes\" and words\n"
        'const a = t("x.y"); const b = "two words"; /* "no no" */ const c = /"a b"/.test(d);\n'
        "const e = 'it\\'s here'; const f = tp( 'p.q', 2);\n"
    )

    assert script_literals(source) == [
        ("x.y", source.index('"x.y"'), True),
        ("two words", source.index('"two words"'), False),
        ("it\\'s here", source.index("'it"), False),
        ("p.q", source.index("'p.q'"), True),
    ]
