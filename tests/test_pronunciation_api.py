"""The pronunciation flags and the saved respellings: the store, the /api/pronunciation routes, the files."""

import json
import logging

import pytest

from app.engines.tts_piper import Pronunciations
from app.pronunciation_store import MAX_FLAGS, MAX_TEXT, FlagStore, TableFileError, save_pronunciation

LINE = "Moien, ech si Stefan an ech wunnen zu Lëtzebuerg."


@pytest.fixture
def state(client):
    return client.app.state


@pytest.fixture
def flags_path(settings):
    return settings.data_dir / "pronunciation_flags.json"


@pytest.fixture
def table_path(settings):
    return settings.data_dir / "pronunciations.json"


def flag(client, word="Stefan", lang="lb", text=LINE):
    return client.post("/api/pronunciation/flags", json={"lang": lang, "word": word, "text": text})


def file_flags(path):
    return json.loads(path.read_text(encoding="utf-8"))["flags"]


# --- flag, list, dismiss ---


def test_a_flag_is_kept_with_its_line_and_listed(client, flags_path):
    response = flag(client)

    assert response.status_code == 200
    made = response.json()
    assert set(made) == {"id", "lang", "word", "text", "flagged_at", "phonemes", "count"}
    assert (made["lang"], made["word"], made["text"], made["count"]) == ("lb", "Stefan", LINE, 1)
    assert made["phonemes"] == ""  # the fake engine has no phonemes
    assert len(made["id"]) == 8 and made["flagged_at"].endswith("+00:00")
    assert file_flags(flags_path) == [made]
    assert not flags_path.with_name(flags_path.name + ".tmp").exists()
    assert not flags_path.read_bytes().startswith(b"\xef\xbb\xbf")

    listed = client.get("/api/pronunciation/flags").json()
    assert listed == {"flags": [{**made, "respelling": ""}]}


def test_the_phonemes_come_from_the_engine_when_the_flag_is_made(client, state, monkeypatch):
    calls = []

    def phonemes(lang, text):
        calls.append((lang, text))
        return "ʃtɛfan"

    monkeypatch.setattr(state.tts, "phonemes", phonemes, raising=False)

    made = flag(client, "Stefan", "lb").json()
    english = flag(client, "Hello", "en", "Hello there").json()
    other = flag(client, "Bonjour", "fr", "Bonjour tout le monde").json()

    assert made["phonemes"] == english["phonemes"] == other["phonemes"] == "ʃtɛfan"
    assert calls == [("lb", "Stefan"), ("en", "Hello"), ("en", "Bonjour")]
    # They are stored, not asked again when the list is read.
    monkeypatch.setattr(state.tts, "phonemes", lambda lang, text: 1 / 0, raising=False)
    assert {item["phonemes"] for item in client.get("/api/pronunciation/flags").json()["flags"]} == {"ʃtɛfan"}


def test_an_engine_that_fails_to_make_phonemes_still_flags_the_word(client, state, monkeypatch, caplog):
    monkeypatch.setattr(state.tts, "phonemes", lambda lang, text: 1 / 0, raising=False)

    with caplog.at_level(logging.WARNING):
        response = flag(client)

    assert response.status_code == 200 and response.json()["phonemes"] == ""
    assert "ZeroDivisionError" in caplog.text


def test_a_second_flag_of_a_word_refreshes_the_entry_and_counts_again(client, state, monkeypatch, flags_path):
    first = flag(client, "Stefan", "lb", "Moien, Stefan!").json()
    other_language = flag(client, "Stefan", "en", "Hello, Stefan.").json()
    monkeypatch.setattr(state.tts, "phonemes", lambda lang, text: "new", raising=False)

    again = flag(client, "STEFAN", "lb", "  Gudde Moien, Stefan.  ").json()

    assert again["id"] == first["id"] and other_language["id"] != first["id"]
    assert (again["count"], again["text"], again["phonemes"], again["word"]) == (
        2,
        "Gudde Moien, Stefan.",
        "new",
        "Stefan",
    )
    assert again["flagged_at"] >= first["flagged_at"]
    stored = file_flags(flags_path)
    assert [(item["lang"], item["count"]) for item in stored] == [("en", 1), ("lb", 2)]
    assert [item["id"] for item in client.get("/api/pronunciation/flags").json()["flags"]] == [
        first["id"],
        other_language["id"],
    ]


def test_any_language_but_lb_is_english(client):
    assert flag(client, "Bonjour", "fr").json()["lang"] == "en"
    assert flag(client, "Bonjour", "en").json()["count"] == 2


def test_the_line_is_kept_up_to_500_characters(client):
    made = flag(client, text="Stefan " + "x" * 600).json()

    assert len(made["text"]) == MAX_TEXT


def test_the_list_shows_the_respelling_the_table_has_now(client, table_path):
    flag(client, "Luxembourgish", "en", "Say Luxembourgish.")
    flag(client, "Stefan", "lb")
    flag(client, "Moien", "lb")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(json.dumps({"lb": {"STEFAN": "[[ ʃtɛfan ]]", "moien": ""}}), encoding="utf-8")

    listed = client.get("/api/pronunciation/flags").json()["flags"]

    assert [(item["word"], item["respelling"]) for item in listed] == [
        ("Moien", ""),
        ("Stefan", "[[ ʃtɛfan ]]"),
        ("Luxembourgish", "Luxembourghish"),
    ]


def test_a_flag_is_dismissed_once(client, flags_path):
    first, second = flag(client, "Stefan").json(), flag(client, "Marc").json()

    response = client.delete(f"/api/pronunciation/flags/{first['id']}")

    assert (response.status_code, response.json()) == (200, {"dismissed": first["id"]})
    assert [item["id"] for item in file_flags(flags_path)] == [second["id"]]
    again = client.delete(f"/api/pronunciation/flags/{first['id']}")
    assert again.status_code == 404
    assert set(again.json()) == {"error", "fix"}
    assert client.delete("/api/pronunciation/flags/nothing").status_code == 404


def test_there_are_at_most_200_flags_and_the_oldest_goes_first(flags_path):
    store = FlagStore(flags_path)
    words = [f"w{chr(97 + n // 26)}{chr(97 + n % 26)}" for n in range(MAX_FLAGS + 5)]

    for word in words:
        store.flag("lb", word, "line")
    store.flag("lb", words[5], "line again")  # refreshed: now the newest, not the oldest

    kept = [item["word"] for item in store.list()]
    assert len(kept) == MAX_FLAGS
    assert kept[0] == words[5]
    assert words[0] not in kept and words[4] not in kept and words[6] in kept


def test_a_flags_file_that_cannot_be_read_is_renamed_and_the_list_starts_empty(client, flags_path, caplog):
    flags_path.parent.mkdir(parents=True, exist_ok=True)
    flags_path.write_text('{"flags": [{"id": 1}', encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        listed = client.get("/api/pronunciation/flags")

    assert listed.json() == {"flags": []}
    assert not flags_path.exists()
    assert flags_path.with_name(flags_path.name + ".corrupt").read_text(encoding="utf-8").startswith('{"flags"')
    assert "pronunciation_flags.json" in caplog.text and ".corrupt" in caplog.text
    assert flag(client).json()["count"] == 1
    assert len(file_flags(flags_path)) == 1


@pytest.mark.parametrize("content", ["[]", '{"flags": 3}', '{"flags": ["x"]}', '{"flags": [{"id": "a"}]}', "{}"])
def test_a_flags_file_of_another_shape_is_renamed_too(client, flags_path, content):
    flags_path.parent.mkdir(parents=True, exist_ok=True)
    flags_path.write_text(content, encoding="utf-8")

    assert client.get("/api/pronunciation/flags").json() == {"flags": []}
    assert flags_path.with_name(flags_path.name + ".corrupt").is_file()


# --- the rules ---


@pytest.mark.parametrize("word", ["a", "d'", "3", "x y", "", "ab3", "x" * 41, "[[ a ]]"])
def test_a_flag_needs_one_word_of_letters(client, flags_path, word):
    response = flag(client, word)

    assert response.status_code == 400
    body = response.json()
    assert set(body) == {"error", "fix"} and body["error"].startswith("bad word") and body["fix"]
    assert not flags_path.exists()


@pytest.mark.parametrize("body", [{}, {"lang": "lb", "word": "Stefan"}, {"lang": "lb", "word": 3, "text": "x"}])
def test_a_flag_with_missing_fields_is_a_422(client, body):
    assert client.post("/api/pronunciation/flags", json=body).status_code == 422


# --- save ---


def save(client, word="Stefan", respelling="[[ ʃtɛfan ]]", lang="lb"):
    return client.post("/api/pronunciation/words", json={"lang": lang, "word": word, "respelling": respelling})


def test_save_writes_the_table_removes_the_flag_and_applies_on_the_next_synthesis(client, state, flags_path, table_path):
    flag(client, "Stefan", "lb")
    flag(client, "Marc", "lb")
    english = flag(client, "Stefan", "en").json()
    assert state.pronunciations.apply("lb", "Moien Stefan") == "Moien Stefan"

    response = save(client, "Stefan", " [[ ʃtɛfan ]] ")

    assert (response.status_code, response.json()) == (
        200,
        {"lang": "lb", "word": "Stefan", "respelling": "[[ ʃtɛfan ]]"},
    )
    raw = table_path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert raw.decode("utf-8") == json.dumps({"lb": {"stefan": "[[ ʃtɛfan ]]"}}, ensure_ascii=False, indent=2)
    assert not table_path.with_name(table_path.name + ".tmp").exists()
    assert [(item["word"], item["lang"]) for item in file_flags(flags_path)] == [("Marc", "lb"), ("Stefan", "en")]
    assert file_flags(flags_path)[1]["id"] == english["id"]
    # The engine's table re-reads the file: no restart.
    assert state.pronunciations.apply("lb", "Moien Stefan") == "Moien [[ ʃtɛfan ]]"
    assert Pronunciations(table_path).apply("lb", "stefan, STEFAN") == "[[ ʃtɛfan, ]] [[ ʃtɛfan ]]"
    assert Pronunciations(table_path).apply("en", "Stefan") == "Stefan"
    listed = client.get("/api/pronunciation/flags").json()["flags"]
    assert [(item["word"], item["lang"], item["respelling"]) for item in listed] == [
        ("Stefan", "en", ""),
        ("Marc", "lb", ""),
    ]


def test_save_keeps_the_other_entries_and_replaces_the_same_word_in_any_case(client, table_path):
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(
        json.dumps({"en": {"Hello": "hullo"}, "lb": {"Moien": "mojen", "Marc": "mark"}}), encoding="utf-8-sig"
    )

    assert save(client, "moien", "mo-yen").status_code == 200

    assert json.loads(table_path.read_text(encoding="utf-8")) == {
        "en": {"Hello": "hullo"},
        "lb": {"Marc": "mark", "moien": "mo-yen"},
    }


def test_save_in_another_language_than_lb_goes_to_the_english_table(client, table_path):
    assert save(client, "Bonjour", "bonzhoor", lang="fr").json()["lang"] == "en"

    assert json.loads(table_path.read_text(encoding="utf-8")) == {"en": {"bonjour": "bonzhoor"}}


@pytest.mark.parametrize("word", ["a", "d'", "3", "x y", ""])
def test_save_refuses_a_word_that_cannot_be_a_key(client, table_path, word):
    response = save(client, word)

    assert response.status_code == 400 and response.json()["error"].startswith("bad word")
    assert not table_path.exists()


@pytest.mark.parametrize("respelling", ["", "   ", "x" * 61, "a\nb", "[[ a", "a ]]", "[[ a ]] [[ b ]]", "[a]"])
def test_save_refuses_a_respelling_that_breaks_the_rules(client, table_path, respelling):
    response = save(client, "Stefan", respelling)

    assert response.status_code == 400
    body = response.json()
    assert set(body) == {"error", "fix"} and body["error"].startswith("bad respelling") and body["fix"]
    assert not table_path.exists()


@pytest.mark.parametrize(
    "content",
    [b'{"lb": {"stefan": "x"}', b"", b'["x"]', b'{"fr": {}}', b'{"lb": {"a": 3}}', b"\xff\xfe nonsense"],
)
def test_save_does_not_write_over_a_table_that_cannot_be_read(client, table_path, flags_path, content):
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_bytes(content)
    flag(client, "Stefan")

    response = save(client)

    assert response.status_code == 409
    body = response.json()
    assert set(body) == {"error", "fix"} and str(table_path) in body["fix"]
    assert table_path.read_bytes() == content
    assert not table_path.with_name(table_path.name + ".tmp").exists()
    assert len(file_flags(flags_path)) == 1


def test_save_pronunciation_raises_for_a_table_that_cannot_be_read(tmp_path):
    path = tmp_path / "pronunciations.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(TableFileError, match="JSONDecodeError"):
        save_pronunciation(path, "lb", "Stefan", "stef")
    assert path.read_text(encoding="utf-8") == "{"


def test_save_creates_the_data_folder_when_it_is_not_there(tmp_path):
    path = tmp_path / "fresh" / "pronunciations.json"

    save_pronunciation(path, "lb", "Stefan", "stef")

    assert json.loads(path.read_text(encoding="utf-8")) == {"lb": {"stefan": "stef"}}


# --- trial and scope ---


def test_a_trial_respelling_reaches_the_engine_and_writes_no_file(client, state, flags_path, table_path):
    flag(client, "Stefan")
    before = flags_path.read_bytes()

    response = client.post(
        "/api/tts", json={"segments": [{"lang": "lb", "text": LINE}], "respell": {"Stefan": "stef"}}
    )

    assert response.status_code == 200
    assert state.tts.respell == {"Stefan": "stef"}
    assert not table_path.exists()
    assert flags_path.read_bytes() == before


def test_the_flags_stay_when_the_learner_deletes_everything(client, flags_path):
    flag(client, "Stefan")

    assert client.delete("/api/learner").status_code == 200

    assert len(file_flags(flags_path)) == 1
