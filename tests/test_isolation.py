import os

from tests.realdata import REAL_DATA, changed, snapshot


def stat_of(path):
    if not path.is_file():
        return None
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def test_settings_point_at_a_temporary_data_folder(settings, tmp_path):
    assert settings.data_dir == tmp_path / "data"
    assert settings.data_dir != REAL_DATA


def test_the_app_starts_without_touching_the_real_data_folder(client):
    before = snapshot(REAL_DATA)

    assert client.get("/api/health").status_code == 200

    # The autouse guard compares the real folder again at teardown.
    assert changed(before, snapshot(REAL_DATA)) == []


def test_a_write_into_the_test_data_folder_is_not_a_write_into_the_real_one(settings):
    real_phrases = REAL_DATA / "phrases.json"
    before = stat_of(real_phrases)

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "phrases.json").write_text("[]", encoding="utf-8")

    assert (settings.data_dir / "phrases.json").read_text(encoding="utf-8") == "[]"
    assert stat_of(real_phrases) == before


def test_changed_names_a_file_that_was_added_changed_or_removed(tmp_path):
    folder = tmp_path / "data"
    (folder / "practice" / "2026-10-14-abcd1234").mkdir(parents=True)
    (folder / "learner").mkdir()
    (folder / "phrases.json").write_text("[]", encoding="utf-8")
    (folder / "practice" / "2026-10-14-abcd1234" / "answers.jsonl").write_text("{}\n", encoding="utf-8")
    (folder / "learner" / "profile.json").write_text("{}", encoding="utf-8")
    before = snapshot(folder)

    (folder / "learner" / "added.json").write_text("{}", encoding="utf-8")
    phrases = folder / "phrases.json"
    os.utime(phrases, ns=(phrases.stat().st_atime_ns, phrases.stat().st_mtime_ns + 1_000_000_000))
    (folder / "practice" / "2026-10-14-abcd1234" / "answers.jsonl").unlink()

    assert changed(before, snapshot(folder)) == [
        "learner/added.json",
        "phrases.json",
        "practice/2026-10-14-abcd1234/answers.jsonl",
    ]
    assert changed(before, before) == []
    assert snapshot(tmp_path / "missing") == []
