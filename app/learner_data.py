"""Everything the app keeps about the learner on this PC, taken as one: its size, the export ZIP
and the delete of all of it.

The learner's files are the learner store's folder (the database with its -wal and -shm, the
backups, the pre-migration copy), the practice folder and phrases.json with its `.corrupt` and
`.tmp`. Nothing else in the data folder is the learner's: the LOD dictionary and its recordings,
providers.json, pronunciations.json and pronunciation_flags.json are the app's.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import IO

from app.learner_store import DATABASE, LEARNER_FOLDER, PHRASES_FILE, PRACTICE_FOLDER, LearnerStore
from app.orchestrator import PhraseMemory
from app.practice_store import ANSWERS_FILE, PracticeStore, StoreError

PHRASE_FILES = (PHRASES_FILE, PHRASES_FILE + ".corrupt", PHRASES_FILE + ".tmp")
EXPORT_JSON = "learner.json"
ORIGINALS = "originals"  # in the ZIP: phrases.json and each answers.jsonl as they are on disk
AUDIO = "audio"  # in the ZIP: the recordings, by their place in the data folder
SPOOL_BYTES = 8 * 1024 * 1024  # a ZIP up to this size stays in memory
CHUNK_BYTES = 64 * 1024


def learner_paths(data_dir: Path) -> list[Path]:
    """Where the learner's files are: the learner store's folder, the practice folder, phrases.json*."""
    data_dir = Path(data_dir)
    return [data_dir / LEARNER_FOLDER, data_dir / PRACTICE_FOLDER, *(data_dir / name for name in PHRASE_FILES)]


def kept_bytes(data_dir: Path) -> int:
    """The bytes of every learner file; a link counts as itself, not as what it leads to."""
    size = 0
    for path in learner_paths(data_dir):
        try:
            if path.is_symlink() or path.is_junction() or not path.is_dir():
                size += path.lstat().st_size if os.path.lexists(path) else 0
                continue
            # os.walk does not go into a link to a folder.
            for folder, _, names in os.walk(path):
                size += sum((Path(folder) / name).lstat().st_size for name in names)
        except OSError:
            continue  # a file that went while it was counted
    return size


def export_name(exported_at: str) -> str:
    return f"luxembourgish-tutor-export-{exported_at[:10]}.zip"


def export_zip(store: LearnerStore) -> tuple[IO[bytes], str]:
    """(the export ZIP, at its start, the UTC time of the export).

    `learner.json` holds every row of every table, `learner.sqlite` a copy of the database
    with the same rows, `originals/` phrases.json, phrases.json.corrupt and each answers.jsonl,
    `audio/` the recordings, under their place in the data folder. The backups and the
    pre-migration copy are copies of these and are not in it.
    """
    spool = tempfile.SpooledTemporaryFile(max_size=SPOOL_BYTES)
    copies = Path(tempfile.mkdtemp(prefix="learner-export-"))
    try:
        exported = store.export(copies / DATABASE)
        with zipfile.ZipFile(spool, "w") as archive:
            text = json.dumps(exported, ensure_ascii=False, indent=2)
            archive.writestr(EXPORT_JSON, text, compress_type=zipfile.ZIP_DEFLATED)
            archive.write(copies / DATABASE, DATABASE, compress_type=zipfile.ZIP_STORED)
            for name, path in store.kept_files():
                if name in PHRASE_FILES or path.name == ANSWERS_FILE:
                    archive.write(path, f"{ORIGINALS}/{name}", compress_type=zipfile.ZIP_DEFLATED)
                else:
                    # A recording is compressed already.
                    archive.write(path, f"{AUDIO}/{name}", compress_type=zipfile.ZIP_STORED)
    except OSError as exc:
        spool.close()
        raise StoreError(f"the export was not made: {type(exc).__name__}: {exc}") from exc
    except BaseException:
        spool.close()
        raise
    finally:
        shutil.rmtree(copies, ignore_errors=True)
    spool.seek(0)
    return spool, exported["exported_at"]


def chunks(file: IO[bytes]) -> Iterator[bytes]:
    """The bytes of `file` in pieces; the file is closed at the end."""
    try:
        while piece := file.read(CHUNK_BYTES):
            yield piece
    finally:
        file.close()


def delete_everything(
    store: LearnerStore, practice_store: PracticeStore, phrases: PhraseMemory
) -> tuple[list[str], list[str]]:
    """Deletes every learner file: the learner store (closed, emptied and removed), the practice
    folder, phrases.json and its `.corrupt` and `.tmp`. Each part is tried even when one before
    fails. (what is still there of the learner paths, why a part was not deleted); both are
    empty when everything is gone."""
    problems = []
    for delete in (store.delete_all, practice_store.delete_all, phrases.delete):
        try:
            delete()
        except StoreError as exc:
            problems.append(str(exc))
        except OSError as exc:
            problems.append(f"{type(exc).__name__}: {exc}")
    left = [str(path) for path in learner_paths(store.data_dir) if os.path.lexists(path)]
    return left, problems
