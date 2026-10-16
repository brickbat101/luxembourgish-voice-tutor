"""The real data folder of the app, and a stat-only snapshot of what tests must never change in it."""

from __future__ import annotations

from pathlib import Path

from app.config import ROOT

REAL_DATA = ROOT / "data"

WATCHED_FILES = (
    "phrases.json",
    "phrases.json.corrupt",
    "phrases.json.tmp",
    "lod/lod.sqlite",
    "pronunciations.json",
    "pronunciation_flags.json",
)
WATCHED_FOLDERS = ("practice", "learner")


def snapshot(folder: Path) -> list[tuple[str, int, int]]:
    """Sorted (relative posix path, size, mtime_ns) of the watched files, plus (name, 0, 0) per top-level entry."""
    if not folder.is_dir():
        return []
    entries = {(entry.name, 0, 0) for entry in folder.iterdir()}
    files = [folder / name for name in WATCHED_FILES]
    for name in WATCHED_FOLDERS:
        if (folder / name).is_dir():
            files.extend((folder / name).rglob("*"))
    for path in files:
        if path.is_file():
            stat = path.stat()
            entries.add((path.relative_to(folder).as_posix(), stat.st_size, stat.st_mtime_ns))
    return sorted(entries)


def changed(before: list[tuple[str, int, int]], after: list[tuple[str, int, int]]) -> list[str]:
    """The paths that were added, removed or changed between two snapshots."""
    return sorted({path for path, _, _ in set(before) ^ set(after)})
