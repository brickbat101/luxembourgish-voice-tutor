"""Download the two CC0 LOD dumps into <data_dir>/lod/raw/ and build <data_dir>/lod/lod.sqlite.

Run from anywhere: python scripts/download_lod.py
The zips are data only: they are unpacked with zipfile and parsed, nothing in them is run.
"""

from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.lod.build import ART_XML, TAB_XML, build  # noqa: E402

BASE = "https://download.data.public.lu/resources/"
URLS = (
    BASE + "letzebuerger-online-dictionnaire-lod-flexiounstabellen/20260727-101009/260727-new-lod-tab.zip",
    BASE + "letzebuerger-online-dictionnaire-lod-linguistesch-daten/20260727-095857/260727-new-lod-art.zip",
)
MAX_UNPACKED_BYTES = 1_000_000_000


def fetch(url: str, raw_dir: Path) -> Path:
    """Download `url` into `raw_dir` unless the file is already there."""
    target = raw_dir / url.rsplit("/", 1)[1]
    if target.exists():
        print(f"present  {target.name}")
        return target
    part = target.with_name(target.name + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with part.open("wb") as out:
            for chunk in response.iter_bytes():
                out.write(chunk)
    part.replace(target)
    print(f"fetched  {target.name} ({target.stat().st_size:,} bytes)")
    return target


def extract(zip_path: Path, raw_dir: Path) -> list[Path]:
    """Unpack the .xml members of `zip_path` flat into `raw_dir`, by base name only."""
    extracted = []
    with zipfile.ZipFile(zip_path) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        if sum(m.file_size for m in members) > MAX_UNPACKED_BYTES:
            raise ValueError(f"{zip_path.name} unpacks to more than {MAX_UNPACKED_BYTES:,} bytes")
        for member in members:
            # Only the base name is used, so "../x" or "C:/x" cannot leave raw_dir.
            name = PurePosixPath(member.filename.replace("\\", "/")).name
            if not re.fullmatch(r"[\w.-]+\.xml", name):
                print(f"skipped  {member.filename!r} in {zip_path.name}")
                continue
            target = raw_dir / name
            with archive.open(member) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out)
            extracted.append(target)
    return extracted


def download(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for url in URLS:
        for path in extract(fetch(url, raw_dir), raw_dir):
            print(f"unpacked {path.name} ({path.stat().st_size:,} bytes)")


def main() -> None:
    lod_dir = Settings.from_env().data_dir / "lod"
    raw_dir = lod_dir / "raw"
    download(raw_dir)
    counts = build(raw_dir / TAB_XML, raw_dir / ART_XML, lod_dir / "lod.sqlite")
    print(f"built    {lod_dir / 'lod.sqlite'}: {counts}")


if __name__ == "__main__":
    main()
