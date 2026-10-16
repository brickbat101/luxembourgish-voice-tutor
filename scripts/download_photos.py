"""Download the photos of practice mode from Wikimedia Commons into static/photos/.

    python scripts/download_photos.py            fetch the photos that are not there yet
    python scripts/download_photos.py --check    only compare the licences; writes nothing

The photos are the entries of app/content/photos.toml. Commons is asked for the licence of
each; a photo whose licence there is not "Public domain" or "CC0", or not the one the entry
records, is refused. Every request goes to commons.wikimedia.org and no redirect is followed.
The images are data only: they are saved, nothing in them is run.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ROOT  # noqa: E402
from app.content import PHOTO_LICENCES, Photo, load_photos  # noqa: E402

API = "https://commons.wikimedia.org/w/api.php"
RENDITION = "https://commons.wikimedia.org/w/thumb.php"  # a file scaled to a width, as a JPEG
# Wikimedia refuses clients that do not say who they are.
USER_AGENT = "loci-language-app/0.1 (practice photo downloader)"
WIDTH = 1200  # pixels
STATIC_DIR = ROOT / "static"
TRIES = 5  # of a request that Commons answers with 429, too many requests
PAUSE_S = 1.0  # after an image download
_TAG = re.compile(r"<[^>]+>")


def get(client: httpx.Client, url: str, params: dict[str, str]) -> httpx.Response:
    """The response to a GET; after a 429 it waits as long as Commons asks and tries again."""
    for _ in range(TRIES):
        response = client.get(url, params=params)
        if response.status_code != 429:
            break
        time.sleep(float(response.headers.get("retry-after", "5")) + 1)
    response.raise_for_status()
    return response


def say(line: str) -> None:
    """Prints `line`; a character the console cannot show is printed as its escape."""
    encoding = sys.stdout.encoding or "utf-8"
    print(line.encode(encoding, "backslashreplace").decode(encoding))


def plain(html: str) -> str:
    """`html` of a Commons metadata field as one line of text."""
    return " ".join(_TAG.sub(" ", html).split())


def commons_info(client: httpx.Client, photos: Sequence[Photo]) -> dict[str, dict[str, str]]:
    """Per photo id the `licence` and `author` Commons has for its file, asked in one request.

    A photo whose file Commons does not have is left out.
    """
    params = {
        "action": "query",
        "format": "json",
        "titles": "|".join(f"File:{photo.file}" for photo in photos),
        "prop": "imageinfo",
        "iiprop": "extmetadata",
    }
    query = get(client, API, params).json().get("query", {})
    # Commons answers under its own spelling of a title, e.g. with spaces for underscores.
    spelled = {entry["from"]: entry["to"] for entry in query.get("normalized", [])}
    pages = {page.get("title"): page for page in query.get("pages", {}).values()}
    found = {}
    for photo in photos:
        title = f"File:{photo.file}"
        infos = pages.get(spelled.get(title, title), {}).get("imageinfo")
        if infos:
            meta = infos[0].get("extmetadata", {})
            found[photo.id] = {
                "licence": plain(meta.get("LicenseShortName", {}).get("value", "")),
                "author": plain(meta.get("Artist", {}).get("value", "")),
            }
    return found


def refusal(photo: Photo, info: dict[str, str] | None) -> str:
    """Why `photo` is not taken, given what Commons has on it; "" when it is."""
    if info is None:
        return f"Commons has no File:{photo.file}"
    if info["licence"] not in PHOTO_LICENCES:
        return f"its licence on Commons is {info['licence']!r}, not one of {', '.join(PHOTO_LICENCES)}"
    if info["licence"] != photo.licence:
        return f"its licence on Commons is {info['licence']!r}, the entry records {photo.licence!r}"
    return ""


def fetch(client: httpx.Client, photo: Photo, target: Path) -> int:
    """Saves the rendition of `photo`, WIDTH pixels wide, as `target`; its size in bytes."""
    response = get(client, RENDITION, {"f": photo.file, "w": str(WIDTH)})
    kind = response.headers.get("content-type", "")
    if not kind.startswith("image/jpeg"):
        raise ValueError(f"Commons sent {kind or 'no content type'}, not a JPEG")
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    part.write_bytes(response.content)
    part.replace(target)
    return target.stat().st_size


def download(client: httpx.Client, photos: Sequence[Photo], static_dir: Path, check: bool = False) -> list[str]:
    """Fetches the image of each of `photos` that is not under `static_dir` yet; the refusals.

    With `check` nothing is fetched or written: only the licences are compared. A line per
    photo is printed, with the author and the licence Commons has for it.
    """
    found = commons_info(client, photos)
    refused = []
    for photo in photos:
        info = found.get(photo.id)
        target = static_dir / photo.image
        why = refusal(photo, info)
        if why:
            pass
        elif check:
            say(f"ok       {photo.id}: {info['author']}, {info['licence']}")
        elif target.exists():
            say(f"present  {photo.id} ({target.stat().st_size:,} bytes): {info['author']}, {info['licence']}")
        else:
            try:
                size = fetch(client, photo, target)
            except (httpx.HTTPError, ValueError) as exc:
                why = str(exc)
            else:
                say(f"fetched  {photo.id} ({size:,} bytes): {info['author']}, {info['licence']}")
                time.sleep(PAUSE_S)
        if why:
            refused.append(f"{photo.id}: {why}")
            say(f"refused  {photo.id}: {why}")
    return refused


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only compare the licences; writes nothing")
    args = parser.parse_args(argv)
    photos = load_photos()
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60) as client:
        try:
            refused = download(client, photos, STATIC_DIR, check=args.check)
        except httpx.HTTPError as exc:
            say(f"failed   {exc}")
            return 1
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
