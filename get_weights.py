"""
FireSense - fetch candidate fire/smoke detector weights.

Roboflow Universe needs an account, an API key and a download dialog. Hugging
Face serves the .pt file over plain HTTPS, so this is one command and no
signup:

    python get_weights.py --list          # what is on offer
    python get_weights.py --all           # fetch every candidate
    python get_weights.py yolov10         # fetch one

Files land in models/. Nothing here trains anything - these are other people's
pretrained detectors, and treating them as a COMPONENT rather than as our
contribution is the honest framing. Which one to actually use is decided by
measurement, not by the README: see pick_vision_model.py.

On licences
-----------
For an academic project this matters, so it is recorded per candidate below.
`yolov10` is Apache-2.0. The SHOU-ISD repository states no licence at all,
which means no permission is granted in writing - fine for a private
experiment, not something to redistribute or build a product on. The table
prints this every time on purpose.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

MODELS_DIR = Path("models")

# name -> (url, saved filename, arch, licence, note)
CANDIDATES: dict[str, tuple[str, str, str, str, str]] = {
    "yolov10": (
        "https://huggingface.co/TommyNgx/YOLOv10-Fire-and-Smoke-Detection"
        "/resolve/main/best.pt",
        "fire_yolov10.pt", "YOLOv10", "Apache-2.0",
        "fire + smoke; the safest licence of the three",
    ),
    "yolov8n": (
        "https://huggingface.co/SHOU-ISD/fire-and-smoke"
        "/resolve/main/yolov8n_1.pt",
        "fire_yolov8n.pt", "YOLOv8n", "not stated",
        "small and fast; good first thing to try on a laptop",
    ),
    "yolov8-ema": (
        "https://huggingface.co/SHOU-ISD/fire-and-smoke"
        "/resolve/main/bestEMA.pt",
        "fire_yolov8_ema.pt", "YOLOv8", "not stated",
        "EMA weights from the same repo; usually the stronger of the two",
    ),
}


def show_table() -> None:
    print()
    print(f"  {'name':12s} {'arch':10s} {'licence':12s} note")
    print("  " + "-" * 70)
    for key, (_, fname, arch, lic, note) in CANDIDATES.items():
        here = " [have]" if (MODELS_DIR / fname).exists() else ""
        print(f"  {key:12s} {arch:10s} {lic:12s} {note}{here}")
    print()
    print("  These are pretrained by other people on other people's data.")
    print("  They are a component of this project, not a result of it.")
    print()


def download(key: str, force: bool = False) -> Path | None:
    url, fname, arch, lic, _ = CANDIDATES[key]
    MODELS_DIR.mkdir(exist_ok=True)
    dest = MODELS_DIR / fname
    if dest.exists() and not force:
        mb = dest.stat().st_size / 1e6
        print(f"  {key:12s} already here ({mb:.1f} MB) - --force to replace")
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  {key:12s} downloading {arch}, licence {lic}")
    req = urllib.request.Request(url, headers={"User-Agent": "firesense/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            with tmp.open("wb") as fh:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    fh.write(chunk)
                    got += len(chunk)
                    if total:
                        pct = 100 * got / total
                        sys.stdout.write(f"\r    {got/1e6:6.1f} / "
                                         f"{total/1e6:.1f} MB  {pct:5.1f}%")
                    else:
                        sys.stdout.write(f"\r    {got/1e6:6.1f} MB")
                    sys.stdout.flush()
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        print(f"\r    FAILED  HTTP {exc.code} - the file may have moved. "
              f"Open the repo in a browser and check.")
        return None
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        print(f"\r    FAILED  {type(exc).__name__}: {exc}")
        print("            If this is a proxy or firewall, download it in a")
        print("            browser instead:  python get_weights.py --urls")
        return None

    # A .pt is a zip archive. An HTML error page saved under a .pt name is the
    # classic way this fails silently, so check the magic bytes before
    # committing the file.
    with tmp.open("rb") as fh:
        magic = fh.read(2)
    if magic != b"PK":
        tmp.unlink(missing_ok=True)
        print(f"\r    FAILED  that was not a model file (got {magic!r}) - "
              f"probably an error page.")
        return None

    shutil.move(str(tmp), str(dest))
    print(f"\r    saved  {dest}  ({dest.stat().st_size/1e6:.1f} MB)"
          + " " * 20)
    return dest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", help="which candidates to fetch")
    p.add_argument("--list", action="store_true", help="show the table and exit")
    p.add_argument("--all", action="store_true", help="fetch every candidate")
    p.add_argument("--force", action="store_true", help="re-download")
    p.add_argument("--urls", action="store_true",
                   help="just print the URLs, to download in a browser instead")
    a = p.parse_args()

    if a.urls:
        print()
        for key, (url, fname, _, lic, _n) in CANDIDATES.items():
            print(f"  {key}   (licence {lic})")
            print(f"    {url}")
            print(f"    save as  models/{fname}")
            print()
        return 0

    if a.list or (not a.names and not a.all):
        show_table()
        print("  fetch them:  python get_weights.py --all")
        return 0

    wanted = list(CANDIDATES) if a.all else a.names
    unknown = [n for n in wanted if n not in CANDIDATES]
    if unknown:
        print(f"unknown: {', '.join(unknown)}")
        show_table()
        return 1

    print()
    got = [download(n, a.force) for n in wanted]
    ok = [g for g in got if g]
    print()
    print(f"  {len(ok)}/{len(wanted)} downloaded into {MODELS_DIR}/")
    if ok:
        print()
        print("  now measure them instead of trusting them:")
        print("    python pick_vision_model.py --capture-negative")
        print("    python pick_vision_model.py --capture-positive   # lighter in frame")
        print("    python pick_vision_model.py --compare")
    return 0 if len(ok) == len(wanted) else 1


if __name__ == "__main__":
    raise SystemExit(main())
