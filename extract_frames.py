"""
FireSense - save every Nth frame of a video as JPEG.

    python extract_frames.py phone_candle.mp4                 # every 30th frame
    python extract_frames.py phone_candle.mp4 --every 10
    python extract_frames.py phone_candle.mp4 --out data/vision_check_phone/candle

Default output: data/vision_check_phone/<video name>/.

Not data/vision_check/. That folder is the adoption test for a vision model
(CLAUDE.md rule 10, evaluate_on_frames.py, pick_vision_model.py) and must
hold frames from the deployment camera only. A phone has a different sensor,
lens, white balance, resolution and JPEG encoder from the ESP32-CAM's OV3660;
a model that separates fire from no-fire on phone footage has not shown that
it does so on the ESP32-CAM. So this script refuses to write inside
data/vision_check/, and says why every time it runs.

Phone frames are still useful: a first look at a model on the real room, and
`evaluate_on_frames.py --dir data/vision_check_phone/<name>` if you sort them
into normal/ and flame/ yourself - reported as phone results, not as the check.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import camera as cam_mod

PHONE_DIR = Path("data") / "vision_check_phone"
CHECK_DIR = Path("data") / "vision_check"        # the ESP32-CAM folder: never written here
DEFAULT_EVERY = 30                               # about one frame a second at 30 fps
JPEG_QUALITY = 95

WARNING = (
    "  NOTE: these frames are recorded footage (phone or other camera), not\n"
    "  ESP32-CAM frames. Keep them out of data/vision_check/: that folder decides\n"
    "  whether a vision model is adopted, and must come from the deployment camera.")


def default_out(video: Path) -> Path:
    return PHONE_DIR / video.stem


def inside_check_dir(out: Path) -> bool:
    """True if `out` is data/vision_check or anything below it."""
    try:
        out.resolve().relative_to(CHECK_DIR.resolve())
        return True
    except ValueError:
        return False


def extract(video: Path, out: Path, every: int, limit: int = 0,
            force: bool = False) -> tuple[int, str]:
    """-> (frames saved, error). Pure enough to test: no prompts, no exit."""
    if every < 1:
        return 0, "--every must be 1 or more"
    if inside_check_dir(out):
        return 0, (f"refusing to write into {CHECK_DIR}: it holds ESP32-CAM frames "
                   f"only. Use the default {PHONE_DIR}/<name>/ or another folder.")
    if out.exists() and any(out.glob("*.jpg")) and not force:
        return 0, (f"{out} already has JPEGs. Choose another --out, or pass "
                   f"--force to write alongside them (same-named files are replaced).")

    src = cam_mod.VideoFileSource(video, loop=False, realtime=False)
    if not src.open():
        return 0, src.error
    out.mkdir(parents=True, exist_ok=True)
    saved = index = 0
    try:
        while True:
            f = src.read()
            if not f.ok:
                if f.end_of_stream:
                    break
                return saved, f"read failed at frame {index}: {f.error}"
            if index % every == 0:
                name = out / f"{video.stem}_{index:06d}.jpg"
                if not cam_mod.cv2.imwrite(str(name), f.image,
                                           [cam_mod.cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
                    return saved, f"could not write {name}"
                saved += 1
                if limit and saved >= limit:
                    break
            index += 1
    finally:
        src.close()
    return saved, ""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", help=".mp4 / .mov / .avi / .mkv")
    p.add_argument("--every", type=int, default=DEFAULT_EVERY,
                   help=f"keep every Nth frame (default {DEFAULT_EVERY})")
    p.add_argument("--out", default=None,
                   help=f"output folder (default {PHONE_DIR}/<video name>/)")
    p.add_argument("--max", type=int, default=0, help="stop after this many (0 = all)")
    p.add_argument("--force", action="store_true",
                   help="write into a folder that already has JPEGs")
    a = p.parse_args()

    video = Path(a.video).expanduser()
    out = Path(a.out).expanduser() if a.out else default_out(video)
    print(WARNING)
    print()
    saved, err = extract(video, out, a.every, a.max, a.force)
    if err:
        print(f"  {err}")
        if saved:
            print(f"  ({saved} frames were saved to {out} before that)")
        return 1
    print(f"  saved {saved} frames (every {a.every}th) -> {out}")
    if saved == 0:
        print("  the video had no frames to save")
    return 0


if __name__ == "__main__":
    sys.exit(main())
