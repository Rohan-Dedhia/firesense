"""
FireSense - automatic HSV tuner.

Instead of dragging sliders by eye, this learns the flame's colour signature
from your actual camera, in your actual room:

  1. records a few seconds of the scene with NO flame   (the background)
  2. records a few seconds WITH the flame in view
  3. finds the pixels that changed and are bright - that's the flame
  4. takes percentiles of their hue/saturation/value to build the HSV range
  5. validates the range against both recordings and picks a trigger threshold
     that sits between the worst background frame and the typical flame frame

    python auto_tune_vision.py            # tune and print
    python auto_tune_vision.py --write    # tune and write into config.py

Defensible answer when your evaluator asks how you chose these numbers: they
were derived from the measured pixel distribution of the actual flame against
the actual background, with a validated separation margin - not guessed.
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

import config as C

N_FRAMES = 90            # ~3-6 s depending on the camera
CHANGE_THRESHOLD = 45    # per-pixel BGR change that counts as "something new"
MIN_VALUE = 110          # flames are bright; ignore dim changes (shadows)
PCT_LOW, PCT_HIGH = 3, 97


def grab(cap, n, label):
    frames = []
    print(f"  recording {label} ", end="", flush=True)
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            continue
        frames.append(f)
        if len(frames) % 15 == 0:
            print(".", end="", flush=True)
        cv2.imshow("FireSense auto-tune (recording)", f)
        cv2.waitKey(1)
    print(" done")
    return frames


def flame_pixels(bg_frames, fl_frames):
    """HSV values of pixels that appeared, and are bright, in the flame clips."""
    bg = np.median(np.stack(bg_frames), axis=0).astype(np.int16)
    collected = []
    for f in fl_frames:
        diff = np.abs(f.astype(np.int16) - bg).sum(axis=2)
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        mask = (diff > CHANGE_THRESHOLD) & (hsv[:, :, 2] > MIN_VALUE)
        if mask.sum():
            collected.append(hsv[mask])
    if not collected:
        return None
    return np.concatenate(collected, axis=0)


def derive_range(px):
    h, s, v = px[:, 0], px[:, 1], px[:, 2]
    # Flame hue sits at the red/orange/yellow end. Red wraps around 179, so
    # fold the high-hue tail down before taking percentiles.
    wrapped = (h > 150).mean()
    h = h.astype(np.int16)
    h[h > 150] -= 180
    lo_h = max(0, int(np.percentile(h, PCT_LOW)))
    hi_h = int(np.percentile(h, PCT_HIGH))
    lower = (lo_h, int(np.percentile(s, PCT_LOW)), int(np.percentile(v, PCT_LOW)))
    upper = (min(179, hi_h), 255, 255)
    return lower, upper, wrapped


def ratio_series(frames, lower, upper):
    lo, up = np.array(lower, np.uint8), np.array(upper, np.uint8)
    k = np.ones((5, 5), np.uint8)
    out = []
    for f in frames:
        m = cv2.inRange(cv2.cvtColor(f, cv2.COLOR_BGR2HSV), lo, up)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        out.append(np.count_nonzero(m) / m.size)
    return np.array(out)


def choose_trigger(bg_ratios, fl_ratios):
    """Sit between the worst background frame and the typical flame frame."""
    bg_worst = float(bg_ratios.max())
    fl_typical = float(np.percentile(fl_ratios, 25))
    if fl_typical <= bg_worst:
        return None, bg_worst, fl_typical
    return (bg_worst + fl_typical) / 2.0, bg_worst, fl_typical


def write_config(lower, upper, trigger):
    import re
    from pathlib import Path
    p = Path(__file__).with_name("config.py")
    s = p.read_text()
    s = re.sub(r"HSV_LOWER = \([^)]*\)", f"HSV_LOWER = {tuple(int(x) for x in lower)}", s)
    s = re.sub(r"HSV_UPPER = \([^)]*\)", f"HSV_UPPER = {tuple(int(x) for x in upper)}", s)
    s = re.sub(r"FIRE_PIXEL_RATIO_TRIGGER = [0-9.]+",
               f"FIRE_PIXEL_RATIO_TRIGGER = {trigger:.4f}", s)
    p.write_text(s)
    print(f"\n  written into {p}")


def main():
    cap = cv2.VideoCapture(C.CAMERA_INDEX)
    if not cap.isOpened():
        print("camera did not open - try a different CAMERA_INDEX in config.py")
        return
    for _ in range(20):     # let auto-exposure settle
        cap.read()

    print("\nFireSense automatic HSV tuning")
    print("Set the camera exactly as it will be for the demo, then:\n")
    input("  1) NO flame in view. Press Enter to record the background...")
    bg = grab(cap, N_FRAMES, "background")
    input("\n  2) Light the candle and hold it where you will during the demo.\n"
          "     Press Enter to record the flame...")
    fl = grab(cap, N_FRAMES, "flame")
    cap.release()
    cv2.destroyAllWindows()

    px = flame_pixels(bg, fl)
    if px is None or len(px) < 200:
        print("\nCould not isolate the flame. Usually this means the flame was too "
              "small or too far away.\nMove it closer to the lens and run again.")
        return

    lower, upper, wrapped = derive_range(px)
    bg_r = ratio_series(bg, lower, upper)
    fl_r = ratio_series(fl, lower, upper)
    trigger, bg_worst, fl_typical = choose_trigger(bg_r, fl_r)

    print(f"\n  flame pixels sampled : {len(px):,}")
    print(f"  background worst     : {bg_worst*100:.3f} % of frame")
    print(f"  flame typical        : {fl_typical*100:.3f} % of frame")
    if wrapped > 0.15:
        print(f"  note: {wrapped*100:.0f} % of flame pixels were deep red (hue wraps "
              f"past 179). Keep H low at 0.")

    if trigger is None:
        print("\n  FAILED TO SEPARATE. The background already contains as much "
              "flame-coloured\n  area as the flame does. Fix the scene, not the "
              "numbers: remove orange/warm\n  objects from view, dim warm lamps, "
              "or point the camera at a plainer wall.")
        return

    margin = fl_typical / max(bg_worst, 1e-6)
    print(f"\n  HSV_LOWER = {tuple(int(x) for x in lower)}")
    print(f"  HSV_UPPER = {tuple(int(x) for x in upper)}")
    print(f"  FIRE_PIXEL_RATIO_TRIGGER = {trigger:.4f}")
    print(f"  separation margin: {margin:.1f}x "
          f"({'comfortable' if margin >= 3 else 'tight - re-run closer if you can'})")

    if "--write" in sys.argv:
        write_config(lower, upper, trigger)
    else:
        print("\n  re-run with --write to put these into config.py")


if __name__ == "__main__":
    main()
