"""
FireSense - choose the vision model and its threshold by measurement.

The vision channel is somebody else's pretrained detector. Which one to use,
and at what confidence, is therefore an empirical question about OUR camera in
OUR room - and it has a known wrong answer: on this project's webcam a
community fire model reported `fire` on an empty room for twenty consecutive
frames at 0.40-0.45 confidence. Picking a model from its README would have
shipped that.

So: capture the two cases, then sweep.

    python get_weights.py --all                      # candidates into models/

    python pick_vision_model.py --capture normal     # ordinary room, no flame
    python pick_vision_model.py --capture flame      # SAME room, lighter lit

    python pick_vision_model.py --compare

The comparison prints, for every model and every threshold, how often it cried
fire at nothing and how often it saw a real flame. The threshold you want is
the lowest one with no false positives - and if a model has no such threshold,
that is the model telling you it is unusable here, which is worth knowing
before four days of recording rather than after.

Frames live in data/vision_check/{normal,flame}/ and are reusable: re-run
--compare after downloading another candidate without re-capturing anything.
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

CHECK_DIR = Path("data") / "vision_check"
CASES = ("normal", "flame")

# Detections are collected once at this floor, then filtered in Python, so a
# whole sweep costs one inference pass rather than one per threshold.
SWEEP_FLOOR = 0.10
SWEEP = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75)


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------
def capture(case: str, n: int, camera_spec: str, delay: float) -> int:
    import camera as cam_mod

    if case not in CASES:
        print(f"case must be one of {CASES}")
        return 1
    # Checked BEFORE the old frames below are deleted. These folders are the
    # adoption test (CLAUDE.md rule 10) and must hold deployment-camera
    # frames only; phone or other recorded footage goes elsewhere.
    if cam_mod.is_video_spec(camera_spec):
        print(f"  refusing to capture from a video file ({camera_spec}) into {CHECK_DIR}:")
        print("  these frames must come from the deployment camera (the ESP32-CAM).")
        print(f"  For recorded footage use:  python extract_frames.py {camera_spec}")
        return 1

    out = CHECK_DIR / case

    # Prove the camera works BEFORE touching the old frames: open it and read
    # one frame. A camera that is unplugged, off WiFi or busy must leave the
    # previous capture exactly as it was, not an empty folder.
    cam = cam_mod.make_camera(camera_spec)
    if not cam.open():
        print(f"camera unavailable: {getattr(cam, 'error', '?')}")
        print(f"  existing frames in {out} were left untouched")
        return 1
    probe = cam.read()
    if not probe.ok:
        cam.close()
        print(f"camera opened but the first frame failed: {probe.error}")
        print(f"  existing frames in {out} were left untouched")
        return 1

    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()

    print()
    if case == "normal":
        print("  CAPTURING: the ordinary room, NO flame anywhere in frame.")
        print("  Leave the scene exactly as it normally is - warm lamps, orange")
        print("  objects and all. Those are what cause the false positives, so")
        print("  hiding them would be cheating your own test.")
    else:
        print("  CAPTURING: the SAME room, with a lit flame in frame.")
        print("  A lighter or candle, roughly where a real one would be.")
        print("  Keep everything else identical to the normal capture.")
    print()
    for s in range(int(delay), 0, -1):
        sys.stdout.write(f"\r  starting in {s}... ")
        sys.stdout.flush()
        time.sleep(1)
    print("\r  capturing...      ")

    saved = 0
    for i in range(n):
        f = cam.read()
        if f.ok:
            cam_mod.cv2.imwrite(str(out / f"{i:03d}.jpg"), f.image)
            saved += 1
            sys.stdout.write(f"\r  {saved}/{n} frames")
            sys.stdout.flush()
        else:
            print(f"\n  frame {i} failed: {f.error}")
        time.sleep(0.25)
    cam.close()

    print(f"\n  saved {saved} frames -> {out}")
    other = CHECK_DIR / (CASES[1] if case == CASES[0] else CASES[0])
    if not list(other.glob("*.jpg")):
        print(f"\n  now capture the other case:")
        print(f"    python pick_vision_model.py --capture "
              f"{'flame' if case == 'normal' else 'normal'}")
    else:
        print(f"\n  both cases captured. Compare:")
        print(f"    python pick_vision_model.py --compare")
    return 0 if saved else 1


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
def _load_frames(case: str):
    import cv2
    paths = sorted((CHECK_DIR / case).glob("*.jpg"))
    frames = [(p.name, cv2.imread(str(p))) for p in paths]
    return [(n, f) for n, f in frames if f is not None]


def _collect(detector, frames) -> list[list]:
    """One inference pass per frame, every detection kept above SWEEP_FLOOR."""
    per_frame = []
    for name, frame in frames:
        res = detector.detect(frame)
        if not res.available:
            print(f"    {name}: {res.note}")
            per_frame.append([])
        else:
            per_frame.append(list(res.detections))
    return per_frame


def compare(weights_glob: str, device: str | None) -> int:
    import vision_ml as V

    frames = {c: _load_frames(c) for c in CASES}
    missing = [c for c in CASES if not frames[c]]
    if missing:
        print(f"\n  no frames for: {', '.join(missing)}")
        for c in missing:
            print(f"    python pick_vision_model.py --capture {c}")
        return 1

    paths = sorted(glob.glob(weights_glob))
    if not paths:
        print(f"\n  no weights matched {weights_glob}")
        print("    python get_weights.py --all")
        return 1

    print()
    print(f"  {len(frames['normal'])} normal frames, "
          f"{len(frames['flame'])} flame frames, {len(paths)} model(s)")

    summary = []
    for wp in paths:
        print()
        print("=" * 72)
        print(f"  {wp}")
        print("=" * 72)
        det = V.YoloFireDetector(wp, conf=SWEEP_FLOOR, device=device)
        if not det.load():
            print(f"  could not load: {det.error}")
            continue

        names = det.class_names
        print(f"  classes: {names}")
        unmapped = sorted({
            raw for raw in names.values()
            if V.normalize_label(raw) not in V.CLASS_WEIGHTS
        })
        if unmapped:
            print(f"  NOTE: {unmapped} are not recognised names, so they score at")
            print(f"        DEFAULT_CLASS_WEIGHT ({V.DEFAULT_CLASS_WEIGHT}). If any")
            print(f"        of them means fire or smoke, add it to LABEL_ALIASES.")

        t0 = time.time()
        got = {c: _collect(det, frames[c]) for c in CASES}
        ms = (time.time() - t0) * 1000 / max(1, sum(len(frames[c]) for c in CASES))

        print()
        print(f"  {'conf':>6}  {'false alarms':>14}  {'flame seen':>12}   verdict")
        print("  " + "-" * 62)
        best = None
        for thr in SWEEP:
            fp = sum(1 for d in got["normal"] if any(x.confidence >= thr for x in d))
            tp = sum(1 for d in got["flame"] if any(x.confidence >= thr for x in d))
            n_norm = len(frames["normal"]) or 1
            n_flame = len(frames["flame"]) or 1
            fp_r, tp_r = fp / n_norm, tp / n_flame
            if fp == 0 and best is None:
                best = (thr, tp_r)
                mark = "  <- lowest clean threshold"
            elif fp == 0:
                mark = ""
            else:
                mark = ""
            print(f"  {thr:6.2f}  {fp:4d}/{n_norm:<3d} {fp_r:5.0%}  "
                  f"{tp:4d}/{n_flame:<3d} {tp_r:4.0%}{mark}")

        # -- separation, which is what actually matters ------------------
        # The lowest clean threshold has zero margin by construction: one
        # slightly warmer evening and it is no longer clean. What you want is
        # the GAP between the loudest false positive and the quietest true
        # one, and a threshold sitting in the middle of it.
        fp_confs = [max((x.confidence for x in d), default=0.0)
                    for d in got["normal"]]
        tp_confs = [max((x.confidence for x in d), default=0.0)
                    for d in got["flame"]]
        worst_fp = max(fp_confs, default=0.0)
        seen = sorted(c for c in tp_confs if c > 0)
        typical_tp = seen[len(seen) // 2] if seen else 0.0
        gap = typical_tp - worst_fp
        # Mid-gap, but clamped. With no observed false positives at all the
        # midpoint can land absurdly low, and twelve frames of an empty room
        # is not enough evidence to trust 0.15 on a different evening.
        safe = (min(0.80, max(0.30, round((worst_fp + typical_tp) / 2, 2)))
                if gap > 0 else None)

        print()
        print(f"  loudest false positive : {worst_fp:.2f}")
        print(f"  typical real flame     : {typical_tp:.2f}"
              f"   ({len(seen)}/{len(tp_confs)} frames detected at all)")
        if gap > 0:
            print(f"  separation             : {gap:.2f}"
                  f"   -> threshold {safe:.2f} sits in the middle")
        else:
            print(f"  separation             : none. The model is no more "
                  f"confident about a")
            print(f"                           real flame than about your "
                  f"empty room.")

        print(f"\n  mean inference {ms:.0f} ms/frame")
        if best is None:
            print("  UNUSABLE HERE: false alarms at every threshold tested.")
            print("  Either the scene is the problem or the model is. Try another")
            print("  candidate before blaming the room.")
            summary.append((wp, None, 0.0, ms, gap, safe))
        else:
            thr, tp_r = best
            print(f"  clean from conf {thr:.2f}, where it still sees the flame "
                  f"in {tp_r:.0%} of frames")
            if tp_r < 0.5:
                print("  but that is poor detection - a model that is only quiet")
                print("  because it is blind is not a useful model.")
            summary.append((wp, thr, tp_r, ms, gap, safe))

    # -- the recommendation ------------------------------------------------
    print()
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  {'model':30s} {'clean':>6s} {'sep':>6s} {'use':>5s} "
          f"{'flame':>6s} {'ms':>6s}")
    print("  " + "-" * 66)
    for wp, thr, tp_r, ms, gap, safe in summary:
        t = f"{thr:.2f}" if thr is not None else "    --"
        u = f"{safe:.2f}" if safe is not None else "   --"
        print(f"  {Path(wp).name:30s} {t:>6s} {gap:6.2f} {u:>5s} "
              f"{tp_r:5.0%} {ms:6.0f}")
    print()
    print("  clean = lowest threshold with no false alarms")
    print("  sep   = gap between loudest false positive and typical real flame")
    print("  use   = the middle of that gap, which is the one to actually set")

    # Require real separation, not just a threshold that happens to be clean
    # on this many frames. A model with a 0.03 gap is one warm evening from
    # being useless.
    usable = [s for s in summary
              if s[1] is not None and s[2] >= 0.5 and s[4] >= 0.10]
    print()
    if not usable:
        print("  Nothing here is usable as-is - either false alarms at every")
        print("  threshold, poor detection, or a separation gap under 0.10.")
        print("  Options, in order of effort:")
        print("    1. try the remaining candidates in get_weights.py")
        print("    2. improve the scene - the camera pointing at a warm lamp is")
        print("       a real cause, not an excuse")
        print("    3. proceed without the vision channel. The sensors and the")
        print("       VLM still work, and the escalator rule means a missing")
        print("       vision channel degrades the system rather than breaking it.")
        return 2

    # Widest separation first - that is robustness. Detection rate breaks ties.
    usable.sort(key=lambda s: (-s[4], -s[2]))
    wp, thr, tp_r, ms, gap, safe = usable[0]
    print(f"  USE  {wp}")
    print(f"       CONF_THRESHOLD = {safe:.2f}   (in vision_ml.py)")
    print(f"       separation {gap:.2f}; no false alarms on "
          f"{len(frames['normal'])} frames of your")
    print(f"       own room; sees a real flame in {tp_r:.0%} of frames; "
          f"{ms:.0f} ms/frame")
    print()
    print("  Set it, then confirm end to end:")
    print(f"    python vision_ml.py webcam {wp} {safe:.2f}")
    print()
    print("  Worth saying in the report: this threshold was measured on the")
    print("  deployment scene, not copied from a paper. That is the difference")
    print("  between a tuned system and a borrowed one.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture", choices=CASES,
                   help="capture frames for one case")
    p.add_argument("--compare", action="store_true",
                   help="evaluate every model in models/ against both cases")
    p.add_argument("--n", type=int, default=20, help="frames to capture")
    p.add_argument("--camera", default="webcam", help="camera spec")
    p.add_argument("--delay", type=float, default=5,
                   help="seconds before capture starts")
    p.add_argument("--weights", default="models/*.pt")
    p.add_argument("--device", default=None, help='e.g. "mps" on Apple Silicon')
    a = p.parse_args()

    if a.capture:
        return capture(a.capture, a.n, a.camera, a.delay)
    if a.compare:
        return compare(a.weights, a.device)
    p.print_help()
    print()
    print("  the usual order:")
    print("    python get_weights.py --all")
    print("    python pick_vision_model.py --capture normal")
    print("    python pick_vision_model.py --capture flame")
    print("    python pick_vision_model.py --compare")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
