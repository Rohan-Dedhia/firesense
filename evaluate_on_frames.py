"""
FireSense - check a trained vision model on frames from the deployment camera.

    python evaluate_on_frames.py --model models/maxvit_dfire/best.pt
    python evaluate_on_frames.py --model models/yolo_dfire/weights/best.pt
    python evaluate_on_frames.py --model models/maxvit_dfire/best.pt \
        --frames-dir data/vision_check_phone/candle      # phone footage

Reads the same folders as pick_vision_model.py:

    data/vision_check/normal/*.jpg    the ordinary room, no flame
    data/vision_check/flame/*.jpg     the same room, a lit flame in frame

(capture them with `python pick_vision_model.py --capture normal|flame`).

--frames-dir (or --dir) points it at any other folder with the same
normal/ and flame/ layout - for example frames from a phone video, made with
extract_frames.py and sorted by hand. The output is identical, plus a warning:
phone footage is not ESP32-CAM footage, so those numbers are a preview and do
not replace the check on data/vision_check.

Prints the false-alarm rate on `normal` and the detection rate on `flame`:
first at the thresholds chosen on D-Fire validation (what the model would
ship with), then as a sweep, then the separation between the loudest false
alarm and the typical real flame - the number that says whether ANY
threshold will work in this room. A D-Fire score says nothing about an
indoor ESP32-CAM; this is the check that does (CLAUDE.md rule 10).

Works with either model trained here:
  * MaxViT checkpoints from train_vision.py (two sigmoid outputs)
  * YOLO weights from train_yolo.py (image score = most confident box per class)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CHECK_DIR = Path("data") / "vision_check"
CASES = ("normal", "flame")
SWEEP = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75)
NOT_ESP32_WARNING = (
    "  WARNING: these frames are not from data/vision_check. If they are phone\n"
    "  or other recorded footage, they are NOT ESP32-CAM footage (different\n"
    "  sensor, lens, colour and JPEG encoder). Read the numbers below as a\n"
    "  preview; they do not replace the check on the deployment camera\n"
    "  (CLAUDE.md rule 10).")
OUTPUTS = ("fire", "smoke")


# ---------------------------------------------------------------------------
# scorers: every model becomes "frame -> (fire score, smoke score)"
# ---------------------------------------------------------------------------
class MaxVitScorer:
    def __init__(self, path: Path, device: str):
        import torch
        import train_vision as T

        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model = T.make_model(pretrained=False)
        self.model.load_state_dict(ck["model"])
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.mean, self.std = T._normalise(self.model)
        self.size = ck.get("size", T.SIZE)
        self.thresholds = _validated_thresholds(path.parent / "val_results.json")

    def score(self, paths: list[Path]) -> np.ndarray:
        import torch
        import train_vision as T
        from PIL import Image

        out = []
        for i in range(0, len(paths), 32):
            # Same preprocessing as training: whole frame resized, not cropped.
            batch = np.stack([np.asarray(Image.open(p).convert("RGB").resize(
                (self.size, self.size), Image.BICUBIC), dtype=np.uint8)
                for p in paths[i:i + 32]])
            with torch.no_grad():
                x = T.to_tensor(batch, self.mean, self.std, self.device)
                out.append(torch.sigmoid(self.model(x)).float().cpu().numpy())
        return np.concatenate(out)


class YoloScorer:
    def __init__(self, path: Path, device: str):
        from ultralytics import YOLO

        self.model = YOLO(str(path))
        self.device = device
        names = {int(k): str(v).lower() for k, v in self.model.names.items()}
        self.ids = {o: [k for k, v in names.items() if v == o] for o in OUTPUTS}
        missing = [o for o in OUTPUTS if not self.ids[o]]
        if missing:
            raise SystemExit(f"model classes {names} have no {missing}")
        # weights/best.pt -> the run folder holds val_results.json
        self.thresholds = _validated_thresholds(path.parent.parent / "val_results.json")

    def score(self, paths: list[Path]) -> np.ndarray:
        out = []
        for i in range(0, len(paths), 16):
            res = self.model.predict([str(p) for p in paths[i:i + 16]], conf=0.01,
                                     device=self.device, verbose=False)
            for r in res:
                cls = r.boxes.cls.cpu().numpy().astype(int)
                conf = r.boxes.conf.cpu().numpy()
                out.append([float(conf[np.isin(cls, self.ids[o])].max())
                            if np.isin(cls, self.ids[o]).any() else 0.0
                            for o in OUTPUTS])
        return np.array(out, dtype=np.float32)


def _validated_thresholds(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())["thresholds"]
    return None


def is_check_dir(path: Path) -> bool:
    """True for data/vision_check itself - the deployment-camera folder."""
    try:
        return Path(path).resolve() == CHECK_DIR.resolve()
    except OSError:
        return False


def make_scorer(path: Path, device: str):
    import torch

    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        ck = None
    if isinstance(ck, dict) and "model_name" in ck and "outputs" in ck:
        return MaxVitScorer(path, device)
    return YoloScorer(path, device)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def evaluate(model_path: str, check_dir: Path, device: str, scorer=None) -> int:
    """`scorer` is for tests: anything with .score(paths) and .thresholds."""
    check_dir = Path(check_dir)
    deployment = is_check_dir(check_dir)
    if not deployment:
        print()
        print(NOT_ESP32_WARNING)

    frames = {c: sorted((check_dir / c).glob("*.jpg")) for c in CASES}
    missing = [c for c in CASES if not frames[c]]
    if missing:
        print(f"\n  no frames for: {', '.join(missing)} (looked in {check_dir})")
        for c in missing:
            if deployment:
                print(f"    python pick_vision_model.py --capture {c}")
            else:
                print(f"    put the {c} frames in {check_dir / c}/  "
                      f"(extract_frames.py output, sorted by hand)")
        return 1

    scorer = scorer or make_scorer(Path(model_path), device)
    s = {c: scorer.score(frames[c]) for c in CASES}
    n_norm, n_flame = len(frames["normal"]), len(frames["flame"])

    print()
    print("=" * 72)
    print(f"  {model_path}")
    print(f"  {n_norm} normal frames, {n_flame} flame frames from {check_dir}")
    if not deployment:
        print("  (not the ESP32-CAM check folder - see the warning above)")
    print("=" * 72)

    thr = scorer.thresholds
    if thr:
        print(f"\n  at the thresholds chosen on D-Fire validation "
              f"(fire {thr['fire']:.2f}, smoke {thr['smoke']:.2f}):")
        fa_any = int(((s["normal"][:, 0] >= thr["fire"]) |
                      (s["normal"][:, 1] >= thr["smoke"])).sum())
        fa_fire = int((s["normal"][:, 0] >= thr["fire"]).sum())
        det = int((s["flame"][:, 0] >= thr["fire"]).sum())
        print(f"    false alarms on normal, any output : {fa_any:4d}/{n_norm:<4d} "
              f"{fa_any / n_norm:5.0%}")
        print(f"    false alarms on normal, fire output: {fa_fire:4d}/{n_norm:<4d} "
              f"{fa_fire / n_norm:5.0%}")
        print(f"    flame detected (fire output)       : {det:4d}/{n_flame:<4d} "
              f"{det / n_flame:5.0%}")
    else:
        print("\n  (no val_results.json next to the model: sweep only)")

    # The sweep is on the fire score: a flame frame should be called fire.
    print()
    print(f"  {'conf':>6}  {'false alarms':>14}  {'flame seen':>12}   verdict")
    print("  " + "-" * 62)
    best = None
    for t in SWEEP:
        fp = int((s["normal"][:, 0] >= t).sum())
        tp = int((s["flame"][:, 0] >= t).sum())
        mark = ""
        if fp == 0 and best is None:
            best = (t, tp / n_flame)
            mark = "  <- lowest clean threshold"
        print(f"  {t:6.2f}  {fp:4d}/{n_norm:<3d} {fp / n_norm:5.0%}  "
              f"{tp:4d}/{n_flame:<3d} {tp / n_flame:4.0%}{mark}")

    worst_fp = float(s["normal"][:, 0].max())
    typical_tp = float(np.median(s["flame"][:, 0]))
    gap = typical_tp - worst_fp
    print()
    print(f"  loudest false positive : {worst_fp:.2f}")
    print(f"  typical real flame     : {typical_tp:.2f}   (median fire score)")
    if gap > 0:
        print(f"  separation             : {gap:.2f}   -> threshold "
              f"{(worst_fp + typical_tp) / 2:.2f} sits in the middle")
    else:
        print("  separation             : none. The model is no more confident "
              "about a")
        print("                           real flame than about your empty room.")

    print()
    if best is None:
        print("  UNUSABLE HERE: false alarms at every threshold tested.")
        return 2
    t, rate = best
    print(f"  clean from conf {t:.2f}, where it still sees the flame in {rate:.0%} "
          f"of frames")
    if rate < 0.5:
        print("  but that is poor detection - a model that is only quiet")
        print("  because it is blind is not a useful model.")
        return 2
    if gap < 0.10:
        print("  separation under 0.10: one warm evening from being useless.")
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="MaxViT best.pt or YOLO .pt")
    p.add_argument("--frames-dir", "--dir", dest="frames_dir", default=str(CHECK_DIR),
                   help="folder holding normal/ and flame/ (default data/vision_check)")
    p.add_argument("--device", default="mps")
    return p


def main() -> int:
    a = build_parser().parse_args()
    return evaluate(a.model, Path(a.frames_dir), a.device)


if __name__ == "__main__":
    raise SystemExit(main())
