"""
FireSense - the MaxViT-tiny vision channel, and YOLO boxes for the alert photo.

    python pipeline.py --source serial --camera 192.168.1.42 --vision maxvit
    python pipeline.py ... --vision maxvit --alert-boxes      # + YOLO v2 boxes

MaxViT-tiny was trained on D-Fire by train_vision.py (models/maxvit_dfire/).
It is an image classifier with two sigmoid outputs, fire and smoke, so it says
WHETHER a frame shows fire or smoke but not WHERE. The thresholds are the ones
chosen on D-Fire validation (val_results.json beside the weights) - never
typed in here. If that file is missing the channel refuses to load rather than
guess a threshold.

D-Fire is mostly outdoor footage. Its numbers do not transfer to an indoor
ESP32-CAM, so this channel is opt-in (`--vision maxvit`). Whether it is
adopted is decided by `evaluate_on_frames.py` on frames of the actual room
(CLAUDE.md rule 10), not by its D-Fire scores.

The channel plugs into the pipeline exactly like vision_ml.VisionChannel: the
same VisionResult, the same 2-frame debounce, the same escalator treatment in
pipeline.combine(). It can only raise the level, and on its own it is capped
at VISION_ALONE_MAX, below CRITICAL.

Alert boxes. A classifier has no boxes to show. When `--alert-boxes` is on, the
YOLOv8n run-2 detector (models/yolo_dfire_v2) draws boxes on the photo that
goes out with an alert - and does nothing else. It never feeds the score: its
D-Fire false-alarm rate on "neither" images was 8.0% on validation against
MaxViT's 5.8%, so it is shown to the person, not trusted by the system.

Nothing here imports torch, timm or ultralytics at module level: the tests
and the hosted app must import this file on a machine with none of them.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

import vision_ml as V

MAXVIT_WEIGHTS = Path("models") / "maxvit_dfire" / "best.pt"
YOLO_BOX_WEIGHTS = Path("models") / "yolo_dfire_v2" / "weights" / "best.pt"
OUTPUTS = ("fire", "smoke")         # the classifier's output order (train_vision.py)


def load_thresholds(path: Path) -> Optional[dict]:
    """Validated per-output thresholds from a val_results.json, or None."""
    try:
        thr = json.loads(Path(path).read_text())["thresholds"]
        return {o: float(thr[o]) for o in OUTPUTS}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# scoring - pure, no model, fully testable
# ---------------------------------------------------------------------------
def classify_probabilities(p_fire: float, p_smoke: float, thresholds: dict,
                           inference_ms: float = 0.0) -> V.VisionResult:
    """
    Two sigmoid outputs -> a VisionResult on the same 0-100 scale as YOLO.

    An output counts only at or above its validated threshold. The score is
    the strongest counting output, weighted like vision_ml (smoke 0.75 of
    fire: smoke alone is the smouldering case, real but weaker evidence of
    flame). Below both thresholds the score is 0 - "the camera saw nothing",
    which the escalator rule treats as no evidence either way.
    """
    fire_on = p_fire >= thresholds["fire"]
    smoke_on = p_smoke >= thresholds["smoke"]
    score = 0.0
    if fire_on:
        score = max(score, p_fire * V.CLASS_WEIGHTS["fire"])
    if smoke_on:
        score = max(score, p_smoke * V.CLASS_WEIGHTS["smoke"])
    score = float(max(0.0, min(100.0, score * 100.0)))
    seen = [n for n, on in (("fire", fire_on), ("smoke", smoke_on)) if on]
    note = (f"maxvit fire {p_fire:.2f} smoke {p_smoke:.2f} "
            f"(thresholds {thresholds['fire']:.2f}/{thresholds['smoke']:.2f})"
            + (f": {' + '.join(seen)}" if seen else ""))
    return V.VisionResult(
        available=True,
        raw_detected=bool(seen),
        fire_detected=bool(seen),       # the channel applies the debounce
        score=score,
        has_fire_class=fire_on,
        has_smoke_class=smoke_on,
        detections=[],                  # a classifier has no boxes
        inference_ms=inference_ms,
        note=note,
    )


def filter_boxes(dets: list, thresholds: dict) -> list:
    """Keep detections at or above their own class's validated threshold.
    Unknown classes are dropped: a box we cannot name is not shown."""
    keep = []
    for d in dets:
        label = V.normalize_label(d.label)
        label = "fire" if label == "flame" else label
        if label in thresholds and d.confidence >= thresholds[label]:
            keep.append(d)
    return keep


# ---------------------------------------------------------------------------
# the model - a thin shell, same interface as vision_ml.YoloFireDetector
# ---------------------------------------------------------------------------
def pick_device(cuda_ok: bool, mps_ok: bool) -> str:
    """CUDA, then Apple MPS, then CPU. Pure, so the order can be tested on a
    machine that has neither accelerator."""
    if cuda_ok:
        return "cuda"
    if mps_ok:
        return "mps"
    return "cpu"


def default_device() -> str:
    """The best device this machine has. Without torch (or if a backend check
    itself fails) the answer is CPU, never an error."""
    try:
        import torch
    except Exception:
        return "cpu"
    try:
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False
    try:
        mps_ok = bool(torch.backends.mps.is_available())
    except Exception:
        mps_ok = False
    return pick_device(cuda_ok, mps_ok)


class MaxVitClassifier:
    """Loads models/maxvit_dfire/best.pt. Absent weights, thresholds, torch or
    timm, it reports itself unavailable instead of raising."""

    def __init__(self, weights: str | Path = MAXVIT_WEIGHTS,
                 device: Optional[str] = None):
        self.weights = Path(weights)
        self.device = device
        self.model = None
        self.thresholds: Optional[dict] = None
        self.error = ""
        self.class_names = {0: "fire", 1: "smoke"}
        self._mean = self._std = None
        self._size = 224

    def load(self) -> bool:
        if not self.weights.exists():
            self.error = (f"no MaxViT weights at {self.weights}. "
                          f"Train them with `python train_vision.py`.")
            return False
        self.thresholds = load_thresholds(self.weights.parent / "val_results.json")
        if self.thresholds is None:
            self.error = (f"no validated thresholds in "
                          f"{self.weights.parent / 'val_results.json'}; "
                          f"refusing to guess one")
            return False
        try:
            import torch
            import train_vision as T
        except Exception as exc:
            self.error = f"torch/timm not installed ({exc}). pip install -r requirements-ml.txt"
            return False
        try:
            ck = torch.load(self.weights, map_location="cpu", weights_only=False)
            if tuple(ck.get("outputs", ())) != OUTPUTS:
                self.error = f"checkpoint outputs {ck.get('outputs')} are not {OUTPUTS}"
                return False
            model = T.make_model(pretrained=False)
            model.load_state_dict(ck["model"])
            self.device = self.device or default_device()
            self.model = model.to(self.device).eval()
            self._mean, self._std = T._normalise(model)
            self._size = int(ck.get("size", T.SIZE))
            return True
        except Exception as exc:
            self.error = f"could not load the MaxViT checkpoint: {exc}"
            self.model = None
            return False

    def probabilities(self, frame: np.ndarray) -> tuple[float, float]:
        """BGR frame -> (p_fire, p_smoke). Same preprocessing as training:
        the whole frame squashed to 224x224 with PIL bicubic, not cropped."""
        import torch
        from PIL import Image

        import train_vision as T

        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        small = np.array(Image.fromarray(rgb).resize(   # writable copy for torch
            (self._size, self._size), Image.BICUBIC), dtype=np.uint8)
        with torch.no_grad():
            x = T.to_tensor(small[None], self._mean, self._std, self.device)
            p = torch.sigmoid(self.model(x)).float().cpu().numpy()[0]
        return float(p[0]), float(p[1])

    def detect(self, frame: np.ndarray) -> V.VisionResult:
        if self.model is None:
            return V.VisionResult(available=False, note=self.error or "model not loaded")
        if frame is None or getattr(frame, "size", 0) == 0:
            return V.VisionResult(available=False, note="empty frame")
        t0 = time.time()
        try:
            p_fire, p_smoke = self.probabilities(frame)
        except Exception as exc:
            return V.VisionResult(available=False, note=f"inference failed: {exc}")
        return classify_probabilities(p_fire, p_smoke, self.thresholds,
                                      inference_ms=(time.time() - t0) * 1000.0)


class MaxVitChannel(V.VisionChannel):
    """What the pipeline talks to. The detector is swapped for the classifier;
    the debounce (CONFIRM_FRAMES consecutive frames) is inherited unchanged."""

    def __init__(self, weights: str | Path = MAXVIT_WEIGHTS,
                 device: Optional[str] = None,
                 confirm_frames: int = V.CONFIRM_FRAMES):
        self.detector = MaxVitClassifier(weights, device=device)
        self.confirm_frames = max(1, confirm_frames)
        self._streak = 0
        self.available = False


# ---------------------------------------------------------------------------
# boxes for the alert photo - display only, never scored
# ---------------------------------------------------------------------------
class AlertBoxes:
    """Draws YOLO v2 boxes on the alert photo at its validated thresholds."""

    def __init__(self, weights: str | Path = YOLO_BOX_WEIGHTS,
                 device: Optional[str] = None):
        self.weights = Path(weights)
        # weights/best.pt -> the run folder holds val_results.json
        self.thresholds = load_thresholds(self.weights.parent.parent / "val_results.json")
        floor = min(self.thresholds.values()) if self.thresholds else V.CONF_THRESHOLD
        self.detector = V.YoloFireDetector(self.weights, conf=floor, device=device)
        self.error = ""

    def load(self) -> bool:
        if self.thresholds is None:
            self.error = (f"no validated thresholds next to {self.weights}; "
                          f"run `python train_yolo_v2.py --validate`")
            return False
        # Left to itself ultralytics picks CUDA or CPU and never MPS, so the
        # device is chosen here, by the same rule as the MaxViT channel.
        self.detector.device = self.detector.device or default_device()
        if not self.detector.load():
            self.error = self.detector.error
            return False
        return True

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Frame with boxes; the frame unchanged if detection fails."""
        res = self.detector.detect(frame)
        if not res.available:
            return frame
        keep = filter_boxes(res.detections, self.thresholds)
        h, w = frame.shape[:2]
        shown = V.build_result(keep, float(w * h))
        shown.available = False          # no "vision N/100" banner: not a score
        return V.annotate(frame, shown)
