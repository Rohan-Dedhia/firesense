"""
FireSense - the learned vision channel.

Replaces the HSV colour thresholding that failed measurably in our own demo
room (the auto-tuner could not separate flame from a warm-coloured background;
separation margin stuck at 1.0x). A trained detector does not care that the
desk is orange.

    from vision_ml import VisionChannel

    vision = VisionChannel("models/fire_yolo.pt")
    vision.load()
    result = vision.update(frame)        # frame is BGR, from camera.py
    result.score                         # 0-100, feeds the fusion engine

Getting a model
---------------
No training required - use a pretrained fire/smoke detector:

  1. roboflow.com/universe -> search "fire smoke detection"
  2. pick a model with a decent mAP and both `fire` and `smoke` classes
  3. download the YOLOv8/v11 weights (.pt) into models/

Then:  pip install ultralytics

On Apple Silicon pass device="mps" for GPU acceleration.

Design notes
------------
SMOKE IS NOT A LESSER FIRE. A Roboflow model usually exposes `fire` and
`smoke` separately, and we keep them separate: smoke with no flame is the
smouldering case - our hardest one, the one the sensors nearly missed - so it
gets its own weight rather than being folded into a single "fire" score.

CONFIDENCE LEADS, AREA MODULATES. Under HSV, area *was* the signal. With a
detector the confidence is the signal; a small flame is still a flame, so area
only scales the score between half and full rather than driving it.

THIS CHANNEL CANNOT SAY "SAFE". Like the IR sensor, it is an escalator: seeing
no fire means the camera saw no fire, not that there is none. It returns 0 and
lets the fusion engine decide what that is worth.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Class weighting. Flame visible is stronger evidence than smoke alone, but
# smoke alone is exactly the smouldering signature, so it stays substantial.
CLASS_WEIGHTS = {
    "fire": 1.0,
    "flame": 1.0,
    "smoke": 0.75,
}
DEFAULT_CLASS_WEIGHT = 0.6          # an unfamiliar class name, treated cautiously

# Community fire models are trained all over the world and their class names
# come back in the trainer's language. A model that labels flames 火 is not an
# unfamiliar class, it is the same class in Chinese, and treating it as
# unknown would quietly discount every real detection it makes.
LABEL_ALIASES = {
    # Chinese
    "火": "fire", "火焰": "fire", "明火": "fire", "火灾": "fire",
    "烟": "smoke", "烟雾": "smoke", "煙": "smoke", "煙霧": "smoke",
    # common English variants
    "fire ": "fire", "flames": "fire", "wildfire": "fire",
    "smoking": "smoke", "smoke ": "smoke",
}


def normalize_label(raw: str) -> str:
    """Map whatever the model calls a class onto fire / smoke where possible."""
    v = (raw or "").strip().lower()
    if v in CLASS_WEIGHTS:
        return v
    return LABEL_ALIASES.get(v, LABEL_ALIASES.get((raw or "").strip(), v))

# Community fire models are trained on obvious fires and are noticeably
# trigger-happy on ordinary indoor scenes. Measured on this project's own
# webcam with nothing burning: persistent detections at 0.40-0.45. So the
# floor sits above that, and you should check it against YOUR room - see the
# --conf option on the CLI below.
CONF_THRESHOLD = 0.55               # below this a detection is not reported
AREA_FULL_SCALE = 0.06              # 6 % of frame counts as a "large" detection
AREA_FLOOR = 0.5                    # a tiny detection keeps half its weight
MULTI_DETECTION_BONUS = 4.0         # per extra detection
MULTI_DETECTION_CAP = 12.0

# A real fire persists. One frame is a glitch, a reflection, or a person
# walking past a screen - the same reasoning as FLAME_CONFIRM_SAMPLES.
CONFIRM_FRAMES = 2


# ---------------------------------------------------------------------------
# pure data
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass
class VisionResult:
    available: bool = False
    fire_detected: bool = False         # after debounce
    raw_detected: bool = False          # this frame alone
    score: float = 0.0                  # 0-100
    has_fire_class: bool = False
    has_smoke_class: bool = False
    detections: list = field(default_factory=list)
    inference_ms: float = 0.0
    note: str = ""


# ---------------------------------------------------------------------------
# scoring - pure, no model, fully testable
# ---------------------------------------------------------------------------
def _area_factor(area_fraction: float) -> float:
    """Small detections keep AREA_FLOOR of their weight; large ones get all."""
    if area_fraction <= 0:
        return AREA_FLOOR
    scaled = min(1.0, area_fraction / AREA_FULL_SCALE)
    return AREA_FLOOR + (1.0 - AREA_FLOOR) * scaled


def score_detections(dets: list[Detection], frame_area: float) -> float:
    """
    Detections -> 0-100.

    The strongest single detection sets the score; additional ones add a small
    bonus, because two independent detections of fire is better evidence than
    one, but not twice as much.
    """
    if not dets or frame_area <= 0:
        return 0.0

    best = 0.0
    for d in dets:
        weight = CLASS_WEIGHTS.get(normalize_label(d.label), DEFAULT_CLASS_WEIGHT)
        best = max(best, d.confidence * weight * _area_factor(d.area / frame_area))

    score = best * 100.0
    if len(dets) > 1:
        score += min(MULTI_DETECTION_CAP, MULTI_DETECTION_BONUS * (len(dets) - 1))
    return float(max(0.0, min(100.0, score)))


# ---------------------------------------------------------------------------
# the model - a thin shell around ultralytics
# ---------------------------------------------------------------------------
class YoloFireDetector:
    """Wraps a pretrained YOLO fire/smoke model. Absent ultralytics or weights,
    it reports itself unavailable rather than raising - the rest of the system
    runs on chemical + thermal + IR exactly as it does today."""

    def __init__(self, weights: str | Path = "models/fire_yolo.pt",
                 conf: float = CONF_THRESHOLD, device: Optional[str] = None,
                 imgsz: int = 640):
        self.weights = Path(weights)
        self.conf = conf
        self.device = device
        self.imgsz = imgsz
        self.model = None
        self.error = ""
        self.class_names: dict = {}

    def load(self) -> bool:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            self.error = (f"ultralytics not installed ({exc}). "
                          f"pip install ultralytics")
            return False
        if not self.weights.exists():
            self.error = (f"no weights at {self.weights}. Download a pretrained "
                          f"fire/smoke model from roboflow.com/universe into "
                          f"models/ - see the module docstring.")
            return False
        try:
            self.model = YOLO(str(self.weights))
            self.class_names = dict(getattr(self.model, "names", {}) or {})
            return True
        except Exception as exc:
            self.error = f"could not load the model: {exc}"
            return False

    def detect(self, frame: np.ndarray) -> VisionResult:
        if self.model is None:
            return VisionResult(available=False, note=self.error or "model not loaded")
        if frame is None or getattr(frame, "size", 0) == 0:
            return VisionResult(available=False, note="empty frame")

        t0 = time.time()
        try:
            kw = {"conf": self.conf, "imgsz": self.imgsz, "verbose": False}
            if self.device:
                kw["device"] = self.device
            results = self.model(frame, **kw)
        except Exception as exc:
            return VisionResult(available=False, note=f"inference failed: {exc}")
        ms = (time.time() - t0) * 1000.0

        dets: list[Detection] = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for b in boxes:
                try:
                    cls_id = int(b.cls[0])
                    conf = float(b.conf[0])
                    x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                except Exception:
                    continue
                label = self.class_names.get(cls_id, str(cls_id))
                dets.append(Detection(label, conf, x1, y1, x2, y2))

        h, w = frame.shape[:2]
        return build_result(dets, float(w * h), inference_ms=ms)


def build_result(dets: list[Detection], frame_area: float,
                 inference_ms: float = 0.0) -> VisionResult:
    """Detections -> a VisionResult. Separate from the model so it can be
    tested without one."""
    labels = {normalize_label(d.label) for d in dets}
    score = score_detections(dets, frame_area)
    has_fire = bool(labels & {"fire", "flame"})
    has_smoke = "smoke" in labels
    if not dets:
        note = "no fire or smoke detected"
    else:
        note = ", ".join(f"{d.label} {d.confidence:.2f}" for d in dets[:4])
    return VisionResult(
        available=True,
        raw_detected=bool(dets),
        fire_detected=bool(dets),        # the channel applies the debounce
        score=score,
        has_fire_class=has_fire,
        has_smoke_class=has_smoke,
        detections=dets,
        inference_ms=inference_ms,
        note=note,
    )


# ---------------------------------------------------------------------------
# the channel - adds debounce, matching the IR sensor's behaviour
# ---------------------------------------------------------------------------
class VisionChannel:
    """
    What the fusion engine talks to.

    Holds the frame-to-frame state the detector itself is stateless about: a
    detection is only trusted once it has persisted for CONFIRM_FRAMES, and a
    single isolated hit is reported as noise rather than acted on.
    """

    def __init__(self, weights: str | Path = "models/fire_yolo.pt",
                 conf: float = CONF_THRESHOLD, device: Optional[str] = None,
                 confirm_frames: int = CONFIRM_FRAMES):
        self.detector = YoloFireDetector(weights, conf=conf, device=device)
        self.confirm_frames = max(1, confirm_frames)
        self._streak = 0
        self.available = False

    def load(self) -> bool:
        self.available = self.detector.load()
        return self.available

    @property
    def error(self) -> str:
        return self.detector.error

    def update(self, frame: np.ndarray) -> VisionResult:
        res = self.detector.detect(frame)
        if not res.available:
            self._streak = 0
            return res
        return self.apply_debounce(res)

    def apply_debounce(self, res: VisionResult) -> VisionResult:
        """Split out so the debounce can be tested without a model."""
        self._streak = self._streak + 1 if res.raw_detected else 0
        confirmed = self._streak >= self.confirm_frames
        res.fire_detected = confirmed
        if not confirmed:
            if res.raw_detected:
                res.note = (f"single unconfirmed detection ({res.note}) - "
                            f"treated as noise until {self.confirm_frames} "
                            f"consecutive frames agree")
            res.score = 0.0
        return res

    def reset(self) -> None:
        self._streak = 0


# ---------------------------------------------------------------------------
# drawing, for the dashboard and the demo
# ---------------------------------------------------------------------------
def annotate(frame: np.ndarray, res: VisionResult) -> np.ndarray:
    """Boxes and labels on a copy of the frame. Returns the frame unchanged if
    OpenCV is missing."""
    try:
        import cv2
    except Exception:
        return frame
    out = frame.copy()
    for d in res.detections:
        colour = ((0, 0, 220) if normalize_label(d.label) in ("fire", "flame")
                  else (0, 170, 220))
        p1 = (int(d.x1), int(d.y1))
        p2 = (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, colour, 2)
        cv2.putText(out, f"{d.label} {d.confidence:.2f}",
                    (p1[0], max(14, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    if res.available:
        banner = f"vision {res.score:.0f}/100"
        if not res.fire_detected and res.raw_detected:
            banner += "  (unconfirmed)"
        cv2.putText(out, banner, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# manual check:  python vision_ml.py [webcam | path/to/image | esp32 url]
# ---------------------------------------------------------------------------
if __name__ == "__main__":                                  # pragma: no cover
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "webcam"
    weights = sys.argv[2] if len(sys.argv) > 2 else "models/fire_yolo.pt"
    conf = float(sys.argv[3]) if len(sys.argv) > 3 else CONF_THRESHOLD

    ch = VisionChannel(weights, conf=conf, device=None)
    print(f"confidence threshold {conf}")
    if not ch.load():
        print(f"vision channel unavailable:\n  {ch.error}")
        raise SystemExit(1)
    print(f"loaded {weights}")
    print(f"classes: {ch.detector.class_names}")

    if Path(target).is_file():
        import cv2
        frame = cv2.imread(target)
        for i in range(ch.confirm_frames):          # satisfy the debounce
            res = ch.update(frame)
        print(f"  score {res.score:.0f}/100   {res.note}")
        out = annotate(frame, res)
        cv2.imwrite("vision_check.jpg", out)
        print("  annotated -> vision_check.jpg")
    else:
        import camera
        cam = camera.make_camera(target)
        if not cam.open():
            print(f"camera unavailable: {getattr(cam, 'error', '?')}")
            raise SystemExit(1)
        print("  20 frames, Ctrl+C to stop early")
        try:
            for i in range(20):
                f = cam.read()
                if not f.ok:
                    print(f"  frame failed: {f.error}")
                    continue
                res = ch.update(f.image)
                flag = "FIRE" if res.fire_detected else "   -"
                print(f"  {i:2d} {flag}  {res.score:5.1f}/100  "
                      f"{res.inference_ms:5.0f} ms  {res.note}")
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        cam.close()
