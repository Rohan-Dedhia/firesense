"""
FireSense - optical channel: HSV fire-colour detection.

This is RULE-BASED computer vision, not machine learning. There is no model,
no training data and no dataset: every frame is converted to HSV colour space
and we measure what fraction of pixels fall inside the flame colour band.

Report wording (use this, do not overclaim):
    "Visual fire detection is performed by rule-based HSV colour-space
     thresholding, not a trained image classifier. It requires no training
     data, runs at full frame rate on a laptop CPU, and its decision is fully
     explainable - the reported confidence is literally the proportion of the
     frame occupying the flame colour band."

Frame ACQUISITION and frame ANALYSIS are deliberately separate functions, so
migrating from the laptop webcam to an ESP32-CAM later needs no change to the
detection or fusion pipeline - only a different get_frame_* function.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import config as C

try:
    import cv2
    import numpy as np
    CV_AVAILABLE = True
except Exception:                                    # pragma: no cover
    cv2 = None
    np = None
    CV_AVAILABLE = False


@dataclass
class VisualResult:
    available: bool = False
    fire_detected: bool = False
    confidence: float = 0.0     # 0-100
    pixel_ratio: float = 0.0    # raw fraction of frame in the flame band
    note: str = ""


# ---------------------------------------------------------------------------
# FRAME SOURCES  (swappable - this is the seam for the future camera module)
# ---------------------------------------------------------------------------
class WebcamSource:
    """Laptop / Surface built-in camera."""

    def __init__(self, index: int = C.CAMERA_INDEX):
        self.index = index
        self.cap = None

    def open(self) -> bool:
        if not CV_AVAILABLE:
            return False
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
        if not self.cap or not self.cap.isOpened():
            # CAP_DSHOW is a Windows backend; fall back to the default one
            self.cap = cv2.VideoCapture(self.index)
        return bool(self.cap and self.cap.isOpened())

    def get_frame(self):
        if not self.cap:
            return None
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None


class Esp32CamSource:
    """
    Future drop-in: ESP32-CAM serving an MJPEG/JPEG endpoint over WiFi.
    Detection code below is untouched by this swap - that is the whole point.
    """

    def __init__(self, url: str):
        self.url = url

    def open(self) -> bool:
        return CV_AVAILABLE

    def get_frame(self):                             # pragma: no cover
        import urllib.request
        with urllib.request.urlopen(self.url, timeout=2) as resp:
            buf = np.asarray(bytearray(resp.read()), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def release(self):
        pass


# ---------------------------------------------------------------------------
# DETECTION  (source-agnostic: give it any frame array)
# ---------------------------------------------------------------------------
def detect_fire(frame, lower=None, upper=None, trigger=None) -> VisualResult:
    if not CV_AVAILABLE or frame is None:
        return VisualResult(available=False, note="no frame / OpenCV unavailable")

    lower = np.array(lower if lower is not None else C.HSV_LOWER, dtype=np.uint8)
    upper = np.array(upper if upper is not None else C.HSV_UPPER, dtype=np.uint8)
    trigger = C.FIRE_PIXEL_RATIO_TRIGGER if trigger is None else trigger

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    # Morphological opening kills isolated hot pixels (sensor noise, specular
    # highlights) so a single bright speck cannot trip the alarm.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    ratio = float(np.count_nonzero(mask)) / float(mask.size)
    conf = min(100.0, ratio / max(C.VISION_CONFIDENCE_FULL_SCALE, 1e-6) * 100.0)
    return VisualResult(
        available=True,
        fire_detected=ratio >= trigger,
        confidence=round(conf, 1),
        pixel_ratio=round(ratio, 5),
        note=f"{ratio*100:.2f} % of frame in flame colour band",
    )


def annotate(frame, result: VisualResult, lower=None, upper=None):
    """Return a display frame with the detected region outlined (for the UI)."""
    if not CV_AVAILABLE or frame is None:
        return None
    lower = np.array(lower if lower is not None else C.HSV_LOWER, dtype=np.uint8)
    upper = np.array(upper if upper is not None else C.HSV_UPPER, dtype=np.uint8)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    out = frame.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 200:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
    colour = (0, 0, 255) if result.fire_detected else (0, 200, 0)
    cv2.putText(out, f"fire {result.pixel_ratio*100:.2f}%  conf {result.confidence:.0f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    return out


class VisionChannel:
    """Convenience wrapper the dashboard uses; degrades gracefully if no camera."""

    def __init__(self, source=None):
        self.source = source or WebcamSource()
        self.ready = False
        self.last = VisualResult()

    def start(self) -> bool:
        self.ready = self.source.open()
        if not self.ready:
            self.last = VisualResult(available=False, note="camera not available")
        return self.ready

    def poll(self, want_frame: bool = False):
        if not self.ready:
            return self.last, None
        frame = self.source.get_frame()
        self.last = detect_fire(frame)
        return self.last, (annotate(frame, self.last) if want_frame else None)

    def stop(self):
        self.source.release()
        self.ready = False
