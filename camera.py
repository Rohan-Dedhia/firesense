"""
FireSense - camera frame sources.

One interface, two implementations, so where the frames come from is a runtime
detail rather than something baked into the vision code:

    WebcamSource     - the laptop's own camera. Use this NOW, before the
                       ESP32-CAM is flashed, so hardware delivery never blocks
                       the vision work.
    Esp32CamSource   - an ESP32-CAM serving JPEGs over WiFi. Point it at the
                       /capture endpoint the stock CameraWebServer sketch
                       already provides.
    VideoFileSource  - a recorded video (.mp4 .mov .avi .mkv), played back at
                       real-time speed by default. For testing and demos
                       without the ESP32-CAM - NOT for the vision check
                       frames, which must come from the deployment camera.

Swapping between them is one line at the call site. Everything downstream -
recording, detection, fusion - sees the same BGR numpy frames either way.

    cam = make_camera("webcam:0")
    cam = make_camera("http://192.168.1.42/capture")
    cam = make_camera("demo.mp4", loop=True)
"""
from __future__ import annotations

import time
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
    CV2_ERROR = ""
except Exception as _exc:                                   # pragma: no cover
    cv2 = None
    CV2_AVAILABLE = False
    CV2_ERROR = str(_exc)


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------
@dataclass
class Frame:
    """One captured image, or a clear explanation of why there isn't one."""
    ok: bool
    image: Optional[np.ndarray] = None          # BGR, as OpenCV produces
    ts: float = 0.0                             # wall clock at capture
    error: str = ""
    end_of_stream: bool = False                 # a finished video, not a fault

    @property
    def shape(self) -> tuple:
        return self.image.shape if self.image is not None else ()


# ---------------------------------------------------------------------------
# base
# ---------------------------------------------------------------------------
class FrameSource:
    name = "base"

    def open(self) -> bool:
        raise NotImplementedError

    def read(self) -> Frame:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ---------------------------------------------------------------------------
# laptop webcam
# ---------------------------------------------------------------------------
class WebcamSource(FrameSource):
    """The laptop's built-in camera. No hardware, no network, no excuses."""

    name = "webcam"

    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self.index = index
        self.width = width
        self.height = height
        self.cap = None
        self.error = ""

    def open(self) -> bool:
        if not CV2_AVAILABLE:
            self.error = f"OpenCV not installed ({CV2_ERROR}). pip install opencv-python"
            return False
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            self.error = (f"camera index {self.index} would not open. Try another "
                          f"index, close any app already using the camera, and "
                          f"check the OS camera privacy setting.")
            self.cap = None
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return True

    def read(self) -> Frame:
        if self.cap is None:
            return Frame(ok=False, ts=time.time(),
                         error=self.error or "camera not opened")
        got, img = self.cap.read()
        if not got or img is None:
            return Frame(ok=False, ts=time.time(), error="frame grab failed")
        return Frame(ok=True, image=img, ts=time.time())

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ---------------------------------------------------------------------------
# ESP32-CAM over WiFi
# ---------------------------------------------------------------------------
class Esp32CamSource(FrameSource):
    """
    Pulls single JPEGs from an ESP32-CAM.

    The stock Arduino CameraWebServer sketch serves:
        http://<ip>/capture   one JPEG        <- this is what we use
        http://<ip>:81/stream MJPEG stream

    Polling /capture is slower than the MJPEG stream but far simpler and it
    recovers cleanly from the board dropping off WiFi, which a long-lived
    stream does not. At the frame rates fire detection needs, simple wins.
    """

    name = "esp32cam"

    def __init__(self, url: str, timeout: float = 4.0, retries: int = 2):
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        # A bare host gets the stock sketch's /capture endpoint. Anything with
        # a path of its own is left exactly as given - the caller may be
        # pointing at a custom endpoint, and silently rewriting it would be
        # both surprising and hard to debug.
        if urllib.parse.urlparse(url).path in ("", "/"):
            url = url.rstrip("/") + "/capture"
        self.url = url
        self.timeout = timeout
        self.retries = retries
        self.consecutive_failures = 0
        self.error = ""

    def open(self) -> bool:
        if not CV2_AVAILABLE:
            self.error = f"OpenCV not installed ({CV2_ERROR}). pip install opencv-python"
            return False
        probe = self.read()
        if not probe.ok:
            self.error = probe.error
        return probe.ok

    def _fetch(self) -> bytes:
        req = urllib.request.Request(self.url, headers={"User-Agent": "FireSense"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return r.read()

    def read(self) -> Frame:
        if not CV2_AVAILABLE:
            return Frame(ok=False, ts=time.time(), error=self.error)
        last = ""
        for attempt in range(self.retries + 1):
            try:
                raw = self._fetch()
                if not raw:
                    last = "empty response from the camera"
                    continue
                buf = np.frombuffer(raw, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is None:
                    last = f"response was not a decodable JPEG ({len(raw)} bytes)"
                    continue
                self.consecutive_failures = 0
                return Frame(ok=True, image=img, ts=time.time())
            except urllib.error.URLError as exc:
                last = f"cannot reach {self.url}: {exc.reason}"
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"
            if attempt < self.retries:
                time.sleep(0.25)
        self.consecutive_failures += 1
        return Frame(ok=False, ts=time.time(), error=last)


# ---------------------------------------------------------------------------
# recorded video
# ---------------------------------------------------------------------------
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")
END_OF_STREAM = "end of stream"
FALLBACK_FPS = 25.0                 # when a file does not report its own rate


def is_video_spec(spec: str) -> bool:
    """True when a camera spec names a video file (by extension)."""
    return Path((spec or "").strip()).suffix.lower() in VIDEO_EXTENSIONS


class VideoFileSource(FrameSource):
    """
    A video file played back as if it were a camera.

    realtime=True (default): the frame returned is the one at the current
    wall-clock position in the video, like a live camera. Reading slower than
    the video's frame rate skips frames; reading faster returns the same frame
    again, exactly as polling an ESP32-CAM /capture faster than the scene
    changes would. `fps` replaces the file's own rate on that clock: a
    30 fps file played with fps=10 runs three times slower than recorded.

    realtime=False: every read returns the next frame, as fast as it is
    called. For tools that want every frame (extract_frames.py, tests).

    At the end of the file it rewinds when loop=True; otherwise every read
    returns Frame(ok=False, end_of_stream=True, error="end of stream").
    """

    name = "video"

    def __init__(self, path: str | Path, loop: bool = False, realtime: bool = True,
                 fps: Optional[float] = None):
        self.path = Path(path).expanduser()
        self.loop = loop
        self.realtime = realtime
        self.fps_override = fps
        self.cap = None
        self.error = ""
        self.native_fps = 0.0
        self.play_fps = 0.0
        self.frame_count = 0
        self.ended = False
        self._pos = 0               # index of the next frame the decoder returns
        self._last = None           # last decoded frame (realtime repeats it)
        self._last_index = -1
        self._t0 = 0.0

    # -- open / close -----------------------------------------------------
    def _open_capture(self) -> bool:
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(str(self.path))
        return self.cap.isOpened()

    def open(self) -> bool:
        if not CV2_AVAILABLE:
            self.error = f"OpenCV not installed ({CV2_ERROR}). pip install opencv-python"
            return False
        if not self.path.is_file():
            self.error = f"video file not found: {self.path}"
            return False
        if self.fps_override is not None and not self.fps_override > 0:
            self.error = f"--camera-fps must be above 0 (got {self.fps_override})"
            return False
        if not self._open_capture():
            self.error = (f"OpenCV could not open {self.path} as a video "
                          f"(unsupported codec or not a video file)")
            self.close()
            return False
        # Opening is not enough: a corrupt file can open and yield nothing.
        got, img = self.cap.read()
        if not got or img is None:
            self.error = f"{self.path} opened but no frame could be decoded"
            self.close()
            return False
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.native_fps = fps if 0 < fps < 1000 else FALLBACK_FPS
        self.play_fps = float(self.fps_override or self.native_fps)
        self.frame_count = max(0, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        self._open_capture()        # back to frame 0; reopening beats seeking
        self._pos, self._last, self._last_index = 0, None, -1
        self.ended = False
        self._t0 = time.monotonic()
        return True

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # -- reading ----------------------------------------------------------
    def _rewind(self) -> bool:
        if not self._open_capture():
            return False
        self._pos, self._t0 = 0, time.monotonic()
        return True

    def _end(self) -> Frame:
        self.ended = True
        return Frame(ok=False, ts=time.time(), error=END_OF_STREAM,
                     end_of_stream=True)

    def _decode_next(self) -> Optional[np.ndarray]:
        got, img = self.cap.read()
        if not got or img is None:
            return None
        self._last, self._last_index = img, self._pos
        self._pos += 1
        return img

    def read(self) -> Frame:
        if self.cap is None:
            if self.ended:
                return self._end()
            return Frame(ok=False, ts=time.time(), error=self.error or "video not opened")
        if self.ended:
            return self._end()

        if self.realtime:
            target = int((time.monotonic() - self._t0) * self.play_fps)
            if self._last is not None and target <= self._last_index:
                # polled faster than the video runs: the scene has not moved on
                return Frame(ok=True, image=self._last.copy(), ts=time.time())
            while self._pos < target:           # skip, without decoding
                if not self.cap.grab():
                    break
                self._pos += 1

        img = self._decode_next()
        if img is None:                         # ran off the end of the file
            if not self.loop:
                return self._end()
            if not self._rewind() or self._decode_next() is None:
                self.error = f"could not rewind {self.path}"
                return Frame(ok=False, ts=time.time(), error=self.error)
            img = self._last
        return Frame(ok=True, image=img.copy(), ts=time.time())


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def make_camera(spec: str, *, loop: bool = False, fps: Optional[float] = None,
                realtime: bool = True) -> FrameSource:
    """
    Turn a string into a frame source.

        "webcam"                       -> laptop camera, index 0
        "webcam:1"                     -> laptop camera, index 1
        "192.168.1.42"                 -> ESP32-CAM, /capture appended
        "http://192.168.1.42/capture"  -> ESP32-CAM, exactly that URL
        "demo.mp4" (.mov .avi .mkv)    -> VideoFileSource

    loop / fps / realtime apply to video files only. A video spec is routed
    by its extension even if the file is missing, so open() can say "file not
    found" instead of trying to reach http://demo.mp4.
    """
    spec = (spec or "webcam").strip()
    if is_video_spec(spec):
        return VideoFileSource(spec, loop=loop, realtime=realtime, fps=fps)
    low = spec.lower()
    if low in ("webcam", "laptop", "usb"):
        return WebcamSource(0)
    if low.startswith(("webcam:", "usb:")):
        try:
            return WebcamSource(int(spec.split(":", 1)[1]))
        except ValueError:
            return WebcamSource(0)
    return Esp32CamSource(spec)


# ---------------------------------------------------------------------------
# manual check:  python camera.py [spec]
# ---------------------------------------------------------------------------
if __name__ == "__main__":                                  # pragma: no cover
    import sys

    spec = sys.argv[1] if len(sys.argv) > 1 else "webcam"
    print(f"opening {spec!r} ...")
    cam = make_camera(spec)
    if not cam.open():
        print(f"  FAILED: {getattr(cam, 'error', 'unknown error')}")
        raise SystemExit(1)

    print(f"  OK - source is {cam.name}")
    got = 0
    t0 = time.time()
    for i in range(10):
        f = cam.read()
        if f.ok:
            got += 1
            if got == 1:
                print(f"  first frame: {f.shape[1]}x{f.shape[0]}")
        else:
            print(f"  frame {i} failed: {f.error}")
        time.sleep(0.1)
    cam.close()
    dt = time.time() - t0
    print(f"  {got}/10 frames in {dt:.1f}s")
    print("  camera is working." if got >= 8 else "  camera is unreliable.")
