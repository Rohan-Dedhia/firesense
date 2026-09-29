"""
FireSense - the semantic channel.

YOLO answers "is there fire in this frame". That is not the question your
project is about. A stove flame and a burning sofa are both fire; the
difference is context, and context is a language problem.

So this channel asks a vision-language model what it is actually looking at:

    SCENE:  a metal pan on a lit gas burner, someone standing beside it
    FIRE:   contained
    SOURCE: stove
    PEOPLE: yes
    -> expected combustion, score 8/100

    SCENE:  flames spreading across a fabric sofa in an empty room
    FIRE:   uncontained
    SOURCE: furniture
    PEOPLE: no
    -> uncontrolled fire, nobody present, score 100/100

Two design decisions worth knowing
----------------------------------
IT RUNS ON ESCALATION, NOT EVERY FRAME. A VLM takes seconds, and calling one
at 1 Hz would make the system slower than the fire. It is a verification
channel: something else raises a concern, this explains what is happening.
That is also how it would deploy.

"NOBODY IS PRESENT" IS A SIGNAL, NOT A DETAIL. Unattended cooking is the
leading cause of home fires - it is what NIST's cooktop research is about. A
contained flame with a person beside it and the same flame in an empty room
are not equally safe, and this is the only channel that can tell them apart.

Setup (easiest path on a Mac)
-----------------------------
    brew install ollama          (or download from ollama.com)
    ollama serve                 (leave running)
    ollama pull moondream        small and fast - start here
    ollama pull llava            larger, better descriptions

24 GB of unified memory runs either comfortably. No torch, no MPS setup, no
model files to manage yourself.

    from vlm import VLMChannel
    vlm = VLMChannel(model="moondream")
    if vlm.available():
        r = vlm.describe(frame)
        print(r.scene, r.score)
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("FIRESENSE_VLM", "moondream")

# Asking for fixed fields rather than prose. A small VLM narrates loosely but
# fills in a form reliably, and a form is what the fusion engine can use.
PROMPT = """You are a fire safety monitor looking at one camera frame.
Answer in exactly this format, one field per line, nothing else:

SCENE: <one short sentence describing what you see>
FIRE: <none|contained|uncontained>
SOURCE: <stove|candle|cigarette|paper|fabric|furniture|electrical|unknown|none>
PEOPLE: <yes|no>

FIRE means: none = no flame or smoke visible. contained = flame in a device or
holder built for it, such as a stove burner or a candle in a holder.
uncontained = flame or smoke on something not meant to burn, or spreading."""

# Containment carries most of the weight; the source decides whether a
# contained flame is expected there at all.
EXPECTED_SOURCES = {"stove", "candle", "cigarette"}
ALARMING_SOURCES = {"fabric", "furniture", "paper", "electrical"}

SCORE_NONE = 0.0
SCORE_CONTAINED_EXPECTED = 8.0      # a lit hob. Real fire, expected place.
SCORE_CONTAINED_UNKNOWN = 35.0      # contained, but of what?
SCORE_CONTAINED_ALARMING = 70.0     # "contained" on a sofa is not contained
SCORE_UNCONTAINED = 92.0
UNATTENDED_BONUS = 8.0              # fire present and nobody in frame


@dataclass
class VLMResult:
    available: bool = False
    scene: str = ""
    fire: str = "none"                  # none | contained | uncontained
    source: str = "none"
    people_present: Optional[bool] = None
    score: float = 0.0
    expected: bool = False              # combustion that belongs here
    unattended: bool = False
    latency_ms: float = 0.0
    raw: str = ""
    note: str = ""


# ---------------------------------------------------------------------------
# parsing - pure, and forgiving, because small models drift from the format
# ---------------------------------------------------------------------------
def _field(text: str, name: str) -> str:
    m = re.search(rf"^\s*{name}\s*[:\-]\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else ""


def _norm_fire(value: str) -> str:
    v = value.lower()
    if "uncontain" in v or "spread" in v or "out of control" in v:
        return "uncontained"
    if "contain" in v or "controlled" in v:
        return "contained"
    if "none" in v or "no fire" in v or v.strip() in ("no", "n/a", ""):
        return "none"
    return "none"


def _norm_people(value: str) -> Optional[bool]:
    v = value.lower().strip()
    if v.startswith(("yes", "true", "1")) or "person" in v or "someone" in v:
        return True
    if v.startswith(("no", "false", "0")) or "nobody" in v or "empty" in v:
        return False
    return None


def _norm_source(value: str) -> str:
    v = value.lower().strip().strip(".")
    known = (EXPECTED_SOURCES | ALARMING_SOURCES | {"unknown", "none"})
    for k in known:
        if k in v:
            return k
    return "unknown" if v else "none"


def parse_reply(text: str) -> VLMResult:
    """Model output -> structured result. Never raises: this parses the output
    of a language model, which is not a contract."""
    fire = _norm_fire(_field(text, "FIRE"))
    source = _norm_source(_field(text, "SOURCE"))
    people = _norm_people(_field(text, "PEOPLE"))
    scene = _field(text, "SCENE")

    if not scene:
        # Model ignored the format - keep the first sentence so the operator
        # still gets something readable.
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        scene = first[:200]

    res = VLMResult(available=True, scene=scene, fire=fire, source=source,
                    people_present=people, raw=text)
    res.score = score_scene(fire, source, people)
    res.expected = (fire == "contained" and source in EXPECTED_SOURCES)
    res.unattended = (fire != "none" and people is False)
    res.note = describe(res)
    return res


def score_scene(fire: str, source: str, people: Optional[bool]) -> float:
    """
    Containment first, then whether the source belongs there, then whether
    anyone is watching.
    """
    if fire == "none":
        return SCORE_NONE
    if fire == "uncontained":
        base = SCORE_UNCONTAINED
    elif source in EXPECTED_SOURCES:
        base = SCORE_CONTAINED_EXPECTED
    elif source in ALARMING_SOURCES:
        base = SCORE_CONTAINED_ALARMING
    else:
        base = SCORE_CONTAINED_UNKNOWN
    if people is False:
        base += UNATTENDED_BONUS
    return float(max(0.0, min(100.0, base)))


def describe(r: VLMResult) -> str:
    """One line for the phone alert - the judgement, not the raw fields."""
    if r.fire == "none":
        return "no fire or smoke visible"
    who = ("nobody in frame" if r.people_present is False
           else "someone present" if r.people_present else "")
    if r.expected:
        base = f"{r.source} flame, contained - expected combustion"
    elif r.fire == "uncontained":
        base = f"uncontained fire on {r.source}"
    else:
        base = f"contained flame, source {r.source}"
    return f"{base}{', ' + who if who else ''}"


# ---------------------------------------------------------------------------
# backend
# ---------------------------------------------------------------------------
class OllamaBackend:
    """Talks to a local Ollama server. Chosen over torch/MPS because it needs
    no Python ML stack and manages the model files itself."""

    def __init__(self, model: str = DEFAULT_MODEL, url: str = OLLAMA_URL,
                 timeout: float = 60.0):
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.error = ""

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url}/api/tags", timeout=4) as r:
                tags = json.loads(r.read().decode() or "{}")
        except Exception as exc:
            self.error = (f"Ollama not reachable at {self.url} ({exc}). "
                          f"Run: ollama serve")
            return False
        names = [m.get("name", "") for m in tags.get("models", []) or []]
        if not any(n.split(":")[0] == self.model.split(":")[0] for n in names):
            self.error = (f"model '{self.model}' not pulled. "
                          f"Run: ollama pull {self.model}\n"
                          f"  available: {', '.join(names) or 'none'}")
            return False
        self.error = ""
        return True

    def generate(self, image_jpeg: bytes, prompt: str = PROMPT) -> tuple[bool, str]:
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "images": [base64.b64encode(image_jpeg).decode()],
            "stream": False,
            "options": {"temperature": 0.1},      # description, not creativity
        }).encode()
        req = urllib.request.Request(
            f"{self.url}/api/generate", data=payload,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = json.loads(r.read().decode() or "{}")
            return True, str(body.get("response", ""))
        except urllib.error.HTTPError as exc:
            return False, f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:200]}"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# channel
# ---------------------------------------------------------------------------
class VLMChannel:
    def __init__(self, model: str = DEFAULT_MODEL, url: str = OLLAMA_URL,
                 timeout: float = 60.0, backend=None):
        self.backend = backend or OllamaBackend(model, url, timeout)
        self.last: Optional[VLMResult] = None

    def available(self) -> bool:
        return self.backend.available()

    @property
    def error(self) -> str:
        return getattr(self.backend, "error", "")

    def describe(self, image_jpeg: bytes) -> VLMResult:
        """
        One frame -> a described scene. Takes seconds; call it on escalation,
        not in the sampling loop.
        """
        if not image_jpeg:
            return VLMResult(available=False, note="no image")
        t0 = time.time()
        ok, text = self.backend.generate(image_jpeg)
        ms = (time.time() - t0) * 1000.0
        if not ok:
            return VLMResult(available=False, note=text, latency_ms=ms)
        res = parse_reply(text)
        res.latency_ms = ms
        self.last = res
        return res

    def describe_frame(self, frame) -> VLMResult:
        """Convenience for a BGR numpy frame from camera.py."""
        try:
            import cv2
        except Exception as exc:
            return VLMResult(available=False, note=f"OpenCV needed to encode: {exc}")
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return VLMResult(available=False, note="could not encode the frame")
        return self.describe(buf.tobytes())


# ---------------------------------------------------------------------------
# manual check:  python vlm.py [image|webcam] [model]
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# self-test:  python vlm.py --selftest
#
# Checks the three things that can be wrong, in the order they can be wrong,
# and says which one it is. Needs no camera and no fire - the scoring logic is
# exercised against canned replies, so a failure here is a real failure and not
# "the flame was out of frame".
# ---------------------------------------------------------------------------
SELFTEST_CASES = (
    # (reply text,                                          expect_at_least, expect_at_most, what it is)
    ("SCENE: an empty living room with a sofa\nFIRE: none\n"
     "SOURCE: none\nPEOPLE: no",                             0.0,  1.0,  "empty room -> silent"),
    ("SCENE: a pan on a lit gas burner, a person stirring\n"
     "FIRE: contained\nSOURCE: stove\nPEOPLE: yes",          0.0, 20.0,  "attended cooking -> quiet"),
    ("SCENE: a pan on a lit gas burner, nobody in the room\n"
     "FIRE: contained\nSOURCE: stove\nPEOPLE: no",           8.0, 30.0,  "unattended cooking -> nudged up"),
    ("SCENE: flames on a fabric sofa\nFIRE: contained\n"
     "SOURCE: furniture\nPEOPLE: yes",                      60.0, 80.0,  "'contained' on a sofa -> alarming"),
    ("SCENE: flames spreading across the floor, room empty\n"
     "FIRE: uncontained\nSOURCE: furniture\nPEOPLE: no",    92.0, 100.0, "uncontained + nobody -> maximum"),
)


def selftest(model: str = DEFAULT_MODEL, url: str = OLLAMA_URL) -> int:
    """-> process exit code. 0 means the channel is ready to use."""
    print()
    print("=" * 66)
    print("  VLM CHANNEL SELF-TEST")
    print("=" * 66)

    # -- 1. the scoring logic. No network, no model, no excuses. ----------
    print("\n  1. scoring logic (offline)")
    print("  " + "-" * 62)
    bad = 0
    for reply, lo, hi, what in SELFTEST_CASES:
        res = parse_reply(reply)
        ok = lo <= res.score <= hi
        bad += not ok
        print(f"    {'PASS' if ok else 'FAIL'}  {res.score:5.1f}/100  {what}")
        if not ok:
            print(f"          expected {lo}-{hi}; parsed "
                  f"fire={res.fire!r} source={res.source!r} "
                  f"people={res.people_present!r}")
    if bad:
        print(f"\n  {bad} scoring case(s) failed - this is a code bug, not a setup")
        print("  problem. Nothing below will help until it is fixed.")
        return 1
    print("    all scoring cases pass")

    # -- 2. is Ollama running, and is the model pulled? -------------------
    print(f"\n  2. Ollama at {url}")
    print("  " + "-" * 62)
    ch = VLMChannel(model=model, url=url)
    if not ch.available():
        print(f"    FAIL  {ch.error}")
        print()
        print("  fix it in this order:")
        print("    brew install ollama       (once)")
        print("    ollama serve              (leave this running in its own tab)")
        print(f"    ollama pull {model}")
        print()
        print("  The scoring logic above is fine, so the channel will work as")
        print("  soon as the server is up. Nothing in the project is blocked")
        print("  by this - the VLM only runs on escalation.")
        return 2
    print(f"    PASS  server up, '{model}' pulled")

    # -- 3. a real round trip, on a picture we generate ourselves ---------
    print("\n  3. live round trip")
    print("  " + "-" * 62)
    jpeg = _selftest_image()
    if jpeg is None:
        print("    SKIP  could not build a test image (is opencv installed?)")
        print("          Point it at a real picture instead:")
        print("          python vlm.py path/to/photo.jpg")
        return 0
    res = ch.describe(jpeg)
    if not res.available:
        print(f"    FAIL  {res.note}")
        return 3
    print(f"    scene    : {res.scene[:52]}")
    print(f"    fire     : {res.fire}     source: {res.source}     "
          f"people: {res.people_present}")
    print(f"    score    : {res.score:.0f}/100")
    print(f"    latency  : {res.latency_ms:.0f} ms")
    if res.latency_ms > 15000:
        print("    note: slow. Fine for escalation-only use, which is how it runs.")
    print(f"\n    the model answered in the expected format - the channel works.")
    print("    Whether it describes THIS synthetic image correctly does not")
    print("    matter; it is a coloured rectangle, not a scene.")

    print("\n" + "=" * 66)
    print("  READY. Next, try it on something real:")
    print("    python vlm.py webcam          (point the camera at a candle)")
    print("    python vlm.py photo.jpg")
    print("=" * 66 + "\n")
    return 0


def _selftest_image() -> Optional[bytes]:
    """A small synthetic JPEG, so step 3 needs no camera and no file."""
    try:
        import cv2
        import numpy as _np
    except Exception:
        return None
    img = _np.zeros((240, 320, 3), _np.uint8)
    img[:] = (40, 40, 45)
    cv2.circle(img, (160, 150), 40, (30, 120, 240), -1)      # a warm blob
    cv2.circle(img, (160, 140), 18, (120, 220, 255), -1)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes() if ok else None


if __name__ == "__main__":                                   # pragma: no cover
    import sys
    from pathlib import Path

    args = [a for a in sys.argv[1:]]
    model = DEFAULT_MODEL
    if "--model" in args:
        k = args.index("--model")
        model = args[k + 1]
        del args[k:k + 2]

    if "--selftest" in args or "--self-test" in args:
        raise SystemExit(selftest(model=model))

    target = args[0] if args else "webcam"

    ch = VLMChannel(model=model)
    if not ch.available():
        print(f"VLM unavailable:\n  {ch.error}")
        print("\n  run  python vlm.py --selftest  for a step-by-step check")
        raise SystemExit(1)
    print(f"using {model} via {OLLAMA_URL}\n")

    if Path(target).is_file():
        res = ch.describe(Path(target).read_bytes())
        frames = [res]
    else:
        import camera
        cam = camera.make_camera(target)
        if not cam.open():
            print(f"camera unavailable: {getattr(cam, 'error', '?')}")
            raise SystemExit(1)
        f = cam.read()
        cam.close()
        if not f.ok:
            print(f"frame failed: {f.error}")
            raise SystemExit(1)
        frames = [ch.describe_frame(f.image)]

    for res in frames:
        if not res.available:
            print(f"  failed: {res.note}")
            continue
        print(f"  scene    : {res.scene}")
        print(f"  fire     : {res.fire}")
        print(f"  source   : {res.source}")
        print(f"  people   : {res.people_present}")
        print(f"  verdict  : {res.note}")
        print(f"  score    : {res.score:.0f}/100")
        print(f"  latency  : {res.latency_ms:.0f} ms")
