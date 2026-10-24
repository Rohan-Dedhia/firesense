"""
Tests that keep room photos and the vision check frames safe.

    1. data/vision_check_phone/ is git-ignored (photos of a private home).
    2. pick_vision_model.py --capture deletes the previous frames only after
       the camera has opened AND delivered a first frame. A failed camera
       must never leave an empty folder where a good capture used to be.
    3. evaluate_on_frames.py runs on any folder (--frames-dir), with the same
       report, and warns when that folder is not the ESP32-CAM check folder.

No weights, no camera, no model: the camera and the scorer are stubbed, and
the check folders are redirected to temporary directories.
"""
from __future__ import annotations

import io
import shutil
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import camera as cam_mod               # noqa: E402
import evaluate_on_frames as EV        # noqa: E402
import pick_vision_model as PVM        # noqa: E402

# Rule 9: module-level state a test changes is always put back.
_DEFAULTS = (PVM.CHECK_DIR, EV.CHECK_DIR, cam_mod.make_camera)


def _restore():
    PVM.CHECK_DIR, EV.CHECK_DIR, cam_mod.make_camera = _DEFAULTS


def _quiet(fn, *a, **kw):
    """Run fn with its printing captured -> (return value, printed text)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        rv = fn(*a, **kw)
    return rv, buf.getvalue()


# ---------------------------------------------------------------------------
# 1. .gitignore
# ---------------------------------------------------------------------------
def test_phone_frames_are_git_ignored():
    lines = [l.strip() for l in (REPO / ".gitignore").read_text().splitlines()]
    assert "data/vision_check_phone/" in lines
    if shutil.which("git") is None:
        return
    r = subprocess.run(["git", "check-ignore", "-q",
                        "data/vision_check_phone/candle/candle_000030.jpg"],
                       cwd=REPO, capture_output=True)
    if r.returncode == 128:              # not a git checkout (e.g. a zip copy)
        return
    assert r.returncode == 0, "git does not ignore data/vision_check_phone/"


# ---------------------------------------------------------------------------
# 2. capture never wipes frames on a failed camera
# ---------------------------------------------------------------------------
class _Cam:
    """open_ok / read_ok choose how the stub camera fails."""

    def __init__(self, open_ok=True, read_ok=True):
        self.open_ok, self.read_ok = open_ok, read_ok
        self.error = "" if open_ok else "stub camera unplugged"
        self.closed = False

    def open(self):
        return self.open_ok

    def read(self):
        if not self.read_ok:
            return cam_mod.Frame(ok=False, error="stub frame timeout")
        return cam_mod.Frame(ok=True, image=np.zeros((24, 32, 3), dtype=np.uint8))

    def close(self):
        self.closed = True


def _capture_with(cam, folder: Path):
    """Run capture('normal') into `folder` with `cam`; previous frames planted."""
    case_dir = folder / "normal"
    case_dir.mkdir(parents=True)
    old = [case_dir / f"{i:03d}.jpg" for i in range(3)]
    for f in old:
        f.write_bytes(b"previous good capture")
    try:
        PVM.CHECK_DIR = folder
        cam_mod.make_camera = lambda spec, **kw: cam
        rc, out = _quiet(PVM.capture, "normal", 2, "192.168.1.42", 0)
    finally:
        _restore()
    return rc, out, old


def test_a_camera_that_will_not_open_leaves_old_frames():
    with tempfile.TemporaryDirectory() as d:
        rc, out, old = _capture_with(_Cam(open_ok=False), Path(d))
        assert rc == 1
        assert all(f.exists() and f.read_bytes() == b"previous good capture" for f in old), "old frames were deleted"
        assert "left untouched" in out


def test_a_camera_whose_first_frame_fails_leaves_old_frames():
    with tempfile.TemporaryDirectory() as d:
        cam = _Cam(open_ok=True, read_ok=False)
        rc, out, old = _capture_with(cam, Path(d))
        assert rc == 1 and cam.closed
        assert all(f.exists() and f.read_bytes() == b"previous good capture" for f in old), "old frames were deleted"
        assert "first frame failed" in out and "left untouched" in out


def test_a_working_camera_replaces_the_old_frames():
    if not cam_mod.CV2_AVAILABLE:
        return
    with tempfile.TemporaryDirectory() as d:
        rc, out, old = _capture_with(_Cam(), Path(d))
        case_dir = Path(d) / "normal"
        new = sorted(case_dir.glob("*.jpg"))
        assert [f.name for f in new] == ["000.jpg", "001.jpg"]   # n = 2
        assert not (case_dir / "002.jpg").exists()               # old extra gone
        assert all(f.read_bytes() != b"previous good capture" for f in new)


def test_restore_puts_the_check_folder_back():
    PVM.CHECK_DIR = Path("elsewhere")
    _restore()
    assert PVM.CHECK_DIR == Path("data") / "vision_check"


# ---------------------------------------------------------------------------
# 3. evaluate_on_frames on another folder
# ---------------------------------------------------------------------------
class _Scorer:
    """Flame frames score 0.90 fire; normal frames 0.10. Thresholds as if
    validated. Paths are only looked at for their folder name."""
    thresholds = {"fire": 0.50, "smoke": 0.50}

    def score(self, paths):
        return np.array([[0.90, 0.05] if p.parent.name == "flame" else [0.10, 0.05]
                         for p in paths], dtype=np.float32)


def _frames(root: Path, n_normal=4, n_flame=4) -> Path:
    for case, n in (("normal", n_normal), ("flame", n_flame)):
        (root / case).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (root / case / f"{i:03d}.jpg").write_bytes(b"jpg")   # never decoded
    return root


def _report_body(text: str) -> list[str]:
    """The report lines that must not depend on which folder was used."""
    skip = ("WARNING", "NOT ESP32", "sensor, lens", "preview", "(CLAUDE.md",
            "frames from", "not the ESP32-CAM", "models/")
    return [l for l in text.splitlines() if l.strip() and not any(s in l for s in skip)]


def test_frames_dir_and_dir_are_the_same_option():
    p = EV.build_parser()
    assert p.parse_args(["--model", "m.pt"]).frames_dir == str(EV.CHECK_DIR)
    a = p.parse_args(["--model", "m.pt", "--frames-dir", "data/vision_check_phone/candle"])
    b = p.parse_args(["--model", "m.pt", "--dir", "data/vision_check_phone/candle"])
    assert a.frames_dir == b.frames_dir == "data/vision_check_phone/candle"


def test_a_phone_folder_gets_the_same_report_plus_a_warning():
    with tempfile.TemporaryDirectory() as d:
        check = _frames(Path(d) / "vision_check")
        phone = _frames(Path(d) / "vision_check_phone" / "candle")
        try:
            EV.CHECK_DIR = check
            rc_check, out_check = _quiet(EV.evaluate, "models/x.pt", check, "cpu",
                                         scorer=_Scorer())
            rc_phone, out_phone = _quiet(EV.evaluate, "models/x.pt", phone, "cpu",
                                         scorer=_Scorer())
        finally:
            _restore()
    assert rc_check == rc_phone == 0
    assert "WARNING" not in out_check and "not ESP32-CAM footage" not in out_check
    assert "NOT ESP32-CAM footage" in out_phone
    assert "not the ESP32-CAM check folder" in out_phone
    assert _report_body(out_check) == _report_body(out_phone)
    assert "lowest clean threshold" in out_phone and "separation" in out_phone


def test_a_phone_folder_without_sorted_frames_says_how_to_fix_it():
    with tempfile.TemporaryDirectory() as d:
        phone = _frames(Path(d) / "candle", n_flame=0)
        rc, out = _quiet(EV.evaluate, "models/x.pt", phone, "cpu", scorer=_Scorer())
    assert rc == 1
    assert "no frames for: flame" in out
    assert "sorted by hand" in out
    assert "pick_vision_model.py --capture" not in out   # that would write to the ESP32 folder


def test_evaluating_a_phone_folder_never_touches_the_check_folder():
    with tempfile.TemporaryDirectory() as d:
        check = _frames(Path(d) / "vision_check", n_normal=2, n_flame=2)
        before = sorted((p.name, p.read_bytes()) for p in check.rglob("*.jpg"))
        phone = _frames(Path(d) / "phone")
        try:
            EV.CHECK_DIR = check
            _quiet(EV.evaluate, "models/x.pt", phone, "cpu", scorer=_Scorer())
        finally:
            _restore()
        after = sorted((p.name, p.read_bytes()) for p in check.rglob("*.jpg"))
        assert before == after
