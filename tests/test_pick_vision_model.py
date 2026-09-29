"""
Tests for the vision-model chooser.

These matter because the chooser's whole job is to REJECT things, and a
rejector that silently accepts is worse than no rejector. So each test is one
kind of bad model that must not be recommended:

    noisy      false alarms at every threshold
    blind      quiet only because it detects nothing
    tightgap   clean today, one warm evening from useless
    good       the one that should win

No ultralytics, no weights, no camera - the detector is stubbed, because what
is being tested is the decision logic, not somebody else's model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pick_vision_model as P          # noqa: E402
import vision_ml as V                  # noqa: E402


# ---------------------------------------------------------------------------
# a stub detector whose character is chosen by name
# ---------------------------------------------------------------------------
CHARACTERS = {
    #            flame conf   normal-frame conf
    "good":     (0.85, 0.00),
    "noisy":    (0.78, 0.70),     # confident about nothing
    "lowconf":  (0.70, 0.30),     # usable, but needs a higher threshold
    "tightgap": (0.52, 0.46),     # separation 0.06
    "blind":    (0.00, 0.00),     # sees nothing at all
}


def _stub_detector(kind: str):
    flame_c, norm_c = CHARACTERS[kind]

    class Stub:
        def __init__(self, weights, conf=0.1, device=None):
            self.class_names = {0: "fire", 1: "smoke"}
            self.error = ""

        def load(self):
            return True

        def detect(self, frame):
            bright = bool(frame[:, :, 2].max() > 200)
            c = flame_c if bright else norm_c
            dets = ([V.Detection("fire", c, 10, 10, 110, 110)]
                    if c > 0 else [])
            return V.build_result(dets, float(frame.shape[0] * frame.shape[1]),
                                  inference_ms=10.0)

    return Stub


def _frames(n=12):
    """(normal, flame) frame lists. 'Bright' stands in for a real flame."""
    dark = np.full((120, 160, 3), 40, np.uint8)
    lit = dark.copy()
    lit[40:80, 60:100, 2] = 255
    return ([(f"n{i}", dark.copy()) for i in range(n)],
            [(f"f{i}", lit.copy()) for i in range(n)])


def _run(kind: str, monkeypatched: dict | None = None):
    """-> the summary row the chooser would build for one stubbed model."""
    normal, flame = _frames()
    real_detector, real_load = V.YoloFireDetector, P._load_frames
    try:
        V.YoloFireDetector = _stub_detector(kind)
        P._load_frames = lambda case: normal if case == "normal" else flame
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = P.compare(str(HERE / "_stub.pt"), None)
        return rc, buf.getvalue()
    finally:
        V.YoloFireDetector = real_detector
        P._load_frames = real_load


def _write_stub():
    # compare() globs for weights files; the stub detector never reads them.
    p = HERE / "_stub.pt"
    if not p.exists():
        p.write_bytes(b"PK stub")
    return p


# ---------------------------------------------------------------------------
# the tests
# ---------------------------------------------------------------------------
def test_a_clean_confident_model_is_recommended():
    _write_stub()
    rc, out = _run("good")
    assert rc == 0, out
    assert "USE" in out, out
    assert "CONF_THRESHOLD" in out, out


def test_a_model_that_alarms_at_every_threshold_is_rejected():
    _write_stub()
    rc, out = _run("noisy")
    assert rc == 2, f"expected rejection, got {rc}\n{out}"
    assert "Nothing here is usable" in out, out


def test_a_blind_model_is_rejected_even_though_it_never_false_alarms():
    """The failure mode that a false-positive-only check would miss."""
    _write_stub()
    rc, out = _run("blind")
    assert rc == 2, f"a model that detects nothing must not win\n{out}"


def test_a_model_with_no_separation_margin_is_rejected():
    """Clean on twelve frames is not the same as robust."""
    _write_stub()
    rc, out = _run("tightgap")
    assert rc == 2, f"separation 0.06 must not be recommended\n{out}"
    assert "separation" in out.lower(), out


def test_a_usable_model_needing_a_higher_threshold_is_accepted():
    _write_stub()
    rc, out = _run("lowconf")
    assert rc == 0, out
    assert "USE" in out, out


def test_the_recommended_threshold_is_clamped_into_a_sane_range():
    """worst_fp of 0.0 must not produce a recommendation near zero."""
    _write_stub()
    rc, out = _run("good")
    line = [l for l in out.splitlines() if "CONF_THRESHOLD" in l][0]
    value = float(line.split("=")[1].split()[0])
    assert 0.30 <= value <= 0.80, f"recommended {value}, out of range\n{line}"


def test_missing_frames_do_not_crash_and_say_what_to_do():
    real = P._load_frames
    try:
        P._load_frames = lambda case: []
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = P.compare("models/*.pt", None)
        assert rc == 1
        assert "--capture" in buf.getvalue()
    finally:
        P._load_frames = real
