"""
Tests for the MaxViT vision channel and its place in the pipeline.

No weights, no torch, no timm, no ultralytics, no camera: the classifier and
the camera are stubbed. What is tested is the logic around the model - how
two probabilities become a score, the debounce, refusing to run without
validated thresholds, and above all the escalator rule:

    vision can only RAISE the level, and on its own it cannot reach CRITICAL.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import config as C                     # noqa: E402
import ingest                          # noqa: E402
import pipeline as P                   # noqa: E402
import vision_maxvit as M              # noqa: E402
import vision_ml as V                  # noqa: E402

THR = {"fire": 0.36, "smoke": 0.88}    # the shape of val_results.json thresholds
FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# stubs
# ---------------------------------------------------------------------------
class StubClassifier:
    """Stands in for MaxVitClassifier: returns fixed probabilities."""

    def __init__(self, p_fire: float, p_smoke: float):
        self.p = (p_fire, p_smoke)
        self.thresholds = dict(THR)
        self.error = ""

    def load(self):
        return True

    def detect(self, frame):
        return M.classify_probabilities(*self.p, self.thresholds)


def stub_channel(p_fire: float, p_smoke: float) -> M.MaxVitChannel:
    ch = M.MaxVitChannel("unused.pt")
    ch.detector = StubClassifier(p_fire, p_smoke)
    ch.available = True
    return ch


class StubFrame:
    ok, error = True, ""

    def __init__(self):
        self.image = FRAME


class StubCamera:
    def open(self):
        return True

    def read(self):
        return StubFrame()

    def close(self):
        pass


class StubCameraModule:
    @staticmethod
    def make_camera(spec):
        return StubCamera()


def quiet_pipeline(vision="maxvit", **kw) -> P.FireSensePipeline:
    return P.FireSensePipeline(ingest.MockSource("normal"), vision=vision,
                               log_csv=False, send_alerts=False, vision_every=1, **kw)


# Rule 9: anything module-level that a test changes is put back.
_CAMERA_DEFAULT = (P.camera_mod, P.HAVE_CAMERA)


def _restore_camera():
    P.camera_mod, P.HAVE_CAMERA = _CAMERA_DEFAULT


# ---------------------------------------------------------------------------
# probabilities -> score
# ---------------------------------------------------------------------------
def test_below_both_thresholds_scores_zero():
    r = M.classify_probabilities(0.30, 0.80, THR)
    assert r.available and not r.raw_detected
    assert r.score == 0.0
    assert not r.has_fire_class and not r.has_smoke_class


def test_fire_over_threshold_scores_its_probability():
    r = M.classify_probabilities(0.90, 0.10, THR)
    assert r.raw_detected and r.has_fire_class and not r.has_smoke_class
    assert abs(r.score - 90.0) < 1e-6


def test_smoke_alone_is_weighted_below_fire():
    r = M.classify_probabilities(0.10, 0.96, THR)
    assert r.raw_detected and r.has_smoke_class and not r.has_fire_class
    assert abs(r.score - 0.96 * V.CLASS_WEIGHTS["smoke"] * 100.0) < 1e-6
    assert r.score < M.classify_probabilities(0.96, 0.0, THR).score


def test_threshold_is_inclusive():
    assert M.classify_probabilities(0.36, 0.0, THR).raw_detected
    assert M.classify_probabilities(0.0, 0.88, THR).raw_detected


def test_a_classifier_result_carries_no_boxes():
    assert M.classify_probabilities(0.99, 0.99, THR).detections == []


# ---------------------------------------------------------------------------
# the channel: debounce inherited from vision_ml
# ---------------------------------------------------------------------------
def test_one_frame_is_not_enough():
    ch = stub_channel(0.95, 0.0)
    first = ch.update(FRAME)
    assert first.raw_detected and not first.fire_detected
    assert first.score == 0.0
    second = ch.update(FRAME)
    assert second.fire_detected and second.score > 0


def test_a_quiet_frame_resets_the_streak():
    ch = stub_channel(0.95, 0.0)
    ch.update(FRAME)
    ch.detector.p = (0.0, 0.0)
    ch.update(FRAME)
    ch.detector.p = (0.95, 0.0)
    assert not ch.update(FRAME).fire_detected


# ---------------------------------------------------------------------------
# refusing to run without what it needs
# ---------------------------------------------------------------------------
def test_missing_weights_report_unavailable_without_raising():
    ch = M.MaxVitChannel(HERE / "no_such_maxvit.pt")
    assert ch.load() is False
    assert "train_vision.py" in ch.error


def test_weights_without_validated_thresholds_are_refused():
    with tempfile.TemporaryDirectory() as d:
        w = Path(d) / "best.pt"
        w.write_bytes(b"not a checkpoint")
        ch = M.MaxVitChannel(w)
        assert ch.load() is False
        assert "threshold" in ch.error


def test_thresholds_are_read_from_val_results():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "val_results.json"
        f.write_text(json.dumps({"thresholds": {"fire": 0.41, "smoke": 0.77}}))
        assert M.load_thresholds(f) == {"fire": 0.41, "smoke": 0.77}
        f.write_text(json.dumps({"thresholds": {"fire": 0.41}}))
        assert M.load_thresholds(f) is None


def test_alert_boxes_without_thresholds_are_refused():
    ab = M.AlertBoxes(HERE / "no_run" / "weights" / "best.pt")
    assert ab.load() is False
    assert "threshold" in ab.error


def test_alert_boxes_keep_each_class_at_its_own_threshold():
    thr = {"fire": 0.20, "smoke": 0.16}
    dets = [V.Detection("fire", 0.25, 0, 0, 5, 5),     # kept
            V.Detection("fire", 0.18, 0, 0, 5, 5),     # under fire's 0.20
            V.Detection("smoke", 0.18, 0, 0, 5, 5),    # over smoke's 0.16
            V.Detection("flame", 0.30, 0, 0, 5, 5),    # alias of fire, kept
            V.Detection("person", 0.99, 0, 0, 5, 5)]   # not ours, dropped
    kept = M.filter_boxes(dets, thr)
    assert [(d.label, d.confidence) for d in kept] == [
        ("fire", 0.25), ("smoke", 0.18), ("flame", 0.30)]


# ---------------------------------------------------------------------------
# the escalator rule and the cap, through combine()
# ---------------------------------------------------------------------------
def test_vision_alone_never_reaches_critical():
    for engine in np.arange(0.0, C.BAND_WARNING, 2.5):
        for p in (0.36, 0.6, 0.9, 1.0):
            r = M.classify_probabilities(p, p, THR)
            score, level, _ = P.combine(float(engine), "NORMAL", r.score, True)
            assert level != "CRITICAL", (engine, p, score)
            assert score <= max(engine, P.VISION_ALONE_MAX) + 1e-9


def test_vision_never_lowers_the_engine():
    quiet = M.classify_probabilities(0.0, 0.0, THR)
    for engine, lvl in ((50.0, "WARNING"), (80.0, "CRITICAL")):
        score, level, by = P.combine(engine, lvl, quiet.score, False)
        assert (score, level, by) == (engine, lvl, "")
        # an unconfirmed (debounced) hit is no evidence either
        score, level, _ = P.combine(engine, lvl, 0.0, False)
        assert level == lvl and score >= engine


def test_vision_corroborates_an_already_concerned_engine():
    loud = M.classify_probabilities(0.99, 0.99, THR)
    score, level, by = P.combine(70.0, "WARNING", loud.score, True)
    assert score > 70.0 and by == "vision"


# ---------------------------------------------------------------------------
# the pipeline with --vision maxvit (stub camera, stub classifier)
# ---------------------------------------------------------------------------
def test_pipeline_rejects_an_unknown_vision_kind():
    try:
        quiet_pipeline(vision="hsv")
    except ValueError:
        return
    raise AssertionError("an unknown --vision value was accepted")


def test_pipeline_without_camera_runs_without_vision():
    pipe = quiet_pipeline()
    assert pipe.vision is None and pipe.alert_boxes is None
    assert "MaxViT" not in pipe.channel_summary()


def test_pipeline_says_why_maxvit_is_off():
    try:
        P.camera_mod, P.HAVE_CAMERA = StubCameraModule, True
        pipe = quiet_pipeline(camera_spec="stub",
                              weights=str(HERE / "no_such_maxvit.pt"),
                              box_weights=str(HERE / "no_run" / "weights" / "best.pt"))
        assert pipe.vision is None
        notes = " ".join(pipe.startup_notes)
        assert "vision channel off" in notes and "train_vision.py" in notes
        assert "alert boxes off" in notes
    finally:
        _restore_camera()


def test_pipeline_with_a_confident_maxvit_stays_below_critical():
    try:
        P.camera_mod, P.HAVE_CAMERA = StubCameraModule, True
        pipe = quiet_pipeline(camera_spec="stub",
                              weights=str(HERE / "no_such_maxvit.pt"))
        assert pipe.camera is not None and pipe.vision is None
        pipe.vision = stub_channel(0.99, 0.99)     # certain of fire, every frame
        assert "MaxViT vision" in pipe.channel_summary()
        seen_vision = False
        for _ in range(40):
            res = pipe.step()
            if res is None or res.engine is None:
                continue
            assert res.final_level != "CRITICAL"
            assert res.final_score >= res.engine.score
            rank = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}
            assert rank[res.final_level] >= rank[res.engine.level]
            seen_vision |= res.vision_available and res.vision_score > 0
        assert seen_vision, "the stub channel was never consulted"
    finally:
        _restore_camera()


# ---------------------------------------------------------------------------
# device fallback: CUDA, then MPS, then CPU - for MaxViT and the alert boxes
# ---------------------------------------------------------------------------
class _FakeTorch:
    """Just enough of torch for default_device(); `None` = the check raises."""

    def __init__(self, cuda, mps):
        def check(v):
            def is_available():
                if v is None:
                    raise RuntimeError("backend check failed")
                return v
            return is_available
        self.cuda = type("cuda", (), {"is_available": staticmethod(check(cuda))})
        self.backends = type("backends", (), {
            "mps": type("mps", (), {"is_available": staticmethod(check(mps))})})


def _with_torch(fake, fn):
    """Run fn with `import torch` returning `fake` (None = not installed);
    the real module, if any, is always put back."""
    had, real = "torch" in sys.modules, sys.modules.get("torch")
    sys.modules["torch"] = fake
    try:
        return fn()
    finally:
        if had:
            sys.modules["torch"] = real
        else:
            sys.modules.pop("torch", None)


def test_device_order_is_cuda_then_mps_then_cpu():
    assert M.pick_device(cuda_ok=True, mps_ok=True) == "cuda"
    assert M.pick_device(cuda_ok=True, mps_ok=False) == "cuda"
    assert M.pick_device(cuda_ok=False, mps_ok=True) == "mps"
    assert M.pick_device(cuda_ok=False, mps_ok=False) == "cpu"


def test_default_device_follows_what_the_machine_has():
    assert _with_torch(_FakeTorch(cuda=True, mps=False), M.default_device) == "cuda"
    assert _with_torch(_FakeTorch(cuda=False, mps=True), M.default_device) == "mps"
    assert _with_torch(_FakeTorch(cuda=False, mps=False), M.default_device) == "cpu"


def test_default_device_is_cpu_without_torch_or_when_checks_fail():
    assert _with_torch(None, M.default_device) == "cpu"          # not installed
    assert _with_torch(_FakeTorch(cuda=None, mps=None), M.default_device) == "cpu"
    # one broken backend check does not hide a working one
    assert _with_torch(_FakeTorch(cuda=None, mps=True), M.default_device) == "mps"


def test_an_explicit_device_is_kept():
    assert M.MaxVitChannel("unused.pt", device="cpu").detector.device == "cpu"
    assert M.AlertBoxes("unused.pt", device="cpu").detector.device == "cpu"


def test_alert_boxes_choose_a_device_instead_of_leaving_it_to_ultralytics():
    # ultralytics on its own picks CUDA or CPU, never MPS; the boxes must use
    # the same rule as the MaxViT channel.
    with tempfile.TemporaryDirectory() as d:
        run = Path(d) / "run"
        (run / "weights").mkdir(parents=True)
        (run / "val_results.json").write_text(
            json.dumps({"thresholds": {"fire": 0.2, "smoke": 0.16}}))
        for fake, want in ((_FakeTorch(cuda=False, mps=True), "mps"),
                           (_FakeTorch(cuda=False, mps=False), "cpu")):
            ab = M.AlertBoxes(run / "weights" / "best.pt")
            # Stub the model load: importing ultralytics under a fake torch
            # would leave half-imported modules behind for later tests.
            ab.detector.load = lambda: False
            ok = _with_torch(fake, ab.load)
            assert ok is False                                 # refused, not raised
            assert ab.detector.device == want
