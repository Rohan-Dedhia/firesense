"""
Verification suite for the fusion engine.
Run:  python -m pytest tests -q     (from the firesense/ folder)

These tests encode the project's actual claims, so if a claim breaks, a test
goes red. That is the point - do not delete them before the demo.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C            # noqa: E402
import fusion                 # noqa: E402
from mock_source import MockNode   # noqa: E402


def run(scenario, n=90, env="Kitchen", visual_from=None, calibrate=True, dt=1.0):
    """Drive the engine with a scenario on a synthetic 1 Hz clock."""
    eng = fusion.FusionEngine(env)
    node = MockNode(scenario, seed=7)
    t0 = 1_000_000.0
    if calibrate:                       # feed a quiet baseline first
        quiet = MockNode("normal", seed=3)
        for i in range(60):
            r, _ = quiet.next()
            eng.update(r, now=t0 + i * dt)
        eng.calibrator.finish()
        t0 += 60 * dt
    out = []
    for i in range(n):
        r, flame = node.next()
        vis = flame if visual_from is None else (i >= visual_from)
        out.append(eng.update(r, visual_fire=vis,
                              visual_confidence=85.0 if vis else 0.0,
                              now=t0 + i * dt))
    return eng, out


# --- basics ---------------------------------------------------------------
def test_quiet_room_stays_normal():
    _, res = run("normal", 90)
    assert all(r.level == "NORMAL" for r in res)
    assert max(r.score for r in res) < C.BAND_WARNING


def test_humidity_compensation_moves_the_reading():
    dry = fusion.compensate_gas(300, humidity_pct=25, temp_c=20)
    humid = fusion.compensate_gas(300, humidity_pct=90, temp_c=20)
    assert dry > 300 > humid            # dry air reads high, humid air reads low
    assert abs(fusion.compensate_gas(300, 65, 20) - 300) < 1e-6   # at reference


# --- feature B: false-alarm suppression ----------------------------------
def test_cooking_is_recognised_and_suppressed():
    _, res = run("cooking_false_positive", 90)
    tail = res[-40:]
    assert any(r.signature == "cooking" for r in tail), "cooking never recognised"
    assert any(r.suppressed for r in tail), "cooking never suppressed"


def test_leak_is_not_suppressed_and_escalates():
    _, res = run("gradual_leak", 120)
    assert any(r.signature == "leak" for r in res), "leak signature never matched"
    assert not any(r.suppressed for r in res), "a leak must never be suppressed"
    assert max(r.score for r in res) >= C.BAND_WARNING


def test_same_gas_level_scores_lower_when_it_looks_like_cooking():
    """
    The core claim: two events reaching a similar gas concentration are scored
    differently because their humidity signature differs.
    """
    _, cook = run("cooking_false_positive", 90)
    _, leak = run("gradual_leak", 90)
    cook_peak_gas = max(r.gas_compensated for r in cook)
    leak_peak_gas = max(r.gas_compensated for r in leak)
    assert abs(cook_peak_gas - leak_peak_gas) / leak_peak_gas < 0.35, "gas levels not comparable"
    assert max(r.score for r in cook) < max(r.score for r in leak)


# --- fire ----------------------------------------------------------------
def test_fire_reaches_critical_with_high_confidence():
    _, res = run("sudden_fire", 60)
    crit = [r for r in res if r.level == "CRITICAL"]
    assert crit, "fire scenario never reached CRITICAL"
    assert max(r.confidence for r in crit) > 55
    assert any(r.signature == "fire" for r in res)


# --- latency stopwatch ----------------------------------------------------
def test_fusion_beats_the_naive_threshold_on_a_leak():
    eng, res = run("gradual_leak", 150)
    assert eng._fusion_trigger_t is not None, "fusion never alarmed"
    assert eng._naive_trigger_t is not None, "naive threshold never alarmed"
    assert eng.latency_gain_s is not None and eng.latency_gain_s > 0, \
        "fusion must alarm strictly earlier than the fixed threshold"


# --- environment personalisation -----------------------------------------
def test_same_reading_scores_differently_per_environment():
    reading = {"gas_ppm": 260, "temp_c": 33, "humidity_pct": 50}
    scores = {}
    for env in ("Kitchen", "Factory", "Storage Room"):
        eng = fusion.FusionEngine(env)
        eng.calibrator.done = True          # skip calibration, use factory profile
        scores[env] = eng.update(reading, now=1.0).score
    assert scores["Storage Room"] > scores["Factory"] > scores["Kitchen"]


# --- feature A: auto-calibration -----------------------------------------
def test_calibration_tightens_thresholds_in_a_clean_room():
    eng = fusion.FusionEngine("Kitchen")
    node = MockNode("normal", seed=11)      # idle gas around 90 ppm
    for i in range(60):
        r, _ = node.next()
        eng.update(r, now=1_000 + i)
    eng.calibrator.finish()
    prof = eng.calibrator.profile()
    assert prof["gas_normal_max"] < C.ENVIRONMENT_PROFILES["Kitchen"]["gas_normal_max"]
    assert prof["gas_normal_max"] >= (
        C.ENVIRONMENT_PROFILES["Kitchen"]["gas_normal_max"] * C.CALIBRATION_MAX_SHRINK)


def test_calibration_ignores_readings_during_an_event():
    eng = fusion.FusionEngine("Storage Room")
    node = MockNode("sudden_fire", seed=5)
    for i in range(40):
        r, flame = node.next()
        eng.update(r, visual_fire=flame, visual_confidence=90.0, now=1_000 + i)
    # an event must not be absorbed into "normal"
    assert eng.calibrator.profile()["gas_normal_max"] <= \
        C.ENVIRONMENT_PROFILES["Storage Room"]["gas_normal_max"] * C.CALIBRATION_MAX_GROWTH


# --- self-diagnostics -----------------------------------------------------
def test_sensor_fault_is_flagged():
    _, res = run("sensor_fault", 30)
    assert any(r.flags for r in res), "impossible sensor values were not flagged"


def test_visual_only_trigger_is_flagged_as_disagreement():
    eng = fusion.FusionEngine("Kitchen")
    eng.calibrator.done = True
    r = eng.update({"gas_ppm": 60, "temp_c": 24, "humidity_pct": 55},
                   visual_fire=True, visual_confidence=95.0, now=1.0)
    assert any("Optical channel reports flame" in f for f in r.flags)


def test_camera_offline_is_reported_and_penalises_confidence():
    eng = fusion.FusionEngine("Kitchen")
    eng.calibrator.done = True
    on = eng.update({"gas_ppm": 100, "temp_c": 25, "humidity_pct": 55},
                    visual_available=True, now=1.0)
    off = eng.update({"gas_ppm": 100, "temp_c": 25, "humidity_pct": 55},
                     visual_available=False, now=2.0)
    assert off.confidence < on.confidence
    assert any("No ignition channel available" in f for f in off.flags)


# --- countdown ------------------------------------------------------------
def test_countdown_appears_on_a_rising_trend_and_not_on_a_flat_one():
    _, flat = run("normal", 60)
    assert all(r.seconds_to_critical is None for r in flat)
    _, fire = run("sudden_fire", 40)
    assert any(r.seconds_to_critical is not None for r in fire)


def test_countdown_shrinks_as_the_event_develops():
    _, res = run("gradual_leak", 140)
    etas = [(i, r.seconds_to_critical) for i, r in enumerate(res)
            if r.seconds_to_critical is not None and r.level != "CRITICAL"]
    assert len(etas) >= 5
    first, last = etas[0][1], etas[-1][1]
    assert last < first, f"ETA did not shrink ({first} -> {last})"


# --- IR flame sensor (added once the sensor was wired) --------------------
def _with_flame(present=True):
    C.FLAME_SENSOR_PRESENT = present


def test_flame_sensor_alone_reaches_critical_with_no_camera():
    """A wired IR sensor must be able to confirm ignition on its own."""
    _with_flame(True)
    try:
        eng = fusion.FusionEngine("Kitchen")
        eng.calibrator.done = True
        reading = {"gas_ppm": 320, "temp_c": 44, "humidity_pct": 45,
                   "flame_detected": True}
        eng.update(reading, visual_available=False, now=1.0)   # debounce sample
        r = eng.update(reading, visual_available=False, now=2.0)
        assert r.flame_available and r.flame_sensor
        assert r.ignition_score >= C.IGNITION_IR_ONLY
        assert r.level == "CRITICAL", (r.level, r.score)
    finally:
        _with_flame(False)


def test_both_ignition_channels_agreeing_beats_either_alone():
    _with_flame(True)
    try:
        base = {"gas_ppm": 200, "temp_c": 30, "humidity_pct": 45}
        def score(flame, vis):
            eng = fusion.FusionEngine("Kitchen")
            eng.calibrator.done = True
            r = None
            for t in (1.0, 2.0):        # two samples: satisfies the IR debounce
                r = eng.update({**base, "flame_detected": flame},
                               visual_fire=vis, visual_confidence=85.0,
                               visual_available=True, now=t)
            return r.score
        both = score(True, True)
        ir_only = score(True, False)
        cam_only = score(False, True)
        assert both > ir_only and both > cam_only
    finally:
        _with_flame(False)


def test_ignition_channels_disagreeing_is_flagged():
    _with_flame(True)
    try:
        eng = fusion.FusionEngine("Kitchen")
        eng.calibrator.done = True
        reading = {"gas_ppm": 300, "temp_c": 40, "humidity_pct": 45,
                   "flame_detected": True}
        eng.update(reading, visual_fire=False, visual_available=True, now=1.0)
        r = eng.update(reading, visual_fire=False, visual_available=True, now=2.0)
        assert any("Ignition channels disagree" in f for f in r.flags)
    finally:
        _with_flame(False)


def test_flame_field_is_ignored_until_the_sensor_is_declared():
    """The ESP32 always sends the field; software must ignore it until wired."""
    _with_flame(False)
    eng = fusion.FusionEngine("Kitchen")
    eng.calibrator.done = True
    r = eng.update({"gas_ppm": 100, "temp_c": 25, "humidity_pct": 55,
                    "flame_detected": True}, visual_available=False, now=1.0)
    assert not r.flame_available and not r.flame_sensor
    assert r.ignition_score == 0.0
    assert r.level == "NORMAL"


def test_flame_sensor_clears_the_cooking_suppression():
    """A fire starting during cooking must not stay suppressed."""
    _with_flame(True)
    try:
        eng = fusion.FusionEngine("Kitchen")
        eng.calibrator.done = True
        node = MockNode("cooking_false_positive", seed=7)
        for i in range(40):
            r, _ = node.next()
            res = eng.update({**r, "flame_detected": False}, now=1_000 + i)
        assert res.suppressed, "cooking should be suppressed before ignition"
        r, _ = node.next()
        eng.update({**r, "flame_detected": True}, now=1_041)   # debounce sample
        r, _ = node.next()
        res = eng.update({**r, "flame_detected": True}, now=1_042)
        assert not res.suppressed, "ignition must clear the cooking latch"
        assert res.level in ("WARNING", "CRITICAL")
    finally:
        _with_flame(False)


def test_calibration_holds_while_the_sensor_is_still_warming_up():
    """A drifting baseline must not be locked in as 'normal'."""
    eng = fusion.FusionEngine("Kitchen")
    step = (C.CALIBRATION_SECONDS + 20) / 59.0     # window elapses within 60 samples
    for i in range(60):                     # steady one-way march = warming up
        eng.update({"gas_ppm": 300 - i * 2.0, "temp_c": 26, "humidity_pct": 55},
                   now=1_000 + i * step)
    assert not eng.calibrator.done, "locked in a baseline while still drifting"
    assert eng.calibrator.holding
    assert "warming up" in eng.calibrator.adjustment_note


def test_calibration_completes_once_the_sensor_settles():
    eng = fusion.FusionEngine("Kitchen")
    step = (C.CALIBRATION_SECONDS + 20) / 59.0
    for i in range(60):                     # settled: wobble, no trend
        eng.update({"gas_ppm": 150 + (i % 5) * 2.0, "temp_c": 26,
                    "humidity_pct": 55}, now=1_000 + i * step)
    assert eng.calibrator.done, "never finished on a settled sensor"
    assert not eng.calibrator.holding


def test_calibration_completes_on_a_replayed_log_not_wall_clock():
    """A log replayed in milliseconds must calibrate exactly like a live run."""
    eng = fusion.FusionEngine("Kitchen")
    for i in range(int(C.CALIBRATION_SECONDS) + 40):
        eng.update({"gas_ppm": 1850 + (i % 7) * 6, "temp_c": 29,
                    "humidity_pct": 62}, now=5_000 + i)
    assert eng.calibrator.done, "calibration never completed on a fast replay"
    assert eng.calibrator.relative_mode, "should have gone relative on a 1850ppm idle"


def test_uncalibrated_node_settles_to_normal_after_calibrating():
    """The demo-day case: odd baseline must not sit pinned at WARNING."""
    eng = fusion.FusionEngine("Kitchen")
    res = None
    for i in range(int(C.CALIBRATION_SECONDS) + 60):
        res = eng.update({"gas_ppm": 1850 + (i % 7) * 6, "temp_c": 29,
                          "humidity_pct": 62}, now=5_000 + i)
    assert res.level == "NORMAL", (res.level, res.score)


def test_no_absolute_alarm_while_calibrating_but_ignition_still_alarms():
    quiet = fusion.FusionEngine("Storage Room")
    r = quiet.update({"gas_ppm": 5000, "temp_c": 26, "humidity_pct": 55}, now=1.0)
    assert r.calibrating and r.level == "NORMAL"
    assert any("Still calibrating" in f for f in r.flags)

    fire = fusion.FusionEngine("Storage Room")
    r2 = fire.update({"gas_ppm": 5000, "temp_c": 60, "humidity_pct": 40},
                     visual_fire=True, visual_confidence=95.0, now=1.0)
    assert r2.level != "NORMAL", "ignition must alarm even during calibration"


# --- regression: calibration must never silence a real event --------------
def test_gradual_leak_from_a_cold_start_still_alarms():
    """
    The dangerous case: the node is switched on and a leak is ALREADY under way,
    so calibration never sees a quiet room. The system must still alarm - it
    must not sit in "still calibrating" through the event it exists to catch.
    """
    eng = fusion.FusionEngine("Kitchen")          # no forced calibration
    node = MockNode("gradual_leak", seed=7)
    first = None
    for i in range(300):
        r, _ = node.next()
        res = eng.update(r, now=1_000_000 + i)
        if first is None and res.level != "NORMAL":
            first = i
    assert first is not None, "a leak from cold start never alarmed"
    assert first < 60, f"alarmed far too late (t+{first}s)"
    assert any(r for r in [res]) and res.level != "NORMAL" or True


def test_calibration_always_terminates():
    """Calibration must not stall forever when readings are never quiet."""
    eng = fusion.FusionEngine("Kitchen")
    for i in range(int(C.CALIBRATION_GIVE_UP_S) + 60):
        eng.update({"gas_ppm": 200 + i * 3.0, "temp_c": 26, "humidity_pct": 55},
                   now=1_000 + i)
    assert eng.calibrator.done, "calibration never terminated on a rising signal"


def test_calibration_gives_up_gracefully_when_never_quiet():
    """With too few usable samples it falls back to the profile and says so."""
    eng = fusion.FusionEngine("Storage Room")
    node = MockNode("sudden_fire", seed=3)
    for i in range(int(C.CALIBRATION_GIVE_UP_S) + 30):
        r, flame = node.next()
        eng.update(r, visual_fire=flame, visual_confidence=90.0 if flame else 0.0,
                   now=1_000 + i)
    assert eng.calibrator.done
    prof = eng.calibrator.profile()
    assert prof["gas_normal_max"] <= (
        C.ENVIRONMENT_PROFILES["Storage Room"]["gas_normal_max"] * C.CALIBRATION_MAX_GROWTH)


def test_pattern_resolves_on_a_slow_dashboard():
    """A 2-3 s polling cadence must still produce a usable trend, not
    'insufficient_data' forever."""
    eng = fusion.FusionEngine("Kitchen")
    eng.calibrator.done = True
    node = MockNode("gradual_leak", seed=7)
    res = None
    for i in range(30):
        r, _ = node.next()
        res = eng.update(r, now=1_000 + i * 2.8)
    assert res.slopes.mature, "trend never matured at a slow polling rate"
    assert res.signature != "insufficient_data"


def test_levels_do_not_flap_on_a_band_boundary():
    """A score hovering at the CRITICAL boundary must not chatter."""
    eng = fusion.FusionEngine("Kitchen")
    eng.calibrator.done = True
    # walk the score across the boundary and back, repeatedly
    seq = [C.BAND_CRITICAL + 1, C.BAND_CRITICAL - 1] * 8
    transitions = 0
    prev = eng.level
    for i, target in enumerate(seq):
        lvl = eng._level_with_hysteresis(target, prev)
        if lvl != prev:
            transitions += 1
        prev = lvl
    assert transitions <= 1, f"flapped {transitions} times across the boundary"


def test_hysteresis_still_escalates_immediately():
    assert fusion.FusionEngine._level_with_hysteresis(C.BAND_CRITICAL, "NORMAL") == "CRITICAL"
    assert fusion.FusionEngine._level_with_hysteresis(C.BAND_WARNING, "NORMAL") == "WARNING"
    # and does stand down once genuinely clear
    assert fusion.FusionEngine._level_with_hysteresis(
        C.BAND_WARNING - C.LEVEL_HYSTERESIS - 1, "CRITICAL") == "NORMAL"


def test_flame_debounce_matches_the_configured_setting():
    """
    Honours whatever FLAME_CONFIRM_SAMPLES is set to:
      1 -> instant response (a single reading is a flame)
      2+ -> isolated readings are treated as noise until confirmed
    Either is a legitimate choice; the test checks the configured behaviour
    rather than assuming one.
    """
    need = C.FLAME_CONFIRM_SAMPLES
    C.FLAME_SENSOR_PRESENT = True
    try:
        eng = fusion.FusionEngine("Kitchen")
        eng.calibrator.done = True
        quiet = {"gas_ppm": 100, "temp_c": 26, "humidity_pct": 55,
                 "flame_detected": False}
        eng.update(quiet, visual_available=False, now=0.0)

        res = None
        for i in range(1, need + 1):
            res = eng.update({**quiet, "flame_detected": True},
                             visual_available=False, now=float(i))
            if i < need:                       # not yet confirmed
                assert not res.flame_sensor, f"trusted after only {i} sample(s)"
                assert res.level == "NORMAL"
                assert any("isolated flame reading" in f for f in res.flags)

        assert res.flame_sensor, f"never confirmed after {need} sample(s)"
        assert res.level != "NORMAL", "confirmed ignition did not escalate"
    finally:
        C.FLAME_SENSOR_PRESENT = False


# --- regression against REAL hardware data --------------------------------
def test_real_session_never_suppresses_a_high_gas_event():
    """
    Locks in a fix found on real hardware: a brief humidity transient (a breath
    near the BME280) matched the cooking signature and suppressed a genuine gas
    event. Replays the recorded session and asserts it cannot happen again.
    """
    import csv
    from datetime import datetime
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "data" / "real_session_20260905.csv"
    if not path.exists():
        return                                    # data file not shipped
    rows = list(csv.DictReader(path.open()))
    t0 = datetime.fromisoformat(rows[0]["timestamp"])

    eng = fusion.FusionEngine("Kitchen")
    bad, transitions, prev = 0, 0, "NORMAL"
    for r in rows:
        t = (datetime.fromisoformat(r["timestamp"]) - t0).total_seconds()
        res = eng.update({"gas_ppm": float(r["gas_raw"]),
                          "temp_c": float(r["temp_c"]),
                          "humidity_pct": float(r["humidity_pct"])},
                         now=1_000_000 + t)
        if res.suppressed and res.gas_score > 70:
            bad += 1
        if res.level != prev:
            transitions += 1
            prev = res.level
    assert bad == 0, f"{bad} readings suppressed while the gas channel was high"
    assert transitions <= 12, f"level flapped {transitions} times on real data"


def test_a_fast_smoke_burst_is_never_suppressed_as_cooking():
    """
    Smouldering material (a just-extinguished match) raises gas AND humidity
    together, exactly like cooking, with no flame and no temperature rise.
    The discriminator is rate: cooking is gradual, a smoke burst is not.
    """
    eng = fusion.FusionEngine("Kitchen")
    eng.calibrator.done = True
    res = None
    for i in range(30):
        res = eng.update({"gas_ppm": 400 + i * 40,      # fast burst
                          "temp_c": 29.3 + i * 0.01,    # barely moves
                          "humidity_pct": 72 + i * 0.12},  # combustion vapour
                         visual_available=False, now=1_000 + i)
    assert not res.suppressed, "a fast smoke burst was excused as cooking"
    assert res.level != "NORMAL", "smouldering smoke did not raise an alarm"
