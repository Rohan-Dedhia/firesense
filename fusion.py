"""
FireSense - fusion engine.

Pure functions + one stateful FusionEngine class. No I/O, no hardware, no
Streamlit imports: this module is fully unit-testable and is the piece the
whole project is actually graded on.

Pipeline per reading:
    raw reading  ->  sanity check
                 ->  humidity/temperature compensation of the MQ2 value
                 ->  auto-calibrated environment profile          (feature A)
                 ->  per-channel normalisation into 0-100 sub-scores
                 ->  weighted fusion  ->  raw risk score
                 ->  signature analysis on slopes                 (feature B)
                     -> cooking suppression / leak confirmation
                 ->  confidence (cross-modality agreement)
                 ->  disagreement + fault flags
                 ->  time-to-critical countdown
                 ->  latency stopwatch vs naive single-sensor threshold
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional

import config as C


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _linfit(xs, ys):
    """Least-squares slope of ys over xs. Returns 0.0 when undetermined."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom < 1e-9:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def _percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


# ---------------------------------------------------------------------------
# 1. compensation
# ---------------------------------------------------------------------------
def compensate_gas(raw_ppm: float, humidity_pct: float, temp_c: float) -> float:
    """
    MQ2 sensing-resistance drifts with ambient humidity and temperature, so the
    same true gas concentration reads differently on a humid evening than on a
    dry morning. We correct the reading toward the datasheet reference point
    (20 C / 65 %RH) BEFORE it reaches the risk score.

    This is what makes this a fusion system rather than two numbers displayed
    side by side: BME280 is actively correcting MQ2, not just sitting next to it.
    """
    rh_term = 1.0 + C.RH_COEFF * (C.RH_REFERENCE - humidity_pct)
    t_term = 1.0 + C.TEMP_COEFF * (C.TEMP_REFERENCE - temp_c)
    return max(0.0, raw_ppm * rh_term * t_term)


# ---------------------------------------------------------------------------
# 2. normalisation
# ---------------------------------------------------------------------------
def normalize(value: float, normal_max: float, critical: float) -> float:
    """
    Map a physical reading onto 0-100.
      0   .. normal_max   ->  0 .. 40   (inside the profile's 'normal' band)
      normal_max .. crit  -> 40 .. 100  (escalating)
    """
    if normal_max <= 0:
        return 0.0
    if value <= normal_max:
        return _clamp(value / normal_max * 40.0)
    span = max(critical - normal_max, 1e-6)
    excess = min(value - normal_max, span)
    return _clamp(40.0 + excess / span * 60.0)


# ---------------------------------------------------------------------------
# 3. auto-calibration  (feature A)
# ---------------------------------------------------------------------------
class BaselineCalibrator:
    """
    Learns THIS room's idle baseline instead of trusting hard-coded numbers.

    On start-up (or whenever the environment profile is switched) the node spends
    CALIBRATION_SECONDS observing. The p90 of the idle gas readings x a headroom
    factor becomes the effective 'gas_normal_max' for that room, clamped so a
    calibration performed during an actual leak can never blow the threshold out
    to something useless.
    """

    def __init__(self, environment: str):
        self.reset(environment)

    def reset(self, environment: str) -> None:
        self.environment = environment
        self.base = dict(C.ENVIRONMENT_PROFILES[environment])
        self.samples: list[tuple[float, float, float]] = []   # (gas, temp, hum)
        # Data clock, not wall clock: set from the first reading's timestamp so
        # a replayed or accelerated log calibrates exactly like a live one.
        self.started: float | None = None
        self._last_now: float = 0.0
        self.done = False
        self.holding = False
        self.effective = dict(self.base)
        self.relative_mode = False
        self.adjustment_note = "using factory profile (not yet calibrated)"

    # -- lifecycle ---------------------------------------------------------
    @property
    def progress(self) -> float:
        if self.done:
            return 1.0
        if self.started is None:
            return 0.0
        return min(1.0, (self._last_now - self.started) / C.CALIBRATION_SECONDS)

    def should_observe(self, gas_c: float) -> bool:
        """
        Decide whether a sample is 'quiet enough' to define normal. Judged
        against what this node has ALREADY seen, not against the factory
        profile - because on an uncalibrated MQ2 the profile's absolute ppm
        numbers may be meaningless. Early samples are all accepted so the
        calibrator can find the room's level at all; after that, anything well
        above the running median is treated as an event and excluded.
        """
        if self.done:
            return False
        if len(self.samples) < 10:
            return True
        seen = sorted(x[0] for x in self.samples)
        median = seen[len(seen) // 2]
        return gas_c <= max(median * 1.6, median + 40.0)

    def is_drifting(self) -> bool:
        """
        Compare the first third of the collected samples with the last third.
        A still-warming MQ2 shows a steady one-way march; a settled one wobbles
        around a level. This is how we tell "warming up" from "ready".
        """
        n = len(self.samples)
        if n < max(C.CALIBRATION_MIN_SAMPLES, 12):
            return False
        k = n // 3
        first = sum(x[0] for x in self.samples[:k]) / k
        last = sum(x[0] for x in self.samples[-k:]) / k
        if first <= 0:
            return False
        return abs(last - first) / first > C.CALIBRATION_DRIFT_TOLERANCE

    def tick(self, now: float | None = None) -> None:
        """
        Advance the calibration clock. Called for EVERY reading, including ones
        too noisy to learn from - otherwise a rising leak (whose samples are all
        rejected) would freeze the clock and calibration would never end.
        """
        if self.done:
            return
        now = time.time() if now is None else now
        if self.started is None:
            self.started = now
        self._last_now = now
        elapsed = now - self.started
        if elapsed < C.CALIBRATION_SECONDS:
            return

        if len(self.samples) >= C.CALIBRATION_MIN_SAMPLES:
            if self.is_drifting() and elapsed < C.CALIBRATION_GIVE_UP_S:
                self.holding = True                     # sensor still warming up
                self.samples = self.samples[-600:]
                direction = ("up" if self.samples[-1][0] > self.samples[0][0]
                             else "down")
                self.adjustment_note = (
                    f"sensor still warming up - baseline has moved {direction} "
                    f"over the last {elapsed/60:.0f} min, so calibration is being "
                    f"held rather than locking in a wrong baseline")
                return
            self.holding = False
            self.finish()
        elif elapsed >= C.CALIBRATION_GIVE_UP_S:
            self.give_up()

    def give_up(self) -> None:
        """Room never went quiet - fall back to the profile and say so."""
        self.holding = False
        self.done = True
        self.relative_mode = False
        self.effective = dict(self.base)
        self.adjustment_note = (
            f"could not learn a baseline - the air was never quiet enough during "
            f"start-up (only {len(self.samples)} usable samples). Using the "
            f"factory {self.environment} profile instead. If nothing is actually "
            f"wrong, press Recalibrate once the room settles.")

    def observe(self, gas_c: float, temp_c: float, hum: float,
                now: float | None = None) -> None:
        if self.done:
            return
        self.samples.append((gas_c, temp_c, hum))

    def finish(self) -> None:
        """Compute the room-specific thresholds and lock them in."""
        if not self.samples:
            self.done = True
            return
        gas = [s[0] for s in self.samples]
        temp = [s[1] for s in self.samples]

        idle_gas = _percentile(gas, C.CALIBRATION_PERCENTILE)
        learned_gas = idle_gas * C.CALIBRATION_HEADROOM
        self.relative_mode = (
            learned_gas > self.base["gas_normal_max"] * C.RELATIVE_MODE_TRIGGER)

        if self.relative_mode:
            # The absolute ppm scale is not trustworthy (uncalibrated R0, or a
            # genuinely dirty environment). Work in multiples of this room's own
            # idle level instead of absolute concentration.
            gas_norm = idle_gas * C.RELATIVE_NORMAL_MULT
        else:
            lo = self.base["gas_normal_max"] * C.CALIBRATION_MAX_SHRINK
            hi = self.base["gas_normal_max"] * C.CALIBRATION_MAX_GROWTH
            gas_norm = _clamp(learned_gas, lo, hi)

        learned_temp = _percentile(temp, C.CALIBRATION_PERCENTILE) + 8.0
        temp_norm = _clamp(
            learned_temp,
            self.base["temp_normal_max"] * C.CALIBRATION_MAX_SHRINK,
            self.base["temp_normal_max"] * 1.4,
        )

        self.effective = dict(self.base)
        self.effective["gas_normal_max"] = round(gas_norm, 1)
        self.effective["temp_normal_max"] = round(temp_norm, 1)
        self.effective["temp_critical"] = max(self.base["temp_critical"], temp_norm + 12)

        if self.relative_mode:
            self.effective["gas_critical"] = round(idle_gas * C.RELATIVE_CRITICAL_MULT, 1)
            self.adjustment_note = (
                f"RELATIVE mode ({len(self.samples)} idle samples): this room's "
                f"idle level is {idle_gas:.0f} ppm-equivalent, far above the "
                f"{self.environment} profile - the absolute ppm scale is not "
                f"trustworthy (MQ2 R0 uncalibrated), so thresholds are set as "
                f"multiples of this node's own baseline: normal < "
                f"{gas_norm:.0f}, critical at {idle_gas * C.RELATIVE_CRITICAL_MULT:.0f}"
            )
        else:
            # critical stays anchored to the profile, but never below normal_max
            self.effective["gas_critical"] = max(self.base["gas_critical"],
                                                 gas_norm * 1.6)
            direction = ("loosened" if gas_norm > self.base["gas_normal_max"]
                         else "tightened")
            self.adjustment_note = (
                f"calibrated on {len(self.samples)} idle samples: gas_normal_max "
                f"{self.base['gas_normal_max']:.0f} -> {gas_norm:.0f} ppm ({direction})"
            )
        self.done = True

    def profile(self) -> dict:
        return self.effective


# ---------------------------------------------------------------------------
# 4. signature analysis  (feature B)
# ---------------------------------------------------------------------------
@dataclass
class Slopes:
    gas: float = 0.0        # ppm / s
    temp: float = 0.0       # C / s
    humidity: float = 0.0   # %RH / s
    window_s: float = 0.0
    samples: int = 0
    mature: bool = False    # enough samples over enough time to trust the slope


def classify_signature(sl: Slopes, ignition: bool) -> tuple[str, str]:
    """
    Physical reasoning, not a trained model:

      cooking / steam  -> gas UP and humidity UP together (combustion + water
                          vapour from the food), temperature drifting up slowly
      gas leak         -> gas UP while humidity is flat or FALLING (escaping LPG
                          displaces room air; no water vapour is produced)
      fire             -> gas UP, temperature UP fast, humidity DOWN, and the
                          IR flame sensor confirms ignition

    Returns (label, human-readable reason).
    """
    if ignition and sl.temp > 0.05:
        return "fire", "flame confirmed by an ignition channel while temperature is rising"
    if not sl.mature:
        return "insufficient_data", "not enough history yet to judge the pattern"
    if sl.gas < C.COOKING_GAS_SLOPE_MIN * 0.4:
        return "steady", "no meaningful gas trend"
    if sl.gas >= C.COOKING_GAS_SLOPE_MIN and sl.humidity >= C.COOKING_HUM_SLOPE_MIN:
        return "cooking", (
            f"gas and humidity are rising together "
            f"(+{sl.gas:.1f} ppm/s with +{sl.humidity:.2f} %RH/s) - "
            f"water-vapour signature of cooking, not a leak"
        )
    if sl.gas >= C.COOKING_GAS_SLOPE_MIN and sl.humidity <= C.LEAK_HUM_SLOPE_MAX:
        return "leak", (
            f"gas rising (+{sl.gas:.1f} ppm/s) with humidity flat or falling "
            f"({sl.humidity:+.2f} %RH/s) - no cooking signature"
        )
    return "rising", "gas trending up, pattern not yet distinctive"


# ---------------------------------------------------------------------------
# 5. result container
# ---------------------------------------------------------------------------
@dataclass
class RiskResult:
    ts: float
    environment: str
    # inputs
    gas_raw: float = 0.0
    gas_compensated: float = 0.0
    temp_c: float = 0.0
    humidity_pct: float = 0.0
    flame_sensor: bool = False
    flame_available: bool = False
    # sub-scores
    gas_score: float = 0.0
    temp_score: float = 0.0
    ignition_score: float = 0.0
    # outputs
    score_raw: float = 0.0
    score: float = 0.0
    level: str = "NORMAL"
    confidence: float = 0.0
    signature: str = "steady"
    signature_reason: str = ""
    suppressed: bool = False
    rapid_rise: bool = False
    explanation: str = ""
    flags: list = field(default_factory=list)
    slopes: Slopes = field(default_factory=Slopes)
    seconds_to_critical: Optional[float] = None
    calibrating: bool = False
    calibration_progress: float = 0.0
    calibration_note: str = ""
    naive_alarm: bool = False
    latency_gain_s: Optional[float] = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["slopes"] = asdict(self.slopes)
        return d


# ---------------------------------------------------------------------------
# 6. the engine
# ---------------------------------------------------------------------------
class FusionEngine:
    def __init__(self, environment: str = C.DEFAULT_ENVIRONMENT):
        self.environment = environment
        self.calibrator = BaselineCalibrator(environment)
        self.history: deque = deque(maxlen=600)      # (t, gas_c, temp, hum)
        self.score_history: deque = deque(maxlen=600)  # (t, score)
        self.events: list[dict] = []
        self.level = "NORMAL"
        # latency stopwatch
        self._naive_trigger_t: Optional[float] = None
        self._fusion_trigger_t: Optional[float] = None
        self.latency_gain_s: Optional[float] = None
        # cooking-context latch (see update())
        self._cooking_until: float = 0.0
        self._cooking_streak: int = 0
        self._cooking_long_until: float = 0.0
        self._flame_streak: int = 0        # IR debounce
        self._gas_window: deque = deque(maxlen=max(C.GAS_MEDIAN_WINDOW, 1))
        self._level_since: float = 0.0

    # -- environment -------------------------------------------------------
    def set_environment(self, environment: str, recalibrate: bool = True) -> None:
        if environment == self.environment:
            return
        self.environment = environment
        if recalibrate:
            self.calibrator.reset(environment)
        else:
            self.calibrator.environment = environment
            self.calibrator.base = dict(C.ENVIRONMENT_PROFILES[environment])
            self.calibrator.effective = dict(C.ENVIRONMENT_PROFILES[environment])

    def reset_latency_stopwatch(self) -> None:
        self._naive_trigger_t = None
        self._fusion_trigger_t = None
        self.latency_gain_s = None

    # -- sanity ------------------------------------------------------------
    @staticmethod
    def check_sanity(reading: dict) -> list[str]:
        flags = []
        for field_name, (lo, hi) in C.SANE_RANGES.items():
            v = reading.get(field_name)
            if v is None:
                flags.append(f"{field_name}: missing from payload")
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                flags.append(f"{field_name}: non-numeric value")
                continue
            if not (lo <= v <= hi):
                flags.append(f"{field_name}: {v} outside sane range ({lo}-{hi}) - sensor fault?")
            if isinstance(v, float) and math.isnan(v):
                flags.append(f"{field_name}: NaN - sensor disconnected?")
        return flags

    # -- slopes ------------------------------------------------------------
    def _slopes(self, now: float) -> Slopes:
        window = [h for h in self.history if now - h[0] <= C.SLOPE_WINDOW_S]
        if len(window) < 3:
            return Slopes(samples=len(window))
        ts = [h[0] for h in window]
        span = ts[-1] - ts[0]
        return Slopes(
            gas=_linfit(ts, [h[1] for h in window]),
            temp=_linfit(ts, [h[2] for h in window]),
            humidity=_linfit(ts, [h[3] for h in window]),
            window_s=span,
            samples=len(window),
            mature=(len(window) >= C.RATE_MIN_SAMPLES
                    and span >= C.SLOPE_WINDOW_S * C.RATE_MIN_WINDOW_FRACTION),
        )

    # -- countdown ---------------------------------------------------------
    def _time_to_critical(self, now: float, current_score: float) -> Optional[float]:
        """
        Short-horizon linear extrapolation of the CURRENT session's risk trend.
        Deliberately NOT a trained forecasting model - see README for the exact
        wording to use in the report.
        """
        if current_score >= C.COUNTDOWN_TARGET:
            return 0.0
        # Gate 1: only extrapolate once the situation is actually developing,
        # so ordinary idle wobble never produces a scary countdown.
        if current_score < C.COUNTDOWN_MIN_SCORE:
            return None
        window = [s for s in self.score_history if now - s[0] <= C.SLOPE_WINDOW_S]
        if len(window) < 5:
            return None
        slope = _linfit([s[0] for s in window], [s[1] for s in window])
        # Gate 2: the trend must be meaningfully upward.
        if slope < C.COUNTDOWN_MIN_SLOPE:
            return None
        eta = (C.COUNTDOWN_TARGET - current_score) / slope
        if eta <= 0 or eta > C.COUNTDOWN_MAX_HORIZON_S:
            return None            # beyond the horizon the extrapolation is noise
        return round(eta, 1)

    @staticmethod
    def _level_with_hysteresis(score: float, current: str) -> str:
        """
        Escalate immediately, de-escalate reluctantly. A score hovering on a
        band boundary must not flap between levels every reading.
        """
        h = C.LEVEL_HYSTERESIS
        if score >= C.BAND_CRITICAL:
            return "CRITICAL"
        if current == "CRITICAL" and score >= C.BAND_CRITICAL - h:
            return "CRITICAL"
        if score >= C.BAND_WARNING:
            return "WARNING"
        if current in ("WARNING", "CRITICAL") and score >= C.BAND_WARNING - h:
            return "WARNING"
        return "NORMAL"

    # -- confidence --------------------------------------------------------
    @staticmethod
    def _confidence(sub_scores: list[float], ignition_available: bool) -> float:
        """
        Cross-modality agreement. Independent physics (chemical, thermal,
        infrared) pointing the same way = high confidence. One channel shouting
        while the others are calm = low confidence, flag for human verification.
        """
        n = len(sub_scores)
        if n < 2:
            return 40.0
        mean = sum(sub_scores) / n
        var = sum((s - mean) ** 2 for s in sub_scores) / n
        spread = math.sqrt(var)
        conf = _clamp(100.0 - spread * 1.6)
        if not ignition_available:
            conf *= 0.8     # a missing modality is itself a confidence penalty
        return round(conf, 1)

    # -- main --------------------------------------------------------------
    def update(
        self,
        reading: dict,
        now: Optional[float] = None,
    ) -> RiskResult:
        now = now if now is not None else time.time()
        flags = self.check_sanity(reading)

        gas_raw = float(reading.get("gas_ppm", 0.0) or 0.0)
        temp = float(reading.get("temp_c", 0.0) or 0.0)
        hum = float(reading.get("humidity_pct", 0.0) or 0.0)

        gas_c = compensate_gas(gas_raw, hum, temp)
        # de-spike: median of the last few readings (see GAS_MEDIAN_WINDOW)
        self._gas_window.append(gas_c)
        if len(self._gas_window) >= 2:
            gas_c = sorted(self._gas_window)[len(self._gas_window) // 2]

        # ---- calibration (feature A) ------------------------------------
        # Only learn from quiet readings; never absorb an event into "normal".
        if not self.calibrator.done:
            if self.calibrator.should_observe(gas_c):
                self.calibrator.observe(gas_c, temp, hum, now=now)
            self.calibrator.tick(now)      # clock advances regardless
        profile = self.calibrator.profile()

        self.history.append((now, gas_c, temp, hum))
        sl = self._slopes(now)

        # ---- sub-scores --------------------------------------------------
        gas_level_score = normalize(gas_c, profile["gas_normal_max"],
                                    profile["gas_critical"])
        # Rate-of-rise term: HOW FAST the gas is climbing is information a fixed
        # threshold throws away. A slow drift to 300 ppm and a 30-second sprint
        # to 300 ppm are not the same event.
        rate_bonus = _clamp(sl.gas / max(C.RATE_ALERT_GAS_SLOPE, 1e-6)
                            * C.RATE_BONUS_MAX, 0.0, C.RATE_BONUS_MAX)
        if not sl.mature:
            rate_bonus = 0.0
        gas_score = _clamp(gas_level_score + rate_bonus)
        temp_score = normalize(temp, profile["temp_normal_max"],
                               profile["temp_critical"])

        # ---- ignition channel: IR flame sensor ---------------------------
        # The IR sensor measures a flame's infrared emission directly. That is
        # a physical measurement rather than an inference from colour, so it is
        # trusted on its own - but only after a debounce, because a single
        # isolated positive is far more likely to be a glitch or a reflection
        # than a fire, and ignition hard-escalates the score.
        flame_available = C.FLAME_SENSOR_PRESENT and ("flame_detected" in reading)
        flame_raw = bool(reading.get("flame_detected", False)) and flame_available
        self._flame_streak = self._flame_streak + 1 if flame_raw else 0
        flame_sensor = self._flame_streak >= C.FLAME_CONFIRM_SAMPLES

        ignition_score = C.IGNITION_IR_ONLY if flame_sensor else 0.0
        ignition = ignition_score > 0.0
        ignition_available = flame_available

        # Chemical + thermal, renormalised over the two channels that are always
        # present. The ignition channel is an ESCALATOR, not a fixed-weight
        # input: seeing no flame is not evidence of safety (the source may be
        # outside the sensor's cone), so it can only ever raise the score,
        # never dilute it.
        wg, wt = C.WEIGHTS["gas"], C.WEIGHTS["temp"]
        chem_thermal = (wg * gas_score + wt * temp_score) / (wg + wt)
        if ignition:
            wi = C.WEIGHTS["ignition"]
            score_raw = (1.0 - wi) * chem_thermal + wi * ignition_score
            # A confirmed flame is a direct physical measurement, not a
            # statistical inference, so it hard-escalates rather than merely
            # contributing weight.
            score_raw = max(score_raw, C.IGNITION_FLOOR_WARNING)
            if chem_thermal >= C.IGNITION_CRITICAL_CHEM_MIN:
                score_raw = max(score_raw, C.IGNITION_FLOOR_CRITICAL)
        else:
            score_raw = chem_thermal

        # ---- rapid-rise override ----------------------------------------
        rapid = sl.mature and sl.gas >= C.RATE_ALERT_GAS_SLOPE
        if rapid:
            score_raw = max(score_raw, C.BAND_WARNING + 8.0)

        # ---- signature analysis + suppression (feature B) ----------------
        signature, reason = classify_signature(sl, ignition)
        score = score_raw
        suppressed = False

        # Cooking-context latch: once a cooking episode is recognised, elevated
        # gas stays EXPLAINED for a while after the rise flattens off - otherwise
        # the system alarms the moment someone stops stirring, which is exactly
        # the nuisance behaviour we are trying to remove. The latch is cleared
        # the instant the pattern turns into a leak or a fire, so a real leak
        # starting during or right after cooking is not masked.
        if signature in ("leak", "fire") or ignition:
            self._cooking_until = 0.0
            self._cooking_long_until = 0.0
            self._cooking_streak = 0
        elif signature == "cooking":
            self._cooking_streak += 1
            if self._cooking_streak >= C.COOKING_CONFIRM_SAMPLES:
                self._cooking_until = now + C.COOKING_LATCH_S
            if self._cooking_streak >= C.SUPPRESSION_HIGH_GAS_STREAK:
                # a long, established episode - remember it through the decay
                self._cooking_long_until = now + C.COOKING_LATCH_S
        else:
            self._cooking_streak = 0

        cooking_confirmed = self._cooking_streak >= C.COOKING_CONFIRM_SAMPLES
        cooking_context = cooking_confirmed or now < self._cooking_until

        # Cooking is a gradual process. A fast burst of combustion products -
        # smouldering material, a match just extinguished, smoke wafted in - is
        # not a cooking signature whatever the humidity is doing, so a rapid
        # rise refuses suppression outright. (Measured: a real smouldering
        # sample peaked at 72 ppm/s; simulated cooking peaks near 5 ppm/s.)
        may_suppress = (cooking_context and not ignition and not rapid
                        and score_raw < C.SUPPRESSION_MAX_SCORE)
        if gas_score >= C.SUPPRESSION_MAX_GAS_SCORE:
            # chemical channel is shouting - only a long-established cooking
            # episode may explain it, never a few seconds of humidity
            established = (self._cooking_streak >= C.SUPPRESSION_HIGH_GAS_STREAK
                           or now < self._cooking_long_until)
            may_suppress = may_suppress and established
        if may_suppress:
            score = score_raw * C.SUPPRESSION_FACTOR
            suppressed = True
            if not cooking_confirmed:
                reason = ("elevated gas still explained by the cooking episode "
                          "detected moments ago (humidity rose with it)")
        elif signature == "leak":
            score = min(100.0, score_raw * 1.15)   # no cooking excuse -> escalate

        score = round(_clamp(score), 1)

        # ---- level -------------------------------------------------------
        level = self._level_with_hysteresis(score, self.level)
        # minimum dwell: escalate freely, but do not stand down immediately
        rank = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}
        if (rank[level] < rank[self.level]
                and (now - self._level_since) < C.LEVEL_MIN_DWELL_S):
            level = self.level

        # Until calibration completes, the absolute thresholds are not trusted -
        # a warming sensor sitting at an odd baseline must not paint the whole
        # dashboard red for its first two minutes. Real ignition and a genuine
        # rapid rise still alarm, because those do not depend on the baseline.
        # Suppressing alarms while the baseline is unknown protects against a
        # warming sensor painting the dashboard red. It must NEVER silence a
        # recognised event: a confirmed ignition, a rapid rise, a leak or fire
        # signature, or a score that has reached CRITICAL all override it.
        if (not self.calibrator.done
                and level != "NORMAL"
                and not (ignition or rapid)
                and signature not in ("leak", "fire")
                and score < C.BAND_CRITICAL):
            level = "NORMAL"
            flags.append(
                "Still calibrating - elevated reading not yet treated as an "
                "alarm because this node's normal level is still being learned.")

        # ---- confidence + disagreement -----------------------------------
        # Agreement is measured over the channels that are actually saying
        # something: a quiet ignition channel in a quiet room is agreement, not
        # disagreement, so a zero is only included once ignition has fired.
        subs = [gas_score, temp_score]
        if ignition:
            subs.append(ignition_score)
        confidence = self._confidence(subs, ignition_available)

        if ignition and gas_score < 30 and temp_score < 30:
            flags.append(
                "IR flame sensor triggered while gas and temperature are still "
                "normal - very early ignition, or a heat source in the sensor's "
                "cone. Verify.")
        if flame_raw and not flame_sensor:
            flags.append(
                f"Single isolated flame reading from the IR sensor - treated as "
                f"noise until {C.FLAME_CONFIRM_SAMPLES} consecutive samples "
                f"confirm it.")
        if (gas_score > 60 or temp_score > 60) and not ignition and ignition_available:
            flags.append(
                "Chemical/thermal anomaly without ignition confirmation - the "
                "source may be outside the sensor's line of sight. Verify manually.")
        if not ignition_available:
            flags.append("No ignition channel available (no flame sensor "
                         "reported) - running on chemical + thermal only.")

        self.score_history.append((now, score))

        # ---- countdown ---------------------------------------------------
        eta = self._time_to_critical(now, score)

        # ---- latency stopwatch vs the naive detector ---------------------
        naive_alarm = gas_raw >= C.NAIVE_GAS_THRESHOLD_PPM
        if naive_alarm and self._naive_trigger_t is None:
            self._naive_trigger_t = now
        if level != "NORMAL" and self._fusion_trigger_t is None:
            self._fusion_trigger_t = now
        if (
            self._fusion_trigger_t is not None
            and self._naive_trigger_t is not None
            and self.latency_gain_s is None
        ):
            self.latency_gain_s = round(self._naive_trigger_t - self._fusion_trigger_t, 2)

        # ---- state transition log ---------------------------------------
        if level != self.level:
            self._level_since = now
            self.events.append({
                "t": now, "from": self.level, "to": level,
                "score": score, "signature": signature, "reason": reason,
            })
            self.level = level

        res = RiskResult(
            ts=now, environment=self.environment,
            gas_raw=round(gas_raw, 1), gas_compensated=round(gas_c, 1),
            temp_c=round(temp, 2), humidity_pct=round(hum, 2),
            flame_sensor=flame_sensor, flame_available=flame_available,
            gas_score=round(gas_score, 1), temp_score=round(temp_score, 1),
            ignition_score=round(ignition_score, 1),
            score_raw=round(score_raw, 1), score=score, level=level,
            confidence=confidence, signature=signature, signature_reason=reason,
            suppressed=suppressed, rapid_rise=rapid, flags=flags, slopes=sl,
            seconds_to_critical=eta,
            calibrating=not self.calibrator.done,
            calibration_progress=round(self.calibrator.progress, 3),
            calibration_note=self.calibrator.adjustment_note,
            naive_alarm=naive_alarm, latency_gain_s=self.latency_gain_s,
        )
        res.explanation = explain(res)
        return res


# ---------------------------------------------------------------------------
# 7. explainability
# ---------------------------------------------------------------------------
def _band_word(sub: float) -> str:
    if sub < 25:
        return "normal"
    if sub < 45:
        return "slightly elevated"
    if sub < 70:
        return "elevated"
    return "high"


def _ignition_words(r: RiskResult) -> str:
    if not r.flame_available:
        return "IR flame sensor: not reporting"
    return "IR flame sensor: " + ("TRIGGERED" if r.flame_sensor else "clear")


def explain(r: RiskResult) -> str:
    """Plain-language narration of WHY the score is what it is."""
    bits = [
        f"Gas {_band_word(r.gas_score)} ({r.gas_compensated:.0f} ppm compensated "
        f"from {r.gas_raw:.0f} raw)",
        f"temperature {_band_word(r.temp_score)} ({r.temp_c:.1f} C)",
        f"humidity {r.humidity_pct:.0f} %",
        _ignition_words(r),
    ]
    line = f"{r.environment}: " + ", ".join(bits) + f" -> {r.level} ({r.score:.0f}/100)."
    if r.suppressed:
        line += f" Suppressed: {r.signature_reason}."
    elif r.signature in ("leak", "fire"):
        line += f" Pattern: {r.signature} - {r.signature_reason}."
    if r.rapid_rise:
        line += f" Rapid rise: {r.slopes.gas:+.1f} ppm/s."
    if r.seconds_to_critical is not None and r.level != "CRITICAL":
        line += f" Estimated {r.seconds_to_critical:.0f} s to CRITICAL if the trend holds."
    return line
