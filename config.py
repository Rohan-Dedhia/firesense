"""
FireSense - central configuration.
Everything tunable lives here. No magic numbers anywhere else in the codebase.
"""

import os

# ---------------------------------------------------------------------------
# DATA CONTRACT  (the hardware <-> software seam - DO NOT change without telling
# the hardware side immediately; both tracks build against exactly this shape)
# ---------------------------------------------------------------------------
# The ESP32 publishes this JSON object, once per second:
#
# {
#   "node_id":        "node-01",                 # str
#   "timestamp":      "2026-08-30T14:23:00Z",    # ISO-8601 UTC
#   "gas_ppm":        340.0,                     # float, MQ2 converted reading
#   "gas_raw_adc":    1820,                      # int, 0-4095 raw ADC (debug)
#   "temp_c":         26.4,                      # float, BME280
#   "humidity_pct":   55.0,                      # float, BME280
#   "pressure_hpa":   1008.2,                    # float, BME280 (not scored)
#   "flame_detected": false                      # bool - IR flame sensor
# }
#
# flame_detected: set FLAME_SENSOR_PRESENT below to False if the IR flame
# sensor is ever removed. The ESP32 then sends false and the software ignores
# the field entirely, so the contract never changes either way.
REQUIRED_FIELDS = ("timestamp", "gas_ppm", "temp_c", "humidity_pct")
OPTIONAL_FIELDS = ("node_id", "gas_raw_adc", "pressure_hpa", "flame_detected")

# >>> THE ONLY TWO LINES THE HARDWARE SIDE NEEDS TO TOUCH <<<
# Put the ESP32's IP here once it is on WiFi, then set DEFAULT_SOURCE = "http".
# Until then leave it on "mock" and everything runs off the simulator.
ESP32_URL = "http://192.168.1.50/reading"
DEFAULT_SOURCE = "mock"          # "mock" | "http" | "mqtt"
# Polling an ESP32's tiny single-threaded web server needs patience: it serves
# one request at a time and can take a moment while it is also reading sensors.
HTTP_TIMEOUT_S = 3.0
HTTP_RETRIES = 2
HTTP_MIN_INTERVAL_S = 0.5     # never poll a real node faster than this
HTTP_STALE_GRACE_S = 6.0      # keep showing the last good reading this long

MQTT_HOST = "broker.hivemq.com"
MQTT_TOPIC = "firesense/node01"

# A real MQ2 is spiky - single-sample excursions of several hundred ppm are
# normal when gas is wafted past it. A 3-point median removes those spikes
# without delaying a genuine sustained rise (unlike an average, a median is
# immune to one outlier).
GAS_MEDIAN_WINDOW = 3

# Once a level has been raised, hold it for at least this long before standing
# down. Without it a spiky sensor produces a flurry of state changes for what a
# human would call one event - noisy alerts, and an unstable-looking log.
LEVEL_MIN_DWELL_S = 20.0

SAMPLE_INTERVAL_S = 1.0          # normal sampling cadence
FAST_SAMPLE_INTERVAL_S = 0.2     # adaptive: used once score enters WARNING

# ---------------------------------------------------------------------------
# ENVIRONMENT PROFILES  (the "personalisation" feature)
# Same sensors, different notion of "normal" per deployment context.
# ---------------------------------------------------------------------------
ENVIRONMENT_PROFILES = {
    "Kitchen": {
        "label": "Kitchen / domestic LPG",
        "gas_normal_max": 300.0,   # cooking fumes expected -> tolerant
        "gas_critical": 700.0,
        "temp_normal_max": 40.0,   # cooking heat expected
        "temp_critical": 60.0,
        "humidity_normal_max": 75.0,
        "notes": "High tolerance for correlated gas+humidity rise (cooking).",
    },
    "Factory": {
        "label": "Factory / workshop",
        "gas_normal_max": 150.0,   # combustible vapour NOT expected -> sensitive
        "gas_critical": 500.0,
        "temp_normal_max": 45.0,   # machinery runs warm
        "temp_critical": 65.0,
        "humidity_normal_max": 70.0,
        "notes": "Low gas tolerance, high thermal tolerance.",
    },
    "Storage Room": {
        "label": "Storage / godown",
        "gas_normal_max": 100.0,   # nothing should be off-gassing
        "gas_critical": 400.0,
        "temp_normal_max": 32.0,
        "temp_critical": 50.0,
        "humidity_normal_max": 65.0,
        "notes": "Most sensitive profile - unoccupied, no legitimate gas source.",
    },
    "Lab": {
        "label": "Laboratory",
        "gas_normal_max": 120.0,
        "gas_critical": 450.0,
        "temp_normal_max": 35.0,
        "temp_critical": 55.0,
        "humidity_normal_max": 60.0,
        "notes": "Solvent vapour possible but should be extracted.",
    },
}
DEFAULT_ENVIRONMENT = "Kitchen"

# ---------------------------------------------------------------------------
# FUSION WEIGHTS
# ---------------------------------------------------------------------------
# IS THE IR FLAME SENSOR WIRED? Flip this to False if it is ever removed.
# Nothing else needs changing - the contract already carries the field, and the
# software falls back to chemical + thermal on its own.
FLAME_SENSOR_PRESENT = True

WEIGHTS = {
    "gas": 0.35,        # chemical modality (MQ2)    - always-present channel
    "temp": 0.25,       # thermal modality (BME280)  - always-present channel
    "ignition": 0.40,   # IGNITION channel (IR flame sensor). An escalator, not
                        # a fixed weight: no flame seen is NOT evidence of
                        # safety (the source may be outside the sensor's cone),
                        # so this channel can only ever raise the score.
}
IGNITION_IR_ONLY = 92.0       # a confirmed IR flame: a real physical measurement
# A confirmed flame is a direct physical measurement, not a statistical
# inference, so it hard-escalates rather than merely contributing weight.
IGNITION_FLOOR_WARNING = 50.0   # trusted ignition -> at least WARNING
IGNITION_FLOOR_CRITICAL = 77.0  # trusted ignition + elevated gas/temp -> CRITICAL
IGNITION_CRITICAL_CHEM_MIN = 40.0
# A real flame stays in view for many consecutive samples. A single isolated
# 'flame' reading is far more likely to be an IR glitch or a reflection, and
# because ignition hard-escalates the score, one bad sample must not be able to
# throw a CRITICAL. Require this many consecutive positives.
FLAME_CONFIRM_SAMPLES = 2

# Risk bands (0-100)
BAND_WARNING = 45.0
BAND_CRITICAL = 75.0
COUNTDOWN_TARGET = BAND_CRITICAL   # "time to critical" extrapolates to this
# Hysteresis: escalate the instant the score crosses a band, but require it to
# fall this far BELOW the band before standing down. Without it a score sitting
# on a boundary flaps between levels every few seconds, spamming alerts and
# making a stable situation look unstable.
LEVEL_HYSTERESIS = 5.0

# Naive single-sensor baseline used ONLY for the latency comparison demo.
# This is the "conventional detector" we benchmark against.
NAIVE_GAS_THRESHOLD_PPM = 450.0

# ---------------------------------------------------------------------------
# HUMIDITY / TEMPERATURE COMPENSATION  (MQ2 output drifts with RH and T)
# ---------------------------------------------------------------------------
RH_REFERENCE = 65.0     # datasheet characterisation point
RH_COEFF = 0.004        # ~0.4 % per %RH deviation
TEMP_REFERENCE = 20.0
TEMP_COEFF = 0.002      # secondary temperature term

# ---------------------------------------------------------------------------
# AUTO-CALIBRATION (feature A)
# ---------------------------------------------------------------------------
CALIBRATION_SECONDS = 120          # demo-friendly; 600-1800 for real deploys
CALIBRATION_MIN_SAMPLES = 30
CALIBRATION_PERCENTILE = 90        # p90 of idle readings ...
CALIBRATION_HEADROOM = 1.25        # ... times this becomes gas_normal_max
CALIBRATION_MAX_SHRINK = 0.5       # never tighten below 50 % of profile value
CALIBRATION_MAX_GROWTH = 2.5       # never loosen beyond 250 % of it
# A freshly powered MQ2 drifts for 20-30 min while the heater reaches thermal
# equilibrium. Calibrating during that drift learns a baseline that is simply
# wrong. So the calibrator refuses to lock in while the samples are still
# trending, and says so, instead of quietly producing bad thresholds.
CALIBRATION_DRIFT_TOLERANCE = 0.12   # >12 % move across the window = still warming
CALIBRATION_MAX_SECONDS = 1800.0     # (retained for reference; the cap that
                                     #  actually applies is GIVE_UP below)
# If the room is never quiet enough to learn a baseline (someone is cooking, or
# there is a real leak in progress at start-up), we must NOT wait forever - the
# system would stay silent through the very event it exists to catch. After this
# long we give up, say so, and fall back to the factory profile for the chosen
# environment.
CALIBRATION_GIVE_UP_S = 240.0
# If the learned idle level is WAY above the profile, the MQ2's R0 has not been
# calibrated (very likely on demo day). Rather than sit pinned at CRITICAL, the
# calibrator switches to RELATIVE mode: thresholds become multiples of whatever
# this room's idle level actually is. The system then works on "how far above
# ITS OWN normal is this" instead of absolute ppm - which is what it should have
# been doing anyway, and is the honest answer to "does it generalise".
RELATIVE_MODE_TRIGGER = 3.0        # learned idle > 3x profile -> go relative
RELATIVE_NORMAL_MULT = 1.4         # normal band  = idle x 1.4
RELATIVE_CRITICAL_MULT = 3.0       # critical     = idle x 3.0

# ---------------------------------------------------------------------------
# SLOPES / SIGNATURE ANALYSIS (feature B)
# ---------------------------------------------------------------------------
SLOPE_WINDOW_S = 20                # rolling window for d/dt and for countdown
COOKING_GAS_SLOPE_MIN = 0.8        # ppm per second
COOKING_HUM_SLOPE_MIN = 0.05       # %RH per second - humidity rises WITH gas
LEAK_HUM_SLOPE_MAX = 0.0           # leak: humidity flat/falling while gas rises
SUPPRESSION_FACTOR = 0.55          # score multiplier when cooking sig. matched
COOKING_CONFIRM_SAMPLES = 4        # consecutive cooking classifications needed
                                   # before suppression engages (debounce: one
                                   # noisy humidity sample must not excuse a leak)
COOKING_LATCH_S = 150.0            # how long a recognised cooking episode keeps
                                   # explaining elevated gas after the rise stops
                                   # (cleared instantly by a leak/fire signature)
SUPPRESSION_MAX_SCORE = 85.0       # never suppress a near-critical event away
# Real-hardware finding: a brief humidity transient (a breath near the BME280,
# a hand passing over it) can look exactly like the cooking signature and
# suppressed a genuine gas event. Context may excuse a MODERATE rise; it must
# never excuse the chemical channel on its own shouting.
SUPPRESSION_MAX_GAS_SCORE = 70.0
# ... but a genuine cooking episode is sustained for minutes, while a breath or
# a passing hand lasts seconds. So at high gas we do not refuse suppression
# outright - we demand that the cooking signature has held this many
# consecutive samples first. Real cooking clears that bar easily; a transient
# never does.
SUPPRESSION_HIGH_GAS_STREAK = 12
RATE_ALERT_GAS_SLOPE = 6.0         # ppm/s - rapid-rise trigger regardless of level
RATE_BONUS_MAX = 25.0              # max points a fast rise adds to the gas sub-score
# A slope fitted through 3 noisy samples is meaningless - on a sensor with
# +/-60 ppm of noise it can read tens of ppm/s. Rate-of-rise is only allowed to
# influence anything once the window is actually populated.
# 8 samples assumed ~1 Hz polling. A Streamlit rerun over live charts can take
# 2-3 s, which left the window short and the pattern stuck on
# "insufficient_data". 5 samples spread over at least half the window is still
# a 4-sigma test against sensor noise.
RATE_MIN_SAMPLES = 5
RATE_MIN_WINDOW_FRACTION = 0.5     # of SLOPE_WINDOW_S

# Countdown gating - never extrapolate off ordinary idle wobble
COUNTDOWN_MIN_SLOPE = 0.25         # risk-points per second
COUNTDOWN_MIN_SCORE = 36.0         # only once the situation is actually developing
COUNTDOWN_MAX_HORIZON_S = 900.0

# ---------------------------------------------------------------------------
# SENSOR SANITY (self-diagnostic + fault detection)
# ---------------------------------------------------------------------------
SANE_RANGES = {
    "gas_ppm": (0.0, 10000.0),
    "temp_c": (-10.0, 100.0),
    "humidity_pct": (0.0, 100.0),
}
STALE_READING_S = 10.0             # no data for this long -> node offline

# ---------------------------------------------------------------------------
# ALERTS
# ---------------------------------------------------------------------------
# Credentials are resolved in this order, first non-empty wins:
#
#   1. Streamlit "Secrets"            <- use this when the app is HOSTED
#        app.py copies them into the environment at start-up. On Streamlit
#        Community Cloud: Manage app -> Settings -> Secrets, and paste
#            TELEGRAM_TOKEN = "123456:ABC..."
#            TELEGRAM_CHAT_ID = "1493462776"
#        Locally the same lines live in .streamlit/secrets.toml, which is
#        git-ignored so the token never reaches GitHub.
#
#   2. environment variables TELEGRAM_TOKEN / TELEGRAM_CHAT_ID
#
#   3. the literals below - convenient for a laptop-only demo, but NEVER
#      commit a real token here: a token pushed to a public repository is a
#      public token, and Telegram revokes it.
#
# Get them from: @BotFather -> /newbot (token), then message the bot and open
# https://api.telegram.org/bot<TOKEN>/getUpdates -> result[0].message.chat.id
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
ALERT_COOLDOWN_S = 60   # don't spam the same state
