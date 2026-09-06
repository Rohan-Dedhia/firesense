# FireSense

Multi-sensor fire and gas-leak detection. A conventional detector alarms when
one number crosses one line. FireSense fuses four channels and reasons about
*how* the readings are changing, which is what lets it catch a leak earlier and
stay quiet through cooking.

```
ESP32 ──USB serial──▶  ingest  ──▶  fusion engine  ──▶  Streamlit dashboard
 MQ2 · BME280 · IR flame                │                     └─▶ Telegram alert
 (+ laptop webcam, optional)            └─▶ CSV incident log
```

---

## Run it

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows the three numbered `.bat` files do the same thing by double-click:

| file | what it does |
|---|---|
| `1_SETUP.bat` | installs dependencies, checks the machine, runs the tests |
| `2_RUN_DASHBOARD.bat` | starts the dashboard |
| `3_CHECK_HARDWARE.bat` | proves the USB link to the node before you rely on it |

Close the Arduino Serial Monitor before connecting — only one program may hold
a COM port.

For deploying a shareable copy, see **STREAMLIT_HOSTING.md**. For the demo
running order, see **DEMO_SCRIPT.md**.

---

## How the score is built

Four channels, each scored 0–100, then combined:

| channel | sensor | weight | reads |
|---|---|---|---|
| chemical | MQ2 | 0.35 | combustible gas, humidity/temperature compensated |
| thermal | BME280 | 0.25 | temperature, and its rate of rise |
| optical | webcam | ignition | flame colour by HSV threshold |
| IR flame | flame sensor | ignition | direct flame presence |

The two ignition channels fuse into one and act as an **escalator**: they can
only raise the score, never dilute it, and a trusted ignition source forces a
hard floor under the result. That is deliberate — a fire that the camera can
see should never be averaged down by a calm gas reading.

On top of the levels, the engine classifies the *pattern* over a rolling window:

- **cooking** — gas and humidity rising together (combustion produces water
  vapour). Suppressed, because this is the classic nuisance alarm.
- **leak** — gas rising while humidity is flat or falling. Escalated.
- **fire** — gas rising, temperature climbing fast, ignition confirmed.
- **rising / steady** — not yet distinctive.

Suppression has hard limits: it is refused when an ignition channel fires, when
the chemical channel is already shouting, or when the rise is *rapid*. That
last one came out of real hardware — a smouldering match measured **+72 ppm/s**
against a cooking peak near **+5 ppm/s**, so a fast burst can never be excused
as cooking however the humidity behaves.

---

## Environments, and why sample counts don't matter

Two independent mechanisms, because rooms differ in two independent ways.

**1. Profile — what kind of room is this?** `ENVIRONMENT_PROFILES` in
`config.py` sets different notions of normal:

| | gas normal | gas critical | temp normal |
|---|---|---|---|
| Kitchen | 300 | 700 | 40 °C |
| Factory | 150 | 500 | 45 °C |
| Storage Room | see config | | |
| Lab | 120 | 450 | 35 °C |

A kitchen tolerates cooking fumes; a lab should not have solvent vapour in the
air at all, so the same reading scores very differently in each.

**2. Calibration — what is *this particular room* actually like?** On start-up
(and whenever you switch environment) the engine watches the air and sets the
thresholds from what it sees, so one node's baseline never judges another
room's air.

Calibration is driven by **elapsed time plus a minimum sample floor**, not by a
fixed number of readings:

- `CALIBRATION_SECONDS = 120` — how long to observe
- `CALIBRATION_MIN_SAMPLES = 30` — the floor before it will commit
- `CALIBRATION_GIVE_UP_S = 240` — hard stop; falls back to the profile and says
  so, rather than stalling forever

So a fast serial link delivering many readings and a slow dashboard delivering
few both calibrate correctly: the fast one simply collects more samples in the
same two minutes. The same applies to trend detection, which needs
`RATE_MIN_SAMPLES = 5` spread over at least half the slope window before it
will call a pattern — it reports `insufficient_data` instead of guessing from
three points.

If a room's idle level is so far above the profile that the absolute ppm scale
is meaningless (an uncalibrated MQ2 R0, or genuinely dirty air), calibration
switches to **relative mode** and sets thresholds as multiples of that node's
own baseline instead.

---

## Files

| file | role |
|---|---|
| `config.py` | every tunable number and the hardware data contract |
| `fusion.py` | calibration, scoring, pattern classification, suppression |
| `ingest.py` | data sources: serial, mock, HTTP, MQTT, replay |
| `serial_parse.py` | parses the node's Serial Monitor text into the contract |
| `vision.py` | webcam capture and HSV flame detection |
| `alerts.py` | Telegram alerting with cooldown, plus the CSV incident log |
| `app.py` | the dashboard |
| `mock_source.py` | synthetic scenarios for Test Mode |
| `auto_tune_vision.py` | learns the flame's colour signature in your room |
| `benchmark.py` | fusion vs a fixed-threshold detector, across scenarios |
| `replay_csv.py` | re-runs a captured session CSV through the engine |
| `read_serial.py` | standalone USB link check |
| `check_setup.py` | environment sanity check |
| `run_tests.py` | test runner (34 tests, no pytest needed) |
| `data/` | a real captured hardware session, used as a regression fixture |

---

## Tests

```bash
python run_tests.py
```

34 tests. They cover the ordinary cases, but the ones worth knowing about came
from things that actually went wrong on real hardware:

- a leak already under way at power-on still alarms (calibration cannot deadlock)
- calibration always terminates, even if the room never goes quiet
- levels do not flap across a band boundary
- a real captured session is replayed and must never suppress a high-gas event
- a fast smoke burst is never excused as cooking

---

## Honest limits

- The optical channel is HSV colour thresholding, not a trained model. It needs
  tuning per room and it is fooled by warm-coloured backgrounds.
- Nothing here is a trained classifier, so there is no accuracy figure to quote —
  the validation is the test suite plus replay against real captured data.
- MQ2 ppm values are not calibrated against a reference gas. Relative mode
  exists precisely because the absolute scale cannot be trusted.
- Time-to-critical is a short-horizon linear extrapolation, not a forecast.
