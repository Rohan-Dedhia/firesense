# FireSense

Multi-sensor fire and gas detection that tries to solve the problem the
industry has openly failed to solve: **telling a real fire apart from cooking.**

NIST's own 2024 write-up on residential alarms is titled *"New Smoke Alarms Are
Better at Detecting Fires but Still Beep for Bacon."* That nuisance-alarm
problem is the thesis of this project. A conventional detector alarms when one
number crosses one line. FireSense fuses chemical, thermal, ignition, visual and
semantic evidence, and reasons about *how* the readings are changing — which is
what lets it catch a leak earlier and stay quiet through a stir-fry.

```
ESP32 ──USB/WiFi──▶ ingest ─┐
 MQ2 · BME280 · IR flame    │
                            ├──▶ rule engine (baseline) ─┐
ESP32-CAM / webcam ─────────┼──▶ YOLO fire/smoke ────────┼──▶ decision
                            └──▶ VLM, on escalation ─────┘      │
                                                                ├──▶ Telegram
                                                                │    photo + buttons
                                                                └──▶ CSV log
                                                                     ↓
                                                            labelled examples
                                                                     ↓
                                                            learned model
```

The loop at the bottom is the point. Every alert the system sends is answerable
with one tap — **Real** or **False alarm** — and each tap becomes a labelled
training example with the full sensor window that caused it. The system gets
better the longer it runs.

---

## Two ways to run it

**The dashboard** — what you show someone. Sensors only, hosted-friendly.

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

**The pipeline** — the actual system, with camera and VLM. Local only.

```bash
python -m pip install -r requirements-ml.txt

python pipeline.py --source mock --scenario sudden_fire        # no hardware
python pipeline.py --source mock --scenario cooking_false_positive
python pipeline.py --source serial --camera webcam             # node plugged in
python pipeline.py --source serial --camera 192.168.1.42 --vlm # ESP32-CAM + VLM
```

Can't find the node? Run the diagnostic:

```bash
python ingest.py
```

It lists every serial port, marks the ones that could plausibly be hardware, and
says which one auto-pick will take. On macOS `/dev/cu.debug-console` and the
Bluetooth port are always visible and are **not** your node — they're filtered
out. Close the Arduino Serial Monitor first; only one program may hold a port.

For deploying a shareable copy see **STREAMLIT_HOSTING.md**; for the demo
running order, **DEMO_SCRIPT.md**; for data collection,
**RECORDING_RUNBOOK.md**.

---

## The five channels

| channel | source | weight | reads |
|---|---|---|---|
| chemical | MQ2 | 0.35 | combustible gas, humidity/temperature compensated |
| thermal | BME280 | 0.25 | temperature, and its rate of rise |
| ignition | IR flame sensor | 0.40 | direct flame presence, debounced |
| visual | YOLO on camera frames | escalator | fire and smoke, separately weighted |
| semantic | VLM (Ollama) | escalator | *what kind* of fire, and whether anyone is there |

### The escalator rule

Ignition, visual and semantic channels can only **raise** the score, never
dilute it. This is the single most important design decision in the project and
it is justified by measurement, not taste:

- A flame outside the IR sensor's narrow cone is invisible to it. A zero there
  is absence of evidence, not evidence of safety.
- On this project's own webcam, a community fire model reported *fire* on an
  empty room for **twenty consecutive frames** at 0.40–0.45 confidence. A
  channel that can be wrong in that direction must never be allowed to
  contribute a confident zero.

A confirmed flame requires **two consecutive** positive samples; single isolated
positives show on screen as noise rather than being acted on.

### Pattern classification

On top of the levels, the engine classifies the *pattern* over a rolling window:

- **cooking** — gas and humidity rising together (combustion makes water
  vapour). Suppressed; this is the classic nuisance alarm.
- **leak** — gas rising while humidity is flat or falling. Escalated.
- **fire** — gas rising, temperature climbing fast, ignition confirmed.
- **rising / steady** — not yet distinctive.

Suppression has hard limits. It is refused when an ignition channel fires, when
the chemical channel is already shouting, or when the rise is *rapid*. That last
one came out of real hardware: a smouldering match measured **+72 ppm/s** against
a cooking peak near **+5 ppm/s**, so a fast burst can never be excused as cooking
however the humidity behaves.

### What the VLM adds that YOLO cannot

YOLO answers *is there fire in this frame*. That is not the question this project
is about — a stove flame and a burning sofa are both fire. The VLM answers in a
fixed format and the answer is scored:

```
SCENE:  a metal pan on a lit gas burner, someone standing beside it
FIRE:   contained     SOURCE: stove      PEOPLE: yes     ->  8/100
SCENE:  flames spreading across a fabric sofa in an empty room
FIRE:   uncontained   SOURCE: furniture  PEOPLE: no      -> 92/100 (+8 unattended)
```

"Contained" on a stove is 8. "Contained" on a sofa is 70. Nobody in frame adds 8,
because unattended cooking is the leading cause of home fires — the thing NIST's
cooktop research is actually about.

It runs **on escalation only**, not every frame. A VLM takes seconds; calling one
at 1 Hz would make the system slower than the fire. It is a verification channel,
which is also how it would deploy.

Vision alone is capped at **60** (`VISION_ALONE_MAX`) when the sensors are calm,
so a hallucinated detection can warn but never reach CRITICAL on its own. Once
the sensors agree, the cap lifts and vision corroborates freely.

---

## Environments, and why sample counts don't matter

Two independent mechanisms, because rooms differ in two independent ways.

**1. Profile — what kind of room is this?** `ENVIRONMENT_PROFILES` in
`config.py`:

| | gas normal | gas critical | temp normal |
|---|---|---|---|
| Kitchen | 300 | 700 | 40 °C |
| Factory | 150 | 500 | 45 °C |
| Storage Room | 100 | 400 | 32 °C |
| Lab | 120 | 450 | 35 °C |

A kitchen tolerates cooking fumes; a lab should not have solvent vapour in the
air at all, so the same reading scores very differently in each.

**2. Calibration — what is *this particular room* actually like?** On start-up
(and whenever you switch environment) the engine watches the air and sets
thresholds from what it sees. Driven by **elapsed time plus a sample floor**,
not a fixed count:

- `CALIBRATION_SECONDS = 120` — how long to observe
- `CALIBRATION_MIN_SAMPLES = 30` — the floor before it will commit
- `CALIBRATION_GIVE_UP_S = 240` — hard stop; falls back to the profile and says
  so, rather than stalling forever

So a fast serial link and a slow dashboard both calibrate correctly — the fast
one simply collects more samples in the same two minutes. Trend detection needs
`RATE_MIN_SAMPLES = 5` over at least half the slope window before it will name a
pattern; otherwise it reports `insufficient_data` instead of guessing from three
points.

If a room's idle level is so far above the profile that absolute ppm is
meaningless (an uncalibrated MQ2 R0, or genuinely dirty air), calibration
switches to **relative mode** and sets thresholds as multiples of that node's own
baseline.

---

## The ML pipeline

### Recording

```bash
python record_session.py cooking --minutes 10
python record_session.py normal --minutes 15 --camera none
python record_session.py candle --camera 192.168.1.42
```

Ten fixed labels — six nuisances that **must not** alarm (`normal`, `cooking`,
`steam`, `incense`, `spray`, `dust`) and four dangers that **must**
(`candle`, `smoulder`, `fire`, `gas`). The vocabulary is fixed on purpose: a
dataset where one session says `cooking` and another says `cooking_oil` is a
dataset you have to clean later.

Each session becomes one folder — `meta.json`, `sensors.csv`, `frames/*.jpg`
named by milliseconds — all on **one shared wall clock**, so sensors and frames
align afterwards by timestamp with no further bookkeeping.

### Features

`dataset.py` makes three decisions that matter more than model architecture:

1. **Split by session, never by window.** Windows from one recording are hugely
   correlated. Shuffle them across train and test and the model effectively sees
   the test set — accuracy looks superb and means nothing. `split_by_session()`
   is the *only* split function in the file, deliberately. There is no
   `split_by_window()` to reach for by accident.
2. **Relative features, not raw ppm.** Every feature is expressed against the
   session's own quiet baseline. Verified: two synthetic `normal` sessions at
   100 ppm and 300 ppm baselines produce *identical* `gas_rel`. This is what
   makes training across different hardware and different rooms possible at all.
3. **Rates of change are first-class.** Level alone cannot separate cooking from
   a leak; the correlation between gas and humidity *slopes* can.

Eleven features: `gas_rel`, `gas_delta`, `gas_slope`, `gas_slope_rel`,
`gas_std`, `temp_delta`, `temp_slope`, `hum_delta`, `hum_slope`,
`gas_hum_agree`, `flame_frac`.

### Augmentation

Five transforms, each with a physical justification — a different MQ2 unit
(`scale_gas`, because R0 scales the whole curve), a different room
(`shift_baseline`), a slower or faster event (`time_warp`), a noisier board
(`sensor_noise`), a warmer room (`climate_offset`). Generic jitter would say
nothing true about deployment; these five do, which is the difference between
augmentation you can defend in a viva and augmentation that makes the number go
up.

Time warping is drawn from a **fixed ladder**, not sampled, and that is not
fussiness. Warping changes a session's duration, which changes how many windows
it yields. Sampled independently per session, the class balance drifts — measured
on this project's own data, the danger fraction moved from 0.50 to **0.56**
purely from augmentation. Cycling one ladder gives every session the identical
set of warps, so the effect cancels exactly.

Augment the **training** side of a split that has already happened. Never the
test set.

### Training

```bash
python train.py              # train, evaluate, compare against the rule engine
python train.py --cv         # GroupKFold, grouped by session
python train.py --augment 6  # 6x the training set, test set untouched
```

Three candidates (logistic regression, random forest, gradient boosting) and two
non-negotiables:

- **Grouped evaluation.** With ~15 recordings a single train/test split is mostly
  noise, so `--cv` runs `GroupKFold` and reports the spread as well as the mean.
  A number without a spread, on this much data, is not a number. Augmented copies
  share their original's group, or the fold boundary would cut between a session
  and its own copy.
- **An honest baseline.** The comparison is `fusion.FusionEngine` replayed *live*
  over the same held-out sessions, calibrating from each session head exactly as
  it would on the bench — not a strawman threshold. The rule engine passes 34
  tests and was tuned against real hardware, so beating it means something.

**`fusion.py` is deliberately never improved again.** It is the baseline. A
baseline you keep improving is not a baseline. Vision and the VLM combine *on top
of* its verdict in `pipeline.py`, and when the learned model is ready it replaces
that combination step rather than the engine.

### Vision weights

No training needed — these are pretrained detectors, fetched with one command
(Hugging Face serves the `.pt` directly; no account, no Roboflow dialog):

```bash
python get_weights.py --list     # candidates, with licences
python get_weights.py --all      # into models/
python get_weights.py --urls     # or download in a browser
```

**Then measure them instead of trusting them.** Which detector to use, and at
what confidence, is an empirical question about *our* camera in *our* room:

```bash
python pick_vision_model.py --capture normal   # the ordinary room, no flame
python pick_vision_model.py --capture flame    # same room, lighter lit
python pick_vision_model.py --compare
```

That sweeps every downloaded model across six confidence thresholds and reports
how often each cried fire at nothing versus how often it saw a real flame. It
recommends the model with the widest **separation** — the gap between the
loudest false positive and the typical real detection — and sets the threshold
in the middle of that gap, not at the edge where one warm evening breaks it.

A model is rejected if it false-alarms at every threshold, if it is quiet only
because it detects nothing, or if its separation is under 0.10. All three
rejections are tested.

Smoke is not a lesser fire, so classes are weighted: `fire` 1.0, `flame` 1.0,
`smoke` 0.75, anything unrecognised 0.6. Community models often ship non-English
class names — this project's first real run reported `{0: '火', 1: '烟'}` — so
`LABEL_ALIASES` maps the Chinese and Traditional forms onto `fire` and `smoke`.
`CONF_THRESHOLD` is **0.55**, raised from 0.35 after measuring the empty-room
false positives described above.

---

## Files

**Running system**

| file | role |
|---|---|
| `pipeline.py` | the whole system: sensors + camera + VLM + alerts + log |
| `app.py` | the Streamlit dashboard (sensors only, hosting-friendly) |
| `config.py` | every tunable number and the hardware data contract |
| `fusion.py` | calibration, scoring, patterns, suppression — **the baseline** |
| `ingest.py` | data sources: serial, mock, HTTP, MQTT, replay + port diagnostic |
| `serial_parse.py` | parses the node's Serial Monitor text into the contract |
| `camera.py` | webcam and ESP32-CAM frame sources behind one interface |
| `alerts.py` | Telegram text alerts with cooldown, plus CSV incident log |
| `feedback.py` | photo alerts with Real / False-alarm buttons; collects labels |

**ML pipeline**

| file | role |
|---|---|
| `record_session.py` | labelled capture — sensors and frames on one clock |
| `dataset.py` | sessions → windows → features; session-level splitting |
| `augment.py` | five physically motivated transforms |
| `train.py` | three candidates, GroupKFold, and the rule-engine comparison |
| `vision_ml.py` | the YOLO channel |
| `vlm.py` | the semantic channel (Ollama); `--selftest` checks the setup |
| `get_weights.py` | fetches candidate detectors from Hugging Face |
| `pick_vision_model.py` | chooses model and threshold by measurement |

**Utilities**

| file | role |
|---|---|
| `mock_source.py` | synthetic scenarios, for development without hardware |
| `benchmark.py` | fusion vs a fixed-threshold detector across scenarios |
| `replay_csv.py` | re-runs a captured session CSV through the engine |
| `read_serial.py` | standalone USB link check |
| `check_setup.py` | environment sanity check |
| `extract_frames.py` | every Nth frame of a video → `data/vision_check_phone/<name>/` |
| `run_tests.py` | test runner (83 tests, no pytest needed) |
| `data/` | a real captured hardware session, used as a regression fixture |

---

## Testing without hardware

Everything runs without the ESP32 node or the ESP32-CAM. Sensors come from
`--source mock` (scripted scenarios), and the camera can be a video file:

```bash
python pipeline.py --source mock --scenario sudden_fire --camera demo.mp4 --vision maxvit
python pipeline.py --source mock --scenario cooking_false_positive --camera kitchen.mov --camera-loop
```

- `--camera` takes `.mp4`, `.mov`, `.avi` or `.mkv`. The video plays at
  real-time speed, so the pipeline samples it like a live camera.
  `--camera-fps 5` slows it down (or speeds it up), and `--camera-loop`
  restarts it at the end.
- Without `--camera-loop`, the pipeline says "video ended" once and carries on
  with the sensors only. A missing or unreadable file gives a clear startup
  note instead of a crash.
- `record_session.py --camera demo.mp4` also works, and `meta.json` records the
  video as the camera source. Don't record training sessions this way with
  real sensors: the frames would not show what the sensors measured.

**Phone footage is not ESP32-CAM footage.** To look at a model on frames from
a phone video:

```bash
python extract_frames.py candle.mov --every 15     # -> data/vision_check_phone/candle/
```

`data/vision_check/` is the adoption test for a vision model (hard rule 10) and
holds frames from the deployment camera only. `extract_frames.py` refuses to
write there, and `pick_vision_model.py --capture` refuses a video file.
Results on phone frames are phone results. They don't replace the ESP32-CAM
check.

---

## Tests

```bash
python run_tests.py            # everything
python run_tests.py fusion     # one file
```

83 tests. They cover the ordinary cases, but the ones worth knowing about came
from things that actually went wrong:

- a leak already under way at power-on still alarms (calibration cannot deadlock)
- calibration always terminates, even if the room never goes quiet
- levels do not flap across a band boundary
- a real captured session is replayed and must never suppress a high-gas event
- a fast smoke burst is never excused as cooking
- `FLAME_SENSOR_PRESENT` is restored after every test that changes it — a leaked
  global was harmless until IR became the only ignition channel, then fatal
- a vision model that never false-alarms *because it detects nothing* is
  rejected, and so is one whose separation margin is too thin to survive a
  warmer evening
- vision (YOLO or MaxViT) can only raise the level and never reaches CRITICAL
  on its own, on CUDA, MPS or CPU
- the video-file camera loops, ends cleanly and reports a missing or broken
  file, all on a synthetic video written by the test itself

---

## Honest limits

- The IR flame sensor sees a narrow cone. A flame outside it is invisible to the
  ignition channel — exactly why that channel can only raise the score.
- MQ2 ppm is not calibrated against a reference gas. Relative mode exists
  precisely because the absolute scale cannot be trusted.
- The YOLO weights are pretrained by someone else on someone else's data. They
  are a component, not a contribution, and they demonstrably false-positive on
  an ordinary room — hence the confidence threshold and the escalator rule.
- The VLM is a small local model. Its scene descriptions are sometimes wrong;
  it is a verification channel, not an oracle.
- Time-to-critical is a short-horizon linear extrapolation, not a forecast.
- Until the learned model is trained and compared, **there is no accuracy figure
  to quote.** The validation so far is the test suite plus replay against real
  captured data. That is the honest answer, and it is a better answer than a
  number with no held-out data behind it.
