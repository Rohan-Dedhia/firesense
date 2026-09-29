# CLAUDE.md — working rules for this project

Read this before changing anything. It records decisions that were made for
reasons, several of them after something went wrong, and reversing one by
accident costs more than it looks.

## What this project is

FireSense: multi-sensor fire and gas detection whose research claim is
**nuisance-alarm discrimination** — telling cooking from fire. The anchor is
NIST 2024, *"New Smoke Alarms Are Better at Detecting Fires but Still Beep for
Bacon."* Detecting fire is solved; not alarming on bacon is not. Every design
decision serves that claim.

Owner: Riti Shah (B.Tech Data Science). She does all of the work, including all
recording. A teammate supplies the hardware node and nothing else — do not
assign tasks to anyone but her.

## Hard rules

**1. `fusion.py` is frozen.** It is the baseline the learned model must beat. A
baseline you keep improving is not a baseline. Fix an outright bug if you find
one; do not tune it, do not add channels to it, do not make it smarter. New
capability goes in `pipeline.py`, on top of its verdict.

**2. The escalator rule.** Ignition, visual and semantic channels may only
*raise* the score, never lower it. A zero from any of them is absence of
evidence, not evidence of safety. This was earned: a community fire model
reported *fire* on an empty room for twenty consecutive frames at 0.40–0.45
confidence. Never let a channel that can be wrong that way contribute a
confident zero.

**3. Split by session, never by window.** `dataset.py` has exactly one split
function, `split_by_session()`, and the absence of `split_by_window()` is
deliberate. Windows from one recording are hugely correlated; shuffling them
across train and test leaks and every number afterwards is inflated. If you add
cross-validation, group by session. Augmented copies share their original's
group (`augment.augmented_name()`).

**4. Augment the training side only, after the split.** Never the test set.
An augmented copy of a test session is still that session.

**5. Every feature is relative to the session's own baseline.** Raw ppm encodes
one uncalibrated MQ2 in one room. Relative features are why cross-hardware and
cross-room training works at all. Do not add an absolute-ppm feature without
saying in the code comment why it is defensible.

**6. `TIME_WARP_LADDER` is a fixed ladder, not a distribution.** Warping changes
session duration, which changes window count, which silently reweights classes —
measured drift from 0.50 to 0.56 danger fraction. Cycling one ladder makes the
effect cancel exactly across classes. Do not "improve" this by sampling it.

**7. Keep `requirements.txt` thin.** Streamlit Cloud installs it. `opencv`,
`ultralytics`, `torch`, `scikit-learn` go in `requirements-ml.txt` (local only).
There must be **no `packages.txt`** — adding one with `libglib2.0-0` broke the
hosted build on Debian trixie. `app.py` must import nothing that needs a camera.

**8. Secrets come from the environment.** `config.py` reads `TELEGRAM_TOKEN` and
`TELEGRAM_CHAT_ID` via `os.environ.get(...)`, and `app.py` bridges `st.secrets`
into `os.environ`. Never hardcode a token, not even temporarily.

**9. Tests that mutate module-level config must restore it.** `C.FLAME_SENSOR_PRESENT`
was once left `False` by a `finally` block and poisoned every later test. Use the
`_FLAME_DEFAULT` / `_restore_flame()` pattern.

**10. A vision model is never adopted on its README.** It is adopted on a
measured separation margin against frames of the actual deployment scene. A
detector that is quiet because it is blind must fail that check — that is the
failure mode a false-positive-only test misses, and it is tested for.

## Conventions

- Python 3.11+, `from __future__ import annotations`, standard library first.
- No pytest. `run_tests.py` is a plain runner; keep it that way — it has to work
  on a fresh machine with nothing installed. It picks up every `tests/test_*.py`
  automatically, so a new test file needs no change to the runner.
- Streamlit live updates use `@st.fragment(run_every=...)`. Never `time.sleep()`
  plus `st.rerun()` in the main body; that was the source of the UI glitches.
- Comments explain *why*, and cite the measurement when there was one. A number
  in this codebase should be traceable to an observation, not a preference.
- New label vocabulary is added deliberately, in `record_session.LABELS`, never
  ad hoc at recording time.
- Environment: `conda activate firesense`. macOS, Apple Silicon (M5, 24 GB).
  The project lives at `~/Desktop/firesense`.

## Hardware facts

- ESP32 node over USB serial: MQ2 (gas), BME280 (temp/humidity/pressure),
  IR flame sensor. `serial_parse.py` reads the Serial Monitor text format.
- ESP32-CAM, **OV3660** sensor — it has a hardware JPEG encoder, so streaming is
  fine. Serves `/capture` from the stock CameraWebServer sketch.
- macOS always shows `/dev/cu.debug-console` and `/dev/cu.Bluetooth-Incoming-Port`.
  Neither is the node. `SerialSource.likely_ports()` filters them; `python ingest.py`
  is the diagnostic. Close the Arduino Serial Monitor — one program per port.
- MQ2 R0 is uncalibrated and differs between units. This is a premise of the
  design, not a defect to fix.

## What is not built yet

- The learned fusion model is untrained — no data recorded yet. **There is no
  accuracy figure, and inventing one is the worst thing you could do here.**
  Until `train.py` has run on real held-out sessions, the honest answer to "what
  is the accuracy" is the test suite and the replay fixture.
- MaxViT (`timm`) is planned as a vision backbone comparison and is commented out
  in `requirements-ml.txt`. Not started.
- No YOLO weights in `models/` yet. `get_weights.py` fetches candidates;
  `pick_vision_model.py` decides which one and at what threshold, by measurement
  on her own room. Do not set `CONF_THRESHOLD` from a README — the current 0.55
  is itself a measured value and should be replaced by whatever the chooser
  reports.
- Ollama and `moondream` not yet installed locally. `python vlm.py --selftest`
  validates the scoring logic offline first, so a setup failure is never
  confused with a code failure.

## Tone

Riti presents this to a professor who asked for something that "differentiates
from already available products and normal engineering projects." Explanations
should be defensible in a viva: name the mechanism, give the measurement, state
the limit. Do not oversell. A limitation stated plainly reads as competence; a
number with nothing behind it does not survive one follow-up question.
