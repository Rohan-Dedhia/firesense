# Data collection — start this week

This is the slowest part of the project and it needs **no new hardware**. Every
day not recording is a day lost. The ESP32-CAM, the models, the fusion head —
all of that can be built later, but none of it is worth anything without data.

The point of this dataset is the thing nobody has published: **nuisance events
recorded with synchronised gas, temperature, humidity and video.** Fire
datasets are all photographs of obvious fires. Yours will contain the hard
cases — the ones that look like fire and aren't.

---

## Recording one session

```bash
python record_session.py cooking
```

That records 10 minutes of sensors plus one frame per second from your laptop
webcam, into `data/sessions/<timestamp>_cooking/`.

Useful variations:

```bash
python record_session.py normal --minutes 30          # long, unattended
python record_session.py candle --minutes 5           # short, supervised
python record_session.py smoulder --port COM7         # name the port explicitly
python record_session.py steam --camera none          # sensors only
python record_session.py fire --camera http://192.168.1.42/capture
```

Stop early with **Ctrl+C** — the session still saves.

Before you start: plug the node in, **close the Arduino Serial Monitor**, and
let the MQ2 warm up for 20–30 minutes. A cold sensor drifts and the readings
are not representative.

---

## The labels

Keep to these exactly. A dataset where one session says `cooking` and another
says `cooking_oil` is a dataset you have to clean later.

### Nuisances — these must NOT alarm

| label | what to do |
|---|---|
| `normal` | Nothing. Leave it running in a quiet room. |
| `cooking` | Fry or boil something on a stove. Gas and humidity rise together. |
| `steam` | Kettle or boiling water only — humidity without much gas. |
| `incense` | Incense stick. Real smoke, no flame, no heat. |
| `spray` | Deodorant or air freshener near the sensor. Spikes an MQ2 hard. |
| `dust` | Disturb dust near the sensor — shake a cloth, sweep. |

### Real events — these MUST alarm

| label | what to do |
|---|---|
| `gas` | Unlit lighter, release butane near the sensor. No flame. |
| `candle` | Open flame in view. Little smoke, should trip the IR sensor. |
| `smoulder` | Light a match, blow it out, hold the smoking head near the sensor. |
| `fire` | A small, controlled, supervised fire. See safety below. |

---

## How much

Target roughly this. It's about 14 hours of recording spread over three weeks,
and `normal` mostly runs itself.

| label | sessions | notes |
|---|---|---|
| `normal` | 20 | Cheapest data you will ever collect. Run it overnight, in different rooms, at different times of day. |
| `cooking` | 10 | Vary it — frying, boiling, toasting. Different distances from the stove. |
| `candle` | 10 | Vary distance and angle to the IR sensor. |
| `smoulder` | 10 | Your hardest case. Worth the effort. |
| `gas` | 8 | |
| `steam` | 6 | |
| `incense` | 6 | |
| `spray` | 6 | |
| `fire` | 6 | Short sessions, supervised. |
| `dust` | 4 | |

**Record `normal` more than you think you need.** A model that has seen a lot
of ordinary air is a model that doesn't panic at ordinary air, and it's the one
class you can collect while doing something else.

---

## Setting up the camera

### The rig

Build one standard arrangement and reuse it. Consistency within a session type
is what lets the model learn the event rather than learning that you moved the
furniture.

```
   [camera]  ~50-80 cm back, slightly above, looking down at the scene
        \
         \        ~20-30 cm
          v     node <--------> event (candle / pan / incense)
        ============ table, tile, or metal tray ============
```

**The node goes close — 20–30 cm.** The MQ2 works by diffusion; smoke has to
physically reach it. Too far and the chemical channel sees nothing while the
camera sees everything, which teaches the model the wrong thing.

**The camera goes back — 50–80 cm**, framing *both* the node and the event
source. You want the video and the sensor readings to obviously describe the
same scene.

**Don't go further than about a metre.** A candle flame at 1 m in a 640×480
frame is only a few pixels tall — not enough for any model to work with.

### Mounting

- **Laptop webcam:** put the laptop on a stack of books so the camera looks
  slightly down at the scene. **Mark the laptop's position with masking tape**
  so you can put it back in the same place next session.
- **A phone as a webcam** (via Iriun, DroidCam, or Windows' built-in phone link)
  is often easier than moving the laptop around, and the image is better.
- **ESP32-CAM later:** blu-tack or tape it to a small box, or clip it to a
  phone tripod. It only needs to be steady and repeatable.

Whatever you use, **the camera must not move during a session.**

### Lighting and background — do this deliberately

This is where you make the dataset genuinely useful rather than just large.

- **Vary the lighting across sessions.** Bright ceiling light, dim room,
  daylight, and — importantly — **warm yellow lamp light**. Warm light is the
  classic false positive for fire detection, and a model that has never seen it
  will fail on it.
- **Deliberately include warm-coloured objects in some sessions.** Wooden desks,
  orange notebooks, skin tone in frame. These are your hard negatives. The HSV
  tuner failed on exactly this, so make sure the learned model gets to see it.
- **Never point the camera at a window.** Backlight and moving sunlight will
  dominate everything.
- **Take one photo of each rig arrangement** and keep it — you will want it for
  the report, and for rebuilding the setup in six weeks.

### Per-session checklist

1. Node warmed up 20–30 minutes, Arduino Serial Monitor closed
2. Camera in position, not going to be nudged
3. `python camera.py` once to confirm frames are arriving
4. Start the recorder **before** you light or start anything
5. Let it run 20–30 seconds of nothing first, so the session contains the
   transition into the event
6. Do the thing
7. Ctrl+C, then check `meta.json` — `readings` should be non-zero and
   `camera_failures` should be near zero

---

## Making the data good, not just plentiful

**Vary things deliberately.** Same event, different distance, different angle,
different room, different time of day, window open and closed. A dataset of
twenty identical cooking sessions teaches the model one cooking session.

**Record the boring beginning.** Start recording *before* you light anything,
so each session contains the transition from normal into the event. The
transition is what the model has to learn to spot — a session that begins
mid-event has thrown that away.

**Use `--note` when something is odd.** `--note "window open, fan on"` takes
five seconds now and saves you guessing at a strange session in six weeks.

**Check the session afterwards.** Open `meta.json`. If `readings` is 0 the node
wasn't connected. If `camera_failures` is high, something was wrong with the
camera. Better to find out now than when you're training.

---

## Safety — read before the `fire` sessions

This is the only part of the project that can hurt you.

- **Never record a `fire` session alone.** Two people, always.
- Keep it **small** — a tealight, a scrap of paper in a metal bowl, a few
  centimetres of flame. You are not trying to make an impressive fire.
- Do it **in a sink, on a tile, or in a metal tray**. Never on a desk, never
  near paper, curtains, or your laptop.
- **Water and a lid within arm's reach** before you light anything.
- Keep the **node and laptop well back**. You need the sensors to smell it, not
  to melt.
- Do it **in a ventilated room**, and stop if anyone starts coughing.
- If your college has rules about open flames in the lab, follow them. A
  `smoulder` session gives you most of the value with none of the risk, so lean
  on those and keep `fire` sessions few and short.

Your supervisor being able to say "they did this carefully" is worth more than
a couple of extra sessions.

---

## What you end up with

```
data/sessions/20260926_143000_cooking/
    meta.json      label, class, duration, source, counts, your note
    sensors.csv    one row per reading, wall-clock and elapsed timestamps
    frames/        00000000.jpg, 00001000.jpg, ...  (milliseconds since start)
```

Sensors and frames share one clock, so they line up by timestamp with no
further work. That is the whole point of recording them together rather than
separately.
