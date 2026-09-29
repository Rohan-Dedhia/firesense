# Recording runbook

Follow this literally. One evening gets 3–5 sessions and takes about 90 minutes,
of which the first 30 is just waiting for the sensor to warm up.

---

## Part 1 — Setup (start 30 minutes before you record)

**1. Power the node.** Plug the ESP32 into the laptop's USB. Leave it.

**2. Close the Arduino IDE Serial Monitor.** Completely. Only one program can
hold a COM port, and if the Serial Monitor has it, the recorder gets nothing.
This is the single most common failure.

**3. Set a timer for 25 minutes.** The MQ2 has a heater that needs to reach
thermal equilibrium. Readings before that are drifting and useless. Don't skip
it — you only pay this once per evening, not per session.

While you wait, do steps 4–7.

**4. Build the rig.**

```
   [camera]  50-80 cm back, slightly above, looking down
        \
         \        20-30 cm
          v     node  <------->  where the event will happen
        =========== table or metal tray ===========
```

Node close (it has to physically smell the smoke), camera back far enough to
see both the node and the event.

**5. Tape the positions.** Masking tape on the table where the node sits, where
the laptop sits. You want to reproduce this next time.

**6. Get the code and check the camera.**

```bash
cd firesense
git pull
python camera.py
```

Expect `10/10 frames` and `camera is working`. On a Mac, if it fails, allow
Terminal under System Settings → Privacy & Security → Camera.

**7. Lay out tonight's materials.** Decide your 3–5 sessions now and put
everything within reach — matches, candle, lighter, incense, deodorant, pan.
Fumbling for a lighter mid-session wastes the recording.

**8. When the timer goes, do a 1-minute test:**

```bash
python record_session.py normal --minutes 1
```

Look at the last lines. `readings` must be greater than zero. If it's zero, the
Serial Monitor is still open or the node isn't connected — fix that before
recording anything real. Delete this test folder afterwards.

---

## Part 2 — Each session

Repeat for each of tonight's 3–5 sessions.

**1. Start the recorder.**

```bash
python record_session.py <label> --minutes <N>
```

**2. Do nothing for 30 seconds.** Stand back. Let it record ordinary air first.
This baseline is what the model learns "before" from — a session that starts
mid-event has thrown it away.

**3. Do the event** (see the table below for exactly what).

**4. Let it run to the end**, or press `Ctrl+C` when you're done.

**5. Check the summary line.** `readings` non-zero, `camera_failures` near
zero. If either looks wrong, redo the session — a broken session is worse than
no session, because you'll trust it later.

**6. Wait 3 minutes before the next session.** Open a window. Let the air clear
and the MQ2 settle back down. **If you skip this, the next session starts
contaminated** and its label is a lie.

---

## Part 3 — What to actually do, per label

| label | minutes | exactly what to do |
|---|---|---|
| `normal` | 20–30 | Nothing. Start it and leave the room. Record these while you eat dinner. |
| `cooking` | 15 | Heat oil in a pan on the stove, fry something — onion, bread, egg. Real cooking, not a demonstration of cooking. |
| `steam` | 10 | Boil water in an open pan or an uncapped kettle. No food, no oil. Humidity without much gas. |
| `incense` | 10 | Light an incense stick, stand it 20 cm from the node. Let it burn. |
| `spray` | 5 | Two or three 1-second bursts of deodorant about 30 cm from the node, spaced a minute apart. Not one long spray. |
| `dust` | 5 | Shake a dusty cloth or sweep near the node. |
| `candle` | 10 | Light a tealight 20–30 cm from the node, **in view of the IR flame sensor**. Move it slightly between sessions so the angle varies. |
| `smoulder` | 8 | Light a match, blow it out, hold the smoking head ~10 cm from the node for about 10 seconds. Repeat 3–4 times with a minute between. |
| `gas` | 8 | Hold an **unlit** lighter 15 cm from the node and press the valve for 2–3 seconds. No flame. Repeat 3–4 times with a minute between. |
| `fire` | 5 | **Two people.** A tissue or scrap of paper in a metal bowl, in the sink. Light it, let it burn out. Water and a lid within arm's reach. See the safety section in DATA_COLLECTION.md before the first one. |

---

## Part 4 — End of the evening

**1. Back tonight's folders up.** Everything new under `data/sessions/`. The
folder names already carry the date and label. Copy them somewhere off this
laptop — Drive, an external disk, anywhere. These recordings are the only part
of the project that cannot be rebuilt from scratch.

**2. Note what you recorded** — labels and anything odd, in
`data/sessions/LOG.md`. "3 cooking, 1 smoulder, kitchen light was off for the
second one." You will not remember this in three weeks, and `--note` on the
command line only captures what you thought to say at the time.

```bash
python run_tests.py          # 10 seconds, catches a broken working copy
```

---

## Vary things deliberately

Across the three weeks, change:

- **the room** — kitchen, bedroom, lab
- **the time of day** — daylight versus evening
- **the lighting** — bright white, dim, and especially **warm yellow lamps**
- **what's in frame** — sometimes put a wooden board or an orange notebook in
  view on purpose. These are the hard negatives; the model needs to see them.
- **the distance and angle** of the event from the node

Twelve sessions in the same kitchen under the same bulb is one session recorded
twelve times. The variety is what makes the dataset worth having.

---

## If something goes wrong

| symptom | cause |
|---|---|
| `readings` is 0 | Arduino Serial Monitor is open, or the node is unplugged |
| "No serial ports found" | Node unplugged, or pyserial not installed |
| Camera fails on Mac | Terminal lacks camera permission in System Settings |
| Gas readings look mad | Sensor not warmed up, or the previous session's smoke hasn't cleared |
| Node keeps resetting | Power — use a proper 5V 2A supply, not a thin cable |

When in doubt, redo the session. They cost ten minutes; bad data costs weeks.
