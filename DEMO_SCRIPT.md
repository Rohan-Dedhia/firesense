# Demo script — the exact sequence to run in front of your sir

Total: about 4 minutes. Rehearse it once end-to-end before he walks in.

---

## Before he arrives (30 minutes, non-negotiable)

1. **Power the node and leave it running.** MQ2 needs to warm up; the longer the
   better, minimum 20–30 minutes before the demo.
2. **`python auto_tune_vision.py --write`** in the demo room, camera positioned
   exactly as it will be. It records the scene without the flame, then with it,
   works out the flame's colour signature from the pixels that changed, and
   writes the HSV range and trigger threshold into `config.py` itself. It also
   prints a **separation margin** — if that is under 3x, something warm-coloured
   is in view; move it or re-aim rather than accepting the numbers. A margin
   stuck near 1x means the room itself is the problem (warm lamps, wooden
   surfaces, skin tone in frame), not the settings.
3. **Set the table.** Camera must see the trigger point. Node beside the laptop.
   Decide exactly where you will hold the lighter and don't improvise later.
4. **Start the dashboard, pick the environment, let auto-calibration finish**
   (progress bar, ~2 min). Do not skip this — it is what makes the thresholds fit
   this room.
5. **Open a second browser tab on the dashboard in Mock mode** as your fallback.

---

## Choosing your trigger — read this before you buy anything

| trigger | chemical (MQ2) | thermal (BME280) | optical (camera) | use it for |
|---|---|---|---|---|
| **Lighter, UNLIT** (butane released) | **strong, fast** | none | none | the gas-leak demo — your best moment |
| **Tea-light candle** | weak | slight if close | **best** — steady, orange, hands-free | the optical/ignition demo |
| **Soft-flame lighter, lit** | weak | slight if close | good, but small and you must hold it | optical, if you have no candle |
| **Incense stick** | **good** — MQ2 reads smoke well | none | **poor** — see below | a secondary smoke demo only |
| **Jet/torch lighter** | weak | some | **useless — burns blue** | don't |

**On incense specifically:** it's a genuinely good smoke source for the MQ2 —
that channel will respond, though more slowly and weakly than butane, so hold it
within a few centimetres and waft the smoke toward the sensor. But the ember is
2–3 mm of dull orange glow. At any sane distance from the webcam that is a few
pixels — far below any trigger threshold you could set without the system
false-alarming on every warm-coloured thing in the room. **Do not plan the
optical part of the demo around incense.** Light it with a lighter and the
lighter's flame is what the camera sees, not the incense.

**One trap to avoid:** do not present incense as the "nuisance event our system
suppresses". The suppression logic keys on *humidity rising together with gas* —
the cooking signature. Incense smoke doesn't do that, so the system will treat
it as a leak and alarm. That would look like a false positive of your own
system, in front of him. Use **Mock mode → Cooking False Positive** for the
suppression demo; that's what it's for.

**If you can only get one thing:** a pack of tea-light candles plus the lighter
you already own. The candle covers the camera, the unlit lighter covers the gas.

---

## The demo

**1. "This is the room, quiet."** (15 s)
Point at NORMAL, the confidence figure, and the calibration line — *"it learned
this room's baseline itself two minutes ago, it isn't using numbers we typed in."*

**2. Gas first — lighter UNLIT, release butane near the sensor.** (30 s)
This is the strongest part of the demo and doesn't need the camera at all.
Watch: gas rises, humidity stays flat → pattern flips to **leak** → score climbs
→ WARNING. Point at the **fixed-threshold panel still saying "quiet"** and at the
countdown. Say: *"a conventional detector hasn't noticed yet."*
Then keep going until the fixed threshold trips, and show the **latency number**.

**3. Now the flame — light the candle in view of the camera and the IR sensor.** (20 s)
Both ignition channels confirm → score hard-escalates to CRITICAL → buzzer →
Telegram on your phone. Say: *"four independent physics agreeing — chemical,
thermal, infrared and optical — that's why confidence is high."*
Then, if you want the strongest single moment: **block the camera's view with
your hand while keeping the flame on the IR sensor.** The system stays CRITICAL
on the IR channel and raises "ignition channels disagree — one of them may not
be pointed at the source". That is redundancy working, live, and almost nobody
demonstrates it.

**4. The false alarm — this is the part that wins it.** (45 s)
Switch to Mock mode → **Cooking False Positive**. Show the fixed threshold
alarming while the fusion score stays NORMAL, and read the suppression line out
loud: *gas and humidity rising together is the water-vapour signature of cooking,
not a leak.* This is your actual research claim, made visible.

**5. Environment flip.** (15 s)
Change the dropdown Kitchen → Storage Room. Same reading, different colour.
*"Same hardware, different context."*

**6. If anything misbehaves:** switch to Mock mode, run **Sudden Fire**, and say
plainly *"this is our labelled test mode — commercial detectors have a self-test
too."* That is a normal engineering answer, not an excuse.

---

## Questions he will ask, and the honest answers

**"What's your accuracy?"**
*"There's no trained model to measure accuracy of — the visual detection is
rule-based HSV colour thresholding and the leak-vs-cooking logic is derived from
physics, so neither needs training data. What we measure instead is the
comparison that matters: our system versus a conventional fixed-threshold
detector on the same event, and that number is on screen live."*

**"Is this AI / machine learning?"**
*"No, and deliberately not. It's sensor fusion with rule-based signature
analysis. We can explain every decision the system makes, which a trained model
on a week of data could not. A learned model is our stated future work."*

**"Where's your dataset?"**
*"The baseline is collected in-situ at run time — the node learns this room's
normal in two minutes. Public MQ-sensor datasets are recorded on different
sensor units with different R0 values, so they wouldn't transfer to our
hardware. Cross-validation against them is future work."*

**"Why two flame detectors?"**
*"They fail differently. The IR sensor measures infrared emission and doesn't
care about room lighting; the camera measures colour and can be fooled by warm
light, but it covers a wide area rather than a narrow cone. We fuse them, weight
agreement higher than either alone, and flag disagreement instead of hiding it —
which is also our sensor-fault detector."*

**"Will this work in a factory?"**
*"The thresholds are per-environment and the node auto-calibrates to whatever
room it's in, so yes in principle — but we've only validated it here, and we say
that rather than claiming more."*
