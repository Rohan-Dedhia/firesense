# Demo script — the exact sequence to run in front of your sir

Total: about 4 minutes. Rehearse it once end-to-end before he walks in.

---

## Before he arrives (30 minutes, non-negotiable)

1. **Power the node and leave it running.** The MQ2 heater needs to reach
   thermal equilibrium; the longer the better, minimum 20–30 minutes. Calibration
   refuses to lock in while readings are still drifting more than 12 %, so a cold
   sensor costs you time you won't have.
2. **Close the Arduino Serial Monitor.** Only one program can hold a COM port.
   This is the single most common reason the dashboard shows nothing.
3. **Set the table.** Node beside the laptop. Decide exactly where you will hold
   the lighter and the flame, and don't improvise later.
4. **Start the dashboard, pick the environment, let auto-calibration finish**
   (progress bar, ~2 min). Do not skip this — it is what makes the thresholds fit
   this room.
5. **Open a second browser tab in Test Mode** as your fallback.

---

## Choosing your trigger — read this before you buy anything

| trigger | chemical (MQ2) | thermal (BME280) | IR flame sensor | use it for |
|---|---|---|---|---|
| **Lighter, UNLIT** (butane released) | **strong, fast** | none | none | the gas-leak demo — your best moment |
| **Soft-flame lighter, lit** | weak | slight if close | **yes** | the ignition demo |
| **Tea-light candle** | weak | slight if close | **yes** — steady and hands-free | the ignition demo, if you'd rather not hold it |
| **Incense stick** | **good** — MQ2 reads smoke well | none | no | a secondary smoke demo only |
| **Jet/torch lighter** | weak | some | yes | works, but the soft flame is easier to aim |

**On incense specifically:** it's a genuinely good smoke source for the MQ2 —
that channel will respond, though more slowly and weakly than butane, so hold it
within a few centimetres and waft the smoke toward the sensor. It will not
trigger the IR flame sensor, and that is correct behaviour: there is no flame.

**One trap to avoid:** do not present incense as the "nuisance event our system
suppresses". The suppression logic keys on *humidity rising together with gas* —
the cooking signature. Incense smoke doesn't do that, so the system will treat
it as a leak and alarm. That would look like a false positive of your own
system, in front of him. Use **Test Mode → Cooking False Positive** for the
suppression demo; that's what it's for.

**If you can only get one thing:** the lighter you already own. Unlit it covers
the gas demo, lit it covers the ignition demo.

---

## The demo

**1. "This is the room, quiet."** (15 s)
Point at NORMAL, the confidence figure, and the calibration line — *"it learned
this room's baseline itself two minutes ago, it isn't using numbers we typed in."*

**2. Point at "Why this score" before anything happens.** (15 s)
Three channels, each with its own sub-score and weight. Establish that every
number is traceable *before* you make it move — it lands better than explaining
it afterwards.

**3. Gas first — lighter UNLIT, release butane near the sensor.** (30 s)
This is the strongest part of the demo. Watch: gas rises, humidity stays flat →
pattern flips to **leak** → score climbs → WARNING. Point at the
**fixed-threshold panel still saying "quiet"** and at the countdown. Say:
*"a conventional detector hasn't noticed yet."*
Then keep going until the fixed threshold trips, and show the **latency number**.

**4. Now the flame — light it in view of the IR sensor and hold it steady.** (20 s)
Hold it for **at least three seconds**. The system requires two consecutive
positive samples before it believes the sensor, so a quick flick is correctly
ignored — that's the glitch filter working, but it isn't what you want him to
see first.
Watch the ignition bar fill and the score hard-escalate to CRITICAL, then show
the Telegram alert on your phone. Say: *"a confirmed flame is a direct physical
measurement, so it sets a floor under the score rather than just adding to it."*

**5. The false alarm — this is the part that wins it.** (45 s)
Switch to Test Mode → **Cooking False Positive**. Show the fixed threshold
alarming while the fusion score stays NORMAL, and read the suppression line out
loud: *gas and humidity rising together is the water-vapour signature of cooking,
not a leak.* This is your actual research claim, made visible.

**6. Environment flip.** (15 s)
Change the dropdown Kitchen → Storage Room. Same reading, different colour.
*"Same hardware, different context."*

**7. If anything misbehaves:** switch to Test Mode, run **Sudden Fire**, and say
plainly *"this is our labelled test mode — commercial detectors have a self-test
too."* That is a normal engineering answer, not an excuse.

---

## Questions he will ask, and the honest answers

**"What's your accuracy?"**
*"There's no trained model to measure the accuracy of. The leak-versus-cooking
logic is derived from physics, so it needs no training data. What we measure
instead is the comparison that matters: our system versus a conventional
fixed-threshold detector on the same event, and that number is on screen live."*

**"Is this AI / machine learning?"**
*"No, and deliberately not. It's sensor fusion with rule-based signature
analysis. We can explain every decision the system makes, which a trained model
on a week of data could not. A learned model is our stated future work."*

**"Where's your dataset?"**
*"The baseline is collected in-situ at run time — the node learns this room's
normal in two minutes. Public MQ-sensor datasets are recorded on different
sensor units with different R0 values, so they wouldn't transfer to our
hardware. Cross-validation against them is future work."*

**"Why does the flame sensor need two samples?"**
*"Because ignition hard-escalates the score. One glitched reading or a
reflection could otherwise throw a CRITICAL on its own, so we require two
consecutive positives and flag the single ones as noise on screen rather than
hiding them."*

**"What if the flame is out of the sensor's view?"**
*"Then the ignition channel stays at zero — and that's why it can only ever
raise the score, never lower it. Not seeing a flame is not evidence of safety,
so the chemical and thermal channels still alarm on their own, and the system
says on screen that the source may be outside the sensor's line of sight."*

**"Will this work in a factory?"**
*"The thresholds are per-environment and the node auto-calibrates to whatever
room it's in, so yes in principle — but we've only validated it here, and we say
that rather than claiming more."*
