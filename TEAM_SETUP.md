# Working on two laptops

The code lives in GitHub. The recordings live in a shared drive. The node lives
on WiFi. Once those three things are true, it stops mattering whose laptop
anything is on.

---

## Setting up your laptop

You already have the repo — `github.com/Rohan-Dedhia/firesense`. Clone it:

```bash
git clone https://github.com/Rohan-Dedhia/firesense.git
cd firesense
python -m pip install -r requirements.txt
python -m pip install -r requirements-ml.txt
python run_tests.py
```

If that prints `34/34 passed`, your laptop is ready. **No hardware needed** —
the fusion engine has no dependency on the node, which is exactly why it was
built that way.

Ask Rohan to add you as a collaborator on the repo (Settings → Collaborators)
so you can push, not just pull.

What runs on your laptop with nothing plugged in:

- the dashboard in Test Mode (`streamlit run app.py`)
- the camera module against your own webcam (`python camera.py`)
- replaying any session either of you recorded
- all the model training and evaluation work

---

## Sharing code

Normal git. Before you start work:

```bash
git pull
```

After you finish something:

```bash
git add .
git status          # check nothing secret or huge crept in
git commit -m "what you did"
git push
```

**The one rule that avoids pain: don't both edit the same file at the same
time.** Agree who owns what. A reasonable split for this project:

| you | Rohan |
|---|---|
| `camera.py`, the vision and model code, the fusion training | `firmware/`, the ESP32 sketches, wiring |
| the dashboard and analysis | recording sessions with the node |

If you do collide, git will tell you and you fix it together — it's recoverable,
just annoying.

---

## Sharing recordings

**Recordings do not go in git.** A session is hundreds of JPEGs; a few dozen
sessions would make the repo enormous and slow for both of you. `.gitignore`
already excludes `data/sessions/`.

Instead: make a shared **Google Drive folder** called `firesense-data`, and
inside it keep the session folders exactly as the recorder produces them:

```
firesense-data/
    20260926_143000_cooking/
    20260926_151200_normal/
    20260927_100500_smoulder/
```

Whoever records a session uploads that folder. Whoever trains downloads what
they need into their local `data/sessions/`. The folder names already carry the
timestamp and label, so nothing gets confused.

Install the Google Drive desktop app on both laptops and it syncs by itself.

---

## Sharing the node — the part that actually changes things

**Right now** the node is plugged into Rohan's laptop over USB, so only he can
record. That's the real constraint, and it's a bigger deal than the code.

**After the WiFi migration** (week 1 of the plan) the node publishes over the
network instead of a cable. Then *either* of you can record from it, as long as
you're on the same WiFi — you point the recorder at its address instead of a
COM port:

```bash
python record_session.py cooking --source http --url http://192.168.1.50/reading
```

So do the WiFi migration early. It isn't just a feature for the demo — it's
what stops one person being the bottleneck for all data collection.

Until then, the honest split is: **Rohan records, you build.** There's plenty
for you to build that doesn't need the node, and the recordings arrive through
the shared drive as he makes them.

---

## Keep one machine as the demo machine

For the actual presentation, decide early which laptop drives the demo and do
the final rehearsals on that one. Two laptops that both *nearly* work is a
worse position than one that definitely does.
