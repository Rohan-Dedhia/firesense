# Hosting FireSense on Streamlit

## The one thing to know first

A hosted app runs on Streamlit's server, not on your laptop. That server has
no USB ports and no camera, so **it cannot see your ESP32 and it cannot see a
webcam**. There is no setting that changes this — it is what "hosted" means.

| | Local (`streamlit run app.py`) | Hosted on Streamlit Cloud |
|---|---|---|
| Serial (USB) — the live node | works | not possible |
| Optical channel (webcam) | works | not possible |
| Test Mode (synthetic scenarios) | works | works |
| Replay an uploaded log | works | works |
| HTTP endpoint | works if the node is reachable | only if the node is on the public internet |
| Telegram alerts | works | works |

So: **run it locally for the live demo in front of sir**, and use the hosted
link as the shareable version — anyone can open the URL, run the scenarios, and
upload a captured log to watch the engine reason over real readings.

The app detects this by itself. Hosted, the Serial option explains that no
ports are visible instead of throwing an error, and the camera toggle
disappears if OpenCV cannot load.

---

## Run it locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Close the Arduino Serial Monitor first — only one program may hold a COM port.

---

## Deploy it

1. Push this folder to a GitHub repository.
   `.gitignore` already excludes `.streamlit/secrets.toml`, `logs/` and
   `__pycache__/`, so your Telegram token does not go with it.
2. Go to <https://share.streamlit.io>, sign in with GitHub, **New app**.
3. Point it at your repo, branch `main`, main file `app.py`. Deploy.
4. Once it is up: **Manage app → Settings → Secrets**, and paste

   ```toml
   TELEGRAM_TOKEN = "your-botfather-token"
   TELEGRAM_CHAT_ID = "your-chat-id"
   ```

   Save. The app restarts and the sidebar will read *Telegram: configured*.

`requirements.txt` deliberately does **not** install OpenCV. A hosted container
has no camera, so the optical channel could never run there anyway, and OpenCV
pulls in system libraries (`libglib2.0-0`, `libffi7`) that no longer resolve on
Streamlit Cloud's current Debian base — the build fails with *"held broken
packages"*. Leaving it out makes the deploy both faster and reliable; the
dashboard notices OpenCV is missing and runs on chemical + thermal + IR.

Locally, `python -m pip install opencv-python` re-enables the camera.

---

## Credentials, in order of precedence

1. **Streamlit Secrets** — hosted, and locally via `.streamlit/secrets.toml`
   (already filled in for you, and git-ignored).
2. Environment variables `TELEGRAM_TOKEN` / `TELEGRAM_CHAT_ID`.
3. The literals in `config.py` — fine for a laptop demo, but never commit a
   real token. A token in a public repo is a public token, and Telegram
   revokes it.

Because the token now lives in `.streamlit/secrets.toml` rather than in
`config.py`, re-extracting the project no longer wipes it.

---

## If the hosted app misbehaves

- **Build fails on `libglib2.0-0` / "held broken packages"** — a `packages.txt`
  has crept back into the repo, or OpenCV is back in `requirements.txt`.
  Neither belongs there; delete and push again.
- **Sidebar says Telegram not configured** — the Secrets were saved but the app
  was not restarted; use *Manage app → Reboot*.
- **App sleeps** — free Community Cloud apps idle out. Opening the URL wakes
  it, but give it ~30 seconds before a demo rather than during one.
