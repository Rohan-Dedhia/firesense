"""
FireSense - live multi-sensor fusion dashboard.

Run locally (full hardware: USB serial + webcam):
    streamlit run app.py

Hosted on Streamlit Community Cloud the server has no COM port and no camera,
so those two channels detect their own absence and step aside; Test Mode and
Replay still work, which is what a shared link is good for. The live hardware
demo must be run locally.

UI notes
--------
The live panel is a st.fragment with run_every, NOT a sleep + st.rerun loop.
That is the difference between a dashboard whose sidebar keeps working while
data streams, and one that ignores every click for two seconds at a time.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st

import alerts
import config as C
import fusion
import ingest
from mock_source import SCENARIOS

# --- optional channel: the camera needs OpenCV, which a headless host may lack
try:
    import vision
    VISION_IMPORTED, VISION_ERROR = True, ""
except Exception as _exc:                                   # pragma: no cover
    vision, VISION_IMPORTED, VISION_ERROR = None, False, str(_exc)

_HAS_FRAGMENT = hasattr(st, "fragment")

st.set_page_config(
    page_title="FireSense",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# credentials: Streamlit Secrets -> environment -> config.py literals
# alerts.py already reads the environment first, so bridging here is enough.
# ---------------------------------------------------------------------------
def _bridge_secrets() -> None:
    try:
        for key in ("TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID"):
            val = st.secrets.get(key, "")
            if val:
                os.environ[key] = str(val)
    except Exception:
        pass          # no secrets.toml locally is normal, not an error


_bridge_secrets()

TONE = {"NORMAL": "#16a34a", "WARNING": "#d97706", "CRITICAL": "#dc2626"}


def _tone_for(score: float) -> str:
    return TONE["NORMAL"] if score < 40 else (
        TONE["WARNING"] if score < 70 else TONE["CRITICAL"])


# ---------------------------------------------------------------------------
# styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px;}
      div[data-testid="stMetricValue"] {font-size: 1.55rem;}
      div[data-testid="stMetricLabel"] {opacity: .72; font-size: .78rem;
          text-transform: uppercase; letter-spacing: .05em;}

      .fs-banner {border-radius: 14px; padding: 20px 24px; margin-bottom: 4px;
          border: 1px solid rgba(128,128,128,.22);
          background: linear-gradient(100deg, var(--fs-wash), transparent 62%);
          border-left: 8px solid var(--fs-tone);}
      .fs-level {font-size: 2.5rem; font-weight: 750; line-height: 1.05;
          letter-spacing: -.02em; color: var(--fs-tone);}
      .fs-score {font-size: 1.35rem; font-weight: 600; opacity: .6;
          margin-left: .5rem; letter-spacing: 0;}
      .fs-why {margin-top: 8px; font-size: .95rem; opacity: .88; max-width: 78ch;}
      .fs-sub {margin-top: 10px; font-size: .8rem; opacity: .6;
          text-transform: uppercase; letter-spacing: .06em;}
      .fs-crit {animation: fs-pulse 1.6s ease-in-out infinite;}
      @keyframes fs-pulse {
        0%,100% {box-shadow: 0 0 0 0 rgba(220,38,38,.30);}
        50%     {box-shadow: 0 0 0 12px rgba(220,38,38,0);}
      }

      .fs-ch {margin-bottom: 15px;}
      .fs-ch-top {display:flex; justify-content:space-between; align-items:baseline;
          gap:10px; margin-bottom:5px;}
      .fs-ch-name {font-weight:600; font-size:.9rem;}
      .fs-ch-read {font-size:.82rem; opacity:.7; text-align:right;}
      .fs-ch-track {height:7px; border-radius:99px; overflow:hidden;
          background: rgba(128,128,128,.18);}
      .fs-ch-fill {height:100%; border-radius:99px; transition: width .35s ease;}
      .fs-ch-foot {font-size:.72rem; opacity:.55; margin-top:4px;}
      .fs-off .fs-ch-name, .fs-off .fs-ch-read {opacity:.42;}

      .fs-card {border:1px solid rgba(128,128,128,.22); border-radius:12px;
          padding:14px 16px; margin-bottom:12px;}
      .fs-card h4 {margin:0 0 8px 0; font-size:.78rem; opacity:.6;
          text-transform:uppercase; letter-spacing:.06em; font-weight:600;}
      .fs-kv {display:flex; justify-content:space-between; font-size:.9rem;
          padding:3px 0;}
      .fs-kv b {font-weight:650;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# session state
# ---------------------------------------------------------------------------
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("engine", fusion.FusionEngine(C.DEFAULT_ENVIRONMENT))
    ss.setdefault("source", ingest.MockSource("normal"))
    ss.setdefault("source_kind", "mock")
    ss.setdefault("scenario", "normal")
    ss.setdefault("alerts", alerts.AlertManager())
    ss.setdefault("vision", None)
    ss.setdefault("show_cam", True)
    ss.setdefault("running", True)
    ss.setdefault("logging", False)
    ss.setdefault("rows", [])
    ss.setdefault("last_result", None)
    ss.setdefault("cadence", "slow")
    ss.setdefault("notice", None)


init_state()
ss = st.session_state


def swap_source(source, kind: str, environment: str, keep_calibration=False) -> None:
    """Replace the data source and start the engine clean, so a switch mid-demo
    never leaves one room's baseline judging another room's air."""
    try:
        ss.source.stop()
    except Exception:
        pass
    ss.source = source
    ss.source_kind = kind
    ss.rows = []
    ss.last_result = None
    ss.engine = fusion.FusionEngine(environment)
    if keep_calibration:
        ss.engine.calibrator.done = True
    ss.alerts = alerts.AlertManager()


# ---------------------------------------------------------------------------
# sidebar - every control lives here, outside the auto-refreshing fragment,
# so streaming data can never steal a click.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔥 FireSense")
    st.caption("Multi-sensor fusion · gas · thermal · ignition")

    eng: fusion.FusionEngine = ss.engine

    env = st.selectbox(
        "Deployment environment",
        list(C.ENVIRONMENT_PROFILES),
        index=list(C.ENVIRONMENT_PROFILES).index(eng.environment),
        help="Same sensors, a different notion of normal. Switching restarts "
             "auto-calibration for the new room.",
    )
    if env != eng.environment:
        eng.set_environment(env, recalibrate=True)
        eng.reset_latency_stopwatch()
        ss.rows = []
    st.caption(C.ENVIRONMENT_PROFILES[env]["notes"])

    st.divider()

    KINDS = ["Serial (USB)", "Test Mode", "HTTP endpoint", "Replay a log", "MQTT"]
    KEY_OF = {"Serial (USB)": "serial", "Test Mode": "mock", "HTTP endpoint": "http",
              "Replay a log": "replay", "MQTT": "mqtt"}
    _default = {"serial": 0, "mock": 1, "http": 2, "replay": 3, "mqtt": 4}.get(
        C.DEFAULT_SOURCE, 1)
    kind = st.radio("Data source", KINDS,
                    index=ss.get("source_ix", _default), key="source_radio")
    ss["source_ix"] = KINDS.index(kind)
    picked = KEY_OF[kind]

    # -- serial ------------------------------------------------------------
    if picked == "serial":
        ports = ingest.SerialSource.list_ports()
        if not ports:
            st.warning(
                "No USB serial ports visible.\n\n"
                "• Locally: plug the node in, close the Arduino Serial Monitor, "
                "and make sure `pyserial` is installed.\n"
                "• Hosted: a cloud server cannot see your ESP32 — run the app on "
                "the laptop the node is plugged into.")
        else:
            labels = [f"{dev} — {desc}" for dev, desc in ports]
            ix = st.selectbox("Port", range(len(ports)),
                              format_func=lambda i: labels[i])
            baud = st.selectbox("Baud rate", ["auto", 115200, 9600, 57600, 38400])
            low = st.checkbox(
                "Flame sensor is active-LOW (0 = flame)", value=True,
                help="Most IR flame modules pull the line LOW on a flame. "
                     "If the dashboard reports flame when there is none, untick this.")
            st.caption("Close the Arduino Serial Monitor first — only one "
                       "program may hold a COM port.")
            if st.button("Connect to node", use_container_width=True,
                         type="primary"):
                try:
                    src = ingest.SerialSource(ports[ix][0], baud=baud,
                                              flame_active_low=low)
                    src.start()
                    swap_source(src, "serial", env)
                    ss.notice = ("success",
                                 f"Connected on {ports[ix][0]} at {src.baud_used} baud.")
                except Exception as exc:
                    ss.notice = ("error", f"Could not open the port: {exc}")
                st.rerun()

    # -- test mode ---------------------------------------------------------
    elif picked == "mock":
        if ss.source_kind != "mock":
            swap_source(ingest.MockSource(ss.scenario), "mock", env,
                        keep_calibration=True)
            st.rerun()
        st.caption("Synthetic scenarios — for checking the logic, clearly "
                   "labelled as such on screen.")
        for name in SCENARIOS:
            if st.button(name.replace("_", " ").title(), use_container_width=True,
                         type="primary" if ss.scenario == name else "secondary"):
                ss.scenario = name
                swap_source(ingest.MockSource(name), "mock", env,
                            keep_calibration=True)
                st.rerun()
        st.caption(SCENARIOS[ss.scenario])

    # -- http --------------------------------------------------------------
    elif picked == "http":
        url = st.text_input("JSON endpoint", C.ESP32_URL,
                            help="ESP32 web server, Firebase REST URL, or Apps Script.")
        if st.button("Connect", use_container_width=True, type="primary"):
            swap_source(ingest.HttpPollSource(url), "http", env)
            st.rerun()
        if ss.source_kind == "http":
            fails = getattr(ss.source, "consecutive_failures", 0)
            if 0 < fails < 4:
                st.caption(f"reconnecting to the node ({fails})…")
            elif fails >= 4:
                st.error(f"No data after {fails} tries. "
                         f"{getattr(ss.source, '_last_error', '')}\n\n"
                         "Is the node powered and on the same network as this "
                         "machine? Try opening the URL above in a browser tab.")

    # -- replay ------------------------------------------------------------
    elif picked == "replay":
        st.caption("Real readings captured earlier — Serial Monitor text, or a "
                   "JSON/JSONL log.")
        up = st.file_uploader("Log file", type=["txt", "log", "json", "jsonl"])
        if up is not None and st.button("Load log", use_container_width=True,
                                        type="primary"):
            try:
                folder = Path("logs")
                folder.mkdir(exist_ok=True)
                f = folder / "uploaded_log.txt"
                f.write_bytes(up.getvalue())
                src = ingest.ReplaySource(str(f), loop=True)
                swap_source(src, "replay", env)
                ss.notice = ("success",
                             f"Loaded {len(src.rows)} real readings "
                             f"({src.bad_lines} unparsable lines skipped).")
            except Exception as exc:
                ss.notice = ("error", str(exc))
            st.rerun()

    # -- mqtt --------------------------------------------------------------
    else:
        host = st.text_input("Broker", C.MQTT_HOST)
        topic = st.text_input("Topic", C.MQTT_TOPIC)
        if st.button("Subscribe", use_container_width=True, type="primary"):
            try:
                src = ingest.MqttSource(host=host, topic=topic)
                src.start()
                swap_source(src, "mqtt", env)
                ss.notice = ("success", f"Subscribed to {topic}.")
            except Exception as exc:
                ss.notice = ("error", str(exc))
            st.rerun()

    st.divider()

    # -- optical channel ---------------------------------------------------
    if VISION_IMPORTED:
        cam_on = st.toggle("Optical channel (webcam)", value=ss.vision is not None)
        if cam_on and ss.vision is None:
            v = vision.VisionChannel()
            ss.vision = v if v.start() else None
            if ss.vision is None:
                st.error("Camera unavailable — check the index in config.py, "
                         "permissions, or whether another app holds it. "
                         "A hosted server has no camera at all.")
        elif not cam_on and ss.vision is not None:
            ss.vision.stop()
            ss.vision = None
        ss.show_cam = st.checkbox("Show camera preview", value=ss.show_cam,
                                  disabled=ss.vision is None)
    else:
        st.caption(f"Optical channel unavailable (OpenCV not importable). "
                   f"Chemical + thermal + IR still run.")

    st.divider()

    ss.running = st.toggle("Live", value=ss.running,
                           help="Pause to freeze the screen while you explain it.")
    ss.logging = st.toggle("Log to CSV", value=ss.logging)

    c_a, c_b = st.columns(2)
    if c_a.button("Recalibrate", use_container_width=True):
        ss.engine.calibrator.reset(env)
        ss.notice = ("info", "Re-learning this room's baseline.")
        st.rerun()
    if c_b.button("Reset timer", use_container_width=True):
        ss.engine.reset_latency_stopwatch()
        st.rerun()

    if st.button("Send test Telegram alert", use_container_width=True):
        if ss.last_result is None:
            ss.notice = ("error", "No reading yet — wait for the first sample.")
        else:
            ok, msg = ss.alerts.maybe_alert(ss.last_result, force=True)
            ss.notice = ("success" if ok else "error",
                         "Telegram alert sent." if ok else f"Telegram: {msg}")
        st.rerun()

    tok = os.environ.get("TELEGRAM_TOKEN", C.TELEGRAM_TOKEN)
    chat = os.environ.get("TELEGRAM_CHAT_ID", C.TELEGRAM_CHAT_ID)
    st.caption("Telegram: configured ✓" if tok and chat else
               "Telegram: not configured — set TELEGRAM_TOKEN / TELEGRAM_CHAT_ID "
               "in Streamlit Secrets or config.py.")


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------
def channel_bar(name: str, reading: str, score: float, weight: str,
                available: bool = True) -> str:
    pct = max(0.0, min(100.0, float(score)))
    tone = _tone_for(pct) if available else "rgba(128,128,128,.35)"
    off = "" if available else " fs-off"
    return (
        f'<div class="fs-ch{off}">'
        f'  <div class="fs-ch-top"><span class="fs-ch-name">{name}</span>'
        f'  <span class="fs-ch-read">{reading}</span></div>'
        f'  <div class="fs-ch-track"><div class="fs-ch-fill" '
        f'       style="width:{pct:.0f}%;background:{tone}"></div></div>'
        f'  <div class="fs-ch-foot">sub-score {pct:.0f}/100 · {weight}</div>'
        f'</div>'
    )


def read_once():
    """One sample: source -> optional camera -> fusion -> alerts -> history."""
    reading, flame_hint = ss.source.read()

    vis_res = vision.VisualResult(available=False) if VISION_IMPORTED else None
    frame = None
    if ss.vision is not None:
        vis_res, frame = ss.vision.poll(want_frame=ss.show_cam)

    problem = None
    if reading:
        problems = ingest.validate(reading)
        if problems:
            problem = "Contract violation from the node: " + "; ".join(problems)
        else:
            if ss.vision is not None and vis_res is not None and vis_res.available:
                v_fire, v_conf, v_avail = vis_res.fire_detected, vis_res.confidence, True
            elif flame_hint is not None:
                # Test Mode with no camera: the scenario supplies optical truth.
                v_fire = bool(flame_hint)
                v_conf, v_avail = (88.0 if flame_hint else 0.0), True
            else:
                v_fire, v_conf, v_avail = False, 0.0, False

            res = ss.engine.update(reading, visual_fire=v_fire,
                                   visual_confidence=v_conf,
                                   visual_available=v_avail)
            ss.last_result = res
            ss.rows.append({
                "time": pd.Timestamp.fromtimestamp(res.ts),
                "Risk score": res.score,
                "Gas compensated (ppm)": res.gas_compensated,
                "Gas raw (ppm)": res.gas_raw,
                "Temperature (°C)": res.temp_c,
                "Humidity (%RH)": res.humidity_pct,
            })
            ss.rows = ss.rows[-400:]
            ss.alerts.maybe_alert(res)
            if ss.logging:
                try:
                    alerts.append_log(res)
                except Exception as exc:
                    problem = f"Could not write the CSV log: {exc}"

    return ss.last_result, vis_res, frame, problem


def render(res, vis_res, frame, problem) -> None:
    if problem:
        st.error(problem)

    if res is None:
        st.info("Waiting for the first reading from the node…")
        if ss.source_kind == "serial":
            st.caption("If nothing arrives: the Arduino Serial Monitor may still "
                       "be holding the port, or the baud rate is wrong.")
        return

    if ss.source_kind == "mock":
        st.warning(f"TEST MODE — synthetic scenario **{ss.scenario}**, "
                   f"not live sensor data.")

    tone = TONE[res.level]
    wash = {"NORMAL": "rgba(22,163,74,.10)", "WARNING": "rgba(217,119,6,.12)",
            "CRITICAL": "rgba(220,38,38,.14)"}[res.level]
    pulse = " fs-crit" if res.level == "CRITICAL" else ""
    src_label = {"serial": "USB serial", "mock": "test mode", "http": "HTTP",
                 "replay": "replay", "mqtt": "MQTT"}[ss.source_kind]

    st.markdown(
        f'<div class="fs-banner{pulse}" style="--fs-tone:{tone};--fs-wash:{wash}">'
        f'  <div class="fs-level">{res.level}'
        f'    <span class="fs-score">{res.score:.0f}/100</span></div>'
        f'  <div class="fs-why">{res.explanation}</div>'
        f'  <div class="fs-sub">{res.environment} · {src_label} · '
        f'   pattern: {res.signature.replace("_", " ")}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if res.calibrating:
        st.progress(res.calibration_progress,
                    text=f"Learning this room's baseline — "
                         f"{res.calibration_progress * 100:.0f} %")
    else:
        st.caption(f"Baseline: {res.calibration_note}")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Gas", f"{res.gas_compensated:.0f} ppm",
              f"{res.gas_compensated - res.gas_raw:+.0f} vs raw",
              delta_color="off")
    k2.metric("Temperature", f"{res.temp_c:.1f} °C",
              f"{res.slopes.temp:+.2f} °C/s" if res.slopes.mature else None,
              delta_color="off")
    k3.metric("Humidity", f"{res.humidity_pct:.0f} %",
              f"{res.slopes.humidity:+.2f} %/s" if res.slopes.mature else None,
              delta_color="off")
    k4.metric("Confidence", f"{res.confidence:.0f} %",
              "channels agree" if res.confidence >= 65 else "channels disagree",
              delta_color="off")
    k5.metric("Time to CRITICAL",
              f"{res.seconds_to_critical:.0f} s" if res.seconds_to_critical else "—",
              "trend extrapolation" if res.seconds_to_critical else "no upward trend",
              delta_color="off")

    st.write("")
    left, right = st.columns([3, 2], gap="large")

    # ---- left: the trend story ------------------------------------------
    with left:
        if ss.rows:
            df = pd.DataFrame(ss.rows).set_index("time")
            st.markdown("**Risk score**")
            st.line_chart(df[["Risk score"]], height=190, color=tone)
            st.markdown("**Gas — raw vs humidity/temperature compensated**")
            st.line_chart(df[["Gas raw (ppm)", "Gas compensated (ppm)"]],
                          height=165, color=["#94a3b8", "#2563eb"])
            st.markdown("**Thermal channel**")
            st.line_chart(df[["Temperature (°C)", "Humidity (%RH)"]],
                          height=165, color=["#ea580c", "#0891b2"])
        else:
            st.caption("Collecting history…")

    # ---- right: the reasoning -------------------------------------------
    with right:
        st.markdown("#### Why this score")

        flame_read = ("TRIGGERED" if res.flame_sensor else "clear") \
            if res.flame_available else "not fitted"
        optical_read = ("flame colour" if res.visual_fire else "no flame colour") \
            if (ss.vision is not None or ss.source_kind == "mock") else "channel off"

        st.markdown(
            channel_bar("Chemical · MQ2", f"{res.gas_compensated:.0f} ppm",
                        res.gas_score, f"weight {C.WEIGHTS['gas']:.2f}")
            + channel_bar("Thermal · BME280", f"{res.temp_c:.1f} °C",
                          res.temp_score, f"weight {C.WEIGHTS['temp']:.2f}")
            + channel_bar("Optical · camera", optical_read, res.visual_score,
                          "ignition evidence",
                          available=(ss.vision is not None or ss.source_kind == "mock"))
            + channel_bar("IR flame sensor", flame_read,
                          100.0 if res.flame_sensor else 0.0, "ignition evidence",
                          available=res.flame_available)
            + channel_bar("Ignition · fused",
                          "confirmed" if res.ignition_score else "none",
                          res.ignition_score, f"weight {C.WEIGHTS['visual']:.2f}"),
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="fs-card"><h4>Pattern</h4>'
            f'<div style="font-size:.92rem">{res.signature_reason}</div>'
            f'<div class="fs-kv" style="margin-top:8px"><span>gas trend</span>'
            f'<b>{res.slopes.gas:+.2f} ppm/s</b></div>'
            f'<div class="fs-kv"><span>temperature trend</span>'
            f'<b>{res.slopes.temp:+.3f} °C/s</b></div>'
            f'<div class="fs-kv"><span>humidity trend</span>'
            f'<b>{res.slopes.humidity:+.3f} %RH/s</b></div></div>',
            unsafe_allow_html=True,
        )

        if res.suppressed:
            st.success("False alarm suppressed by fusion — a fixed-threshold "
                       "detector would be alarming right now.")
        for flag in res.flags:
            st.warning(flag)

        st.markdown("#### Fusion vs fixed threshold")
        naive = "ALARMING" if res.naive_alarm else "quiet"
        st.markdown(
            f'<div class="fs-card">'
            f'<div class="fs-kv"><span>Conventional detector '
            f'(raw MQ2 ≥ {C.NAIVE_GAS_THRESHOLD_PPM:.0f} ppm)</span>'
            f'<b>{naive}</b></div>'
            f'<div class="fs-kv"><span>FireSense fusion</span>'
            f'<b style="color:{tone}">{res.level}</b></div></div>',
            unsafe_allow_html=True,
        )
        gain = ss.engine.latency_gain_s
        if gain is not None:
            st.metric("Fusion alarmed earlier by", f"{gain:.1f} s")
        elif getattr(ss.engine, "_fusion_trigger_t", None) and not res.naive_alarm:
            st.info("Fusion has alarmed; the fixed threshold still has not.")
        elif res.naive_alarm and res.level == "NORMAL":
            st.success("The fixed threshold is false-alarming; fusion is "
                       "correctly quiet.")

        if ss.show_cam and frame is not None:
            st.markdown("#### Optical channel")
            st.image(frame[:, :, ::-1], channels="RGB",
                     use_container_width=True)
            if vis_res is not None:
                st.caption(vis_res.note)

        st.markdown("#### Event log")
        if ss.engine.events:
            ev = pd.DataFrame(ss.engine.events)
            ev["time"] = pd.to_datetime(ev["t"], unit="s").dt.strftime("%H:%M:%S")
            st.dataframe(ev[["time", "from", "to", "score", "signature"]],
                         hide_index=True, use_container_width=True, height=190)
        else:
            st.caption("No state changes yet.")

        if ss.rows:
            st.download_button(
                "Download session CSV",
                pd.DataFrame(ss.rows).to_csv(index=False).encode(),
                file_name="firesense_session.csv", mime="text/csv",
                use_container_width=True,
            )


def refresh_interval() -> float:
    """Sample faster once something is developing; never hammer a real ESP32."""
    level = ss.last_result.level if ss.last_result else "NORMAL"
    interval = C.FAST_SAMPLE_INTERVAL_S if level != "NORMAL" else C.SAMPLE_INTERVAL_S
    if ss.source_kind == "http":
        interval = max(interval, C.HTTP_MIN_INTERVAL_S)
    return float(interval)


# ---------------------------------------------------------------------------
# one-shot notices raised by sidebar actions
# ---------------------------------------------------------------------------
if ss.notice:
    kind, text = ss.notice
    {"success": st.success, "error": st.error, "info": st.info}[kind](text)
    ss.notice = None


# ---------------------------------------------------------------------------
# the live panel
# ---------------------------------------------------------------------------
_INTERVAL = refresh_interval()

if _HAS_FRAGMENT:

    @st.fragment(run_every=_INTERVAL if ss.running else None)
    def live_panel() -> None:
        res, vis_res, frame, problem = read_once()
        render(res, vis_res, frame, problem)

        # re-arm the fragment when the sampling regime changes
        want = "fast" if (res and res.level != "NORMAL") else "slow"
        if want != ss.cadence:
            ss.cadence = want
            try:
                st.rerun(scope="app")
            except TypeError:                          # older Streamlit
                st.rerun()

    live_panel()

else:                                                   # Streamlit < 1.37
    res, vis_res, frame, problem = read_once()
    render(res, vis_res, frame, problem)
    if ss.running:
        time.sleep(_INTERVAL)
        st.rerun()

if not ss.running:
    st.caption("Paused — flip **Live** back on in the sidebar to resume streaming.")
