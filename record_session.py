"""
FireSense - synchronised data capture.

Records sensor readings and camera frames from one event into a single folder,
with a label. This is the slowest part of the whole project and it needs no new
hardware, so it starts first and runs while everything else is built.

    python record_session.py cooking
    python record_session.py normal --minutes 15
    python record_session.py smoulder --source serial --port COM7
    python record_session.py candle --camera http://192.168.1.42/capture

Stop early with Ctrl+C - the session is still saved.

Output:  data/sessions/20260926_143000_cooking/
             meta.json     what this session is, and how it was captured
             sensors.csv   one row per reading, wall-clock timestamped
             frames/       JPEGs named by milliseconds since session start

Sensors and frames share one clock, so they can be aligned afterwards by
timestamp without any further bookkeeping.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import camera as cam_mod
import config as C
import ingest

# ---------------------------------------------------------------------------
# The label vocabulary. Keep it small and fixed - a dataset where one session
# says "cooking" and another says "cooking_oil" is a dataset you have to clean
# later. Add a new label deliberately, not accidentally.
# ---------------------------------------------------------------------------
LABELS = {
    # nothing happening - the majority of what you should record
    "normal":   "Quiet room, nobody doing anything unusual.",
    # nuisances: these MUST NOT alarm
    "cooking":  "Frying or boiling on a stove. Gas and humidity rise together.",
    "steam":    "Kettle or boiling water only. Humidity with little gas.",
    "incense":  "Incense stick burning. Real smoke, no flame, no heat.",
    "spray":    "Aerosol - deodorant, air freshener. Spikes an MQ2 hard.",
    "dust":     "Dust disturbed near the sensor.",
    # real events: these MUST alarm
    "candle":   "Open flame, little smoke. Should trip the IR flame sensor.",
    "smoulder": "Just-extinguished match or paper. Smoke, no flame, no heat.",
    "fire":     "A real (small, controlled, supervised) fire.",
    "gas":      "Unlit lighter or gas released with no ignition.",
}

NUISANCE = {"normal", "cooking", "steam", "incense", "spray", "dust"}
DANGER = {"candle", "smoulder", "fire", "gas"}

SESSIONS_DIR = Path("data") / "sessions"


def build_sensor_source(kind: str, port: str | None, baud, url: str | None,
                        scenario: str):
    """Whatever the node is talking over today, hand back something with read()."""
    kind = kind.lower()
    if kind == "serial":
        if not port:
            ports = ingest.SerialSource.likely_ports()
            if not ports:
                raise SystemExit(
                    "No serial ports found. Plug the node in, close the Arduino "
                    "Serial Monitor, and check that pyserial is installed.")
            port = ports[0][0]
            print(f"  (no --port given, using {port})")
        src = ingest.SerialSource(port, baud=baud)
        src.start()
        return src, f"serial:{port}@{getattr(src, 'baud_used', baud)}"
    if kind == "http":
        return ingest.HttpPollSource(url or C.ESP32_URL), f"http:{url or C.ESP32_URL}"
    if kind == "mock":
        return ingest.MockSource(scenario), f"mock:{scenario}"
    raise SystemExit(f"unknown --source {kind!r}; use serial, http or mock")


def record(label: str, minutes: float, sensor_kind: str, port: str | None,
           baud, url: str | None, scenario: str, camera_spec: str | None,
           fps: float, environment: str, note: str) -> Path:

    if label not in LABELS:
        raise SystemExit(
            f"unknown label {label!r}.\nUse one of:\n  " +
            "\n  ".join(f"{k:9s} {v}" for k, v in LABELS.items()))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = SESSIONS_DIR / f"{stamp}_{label}"
    frames_dir = folder / "frames"
    folder.mkdir(parents=True, exist_ok=True)

    # -- sensors ----------------------------------------------------------
    sensor, sensor_desc = build_sensor_source(sensor_kind, port, baud, url,
                                              scenario)

    # -- camera (optional - a sensor-only session is still worth having) ---
    cam = None
    cam_desc = "none"
    if camera_spec and camera_spec.lower() not in ("none", "off"):
        cam = cam_mod.make_camera(camera_spec)
        if cam.open():
            cam_desc = f"{cam.name}:{camera_spec}"
            frames_dir.mkdir(exist_ok=True)
        else:
            print(f"  camera unavailable ({getattr(cam, 'error', '?')}) - "
                  f"recording sensors only.")
            cam = None

    print()
    print("=" * 64)
    print(f"  RECORDING  {label}   ({'nuisance' if label in NUISANCE else 'danger'})")
    print(f"  {LABELS[label]}")
    print("=" * 64)
    print(f"  sensors : {sensor_desc}")
    print(f"  camera  : {cam_desc}")
    print(f"  folder  : {folder}")
    print(f"  stop    : Ctrl+C (the session is saved either way)")
    print()

    t0 = time.time()
    deadline = t0 + minutes * 60 if minutes else None
    next_frame_at = t0
    frame_interval = 1.0 / fps if fps > 0 else None

    rows = 0
    frames = 0
    cam_failures = 0
    fields = ["t", "elapsed_s", "gas_ppm", "temp_c", "humidity_pct",
              "pressure_hpa", "flame_detected", "node_id"]

    sensors_csv = folder / "sensors.csv"
    try:
        with sensors_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()

            while deadline is None or time.time() < deadline:
                now = time.time()

                reading, _ = sensor.read()
                if reading:
                    row = dict(reading)
                    row["t"] = round(now, 3)
                    row["elapsed_s"] = round(now - t0, 3)
                    writer.writerow(row)
                    rows += 1
                    if rows % 10 == 0:
                        fh.flush()

                if cam is not None and frame_interval and now >= next_frame_at:
                    f = cam.read()
                    if f.ok:
                        ms = int((now - t0) * 1000)
                        cam_mod.cv2.imwrite(str(frames_dir / f"{ms:08d}.jpg"),
                                            f.image)
                        frames += 1
                    else:
                        cam_failures += 1
                        if cam_failures in (1, 10, 50):
                            print(f"  camera: {f.error}")
                    next_frame_at = now + frame_interval

                elapsed = now - t0
                sys.stdout.write(
                    f"\r  {elapsed:6.1f}s   {rows:5d} readings   "
                    f"{frames:5d} frames   "
                    + (f"gas {reading.get('gas_ppm', 0):6.1f} ppm  "
                       f"{reading.get('temp_c', 0):5.1f} C  "
                       f"{reading.get('humidity_pct', 0):5.1f} %RH"
                       if reading else "waiting for the node...")
                )
                sys.stdout.flush()
                time.sleep(max(0.0, C.SAMPLE_INTERVAL_S - (time.time() - now)))

    except KeyboardInterrupt:
        print("\n  stopped.")
    finally:
        try:
            sensor.stop()
        except Exception:
            pass
        if cam is not None:
            cam.close()

    duration = time.time() - t0
    meta = {
        "label": label,
        "class": "nuisance" if label in NUISANCE else "danger",
        "description": LABELS[label],
        "note": note,
        "environment": environment,
        "started_iso": datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
        "duration_s": round(duration, 1),
        "sensor_source": sensor_desc,
        "camera_source": cam_desc,
        "frame_fps_target": fps if cam is not None else 0,
        "readings": rows,
        "frames": frames,
        "camera_failures": cam_failures,
        "schema": fields,
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))

    print()
    print("-" * 64)
    print(f"  saved {rows} readings and {frames} frames over {duration:.0f}s")
    print(f"  -> {folder}")
    if rows == 0:
        print("  WARNING: no sensor readings. Is the node connected and the")
        print("           Arduino Serial Monitor closed?")
    if cam_failures:
        print(f"  note: {cam_failures} camera reads failed")
    print("-" * 64)
    return folder


def main() -> None:
    p = argparse.ArgumentParser(
        description="Record one labelled FireSense session.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="labels:\n  " + "\n  ".join(f"{k:9s} {v}" for k, v in LABELS.items()))
    p.add_argument("label", help="what is happening in this session")
    p.add_argument("--minutes", type=float, default=10.0,
                   help="how long to record (0 = until Ctrl+C). Default 10.")
    p.add_argument("--source", default="serial",
                   help="serial | http | mock   (default serial)")
    p.add_argument("--port", default=None, help="COM port for --source serial")
    p.add_argument("--baud", default="auto")
    p.add_argument("--url", default=None, help="endpoint for --source http")
    p.add_argument("--scenario", default="normal", help="for --source mock")
    p.add_argument("--camera", default="webcam",
                   help="webcam | webcam:1 | <esp32 ip or url> | none")
    p.add_argument("--fps", type=float, default=1.0,
                   help="frames per second to save (default 1)")
    p.add_argument("--environment", default=C.DEFAULT_ENVIRONMENT)
    p.add_argument("--note", default="", help="anything unusual about this run")
    a = p.parse_args()

    record(a.label, a.minutes, a.source, a.port, a.baud, a.url, a.scenario,
           a.camera, a.fps, a.environment, a.note)


if __name__ == "__main__":
    main()
