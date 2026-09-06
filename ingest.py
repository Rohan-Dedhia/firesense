"""
FireSense - data source abstraction.

The ONLY place that knows where readings come from. Everything downstream
(fusion, dashboard, alerts) consumes the contract dict and nothing else, so
Day 6 integration is a dropdown change, not a rewrite.

Sources:
  MockSource      - scenario generator (Test Mode) .............. works today
  HttpPollSource  - GET a JSON endpoint (ESP32 web server, or a Firebase /
                    Apps Script REST URL) ....................... Day 6 swap
  MqttSource      - subscribe to a topic (paho-mqtt, optional) .. Day 6 swap
"""
from __future__ import annotations

import json
import queue
import time
import threading
import time
from typing import Optional

import config as C
import mock_source
import serial_parse


class BaseSource:
    name = "base"

    def start(self):
        pass

    def read(self) -> tuple[Optional[dict], Optional[bool]]:
        """Return (reading_dict, visual_flame_hint_or_None)."""
        raise NotImplementedError

    def stop(self):
        pass


class MockSource(BaseSource):
    name = "mock"

    def __init__(self, scenario="normal", seed=7):
        self.node = mock_source.MockNode(scenario, seed)

    def set_scenario(self, scenario):
        self.node.set_scenario(scenario)

    def read(self):
        return self.node.next()


class HttpPollSource(BaseSource):
    """
    Poll a URL that returns the contract JSON.
    Works with: ESP32 running a tiny web server, Firebase RTDB REST
    (https://<db>.firebaseio.com/latest.json), or a Google Apps Script endpoint.
    """
    name = "http"

    def __init__(self, url: str, timeout: float | None = None,
                 retries: int | None = None):
        self.url = url
        self.timeout = C.HTTP_TIMEOUT_S if timeout is None else timeout
        self.retries = C.HTTP_RETRIES if retries is None else retries
        self._last_error = ""
        self._last_good = None
        self._last_good_at = 0.0
        self.consecutive_failures = 0

    def _fetch_once(self):
        import urllib.request
        req = urllib.request.Request(self.url, headers={"Connection": "close"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode())
        if isinstance(data, dict) and "gas_ppm" not in data:
            # Firebase-style {"push_id": {...}} -> take the newest child
            children = [v for v in data.values() if isinstance(v, dict)]
            if children:
                data = sorted(children, key=lambda d: d.get("timestamp", ""))[-1]
        return data

    def read(self):
        """
        A single dropped poll is normal against a small embedded web server -
        it serves one request at a time and may be busy reading its sensors.
        Retry, and if the node is briefly unreachable keep serving the last
        good reading for a few seconds rather than blanking the dashboard.
        """
        last_exc = None
        for attempt in range(self.retries + 1):
            try:
                data = self._fetch_once()
                self._last_error = ""
                self._last_good = data
                self._last_good_at = time.time()
                self.consecutive_failures = 0
                return data, None
            except Exception as exc:
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(0.15)

        self.consecutive_failures += 1
        self._last_error = str(last_exc)
        if (self._last_good is not None
                and time.time() - self._last_good_at <= C.HTTP_STALE_GRACE_S):
            return self._last_good, None      # ride out a brief hiccup
        return None, None


class SerialSource(BaseSource):
    """
    Read the node over USB. This is the simplest and most reliable link when the
    board is plugged into the same laptop that runs the dashboard: no WiFi, no
    IP address, no network at all. It accepts both the human-readable Serial
    output and JSON firmware, so the sketch does not have to change.

    IMPORTANT: only one program may hold a COM port. The Arduino IDE's Serial
    Monitor must be CLOSED before this can connect.
    """
    name = "serial"

    def __init__(self, port: str, baud: int | str = "auto",
                 flame_active_low: bool = True):
        self.port = port
        self.baud = baud
        self.parser = serial_parse.BlockParser(flame_active_low)
        self.ser = None
        self._latest = None
        self._buf = ""
        self._last_error = ""
        self.baud_used = None

    # -- discovery ---------------------------------------------------------
    @staticmethod
    def list_ports() -> list[tuple[str, str]]:
        try:
            from serial.tools import list_ports
        except Exception:
            return []
        return [(p.device, p.description or "") for p in list_ports.comports()]

    def _open(self, baud: int):
        import serial
        return serial.Serial(self.port, baud, timeout=0.2)

    def start(self):
        import serial                                    # noqa: F401
        candidates = ([115200, 9600, 57600, 38400]
                      if self.baud == "auto" else [int(self.baud)])
        last = None
        for baud in candidates:
            try:
                ser = self._open(baud)
            except Exception as exc:
                last = exc
                continue
            # sample briefly: the right baud rate produces parseable lines
            probe = serial_parse.BlockParser()
            got = 0
            deadline = time.time() + (2.5 if self.baud == "auto" else 0.5)
            buf = ""
            while time.time() < deadline:
                try:
                    buf += ser.read(512).decode(errors="replace")
                except Exception:
                    break
                while "\n" in buf:
                    line, _, buf = buf.partition("\n")
                    if probe.feed(line):
                        got += 1
                if got >= 1 and self.baud == "auto":
                    break
            if got or self.baud != "auto":
                self.ser = ser
                self.baud_used = baud
                return
            ser.close()
        raise RuntimeError(
            f"could not read the node on {self.port}. "
            f"Close the Arduino IDE Serial Monitor (only one program can use a "
            f"COM port), check the cable, and confirm the port. Last error: {last}")

    # -- reading -----------------------------------------------------------
    def read(self):
        if self.ser is None:
            return None, None
        try:
            self._buf += self.ser.read(4096).decode(errors="replace")
        except Exception as exc:
            self._last_error = str(exc)
            return self._latest, None
        while "\n" in self._buf:
            line, _, self._buf = self._buf.partition("\n")
            r = self.parser.feed(line)
            if r:
                self._latest = r
        return self._latest, None

    def stop(self):
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None


class ReplaySource(BaseSource):
    """
    Replay a log of REAL readings captured from the node.

    This is how the two halves integrate when the hardware is at one person's
    house and the software at another's: the hardware half copies a minute of
    Serial Monitor output into a text file and sends it over. Everything except
    the live network link is then testable - field names, units, this specific
    sensor's baseline, whether the flame field toggles, how calibration behaves
    on real noise.

    Tolerant of real Serial Monitor output: non-JSON lines (boot messages, the
    [info] lines, garbage from the reset) are skipped rather than fatal.
    """
    name = "replay"

    def __init__(self, path: str, loop: bool = True):
        self.path = path
        self.loop = loop
        self.rows: list[dict] = []
        self.bad_lines = 0
        self.i = 0
        self._load()

    def _load(self):
        with open(self.path, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                start = line.find("{")
                if start == -1:
                    self.bad_lines += 1
                    continue
                try:
                    obj = json.loads(line[start:])
                except Exception:
                    self.bad_lines += 1
                    continue
                if isinstance(obj, dict) and "gas_ppm" in obj:
                    self.rows.append(obj)
                else:
                    self.bad_lines += 1
        if not self.rows:
            raise ValueError(
                f"no usable readings found in {self.path}. Expected one JSON "
                f"object per line containing gas_ppm - copy the Serial Monitor "
                f"output straight into a .txt file.")

    def read(self):
        if self.i >= len(self.rows):
            if not self.loop:
                return None, None
            self.i = 0
        row = self.rows[self.i]
        self.i += 1
        return row, None


class MqttSource(BaseSource):
    """Subscribe to the ESP32's MQTT topic. Requires `pip install paho-mqtt`."""
    name = "mqtt"

    def __init__(self, host="broker.hivemq.com", port=1883, topic="firesense/node01"):
        self.host, self.port, self.topic = host, port, topic
        self.q: "queue.Queue[dict]" = queue.Queue(maxsize=200)
        self.client = None
        self._latest = None

    def start(self):
        import paho.mqtt.client as mqtt

        def on_connect(client, *_):
            client.subscribe(self.topic)

        def on_message(_c, _u, msg):
            try:
                self.q.put_nowait(json.loads(msg.payload.decode()))
            except Exception:
                pass

        self.client = mqtt.Client()
        self.client.on_connect = on_connect
        self.client.on_message = on_message
        self.client.connect(self.host, self.port, 60)
        threading.Thread(target=self.client.loop_forever, daemon=True).start()

    def read(self):
        try:
            while True:
                self._latest = self.q.get_nowait()
        except queue.Empty:
            pass
        return self._latest, None

    def stop(self):
        if self.client:
            self.client.disconnect()


def validate(reading: dict) -> list[str]:
    """Contract check - catches a hardware-side field rename immediately."""
    problems = []
    if not isinstance(reading, dict):
        return ["payload is not a JSON object"]
    for f in C.REQUIRED_FIELDS:
        if f not in reading:
            problems.append(f"missing required field '{f}'")
    return problems


def make_source(kind: str, **kw) -> BaseSource:
    kind = kind.lower()
    if kind == "mock":
        return MockSource(**kw)
    if kind == "http":
        return HttpPollSource(**kw)
    if kind == "mqtt":
        return MqttSource(**kw)
    if kind == "replay":
        return ReplaySource(**kw)
    if kind == "serial":
        return SerialSource(**kw)
    raise ValueError(f"unknown source {kind!r}")
