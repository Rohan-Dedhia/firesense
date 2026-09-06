"""
FireSense - parser for the node's human-readable Serial output.

The hardware half's sketch prints blocks like:

    --------------------------------
    Temperature : 29.13 C
    Humidity    : 74.53 %
    Pressure    : 1009.60 hPa
    MQ2 Value   : 117
    Flame Raw   : 1
    Gas Alert   : NO
    Flame Alert : NO
    Buzzer      : OFF

Rather than make them re-flash a working board, we translate this into the
contract dict the rest of the system already speaks. JSON lines are also
accepted, so either firmware works with no code change here.

Flame polarity: these IR modules pull the digital line LOW when they see a
flame, so `Flame Raw : 0` means FLAME. `Flame Alert : YES` is trusted first
when the sketch prints it, since that is the sketch's own interpretation.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

# label (case/space-insensitive) -> contract field
FIELD_MAP = {
    "temperature": "temp_c",
    "temp": "temp_c",
    "humidity": "humidity_pct",
    "pressure": "pressure_hpa",
    "mq2value": "gas_ppm",
    "mq2": "gas_ppm",
    "gas": "gas_ppm",
    "gasvalue": "gas_ppm",
    "flameraw": "_flame_raw",
    "flame": "_flame_raw",
    "flamealert": "_flame_alert",
    "gasalert": "_gas_alert",
    "buzzer": "_buzzer",
}
NUM = re.compile(r"[-+]?\d*\.?\d+")
REQUIRED = ("temp_c", "humidity_pct", "gas_ppm")


def _iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class BlockParser:
    """Feed it lines; it yields a contract dict whenever a block completes."""

    def __init__(self, flame_active_low: bool = True):
        self.cur: dict = {}
        self.flame_active_low = flame_active_low
        self.parsed = 0
        self.skipped = 0

    def feed(self, line: str):
        line = (line or "").strip()
        if not line:
            return None

        # a JSON firmware line - pass straight through
        if line.startswith("{"):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "gas_ppm" in obj:
                    self.parsed += 1
                    return obj
            except Exception:
                pass

        if set(line) <= set("-=_ *"):          # separator row ends a block
            return self._emit()

        if ":" not in line:
            self.skipped += 1
            return None

        label, _, value = line.partition(":")
        key = FIELD_MAP.get(re.sub(r"[^a-z0-9]", "", label.lower()))
        if key is None:
            self.skipped += 1
            return None

        v = value.strip()
        if key in ("_flame_alert", "_gas_alert", "_buzzer"):
            self.cur[key] = v.upper().startswith(("Y", "ON", "TRUE", "1"))
        else:
            m = NUM.search(v)
            if not m:
                return None
            self.cur[key] = float(m.group())

        # emit as soon as a block is complete, in case there is no separator
        if all(f in self.cur for f in REQUIRED) and (
                "_flame_raw" in self.cur or "_flame_alert" in self.cur):
            if "_buzzer" in self.cur or "_flame_alert" in self.cur:
                return self._emit()
        return None

    def _emit(self):
        if not all(f in self.cur for f in REQUIRED):
            self.cur = {}
            return None
        c = self.cur
        if "_flame_alert" in c:
            flame = bool(c["_flame_alert"])
        else:
            raw = c.get("_flame_raw", 1)
            flame = (raw == 0) if self.flame_active_low else (raw == 1)

        reading = {
            "node_id": "serial-node",
            "timestamp": _iso(),
            "gas_ppm": float(c["gas_ppm"]),
            "gas_raw_adc": int(c["gas_ppm"]),
            "temp_c": float(c["temp_c"]),
            "humidity_pct": float(c["humidity_pct"]),
            "pressure_hpa": float(c.get("pressure_hpa", 0.0)),
            "flame_detected": bool(flame),
        }
        self.cur = {}
        self.parsed += 1
        return reading


def parse_text(text: str, flame_active_low: bool = True) -> list[dict]:
    p = BlockParser(flame_active_low)
    out = []
    for line in text.splitlines():
        r = p.feed(line)
        if r:
            out.append(r)
    r = p._emit()
    if r:
        out.append(r)
    return out
