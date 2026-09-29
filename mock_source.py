"""
FireSense - mock data source == the project's Test / Simulation Mode.

Two jobs, one piece of code:
  1. Day 1-5: lets the software half be built and tested with zero hardware.
  2. Demo day: a labelled, repeatable way to drive a full CRITICAL event in
     front of evaluators without an open flame - and the fallback if the live
     trigger misbehaves.

Every scenario emits EXACTLY the contract shape in config.py, so swapping to
the real ESP32 feed is a one-line change in ingest.py.
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timezone

import config as C

SCENARIOS = {
    "normal": "Quiet room. Small random drift only.",
    "cooking_false_positive": "Gas AND humidity rise together (steam + burner). "
                              "A threshold detector alarms; fusion should suppress.",
    "gradual_leak": "Gas rises steadily, humidity flat/falling. No flame. "
                    "Fusion should escalate well before the naive threshold.",
    "sudden_fire": "Fast gas + temperature rise, humidity drops, IR flame "
                   "sensor fires. Everything agrees -> CRITICAL, high confidence.",
    "sensor_fault": "BME280 returns garbage while MQ2 looks normal - exercises "
                    "the sanity check and disagreement flag.",
}


def _iso(ts=None) -> str:
    return (ts or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")


class MockNode:
    """Stateful generator so ramps are smooth and reproducible (seeded)."""

    def __init__(self, scenario: str = "normal", seed: int | None = 7):
        self.rng = random.Random(seed)
        self.step = 0
        self.set_scenario(scenario)

    def set_scenario(self, scenario: str) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; choose from {list(SCENARIOS)}")
        self.scenario = scenario
        self.step = 0

    # ------------------------------------------------------------------
    def next(self) -> dict:
        s = self.step
        self.step += 1
        j = lambda a: self.rng.uniform(-a, a)          # noqa: E731  (jitter)

        gas, temp, hum, flame = 90.0, 26.0, 55.0, False

        if self.scenario == "normal":
            gas = 90 + 12 * math.sin(s / 25.0) + j(6)
            temp = 26 + 0.4 * math.sin(s / 40.0) + j(0.15)
            hum = 55 + 1.2 * math.sin(s / 33.0) + j(0.4)

        elif self.scenario == "cooking_false_positive":
            # both gas and humidity climb together - the classic nuisance event -
            # then the burner goes off and the room ventilates back down.
            ramp = min(s, 90)
            decay = max(0, s - 90)
            fade = math.exp(-decay / 45.0)
            gas = 90 + ramp * 5.0 * fade + j(10)
            hum = 55 + ramp * 0.32 * fade + j(0.4)
            temp = 26 + ramp * 0.12 * fade + j(0.2)

        elif self.scenario == "gradual_leak":
            # gas climbs, humidity DROPS slightly (escaping LPG displaces air)
            ramp = min(s, 120)
            gas = 90 + ramp * 4.6 + j(8)
            hum = 55 - ramp * 0.06 + j(0.25)
            temp = 26 + j(0.15)

        elif self.scenario == "sudden_fire":
            ramp = min(s, 60)
            gas = 90 + ramp * 14.0 + j(15)
            temp = 26 + ramp * 0.85 + j(0.3)
            hum = 55 - ramp * 0.45 + j(0.4)
            flame = s >= 6            # IR sensor sees it shortly after ignition

        elif self.scenario == "sensor_fault":
            gas = 95 + j(8)
            temp = -273.0 if s % 3 == 0 else 26 + j(0.2)   # impossible value
            hum = 150.0 if s % 3 == 0 else 55 + j(0.5)     # out of range

        reading = {
            "node_id": "mock-01",
            "timestamp": _iso(),
            "gas_ppm": round(max(0.0, gas), 1),
            "gas_raw_adc": int(max(0, min(4095, gas * 4))),
            "temp_c": round(temp, 2),
            "humidity_pct": round(hum, 2),
            "pressure_hpa": round(1008 + j(1.5), 2),
            # True only in the fire scenario; ignored by the software unless
            # config.FLAME_SENSOR_PRESENT is set.
            "flame_detected": bool(flame),
        }
        # second element retained so every source shares one read() shape;
        # ignition now travels inside the payload as flame_detected.
        return reading, flame


def stream(scenario: str = "normal", n: int = 60, seed: int | None = 7):
    node = MockNode(scenario, seed)
    for _ in range(n):
        yield node.next()


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "gradual_leak"
    print(f"--- {name}: {SCENARIOS[name]}")
    for i, (r, flame) in enumerate(stream(name, 12)):
        print(f"{i:3d} gas={r['gas_ppm']:7.1f} T={r['temp_c']:6.2f} "
              f"RH={r['humidity_pct']:6.2f} flame={flame}")
