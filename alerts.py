"""
FireSense - Telegram alerting with cooldown + a CSV incident log.

Setup (about 15 minutes, do it on Day 5):
  1. Telegram -> @BotFather -> /newbot -> copy the token
  2. Send your bot any message, then open
     https://api.telegram.org/bot<TOKEN>/getUpdates  and copy result[0].message.chat.id
  3. Put both in config.py, or set env TELEGRAM_TOKEN / TELEGRAM_CHAT_ID
"""
from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from pathlib import Path

import config as C

LOG_PATH = Path("logs")
LOG_PATH.mkdir(exist_ok=True)


def _creds():
    return (os.environ.get("TELEGRAM_TOKEN", C.TELEGRAM_TOKEN),
            os.environ.get("TELEGRAM_CHAT_ID", C.TELEGRAM_CHAT_ID))


def send_telegram(text: str) -> tuple[bool, str]:
    token, chat = _creds()
    if not token or not chat:
        return False, "Telegram not configured (set TELEGRAM_TOKEN / TELEGRAM_CHAT_ID)"
    try:
        import urllib.parse
        import urllib.request
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
        with urllib.request.urlopen(url, data=data, timeout=5) as r:
            r.read()
        return True, "sent"
    except Exception as exc:
        return False, f"send failed: {exc}"


class AlertManager:
    """Fires on state TRANSITIONS only, with a cooldown, so no spam."""

    def __init__(self, cooldown_s: float = C.ALERT_COOLDOWN_S):
        self.cooldown = cooldown_s
        self.last_level = "NORMAL"
        self.last_sent_at = 0.0
        self.sent: list[dict] = []

    def maybe_alert(self, result, force: bool = False) -> tuple[bool, str]:
        level = result.level
        transition = level != self.last_level
        escalation = transition and level in ("WARNING", "CRITICAL")
        self.last_level = level
        if not (force or escalation):
            return False, "no transition"
        if time.time() - self.last_sent_at < self.cooldown and not force:
            return False, "cooldown"

        icon = {"WARNING": "[WARNING]", "CRITICAL": "[CRITICAL]"}.get(level, "[INFO]")
        eta = (f"\nETA to CRITICAL: {result.seconds_to_critical:.0f}s"
               if result.seconds_to_critical else "")
        text = (
            f"{icon} FireSense - {result.environment}\n"
            f"Risk {result.score:.0f}/100  (confidence {result.confidence:.0f} %)\n"
            f"Gas {result.gas_compensated:.0f} ppm | {result.temp_c:.1f} C | "
            f"{result.humidity_pct:.0f} %RH | flame: "
            f"{'YES' if result.visual_fire else 'no'}\n"
            f"Pattern: {result.signature}{eta}\n"
            f"{result.explanation}"
        )
        ok, msg = send_telegram(text)
        self.last_sent_at = time.time()
        self.sent.append({"t": time.time(), "level": level, "ok": ok, "msg": msg})
        return ok, msg


def append_log(result, path: Path | None = None) -> Path:
    """Append one row to the session CSV (feeds the exportable incident report)."""
    path = path or LOG_PATH / f"session_{datetime.now():%Y%m%d}.csv"
    new = not path.exists()
    row = {
        "timestamp": datetime.fromtimestamp(result.ts).isoformat(timespec="seconds"),
        "environment": result.environment,
        "gas_raw": result.gas_raw, "gas_compensated": result.gas_compensated,
        "temp_c": result.temp_c, "humidity_pct": result.humidity_pct,
        "visual_fire": result.visual_fire, "visual_conf": result.visual_confidence,
        # the ignition channels - without these a log cannot explain why an
        # event was or was not classified as a fire
        "flame_sensor": result.flame_sensor,
        "flame_available": result.flame_available,
        "ignition_score": result.ignition_score,
        "gas_score": result.gas_score, "temp_score": result.temp_score,
        "visual_score": result.visual_score,
        "score": result.score, "level": result.level,
        "confidence": result.confidence, "signature": result.signature,
        "suppressed": result.suppressed,
        "temp_slope_c_s": round(result.slopes.temp, 4),
        "gas_slope_ppm_s": round(result.slopes.gas, 3),
        "hum_slope_pct_s": round(result.slopes.humidity, 3),
        "eta_critical_s": result.seconds_to_critical,
        "naive_alarm": result.naive_alarm,
        "flags": " | ".join(result.flags),
    }
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    return path
