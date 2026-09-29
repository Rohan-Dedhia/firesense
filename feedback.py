"""
FireSense - alerts with a photo, and the human-in-the-loop labelling that
comes back from them.

When the system alarms it sends the phone a picture and two buttons:

        [ Real ]   [ False alarm ]

Tapping one does two things. It tells the person watching that their answer
was heard - and it produces a labelled training example, captured with the
full sensor window that caused the alert.

That is the point. The dataset nobody has published is nuisance events on real
hardware, and this collects them from ordinary use rather than from someone
standing in a kitchen with a stopwatch. A deployed system that gets better the
longer it runs is a different kind of project from one that was trained once.

    from feedback import FeedbackLoop

    loop = FeedbackLoop()
    loop.alert(result, frame)          # sends photo + buttons, stores context
    loop.poll()                        # collects taps, writes labels

    python feedback.py --listen        # run the collector on its own
    python feedback.py --export        # labelled examples -> CSV

Setup is the same Telegram bot as the text alerts - see config.py.
"""
from __future__ import annotations

import csv
import json
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import config as C

# Overridable so the tests can point at a local stand-in rather than Telegram.
API_BASE = os.environ.get("TELEGRAM_API_BASE", "https://api.telegram.org")

ALERTS_DIR = Path("data") / "alerts"
FEEDBACK_CSV = Path("data") / "feedback.csv"
OFFSET_FILE = Path("data") / ".telegram_offset"

VERDICT_REAL = "real"
VERDICT_FALSE = "false_alarm"


def _creds() -> tuple[str, str]:
    return (os.environ.get("TELEGRAM_TOKEN", C.TELEGRAM_TOKEN),
            os.environ.get("TELEGRAM_CHAT_ID", C.TELEGRAM_CHAT_ID))


def _api(method: str) -> str:
    token, _ = _creds()
    return f"{API_BASE}/bot{token}/{method}"


# ---------------------------------------------------------------------------
# multipart, by hand - no third-party HTTP library needed
# ---------------------------------------------------------------------------
def _multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    """fields: name -> str.  files: name -> (filename, bytes)."""
    boundary = f"----FireSense{uuid.uuid4().hex}"
    out = bytearray()
    for name, value in fields.items():
        out += f"--{boundary}\r\n".encode()
        out += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        out += f"{value}\r\n".encode()
    for name, (filename, blob) in files.items():
        ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        out += f"--{boundary}\r\n".encode()
        out += (f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\n').encode()
        out += f"Content-Type: {ctype}\r\n\r\n".encode()
        out += blob + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def _post(method: str, data: bytes, content_type: str,
          timeout: float = 15.0) -> tuple[bool, dict]:
    req = urllib.request.Request(_api(method), data=data,
                                 headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return True, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:300]
        return False, {"error": f"HTTP {exc.code}: {body}"}
    except Exception as exc:
        return False, {"error": f"{type(exc).__name__}: {exc}"}


def _post_form(method: str, fields: dict, timeout: float = 15.0):
    body = urllib.parse.urlencode(fields).encode()
    return _post(method, body, "application/x-www-form-urlencoded", timeout)


# ---------------------------------------------------------------------------
# the keyboard
# ---------------------------------------------------------------------------
def verdict_keyboard(alert_id: str) -> str:
    """Two buttons. The alert id rides in callback_data so a tap can be matched
    back to the exact reading that produced it, even hours later."""
    return json.dumps({
        "inline_keyboard": [[
            {"text": "✅ Real", "callback_data": f"{VERDICT_REAL}:{alert_id}"},
            {"text": "❌ False alarm",
             "callback_data": f"{VERDICT_FALSE}:{alert_id}"},
        ]]
    })


def parse_callback(update: dict) -> Optional[tuple[str, str, str]]:
    """
    One getUpdates entry -> (verdict, alert_id, callback_query_id), or None.

    Anything unexpected returns None rather than raising: this parses data
    arriving from the internet, and a malformed update must not stop the
    detector from running.
    """
    cq = (update or {}).get("callback_query")
    if not isinstance(cq, dict):
        return None
    data = cq.get("data")
    if not isinstance(data, str) or ":" not in data:
        return None
    verdict, _, alert_id = data.partition(":")
    if verdict not in (VERDICT_REAL, VERDICT_FALSE) or not alert_id:
        return None
    return verdict, alert_id, str(cq.get("id", ""))


# ---------------------------------------------------------------------------
# what an alert looked like when it fired
# ---------------------------------------------------------------------------
@dataclass
class AlertContext:
    alert_id: str
    sent_at: float
    level: str
    score: float
    signature: str
    gas_ppm: float
    gas_raw: float
    temp_c: float
    humidity_pct: float
    flame_sensor: bool
    gas_score: float
    temp_score: float
    ignition_score: float
    vision_score: float = 0.0
    environment: str = ""
    frame_path: str = ""

    @classmethod
    def from_result(cls, result, alert_id: str, vision_score: float = 0.0,
                    frame_path: str = "") -> "AlertContext":
        return cls(
            alert_id=alert_id, sent_at=time.time(),
            level=getattr(result, "level", ""),
            score=float(getattr(result, "score", 0.0)),
            signature=getattr(result, "signature", ""),
            gas_ppm=float(getattr(result, "gas_compensated", 0.0)),
            gas_raw=float(getattr(result, "gas_raw", 0.0)),
            temp_c=float(getattr(result, "temp_c", 0.0)),
            humidity_pct=float(getattr(result, "humidity_pct", 0.0)),
            flame_sensor=bool(getattr(result, "flame_sensor", False)),
            gas_score=float(getattr(result, "gas_score", 0.0)),
            temp_score=float(getattr(result, "temp_score", 0.0)),
            ignition_score=float(getattr(result, "ignition_score", 0.0)),
            vision_score=float(vision_score),
            environment=getattr(result, "environment", ""),
            frame_path=frame_path,
        )


def caption_for(ctx: AlertContext, explanation: str = "") -> str:
    """What the person reads on their phone. They may be in a meeting, so it
    leads with the judgement and keeps the numbers to one line."""
    lines = [
        f"{'CRITICAL' if ctx.level == 'CRITICAL' else 'WARNING'} - "
        f"{ctx.environment or 'FireSense'}",
        f"Risk {ctx.score:.0f}/100   pattern: {ctx.signature.replace('_', ' ')}",
        f"Gas {ctx.gas_ppm:.0f} ppm | {ctx.temp_c:.1f} C | "
        f"{ctx.humidity_pct:.0f} %RH | flame: "
        f"{'YES' if ctx.flame_sensor else 'no'}",
    ]
    if explanation:
        lines.append(explanation)
    lines.append("\nIs this real?")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------
class FeedbackStore:
    """Alert contexts as JSON, verdicts as one CSV. Deliberately plain files -
    this has to survive being copied between two laptops."""

    FIELDS = ("alert_id", "verdict", "answered_at", "sent_at", "level", "score",
              "signature", "gas_ppm", "gas_raw", "temp_c", "humidity_pct",
              "flame_sensor", "gas_score", "temp_score", "ignition_score",
              "vision_score", "environment", "frame_path")

    def __init__(self, alerts_dir: Path = ALERTS_DIR, csv_path: Path = FEEDBACK_CSV):
        self.alerts_dir = Path(alerts_dir)
        self.csv_path = Path(csv_path)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def save_context(self, ctx: AlertContext) -> Path:
        p = self.alerts_dir / f"{ctx.alert_id}.json"
        p.write_text(json.dumps(asdict(ctx), indent=2))
        return p

    def load_context(self, alert_id: str) -> Optional[AlertContext]:
        p = self.alerts_dir / f"{alert_id}.json"
        if not p.exists():
            return None
        try:
            return AlertContext(**json.loads(p.read_text()))
        except Exception:
            return None

    def already_answered(self, alert_id: str) -> bool:
        if not self.csv_path.exists():
            return False
        with self.csv_path.open(newline="") as fh:
            return any(row.get("alert_id") == alert_id
                       for row in csv.DictReader(fh))

    def record(self, alert_id: str, verdict: str) -> bool:
        """-> True if this produced a new labelled example."""
        ctx = self.load_context(alert_id)
        if ctx is None or self.already_answered(alert_id):
            return False
        row = asdict(ctx)
        row.update({"verdict": verdict,
                    "answered_at": round(time.time(), 3)})
        new = not self.csv_path.exists()
        with self.csv_path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(self.FIELDS),
                               extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
        return True

    def counts(self) -> dict:
        out = {VERDICT_REAL: 0, VERDICT_FALSE: 0}
        if not self.csv_path.exists():
            return out
        with self.csv_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                v = row.get("verdict")
                if v in out:
                    out[v] += 1
        return out


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
class FeedbackLoop:
    def __init__(self, store: Optional[FeedbackStore] = None):
        self.store = store or FeedbackStore()
        self.offset_file = OFFSET_FILE

    # -- sending ----------------------------------------------------------
    def alert(self, result, image_bytes: Optional[bytes] = None,
              vision_score: float = 0.0, frame_path: str = "",
              explanation: str = "") -> tuple[bool, str, str]:
        """
        Send one alert with buttons. -> (ok, message, alert_id)

        The context is stored BEFORE the send, so a tap can always be matched
        even if the network hiccups between the two.
        """
        token, chat = _creds()
        alert_id = uuid.uuid4().hex[:12]
        ctx = AlertContext.from_result(result, alert_id,
                                       vision_score=vision_score,
                                       frame_path=frame_path)
        self.store.save_context(ctx)

        if not token or not chat:
            return False, "Telegram not configured", alert_id

        caption = caption_for(ctx, explanation or
                              getattr(result, "explanation", ""))
        markup = verdict_keyboard(alert_id)

        if image_bytes:
            body, ctype = _multipart(
                {"chat_id": chat, "caption": caption[:1024],
                 "reply_markup": markup},
                {"photo": (f"{alert_id}.jpg", image_bytes)})
            ok, resp = _post("sendPhoto", body, ctype)
        else:
            ok, resp = _post_form("sendMessage", {
                "chat_id": chat, "text": caption, "reply_markup": markup})

        if not ok:
            return False, str(resp.get("error", "send failed")), alert_id
        if not resp.get("ok", True):
            return False, str(resp.get("description", resp)), alert_id
        return True, "sent", alert_id

    # -- receiving --------------------------------------------------------
    def _read_offset(self) -> int:
        try:
            return int(self.offset_file.read_text().strip())
        except Exception:
            return 0

    def _write_offset(self, value: int) -> None:
        try:
            self.offset_file.parent.mkdir(parents=True, exist_ok=True)
            self.offset_file.write_text(str(value))
        except Exception:
            pass

    def poll(self, timeout: int = 0) -> list[tuple[str, str]]:
        """
        Collect taps since the last poll. -> [(alert_id, verdict), ...]

        The offset is persisted, so restarting the collector does not replay
        every button press ever made.
        """
        token, _ = _creds()
        if not token:
            return []
        offset = self._read_offset()
        fields = {"timeout": timeout, "allowed_updates": '["callback_query"]'}
        if offset:
            fields["offset"] = offset
        ok, resp = _post_form("getUpdates", fields, timeout=timeout + 15)
        if not ok or not resp.get("ok", False):
            return []

        collected = []
        highest = offset
        for update in resp.get("result", []) or []:
            uid = int(update.get("update_id", 0) or 0)
            highest = max(highest, uid + 1)
            parsed = parse_callback(update)
            if parsed is None:
                continue
            verdict, alert_id, cq_id = parsed
            if self.store.record(alert_id, verdict):
                collected.append((alert_id, verdict))
            self._acknowledge(cq_id, verdict)
        if highest > offset:
            self._write_offset(highest)
        return collected

    def _acknowledge(self, callback_query_id: str, verdict: str) -> None:
        """Stop the button spinning, and confirm the answer landed."""
        if not callback_query_id:
            return
        text = ("Logged as a real event - thank you."
                if verdict == VERDICT_REAL else
                "Logged as a false alarm. The system will learn from this.")
        _post_form("answerCallbackQuery",
                   {"callback_query_id": callback_query_id, "text": text})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _listen(interval: float) -> None:                        # pragma: no cover
    loop = FeedbackLoop()
    token, chat = _creds()
    if not token or not chat:
        print("Telegram is not configured. Set TELEGRAM_TOKEN and "
              "TELEGRAM_CHAT_ID (see config.py).")
        return
    print(f"listening for verdicts every {interval:.0f}s - Ctrl+C to stop")
    try:
        while True:
            for alert_id, verdict in loop.poll():
                print(f"  {datetime.now():%H:%M:%S}  {alert_id}  {verdict}")
            c = loop.store.counts()
            print(f"\r  real {c[VERDICT_REAL]}   "
                  f"false alarms {c[VERDICT_FALSE]}   ", end="", flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nstopped.")


def _export() -> None:                                       # pragma: no cover
    store = FeedbackStore()
    c = store.counts()
    total = c[VERDICT_REAL] + c[VERDICT_FALSE]
    if not total:
        print("No verdicts yet. Alerts collect them once the system is running.")
        return
    print(f"\n{total} labelled examples from real use")
    print(f"  real         {c[VERDICT_REAL]}")
    print(f"  false alarms {c[VERDICT_FALSE]}")
    print(f"\n  -> {store.csv_path}")
    print("  Each row carries the full sensor context that caused the alert,")
    print("  so these join straight onto the recorded sessions.\n")


if __name__ == "__main__":                                   # pragma: no cover
    import argparse
    p = argparse.ArgumentParser(description="FireSense alert feedback loop.")
    p.add_argument("--listen", action="store_true", help="collect verdicts")
    p.add_argument("--export", action="store_true", help="show what was collected")
    p.add_argument("--interval", type=float, default=5.0)
    a = p.parse_args()
    if a.listen:
        _listen(a.interval)
    else:
        _export()
