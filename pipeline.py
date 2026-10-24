"""
FireSense - the running system.

Everything else in this project is a component. This is the thing that runs:

    sensors ──┐
              ├──► rule engine ──┐
    camera ───┼──► YOLO/MaxViT ──┼──► decision ──► alert + photo + buttons
              └──► VLM ──────────┘                      │
                  (on escalation only)                  └──► CSV log

    python pipeline.py --source mock --scenario sudden_fire
    python pipeline.py --source serial --camera webcam
    python pipeline.py --source serial --camera 192.168.1.42 --vlm
    python pipeline.py --source serial --camera webcam --vision maxvit --alert-boxes

Why the rule engine is still here
---------------------------------
fusion.FusionEngine is deliberately untouched by the vision channels. It is
the BASELINE the learned model has to beat, and a baseline you keep improving
is not a baseline. So vision and the VLM are combined here, at the pipeline
level, on top of its verdict - and when the learned model is ready it replaces
this combination step rather than the engine.

How the channels combine
------------------------
Same escalator rule as everywhere else in this project: vision can raise the
level, never lower it. A model that scores 0 has seen no fire, which is not
the same as there being none - and on this project's own webcam, a community
fire model reported fire on an empty room for twenty consecutive frames. That
is the argument for the rule, made by measurement rather than assertion.

The vision channel is either the YOLO detector (vision_ml, default) or the
MaxViT-tiny classifier trained on D-Fire (vision_maxvit, `--vision maxvit`).
Both feed combine() through the same VisionResult, so the escalator rule and
the VISION_ALONE_MAX cap apply to either. `--alert-boxes` adds YOLO v2 boxes
to the alert photo for the person reading it; the boxes are not scored.

The VLM runs only on escalation. It takes seconds; calling it every second
would make the system slower than the fire.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import alerts
import config as C
import fusion
import ingest

# Optional channels. Each is absent in ordinary development and the system
# must run without it, saying so, rather than refusing to start.
try:
    import camera as camera_mod
    HAVE_CAMERA = True
except Exception:                                           # pragma: no cover
    camera_mod, HAVE_CAMERA = None, False

try:
    import vision_ml
    HAVE_VISION = True
except Exception:                                           # pragma: no cover
    vision_ml, HAVE_VISION = None, False

try:
    import vision_maxvit
    HAVE_MAXVIT = True
except Exception:                                           # pragma: no cover
    vision_maxvit, HAVE_MAXVIT = None, False

try:
    import vlm as vlm_mod
    HAVE_VLM = True
except Exception:                                           # pragma: no cover
    vlm_mod, HAVE_VLM = None, False

try:
    import feedback as feedback_mod
    HAVE_FEEDBACK = True
except Exception:                                           # pragma: no cover
    feedback_mod, HAVE_FEEDBACK = None, False


# How far a confirmed visual detection may push the level on its own, when the
# chemical and thermal channels are quiet. Deliberately capped below CRITICAL:
# a camera that has been wrong before does not get to declare an emergency by
# itself, it gets to raise a warning and bring the other channels' attention.
VISION_ALONE_MAX = 60.0
VISION_WEIGHT = 0.40                  # matches the ignition weight in config
VLM_ESCALATION_LEVELS = ("WARNING", "CRITICAL")
VISION_KINDS = ("yolo", "maxvit")
DEFAULT_WEIGHTS = {"yolo": "models/fire_yolo.pt",
                   "maxvit": "models/maxvit_dfire/best.pt"}
DEFAULT_BOX_WEIGHTS = "models/yolo_dfire_v2/weights/best.pt"


@dataclass
class PipelineResult:
    """One pass of the whole system."""
    ts: float
    engine: object = None                # the fusion.RiskResult, untouched
    vision_score: float = 0.0
    vision_note: str = ""
    vision_available: bool = False
    vlm_scene: str = ""
    vlm_score: float = 0.0
    vlm_ran: bool = False
    final_score: float = 0.0
    final_level: str = "NORMAL"
    escalated_by: str = ""               # what raised it above the engine
    alert_sent: bool = False
    alert_id: str = ""
    notes: list = field(default_factory=list)


def combine(engine_score: float, engine_level: str, vision_score: float,
            vision_confirmed: bool, vlm_score: float = 0.0,
            vlm_ran: bool = False) -> tuple[float, str, str]:
    """
    Engine verdict + vision -> final verdict. Pure, so it can be tested.

    -> (score, level, what_escalated)
    """
    score = engine_score
    by = ""

    if vision_confirmed and vision_score > 0:
        blended = (1.0 - VISION_WEIGHT) * engine_score + VISION_WEIGHT * vision_score
        raised = max(engine_score, blended)
        # The cap limits what vision may do ON ITS OWN, and only then. With
        # the chemical and thermal channels quiet, a camera that has been
        # wrong before gets to raise a warning, not declare an emergency. Once
        # the engine is itself concerned, vision is corroborating rather than
        # acting alone, and the cap would only suppress agreement between
        # independent channels - which is the opposite of what fusion is for.
        if engine_score < C.BAND_WARNING:
            raised = min(raised, VISION_ALONE_MAX)
        if raised > score:
            score = raised
            by = "vision"

    if vlm_ran and vlm_score > score:
        # The VLM has the most context of any channel, so when it disagrees
        # upward it wins - but it only ever runs after something else was
        # already concerned.
        score = vlm_score
        by = "vlm" if not by else "vision+vlm"

    score = max(0.0, min(100.0, score))
    if score >= C.BAND_CRITICAL:
        level = "CRITICAL"
    elif score >= C.BAND_WARNING:
        level = "WARNING"
    else:
        level = "NORMAL"

    # Never de-escalate below what the engine itself concluded.
    rank = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}
    if rank[level] < rank[engine_level]:
        level = engine_level
        by = ""
    return round(score, 1), level, by


class FireSensePipeline:
    def __init__(self, source, camera_spec: Optional[str] = None,
                 weights: Optional[str] = None,
                 use_vlm: bool = False, vlm_model: str = "moondream",
                 environment: str = C.DEFAULT_ENVIRONMENT,
                 vision_every: int = 2, log_csv: bool = True,
                 send_alerts: bool = True, vision: str = "yolo",
                 box_weights: Optional[str] = None):
        if vision not in VISION_KINDS:
            raise ValueError(f"vision must be one of {VISION_KINDS}, not {vision!r}")
        self.source = source
        self.vision_kind = vision
        self.engine = fusion.FusionEngine(environment)
        self.log_csv = log_csv
        self.send_alerts = send_alerts
        self.vision_every = max(1, vision_every)
        self._tick = 0
        self._last_level = "NORMAL"
        self.startup_notes: list[str] = []

        # -- camera -------------------------------------------------------
        self.camera = None
        if camera_spec and HAVE_CAMERA:
            cam = camera_mod.make_camera(camera_spec)
            if cam.open():
                self.camera = cam
            else:
                self.startup_notes.append(
                    f"camera unavailable ({getattr(cam, 'error', '?')}) - "
                    f"running without the visual channels")

        # -- vision: YOLO detector or MaxViT classifier ---------------------
        self.vision = None
        weights = weights or DEFAULT_WEIGHTS[vision]
        if self.camera is not None:
            ch = None
            if vision == "maxvit" and HAVE_MAXVIT:
                ch = vision_maxvit.MaxVitChannel(weights)
            elif vision == "yolo" and HAVE_VISION:
                ch = vision_ml.VisionChannel(weights)
            if ch is not None and ch.load():
                self.vision = ch
                if vision == "maxvit":
                    thr = ch.detector.thresholds
                    self.startup_notes.append(
                        f"MaxViT thresholds fire {thr['fire']:.2f} / smoke "
                        f"{thr['smoke']:.2f} come from D-Fire validation, not this "
                        f"room - check with evaluate_on_frames.py")
            elif ch is not None:
                self.startup_notes.append(f"vision channel off: {ch.error}")

        # -- boxes for the alert photo (display only) ---------------------
        self.alert_boxes = None
        if box_weights and self.camera is not None and HAVE_MAXVIT:
            ab = vision_maxvit.AlertBoxes(box_weights)
            if ab.load():
                self.alert_boxes = ab
            else:
                self.startup_notes.append(f"alert boxes off: {ab.error}")

        # -- VLM ----------------------------------------------------------
        self.vlm = None
        if use_vlm and self.camera is not None and HAVE_VLM:
            v = vlm_mod.VLMChannel(model=vlm_model)
            if v.available():
                self.vlm = v
            else:
                self.startup_notes.append(f"VLM off: {v.error}")

        # -- alerting -----------------------------------------------------
        self.feedback = (feedback_mod.FeedbackLoop()
                         if (send_alerts and HAVE_FEEDBACK) else None)
        self.alerts = alerts.AlertManager()

    # -- one pass ---------------------------------------------------------
    def step(self) -> Optional[PipelineResult]:
        self._tick += 1
        reading, _ = self.source.read()
        if not reading:
            return None

        problems = ingest.validate(reading)
        if problems:
            return PipelineResult(ts=time.time(),
                                  notes=[f"contract violation: {'; '.join(problems)}"])

        res = PipelineResult(ts=time.time())
        res.engine = self.engine.update(reading)

        frame = None
        if self.camera is not None and (self._tick % self.vision_every == 0):
            f = self.camera.read()
            if f.ok:
                frame = f.image
            else:
                res.notes.append(f"camera: {f.error}")

        # -- vision -------------------------------------------------------
        vision_confirmed = False
        if self.vision is not None and frame is not None:
            v = self.vision.update(frame)
            res.vision_available = v.available
            res.vision_score = v.score
            res.vision_note = v.note
            vision_confirmed = v.fire_detected

        # -- combine ------------------------------------------------------
        score, level, by = combine(
            res.engine.score, res.engine.level, res.vision_score, vision_confirmed)

        # -- VLM, only when something is already concerning -----------------
        escalating = (level in VLM_ESCALATION_LEVELS
                      and self._last_level == "NORMAL")
        if self.vlm is not None and frame is not None and escalating:
            vr = self.vlm.describe_frame(frame)
            if vr.available:
                res.vlm_ran = True
                res.vlm_scene = vr.scene
                res.vlm_score = vr.score
                res.notes.append(vr.note)
                score, level, by = combine(
                    res.engine.score, res.engine.level, res.vision_score,
                    vision_confirmed, vr.score, True)

        res.final_score, res.final_level, res.escalated_by = score, level, by

        # -- act ----------------------------------------------------------
        if self.log_csv:
            try:
                alerts.append_log(res.engine)
            except Exception as exc:
                res.notes.append(f"log failed: {exc}")

        if (self.send_alerts and level != self._last_level
                and level in ("WARNING", "CRITICAL")):
            res.alert_sent, res.alert_id = self._alert(res, frame)

        self._last_level = level
        return res

    def _alert(self, res: PipelineResult, frame) -> tuple[bool, str]:
        image = None
        if frame is not None and HAVE_CAMERA:
            try:
                import cv2
                if self.alert_boxes is not None:
                    try:
                        frame = self.alert_boxes.draw(frame)
                    except Exception as exc:     # a box failure never blocks the alert
                        res.notes.append(f"alert boxes failed: {exc}")
                ok, buf = cv2.imencode(".jpg", frame)
                image = buf.tobytes() if ok else None
            except Exception:
                image = None
        if self.feedback is not None:
            ok, _, alert_id = self.feedback.alert(
                res.engine, image_bytes=image, vision_score=res.vision_score,
                explanation=res.vlm_scene or res.engine.explanation)
            return ok, alert_id
        ok, _ = self.alerts.maybe_alert(res.engine, force=True)
        return ok, ""

    def poll_feedback(self) -> list:
        return self.feedback.poll() if self.feedback is not None else []

    def close(self) -> None:
        try:
            self.source.stop()
        except Exception:
            pass
        if self.camera is not None:
            self.camera.close()

    # -- what is actually switched on -------------------------------------
    def channel_summary(self) -> str:
        on = ["chemical+thermal+IR (rule engine)"]
        if self.vision is not None:
            on.append({"yolo": "YOLO vision",
                       "maxvit": "MaxViT vision"}[self.vision_kind])
        if self.alert_boxes is not None:
            on.append("YOLO v2 boxes on alert photo")
        if self.vlm is not None:
            on.append("VLM")
        if self.feedback is not None:
            on.append("Telegram + feedback")
        return ", ".join(on)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run(args) -> int:                                        # pragma: no cover
    if args.source == "serial":
        ports = ingest.SerialSource.likely_ports()
        if not ports and not args.port:
            print("No serial ports found. Plug the node in and close the "
                  "Arduino Serial Monitor.")
            return 1
        port = args.port or ports[0][0]
        src = ingest.SerialSource(port, baud=args.baud)
        src.start()
        desc = f"serial:{port}"
    elif args.source == "http":
        src, desc = ingest.HttpPollSource(args.url or C.ESP32_URL), "http"
    else:
        src, desc = ingest.MockSource(args.scenario), f"mock:{args.scenario}"

    pipe = FireSensePipeline(
        src, camera_spec=args.camera, weights=args.weights,
        use_vlm=args.vlm, vlm_model=args.vlm_model,
        environment=args.environment, vision_every=args.vision_every,
        log_csv=not args.no_log, send_alerts=not args.no_alerts,
        vision=args.vision, box_weights=args.alert_boxes)

    print("=" * 70)
    print(f"  FireSense   source {desc}   env {args.environment}")
    print(f"  channels: {pipe.channel_summary()}")
    for n in pipe.startup_notes:
        print(f"  note: {n}")
    print("=" * 70)
    print("  Ctrl+C to stop\n")

    deadline = time.time() + args.minutes * 60 if args.minutes else None
    last_poll = 0.0
    try:
        while deadline is None or time.time() < deadline:
            t0 = time.time()
            res = pipe.step()
            if res is not None and res.engine is not None:
                e = res.engine
                flag = {"NORMAL": "  ", "WARNING": "! ",
                        "CRITICAL": "!!"}[res.final_level]
                line = (f"{flag} {res.final_level:8s} {res.final_score:5.1f}  "
                        f"gas {e.gas_compensated:6.0f}  {e.temp_c:5.1f}C  "
                        f"{e.humidity_pct:4.0f}%  {e.signature}")
                if res.vision_available and res.vision_score:
                    line += f"  | vision {res.vision_score:.0f}"
                if res.escalated_by:
                    line += f"  <- {res.escalated_by}"
                if res.vlm_ran:
                    line += f"\n     VLM: {res.vlm_scene}"
                if res.alert_sent:
                    line += "\n     alert sent to Telegram"
                print(line)
                for n in res.notes:
                    print(f"     {n}")

            if time.time() - last_poll > 10:
                for alert_id, verdict in pipe.poll_feedback():
                    print(f"     feedback: {alert_id} -> {verdict}")
                last_poll = time.time()

            time.sleep(max(0.0, C.SAMPLE_INTERVAL_S - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        pipe.close()
    return 0


def main() -> None:                                          # pragma: no cover
    p = argparse.ArgumentParser(description="Run the whole FireSense system.")
    p.add_argument("--source", default="mock", help="serial | http | mock")
    p.add_argument("--port", default=None)
    p.add_argument("--baud", default="auto")
    p.add_argument("--url", default=None)
    p.add_argument("--scenario", default="sudden_fire")
    p.add_argument("--camera", default=None,
                   help="webcam | webcam:1 | <esp32 ip> - omit for sensors only")
    p.add_argument("--vision", choices=VISION_KINDS, default="yolo",
                   help="yolo = vision_ml detector; maxvit = D-Fire classifier")
    p.add_argument("--weights", default=None,
                   help="default: models/fire_yolo.pt (yolo) or "
                        "models/maxvit_dfire/best.pt (maxvit)")
    p.add_argument("--alert-boxes", nargs="?", const=DEFAULT_BOX_WEIGHTS,
                   default=None, metavar="WEIGHTS",
                   help="draw YOLO boxes on the alert photo (display only); "
                        f"default weights {DEFAULT_BOX_WEIGHTS}")
    p.add_argument("--vlm", action="store_true", help="describe scenes on escalation")
    p.add_argument("--vlm-model", default="moondream")
    p.add_argument("--environment", default=C.DEFAULT_ENVIRONMENT)
    p.add_argument("--vision-every", type=int, default=2,
                   help="run the vision model every Nth reading (default 2)")
    p.add_argument("--minutes", type=float, default=0.0, help="0 = until Ctrl+C")
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--no-alerts", action="store_true")
    sys.exit(run(p.parse_args()))


if __name__ == "__main__":
    main()
