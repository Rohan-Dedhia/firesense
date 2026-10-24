"""
FireSense - continue the YOLOv8n fine-tune as a NEW run (validation only).

    python train_yolo_v2.py              # up to 15 epochs, then image-level validation
    python train_yolo_v2.py --validate   # only redo the validation step

The first run (train_yolo.py -> models/yolo_dfire) stopped after 6 epochs
with box mAP50 still rising. This run starts from its weights/last.pt and
writes to models/yolo_dfire_v2; models/yolo_dfire is never written.

It is a new run, not an ultralytics `resume=True`: the optimiser, learning-rate
warm-up and schedule start again (ultralytics defaults, same as run 1). The
grouped train/validation split and the image-level metrics are the same as
train_yolo.py and train_vision.py, so the numbers are comparable.

There is deliberately no --test. The D-Fire test split was already used once
for YOLO (models/yolo_dfire/test_results.json); a second look would let test
results steer the model.

Epoch guard. In run 1 one epoch took 254 min against 8-13 min for the others,
and the cause was not found. Here every epoch's wall time is logged, and an
epoch over 20 minutes ends the run:
  * a batch callback stops training cleanly once the epoch's training part
    passes 20 min (ultralytics then validates and saves what it has);
  * a watchdog thread kills the process if an epoch reaches 25 min - the
    backstop for a stall inside a single step, where no callback can fire.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import train_yolo as Y

START = Path("models") / "yolo_dfire" / "weights" / "last.pt"
OUT = Path("models") / "yolo_dfire_v2"
EPOCHS = 15
EPOCH_LIMIT_S = 20 * 60       # the user's limit: an epoch over this stops the run
WATCHDOG_S = 25 * 60          # hard kill if an epoch is stuck past this
EPOCH_LOG = OUT / "epoch_times.csv"
HYPER = {"model": "continued from models/yolo_dfire/weights/last.pt (6 epochs)",
         "epochs_requested": EPOCHS, "time_cap_hours": None,
         "epoch_limit_minutes": EPOCH_LIMIT_S / 60,
         "schedule": "new run: ultralytics default warm-up and lr schedule restart"}


class EpochGuard:
    """Times each epoch, logs it, and enforces the 20-minute limit."""

    def __init__(self):
        self.epoch_start = None
        self.epoch = 0
        self.tripped = None           # reason string once the limit is hit
        self._done = threading.Event()

    def _log(self, line: str) -> None:
        with open(EPOCH_LOG, "a") as f:
            f.write(line + "\n")

    def on_train_start(self, trainer) -> None:
        EPOCH_LOG.write_text("epoch,wall_seconds,wall_minutes,box_map50,note\n")
        threading.Thread(target=self._watchdog, daemon=True).start()

    def on_train_epoch_start(self, trainer) -> None:
        self.epoch = trainer.epoch + 1
        self.epoch_start = time.time()

    def on_train_batch_end(self, trainer) -> None:
        if time.time() - self.epoch_start > EPOCH_LIMIT_S and not self.tripped:
            self.tripped = f"epoch {self.epoch} passed {EPOCH_LIMIT_S // 60} min while training"
            print(f"\n  EPOCH GUARD: {self.tripped} - stopping the run")
            trainer.stop = True

    def on_fit_epoch_end(self, trainer) -> None:
        # Fires after validation and saving, so this is the whole epoch.
        wall = time.time() - self.epoch_start
        m50 = trainer.metrics.get("metrics/mAP50(B)", float("nan"))
        note = ""
        if wall > EPOCH_LIMIT_S:
            note = "OVER LIMIT"
            if not self.tripped:
                self.tripped = f"epoch {self.epoch} took {wall / 60:.1f} min (incl. validation)"
                print(f"\n  EPOCH GUARD: {self.tripped} - stopping the run")
            trainer.stop = True
        self._log(f"{self.epoch},{wall:.0f},{wall / 60:.1f},{m50:.4f},{note}")
        self.epoch_start = time.time()   # covers the gap until the next epoch starts

    def on_train_end(self, trainer) -> None:
        self._done.set()

    def _watchdog(self) -> None:
        while not self._done.wait(30):
            if self.epoch_start and time.time() - self.epoch_start > WATCHDOG_S:
                self._log(f"{self.epoch},,,,"
                          f"KILLED by watchdog: no epoch end after {WATCHDOG_S // 60} min")
                print(f"\n  WATCHDOG: epoch {self.epoch} stuck past {WATCHDOG_S // 60} min - "
                      "killing the process", flush=True)
                os._exit(3)


def train(device: str) -> int:
    from ultralytics import YOLO

    if OUT.exists() and any(OUT.iterdir()):
        print(f"  {OUT} already has files; refusing to overwrite a finished run.")
        return 1
    yaml = Y.write_split_files()      # same grouped split as run 1 and MaxViT
    OUT.mkdir(parents=True, exist_ok=True)
    guard = EpochGuard()
    model = YOLO(str(START))
    for event in ("on_train_start", "on_train_epoch_start", "on_train_batch_end",
                  "on_fit_epoch_end", "on_train_end"):
        model.add_callback(event, getattr(guard, event))
    model.train(data=str(yaml), epochs=EPOCHS, patience=10,
                imgsz=Y.IMGSZ, batch=Y.BATCH, device=device, workers=4, seed=Y.T.SEED,
                deterministic=False, project=str(OUT.parent.resolve()),
                name=OUT.name, exist_ok=True, plots=True, verbose=True)
    (OUT / "guard.json").write_text(json.dumps(
        {"tripped": guard.tripped, "epoch_limit_s": EPOCH_LIMIT_S}, indent=1))
    if guard.tripped:
        print(f"\n  stopped by the epoch guard: {guard.tripped}")
        print("  not continuing; validation is not run on a cut-short epoch.")
        return 2
    return Y.validate(device, out=OUT, hyper=HYPER)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--validate", action="store_true",
                   help="only redo the image-level validation step")
    p.add_argument("--device", default="mps")
    a = p.parse_args()
    if a.validate:
        return Y.validate(a.device, out=OUT, hyper=HYPER)
    return train(a.device)


if __name__ == "__main__":
    raise SystemExit(main())
