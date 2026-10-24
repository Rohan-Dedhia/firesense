"""
FireSense - fine-tune YOLOv8n on D-Fire and score it like the MaxViT model.

    python train_yolo.py                 # train (2-hour cap), then validation
    python train_yolo.py --test          # ONCE, at the very end

The detector is trained on the SAME grouped train/validation split as
train_vision.py (read from models/maxvit_dfire/split.json), so the two
models are compared on equal terms. The D-Fire test split is not seen until
--test.

To compare a detector with a classifier, the detector is reduced to image
level: an image's fire score is its most confident fire box (0 if none), and
likewise for smoke. Thresholds on those scores are chosen on validation with
the same rule as MaxViT (per-output max F1), and the same metrics follow.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import train_vision as T

OUT = Path("models") / "yolo_dfire"
SPLIT_DIR = T.DATA / "yolo_split"
TIME_CAP_H = 2.0
IMGSZ = 640       # YOLOv8 default; D-Fire images are mostly 1280x720
BATCH = 16
SCORE_CONF = 0.01  # keep nearly every box so the score sweep has range


def write_split_files() -> Path:
    """Image-list files and a dataset yaml for ultralytics, from split.json."""
    split = json.loads((Path("models") / "maxvit_dfire" / "split.json").read_text())
    val = set(split["val_names"])
    img_dir = T.DATA / "extracted" / "train" / "images"
    names = sorted(p.name for p in img_dir.glob("*.jpg"))
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    (SPLIT_DIR / "train.txt").write_text(
        "\n".join(str(img_dir / n) for n in names if n not in val) + "\n")
    (SPLIT_DIR / "val.txt").write_text(
        "\n".join(str(img_dir / n) for n in names if n in val) + "\n")
    yaml = SPLIT_DIR / "dfire.yaml"
    # Labels are found by ultralytics by swapping /images/ for /labels/.
    yaml.write_text(
        f"path: {SPLIT_DIR}\ntrain: {SPLIT_DIR / 'train.txt'}\n"
        f"val: {SPLIT_DIR / 'val.txt'}\nnames:\n  0: smoke\n  1: fire\n")
    return yaml


def image_scores(model, paths: list[Path], device: str) -> np.ndarray:
    """(n, 2) fire, smoke: the most confident box of each class per image."""
    ids = {"smoke": 0, "fire": 1}
    out = []
    for i in range(0, len(paths), 32):
        res = model.predict([str(p) for p in paths[i:i + 32]], conf=SCORE_CONF,
                            imgsz=IMGSZ, device=device, verbose=False)
        for r in res:
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            out.append([float(conf[cls == ids[o]].max()) if (cls == ids[o]).any()
                        else 0.0 for o in T.OUTPUTS])
    return np.array(out, dtype=np.float32)


def labels_for(split: str, names: list[str]) -> np.ndarray:
    lab = {n: (f, s) for n, f, s in T.list_split(split)}
    return np.array([lab[n] for n in names], dtype=np.float32)


def train(device: str) -> int:
    from ultralytics import YOLO

    yaml = write_split_files()
    model = YOLO("yolov8n.pt")   # COCO-pretrained; ultralytics downloads it once
    model.train(data=str(yaml), epochs=100, time=TIME_CAP_H, patience=10,
                imgsz=IMGSZ, batch=BATCH, device=device, workers=4, seed=T.SEED,
                deterministic=False, project=str(OUT.parent.resolve()),
                name=OUT.name, exist_ok=True, plots=True, verbose=True)
    return validate(device)


def validate(device: str, out: Path = OUT, hyper: dict | None = None) -> int:
    """Image-level validation of out/weights/best.pt; `hyper` overrides the
    recorded hyperparameters (train_yolo_v2.py uses a different schedule)."""
    from ultralytics import YOLO

    model = YOLO(str(out / "weights" / "best.pt"))
    split = json.loads((Path("models") / "maxvit_dfire" / "split.json").read_text())
    names = sorted(split["val_names"])
    paths = [T.DATA / "extracted" / "train" / "images" / n for n in names]
    p = image_scores(model, paths, device)
    y = labels_for("train", names)
    thr = {o: T.best_f1_threshold(y[:, k], p[:, k]) for k, o in enumerate(T.OUTPUTS)}
    m = T.image_metrics(y, p, thr)
    T.print_metrics("YOLO VALIDATION, image level (thresholds chosen here)", m)
    np.save(out / "val_scores.npy", p)
    hp = {"model": "yolov8n.pt (COCO-pretrained)", "imgsz": IMGSZ,
          "batch": BATCH, "time_cap_hours": TIME_CAP_H,
          "epochs_requested": 100, "patience": 10,
          "optimizer": "ultralytics default (auto)",
          "image_score": "max box confidence per class",
          "box_conf_floor": SCORE_CONF}
    hp.update(hyper or {})
    (out / "val_results.json").write_text(json.dumps(
        {"thresholds": thr, "metrics": m, "hyperparameters": hp}, indent=1))
    return 0


def test(device: str) -> int:
    from ultralytics import YOLO

    done = OUT / "test_results.json"
    if done.exists():
        print(f"  {done} exists: the test set has been used. Refusing to re-run.")
        return 1
    thr = json.loads((OUT / "val_results.json").read_text())["thresholds"]
    model = YOLO(str(OUT / "weights" / "best.pt"))
    names = [n for n, _, _ in T.list_split("test")]
    paths = [T.DATA / "extracted" / "test" / "images" / n for n in names]
    p = image_scores(model, paths, device)
    y = labels_for("test", names)
    np.save(OUT / "test_scores.npy", p)
    clean = set(json.loads((T.DATA / "extracted" / "audit.json").read_text())
                ["test_without_near_dup"])
    ci = np.array([i for i, n in enumerate(names) if n in clean])
    res = {"thresholds_from_validation": thr,
           "official_test": T.image_metrics(y, p, thr),
           "test_without_near_dup_in_train": T.image_metrics(y[ci], p[ci], thr)}
    T.print_metrics("YOLO TEST, official D-Fire split", res["official_test"])
    T.print_metrics("YOLO TEST, images with no near-duplicate in train",
                    res["test_without_near_dup_in_train"])
    done.write_text(json.dumps(res, indent=1))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", action="store_true", help="evaluate the test set ONCE")
    p.add_argument("--validate", action="store_true",
                   help="only redo the image-level validation step")
    p.add_argument("--device", default="mps")
    a = p.parse_args()
    if a.test:
        return test(a.device)
    if a.validate:
        return validate(a.device)
    return train(a.device)


if __name__ == "__main__":
    raise SystemExit(main())
