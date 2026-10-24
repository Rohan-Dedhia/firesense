"""
FireSense - train the camera channel: MaxViT-tiny, fire and smoke, on D-Fire.

    python dfire_audit.py                       # once: counts, leakage, groups.json
    python train_vision.py --cache              # once: decode every image to 224x224
    python train_vision.py                      # train (resumable: just run it again)
    python train_vision.py --test               # ONCE, at the very end

What the model is
-----------------
An ImageNet-pretrained MaxViT-tiny (timm) with two independent outputs,
fire and smoke (multi-label, sigmoid). "Neither" is both outputs below their
thresholds. Image-level labels come from the D-Fire YOLO box files: any fire
box makes the image fire, any smoke box makes it smoke.

How it is kept honest
---------------------
* The D-Fire test split is never read during training, model selection or
  threshold choice. `--test` evaluates it once and refuses to run twice.
* Validation is 15% of D-Fire train, split by GROUP, not by image. The groups
  (groups.json, from dfire_audit.py) join consecutive video frames and
  near-duplicate images, so near-copies never sit on both sides. Same idea as
  CLAUDE.md rule 3 for sensor sessions.
* The best epoch is chosen by validation mean average precision, which needs
  no threshold. Thresholds are then chosen on validation only.
* The test set is reported twice: the official split, and the subset of test
  images that have no near-duplicate in train (dfire_audit.py found 2,211 of
  4,306 do). The second number is the fairer guess for an unseen scene, and
  even it says nothing about a kitchen: D-Fire is mostly outdoor.

The number that matters for FireSense is the false-alarm rate on "neither"
images (any output firing on an image with no fire and no smoke). That is
the bacon problem in camera form.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DATA = Path.home() / "datasets" / "d-fire"
OUT = Path("models") / "maxvit_dfire"

MODEL_NAME = "maxvit_tiny_tf_224.in1k"   # timm id; ImageNet-1k pretrained
SIZE = 224                               # MaxViT-tiny's native input
OUTPUTS = ("fire", "smoke")              # column order everywhere
VAL_FRACTION = 0.15
SEED = 0

# Training settings. Ordinary fine-tuning values for a small ViT-family model,
# not tuned on test (nothing is). Batch 32 fits comfortably in 24 GB unified
# memory; larger batches gave no reason to risk swapping.
BATCH = 32
LR = 1e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 1
# Measured on the M5: ~1.3 s per batch of 32, so ~10 min per epoch. 15 epochs
# lets the cosine schedule finish inside the 3-hour cap instead of being cut.
MAX_EPOCHS = 15
PATIENCE = 4                 # epochs without a better val mAP before stopping
DROP_PATH = 0.1
WALL_CAP_S = 3 * 3600        # total training time across resumes


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def list_split(split: str) -> list[tuple[str, int, int]]:
    """[(filename, fire, smoke)] for one D-Fire split, sorted by name."""
    from dfire_audit import image_label

    root = DATA / "extracted" / split
    return [(p.name, *image_label(root / "labels" / (p.stem + ".txt")))
            for p in sorted((root / "images").glob("*.jpg"))]


def _load_resized(path: str) -> np.ndarray:
    from PIL import Image

    # Squash the whole frame to 224x224 rather than centre-cropping it: a
    # 16:9 crop would cut off the left and right edges, which is exactly
    # where a small fire can be. MaxViT tolerates the aspect change.
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC),
                          dtype=np.uint8)


def build_cache(workers: int) -> None:
    """Decode every image once into a uint8 array per split.

    Decoding 1280x720 JPEGs every epoch would make the data loader, not the
    GPU, the bottleneck. The cache is ~3.2 GB on disk, memory-mapped.
    """
    for split in ("train", "test"):
        items = list_split(split)
        out = DATA / f"cache{SIZE}_{split}.npy"
        if out.exists():
            print(f"  {out} exists, skipping")
            continue
        paths = [str(DATA / "extracted" / split / "images" / n) for n, _, _ in items]
        arr = np.lib.format.open_memmap(str(out) + ".part", mode="w+",
                                        dtype=np.uint8,
                                        shape=(len(paths), SIZE, SIZE, 3))
        t0 = time.time()
        with ProcessPoolExecutor(workers) as ex:
            for i, img in enumerate(ex.map(_load_resized, paths, chunksize=64)):
                arr[i] = img
                if i % 2000 == 0:
                    print(f"  {split}: {i}/{len(paths)}  {time.time() - t0:.0f}s")
        arr.flush()
        del arr
        Path(str(out) + ".part").rename(out)
        (DATA / f"cache{SIZE}_{split}.json").write_text(
            json.dumps([n for n, _, _ in items]))
        print(f"  wrote {out} ({len(paths)} images)")


def load_cache(split: str):
    names = json.loads((DATA / f"cache{SIZE}_{split}.json").read_text())
    arr = np.load(DATA / f"cache{SIZE}_{split}.npy", mmap_mode="r")
    labels = {n: (f, s) for n, f, s in list_split(split)}
    y = np.array([labels[n] for n in names], dtype=np.float32)
    return names, arr, y


def group_split(names: list[str], y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices (train, val): whole groups go to validation until 15% is reached."""
    groups = json.loads((DATA / "extracted" / "groups.json").read_text())["groups"]
    by_group: dict[int, list[int]] = {}
    for i, n in enumerate(names):
        by_group.setdefault(groups[n], []).append(i)
    order = sorted(by_group)
    random.Random(SEED).shuffle(order)
    target = VAL_FRACTION * len(names)
    val: list[int] = []
    for g in order:
        if len(val) >= target:
            break
        # Skip a group that would overshoot by a lot (the largest is ~1,000
        # frames of one camera) - it stays in train instead.
        if len(val) + len(by_group[g]) > target * 1.1:
            continue
        val.extend(by_group[g])
    val_set = set(val)
    tr = np.array([i for i in range(len(names)) if i not in val_set])
    return tr, np.array(sorted(val))


# ---------------------------------------------------------------------------
# metrics - shared with train_yolo.py and evaluate_on_frames.py
# ---------------------------------------------------------------------------
def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold that maximises F1 for one output. Chosen on validation only."""
    best_t, best_f = 0.5, -1.0
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        pred = p >= t
        tp = float((pred & (y == 1)).sum())
        fp = float((pred & (y == 0)).sum())
        fn = float((~pred & (y == 1)).sum())
        f = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t


def image_metrics(y: np.ndarray, p: np.ndarray, thr: dict[str, float]) -> dict:
    """Per-output precision/recall/F1/AP, the neither false-alarm rate, and a
    4x4 confusion matrix over {neither, fire only, smoke only, both}.

    y, p: arrays of shape (n, 2) in OUTPUTS order; p are scores in [0, 1].
    """
    from sklearn.metrics import average_precision_score

    out: dict = {"n": int(len(y)), "thresholds": thr}
    pred = np.stack([p[:, k] >= thr[o] for k, o in enumerate(OUTPUTS)], axis=1)
    for k, o in enumerate(OUTPUTS):
        yt, yp = y[:, k] == 1, pred[:, k]
        tp, fp, fn = int((yp & yt).sum()), int((yp & ~yt).sum()), int((~yp & yt).sum())
        out[o] = {
            "positives": int(yt.sum()),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
            "ap": float(average_precision_score(yt, p[:, k])) if yt.any() else None,
            "tp": tp, "fp": fp, "fn": fn,
        }
    neither = (y.sum(axis=1) == 0)
    alarms = pred.any(axis=1)
    out["neither_images"] = int(neither.sum())
    out["neither_false_alarms"] = int((alarms & neither).sum())
    out["neither_false_alarm_rate"] = (float((alarms & neither).sum() / neither.sum())
                                       if neither.any() else None)
    out["neither_fire_false_alarm_rate"] = (float((pred[:, 0] & neither).sum() / neither.sum())
                                            if neither.any() else None)
    # Detection: of images that DO have fire or smoke, how many raised any alarm.
    hazard = ~neither
    out["hazard_detection_rate"] = (float((alarms & hazard).sum() / hazard.sum())
                                    if hazard.any() else None)
    ti = (y[:, 0] + 2 * y[:, 1]).astype(int)
    pi = (pred[:, 0].astype(int) + 2 * pred[:, 1].astype(int))
    cm = np.zeros((4, 4), dtype=int)
    for a, b in zip(ti, pi):
        cm[a, b] += 1
    out["confusion"] = {"labels": ["neither", "fire only", "smoke only", "both"],
                        "rows_true_cols_pred": cm.tolist()}
    return out


def print_metrics(title: str, m: dict) -> None:
    print(f"\n  {title}  (n = {m['n']})")
    for o in OUTPUTS:
        r = m[o]
        pr = f"{r['precision']:.3f}" if r["precision"] is not None else "  -  "
        rc = f"{r['recall']:.3f}" if r["recall"] is not None else "  -  "
        ap = f"{r['ap']:.3f}" if r["ap"] is not None else "  -  "
        print(f"    {o:6s} thr {m['thresholds'][o]:.2f}  precision {pr}  recall {rc}"
              f"  F1 {r['f1']:.3f}  AP {ap}  (positives {r['positives']})")
    if m["neither_false_alarm_rate"] is not None:
        print(f"    false alarms on neither images: {m['neither_false_alarms']}/"
              f"{m['neither_images']} = {m['neither_false_alarm_rate']:.1%}"
              f"   (fire output alone: {m['neither_fire_false_alarm_rate']:.1%})")
    if m["hazard_detection_rate"] is not None:
        print(f"    fire-or-smoke images raising an alarm: {m['hazard_detection_rate']:.1%}")
    cm = m["confusion"]
    print("    confusion (rows true, cols predicted): " + ", ".join(cm["labels"]))
    for lab, row in zip(cm["labels"], cm["rows_true_cols_pred"]):
        print(f"      {lab:11s} " + " ".join(f"{v:6d}" for v in row))


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def make_model(pretrained: bool):
    import timm

    return timm.create_model(MODEL_NAME, pretrained=pretrained,
                             num_classes=len(OUTPUTS), drop_path_rate=DROP_PATH)


def _normalise(model):
    import torch

    cfg = model.pretrained_cfg
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
    return mean, std


def to_tensor(batch_u8: np.ndarray, mean, std, device):
    import torch

    x = torch.from_numpy(np.ascontiguousarray(batch_u8)).to(device)
    x = x.permute(0, 3, 1, 2).float().div_(255.0)
    return (x - mean.to(device)) / std.to(device)


def augment(batch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Horizontal flip and mild brightness/contrast. No hue shift (it would
    turn fire into something that is not fire) and no cropping (it can cut a
    small fire out while the label still says fire)."""
    b = batch.astype(np.float32)
    flip = rng.random(len(b)) < 0.5
    b[flip] = b[flip, :, ::-1]
    bright = rng.uniform(0.8, 1.2, (len(b), 1, 1, 1)).astype(np.float32)
    contrast = rng.uniform(0.8, 1.2, (len(b), 1, 1, 1)).astype(np.float32)
    mean = b.mean(axis=(1, 2, 3), keepdims=True)
    b = (b - mean) * contrast + mean * bright
    return np.clip(b, 0, 255).astype(np.uint8)


def predict(model, arr, idx: np.ndarray, device, batch: int = 64) -> np.ndarray:
    """Sigmoid scores (len(idx), 2) for images arr[idx], in idx order."""
    import torch

    mean, std = _normalise(model)
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            x = to_tensor(arr[idx[s:s + batch]], mean, std, device)
            out.append(torch.sigmoid(model(x)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(OUTPUTS)), np.float32)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def train(device_name: str) -> int:
    import torch
    from sklearn.metrics import average_precision_score

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    names, arr, y = load_cache("train")
    tr, va = group_split(names, y)
    split_info = {"train": len(tr), "val": len(va),
                  "val_class_counts": {
                      c: int(((y[va, 0] + 2 * y[va, 1]) == k).sum())
                      for k, c in enumerate(["neither", "fire only", "smoke only", "both"])},
                  "train_class_counts": {
                      c: int(((y[tr, 0] + 2 * y[tr, 1]) == k).sum())
                      for k, c in enumerate(["neither", "fire only", "smoke only", "both"])},
                  "val_names": [names[i] for i in va]}
    (OUT / "split.json").write_text(json.dumps(split_info))
    print(f"  train {len(tr)}  val {len(va)}  (grouped split, seed {SEED})")
    print(f"  val classes: {split_info['val_class_counts']}")

    ckpt_path = OUT / "last.pt"
    resumed = ckpt_path.exists()
    model = make_model(pretrained=not resumed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(tr) / BATCH)
    total = MAX_EPOCHS * steps_per_epoch
    warm = WARMUP_EPOCHS * steps_per_epoch

    def lr_at(step: int) -> float:
        if step < warm:
            return LR * (step + 1) / warm
        return LR * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    state = {"epoch": 0, "step": 0, "elapsed": 0.0, "best_map": -1.0,
             "best_epoch": -1, "bad_epochs": 0, "history": []}
    if resumed:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        print(f"  resumed from epoch {state['epoch']} "
              f"({state['elapsed'] / 60:.0f} min used of {WALL_CAP_S / 60:.0f})")

    mean, std = _normalise(model)
    mean, std = mean.to(device), std.to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    stop_reason = None

    while state["epoch"] < MAX_EPOCHS:
        if state["bad_epochs"] >= PATIENCE:
            stop_reason = f"early stop: no val mAP gain in {PATIENCE} epochs"
            break
        if state["elapsed"] >= WALL_CAP_S:
            stop_reason = "wall-time cap reached"
            break
        ep = state["epoch"]
        rng = np.random.default_rng(SEED + ep)
        perm = tr[rng.permutation(len(tr))]
        model.train()
        t0 = time.time()
        run_loss, seen = 0.0, 0
        capped = False
        for s in range(0, len(perm), BATCH):
            bi = np.sort(perm[s:s + BATCH])
            xb = to_tensor(augment(arr[bi], rng), mean, std, device)
            yb = torch.from_numpy(y[bi]).to(device)
            for g in opt.param_groups:
                g["lr"] = lr_at(state["step"])
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            state["step"] += 1
            run_loss += float(loss) * len(bi)
            seen += len(bi)
            if (s // BATCH) % 100 == 0:
                print(f"    epoch {ep + 1} batch {s // BATCH}/{steps_per_epoch} "
                      f"loss {run_loss / seen:.4f}  {time.time() - t0:.0f}s", flush=True)
            if state["elapsed"] + (time.time() - t0) >= WALL_CAP_S:
                capped = True
                break
        train_loss = run_loss / max(1, seen)

        pv = predict(model, arr, va, device)
        yv = y[va]
        val_loss = float(-np.mean(yv * np.log(np.clip(pv, 1e-7, 1))
                                  + (1 - yv) * np.log(np.clip(1 - pv, 1e-7, 1))))
        aps = [float(average_precision_score(yv[:, k], pv[:, k])) for k in range(len(OUTPUTS))]
        vmap = float(np.mean(aps))
        state["elapsed"] += time.time() - t0
        state["epoch"] = ep + 1
        rec = {"epoch": ep + 1, "train_loss": train_loss, "val_loss": val_loss,
               "val_ap_fire": aps[0], "val_ap_smoke": aps[1], "val_map": vmap,
               "lr_end": lr_at(state["step"] - 1), "minutes": state["elapsed"] / 60,
               "partial_epoch": capped}
        state["history"].append(rec)
        print(f"  epoch {ep + 1}: train loss {train_loss:.4f}  val loss {val_loss:.4f}  "
              f"val AP fire {aps[0]:.3f} smoke {aps[1]:.3f}  mAP {vmap:.3f}  "
              f"[{state['elapsed'] / 60:.0f} min]", flush=True)

        if vmap > state["best_map"]:
            state["best_map"], state["best_epoch"], state["bad_epochs"] = vmap, ep + 1, 0
            torch.save({"model": model.state_dict(), "model_name": MODEL_NAME,
                        "size": SIZE, "outputs": OUTPUTS, "epoch": ep + 1,
                        "val_map": vmap}, OUT / "best.pt")
            np.save(OUT / "val_scores_best.npy", pv)
        else:
            state["bad_epochs"] += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "state": state}, ckpt_path)
        (OUT / "history.json").write_text(json.dumps(state["history"], indent=1))
        if capped:
            stop_reason = "wall-time cap reached mid-epoch"
            break
    else:
        stop_reason = stop_reason or f"reached MAX_EPOCHS ({MAX_EPOCHS})"
    if stop_reason is None:
        stop_reason = f"reached MAX_EPOCHS ({MAX_EPOCHS})"
    print(f"\n  stopped: {stop_reason}. best epoch {state['best_epoch']} "
          f"(val mAP {state['best_map']:.3f})")

    finalize_validation(y[va], np.load(OUT / "val_scores_best.npy"), state, stop_reason)
    plot_curves(state["history"], OUT / "curves.png")
    return 0


def finalize_validation(yv: np.ndarray, pv: np.ndarray, state: dict, stop_reason: str) -> None:
    """Choose thresholds on validation (best epoch's scores) and save them."""
    thr = {o: best_f1_threshold(yv[:, k], pv[:, k]) for k, o in enumerate(OUTPUTS)}
    m = image_metrics(yv, pv, thr)
    print_metrics("VALIDATION (best epoch, thresholds chosen here)", m)
    (OUT / "val_results.json").write_text(json.dumps(
        {"thresholds": thr, "metrics": m, "best_epoch": state["best_epoch"],
         "best_val_map": state["best_map"], "stop_reason": stop_reason,
         "epochs_run": state["epoch"], "train_minutes": state["elapsed"] / 60,
         "hyperparameters": hyperparameters()}, indent=1))


def hyperparameters() -> dict:
    return {"model": MODEL_NAME, "input": f"{SIZE}x{SIZE} (whole frame resized)",
            "batch": BATCH, "optimizer": "AdamW", "lr": LR,
            "weight_decay": WEIGHT_DECAY, "schedule": f"{WARMUP_EPOCHS} epoch linear "
            "warm-up, then cosine to 0 over MAX_EPOCHS", "max_epochs": MAX_EPOCHS,
            "early_stop_patience": PATIENCE, "drop_path": DROP_PATH,
            "loss": "BCEWithLogits, 2 outputs", "augment": "hflip, brightness/contrast 0.8-1.2",
            "val_fraction": VAL_FRACTION, "seed": SEED,
            "selection": "best val mAP; thresholds = per-output max F1 on val",
            "wall_cap_hours": WALL_CAP_S / 3600}


def plot_curves(history: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(ep, [h["train_loss"] for h in history], marker="o", label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], marker="o", label="validation")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("BCE loss"); ax[0].legend()
    ax[0].set_title("Loss")
    ax[1].plot(ep, [h["val_ap_fire"] for h in history], marker="o", label="fire AP")
    ax[1].plot(ep, [h["val_ap_smoke"] for h in history], marker="o", label="smoke AP")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("average precision"); ax[1].legend()
    ax[1].set_title("Validation AP")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# test - once
# ---------------------------------------------------------------------------
def test(device_name: str) -> int:
    import torch

    done = OUT / "test_results.json"
    if done.exists():
        print(f"  {done} already exists: the test set has been used. Refusing to")
        print("  evaluate it again - a test set you look at twice is a validation set.")
        return 1
    val = json.loads((OUT / "val_results.json").read_text())
    thr = val["thresholds"]
    ck = torch.load(OUT / "best.pt", map_location="cpu", weights_only=False)
    model = make_model(pretrained=False)
    model.load_state_dict(ck["model"])
    device = torch.device(device_name)
    model.to(device)

    names, arr, y = load_cache("test")
    p = predict(model, arr, np.arange(len(names)), device)
    np.save(OUT / "test_scores.npy", p)
    audit = json.loads((DATA / "extracted" / "audit.json").read_text())
    clean = set(audit["test_without_near_dup"])
    ci = np.array([i for i, n in enumerate(names) if n in clean])

    res = {"thresholds_from_validation": thr, "best_epoch": ck["epoch"],
           "official_test": image_metrics(y, p, thr),
           "test_without_near_dup_in_train": image_metrics(y[ci], p[ci], thr)}
    print_metrics("TEST, official D-Fire split", res["official_test"])
    print_metrics("TEST, images with no near-duplicate in train",
                  res["test_without_near_dup_in_train"])
    done.write_text(json.dumps(res, indent=1))
    print(f"\n  wrote {done}")
    return 0


def main() -> int:
    global DATA, OUT
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--cache", action="store_true", help="build the 224px cache")
    p.add_argument("--test", action="store_true", help="evaluate the test set ONCE")
    p.add_argument("--device", default="mps")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    DATA, OUT = Path(a.data).expanduser(), Path(a.out)

    if a.cache:
        build_cache(a.workers)
        return 0
    if a.test:
        return test(a.device)
    return train(a.device)


if __name__ == "__main__":
    raise SystemExit(main())
