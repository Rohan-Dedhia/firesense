"""
FireSense - turning recorded sessions into training data.

A session on disk is a folder; a model wants fixed-length windows of features
with a label. This module is the bridge, and it makes three decisions that
matter more than the model architecture does:

1. SPLIT BY SESSION, NEVER BY WINDOW.
   Windows from one recording are hugely correlated. Shuffle them into train
   and test and the model effectively sees the test set during training - your
   accuracy looks superb and means nothing. split_by_session() is the only
   split function here, deliberately.

2. RELATIVE FEATURES, NOT RAW ppm.
   An MQ2's absolute reading depends on its uncalibrated R0 and the room's
   baseline. A model trained on raw ppm learns one sensor in one room. So every
   feature is expressed against the session's own quiet baseline - exactly the
   reasoning behind relative mode in fusion.py. This is also what makes
   training across different datasets and different hardware possible at all.

3. RATES OF CHANGE ARE FIRST-CLASS.
   Level alone cannot separate cooking from a leak; the correlation between
   gas and humidity slopes can. The features carry both.

    from dataset import load_sessions, split_by_session, make_windows

    sessions = load_sessions("data/sessions")
    train, test = split_by_session(sessions, test_frac=0.3, seed=7)
    X, y, meta = make_windows(train)
"""
from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

# The labels the recorder writes, and how they map to the thing we care about.
NUISANCE = {"normal", "cooking", "steam", "incense", "spray", "dust"}
DANGER = {"candle", "smoulder", "fire", "gas"}

SENSOR_COLUMNS = ("gas_ppm", "temp_c", "humidity_pct")

# Window geometry. 30 s at ~1 Hz gives the slopes something to fit, and is
# short enough that a developing event is several windows rather than one.
WINDOW_S = 30.0
STRIDE_S = 10.0
MIN_ROWS_PER_WINDOW = 8

# The baseline is the quiet head of the session - recorded before the event
# starts, which is why the runbook insists on 30 seconds of nothing first.
BASELINE_S = 20.0


# ---------------------------------------------------------------------------
# one session on disk
# ---------------------------------------------------------------------------
@dataclass
class Session:
    path: Path
    label: str
    meta: dict
    t: np.ndarray                     # wall clock, seconds
    gas: np.ndarray
    temp: np.ndarray
    hum: np.ndarray
    flame: np.ndarray                 # bool
    frame_ms: list = field(default_factory=list)   # sorted, may be empty

    # -- identity ---------------------------------------------------------
    @property
    def name(self) -> str:
        return self.path.name

    @property
    def is_danger(self) -> bool:
        return self.label in DANGER

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    @property
    def n_rows(self) -> int:
        return len(self.t)

    # -- baseline ---------------------------------------------------------
    def baseline(self) -> tuple[float, float, float]:
        """
        This room's quiet level, from the head of the session. Median rather
        than mean so one early spike cannot move it.
        """
        if self.n_rows == 0:
            return 0.0, 0.0, 0.0
        head = self.t <= (self.t[0] + BASELINE_S)
        if head.sum() < 3:                     # very short session
            head = np.arange(self.n_rows) < max(3, self.n_rows // 4)
        return (float(np.median(self.gas[head])),
                float(np.median(self.temp[head])),
                float(np.median(self.hum[head])))

    def frame_for(self, t_wall: float) -> Optional[Path]:
        """The frame captured nearest this moment, if any were captured."""
        if not self.frame_ms:
            return None
        want_ms = (t_wall - self.t[0]) * 1000.0
        best = min(self.frame_ms, key=lambda ms: abs(ms - want_ms))
        return self.path / "frames" / f"{best:08d}.jpg"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def _to_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _to_bool(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes", "y")


def load_session(folder: str | Path) -> Optional[Session]:
    """Read one session folder. Returns None if it is unusable, with a reason
    printed - a silently dropped session is a bug you find three weeks later."""
    folder = Path(folder)
    meta_p, csv_p = folder / "meta.json", folder / "sensors.csv"
    if not meta_p.exists() or not csv_p.exists():
        return None

    meta = json.loads(meta_p.read_text())
    label = meta.get("label", "unknown")

    t, gas, temp, hum, flame = [], [], [], [], []
    with csv_p.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if "t" not in row or row["t"] in (None, ""):
                continue
            t.append(_to_float(row["t"]))
            gas.append(_to_float(row.get("gas_ppm")))
            temp.append(_to_float(row.get("temp_c")))
            hum.append(_to_float(row.get("humidity_pct")))
            flame.append(_to_bool(row.get("flame_detected")))

    if len(t) < MIN_ROWS_PER_WINDOW:
        print(f"  skipping {folder.name}: only {len(t)} readings")
        return None

    frames_dir = folder / "frames"
    frame_ms = []
    if frames_dir.is_dir():
        for f in frames_dir.glob("*.jpg"):
            try:
                frame_ms.append(int(f.stem))
            except ValueError:
                pass
        frame_ms.sort()

    order = np.argsort(np.asarray(t))          # never trust file order
    return Session(
        path=folder, label=label, meta=meta,
        t=np.asarray(t, float)[order],
        gas=np.asarray(gas, float)[order],
        temp=np.asarray(temp, float)[order],
        hum=np.asarray(hum, float)[order],
        flame=np.asarray(flame, bool)[order],
        frame_ms=frame_ms,
    )


def load_sessions(root: str | Path = "data/sessions") -> list[Session]:
    root = Path(root)
    if not root.is_dir():
        print(f"no session folder at {root}")
        return []
    out = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        s = load_session(folder)
        if s is not None:
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# splitting - by session, always
# ---------------------------------------------------------------------------
def split_by_session(sessions: list[Session], test_frac: float = 0.3,
                     seed: int = 7) -> tuple[list[Session], list[Session]]:
    """
    Hold out whole sessions, stratified by label so both sides see every class.

    There is deliberately no split_by_window() in this module. If you want one,
    read the note at the top of the file first.
    """
    rng = random.Random(seed)
    by_label: dict[str, list[Session]] = {}
    for s in sessions:
        by_label.setdefault(s.label, []).append(s)

    train, test = [], []
    for label, group in sorted(by_label.items()):
        group = group[:]
        rng.shuffle(group)
        n_test = max(1, round(len(group) * test_frac)) if len(group) > 1 else 0
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    return train, test


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def _slope(xs: np.ndarray, ys: np.ndarray) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = xs.mean(), ys.mean()
    denom = float(((xs - mx) ** 2).sum())
    if denom < 1e-9:
        return 0.0
    return float(((xs - mx) * (ys - my)).sum() / denom)


FEATURE_NAMES = (
    "gas_rel",          # gas as a multiple of this session's baseline
    "gas_delta",        # gas above baseline, in ppm
    "gas_slope",        # ppm / s
    "gas_slope_rel",    # slope as a fraction of baseline per second
    "gas_std",          # roughness over the window
    "temp_delta",       # degrees above baseline
    "temp_slope",       # C / s
    "hum_delta",        # %RH above baseline
    "hum_slope",        # %RH / s
    "gas_hum_agree",    # +1 both rising, -1 gas up while humidity falls
    "flame_frac",       # fraction of the window the IR sensor fired
)


def window_features(s: Session, i0: int, i1: int) -> np.ndarray:
    """
    Features for rows [i0, i1). Everything is relative to the session's own
    baseline, so a different MQ2 or a different room produces comparable
    numbers - which is what lets one model work on hardware it never saw.
    """
    gb, tb, hb = s.baseline()
    t = s.t[i0:i1] - s.t[i0]
    gas, temp, hum = s.gas[i0:i1], s.temp[i0:i1], s.hum[i0:i1]

    gas_slope = _slope(t, gas)
    hum_slope = _slope(t, hum)
    denom = max(gb, 1.0)

    # The cooking-versus-leak discriminator, as one number: do gas and
    # humidity move together, or in opposition?
    if abs(gas_slope) < 1e-6:
        agree = 0.0
    else:
        agree = math.copysign(1.0, gas_slope) * math.copysign(1.0, hum_slope) \
            if abs(hum_slope) > 1e-6 else 0.0
        agree *= min(1.0, abs(hum_slope) / 0.05)      # weak signal, weak vote

    return np.array([
        float(np.median(gas) / denom),
        float(np.median(gas) - gb),
        gas_slope,
        gas_slope / denom,
        float(gas.std()),
        float(np.median(temp) - tb),
        _slope(t, temp),
        float(np.median(hum) - hb),
        hum_slope,
        agree,
        float(s.flame[i0:i1].mean()),
    ], dtype=np.float32)


def iter_windows(s: Session, window_s: float = WINDOW_S,
                 stride_s: float = STRIDE_S) -> Iterator[tuple[int, int, float]]:
    """Yield (start_index, end_index, window_end_walltime)."""
    if s.n_rows < MIN_ROWS_PER_WINDOW:
        return
    start_t = s.t[0]
    while start_t + window_s <= s.t[-1]:
        i0 = int(np.searchsorted(s.t, start_t, "left"))
        i1 = int(np.searchsorted(s.t, start_t + window_s, "right"))
        if i1 - i0 >= MIN_ROWS_PER_WINDOW:
            yield i0, i1, float(s.t[i1 - 1])
        start_t += stride_s


def make_windows(sessions: list[Session], window_s: float = WINDOW_S,
                 stride_s: float = STRIDE_S):
    """
    -> X  (n, n_features) float32
       y  (n,) int64       1 = danger, 0 = nuisance
       meta  list of dicts: session name, label, time, and the frame path

    The frame path is carried rather than the image, so the vision model can
    be run later, in batches, without loading every JPEG into memory here.
    """
    X, y, meta = [], [], []
    for s in sessions:
        for i0, i1, t_end in iter_windows(s, window_s, stride_s):
            X.append(window_features(s, i0, i1))
            y.append(1 if s.is_danger else 0)
            frame = s.frame_for(t_end)
            meta.append({
                "session": s.name,
                "label": s.label,
                "t_end": t_end,
                "frame": str(frame) if frame else None,
            })
    if not X:
        return (np.zeros((0, len(FEATURE_NAMES)), np.float32),
                np.zeros((0,), np.int64), [])
    return np.stack(X), np.asarray(y, np.int64), meta


# ---------------------------------------------------------------------------
# summary:  python dataset.py [folder]
# ---------------------------------------------------------------------------
def summarise(root: str | Path = "data/sessions") -> None:   # pragma: no cover
    sessions = load_sessions(root)
    if not sessions:
        print("No sessions found. Record one:  python record_session.py normal")
        return

    print(f"\n{len(sessions)} sessions in {root}\n")
    print(f"  {'label':10s} {'n':>3s}  {'minutes':>8s}  {'readings':>9s}  {'frames':>7s}")
    print("  " + "-" * 46)
    by_label: dict[str, list[Session]] = {}
    for s in sessions:
        by_label.setdefault(s.label, []).append(s)
    for label in sorted(by_label):
        g = by_label[label]
        mins = sum(x.duration_s for x in g) / 60.0
        print(f"  {label:10s} {len(g):3d}  {mins:8.1f}  "
              f"{sum(x.n_rows for x in g):9d}  {sum(len(x.frame_ms) for x in g):7d}")

    n_danger = sum(1 for s in sessions if s.is_danger)
    print("  " + "-" * 46)
    print(f"  danger {n_danger}   nuisance {len(sessions) - n_danger}")

    train, test = split_by_session(sessions, test_frac=0.3)
    Xtr, ytr, _ = make_windows(train)
    Xte, yte, _ = make_windows(test)
    print(f"\n  split by session (never by window):")
    print(f"    train {len(train):2d} sessions -> {len(Xtr):5d} windows  "
          f"({int(ytr.sum())} danger)")
    print(f"    test  {len(test):2d} sessions -> {len(Xte):5d} windows  "
          f"({int(yte.sum())} danger)")

    missing = [lbl for lbl in (NUISANCE | DANGER) if lbl not in by_label]
    if missing:
        print(f"\n  not yet recorded: {', '.join(sorted(missing))}")
    if len(sessions) < 12:
        print(f"  {12 - len(sessions)} more sessions to reach a trainable set.")
    print()


if __name__ == "__main__":                                   # pragma: no cover
    import sys
    summarise(sys.argv[1] if len(sys.argv) > 1 else "data/sessions")
