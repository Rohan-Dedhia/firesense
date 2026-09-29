"""
FireSense - physically motivated augmentation.

Fifteen recorded sessions is not many. Augmentation multiplies them, but only
if the transformations correspond to something that actually varies in the
world - otherwise you are training on noise you invented and measuring your
own imagination.

So every augmentation here has a physical justification:

    scale_gas        a different MQ2 unit. R0 varies between sensors by a
                     large factor and is not calibrated on either of ours -
                     which is exactly why fusion.py has a relative mode.
    shift_baseline   a different room. A kitchen idles higher than a bedroom.
    time_warp        a slower or faster event. A leak from a loose fitting
                     develops over minutes; a knocked-over canister does not.
                     Drawn from a fixed ladder, not sampled - see the note by
                     TIME_WARP_LADDER for why that matters.
    sensor_noise     the same event on a noisier day, or a slightly different
                     board.
    temp_offset      a warmer or cooler room.

Generic jitter would do none of that. These five say something true about
deployment, and that is the difference between augmentation you can defend in
a viva and augmentation that just makes the number go up.

    from augment import augment_sessions

    train, test = split_by_session(sessions, test_frac=0.3)
    train = augment_sessions(train, factor=6)      # test set untouched

THE RULE THAT MAKES THIS HONEST
-------------------------------
Augment the TRAINING set only. Never the test set. An augmented copy of a test
session is still that session, so putting one in training and one in test
leaks, and your reported score becomes meaningless. augment_sessions() is
built to be called on one side of a split that has already happened.
"""
from __future__ import annotations

import copy
import random
from dataclasses import replace
from typing import Optional

import numpy as np

from dataset import Session

# Ranges chosen to cover realistic variation without inventing physics.
GAS_SCALE_RANGE = (0.6, 1.8)        # MQ2 R0 spread between units
BASELINE_SHIFT_RANGE = (-40.0, 90.0)  # ppm - a quieter or dirtier room
# Time warping is drawn from a FIXED LADDER rather than sampled, and this is
# not fussiness. Warping changes a session's duration, which changes how many
# windows it yields. Sample it independently per session and the classes get
# silently reweighted - measured on this project's own data, the danger
# fraction drifted from 0.50 to 0.56 purely from augmentation. Cycling one
# ladder gives every session the identical set of warps, so the effect cancels
# exactly across classes.
TIME_WARP_LADDER = (0.75, 0.9, 1.0, 1.15, 1.3)
NOISE_SIGMA_RANGE = (0.0, 8.0)      # ppm of extra sensor noise
TEMP_OFFSET_RANGE = (-4.0, 6.0)     # degrees - a cooler or warmer room
HUM_OFFSET_RANGE = (-8.0, 10.0)     # %RH


def _clone(s: Session, suffix: str, **changes) -> Session:
    out = copy.copy(s)
    for k, v in changes.items():
        setattr(out, k, v)
    out.meta = dict(s.meta)
    out.meta["augmented_from"] = s.name
    out.meta["augmentation"] = suffix
    # Frames belong to the original recording; an augmented sensor trace no
    # longer lines up with them, so the copy carries none.
    out.frame_ms = []
    return out


# ---------------------------------------------------------------------------
# individual transformations
# ---------------------------------------------------------------------------
def scale_gas(s: Session, factor: float) -> Session:
    """A different MQ2. Multiplicative, because R0 scales the whole curve -
    baseline and excursion together - rather than offsetting it."""
    return _clone(s, f"gas_x{factor:.2f}", gas=np.maximum(0.0, s.gas * factor))


def shift_baseline(s: Session, delta_ppm: float) -> Session:
    """A different room. Additive: the idle level moves, the event on top of
    it does not change size."""
    return _clone(s, f"base{delta_ppm:+.0f}",
                  gas=np.maximum(0.0, s.gas + delta_ppm))


def time_warp(s: Session, factor: float) -> Session:
    """The same event, slower or faster. Timestamps stretch; the readings
    themselves are untouched, so every value remains one this sensor really
    produced."""
    if s.n_rows < 2 or factor <= 0:
        return _clone(s, "warp1.00")
    t0 = s.t[0]
    return _clone(s, f"warp{factor:.2f}", t=t0 + (s.t - t0) * factor)


def sensor_noise(s: Session, sigma_ppm: float, rng: np.random.Generator) -> Session:
    """A noisier board, or a draughtier room."""
    if sigma_ppm <= 0:
        return _clone(s, "noise0")
    noisy = s.gas + rng.normal(0.0, sigma_ppm, size=s.gas.shape)
    return _clone(s, f"noise{sigma_ppm:.0f}", gas=np.maximum(0.0, noisy))


def climate_offset(s: Session, d_temp: float, d_hum: float) -> Session:
    """A warmer or more humid room. Offsets, not scales - the trend within the
    session is the signal and must survive intact."""
    return _clone(s, f"clim{d_temp:+.1f}/{d_hum:+.1f}",
                  temp=s.temp + d_temp,
                  hum=np.clip(s.hum + d_hum, 0.0, 100.0))


# ---------------------------------------------------------------------------
# combining them
# ---------------------------------------------------------------------------
def augment_session(s: Session, rng: np.random.Generator,
                    index: int = 0) -> Session:
    """One randomly transformed copy. All five transformations are applied,
    each sampled independently, because in deployment they vary together: a
    different sensor in a different room on a different day."""
    gas_factor = rng.uniform(*GAS_SCALE_RANGE)
    base_delta = rng.uniform(*BASELINE_SHIFT_RANGE)
    warp = TIME_WARP_LADDER[index % len(TIME_WARP_LADDER)]   # see the note above
    sigma = rng.uniform(*NOISE_SIGMA_RANGE)
    d_temp = rng.uniform(*TEMP_OFFSET_RANGE)
    d_hum = rng.uniform(*HUM_OFFSET_RANGE)

    out = scale_gas(s, gas_factor)
    out = shift_baseline(out, base_delta)
    out = sensor_noise(out, sigma, rng)
    out = time_warp(out, warp)
    out = climate_offset(out, d_temp, d_hum)

    out.path = s.path
    out.label = s.label                     # the label is never touched
    out.meta = dict(s.meta)
    out.meta.update({
        "augmented_from": s.name,
        "augmentation": (f"gas x{gas_factor:.2f}, base {base_delta:+.0f} ppm, "
                         f"warp x{warp:.2f}, noise {sigma:.1f}, "
                         f"T {d_temp:+.1f}, RH {d_hum:+.1f}"),
        "augment_index": index,
    })
    out._aug_name = f"{s.name}__aug{index:02d}"
    return out


def augment_sessions(sessions: list[Session], factor: int = 5,
                     seed: int = 7, keep_originals: bool = True) -> list[Session]:
    """
    -> the originals plus (factor - 1) transformed copies of each.

    Call this on the TRAINING split only. Augmenting before you split, or
    augmenting the test set, leaks the same recording into both sides and
    every number you report afterwards is inflated.
    """
    if factor <= 1:
        return list(sessions) if keep_originals else []
    rng = np.random.default_rng(seed)
    out: list[Session] = list(sessions) if keep_originals else []
    for s in sessions:
        for i in range(factor - 1):
            out.append(augment_session(s, rng, index=i))
    return out


def augmented_name(s: Session) -> str:
    """The grouping key for cross-validation: an augmented copy must share its
    original's group, or GroupKFold will happily split them apart."""
    return s.meta.get("augmented_from") or s.name


# ---------------------------------------------------------------------------
# summary:  python augment.py [folder] [factor]
# ---------------------------------------------------------------------------
if __name__ == "__main__":                                   # pragma: no cover
    import sys
    from dataset import load_sessions, make_windows, split_by_session

    folder = sys.argv[1] if len(sys.argv) > 1 else "data/sessions"
    factor = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    sessions = load_sessions(folder)
    if not sessions:
        print("No sessions yet. Record one:  python record_session.py normal")
        raise SystemExit(0)

    train, test = split_by_session(sessions, test_frac=0.3)
    Xb, yb, _ = make_windows(train)
    aug = augment_sessions(train, factor=factor)
    Xa, ya, _ = make_windows(aug)
    Xt, _, _ = make_windows(test)

    print(f"\n  split by session first, then augment the training side only")
    print(f"  {'-' * 56}")
    print(f"  train sessions   {len(train):4d}  ->  {len(aug):4d} after x{factor}")
    print(f"  train windows    {len(Xb):4d}  ->  {len(Xa):4d}")
    print(f"  test windows     {len(Xt):4d}      (untouched, deliberately)")
    if len(yb):
        print(f"  danger fraction  {yb.mean():.2f}  ->  {ya.mean():.2f}"
              f"      (should barely move)")
    print(f"\n  example transformations:")
    for s in aug[len(train):len(train) + 3]:
        print(f"    {s.meta['augmented_from']:28s} {s.meta['augmentation']}")
    print()
