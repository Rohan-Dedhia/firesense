"""
FireSense - train the learned fusion model, and measure it against the
hand-tuned rule engine it is supposed to replace.

    python train.py                  # train, evaluate, compare
    python train.py --cv             # grouped cross-validation (small datasets)
    python train.py --augment 6      # 6x the training set, test set untouched
    python train.py --folder path/to/sessions

Two things this file insists on, because they are what separate a real result
from a flattering one:

  GROUPED EVALUATION. Every split groups by session. With fifteen recordings a
  single train/test split is mostly noise, so --cv runs GroupKFold and reports
  the spread as well as the mean. A number without a spread, on this much data,
  is not a number.

  A HONEST BASELINE. The comparison is against fusion.FusionEngine run live
  over the same held-out sessions - calibrating from the session head exactly
  as it would on the bench, not a strawman threshold. Beating a weak baseline
  proves nothing; the rule engine passes 34 tests and was tuned against real
  hardware, so beating it means something.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import augment as AUG
import dataset as D

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN = True
except Exception as exc:                                    # pragma: no cover
    SKLEARN = False
    print(f"scikit-learn is required: {exc}\n  pip install scikit-learn")

MODELS_DIR = Path("models")


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Recall matters more than precision here: a missed fire is not the same
    kind of mistake as a false alarm, and the report should not average them
    into one number that hides which one you are making."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1,
            "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
            "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0}


def fmt(s: dict) -> str:
    return (f"precision {s['precision']:.2f}   recall {s['recall']:.2f}   "
            f"F1 {s['f1']:.2f}   false-alarm rate {s['false_alarm_rate']:.2f}")


# ---------------------------------------------------------------------------
# the baseline: the existing rule engine, run as it runs live
# ---------------------------------------------------------------------------
def rule_engine_verdicts(sessions: list[D.Session]) -> tuple[np.ndarray, np.ndarray]:
    """
    Replay each session through FusionEngine and ask the question the product
    actually asks: did this session raise an alarm at any point?

    -> (y_true, y_pred) at SESSION level, one entry per session.
    """
    import fusion

    y_true, y_pred = [], []
    for s in sessions:
        eng = fusion.FusionEngine(s.meta.get("environment", "Kitchen"))
        alarmed = False
        for i in range(s.n_rows):
            reading = {
                "timestamp": "",
                "gas_ppm": float(s.gas[i]),
                "temp_c": float(s.temp[i]),
                "humidity_pct": float(s.hum[i]),
                "flame_detected": bool(s.flame[i]),
            }
            res = eng.update(reading, now=float(s.t[i]))
            if res.level != "NORMAL":
                alarmed = True
        y_true.append(1 if s.is_danger else 0)
        y_pred.append(1 if alarmed else 0)
    return np.asarray(y_true), np.asarray(y_pred)


def model_session_verdicts(model, sessions: list[D.Session],
                           min_windows: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Same question, asked of the learned model. A session counts as alarmed once
    `min_windows` windows are predicted dangerous - the single-window
    equivalent of the debounce the rule engine already applies to the flame
    sensor, and for the same reason.
    """
    y_true, y_pred = [], []
    for s in sessions:
        X, _, _ = D.make_windows([s])
        if len(X) == 0:
            continue
        p = model.predict(X)
        y_true.append(1 if s.is_danger else 0)
        y_pred.append(1 if int(p.sum()) >= min_windows else 0)
    return np.asarray(y_true), np.asarray(y_pred)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def candidates() -> dict:
    return {
        "logistic": make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=2000,
                                                     class_weight="balanced")),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=7, n_jobs=-1),
        "gradient_boost": GradientBoostingClassifier(random_state=7),
    }


def cross_validate(sessions: list[D.Session], n_splits: int = 4) -> dict:
    """GroupKFold over sessions. The groups are what stop windows from one
    recording appearing on both sides of a fold."""
    X, y, meta = D.make_windows(sessions)
    if len(X) == 0:
        return {}
    groups = np.array([m["session"] for m in meta])
    n_groups = len(set(groups))
    n_splits = max(2, min(n_splits, n_groups))

    out = {}
    for name, mk in candidates().items():
        fold_f1, fold_rec, fold_far = [], [], []
        for tr_i, te_i in GroupKFold(n_splits=n_splits).split(X, y, groups):
            if len(set(y[tr_i])) < 2 or len(set(y[te_i])) < 2:
                continue
            model = candidates()[name]
            model.fit(X[tr_i], y[tr_i])
            s = scores(y[te_i], model.predict(X[te_i]))
            fold_f1.append(s["f1"])
            fold_rec.append(s["recall"])
            fold_far.append(s["false_alarm_rate"])
        if fold_f1:
            out[name] = {
                "f1_mean": float(np.mean(fold_f1)), "f1_std": float(np.std(fold_f1)),
                "recall_mean": float(np.mean(fold_rec)),
                "far_mean": float(np.mean(fold_far)),
                "folds": len(fold_f1),
            }
    return out


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def per_label_errors(model, sessions: list[D.Session]) -> dict:
    """Which kinds of event does it get wrong? A model that fails only on
    `spray` needs different work from one that fails on `smoulder`."""
    out: dict[str, dict] = {}
    for s in sessions:
        X, y, _ = D.make_windows([s])
        if len(X) == 0:
            continue
        pred = model.predict(X)
        want = 1 if s.is_danger else 0
        d = out.setdefault(s.label, {"windows": 0, "wrong": 0, "sessions": 0})
        d["windows"] += len(pred)
        d["wrong"] += int((pred != want).sum())
        d["sessions"] += 1
    for d in out.values():
        d["error_rate"] = d["wrong"] / max(1, d["windows"])
    return out


def report(folder: str, do_cv: bool, seed: int, augment: int = 1) -> int:
    sessions = D.load_sessions(folder)
    if len(sessions) < 4:
        print(f"\nOnly {len(sessions)} sessions in {folder}.")
        print("Record a few more first:  python record_session.py normal")
        return 1

    n_danger = sum(1 for s in sessions if s.is_danger)
    print(f"\n{'=' * 62}")
    print(f"  {len(sessions)} sessions   {n_danger} danger / "
          f"{len(sessions) - n_danger} nuisance")
    print(f"{'=' * 62}")

    if n_danger == 0 or n_danger == len(sessions):
        print("\n  Only one class present. Record some of the other kind before")
        print("  training - a model cannot learn a distinction it never sees.")
        return 1

    train, test = D.split_by_session(sessions, test_frac=0.3, seed=seed)

    # Split FIRST, then augment one side. The other order puts copies of the
    # same recording into both halves and every number below becomes fiction.
    if augment > 1:
        before = len(train)
        train = AUG.augment_sessions(train, factor=augment, seed=seed)
        print(f"\n  augmented the training split x{augment}: "
              f"{before} -> {len(train)} sessions (test set untouched)")

    Xtr, ytr, _ = D.make_windows(train)
    Xte, yte, _ = D.make_windows(test)
    print(f"\n  train {len(train):2d} sessions / {len(Xtr):5d} windows")
    print(f"  test  {len(test):2d} sessions / {len(Xte):5d} windows")

    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        print("\n  The split put all of one class on one side. More sessions,")
        print("  or a different --seed, will fix it.")
        return 1

    # -- window-level ------------------------------------------------------
    print(f"\n  WINDOW LEVEL (held-out sessions)")
    print("  " + "-" * 58)
    best_name, best_model, best_f1 = None, None, -1.0
    for name, model in candidates().items():
        model.fit(Xtr, ytr)
        s = scores(yte, model.predict(Xte))
        print(f"  {name:16s} {fmt(s)}")
        if s["f1"] > best_f1:
            best_name, best_model, best_f1 = name, model, s["f1"]
    print(f"\n  best: {best_name}")

    # -- cross-validation --------------------------------------------------
    if do_cv:
        print(f"\n  GROUPED CROSS-VALIDATION")
        print("  " + "-" * 58)
        cv = cross_validate(sessions)
        for name, r in sorted(cv.items(), key=lambda kv: -kv[1]["f1_mean"]):
            print(f"  {name:16s} F1 {r['f1_mean']:.2f} +/- {r['f1_std']:.2f}   "
                  f"recall {r['recall_mean']:.2f}   "
                  f"false-alarm {r['far_mean']:.2f}   ({r['folds']} folds)")
        if cv:
            spread = max(r["f1_std"] for r in cv.values())
            if spread > 0.15:
                print(f"\n  Fold spread is {spread:.2f} - that is large. These")
                print(f"  numbers are not yet stable; record more sessions before")
                print(f"  drawing conclusions from them.")

    # -- THE experiment: learned vs the rule engine ------------------------
    print(f"\n  SESSION LEVEL - learned model vs the existing rule engine")
    print("  " + "-" * 58)
    yt_r, yp_r = rule_engine_verdicts(test)
    yt_m, yp_m = model_session_verdicts(best_model, test)
    s_rule = scores(yt_r, yp_r)
    s_model = scores(yt_m, yp_m)
    print(f"  rule engine      {fmt(s_rule)}")
    print(f"  learned ({best_name:8s}) {fmt(s_model)}")

    gap = s_model["f1"] - s_rule["f1"]
    print()
    if gap > 0.02:
        print(f"  -> the learned model is ahead by {gap:+.2f} F1")
    elif gap < -0.02:
        print(f"  -> the rule engine is still ahead by {-gap:.2f} F1.")
        print(f"     That is a real result, not a failure - say so, and look at")
        print(f"     which labels it loses on below.")
    else:
        print(f"  -> level pegging ({gap:+.2f} F1). With this few sessions that")
        print(f"     is the honest reading; more data will separate them.")

    # -- what it gets wrong ------------------------------------------------
    print(f"\n  ERROR RATE BY LABEL (held-out)")
    print("  " + "-" * 58)
    for label, d in sorted(per_label_errors(best_model, test).items(),
                           key=lambda kv: -kv[1]["error_rate"]):
        bar = "#" * int(d["error_rate"] * 30)
        print(f"  {label:10s} {d['error_rate']:5.1%}  {bar}")

    # -- what it is using --------------------------------------------------
    importances = getattr(best_model, "feature_importances_", None)
    if importances is not None:
        print(f"\n  WHAT THE MODEL IS USING")
        print("  " + "-" * 58)
        for i in np.argsort(importances)[::-1][:6]:
            bar = "#" * int(importances[i] * 40)
            print(f"  {D.FEATURE_NAMES[i]:16s} {importances[i]:5.3f}  {bar}")

    # -- save --------------------------------------------------------------
    MODELS_DIR.mkdir(exist_ok=True)
    try:
        import joblib
        best_model.fit(*D.make_windows(sessions)[:2])      # refit on everything
        joblib.dump(best_model, MODELS_DIR / "fusion_model.joblib")
        (MODELS_DIR / "fusion_model.json").write_text(json.dumps({
            "model": best_name,
            "features": list(D.FEATURE_NAMES),
            "sessions": len(sessions),
            "window_s": D.WINDOW_S,
            "test_f1": s_model["f1"],
            "rule_engine_f1": s_rule["f1"],
        }, indent=2))
        print(f"\n  saved -> {MODELS_DIR / 'fusion_model.joblib'}")
    except Exception as exc:
        print(f"\n  could not save the model: {exc}")

    print()
    return 0


def main() -> None:
    if not SKLEARN:
        sys.exit(1)
    p = argparse.ArgumentParser(description="Train and evaluate the fusion model.")
    p.add_argument("--folder", default="data/sessions")
    p.add_argument("--cv", action="store_true",
                   help="grouped cross-validation - worth it under ~30 sessions")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--augment", type=int, default=1,
                   help="multiply the TRAINING split by this. 5-8 is sensible "
                        "under ~20 sessions; 1 disables it.")
    a = p.parse_args()
    sys.exit(report(a.folder, a.cv, a.seed, a.augment))


if __name__ == "__main__":
    main()
