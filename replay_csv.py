"""Replay a logged session CSV back through the fusion engine (for tuning)."""
import csv, sys
from datetime import datetime
import config as C, fusion

def load(path):
    rows = list(csv.DictReader(open(path)))
    t0 = datetime.fromisoformat(rows[0]["timestamp"])
    out = []
    for r in rows:
        t = (datetime.fromisoformat(r["timestamp"]) - t0).total_seconds()
        out.append((t, {
            "gas_ppm": float(r["gas_raw"]),
            "temp_c": float(r["temp_c"]),
            "humidity_pct": float(r["humidity_pct"]),
            # older logs carry visual_fire; newer ones flame_sensor
            "flame_detected": (r.get("flame_sensor")
                               or r.get("visual_fire", "False")) == "True",
        }))
    return out

def run(path, env="Kitchen", verbose=False):
    data = load(path)
    eng = fusion.FusionEngine(env)
    res = []
    for t, reading in data:
        res.append((t, eng.update(reading, now=1_000_000 + t)))
    trans = sum(1 for a, b in zip(res, res[1:]) if a[1].level != b[1].level)
    bad_supp = [(t, r) for t, r in res if r.suppressed and r.gas_score > 70]
    first_fus = next((t for t, r in res if r.level != "NORMAL"), None)
    first_nai = next((t for t, r in res if r.naive_alarm), None)
    peak = max(r.score for _, r in res)
    if verbose:
        prev = "NORMAL"
        for t, r in res:
            if r.level != prev:
                print(f"   t+{t:6.0f}s {prev:>8} -> {r.level:<8} score {r.score:5.1f} "
                      f"sig {r.signature}")
                prev = r.level
    return dict(transitions=trans, suppressed_high_gas=len(bad_supp),
                first_fusion=first_fus, first_naive=first_nai, peak=peak,
                res=res)

if __name__ == "__main__":
    m = run(sys.argv[1] if len(sys.argv) > 1 else "real_session.csv",
            verbose="-v" in sys.argv)
    print(f"level changes            : {m['transitions']}")
    print(f"suppressed while gas high: {m['suppressed_high_gas']}")
    print(f"first fusion alarm       : t+{m['first_fusion']}s")
    print(f"first fixed-threshold    : t+{m['first_naive']}s")
    if m['first_fusion'] is not None and m['first_naive'] is not None:
        print(f"lead                     : {m['first_naive']-m['first_fusion']:.1f} s")
    print(f"peak score               : {m['peak']}")
