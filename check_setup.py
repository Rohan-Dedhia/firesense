"""
FireSense - Day 1 environment check. Run this FIRST.

    python check_setup.py

Written for the Surface Pro 10 (Snapdragon X / Windows ARM64): the one real
risk on that machine is the Python + OpenCV install, and 60 seconds here beats
discovering it on Day 4.
"""
import importlib
import platform
import sys

print("=" * 62)
print("FireSense environment check")
print("=" * 62)
print(f"Python      : {sys.version.split()[0]}  ({platform.python_implementation()})")
print(f"Platform    : {platform.system()} {platform.release()}")
print(f"Machine     : {platform.machine()}")
if platform.machine().upper() in ("ARM64", "AARCH64"):
    print("              native ARM64 build - good (avoids x86 emulation)")
elif platform.system() == "Windows":
    print("              x86/x64 build - normal on an Intel/AMD machine. On an")
    print("              ARM Surface it runs under emulation; fine either way.")
print("-" * 62)

ok = True
for mod, why in [
    ("numpy", "maths"),
    ("cv2", "webcam + HSV detection  (pip install opencv-python)"),
    ("pandas", "dashboard tables/charts"),
    ("streamlit", "dashboard"),
]:
    try:
        m = importlib.import_module(mod)
        print(f"  OK    {mod:<12} {getattr(m, '__version__', '')}")
    except Exception as exc:
        ok = False
        print(f"  MISS  {mod:<12} needed for {why}  [{exc}]")

print("-" * 62)
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        got, frame = cap.read()
        print(f"  OK    camera index 0 -> frame {frame.shape if got else 'none'}")
        cap.release()
    else:
        ok = False
        print("  MISS  camera index 0 did not open. Try index 1, and check")
        print("        Settings > Privacy > Camera > let desktop apps use camera.")
except Exception as exc:
    print(f"  SKIP  camera test ({exc})")

print("-" * 62)
try:
    import fusion, mock_source          # noqa: F401
    eng = fusion.FusionEngine("Kitchen")
    r = eng.update({"gas_ppm": 120, "temp_c": 26, "humidity_pct": 55})
    print(f"  OK    fusion engine -> {r.level} {r.score}")
except Exception as exc:
    ok = False
    print(f"  FAIL  fusion engine: {exc}")

print("=" * 62)
print("READY - run `python run_tests.py`, then `streamlit run app.py`"
      if ok else "Fix the MISS/FAIL lines above, then re-run.")
