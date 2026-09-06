"""
Minimal test runner - use this if pytest isn't installed.
    python run_tests.py
(`python -m pytest tests -q` works too and gives nicer output.)
"""
import sys, traceback, importlib.util, pathlib

spec = importlib.util.spec_from_file_location(
    "test_fusion", pathlib.Path(__file__).parent / "tests" / "test_fusion.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

tests = [(n, getattr(mod, n)) for n in dir(mod) if n.startswith("test_")]
passed, failed = 0, []
for name, fn in tests:
    try:
        fn()
        print(f"  PASS  {name}")
        passed += 1
    except Exception as exc:
        print(f"  FAIL  {name}: {exc}")
        failed.append((name, traceback.format_exc()))
print(f"\n{passed}/{len(tests)} passed")
for name, tb in failed:
    print(f"\n--- {name} ---\n{tb}")
sys.exit(1 if failed else 0)
