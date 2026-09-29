"""
Minimal test runner - use this if pytest isn't installed.

    python run_tests.py
    python run_tests.py fusion          # only tests/test_fusion.py

(`python -m pytest tests -q` works too and gives nicer output.)

Picks up every tests/test_*.py automatically, so a new test file needs no
change here.
"""
import importlib.util
import pathlib
import sys
import traceback

HERE = pathlib.Path(__file__).parent
only = sys.argv[1:] 

files = sorted((HERE / "tests").glob("test_*.py"))
if only:
    files = [f for f in files
             if any(o in f.stem for o in only)]
    if not files:
        print(f"no test file matched {only}")
        raise SystemExit(1)

passed, failed = 0, []
for path in files:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        print(f"  ERROR loading {path.name}")
        failed.append((path.name, traceback.format_exc()))
        continue
    names = [n for n in dir(mod) if n.startswith("test_")]
    if len(files) > 1:
        print(f"\n  {path.name}")
    for name in names:
        try:
            getattr(mod, name)()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {name}: {exc}")
            failed.append((name, traceback.format_exc()))

print(f"\n{passed}/{passed + len(failed)} passed")
for name, tb in failed:
    print(f"\n--- {name} ---\n{tb}")
sys.exit(1 if failed else 0)
