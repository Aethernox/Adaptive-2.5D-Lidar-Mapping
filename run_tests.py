"""
Minimal test runner used because this sandbox has no network access to
`pip install pytest`. Discovers every `test_*` function in tests/test_*.py
and runs it, printing PASS/FAIL per test. The test files are written in
plain pytest style (bare functions + assert), so they will also run
unmodified under real pytest wherever that's available:
    pip install pytest && pytest tests/ -v
"""
import importlib
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")


def discover_test_modules():
    for fname in sorted(os.listdir(TEST_DIR)):
        if fname.startswith("test_") and fname.endswith(".py"):
            yield fname[:-3]


def main():
    total, failed = 0, 0
    for modname in discover_test_modules():
        mod = importlib.import_module(f"tests.{modname}")
        for name in dir(mod):
            if name.startswith("test_") and callable(getattr(mod, name)):
                total += 1
                fn = getattr(mod, name)
                try:
                    fn()
                    print(f"  PASS  {modname}.{name}")
                except Exception:
                    failed += 1
                    print(f"  FAIL  {modname}.{name}")
                    traceback.print_exc()
    print(f"\n{total - failed}/{total} tests passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
