"""Tiny plain-script test runner: no pytest, no fixtures. Each test file calls
run_all(globals()) at the bottom; every top-level `test_*` callable taking no
arguments is invoked in definition order, failures are reported with a
traceback, and the process exits non-zero if anything failed."""

import sys
import traceback


def run_all(namespace: dict) -> None:
    tests = [(k, v) for k, v in namespace.items() if k.startswith("test_") and callable(v)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            print(f"FAIL {name}")
            traceback.print_exc()
            failures.append(name)
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        print("failed: " + ", ".join(failures))
        sys.exit(1)
