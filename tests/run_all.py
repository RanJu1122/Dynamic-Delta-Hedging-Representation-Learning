"""Dependency-free runner for environments where pytest is not installed."""

from __future__ import annotations

import importlib

MODULES = (
    "tests.test_core",
    "tests.test_pricing_calibration",
    "tests.test_dynamic_alpha_hedging",
)


def main() -> int:
    tests = []
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        tests.extend((f"{module_name}.{name}", value)
                     for name, value in vars(module).items()
                     if name.startswith("test_") and callable(value))
    failed = 0
    for name, function in sorted(tests):
        try:
            function()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
