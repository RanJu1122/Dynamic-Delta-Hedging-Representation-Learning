"""Dependency-free runner for environments where pytest is not installed."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from tempfile import TemporaryDirectory

MODULES = (
    "tests.test_core",
    "tests.test_pricing_calibration",
    "tests.test_dynamic_alpha_hedging",
    "tests.test_step07",
    "tests.test_precompute",
    "tests.test_shared_precompute",
    "tests.test_full63_grid",
    "tests.test_exclusion_workflow",
    "tests.test_daily_only",
    "tests.test_shared_step07",
    "tests.test_fixed_book",
    "tests.test_mc_revision",
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
            parameters = inspect.signature(function).parameters
            if parameters:
                if set(parameters) != {"tmp_path"}:
                    raise TypeError(f"unsupported test parameters: {list(parameters)}")
                with TemporaryDirectory() as directory:
                    function(tmp_path=Path(directory))
            else:
                function()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
