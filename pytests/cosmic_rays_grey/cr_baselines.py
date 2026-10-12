"""Measured-baseline registry for the grey-CR tests (Phase A test plan X1/X2).

Every number a test is calibrated against lives in ``baselines.json`` next to this
file, keyed by ``"<file>::<test>"`` and a metric name, with the date it was measured.
Docstrings refer to the registry instead of quoting the numbers, so they cannot go
stale (X1). Tests gate on it through :class:`Baselines` (X2):

- ``upper(metric, value, factor=1.5, floor=0)``: an error, ``value <= max(factor *
  baseline, floor)``; ``floor`` keeps round-off-sized metrics from gating on their own
  noise. A value far *below* the baseline only prints a note (refresh the registry).
- ``lower(metric, value, margin)``: a convergence order, ``value >= baseline - margin``.
- ``close(metric, value, rel=..., abs_tol=...)``: a physical quantity (plateau value,
  front position), two-sided.

The physics gates (literature orders, invariants) stay hard-coded in the tests; the
registry only adds "no worse than measured" gates on top of them.

Refresh: run the tests with ``CR_UPDATE_BASELINES=1``. Every check then records the
measured value instead of comparing (the physics gates still apply), and
``baselines.json`` is rewritten. Review its diff before committing it.
"""

import datetime
import fcntl
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

BASELINE_FILE = Path(__file__).with_name("baselines.json")
DEFAULT_FACTOR = 1.5
UPDATE = os.environ.get("CR_UPDATE_BASELINES", "0") not in ("", "0")


def _round(value: float, digits: int = 4) -> float:
    if value == 0.0 or not math.isfinite(value):
        return float(value)
    return float(f"{value:.{digits}g}")


def load() -> dict:
    with open(BASELINE_FILE) as f:
        return json.load(f)


def baseline(test_id: str, metric: str) -> float:
    """The registered value of ``metric`` for ``test_id``."""
    return load()[test_id][metric]["value"]


class Baselines:
    """Baseline checks for one test, collected and raised together by :meth:`check`."""

    def __init__(self, test_id: str):
        self.test_id = test_id
        self.measured = {}
        self.failures = []
        self.entries = {} if UPDATE else load().get(test_id, {})

    def _base(self, metric: str, value: float):
        self.measured[metric] = float(value)
        if UPDATE:
            return None
        if metric not in self.entries:
            self.failures.append(f"{metric}: no baseline (run with CR_UPDATE_BASELINES=1)")
            return None
        return self.entries[metric]["value"]

    def upper(self, metric: str, value: float, factor: float = DEFAULT_FACTOR, floor: float = 0.0) -> None:
        base = self._base(metric, value)
        if base is None:
            return
        bound = max(factor * base, floor)
        print(f"  baseline {metric}: {value:.4g} (registry {base:.4g}, gate <= {bound:.4g})")
        if not value <= bound:
            self.failures.append(f"{metric} {value:.4g} > {bound:.4g} ({factor:g} x baseline {base:.4g})")
        elif value < base / factor and base / factor > floor:
            print(f"  NOTE {self.test_id} {metric}: {value:.4g} well below baseline {base:.4g}; refresh?")

    def lower(self, metric: str, value: float, margin: float) -> None:
        base = self._base(metric, value)
        if base is None:
            return
        print(f"  baseline {metric}: {value:.4g} (registry {base:.4g}, gate >= -{margin:g})")
        if not value >= base - margin:
            self.failures.append(f"{metric} {value:.4g} < baseline {base:.4g} - {margin:g}")

    def close(self, metric: str, value: float, rel: float = 0.0, abs_tol: float = 0.0) -> None:
        base = self._base(metric, value)
        if base is None:
            return
        tol = rel * abs(base) + abs_tol
        print(f"  baseline {metric}: {value:.6g} (registry {base:.6g}, gate +- {tol:.3g})")
        if not abs(value - base) <= tol:
            self.failures.append(f"{metric} {value:.6g} differs from baseline {base:.6g} by more than {tol:.3g}")

    def check(self) -> None:
        """Raise on failed checks; in update mode write the measured values instead."""
        if UPDATE:
            _write(self.test_id, self.measured)
            print(f"  baselines recorded for {self.test_id}: {len(self.measured)} metrics")
            return
        assert not self.failures, f"{self.test_id} vs baselines.json: " + "; ".join(self.failures)


def _write(test_id: str, measured: dict) -> None:
    today = datetime.date.today().isoformat()
    # Parallel test processes may record at once; the lock lives outside the repository.
    lock_path = Path(tempfile.gettempdir()) / f"cr_baselines_{hashlib.md5(str(BASELINE_FILE).encode()).hexdigest()[:12]}.lock"
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = load() if BASELINE_FILE.exists() else {}
        entry = data.setdefault(test_id, {})
        for metric, value in measured.items():
            entry[metric] = {"value": _round(value), "date": today}
        with open(BASELINE_FILE, "w") as f:
            json.dump(data, f, indent=1, sort_keys=True)
            f.write("\n")
