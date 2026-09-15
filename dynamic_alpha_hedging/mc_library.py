"""Versioned date-local pricing library. Prediction/training settings are NOT keys.

Each shard is one date/tenor, with all precomputed levels and alpha nodes.
Readers take exact subsets, never interpolate dates/tenors/levels or run MC.
"""

from dataclasses import asdict, replace
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import file_sha256
from .data_loader import date_at_tau, EXCLUDED_OBSERVATIONS
from .hedging import MCMapStore
from .step03 import _anchor_at_sticky_strike, _cell_quality


NUMERICAL_FIELDS = (
    "step3_spot_bump_fraction", "step3_n_paths", "step3_seed", "step3_antithetic",
    "step3_n_substeps", "step3_n_ratio", "step3_ratio_min", "step3_ratio_max",
    "step3_vol_floor", "step3_vol_cap",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def engine_signature():
    root = Path(__file__).resolve().parent.parent
    files = [f"svi_localvol/{n}.py" for n in (
        "montecarlo", "surface", "svi", "blackscholes", "conventions", "params")]
    return digest({
        "files": {name: file_sha256(root / name) for name in files},
        "measure": inspect.getsource(MCMapStore.measure),
        "measure_many": inspect.getsource(MCMapStore.measure_many),
        "anchor": inspect.getsource(_anchor_at_sticky_strike),
        "expiry": inspect.getsource(date_at_tau),
    })


def snapshot_signature(surface):
    # Date, rates/calendar and this day's quotes; never the next day's data.
    return digest({"market": asdict(surface.market), "quotes": asdict(surface.quotes),
                   "beta_clamp": getattr(surface, "beta_clamp", None),
                   "calendar_repair": getattr(surface, "calendar_repair", False)})


def axis(values, name):
    values = tuple(float(x) for x in values)
    if (not values or not np.isfinite(values).all() or min(values) <= 0
            or tuple(sorted(set(values))) != values):
        raise ValueError(f"{name} must be finite, positive, unique and increasing")
    return values


def subset_axis(requested, available, name):
    selected = []
    for value in requested:
        hits = [x for x in available if np.isclose(x, value, rtol=0., atol=1e-12)]
        if len(hits) != 1:
            raise ValueError(f"MC library does not cover {name}={value}; no extrapolation")
        selected.append(hits[0])
    return selected


def _atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _audit(frame, config):
    # Quality thresholds can change without repricing; no stale stored pass flag.
    frame = frame.drop(columns=["quality_pass", "inverse_available", "quality_failures"],
                       errors="ignore")
    quality = _cell_quality(frame, config)
    return frame.merge(quality[["tenor", "level", "quality_pass", "inverse_available",
                                "quality_failures"]], on=["tenor", "level"],
                       validate="many_to_one")


class MCLibrary:
    def __init__(self, root, config, tenors, levels, *, writable=False):
        self.root, self.writable = Path(root), writable
        self.config = config
        self.tenors, self.levels = axis(tenors, "tenors"), axis(levels, "levels")
        path = self.root / "library.json"
        if not path.exists() and not writable:
            raise FileNotFoundError(f"missing pricing library: {path}; precompute first")
        expected = {"schema": 1, "engine": engine_signature(),
                    "numerical": {key: getattr(config, key) for key in NUMERICAL_FIELDS},
                    "tenors": list(self.tenors), "levels": list(self.levels),
                    "alphas": list(config.step3_alphas),
                    "beta_policy": "raw_minus_same_date_alpha_one; raw MC delta unchanged"}
        if path.exists():
            self.metadata = json.loads(path.read_text())
            for key in ("schema", "engine", "numerical", "beta_policy"):
                if self.metadata[key] != expected[key]:
                    raise ValueError(f"MC library {key} mismatch; use its frozen settings or a new library")
            if writable and any(self.metadata[k] != expected[k] for k in ("tenors", "levels", "alphas")):
                raise ValueError("cannot overwrite a library with different axes; use a new directory")
            subset_axis(self.tenors, self.metadata["tenors"], "tenor")
            subset_axis(self.levels, self.metadata["levels"], "level")
            subset_axis(config.step3_alphas, self.metadata["alphas"], "alpha")
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            self.metadata = expected
            _atomic_json(path, expected)
        self.index_path = self.root / "index.json"
        if not self.index_path.exists() and not writable:
            raise FileNotFoundError(f"missing pricing index: {self.index_path}; precompute first")
        self.index = json.loads(self.index_path.read_text()) if self.index_path.exists() else {}

    @staticmethod
    def configured(root, config):
        """Only MC numerical settings/alpha grid come from the library, not model settings."""
        metadata = json.loads((Path(root) / "library.json").read_text())
        return replace(config, **metadata["numerical"], step3_alphas=tuple(metadata["alphas"]))

    def _key(self, surface, tenor):
        date = surface.market.pricing_date
        if date in EXCLUDED_OBSERVATIONS:
            raise ValueError(f"excluded pricing snapshot: {date}")
        tenor = subset_axis([tenor], self.metadata["tenors"], "tenor")[0]
        return f"{date}_{tenor:.12g}", tenor

    def read(self, surface, tenor):
        key, tenor = self._key(surface, tenor)
        entry = self.index.get(key)
        if entry is None:
            raise FileNotFoundError(f"missing MC shard {key}; read-only backtest never starts MC")
        if entry["snapshot"] != snapshot_signature(surface):
            raise ValueError(f"changed market/quotes for MC shard {key}; reprice that snapshot")
        path = self.root / "shards" / f"{key}.csv"
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"MC shard checksum mismatch: {path}")
        frame = pd.read_csv(path)
        if not pd.to_datetime(frame.calibration_date).dt.date.eq(surface.market.pricing_date).all():
            raise ValueError("MC shard date differs from requested decision date")
        for column, available in (("level", self.metadata["levels"]),
                                   ("alpha", self.metadata["alphas"])):
            frame[column] = subset_axis(frame[column], available, column)
        if (len(frame) != len(self.metadata["levels"])*len(self.metadata["alphas"])
                or frame.duplicated(["level", "alpha"]).any()
                or not np.isclose(frame.tenor, tenor, atol=1e-12, rtol=0).all()):
            raise ValueError(f"invalid MC shard axes: {key}")
        if not np.allclose(frame.strike, frame.level*surface.ref_spot):
            raise ValueError(f"MC shard strike/refSpot mismatch: {key}")
        # Preserve caller's exact float axes; all stored alpha nodes support interpolation.
        rows = []
        for level in self.levels:
            part = frame[np.isclose(frame.level, level, atol=1e-12, rtol=0)].copy()
            part["level"], part["tenor"] = level, tenor
            rows.append(part)
        return _audit(pd.concat(rows, ignore_index=True), self.config)

    def get(self, surface):
        return pd.concat([self.read(surface, t) for t in self.tenors], ignore_index=True)

    def ensure(self, surface, tenor):
        return next(self.ensure_many(surface, (tenor,)))[1]

    def ensure_many(self, surface, tenors):
        """Yield cached shards, then price missing tenors together for one date.

        Shards remain independently checksummed and atomically resumable. No
        cached tenor is simulated merely to include it in the shared batch.
        """
        if not self.writable:
            raise RuntimeError("read-only MC library cannot precompute")
        missing = {}
        for tenor in dict.fromkeys(tenors):
            key, tenor = self._key(surface, tenor)
            expiry = date_at_tau(surface, tenor)
            if not surface.taus[0] <= surface.tau_vol(expiry) <= surface.taus[-1]:
                raise ValueError("requested maturity is outside this snapshot's quotes")
            if key in self.index:
                yield tenor, self.read(surface, tenor)
            else:
                missing[tenor] = (key, expiry)
        if not missing:
            return
        pricer = MCMapStore(self.config, missing, self.metadata["levels"], self.root / "shards")
        frames = pricer.measure_many(surface, missing)
        for tenor, (key, expiry) in missing.items():
            frame = frames[tenor]
            frame["actual_expiry"] = expiry
            frame["actual_tau"] = surface.tau_vol(expiry)
            frame["ref_spot"] = surface.ref_spot
            path = self.root / "shards" / f"{key}.csv"
            temporary = path.with_suffix(".tmp")
            frame.to_csv(temporary, index=False)
            temporary.replace(path)
            self.index[key] = {"snapshot": snapshot_signature(surface), "sha256": file_sha256(path)}
            _atomic_json(self.index_path, self.index)
            yield tenor, self.read(surface, tenor)
