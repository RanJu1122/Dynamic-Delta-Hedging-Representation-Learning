"""Read-only comparison of the two supplied, trusted SVI pickles.

Run from the project root with .venv/bin/python scripts/audit_spot_plateau.py.
Date alignment is reported as observed evidence; input dates/prices are never repaired.
"""
from pathlib import Path
import datetime as dt
import hashlib
import json
import pickle

import numpy as np
import pandas as pd

from dynamic_alpha_hedging.config import DynamicAlphaConfig
from dynamic_alpha_hedging.data_loader import build_surface

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "spot_plateau_audit_20260914"
FIELDS = ("ATMVol", "Skew", "Putwing", "Callwing", "Kurt", "StickinessRatio")
START, END = dt.date(2026, 3, 25), dt.date(2026, 4, 9)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal_records(left, right):
    if set(left) != set(right):
        return False
    return all(np.array_equal(np.asarray(left[k]), np.asarray(right[k])) for k in left)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, metadata, windows, changes = {}, {}, [], []
    for name in ("svi_data.pkl", "svi_param.pkl"):
        path = ROOT / "data" / name
        before = sha(path)
        with path.open("rb") as handle:
            history = pickle.load(handle)
        records[name] = history
        dates = sorted(history)
        metadata[name] = {"sha256": before, "records": len(dates), "start": str(dates[0]),
            "end": str(dates[-1]), "weekend_date_keys": sum(d.weekday() >= 5 for d in dates),
            "fields": sorted(set().union(*(r.keys() for r in history.values())))}
        for i, date in enumerate(dates):
            if not START <= date <= END:
                continue
            row = history[date]
            previous = history[dates[i-1]] if i else None
            windows.append(dict(file=name, raw_date=date, spot=float(row["Spot"]),
                previous_raw_date=dates[i-1] if i else None,
                zero_spot_change=previous is not None and row["Spot"] == previous["Spot"],
                n_vol_dates=len(row["VolDate"]), first_expiry=row["VolDate"][0],
                **{field+"_first": float(row[field][0]) for field in FIELDS}))
            if previous is not None:
                # Compare matching calendar expiries, retaining source positions for audit.
                prior = {expiry: j for j, expiry in enumerate(previous["VolDate"])}
                pairs = [(j, prior[e]) for j, e in enumerate(row["VolDate"]) if e in prior]
                for field in FIELDS:
                    difference = np.array([row[field][j]-previous[field][k] for j, k in pairs])
                    changes.append(dict(file=name, raw_date=date, previous_raw_date=dates[i-1],
                        field=field, common_expiries=len(pairs), changed_values=int(np.count_nonzero(difference)),
                        max_absolute_change=float(np.abs(difference).max()) if pairs else None))
        assert before == sha(path), "input file changed during read-only audit"
    old, new = records["svi_data.pkl"], records["svi_param.pkl"]
    alignment = []
    for new_date in sorted(new):
        if not START <= new_date <= END:
            continue
        old_date = new_date+dt.timedelta(days=1)
        if old_date in old:
            alignment.append(dict(param_date=new_date, data_date=old_date,
                spot=float(new[new_date]["Spot"]), all_fields_equal=equal_records(new[new_date], old[old_date]),
                **{k+"_equal": np.array_equal(np.asarray(new[new_date][k]), np.asarray(old[old_date][k]))
                   for k in new[new_date]}))
    metadata["alignment"] = {"calendar_shift_days_in_reported_window": 1,
        "matching_window_records": sum(r["all_fields_equal"] for r in alignment),
        "compared_window_records": len(alignment),
        "matching_one_day_shift_records_over_whole_old_file": sum(
            d-dt.timedelta(days=1) in new and equal_records(v, new[d-dt.timedelta(days=1)])
            for d, v in old.items()),
        "caution": "not a proven timezone conversion for all records; both files have date-only keys"}
    # Demonstrate that identical normalized SVI shapes allow different absolute spot scales.
    config = DynamicAlphaConfig()
    date = dt.date(2026, 3, 31)
    record = new[date]
    expiry = record["VolDate"][0]
    levels = np.array([.8, 1., 1.1])
    iv_rows = []
    for hypothetical_spot in (float(record["Spot"]), 6500., 6700.):
        surface = build_surface(date, {**record, "Spot": hypothetical_spot}, config.market_conventions)
        iv = surface.implied_vol(expiry, levels*hypothetical_spot)
        iv_rows.append(dict(hypothetical_spot=hypothetical_spot, **{f"iv_level_{l:g}": v for l,v in zip(levels,iv)}))
    matrix = pd.DataFrame(iv_rows)
    maxdiff = float(np.max(np.abs(matrix.iloc[:, 1:].to_numpy()-matrix.iloc[0, 1:].to_numpy())))
    assert maxdiff < 1e-12
    metadata["spot_scale_nonidentifiability"] = {"date": str(date), "expiry": str(expiry),
        "max_iv_difference_at_identical_levels": maxdiff,
        "hypothetical_spots_are_not_estimates": True}
    for filename, rows in (("raw_window.csv", windows), ("parameter_changes.csv", changes),
                           ("date_alignment.csv", alignment), ("spot_scale_example.csv", iv_rows)):
        pd.DataFrame(rows).to_csv(OUT / filename, index=False)
    for name in records:
        assert sha(ROOT / "data" / name) == metadata[name]["sha256"]
    (OUT / "audit.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame(alignment)[["param_date", "data_date", "spot", "all_fields_equal"]].to_string(index=False))
    print("Input hashes unchanged; normalized-IV scale-invariance check passed.")
    print(OUT)


if __name__ == "__main__":
    main()
