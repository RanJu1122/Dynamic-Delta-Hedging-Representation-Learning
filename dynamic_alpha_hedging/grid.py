"""Validate surface-node coverage at data and model boundaries."""

import numpy as np
import pandas as pd


def validate_cells(frame, tenors, levels, *, source):
    """Require the exact Cartesian grid, tolerating only CSV float round-off.

    Missing observations/NaN labels are handled by each stage separately.
    This checks that an entire tenor/level cell has not silently disappeared.
    """
    columns = ("tenor", "level")
    if not set(columns).issubset(frame.columns):
        raise ValueError(f"{source} must contain tenor and level columns")
    cells = frame[list(columns)].drop_duplicates().copy()
    for column, axis in zip(columns, (tenors, levels)):
        grid = np.asarray(axis, dtype=float)
        values = cells[column].to_numpy(dtype=float)
        distances = np.abs(values[:, None] - grid[None, :])
        nearest = distances.argmin(axis=1)
        valid = np.isfinite(values) & (distances[np.arange(len(values)), nearest] <= 1e-12)
        if not valid.all():
            raise ValueError(f"{source} grid mismatch: unexpected {column}={values[~valid].tolist()}; "
                             "rebuild upstream with matching axes")
        cells[column] = grid[nearest]
    expected = pd.MultiIndex.from_product([tenors, levels], names=columns)
    observed = pd.MultiIndex.from_frame(cells).unique()
    missing = expected.difference(observed)
    if len(missing):
        raise ValueError(f"{source} grid mismatch: missing cells={list(missing)}; "
                         "rebuild upstream with matching axes")
