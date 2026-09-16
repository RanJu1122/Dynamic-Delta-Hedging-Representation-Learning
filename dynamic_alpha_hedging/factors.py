"""Factor schemas shared by decomposition, prediction and hedging.

Old, untagged artifacts are accepted only as ATM-anchored artifacts. New
artifacts carry the method and a fitted-basis ID to prevent mixed runs.
"""
from dataclasses import dataclass
import hashlib
import json

import numpy as np


@dataclass(frozen=True)
class FactorSchema:
    method: str
    scores: tuple[str, ...]
    loadings: tuple[str, ...]
    labels: tuple[str, ...]

    @property
    def last(self):
        return tuple(f"last_{name}" for name in self.scores)


SCHEMAS = {
    "atm_anchored": FactorSchema(
        "atm_anchored", ("atm_beta_factor", "shape_score_1", "shape_score_2"),
        ("atm_beta_loading", "shape_loading_1", "shape_loading_2"),
        ("Observed ATM beta", "Residual PC1", "Residual PC2")),
    "pca": FactorSchema(
        "pca", ("pc_score_1", "pc_score_2", "pc_score_3"),
        ("pc_loading_1", "pc_loading_2", "pc_loading_3"),
        ("PC1", "PC2", "PC3")),
}
METADATA = ("factor_method", "factor_basis_id")


def metadata_value(frame, column):
    if column not in frame:
        return None
    values = frame[column].dropna().unique()
    if len(values) != 1:
        raise ValueError(f"factor artifact must have one {column}")
    return str(values[0])


def schema_for(value="atm_anchored"):
    if isinstance(value, str):
        method = value
    else:
        method = metadata_value(value, "factor_method")
        if method is None:
            if any(c in value for c in (*SCHEMAS["pca"].scores, *SCHEMAS["pca"].loadings)):
                raise ValueError("PCA artifacts require factor_method metadata")
            method = "atm_anchored"
    if method not in SCHEMAS:
        raise ValueError(f"unknown factor method: {method}")
    return SCHEMAS[method]


def validate_factor_pair(scores, loadings, config):
    schemas = [schema_for(frame) for frame in (scores, loadings)]
    if any(s.method != config.step4_factor_method for s in schemas):
        raise ValueError("factor method mismatch: scores, loadings and config must agree")
    ids = [metadata_value(frame, "factor_basis_id") for frame in (scores, loadings)]
    if ids[0] != ids[1] or (schemas[0].method == "pca" and ids[0] is None):
        raise ValueError("factor basis mismatch: rebuild scores and loadings together")
    for frame, columns in ((scores, schemas[0].scores), (loadings, schemas[0].loadings)):
        missing = set(columns).difference(frame.columns)
        if missing:
            raise ValueError(f"factor artifact misses {sorted(missing)}")
    return schemas[0]


def copy_metadata(source, target):
    """Broadcast metadata even to dates without usable factor observations."""
    for column in METADATA:
        value = metadata_value(source, column)
        if value is not None:
            target[column] = value
    return target


def basis_id(method, axes, train_dates, train_values, intercept, loadings):
    digest = hashlib.sha256()
    digest.update(json.dumps([method, axes, list(map(str, train_dates))]).encode())
    for array in (train_values, intercept, loadings):
        digest.update(np.asarray(array, dtype="<f8").tobytes())
    return digest.hexdigest()
