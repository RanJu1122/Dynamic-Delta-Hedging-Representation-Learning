"""Small, stable manifest helpers shared by dynamic-alpha stages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: str | Path, *, stage: str, config: Any,
                   inputs: dict[str, str], validation: dict[str, Any]) -> Path:
    """Write the provenance contract that lets a later stage verify inputs."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    config_data = asdict(config) if is_dataclass(config) else dict(config)
    payload = {
        "stage": stage,
        "config": config_data,
        "inputs": inputs,
        "validation": validation,
    }
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return target


def read_manifest(path: str | Path, *, expected_stage: str | None = None) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if expected_stage is not None and payload.get("stage") != expected_stage:
        raise ValueError(
            f"expected stage {expected_stage!r}, got {payload.get('stage')!r}")
    return payload
