"""Read and update ``results/run_config.json`` (settings, env snapshot, scope cuts)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phantom import config


def utc_now() -> str:
    """Current UTC time as an ISO-8601 string with a ``Z`` suffix."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load(path: Path = config.RUN_CONFIG_PATH) -> dict[str, Any]:
    """Return the run config, or a fresh skeleton if the file does not exist."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"cuts": []}


def save(data: dict[str, Any], path: Path = config.RUN_CONFIG_PATH) -> None:
    """Write the run config as pretty, key-sorted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update(section: str, values: Any, path: Path = config.RUN_CONFIG_PATH) -> dict[str, Any]:
    """Replace one top-level ``section`` and save; returns the full config."""
    data = load(path)
    data[section] = values
    save(data, path)
    return data


def record_cut(cut: str, reason: str, phase: str, path: Path = config.RUN_CONFIG_PATH) -> dict[str, Any]:
    """Append a scope cut (what was dropped, why, in which phase) and save."""
    data = load(path)
    data.setdefault("cuts", []).append(
        {"cut": cut, "reason": reason, "phase": phase, "ts": utc_now()}
    )
    save(data, path)
    return data
