"""Strict hardware calibration for real-robot ``ObjectHeld`` evidence.

Production calibration is object-independent: it measures only the installed
GN01 empty-close width distribution and gripper-width sensor resolution.
Runtime attributes a non-empty obstruction to the current subject through the
episode's registration-backed continuous pose and reconstructed geometry; no
held object, arm probe, rigidity threshold, or task-specific grasp is part of
calibration.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oap.utils.io import sha256_file

SCHEMA = "oap_object_held_evidence_calibration_v2"
LEGACY_SCHEMA = "oap_object_held_evidence_calibration_v1"

_VALUE_FIELDS = {
    "gripper_air_close_width_m": (0.0, 0.1),
    "gripper_width_resolution_m": (1e-6, 0.02),
}
_BASE_FIELDS = {
    "schema",
    "source",
    "captured_at",
    "hardware_id",
}


@dataclass(frozen=True)
class ObjectHeldEvidenceCalibration:
    """Hardware-only constants consumed by the real ``ObjectHeld`` channel."""

    schema: str
    source: str
    captured_at: str
    hardware_id: str
    gripper_air_close_width_m: float
    gripper_width_resolution_m: float
    sha256: str
    path: Path
    migrated_from_v1: bool = False

    def loop_config_values(self) -> dict[str, float]:
        """Return only terminal-relevant hardware measurements."""
        return {
            name: float(getattr(self, name))
            for name in _VALUE_FIELDS
        }


def _required_text(document: dict[str, Any], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"hardware evidence calibration {key!r} must be non-empty"
        )
    text = value.strip()
    if "PLACEHOLDER" in text.upper() or "DO_NOT_USE" in text.upper():
        raise ValueError(
            f"hardware evidence calibration {key!r} is still a template "
            "placeholder"
        )
    return text


def canonical_runtime_hardware_identity_sha256(
    identity: Any,
) -> str:
    """Hash one JSON hardware identity with the production canonical form."""
    if not isinstance(identity, dict) or not identity:
        raise ValueError(
            "runtime_hardware_identity must be a non-empty JSON object"
        )
    try:
        payload = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "runtime_hardware_identity must contain only finite JSON values"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _required_sha256(document: dict[str, Any], key: str) -> str:
    value = _required_text(document, key)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(
            f"hardware evidence calibration {key!r} must be a lowercase "
            "SHA-256"
        )
    return value


def _measured_number(
    document: dict[str, Any],
    name: str,
    bounds: tuple[float, float],
) -> float:
    raw = document.get(name)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(
            f"hardware evidence calibration {name!r} must be a measured number"
        )
    value = float(raw)
    lower, upper = bounds
    if not math.isfinite(value) or value < lower or value > upper:
        raise ValueError(
            f"hardware evidence calibration {name!r}={raw!r} is outside "
            f"[{lower}, {upper}]"
        )
    return value


def load_object_held_evidence_calibration(
    path: Path | str,
) -> ObjectHeldEvidenceCalibration:
    """Load the production v2 hardware calibration.

    Legacy v1 artifacts may still be inspected by external migration tooling,
    but they are intentionally not production-ready.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(
            f"hardware evidence calibration does not exist: {resolved}"
        )
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot parse hardware evidence calibration {resolved}: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise ValueError("hardware evidence calibration JSON must be an object")

    schema = document.get("schema")
    if schema == LEGACY_SCHEMA:
        raise ValueError(
            "legacy ObjectHeld calibration v1 is migration-only and cannot "
            "satisfy production readiness or authorization"
        )
    if schema != SCHEMA:
        raise ValueError(
            "hardware evidence calibration schema must be "
            f"{SCHEMA!r}, got {schema!r}"
        )
    allowed = _BASE_FIELDS | set(_VALUE_FIELDS)
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise ValueError(
            f"unknown hardware evidence calibration fields: {unknown}"
        )

    values = {
        name: _measured_number(document, name, bounds)
        for name, bounds in _VALUE_FIELDS.items()
    }
    return ObjectHeldEvidenceCalibration(
        schema=str(schema),
        source=_required_text(document, "source"),
        captured_at=_required_text(document, "captured_at"),
        hardware_id=_required_sha256(document, "hardware_id"),
        sha256=sha256_file(resolved),
        path=resolved,
        migrated_from_v1=False,
        **values,
    )


def program_uses_object_held(program: Any) -> bool:
    """Return whether the typed program explicitly requests held evidence."""
    from oap.program.predicates import ObjectHeld, TemporalHold

    for stage in getattr(program, "stages", ()):
        for predicate in (*stage.running, *stage.terminal):
            while isinstance(predicate, TemporalHold):
                predicate = predicate.inner
            if isinstance(predicate, ObjectHeld):
                return True
    return False


def require_real_object_held_evidence(
    *,
    program: Any,
    execute: bool,
    prepare_real_execution: bool,
    calibration: ObjectHeldEvidenceCalibration | None,
    foundationpose_mode: str,
    disable_held_verify: bool,
) -> None:
    """Fail closed before connection when real held evidence is incomplete."""
    if not (execute or prepare_real_execution) or not program_uses_object_held(
        program
    ):
        return
    if calibration is None:
        raise ValueError(
            "the program contains ObjectHeld, so real-execution readiness "
            "requires --hardware-evidence-calibration-json with measured "
            "empty-close baseline and width resolution"
        )
    if str(foundationpose_mode) != "actual":
        raise ValueError(
            "the program contains ObjectHeld, so real-execution readiness "
            "requires --foundationpose-mode actual for current "
            "registration-backed spatial attribution"
        )
    if disable_held_verify:
        raise ValueError(
            "--disable-held-verify is incompatible with real execution of a "
            "program containing ObjectHeld"
        )
