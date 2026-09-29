"""Registered execution-budget protocols for reproducible paper runs."""
from __future__ import annotations

from typing import Any


PAPER_GLOBAL_250_V1 = "paper_global_250_v1"
_KNOWN_PROTOCOLS = {PAPER_GLOBAL_250_V1}


def validate_execution_budget_protocol(
    value: Any,
    *,
    allow_none: bool = False,
) -> str | None:
    """Return a registered protocol name, failing closed otherwise."""
    if value is None and allow_none:
        return None
    name = str(value).strip()
    if name not in _KNOWN_PROTOCOLS:
        raise ValueError(
            f"unknown execution-budget protocol {name!r}; "
            f"known: {sorted(_KNOWN_PROTOCOLS)!r}"
        )
    return name


def apply_execution_budget_protocol(
    values: dict[str, Any],
    protocol: Any,
) -> dict[str, Any]:
    """Apply only the registered loop-budget fields to resolved values."""
    resolved = dict(values)
    name = validate_execution_budget_protocol(protocol, allow_none=True)
    if name is None:
        return resolved
    if name == PAPER_GLOBAL_250_V1:
        resolved.update({
            "max_stage_cycles": 250,
            "mppi_cycle_budget_scope": "episode",
        })
    return resolved


def execution_budget_identity(protocol: Any) -> dict[str, Any] | None:
    """Return the artifact identity for a registered budget protocol."""
    name = validate_execution_budget_protocol(protocol, allow_none=True)
    if name is None:
        return None
    if name == PAPER_GLOBAL_250_V1:
        return {
            "protocol": name,
            "limit": 250,
            "scope": "episode",
            "unit": "receding_horizon_cycle",
            "commanded_motion_s_max": 20.0,
            "source": "fixed_budget_entrypoint",
        }
    raise AssertionError(f"unhandled execution-budget protocol {name!r}")
