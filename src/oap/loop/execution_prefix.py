"""Execution-prefix configuration shared by local and remote MPC paths."""
from __future__ import annotations

from dataclasses import dataclass
import math

DEFAULT_EXECUTION_PREFIX_FRACTION = 1.0 / 3.0


@dataclass(frozen=True)
class ResolvedExecutionPrefix:
    """One horizon-specific execution-prefix resolution.

    Fractions preserve the historical ``ceil(H * fraction)`` behavior.
    Explicit steps take precedence and must fit the available horizon;
    their effective fraction is the exact ``resolved_steps / H`` ratio passed
    through the existing fraction-based planner and executor interfaces.
    """

    requested_fraction: float
    requested_steps: int | None
    resolved_steps: int
    effective_fraction: float


def validate_execution_prefix_fraction(value: object) -> float:
    """Return a finite execution-prefix fraction in ``(0, 1]``."""
    try:
        fraction = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "execution_prefix_fraction must be finite and in (0, 1]"
        ) from exc
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError(
            "execution_prefix_fraction must be finite and in (0, 1], got "
            f"{value!r}"
        )
    return fraction


def validate_execution_prefix_steps(value: object) -> int:
    """Return a positive integer execution-prefix step count."""
    if isinstance(value, bool):
        raise ValueError("execution_prefix_steps must be an integer >= 1")
    try:
        steps = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "execution_prefix_steps must be an integer >= 1"
        ) from exc
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "execution_prefix_steps must be an integer >= 1"
        ) from exc
    if (
        steps < 1
        or not math.isfinite(numeric)
        or numeric != float(steps)
    ):
        raise ValueError(
            "execution_prefix_steps must be an integer >= 1, got "
            f"{value!r}"
        )
    return steps


def validate_mppi_stage_execution_prefix_steps(
    value: object,
) -> tuple[int, ...]:
    """Return an explicit non-empty MPPI prefix-step tuple.

    The CLI accepts a comma-separated sequence while ``LoopConfig`` and
    library callers may provide a tuple/list.  Stage-count and horizon checks
    require the loaded program and are therefore performed by the runner and
    episode loop, respectively.
    """
    if isinstance(value, str):
        parts: list[object] = [part.strip() for part in value.split(",")]
        if any(not part for part in parts):
            raise ValueError(
                "mppi_stage_execution_prefix_steps must be a comma-separated "
                "non-empty sequence of integers >= 1"
            )
    elif isinstance(value, (tuple, list)):
        parts = list(value)
    else:
        raise ValueError(
            "mppi_stage_execution_prefix_steps must be a comma-separated "
            "non-empty sequence of integers >= 1"
        )
    if not parts:
        raise ValueError(
            "mppi_stage_execution_prefix_steps must contain at least one step"
        )
    try:
        return tuple(validate_execution_prefix_steps(part) for part in parts)
    except ValueError as exc:
        raise ValueError(
            "mppi_stage_execution_prefix_steps must contain only integers >= 1"
        ) from exc


def resolve_execution_prefix(
    *,
    horizon_steps: object,
    execution_prefix_fraction: object = DEFAULT_EXECUTION_PREFIX_FRACTION,
    execution_prefix_steps: object | None = None,
) -> ResolvedExecutionPrefix:
    """Resolve fraction/step input against one discrete horizon.

    ``execution_prefix_steps`` has precedence when present. Values larger than
    the horizon are rejected rather than silently changing an exact physical
    interval. Without an explicit step
    count, the legacy fraction is retained verbatim and its discrete prefix is
    ``ceil(H * fraction)``, with a small integer-boundary tolerance matching
    the controller-prefix endpoint calculation.
    """
    if isinstance(horizon_steps, bool):
        raise ValueError("horizon_steps must be an integer >= 1")
    try:
        horizon = int(horizon_steps)
    except (TypeError, ValueError) as exc:
        raise ValueError("horizon_steps must be an integer >= 1") from exc
    if horizon < 1 or horizon != horizon_steps:
        raise ValueError(
            f"horizon_steps must be an integer >= 1, got {horizon_steps!r}"
        )

    fraction = validate_execution_prefix_fraction(
        execution_prefix_fraction
    )
    if execution_prefix_steps is not None:
        requested_steps = validate_execution_prefix_steps(
            execution_prefix_steps
        )
        if requested_steps > horizon:
            raise ValueError(
                "execution_prefix_steps cannot exceed horizon_steps: "
                f"{requested_steps} > {horizon}"
            )
        resolved_steps = requested_steps
        return ResolvedExecutionPrefix(
            requested_fraction=fraction,
            requested_steps=requested_steps,
            resolved_steps=resolved_steps,
            effective_fraction=float(resolved_steps) / float(horizon),
        )

    raw_steps = float(horizon) * fraction
    nearest = round(raw_steps)
    resolved_steps = (
        int(nearest)
        if math.isclose(raw_steps, nearest, rel_tol=0.0, abs_tol=1e-12)
        else int(math.ceil(raw_steps))
    )
    resolved_steps = max(1, min(horizon, resolved_steps))
    return ResolvedExecutionPrefix(
        requested_fraction=fraction,
        requested_steps=None,
        resolved_steps=resolved_steps,
        effective_fraction=fraction,
    )
