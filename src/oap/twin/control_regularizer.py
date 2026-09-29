"""Registered calibration and frozen score-active control regularizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


CONTROL_REGULARIZER_CALIBRATION_V1 = (
    "unified_control_regularizer_calibration_v1"
)
UNIFIED_CONTROL_REGULARIZER_V1 = "unified_control_regularizer_v1"
CONTROL_REGULARIZER_CALIBRATION_PROFILES = (
    CONTROL_REGULARIZER_CALIBRATION_V1,
    UNIFIED_CONTROL_REGULARIZER_V1,
)
CONTROL_REGULARIZER_ACTION_UNITS = (
    "rad_s",
    "rad_s",
    "rad_s",
    "rad_s",
    "rad_s",
    "rad_s",
    "rad_s",
    "newton",
)
CONTROL_REGULARIZER_ACTION_HALFRANGE = (
    0.2,
    0.2,
    0.2,
    0.2,
    0.2,
    0.2,
    0.2,
    80.0,
)
CONTROL_REGULARIZER_TAU_REF_S = 1.0
UNIFIED_CONTROL_REGULARIZER_WEIGHTS = (
    3.0,
    0.2,
    0.00005,
)
UNIFIED_CONTROL_REGULARIZER_WEIGHT_SOURCE = {
    "schema": "oap_rtx_four_task_one_cycle_calibration_v1",
    "base_commit": "842fd664f4a6cf72506c11b20c1c0b39f0523866",
    "hardware": "rtx5080",
    "calibration_identity": (
        "joint_velocity_force_four_task_registry_v1:rtx5080:seed1200:"
        "smoke1:control_regularizer_calibration_v1"
    ),
    "result_root_label": (
        "oap-jvf-regularizer-calibration-four-task-rtx-seed1200-"
        "842fd66-20260814"
    ),
    "task_ids": ["pick_v9", "cup_065e_v7", "push_v11", "flip_v1"],
    "seed": 1200,
    "cycles_per_task": 1,
    "population_policy": "optimization_population_valid_candidates",
    "pooled_valid_count": 3995,
    "pooled_valid_medians": {
        "realized_arm_qvel_si_time_mean": 0.0034093018621206284,
        "command_envelope_normalized_magnitude_time_mean": (
            0.15298575162887573
        ),
        "boundary_complete_normalized_command_rate_time_mean": (
            444.3421325683594
        ),
    },
}


def validate_control_regularizer_calibration_profile(
    value: Any,
    *,
    allow_none: bool = False,
) -> str | None:
    """Return the sole registered calibration profile or fail closed."""
    if value is None and allow_none:
        return None
    if (
        not isinstance(value, str)
        or value not in CONTROL_REGULARIZER_CALIBRATION_PROFILES
    ):
        raise ValueError(
            "control_regularizer_calibration_profile must be one of "
            f"{CONTROL_REGULARIZER_CALIBRATION_PROFILES}, got {value!r}"
        )
    return value


def control_regularizer_formula_identity(value: Any) -> dict[str, Any]:
    """Return the immutable formula/reduction identity for evidence/cache keys."""
    profile = validate_control_regularizer_calibration_profile(value)
    common = {
        "physical_action_units": list(CONTROL_REGULARIZER_ACTION_UNITS),
        "normalization_halfrange": list(
            CONTROL_REGULARIZER_ACTION_HALFRANGE
        ),
        "tau_ref_s": CONTROL_REGULARIZER_TAU_REF_S,
        "native_time_reduction": "sum_dt_over_total_time",
        "coordinate_reduction": "mean_squared_over_physical_coordinates",
        "realized_arm_qvel_formula": (
            "sum_t(dt*mean_j(qvel_t_j^2))/sum_t(dt)"
        ),
        "command_envelope_formula": (
            "sum_t(dt*mean_i((u_t_i/halfrange_i)^2))/sum_t(dt)"
        ),
        "boundary_complete_rate_formula": (
            "sum_t(dt*mean_i((((u_t_i-u_prev_t_i)/halfrange_i)*"
            "tau_ref/dt)^2))/sum_t(dt)"
        ),
        "boundary_policy": (
            "cycle0_registered_ack_hold_then_successful_selected_prefix_ctrl"
        ),
    }
    if profile == UNIFIED_CONTROL_REGULARIZER_V1:
        return {
            "schema": "oap_unified_control_regularizer_identity_v1",
            "profile": profile,
            "mode": "frozen_score_active",
            "weights": {
                "realized_arm_qvel": UNIFIED_CONTROL_REGULARIZER_WEIGHTS[0],
                "command_envelope_magnitude": (
                    UNIFIED_CONTROL_REGULARIZER_WEIGHTS[1]
                ),
                "boundary_complete_command_rate": (
                    UNIFIED_CONTROL_REGULARIZER_WEIGHTS[2]
                ),
            },
            "weight_source": {
                key: (
                    dict(value)
                    if key == "pooled_valid_medians"
                    else list(value)
                    if key == "task_ids"
                    else value
                )
                for key, value in UNIFIED_CONTROL_REGULARIZER_WEIGHT_SOURCE.items()
            },
            "score_policy": "base_total_plus_weighted_raw_sum",
            "validity_policy": "unchanged_existing_finite_cost_mask",
            **common,
        }
    return {
        "schema": "oap_control_regularizer_calibration_identity_v1",
        "profile": profile,
        "mode": "calibration_only_zero_weight",
        "weights": {
            "realized_arm_qvel": 0.0,
            "command_envelope_magnitude": 0.0,
            "boundary_complete_command_rate": 0.0,
        },
        **common,
    }


def control_regularizer_cache_identity(value: Any) -> tuple[Any, ...]:
    """Return no key suffix for None and a complete immutable active suffix."""
    profile = validate_control_regularizer_calibration_profile(
        value,
        allow_none=True,
    )
    if profile is None:
        return ()
    identity = control_regularizer_formula_identity(profile)
    suffix = (
        identity["schema"],
        profile,
        tuple(identity["weights"].items()),
        tuple(identity["physical_action_units"]),
        tuple(identity["normalization_halfrange"]),
        identity["tau_ref_s"],
        identity["native_time_reduction"],
        identity["coordinate_reduction"],
        identity["realized_arm_qvel_formula"],
        identity["command_envelope_formula"],
        identity["boundary_complete_rate_formula"],
        identity["boundary_policy"],
    )
    if profile == CONTROL_REGULARIZER_CALIBRATION_V1:
        return suffix
    source = identity["weight_source"]
    return suffix + (
        identity["score_policy"],
        identity["validity_policy"],
        source["schema"],
        source["base_commit"],
        source["hardware"],
        source["calibration_identity"],
        source["result_root_label"],
        tuple(source["task_ids"]),
        source["seed"],
        source["cycles_per_task"],
        source["population_policy"],
        source["pooled_valid_count"],
        tuple(source["pooled_valid_medians"].items()),
    )


def control_regularizer_weighted_terms(
    raw: Any,
    *,
    profile: Any,
    array_module: Any = np,
) -> Any:
    """Apply the registered three-term weights without exposing free knobs."""
    registered = validate_control_regularizer_calibration_profile(profile)
    xp = array_module
    values = xp.asarray(raw)
    if values.ndim < 1 or values.shape[0] != 3:
        raise ValueError("control regularizer raw terms must start with axis 3")
    if registered == CONTROL_REGULARIZER_CALIBRATION_V1:
        weights = (0.0, 0.0, 0.0)
    else:
        weights = UNIFIED_CONTROL_REGULARIZER_WEIGHTS
    reshape = (3,) + (1,) * (values.ndim - 1)
    return values * xp.asarray(weights, dtype=values.dtype).reshape(reshape)


def control_regularizer_total_cost(
    base_cost: Any,
    raw: Any,
    *,
    profile: Any,
    array_module: Any = np,
) -> Any:
    """Add the categorical profile's weighted raw terms to a base score."""
    xp = array_module
    base = xp.asarray(base_cost)
    weighted = control_regularizer_weighted_terms(
        raw,
        profile=profile,
        array_module=xp,
    )
    if weighted.shape[1:] != base.shape:
        raise ValueError("control regularizer raw population must align with cost")
    return base + xp.sum(weighted, axis=0)


def control_regularizer_base_valid_raw_violation(
    raw: Any,
    base_valid: Any,
    *,
    array_module: Any = np,
) -> Any:
    """Return whether a pre-regularizer valid candidate has non-finite raw data."""
    xp = array_module
    values = xp.asarray(raw)
    valid = xp.asarray(base_valid)
    if values.ndim < 1 or values.shape[0] != 3:
        raise ValueError("control regularizer raw terms must start with axis 3")
    if values.shape[1:] != valid.shape:
        raise ValueError("base validity must align with raw candidate axes")
    return xp.any(valid & (~xp.all(xp.isfinite(values), axis=0)))


def require_control_regularizer_result_identity(
    diagnostics: Any,
    *,
    profile: Any,
    phase: str,
) -> None:
    """Require the frozen profile identity on optimization/replay evidence."""
    registered = validate_control_regularizer_calibration_profile(
        profile,
        allow_none=True,
    )
    if registered != UNIFIED_CONTROL_REGULARIZER_V1:
        return
    evidence = (
        diagnostics.get("control_regularizer")
        if isinstance(diagnostics, dict)
        else None
    )
    expected = control_regularizer_formula_identity(
        UNIFIED_CONTROL_REGULARIZER_V1
    )
    if not isinstance(evidence, dict) or evidence.get("identity") != expected:
        raise RuntimeError(
            f"{phase} omitted the frozen control-regularizer identity"
        )


def control_regularizer_raw_diagnostics(
    *,
    realized_arm_qvel: Any,
    physical_ctrl: Any,
    previous_executed_action: Any,
    native_dt_s: float,
    boundary_valid: Any,
    array_module: Any = np,
) -> dict[str, Any]:
    """Compute the three raw, zero-weight diagnostics on a native-step tape."""
    xp = array_module
    qvel = xp.asarray(realized_arm_qvel)
    ctrl = xp.asarray(physical_ctrl)
    if qvel.ndim < 2 or qvel.shape[-1] != 7:
        raise ValueError("realized_arm_qvel must end in (H, 7)")
    if ctrl.ndim < 2 or ctrl.shape[-1] != 8:
        raise ValueError("physical_ctrl must end in (H, 8)")
    if qvel.shape[-2] != ctrl.shape[-2]:
        raise ValueError("qvel and control must use the same native horizon")
    dt = xp.asarray(native_dt_s, dtype=ctrl.dtype)
    horizon = int(ctrl.shape[-2])
    if horizon < 1 or not np.isfinite(float(native_dt_s)) or native_dt_s <= 0.0:
        raise ValueError("native horizon and native_dt_s must be positive")
    duration = dt * horizon
    qvel_raw = xp.sum(
        dt * xp.mean(qvel * qvel, axis=-1),
        axis=-1,
    ) / duration
    command_raw = control_regularizer_command_diagnostics(
        physical_ctrl=ctrl,
        previous_executed_action=previous_executed_action,
        native_dt_s=native_dt_s,
        boundary_valid=boundary_valid,
        array_module=xp,
    )
    return {
        "realized_arm_qvel_si_time_mean": qvel_raw,
        **command_raw,
    }


def control_regularizer_command_diagnostics(
    *,
    physical_ctrl: Any,
    previous_executed_action: Any,
    native_dt_s: float,
    boundary_valid: Any,
    array_module: Any = np,
) -> dict[str, Any]:
    """Compute physical command magnitude/rate without a state-tape input."""
    xp = array_module
    ctrl = xp.asarray(physical_ctrl)
    previous = xp.asarray(previous_executed_action, dtype=ctrl.dtype)
    if ctrl.ndim < 2 or ctrl.shape[-1] != 8:
        raise ValueError("physical_ctrl must end in (H, 8)")
    if previous.shape != ctrl.shape[:-2] + (8,):
        if not (ctrl.ndim == 2 and previous.shape == (8,)):
            raise ValueError(
                "previous_executed_action must align with the control batch"
            )
    horizon = int(ctrl.shape[-2])
    if horizon < 1 or not np.isfinite(float(native_dt_s)) or native_dt_s <= 0.0:
        raise ValueError("native horizon and native_dt_s must be positive")
    dt = xp.asarray(native_dt_s, dtype=ctrl.dtype)
    duration = dt * horizon
    span = xp.asarray(CONTROL_REGULARIZER_ACTION_HALFRANGE, dtype=ctrl.dtype)
    magnitude_raw = xp.sum(
        dt * xp.mean((ctrl / span) ** 2, axis=-1),
        axis=-1,
    ) / duration
    previous_row = xp.expand_dims(previous, axis=-2)
    delta = xp.concatenate(
        (ctrl[..., :1, :] - previous_row, xp.diff(ctrl, axis=-2)),
        axis=-2,
    )
    normalized_rate = (
        delta / span * CONTROL_REGULARIZER_TAU_REF_S / dt
    )
    rate_raw = xp.sum(
        dt * xp.mean(normalized_rate * normalized_rate, axis=-1),
        axis=-1,
    ) / duration
    rate_raw = xp.where(
        xp.asarray(boundary_valid),
        rate_raw,
        xp.asarray(xp.nan, dtype=ctrl.dtype),
    )
    return {
        "command_envelope_normalized_magnitude_time_mean": magnitude_raw,
        "boundary_complete_normalized_command_rate_time_mean": rate_raw,
    }


@dataclass
class ExecutedActionState:
    """Typed last successfully executed profile-native 8D command."""

    action: np.ndarray
    action_units: tuple[str, ...]
    source: str
    cycle: int | None
    stage_index: int | None
    boundary_valid: bool

    def __post_init__(self) -> None:
        action = np.asarray(self.action, dtype=float).reshape(-1)
        if action.shape != (8,):
            raise ValueError("ExecutedActionState action must be finite 8D")
        if not np.all(np.isfinite(action)):
            raise ValueError("ExecutedActionState action must be finite 8D")
        if tuple(self.action_units) != CONTROL_REGULARIZER_ACTION_UNITS:
            raise ValueError("ExecutedActionState units must be profile-native")
        # ``prefix_ctrl`` is emitted by the production MJWarp/JAX float32
        # rollout.  Compare against those exact representable actuator bounds
        # after lossless promotion to host float64: float32(0.2) is slightly
        # above Python's 0.2, while the next outward float32 value must remain
        # invalid.  This is a representation contract, not an epsilon, and it
        # never clips, rounds, or otherwise mutates the acknowledged command.
        span = np.asarray(
            CONTROL_REGULARIZER_ACTION_HALFRANGE,
            dtype=np.float32,
        ).astype(np.float64)
        if np.any(np.abs(action) > span):
            raise ValueError("ExecutedActionState exceeds the command envelope")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("ExecutedActionState source must be non-empty")
        if not isinstance(self.boundary_valid, bool):
            raise ValueError("ExecutedActionState boundary_valid must be Boolean")
        self.action = action.copy()
        self.action_units = tuple(self.action_units)

    @classmethod
    def cycle0_profile_native_hold(
        cls,
        *,
        jaw_effort_n: float,
        source: str,
    ) -> "ExecutedActionState":
        """Build the explicit zero-arm/acknowledged-effort episode boundary."""
        return cls(
            action=np.asarray([0.0] * 7 + [float(jaw_effort_n)]),
            action_units=CONTROL_REGULARIZER_ACTION_UNITS,
            source=source,
            cycle=None,
            stage_index=None,
            boundary_valid=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "oap_executed_action_state_v1",
            "action": self.action.tolist(),
            "action_units": list(self.action_units),
            "source": self.source,
            "cycle": self.cycle,
            "stage_index": self.stage_index,
            "boundary_valid": self.boundary_valid,
        }

    def update_from_successful_selected_rollout(
        self,
        selected_rollout: Any,
        *,
        cycle: int,
        stage_index: int,
    ) -> bool:
        """Acknowledge only the executed winner's native prefix endpoint."""
        if not isinstance(selected_rollout, dict):
            raise ValueError("successful selected_rollout must be an object")
        prefix_ctrl = selected_rollout.get("prefix_ctrl")
        if prefix_ctrl is None:
            raise ValueError(
                "successful selected_rollout omitted physical 8D prefix_ctrl"
            )
        updated = ExecutedActionState(
            action=np.asarray(prefix_ctrl, dtype=float),
            action_units=CONTROL_REGULARIZER_ACTION_UNITS,
            source="successful_selected_rollout.prefix_ctrl",
            cycle=int(cycle),
            stage_index=int(stage_index),
            boundary_valid=True,
        )
        self.action = updated.action
        self.action_units = updated.action_units
        self.source = updated.source
        self.cycle = updated.cycle
        self.stage_index = updated.stage_index
        self.boundary_valid = updated.boundary_valid
        return True

    def update_after_execution_attempt(
        self,
        *,
        selected_rollout: Any,
        cycle: int,
        stage_index: int,
        executed: bool,
        execution_valid: bool,
    ) -> bool:
        """Leave refusal/failed execution unchanged; acknowledge only success."""
        if not isinstance(executed, bool) or not isinstance(
            execution_valid, bool
        ):
            raise ValueError("execution acknowledgement flags must be Boolean")
        if not (executed and execution_valid):
            return False
        return self.update_from_successful_selected_rollout(
            selected_rollout,
            cycle=cycle,
            stage_index=stage_index,
        )
