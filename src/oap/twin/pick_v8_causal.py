"""Closed single-variable diagnostics around the successful Pick-v8 run."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PICK_V8_PREFIX_ONLY_V1 = "pick_v8_prefix_only_v1"
PICK_V8_GRIPPER_ONLY_V1 = "pick_v8_gripper_only_v1"
PICK_V8_BASELINE_V2 = "pick_v8_baseline_v2"
PICK_V8_PREFIX_ONLY_V2 = "pick_v8_prefix_only_v2"
PICK_V8_GRIPPER_ONLY_V2 = "pick_v8_gripper_only_v2"
PICK_V8_COMBINED_V2 = "pick_v8_combined_v2"
PICK_V8_COMBINED_DIRECT_RELEASE_V3 = "pick_v8_combined_direct_release_v3"
PICK_V8_COMBINED_RELEASE_LATERAL_V5 = "pick_v8_combined_release_lateral_v5"
PICK_V8_COMBINED_RELEASE_LATERAL_UNIFORM30_V6 = (
    "pick_v8_combined_release_lateral_uniform30_v6"
)
PICK_V8_COMBINED_RELEASE_LATERAL_UNIFORM20_V6 = (
    "pick_v8_combined_release_lateral_uniform20_v6"
)
PICK_V8_CAUSAL_PROFILES = (
    PICK_V8_PREFIX_ONLY_V1,
    PICK_V8_GRIPPER_ONLY_V1,
    PICK_V8_BASELINE_V2,
    PICK_V8_PREFIX_ONLY_V2,
    PICK_V8_GRIPPER_ONLY_V2,
    PICK_V8_COMBINED_V2,
    PICK_V8_COMBINED_DIRECT_RELEASE_V3,
    PICK_V8_COMBINED_RELEASE_LATERAL_V5,
    PICK_V8_COMBINED_RELEASE_LATERAL_UNIFORM30_V6,
    PICK_V8_COMBINED_RELEASE_LATERAL_UNIFORM20_V6,
)
PICK_V8_CAUSAL_PROGRAM_PATH = (
    "experiments/placement/minimal_eraser_on_box_task_program_paper_mppi.json"
)
PICK_V8_CAUSAL_INITIAL_OBSERVATION_PATH = "eraser/pose/eraser_refined.json"
PICK_V8_CAUSAL_INSTRUCTION = (
    "Place the black felt blackboard eraser on top of the white product box "
    "and release it."
)
PICK_V8_SUCCESS_STAGE_PREFIXES = (50, 50, 20, 20, 20)
# The exact successful-v8 release state loses its two-sided hold on the first
# native step at every tested opening effort from -0.1 N through -80 N.  One
# Newton is the preregistered round, actively-opening value: it leaves margin
# over that single-state minimum while avoiding the 80 N free-space over-open.
#
# SCOPE, and the limit of that calibration (measured 2026-08-19).  Despite the
# pick_v8 name this is the opening authority for EVERY stage of EVERY task that
# routes through unified_mppi_effort_uses_proven_pick_carrier -- all four of
# Pick, Cup, Push and ToolPush -- while closing keeps the full 80 N
# (GN01_DIRECT_FORCE_LIMIT_N).  Closing authority is therefore 80x opening
# authority.  The calibration above was taken on an UNLOADED release; under a
# loaded fingertip the asymmetry makes jaw closure effectively irreversible.
# Measured on ToolPush v10 chunk 127, arm pose held fixed: the actually
# commanded -0.945 N (98% of the whole opening budget) moves the width from
# -0.000409 m to -0.000174 m, i.e. +0.23 mm, and then stalls; teleporting the
# eraser 1 m away gives +7.53 mm from the same force; -80 N gives +101.8 mm.
# The eraser, not the actuator, is what jams it, and the optimizer had been
# commanding ~-0.94 on all 12 knots since chunk 50 without being able to open.
#
# This is deliberately NOT changed here.  It never fired in any run recorded
# under the repaired jaw pricing: across route-B's ToolPush and Push episodes
# the measured width never went below 0.045 m and 0.086 m respectively, so the
# ratchet has no measured victim once gripper_command is scaled correctly (see
# device_cost.py's GripperCommand branch).  Raising it would change the action
# space and make every recorded result non-comparable, which is not warranted
# by any evidence currently in hand.  If it is ever revisited, it needs its own
# preregistered sweep of the loaded-open force, not a guess.
PICK_V8_GRIPPER_ONLY_V2_OPEN_FORCE_LIMIT_N = 1.0
PICK_V8_CAUSAL_MPPI_TEMPERATURE_POLICY = "legacy_v8_adaptive_eta_v1"
PICK_V8_CAUSAL_MPPI_ETA_MIN = 5.0
PICK_V8_CAUSAL_MPPI_ETA_MAX = 10.0
PICK_V8_CAUSAL_MPPI_DECREASE = 0.9
PICK_V8_CAUSAL_MPPI_INCREASE = 1.2
PICK_V8_CAUSAL_VALUES: dict[str, Any] = {
    "pool_size": 1001,
    "horizon_steps": 600,
    "num_knots": 12,
    "sigma_fraction": 0.06,
    "sampler_mode": "single_scale",
    "mtp_jaw_mode": "preserve_nominal",
    "local_sigma_fraction": 0.02,
    "optimizer": "mppi",
    "cem_rounds": 1,
    "cem_elite_fraction": 0.1,
    "cem_min_std_fraction": 0.01,
    "mppi_temperature": 1.0,
    "mppi_execution_mode": "softmax_weighted_mean",
    "candidate_selection_mode": "valid_then_total_cost",
    # Preserve the literal successful-v8 config. Exact step overrides and the
    # stage schedule determine the effective fractions used by the device.
    "execution_prefix_fraction": 1.0 / 3.0,
    "execution_prefix_steps": 50,
    "max_stage_cycles": 180,
    "mppi_cycle_budget_scope": "stage",
    "seed": 1012,
    "arm_velocity_weight": 0.1,
    "record_candidate_cost_telemetry": False,
}
PICK_V8_SUCCESS_BASE_CONTROLLER: dict[str, Any] = {
    **PICK_V8_CAUSAL_VALUES,
    "control_profile": "legacy_velocity_width",
    "mppi_stage_execution_prefix_steps": PICK_V8_SUCCESS_STAGE_PREFIXES,
    "interpolation": "linear_degree_one",
}
PICK_V8_V2_SUCCESS_BASE_CONTROLLER: dict[str, Any] = {
    **PICK_V8_SUCCESS_BASE_CONTROLLER,
    "jaw_actuator_decode": "continuous_width_target_m",
    "gripper_command_cost": "legacy_v8_physical_width_equivalent_v1",
    "cost_shaping_profile": None,
}


def pick_v8_causal_values(profile: Any) -> dict[str, Any]:
    """Return frozen controller values, changing only the v6 uniform prefix."""
    value = validate_pick_v8_causal_profile(profile)
    values = dict(PICK_V8_CAUSAL_VALUES)
    prefix_steps = {
        PICK_V8_COMBINED_RELEASE_LATERAL_UNIFORM30_V6: 30,
        PICK_V8_COMBINED_RELEASE_LATERAL_UNIFORM20_V6: 20,
    }.get(value)
    if prefix_steps is not None:
        values["execution_prefix_steps"] = prefix_steps
        values["execution_prefix_fraction"] = prefix_steps / float(
            values["horizon_steps"]
        )
    return values


def pick_v8_causal_cost_shaping_profile(profile: Any) -> str | None:
    """Map categorical experiments onto their exact registered cost profile."""
    value = validate_pick_v8_causal_profile(profile)
    if value == PICK_V8_COMBINED_DIRECT_RELEASE_V3:
        return PICK_V8_COMBINED_DIRECT_RELEASE_V3
    if value in {
        PICK_V8_COMBINED_RELEASE_LATERAL_V5,
        PICK_V8_COMBINED_RELEASE_LATERAL_UNIFORM30_V6,
        PICK_V8_COMBINED_RELEASE_LATERAL_UNIFORM20_V6,
    }:
        return PICK_V8_COMBINED_RELEASE_LATERAL_V5
    return None


def pick_v8_causal_device_prefix_fractions(profile: Any) -> tuple[float, ...]:
    """Return the only device prefix fractions allowed by this category."""
    value = validate_pick_v8_causal_profile(profile)
    if pick_v8_causal_uses_original_prefix_schedule(value):
        return (50.0 / 600.0, 20.0 / 600.0)
    values = pick_v8_causal_values(value)
    return (
        float(values["execution_prefix_steps"])
        / float(values["horizon_steps"]),
    )


def pick_v8_causal_is_v2(profile: Any) -> bool:
    value = validate_pick_v8_causal_profile(profile)
    return value in {
        PICK_V8_BASELINE_V2,
        PICK_V8_PREFIX_ONLY_V2,
        PICK_V8_GRIPPER_ONLY_V2,
        PICK_V8_COMBINED_V2,
        PICK_V8_COMBINED_DIRECT_RELEASE_V3,
        PICK_V8_COMBINED_RELEASE_LATERAL_V5,
        PICK_V8_COMBINED_RELEASE_LATERAL_UNIFORM30_V6,
        PICK_V8_COMBINED_RELEASE_LATERAL_UNIFORM20_V6,
    }


def pick_v8_causal_uses_width_jaw(profile: Any) -> bool:
    value = validate_pick_v8_causal_profile(profile)
    return value in {
        PICK_V8_PREFIX_ONLY_V1,
        PICK_V8_BASELINE_V2,
        PICK_V8_PREFIX_ONLY_V2,
    }


def pick_v8_causal_uses_original_prefix_schedule(profile: Any) -> bool:
    value = validate_pick_v8_causal_profile(profile)
    return value in {
        PICK_V8_GRIPPER_ONLY_V1,
        PICK_V8_BASELINE_V2,
        PICK_V8_GRIPPER_ONLY_V2,
    }


def pick_v8_causal_uses_asymmetric_effort(profile: Any) -> bool:
    return validate_pick_v8_causal_profile(profile) in {
        PICK_V8_GRIPPER_ONLY_V2,
        PICK_V8_COMBINED_V2,
        PICK_V8_COMBINED_DIRECT_RELEASE_V3,
        PICK_V8_COMBINED_RELEASE_LATERAL_V5,
        PICK_V8_COMBINED_RELEASE_LATERAL_UNIFORM30_V6,
        PICK_V8_COMBINED_RELEASE_LATERAL_UNIFORM20_V6,
    }


@dataclass(frozen=True)
class PickV8CausalMppiTemperatureState:
    """Internal proof of one authentic legacy-v8 adaptive transition."""

    temperature: float
    previous_temperature: float
    effective_samples: float
    policy: str = PICK_V8_CAUSAL_MPPI_TEMPERATURE_POLICY


def _legacy_v8_next_mppi_temperature(
    current_temperature: float,
    effective_samples: float,
) -> float:
    current = float(current_temperature)
    eta = float(effective_samples)
    if not math.isfinite(current) or current <= 0.0:
        raise ValueError("Pick-v8 causal adaptive temperature must be positive")
    if not math.isfinite(eta) or eta < 0.0:
        raise ValueError(
            "Pick-v8 causal adaptive temperature ESS must be finite and nonnegative"
        )
    multiplier = 1.0
    # A valid population always contributes its minimum-cost member with
    # exp(0)=1, so eta==0 is exactly the old update's count==0 hold branch.
    if eta > PICK_V8_CAUSAL_MPPI_ETA_MAX:
        multiplier = PICK_V8_CAUSAL_MPPI_DECREASE
    elif 0.0 < eta < PICK_V8_CAUSAL_MPPI_ETA_MIN:
        multiplier = PICK_V8_CAUSAL_MPPI_INCREASE
    return float(np.float32(current) * np.float32(multiplier))


def _float32_ulp_distance(left: float, right: float) -> int | None:
    """Return positive-float carrier distance, or None for invalid values."""
    left32 = np.float32(left)
    right32 = np.float32(right)
    if (
        not np.isfinite(left32)
        or not np.isfinite(right32)
        or left32 <= 0.0
        or right32 <= 0.0
    ):
        return None
    left_bits = int(np.asarray(left32).view(np.uint32))
    right_bits = int(np.asarray(right32).view(np.uint32))
    return abs(left_bits - right_bits)


def _matches_float32_transition(actual: float, expected: float) -> bool:
    distance = _float32_ulp_distance(actual, expected)
    return distance is not None and distance <= 1


def require_pick_v8_causal_mppi_temperature(
    *,
    current_temperature: float,
    state: PickV8CausalMppiTemperatureState | None,
) -> None:
    """Fail closed unless temperature is initial or proven by old-v8 policy."""
    current = float(current_temperature)
    if state is None:
        if current != float(PICK_V8_CAUSAL_VALUES["mppi_temperature"]):
            raise ValueError(
                "Pick-v8 causal adaptive temperature must start at 1.0"
            )
        return
    if not isinstance(state, PickV8CausalMppiTemperatureState):
        raise ValueError("Pick-v8 causal adaptive temperature state is invalid")
    if state.policy != PICK_V8_CAUSAL_MPPI_TEMPERATURE_POLICY:
        raise ValueError("Pick-v8 causal temperature policy identity mismatch")
    expected = _legacy_v8_next_mppi_temperature(
        state.previous_temperature,
        state.effective_samples,
    )
    if (
        current != state.temperature
        or not _matches_float32_transition(state.temperature, expected)
    ):
        raise ValueError(
            "Pick-v8 causal adaptive temperature is not an authentic "
            "legacy-v8 state transition"
        )


def advance_pick_v8_causal_mppi_temperature(
    *,
    previous_state: PickV8CausalMppiTemperatureState | None,
    current_temperature: float,
    effective_samples: float | None,
    reported_next_temperature: float | None,
) -> PickV8CausalMppiTemperatureState:
    """Validate one device update and mint the next internal state proof."""
    require_pick_v8_causal_mppi_temperature(
        current_temperature=current_temperature,
        state=previous_state,
    )
    if effective_samples is None or reported_next_temperature is None:
        raise ValueError(
            "Pick-v8 causal adaptive temperature update telemetry is absent"
        )
    expected = _legacy_v8_next_mppi_temperature(
        current_temperature,
        effective_samples,
    )
    reported = float(reported_next_temperature)
    if not _matches_float32_transition(reported, expected):
        raise ValueError(
            "Pick-v8 causal adaptive temperature device update violates "
            "the legacy-v8 policy"
        )
    return PickV8CausalMppiTemperatureState(
        # Preserve the device carrier verbatim. Replacing it with the host
        # expectation accumulates rounding drift over long adaptive runs.
        temperature=reported,
        previous_temperature=float(current_temperature),
        effective_samples=float(effective_samples),
    )


def pick_v8_causal_controller_identity(profile: Any) -> dict[str, Any]:
    """Expose the exact one-field delta from the successful v8 controller."""
    profile = validate_pick_v8_causal_profile(profile)
    identity = dict(
        PICK_V8_V2_SUCCESS_BASE_CONTROLLER
        if pick_v8_causal_is_v2(profile)
        else PICK_V8_SUCCESS_BASE_CONTROLLER
    )
    identity.update(pick_v8_causal_values(profile))
    identity["pick_v8_causal_profile"] = profile
    if profile in (
        PICK_V8_PREFIX_ONLY_V1,
        PICK_V8_PREFIX_ONLY_V2,
        PICK_V8_COMBINED_V2,
        PICK_V8_COMBINED_DIRECT_RELEASE_V3,
        PICK_V8_COMBINED_RELEASE_LATERAL_V5,
        PICK_V8_COMBINED_RELEASE_LATERAL_UNIFORM30_V6,
        PICK_V8_COMBINED_RELEASE_LATERAL_UNIFORM20_V6,
    ):
        identity["mppi_stage_execution_prefix_steps"] = None
    if profile in (
        PICK_V8_GRIPPER_ONLY_V1,
        PICK_V8_GRIPPER_ONLY_V2,
        PICK_V8_COMBINED_V2,
        PICK_V8_COMBINED_DIRECT_RELEASE_V3,
        PICK_V8_COMBINED_RELEASE_LATERAL_V5,
        PICK_V8_COMBINED_RELEASE_LATERAL_UNIFORM30_V6,
        PICK_V8_COMBINED_RELEASE_LATERAL_UNIFORM20_V6,
    ):
        identity["control_profile"] = "legacy_velocity_effort"
    if pick_v8_causal_uses_asymmetric_effort(profile):
        identity["jaw_actuator_decode"] = (
            "asymmetric_continuous_effort_open_1n_close_80n"
        )
    if pick_v8_causal_is_v2(profile):
        identity["cost_shaping_profile"] = pick_v8_causal_cost_shaping_profile(
            profile
        )
    return identity


def validate_pick_v8_causal_profile(
    value: Any,
    *,
    allow_none: bool = False,
) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or value not in PICK_V8_CAUSAL_PROFILES:
        raise ValueError(
            "pick_v8_causal_profile must be one of "
            f"{PICK_V8_CAUSAL_PROFILES}, got {value!r}"
        )
    return value


def require_pick_v8_causal_contract(
    value: Any,
    *,
    program: Any | None = None,
) -> str | None:
    """Fail closed unless config retains the exact Pick-v8 causal boundary."""
    profile = validate_pick_v8_causal_profile(
        getattr(value, "pick_v8_causal_profile", None),
        allow_none=True,
    )
    if profile is None:
        return None
    mismatches = [
        f"{name}={getattr(value, name, None)!r} (required {expected!r})"
        for name, expected in pick_v8_causal_values(profile).items()
        if getattr(value, name, None) != expected
    ]
    expected_control = (
        "legacy_velocity_width"
        if pick_v8_causal_uses_width_jaw(profile)
        else "legacy_velocity_effort"
    )
    expected_schedule = (
        PICK_V8_SUCCESS_STAGE_PREFIXES
        if pick_v8_causal_uses_original_prefix_schedule(profile)
        else None
    )
    if getattr(value, "control_profile", None) != expected_control:
        mismatches.append(f"control_profile must be {expected_control!r}")
    if getattr(value, "mppi_stage_execution_prefix_steps", None) != expected_schedule:
        mismatches.append(f"stage prefix schedule must be {expected_schedule!r}")
    if getattr(value, "task", None) != PICK_V8_CAUSAL_INSTRUCTION:
        mismatches.append("task differs from Pick-v8 authority")
    for field, suffix in (
        ("program_json", PICK_V8_CAUSAL_PROGRAM_PATH),
        ("initial_obs_json", PICK_V8_CAUSAL_INITIAL_OBSERVATION_PATH),
    ):
        actual = getattr(value, field, None)
        if actual is None or not str(actual).replace("\\", "/").endswith(suffix):
            mismatches.append(f"{field} differs from Pick-v8 authority")
    if program is not None:
        from oap.program import TaskProgram

        authority = TaskProgram.from_json(
            pick_v8_causal_program_authority().read_text(encoding="utf-8")
        )
        if program.to_dict() != authority.to_dict():
            mismatches.append("TaskProgram differs from exact Pick-v8 authority")
    expected_cost_shaping = pick_v8_causal_cost_shaping_profile(profile)
    if getattr(value, "cost_shaping_profile", None) != expected_cost_shaping:
        mismatches.append(
            f"cost_shaping_profile must be {expected_cost_shaping!r}"
        )
    for field in (
        "unified_mppi_effort_profile",
        "joint_velocity_force_task_id",
        "joint_velocity_force_trial_budget",
        "joint_velocity_force_algorithm_arm",
        "joint_velocity_force_dial_arm",
        "joint_velocity_force_phase3_finalist_arm",
        "joint_velocity_force_phase3_seed",
        "control_regularizer_calibration_profile",
    ):
        if getattr(value, field, None) is not None:
            mismatches.append(f"{field} must be absent")
    if not bool(getattr(value, "offline", False)):
        mismatches.append("offline must be true")
    if mismatches:
        raise ValueError("pick_v8_causal contract mismatch: " + "; ".join(mismatches))
    return profile


def pick_v8_causal_program_authority() -> Path:
    from oap.twin.control_profile import external_input_path
    return external_input_path(
        PICK_V8_CAUSAL_PROGRAM_PATH, purpose="registered program authority"
    )


def require_pick_v8_causal_input_identity(
    *,
    profile: str,
    control_profile: str,
    bundle_manifest: Any,
    initial_observation: dict[str, Any],
    subject_name: str,
    reference_name: str | None,
) -> dict[str, Any]:
    """Reuse only the shared sealed Pick scene verifier, reporting v8."""
    profile = validate_pick_v8_causal_profile(profile)
    from oap.twin.control_profile import (
        require_s0_control_profile_input_identity,
    )

    identity = require_s0_control_profile_input_identity(
        control_profile=control_profile,
        joint_velocity_force_task_id="pick_v9",
        bundle_manifest=bundle_manifest,
        initial_observation=initial_observation,
        subject_name=subject_name,
        reference_name=reference_name,
    )
    return {
        **dict(identity or {}),
        "schema": "oap_pick_v8_single_variable_input_v1",
        "pick_v8_causal_profile": profile,
        "registered_task_id": "pick_v8",
        "program_path": PICK_V8_CAUSAL_PROGRAM_PATH,
        "program_instruction": PICK_V8_CAUSAL_INSTRUCTION,
        "program_provenance": "reviewed:pick-place-result-focused-v8:2026-08-11",
        "program_stage_count": 5,
        "program_stage_names": [
            "approach_eraser_open",
            "grasp_eraser_continuous_width",
            "center_eraser_over_box_held",
            "lower_centered_eraser_to_release_hover",
            "release_eraser",
        ],
    }
