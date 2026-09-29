"""Fail-closed MPPI effort controllers over sealed TaskProgram authorities."""
from __future__ import annotations

import copy
import os
import math
from typing import Any

from oap.execution_budget import apply_execution_budget_protocol
from oap.program import TaskProgram
from oap.program.cost_shaping import (
    BASELINE_COST_SHAPING,
    CUP_V7_S1_RUNNING_POSITION_FOCUS_V1,
    CUP_SIMPLE_TILT_TERMINAL_HELD_60_V1,
    FLIP_V1_S0_RUNNING_POSITION_FOCUS_V1,
    PICK_V8_AT_REST_TERMINAL_V6,
    PICK_V8_COMBINED_DIRECT_RELEASE_V3,
    TOOLPUSH_8CM_MEDIATION_WEIGHT_4_V1,
    TOOLPUSH_8CM_MEDIATION_DURATION_WEIGHT_4_V2,
)
from oap.twin.control_profile import (
    JOINT_VELOCITY_FORCE_TASK_REGISTRY,
    evaluation_device_seed,
    external_input_path,
)
from oap.twin import lab_cup_fixture


UNIFIED_MPPI_EFFORT_V1 = "unified_mppi_effort_v1"
UNIFIED_MPPI_EFFORT_UNIFORM20_V2 = "unified_mppi_effort_uniform20_v2"
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_V3 = (
    "unified_mppi_effort_uniform20_n2001_v3"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_V1 = (
    "unified_mppi_effort_uniform20_n2001_toolpush_v1"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_V2 = (
    "unified_mppi_effort_uniform20_n2001_toolpush_budget240_v2"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H300_P20 = (
    "unified_mppi_effort_uniform20_n2001_toolpush_budget240_h300_p20"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H600_P40 = (
    "unified_mppi_effort_uniform20_n2001_toolpush_budget240_h600_p40"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_H600_P40_V9 = (
    "unified_mppi_effort_uniform20_n2001_pick_h600_p40_v9"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_DIRECT_RELEASE_H600_P40_V10 = (
    "unified_mppi_effort_uniform20_n2001_pick_direct_release_h600_p40_v10"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_AT_REST_H600_P40_V12 = (
    "unified_mppi_effort_uniform20_n2001_"
    "pick_at_rest_h600_p40_v12"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_S0_ATT20_H600_P40_V13 = (
    "unified_mppi_effort_uniform20_n2001_"
    "pick_s0_att20_h600_p40_v13"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_DIRECT_RELEASE_TF0_H600_P40_V11 = (
    "unified_mppi_effort_uniform20_n2001_"
    "pick_direct_release_tf0_h600_p40_v11"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PUSH_6CM_ONE_SIDED_H600_P40_V20 = (
    "unified_mppi_effort_uniform20_n2001_"
    "push_6cm_one_sided_h600_p40_v20"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_H600_P40_V3 = (
    "unified_mppi_effort_uniform20_n2001_toolpush_8cm_h600_p40_v3"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_MEDIATION4_H600_P40_V4 = (
    "unified_mppi_effort_uniform20_n2001_toolpush_8cm_mediation4_h600_p40_v4"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_DURATION4_H600_P40_V5 = (
    "unified_mppi_effort_uniform20_n2001_toolpush_8cm_duration4_h600_p40_v5"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_6CM_RL10_H600_P40_V14 = (
    "unified_mppi_effort_uniform20_n2001_"
    "toolpush_6cm_rl10_h600_p40_v14"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_DURATION4_TF0_H600_P40_V6 = (
    "unified_mppi_effort_uniform20_n2001_"
    "toolpush_8cm_duration4_tf0_h600_p40_v6"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_S1_FOCUS_V4 = (
    "unified_mppi_effort_uniform20_n2001_cup_s1_focus_v4"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SEQUENTIAL_V5 = (
    "unified_mppi_effort_uniform20_n2001_cup_sequential_v5"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_COARSE_FINE_V6 = (
    "unified_mppi_effort_uniform20_n2001_cup_coarse_fine_v6"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_ORIENTATION_FINE_V7 = (
    "unified_mppi_effort_uniform20_n2001_cup_orientation_fine_v7"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_FINE_EARLIEST_V8 = (
    "unified_mppi_effort_uniform20_n2001_cup_fine_earliest_v8"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_V9 = (
    "unified_mppi_effort_uniform20_n2001_cup_simple_tilt_v9"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_NO_RUNNING_HELD_V10 = (
    "unified_mppi_effort_uniform20_n2001_"
    "cup_simple_tilt_no_running_held_v10"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_H600_P40_V14 = (
    "unified_mppi_effort_uniform20_n2001_"
    "cup_side_grasp_h600_p40_v14"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_H600_P40_V15 = (
    "unified_mppi_effort_uniform20_n2001_"
    "cup_side_grasp_at_rest_h600_p40_v15"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_ALIGN8_H600_P40_V16 = (
    "unified_mppi_effort_uniform20_n2001_"
    "cup_side_grasp_at_rest_align8_h600_p40_v16"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_V7_FUNNEL_AT_REST_H600_P40_V17 = (
    "unified_mppi_effort_uniform20_n2001_"
    "cup_v7_funnel_at_rest_h600_p40_v17"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_UNCONSTRAINED_GRASP_H600_P40_V18 = (
    "unified_mppi_effort_uniform20_n2001_"
    "cup_unconstrained_grasp_h600_p40_v18"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TAKES_WEIGHT_H600_P40_V19 = (
    "unified_mppi_effort_uniform20_n2001_"
    "cup_grasp_takes_weight_h600_p40_v19"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT30_H600_P40_V20 = (
    "unified_mppi_effort_uniform20_n2001_"
    "cup_grasp_tilt30_h600_p40_v20"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_TIGHT_POS_H600_P40_V21 = (
    "unified_mppi_effort_uniform20_n2001_"
    "cup_tight_pos_h600_p40_v21"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_BOUNDED_GRIP_H600_P40_V22 = (
    "unified_mppi_effort_uniform20_n2001_"
    "cup_bounded_grip_h600_p40_v22"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_ONLY_H600_P40_V23 = (
    "unified_mppi_effort_uniform20_n2001_"
    "cup_grasp_tilt_only_h600_p40_v23"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_OFFAXIS10_H600_P40_V24 = (
    "unified_mppi_effort_uniform20_n2001_"
    "cup_grasp_tilt_offaxis10_h600_p40_v24"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_S2_H600_P40_V13 = (
    "unified_mppi_effort_uniform20_n2001_"
    "cup_simple_tilt_held60_s2_h600_p40_v13"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_H600_P40_V12 = (
    "unified_mppi_effort_uniform20_n2001_"
    "cup_simple_tilt_held60_h600_p40_v12"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_SOFT_HELD_H600_P40_V11 = (
    "unified_mppi_effort_uniform20_n2001_"
    "cup_simple_tilt_soft_held_h600_p40_v11"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_S0_FOCUS_V4 = (
    "unified_mppi_effort_uniform20_n2001_flip_s0_focus_v4"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_SEQUENTIAL_V5 = (
    "unified_mppi_effort_uniform20_n2001_flip_sequential_v5"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_ORIENTATION_SEQUENTIAL_V6 = (
    "unified_mppi_effort_uniform20_n2001_flip_orientation_sequential_v6"
)
UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_FULL_POSE_EARLIEST_V7 = (
    "unified_mppi_effort_uniform20_n2001_flip_full_pose_earliest_v7"
)
UNIFIED_MPPI_EFFORT_PROFILES = (
    UNIFIED_MPPI_EFFORT_V1,
    UNIFIED_MPPI_EFFORT_UNIFORM20_V2,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_V3,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_V1,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_V2,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H300_P20,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H600_P40,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_H600_P40_V9,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_DIRECT_RELEASE_H600_P40_V10,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_DIRECT_RELEASE_TF0_H600_P40_V11,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_AT_REST_H600_P40_V12,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_S0_ATT20_H600_P40_V13,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PUSH_6CM_ONE_SIDED_H600_P40_V20,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_H600_P40_V3,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_MEDIATION4_H600_P40_V4,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_DURATION4_H600_P40_V5,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_DURATION4_TF0_H600_P40_V6,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_6CM_RL10_H600_P40_V14,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_S1_FOCUS_V4,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SEQUENTIAL_V5,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_COARSE_FINE_V6,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_ORIENTATION_FINE_V7,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_FINE_EARLIEST_V8,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_V9,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_NO_RUNNING_HELD_V10,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_SOFT_HELD_H600_P40_V11,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_H600_P40_V12,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_S2_H600_P40_V13,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_H600_P40_V14,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_H600_P40_V15,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_ALIGN8_H600_P40_V16,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_V7_FUNNEL_AT_REST_H600_P40_V17,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_UNCONSTRAINED_GRASP_H600_P40_V18,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TAKES_WEIGHT_H600_P40_V19,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT30_H600_P40_V20,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_TIGHT_POS_H600_P40_V21,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_BOUNDED_GRIP_H600_P40_V22,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_ONLY_H600_P40_V23,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_OFFAXIS10_H600_P40_V24,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_S0_FOCUS_V4,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_SEQUENTIAL_V5,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_ORIENTATION_SEQUENTIAL_V6,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_FULL_POSE_EARLIEST_V7,
)

# These OaP controller profiles use VLM-generated task costs.
_GENERATED_PROGRAM_PROFILES = frozenset({
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PUSH_6CM_ONE_SIDED_H600_P40_V20,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_6CM_RL10_H600_P40_V14,
})

_TRADITIONAL_H600_P40_PROFILES = frozenset({
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_H600_P40_V9,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_DIRECT_RELEASE_H600_P40_V10,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_DIRECT_RELEASE_TF0_H600_P40_V11,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_AT_REST_H600_P40_V12,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_S0_ATT20_H600_P40_V13,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PUSH_6CM_ONE_SIDED_H600_P40_V20,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_H600_P40_V3,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_MEDIATION4_H600_P40_V4,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_DURATION4_H600_P40_V5,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_DURATION4_TF0_H600_P40_V6,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_6CM_RL10_H600_P40_V14,
})
_TRADITIONAL_P40_PROFILES = _TRADITIONAL_H600_P40_PROFILES

# These are controller values, not task overrides.  ``sigma_fraction`` remains
# the successful v8 CLI identity; the MPPI device implementation's actual
# fixed normalized covariance/std are bound separately below.
UNIFIED_MPPI_EFFORT_VALUES: dict[str, Any] = {
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
    "execution_prefix_fraction": 1.0 / 12.0,
    "execution_prefix_steps": 50,
    "max_stage_cycles": 180,
    "mppi_cycle_budget_scope": "stage",
    "seed": 1012,
    "arm_velocity_weight": 0.1,
    "record_candidate_cost_telemetry": False,
}

UNIFIED_MPPI_EFFORT_IDENTITY: dict[str, Any] = {
    "schema": "oap_unified_mppi_effort_v1",
    "profile": UNIFIED_MPPI_EFFORT_V1,
    "optimizer_action_half_span": [0.2] * 7 + [1.0],
    "physical_action_half_span": [0.2] * 7 + [80.0],
    "action_units": ["rad_s"] * 7 + ["signed_effort_n"],
    "jaw_latent_decode_n_per_unit": 80.0,
    "proposal_distribution": "fixed_gaussian_halton_single_scale",
    "proposal_covariance_normalized": 0.1,
    "proposal_std_normalized": math.sqrt(0.1),
    "device_proposal_seed": 0,
    "halton_panel_reused_across_mpc_cycles": True,
    "optimization_rollouts": 1000,
    "exact_singleton_replay_rollouts": 1,
    "task_program_terminal_predicates": "unchanged",
    "hard_validity": "unchanged",
    "control_regularizer": None,
    "table_contact_force_weight": 0.0,
}

UNIFIED_MPPI_EFFORT_DEVICE_VALUES: dict[str, Any] = {
    "control_profile": "joint_velocity_force",
    "total_rollouts": 1001,
    "n_steps": 600,
    "num_knots": 12,
    "temperature": 1.0,
    "execution_mode": "softmax_weighted_mean",
    "sampler_mode": "single_scale",
    "mtp_jaw_mode": "preserve_nominal",
    "seed": 0,
    "arm_velocity_weight": 0.1,
    "prefer_earliest": False,
    "regularizer_profile": None,
    "table_contact_force_weight": 0.0,
    "dial_arm": None,
    "execution_prefix_frac": 50.0 / 600.0,
}


_ABLATION_MARK = "__ablate_"

# One shared parameter set per ablation, applied to every task.  ``values`` and
# ``device`` name the two resolvers each override reaches; a knob that lives in
# both must be stated in both, because the device dict is what the GPU reads.
UNIFIED_MPPI_EFFORT_ABLATIONS: dict[str, dict[str, dict[str, Any]]] = {
    # Fewer samples.  The reference (arXiv 2307.09105) uses 750; 751 keeps the
    # one exact-replay rollout the production 2001 also carries.
    # The sample count lives in BOTH dicts and the two must move together:
    # cli/run.py reads unified_values["pool_size"] to size the pool, while the
    # device dict's total_rollouts is what the identity block and the launcher
    # checks report.  Declaring only the device side produced a run that logged
    # "2001 rollouts" while every check said 1001.  k8 and p80 already declared
    # both sides; these two did not.
    "n751": {"values": {"pool_size": 751},
             "device": {"total_rollouts": 751}},
    # 1000 samples plus the one exact-replay rollout, matching the 2001 = 2000+1
    # convention the production set already uses.
    "n1001": {"values": {"pool_size": 1001},
               "device": {"total_rollouts": 1001}},
    # EXECUTOR ABLATION (Addendum 71), never pooled with sealed-arm results:
    # iterated sampling under the same rollout budget. 4 rounds x 251 = 1004
    # rollouts against the production single batch of 1001; each round refits
    # diagonal mean/std from the elites (cem_elite_refit, min_std floored) and
    # the returned trajectory is the best-of-budget incumbent. Reachable only
    # via a profile name no sealed arm ever used, so adding this key cannot
    # change any recorded run's resolution.
    "cem4n251": {"values": {"pool_size": 1004, "cem_rounds": 4},
                 "device": {"total_rollouts": 1004}},
    # Fewer decision variables: 8 x 7 knots instead of 12 x 7.
    "k8": {"values": {"num_knots": 8}, "device": {"num_knots": 8}},
    # Execute more of each solve: 80 steps (160 ms) instead of 40.
    "p80": {
        "values": {
            "execution_prefix_steps": 80,
            "execution_prefix_fraction": 80.0 / 600.0,
        },
        "device": {"execution_prefix_frac": 80.0 / 600.0},
    },
}


def split_unified_mppi_effort_ablation(value: Any) -> tuple[Any, str | None]:
    """Split ``<base>__ablate_<key>`` into its parts, or pass a plain name."""
    if not isinstance(value, str) or _ABLATION_MARK not in value:
        return value, None
    base, _, key = value.partition(_ABLATION_MARK)
    if key not in UNIFIED_MPPI_EFFORT_ABLATIONS:
        raise ValueError(
            "unknown ablation key "
            f"{key!r}; known: {sorted(UNIFIED_MPPI_EFFORT_ABLATIONS)!r}"
        )
    return base, key


def unified_mppi_effort_ablated_profile(base: str, key: str) -> str:
    """Name the ablation of ``base``, failing closed on an unknown knob."""
    if key not in UNIFIED_MPPI_EFFORT_ABLATIONS:
        raise ValueError(
            "unknown ablation key "
            f"{key!r}; known: {sorted(UNIFIED_MPPI_EFFORT_ABLATIONS)!r}"
        )
    validate_unified_mppi_effort_profile(base)
    return f"{base}{_ABLATION_MARK}{key}"


def _apply_ablation(
    values: dict[str, Any], key: str | None, scope: str
) -> dict[str, Any]:
    """Override only the declared fields, and only ones that already exist."""
    if key is None:
        return values
    for field, value in UNIFIED_MPPI_EFFORT_ABLATIONS[key].get(scope, {}).items():
        if field not in values:
            raise ValueError(
                f"ablation {key!r} names {scope} field {field!r}, "
                "which the base profile does not resolve"
            )
        values[field] = value
    return values


def unified_mppi_effort_values(profile: Any) -> dict[str, Any]:
    """Return the frozen numeric tuple for one unified controller category."""
    profile, _ablation = split_unified_mppi_effort_ablation(profile)
    profile = validate_unified_mppi_effort_profile(profile)
    values = dict(UNIFIED_MPPI_EFFORT_VALUES)
    if unified_mppi_effort_uses_proven_pick_carrier(profile):
        values.update({
            "execution_prefix_fraction": 20.0 / 600.0,
            "execution_prefix_steps": 20,
            # Diagnostic-only: the objective and selection still use the same
            # device total; this only copies its existing per-term breakdown.
            "record_candidate_cost_telemetry": True,
        })
    if profile in {
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_V3,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_V1,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_V2,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H300_P20,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H600_P40,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_S1_FOCUS_V4,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SEQUENTIAL_V5,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_COARSE_FINE_V6,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_ORIENTATION_FINE_V7,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_FINE_EARLIEST_V8,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_V9,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_NO_RUNNING_HELD_V10,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_SOFT_HELD_H600_P40_V11,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_H600_P40_V12,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_S2_H600_P40_V13,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_H600_P40_V14,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_H600_P40_V15,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_ALIGN8_H600_P40_V16,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_V7_FUNNEL_AT_REST_H600_P40_V17,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_UNCONSTRAINED_GRASP_H600_P40_V18,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TAKES_WEIGHT_H600_P40_V19,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT30_H600_P40_V20,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_TIGHT_POS_H600_P40_V21,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_BOUNDED_GRIP_H600_P40_V22,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_ONLY_H600_P40_V23,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_OFFAXIS10_H600_P40_V24,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_S0_FOCUS_V4,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_SEQUENTIAL_V5,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_ORIENTATION_SEQUENTIAL_V6,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_FULL_POSE_EARLIEST_V7,
    }:
        values["pool_size"] = 2001
    if profile in _TRADITIONAL_P40_PROFILES:
        values["pool_size"] = 2001
    if profile in {
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_V2,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H300_P20,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H600_P40,
    }:
        values["max_stage_cycles"] = 240
    if (
        profile
        == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H300_P20
    ):
        values.update({
            "horizon_steps": 300,
            "execution_prefix_fraction": 20.0 / 300.0,
        })
    if (
        profile
        == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H600_P40
    ):
        values.update({
            "execution_prefix_fraction": 40.0 / 600.0,
            "execution_prefix_steps": 40,
        })
    if (
        profile
        in {
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_SOFT_HELD_H600_P40_V11,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_H600_P40_V12,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_S2_H600_P40_V13,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_H600_P40_V14,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_H600_P40_V15,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_ALIGN8_H600_P40_V16,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_V7_FUNNEL_AT_REST_H600_P40_V17,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_UNCONSTRAINED_GRASP_H600_P40_V18,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TAKES_WEIGHT_H600_P40_V19,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT30_H600_P40_V20,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_TIGHT_POS_H600_P40_V21,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_BOUNDED_GRIP_H600_P40_V22,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_ONLY_H600_P40_V23,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_OFFAXIS10_H600_P40_V24,
        }
    ):
        values.update({
            "execution_prefix_fraction": 40.0 / 600.0,
            "execution_prefix_steps": 40,
            "max_stage_cycles": 240,
        })
    if profile in _TRADITIONAL_P40_PROFILES:
        values.update({
            "execution_prefix_fraction": 40.0 / 600.0,
            "execution_prefix_steps": 40,
            "max_stage_cycles": 240,
        })
    if profile in {
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_FINE_EARLIEST_V8,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_FULL_POSE_EARLIEST_V7,
    }:
        values["candidate_selection_mode"] = "result_terminal_earliest"
    if profile in _PRODUCTION_PROFILES:
        values["max_stage_cycles"] = 480

    return _apply_ablation(values, _ablation, "values")


def unified_mppi_effort_uses_proven_pick_carrier(profile: Any) -> bool:
    profile = validate_unified_mppi_effort_profile(profile)
    return profile in _TRADITIONAL_P40_PROFILES or profile in {
        UNIFIED_MPPI_EFFORT_UNIFORM20_V2,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_V3,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_V1,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_V2,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H300_P20,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H600_P40,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_S1_FOCUS_V4,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SEQUENTIAL_V5,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_COARSE_FINE_V6,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_ORIENTATION_FINE_V7,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_FINE_EARLIEST_V8,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_V9,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_NO_RUNNING_HELD_V10,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_SOFT_HELD_H600_P40_V11,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_H600_P40_V12,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_S2_H600_P40_V13,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_H600_P40_V14,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_H600_P40_V15,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_ALIGN8_H600_P40_V16,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_V7_FUNNEL_AT_REST_H600_P40_V17,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_UNCONSTRAINED_GRASP_H600_P40_V18,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TAKES_WEIGHT_H600_P40_V19,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT30_H600_P40_V20,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_TIGHT_POS_H600_P40_V21,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_BOUNDED_GRIP_H600_P40_V22,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_ONLY_H600_P40_V23,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_OFFAXIS10_H600_P40_V24,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_S0_FOCUS_V4,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_SEQUENTIAL_V5,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_ORIENTATION_SEQUENTIAL_V6,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_FULL_POSE_EARLIEST_V7,
    }


def unified_mppi_effort_requires_tool_mediation_success(profile: Any) -> bool:
    """Whether measured terminal success also requires exact mechanism evidence."""
    if profile is None:
        return False
    return validate_unified_mppi_effort_profile(profile) in {
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_V1,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_V2,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H300_P20,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H600_P40,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_H600_P40_V3,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_MEDIATION4_H600_P40_V4,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_DURATION4_H600_P40_V5,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_DURATION4_TF0_H600_P40_V6,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_6CM_RL10_H600_P40_V14,
    }


def unified_mppi_effort_table_force_weight(profile: Any) -> float:
    """Return the registered physical table-force penalty for the carrier."""
    return (
        0.1
        if unified_mppi_effort_uses_proven_pick_carrier(profile)
        else 0.0
    )


def unified_mppi_effort_cost_shaping_profile(profile: Any) -> str | None:
    """Return the sole categorical cost delta for a unified profile."""
    profile = validate_unified_mppi_effort_profile(profile)
    if profile in _GENERATED_PROGRAM_PROFILES:
        return os.environ.get("OAP_EXEC_SHAPING_OVERRIDE") or "typed_residual_library_v3"
    if profile in {UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_AT_REST_H600_P40_V12, UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_S0_ATT20_H600_P40_V13}:
        return PICK_V8_AT_REST_TERMINAL_V6
    if profile in {
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_H600_P40_V12,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_S2_H600_P40_V13,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_H600_P40_V14,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_H600_P40_V15,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_ALIGN8_H600_P40_V16,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_V7_FUNNEL_AT_REST_H600_P40_V17,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_UNCONSTRAINED_GRASP_H600_P40_V18,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TAKES_WEIGHT_H600_P40_V19,
    }:
        # V12-V19 only.  This profile seals the stage identity
        # ``tilt_held_cup_positive_x_60deg``; V20 onward renamed the tilt stage
        # to ``..._30deg``, so stage_cost_shaping_plan fell through to its
        # per-stage all-1.0 branch and the advertised terminal[0] = 60.0 boost
        # on ObjectHeld was inert on V20-V24 while still being reported in
        # their identity.  Measured: for V20/V21/V22/V23/V24 every stage's
        # multipliers under this profile equal the baseline's, and for
        # V17/V18/V19 they do not.  Declaring baseline is therefore a
        # zero-behaviour change that makes the record honest.
        # cost_shaping_profile_binds() now refuses this class of silent no-op.
        return CUP_SIMPLE_TILT_TERMINAL_HELD_60_V1
    if profile in {
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT30_H600_P40_V20,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_TIGHT_POS_H600_P40_V21,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_BOUNDED_GRIP_H600_P40_V22,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_ONLY_H600_P40_V23,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_OFFAXIS10_H600_P40_V24,
    }:
        return BASELINE_COST_SHAPING
    if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_S1_FOCUS_V4:
        return CUP_V7_S1_RUNNING_POSITION_FOCUS_V1
    if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_S0_FOCUS_V4:
        return FLIP_V1_S0_RUNNING_POSITION_FOCUS_V1
    if (
        profile
        in {
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_DIRECT_RELEASE_H600_P40_V10,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_DIRECT_RELEASE_TF0_H600_P40_V11,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_AT_REST_H600_P40_V12,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_S0_ATT20_H600_P40_V13,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_DIRECT_RELEASE_TF0_H600_P40_V11,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_AT_REST_H600_P40_V12,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_S0_ATT20_H600_P40_V13,
        }
    ):
        return PICK_V8_COMBINED_DIRECT_RELEASE_V3
    if (
        profile
        == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_MEDIATION4_H600_P40_V4
    ):
        return TOOLPUSH_8CM_MEDIATION_WEIGHT_4_V1
    if profile in {
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_DURATION4_H600_P40_V5,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_DURATION4_TF0_H600_P40_V6,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_6CM_RL10_H600_P40_V14,
    }:
        return TOOLPUSH_8CM_MEDIATION_DURATION_WEIGHT_4_V2
    return None


# The four production profiles that generate the deliverable videos.  The
# geom-scoped table-force sensor makes the robot push rather than slam, which
# is slower: the filtered Push run reached 71.740 mm and a 1.020 mm hinge with
# 518 of 520 samples reading exactly 0 N, but exhausted 240 stage cycles.
# Historical profiles keep 240 so their sealed episodes stay comparable.
_PRODUCTION_PROFILES = frozenset({
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_DIRECT_RELEASE_TF0_H600_P40_V11,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_AT_REST_H600_P40_V12,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_S0_ATT20_H600_P40_V13,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PUSH_6CM_ONE_SIDED_H600_P40_V20,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_DURATION4_TF0_H600_P40_V6,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_6CM_RL10_H600_P40_V14,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_H600_P40_V12,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_S2_H600_P40_V13,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_H600_P40_V14,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_H600_P40_V15,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_ALIGN8_H600_P40_V16,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_V7_FUNNEL_AT_REST_H600_P40_V17,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_UNCONSTRAINED_GRASP_H600_P40_V18,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TAKES_WEIGHT_H600_P40_V19,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT30_H600_P40_V20,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_TIGHT_POS_H600_P40_V21,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_BOUNDED_GRIP_H600_P40_V22,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_ONLY_H600_P40_V23,
    UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_OFFAXIS10_H600_P40_V24,
})


def unified_mppi_effort_device_values(profile: Any) -> dict[str, Any]:
    profile, _ablation = split_unified_mppi_effort_ablation(profile)
    profile = validate_unified_mppi_effort_profile(profile)
    values = dict(UNIFIED_MPPI_EFFORT_DEVICE_VALUES)
    explicit_device_seed = evaluation_device_seed()
    if explicit_device_seed is not None:
        values["seed"] = explicit_device_seed
    if unified_mppi_effort_uses_proven_pick_carrier(profile):
        values.pop("temperature")
        values.update({
            "control_profile": "legacy_velocity_effort",
            "execution_prefix_frac": 20.0 / 600.0,
            # Resolve rather than hardcode so a profile that registers a
            # different penalty (v18) actually reaches the device.
            "table_contact_force_weight": (
                unified_mppi_effort_table_force_weight(profile)
            ),
        })
    if profile in {
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_V3,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_V1,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_V2,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H300_P20,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H600_P40,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_S1_FOCUS_V4,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SEQUENTIAL_V5,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_COARSE_FINE_V6,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_ORIENTATION_FINE_V7,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_FINE_EARLIEST_V8,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_V9,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_NO_RUNNING_HELD_V10,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_SOFT_HELD_H600_P40_V11,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_H600_P40_V12,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_S2_H600_P40_V13,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_H600_P40_V14,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_H600_P40_V15,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_ALIGN8_H600_P40_V16,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_V7_FUNNEL_AT_REST_H600_P40_V17,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_UNCONSTRAINED_GRASP_H600_P40_V18,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TAKES_WEIGHT_H600_P40_V19,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT30_H600_P40_V20,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_TIGHT_POS_H600_P40_V21,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_BOUNDED_GRIP_H600_P40_V22,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_ONLY_H600_P40_V23,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_OFFAXIS10_H600_P40_V24,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_S0_FOCUS_V4,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_SEQUENTIAL_V5,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_ORIENTATION_SEQUENTIAL_V6,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_FULL_POSE_EARLIEST_V7,
    }:
        values["total_rollouts"] = 2001
    if profile in _TRADITIONAL_P40_PROFILES:
        values["total_rollouts"] = 2001
    if (
        profile
        == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H300_P20
    ):
        values.update({
            "n_steps": 300,
            "execution_prefix_frac": 20.0 / 300.0,
        })
    if (
        profile
        == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H600_P40
    ):
        values["execution_prefix_frac"] = 40.0 / 600.0
    if (
        profile
        in {
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_SOFT_HELD_H600_P40_V11,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_H600_P40_V12,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_S2_H600_P40_V13,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_H600_P40_V14,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_H600_P40_V15,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_ALIGN8_H600_P40_V16,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_V7_FUNNEL_AT_REST_H600_P40_V17,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_UNCONSTRAINED_GRASP_H600_P40_V18,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TAKES_WEIGHT_H600_P40_V19,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT30_H600_P40_V20,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_TIGHT_POS_H600_P40_V21,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_BOUNDED_GRIP_H600_P40_V22,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_ONLY_H600_P40_V23,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_OFFAXIS10_H600_P40_V24,
        }
    ):
        values["execution_prefix_frac"] = 40.0 / 600.0
    if profile in _TRADITIONAL_P40_PROFILES:
        values["execution_prefix_frac"] = 40.0 / 600.0
    return _apply_ablation(values, _ablation, "device")


def unified_mppi_effort_identity(
    profile: Any,
    *,
    execution_budget_protocol: Any = None,
) -> dict[str, Any]:
    profile, _ablation = split_unified_mppi_effort_ablation(profile)
    profile = validate_unified_mppi_effort_profile(profile)
    identity = dict(UNIFIED_MPPI_EFFORT_IDENTITY)
    explicit_device_seed = evaluation_device_seed()
    if explicit_device_seed is not None:
        identity["device_proposal_seed"] = explicit_device_seed
        identity["proposal_seed_policy"] = "explicit_evaluation_device_seed"
    identity["profile"] = profile
    # Mirror the resolver rather than the module default.  The proven-pick
    # carrier registers a 0.1 table-force penalty that reaches the ranked cost
    # (``unified_mppi_effort_device_values``), so inheriting the 0.0 default
    # here sealed identities contradicting their own planner telemetry.
    # Resolving unconditionally keeps the two in step for profiles added later.
    identity["table_contact_force_weight"] = (
        unified_mppi_effort_table_force_weight(profile)
    )
    if unified_mppi_effort_uses_proven_pick_carrier(profile):
        identity.update({
            "schema": "oap_unified_mppi_effort_uniform20_v2",
            "physical_action_half_span": [0.2] * 7 + [80.0],
            "jaw_latent_decode_n_per_unit": {"open": 1.0, "close": float(__import__("os").environ.get("OAP_JAW_CLOSE_FORCE_N", "80") or 80.0)},  # V11F
            "action_interpolation": "linear_degree_one",
            "jaw_actuator_decode": (
                "asymmetric_continuous_effort_open_1n_close_80n"
            ),
            "adaptive_temperature_policy": "legacy_v8_adaptive_eta_v1",
            "candidate_cost_telemetry": "per_term_enabled",
            "evidence_artifacts": ["episode.json", "gpu_step_trace"],
        })
    if profile in {
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_V3,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_V1,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_V2,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H300_P20,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H600_P40,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_S1_FOCUS_V4,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SEQUENTIAL_V5,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_COARSE_FINE_V6,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_ORIENTATION_FINE_V7,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_FINE_EARLIEST_V8,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_V9,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_NO_RUNNING_HELD_V10,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_SOFT_HELD_H600_P40_V11,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_H600_P40_V12,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_S2_H600_P40_V13,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_H600_P40_V14,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_H600_P40_V15,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_ALIGN8_H600_P40_V16,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_V7_FUNNEL_AT_REST_H600_P40_V17,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_UNCONSTRAINED_GRASP_H600_P40_V18,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TAKES_WEIGHT_H600_P40_V19,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT30_H600_P40_V20,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_TIGHT_POS_H600_P40_V21,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_BOUNDED_GRIP_H600_P40_V22,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_ONLY_H600_P40_V23,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_OFFAXIS10_H600_P40_V24,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_S0_FOCUS_V4,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_SEQUENTIAL_V5,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_ORIENTATION_SEQUENTIAL_V6,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_FULL_POSE_EARLIEST_V7,
    } or profile in _TRADITIONAL_P40_PROFILES:
        identity.update({
            "schema": "oap_unified_mppi_effort_uniform20_n2001_v3",
            "optimization_rollouts": 2000,
            "short_screen_required_telemetry": [
                (
                    "call_sites.trajectory_cost.batch.solve_diagnostics."
                    "joint_feasibility.population"
                ),
                (
                    "call_sites.trajectory_cost.batch.solve_diagnostics."
                    "joint_feasibility.winner_is_joint_feasible"
                ),
                "chunks.terminal_gate_evidence.measured_residuals",
            ],
        })
    if profile in _TRADITIONAL_P40_PROFILES:
        identity.update({
            "schema": "oap_traditional_softmax_h600_p40_v1",
            "horizon_steps": 600,
            "execution_prefix_steps": 40,
            "max_stage_cycles": 240,
            "candidate_selection_mode": "valid_then_total_cost",
            "mppi_execution_mode": "softmax_weighted_mean",
            "task_specific_selector": False,
            "cost_shaping_profile": None,
        })
    if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_H600_P40_V9:
        identity.update({
            "task_scope": "pick_v8_only",
            "task_program_change": "none_reuse_verified_pick_v8",
        })
    if (
        profile
        in {
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_DIRECT_RELEASE_H600_P40_V10,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_DIRECT_RELEASE_TF0_H600_P40_V11,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_AT_REST_H600_P40_V12,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_S0_ATT20_H600_P40_V13,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_DIRECT_RELEASE_TF0_H600_P40_V11,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_AT_REST_H600_P40_V12,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_S0_ATT20_H600_P40_V13,
        }
    ):
        identity.update({
            "task_scope": "pick_v8_only",
            "task_program_change": "none_reuse_verified_pick_v8",
            "cost_shaping_profile": PICK_V8_COMBINED_DIRECT_RELEASE_V3,
            "science_delta": (
                "release_running_zero_min_distance_point_on_line_at_rest"
            ),
        })
    if (
        profile
        == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PUSH_6CM_ONE_SIDED_H600_P40_V20
    ):
        identity.update({
            "schema": "oap_traditional_softmax_push_6cm_one_sided_v20",
            "task_scope": "push_v11_only",
            "task_program_source": "vlm_generated",
            "cost_shaping_profile": unified_mppi_effort_cost_shaping_profile(profile),
        })
    if profile in {
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_H600_P40_V3,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_MEDIATION4_H600_P40_V4,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_DURATION4_H600_P40_V5,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_DURATION4_TF0_H600_P40_V6,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_6CM_RL10_H600_P40_V14,
    }:
        identity.update({
            "task_scope": "toolpush_v1_only",
            "task_program_change": "goal_distance_5cm_to_8cm",
            "scene_bundle_input": (
                "existing_loader_validated_joint_velocity_force_canonical_scene"
            ),
            "scene_roles": {"subject": "eraser", "reference": "spacemouse_box"},
            "episode_success_gate": "terminal_and_tool_mediation_true",
        })
    if (
        profile
        == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_6CM_RL10_H600_P40_V14
    ):
        identity.update({
            "task_program_change": None,
            "task_program_source": "vlm_generated",
            "cost_shaping_profile": unified_mppi_effort_cost_shaping_profile(profile),
        })
    if (
        profile
        == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_MEDIATION4_H600_P40_V4
    ):
        identity.update({
            "task_scope": "toolpush_v1_only",
            "task_program_change": "goal_distance_5cm_to_8cm",
            "scene_bundle_input": (
                "existing_loader_validated_joint_velocity_force_canonical_scene"
            ),
            "scene_roles": {"subject": "eraser", "reference": "spacemouse_box"},
            "episode_success_gate": "terminal_and_tool_mediation_true",
            "cost_shaping_profile": TOOLPUSH_8CM_MEDIATION_WEIGHT_4_V1,
            "science_delta": "running_tool_mediation_multiplier_1_to_4",
        })
    if (
        profile
        == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_DURATION4_H600_P40_V5
    ):
        identity.update({
            "task_scope": "toolpush_v1_only",
            "task_program_change": "goal_distance_5cm_to_8cm",
            "scene_bundle_input": (
                "existing_loader_validated_joint_velocity_force_canonical_scene"
            ),
            "scene_roles": {"subject": "eraser", "reference": "spacemouse_box"},
            "episode_success_gate": "terminal_and_tool_mediation_true",
            "cost_shaping_profile": TOOLPUSH_8CM_MEDIATION_DURATION_WEIGHT_4_V2,
            "science_delta": (
                "running_tool_mediation_contact_existence_to_missing_fraction"
            ),
        })
    if (
        profile
        == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_DURATION4_TF0_H600_P40_V6
    ):
        identity.update({
            "schema": "oap_traditional_softmax_toolpush_tf0_h600_p40_v6",
            "task_scope": "toolpush_v1_only",
            "task_program_change": "goal_distance_5cm_to_8cm",
            "scene_bundle_input": (
                "existing_loader_validated_joint_velocity_force_canonical_scene"
            ),
            "scene_roles": {"subject": "eraser", "reference": "spacemouse_box"},
            "episode_success_gate": "terminal_and_tool_mediation_true",
            "cost_shaping_profile": TOOLPUSH_8CM_MEDIATION_DURATION_WEIGHT_4_V2,
            "science_delta": (
                "carrier_table_contact_force_weight_0.1_to_0.0"
            ),
        })
    if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_S1_FOCUS_V4:
        identity.update({
            "schema": (
                "oap_unified_mppi_effort_uniform20_n2001_"
                "cup_s1_focus_v4"
            ),
            "cost_shaping_profile": CUP_V7_S1_RUNNING_POSITION_FOCUS_V1,
            "task_scope": "cup_065e_v7_only",
            "science_delta": "s1_running_relative_position_multiplier_0",
        })
    if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SEQUENTIAL_V5:
        identity.update({
            "schema": (
                "oap_unified_mppi_effort_uniform20_n2001_"
                "cup_sequential_v5"
            ),
            "task_scope": "cup_065e_v7_only",
            "task_program_change": "split_s1_align_then_recenter",
            "task_program_terminal_final": "unchanged",
        })
    if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_COARSE_FINE_V6:
        identity.update({
            "schema": "oap_unified20_n2001_cup_coarse_fine_v6",
            "task_scope": "cup_065e_v7_only",
            "task_program_change": "coarse_position_gate_then_exact_fine_recenter",
            "coarse_position_terminal_tol_m": 0.14,
            "coarse_gate_evidence": {
                "episode_profile": "cup_sequential_v5_3ec7d7e",
                "first_pass_global_chunk": 303,
            },
            "task_program_terminal_final": "unchanged",
        })
    if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_ORIENTATION_FINE_V7:
        identity.update({
            "schema": "oap_unified20_n2001_cup_orientation_fine_v7",
            "task_scope": "cup_065e_v7_only",
            "task_program_change": (
                "exact_orientation_open_gate_then_exact_fine_recenter"
            ),
            "evidence_episode_sha256": (
                "93dc357e420019556a684cc3c8dd8e2ae6a6467dac2a02178b5b60923b9bb18b"
            ),
            "orientation_open_first_joint_pass_chunk": 436,
            "orientation_open_continuous_pass_through_chunk": 496,
            "tolerance_policy": "reuse_original_exact_no_relaxation",
            "task_program_terminal_final": "unchanged",
        })
    if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_FINE_EARLIEST_V8:
        identity.update({
            "schema": "oap_unified20_cup_fine_earliest_v8",
            "task_scope": "cup_065e_v7_only",
            "task_program_change": "none_reuse_orientation_fine_v7",
            "candidate_selection_mode": "result_terminal_earliest",
            "candidate_selection_scope": (
                "sealed_fine_recenter_side_preapproach_cup_open_only"
            ),
            "selection_order": (
                "endpoint_joint_then_earliest_pose_sample_then_cost_"
                "then_stable_index_exact_singleton"
            ),
            "other_stage_selection": "valid_then_total_cost_softmax_unchanged",
            "natural_failure_evidence": {
                "stage_cycles": 180,
                "total_cycles": 586,
                "population_joint_feasible_candidates": 2708,
                "cycles_with_population_joint_feasible": 111,
                "max_population_joint_feasible_per_cycle": 50,
                "cycles_selected_joint_feasible": 0,
                "refusal_count": 0,
                "hard_failure_count": 0,
            },
            "task_program_terminal_final": "unchanged",
        })
    if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_V9:
        identity.update({
            "schema": "oap_unified20_cup_simple_tilt_v9",
            "task_scope": "cup_065e_v7_only",
            "task_program_change": (
                "replace_eleven_stage_pour_with_four_stage_grasp_lift_tilt"
            ),
            "removed_non_task_costs": [
                "distant_axis_prealignment",
                "coarse_recenter",
                "fine_recenter",
                "dual_axis_alignment",
                "45_degree_checkpoint",
                "settle_at_rest",
            ],
            "minimal_cost_necessity": {
                "approach": (
                    "position_reaches_cup; open_prevents_premature_closure"
                ),
                "grasp": (
                    "same_position_maintains_contact_geometry; "
                    "closed_acquires_hold"
                ),
                "lift": (
                    "held_prevents_drop; closed_preserves_grasp; "
                    "clearance_completes_lift"
                ),
                "tilt": (
                    "held_prevents_drop; closed_preserves_grasp; "
                    "signed_pose_completes_directed_60deg_tilt"
                ),
            },
            "hard_validity": "unchanged",
            "candidate_selection_mode": "valid_then_total_cost",
            "task_program_terminal_final": "held_and_directed_60deg_tilt",
        })
    if (
        profile
        == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_NO_RUNNING_HELD_V10
    ):
        identity.update({
            "schema": "oap_unified20_cup_simple_tilt_v10",
            "task_scope": "cup_065e_v7_only",
            "science_delta": "final_tilt_running_object_held_removed",
            "task_program_change": (
                "v9_four_stage_program_with_final_running_held_removed"
            ),
            "terminal_object_held": "unchanged_required",
            "hard_prefix_held_validity": "unchanged",
            "gripper_closed_running": "unchanged",
            "natural_failure_evidence": {
                "last_measured_orientation_residual": 0.620486,
                "last_measured_object_held": True,
                "consecutive_no_valid_plan_refusals": 31,
                "refusal_reason": "first_fail_prefix_hold",
            },
            "candidate_selection_mode": "valid_then_total_cost",
            "task_program_terminal_final": "held_and_directed_60deg_tilt",
        })
    if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_S0_FOCUS_V4:
        identity.update({
            "schema": (
                "oap_unified_mppi_effort_uniform20_n2001_"
                "flip_s0_focus_v4"
            ),
            "cost_shaping_profile": FLIP_V1_S0_RUNNING_POSITION_FOCUS_V1,
            "task_scope": "flip_v1_only",
            "science_delta": "s0_running_relative_position_multiplier_0",
        })
    if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_SEQUENTIAL_V5:
        identity.update({
            "schema": (
                "oap_unified_mppi_effort_uniform20_n2001_"
                "flip_sequential_v5"
            ),
            "task_scope": "flip_v1_only",
            "task_program_change": "split_s0_position_then_full_pose",
            "task_program_terminal_final": "unchanged",
        })
    if (
        profile
        == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_ORIENTATION_SEQUENTIAL_V6
    ):
        identity.update({
            "schema": (
                "oap_unified_mppi_effort_uniform20_n2001_"
                "flip_orientation_sequential_v6"
            ),
            "task_scope": "flip_v1_only",
            "task_program_change": (
                "insert_exact_world_up_gate_before_full_pose"
            ),
            "task_program_terminal_final": "unchanged",
            "evidence_episode_sha256": (
                "6794789cbdca8c0f036cec91e7bd302142f3c0d2706e742427f3a718cb939155"
            ),
            "world_up_first_pass_chunk": 195,
            "intermediate_orientation_tolerance_deg": 12.0,
            "tolerance_policy": "reuse_original_exact_no_relaxation",
        })
    if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_FULL_POSE_EARLIEST_V7:
        identity.update({
            "schema": "oap_unified20_flip_full_pose_earliest_v7",
            "task_scope": "flip_v1_only",
            "task_program_change": "none_reuse_orientation_sequential_v6",
            "candidate_selection_mode": "result_terminal_earliest",
            "candidate_selection_scope": (
                "sealed_align_robot_proximal_eraser_end_open_full_pose_only"
            ),
            "selection_order": (
                "endpoint_joint_then_earliest_pose_sample_then_cost_"
                "then_stable_index_exact_singleton"
            ),
            "other_stage_selection": "valid_then_total_cost_softmax_unchanged",
            "natural_failure_evidence": {
                "stage_cycles": 180,
                "population_joint_feasible_candidates": 5139,
                "cycles_with_population_joint_feasible": 104,
                "cycles_selected_joint_feasible": 58,
                "measured_joint_pass_cycles": 0,
            },
            "task_program_terminal_final": "unchanged",
        })
    if profile in {
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_V1,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_V2,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H300_P20,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H600_P40,
    }:
        identity.update({
            "schema": "oap_unified20_n2001_toolpush_v1",
            "task_scope": "toolpush_v1_only",
            "task_program_change": "new_canonical_eraser_toolpush_positive_x_5cm",
            "scene_bundle_input": (
                "existing_loader_validated_joint_velocity_force_canonical_scene"
            ),
            "scene_bundle_materializer": None,
            "scene_roles": {"subject": "eraser", "reference": "spacemouse_box"},
            "initial_observation_identity": "sealed_eraser_refined",
            "episode_success_gate": "terminal_and_tool_mediation_true",
            "tool_mediation_evidence": {
                "required_tool_target_contact": True,
                "required_robot_target_contact_frames": 0,
                "missing_evidence": "CANNOT_VERIFY_TOOL_MEDIATION",
            },
            "terminal_success_field": "retained_as_measured_fact",
            "episode_result_evidence_fields": [
                "terminal_success",
                "mechanism_valid",
                "tool_use_success",
                "outcome",
            ],
            "video_render_evidence": (
                "retain_episode_json_gpu_step_trace_and_stage_frames"
            ),
            "cost_shaping_profile": None,
        })
    if (
        profile
        == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_SOFT_HELD_H600_P40_V11
    ):
        identity.update({
            "schema": "oap_cup_simple_tilt_soft_held_h600_p40_v11",
            "task_scope": "cup_065e_v7_only",
            "task_program_change": "none_reuse_cup_simple_tilt_v9",
            "science_delta": (
                "uniform_prefix20_to40_and_soft_running_held_not_hard_prefix_gate"
            ),
            "soft_running_object_held_cost": "enabled_multiplier_1",
            # The profile override that disabled this gate was removed:
            # it changed the admissible set for one task category, which is
            # a task-specific selector.  The sealed v11 episode was ranked
            # with the gate off and is therefore not reproducible at or
            # after that removal; this field records what that run did, not
            # what the code does now.
            "hard_prefix_held_refusal": (
                "disabled_for_soft_semantics_in_the_sealed_v11_episode_only"
            ),
            "terminal_object_held": "unchanged_required",
            "candidate_selection_mode": "valid_then_total_cost",
            "mppi_execution_mode": "softmax_weighted_mean",
            "cost_shaping_profile": None,
            "task_program_terminal_final": "held_and_directed_60deg_tilt",
        })
    if profile in {
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_H600_P40_V12,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_S2_H600_P40_V13,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_H600_P40_V14,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_H600_P40_V15,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_ALIGN8_H600_P40_V16,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_V7_FUNNEL_AT_REST_H600_P40_V17,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_UNCONSTRAINED_GRASP_H600_P40_V18,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TAKES_WEIGHT_H600_P40_V19,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT30_H600_P40_V20,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_TIGHT_POS_H600_P40_V21,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_BOUNDED_GRIP_H600_P40_V22,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_ONLY_H600_P40_V23,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_OFFAXIS10_H600_P40_V24,
    }:
        identity.update({
            "schema": "oap_cup_simple_tilt_held60_h600_p40_v12",
            "task_scope": "cup_065e_v7_only",
            "task_program_change": "none_reuse_cup_simple_tilt_v10",
            # Resolve, never restate: V20 onward renamed the tilt stage to
            # ``..._30deg``, which the held-60 profile does not seal, so a
            # hardcoded constant here reported a boost the compiler never
            # applied.  Taking it from the resolver is what keeps identity
            # and behaviour from drifting apart.
            "cost_shaping_profile": (
                unified_mppi_effort_cost_shaping_profile(profile)
            ),
            "science_delta": (
                "terminal_object_held_multiplier_1_to_60"
            ),
        })
    if (
        profile
        == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_OFFAXIS10_H600_P40_V24
    ):
        # The block above describes v12's delta and would be a false record of
        # a run whose whole point is a NEW running term.  Restate what actually
        # moved, and take cost_shaping_profile from the same constant the
        # resolver returns so identity and resolver cannot drift apart.
        identity.update({
            "task_program_change": (
                "tilt_stage_running_add_axis_parallel_"
                "cup_frame_y_world_y_tol10"
            ),
            "task_program_terminal_final": "unchanged",
            # Resolve, never restate: V20 onward renamed the tilt stage to
            # ``..._30deg``, which the held-60 profile does not seal, so a
            # hardcoded constant here reported a boost the compiler never
            # applied.  Taking it from the resolver is what keeps identity
            # and behaviour from drifting apart.
            "cost_shaping_profile": (
                unified_mppi_effort_cost_shaping_profile(profile)
            ),
            "science_delta": (
                "tilt_off_axis_rotation_priced_separately_from_the_"
                "so3_geodesic_terminal_unchanged"
            ),
        })
    if (
        profile
        == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_V2
    ):
        identity.update({
            "schema": "oap_unified20_n2001_toolpush_budget240_v2",
            "task_scope": "toolpush_v1_only",
            "task_program_change": (
                "new_canonical_eraser_toolpush_positive_x_5cm"
            ),
            "science_delta": "max_stage_cycles_180_to_240",
            "all_other_controller_values": "byte_identical_to_toolpush_v1",
            "cost_shaping_profile": None,
            "evidence": {
                "episode_sha256": (
                    "bc3052b236fb0b2b79f3a8b49bf13854f7ab3fb50c12c9739f06e430e054f833"
                ),
                "s0_cycle_179_position_excess_m": 0.003402996,
                "s0_last_10_slope_m_per_cycle": -2.16e-05,
            },
        })
    if (
        profile
        == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H300_P20
    ):
        identity.update({
            "schema": "oap_toolpush_budget240_h300_p20_v1",
            "task_scope": "toolpush_v1_only",
            "science_delta": "horizon_steps_600_to_300_prefix_steps_20_fixed",
            "max_stage_cycles": 240,
            "horizon_steps": 300,
            "execution_prefix_steps": 20,
            "task_program_terminal_predicates": "unchanged",
            "hard_validity": "unchanged",
            "cost_shaping_profile": None,
        })
    if (
        profile
        == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H600_P40
    ):
        identity.update({
            "schema": "oap_toolpush_budget240_h600_p40_v1",
            "task_scope": "toolpush_v1_only",
            "science_delta": "execution_prefix_steps_20_to_40_horizon600_fixed",
            "max_stage_cycles": 240,
            "horizon_steps": 600,
            "execution_prefix_steps": 40,
            "task_program_terminal_predicates": "unchanged",
            "hard_validity": "unchanged",
            "cost_shaping_profile": None,
        })
    # Same reason as the table-force weight above: b70c6e0 raised the
    # production budget in unified_mppi_effort_values and not here, so every
    # sealed episode carried 240 in its identity block while four other fields
    # and the actual loop used 480.  Resolve it, do not restate it.
    effective_values = apply_execution_budget_protocol(
        unified_mppi_effort_values(profile),
        execution_budget_protocol,
    )
    identity["max_stage_cycles"] = int(effective_values["max_stage_cycles"])
    identity["mppi_cycle_budget_scope"] = str(
        effective_values["mppi_cycle_budget_scope"]
    )
    # An ablated episode must say so, and say what it was ablated from.  A
    # sealed artifact that merely reports fewer samples is indistinguishable
    # from a production run configured differently, which is the confusion the
    # identity block exists to prevent.
    identity["ablation"] = _ablation
    identity["ablation_base_profile"] = profile if _ablation else None
    if _ablation is not None:
        identity["profile"] = unified_mppi_effort_ablated_profile(
            profile, _ablation
        )
        resolved = unified_mppi_effort_values(identity["profile"])
        device = unified_mppi_effort_device_values(identity["profile"])
        for scope in ("values", "device"):
            source = resolved if scope == "values" else device
            for field in UNIFIED_MPPI_EFFORT_ABLATIONS[_ablation].get(scope, {}):
                if field in identity:
                    identity[field] = source[field]
    # Derive the proposal count from the device resolver.  Profiles ending in
    # ``__ablate_n1001`` execute 1000 optimization proposals plus one exact
    # replay; inheriting a base-profile literal here made their artifacts
    # incorrectly self-report 2000 proposals.
    device = unified_mppi_effort_device_values(identity["profile"])
    exact_replays = int(identity["exact_singleton_replay_rollouts"])
    total_rollouts = int(device["total_rollouts"])
    if total_rollouts < exact_replays:
        raise ValueError("device rollout total is smaller than exact replays")
    identity["optimization_rollouts"] = total_rollouts - exact_replays
    identity["total_rollouts"] = total_rollouts
    # Resolve, never restate.  Twenty-seven branches above each spell out a
    # cost_shaping_profile by hand, and three of them were wrong: Pick v13 said
    # pick_v8_combined_direct_release_v3 where the resolver applies
    # pick_v8_at_rest_terminal_v6, and both ToolPush profiles said None -- which
    # reads as "no shaping, every multiplier is 1" -- where the resolver applies
    # a shaping that sets running ToolMediation to 4.  The applied value is
    # whatever unified_mppi_effort_cost_shaping_profile returns, because that is
    # what the CLI passes to the cost compiler, so identity has to be derived
    # from it rather than maintained in parallel with it.
    identity["cost_shaping_profile"] = unified_mppi_effort_cost_shaping_profile(
        profile
    )
    return identity


def require_unified_mppi_effort_device_contract(
    values: dict[str, Any],
    *,
    profile: str = UNIFIED_MPPI_EFFORT_V1,
    flip_v7_full_pose_terminal_earliest_authorized: bool = False,
    cup_v8_fine_terminal_earliest_authorized: bool = False,
) -> None:
    """Fail closed unless one device batch has the frozen shared tuple."""
    # Same distinction the run-level contract needs: EXPECTATIONS come from the
    # requested name (so an ablation is honoured), while identity BRANCHES use
    # the base.  Resolving expectations from the base rejected a correct N=1001
    # batch with "total_rollouts {'actual': 1001, 'required': 2001}".
    requested = validate_unified_mppi_effort_profile_arg(profile)
    profile = validate_unified_mppi_effort_profile(profile)
    runtime_prefer_earliest = values.get("prefer_earliest") is True
    sealed_flip_v7_authorization = bool(
        flip_v7_full_pose_terminal_earliest_authorized
    )
    sealed_cup_v8_authorization = bool(
        cup_v8_fine_terminal_earliest_authorized
    )
    if sealed_flip_v7_authorization and sealed_cup_v8_authorization:
        raise ValueError("terminal-earliest stage authorizations conflict")
    flip_v7_profile = (
        profile
        == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_FULL_POSE_EARLIEST_V7
    )
    cup_v8_profile = (
        profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_FINE_EARLIEST_V8
    )
    if sealed_flip_v7_authorization != (
        flip_v7_profile and runtime_prefer_earliest
    ):
        if sealed_flip_v7_authorization or flip_v7_profile:
            raise ValueError(
                "prefer_earliest=True requires the sealed Flip-v7 full-pose "
                "stage runtime authorization"
            )
    if sealed_cup_v8_authorization != (
        cup_v8_profile and runtime_prefer_earliest
    ):
        if sealed_cup_v8_authorization or cup_v8_profile:
            raise ValueError(
                "prefer_earliest=True requires the sealed Cup-v8 fine stage "
                "runtime authorization"
            )
    if runtime_prefer_earliest and not (
        sealed_flip_v7_authorization or sealed_cup_v8_authorization
    ):
        raise ValueError(
            "prefer_earliest=True requires one sealed stage runtime authorization"
        )
    expected_values = unified_mppi_effort_device_values(requested)
    if sealed_flip_v7_authorization or sealed_cup_v8_authorization:
        expected_values["prefer_earliest"] = True
    mismatches = {
        name: {"actual": values.get(name), "required": expected}
        for name, expected in expected_values.items()
        if not (
            math.isclose(
                float(values.get(name)),
                float(expected),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            if isinstance(expected, float)
            and values.get(name) is not None
            else values.get(name) == expected
        )
    }
    if mismatches:
        raise ValueError(
            "unified_mppi_effort_v1 device mismatch: "
            f"{mismatches!r}"
        )


def require_unified_mppi_effort_input_identity(
    *,
    profile: str = UNIFIED_MPPI_EFFORT_V1,
    task_id: str,
    control_profile: str,
    bundle_manifest: Any,
    initial_observation: dict[str, Any],
    subject_name: str,
    reference_name: str | None,
) -> dict[str, Any]:
    """Verify shared scene inputs while retaining the exact unified authority."""
    profile = validate_unified_mppi_effort_profile(profile)
    if (
        task_id not in UNIFIED_MPPI_EFFORT_TASK_REGISTRY
        and task_id not in UNIFIED_MPPI_EFFORT_TOOLPUSH_TASK_REGISTRY
        and task_id != lab_cup_fixture.TASK_ID
    ):
        raise ValueError(f"unregistered unified MPPI effort task: {task_id!r}")
    _require_unified_profile_task_scope(profile, task_id)
    if (
        profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_S1_FOCUS_V4
        and task_id != "cup_065e_v7"
    ):
        raise ValueError(
            "unified N2001 Cup-S1 focus profile requires canonical cup_065e_v7"
        )
    if (
        profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SEQUENTIAL_V5
        and task_id != "cup_065e_v7"
    ):
        raise ValueError(
            "unified N2001 Cup sequential profile requires canonical cup_065e_v7"
        )
    if (
        profile in {
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_COARSE_FINE_V6,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_ORIENTATION_FINE_V7,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_FINE_EARLIEST_V8,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_V9,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_NO_RUNNING_HELD_V10,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_SOFT_HELD_H600_P40_V11,
        }
        and task_id != "cup_065e_v7"
    ):
        raise ValueError("Cup coarse-fine profile requires canonical cup_065e_v7")
    if (
        profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_S0_FOCUS_V4
        and task_id != "flip_v1"
    ):
        raise ValueError(
            "unified N2001 Flip-S0 focus profile requires canonical flip_v1"
        )
    if (
        profile in {
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_SEQUENTIAL_V5,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_ORIENTATION_SEQUENTIAL_V6,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_FULL_POSE_EARLIEST_V7,
        }
        and task_id != "flip_v1"
    ):
        raise ValueError(
            "unified N2001 Flip sequential profile requires canonical flip_v1"
        )
    from oap.twin.control_profile import (
        require_s0_control_profile_input_identity,
    )

    # Pick v8, ToolPush, and Pick v9 share the same loader-valid scene, initial
    # eraser observation, and eraser/box roles. The legacy input verifier names
    # that scene row as Pick v9; restore the external TaskProgram authority below.
    verifier_task_id = (
        "pick_v9" if task_id in {"pick_v8", "toolpush_v1"} else task_id
    )
    identity = require_s0_control_profile_input_identity(
        control_profile=control_profile,
        joint_velocity_force_task_id=verifier_task_id,
        registered_control_profile=(
            "legacy_velocity_effort"
            if unified_mppi_effort_uses_proven_pick_carrier(profile)
            else None
        ),
        bundle_manifest=bundle_manifest,
        initial_observation=initial_observation,
        subject_name=subject_name,
        reference_name=reference_name,
    ) if task_id != lab_cup_fixture.TASK_ID else lab_cup_fixture.input_identity(
        control_profile=control_profile,
        bundle_manifest=bundle_manifest,
        initial_observation=initial_observation,
        subject_name=subject_name,
        reference_name=reference_name,
    )
    registration = unified_mppi_effort_task_registration(task_id, profile)
    return {
        **dict(identity or {}),
        "unified_mppi_effort_profile": profile,
        "registered_task_id": task_id,
        "program_path": registration["program_path"],
        "program_instruction": registration["instruction"],
        "program_provenance": registration["provenance"],
        **({
            "program_stage_count": registration["stage_count"],
            "program_stage_names": list(registration["stage_names"]),
        } if "stage_count" in registration else {}),
        "planner_table_contact_force_weight": (
            unified_mppi_effort_table_force_weight(profile)
        ),
        "table_force_policy": (
            "pick_proven_legacy_velocity_effort_0.1"
            if unified_mppi_effort_uses_proven_pick_carrier(profile)
            else "registered_common_zero"
        ),
    }


UNIFIED_MPPI_EFFORT_TASK_REGISTRY = copy.deepcopy(
    JOINT_VELOCITY_FORCE_TASK_REGISTRY
)
UNIFIED_MPPI_EFFORT_TASK_REGISTRY["pick_v8"] = (
    UNIFIED_MPPI_EFFORT_TASK_REGISTRY.pop("pick_v9")
)
UNIFIED_MPPI_EFFORT_TASK_REGISTRY["pick_v8"].update({
    "program_path": (
        "experiments/placement/"
        "minimal_eraser_on_box_task_program_paper_mppi.json"
    ),
    "provenance": "reviewed:pick-place-result-focused-v8:2026-08-11",
    "stage_count": 5,
    "stage_names": [
        "approach_eraser_open",
        "grasp_eraser_continuous_width",
        "center_eraser_over_box_held",
        "lower_centered_eraser_to_release_hover",
        "release_eraser",
    ],
})

_PICK_AT_REST_REGISTRATION = copy.deepcopy(
    UNIFIED_MPPI_EFFORT_TASK_REGISTRY["pick_v8"]
)
_PICK_AT_REST_REGISTRATION.update({
    "program_path": (
        "experiments/placement/"
        "minimal_eraser_on_box_task_program_at_rest_v9.json"
    ),
    "provenance": (
        "reviewed:pick-place-result-focused-at-rest-terminal-v9:2026-08-17"
    ),
})
UNIFIED_MPPI_EFFORT_TASK_REGISTRY = {
    task_id: UNIFIED_MPPI_EFFORT_TASK_REGISTRY[task_id]
    for task_id in ("pick_v8", "cup_065e_v7", "push_v11", "flip_v1")
}

_TOOLPUSH_REGISTRATION = copy.deepcopy(
    UNIFIED_MPPI_EFFORT_TASK_REGISTRY["pick_v8"]
)
_TOOLPUSH_REGISTRATION.update({
    "program_path": (
        "experiments/task_programs/eraser_toolpush_box_positive_x_5cm_v1.json"
    ),
    "instruction": (
        "Use the black felt blackboard eraser to push the white product box "
        "at least 5 cm along world +X while keeping the eraser held."
    ),
    "provenance": "reviewed:canonical-eraser-toolpush-positive-x-5cm-v1:2026-08-15",
    "stage_count": 4,
    "stage_names": [
        "approach_eraser_open",
        "grasp_eraser_continuous_effort",
        "acquire_eraser_box_contact_held",
        "push_box_positive_x_5cm_with_eraser_held",
    ],
})
UNIFIED_MPPI_EFFORT_TOOLPUSH_TASK_REGISTRY = {
    "toolpush_v1": _TOOLPUSH_REGISTRATION,
}


# The generated-program task keeps the registered scene and instruction.
_PUSH_6CM_REGISTRATION = copy.deepcopy(UNIFIED_MPPI_EFFORT_TASK_REGISTRY["push_v11"])
_PUSH_6CM_REGISTRATION.update({
    "program_path": "experiments/task_programs/push_box_forward_6cm_nonprehensile.json",
    "instruction": (
        "Push the white box straight along robot-base +X by at least 6 cm, "
        "without grasping, lifting, or rotating it."
    ),
    "provenance": "reviewed:push-6cm-one-sided-unified-0060:2026-09-04",
    "stage_count": 2,
    "stage_names": [
        "acquire_fresh_robot_box_contact",
        "maintain_contact_and_push_box_forward_6cm",
    ],
})

_TOOLPUSH_8CM_REGISTRATION = copy.deepcopy(_TOOLPUSH_REGISTRATION)
_TOOLPUSH_8CM_REGISTRATION.update({
    "program_path": (
        "experiments/task_programs/eraser_toolpush_box_positive_x_8cm_v2.json"
    ),
    "instruction": (
        "Use the black felt blackboard eraser to push the white product box "
        "at least 8 cm along world +X while keeping the eraser held."
    ),
    "provenance": "reviewed:canonical-eraser-toolpush-positive-x-8cm-v2:2026-08-15",
    "stage_names": [
        "approach_eraser_open",
        "grasp_eraser_continuous_effort",
        "acquire_eraser_box_contact_held",
        "push_box_positive_x_8cm_with_eraser_held",
    ],
})

# v13 widens stage 0's approach attitude 10 -> 20 deg.  Same change and same
# evidence as ToolPush v5: the two stage-0 gates are structurally identical, and
# ToolPush's measurement showed the halved pool cannot afford the attitude while
# the position term was actually BETTER than in the run that succeeded.
_PICK_S0_ATT20_REGISTRATION = copy.deepcopy(_PICK_AT_REST_REGISTRATION)
_PICK_S0_ATT20_REGISTRATION.update({
    "program_path": (
        "experiments/placement/"
        "minimal_eraser_on_box_task_program_s0_attitude20_v10.json"
    ),
    "provenance": "reviewed:pick-at-rest-s0-attitude-20deg-v10:2026-08-18",
})

# Metadata shared by the current generated ToolPush program.
_TOOLPUSH_6CM_RL10_REGISTRATION = copy.deepcopy(_TOOLPUSH_REGISTRATION)
_TOOLPUSH_6CM_RL10_REGISTRATION.update({
    "program_path": (
        "experiments/task_programs/"
        "eraser_toolpush_box_positive_x_6cm_running_line10_v10.json"
    ),
    "instruction": (
        "Use the blackboard eraser to push the white product box at "
        "least 6 cm along robot-base +X while keeping the eraser held."
    ),
    "provenance": "reviewed:toolpush-6cm-running-line10-unified-0060:2026-09-04",
    "stage_names": [
        "approach_eraser_open",
        "grasp_eraser_continuous_effort",
        "acquire_eraser_box_contact_held",
        "push_box_positive_x_6cm_with_eraser_held",
    ],
})

_CUP_SEQUENTIAL_REGISTRATION = copy.deepcopy(
    UNIFIED_MPPI_EFFORT_TASK_REGISTRY["cup_065e_v7"]
)
_CUP_SEQUENTIAL_REGISTRATION.update({
    "program_path": (
        "experiments/task_programs/"
        "ycb_065e_cup_side_grasp_pour_60deg_sequential_v8.json"
    ),
    "provenance": "reviewed:ycb065e-sequential-align-recenter-v8:2026-08-15",
    "stage_count": 9,
    "stage_names": [
        "reach_side_preapproach_cup_open",
        "align_side_preapproach_cup_open_orientation",
        "recenter_side_preapproach_cup_open",
        "side_approach_cup_open",
        "side_grasp_cup_body",
        "lift_side_grasped_cup",
        "tilt_cup_toward_positive_x_45deg_checkpoint",
        "pour_cup_toward_positive_x_60deg",
        "settle_cup_at_positive_x_60deg",
    ],
})

_CUP_COARSE_FINE_REGISTRATION = copy.deepcopy(
    UNIFIED_MPPI_EFFORT_TASK_REGISTRY["cup_065e_v7"]
)
_CUP_COARSE_FINE_REGISTRATION.update({
    "program_path": (
        "experiments/task_programs/"
        "ycb_065e_cup_side_grasp_pour_60deg_coarse_fine_v9.json"
    ),
    "provenance": "reviewed:ycb065e-coarse-fine-recenter-c303-v9:2026-08-15",
    "stage_count": 10,
    "stage_names": [
        "reach_side_preapproach_cup_open",
        "align_side_preapproach_cup_open_orientation",
        "coarse_recenter_side_preapproach_cup_open",
        "fine_recenter_side_preapproach_cup_open",
        "side_approach_cup_open",
        "side_grasp_cup_body",
        "lift_side_grasped_cup",
        "tilt_cup_toward_positive_x_45deg_checkpoint",
        "pour_cup_toward_positive_x_60deg",
        "settle_cup_at_positive_x_60deg",
    ],
})

_CUP_ORIENTATION_FINE_REGISTRATION = copy.deepcopy(
    UNIFIED_MPPI_EFFORT_TASK_REGISTRY["cup_065e_v7"]
)
_CUP_ORIENTATION_FINE_REGISTRATION.update({
    "program_path": (
        "experiments/task_programs/"
        "ycb_065e_cup_side_grasp_pour_60deg_orientation_fine_v10.json"
    ),
    "provenance": (
        "reviewed:ycb065e-orientation-fine-joint-c436-v10:2026-08-15"
    ),
    "stage_count": 11,
    "stage_names": [
        "reach_side_preapproach_cup_open",
        "align_side_preapproach_cup_open_orientation",
        "coarse_recenter_side_preapproach_cup_open",
        "orient_fine_recenter_side_preapproach_cup_open",
        "fine_recenter_side_preapproach_cup_open",
        "side_approach_cup_open",
        "side_grasp_cup_body",
        "lift_side_grasped_cup",
        "tilt_cup_toward_positive_x_45deg_checkpoint",
        "pour_cup_toward_positive_x_60deg",
        "settle_cup_at_positive_x_60deg",
    ],
})

_CUP_SIMPLE_TILT_REGISTRATION = copy.deepcopy(
    UNIFIED_MPPI_EFFORT_TASK_REGISTRY["cup_065e_v7"]
)
_CUP_SIMPLE_TILT_REGISTRATION.update({
    "program_path": (
        "experiments/task_programs/"
        "ycb_065e_cup_grasp_lift_tilt_60deg_simple_v11.json"
    ),
    "provenance": (
        "reviewed:ycb065e-simple-grasp-lift-directed-tilt-v11:2026-08-15"
    ),
    "stage_count": 4,
    "stage_names": [
        "approach_cup_open_position_only",
        "grasp_cup_same_position",
        "lift_held_cup_15cm",
        "tilt_held_cup_positive_x_60deg",
    ],
})

_CUP_SIMPLE_TILT_NO_RUNNING_HELD_REGISTRATION = copy.deepcopy(
    UNIFIED_MPPI_EFFORT_TASK_REGISTRY["cup_065e_v7"]
)
_CUP_SIMPLE_TILT_NO_RUNNING_HELD_REGISTRATION.update({
    "program_path": (
        "experiments/task_programs/"
        "ycb_065e_cup_grasp_lift_tilt_60deg_no_running_held_v12.json"
    ),
    "provenance": (
        "reviewed:ycb065e-simple-tilt-no-running-held-v12:2026-08-15"
    ),
    "stage_count": 4,
    "stage_names": [
        "approach_cup_open_position_only",
        "grasp_cup_same_position",
        "lift_held_cup_15cm",
        "tilt_held_cup_positive_x_60deg",
    ],
})

_CUP_SIMPLE_TILT_S2_NO_RUNNING_HELD_REGISTRATION = copy.deepcopy(
    _CUP_SIMPLE_TILT_NO_RUNNING_HELD_REGISTRATION
)
_CUP_SIMPLE_TILT_S2_NO_RUNNING_HELD_REGISTRATION.update({
    "program_path": (
        "experiments/task_programs/"
        "ycb_065e_cup_grasp_lift_tilt_60deg_no_running_held_v13.json"
    ),
    "provenance": (
        "reviewed:ycb065e-simple-tilt-no-running-held-s2-and-s3-v13:2026-08-17"
    ),
})

_CUP_SIDE_GRASP_REGISTRATION = copy.deepcopy(
    _CUP_SIMPLE_TILT_S2_NO_RUNNING_HELD_REGISTRATION
)
_CUP_SIDE_GRASP_REGISTRATION.update({
    "program_path": (
        "experiments/task_programs/"
        "ycb_065e_cup_grasp_lift_tilt_60deg_side_grasp_v14.json"
    ),
    "provenance": (
        "reviewed:ycb065e-simple-tilt-side-grasp-orientation-v14:2026-08-17"
    ),
})

_CUP_SIDE_GRASP_AT_REST_REGISTRATION = copy.deepcopy(
    _CUP_SIDE_GRASP_REGISTRATION
)
_CUP_SIDE_GRASP_AT_REST_REGISTRATION.update({
    "program_path": (
        "experiments/task_programs/"
        "ycb_065e_cup_grasp_lift_tilt_60deg_side_grasp_at_rest_v15.json"
    ),
    "provenance": (
        "reviewed:ycb065e-side-grasp-pregrasp-at-rest-v15:2026-08-17"
    ),
})

# v16 tightens ONE running tolerance inside an unchanged terminal band.  Do not
# bind this program to a cost-shaping profile that treats a running term
# duplicating a terminal term as redundant: the tightened band is the fix, and a
# profile that zeroed it would delete the change silently.
_CUP_SIDE_GRASP_AT_REST_ALIGN8_REGISTRATION = copy.deepcopy(
    _CUP_SIDE_GRASP_AT_REST_REGISTRATION
)
_CUP_SIDE_GRASP_AT_REST_ALIGN8_REGISTRATION.update({
    "program_path": (
        "experiments/task_programs/"
        "ycb_065e_cup_grasp_lift_tilt_60deg_side_grasp_at_rest_align8_v16.json"
    ),
    "provenance": (
        "reviewed:ycb065e-side-grasp-at-rest-s0-running-align8-v16:2026-08-17"
    ),
})

# v17's six-stage identity differs from its four-stage ancestor. The program
# itself is supplied externally when this legacy registration is selected.
_CUP_V7_FUNNEL_AT_REST_REGISTRATION = copy.deepcopy(
    _CUP_SIDE_GRASP_AT_REST_ALIGN8_REGISTRATION
)
_CUP_V7_FUNNEL_AT_REST_REGISTRATION.update({
    "program_path": (
        "experiments/task_programs/ycb_065e_cup_v7_funnel_at_rest_v17.json"
    ),
    "provenance": "reviewed:ycb065e-v7-funnel-plus-at-rest-v17:2026-08-17",
    "stage_count": 6,
    "stage_names": [
        "reach_side_preapproach_cup_open",
        "align_side_preapproach_cup_open",
        "side_approach_cup_open",
        "side_grasp_cup_body",
        "lift_held_cup_15cm",
        "tilt_held_cup_positive_x_60deg",
    ],
})

# v18 drops the prescribed grasp pose entirely: no axis_parallel anywhere, and
# a zero offset with a 40 mm tolerance, so the only pre-grasp requirement is
# presence at the cup.  Four stages again, with new names.
_CUP_UNCONSTRAINED_GRASP_REGISTRATION = copy.deepcopy(
    _CUP_V7_FUNNEL_AT_REST_REGISTRATION
)
_CUP_UNCONSTRAINED_GRASP_REGISTRATION.update({
    "program_path": (
        "experiments/task_programs/ycb_065e_cup_unconstrained_grasp_v18.json"
    ),
    "provenance": "reviewed:ycb065e-unconstrained-grasp-pose-v18:2026-08-17",
    "stage_count": 4,
    "stage_names": [
        "approach_cup_any_pose",
        "grasp_cup_any_pose",
        "lift_held_cup_15cm",
        "tilt_held_cup_positive_x_60deg",
    ],
})

# v19 keeps v18's unprescribed grasp pose and makes the grasp gate test the
# grasp's CAPABILITY: held, upright, and 40 mm off the start.  A rim pivot buys
# 16.27 mm of centre rise at the price of 46.8 deg of tilt, so the elevation and
# the upright term cannot both be satisfied by tipping.
_CUP_GRASP_TAKES_WEIGHT_REGISTRATION = copy.deepcopy(
    _CUP_UNCONSTRAINED_GRASP_REGISTRATION
)
_CUP_GRASP_TAKES_WEIGHT_REGISTRATION.update({
    "program_path": (
        "experiments/task_programs/ycb_065e_cup_grasp_takes_weight_v19.json"
    ),
    "provenance": "reviewed:ycb065e-grasp-must-take-weight-v19:2026-08-17",
    "stage_count": 4,
    "stage_names": [
        "approach_cup_any_pose",
        "grasp_cup_and_take_its_weight",
        "lift_held_cup_15cm",
        "tilt_held_cup_positive_x_60deg",
    ],
})

# v20 drops the 15 cm lift stage and uses the 30 +/- 15 deg task target. The
# tilt stage is RENAMED,
# so it no longer matches the sealed _CUP_SIMPLE_TILT_STAGES entry and its
# terminal ObjectHeld multiplier falls back to 1.0 from 60.  That multiplier
# has never been validated -- v12 and v13 both stalled before reaching the
# stage it applies to -- so this removes an unproven moving part rather than a
# proven one.
_CUP_GRASP_TILT30_REGISTRATION = copy.deepcopy(
    _CUP_GRASP_TAKES_WEIGHT_REGISTRATION
)
_CUP_GRASP_TILT30_REGISTRATION.update({
    "program_path": (
        "experiments/task_programs/ycb_065e_cup_grasp_and_tilt_30deg_v20.json"
    ),
    "instruction": (
        "Approach the larger handleless YCB cup, grasp it, and tilt it about "
        "30 degrees toward robot-base +X while keeping it held."
    ),
    "provenance": "reviewed:ycb065e-grasp-and-tilt-30deg-v20:2026-08-18",
    "stage_count": 3,
    "stage_names": [
        "approach_cup_any_pose",
        "grasp_cup_and_take_its_weight",
        "tilt_held_cup_positive_x_30deg",
    ],
})

# v21 tightens the approach/grasp position back to 12 mm while keeping v20's
# zero attitude constraints.  With 40 mm of slack the optimiser shut the jaw at
# 35.01 mm from the cup axis and dragged the cup 8.05 mm by friction; the cup's
# own radius is 37.64 mm, so a 40 mm ball contains poses from which no grasp
# exists at all.
_CUP_TIGHT_POS_REGISTRATION = copy.deepcopy(_CUP_GRASP_TILT30_REGISTRATION)
_CUP_TIGHT_POS_REGISTRATION.update({
    "program_path": (
        "experiments/task_programs/ycb_065e_cup_tight_pos_free_attitude_v21.json"
    ),
    "provenance": "reviewed:ycb065e-tight-position-free-attitude-v21:2026-08-18",
})

# v22 gives the closed-jaw command an explicit 72 mm width.  v21 cleared the
# grasp gate with the jaw 7.6 mm inside a 75.28 mm cup and the cup slipped
# straight back out: "closed" resolves to a target of 0.0, so the jaw sees a
# 75 mm error and applies full authority.  A bounded 3.3 mm error is a grip.
_CUP_BOUNDED_GRIP_REGISTRATION = copy.deepcopy(_CUP_TIGHT_POS_REGISTRATION)
_CUP_BOUNDED_GRIP_REGISTRATION.update({
    "program_path": (
        "experiments/task_programs/ycb_065e_cup_bounded_grip_v22.json"
    ),
    "provenance": "reviewed:ycb065e-bounded-grip-width-v22:2026-08-18",
})

# v23 is grasp-and-tilt with no lift requirement.  v22's bounded-grip idea was
# refused by the program validator itself (synthesis.py:501-516: neither
# gripper_command nor gripper_state may carry a numeric width_m, because
# open/closed is grounded by robot capability), so the crush cannot be bounded
# from the program and a clean lift cannot be asked for honestly.
_CUP_GRASP_TILT_ONLY_REGISTRATION = copy.deepcopy(_CUP_BOUNDED_GRIP_REGISTRATION)
_CUP_GRASP_TILT_ONLY_REGISTRATION.update({
    "program_path": (
        "experiments/task_programs/ycb_065e_cup_grasp_tilt_only_v23.json"
    ),
    "provenance": "reviewed:ycb065e-grasp-and-tilt-only-v23:2026-08-18",
    "stage_names": [
        "approach_cup_any_pose",
        "grasp_cup_any_pose",
        "tilt_held_cup_positive_x_30deg",
    ],
})

# v24 adds ONE running term to the tilt stage and leaves every terminal alone,
# so acceptance is byte-identical to v23.  v23 ended STAGE_UNSATISFIED:2 with
# the cup held for all 547 cycles.  Stage 2 is chunks 67-546 -- 480 cycles, its
# whole budget -- read off the measured terminal residual types rather than
# assumed.  Over those 480 states the single SO(3) geodesic was nearly constant
# (mean 18.04 deg, std 2.99, minimum 15.50, never inside the 15 deg band) while
# its parts were not: on-axis (+Y) mean 17.15 std 3.72, off-axis mean 12.29 std
# 2.81.  The optimiser bought +Y up to 20.94 deg and paid for it in off-axis,
# sliding along one contour of one ball with neither component priced
# separately.  At the +Y it did reach, the terminal admits an off-axis proxy of
# only 12.03 deg; the best cycle measured 13.86.
# axis_parallel(cup_frame_y, world_y) reads exactly those two off-axis degrees
# of freedom -- rotation about the cup's material +Y leaves cup_frame_y fixed,
# so the term is BLIND to on-axis progress.  It does NOT avoid double-charging
# the geodesic, which is a function of both components; what it avoids is
# adding a second charge for on-axis travel.
# tol_deg 10.0 is bracketed by two measurements, and the angular divisor IS the
# declared tolerance.  Below 8.6850 the worst visited state (17.3700 deg, chunk
# 89) costs more than the ((30-15)/15)^2 = 1.000 it takes to give the tilt back
# and stand the cup up; above 12.0286 the term is flat where the terminal is
# decided.  sqrt(8.6850*12.0286) = 10.2210.  An earlier draft declared 8.0 off
# a chunks-100-546 window that dropped the worst state, and at 8.0 the
# incentive inverts at 8 of the 480 states -- do not re-try it.  At 10.0 the
# worst state costs 0.543 and pressing the off-axis proxy to 10 deg puts the
# geodesic at 13.45 deg with no further +Y progress at all.
_CUP_GRASP_TILT_OFFAXIS10_REGISTRATION = copy.deepcopy(
    _CUP_GRASP_TILT_ONLY_REGISTRATION
)
_CUP_GRASP_TILT_OFFAXIS10_REGISTRATION.update({
    "program_path": (
        "experiments/task_programs/ycb_065e_cup_grasp_tilt_offaxis10_v24.json"
    ),
    "provenance": "reviewed:ycb065e-grasp-tilt-off-axis-10deg-v24:2026-08-18",
})

_CUP_MINIMUM_RELATIVE20_INSTRUCTION = (
    "Approach the larger handleless YCB cup, grasp it, and tilt it by at least "
    "20 degrees in the positive direction about its initial measured material "
    "+Y axis while keeping it held. In this upright scene this is toward "
    "robot-base +X. Measure signed rotation-vector progress relative to the "
    "first recorded measured execution endpoint after initial seating."
)


def _cup_minimum_relative20_registration(
    value: Any, profile: str, registration: dict[str, Any]
) -> dict[str, Any]:
    """Bind the separately registered Cup20 instruction to the same controller.

    This generated-program experiment changes the task goal, not the canonical
    scene, input identities, program filename, or controller parameters.
    """
    if (
        profile
        == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_OFFAXIS10_H600_P40_V24
        and bool(getattr(value, "vlm_generated_program", False))
        and getattr(value, "task", None) == _CUP_MINIMUM_RELATIVE20_INSTRUCTION
    ):
        return {**registration, "instruction": _CUP_MINIMUM_RELATIVE20_INSTRUCTION}
    return registration


_FLIP_SEQUENTIAL_REGISTRATION = copy.deepcopy(
    UNIFIED_MPPI_EFFORT_TASK_REGISTRY["flip_v1"]
)
_FLIP_SEQUENTIAL_REGISTRATION.update({
    "program_path": (
        "experiments/task_programs/"
        "grasp_robot_proximal_eraser_and_rotate_clockwise_"
        "sagittal_180_sequential_v2.json"
    ),
    "provenance": (
        "reviewed:eraser-proximal-sequential-position-pose-v2:2026-08-15"
    ),
    "stage_count": 6,
    "stage_names": [
        "approach_robot_proximal_eraser_end_position_open",
        "align_robot_proximal_eraser_end_open_full_pose",
        "grasp_robot_proximal_eraser_end",
        "lift_held_eraser_to_flip_clearance",
        "rotate_held_eraser_clockwise_to_sagittal_90",
        "rotate_eraser_clockwise_to_sagittal_180_result",
    ],
})

_FLIP_ORIENTATION_SEQUENTIAL_REGISTRATION = copy.deepcopy(
    UNIFIED_MPPI_EFFORT_TASK_REGISTRY["flip_v1"]
)
_FLIP_ORIENTATION_SEQUENTIAL_REGISTRATION.update({
    "program_path": (
        "experiments/task_programs/"
        "grasp_robot_proximal_eraser_and_rotate_clockwise_"
        "sagittal_180_orientation_sequential_v3.json"
    ),
    "provenance": (
        "reviewed:eraser-orientation-sequential-world-up-c195-v3:2026-08-15"
    ),
    "stage_count": 7,
    "stage_names": [
        "approach_robot_proximal_eraser_end_position_open",
        "align_robot_proximal_eraser_end_open_world_up",
        "align_robot_proximal_eraser_end_open_full_pose",
        "grasp_robot_proximal_eraser_end",
        "lift_held_eraser_to_flip_clearance",
        "rotate_held_eraser_clockwise_to_sagittal_90",
        "rotate_eraser_clockwise_to_sagittal_180_result",
    ],
})


def validate_unified_mppi_effort_profile(
    value: Any,
    *,
    allow_none: bool = False,
) -> str | None:
    """Return the sole unified category, or fail closed.

    An ablation name resolves to its base, so every resolver that dispatches on
    profile identity -- program, cost shaping, task id, stage budget -- keeps
    working unchanged.  Only the three functions that return numbers re-apply
    the override, and they split before calling this.
    """
    if value is None and allow_none:
        return None
    value, _ablation = split_unified_mppi_effort_ablation(value)
    if value not in UNIFIED_MPPI_EFFORT_PROFILES:
        raise ValueError(
            "unified_mppi_effort_profile must be one of "
            f"{UNIFIED_MPPI_EFFORT_PROFILES!r}, got {value!r}"
        )
    return value


def validate_unified_mppi_effort_profile_arg(
    value: Any,
    *,
    allow_none: bool = False,
) -> str | None:
    """Validate a profile WITHOUT discarding an ablation suffix.

    ``validate_unified_mppi_effort_profile`` deliberately returns the BASE name,
    because a dozen internal callers use its result for set membership.  That
    makes it the wrong function for an entry point that has to carry the user's
    choice forward: argparse used it as its ``type=`` and every ablated run
    silently executed the base profile.  The first N=1001 batch ran at N=2001
    and its logs say ``2001 rollouts`` -- the request was destroyed at argument
    parsing, before anything downstream could honour it.

    This validates exactly the same way and returns the value unchanged.
    """
    if value is None and allow_none:
        return None
    validate_unified_mppi_effort_profile(value)
    return value


def _require_unified_profile_task_scope(profile: str, task_id: str) -> None:
    """Keep the new tool-use authority mutually exclusive with old tasks."""
    if task_id == lab_cup_fixture.TASK_ID:
        if profile != UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_OFFAXIS10_H600_P40_V24:
            raise ValueError("Lab cup uses only the unchanged Cup v24 controller profile")
        return
    is_toolpush_profile = profile in {
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_V1,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_V2,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H300_P20,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H600_P40,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_H600_P40_V3,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_MEDIATION4_H600_P40_V4,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_DURATION4_H600_P40_V5,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_DURATION4_TF0_H600_P40_V6,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_6CM_RL10_H600_P40_V14,
    }
    is_toolpush_task = task_id == "toolpush_v1"
    if is_toolpush_profile != is_toolpush_task:
        raise ValueError(
            "toolpush_v1 requires its sealed uniform20 N2001 profile and "
            "that profile requires canonical toolpush_v1"
        )
    if profile in {
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_H600_P40_V9,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_DIRECT_RELEASE_H600_P40_V10,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_DIRECT_RELEASE_TF0_H600_P40_V11,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_AT_REST_H600_P40_V12,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_S0_ATT20_H600_P40_V13,
    } and task_id != "pick_v8":
        raise ValueError("Pick H600/p40 profile requires canonical pick_v8")
    is_push_profile = (
        profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PUSH_6CM_ONE_SIDED_H600_P40_V20
    )
    if is_push_profile != (task_id == "push_v11"):
        raise ValueError("Push requires the generated-program Push v20 profile")


def _path_has_suffix(value: Any, suffix: str) -> bool:
    return value is not None and str(value).replace("\\", "/").endswith(suffix)


def unified_mppi_effort_task_registration(
    task_id: str,
    profile: Any,
) -> dict[str, Any]:
    """Return task metadata; program authorities are loaded only by contracts."""
    profile = validate_unified_mppi_effort_profile(profile)
    _require_unified_profile_task_scope(profile, task_id)
    if task_id == lab_cup_fixture.TASK_ID:
        return lab_cup_fixture.registration()
    if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PUSH_6CM_ONE_SIDED_H600_P40_V20:
        return _PUSH_6CM_REGISTRATION
    if profile in {
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_H600_P40_V3,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_MEDIATION4_H600_P40_V4,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_DURATION4_H600_P40_V5,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_DURATION4_TF0_H600_P40_V6,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_6CM_RL10_H600_P40_V14,
    }:
        if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_6CM_RL10_H600_P40_V14:
            return _TOOLPUSH_6CM_RL10_REGISTRATION
        return _TOOLPUSH_8CM_REGISTRATION
    if profile in {
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_V1,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_V2,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H300_P20,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H600_P40,
    }:
        return _TOOLPUSH_REGISTRATION
    if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SEQUENTIAL_V5:
        if task_id != "cup_065e_v7":
            raise ValueError(
                "unified N2001 Cup sequential profile requires canonical "
                "cup_065e_v7"
            )
        return _CUP_SEQUENTIAL_REGISTRATION
    if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_COARSE_FINE_V6:
        if task_id != "cup_065e_v7":
            raise ValueError("Cup coarse-fine profile requires canonical cup_065e_v7")
        return _CUP_COARSE_FINE_REGISTRATION
    if profile in {
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_ORIENTATION_FINE_V7,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_FINE_EARLIEST_V8,
    }:
        if task_id != "cup_065e_v7":
            raise ValueError(
                "Cup orientation-fine profile requires canonical cup_065e_v7"
            )
        return _CUP_ORIENTATION_FINE_REGISTRATION
    if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_V9:
        if task_id != "cup_065e_v7":
            raise ValueError(
                "Cup simple-tilt profile requires canonical cup_065e_v7"
            )
        return _CUP_SIMPLE_TILT_REGISTRATION
    if (
        profile
        in {
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_NO_RUNNING_HELD_V10,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_H600_P40_V12,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_S2_H600_P40_V13,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_H600_P40_V14,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_H600_P40_V15,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_ALIGN8_H600_P40_V16,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_V7_FUNNEL_AT_REST_H600_P40_V17,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_UNCONSTRAINED_GRASP_H600_P40_V18,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TAKES_WEIGHT_H600_P40_V19,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT30_H600_P40_V20,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_TIGHT_POS_H600_P40_V21,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_BOUNDED_GRIP_H600_P40_V22,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_ONLY_H600_P40_V23,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_OFFAXIS10_H600_P40_V24,
        }
    ):
        if task_id != "cup_065e_v7":
            raise ValueError(
                "Cup simple-tilt profile requires canonical cup_065e_v7"
            )
        if (
            profile
            == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_OFFAXIS10_H600_P40_V24
        ):
            return _CUP_GRASP_TILT_OFFAXIS10_REGISTRATION
        if (
            profile
            == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_ONLY_H600_P40_V23
        ):
            return _CUP_GRASP_TILT_ONLY_REGISTRATION
        if (
            profile
            == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_BOUNDED_GRIP_H600_P40_V22
        ):
            return _CUP_BOUNDED_GRIP_REGISTRATION
        if (
            profile
            == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_TIGHT_POS_H600_P40_V21
        ):
            return _CUP_TIGHT_POS_REGISTRATION
        if (
            profile
            == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT30_H600_P40_V20
        ):
            return _CUP_GRASP_TILT30_REGISTRATION
        if (
            profile
            == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TAKES_WEIGHT_H600_P40_V19
        ):
            return _CUP_GRASP_TAKES_WEIGHT_REGISTRATION
        if (
            profile
            == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_UNCONSTRAINED_GRASP_H600_P40_V18
        ):
            return _CUP_UNCONSTRAINED_GRASP_REGISTRATION
        if (
            profile
            == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_V7_FUNNEL_AT_REST_H600_P40_V17
        ):
            return _CUP_V7_FUNNEL_AT_REST_REGISTRATION
        if (
            profile
            == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_ALIGN8_H600_P40_V16
        ):
            return _CUP_SIDE_GRASP_AT_REST_ALIGN8_REGISTRATION
        if (
            profile
            == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_H600_P40_V15
        ):
            return _CUP_SIDE_GRASP_AT_REST_REGISTRATION
        if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_H600_P40_V14:
            return _CUP_SIDE_GRASP_REGISTRATION
        if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_S2_H600_P40_V13:
            return _CUP_SIMPLE_TILT_S2_NO_RUNNING_HELD_REGISTRATION
        return _CUP_SIMPLE_TILT_NO_RUNNING_HELD_REGISTRATION
    if (
        profile
        == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_SOFT_HELD_H600_P40_V11
    ):
        if task_id != "cup_065e_v7":
            raise ValueError(
                "Cup simple-tilt profile requires canonical cup_065e_v7"
            )
        return _CUP_SIMPLE_TILT_REGISTRATION
    if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_SEQUENTIAL_V5:
        if task_id != "flip_v1":
            raise ValueError(
                "unified N2001 Flip sequential profile requires canonical "
                "flip_v1"
            )
        return _FLIP_SEQUENTIAL_REGISTRATION
    if (
        profile
        == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_ORIENTATION_SEQUENTIAL_V6
    ):
        if task_id != "flip_v1":
            raise ValueError(
                "Flip orientation-sequential profile requires canonical flip_v1"
            )
        return _FLIP_ORIENTATION_SEQUENTIAL_REGISTRATION
    if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_FULL_POSE_EARLIEST_V7:
        if task_id != "flip_v1":
            raise ValueError(
                "Flip full-pose-earliest profile requires canonical flip_v1"
            )
        return _FLIP_ORIENTATION_SEQUENTIAL_REGISTRATION
    try:
        if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_S0_ATT20_H600_P40_V13:
            return _PICK_S0_ATT20_REGISTRATION
        if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_AT_REST_H600_P40_V12:
            return _PICK_AT_REST_REGISTRATION
        return UNIFIED_MPPI_EFFORT_TASK_REGISTRY[task_id]
    except KeyError as exc:
        raise ValueError(
            f"unregistered unified MPPI effort task: {task_id!r}"
        ) from exc


def unified_mppi_effort_task_id(value: Any) -> str:
    """Resolve the exact task from its canonical instruction and program."""
    profile = validate_unified_mppi_effort_profile(
        getattr(value, "unified_mppi_effort_profile", None)
    )
    if _path_has_suffix(getattr(value, "program_json", None), lab_cup_fixture.PROGRAM_PATH):
        _require_unified_profile_task_scope(profile, lab_cup_fixture.TASK_ID)
        if getattr(value, "task", None) != lab_cup_fixture.INSTRUCTION:
            raise ValueError("Lab cup requires its separately registered instruction")
        return lab_cup_fixture.TASK_ID
    if profile in {
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PUSH_6CM_ONE_SIDED_H600_P40_V20,
    }:
        registration = _PUSH_6CM_REGISTRATION
        if (
            getattr(value, "task", None) != registration["instruction"]
            or not _path_has_suffix(
                getattr(value, "program_json", None), registration["program_path"]
            )
        ):
            raise ValueError("Push v20 profile requires its registered task authority")
        return "push_v11"
    if profile in {
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_H600_P40_V3,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_MEDIATION4_H600_P40_V4,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_DURATION4_H600_P40_V5,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_8CM_DURATION4_TF0_H600_P40_V6,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_6CM_RL10_H600_P40_V14,
    }:
        registration = (
            _TOOLPUSH_6CM_RL10_REGISTRATION
            if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_6CM_RL10_H600_P40_V14
            else _TOOLPUSH_8CM_REGISTRATION
        )
        if (
            getattr(value, "task", None) != registration["instruction"]
            or not _path_has_suffix(
                getattr(value, "program_json", None), registration["program_path"]
            )
        ):
            raise ValueError("ToolPush 8 cm profile requires its canonical authority")
        return "toolpush_v1"
    if profile in {
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_V1,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_V2,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H300_P20,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_TOOLPUSH_BUDGET240_H600_P40,
    }:
        registration = _TOOLPUSH_REGISTRATION
        if (
            getattr(value, "task", None) != registration["instruction"]
            or not _path_has_suffix(
                getattr(value, "program_json", None),
                registration["program_path"],
            )
        ):
            raise ValueError(
                "toolpush profile requires canonical toolpush_v1 authority"
            )
        return "toolpush_v1"
    if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SEQUENTIAL_V5:
        registration = _CUP_SEQUENTIAL_REGISTRATION
        if (
            getattr(value, "task", None) != registration["instruction"]
            or not _path_has_suffix(
                getattr(value, "program_json", None),
                registration["program_path"],
            )
        ):
            raise ValueError(
                "unified N2001 Cup sequential profile requires canonical "
                "cup_065e_v7"
            )
        return "cup_065e_v7"
    if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_COARSE_FINE_V6:
        registration = _CUP_COARSE_FINE_REGISTRATION
        if (
            getattr(value, "task", None) != registration["instruction"]
            or not _path_has_suffix(
                getattr(value, "program_json", None), registration["program_path"]
            )
        ):
            raise ValueError("Cup coarse-fine profile requires canonical cup_065e_v7")
        return "cup_065e_v7"
    if profile in {
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_ORIENTATION_FINE_V7,
        UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_FINE_EARLIEST_V8,
    }:
        registration = _CUP_ORIENTATION_FINE_REGISTRATION
        if (
            getattr(value, "task", None) != registration["instruction"]
            or not _path_has_suffix(
                getattr(value, "program_json", None), registration["program_path"]
            )
        ):
            raise ValueError(
                "Cup orientation-fine profile requires canonical cup_065e_v7"
            )
        return "cup_065e_v7"
    if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_V9:
        registration = _CUP_SIMPLE_TILT_REGISTRATION
        if (
            getattr(value, "task", None) != registration["instruction"]
            or not _path_has_suffix(
                getattr(value, "program_json", None),
                registration["program_path"],
            )
        ):
            raise ValueError(
                "Cup simple-tilt profile requires canonical cup_065e_v7"
            )
        return "cup_065e_v7"
    if (
        profile
        in {
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_NO_RUNNING_HELD_V10,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_H600_P40_V12,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_S2_H600_P40_V13,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_H600_P40_V14,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_H600_P40_V15,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_ALIGN8_H600_P40_V16,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_V7_FUNNEL_AT_REST_H600_P40_V17,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_UNCONSTRAINED_GRASP_H600_P40_V18,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TAKES_WEIGHT_H600_P40_V19,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT30_H600_P40_V20,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_TIGHT_POS_H600_P40_V21,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_BOUNDED_GRIP_H600_P40_V22,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_ONLY_H600_P40_V23,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_OFFAXIS10_H600_P40_V24,
        }
    ):
        registration = (
            _CUP_GRASP_TILT_OFFAXIS10_REGISTRATION
            if profile
            == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_OFFAXIS10_H600_P40_V24
            else _CUP_GRASP_TILT_ONLY_REGISTRATION
            if profile
            == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT_ONLY_H600_P40_V23
            else _CUP_BOUNDED_GRIP_REGISTRATION
            if profile
            == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_BOUNDED_GRIP_H600_P40_V22
            else _CUP_TIGHT_POS_REGISTRATION
            if profile
            == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_TIGHT_POS_H600_P40_V21
            else _CUP_GRASP_TILT30_REGISTRATION
            if profile
            == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TILT30_H600_P40_V20
            else _CUP_GRASP_TAKES_WEIGHT_REGISTRATION
            if profile
            == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_GRASP_TAKES_WEIGHT_H600_P40_V19
            else _CUP_UNCONSTRAINED_GRASP_REGISTRATION
            if profile
            == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_UNCONSTRAINED_GRASP_H600_P40_V18
            else _CUP_V7_FUNNEL_AT_REST_REGISTRATION
            if profile
            == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_V7_FUNNEL_AT_REST_H600_P40_V17
            else _CUP_SIDE_GRASP_AT_REST_ALIGN8_REGISTRATION
            if profile
            == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_ALIGN8_H600_P40_V16
            else _CUP_SIDE_GRASP_AT_REST_REGISTRATION
            if profile
            == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_AT_REST_H600_P40_V15
            else _CUP_SIDE_GRASP_REGISTRATION
            if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIDE_GRASP_H600_P40_V14
            else _CUP_SIMPLE_TILT_S2_NO_RUNNING_HELD_REGISTRATION
            if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_HELD60_S2_H600_P40_V13
            else _CUP_SIMPLE_TILT_NO_RUNNING_HELD_REGISTRATION
        )
        registration = _cup_minimum_relative20_registration(
            value, profile, registration
        )
        if (
            getattr(value, "task", None) != registration["instruction"]
            or not _path_has_suffix(
                getattr(value, "program_json", None),
                registration["program_path"],
            )
        ):
            raise ValueError(
                "Cup simple-tilt profile requires canonical cup_065e_v7"
            )
        return "cup_065e_v7"
    if (
        profile
        == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_SOFT_HELD_H600_P40_V11
    ):
        registration = _CUP_SIMPLE_TILT_REGISTRATION
        if (
            getattr(value, "task", None) != registration["instruction"]
            or not _path_has_suffix(
                getattr(value, "program_json", None),
                registration["program_path"],
            )
        ):
            raise ValueError(
                "Cup simple-tilt profile requires canonical cup_065e_v7"
            )
        return "cup_065e_v7"
    if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_SEQUENTIAL_V5:
        registration = _FLIP_SEQUENTIAL_REGISTRATION
        if (
            getattr(value, "task", None) != registration["instruction"]
            or not _path_has_suffix(
                getattr(value, "program_json", None),
                registration["program_path"],
            )
        ):
            raise ValueError(
                "unified N2001 Flip sequential profile requires canonical "
                "flip_v1"
            )
        return "flip_v1"
    if (
        profile
        == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_ORIENTATION_SEQUENTIAL_V6
    ):
        registration = _FLIP_ORIENTATION_SEQUENTIAL_REGISTRATION
        if (
            getattr(value, "task", None) != registration["instruction"]
            or not _path_has_suffix(
                getattr(value, "program_json", None),
                registration["program_path"],
            )
        ):
            raise ValueError(
                "Flip orientation-sequential profile requires canonical flip_v1"
            )
        return "flip_v1"
    if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_FULL_POSE_EARLIEST_V7:
        registration = _FLIP_ORIENTATION_SEQUENTIAL_REGISTRATION
        if (
            getattr(value, "task", None) != registration["instruction"]
            or not _path_has_suffix(
                getattr(value, "program_json", None),
                registration["program_path"],
            )
        ):
            raise ValueError(
                "Flip full-pose-earliest profile requires canonical flip_v1"
            )
        return "flip_v1"
    if profile in {UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_AT_REST_H600_P40_V12, UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_S0_ATT20_H600_P40_V13}:
        registration = (
            _PICK_S0_ATT20_REGISTRATION
            if profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_PICK_S0_ATT20_H600_P40_V13
            else _PICK_AT_REST_REGISTRATION
        )
        if (
            getattr(value, "task", None) != registration["instruction"]
            or not _path_has_suffix(
                getattr(value, "program_json", None),
                registration["program_path"],
            )
        ):
            raise ValueError(
                "Pick at-rest profile requires its canonical pick_v8 program"
            )
        return "pick_v8"
    matches = [
        task_id
        for task_id, registration in UNIFIED_MPPI_EFFORT_TASK_REGISTRY.items()
        if getattr(value, "task", None) == registration["instruction"]
        and _path_has_suffix(
            getattr(value, "program_json", None), registration["program_path"]
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            "unified_mppi_effort_v1 requires exactly one canonical task and "
            "TaskProgram authority"
        )
    task_id = matches[0]
    _require_unified_profile_task_scope(profile, task_id)
    return task_id


def require_unified_mppi_effort_contract(
    value: Any,
    *,
    program: TaskProgram | None = None,
) -> str | None:
    """Validate the full shared controller and exact task authority."""
    # Keep the ablation: the numeric comparison below resolves the expected
    # values from THIS name, so stripping it here would check the run against
    # the base profile's numbers and reject its own ablation --
    #   "pool_size=1001 (required 2001)"
    # which is exactly what happened.  Everything after this that branches on
    # profile identity uses ``base``.
    requested = validate_unified_mppi_effort_profile_arg(
        getattr(value, "unified_mppi_effort_profile", None),
        allow_none=True,
    )
    profile = validate_unified_mppi_effort_profile(
        getattr(value, "unified_mppi_effort_profile", None),
        allow_none=True,
    )
    if profile is None:
        return None
    if (
        profile in _GENERATED_PROGRAM_PROFILES
        and not bool(getattr(value, "vlm_generated_program", False))
    ):
        raise ValueError(
            "This controller profile requires a VLM-generated TaskProgram"
        )
    task_id = unified_mppi_effort_task_id(value)
    if (
        profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_S1_FOCUS_V4
        and task_id != "cup_065e_v7"
    ):
        raise ValueError(
            "unified N2001 Cup-S1 focus profile requires canonical cup_065e_v7"
        )
    if (
        profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SEQUENTIAL_V5
        and task_id != "cup_065e_v7"
    ):
        raise ValueError(
            "unified N2001 Cup sequential profile requires canonical cup_065e_v7"
        )
    if (
        profile in {
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_COARSE_FINE_V6,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_ORIENTATION_FINE_V7,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_FINE_EARLIEST_V8,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_V9,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_NO_RUNNING_HELD_V10,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_SIMPLE_TILT_SOFT_HELD_H600_P40_V11,
        }
        and task_id != "cup_065e_v7"
    ):
        raise ValueError("Cup coarse-fine profile requires canonical cup_065e_v7")
    if (
        profile == UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_S0_FOCUS_V4
        and task_id != "flip_v1"
    ):
        raise ValueError(
            "unified N2001 Flip-S0 focus profile requires canonical flip_v1"
        )
    if (
        profile in {
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_SEQUENTIAL_V5,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_ORIENTATION_SEQUENTIAL_V6,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_FULL_POSE_EARLIEST_V7,
        }
        and task_id != "flip_v1"
    ):
        raise ValueError(
            "unified N2001 Flip sequential profile requires canonical flip_v1"
        )
    registration = _cup_minimum_relative20_registration(
        value, profile, unified_mppi_effort_task_registration(task_id, profile)
    )
    expected_values = apply_execution_budget_protocol(
        unified_mppi_effort_values(requested),
        getattr(value, "execution_budget_protocol", None),
    )
    # The episode seed selects an evaluation replicate rather than changing
    # the controller.  ``config_from_args`` supplies the profile default when
    # omitted and records an explicit CLI seed when a study pairs replicates.
    expected_values.pop("seed", None)
    mismatches = [
        f"{name}={getattr(value, name, None)!r} (required {expected!r})"
        for name, expected in expected_values.items()
        if getattr(value, name, None) != expected
    ]
    required_absent = (
        "mppi_stage_execution_prefix_steps",
        "direct_torque_s0_coverage_k",
        "direct_torque_s0_horizon_prefix_arm",
        "joint_velocity_force_task_id",
        "joint_velocity_force_trial_budget",
        "joint_velocity_force_algorithm_arm",
        "joint_velocity_force_dial_arm",
        "joint_velocity_force_phase3_finalist_arm",
        "joint_velocity_force_phase3_seed",
        "control_regularizer_calibration_profile",
    )
    mismatches.extend(
        f"{name} must be absent"
        for name in required_absent
        if getattr(value, name, None) is not None
    )
    expected_control = (
        "legacy_velocity_effort"
        if unified_mppi_effort_uses_proven_pick_carrier(profile)
        else "joint_velocity_force"
    )
    vlm_generated = bool(getattr(value, "vlm_generated_program", False))
    # Generated programs use the typed residual library. The pipeline chooses
    # its revision explicitly; term choice remains with the generated program.
    expected_cost_shaping = (
        (os.environ.get("OAP_EXEC_SHAPING_OVERRIDE")
         or "typed_residual_library_v3")
        if vlm_generated
        else unified_mppi_effort_cost_shaping_profile(profile)
    )
    if getattr(value, "cost_shaping_profile", None) != expected_cost_shaping:
        mismatches.append(
            "cost_shaping_profile must be " f"{expected_cost_shaping!r}"
        )
    if getattr(value, "control_profile", None) != expected_control:
        mismatches.append(
            f"control_profile must be {expected_control!r}"
        )
    table_weight = getattr(value, "table_contact_force_weight", None)
    expected_table_weight = unified_mppi_effort_table_force_weight(profile)
    if table_weight != expected_table_weight:
        mismatches.append(
            "table_contact_force_weight must be "
            f"{expected_table_weight!r}"
        )
    if not bool(getattr(value, "offline", False)):
        mismatches.append("offline must be true")
    if bool(getattr(value, "execute", False)):
        mismatches.append("execute must be false")
    if bool(getattr(value, "prepare_real_execution", False)):
        mismatches.append("prepare_real_execution must be false")
    if getattr(value, "remote_planner_url", None) is not None:
        mismatches.append("remote_planner_url must be absent")
    if not _path_has_suffix(
        getattr(value, "initial_obs_json", None),
        registration["initial_observation_path"],
    ):
        mismatches.append("initial_obs_json differs from canonical authority")
    if program is not None:
        if vlm_generated:
            # The ONE relaxed check. Everything else the seal guarantees --
            # instruction identity, scene, initial observation, the full
            # numeric controller tuple -- still binds above. In exchange the
            # program must SAY it is machine-synthesized: parse_program stamps
            # '<backend>:<model>[:repairN]' provenance on every program it
            # validates, and a reviewed:* (hand-written) program is refused
            # here so this flag cannot smuggle a hand-tuned cost past the
            # authority comparison.
            provenance = str(getattr(program, "provenance", "") or "")
            if not provenance.startswith(("anthropic:", "openrouter:", "openai:")):
                mismatches.append(
                    "vlm_generated_program requires machine-synthesis "
                    f"provenance, got {provenance!r}"
                )
            if (
                getattr(program, "instruction", None)
                != registration["instruction"]
            ):
                mismatches.append(
                    "generated program instruction differs from canonical task"
                )
        else:
            authority_path = external_input_path(
                registration["program_path"], purpose="registered program authority"
            )
            authority = TaskProgram.from_json(
                authority_path.read_text(encoding="utf-8")
            )
            if program.to_dict() != authority.to_dict():
                mismatches.append("TaskProgram differs from canonical authority")
    if mismatches:
        raise ValueError(
            "unified_mppi_effort_v1 contract mismatch: " + "; ".join(mismatches)
        )
    return task_id
