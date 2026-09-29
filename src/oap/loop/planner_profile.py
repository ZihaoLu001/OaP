"""Canonical identity for one sampling-MPC runtime profile.

The profile contains only controller parameters.  It carries no task, object,
stage, waypoint, pose proposal, or analytic initialization.  The exact sampler
distribution and both possible Gaussian scales are identity-bound alongside
the standard controller parameters.  The same values are bound into local
evidence, remote-planner requests, deployment identity, and real-execution
authorization so the process that authorizes a plan cannot silently describe
different sampling than the process that solves it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping

from oap.execution_budget import (
    apply_execution_budget_protocol,
    validate_execution_budget_protocol,
)
from oap.loop.sampling import (
    CANDIDATE_SELECTION_RESULT_TERMINAL_EARLIEST,
    CANDIDATE_SELECTION_VALID_THEN_TOTAL_COST,
    validate_candidate_selection_mode,
)
from oap.program.cost_shaping import (
    PICK_V9_S4_DIRECT_RELEASE_V1,
    validate_cost_shaping_profile,
)
from oap.twin.batched_rollout import (
    MPPI_EXECUTION_BEST_VALID_SAMPLE,
    validate_mppi_execution_mode,
    validate_mppi_temperature,
    validate_arm_velocity_weight,
)
from oap.twin.control_profile import (
    CONTROL_PROFILE_DIRECT_TORQUE_FORCE,
    JOINT_VELOCITY_FORCE_COST_FOCUS_STAGE90_VALUES,
    JOINT_VELOCITY_FORCE_COST_FOCUS_SCREEN30_VALUES,
    JOINT_VELOCITY_FORCE_TRIAL_BUDGET_COST_FOCUS_STAGE90_V1,
    JOINT_VELOCITY_FORCE_TRIAL_BUDGET_COST_FOCUS_SCREEN30_V1,
    JOINT_VELOCITY_FORCE_TRIAL_BUDGET_STAGE30_V1,
    direct_torque_s0_horizon_prefix_values,
    joint_velocity_force_algorithm_arm_values,
    joint_velocity_force_dial_arm_values,
    joint_velocity_force_phase3_finalist_arm_values,
    validate_joint_velocity_force_trial_budget,
    validate_joint_velocity_force_algorithm_arm,
    validate_joint_velocity_force_dial_arm,
    validate_joint_velocity_force_phase3_finalist_arm,
    validate_joint_velocity_force_phase3_seed,
    validate_direct_torque_s0_coverage_k,
    validate_direct_torque_s0_horizon_prefix_arm,
)
from oap.twin.control_regularizer import (
    UNIFIED_CONTROL_REGULARIZER_V1,
    validate_control_regularizer_calibration_profile,
)
from oap.twin.unified_mppi_effort import (
    unified_mppi_effort_cost_shaping_profile,
    unified_mppi_effort_values,
    unified_mppi_effort_table_force_weight,
    require_unified_mppi_effort_contract,
    validate_unified_mppi_effort_profile_arg,
)
from oap.twin.pick_v8_causal import (
    pick_v8_causal_values,
    require_pick_v8_causal_contract,
    validate_pick_v8_causal_profile,
)

__all__ = [
    "FIXED_EFFORT_MPPI_PROFILE",
    "PICK_V9_S4_DIRECT_RELEASE_PLANNER_PROFILE",
    "PlannerProfile",
    "require_fixed_effort_mppi_profile",
    "require_mppi_runtime_profile",
    "require_pick_v9_s4_direct_release_profile",
]

_FIXED_EFFORT_SAMPLER_VARIANTS = {
    "mtp_global_local": frozenset({
        ("sample", MPPI_EXECUTION_BEST_VALID_SAMPLE),
    }),
}


FIXED_EFFORT_MPPI_PROFILE = {
    # Project-wide direct-effort profile.  Six hundred native 2 ms steps give
    # 1.2 s of lookahead. Thirty zero-order-held actions preserve the cited
    # whole-body controller's 40 ms effort-command period. MTP devotes 500
    # proposals to a full-bound global graph, 499 to a 0.02 local Gaussian,
    # one to the incumbent, and OaP reserves one additional rollout for
    # exact singleton execution validation.
    "pool_size": 1001,
    "horizon_steps": 600,
    "num_knots": 30,
    "sigma_fraction": 1.0,
    "sampler_mode": "mtp_global_local",
    "mtp_jaw_mode": "sample",
    "local_sigma_fraction": 0.02,
    "mppi_temperature": 0.1,
    "mppi_execution_mode": "best_valid_sample",
    "candidate_selection_mode": CANDIDATE_SELECTION_RESULT_TERMINAL_EARLIEST,
    "execution_prefix_steps": 50,
    "max_stage_cycles": 360,
    "mppi_cycle_budget_scope": "stage",
}


PICK_V9_S4_DIRECT_RELEASE_PLANNER_PROFILE = {
    # Frozen legacy Pick-v9 controller. The opt-in cost profile changes only
    # the exact reviewed S4 running multipliers; it is not a cost-focus/JVF
    # experiment and must not inherit that controller's numeric tuple.
    "pool_size": 1001,
    "horizon_steps": 600,
    "num_knots": 12,
    "sigma_fraction": 0.06,
    "sampler_mode": "single_scale",
    "mtp_jaw_mode": "preserve_nominal",
    "local_sigma_fraction": 0.02,
    "cem_rounds": 1,
    "cem_elite_fraction": 0.1,
    "cem_min_std_fraction": 0.01,
    "mppi_temperature": 1.0,
    "mppi_execution_mode": "softmax_weighted_mean",
    "candidate_selection_mode": CANDIDATE_SELECTION_RESULT_TERMINAL_EARLIEST,
    "execution_prefix_steps": 50,
    "max_stage_cycles": 180,
    "arm_velocity_weight": 0.1,
}

_PICK_V9_INSTRUCTION = (
    "Place the black felt blackboard eraser on top of the white product box "
    "and release it."
)
_PICK_V9_PROGRAM_SUFFIX = (
    "experiments/placement/"
    "minimal_eraser_on_box_task_program_paper_mppi_v9.json"
)
_PICK_V9_STAGE_PREFIXES = (50, 50, 20, 20, 20, 20)


def _profile_value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return (
            isinstance(actual, Real)
            and not isinstance(actual, bool)
            and math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-12
            )
        )
    return actual == expected


def _normalized_path(value: Any) -> str:
    return str(value).replace("\\", "/")


def _require_pick_v9_direct_release_planner_values(value: Any) -> list[str]:
    mismatches = [
        f"{name}={getattr(value, name, None)!r} (required {expected!r})"
        for name, expected in PICK_V9_S4_DIRECT_RELEASE_PLANNER_PROFILE.items()
        if not _profile_value_matches(getattr(value, name, None), expected)
    ]
    forbidden = (
        "direct_torque_s0_horizon_prefix_arm",
        "table_contact_force_weight",
        "control_regularizer_calibration_profile",
        "joint_velocity_force_trial_budget",
        "joint_velocity_force_algorithm_arm",
        "joint_velocity_force_dial_arm",
        "joint_velocity_force_phase3_finalist_arm",
        "joint_velocity_force_phase3_seed",
    )
    for name in forbidden:
        if getattr(value, name, None) is not None:
            mismatches.append(f"{name} must be absent")
    return mismatches


def require_pick_v9_s4_direct_release_profile(value: Any) -> None:
    """Bind the S4 shaping opt-in to the exact full legacy Pick-v9 run."""
    profile = validate_cost_shaping_profile(
        getattr(value, "cost_shaping_profile", None),
        allow_none=True,
    )
    mismatches: list[str] = []
    if profile != PICK_V9_S4_DIRECT_RELEASE_V1:
        mismatches.append(
            f"cost_shaping_profile={profile!r} (required "
            f"{PICK_V9_S4_DIRECT_RELEASE_V1!r})"
        )
    mismatches.extend(_require_pick_v9_direct_release_planner_values(value))

    runtime_expected = {
        "optimizer": "mppi",
        "mppi_stage_execution_prefix_steps": _PICK_V9_STAGE_PREFIXES,
        "mppi_cycle_budget_scope": "stage",
        "seed": 1014,
        "execution_prefix_fraction": 1.0 / 3.0,
        "record_candidate_cost_telemetry": False,
        "control_profile": None,
        "control_profile_smoke_one_cycle": False,
        "direct_torque_s0_coverage_k": None,
        "joint_velocity_force_task_id": None,
        "offline": True,
        "execute": False,
        "prepare_real_execution": False,
        "remote_planner_url": None,
        "real_table_z": -0.021,
        "sim_tcp_m": 0.19812,
        "sim_tcp_anchor": "site",
    }
    for name, expected in runtime_expected.items():
        actual = getattr(value, name, None)
        if not _profile_value_matches(actual, expected):
            mismatches.append(f"{name}={actual!r} (required {expected!r})")

    if getattr(value, "task", None) != _PICK_V9_INSTRUCTION:
        mismatches.append("task must equal the exact Pick-v9 instruction")
    bundle = getattr(value, "bundle_manifest", None)
    if bundle is None or _normalized_path(bundle).split("/")[-1] != "manifest.json":
        mismatches.append("bundle_manifest must name manifest.json")
    initial = getattr(value, "initial_obs_json", None)
    if initial is None or _normalized_path(initial).split("/")[-1] != (
        "eraser_refined.json"
    ):
        mismatches.append("initial_obs_json must name eraser_refined.json")

    program_path = getattr(value, "program_json", None)
    normalized_program = _normalized_path(program_path)
    if program_path is None or not normalized_program.endswith(
        _PICK_V9_PROGRAM_SUFFIX
    ):
        mismatches.append("program_json must name the exact full Pick-v9 authority")
    else:
        try:
            from oap.program import TaskProgram

            supplied = TaskProgram.from_json(
                Path(program_path).read_text(encoding="utf-8")
            )
            from oap.twin.control_profile import external_input_path
            authority_path = external_input_path(
                _PICK_V9_PROGRAM_SUFFIX, purpose="registered program authority"
            )
            authority = TaskProgram.from_json(
                authority_path.read_text(encoding="utf-8")
            )
            if supplied.to_dict() != authority.to_dict():
                mismatches.append("program differs from exact full Pick-v9 authority")
            if len(supplied.stages) != len(_PICK_V9_STAGE_PREFIXES):
                mismatches.append("program must contain the exact six Pick-v9 stages")
        except (OSError, TypeError, ValueError) as exc:
            mismatches.append(f"program_json is unreadable or invalid: {exc}")

    if mismatches:
        raise ValueError(
            "Pick-v9 direct-release profile mismatch: " + "; ".join(mismatches)
        )


def require_mppi_runtime_profile(value: Any) -> None:
    """Dispatch MPPI validation without treating all shaping as cost-focus."""
    causal_profile = validate_pick_v8_causal_profile(
        getattr(value, "pick_v8_causal_profile", None),
        allow_none=True,
    )
    if causal_profile is not None:
        require_pick_v8_causal_contract(value)
        return
    unified_profile = validate_unified_mppi_effort_profile_arg(
        getattr(value, "unified_mppi_effort_profile", None),
        allow_none=True,
    )
    if unified_profile is not None:
        require_unified_mppi_effort_contract(value)
        return
    profile = validate_cost_shaping_profile(
        getattr(value, "cost_shaping_profile", None),
        allow_none=True,
    )
    from oap.program.cost_shaping import (
        PICK_V8_COMBINED_DIRECT_RELEASE_V3,
        PICK_V8_COMBINED_RELEASE_LATERAL_V5,
    )

    if profile in (
        PICK_V8_COMBINED_DIRECT_RELEASE_V3,
        PICK_V8_COMBINED_RELEASE_LATERAL_V5,
    ):
        raise ValueError(
            f"{profile} cost shaping requires its "
            "sealed Pick-v8 causal category"
        )
    if profile == PICK_V9_S4_DIRECT_RELEASE_V1:
        require_pick_v9_s4_direct_release_profile(value)
        return
    require_fixed_effort_mppi_profile(value)


def require_fixed_effort_mppi_profile(value: Any) -> None:
    """Refuse any task-specific change to the paper experiment controller."""
    mismatches: list[str] = []
    coverage_k = validate_direct_torque_s0_coverage_k(
        getattr(value, "direct_torque_s0_coverage_k", None),
        allow_none=True,
    )
    matrix_arm = validate_direct_torque_s0_horizon_prefix_arm(
        getattr(value, "direct_torque_s0_horizon_prefix_arm", None),
        allow_none=True,
    )
    algorithm_arm = validate_joint_velocity_force_algorithm_arm(
        getattr(value, "joint_velocity_force_algorithm_arm", None),
        allow_none=True,
    )
    dial_arm = validate_joint_velocity_force_dial_arm(
        getattr(value, "joint_velocity_force_dial_arm", None),
        allow_none=True,
    )
    phase3_arm = validate_joint_velocity_force_phase3_finalist_arm(
        getattr(value, "joint_velocity_force_phase3_finalist_arm", None),
        allow_none=True,
    )
    phase3_seed = validate_joint_velocity_force_phase3_seed(
        getattr(value, "joint_velocity_force_phase3_seed", None),
        allow_none=True,
    )
    if sum(item is not None for item in (algorithm_arm, dial_arm, phase3_arm)) > 1:
        raise ValueError(
            "Phase-A, Phase-B DIAL, and Phase-3 finalist arms are mutually exclusive"
        )
    if phase3_arm is None and phase3_seed is not None:
        raise ValueError("Phase-3 seed requires a Phase-3 finalist arm")
    trial_budget = validate_joint_velocity_force_trial_budget(
        getattr(value, "joint_velocity_force_trial_budget", None),
        allow_none=True,
    )
    cost_focus_stage90 = (
        trial_budget
        == JOINT_VELOCITY_FORCE_TRIAL_BUDGET_COST_FOCUS_STAGE90_V1
    )
    cost_focus_screen30 = (
        trial_budget
        == JOINT_VELOCITY_FORCE_TRIAL_BUDGET_COST_FOCUS_SCREEN30_V1
    )
    cost_focus_values = (
        JOINT_VELOCITY_FORCE_COST_FOCUS_STAGE90_VALUES
        if cost_focus_stage90
        else JOINT_VELOCITY_FORCE_COST_FOCUS_SCREEN30_VALUES
        if cost_focus_screen30
        else {}
    )
    if coverage_k is not None and matrix_arm is not None:
        raise ValueError(
            "coverage K and horizon-prefix matrix are mutually exclusive"
        )
    matrix_values = (
        direct_torque_s0_horizon_prefix_values(matrix_arm)
        if matrix_arm is not None
        else {}
    )
    algorithm_values = (
        joint_velocity_force_algorithm_arm_values(algorithm_arm)
        if algorithm_arm is not None
        else {}
    )
    dial_values = (
        joint_velocity_force_dial_arm_values(dial_arm)
        if dial_arm is not None
        else {}
    )
    phase3_values = (
        joint_velocity_force_phase3_finalist_arm_values(phase3_arm)
        if phase3_arm is not None
        else {}
    )
    for name, expected in FIXED_EFFORT_MPPI_PROFILE.items():
        actual = getattr(value, name, None)
        if name in algorithm_values:
            expected = algorithm_values[name]
        if name in dial_values:
            expected = dial_values[name]
        if name in phase3_values:
            expected = phase3_values[name]
        if (
            cost_focus_values
            and name in cost_focus_values
        ):
            expected = cost_focus_values[name]
        if (
            name == "num_knots"
            and coverage_k is not None
            and getattr(value, "control_profile", None)
            == CONTROL_PROFILE_DIRECT_TORQUE_FORCE
            and getattr(value, "control_profile_smoke_one_cycle", False)
            is False
        ):
            # The registered direct-torque S0 coverage matrix changes exactly
            # K while retaining every other fixed-effort controller value.
            expected = coverage_k
        if (
            name in matrix_values
            and getattr(value, "control_profile", None)
            == CONTROL_PROFILE_DIRECT_TORQUE_FORCE
            and getattr(value, "control_profile_smoke_one_cycle", False)
            is False
        ):
            expected = matrix_values[name]
        if (
            name == "max_stage_cycles"
            and getattr(value, "control_profile", None) is not None
            and getattr(value, "control_profile_smoke_one_cycle", False) is True
        ):
            # The experiment-only smoke changes only the completed-cycle
            # limit.  Its stricter Pick-v9 S0 contract is validated before
            # this general fixed-controller profile check.
            expected = 1
        if (
            name == "max_stage_cycles"
            and trial_budget is not None
        ):
            expected = 90 if cost_focus_stage90 else 30
        if (
            algorithm_arm is None
            and dial_arm is None
            and phase3_arm is None
            and name in {
                "sampler_mode",
                "mtp_jaw_mode",
                "mppi_execution_mode",
            }
        ):
            continue
        if isinstance(expected, float):
            matches = (
                isinstance(actual, Real)
                and not isinstance(actual, bool)
                and math.isclose(
                    float(actual), expected, rel_tol=0.0, abs_tol=1e-12
                )
            )
        else:
            matches = actual == expected
        if not matches:
            mismatches.append(f"{name}={actual!r} (required {expected!r})")
    sampler = getattr(value, "sampler_mode", None)
    jaw_mode = getattr(value, "mtp_jaw_mode", None)
    try:
        execution_mode = validate_mppi_execution_mode(
            getattr(value, "mppi_execution_mode", None)
        )
    except ValueError:
        execution_mode = getattr(value, "mppi_execution_mode", None)
    allowed_pairs = _FIXED_EFFORT_SAMPLER_VARIANTS.get(sampler)
    if any(item is not None for item in (algorithm_arm, dial_arm, phase3_arm)):
        registered_values = (
            algorithm_values
            if algorithm_arm is not None
            else dial_values
            if dial_arm is not None
            else phase3_values
        )
        expected_pair = (
            registered_values["mtp_jaw_mode"],
            MPPI_EXECUTION_BEST_VALID_SAMPLE,
        )
        pair_matches = (
            sampler == registered_values["sampler_mode"]
            and (jaw_mode, execution_mode) == expected_pair
        )
    else:
        pair_matches = (
            allowed_pairs is not None
            and (jaw_mode, execution_mode) in allowed_pairs
        )
    if not pair_matches:
        mismatches.append(
            "sampler/mtp_jaw/mppi_execution variant="
            f"{(sampler, jaw_mode, execution_mode)!r} (required one of the "
            "registered uniform paper-effort variants)"
        )
    if dial_arm is not None:
        # These fields complete the frozen Phase-B planner tuple but are not
        # part of the older project-wide fixed-effort dictionary. Keep this
        # branch DIAL-only so Phase-A and default runtime semantics are
        # unchanged.
        dial_only_expected = {
            "cem_rounds": 1,
            "cem_elite_fraction": 0.1,
            "cem_min_std_fraction": 0.01,
            "arm_velocity_weight": 0.0,
            "table_contact_force_weight": 0.0,
            "control_regularizer_calibration_profile": (
                UNIFIED_CONTROL_REGULARIZER_V1
            ),
            "joint_velocity_force_trial_budget": (
                JOINT_VELOCITY_FORCE_TRIAL_BUDGET_STAGE30_V1
            ),
        }
        for name, expected in dial_only_expected.items():
            actual = getattr(value, name, None)
            if isinstance(expected, float):
                matches = (
                    isinstance(actual, Real)
                    and not isinstance(actual, bool)
                    and math.isclose(
                        float(actual), expected, rel_tol=0.0, abs_tol=1e-12
                    )
                )
            else:
                matches = actual == expected
            if not matches:
                mismatches.append(
                    f"{name}={actual!r} (required {expected!r})"
                )
    if phase3_arm is not None:
        phase3_only_expected = {
            "cem_rounds": 1,
            "cem_elite_fraction": 0.1,
            "cem_min_std_fraction": 0.01,
            "arm_velocity_weight": 0.0,
            "table_contact_force_weight": 0.0,
            "control_regularizer_calibration_profile": (
                UNIFIED_CONTROL_REGULARIZER_V1
            ),
            "joint_velocity_force_trial_budget": None,
            "joint_velocity_force_phase3_seed": phase3_seed,
        }
        for name, expected in phase3_only_expected.items():
            actual = getattr(value, name, None)
            if isinstance(expected, float):
                matches = (
                    isinstance(actual, Real)
                    and not isinstance(actual, bool)
                    and math.isclose(
                        float(actual), expected, rel_tol=0.0, abs_tol=1e-12
                    )
                )
            else:
                matches = actual == expected
            if not matches:
                mismatches.append(
                    f"{name}={actual!r} (required {expected!r})"
                )
    if cost_focus_values:
        cost_focus_expected = {
            name: expected
            for name, expected in cost_focus_values.items()
            if name not in {"seed", "mppi_cycle_budget_scope"}
        }
        cost_focus_expected.update({
            "control_regularizer_calibration_profile": (
                UNIFIED_CONTROL_REGULARIZER_V1
            ),
        })
        for name, expected in cost_focus_expected.items():
            actual = getattr(value, name, None)
            if not (
                math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
                if isinstance(expected, float)
                and isinstance(actual, Real)
                and not isinstance(actual, bool)
                else actual == expected
            ):
                mismatches.append(
                    f"{name}={actual!r} (required {expected!r})"
                )
        if getattr(value, "cost_shaping_profile", None) is None:
            mismatches.append(
                "cost_shaping_profile=None (required explicit registered profile)"
            )
        if any(item is not None for item in (algorithm_arm, dial_arm, phase3_arm)):
            mismatches.append(
                "cost-focus trial forbids Phase-A, Phase-B DIAL, and Phase-3 arms"
            )
    stage_prefixes = getattr(value, "mppi_stage_execution_prefix_steps", None)
    if stage_prefixes is not None:
        mismatches.append(
            "mppi_stage_execution_prefix_steps must be absent; the fixed "
            "execution_prefix_steps applies to every stage"
        )
    if mismatches:
        raise ValueError(
            "fixed effort-MPPI profile mismatch: " + "; ".join(mismatches)
        )


def _integer_at_least(value: Any, minimum: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(
            f"{label} must be an integer >= {minimum}, got {value!r}"
        )
    integer = int(value)
    if integer < minimum:
        raise ValueError(
            f"{label} must be an integer >= {minimum}, got {value!r}"
        )
    return integer


@dataclass(frozen=True)
class PlannerProfile:
    """The complete task-agnostic numerical sampling profile."""

    pool_size: int
    horizon_steps: int
    num_knots: int
    sigma_fraction: float
    execution_prefix_steps: int
    max_stage_cycles: int
    sampler_mode: str = "single_scale"
    local_sigma_fraction: float = 0.02
    cem_rounds: int = 4
    cem_elite_fraction: float = 0.1
    cem_min_std_fraction: float = 0.01
    mtp_jaw_mode: str = "preserve_nominal"
    candidate_selection_mode: str = (
        CANDIDATE_SELECTION_VALID_THEN_TOTAL_COST
    )
    mppi_execution_mode: str = "softmax_weighted_mean"
    mppi_temperature: float = 0.1
    arm_velocity_weight: float = 0.1
    direct_torque_s0_horizon_prefix_arm: str | None = None
    table_contact_force_weight: float | None = None
    control_regularizer_calibration_profile: str | None = None
    joint_velocity_force_trial_budget: str | None = None
    joint_velocity_force_algorithm_arm: str | None = None
    joint_velocity_force_dial_arm: str | None = None
    joint_velocity_force_phase3_finalist_arm: str | None = None
    joint_velocity_force_phase3_seed: int | None = None
    cost_shaping_profile: str | None = None
    unified_mppi_effort_profile: str | None = None
    pick_v8_causal_profile: str | None = None
    execution_budget_protocol: str | None = None
    # Mirrors LoopConfig.vlm_generated_program: a machine-synthesized program
    # runs under the typed residual library, so the planner expects that
    # shaping instead of the profile's sealed resolver value.
    vlm_generated_program: bool = False

    def __post_init__(self) -> None:
        pool_size = _integer_at_least(
            self.pool_size, 2, "pool_size"
        )
        horizon_steps = _integer_at_least(
            self.horizon_steps, 2, "horizon_steps"
        )
        num_knots = _integer_at_least(
            self.num_knots, 2, "num_knots"
        )
        execution_prefix_steps = _integer_at_least(
            self.execution_prefix_steps, 1, "execution_prefix_steps"
        )
        max_stage_cycles = _integer_at_least(
            self.max_stage_cycles, 1, "max_stage_cycles"
        )
        cem_rounds = _integer_at_least(
            self.cem_rounds, 1, "cem_rounds"
        )
        if (
            isinstance(self.sigma_fraction, bool)
            or not isinstance(self.sigma_fraction, Real)
        ):
            raise ValueError(
                "sigma_fraction must be a real number in (0, 1], got "
                f"{self.sigma_fraction!r}"
            )
        sigma_fraction = float(self.sigma_fraction)
        if (
            not math.isfinite(sigma_fraction)
            or not 0.0 < sigma_fraction <= 1.0
        ):
            raise ValueError(
                "sigma_fraction must be finite and in (0, 1], got "
                f"{self.sigma_fraction!r}"
            )
        if (
            not isinstance(self.sampler_mode, str)
            or self.sampler_mode not in {
                "single_scale",
                "two_scale",
                "mtp_global_local",
                "reference_effort_halton",
            }
        ):
            raise ValueError(
                "sampler_mode must be 'single_scale', 'two_scale', "
                "'mtp_global_local', or 'reference_effort_halton', got "
                f"{self.sampler_mode!r}"
            )
        if (
            not isinstance(self.mtp_jaw_mode, str)
            or self.mtp_jaw_mode not in {"preserve_nominal", "sample"}
        ):
            raise ValueError(
                "mtp_jaw_mode must be 'preserve_nominal' or 'sample', got "
                f"{self.mtp_jaw_mode!r}"
            )
        if (
            self.mtp_jaw_mode == "sample"
            and self.sampler_mode != "mtp_global_local"
        ):
            raise ValueError(
                "sampled MTP jaw mode requires "
                "sampler_mode='mtp_global_local'"
            )
        candidate_selection_mode = validate_candidate_selection_mode(
            self.candidate_selection_mode
        )
        mppi_execution_mode = validate_mppi_execution_mode(
            self.mppi_execution_mode
        )
        if (
            isinstance(self.mppi_temperature, bool)
            or not isinstance(self.mppi_temperature, Real)
        ):
            raise ValueError("mppi_temperature must be a real number")
        mppi_temperature = validate_mppi_temperature(self.mppi_temperature)
        arm_velocity_weight = validate_arm_velocity_weight(
            self.arm_velocity_weight
        )
        matrix_arm = validate_direct_torque_s0_horizon_prefix_arm(
            self.direct_torque_s0_horizon_prefix_arm,
            allow_none=True,
        )
        table_force_weight = self.table_contact_force_weight
        regularizer_profile = validate_control_regularizer_calibration_profile(
            self.control_regularizer_calibration_profile,
            allow_none=True,
        )
        trial_budget = validate_joint_velocity_force_trial_budget(
            self.joint_velocity_force_trial_budget,
            allow_none=True,
        )
        algorithm_arm = validate_joint_velocity_force_algorithm_arm(
            self.joint_velocity_force_algorithm_arm,
            allow_none=True,
        )
        dial_arm = validate_joint_velocity_force_dial_arm(
            self.joint_velocity_force_dial_arm,
            allow_none=True,
        )
        phase3_arm = validate_joint_velocity_force_phase3_finalist_arm(
            self.joint_velocity_force_phase3_finalist_arm,
            allow_none=True,
        )
        phase3_seed = validate_joint_velocity_force_phase3_seed(
            self.joint_velocity_force_phase3_seed,
            allow_none=True,
        )
        cost_shaping_profile = validate_cost_shaping_profile(
            self.cost_shaping_profile,
            allow_none=True,
        )
        unified_profile = validate_unified_mppi_effort_profile_arg(
            self.unified_mppi_effort_profile,
            allow_none=True,
        )
        execution_budget_protocol = validate_execution_budget_protocol(
            self.execution_budget_protocol,
            allow_none=True,
        )
        causal_profile = validate_pick_v8_causal_profile(
            self.pick_v8_causal_profile,
            allow_none=True,
        )
        if unified_profile is not None and causal_profile is not None:
            raise ValueError("Pick-v8 causal and unified profiles conflict")
        if causal_profile is not None:
            expected_causal = {
                name: expected
                for name, expected in pick_v8_causal_values(
                    causal_profile
                ).items()
                if hasattr(self, name)
            }
            mismatches = {
                name: {"actual": getattr(self, name), "required": expected}
                for name, expected in expected_causal.items()
                if getattr(self, name) != expected
            }
            if mismatches:
                raise ValueError(
                    "pick_v8_causal planner mismatch: " f"{mismatches!r}"
                )
        if unified_profile is not None:
            resolved_unified = apply_execution_budget_protocol(
                unified_mppi_effort_values(unified_profile),
                execution_budget_protocol,
            )
            expected_unified = {
                name: expected
                for name, expected in resolved_unified.items()
                if hasattr(self, name)
            }
            expected_unified["table_contact_force_weight"] = (
                unified_mppi_effort_table_force_weight(unified_profile)
            )
            mismatches = {
                name: {"actual": getattr(self, name), "required": expected}
                for name, expected in expected_unified.items()
                if getattr(self, name) != expected
            }
            forbidden = (
                "direct_torque_s0_horizon_prefix_arm",
                "control_regularizer_calibration_profile",
                "joint_velocity_force_trial_budget",
                "joint_velocity_force_algorithm_arm",
                "joint_velocity_force_dial_arm",
                "joint_velocity_force_phase3_finalist_arm",
                "joint_velocity_force_phase3_seed",
            )
            mismatches.update({
                name: {"actual": getattr(self, name), "required": None}
                for name in forbidden
                if getattr(self, name) is not None
            })
            expected_cost_shaping = (
                # Executor ablation (Addendum 77): env-gated substitute for the
                # typed library on VLM programs; unset = sealed behaviour.
                (os.environ.get("OAP_EXEC_SHAPING_OVERRIDE")
                 or "typed_residual_library_v3")
                if self.vlm_generated_program
                else unified_mppi_effort_cost_shaping_profile(unified_profile)
            )
            if cost_shaping_profile != expected_cost_shaping:
                mismatches["cost_shaping_profile"] = {
                    "actual": cost_shaping_profile,
                    "required": expected_cost_shaping,
                }
            if mismatches:
                raise ValueError(
                    "unified_mppi_effort_v1 planner mismatch: "
                    f"{mismatches!r}"
                )
        if cost_shaping_profile == PICK_V9_S4_DIRECT_RELEASE_V1:
            direct_release_mismatches = (
                _require_pick_v9_direct_release_planner_values(self)
            )
            if direct_release_mismatches:
                raise ValueError(
                    "Pick-v9 direct-release planner profile mismatch: "
                    + "; ".join(direct_release_mismatches)
                )
        cost_focus_stage90 = (
            trial_budget
            == JOINT_VELOCITY_FORCE_TRIAL_BUDGET_COST_FOCUS_STAGE90_V1
        )
        cost_focus_screen30 = (
            trial_budget
            == JOINT_VELOCITY_FORCE_TRIAL_BUDGET_COST_FOCUS_SCREEN30_V1
        )
        cost_focus_values = (
            JOINT_VELOCITY_FORCE_COST_FOCUS_STAGE90_VALUES
            if cost_focus_stage90
            else JOINT_VELOCITY_FORCE_COST_FOCUS_SCREEN30_VALUES
            if cost_focus_screen30
            else {}
        )
        if sum(
            item is not None for item in (algorithm_arm, dial_arm, phase3_arm)
        ) > 1:
            raise ValueError(
                "Phase-A, Phase-B DIAL, and Phase-3 finalist arms are "
                "mutually exclusive"
            )
        if phase3_arm is None and phase3_seed is not None:
            raise ValueError("Phase-3 seed requires a Phase-3 finalist arm")
        required_trial_cycles = 90 if cost_focus_stage90 else 30
        if (
            trial_budget is not None
            and max_stage_cycles != required_trial_cycles
        ):
            raise ValueError(
                "joint_velocity_force_trial_budget requires max_stage_cycles="
                f"{required_trial_cycles}"
            )
        if (
            trial_budget is not None
            and regularizer_profile not in (
                None,
                UNIFIED_CONTROL_REGULARIZER_V1,
            )
        ):
            raise ValueError(
                "joint_velocity_force_trial_budget forbids the calibration "
                "regularizer profile"
            )
        if algorithm_arm is not None:
            if (
                trial_budget is None
                or regularizer_profile != UNIFIED_CONTROL_REGULARIZER_V1
            ):
                raise ValueError(
                    "joint_velocity_force_algorithm_arm requires formal "
                    "unified regularizer stage30 pilot"
                )
            expected_algorithm = {
                **joint_velocity_force_algorithm_arm_values(algorithm_arm),
                "sigma_fraction": 1.0,
                "mppi_temperature": 0.1,
                "mppi_execution_mode": MPPI_EXECUTION_BEST_VALID_SAMPLE,
                "candidate_selection_mode": (
                    CANDIDATE_SELECTION_RESULT_TERMINAL_EARLIEST
                ),
                "cem_rounds": 1,
                "cem_elite_fraction": 0.1,
                "cem_min_std_fraction": 0.01,
                "max_stage_cycles": 30,
                "arm_velocity_weight": 0.0,
                "table_contact_force_weight": 0.0,
            }
            algorithm_actual = {
                name: getattr(self, name)
                for name in expected_algorithm
            }
            mismatches = {
                name: {"actual": algorithm_actual[name], "required": expected}
                for name, expected in expected_algorithm.items()
                if algorithm_actual[name] != expected
            }
            if mismatches:
                raise ValueError(
                    f"joint_velocity_force algorithm arm {algorithm_arm!r} "
                    f"planner mismatch: {mismatches!r}"
                )
        if dial_arm is not None:
            if (
                trial_budget is None
                or regularizer_profile != UNIFIED_CONTROL_REGULARIZER_V1
                or algorithm_arm is not None
            ):
                raise ValueError(
                    "joint_velocity_force_dial_arm requires formal unified "
                    "regularizer stage30 pilot and forbids Phase-A arms"
                )
            expected_dial = {
                **joint_velocity_force_dial_arm_values(dial_arm),
                "sigma_fraction": 1.0,
                "mppi_temperature": 0.1,
                "mppi_execution_mode": MPPI_EXECUTION_BEST_VALID_SAMPLE,
                "candidate_selection_mode": (
                    CANDIDATE_SELECTION_RESULT_TERMINAL_EARLIEST
                ),
                "cem_rounds": 1,
                "cem_elite_fraction": 0.1,
                "cem_min_std_fraction": 0.01,
                "max_stage_cycles": 30,
                "arm_velocity_weight": 0.0,
                "table_contact_force_weight": 0.0,
            }
            dial_actual = {
                name: getattr(self, name)
                for name in expected_dial
            }
            mismatches = {
                name: {"actual": dial_actual[name], "required": expected}
                for name, expected in expected_dial.items()
                if dial_actual[name] != expected
            }
            if mismatches:
                raise ValueError(
                    f"joint_velocity_force DIAL arm {dial_arm!r} planner "
                    f"mismatch: {mismatches!r}"
                )
        if phase3_arm is not None:
            if (
                phase3_seed is None
                or trial_budget is not None
                or regularizer_profile != UNIFIED_CONTROL_REGULARIZER_V1
            ):
                raise ValueError(
                    "Phase-3 finalist requires formal unified regularizer "
                    "full360 mode and an explicit registered seed"
                )
            expected_phase3 = {
                **joint_velocity_force_phase3_finalist_arm_values(phase3_arm),
                "sigma_fraction": 1.0,
                "mppi_temperature": 0.1,
                "mppi_execution_mode": MPPI_EXECUTION_BEST_VALID_SAMPLE,
                "candidate_selection_mode": (
                    CANDIDATE_SELECTION_RESULT_TERMINAL_EARLIEST
                ),
                "cem_rounds": 1,
                "cem_elite_fraction": 0.1,
                "cem_min_std_fraction": 0.01,
                "max_stage_cycles": 360,
                "arm_velocity_weight": 0.0,
                "table_contact_force_weight": 0.0,
            }
            phase3_actual = {
                name: getattr(self, name)
                for name in expected_phase3
            }
            mismatches = {
                name: {"actual": phase3_actual[name], "required": expected}
                for name, expected in expected_phase3.items()
                if phase3_actual[name] != expected
            }
            if mismatches:
                raise ValueError(
                    f"Phase-3 finalist arm {phase3_arm!r} planner mismatch: "
                    f"{mismatches!r}"
                )
        if cost_focus_values:
            if (
                regularizer_profile != UNIFIED_CONTROL_REGULARIZER_V1
                or cost_shaping_profile is None
                or algorithm_arm is not None
                or dial_arm is not None
                or phase3_arm is not None
                or phase3_seed is not None
            ):
                raise ValueError(
                    "cost-focus trial requires formal unified regularizer, "
                    "an explicit cost profile, and no algorithm/DIAL/Phase-3 "
                    "identity"
                )
            expected_cost_focus = {
                name: expected
                for name, expected in cost_focus_values.items()
                if name not in {"seed", "mppi_cycle_budget_scope"}
            }
            actual_cost_focus = {
                name: getattr(self, name)
                for name in expected_cost_focus
            }
            mismatches = {
                name: {
                    "actual": actual_cost_focus[name],
                    "required": expected,
                }
                for name, expected in expected_cost_focus.items()
                if actual_cost_focus[name] != expected
            }
            if mismatches:
                raise ValueError(
                    f"cost-focus stage90 planner mismatch: {mismatches!r}"
                )
        if table_force_weight is not None:
            if (
                isinstance(table_force_weight, bool)
                or not isinstance(table_force_weight, Real)
                or not math.isfinite(float(table_force_weight))
                or float(table_force_weight) < 0.0
            ):
                raise ValueError(
                    "table_contact_force_weight must be a finite "
                    "non-negative number or None"
                )
            table_force_weight = float(table_force_weight)
        if matrix_arm is not None:
            matrix_values = direct_torque_s0_horizon_prefix_values(
                matrix_arm
            )
            matrix_actual = {
                "horizon_steps": horizon_steps,
                "num_knots": num_knots,
                "execution_prefix_steps": execution_prefix_steps,
            }
            if matrix_actual != matrix_values:
                raise ValueError(
                    f"planner profile horizon-prefix arm {matrix_arm!r} "
                    f"values mismatch: got {matrix_actual!r}, required "
                    f"{matrix_values!r}"
                )
        if (
            candidate_selection_mode
            == CANDIDATE_SELECTION_RESULT_TERMINAL_EARLIEST
            and self.sampler_mode not in {
                "single_scale",
                "reference_effort_halton",
                "mtp_global_local",
            }
        ):
            raise ValueError(
                "result_terminal_earliest requires an MPPI sampler"
            )
        if (
            isinstance(self.local_sigma_fraction, bool)
            or not isinstance(self.local_sigma_fraction, Real)
        ):
            raise ValueError(
                "local_sigma_fraction must be a real number in (0, 1], got "
                f"{self.local_sigma_fraction!r}"
            )
        local_sigma_fraction = float(self.local_sigma_fraction)
        if (
            not math.isfinite(local_sigma_fraction)
            or not 0.0 < local_sigma_fraction <= 1.0
        ):
            raise ValueError(
                "local_sigma_fraction must be finite and in (0, 1], got "
                f"{self.local_sigma_fraction!r}"
            )
        if (
            self.sampler_mode == "two_scale"
            and local_sigma_fraction >= sigma_fraction
        ):
            raise ValueError(
                "two_scale requires local_sigma_fraction < sigma_fraction "
                f"(broad), got {local_sigma_fraction} >= {sigma_fraction}"
            )
        for label, value in (
            ("cem_elite_fraction", self.cem_elite_fraction),
            ("cem_min_std_fraction", self.cem_min_std_fraction),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{label} must be a real number in (0, 1)")
            numeric = float(value)
            if not math.isfinite(numeric) or not 0.0 < numeric < 1.0:
                raise ValueError(f"{label} must be finite and in (0, 1)")
            object.__setattr__(self, label, numeric)
        if pool_size < 2 * cem_rounds or pool_size % cem_rounds != 0:
            raise ValueError(
                "pool_size is the total rollout budget and must be divisible "
                "by cem_rounds with at least two samples per round"
            )
        if execution_prefix_steps > horizon_steps:
            raise ValueError(
                "execution_prefix_steps cannot exceed horizon_steps: "
                f"{execution_prefix_steps} > {horizon_steps}"
            )
        object.__setattr__(self, "pool_size", pool_size)
        object.__setattr__(self, "horizon_steps", horizon_steps)
        object.__setattr__(self, "num_knots", num_knots)
        object.__setattr__(
            self, "sigma_fraction", sigma_fraction
        )
        object.__setattr__(
            self, "local_sigma_fraction", local_sigma_fraction
        )
        object.__setattr__(
            self, "execution_prefix_steps", execution_prefix_steps
        )
        object.__setattr__(
            self, "max_stage_cycles", max_stage_cycles
        )
        object.__setattr__(self, "cem_rounds", cem_rounds)
        object.__setattr__(
            self, "candidate_selection_mode", candidate_selection_mode
        )
        object.__setattr__(self, "mppi_execution_mode", mppi_execution_mode)
        object.__setattr__(self, "mppi_temperature", mppi_temperature)
        object.__setattr__(
            self, "arm_velocity_weight", arm_velocity_weight
        )
        object.__setattr__(
            self, "direct_torque_s0_horizon_prefix_arm", matrix_arm
        )
        object.__setattr__(
            self, "table_contact_force_weight", table_force_weight
        )
        object.__setattr__(
            self,
            "control_regularizer_calibration_profile",
            regularizer_profile,
        )
        object.__setattr__(
            self,
            "joint_velocity_force_trial_budget",
            trial_budget,
        )
        object.__setattr__(
            self,
            "joint_velocity_force_algorithm_arm",
            algorithm_arm,
        )
        object.__setattr__(
            self,
            "joint_velocity_force_dial_arm",
            dial_arm,
        )
        object.__setattr__(
            self,
            "joint_velocity_force_phase3_finalist_arm",
            phase3_arm,
        )
        object.__setattr__(
            self,
            "joint_velocity_force_phase3_seed",
            phase3_seed,
        )
        object.__setattr__(
            self,
            "cost_shaping_profile",
            cost_shaping_profile,
        )
        object.__setattr__(
            self,
            "unified_mppi_effort_profile",
            unified_profile,
        )
        object.__setattr__(
            self,
            "execution_budget_protocol",
            execution_budget_protocol,
        )
        object.__setattr__(self, "pick_v8_causal_profile", causal_profile)

    def to_dict(self) -> dict[str, int | float | str]:
        """Return the canonical exact-profile JSON object."""
        out = asdict(self)
        if out["direct_torque_s0_horizon_prefix_arm"] is None:
            out.pop("direct_torque_s0_horizon_prefix_arm")
        if out["table_contact_force_weight"] is None:
            out.pop("table_contact_force_weight")
        if out["control_regularizer_calibration_profile"] is None:
            out.pop("control_regularizer_calibration_profile")
        if out["joint_velocity_force_trial_budget"] is None:
            out.pop("joint_velocity_force_trial_budget")
        if out["joint_velocity_force_algorithm_arm"] is None:
            out.pop("joint_velocity_force_algorithm_arm")
        if out["joint_velocity_force_dial_arm"] is None:
            out.pop("joint_velocity_force_dial_arm")
        if out["joint_velocity_force_phase3_finalist_arm"] is None:
            out.pop("joint_velocity_force_phase3_finalist_arm")
        if out["joint_velocity_force_phase3_seed"] is None:
            out.pop("joint_velocity_force_phase3_seed")
        if out["cost_shaping_profile"] is None:
            out.pop("cost_shaping_profile")
        if out["unified_mppi_effort_profile"] is None:
            out.pop("unified_mppi_effort_profile")
        if out["pick_v8_causal_profile"] is None:
            out.pop("pick_v8_causal_profile")
        if out["execution_budget_protocol"] is None:
            out.pop("execution_budget_protocol")
        return out

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlannerProfile":
        """Parse an exact profile and reject missing or unknown fields."""
        expected = {
            "pool_size",
            "horizon_steps",
            "num_knots",
            "sigma_fraction",
            "sampler_mode",
            "candidate_selection_mode",
            "mtp_jaw_mode",
            "local_sigma_fraction",
            "cem_rounds",
            "cem_elite_fraction",
            "cem_min_std_fraction",
            "execution_prefix_steps",
            "max_stage_cycles",
            "mppi_execution_mode",
            "mppi_temperature",
            "arm_velocity_weight",
        }
        optional = {
            "direct_torque_s0_horizon_prefix_arm",
            "table_contact_force_weight",
            "control_regularizer_calibration_profile",
            "joint_velocity_force_trial_budget",
            "joint_velocity_force_algorithm_arm",
            "joint_velocity_force_dial_arm",
            "joint_velocity_force_phase3_finalist_arm",
            "joint_velocity_force_phase3_seed",
            "cost_shaping_profile",
            "unified_mppi_effort_profile",
            "pick_v8_causal_profile",
            "execution_budget_protocol",
        }
        actual = set(value)
        if not expected.issubset(actual) or not actual.issubset(expected | optional):
            raise ValueError(
                "planner profile fields mismatch: "
                f"missing={sorted(expected - actual)}, "
                f"unknown={sorted(actual - expected - optional)}"
            )
        kwargs = {name: value[name] for name in expected}
        for name in optional:
            if name in value:
                kwargs[name] = value[name]
        return cls(**kwargs)

    @property
    def sha256(self) -> str:
        """Stable identity of the exact controller profile."""
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
