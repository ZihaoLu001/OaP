"""Fail-closed control-profile adapters and registered offline experiments.

The original path remains the Pick-v9 S0 diagnostic.  The explicit
``joint_velocity_force`` four-task registry additionally permits only four
sealed multi-stage programs with one shared controller/cost configuration.
"""
from __future__ import annotations

import hashlib
import math
import json
import os
from numbers import Real
from pathlib import Path
from typing import Any

#: Restore the pre-registration seal as a REFUSAL rather than a record. Needed
#: to reproduce a result whose claim rests on the seal; off by default, because
#: a scene reconstructed after the camera moved can never match it.
STRICT_SCENE_SEAL_ENV = "OAP_STRICT_SCENE_SEAL"


def external_input_path(relative_path: str, *, purpose: str) -> Path:
    """Resolve a required artifact supplied outside this source-only package."""
    configured_root = (os.environ.get("OAP_INPUT_ROOT") or "").strip()
    if not configured_root:
        raise ValueError(
            f"{purpose} requires OAP_INPUT_ROOT pointing to your external input "
            f"directory; expected {relative_path!r} below it."
        )
    root = Path(configured_root).expanduser().resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{purpose} must remain within OAP_INPUT_ROOT: {relative_path!r}")
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing external {purpose}: {path}. This source-only package does "
            "not include experiment programs, observations, or scene assets."
        )
    return path


def _external_json_object(relative_path: str, *, purpose: str) -> dict[str, Any]:
    path = external_input_path(relative_path, purpose=purpose)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{purpose} must contain a JSON object: {path}")
    return value


def _env_flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _sha256_path(path: Any) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


CONTROL_PROFILE_LEGACY_VELOCITY_WIDTH = "legacy_velocity_width"
CONTROL_PROFILE_LEGACY_VELOCITY_EFFORT = "legacy_velocity_effort"
CONTROL_PROFILE_DIRECT_TORQUE_FORCE = "direct_torque_force"
CONTROL_PROFILE_JOINT_VELOCITY_FORCE = "joint_velocity_force"
CONTROL_PROFILES = (
    CONTROL_PROFILE_LEGACY_VELOCITY_WIDTH,
    CONTROL_PROFILE_LEGACY_VELOCITY_EFFORT,
    CONTROL_PROFILE_DIRECT_TORQUE_FORCE,
    CONTROL_PROFILE_JOINT_VELOCITY_FORCE,
)

JOINT_VELOCITY_FORCE_TASK_IDS = (
    "pick_v9",
    "cup_065e_v7",
    "flip_v1",
    "push_v11",
)
JOINT_VELOCITY_FORCE_TRIAL_BUDGET_STAGE30_V1 = (
    "unified_tournament_stage30_v1"
)
JOINT_VELOCITY_FORCE_TRIAL_BUDGET_COST_FOCUS_STAGE90_V1 = (
    "unified_cost_focus_stage90_v1"
)
JOINT_VELOCITY_FORCE_TRIAL_BUDGET_COST_FOCUS_SCREEN30_V1 = (
    "unified_cost_focus_screen30_v1"
)
JOINT_VELOCITY_FORCE_TRIAL_BUDGETS = (
    JOINT_VELOCITY_FORCE_TRIAL_BUDGET_STAGE30_V1,
    JOINT_VELOCITY_FORCE_TRIAL_BUDGET_COST_FOCUS_STAGE90_V1,
    JOINT_VELOCITY_FORCE_TRIAL_BUDGET_COST_FOCUS_SCREEN30_V1,
)
JOINT_VELOCITY_FORCE_ALGORITHM_ARMS: dict[str, dict[str, Any]] = {
    "mtp_base": {
        "pool_size": 1001,
        "horizon_steps": 600,
        "num_knots": 30,
        "execution_prefix_steps": 50,
        "sampler_mode": "mtp_global_local",
        "mtp_jaw_mode": "sample",
        "local_sigma_fraction": 0.02,
    },
    "mppi_single": {
        "pool_size": 1001,
        "horizon_steps": 600,
        "num_knots": 30,
        "execution_prefix_steps": 50,
        "sampler_mode": "single_scale",
        "mtp_jaw_mode": "preserve_nominal",
        "local_sigma_fraction": 0.02,
    },
    "short_h": {
        "pool_size": 1001,
        "horizon_steps": 300,
        "num_knots": 15,
        "execution_prefix_steps": 50,
        "sampler_mode": "mtp_global_local",
        "mtp_jaw_mode": "sample",
        "local_sigma_fraction": 0.02,
    },
    "long_prefix": {
        "pool_size": 1001,
        "horizon_steps": 600,
        "num_knots": 30,
        "execution_prefix_steps": 100,
        "sampler_mode": "mtp_global_local",
        "mtp_jaw_mode": "sample",
        "local_sigma_fraction": 0.02,
    },
    "n_high": {
        "pool_size": 2001,
        "horizon_steps": 600,
        "num_knots": 30,
        "execution_prefix_steps": 50,
        "sampler_mode": "mtp_global_local",
        "mtp_jaw_mode": "sample",
        "local_sigma_fraction": 0.02,
    },
    "local_wide": {
        "pool_size": 1001,
        "horizon_steps": 600,
        "num_knots": 30,
        "execution_prefix_steps": 50,
        "sampler_mode": "mtp_global_local",
        "mtp_jaw_mode": "sample",
        "local_sigma_fraction": 0.10,
    },
}
DIAL_ANNEALED_R2_EQUAL_BUDGET_V1 = "dial_annealed_r2_equal_budget_v1"
JOINT_VELOCITY_FORCE_DIAL_ARMS = (DIAL_ANNEALED_R2_EQUAL_BUDGET_V1,)
DIAL_MPC_OFFICIAL_COMMIT = "871c84f1cefbe9fa9d83a06b05ab4f0243c1a05c"
_DIAL_ANNEALED_FIXED_VALUES: dict[str, Any] = {
    "pool_size": 1001,
    "horizon_steps": 600,
    "num_knots": 30,
    "execution_prefix_steps": 50,
    "sampler_mode": "single_scale",
    "mtp_jaw_mode": "preserve_nominal",
    "local_sigma_fraction": 0.02,
}
JOINT_VELOCITY_FORCE_PHASE3_FINALIST_ARMS: dict[str, dict[str, Any]] = {
    "n_high": {
        "pool_size": 2001,
        "horizon_steps": 600,
        "num_knots": 30,
        "execution_prefix_steps": 50,
        "sampler_mode": "mtp_global_local",
        "mtp_jaw_mode": "sample",
        "local_sigma_fraction": 0.02,
    },
    "short_h": {
        "pool_size": 1001,
        "horizon_steps": 300,
        "num_knots": 15,
        "execution_prefix_steps": 50,
        "sampler_mode": "mtp_global_local",
        "mtp_jaw_mode": "sample",
        "local_sigma_fraction": 0.02,
    },
}
JOINT_VELOCITY_FORCE_PHASE3_SEEDS = (1200, 1201, 1202)
JOINT_VELOCITY_FORCE_COST_FOCUS_STAGE90_VALUES: dict[str, Any] = {
    "pool_size": 2001,
    "horizon_steps": 600,
    "num_knots": 30,
    "execution_prefix_steps": 50,
    "sampler_mode": "mtp_global_local",
    "mtp_jaw_mode": "sample",
    "local_sigma_fraction": 0.02,
    "sigma_fraction": 1.0,
    "mppi_temperature": 0.1,
    "mppi_execution_mode": "best_valid_sample",
    "candidate_selection_mode": "result_terminal_earliest",
    "cem_rounds": 1,
    "cem_elite_fraction": 0.1,
    "cem_min_std_fraction": 0.01,
    "max_stage_cycles": 90,
    "mppi_cycle_budget_scope": "stage",
    "seed": 1200,
    "arm_velocity_weight": 0.0,
    "table_contact_force_weight": 0.0,
}
JOINT_VELOCITY_FORCE_COST_FOCUS_SCREEN30_VALUES: dict[str, Any] = {
    **JOINT_VELOCITY_FORCE_COST_FOCUS_STAGE90_VALUES,
    "max_stage_cycles": 30,
}
JOINT_VELOCITY_FORCE_COST_FOCUS_TRIAL_BUDGETS = (
    JOINT_VELOCITY_FORCE_TRIAL_BUDGET_COST_FOCUS_STAGE90_V1,
    JOINT_VELOCITY_FORCE_TRIAL_BUDGET_COST_FOCUS_SCREEN30_V1,
)
JOINT_VELOCITY_FORCE_TASK_REGISTRY_BASE_COMMIT = (
    "d0dff2492a87d39d154096d5a635b51f1a9483f4"
)

# Source-only release: these declared authorities are supplied externally.
_SCENE_MANIFEST_AUTHORITY_PATH = "authorities/joint_velocity_force_scene_manifest.json"

_NO_DATASET_SUBJECT_REWRITE = {
    "offline_dataset_subject_source_name": None,
    "offline_dataset_subject_name": None,
    "offline_dataset_subject_label": None,
    "offline_dataset_visual_mesh": None,
    "offline_dataset_size_lwh": None,
    "offline_dataset_mass_kg": None,
    "offline_dataset_rgba": None,
    "offline_dataset_collision_primitive": None,
}

# This is a categorical science registry, not a collection of task knobs.  The
# paths name tracked authority artifacts in the exact source checkout.  Runtime
# paths may be relocated with that checkout, but their parsed contents and all
# scene rewrite values below must remain identical.
JOINT_VELOCITY_FORCE_TASK_REGISTRY: dict[str, dict[str, Any]] = {
    "pick_v9": {
        "program_path": (
            "experiments/placement/"
            "minimal_eraser_on_box_task_program_paper_mppi_v9.json"
        ),
        "initial_observation_path": "eraser/pose/eraser_refined.json",
        "instruction": (
            "Place the blackboard eraser on top of the white "
            "product box and release it."
        ),
        "provenance": (
            "reviewed:pick-place-result-terminal-earliest-contact-settle-v9:"
            "2026-08-12"
        ),
        "stage_count": 6,
        "stage_names": [
            "approach_eraser_open",
            "grasp_eraser_continuous_width",
            "center_eraser_over_box_held",
            "lower_centered_eraser_to_release_hover",
            "release_eraser",
            "settle_eraser_on_box_contact",
        ],
        "subject": "eraser",
        "reference": "spacemouse_box",
        "scene": dict(_NO_DATASET_SUBJECT_REWRITE),
        "initial_observation_identity": "sealed_eraser_refined",
        "manifest_contains_registered_subject": True,
    },
    "cup_065e_v7": {
        "program_path": (
            "experiments/task_programs/"
            "ycb_065e_cup_side_grasp_pour_60deg.json"
        ),
        "initial_observation_path": (
            "experiments/initial_observations/"
            "ycb_065e_cup_side_grasp_offline.json"
        ),
        "instruction": (
            "Approach the larger handleless YCB cup from its side, grasp its "
            "body without entering the rim, lift it 15 cm, and tilt it 60 "
            "degrees toward robot-base +X while keeping it securely held."
        ),
        "provenance": (
            "reviewed:ycb065e-directional-pour-without-height-gate-v7:"
            "2026-08-11"
        ),
        "stage_count": 8,
        "stage_names": [
            "reach_side_preapproach_cup_open",
            "align_side_preapproach_cup_open",
            "side_approach_cup_open",
            "side_grasp_cup_body",
            "lift_side_grasped_cup",
            "tilt_cup_toward_positive_x_45deg_checkpoint",
            "pour_cup_toward_positive_x_60deg",
            "settle_cup_at_positive_x_60deg",
        ],
        "subject": "cup",
        "reference": None,
        "scene": {
            "offline_dataset_subject_source_name": "eraser",
            "offline_dataset_subject_name": "cup",
            "offline_dataset_subject_label": (
                "YCB 065-e larger handleless blue cup"
            ),
            "offline_dataset_visual_mesh": (
                "assets/datasets/ycb_065e_cup/cup_centered.obj"
            ),
            "offline_dataset_size_lwh": (0.075543, 0.075274, 0.070778),
            "offline_dataset_mass_kg": 0.021,
            "offline_dataset_rgba": (0.16, 0.48, 0.90, 1.0),
            "offline_dataset_collision_primitive": "cylinder",
        },
        "initial_observation_identity": "tracked_ycb065e_offline_pose",
        "manifest_contains_registered_subject": False,
    },
    "flip_v1": {
        "program_path": (
            "experiments/task_programs/"
            "grasp_robot_proximal_eraser_and_rotate_clockwise_sagittal_180.json"
        ),
        "initial_observation_path": "eraser/pose/eraser_refined.json",
        "instruction": (
            "Grasp the robot-proximal end of the blackboard eraser, lift it "
            "clear, then rotate it clockwise in the robot-forward sagittal "
            "plane through a signed +90 checkpoint to a signed +180 final "
            "orientation."
        ),
        "provenance": (
            "reviewed:eraser-proximal-grasp-clockwise-sagittal-180-v1:"
            "2026-08-12"
        ),
        "stage_count": 5,
        "stage_names": [
            "approach_robot_proximal_eraser_end_open",
            "grasp_robot_proximal_eraser_end",
            "lift_held_eraser_to_flip_clearance",
            "rotate_held_eraser_clockwise_to_sagittal_90",
            "rotate_eraser_clockwise_to_sagittal_180_result",
        ],
        "subject": "eraser",
        "reference": None,
        "scene": dict(_NO_DATASET_SUBJECT_REWRITE),
        "initial_observation_identity": "sealed_eraser_refined",
        "manifest_contains_registered_subject": True,
    },
    "push_v11": {
        "program_path": (
            "experiments/task_programs/push_box_forward_5cm_nonprehensile.json"
        ),
        "initial_observation_path": (
            "spacemouse_box/pose/spacemouse_box_refined.json"
        ),
        "instruction": (
            "Push the white product box 5 cm forward along robot-base +X "
            "without grasping it, keep it on the table with its initial "
            "orientation, and leave it at rest."
        ),
        "provenance": (
            "reviewed:staged-rsc-mtp-global-local-best-candidate-push-v11:"
            "2026-08-11"
        ),
        "stage_count": 2,
        "stage_names": [
            "acquire_fresh_robot_box_contact",
            "maintain_contact_and_push_box_forward_5cm",
        ],
        "subject": "spacemouse_box",
        "reference": None,
        "scene": dict(_NO_DATASET_SUBJECT_REWRITE),
        "initial_observation_identity": "sealed_spacemouse_box_refined",
        "manifest_contains_registered_subject": True,
    },
}


def control_profile_uses_joint_velocity_arm(value: Any) -> bool:
    """Return whether the profile's seven arm actions are physical rad/s."""
    profile = validate_control_profile(value, allow_none=True)
    return profile in (
        CONTROL_PROFILE_LEGACY_VELOCITY_WIDTH,
        CONTROL_PROFILE_LEGACY_VELOCITY_EFFORT,
        CONTROL_PROFILE_JOINT_VELOCITY_FORCE,
    )


def control_profile_uses_width_jaw(value: Any) -> bool:
    """Return whether the jaw actuator command is a physical width target."""
    return (
        validate_control_profile(value, allow_none=True)
        == CONTROL_PROFILE_LEGACY_VELOCITY_WIDTH
    )

# Registered single-axis coverage matrix for the direct-torque Pick-v9 S0
# diagnostic.  Every value divides the frozen 600-step horizon exactly, so K
# changes only the number/duration of equal ZOH action intervals.  Proposal
# range and per-coordinate local covariance remain fixed below.
DIRECT_TORQUE_S0_COVERAGE_K = (6, 12, 20, 30)

# Registered horizon-by-execution-prefix matrix.  Horizon duration is paired
# with K so every arm retains the same 20 native-step / 40 ms action cadence.
DIRECT_TORQUE_S0_HORIZON_PREFIX_ARMS = {
    "h300_k15_prefix50": {
        "horizon_steps": 300,
        "num_knots": 15,
        "execution_prefix_steps": 50,
    },
    "h300_k15_prefix100": {
        "horizon_steps": 300,
        "num_knots": 15,
        "execution_prefix_steps": 100,
    },
    "h600_k30_prefix50": {
        "horizon_steps": 600,
        "num_knots": 30,
        "execution_prefix_steps": 50,
    },
    "h600_k30_prefix100": {
        "horizon_steps": 600,
        "num_knots": 30,
        "execution_prefix_steps": 100,
    },
}

PICK_V9_INSTRUCTION = (
    "Place the blackboard eraser on top of the white product box "
    "and release it."
)
PICK_V9_PROVENANCE = (
    "reviewed:pick-place-result-terminal-earliest-contact-settle-v9:2026-08-12"
)
_PICK_V9_S0_STAGE_PATH = "authorities/pick_v9_s0_stage.json"
_PICK_INPUT_IDENTITY_PATH = "authorities/pick_input_identity.json"

_FIXED_EXPERIMENT_VALUES = {
    # Scene/input mutators are part of the preregistered state, not free
    # nuisance knobs.  The manifest and refined pose may be relocated between
    # hosts, but their typed identities are checked separately below.
    "relocalize_all_each_chunk": False,
    "offline_dataset_subject_source_name": None,
    "offline_dataset_subject_name": None,
    "offline_dataset_subject_label": None,
    "offline_dataset_visual_mesh": None,
    "offline_dataset_size_lwh": None,
    "offline_dataset_mass_kg": None,
    "offline_dataset_rgba": None,
    "offline_dataset_collision_primitive": None,
    "table_spec_json": None,
    "table_spec_height": False,
    "real_table_z": -0.021,
    "snap_subject_to_table": False,
    "collision_from_mesh": False,
    "sim_tcp_m": 0.19812,
    "sim_tcp_anchor": "site",
    "home_posture_json": None,
    "pool_size": 1001,
    "horizon_steps": 600,
    "num_knots": 30,
    "sigma_fraction": 1.0,
    "sampler_mode": "mtp_global_local",
    "mtp_jaw_mode": "sample",
    "local_sigma_fraction": 0.02,
    "optimizer": "mppi",
    "cem_rounds": 1,
    "cem_elite_fraction": 0.1,
    "cem_min_std_fraction": 0.01,
    "mppi_temperature": 0.1,
    "mppi_execution_mode": "best_valid_sample",
    "candidate_selection_mode": "result_terminal_earliest",
    "execution_prefix_steps": 50,
    "mppi_cycle_budget_scope": "stage",
    "seed": 1200,
    "arm_velocity_weight": 0.0,
    "record_candidate_cost_telemetry": True,
}


def validate_control_profile(value: Any, *, allow_none: bool = False) -> str | None:
    """Return one registered experiment profile or fail closed."""
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or value not in CONTROL_PROFILES:
        raise ValueError(
            f"control_profile must be one of {CONTROL_PROFILES}, got {value!r}"
        )
    return value


def validate_joint_velocity_force_task_id(
    value: Any,
    *,
    allow_none: bool = False,
) -> str | None:
    """Return one categorical four-task registration or fail closed."""
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or value not in JOINT_VELOCITY_FORCE_TASK_REGISTRY:
        raise ValueError(
            "joint_velocity_force_task_id must be one of "
            f"{JOINT_VELOCITY_FORCE_TASK_IDS}, got {value!r}"
        )
    return value


def validate_joint_velocity_force_trial_budget(
    value: Any,
    *,
    allow_none: bool = False,
) -> str | None:
    """Return one evidence-bound trial budget or fail closed."""
    if value is None and allow_none:
        return None
    if (
        not isinstance(value, str)
        or value not in JOINT_VELOCITY_FORCE_TRIAL_BUDGETS
    ):
        raise ValueError(
            "joint_velocity_force_trial_budget must be one of "
            f"{JOINT_VELOCITY_FORCE_TRIAL_BUDGETS}, got {value!r}"
        )
    return value


def validate_joint_velocity_force_algorithm_arm(
    value: Any,
    *,
    allow_none: bool = False,
) -> str | None:
    """Return one closed Phase-A algorithm arm or fail closed."""
    if value is None and allow_none:
        return None
    if (
        not isinstance(value, str)
        or value not in JOINT_VELOCITY_FORCE_ALGORITHM_ARMS
    ):
        raise ValueError(
            "joint_velocity_force_algorithm_arm must be one of "
            f"{tuple(JOINT_VELOCITY_FORCE_ALGORITHM_ARMS)}, got {value!r}"
        )
    return value


def joint_velocity_force_algorithm_arm_values(value: Any) -> dict[str, Any]:
    """Return an isolated copy of one registered Phase-A arm mapping."""
    arm = validate_joint_velocity_force_algorithm_arm(value)
    return dict(JOINT_VELOCITY_FORCE_ALGORITHM_ARMS[arm])


def validate_joint_velocity_force_dial_arm(
    value: Any,
    *,
    allow_none: bool = False,
) -> str | None:
    """Return the sole closed Phase-B DIAL-inspired arm or fail closed."""
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or value not in JOINT_VELOCITY_FORCE_DIAL_ARMS:
        raise ValueError(
            "joint_velocity_force_dial_arm must be one of "
            f"{JOINT_VELOCITY_FORCE_DIAL_ARMS}, got {value!r}"
        )
    return value


def joint_velocity_force_dial_arm_values(value: Any) -> dict[str, Any]:
    """Return a copy of the one registered Phase-B numeric contract."""
    validate_joint_velocity_force_dial_arm(value)
    return dict(_DIAL_ANNEALED_FIXED_VALUES)


def validate_joint_velocity_force_phase3_finalist_arm(
    value: Any,
    *,
    allow_none: bool = False,
) -> str | None:
    """Return one closed full360 Phase-3 finalist arm or fail closed."""
    if value is None and allow_none:
        return None
    if (
        not isinstance(value, str)
        or value not in JOINT_VELOCITY_FORCE_PHASE3_FINALIST_ARMS
    ):
        raise ValueError(
            "joint_velocity_force_phase3_finalist_arm must be one of "
            f"{tuple(JOINT_VELOCITY_FORCE_PHASE3_FINALIST_ARMS)}, got {value!r}"
        )
    return value


def joint_velocity_force_phase3_finalist_arm_values(
    value: Any,
) -> dict[str, Any]:
    """Return an isolated copy of one Phase-3 finalist mapping."""
    arm = validate_joint_velocity_force_phase3_finalist_arm(value)
    return dict(JOINT_VELOCITY_FORCE_PHASE3_FINALIST_ARMS[arm])


def validate_joint_velocity_force_phase3_seed(
    value: Any,
    *,
    allow_none: bool = False,
) -> int | None:
    """Return one preregistered Phase-3 episode seed or fail closed."""
    if value is None and allow_none:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in JOINT_VELOCITY_FORCE_PHASE3_SEEDS
    ):
        raise ValueError(
            "joint_velocity_force_phase3_seed must be one of "
            f"{JOINT_VELOCITY_FORCE_PHASE3_SEEDS}, got {value!r}"
        )
    return value


def evaluation_device_seed() -> int | None:
    """Optional offline-evaluation proposal panel; absent preserves old runs."""
    raw = os.environ.get("OAP_EVAL_DEVICE_SEED")
    if raw is None:
        return None
    if not raw.isdecimal() or not 0 <= int(raw) <= 2**31 - 1:
        raise ValueError("OAP_EVAL_DEVICE_SEED must be an integer in [0, 2**31-1]")
    return int(raw)


def joint_velocity_force_device_proposal_seed(
    *,
    trial_budget: Any = None,
    dial_arm: Any = None,
    phase3_arm: Any = None,
    phase3_seed: Any = None,
) -> int:
    """Resolve the one active numeric device proposal seed.

    The stage90 cost experiment deliberately uses its registered episode seed
    without borrowing the Phase-3 arm/seed identity.  Every inactive/default
    and Phase-A configuration retains the historical fixed-zero panel. Policy
    labels remain local to each evidence schema so legacy packet bytes do not
    change when only the numeric execution path needs sharing.
    """
    budget = validate_joint_velocity_force_trial_budget(
        trial_budget,
        allow_none=True,
    )
    dial = validate_joint_velocity_force_dial_arm(dial_arm, allow_none=True)
    arm = validate_joint_velocity_force_phase3_finalist_arm(
        phase3_arm,
        allow_none=True,
    )
    seed = validate_joint_velocity_force_phase3_seed(
        phase3_seed,
        allow_none=True,
    )
    if (arm is None) != (seed is None):
        raise ValueError("Phase-3 finalist arm and seed must be selected together")
    cost_focus_active = budget in JOINT_VELOCITY_FORCE_COST_FOCUS_TRIAL_BUDGETS
    active = int(cost_focus_active) + int(dial is not None) + int(arm is not None)
    if active > 1:
        raise ValueError(
            "cost-focus, DIAL, and Phase-3 device seed policies are mutually exclusive"
        )
    evaluation_seed = evaluation_device_seed()
    if evaluation_seed is not None:
        if active:
            raise ValueError("evaluation device seed conflicts with a registered seed experiment")
        return evaluation_seed
    if arm is not None:
        assert seed is not None
        return int(seed)
    if dial is not None:
        return 1200
    if cost_focus_active:
        values = (
            JOINT_VELOCITY_FORCE_COST_FOCUS_STAGE90_VALUES
            if budget == JOINT_VELOCITY_FORCE_TRIAL_BUDGET_COST_FOCUS_STAGE90_V1
            else JOINT_VELOCITY_FORCE_COST_FOCUS_SCREEN30_VALUES
        )
        return int(values["seed"])
    return 0


def joint_velocity_force_phase3_finalist_arm_telemetry(
    *,
    selected_arm: Any,
    selected_seed: Any,
    pool_size: Any,
    horizon_steps: Any,
    num_knots: Any,
    execution_prefix_steps: Any,
    sampler_mode: Any,
    mtp_jaw_mode: Any,
    local_sigma_fraction: Any,
    sigma_fraction: Any,
    mppi_temperature: Any,
    mppi_execution_mode: Any,
    candidate_selection_mode: Any,
    cem_rounds: Any,
    cem_elite_fraction: Any,
    cem_min_std_fraction: Any,
    arm_velocity_weight: Any,
    max_stage_cycles: Any,
    mppi_cycle_budget_scope: Any,
) -> dict[str, Any] | None:
    """Revalidate and describe one closed full360 Phase-3 finalist."""
    arm = validate_joint_velocity_force_phase3_finalist_arm(
        selected_arm,
        allow_none=True,
    )
    seed = validate_joint_velocity_force_phase3_seed(
        selected_seed,
        allow_none=True,
    )
    if arm is None and seed is None:
        return None
    if arm is None or seed is None:
        raise ValueError("Phase-3 finalist arm and seed must be selected together")
    expected = {
        **joint_velocity_force_phase3_finalist_arm_values(arm),
        "sigma_fraction": 1.0,
        "mppi_temperature": 0.1,
        "mppi_execution_mode": "best_valid_sample",
        "candidate_selection_mode": "result_terminal_earliest",
        "cem_rounds": 1,
        "cem_elite_fraction": 0.1,
        "cem_min_std_fraction": 0.01,
        "arm_velocity_weight": 0.0,
        "max_stage_cycles": 360,
        "mppi_cycle_budget_scope": "stage",
    }
    actual = {
        "pool_size": pool_size,
        "horizon_steps": horizon_steps,
        "num_knots": num_knots,
        "execution_prefix_steps": execution_prefix_steps,
        "sampler_mode": sampler_mode,
        "mtp_jaw_mode": mtp_jaw_mode,
        "local_sigma_fraction": local_sigma_fraction,
        "sigma_fraction": sigma_fraction,
        "mppi_temperature": mppi_temperature,
        "mppi_execution_mode": mppi_execution_mode,
        "candidate_selection_mode": candidate_selection_mode,
        "cem_rounds": cem_rounds,
        "cem_elite_fraction": cem_elite_fraction,
        "cem_min_std_fraction": cem_min_std_fraction,
        "arm_velocity_weight": arm_velocity_weight,
        "max_stage_cycles": max_stage_cycles,
        "mppi_cycle_budget_scope": mppi_cycle_budget_scope,
    }
    mismatches = {
        name: {"actual": actual[name], "required": required}
        for name, required in expected.items()
        if not _same(actual[name], required)
    }
    if mismatches:
        raise ValueError(
            f"Phase-3 finalist arm {arm!r} mismatch: {mismatches!r}"
        )
    return {
        "schema": "oap_joint_velocity_force_phase3_finalist_v1",
        "arm_id": arm,
        "episode_seed": seed,
        "config_episode_seed": seed,
        "device_proposal_seed": seed,
        "device_seed_policy": "registered_phase3_seed_active_only",
        "registered_values": expected,
        "registered_seeds": list(JOINT_VELOCITY_FORCE_PHASE3_SEEDS),
        "physics_dt_s": 0.002,
        "zoh_native_steps_per_action": (
            int(horizon_steps) // int(num_knots)
        ),
        "action_cadence_s": (
            int(horizon_steps) // int(num_knots)
        ) * 0.002,
        "only_registered_variables": ["arm_id", "episode_seed"],
        "ranking_contract": {
            "schema": "oap_phase3_closed_finalist_ranking_v1",
            "authority": "source_registered_evidence_only",
            "runtime_effect": "arm_dependent_expected_and_measured_ranked_last",
            "changes_cost": False,
            "changes_terminal": False,
            "changes_control_profile": False,
            "changes_sampler_configuration": True,
            "registered_sampler_differences": [
                "pool_size_N",
                "horizon_steps_H",
                "num_knots_K",
                "device_proposal_seed",
            ],
        },
    }


def joint_velocity_force_cost_focus_stage90_telemetry(
    *,
    selected_budget: Any,
    cost_shaping_profile: Any,
    pool_size: Any,
    horizon_steps: Any,
    num_knots: Any,
    execution_prefix_steps: Any,
    sampler_mode: Any,
    mtp_jaw_mode: Any,
    local_sigma_fraction: Any,
    sigma_fraction: Any,
    mppi_temperature: Any,
    mppi_execution_mode: Any,
    candidate_selection_mode: Any,
    cem_rounds: Any,
    cem_elite_fraction: Any,
    cem_min_std_fraction: Any,
    max_stage_cycles: Any,
    mppi_cycle_budget_scope: Any,
    seed: Any,
    arm_velocity_weight: Any,
    table_contact_force_weight: Any,
) -> dict[str, Any] | None:
    """Revalidate and describe the closed cost-only stage90 ablation."""
    budget = validate_joint_velocity_force_trial_budget(
        selected_budget,
        allow_none=True,
    )
    if budget != JOINT_VELOCITY_FORCE_TRIAL_BUDGET_COST_FOCUS_STAGE90_V1:
        return None
    from oap.program.cost_shaping import validate_cost_shaping_profile

    shaping = validate_cost_shaping_profile(cost_shaping_profile)
    actual = {
        "pool_size": pool_size,
        "horizon_steps": horizon_steps,
        "num_knots": num_knots,
        "execution_prefix_steps": execution_prefix_steps,
        "sampler_mode": sampler_mode,
        "mtp_jaw_mode": mtp_jaw_mode,
        "local_sigma_fraction": local_sigma_fraction,
        "sigma_fraction": sigma_fraction,
        "mppi_temperature": mppi_temperature,
        "mppi_execution_mode": mppi_execution_mode,
        "candidate_selection_mode": candidate_selection_mode,
        "cem_rounds": cem_rounds,
        "cem_elite_fraction": cem_elite_fraction,
        "cem_min_std_fraction": cem_min_std_fraction,
        "max_stage_cycles": max_stage_cycles,
        "mppi_cycle_budget_scope": mppi_cycle_budget_scope,
        "seed": seed,
        "arm_velocity_weight": arm_velocity_weight,
        "table_contact_force_weight": table_contact_force_weight,
    }
    mismatches = {
        name: {"actual": actual[name], "required": required}
        for name, required in JOINT_VELOCITY_FORCE_COST_FOCUS_STAGE90_VALUES.items()
        if not _same(actual[name], required)
    }
    if mismatches:
        raise ValueError(f"cost-focus stage90 planner mismatch: {mismatches!r}")
    return {
        "schema": "oap_joint_velocity_force_cost_focus_stage90_v1",
        "trial_budget": budget,
        "cost_shaping_profile": shaping,
        "registered_values": dict(
            JOINT_VELOCITY_FORCE_COST_FOCUS_STAGE90_VALUES
        ),
        "episode_seed": 1200,
        "device_proposal_seed": 1200,
        "device_seed_policy": "registered_cost_focus_seed_active_only",
        "only_registered_variable": "cost_shaping_profile",
        "terminal_predicates_thresholds_selectors": "unchanged",
        "hard_validity_and_safety": "unchanged",
        "control_regularizer": "unified_control_regularizer_v1_unchanged",
    }


def joint_velocity_force_cost_focus_screen30_telemetry(
    *,
    selected_budget: Any,
    cost_shaping_profile: Any,
    pool_size: Any,
    horizon_steps: Any,
    num_knots: Any,
    execution_prefix_steps: Any,
    sampler_mode: Any,
    mtp_jaw_mode: Any,
    local_sigma_fraction: Any,
    sigma_fraction: Any,
    mppi_temperature: Any,
    mppi_execution_mode: Any,
    candidate_selection_mode: Any,
    cem_rounds: Any,
    cem_elite_fraction: Any,
    cem_min_std_fraction: Any,
    max_stage_cycles: Any,
    mppi_cycle_budget_scope: Any,
    seed: Any,
    arm_velocity_weight: Any,
    table_contact_force_weight: Any,
) -> dict[str, Any] | None:
    """Describe the closed diagnostic-only 30-cycle cost screen."""
    budget = validate_joint_velocity_force_trial_budget(
        selected_budget,
        allow_none=True,
    )
    if budget != JOINT_VELOCITY_FORCE_TRIAL_BUDGET_COST_FOCUS_SCREEN30_V1:
        return None
    from oap.program.cost_shaping import validate_cost_shaping_profile

    shaping = validate_cost_shaping_profile(cost_shaping_profile)
    actual = {
        "pool_size": pool_size,
        "horizon_steps": horizon_steps,
        "num_knots": num_knots,
        "execution_prefix_steps": execution_prefix_steps,
        "sampler_mode": sampler_mode,
        "mtp_jaw_mode": mtp_jaw_mode,
        "local_sigma_fraction": local_sigma_fraction,
        "sigma_fraction": sigma_fraction,
        "mppi_temperature": mppi_temperature,
        "mppi_execution_mode": mppi_execution_mode,
        "candidate_selection_mode": candidate_selection_mode,
        "cem_rounds": cem_rounds,
        "cem_elite_fraction": cem_elite_fraction,
        "cem_min_std_fraction": cem_min_std_fraction,
        "max_stage_cycles": max_stage_cycles,
        "mppi_cycle_budget_scope": mppi_cycle_budget_scope,
        "seed": seed,
        "arm_velocity_weight": arm_velocity_weight,
        "table_contact_force_weight": table_contact_force_weight,
    }
    mismatches = {
        name: {"actual": actual[name], "required": required}
        for name, required in JOINT_VELOCITY_FORCE_COST_FOCUS_SCREEN30_VALUES.items()
        if not _same(actual[name], required)
    }
    if mismatches:
        raise ValueError(f"cost-focus screen30 planner mismatch: {mismatches!r}")
    return {
        "schema": "oap_joint_velocity_force_cost_focus_screen30_v1",
        "trial_budget": budget,
        "cost_shaping_profile": shaping,
        "registered_values": dict(
            JOINT_VELOCITY_FORCE_COST_FOCUS_SCREEN30_VALUES
        ),
        "episode_seed": 1200,
        "device_proposal_seed": 1200,
        "device_seed_policy": "registered_cost_focus_seed_active_only",
        "only_registered_variable": "cost_shaping_profile",
        "diagnostic_only": True,
        "stage90_ranking_eligible": False,
        "progress_packet_cycles": [10, 20, 30],
        "screen_completion_policy": "natural_outcome_or_full_30_no_external_kill",
        "promotion_authority": "external_screen_scorer_only",
        "terminal_predicates_thresholds_selectors": "unchanged",
        "hard_validity_and_safety": "unchanged",
        "control_regularizer": "unified_control_regularizer_v1_unchanged",
    }


def joint_velocity_force_dial_arm_telemetry(
    *,
    selected_arm: Any,
    pool_size: Any,
    horizon_steps: Any,
    num_knots: Any,
    execution_prefix_steps: Any,
    sampler_mode: Any,
    mtp_jaw_mode: Any,
    local_sigma_fraction: Any,
    sigma_fraction: Any,
    mppi_temperature: Any,
    mppi_execution_mode: Any,
    candidate_selection_mode: Any,
    cem_rounds: Any,
    cem_elite_fraction: Any,
    cem_min_std_fraction: Any,
    seed: Any,
    arm_velocity_weight: Any,
) -> dict[str, Any] | None:
    """Revalidate and describe the equal-budget Phase-B adaptation."""
    arm = validate_joint_velocity_force_dial_arm(selected_arm, allow_none=True)
    if arm is None:
        return None
    expected = {
        **joint_velocity_force_dial_arm_values(arm),
        "sigma_fraction": 1.0,
        "mppi_temperature": 0.1,
        "mppi_execution_mode": "best_valid_sample",
        "candidate_selection_mode": "result_terminal_earliest",
        "cem_rounds": 1,
        "cem_elite_fraction": 0.1,
        "cem_min_std_fraction": 0.01,
        "seed": 1200,
        "arm_velocity_weight": 0.0,
    }
    actual = {
        "pool_size": pool_size,
        "horizon_steps": horizon_steps,
        "num_knots": num_knots,
        "execution_prefix_steps": execution_prefix_steps,
        "sampler_mode": sampler_mode,
        "mtp_jaw_mode": mtp_jaw_mode,
        "local_sigma_fraction": local_sigma_fraction,
        "sigma_fraction": sigma_fraction,
        "mppi_temperature": mppi_temperature,
        "mppi_execution_mode": mppi_execution_mode,
        "candidate_selection_mode": candidate_selection_mode,
        "cem_rounds": cem_rounds,
        "cem_elite_fraction": cem_elite_fraction,
        "cem_min_std_fraction": cem_min_std_fraction,
        "seed": seed,
        "arm_velocity_weight": arm_velocity_weight,
    }
    mismatches = {
        name: {"actual": actual[name], "required": required}
        for name, required in expected.items()
        if not _same(actual[name], required)
    }
    if mismatches:
        raise ValueError(
            f"joint_velocity_force DIAL arm {arm!r} mismatch: {mismatches!r}"
        )
    return {
        "schema": "oap_joint_velocity_force_dial_arm_v1",
        "arm_id": arm,
        "phase": "phase_b_independent_candidate",
        "registered_values": expected,
        "official_dial_mpc_commit": DIAL_MPC_OFFICIAL_COMMIT,
        "official_source_files": [
            "dial_mpc/core/dial_core.py",
            "dial_mpc/core/dial_config.py",
        ],
        "port_claim": (
            "official_code_inspired_jvf_adaptation_not_exact_paper_port"
        ),
        "annealing_rounds": 2,
        "optimization_rollouts_per_round": 500,
        "final_exact_replay_rollouts": 1,
        "rollout_budget_closure": "2*500+1=1001",
        "round_population": "499_normal_samples_plus_exact_round_nominal",
        "round_rng": "jax_default_threefry2x32_normal_float32",
        "round_seed_offsets": [0, 104729],
        "round_seeds_for_registered_seed1200": [1200, 105929],
        "trajectory_std_factor": 0.5,
        "trajectory_std_scales": [1.0, 0.5],
        "trajectory_std_exponents": [0, 1],
        "horizon_std_factor": 0.9,
        "horizon_std_exponents": list(range(29, -1, -1)),
        "horizon_std_first_last": [0.9**29, 1.0],
        "covariance_is_std_squared": True,
        "normalized_action_dimension": 8,
        "optimizer_action_half_span": [0.2] * 7 + [1.0],
        "physical_action_half_span": [0.2] * 7 + [80.0],
        "action_units": ["rad_s"] * 7 + ["signed_effort_n"],
        "jaw_latent_decode_n_per_unit": 80.0,
        "temperature_policy": "fixed_0.1_each_round",
        "ess_temperature_adaptation": False,
        "elite_refit": False,
        "covariance_refit": False,
        "mtp_global_local_mixture": False,
        "selector_scope": "final_round_only",
        "clipping_policy": "elementwise_optimizer_action_bounds",
        "clipping_telemetry": "preclip_out_of_bounds_coordinate_count",
        "zoh_native_steps_per_knot": 20,
        "physics_dt_s": 0.002,
        "action_cadence_s": 0.04,
    }


def joint_velocity_force_algorithm_arm_telemetry(
    *,
    selected_arm: Any,
    pool_size: Any,
    horizon_steps: Any,
    num_knots: Any,
    execution_prefix_steps: Any,
    sampler_mode: Any,
    mtp_jaw_mode: Any,
    local_sigma_fraction: Any,
    sigma_fraction: Any,
    mppi_temperature: Any,
    mppi_execution_mode: Any,
    candidate_selection_mode: Any,
    cem_rounds: Any,
    cem_elite_fraction: Any,
    cem_min_std_fraction: Any,
    seed: Any,
    arm_velocity_weight: Any,
) -> dict[str, Any] | None:
    """Revalidate and describe one active Phase-A algorithm arm."""
    arm = validate_joint_velocity_force_algorithm_arm(
        selected_arm,
        allow_none=True,
    )
    if arm is None:
        return None
    expected = {
        **joint_velocity_force_algorithm_arm_values(arm),
        "sigma_fraction": 1.0,
        "mppi_temperature": 0.1,
        "mppi_execution_mode": "best_valid_sample",
        "candidate_selection_mode": "result_terminal_earliest",
        "cem_rounds": 1,
        "cem_elite_fraction": 0.1,
        "cem_min_std_fraction": 0.01,
        "seed": 1200,
        "arm_velocity_weight": 0.0,
    }
    actual = {
        "pool_size": pool_size,
        "horizon_steps": horizon_steps,
        "num_knots": num_knots,
        "execution_prefix_steps": execution_prefix_steps,
        "sampler_mode": sampler_mode,
        "mtp_jaw_mode": mtp_jaw_mode,
        "local_sigma_fraction": local_sigma_fraction,
        "sigma_fraction": sigma_fraction,
        "mppi_temperature": mppi_temperature,
        "mppi_execution_mode": mppi_execution_mode,
        "candidate_selection_mode": candidate_selection_mode,
        "cem_rounds": cem_rounds,
        "cem_elite_fraction": cem_elite_fraction,
        "cem_min_std_fraction": cem_min_std_fraction,
        "seed": seed,
        "arm_velocity_weight": arm_velocity_weight,
    }
    mismatches = {
        name: {"actual": actual[name], "required": required}
        for name, required in expected.items()
        if not _same(actual[name], required)
    }
    if mismatches:
        raise ValueError(
            f"joint_velocity_force algorithm arm {arm!r} mismatch: "
            f"{mismatches!r}"
        )
    optimization_population = int(expected["pool_size"]) - 1
    packet: dict[str, Any] = {
        "schema": "oap_joint_velocity_force_algorithm_arm_v1",
        "arm_id": arm,
        "registered_values": expected,
        "only_registered_variables": sorted(
            JOINT_VELOCITY_FORCE_ALGORITHM_ARMS[arm]
        ),
        "action_dimension": 8,
        "action_units": ["rad_s"] * 7 + ["signed_effort_n"],
        "zoh_native_steps_per_knot": 20,
        "physics_dt_s": 0.002,
        "action_cadence_s": 0.04,
        "total_rollout_budget": int(expected["pool_size"]),
        "optimization_population": optimization_population,
        "exact_singleton_replay_population": 1,
    }
    if arm == "mppi_single":
        packet.update({
            "proposal_semantics": (
                "fixed_covariance_diagonal_gaussian_all_8d"
            ),
            "jaw_sampling": "ordinary_mppi_gaussian_8d",
            "mtp_jaw_mode_role": "inactive_contract_must_preserve_nominal",
            "covariance_update": False,
            "fixed_std_fraction_of_action_half_range": math.sqrt(0.1),
            "halton_spline_degree": 1,
            "halton_knot_scale": 2,
            "halton_smoothing": 0.5,
            "sample_null_action": True,
            "sample_previous_plan": True,
            "ess_semantics": (
                "inverse_sum_squared_normalized_softmax_weights"
            ),
            "temperature_update": "adaptive_from_ess",
        })
    else:
        global_count = int(round(0.5 * optimization_population))
        packet.update({
            "proposal_semantics": "mtp_global_local_linear_m3_n50",
            "jaw_sampling": "mtp_sampled_signed_latent_with_extremes",
            "global_local_beta": 0.5,
            "graph_depth": 3,
            "graph_width": 50,
            "elite_refit_count": 20,
            "temperature_update": "fixed_0.1_no_ess_update",
            "nominal_count": 1,
            "global_count": global_count,
            "local_count": optimization_population - 1 - global_count,
        })
    return packet


def joint_velocity_force_registered_table_force_weight(value: Any) -> float:
    """The one planner-level table-force coefficient shared by all four tasks."""
    validate_joint_velocity_force_task_id(value)
    return 0.0


def _path_has_registered_suffix(actual: Any, expected: str) -> bool:
    normalized = str(actual).replace("\\", "/")
    return normalized == expected or normalized.endswith(f"/{expected}")


def validate_direct_torque_s0_coverage_k(
    value: Any,
    *,
    allow_none: bool = False,
) -> int | None:
    """Return one registered K for the direct-torque S0 coverage matrix."""
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            "direct_torque_s0_coverage_k must be one of "
            f"{DIRECT_TORQUE_S0_COVERAGE_K}, got {value!r}"
        )
    selected = int(value)
    if selected not in DIRECT_TORQUE_S0_COVERAGE_K:
        raise ValueError(
            "direct_torque_s0_coverage_k must be one of "
            f"{DIRECT_TORQUE_S0_COVERAGE_K}, got {value!r}"
        )
    return selected


def validate_direct_torque_s0_horizon_prefix_arm(
    value: Any,
    *,
    allow_none: bool = False,
) -> str | None:
    """Return one registered horizon-prefix arm or fail closed."""
    if value is None and allow_none:
        return None
    if (
        not isinstance(value, str)
        or value not in DIRECT_TORQUE_S0_HORIZON_PREFIX_ARMS
    ):
        raise ValueError(
            "direct_torque_s0_horizon_prefix_arm must be one of "
            f"{tuple(DIRECT_TORQUE_S0_HORIZON_PREFIX_ARMS)}, got {value!r}"
        )
    return value


def direct_torque_s0_horizon_prefix_values(value: Any) -> dict[str, int]:
    """Return an isolated copy of one registered arm's three numeric values."""
    arm = validate_direct_torque_s0_horizon_prefix_arm(value)
    return dict(DIRECT_TORQUE_S0_HORIZON_PREFIX_ARMS[arm])


def direct_torque_s0_horizon_prefix_telemetry(
    *,
    selected_arm: Any,
    horizon_steps: Any,
    num_knots: Any,
    execution_prefix_steps: Any,
    execution_dt_s: Any,
    pool_size: Any,
    sigma_fraction: Any,
    local_sigma_fraction: Any,
) -> dict[str, Any] | None:
    """Describe and revalidate the registered horizon-by-prefix arm."""
    arm = validate_direct_torque_s0_horizon_prefix_arm(
        selected_arm,
        allow_none=True,
    )
    if arm is None:
        return None
    expected = direct_torque_s0_horizon_prefix_values(arm)
    actual = {
        "horizon_steps": horizon_steps,
        "num_knots": num_knots,
        "execution_prefix_steps": execution_prefix_steps,
    }
    if actual != expected:
        raise ValueError(
            f"horizon-prefix arm {arm!r} values mismatch: "
            f"got {actual!r}, required {expected!r}"
        )
    dt = float(execution_dt_s)
    broad = float(sigma_fraction)
    local = float(local_sigma_fraction)
    if not math.isclose(dt, 0.002, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("horizon-prefix matrix requires 2 ms physics")
    if pool_size != 1001:
        raise ValueError("horizon-prefix matrix requires pool_size=1001")
    if not math.isclose(broad, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("horizon-prefix matrix requires sigma_fraction=1")
    if not math.isclose(local, 0.02, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "horizon-prefix matrix requires local_sigma_fraction=0.02"
        )
    horizon = expected["horizon_steps"]
    knots = expected["num_knots"]
    prefix = expected["execution_prefix_steps"]
    if horizon % knots or horizon // knots != 20:
        raise ValueError("horizon-prefix matrix lost its 40 ms knot cadence")
    return {
        "schema": "oap_direct_torque_s0_horizon_prefix_2x2_v1",
        "selected_arm": arm,
        "registered_arms": {
            name: dict(values)
            for name, values in DIRECT_TORQUE_S0_HORIZON_PREFIX_ARMS.items()
        },
        "registered_axes": [
            "horizon_duration_with_paired_knot_count",
            "execution_prefix_steps",
        ],
        "horizon_axis_explanation": (
            "horizon_steps and num_knots co-vary to preserve 40 ms cadence"
        ),
        "prefix_axis_explanation": (
            "execution_prefix_steps changes without changing horizon or K"
        ),
        "horizon_steps": horizon,
        "num_knots": knots,
        "execution_prefix_steps": prefix,
        "physics_dt_s": dt,
        "zoh_steps_per_action": horizon // knots,
        "action_dt_s": (horizon // knots) * dt,
        "horizon_s": horizon * dt,
        "execution_prefix_s": prefix * dt,
        "execution_prefix_fraction": prefix / float(horizon),
        "fixed": {
            "pool_size": 1001,
            "mppi_temperature": 0.1,
            "sigma_fraction": broad,
            "local_sigma_fraction": local,
            "episode_seed": 1200,
            "device_seed": 0,
            "arm_velocity_weight": 0.0,
            "candidate_selection_mode": "result_terminal_earliest",
        },
        "unchanged": [
            "control_profile_direct_torque_force",
            "pick_v9_s0_task_program",
            "full360_cycle_budget",
            "mtp_global_local_sampler",
            "sampled_jaw",
            "terminal",
            "cost",
        ],
    }


def direct_torque_s0_coverage_telemetry(
    *,
    selected_k: Any,
    horizon_steps: Any,
    execution_dt_s: Any,
    sigma_fraction: Any,
    local_sigma_fraction: Any,
) -> dict[str, Any] | None:
    """Describe the registered K-only coverage axis without changing it."""
    selected = validate_direct_torque_s0_coverage_k(
        selected_k, allow_none=True
    )
    if selected is None:
        return None
    if isinstance(horizon_steps, bool) or not isinstance(horizon_steps, int):
        raise ValueError("coverage horizon_steps must be an integer")
    horizon = int(horizon_steps)
    if horizon != 600 or horizon % selected:
        raise ValueError(
            "direct-torque S0 coverage requires H=600 and equal ZOH "
            f"intervals, got H={horizon}, K={selected}"
        )
    dt = float(execution_dt_s)
    broad = float(sigma_fraction)
    local = float(local_sigma_fraction)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("coverage execution_dt_s must be finite and positive")
    if not math.isclose(broad, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("coverage broad sigma fraction must remain 1.0")
    if not math.isclose(local, 0.02, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("coverage local sigma fraction must remain 0.02")
    return {
        "schema": "oap_direct_torque_s0_sampling_coverage_v1",
        "axis": "num_knots",
        "registered_k": list(DIRECT_TORQUE_S0_COVERAGE_K),
        "selected_k": selected,
        "fixed_pool_size": 1001,
        "fixed_horizon_steps": horizon,
        "fixed_mppi_temperature": 0.1,
        "fixed_episode_seed": 1200,
        "fixed_arm_velocity_weight": 0.0,
        "fixed_candidate_selection_mode": "result_terminal_earliest",
        "zoh_steps_per_action": horizon // selected,
        "action_dt_s": (horizon // selected) * dt,
        "action_dimension": 8,
        "proposal_seed": 0,
        "global_mtp_distribution": "uniform_nodes_over_full_action_bounds",
        "registered_sigma_fraction_of_half_range": broad,
        "global_mtp_uses_sigma_fraction": False,
        "local_gaussian_std_fraction_of_half_range": local,
        "local_std_shape": [selected, 8],
        "local_covariance": "diagonal_per_knot_action_coordinate",
        "local_marginal_variance_fraction_of_half_range_squared": local**2,
        "jaw_exploration": "sampled_signed_latent_with_global_extremes",
        "covariance_dimension_changes_with_k": True,
        "marginal_std_rescaled_with_k": False,
        "covariance_rescaled_with_k": False,
        "only_registered_variable": "num_knots",
    }


def control_profile_compound_experiment_id(
    *,
    control_profile: Any,
    smoke_one_cycle: Any,
    direct_torque_s0_coverage_k: Any = None,
    direct_torque_s0_horizon_prefix_arm: Any = None,
    joint_velocity_force_task_id: Any = None,
    joint_velocity_force_trial_budget: Any = None,
    joint_velocity_force_algorithm_arm: Any = None,
    joint_velocity_force_dial_arm: Any = None,
    joint_velocity_force_phase3_finalist_arm: Any = None,
    joint_velocity_force_phase3_seed: Any = None,
    control_regularizer_calibration_profile: Any = None,
    cost_shaping_profile: Any = None,
) -> str:
    """Return the formal experiment identity, separate from numeric profile."""
    profile = validate_control_profile(control_profile)
    if not isinstance(smoke_one_cycle, bool):
        raise ValueError("smoke_one_cycle must be a boolean")
    coverage_k = validate_direct_torque_s0_coverage_k(
        direct_torque_s0_coverage_k,
        allow_none=True,
    )
    matrix_arm = validate_direct_torque_s0_horizon_prefix_arm(
        direct_torque_s0_horizon_prefix_arm,
        allow_none=True,
    )
    task_id = validate_joint_velocity_force_task_id(
        joint_velocity_force_task_id,
        allow_none=True,
    )
    from oap.twin.control_regularizer import (
        UNIFIED_CONTROL_REGULARIZER_V1,
        validate_control_regularizer_calibration_profile,
    )
    trial_budget = validate_joint_velocity_force_trial_budget(
        joint_velocity_force_trial_budget,
        allow_none=True,
    )
    algorithm_arm = validate_joint_velocity_force_algorithm_arm(
        joint_velocity_force_algorithm_arm,
        allow_none=True,
    )
    dial_arm = validate_joint_velocity_force_dial_arm(
        joint_velocity_force_dial_arm,
        allow_none=True,
    )
    phase3_arm = validate_joint_velocity_force_phase3_finalist_arm(
        joint_velocity_force_phase3_finalist_arm,
        allow_none=True,
    )
    phase3_seed = validate_joint_velocity_force_phase3_seed(
        joint_velocity_force_phase3_seed,
        allow_none=True,
    )

    regularizer_profile = validate_control_regularizer_calibration_profile(
        control_regularizer_calibration_profile,
        allow_none=True,
    )
    from oap.program.cost_shaping import validate_cost_shaping_profile

    shaping_profile = validate_cost_shaping_profile(
        cost_shaping_profile,
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
    cost_focus = cost_focus_stage90 or cost_focus_screen30
    if coverage_k is not None and matrix_arm is not None:
        raise ValueError(
            "coverage K and horizon-prefix matrix are mutually exclusive"
        )
    if coverage_k is not None:
        if profile != CONTROL_PROFILE_DIRECT_TORQUE_FORCE:
            raise ValueError("sampling coverage identity requires direct torque")
        if smoke_one_cycle:
            raise ValueError("sampling coverage identity requires full360")
    if matrix_arm is not None:
        if profile != CONTROL_PROFILE_DIRECT_TORQUE_FORCE:
            raise ValueError("horizon-prefix identity requires direct torque")
        if smoke_one_cycle:
            raise ValueError("horizon-prefix identity requires full360")
    if task_id is not None:
        if profile != CONTROL_PROFILE_JOINT_VELOCITY_FORCE:
            raise ValueError(
                "four-task registry identity requires joint_velocity_force"
            )
    if trial_budget is not None:
        if (
            task_id is None
            or smoke_one_cycle
            or profile != CONTROL_PROFILE_JOINT_VELOCITY_FORCE
            or regularizer_profile not in (
                None,
                UNIFIED_CONTROL_REGULARIZER_V1,
            )
        ):
            raise ValueError(
                "joint_velocity_force trial budget requires registered "
                "non-smoke baseline or formal unified-regularizer mode; "
                "calibration is forbidden"
            )
    if cost_focus and (
        regularizer_profile != UNIFIED_CONTROL_REGULARIZER_V1
        or shaping_profile is None
        or algorithm_arm is not None
        or dial_arm is not None
        or phase3_arm is not None
        or phase3_seed is not None
    ):
        raise ValueError(
            "cost-focus trial requires the formal unified regularizer, an "
            "explicit registered cost-shaping profile, and forbids Phase-A, "
            "Phase-B DIAL, and Phase-3 identities"
        )
    if algorithm_arm is not None and (
        task_id is None
        or trial_budget != JOINT_VELOCITY_FORCE_TRIAL_BUDGET_STAGE30_V1
        or smoke_one_cycle
        or profile != CONTROL_PROFILE_JOINT_VELOCITY_FORCE
        or regularizer_profile != UNIFIED_CONTROL_REGULARIZER_V1
    ):
        raise ValueError(
            "joint_velocity_force algorithm arm requires registered four-task "
            "JVF formal-regularizer stage30 mode"
        )
    if dial_arm is not None and (
        task_id is None
        or trial_budget != JOINT_VELOCITY_FORCE_TRIAL_BUDGET_STAGE30_V1
        or smoke_one_cycle
        or profile != CONTROL_PROFILE_JOINT_VELOCITY_FORCE
        or regularizer_profile != UNIFIED_CONTROL_REGULARIZER_V1
        or algorithm_arm is not None
    ):
        raise ValueError(
            "joint_velocity_force DIAL arm requires registered four-task JVF "
            "formal-regularizer stage30 mode and forbids Phase-A arms"
        )
    if phase3_arm is None and phase3_seed is not None:
        raise ValueError("Phase-3 seed requires a Phase-3 finalist arm")
    if phase3_arm is not None and (
        task_id is None
        or phase3_seed is None
        or trial_budget is not None
        or smoke_one_cycle
        or profile != CONTROL_PROFILE_JOINT_VELOCITY_FORCE
        or regularizer_profile != UNIFIED_CONTROL_REGULARIZER_V1
        or algorithm_arm is not None
        or dial_arm is not None
    ):
        raise ValueError(
            "Phase-3 finalist requires registered four-task JVF formal-"
            "regularizer full360 mode and forbids earlier algorithm arms"
        )
    mode_label = (
        "smoke1"
        if smoke_one_cycle
        else "stage90_cost_focus"
        if cost_focus_stage90
        else "screen30_cost_focus"
        if cost_focus_screen30
        else "stage30_pilot"
        if trial_budget is not None
        else "full360"
    )
    base = (
        "pick_v9_s0:control_profile_contract_v2:mtp_fixed_zero:"
        f"{profile}:{mode_label}"
    )
    identity = base + (
        f":coverage_k{coverage_k}" if coverage_k is not None else ""
    )
    if matrix_arm is not None:
        identity += f":horizon_prefix_2x2:{matrix_arm}"
    if task_id is not None:
        identity = (
            "joint_velocity_force_four_task_registry_v1:"
            f"based_on_{JOINT_VELOCITY_FORCE_TASK_REGISTRY_BASE_COMMIT}:"
            f"{task_id}:control_profile_contract_v2:{profile}:"
            f"{mode_label}"
        )
    if trial_budget is not None:
        identity += (
            ":stage90_cost_focus_v1"
            if cost_focus_stage90
            else ":screen30_cost_focus_v1"
            if cost_focus_screen30
            else ":stage30_pilot_v1"
        )
    if regularizer_profile is not None:
        if task_id is None:
            raise ValueError(
                "control regularizer calibration requires four-task registry"
            )
        identity += (
            ":unified_control_regularizer_v1"
            if regularizer_profile == UNIFIED_CONTROL_REGULARIZER_V1
            else ":regularizer_calibration_only_v1"
        )
    if algorithm_arm is not None:
        identity += f":algorithm_arm:{algorithm_arm}"
    if dial_arm is not None:
        identity += f":phase_b:{dial_arm}"
    if phase3_arm is not None:
        identity += f":phase3_finalist:{phase3_arm}:seed{phase3_seed}"
    if shaping_profile is not None:
        identity += f":cost_shaping:{shaping_profile}"
    return identity


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return (
            isinstance(actual, Real)
            and not isinstance(actual, bool)
            and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        )
    return actual == expected


def require_s0_control_profile_contract(
    value: Any,
    *,
    program: Any | None = None,
) -> str | None:
    """Validate the complete experiment-only launch and program contract."""
    profile = validate_control_profile(
        getattr(value, "control_profile", None), allow_none=True
    )
    coverage_k = validate_direct_torque_s0_coverage_k(
        getattr(value, "direct_torque_s0_coverage_k", None),
        allow_none=True,
    )
    matrix_arm = validate_direct_torque_s0_horizon_prefix_arm(
        getattr(value, "direct_torque_s0_horizon_prefix_arm", None),
        allow_none=True,
    )
    task_id = validate_joint_velocity_force_task_id(
        getattr(value, "joint_velocity_force_task_id", None),
        allow_none=True,
    )
    if task_id == "push_v11":
        raise ValueError(
            "Direct Push requires the Push v20 unified profile "
            "and a VLM-generated TaskProgram"
        )
    from oap.twin.control_regularizer import (
        UNIFIED_CONTROL_REGULARIZER_V1,
        validate_control_regularizer_calibration_profile,
    )
    trial_budget = validate_joint_velocity_force_trial_budget(
        getattr(value, "joint_velocity_force_trial_budget", None),
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

    regularizer_profile = validate_control_regularizer_calibration_profile(
        getattr(value, "control_regularizer_calibration_profile", None),
        allow_none=True,
    )
    from oap.program.cost_shaping import validate_cost_shaping_profile

    shaping_profile = validate_cost_shaping_profile(
        getattr(value, "cost_shaping_profile", None),
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
    cost_focus = cost_focus_stage90 or cost_focus_screen30
    smoke = getattr(value, "control_profile_smoke_one_cycle", False)
    if not isinstance(smoke, bool):
        raise ValueError(
            "control-profile smoke flag must be a Boolean"
        )
    if smoke and profile is None:
        raise ValueError(
            "control-profile smoke requires an explicit control_profile"
        )
    if coverage_k is not None and profile is None:
        raise ValueError(
            "direct-torque S0 coverage requires an explicit control_profile"
        )
    if matrix_arm is not None and profile is None:
        raise ValueError(
            "horizon-prefix matrix requires an explicit control_profile"
        )
    if task_id is not None and profile is None:
        raise ValueError(
            "joint_velocity_force_task_id requires an explicit control_profile"
        )
    if regularizer_profile is not None and task_id is None:
        raise ValueError(
            "control regularizer calibration requires four-task registry"
        )
    if trial_budget is not None and task_id is None:
        raise ValueError(
            "joint_velocity_force_trial_budget requires four-task registry"
        )
    if algorithm_arm is not None and task_id is None:
        raise ValueError(
            "joint_velocity_force_algorithm_arm requires four-task registry"
        )
    if dial_arm is not None and task_id is None:
        raise ValueError(
            "joint_velocity_force_dial_arm requires four-task registry"
        )
    if phase3_arm is not None and task_id is None:
        raise ValueError(
            "joint_velocity_force_phase3_finalist_arm requires four-task registry"
        )
    if phase3_arm is None and phase3_seed is not None:
        raise ValueError("Phase-3 seed requires a Phase-3 finalist arm")
    if trial_budget is not None and smoke:
        raise ValueError(
            "joint_velocity_force_trial_budget conflicts with smoke mode"
        )
    if trial_budget is not None and regularizer_profile not in (
        None,
        UNIFIED_CONTROL_REGULARIZER_V1,
    ):
        raise ValueError(
            "registered trial budgets forbid the calibration regularizer profile"
        )
    if cost_focus and (
        profile != CONTROL_PROFILE_JOINT_VELOCITY_FORCE
        or regularizer_profile != UNIFIED_CONTROL_REGULARIZER_V1
        or shaping_profile is None
        or smoke
        or algorithm_arm is not None
        or dial_arm is not None
        or phase3_arm is not None
        or phase3_seed is not None
    ):
        raise ValueError(
            "cost-focus trial requires registered four-task JVF, formal "
            "unified regularizer, an explicit cost profile, and forbids "
            "smoke and all algorithm/DIAL/Phase-3 identities"
        )
    if regularizer_profile == UNIFIED_CONTROL_REGULARIZER_V1 and smoke:
        raise ValueError(
            "unified_control_regularizer_v1 forbids smoke mode"
        )
    if algorithm_arm is not None and (
        profile != CONTROL_PROFILE_JOINT_VELOCITY_FORCE
        or trial_budget != JOINT_VELOCITY_FORCE_TRIAL_BUDGET_STAGE30_V1
        or regularizer_profile != UNIFIED_CONTROL_REGULARIZER_V1
        or smoke
    ):
        raise ValueError(
            "joint_velocity_force algorithm arm requires registered four-task "
            "JVF formal-regularizer stage30 mode"
        )
    if dial_arm is not None and (
        profile != CONTROL_PROFILE_JOINT_VELOCITY_FORCE
        or trial_budget != JOINT_VELOCITY_FORCE_TRIAL_BUDGET_STAGE30_V1
        or regularizer_profile != UNIFIED_CONTROL_REGULARIZER_V1
        or smoke
        or algorithm_arm is not None
    ):
        raise ValueError(
            "joint_velocity_force DIAL arm requires registered four-task JVF "
            "formal-regularizer stage30 mode and forbids Phase-A arms"
        )
    if phase3_arm is not None and (
        profile != CONTROL_PROFILE_JOINT_VELOCITY_FORCE
        or phase3_seed is None
        or trial_budget is not None
        or regularizer_profile != UNIFIED_CONTROL_REGULARIZER_V1
        or smoke
        or algorithm_arm is not None
        or dial_arm is not None
    ):
        raise ValueError(
            "Phase-3 finalist requires registered four-task JVF formal-"
            "regularizer full360 mode and forbids earlier algorithm arms"
        )
    if profile is None:
        return None

    if task_id is not None:
        registration = JOINT_VELOCITY_FORCE_TASK_REGISTRY[task_id]
        mismatches: list[str] = []
        if regularizer_profile is not None and not _same(
            getattr(value, "arm_velocity_weight", None), 0.0
        ):
            mismatches.append(
                "control regularizer calibration weights are fixed zero; "
                "arm_velocity_weight must remain 0.0"
            )
        if profile != CONTROL_PROFILE_JOINT_VELOCITY_FORCE:
            mismatches.append(
                "joint_velocity_force_task_id requires "
                "control_profile='joint_velocity_force'"
            )
        if coverage_k is not None or matrix_arm is not None:
            mismatches.append(
                "four-task registry forbids Pick-only sampling matrices"
            )
        expected_values = dict(_FIXED_EXPERIMENT_VALUES)
        if algorithm_arm is not None:
            expected_values.update(
                joint_velocity_force_algorithm_arm_values(algorithm_arm)
            )
            expected_values["execution_prefix_fraction"] = 1.0 / 3.0
        if dial_arm is not None:
            expected_values.update(
                joint_velocity_force_dial_arm_values(dial_arm)
            )
            expected_values["execution_prefix_fraction"] = 1.0 / 3.0
        if phase3_arm is not None:
            expected_values.update(
                joint_velocity_force_phase3_finalist_arm_values(phase3_arm)
            )
            expected_values["execution_prefix_fraction"] = 1.0 / 3.0
            expected_values["seed"] = phase3_seed
        if cost_focus:
            expected_values.update(
                JOINT_VELOCITY_FORCE_COST_FOCUS_STAGE90_VALUES
                if cost_focus_stage90
                else JOINT_VELOCITY_FORCE_COST_FOCUS_SCREEN30_VALUES
            )
            expected_values["execution_prefix_fraction"] = 1.0 / 3.0
        expected_values.update(registration["scene"])
        for name, expected in expected_values.items():
            actual = getattr(value, name, None)
            if name == "offline_dataset_visual_mesh" and expected is not None:
                if not _path_has_registered_suffix(actual, str(expected)):
                    mismatches.append(
                        f"{name}={actual!r} differs from registered scene "
                        f"artifact {expected!r}"
                    )
            elif not _same(actual, expected):
                mismatches.append(
                    f"{name}={actual!r} differs from registered scene/controller "
                    f"value {expected!r}"
                )
        required_cycles = (
            90
            if cost_focus_stage90
            else 30
            if cost_focus_screen30
            else 30
            if trial_budget is not None
            else 1
            if smoke
            else 360
        )
        if getattr(value, "max_stage_cycles", None) != required_cycles:
            mismatches.append(
                "max_stage_cycles must remain "
                f"{required_cycles} for registered "
                f"{'smoke' if smoke else 'full'} mode"
            )
        if not bool(getattr(value, "offline", False)):
            mismatches.append("offline must be true")
        if bool(getattr(value, "execute", False)):
            mismatches.append("execute must be false")
        if bool(getattr(value, "prepare_real_execution", False)):
            mismatches.append("prepare_real_execution must be false")
        if getattr(value, "remote_planner_url", None) is not None:
            mismatches.append("remote_planner_url must be absent")
        if getattr(value, "mppi_stage_execution_prefix_steps", None) is not None:
            mismatches.append("stage-specific execution prefixes are forbidden")
        if getattr(value, "task", None) != registration["instruction"]:
            mismatches.append(
                "task must equal the sealed TaskProgram canonical instruction"
            )
        program_json = getattr(value, "program_json", None)
        if program_json is None or not _path_has_registered_suffix(
            program_json,
            registration["program_path"],
        ):
            mismatches.append("program_json must name the registered authority file")
        initial_obs = getattr(value, "initial_obs_json", None)
        if initial_obs is None or not _path_has_registered_suffix(
            initial_obs,
            registration["initial_observation_path"],
        ):
            mismatches.append(
                "initial_obs_json must name the registered observation artifact"
            )
        bundle_manifest = getattr(value, "bundle_manifest", None)
        if (
            bundle_manifest is None
            or str(bundle_manifest).replace("\\", "/").split("/")[-1]
            != "manifest.json"
        ):
            mismatches.append("bundle_manifest must name sealed manifest.json")
        if program is not None:
            authority_path = external_input_path(
                registration["program_path"], purpose="registered program authority"
            )
            from oap.program import TaskProgram

            authority = TaskProgram.from_json(
                authority_path.read_text(encoding="utf-8")
            )
            if program.to_dict() != authority.to_dict():
                mismatches.append("program differs from sealed TaskProgram authority")
            if getattr(program, "instruction", None) != registration["instruction"]:
                mismatches.append("program canonical instruction differs")
            if getattr(program, "provenance", None) != registration["provenance"]:
                mismatches.append("program provenance differs")
            stages = list(getattr(program, "stages", ()))
            if len(stages) != registration["stage_count"]:
                mismatches.append("program stage count differs")
            if [stage.name for stage in stages] != registration["stage_names"]:
                mismatches.append("program stage names differ")
        if mismatches:
            raise ValueError(
                "joint_velocity_force four-task contract mismatch: "
                + "; ".join(mismatches)
            )
        return profile

    expected_values = dict(_FIXED_EXPERIMENT_VALUES)
    if coverage_k is not None:
        expected_values["num_knots"] = coverage_k
    if matrix_arm is not None:
        expected_values.update(
            direct_torque_s0_horizon_prefix_values(matrix_arm)
        )
    mismatches = [
        f"{name}={getattr(value, name, None)!r} (required {expected!r})"
        for name, expected in expected_values.items()
        if not _same(getattr(value, name, None), expected)
    ]
    if coverage_k is not None:
        if profile != CONTROL_PROFILE_DIRECT_TORQUE_FORCE:
            mismatches.append(
                "direct-torque S0 coverage requires "
                "control_profile='direct_torque_force'"
            )
        if smoke:
            mismatches.append(
                "direct-torque S0 coverage forbids smoke mode; "
                "max_stage_cycles must remain 360"
            )
    if matrix_arm is not None:
        if coverage_k is not None:
            mismatches.append(
                "coverage K and horizon-prefix matrix are mutually exclusive"
            )
        if profile != CONTROL_PROFILE_DIRECT_TORQUE_FORCE:
            mismatches.append(
                "horizon-prefix matrix requires direct-torque control"
            )
        if smoke:
            mismatches.append(
                "horizon-prefix matrix forbids smoke and requires full360"
            )
    required_cycles = 1 if smoke else 360
    if getattr(value, "max_stage_cycles", None) != required_cycles:
        mismatches.append(
            "max_stage_cycles="
            f"{getattr(value, 'max_stage_cycles', None)!r} "
            f"(required {required_cycles!r} for "
            f"{'smoke' if smoke else 'full'} control-profile mode)"
        )
    if not bool(getattr(value, "offline", False)):
        mismatches.append("offline must be true")
    if bool(getattr(value, "execute", False)):
        mismatches.append("execute must be false")
    if bool(getattr(value, "prepare_real_execution", False)):
        mismatches.append("prepare_real_execution must be false")
    if getattr(value, "remote_planner_url", None) is not None:
        mismatches.append("remote_planner_url must be absent")
    if getattr(value, "mppi_stage_execution_prefix_steps", None) is not None:
        mismatches.append("mppi_stage_execution_prefix_steps must be absent")
    if getattr(value, "task", None) != PICK_V9_INSTRUCTION:
        mismatches.append("task must equal the frozen Pick-v9 instruction")
    if getattr(value, "program_json", None) is None:
        mismatches.append("program_json must name the frozen S0-only program")
    elif str(getattr(value, "program_json")).replace("\\", "/").split("/")[-1] != (
        "minimal_eraser_on_box_task_program_paper_mppi_v9_"
        "s0_control_profile.json"
    ):
        mismatches.append("program_json must be the frozen Pick-v9 S0 file")
    initial_obs = getattr(value, "initial_obs_json", None)
    if initial_obs is None:
        mismatches.append("initial_obs_json must name the frozen eraser pose")
    elif str(initial_obs).replace("\\", "/").split("/")[-1] != "eraser_refined.json":
        mismatches.append("initial_obs_json must be eraser_refined.json")
    bundle_manifest = getattr(value, "bundle_manifest", None)
    if (
        bundle_manifest is None
        or str(bundle_manifest).replace("\\", "/").split("/")[-1]
        != "manifest.json"
    ):
        mismatches.append("bundle_manifest must name the sealed manifest.json")

    if program is not None:
        if getattr(program, "instruction", None) != PICK_V9_INSTRUCTION:
            mismatches.append("program instruction differs from Pick-v9")
        if getattr(program, "provenance", None) != PICK_V9_PROVENANCE:
            mismatches.append("program provenance differs from Pick-v9")
        stages = list(getattr(program, "stages", ()))
        if len(stages) != 1:
            mismatches.append(f"program must contain exactly S0, got {len(stages)} stages")
        elif stages[0].to_dict() != _external_json_object(
            _PICK_V9_S0_STAGE_PATH, purpose="Pick-v9 diagnostic stage authority"
        ):
            mismatches.append("program S0 differs from the reviewed Pick-v9 stage0")

    if mismatches:
        raise ValueError(
            "control-profile diagnostic contract mismatch: " + "; ".join(mismatches)
        )
    return profile


def require_s0_control_profile_input_identity(
    *,
    control_profile: Any,
    joint_velocity_force_task_id: Any = None,
    bundle_manifest: Any,
    initial_observation: dict[str, Any],
    subject_name: str,
    reference_name: str | None,
    registered_control_profile: Any = None,
) -> dict[str, Any] | None:
    """Verify the pre-registered Pick input without calculating a new digest."""
    profile = validate_control_profile(control_profile, allow_none=True)
    if profile is None:
        return None
    pick_input_identity = _external_json_object(
        _PICK_INPUT_IDENTITY_PATH, purpose="registered input identity"
    )
    task_id = validate_joint_velocity_force_task_id(
        joint_velocity_force_task_id,
        allow_none=True,
    )
    registered_profile = validate_control_profile(
        registered_control_profile,
        allow_none=True,
    )
    if task_id is None and registered_profile is not None:
        raise ValueError(
            "registered_control_profile requires a registered task"
        )
    if task_id is not None:
        causal_pick_input = (
            task_id == "pick_v9"
            and profile in (
                CONTROL_PROFILE_LEGACY_VELOCITY_WIDTH,
                CONTROL_PROFILE_LEGACY_VELOCITY_EFFORT,
            )
        )
        expected_registered_profile = (
            CONTROL_PROFILE_JOINT_VELOCITY_FORCE
            if registered_profile is None
            else registered_profile
        )
        if profile != expected_registered_profile and not causal_pick_input:
            raise ValueError(
                "registered input identity requires control_profile="
                f"{expected_registered_profile!r}"
            )
        registration = JOINT_VELOCITY_FORCE_TASK_REGISTRY[task_id]
        manifest = json.loads(Path(bundle_manifest).read_text(encoding="utf-8"))
        errors: list[str] = []
        scene_manifest_authority = _external_json_object(
            _SCENE_MANIFEST_AUTHORITY_PATH, purpose="registered scene manifest authority"
        )
        if manifest != scene_manifest_authority:
            errors.append("complete parsed scene manifest authority")
        raw_camera = manifest.get("camera_calibration")
        camera = raw_camera if isinstance(raw_camera, dict) else {}
        authority_camera = scene_manifest_authority.get("camera_calibration")
        if not isinstance(authority_camera, dict) or not authority_camera.get("sha256"):
            raise ValueError("External scene authority must declare camera calibration identity")
        if (
            camera.get("camera_serial") != authority_camera.get("camera_serial")
            or camera.get("sha256") != pick_input_identity["camera_sha256"]
        ):
            errors.append("camera calibration identity")
        objects = {
            row.get("name"): row
            for row in manifest.get("objects", [])
            if isinstance(row, dict)
        }
        eraser = objects.get("eraser") or {}
        box = objects.get("spacemouse_box") or {}
        if (
            eraser.get("label") != "felt blackboard eraser"
            or eraser.get("refined_json")
            != "eraser/pose/eraser_refined.json"
            or (eraser.get("artifact_sha256") or {}).get(
                "eraser/pose/eraser_refined.json"
            )
            != pick_input_identity["eraser_refined_sha256"]
        ):
            errors.append("manifest eraser source identity")
        if (
            box.get("label") != "white product box"
            or box.get("refined_json")
            != "spacemouse_box/pose/spacemouse_box_refined.json"
            or (box.get("artifact_sha256") or {}).get(
                "spacemouse_box/pose/spacemouse_box_refined.json"
            )
            != pick_input_identity["box_refined_sha256"]
        ):
            errors.append("manifest box identity")
        if (
            subject_name != registration["subject"]
            or reference_name != registration["reference"]
        ):
            errors.append("bound task roles")

        observation = dict(initial_observation or {})
        authority_relative_path = registration["initial_observation_path"]
        if registration["manifest_contains_registered_subject"]:
            authority_path = Path(bundle_manifest).parent / authority_relative_path
            authority_source = "sealed_manifest_refined_json"
        else:
            authority_path = external_input_path(
                authority_relative_path, purpose="registered initial observation"
            )
            authority_source = "external_input_artifact"
        try:
            authority_observation = json.loads(
                authority_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            if _env_flag(STRICT_SCENE_SEAL_ENV):
                raise ValueError(
                    "joint_velocity_force four-task initial observation "
                    f"authority is unavailable or invalid: "
                    f"{authority_relative_path}"
                ) from exc
            # A scene that was never registered has no authority file to read.
            # That is the ordinary case now, not an error.
            authority_observation = None
            errors.append("initial observation authority unavailable")
        if authority_observation is not None and observation != authority_observation:
            errors.append("initial observation parsed JSON deep equality")
        if errors:
            # The seal answers "were these the pre-registered inputs?". It
            # cannot answer "what did this episode actually use?", and treating
            # the first as a precondition for RUNNING made every
            # re-reconstruction unrunnable: the ZED moved 290.3 mm on
            # 2026-08-17, so the sealed manifest describes a scene that no
            # longer physically exists and nothing reconstructed since can
            # match it. It records instead of blocking now. Set
            # OAP_STRICT_SCENE_SEAL=1 to restore the refusal -- which is
            # what reproducing a result whose claim rests on the seal needs.
            if _env_flag(STRICT_SCENE_SEAL_ENV):
                raise ValueError(
                    "joint_velocity_force four-task input identity mismatch: "
                    + ", ".join(errors)
                )
            return {
                "schema": "oap_unregistered_scene_input_v1",
                "input_identity": "unregistered_scene",
                "preregistered": False,
                "control_profile": profile,
                "registered_task_id": task_id,
                "mismatched_against_registry": list(errors),
                # Not pre-registered is not the same as unprovenanced: the
                # digest of what actually ran is recorded either way.
                "bundle_manifest": str(bundle_manifest),
                "bundle_manifest_sha256": _sha256_path(bundle_manifest),
                "subject_name": subject_name,
                "reference_name": reference_name,
            }
        return {
            "schema": "oap_joint_velocity_force_four_task_input_v1",
            "registry_base_commit": (
                JOINT_VELOCITY_FORCE_TASK_REGISTRY_BASE_COMMIT
            ),
            "runtime_source_identity_scope": (
                "not_claimed_by_registry; require exact deployment HEAD evidence"
            ),
            "control_profile": profile,
            "registered_task_id": task_id,
            "program_path": registration["program_path"],
            "program_instruction": registration["instruction"],
            "program_provenance": registration["provenance"],
            "program_stage_count": registration["stage_count"],
            "program_stage_names": list(registration["stage_names"]),
            "subject": registration["subject"],
            "reference": registration["reference"],
            "subject_identity_source": (
                "explicit_offline_dataset_rewrite"
                if not registration["manifest_contains_registered_subject"]
                else "sealed_manifest_object"
            ),
            "manifest_contains_registered_subject": registration[
                "manifest_contains_registered_subject"
            ],
            "manifest_subject_source": (
                registration["scene"]["offline_dataset_subject_source_name"]
            ),
            "visual_mesh_identity_source": (
                "tracked_repository_artifact"
                if task_id == "cup_065e_v7"
                else "sealed_manifest_artifact"
            ),
            "scene": dict(registration["scene"]),
            "real_table_z": -0.021,
            "sim_tcp_m": 0.19812,
            "sim_tcp_anchor": "site",
            "planner_table_contact_force_weight": (
                joint_velocity_force_registered_table_force_weight(task_id)
            ),
            "legacy_unregistered_paper_mppi_default_table_force_weight": 0.1,
            "table_force_policy": (
                "registered_common_zero; contact permission remains in typed "
                "program predicates and hard validity"
            ),
            "initial_observation_identity": registration[
                "initial_observation_identity"
            ],
            "initial_observation_authority_path": authority_relative_path,
            "initial_observation_authority_source": authority_source,
            "initial_observation_comparison": (
                "parsed_json_deep_equality_no_allowlist"
            ),
            "camera_serial": camera.get("camera_serial"),
            "camera_sha256": camera.get("sha256"),
            "eraser_refined_sha256": pick_input_identity[
                "eraser_refined_sha256"
            ],
            "box_refined_sha256": pick_input_identity["box_refined_sha256"],
        }
    manifest = json.loads(Path(bundle_manifest).read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest.get("schema") != "real2sim2real_scene_manifest_v1":
        errors.append("bundle schema")
    camera = manifest.get("camera_calibration") or {}
    if camera.get("sha256") != pick_input_identity["camera_sha256"]:
        errors.append("camera calibration identity")
    objects = {
        row.get("name"): row
        for row in manifest.get("objects", [])
        if isinstance(row, dict)
    }
    eraser = objects.get("eraser") or {}
    box = objects.get("spacemouse_box") or {}
    if (
        eraser.get("label") != "felt blackboard eraser"
        or eraser.get("refined_json") != "eraser/pose/eraser_refined.json"
        or (eraser.get("artifact_sha256") or {}).get(
            "eraser/pose/eraser_refined.json"
        )
        != pick_input_identity["eraser_refined_sha256"]
    ):
        errors.append("eraser bundle identity")
    if (
        box.get("label") != "white product box"
        or box.get("refined_json")
        != "spacemouse_box/pose/spacemouse_box_refined.json"
        or (box.get("artifact_sha256") or {}).get(
            "spacemouse_box/pose/spacemouse_box_refined.json"
        )
        != pick_input_identity["box_refined_sha256"]
    ):
        errors.append("reference bundle identity")
    # This diagnostic intentionally contains only Pick-v9 S0.  S0 names the
    # movable eraser but not the later placement box, so structural binding
    # must produce no external reference even though the sealed bundle still
    # verifies the box artifact above.
    if subject_name != "eraser" or reference_name is not None:
        errors.append("bound task roles")
    observation = dict(initial_observation or {})
    if observation.get("schema") != "any6d_foundationpose_pose_scale_v1":
        errors.append("initial observation schema")
    for field, expected in (
        ("pos_base", pick_input_identity["initial_pos_base"]),
        ("size_LWH_m", pick_input_identity["initial_size_lwh_m"]),
    ):
        actual = observation.get(field)
        if (
            not isinstance(actual, list)
            or len(actual) != len(expected)
            or any(
                not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)
                for a, b in zip(actual, expected)
            )
        ):
            errors.append(f"initial observation {field}")
    if not math.isclose(
        float(observation.get("yaw_base_rad", math.nan)),
        float(pick_input_identity["initial_yaw_base_rad"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        errors.append("initial observation yaw_base_rad")
    if not bool((observation.get("quality_gate") or {}).get("passed", False)):
        errors.append("initial observation quality gate")
    if observation.get("size_lwh_obb") is not None:
        errors.append("initial observation size_lwh_obb must be absent")
    if errors:
        raise ValueError(
            "control-profile diagnostic input identity mismatch: "
            + ", ".join(errors)
        )
    return {
        "schema": "oap_pick_v9_s0_input_identity_v1",
        "control_profile": profile,
        "bundle_schema": manifest["schema"],
        "camera_sha256": camera["sha256"],
        "eraser_refined_sha256": pick_input_identity["eraser_refined_sha256"],
        "box_refined_sha256": pick_input_identity["box_refined_sha256"],
        "subject": subject_name,
        "reference": None,
        "initial_pos_base": list(pick_input_identity["initial_pos_base"]),
        "initial_size_lwh_m": list(pick_input_identity["initial_size_lwh_m"]),
        "initial_yaw_base_rad": pick_input_identity["initial_yaw_base_rad"],
    }
