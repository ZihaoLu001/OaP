"""Top-level closed-loop orchestrator for ``oap-run``.

The flow is observe -> synthesize/load and validate one program -> bind its
explicit scene roles -> build the twin -> run the program's declared stages.
Each stage repeatedly plans, applies one prefix, reobserves, and evaluates its
measured terminal. An evaluably false terminal consumes the same stage's
remaining cycle budget; an unknown terminal stops fail-closed.

One :class:`~oap.program.TaskProgram` object is created per episode
and its canonical hash is logged at trajectory cost, physics feasibility,
terminal evaluation, and final verification. There are no task-family or
stage-name branches: task semantics enter through the program's running and
terminal predicates. Every stage starts from a content-free hold of the latest
acknowledged actuator command; only later cycles of that same stage shift the
previous winner.
"""
from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from oap.execution_budget import (
    execution_budget_identity,
    validate_execution_budget_protocol,
)
from oap.loop import observe as observe_mod
from oap.loop import safety
from oap.loop.evidence_calibration import (
    ObjectHeldEvidenceCalibration,
    load_object_held_evidence_calibration,
    require_real_object_held_evidence,
)
from oap.loop.object_held_geometry import (
    OrientedBox,
    closing_region_intersection,
)
from oap.loop.executor_readiness import evaluate_executor_readiness
from oap.loop.execution_prefix import (
    DEFAULT_EXECUTION_PREFIX_FRACTION,
    resolve_execution_prefix,
    validate_execution_prefix_fraction,
    validate_execution_prefix_steps,
    validate_mppi_stage_execution_prefix_steps,
)
from oap.loop.episode import EpisodeLog
from oap.loop.plan import num_knots_from_env, validate_num_knots
from oap.loop.sampling import (
    CANDIDATE_SELECTION_RESULT_TERMINAL_EARLIEST,
    CANDIDATE_SELECTION_VALID_THEN_TOTAL_COST,
    pool_size_from_env,
    validate_candidate_selection_mode,
    validate_pool_size,
)
from oap.program import (
    TaskProgram,
    ProgramBindingError,
    bind_program_to_scene,
    groundability_failures,
    synthesize,
)
from oap.program.cost_shaping import (
    cost_shaping_identity,
    cost_shaping_profile_binds,
    validate_cost_shaping_profile,
)
from oap.twin import (
    TABLE_TOP_Z,
    REAL_GN01_TCP_M,
    SceneObject,
    build_plan_scene,
    load_obj_vertices,
    load_scene_manifest,
    load_world,
    mesh_obb_from_vertices,
    object_body_quat,
    patch_sim_gripper_tcp,
    reset_object_pose,
    roll_q_om_for_standing_dims,
)
from oap.twin.assets import GN01_OPEN_WIDTH_M
from oap.twin.obb import seat_unheld_pose_on_support
from oap.twin.manifest import _project_objects_to_plan_support
from oap.twin.types import LAB_TABLE_POS, LAB_TABLE_SIZE
from oap.twin.control_profile import (
    CONTROL_PROFILE_DIRECT_TORQUE_FORCE,
    CONTROL_PROFILE_JOINT_VELOCITY_FORCE,
    CONTROL_PROFILE_LEGACY_VELOCITY_WIDTH,
    JOINT_VELOCITY_FORCE_TRIAL_BUDGET_COST_FOCUS_STAGE90_V1,
    JOINT_VELOCITY_FORCE_TRIAL_BUDGET_COST_FOCUS_SCREEN30_V1,
    validate_joint_velocity_force_trial_budget,
    validate_joint_velocity_force_algorithm_arm,
    validate_joint_velocity_force_dial_arm,
    validate_joint_velocity_force_phase3_finalist_arm,
    validate_joint_velocity_force_phase3_seed,
    control_profile_compound_experiment_id,
    joint_velocity_force_algorithm_arm_telemetry,
    joint_velocity_force_dial_arm_telemetry,
    joint_velocity_force_phase3_finalist_arm_telemetry,
    joint_velocity_force_cost_focus_stage90_telemetry,
    joint_velocity_force_cost_focus_screen30_telemetry,
    joint_velocity_force_device_proposal_seed,
    evaluation_device_seed,
    direct_torque_s0_coverage_telemetry,
    direct_torque_s0_horizon_prefix_telemetry,
    joint_velocity_force_registered_table_force_weight,
    require_s0_control_profile_contract,
    require_s0_control_profile_input_identity,
    validate_control_profile,
    validate_joint_velocity_force_task_id,
    validate_direct_torque_s0_coverage_k,
    validate_direct_torque_s0_horizon_prefix_arm,
)
from oap.twin.control_regularizer import (
    ExecutedActionState,
    UNIFIED_CONTROL_REGULARIZER_V1,
    control_regularizer_formula_identity,
    validate_control_regularizer_calibration_profile,
)
from oap.twin.unified_mppi_effort import (
    unified_mppi_effort_identity,
    unified_mppi_effort_table_force_weight,
    unified_mppi_effort_requires_tool_mediation_success,
    unified_mppi_effort_uses_proven_pick_carrier,
    split_unified_mppi_effort_ablation,
    require_unified_mppi_effort_contract,
    unified_mppi_effort_task_id,
    require_unified_mppi_effort_input_identity,
    validate_unified_mppi_effort_profile_arg,
)
from oap.twin.pick_v8_causal import (
    pick_v8_causal_controller_identity,
    require_pick_v8_causal_input_identity,
    require_pick_v8_causal_contract,
    validate_pick_v8_causal_profile,
)
from oap.twin.batched_rollout import (
    MTP_JAW_MODE_PRESERVE_NOMINAL,
    PREDICTIVE_SAMPLER_MPPI_HALTON_MODES,
    PREDICTIVE_SAMPLER_MTP_GLOBAL_LOCAL,
    PREDICTIVE_SAMPLER_REFERENCE_EFFORT_HALTON,
    PREDICTIVE_SAMPLER_SINGLE_SCALE,
    PREDICTIVE_TWO_SCALE_LOCAL_SIGMA_FRACTION,
    RIZON4S_MPC_TORQUE_LIMIT_NM,
    DEFAULT_ARM_VELOCITY_WEIGHT,
    DEFAULT_MPPI_TEMPERATURE,
    MPPI_EXECUTION_SOFTMAX_WEIGHTED_MEAN,
    cem_elite_fraction_from_env,
    cem_min_std_fraction_from_env,
    cem_rounds_from_env,
    horizon_steps_from_env,
    local_sigma_fraction_from_env,
    mppi_gripper_command_channels,
    validate_cem_elite_fraction,
    validate_cem_rounds,
    validate_horizon_steps,
    validate_local_sigma_fraction,
    validate_mtp_jaw_mode,
    validate_predictive_sampler_mode,
    validate_mppi_temperature,
    validate_mppi_execution_mode,
    validate_arm_velocity_weight,
)
from oap.utils.io import write_json_atomic

logger = logging.getLogger("oap.loop.runner")


def _initialize_control_profile_jaw_carrier(
    plan_world: Any,
    *,
    control_profile: str,
    pick_v8_causal_profile: str | None = None,
    unified_mppi_effort_profile: str | None = None,
) -> dict[str, Any]:
    """Put the scratch experiment's host carrier in profile-native units."""
    from oap.twin.control_knots import NJ, JawCommandState

    profile = validate_control_profile(control_profile)
    asymmetric_effort = False
    if pick_v8_causal_profile is not None:
        from oap.twin.pick_v8_causal import (
            pick_v8_causal_uses_asymmetric_effort,
        )

        asymmetric_effort = pick_v8_causal_uses_asymmetric_effort(
            pick_v8_causal_profile
        )
    if unified_mppi_effort_profile is not None:
        asymmetric_effort = unified_mppi_effort_uses_proven_pick_carrier(
            unified_mppi_effort_profile
        )
    ctrl = np.asarray(plan_world.data.ctrl, dtype=float).reshape(-1)
    if ctrl.size <= NJ:
        raise RuntimeError(
            "control-profile diagnostic requires the GN01 host carrier column"
        )
    seeded_effort_n = float(ctrl[NJ])
    if not np.isclose(seeded_effort_n, -80.0, rtol=0.0, atol=1e-9):
        raise RuntimeError(
            "control-profile diagnostic expected clean-2ef to seed the GN01 "
            f"direct open command at -80 N, got {seeded_effort_n!r}"
        )
    direct_state = JawCommandState.from_continuous_effort(
        seeded_effort_n,
        source="clean_2ef_load_world_direct_effort_seed",
    )
    native_command, semantic_latent = mppi_gripper_command_channels(
        np.asarray(direct_state.latent),
        control_profile=profile,
        pick_v8_causal_profile=pick_v8_causal_profile,
        unified_mppi_effort_profile=unified_mppi_effort_profile,
    )
    native_value = float(np.asarray(native_command))
    ctrl[NJ] = native_value
    plan_world.data.ctrl[:] = ctrl
    return {
        "schema": "oap_control_profile_stage_entry_v1",
        "control_profile": profile,
        "source_profile": CONTROL_PROFILE_DIRECT_TORQUE_FORCE,
        "source_command": seeded_effort_n,
        "source_command_units": "effort_n",
        "semantic_latent": float(np.asarray(semantic_latent)),
        "carrier_command": native_value,
        "carrier_command_units": (
            "width_m"
            if profile == CONTROL_PROFILE_LEGACY_VELOCITY_WIDTH
            else "effort_n"
        ),
        "transcode": (
            "direct_effort_to_signed_latent_to_legacy_width_once"
            if profile == CONTROL_PROFILE_LEGACY_VELOCITY_WIDTH
            else "direct_effort_to_signed_latent_to_asymmetric_effort_once"
            if asymmetric_effort
            else "direct_effort_identity"
        ),
        "carrier_role": "host_orchestration_only_not_physics_or_measurement",
    }


def _fixed_mpc_torque_envelope(
    effective_joint_limits: dict[str, Any],
) -> np.ndarray:
    """Bind hardware to the same fixed torque envelope as every experiment."""
    runtime_max = np.asarray(
        effective_joint_limits.get("base_torque_max_nm"), dtype=float
    )
    profile_scale = float(
        effective_joint_limits.get("max_joint_torque_scale", 0.0)
    )
    fixed = np.asarray(RIZON4S_MPC_TORQUE_LIMIT_NM, dtype=float)
    if (
        runtime_max.shape != fixed.shape
        or not np.all(np.isfinite(runtime_max))
        or np.any(runtime_max <= 0.0)
        or not np.isfinite(profile_scale)
        or not 0.0 < profile_scale <= 1.0
    ):
        raise safety.SafetyError(
            "controller did not publish a valid Rizon torque envelope"
        )
    available = runtime_max * profile_scale
    if np.any(available + 1e-12 < fixed):
        raise safety.SafetyError(
            "controller/profile torque envelope is below the fixed project MPC "
            "envelope; refusing a hardware-specific silent rescale"
        )
    return fixed.copy()

__all__ = [
    "LoopConfig",
    "measured_program_success",
    "profile_sealed_summary_success",
    "run_episode",
]


def measured_program_success(
    *,
    terminal_success: object,
) -> bool:
    """Return the only condition allowed to become process success.

    ``terminal_success`` is the final :class:`TaskProgram` terminal
    evaluated on the fresh post-prefix observation. Missing, ungroundable, or
    unreliable terminal evidence leaves this value false. Running predicates,
    mechanism diagnostics, and outcome labels never enter process success.
    """
    return terminal_success is True


def profile_sealed_summary_success(
    *,
    terminal_success: object,
    result_success: object,
    unified_mppi_effort_profile: str | None,
) -> bool:
    """Promote the exact EpisodeResult gate only for sealed toolpush."""
    if unified_mppi_effort_requires_tool_mediation_success(
        unified_mppi_effort_profile
    ):
        return result_success is True
    return measured_program_success(terminal_success=terminal_success)


def _validate_candidate_selection_config(
    value: Any,
    *,
    optimizer: Any,
    sampler_mode: str,
) -> str:
    """Validate the task-aware result selector at the runner boundary."""
    mode = validate_candidate_selection_mode(value)
    if mode == CANDIDATE_SELECTION_RESULT_TERMINAL_EARLIEST:
        if str(optimizer).strip().lower() != "mppi":
            raise ValueError(
                "result_terminal_earliest requires optimizer='mppi'"
            )
        if sampler_mode not in (
            *PREDICTIVE_SAMPLER_MPPI_HALTON_MODES,
            PREDICTIVE_SAMPLER_MTP_GLOBAL_LOCAL,
        ):
            raise ValueError(
                "result_terminal_earliest requires an MPPI sampler"
            )
    return mode


def _candidate_selection_call_kwargs(mode: Any) -> dict[str, str]:
    """Omit the legacy default from downstream call packets exactly."""
    validated = validate_candidate_selection_mode(mode)
    if validated == CANDIDATE_SELECTION_VALID_THEN_TOTAL_COST:
        return {}
    return {"candidate_selection_mode": validated}


def _mppi_execution_call_kwargs(mode: Any) -> dict[str, str]:
    """Omit the legacy softmax execution policy from call packets exactly."""
    validated = validate_mppi_execution_mode(mode)
    if validated == MPPI_EXECUTION_SOFTMAX_WEIGHTED_MEAN:
        return {}
    return {"mppi_execution_mode": validated}


# ==========================================================================
# Configuration
# ==========================================================================
@dataclass
class LoopConfig:
    """Every knob of one closed-loop episode (the CLI maps 1:1 onto this).

    Dry-run is the DEFAULT: with ``execute=False`` nothing ever connects to the
    robot. Real motion additionally requires ``i_confirm_real_motion`` and a
    home-gate pass (see :mod:`oap.loop.safety`).
    """

    # scene + task
    bundle_manifest: Path = Path("manifest.json")
    task: str = ""
    out_dir: Path = Path("runs/episode")
    # program synthesis
    synth_backend: str = "anthropic"
    program_json: Path | None = None
    # observation
    observer_python: Path | None = None
    foundationpose_python: Path | None = None
    foundationpose_root: Path | None = None
    foundationpose_mesh: Path | None = None
    foundationpose_timeout_s: float = 300.0
    # Maximum wait for a fresh packet from the already-running continuous
    # tracker.  This is deliberately separate from the slower worker/camera
    # RPC budget above and has no guessed real-execution default.
    foundationpose_checkpoint_timeout_s: float | None = None
    foundationpose_mode: str = "off"                 # 'off' | 'actual'
    subject_pose_policy: str = "foundationpose"      # 'foundationpose' | 'gravity_obb'
    fp_seed_from_obb: bool = False
    relocalize_all_each_chunk: bool = False
    disable_held_verify: bool = False
    offline: bool = False
    initial_obs_json: Path | None = None
    # Explicit, offline-only dataset visual/collision substitution. This is a
    # simulation demo adapter, never an authorization path for real motion.
    offline_dataset_subject_source_name: str | None = None
    offline_dataset_subject_name: str | None = None
    offline_dataset_subject_label: str | None = None
    offline_dataset_visual_mesh: Path | None = None
    offline_dataset_size_lwh: tuple[float, float, float] | None = None
    offline_dataset_mass_kg: float | None = None
    offline_dataset_rgba: tuple[float, float, float, float] | None = None
    offline_dataset_collision_primitive: str | None = None
    # twin
    table_spec_json: Path | None = None
    table_spec_height: bool = False
    real_table_z: float | None = None
    snap_subject_to_table: bool = False
    collision_from_mesh: bool = False
    sim_tcp_m: float | None = None
    sim_tcp_anchor: str = "site"
    home_posture_json: Path | None = None
    # planning
    pool_size: int = field(default_factory=pool_size_from_env)
    horizon_steps: int = field(default_factory=horizon_steps_from_env)
    num_knots: int = field(default_factory=num_knots_from_env)
    sigma_fraction: float = field(
        default_factory=local_sigma_fraction_from_env
    )
    sampler_mode: str = PREDICTIVE_SAMPLER_SINGLE_SCALE
    mtp_jaw_mode: str = MTP_JAW_MODE_PRESERVE_NOMINAL
    local_sigma_fraction: float = (
        PREDICTIVE_TWO_SCALE_LOCAL_SIGMA_FRACTION
    )
    cem_rounds: int = field(default_factory=cem_rounds_from_env)
    cem_elite_fraction: float = field(
        default_factory=cem_elite_fraction_from_env
    )
    cem_min_std_fraction: float = field(
        default_factory=cem_min_std_fraction_from_env
    )
    optimizer: str = "cem"
    control_profile: str | None = None
    control_profile_smoke_one_cycle: bool = False
    direct_torque_s0_coverage_k: int | None = None
    mppi_temperature: float = DEFAULT_MPPI_TEMPERATURE
    mppi_execution_mode: str = MPPI_EXECUTION_SOFTMAX_WEIGHTED_MEAN
    execution_prefix_fraction: float = DEFAULT_EXECUTION_PREFIX_FRACTION
    execution_prefix_steps: int | None = None
    # Optional offline diagnostic cadence, one exact prefix per semantic
    # stage. The default remains the single paper-style episode prefix.
    mppi_stage_execution_prefix_steps: tuple[int, ...] | None = None
    max_stage_cycles: int = 9          # one continuous stage visit; no cold attempts
    # The paper's single-objective demo uses one episode-wide loop.  Our
    # multi-stage task programs can explicitly opt into a fresh budget after
    # each measured semantic gate instead.
    mppi_cycle_budget_scope: str = "episode"
    seed: int = 27
    # Optional H100 solver. All fields are required together; URL must be an
    # SSH-forwarded loopback endpoint. Hardware ownership remains local.
    remote_planner_url: str | None = None
    remote_planner_model_sha256: str | None = None
    remote_planner_assets_sha256: str | None = None
    remote_planner_sha256: str | None = None
    remote_planner_service_instance_id: str | None = None
    remote_planner_deadline_s: float | None = None
    # execution
    execute: bool = False
    prepare_real_execution: bool = False
    # Human verification latch.  The two booleans below are derived from this
    # exact approved file at runtime and are never exposed as CLI switches.
    joint_executor_verification_json: Path | None = None
    joint_executor_verification_sha256: str | None = None
    joint_executor_hardware_verified: bool = False
    joint_executor_verification_latch_verified: bool = False
    joint_executor_software_fingerprint_sha256: str | None = None
    i_confirm_real_motion: bool = False
    approved_program_sha256: str | None = None
    readiness_authorization_token: Path | None = None
    operator_estop_packet_json: Path | None = None
    # These fields are written only by the signed-token consumer immediately
    # before controller access.  They have no CLI boolean/string bypass.
    execution_authorization_consumed: bool = False
    execution_authorization_token_id: str | None = None
    execution_authorization_token_sha256: str | None = None
    execution_authorization_expires_at: str | None = None
    execution_authorization_server_identity: dict[str, Any] | None = None
    hardware_evidence_calibration_json: Path | None = None
    hardware_evidence_calibration_sha256: str | None = None
    hardware_evidence_calibration_source: str | None = None
    hardware_evidence_calibration_hardware_id: str | None = None
    # Hardware-calibrated evidence thresholds. None means "cannot verify",
    # never a guessed default.
    gripper_air_close_width_m: float | None = None
    gripper_width_resolution_m: float | None = None
    # Hardware-calibrated timing/tracking limits. None means no real-motion
    # readiness contract; execute/prepare therefore fail closed.
    max_observation_age_s: float | None = None
    max_camera_robot_skew_s: float | None = None
    max_plan_staleness_s: float | None = None
    max_final_joint_tracking_error_rad: float | None = None
    min_terminal_anchor_reliability: float | None = None
    server_host: str = "ROBOT-HOST-PLACEHOLDER"
    server_port: int = 8766
    grasp_tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    home_first: bool = False
    allow_home_drift: bool = False
    per_chunk_confirm: bool = True
    # Default exit remains the safe automatic lift/open/home ritual. This
    # explicit experiment mode instead stops, records final_posture.json, and
    # leaves restoration to the attended operator.
    manual_post_experiment_restore: bool = False
    # Appended to retain the positional order of every pre-existing field.
    candidate_selection_mode: str = (
        CANDIDATE_SELECTION_VALID_THEN_TOTAL_COST
    )
    arm_velocity_weight: float = DEFAULT_ARM_VELOCITY_WEIGHT
    record_candidate_cost_telemetry: bool = False
    direct_torque_s0_horizon_prefix_arm: str | None = None
    joint_velocity_force_task_id: str | None = None
    # Derived from the categorical four-task registration.  There is no CLI
    # knob: None preserves all existing profiles, while a registered task is
    # normalized to the one common value 0.0 below.
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
    # The program is machine-synthesized (VLM), not the sealed authority file.
    # Under a unified profile this relaxes exactly ONE check -- the
    # program==authority equality -- and in exchange REQUIRES machine
    # provenance on the program and baseline (all-1.0) cost shaping, so a
    # hand-edited program cannot ride in under the flag and no hand-tuned
    # stage multiplier can leak into a generated program's run.
    vlm_generated_program: bool = False
    # Derived from exact task/program authority; never a CLI selector.
    unified_mppi_effort_task_id: str | None = None
    # Protocol-level limit applied by the paper entry point.  It is separate
    # from controller ablations and is recorded as such in every artifact.
    execution_budget_protocol: str | None = None
    # Tier-2 judge (veto-only)

    def __post_init__(self) -> None:
        explicit_device_seed = evaluation_device_seed()
        if explicit_device_seed is not None:
            if not self.offline or self.execute or self.unified_mppi_effort_profile is None:
                raise ValueError("evaluation device seed requires an offline unified-MPPI run")
            if self.seed != explicit_device_seed:
                raise ValueError("evaluation device seed must match the recorded --seed")
        self.pool_size = (
            pool_size_from_env()
            if self.pool_size is None
            else validate_pool_size(self.pool_size)
        )
        self.horizon_steps = validate_horizon_steps(self.horizon_steps)
        self.num_knots = validate_num_knots(self.num_knots)
        self.sigma_fraction = validate_local_sigma_fraction(
            self.sigma_fraction
        )
        self.sampler_mode = validate_predictive_sampler_mode(
            self.sampler_mode
        )
        self.candidate_selection_mode = (
            _validate_candidate_selection_config(
                self.candidate_selection_mode,
                optimizer=self.optimizer,
                sampler_mode=self.sampler_mode,
            )
        )
        self.mtp_jaw_mode = validate_mtp_jaw_mode(self.mtp_jaw_mode)
        if (
            self.sampler_mode != PREDICTIVE_SAMPLER_MTP_GLOBAL_LOCAL
            and self.mtp_jaw_mode != MTP_JAW_MODE_PRESERVE_NOMINAL
        ):
            raise ValueError(
                "sampled MTP jaw mode requires "
                "sampler_mode='mtp_global_local'"
            )
        self.local_sigma_fraction = validate_local_sigma_fraction(
            self.local_sigma_fraction
        )
        self.mppi_execution_mode = validate_mppi_execution_mode(
            self.mppi_execution_mode
        )
        self.arm_velocity_weight = validate_arm_velocity_weight(
            self.arm_velocity_weight
        )
        self.control_profile = validate_control_profile(
            self.control_profile, allow_none=True
        )
        self.direct_torque_s0_coverage_k = (
            validate_direct_torque_s0_coverage_k(
                self.direct_torque_s0_coverage_k,
                allow_none=True,
            )
        )
        self.direct_torque_s0_horizon_prefix_arm = (
            validate_direct_torque_s0_horizon_prefix_arm(
                self.direct_torque_s0_horizon_prefix_arm,
                allow_none=True,
            )
        )
        self.joint_velocity_force_task_id = validate_joint_velocity_force_task_id(
            self.joint_velocity_force_task_id,
            allow_none=True,
        )
        self.control_regularizer_calibration_profile = (
            validate_control_regularizer_calibration_profile(
                self.control_regularizer_calibration_profile,
                allow_none=True,
            )
        )
        self.joint_velocity_force_trial_budget = (
            validate_joint_velocity_force_trial_budget(
                self.joint_velocity_force_trial_budget,
                allow_none=True,
            )
        )
        self.joint_velocity_force_algorithm_arm = (
            validate_joint_velocity_force_algorithm_arm(
                self.joint_velocity_force_algorithm_arm,
                allow_none=True,
            )
        )
        self.joint_velocity_force_dial_arm = (
            validate_joint_velocity_force_dial_arm(
                self.joint_velocity_force_dial_arm,
                allow_none=True,
            )
        )
        self.joint_velocity_force_phase3_finalist_arm = (
            validate_joint_velocity_force_phase3_finalist_arm(
                self.joint_velocity_force_phase3_finalist_arm,
                allow_none=True,
            )
        )
        self.joint_velocity_force_phase3_seed = (
            validate_joint_velocity_force_phase3_seed(
                self.joint_velocity_force_phase3_seed,
                allow_none=True,
            )
        )
        self.cost_shaping_profile = validate_cost_shaping_profile(
            self.cost_shaping_profile,
            allow_none=True,
        )
        # Preserve any ablation suffix.  This assignment normalised the field
        # to the BASE profile, and every consumer downstream -- including
        # require_unified_mppi_effort_contract, which builds its numeric
        # expectations from it -- then saw a run that had asked for N=1001 as if
        # it had asked for the base, and rejected it against the base's numbers.
        self.unified_mppi_effort_profile = (
            validate_unified_mppi_effort_profile_arg(
                self.unified_mppi_effort_profile,
                allow_none=True,
            )
        )
        self.pick_v8_causal_profile = validate_pick_v8_causal_profile(
            self.pick_v8_causal_profile,
            allow_none=True,
        )
        self.execution_budget_protocol = validate_execution_budget_protocol(
            self.execution_budget_protocol,
            allow_none=True,
        )
        if (
            self.execution_budget_protocol is not None
            and self.unified_mppi_effort_profile is None
        ):
            raise ValueError(
                "execution-budget protocol requires a unified MPPI profile"
            )
        if self.execution_budget_protocol is not None:
            _, controller_modifier = split_unified_mppi_effort_ablation(
                self.unified_mppi_effort_profile
            )
            if controller_modifier != "n1001":
                raise ValueError(
                    "paper execution-budget protocol requires 1000 "
                    "optimization proposals plus the registered verification "
                    "replays (__ablate_n1001)"
                )
        if (
            self.unified_mppi_effort_profile is not None
            and self.pick_v8_causal_profile is not None
        ):
            raise ValueError("Pick-v8 causal and unified profiles conflict")
        if self.unified_mppi_effort_profile is not None:
            derived_task_id = unified_mppi_effort_task_id(self)
            if self.unified_mppi_effort_task_id not in (None, derived_task_id):
                raise ValueError(
                    "unified_mppi_effort_v1 derived task identity mismatch"
                )
            self.unified_mppi_effort_task_id = derived_task_id
            expected_table_weight = unified_mppi_effort_table_force_weight(
                self.unified_mppi_effort_profile
            )
            if self.table_contact_force_weight not in (
                None,
                expected_table_weight,
            ):
                raise ValueError(
                    "unified MPPI profile fixes table_contact_force_weight="
                    f"{expected_table_weight!r}"
                )
            self.table_contact_force_weight = expected_table_weight
        elif self.joint_velocity_force_task_id is None:
            if self.table_contact_force_weight is not None:
                raise ValueError(
                    "table_contact_force_weight is registry-derived and "
                    "cannot be set independently"
                )
        else:
            if self.table_contact_force_weight not in (None, 0, 0.0):
                raise ValueError(
                    "four-task registry fixes table_contact_force_weight=0.0"
                )
            self.table_contact_force_weight = (
                joint_velocity_force_registered_table_force_weight(
                    self.joint_velocity_force_task_id
                )
            )
        if (
            self.sampler_mode not in PREDICTIVE_SAMPLER_MPPI_HALTON_MODES
            and self.local_sigma_fraction >= self.sigma_fraction
        ):
            raise ValueError(
                "two_scale requires local_sigma_fraction < sigma_fraction "
                "(broad)"
            )
        if (
            self.sampler_mode == PREDICTIVE_SAMPLER_REFERENCE_EFFORT_HALTON
            and str(self.optimizer).strip().lower() != "mppi"
        ):
            raise ValueError(
                "reference_effort_halton requires optimizer='mppi'"
            )
        self.execution_prefix_fraction = validate_execution_prefix_fraction(
            self.execution_prefix_fraction
        )
        if self.execution_prefix_steps is not None:
            self.execution_prefix_steps = validate_execution_prefix_steps(
                self.execution_prefix_steps
            )
        if self.mppi_stage_execution_prefix_steps is not None:
            self.mppi_stage_execution_prefix_steps = (
                validate_mppi_stage_execution_prefix_steps(
                    self.mppi_stage_execution_prefix_steps
                )
            )
            for stage_steps in self.mppi_stage_execution_prefix_steps:
                resolve_execution_prefix(
                    horizon_steps=self.horizon_steps,
                    execution_prefix_fraction=self.execution_prefix_fraction,
                    execution_prefix_steps=stage_steps,
                )
            if str(self.optimizer).strip().lower() != "mppi":
                raise safety.SafetyError(
                    "stage execution-prefix steps are supported only by MPPI"
                )
            if (
                not self.offline
                or self.execute
                or self.prepare_real_execution
                or self.remote_planner_url is not None
            ):
                raise safety.SafetyError(
                    "stage execution-prefix steps are offline local-MPPI-only"
                )
        if self.pick_v8_causal_profile is not None:
            require_pick_v8_causal_contract(self)
        elif self.unified_mppi_effort_profile is not None:
            require_unified_mppi_effort_contract(self)
        else:
            require_s0_control_profile_contract(self)

    def snapshot(self) -> dict[str, Any]:
        """JSON-safe copy of the config for the episode evidence packet."""
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if (
                (
                    f.name in {
                        "mppi_stage_execution_prefix_steps",
                        "control_profile",
                        "direct_torque_s0_coverage_k",
                        "direct_torque_s0_horizon_prefix_arm",
                        "joint_velocity_force_task_id",
                        "table_contact_force_weight",
                        "control_regularizer_calibration_profile",
                        "joint_velocity_force_trial_budget",
                        "joint_velocity_force_algorithm_arm",
                        "joint_velocity_force_dial_arm",
                        "joint_velocity_force_phase3_finalist_arm",
                        "joint_velocity_force_phase3_seed",
                        "cost_shaping_profile",
                        "unified_mppi_effort_profile",
                        "unified_mppi_effort_task_id",
                        "execution_budget_protocol",
                        "pick_v8_causal_profile",
                    }
                    and v is None
                )
                or (
                    f.name == "control_profile_smoke_one_cycle"
                    and v is False
                )
            ):
                # Preserve the historical default evidence packet exactly.
                continue
            out[f.name] = str(v) if isinstance(v, Path) else v
        if self.control_profile is not None:
            out["diagnostic_smoke_only"] = bool(
                self.control_profile_smoke_one_cycle
            )
            out["completed_cycle_limit"] = int(self.max_stage_cycles)
        return out


def _device_proposal_seed_evidence(
    *,
    trial_budget: str | None,
    dial_arm: str | None,
    phase3_arm: str | None,
    phase3_seed: int | None,
) -> dict[str, Any]:
    """Bind the numeric seed while preserving the runner packet vocabulary."""
    active_seed = joint_velocity_force_device_proposal_seed(
        trial_budget=trial_budget,
        dial_arm=dial_arm,
        phase3_arm=phase3_arm,
        phase3_seed=phase3_seed,
    )
    policy = (
        "explicit_evaluation_device_seed"
        if evaluation_device_seed() is not None
        else
        "registered_cost_focus_seed_active_only"
        if trial_budget in (
            JOINT_VELOCITY_FORCE_TRIAL_BUDGET_COST_FOCUS_STAGE90_V1,
            JOINT_VELOCITY_FORCE_TRIAL_BUDGET_COST_FOCUS_SCREEN30_V1,
        )
        else "registered_phase3_episode_seed"
        if phase3_arm is not None
        else "fixed_zero"
    )
    seed = active_seed if policy != "fixed_zero" else 0
    return {"proposal_seed_policy": policy, "device_seed": seed}


def _apply_collision_from_mesh(objects: list[SceneObject]) -> None:
    """Derive each object's collision cuboid from the ACTUAL visual mesh AABB.

    Compatibility-only library helper retained for focused geometry tests.
    The production CLI fixes this path off: reconstruction and the
    integrity-bound manifest select collision geometry automatically.
    """
    for o in objects:
        try:
            # load_scene_manifest derives q_om for every object at load time.
            assert o.q_om is not None
            ext, ctr = mesh_obb_from_vertices(load_obj_vertices(o.mesh), o.q_om)
            old = tuple(round(float(v), 4) for v in o.size_lwh)
            o.size_lwh = (float(ext[0]), float(ext[1]), float(ext[2]))
            o.collision_center_obj = (float(ctr[0]), float(ctr[1]), float(ctr[2]))
            logger.info("[collision-from-mesh] %s: size_lwh %s -> %s m, box center %s",
                        o.name, old, tuple(round(v, 4) for v in o.size_lwh),
                        [round(v, 4) for v in o.collision_center_obj])
        except Exception as exc:  # noqa: BLE001 - keep the reconstructed size on failure
            logger.warning("[collision-from-mesh] %s: SKIPPED (%r); keeping size_lwh", o.name, exc)


def _apply_offline_dataset_subject_override(
    cfg: LoopConfig,
    objects: list[SceneObject],
) -> None:
    """Replace one reconstructed subject with an explicit dataset mesh.

    The normal bundle contract remains untouched. This adapter exists only for
    offline simulator demonstrations of public dataset objects whose real
    camera reconstruction is intentionally unavailable. Every required field
    must be supplied together, and real/readiness execution is refused.
    """
    values = (
        cfg.offline_dataset_subject_source_name,
        cfg.offline_dataset_subject_name,
        cfg.offline_dataset_subject_label,
        cfg.offline_dataset_visual_mesh,
        cfg.offline_dataset_size_lwh,
        cfg.offline_dataset_mass_kg,
        cfg.offline_dataset_rgba,
        cfg.offline_dataset_collision_primitive,
    )
    if all(value is None for value in values):
        return
    if any(value is None for value in values):
        raise safety.SafetyError(
            "offline dataset subject override requires every dataset-subject flag"
        )
    if not cfg.offline or cfg.execute or cfg.prepare_real_execution:
        raise safety.SafetyError(
            "offline dataset subject override is simulation-only"
        )
    source_name = str(cfg.offline_dataset_subject_source_name)
    matches = [obj for obj in objects if obj.name == source_name]
    if len(matches) != 1 or not matches[0].movable:
        raise safety.SafetyError(
            f"offline dataset subject source {source_name!r} must name one movable object"
        )
    new_name = str(cfg.offline_dataset_subject_name)
    if (
        not new_name
        or new_name in {".", ".."}
        or Path(new_name).name != new_name
        or not new_name.replace("_", "").isalnum()
    ):
        raise safety.SafetyError(f"invalid offline dataset subject name {new_name!r}")
    if any(obj is not matches[0] and obj.name == new_name for obj in objects):
        raise safety.SafetyError(f"duplicate offline dataset subject name {new_name!r}")
    mesh = Path(cfg.offline_dataset_visual_mesh).expanduser().resolve()
    if not mesh.is_file() or mesh.suffix.lower() != ".obj":
        raise safety.SafetyError(
            f"offline dataset visual mesh must be an existing OBJ: {mesh}"
        )
    size = tuple(float(value) for value in cfg.offline_dataset_size_lwh)
    rgba = tuple(float(value) for value in cfg.offline_dataset_rgba)
    mass = float(cfg.offline_dataset_mass_kg)
    primitive = str(cfg.offline_dataset_collision_primitive)
    if (
        len(size) != 3
        or not np.all(np.isfinite(size))
        or min(size) <= 0.0
        or len(rgba) != 4
        or not np.all(np.isfinite(rgba))
        or min(rgba) < 0.0
        or max(rgba) > 1.0
        or not np.isfinite(mass)
        or mass <= 0.0
        or primitive not in {"box", "cylinder"}
    ):
        raise safety.SafetyError("invalid offline dataset geometry or physical properties")

    subject = matches[0]
    subject.name = new_name
    subject.label = str(cfg.offline_dataset_subject_label)
    subject.sam3_prompt = str(cfg.offline_dataset_subject_label)
    subject.mesh = mesh
    subject.visual_mesh = mesh
    subject.texture = None
    subject.size_lwh = size
    subject.q_om = np.array([1.0, 0.0, 0.0, 0.0])
    subject.mesh_long_axis = np.array([1.0, 0.0, 0.0])
    subject.collision_center_obj = (0.0, 0.0, 0.0)
    subject.mass_kg = mass
    subject.rgba = rgba
    subject.pose_quat = [1.0, 0.0, 0.0, 0.0]
    subject.yaw_ref_rad = 0.0
    subject.collision_primitive = primitive
    subject.open_top_container = False
    if getattr(cfg, "unified_mppi_effort_task_id", None) == "lab_cup_min20_v1":
        from oap.twin.lab_cup_fixture import apply_materials
        apply_materials(subject)
    subject.z_snapped_plan_frame = True
    if subject.pose_base is not None:
        subject.pose_base = [
            float(subject.pose_base[0]),
            float(subject.pose_base[1]),
            float(TABLE_TOP_Z + 0.5 * size[2]),
        ]
    logger.info(
        "[offline-dataset-subject] %s -> %s mesh=%s size=%s primitive=%s",
        source_name,
        new_name,
        mesh,
        size,
        primitive,
    )


# ==========================================================================
# Worker-client construction (lazy; heavy models stay in external envs)
# ==========================================================================
def _construct_worker(workers_client: Any, class_names: tuple[str, ...],
                      **kwargs: Any) -> Any:
    """Instantiate the first matching client class, filtering unknown kwargs."""
    for name in class_names:
        cls = getattr(workers_client, name, None)
        if cls is None:
            continue
        params = inspect.signature(cls).parameters
        kw = {k: v for k, v in kwargs.items() if k in params}
        return cls(**kw)
    raise RuntimeError(
        f"oap.workers.client provides none of {class_names}; the worker "
        f"package and the loop disagree -- update one side")


def _build_worker_clients(cfg: LoopConfig, out_dir: Path) -> dict[str, Any]:
    """Build only the worker clients this run needs (None otherwise)."""
    clients: dict[str, Any] = {"fp": None, "vlm": None}
    need_fp = (not cfg.offline and cfg.foundationpose_mode == "actual"
               and cfg.subject_pose_policy == "foundationpose")
    if not need_fp:
        return clients
    try:
        from oap.workers import client as workers_client
    except ImportError as exc:
        raise RuntimeError(
            "oap.workers.client is unavailable but this run needs a "
            "persistent worker (FoundationPose / VLM judge) -- "
            "reinstall the oap package") from exc
    if need_fp:
        clients["fp"] = _construct_worker(
            workers_client,
            ("FoundationPoseClient", "FoundationPoseWorkerClient", "FpWorkerClient"),
            python=cfg.foundationpose_python, py=cfg.foundationpose_python,
            root=cfg.foundationpose_root,
            log_path=out_dir / "foundationpose_worker.log", out_dir=out_dir)
    return clients


def _call_optional(obj: Any, method_names: tuple[str, ...]) -> None:
    """Call the first available method (shutdown hooks), tolerantly."""
    if obj is None:
        return
    for name in method_names:
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - a shutdown fault must not crash the run
                logger.warning("[workers] %s() failed: %r", name, exc)
            return


def _registration_backed_track_state_valid(
    visual: dict[str, Any],
    *,
    tracking_state: dict[str, Any],
    registration_score: float | None,
    min_registration_score: float | None,
    synchronized_capture: bool,
) -> bool:
    """Validate current maskless pose using scored registration provenance.

    The current frame deliberately has no numeric confidence. Identity quality
    comes from the episode's legal mask-scored registration, while freshness,
    loss, and pose validity come from the current continuous state.
    """
    if registration_score is None or min_registration_score is None:
        return False
    try:
        score = float(registration_score)
        threshold = float(min_registration_score)
        visibility = float(visual.get("visibility", 0.0))
        position = np.asarray(visual["pos_base"], dtype=float).reshape(3)
        quaternion = np.asarray(
            visual["quat_wxyz"], dtype=float
        ).reshape(4)
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        np.isfinite(score)
        and np.isfinite(threshold)
        and score >= threshold
        and tracking_state.get("state_valid") is True
        and visibility > 0.0
        and not bool(visual.get("track_lost", False))
        and synchronized_capture
        and np.all(np.isfinite(position))
        and np.all(np.isfinite(quaternion))
    )


def _current_subject_closing_region_evidence(
    plan_world: Any,
    visual: dict[str, Any],
) -> dict[str, Any]:
    """Attribute the current tracked subject to the measured GN01 pad gap.

    A scratch ``MjData`` receives both the current measured arm/gripper state
    and the current continuous FoundationPose subject pose before one FK
    refresh. All contact-active subject collision boxes are derived through the
    subject freejoint/body role, never an object or task name. The result reads
    only current transforms and dimensions; simulator contacts/history are not
    evidence for this predicate.
    """
    source = (
        "current_foundationpose_subject_pose+measured_robot_gripper_fk+"
        "integrity_bound_collision_geometry"
    )
    evidence: dict[str, Any] = {
        "schema": "oap_object_held_spatial_evidence_v1",
        "source": source,
        "observable": False,
        "intersects_closing_region": None,
        "reason_code": None,
        "reason": None,
        "contact_history_used": False,
    }

    def _fail(code: str, detail: object | None = None) -> dict[str, Any]:
        evidence["reason_code"] = code
        evidence["reason"] = code if detail is None else f"{code}:{detail}"
        return evidence

    try:
        import mujoco

        model = plan_world.model
        current_data = plan_world.data
        subject_qpos_addr = int(plan_world.object_qpos_addr)
        subject_joints = [
            joint_id
            for joint_id in range(int(model.njnt))
            if (
                int(model.jnt_qposadr[joint_id]) == subject_qpos_addr
                and int(model.jnt_type[joint_id])
                == int(mujoco.mjtJoint.mjJNT_FREE)
            )
        ]
        if len(subject_joints) != 1:
            return _fail(
                "subject_freejoint_role_ambiguous",
                len(subject_joints),
            )
        subject_body_id = int(model.jnt_bodyid[subject_joints[0]])

        def _belongs_to_subject(body_id: int) -> bool:
            current = int(body_id)
            visited: set[int] = set()
            while current >= 0 and current not in visited:
                if current == subject_body_id:
                    return True
                visited.add(current)
                parent = int(model.body_parentid[current])
                if parent == current:
                    break
                current = parent
            return False

        subject_geom_ids = [
            geom_id
            for geom_id in range(int(model.ngeom))
            if (
                _belongs_to_subject(int(model.geom_bodyid[geom_id]))
                and (
                    int(model.geom_contype[geom_id]) != 0
                    or int(model.geom_conaffinity[geom_id]) != 0
                )
            )
        ]
        if not subject_geom_ids:
            return _fail("subject_collision_geometry_missing")
        unsupported_subject_geoms = [
            int(geom_id)
            for geom_id in subject_geom_ids
            if int(model.geom_type[geom_id])
            != int(mujoco.mjtGeom.mjGEOM_BOX)
        ]
        if unsupported_subject_geoms:
            return _fail(
                "subject_collision_geometry_unsupported_non_box",
                unsupported_subject_geoms,
            )
        pad_names = {
            "left_pad": "gn01_left_finger_tip_collision",
            "right_pad": "gn01_right_finger_tip_collision",
        }
        pad_ids = {
            role: int(
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    name,
                )
            )
            for role, name in pad_names.items()
        }
        missing_pads = sorted(
            pad_names[role]
            for role, geom_id in pad_ids.items()
            if geom_id < 0
        )
        if missing_pads:
            return _fail("gn01_pad_geometry_missing", missing_pads)
        non_box_pads = sorted(
            pad_names[role]
            for role, geom_id in pad_ids.items()
            if int(model.geom_type[geom_id])
            != int(mujoco.mjtGeom.mjGEOM_BOX)
        )
        if non_box_pads:
            return _fail("gn01_pad_geometry_unsupported_non_box", non_box_pads)
        body_position = np.asarray(
            visual["pos_plan"], dtype=float
        ).reshape(3)
        body_quaternion = np.asarray(
            visual["quat_wxyz"], dtype=float
        ).reshape(4)
        quaternion_norm = float(np.linalg.norm(body_quaternion))
        if (
            not np.all(np.isfinite(body_position))
            or not np.all(np.isfinite(body_quaternion))
            or not np.isfinite(quaternion_norm)
            or quaternion_norm <= 0.0
        ):
            return _fail("tracked_subject_pose_invalid")
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = current_data.qpos
        scratch.qvel[:] = current_data.qvel
        scratch.ctrl[:] = current_data.ctrl
        if int(model.na) > 0:
            scratch.act[:] = current_data.act
        scratch.qpos[subject_qpos_addr:subject_qpos_addr + 3] = body_position
        scratch.qpos[subject_qpos_addr + 3:subject_qpos_addr + 7] = (
            body_quaternion / quaternion_norm
        )
        # Current-state geometry only: update body/site/geom transforms from
        # measured qpos without running CPU dynamics, constraints, or contacts.
        mujoco.mj_kinematics(model, scratch)

        def _world_box(geom_id: int, *, name: str) -> OrientedBox:
            return OrientedBox.validated(
                center=scratch.geom_xpos[int(geom_id)],
                axes=np.asarray(
                    scratch.geom_xmat[int(geom_id)], dtype=float
                ).reshape(3, 3),
                half_extents=model.geom_size[int(geom_id)],
                name=name,
            )

        left_pad = _world_box(pad_ids["left_pad"], name="left pad")
        right_pad = _world_box(pad_ids["right_pad"], name="right pad")
        gripper_axes = np.asarray(
            scratch.site_xmat[int(plan_world.site_id)], dtype=float
        ).reshape(3, 3)
        subjects = [
            _world_box(geom_id, name=f"subject collision geom {geom_id}")
            for geom_id in subject_geom_ids
        ]
        results = [
            closing_region_intersection(
                subject=subject_box,
                left_pad=left_pad,
                right_pad=right_pad,
                gripper_axes=gripper_axes,
            )
            for subject_box in subjects
        ]
        intersects = any(result.intersects for result in results)
        region = results[0].region
        subject_geometry = []
        for geom_id, subject_box, result in zip(
            subject_geom_ids,
            subjects,
            results,
            strict=True,
        ):
            subject_geometry.append({
                "geom_id": int(geom_id),
                "geom_name": mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    int(geom_id),
                ),
                "center": subject_box.center.tolist(),
                "axes": subject_box.axes.tolist(),
                "half_extents_m": subject_box.half_extents.tolist(),
                "intersects_closing_region": bool(result.intersects),
            })
    except Exception as exc:  # noqa: BLE001 - terminal evidence fails closed
        return _fail("spatial_geometry_evaluation_failed", repr(exc))
    evidence.update({
        "observable": True,
        "intersects": bool(intersects),
        "intersects_closing_region": bool(intersects),
        "reason_code": (
            "tracked_subject_intersects_physical_closing_region"
            if intersects
            else "tracked_subject_outside_physical_closing_region"
        ),
        "reason": (
            "tracked_subject_intersects_physical_closing_region"
            if intersects
            else "tracked_subject_outside_physical_closing_region"
        ),
        "closing_region_center": region.center.tolist(),
        "closing_region_axes": region.axes.tolist(),
        "closing_region_half_extents_m": region.half_extents.tolist(),
        "subject_body_id": subject_body_id,
        "subject_collision_geometry": subject_geometry,
        "pad_geometry_names": pad_names,
        "current_pose_applied_before_fk": True,
        "fk_refresh": "mujoco.mj_kinematics_scratch",
    })
    return evidence


def _object_held_terminal_state(
    *,
    commanded_closed: bool,
    obstruction: bool | None,
    track_valid: bool,
    intersects_closing_region: bool | None,
    measured_width_m: float | None,
    commanded_width_m: float | None,
    width_resolution_m: float | None,
) -> bool | None:
    """Fuse the three necessary-and-sufficient real ``ObjectHeld`` facts."""
    if commanded_closed:
        if obstruction is False:
            return False
        if obstruction is not True or not track_valid:
            return None
        if isinstance(intersects_closing_region, (bool, np.bool_)):
            return bool(intersects_closing_region)
        return None
    if (
        measured_width_m is None
        or commanded_width_m is None
        or width_resolution_m is None
        or not track_valid
    ):
        return None
    try:
        reached_open_command = bool(
            abs(float(measured_width_m) - float(commanded_width_m))
            <= float(width_resolution_m)
        )
    except (TypeError, ValueError):
        return None
    # Failure to reach an open target proves only unresolved release, never a
    # held subject. Reaching it is affirmative evidence that the subject is
    # not held.
    return False if reached_open_command else None


def _require_mask_scored_tracker_ready(
    scene: dict[str, Any],
    *,
    required_body_names: set[str],
    max_age_s: float,
    min_mask_iou: float = 0.0,
    now_s: float | None = None,
) -> dict[str, Any]:
    """Fail closed unless one fresh synchronized packet scores every body."""
    threshold = float(min_mask_iou)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("min_mask_iou must be finite in [0, 1]")
    now = float(time.time() if now_s is None else now_s)
    try:
        stamp = float(scene.get("capture_timestamp_s"))
    except (TypeError, ValueError) as exc:
        raise safety.SafetyError(
            "tracker-ready packet has no numeric capture timestamp"
        ) from exc
    age = now - stamp
    if (
        not np.isfinite(stamp)
        or not np.isfinite(age)
        or age < 0.0
        or age > float(max_age_s)
    ):
        raise safety.SafetyError(
            "tracker-ready packet is missing, future-dated, or stale"
        )
    if scene.get("synchronized_capture") is not True:
        raise safety.SafetyError(
            "tracker-ready packet is not one synchronized exposure"
        )
    capture_id = scene.get("capture_id")
    if (
        not isinstance(capture_id, str)
        or not capture_id
        or any(ch.isspace() for ch in capture_id)
    ):
        raise safety.SafetyError(
            "tracker-ready packet has no exact capture_id"
        )
    bodies = scene.get("bodies")
    if not isinstance(bodies, dict):
        raise safety.SafetyError("tracker-ready packet has no bodies")
    missing = sorted(required_body_names - set(bodies))
    if missing:
        raise safety.SafetyError(
            f"tracker-ready packet is missing bodies: {missing}"
        )
    scores: dict[str, float] = {}
    for name in sorted(required_body_names):
        body = bodies[name]
        lost = body.get("track_lost")
        score = body.get("mask_iou")
        confidence = body.get("confidence")
        if (
            not isinstance(lost, bool)
            or lost
            or body.get("mask_scored") is not True
            or score is None
            or confidence is None
        ):
            raise safety.SafetyError(
                f"tracker-ready body {name!r} is not mask-scored/reliable"
            )
        try:
            score_f = float(score)
            confidence_f = float(confidence)
        except (TypeError, ValueError) as exc:
            raise safety.SafetyError(
                f"tracker-ready body {name!r} has non-numeric quality"
            ) from exc
        if (
            not np.isfinite(score_f)
            or not 0.0 <= score_f <= 1.0
            or not np.isfinite(confidence_f)
            or not 0.0 <= confidence_f <= 1.0
        ):
            raise safety.SafetyError(
                f"tracker-ready body {name!r} has invalid quality"
            )
        if score_f < threshold:
            raise safety.SafetyError(
                f"tracker-ready body {name!r} mask IoU {score_f:.3f} is "
                f"below the required {threshold:.3f}"
            )
        scores[name] = score_f
        if body.get("capture_id") != capture_id:
            raise safety.SafetyError(
                f"tracker-ready body {name!r} has a different capture_id"
            )
    return {
        "capture_timestamp_s": stamp,
        "capture_id": capture_id,
        "age_s": age,
        "required_body_names": sorted(required_body_names),
        "min_mask_iou": threshold,
        "mask_iou": scores,
    }


def _connect_after_tracker_ready(
    scene: dict[str, Any],
    *,
    required_body_names: set[str],
    max_age_s: float,
    connect: Callable[[], Any],
    min_mask_iou: float = 0.0,
    now_s: float | None = None,
) -> tuple[dict[str, Any], Any]:
    """Make the readiness attestation the predecessor of controller access."""
    evidence = _require_mask_scored_tracker_ready(
        scene,
        required_body_names=required_body_names,
        max_age_s=max_age_s,
        min_mask_iou=min_mask_iou,
        now_s=now_s,
    )
    return evidence, connect()


def _capture_live_synthesis_context(
    cfg: LoopConfig,
    objects: list[SceneObject],
    out_dir: Path,
) -> tuple[Any, Path, dict[str, Any]]:
    """Observe, ground, and annotate one current scene for live synthesis.

    Program synthesis happens before task-role binding, so it cannot select one
    object as the tracking subject.  Use one synchronized, segmented RGB-D
    capture for every manifest object and the task-independent gravity OBB
    observer for each body in that same packet.  The returned anchors and image
    therefore describe the same physical instant; manifest poses are never
    substituted as live geometry.

    FoundationPose remains the closed-loop pose backend after the program binds
    the controlled object.  Forcing the pre-bind observation to gravity OBB is
    intentional: it needs no guessed subject or subject-specific persistent
    worker, and every object is localized from the shared frame.
    """
    if cfg.offline:
        raise ValueError("live synthesis context requires a live observer")
    if cfg.observer_python is None:
        raise RuntimeError(
            "live VLM synthesis requires --observer-python so its anchors and "
            "annotated image come from one current scene observation"
        )
    if not objects:
        raise RuntimeError("live VLM synthesis requires at least one scene object")

    observation_cfg = replace(
        cfg,
        foundationpose_mode="off",
        subject_pose_policy="gravity_obb",
    )
    observation = observe_mod.observe_scene(
        observation_cfg,
        objects,
        None,
        "synthesis_scene",
        purpose="replan",
        plan_world=None,
        fp_client=None,
        z_real_to_plan=0.0,
    )
    table_z = (
        float(cfg.real_table_z)
        if cfg.real_table_z is not None
        else float(TABLE_TOP_Z)
    )
    anchors = observe_mod.ground_observation(
        observation,
        None,
        objects,
        table_top_z=table_z,
    )
    missing = [
        obj.name
        for obj in objects
        if f"{obj.name}_center" not in anchors
    ]
    if missing:
        raise RuntimeError(
            "current synthesis observation did not localize every scene object: "
            + ", ".join(sorted(missing))
        )

    capture_dir_raw = observation.get("capture_dir")
    if not capture_dir_raw:
        raise RuntimeError(
            "current synthesis observation has no synchronized capture directory"
        )
    capture_dir = Path(str(capture_dir_raw))
    packet: Path | None = None
    for obj in objects:
        candidate = capture_dir / obj.name
        if (
            (candidate / "rgb.png").is_file()
            and (candidate / "cam_K.txt").is_file()
        ):
            packet = candidate
            break
    if packet is None:
        raise RuntimeError(
            "synchronized synthesis observation has no RGB/intrinsics evidence "
            f"under {capture_dir}"
        )

    from oap.program.annotate import render_anchor_annotations

    trace_dir = Path(out_dir) / "synthesis"
    scene = render_anchor_annotations(
        packet / "rgb.png",
        anchors,
        np.loadtxt(packet / "cam_K.txt"),
        out_dir=trace_dir,
    )
    write_json_atomic(trace_dir / "observation.json", observation)
    logger.info(
        "[synthesis] current synchronized scene: %s "
        "(%d anchors in frame, %d off-screen)",
        scene.annotated_path,
        len(scene.visible),
        len(scene.off_screen),
    )
    return anchors, scene.annotated_path, observation


def _record_live_synthesis_trace(
    *,
    out_dir: Path,
    cfg: LoopConfig,
    program: TaskProgram,
    program_sha256: str,
    episode: EpisodeLog,
) -> None:
    """Link the exact VLM exchange and current observation to the program hash."""
    trace_dir = Path(out_dir) / "synthesis"
    raw_responses = sorted(
        path.name for path in trace_dir.glob("raw_response_*.txt")
    )
    provider, separator, model = str(program.provenance).partition(":")
    index = {
        "schema": "oap_live_synthesis_trace_v1",
        "backend": str(cfg.synth_backend),
        "provider": provider,
        "model": model if separator else None,
        "program_provenance": str(program.provenance),
        "program_sha256": str(program_sha256),
        "observation": "observation.json",
        "prompt_system": "prompt_system.txt",
        "prompt_user": "prompt_user.txt",
        "prompt_image": "prompt_image.png",
        "raw_responses": raw_responses,
        "program": "../program.json",
    }
    write_json_atomic(trace_dir / "trace.json", index)
    episode.record_event("live_program_synthesis", index)


# ==========================================================================
# The episode
# ==========================================================================
def run_episode(cfg: LoopConfig) -> dict[str, Any]:
    """Run ONE closed-loop episode; returns the summary dict (also on disk).

    See the module docstring for the state machine. Everything the paper needs
    to audit the run is in ``<out>/episode.json`` (schema
    ``oap_episode_v3``); ``summary.json`` is the compact human recap.
    """
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    # Production has one algorithm: one device-resident nominal-plus-Gaussian
    # predictive batch. Generic controller parameters were resolved once when
    # LoopConfig was constructed.
    if int(cfg.max_stage_cycles) < 1:
        raise ValueError("max_stage_cycles must be >= 1")
    cfg.max_stage_cycles = int(cfg.max_stage_cycles)
    cfg.mppi_cycle_budget_scope = str(
        cfg.mppi_cycle_budget_scope
    ).strip().lower()
    if cfg.mppi_cycle_budget_scope not in {"episode", "stage"}:
        raise ValueError(
            "mppi_cycle_budget_scope must be 'episode' or 'stage'"
        )
    cfg.horizon_steps = validate_horizon_steps(cfg.horizon_steps)
    if cfg.mppi_stage_execution_prefix_steps is not None:
        cfg.mppi_stage_execution_prefix_steps = (
            validate_mppi_stage_execution_prefix_steps(
                cfg.mppi_stage_execution_prefix_steps
            )
        )
        for stage_steps in cfg.mppi_stage_execution_prefix_steps:
            resolve_execution_prefix(
                horizon_steps=cfg.horizon_steps,
                execution_prefix_fraction=cfg.execution_prefix_fraction,
                execution_prefix_steps=stage_steps,
            )
        if str(cfg.optimizer).strip().lower() != "mppi":
            raise safety.SafetyError(
                "stage execution-prefix steps are supported only by MPPI"
            )
        if (
            not cfg.offline
            or cfg.execute
            or cfg.prepare_real_execution
            or cfg.remote_planner_url is not None
        ):
            raise safety.SafetyError(
                "stage execution-prefix steps are offline local-MPPI-only"
            )
    if (
        (cfg.execute or cfg.prepare_real_execution)
        and cfg.execution_prefix_steps is None
    ):
        raise safety.SafetyError(
            "real execution and readiness require explicit "
            "execution_prefix_steps; a fraction-only interval is not "
            "authorizable"
        )
    cfg.pool_size = validate_pool_size(cfg.pool_size)
    if str(cfg.optimizer).strip().lower() == "mppi":
        from oap.loop.planner_profile import (
            require_mppi_runtime_profile,
        )

        try:
            require_mppi_runtime_profile(cfg)
        except ValueError as exc:
            raise safety.SafetyError(str(exc)) from exc
    remote_values = (
        cfg.remote_planner_url,
        cfg.remote_planner_model_sha256,
        cfg.remote_planner_assets_sha256,
        cfg.remote_planner_sha256,
        cfg.remote_planner_service_instance_id,
        cfg.remote_planner_deadline_s,
    )
    if any(value is not None for value in remote_values):
        if not all(value is not None for value in remote_values):
            raise safety.SafetyError(
                "remote planning requires URL, model/assets/planner SHA, "
                "service instance id, and deadline together"
            )
        if not cfg.execute:
            raise safety.SafetyError(
                "remote planning is only supported on the lab-executor path; "
                "dry-run needs the local GPU prefix state"
            )
        if str(cfg.optimizer).strip().lower() != "cem":
            raise safety.SafetyError(
                "remote planner protocol v3 does not carry optimizer or "
                "MPPI temperature; refusing to silently execute an MPPI "
                "request with the remote service's CEM semantics"
            )
        assert cfg.remote_planner_deadline_s is not None
        if float(cfg.remote_planner_deadline_s) <= 0.0:
            raise safety.SafetyError(
                "remote planner deadline must be positive"
            )
        if (
            cfg.max_plan_staleness_s is not None
            and float(cfg.remote_planner_deadline_s)
            > float(cfg.max_plan_staleness_s)
        ):
            raise safety.SafetyError(
                "remote planner deadline may not exceed the measured "
                "plan-staleness limit"
            )
    evidence_calibration: ObjectHeldEvidenceCalibration | None = None
    if cfg.hardware_evidence_calibration_json is not None:
        try:
            evidence_calibration = load_object_held_evidence_calibration(
                cfg.hardware_evidence_calibration_json
            )
        except ValueError as exc:
            raise safety.SafetyError(str(exc)) from exc
        cfg.hardware_evidence_calibration_json = evidence_calibration.path
        cfg.hardware_evidence_calibration_sha256 = evidence_calibration.sha256
        cfg.hardware_evidence_calibration_source = evidence_calibration.source
        cfg.hardware_evidence_calibration_hardware_id = (
            evidence_calibration.hardware_id
        )
        for name, value in evidence_calibration.loop_config_values().items():
            setattr(cfg, name, value)
    cfg.joint_executor_hardware_verified = False
    cfg.joint_executor_verification_latch_verified = False
    cfg.joint_executor_software_fingerprint_sha256 = None
    if cfg.execute or cfg.prepare_real_execution:
        executor_readiness = evaluate_executor_readiness(
            cfg.joint_executor_verification_json,
            cfg.joint_executor_verification_sha256,
        )
        write_json_atomic(out / "executor_readiness.json", executor_readiness)
        cfg.joint_executor_hardware_verified = bool(
            executor_readiness["hardware_verified"]
        )
        cfg.joint_executor_verification_latch_verified = bool(
            executor_readiness["verification_latch_valid"]
        )
        cfg.joint_executor_software_fingerprint_sha256 = (
            executor_readiness["flexiv_control"][
                "software_fingerprint_sha256"
            ]
        )
        if (
            cfg.prepare_real_execution
            and not executor_readiness["software_ready"]
        ):
            raise safety.SafetyError(
                "joint executor software preflight failed; no robot connection "
                f"was attempted. See {out / 'executor_readiness.json'}"
            )
    safety.require_joint_executor_readiness(
        execute=cfg.execute,
        prepare_real_execution=cfg.prepare_real_execution,
        hardware_verified=cfg.joint_executor_hardware_verified,
        verification_latch_verified=(
            cfg.joint_executor_verification_latch_verified
        ),
    )
    observation_readiness = safety.require_observation_readiness_limits(
        required=bool(cfg.execute or cfg.prepare_real_execution),
        max_observation_age_s=cfg.max_observation_age_s,
        max_camera_robot_skew_s=cfg.max_camera_robot_skew_s,
        max_plan_staleness_s=cfg.max_plan_staleness_s,
        max_final_joint_tracking_error_rad=(
            cfg.max_final_joint_tracking_error_rad
        ),
        min_terminal_anchor_reliability=(
            cfg.min_terminal_anchor_reliability
        ),
    )
    safety.require_execution_allowed(execute=cfg.execute,
                                     i_confirm_real_motion=cfg.i_confirm_real_motion)
    episode = EpisodeLog(out, instruction=cfg.task, bundle_manifest=cfg.bundle_manifest,
                         config_snapshot=cfg.snapshot())

    # ---------------------------------------------------------------- scene
    objects = load_scene_manifest(
        Path(cfg.bundle_manifest),
        z_real_to_plan=(None if cfg.real_table_z is None
                        else TABLE_TOP_Z - float(cfg.real_table_z)),
    )
    _apply_offline_dataset_subject_override(cfg, objects)
    if cfg.collision_from_mesh:
        _apply_collision_from_mesh(objects)
    # A reviewed/offline program still validates against the complete manifest
    # vocabulary before its roles bind. A LIVE synthesized program replaces
    # these values below with one current synchronized whole-scene observation.
    synthesis_anchors = observe_mod.anchors_from_scene_manifest(
        objects, table_top_z=TABLE_TOP_Z)
    synthesis_image: Path | None = None
    live_synthesis_observation: dict[str, Any] | None = None
    if cfg.program_json is None and not cfg.offline:
        (
            synthesis_anchors,
            synthesis_image,
            live_synthesis_observation,
        ) = _capture_live_synthesis_context(cfg, objects, out)
    if cfg.program_json is not None:
        program = TaskProgram.from_json(
            Path(cfg.program_json).read_text(encoding="utf-8"))
        logger.info("[program] loaded from %s", cfg.program_json)
    else:
        program = synthesize(
            cfg.task, synthesis_anchors, backend=cfg.synth_backend,
            trace_dir=out / "synthesis", image_path=synthesis_image)
    if (
        cfg.mppi_stage_execution_prefix_steps is not None
        and len(cfg.mppi_stage_execution_prefix_steps) != len(program.stages)
    ):
        raise safety.SafetyError(
            "mppi_stage_execution_prefix_steps must provide exactly one "
            f"entry per program stage: got "
            f"{len(cfg.mppi_stage_execution_prefix_steps)} for "
            f"{len(program.stages)} stages"
        )
    if cfg.pick_v8_causal_profile is not None:
        require_pick_v8_causal_contract(cfg, program=program)
    elif cfg.unified_mppi_effort_profile is not None:
        require_unified_mppi_effort_contract(cfg, program=program)
    else:
        require_s0_control_profile_contract(cfg, program=program)
    from oap.program.synthesis import validate_program

    # The manifest vocabulary predates runtime signed material-frame anchors.
    # Defer only that capability check; the grounded pass below remains strict.
    program_errors = validate_program(
        program,
        synthesis_anchors,
        require_grounded_frame_contract=False,
    )
    if program_errors:
        reason = "invalid_program:" + ";".join(program_errors[:4])
        logger.error("[program] CANNOT_VERIFY (%s)", reason)
        write_json_atomic(out / "verdict.json", {
            "outcome": "CANNOT_VERIFY",
            "provenance": "program_validation",
            "reason": reason,
        })
        episode.record_verdict({
            "outcome": "CANNOT_VERIFY",
            "tier": "refused",
            "provenance": "program_validation",
            "reason": reason,
        })
        episode.finalize(success=False, outcome="CANNOT_VERIFY")
        return {
            "success": False,
            "outcome": "CANNOT_VERIFY",
            "reason": reason,
            "chunks": 0,
        }
    safety.refuse_offline_program_execution(program, execute=cfg.execute)
    program_sha = episode.set_program(program)
    if live_synthesis_observation is not None:
        _record_live_synthesis_trace(
            out_dir=out,
            cfg=cfg,
            program=program,
            program_sha256=program_sha,
            episode=episode,
        )
    safety.require_real_tool_mediation_evidence(
        program,
        execute=cfg.execute,
        prepare_real_execution=cfg.prepare_real_execution,
    )
    if cfg.execute:
        approval_gate = getattr(safety, "require_program_approval", None)
        if approval_gate is None:
            raise safety.SafetyError(
                "canonical live-VLM program approval gate is unavailable in "
                "this build; refusing real motion"
            )
        approval_gate(
            program,
            execute=True,
            approved_program_sha256=cfg.approved_program_sha256,
        )
    try:
        require_real_object_held_evidence(
            program=program,
            execute=cfg.execute,
            prepare_real_execution=cfg.prepare_real_execution,
            calibration=evidence_calibration,
            foundationpose_mode=cfg.foundationpose_mode,
            disable_held_verify=cfg.disable_held_verify,
        )
    except ValueError as exc:
        raise safety.SafetyError(str(exc)) from exc
    if evidence_calibration is not None:
        episode.record_event(
            "object_held_evidence_calibration",
            {
                "schema": "oap_object_held_evidence_calibration_ref_v1",
                "path": str(evidence_calibration.path),
                "sha256": evidence_calibration.sha256,
                "source": evidence_calibration.source,
                "captured_at": evidence_calibration.captured_at,
                "hardware_id": evidence_calibration.hardware_id,
                **evidence_calibration.loop_config_values(),
            },
        )
    logger.info("[program] (%s, sha %s): %d stages, fully_geometric=%s non_geometric=%s",
                program.provenance, program_sha[:12], len(program.stages),
                program.is_fully_geometric(), program.non_geometric_reasons())
    try:
        bindings = bind_program_to_scene(program, objects)
    except ProgramBindingError as exc:
        reason = f"ungroundable:program_binding:{exc}"
        logger.error("[program] CANNOT_VERIFY (%s) -- refusing to move", reason)
        write_json_atomic(out / "verdict.json", {
            "outcome": "CANNOT_VERIFY", "provenance": "program_binding",
            "reason": reason, "program_provenance": program.provenance})
        episode.record_verdict({"outcome": "CANNOT_VERIFY", "tier": "refused",
                                "provenance": "program_binding", "reason": reason})
        episode.finalize(success=False, outcome="CANNOT_VERIFY")
        return {"success": False, "outcome": "CANNOT_VERIFY", "reason": reason,
                "chunks": 0}
    subject, reference = bindings.subject, bindings.reference
    episode.record_event("task_roles", {
        "source": "constraint_program", "subject": subject.name,
        "reference": reference.name if reference else None})

    clients = _build_worker_clients(cfg, out)
    fp_client = clients["fp"]

    # ------------------------------------------------------- observe (init)
    obs0 = observe_mod.observe_subject(cfg, subject, "chunk_00_before", plan_world=None,
                                       fp_client=fp_client)
    episode.record_observation(0, "before", obs0)
    if cfg.control_profile is not None:
        if cfg.pick_v8_causal_profile is not None:
            input_identity = require_pick_v8_causal_input_identity(
                profile=cfg.pick_v8_causal_profile,
                control_profile=cfg.control_profile,
                bundle_manifest=cfg.bundle_manifest,
                initial_observation=obs0,
                subject_name=subject.name,
                reference_name=(
                    reference.name if reference is not None else None
                ),
            )
        elif cfg.unified_mppi_effort_task_id is not None:
            input_identity = require_unified_mppi_effort_input_identity(
                profile=cfg.unified_mppi_effort_profile,
                task_id=cfg.unified_mppi_effort_task_id,
                control_profile=cfg.control_profile,
                bundle_manifest=cfg.bundle_manifest,
                initial_observation=obs0,
                subject_name=subject.name,
                reference_name=(
                    reference.name if reference is not None else None
                ),
            )
        else:
            input_identity = require_s0_control_profile_input_identity(
                control_profile=cfg.control_profile,
                joint_velocity_force_task_id=cfg.joint_velocity_force_task_id,
                bundle_manifest=cfg.bundle_manifest,
                initial_observation=obs0,
                subject_name=subject.name,
                reference_name=(
                    reference.name if reference is not None else None
                ),
            )
        episode.record_event(
            "control_profile_input_identity",
            input_identity,
        )
    logger.info("[obs0] %s", obs0)
    # Under the FP policy obs0 carries no size_lwh_obb, so the standing-dims
    # correction below never fired: an Any6D thin-axis under-read then becomes
    # the twin's HEIGHT (a seated 11 cm bear twinned as a lying 4 cm slab) and
    # every grasp targets the slab's mid-height -- the legs, a marginal fabric
    # pinch that slipped mid-carry on camera. Fit the gravity OBB on the SAME
    # capture packet (free: depth already on disk) purely for the standing
    # dims; the FP pose stays authoritative for position/yaw.
    if (obs0.get("size_lwh_obb") is None and not cfg.offline
            and cfg.foundationpose_mode == "actual"):
        try:
            _tz = float(cfg.real_table_z) if cfg.real_table_z is not None else TABLE_TOP_Z
            _, _, obb_probe_dims = observe_mod.subject_obb_from_capture(
                Path(cfg.out_dir) / "fp_obs" / "chunk_00_before", subject, _tz)
            obs0["size_lwh_obb"] = [float(v) for v in obb_probe_dims]
            logger.info("[obb-probe] standing dims from obs0 depth: %s",
                        [round(float(v), 4) for v in obb_probe_dims])
        except Exception as exc:  # noqa: BLE001 -- size probe is best-effort
            logger.warning("[obb-probe] standing-dims probe unavailable (%r); "
                           "keeping reconstruction dims", exc)
    # gravity_obb recovers the STANDING dims from depth -> adopt them as the
    # subject's twin size (which side is vertical) and roll q_om so the mesh
    # renders/collides on the right face. Initial on-table observation only.
    if obs0.get("size_lwh_obb") is not None:
        obb_l, obb_w, obb_h = (float(v) for v in obs0["size_lwh_obb"])
        obb_dims = (obb_l, obb_w, obb_h)
        if tuple(round(x, 5) for x in obb_dims) != tuple(round(x, 5) for x in subject.size_lwh):
            logger.info("[obb] subject.size_lwh %s -> %s (standing orientation from depth)",
                        tuple(round(x, 4) for x in subject.size_lwh),
                        tuple(round(x, 4) for x in obb_dims))
            # load_scene_manifest derives q_om for every object at load time.
            assert subject.q_om is not None
            subject.q_om = roll_q_om_for_standing_dims(subject.q_om, subject.size_lwh, obb_dims)
        subject.size_lwh = obb_dims
    real_table_z = (float(cfg.real_table_z) if cfg.real_table_z is not None
                    else float(obs0.get("table_z_base", TABLE_TOP_Z)))
    z_real_to_plan = TABLE_TOP_Z - real_table_z
    _project_objects_to_plan_support(objects, z_real_to_plan=z_real_to_plan)
    logger.info("[frames] real table z=%.4f  plan offset=%+.4f", real_table_z, z_real_to_plan)
    obs0 = observe_mod.snap_observation_to_table(
        obs0, subject.size_lwh, cfg.real_table_z, "obs0", enabled=cfg.snap_subject_to_table,
        containers=observe_mod.open_top_container_rests(objects, subject, cfg.real_table_z))
    # Localize the reference LIVE at scene init too (no hardcoded initial pose)
    # when per-chunk relocalization is on; plan_world=None -> pose_base only.
    if reference is not None and cfg.relocalize_all_each_chunk and not cfg.offline:
        rec_xy = np.round(np.asarray(reference.pose_base, dtype=float)[:2], 3).tolist()
        try:
            dmm, ddeg = observe_mod.relocalize_static_object(
                cfg, reference, None, z_real_to_plan, "reference_init_localize",
                fp_client=fp_client)
            logger.info("[init-relocalize] %s recorded %s -> live %s (drift %.0fmm/%.1fdeg)",
                        reference.name, rec_xy,
                        np.round(np.asarray(reference.pose_base)[:2], 3).tolist(), dmm, ddeg)
        except Exception as exc:  # noqa: BLE001 - init relocalize is best-effort
            logger.warning("[init-relocalize] %s SKIPPED (%r); keeping recorded pose %s",
                           reference.name, exc, rec_xy)

    def to_plan_pose(obs: dict[str, Any]) -> tuple[float, float, float, float]:
        return (obs["pos_base"][0], obs["pos_base"][1],
                obs["pos_base"][2] + (0.0 if str(obs.get("schema", "")).startswith("offline")
                                      else z_real_to_plan),
                obs["yaw_base_rad"])

    subject_pose0 = to_plan_pose(obs0)
    # Seat the subject on its support, the way every OTHER movable body already
    # is.  manifest.py projects each non-subject body; the subject's pose comes
    # straight from the observation and skipped that path, so every task spawned
    # its subject afloat and it free-fell on the first physics step -- the Push
    # box 6.850 mm, the Pick and ToolPush eraser 4.780 mm.
    #
    # The float is not cosmetic: ``*_initial_center`` anchors are frozen from
    # this t=0 pose and never re-measured, so the drop becomes a permanent z
    # component of every displacement residual taken against them.  For Direct
    # Push it consumed 46.9% of the 10 mm tolerance ball's area, leaving
    # 7.282 mm in-plane against a declared 10.000 -- a gate 27% tighter than the
    # program states, which decided one run's pass (0.266 mm margin, 1 qualifying
    # cycle of 435) and the next run's failure (short by 0.572 mm).
    #
    # It MUST be seated here rather than inside build_plan_scene.  Seating there
    # was tried and measured: the emitted XML and its keyframe were both correct
    # at 0.06785, and the run still started at 0.074700, because ``subject_pose0``
    # is local to this function and line ~1998 resets the loaded world from it,
    # as does the obj_pose7 built further down.  One seating point, before every
    # consumer, is the only arrangement that holds.
    #
    # The half extent uses the quaternion the scene builder will actually EMIT
    # (the same expression reset_object_pose is given below), not the
    # reconstruction quat: for the Push box the latter carries a 1.267 deg tilt
    # and returns 0.0499579 against the emitted frame's true 0.047850, leaving
    # 2.108 mm of float and fixing nothing.
    _seated_z, _seated = seat_unheld_pose_on_support(
        subject_pose0[:3],
        object_body_quat(subject_pose0[3], subject.q_om),
        subject.size_lwh,
        subject.q_om,
        TABLE_TOP_Z,
        held=False,
        support_center_xy=LAB_TABLE_POS[:2],
        support_half_size_xy=LAB_TABLE_SIZE[:2],
        excluded_footprints=[
            (
                container.name,
                tuple(container.pose_base[:2]),
                0.5 * float(max(container.size_lwh)),
            )
            for container in objects
            if (
                container is not subject
                and container.open_top_container
                and container.pose_base is not None
            )
        ],
    )
    if _seated:
        logger.info(
            "[table-support] subject %s: z %.6f -> %.6f (exact seat on the "
            "emitted body frame; was %+.3f mm off the table)",
            subject.name, float(subject_pose0[2]), float(_seated_z),
            1000.0 * (float(subject_pose0[2]) - float(_seated_z)),
        )
        subject_pose0 = (subject_pose0[0], subject_pose0[1],
                         float(_seated_z), subject_pose0[3])
    sub_H = float(subject.size_lwh[2])

    # ------------------------------------------------------------- twin
    # Build and reset the CPU model before offline grounded validation.  The
    # configured reconstruction quaternion is an input to scene construction,
    # but the released model actually starts from yaw composed with q_om (and
    # support projection).  RelativeOrientation must bind to that frozen qpos,
    # which is also the state the first GPU rollout receives.
    assert subject.q_om is not None
    plan_xml = build_plan_scene(
        objects, subject, subject_pose0, z_real_to_plan, out,
        table_spec_json=cfg.table_spec_json, table_spec_height=cfg.table_spec_height,
        home_posture_json=cfg.home_posture_json)
    if cfg.execute or cfg.prepare_real_execution:
        if (cfg.sim_tcp_m is None
                or abs(float(cfg.sim_tcp_m) - REAL_GN01_TCP_M) > 5e-4):
            raise safety.SafetyError(
                "real-execution preflight requires an explicit "
                f"--sim-tcp-m {REAL_GN01_TCP_M:.5f}; the live closed-tip/grasp "
                "frame and compiled planning site must agree within 0.5 mm")
    if cfg.sim_tcp_m is not None:
        patch_sim_gripper_tcp(plan_xml, float(cfg.sim_tcp_m), anchor=cfg.sim_tcp_anchor)
    plan_world = load_world(plan_xml, "W_plan_real")
    reset_object_pose(plan_world, list(subject_pose0[:3]),
                      object_body_quat(subject_pose0[3], subject.q_om))
    executed_action_state: ExecutedActionState | None = None
    if cfg.control_profile is not None:
        stage_entry_control_evidence = (
            _initialize_control_profile_jaw_carrier(
                plan_world,
                control_profile=cfg.control_profile,
                pick_v8_causal_profile=cfg.pick_v8_causal_profile,
                unified_mppi_effort_profile=cfg.unified_mppi_effort_profile,
            )
        )
        episode.record_event(
            "control_profile_stage_entry",
            stage_entry_control_evidence,
        )
        if cfg.control_regularizer_calibration_profile is not None:
            if stage_entry_control_evidence.get(
                "carrier_command_units"
            ) != "effort_n":
                raise RuntimeError(
                    "control regularizer cycle0 boundary requires an "
                    "acknowledged physical jaw effort"
                )
            executed_action_state = (
                ExecutedActionState.cycle0_profile_native_hold(
                    jaw_effort_n=float(
                        stage_entry_control_evidence["carrier_command"]
                    ),
                    source="registered_control_profile_stage_entry_ack",
                )
            )
            episode.record_event(
                "control_regularizer_cycle0_boundary",
                executed_action_state.to_dict(),
            )

    # Establish the episode-zero scene used by validation, scoring and measured
    # Stage evaluation.  Offline simulation reads the exact post-reset qpos;
    # live execution retains the synchronized observer-owned scene.
    if cfg.offline:
        scene_obs0, anchors0 = observe_mod.ground_offline_plan_world(
            cfg,
            objects,
            subject,
            plan_world,
            tag="scene_00_before",
            table_top_z=TABLE_TOP_Z,
            reference=reference,
            z_real_to_plan=z_real_to_plan,
        )
    else:
        scene_obs0 = observe_mod.observe_scene(
            cfg, objects, subject, "scene_00_before", purpose="replan",
            plan_world=None, fp_client=fp_client,
            z_real_to_plan=z_real_to_plan, observed_subject=obs0)
        anchors0 = observe_mod.ground_observation(
            scene_obs0, subject, objects, table_top_z=TABLE_TOP_Z,
            reference=reference)
    episode.record_event("scene_observation_initial", scene_obs0)

    # Grounded re-validation: the contact structural checks (actor/dynamic
    # target/shared target) are decidable only on this role-bound vocabulary,
    # never on the pre-bind manifest/synthesis anchors validated above. Refuse
    # here -- after CPU model reset but before any GPU compile/sampling -- with
    # the same validator preflight re-runs after its grounding.
    grounded_errors = validate_program(program, anchors0)
    if grounded_errors:
        reason = "invalid_program_grounded:" + ";".join(grounded_errors[:4])
        logger.error("[program] CANNOT_VERIFY (%s)", reason)
        write_json_atomic(out / "verdict.json", {
            "outcome": "CANNOT_VERIFY",
            "provenance": "program_validation_grounded",
            "reason": reason,
        })
        episode.record_verdict({
            "outcome": "CANNOT_VERIFY",
            "tier": "refused",
            "provenance": "program_validation_grounded",
            "reason": reason,
        })
        episode.finalize(success=False, outcome="CANNOT_VERIFY")
        return {"success": False, "outcome": "CANNOT_VERIFY",
                "reason": reason, "chunks": 0}

    # A declared shaping profile that moves no multiplier on THIS program is a
    # silent no-op: it is registered, passes every identity assertion, and is
    # reported in the episode while changing nothing.  That is not theoretical
    # -- cup_simple_tilt_terminal_held_60_v1 seals the stage identity
    # tilt_held_cup_positive_x_60deg, Cup v20 renamed its tilt stage to
    # ..._30deg, and the advertised terminal ObjectHeld boost was inert on five
    # profiles while their identity kept reporting it.  Refuse here, on the
    # grounded program, before any GPU compile.
    if not cost_shaping_profile_binds(program, cfg.cost_shaping_profile):
        reason = (
            "cost_shaping_no_op:"
            f"{cfg.cost_shaping_profile!r} moves no multiplier on any stage of "
            "this program"
        )
        logger.error("[program] CANNOT_VERIFY (%s)", reason)
        write_json_atomic(out / "verdict.json", {
            "outcome": "CANNOT_VERIFY",
            "provenance": "cost_shaping_binding",
            "reason": reason,
        })
        episode.record_verdict({
            "outcome": "CANNOT_VERIFY",
            "tier": "refused",
            "provenance": "cost_shaping_binding",
            "reason": reason,
        })
        episode.finalize(success=False, outcome="CANNOT_VERIFY")
        return {"success": False, "outcome": "CANNOT_VERIFY",
                "reason": reason, "chunks": 0}

    # The planning world's support datum is fixed. Container faces remain
    # ordinary observed anchors used explicitly by running/terminal predicates;
    # a task predicate must never mutate the physics/reference frame.
    band_floor = TABLE_TOP_Z

    # Up-front refusal (force-free safe): if the program references anchors
    # absent/non-finite in the grounded scene, OR any Stage terminal is empty
    # or contains an unmeasurable primitive, that sub-task can NEVER be checked
    # in full -- refuse BEFORE any motion, with the same contract used by the
    # measured Stage evaluator.
    # Robot-state anchors such as ``gripper_center`` and
    # ``gripper_closing_axis`` are not perception hallucinations. They are
    # grounded from FK immediately after the planning model is built above,
    # and from measured/FK state at every later verdict. All scene anchors
    # still have to be finite now; ``groundability_failures`` owns the single
    # runtime-FK exception list.
    missing = groundability_failures(program, anchors0)
    if not program.has_verifiable_success():
        missing = ["unverifiable_stage_terminal:" + ";".join(
            program.non_geometric_reasons()
            or ["empty_or_unmeasurable_terminal"])] + missing
    # INERT-SUCCESS refusal (generality audit, 2026-07-24): strengthen the
    # all-Stage measurability question above with a final-goal influence check:
    # a final terminal the controlled state cannot influence -- e.g. a goal on
    # a body the twin cannot move -- is unpursuable and ungradable. Every
    # candidate scores identically, the ranking falls through to the physics
    # tiebreak, and the episode reports a success the program never tested
    # (measured on 9 of 16 audited tasks). Structural, not numerical (see
    # observable_terminal), and task-independent: this check reads the
    # program's own predicates and the grounded scene, never the task text.
    if program is not None and program.stages and not missing:
        from oap.program.verdict import observable_terminal
        if not observable_terminal(program.stages[-1], anchors0):
            missing = ["inert_success:" +
                       (program.stages[-1].name or "final_stage")] + missing
    if missing:
        reason = f"ungroundable:{','.join(missing)}"
        logger.error("[program] CANNOT_VERIFY (%s) -- refusing to move. "
                     "Re-scene or restate the task.", reason)
        write_json_atomic(out / "verdict.json", {
            "outcome": "CANNOT_VERIFY", "provenance": "groundability", "reason": reason,
            "program_provenance": program.provenance,
            "non_geometric": program.non_geometric_reasons()})
        episode.record_verdict({"outcome": "CANNOT_VERIFY", "tier": "refused",
                                "provenance": "groundability", "reason": reason})
        episode.finalize(success=False, outcome="CANNOT_VERIFY")
        return {"success": False, "outcome": "CANNOT_VERIFY", "reason": reason, "chunks": 0}

    from oap.twin.batched_rollout import plan_dt_from_env

    execution_dt_s = float(plan_world.model.opt.timestep)
    planning_dt_s = plan_dt_from_env() or execution_dt_s
    if cfg.execute and not np.isclose(
        planning_dt_s,
        execution_dt_s,
        rtol=0.0,
        atol=1e-12,
    ):
        raise safety.SafetyError(
            "real execution requires planning_dt_s == execution_dt_s so the "
            "selected GPU prefix control row is the atomic controller endpoint"
        )
    horizon_s = float(cfg.horizon_steps) * execution_dt_s
    num_knots = validate_num_knots(cfg.num_knots)
    sigma_fraction = validate_local_sigma_fraction(
        cfg.sigma_fraction
    )
    sampler_mode = validate_predictive_sampler_mode(cfg.sampler_mode)
    mtp_jaw_mode = validate_mtp_jaw_mode(cfg.mtp_jaw_mode)
    if (
        sampler_mode != PREDICTIVE_SAMPLER_MTP_GLOBAL_LOCAL
        and mtp_jaw_mode != MTP_JAW_MODE_PRESERVE_NOMINAL
    ):
        raise ValueError(
            "sampled MTP jaw mode requires sampler_mode='mtp_global_local'"
        )
    local_sigma_fraction = validate_local_sigma_fraction(
        cfg.local_sigma_fraction
    )
    cem_rounds = validate_cem_rounds(cfg.cem_rounds)
    cem_elite_fraction = validate_cem_elite_fraction(
        cfg.cem_elite_fraction
    )
    cem_min_std_fraction = validate_local_sigma_fraction(
        cfg.cem_min_std_fraction
    )
    optimizer = str(cfg.optimizer).strip().lower()
    if optimizer not in {"cem", "mppi"}:
        raise ValueError("optimizer must be 'cem' or 'mppi'")
    candidate_selection_mode = _validate_candidate_selection_config(
        cfg.candidate_selection_mode,
        optimizer=optimizer,
        sampler_mode=sampler_mode,
    )
    candidate_selection_kwargs = _candidate_selection_call_kwargs(
        candidate_selection_mode
    )
    mppi_temperature = validate_mppi_temperature(cfg.mppi_temperature)
    mppi_execution_mode = validate_mppi_execution_mode(
        cfg.mppi_execution_mode
    )
    arm_velocity_weight = validate_arm_velocity_weight(
        cfg.arm_velocity_weight
    )
    if (
        optimizer != "mppi"
        and mppi_execution_mode != MPPI_EXECUTION_SOFTMAX_WEIGHTED_MEAN
    ):
        raise ValueError("best_valid_sample requires optimizer='mppi'")
    mppi_execution_kwargs = _mppi_execution_call_kwargs(
        mppi_execution_mode
    )
    prefix = resolve_execution_prefix(
        horizon_steps=cfg.horizon_steps,
        execution_prefix_fraction=cfg.execution_prefix_fraction,
        execution_prefix_steps=cfg.execution_prefix_steps,
    )
    prefix_frac = prefix.effective_fraction
    continuous_requested_prefix_s = horizon_s * prefix_frac
    gpu_discrete_prefix_steps = prefix.resolved_steps
    gpu_discrete_prefix_s = (
        gpu_discrete_prefix_steps * execution_dt_s
    )
    planning_steps = max(2, int(round(horizon_s / planning_dt_s)))
    from oap.loop.planner_profile import PlannerProfile

    runtime_planner_profile = PlannerProfile(
        pool_size=int(cfg.pool_size),
        horizon_steps=int(cfg.horizon_steps),
        num_knots=num_knots,
        sigma_fraction=sigma_fraction,
        sampler_mode=sampler_mode,
        candidate_selection_mode=candidate_selection_mode,
        mtp_jaw_mode=mtp_jaw_mode,
        local_sigma_fraction=local_sigma_fraction,
        cem_rounds=cem_rounds,
        cem_elite_fraction=cem_elite_fraction,
        cem_min_std_fraction=cem_min_std_fraction,
        execution_prefix_steps=int(prefix.resolved_steps),
        max_stage_cycles=int(cfg.max_stage_cycles),
        mppi_execution_mode=mppi_execution_mode,
        mppi_temperature=mppi_temperature,
        arm_velocity_weight=arm_velocity_weight,
        direct_torque_s0_horizon_prefix_arm=(
            cfg.direct_torque_s0_horizon_prefix_arm
        ),
        table_contact_force_weight=cfg.table_contact_force_weight,
        control_regularizer_calibration_profile=(
            cfg.control_regularizer_calibration_profile
        ),
        joint_velocity_force_trial_budget=(
            cfg.joint_velocity_force_trial_budget
        ),
        joint_velocity_force_algorithm_arm=(
            cfg.joint_velocity_force_algorithm_arm
        ),
        joint_velocity_force_dial_arm=cfg.joint_velocity_force_dial_arm,
        joint_velocity_force_phase3_finalist_arm=(
            cfg.joint_velocity_force_phase3_finalist_arm
        ),
        joint_velocity_force_phase3_seed=cfg.joint_velocity_force_phase3_seed,
        cost_shaping_profile=cfg.cost_shaping_profile,
        unified_mppi_effort_profile=cfg.unified_mppi_effort_profile,
        execution_budget_protocol=cfg.execution_budget_protocol,
        pick_v8_causal_profile=cfg.pick_v8_causal_profile,
        vlm_generated_program=bool(cfg.vlm_generated_program),
    )
    planner_config = {
        # Evidence only; nothing reads it back.  ``prefix_qpos`` is a vector in
        # this model's coordinates, so a recorded episode cannot be replayed or
        # rendered without naming the model it was measured against.  A plain
        # string: no file is opened and no hash is computed here.
        "plan_xml": str(plan_xml),
        "algorithm": (
            "mppi" if optimizer == "mppi" else
            "cem" if cem_rounds > 1 else "predictive_sampling"
        ),
        "backend": "mjwarp",
        "physics_and_program_cost_device": "gpu",
        "host_role": "orchestration_and_selected_payload_only",
        "total_rollout_budget": int(cfg.pool_size),
        "samples_per_round": int(cfg.pool_size) // cem_rounds,
        "optimization_rounds": cem_rounds,
        "cem_elite_fraction": cem_elite_fraction,
        "cem_min_std_fraction": cem_min_std_fraction,
        "mppi_temperature": (
            mppi_temperature if optimizer == "mppi" else None
        ),
        "mppi_execution_mode": (
            mppi_execution_mode if optimizer == "mppi" else None
        ),
        "arm_velocity_weight": (
            arm_velocity_weight if optimizer == "mppi" else None
        ),
        **(
            {"control_profile": cfg.control_profile}
            if cfg.control_profile is not None else {}
        ),
        **(
            {
                "execution_budget": execution_budget_identity(
                    cfg.execution_budget_protocol
                )
            }
            if cfg.execution_budget_protocol is not None
            else {}
        ),
        **(
            {
                (
                    "control_regularizer"
                    if cfg.control_regularizer_calibration_profile
                    == UNIFIED_CONTROL_REGULARIZER_V1
                    else "control_regularizer_calibration"
                ): control_regularizer_formula_identity(
                    cfg.control_regularizer_calibration_profile
                )
            }
            if cfg.control_regularizer_calibration_profile is not None
            else {}
        ),
        **(
            {"cost_shaping": cost_shaping_identity(cfg.cost_shaping_profile)}
            if cfg.cost_shaping_profile is not None
            else {}
        ),
        "record_candidate_cost_telemetry": bool(
            cfg.record_candidate_cost_telemetry
        ),
        "num_knots": num_knots,
        # Paper-compatible MPPI actions are generalized efforts held constant
        # for native simulation steps.  CEM retains its legacy position spline.
        "spline": (
            "linear"
            if cfg.pick_v8_causal_profile is not None
            or (
                cfg.unified_mppi_effort_profile is not None
                and unified_mppi_effort_uses_proven_pick_carrier(
                    cfg.unified_mppi_effort_profile
                )
            )
            else "zero_order_hold"
            if optimizer == "mppi"
            else "linear"
        ),
        "horizon_steps": int(cfg.horizon_steps),
        "execution_steps": int(cfg.horizon_steps),
        "planning_steps": planning_steps,
        "execution_dt_s": execution_dt_s,
        "planning_dt_s": float(planning_dt_s),
        "horizon_s": horizon_s,
        "execution_prefix_frac": prefix_frac,
        "execution_prefix_steps": prefix.requested_steps,
        "resolved_execution_prefix_steps": prefix.resolved_steps,
        "continuous_requested_prefix_s": continuous_requested_prefix_s,
        "gpu_discrete_prefix_steps": gpu_discrete_prefix_steps,
        "gpu_discrete_prefix_s": gpu_discrete_prefix_s,
        "simulation_prefix_s": gpu_discrete_prefix_s,
        # The integer real-controller duration is bound later from the leased
        # server's top-level control_hz and recorded with every execution.
        "real_execution_prefix_s": None,
        "stage_entry_nominal": "current_controller_hold",
        "within_stage_nominal": "shift_previous_winner_last_hold",
        "sampler_mode": sampler_mode,
        "candidate_selection_mode": candidate_selection_mode,
        "mtp_jaw_mode": mtp_jaw_mode,
        "proposal_families": (
            ["nominal", "diagonal_gaussian"]
            if sampler_mode in PREDICTIVE_SAMPLER_MPPI_HALTON_MODES
            else [
                "nominal",
                "local_diagonal_gaussian",
                "broad_diagonal_gaussian",
            ]
        ),
        "sigma_half_range": sigma_fraction,
        "local_sigma_half_range": local_sigma_fraction,
        "max_stage_cycles": int(cfg.max_stage_cycles),
        "mppi_cycle_budget_scope": cfg.mppi_cycle_budget_scope,
        "profile": runtime_planner_profile.to_dict(),
        "profile_sha256": runtime_planner_profile.sha256,
        **(
            {
                "unified_mppi_effort": unified_mppi_effort_identity(
                    cfg.unified_mppi_effort_profile,
                    execution_budget_protocol=cfg.execution_budget_protocol,
                )
            }
            if cfg.unified_mppi_effort_profile is not None
            else {}
        ),
        **(
            {
                "pick_v8_causal": pick_v8_causal_controller_identity(
                    cfg.pick_v8_causal_profile
                )
            }
            if cfg.pick_v8_causal_profile is not None
            else {}
        ),
    }
    coverage_telemetry = direct_torque_s0_coverage_telemetry(
        selected_k=cfg.direct_torque_s0_coverage_k,
        horizon_steps=int(cfg.horizon_steps),
        execution_dt_s=execution_dt_s,
        sigma_fraction=sigma_fraction,
        local_sigma_fraction=local_sigma_fraction,
    )
    if coverage_telemetry is not None:
        planner_config["sampling_coverage_matrix"] = coverage_telemetry
    horizon_prefix_telemetry = direct_torque_s0_horizon_prefix_telemetry(
        selected_arm=cfg.direct_torque_s0_horizon_prefix_arm,
        horizon_steps=int(cfg.horizon_steps),
        num_knots=num_knots,
        execution_prefix_steps=int(prefix.resolved_steps),
        execution_dt_s=execution_dt_s,
        pool_size=int(cfg.pool_size),
        sigma_fraction=sigma_fraction,
        local_sigma_fraction=local_sigma_fraction,
    )
    if horizon_prefix_telemetry is not None:
        planner_config["horizon_prefix_matrix"] = horizon_prefix_telemetry
    algorithm_arm_telemetry = joint_velocity_force_algorithm_arm_telemetry(
        selected_arm=cfg.joint_velocity_force_algorithm_arm,
        pool_size=int(cfg.pool_size),
        horizon_steps=int(cfg.horizon_steps),
        num_knots=num_knots,
        execution_prefix_steps=int(prefix.resolved_steps),
        sampler_mode=sampler_mode,
        mtp_jaw_mode=mtp_jaw_mode,
        local_sigma_fraction=local_sigma_fraction,
        sigma_fraction=sigma_fraction,
        mppi_temperature=mppi_temperature,
        mppi_execution_mode=mppi_execution_mode,
        candidate_selection_mode=candidate_selection_mode,
        cem_rounds=cem_rounds,
        cem_elite_fraction=cem_elite_fraction,
        cem_min_std_fraction=cem_min_std_fraction,
        seed=int(cfg.seed),
        arm_velocity_weight=arm_velocity_weight,
    )
    if algorithm_arm_telemetry is not None:
        planner_config["joint_velocity_force_algorithm_arm"] = (
            algorithm_arm_telemetry
        )
    dial_arm_telemetry = joint_velocity_force_dial_arm_telemetry(
        selected_arm=cfg.joint_velocity_force_dial_arm,
        pool_size=int(cfg.pool_size),
        horizon_steps=int(cfg.horizon_steps),
        num_knots=num_knots,
        execution_prefix_steps=int(prefix.resolved_steps),
        sampler_mode=sampler_mode,
        mtp_jaw_mode=mtp_jaw_mode,
        local_sigma_fraction=local_sigma_fraction,
        sigma_fraction=sigma_fraction,
        mppi_temperature=mppi_temperature,
        mppi_execution_mode=mppi_execution_mode,
        candidate_selection_mode=candidate_selection_mode,
        cem_rounds=cem_rounds,
        cem_elite_fraction=cem_elite_fraction,
        cem_min_std_fraction=cem_min_std_fraction,
        seed=int(cfg.seed),
        arm_velocity_weight=arm_velocity_weight,
    )
    if dial_arm_telemetry is not None:
        planner_config["joint_velocity_force_dial_arm"] = dial_arm_telemetry
    phase3_finalist_telemetry = (
        joint_velocity_force_phase3_finalist_arm_telemetry(
            selected_arm=cfg.joint_velocity_force_phase3_finalist_arm,
            selected_seed=cfg.joint_velocity_force_phase3_seed,
            pool_size=int(cfg.pool_size),
            horizon_steps=int(cfg.horizon_steps),
            num_knots=num_knots,
            execution_prefix_steps=int(prefix.resolved_steps),
            sampler_mode=sampler_mode,
            mtp_jaw_mode=mtp_jaw_mode,
            local_sigma_fraction=local_sigma_fraction,
            sigma_fraction=sigma_fraction,
            mppi_temperature=mppi_temperature,
            mppi_execution_mode=mppi_execution_mode,
            candidate_selection_mode=candidate_selection_mode,
            cem_rounds=cem_rounds,
            cem_elite_fraction=cem_elite_fraction,
            cem_min_std_fraction=cem_min_std_fraction,
            arm_velocity_weight=arm_velocity_weight,
            max_stage_cycles=int(cfg.max_stage_cycles),
            mppi_cycle_budget_scope=cfg.mppi_cycle_budget_scope,
        )
    )
    if phase3_finalist_telemetry is not None:
        planner_config["joint_velocity_force_phase3_finalist"] = (
            phase3_finalist_telemetry
        )
    cost_focus_stage90_telemetry = (
        joint_velocity_force_cost_focus_stage90_telemetry(
            selected_budget=cfg.joint_velocity_force_trial_budget,
            cost_shaping_profile=cfg.cost_shaping_profile,
            pool_size=int(cfg.pool_size),
            horizon_steps=int(cfg.horizon_steps),
            num_knots=num_knots,
            execution_prefix_steps=int(prefix.resolved_steps),
            sampler_mode=sampler_mode,
            mtp_jaw_mode=mtp_jaw_mode,
            local_sigma_fraction=local_sigma_fraction,
            sigma_fraction=sigma_fraction,
            mppi_temperature=mppi_temperature,
            mppi_execution_mode=mppi_execution_mode,
            candidate_selection_mode=candidate_selection_mode,
            cem_rounds=cem_rounds,
            cem_elite_fraction=cem_elite_fraction,
            cem_min_std_fraction=cem_min_std_fraction,
            max_stage_cycles=int(cfg.max_stage_cycles),
            mppi_cycle_budget_scope=cfg.mppi_cycle_budget_scope,
            seed=int(cfg.seed),
            arm_velocity_weight=arm_velocity_weight,
            table_contact_force_weight=cfg.table_contact_force_weight,
        )
    )
    if cost_focus_stage90_telemetry is not None:
        planner_config["joint_velocity_force_cost_focus_stage90"] = (
            cost_focus_stage90_telemetry
        )
    cost_focus_screen30_telemetry = (
        joint_velocity_force_cost_focus_screen30_telemetry(
            selected_budget=cfg.joint_velocity_force_trial_budget,
            cost_shaping_profile=cfg.cost_shaping_profile,
            pool_size=int(cfg.pool_size),
            horizon_steps=int(cfg.horizon_steps),
            num_knots=num_knots,
            execution_prefix_steps=int(prefix.resolved_steps),
            sampler_mode=sampler_mode,
            mtp_jaw_mode=mtp_jaw_mode,
            local_sigma_fraction=local_sigma_fraction,
            sigma_fraction=sigma_fraction,
            mppi_temperature=mppi_temperature,
            mppi_execution_mode=mppi_execution_mode,
            candidate_selection_mode=candidate_selection_mode,
            cem_rounds=cem_rounds,
            cem_elite_fraction=cem_elite_fraction,
            cem_min_std_fraction=cem_min_std_fraction,
            max_stage_cycles=int(cfg.max_stage_cycles),
            mppi_cycle_budget_scope=cfg.mppi_cycle_budget_scope,
            seed=int(cfg.seed),
            arm_velocity_weight=arm_velocity_weight,
            table_contact_force_weight=cfg.table_contact_force_weight,
        )
    )
    if cost_focus_screen30_telemetry is not None:
        planner_config["joint_velocity_force_cost_focus_screen30"] = (
            cost_focus_screen30_telemetry
        )
    device_seed_evidence = _device_proposal_seed_evidence(
        trial_budget=cfg.joint_velocity_force_trial_budget,
        dial_arm=cfg.joint_velocity_force_dial_arm,
        phase3_arm=cfg.joint_velocity_force_phase3_finalist_arm,
        phase3_seed=cfg.joint_velocity_force_phase3_seed,
    )
    if cfg.control_profile is not None:
        # Bind the scratch-only actuator arm to evidence beside the established
        # planner profile.  The production profile object and its SHA stay
        # byte-stable when the experiment option is absent.
        planner_config["experiment_identity"] = {
            "scope": (
                "offline_pick_v8_single_variable_causal"
                if cfg.pick_v8_causal_profile is not None
                else "offline_unified_mppi_effort_four_task_registry"
                if cfg.unified_mppi_effort_profile is not None
                else "offline_joint_velocity_force_four_task_registry"
                if cfg.joint_velocity_force_task_id is not None
                else "offline_pick_v9_s0_direct_torque_sampling_coverage"
                if coverage_telemetry is not None
                else "offline_pick_v9_s0_direct_torque_horizon_prefix_2x2"
                if horizon_prefix_telemetry is not None
                else "offline_pick_v9_s0_control_profile"
            ),
            "control_profile": cfg.control_profile,
            "diagnostic_smoke_only": bool(
                cfg.control_profile_smoke_one_cycle
            ),
            "completed_cycle_limit": int(cfg.max_stage_cycles),
            **(
                {
                    "registered_task_id": cfg.joint_velocity_force_task_id,
                    "planner_table_contact_force_weight": (
                        joint_velocity_force_registered_table_force_weight(
                            cfg.joint_velocity_force_task_id
                        )
                    ),
                    "table_force_policy": (
                        "common_zero_not_a_task_or_controller_knob"
                    ),
                }
                if cfg.joint_velocity_force_task_id is not None
                else {}
            ),
            **(
                {
                    "joint_velocity_force_trial_budget": (
                        cfg.joint_velocity_force_trial_budget
                    ),
                    "trial_budget_semantics": (
                        "evidence_only_stage_completed_cycle_limit"
                    ),
                }
                if cfg.joint_velocity_force_trial_budget is not None
                else {}
            ),
            "compound_experiment_id": (
                "pick_v8_single_variable:" + cfg.pick_v8_causal_profile
                if cfg.pick_v8_causal_profile is not None
                else control_profile_compound_experiment_id(
                    control_profile=cfg.control_profile,
                    smoke_one_cycle=cfg.control_profile_smoke_one_cycle,
                    direct_torque_s0_coverage_k=(
                        cfg.direct_torque_s0_coverage_k
                    ),
                    direct_torque_s0_horizon_prefix_arm=(
                        cfg.direct_torque_s0_horizon_prefix_arm
                    ),
                    joint_velocity_force_task_id=(
                        cfg.joint_velocity_force_task_id
                    ),
                    joint_velocity_force_trial_budget=(
                        cfg.joint_velocity_force_trial_budget
                    ),
                    joint_velocity_force_algorithm_arm=(
                        cfg.joint_velocity_force_algorithm_arm
                    ),
                    joint_velocity_force_dial_arm=(
                        cfg.joint_velocity_force_dial_arm
                    ),
                    joint_velocity_force_phase3_finalist_arm=(
                        cfg.joint_velocity_force_phase3_finalist_arm
                    ),
                    joint_velocity_force_phase3_seed=(
                        cfg.joint_velocity_force_phase3_seed
                    ),
                    control_regularizer_calibration_profile=(
                        cfg.control_regularizer_calibration_profile
                    ),
                    cost_shaping_profile=cfg.cost_shaping_profile,
                )
            ),
            **device_seed_evidence,
            "episode_seed": int(cfg.seed),
            "stage_entry_jaw_source": "clean_2ef_load_world_direct_effort_seed",
            "stage_entry_jaw_semantics": "signed_latent_open_minus1_closed_plus1",
            "legacy_initial_transcode": (
                "direct_effort_n_to_signed_latent_to_width_m_once"
            ),
            "carrier_semantics": (
                "profile_native_command_not_measured_qpos_or_second_physics"
            ),
            "private_rollout_xml_identity": (
                "ab10_velocity_kv600_gn01_width_kp2700"
                if cfg.control_profile == CONTROL_PROFILE_LEGACY_VELOCITY_WIDTH
                else "ab10_velocity_linear_v8_gn01_direct_signed_force_pm80"
                if cfg.control_profile == "legacy_velocity_effort"
                else "ab10_velocity_kv600_gn01_direct_signed_force_pm80"
                if cfg.control_profile == CONTROL_PROFILE_JOINT_VELOCITY_FORCE
                else "clean_2ef_native_direct_torque_force"
            ),
            **(
                {
                    "hybrid_native_jaw_transcode": "direct_effort_identity",
                    "arm_action_units": "velocity_rad_s",
                    "jaw_action_units": "signed_effort_n",
                    "measured_width_role": "state_only_never_action",
                }
                if cfg.control_profile
                == CONTROL_PROFILE_JOINT_VELOCITY_FORCE
                else {}
            ),
            "base_planner_profile": runtime_planner_profile.to_dict(),
            "base_planner_profile_sha256": runtime_planner_profile.sha256,
            **(
                {"sampling_coverage_matrix": coverage_telemetry}
                if coverage_telemetry is not None
                else {}
            ),
            **(
                {
                    "joint_velocity_force_algorithm_arm": (
                        algorithm_arm_telemetry
                    )
                }
                if algorithm_arm_telemetry is not None
                else {}
            ),
            **(
                {"joint_velocity_force_dial_arm": dial_arm_telemetry}
                if dial_arm_telemetry is not None
                else {}
            ),
            **(
                {
                    "joint_velocity_force_phase3_finalist": (
                        phase3_finalist_telemetry
                    )
                }
                if phase3_finalist_telemetry is not None
                else {}
            ),
            **(
                {"horizon_prefix_matrix": horizon_prefix_telemetry}
                if horizon_prefix_telemetry is not None
                else {}
            ),
            "program_instruction": program.instruction,
            "program_provenance": program.provenance,
            "program_stage_names": [stage.name for stage in program.stages],
        }
    if cfg.mppi_stage_execution_prefix_steps is not None:
        planner_config["mppi_stage_execution_prefix_steps"] = list(
            cfg.mppi_stage_execution_prefix_steps
        )
    remote_solve_step = None
    if cfg.remote_planner_url is not None:
        from oap.remote_planner import (
            HttpPlannerTransport,
            RemotePlannerClient,
            RemoteSolveStep,
            assets_sha256,
            model_sha256,
            planner_source_sha256,
        )

        assert cfg.remote_planner_model_sha256 is not None
        assert cfg.remote_planner_assets_sha256 is not None
        assert cfg.remote_planner_sha256 is not None
        assert cfg.remote_planner_service_instance_id is not None
        assert cfg.remote_planner_deadline_s is not None
        actual_identities = {
            "model_sha256": model_sha256(plan_xml),
            "assets_sha256": assets_sha256(plan_xml),
            "planner_sha256": planner_source_sha256(),
        }
        expected_identities = {
            "model_sha256": cfg.remote_planner_model_sha256,
            "assets_sha256": cfg.remote_planner_assets_sha256,
            "planner_sha256": cfg.remote_planner_sha256,
        }
        if actual_identities != expected_identities:
            raise safety.SafetyError(
                "local planning twin/source does not match approved remote "
                f"identity: expected={expected_identities}, "
                f"actual={actual_identities}"
            )
        try:
            remote_client = RemotePlannerClient(
                HttpPlannerTransport(cfg.remote_planner_url)
            )
            remote_info = remote_client.require_identity(
                service_instance_id=(
                    cfg.remote_planner_service_instance_id
                ),
                planner_sha256=cfg.remote_planner_sha256,
                model_sha256=cfg.remote_planner_model_sha256,
                assets_sha256=cfg.remote_planner_assets_sha256,
                planner_profile=runtime_planner_profile,
            )
        except Exception as exc:
            raise safety.SafetyError(
                "remote planner identity preflight failed; no robot "
                f"connection was requested ({exc})"
            ) from exc
        remote_solve_step = RemoteSolveStep(
            client=remote_client,
            episode_id=episode.episode_id,
            model_sha256=cfg.remote_planner_model_sha256,
            assets_sha256=cfg.remote_planner_assets_sha256,
            planner_sha256=cfg.remote_planner_sha256,
            service_instance_id=(
                cfg.remote_planner_service_instance_id
            ),
            max_latency_s=float(cfg.remote_planner_deadline_s),
            planner_profile=runtime_planner_profile,
        )
        planner_config.update({
            "backend": "remote_mjwarp_h100",
            "remote_url": cfg.remote_planner_url,
            "remote_service_instance_id": (
                cfg.remote_planner_service_instance_id
            ),
            **actual_identities,
        })
        episode.record_event("remote_planner_identity", remote_info)
    episode.record_planner_config(planner_config)
    from oap.program.anchors import Anchor

    gripper_center = np.asarray(
        plan_world.data.site_xpos[int(plan_world.site_id)],
        dtype=float,
    ).copy()
    gripper_rotation = np.asarray(
        plan_world.data.site_xmat[int(plan_world.site_id)],
        dtype=float,
    ).reshape(3, 3)
    gripper_axis = gripper_rotation[:, 2].copy()
    gripper_closing_axis = gripper_rotation[:, 1].copy()
    if (
        gripper_center.shape != (3,)
        or not np.all(np.isfinite(gripper_center))
        or gripper_axis.shape != (3,)
        or not np.all(np.isfinite(gripper_axis))
        or np.linalg.norm(gripper_axis) <= 1e-9
        or gripper_closing_axis.shape != (3,)
        or not np.all(np.isfinite(gripper_closing_axis))
        or np.linalg.norm(gripper_closing_axis) <= 1e-9
    ):
        raise safety.SafetyError(
            "planning model cannot ground reserved gripper point/axes from FK"
        )
    anchors0.add(
        "gripper_center",
        Anchor(point=gripper_center, axis=gripper_axis, kind="robot"),
    )
    anchors0.add(
        "gripper_closing_axis",
        Anchor(point=gripper_center, axis=gripper_closing_axis, kind="robot"),
    )
    sim_tcp_site_z = float(plan_world.model.site_pos[plan_world.site_id][2])
    tcp_dz = sim_tcp_site_z - REAL_GN01_TCP_M
    logger.info("[tcp-site] compiled planning site z=%.5fm; live closed-tip "
                "calibration=%.5fm; frame offset=%+.2fmm",
                sim_tcp_site_z, REAL_GN01_TCP_M, 1000.0 * tcp_dz)
    if ((cfg.execute or cfg.prepare_real_execution)
            and abs(tcp_dz) > 5e-4):
        raise safety.SafetyError(
            f"compiled planning site {sim_tcp_site_z:.5f} m does not match the "
            f"live profile {REAL_GN01_TCP_M:.5f} m within 0.5 mm")
    # Run one joint-space orchestrator over the complete explicit program,
    # carrying measured physical state across stages. Dry-run prefixes use the
    # selected MJWarp state. Real execution supplies the two hooks below:
    # numeric joint/gripper prefix execution followed by whole-scene
    # re-observation and twin synchronization.
    #
    # FoundationPose keeps its persistent register-once -> track state across
    # the episode.  After each executed prefix, one current tracking snapshot
    # updates the twin and grades the Stage terminal.  Only the program's
    # measured terminal predicates advance the Stage; no second observation
    # or host-side completion condition is added.
    from oap.loop.mpc_episode import run_mpc_episode
    obj_pose7 = np.array([subject_pose0[0], subject_pose0[1], subject_pose0[2],
                          *object_body_quat(subject_pose0[3], subject.q_om)])
    session = None
    execution_feasibility: dict[str, Any] | None = None
    execute_prefix_fn = None
    reobserve_fn = None
    camera_client = None
    continuous_tracker = None
    tracker_registration_evidence: dict[str, Any] | None = None
    tracker_ready = False
    last_scene_observation = scene_obs0
    initial_robot_timestamp_s: float | None = None

    def _anchor_reliability(
        scene: dict[str, Any],
    ) -> dict[str, dict[str, float]]:
        measured = observe_mod.ground_observation(
            scene,
            subject,
            objects,
            table_top_z=TABLE_TOP_Z,
            reference=reference,
            initial_anchors=anchors0,
        )
        return {
            name: {
                "confidence": float(measured[name].confidence),
                "visibility": float(measured[name].visibility),
            }
            for name in measured.names()
        }

    if cfg.execute:
        from oap.loop import execute as execute_mod
        from oap.loop.continuous_tracking import (
            ContinuousSceneTracker,
        )
        from oap.loop.execution_authorization import (
            require_and_consume_execution_authorization,
        )
        from oap.workers.client import ZedStreamClient

        authorization = require_and_consume_execution_authorization(
            cfg,
            program_sha256=program_sha,
            bundle_manifest=Path(cfg.bundle_manifest),
            out_dir=out,
        )
        assert authorization is not None
        checkpoint_timeout_raw = cfg.foundationpose_checkpoint_timeout_s
        if checkpoint_timeout_raw is None:
            raise safety.SafetyError(
                "real execution requires an explicit FoundationPose "
                "checkpoint timeout"
            )
        checkpoint_timeout_s = float(checkpoint_timeout_raw)
        if (
            not np.isfinite(checkpoint_timeout_s)
            or checkpoint_timeout_s <= 0.0
        ):
            raise safety.SafetyError(
                "FoundationPose checkpoint timeout must be finite and positive"
            )
        episode.record_event(
            "execution_authorization",
            authorization.to_dict(),
        )
        camera_client = ZedStreamClient(
            py=cfg.observer_python,
            log_path=out / "zed_stream_worker.log",
            timeout_s=float(cfg.foundationpose_timeout_s),
        )
        required_body_names = {obj.name for obj in objects}
        assert observation_readiness is not None

        # One mask-scored frame establishes object identity and registration.
        # Every planning checkpoint after that consumes the already-running,
        # maskless FoundationPose state; no second segmentation/verdict frame
        # is captured.
        try:
            tracker_seed_scene = observe_mod.observe_scene(
                cfg,
                objects,
                subject,
                "scene_tracker_seed",
                purpose="replan",
                plan_world=None,
                fp_client=fp_client,
                previous=last_scene_observation,
                z_real_to_plan=z_real_to_plan,
                subject_track_only=False,
                camera_client=camera_client,
                capture_segmentation=True,
                allow_same_frame_recovery=True,
                require_mask_score=True,
            )
        except BaseException:
            _call_optional(
                camera_client,
                ("stop", "close", "shutdown"),
            )
            for client in clients.values():
                _call_optional(
                    client,
                    ("stop", "close", "shutdown"),
                )
            raise

        def _continuous_scene_producer(
            sequence: int,
            previous: dict[str, Any] | None,
        ) -> dict[str, Any]:
            assert camera_client is not None
            scene = observe_mod.observe_scene(
                cfg,
                objects,
                subject,
                f"scene_stream_slot_{sequence % 2}",
                purpose="replan",
                plan_world=None,
                fp_client=fp_client,
                previous=previous,
                z_real_to_plan=z_real_to_plan,
                subject_track_only=True,
                camera_client=camera_client,
                capture_segmentation=False,
                allow_same_frame_recovery=False,
                require_mask_score=False,
            )
            scene["capture_dir"] = None
            scene["capture_storage"] = (
                "ephemeral_two_slot_continuous_tracking"
            )
            return scene

        continuous_tracker = ContinuousSceneTracker(
            _continuous_scene_producer,
            previous=tracker_seed_scene,
            accept_unscored_tracking=True,
        )
        continuous_tracker.start()

        # No command is permitted until both identity registration and one
        # fresh state from the sole continuous producer are available.
        try:
            tracker_ready_scene = continuous_tracker.consume_latest(
                max_age_s=float(
                    observation_readiness.max_observation_age_s
                ),
                timeout_s=checkpoint_timeout_s,
                required_body_names=required_body_names,
            )
            home = safety.load_home_posture(cfg.home_posture_json)
            tracker_registration_evidence, session = (
                _connect_after_tracker_ready(
                    tracker_seed_scene,
                    required_body_names=required_body_names,
                    max_age_s=float(
                        observation_readiness.max_observation_age_s
                    ),
                    min_mask_iou=float(
                        observation_readiness.min_terminal_anchor_reliability
                    ),
                    connect=lambda: execute_mod.connect_robot(
                        cfg,
                        home,
                        z_real_to_plan=z_real_to_plan,
                        table_top_z=TABLE_TOP_Z,
                        out_dir=out,
                    ),
                )
            )
        except BaseException:
            continuous_tracker.stop()
            _call_optional(
                camera_client,
                ("stop", "close", "shutdown"),
            )
            for client in clients.values():
                _call_optional(
                    client,
                    ("stop", "close", "shutdown"),
                )
            raise
        tracker_ready = True
        tracker_ready_evidence = {
            "mask_scored_registration": tracker_registration_evidence,
            "tracking_state": dict(
                tracker_ready_scene.get("continuous_tracking") or {}
            ),
            "tracking_state_valid": True,
        }
        episode.record_event(
            "continuous_tracker_ready",
            tracker_ready_evidence,
        )
        episode.record_observation(
            0, "tracker_ready", tracker_ready_scene
        )
        last_scene_observation = tracker_ready_scene

        effective = session.extra["controller_effective_joint_limits"]
        gripper_runtime = session.extra["controller_gripper_limits"]
        execution_speed_scale = 0.3
        if (
            execution_speed_scale
            > float(effective["max_joint_speed_scale"]) + 1e-12
        ):
            raise safety.SafetyError(
                "OaP execution speed scale exceeds the leased controller "
                "ceiling before planning"
            )
        prefix_knot_times = execute_mod.joint_prefix_knot_times(
            num_knots,
            prefix_frac,
            horizon_steps=int(cfg.horizon_steps),
        )
        prefix_timing_contract = execute_mod.execution_prefix_schedule(
            horizon_steps=int(cfg.horizon_steps),
            execution_dt_s=execution_dt_s,
            prefix_frac=prefix_frac,
            prefix_knot_times=prefix_knot_times,
            control_hz=float(session.extra["controller_control_hz"]),
        )
        if not prefix_timing_contract["controller_timing_compatible"]:
            raise safety.SafetyError(
                "GPU prefix/controller cumulative timing quantization exceeds "
                "half a controller tick; refusing to plan an undispatchable batch"
            )
        execution_feasibility = {
            "control_hz": float(session.extra["controller_control_hz"]),
            "execution_dt_s": float(execution_dt_s),
            "controller_timing_compatible": True,
            "controller_prefix_knot_times": list(
                prefix_timing_contract["spline_prefix_knot_times"]
            ),
            "controller_segment_durations_s": list(
                prefix_timing_contract["segment_durations_s"]
            ),
            "joint_position_min_rad": list(
                effective["enforced_position_min_rad"]
            ),
            "joint_position_max_rad": list(
                effective["enforced_position_max_rad"]
            ),
            "joint_velocity_max_rad_s": (
                np.asarray(
                    effective["base_velocity_max_rad_s"],
                    dtype=float,
                )
                * execution_speed_scale
            ).tolist(),
            "joint_torque_max_nm": _fixed_mpc_torque_envelope(
                effective
            ).tolist(),
            "effective_joint_limits_sha256": effective["sha256"],
            "gripper_width_min_m": float(
                gripper_runtime["min_width_m"]
            ),
            "gripper_width_max_m": float(
                gripper_runtime["max_width_m"]
            ),
            "gripper_velocity_min_m_s": float(
                gripper_runtime["min_velocity_m_s"]
            ),
            "gripper_velocity_max_m_s": float(
                gripper_runtime["max_velocity_m_s"]
            ),
            "gripper_limits_sha256": gripper_runtime["sha256"],
        }
        episode.record_event(
            "real_action_feasibility",
            {
                "schema": "oap_real_action_feasibility_v1",
                **execution_feasibility,
                "prefix_timing": prefix_timing_contract,
            },
        )

        # The calibrated home gate is a safety envelope, not a state estimate.
        # Start the first GPU rollout from live arm/gripper telemetry and one
        # subsequent state from the same continuously running tracker.
        #
        # Nothing has been commanded yet, so the jaw command state is not
        # inferred from the aperture -- it is the attended bring-up
        # precondition that the operator confirms and the readiness gate
        # verifies: a fully open, empty gripper. Stating it here keeps the
        # command channel explicit instead of letting a measured width
        # masquerade as a command.
        initial_state = execute_mod.resync_robot_state(
            session,
            plan_world,
            acknowledged_jaw_effort_n=-80.0,
        )
        initial_stamp_raw = getattr(initial_state, "stamp", None)
        initial_robot_timestamp_s = (
            None
            if initial_stamp_raw is None
            else float(initial_stamp_raw)
        )
        initial_live_scene = continuous_tracker.consume_latest(
            max_age_s=float(
                observation_readiness.max_observation_age_s
            ),
            timeout_s=checkpoint_timeout_s,
            min_capture_timestamp_s=initial_robot_timestamp_s,
            required_body_names=required_body_names,
        )
        observe_mod.hot_patch_twin(
            initial_live_scene,
            plan_world,
            subject,
            objects,
        )
        initial_live_scene["subject_name"] = subject.name
        initial_live_scene["evidence"] = {
            "schema": "oap_real_observation_readiness_v1",
            "robot_state_timestamp_s": initial_robot_timestamp_s,
            "tracking_state_valid": True,
            "tracking_state": dict(
                initial_live_scene.get("continuous_tracking") or {}
            ),
            "mask_scored_registration": tracker_registration_evidence,
            "anchor_reliability": _anchor_reliability(
                initial_live_scene
            ),
        }
        last_scene_observation = initial_live_scene
        scene_obs0 = initial_live_scene
        obj_pose7 = np.asarray(
            plan_world.data.qpos[
                int(plan_world.object_qpos_addr):
                int(plan_world.object_qpos_addr) + 7
            ],
            dtype=float,
        ).copy()
        episode.record_observation(0, "planning", initial_live_scene)

        def execute_prefix_fn(*, knots: np.ndarray,
                              action_plan: np.ndarray | None,
                              prefix_frac: float,
                              chunk_idx: int) -> dict[str, Any]:
            assert session is not None
            if not tracker_ready:
                raise safety.SafetyError(
                    "continuous tracker is not mask-scored/ready; refusing "
                    "the first controller command"
                )
            return execute_mod.execute_joint_trajectory(
                session, cfg, knots, action_plan=action_plan,
                chunk_idx=chunk_idx, out_dir=out,
                plan_world=plan_world, prefix_frac=prefix_frac,
                horizon_steps=int(cfg.horizon_steps),
                execution_dt_s=execution_dt_s,
                camera_client=camera_client,
                checkpoint_objects=[
                    (obj.name, obj.sam3_prompt)
                    for obj in objects
                ],
            )

        def reobserve_fn(*, purpose: str, chunk_idx: int,
                         execution_record: dict[str, Any]) -> dict[str, Any]:
            nonlocal last_scene_observation
            assert session is not None
            acknowledged_target = execution_record.get(
                "last_acknowledged_controller_target"
            )
            if acknowledged_target is None:
                raise safety.SafetyError(
                    "successful real prefix did not report its last "
                    "acknowledged controller target"
                )
            efforts = execution_record.get("commanded_gripper_efforts_n") or []
            commanded_effort = (
                None if not efforts else float(
                    execution_record.get("last_commanded_gripper_effort_n",
                                         efforts[-1])))
            commanded_closed = bool(
                commanded_effort is not None and commanded_effort > 0.0
            )
            endpoint = execution_record.get("measured_robot_state_after")
            if not isinstance(endpoint, dict):
                raise safety.SafetyError(
                    "real prefix has no endpoint RobotState captured before "
                    "the post-stop camera frame"
                )
            q_raw = endpoint.get("joint_positions")
            width_raw = endpoint.get("gripper_width_m")
            force_raw = endpoint.get("gripper_force_n")
            state_stamp = endpoint.get("stamp_s")
            width = None if width_raw is None else float(width_raw)
            force = None if force_raw is None else float(force_raw)
            state_stamp = (
                None if state_stamp is None else float(state_stamp)
            )
            if state_stamp is None or not np.isfinite(state_stamp):
                raise safety.SafetyError(
                    "checkpoint robot-state timestamp is missing/non-finite"
                )
            if q_raw is None or width is None:
                raise safety.SafetyError(
                    "checkpoint endpoint joint/gripper telemetry is missing"
                )
            execute_mod.sync_robot_state_to_twin(
                plan_world,
                q_raw,
                width,
                controller_target=acknowledged_target,
            )
            assert observation_readiness is not None
            assert continuous_tracker is not None
            endpoint_monotonic = execution_record.get(
                "endpoint_state_received_monotonic_s"
            )
            scene = continuous_tracker.consume_latest(
                max_age_s=float(
                    observation_readiness.max_observation_age_s
                ),
                timeout_s=checkpoint_timeout_s,
                min_capture_timestamp_s=state_stamp,
                min_capture_monotonic_s=endpoint_monotonic,
                required_body_names=required_body_names,
            )
            # The video window and post-stop packet remain audit evidence. They
            # never pause FoundationPose and are not a second planning/terminal
            # observation.
            scene["recording_evidence"] = {
                "schema": "oap_nonplanning_recording_evidence_v1",
                "video_dir": execution_record.get("video_dir"),
                "video_coverage": execution_record.get("video_coverage"),
                "post_stop_packet": execution_record.get(
                    "post_stop_observation_packet"
                ),
                "post_stop_capture_id": execution_record.get(
                    "post_stop_capture_id"
                ),
                "used_for_planning": False,
            }
            visual = scene["bodies"][subject.name]
            registration_scores = dict(
                (tracker_registration_evidence or {}).get("mask_iou") or {}
            )
            registration_score_raw = registration_scores.get(subject.name)
            try:
                registration_score = (
                    None
                    if registration_score_raw is None
                    else float(registration_score_raw)
                )
            except (TypeError, ValueError):
                registration_score = None
            if (
                registration_score is not None
                and not np.isfinite(registration_score)
            ):
                registration_score = None
            tracking_state = dict(
                scene.get("continuous_tracking") or {}
            )
            min_conf = float(
                observation_readiness.min_terminal_anchor_reliability
            )
            track_reliable = _registration_backed_track_state_valid(
                visual,
                tracking_state=tracking_state,
                registration_score=registration_score,
                min_registration_score=min_conf,
                synchronized_capture=bool(
                    scene.get("synchronized_capture", False)
                ),
            )

            resolution = cfg.gripper_width_resolution_m
            air_close = cfg.gripper_air_close_width_m
            obstruction: bool | None = None
            obstruction_baseline: float | None = None
            if (commanded_closed and width is not None
                    and resolution is not None and air_close is not None):
                obstruction_baseline = float(air_close)
                obstruction = bool(
                    width - obstruction_baseline >= float(resolution))

            spatial_evidence = (
                _current_subject_closing_region_evidence(
                    plan_world,
                    visual,
                )
                if track_reliable
                else {
                    "schema": "oap_object_held_spatial_evidence_v1",
                    "source": (
                        "current_foundationpose_subject_pose+"
                        "measured_robot_gripper_fk+"
                        "integrity_bound_collision_geometry"
                    ),
                    "observable": False,
                    "intersects_closing_region": None,
                    "reason_code": "current_registered_track_invalid",
                    "reason": "current_registered_track_invalid",
                    "contact_history_used": False,
                }
            )
            spatial_intersection = spatial_evidence.get(
                "intersects_closing_region"
            )
            subject_held = _object_held_terminal_state(
                commanded_closed=commanded_closed,
                obstruction=obstruction,
                track_valid=track_reliable,
                intersects_closing_region=(
                    spatial_intersection
                    if isinstance(spatial_intersection, (bool, np.bool_))
                    else None
                ),
                measured_width_m=width,
                commanded_width_m=float(GN01_OPEN_WIDTH_M),
                width_resolution_m=resolution,
            )
            grasp_observable = subject_held is not None
            observe_mod.hot_patch_twin(
                scene,
                plan_world,
                subject,
                objects,
            )
            proxy = execute_mod.observed_contact_proxies(
                plan_world,
                target_name=(reference.name if reference is not None
                             and reference.movable else None))
            scene["subject_name"] = subject.name
            scene["evidence"] = {
                "schema": "oap_real_observation_evidence_v1",
                "gripper_width_m": width,
                "gripper_force_n": force,
                "gripper_measurement_stamp_s": state_stamp,
                "robot_state_timestamp_s": state_stamp,
                "gripper_commanded_effort_n": commanded_effort,
                "gripper_commanded_closed": commanded_closed,
                "held_track_pose_base": visual["pos_base"],
                "held_track_pose_plan": visual["pos_plan"],
                "held_track_quat_wxyz": visual["quat_wxyz"],
                "held_registration_score": registration_score,
                "held_registration_threshold": min_conf,
                "held_registration_source": (
                    "episode_start_mask_scored_registration"
                ),
                "held_tracking_state_valid": track_reliable,
                "held_pointcloud_support": visual.get(
                    "pointcloud_support"
                ),
                "held_track_lost": visual.get("track_lost"),
                "gripper_obstruction_observed": obstruction,
                "gripper_obstruction": {
                    "observed": obstruction,
                    "measured_width_m": width,
                    "baseline_width_m": obstruction_baseline,
                    "required_sensor_increment_m": resolution,
                    "calibration_sha256": (
                        cfg.hardware_evidence_calibration_sha256
                    ),
                    "calibration_source": (
                        cfg.hardware_evidence_calibration_source
                    ),
                    "calibration_hardware_id": (
                        cfg.hardware_evidence_calibration_hardware_id
                    ),
                },
                "held_spatial_attribution": spatial_evidence,
                "held_terminal_definition": (
                    "calibrated_nonempty_obstruction+"
                    "current_registration_backed_track+"
                    "tracked_subject_intersects_measured_fk_closing_region"
                ),
                "grasp_evidence_observable": grasp_observable,
                "subject_held": subject_held,
                "contact_proxy": proxy,
                # Endpoint camera/twin proxies cannot certify causal contact.
                "tool_mediation_observable": False,
                "tool_mediation_source": "unobserved",
                "tool_mediation_coverage": None,
                "anchor_reliability": _anchor_reliability(scene),
                "tracking_state_valid": True,
                "tracking_state": tracking_state,
                "mask_scored_registration": (
                    tracker_registration_evidence
                ),
            }
            last_scene_observation = scene
            episode.record_observation(chunk_idx, purpose, scene)
            return scene

    regularizer_episode_kwargs = (
        {}
        if cfg.control_regularizer_calibration_profile is None
        else {
            "control_regularizer_calibration_profile": (
                cfg.control_regularizer_calibration_profile
            ),
            "executed_action_state": executed_action_state,
        }
    )
    unified_episode_kwargs = (
        {}
        if cfg.unified_mppi_effort_profile is None
        else {
            "unified_mppi_effort_profile": cfg.unified_mppi_effort_profile
        }
    )
    if cfg.pick_v8_causal_profile is not None:
        unified_episode_kwargs["pick_v8_causal_profile"] = (
            cfg.pick_v8_causal_profile
        )
    try:
        jres = run_mpc_episode(
            world=plan_world, plan_xml=plan_xml, program=program, anchors0=anchors0,
            obj_pose7=obj_pose7,
            sub_h=sub_H, band_floor=band_floor, rng=np.random.default_rng(cfg.seed),
            pool_size=cfg.pool_size,
            out_dir=out,
            subject=subject, objects=objects, reference=reference, episode=episode,
            max_stage_cycles=int(cfg.max_stage_cycles),
            mppi_cycle_budget_scope=cfg.mppi_cycle_budget_scope,
            horizon_steps=int(cfg.horizon_steps),
            num_knots=num_knots,
            sigma_fraction=sigma_fraction,
            sampler_mode=sampler_mode,
            mtp_jaw_mode=mtp_jaw_mode,
            local_sigma_fraction=local_sigma_fraction,
            cem_rounds=cem_rounds,
            cem_elite_fraction=cem_elite_fraction,
            cem_min_std_fraction=cem_min_std_fraction,
            optimizer=optimizer,
            mppi_temperature=mppi_temperature,
            control_profile=cfg.control_profile,
            **unified_episode_kwargs,
            arm_velocity_weight=arm_velocity_weight,
            record_candidate_cost_telemetry=bool(
                cfg.record_candidate_cost_telemetry
            ),
            table_contact_force_weight=cfg.table_contact_force_weight,
            joint_velocity_force_dial_arm=cfg.joint_velocity_force_dial_arm,
            joint_velocity_force_phase3_finalist_arm=(
                cfg.joint_velocity_force_phase3_finalist_arm
            ),
            joint_velocity_force_phase3_seed=(
                cfg.joint_velocity_force_phase3_seed
            ),
            joint_velocity_force_trial_budget=(
                cfg.joint_velocity_force_trial_budget
            ),
            cost_shaping_profile=cfg.cost_shaping_profile,
            **regularizer_episode_kwargs,
            execution_prefix_fraction=prefix_frac,
            mppi_stage_execution_prefix_steps=(
                cfg.mppi_stage_execution_prefix_steps
            ),
            execute_prefix_fn=execute_prefix_fn,
            reobserve_fn=reobserve_fn,
            solve_step_fn=remote_solve_step,
            execution_feasibility=execution_feasibility,
            observation_readiness=(
                observation_readiness if cfg.execute else None
            ),
            initial_observation_timestamp_s=(
                scene_obs0.get("capture_timestamp_s")
                if cfg.execute
                else None
            ),
            initial_robot_timestamp_s=(
                initial_robot_timestamp_s if cfg.execute else None
            ),
            initial_terminal_evidence={
                **dict(scene_obs0.get("evidence", {})),
                "anchor_reliability": _anchor_reliability(scene_obs0),
            },
            exact_plan_world_grounding=bool(cfg.offline),
            **candidate_selection_kwargs,
            **mppi_execution_kwargs,
        )
    except BaseException as exc:
        # Any failure after the program is fixed must leave a complete,
        # auditable packet. It may occur before the first GPU scorer is built,
        # after physics, or during a real prefix. Record every call site that
        # was never reached as explicitly NOT EVALUATED; absence must not be
        # confused with a hash/schema bug, and it must never authorize motion.
        from oap.program import REQUIRED_CALL_SITES, log_call_site

        existing_sites = {
            record.site for record in episode.call_site_records}
        failure_outcome = (
            f"CANNOT_VERIFY:EXECUTION_ABORTED:{type(exc).__name__}")
        for site in REQUIRED_CALL_SITES:
            if site not in existing_sites:
                log_call_site(episode, site, program, payload={
                    "evaluation_status": "not_evaluated_due_to_failure",
                    "failure_site": "run_mpc_episode",
                    "outcome": failure_outcome,
                    "reason": str(exc),
                    "exception_type": type(exc).__name__,
                })
        failure_verdict = {
            "outcome": "CANNOT_VERIFY",
            "tier": "refused",
            "provenance": "execution_failure",
            "reason": str(exc),
        }
        write_json_atomic(out / "verdict.json", failure_verdict)
        episode.record_verdict(
            failure_verdict,
            context={"exception_type": type(exc).__name__},
        )
        episode.finalize(success=False, outcome=failure_outcome)
        raise
    finally:
        try:
            if session is not None:
                result = locals().get("jres")
                held_close_width_m = getattr(
                    result, "held_close_width_m", None
                )
                held_status = getattr(result, "held_status", None)
                close_was_commanded = getattr(
                    result, "close_was_commanded", None
                )
                if cfg.manual_post_experiment_restore:
                    # Explicit experiment protocol: do not lift, open, move
                    # home, or infer a recovery trajectory. Stop and preserve
                    # the measured state for the operator's manual restore.
                    execute_mod.stop_and_record_manual_restore(
                        session,
                        out_dir=out,
                        held_close_width_m=held_close_width_m,
                        held_status=held_status,
                        close_was_commanded=close_was_commanded,
                    )
                elif result is not None:
                    # On exceptions the held state is unresolved: release the
                    # lease but never perform an automatic sweep/home.
                    execute_mod.go_home_safe_or_hold(
                        session,
                        held_close_width_m=held_close_width_m,
                        held_status=held_status,
                        close_was_commanded=bool(close_was_commanded),
                    )
        finally:
            try:
                if session is not None:
                    session.close()
            finally:
                # A failed real prefix must not strand either the robot lease
                # or a persistent FoundationPose worker.
                if continuous_tracker is not None:
                    continuous_tracker.stop()
                _call_optional(
                    camera_client,
                    ("stop", "close", "shutdown"),
                )
                for c in clients.values():
                    _call_optional(c, ("stop", "close", "shutdown"))
    tool_mediation_required = safety.program_uses_tool_mediation(program)
    terminal_success = bool(jres.terminal_success)
    mechanism_gated_success = unified_mppi_effort_requires_tool_mediation_success(
        cfg.unified_mppi_effort_profile
    )
    summary_success = profile_sealed_summary_success(
        terminal_success=terminal_success,
        result_success=jres.success,
        unified_mppi_effort_profile=cfg.unified_mppi_effort_profile,
    )
    episode.finalize(success=summary_success, outcome=jres.outcome)
    summary = {
        "schema": "oap_run_summary_v1",
        "task": cfg.task,
        "subject": subject.name,
        "reference": reference.name if reference else None,
        "role_source": "constraint_program",
        "program_sha256": program_sha,
        "program_provenance": program.provenance,
        "success": summary_success,
        "success_semantics": (
            "measured_stage_terminal_and_exact_tool_mediation"
            if mechanism_gated_success
            else "measured_stage_terminal"
        ),
        "terminal_success": terminal_success,
        "tool_mediation_required": tool_mediation_required,
        "mechanism_valid": jres.tool_mediation_valid,
        "tool_use_success": jres.tool_use_success,
        "mechanism_semantics": (
            "profile_sealed_episode_final_gate"
            if mechanism_gated_success
            else "diagnostic_running_constraint"
        ),
        "outcome": jres.outcome,               # verified, exhausted, unknown, or hard refusal
        "chunks": jres.chunks,                 # actual MPC cycles, globally indexed
        "executed_on_real": bool(cfg.execute),
        "execution_readiness": (
            "hardware-verified" if cfg.execute
            else "prepare-no-connection" if cfg.prepare_real_execution
            else "dry-run"),
        "bundle_manifest": str(cfg.bundle_manifest),
        "runtime_s": time.time() - started,
    }
    write_json_atomic(out / "summary.json", summary)
    logger.info("[summary] %s", summary)
    return summary
