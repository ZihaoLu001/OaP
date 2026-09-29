"""The joint-space MPC step + receding-horizon loop (M1-M5, minus real-robot M4).

The decision variable is a control-knot sequence; nothing here builds an
execution chunk. One MPC step:

    measured knot zero + future acknowledged hold -> one nominal (K x 8)
    sample_once  ->  one large predictive/random-shooting batch on the knots
                    (later nominal = shifted previous winner + last hold)
    (physics is IN the loop: exact-mesh MJWarp scores every stage; an
     unavailable GPU fails closed)
    selected MJWarp prefix ->  advance the dry-run plant on the same GPU rollout

The decision, the rollout, the feasibility check, and the sim execution are all
    joint. The one piece NOT here is the real-robot executor (stream the same joint
spline to flexiv joint control) -- that is M4, blocked on the lab robot PC + the
physically-present-with-e-stop safety gate, so it lands on the robot.
:func:`solve_mpc_step` is the reusable one-step unit; :func:`run_receding_horizon`
replans a stage from the shifted previous best
(:func:`~oap.twin.control_knots.shift_warm_start`).
"""
from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from oap.program.cost_shaping import validate_cost_shaping_profile
from oap.loop.sampling import (
    CANDIDATE_SELECTION_VALID_THEN_TOTAL_COST,
    SamplerStats,
    sample_once,
    validate_candidate_selection_mode,
)
from oap.loop.execution_prefix import (
    DEFAULT_EXECUTION_PREFIX_FRACTION,
    validate_execution_prefix_fraction,
)
from oap.loop.sim_grasp_evidence import (
    current_sim_closing_region_evidence,
    fuse_sim_held_evidence,
    fuse_contact_width_evidence,
    SIM_HELD_EVIDENCE_CONTACT_WIDTH_V1,
    resolve_grasp_bridge_telemetry_mode,
    resolve_sim_held_evidence_mode,
    should_record_canonical_grasp_evidence,
)
from oap.twin.batched_rollout import (
    DEFAULT_CEM_ELITE_FRACTION,
    DEFAULT_CEM_MIN_STD_FRACTION,
    DEFAULT_CEM_ROUNDS,
    DEFAULT_EXECUTION_STEPS,
    DEFAULT_ARM_VELOCITY_WEIGHT,
    DEFAULT_MPPI_TEMPERATURE,
    MPPI_EXECUTION_SOFTMAX_WEIGHTED_MEAN,
    MTP_JAW_MODE_PRESERVE_NOMINAL,
    PREDICTIVE_SAMPLER_SINGLE_SCALE,
    PREDICTIVE_SIGMA_FRACTION,
    PREDICTIVE_TWO_SCALE_LOCAL_SIGMA_FRACTION,
    validate_horizon_steps,
    validate_cem_elite_fraction,
    validate_cem_rounds,
    validate_local_sigma_fraction,
    validate_mtp_jaw_mode,
    validate_predictive_sampler_mode,
    validate_mppi_temperature,
    validate_mppi_execution_mode,
    validate_arm_velocity_weight,
)
from oap.twin.control_knots import ControlKnots
from oap.twin.control_profile import (
    joint_velocity_force_device_proposal_seed,
    validate_control_profile,
    validate_joint_velocity_force_phase3_finalist_arm,
    validate_joint_velocity_force_phase3_seed,
)

logger = logging.getLogger("oap.loop.mpc_loop")

__all__ = [
    "DEFAULT_EXECUTION_PREFIX_FRACTION",
    "MpcStepResult",
    "execution_prefix_fraction",
    "run_receding_horizon",
    "solve_mpc_step",
    "validate_execution_prefix_fraction",
]


def _persist_selected_gpu_step_trace(
    *,
    sampler_stats: SamplerStats,
    episode: Any,
    chunk_index: int,
    start_qpos: np.ndarray,
) -> dict[str, Any] | None:
    """Persist an enabled winner tape; otherwise preserve the normal path."""
    trace = getattr(sampler_stats, "selected_prefix_qpos_trace", None)
    if trace is None:
        return None
    if episode is None or not hasattr(episode, "out_dir"):
        raise RuntimeError(
            "GPU step tracing requires an EpisodeLog output directory"
        )
    endpoint = getattr(sampler_stats, "selected_prefix_state", None)
    if endpoint is None or not np.array_equal(
        np.asarray(trace)[-1],
        np.asarray(endpoint[0]),
    ):
        raise RuntimeError(
            "GPU step trace endpoint does not match the selected plant prefix"
        )
    dt = getattr(sampler_stats, "selected_prefix_step_dt_s", None)
    if dt is None:
        raise RuntimeError("GPU step trace omitted its physics timestep")
    batch = dict(getattr(sampler_stats, "batch", None) or {})
    selected_index = batch.get("selected_index")
    selected_round = batch.get("selected_round")
    from oap.loop.gpu_step_trace import write_gpu_step_trace

    return write_gpu_step_trace(
        episode_dir=episode.out_dir,
        chunk_index=chunk_index,
        qpos=np.asarray(trace),
        start_qpos=np.asarray(start_qpos),
        step_dt_s=float(dt),
        selected_candidate_index=(
            None if selected_index is None else int(selected_index)
        ),
        selected_round=(
            None if selected_round is None else int(selected_round)
        ),
    )


@dataclass
class MpcStepResult:
    """The outcome of one MPC step (optimize; the single output IS the plan)."""

    best: ControlKnots                    # lowest-cost feasible sampled plan
    sampler_stats: SamplerStats
    metadata: dict[int, dict[str, Any]]
    plan_valid: bool                     # final physics batch contained a valid plan


def solve_mpc_step(
    *, world: Any, plan_xml: Any, candidates: list[ControlKnots],
    metadata: dict[int, dict[str, Any]], obj_pose7: np.ndarray,
    rng: np.random.Generator, program: Any = None, stage_idx: int = 0,
    anchors: Any = None, cost_scale_anchors: Any = None,
    chunk_idx: int = 0, stage_cycle_idx: int = 0, sub_h: float = 0.0,
    episode: Any = None,
    start_state: tuple[np.ndarray, np.ndarray] | None = None,
    pool_size: int = 2048,
    horizon_steps: int = DEFAULT_EXECUTION_STEPS,
    sigma_fraction: float = PREDICTIVE_SIGMA_FRACTION,
    sampler_mode: str = PREDICTIVE_SAMPLER_SINGLE_SCALE,
    candidate_selection_mode: str = (
        CANDIDATE_SELECTION_VALID_THEN_TOTAL_COST
    ),
    mtp_jaw_mode: str = MTP_JAW_MODE_PRESERVE_NOMINAL,
    local_sigma_fraction: float = (
        PREDICTIVE_TWO_SCALE_LOCAL_SIGMA_FRACTION
    ),
    cem_rounds: int = DEFAULT_CEM_ROUNDS,
    cem_elite_fraction: float = DEFAULT_CEM_ELITE_FRACTION,
    cem_min_std_fraction: float = DEFAULT_CEM_MIN_STD_FRACTION,
    optimizer: str = "cem",
    mppi_temperature: float = DEFAULT_MPPI_TEMPERATURE,
    mppi_execution_mode: str = MPPI_EXECUTION_SOFTMAX_WEIGHTED_MEAN,
    control_profile: str | None = None,
    unified_mppi_effort_profile: str | None = None,
    pick_v8_causal_profile: str | None = None,
    pick_v8_causal_temperature_state: Any | None = None,
    mppi_halton_seed: int | None = None,
    mppi_nominal_actions: np.ndarray | None = None,
    arm_velocity_weight: float = DEFAULT_ARM_VELOCITY_WEIGHT,
    record_candidate_cost_telemetry: bool = False,
    table_contact_force_weight: float | None = None,
    control_regularizer_calibration_profile: str | None = None,
    joint_velocity_force_dial_arm: str | None = None,
    joint_velocity_force_phase3_finalist_arm: str | None = None,
    joint_velocity_force_phase3_seed: int | None = None,
    joint_velocity_force_trial_budget: str | None = None,
    cost_shaping_profile: str | None = None,
    previous_executed_action: np.ndarray | None = None,
    control_regularizer_boundary_valid: bool | None = None,
    execution_prefix_frac: float = DEFAULT_EXECUTION_PREFIX_FRACTION,
    movable_grounder: Any = None,
    execution_feasibility: Mapping[str, Any] | None = None,
    controlled_subject_contact_evidence_name: str | None = None,
) -> MpcStepResult:
    """One MPC step: optimize a fixed rollout budget; return the best valid plan.

    Takes the content-free joint-hold nominal from
    :func:`~oap.loop.plan.sample_candidates` and runs iterative GPU CEM
    (or the explicit one-round baseline). At later cycles the input nominal is the
    previous winner shifted by the executed horizon fraction with a last-hold
    tail.
    The optimizer's single output is executed as-is, with no post-hoc CPU
    re-ranking. Physics is in the loop: every stage uses the same exact-mesh
    GPU Warp backend, rolling the real gn01 linkage and contact-active arm
    meshes from the live state. An unavailable or failed GPU rollout leaves
    ``plan_valid=False`` and moves zero joints. Model error is handled by the
    closed loop (execute a prefix,
    re-measure, per-cycle stage check, bounded re-entry). No execution chunk
    is ever built; the dry-run adopts the selected rollout's exact MJWarp prefix
    state, while the robot path streams the same knots through flexiv joint
    control.
    """
    if not candidates:
        raise ValueError("solve_mpc_step: empty candidate pool")
    if int(stage_cycle_idx) < 0:
        raise ValueError("stage_cycle_idx must be non-negative")
    horizon_steps = validate_horizon_steps(horizon_steps)
    sigma_fraction = validate_local_sigma_fraction(
        sigma_fraction
    )
    sampler_mode = validate_predictive_sampler_mode(sampler_mode)
    candidate_selection_mode = validate_candidate_selection_mode(
        candidate_selection_mode
    )
    candidate_selection_kwargs = (
        {}
        if candidate_selection_mode
        == CANDIDATE_SELECTION_VALID_THEN_TOTAL_COST
        else {"candidate_selection_mode": candidate_selection_mode}
    )
    mtp_jaw_mode = validate_mtp_jaw_mode(mtp_jaw_mode)
    local_sigma_fraction = validate_local_sigma_fraction(
        local_sigma_fraction
    )
    cem_rounds = validate_cem_rounds(cem_rounds)
    cem_elite_fraction = validate_cem_elite_fraction(cem_elite_fraction)
    cem_min_std_fraction = validate_local_sigma_fraction(
        cem_min_std_fraction
    )
    optimizer = str(optimizer).strip().lower()
    if optimizer not in {"cem", "mppi"}:
        raise ValueError("optimizer must be 'cem' or 'mppi'")
    mppi_temperature = validate_mppi_temperature(mppi_temperature)
    mppi_execution_mode = validate_mppi_execution_mode(mppi_execution_mode)
    control_profile = validate_control_profile(control_profile, allow_none=True)
    cost_shaping_profile = validate_cost_shaping_profile(
        cost_shaping_profile,
        allow_none=True,
    )
    arm_velocity_weight = validate_arm_velocity_weight(arm_velocity_weight)
    if (
        optimizer != "mppi"
        and mppi_execution_mode != MPPI_EXECUTION_SOFTMAX_WEIGHTED_MEAN
    ):
        raise ValueError("best_valid_sample requires optimizer='mppi'")
    mppi_execution_kwargs = (
        {}
        if mppi_execution_mode == MPPI_EXECUTION_SOFTMAX_WEIGHTED_MEAN
        else {"mppi_execution_mode": mppi_execution_mode}
    )
    execution_prefix_frac = validate_execution_prefix_fraction(
        execution_prefix_frac
    )
    regularizer_sampling_kwargs = (
        {}
        if control_regularizer_calibration_profile is None
        else {
            "control_regularizer_calibration_profile": (
                control_regularizer_calibration_profile
            ),
            "previous_executed_action": previous_executed_action,
            "control_regularizer_boundary_valid": (
                control_regularizer_boundary_valid
            ),
        }
    )
    dial_sampling_kwargs = (
        {}
        if joint_velocity_force_dial_arm is None
        else {
            "joint_velocity_force_dial_arm": joint_velocity_force_dial_arm
        }
    )
    phase3_sampling_kwargs = (
        {}
        if joint_velocity_force_phase3_finalist_arm is None
        else {
            "joint_velocity_force_phase3_finalist_arm": (
                joint_velocity_force_phase3_finalist_arm
            ),
            "joint_velocity_force_phase3_seed": (
                joint_velocity_force_phase3_seed
            ),
        }
    )
    unified_sampling_kwargs = (
        {}
        if unified_mppi_effort_profile is None
        else {"unified_mppi_effort_profile": unified_mppi_effort_profile}
    )
    from oap.twin.unified_mppi_effort import (
        unified_mppi_effort_uses_proven_pick_carrier,
    )
    adaptive_unified = (
        unified_mppi_effort_profile is not None
        and unified_mppi_effort_uses_proven_pick_carrier(
            unified_mppi_effort_profile
        )
    )
    if pick_v8_causal_profile is not None:
        unified_sampling_kwargs["pick_v8_causal_profile"] = (
            pick_v8_causal_profile
        )
        unified_sampling_kwargs["pick_v8_causal_temperature_state"] = (
            pick_v8_causal_temperature_state
        )
    elif adaptive_unified:
        unified_sampling_kwargs["pick_v8_causal_temperature_state"] = (
            pick_v8_causal_temperature_state
        )
    # One optimizer and one exact-mesh GPU backend for reach, grasp, carry,
    # pour, and indirect tool interaction. There is no task-family route and no
    # CPU/no-GPU fallback. The production algorithm is exactly one shooting
    # batch; shift + last-hold supplies the next cycle's nominal.
    ranked, meta, stats = sample_once(
        world=world, plan_xml=plan_xml, candidates=candidates, metadata=metadata,
        obj_pose7=obj_pose7, rng=rng, program=program,
        stage_idx=stage_idx, anchors=anchors,
        cost_scale_anchors=cost_scale_anchors,
        chunk_idx=chunk_idx, sub_h=sub_h,
        start_state=start_state,
        episode=episode, pool_size=pool_size,
        horizon_steps=horizon_steps,
        sigma_fraction=sigma_fraction,
        sampler_mode=sampler_mode,
        mtp_jaw_mode=mtp_jaw_mode,
        local_sigma_fraction=local_sigma_fraction,
        cem_rounds=cem_rounds,
        cem_elite_fraction=cem_elite_fraction,
        cem_min_std_fraction=cem_min_std_fraction,
        optimizer=optimizer,
        mppi_temperature=mppi_temperature,
        control_profile=control_profile,
        **unified_sampling_kwargs,
        mppi_halton_seed=mppi_halton_seed,
        mppi_nominal_actions=mppi_nominal_actions,
        arm_velocity_weight=arm_velocity_weight,
        record_candidate_cost_telemetry=record_candidate_cost_telemetry,
        table_contact_force_weight=table_contact_force_weight,
        cost_shaping_profile=cost_shaping_profile,
        joint_velocity_force_trial_budget=joint_velocity_force_trial_budget,
        **regularizer_sampling_kwargs,
        **dial_sampling_kwargs,
        **phase3_sampling_kwargs,
        movable_grounder=movable_grounder,
        execution_prefix_frac=execution_prefix_frac,
        execution_feasibility=execution_feasibility,
        controlled_subject_contact_evidence_name=(
            controlled_subject_contact_evidence_name
        ),
        **candidate_selection_kwargs,
        **mppi_execution_kwargs,
    )
    # Single-plan contract: sample_once returns the best sample of its only
    # physics batch (or the seed when the backend fails closed).
    ordered = list(ranked)
    best = ordered[0]
    # Route the joint pipeline's evaluations through log_call_site so the episode
    # packet carries the same four-call-site audit trail the EE path does. The
    # PHYSICS gate records that the program's physics evaluation happened
    # inside the exact-mesh GPU optimization loop; the
    # separate post-hoc certifier was deleted. TRAJECTORY_COST (call site 1) is normally
    # logged inside sample_once. If the GPU path fails before evaluating a
    # batch, log the failed content-free nominal explicitly so the packet stays
    # conformant without pretending any plan was validated.
    if program is not None:
        from oap.program import CallSite, log_call_site
        idx = max(0, min(int(stage_idx), max(0, len(program.stages) - 1)))
        stage_name = program.stages[idx].name if program.stages else ""
        if not stats.ran:
            log_call_site(episode, CallSite.TRAJECTORY_COST, program, payload={
                "chunk_index": int(chunk_idx),
                "stage_index": idx, "stage_name": stage_name,
                "planner": "sampling_failed_nominal_retained",
                "pool": len(ordered),
                "failure_reason": stats.failure_reason,
            })
        # The payload must attest honestly: when the optimizer no-op'd
        # (exception-degraded to the unchanged pool)
        # NO physics touched this plan -- record that, never a false
        # "in_loop_rollout" (the audit packet is evidence, not decoration).
        log_call_site(episode, CallSite.PHYSICS_GATE, program, payload={
            "stage_index": idx, "stage_name": stage_name,
            "physics": ("in_loop_rollout" if stats.ran
                        else "content_free_sampling_failed"),
            "pool": (
                int(stats.pool_size) if stats.ran else len(ordered)
            ),
            "selected_payloads": len(ordered),
            "best_candidate": int(best.candidate_id),
            "initial_robot_scene_contact":
                stats.initial_robot_scene_contact,
        })
    logger.info("[mpc-step %d] best=%d", chunk_idx,
                int(best.candidate_id))
    plan_valid = (
        stats.ran
        and stats.final_n_valid is not None
        and int(stats.final_n_valid) > 0
        and stats.execution_plan_accepted is not False
    )
    return MpcStepResult(
        best=best,
        sampler_stats=stats,
        metadata=meta,
        plan_valid=bool(plan_valid),
    )


@dataclass
class RecedingHorizonCycle:
    """One receding-horizon cycle's outcome (the shift-init loop's record)."""

    cycle: int
    seeded: str                          # episode-entry source or "shift"
    step: MpcStepResult | None
    obj_pose7_after: np.ndarray          # the re-observed object pose after the prefix
    # Composed measured gate: explicit terminal predicates plus observable
    # endpoint evidence (for example ObjectHeld) and generic ejection safety.
    terminal_gate_status: bool | None = None
    execution_contact: dict[str, Any] | None = None
    execution_contact_cumulative: dict[str, Any] | None = None
    execution_valid: bool | None = True
    execution_tool_mediation_valid: bool | None = None
    execution_evidence: dict[str, Any] | None = None
    scene_observation: dict[str, Any] | None = None
    terminal_gate_evidence: Any = None
    executed: bool = True
    stop_reason: str | None = None


def execution_prefix_fraction(
    value: object = DEFAULT_EXECUTION_PREFIX_FRACTION,
) -> float:
    """Validate one explicit prefix value (compatibility helper)."""
    return validate_execution_prefix_fraction(value)


def _merge_execution_contact(
    cumulative: dict[str, Any] | None,
    current: dict[str, Any] | None,
    *,
    required_tool_target_contact: bool = False,
) -> dict[str, Any] | None:
    """Merge exact executed-contact evidence across cycles and stages.

    Robot-to-target contact remains a whole-episode invariant.  Positive
    subject-to-target evidence is also retained in a second map, but only for
    typed stages whose ``ToolMediation`` declaration explicitly requires that
    contact.  This prevents an accidental contact during acquisition from
    satisfying a later interaction stage.
    """
    if cumulative is None and current is None:
        return None
    total = {
        "subject_movable_frames": 0,
        "robot_movable_frames": 0,
        "max_subject_movable_pen_m": 0.0,
        "max_robot_movable_pen_m": 0.0,
    }
    for source in (cumulative or {}, current or {}):
        total["subject_movable_frames"] += int(
            source.get("subject_movable_frames", 0))
        total["robot_movable_frames"] += int(
            source.get("robot_movable_frames", 0))
        total["max_subject_movable_pen_m"] = max(
            float(total["max_subject_movable_pen_m"]),
            float(source.get("max_subject_movable_pen_m", 0.0)))
        total["max_robot_movable_pen_m"] = max(
            float(total["max_robot_movable_pen_m"]),
            float(source.get("max_robot_movable_pen_m", 0.0)))
    map_specs = (
        ("subject_target_frames_by_body", int, sum),
        ("robot_target_frames_by_body", int, sum),
        ("max_subject_target_pen_m_by_body", float, max),
        ("max_robot_target_pen_m_by_body", float, max),
    )
    for key, cast, reducer in map_specs:
        if not any(isinstance(source.get(key), dict)
                   for source in (cumulative or {}, current or {})):
            continue
        merged: dict[str, Any] = {}
        for source in (cumulative or {}, current or {}):
            values = source.get(key, {})
            if not isinstance(values, dict):
                continue
            for body, value in values.items():
                name = str(body)
                typed = cast(value)
                if name not in merged:
                    merged[name] = typed
                else:
                    merged[name] = reducer((merged[name], typed))
        total[key] = merged
    required_key = "required_subject_target_frames_by_body"
    prior_required = (cumulative or {}).get(required_key)
    explicit_current_required = (current or {}).get(required_key)
    current_required = (
        explicit_current_required
        if isinstance(explicit_current_required, dict)
        else (
            (current or {}).get("subject_target_frames_by_body")
            if required_tool_target_contact
            else None
        )
    )
    if isinstance(prior_required, dict) or isinstance(current_required, dict):
        required_merged: dict[str, int] = {}
        for values in (prior_required, current_required):
            if not isinstance(values, dict):
                continue
            for body, value in values.items():
                name = str(body)
                required_merged[name] = (
                    required_merged.get(name, 0) + int(value)
                )
        total[required_key] = required_merged
    target_names = tuple(dict.fromkeys(
        str(name)
        for source in (cumulative or {}, current or {})
        for name in source.get("mediation_target_bodies", ())
        if name
    ))
    if target_names:
        total["mediation_target_bodies"] = list(target_names)
    # These are state/recency observations, not additive counters.  Keep the
    # newest observation that actually reported each field; never turn missing
    # evidence into a measured zero (UNKNOWN must remain UNKNOWN).
    for key in (
        "subject_two_sided_frames",
        "subject_two_sided_recent",
        "subject_two_sided_now",
    ):
        for source in (cumulative or {}, current or {}):
            if key in source:
                total[key] = source[key]
    return total


def _hard_stop_reason(evaluation: Any) -> str | None:
    """Return an irreversible measured failure, independent of goal progress.

    Terminal conditions, including ``ObjectHeld``, may legitimately be false for
    many cycles and remain finite program costs. Ejection is a task-independent
    physical failure. Tool contact and holding are task semantics, not hidden
    execution vetoes.
    """
    if evaluation is None:
        return None
    if tuple(getattr(evaluation, "ejected", ()) or ()):
        return "ejected"
    return None


def _terminal_gate_evidence_payload(evaluation: Any) -> Any:
    """Return JSON-native terminal evidence without flattening typed detail."""
    if evaluation is None:
        return None
    to_dict = getattr(evaluation, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(evaluation, dict):
        return dict(evaluation)
    if isinstance(evaluation, (bool, int, float, str)):
        return evaluation
    # Retain the legacy lightweight test seam: pre-existing namespace-like
    # doubles carried only ``status`` and were deliberately not episode
    # evidence. Production StageEvaluation instances take the lossless branch
    # above; never stringify an unknown object into an apparently authoritative
    # packet.
    return None


def run_receding_horizon(
    *, world: Any, plan_xml: Any, seed_pool_fn: Any, obj_pose7: np.ndarray,
    rng: np.random.Generator, n_cycles: int,
    program: Any = None, stage_idx: int = 0, anchors: Any = None,
    cost_scale_anchors: Any = None,
    bounds: Any = None, pool_size: int = 2048,
    horizon_steps: int = DEFAULT_EXECUTION_STEPS,
    sigma_fraction: float = PREDICTIVE_SIGMA_FRACTION,
    sampler_mode: str = PREDICTIVE_SAMPLER_SINGLE_SCALE,
    candidate_selection_mode: str = (
        CANDIDATE_SELECTION_VALID_THEN_TOTAL_COST
    ),
    mtp_jaw_mode: str = MTP_JAW_MODE_PRESERVE_NOMINAL,
    local_sigma_fraction: float = (
        PREDICTIVE_TWO_SCALE_LOCAL_SIGMA_FRACTION
    ),
    cem_rounds: int = DEFAULT_CEM_ROUNDS,
    cem_elite_fraction: float = DEFAULT_CEM_ELITE_FRACTION,
    cem_min_std_fraction: float = DEFAULT_CEM_MIN_STD_FRACTION,
    optimizer: str = "cem",
    mppi_temperature: float = DEFAULT_MPPI_TEMPERATURE,
    mppi_execution_mode: str = MPPI_EXECUTION_SOFTMAX_WEIGHTED_MEAN,
    control_profile: str | None = None,
    unified_mppi_effort_profile: str | None = None,
    pick_v8_causal_profile: str | None = None,
    arm_velocity_weight: float = DEFAULT_ARM_VELOCITY_WEIGHT,
    record_candidate_cost_telemetry: bool = False,
    table_contact_force_weight: float | None = None,
    control_regularizer_calibration_profile: str | None = None,
    joint_velocity_force_dial_arm: str | None = None,
    joint_velocity_force_phase3_finalist_arm: str | None = None,
    joint_velocity_force_phase3_seed: int | None = None,
    joint_velocity_force_trial_budget: str | None = None,
    cost_shaping_profile: str | None = None,
    executed_action_state: Any = None,
    execution_prefix_frac: float = DEFAULT_EXECUTION_PREFIX_FRACTION,
    sub_h: float = 0.0, renderer: Any = None,
    out_dir: Any = None, chunk_idx0: int = 0,
    start_state: tuple[np.ndarray, np.ndarray] | None = None,
    episode: Any = None,
    terminal_gate_fn: Any = None,
    reground_anchors_fn: Any = None,
    movable_grounder: Any = None,
    execution_feasibility: Mapping[str, Any] | None = None,
    execution_contact_prior: dict[str, Any] | None = None,
    execute_prefix_fn: Any = None,
    reobserve_fn: Any = None,
    solve_step_fn: Any = None,
    observation_readiness: Any = None,
    initial_observation_timestamp_s: float | None = None,
    initial_robot_timestamp_s: float | None = None,
    initial_terminal_evidence: Mapping[str, Any] | None = None,
    controlled_subject_contact_evidence_name: str | None = None,
) -> list["RecedingHorizonCycle"]:
    """Standard receding-horizon MPC for ONE stage, in sim -- THE shift-init loop.

    This is the loop the audit found missing (criterion 5), and the first
    production caller of :func:`~oap.twin.control_knots.shift_warm_start`
    and the sampler's ``warm_start``. It follows the hydrax/mjpc/STORM contract:

      * the first cycle of each explicit stage obtains one content-free
        measured-boundary/future acknowledged hold through ``seed_pool_fn``;
      * every LATER cycle in that same stage SHIFTS the previous best
        (:func:`shift_warm_start` by exactly the executed horizon fraction) and
        hands it
        to the optimizer as the nominal -- no task-specific initialization,
        cross-stage trajectory inheritance, or driver-side
        noise copies (iteration-0 sampling lives inside the optimizer);
      * each cycle samples (:func:`solve_mpc_step`), advances the dry-run plant
        to the selected rollout's exact MJWarp state after that fraction,
        re-reads the object, and warm-starts the next cycle from the shifted plan.

    ``seed_pool_fn(obj_pose7) -> (list[ControlKnots], metadata)`` supplies the
    stage-entry nominal (the caller wires it to
    :func:`~oap.loop.plan.sample_candidates`). There is deliberately no
    external stage-entry warm-start input: every explicit Stage begins from
    the latest measured/controller hold.
    ``terminal_gate_fn(obj_pose7, cumulative_contact) -> StageEvaluation`` is
    the optional per-cycle termination criterion. It evaluates only the
    explicit Stage ``terminal`` predicates on the same post-prefix measured
    state. Running costs and tool/contact diagnostics cannot add a hidden
    completion condition. The driver consumes only its three-valued
    ``status``; true ends the stage immediately -- the Stage boundary is the
    condition being met, not the cycle budget.
    ``reground_anchors_fn(obj_pose7) -> AnchorSet`` optionally rebuilds the
    optimizer's cost references from the latest observation before every
    solve. Initial-scene anchors remain frozen inside that grounding function;
    current robot and movable-object anchors do not. Returns one
    :class:`RecedingHorizonCycle` per cycle (the ``seeded`` field records
    cold-start-vs-shift). No candidate is
    executed unless the optimizer's final physics batch contains at least one
    valid plan. The final action column is always a sampled binary-jaw latent,
    independent of stage predicates and measured held state. It is
    interpolated and thresholded per control step in both simulation and real
    execution; finger physics and every held measurement remain outcomes.

    ``solve_step_fn`` is the narrow remote-planner injection seam.  It must
    accept the same keyword arguments and return :class:`MpcStepResult`;
    perception, terminal evaluation, safety, re-observation, and prefix
    execution remain in this lab-side loop.  ``None`` selects the local
    :func:`solve_mpc_step`, preserving the existing production path.
    """
    from oap.twin.control_regularizer import (
        ExecutedActionState,
        validate_control_regularizer_calibration_profile,
    )

    regularizer_profile = validate_control_regularizer_calibration_profile(
        control_regularizer_calibration_profile,
        allow_none=True,
    )
    if regularizer_profile is not None and not isinstance(
        executed_action_state, ExecutedActionState
    ):
        raise ValueError(
            "active control regularizer requires episode ExecutedActionState"
        )
    from oap.twin.control_knots import (
        NJ,
        ControlBounds,
        ControlKnots,
        JawCommandState,
        JawCommandUnknown,
        action_knot_zero,
        jaw_command_state_from_native_control,
        shift_mppi_effort_warm_start,
        shift_warm_start,
    )

    import mujoco

    if bounds is None:
        bounds = ControlBounds.from_world(world)
    horizon_steps = validate_horizon_steps(horizon_steps)
    sigma_fraction = validate_local_sigma_fraction(
        sigma_fraction
    )
    sampler_mode = validate_predictive_sampler_mode(sampler_mode)
    candidate_selection_mode = validate_candidate_selection_mode(
        candidate_selection_mode
    )
    candidate_selection_kwargs = (
        {}
        if candidate_selection_mode
        == CANDIDATE_SELECTION_VALID_THEN_TOTAL_COST
        else {"candidate_selection_mode": candidate_selection_mode}
    )
    mtp_jaw_mode = validate_mtp_jaw_mode(mtp_jaw_mode)
    local_sigma_fraction = validate_local_sigma_fraction(
        local_sigma_fraction
    )
    cem_rounds = validate_cem_rounds(cem_rounds)
    cem_elite_fraction = validate_cem_elite_fraction(cem_elite_fraction)
    cem_min_std_fraction = validate_local_sigma_fraction(
        cem_min_std_fraction
    )
    optimizer = str(optimizer).strip().lower()
    if optimizer not in {"cem", "mppi"}:
        raise ValueError("optimizer must be 'cem' or 'mppi'")
    mppi_temperature = validate_mppi_temperature(mppi_temperature)
    pick_v8_causal_temperature_state = None
    from oap.twin.unified_mppi_effort import (
        unified_mppi_effort_uses_proven_pick_carrier,
    )
    adaptive_unified = (
        unified_mppi_effort_profile is not None
        and unified_mppi_effort_uses_proven_pick_carrier(
            unified_mppi_effort_profile
        )
    )
    mppi_execution_mode = validate_mppi_execution_mode(mppi_execution_mode)
    control_profile = validate_control_profile(control_profile, allow_none=True)
    cost_shaping_profile = validate_cost_shaping_profile(
        cost_shaping_profile,
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
    if (phase3_arm is None) != (phase3_seed is None):
        raise ValueError("Phase-3 finalist arm and seed must be supplied together")
    if phase3_arm is not None and joint_velocity_force_dial_arm is not None:
        raise ValueError("Phase-3 finalist conflicts with DIAL arm")
    arm_velocity_weight = validate_arm_velocity_weight(arm_velocity_weight)
    if (
        optimizer != "mppi"
        and mppi_execution_mode != MPPI_EXECUTION_SOFTMAX_WEIGHTED_MEAN
    ):
        raise ValueError("best_valid_sample requires optimizer='mppi'")
    mppi_execution_kwargs = (
        {}
        if mppi_execution_mode == MPPI_EXECUTION_SOFTMAX_WEIGHTED_MEAN
        else {"mppi_execution_mode": mppi_execution_mode}
    )
    execution_prefix_frac = validate_execution_prefix_fraction(
        execution_prefix_frac
    )
    sim_held_evidence_mode = resolve_sim_held_evidence_mode()
    grasp_bridge_telemetry_mode = resolve_grasp_bridge_telemetry_mode()
    m, d = world.model, world.data

    def _refresh_host_kinematics() -> None:
        """Update pose caches only; never run host dynamics or contact."""
        mujoco.mj_kinematics(m, d)

    oadr = int(world.object_qpos_addr)
    obj = np.asarray(obj_pose7, dtype=float).copy()
    warm_start: ControlKnots | None = None
    mppi_warm_start_actions: np.ndarray | None = None
    # The experiment registry resolves the actual device proposal key. The
    # unrelated published reference-effort Halton profile retains seed 42.
    registered_device_seed = joint_velocity_force_device_proposal_seed(
        trial_budget=joint_velocity_force_trial_budget,
        dial_arm=joint_velocity_force_dial_arm,
        phase3_arm=phase3_arm,
        phase3_seed=phase3_seed,
    )
    mppi_halton_seed = (
        42
        if optimizer == "mppi"
        and sampler_mode == "reference_effort_halton"
        else registered_device_seed
        if optimizer == "mppi"
        else None
    )
    out: list[RecedingHorizonCycle] = []
    nid = 1_000_000                      # id base for driver-created candidates
    # The PHYSICAL state carried across cycles. Standard receding-horizon MPC
    # advances the real system by the executed prefix and CONTINUES from there --
    # it does NOT teleport back to the plan's start each cycle (which flung the
    # object / never let the grasp progress). We snapshot (qpos, qvel) as the
    # physical execution state; GPU rollouts use independent device state.
    # ``start_state`` (the previous STAGE's physical state) makes a continuing stage
    # -- e.g. carry/lift while HELD -- start from the held state instead of
    # resetting the object onto the table (which would drop it).
    saved_state: tuple[np.ndarray, np.ndarray] = (
        (np.asarray(start_state[0]).copy(), np.asarray(start_state[1]).copy())
        if start_state is not None
        else (d.qpos.copy(), d.qvel.copy())
    )
    # ``runner`` initializes the host orchestration carrier in profile-native
    # units: legacy width metres or direct effort Newtons.  It is command
    # continuity only, never measured aperture and never a second physics
    # plant.  Decode that native command exactly once to the common latent.
    # CEM does not sample knot
    # zero, so this value is executed verbatim at the head of each prefix;
    # deriving it from the measured aperture instead put a spurious open
    # command at the start of all 30 carries in the 2026-08-07 pilot (a jaw
    # holding the eraser is obstructed at 44-67 mm and read as "open").
    # A world without a jaw actuator column (reduced unit-test twins) has no
    # command state to carry; knot zero is then never re-anchored either, and
    # ``action_knot_zero`` refuses if that ever changes.
    _entry_ctrl = np.asarray(d.ctrl, dtype=float).reshape(-1)
    from functools import partial

    native_jaw_state_factory = partial(
        jaw_command_state_from_native_control,
        control_profile=control_profile,
        pick_v8_causal_profile=pick_v8_causal_profile,
        unified_mppi_effort_profile=unified_mppi_effort_profile,
    )
    host_jaw_state_factory = native_jaw_state_factory
    acknowledged_jaw: JawCommandState | None = (
        host_jaw_state_factory(
            float(_entry_ctrl[NJ]), source="stage_entry_acknowledged_ctrl"
        )
        if _entry_ctrl.size > NJ
        else None
    )
    if episode is not None and acknowledged_jaw is not None:
        episode.record_event(
            "jaw_command_state",
            {
                **acknowledged_jaw.to_dict(),
                "stage_index": int(stage_idx),
                "cycle": None,
            },
        )
    cumulative_contact = (
        _merge_execution_contact(execution_contact_prior, None)
        if execute_prefix_fn is None or execution_contact_prior is not None
        else None
    )
    planning_observation_timestamp_s = initial_observation_timestamp_s
    planning_robot_timestamp_s = initial_robot_timestamp_s

    def _observation_timing(
        observation: dict[str, Any],
    ) -> tuple[dict[str, float], float, float]:
        from oap.loop.safety import require_observation_timing

        evidence = dict(observation.get("evidence", {}))
        observation_stamp = observation.get("capture_timestamp_s")
        robot_stamp = evidence.get("robot_state_timestamp_s")
        timing = require_observation_timing(
            observation_timestamp_s=observation_stamp,
            robot_timestamp_s=robot_stamp,
            limits=observation_readiness,
        )
        return timing, float(observation_stamp), float(robot_stamp)

    def _apply_scene_observation(
        observation: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Extract the measured subject pose/velocity from either seam schema."""
        if "subject_pose7" in observation:
            pose = np.asarray(observation["subject_pose7"], dtype=float)
            raw_velocity = observation.get("subject_velocity6")
            velocity = (
                None
                if raw_velocity is None
                else np.asarray(raw_velocity, dtype=float)
            )
            return pose, velocity
        subject_name = str(observation.get("subject_name", ""))
        body = dict(observation.get("bodies", {})).get(subject_name)
        if body is None:
            raise RuntimeError(
                "reobserve_fn must return subject_pose7 or "
                "subject_name + bodies[subject_name]"
            )
        pose = np.r_[body["pos_plan"], body["quat_wxyz"]].astype(float)
        linear = body.get("linear_velocity_mps")
        angular = body.get("angular_velocity_rps")
        velocity = (
            None
            if linear is None or angular is None
            else np.r_[linear, angular].astype(float)
        )
        return pose, velocity

    def _world_movable_state(*, velocity_observable: bool) -> dict[str, Any]:
        """Snapshot the planning twin without overclaiming real velocity."""
        rows: dict[str, Any] = {}
        qpos_addrs = {
            str(name): int(addr)
            for name, addr in (
                getattr(world, "movable_qpos_addr", {}) or {}
            ).items()
        }
        qpos_addrs.setdefault("subject", oadr)
        for name, qaddr in qpos_addrs.items():
            joint_ids = [
                jid
                for jid in range(m.njnt)
                if int(m.jnt_qposadr[jid]) == int(qaddr)
            ]
            velocity = None
            if joint_ids:
                dof = int(m.jnt_dofadr[joint_ids[0]])
                velocity = np.asarray(d.qvel[dof:dof + 6], dtype=float)
            rows[name] = {
                "pose7": np.asarray(
                    d.qpos[qaddr:qaddr + 7], dtype=float
                ).tolist(),
                "linear_velocity_mps": (
                    velocity[:3].tolist()
                    if velocity_observable and velocity is not None
                    else None
                ),
                "angular_velocity_rps": (
                    velocity[3:].tolist()
                    if velocity_observable and velocity is not None
                    else None
                ),
                "velocity_observable": bool(
                    velocity_observable and velocity is not None
                ),
                "source": (
                    "sim_qpos_qvel"
                    if velocity_observable
                    else "planning_twin_state"
                ),
            }
        return rows

    def _observed_movable_state(
        observation: dict[str, Any],
        measured_velocity: np.ndarray | None,
    ) -> dict[str, Any]:
        rows = {
            str(name): {
                "pose7": [*body["pos_plan"], *body["quat_wxyz"]],
                "linear_velocity_mps": body.get("linear_velocity_mps"),
                "angular_velocity_rps": body.get("angular_velocity_rps"),
                "velocity_observable": bool(
                    body.get("velocity_observable", False)
                ),
                "source": body.get("source"),
            }
            for name, body in observation.get("bodies", {}).items()
            if bool(body.get("movable", False))
        }
        if not rows and "subject_pose7" in observation:
            rows["subject"] = {
                "pose7": list(observation["subject_pose7"]),
                "linear_velocity_mps": (
                    None
                    if measured_velocity is None
                    else measured_velocity[:3].tolist()
                ),
                "angular_velocity_rps": (
                    None
                    if measured_velocity is None
                    else measured_velocity[3:].tolist()
                ),
                "velocity_observable": measured_velocity is not None,
                "source": observation.get(
                    "source", "external_observation_callback"
                ),
            }
        return rows

    def _call_terminal_gate(
        obj: np.ndarray,
        cumulative_contact: dict[str, Any],
        evidence: dict[str, Any],
    ) -> Any:
        """Call the evidence-aware seam, retaining the old two-arg test seam."""
        if terminal_gate_fn is None:
            return None
        import inspect

        try:
            parameters = tuple(
                inspect.signature(terminal_gate_fn).parameters.values()
            )
            positional = tuple(
                parameter
                for parameter in parameters
                if parameter.kind
                in (
                    parameter.POSITIONAL_ONLY,
                    parameter.POSITIONAL_OR_KEYWORD,
                )
            )
            variadic = any(
                parameter.kind is parameter.VAR_POSITIONAL
                for parameter in parameters
            )
        except (TypeError, ValueError):
            positional, variadic = (), True
        if variadic or len(positional) >= 3:
            return terminal_gate_fn(obj, cumulative_contact, evidence)
        return terminal_gate_fn(obj, cumulative_contact)

    if terminal_gate_fn is not None:
        entry_evidence = dict(initial_terminal_evidence or {})
        entry_evidence.setdefault("observation_purpose", "replan")
        entry_evidence.setdefault(
            "measured_movable_state_after",
            _world_movable_state(velocity_observable=True),
        )
        entry_evaluation = _call_terminal_gate(
            obj,
            cumulative_contact or {},
            entry_evidence,
        )
        entry_status = getattr(
            entry_evaluation,
            "status",
            entry_evaluation,
        )
        if entry_status is True:
            entry_contact = (
                None
                if cumulative_contact is None
                else dict(cumulative_contact)
            )
            if episode is not None:
                episode.record_chunk(int(chunk_idx0), {
                    "planner_elapsed_s": 0.0,
                    "cycle_elapsed_s": 0.0,
                    "execution_prefix_frac": 0.0,
                    "selected_rollout": None,
                    "executed": False,
                    "stop_reason": "measured_terminal_true_before_solve",
                    "execution_contact": None,
                    "execution_contact_cumulative": entry_contact,
                    "execution_valid": True,
                    "execution_tool_mediation_valid": None,
                    "execution_evidence": entry_evidence,
                    "scene_observation": None,
                    "terminal_gate_evidence": (
                        _terminal_gate_evidence_payload(entry_evaluation)
                    ),
                })
            return [RecedingHorizonCycle(
                cycle=int(chunk_idx0),
                seeded="measured_terminal",
                step=None,
                obj_pose7_after=obj.copy(),
                terminal_gate_status=True,
                execution_contact_cumulative=entry_contact,
                execution_valid=True,
                execution_evidence=entry_evidence,
                terminal_gate_evidence=entry_evaluation,
                executed=False,
                stop_reason=None,
            )]

    # Absorbing refusal-loop early termination (default OFF; behaviour is
    # byte-identical when the variable is unset/0). Measured basis: across
    # five absorbed episodes (REF_pick 222+, three abl5 slips 400-480, and
    # abl14_cup_gen02 683) not one of ~1800 consecutively refused cycles ever
    # recovered, while each absorption burned 2-4 GPU-hours. When the limit
    # is reached the loop breaks after appending the refusal chunk, so the
    # episode terminates with the SAME stop_reason/no_valid_plan path and the
    # SAME STAGE_NO_VALID_PLAN outcome as budget exhaustion under refusal.
    import os as _os
    _refusal_loop_limit = int(_os.environ.get(
        "OAP_REFUSAL_LOOP_LIMIT", "0") or 0)
    _consecutive_refusals = 0
    for cycle in range(int(n_cycles)):
        cycle_t0 = time.perf_counter()
        cycle_id = int(chunk_idx0) + cycle
        readiness_evidence: dict[str, Any] = {}
        if observation_readiness is not None and execute_prefix_fn is not None:
            from oap.loop.safety import require_observation_timing

            readiness_evidence.update(require_observation_timing(
                observation_timestamp_s=(
                    planning_observation_timestamp_s
                ),
                robot_timestamp_s=planning_robot_timestamp_s,
                limits=observation_readiness,
            ))
        if warm_start is None:
            pool, meta = seed_pool_fn(obj)  # ONE applied-control hold (stage entry)
            # Single-NOMINAL contract (advisor 2026-07-24, closing the loop on
            # the 2026-07-22 twelve-seed deletion): the driver hands the
            # optimizer exactly ONE nominal. The old inflate-to-pool_size at
            # 0.25*frac0 here was the twelve-seed jitter reborn one layer down
            # -- near-identical copies at a second, ad-hoc noise scale. Its
            # 0.25 calibration guarded the softmax MEAN, which PS retired.
            # CEM sampling and valid-elite refits happen inside sample_once.
            # The optional one-round baseline still launches one population.
            # The episode-entry nominal is the content-free current hold.
            # Preserve that provenance in the episode evidence.
            first_id = int(pool[0].candidate_id)
            seeded = str(meta.get(first_id, {}).get("source", "current_hold"))
            nominal_source = "current_hold"
        else:
            base = np.asarray(warm_start.knots, dtype=float)  # SHIFT: the nominal IS the shifted best
            pool = [ControlKnots(candidate_id=nid, knots=base.copy())]
            meta = {int(pool[0].candidate_id): {"source": "shift",
                                                "opt_update": "shift_init"}}
            nid += 1
            seeded = "shift"
            nominal_source = "shift_last_hold"
        # Evidence consumers must never infer provenance from knot values.
        # A stage-entry tape can be non-constant because knot zero is measured
        # while future knots hold the last acknowledged controller target.
        # Record the production branch that built this exact candidate zero.
        first_id = int(pool[0].candidate_id)
        meta.setdefault(first_id, {})["nominal_source"] = nominal_source

        # An acknowledged/shifted control is an optimizer warm start, not a
        # physical-state continuity claim.  For every solve, replace only knot
        # zero with the latest measured arm q and gripper width carried in the
        # physical state, saturated to the actuator command range.  The raw
        # measured state is still passed separately as ``start_state``; a
        # position-controlled joint can physically overshoot its command range,
        # but that overshoot must never be emitted as the next control target.
        # Future knots retain the shifted/last-hold nominal.
        robot_metadata = getattr(world, "robot", None)
        gripper_joint_ids = (
            robot_metadata.get("gripper_joint_ids")
            if isinstance(robot_metadata, dict)
            else None
        )
        has_measured_action_mapping = (
            isinstance(robot_metadata, dict)
            and "qpos_addr" in robot_metadata
            and isinstance(gripper_joint_ids, dict)
            and "finger_width" in gripper_joint_ids
            and hasattr(m, "jnt_qposadr")
        )
        if has_measured_action_mapping:
            measured_start = action_knot_zero(
                world,
                acknowledged_jaw=acknowledged_jaw,
                qpos=saved_state[0],
            )
            anchored_pool: list[ControlKnots] = []
            for candidate in pool:
                candidate_knots = np.asarray(
                    candidate.knots,
                    dtype=float,
                ).copy()
                if (
                    candidate_knots.ndim != 2
                    or candidate_knots.shape[0] < 2
                    or candidate_knots.shape[1] != len(measured_start)
                ):
                    raise ValueError(
                        "every solve nominal must be a (K, 8) knot array "
                        "with K >= 2"
                    )
                candidate_knots[0] = bounds.clip(
                    measured_start[None, :],
                )[0]
                anchored_pool.append(ControlKnots(
                    candidate_id=int(candidate.candidate_id),
                    knots=candidate_knots,
                ))
            pool = anchored_pool
        cycle_prefix_frac = execution_prefix_frac
        knot_count = int(np.asarray(pool[0].knots).shape[0])
        if knot_count < 2 or any(
            int(np.asarray(candidate.knots).shape[0]) != knot_count
            for candidate in pool[1:]
        ):
            raise ValueError("all candidates in one sampling batch need the same K")
        cycle_anchors = (
            reground_anchors_fn(obj)
            if reground_anchors_fn is not None
            else anchors
        )
        step_solver = solve_mpc_step if solve_step_fn is None else solve_step_fn
        regularizer_solve_kwargs = (
            {}
            if regularizer_profile is None
            else {
                "control_regularizer_calibration_profile": regularizer_profile,
                "previous_executed_action": (
                    executed_action_state.action.copy()
                ),
                "control_regularizer_boundary_valid": (
                    executed_action_state.boundary_valid
                ),
            }
        )
        dial_solve_kwargs = (
            {}
            if joint_velocity_force_dial_arm is None
            else {
                "joint_velocity_force_dial_arm": (
                    joint_velocity_force_dial_arm
                )
            }
        )
        phase3_solve_kwargs = (
            {}
            if phase3_arm is None
            else {
                "joint_velocity_force_phase3_finalist_arm": phase3_arm,
                "joint_velocity_force_phase3_seed": phase3_seed,
            }
        )
        unified_solve_kwargs = (
            {}
            if unified_mppi_effort_profile is None
            else {"unified_mppi_effort_profile": unified_mppi_effort_profile}
        )
        if pick_v8_causal_profile is not None:
            unified_solve_kwargs["pick_v8_causal_profile"] = (
                pick_v8_causal_profile
            )
            unified_solve_kwargs["pick_v8_causal_temperature_state"] = (
                pick_v8_causal_temperature_state
            )
        elif adaptive_unified:
            unified_solve_kwargs["pick_v8_causal_temperature_state"] = (
                pick_v8_causal_temperature_state
            )
        res = step_solver(
            world=world, plan_xml=plan_xml, candidates=pool, metadata=meta,
            obj_pose7=obj, rng=rng, program=program, stage_idx=stage_idx,
            anchors=cycle_anchors,
            cost_scale_anchors=cost_scale_anchors,
            chunk_idx=cycle_id, stage_cycle_idx=cycle, sub_h=sub_h,
            start_state=saved_state,
            episode=episode,
            pool_size=int(pool_size), horizon_steps=horizon_steps,
            sigma_fraction=sigma_fraction,
            sampler_mode=sampler_mode,
            mtp_jaw_mode=mtp_jaw_mode,
            local_sigma_fraction=local_sigma_fraction,
            cem_rounds=cem_rounds,
            cem_elite_fraction=cem_elite_fraction,
            cem_min_std_fraction=cem_min_std_fraction,
            optimizer=optimizer,
            mppi_temperature=mppi_temperature,
            control_profile=control_profile,
            **unified_solve_kwargs,
            mppi_halton_seed=mppi_halton_seed,
            mppi_nominal_actions=mppi_warm_start_actions,
            arm_velocity_weight=arm_velocity_weight,
            record_candidate_cost_telemetry=(
                record_candidate_cost_telemetry
            ),
            table_contact_force_weight=table_contact_force_weight,
            cost_shaping_profile=cost_shaping_profile,
            joint_velocity_force_trial_budget=(
                joint_velocity_force_trial_budget
            ),
            **regularizer_solve_kwargs,
            **dial_solve_kwargs,
            **phase3_solve_kwargs,
            execution_prefix_frac=cycle_prefix_frac,
            movable_grounder=movable_grounder,
            execution_feasibility=execution_feasibility,
            controlled_subject_contact_evidence_name=(
                controlled_subject_contact_evidence_name
            ),
            **candidate_selection_kwargs,
            **mppi_execution_kwargs,
        )
        sampler_stats = getattr(res, "sampler_stats", None)
        legacy_temperature_update = (
            optimizer == "mppi"
            and sampler_stats is not None
            and unified_mppi_effort_profile is None
        )
        if legacy_temperature_update or (
            optimizer == "mppi"
            and sampler_stats is not None
            and adaptive_unified
        ):
            sampler_batch = getattr(sampler_stats, "batch", None)
            if isinstance(sampler_batch, dict):
                proposed_temperature = sampler_batch.get(
                    "mppi_next_temperature"
                )
                if proposed_temperature is not None:
                    if pick_v8_causal_profile is not None or adaptive_unified:
                        from oap.twin.pick_v8_causal import (
                            advance_pick_v8_causal_mppi_temperature,
                        )

                        pick_v8_causal_temperature_state = (
                            advance_pick_v8_causal_mppi_temperature(
                                previous_state=(
                                    pick_v8_causal_temperature_state
                                ),
                                current_temperature=mppi_temperature,
                                effective_samples=sampler_batch.get(
                                    "mppi_effective_samples"
                                ),
                                reported_next_temperature=(
                                    proposed_temperature
                                ),
                            )
                        )
                    mppi_temperature = validate_mppi_temperature(
                        proposed_temperature
                    )
        planner_elapsed_s = float(
            getattr(sampler_stats, "elapsed_s", 0.0)
            or 0.0
        )
        selected_rollout = getattr(
            sampler_stats,
            "selected_rollout_diagnostics",
            None,
        )
        # Retire the host contact pre-check: its ``mj_forward`` used a second
        # physics backend before planning.  Instead, the exact MJWarp rollout
        # attests robot/scene contact at its first post-step state.  Apply that
        # integrity gate only to the episode's global first chunk, before any
        # prefix can move.  Later contact is manipulation, not initialization.
        initial_contact = getattr(
            sampler_stats,
            "initial_robot_scene_contact",
            None,
        )
        sampling_failure_reason = getattr(
            sampler_stats,
            "failure_reason",
            None,
        )
        refusal_reason: str | None = (
            str(sampling_failure_reason)
            if sampling_failure_reason
            else None
        )
        refusal_source = (
            "sampling_backend"
            if refusal_reason is not None
            else "mjwarp_first_post_step"
        )
        if refusal_reason is None and cycle_id == 0:
            if initial_contact is None:
                refusal_reason = (
                    "initial_robot_scene_contact_unavailable"
                )
            elif bool(initial_contact):
                refusal_reason = "initial_robot_scene_contact"
        if refusal_reason is not None:
            d.qpos[:] = saved_state[0]
            d.qvel[:] = saved_state[1]
            _refresh_host_kinematics()
            obj = np.asarray(
                saved_state[0][oadr:oadr + 7], dtype=float
            ).copy()
            refusal_evidence = {
                "schema": "oap_execution_refusal_v1",
                "reason": refusal_reason,
                "source": refusal_source,
                "initial_robot_scene_contact": initial_contact,
                "measured_movable_state_before": _world_movable_state(
                    velocity_observable=execute_prefix_fn is None
                ),
            }
            if episode is not None:
                episode.record_chunk(cycle_id, {
                    "planner_elapsed_s": planner_elapsed_s,
                    "cycle_elapsed_s": time.perf_counter() - cycle_t0,
                    "execution_prefix_frac": float(cycle_prefix_frac),
                    "selected_rollout": selected_rollout,
                    "executed": False,
                    "stop_reason": refusal_reason,
                    "execution_contact": (
                        {} if execute_prefix_fn is None else None
                    ),
                    "execution_contact_cumulative": (
                        None
                        if cumulative_contact is None
                        else dict(cumulative_contact)
                    ),
                    "execution_valid": False,
                    "execution_tool_mediation_valid": None,
                    "execution_evidence": refusal_evidence,
                    "scene_observation": None,
                    "terminal_gate_evidence": None,
                })
            out.append(RecedingHorizonCycle(
                cycle=cycle_id,
                seeded=seeded,
                step=res,
                obj_pose7_after=obj.copy(),
                terminal_gate_status=None,
                execution_contact=(
                    {} if execute_prefix_fn is None else None
                ),
                execution_contact_cumulative=(
                    None
                    if cumulative_contact is None
                    else dict(cumulative_contact)
                ),
                execution_valid=False,
                execution_evidence=refusal_evidence,
                scene_observation=None,
                terminal_gate_evidence=None,
                executed=False,
                stop_reason=refusal_reason,
            ))
            logger.error(
                "[recede stage %d cycle %d] %s -- execution refused; "
                "state unchanged",
                stage_idx,
                cycle_id,
                refusal_reason,
            )
            break
        # A ranked "least invalid" sample is not permission to move. Restore
        # the exact pre-optimization physical state and record the refused
        # cycle. The next bounded MPC cycle may safely draw a fresh batch from
        # this unchanged state; a transient all-invalid GPU reduction is not an
        # irreversible physical failure.
        if not bool(getattr(res, "plan_valid", False)):
            restore = saved_state
            d.qpos[:] = restore[0]
            d.qvel[:] = restore[1]
            _refresh_host_kinematics()
            saved_state = (d.qpos.copy(), d.qvel.copy())
            obj = np.asarray(saved_state[0][oadr:oadr + 7], dtype=float).copy()
            # The rejected rollout is not an acknowledged command, so it never
            # replaces the nominal. Preserve the last *executed and shifted*
            # winner instead. Resetting to current-hold after a transient
            # all-invalid batch destroys the standard receding-horizon prior;
            # for an already held object that hold can itself predict a drop,
            # making every later batch invalid. Resampling around the last
            # acknowledged shift is task-independent and matches the usual
            # shift-plus-last-hold MPC contract.
            if episode is not None:
                refusal_evidence = {
                    "schema": "oap_execution_refusal_v1",
                    "reason": "no_valid_plan",
                    "measured_movable_state_before": _world_movable_state(
                        velocity_observable=execute_prefix_fn is None
                    ),
                }
                episode.record_chunk(cycle_id, {
                    "planner_elapsed_s": planner_elapsed_s,
                    "cycle_elapsed_s": time.perf_counter() - cycle_t0,
                    "execution_prefix_frac": float(cycle_prefix_frac),
                    "selected_rollout": selected_rollout,
                    "executed": False,
                    "stop_reason": "no_valid_plan",
                    "execution_contact": (
                        {} if execute_prefix_fn is None else None
                    ),
                    "execution_contact_cumulative": (
                        None
                        if cumulative_contact is None
                        else dict(cumulative_contact)
                    ),
                    "execution_valid": False,
                    "execution_tool_mediation_valid": None,
                    "execution_evidence": refusal_evidence,
                    "scene_observation": None,
                    "terminal_gate_evidence": None,
                })
            out.append(RecedingHorizonCycle(
                cycle=cycle_id,
                seeded=seeded,
                step=res,
                obj_pose7_after=obj.copy(),
                terminal_gate_status=None,
                execution_contact=(
                    {} if execute_prefix_fn is None else None
                ),
                execution_contact_cumulative=(
                    None
                    if cumulative_contact is None
                    else dict(cumulative_contact)
                ),
                execution_valid=False,
                execution_evidence={
                    "schema": "oap_execution_refusal_v1",
                    "reason": "no_valid_plan",
                    "measured_movable_state_before": _world_movable_state(
                        velocity_observable=execute_prefix_fn is None
                    ),
                },
                scene_observation=None,
                terminal_gate_evidence=None,
                executed=False,
                stop_reason="no_valid_plan",
            ))
            logger.warning(
                "[recede stage %d cycle %d] no valid physics rollout -- "
                "execution refused; state unchanged; retaining the last "
                "acknowledged shifted nominal%s",
                stage_idx,
                cycle_id,
                (
                    ", resampling within the stage budget"
                    if cycle + 1 < int(n_cycles)
                    else ", cycle budget exhausted"
                ),
            )
            _consecutive_refusals += 1
            if (_refusal_loop_limit > 0
                    and _consecutive_refusals >= _refusal_loop_limit):
                logger.warning(
                    "[recede stage %d cycle %d] %d consecutive refused "
                    "cycles reached OAP_REFUSAL_LOOP_LIMIT=%d -- "
                    "terminating the stage early on the no_valid_plan path "
                    "(recovery from an absorbed refusal loop has never been "
                    "observed)",
                    stage_idx, cycle_id,
                    _consecutive_refusals, _refusal_loop_limit,
                )
                break
            continue
        _consecutive_refusals = 0
        if execute_prefix_fn is None and sampler_stats is not None:
            trace_metadata = _persist_selected_gpu_step_trace(
                sampler_stats=sampler_stats,
                episode=episode,
                chunk_index=cycle_id,
                start_qpos=saved_state[0],
            )
            if trace_metadata is not None:
                selected_rollout = dict(selected_rollout or {})
                selected_rollout["gpu_step_trace"] = trace_metadata
        if observation_readiness is not None and execute_prefix_fn is not None:
            from oap.loop.safety import require_plan_freshness

            readiness_evidence["plan_staleness_s"] = (
                require_plan_freshness(
                    planning_observation_timestamp_s=(
                        planning_observation_timestamp_s
                    ),
                    limits=observation_readiness,
                )
            )
        # Execute the plan's PREFIX and re-observe (the "execute prefix, re-observe"
        # half of MPC) -- textbook: the optimizer's output streams as-is, no
        # post-hoc gate (closing/held stages already optimized on the very
        # physics the deleted certifier used; errors are the closed loop's job:
        # re-measure, per-cycle stage check, bounded continuation). Stage entry
        # SEEDS the plan's start; later cycles RESTORE the persisted physical
        # state and CONTINUE (the optimizer's rollouts reset world.data).
        cycle_contact: dict[str, Any] | None = (
            {} if execute_prefix_fn is None else None
        )
        execution_record: dict[str, Any] = {}
        scene_observation: dict[str, Any] | None = None
        measured_velocity: np.ndarray | None = None
        if execute_prefix_fn is None:
            # The selected candidate was already integrated by the exact
            # MJWarp backend.  Its executed-fraction endpoint is the dry-run
            # plant state: adopting it avoids a second, CPU-MuJoCo simulator
            # with different solver/contact semantics.  Missing endpoint
            # evidence is a failed-closed execution error, never permission to
            # replay the knots on CPU.
            sampler_stats = getattr(res, "sampler_stats", None)
            prefix_state = getattr(
                sampler_stats,
                "selected_prefix_state",
                None,
            )
            prefix_control = getattr(
                sampler_stats,
                "selected_prefix_control",
                None,
            )
            prefix_contact = getattr(
                sampler_stats,
                "selected_prefix_contact",
                None,
            )
            if (
                prefix_state is None
                or prefix_control is None
                or not isinstance(prefix_contact, dict)
            ):
                raise RuntimeError(
                    "valid GPU plan omitted selected-prefix state/control/"
                    "contact; refusing CPU simulation fallback"
                )
            saved_state = (
                np.asarray(prefix_state[0], dtype=float).copy(),
                np.asarray(prefix_state[1], dtype=float).copy(),
            )
            if saved_state[0].shape != d.qpos.shape:
                raise RuntimeError(
                    "selected GPU prefix qpos shape does not match the "
                    f"planning twin: {saved_state[0].shape} != {d.qpos.shape}"
                )
            if saved_state[1].shape != d.qvel.shape:
                raise RuntimeError(
                    "selected GPU prefix qvel shape does not match the "
                    f"planning twin: {saved_state[1].shape} != {d.qvel.shape}"
                )
            prefix_control = np.asarray(prefix_control, dtype=float)
            if prefix_control.shape != d.ctrl.shape:
                raise RuntimeError(
                    "selected GPU prefix ctrl shape does not match the "
                    f"planning twin: {prefix_control.shape} != {d.ctrl.shape}"
                )
            d.qpos[:] = saved_state[0]
            d.qvel[:] = saved_state[1]
            d.ctrl[:] = prefix_control
            # The jaw command actually applied at the prefix endpoint. This is
            # the scored winner's own decoded command, so the jaw carried into
            # the next solve is exactly the jaw that was executed.
            _executed_ctrl = np.asarray(
                prefix_control, dtype=float
            ).reshape(-1)
            if _executed_ctrl.size > NJ:
                selected_diagnostics = getattr(
                    sampler_stats, "selected_rollout_diagnostics", None
                ) or {}
                executed_jaw_command = selected_diagnostics.get(
                    "prefix_gripper_actuator_command"
                    if control_profile is not None
                    else "prefix_gripper_command"
                )
                if executed_jaw_command is None:
                    raise JawCommandUnknown(
                        "MPPI omitted its exact native prefix jaw actuator "
                        "command"
                    )
                acknowledged_jaw = native_jaw_state_factory(
                    float(executed_jaw_command),
                    source="mjwarp_executed_prefix_endpoint",
                )
            # Refresh only body/geom/site pose caches.  ``mj_kinematics`` does
            # not integrate, solve constraints, or compute contacts/dynamics;
            # all plant evolution and execution evidence came from MJWarp.
            _refresh_host_kinematics()
            cycle_contact = dict(prefix_contact)
            obj = np.asarray(
                saved_state[0][oadr:oadr + 7], dtype=float
            ).copy()
            joint_ids = [
                jid
                for jid in range(m.njnt)
                if int(m.jnt_qposadr[jid]) == oadr
            ]
            if joint_ids:
                dof = int(m.jnt_dofadr[joint_ids[0]])
                measured_velocity = np.asarray(
                    saved_state[1][dof:dof + 6], dtype=float
                )
            robot_spec = getattr(world, "robot", {}) or {}
            width_jid = (
                robot_spec.get("gripper_joint_ids", {})
                .get("finger_width")
            )
            measured_width = None
            measured_held = None
            grasp_observable = False
            if width_jid is not None:
                width_qaddr = int(m.jnt_qposadr[int(width_jid)])
                measured_width = float(d.qpos[width_qaddr])
                measured_held = bool(
                    cycle_contact.get("subject_two_sided_now", False)
                )
                grasp_observable = True
            sim_held_evidence = None
            if should_record_canonical_grasp_evidence(
                held_evidence_mode=sim_held_evidence_mode,
                telemetry_mode=grasp_bridge_telemetry_mode,
            ):
                bilateral_contact = cycle_contact.get("subject_two_sided_now")
                if sim_held_evidence_mode == SIM_HELD_EVIDENCE_CONTACT_WIDTH_V1:
                    sim_held_evidence = fuse_contact_width_evidence(
                        bilateral_contact=bilateral_contact,
                        measured_width_m=measured_width,
                    )
                    measured_held = sim_held_evidence["subject_held"]
                    grasp_observable = sim_held_evidence["observable"]
                else:
                    spatial_evidence = current_sim_closing_region_evidence(world)
                    sim_held_evidence = fuse_sim_held_evidence(
                        bilateral_contact=(
                            bilateral_contact
                            if isinstance(bilateral_contact, (bool, np.bool_))
                            else None
                        ),
                        measured_width_m=measured_width,
                        closing_region_observable=bool(
                            spatial_evidence.get("observable", False)
                        ),
                        subject_intersects_closing_region=(
                            spatial_evidence.get("intersects_closing_region")
                        ),
                    )
                    sim_held_evidence["spatial_evidence"] = spatial_evidence
                sim_held_evidence["recording_mode"] = (
                    grasp_bridge_telemetry_mode
                )
            execution_record = {
                "schema": "oap_gpu_sim_execution_evidence_v1",
                "execution_source": "mjwarp_selected_prefix",
                **(
                    {"executed_prefix_sample_id": int(cycle_id)}
                    if sim_held_evidence is not None
                    else {}
                ),
                "contact_measurement": dict(cycle_contact or {}),
                "tool_mediation_observable": True,
                "tool_mediation_source": "mjwarp_selected_prefix_contacts",
                "tool_mediation_coverage":
                    "every_mjwarp_step_in_executed_prefix",
                "grasp_evidence_observable": grasp_observable,
                # Frozen mode retains its raw contact channel. contact_width_v1
                # uses the same qualified held value for costs and stage gates.
                "subject_held": measured_held,
                "gripper_width_m": measured_width,
                **(
                    {"sim_held_evidence": sim_held_evidence}
                    if sim_held_evidence is not None
                    else {}
                ),
            }
        else:
            if reobserve_fn is None:
                raise ValueError(
                    "real execute_prefix_fn requires reobserve_fn after "
                    "every prefix"
                )
            d.qpos[:] = saved_state[0]
            d.qvel[:] = saved_state[1]
            _refresh_host_kinematics()
            execution_record = dict(
                execute_prefix_fn(
                    knots=res.best.knots,
                    action_plan=(
                        None
                        if sampler_stats is None
                        else getattr(
                            sampler_stats,
                            "mppi_selected_action_plan",
                            None,
                        )
                    ),
                    prefix_frac=cycle_prefix_frac,
                    chunk_idx=cycle_id,
                )
            )
            if (
                execution_record.get("success") is False
                or bool(execution_record.get("clipped", False))
            ):
                from oap.loop.safety import SafetyError

                failure_evidence = dict(execution_record)
                failure_evidence["measured_movable_state_before"] = (
                    _world_movable_state(velocity_observable=False)
                )
                if episode is not None:
                    episode.record_chunk(cycle_id, {
                        "planner_elapsed_s": planner_elapsed_s,
                        "cycle_elapsed_s": time.perf_counter() - cycle_t0,
                        "execution_prefix_frac": float(cycle_prefix_frac),
                        "selected_rollout": selected_rollout,
                        "executed": True,
                        "stop_reason": "segment_failed_or_clipped",
                        "execution_contact": None,
                        "execution_contact_cumulative": (
                            None
                            if cumulative_contact is None
                            else dict(cumulative_contact)
                        ),
                        "execution_valid": False,
                        "execution_tool_mediation_valid": None,
                        "execution_evidence": failure_evidence,
                        "scene_observation": None,
                        "terminal_gate_evidence": None,
                    })
                raise SafetyError(
                    "real joint prefix failed/clipped; refusing "
                    "re-observation, warm start, gripper transition, and "
                    "all later commands"
                )
            if observation_readiness is not None:
                from oap.loop.safety import (
                    require_final_tracking_error,
                )

                readiness_evidence[
                    "final_joint_tracking_error_rad"
                ] = require_final_tracking_error(
                    final_joint_tracking_error_rad=execution_record.get(
                        "final_joint_tracking_error_rad"
                    ),
                    limits=observation_readiness,
                )
            scene_observation = dict(
                reobserve_fn(
                    purpose="replan",
                    chunk_idx=cycle_id,
                    execution_record=execution_record,
                )
            )
            if observation_readiness is not None:
                (
                    observation_timing,
                    planning_observation_timestamp_s,
                    planning_robot_timestamp_s,
                ) = _observation_timing(scene_observation)
                readiness_evidence.update(observation_timing)
            obj, measured_velocity = _apply_scene_observation(
                scene_observation
            )
            # The callback resynchronizes robot q and hot-patches every measured
            # movable body into this same planning twin.
            saved_state = (d.qpos.copy(), d.qvel.copy())
            # Real lane: the controller's own acknowledged ending gripper
            # target is the command state. It must be present -- an executed
            # prefix whose jaw command cannot be confirmed leaves the next
            # solve's action boundary unknown, and guessing it is exactly the
            # defect this path exists to prevent.
            # Both lanes carry the command state in the SAME place: the plan
            # world's acknowledged actuator vector, which the re-observation
            # callback synchronizes from the controller's atomic target.
            # ``sync_robot_state_to_twin`` refuses to seed that jaw column
            # from a measured aperture, so this read is a command by
            # construction; a value at neither level means the command state
            # was lost and planning stops.
            _post_ctrl = np.asarray(d.ctrl, dtype=float).reshape(-1)
            if _post_ctrl.size > NJ:
                acknowledged_jaw = host_jaw_state_factory(
                    float(_post_ctrl[NJ]),
                    source="real_reobserved_acknowledged_ctrl",
                    acknowledged_at_s=execution_record.get(
                        "motion_end_unix_s"
                    ),
                )
            elif acknowledged_jaw is not None:
                # The twin had a jaw command column at stage entry and lost
                # it: the command channel is broken, not absent by design.
                raise JawCommandUnknown(
                    "the twin no longer exposes a jaw command column after a "
                    "real prefix; refusing to plan the next cycle"
                )
            # Cross-check against the server's own atomic acknowledgement
            # when the executor reports it: the jaw that was scored must be
            # the jaw that was executed, with no post-hoc substitution.
            reported_target = execution_record.get(
                "last_acknowledged_controller_target"
            )
            if reported_target is not None:
                reported = np.asarray(reported_target, dtype=float).reshape(-1)
                if reported.size > NJ and not np.isclose(
                    float(reported[NJ]),
                    float(acknowledged_jaw.commanded_effort_n),
                    atol=1e-6,
                ):
                    raise JawCommandUnknown(
                        "controller-acknowledged jaw target "
                        f"{float(reported[NJ]):.6f} N disagrees with the "
                        f"twin command column "
                        f"{float(acknowledged_jaw.commanded_effort_n):.6f} N"
                    )
            contact = execution_record.get("contact_measurement")
            if (
                bool(execution_record.get(
                    "tool_mediation_observable", False
                ))
                and isinstance(contact, dict)
            ):
                cycle_contact = dict(contact)

        contact_observable = bool(
            execution_record.get("tool_mediation_observable", False)
        )
        if contact_observable:
            required_tool_target_contact = False
            if program is not None and program.stages:
                from oap.program.verdict import (
                    stage_requires_tool_target_contact,
                )

                idx = max(
                    0,
                    min(int(stage_idx), len(program.stages) - 1),
                )
                required_tool_target_contact = (
                    stage_requires_tool_target_contact(
                        program.stages[idx],
                        anchors,
                    )
                )
            cumulative_contact = _merge_execution_contact(
                cumulative_contact,
                cycle_contact,
                required_tool_target_contact=required_tool_target_contact,
            )
        evidence = dict(execution_record)
        evidence.update(dict((scene_observation or {}).get("evidence", {})))
        if readiness_evidence:
            evidence["observation_readiness"] = dict(
                readiness_evidence
            )
        evidence["observation_purpose"] = "replan"
        evidence["measured_subject_velocity6"] = (
            None
            if measured_velocity is None
            else measured_velocity.tolist()
        )
        evidence["measured_movable_state_after"] = (
            _world_movable_state(velocity_observable=True)
            if scene_observation is None
            else _observed_movable_state(
                scene_observation, measured_velocity
            )
        )
        # Command-state continuity evidence: what the jaw was told to do at
        # this prefix's end. It pins knot zero of the next solve AND is the
        # channel a terminal GripperCommand is graded against, so it must be
        # present before the gate runs -- a Stage may not exit on a measured
        # hold whose command has already been released.
        if acknowledged_jaw is not None:
            evidence["acknowledged_jaw_command"] = acknowledged_jaw.to_dict()
        evaluation = _call_terminal_gate(
            obj, cumulative_contact or {}, evidence
        )
        satisfied = (
            None
            if evaluation is None
            else getattr(evaluation, "status", evaluation)
        )
        # This exact post-prefix tracking snapshot both synchronizes the twin
        # and grades the Stage terminal.  FoundationPose keeps its persistent
        # tracking state; Stage advancement must not trigger a second global
        # registration or count a second measurement.
        stop_reason = _hard_stop_reason(evaluation)
        tool_mediation_valid = (
            True
            if evaluation is None
            else getattr(
                evaluation, "tool_mediation_satisfied", None
            )
        )
        evidence["execution_tool_mediation_valid"] = (
            tool_mediation_valid
        )
        evidence["hard_failure_reason"] = stop_reason
        if regularizer_profile is not None:
            executed_action_state.update_after_execution_attempt(
                selected_rollout=selected_rollout,
                cycle=cycle_id,
                stage_index=stage_idx,
                executed=True,
                execution_valid=True,
            )
            evidence["executed_action_state"] = (
                executed_action_state.to_dict()
            )
        if episode is not None:
            episode.record_chunk(cycle_id, {
                "planner_elapsed_s": planner_elapsed_s,
                "cycle_elapsed_s": time.perf_counter() - cycle_t0,
                "execution_prefix_frac": float(cycle_prefix_frac),
                "selected_rollout": selected_rollout,
                "executed": True,
                "stop_reason": stop_reason,
                "execution_contact": (
                    dict(cycle_contact)
                    if contact_observable and cycle_contact is not None
                    else None
                ),
                "execution_contact_cumulative": (
                    None
                    if cumulative_contact is None
                    else dict(cumulative_contact)
                ),
                "execution_valid": True,
                "execution_tool_mediation_valid":
                    tool_mediation_valid,
                "execution_evidence": evidence,
                "scene_observation": scene_observation,
                "terminal_gate_evidence": (
                    _terminal_gate_evidence_payload(evaluation)
                ),
            })
        out.append(RecedingHorizonCycle(
            cycle=cycle_id,
            seeded=seeded,
            step=res,
            obj_pose7_after=obj.copy(),
            terminal_gate_status=satisfied,
            execution_contact=(
                dict(cycle_contact)
                if cycle_contact is not None
                else None
            ),
            execution_contact_cumulative=(
                None
                if cumulative_contact is None
                else dict(cumulative_contact)
            ),
            execution_valid=True,
            execution_tool_mediation_valid=tool_mediation_valid,
            execution_evidence=evidence,
            scene_observation=scene_observation,
            terminal_gate_evidence=evaluation,
            executed=True,
            stop_reason=stop_reason,
        ))
        # shift the plan forward by the executed prefix -> next cycle's warm start.
        logger.info("[recede stage %d cycle %d] seeded=%s obj_z=%.3f satisfied=%s",
                    stage_idx, cycle_id, seeded, float(obj[2]), satisfied)
        if stop_reason is not None:
            logger.error(
                "[recede stage %d cycle %d] irreversible failure: %s",
                stage_idx,
                cycle_id,
                stop_reason,
            )
            break
        if satisfied:
            break                        # stage complete NOW; no shift, no next cycle
        if optimizer == "mppi":
            updated_actions = getattr(
                sampler_stats, "mppi_updated_action_plan", None
            )
            if updated_actions is None:
                raise RuntimeError("MPPI solve omitted action-space warm start")
            mppi_warm_start_actions = shift_mppi_effort_warm_start(
                updated_actions,
                cycle_prefix_frac,
            )
            # ``best.knots`` is the realized position trajectory.  It remains
            # useful as the ordinary ControlKnots carrier, but it is never
            # reinterpreted as MPPI torque actions; those travel through the
            # explicit action-space field above.
            shifted = shift_warm_start(
                res.best.knots,
                cycle_prefix_frac,
                bounds=bounds,
            )
        else:
            shifted = shift_warm_start(
                res.best.knots,
                cycle_prefix_frac,
                bounds=bounds,
            )
        warm_start = ControlKnots(candidate_id=nid, knots=shifted)
        nid += 1
    # Leave world.data at the final PHYSICAL state (the certifier may have reset
    # it), so the caller can snapshot it and CONTINUE the next stage from here.
    d.qpos[:] = saved_state[0]
    d.qvel[:] = saved_state[1]
    _refresh_host_kinematics()
    return out
