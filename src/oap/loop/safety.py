"""Safety policy of the online closed loop, in ONE place.

Role in the two-stage pipeline: every rule that keeps ``oap-run`` from
moving the real arm unsafely is defined and enforced here, nowhere else:

  * DRY-RUN IS THE DEFAULT. Real motion requires ``--execute`` AND
    ``--i-confirm-real-motion`` AND a home-gate pass against the canonical
    packaged home posture.
  * HARDWARE READINESS: the joint executor remains explicitly unverified.
    ``--prepare-real-execution`` performs an offline readiness run and can
    never open a robot connection. The internal hardware-verified bit has no
    CLI switch; it is enabled only after an attended bring-up/sign-off.
  * LEGACY-OFFLINE HARD REFUSAL: a loaded program whose provenance starts with
    ``offline`` is NEVER executed on hardware. The compiler that emitted such
    artifacts is gone; retaining the gate quarantines historical JSON.
  * TOOL-MEDIATION OBSERVABILITY: prefix-complete tool-target and robot-target
    contact history is reported when available. Its absence is warned and
    remains diagnostic because measured terminal predicates are the universal
    task-success contract; predicted ``running`` constraints still shape every
    sampled trajectory.
  * WORKSPACE BOUNDS: every joint waypoint is FK'd before streaming; both the
    planning grasp site and the calibrated RDK TCP point must remain inside the
    active server profile. Joint ranges are checked client-side and the server
    still supplies the authoritative speed filter/cap.
  * HOME GATE: a startup joint drift beyond tolerance means the arm is NOT at
    the posture the twin was seeded from -- refuse to move.
  * SAFE HOMING: the finally-block home restore REFUSES while a held object is
    unresolved (a go_home once opened the gripper and DROPPED a successfully
    grasped box).
  * PER-CHUNK CONFIRMATION: on by default under ``--execute``; an empty input
    re-prompts (a stray buffered newline must not silently abort a session).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from oap.program import TaskProgram
from oap.twin import GN01_OPEN_WIDTH_M
from oap.utils.io import package_config_path, read_json

logger = logging.getLogger("oap.loop.safety")

__all__ = [
    "HOME_DRIFT_TOL_RAD",
    "ObservationReadinessLimits",
    "SafetyError",
    "check_waypoint_bounds",
    "confirm_chunk",
    "default_open_width",
    "home_gate",
    "load_home_posture",
    "refuse_joint_real_execution",
    "refuse_offline_program_execution",
    "program_uses_tool_mediation",
    "require_program_approval",
    "require_execution_allowed",
    "require_joint_executor_readiness",
    "require_observation_readiness_limits",
    "require_observation_timing",
    "require_plan_freshness",
    "require_final_tracking_error",
    "require_real_tool_mediation_evidence",
    "safe_to_home",
]

#: F2 hard gate: startup joint drift beyond this means the arm is not at the
#: posture the twin was seeded from (the certified geometry would be commanded
#: from the wrong base configuration).
HOME_DRIFT_TOL_RAD = 0.09


class SafetyError(RuntimeError):
    """A safety rule refused an action (never catch-and-proceed)."""


@dataclass(frozen=True)
class ObservationReadinessLimits:
    """Calibrated, task-independent limits for one real MPC deployment.

    There are intentionally no shipped numeric defaults.  Camera rate,
    perception latency, telemetry clock synchronization, servo tracking, and
    acceptable terminal confidence are hardware facts that must be measured
    on the deployed stack rather than inferred from a simulation task.
    """

    max_observation_age_s: float
    max_camera_robot_skew_s: float
    max_plan_staleness_s: float
    max_final_joint_tracking_error_rad: float
    min_terminal_anchor_reliability: float


def require_observation_readiness_limits(
    *,
    required: bool,
    max_observation_age_s: float | None,
    max_camera_robot_skew_s: float | None,
    max_plan_staleness_s: float | None,
    max_final_joint_tracking_error_rad: float | None,
    min_terminal_anchor_reliability: float | None,
) -> ObservationReadinessLimits | None:
    """Validate the complete calibrated readiness contract.

    Dry runs may omit it.  Real execution and connection-free readiness
    preparation must provide every value together; a partial contract cannot
    silently disable the missing gate.
    """
    values = {
        "max_observation_age_s": max_observation_age_s,
        "max_camera_robot_skew_s": max_camera_robot_skew_s,
        "max_plan_staleness_s": max_plan_staleness_s,
        "max_final_joint_tracking_error_rad":
            max_final_joint_tracking_error_rad,
        "min_terminal_anchor_reliability":
            min_terminal_anchor_reliability,
    }
    supplied = any(value is not None for value in values.values())
    if required and not all(value is not None for value in values.values()):
        missing = [name for name, value in values.items() if value is None]
        raise SafetyError(
            "real observation readiness requires calibrated values for "
            + ", ".join(missing)
        )
    if not supplied:
        return None
    if not all(value is not None for value in values.values()):
        missing = [name for name, value in values.items() if value is None]
        raise ValueError(
            "observation readiness configuration is partial; missing "
            + ", ".join(missing)
        )
    numeric = {name: float(value) for name, value in values.items()}
    for name in (
        "max_observation_age_s",
        "max_camera_robot_skew_s",
        "max_plan_staleness_s",
        "max_final_joint_tracking_error_rad",
    ):
        if not np.isfinite(numeric[name]) or numeric[name] <= 0.0:
            raise ValueError(f"{name} must be finite and > 0")
    reliability = numeric["min_terminal_anchor_reliability"]
    if not np.isfinite(reliability) or not 0.0 < reliability <= 1.0:
        raise ValueError(
            "min_terminal_anchor_reliability must be finite and in (0, 1]"
        )
    return ObservationReadinessLimits(**numeric)


def _finite_timestamp(value: Any, *, label: str) -> float:
    if value is None:
        raise SafetyError(f"{label} is missing")
    stamp = float(value)
    if not np.isfinite(stamp):
        raise SafetyError(f"{label} is not finite")
    return stamp


def require_observation_timing(
    *,
    observation_timestamp_s: float | None,
    robot_timestamp_s: float | None,
    limits: ObservationReadinessLimits,
    now_s: float | None = None,
) -> dict[str, float]:
    """Hard-gate observation age and camera/robot timestamp skew."""
    observation_stamp = _finite_timestamp(
        observation_timestamp_s,
        label="camera observation timestamp",
    )
    robot_stamp = _finite_timestamp(
        robot_timestamp_s,
        label="robot-state timestamp",
    )
    now = float(time.time() if now_s is None else now_s)
    if not np.isfinite(now):
        raise SafetyError("readiness clock is not finite")
    age = now - observation_stamp
    if age < 0.0:
        raise SafetyError(
            "camera observation timestamp is in the future relative to the "
            "readiness clock"
        )
    if age > limits.max_observation_age_s:
        raise SafetyError(
            f"camera observation age {age:.6f}s exceeds calibrated limit "
            f"{limits.max_observation_age_s:.6f}s"
        )
    skew = abs(observation_stamp - robot_stamp)
    if skew > limits.max_camera_robot_skew_s:
        raise SafetyError(
            f"camera/robot timestamp skew {skew:.6f}s exceeds calibrated "
            f"limit {limits.max_camera_robot_skew_s:.6f}s"
        )
    return {
        "observation_age_s": float(age),
        "camera_robot_timestamp_skew_s": float(skew),
    }


def require_plan_freshness(
    *,
    planning_observation_timestamp_s: float | None,
    limits: ObservationReadinessLimits,
    now_s: float | None = None,
) -> float:
    """Refuse a plan whose measured start state aged out during solving."""
    stamp = _finite_timestamp(
        planning_observation_timestamp_s,
        label="planning observation timestamp",
    )
    now = float(time.time() if now_s is None else now_s)
    staleness = now - stamp
    if not np.isfinite(staleness) or staleness < 0.0:
        raise SafetyError("plan staleness clock is invalid")
    if staleness > limits.max_plan_staleness_s:
        raise SafetyError(
            f"plan staleness {staleness:.6f}s exceeds calibrated limit "
            f"{limits.max_plan_staleness_s:.6f}s; refusing prefix execution"
        )
    return float(staleness)


def require_final_tracking_error(
    *,
    final_joint_tracking_error_rad: float | None,
    limits: ObservationReadinessLimits,
) -> float:
    """Require measured final joint error before accepting an executed prefix."""
    if final_joint_tracking_error_rad is None:
        raise SafetyError("final joint tracking error is unobservable")
    error = float(final_joint_tracking_error_rad)
    if not np.isfinite(error) or error < 0.0:
        raise SafetyError("final joint tracking error is invalid")
    if error > limits.max_final_joint_tracking_error_rad:
        raise SafetyError(
            f"final joint tracking error {error:.6f}rad exceeds calibrated "
            f"limit {limits.max_final_joint_tracking_error_rad:.6f}rad"
        )
    return error


def require_execution_allowed(*, execute: bool, i_confirm_real_motion: bool) -> None:
    """Enforce the double-flag rule for real motion.

    A dry run (``execute=False``) always passes. Real motion additionally
    requires the explicit ``--i-confirm-real-motion`` acknowledgement.
    """
    if execute and not i_confirm_real_motion:
        raise SafetyError(
            "--execute requires --i-confirm-real-motion (dry-run is the default; "
            "real motion needs both flags AND a home-gate pass)")


def require_joint_executor_readiness(
    *,
    execute: bool,
    prepare_real_execution: bool,
    hardware_verified: bool,
    verification_latch_verified: bool = False,
) -> None:
    """Gate the unverified joint executor without weakening dry-run usability.

    Preparation mode is deliberately connection-free and therefore mutually
    exclusive with ``execute``. ``hardware_verified`` is an internal
    deployment/sign-off fact derived from an exact approved verification JSON,
    not a user acknowledgement.  The CLI intentionally has no boolean switch
    that can promote an unverified executor.
    """
    if execute and prepare_real_execution:
        raise SafetyError(
            "--execute and --prepare-real-execution are mutually exclusive; "
            "preparation mode never connects to or moves the robot")
    if execute and not hardware_verified:
        raise SafetyError(
            "joint executor hardware status is UNVERIFIED: real motion is "
            "refused. Run oap-preflight for the connection-free readiness "
            "report, then supply the exact attended verification JSON and its "
            "approved SHA-256.")
    if execute and not verification_latch_verified:
        raise SafetyError(
            "joint executor verification latch is absent or invalid: a bare "
            "joint_executor_hardware_verified boolean cannot authorize motion")
    if prepare_real_execution:
        logger.warning(
            "[readiness] PREPARE ONLY: planning/FK/safety checks run offline; "
            "no robot connection or motion is permitted")


def refuse_offline_program_execution(program: TaskProgram, *, execute: bool) -> None:
    """Hard-refuse executing a legacy offline-synthesized program on hardware.

    The removed keyword compiler emitted provenance ``offline:*``. Historical
    JSON artifacts may still carry that provenance, so the real-motion gate
    remains as a quarantine even though no current backend can create them.
    """
    if execute and str(program.provenance).startswith("offline"):
        raise SafetyError(
            f"program provenance {program.provenance!r} is offline-synthesized; "
            f"executing it on the real arm is forbidden. Synthesize with a live "
            f"VLM backend (--synth-backend anthropic) or ship a pre-synthesized "
            f"--program-json, then re-run with --execute.")


def program_uses_tool_mediation(program: TaskProgram) -> bool:
    """Whether any stage declares the indirect-tool running relation."""
    from oap.program.predicates import TemporalHold, ToolMediation

    for stage in program.stages:
        for predicate in (*stage.running, *stage.terminal):
            while isinstance(predicate, TemporalHold):
                predicate = predicate.inner
            if isinstance(predicate, ToolMediation):
                return True
    return False


def require_real_tool_mediation_evidence(
    program: TaskProgram,
    *,
    execute: bool,
    prepare_real_execution: bool = False,
) -> bool:
    """Report whether real prefix contact history is observable in this build.

    Simulation receives exact per-step MJWarp contact reductions. The current
    real executor deliberately returns ``contact_measurement=None`` and an
    endpoint hot-patched twin supplies only clearance proxies. Neither can
    establish that the tool contacted the target or that the robot did not.

    Tool mediation is an explicit ``running`` relation, not a second terminal
    definition. Therefore missing real contact history is a visible diagnostic,
    not an execution-authority gate. Once a calibrated provider is wired into
    every prefix this function should return its concrete capability status.

    Returns:
        ``True`` when no live tool-mediation evidence is required for this run;
        ``False`` when the current real build will report mechanism evidence as
        unavailable. The return value never authorizes motion.
    """
    if not (execute or prepare_real_execution):
        return True
    if not program_uses_tool_mediation(program):
        return True
    logger.warning(
        "[diagnostic] program declares running ToolMediation, but this build "
        "has no observable real contact-history provider over every executed "
        "prefix. Mechanism/tool-use evidence will be unavailable; explicit "
        "measured terminal predicates remain the task-success contract."
    )
    return False


def require_program_approval(
    program: TaskProgram,
    *,
    execute: bool,
    approved_program_sha256: str | None,
) -> str:
    """Require exact operator approval of the canonical program under execution.

    The returned digest is always :func:`oap.program.program_sha256`, i.e.
    SHA-256 over ``program.to_json()``. Dry runs compute and return the identity
    without requiring approval. Real execution first preserves the hard
    offline-provenance refusal, then requires the supplied digest to match
    exactly; prefixes, case changes, and surrounding whitespace are not
    accepted as approval of a different byte identity.

    This helper is deliberately independent of CLI/config plumbing. The runner
    should call it after loading/synthesizing the final program and before any
    robot connection or lease is attempted.
    """
    # Import through the public program package so this stays the one canonical
    # hash implementation used by episode logs and all four evaluation sites.
    from oap.program import program_sha256

    digest = program_sha256(program)
    if not execute:
        return digest
    refuse_offline_program_execution(program, execute=True)
    if approved_program_sha256 is None:
        raise SafetyError(
            "--execute requires --approved-program-sha256 with the full canonical "
            f"program digest (current program: {digest})")
    if approved_program_sha256 != digest:
        raise SafetyError(
            "approved program SHA-256 does not exactly match the canonical "
            f"program (approved={approved_program_sha256!r}, actual={digest})")
    return digest


def refuse_joint_real_execution(*, execute: bool) -> None:
    """Compatibility wrapper for callers that do not carry readiness state."""
    require_joint_executor_readiness(
        execute=execute,
        prepare_real_execution=False,
        hardware_verified=False,
        verification_latch_verified=False,
    )


def load_home_posture(path: Path | None = None) -> dict[str, Any]:
    """Load the canonical home posture json (packaged default).

    The SAME file seeds the twin keyframe and gates the real arm, so the plan
    and the home-gate can never disagree about where "home" is.
    """
    home_path = Path(path) if path is not None else package_config_path(
        "calibration/lab_home_posture.json")
    home = read_json(home_path)
    if "q_rad" not in home:
        raise SafetyError(f"home posture json {home_path} has no 'q_rad' key")
    return dict(home)


def home_gate(measured_q: np.ndarray | list[float], home_q: np.ndarray | list[float], *,
              tol_rad: float = HOME_DRIFT_TOL_RAD, allow_drift: bool = False) -> float:
    """The startup home-gate: refuse to move from a drifted posture.

    Returns the measured max joint drift (rad). Raises :class:`SafetyError`
    when the drift exceeds tolerance and ``allow_drift`` is False.
    """
    drift = float(np.abs(np.asarray(measured_q, dtype=float)
                         - np.asarray(home_q, dtype=float)).max())
    logger.info("[home-check] max joint drift from canonical home: %.4f rad", drift)
    if drift > float(tol_rad) and not allow_drift:
        raise SafetyError(
            f"[home-gate] ABORT: startup joint drift {drift:.4f} rad exceeds "
            f"tolerance {float(tol_rad):.2f} rad. The arm is not at canonical "
            f"home -- re-home (--home-first / jog to the home posture) before "
            f"--execute, or pass --allow-home-drift to override.")
    logger.info("[home-gate] %s (drift %.4f rad, tol %.2f rad, allow_home_drift=%s)",
                "PASS" if drift <= float(tol_rad) else "OVERRIDDEN",
                drift, float(tol_rad), bool(allow_drift))
    return drift


def check_waypoint_bounds(position: np.ndarray | list[float], *,
                          ws_x: tuple[float, float], ws_y: tuple[float, float],
                          ws_z: tuple[float, float]) -> None:
    """Bounds-check ONE commanded waypoint position against the workspace box."""
    p = np.asarray(position, dtype=float).reshape(3)
    if not (ws_x[0] <= p[0] <= ws_x[1]
            and ws_y[0] <= p[1] <= ws_y[1]
            and ws_z[0] <= p[2] <= ws_z[1]):
        raise SafetyError(
            f"commanded waypoint {p.tolist()} outside the workspace bounds "
            f"x={list(ws_x)} y={list(ws_y)} z={list(ws_z)}")


def safe_to_home(
    held_close_width_m: float | None = None,
    *,
    held_status: bool | None = None,
    close_was_commanded: bool = False,
) -> bool:
    """Is the finally-block home restore SAFE right now?

    Homing opens the gripper and sweeps the arm; while a held object is
    unresolved that DROPS it (it has happened -- a successfully grasped box was
    dropped by the exit ritual). The restore must be skipped and the operator
    told to recover manually.
    """
    if held_status is False:
        return True
    if held_status is True:
        return False
    # UNKNOWN after any close is unsafe: opening/sweeping could drop an object.
    if close_was_commanded:
        return False
    return held_close_width_m is None


def confirm_initial_motion(token_id: str) -> bool:
    """Require a fresh, session-specific confirmation before bootstrap motion.

    Connecting and reading controller identity is not motion, but ``home-first``
    and the verified gripper-open bootstrap are.  The signed operator packet
    can be several minutes old by then, so the attended process must confirm
    the current workspace and E-stop state immediately before the first lease.
    A token suffix prevents a buffered ``y`` from an earlier prompt or pasted
    command from authorizing motion.
    """
    token = str(token_id).strip()
    if len(token) < 8:
        return False
    expected = f"START {token[-8:]}"
    answer = input(
        "FIRST REAL MOTION: confirm workspace/hand are clear, robot is "
        "stationary with no contact, E-stop is in hand, and Auto Remote is "
        f"enabled. Type '{expected}' to continue: "
    ).strip()
    return answer == expected


def confirm_chunk(chunk_idx: int) -> bool:
    """Per-chunk E-stop/workspace confirmation (empty input re-prompts).

    A stray buffered newline (e.g. from pasting a multi-line command) must NOT
    authorize or silently abort the whole run, so empty input re-prompts.  A
    generic ``y`` is deliberately insufficient. Returns True to execute,
    False to abort.
    """
    expected = f"EXECUTE CHUNK {int(chunk_idx)}"
    answer = ""
    while not answer:
        answer = input(
            "REAL MOTION: confirm workspace/hand are clear and E-stop is in "
            f"hand. Type '{expected}' to execute: "
        ).strip()
        if not answer:
            print(
                f"  (no input -- type '{expected}' to execute; anything "
                "else aborts)"
            )
    return answer == expected


def default_open_width(home: dict[str, Any]) -> float:
    """The gripper width the home posture expects (default: GN01 open)."""
    return float(home.get("gripper_width_m", GN01_OPEN_WIDTH_M))
