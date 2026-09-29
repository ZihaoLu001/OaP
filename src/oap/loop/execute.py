"""EXECUTE layer: replay the joint control in the twin, and (M4) on the real arm.

Role in the pipeline: the joint MPC (:mod:`oap.loop.mpc_loop`) samples a
zero-order-held sequence of seven arm torques plus one signed gripper force.
:func:`replay_in_sim` applies those generalized efforts in the twin (no IK and
no motion initializer), and :func:`execute_joint_trajectory` sends the SAME
selected effort endpoints through flexiv-control's strict joint-torque RPC.
Positive gripper effort closes, negative effort opens, and aperture remains
measured state only.  The real adapter interpolates consecutive torque
endpoints at 1 kHz solely to satisfy Flexiv's smooth RT command contract; it
never reinterprets them as positions or velocities.  The external ZED records
the motion, and the twin is re-synced from the measured real joints before the
next MPC step.

Session lifecycle: :func:`connect_robot` opens the RemoteRobot lease, verifies
the active safety profile (envelope + table-floor cross-check), optionally
homes first, enforces the startup home-gate, records the pre-open state, and
opens/verifies the gripper before recording the actual initial posture.
:func:`go_home_safe_or_hold` is the default exit ritual -- and it REFUSES to
home while a held object is unresolved (homing once opened the gripper and
dropped a successfully grasped box). An explicitly selected manual-restore
run instead calls :func:`stop_and_record_manual_restore`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from oap.loop import safety
from oap.loop.executor_readiness import (
    require_executor_readiness,
    validate_robot_state_telemetry,
)
from oap.loop.execution_authorization import (
    require_authorized_calibration_hardware_identity,
    require_authorized_server_identity,
    require_consumed_execution_authorization,
)
from oap.loop.observe import observation_payload_path
from oap.twin import (
    SimWorld,
    reset_robot_qpos,
)
from oap.twin.assets import GN01_OPEN_WIDTH_M, gripper_joint_targets
from oap.utils.io import write_json_atomic

if TYPE_CHECKING:
    from oap.loop.runner import LoopConfig

logger = logging.getLogger("oap.loop.execute")

__all__ = [
    "RobotSession",
    "connect_robot",
    "controller_effective_joint_limits",
    "controller_gripper_limits",
    "controller_runtime_contract",
    "controller_runtime_control_hz",
    "execution_prefix_schedule",
    "gpu_prefix_endpoint_fraction",
    "execute_joint_trajectory",
    "go_home_safe_or_hold",
    "joint_prefix_knot_times",
    "joint_prefix_knots",
    "observed_contact_proxies",
    "replay_in_sim",
    "resync_robot_qpos",
    "resync_robot_state",
    "sync_robot_state_to_twin",
    "start_zed_recording",
    "stop_and_record_manual_restore",
    "validate_controller_gripper_path",
    "validate_controller_joint_path",
    "validate_joint_trajectory",
]


@dataclass
class RobotSession:
    """An open flexiv-control lease + the resolved active safety envelope."""

    robot: Any
    active_profile: Any
    home_q: np.ndarray
    home: dict[str, Any]
    workspace_x: tuple[float, float]
    workspace_y: tuple[float, float]
    workspace_z: tuple[float, float]
    home_drift_rad: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def close(self) -> None:
        """Release the RemoteRobot lease (never raises)."""
        try:
            self.robot.__exit__(None, None, None)
        except Exception as exc:  # noqa: BLE001 - closing must never mask the run
            logger.warning("[robot] lease release failed: %r", exc)


def _verify_open_gripper_width(
    *,
    requested_width_m: float,
    command_settled_width_m: Any,
    measured_width_m: float,
    resolution_m: float | None,
) -> tuple[float, float]:
    """Require command acknowledgement and telemetry to agree with open."""
    try:
        settled_width_m = float(command_settled_width_m)
    except (TypeError, ValueError) as exc:
        raise safety.SafetyError(
            "blocking gripper-open command returned no finite settled "
            "width; refusing to plan from an unverified initial posture"
        ) from exc
    tolerance_m = 0.0 if resolution_m is None else float(resolution_m)
    values = (
        float(requested_width_m),
        settled_width_m,
        float(measured_width_m),
        tolerance_m,
    )
    if not all(np.isfinite(value) for value in values):
        raise safety.SafetyError(
            "gripper-open verification received a non-finite width or "
            "resolution; refusing the initial posture"
        )
    if tolerance_m < 0.0:
        raise safety.SafetyError(
            "gripper-width resolution must be non-negative before verifying "
            "the initial open posture"
        )
    settled_error_m = abs(settled_width_m - requested_width_m)
    measured_error_m = abs(measured_width_m - requested_width_m)
    settled_measured_error_m = abs(
        settled_width_m - measured_width_m
    )
    if (
        settled_error_m > tolerance_m
        or measured_error_m > tolerance_m
        or settled_measured_error_m > tolerance_m
    ):
        raise safety.SafetyError(
            "gripper failed verified-open initial posture: requested "
            f"{requested_width_m:.6f} m, command settled at "
            f"{settled_width_m:.6f} m, telemetry measured "
            f"{measured_width_m:.6f} m, allowed error "
            f"{tolerance_m:.6f} m"
        )
    return settled_width_m, tolerance_m


def controller_runtime_control_hz(
    server_info: Mapping[str, Any],
    *,
    require_top_level: bool = True,
) -> tuple[float, str]:
    """Resolve the controller command rate from its read-only runtime identity.

    ``control_hz`` at the top level is the production protocol field.  The two
    historical nested locations are inspected only so an old deployment fails
    with an actionable compatibility error instead of looking like a missing
    value.  Real execution must never infer or hard-code the controller rate:
    it determines the exact integer frame schedule sent to the robot.
    """
    if not isinstance(server_info, Mapping):
        raise safety.SafetyError(
            "controller server_info is not an object; cannot bind control_hz"
        )
    source = "control_hz"
    raw = server_info.get("control_hz")
    if raw is None:
        legacy: list[tuple[str, Any]] = []
        for container_name in ("execution_runtime", "runtime"):
            container = server_info.get(container_name)
            if isinstance(container, Mapping) and container.get("control_hz") is not None:
                legacy.append(
                    (f"{container_name}.control_hz", container["control_hz"])
                )
        if legacy:
            source, raw = legacy[0]
            if require_top_level:
                raise safety.SafetyError(
                    "controller reports legacy "
                    f"{source}, but production execution requires top-level "
                    "server_info.control_hz; update flexiv-control before motion"
                )
        else:
            raise safety.SafetyError(
                "controller server_info is missing top-level control_hz; "
                "refusing to infer the real command rate"
            )
    if isinstance(raw, bool):
        raise safety.SafetyError(
            f"controller {source} must be a finite positive rate, got {raw!r}"
        )
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise safety.SafetyError(
            f"controller {source} must be a finite positive rate, got {raw!r}"
        ) from exc
    if not np.isfinite(value) or value <= 0.0:
        raise safety.SafetyError(
            f"controller {source} must be a finite positive rate, got {raw!r}"
        )
    return value, source


def controller_effective_joint_limits(
    server_info: Mapping[str, Any],
    *,
    joint_count: int = 7,
) -> dict[str, Any]:
    """Validate the controller's signed-by-content joint limit contract.

    The v3 server intersects configured, hardware, and firmware limits before
    publishing this object. Real execution consumes the enforced position
    interval and per-joint velocity ceilings directly; it never substitutes
    the planning model ranges or the historical uniform 2 rad/s fallback.
    """
    raw = server_info.get("effective_joint_limits")
    if not isinstance(raw, Mapping):
        raise safety.SafetyError(
            "controller server_info v3 is missing effective_joint_limits"
        )
    required = (
        "sources",
        "hard_position_min_rad",
        "hard_position_max_rad",
        "enforced_position_min_rad",
        "enforced_position_max_rad",
        "base_velocity_max_rad_s",
        "max_joint_speed_scale",
        "base_torque_max_nm",
        "max_joint_torque_scale",
        "sha256",
    )
    missing = [name for name in required if name not in raw]
    if missing:
        raise safety.SafetyError(
            "controller effective_joint_limits is missing fields: "
            + ", ".join(missing)
        )
    arrays: dict[str, np.ndarray] = {}
    for name in required[1:6]:
        try:
            value = np.asarray(raw[name], dtype=float)
        except (TypeError, ValueError) as exc:
            raise safety.SafetyError(
                f"controller effective_joint_limits.{name} is not numeric"
            ) from exc
        if value.shape != (joint_count,) or not np.all(np.isfinite(value)):
            raise safety.SafetyError(
                f"controller effective_joint_limits.{name} must contain "
                f"exactly {joint_count} finite values"
            )
        arrays[name] = value
    for name in ("base_torque_max_nm",):
        try:
            value = np.asarray(raw[name], dtype=float)
        except (TypeError, ValueError) as exc:
            raise safety.SafetyError(
                f"controller effective_joint_limits.{name} is not numeric"
            ) from exc
        if (
            value.shape != (joint_count,)
            or not np.all(np.isfinite(value))
            or np.any(value <= 0.0)
        ):
            raise safety.SafetyError(
                f"controller effective_joint_limits.{name} must contain "
                f"exactly {joint_count} finite positive values"
            )
        arrays[name] = value
    if np.any(arrays["hard_position_min_rad"] >= arrays["hard_position_max_rad"]):
        raise safety.SafetyError(
            "controller effective hard joint position intervals are invalid"
        )
    if np.any(
        arrays["enforced_position_min_rad"]
        >= arrays["enforced_position_max_rad"]
    ):
        raise safety.SafetyError(
            "controller effective enforced joint position intervals are invalid"
        )
    if np.any(
        arrays["enforced_position_min_rad"]
        < arrays["hard_position_min_rad"]
    ) or np.any(
        arrays["enforced_position_max_rad"]
        > arrays["hard_position_max_rad"]
    ):
        raise safety.SafetyError(
            "controller enforced joint interval expands outside hard limits"
        )
    if np.any(arrays["base_velocity_max_rad_s"] <= 0.0):
        raise safety.SafetyError(
            "controller effective per-joint velocity limits must be positive"
        )
    try:
        scale_ceiling = float(raw["max_joint_speed_scale"])
        torque_scale_ceiling = float(raw["max_joint_torque_scale"])
    except (TypeError, ValueError) as exc:
        raise safety.SafetyError(
            "controller effective max_joint_speed_scale is not numeric"
        ) from exc
    if not np.isfinite(scale_ceiling) or not 0.0 < scale_ceiling <= 1.0:
        raise safety.SafetyError(
            "controller effective max_joint_speed_scale must be in (0, 1]"
        )
    if (
        not np.isfinite(torque_scale_ceiling)
        or not 0.0 < torque_scale_ceiling <= 1.0
    ):
        raise safety.SafetyError(
            "controller effective max_joint_torque_scale must be in (0, 1]"
        )
    sources = raw["sources"]
    if (
        not isinstance(sources, Sequence)
        or isinstance(sources, (str, bytes))
        or not sources
        or any(not str(value).strip() for value in sources)
    ):
        raise safety.SafetyError(
            "controller effective_joint_limits.sources must be non-empty"
        )
    fingerprint = str(raw["sha256"])
    unsigned = {name: raw[name] for name in raw if name != "sha256"}
    calculated = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if fingerprint != calculated:
        raise safety.SafetyError(
            "controller effective_joint_limits sha256 does not match content"
        )
    return {
        "sources": [str(value) for value in sources],
        **{name: value.tolist() for name, value in arrays.items()},
        "max_joint_speed_scale": scale_ceiling,
        "max_joint_torque_scale": torque_scale_ceiling,
        "sha256": fingerprint,
    }


def controller_gripper_limits(
    server_info: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate runtime GN01 width/velocity/force limits from server_info."""
    raw = server_info.get("gripper_limits")
    if not isinstance(raw, Mapping):
        raise safety.SafetyError(
            "controller server_info v3 is missing runtime gripper_limits"
        )
    numeric_names = (
        "min_width_m",
        "max_width_m",
        "min_velocity_m_s",
        "max_velocity_m_s",
        "min_force_n",
        "max_force_n",
    )
    values: dict[str, float] = {}
    for name in numeric_names:
        try:
            value = float(raw[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise safety.SafetyError(
                f"controller gripper_limits.{name} must be finite"
            ) from exc
        if not np.isfinite(value):
            raise safety.SafetyError(
                f"controller gripper_limits.{name} must be finite"
            )
        values[name] = value
    for lower, upper in (
        ("min_width_m", "max_width_m"),
        ("min_velocity_m_s", "max_velocity_m_s"),
        ("min_force_n", "max_force_n"),
    ):
        if values[lower] > values[upper]:
            raise safety.SafetyError(
                f"controller gripper limits {lower}/{upper} are not ordered"
            )
    if values["max_velocity_m_s"] <= 0.0:
        raise safety.SafetyError(
            "controller gripper max_velocity_m_s must be positive"
        )
    normalized = {
        "source": str(raw.get("source", "")).strip(),
        "device_name": str(raw.get("device_name", "")).strip(),
        **values,
    }
    if not normalized["source"] or not normalized["device_name"]:
        raise safety.SafetyError(
            "controller gripper_limits needs non-empty source/device_name"
        )
    normalized["sha256"] = hashlib.sha256(
        json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return normalized


def controller_runtime_contract(
    server_info: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the exact local v5 protocol plus live timing/limit facts."""
    if not isinstance(server_info, Mapping):
        raise safety.SafetyError("controller server_info is not an object")
    try:
        from flexiv_control import __version__ as flexiv_control_version
        from flexiv_control.server import protocol
    except Exception as exc:
        raise safety.SafetyError(
            "cannot import the local flexiv-control v5 protocol identity"
        ) from exc
    if (
        protocol.SERVER_INFO_SCHEMA != "flexiv-control.server-info.v5"
        or protocol.PROTOCOL_ID != "flexiv-control.trajectory-rpc.v5"
    ):
        raise safety.SafetyError(
            "local flexiv-control is not the required server-info/RPC v5 client"
        )
    expected = {
        "schema": protocol.SERVER_INFO_SCHEMA,
        "package": "flexiv-control",
        "package_version": flexiv_control_version,
        "protocol_id": protocol.PROTOCOL_ID,
        "protocol_fingerprint_sha256": protocol.PROTOCOL_FINGERPRINT_SHA256,
        "source_fingerprint_sha256": protocol.SOURCE_FINGERPRINT_SHA256,
    }
    mismatches = [
        f"{name}: local={value!r}, controller={server_info.get(name)!r}"
        for name, value in expected.items()
        if server_info.get(name) != value
    ]
    if mismatches:
        raise safety.SafetyError(
            "controller server_info v3 identity mismatch: " + "; ".join(mismatches)
        )
    active_profile = str(server_info.get("active_safety_profile", "")).strip()
    if not active_profile:
        raise safety.SafetyError(
            "controller server_info v3 is missing active_safety_profile"
        )
    control_hz, control_hz_source = controller_runtime_control_hz(
        server_info,
        require_top_level=True,
    )
    limits = controller_effective_joint_limits(server_info)
    gripper_limits = controller_gripper_limits(server_info)
    return {
        "control_hz": control_hz,
        "control_hz_source": control_hz_source,
        "active_safety_profile": active_profile,
        "effective_joint_limits": limits,
        "gripper_limits": gripper_limits,
    }


def connect_robot(cfg: LoopConfig, home: dict[str, Any], *,
                  z_real_to_plan: float, table_top_z: float,
                  out_dir: Path) -> RobotSession:
    """Connect to flexiv-control serve and run the full startup safety ritual.

    In order: double-flag check, lease, ACTIVE-profile fetch (the ONE source of
    truth for the executable envelope -- the pre-send bounds check uses these
    bounds so a waypoint the loop accepts can never be silently displaced by
    the server-side filter), table-floor cross-check (refuse a profile whose z
    floor sits meaningfully below the MEASURED table plane), optional
    ``--home-first`` restore, Cartesian-impedance start, the HOME-GATE, the
    pre-open evidence record, gripper open/verification, and only then the
    initial-posture evidence record.
    """
    # This must remain the first executable gate in the connection layer.
    # Failure occurs before RemoteRobot is imported/constructed, so there is
    # no TCP connection, lease, home, control-mode, gripper, or trajectory RPC.
    require_consumed_execution_authorization(cfg)
    require_authorized_calibration_hardware_identity(cfg)
    readiness = require_executor_readiness(
        cfg.joint_executor_verification_json,
        cfg.joint_executor_verification_sha256,
    )
    cfg.joint_executor_hardware_verified = bool(
        readiness["hardware_verified"]
    )
    cfg.joint_executor_verification_latch_verified = bool(
        readiness["verification_latch_valid"]
    )
    cfg.joint_executor_software_fingerprint_sha256 = readiness[
        "flexiv_control"
    ]["software_fingerprint_sha256"]
    safety.require_joint_executor_readiness(
        execute=bool(cfg.execute),
        prepare_real_execution=bool(cfg.prepare_real_execution),
        hardware_verified=bool(cfg.joint_executor_hardware_verified),
        verification_latch_verified=bool(
            cfg.joint_executor_verification_latch_verified
        ),
    )
    safety.require_execution_allowed(execute=cfg.execute,
                                     i_confirm_real_motion=cfg.i_confirm_real_motion)
    if not safety.confirm_initial_motion(
        str(cfg.execution_authorization_token_id)
    ):
        raise KeyboardInterrupt(
            "operator declined the immediate pre-motion confirmation"
        )
    # The operator may wait at the prompt. Recheck expiry after the exact
    # session phrase and before importing/constructing the robot client.
    require_consumed_execution_authorization(cfg)
    from flexiv_control import RemoteRobot
    from flexiv_control.types import GripperCommand

    home_q = np.asarray(home["q_rad"], dtype=float)
    robot = RemoteRobot(cfg.server_host, port=cfg.server_port, owner="oap-run")
    robot.__enter__()
    try:
        # Close the probe->lease race: re-read identity on the exact leased
        # session before profile access, home, mode changes, or gripper motion.
        leased_server_identity = robot.get_server_info()
        require_authorized_server_identity(cfg, leased_server_identity)
        runtime_contract = controller_runtime_contract(leased_server_identity)
        controller_control_hz = runtime_contract["control_hz"]
        controller_control_hz_source = runtime_contract["control_hz_source"]
        effective_joint_limits = runtime_contract["effective_joint_limits"]
        gripper_limits = runtime_contract["gripper_limits"]
        logger.info(
            "[controller-runtime] control_hz=%.9g source=%s limits_sha256=%s",
            controller_control_hz,
            controller_control_hz_source,
            effective_joint_limits["sha256"],
        )
        logger.info(
            "[controller-runtime] gripper=%s limits_sha256=%s",
            gripper_limits["device_name"],
            gripper_limits["sha256"],
        )
        active_profile = robot.get_safety_profile()
        if active_profile.name != runtime_contract["active_safety_profile"]:
            raise safety.SafetyError(
                "controller active profile changed after server_info: "
                f"identity={runtime_contract['active_safety_profile']!r}, "
                f"profile_rpc={active_profile.name!r}"
            )
        logger.info("[profile] active=%r x=%s y=%s z=%s (%s) caps=%sm/s",
                    active_profile.name, list(active_profile.ws_x),
                    list(active_profile.ws_y), list(active_profile.ws_z),
                    active_profile.workspace_action, active_profile.max_linear_speed)
        if active_profile.name != "oap_lab":
            logger.warning("[profile] WARNING: expected 'oap_lab' "
                           "(serve --config rizon4s_oap_lab), got %r",
                           active_profile.name)
        # Held-transport payload allowance is granted server-side by the
        # active profile's contact.max_allowance. The exact leased v5
        # server identity has already been re-checked above, so a zero grant
        # means the active profile lacks the required allowance; every held
        # transport would instant-stop on the payload's static wrench.
        allowance_grant = np.asarray(
            getattr(active_profile, "max_wrench_allowance", np.zeros(6)), float)
        if not np.any(allowance_grant > 0):
            logger.warning("[profile] WARNING: zero contact-wrench allowance grant "
                           "-- active profile lacks contact.max_allowance; held "
                           "transports will trip the wrench guard. Verify the "
                           "exact safety profile configuration before real runs.")
        # Safety cross-check: the profile's z floor is the last-resort guard
        # against driving the TCP into the table. We KNOW the real table plane
        # (measured at session start: base z = plan table_top_z - z_real_to_plan),
        # so refuse to execute if the floor sits meaningfully below it.
        table_z_base = float(table_top_z - z_real_to_plan)
        if active_profile.ws_z[0] < table_z_base - 0.03:
            raise safety.SafetyError(
                f"safety: profile z floor {active_profile.ws_z[0]:.3f} m is below the "
                f"measured table plane {table_z_base:.3f} m (base frame); fix the "
                f"'{active_profile.name}' profile's workspace z before --execute")
        if cfg.home_first:
            logger.info("[home-first] resetting arm to canonical home (go_home_safe, lift 0.10m) ...")
            hr = robot.go_home_safe(q_home=home_q, lift_m=0.10)
            logger.info("[home-first] %s", "done" if getattr(hr, "success", True) else f"WARNING: {hr}")
        # Chunks run under the NRT Cartesian impedance mode; start it explicitly
        # (execute_cartesian_trajectory also auto-ensures it, but the runbook should
        # not rely on the implicit path).
        robot.start_cartesian_impedance()
        s = robot.get_state()
        telemetry_contract = validate_robot_state_telemetry(
            s,
            expected_joint_count=len(home_q),
        )
        drift = safety.home_gate(np.asarray(s.q, dtype=float), home_q,
                                 allow_drift=cfg.allow_home_drift)
        write_json_atomic(out_dir / "pre_open_posture.json", {
            "schema": "oap_pre_open_posture_v1",
            "q_rad": [float(v) for v in s.q],
            "tcp_pose": [float(v) for v in s.tcp_pose],
            "robot_state_stamp_s": telemetry_contract["stamp_s"],
            "gripper_width_m": telemetry_contract["gripper_width_m"],
            "gripper_force_n": telemetry_contract["gripper_force_n"],
            "telemetry_contract": telemetry_contract,
            "canonical_home_q": home_q.tolist(),
            "drift_rad": float(drift),
            "active_safety_profile": active_profile.name,
            "unix_s": time.time(),
        })
        # Align the real gripper with the sim home: OPEN before anything else
        # (the Grav parks closed; the GN01 sim home is open). Blocking gripper
        # command -- no do-nothing hold-pose chunk as a wait workaround.
        requested_open_width_m = safety.default_open_width(home)
        final_w = robot.command_gripper(
            GripperCommand(width=requested_open_width_m),
            wait=True, timeout=8.0)
        opened_state = robot.get_state()
        opened_telemetry = validate_robot_state_telemetry(
            opened_state,
            expected_joint_count=len(home_q),
        )
        resolution_raw = getattr(cfg, "gripper_width_resolution_m", None)
        measured_open_width_m = float(
            opened_telemetry["gripper_width_m"]
        )
        settled_width_m, open_tolerance_m = _verify_open_gripper_width(
            requested_width_m=requested_open_width_m,
            command_settled_width_m=final_w,
            measured_width_m=measured_open_width_m,
            resolution_m=resolution_raw,
        )
        write_json_atomic(out_dir / "initial_posture.json", {
            "schema": "oap_initial_posture_v2",
            "q_rad": [float(v) for v in opened_state.q],
            "tcp_pose": [float(v) for v in opened_state.tcp_pose],
            "robot_state_stamp_s": opened_telemetry["stamp_s"],
            "gripper_width_m": measured_open_width_m,
            "gripper_force_n": opened_telemetry["gripper_force_n"],
            "telemetry_contract": opened_telemetry,
            "canonical_home_q": home_q.tolist(),
            "drift_rad": float(drift),
            "active_safety_profile": active_profile.name,
            "requested_open_width_m": requested_open_width_m,
            "command_settled_width_m": settled_width_m,
            "open_verification_tolerance_m": open_tolerance_m,
            "open_verified": True,
            "pre_open_posture_artifact": "pre_open_posture.json",
            "unix_s": time.time(),
        })
        logger.info(
            "[gripper] opened and verified, settled width=%.6f m "
            "(telemetry %.6f m, tolerance %.6f m)",
            settled_width_m,
            measured_open_width_m,
            open_tolerance_m,
        )
    except BaseException:
        robot.__exit__(None, None, None)
        raise
    return RobotSession(
        robot=robot, active_profile=active_profile, home_q=home_q, home=home,
        workspace_x=tuple(active_profile.ws_x),
        workspace_y=tuple(active_profile.ws_y),
        workspace_z=tuple(active_profile.ws_z),
        home_drift_rad=float(drift),
        extra={
            "execution_authorization_token_id": (
                cfg.execution_authorization_token_id
            ),
            "execution_authorization_token_sha256": (
                cfg.execution_authorization_token_sha256
            ),
            "execution_authorization_expires_at": (
                cfg.execution_authorization_expires_at
            ),
            "authorized_server_identity": dict(
                cfg.execution_authorization_server_identity or {}
            ),
            "verification_latch_verified": True,
            "verification_latch_sha256": readiness[
                "verification_latch"
            ]["sha256"],
            "verification_hardware_id": readiness[
                "verification_latch"
            ]["hardware_id"],
            "executor_software_fingerprint_sha256": readiness[
                "flexiv_control"
            ]["software_fingerprint_sha256"],
            "joint_telemetry_verified": True,
            "joint_telemetry_source": opened_telemetry["source"],
            "joint_telemetry_contract": opened_telemetry,
            "joint_speed_scale_verified": bool(
                readiness["verification_latch"]["attestations"].get(
                    "max_joint_speed_scale_enforced"
                )
            ),
            "joint_torque_stream_verified": bool(
                readiness["verification_latch"]["attestations"].get(
                    "rt_joint_torque_stream_verified"
                )
                and readiness["verification_latch"]["attestations"].get(
                    "max_joint_torque_scale_enforced"
                )
            ),
            "controller_control_hz": controller_control_hz,
            "controller_control_hz_source": controller_control_hz_source,
            "controller_effective_joint_limits": effective_joint_limits,
            "controller_effective_joint_limits_sha256": (
                effective_joint_limits["sha256"]),
            "controller_gripper_limits": gripper_limits,
            "controller_gripper_limits_sha256": gripper_limits["sha256"],
        },
    )


# --------------------------------------------------------------------------
# Per-chunk execution (+ external ZED recording)
# --------------------------------------------------------------------------
def start_zed_recording(cfg: LoopConfig, rec_dir: Path, duration_s: float,
                        *, fps: float = 15.0) -> subprocess.Popen:
    """Start the external ZED recording payload and wait for its LIVE flag.

    The recorder (an observer-env payload) drops ``.recording_live`` in
    ``rec_dir`` once frames are streaming; motion is only commanded after that
    so the clip captures the chunk from its first frame of real motion, not
    after the ~2 s camera open. Missing configuration, an early process exit,
    or no LIVE flag is fatal: an unrecorded real prefix cannot later support a
    real contact/tool-use claim.
    """
    if cfg.observer_python is None:
        raise safety.SafetyError(
            "real joint execution requires observer_python; refusing an "
            "unrecorded prefix")
    rec_dir.mkdir(parents=True, exist_ok=True)
    live_flag = rec_dir / ".recording_live"
    live_flag.unlink(missing_ok=True)
    try:
        argv = [str(cfg.observer_python), str(observation_payload_path("zed_record.py"))]
    except RuntimeError:
        # Single-payload deployments fold the recorder into the capture payload
        # behind a --record switch; same flags either way.
        argv = [str(cfg.observer_python), str(observation_payload_path("zed_capture.py")),
                "--record"]
    rec = subprocess.Popen([*argv, "--out-dir", str(rec_dir),
                            "--duration-s", str(float(duration_s)), "--fps", str(float(fps))])
    t0 = time.time()
    while not live_flag.exists() and (time.time() - t0) < 8.0:
        if rec.poll() is not None:
            break
        time.sleep(0.05)
    if not live_flag.exists():
        code = rec.poll()
        if code is None:
            rec.kill()
            rec.wait(timeout=10)
        raise safety.SafetyError(
            "ZED recorder did not attest a live stream before motion "
            f"(returncode={code})")
    return rec


def _recording_coverage(
    rec_dir: Path,
    *,
    motion_start_monotonic_s: float,
    motion_end_monotonic_s: float,
    returncode: int | None,
) -> dict[str, Any]:
    """Verify that saved, timestamped frames bracket the whole real prefix."""
    ts_path = rec_dir / "timestamps.txt"
    timestamps: list[float] = []
    if ts_path.exists():
        for line in ts_path.read_text(encoding="utf-8").splitlines():
            try:
                value = float(line.strip())
            except ValueError:
                continue
            if np.isfinite(value):
                timestamps.append(value)
    frame_count = len(list(rec_dir.glob("frame_*.png")))
    bracketed = bool(
        len(timestamps) >= 2
        and timestamps[0] <= float(motion_start_monotonic_s)
        and timestamps[-1] >= float(motion_end_monotonic_s))
    complete = bool(returncode == 0 and frame_count >= len(timestamps) >= 2
                    and bracketed)
    return {
        "observable": complete,
        "source": "zed_timestamped_frames",
        "returncode": returncode,
        "frame_count": frame_count,
        "timestamp_count": len(timestamps),
        "first_monotonic_s": timestamps[0] if timestamps else None,
        "last_monotonic_s": timestamps[-1] if timestamps else None,
        "motion_start_monotonic_s": float(motion_start_monotonic_s),
        "motion_end_monotonic_s": float(motion_end_monotonic_s),
        "bracketed_motion": bracketed,
    }


def sync_robot_state_to_twin(
    world: SimWorld,
    arm_q: np.ndarray | list[float],
    gripper_width_m: float,
    *,
    controller_target: np.ndarray | list[float] | None = None,
    acknowledged_jaw_effort_n: float | None = None,
) -> float:
    """Write measured state and the latest control target into the twin.

    This is state estimation, not an initializer: no task, stage, or desired
    gripper intent enters. The seven arm joints and all seven coupled GN01
    positions are updated from one live measurement. Position-actuator targets
    remain the latest successfully acknowledged controller command when
    ``controller_target`` is supplied. Keeping measured aperture and commanded
    effort separate preserves the action/state boundary.

    Before the first command, omitting ``controller_target`` initializes a
    neutral ARM hold from the measurement. The jaw is different in kind: its
    command lane is signed effort, so an obstructed aperture is not a command.
    That column therefore comes from ``acknowledged_jaw_effort_n``, and its
    absence is refused rather than filled from telemetry.

    Returns the maximum arm-joint correction in radians.  Missing, non-finite,
    or out-of-model telemetry is a hard safety error; it is never clipped into
    an apparently valid state.
    """
    from oap.twin.control_knots import (
        JawCommandUnknown,
    )
    import mujoco

    robot = world.robot
    arm_addr = np.asarray(robot["qpos_addr"], dtype=int)
    arm_dof = np.asarray(robot["dof_addr"], dtype=int)
    measured_q = np.asarray(arm_q, dtype=float).reshape(-1)
    if measured_q.shape != (len(arm_addr),):
        raise safety.SafetyError(
            "live robot state must contain exactly "
            f"{len(arm_addr)} arm joints, got {measured_q.shape}"
        )
    if not np.all(np.isfinite(measured_q)):
        raise safety.SafetyError("live arm telemetry contains NaN/Inf")

    width = float(gripper_width_m)
    if not np.isfinite(width):
        raise safety.SafetyError("live gripper-width telemetry is missing or non-finite")

    gripper_actuator = int(robot["gripper_actuator_id"])
    ctrl_range = np.asarray(
        world.model.actuator_ctrlrange[gripper_actuator],
        dtype=float,
    )
    if (
        ctrl_range.shape != (2,)
        or not np.all(np.isfinite(ctrl_range))
        or width < -1e-9
        or width > float(GN01_OPEN_WIDTH_M) + 1e-9
    ):
        raise safety.SafetyError(
            f"live gripper width {width!r} m is outside the planning model "
            f"physical range [0, {GN01_OPEN_WIDTH_M}]"
        )

    actuator_ids = np.concatenate([
        np.asarray(robot["arm_actuator_ids"], dtype=int),
        np.asarray([gripper_actuator], dtype=int),
    ])
    if controller_target is None:
        if acknowledged_jaw_effort_n is None:
            raise JawCommandUnknown(
                "twin sync without a controller target needs the "
                "acknowledged jaw command effort; the measured aperture is "
                "plant state, not a command, and must not seed data.ctrl"
            )
        effort = float(acknowledged_jaw_effort_n)
        if not np.isfinite(effort) or not (
            float(ctrl_range[0]) <= effort <= float(ctrl_range[1])
        ):
            raise safety.SafetyError(
                "acknowledged jaw effort is outside the planning model range"
            )
        target = np.concatenate([
            measured_q,
            np.asarray([effort]),
        ])
    else:
        target = np.asarray(controller_target, dtype=float).reshape(-1)
    if target.shape != actuator_ids.shape:
        raise safety.SafetyError(
            "controller target must contain exactly "
            f"{len(actuator_ids)} actuator values, got {target.shape}"
        )
    if not np.all(np.isfinite(target)):
        raise safety.SafetyError("controller target contains NaN/Inf")
    target_ranges = np.asarray(
        world.model.actuator_ctrlrange[actuator_ids],
        dtype=float,
    )
    outside = np.argwhere(
        (target < target_ranges[:, 0] - 1e-9)
        | (target > target_ranges[:, 1] + 1e-9)
    )
    if len(outside):
        actuator = int(outside[0, 0])
        raise safety.SafetyError(
            f"controller target actuator {actuator}={target[actuator]!r} "
            f"is outside {target_ranges[actuator].tolist()}"
        )

    previous = np.asarray(world.data.qpos[arm_addr], dtype=float).copy()
    arm_delta = (
        float(np.max(np.abs(measured_q - previous)))
        if len(arm_addr)
        else 0.0
    )
    world.data.qpos[arm_addr] = measured_q
    world.data.qvel[arm_dof] = 0.0
    world.data.ctrl[actuator_ids] = target

    joint_order = (
        "finger_width",
        "left_outer",
        "left_inner",
        "left_finger",
        "right_outer",
        "right_inner",
        "right_finger",
    )
    targets = gripper_joint_targets(width)
    for key, value in zip(joint_order, targets):
        joint_id = int(robot["gripper_joint_ids"][key])
        qpos_addr = int(world.model.jnt_qposadr[joint_id])
        dof_addr = int(world.model.jnt_dofadr[joint_id])
        world.data.qpos[qpos_addr] = float(value)
        world.data.qvel[dof_addr] = 0.0
    # Host state is an observed kinematic mirror. Physics/contact is evaluated
    # only by the exact MJWarp rollout.
    mujoco.mj_kinematics(world.model, world.data)
    return arm_delta


def resync_robot_state(
    session: RobotSession,
    world: SimWorld,
    *,
    controller_target: np.ndarray | list[float] | None = None,
    acknowledged_jaw_effort_n: float | None = None,
) -> Any:
    """Re-sync the twin from one timestamped REAL arm+gripper measurement.

    The returned RobotState is the exact record written into the twin, so the
    observation/evidence path need not fetch a second, potentially different
    sample. Without an acknowledged ``controller_target``, the caller must
    still state the acknowledged jaw command; see
    :func:`sync_robot_state_to_twin`.
    """
    state = session.robot.get_state()
    validate_robot_state_telemetry(
        state,
        expected_joint_count=len(np.asarray(world.robot["qpos_addr"])),
    )
    width = getattr(state, "gripper_width", None)
    if width is None:
        raise safety.SafetyError(
            "live RobotState has no gripper_width; refusing stale-state planning"
        )
    arm_delta = sync_robot_state_to_twin(
        world,
        state.q,
        float(width),
        controller_target=controller_target,
        acknowledged_jaw_effort_n=acknowledged_jaw_effort_n,
    )
    logger.info(
        "[robot-resync] state <- live telemetry; control <- %s "
        "(max arm delta %.4f rad, width %.4f m)",
        (
            "measured first-command hold"
            if controller_target is None
            else "last acknowledged target"
        ),
        arm_delta,
        float(width),
    )
    return state


def resync_robot_qpos(session: RobotSession, world: SimWorld) -> float:
    """Re-sync only the twin arm from measured joints.

    Kept for explicit arm-only diagnostics. Production must use
    :func:`resync_robot_state`, because an arm-only update leaves stale gripper
    qpos/control and is not a complete MPC start state.
    """
    real_q = session.robot.get_state().q
    arm_delta = reset_robot_qpos(world, real_q)
    logger.info("[robot-resync] sim arm <- real q (max delta from prev %.4f rad)", arm_delta)
    return float(arm_delta)


def gpu_prefix_endpoint_fraction(
    horizon_steps: int,
    prefix_frac: float,
) -> float:
    """Return the normalized control sample selected at the GPU prefix.

    ``knots_to_ctrl`` samples the spline at ``H`` points including both
    endpoints. The rollout prefix after ``ceil(H * f)`` complete steps ends at
    control row ``ceil(H * f) - 1``, whose normalized spline coordinate is the
    row index divided by ``H - 1``.
    """
    if isinstance(horizon_steps, bool):
        raise safety.SafetyError("horizon_steps must be an integer >= 2")
    try:
        steps = int(horizon_steps)
        fraction = float(prefix_frac)
    except (TypeError, ValueError) as exc:
        raise safety.SafetyError("GPU prefix values must be finite") from exc
    if steps < 2 or steps != horizon_steps:
        raise safety.SafetyError("horizon_steps must be an integer >= 2")
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise safety.SafetyError("prefix_frac must be finite and in (0, 1]")
    raw_steps = steps * fraction
    nearest = round(raw_steps)
    prefix_steps = (
        int(nearest)
        if math.isclose(raw_steps, nearest, rel_tol=0.0, abs_tol=1e-12)
        else int(math.ceil(raw_steps))
    )
    return float(prefix_steps - 1) / float(steps - 1)


def joint_prefix_knot_times(
    knot_count: int,
    prefix_frac: float,
    *,
    horizon_steps: int | None = None,
) -> np.ndarray:
    """Return original spline breaks plus the selected GPU endpoint sample."""
    if isinstance(knot_count, bool):
        raise safety.SafetyError("knot_count must be a positive integer")
    try:
        count = int(knot_count)
    except (TypeError, ValueError) as exc:
        raise safety.SafetyError("knot_count must be a positive integer") from exc
    if count < 1 or count != knot_count:
        raise safety.SafetyError("knot_count must be a positive integer")
    try:
        requested_frac = float(prefix_frac)
    except (TypeError, ValueError) as exc:
        raise safety.SafetyError("prefix fraction must be finite and in (0, 1]") from exc
    if not np.isfinite(requested_frac) or not 0.0 < requested_frac <= 1.0:
        raise safety.SafetyError(
            f"prefix fraction must be finite and in (0, 1], got {prefix_frac}"
        )
    endpoint_frac = (
        requested_frac
        if horizon_steps is None
        else gpu_prefix_endpoint_fraction(horizon_steps, requested_frac)
    )
    if count == 1:
        return np.asarray([0.0], dtype=float)
    if endpoint_frac <= 0.0:
        return np.asarray([0.0], dtype=float)
    original = np.linspace(0.0, 1.0, count)
    if endpoint_frac == 1.0:
        return original
    interior = original[(original > 0.0) & (original < endpoint_frac)]
    return np.concatenate(
        (np.asarray([0.0]), interior, np.asarray([endpoint_frac]))
    )


def joint_prefix_knots(
    knots: np.ndarray,
    prefix_frac: float,
    *,
    horizon_steps: int | None = None,
) -> np.ndarray:
    """Preserve the GPU spline through its exact selected prefix control."""
    values = np.asarray(knots, dtype=float)
    if values.ndim != 2 or values.shape[1] != 8 or len(values) < 1:
        raise safety.SafetyError(
            f"joint trajectory must be finite (K, 8), got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise safety.SafetyError("joint trajectory contains NaN/Inf")
    original_t = np.linspace(0.0, 1.0, len(values))
    prefix_t = joint_prefix_knot_times(
        len(values),
        prefix_frac,
        horizon_steps=horizon_steps,
    )
    return np.stack(
        [
            np.interp(prefix_t, original_t, values[:, column])
            for column in range(values.shape[1])
        ],
        axis=1,
    )


def execution_prefix_schedule(
    *,
    horizon_steps: int,
    execution_dt_s: float,
    prefix_frac: float,
    prefix_knot_times: Sequence[float],
    control_hz: float,
) -> dict[str, Any]:
    """Map GPU prefix step boundaries to cumulative controller ticks.

    GPU prefix state/evidence is after ``ceil(H * t)`` complete physics steps.
    Every original spline break inside the prefix is retained. Controller tick
    boundaries are derived cumulatively from those GPU step boundaries. A
    bounded nearest-tick schedule is compatible when every cumulative boundary
    error is at most half a controller tick.
    """
    if isinstance(horizon_steps, bool):
        raise safety.SafetyError("horizon_steps must be an integer >= 2")
    try:
        steps = int(horizon_steps)
        dt_s = float(execution_dt_s)
        fraction = float(prefix_frac)
        rate_hz = float(control_hz)
    except (TypeError, ValueError) as exc:
        raise safety.SafetyError("prefix timing values must be finite numbers") from exc
    if steps < 2 or steps != horizon_steps:
        raise safety.SafetyError("horizon_steps must be an integer >= 2")
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise safety.SafetyError("execution_dt_s must be finite and positive")
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise safety.SafetyError("prefix_frac must be finite and in (0, 1]")
    if not np.isfinite(rate_hz) or rate_hz <= 0.0:
        raise safety.SafetyError("control_hz must be finite and positive")
    times = np.asarray(prefix_knot_times, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.all(np.isfinite(times))
        or abs(float(times[0])) > 1e-12
        or abs(
            float(times[-1])
            - gpu_prefix_endpoint_fraction(steps, fraction)
        ) > 1e-12
        or np.any(np.diff(times) <= 0.0)
    ):
        raise safety.SafetyError(
            "prefix_knot_times must increase from 0 to the selected GPU "
            "prefix control sample"
        )

    raw_step_boundaries = times * steps
    nearest_steps = np.rint(raw_step_boundaries)
    step_boundaries = np.where(
        np.isclose(
            raw_step_boundaries,
            nearest_steps,
            rtol=0.0,
            atol=1e-12,
        ),
        nearest_steps,
        np.ceil(raw_step_boundaries),
    ).astype(int)
    segment_gpu_steps = np.diff(step_boundaries)
    if np.any(segment_gpu_steps <= 0):
        raise safety.SafetyError(
            "GPU prefix has a spline segment shorter than one physics step"
        )
    required_cumulative_ticks = step_boundaries.astype(float) * dt_s * rate_hz
    rounded_cumulative_ticks = np.floor(required_cumulative_ticks + 0.5).astype(int)
    segment_n_frames = np.diff(rounded_cumulative_ticks)
    if np.any(segment_n_frames <= 0):
        raise safety.SafetyError(
            "controller prefix has a spline segment shorter than one controller tick"
        )
    timing_error_s = (
        rounded_cumulative_ticks.astype(float) / rate_hz
        - step_boundaries.astype(float) * dt_s
    )
    timing_exact = bool(
        np.allclose(
            required_cumulative_ticks,
            rounded_cumulative_ticks.astype(float),
            rtol=0.0,
            atol=1e-9,
        )
    )
    quantization_bound_s = 0.5 / rate_hz
    timing_compatible = bool(
        np.all(np.abs(timing_error_s) <= quantization_bound_s + 1e-12)
    )
    controller_ticks = int(rounded_cumulative_ticks[-1])
    gpu_prefix_steps = int(step_boundaries[-1])
    return {
        "requested_prefix_fraction": fraction,
        "gpu_prefix_endpoint_sample_index": gpu_prefix_steps - 1,
        "gpu_prefix_endpoint_spline_fraction": float(times[-1]),
        "continuous_horizon_duration_s": float(steps * dt_s),
        "continuous_requested_prefix_duration_s": float(steps * dt_s * fraction),
        "spline_prefix_knot_times": times.tolist(),
        "spline_segment_durations_s": (
            np.diff(times) * steps * dt_s
        ).tolist(),
        "gpu_cumulative_step_boundaries": step_boundaries.tolist(),
        "gpu_segment_steps": segment_gpu_steps.tolist(),
        "gpu_segment_durations_s": (segment_gpu_steps.astype(float) * dt_s).tolist(),
        "gpu_discrete_prefix_steps": gpu_prefix_steps,
        "gpu_discrete_prefix_duration_s": float(gpu_prefix_steps * dt_s),
        "controller_control_hz": rate_hz,
        "controller_required_cumulative_ticks": required_cumulative_ticks.tolist(),
        "controller_cumulative_ticks": rounded_cumulative_ticks.tolist(),
        "controller_timing_error_s": timing_error_s.tolist(),
        "controller_timing_exact": timing_exact,
        "controller_timing_compatible": timing_compatible,
        "controller_timing_quantization_bound_s": quantization_bound_s,
        "controller_prefix_ticks": controller_ticks,
        "controller_scheduled_prefix_duration_s": float(controller_ticks / rate_hz),
        "segment_n_frames": segment_n_frames.tolist(),
        "segment_durations_s": (segment_n_frames.astype(float) / rate_hz).tolist(),
    }


def effort_prefix_schedule(
    *,
    action_count: int,
    horizon_steps: int,
    execution_dt_s: float,
    prefix_frac: float,
    control_hz: float,
) -> dict[str, Any]:
    """Map zero-order-held MPPI efforts to real-controller segments.

    The GPU applies action ``floor(step * K / H)`` at native step ``step``.
    This helper retains exactly the actions touched by the executed native
    prefix and assigns each one the duration of those same physics steps.  The
    Flexiv adapter may interpolate between consecutive effort endpoints at
    1 kHz for the RDK smooth-command contract, but it must not reinterpret the
    actions as joint positions, velocities, or a ``K-1`` position spline.
    """
    if isinstance(action_count, bool) or int(action_count) != action_count:
        raise safety.SafetyError("action_count must be an integer >= 1")
    count = int(action_count)
    if count < 1:
        raise safety.SafetyError("action_count must be an integer >= 1")
    if isinstance(horizon_steps, bool) or int(horizon_steps) != horizon_steps:
        raise safety.SafetyError("horizon_steps must be an integer >= 2")
    steps = int(horizon_steps)
    if steps < 2:
        raise safety.SafetyError("horizon_steps must be an integer >= 2")
    try:
        dt_s = float(execution_dt_s)
        fraction = float(prefix_frac)
        rate_hz = float(control_hz)
    except (TypeError, ValueError) as exc:
        raise safety.SafetyError("effort prefix timing values must be numeric") from exc
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise safety.SafetyError("execution_dt_s must be finite and positive")
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise safety.SafetyError("prefix_frac must be finite and in (0, 1]")
    if not np.isfinite(rate_hz) or rate_hz <= 0.0:
        raise safety.SafetyError("control_hz must be finite and positive")

    prefix_steps = int(math.ceil(float(steps) * fraction - 1e-12))
    prefix_steps = max(1, min(prefix_steps, steps))
    indices = (np.arange(prefix_steps, dtype=int) * count) // steps
    starts = np.concatenate(
        (np.asarray([0], dtype=int), np.flatnonzero(np.diff(indices)) + 1)
    )
    ends = np.concatenate((starts[1:], np.asarray([prefix_steps], dtype=int)))
    action_indices = indices[starts]
    segment_steps = ends - starts
    cumulative_steps = np.cumsum(segment_steps)
    required_ticks = cumulative_steps.astype(float) * dt_s * rate_hz
    rounded_ticks = np.floor(required_ticks + 0.5).astype(int)
    segment_ticks = np.diff(
        np.concatenate((np.asarray([0], dtype=int), rounded_ticks))
    )
    if np.any(segment_ticks <= 0):
        raise safety.SafetyError(
            "controller prefix has an effort segment shorter than one tick"
        )
    timing_error_s = (
        rounded_ticks.astype(float) / rate_hz
        - cumulative_steps.astype(float) * dt_s
    )
    quantization_bound_s = 0.5 / rate_hz
    timing_compatible = bool(
        np.all(np.abs(timing_error_s) <= quantization_bound_s + 1e-12)
    )
    return {
        "requested_prefix_fraction": fraction,
        "gpu_prefix_endpoint_sample_index": prefix_steps - 1,
        "gpu_prefix_endpoint_spline_fraction": prefix_steps / float(steps),
        "continuous_horizon_duration_s": float(steps * dt_s),
        "continuous_requested_prefix_duration_s": float(steps * dt_s * fraction),
        "gpu_discrete_prefix_steps": prefix_steps,
        "gpu_discrete_prefix_duration_s": float(prefix_steps * dt_s),
        "gpu_action_indices": action_indices.tolist(),
        "gpu_segment_steps": segment_steps.tolist(),
        "gpu_segment_durations_s": (segment_steps.astype(float) * dt_s).tolist(),
        "controller_control_hz": rate_hz,
        "controller_required_cumulative_ticks": required_ticks.tolist(),
        "controller_cumulative_ticks": rounded_ticks.tolist(),
        "controller_timing_error_s": timing_error_s.tolist(),
        "controller_timing_exact": bool(
            np.allclose(required_ticks, rounded_ticks, rtol=0.0, atol=1e-9)
        ),
        "controller_timing_compatible": timing_compatible,
        "controller_timing_quantization_bound_s": quantization_bound_s,
        "controller_prefix_ticks": int(rounded_ticks[-1]),
        "controller_scheduled_prefix_duration_s": float(
            rounded_ticks[-1] / rate_hz
        ),
        "segment_n_frames": segment_ticks.tolist(),
        "segment_durations_s": (segment_ticks.astype(float) / rate_hz).tolist(),
        "real_effort_transition": "linear_1khz_between_zoh_endpoints",
    }


def binary_gripper_tick_knots(
    prefix_knots: np.ndarray,
    segment_n_frames: Sequence[int],
) -> np.ndarray:
    """Legacy helper: expand a latent spline to binary width commands.

    Production MPPI execution uses :func:`continuous_effort_tick_knots`.
    """
    from oap.twin.batched_rollout import width_command_ctrl
    from oap.twin.control_knots import NJ

    values = np.asarray(prefix_knots, dtype=float)
    frames = np.asarray(segment_n_frames, dtype=int)
    if (
        values.ndim != 2
        or values.shape[1] != NJ + 1
        or len(values) < 2
        or not np.all(np.isfinite(values))
    ):
        raise safety.SafetyError(
            f"prefix knots must be finite (K, {NJ + 1}) with K >= 2"
        )
    if frames.shape != (len(values) - 1,) or np.any(frames < 1):
        raise safety.SafetyError(
            "segment_n_frames must provide one positive tick count per segment"
        )

    rows = [values[0].copy()]
    rows[0][NJ] = float(width_command_ctrl(values[0, NJ]))
    for segment_index, tick_count in enumerate(frames):
        start = values[segment_index]
        end = values[segment_index + 1]
        for tick in range(1, int(tick_count) + 1):
            alpha = float(tick) / float(tick_count)
            row = start * (1.0 - alpha) + end * alpha
            row[NJ] = float(width_command_ctrl(row[NJ]))
            rows.append(row)
    return np.asarray(rows, dtype=float)


def continuous_effort_tick_knots(
    prefix_knots: np.ndarray,
    segment_n_frames: Sequence[int],
    *,
    pick_v8_causal_profile: str | None = None,
) -> np.ndarray:
    """Expand arm/jaw knots to controller ticks and map jaw to Newtons."""
    from oap.twin.batched_rollout import mppi_continuous_effort_ctrl
    from oap.twin.control_knots import NJ

    values = np.asarray(prefix_knots, dtype=float)
    frames = np.asarray(segment_n_frames, dtype=int)
    if (
        values.ndim != 2
        or values.shape[1] != NJ + 1
        or len(values) < 2
        or not np.all(np.isfinite(values))
    ):
        raise safety.SafetyError(
            f"prefix knots must be finite (K, {NJ + 1}) with K >= 2"
        )
    if frames.shape != (len(values) - 1,) or np.any(frames < 1):
        raise safety.SafetyError(
            "segment_n_frames must provide one positive tick count per segment"
        )
    rows = [values[0].copy()]
    for segment_index, tick_count in enumerate(frames):
        start = values[segment_index]
        end = values[segment_index + 1]
        for tick in range(1, int(tick_count) + 1):
            alpha = float(tick) / float(tick_count)
            rows.append(start * (1.0 - alpha) + end * alpha)
    expanded = np.asarray(rows, dtype=float)
    open_force_limit_n = None
    if pick_v8_causal_profile is not None:
        from oap.twin.pick_v8_causal import (
            PICK_V8_GRIPPER_ONLY_V2_OPEN_FORCE_LIMIT_N,
            pick_v8_causal_uses_asymmetric_effort,
        )

        if pick_v8_causal_uses_asymmetric_effort(pick_v8_causal_profile):
            open_force_limit_n = PICK_V8_GRIPPER_ONLY_V2_OPEN_FORCE_LIMIT_N
    expanded[:, NJ] = mppi_continuous_effort_ctrl(
        expanded[:, NJ], open_force_limit_n=open_force_limit_n
    )
    return expanded


def validate_controller_joint_path(
    joint_waypoints: np.ndarray,
    segment_durations_s: Sequence[float],
    effective_joint_limits: Mapping[str, Any],
    *,
    max_joint_speed_scale: float,
    start_joint_positions: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Fail closed before RPC if v3 position/slope limits would stretch."""
    q = np.asarray(joint_waypoints, dtype=float)
    if q.ndim != 2 or q.shape[1] < 1 or len(q) < 2 or not np.all(np.isfinite(q)):
        raise safety.SafetyError("controller joint path must be finite (K, J), K >= 2")
    durations = np.asarray(segment_durations_s, dtype=float)
    if (
        durations.shape != (len(q) - 1,)
        or not np.all(np.isfinite(durations))
        or np.any(durations <= 0.0)
    ):
        raise safety.SafetyError(
            "controller joint path needs one finite positive duration per segment"
        )
    limits = controller_effective_joint_limits(
        {"effective_joint_limits": effective_joint_limits},
        joint_count=q.shape[1],
    )
    try:
        scale = float(max_joint_speed_scale)
    except (TypeError, ValueError) as exc:
        raise safety.SafetyError("max_joint_speed_scale must be finite") from exc
    ceiling = float(limits["max_joint_speed_scale"])
    if not np.isfinite(scale) or not 0.0 < scale <= ceiling + 1e-12:
        raise safety.SafetyError(
            "requested max_joint_speed_scale exceeds the leased controller ceiling: "
            f"requested={scale!r}, ceiling={ceiling:.9g}"
        )
    lower = np.asarray(limits["enforced_position_min_rad"], dtype=float)
    upper = np.asarray(limits["enforced_position_max_rad"], dtype=float)
    bad_position = np.argwhere((q < lower) | (q > upper))
    if len(bad_position):
        waypoint, joint = (int(value) for value in bad_position[0])
        raise safety.SafetyError(
            f"controller joint waypoint {waypoint} joint {joint + 1}="
            f"{q[waypoint, joint]:.9g} outside leased enforced interval "
            f"[{lower[joint]:.9g}, {upper[joint]:.9g}] rad"
        )
    starts = q[:-1].copy()
    if start_joint_positions is not None:
        live = np.asarray(start_joint_positions, dtype=float)
        if live.shape != (q.shape[1],) or not np.all(np.isfinite(live)):
            raise safety.SafetyError(
                "live controller start joints must match the finite path dimension"
            )
        starts[0] = live
    requested_speed = np.abs(q[1:] - starts) / durations[:, None]
    allowed_speed = (
        np.asarray(limits["base_velocity_max_rad_s"], dtype=float) * scale
    )
    bad_speed = np.argwhere(requested_speed > allowed_speed[None, :] + 1e-12)
    if len(bad_speed):
        segment, joint = (int(value) for value in bad_speed[0])
        raise safety.SafetyError(
            f"controller joint segment {segment} joint {joint + 1} requires "
            f"{requested_speed[segment, joint]:.9g} rad/s but leased limit is "
            f"{allowed_speed[joint]:.9g} rad/s; refusing controller time-stretch"
        )
    ratio = requested_speed / allowed_speed[None, :]
    return {
        "effective_joint_limits_sha256": limits["sha256"],
        "max_joint_speed_scale": scale,
        "max_requested_joint_speed_rad_s": float(np.max(requested_speed)),
        "max_joint_speed_utilization": float(np.max(ratio)),
    }


def validate_controller_gripper_path(
    widths_m: Sequence[float],
    segment_durations_s: Sequence[float],
    gripper_limits: Mapping[str, Any],
    *,
    force_n: float,
) -> dict[str, Any]:
    """Fail closed before RPC if exact gripper timing exceeds runtime limits."""
    widths = np.asarray(widths_m, dtype=float)
    durations = np.asarray(segment_durations_s, dtype=float)
    if (
        widths.ndim != 1
        or len(widths) < 2
        or not np.all(np.isfinite(widths))
        or durations.shape != (len(widths) - 1,)
        or not np.all(np.isfinite(durations))
        or np.any(durations <= 0.0)
    ):
        raise safety.SafetyError(
            "controller gripper path needs finite widths and one positive "
            "duration per segment"
        )
    limits = controller_gripper_limits({"gripper_limits": gripper_limits})
    try:
        force = float(force_n)
    except (TypeError, ValueError) as exc:
        raise safety.SafetyError("gripper force must be finite") from exc
    if not np.isfinite(force) or not (
        limits["min_force_n"] <= force <= limits["max_force_n"]
    ):
        raise safety.SafetyError(
            f"gripper force {force!r} N is outside leased runtime interval "
            f"[{limits['min_force_n']}, {limits['max_force_n']}]"
        )
    bad_width = np.flatnonzero(
        (widths < limits["min_width_m"])
        | (widths > limits["max_width_m"])
    )
    if len(bad_width):
        index = int(bad_width[0])
        raise safety.SafetyError(
            f"gripper knot {index} width {widths[index]:.9g} m is outside "
            "leased runtime interval "
            f"[{limits['min_width_m']}, {limits['max_width_m']}]"
        )
    speeds = np.abs(np.diff(widths)) / durations
    moving = speeds > 1e-12
    invalid_speed = moving & (
        (speeds < limits["min_velocity_m_s"] - 1e-12)
        | (speeds > limits["max_velocity_m_s"] + 1e-12)
    )
    if np.any(invalid_speed):
        segment = int(np.flatnonzero(invalid_speed)[0])
        raise safety.SafetyError(
            f"gripper segment {segment} requires {speeds[segment]:.9g} m/s "
            "for exact timing but leased runtime interval is "
            f"[{limits['min_velocity_m_s']}, "
            f"{limits['max_velocity_m_s']}] m/s"
        )
    utilization = speeds / float(limits["max_velocity_m_s"])
    return {
        "gripper_limits_sha256": limits["sha256"],
        "gripper_force_n": force,
        "max_requested_gripper_speed_m_s": float(np.max(speeds)),
        "max_gripper_speed_utilization": float(np.max(utilization)),
    }


def validate_joint_trajectory(
    session: RobotSession,
    world: SimWorld,
    knots: np.ndarray,
    *,
    seg_duration_s: float | None = None,
    segment_durations_s: Sequence[float] | None = None,
    max_joint_speed_scale: float,
    effective_joint_limits: Mapping[str, Any],
    expected_grasp_site_flange_m: float | None = None,
) -> dict[str, Any]:
    """Client-side finite/range/FK workspace preflight for every waypoint.

    The planner's site is the pad-centre/grasp frame. The real controller's
    active RDK TCP is the calibrated closed-tip frame. They may differ (the
    stock sim site is 0.174 m; the live profile must explicitly patch it to
    0.19812 m), so both points are checked rather than comparing unlike frame
    names and either ignoring or double-applying the 24.12 mm offset.
    """
    import mujoco

    from oap.twin import REAL_GN01_TCP_M

    values = np.asarray(knots, dtype=float)
    if values.ndim != 2 or values.shape[1] != 8 or not len(values):
        raise safety.SafetyError(
            f"joint trajectory must be (K, 8), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise safety.SafetyError("joint trajectory contains NaN/Inf")
    if segment_durations_s is None:
        if seg_duration_s is None:
            raise safety.SafetyError(
                "joint waypoint durations are required"
            )
        durations = np.full(
            max(0, len(values) - 1),
            float(seg_duration_s),
            dtype=float,
        )
    else:
        durations = np.asarray(segment_durations_s, dtype=float)
        if durations.shape != (max(0, len(values) - 1),):
            raise safety.SafetyError(
                "joint waypoint durations must contain exactly one positive "
                "duration per future knot"
            )
    if (
        not len(durations)
        or not np.all(np.isfinite(durations))
        or np.any(durations <= 0.0)
    ):
        raise safety.SafetyError(
            "joint waypoint durations must be finite and positive"
        )
    q = values[:, :7]
    controller_path = validate_controller_joint_path(
        q,
        durations,
        effective_joint_limits,
        max_joint_speed_scale=max_joint_speed_scale,
    )
    ranges = np.asarray(world.robot["joint_ranges"], dtype=float)
    bad = np.argwhere((q < ranges[:, 0] - 1e-8)
                      | (q > ranges[:, 1] + 1e-8))
    if len(bad):
        waypoint, joint = (int(v) for v in bad[0])
        raise safety.SafetyError(
            f"joint waypoint {waypoint} joint {joint + 1}={q[waypoint, joint]:.4f} "
            f"outside [{ranges[joint, 0]:.4f}, {ranges[joint, 1]:.4f}] rad")
    gripper_actuator = int(world.robot["gripper_actuator_id"])
    gripper_range = np.asarray(
        world.model.actuator_ctrlrange[gripper_actuator],
        dtype=float,
    ).copy()
    if gripper_range.shape != (2,) or not np.all(np.isfinite(gripper_range)):
        raise safety.SafetyError(
            "planning model has no finite gripper control range"
        )
    gripper_efforts = values[:, 7]
    bad_efforts = np.argwhere(
        (gripper_efforts < gripper_range[0] - 1e-9)
        | (gripper_efforts > gripper_range[1] + 1e-9)
    )
    if len(bad_efforts):
        waypoint = int(bad_efforts[0, 0])
        raise safety.SafetyError(
            f"gripper waypoint {waypoint} force "
            f"{gripper_efforts[waypoint]:.5f} N is outside "
            f"[{gripper_range[0]:.5f}, {gripper_range[1]:.5f}] N"
        )

    m = world.model
    site_local_z = float(m.site_pos[int(world.site_id)][2])
    if (expected_grasp_site_flange_m is not None
            and abs(site_local_z - float(expected_grasp_site_flange_m)) > 5e-4):
        raise safety.SafetyError(
            f"compiled grasp-site frame is {site_local_z:.5f} m from flange, "
            f"expected {float(expected_grasp_site_flange_m):.5f} m (>0.5 mm); "
            "refusing execution")
    scratch = mujoco.MjData(m)
    scratch.qpos[:] = world.data.qpos
    qaddr = np.asarray(world.robot["qpos_addr"], dtype=int)
    max_site_offset = 0.0
    for waypoint, joint_q in enumerate(q):
        scratch.qpos[qaddr] = joint_q
        # Workspace validation needs FK only; never invoke host dynamics or
        # contact solving in the GPU-only production path.
        mujoco.mj_kinematics(m, scratch)
        grasp = np.asarray(scratch.site_xpos[int(world.site_id)], dtype=float)
        R = np.asarray(
            scratch.site_xmat[int(world.site_id)], dtype=float).reshape(3, 3)
        # Same tool axis, explicitly shifted to the RDK/closed-tip calibration.
        rdk_tcp = grasp + R[:, 2] * (REAL_GN01_TCP_M - site_local_z)
        for frame, point in (("grasp_site", grasp), ("rdk_tcp", rdk_tcp)):
            try:
                safety.check_waypoint_bounds(
                    point, ws_x=session.workspace_x, ws_y=session.workspace_y,
                    ws_z=session.workspace_z)
            except safety.SafetyError as exc:
                raise safety.SafetyError(
                    f"{frame} at joint waypoint {waypoint}: {exc}") from exc
        max_site_offset = max(
            max_site_offset, float(np.linalg.norm(rdk_tcp - grasp)))
    return {
        **controller_path,
        "n_waypoints": float(len(values)),
        "grasp_site_flange_m": site_local_z,
        "rdk_tcp_flange_m": float(REAL_GN01_TCP_M),
        "frame_offset_m": max_site_offset,
        "max_joint_speed_scale": float(max_joint_speed_scale),
        "segment_duration_min_s": float(np.min(durations)),
        "segment_duration_max_s": float(np.max(durations)),
        "gripper_effort_min_n": float(np.min(gripper_efforts)),
        "gripper_effort_max_n": float(np.max(gripper_efforts)),
    }


def observed_contact_proxies(
    world: SimWorld,
    *,
    target_name: str | None,
) -> dict[str, Any]:
    """Endpoint-only observed-twin clearance proxies (never contact evidence).

    Camera poses and measured joints hot-patch ``world`` before this runs. The
    returned distances are useful diagnostics, but an endpoint twin cannot tell
    which body caused a target's motion. Consequently
    ``tool_mediation_observable`` is always False; callers must route an
    indirect-tool verdict to CANNOT_VERIFY unless a real contact sensor/evidence
    provider supplies a stronger schema.
    """
    import mujoco

    result: dict[str, Any] = {
        "schema": "oap_observed_twin_contact_proxy_v1",
        "tool_mediation_observable": False,
        "reason": "endpoint twin proxy is not measured contact history",
        "target_name": target_name,
    }
    if not target_name:
        return result
    m, d = world.model, world.data
    target = mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_GEOM, f"support_{target_name}_collision")
    subject = mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_GEOM, "pick_object_collision")
    if target < 0:
        result["reason"] = f"target collision geom not found for {target_name!r}"
        return result
    from oap.twin.batched_rollout import exact_arm_collision_geoms

    robot = list(exact_arm_collision_geoms(m))
    robot.extend(
        gid
        for gid in range(m.ngeom)
        if (
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        ).startswith("gn01_")
        and (
            int(m.geom_contype[gid])
            or int(m.geom_conaffinity[gid])
        )
    )

    def distance(g1: int, g2: int) -> float | None:
        fromto = np.zeros(6)
        try:
            value = mujoco.mj_geomDistance(m, d, int(g1), int(g2), 1.0, fromto)
        except Exception:  # noqa: BLE001 - diagnostic only, never a gate pass
            return None
        return float(value)

    subject_distance = distance(subject, target) if subject >= 0 else None
    robot_distances = [v for v in (distance(g, target) for g in robot)
                       if v is not None]
    robot_distance = min(robot_distances) if robot_distances else None
    result.update({
        "subject_target_clearance_m": subject_distance,
        "robot_target_clearance_m": robot_distance,
        "subject_target_contact_proxy": (
            None if subject_distance is None else subject_distance <= 0.0),
        "robot_target_contact_proxy": (
            None if robot_distance is None else robot_distance <= 0.0),
    })
    return result


# --------------------------------------------------------------------------
# Sim replay of the executed chunk (the sim half of the evidence video)
# --------------------------------------------------------------------------
def replay_in_sim(world: SimWorld, knots: np.ndarray, renderer: Any,
                        frames_dir: Path, *, n_steps: int = 480,
                        spline_order: str = "linear", render_stride: int = 8,
                        prefix_frac: float = 1.0,
                        seed_obj_pose7: np.ndarray | None = None,
                        contact_out: dict[str, Any] | None = None,
                        mediation_target_bodies: tuple[str, ...] = (),
                        ) -> tuple[list[Path], np.ndarray]:
    """CPU test/video utility; never the production dry-run plant.

    Production adopts the selected MJWarp prefix state/contact directly.

    ``prefix_frac`` in (0, 1] executes only the FIRST fraction of the horizon --
    the receding-horizon "execute the plan's prefix, then re-observe and replan"
    step (the driver shifts the remaining plan by the same fraction). 1.0 runs
    the whole plan (single-shot chunk execution).

    The joint-space sibling of :func:`replay_chunk_in_sim`, and the M4 execution
    logic exercised in sim: NO execution-time IK and NO an execution chunk
    -- the (K, 8) knots are splined to per-step joint targets
    (:func:`~oap.twin.control_knots.knots_to_ctrl`) and written straight
    to ``data.ctrl`` and PD-tracked, exactly what the real flexiv joint executor
    (M4, on the robot) will stream. Rolls forward from the CURRENT ``world.data``
    (it does NOT re-seed the object -- the caller places it; note the
    GPU optimizer is side-effect free with respect to ``world.data``), leaves
    the twin in the
    post-execution state (the object ends where this chunk leaves it), renders
    every ``render_stride`` steps when a renderer + ZED camera are available, and
    returns ``(frames, final object pose7)`` -- the pose the next chunk plans from
    in a sim run (perception replaces this read on the real robot).
    ``contact_out`` is an optional caller-owned mapping populated with contact
    evidence from the commands that actually ran.  It keeps the public return
    shape stable while avoiding a process-global ``last_exec_contact`` side
    channel that no caller consumed. ``mediation_target_bodies`` is resolved
    structurally from program terminal anchors by the caller; contacts with
    other movable bodies remain diagnostics but are not causal task evidence.
    """
    import mujoco as mj

    from oap.twin.batched_rollout import mppi_continuous_effort_ctrl
    from oap.twin.control_knots import NJ, knots_to_ctrl
    from oap.twin.scene import set_arm_ctrl

    m, d = world.model, world.data
    oadr = int(world.object_qpos_addr)
    gaid = int(world.robot["gripper_actuator_id"])
    knots = np.asarray(knots, dtype=float)
    if seed_obj_pose7 is not None:
        # Place the CURRENT episode state before streaming (the receding-horizon
        # driver's re-observed object + the plan's start arm config), so a
        # preceding certify's world.data reset does not leak in.
        mj.mj_resetData(m, d)
        d.qpos[oadr:oadr + 7] = np.asarray(seed_obj_pose7, dtype=float)
        d.qpos[np.asarray(world.robot["qpos_addr"])] = knots[0, :NJ]
        mj.mj_forward(m, d)
    ctrl = knots_to_ctrl(knots, int(n_steps), order=spline_order)
    ctrl[:, NJ] = mppi_continuous_effort_ctrl(ctrl[:, NJ])
    n_exec = max(1, int(round(float(np.clip(prefix_frac, 1e-3, 1.0)) * len(ctrl))))
    ctrl = ctrl[:n_exec]                         # receding-horizon prefix
    has_cam = (renderer is not None
               and mj.mj_name2id(m, mj.mjtObj.mjOBJ_CAMERA, "zed2i_rgbd_sim") >= 0)
    frames: list[Path] = []
    if has_cam:
        from PIL import Image
        frames_dir.mkdir(parents=True, exist_ok=True)
    i = 0
    # NoSlip ON for the sim execution: the real gn01 grip holds only under the
    # complete solver (measured 2026-07-17), and the certifier certifies under
    # NoSlip -- streaming without it would slip the grasp in sim and diverge from
    # both the certifier and the real robot. Save/restore around the roll.
    _saved_noslip = int(m.opt.noslip_iterations)
    m.opt.noslip_iterations = _saved_noslip if _saved_noslip > 0 else 10
    # Contact evidence for what was EXECUTED, not what was planned. The tool
    # rule lives in the optimizer's validity, so it can refuse to PROPOSE a
    # plan that shoves the goal body with the gripper -- but the certifier
    # grades measured anchor positions and structurally cannot see contact, so
    # nothing checks that the plan which actually ran used the tool. Tally it
    # here and hand it to the caller, which persists and grades the cumulative
    # evidence before accepting a positional success verdict.
    from oap.twin.batched_rollout import (
        _movable_geom_groups,
        exact_arm_collision_geoms,
        _movable_and_robot_geoms,
    )
    _mov_g, _rob_g = _movable_and_robot_geoms(m)
    _mov_groups = _movable_geom_groups(m)
    _target_names = tuple(dict.fromkeys(
        str(name) for name in mediation_target_bodies if name
    ))
    _unknown_targets = sorted(set(_target_names) - set(_mov_groups))
    if _unknown_targets:
        raise ValueError(
            "execution mediation targets are not dynamic bodies in the twin: "
            f"{_unknown_targets}; available={sorted(_mov_groups)}"
        )
    _target_groups = {
        name: _mov_groups[name]
        for name in _target_names
    }
    # Execution and GPU planning use the same exact arm hull set. A link shove
    # counts at any real penetration depth; there is no proxy-geometry band.
    _rob_g |= set(exact_arm_collision_geoms(m))
    _subj_g = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "pick_object_collision")
    _left_pad_g = mj.mj_name2id(
        m,
        mj.mjtObj.mjOBJ_GEOM,
        "gn01_left_finger_tip_collision",
    )
    _right_pad_g = mj.mj_name2id(
        m,
        mj.mjtObj.mjOBJ_GEOM,
        "gn01_right_finger_tip_collision",
    )
    exec_contact = {"subject_movable_frames": 0, "robot_movable_frames": 0,
                    "max_subject_movable_pen_m": 0.0,
                    "max_robot_movable_pen_m": 0.0,
                    "mediation_target_bodies": list(_target_names),
                    "subject_target_frames_by_body": {
                        name: 0 for name in _target_names
                    },
                    "robot_target_frames_by_body": {
                        name: 0 for name in _target_names
                    },
                    "max_subject_target_pen_m_by_body": {
                        name: 0.0 for name in _target_names
                    },
                    "max_robot_target_pen_m_by_body": {
                        name: 0.0 for name in _target_names
                    },
                    "subject_two_sided_frames": 0,
                    "subject_two_sided_recent": 0,
                    "subject_two_sided_now": False}
    held_recent: list[bool] = []
    try:
        for step, u in enumerate(ctrl):
            set_arm_ctrl(d, world.robot, u[:NJ])
            d.ctrl[gaid] = float(u[NJ])
            mj.mj_step(m, d)
            _sm = _rm = False
            _sm_targets: set[str] = set()
            _rm_targets: set[str] = set()
            _left = _right = False
            for _c in range(d.ncon):
                _con = d.contact[_c]
                _pair = {int(_con.geom1), int(_con.geom2)}
                _touching = float(_con.dist) <= 0.0
                if _touching and _pair == {_subj_g, _left_pad_g}:
                    _left = True
                elif _touching and _pair == {_subj_g, _right_pad_g}:
                    _right = True
                if not _touching or not (_mov_g & _pair):
                    continue
                _pen = max(0.0, -float(_con.dist))
                _contact_targets = {
                    name
                    for name, geoms in _target_groups.items()
                    if geoms & _pair
                }
                if _subj_g in _pair:
                    _sm = True
                    exec_contact["max_subject_movable_pen_m"] = max(
                        exec_contact["max_subject_movable_pen_m"], _pen)
                    for _target in _contact_targets:
                        _sm_targets.add(_target)
                        values = exec_contact[
                            "max_subject_target_pen_m_by_body"
                        ]
                        values[_target] = max(values[_target], _pen)
                elif _rob_g & _pair:
                    _rm = True
                    exec_contact["max_robot_movable_pen_m"] = max(
                        exec_contact["max_robot_movable_pen_m"], _pen)
                    for _target in _contact_targets:
                        _rm_targets.add(_target)
                        values = exec_contact[
                            "max_robot_target_pen_m_by_body"
                        ]
                        values[_target] = max(values[_target], _pen)
            exec_contact["subject_movable_frames"] += int(_sm)
            exec_contact["robot_movable_frames"] += int(_rm)
            for _target in _sm_targets:
                exec_contact[
                    "subject_target_frames_by_body"
                ][_target] += 1
            for _target in _rm_targets:
                exec_contact[
                    "robot_target_frames_by_body"
                ][_target] += 1
            held_now = bool(_left and _right)
            exec_contact["subject_two_sided_frames"] += int(held_now)
            held_recent.append(held_now)
            held_recent = held_recent[-5:]
            if has_cam and step % max(1, int(render_stride)) == 0:
                renderer.update_scene(d, camera="zed2i_rgbd_sim")
                p = frames_dir / f"frame_{i:05d}.png"
                Image.fromarray(renderer.render()).save(p)
                frames.append(p)
                i += 1
    finally:
        m.opt.noslip_iterations = _saved_noslip
    exec_contact["subject_two_sided_recent"] = int(sum(held_recent))
    exec_contact["subject_two_sided_now"] = bool(
        held_recent[-1] if held_recent else False
    )
    if exec_contact["subject_movable_frames"] or exec_contact["robot_movable_frames"]:
        logger.info("[replay] executed contact with movable bodies: subject %d "
                    "frames (max %.2f mm), robot %d frames (max %.2f mm)",
                    exec_contact["subject_movable_frames"],
                    1000 * exec_contact["max_subject_movable_pen_m"],
                    exec_contact["robot_movable_frames"],
                    1000 * exec_contact["max_robot_movable_pen_m"])
    if contact_out is not None:
        contact_out.clear()
        contact_out.update(exec_contact)
    return frames, np.asarray(d.qpos[oadr:oadr + 7], dtype=float).copy()


def _measured_robot_state(state: Any) -> dict[str, Any]:
    """Extract only fields supplied by a timestamped RobotState snapshot."""
    q = getattr(state, "q", None)
    tcp = getattr(state, "tcp_pose", None)
    width = getattr(state, "gripper_width", None)
    force = getattr(state, "gripper_force", None)
    stamp = getattr(state, "stamp", None)
    return {
        "stamp_s": None if stamp is None else float(stamp),
        "joint_positions": (
            None if q is None else [float(v) for v in np.asarray(q).ravel()]),
        "tcp_pose_base": (
            None if tcp is None else [float(v) for v in np.asarray(tcp).ravel()]),
        "gripper_width_m": None if width is None else float(width),
        "gripper_force_n": None if force is None else float(force),
    }


def _post_stop_capture_follows_endpoint(
    capture_monotonic_s: object,
    endpoint_state_received_monotonic_s: object,
) -> bool:
    """Compare endpoint/camera ordering only inside the host monotonic clock."""
    try:
        capture = float(capture_monotonic_s)
        endpoint = float(endpoint_state_received_monotonic_s)
    except (TypeError, ValueError):
        return False
    return bool(
        np.isfinite(capture)
        and np.isfinite(endpoint)
        and capture >= endpoint
    )


def execute_joint_trajectory(
    session: RobotSession,
    cfg: "LoopConfig",
    knots: np.ndarray,
    *,
    action_plan: np.ndarray | None = None,
    chunk_idx: int,
    out_dir: Path,
    plan_world: SimWorld,
    prefix_frac: float = 1.0,
    horizon_steps: int | None = None,
    execution_dt_s: float | None = None,
    max_joint_speed_scale: float = 0.3,
    camera_client: Any | None = None,
    checkpoint_objects: list[tuple[str, str]] | None = None,
    reserve_checkpoint: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Execute one selected effort plan on the real arm (M4).

    *** NOT YET VERIFIED ON HARDWARE -- bring up with the e-stop in hand. ***

    The selected ``(K, 8)`` action plan is seven normalized Rizon joint torques
    plus one normalized GN01 force.  No desired arm pose, velocity, gripper
    width, IK target, waypoint, or hidden grasp state is reconstructed here.
    The seven arm components are mapped through the fixed paper profile's
    physical torque envelope; the last component maps linearly to signed
    native gripper force.  One strict RPC carries the previous acknowledged
    torque command and every retained endpoint with exact controller frame
    counts.  flexiv-control interpolates only between effort endpoints at
    1 kHz, checks ``RobotInfo.tau_max`` and the active torque scale, and calls
    the native gravity-compensated/soft-limited torque stream.  The GN01 call
    is native ``Gripper.Grasp(force)``; width is telemetry only.

    Execution remains gated by the consumed human authorization, an attended
    hardware-verification latch, controller/runtime identity, timestamped
    state telemetry, force/torque envelopes, and the per-chunk confirmation.

    Returns an execution record analogous to :func:`execute_chunk`'s.
    """
    # Recheck the consumed capability before every motion prefix.  This also
    # prevents a long-running episode from continuing after token expiry.
    require_consumed_execution_authorization(cfg)
    try:
        from flexiv_control import (
            JointGripperForceTarget,
            JointTorqueTrajectory,
            JointTorqueWaypoint,
        )
    except ImportError as exc:
        raise safety.SafetyError(
            "installed flexiv-control lacks the verified joint trajectory "
            "API; refusing real motion") from exc

    from oap.twin.control_knots import NJ

    safety.require_joint_executor_readiness(
        execute=True,
        prepare_real_execution=bool(
            getattr(cfg, "prepare_real_execution", False)),
        hardware_verified=bool(
            getattr(cfg, "joint_executor_hardware_verified", False)),
        verification_latch_verified=bool(
            getattr(
                cfg,
                "joint_executor_verification_latch_verified",
                False,
            )
        ),
    )
    capabilities = dict(getattr(session, "extra", {}) or {})
    if (
        capabilities.get("execution_authorization_token_id")
        != getattr(cfg, "execution_authorization_token_id", None)
        or capabilities.get("execution_authorization_token_sha256")
        != getattr(cfg, "execution_authorization_token_sha256", None)
    ):
        raise safety.SafetyError(
            "real joint execution is blocked: RobotSession and LoopConfig "
            "execution-authorization identities differ"
        )
    if not capabilities.get("verification_latch_verified", False):
        raise safety.SafetyError(
            "real joint execution is blocked: the active RobotSession was not "
            "created under a valid human verification latch"
        )
    if (
        capabilities.get("executor_software_fingerprint_sha256")
        != getattr(
            cfg,
            "joint_executor_software_fingerprint_sha256",
            None,
        )
    ):
        raise safety.SafetyError(
            "real joint execution is blocked: RobotSession and LoopConfig "
            "flexiv-control fingerprints differ"
        )
    if not capabilities.get("joint_telemetry_verified", False):
        raise safety.SafetyError(
            "real joint execution is blocked until timestamped joint/gripper "
            "RobotState telemetry is verified end-to-end")
    if not capabilities.get("joint_torque_stream_verified", False):
        raise safety.SafetyError(
            "real joint execution is blocked: this flexiv-control deployment "
            "has not verified RT_JOINT_TORQUE streaming and torque-scale "
            "enforcement on the attended hardware")
    if not hasattr(session.robot, "execute_joint_torque_trajectory"):
        raise safety.SafetyError(
            "active flexiv-control client has no verified "
            "execute_joint_torque_trajectory endpoint")
    if horizon_steps is None or execution_dt_s is None:
        raise safety.SafetyError(
            "real joint execution requires the signed GPU horizon_steps and "
            "execution_dt_s; refusing an inferred waypoint duration"
        )
    carrier_knots = np.asarray(knots, dtype=float)
    if (
        carrier_knots.ndim != 2
        or carrier_knots.shape[1] != NJ + 1
        or len(carrier_knots) < 1
        or not np.all(np.isfinite(carrier_knots))
    ):
        raise safety.SafetyError("realized carrier must be finite (K, 8)")
    try:
        controller_control_hz = float(
            capabilities["controller_control_hz"]
        )
        controller_control_hz_source = str(
            capabilities["controller_control_hz_source"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise safety.SafetyError(
            "real joint execution has no leased top-level controller "
            "control_hz binding"
        ) from exc
    if (
        controller_control_hz_source != "control_hz"
        or not np.isfinite(controller_control_hz)
        or controller_control_hz <= 0.0
    ):
        raise safety.SafetyError(
            "real joint execution requires a finite positive control_hz "
            "bound from top-level server_info.control_hz"
        )
    if action_plan is None:
        raise safety.SafetyError(
            "real MPPI torque execution requires the exact selected action plan"
        )
    action_plan = np.asarray(action_plan, dtype=float)
    if (
        action_plan.shape != carrier_knots.shape
        or not np.all(np.isfinite(action_plan))
    ):
        raise safety.SafetyError(
            "selected torque action plan must match the full finite (K,8) carrier"
        )
    prefix_timing = effort_prefix_schedule(
        action_count=len(action_plan),
        horizon_steps=horizon_steps,
        execution_dt_s=execution_dt_s,
        prefix_frac=prefix_frac,
        control_hz=controller_control_hz,
    )
    if not prefix_timing["controller_timing_compatible"]:
        raise safety.SafetyError(
            "GPU prefix/controller cumulative timing quantization exceeds "
            "half a controller tick; refusing dispatch"
        )
    segment_n_frames = [
        int(value) for value in prefix_timing["segment_n_frames"]
    ]
    action_indices = np.asarray(
        prefix_timing["gpu_action_indices"], dtype=int
    )
    knots = action_plan[action_indices]
    raw_planned_gripper_latents = knots[:, NJ].copy()
    logger.info(
        "[real-joint-timing] requested=%.6fs; GPU=%d steps/%.6fs; "
        "controller=%d ticks/%.6fs @ %.9gHz; segments=%s",
        prefix_timing["continuous_requested_prefix_duration_s"],
        prefix_timing["gpu_discrete_prefix_steps"],
        prefix_timing["gpu_discrete_prefix_duration_s"],
        prefix_timing["controller_prefix_ticks"],
        prefix_timing["controller_scheduled_prefix_duration_s"],
        prefix_timing["controller_control_hz"],
        segment_n_frames,
    )
    try:
        effective_joint_limits = capabilities[
            "controller_effective_joint_limits"
        ]
    except KeyError as exc:
        raise safety.SafetyError(
            "real joint execution has no leased effective_joint_limits"
        ) from exc
    del plan_world, max_joint_speed_scale
    torque_max = np.asarray(
        effective_joint_limits.get("base_torque_max_nm"), dtype=float
    )
    torque_scale = float(
        effective_joint_limits.get("max_joint_torque_scale", 0.0)
    )
    if (
        torque_max.shape != (NJ,)
        or not np.all(np.isfinite(torque_max))
        or np.any(torque_max <= 0.0)
        or not 0.0 < torque_scale <= 1.0
    ):
        raise safety.SafetyError(
            "leased controller has no valid joint torque limit contract"
        )
    from oap.twin.batched_rollout import RIZON4S_MPC_TORQUE_LIMIT_NM

    fixed_torque_limit = np.asarray(RIZON4S_MPC_TORQUE_LIMIT_NM, dtype=float)
    if np.any(torque_max * torque_scale + 1e-12 < fixed_torque_limit):
        raise safety.SafetyError(
            "leased controller torque envelope is below the fixed project MPC "
            "envelope; refusing a silent hardware-specific rescale"
        )
    arm_torques_nm = knots[:, :NJ] * fixed_torque_limit[None, :]
    try:
        gripper_limits = capabilities["controller_gripper_limits"]
    except KeyError as exc:
        raise safety.SafetyError(
            "real joint execution has no leased gripper_limits"
        ) from exc
    efforts_n = knots[:, NJ] * 80.0
    min_force_n = float(gripper_limits["min_force_n"])
    max_force_n = float(gripper_limits["max_force_n"])
    if (
        not np.all(np.isfinite(efforts_n))
        or np.any(efforts_n < min_force_n)
        or np.any(efforts_n > max_force_n)
    ):
        raise safety.SafetyError(
            "sampled GN01 force path is outside controller runtime limits"
        )
    preflight = {
        "arm_action_mode": "gravity_compensated_direct_joint_torque",
        "arm_torque_range_nm": [
            np.min(arm_torques_nm, axis=0).tolist(),
            np.max(arm_torques_nm, axis=0).tolist(),
        ],
        "gripper_action_mode": "signed_direct_force",
        "gripper_force_range_n": [
            float(np.min(efforts_n)),
            float(np.max(efforts_n)),
        ],
    }
    if cfg.per_chunk_confirm and not safety.confirm_chunk(chunk_idx):
        raise KeyboardInterrupt("user declined chunk execution")
    # The operator can wait at the prompt; never dispatch under a capability
    # that expired while they were checking the workspace or E-stop.
    require_consumed_execution_authorization(cfg)
    profile = session.active_profile.name
    previous_torque = np.asarray(
        capabilities.get(
            "last_acknowledged_joint_torque_nm",
            np.zeros(NJ, dtype=float),
        ),
        dtype=float,
    )
    if previous_torque.shape != (NJ,) or not np.all(np.isfinite(previous_torque)):
        raise safety.SafetyError(
            "RobotSession has no valid previous acknowledged joint torque"
        )
    trajectory = JointTorqueTrajectory(
        initial_torques=previous_torque,
        waypoints=[
            JointTorqueWaypoint(
                torques=np.asarray(arm_torques_nm[index], dtype=float),
                n_frames=segment_n_frames[index],
                gripper=JointGripperForceTarget(force=float(efforts_n[index])),
            )
            for index in range(len(knots))
        ],
        max_joint_torque_scale=float(
            np.max(fixed_torque_limit / torque_max)
        ),
        safety_profile=profile,
    )

    if camera_client is None:
        raise safety.SafetyError(
            "real joint execution requires the episode-long single-owner "
            "ZED stream; refusing to open a competing per-prefix camera"
        )
    if not checkpoint_objects:
        raise safety.SafetyError(
            "real joint execution requires canonical object labels for the "
            "same-frame post-stop checkpoint packet"
        )
    rec_dir = out_dir / "real_video" / f"chunk_{chunk_idx:02d}"
    checkpoint_dir = rec_dir / "post_stop_packet"
    if rec_dir.exists():
        raise safety.SafetyError(
            f"real-video chunk directory already exists: {rec_dir}; "
            "refusing to overwrite motion/checkpoint evidence"
        )
    camera_client.start_recording(out_dir=rec_dir)
    motion_start_unix_s = time.time()
    motion_start = time.monotonic()
    state_before = None
    state_after = None
    endpoint_state_received_monotonic_s: float | None = None
    checkpoint_reserved_ok = reserve_checkpoint is None
    result: Any | None = None
    aborted = False
    try:
        state_before = session.robot.get_state()
        validate_robot_state_telemetry(
            state_before,
            expected_joint_count=NJ,
        )
        # One atomic strict RPC carries explicit target knot 0, every future arm
        # waypoint, cumulative n_frames-derived timing, and the synchronized
        # gripper target at each segment boundary. No blocking gripper RPC may
        # serialize arm and hand motion.
        result = session.robot.execute_joint_torque_trajectory(trajectory)
        if bool(result.success) and not bool(result.clipped):
            acknowledged = np.asarray(
                result.log.get("acknowledged_ending_joint_torque_nm"),
                dtype=float,
            )
            if acknowledged.shape != (NJ,) or not np.all(np.isfinite(acknowledged)):
                try:
                    session.robot.stop()
                finally:
                    session.extra["last_acknowledged_joint_torque_nm"] = (
                        np.zeros(NJ, dtype=float).tolist()
                    )
                raise safety.SafetyError(
                    "controller omitted the acknowledged ending torque command"
                )
            session.extra["last_acknowledged_joint_torque_nm"] = (
                acknowledged.tolist()
            )
        if not bool(result.success) or bool(result.clipped):
            aborted = True
            try:
                session.robot.stop()
            except Exception:  # noqa: BLE001 - already stopped/degraded
                logger.warning(
                    "[real-joint] defensive stop RPC failed after atomic prefix "
                    "abort", exc_info=True)
            finally:
                session.extra["last_acknowledged_joint_torque_nm"] = (
                    np.zeros(NJ, dtype=float).tolist()
                )
    finally:
        motion_end = time.monotonic()
        motion_end_unix_s = time.time()
        # Freeze one endpoint robot state before asking the sole camera owner
        # for its post-stop exposure.  The returned RGB-D packet therefore
        # provably succeeds this exact telemetry sample, and the runner reuses
        # it instead of taking a later, mismatched RobotState.
        try:
            state_after = session.robot.get_state()
            validate_robot_state_telemetry(
                state_after,
                expected_joint_count=NJ,
            )
            endpoint_state_received_monotonic_s = time.monotonic()
        except Exception:  # noqa: BLE001 - motion ended; still close camera
            logger.exception(
                "[real-joint] endpoint RobotState unavailable before "
                "post-stop camera grab"
            )
        if state_after is not None and reserve_checkpoint is not None:
            try:
                reserve_checkpoint()
                checkpoint_reserved_ok = True
            except Exception:  # noqa: BLE001 - still close camera/evidence
                logger.exception(
                    "[real-joint] could not reserve perception checkpoint"
                )
        try:
            recording_result = camera_client.stop_recording(
                checkpoint_out_dir=checkpoint_dir,
                objects=checkpoint_objects,
            )
        except Exception:  # noqa: BLE001 - movement already ended; fail closed
            recording_result = {
                "ok": False,
                "error": "single-owner ZED recording stop failed",
            }
            logger.exception(
                "[real-joint] failed to close single-owner ZED evidence window"
            )
    assert state_before is not None
    if state_after is None:
        raise safety.SafetyError(
            "real prefix ended without valid endpoint RobotState; "
            "post-stop observation cannot be synchronized"
        )
    coverage = _recording_coverage(
        rec_dir, motion_start_monotonic_s=motion_start,
        motion_end_monotonic_s=motion_end,
        returncode=(0 if recording_result.get("ok") else 1))
    coverage["camera_owner"] = "episode_zed_stream"
    coverage["stream_attestation"] = dict(recording_result)
    result_log_raw = getattr(result, "log", {}) if result is not None else {}
    result_log = (
        dict(result_log_raw) if isinstance(result_log_raw, Mapping) else {}
    )
    requested_ticks = result_log.get("requested_segment_ticks")
    scheduled_ticks = result_log.get("scheduled_segment_ticks")
    timing_acknowledged = (
        requested_ticks == segment_n_frames
        and scheduled_ticks == segment_n_frames
        and result_log.get("requested_total_ticks")
        == prefix_timing["controller_prefix_ticks"]
        and result_log.get("scheduled_total_ticks")
        == prefix_timing["controller_prefix_ticks"]
        and bool(result_log.get("strict_timing", False))
    )
    expected_final_torque = np.asarray(arm_torques_nm[-1], dtype=float)
    acknowledged_final_torque = np.asarray(
        result_log.get("acknowledged_ending_joint_torque_nm", []),
        dtype=float,
    )
    acknowledged_final_force = result_log.get("ending_gripper_force_n")
    target_acknowledged = bool(
        acknowledged_final_torque.shape == expected_final_torque.shape
        and np.all(np.isfinite(acknowledged_final_torque))
        and np.allclose(
            acknowledged_final_torque,
            expected_final_torque,
            rtol=0.0,
            atol=1e-12,
        )
        and acknowledged_final_force is not None
        and np.isfinite(float(acknowledged_final_force))
        and abs(float(acknowledged_final_force) - float(efforts_n[-1])) <= 1e-12
    )
    clipped = bool(result is not None and bool(result.clipped))
    ok = bool(
        result is not None
        and bool(result.success)
        and not clipped
        and not aborted
        and timing_acknowledged
        and target_acknowledged
        and coverage["observable"]
    )
    if not ok:
        logger.warning(
            "[real-joint] WARNING: degraded atomic execution "
            "(clipped=%s timing_ack=%s target_ack=%s) -- controller evidence "
            "does not match the certified prefix",
            clipped,
            timing_acknowledged,
            target_acknowledged,
        )
    before_row = _measured_robot_state(state_before)
    after_row = _measured_robot_state(state_after)
    post_stop_capture_raw = recording_result.get(
        "post_stop_capture_monotonic_s"
    )
    post_stop_after_endpoint_state = _post_stop_capture_follows_endpoint(
        post_stop_capture_raw,
        endpoint_state_received_monotonic_s,
    )
    coverage["post_stop_after_endpoint_state"] = (
        post_stop_after_endpoint_state
    )
    if not post_stop_after_endpoint_state or not checkpoint_reserved_ok:
        ok = False
    final_joint_tracking_error_rad = None
    last_gripper_target = float(efforts_n[-1])
    last_acknowledged_target = (
        np.concatenate([
            np.asarray(after_row["joint_positions"], dtype=float),
            np.asarray([last_gripper_target], dtype=float),
        ]).tolist()
        if ok
        else None
    )
    raw_gripper_events = result_log.get("gripper_events", [])
    gripper_events = (
        [dict(event) for event in raw_gripper_events]
        if isinstance(raw_gripper_events, list)
        and all(isinstance(event, Mapping) for event in raw_gripper_events)
        else []
    )
    reported_duration_raw = getattr(result, "executed_duration", None)
    try:
        reported_duration = float(reported_duration_raw)
    except (TypeError, ValueError):
        reported_duration = None
    if reported_duration is not None and (
        not np.isfinite(reported_duration) or reported_duration < 0.0
    ):
        reported_duration = None
    path_tracking_error_raw = getattr(result, "path_tracking_error", None)
    try:
        path_tracking_error_rad = float(path_tracking_error_raw)
    except (TypeError, ValueError):
        path_tracking_error_rad = None
    if path_tracking_error_rad is not None and (
        not np.isfinite(path_tracking_error_rad)
        or path_tracking_error_rad < 0.0
    ):
        path_tracking_error_rad = None
    controller_result_timing = [{
        "requested_segment_ticks": requested_ticks,
        "scheduled_segment_ticks": scheduled_ticks,
        "requested_total_ticks": result_log.get("requested_total_ticks"),
        "scheduled_total_ticks": result_log.get("scheduled_total_ticks"),
        "controller_reported_execution_duration_s": reported_duration,
        "timing_acknowledged": timing_acknowledged,
        "target_acknowledged": target_acknowledged,
        "controller_log": result_log,
    }]
    controller_reported_arm_execution_duration_s = reported_duration
    return {
        "schema": "oap_real_joint_execution_v1",
        "execution_source": "real_joint",
        "success": ok,
        "clipped": clipped,
        "n_segments": len(segment_n_frames),
        "n_controller_rpcs": 1 if result is not None else 0,
        "stop_reason": (
            "video_coverage_unverified" if not coverage["observable"]
            else str(result.stop_reason) if result is not None else "no_result"),
        "executed_duration_s": reported_duration,
        "prefix_timing": prefix_timing,
        "controller_result_timing": controller_result_timing,
        "controller_reported_arm_execution_duration_s":
            controller_reported_arm_execution_duration_s,
        "host_motion_window_duration_s": float(motion_end - motion_start),
        "path_tracking_error_m": None,
        "path_tracking_error_rad": path_tracking_error_rad,
        "final_joint_tracking_error_rad":
            final_joint_tracking_error_rad,
        "gripper_width_final_m": after_row["gripper_width_m"],
        "gripper_force_final_n": after_row["gripper_force_n"],
        "gripper_measurement_stamp_s": after_row["stamp_s"],
        "final_joint_positions": after_row["joint_positions"],
        "measured_robot_state_before": before_row,
        "measured_robot_state_after": after_row,
        "prefix_anchor_joint_positions": before_row["joint_positions"],
        "commanded_joint_torques_nm": arm_torques_nm.tolist(),
        "commanded_gripper_efforts_n": efforts_n.tolist(),
        "planned_gripper_latent_knots": (
            raw_planned_gripper_latents.tolist()
        ),
        "gripper_command_mapping": {
            "latent_positive": "positive_closing_force",
            "latent_negative": "negative_opening_force",
            "latent_zero": "zero_force",
            "mapping": "force_n=clip(latent,-1,1)*80",
            "mapping_point": "after_interpolation_per_controller_tick",
        },
        "last_commanded_gripper_effort_n": last_gripper_target,
        # Only the server's atomic target acknowledgement advances the
        # controller-target boundary used by the next MPC cycle.
        "last_acknowledged_controller_target": last_acknowledged_target,
        "gripper_events": gripper_events,
        "video_dir": str(rec_dir),
        "video_coverage": coverage,
        "motion_start_unix_s": float(motion_start_unix_s),
        "motion_end_unix_s": float(motion_end_unix_s),
        "post_stop_observation_packet": recording_result.get(
            "checkpoint_out_dir"
        ),
        "post_stop_capture_unix_s": recording_result.get(
            "post_stop_capture_unix_s"
        ),
        "post_stop_capture_monotonic_s": recording_result.get(
            "post_stop_capture_monotonic_s"
        ),
        "endpoint_state_received_monotonic_s":
            endpoint_state_received_monotonic_s,
        "post_stop_capture_id": recording_result.get("capture_id"),
        "post_stop_rgb_sha256": recording_result.get(
            "post_stop_rgb_sha256"
        ),
        "post_stop_depth_m_sha256": recording_result.get(
            "post_stop_depth_m_sha256"
        ),
        "post_stop_sidecar": recording_result.get(
            "post_stop_sidecar"
        ),
        "contact_measurement": None,
        "tool_mediation_observable": False,
        "tool_mediation_source": "unobserved",
        "tool_mediation_coverage": None,
        "preflight": preflight,
        "telemetry_source": capabilities.get(
            "joint_telemetry_source", "timestamped_robot_state"),
        "joint_telemetry_observable": True,
        "joint_telemetry_contract": capabilities.get(
            "joint_telemetry_contract"
        ),
        "joint_speed_scale_verified": bool(
            capabilities.get("joint_speed_scale_verified", False)
        ),
        "joint_torque_stream_verified": bool(
            capabilities.get("joint_torque_stream_verified", False)
        ),
        "executor_verification_latch_sha256": capabilities.get(
            "verification_latch_sha256"
        ),
        "executor_software_fingerprint_sha256": capabilities.get(
            "executor_software_fingerprint_sha256"
        ),
        "hardware_verified": bool(
            getattr(cfg, "joint_executor_hardware_verified", False)),
    }


# --------------------------------------------------------------------------
# End-of-run rituals
# --------------------------------------------------------------------------
def stop_and_record_manual_restore(
    session: RobotSession,
    *,
    out_dir: Path,
    held_close_width_m: float | None = None,
    held_status: bool | None = None,
    close_was_commanded: bool | None = None,
) -> None:
    """Stop and leave the measured posture for an operator to restore.

    This is the explicit ``--manual-post-experiment-restore`` path. It never
    sends lift, gripper, Cartesian, joint, or home commands. It requests the
    controller's stop primitive, reads one final timestamped state, and writes
    ``final_posture.json`` next to the verified ``initial_posture.json``. The
    subsequent lease release independently stops the backend again.

    Cleanup must not mask the episode result, so failures are recorded and
    logged instead of raised.
    """
    stop_acknowledged = False
    stop_error: str | None = None
    try:
        session.robot.stop()
        stop_acknowledged = True
    except Exception as exc:  # noqa: BLE001 - cleanup must preserve evidence
        stop_error = f"{type(exc).__name__}: {exc}"
        logger.error(
            "[manual-restore] stop RPC failed; lease release will issue the "
            "independent backend stop: %s",
            stop_error,
        )

    state_error: str | None = None
    state_payload: dict[str, Any] | None = None
    try:
        state = session.robot.get_state()
        telemetry = validate_robot_state_telemetry(
            state,
            expected_joint_count=len(session.home_q),
        )
        state_payload = {
            "q_rad": [float(v) for v in state.q],
            "tcp_pose": [float(v) for v in state.tcp_pose],
            "robot_state_stamp_s": telemetry["stamp_s"],
            "gripper_width_m": telemetry["gripper_width_m"],
            "gripper_force_n": telemetry["gripper_force_n"],
            "telemetry_contract": telemetry,
        }
    except Exception as exc:  # noqa: BLE001 - preserve cleanup evidence
        state_error = f"{type(exc).__name__}: {exc}"
        logger.error(
            "[manual-restore] final posture telemetry unavailable: %s",
            state_error,
        )

    write_json_atomic(out_dir / "final_posture.json", {
        "schema": "oap_final_posture_v1",
        "restore_mode": "manual_post_experiment",
        "automatic_lift_open_home_performed": False,
        "stop_rpc_acknowledged": stop_acknowledged,
        "stop_error": stop_error,
        "state_observed": state_payload is not None,
        "state_error": state_error,
        "state": state_payload,
        "held_close_width_m": held_close_width_m,
        "held_status": held_status,
        "close_was_commanded": close_was_commanded,
        "restore_target_artifact": "initial_posture.json",
        "pre_open_state_artifact": "pre_open_posture.json",
        "operator_action_required": (
            "manually restore q_rad and the verified open gripper width from "
            "initial_posture.json before the next experiment"
        ),
        "unix_s": time.time(),
    })
    logger.warning(
        "[manual-restore] automatic lift/open/home disabled; final posture "
        "recorded in %s and controller stop requested. Restore manually from %s",
        out_dir / "final_posture.json",
        out_dir / "initial_posture.json",
    )


def go_home_safe_or_hold(session: RobotSession, *,
                         held_close_width_m: float | None = None,
                         held_status: bool | None = None,
                         close_was_commanded: bool = False) -> None:
    """Restore canonical home in the finally block -- UNLESS an object is held.

    The whole exit ritual (lift -> open gripper -> joint-home, speed-capped,
    each stage tolerating the previous one failing) is one library call. But
    homing OPENS THE GRIPPER: while a held object is unresolved that DROPS it
    (it has happened). So with ``held_close_width_m`` set this REFUSES to home,
    keeps the grip closed, and tells the operator to recover manually. Never
    raises -- it runs on every exit path.
    """
    if not safety.safe_to_home(
            held_close_width_m, held_status=held_status,
            close_was_commanded=close_was_commanded):
        held_width_m = float(held_close_width_m) if held_close_width_m is not None else float("nan")
        logger.error(
            "[home] REFUSING to home: held state is %s after_close=%s "
            "(close width %.4f m). Homing would open the gripper and may DROP it. "
            "Recover manually: take the object / set it down, then home via "
            "flexiv-control serve.", held_status, close_was_commanded,
            held_width_m)
        return
    try:
        logger.info("[home] restoring canonical posture (go_home_safe)")
        r = session.robot.go_home_safe(
            q_home=session.home_q, lift_m=0.10,
            open_gripper_width=safety.default_open_width(session.home),
            max_tcp_speed=0.10, max_joint_speed=0.3)
        logger.info("[home] %s stages=%s", r.summary(), r.log)
        if not r.success:
            logger.error("[home] RESTORE FAILED -- manual recovery needed")
    except Exception as exc:  # noqa: BLE001 - the exit ritual must never mask the run
        logger.error("[home] RESTORE FAILED -- manual recovery needed: %r", exc)
