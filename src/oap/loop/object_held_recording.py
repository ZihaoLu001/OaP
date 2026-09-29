"""Attended, object-independent ``ObjectHeld`` hardware calibration.

The backend-neutral recorder imports neither a robot driver nor a perception
model. Production calibration records only empty GN01 closes and width-sensor
readback. It has no held-object/tracker phase and cannot command arm motion.
Every gripper command, telemetry sample, and file digest is retained.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from oap.loop.evidence_calibration import (
    canonical_runtime_hardware_identity_sha256,
)
from oap.loop.executor_readiness import validate_robot_state_telemetry
from oap.twin.assets import GN01_OPEN_WIDTH_M

SAMPLE_SCHEMA = "oap_object_held_evidence_samples_v4"
PHASE_SCHEMA = "oap_object_held_evidence_phase_v4"
MANIFEST_SCHEMA = "oap_object_held_evidence_sha256_manifest_v1"
SERVER_INFO_SCHEMA = "flexiv-control.server-info.v5"
PROTOCOL_ID = "flexiv-control.trajectory-rpc.v5"

EMPTY_FIT_TRIALS = 5
DEFAULT_HOLDOUT_TRIALS = 1
PRODUCTION_CLOSE_WIDTH_M = 0.0
PRODUCTION_CLOSE_FORCE_N = 40.0
PRODUCTION_GRIPPER_VELOCITY_M_S = 0.1
_FRESH_STATE_TIMEOUT_TICKS = 10.0

__all__ = [
    "DEFAULT_HOLDOUT_TRIALS",
    "EMPTY_FIT_TRIALS",
    "ObjectHeldEvidenceRecorder",
    "RuntimeContract",
    "SAMPLE_SCHEMA",
    "hash_evidence_tree",
    "validate_runtime_contract",
]


class RobotIO(Protocol):
    """Narrow, gripper-only seam used by the hardware recorder."""

    def get_server_info(self) -> Mapping[str, Any]: ...

    def get_safety_profile(self) -> Any: ...

    def get_state(self) -> Any: ...

    def command_gripper(
        self,
        *,
        width_m: float,
        force_n: float,
        velocity_m_s: float,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class RuntimeContract:
    """Exact controller facts that authorize the calibration commands."""

    control_hz: float
    active_safety_profile: str
    max_joint_speed_scale: float
    hardware_id: str
    runtime_hardware_identity: dict[str, Any]
    gripper_limits: dict[str, Any]
    effective_joint_limits: dict[str, Any]
    protocol_fingerprint_sha256: str
    source_fingerprint_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SERVER_INFO_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "control_hz": self.control_hz,
            "active_safety_profile": self.active_safety_profile,
            "max_joint_speed_scale": self.max_joint_speed_scale,
            "hardware_id": self.hardware_id,
            "runtime_hardware_identity": self.runtime_hardware_identity,
            "gripper_limits": self.gripper_limits,
            "effective_joint_limits": self.effective_joint_limits,
            "protocol_fingerprint_sha256": self.protocol_fingerprint_sha256,
            "source_fingerprint_sha256": self.source_fingerprint_sha256,
        }


def _finite(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"{label} is not finite")
    return number


def _sha256_text(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise RuntimeError(f"{label} is not a lowercase SHA-256")
    return digest


def validate_runtime_contract(server_info: Mapping[str, Any], safety_profile: Any) -> RuntimeContract:
    """Reject old, incomplete, or internally inconsistent controllers."""
    if server_info.get("schema") != SERVER_INFO_SCHEMA:
        raise RuntimeError(
            "ObjectHeld calibration requires flexiv-control server-info.v5; "
            "legacy controller identities are rejected before motion"
        )
    if server_info.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("unexpected flexiv-control protocol_id")
    control_hz = _finite(server_info.get("control_hz"), label="control_hz")
    if control_hz <= 0.0:
        raise RuntimeError("control_hz must be > 0")
    profile_name = str(server_info.get("active_safety_profile", "")).strip()
    if not profile_name or str(getattr(safety_profile, "name", "")) != profile_name:
        raise RuntimeError("leased safety profile does not match server_info active profile")
    speed_scale = _finite(
        getattr(safety_profile, "max_joint_speed_scale", None),
        label="active profile max_joint_speed_scale",
    )
    if not 0.0 < speed_scale <= 1.0:
        raise RuntimeError("active profile max_joint_speed_scale must be in (0, 1]")

    identity = server_info.get("runtime_hardware_identity")
    if not isinstance(identity, Mapping) or not identity:
        raise RuntimeError("server_info lacks runtime_hardware_identity")
    try:
        hardware_id = canonical_runtime_hardware_identity_sha256(
            dict(identity)
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    raw_gripper = server_info.get("gripper_limits")
    if not isinstance(raw_gripper, Mapping):
        raise RuntimeError("server_info lacks runtime gripper_limits")
    gripper = dict(raw_gripper)
    required_gripper = (
        "min_width_m",
        "max_width_m",
        "min_velocity_m_s",
        "max_velocity_m_s",
        "min_force_n",
        "max_force_n",
    )
    for key in required_gripper:
        gripper[key] = _finite(gripper.get(key), label=f"gripper_limits.{key}")
    if gripper["min_width_m"] > gripper["max_width_m"]:
        raise RuntimeError("gripper width limits are inverted")
    if gripper["min_velocity_m_s"] > gripper["max_velocity_m_s"]:
        raise RuntimeError("gripper velocity limits are inverted")
    if gripper["min_force_n"] > gripper["max_force_n"]:
        raise RuntimeError("gripper force limits are inverted")
    for value, low_key, high_key, label in (
        (PRODUCTION_CLOSE_WIDTH_M, "min_width_m", "max_width_m", "close width"),
        (GN01_OPEN_WIDTH_M, "min_width_m", "max_width_m", "canonical open width"),
        (PRODUCTION_CLOSE_FORCE_N, "min_force_n", "max_force_n", "close force"),
        (PRODUCTION_GRIPPER_VELOCITY_M_S, "min_velocity_m_s", "max_velocity_m_s", "gripper velocity"),
    ):
        if not gripper[low_key] <= value <= gripper[high_key]:
            raise RuntimeError(
                f"production {label} {value} is outside runtime limits [{gripper[low_key]}, {gripper[high_key]}]"
            )
    effective = server_info.get("effective_joint_limits")
    if not isinstance(effective, Mapping):
        raise RuntimeError("server_info lacks effective_joint_limits")
    effective_dict = dict(effective)
    recorded_effective_sha256 = _require_sha256(effective_dict.get("sha256"), label="effective_joint_limits.sha256")
    unsigned_effective = {key: value for key, value in effective_dict.items() if key != "sha256"}
    if _sha256_text(unsigned_effective) != recorded_effective_sha256:
        raise RuntimeError("effective_joint_limits SHA-256 does not match its payload")
    for key in (
        "enforced_position_min_rad",
        "enforced_position_max_rad",
        "base_velocity_max_rad_s",
    ):
        values = np.asarray(effective_dict.get(key), dtype=float)
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise RuntimeError(f"effective_joint_limits.{key} must be finite length 7")
    lower = np.asarray(effective_dict["enforced_position_min_rad"], dtype=float)
    upper = np.asarray(effective_dict["enforced_position_max_rad"], dtype=float)
    velocity = np.asarray(effective_dict["base_velocity_max_rad_s"], dtype=float)
    if np.any(lower >= upper):
        raise RuntimeError("effective joint position limits are not ordered")
    if np.any(velocity <= 0.0):
        raise RuntimeError("effective joint velocity limits must be positive")
    recorded_scale = _finite(
        effective_dict.get("max_joint_speed_scale"),
        label="effective_joint_limits.max_joint_speed_scale",
    )
    if not math.isclose(recorded_scale, speed_scale, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("effective joint contract and active profile speed scales differ")
    return RuntimeContract(
        control_hz=control_hz,
        active_safety_profile=profile_name,
        max_joint_speed_scale=speed_scale,
        hardware_id=hardware_id,
        runtime_hardware_identity=dict(identity),
        gripper_limits=gripper,
        effective_joint_limits=effective_dict,
        protocol_fingerprint_sha256=_require_sha256(
            server_info.get("protocol_fingerprint_sha256"), label="protocol_fingerprint_sha256"
        ),
        source_fingerprint_sha256=_require_sha256(
            server_info.get("source_fingerprint_sha256"), label="source_fingerprint_sha256"
        ),
    )


def _state_row(state: Any) -> dict[str, Any]:
    contract = validate_robot_state_telemetry(state, expected_joint_count=7)
    q = np.asarray(getattr(state, "q"), dtype=float).reshape(7)
    tcp = np.asarray(getattr(state, "tcp_pose"), dtype=float).reshape(7)
    if contract["gripper_force_n"] is None:
        raise RuntimeError("RobotState gripper force telemetry is missing")
    return {
        "stamp_s": float(contract["stamp_s"]),
        "q_rad": q.tolist(),
        "tcp_pose_base": tcp.tolist(),
        "gripper_width_m": float(contract["gripper_width_m"]),
        "gripper_force_n": float(contract["gripper_force_n"]),
        "gripper_is_moving": bool(
            getattr(state, "gripper_is_moving", False)
        ),
        "telemetry_contract": contract,
    }


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def hash_evidence_tree(root: Path) -> dict[str, Any]:
    """Hash every materialized evidence file, excluding the manifest itself."""
    root = Path(root).resolve()
    rows: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "sha256_manifest.json":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        rows[path.relative_to(root).as_posix()] = digest.hexdigest()
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generated_at_unix_s": time.time(),
        "root": str(root),
        "files": rows,
    }
    _write_json(root / "sha256_manifest.json", manifest)
    return manifest


def _readback_resolution_upper_bound(values: Sequence[float]) -> float:
    """Return the smallest width change actually distinguished by telemetry.

    This is an empirical upper bound on sensor resolution, not a command-grid
    spacing.  It deliberately consumes raw readbacks from the ordinary
    empty-close/open trials; no actuator target is treated as a measurement.
    A degenerate trace fails closed.
    """
    unique = np.unique(np.asarray(values, dtype=float))
    increments = np.diff(unique)
    increments = increments[increments > 0.0]
    if len(unique) < 5 or not len(increments):
        raise RuntimeError(
            "raw gripper telemetry trace did not produce five distinct "
            "measurements; sensor resolution cannot be bounded"
        )
    return float(np.min(increments))


def _require_directional_response(
    row: Mapping[str, Any],
    *,
    opening: bool,
) -> None:
    """Reject a stale command return without inventing a width tolerance."""
    before = float(row["telemetry_before"]["gripper_width_m"])
    after = float(row["telemetry_after"]["gripper_width_m"])
    changed_in_commanded_direction = after > before if opening else after < before
    if not changed_in_commanded_direction:
        direction = "increase" if opening else "decrease"
        raise RuntimeError(
            f"{row['kind']} readback did not {direction} after the command; "
            "refusing stale gripper telemetry"
        )


def _telemetry_trace(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flatten strict transition telemetry while retaining event provenance."""
    trace: list[dict[str, Any]] = []
    for event_sequence, event in enumerate(events):
        raw_transition = event["result"].get("transition_telemetry")
        if not isinstance(raw_transition, list) or not raw_transition:
            raise RuntimeError(
                f"{event['kind']} has no raw transition telemetry"
            )
        candidates: list[tuple[str, Mapping[str, Any]]] = [
            ("before", event["telemetry_before"]),
            *[("transition", sample) for sample in raw_transition],
            ("after", event["telemetry_after"]),
        ]
        event_rows: list[dict[str, Any]] = []
        for sample_position, sample in candidates:
            if not isinstance(sample, Mapping):
                raise RuntimeError(
                    f"{event['kind']} transition sample is not a mapping"
                )
            try:
                stamp = float(sample["stamp_s"])
                width = float(sample["gripper_width_m"])
                moving = bool(sample["gripper_is_moving"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{event['kind']} transition sample is incomplete"
                ) from exc
            if not math.isfinite(stamp) or not math.isfinite(width):
                raise RuntimeError(
                    f"{event['kind']} transition sample is not finite"
                )
            row = {
                "event_sequence": int(event_sequence),
                "event_kind": str(event["kind"]),
                "sample_index": len(event_rows),
                "sample_position": sample_position,
                "stamp_s": stamp,
                "gripper_width_m": width,
                "gripper_is_moving": moving,
            }
            if event_rows and stamp <= event_rows[-1]["stamp_s"]:
                previous = event_rows[-1]
                duplicate = (
                    stamp == previous["stamp_s"]
                    and width == previous["gripper_width_m"]
                    and moving == previous["gripper_is_moving"]
                )
                if duplicate and sample_position == "transition":
                    continue
                if (
                    duplicate
                    and sample_position == "after"
                    and previous["sample_position"] == "transition"
                ):
                    event_rows.pop()
                    row["sample_index"] = len(event_rows)
                else:
                    raise RuntimeError(
                        f"{event['kind']} transition timestamps are not "
                        "strictly increasing"
                    )
            event_rows.append(row)
        positions = [row["sample_position"] for row in event_rows]
        if (
            positions[0] != "before"
            or positions[-1] != "after"
            or "transition" not in positions[1:-1]
        ):
            raise RuntimeError(
                f"{event['kind']} lacks a complete transition trace"
            )
        if trace and event_rows[0]["stamp_s"] <= trace[-1]["stamp_s"]:
            raise RuntimeError(
                "gripper event timestamps are not strictly increasing"
            )
        for sample_index, row in enumerate(event_rows):
            row["sample_index"] = sample_index
        trace.extend(event_rows)
    return trace


class ObjectHeldEvidenceRecorder:
    """Record strict, object-independent empty-gripper calibration."""

    def __init__(
        self,
        robot: RobotIO,
        *,
        out_dir: Path,
    ) -> None:
        self.robot = robot
        self.out_dir = Path(out_dir).resolve()
        self._control_hz: float | None = None
        self._last_after_stamp_s: float | None = None

    def _contract(self) -> RuntimeContract:
        return validate_runtime_contract(self.robot.get_server_info(), self.robot.get_safety_profile())

    def _fresh_state_before_command(self) -> dict[str, Any]:
        """Wait briefly for a telemetry tick after the preceding event."""
        row = _state_row(self.robot.get_state())
        previous_stamp = self._last_after_stamp_s
        if previous_stamp is None or row["stamp_s"] > previous_stamp:
            return row
        if self._control_hz is None or self._control_hz <= 0.0:
            raise RuntimeError("runtime control_hz is unavailable")
        poll_period_s = 1.0 / self._control_hz
        deadline = time.monotonic() + _FRESH_STATE_TIMEOUT_TICKS * poll_period_s
        while row["stamp_s"] <= previous_stamp:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                raise RuntimeError(
                    "gripper telemetry did not advance before the next command"
                )
            time.sleep(min(poll_period_s, remaining_s))
            row = _state_row(self.robot.get_state())
        return row

    def _gripper_event(self, *, kind: str, width_m: float, force_n: float) -> dict[str, Any]:
        before = self._fresh_state_before_command()
        command_stamp = time.time()
        result = dict(
            self.robot.command_gripper(
                width_m=float(width_m),
                force_n=float(force_n),
                velocity_m_s=PRODUCTION_GRIPPER_VELOCITY_M_S,
            )
        )
        after = _state_row(self.robot.get_state())
        self._last_after_stamp_s = float(after["stamp_s"])
        return {
            "kind": kind,
            "command_stamp_s": command_stamp,
            "command": {
                "width_m": float(width_m),
                "force_n": float(force_n),
                "velocity_m_s": PRODUCTION_GRIPPER_VELOCITY_M_S,
                "grasp": False,
            },
            "result": result,
            "telemetry_before": before,
            "telemetry_after": after,
        }

    def record_empty(
        self,
        *,
        confirmed_empty_and_clear: bool,
        fit_trials: int = EMPTY_FIT_TRIALS,
        holdout_trials: int = DEFAULT_HOLDOUT_TRIALS,
    ) -> dict[str, Any]:
        """Record empty closes and sensor resolution, then restore fully open."""
        if not confirmed_empty_and_clear:
            raise RuntimeError("empty phase requires explicit empty-gripper/workspace-clear confirmation")
        if int(fit_trials) < EMPTY_FIT_TRIALS or int(holdout_trials) < 1:
            raise ValueError("empty phase needs >=5 fit and >=1 holdout trials")
        phase_dir = self.out_dir / "empty"
        if self.out_dir.exists():
            raise FileExistsError(f"calibration output already exists: {self.out_dir}")
        contract = self._contract()
        self._control_hz = contract.control_hz
        self._last_after_stamp_s = None
        phase_dir.mkdir(parents=True)
        open_width = float(GN01_OPEN_WIDTH_M)
        fit_rows: list[dict[str, Any]] = []
        holdout_rows: list[dict[str, Any]] = []
        reset_rows: list[dict[str, Any]] = []
        try:
            for index in range(int(fit_trials) + int(holdout_trials)):
                row = self._gripper_event(
                    kind="air_close",
                    width_m=PRODUCTION_CLOSE_WIDTH_M,
                    force_n=PRODUCTION_CLOSE_FORCE_N,
                )
                _require_directional_response(row, opening=False)
                row["index"] = index
                row["fit"] = index < int(fit_trials)
                (fit_rows if row["fit"] else holdout_rows).append(row)
                _write_json(phase_dir / f"air_close_{index:02d}.json", row)
                reset = self._gripper_event(
                    kind="reset_open",
                    width_m=open_width,
                    force_n=PRODUCTION_CLOSE_FORCE_N,
                )
                _require_directional_response(reset, opening=True)
                reset["index"] = index
                reset_rows.append(reset)
                _write_json(phase_dir / f"reset_open_{index:02d}.json", reset)
        finally:
            open_event = self._gripper_event(
                kind="final_restore_open",
                width_m=open_width,
                force_n=PRODUCTION_CLOSE_FORCE_N,
            )
            _write_json(phase_dir / "final_restore_open.json", open_event)

        fit_air = [float(row["telemetry_after"]["gripper_width_m"]) for row in fit_rows]
        holdout_air = [
            float(row["telemetry_after"]["gripper_width_m"])
            for row in holdout_rows
        ]
        raw_trace = _telemetry_trace(
            [
                event
                for close, reset in zip(
                    [*fit_rows, *holdout_rows],
                    reset_rows,
                    strict=True,
                )
                for event in (close, reset)
            ]
        )
        readback = [
            float(sample["gripper_width_m"])
            for sample in raw_trace
        ]
        resolution = _readback_resolution_upper_bound(readback)
        final_open_width = float(open_event["telemetry_after"]["gripper_width_m"])
        final_open_error = abs(final_open_width - open_width)
        if final_open_width <= max([*fit_air, *holdout_air]):
            raise RuntimeError(
                "final gripper-open readback is not above every measured "
                "empty-close width"
            )
        phase = {
            "schema": PHASE_SCHEMA,
            "phase": "empty",
            "complete": True,
            "captured_at_unix_s": time.time(),
            "runtime_contract": contract.to_dict(),
            "air_close_fit": fit_rows,
            "air_close_holdout": holdout_rows,
            "reset_open": reset_rows,
            "air_close_width_m": fit_air,
            "gripper_width_readback_trace": raw_trace,
            "measured_readback_resolution_m": resolution,
            "final_open": open_event,
            "final_open_error_m": final_open_error,
        }
        samples = {
            "schema": SAMPLE_SCHEMA,
            "runtime_contract": contract.to_dict(),
            "air_close_fit_width_m": fit_air,
            "air_close_holdout_width_m": holdout_air,
            "gripper_width_readback_trace": raw_trace,
        }
        _write_json(self.out_dir / "samples.fit.json", samples)
        _write_json(phase_dir / "phase.json", phase)
        hash_evidence_tree(self.out_dir)
        return phase
