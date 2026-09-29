"""Fail-closed software and human-verification gate for the real executor.

This module is deliberately connection-free.  It may import ``flexiv_control``
to prove that the installed package exposes the API OaP calls, but it
never constructs ``RemoteRobot`` and never opens a robot, gripper, or camera.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np

from oap.loop.safety import SafetyError

__all__ = [
    "EXECUTOR_ATTESTATIONS",
    "evaluate_executor_readiness",
    "inspect_flexiv_control",
    "require_executor_readiness",
    "validate_robot_state_telemetry",
]

_DIST_NAME = "flexiv-control"
_MODULE_NAME = "flexiv_control"
_LATCH_SCHEMA = "oap_joint_executor_verification_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# These are not optional policy preferences.  They are exactly the unresolved
# hardware facts that must be demonstrated before this executor can be enabled.
EXECUTOR_ATTESTATIONS = (
    "attended_estop_bringup",
    "joint_order_sign_units_limits",
    "trajectory_prefix_semantics",
    "numeric_gripper_width_and_force_limit",
    "timestamped_joint_gripper_telemetry",
    "max_joint_speed_scale_enforced",
    "rt_joint_torque_stream_verified",
    "max_joint_torque_scale_enforced",
)

_TOP_LEVEL_API = (
    "RemoteRobot",
    "GripperCommand",
    "JointTrajectory",
    "JointWaypoint",
    "JointTorqueTrajectory",
    "JointTorqueWaypoint",
)
_REMOTE_ROBOT_API = (
    "__enter__",
    "__exit__",
    "command_gripper",
    "execute_joint_trajectory",
    "execute_joint_torque_trajectory",
    "get_server_info",
    "get_safety_profile",
    "get_state",
    "go_home_safe",
    "start_cartesian_impedance",
    "stop",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _editable_project_path(distribution: Any) -> Path | None:
    try:
        raw = distribution.read_text("direct_url.json")
        if not raw:
            return None
        direct = json.loads(raw)
        if not bool((direct.get("dir_info") or {}).get("editable")):
            return None
        parsed = urlparse(str(direct.get("url", "")))
        if parsed.scheme != "file":
            return None
        decoded = unquote(parsed.path)
        if (
            len(decoded) >= 3
            and decoded[0] == "/"
            and decoded[1].isalpha()
            and decoded[2] == ":"
        ):
            decoded = decoded[1:]
        return Path(decoded)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _package_tree_sha256(roots: list[Path]) -> str | None:
    """Hash importable package bytes, excluding interpreter caches."""
    files: list[tuple[str, Path]] = []
    for root_index, root in enumerate(roots):
        if root.is_file():
            files.append((f"{root_index}/{root.name}", root))
            continue
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or path.suffix in {".pyc", ".pyo"}
            ):
                continue
            files.append((f"{root_index}/{path.relative_to(root).as_posix()}", path))
    if not files:
        return None
    digest = hashlib.sha256()
    for relative, path in files:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def inspect_flexiv_control() -> dict[str, Any]:
    """Inspect the installed dependency and required API without connecting.

    The report intentionally distinguishes an installed distribution record
    from an importable package.  A stale editable ``.pth`` can satisfy
    ``pip show`` while exposing no Python module; that condition is a hard
    software failure with the stale source path included in the diagnostic.
    """
    errors: list[str] = []
    version: str | None = None
    distribution_root: str | None = None
    editable_path: Path | None = None
    try:
        distribution = importlib.metadata.distribution(_DIST_NAME)
    except importlib.metadata.PackageNotFoundError:
        distribution = None
        errors.append(f"Python distribution {_DIST_NAME!r} is not installed")
    except Exception as exc:  # noqa: BLE001 - diagnostic boundary
        distribution = None
        errors.append(
            f"distribution metadata failed: {type(exc).__name__}: {exc}"
        )
    if distribution is not None:
        version = str(distribution.version)
        distribution_root = str(Path(distribution.locate_file("")).resolve())
        editable_path = _editable_project_path(distribution)
        if editable_path is not None and not editable_path.is_dir():
            errors.append(
                "editable project directory does not exist: "
                f"{editable_path}"
            )

    try:
        spec = importlib.util.find_spec(_MODULE_NAME)
    except Exception as exc:  # noqa: BLE001 - diagnostic boundary
        spec = None
        errors.append(f"module discovery failed: {type(exc).__name__}: {exc}")
    if spec is None:
        errors.append(
            f"distribution metadata exists but import spec {_MODULE_NAME!r} "
            "is missing"
        )

    roots: list[Path] = []
    origin: str | None = None
    if spec is not None:
        if spec.origin and spec.origin not in {"built-in", "frozen"}:
            origin_path = Path(spec.origin).resolve()
            origin = str(origin_path)
            if origin_path.is_file():
                roots.append(origin_path)
        if spec.submodule_search_locations:
            roots.extend(
                Path(location).resolve()
                for location in spec.submodule_search_locations
            )
    roots = list(dict.fromkeys(roots))
    missing_roots = [str(root) for root in roots if not root.exists()]
    if missing_roots:
        errors.append(f"import roots do not exist: {missing_roots}")
    package_tree_sha256 = _package_tree_sha256(roots)
    if spec is not None and package_tree_sha256 is None:
        errors.append("import spec resolves, but no package bytes can be hashed")

    missing_api: list[str] = []
    import_error: str | None = None
    module_version: str | None = None
    if spec is not None:
        try:
            module = importlib.import_module(_MODULE_NAME)
            types_module = importlib.import_module(f"{_MODULE_NAME}.types")
            raw_module_version = getattr(module, "__version__", None)
            module_version = (
                None
                if raw_module_version is None
                else str(raw_module_version)
            )
            if module_version is None:
                errors.append(
                    f"module {_MODULE_NAME!r} has no __version__ identity"
                )
            elif version is not None and module_version != version:
                errors.append(
                    "distribution/module version mismatch "
                    f"(distribution={version}, module={module_version})"
                )
            if editable_path is not None and origin is not None:
                try:
                    Path(origin).relative_to(editable_path.resolve())
                except ValueError:
                    errors.append(
                        "editable distribution resolves to a module outside "
                        f"its project tree (project={editable_path}, "
                        f"module={origin})"
                    )
            for symbol in _TOP_LEVEL_API:
                if not hasattr(module, symbol):
                    missing_api.append(f"{_MODULE_NAME}.{symbol}")
            if not hasattr(types_module, "GripperCommand"):
                missing_api.append(f"{_MODULE_NAME}.types.GripperCommand")
            remote_robot = getattr(module, "RemoteRobot", None)
            if remote_robot is not None:
                for method in _REMOTE_ROBOT_API:
                    if not hasattr(remote_robot, method):
                        missing_api.append(f"RemoteRobot.{method}")
        except Exception as exc:  # noqa: BLE001 - diagnostic boundary
            import_error = f"{type(exc).__name__}: {exc}"
            errors.append(f"import {_MODULE_NAME!r} failed: {import_error}")
    if missing_api:
        errors.append(f"required executor API is missing: {missing_api}")

    fingerprint: str | None = None
    if version is not None and package_tree_sha256 is not None:
        material = {
            "distribution": _DIST_NAME,
            "version": version,
            "package_tree_sha256": package_tree_sha256,
            "required_top_level_api": list(_TOP_LEVEL_API),
            "required_remote_robot_api": list(_REMOTE_ROBOT_API),
        }
        fingerprint = hashlib.sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    return {
        "schema": "oap_flexiv_control_inspection_v1",
        "software_ready": not errors,
        "distribution": _DIST_NAME,
        "version": version,
        "module_version": module_version,
        "distribution_root": distribution_root,
        "editable_project_path": (
            None if editable_path is None else str(editable_path)
        ),
        "editable_project_exists": (
            None if editable_path is None else editable_path.is_dir()
        ),
        "module": _MODULE_NAME,
        "module_origin": origin,
        "module_search_roots": [str(root) for root in roots],
        "package_tree_sha256": package_tree_sha256,
        "software_fingerprint_sha256": fingerprint,
        "missing_api": missing_api,
        "import_error": import_error,
        "errors": errors,
        "connection_attempted": False,
        "motion_attempted": False,
    }


def _parse_utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _inspect_latch(
    path: Path | None,
    approved_sha256: str | None,
    *,
    software_fingerprint_sha256: str | None,
) -> dict[str, Any]:
    errors: list[str] = []
    actual_sha256: str | None = None
    document: dict[str, Any] | None = None
    if path is None:
        return {
            "provided": False,
            "valid": False,
            "path": None,
            "sha256": None,
            "approved_sha256": approved_sha256,
            "hardware_id": None,
            "verified_by": None,
            "verified_at": None,
            "attestations": {},
            "errors": [
                "no joint-executor verification JSON was supplied; attended "
                "hardware verification remains pending"
            ],
        }
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        errors.append(f"verification JSON does not exist: {resolved}")
    else:
        actual_sha256 = _sha256_file(resolved)
        try:
            loaded = json.loads(resolved.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("top level must be an object")
            document = loaded
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"cannot parse verification JSON: {exc}")

    if approved_sha256 is None:
        errors.append(
            "no approved verification SHA-256 was supplied; the operator must "
            "approve the exact latch file"
        )
    elif not _SHA256_RE.fullmatch(approved_sha256):
        errors.append("approved verification SHA-256 must be 64 lowercase hex")
    elif actual_sha256 is not None and approved_sha256 != actual_sha256:
        errors.append(
            "approved verification SHA-256 does not match the latch "
            f"(approved={approved_sha256}, actual={actual_sha256})"
        )

    if document is not None:
        if document.get("schema") != _LATCH_SCHEMA:
            errors.append(
                f"schema must be {_LATCH_SCHEMA!r}, got {document.get('schema')!r}"
            )
        for key in ("hardware_id", "verified_by"):
            if not isinstance(document.get(key), str) or not document[key].strip():
                errors.append(f"{key} must be a non-empty string")
        if not _parse_utc_timestamp(document.get("verified_at")):
            errors.append("verified_at must be an ISO-8601 timestamp with timezone")
        if software_fingerprint_sha256 is None:
            errors.append(
                "installed flexiv-control has no software fingerprint to bind"
            )
        elif (
            document.get("software_fingerprint_sha256")
            != software_fingerprint_sha256
        ):
            errors.append(
                "verification latch is for a different flexiv-control build "
                f"(latch={document.get('software_fingerprint_sha256')!r}, "
                f"installed={software_fingerprint_sha256!r})"
            )
        attestations = document.get("attestations")
        if not isinstance(attestations, dict):
            errors.append("attestations must be an object")
            attestations = {}
        for name in EXECUTOR_ATTESTATIONS:
            if attestations.get(name) is not True:
                errors.append(f"attestation {name!r} is not explicitly true")
    else:
        attestations = {}

    return {
        "provided": True,
        "valid": not errors,
        "path": str(resolved),
        "sha256": actual_sha256,
        "approved_sha256": approved_sha256,
        "hardware_id": None if document is None else document.get("hardware_id"),
        "verified_by": None if document is None else document.get("verified_by"),
        "verified_at": None if document is None else document.get("verified_at"),
        "attestations": dict(attestations),
        "errors": errors,
    }


def evaluate_executor_readiness(
    verification_json: Path | None,
    approved_verification_sha256: str | None,
) -> dict[str, Any]:
    """Return one connection-free report; never infer hardware verification."""
    software = inspect_flexiv_control()
    latch = _inspect_latch(
        verification_json,
        approved_verification_sha256,
        software_fingerprint_sha256=software["software_fingerprint_sha256"],
    )
    hardware_verified = bool(software["software_ready"] and latch["valid"])
    return {
        "schema": "oap_joint_executor_readiness_v1",
        "software_ready": bool(software["software_ready"]),
        "verification_latch_valid": bool(latch["valid"]),
        "hardware_verified": hardware_verified,
        "flexiv_control": software,
        "verification_latch": latch,
        "connection_attempted": False,
        "motion_attempted": False,
    }


def require_executor_readiness(
    verification_json: Path | None,
    approved_verification_sha256: str | None,
) -> dict[str, Any]:
    """Evaluate and raise a diagnostic refusal unless every gate passes."""
    report = evaluate_executor_readiness(
        verification_json,
        approved_verification_sha256,
    )
    if report["hardware_verified"]:
        return report
    software_errors = report["flexiv_control"]["errors"]
    latch_errors = report["verification_latch"]["errors"]
    reasons = [*software_errors, *latch_errors]
    raise SafetyError(
        "joint executor readiness refused before robot connection: "
        + "; ".join(str(reason) for reason in reasons)
    )


def validate_robot_state_telemetry(
    state: Any,
    *,
    expected_joint_count: int,
) -> dict[str, Any]:
    """Validate the exact state keys consumed by planning and evidence.

    This validates observations only.  It never fills missing telemetry with
    zeros, guesses a timestamp, or treats a command acknowledgement as state.
    """
    missing = [
        name
        for name in ("stamp", "q", "tcp_pose", "gripper_width")
        if getattr(state, name, None) is None
    ]
    if missing:
        raise SafetyError(
            f"RobotState telemetry is missing required keys: {missing}"
        )
    try:
        stamp = float(state.stamp)
        q = np.asarray(state.q, dtype=float).reshape(-1)
        tcp_pose = np.asarray(state.tcp_pose, dtype=float).reshape(-1)
        gripper_width = float(state.gripper_width)
    except (TypeError, ValueError) as exc:
        raise SafetyError(f"RobotState telemetry has invalid types: {exc}") from exc
    if not math.isfinite(stamp) or stamp <= 0.0:
        raise SafetyError("RobotState stamp must be finite and positive")
    if q.shape != (int(expected_joint_count),) or not np.all(np.isfinite(q)):
        raise SafetyError(
            "RobotState q must contain exactly "
            f"{expected_joint_count} finite joint positions, got {q.shape}"
        )
    if tcp_pose.shape != (7,) or not np.all(np.isfinite(tcp_pose)):
        raise SafetyError(
            "RobotState tcp_pose must contain seven finite values (xyz+wxyz)"
        )
    if float(np.linalg.norm(tcp_pose[3:7])) <= 1e-9:
        raise SafetyError("RobotState tcp_pose quaternion is degenerate")
    if not math.isfinite(gripper_width) or gripper_width < 0.0:
        raise SafetyError(
            "RobotState gripper_width must be finite and non-negative"
        )
    force_raw = getattr(state, "gripper_force", None)
    gripper_force = None
    if force_raw is not None:
        gripper_force = float(force_raw)
        if not math.isfinite(gripper_force):
            raise SafetyError(
                "RobotState gripper_force must be finite when supplied"
            )
    return {
        "schema": "oap_robot_state_telemetry_contract_v1",
        "source": "flexiv_control.RemoteRobot.get_state",
        "stamp_s": stamp,
        "joint_count": int(len(q)),
        "joint_positions_finite": True,
        "tcp_pose_finite": True,
        "gripper_width_m": gripper_width,
        "gripper_force_n": gripper_force,
        "required_keys": ["stamp", "q", "tcp_pose", "gripper_width"],
    }
