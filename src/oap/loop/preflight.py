"""Read-only readiness checks before an attended real-robot episode.

The software check imports ``flexiv_control`` to verify its installed API.  The
robot-side probe opens a client connection and sends only ``ping`` plus the
read-only ``get_server_info`` request.  It never enters the client context,
acquires a lease, reads the backend, changes mode, or sends motion.
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import socket
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from oap.loop.safety import SafetyError, require_program_approval
from oap.loop.evidence_calibration import (
    load_object_held_evidence_calibration,
    program_uses_object_held,
)
from oap.loop.executor_readiness import evaluate_executor_readiness
from oap.program import (
    TaskProgram,
    groundability_failures,
    program_sha256,
)
from oap.reconstruct.external import ExternalEnvs
from oap.twin import TABLE_TOP_Z, load_scene_manifest
from oap.utils.io import sha256_file

CheckStatus = Literal["PASS", "FAIL", "CANNOT_CHECK"]
CheckScope = Literal["software", "attended_hardware"]

_SHA256_LINE = re.compile(r"^([0-9a-fA-F]{64})[ \t]+\*?(.+?)\s*$")

__all__ = [
    "PreflightCheck",
    "PreflightConfig",
    "PreflightReport",
    "run_preflight",
]


@dataclass(frozen=True)
class PreflightCheck:
    """One machine-readable readiness check."""

    name: str
    status: CheckStatus
    scope: CheckScope
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "status": self.status,
            "scope": self.scope,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class PreflightConfig:
    """Inputs for a read-only real-episode preflight."""

    bundle_manifest: Path
    program_json: Path
    approved_program_sha256: str | None
    asset_checksums: Path | None
    checksum_root: Path
    hardware_evidence_calibration_json: Path | None = None
    joint_executor_verification_json: Path | None = None
    joint_executor_verification_sha256: str | None = None
    external_envs: str | None = "lab"
    required_external_tools: tuple[str, ...] = ("observer", "foundationpose")
    camera_device: Path = Path("/dev/video0")
    robot_host: str = "ROBOT-HOST-PLACEHOLDER"
    robot_port: int = 8766
    robot_timeout_s: float = 1.0


@dataclass(frozen=True)
class PreflightReport:
    """Complete readiness result with software/hardware kept distinct."""

    checks: tuple[PreflightCheck, ...]
    program_sha256: str | None
    software_ready: bool
    attended_hardware_checks_passed: bool
    readiness: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "oap_preflight_v1",
            "readiness": self.readiness,
            "software_ready": self.software_ready,
            "attended_hardware_checks_passed": self.attended_hardware_checks_passed,
            # These are invariants of this read-only command, not conditions
            # derived from the probes. Camera/TCP availability cannot prove
            # home, calibration, E-stop, active tool, or executor bring-up.
            "motion_authorized": False,
            "sends_motion": False,
            "program_sha256": self.program_sha256,
            "checks": [check.to_dict() for check in self.checks],
        }


def _check(
    name: str,
    status: CheckStatus,
    scope: CheckScope,
    detail: str,
) -> PreflightCheck:
    return PreflightCheck(name=name, status=status, scope=scope, detail=detail)


def _ground_program(
    program: TaskProgram,
    objects: list[Any],
) -> tuple[Any, list[str], str]:
    """Build the task-neutral scene vocabulary, then add structural aliases."""
    from oap.loop import observe as observe_mod
    from oap.program.bindings import bind_program_to_scene
    from oap.program.synthesis import validate_program

    anchors = observe_mod.anchors_from_scene_manifest(
        objects, table_top_z=TABLE_TOP_Z
    )
    # This is the exact task-neutral vocabulary and validator used by runner
    # before it binds roles or authorizes any motion.
    # The manifest vocabulary predates runtime signed material-frame anchors.
    # Defer only that capability check; the grounded pass below remains strict.
    validation_errors = validate_program(
        program,
        anchors,
        require_grounded_frame_contract=False,
    )
    if validation_errors:
        raise ValueError(
            "invalid_program:" + ";".join(validation_errors[:4])
        )
    bindings = bind_program_to_scene(program, objects)
    subject, reference = bindings.subject, bindings.reference

    # Add only aliases produced by the same observation grounder the loop uses.
    # The observation is the manifest's measured initial pose; no camera is
    # touched during preflight.
    observed = {
        "pos_base": [float(v) for v in subject.pose_base],
        "quat_wxyz": [float(v) for v in subject.pose_quat],
        "schema": "preflight_manifest_observation",
    }
    aliases = observe_mod.anchors_from_observation(
        observed,
        subject,
        objects,
        table_top_z=TABLE_TOP_Z,
        reference=reference,
    )
    for name in aliases.names():
        anchors.add(name, aliases[name])
    # The contact structural checks (actor/dynamic-target/shared-target) are
    # decidable only now that role-bound aliases exist. Re-run the SAME
    # validator on the grounded vocabulary so preflight fails closed on
    # exactly the programs the runner's grounded validation refuses.
    grounded_errors = validate_program(program, anchors)
    if grounded_errors:
        raise ValueError(
            "invalid_program_grounded:" + ";".join(grounded_errors[:4])
        )
    return anchors, groundability_failures(program, anchors), (
        "anchors_from_scene_manifest+bind_program_to_scene"
    )


def _check_asset_checksums(path: Path | None, root: Path) -> PreflightCheck:
    if path is None:
        return _check(
            "asset_checksums",
            "CANNOT_CHECK",
            "software",
            "no --asset-checksums file supplied",
        )
    path = path.expanduser()
    if not path.is_file():
        return _check(
            "asset_checksums",
            "FAIL",
            "software",
            f"checksum file does not exist: {path}",
        )
    root = root.expanduser().resolve()
    failures: list[str] = []
    checked = 0
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        match = _SHA256_LINE.match(raw)
        if match is None:
            failures.append(f"line {line_no}: invalid sha256sum syntax")
            continue
        expected, rel = match.groups()
        candidate = (root / rel).resolve() if not Path(rel).is_absolute() else Path(rel).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            failures.append(f"line {line_no}: path escapes checksum root: {rel}")
            continue
        if not candidate.is_file():
            failures.append(f"line {line_no}: missing asset: {rel}")
            continue
        checked += 1
        actual = sha256_file(candidate)
        if actual != expected.lower():
            failures.append(
                f"line {line_no}: digest mismatch for {rel} "
                f"(expected {expected.lower()}, actual {actual})"
            )
    if checked == 0 and not failures:
        failures.append("checksum file contains no asset entries")
    if failures:
        return _check("asset_checksums", "FAIL", "software", "; ".join(failures))
    return _check(
        "asset_checksums",
        "PASS",
        "software",
        f"verified {checked} assets under {root}",
    )


def _check_object_held_evidence_calibration(
    program: TaskProgram | None,
    path: Path | None,
) -> PreflightCheck:
    if program is None:
        return _check(
            "object_held_evidence_calibration",
            "CANNOT_CHECK",
            "software",
            "requires a parsed program",
        )
    if not program_uses_object_held(program):
        return _check(
            "object_held_evidence_calibration",
            "PASS",
            "software",
            "program contains no ObjectHeld term; calibration is not required",
        )
    if path is None:
        return _check(
            "object_held_evidence_calibration",
            "FAIL",
            "software",
            "program contains ObjectHeld but no calibration JSON was supplied",
        )
    try:
        calibration = load_object_held_evidence_calibration(path)
    except ValueError as exc:
        return _check(
            "object_held_evidence_calibration",
            "FAIL",
            "software",
            str(exc),
        )
    return _check(
        "object_held_evidence_calibration",
        "PASS",
        "software",
        (
            f"hardware_id={calibration.hardware_id}; "
            f"source={calibration.source}; sha256={calibration.sha256}"
        ),
    )


def _check_executor_readiness(
    verification_json: Path | None,
    verification_sha256: str | None,
) -> tuple[PreflightCheck, PreflightCheck]:
    """Check import/API identity and the attended human latch separately."""
    report = evaluate_executor_readiness(
        verification_json,
        verification_sha256,
    )
    dependency = report["flexiv_control"]
    if report["software_ready"]:
        software = _check(
            "flexiv_control",
            "PASS",
            "software",
            (
                f"version={dependency['version']}; "
                "required executor API importable; "
                "software_fingerprint_sha256="
                f"{dependency['software_fingerprint_sha256']}"
            ),
        )
    else:
        software = _check(
            "flexiv_control",
            "FAIL",
            "software",
            "; ".join(dependency["errors"]),
        )

    latch = report["verification_latch"]
    if latch["valid"]:
        latch_check = _check(
            "joint_executor_verification_latch",
            "PASS",
            "attended_hardware",
            (
                f"hardware_id={latch['hardware_id']}; "
                f"verified_by={latch['verified_by']}; "
                f"verified_at={latch['verified_at']}; sha256={latch['sha256']}"
            ),
        )
    elif not latch["provided"]:
        latch_check = _check(
            "joint_executor_verification_latch",
            "CANNOT_CHECK",
            "attended_hardware",
            "; ".join(latch["errors"]),
        )
    else:
        latch_check = _check(
            "joint_executor_verification_latch",
            "FAIL",
            "attended_hardware",
            "; ".join(latch["errors"]),
        )
    return software, latch_check


def _check_warp_gpu() -> PreflightCheck:
    try:
        # Warp's first import/init is chatty. Keep stdout as one valid JSON
        # document for automation while still reporting any exception below.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import warp as wp

            wp.init()
            devices = list(wp.get_cuda_devices())
    except Exception as exc:
        return _check("warp_gpu", "FAIL", "software", f"{type(exc).__name__}: {exc}")
    if not devices:
        return _check("warp_gpu", "FAIL", "software", "Warp reports no CUDA device")
    labels = [str(device) for device in devices]
    return _check("warp_gpu", "PASS", "software", f"CUDA devices: {labels}")


def _check_external_envs(
    spec: str | None,
    required_tools: tuple[str, ...],
) -> PreflightCheck:
    try:
        envs = ExternalEnvs.load(spec)
        resolved: list[str] = []
        for name in required_tools:
            python = envs.python(name)
            if not os.access(python, os.X_OK):
                raise RuntimeError(f"{name} interpreter is not executable: {python}")
            tool = envs.tool(name)
            paths = list(tool.pythonpath_extra) + list(tool.ld_library_path_extra)
            paths.extend(
                value
                for key, value in tool.extras.items()
                if key == "root" or key.endswith(("_root", "_path"))
            )
            missing = [
                value
                for value in paths
                if Path(value).is_absolute() and not Path(value).exists()
            ]
            if missing:
                raise RuntimeError(
                    f"{name} has missing configured paths: {', '.join(missing)}"
                )
            resolved.append(f"{name}={python}")
    except Exception as exc:
        return _check(
            "external_envs",
            "FAIL",
            "software",
            f"{type(exc).__name__}: {exc}",
        )
    return _check(
        "external_envs",
        "PASS",
        "software",
        f"profile={envs.profile}; " + ", ".join(resolved),
    )


def _check_camera_device(path: Path) -> PreflightCheck:
    try:
        info = path.stat()
    except FileNotFoundError:
        return _check(
            "camera_device",
            "FAIL",
            "attended_hardware",
            f"device does not exist: {path}",
        )
    except OSError as exc:
        return _check(
            "camera_device",
            "CANNOT_CHECK",
            "attended_hardware",
            f"cannot stat {path}: {exc}",
        )
    if not stat.S_ISCHR(info.st_mode):
        return _check(
            "camera_device",
            "FAIL",
            "attended_hardware",
            f"path is not a character device: {path}",
        )
    if not os.access(path, os.R_OK | os.W_OK):
        return _check(
            "camera_device",
            "FAIL",
            "attended_hardware",
            f"camera device is not readable/writable by this user: {path}",
        )
    return _check(
        "camera_device",
        "PASS",
        "attended_hardware",
        f"device present and accessible: {path}",
    )


def _check_robot_server_identity(
    host: str,
    port: int,
    timeout_s: float,
) -> PreflightCheck:
    """Bind the reachable server to the exact local trajectory protocol."""
    try:
        from flexiv_control import __version__ as flexiv_control_version
        from flexiv_control.client import RemoteRobot, RemoteRobotError
        from flexiv_control.server import protocol
    except Exception as exc:
        return _check(
            "robot_server_identity",
            "FAIL",
            "attended_hardware",
            f"cannot import identity client: {type(exc).__name__}: {exc}",
        )

    expected = {
        "schema": protocol.SERVER_INFO_SCHEMA,
        "package": "flexiv-control",
        "package_version": flexiv_control_version,
        "protocol_id": protocol.PROTOCOL_ID,
        "protocol_fingerprint_sha256":
            protocol.PROTOCOL_FINGERPRINT_SHA256,
        "source_fingerprint_sha256": protocol.SOURCE_FINGERPRINT_SHA256,
    }
    robot = None
    try:
        robot = RemoteRobot(
            host,
            int(port),
            owner="oap-readonly-preflight",
            timeout=float(timeout_s),
            motion_timeout=float(timeout_s),
        )
        robot.connect()
        remote = robot.get_server_info()
    except ConnectionRefusedError as exc:
        return _check(
            "robot_server_identity",
            "FAIL",
            "attended_hardware",
            f"{host}:{port} refused the TCP connection: {exc}",
        )
    except (socket.timeout, socket.gaierror, TimeoutError) as exc:
        return _check(
            "robot_server_identity",
            "CANNOT_CHECK",
            "attended_hardware",
            f"reachability unknown for {host}:{port}: {type(exc).__name__}: {exc}",
        )
    except RemoteRobotError as exc:
        return _check(
            "robot_server_identity",
            "FAIL",
            "attended_hardware",
            (
                f"{host}:{port} does not expose the required read-only "
                f"trajectory protocol identity: {exc}"
            ),
        )
    except OSError as exc:
        return _check(
            "robot_server_identity",
            "CANNOT_CHECK",
            "attended_hardware",
            f"cannot establish reachability for {host}:{port}: {exc}",
        )
    except Exception as exc:
        return _check(
            "robot_server_identity",
            "FAIL",
            "attended_hardware",
            f"identity probe failed: {type(exc).__name__}: {exc}",
        )
    finally:
        if robot is not None:
            with contextlib.suppress(Exception):
                robot.close()

    compared_fields = (
        "schema",
        "package",
        "package_version",
        "protocol_id",
        "protocol_fingerprint_sha256",
        "source_fingerprint_sha256",
    )
    mismatches = [
        f"{field}: local={expected.get(field)!r}, remote={remote.get(field)!r}"
        for field in compared_fields
        if remote.get(field) != expected.get(field)
    ]
    if mismatches:
        return _check(
            "robot_server_identity",
            "FAIL",
            "attended_hardware",
            "client/server identity mismatch: " + "; ".join(mismatches),
        )
    try:
        from oap.loop.execute import controller_runtime_contract

        runtime_contract = controller_runtime_contract(remote)
        control_hz = runtime_contract["control_hz"]
        control_hz_source = runtime_contract["control_hz_source"]
        effective_limits_sha256 = runtime_contract[
            "effective_joint_limits"
        ]["sha256"]
    except Exception as exc:
        return _check(
            "robot_server_identity",
            "FAIL",
            "attended_hardware",
            "controller runtime timing contract is invalid: "
            f"{type(exc).__name__}: {exc}",
        )
    return _check(
        "robot_server_identity",
        "PASS",
        "attended_hardware",
        (
            f"{host}:{port}; version={remote['package_version']}; "
            f"protocol_id={remote['protocol_id']}; "
            "protocol_fingerprint_sha256="
            f"{remote['protocol_fingerprint_sha256']}; "
            f"source_fingerprint_sha256={remote['source_fingerprint_sha256']}; "
            f"control_hz={control_hz:.9g}; "
            f"control_hz_source={control_hz_source}; "
            f"effective_joint_limits_sha256={effective_limits_sha256}; "
            "no lease or motion"
        ),
    )


def run_preflight(
    cfg: PreflightConfig,
    *,
    warp_check: Callable[[], PreflightCheck] = _check_warp_gpu,
    external_env_check: Callable[[str | None, tuple[str, ...]], PreflightCheck] = _check_external_envs,
    executor_check: Callable[
        [Path | None, str | None],
        tuple[PreflightCheck, PreflightCheck],
    ] = _check_executor_readiness,
    camera_check: Callable[[Path], PreflightCheck] = _check_camera_device,
    robot_server_check: Callable[
        [str, int, float],
        PreflightCheck,
    ] = _check_robot_server_identity,
) -> PreflightReport:
    """Run every check without acquiring robot control or sending motion."""
    checks: list[PreflightCheck] = []
    objects: list[Any] | None = None
    program: TaskProgram | None = None
    digest: str | None = None

    try:
        objects = load_scene_manifest(Path(cfg.bundle_manifest))
    except Exception as exc:
        checks.append(_check(
            "bundle",
            "FAIL",
            "software",
            f"{type(exc).__name__}: {exc}",
        ))
    else:
        checks.append(_check(
            "bundle",
            "PASS",
            "software",
            f"parsed {len(objects)} objects from {cfg.bundle_manifest}",
        ))

    try:
        text = Path(cfg.program_json).read_text(encoding="utf-8")
        program = TaskProgram.from_json(text)
        if not program.stages:
            raise ValueError("program has no stages")
        if not program.has_verifiable_success():
            raise ValueError(
                "every stage needs a non-empty, fully measurable geometric "
                "terminal"
            )
        from oap.program.synthesis import validate_program

        validation_errors = validate_program(program)
        if validation_errors:
            raise ValueError(
                "invalid_program:" + ";".join(validation_errors[:4])
            )
    except Exception as exc:
        checks.append(_check(
            "program",
            "FAIL",
            "software",
            f"{type(exc).__name__}: {exc}",
        ))
    else:
        checks.append(_check(
            "program",
            "PASS",
            "software",
            f"parsed {len(program.stages)} stages; provenance={program.provenance!r}",
        ))

    if objects is None or program is None:
        checks.append(_check(
            "program_grounding",
            "CANNOT_CHECK",
            "software",
            "requires both a parsed bundle and parsed program",
        ))
    else:
        try:
            anchors, missing, route = _ground_program(program, objects)
            if missing:
                raise ValueError(f"ungroundable anchors: {','.join(missing)}")
        except Exception as exc:
            checks.append(_check(
                "program_grounding",
                "FAIL",
                "software",
                f"{type(exc).__name__}: {exc}",
            ))
        else:
            checks.append(_check(
                "program_grounding",
                "PASS",
                "software",
                f"{len(program.referenced_anchors())} references grounded via {route}; "
                f"scene vocabulary has {len(anchors.names())} anchors",
            ))

    if program is None:
        checks.append(_check(
            "program_approval",
            "CANNOT_CHECK",
            "software",
            "requires a parsed program",
        ))
    else:
        # Publish the actual canonical identity even when approval is absent or
        # mismatched, so the operator has the exact digest to review next.
        digest = program_sha256(program)
        try:
            require_program_approval(
                program,
                execute=True,
                approved_program_sha256=cfg.approved_program_sha256,
            )
        except SafetyError as exc:
            checks.append(_check(
                "program_approval",
                "FAIL",
                "software",
                str(exc),
            ))
        else:
            checks.append(_check(
                "program_approval",
                "PASS",
                "software",
                f"exact canonical SHA-256 approved: {digest}",
            ))

    checks.append(_check_object_held_evidence_calibration(
        program,
        cfg.hardware_evidence_calibration_json,
    ))
    checks.append(_check_asset_checksums(cfg.asset_checksums, cfg.checksum_root))
    checks.extend(executor_check(
        cfg.joint_executor_verification_json,
        cfg.joint_executor_verification_sha256,
    ))
    checks.append(warp_check())
    checks.append(external_env_check(cfg.external_envs, cfg.required_external_tools))
    checks.append(camera_check(cfg.camera_device))
    checks.append(robot_server_check(
        cfg.robot_host,
        cfg.robot_port,
        cfg.robot_timeout_s,
    ))

    software = [check for check in checks if check.scope == "software"]
    hardware = [check for check in checks if check.scope == "attended_hardware"]
    software_ready = bool(software) and all(check.status == "PASS" for check in software)
    hardware_ready = bool(hardware) and all(check.status == "PASS" for check in hardware)
    if software_ready and hardware_ready:
        readiness = "READY_FOR_ATTENDED_PREFLIGHT"
    elif software_ready:
        readiness = "SOFTWARE_READY_HARDWARE_PENDING"
    else:
        readiness = "NOT_READY"
    return PreflightReport(
        checks=tuple(checks),
        program_sha256=digest,
        software_ready=software_ready,
        attended_hardware_checks_passed=hardware_ready,
        readiness=readiness,
    )
