"""One-shot, signed authorization for an attended real-motion session.

Readiness reports are evidence, not capabilities.  This module turns a
reviewed evidence set into a narrowly-scoped capability: an Ed25519-signed,
short-lived token bound to the exact program, generated scene bundle, source
commit, controller protocol/source identity, rig calibrations, measured
limits, resolved planner/tracker runtime profile, ObjectHeld calibration,
attended executor latch, and operator/E-stop packet.  The token is consumed
atomically before a controller lease can be acquired.

The trust root and replay ledger deliberately have no CLI overrides.  A dry
run never reads a token or writes the ledger.
"""
from __future__ import annotations

import base64
import binascii
import json
import math
import os
import subprocess
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from oap.loop.evidence_calibration import (
    SCHEMA as OBJECT_HELD_SCHEMA,
    canonical_runtime_hardware_identity_sha256,
    load_object_held_evidence_calibration,
)
from oap.loop.executor_readiness import require_executor_readiness
from oap.loop.planner_profile import PlannerProfile
from oap.loop.safety import SafetyError
from oap.loop.execution_prefix import resolve_execution_prefix
from oap.reconstruct.pose import load_fixed_camera_calibration
from oap.utils.io import (
    package_config_path,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)

if TYPE_CHECKING:
    from oap.loop.runner import LoopConfig

TOKEN_SCHEMA = "oap_execution_authorization_v2"
TRUST_SCHEMA = "oap_execution_authority_trust_v1"
OPERATOR_ESTOP_SCHEMA = "oap_operator_estop_packet_v1"
TOKEN_AUDIENCE = "oap-real-motion"
MAX_TOKEN_LIFETIME_S = 15 * 60
# Deliberately unset while parameter/method screening is in progress.  A
# reviewed release commit must replace ``None`` with exactly one complete
# profile.  Until then authorization collection and real launch both fail
# closed; old experimental values are never silently treated as production.
REQUIRED_REAL_PLANNER_PROFILE: PlannerProfile | None = None

# System-owned deployment state.  Neither path is configurable by the run CLI.
_TRUST_STORE_PATH = Path("/etc/oap/execution_authority.json")
_LEDGER_DIR = Path.home() / ".local/state/oap/execution-token-ledger"

_TOKEN_KEYS = {
    "schema",
    "token_id",
    "audience",
    "issued_at",
    "not_before",
    "expires_at",
    "claims",
    "signature",
}
_CLAIM_KEYS = {
    "oap_source",
    "program",
    "bundle",
    "robot_server",
    "planner_deployment",
    "resolved_runtime_profile",
    "rig_calibrations",
    "execution_policy",
    "observation_limits",
    "object_held_evidence_calibration",
    "joint_executor_verification_latch",
    "operator_estop_packet",
}
_SERVER_PROTOCOL_IDENTITY_FIELDS = (
    "schema",
    "package",
    "package_version",
    "protocol_id",
    "protocol_fingerprint_sha256",
    "source_fingerprint_sha256",
    "control_hz",
    "active_safety_profile",
    "effective_joint_limits",
    "gripper_limits",
)
_SERVER_IDENTITY_FIELDS = (
    *_SERVER_PROTOCOL_IDENTITY_FIELDS,
    "runtime_hardware_identity_sha256",
)
_OPERATOR_ATTESTATIONS = (
    "operator_present",
    "estop_in_hand",
    "workspace_clear",
    "attended_run",
)
_BUNDLE_PATH_FIELDS = (
    "mesh",
    "collision",
    "refined_json",
    "priors_json",
    "visual_mesh",
    "texture",
    "tracking_metadata",
)

__all__ = [
    "ExecutionAuthorizationReceipt",
    "OPERATOR_ESTOP_SCHEMA",
    "TOKEN_SCHEMA",
    "collect_execution_authorization_claims",
    "require_and_consume_execution_authorization",
    "require_authorized_calibration_hardware_identity",
    "require_authorized_server_identity",
    "require_consumed_execution_authorization",
]


@dataclass(frozen=True)
class _VerifiedToken:
    path: Path
    sha256: str
    token_id: str
    key_id: str
    issued_at: datetime
    not_before: datetime
    expires_at: datetime
    claims: dict[str, Any]


@dataclass(frozen=True)
class ExecutionAuthorizationReceipt:
    """Proof that one token was validated and atomically consumed."""

    token_id: str
    token_sha256: str
    authority_key_id: str
    consumed_at: str
    expires_at: str
    ledger_record: str

    def to_dict(self) -> dict[str, str]:
        return {
            "schema": "oap_execution_authorization_receipt_v1",
            "token_id": self.token_id,
            "token_sha256": self.token_sha256,
            "authority_key_id": self.authority_key_id,
            "consumed_at": self.consumed_at,
            "expires_at": self.expires_at,
            "ledger_record": self.ledger_record,
        }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp(document: Mapping[str, Any], name: str) -> datetime:
    raw = document.get(name)
    if not isinstance(raw, str) or not raw.strip():
        raise SafetyError(f"execution authorization {name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SafetyError(
            f"execution authorization {name} is not a valid ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None:
        raise SafetyError(f"execution authorization {name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _canonical_signed_bytes(document: Mapping[str, Any]) -> bytes:
    payload = {key: value for key, value in document.items() if key != "signature"}
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _load_trusted_public_key(path: Path, key_id: str) -> bytes:
    resolved = Path(path).expanduser().resolve()
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SafetyError(
            f"execution authority trust store is unavailable or invalid at {resolved}: {exc}"
        ) from exc
    if not isinstance(document, dict) or document.get("schema") != TRUST_SCHEMA:
        raise SafetyError(
            f"execution authority trust store must use schema {TRUST_SCHEMA!r}"
        )
    keys = document.get("keys")
    if not isinstance(keys, list):
        raise SafetyError("execution authority trust store keys must be a list")
    matches = [
        entry
        for entry in keys
        if isinstance(entry, dict)
        and entry.get("key_id") == key_id
        and entry.get("algorithm") == "ed25519"
        and entry.get("enabled") is True
    ]
    if len(matches) != 1:
        raise SafetyError(
            f"execution authorization key {key_id!r} is not uniquely enabled "
            f"in the system trust store"
        )
    raw = matches[0].get("public_key_b64")
    if not isinstance(raw, str):
        raise SafetyError("trusted Ed25519 public key must be base64 text")
    try:
        decoded = base64.b64decode(raw, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise SafetyError("trusted Ed25519 public key is not valid base64") from exc
    if len(decoded) != 32:
        raise SafetyError("trusted Ed25519 public key must contain exactly 32 bytes")
    return decoded


def _verify_token(
    path: Path,
    *,
    now: datetime,
    trust_store_path: Path,
) -> _VerifiedToken:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise SafetyError(f"execution authorization token does not exist: {resolved}")
    raw_bytes = resolved.read_bytes()
    try:
        document = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise SafetyError(f"execution authorization token is invalid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise SafetyError("execution authorization token must be a JSON object")
    unknown = sorted(set(document) - _TOKEN_KEYS)
    missing = sorted(_TOKEN_KEYS - set(document))
    if unknown or missing:
        raise SafetyError(
            "execution authorization token fields are not exact "
            f"(missing={missing}, unknown={unknown})"
        )
    if document.get("schema") != TOKEN_SCHEMA:
        raise SafetyError(
            f"execution authorization schema must be {TOKEN_SCHEMA!r}"
        )
    if document.get("audience") != TOKEN_AUDIENCE:
        raise SafetyError(
            f"execution authorization audience must be {TOKEN_AUDIENCE!r}"
        )
    token_id = str(document.get("token_id", ""))
    try:
        parsed_id = uuid.UUID(token_id)
    except (ValueError, AttributeError) as exc:
        raise SafetyError("execution authorization token_id must be a UUID") from exc
    if str(parsed_id) != token_id:
        raise SafetyError("execution authorization token_id must be canonical lowercase UUID")

    issued_at = _timestamp(document, "issued_at")
    not_before = _timestamp(document, "not_before")
    expires_at = _timestamp(document, "expires_at")
    if not_before < issued_at:
        raise SafetyError("execution authorization not_before precedes issued_at")
    lifetime_s = (expires_at - issued_at).total_seconds()
    if lifetime_s <= 0.0 or lifetime_s > MAX_TOKEN_LIFETIME_S:
        raise SafetyError(
            "execution authorization lifetime must be positive and no longer "
            f"than {MAX_TOKEN_LIFETIME_S} seconds"
        )
    if now < not_before:
        raise SafetyError("execution authorization is not yet valid")
    if now >= expires_at:
        raise SafetyError("execution authorization has expired")

    claims = document.get("claims")
    if not isinstance(claims, dict) or set(claims) != _CLAIM_KEYS:
        raise SafetyError(
            "execution authorization claims must contain exactly "
            f"{sorted(_CLAIM_KEYS)}"
        )
    signature = document.get("signature")
    if (
        not isinstance(signature, dict)
        or set(signature) != {"algorithm", "key_id", "signature_b64"}
        or signature.get("algorithm") != "ed25519"
        or not isinstance(signature.get("key_id"), str)
        or not signature["key_id"].strip()
        or not isinstance(signature.get("signature_b64"), str)
    ):
        raise SafetyError("execution authorization has an invalid Ed25519 signature block")
    public_key_bytes = _load_trusted_public_key(
        trust_store_path, signature["key_id"]
    )
    try:
        signature_bytes = base64.b64decode(
            signature["signature_b64"], validate=True
        )
    except (ValueError, binascii.Error) as exc:
        raise SafetyError("execution authorization signature is not valid base64") from exc
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except ImportError as exc:
        raise SafetyError(
            "cryptography with Ed25519 support is required for real motion"
        ) from exc
    try:
        Ed25519PublicKey.from_public_bytes(public_key_bytes).verify(
            signature_bytes,
            _canonical_signed_bytes(document),
        )
    except (InvalidSignature, ValueError) as exc:
        raise SafetyError("execution authorization signature verification failed") from exc
    return _VerifiedToken(
        path=resolved,
        sha256=sha256_bytes(raw_bytes),
        token_id=token_id,
        key_id=signature["key_id"],
        issued_at=issued_at,
        not_before=not_before,
        expires_at=expires_at,
        claims=dict(claims),
    )


def _inspect_oap_source() -> dict[str, Any]:
    """Return the exact clean Git identity of the imported OaP source."""
    repository = Path(__file__).resolve().parents[3]

    def git(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(repository), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=10.0,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise SafetyError(
                "real execution requires an exact Git source identity for "
                f"{repository}: {exc}"
            ) from exc
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    dirty = git(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "src/oap",
        "pyproject.toml",
    )
    if dirty:
        raise SafetyError(
            "OaP source/config is dirty; commit the exact real-motion "
            "build before issuing an authorization token"
        )
    return {
        "git_commit": commit,
        "git_tree": tree,
        "tracked_source_clean": True,
    }


def _bundle_artifact_path(
    root: Path,
    raw: Any,
    *,
    role: str,
) -> tuple[str, Path]:
    if not isinstance(raw, str):
        raise SafetyError(f"scene bundle artifact {role!r} is not a path")
    posix = PurePosixPath(raw)
    if (
        posix.is_absolute()
        or "\\" in raw
        or raw != posix.as_posix()
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise SafetyError(
            f"scene bundle artifact path is not canonical: {raw!r}"
        )
    artifact = (root / Path(*posix.parts)).resolve()
    try:
        artifact.relative_to(root)
    except ValueError as exc:
        raise SafetyError(
            f"scene bundle artifact escapes bundle root: {raw!r}"
        ) from exc
    if not artifact.is_file():
        raise SafetyError(f"scene bundle artifact is missing: {artifact}")
    return raw, artifact


def _bind_bundle_artifact(
    artifacts: dict[str, str],
    root: Path,
    raw: Any,
    *,
    role: str,
    expected_sha256: Any,
) -> None:
    relative, artifact = _bundle_artifact_path(root, raw, role=role)
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_sha256
        )
    ):
        raise SafetyError(
            f"scene bundle artifact {relative!r} has no canonical "
            "declared SHA-256"
        )
    actual_sha256 = sha256_file(artifact)
    if actual_sha256 != expected_sha256:
        raise SafetyError(
            f"scene bundle artifact hash mismatch for {relative!r}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    # A path can be referenced by more than one role, but it appears exactly
    # once in the authorization identity and checksum manifest.
    artifacts[relative] = actual_sha256


def _bundle_identity(path: Path) -> dict[str, Any]:
    manifest = Path(path).expanduser().resolve()
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SafetyError(f"cannot bind scene bundle manifest {manifest}: {exc}") from exc
    if not isinstance(document, dict):
        raise SafetyError("scene bundle manifest must be a JSON object")
    root = manifest.parent.resolve()
    artifacts: dict[str, str] = {}

    camera_calibration = document.get("camera_calibration")
    if not isinstance(camera_calibration, Mapping):
        raise SafetyError(
            "scene bundle manifest has no camera_calibration object to bind"
        )
    _bind_bundle_artifact(
        artifacts,
        root,
        camera_calibration.get("config"),
        role="camera_calibration.config",
        expected_sha256=camera_calibration.get("sha256"),
    )

    objects = document.get("objects")
    if not isinstance(objects, list) or not objects:
        raise SafetyError("scene bundle manifest has no object artifacts to bind")
    for entry in objects:
        if not isinstance(entry, dict):
            raise SafetyError("scene bundle object entry must be an object")
        declared = entry.get("artifact_sha256")
        if not isinstance(declared, Mapping):
            raise SafetyError(
                "scene bundle object has no artifact_sha256 object to bind"
            )
        for field in _BUNDLE_PATH_FIELDS:
            raw = entry.get(field)
            if raw is None:
                continue
            expected_sha256 = declared.get(raw) if isinstance(raw, str) else None
            _bind_bundle_artifact(
                artifacts,
                root,
                raw,
                role=field,
                expected_sha256=expected_sha256,
            )
    return {
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "artifacts_sha256": dict(sorted(artifacts.items())),
    }


def _artifact_ref(path: Path) -> dict[str, str]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise SafetyError(f"authorization-bound artifact does not exist: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _interpreter_ref(path: Path | None, *, label: str) -> dict[str, Any]:
    """Bind one external interpreter and its effective installed package set."""
    if path is None:
        raise SafetyError(f"real execution requires {label} interpreter identity")
    executable = Path(path).expanduser().resolve()
    try:
        frozen = subprocess.run(
            [
                str(executable),
                "-m",
                "pip",
                "--disable-pip-version-check",
                "freeze",
                "--all",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60.0,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise SafetyError(
            f"cannot fingerprint {label} environment at {executable}: {exc}"
        ) from exc
    normalized = "\n".join(
        sorted(line.strip() for line in frozen.splitlines() if line.strip())
    )
    result = {
        **_artifact_ref(executable),
        "packages_sha256": sha256_bytes(normalized.encode("utf-8")),
    }
    conda_history = executable.parent.parent / "conda-meta/history"
    result["conda_history"] = (
        _artifact_ref(conda_history) if conda_history.is_file() else None
    )
    return result


def _git_checkout_ref(path: Path, *, label: str) -> dict[str, Any]:
    """Bind one checkout, including any modified/untracked runtime files."""
    root = Path(path).expanduser().resolve()

    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=20.0,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise SafetyError(f"cannot fingerprint {label} checkout: {exc}") from exc

    changed = set(git("diff", "--name-only", "HEAD").splitlines())
    changed.update(
        git("ls-files", "--others", "--exclude-standard").splitlines()
    )
    changed_refs: dict[str, Any] = {}
    for relative in sorted(name for name in changed if name):
        candidate = root / relative
        changed_refs[relative] = (
            _artifact_ref(candidate) if candidate.is_file() else {"deleted": True}
        )
    return {
        "path": str(root),
        "git_commit": git("rev-parse", "HEAD"),
        "git_tree": git("rev-parse", "HEAD^{tree}"),
        "changed_files": changed_refs,
    }


def _sam3_ref() -> dict[str, Any]:
    """Bind the exact cached ``facebook/sam3`` revision and model files."""
    hub = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
            / "hub",
        )
    ).expanduser().resolve()
    root = hub / "models--facebook--sam3"
    try:
        revision = (root / "refs/main").read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise SafetyError(f"cannot bind cached facebook/sam3 revision: {exc}") from exc
    try:
        revision_is_hex = len(revision) == 40 and int(revision, 16) >= 0
    except ValueError:
        revision_is_hex = False
    if not revision_is_hex:
        raise SafetyError("cached facebook/sam3 revision is not a Git SHA")
    snapshot = root / "snapshots" / revision
    files = {
        path.name: sha256_file(path)
        for path in sorted(snapshot.iterdir())
        if path.is_file()
    }
    if not revision or not files:
        raise SafetyError("cached facebook/sam3 snapshot is incomplete")
    service = (
        Path.home()
        / "countersim-current/services/sam3_service/serve_hf_sam3.py"
    )
    return {
        "model_id": "facebook/sam3",
        "revision": revision,
        "files_sha256": files,
        "service": _artifact_ref(service),
    }


def _tracker_runtime_ref(cfg: "LoopConfig") -> dict[str, Any]:
    """Bind external tracking code, environments, models, and quality path."""
    oap = Path(__file__).resolve().parents[1]
    timeout_s = float(cfg.foundationpose_timeout_s)
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise SafetyError("FoundationPose timeout must be finite and positive")
    checkpoint_timeout_raw = getattr(
        cfg,
        "foundationpose_checkpoint_timeout_s",
        None,
    )
    if checkpoint_timeout_raw is None:
        raise SafetyError(
            "real execution requires an explicit FoundationPose checkpoint "
            "timeout"
        )
    checkpoint_timeout_s = float(checkpoint_timeout_raw)
    if (
        not math.isfinite(checkpoint_timeout_s)
        or checkpoint_timeout_s <= 0.0
    ):
        raise SafetyError(
            "FoundationPose checkpoint timeout must be finite and positive"
        )
    result: dict[str, Any] = {
        "mode": str(cfg.foundationpose_mode),
        "subject_pose_policy": str(cfg.subject_pose_policy),
        "seed_from_obb": bool(cfg.fp_seed_from_obb),
        "timeout_s": timeout_s,
        "checkpoint_timeout_s": checkpoint_timeout_s,
        "observer_environment": _interpreter_ref(
            cfg.observer_python,
            label="observer",
        ),
        "sam3": _sam3_ref(),
        "quality_code": {
            name: _artifact_ref(oap / relative)
            for name, relative in {
                "continuous_tracking": "loop/continuous_tracking.py",
                "foundationpose_worker": "workers/foundationpose_worker.py",
                "zed_stream_worker": (
                    "reconstruct/payloads/zed_stream_worker.py"
                ),
            }.items()
        },
    }
    if (
        result["mode"] == "actual"
        and result["subject_pose_policy"] == "foundationpose"
    ):
        if cfg.foundationpose_root is None:
            raise SafetyError("FoundationPose tracking requires its checkout root")
        root = Path(cfg.foundationpose_root).expanduser().resolve()
        weights = {
            str(path.relative_to(root)): sha256_file(path)
            for path in sorted(root.glob("weights/*/model_best.pth"))
            if path.is_file()
        }
        if not weights:
            raise SafetyError("FoundationPose deployment has no model weights")
        result["foundationpose"] = {
            "environment": _interpreter_ref(
                cfg.foundationpose_python,
                label="FoundationPose",
            ),
            "checkout": _git_checkout_ref(root, label="FoundationPose"),
            "weights_sha256": weights,
        }
    else:
        result["foundationpose"] = None
    return result


def _required_real_planner_profile() -> PlannerProfile:
    profile = REQUIRED_REAL_PLANNER_PROFILE
    if profile is None:
        raise SafetyError(
            "real-execution planner profile is not frozen/reviewed; "
            "parameter screening may continue in simulation, but no "
            "authorization request or real launch may be created"
        )
    if not isinstance(profile, PlannerProfile):
        raise SafetyError(
            "real-execution planner profile freeze is malformed"
        )
    return profile


def _resolved_runtime_profile(cfg: "LoopConfig") -> dict[str, Any]:
    """Resolve the exact planner/tracker knobs used by the eventual run."""
    from oap.loop.plan import validate_num_knots
    from oap.loop.sampling import validate_pool_size
    from oap.twin.batched_rollout import (
        DEFAULT_CEM_ELITE_FRACTION,
        DEFAULT_CEM_MIN_STD_FRACTION,
        DEFAULT_CEM_ROUNDS,
        PREDICTIVE_SAMPLER_MPPI_HALTON_MODES,
        PREDICTIVE_SAMPLER_SINGLE_SCALE,
        PREDICTIVE_TWO_SCALE_LOCAL_SIGMA_FRACTION,
        plan_dt_from_env,
        validate_horizon_steps,
        validate_local_sigma_fraction,
        validate_predictive_sampler_mode,
    )

    pool_size = validate_pool_size(cfg.pool_size)
    horizon_steps = validate_horizon_steps(cfg.horizon_steps)
    num_knots = validate_num_knots(cfg.num_knots)
    sigma_fraction = validate_local_sigma_fraction(
        cfg.sigma_fraction
    )
    sampler_mode = validate_predictive_sampler_mode(
        getattr(cfg, "sampler_mode", PREDICTIVE_SAMPLER_SINGLE_SCALE)
    )
    local_sigma_fraction = validate_local_sigma_fraction(
        getattr(
            cfg,
            "local_sigma_fraction",
            PREDICTIVE_TWO_SCALE_LOCAL_SIGMA_FRACTION,
        )
    )
    cem_rounds = int(getattr(cfg, "cem_rounds", DEFAULT_CEM_ROUNDS))
    cem_elite_fraction = float(
        getattr(
            cfg,
            "cem_elite_fraction",
            DEFAULT_CEM_ELITE_FRACTION,
        )
    )
    cem_min_std_fraction = float(
        getattr(
            cfg,
            "cem_min_std_fraction",
            DEFAULT_CEM_MIN_STD_FRACTION,
        )
    )
    requested_prefix_steps = getattr(
        cfg,
        "execution_prefix_steps",
        None,
    )
    if requested_prefix_steps is None:
        raise SafetyError(
            "real execution requires explicit execution_prefix_steps; "
            "a fraction-only profile is not authorizable"
        )
    prefix = resolve_execution_prefix(
        horizon_steps=horizon_steps,
        execution_prefix_fraction=cfg.execution_prefix_fraction,
        execution_prefix_steps=requested_prefix_steps,
    )
    prefix_fraction = prefix.effective_fraction
    try:
        actual_real_profile = PlannerProfile(
            pool_size=pool_size,
            horizon_steps=horizon_steps,
            num_knots=num_knots,
            sigma_fraction=sigma_fraction,
            sampler_mode=sampler_mode,
            local_sigma_fraction=local_sigma_fraction,
            cem_rounds=cem_rounds,
            cem_elite_fraction=cem_elite_fraction,
            cem_min_std_fraction=cem_min_std_fraction,
            execution_prefix_steps=prefix.resolved_steps,
            max_stage_cycles=int(cfg.max_stage_cycles),
        )
    except ValueError as exc:
        raise SafetyError(
            f"resolved planner profile is invalid: {exc}"
        ) from exc
    required_real_profile = _required_real_planner_profile()
    if actual_real_profile != required_real_profile:
        raise SafetyError(
            "real execution requires the frozen GPU CEM-MPC profile "
            f"{required_real_profile.to_dict()}; got "
            f"{actual_real_profile.to_dict()}. Simulation ablations may "
            "override these values."
        )
    plan_dt_override_s = plan_dt_from_env()
    if plan_dt_override_s is not None:
        raise SafetyError(
            "real execution forbids OAP_PLAN_DT; planning and execution "
            "must use the model timestep"
        )
    profile_dict = actual_real_profile.to_dict()
    return {
        "planner": {
            "pool_size": pool_size,
            "optimization_rounds": cem_rounds,
            "algorithm": (
                "predictive_sampling" if cem_rounds == 1 else "cem"
            ),
            "total_rollout_budget": pool_size,
            "samples_per_round": pool_size // cem_rounds,
            "physics_backend": "mjwarp_gpu",
            "horizon_steps": horizon_steps,
            "max_stage_cycles": int(cfg.max_stage_cycles),
            "seed": int(cfg.seed),
            "plan_dt_override_s": plan_dt_override_s,
            "num_knots": num_knots,
            "execution_prefix_fraction": prefix_fraction,
            "execution_prefix_steps": prefix.resolved_steps,
            "resolved_execution_prefix_steps": prefix.resolved_steps,
            "profile": profile_dict,
            "profile_sha256": actual_real_profile.sha256,
            "sampling_profile": {
                "algorithm": (
                    "predictive_sampling" if cem_rounds == 1 else "cem"
                ),
                "optimization_rounds": cem_rounds,
                "samples_per_round": pool_size // cem_rounds,
                "cem_elite_fraction": cem_elite_fraction,
                "cem_min_std_fraction": cem_min_std_fraction,
                "sampler_mode": sampler_mode,
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
            },
        },
        "tracker": _tracker_runtime_ref(cfg),
    }


def _rig_calibrations(cfg: "LoopConfig") -> dict[str, Any]:
    fixed_matrix, fixed_source, camera_serial, fixed_sha256 = (
        load_fixed_camera_calibration()
    )
    del fixed_matrix
    home_path = (
        Path(cfg.home_posture_json)
        if cfg.home_posture_json is not None
        else package_config_path("calibration/lab_home_posture.json")
    )
    if cfg.table_spec_json is None:
        raise SafetyError(
            "real execution requires a measured table-spec JSON bound by the "
            "authorization token"
        )
    if cfg.real_table_z is None or not math.isfinite(float(cfg.real_table_z)):
        raise SafetyError(
            "real execution requires a finite measured real_table_z bound by "
            "the authorization token"
        )
    if cfg.sim_tcp_m is None or not math.isfinite(float(cfg.sim_tcp_m)):
        raise SafetyError(
            "real execution requires a finite calibrated sim_tcp_m bound by "
            "the authorization token"
        )
    offset = [float(value) for value in cfg.grasp_tcp_offset]
    if len(offset) != 3 or not all(math.isfinite(value) for value in offset):
        raise SafetyError("grasp_tcp_offset must contain exactly three finite values")
    return {
        "fixed_camera": {
            "path": str(Path(fixed_source).resolve()),
            "sha256": fixed_sha256,
            "camera_serial": camera_serial,
        },
        "home_posture": _artifact_ref(home_path),
        "table_spec": _artifact_ref(Path(cfg.table_spec_json)),
        "gn01_mount": _artifact_ref(
            package_config_path("robot/gn01_mount_relative.yaml")
        ),
        "gn01_tool": _artifact_ref(package_config_path("robot/gn01_tool.yaml")),
        "real_table_z_m": float(cfg.real_table_z),
        "sim_tcp_m": float(cfg.sim_tcp_m),
        "sim_tcp_anchor": str(cfg.sim_tcp_anchor),
        "grasp_tcp_offset_tool_m": offset,
    }


def _observation_limits(cfg: "LoopConfig") -> dict[str, float]:
    names = (
        "max_observation_age_s",
        "max_camera_robot_skew_s",
        "max_plan_staleness_s",
        "max_final_joint_tracking_error_rad",
        "min_terminal_anchor_reliability",
    )
    result: dict[str, float] = {}
    for name in names:
        raw = getattr(cfg, name)
        if raw is None or not math.isfinite(float(raw)):
            raise SafetyError(
                f"real execution requires measured {name} bound by the "
                "authorization token"
            )
        result[name] = float(raw)
    return result


def _execution_policy(cfg: "LoopConfig") -> dict[str, Any]:
    """Bind motion-affecting bootstrap/observation choices, not planner taste."""
    return {
        "home_first": bool(cfg.home_first),
        "per_chunk_confirm": bool(cfg.per_chunk_confirm),
        "manual_post_experiment_restore": bool(
            cfg.manual_post_experiment_restore
        ),
        "foundationpose_mode": str(cfg.foundationpose_mode),
        "subject_pose_policy": str(cfg.subject_pose_policy),
        "relocalize_all_each_chunk": bool(cfg.relocalize_all_each_chunk),
        "held_verification_enabled": not bool(cfg.disable_held_verify),
    }


def _planner_deployment(cfg: "LoopConfig") -> dict[str, Any]:
    required_real_profile = _required_real_planner_profile()
    values = {
        "url": cfg.remote_planner_url,
        "model_sha256": cfg.remote_planner_model_sha256,
        "assets_sha256": cfg.remote_planner_assets_sha256,
        "planner_sha256": cfg.remote_planner_sha256,
        "service_instance_id": cfg.remote_planner_service_instance_id,
        "deadline_s": cfg.remote_planner_deadline_s,
    }
    provided = {name: value is not None for name, value in values.items()}
    if any(provided.values()) and not all(provided.values()):
        missing = sorted(name for name, present in provided.items() if not present)
        raise SafetyError(
            "remote planner authorization identity is incomplete; missing "
            + ", ".join(missing)
        )
    if not any(provided.values()):
        return {
            "mode": "local_gpu",
            "profile_sha256": required_real_profile.sha256,
        }
    return {
        "mode": "remote_h100",
        "url": str(values["url"]),
        "model_sha256": str(values["model_sha256"]),
        "assets_sha256": str(values["assets_sha256"]),
        "planner_sha256": str(values["planner_sha256"]),
        "service_instance_id": str(values["service_instance_id"]),
        "deadline_s": float(values["deadline_s"]),
        "profile_sha256": required_real_profile.sha256,
    }


def _object_held_calibration(cfg: "LoopConfig") -> dict[str, Any]:
    if cfg.hardware_evidence_calibration_json is None:
        raise SafetyError(
            "real execution requires the measured ObjectHeld calibration even "
            "when the current program does not consume ObjectHeld"
        )
    try:
        calibration = load_object_held_evidence_calibration(
            cfg.hardware_evidence_calibration_json
        )
    except ValueError as exc:
        raise SafetyError(str(exc)) from exc
    return {
        "schema": OBJECT_HELD_SCHEMA,
        "path": str(calibration.path),
        "sha256": calibration.sha256,
        "hardware_id": calibration.hardware_id,
        "source": calibration.source,
        "captured_at": calibration.captured_at,
        "values": calibration.loop_config_values(),
    }


def _executor_latch(cfg: "LoopConfig") -> dict[str, Any]:
    readiness = require_executor_readiness(
        cfg.joint_executor_verification_json,
        cfg.joint_executor_verification_sha256,
    )
    latch = readiness["verification_latch"]
    software = readiness["flexiv_control"]
    return {
        "path": str(Path(latch["path"]).resolve()),
        "sha256": latch["sha256"],
        "hardware_id": latch["hardware_id"],
        "verified_by": latch["verified_by"],
        "verified_at": latch["verified_at"],
        "executor_software_fingerprint_sha256": software[
            "software_fingerprint_sha256"
        ],
    }


def _operator_estop_packet(
    path: Path | None,
    *,
    token_expires_at: datetime | None,
    now: datetime | None = None,
) -> dict[str, Any]:
    if path is None:
        raise SafetyError(
            "real execution requires an operator/E-stop packet bound by the "
            "authorization token"
        )
    resolved = Path(path).expanduser().resolve()
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SafetyError(f"operator/E-stop packet is unavailable or invalid: {exc}") from exc
    allowed = {
        "schema",
        "operator_id",
        "robot_hardware_id",
        "attested_at",
        "expires_at",
        "attestations",
    }
    if not isinstance(document, dict) or set(document) != allowed:
        raise SafetyError(
            "operator/E-stop packet must contain exactly "
            f"{sorted(allowed)}"
        )
    if document.get("schema") != OPERATOR_ESTOP_SCHEMA:
        raise SafetyError(
            f"operator/E-stop packet schema must be {OPERATOR_ESTOP_SCHEMA!r}"
        )
    for name in ("operator_id", "robot_hardware_id"):
        if not isinstance(document.get(name), str) or not document[name].strip():
            raise SafetyError(f"operator/E-stop packet {name} must be non-empty")
    attested_at = _timestamp(document, "attested_at")
    expires_at = _timestamp(document, "expires_at")
    if expires_at <= attested_at:
        raise SafetyError("operator/E-stop packet expiry must follow attestation")
    if now is not None:
        current = now.astimezone(timezone.utc)
        if attested_at > current:
            raise SafetyError("operator/E-stop packet attestation is in the future")
        if current >= expires_at:
            raise SafetyError("operator/E-stop packet has expired")
    if token_expires_at is not None and expires_at < token_expires_at:
        raise SafetyError(
            "operator/E-stop packet expires before the execution token"
        )
    attestations = document.get("attestations")
    if not isinstance(attestations, dict) or set(attestations) != set(
        _OPERATOR_ATTESTATIONS
    ):
        raise SafetyError(
            "operator/E-stop packet attestations must contain exactly "
            f"{list(_OPERATOR_ATTESTATIONS)}"
        )
    for name in _OPERATOR_ATTESTATIONS:
        if attestations.get(name) is not True:
            raise SafetyError(
                f"operator/E-stop packet attestation {name!r} is not true"
            )
    return {
        **_artifact_ref(resolved),
        "operator_id": document["operator_id"],
        "robot_hardware_id": document["robot_hardware_id"],
        "attested_at": _iso_utc(attested_at),
        "expires_at": _iso_utc(expires_at),
        "attestations": dict(attestations),
    }


def _read_server_identity(host: str, port: int) -> dict[str, Any]:
    """Read identity without entering the controller context or taking a lease."""
    try:
        from flexiv_control.client import RemoteRobot
    except ImportError as exc:
        raise SafetyError(
            "cannot import flexiv-control identity client for authorization"
        ) from exc
    robot = RemoteRobot(
        str(host),
        int(port),
        owner="oap-authorization-readonly",
        timeout=2.0,
        motion_timeout=2.0,
    )
    try:
        robot.connect()
        remote = robot.get_server_info()
    except Exception as exc:
        raise SafetyError(
            "cannot verify the authorization-bound robot server identity "
            f"without taking a lease: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        try:
            robot.close()
        except Exception:
            pass
    if not isinstance(remote, Mapping):
        raise SafetyError("robot server identity response is not an object")
    missing = [
        name
        for name in _SERVER_PROTOCOL_IDENTITY_FIELDS
        if remote.get(name) is None
    ]
    if missing:
        raise SafetyError(f"robot server identity is missing fields: {missing}")
    try:
        runtime_hardware_identity_sha256 = (
            canonical_runtime_hardware_identity_sha256(
                remote.get("runtime_hardware_identity")
            )
        )
    except ValueError as exc:
        raise SafetyError(
            "robot server identity lacks a canonical "
            "runtime_hardware_identity"
        ) from exc
    return {
        "host": str(host),
        "port": int(port),
        **{
            name: remote[name]
            for name in _SERVER_PROTOCOL_IDENTITY_FIELDS
        },
        "runtime_hardware_identity_sha256": (
            runtime_hardware_identity_sha256
        ),
    }


def _normalized_server_identity(
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize a live response or an already-collected signed claim."""
    missing = [
        name
        for name in _SERVER_PROTOCOL_IDENTITY_FIELDS
        if identity.get(name) is None
    ]
    if missing:
        raise SafetyError(f"robot server identity is missing fields: {missing}")
    claimed_hash = identity.get("runtime_hardware_identity_sha256")
    raw_identity = identity.get("runtime_hardware_identity")
    if raw_identity is not None:
        try:
            derived_hash = canonical_runtime_hardware_identity_sha256(
                raw_identity
            )
        except ValueError as exc:
            raise SafetyError(
                "robot server runtime_hardware_identity is invalid"
            ) from exc
        if claimed_hash is not None and claimed_hash != derived_hash:
            raise SafetyError(
                "robot server runtime hardware identity hash is inconsistent"
            )
        claimed_hash = derived_hash
    if (
        not isinstance(claimed_hash, str)
        or len(claimed_hash) != 64
        or any(
            character not in "0123456789abcdef"
            for character in claimed_hash
        )
    ):
        raise SafetyError(
            "robot server identity lacks a canonical runtime hardware "
            "identity SHA-256"
        )
    return {
        name: identity[name]
        for name in _SERVER_PROTOCOL_IDENTITY_FIELDS
    } | {"runtime_hardware_identity_sha256": claimed_hash}


def _reject_unsigned_bypasses(cfg: "LoopConfig") -> None:
    forbidden: list[str] = []
    if bool(cfg.allow_home_drift):
        forbidden.append("--allow-home-drift")
    if not bool(cfg.per_chunk_confirm):
        forbidden.append("--no-per-chunk-confirm")
    if bool(cfg.disable_held_verify):
        forbidden.append("--disable-held-verify")
    if bool(cfg.offline):
        forbidden.append("--offline")
    if bool(cfg.snap_subject_to_table):
        forbidden.append("snap_subject_to_table")
    if bool(cfg.collision_from_mesh):
        forbidden.append("collision_from_mesh")
    if bool(cfg.table_spec_height):
        forbidden.append("table_spec_height")
    if cfg.foundationpose_mesh is not None:
        forbidden.append("foundationpose_mesh override")
    if forbidden:
        raise SafetyError(
            "real execution forbids unsigned safety/calibration bypasses: "
            + ", ".join(forbidden)
        )


def collect_execution_authorization_claims(
    cfg: "LoopConfig",
    *,
    program_sha256: str,
    bundle_manifest: Path,
    token_expires_at: datetime | None = None,
    now: datetime | None = None,
    source_identity_fn: Callable[[], dict[str, Any]] = _inspect_oap_source,
    server_identity_fn: Callable[[str, int], dict[str, Any]] = _read_server_identity,
) -> dict[str, Any]:
    """Collect the exact facts an external authority must sign."""
    _reject_unsigned_bypasses(cfg)
    executor_latch = _executor_latch(cfg)
    operator_packet = _operator_estop_packet(
        cfg.operator_estop_packet_json,
        token_expires_at=token_expires_at,
        now=now,
    )
    if operator_packet["robot_hardware_id"] != executor_latch["hardware_id"]:
        raise SafetyError(
            "operator/E-stop packet robot_hardware_id differs from the "
            "verified executor latch hardware_id"
        )
    object_held_calibration = _object_held_calibration(cfg)
    robot_server = _normalized_server_identity(
        server_identity_fn(cfg.server_host, int(cfg.server_port))
    )
    if (
        object_held_calibration["hardware_id"]
        != robot_server["runtime_hardware_identity_sha256"]
    ):
        raise SafetyError(
            "ObjectHeld calibration hardware_id differs from the live "
            "controller runtime_hardware_identity"
        )
    return {
        "oap_source": source_identity_fn(),
        "program": {"canonical_sha256": str(program_sha256)},
        "bundle": _bundle_identity(bundle_manifest),
        "robot_server": {
            "host": str(cfg.server_host),
            "port": int(cfg.server_port),
            **robot_server,
        },
        "planner_deployment": _planner_deployment(cfg),
        "resolved_runtime_profile": _resolved_runtime_profile(cfg),
        "rig_calibrations": _rig_calibrations(cfg),
        "execution_policy": _execution_policy(cfg),
        "observation_limits": _observation_limits(cfg),
        "object_held_evidence_calibration": object_held_calibration,
        "joint_executor_verification_latch": executor_latch,
        "operator_estop_packet": operator_packet,
    }


def _claim_mismatch(expected: Any, actual: Any, path: str = "claims") -> str:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return (
                f"{path} keys differ (token={sorted(expected)}, "
                f"runtime={sorted(actual)})"
            )
        for key in sorted(expected):
            mismatch = _claim_mismatch(
                expected[key], actual[key], f"{path}.{key}"
            )
            if mismatch:
                return mismatch
        return ""
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return f"{path} lengths differ (token={len(expected)}, runtime={len(actual)})"
        for index, (left, right) in enumerate(zip(expected, actual)):
            mismatch = _claim_mismatch(left, right, f"{path}[{index}]")
            if mismatch:
                return mismatch
        return ""
    if expected != actual:
        return f"{path} differs (token={expected!r}, runtime={actual!r})"
    return ""


def _consume_token(
    token: _VerifiedToken,
    *,
    now: datetime,
    ledger_dir: Path,
    program_sha256: str,
) -> ExecutionAuthorizationReceipt:
    ledger = Path(ledger_dir).expanduser().resolve()
    try:
        ledger.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(ledger, 0o700)
    except OSError as exc:
        raise SafetyError(
            f"cannot initialize the execution-token replay ledger {ledger}: {exc}"
        ) from exc
    record = ledger / f"{token.token_id}.json"
    payload = {
        "schema": "oap_execution_authorization_consumption_v1",
        "token_id": token.token_id,
        "token_sha256": token.sha256,
        "authority_key_id": token.key_id,
        "program_sha256": program_sha256,
        "consumed_at": _iso_utc(now),
        "expires_at": _iso_utc(token.expires_at),
        "pid": os.getpid(),
    }
    data = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(
            record,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise SafetyError(
            f"execution authorization token {token.token_id} was already consumed"
        ) from exc
    except OSError as exc:
        raise SafetyError(
            f"cannot atomically consume execution authorization token: {exc}"
        ) from exc
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("zero-byte write to replay ledger")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    # POSIX can make the exclusive create durable by fsyncing the directory.
    # Windows has neither O_DIRECTORY nor a portable directory fsync; the
    # exclusive file creation above remains the replay lock there.
    if hasattr(os, "O_DIRECTORY"):
        try:
            directory_descriptor = os.open(
                ledger,
                os.O_RDONLY | os.O_DIRECTORY,
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError as exc:
            raise SafetyError(
                "execution token was consumed, but the replay-ledger "
                f"directory could not be synchronized: {exc}"
            ) from exc
    return ExecutionAuthorizationReceipt(
        token_id=token.token_id,
        token_sha256=token.sha256,
        authority_key_id=token.key_id,
        consumed_at=_iso_utc(now),
        expires_at=_iso_utc(token.expires_at),
        ledger_record=str(record),
    )


def require_and_consume_execution_authorization(
    cfg: "LoopConfig",
    *,
    program_sha256: str,
    bundle_manifest: Path,
    out_dir: Path,
    now: datetime | None = None,
    trust_store_path: Path = _TRUST_STORE_PATH,
    ledger_dir: Path = _LEDGER_DIR,
    source_identity_fn: Callable[[], dict[str, Any]] = _inspect_oap_source,
    server_identity_fn: Callable[[str, int], dict[str, Any]] = _read_server_identity,
) -> ExecutionAuthorizationReceipt | None:
    """Validate and consume one exact capability before any controller lease."""
    if not bool(cfg.execute):
        return None
    if cfg.readiness_authorization_token is None:
        raise SafetyError(
            "--execute requires a signed --readiness-authorization-token; "
            "a standalone preflight report cannot authorize motion"
        )
    current = (now or _utc_now()).astimezone(timezone.utc)
    token = _verify_token(
        cfg.readiness_authorization_token,
        now=current,
        trust_store_path=trust_store_path,
    )
    actual_claims = collect_execution_authorization_claims(
        cfg,
        program_sha256=program_sha256,
        bundle_manifest=bundle_manifest,
        token_expires_at=token.expires_at,
        now=current,
        source_identity_fn=source_identity_fn,
        server_identity_fn=server_identity_fn,
    )
    mismatch = _claim_mismatch(token.claims, actual_claims)
    if mismatch:
        raise SafetyError(
            "execution authorization does not match the exact runtime evidence: "
            + mismatch
        )
    # Recheck the clock immediately before the irreversible one-shot consume.
    current = (now or _utc_now()).astimezone(timezone.utc)
    if current < token.not_before or current >= token.expires_at:
        raise SafetyError("execution authorization expired before consumption")
    receipt = _consume_token(
        token,
        now=current,
        ledger_dir=ledger_dir,
        program_sha256=program_sha256,
    )
    write_json_atomic(
        Path(out_dir) / "execution_authorization_receipt.json",
        receipt.to_dict(),
    )
    cfg.execution_authorization_consumed = True
    cfg.execution_authorization_token_id = receipt.token_id
    cfg.execution_authorization_token_sha256 = receipt.token_sha256
    cfg.execution_authorization_expires_at = receipt.expires_at
    cfg.execution_authorization_server_identity = dict(
        actual_claims["robot_server"]
    )
    return receipt


def require_consumed_execution_authorization(
    cfg: "LoopConfig",
    *,
    now: datetime | None = None,
) -> None:
    """Connection-layer guard shared by lease acquisition and every trajectory."""
    if not bool(cfg.execute):
        return
    if (
        not bool(cfg.execution_authorization_consumed)
        or not cfg.execution_authorization_token_id
        or not cfg.execution_authorization_token_sha256
        or not cfg.execution_authorization_expires_at
        or not isinstance(cfg.execution_authorization_server_identity, dict)
    ):
        raise SafetyError(
            "real controller access refused: no signed one-shot execution "
            "authorization was consumed"
        )
    expires = _timestamp(
        {"expires_at": cfg.execution_authorization_expires_at},
        "expires_at",
    )
    current = (now or _utc_now()).astimezone(timezone.utc)
    if current >= expires:
        raise SafetyError(
            "real controller access refused: consumed execution authorization "
            "has expired"
        )


def require_authorized_server_identity(
    cfg: "LoopConfig",
    observed: Mapping[str, Any],
) -> None:
    """Bind the leased controller session to the pre-lease identity probe."""
    require_consumed_execution_authorization(cfg)
    expected = cfg.execution_authorization_server_identity
    assert isinstance(expected, dict)
    normalized = _normalized_server_identity(observed)
    mismatches = [
        f"{field}: authorized={expected.get(field)!r}, "
        f"observed={normalized.get(field)!r}"
        for field in _SERVER_IDENTITY_FIELDS
        if normalized.get(field) != expected.get(field)
    ]
    if mismatches:
        raise SafetyError(
            "leased controller identity differs from the signed authorization: "
            + "; ".join(mismatches)
        )


def require_authorized_calibration_hardware_identity(
    cfg: "LoopConfig",
) -> None:
    """Reject a calibration/controller hardware mismatch before connection."""
    require_consumed_execution_authorization(cfg)
    if not bool(cfg.execute):
        return
    expected = cfg.execution_authorization_server_identity
    assert isinstance(expected, dict)
    expected_hardware_id = expected.get(
        "runtime_hardware_identity_sha256"
    )
    try:
        calibration = load_object_held_evidence_calibration(
            cfg.hardware_evidence_calibration_json
        )
    except (TypeError, ValueError) as exc:
        raise SafetyError(
            "cannot validate the authorization-bound ObjectHeld calibration"
        ) from exc
    if calibration.hardware_id != expected_hardware_id:
        raise SafetyError(
            "ObjectHeld calibration hardware_id differs from the "
            "authorization-bound controller runtime_hardware_identity"
        )
