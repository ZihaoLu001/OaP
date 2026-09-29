"""Non-motion helpers for OaP's attended real-execution authorization.

This command never constructs a controller lease and never sends a robot or
gripper command.  It creates immutable evidence artifacts, collects the exact
runtime claims through the read-only server identity endpoint, and signs a
short-lived one-shot token with an independently held Ed25519 key.

Human and measured facts are never inferred.  Every attestation name must be
supplied explicitly, and token signing requires the separately reviewed
SHA-256 of the exact unsigned request.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from oap.loop import execution_authorization as authorization
from oap.loop.executor_readiness import (
    EXECUTOR_ATTESTATIONS,
    inspect_flexiv_control,
    require_executor_readiness,
)
from oap.loop import safety
from oap.loop.safety import SafetyError
from oap.program import TaskProgram, program_sha256
from oap.utils.io import sha256_file

REQUEST_SCHEMA = "oap_execution_authorization_request_v1"
EXECUTOR_LATCH_SCHEMA = "oap_joint_executor_verification_v1"
OPERATOR_ATTESTATIONS = (
    "operator_present",
    "estop_in_hand",
    "workspace_clear",
    "attended_run",
)
MAX_OPERATOR_PACKET_LIFETIME_S = 30 * 60
_KEY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_UNSIGNED_TOKEN_KEYS = authorization._TOKEN_KEYS - {"signature"}

__all__ = ["build_parser", "main"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_new_file(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.exists():
        raise SafetyError(f"refusing to overwrite immutable evidence: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_text_new(path: Path, text: str, *, mode: int = 0o644) -> Path:
    resolved = _require_new_file(path)
    descriptor = os.open(
        resolved,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        mode,
    )
    try:
        data = text.encode("utf-8")
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("zero-byte write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return resolved


def _write_json_new(path: Path, document: dict[str, Any]) -> Path:
    text = json.dumps(
        document,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    return _write_text_new(path, text)


def _exact_attestations(
    supplied: Sequence[str],
    expected: Sequence[str],
    *,
    label: str,
) -> dict[str, bool]:
    supplied_set = set(supplied)
    expected_set = set(expected)
    unknown = sorted(supplied_set - expected_set)
    missing = sorted(expected_set - supplied_set)
    duplicates = sorted(
        name for name in supplied_set if list(supplied).count(name) != 1
    )
    if unknown or missing or duplicates:
        raise SafetyError(
            f"{label} attestations must be supplied exactly once "
            f"(missing={missing}, unknown={unknown}, duplicates={duplicates})"
        )
    return {name: True for name in expected}


def _artifact_result(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
    }


def _create_executor_latch(args: argparse.Namespace) -> dict[str, Any]:
    report = inspect_flexiv_control()
    if not report["software_ready"]:
        raise SafetyError(
            "executor software is not ready: "
            + "; ".join(str(value) for value in report["errors"])
        )
    fingerprint = report["software_fingerprint_sha256"]
    if not isinstance(fingerprint, str):
        raise SafetyError("executor software has no fingerprint")
    document = {
        "schema": EXECUTOR_LATCH_SCHEMA,
        "hardware_id": str(args.hardware_id).strip(),
        "verified_by": str(args.verified_by).strip(),
        "verified_at": _iso_utc(_utc_now()),
        "software_fingerprint_sha256": fingerprint,
        "attestations": _exact_attestations(
            args.attest,
            EXECUTOR_ATTESTATIONS,
            label="executor",
        ),
    }
    if not document["hardware_id"] or not document["verified_by"]:
        raise SafetyError("hardware-id and verified-by must be non-empty")
    path = _write_json_new(args.out, document)
    result = _artifact_result(path)
    result["software_fingerprint_sha256"] = fingerprint
    return result


def _create_operator_packet(args: argparse.Namespace) -> dict[str, Any]:
    readiness = require_executor_readiness(
        args.joint_executor_verification_json,
        args.joint_executor_verification_sha256,
    )
    hardware_id = readiness["verification_latch"]["hardware_id"]
    if not isinstance(hardware_id, str) or not hardware_id.strip():
        raise SafetyError("verified executor latch has no hardware identity")
    lifetime_s = int(args.valid_for_s)
    if lifetime_s < 60 or lifetime_s > MAX_OPERATOR_PACKET_LIFETIME_S:
        raise SafetyError(
            "operator packet validity must be between 60 and "
            f"{MAX_OPERATOR_PACKET_LIFETIME_S} seconds"
        )
    now = _utc_now()
    operator_id = str(args.operator_id).strip()
    if not operator_id:
        raise SafetyError("operator-id must be non-empty")
    document = {
        "schema": authorization.OPERATOR_ESTOP_SCHEMA,
        "operator_id": operator_id,
        "robot_hardware_id": hardware_id,
        "attested_at": _iso_utc(now),
        "expires_at": _iso_utc(now + timedelta(seconds=lifetime_s)),
        "attestations": _exact_attestations(
            args.attest,
            OPERATOR_ATTESTATIONS,
            label="operator/E-stop",
        ),
    }
    path = _write_json_new(args.out, document)
    result = _artifact_result(path)
    result["expires_at"] = document["expires_at"]
    result["robot_hardware_id"] = hardware_id
    return result


def _bundle_checksum_manifest(args: argparse.Namespace) -> dict[str, Any]:
    bundle = Path(args.bundle).expanduser().resolve()
    root = Path(args.root).expanduser().resolve()
    identity = authorization._bundle_identity(bundle)
    files = [bundle]
    files.extend(
        bundle.parent / Path(relative)
        for relative in identity["artifacts_sha256"]
    )
    rows: list[tuple[str, str]] = []
    for path in files:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise SafetyError(
                f"bundle asset is outside checksum root {root}: {resolved}"
            ) from exc
        rows.append((relative.as_posix(), sha256_file(resolved)))
    text = "".join(
        f"{digest}  {relative}\n"
        for relative, digest in sorted(rows)
    )
    path = _write_text_new(args.out, text)
    result = _artifact_result(path)
    result["entries"] = len(rows)
    result["root"] = str(root)
    return result


def _run_config_from_remainder(run_args: Sequence[str]) -> Any:
    values = list(run_args)
    if values and values[0] == "--":
        values = values[1:]
    if not values:
        raise SafetyError(
            "collect-request requires '--' followed by the exact oap-run "
            "arguments"
        )
    from oap.cli.run import build_parser as build_run_parser
    from oap.cli.run import config_from_args

    try:
        parsed = build_run_parser().parse_args(values)
    except SystemExit as exc:
        raise SafetyError("invalid oap-run arguments") from exc
    cfg = config_from_args(parsed)
    if not cfg.execute or not cfg.i_confirm_real_motion:
        raise SafetyError(
            "the authorization request must describe the exact eventual "
            "--execute --i-confirm-real-motion invocation"
        )
    if cfg.readiness_authorization_token is not None:
        raise SafetyError(
            "do not include --readiness-authorization-token while collecting "
            "the unsigned request"
        )
    if cfg.program_json is None:
        raise SafetyError(
            "authorization collection requires one reviewed --program-json"
        )
    if cfg.prepare_real_execution:
        raise SafetyError(
            "--prepare-real-execution and --execute are mutually exclusive"
        )
    return cfg


def _build_request(
    cfg: Any,
    *,
    lifetime_s: int,
    now: datetime,
    collect_fn: Callable[..., dict[str, Any]] = (
        authorization.collect_execution_authorization_claims
    ),
) -> dict[str, Any]:
    if lifetime_s < 60 or lifetime_s > authorization.MAX_TOKEN_LIFETIME_S:
        raise SafetyError(
            "token lifetime must be between 60 and "
            f"{authorization.MAX_TOKEN_LIFETIME_S} seconds"
        )
    program_path = Path(cfg.program_json).expanduser().resolve()
    try:
        program = TaskProgram.from_json(
            program_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SafetyError(f"cannot load reviewed program: {exc}") from exc
    digest = program_sha256(program)
    safety.refuse_offline_program_execution(program, execute=True)
    if cfg.approved_program_sha256 != digest:
        raise SafetyError(
            "approved program digest does not match canonical program "
            f"(approved={cfg.approved_program_sha256!r}, actual={digest})"
        )
    safety.require_execution_allowed(
        execute=True,
        i_confirm_real_motion=bool(cfg.i_confirm_real_motion),
    )
    safety.require_observation_readiness_limits(
        required=True,
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
    expires = now + timedelta(seconds=lifetime_s)
    claims = collect_fn(
        cfg,
        program_sha256=digest,
        bundle_manifest=Path(cfg.bundle_manifest),
        token_expires_at=expires,
        now=now,
    )
    unsigned = {
        "schema": authorization.TOKEN_SCHEMA,
        "token_id": str(uuid.uuid4()),
        "audience": authorization.TOKEN_AUDIENCE,
        "issued_at": _iso_utc(now),
        "not_before": _iso_utc(now),
        "expires_at": _iso_utc(expires),
        "claims": claims,
    }
    return {
        "schema": REQUEST_SCHEMA,
        "unsigned_token": unsigned,
    }


def _collect_request(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _run_config_from_remainder(args.run_args)
    request = _build_request(
        cfg,
        lifetime_s=int(args.lifetime_s),
        now=_utc_now(),
    )
    path = _write_json_new(args.out, request)
    result = _artifact_result(path)
    result["token_id"] = request["unsigned_token"]["token_id"]
    result["expires_at"] = request["unsigned_token"]["expires_at"]
    return result


def _load_request(path: Path, *, now: datetime) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        request = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SafetyError(f"authorization request is unavailable or invalid: {exc}") from exc
    if (
        not isinstance(request, dict)
        or set(request) != {"schema", "unsigned_token"}
        or request.get("schema") != REQUEST_SCHEMA
    ):
        raise SafetyError(f"authorization request must use exact schema {REQUEST_SCHEMA!r}")
    token = request["unsigned_token"]
    if not isinstance(token, dict) or set(token) != _UNSIGNED_TOKEN_KEYS:
        raise SafetyError("unsigned token fields are not exact")
    if token.get("schema") != authorization.TOKEN_SCHEMA:
        raise SafetyError("unsigned token schema is invalid")
    if token.get("audience") != authorization.TOKEN_AUDIENCE:
        raise SafetyError("unsigned token audience is invalid")
    try:
        parsed_id = uuid.UUID(str(token.get("token_id", "")))
    except (ValueError, AttributeError) as exc:
        raise SafetyError("unsigned token_id must be a UUID") from exc
    if str(parsed_id) != token["token_id"]:
        raise SafetyError("unsigned token_id must be canonical lowercase UUID")
    issued = authorization._timestamp(token, "issued_at")
    not_before = authorization._timestamp(token, "not_before")
    expires = authorization._timestamp(token, "expires_at")
    if not_before < issued:
        raise SafetyError("unsigned token not_before precedes issued_at")
    lifetime_s = (expires - issued).total_seconds()
    if lifetime_s <= 0 or lifetime_s > authorization.MAX_TOKEN_LIFETIME_S:
        raise SafetyError("unsigned token lifetime is outside the permitted window")
    if now < not_before:
        raise SafetyError("unsigned token is not yet valid")
    if now >= expires:
        raise SafetyError("unsigned token request has expired")
    claims = token.get("claims")
    if not isinstance(claims, dict) or set(claims) != authorization._CLAIM_KEYS:
        raise SafetyError("unsigned token claims are not exact")
    operator = claims.get("operator_estop_packet")
    if not isinstance(operator, dict):
        raise SafetyError("unsigned request has no operator/E-stop claim")
    operator_attested = authorization._timestamp(operator, "attested_at")
    operator_expires = authorization._timestamp(operator, "expires_at")
    if operator_attested > now or operator_expires < expires:
        raise SafetyError(
            "operator/E-stop packet is not valid for the complete token lifetime"
        )
    latch = claims.get("joint_executor_verification_latch")
    if (
        not isinstance(latch, dict)
        or operator.get("robot_hardware_id") != latch.get("hardware_id")
    ):
        raise SafetyError(
            "operator/E-stop and executor-latch hardware identities differ"
        )
    return request


def _inspect_request(args: argparse.Namespace) -> dict[str, Any]:
    request = _load_request(args.request, now=_utc_now())
    token = request["unsigned_token"]
    claims = token["claims"]
    return {
        **_artifact_result(Path(args.request).expanduser().resolve()),
        "token_id": token["token_id"],
        "issued_at": token["issued_at"],
        "expires_at": token["expires_at"],
        "program_sha256": claims["program"]["canonical_sha256"],
        "oap_git_commit": claims["oap_source"]["git_commit"],
        "bundle_manifest_sha256": claims["bundle"]["manifest_sha256"],
        "robot_server": claims["robot_server"],
        "robot_hardware_id": claims["joint_executor_verification_latch"][
            "hardware_id"
        ],
        "operator_id": claims["operator_estop_packet"]["operator_id"],
        "observation_limits": claims["observation_limits"],
    }


def _load_private_key(path: Path) -> Any:
    resolved = Path(path).expanduser().resolve()
    if os.name != "nt" and resolved.stat().st_mode & 0o077:
        raise SafetyError(
            f"authority private key permissions must be 0600: {resolved}"
        )
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )

        key = serialization.load_pem_private_key(
            resolved.read_bytes(),
            password=None,
        )
    except (OSError, ValueError, TypeError) as exc:
        raise SafetyError(f"cannot load Ed25519 authority private key: {exc}") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise SafetyError("authority private key is not Ed25519")
    return key


def _sign_token(args: argparse.Namespace) -> dict[str, Any]:
    request_path = Path(args.request).expanduser().resolve()
    actual_request_sha256 = sha256_file(request_path)
    if args.approved_request_sha256 != actual_request_sha256:
        raise SafetyError(
            "approved request SHA-256 does not match the exact reviewed request "
            f"(approved={args.approved_request_sha256!r}, "
            f"actual={actual_request_sha256})"
        )
    request = _load_request(request_path, now=_utc_now())
    key_id = str(args.key_id)
    private_key = _load_private_key(args.private_key)
    try:
        from cryptography.hazmat.primitives import serialization

        public_raw = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    except (ValueError, TypeError) as exc:
        raise SafetyError(f"cannot derive authority public key: {exc}") from exc
    trusted = authorization._load_trusted_public_key(args.trust_store, key_id)
    if trusted != public_raw:
        raise SafetyError(
            "authority private key does not match the enabled trust-store key"
        )
    token = dict(request["unsigned_token"])
    signature = private_key.sign(authorization._canonical_signed_bytes(token))
    token["signature"] = {
        "algorithm": "ed25519",
        "key_id": key_id,
        "signature_b64": base64.b64encode(signature).decode("ascii"),
    }
    path = _write_json_new(args.out, token)
    result = _artifact_result(path)
    result["token_id"] = token["token_id"]
    result["expires_at"] = token["expires_at"]
    result["authority_key_id"] = key_id
    return result


def _generate_keypair(args: argparse.Namespace) -> dict[str, Any]:
    key_id = str(args.key_id)
    if not _KEY_ID_RE.fullmatch(key_id):
        raise SafetyError(
            "key-id must be 1-64 ASCII letters, digits, dot, underscore, or hyphen"
        )
    private_path = _require_new_file(args.private_key_out)
    trust_path = _require_new_file(args.trust_store_out)
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )

    private_key = Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    _write_text_new(private_path, private_pem.decode("ascii"), mode=0o600)
    public_raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    trust = {
        "schema": authorization.TRUST_SCHEMA,
        "keys": [
            {
                "key_id": key_id,
                "algorithm": "ed25519",
                "enabled": True,
                "public_key_b64": base64.b64encode(public_raw).decode("ascii"),
            }
        ],
    }
    _write_json_new(trust_path, trust)
    return {
        "private_key": str(private_path),
        "private_key_permissions": "0600",
        "trust_store_candidate": str(trust_path),
        "trust_store_candidate_sha256": sha256_file(trust_path),
        "key_id": key_id,
        "installation_performed": False,
    }


def _add_attest_argument(parser: argparse.ArgumentParser, choices: Sequence[str]) -> None:
    parser.add_argument(
        "--attest",
        action="append",
        required=True,
        choices=choices,
        metavar="FACT",
        help="fact physically demonstrated and reviewed; repeat once per required fact",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oap-real-authorization")
    subparsers = parser.add_subparsers(dest="command", required=True)

    latch = subparsers.add_parser(
        "create-executor-latch",
        help="write an exact latch after all six attended facts are true",
    )
    latch.add_argument("--out", type=Path, required=True)
    latch.add_argument("--hardware-id", required=True)
    latch.add_argument("--verified-by", required=True)
    _add_attest_argument(latch, EXECUTOR_ATTESTATIONS)
    latch.set_defaults(handler=_create_executor_latch)

    operator = subparsers.add_parser(
        "create-operator-packet",
        help="write a fresh attended operator/E-stop packet",
    )
    operator.add_argument("--out", type=Path, required=True)
    operator.add_argument("--operator-id", required=True)
    operator.add_argument(
        "--joint-executor-verification-json",
        type=Path,
        required=True,
    )
    operator.add_argument(
        "--joint-executor-verification-sha256",
        required=True,
    )
    operator.add_argument("--valid-for-s", type=int, required=True)
    _add_attest_argument(operator, OPERATOR_ATTESTATIONS)
    operator.set_defaults(handler=_create_operator_packet)

    checksums = subparsers.add_parser(
        "bundle-checksums",
        help="write sha256sum evidence for a bundle manifest and every referenced asset",
    )
    checksums.add_argument("--bundle", type=Path, required=True)
    checksums.add_argument("--root", type=Path, required=True)
    checksums.add_argument("--out", type=Path, required=True)
    checksums.set_defaults(handler=_bundle_checksum_manifest)

    collect = subparsers.add_parser(
        "collect-request",
        help="collect exact read-only runtime claims for external signing",
    )
    collect.add_argument("--out", type=Path, required=True)
    collect.add_argument("--lifetime-s", type=int, default=600)
    collect.add_argument(
        "run_args",
        nargs=argparse.REMAINDER,
        help="'--' followed by the exact eventual oap-run arguments",
    )
    collect.set_defaults(handler=_collect_request)

    inspect = subparsers.add_parser(
        "inspect-request",
        help="validate and summarize an unsigned request before approval",
    )
    inspect.add_argument("--request", type=Path, required=True)
    inspect.set_defaults(handler=_inspect_request)

    sign = subparsers.add_parser(
        "sign-token",
        help="sign one exact separately reviewed, unexpired request",
    )
    sign.add_argument("--request", type=Path, required=True)
    sign.add_argument("--approved-request-sha256", required=True)
    sign.add_argument("--private-key", type=Path, required=True)
    sign.add_argument("--trust-store", type=Path, required=True)
    sign.add_argument("--key-id", required=True)
    sign.add_argument("--out", type=Path, required=True)
    sign.set_defaults(handler=_sign_token)

    keygen = subparsers.add_parser(
        "generate-keypair",
        help="generate an authority key and trust-store candidate; installs nothing",
    )
    keygen.add_argument("--key-id", required=True)
    keygen.add_argument("--private-key-out", type=Path, required=True)
    keygen.add_argument("--trust-store-out", type=Path, required=True)
    keygen.set_defaults(handler=_generate_keypair)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = args.handler(args)
    except (SafetyError, OSError, ValueError) as exc:
        print(f"SAFETY REFUSAL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
