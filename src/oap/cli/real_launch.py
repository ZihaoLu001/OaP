"""Fail-closed entry point for one frozen, attended real-robot release.

``oap-real-launch`` deliberately accepts no motion-affecting run
overrides.  A reviewed release JSON contains the exact ``oap-run``
argument vector; a short-lived, independently signed execution token supplies
the capability.  The launcher verifies both locally before handing the exact
vector to :mod:`oap.cli.run`.

This module never creates operator attestations, signs tokens, weakens the
runtime authorization gate, or provides a CPU/local-planner fallback.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from oap.cli import run as run_cli
from oap.loop import execution_authorization as authorization
from oap.loop.execution_prefix import resolve_execution_prefix
from oap.loop.planner_profile import PlannerProfile
from oap.loop.safety import SafetyError
from oap.program import TaskProgram, program_sha256
from oap.utils.io import sha256_bytes, sha256_file, write_json_atomic

SCHEMA = "oap_real_launch_release_v1"
DEFAULT_RELEASE = Path("/etc/oap/real_launch_release.json")
_RELEASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_GIT_ID = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")

_RELEASE_FIELDS = {
    "schema",
    "release_id",
    "oap_source",
    "run_argv",
    "run_argv_sha256",
}
_SOURCE_FIELDS = {"git_commit", "git_tree", "tracked_source_clean"}

# Every one of these must occur exactly once in the sealed vector.  This keeps
# environment/default changes from silently changing a real episode.
_REQUIRED_ONCE = {
    "--bundle",
    "--task",
    "--out",
    "--program-json",
    "--approved-program-sha256",
    "--external-envs",
    "--server-host",
    "--server-port",
    "--foundationpose-mode",
    "--foundationpose-timeout-s",
    "--foundationpose-checkpoint-timeout-s",
    "--subject-pose-policy",
    "--relocalize-all-each-chunk",
    "--table-spec-json",
    "--real-table-z",
    "--sim-tcp-m",
    "--sim-tcp-anchor",
    "--pool-size",
    "--horizon-steps",
    "--num-knots",
    "--sigma-fraction",
    "--sampler-mode",
    "--local-sigma-fraction",
    "--cem-rounds",
    "--cem-elite-fraction",
    "--cem-min-std-fraction",
    "--execution-prefix-fraction",
    "--execution-prefix-steps",
    "--max-stage-cycles",
    "--seed",
    "--remote-planner-url",
    "--remote-planner-model-sha256",
    "--remote-planner-assets-sha256",
    "--remote-planner-sha256",
    "--remote-planner-service-instance-id",
    "--remote-planner-deadline-s",
    "--joint-executor-verification-json",
    "--joint-executor-verification-sha256",
    "--operator-estop-packet-json",
    "--hardware-evidence-calibration-json",
    "--max-observation-age-s",
    "--max-camera-robot-skew-s",
    "--max-plan-staleness-s",
    "--max-final-joint-tracking-error-rad",
    "--min-terminal-anchor-reliability",
    "--manual-post-experiment-restore",
    "--execute",
    "--i-confirm-real-motion",
}
_FORBIDDEN = {
    "--allow-home-drift",
    "--disable-held-verify",
    "--home-first",
    "--no-per-chunk-confirm",
    "--offline",
    "--prepare-real-execution",
    "--readiness-authorization-token",
}
_ABSOLUTE_PATH_FIELDS = (
    "bundle_manifest",
    "out_dir",
    "program_json",
    "table_spec_json",
    "joint_executor_verification_json",
    "operator_estop_packet_json",
    "hardware_evidence_calibration_json",
)

__all__ = [
    "DEFAULT_RELEASE",
    "SCHEMA",
    "build_parser",
    "canonical_argv_sha256",
    "load_release",
    "main",
    "seal_main",
    "validate_release",
]


def _default_token_path() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        return Path(runtime) / "oap" / "execution-token.json"
    return Path.home() / ".local/run/oap/execution-token.json"


def canonical_argv_sha256(argv: Sequence[str]) -> str:
    """Hash an argument vector without shell quoting or platform ambiguity."""
    payload = json.dumps(
        list(argv),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def _option_counts(argv: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in argv:
        if value.startswith("--"):
            if "=" in value:
                raise SafetyError(
                    "sealed real-run options must use separate value tokens; "
                    f"'--name=value' is forbidden: {value!r}"
                )
            counts[value] = counts.get(value, 0) + 1
    return counts


def load_release(path: Path) -> dict[str, Any]:
    """Load one exact release document and reject schema drift."""
    resolved = Path(path).expanduser().resolve()
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SafetyError(
            f"real launch release is unavailable or invalid at {resolved}: {exc}"
        ) from exc
    if not isinstance(document, dict) or set(document) != _RELEASE_FIELDS:
        actual = set(document) if isinstance(document, dict) else set()
        raise SafetyError(
            "real launch release fields are not exact "
            f"(missing={sorted(_RELEASE_FIELDS - actual)}, "
            f"unknown={sorted(actual - _RELEASE_FIELDS)})"
        )
    if document.get("schema") != SCHEMA:
        raise SafetyError(f"real launch release must use schema {SCHEMA!r}")
    release_id = document.get("release_id")
    if not isinstance(release_id, str) or not _RELEASE_ID.fullmatch(release_id):
        raise SafetyError("real launch release_id is invalid")
    source = document.get("oap_source")
    if not isinstance(source, dict) or set(source) != _SOURCE_FIELDS:
        raise SafetyError(
            "real launch oap_source must contain exactly "
            f"{sorted(_SOURCE_FIELDS)}"
        )
    if (
        not isinstance(source.get("git_commit"), str)
        or not _GIT_ID.fullmatch(source["git_commit"])
        or not isinstance(source.get("git_tree"), str)
        or not _GIT_ID.fullmatch(source["git_tree"])
        or source.get("tracked_source_clean") is not True
    ):
        raise SafetyError("real launch source identity is malformed or not clean")
    argv = document.get("run_argv")
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(value, str) and value for value in argv)
    ):
        raise SafetyError("real launch run_argv must be a non-empty string list")
    digest = document.get("run_argv_sha256")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise SafetyError("real launch run_argv_sha256 is invalid")
    actual_digest = canonical_argv_sha256(argv)
    if actual_digest != digest:
        raise SafetyError(
            "real launch run_argv does not match run_argv_sha256 "
            f"(declared={digest}, actual={actual_digest})"
        )
    return document


def _parse_frozen_config(argv: Sequence[str]) -> Any:
    counts = _option_counts(argv)
    duplicate = sorted(name for name, count in counts.items() if count != 1)
    if duplicate:
        raise SafetyError(
            "sealed real-run vector contains duplicate options: "
            + ", ".join(duplicate)
        )
    missing = sorted(name for name in _REQUIRED_ONCE if counts.get(name) != 1)
    if missing:
        raise SafetyError(
            "sealed real-run vector is missing exact production options: "
            + ", ".join(missing)
        )
    forbidden = sorted(name for name in _FORBIDDEN if counts.get(name))
    if forbidden:
        raise SafetyError(
            "sealed real-run vector contains forbidden options: "
            + ", ".join(forbidden)
        )
    try:
        namespace = run_cli.build_parser().parse_args(list(argv))
        cfg = run_cli.config_from_args(namespace)
    except (SystemExit, ValueError) as exc:
        raise SafetyError("sealed oap-run vector is invalid") from exc

    if (
        not cfg.execute
        or not cfg.i_confirm_real_motion
        or cfg.offline
        or cfg.prepare_real_execution
        or not cfg.per_chunk_confirm
        or cfg.allow_home_drift
        or cfg.disable_held_verify
        or cfg.home_first
    ):
        raise SafetyError(
            "sealed run does not preserve the real-motion safety gates and "
            "measured-start policy"
        )
    if (
        cfg.foundationpose_mode != "actual"
        or cfg.subject_pose_policy != "foundationpose"
        or not cfg.relocalize_all_each_chunk
    ):
        raise SafetyError(
            "sealed real run must use continuous multi-object FoundationPose"
        )
    tracking_timeouts = {
        "FoundationPose RPC": cfg.foundationpose_timeout_s,
        "FoundationPose checkpoint": (
            cfg.foundationpose_checkpoint_timeout_s
        ),
    }
    for label, raw in tracking_timeouts.items():
        if raw is None:
            raise SafetyError(f"sealed {label} timeout must be explicit")
        timeout_s = float(raw)
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise SafetyError(
                f"sealed {label} timeout must be finite and positive"
            )
    if not cfg.manual_post_experiment_restore:
        raise SafetyError(
            "sealed real run must preserve the verified initial posture for "
            "attended manual restoration"
        )
    if cfg.program_json is None:
        raise SafetyError("sealed real run requires one reviewed program JSON")
    if any(getattr(cfg, name) is None for name in _ABSOLUTE_PATH_FIELDS):
        raise SafetyError("sealed real run is missing an exact artifact path")
    non_absolute = [
        name
        for name in _ABSOLUTE_PATH_FIELDS
        if not Path(getattr(cfg, name)).expanduser().is_absolute()
    ]
    if non_absolute:
        raise SafetyError(
            "sealed real-run artifact paths must be absolute: "
            + ", ".join(non_absolute)
        )
    if Path(cfg.out_dir).exists():
        raise SafetyError(
            f"sealed episode output already exists: {Path(cfg.out_dir)}; "
            "refusing to mix or overwrite real-motion evidence"
        )
    remote_values = (
        cfg.remote_planner_url,
        cfg.remote_planner_model_sha256,
        cfg.remote_planner_assets_sha256,
        cfg.remote_planner_sha256,
        cfg.remote_planner_service_instance_id,
        cfg.remote_planner_deadline_s,
    )
    if not all(value is not None for value in remote_values):
        raise SafetyError(
            "sealed real run requires the identity-bound remote H100 planner"
        )
    try:
        profile = PlannerProfile(
            pool_size=int(cfg.pool_size),
            horizon_steps=int(cfg.horizon_steps),
            num_knots=int(cfg.num_knots),
            sigma_fraction=float(cfg.sigma_fraction),
            sampler_mode=str(cfg.sampler_mode),
            local_sigma_fraction=float(cfg.local_sigma_fraction),
            cem_rounds=int(cfg.cem_rounds),
            cem_elite_fraction=float(cfg.cem_elite_fraction),
            cem_min_std_fraction=float(cfg.cem_min_std_fraction),
            execution_prefix_steps=int(cfg.execution_prefix_steps),
            max_stage_cycles=int(cfg.max_stage_cycles),
        )
    except (TypeError, ValueError) as exc:
        raise SafetyError(f"sealed planner profile is invalid: {exc}") from exc
    required_profile = authorization._required_real_planner_profile()
    if profile != required_profile:
        raise SafetyError(
            "sealed planner profile does not equal the reviewed production "
            f"profile: expected={required_profile.to_dict()}, "
            f"actual={profile.to_dict()}"
        )
    prefix = resolve_execution_prefix(
        horizon_steps=profile.horizon_steps,
        execution_prefix_fraction=float(cfg.execution_prefix_fraction),
        execution_prefix_steps=profile.execution_prefix_steps,
    )
    declared_fraction = float(namespace.execution_prefix_fraction)
    exact_fraction = profile.execution_prefix_steps / profile.horizon_steps
    if not math.isclose(
        declared_fraction,
        exact_fraction,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        prefix.effective_fraction,
        exact_fraction,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise SafetyError(
            "sealed execution-prefix fraction must exactly describe the "
            "authorized integer prefix steps"
        )
    return cfg


def _verify_runtime_token(path: Path, *, now: datetime) -> Any:
    """Verify without consuming; the runner verifies and consumes it again."""
    return authorization._verify_token(
        Path(path),
        now=now,
        trust_store_path=authorization._TRUST_STORE_PATH,
    )


def _expected_local_claims(
    cfg: Any,
    *,
    token_expires_at: datetime,
    now: datetime,
) -> dict[str, Any]:
    program = TaskProgram.from_json(
        Path(cfg.program_json).read_text(encoding="utf-8")
    )
    canonical_program_sha = program_sha256(program)
    if canonical_program_sha != cfg.approved_program_sha256:
        raise SafetyError(
            "sealed approved-program SHA does not match canonical program "
            f"(declared={cfg.approved_program_sha256}, "
            f"actual={canonical_program_sha})"
        )
    return {
        "oap_source": authorization._inspect_oap_source(),
        "program": {"canonical_sha256": canonical_program_sha},
        "bundle": authorization._bundle_identity(cfg.bundle_manifest),
        "planner_deployment": authorization._planner_deployment(cfg),
        "resolved_runtime_profile": authorization._resolved_runtime_profile(cfg),
        "rig_calibrations": authorization._rig_calibrations(cfg),
        "execution_policy": authorization._execution_policy(cfg),
        "observation_limits": authorization._observation_limits(cfg),
        "object_held_evidence_calibration": (
            authorization._object_held_calibration(cfg)
        ),
        "joint_executor_verification_latch": authorization._executor_latch(cfg),
        "operator_estop_packet": authorization._operator_estop_packet(
            cfg.operator_estop_packet_json,
            token_expires_at=token_expires_at,
            now=now,
        ),
    }


def validate_release(
    release: Mapping[str, Any],
    *,
    authorization_token: Path,
    now: datetime | None = None,
) -> tuple[list[str], Any]:
    """Validate all non-network release/token claims before live startup."""
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    argv = list(release["run_argv"])
    cfg = _parse_frozen_config(argv)
    token = _verify_runtime_token(authorization_token, now=current)

    local_claims = _expected_local_claims(
        cfg,
        token_expires_at=token.expires_at,
        now=current,
    )
    if local_claims["oap_source"] != release["oap_source"]:
        raise SafetyError(
            "installed OaP source differs from the sealed launch release"
        )
    for name, actual in local_claims.items():
        expected = token.claims.get(name)
        mismatch = authorization._claim_mismatch(
            expected,
            actual,
            f"claims.{name}",
        )
        if mismatch:
            raise SafetyError(
                "signed execution token differs from the sealed local "
                f"release evidence: {mismatch}"
            )

    claimed_server = token.claims.get("robot_server")
    if not isinstance(claimed_server, dict):
        raise SafetyError("signed token has no robot-server claim")
    if (
        claimed_server.get("host") != cfg.server_host
        or claimed_server.get("port") != int(cfg.server_port)
    ):
        raise SafetyError(
            "signed token robot endpoint differs from the sealed run"
        )

    # The runner owns the irreversible atomic consume and the live server
    # identity re-collection.  Supplying the token only here prevents it from
    # being embedded in a long-lived release file.
    final_argv = [
        *argv,
        "--readiness-authorization-token",
        str(Path(authorization_token).expanduser().resolve()),
    ]
    return final_argv, SimpleNamespace(
        release_id=release["release_id"],
        token_id=token.token_id,
        token_expires_at=token.expires_at,
        operator_id=local_claims["operator_estop_packet"]["operator_id"],
        out_dir=Path(cfg.out_dir),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oap-real-launch",
        description=(
            "Execute one sealed, remote-H100, continuous-tracking release. "
            "Missing or mismatched evidence always refuses motion."
        ),
    )
    parser.add_argument(
        "--release",
        type=Path,
        default=DEFAULT_RELEASE,
        help=f"sealed launch release (default: {DEFAULT_RELEASE})",
    )
    parser.add_argument(
        "--authorization-token",
        type=Path,
        default=None,
        help=(
            "fresh independently signed one-shot token (default: "
            "$XDG_RUNTIME_DIR/oap/execution-token.json)"
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "verify only local release/token evidence; never calls oap-run "
            "and therefore never touches planner, camera, or robot"
        ),
    )
    return parser


def _build_seal_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oap-real-launch-seal",
        description=(
            "Create a connection-free candidate release from the exact "
            "reviewed oap-run argument vector. This does not authorize "
            "or sign real motion."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument(
        "--reviewed-profile-sha256",
        required=True,
        help=(
            "separately reviewed SHA-256 of the currently frozen complete "
            "production PlannerProfile"
        ),
    )
    parser.add_argument(
        "run_argv",
        nargs=argparse.REMAINDER,
        help="'--' followed by the exact eventual oap-run arguments",
    )
    return parser


def seal_main(argv: Sequence[str] | None = None) -> int:
    """Write a non-authorizing candidate release without touching hardware."""
    args = _build_seal_parser().parse_args(argv)
    release_id = str(args.release_id)
    if not _RELEASE_ID.fullmatch(release_id):
        print("SAFETY REFUSAL: invalid release-id", file=sys.stderr)
        return 2
    reviewed = str(args.reviewed_profile_sha256)
    try:
        frozen = authorization._required_real_planner_profile()
    except SafetyError as exc:
        print(f"SAFETY REFUSAL: {exc}", file=sys.stderr)
        return 2
    if not _SHA256.fullmatch(reviewed) or reviewed != frozen.sha256:
        print(
            "SAFETY REFUSAL: reviewed profile SHA does not match the source-"
            f"frozen production profile ({frozen.sha256})",
            file=sys.stderr,
        )
        return 2
    run_argv = list(args.run_argv)
    if run_argv and run_argv[0] == "--":
        run_argv = run_argv[1:]
    try:
        _parse_frozen_config(run_argv)
        source = authorization._inspect_oap_source()
        output = Path(args.out).expanduser().resolve()
        if output.exists():
            raise SafetyError(
                f"candidate release already exists: {output}; refusing overwrite"
            )
        document = {
            "schema": SCHEMA,
            "release_id": release_id,
            "oap_source": source,
            "run_argv": run_argv,
            "run_argv_sha256": canonical_argv_sha256(run_argv),
        }
        write_json_atomic(output, document)
    except (SafetyError, OSError, ValueError) as exc:
        print(f"SAFETY REFUSAL: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "schema": "oap_real_launch_release_candidate_v1",
                "motion_authorized": False,
                "release": str(output),
                "release_sha256": sha256_file(output),
                "run_argv_sha256": document["run_argv_sha256"],
                "planner_profile_sha256": frozen.sha256,
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    token_path = (
        _default_token_path()
        if args.authorization_token is None
        else Path(args.authorization_token)
    )
    try:
        release = load_release(args.release)
        run_argv, identity = validate_release(
            release,
            authorization_token=token_path,
        )
    except (SafetyError, OSError, ValueError) as exc:
        print(f"SAFETY REFUSAL: {exc}", file=sys.stderr)
        return 2
    if args.check_only:
        print(
            json.dumps(
                {
                    "schema": "oap_real_launch_check_v1",
                    "ready_for_live_startup": True,
                    "motion_authorized": False,
                    "release_id": identity.release_id,
                    "token_id": identity.token_id,
                    "token_expires_at": (
                        identity.token_expires_at.astimezone(
                            timezone.utc
                        ).isoformat().replace("+00:00", "Z")
                    ),
                    "operator_id": identity.operator_id,
                    "out_dir": str(identity.out_dir),
                },
                sort_keys=True,
            )
        )
        return 0
    return int(run_cli.main(run_argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
