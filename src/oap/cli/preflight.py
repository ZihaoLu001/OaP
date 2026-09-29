"""Read-only preflight for an attended OaP real-robot episode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from oap.loop.preflight import PreflightConfig, run_preflight

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Build the preflight CLI without importing or connecting a robot client."""
    parser = argparse.ArgumentParser(
        prog="oap-preflight",
        description=(
            "Validate software evidence and observe hardware availability. "
            "This command never acquires robot control and sends no motion."
        ),
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--program-json", type=Path, required=True)
    parser.add_argument(
        "--approved-program-sha256",
        help="full canonical digest reviewed by the operator (required for READY)",
    )
    parser.add_argument(
        "--hardware-evidence-calibration-json",
        type=Path,
        help=(
            "measured ObjectHeld evidence calibration; required when the program "
            "contains ObjectHeld"
        ),
    )
    parser.add_argument(
        "--joint-executor-verification-json",
        type=Path,
        help=(
            "attended joint-executor verification latch; absence remains "
            "hardware-pending and can never authorize motion"
        ),
    )
    parser.add_argument(
        "--joint-executor-verification-sha256",
        help="full lowercase SHA-256 approving the exact verification latch",
    )
    parser.add_argument(
        "--asset-checksums",
        type=Path,
        help="sha256sum-format integrity manifest (absence is CANNOT_CHECK)",
    )
    parser.add_argument(
        "--checksum-root",
        type=Path,
        default=Path.cwd(),
        help="root for relative paths in --asset-checksums (default: current directory)",
    )
    parser.add_argument(
        "--external-envs",
        default="lab",
        help="external-env profile or YAML (default: lab)",
    )
    parser.add_argument(
        "--required-external-tool",
        action="append",
        dest="required_external_tools",
        help=(
            "external tool whose interpreter/paths must exist; repeatable "
            "(default: observer, foundationpose)"
        ),
    )
    parser.add_argument("--camera-device", type=Path, default=Path("/dev/video0"))
    parser.add_argument("--robot-host", default="ROBOT-HOST-PLACEHOLDER")
    parser.add_argument("--robot-port", type=int, default=8766)
    parser.add_argument("--robot-timeout-s", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print one JSON report; zero only permits attended physical preflight."""
    args = build_parser().parse_args(argv)
    required_tools = tuple(
        args.required_external_tools or ("observer", "foundationpose")
    )
    report = run_preflight(PreflightConfig(
        bundle_manifest=args.bundle,
        program_json=args.program_json,
        approved_program_sha256=args.approved_program_sha256,
        hardware_evidence_calibration_json=args.hardware_evidence_calibration_json,
        joint_executor_verification_json=(
            args.joint_executor_verification_json
        ),
        joint_executor_verification_sha256=(
            args.joint_executor_verification_sha256
        ),
        asset_checksums=args.asset_checksums,
        checksum_root=args.checksum_root,
        external_envs=args.external_envs,
        required_external_tools=required_tools,
        camera_device=args.camera_device,
        robot_host=args.robot_host,
        robot_port=args.robot_port,
        robot_timeout_s=args.robot_timeout_s,
    ))
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return (
        0
        if report.attended_hardware_checks_passed and report.software_ready
        else 2
    )


if __name__ == "__main__":  # pragma: no cover - console-script path
    raise SystemExit(main())
