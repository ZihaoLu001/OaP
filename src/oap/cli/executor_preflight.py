"""Connection-free Flexiv executor preflight.

This command inspects Python packaging/API identity and an optional human
verification latch.  It never constructs a robot client or touches a camera.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from oap.loop.executor_readiness import evaluate_executor_readiness
from oap.utils.io import write_json_atomic

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oap-executor-preflight",
        description=__doc__,
    )
    parser.add_argument(
        "--joint-executor-verification-json",
        type=Path,
        help="attended human verification latch to validate",
    )
    parser.add_argument(
        "--joint-executor-verification-sha256",
        help="full lowercase SHA-256 approving the exact latch",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="optional JSON report path (the same report is printed)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate_executor_readiness(
        args.joint_executor_verification_json,
        args.joint_executor_verification_sha256,
    )
    if args.out is not None:
        write_json_atomic(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["hardware_verified"] else 2


if __name__ == "__main__":  # pragma: no cover - console-script path
    raise SystemExit(main())
