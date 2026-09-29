"""Create a canonical H100 warm-up manifest from real planning requests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from oap.remote_planner.protocol import (
    PlanningProtocolError,
    PlanningRequest,
)
from oap.remote_planner.warmup import (
    planner_warmup_manifest_from_request_templates,
)
from oap.utils.io import write_json_atomic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oap-planner-warmup-manifest"
    )
    parser.add_argument(
        "--request",
        action="append",
        type=Path,
        required=True,
        help=(
            "ordinary PlanningRequest JSON captured from the production "
            "lab-side request builder; repeat exactly once per Stage"
        ),
    )
    parser.add_argument("--service-instance-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _read_request(path: Path) -> PlanningRequest:
    source = path.expanduser().resolve()
    try:
        value: Any = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanningProtocolError(
            f"cannot read PlanningRequest template {source}"
        ) from exc
    if not isinstance(value, Mapping):
        raise PlanningProtocolError(
            f"PlanningRequest template {source} must be a JSON object"
        )
    return PlanningRequest.from_dict(value)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(
            "warm-up manifest output already exists; refusing stale "
            f"overwrite: {output}"
        )
    manifest = planner_warmup_manifest_from_request_templates(
        [_read_request(path) for path in args.request],
        service_instance_id=str(args.service_instance_id),
    )
    write_json_atomic(output, manifest.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
