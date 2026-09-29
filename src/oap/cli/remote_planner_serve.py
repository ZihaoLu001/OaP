"""Run one loopback-only, content-addressed H100 planner service."""
from __future__ import annotations

import argparse
import logging
import os
import re
import socket
from pathlib import Path

from oap.remote_planner.deployment import load_release_context
from oap.remote_planner.http import make_http_server
from oap.remote_planner.protocol import PROTOCOL_VERSION
from oap.remote_planner.service import (
    PlannerService,
    PlanningContextRegistry,
)
from oap.remote_planner.warmup import (
    read_planner_warmup_manifest,
)
from oap.twin import mjx_screen
from oap.utils.io import write_json_atomic

logger = logging.getLogger("oap.cli.remote_planner_serve")
_INSTANCE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oap-planner-serve")
    parser.add_argument("--release-bundle", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--service-instance-id", required=True)
    parser.add_argument("--ready-json", type=Path, required=True)
    parser.add_argument(
        "--warmup-requests-json",
        type=Path,
        required=True,
        help=(
            "canonical manifest with one exact PlanningRequest per frozen "
            "TaskProgram Stage"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("OAP_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = build_parser().parse_args(argv)
    if not 1 <= int(args.port) <= 65535 or int(args.port) == 8766:
        raise SystemExit("planner port must be 1..65535 and must not be 8766")
    if not _INSTANCE_RE.fullmatch(str(args.service_instance_id)):
        raise SystemExit("invalid --service-instance-id")
    ready_path = Path(args.ready_json).expanduser().resolve()
    if ready_path.exists():
        raise SystemExit(
            f"ready JSON already exists; refusing stale overwrite: {ready_path}"
        )
    if not mjx_screen.is_available():
        raise SystemExit("GPU_ONLY_REFUSAL: MJWarp backend is unavailable")

    release, context = load_release_context(args.release_bundle)
    warmup_manifest = read_planner_warmup_manifest(
        args.warmup_requests_json
    )
    registry = PlanningContextRegistry()
    registry.register(context)
    service = PlannerService(
        contexts=registry,
        planner_sha256=release.planner_sha256,
        service_instance_id=str(args.service_instance_id),
    )
    warmup_evidence = service.warm_up(
        warmup_manifest.requests,
        manifest_sha256=warmup_manifest.sha256,
    )
    # Only a fully hot service may bind a port or create a readiness artifact.
    server = make_http_server(service, host="127.0.0.1", port=int(args.port))
    write_json_atomic(
        ready_path,
        {
            "schema": "oap_remote_planner_ready_v3",
            "protocol_version": PROTOCOL_VERSION,
            "service_instance_id": str(args.service_instance_id),
            "planner_sha256": release.planner_sha256,
            "model_sha256": release.model_sha256,
            "assets_sha256": release.assets_sha256,
            "planner_profile": release.planner_profile.to_dict(),
            "planner_profile_sha256": (
                release.planner_profile_sha256
            ),
            "release_bundle": str(release.path),
            "compute_host": socket.gethostname(),
            "port": int(args.port),
            "pid": os.getpid(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "warmup_evidence": warmup_evidence,
        },
    )
    logger.info(
        "remote planner ready instance=%s model=%s assets=%s profile=%s "
        "max_hot=%.3fs port=%d",
        args.service_instance_id,
        release.model_sha256[:12],
        release.assets_sha256[:12],
        release.planner_profile_sha256[:12],
        max(
            float(stage["hot_wall_elapsed_s"])
            for stage in warmup_evidence["stages"]
        ),
        int(args.port),
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
