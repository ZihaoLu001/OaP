"""Canonical, stage-complete bootstrap requests for the H100 planner.

The persistent service cannot honestly advertise readiness after merely
loading a MuJoCo model.  JAX/MJWarp specializes both the rollout and the
device-side constraint cost on the real request shapes.  A warm-up manifest
therefore contains one ordinary :class:`PlanningRequest` for every Stage of
the frozen TaskProgram.  The service executes those requests directly
(never through the HTTP/replay dispatch seam) before binding its port.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from oap.program import TaskProgram

from .protocol import (
    PlanningProtocolError,
    PlanningRequest,
    canonical_sha256,
)

WARMUP_MANIFEST_SCHEMA = "oap_remote_planner_warmup_manifest_v1"


@dataclass(frozen=True)
class PlannerWarmupManifest:
    """Strict canonical collection of one bootstrap request per Stage."""

    requests: tuple[PlanningRequest, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": WARMUP_MANIFEST_SCHEMA,
            "requests": [request.to_dict() for request in self.requests],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def program_sha256(self) -> str:
        return self.requests[0].program_sha256

    @property
    def execution_feasibility_sha256(self) -> str:
        feasibility = self.requests[0].config.execution_feasibility
        if feasibility is None:  # pragma: no cover - constructor is strict
            raise PlanningProtocolError(
                "planner warm-up requires execution feasibility"
            )
        return canonical_sha256(feasibility.to_dict())


def _validate_execution_feasibility(
    requests: Sequence[PlanningRequest],
) -> None:
    """Require one exact, controller-compatible actuator contract."""
    first = requests[0].config.execution_feasibility
    if first is None:
        raise PlanningProtocolError(
            "planner warm-up request for Stage "
            f"{requests[0].stage_index} is missing execution_feasibility; "
            "capture an ordinary production request"
        )
    if not first.controller_timing_compatible:
        raise PlanningProtocolError(
            "planner warm-up execution feasibility is not compatible with "
            "the production controller timing"
        )
    expected = first.to_dict()
    for request in requests[1:]:
        feasibility = request.config.execution_feasibility
        if feasibility is None:
            raise PlanningProtocolError(
                "planner warm-up request for Stage "
                f"{request.stage_index} is missing execution_feasibility; "
                "capture an ordinary production request"
            )
        if not feasibility.controller_timing_compatible:
            raise PlanningProtocolError(
                "planner warm-up execution feasibility is not compatible "
                "with the production controller timing"
            )
        if feasibility.to_dict() != expected:
            raise PlanningProtocolError(
                "planner warm-up requests disagree on execution feasibility; "
                "controller prefix arrays and actuator-limit identities must "
                "be identical for every Stage"
            )


def planner_warmup_manifest(
    requests: Sequence[PlanningRequest],
) -> PlannerWarmupManifest:
    """Validate exact program/stage coverage and return a canonical manifest."""
    parsed = tuple(
        PlanningRequest.from_dict(request.to_dict())
        for request in requests
    )
    if not parsed:
        raise PlanningProtocolError(
            "planner warm-up manifest must contain at least one request"
        )
    first = parsed[0]
    program = TaskProgram.from_json(first.program_json)
    if not program.stages:
        raise PlanningProtocolError(
            "planner warm-up program must contain at least one Stage"
        )
    expected_stages = set(range(len(program.stages)))
    actual_stages = [request.stage_index for request in parsed]
    if (
        len(actual_stages) != len(set(actual_stages))
        or set(actual_stages) != expected_stages
    ):
        raise PlanningProtocolError(
            "planner warm-up manifest must contain exactly one request for "
            f"each Stage {sorted(expected_stages)}, got {sorted(actual_stages)}"
        )
    invariant_fields = (
        "program_json",
        "program_sha256",
        "model_sha256",
        "assets_sha256",
        "planner_sha256",
        "service_instance_id",
        "planner_profile",
        "planner_profile_sha256",
    )
    for request in parsed[1:]:
        for field in invariant_fields:
            if getattr(request, field) != getattr(first, field):
                raise PlanningProtocolError(
                    "planner warm-up requests disagree on immutable field "
                    f"{field!r}"
                )
    _validate_execution_feasibility(parsed)
    ordered = tuple(sorted(parsed, key=lambda request: request.stage_index))
    return PlannerWarmupManifest(requests=ordered)


def planner_warmup_manifest_from_request_templates(
    requests: Sequence[PlanningRequest],
    *,
    service_instance_id: str,
) -> PlannerWarmupManifest:
    """Rebind real request templates into non-live bootstrap requests.

    State, grounding, controller feasibility, program, model, assets, source,
    and planner profile are copied byte-for-byte from the ordinary requests.
    Only replay-sensitive wire identity is replaced.  Epoch timestamps make
    the resulting requests unusable as live commands while remaining valid
    inputs to the direct, pre-bind warm-up path.
    """
    parsed = tuple(
        PlanningRequest.from_dict(request.to_dict())
        for request in requests
    )
    if not parsed:
        raise PlanningProtocolError(
            "at least one ordinary PlanningRequest template is required"
        )
    source_instance_ids = {
        request.service_instance_id for request in parsed
    }
    if len(source_instance_ids) != 1:
        raise PlanningProtocolError(
            "PlanningRequest templates came from different service instances"
        )
    instance_digest = hashlib.sha256(
        str(service_instance_id).encode("utf-8")
    ).hexdigest()[:24]
    rebound: list[PlanningRequest] = []
    for request in parsed:
        value = request.to_dict()
        value.update(
            {
                "request_id": (
                    f"warmup.{instance_digest}.stage.{request.stage_index}"
                ),
                "episode_id": f"warmup.{instance_digest}",
                "issued_at_unix_s": 0.0,
                "deadline_unix_s": 1.0,
                "service_instance_id": str(service_instance_id),
            }
        )
        rebound.append(PlanningRequest.from_dict(value))
    return planner_warmup_manifest(rebound)


def planner_warmup_manifest_from_dict(
    value: Mapping[str, Any],
) -> PlannerWarmupManifest:
    """Parse a strict JSON object and validate all embedded wire requests."""
    if not isinstance(value, Mapping):
        raise PlanningProtocolError(
            "planner warm-up manifest must be a JSON object"
        )
    expected = {"schema", "requests"}
    if set(value) != expected:
        raise PlanningProtocolError(
            "planner warm-up manifest fields mismatch: "
            f"missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )
    if value["schema"] != WARMUP_MANIFEST_SCHEMA:
        raise PlanningProtocolError(
            "unsupported planner warm-up manifest schema "
            f"{value['schema']!r}"
        )
    raw_requests = value["requests"]
    if (
        isinstance(raw_requests, (str, bytes))
        or not isinstance(raw_requests, Sequence)
    ):
        raise PlanningProtocolError(
            "planner warm-up manifest requests must be a JSON array"
        )
    requests: list[PlanningRequest] = []
    for index, request in enumerate(raw_requests):
        if not isinstance(request, Mapping):
            raise PlanningProtocolError(
                f"planner warm-up request {index} must be a JSON object"
            )
        requests.append(PlanningRequest.from_dict(request))
    manifest = planner_warmup_manifest(requests)
    # Refuse non-canonical ordering.  This makes the file hash a stable release
    # input rather than accepting two byte-distinct permutations as equivalent.
    if [request.stage_index for request in requests] != [
        request.stage_index for request in manifest.requests
    ]:
        raise PlanningProtocolError(
            "planner warm-up requests must be ordered by stage_index"
        )
    return manifest


def read_planner_warmup_manifest(
    path: Path | str,
) -> PlannerWarmupManifest:
    """Read and strictly validate one UTF-8 warm-up manifest."""
    manifest_path = Path(path).expanduser().resolve()
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanningProtocolError(
            f"cannot read planner warm-up manifest {manifest_path}"
        ) from exc
    return planner_warmup_manifest_from_dict(value)
