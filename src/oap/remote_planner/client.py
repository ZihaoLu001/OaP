"""Lab-side client and ``solve_mpc_step`` adapter for remote planning."""
from __future__ import annotations

import json
import ipaddress
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Protocol
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from oap.loop.execution_prefix import (
    resolve_execution_prefix,
    validate_execution_prefix_fraction,
)
from oap.loop.planner_profile import PlannerProfile
from oap.loop.mpc_loop import MpcStepResult
from oap.loop.sampling import SamplerStats
from oap.program import CallSiteRecord, TaskProgram, program_sha256
from oap.twin.control_knots import ControlKnots

from .grounding import build_rigid_body_grounding
from .protocol import (
    PROTOCOL_VERSION,
    ExecutionFeasibility,
    PlannerConfig,
    PlanningProtocolError,
    PlanningRequest,
    PlanningResponse,
    PlanningState,
    ReplayGuard,
    anchors_to_payload,
    canonical_json,
    validate_response_for_request,
)


class PlannerTransport(Protocol):
    """One request/response transport with an absolute application deadline."""

    def send(
        self,
        payload: dict[str, Any],
        *,
        deadline_unix_s: float,
    ) -> dict[str, Any]:
        ...

    def info(self) -> dict[str, Any]:
        ...


@dataclass
class InProcessPlannerTransport:
    """Connection-free transport used by protocol and service tests."""

    service: Any

    def send(
        self,
        payload: dict[str, Any],
        *,
        deadline_unix_s: float,
    ) -> dict[str, Any]:
        del deadline_unix_s
        result = self.service.handle(payload)
        if not isinstance(result, dict):
            raise PlanningProtocolError(
                "planner service returned a non-object response"
            )
        return result

    def info(self) -> dict[str, Any]:
        result = self.service.info()
        if not isinstance(result, dict):
            raise PlanningProtocolError(
                "planner service returned non-object identity"
            )
        return result


class HttpPlannerTransport:
    """Minimal JSON POST transport intended for an SSH loopback forward."""

    def __init__(
        self,
        endpoint: str,
        *,
        clock: Callable[[], float] = time.time,
        max_response_bytes: int = 8 << 20,
        info_timeout_s: float = 5.0,
    ) -> None:
        parsed = urlsplit(str(endpoint))
        try:
            host = ipaddress.ip_address(parsed.hostname or "")
        except ValueError as exc:
            raise ValueError(
                "remote planner URL must use a literal loopback IP"
            ) from exc
        if (
            parsed.scheme != "http"
            or not host.is_loopback
            or parsed.path != "/v1/plan"
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.port is None
        ):
            raise ValueError(
                "remote planner URL must be "
                "http://<loopback-ip>:<port>/v1/plan"
            )
        self._endpoint = str(endpoint)
        self._info_endpoint = urlunsplit(
            (parsed.scheme, parsed.netloc, "/v1/info", "", "")
        )
        self._clock = clock
        self._max_response_bytes = int(max_response_bytes)
        self._info_timeout_s = float(info_timeout_s)

    def send(
        self,
        payload: dict[str, Any],
        *,
        deadline_unix_s: float,
    ) -> dict[str, Any]:
        remaining = float(deadline_unix_s) - float(self._clock())
        if remaining <= 0.0:
            raise PlanningProtocolError(
                "remote planning deadline expired before transport"
            )
        body = canonical_json(payload).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=remaining) as reply:
                raw = reply.read(self._max_response_bytes + 1)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise PlanningProtocolError(
                "remote planner transport failed"
            ) from exc
        if len(raw) > self._max_response_bytes:
            raise PlanningProtocolError(
                "remote planner response exceeds size limit"
            )
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PlanningProtocolError(
                "remote planner response is not valid JSON"
            ) from exc
        if not isinstance(result, dict):
            raise PlanningProtocolError(
                "remote planner response must be a JSON object"
            )
        return result

    def info(self) -> dict[str, Any]:
        request = urllib.request.Request(
            self._info_endpoint,
            method="GET",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self._info_timeout_s
            ) as reply:
                raw = reply.read(self._max_response_bytes + 1)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise PlanningProtocolError(
                "remote planner identity transport failed"
            ) from exc
        if len(raw) > self._max_response_bytes:
            raise PlanningProtocolError(
                "remote planner identity exceeds size limit"
            )
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PlanningProtocolError(
                "remote planner identity is not valid JSON"
            ) from exc
        if not isinstance(result, dict):
            raise PlanningProtocolError(
                "remote planner identity must be a JSON object"
            )
        return result


class RemotePlannerClient:
    """Consume each request and each matching response exactly once."""

    def __init__(
        self,
        transport: PlannerTransport,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._transport = transport
        self._clock = clock
        self._requests = ReplayGuard()
        self._responses = ReplayGuard()

    def plan(self, request: PlanningRequest) -> PlanningResponse:
        # Round-trip parsing is the single strict validator for locally-built
        # and decoded requests alike.
        request = PlanningRequest.from_dict(request.to_dict())
        self._requests.mark(request.request_id, request.sha256)
        payload = self._transport.send(
            request.to_dict(),
            deadline_unix_s=request.deadline_unix_s,
        )
        response = PlanningResponse.from_dict(payload)
        validate_response_for_request(
            response,
            request,
            now_unix_s=float(self._clock()),
        )
        self._responses.mark(response.request_id, response.request_sha256)
        return response

    def require_identity(
        self,
        *,
        service_instance_id: str,
        planner_sha256: str,
        model_sha256: str,
        assets_sha256: str,
        planner_profile: PlannerProfile,
    ) -> dict[str, Any]:
        """Verify the tunnel endpoint before any planning request."""
        info = self._transport.info()
        if set(info) != {
            "protocol_version",
            "service_instance_id",
            "planner_sha256",
            "contexts",
        }:
            raise PlanningProtocolError(
                "remote planner identity has missing or unknown fields"
            )
        if info.get("protocol_version") != PROTOCOL_VERSION:
            raise PlanningProtocolError(
                "remote planner protocol identity mismatch"
            )
        if info.get("service_instance_id") != service_instance_id:
            raise PlanningProtocolError(
                "remote planner service instance mismatch"
            )
        if info.get("planner_sha256") != planner_sha256:
            raise PlanningProtocolError(
                "remote planner source identity mismatch"
            )
        contexts = info.get("contexts")
        if not isinstance(contexts, list) or {
            "model_sha256": model_sha256,
            "assets_sha256": assets_sha256,
            "planner_profile": planner_profile.to_dict(),
            "planner_profile_sha256": planner_profile.sha256,
        } not in contexts:
            raise PlanningProtocolError(
                "remote planner model/assets/profile context is not loaded"
            )
        return info


class RemoteSolveStep:
    """Callable injection for ``run_receding_horizon(solve_step_fn=...)``.

    It serializes only the optimizer inputs, never callbacks or hardware
    handles.  A transport/protocol failure becomes ``plan_valid=False`` with
    the incoming nominal retained for diagnostics, so the outer lab loop moves
    zero joints.
    """

    def __init__(
        self,
        *,
        client: RemotePlannerClient,
        episode_id: str,
        model_sha256: str,
        assets_sha256: str,
        planner_sha256: str,
        service_instance_id: str,
        max_latency_s: float,
        planner_profile: PlannerProfile,
        clock: Callable[[], float] = time.time,
        request_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if max_latency_s <= 0.0:
            raise ValueError("max_latency_s must be > 0")
        self._client = client
        self._episode_id = str(episode_id)
        self._model_sha256 = str(model_sha256)
        self._assets_sha256 = str(assets_sha256)
        self._planner_sha256 = str(planner_sha256)
        self._service_instance_id = str(service_instance_id)
        self._max_latency_s = float(max_latency_s)
        if not isinstance(planner_profile, PlannerProfile):
            raise TypeError("planner_profile must be a PlannerProfile")
        self._planner_profile = planner_profile
        self._clock = clock
        self._request_id_factory = request_id_factory or (
            lambda: uuid.uuid4().hex
        )

    def __call__(self, **kwargs: Any) -> MpcStepResult:
        candidates = list(kwargs["candidates"])
        metadata = dict(kwargs["metadata"])
        if len(candidates) != 1:
            raise PlanningProtocolError(
                "remote production planning requires one nominal candidate"
            )
        nominal = candidates[0]
        profile = self._planner_profile
        nominal_knots = np.asarray(nominal.knots, dtype=float)
        if nominal_knots.shape != (profile.num_knots, 8):
            raise PlanningProtocolError(
                "remote nominal shape does not match planner profile"
            )
        runtime_pool = int(kwargs.get("pool_size", 2048))
        runtime_horizon = int(kwargs["horizon_steps"])
        runtime_sigma = float(kwargs["sigma_fraction"])
        runtime_sampler_mode = str(kwargs["sampler_mode"])
        runtime_local_sigma = float(kwargs["local_sigma_fraction"])
        runtime_cem_rounds = int(
            kwargs.get("cem_rounds", profile.cem_rounds)
        )
        runtime_cem_elite_fraction = float(
            kwargs.get("cem_elite_fraction", profile.cem_elite_fraction)
        )
        runtime_cem_min_std_fraction = float(
            kwargs.get(
                "cem_min_std_fraction",
                profile.cem_min_std_fraction,
            )
        )
        runtime_cost_shaping_profile = kwargs.get("cost_shaping_profile")
        runtime_prefix_fraction = validate_execution_prefix_fraction(
            kwargs["execution_prefix_frac"]
        )
        resolved_prefix = resolve_execution_prefix(
            horizon_steps=runtime_horizon,
            execution_prefix_fraction=runtime_prefix_fraction,
            execution_prefix_steps=profile.execution_prefix_steps,
        )
        if (
            runtime_pool != profile.pool_size
            or runtime_horizon != profile.horizon_steps
            or abs(
                runtime_sigma - profile.sigma_fraction
            ) > 1e-12
            or runtime_sampler_mode != profile.sampler_mode
            or abs(
                runtime_local_sigma - profile.local_sigma_fraction
            ) > 1e-12
            or runtime_cem_rounds != profile.cem_rounds
            or abs(
                runtime_cem_elite_fraction - profile.cem_elite_fraction
            ) > 1e-12
            or abs(
                runtime_cem_min_std_fraction
                - profile.cem_min_std_fraction
            ) > 1e-12
            or runtime_cost_shaping_profile != profile.cost_shaping_profile
            or abs(
                runtime_prefix_fraction
                - (
                    profile.execution_prefix_steps
                    / profile.horizon_steps
                )
            ) > 1e-12
            or resolved_prefix.resolved_steps
            != profile.execution_prefix_steps
            or abs(
                resolved_prefix.effective_fraction
                - (
                    profile.execution_prefix_steps
                    / profile.horizon_steps
                )
            ) > 1e-12
        ):
            raise PlanningProtocolError(
                "remote solve inputs do not match planner profile"
            )
        world = kwargs["world"]
        raw_start = kwargs.get("start_state")
        start_state = (
            (
                np.asarray(world.data.qpos, dtype=float).copy(),
                np.asarray(world.data.qvel, dtype=float).copy(),
            )
            if raw_start is None
            else (
                np.asarray(raw_start[0], dtype=float).copy(),
                np.asarray(raw_start[1], dtype=float).copy(),
            )
        )
        rng = kwargs["rng"]
        rng_seed = int(
            rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64)
        )
        now = float(self._clock())
        stage_index = int(kwargs.get("stage_idx", 0))
        cycle_index = int(
            kwargs.get(
                "stage_cycle_idx",
                kwargs.get("chunk_idx", 0),
            )
        )
        if not 0 <= cycle_index < profile.max_stage_cycles:
            raise PlanningProtocolError(
                "remote stage cycle index exceeds planner profile"
            )
        program = kwargs.get("program")
        if not isinstance(program, TaskProgram):
            raise PlanningProtocolError(
                "remote planning requires a TaskProgram"
            )
        state = PlanningState(
            qpos=tuple(float(v) for v in start_state[0]),
            qvel=tuple(float(v) for v in start_state[1]),
            applied_ctrl=tuple(
                float(v)
                for v in np.asarray(world.data.ctrl, dtype=float)
            ),
            object_pose7=tuple(
                float(v)
                for v in np.asarray(kwargs["obj_pose7"], dtype=float)
            ),
            nominal_candidate_id=int(nominal.candidate_id),
            nominal_knots=tuple(
                tuple(float(value) for value in row)
                for row in np.asarray(nominal.knots, dtype=float)
            ),
            metadata=dict(metadata.get(int(nominal.candidate_id), {})),
            anchors=anchors_to_payload(kwargs.get("anchors")),
            cost_scale_anchors=anchors_to_payload(
                kwargs.get("cost_scale_anchors")
            ),
            rigid_body_grounding=build_rigid_body_grounding(
                world=world,
                qpos=start_state[0],
                anchors=kwargs.get("anchors"),
            ),
        )
        request = PlanningRequest(
            request_id=(
                f"{self._episode_id}.{stage_index}.{cycle_index}."
                f"{self._request_id_factory()}"
            ),
            episode_id=self._episode_id,
            stage_index=stage_index,
            cycle_index=cycle_index,
            issued_at_unix_s=now,
            deadline_unix_s=now + self._max_latency_s,
            program_json=program.to_json(),
            program_sha256=program_sha256(program),
            state=state,
            state_sha256=state.sha256,
            model_sha256=self._model_sha256,
            assets_sha256=self._assets_sha256,
            planner_sha256=self._planner_sha256,
            service_instance_id=self._service_instance_id,
            planner_profile=profile,
            planner_profile_sha256=profile.sha256,
            config=PlannerConfig(
                pool_size=profile.pool_size,
                horizon_steps=profile.horizon_steps,
                rng_seed=rng_seed,
                execution_prefix_fraction=(
                    profile.execution_prefix_steps
                    / profile.horizon_steps
                ),
                subject_height_m=float(kwargs.get("sub_h", 0.0)),
                execution_prefix_steps=profile.execution_prefix_steps,
                execution_feasibility=(
                    None
                    if kwargs.get("execution_feasibility") is None
                    else ExecutionFeasibility.from_dict(
                        kwargs["execution_feasibility"]
                    )
                ),
            ),
        )
        try:
            response = self._client.plan(request)
        except Exception:
            return self._failed_result(
                nominal,
                metadata,
                request,
                "remote_planning_unavailable",
            )

        self._record_remote_call_sites(
            kwargs.get("episode"), response.call_site_records
        )
        stats = SamplerStats(
            batch={
                "algorithm": response.algorithm,
                "remote_request_sha256": response.request_sha256,
                "remote_state_sha256": response.state_sha256,
                "remote_model_sha256": response.model_sha256,
                "remote_assets_sha256": response.assets_sha256,
                "remote_planner_sha256": response.planner_sha256,
                "remote_planner_profile_sha256": (
                    response.planner_profile_sha256
                ),
                "remote_service_instance_id": response.service_instance_id,
                "remote_rigid_body_grounding_sha256": (
                    request.state.rigid_body_grounding_sha256
                ),
                "device_seed_source": "request_rng_seed",
            },
            pool_size=response.pool_size,
            ran=response.final_n_valid is not None,
            failure_reason=response.failure_reason,
            elapsed_s=response.sampler_elapsed_s,
            flat_objective=response.flat_objective,
            final_n_valid=response.final_n_valid,
            selected_rollout_diagnostics=(
                response.selected_rollout_diagnostics
            ),
            initial_robot_scene_contact=(
                response.initial_robot_scene_contact
            ),
        )
        if not response.plan_valid:
            return MpcStepResult(
                best=nominal,
                sampler_stats=stats,
                metadata=metadata,
                plan_valid=False,
            )
        assert response.knots is not None
        assert response.candidate_id is not None
        best = ControlKnots(
            candidate_id=response.candidate_id,
            knots=np.asarray(response.knots, dtype=float),
        )
        out_meta = dict(metadata)
        out_meta[best.candidate_id] = dict(response.selected_metadata)
        return MpcStepResult(
            best=best,
            sampler_stats=stats,
            metadata=out_meta,
            plan_valid=True,
        )

    @staticmethod
    def _record_remote_call_sites(
        episode: Any,
        records: tuple[dict[str, Any], ...],
    ) -> None:
        if episode is None:
            return
        for raw in records:
            record = CallSiteRecord.from_dict(raw)
            if isinstance(episode, list):
                episode.append(record)
            else:
                episode.record_call_site(record)

    @staticmethod
    def _failed_result(
        nominal: ControlKnots,
        metadata: dict[int, dict[str, Any]],
        request: PlanningRequest,
        reason: str,
    ) -> MpcStepResult:
        return MpcStepResult(
            best=nominal,
            sampler_stats=SamplerStats(
                batch={
                    "algorithm": (
                        "remote_cem"
                        if request.planner_profile.cem_rounds > 1
                        else "remote_predictive_sampling"
                    ),
                    "remote_request_sha256": request.sha256,
                    "remote_planner_profile_sha256": (
                        request.planner_profile_sha256
                    ),
                },
                pool_size=request.config.pool_size,
                ran=False,
                failure_reason=reason,
                final_n_valid=None,
            ),
            metadata=metadata,
            plan_valid=False,
        )
