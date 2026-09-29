"""Fail-closed service-side adapter for remote H100 planning."""
from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from oap.loop.mpc_loop import MpcStepResult
from oap.loop.planner_profile import PlannerProfile
from oap.program import CallSiteRecord, TaskProgram
from oap.twin.batched_rollout import BatchedRollout
from oap.twin.control_knots import ControlKnots

from .grounding import validated_rigid_body_grounder
from .protocol import (
    PROTOCOL_VERSION,
    IdentityMismatch,
    PlannerConfig,
    PlanningProtocolError,
    PlanningRequest,
    PlanningResponse,
    ReplayGuard,
    anchors_from_payload,
    canonical_sha256,
    validate_request_at_service,
)
from .warmup import planner_warmup_manifest

WARMUP_EVIDENCE_SCHEMA = "oap_remote_planner_warmup_evidence_v1"


@dataclass(frozen=True)
class PlanningContext:
    """Server-local, non-serializable objects for one exact twin identity.

    ``world`` and ``plan_xml`` intentionally never cross the wire.  A service
    bootstrap loads them from content-addressed assets and registers the
    resulting runtime objects under the same model/assets hashes the lab sends.
    """

    model_sha256: str
    assets_sha256: str
    world: Any
    plan_xml: Any
    planner_profile: PlannerProfile
    movable_grounder: Any = None


class PlanningContextRegistry:
    """Exact-hash lookup for preloaded H100 planning twins."""

    def __init__(self) -> None:
        self._contexts: dict[tuple[str, str, str], PlanningContext] = {}

    def register(self, context: PlanningContext) -> None:
        key = (
            context.model_sha256,
            context.assets_sha256,
            context.planner_profile.sha256,
        )
        if key in self._contexts:
            raise ValueError(
                "planning context already registered for "
                "model/assets/profile hashes"
            )
        self._contexts[key] = context

    def resolve(
        self,
        model_sha256: str,
        assets_sha256: str,
        planner_profile_sha256: str,
    ) -> PlanningContext:
        try:
            return self._contexts[
                (model_sha256, assets_sha256, planner_profile_sha256)
            ]
        except KeyError as exc:
            raise IdentityMismatch(
                "no deployed planning context matches request "
                "model/assets/profile hashes"
            ) from exc

    def identities(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(sorted(self._contexts))


Solver = Callable[..., MpcStepResult]


class PlannerService:
    """Validate, solve once, and return data only; this class cannot execute."""

    def __init__(
        self,
        *,
        contexts: PlanningContextRegistry,
        planner_sha256: str,
        service_instance_id: str,
        solver: Solver | None = None,
        clock: Callable[[], float] = time.time,
        max_request_ttl_s: float = 600.0,
    ) -> None:
        self._contexts = contexts
        self._planner_sha256 = planner_sha256
        self._service_instance_id = service_instance_id
        self._solver = solver
        self._clock = clock
        self._max_request_ttl_s = float(max_request_ttl_s)
        self._replays = ReplayGuard()
        self._cycles = ReplayGuard()
        # MJWarp model/data caches are mutable and one solve consumes the GPU.
        # A persistent service serializes solves rather than pretending they
        # are re-entrant.
        self._solve_lock = threading.Lock()
        self._warmup_evidence: dict[str, Any] | None = None
        self._warmup_cache_snapshots: dict[
            tuple[str, str, str],
            tuple[frozenset[Any], frozenset[Any]],
        ] = {}
        self._warmup_program_sha256: str | None = None

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle one decoded JSON request.

        Protocol/identity/replay failures raise and therefore cannot yield
        knots.  Backend failures return a typed ``plan_valid=false`` response.
        """
        request = PlanningRequest.from_dict(payload)
        accepted = float(self._clock())
        validate_request_at_service(
            request,
            now_unix_s=accepted,
            planner_sha256=self._planner_sha256,
            service_instance_id=self._service_instance_id,
            max_request_ttl_s=self._max_request_ttl_s,
        )
        self._replays.mark(request.request_id, request.sha256)
        self._cycles.mark(
            "cycle-" + canonical_sha256([
                request.episode_id,
                request.stage_index,
                request.cycle_index,
            ]),
            request.sha256,
        )
        context = self._contexts.resolve(
            request.model_sha256,
            request.assets_sha256,
            request.planner_profile_sha256,
        )
        warm_screen: BatchedRollout | None = None
        warm_snapshot: tuple[
            frozenset[Any], frozenset[Any]
        ] | None = None
        if self._warmup_evidence is not None:
            if request.program_sha256 != self._warmup_program_sha256:
                raise IdentityMismatch(
                    "request program was not compiled by service warm-up"
                )
            context_key = self._context_key(context)
            try:
                warm_snapshot = self._warmup_cache_snapshots[context_key]
            except KeyError as exc:
                raise IdentityMismatch(
                    "request planning context was not compiled by service "
                    "warm-up"
                ) from exc
            warm_screen = BatchedRollout.build(
                context.world,
                context.plan_xml,
                gripper="linkage",
            )
            if warm_screen.compilation_cache_snapshot() != warm_snapshot:
                raise PlanningProtocolError(
                    "GPU executable cache drifted from ready evidence"
                )
        try:
            with self._solve_lock:
                validate_request_at_service(
                    request,
                    now_unix_s=float(self._clock()),
                    planner_sha256=self._planner_sha256,
                    service_instance_id=self._service_instance_id,
                    max_request_ttl_s=self._max_request_ttl_s,
                )
                result, records = self._solve(context, request)
                if (
                    warm_screen is not None
                    and warm_screen.compilation_cache_snapshot()
                    != warm_snapshot
                ):
                    raise PlanningProtocolError(
                        "request required an executable shape absent from "
                        "service warm-up"
                    )
        except PlanningProtocolError:
            raise
        except Exception:
            # Do not leak backend paths or exception strings over the control
            # seam; the lab receives an explicit no-plan result.
            completed = float(self._clock())
            return self._failure_response(
                request,
                accepted_at=accepted,
                completed_at=completed,
                failure_reason="remote_solver_failed",
            ).to_dict()

        completed = float(self._clock())
        if completed > request.deadline_unix_s:
            return self._failure_response(
                request,
                accepted_at=accepted,
                completed_at=completed,
                failure_reason="deadline_exceeded_after_solve",
            ).to_dict()
        response = self._result_response(
            request,
            result,
            records,
            accepted_at=accepted,
            completed_at=completed,
        )
        return response.to_dict()

    def info(self) -> dict[str, Any]:
        """Return immutable, non-hardware service identity."""
        contexts = [
            {
                "model_sha256": model_sha256,
                "assets_sha256": assets_sha256,
                "planner_profile": (
                    self._contexts.resolve(
                        model_sha256,
                        assets_sha256,
                        profile_sha256,
                    ).planner_profile.to_dict()
                ),
                "planner_profile_sha256": profile_sha256,
            }
            for model_sha256, assets_sha256, profile_sha256 in sorted(
                self._contexts.identities()
            )
        ]
        return {
            "protocol_version": PROTOCOL_VERSION,
            "service_instance_id": self._service_instance_id,
            "planner_sha256": self._planner_sha256,
            "contexts": contexts,
        }

    def health(self) -> dict[str, Any]:
        """Return immutable readiness evidence without exposing hardware APIs."""
        return {
            "schema": "oap_remote_planner_health_v1",
            "status": (
                "ready"
                if self._warmup_evidence is not None
                else "initializing"
            ),
            "protocol_version": PROTOCOL_VERSION,
            "service_instance_id": self._service_instance_id,
            "planner_sha256": self._planner_sha256,
            "warmup_evidence": copy.deepcopy(self._warmup_evidence),
        }

    @staticmethod
    def _context_key(
        context: PlanningContext,
    ) -> tuple[str, str, str]:
        return (
            context.model_sha256,
            context.assets_sha256,
            context.planner_profile.sha256,
        )

    def warm_up(
        self,
        requests: list[PlanningRequest] | tuple[PlanningRequest, ...],
        *,
        manifest_sha256: str,
    ) -> dict[str, Any]:
        """Compile and execute every frozen Stage twice before readiness.

        This method deliberately bypasses :meth:`handle`: bootstrap requests
        are not live commands, consume no replay identifiers, and do not use an
        HTTP dispatch.  They still pass the ordinary canonical request parser
        and exact model/assets/profile/program identity checks.  A warm-up is
        accepted only when the second execution adds neither a rollout nor a
        device-score executable, which is the service's deterministic
        definition of a hot shape.
        """
        if self._warmup_evidence is not None:
            raise PlanningProtocolError(
                "planner service warm-up may run only once"
            )
        manifest = planner_warmup_manifest(requests)
        if manifest.sha256 != str(manifest_sha256):
            raise IdentityMismatch(
                "planner warm-up manifest_sha256 does not match requests"
            )
        program = TaskProgram.from_json(
            manifest.requests[0].program_json
        )
        stage_evidence: list[dict[str, Any]] = []
        final_snapshots: dict[
            tuple[str, str, str],
            tuple[frozenset[Any], frozenset[Any]],
        ] = {}
        for request in manifest.requests:
            if request.planner_sha256 != self._planner_sha256:
                raise IdentityMismatch(
                    "warm-up planner_sha256 is not deployed here"
                )
            if request.service_instance_id != self._service_instance_id:
                raise IdentityMismatch(
                    "warm-up service_instance_id is not deployed here"
                )
            context = self._contexts.resolve(
                request.model_sha256,
                request.assets_sha256,
                request.planner_profile_sha256,
            )
            if request.planner_profile != context.planner_profile:
                raise IdentityMismatch(
                    "warm-up planner profile does not match its context"
                )
            screen = BatchedRollout.build(
                context.world,
                context.plan_xml,
                gripper="linkage",
            )
            passes: list[dict[str, Any]] = []
            first_hot_snapshot: tuple[
                frozenset[Any], frozenset[Any]
            ] | None = None
            for repetition in range(2):
                started = time.perf_counter()
                with self._solve_lock:
                    result, _records = self._solve(context, request)
                wall_elapsed_s = time.perf_counter() - started
                stats = result.sampler_stats
                if not stats.ran:
                    raise RuntimeError(
                        "GPU_ONLY_REFUSAL: planner warm-up did not execute "
                        f"Stage {request.stage_index} on the GPU"
                    )
                expected_algorithm = (
                    "cem"
                    if request.planner_profile.cem_rounds > 1
                    else "predictive_sampling"
                )
                if (
                    not isinstance(stats.batch, dict)
                    or stats.batch.get("algorithm")
                    != expected_algorithm
                ):
                    raise RuntimeError(
                        "GPU_ONLY_REFUSAL: planner warm-up did not use the "
                        f"production GPU {expected_algorithm} path"
                    )
                if int(stats.pool_size or 0) != request.planner_profile.pool_size:
                    raise RuntimeError(
                        "planner warm-up pool size differs from frozen profile"
                    )
                if int(stats.batch.get("num_knots", 0)) != (
                    request.planner_profile.num_knots
                ):
                    raise RuntimeError(
                        "planner warm-up knot count differs from frozen profile"
                    )
                sampler_elapsed_s = float(stats.elapsed_s or 0.0)
                if (
                    not np.isfinite(wall_elapsed_s)
                    or wall_elapsed_s <= 0.0
                    or not np.isfinite(sampler_elapsed_s)
                    or sampler_elapsed_s <= 0.0
                ):
                    raise RuntimeError(
                        "planner warm-up returned invalid latency evidence"
                    )
                snapshot = screen.compilation_cache_snapshot()
                inventory = screen.compilation_cache_inventory()
                matching_rollouts = [
                    row
                    for row in inventory["rollout_signatures"]
                    if (
                        row["pool_size"]
                        == request.planner_profile.pool_size
                        and row["horizon_steps"]
                        == request.planner_profile.horizon_steps
                        and row["execution_prefix_steps"]
                        == request.planner_profile.execution_prefix_steps
                        and row["continued_state"] is True
                    )
                ]
                if (
                    inventory["device_score_executable_count"] < 1
                    or not matching_rollouts
                ):
                    raise RuntimeError(
                        "planner warm-up did not compile the exact rollout and "
                        "device-score executable"
                    )
                if repetition == 0:
                    first_hot_snapshot = snapshot
                elif snapshot != first_hot_snapshot:
                    raise RuntimeError(
                        "planner warm-up second pass was cold: executable cache "
                        "changed"
                    )
                passes.append(
                    {
                        "repetition": repetition + 1,
                        "wall_elapsed_s": wall_elapsed_s,
                        "sampler_elapsed_s": sampler_elapsed_s,
                        "plan_valid": bool(result.plan_valid),
                        "final_n_valid": stats.final_n_valid,
                        "cache_inventory": inventory,
                    }
                )
            assert first_hot_snapshot is not None
            final_snapshots[self._context_key(context)] = first_hot_snapshot
            stage = program.stages[request.stage_index]
            stage_evidence.append(
                {
                    "stage_index": request.stage_index,
                    "stage_name": stage.name,
                    "stage_spec_sha256": canonical_sha256(stage.to_dict()),
                    "request_sha256": request.sha256,
                    "pool_size": request.planner_profile.pool_size,
                    "horizon_steps": request.planner_profile.horizon_steps,
                    "num_knots": request.planner_profile.num_knots,
                    "execution_prefix_steps": (
                        request.planner_profile.execution_prefix_steps
                    ),
                    "execution_feasibility_enabled": (
                        request.config.execution_feasibility is not None
                    ),
                    "repetitions": passes,
                    "second_pass_hot": True,
                    "hot_wall_elapsed_s": passes[1]["wall_elapsed_s"],
                    "hot_sampler_elapsed_s": passes[1][
                        "sampler_elapsed_s"
                    ],
                }
            )
        first = manifest.requests[0]
        evidence = {
            "schema": WARMUP_EVIDENCE_SCHEMA,
            "service_instance_id": self._service_instance_id,
            "planner_sha256": self._planner_sha256,
            "model_sha256": first.model_sha256,
            "assets_sha256": first.assets_sha256,
            "planner_profile": first.planner_profile.to_dict(),
            "planner_profile_sha256": first.planner_profile_sha256,
            "program_sha256": first.program_sha256,
            "warmup_manifest_sha256": manifest.sha256,
            "repetition_count": 2,
            "all_second_passes_hot": True,
            "stages": stage_evidence,
        }
        self._warmup_cache_snapshots = final_snapshots
        self._warmup_program_sha256 = first.program_sha256
        self._warmup_evidence = evidence
        return copy.deepcopy(evidence)

    def _solve(
        self,
        context: PlanningContext,
        request: PlanningRequest,
    ) -> tuple[MpcStepResult, list[CallSiteRecord]]:
        data = context.world.data
        qpos = np.asarray(request.state.qpos, dtype=float)
        qvel = np.asarray(request.state.qvel, dtype=float)
        ctrl = np.asarray(request.state.applied_ctrl, dtype=float)
        if np.asarray(data.qpos).shape != qpos.shape:
            raise PlanningProtocolError(
                "request qpos shape does not match deployed model"
            )
        if np.asarray(data.qvel).shape != qvel.shape:
            raise PlanningProtocolError(
                "request qvel shape does not match deployed model"
            )
        if np.asarray(data.ctrl).shape != ctrl.shape:
            raise PlanningProtocolError(
                "request applied_ctrl shape does not match deployed model"
            )
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.ctrl[:] = ctrl

        candidate = ControlKnots(
            candidate_id=request.state.nominal_candidate_id,
            knots=np.asarray(request.state.nominal_knots, dtype=float),
        )
        program = TaskProgram.from_json(request.program_json)
        if request.stage_index >= len(program.stages):
            raise PlanningProtocolError(
                "request stage_index is outside the program"
            )
        request_grounder = validated_rigid_body_grounder(
            world=context.world,
            state=request.state,
            program=program,
            stage_index=request.stage_index,
        )
        anchors = anchors_from_payload(request.state.anchors)
        scale_anchors = (
            None
            if not request.state.cost_scale_anchors
            else anchors_from_payload(request.state.cost_scale_anchors)
        )
        records: list[CallSiteRecord] = []
        solver = self._solver
        if solver is None:
            from oap.loop.mpc_loop import solve_mpc_step

            solver = solve_mpc_step
        result = solver(
            world=context.world,
            plan_xml=context.plan_xml,
            candidates=[candidate],
            metadata={
                candidate.candidate_id: dict(request.state.metadata)
            },
            obj_pose7=np.asarray(request.state.object_pose7, dtype=float),
            rng=np.random.default_rng(request.config.rng_seed),
            program=program,
            stage_idx=request.stage_index,
            anchors=anchors,
            cost_scale_anchors=scale_anchors,
            chunk_idx=request.cycle_index,
            sub_h=request.config.subject_height_m,
            episode=records,
            start_state=(qpos.copy(), qvel.copy()),
            pool_size=request.config.pool_size,
            horizon_steps=request.config.horizon_steps,
            sigma_fraction=(
                request.planner_profile.sigma_fraction
            ),
            sampler_mode=request.planner_profile.sampler_mode,
            local_sigma_fraction=(
                request.planner_profile.local_sigma_fraction
            ),
            cem_rounds=request.planner_profile.cem_rounds,
            cem_elite_fraction=(
                request.planner_profile.cem_elite_fraction
            ),
            cem_min_std_fraction=(
                request.planner_profile.cem_min_std_fraction
            ),
            arm_velocity_weight=(
                request.planner_profile.arm_velocity_weight
            ),
            cost_shaping_profile=(
                request.planner_profile.cost_shaping_profile
            ),
            execution_prefix_frac=(
                request.config.resolved_execution_prefix()[1]
            ),
            movable_grounder=request_grounder,
            execution_feasibility=(
                None
                if request.config.execution_feasibility is None
                else request.config.execution_feasibility.to_device_validity()
            ),
        )
        if not isinstance(result, MpcStepResult):
            raise PlanningProtocolError(
                "planner solver returned the wrong result type"
            )
        return result, records

    def _base_response(
        self,
        request: PlanningRequest,
        *,
        accepted_at: float,
        completed_at: float,
        plan_valid: bool,
        candidate_id: int | None,
        knots: tuple[tuple[float, ...], ...] | None,
        failure_reason: str | None,
        final_n_valid: int | None,
        flat_objective: bool,
        initial_robot_scene_contact: bool | None,
        sampler_elapsed_s: float,
        selected_metadata: dict[str, Any],
        selected_rollout_diagnostics: dict[str, Any] | None,
        call_site_records: tuple[dict[str, Any], ...],
    ) -> PlanningResponse:
        config: PlannerConfig = request.config
        return PlanningResponse(
            request_id=request.request_id,
            episode_id=request.episode_id,
            stage_index=request.stage_index,
            cycle_index=request.cycle_index,
            request_sha256=request.sha256,
            state_sha256=request.state_sha256,
            program_sha256=request.program_sha256,
            model_sha256=request.model_sha256,
            assets_sha256=request.assets_sha256,
            planner_sha256=request.planner_sha256,
            planner_profile_sha256=request.planner_profile_sha256,
            accepted_at_unix_s=accepted_at,
            completed_at_unix_s=completed_at,
            deadline_unix_s=request.deadline_unix_s,
            service_instance_id=self._service_instance_id,
            algorithm=(
                "cem_gpu_warp_exact_joint"
                if request.planner_profile.cem_rounds > 1
                else "predictive_sampling_gpu_warp_exact_joint"
            ),
            pool_size=config.pool_size,
            horizon_steps=config.horizon_steps,
            rng_seed=config.rng_seed,
            plan_valid=plan_valid,
            candidate_id=candidate_id,
            knots=knots,
            failure_reason=failure_reason,
            final_n_valid=final_n_valid,
            flat_objective=flat_objective,
            initial_robot_scene_contact=initial_robot_scene_contact,
            sampler_elapsed_s=sampler_elapsed_s,
            selected_metadata=selected_metadata,
            selected_rollout_diagnostics=selected_rollout_diagnostics,
            call_site_records=call_site_records,
        )

    def _failure_response(
        self,
        request: PlanningRequest,
        *,
        accepted_at: float,
        completed_at: float,
        failure_reason: str,
    ) -> PlanningResponse:
        return self._base_response(
            request,
            accepted_at=accepted_at,
            completed_at=completed_at,
            plan_valid=False,
            candidate_id=None,
            knots=None,
            failure_reason=failure_reason,
            final_n_valid=None,
            flat_objective=False,
            initial_robot_scene_contact=None,
            sampler_elapsed_s=max(0.0, completed_at - accepted_at),
            selected_metadata={},
            selected_rollout_diagnostics=None,
            call_site_records=(),
        )

    def _result_response(
        self,
        request: PlanningRequest,
        result: MpcStepResult,
        records: list[CallSiteRecord],
        *,
        accepted_at: float,
        completed_at: float,
    ) -> PlanningResponse:
        stats = result.sampler_stats
        valid = bool(
            result.plan_valid
            and stats.ran
            and stats.final_n_valid is not None
            and int(stats.final_n_valid) > 0
        )
        failure = None
        if not valid:
            failure = stats.failure_reason or "no_valid_remote_plan"
        candidate_id = int(result.best.candidate_id) if valid else None
        knots = (
            tuple(
                tuple(float(value) for value in row)
                for row in np.asarray(result.best.knots, dtype=float)
            )
            if valid
            else None
        )
        metadata = (
            dict(result.metadata.get(int(result.best.candidate_id), {}))
            if valid
            else {}
        )
        return self._base_response(
            request,
            accepted_at=accepted_at,
            completed_at=completed_at,
            plan_valid=valid,
            candidate_id=candidate_id,
            knots=knots,
            failure_reason=failure,
            final_n_valid=stats.final_n_valid,
            flat_objective=bool(stats.flat_objective),
            initial_robot_scene_contact=(
                stats.initial_robot_scene_contact
            ),
            sampler_elapsed_s=float(stats.elapsed_s or 0.0),
            selected_metadata=metadata,
            selected_rollout_diagnostics=(
                stats.selected_rollout_diagnostics
            ),
            call_site_records=tuple(
                record.to_dict() for record in records
            ),
        )
