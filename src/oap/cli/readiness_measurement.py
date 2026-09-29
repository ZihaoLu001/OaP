"""Collect no-motion camera/tracker/controller/H100 readiness evidence.

This command opens:

* the production single-owner ZED stream;
* the persistent FoundationPose tracker;
* a controller observation-only connection (``connect``, never ``__enter__``);
* the release-bound remote planner's loopback HTTP endpoint.

It never obtains a controller lease and contains no arm, gripper, home, mode,
trajectory, stop, or dispatch operation.  The remote plan is a shadow result:
its knots are hashed as evidence and are never forwarded to a controller.
"""
from __future__ import annotations

import argparse
import math
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from oap.loop import observe as observe_mod
from oap.loop.continuous_tracking import ContinuousSceneTracker
from oap.loop.execute import (
    controller_runtime_contract,
    execution_prefix_schedule,
    joint_prefix_knot_times,
    sync_robot_state_to_twin,
)
from oap.loop.readiness_measurement import (
    ReadOnlyRobotStateBuffer,
    ReadinessMeasurementError,
    source_identity,
    write_evidence_package,
)
from oap.loop.runner import LoopConfig
from oap.program import TaskProgram, program_sha256
from oap.program.anchors import Anchor
from oap.program.bindings import bind_program_to_scene
from oap.remote_planner import (
    ExecutionFeasibility,
    HttpPlannerTransport,
    PlannerConfig,
    PlanningRequest,
    PlanningState,
    RemotePlannerClient,
    build_rigid_body_grounding,
    canonical_sha256,
    read_release_bundle,
)
from oap.remote_planner.protocol import anchors_to_payload
from oap.twin import TABLE_TOP_Z, load_scene_manifest, load_world
from oap.utils.io import write_json_atomic
from oap.workers.client import FoundationPoseClient, ZedStreamClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oap-measure-readiness",
        description=(
            "Read-only production tracking + RobotState + remote-H100 shadow "
            "measurement. This command cannot move the robot."
        ),
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--program-json", type=Path, required=True)
    parser.add_argument("--release-bundle", type=Path, required=True)
    parser.add_argument("--remote-planner-url", required=True)
    parser.add_argument("--remote-planner-service-instance-id", required=True)
    parser.add_argument("--observer-python", type=Path, required=True)
    parser.add_argument("--foundationpose-python", type=Path, required=True)
    parser.add_argument("--foundationpose-root", type=Path, required=True)
    parser.add_argument("--server-host", required=True)
    parser.add_argument("--server-port", type=int, default=8766)
    parser.add_argument("--real-table-z", type=float, required=True)
    parser.add_argument("--stage-index", type=int, default=0)
    parser.add_argument("--cycle-index", type=int, default=0)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--sample-timeout-s", type=float, default=30.0)
    parser.add_argument("--remote-timeout-s", type=float, default=300.0)
    parser.add_argument("--robot-sample-period-s", type=float, default=None)
    parser.add_argument("--out", type=Path, required=True)
    return parser


class _ObservationOnlyController:
    """Narrow wrapper that never exposes a lease or command-capable method."""

    def __init__(self, host: str, port: int) -> None:
        from flexiv_control.client import RemoteRobot

        raw = RemoteRobot(
            str(host),
            port=int(port),
            owner="oap-readiness-observation-only",
        )
        missing = [
            name
            for name in ("connect", "get_state", "get_server_info")
            if not callable(getattr(raw, name, None))
        ]
        if missing:
            raise ReadinessMeasurementError(
                "installed flexiv-control lacks the production read-only "
                f"v3 observation API: {missing}"
            )
        # flexiv-control documents connect() as observation-only. In contrast,
        # __enter__() obtains the lease; this command never calls it.
        raw.connect()
        self._get_state = raw.get_state
        self._get_server_info = raw.get_server_info
        self._close = getattr(
            raw,
            "close",
            getattr(raw, "disconnect", lambda: None),
        )

    def get_state(self) -> Any:
        return self._get_state()

    def get_server_info(self) -> dict[str, Any]:
        result = self._get_server_info()
        if not isinstance(result, Mapping):
            raise ReadinessMeasurementError(
                "controller server_info is not an object"
            )
        return dict(result)

    def close(self) -> None:
        self._close()


def _robot_anchors(world: Any, anchors: Any) -> None:
    center = np.asarray(
        world.data.site_xpos[int(world.site_id)], dtype=float
    ).copy()
    rotation = np.asarray(
        world.data.site_xmat[int(world.site_id)], dtype=float
    ).reshape(3, 3)
    if (
        center.shape != (3,)
        or not np.all(np.isfinite(center))
        or not np.all(np.isfinite(rotation))
    ):
        raise ReadinessMeasurementError(
            "current measured robot FK is not finite"
        )
    anchors.add(
        "gripper_center",
        Anchor(
            point=center,
            axis=rotation[:, 2].copy(),
            kind="robot",
        ),
    )
    anchors.add(
        "gripper_closing_axis",
        Anchor(
            point=center,
            axis=rotation[:, 1].copy(),
            kind="robot",
        ),
    )


def _execution_feasibility(
    *,
    world: Any,
    profile: Any,
    runtime: Mapping[str, Any],
) -> ExecutionFeasibility:
    speed_scale = 0.3
    effective = dict(runtime["effective_joint_limits"])
    gripper = dict(runtime["gripper_limits"])
    if speed_scale > float(effective["max_joint_speed_scale"]) + 1e-12:
        raise ReadinessMeasurementError(
            "production execution speed scale exceeds controller ceiling"
        )
    prefix_fraction = (
        profile.execution_prefix_steps / profile.horizon_steps
    )
    knot_times = joint_prefix_knot_times(
        profile.num_knots,
        prefix_fraction,
        horizon_steps=profile.horizon_steps,
    )
    schedule = execution_prefix_schedule(
        horizon_steps=profile.horizon_steps,
        execution_dt_s=float(world.model.opt.timestep),
        prefix_frac=prefix_fraction,
        prefix_knot_times=knot_times,
        control_hz=float(runtime["control_hz"]),
    )
    if not schedule["controller_timing_compatible"]:
        raise ReadinessMeasurementError(
            "release prefix is not compatible with controller timing"
        )
    return ExecutionFeasibility.from_dict(
        {
            "control_hz": float(runtime["control_hz"]),
            "execution_dt_s": float(world.model.opt.timestep),
            "controller_timing_compatible": True,
            "controller_prefix_knot_times": list(
                schedule["spline_prefix_knot_times"]
            ),
            "controller_segment_durations_s": list(
                schedule["segment_durations_s"]
            ),
            "joint_position_min_rad": list(
                effective["enforced_position_min_rad"]
            ),
            "joint_position_max_rad": list(
                effective["enforced_position_max_rad"]
            ),
            "joint_velocity_max_rad_s": (
                np.asarray(
                    effective["base_velocity_max_rad_s"], dtype=float
                )
                * speed_scale
            ).tolist(),
            "effective_joint_limits_sha256": effective["sha256"],
            "gripper_width_min_m": float(gripper["min_width_m"]),
            "gripper_width_max_m": float(gripper["max_width_m"]),
            "gripper_velocity_min_m_s": float(
                gripper["min_velocity_m_s"]
            ),
            "gripper_velocity_max_m_s": float(
                gripper["max_velocity_m_s"]
            ),
            "gripper_limits_sha256": gripper["sha256"],
        }
    )


def _build_shadow_request(
    *,
    world: Any,
    scene: Mapping[str, Any],
    robot_sample: Any,
    anchors: Any,
    program: TaskProgram,
    release: Any,
    service_instance_id: str,
    stage_index: int,
    cycle_index: int,
    rng_seed: int,
    subject_height_m: float,
    execution_feasibility: ExecutionFeasibility,
    timeout_s: float,
) -> PlanningRequest:
    profile = release.planner_profile
    robot = world.robot
    actuator_ids = np.concatenate(
        [
            np.asarray(robot["arm_actuator_ids"], dtype=int),
            np.asarray([int(robot["gripper_actuator_id"])], dtype=int),
        ]
    )
    hold = np.asarray(world.data.ctrl[actuator_ids], dtype=float)
    if hold.shape != (8,) or not np.all(np.isfinite(hold)):
        raise ReadinessMeasurementError(
            "current measured hold is not an eight-control vector"
        )
    nominal = np.repeat(hold[None, :], profile.num_knots, axis=0)
    object_pose7 = np.asarray(
        world.data.qpos[
            int(world.object_qpos_addr):
            int(world.object_qpos_addr) + 7
        ],
        dtype=float,
    )
    capture_id = str(scene["capture_id"])
    capture_unix_s = float(scene["capture_unix_s"])
    capture_monotonic_s = float(scene["capture_monotonic_s"])
    state = PlanningState(
        qpos=tuple(float(v) for v in np.asarray(world.data.qpos)),
        qvel=tuple(float(v) for v in np.asarray(world.data.qvel)),
        applied_ctrl=tuple(float(v) for v in hold),
        object_pose7=tuple(float(v) for v in object_pose7),
        nominal_candidate_id=0,
        nominal_knots=tuple(
            tuple(float(value) for value in row) for row in nominal
        ),
        metadata={
            "source": "current_measured_hold",
            "readiness_shadow": True,
            "dispatch_permitted": False,
            "capture_id": capture_id,
            "capture_unix_s": capture_unix_s,
            "capture_monotonic_s": capture_monotonic_s,
            "robot_state_stamp_s": float(
                robot_sample.robot_state_stamp_s
            ),
            "robot_state_sequence": int(robot_sample.sequence),
        },
        anchors=anchors_to_payload(anchors),
        cost_scale_anchors=anchors_to_payload(anchors),
        rigid_body_grounding=build_rigid_body_grounding(
            world=world,
            qpos=np.asarray(world.data.qpos, dtype=float),
            anchors=anchors,
        ),
    )
    now = time.time()
    return PlanningRequest(
        request_id=(
            f"readiness-shadow.{stage_index}.{cycle_index}."
            f"{uuid.uuid4().hex}"
        ),
        episode_id=f"readiness-shadow-{uuid.uuid4().hex}",
        stage_index=int(stage_index),
        cycle_index=int(cycle_index),
        issued_at_unix_s=now,
        deadline_unix_s=now + float(timeout_s),
        program_json=program.to_json(),
        program_sha256=program_sha256(program),
        state=state,
        state_sha256=state.sha256,
        model_sha256=release.model_sha256,
        assets_sha256=release.assets_sha256,
        planner_sha256=release.planner_sha256,
        service_instance_id=str(service_instance_id),
        planner_profile=profile,
        planner_profile_sha256=profile.sha256,
        config=PlannerConfig(
            pool_size=profile.pool_size,
            horizon_steps=profile.horizon_steps,
            rng_seed=int(rng_seed),
            execution_prefix_fraction=(
                profile.execution_prefix_steps / profile.horizon_steps
            ),
            subject_height_m=float(subject_height_m),
            execution_prefix_steps=profile.execution_prefix_steps,
            execution_feasibility=execution_feasibility,
        ),
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.samples < 1:
        raise SystemExit("--samples must be positive")
    for name in ("sample_timeout_s", "remote_timeout_s"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.stage_index < 0 or args.cycle_index < 0:
        raise SystemExit("stage/cycle indices must be non-negative")
    if not math.isfinite(float(args.real_table_z)):
        raise SystemExit("--real-table-z must be finite")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    controller: _ObservationOnlyController | None = None
    state_buffer: ReadOnlyRobotStateBuffer | None = None
    camera: ZedStreamClient | None = None
    fp: FoundationPoseClient | None = None
    tracker: ContinuousSceneTracker | None = None
    identities: dict[str, Any] = {
        "source": source_identity(
            Path(__file__).resolve().parents[3]
        ),
        "motion_permitted": False,
        "control_lease_acquired": False,
        "controller_commands_sent": 0,
    }
    try:
        release = read_release_bundle(args.release_bundle)
        program = TaskProgram.from_json(
            Path(args.program_json).read_text(encoding="utf-8")
        )
        objects = load_scene_manifest(Path(args.bundle))
        bindings = bind_program_to_scene(program, objects)
        subject = bindings.subject
        reference = bindings.reference
        if not 0 <= args.stage_index < len(program.stages):
            raise ReadinessMeasurementError(
                "--stage-index is outside the current program"
            )
        world = load_world(
            release.plan_xml,
            "W_plan_readiness_shadow_no_dispatch",
        )
        controller = _ObservationOnlyController(
            args.server_host, args.server_port
        )
        server_info = controller.get_server_info()
        runtime = controller_runtime_contract(server_info)
        period = (
            1.0 / float(runtime["control_hz"])
            if args.robot_sample_period_s is None
            else float(args.robot_sample_period_s)
        )
        state_buffer = ReadOnlyRobotStateBuffer(
            controller.get_state,
            period_s=period,
        )
        state_buffer.start()

        cfg = LoopConfig(
            bundle_manifest=Path(args.bundle),
            task="readiness-measurement-only",
            out_dir=out / "tracking",
            program_json=Path(args.program_json),
            observer_python=Path(args.observer_python),
            foundationpose_python=Path(args.foundationpose_python),
            foundationpose_root=Path(args.foundationpose_root),
            foundationpose_mode="actual",
            subject_pose_policy="foundationpose",
            offline=False,
            real_table_z=float(args.real_table_z),
        )
        fp = FoundationPoseClient(
            py=cfg.foundationpose_python,
            root=cfg.foundationpose_root,
        )
        camera = ZedStreamClient(
            py=cfg.observer_python,
            log_path=out / "zed_stream_worker.log",
            timeout_s=float(args.sample_timeout_s),
        )
        z_real_to_plan = TABLE_TOP_Z - float(args.real_table_z)
        seed = observe_mod.observe_scene(
            cfg,
            objects,
            subject,
            "readiness_tracker_seed",
            purpose="replan",
            plan_world=None,
            fp_client=fp,
            z_real_to_plan=z_real_to_plan,
            subject_track_only=False,
            camera_client=camera,
            capture_segmentation=True,
            allow_same_frame_recovery=True,
            require_mask_score=True,
        )
        initial_anchors = observe_mod.ground_observation(
            seed,
            subject,
            objects,
            table_top_z=TABLE_TOP_Z,
            reference=reference,
        )

        def produce(
            sequence: int,
            previous: dict[str, Any] | None,
        ) -> dict[str, Any]:
            assert camera is not None and fp is not None
            scene = observe_mod.observe_scene(
                cfg,
                objects,
                subject,
                f"readiness_stream_slot_{sequence % 2}",
                purpose="replan",
                plan_world=None,
                fp_client=fp,
                previous=previous,
                z_real_to_plan=z_real_to_plan,
                subject_track_only=True,
                camera_client=camera,
                capture_segmentation=False,
                allow_same_frame_recovery=False,
                require_mask_score=False,
            )
            scene["capture_dir"] = None
            scene["capture_storage"] = (
                "ephemeral_two_slot_continuous_tracking"
            )
            return scene

        tracker = ContinuousSceneTracker(
            produce,
            previous=seed,
            accept_unscored_tracking=True,
        )
        tracker.start()

        transport = HttpPlannerTransport(args.remote_planner_url)
        planner = RemotePlannerClient(transport)
        planner_info = planner.require_identity(
            service_instance_id=args.remote_planner_service_instance_id,
            planner_sha256=release.planner_sha256,
            model_sha256=release.model_sha256,
            assets_sha256=release.assets_sha256,
            planner_profile=release.planner_profile,
        )
        execution_feasibility = _execution_feasibility(
            world=world,
            profile=release.planner_profile,
            runtime=runtime,
        )
        identities.update(
            {
                "controller": server_info,
                "camera_and_tracker": {
                    "observer_python": str(args.observer_python),
                    "foundationpose_python": str(
                        args.foundationpose_python
                    ),
                    "foundationpose_root": str(
                        args.foundationpose_root
                    ),
                },
                "remote_planner": planner_info,
                "release": {
                    "path": str(
                        Path(args.release_bundle).resolve()
                    ),
                    "model_sha256": release.model_sha256,
                    "assets_sha256": release.assets_sha256,
                    "planner_sha256": release.planner_sha256,
                    "planner_profile": (
                        release.planner_profile.to_dict()
                    ),
                    "planner_profile_sha256": (
                        release.planner_profile.sha256
                    ),
                    "program_sha256": program_sha256(program),
                },
            }
        )

        required = {obj.name for obj in objects}
        for index in range(args.samples):
            try:
                scene = tracker.consume_latest(
                    max_age_s=float(args.sample_timeout_s),
                    timeout_s=float(args.sample_timeout_s),
                    required_body_names=required,
                )
                if any(
                    body.get("mask_scored") is not False
                    for body in scene["bodies"].values()
                ):
                    raise ReadinessMeasurementError(
                        "measured tracking sample is not the production "
                        "maskless checkpoint path"
                    )
                if scene.get("capture_storage") != (
                    "ephemeral_two_slot_continuous_tracking"
                ):
                    raise ReadinessMeasurementError(
                        "tracking sample did not come from the production "
                        "continuous two-slot producer"
                    )
                consumed_unix = time.time()
                capture_unix = float(scene["capture_unix_s"])
                capture_monotonic = float(
                    scene["capture_monotonic_s"]
                )
                pair = state_buffer.wait_for_exposure_bracket(
                    capture_unix_s=capture_unix,
                    capture_monotonic_s=capture_monotonic,
                    timeout_s=float(args.sample_timeout_s),
                )
                measured = pair.nearest
                sync_robot_state_to_twin(
                    world,
                    list(measured.q_rad),
                    measured.gripper_width_m,
                )
                observe_mod.hot_patch_twin(
                    scene,
                    world,
                    subject,
                    objects,
                )
                anchors = observe_mod.ground_observation(
                    scene,
                    subject,
                    objects,
                    table_top_z=TABLE_TOP_Z,
                    reference=reference,
                    initial_anchors=initial_anchors,
                )
                _robot_anchors(world, anchors)
                request = _build_shadow_request(
                    world=world,
                    scene=scene,
                    robot_sample=measured,
                    anchors=anchors,
                    program=program,
                    release=release,
                    service_instance_id=(
                        args.remote_planner_service_instance_id
                    ),
                    stage_index=args.stage_index,
                    cycle_index=args.cycle_index,
                    rng_seed=args.rng_seed + index,
                    subject_height_m=float(subject.size_lwh[2]),
                    execution_feasibility=execution_feasibility,
                    timeout_s=float(args.remote_timeout_s),
                )
                sent_unix = time.time()
                sent_monotonic = time.monotonic()
                response = planner.plan(request)
                received_monotonic = time.monotonic()
                received_unix = time.time()
                if not response.plan_valid:
                    raise ReadinessMeasurementError(
                        "remote shadow planner returned plan_valid=false: "
                        f"{response.failure_reason}"
                    )
                samples.append(
                    {
                        "schema": (
                            "oap_readiness_measurement_sample_v1"
                        ),
                        "index": index,
                        "status": "ok",
                        "capture_id": scene["capture_id"],
                        "capture_unix_s": capture_unix,
                        "capture_monotonic_s": capture_monotonic,
                        "tracker_sequence": (
                            scene["continuous_tracking"]["sequence"]
                        ),
                        "tracker_maskless": all(
                            body.get("mask_scored") is False
                            for body in scene["bodies"].values()
                        ),
                        "tracker_lost": any(
                            bool(body.get("track_lost"))
                            for body in scene["bodies"].values()
                        ),
                        "tracker_confidence": {
                            name: body.get("confidence")
                            for name, body in scene["bodies"].items()
                        },
                        "consumed_unix_s": consumed_unix,
                        "observation_age_s": (
                            consumed_unix - capture_unix
                        ),
                        "robot_state_bracketed": True,
                        "robot_state_stamp_s": (
                            measured.robot_state_stamp_s
                        ),
                        "robot_state_sequence": measured.sequence,
                        "robot_state_received_unix_s": (
                            measured.received_unix_s
                        ),
                        "camera_robot_timestamp_skew_s": pair.skew_s,
                        "exposure_telemetry_pair": pair.to_dict(),
                        "shadow_request_id": request.request_id,
                        "shadow_request_sha256": request.sha256,
                        "shadow_state_sha256": request.state.sha256,
                        "shadow_request_capture_binding": {
                            key: request.state.metadata[key]
                            for key in (
                                "capture_id",
                                "capture_unix_s",
                                "capture_monotonic_s",
                                "robot_state_stamp_s",
                                "robot_state_sequence",
                            )
                        },
                        "shadow_sent_unix_s": sent_unix,
                        "shadow_received_unix_s": received_unix,
                        "shadow_round_trip_s": (
                            received_monotonic - sent_monotonic
                        ),
                        "shadow_plan_staleness_s": (
                            received_unix - capture_unix
                        ),
                        "shadow_response_completed_at_unix_s": (
                            response.completed_at_unix_s
                        ),
                        "shadow_sampler_elapsed_s": (
                            response.sampler_elapsed_s
                        ),
                        "shadow_plan_valid": response.plan_valid,
                        "shadow_knots_sha256": canonical_sha256(
                            response.knots
                        ),
                        "shadow_dispatch_permitted": False,
                        "shadow_dispatched": False,
                    }
                )
            except BaseException as exc:  # noqa: BLE001 - keep every failure
                failure = {
                    "index": index,
                    "received_unix_s": time.time(),
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                failures.append(failure)
                samples.append(
                    {
                        "schema": (
                            "oap_readiness_measurement_sample_v1"
                        ),
                        "index": index,
                        "status": "failed",
                        "failure": failure,
                    }
                )
    except BaseException as exc:  # noqa: BLE001 - evidence before fail closed
        failures.append(
            {
                "phase": "setup",
                "received_unix_s": time.time(),
                "type": type(exc).__name__,
                "message": str(exc),
            }
        )
    finally:
        def cleanup(label: str, callback: Any) -> None:
            try:
                callback()
            except BaseException as exc:  # noqa: BLE001 - preserve teardown
                failures.append(
                    {
                        "phase": label,
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )

        if tracker is not None:
            cleanup(
                "tracker_shutdown",
                lambda: tracker.stop(timeout_s=10.0),
            )
        if camera is not None:
            cleanup("camera_shutdown", camera.close)
        if fp is not None:
            cleanup("foundationpose_shutdown", fp.stop)
        if state_buffer is not None:
            cleanup("robot_state_shutdown", state_buffer.stop)
            failures.extend(state_buffer.failures)
        if controller is not None:
            cleanup("controller_observer_shutdown", controller.close)
        manifest = write_evidence_package(
            out,
            samples=samples,
            telemetry_samples=(
                [] if state_buffer is None else state_buffer.samples
            ),
            identities=identities,
            failures=failures,
        )
        write_json_atomic(out / "completion.json", {
            "schema": "oap_readiness_measurement_completion_v1",
            "ok": (
                len(samples) == args.samples
                and all(row.get("status") == "ok" for row in samples)
                and not failures
            ),
            "motion_permitted": False,
            "control_lease_acquired": False,
            "controller_commands_sent": 0,
            "sha256_manifest_sha256": canonical_sha256(manifest),
        })
    return 0 if (
        len(samples) == args.samples
        and all(row.get("status") == "ok" for row in samples)
        and not failures
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
