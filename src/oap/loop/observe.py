"""OBSERVE layer of the online closed loop: ground the REAL scene each chunk.

Role in the two-stage pipeline: ``oap-run`` re-observes the physical scene
between chunks and grounds it into the anchor substrate the TaskProgram
references. Everything here runs on the lab host (the only machine with the
ZED + robot LAN): RGB-D capture goes through the observer-env payload
(``oap.reconstruct.payloads.zed_capture`` under the observer python),
6-DoF subject pose comes from the persistent FoundationPose worker (seeded
once from the generated Any6D reconstruction pose, then tracked), and a
HELD subject is NEVER re-segmented -- it is FK-tracked from the gripper pose
(re-detecting a lifted, occluded object is the "SAM3 found nothing" crash that
used to kill the episode mid-transport) and re-observed normally after release.

Worker access is typed: this module talks to :class:`FoundationPoseClient`
(a Protocol matched by ``oap.workers.client``) so no heavy model is ever
imported in the oap env.
"""
from __future__ import annotations

import copy
import importlib.resources
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from oap.loop.signed_pose_capability import (
    SignedPoseAuthorityRegistry,
    _SIM_CALIBRATION_SHA256,
    _mint_verified_signed_pose,
    _sim_exact_registry,
    signed_pose_frames_authorized,
)
from oap.program import Anchor, AnchorSet
from oap.reconstruct.capture import validate_observation_packet
from oap.twin import TABLE_TOP_Z, SceneObject, observe_object_pose, patch_static_body_pose
from oap.twin.obb import (
    load_base_camera_matrix,
    object_body_quat,
    project_unheld_pose_to_support,
    subject_obb_from_capture,
)
from oap.twin.types import LAB_TABLE_POS, LAB_TABLE_SIZE
from oap.utils.se3 import mat_to_quat, quat_mul, quat_to_mat

if TYPE_CHECKING:  # runtime-cycle-free: runner imports this module, not vice versa
    from oap.loop.runner import LoopConfig

logger = logging.getLogger("oap.loop.observe")

__all__ = [
    "CameraStreamClient",
    "FoundationPoseClient",
    "anchors_from_observation",
    "anchors_from_scene_manifest",
    "capture_observation_packet",
    "capture_synthesis_scene_packet",
    "capture_scene_observation_packet",
    "ground_offline_plan_world",
    "ground_observation",
    "fp_pose_to_base",
    "held_subject_observation_from_gripper",
    "hot_patch_twin",
    "observation_payload_path",
    "observe_scene",
    "observe_subject",
    "observe_subject_fp",
    "observe_subject_obb",
    "project_observation_to_table_support",
    "relocalize_all_static",
    "relocalize_static_object",
    "reset_fp_registry",
    "segment_scene_observation_packet",
    "open_top_container_rests",
    "snap_observation_to_table",
]


# --------------------------------------------------------------------------
# Worker-client contract (implemented by oap.workers.client; kept as a
# Protocol here so the OBSERVE layer never imports the worker package).
# --------------------------------------------------------------------------
@runtime_checkable
class FoundationPoseClient(Protocol):
    """The persistent multi-object FoundationPose worker, as OBSERVE needs it.

    ``request`` performs one register/seed/track call. ``mesh`` routes to the
    per-object estimator inside the multi-object worker (all estimators share
    the heavy nets/glctx), so a request for a DIFFERENT object never tears the
    worker down -- each object keeps its own register-once -> track state.
    ``init_pose`` (a 4x4 camera-frame T_cam_obj) is sent with ``mode="seed"``:
    FP refines FROM it instead of a blind global register.
    """

    def request(self, *, mesh: Path, recording_dir: Path, debug_dir: Path,
                mode: str, timeout_s: float,
                init_pose: np.ndarray | None = None,
                allow_fallback: bool = True,
                capture_id: str | None = None) -> dict[str, Any]:
        """Return the worker response ({'ok', 'pose_txt', 'mode_used', ...})."""
        ...


@runtime_checkable
class CameraStreamClient(Protocol):
    """The single episode-long owner of the physical RGB-D camera."""

    def snapshot(
        self,
        *,
        out_dir: Path,
        objects: list[tuple[str, str]],
    ) -> dict[str, Any]:
        """Materialize one synchronized latest exposure for all objects."""
        ...


# Set of mesh paths whose FoundationPose track has already been initialized
# this episode. Production normally seeds from reconstruction and calls
# track_one; the multi-object worker keeps one pose_last per mesh.
_FP_REGISTERED: set[str] = set()


def reset_fp_registry() -> None:
    """Forget initialized meshes (a restarted worker needs a fresh seed)."""
    _FP_REGISTERED.clear()


# --------------------------------------------------------------------------
# Observer-env payload (RGB-D + SAM3 mask capture under the observer python)
# --------------------------------------------------------------------------
def observation_payload_path(name: str = "zed_capture.py") -> Path:
    """Resolve an observer-env payload shipped under reconstruct/payloads.

    Payloads run under the EXTERNAL observer python (pyzed + SAM3) and import
    nothing from oap, so they are located as package data, never imported.
    """
    try:
        candidate = importlib.resources.files("oap.reconstruct") / "payloads" / name
        with importlib.resources.as_file(candidate) as p:
            path = Path(p)
    except (ModuleNotFoundError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"observer payload {name!r} not found under oap.reconstruct.payloads "
            f"-- is the oap package installed with its payload data "
            f"([tool.setuptools.package-data])?") from exc
    if not path.exists():
        raise RuntimeError(
            f"observer payload {name!r} resolved to {path} but the file is missing")
    return path


def capture_observation_packet(cfg: LoopConfig, obj: SceneObject, tag: str,
                               *, seg: bool = True) -> Path:
    """Capture one ZED RGB-D + SAM3-mask packet for ``obj`` via the observer env.

    Writes ``<out>/fp_obs/<tag>/<obj.name>/{rgb, depth_m.npy, mask_00.png,
    cam_K.txt, foundationpose_recording/}`` -- the FP test-scene format.

    Raises:
        subprocess.CalledProcessError: when the capture fails (e.g. SAM3 found
            nothing) -- callers decide whether that is fatal (pre-grasp) or
            expected (held/occluded, placed inside the container).
    """
    if cfg.observer_python is None:
        raise RuntimeError(
            "no observer python configured -- set observer_python from "
            "configs/external_envs.yaml (key: observer_python) or --observer-python")
    cap = Path(cfg.out_dir) / "fp_obs" / tag
    argv = [str(cfg.observer_python), str(observation_payload_path()),
            "--out", str(cap), "--object", obj.name, obj.sam3_prompt]
    if not seg:
        argv.append("--no-seg")
    capture_started_s = time.time()
    try:
        subprocess.run(argv, check=True, timeout=300)
    except subprocess.CalledProcessError:
        # The ZED wedges transiently after rapid open/close cycles ("CAMERA
        # MOTION SENSORS NOT DETECTED") and recovers within ~30 s -- the wedge
        # killed two live episodes at their decisive observation. One paced
        # retry; a REAL failure (SAM3 found nothing) simply fails again and
        # the callers' fatal-vs-expected routing is unchanged.
        import time as _time
        logger.warning("[observe] capture failed for %s (%s) -- retrying once "
                       "in 20 s (transient ZED wedge recovery)", tag, obj.name)
        _time.sleep(20.0)
        capture_started_s = time.time()
        subprocess.run(argv, check=True, timeout=300)
    validate_observation_packet(
        cap / obj.name,
        require_mask=seg,
        captured_after_s=capture_started_s,
        captured_before_s=time.time(),
    )
    return cap

def capture_synthesis_scene_packet(cfg: LoopConfig, tag: str) -> Path:
    """Capture one read-only RGB-D scene frame before task-role binding.

    This intentionally requests no segmentation and names no manifest object:
    the frame is evidence for synthesis, not an observation of a guessed
    subject. The payload still needs an ``--object`` packet key, so a reserved
    ``__scene__`` key is used. Returns that packet directory.

    A future scene observer may replace the manifest poses with scene-wide
    tracked poses at this seam; callers already consume a complete named anchor
    vocabulary and do not depend on a controlled-object choice.
    """
    if cfg.observer_python is None:
        raise RuntimeError(
            "no observer python configured -- cannot capture synthesis scene")
    cap = Path(cfg.out_dir) / "fp_obs" / tag
    argv = [
        str(cfg.observer_python), str(observation_payload_path()),
        "--out", str(cap), "--object", "__scene__", "tabletop scene",
        "--no-seg",
    ]
    capture_started_s = time.time()
    subprocess.run(argv, check=True, timeout=300)
    capture_finished_s = time.time()
    packet = cap / "__scene__"
    validate_observation_packet(
        packet,
        require_mask=False,
        captured_after_s=capture_started_s,
        captured_before_s=capture_finished_s,
    )
    return packet


def capture_scene_observation_packet(
    cfg: LoopConfig,
    objects: list[SceneObject],
    tag: str,
    *,
    seg: bool = True,
    camera_client: CameraStreamClient | None = None,
) -> Path:
    """Capture all requested objects from one synchronized ZED RGB-D frame."""
    if cfg.observer_python is None:
        raise RuntimeError(
            "no observer python configured for synchronized scene capture")
    if not objects:
        raise ValueError("scene capture needs at least one object")
    cap = Path(cfg.out_dir) / "fp_obs" / tag
    if camera_client is not None:
        camera_client.snapshot(
            out_dir=cap,
            objects=[
                (obj.name, obj.sam3_prompt)
                for obj in objects
            ],
        )
        if seg:
            argv = [
                str(cfg.observer_python),
                str(observation_payload_path()),
                "--out",
                str(cap),
            ]
            for obj in objects:
                argv.extend(
                    ["--object", obj.name, obj.sam3_prompt]
                )
            argv.append("--seg-only")
            subprocess.run(argv, check=True, timeout=300)
        for obj in objects:
            validate_observation_packet(
                cap / obj.name,
                require_mask=seg,
            )
        return cap
    argv = [str(cfg.observer_python), str(observation_payload_path()),
            "--out", str(cap)]
    for obj in objects:
        argv.extend(["--object", obj.name, obj.sam3_prompt])
    if not seg:
        argv.append("--no-seg")
    capture_started_s = time.time()
    try:
        subprocess.run(argv, check=True, timeout=300)
    except subprocess.CalledProcessError:
        import time as _time
        logger.warning(
            "[observe] synchronized scene capture failed for %s -- retrying "
            "once in 20 s", tag)
        _time.sleep(20.0)
        capture_started_s = time.time()
        subprocess.run(argv, check=True, timeout=300)
    capture_finished_s = time.time()
    for obj in objects:
        validate_observation_packet(
            cap / obj.name,
            require_mask=seg,
            captured_after_s=capture_started_s,
            captured_before_s=capture_finished_s,
        )
    return cap


# --------------------------------------------------------------------------
# FP pose conversion
# --------------------------------------------------------------------------
def fp_pose_to_base(pose_txt: Path, t_base_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Map a 4x4 CAMERA(OpenCV)-frame object pose to the ROBOT BASE frame.

    Returns (pos[3], quat_wxyz[4], yaw). The yaw is read off the object x-axis
    in-plane.
    """
    T_co = np.loadtxt(str(pose_txt)).reshape(4, 4)        # camera <- object
    T_bo = np.asarray(t_base_camera, float) @ T_co        # base <- object
    pos = T_bo[:3, 3]
    R = T_bo[:3, :3]
    quat = mat_to_quat(R)
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return pos, quat, yaw


# --------------------------------------------------------------------------
# Subject observers (fp | obb policy)
# --------------------------------------------------------------------------
def segment_observation_packet(cfg: LoopConfig, obj: SceneObject, cap: Path) -> None:
    """Run SAM3 on an ALREADY-CAPTURED packet (the payload's --seg-only pass).

    This is a packet-reconstruction/diagnostic utility, not the production
    maskless tracking recovery policy. Production rejects and rolls back a
    lost update, retries tracking on a later continuous frame, and fails the
    checkpoint closed if no valid fresh pose arrives; it does not call this
    helper to re-register the rejected frame.

    ``cap`` is the capture TAG directory (``<out>/fp_obs/<tag>``) -- the same
    ``--out`` the capture pass used. The payload appends the object itself
    (``out = args.out / name``) in every mode, so passing the packet dir here
    double-nests it.
    """
    argv = [str(cfg.observer_python), str(observation_payload_path()),
            "--out", str(cap), "--object", obj.name, obj.sam3_prompt,
            "--seg-only"]
    subprocess.run(argv, check=True, timeout=300)
    validate_observation_packet(cap / obj.name, require_mask=True)


def segment_scene_observation_packet(
    cfg: LoopConfig,
    objects: list[SceneObject],
    cap: Path,
) -> None:
    """Run one SAM3 process over an existing synchronized scene exposure.

    This operation never opens the camera.  All object masks are synthesized
    from the RGB files already materialized by the single ZED owner.
    """
    if cfg.observer_python is None:
        raise RuntimeError("no observer python configured for SAM3")
    if not objects:
        raise ValueError("scene segmentation needs at least one object")
    argv = [
        str(cfg.observer_python),
        str(observation_payload_path()),
        "--out",
        str(Path(cap)),
    ]
    for obj in objects:
        argv.extend(["--object", obj.name, obj.sam3_prompt])
    argv.append("--seg-only")
    subprocess.run(argv, check=True, timeout=300)
    for obj in objects:
        validate_observation_packet(
            Path(cap) / obj.name,
            require_mask=True,
        )


def observe_subject_fp(cfg: LoopConfig, subject: SceneObject, tag: str,
                       fp_client: FoundationPoseClient, *,
                       track_only: bool = False,
                       purpose: str = "replan",
                       capture_dir: Path | None = None,
                       use_config_mesh_override: bool = True,
                       allow_same_frame_recovery: bool = False,
                       require_mask_score: bool = False) -> dict[str, Any]:
    """FoundationPose REGISTER-once -> TRACK observation of the subject.

    ``purpose="replan"`` is the single closed-loop observation policy.  The
    worker stays alive and preserves ``pose_last`` across the episode: it
    registers each mesh on first sight, then refines that same persistent track
    from every current RGB-D snapshot.  The snapshot taken after an executed
    MPC prefix is used both to update the twin and to grade the Stage terminal;
    terminal satisfaction never restarts the tracker.

    Captures a ZED RGB-D + SAM3 mask (the FP test-scene format) via the
    observer env, then the persistent FP worker REGISTERs on the first call and
    TRACKs (single-hypothesis refine from the last pose) on every call after;
    the 4x4 camera-frame pose is mapped to base frame. The loop then corrects
    the SIMULATED object pose from this TRACKED 6-DoF pose -- including the
    HELD object after it leaves the table (where a table-plane observer fails).

    The FIRST sight is seeded from the integrity-bound Any6D camera-frame pose
    in ``subject.refined_json`` and refined against the current synchronized
    frame.  This is generated reconstruction output, not a task-specific or
    manually aligned initialization.  Runtime never launches FoundationPose's
    252-hypothesis global register, whose transient memory exceeds the lab
    GPU.  Every later sight refines the same persistent track.
    """
    if purpose != "replan":
        raise ValueError(f"purpose must be 'replan', got {purpose!r}")
    # A normal TRACK snapshot does not need a mask -- track_one takes none
    # (official API) and drift is gated by the worker's full-resolution
    # point-cloud support check. Segmentation runs on first sight. Production
    # maskless loss is rejected transactionally; the continuous producer
    # retries a later frame and the checkpoint fails closed if none arrives.
    # It does not lazily segment/re-register the rejected frame.
    configured_mesh = (
        cfg.foundationpose_mesh if use_config_mesh_override else None
    )
    # The manifest's canonical `mesh` is the automatically reconstructed,
    # FP-safe decimated tracking mesh, in the same object frame as the bound
    # pose. `collision` is only the physics proxy and `visual_mesh` is the
    # full-resolution render asset; neither is a tracking substitute. Real
    # execution authorization forbids `foundationpose_mesh` overrides, so
    # production always follows this generated manifest identity.
    mesh = Path(configured_mesh) if configured_mesh else Path(subject.mesh)
    try:
        mesh = mesh.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"FoundationPose tracking mesh is missing: {mesh}"
        ) from exc
    fp_key = str(mesh)
    first_sight = fp_key not in _FP_REGISTERED
    init_pose: np.ndarray | None = None
    if first_sight:
        try:
            refined = json.loads(
                Path(subject.refined_json).read_text(encoding="utf-8")
            )
            rotation = np.asarray(
                refined["R_cam_obj"], dtype=float
            ).reshape(3, 3)
            translation = np.asarray(
                refined["center_cam_m"], dtype=float
            ).reshape(3)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{subject.name!r} has no valid generated Any6D "
                "R_cam_obj/center_cam_m seed"
            ) from exc
        if not (
            np.all(np.isfinite(rotation))
            and np.all(np.isfinite(translation))
        ):
            raise RuntimeError(
                f"{subject.name!r} generated Any6D seed is non-finite"
            )
        init_pose = np.eye(4, dtype=float)
        init_pose[:3, :3] = rotation
        init_pose[:3, 3] = translation
    _seg = first_sight
    _t_cap0 = time.perf_counter()
    cap = (capture_observation_packet(cfg, subject, tag, seg=_seg)
           if capture_dir is None else Path(capture_dir))
    _t_capture = time.perf_counter() - _t_cap0
    manifest_path = cap / subject.name / "observation_manifest.json"
    packet_manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            packet_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "FoundationPose packet manifest is unreadable: "
                f"{manifest_path}"
            ) from exc
    capture_id_raw = packet_manifest.get("capture_id")
    capture_id = (
        None if capture_id_raw is None else str(capture_id_raw)
    )
    rec = cap / subject.name / "foundationpose_recording"
    # Per-object initialize-once -> track: this object's first sight refines its
    # reconstruction seed; every later sight tracks (the worker holds a
    # separate pose_last per generated tracking mesh).
    if track_only and fp_key not in _FP_REGISTERED:
        raise RuntimeError(
            "held/occluded subject has no prior FoundationPose track; refusing "
            "a global register under occlusion")
    mode = "track" if not first_sight else "seed"
    t_base_camera = load_base_camera_matrix()
    _t_fp0 = time.perf_counter()
    fp_request_kwargs: dict[str, Any] = {
        "mesh": mesh,
        "recording_dir": rec.resolve(),
        "debug_dir": (cap / "fp_debug").resolve(),
        "mode": mode,
        "timeout_s": float(cfg.foundationpose_timeout_s),
        "init_pose": init_pose,
        # A bad seed/track is rejected and retried from the last accepted pose.
        # Never invoke the global hypothesis register: it both OOMs this GPU
        # and hallucinates under gripper occlusion.
        "allow_fallback": bool(allow_same_frame_recovery),
    }
    if capture_id is not None:
        fp_request_kwargs["capture_id"] = capture_id
    resp = fp_client.request(**fp_request_kwargs)
    _t_fp = time.perf_counter() - _t_fp0
    lost = resp.get("lost")
    if not isinstance(lost, bool):
        raise RuntimeError(
            "FoundationPose response has no explicit boolean lost state"
        )
    confidence_raw = resp.get("confidence")
    confidence: float | None
    if confidence_raw is None and not require_mask_score:
        confidence = None
    else:
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "FoundationPose response has no numeric confidence"
            ) from exc
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise RuntimeError(
                "FoundationPose response confidence is not finite in [0, 1]"
            )
    if require_mask_score and resp.get("mask_iou") is None:
        raise RuntimeError(
            "FoundationPose checkpoint has no same-frame mask IoU"
        )
    if capture_id is not None and resp.get("capture_id") != capture_id:
        raise RuntimeError(
            "FoundationPose response is not bound to its input capture_id"
        )
    if lost:
        raise RuntimeError(
            f"FoundationPose rejected {subject.name!r} track "
            f"(mask_iou={resp.get('mask_iou')}, "
            f"pc_support={resp.get('pc_support')})"
        )
    _FP_REGISTERED.add(fp_key)
    pos, quat, yaw = fp_pose_to_base(Path(resp["pose_txt"]), t_base_camera)
    logger.info("[fp] %s: mode=%s reregistered=%s conf=%s pos=%s yaw=%.3f",
                tag, resp.get("mode_used"), resp.get("reregistered"),
                resp.get("confidence"), np.round(pos, 4).tolist(), yaw)
    # Latency breakdown, per observe. The stop-to-pose wall clock was
    # UNINSTRUMENTED until 2026-07-16 -- every prior number came from
    # reconstructing log timestamps against file mtimes. capture is the
    # cold-process ZED+SAM3 subprocess; fp is the resident worker's
    # register/track. mode_used tells register (heavy) from track (cheap).
    logger.info("[observe-timing] %s: capture=%.2fs fp=%.2fs (mode=%s) total=%.2fs",
                tag, _t_capture, _t_fp, resp.get("mode_used"),
                _t_capture + _t_fp)
    return {"schema": "foundationpose_track_observation", "tag": tag,
            "pos_base": [float(v) for v in pos], "yaw_base_rad": float(yaw),
            "quat_wxyz": [float(v) for v in quat],
            "fp_mode_used": resp.get("mode_used"),
            "fp_confidence": confidence,
            "fp_lost": lost,
            "fp_mask_iou": resp.get("mask_iou"),
            "fp_pc_support": resp.get("pc_support"),
            "capture_id": capture_id,
            "track_only_probe": bool(track_only)}


def observe_subject_obb(
    cfg: LoopConfig,
    subject: SceneObject,
    tag: str,
    *,
    capture_dir: Path | None = None,
) -> dict[str, Any]:
    """Gravity-aligned depth-OBB localization of the subject.

    Robust to FP's symmetry flips on a textureless box. Captures a ZED RGB-D +
    SAM3 mask (same capture as the FP path), then fits the gravity-aligned OBB.
    The loop places the subject UPRIGHT (yaw-only), so this returns the
    STANDING dims (which side is vertical) + footprint yaw -- correcting a box
    stood on its narrow side that FoundationPose otherwise twins as lying flat.
    """
    cap = (capture_observation_packet(cfg, subject, tag)
           if capture_dir is None else Path(capture_dir))
    _tz = float(cfg.real_table_z) if cfg.real_table_z is not None else TABLE_TOP_Z
    center, yaw, dims = subject_obb_from_capture(cap, subject, _tz)
    logger.info("[obb] %s: center=%s yaw=%.3f dims_LWH=%s (refined dims were %s)",
                tag, np.round(center, 4).tolist(), yaw, np.round(dims, 4).tolist(),
                tuple(round(x, 4) for x in subject.size_lwh))
    return {"schema": "gravity_obb_observation", "tag": tag,
            "pos_base": [float(v) for v in center], "yaw_base_rad": float(yaw),
            "size_lwh_obb": [float(v) for v in dims]}


def observe_subject(cfg: LoopConfig, subject: SceneObject, tag: str,
                    plan_world: Any = None,
                    fp_client: FoundationPoseClient | None = None,
                    track_only: bool = False,
                    purpose: str = "replan",
                    capture_dir: Path | None = None,
                    use_config_mesh_override: bool = True,
                    allow_same_frame_recovery: bool = False,
                    require_mask_score: bool = False) -> dict[str, Any]:
    """Observe the SUBJECT object (real camera, or the plan world when offline).

    ``track_only`` marks a PROBE (held-verify): the FP path returns its cheap
    low-confidence track verbatim instead of the register fallback.

    Policy routing (the fp|obb subject-pose policy):
      * ``cfg.offline``  -> read the pose from the plan world (fully local dry
        loop) or from ``cfg.initial_obs_json`` at scene init;
      * ``subject_pose_policy == 'gravity_obb'`` OR
        ``foundationpose_mode != 'actual'`` -> gravity-aligned depth OBB;
      * otherwise -> FoundationPose register-once -> track (needs ``fp_client``).
    """
    if cfg.offline:
        if plan_world is None and cfg.initial_obs_json is not None:
            return dict(json.loads(Path(cfg.initial_obs_json).read_text(encoding="utf-8")))
        pose = observe_object_pose(plan_world)
        w, x, y, z = pose[3:]
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        ax = R @ subject.mesh_long_axis
        yaw = float(np.arctan2(ax[1], ax[0]))
        return {"schema": "offline_plan_world_observation", "tag": tag,
                "pos_base": [float(v) for v in pose[:3]], "yaw_base_rad": yaw}
    if cfg.subject_pose_policy == "gravity_obb" or cfg.foundationpose_mode != "actual":
        return observe_subject_obb(
            cfg, subject, tag, capture_dir=capture_dir)
    if fp_client is None:
        raise RuntimeError(
            "subject_pose_policy='foundationpose' requires a FoundationPose worker "
            "client -- configure foundationpose_python/foundationpose_root "
            "(configs/external_envs.yaml keys: foundationpose_python, "
            "foundationpose_root) or use --subject-pose-policy gravity_obb")
    return observe_subject_fp(cfg, subject, tag, fp_client, track_only=track_only,
                              purpose=purpose, capture_dir=capture_dir,
                              use_config_mesh_override=use_config_mesh_override,
                              allow_same_frame_recovery=allow_same_frame_recovery,
                              require_mask_score=require_mask_score)


# --------------------------------------------------------------------------
# Held-object tracking + post-observation corrections
# --------------------------------------------------------------------------
def held_subject_observation_from_gripper(robot: Any, grasp_tcp_offset: Any, tag: str,
                                          held_tool_yaw: float | None) -> dict[str, Any]:
    """Track a HELD subject via the gripper instead of re-detecting it on the table.

    Once grasped, the object moves rigidly with the end-effector, so its pose is
    ``grasp_point = TCP + R(tcp_quat) @ grasp_tcp_offset`` (the calibrated grasp
    point). A table-facing SAM3 re-detect of a lifted/occluded object fails
    ("SAM3 found nothing"), which previously crashed the episode mid-transport
    -- so pick-and-place could never reach the place step. The subject is
    re-observed normally once released (open-after-close), when it is visible
    again in/at the target.
    """
    st = robot.get_state()
    tcp = np.asarray(st.tcp_pose, dtype=float).reshape(7)
    pos, q = tcp[:3], tcp[3:]
    off = np.asarray(grasp_tcp_offset, dtype=float)
    box_pos = pos + quat_to_mat(q) @ off
    if held_tool_yaw is not None:
        yaw = float(held_tool_yaw)
    else:
        yaw = float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                               1 - 2 * (q[2] * q[2] + q[3] * q[3])))
    return {"schema": "held_gripper_observation", "tag": tag,
            "pos_base": [float(v) for v in box_pos], "yaw_base_rad": yaw,
            "quat_wxyz": [float(v) for v in q], "held_via_gripper": True}


def open_top_container_rests(objects: list[SceneObject], subject: SceneObject,
                             real_table_z: float | None,
                             ) -> list[tuple[str, tuple[float, float], float]]:
    """Container footprints for :func:`snap_observation_to_table`: every
    open-top container's ``(name, center_xy, cover_radius)``.

    ``center_xy`` is the LIVE pose (relocalized each chunk); the cover radius
    is the circumscribed half-extent -- conservative on purpose (axis
    conventions vary per reconstruction, and over-covering only means keeping
    the honest raw z for a subject standing right against the wall).
    """
    out: list[tuple[str, tuple[float, float], float]] = []
    if real_table_z is None:
        return out
    for o in objects:
        if o is subject or not o.open_top_container or o.pose_base is None:
            continue
        out.append((o.name,
                    (float(o.pose_base[0]), float(o.pose_base[1])),
                    0.5 * float(max(o.size_lwh))))
    return out


def snap_observation_to_table(obs: dict[str, Any], subject_size_lwh: tuple[float, float, float],
                              real_table_z: float | None, tag: str, *,
                              enabled: bool,
                              containers: tuple | list = (),
                              ) -> dict[str, Any]:
    """Snap an ON-TABLE observation's center z so the object RESTS on the table.

    Fixes a HIGH estimator z that floats the gripper (z = real_table_z + half
    height). Only applies when both the snap flag and a measured table z are
    set; a held/lifted observation must never be snapped (the held-verify probe
    reads the RAW z on purpose) -- and neither may a subject INSIDE an open-top
    container's footprint (``containers`` from
    :func:`open_top_container_rests`): it rests on the container floor with an
    unknown effective half-height (it may sit, lie, or lean), and snapping a
    just-placed subject to the TABLE once dragged its z below the container
    bottom, failed the terminal predicate the Tier-2 judge had confirmed at
    0.95, and sent the loop re-grasping air inside the box. Inside a footprint
    the RAW measured z is the honest value for both the terminal predicate
    and the twin re-pose.
    """
    if not (enabled and real_table_z is not None):
        return obs
    x, y = float(obs["pos_base"][0]), float(obs["pos_base"][1])
    for name, (cx, cy), radius in containers:
        if abs(x - cx) <= radius and abs(y - cy) <= radius:
            logger.info("[snap-table] %s subject xy inside %s footprint "
                        "(|%.3f-%.3f|,|%.3f-%.3f| <= %.3f) -> keeping RAW z %.4f",
                        tag, name, x, cx, y, cy, radius, float(obs["pos_base"][2]))
            return obs
    vhe = 0.5 * float(subject_size_lwh[2])
    z0 = float(obs["pos_base"][2])
    obs["pos_base"] = [x, y, float(real_table_z) + vhe]
    logger.info("[snap-table] %s subject z %.4f -> %.4f (table %.3f + half %.4f); "
                "estimator was %+.0fmm off",
                tag, z0, obs["pos_base"][2], float(real_table_z), vhe,
                1000.0 * (z0 - obs["pos_base"][2]))
    return obs


def project_observation_to_table_support(
    obs: dict[str, Any],
    obj: SceneObject,
    table_top_z: float | None,
    tag: str,
    *,
    held: bool | None,
    containers: tuple | list = (),
) -> dict[str, Any]:
    """Apply the generated-observation table-support lower bound.

    Unlike :func:`snap_observation_to_table`, this automatic seam never lowers
    a pose and has no per-object switch or target height.  It only removes a
    physically impossible penetration of the known table by a confirmed
    unheld reconstructed body.
    """
    if table_top_z is None:
        return obs
    if obj.q_om is None:
        raise ValueError(f"{obj.name}: table support projection requires q_om")
    quat = obs.get("quat_wxyz")
    if quat is None:
        quat = object_body_quat(
            float(obs.get("yaw_base_rad", 0.0)),
            np.asarray(obj.q_om, dtype=float),
        )
    raw = np.asarray(obs["pos_base"], dtype=float).reshape(3)
    projected, changed = project_unheld_pose_to_support(
        raw,
        quat,
        obj.size_lwh,
        obj.q_om,
        float(table_top_z),
        held=held,
        support_center_xy=LAB_TABLE_POS[:2],
        support_half_size_xy=LAB_TABLE_SIZE[:2],
        excluded_footprints=containers,
    )
    if not changed:
        return obs
    out = dict(obs)
    out["raw_pos_base"] = raw.tolist()
    out["pos_base"] = projected.tolist()
    out["table_support_projection"] = {
        "source": "global_table_support_lower_bound",
        "raw_z": float(raw[2]),
        "projected_z": float(projected[2]),
    }
    logger.info(
        "[table-support] %s %s z %.4f -> %.4f "
        "(orientation-aware unheld lower bound)",
        tag,
        obj.name,
        float(raw[2]),
        float(projected[2]),
    )
    return out


# --------------------------------------------------------------------------
# Static-object relocalization (reference/container follows the live scene)
# --------------------------------------------------------------------------
def relocalize_static_object(cfg: LoopConfig, obj: SceneObject, plan_world: Any,
                             z_real_to_plan: float, tag: str,
                             fp_client: FoundationPoseClient | None = None,
                             ) -> tuple[float, float]:
    """Re-observe a STATIC (movable:false) object and hot-patch its twin body.

    Updates the ``support_<name>`` twin body + ``pose_base``/``pose_quat`` to
    the live scene, so the planner re-grounds to where the object actually is
    now. Yaw-only about world-up (tilt kept) and the recon RESTING z is kept (a
    static object's height does not change between captures; the observed z is
    noisier). Mutates ``obj`` in place. Returns (pos_drift_mm, yaw_drift_deg).
    Propagates a capture failure (SAM3 found nothing / occluded) so the caller
    can skip-and-keep.

    ``plan_world`` may be None at SCENE INIT (before the twin is built) -- then
    the ``support_<name>`` body patch is skipped and only ``obj.pose_base`` /
    ``pose_quat`` are set, so the twin is built FROM the live observation
    instead of the recorded pose.
    """
    ref_obs = observe_subject(cfg, obj, tag, plan_world=plan_world,
                              fp_client=fp_client)
    old_base = np.asarray(obj.pose_base, dtype=float)
    new_base = np.asarray(ref_obs["pos_base"], dtype=float)
    new_base[2] = old_base[2]                                   # keep recon resting z
    pos_drift_mm = 1000.0 * float(np.linalg.norm(new_base[:2] - old_base[:2]))
    old_quat = np.asarray(obj.pose_quat, dtype=float)
    _r = quat_to_mat(old_quat) @ np.array([1.0, 0.0, 0.0])
    old_yaw = float(np.arctan2(_r[1], _r[0]))
    new_yaw = float(ref_obs.get("yaw_base_rad", old_yaw))
    d_yaw = (new_yaw - old_yaw + np.pi) % (2 * np.pi) - np.pi
    yaw_drift_deg = abs(float(np.degrees(d_yaw)))
    half = 0.5 * d_yaw
    q_dz = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    new_quat = quat_mul(q_dz, old_quat)
    # Keep the cached pose's frame; only raw base-frame objects need translation.
    if plan_world is not None:
        plan_pos = new_base.copy()
        if not obj.z_snapped_plan_frame:
            plan_pos[2] += z_real_to_plan
        patch_static_body_pose(plan_world, f"support_{obj.name}", plan_pos, new_quat)
    obj.pose_base = [float(v) for v in new_base]
    obj.pose_quat = [float(v) for v in new_quat]
    return pos_drift_mm, yaw_drift_deg


def relocalize_all_static(cfg: LoopConfig, objects: list[SceneObject], subject: SceneObject,
                          plan_world: Any, z_real_to_plan: float, chunk_idx: int,
                          fp_client: FoundationPoseClient | None = None,
                          ) -> dict[str, tuple[float, float]]:
    """Re-localize EVERY non-subject STATIC object this chunk (multi-object FP).

    The subject is handled separately (FK while held, FP-track otherwise). Each
    object is FoundationPose register-once-then-tracked in the multi-object
    worker, so this follows a bumped/shifted container through the episode. An
    occluded miss (SAM3 found nothing, e.g. the container hidden behind the
    held object) is skipped, keeping its last good pose. Returns per-object
    (pos_drift_mm, yaw_drift_deg) for the drifts that were applied.
    """
    drifts: dict[str, tuple[float, float]] = {}
    for obj in objects:
        if obj.name == subject.name or obj.movable:
            continue  # subject handled elsewhere; movable non-subjects have no support_ body
        try:
            dmm, ddeg = relocalize_static_object(
                cfg, obj, plan_world, z_real_to_plan,
                f"{obj.name}_relocalize_chunk_{chunk_idx:02d}",
                fp_client=fp_client)
            drifts[obj.name] = (dmm, ddeg)
            warn = "  WARNING: >5mm/>2deg" if (dmm > 5.0 or ddeg > 2.0) else ""
            logger.info("[relocalize-all] %s chunk %d: drift %.1fmm/%.1fdeg%s",
                        obj.name, chunk_idx, dmm, ddeg, warn)
        except subprocess.CalledProcessError:
            logger.info("[relocalize-all] %s chunk %d: SAM3 found nothing "
                        "(occluded?) -> keeping last pose", obj.name, chunk_idx)
        except Exception as exc:  # noqa: BLE001 - relocalize is best-effort per object
            logger.warning("[relocalize-all] %s chunk %d: SKIPPED (%r); keeping last pose",
                           obj.name, chunk_idx, exc)
    return drifts


# --------------------------------------------------------------------------
# Whole-scene observation + twin hot-patch
# --------------------------------------------------------------------------
def _yaw_quat(yaw: float) -> list[float]:
    half = 0.5 * float(yaw)
    return [float(np.cos(half)), 0.0, 0.0, float(np.sin(half))]


def _yaw_from_quat(quat: Any) -> float:
    w, x, y, z = np.asarray(quat, dtype=float).reshape(4)
    return float(np.arctan2(2.0 * (w * z + x * y),
                            1.0 - 2.0 * (y * y + z * z)))


def _freejoint_velocity(plan_world: Any, qpos_addr: int) -> np.ndarray:
    """Read a free body's measured ``[linear, angular]`` velocity."""
    m, d = plan_world.model, plan_world.data
    jid = next((j for j in range(m.njnt)
                if int(m.jnt_qposadr[j]) == int(qpos_addr)), None)
    if jid is None:
        return np.zeros(6)
    dof = int(m.jnt_dofadr[jid])
    return np.asarray(d.qvel[dof:dof + 6], dtype=float).copy()


def _finite_difference_velocity(body: dict[str, Any],
                                previous: dict[str, Any] | None,
                                dt_s: float,
                                ) -> tuple[np.ndarray | None, np.ndarray | None]:
    if previous is None or dt_s <= 1e-6:
        return None, None
    p = np.asarray(body["pos_plan"], dtype=float)
    p0 = np.asarray(previous["pos_plan"], dtype=float)
    yaw = _yaw_from_quat(body["quat_wxyz"])
    yaw0 = _yaw_from_quat(previous["quat_wxyz"])
    dyaw = (yaw - yaw0 + np.pi) % (2.0 * np.pi) - np.pi
    return (p - p0) / dt_s, np.array([0.0, 0.0, dyaw / dt_s])


def _empty_capture_identity() -> dict[str, Any]:
    return {
        "capture_id": None,
        "capture_unix_s": None,
        "capture_monotonic_s": None,
        "camera_serial": None,
        "rgb_array_sha256": None,
        "depth_m_array_sha256": None,
        "calibration_sha256": None,
    }


def _synchronized_capture_identity(
    capture_dir: Path,
    objects: list[SceneObject],
) -> dict[str, Any]:
    """Read and cross-check the exposure identity shared by a scene packet.

    ``capture_id`` is the exact stream-session/frame-sequence binding used by
    post-stop checkpoints. Older packets without it remain readable, but
    callers that need terminal evidence must explicitly reject ``None``.
    """
    identities: list[dict[str, Any]] = []
    for obj in objects:
        manifest_path = (
            Path(capture_dir) / obj.name / "observation_manifest.json"
        )
        if not manifest_path.is_file():
            return _empty_capture_identity()
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw_stamp = manifest.get("capture_unix_s")
            if raw_stamp is None:
                return _empty_capture_identity()
            stamp = float(raw_stamp)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid camera capture timestamp in {manifest_path}"
            ) from exc
        if not np.isfinite(stamp) or stamp <= 0.0:
            raise ValueError(
                f"invalid camera capture timestamp in {manifest_path}"
            )
        monotonic_raw = manifest.get("capture_monotonic_s")
        monotonic = (
            None if monotonic_raw is None else float(monotonic_raw)
        )
        if monotonic is not None and (
            not np.isfinite(monotonic) or monotonic <= 0.0
        ):
            raise ValueError(
                f"invalid monotonic capture timestamp in {manifest_path}"
            )
        identities.append(
            {
                "capture_id": manifest.get("capture_id"),
                "capture_unix_s": stamp,
                "capture_monotonic_s": monotonic,
                "camera_serial": manifest.get("camera_serial"),
                "rgb_array_sha256": manifest.get("rgb_array_sha256"),
                "depth_m_array_sha256": manifest.get(
                    "depth_m_array_sha256"
                ),
                "calibration_sha256": manifest.get("calibration_sha256"),
            }
        )
    if not identities:
        return _empty_capture_identity()
    first = identities[0]
    for identity in identities[1:]:
        if abs(
            float(identity["capture_unix_s"])
            - float(first["capture_unix_s"])
        ) > 1e-6:
            raise ValueError(
                "one synchronized scene packet contains different camera "
                "capture timestamps"
            )
        for field in (
            "capture_id",
            "capture_monotonic_s",
            "camera_serial",
            "rgb_array_sha256",
            "depth_m_array_sha256",
            "calibration_sha256",
        ):
            if identity[field] != first[field]:
                raise ValueError(
                    "one synchronized scene packet contains different "
                    f"{field} values"
                )
    return dict(first)


def _synchronized_capture_timestamp_s(
    capture_dir: Path,
    objects: list[SceneObject],
) -> float | None:
    """Backward-compatible timestamp view of synchronized packet identity."""
    raw = _synchronized_capture_identity(
        capture_dir,
        objects,
    ).get("capture_unix_s")
    return None if raw is None else float(raw)


def observe_scene(
    cfg: LoopConfig,
    objects: list[SceneObject],
    subject: SceneObject | None,
    tag: str,
    *,
    purpose: str,
    plan_world: Any = None,
    fp_client: FoundationPoseClient | None = None,
    previous: dict[str, Any] | None = None,
    z_real_to_plan: float = 0.0,
    observed_subject: dict[str, Any] | None = None,
    body_names: set[str] | None = None,
    timestamp_s: float | None = None,
    subject_track_only: bool = False,
    camera_client: CameraStreamClient | None = None,
    capture_segmentation: bool = True,
    existing_capture_dir: Path | None = None,
    allow_same_frame_recovery: bool = False,
    require_mask_score: bool = False,
    signed_pose_authority_registry: SignedPoseAuthorityRegistry | None = None,
) -> dict[str, Any]:
    """Observe the program-facing scene from either MuJoCo or the real camera.

    The return is deliberately backend-neutral: every requested body has a
    plan-frame pose, confidence/visibility, and linear/angular velocity.  In
    simulation those velocities are the freejoint state; on camera they are a
    finite difference against ``previous``. ``purpose='replan'`` reads the
    current state of the persistent register-once -> track observer.

    ``body_names`` lets the caller limit expensive camera work to bodies the
    program references.  Omitting it returns every manifest body, which is the
    conservative default for synthesis and evidence capture.

    ``subject_track_only`` is the held/occluded-body policy.  The held subject
    is probed with the existing track.  Global register under gripper occlusion
    can hallucinate a pose; a weak/lost probe therefore remains an unknown
    measurement rather than being replaced by FK.
    """
    if purpose != "replan":
        raise ValueError(f"purpose must be 'replan', got {purpose!r}")
    now = float(time.time() if timestamp_s is None else timestamp_s)
    capture_timestamp_s = now if timestamp_s is not None else None
    old_t = float((previous or {}).get("timestamp_s", now))
    dt_s = max(0.0, now - old_t)
    old_bodies = dict((previous or {}).get("bodies", {}))
    wanted = ({o.name for o in objects} if body_names is None
              else {str(n) for n in body_names})
    if subject is not None:
        wanted.add(subject.name)
    bodies: dict[str, dict[str, Any]] = {}
    selected = [o for o in objects if o.name in wanted]
    # One RGB-D exposure for every live body. Sequential per-body capture gives
    # poses from different instants and cannot support velocity/contact/rest.
    synchronized_capture: Path | None = None
    capture_identity = _empty_capture_identity()
    capture_started_s: float | None = None
    if not cfg.offline and selected:
        if existing_capture_dir is None:
            capture_started_s = time.time()
            synchronized_capture = capture_scene_observation_packet(
                cfg,
                selected,
                tag,
                seg=bool(capture_segmentation),
                camera_client=camera_client,
            )
        else:
            synchronized_capture = Path(existing_capture_dir).resolve(
                strict=True
            )
            for obj in selected:
                validate_observation_packet(
                    synchronized_capture / obj.name,
                    require_mask=bool(require_mask_score),
                )
        capture_identity = _synchronized_capture_identity(
            synchronized_capture,
            selected,
        )
        if timestamp_s is None:
            packet_stamp = capture_identity["capture_unix_s"]
            capture_timestamp_s = packet_stamp
            now = (
                time.time()
                if packet_stamp is None
                else float(packet_stamp)
            )
        dt_s = max(0.0, now - old_t)

    for obj in objects:
        if obj.name not in wanted:
            continue
        raw: dict[str, Any]
        exact_velocity: np.ndarray | None = None
        exact_sim_state = False
        qaddr: int | None = None
        if plan_world is not None:
            if obj is subject:
                qaddr = int(plan_world.object_qpos_addr)
            elif obj.movable:
                qaddr = (getattr(plan_world, "movable_qpos_addr", {}) or {}).get(
                    obj.name)

        # A live free body is read straight from MuJoCo.  Static bodies still
        # come from the manifest/cache; a camera run observes them below.
        if cfg.offline and plan_world is not None and qaddr is not None:
            exact_sim_state = True
            pose = np.asarray(
                plan_world.data.qpos[int(qaddr):int(qaddr) + 7], dtype=float)
            raw = {
                "schema": "sim_scene_body_observation",
                "pos_base": pose[:3].tolist(),
                "quat_wxyz": pose[3:7].tolist(),
                "yaw_base_rad": _yaw_from_quat(pose[3:7]),
                "confidence": 1.0,
                "visibility": 1.0,
                "observation_id": f"{tag}:{obj.name}",
                "capture_id": f"sim:{tag}",
                "frame_id": f"qpos:{int(qaddr)}",
                "calibration_sha256": _SIM_CALIBRATION_SHA256,
            }
            exact_velocity = _freejoint_velocity(plan_world, int(qaddr))
        elif subject is not None and obj is subject and observed_subject is not None:
            raw = dict(observed_subject)
        elif not cfg.offline:
            held_probe = bool(subject_track_only
                              and subject is not None and obj is subject)
            raw = observe_subject(
                cfg, obj, f"{tag}_{obj.name}", plan_world=plan_world,
                fp_client=fp_client,
                track_only=held_probe,
                purpose=("replan" if held_probe else purpose),
                capture_dir=synchronized_capture,
                use_config_mesh_override=(
                    subject is not None and obj is subject),
                allow_same_frame_recovery=allow_same_frame_recovery,
                require_mask_score=require_mask_score)
        elif subject is not None and obj is subject:
            raw = observe_subject(
                cfg, obj, tag, plan_world=plan_world, fp_client=fp_client,
                purpose=purpose, capture_dir=synchronized_capture,
                use_config_mesh_override=True)
        else:
            if obj.pose_base is None:
                raise RuntimeError(
                    f"offline scene observation has no pose for {obj.name!r}")
            cached_base = np.asarray(obj.pose_base, dtype=float).copy()
            if obj.z_snapped_plan_frame:
                cached_base[2] -= float(z_real_to_plan)
            raw = {
                "schema": "manifest_cached_body_observation",
                "pos_base": cached_base.tolist(),
                "quat_wxyz": list(obj.pose_quat or _yaw_quat(0.0)),
                "confidence": 1.0,
                "visibility": 1.0,
            }

        pos_base = np.asarray(raw["pos_base"], dtype=float).reshape(3)
        raw_quat = raw.get("quat_wxyz")
        quat = list(
            raw_quat
            or _yaw_quat(float(raw.get("yaw_base_rad", 0.0)))
        )
        raw_schema = str(raw.get("schema", "unknown"))
        observation_id = str(
            raw.get("observation_id") or f"{tag}:{obj.name}"
        )
        capture_id_raw = raw.get(
            "capture_id",
            capture_identity.get("capture_id"),
        )
        capture_id = "" if capture_id_raw is None else str(capture_id_raw)
        frame_id_raw = raw.get(
            "frame_id",
            capture_identity.get("rgb_array_sha256"),
        )
        frame_id = "" if frame_id_raw is None else str(frame_id_raw)
        calibration_raw = raw.get(
            "calibration_sha256",
            capture_identity.get("calibration_sha256"),
        )
        calibration_sha256 = (
            "" if calibration_raw is None else str(calibration_raw)
        )
        authority_registry = (
            _sim_exact_registry(obj.name)
            if exact_sim_state
            else (
                None
                if raw_schema.startswith(("sim_", "offline_plan_world"))
                else signed_pose_authority_registry
            )
        )
        verified_claims = None
        verified_token = None
        if authority_registry is not None:
            authority_id = authority_registry.resolve_authority_id(
                body=obj.name,
                raw_schema=raw_schema,
                calibration_sha256=calibration_sha256,
            )
            if authority_id is not None:
                verified_claims = authority_registry.verify_claims(
                    authority_id=authority_id,
                    body=obj.name,
                    observation_id=observation_id,
                    capture_id=capture_id,
                    frame_id=frame_id,
                    raw_schema=raw_schema,
                    quaternion_wxyz=raw_quat,
                    calibration_sha256=calibration_sha256,
                )
                if verified_claims is not None:
                    verified_token = _mint_verified_signed_pose(
                        verified_claims
                    )
        signed_pose_observable = verified_token is not None
        # Sim observations are already in W_plan. Camera/manifest observations
        # are in robot-base and need the measured table offset.
        already_plan = str(raw.get("schema", "")).startswith(
            ("sim_", "offline_plan_world"))
        pos_plan = pos_base.copy()
        if not already_plan:
            pos_plan[2] += float(z_real_to_plan)
        confidence_raw = raw.get(
            "confidence", raw.get("fp_confidence", 1.0)
        )
        confidence = (
            None
            if confidence_raw is None
            else float(confidence_raw)
        )
        body = {
            "name": obj.name,
            "movable": bool(obj.movable),
            "source": raw_schema,
            "pos_base": pos_base.tolist(),
            "pos_plan": pos_plan.tolist(),
            "quat_wxyz": [float(v) for v in quat],
            "signed_pose_observable": signed_pose_observable,
            "observation_id": observation_id,
            "capture_id": capture_id or None,
            "frame_id": frame_id or None,
            "calibration_sha256": calibration_sha256 or None,
            "signed_pose_source_id": (
                None
                if verified_claims is None
                else verified_claims.authority_source_id
            ),
            "signed_pose_provenance": (
                None
                if verified_claims is None
                else verified_claims.authority_provenance
            ),
            "signed_pose_recording_id": (
                None
                if verified_claims is None
                else verified_claims.authority_recording_id
            ),
            "independent_validation_sha256": (
                None
                if verified_claims is None
                else verified_claims.independent_validation_sha256
            ),
            "signed_pose_verification": (
                None if verified_claims is None else verified_claims.to_dict()
            ),
            "_verified_signed_pose": verified_token,
            "yaw_base_rad": float(raw.get("yaw_base_rad",
                                                _yaw_from_quat(quat))),
            "confidence": confidence,
            "visibility": float(raw.get("visibility", 1.0)),
            "track_lost": bool(raw.get("fp_lost", False)),
            "track_only_probe": bool(raw.get("track_only_probe", False)),
            "mask_iou": raw.get("fp_mask_iou"),
            "mask_scored": raw.get("fp_mask_iou") is not None,
            "pointcloud_support": raw.get("fp_pc_support"),
        }
        if exact_velocity is None:
            lin, ang = _finite_difference_velocity(
                body, old_bodies.get(obj.name), dt_s)
        else:
            lin, ang = exact_velocity[:3], exact_velocity[3:]
        body["linear_velocity_mps"] = (
            None if lin is None else [float(v) for v in lin])
        body["angular_velocity_rps"] = (
            None if ang is None else [float(v) for v in ang])
        body["velocity_observable"] = lin is not None and ang is not None
        body["capture_timestamp_s"] = now
        bodies[obj.name] = body

    return {
        "schema": "oap_scene_observation_v1",
        "tag": tag,
        "purpose": purpose,
        "timestamp_s": now,
        "capture_timestamp_s": capture_timestamp_s,
        "capture_started_s": capture_started_s,
        **capture_identity,
        "synchronized_capture": bool(cfg.offline or synchronized_capture),
        "capture_dir": (None if synchronized_capture is None
                        else str(synchronized_capture)),
        "subject_track_only": bool(subject_track_only),
        "dt_s": dt_s,
        "frame": "W_plan",
        "world_axes": {
            "world_x": [1.0, 0.0, 0.0],
            "world_y": [0.0, 1.0, 0.0],
            "world_up": [0.0, 0.0, 1.0],
        },
        "bodies": bodies,
    }


def hot_patch_twin(
    scene_observation: dict[str, Any],
    plan_world: Any,
    subject: SceneObject,
    objects: list[SceneObject],
    *,
    table_support_z_plan: float | None = None,
    held_state_by_object: dict[str, bool | None] | None = None,
) -> dict[str, Any]:
    """Patch every observed body pose/velocity into the live planning twin.

    Free bodies are written through their freejoint; fixed supports are patched
    through the model body pose.  Robot qpos is intentionally untouched here
    (the real executor resynchronizes it from measured joints separately).
    """
    import mujoco

    body_obs = dict(scene_observation.get("bodies", {}))
    container_footprints = [
        (
            obj.name,
            tuple(body_obs[obj.name]["pos_plan"][:2]),
            0.5 * float(max(obj.size_lwh)),
        )
        for obj in objects
        if obj.open_top_container and obj.name in body_obs
    ]
    applied: dict[str, str] = {}
    for obj in objects:
        body = body_obs.get(obj.name)
        if body is None:
            continue
        pos = np.asarray(body["pos_plan"], dtype=float)
        quat = np.asarray(body["quat_wxyz"], dtype=float)
        quat /= max(float(np.linalg.norm(quat)), 1e-12)
        if (
            table_support_z_plan is not None
            and held_state_by_object is not None
            and obj.name in held_state_by_object
        ):
            if obj.q_om is None:
                raise ValueError(
                    f"{obj.name}: table support projection requires q_om"
                )
            projected, changed = project_unheld_pose_to_support(
                pos,
                quat,
                obj.size_lwh,
                obj.q_om,
                float(table_support_z_plan),
                held=held_state_by_object[obj.name],
                support_center_xy=LAB_TABLE_POS[:2],
                support_half_size_xy=LAB_TABLE_SIZE[:2],
                excluded_footprints=container_footprints,
            )
            if changed:
                delta_z = float(projected[2] - pos[2])
                body["raw_pos_plan"] = pos.tolist()
                body["pos_plan"] = projected.tolist()
                raw_base = np.asarray(body["pos_base"], dtype=float)
                body["raw_pos_base"] = raw_base.tolist()
                raw_base[2] += delta_z
                body["pos_base"] = raw_base.tolist()
                body["table_support_projection"] = {
                    "source": "global_table_support_lower_bound",
                    "raw_z_plan": float(pos[2]),
                    "projected_z_plan": float(projected[2]),
                }
                logger.info(
                    "[table-support] hot-patch %s z %.4f -> %.4f",
                    obj.name,
                    float(pos[2]),
                    float(projected[2]),
                )
                pos = projected
        if obj is subject:
            qaddr = int(plan_world.object_qpos_addr)
        else:
            qaddr = (getattr(plan_world, "movable_qpos_addr", {}) or {}).get(
                obj.name)
        if qaddr is not None:
            qaddr = int(qaddr)
            plan_world.data.qpos[qaddr:qaddr + 3] = pos
            plan_world.data.qpos[qaddr + 3:qaddr + 7] = quat
            jid = next((j for j in range(plan_world.model.njnt)
                        if int(plan_world.model.jnt_qposadr[j]) == qaddr), None)
            if jid is not None:
                dof = int(plan_world.model.jnt_dofadr[jid])
                linear = body.get("linear_velocity_mps")
                angular = body.get("angular_velocity_rps")
                if linear is not None and angular is not None:
                    velocity = np.r_[linear, angular].astype(float)
                    plan_world.data.qvel[dof:dof + 6] = velocity
            applied[obj.name] = "freejoint"
        else:
            patch_static_body_pose(
                plan_world, f"support_{obj.name}", pos, quat)
            applied[obj.name] = "fixed_body"
        # The grounding cache follows the planning twin, not the camera's
        # raw base-frame observation. Preserve that frame explicitly.
        obj.pose_base = [float(v) for v in pos]
        obj.z_snapped_plan_frame = True
        obj.pose_quat = [float(v) for v in quat]
    # Re-observation updates the host kinematic mirror only. All dynamics,
    # contacts, and task cost are evaluated by the following MJWarp batch.
    mujoco.mj_kinematics(plan_world.model, plan_world.data)
    return applied


def _freeze_initial_anchor(anchor: Anchor) -> Anchor:
    """Copy one observed anchor into an immutable episode-start datum."""
    return Anchor(
        point=anchor.point.copy(),
        axis=None if anchor.axis is None else anchor.axis.copy(),
        region_half=(
            None if anchor.region_half is None else anchor.region_half.copy()
        ),
        kind="initial",
        attached_to=None,
        confidence=anchor.confidence,
        visibility=anchor.visibility,
        age_since_seen=anchor.age_since_seen,
        last_pose=(
            None if anchor.last_pose is None else anchor.last_pose.copy()
        ),
        dynamic=False,
    )


def _freeze_initial_material_frame(
    anchors: AnchorSet,
    destination: str,
    source: str,
) -> None:
    """Freeze one complete directed material triad, or freeze none of it.

    A partial triad is not a signed 6-D pose.  ``RelativeOrientation`` then
    stays ungroundable/UNKNOWN instead of silently falling back to yaw or an
    unsigned principal axis.
    """
    source_names = [f"{source}_frame_{axis}" for axis in "xyz"]
    if any(name not in anchors for name in source_names):
        return
    for axis, name in zip("xyz", source_names, strict=True):
        anchors.add(
            f"{destination}_initial_frame_{axis}",
            _freeze_initial_anchor(anchors[name]),
        )


def ground_observation(
    scene_observation: dict[str, Any],
    subject: SceneObject | None,
    objects: list[SceneObject],
    *,
    table_top_z: float,
    reference: SceneObject | None = None,
    initial_anchors: AnchorSet | None = None,
) -> AnchorSet:
    """Ground a whole-scene observation while retaining episode-zero frames.

    Current body anchors move on every re-observation.  ``*_initial_center``,
    ``*_initial_axis`` and ``*_initial_normal`` are copied from the first
    grounding and thereafter remain immutable, so a synthesized program can
    state "move this body 8 cm along world_x from where the episode began"
    without embedding a lab-specific absolute coordinate.
    """
    body_obs = dict(scene_observation.get("bodies", {}))
    if subject is not None and subject.name not in body_obs:
        raise ValueError(
            f"scene observation has no subject body {subject.name!r}")
    if subject is None:
        # Pre-bind grounding: every body keeps only its manifest name. This is
        # sufficient to annotate a live scene and synthesize before the
        # program's explicit manifest anchors bind controlled and goal roles.
        aset = AnchorSet({
            "world_x": Anchor(np.zeros(3), axis=np.array([1.0, 0.0, 0.0]),
                              kind="world"),
            "world_y": Anchor(np.zeros(3), axis=np.array([0.0, 1.0, 0.0]),
                              kind="world"),
            "world_up": Anchor(np.zeros(3), axis=np.array([0.0, 0.0, 1.0]),
                               kind="world"),
        })
        table_xy = np.zeros(2)
        for obj in objects:
            body = body_obs.get(obj.name)
            if body is None:
                continue
            center = np.asarray(body["pos_plan"], dtype=float)
            table_xy = center[:2]
            quat = body["quat_wxyz"]
            L, W, H = obj.size_lwh
            dyn = bool(obj.movable)
            confidence_raw = body.get("confidence", 1.0)
            # Anchor stores require a number. Zero represents an explicitly
            # unscored maskless track, not fabricated positive quality.
            conf = (
                0.0 if confidence_raw is None else float(confidence_raw)
            )
            vis = float(body.get("visibility", 1.0))
            aset.add(obj.name, Anchor(
                point=center.copy(), dynamic=dyn, kind="region",
                attached_to=obj.name if dyn else None,
                confidence=conf, visibility=vis,
                region_half=np.array([L / 2.0, W / 2.0, max(H / 2.0, 0.5)])))
            aset.add(f"{obj.name}_center", Anchor(
                point=center.copy(), dynamic=dyn, kind="object",
                attached_to=obj.name if dyn else None,
                confidence=conf, visibility=vis))
            for name, anchor in _body_anchors(
                obj.name, center,
                _canonical_frame(quat, getattr(obj, "q_om", None)),
                obj.size_lwh,
                attached_to=obj.name if dyn else None,
                include_material_frame=signed_pose_frames_authorized(
                    body,
                    expected_body=obj.name,
                ),
            ).items():
                anchor.dynamic = dyn
                anchor.confidence = conf
                anchor.visibility = vis
                aset.add(name, anchor)
        aset.add("table", Anchor(
            point=np.array([table_xy[0], table_xy[1], table_top_z]),
            axis=np.array([0.0, 0.0, 1.0]), kind="world"))
        if initial_anchors is not None:
            initial_copy = initial_anchors.copy()
            for name in initial_copy.names():
                if "_initial_" in name:
                    aset.add(name, initial_copy[name])
            return aset
        snapshot = aset.copy()
        for obj in objects:
            center_name = f"{obj.name}_center"
            if center_name in snapshot:
                aset.add(
                    f"{obj.name}_initial_center",
                    _freeze_initial_anchor(snapshot[center_name]),
                )
            for suffix in ("axis", "normal", "frame_x", "frame_y", "frame_z"):
                source_name = f"{obj.name}_{suffix}"
                if source_name in snapshot:
                    aset.add(
                        f"{obj.name}_initial_{suffix}",
                        _freeze_initial_anchor(snapshot[source_name]),
                    )
        return aset

    local_objects: list[SceneObject] = []
    local_subject: SceneObject | None = None
    for obj in objects:
        local = copy.copy(obj)
        body = body_obs.get(obj.name)
        if body is not None:
            local.pose_base = [float(v) for v in body["pos_plan"]]
            local.pose_quat = [float(v) for v in body["quat_wxyz"]]
            local.signed_pose_observable = signed_pose_frames_authorized(
                body,
                expected_body=obj.name,
            )
            local.verified_signed_pose_row = body
        local_objects.append(local)
        if obj is subject:
            local_subject = local
    assert local_subject is not None
    subject_obs = dict(body_obs[subject.name])
    subject_obs["pos_base"] = list(subject_obs["pos_plan"])
    aset = anchors_from_observation(
        subject_obs, local_subject, local_objects, table_top_z=table_top_z,
        reference=(next((o for o in local_objects
                         if reference is not None and o.name == reference.name),
                        None)))

    # Carry tracker confidence/visibility onto every anchor owned by that body.
    ownership: list[tuple[str, str]] = [("subject", subject.name)]
    ownership.extend((o.name, o.name) for o in objects if o is not subject)
    if reference is not None:
        ownership.append(("target", reference.name))
    for prefix, body_name in ownership:
        body = body_obs.get(body_name)
        if body is None:
            continue
        for name in aset.names():
            if name == prefix or name.startswith(prefix + "_"):
                confidence_raw = body.get("confidence", 1.0)
                aset[name].confidence = (
                    0.0 if confidence_raw is None else float(confidence_raw)
                )
                aset[name].visibility = float(body.get("visibility", 1.0))

    if initial_anchors is not None:
        initial_copy = initial_anchors.copy()
        for name in initial_copy.names():
            if "_initial_" in name:
                aset.add(name, initial_copy[name])
        return aset

    # First sight: freeze canonical, manifest-name, and target aliases.
    snapshot = aset.copy()
    aliases: list[tuple[str, str]] = [("subject", "subject"),
                                      (subject.name, "subject")]
    aliases.extend((o.name, o.name) for o in objects if o is not subject)
    if reference is not None:
        aliases.append(("target", "target"))
    for destination, source in aliases:
        center_name = (f"{source}_center"
                       if f"{source}_center" in snapshot else source)
        if center_name in snapshot:
            aset.add(
                f"{destination}_initial_center",
                _freeze_initial_anchor(snapshot[center_name]),
            )
        for suffix in ("axis", "normal", "frame_x", "frame_y", "frame_z"):
            source_name = f"{source}_{suffix}"
            if source_name in snapshot:
                aset.add(
                    f"{destination}_initial_{suffix}",
                    _freeze_initial_anchor(snapshot[source_name]),
                )
    return aset


def ground_offline_plan_world(
    cfg: LoopConfig,
    objects: list[SceneObject],
    subject: SceneObject,
    plan_world: Any,
    *,
    tag: str,
    table_top_z: float,
    reference: SceneObject | None = None,
    initial_anchors: AnchorSet | None = None,
    z_real_to_plan: float = 0.0,
) -> tuple[dict[str, Any], AnchorSet]:
    """Ground the exact initialized simulation state used by the planner.

    A configured offline observation is an input to scene construction, not
    necessarily the pose that the released model finally stores: the twin
    composes its yaw with the manifest's mesh-to-object axis map and may apply
    support projection.  Signed material-frame predicates must therefore bind
    to the post-build free-joint quaternion, not to the reconstruction packet.

    This helper keeps that producer boundary explicit and reusable by both the
    runner and CPU-only experiment preflight.  :func:`observe_scene` mints its
    process-local exact-simulation token from ``plan_world.data.qpos``; ordinary
    serialized packets and self-asserted quaternions remain unauthorized.
    """
    if not bool(getattr(cfg, "offline", False)):
        raise ValueError("exact plan-world grounding requires offline mode")
    if plan_world is None:
        raise ValueError("exact plan-world grounding requires a built world")
    scene = observe_scene(
        cfg,
        objects,
        subject,
        tag,
        purpose="replan",
        plan_world=plan_world,
        z_real_to_plan=z_real_to_plan,
    )
    anchors = ground_observation(
        scene,
        subject,
        objects,
        table_top_z=table_top_z,
        reference=reference,
        initial_anchors=initial_anchors,
    )
    # Raw body frames serve signed relative-rotation specifications. They are
    # distinct from q_om-corrected canonical geometry axes and are exposed
    # only by this exact simulation producer, never by an untrusted packet.
    for body_name, row in scene.get("bodies", {}).items():
        if not signed_pose_frames_authorized(row, expected_body=body_name):
            continue
        rotation = quat_to_mat(np.asarray(row["quat_wxyz"], dtype=float))
        aliases = [body_name]
        if body_name == subject.name:
            aliases.append("subject")
        if reference is not None and body_name == reference.name:
            aliases.append("target")
        for alias in dict.fromkeys(aliases):
            base = anchors.get(alias)
            if base is None:
                continue
            owner = base.attached_to
            if owner is None and base.kind == "subject":
                owner = alias
            for i, letter in enumerate("xyz"):
                current_name = f"{alias}_raw_frame_{letter}"
                initial_name = f"{alias}_initial_raw_frame_{letter}"
                anchors.add(current_name, Anchor(
                    point=base.point.copy(), axis=rotation[:, i].copy(),
                    kind="frame_axis", attached_to=owner, dynamic=base.dynamic,
                    confidence=base.confidence, visibility=base.visibility,
                ))
                if initial_anchors is not None and initial_name in initial_anchors:
                    anchors.add(initial_name, initial_anchors.copy()[initial_name])
                else:
                    # Provisional planning reference. run_mpc_episode binds
                    # it once to the first recorded measured endpoint.
                    anchors.add(initial_name, _freeze_initial_anchor(anchors[current_name]))
    return scene, anchors


# --------------------------------------------------------------------------
# Observation -> AnchorSet (the perception -> grounding seam)
# --------------------------------------------------------------------------
def _canonical_frame(quat_wxyz: Any, q_om: Any) -> np.ndarray:
    """The body's CANONICAL object frame in world, as a rotation matrix.

    Both the measured subject quaternion and a manifest object's ``pose_quat``
    orient the MESH frame in world (the twin sets a static body's quat from
    pose_quat verbatim, and the subject's from ``object_body_quat(yaw, q_om)``).
    ``q_om`` is the mesh->object axis map, so composing it out recovers the
    frame the reconstruction's L/W/H are expressed in -- the one an anchor axis
    must mean. Omitting it grades the wrong axis (2026-07-24 audit).
    """
    from oap.program.geometry import quat_to_matrix

    R = quat_to_matrix(np.asarray(quat_wxyz, dtype=float))
    if q_om is None:
        return R
    return R @ quat_to_matrix(np.asarray(q_om, dtype=float)).T


def _body_anchors(
    name: str,
    point: np.ndarray,
    R_ow: np.ndarray,
    size_lwh: Any,
    *,
    attached_to: str | None = None,
    floor_name: str | None = None,
    include_material_frame: bool = True,
) -> dict[str, Anchor]:
    """The UNIFORM anchor vocabulary every body gets -- subject and scene alike.

    One rule for every object, so what a program can say about the subject it
    can also say about a container, a shelf or a tool: the canonical long axis
    and face normal (which is what un-deads ``axis_parallel``,
    ``interaction_align``, ``point_on_line`` and ``along_axis_gap`` against
    passive objects -- their axes were loaded at manifest time and thrown
    away), and the two face-centre points along world up (``_top`` is the datum
    a stack needs: with only a centre point, a physically correct stack graded
    FAIL while interpenetration graded SUCCESS -- measured, and now inverted;
    see below for why the two face datums use DIFFERENT formulas).
    Regions are NOT emitted here: every body already carries one under its bare
    name with the planar (z-inflated) convention, and a second region with
    different semantics under a second name is a trap.
    ``attached_to`` records the physical owner so every alias/keypoint of one
    movable body can be grouped and carried under that body's candidate pose.
    """
    from oap.program.geometry import unit

    half = 0.5 * np.asarray(size_lwh, dtype=float)
    up = np.array([0.0, 0.0, 1.0])
    frame_x = unit(R_ow @ np.array([1.0, 0.0, 0.0]))
    frame_y = unit(R_ow @ np.array([0.0, 1.0, 0.0]))
    frame_z = unit(R_ow @ np.array([0.0, 0.0, 1.0]))
    n_up = frame_z.copy()          # canonical up-face normal
    # SIGN-CORRECT it. The canonical frame comes from the reconstruction's
    # mesh->object axis map, and for a near-symmetric body that map is 180-degree
    # AMBIGUOUS: measured on the 2026-07-27 tool-use bundle, the box's canonical
    # z came back at n_up.z = -0.999, putting `_top` 98 mm BELOW the centre --
    # at the body's own floor. So "place this on top of the box" aimed
    # underneath it and `_normal` pointed at the table. Which face is the TOP is
    # a fact about the world, not about which way the mesh's third axis happens
    # to run. A tilted body still keeps its own normal (that is the whole point
    # of using it instead of world up); only the hemisphere is pinned.
    if float(n_up[2]) < 0.0:
        n_up = -n_up
    # The two datums answer DIFFERENT questions and must not share a formula:
    #  _top  = the top FACE CENTRE, a material point of the body (centre + half
    #          height ALONG ITS OWN normal). This is the placement datum: put
    #          something ON this.
    #  _floor = the SUPPORT PLANE along world up (centre - the OBB's vertical
    #          support half extent). This is the resting-CONTACT datum, and it
    #          matches how the twin seats a body (manifest table-snap sets
    #          pose_base.z = TABLE_TOP_Z + this half extent).
    # Using the support half extent for BOTH -- which this function briefly did
    # -- puts _top strictly OUTSIDE a tilted body (75 mm out at 30 deg) and
    # re-creates the interpenetration inversion on any tilted reference: a
    # MuJoCo-settled stack graded FAIL from 8 deg while a 25 mm hover graded
    # SUCCESS (adversarial review with a physics oracle, 2026-07-26).
    vhe = float(np.sum(np.abs(R_ow[2, :]) * half))
    p = np.asarray(point, dtype=float)
    kw = {"attached_to": attached_to} if attached_to else {}
    x_hat = frame_x
    # Which material axis can the fingers close along? They span the body's
    # extent along that axis, so the graspable one is the narrowest extent
    # that fits the jaw. This is object geometry plus a robot constant --
    # derived by the same fixed rule for every body, carrying no task
    # semantics, and defined for objects the system has never seen. Without
    # it a program can only reference <id>_axis (the LONGEST extent) and
    # measurably asks the fingers to span the wrong dimension.
    _APERTURE_M = 0.085
    _ext = 2.0 * half                      # full extent along each material axis
    _frames = (frame_x, frame_y, frame_z)
    # A body at rest has its underside against its support, so the fingers
    # cannot close along a near-vertical axis however thin the body is there:
    # only the horizontal extents are reachable. Among those, the graspable
    # one is the narrowest that fits the jaw; if none fits, the narrowest
    # horizontal extent is still published and object_held will simply not be
    # achievable, which is the truth about that body.
    _horiz = [i for i in range(3) if abs(float(np.dot(_frames[i], up))) < 0.7]
    _cand = _horiz or list(range(3))
    _fits = [i for i in _cand if float(_ext[i]) < _APERTURE_M]
    _gi = min(_fits or _cand, key=lambda i: float(_ext[i]))
    anchors = {
        f"{name}_grasp_axis": Anchor(point=p.copy(), axis=_frames[_gi].copy(),
                                     kind="axis", **kw),
        f"{name}_axis": Anchor(point=p.copy(), axis=x_hat.copy(),
                               kind="axis", **kw),
        f"{name}_normal": Anchor(point=p.copy(), axis=n_up.copy(), kind="axis", **kw),
        f"{name}_top": Anchor(point=p + half[2] * n_up, axis=n_up.copy(), kind="part", **kw),
        (floor_name or f"{name}_floor"): Anchor(point=p - vhe * up, axis=up.copy(),
                                                kind="part", **kw),
    }
    # Signed grasp points remain attached to fixed material-X locations while
    # the body rotates. Each point is inset 20 mm from its physical end.
    end_inset = min(0.020, 0.5 * float(half[0]))
    grasp_radius = max(0.0, float(half[0]) - end_inset)
    anchors[f"{name}_material_minus_x_grasp"] = Anchor(
        point=p - grasp_radius * frame_x,
        kind="part",
        **kw,
    )
    anchors[f"{name}_material_plus_x_grasp"] = Anchor(
        point=p + grasp_radius * frame_x,
        kind="part",
        **kw,
    )
    if include_material_frame:
        # Unlike ``_normal``, these are raw directed MATERIAL axes.  Never
        # hemisphere-pin them: a 180-degree flip must invert frame_y/frame_z
        # instead of being rewritten to look upright.
        for axis, value in zip(
            "xyz",
            (frame_x, frame_y, frame_z),
            strict=True,
        ):
            anchors[f"{name}_frame_{axis}"] = Anchor(
                point=p.copy(),
                axis=value.copy(),
                kind="frame_axis",
                **kw,
            )
    # The four MATERIAL bottom-face corners (centre - half-height along the
    # body's own normal, +/- the in-plane half extents). Unlike `_floor`
    # (a world-up support datum, tilt-blind by design), these are body points:
    # a tilted body lifts a corner, so full-footprint/support predicates read
    # tilt directly from them.
    y_hat = unit(np.cross(n_up, x_hat))
    bottom = p - half[2] * n_up
    for index, (su, sv) in enumerate(
        ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))
    ):
        anchors[f"{name}_bottom_corner_{index}"] = Anchor(
            point=bottom + su * half[0] * x_hat + sv * half[1] * y_hat,
            kind="part", attached_to=attached_to,
        )
    return anchors


def anchors_from_scene_manifest(objects: list[SceneObject], *,
                                table_top_z: float = TABLE_TOP_Z) -> AnchorSet:
    """Ground every localized manifest body before a controlled subject is known.

    This task-independent vocabulary is what the synthesis VLM sees. In
    particular, each body exposes a fixed ``<name>_initial_center`` alongside
    its movable current anchors, so a program can state a displacement relative
    to the observed episode start. No ``subject``/``target`` alias is guessed;
    :mod:`oap.program.bindings` resolves those roles from the emitted Stage.
    """
    aset = AnchorSet({
        "table": Anchor(point=np.array([0.0, 0.0, float(table_top_z)]),
                        axis=np.array([0.0, 0.0, 1.0]), kind="world"),
        "world_x": Anchor(point=np.zeros(3), axis=np.array([1.0, 0.0, 0.0]),
                          kind="world"),
        "world_y": Anchor(point=np.zeros(3), axis=np.array([0.0, 1.0, 0.0]),
                          kind="world"),
        "world_up": Anchor(point=np.zeros(3), axis=np.array([0.0, 0.0, 1.0]),
                           kind="world"),
    })
    for obj in objects:
        if obj.pose_base is None:
            continue
        center = np.asarray(obj.pose_base, dtype=float)
        L, W, H = obj.size_lwh
        dynamic = bool(obj.movable)
        aset.add(obj.name, Anchor(
            point=center.copy(), kind="region", dynamic=dynamic,
            attached_to=obj.name if dynamic else None,
            region_half=np.array([L / 2.0, W / 2.0, max(H / 2.0, 0.5)])))
        aset.add(f"{obj.name}_center", Anchor(point=center.copy(), kind="object",
                                               dynamic=dynamic,
                                               attached_to=(
                                                   obj.name if dynamic else None
                                               )))
        aset.add(f"{obj.name}_initial_center",
                 Anchor(point=center.copy(), kind="initial"))
        pose_quat = getattr(obj, "pose_quat", None)
        R = _canonical_frame(
            pose_quat or [1.0, 0.0, 0.0, 0.0],
            getattr(obj, "q_om", None),
        )
        for name, anchor in _body_anchors(
            obj.name,
            center,
            R,
            obj.size_lwh,
            attached_to=obj.name if dynamic else None,
            include_material_frame=False,
        ).items():
            anchor.dynamic = dynamic
            aset.add(name, anchor)
        _freeze_initial_material_frame(aset, obj.name, obj.name)
    return aset


def anchors_from_observation(observed: dict[str, Any], subject: SceneObject,
                             objects: list[SceneObject], *, table_top_z: float,
                             reference: SceneObject | None = None) -> AnchorSet:
    """Ground an observation + scene manifest into the program's AnchorSet.

    This is the perception->grounding seam the synthesized TaskProgram
    references and that ``verify()`` re-grounds against. ``subject`` = the
    moving object (point = observed base pose, in-plane long axis from yaw);
    each STATIC manifest object -> a region anchor (region_half from its
    footprint, tall in z so in_region is effectively planar) + a center point
    anchor; plus ``table`` and ``world_up``. confidence / visibility = 1.0 for
    a live observation.

    This is an OBJECT-ONLY observation: deliberately NO ``gripper_width`` or
    ``gripper_command`` anchor. The gripper is robot state, not scene
    perception. Running ``GripperCommand`` is read directly from each sampled
    control trajectory on GPU; terminal ``GripperState`` is grounded from the
    calibrated robot-state observation used by the stage/verdict adapter.
    Neither actuator intent nor jaw state is inferred from an object image.

    ``reference`` (the task's second/goal object) is ALSO exposed under the
    synthesizer's CANONICAL names ``target`` (its center point) and
    ``target_region`` (its footprint region). Without a reference, no
    ``target`` alias exists, so a task that needs one HONESTLY refuses
    (groundability).
    """
    pos = np.asarray(observed["pos_base"], dtype=float)
    yaw = float(observed.get("yaw_base_rad", 0.0))
    quat = observed.get("quat_wxyz")
    signed_pose_observable = signed_pose_frames_authorized(
        observed,
        expected_body=subject.name,
    )
    if quat is not None:
        # Orientation-COMPLETE grounding (adversarial review, 2026-07-24): the
        # long axis is the body x-axis rotated by the MEASURED quaternion.
        # R(q) @ x reduces to [cos yaw, sin yaw, 0] for a pure-yaw pose, so
        # this strictly extends the legacy grounding -- but for a tilted pose
        # the yaw-only axis stays in-plane and every AxisAngle-vs-world_up
        # terminal (a pour's tilt) reads a constant 90 deg: blind to the one
        # thing the stage varies. The scorer's predicted_end_anchors had this
        # exact bug and was fixed the same way (rotate the subject axis).
        # q_om (mesh->object axis map) MUST be composed out: the measured
        # quaternion orients the MESH frame in world (the twin builds the body
        # with object_body_quat(yaw, q_om)), so R(quat) @ x gives the mesh
        # x-axis, not the object's canonical long axis. Reconstructed meshes
        # routinely carry a permuted q_om, and every other consumer composes it
        # (object_body_quat, object_vertical_half_extent, the collision quat) --
        # omitting it here silently graded the WRONG axis (generality audit,
        # 2026-07-24). At rest with pure yaw this reduces EXACTLY to the legacy
        # [cos yaw, sin yaw, 0]: R(q_yaw (x) q_om) @ R(q_om)^T @ x = R(q_yaw) @ x.
        from oap.program.geometry import quat_to_matrix, unit
        R_mesh_world = quat_to_matrix(np.asarray(quat, dtype=float))
        q_om = getattr(subject, "q_om", None)
        R_obj_world = (R_mesh_world @ quat_to_matrix(np.asarray(q_om, dtype=float)).T
                       if q_om is not None else R_mesh_world)
        long_axis = unit(R_obj_world @ np.array([1.0, 0.0, 0.0]))
    else:
        long_axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    aset = AnchorSet({
        "subject": Anchor(point=pos.copy(), axis=long_axis, kind="subject"),
        "subject_axis": Anchor(point=pos.copy(), axis=long_axis, attached_to="subject"),
        "table": Anchor(point=np.array([pos[0], pos[1], table_top_z]),
                        axis=np.array([0.0, 0.0, 1.0]), kind="world"),
        "world_x": Anchor(point=np.zeros(3), axis=np.array([1.0, 0.0, 0.0]),
                          kind="world"),
        "world_y": Anchor(point=np.zeros(3), axis=np.array([0.0, 1.0, 0.0]),
                          kind="world"),
        "world_up": Anchor(point=np.zeros(3), axis=np.array([0.0, 0.0, 1.0]), kind="world"),
    })
    # The subject gets the SAME vocabulary every other body gets (one rule, no
    # exceptions). Its floor point keeps the established name ``subject_base``
    # -- containment tests "seated on the container floor" against it, so a tall
    # object standing in a shallow tray still certifies. ``_axis`` is re-set to
    # the measured long axis above (identical value; kept for the alias).
    R_sub = _canonical_frame(quat, getattr(subject, "q_om", None)) if quat is not None else np.array(
        [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    for k, v in _body_anchors(
        "subject",
        pos,
        R_sub,
        subject.size_lwh,
        attached_to="subject",
        floor_name="subject_base",
        include_material_frame=signed_pose_observable,
    ).items():
        if k != "subject_axis":                       # already grounded above
            aset.add(k, v)
    _freeze_initial_material_frame(aset, "subject", "subject")
    _sL, _sW, _sH = subject.size_lwh
    aset.add("subject_region",                        # same convention as every body
             Anchor(point=pos.copy(), attached_to="subject", kind="region",
                    region_half=np.array([_sL / 2.0, _sW / 2.0, max(_sH / 2.0, 0.5)])))
    # Also expose the controlled body under its manifest id. The pre-observation
    # synthesis vocabulary has no guessed ``subject`` alias, so a multi-movable
    # program names this id directly in its predicates. These aliases ride the
    # canonical subject in candidate predictions; the fixed initial-centre
    # anchor does not.
    if subject.name != "subject":
        aset.add(subject.name, Anchor(
            point=pos.copy(), attached_to="subject", kind="region",
            region_half=np.array([_sL / 2.0, _sW / 2.0, max(_sH / 2.0, 0.5)])))
        aset.add(f"{subject.name}_center",
                 Anchor(point=pos.copy(), attached_to="subject", kind="object"))
        for name, anchor in _body_anchors(
                subject.name, pos, R_sub, subject.size_lwh,
                attached_to="subject",
                include_material_frame=signed_pose_observable).items():
            aset.add(name, anchor)
        _freeze_initial_material_frame(aset, subject.name, subject.name)
    aset.add("subject_initial_center", Anchor(point=pos.copy(), kind="initial"))
    aset.add(f"{subject.name}_initial_center", Anchor(point=pos.copy(), kind="initial"))
    for obj in objects:
        if obj is subject or obj.pose_base is None:
            continue
        c = np.asarray(obj.pose_base, dtype=float)
        L, W, H = obj.size_lwh
        # Whether the physics can move this body is a MANIFEST fact
        # (SceneObject.movable), so it is known here -- before any world exists.
        # The pre-motion inert-success gate runs on exactly this anchor set, and
        # marking the flag only in the live-sim regrounding meant the gate saw a
        # pushable box as immovable and refused every push program as inert:
        # measured, "CANNOT_VERIFY (ungroundable:inert_success:push_box_forward)"
        # on a program whose residual demonstrably responds to the box moving.
        dyn = bool(getattr(obj, "movable", False))
        aset.add(obj.name, Anchor(point=c.copy(), dynamic=dyn,
                                  attached_to=obj.name if dyn else None,
                                  region_half=np.array([L / 2.0, W / 2.0, max(H / 2.0, 0.5)]),
                                  kind="region"))
        aset.add(f"{obj.name}_initial_center",
                 Anchor(point=c.copy(), kind="initial"))
        aset.add(f"{obj.name}_center", Anchor(
            point=c.copy(),
            kind="object",
            dynamic=dyn,
            attached_to=obj.name if dyn else None,
        ))
        # Passive bodies get the uniform vocabulary too: their orientation was
        # loaded at manifest time (pose_quat) and thrown away here, which is why
        # every axis predicate degenerated to "versus world up" and a stack had
        # no top-surface datum.
        pq = getattr(obj, "pose_quat", None)
        if pq is not None:
            for k, v in _body_anchors(obj.name, c,
                                      _canonical_frame(pq, getattr(obj, "q_om", None)),
                                      obj.size_lwh,
                                      attached_to=(
                                          obj.name if dyn else None
                                      ),
                                      include_material_frame=(
                                          signed_pose_frames_authorized(
                                              getattr(
                                                  obj,
                                                  "verified_signed_pose_row",
                                                  None,
                                              ),
                                              expected_body=obj.name,
                                          )
                                      )).items():
                v.dynamic = dyn
                aset.add(k, v)
            _freeze_initial_material_frame(aset, obj.name, obj.name)
    # Canonical aliases for the goal object so the synthesized program grounds.
    if reference is not None and reference is not subject and reference.pose_base is not None:
        rc = np.asarray(reference.pose_base, dtype=float)
        rL, rW, rH = reference.size_lwh
        # Aliases are another vocabulary for the SAME physical body, so they
        # must preserve whether that body has a freejoint. Dropping ``dynamic``
        # here made a program that called a pushable box ``target`` look static
        # to the inert-objective and tool-mediation gates, while the identical
        # program using its manifest name worked.
        ref_dyn = bool(getattr(reference, "movable", False))
        aset.add("target", Anchor(point=rc.copy(), kind="object",
                                  dynamic=ref_dyn,
                                  attached_to=reference.name if ref_dyn else None))
        aset.add("target_initial_center", Anchor(point=rc.copy(), kind="initial"))
        aset.add("target_region", Anchor(point=rc.copy(),
                                         region_half=np.array([rL / 2.0, rW / 2.0, max(rH / 2.0, 0.5)]),
                                         kind="region", dynamic=ref_dyn,
                                         attached_to=reference.name if ref_dyn else None))
        # Open-top container: the FLOOR-plane point (container center dropped by
        # its half-height), re-measured each observation. Containment = the
        # subject's base seated on this floor (AboveBy(subject_base, target_floor,
        # 0)) AND inside the footprint (InRegion vs target_region). No upper
        # bound: the top is open, so a tall object may protrude. Emitted ONLY for
        # open-top references, so every non-container task stays byte-identical.
        if getattr(reference, "open_top_container", False):
            aset.add("target_floor", Anchor(point=np.array([rc[0], rc[1], rc[2] - rH / 2.0]),
                                            axis=np.array([0.0, 0.0, 1.0]),
                                            kind="region", dynamic=ref_dyn,
                                            attached_to=reference.name if ref_dyn else None))
        # the goal object's share of the uniform vocabulary, under the canonical
        # alias -- so "stack ON it" has a top-surface datum by the same rule.
        # ``target_floor`` is DELIBERATELY excluded: that name belongs to the
        # open-top-container contract above (its presence is also what selects
        # the seated-containment program), so emitting it for every reference
        # would turn "place ON the box" into "place INSIDE the box".
        reference_quat = getattr(reference, "pose_quat", None)
        for _k, _v in _body_anchors(
            "target",
            rc,
            _canonical_frame(
                reference_quat or [1.0, 0.0, 0.0, 0.0],
                getattr(reference, "q_om", None),
            ),
            reference.size_lwh,
            attached_to=reference.name if ref_dyn else None,
            include_material_frame=signed_pose_frames_authorized(
                getattr(reference, "verified_signed_pose_row", None),
                expected_body=reference.name,
            ),
        ).items():
            if _k != "target_floor":
                _v.dynamic = ref_dyn
                aset.add(_k, _v)
        _freeze_initial_material_frame(aset, "target", "target")
    return aset
