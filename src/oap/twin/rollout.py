"""Physics rollout machinery for candidate action chunks in the W_plan twin.

Role in the two-stage pipeline: every candidate chunk the sampler proposes is
rolled out here in the twin under strict physical gates (split penetration
semantics, end-state grip evidence, per-chunk metrics including the final
object quaternion for the sim<->real yaw residual). The loop keeps the twin
synchronized with reality between chunks via :func:`reset_object_pose`,
:func:`reset_robot_qpos` (per-chunk robot-qpos resync) and
:func:`patch_static_body_pose` (reference relocalize).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from oap.twin.scene import observe_object_pose
from oap.twin.types import SimWorld

logger = logging.getLogger("oap.twin.rollout")

__all__ = [
    "clone_data", "observe_object_pose", "reset_object_pose",
    "reset_robot_qpos", "patch_static_body_pose", "tcp_pose_T",
]

try:  # canonical video helpers live in oap.utils.video
    from oap.utils.video import annotate_single, write_mp4
except ImportError:  # pragma: no cover - minimal fallbacks, same signatures
    def annotate_single(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
        """Annotate a rendered frame with a title/subtitle banner."""
        from PIL import Image, ImageDraw, ImageFont

        canvas = Image.fromarray(image).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        draw.rectangle((0, 0, canvas.width, 48), fill=(244, 246, 242))
        draw.text((8, 8), title, fill=(10, 10, 10), font=font)
        draw.text((8, 28), subtitle, fill=(35, 35, 35), font=font)
        return np.asarray(canvas)

    def write_mp4(frame_paths: list[Path], video_path: Path, *, fps: float) -> None:
        """Write a list of frame PNGs to an mp4 (imageio/ffmpeg)."""
        import imageio.v2 as iio

        with iio.get_writer(str(video_path), fps=float(fps)) as writer:
            for p in frame_paths:
                # imageio's v2 legacy API annotates the writer as the base
                # class, which lacks append_data; the runtime object is a
                # Format.Writer.
                writer.append_data(iio.imread(str(p)))  # type: ignore[attr-defined]
        if not Path(video_path).exists() or Path(video_path).stat().st_size == 0:
            raise RuntimeError(f"Failed to write video: {video_path}")


def clone_data(model: Any, src: Any) -> Any:
    """CPU diagnostic utility; not used by the GPU-only production path."""
    dst = mujoco.MjData(model)
    dst.qpos[:] = src.qpos
    dst.qvel[:] = src.qvel
    dst.act[:] = src.act
    dst.ctrl[:] = src.ctrl
    if dst.mocap_pos.shape == src.mocap_pos.shape:
        dst.mocap_pos[:] = src.mocap_pos
        dst.mocap_quat[:] = src.mocap_quat
    mujoco.mj_forward(model, dst)
    return dst


# ----------------------------------------------------------- chunk execution
# ------------------------------------------------------- parallel rollouts
_ROLLOUT_WORKER_WORLD: SimWorld | None = None


# ------------------------------------------------------ twin state resyncs
def reset_object_pose(
    world: SimWorld,
    pos_xyz: tuple[float, float, float] | list[float] | np.ndarray,
    quat_wxyz: tuple[float, float, float, float] | list[float] | np.ndarray,
) -> None:
    """Write a re-observed subject pose into the host kinematic mirror."""
    world.data.qpos[world.object_qpos_addr : world.object_qpos_addr + 3] = np.asarray(pos_xyz, dtype=float)
    world.data.qpos[world.object_qpos_addr + 3 : world.object_qpos_addr + 7] = np.asarray(quat_wxyz, dtype=float)
    world.data.qvel[:] = 0.0
    mujoco.mj_kinematics(world.model, world.data)


def reset_robot_qpos(world: SimWorld, real_q: np.ndarray | list[float]) -> float:
    """Re-sync the sim arm to the REAL robot's measured joint angles.

    Writes the N real joint values into world.data.qpos at the SAME per-joint
    addresses map_robot() resolved, zeroes velocity, and refreshes kinematics so
    the TCP site pose is consistent. Returns the max |delta| from the prior sim
    qpos (rad) so the caller can report the drift it just corrected.
    """
    addr = np.asarray(world.robot["qpos_addr"], dtype=int)
    q = np.asarray(real_q, dtype=float).reshape(-1)[: addr.size]
    prev = np.asarray(world.data.qpos[addr], dtype=float).copy()
    max_delta = float(np.abs(q - prev).max()) if addr.size else 0.0
    world.data.qpos[addr] = q
    world.data.qvel[:] = 0.0
    mujoco.mj_kinematics(world.model, world.data)
    return max_delta


def patch_static_body_pose(
    world: SimWorld,
    body_name: str,
    pos_xyz: np.ndarray | list[float],
    quat_wxyz: np.ndarray | list[float],
) -> tuple[float, float]:
    """Relocalize a STATIC (worldbody-fixed) body to a freshly observed pose.

    The reference/container is built as a fixed 'support_<name>' body whose
    pose lives in model.body_pos / model.body_quat (no freejoint -> not in
    qpos), so a re-observation patches the MODEL, then refreshes kinematics
    without CPU dynamics or contact solving. Returns (pos_drift_m,
    ang_drift_rad) from the prior model pose so the caller can report how far
    the recorded twin had drifted from live.
    """
    bid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"patch_static_body_pose: body {body_name!r} not in model")
    prev_pos = np.asarray(world.model.body_pos[bid], dtype=float).copy()
    prev_quat = np.asarray(world.model.body_quat[bid], dtype=float).copy()
    new_pos = np.asarray(pos_xyz, dtype=float).reshape(3)
    new_quat = np.asarray(quat_wxyz, dtype=float).reshape(4)
    pos_drift = float(np.linalg.norm(new_pos - prev_pos))
    dq = np.zeros(4)
    mujoco.mju_mulQuat(dq, new_quat, np.array([prev_quat[0], -prev_quat[1], -prev_quat[2], -prev_quat[3]]))
    ang_drift = float(2.0 * math.atan2(float(np.linalg.norm(dq[1:])), abs(float(dq[0]))))
    world.model.body_pos[bid] = new_pos
    world.model.body_quat[bid] = new_quat
    mujoco.mj_kinematics(world.model, world.data)
    return pos_drift, ang_drift


def tcp_pose_T(world: SimWorld) -> np.ndarray:
    """Homogeneous TCP (grasp-site) pose -- forward kinematics of the twin."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(world.data.site_xmat[world.site_id], dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(world.data.site_xpos[world.site_id], dtype=float)
    return T
