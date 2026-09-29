"""Gravity-aligned OBB recovery and mesh-axis (q_om) helpers for the twin.

Role in the two-stage pipeline: the closed loop (``oap-run``) localizes
the subject from the masked ZED depth by fitting a gravity-aligned oriented
bounding box -- orientation comes from gravity + depth, NOT render-and-compare,
so it does not inherit FoundationPose's symmetry flips on textureless boxes
(the subject-pose policy fix). The same module owns the mesh->object axis-map
math (q_om), the standing-dims roll fix, and the mesh-hugging collision OBB
helper retained for offline validation. Production collision geometry comes
only from the generated, integrity-bound bundle.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from oap.utils.io import package_config_path
from oap.utils.se3 import quat_mul, quat_to_mat

logger = logging.getLogger("oap.twin.obb")

__all__ = [
    "mesh_axis_rotation", "object_body_quat", "object_vertical_half_extent",
    "object_vertical_half_extent_from_q_om",
    "project_unheld_pose_to_support",
    "gravity_aligned_obb", "load_obj_vertices", "mesh_obb_from_vertices",
    "subject_obb_from_capture", "roll_q_om_for_standing_dims",
    "obb_pose_matrix", "load_base_camera_matrix",
]


def mesh_axis_rotation(assignment: tuple[int, int, int] | list[int]) -> np.ndarray:
    """Return R_om: mesh axes -> object axes (x=long in-plane, y=short, z=up)."""
    ax_l, ax_w, ax_h = assignment
    R_om = np.zeros((3, 3))
    R_om[0, ax_l] = 1.0
    R_om[1, ax_w] = 1.0
    R_om[2, ax_h] = 1.0
    if np.linalg.det(R_om) < 0:
        R_om[1] = -R_om[1]
    return R_om


def object_body_quat(yaw: float, q_om: np.ndarray) -> np.ndarray:
    """Compose the world yaw with the mesh->object axis map into a body quat."""
    half = 0.5 * float(yaw)
    q_yaw = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    return quat_mul(q_yaw, q_om)


def object_vertical_half_extent(
    size_lwh: tuple[float, float, float],
    assignment: tuple[int, int, int] | list[int],
    quat_wxyz: np.ndarray | list[float],
) -> float:
    """Return half the object's extent along base +Z at a given orientation.

    Given its LWH dims, the mesh-axis->LWH assignment, and its world quat.
    Used to rest a reconstructed twin ON the table plane: the localization Z
    is the least-reliable axis (a tall open container localizes ~2 cm low; a
    near-symmetric box's Z is coupled to its ambiguous orientation), so the
    resting height is pinned to the measured table rather than the raw recon Z.
    """
    half_local = np.zeros(3)
    for k, a in enumerate(assignment):
        half_local[int(a)] = 0.5 * float(size_lwh[k])
    quat = np.asarray(quat_wxyz, dtype=float).reshape(4)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    R = quat_to_mat(quat)
    return float(np.sum(np.abs(R[2, :]) * half_local))


def object_vertical_half_extent_from_q_om(
    size_lwh: tuple[float, float, float],
    q_om_wxyz: np.ndarray | list[float],
    quat_wxyz: np.ndarray | list[float],
) -> float:
    """Return the world-Z half extent using the loaded mesh-axis map.

    Runtime observations carry a mesh-frame world quaternion while
    :class:`SceneObject` stores ``q_om`` rather than the original integer
    axis assignment.  This is the same oriented-box calculation as
    :func:`object_vertical_half_extent`, expressed in those runtime terms.
    """
    half_object = 0.5 * np.asarray(size_lwh, dtype=float).reshape(3)
    q_om = np.asarray(q_om_wxyz, dtype=float).reshape(4)
    q_om /= max(float(np.linalg.norm(q_om)), 1e-12)
    # R_om maps mesh axes to semantic object L/W/H axes.  Move the semantic
    # half extents back to the mesh frame used by the collision body.
    half_mesh = np.abs(quat_to_mat(q_om)).T @ half_object
    quat = np.asarray(quat_wxyz, dtype=float).reshape(4)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    return float(np.abs(quat_to_mat(quat)[2, :]) @ half_mesh)


def project_unheld_pose_to_support(
    position_xyz: np.ndarray | list[float] | tuple[float, float, float],
    quat_wxyz: np.ndarray | list[float],
    size_lwh: tuple[float, float, float],
    q_om_wxyz: np.ndarray | list[float],
    support_top_z: float,
    *,
    held: bool | None,
    support_center_xy: np.ndarray | list[float] | tuple[float, float],
    support_half_size_xy: np.ndarray | list[float] | tuple[float, float],
    excluded_footprints: tuple | list = (),
) -> tuple[np.ndarray, bool]:
    """Raise a confirmed-unheld pose only enough to clear a known support.

    This is a physical-consistency lower bound, not pose initialization and
    not the legacy exact table snap.  It never changes XY or orientation,
    never lowers an elevated pose, and does nothing when held state is true or
    unknown.  A body whose center is outside the support footprint is treated
    as genuinely off-support.  Open-container footprints are exclusions
    because their contents rest on an internal floor, not the table plane.
    """
    pos = np.asarray(position_xyz, dtype=float).reshape(3).copy()
    if held is not False:
        return pos, False

    center = np.asarray(support_center_xy, dtype=float).reshape(2)
    half = np.asarray(support_half_size_xy, dtype=float).reshape(2)
    if np.any(np.abs(pos[:2] - center) > half):
        return pos, False
    for _name, footprint_center, radius in excluded_footprints:
        cxy = np.asarray(footprint_center, dtype=float).reshape(2)
        if np.all(np.abs(pos[:2] - cxy) <= float(radius)):
            return pos, False

    vertical_half_extent = object_vertical_half_extent_from_q_om(
        size_lwh,
        q_om_wxyz,
        quat_wxyz,
    )
    lower_bound = float(support_top_z) + vertical_half_extent
    if pos[2] >= lower_bound:
        return pos, False
    pos[2] = lower_bound
    return pos, True


def seat_unheld_pose_on_support(
    position_xyz: np.ndarray | list[float] | tuple[float, float, float],
    quat_wxyz: np.ndarray | list[float],
    size_lwh: tuple[float, float, float],
    q_om_wxyz: np.ndarray | list[float],
    support_top_z: float,
    *,
    held: bool | None,
    support_center_xy: np.ndarray | list[float] | tuple[float, float],
    support_half_size_xy: np.ndarray | list[float] | tuple[float, float],
    excluded_footprints: tuple | list = (),
    tolerance_m: float = 1e-9,
) -> tuple[float, bool]:
    """Place a confirmed-unheld body exactly on its support, up OR down.

    :func:`project_unheld_pose_to_support` is a physical-consistency LOWER
    BOUND: it raises a body that has sunk below its support and returns a body
    that is already higher unchanged.  That is right for a reconstructed
    support, whose z is the least reliable axis and may legitimately be
    elevated.  It is wrong for a body the scene builder is spawning fresh,
    because a body spawned ABOVE its rest height simply free-falls on the
    first physics step, and anything frozen from the t=0 pose then measures
    against a height the body was never at and can never return to.

    The three guards are deliberately identical to the lower-bound version, so
    a body that is held, off-support, or inside an open-top container is left
    exactly where it is.  Only bodies whose support is genuinely the named
    plane are seated.

    Returns ``(z, changed)`` rather than a pose: callers own XY and
    orientation, and neither is ever touched here.
    """
    pos = np.asarray(position_xyz, dtype=float).reshape(3)
    if held is not False:
        return float(pos[2]), False

    center = np.asarray(support_center_xy, dtype=float).reshape(2)
    half = np.asarray(support_half_size_xy, dtype=float).reshape(2)
    if np.any(np.abs(pos[:2] - center) > half):
        return float(pos[2]), False
    for _name, footprint_center, radius in excluded_footprints:
        cxy = np.asarray(footprint_center, dtype=float).reshape(2)
        if np.all(np.abs(pos[:2] - cxy) <= float(radius)):
            return float(pos[2]), False

    seated = float(support_top_z) + object_vertical_half_extent_from_q_om(
        size_lwh, q_om_wxyz, quat_wxyz
    )
    if abs(float(pos[2]) - seated) <= float(tolerance_m):
        return float(pos[2]), False
    return seated, True


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """Andrew's monotone-chain convex hull of 2D points (returns hull vertices)."""
    p = np.unique(np.asarray(pts, dtype=float), axis=0)
    if len(p) < 3:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def _half(ps: np.ndarray) -> list:
        h: list = []
        for q in ps:
            while len(h) >= 2:
                d1, d2 = h[-1] - h[-2], q - h[-2]
                if d1[0] * d2[1] - d1[1] * d2[0] > 0:   # 2D cross (NumPy-2.0 safe)
                    break
                h.pop()
            h.append(q)
        return h[:-1]

    return np.asarray(_half(p) + _half(p[::-1]), dtype=float)


def _min_area_rect(xy: np.ndarray) -> tuple[float, tuple[float, float], np.ndarray]:
    """Min-area bounding rectangle of 2D points (rotating calipers over hull
    edges). Returns (yaw_rad of the u-axis, (ext_u, ext_v), center_xy)."""
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 3:
        return 0.0, (0.0, 0.0), (xy.mean(0) if len(xy) else np.zeros(2))
    hull = _convex_hull(xy)
    if len(hull) < 3:
        return 0.0, (float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1]))), xy.mean(0)
    best = None
    for i in range(len(hull)):
        edge = hull[(i + 1) % len(hull)] - hull[i]
        n = float(np.linalg.norm(edge))
        if n < 1e-9:
            continue
        u = edge / n
        v = np.array([-u[1], u[0]])
        pu, pv = xy @ u, xy @ v
        w, h = float(pu.max() - pu.min()), float(pv.max() - pv.min())
        if best is None or (w * h) < best[0]:
            cu, cv = 0.5 * (pu.max() + pu.min()), 0.5 * (pv.max() + pv.min())
            best = (w * h, float(np.arctan2(u[1], u[0])), (w, h), cu * u + cv * v)
    if best is None:  # every hull edge degenerate: axis-aligned fallback
        return 0.0, (float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1]))), xy.mean(0)
    return best[1], best[2], best[3]


def gravity_aligned_obb(
    points_base: np.ndarray,
    table_top_z: float,
    known_dims: np.ndarray | list[float] | tuple[float, ...] | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit the gravity-aligned OBB of a tabletop object's BASE-frame cloud.

    Returns (center_xyz, yaw_rad, dims_LWH) with the object RESTING on the
    table: the vertical extent is dims[2] (H), the footprint is a min-area
    rectangle in the table plane (dims[0]=L >= dims[1]=W) at ``yaw``.
    Orientation comes from gravity + the masked depth (NOT render-and-compare),
    so it does NOT inherit FoundationPose's symmetry flips on a textureless
    box -- which is exactly why a box stood on its narrow side was being
    twinned as if lying flat.

    ``known_dims`` (the object's 3 true side lengths, any order) snaps the
    measured height to the nearest known dimension and the footprint to the
    other two, so a partially-occluded cloud still yields metric dims + the
    right vertical axis.
    """
    pts = np.asarray(points_base, dtype=float).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 3:
        raise ValueError("gravity_aligned_obb: too few valid points")
    height = max(1e-3, float(np.percentile(pts[:, 2], 98)) - float(table_top_z))
    yaw, (ext_u, ext_v), center_xy = _min_area_rect(pts[:, :2])
    if ext_v > ext_u:                       # make yaw index the LONGER footprint axis
        yaw += np.pi / 2.0
    yaw = (yaw + np.pi / 2.0) % np.pi - np.pi / 2.0     # wrap to [-pi/2, pi/2]
    foot = sorted([float(ext_u), float(ext_v)], reverse=True)
    dims = np.array([foot[0], foot[1], height], dtype=float)
    if known_dims is not None:
        kd = np.sort(np.asarray(known_dims, dtype=float))[::-1]
        vi = int(np.argmin(np.abs(kd - height)))            # which known dim is vertical
        rest = np.sort(np.delete(kd, vi))[::-1]
        dims = np.array([rest[0], rest[1], kd[vi]], dtype=float)
    center = np.array([center_xy[0], center_xy[1], float(table_top_z) + 0.5 * dims[2]], dtype=float)
    return center, float(yaw), dims


def load_obj_vertices(path: Path | str) -> np.ndarray:
    """Read the v-line vertices of a Wavefront .obj as an Nx3 array (metres)."""
    vs = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            p = line.split()
            vs.append((float(p[1]), float(p[2]), float(p[3])))
    return np.asarray(vs, dtype=float).reshape(-1, 3)


def mesh_obb_from_vertices(
    vertices: np.ndarray, q_om: np.ndarray | list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Bound the mesh ALONG THE OBJECT AXES (via q_om: mesh->object).

    Returns (extents [L,W,H], center [x,y,z]) in the OBJECT frame so the
    analytic box HUGS the mesh instead of a stale reconstructed size_LWH --
    critical for precise grasps.
    """
    V = np.asarray(vertices, dtype=float).reshape(-1, 3)
    if len(V) < 4:
        raise ValueError("mesh_obb_from_vertices: too few vertices")
    R = quat_to_mat(np.asarray(q_om, dtype=float))
    Vo = V @ R.T
    lo, hi = Vo.min(axis=0), Vo.max(axis=0)
    return (hi - lo), (lo + hi) / 2.0


def load_base_camera_matrix(path: Path | None = None) -> np.ndarray:
    """Load the 4x4 T_base_camera extrinsic from a calibration yaml.

    Defaults to the packaged ``configs/calibration/T_base_zed2i.yaml``.
    Accepts either a top-level ``T_base_zed2i`` / ``T_base_zed2i_candidate``
    section or a bare mapping with a ``matrix`` key.
    """
    from oap.utils.io import load_yaml
    from oap.utils.se3 import validate_se3_matrix

    path = path or package_config_path("calibration/T_base_zed2i.yaml")
    doc = load_yaml(path)
    if "T_base_zed2i" in doc:
        doc = doc["T_base_zed2i"]
    elif "T_base_zed2i_candidate" in doc:
        doc = doc["T_base_zed2i_candidate"]
    if not isinstance(doc, dict) or "matrix" not in doc:
        raise ValueError(f"{path} does not contain a 4x4 T_base_zed2i matrix")
    return validate_se3_matrix(
        doc["matrix"],
        name=f"base<-camera calibration {path}",
    )


def subject_obb_from_capture(
    capture_dir: Path,
    subject: Any,
    table_top_z: float,
    *,
    t_base_camera: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit the subject's gravity-aligned OBB from a saved ZED capture.

    Reads ``<capture_dir>/<subject.name>/{depth_m.npy, mask_00.png, cam_K.txt}``,
    builds the subject's base-frame masked point cloud, and returns its
    gravity-aligned OBB (center_xyz, yaw, dims_LWH). ``t_base_camera`` defaults
    to the packaged hand-eye calibration.
    """
    from PIL import Image

    d = Path(capture_dir) / subject.name
    depth = np.load(d / "depth_m.npy").astype(float)
    mask = np.asarray(Image.open(d / "mask_00.png").convert("L")) > 127
    K = np.loadtxt(d / "cam_K.txt").reshape(3, 3)
    T_bc = (np.asarray(t_base_camera, dtype=float) if t_base_camera is not None
            else load_base_camera_matrix())
    sel = mask & np.isfinite(depth) & (depth > 1e-3)
    vs, us = np.where(sel)
    if len(vs) < 50:
        raise ValueError(f"subject_obb_from_capture: only {len(vs)} masked depth pixels")
    z = depth[vs, us]
    x = (us - K[0, 2]) / K[0, 0] * z
    y = (vs - K[1, 2]) / K[1, 1] * z
    cam = np.stack([x, y, z, np.ones_like(z)], axis=1)          # camera (OpenCV)
    base = (T_bc @ cam.T).T[:, :3]
    return gravity_aligned_obb(base, table_top_z, known_dims=subject.size_lwh)


def roll_q_om_for_standing_dims(
    q_om: np.ndarray,
    size_lwh: tuple[float, float, float],
    obb_dims: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """Roll q_om so the gravity-OBB's newly-vertical axis points up.

    gravity_aligned_obb may reorder WHICH axis is vertical, but q_om still
    encodes the flat refined axis-map -- without also rolling q_om the MESH
    renders/collides on its old (wide) face while only the dims change.
    Canonical object frame: L=x, W=y, H=z; the refined order has the smallest
    dim H=z vertical.

    Args:
        q_om: The refined mesh->object axis-map quaternion (wxyz).
        size_lwh: The subject's PRE-roll size_lwh (the refined flat order).
        obb_dims: The gravity-OBB dims (L, W, H) with H vertical.

    Returns:
        The rolled q_om (unchanged when the vertical axis already matches).
    """
    q_om = np.asarray(q_om, dtype=float)
    nv = float(obb_dims[2])
    vax = int(np.argmin([abs(nv - float(d)) for d in size_lwh]))
    if vax == 1:      # old W (canonical y) now vertical -> roll +90deg about x (narrow edge)
        return quat_mul(np.array([0.70710678, 0.70710678, 0.0, 0.0]), q_om)
    if vax == 0:      # old L (canonical x) now vertical -> roll +90deg about y (stand tall)
        return quat_mul(np.array([0.70710678, 0.0, 0.70710678, 0.0]), q_om)
    return q_om


def obb_pose_matrix(
    center_base: np.ndarray | list[float],
    yaw: float,
    t_base_camera: np.ndarray,
) -> np.ndarray:
    """Build the gravity-OBB pose as a CAMERA-frame 4x4 T_cam_obj (FP seed).

    Seeds FoundationPose's refinement (the joint pose+scale recipe) instead of
    its blind global register -- which fixes the symmetry flip on textureless
    objects. The OBB is gravity-aligned: object +z = world up, object +x = the
    footprint long axis at ``yaw`` about world z, +y = z x x. Built in the
    base frame, then mapped to the camera frame by inv(T_base_camera).
    """
    c, s = np.cos(float(yaw)), np.sin(float(yaw))
    R_base_obj = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    T_base_obj = np.eye(4)
    T_base_obj[:3, :3] = R_base_obj
    T_base_obj[:3, 3] = np.asarray(center_base, dtype=float).reshape(3)
    return np.linalg.inv(np.asarray(t_base_camera, dtype=float)) @ T_base_obj
